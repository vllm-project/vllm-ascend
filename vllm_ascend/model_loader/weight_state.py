# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared enumeration of a model's finalized (post-process) weight state.

After ``process_weights_after_loading``, tensors no longer live only in
``state_dict``: Ascend post-processing stashes derived runtime tensors as
plain module attributes (e.g. ``aclnn_input_scale_reciprocal``), inside
attention ``impl`` carrier objects (e.g. MLA W_UV / W_UK_T), and inside
list/dict containers. Enumerating that state correctly -- without infinite
recursion on shared or cyclic graphs, without double-walking graphs shared
between modules -- is the hard problem three loaders each need:

* netloader: sends the finalized state to a live donor peer (HCCL P2P);
* rfork: transfers checkpoint/processed layouts across fork boundaries;
* bigtensorloader: persists the finalized state to a disk snapshot.

This module owns the traversal so all three share one implementation.
Transfer lifecycles stay per-loader; only enumeration is shared.

Path encoding
-------------
Every yielded tensor carries an *access chain* relative to its owning
module: a tuple of tagged steps, each a single-key dict:

* ``{"a": name}`` -- ``getattr``/``setattr`` step (module attr or plain-object
  member, e.g. the ``impl`` carrier's fields);
* ``{"i": idx}`` -- list/tuple index step (``obj[idx]``);
* ``{"k": key}`` -- dict key step (``obj[key]``).

The dotted ``name`` is derived alongside and is only a human-readable
identifier; the chain is the authoritative restore address, so dict keys
containing dots cannot confuse resolution. Chains are JSON-serializable,
which lets bigtensorloader persist them in its snapshot manifest.
"""

import json
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class WeightStateRecord:
    """One tensor reachable from a module's ``__dict__`` attributes.

    Attributes:
        module: owning module's prefix in ``named_modules`` ("" for root).
        attr: root attribute name on that module; resolution starts at
            ``getattr(module, attr)``.
        name: dotted qualified name (module + attr + path); display/key only.
        path: tagged access chain (see module docstring) from
            ``getattr(module, attr)`` down to the tensor's slot.
        tensor: the tensor itself.
    """

    module: str
    attr: str
    name: str
    path: tuple[dict, ...]
    tensor: torch.Tensor


def iter_tensor_records(
    prefix: str,
    value: Any,
    visited_object_ids: set[int],
    scan_objects: bool = False,
    container_defs: dict[str, dict] | None = None,
    path: tuple[dict, ...] = (),
):
    """Recursively walk ``value``, yielding ``(name, path, tensor)`` records.

    Cycle-safe: every walked list/tuple/dict/plain-object is recorded in
    ``visited_object_ids``, so shared graphs are entered exactly once per
    enumeration (pass one set per model walk to also skip graphs shared
    between modules, or a fresh set per call to re-walk per attribute).

    Skips: ``nn.Module`` (their tensors live in ``state_dict``), strings,
    bytes, and callables. Plain objects are only entered when
    ``scan_objects`` is set (the netloader/rfork policy: only attention
    ``impl`` carriers hold derived tensors in practice; scanning every object
    would walk large config graphs for nothing).

    ``visited_object_ids`` must be provided by the caller; ownership of its
    scope is a policy decision, not a walker decision.

    ``container_defs``: when given, every entered list/tuple/dict is recorded
    as ``{json(chain): {"type": ..., "len": ...}}`` where ``chain`` is the
    full tagged access chain (root ``{"a": attr}`` included) reaching that
    container. Consumers that recreate containers on restore (bigtensorloader)
    need the declared type (dict children must not be reborn as lists) and
    length (a list may hold tensor-free slots after the last tensor).
    ``path`` is the internal accumulator for that chain.
    """
    if isinstance(value, torch.Tensor):
        yield prefix, (), value
        return

    if isinstance(value, (nn.Module, str, bytes)) or callable(value):
        return

    if isinstance(value, (list, tuple)):
        value_id = id(value)
        if value_id in visited_object_ids:
            return
        visited_object_ids.add(value_id)
        if container_defs is not None:
            container_defs[json.dumps(path)] = {
                "type": "tuple" if isinstance(value, tuple) else "list",
                "len": len(value),
            }
        for index, item in enumerate(value):
            for name, sub_path, tensor in iter_tensor_records(
                f"{prefix}.{index}",
                item,
                visited_object_ids,
                scan_objects,
                container_defs,
                path + ({"i": index},),
            ):
                yield name, ({"i": index}, *sub_path), tensor
        return

    if isinstance(value, dict):
        value_id = id(value)
        if value_id in visited_object_ids:
            return
        visited_object_ids.add(value_id)
        if container_defs is not None:
            container_defs[json.dumps(path)] = {"type": "dict"}
        for key, item in value.items():
            for name, sub_path, tensor in iter_tensor_records(
                f"{prefix}.{key}",
                item,
                visited_object_ids,
                scan_objects,
                container_defs,
                path + ({"k": key},),
            ):
                yield name, ({"k": key}, *sub_path), tensor
        return

    if not scan_objects or not hasattr(value, "__dict__"):
        return

    value_id = id(value)
    if value_id in visited_object_ids:
        return
    visited_object_ids.add(value_id)
    for attr_name, attr_value in vars(value).items():
        if attr_name.startswith("_"):
            continue
        for name, sub_path, tensor in iter_tensor_records(
            f"{prefix}.{attr_name}",
            attr_value,
            visited_object_ids,
            scan_objects,
            container_defs,
            path + ({"a": attr_name},),
        ):
            yield name, ({"a": attr_name}, *sub_path), tensor


def iter_tensors_in_value(prefix: str, value: Any, visited_object_ids: set[int], scan_objects: bool = False):
    """Back-compat wrapper: yield ``(name, tensor)`` like the original
    netloader/rfork private helpers. New code should use
    :func:`iter_tensor_records` and keep the access chain.
    """
    for name, _path, tensor in iter_tensor_records(prefix, value, visited_object_ids, scan_objects):
        yield name, tensor


def iter_module_weight_states(
    model: nn.Module,
    *,
    deep_scan_all_objects: bool = False,
    deep_scan_attrs: tuple[str, ...] = (),
    skip_private: bool = True,
    container_defs: dict[str, dict] | None = None,
):
    """Enumerate every tensor reachable from module ``__dict__`` attributes.

    Covers plain tensor attributes, containers (list/tuple/dict), and plain
    object carriers (``impl``-style). Params/buffers are NOT included --
    ``state_dict`` / ``named_parameters`` / ``named_buffers`` own those.

    Args:
        deep_scan_all_objects: enter every plain-object attribute. Use for
            snapshot save (bigtensorloader): capture as much as possible.
        deep_scan_attrs: extra attribute names to deep-scan when
            ``deep_scan_all_objects`` is off (the netloader/rfork policy
            passes ``("impl",)``).
        skip_private: skip attribute names starting with ``_`` (runtime
            scratch by convention; aligned with netloader/rfork).
        container_defs: when given, filled with list/tuple definitions keyed
            ``"<module_prefix>|<json chain>"`` (chain includes the root
            ``{"a": attr}`` step) so restore can recreate missing containers.
    """
    visited: set[int] = set()
    for module_prefix, module in model.named_modules(remove_duplicate=False):
        for attr_name, attr_value in vars(module).items():
            if skip_private and attr_name.startswith("_"):
                continue
            if isinstance(attr_value, nn.Module):
                continue
            scan_objects = deep_scan_all_objects or attr_name in deep_scan_attrs
            # Def keys are module-scoped: the same attr name under different
            # modules must not collide. The shared visited set means a graph
            # shared between modules is recorded under its first owner only,
            # matching how restore reaches it.
            module_defs: dict[str, dict] | None = {} if container_defs is not None else None
            for dotted, path, tensor in iter_tensor_records(
                attr_name, attr_value, visited, scan_objects, module_defs, ({"a": attr_name},)
            ):
                yield WeightStateRecord(
                    module=module_prefix,
                    attr=attr_name,
                    name=f"{module_prefix}.{dotted}" if module_prefix else dotted,
                    path=path,
                    tensor=tensor,
                )
            if module_defs:
                for def_key, d in module_defs.items():
                    container_defs[f"{module_prefix}|{def_key}"] = d
