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

* netloader: sends the finalized state to a live peer (HCCL P2P);
* rfork: transfers it across fork boundaries (YuanRong TransferEngine);
* bigtensorloader: persists it to a disk snapshot (mmap + H2D).

Transfer lifecycles stay per-loader; only enumeration is shared. The
module offers two enumeration strategies over the same state, because the
consumers disagree on what a "correct" enumeration is:

* **Access-chain DFS** (:func:`iter_module_weight_states`,
  :func:`iter_tensor_records`): yields every tensor together with a
  tagged access chain that doubles as its restore address. Used by
  bigtensorloader (snapshot manifests) and netloader (transfer items).
  First-encounter order; containers recorded as definitions; private
  runtime attrs skipped.
* **Canonical shortest-path BFS** (:func:`collect_transferable_tensors`):
  yields each transferable tensor range once under a canonical ID.
  Registered parameters/buffers are claimed first so in-place layout
  changes reach the registered tensor; every remaining tensor takes the
  lexicographically shortest access path, so IDs are independent of
  attribute insertion order; distinct container entries that would share
  one ID label are rejected instead of guessed. Used by rfork (seed keys,
  transfer manifests, structural digests), where both sides of a transfer
  must derive identical IDs.

Both strategies share the transferability predicates at the top of this
module. New enumeration policies belong here rather than in private
walkers inside loaders.

Path encoding
-------------
Every access-chain tensor carries an *access chain* relative to its owning
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

import enum
import json
from collections.abc import Iterator
from dataclasses import dataclass
from types import FunctionType, MethodType
from typing import Any

import regex as re
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Shared transferability predicates
# ---------------------------------------------------------------------------


def is_tensor_on_transfer_device(tensor: torch.Tensor) -> bool:
    """Whether a finalized tensor lives on the local NPU."""
    return tensor.device.type == "npu"


def is_transferable_tensor(tensor: torch.Tensor) -> bool:
    """Whether a finalized tensor carries payload worth transferring.

    netloader and rfork both ship the post-process device state, so they
    share this gate; bigtensorloader snapshots everything and does not
    filter on it.
    """
    return not tensor.is_meta and tensor.numel() > 0 and is_tensor_on_transfer_device(tensor)


def is_non_overlapping_dense_tensor(tensor: torch.Tensor) -> bool:
    """Return whether logical elements occupy one contiguous byte range."""
    if tensor.numel() <= 1:
        return True

    dense_stride = 1
    for stride, size in sorted(
        (int(stride), int(size)) for size, stride in zip(tensor.shape, tensor.stride(), strict=True) if size > 1
    ):
        if stride != dense_stride:
            return False
        dense_stride *= size
    return True


def validate_transferable_tensor_layout(name: str, tensor: torch.Tensor) -> None:
    """Reject tensor views that cannot be represented by RFork byte ranges."""
    if is_non_overlapping_dense_tensor(tensor):
        return
    raise ValueError(
        "RFork cannot transfer a tensor with gapped or overlapping storage: "
        f"{name!r}; shape={tuple(tensor.shape)}, stride={tuple(tensor.stride())}."
    )


# ---------------------------------------------------------------------------
# Strategy A: access-chain DFS enumeration (netloader, bigtensorloader)
# ---------------------------------------------------------------------------


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
    ``scan_objects`` is set (the netloader policy: only attention ``impl``
    carriers hold derived tensors in practice; scanning every object would
    walk large config graphs for nothing).

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
            ``deep_scan_all_objects`` is off (the netloader policy passes
            ``("impl",)``).
        skip_private: skip attribute names starting with ``_`` (runtime
            scratch by convention; aligned with netloader).
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
            if module_defs and container_defs is not None:
                for def_key, d in module_defs.items():
                    container_defs[f"{module_prefix}|{def_key}"] = d


# ---------------------------------------------------------------------------
# Strategy B: canonical shortest-path BFS enumeration (rfork)
# ---------------------------------------------------------------------------

# Exact types only: numeric subclasses can carry tensor attributes.
_TENSOR_ATTRIBUTE_LEAF_TYPES = frozenset({str, bytes, int, float, bool, complex, type(None)})

# Runtime scratch and locally derived caches keep receiver-owned storage.
_RUNTIME_ONLY_TENSOR_NAMES = frozenset({"topk_indices_buffer", "_rope_derived_cos_cache", "_rope_derived_sin_cache"})

_PLAIN_PATH_SEGMENT = re.compile(r"[A-Za-z0-9_]+")


def _is_runtime_only_tensor(name: str) -> bool:
    return name.rsplit(".", 1)[-1] in _RUNTIME_ONLY_TENSOR_NAMES


def _format_tensor_key(key: Any) -> str | None:
    """Render a container key as a typed label that is identical across processes."""
    # Order matters: bool and Enum members subclass int/str.
    if key is None or isinstance(key, bool):
        return json.dumps(key)
    if isinstance(key, enum.Enum):
        key_type = type(key)
        enum_type = f"{key_type.__module__}.{key_type.__qualname__}"
        if isinstance(key, enum.Flag):
            return f"enum({enum_type}:{int(key.value)})" if isinstance(key.value, int) else None
        return f"enum({enum_type}.{key.name})" if key.name is not None else None
    if isinstance(key, str):
        return json.dumps(key, ensure_ascii=True)
    if isinstance(key, int):
        return str(int(key))
    if isinstance(key, float):
        return f"float({float(key).hex()})"
    if isinstance(key, torch.dtype):
        return f"dtype({key})"
    if type(key) is tuple:
        items: list[str] = []
        for item in key:
            label = _format_tensor_key(item)
            if label is None:
                return None
            items.append(label)
        return f"({items[0]},)" if len(items) == 1 else f"({','.join(items)})"
    # Default reprs embed addresses, so unknown key types cannot yield stable IDs.
    return None


def _tensor_child_path(parent_path: str, kind: str, name: Any) -> str:
    if kind in ("index", "key"):
        label = _format_tensor_key(name)
        if label is None:
            name_type = type(name)
            raise ValueError(
                "RFork cannot derive a stable tensor ID from container key type "
                f"{name_type.__module__}.{name_type.__qualname__}"
            )
        return f"{parent_path}[{label}]"
    name = str(name)
    if not _PLAIN_PATH_SEGMENT.fullmatch(name):
        return f"{parent_path}[{json.dumps(name, ensure_ascii=True)}]"
    return f"{parent_path}.{name}" if parent_path else name


def _iter_tensor_children(
    value: Any, scan_objects: bool, processed_layout: bool, registered_only: bool
) -> Iterator[tuple[str, Any, Any, bool]]:
    if isinstance(value, nn.Module):
        # Read local registries: named_parameters/named_modules discard aliases before we can canonicalize them.
        for kind, members in (("module", value._modules), ("parameter", value._parameters), ("buffer", value._buffers)):
            for name, item in members.items():
                if type(item) not in _TENSOR_ATTRIBUTE_LEAF_TYPES:
                    yield kind, name, item, False
        if registered_only:
            return
        if processed_layout:
            for name, item in vars(value).items():
                if not name.startswith("_") and type(item) not in _TENSOR_ATTRIBUTE_LEAF_TYPES:
                    yield "attribute", name, item, name == "impl"
        else:
            impl = getattr(value, "impl", None)
            if type(impl) not in _TENSOR_ATTRIBUTE_LEAF_TYPES:
                yield "attribute", "impl", impl, True
    elif registered_only:
        return
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            if type(item) not in _TENSOR_ATTRIBUTE_LEAF_TYPES:
                yield "index", index, item, scan_objects
    elif isinstance(value, dict):
        for name, item in value.items():
            if type(item) not in _TENSOR_ATTRIBUTE_LEAF_TYPES:
                yield "key", name, item, scan_objects
    else:
        for name, item in vars(value).items():
            if not name.startswith("_") and type(item) not in _TENSOR_ATTRIBUTE_LEAF_TYPES:
                yield "attribute", name, item, scan_objects


def _tensor_child_key(
    kind: str,
    name: Any,
    value: Any,
    scan_objects: bool,
    tensor_metadata: dict[int, tuple[torch.Tensor, tuple[Any, ...] | None]],
    parent_path: str,
) -> tuple[Any, ...] | None:
    if isinstance(value, torch.Tensor):
        # Reject this alias edge only; another alias may still collect the tensor.
        if _is_runtime_only_tensor(str(name)):
            return None
        metadata = tensor_metadata.get(id(value))
        if metadata is not None:
            return metadata[1]
        signature = None
        if is_transferable_tensor(value):
            validate_transferable_tensor_layout(_tensor_child_path(parent_path, kind, name), value)
            signature = (
                value.data_ptr(),
                value.numel(),
                tuple(value.shape),
                value.dtype,
                value.device,
                tuple(value.stride()),
            )
        tensor_metadata[id(value)] = (value, signature)
        return signature
    if isinstance(value, nn.Module):
        return (id(value), False) if kind == "module" else None
    if isinstance(value, (str, bytes)):
        return None
    if isinstance(value, (list, tuple, dict)):
        return (id(value), scan_objects)
    if not scan_objects or isinstance(value, (FunctionType, MethodType, type)):
        return None
    return (id(value), scan_objects) if hasattr(value, "__dict__") else None


def _collect_shortest_path_tensors(
    model: nn.Module,
    processed_layout: bool,
    registered_only: bool,
    claimed: set[tuple[Any, ...]],
    tensor_metadata: dict[int, tuple[torch.Tensor, tuple[Any, ...] | None]],
    collected: list[tuple[str, torch.Tensor]],
) -> None:
    root_key = (id(model), False)
    frontier = {root_key: ""}
    # Keep strong references throughout the scan so object IDs cannot be reused.
    objects: dict[tuple[Any, ...], tuple[Any, bool]] = {root_key: (model, False)}
    visited = set(claimed)

    while frontier:
        # IDs in this layer are final. Ignore back edges, cycles, and longer paths.
        visited.update(frontier)
        next_frontier: dict[tuple[Any, ...], str] = {}
        for key, path in frontier.items():
            value, scan_objects = objects[key]
            if isinstance(value, torch.Tensor):
                claimed.add(key)
                collected.append((path, value))
                continue

            children = _iter_tensor_children(value, scan_objects, processed_layout, registered_only)
            child_paths: set[str] = set()
            for kind, name, item, child_scan_objects in children:
                child_key = _tensor_child_key(kind, name, item, child_scan_objects, tensor_metadata, path)
                if child_key is None:
                    continue
                candidate_path = _tensor_child_path(path, kind, name)
                if candidate_path in child_paths:
                    raise ValueError(
                        f"RFork found distinct container entries with one tensor ID label: {candidate_path!r}"
                    )
                child_paths.add(candidate_path)
                if child_key in visited:
                    continue
                previous_path = next_frontier.get(child_key)
                if previous_path is None or candidate_path < previous_path:
                    # Same-signature aliases can be distinct objects; keep the one the ID names.
                    objects[child_key] = (item, child_scan_objects)
                    next_frontier[child_key] = candidate_path
        frontier = next_frontier


def collect_transferable_tensors(model: nn.Module, processed_layout: bool) -> list[tuple[str, torch.Tensor]]:
    """Collect each tensor range once under its canonical shortest-path ID.

    Registered parameters and buffers take priority over attribute aliases so
    in-place layout changes reach the registered tensor.
    """
    claimed: set[tuple[Any, ...]] = set()
    tensor_metadata: dict[int, tuple[torch.Tensor, tuple[Any, ...] | None]] = {}
    collected: list[tuple[str, torch.Tensor]] = []
    for registered_only in (True, False):
        _collect_shortest_path_tensors(model, processed_layout, registered_only, claimed, tensor_metadata, collected)
    tensor_ids = [tensor_id for tensor_id, _ in collected]
    if len(tensor_ids) != len(set(tensor_ids)):
        raise ValueError("RFork encountered conflicting tensor IDs in one model.")
    return collected
