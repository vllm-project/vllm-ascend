# SPDX-License-Identifier: Apache-2.0
"""Inventory model-owned tensors before request/KV-cache allocation.

Only Python attribute storage and builtin containers are inspected. Properties,
native handles and arbitrary Python objects are not traversed. Runtime reset
hooks remain responsible for request state and tensors allocated after loading.
"""

import torch
from torch import nn


def capture_tensor_inventory(model: nn.Module) -> None:
    """Record paths, not copies, to non-state_dict tensors after weight loading.

    Keep every alias path for validation, but save each tensor object only once.
    Private attributes, nonpersistent buffers and nested list/tuple/dict values
    are included. Do not register buffers or change the model's state_dict ABI.
    """
    registered = {id(t) for t in model.parameters()}
    for module in model.modules():
        registered.update(id(t) for k, t in module._buffers.items() if k not in module._non_persistent_buffers_set)
    paths = []

    def visit(value, path, ancestors):
        if isinstance(value, torch.Tensor):
            # Attention modules have empty KV-cache placeholders at load time.
            # Those acquire request storage later and must not be checkpointed.
            if value.numel() and id(value) not in registered:
                if any(not isinstance(key, (str, int)) for _, key in path[3:]):
                    raise TypeError(f"Unsupported snapshot tensor container key: {path}")
                paths.append(path)
            return
        if id(value) in ancestors:
            return
        ancestors = ancestors | {id(value)}
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, path + (("key", index),), ancestors)
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, path + (("key", key),), ancestors)

    for name, module in model.named_modules():
        for owner_name, owner in (("module", module), ("impl", getattr(module, "impl", None))):
            if owner is None:
                continue
            for attribute, value in vars(owner).items():
                if attribute in ("_modules", "_parameters", "_snapshot_tensor_paths"):
                    continue
                visit(value, (name, owner_name, attribute), set())
    model._snapshot_tensor_paths = tuple(paths)


def inventory_tensors(model: nn.Module) -> dict:
    """Resolve original paths against the live model, preserving tensor aliases."""
    modules = dict(model.named_modules())
    tensors = {}
    for path in model._snapshot_tensor_paths:
        name, owner_name, attribute, *steps = path
        owner = modules[name]
        if owner_name == "impl":
            owner = owner.impl
        value = vars(owner)[attribute]
        for _, key in steps:
            value = value[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Snapshot model tensor was removed or replaced by a non-tensor: {path}")
        tensors[path] = value
    return tensors


def inventory_state(model: nn.Module) -> dict:
    """Produce a weights-only-loadable payload with explicit alias information."""
    tensors, aliases, seen = {}, {}, {}
    for path, tensor in inventory_tensors(model).items():
        if id(tensor) in seen:
            aliases[path] = seen[id(tensor)]
        else:
            seen[id(tensor)] = path
            tensors[path] = tensor.detach()
    return {"version": 1, "tensors": tensors, "aliases": aliases}


def validate_inventory(model: nn.Module, state: dict) -> dict:
    """Reject incompatible checkpoints before changing any tensor contents."""
    if state["version"] != 1:
        raise ValueError("Unsupported snapshot tensor inventory version")
    live = inventory_tensors(model)
    saved = state["tensors"]
    aliases = state["aliases"]
    if set(live) != set(saved) | set(aliases):
        raise ValueError("Snapshot tensor inventory differs; use a new checkpoint directory")
    for path, source in aliases.items():
        if source not in saved or live[path] is not live[source]:
            raise ValueError(f"Snapshot tensor alias mismatch: {path}")
    if len({id(live[path]) for path in saved}) != len(saved):
        raise ValueError("Previously independent snapshot tensors now alias each other")
    for path, tensor in saved.items():
        if live[path].shape != tensor.shape or live[path].dtype != tensor.dtype:
            raise ValueError(f"Snapshot tensor shape/dtype mismatch: {path}")
    return {path: live[path] for path in saved}
