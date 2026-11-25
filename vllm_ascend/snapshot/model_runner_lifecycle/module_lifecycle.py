# SPDX-License-Identifier: Apache-2.0

"""Checkpoint and restore model module state."""

import ctypes
import fcntl
import gc
import os
import time
from collections.abc import Iterable, Iterator

import torch
import torch.nn as nn
from vllm.logger import logger

from vllm_ascend.snapshot.model_runner_lifecycle.h2d_copy import copy_checkpoint_tensor


def dump_state_dict(model: nn.Module, path: str) -> None:
    has_inventory = hasattr(model, "_snapshot_tensor_paths")
    if has_inventory:
        from vllm_ascend.snapshot.model_runner_lifecycle.tensor_inventory import inventory_state, validate_inventory

    # Keep the lock file: unlinking it could let writers lock different inodes.
    with open(f"{path}.lock", "a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if os.path.exists(path):
            if has_inventory:
                saved = torch.load(path, map_location="cpu", mmap=True)
                if "inventory" not in saved or "state_dict" not in saved:
                    raise ValueError(f"Checkpoint lacks tensor inventory; regenerate it in a new directory: {path}")
                validate_inventory(model, saved["inventory"])
            logger.debug(
                "[snapshot][checkpoint] dump skipped: path=%s reason=already_exists",
                path,
            )
            return

        start = time.time()
        state = model.state_dict()
        if has_inventory:
            state = {"state_dict": state, "inventory": inventory_state(model)}
        torch.save(state, path)
    gc.collect()
    torch.npu.empty_cache()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception as e:
        logger.warning(
            "[snapshot][checkpoint] malloc trim failed: error=%s",
            e,
        )

    logger.info(
        "[snapshot][checkpoint] dump completed: path=%s model_type=%s duration=%.4f s",
        path,
        type(model).__name__,
        time.time() - start,
    )


@torch.inference_mode()
def restore_state_dict(model: nn.Module, path: str, label: str) -> None:
    has_inventory = hasattr(model, "_snapshot_tensor_paths")
    if has_inventory:
        from vllm_ascend.snapshot.model_runner_lifecycle.tensor_inventory import validate_inventory

    if not os.path.exists(path):
        logger.error(
            "[snapshot][checkpoint] restore failed: model=%s path=%s reason=not_found",
            label,
            path,
        )
        raise FileNotFoundError(f"Snapshot checkpoint does not exist: {path}")

    start = time.time()
    state_dict = torch.load(path, map_location="cpu", mmap=True)
    inventory = None
    if has_inventory:
        if "inventory" not in state_dict or "state_dict" not in state_dict:
            raise ValueError(f"Checkpoint lacks tensor inventory; regenerate it in a new directory: {path}")
        inventory = state_dict["inventory"]
        destinations = validate_inventory(model, inventory)
        state_dict = state_dict["state_dict"]
    restored = 0
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    for name, cpu_tensor in state_dict.items():
        if name in parameters:
            copy_checkpoint_tensor(parameters[name].data, cpu_tensor)
            restored += 1
        if name in buffers:
            copy_checkpoint_tensor(buffers[name].data, cpu_tensor)
            restored += 1
    if inventory is not None:
        for name, tensor in destinations.items():
            copy_checkpoint_tensor(tensor, inventory["tensors"][name])
    logger.info(
        "[snapshot][checkpoint] restore completed: model=%s path=%s tensors=%d/%d duration=%.4f s",
        label,
        path,
        restored,
        len(state_dict),
        time.time() - start,
    )


def get_drafter_model(runner) -> nn.Module | None:
    """Return the target runner's separately owned draft model, if any."""
    return runner.get_draft_model()


def _iter_modules_and_impls(models: Iterable[nn.Module | None]) -> Iterator[object]:
    """Yield each model module followed by its backend implementation."""
    visited_ids: set[int] = set()
    for model in models:
        if model is None:
            continue
        for module in model.modules():
            for item in (module, getattr(module, "impl", None)):
                if item is None or id(item) in visited_ids:
                    continue
                visited_ids.add(id(item))
                yield item


@torch.inference_mode()
def reset_modules_runtime_state(models: Iterable[nn.Module | None]) -> int:
    """Reset runtime state held by target and drafter model modules.

    Hooks on a module and its backend implementation are dispatched centrally.
    Shared objects are reset only once. Lazy buffers created in an inference
    forward must also be updated under inference mode during restore.
    """
    reset_count = 0
    for item in _iter_modules_and_impls(models):
        # Cache storage belongs to the layer; its layout belongs to the backend.
        # Dispatch once per owning layer, not once per shared implementation.
        restore_kv_cache = getattr(getattr(item, "impl", None), "restore_kv_cache_after_snapshot_restore", None)
        if callable(restore_kv_cache):
            restore_kv_cache(item)
        reset = getattr(item, "reset_runtime_state_after_snapshot_restore", None)
        if callable(reset):
            reset()
            reset_count += 1
    return reset_count


@torch.inference_mode()
def restore_model_non_checkpoint_state(
    model: nn.Module,
    act_dtype: torch.dtype,
    label: str,
) -> None:
    """Restore model-owned constants and lazy tensors after checkpoint loading.

    Constants hooks also repair external cache references. Lazy tensor hooks
    receive checkpoint coverage so they do not rebuild already restored values.
    Request state and native resources are handled by their own lifecycle.
    """
    restored_tensor_ids = {id(t) for t in model.parameters()}
    for module in model.modules():
        restored_tensor_ids.update(
            id(t) for name, t in module._buffers.items() if name not in module._non_persistent_buffers_set
        )
    if hasattr(model, "_snapshot_tensor_paths"):
        from vllm_ascend.snapshot.model_runner_lifecycle.tensor_inventory import inventory_tensors

        restored_tensor_ids.update(id(t) for t in inventory_tensors(model).values())
    for item in _iter_modules_and_impls((model,)):
        restore_constants = getattr(item, "restore_constants_after_snapshot_restore", None)
        if callable(restore_constants):
            restore_constants()
        restore_lazy = getattr(item, "restore_lazy_tensors_after_snapshot_restore", None)
        if callable(restore_lazy):
            restore_lazy(act_dtype, restored_tensor_ids=restored_tensor_ids)
