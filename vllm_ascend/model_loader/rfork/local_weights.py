# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep DeepSeek Engram host loading and readiness inside RFork."""

import sys
import weakref
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class _EngramLocalState:
    owner: weakref.ReferenceType[nn.Module]
    checkpoint: tuple[Any, str]
    tensors: tuple[tuple[weakref.ReferenceType[torch.Tensor], int], ...] = ()


def _engram_tables(model: nn.Module | None) -> Iterator[Any]:
    # Avoid importing NPU model dependencies for unrelated architectures.
    module = sys.modules.get("vllm_ascend.models.deepseek_v41.engram.embedding")
    table_cls = getattr(module, "AscendParallelEngramEmbedding", None)
    if table_cls is None or not isinstance(model, nn.Module):
        return
    for table in model.modules():
        if isinstance(table, table_cls) and table.cpu_offload:
            yield table


def _table_tensors(table: Any) -> tuple[torch.Tensor, ...]:
    return tuple(tensor for tensor in (table.weight, table.weight_scale_inv) if tensor is not None)


def _ready_tensors(table: Any) -> tuple[torch.Tensor, ...]:
    state = getattr(table, "_rfork_engram_state", None)
    if not isinstance(state, _EngramLocalState):
        return ()
    if state.checkpoint != (table._checkpoint_path, table._checkpoint_key):
        return ()
    tensors = _table_tensors(table)
    if len(tensors) != len(state.tensors):
        return ()
    for tensor, (loaded_ref, data_ptr) in zip(tensors, state.tensors):
        if tensor is not loaded_ref() or tensor.device.type != "cpu" or tensor.data_ptr() != data_ptr:
            return ()
    return tensors


def _new_state(model: nn.Module, table: Any) -> _EngramLocalState:
    state = _EngramLocalState(weakref.ref(model), (table._checkpoint_path, table._checkpoint_key))
    table._rfork_engram_state = state
    return state


def _record_tensors(state: _EngramLocalState, table: Any) -> None:
    # Weak refs must not pin replaced tensors or released CANN allocations.
    state.tensors = tuple((weakref.ref(tensor), tensor.data_ptr()) for tensor in _table_tensors(table))


def load_local_weights(model: nn.Module) -> None:
    """Load known host tables before transfer registration or draft reuse."""
    for table in _engram_tables(model):
        if _ready_tensors(table):
            continue
        state = _new_state(model, table)
        table.load_checkpoint(*state.checkpoint)
        # A partial load must never exempt the weight/scale pair from the guard.
        _record_tensors(state, table)


def record_loaded_local_weights(model: nn.Module | None) -> None:
    """Record tables only after the fallback loader returns a complete model."""
    if model is None:
        return
    for table in _engram_tables(model):
        if not _ready_tensors(table):
            _record_tensors(_new_state(model, table), table)


def get_local_tensor_ids(model: nn.Module) -> set[int]:
    """Exempt only still-live tensors whose local loading completed."""
    return {id(tensor) for table in _engram_tables(model) for tensor in _ready_tensors(table)}


def release_local_weights(model: nn.Module) -> None:
    """Release discarded host tables after RFork unregisters transfer memory."""
    for table in _engram_tables(model):
        state = getattr(table, "_rfork_engram_state", None)
        if isinstance(state, _EngramLocalState):
            # A draft may share the target table; never release target storage.
            if state.owner() is not model:
                continue
            state.tensors = ()
        table.close_host_offload()
