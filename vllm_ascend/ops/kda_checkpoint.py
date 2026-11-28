# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
"""Shared Ascend KDA prefill checkpoint capability checks, buffer allocation, and export helpers."""

from __future__ import annotations

import copy
from dataclasses import is_dataclass, replace
from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend.ops.triton.kda.kda import chunk_kda_with_fused_gate
from vllm_ascend.ops.triton.kda.utils import FLA_CHUNK_SIZE

KDA_PREFILL_CHECKPOINT_ALIGNMENT = FLA_CHUNK_SIZE
KDA_FLASH_CHECKPOINT_ALIGNMENT = 16
KDA_MAX_HEAD_DIM = 256
MIN_KDA_CONV_SIZE = 2
CHECKPOINT_TENSOR_FIELDS = ("checkpoint_offsets", "state_indices", "request_rows", "block_cols")
CHECKPOINT_LIST_FIELDS = ("offsets", "query_lens")

_KDA_ALIGNMENT_BY_BACKEND = {
    "flashkda": KDA_FLASH_CHECKPOINT_ALIGNMENT,
    "triton": KDA_PREFILL_CHECKPOINT_ALIGNMENT,
}

TritonKDACheckpointSpec = tuple[tuple[int, ...], torch.dtype]


def clone_with_updates(obj: Any, updates: dict[str, Any]) -> Any:
    """Shallow-copy a dataclass or namespace-like checkpoint object with updated fields."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return replace(obj, **updates)
    cloned = copy.copy(obj)
    for field_name, value in updates.items():
        setattr(cloned, field_name, value)
    return cloned


def kda_prefill_checkpoint_alignment(backend: str = "triton") -> int | None:
    """Return the token alignment for KDA internal prefill checkpoints."""
    return _KDA_ALIGNMENT_BY_BACKEND.get(backend)


def _supports_checkpoint_boundary_policy(vllm_config: Any) -> bool:
    if getattr(vllm_config, "kv_transfer_config", None) is not None:
        return False
    spec = getattr(vllm_config, "speculative_config", None)
    if spec is None or bool(getattr(vllm_config, "num_hash_lookahead_tokens", 0)):
        return True
    use_drop = getattr(spec, "use_eagle_block_drop", None)
    return not use_drop() if callable(use_drop) else True


def _kda_internal_prefill_checkpoints_enabled(vllm_config: Any) -> bool:
    """Return whether KDA internal prefill checkpoints are opted in."""
    try:
        from vllm_ascend.ascend_config import get_ascend_config

        return bool(get_ascend_config().kda_internal_prefill_checkpoints)
    except RuntimeError:
        additional = getattr(vllm_config, "additional_config", None)
        if not isinstance(additional, dict):
            return False
        val = additional.get("kda_internal_prefill_checkpoints", False)
        return val is True or (isinstance(val, str) and val.strip().lower() == "true")


def supports_kda_prefill_checkpoint(
    vllm_config: Any,
    *,
    head_dim: int,
    conv_size: int,
    local_num_heads: int,
) -> bool:
    """Return whether KDA internal prefill checkpoints can run safely for the layer/config."""
    if vllm_config is None or not _kda_internal_prefill_checkpoints_enabled(vllm_config):
        return False
    cache_cfg = getattr(vllm_config, "cache_config", None)
    parallel_cfg = getattr(vllm_config, "parallel_config", None)
    sched_cfg = getattr(vllm_config, "scheduler_config", None)
    if (
        parallel_cfg is None
        or getattr(cache_cfg, "mamba_cache_mode", None) != "align"
        or getattr(cache_cfg, "use_replayssm", False)
        or getattr(cache_cfg, "use_kda_recoverssm", False)
        or getattr(parallel_cfg, "prefill_context_parallel_size", 1) != 1
        or getattr(parallel_cfg, "use_ubatching", False)
        or getattr(sched_cfg, "max_num_seqs", 0) <= 0
        or not (0 < head_dim <= KDA_MAX_HEAD_DIM)
        or conv_size < MIN_KDA_CONV_SIZE
        or local_num_heads <= 0
    ):
        return False
    if not _supports_checkpoint_boundary_policy(vllm_config):
        logger.warning_once(
            "KDA internal prefill checkpoints use existing prefill scheduling with "
            "KV transfer or legacy EAGLE block dropping without lookahead hashes."
        )
        return False

    from vllm.v1.kv_cache_interface import MambaSpec

    return "num_prefill_checkpoint_blocks" in MambaSpec.__dataclass_fields__


def reserve_triton_kda_checkpoint_buffer(
    vllm_config: Any, num_heads: int, head_dim: int, dtype: torch.dtype
) -> TritonKDACheckpointSpec:
    """Reserve the `[max_num_seqs, H, V, K]` workspace spec for Triton KDA checkpoints."""
    spec: TritonKDACheckpointSpec = ((vllm_config.scheduler_config.max_num_seqs, num_heads, head_dim, head_dim), dtype)
    from vllm.v1.worker.workspace import current_workspace_manager, is_workspace_manager_initialized

    if is_workspace_manager_initialized():
        current_workspace_manager().get_simultaneous(spec)
    return spec


def get_triton_kda_checkpoint_buffer(
    checkpoint: Any | None,
    spec: TritonKDACheckpointSpec | None,
    num_seqs: int,
    *,
    reference_tensor: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return `(checkpoint_state, checkpoint_offsets)` for a KDA prefill call or `(None, None)`."""
    if checkpoint is None:
        return None, None
    assert spec is not None
    shape, dtype = spec
    from vllm.v1.worker.workspace import current_workspace_manager, is_workspace_manager_initialized

    if is_workspace_manager_initialized():
        (checkpoint_state,) = current_workspace_manager().get_simultaneous(spec)
    elif reference_tensor is not None:
        checkpoint_state = reference_tensor.new_empty((num_seqs, *shape[1:]), dtype=dtype)
    else:
        raise RuntimeError("KDA prefill checkpoint workspace requires initialized workspace manager.")
    return checkpoint_state[:num_seqs], checkpoint.checkpoint_offsets


def initialize_kda_checkpoints(layer: Any, vllm_config: Any | None = None) -> bool:
    """Initialize KDA internal prefill checkpoint attributes safely on an attention layer."""
    layer._ascend_internal_checkpoint_enabled = False
    layer._checkpoint_exporter = None
    layer._triton_checkpoint_spec = None

    cfg = vllm_config if vllm_config is not None else getattr(layer, "vllm_config", None)
    head_dim = getattr(layer, "head_dim", 0)
    conv_size = getattr(layer, "conv_size", 0)
    local_num_heads = getattr(layer, "local_num_heads", 0)
    if not supports_kda_prefill_checkpoint(
        cfg, head_dim=head_dim, conv_size=conv_size, local_num_heads=local_num_heads
    ):
        return False

    get_state_dtype = getattr(layer, "get_state_dtype", None)
    if not callable(get_state_dtype):
        return False
    state_dtypes = get_state_dtype()
    if len(state_dtypes) < 2:
        return False

    from vllm.model_executor.layers.mamba.checkpoint import MambaPrefillCheckpointExporter

    layer._ascend_internal_checkpoint_enabled = True
    layer._checkpoint_exporter = MambaPrefillCheckpointExporter(state_len=conv_size - 1)
    layer._triton_checkpoint_spec = reserve_triton_kda_checkpoint_buffer(
        cfg, local_num_heads, head_dim, state_dtypes[1]
    )
    return True


def build_kda_kv_cache_spec(layer: Any, vllm_config: Any, base_spec: Any) -> Any:
    """Upgrade or clear KDA `MambaSpec` checkpoint fields to match layer opt-in state."""
    from vllm.v1.kv_cache_interface import MambaSpec

    if not isinstance(base_spec, MambaSpec) or "num_prefill_checkpoint_blocks" not in MambaSpec.__dataclass_fields__:
        return base_spec

    has_align = "prefill_checkpoint_alignment" in MambaSpec.__dataclass_fields__
    if not getattr(layer, "_ascend_internal_checkpoint_enabled", False):
        target_blocks, target_align = 0, None
    elif base_spec.mamba_cache_mode == "align":
        target_blocks, target_align = 1, KDA_PREFILL_CHECKPOINT_ALIGNMENT
    else:
        return base_spec

    updates: dict[str, Any] = {}
    if getattr(base_spec, "num_prefill_checkpoint_blocks", 0) != target_blocks:
        updates["num_prefill_checkpoint_blocks"] = target_blocks
    if has_align and getattr(base_spec, "prefill_checkpoint_alignment", None) != target_align:
        updates["prefill_checkpoint_alignment"] = target_align
    return replace(base_spec, **updates) if updates else base_spec


def export_kda_prefill_checkpoint(
    exporter: Any,
    checkpoint: Any,
    *,
    raw_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    conv_state_is_dim_first: bool,
    recurrent_checkpoint: torch.Tensor | None,
    recurrent_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> None:
    """Normalize conv-state layout to `[block, dim, state_len]` and export KDA checkpoints."""
    normalized_conv_state = conv_state if conv_state_is_dim_first else conv_state.transpose(-1, -2)
    exporter.export(
        checkpoint,
        raw_qkv=raw_qkv,
        conv_state=normalized_conv_state,
        recurrent_checkpoint=recurrent_checkpoint,
        recurrent_state=recurrent_state,
        cu_seqlens=cu_seqlens,
    )


def _as_device_cu_seqlens(cu_seqlens: Any, device: torch.device) -> torch.Tensor:
    if isinstance(cu_seqlens, torch.Tensor):
        return cu_seqlens if cu_seqlens.device == device else cu_seqlens.to(device=device)
    return torch.tensor(cu_seqlens, dtype=torch.int32, device=device)


def _derive_host_keep_mask(cu_seqlens_host: Any, cu_seqlens_kern: Any, expected_len: int) -> list[bool] | None:
    if (
        not isinstance(cu_seqlens_host, (tuple, list))
        or not isinstance(cu_seqlens_kern, (tuple, list))
        or len(cu_seqlens_host) != expected_len + 1
    ):
        return None
    mask = [end > start for start, end in zip(cu_seqlens_host[:-1], cu_seqlens_host[1:])]
    derived = [cu_seqlens_host[0]] + [end for end, keep in zip(cu_seqlens_host[1:], mask) if keep]
    return mask if list(cu_seqlens_kern) == derived else None


def _slice_checkpoint_metadata(
    checkpoint: Any,
    keep_meta: torch.Tensor,
    expected_len: int,
    *,
    keep_meta_host: torch.Tensor | list[bool] | None = None,
) -> Any:
    ckpt_offsets = getattr(checkpoint, "checkpoint_offsets", None)
    if not isinstance(ckpt_offsets, torch.Tensor) or ckpt_offsets.ndim < 1 or ckpt_offsets.shape[0] != expected_len:
        return checkpoint

    updates: dict[str, Any] = {
        name: val[keep_meta]
        for name in CHECKPOINT_TENSOR_FIELDS
        if isinstance((val := getattr(checkpoint, name, None)), torch.Tensor)
        and val.ndim >= 1
        and val.shape[0] == expected_len
    }
    keep_indices: list[Any] | None = None
    is_bool_mask = False
    for name in CHECKPOINT_LIST_FIELDS:
        val = getattr(checkpoint, name, None)
        if isinstance(val, (tuple, list)) and len(val) == expected_len:
            if keep_indices is None:
                if isinstance(keep_meta_host, list):
                    keep_indices, is_bool_mask = keep_meta_host, True
                elif isinstance(keep_meta_host, torch.Tensor):
                    keep_indices, is_bool_mask = keep_meta_host.tolist(), keep_meta_host.dtype == torch.bool
                else:
                    keep_indices, is_bool_mask = keep_meta.tolist(), keep_meta.dtype == torch.bool
            filtered = [x for x, k in zip(val, keep_indices) if k] if is_bool_mask else [val[i] for i in keep_indices]
            updates[name] = tuple(filtered) if isinstance(val, tuple) else filtered
    return clone_with_updates(checkpoint, updates) if updates else checkpoint


def prepare_kda_checkpoint_prefill_inputs(
    *,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    prefill_query_start_loc: torch.Tensor | None,
    chunk_meta: Any,
    checkpoint: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, Any]:
    """Prepare aligned KDA checkpoint prefill metadata and compacted device tensors."""
    device = state_indices.device
    keep_meta = getattr(chunk_meta, "keep_meta", None)
    cu_seqlens_kern = getattr(chunk_meta, "cu_seqlens_kern", None)
    if keep_meta is None:
        cu_source = (
            prefill_query_start_loc
            if isinstance(prefill_query_start_loc, torch.Tensor)
            else (cu_seqlens_kern if cu_seqlens_kern is not None else getattr(chunk_meta, "cu_seqlens_host", None))
        )
        if cu_source is None:
            raise ValueError("KDA prefill checkpoint requires cu_seqlens metadata")
        return state_indices, has_initial_state, _as_device_cu_seqlens(cu_source, device), checkpoint, chunk_meta

    keep_meta_host: torch.Tensor | list[bool] | None = None
    if isinstance(keep_meta, torch.Tensor):
        if keep_meta.device.type == "cpu":
            keep_meta_host = keep_meta
        elif keep_meta.dtype == torch.bool:
            keep_meta_host = _derive_host_keep_mask(
                getattr(chunk_meta, "cu_seqlens_host", None), cu_seqlens_kern, state_indices.shape[0]
            )
        if keep_meta.device != device:
            keep_meta = keep_meta.to(device=device, non_blocking=True)

    expected_len = state_indices.shape[0]
    checkpoint = _slice_checkpoint_metadata(checkpoint, keep_meta, expected_len, keep_meta_host=keep_meta_host)
    if cu_seqlens_kern is not None:
        cu_seqlens = _as_device_cu_seqlens(cu_seqlens_kern, device)
    elif prefill_query_start_loc is not None:
        base_cu = _as_device_cu_seqlens(prefill_query_start_loc, device)
        cu_seqlens = torch.cat((base_cu[:1], base_cu[1:][keep_meta]))
    else:
        raise ValueError("KDA prefill checkpoint requires cu_seqlens metadata")
    # Prebuilt chunk_indices/offsets were computed on uncompacted cu_seqlens_cpu.
    return state_indices[keep_meta], has_initial_state[keep_meta], cu_seqlens, checkpoint, None


def run_kda_checkpoint_prefill(
    *,
    exporter: Any,
    checkpoint_spec: TritonKDACheckpointSpec | None,
    checkpoint: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    raw_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    conv_state_is_dim_first: bool,
    recurrent_state: torch.Tensor,
    beta: torch.Tensor | None = None,
    raw_beta: torch.Tensor | None = None,
    safe_gate: bool | None = None,
    lower_bound: float | None = None,
    prebuilt_meta: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Triton KDA prefill with checkpoint capture and export conv/recurrent states."""
    checkpoint_state, checkpoint_offsets = get_triton_kda_checkpoint_buffer(
        checkpoint, checkpoint_spec, initial_state.shape[0], reference_tensor=recurrent_state
    )
    output, final_state = chunk_kda_with_fused_gate(
        q=q,
        k=k,
        v=v,
        raw_g=raw_gate,
        beta=beta,
        raw_beta=raw_beta,
        A_log=a_log,
        g_bias=dt_bias,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        prebuilt_meta=prebuilt_meta,
        checkpoint_state=checkpoint_state,
        checkpoint_offsets=checkpoint_offsets,
    )
    if checkpoint is not None:
        export_kda_prefill_checkpoint(
            exporter,
            checkpoint,
            raw_qkv=raw_qkv,
            conv_state=conv_state,
            conv_state_is_dim_first=conv_state_is_dim_first,
            recurrent_checkpoint=checkpoint_state,
            recurrent_state=recurrent_state,
            cu_seqlens=cu_seqlens,
        )
    return output, final_state
