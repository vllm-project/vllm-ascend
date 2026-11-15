# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""GDN checkpoint export from FP32 chunk-start recurrent states."""

import torch
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule_checkpoint

GDN_CHECKPOINT_ALIGNMENT = 64


def _supports_checkpoint_boundary_policy(config) -> bool:
    # Ascend's connector-specific split rules and legacy EAGLE block-drop
    # overrides can disagree with the upstream worker checkpoint planner.
    # Never advertise a cache page whose token boundary we cannot reproduce.
    if config.kv_transfer_config is not None:
        return False
    spec = config.speculative_config
    return spec is None or bool(getattr(config, "num_hash_lookahead_tokens", 0)) or not spec.use_eagle_block_drop()


def initialize_gdn_checkpoints(layer) -> None:
    if not get_ascend_config().gdn_internal_prefill_checkpoints:
        return
    # Do not advertise unsupported layers: the upstream all-layer capability
    # check keeps the scheduler's split fallback when any layer cannot export.
    config = layer.vllm_config
    if (
        config.cache_config.mamba_cache_mode != "align"
        or config.parallel_config.prefill_context_parallel_size != 1
        or config.model_config.dtype != torch.bfloat16
        or config.parallel_config.use_ubatching
        or layer.head_k_dim != 128
        or layer.head_v_dim != 128
    ):
        return
    if not _supports_checkpoint_boundary_policy(config):
        logger.warning_once(
            "GDN internal prefill checkpoints use existing prefill scheduling with "
            "KV transfer or legacy EAGLE block dropping without lookahead hashes."
        )
        return
    from vllm.model_executor.layers.mamba.checkpoint import MambaPrefillCheckpointExporter
    from vllm.v1.kv_cache_interface import MambaSpec

    if "num_prefill_checkpoint_blocks" not in MambaSpec.__dataclass_fields__:
        raise ValueError("GDN internal checkpoints require vLLM with PR #60050")
    layer._ascend_internal_checkpoint_enabled = True
    layer.register_buffer("_ascend_checkpoint_workspace", None, persistent=False)
    layer._checkpoint_alignment = GDN_CHECKPOINT_ALIGNMENT
    layer._max_num_batched_tokens = config.scheduler_config.max_num_batched_tokens
    layer._checkpoint_exporter = MambaPrefillCheckpointExporter(state_len=layer.conv_kernel_size - 1)


def ensure_gdn_checkpoint_workspace(layer, tensor):
    if getattr(layer, "_exports_checkpoints", False) and layer._ascend_checkpoint_workspace is None:
        # Allocate during the profiling forward, including when its metadata is
        # None, so KV-cache sizing accounts for the persistent FP32 snapshots.
        shape = (
            layer.vllm_config.scheduler_config.max_num_seqs,
            layer.num_v_heads // layer.tp_size,
            layer.head_v_dim,
            layer.head_k_dim,
        )
        # With ubatching excluded, layer export is ordered before the next
        # layer's recurrence on the worker stream. Share scratch, not KV state.
        context = getattr(layer.vllm_config.compilation_config, "static_forward_context", None) or {}
        for candidate in context.values():
            workspace = getattr(candidate, "_ascend_checkpoint_workspace", None)
            if (
                workspace is not None
                and getattr(candidate, "vllm_config", None) is layer.vllm_config
                and tuple(workspace.shape) == shape
                and workspace.dtype == torch.float32
                and workspace.device == tensor.device
            ):
                layer._ascend_checkpoint_workspace = workspace
                return
        layer._ascend_checkpoint_workspace = tensor.new_empty(shape, dtype=torch.float32)


def run_gdn_checkpoint_prefill(layer, metadata, raw_qkv, q, k, v, g, beta, initial_state):
    checkpoint = metadata.checkpoint
    cu_seqlens = metadata.prefill_query_start_loc
    states = None if checkpoint is None else layer._ascend_checkpoint_workspace[: initial_state.shape[0]]
    output, final_state = chunk_gated_delta_rule_checkpoint(
        q,
        k,
        v,
        g,
        beta,
        initial_state.transpose(-1, -2),
        cu_seqlens,
        metadata.non_spec_prefill_metadata.chunk,
        states,
        None if checkpoint is None else checkpoint.checkpoint_offsets,
        checkpoint_state_dtype=layer.kv_cache[1].dtype,
    )
    if checkpoint is not None:
        # Ascend's conv state is [page, history, channel]; the shared exporter
        # accepts [page, channel, history]. Preserve its padded page strides.
        layer._checkpoint_exporter.export(
            checkpoint,
            raw_qkv=raw_qkv,
            conv_state=layer.kv_cache[0].transpose(-1, -2),
            recurrent_checkpoint=states,
            recurrent_state=layer.kv_cache[1],
            cu_seqlens=cu_seqlens,
        )
    return output, final_state.transpose(-1, -2).contiguous()
