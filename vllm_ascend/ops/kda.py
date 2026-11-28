# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared fla_npu KDA execution; callers own projections and cache updates."""

from collections.abc import Sequence

import torch

# KDA operators from the flashserve/flash-linear-attention-npu ecosystem repository.
# https://github.com/flashserve/flash-linear-attention-npu
from fla_npu.ops.ascendc import chunk_kda_fwd, recurrent_kda

KDA_CHUNK_SIZE = 64


def run_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    lower_bound: float | None,
    beta_is_preprocessed: bool = True,
    num_accepted_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    # Recurrent KDA consumes independent Q/K/V token/head strides directly.
    output, _ = recurrent_kda(
        q,
        k,
        v,
        raw_gate.contiguous(),
        beta.contiguous(),
        state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        A_log=a_log.reshape(-1).contiguous(),
        dt_bias=dt_bias.contiguous(),
        num_accepted_tokens=num_accepted_tokens,
        layout="BSND",
        scale=q.shape[-1] ** -0.5,
        output_final_state=False,
        inplace_final_state=True,
        state_v_first=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=not beta_is_preprocessed,
        allow_neg_eigval=False,
        safe_gate=lower_bound is not None,
        lower_bound=lower_bound if lower_bound is not None else -5.0,
    )
    return output


def run_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: Sequence[int],
    chunk_indices: Sequence[int],
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    lower_bound: float | None,
    use_qk_l2norm_in_kernel: bool = False,
    checkpoint_offsets: torch.Tensor | None = None,
    checkpoint_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Consume preprocessed beta and return output plus the final VK state."""
    # FLA's import-time capability queries can create an unwanted NPU context
    # during operator discovery, before the worker selects its device.
    # TODO: Restore the top-level import once supported vLLM FLA or Triton
    # Ascend versions make these queries safe before device selection.
    from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

    if use_qk_l2norm_in_kernel:
        q = q.contiguous()
        k = k.contiguous()
    else:
        q = l2norm_fwd(q.contiguous())
        k = l2norm_fwd(k.contiguous())

    if (checkpoint_offsets is None) != (checkpoint_state is None):
        raise ValueError("KDA checkpoint offsets and state must be provided together")
    checkpoint_args = (
        {}
        if checkpoint_state is None
        else {
            "checkpoint_offsets": checkpoint_offsets,
            "checkpoint_state": checkpoint_state,
        }
    )
    output, final_state, *_ = chunk_kda_fwd(
        q,
        k,
        v.contiguous(),
        raw_gate.contiguous(),
        beta.contiguous(),
        q.shape[-1] ** -0.5,
        KDA_CHUNK_SIZE,
        layout="BSND",
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        safe_gate=lower_bound is not None,
        lower_bound=lower_bound if lower_bound is not None else -5.0,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=True,
        A_log=a_log.reshape(-1).contiguous(),
        dt_bias=dt_bias.contiguous(),
        disable_recompute=False,
        return_intermediate_states=False,
        state_v_first=True,
        **checkpoint_args,
    )
    return output, final_state


def initialize_kda_checkpoints(layer, config):
    import inspect

    from vllm.logger import logger

    from vllm_ascend.ascend_config import get_ascend_config

    # The inherited CUDA backend cannot establish an Ascend capability.
    layer._checkpoint_alignment = None
    layer._ascend_internal_checkpoint_enabled = False
    if not get_ascend_config().kda_internal_prefill_checkpoints:
        return
    from vllm.v1.worker.workspace import current_workspace_manager, is_workspace_manager_initialized

    from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

    if (
        config.cache_config.mamba_cache_mode != "align"
        or config.model_config.dtype != torch.bfloat16
        or layer.head_dim != 128
        or layer.get_state_dtype()[1] != torch.float32
        or config.parallel_config.use_ubatching
        or not get_current_hardware_profile().supports(HardwareCapability.KDA_INTERNAL_PREFILL_CHECKPOINT)
    ):
        logger.warning_once("KDA internal checkpoints are unavailable for this configuration; keeping split prefill.")
        return
    params = inspect.signature(chunk_kda_fwd).parameters
    if not {"checkpoint_offsets", "checkpoint_state"} <= params.keys():
        raise ValueError("KDA internal checkpoints require a fla_npu AscendC wheel with the FP32 checkpoint ABI")
    from vllm.model_executor.layers.mamba.checkpoint import MambaPrefillCheckpointExporter

    layer._checkpoint_alignment = KDA_CHUNK_SIZE
    layer._ascend_internal_checkpoint_enabled = True
    layer._ascend_kda_checkpoint_spec = (
        (config.scheduler_config.max_num_seqs, layer.local_num_heads, layer.head_dim, layer.head_dim),
        torch.float32,
    )
    layer._checkpoint_exporter = MambaPrefillCheckpointExporter(state_len=layer.conv_size - 1)
    # Match upstream KDA: reserve the runtime layout before KV-cache sizing.
    # The NPU worker initializes this same grow-only workspace manager.
    if is_workspace_manager_initialized():
        current_workspace_manager().get_simultaneous(layer._ascend_kda_checkpoint_spec)


def configure_kda_checkpoint_builder(builder):
    """Match Ascend's scheduler/manager policy for producer prompt hashes."""
    if not getattr(getattr(builder, "kv_cache_spec", None), "num_prefill_checkpoint_blocks", 0):
        return
    from vllm_ascend.patch.platform.patch_kv_cache_coordinator import _skips_eagle_block_drop

    if _skips_eagle_block_drop(builder.vllm_config.kv_transfer_config):
        builder.checkpoint_builder.drop_eagle_block = False


def prepare_kda_checkpoint_prefill(builder, m, metadata, draft_counts):
    """Align RecoverSSM checkpoints with live prefill rows, excluding decode/padding."""
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

    starts = builder._non_spec_cpu_starts(m, metadata, draft_counts)[metadata.num_decodes :]
    starts = starts - metadata.num_decode_tokens
    lengths = m.query_start_loc_cpu.diff()
    rows = list(range(m.num_reqs))
    if metadata.num_spec_decodes:
        # Match _non_spec_cpu_starts; offsets remain sequence-relative under PCP.
        spec = ((draft_counts >= 0) | ((~m.is_prefilling) & (lengths > 0))) & (lengths <= builder.num_spec + 1)
        rows = ((~spec) & (lengths > 0)).nonzero().flatten().tolist()
    rows = rows[metadata.num_decodes :]
    active = starts.diff() > 0
    indices_cpu = active.nonzero().flatten()
    indices = indices_cpu.to(m.query_start_loc.device, non_blocking=True)
    starts = torch.cat((starts[:1], starts[1:][active]))
    metadata.prefill_query_start_loc = starts.to(m.query_start_loc.device, non_blocking=True)
    metadata.prefill_state_indices = metadata.non_spec_state_indices_tensor[metadata.num_decodes :].index_select(
        0, indices
    )
    metadata.prefill_has_initial_state = metadata.has_initial_state[metadata.num_decodes :].index_select(0, indices)
    metadata.checkpoint = builder.checkpoint_builder.build(m, [rows[i] for i in indices_cpu.tolist()])
    if metadata.checkpoint is not None:
        metadata.checkpoint.checkpoint_offsets.masked_fill_(metadata.checkpoint.state_indices == NULL_BLOCK_ID, 0)
    return starts


def ensure_kda_checkpoint_workspace(layer):
    """Account for the attention layout even when profiling skips attention."""
    if layer._ascend_internal_checkpoint_enabled:
        from vllm.v1.worker.workspace import current_workspace_manager

        current_workspace_manager().get_simultaneous(layer._ascend_kda_checkpoint_spec)


def run_kda_checkpoint_prefill(layer, metadata, raw_qkv, q, k, v, raw_gate, beta, initial_state):
    from vllm.v1.worker.workspace import current_workspace_manager

    checkpoint = metadata.checkpoint
    (states,) = current_workspace_manager().get_simultaneous(layer._ascend_kda_checkpoint_spec)
    states = states[: initial_state.shape[0]]
    if checkpoint.checkpoint_offsets.numel() != initial_state.shape[0]:
        raise ValueError("KDA checkpoint rows must match the compact prefill batch")
    # Host plan validation never reads a device value. The shared builder has
    # already masked NULL cache slots on device, so pass that device tensor as is.
    if checkpoint.offsets is None or checkpoint.query_lens is None:
        raise ValueError("KDA checkpoints require the shared builder's host plan")
    if len(checkpoint.offsets) != initial_state.shape[0] or len(checkpoint.query_lens) != initial_state.shape[0]:
        raise ValueError("KDA checkpoint host rows must match the compact prefill batch")
    if raw_qkv is None:
        raise ValueError("KDA checkpoint export requires pre-convolution QKV")
    if any(
        offset and (offset % 64 or not 0 < offset < length)
        for offset, length in zip(checkpoint.offsets, checkpoint.query_lens)
    ):
        raise ValueError("KDA checkpoints require an internal chunk64 boundary")
    chunk = metadata.non_spec_prefill_metadata.chunk
    starts = chunk.cu_seqlens_kern or chunk.cu_seqlens_host
    output, final_state = run_chunk_kda(
        q,
        k,
        v,
        raw_gate,
        beta,
        initial_state,
        starts,
        chunk.chunk_indices_chunk64_host,
        layer.A_log,
        layer.dt_bias,
        lower_bound=layer.gate_lower_bound,
        checkpoint_offsets=checkpoint.checkpoint_offsets,
        checkpoint_state=states,
    )
    layer._checkpoint_exporter.export(
        checkpoint,
        raw_qkv=raw_qkv,
        conv_state=layer.kv_cache[0].transpose(-1, -2),
        recurrent_checkpoint=states,
        recurrent_state=layer.kv_cache[1],
        cu_seqlens=metadata.prefill_query_start_loc,
    )
    return output, final_state
