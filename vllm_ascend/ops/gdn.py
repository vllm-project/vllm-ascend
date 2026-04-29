#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging
import os

import torch
import torch_npu
from einops import rearrange

logger = logging.getLogger(__name__)
_GDN_DEBUG = bool(os.environ.get("GDN_DEBUG", ""))
_ALIGN_TRITON_CONV1D = bool(os.environ.get("GDN_ALIGN_TRITON_CONV1D", ""))


def _dbg(msg: str, *args) -> None:
    """Print debug message only when GDN_DEBUG env var is set."""
    if _GDN_DEBUG:
        print(msg % args if args else msg, flush=True)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fla.ops import (
    fused_recurrent_gated_delta_rule,
)
from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd
from vllm.model_executor.layers.mamba.gdn_linear_attn import GatedDeltaNetAttention
from vllm.triton_utils import triton
from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.attention.utils import maybe_save_kv_layer_to_connector
from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule
from vllm_ascend.ops.triton.fla.fused_qkvzba_split_reshape import fused_qkvzba_split_reshape_cat
from vllm_ascend.ops.triton.fla.sigmoid_gating import fused_sigmoid_gating_delta_rule_update
from vllm_ascend.ops.triton.fla.utils import clear_ssm_states
from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch
from vllm_ascend.ops.triton.mamba.causal_conv1d import (
    causal_conv1d_fwd_npu,
    causal_conv1d_update_npu,
)
from vllm_ascend.utils import enable_sp

# ──────────────────────────────────────────────────────────────────
# All-mode prefix caching helpers (P4)
# ──────────────────────────────────────────────────────────────────


def _nonzero_flat(mask: torch.Tensor) -> torch.Tensor:
    return torch.nonzero(mask, as_tuple=False).flatten()


def _copy_slots_if_needed(
    state_pool: torch.Tensor,
    src_slots: torch.Tensor,
    dst_slots: torch.Tensor,
) -> None:
    copy_idx = _nonzero_flat((src_slots >= 0) & (dst_slots >= 0) & (src_slots != dst_slots))
    if copy_idx.numel() == 0:
        return

    src = src_slots.index_select(0, copy_idx).long()
    dst = dst_slots.index_select(0, copy_idx).long()
    state_pool[dst] = state_pool[src]


def _build_initial_state(
    ssm_state: torch.Tensor,
    metadata,
    num_decodes: int,
    num_prefills: int,
    transpose_state: bool = True,
) -> torch.Tensor:
    """Assemble initial_state tensor for chunk kernel from pool slots.

    Reads SOURCE pool slots for both decode and prefill sequences.
    Decode uses block_state_indices[:num_decodes] (SOURCE = last-computed block).
    Prefill uses block_state_indices[num_decodes:] (SOURCE = last-computed block).
    Sequences without initial state (has_initial_state=False) are zeroed.

    Args:
        ssm_state: Pool tensor [N_pool, H, K/V, V/K]
        metadata: GDNAttentionMetadata with all-mode fields
        num_decodes: Number of decode sequences
        num_prefills: Number of prefill sequences
        transpose_state: If True, transpose pool [V,K]->[K,V] for kernel
                         (Qwen3Next=True, Qwen3.5=False)

    Returns:
        initial_state: [num_seqs, H, K, V] in kernel layout
    """
    num_seqs = num_decodes + num_prefills
    initial = ssm_state.new_zeros(num_seqs, *ssm_state.shape[1:])

    source_slots = metadata.block_state_indices  # [num_seqs]

    if num_decodes > 0:
        d_slots = source_slots[:num_decodes]
        valid_idx = _nonzero_flat(d_slots >= 0)
        if valid_idx.numel() > 0:
            state = ssm_state[d_slots.index_select(0, valid_idx).long()]
            if transpose_state:
                state = state.transpose(-1, -2).contiguous()
            initial.index_copy_(0, valid_idx, state)

    if num_prefills > 0:
        p_slots = source_slots[num_decodes:]
        valid_idx = _nonzero_flat(p_slots >= 0)
        if valid_idx.numel() > 0:
            state = ssm_state[p_slots.index_select(0, valid_idx).long()]
            if transpose_state:
                state = state.transpose(-1, -2).contiguous()
            initial.index_copy_(0, valid_idx + num_decodes, state)
        # Zero out prefill seqs without initial state
        has_init = metadata.has_initial_state
        if has_init is not None:
            no_init = ~has_init[num_decodes:]
            initial[num_decodes:][no_init] = 0

    return initial


def _write_final_states(
    ssm_state: torch.Tensor,
    final_state: torch.Tensor,
    metadata,
    num_decodes: int,
    transpose_state: bool = True,
) -> None:
    """Write kernel final_state back to DEST pool slots.

    DEST = non_spec_state_indices_tensor (overridden to last-scheduled block).

    Args:
        ssm_state: Pool tensor [N_pool, H, K/V, V/K]
        final_state: [num_seqs, H, K, V] from kernel
        metadata: GDNAttentionMetadata with all-mode fields
        num_decodes: Number of decode sequences
        transpose_state: If True, transpose kernel [K,V]->[V,K] for pool
    """
    dest_slots = metadata.non_spec_state_indices_tensor  # [num_seqs]

    if num_decodes > 0:
        d_dest = dest_slots[:num_decodes]
        valid_idx = _nonzero_flat(d_dest >= 0)
        if valid_idx.numel() > 0:
            state = final_state[:num_decodes].index_select(0, valid_idx).to(ssm_state.dtype)
            if transpose_state:
                state = state.transpose(-1, -2).contiguous()
            ssm_state[d_dest.index_select(0, valid_idx).long()] = state

    num_prefills = final_state.shape[0] - num_decodes
    if num_prefills > 0:
        p_dest = dest_slots[num_decodes:]
        valid_idx = _nonzero_flat(p_dest >= 0)
        if valid_idx.numel() > 0:
            state = final_state[num_decodes:].index_select(0, valid_idx).to(ssm_state.dtype)
            if transpose_state:
                state = state.transpose(-1, -2).contiguous()
            ssm_state[p_dest.index_select(0, valid_idx).long()] = state



def _scatter_intermediate_states(
    ssm_state: torch.Tensor,
    chunk_history: torch.Tensor,
    metadata,
    num_decodes: int,
    transpose_state: bool = True,
) -> None:
    """Write intermediate block boundary states from chunk_history to pool.

    Metadata builder precomputes a flattened scatter plan:
    - scatter_src_indices_tensor: indices into chunk_history
    - scatter_dst_slots_tensor: destination pool slots in ssm_state

    Args:
        ssm_state: Pool tensor [N_pool, H, K/V, V/K]
        chunk_history: [total_chunks, H, K, V] intermediate states from kernel
        metadata: GDNAttentionMetadata with all-mode fields
        num_decodes: Number of decode sequences
        transpose_state: If True, transpose kernel [K,V]->[V,K] for pool
    """
    del num_decodes

    src_indices = getattr(metadata, "scatter_src_indices_tensor", None)
    dst_slots = getattr(metadata, "scatter_dst_slots_tensor", None)
    if src_indices is None or dst_slots is None or src_indices.numel() == 0:
        return

    write_states = chunk_history.index_select(0, src_indices.long()).to(ssm_state.dtype)
    if dst_slots.numel() != write_states.shape[0]:
        raise RuntimeError(
            "Scatter plan mismatch: scatter_dst_slots_tensor and gathered "
            "chunk_history rows must have the same length."
        )
    if write_states.numel() == 0:
        return

    if transpose_state:
        write_states = write_states.transpose(-1, -2).contiguous()
    ssm_state[dst_slots.long()] = write_states
    return


def to_int64_tuple(tensor: torch.Tensor) -> tuple[int, ...]:
    tensor = tensor.to(torch.int64)
    if tensor.dim() == 0:
        return (tensor.item(),)
    return tuple(tensor.tolist())


def _require_non_spec_prefill_fallback_meta(attn_metadata, field_name: str):
    fallback_meta = getattr(attn_metadata, "non_spec_prefill_fallback_meta", None)
    if fallback_meta is None:
        raise RuntimeError(
            f"Expected attn_metadata.non_spec_prefill_fallback_meta.{field_name} for patched GDN non-spec prefill path."
        )
    return fallback_meta


def get_non_spec_causal_conv1d_host_args(attn_metadata) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    fallback_meta = _require_non_spec_prefill_fallback_meta(attn_metadata, "causal_conv1d")
    causal_conv1d_meta = fallback_meta.causal_conv1d
    return (
        to_int64_tuple(causal_conv1d_meta.query_start_loc_cpu),
        to_int64_tuple(causal_conv1d_meta.cache_indices_cpu),
        to_int64_tuple(causal_conv1d_meta.has_initial_state_cpu),
    )


def get_non_spec_chunked_prefill_meta(attn_metadata):
    fallback_meta = _require_non_spec_prefill_fallback_meta(attn_metadata, "chunk")
    return fallback_meta.chunk


class AscendGatedDeltaNetAttention(GatedDeltaNetAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        if not self.gqa_interleaved_layout:
            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            num_tokens = mixed_qkvz.size(0)
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            ba, _ = self.in_proj_ba(hidden_states)
            b, a = ba.chunk(2, dim=-1)

            b = b.contiguous()
            a = a.contiguous()
        else:
            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
            projected_states_ba, _ = self.in_proj_ba(hidden_states)
            num_tokens = projected_states_qkvz.size(0)

            mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat(
                projected_states_qkvz,
                projected_states_ba,
                triton.cdiv(self.num_k_heads, self.tp_size),
                triton.cdiv(self.num_v_heads, self.tp_size),
                self.head_k_dim,
                self.head_v_dim,
            )

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        # Note: we should not use torch.empty here like other attention backends,
        # see discussions in https://github.com/vllm-project/vllm/pull/28182
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            self.prefix,
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        maybe_save_kv_layer_to_connector("", [])
        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """
        Core attention computation (called by custom op).
        """
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
            # V1 profile run
            return

        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        conv_state = self_kv_cache[0].transpose(-1, -2)
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        # 1.1: Process the multi-query part
        if spec_sequence_masks is not None:
            mixed_qkv_spec = causal_conv1d_update_npu(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][: attn_metadata.num_spec_decodes],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )

        # 1.2: Process the remaining part
        is_all_mode = getattr(attn_metadata, 'is_all_mode', False)
        if is_all_mode and mixed_qkv_non_spec is not None:
            # All-mode prefix caching: use Triton fwd kernel for prefill,
            # decode kernel with APC params for decode
            num_decodes = attn_metadata.num_decodes
            num_prefills = attn_metadata.num_prefills

            if num_prefills > 0:
                # Compute SOURCE block index for conv state initial state.
                # Kernel expects block INDEX (not pool slot) into block_table_2d.
                num_comp = attn_metadata.num_computed_tokens_all
                initial_state_idx = torch.where(
                    num_comp > 0,
                    (num_comp - 1) // attn_metadata.mamba_block_size,
                    torch.zeros_like(num_comp),  # placeholder; has_initial_state=False skips read
                )
                # Prefill: Triton causal_conv1d_fwd_npu with APC params
                mixed_qkv_non_spec = causal_conv1d_fwd_npu(
                    x=mixed_qkv_non_spec,
                    weight=conv_weights,
                    bias=self.conv1d.bias,
                    conv_states=conv_state,
                    query_start_loc=non_spec_query_start_loc,
                    cache_indices=attn_metadata.block_table_2d,
                    has_initial_state=has_initial_state,
                    activation=self.activation,
                    pad_slot_id=PAD_SLOT_ID,
                    block_idx_first_scheduled_token=(
                        attn_metadata.block_idx_first_scheduled_token),
                    block_idx_last_scheduled_token=(
                        attn_metadata.block_idx_last_scheduled_token),
                    initial_state_idx=initial_state_idx,
                    num_computed_tokens=(
                        attn_metadata.num_computed_tokens_all),
                    block_size_to_align=attn_metadata.mamba_block_size,
                )
            elif num_decodes > 0:
                # Decode-only: pre-copy conv state SOURCE → DEST
                src_slots = attn_metadata.block_state_indices[:num_decodes]
                dst_slots = non_spec_state_indices_tensor[:num_decodes]
                _copy_slots_if_needed(conv_state, src_slots, dst_slots)
                mixed_qkv_non_spec = causal_conv1d_update_npu(
                    mixed_qkv_non_spec,
                    conv_state,
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                    conv_state_indices=non_spec_state_indices_tensor[
                        : attn_metadata.num_actual_tokens],
                    validate_data=True,
                )
            else:
                mixed_qkv_non_spec = None
        elif attn_metadata.num_prefills > 0:
            if mixed_qkv_non_spec is not None:
                if _ALIGN_TRITON_CONV1D:
                    # Triton conv1d for align-mode (fair benchmark vs all-mode).
                    # Same kernel as all-mode with IS_APC_ENABLED=False.
                    # 1D cache_indices works: stride=1, init/last_index=0 → SOURCE==DEST.
                    mixed_qkv_non_spec = causal_conv1d_fwd_npu(
                        x=mixed_qkv_non_spec,
                        weight=conv_weights,
                        bias=self.conv1d.bias,
                        conv_states=conv_state,
                        query_start_loc=non_spec_query_start_loc,
                        cache_indices=non_spec_state_indices_tensor,
                        has_initial_state=has_initial_state,
                        activation=self.activation,
                        pad_slot_id=PAD_SLOT_ID,
                    )
                else:
                    conv_weights_T = conv_weights.transpose(0, 1)
                    activation_num = 1 if self.activation else 0
                    (
                        query_start_loc_opt,
                        cache_indices_opt,
                        initial_state_mode_opt,
                    ) = get_non_spec_causal_conv1d_host_args(attn_metadata)
                    mixed_qkv_non_spec = torch.ops._C_ascend.npu_causal_conv1d_custom(
                        mixed_qkv_non_spec,
                        conv_weights_T,
                        conv_state=self_kv_cache[0],
                        bias_opt=self.conv1d.bias,
                        query_start_loc_opt=query_start_loc_opt,
                        cache_indices_opt=cache_indices_opt,
                        initial_state_mode_opt=initial_state_mode_opt,
                        num_accepted_tokens_opt=[],
                        activation_mode=activation_num,
                        pad_slot_id=PAD_SLOT_ID,
                        run_mode=0,
                    )
        elif attn_metadata.num_decodes > 0:
            mixed_qkv_non_spec = causal_conv1d_update_npu(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[: attn_metadata.num_actual_tokens],
                validate_data=True,
            )
        else:
            mixed_qkv_non_spec = None

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(mixed_qkv_non_spec)

        # 2. Recurrent attention
        if self.gqa_interleaved_layout:
            # Qwen3Next: torch_npu ops support float16/bf16 ssm_state.
            # g/beta are needed for both spec-decode and decode, so compute unconditionally.
            g, beta = fused_gdn_gating_patch(self.A_log, a, b, self.dt_bias)
            if spec_sequence_masks is not None:
                if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                    g_spec = g
                    beta_spec = beta
                    g_non_spec = None
                    beta_non_spec = None
                else:
                    g_spec = g.index_select(1, spec_token_indx)
                    beta_spec = beta.index_select(1, spec_token_indx)
                    g_non_spec = g.index_select(1, non_spec_token_indx)
                    beta_non_spec = beta.index_select(1, non_spec_token_indx)
            else:
                g_spec = None
                beta_spec = None
                g_non_spec = g
                beta_non_spec = beta

            # 2.1: Process the multi-query part
            if spec_sequence_masks is not None:
                cu_seqlens = spec_query_start_loc[: attn_metadata.num_spec_decodes + 1]
                actual_seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                query_spec = l2norm_fwd(query_spec)
                key_spec = l2norm_fwd(key_spec)
                core_attn_out_spec = torch_npu.npu_recurrent_gated_delta_rule(
                    query=query_spec.squeeze(0),
                    key=key_spec.squeeze(0),
                    value=value_spec.squeeze(0),
                    g=g_spec.squeeze(0),
                    beta=beta_spec.squeeze(0),
                    state=ssm_state,
                    scale=key_spec.shape[-1] ** -0.5,
                    actual_seq_lengths=actual_seq_lengths,
                    ssm_state_indices=spec_state_indices_tensor.flatten(),
                    num_accepted_tokens=num_accepted_tokens.to(torch.int32),
                ).unsqueeze(0)
            else:
                core_attn_out_spec, last_recurrent_state = None, None

            # 2.2: Process the remaining part
            if is_all_mode and attn_metadata.num_prefills > 0:
                # All-mode: read SOURCE, run chunk kernel with intermediate
                # states, write DEST + scatter intermediate block states
                # Qwen3Next: pool [V,K] ↔ kernel [K,V] → transpose_state=True
                initial_state = _build_initial_state(
                    ssm_state, attn_metadata,
                    attn_metadata.num_decodes, attn_metadata.num_prefills,
                    transpose_state=True)
                (core_attn_out_non_spec, last_recurrent_state, chunk_history) = (
                    chunk_gated_delta_rule(
                        q=query_non_spec,
                        k=key_non_spec,
                        v=value_non_spec,
                        g=g_non_spec,
                        beta=beta_non_spec,
                        initial_state=initial_state,
                        output_final_state=True,
                        cu_seqlens=non_spec_query_start_loc,
                        prebuilt_meta=get_non_spec_chunked_prefill_meta(
                            attn_metadata),
                        head_first=False,
                        use_qk_l2norm_in_kernel=True,
                        return_intermediate_states=True,
                        state_dtype=ssm_state.dtype,
                    ))
                _write_final_states(
                    ssm_state, last_recurrent_state, attn_metadata,
                    attn_metadata.num_decodes, transpose_state=True)
                if chunk_history is not None:
                    # FLA returns h as [B=1, NT, H, K, V]; squeeze batch dim
                    _scatter_intermediate_states(
                        ssm_state, chunk_history.squeeze(0), attn_metadata,
                        attn_metadata.num_decodes, transpose_state=True)
            elif is_all_mode and attn_metadata.num_decodes > 0:
                # All-mode decode-only: pre-copy SOURCE → DEST, then
                # in-place recurrent kernel reads/writes via DEST slots.
                # Qwen3Next: pool [V,K], no transpose needed for pre-copy
                # (same layout in pool, just different slots)
                src_slots = attn_metadata.block_state_indices
                dst_slots = non_spec_state_indices_tensor
                _copy_slots_if_needed(ssm_state, src_slots, dst_slots)
                cu_seqlens = non_spec_query_start_loc[: attn_metadata.num_decodes + 1]
                actual_seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                query_non_spec = l2norm_fwd(query_non_spec)
                key_non_spec = l2norm_fwd(key_non_spec)
                core_attn_out_non_spec = torch_npu.npu_recurrent_gated_delta_rule(
                    query=query_non_spec.squeeze(0),
                    key=key_non_spec.squeeze(0),
                    value=value_non_spec.squeeze(0),
                    g=g_non_spec.squeeze(0) if g_non_spec is not None else g_non_spec,
                    beta=beta_non_spec.squeeze(0) if beta_non_spec is not None else beta_non_spec,
                    state=ssm_state,
                    scale=key_non_spec.shape[-1] ** -0.5,
                    actual_seq_lengths=actual_seq_lengths,
                    ssm_state_indices=non_spec_state_indices_tensor,
                ).unsqueeze(0)
            elif attn_metadata.num_prefills > 0:
                initial_state = ssm_state[non_spec_state_indices_tensor].transpose(-1, -2).contiguous()
                clear_ssm_states(initial_state, has_initial_state)
                (core_attn_out_non_spec, last_recurrent_state, _) = chunk_gated_delta_rule(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=initial_state,
                    output_final_state=True,
                    cu_seqlens=non_spec_query_start_loc,
                    prebuilt_meta=get_non_spec_chunked_prefill_meta(attn_metadata),
                    head_first=False,
                    use_qk_l2norm_in_kernel=True,
                )
                ssm_state[non_spec_state_indices_tensor] = (
                    last_recurrent_state.transpose(-1, -2).contiguous().to(ssm_state.dtype)
                )
            elif attn_metadata.num_decodes > 0:
                cu_seqlens = non_spec_query_start_loc[: attn_metadata.num_decodes + 1]
                actual_seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                query_non_spec = l2norm_fwd(query_non_spec)
                key_non_spec = l2norm_fwd(key_non_spec)
                core_attn_out_non_spec = torch_npu.npu_recurrent_gated_delta_rule(
                    query=query_non_spec.squeeze(0),
                    key=key_non_spec.squeeze(0),
                    value=value_non_spec.squeeze(0),
                    g=g_non_spec.squeeze(0) if g_non_spec is not None else g_non_spec,
                    beta=beta_non_spec.squeeze(0) if beta_non_spec is not None else beta_non_spec,
                    state=ssm_state,
                    scale=key_non_spec.shape[-1] ** -0.5,
                    actual_seq_lengths=actual_seq_lengths,
                    ssm_state_indices=non_spec_state_indices_tensor,
                ).unsqueeze(0)
        else:
            # Qwen3.5: torch_npu ops do not support float32 ssm_state, use FLA ops instead.
            # NOTE: Once torch_npu supports float32 ssm_state, this branch can be removed.
            if attn_metadata.num_prefills > 0 or spec_sequence_masks is not None:
                g, beta = fused_gdn_gating_patch(self.A_log, a, b, self.dt_bias)
                if spec_sequence_masks is not None:
                    if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                        g_spec = g
                        beta_spec = beta
                        g_non_spec = None
                        beta_non_spec = None
                    else:
                        g_spec = g.index_select(1, spec_token_indx)
                        beta_spec = beta.index_select(1, spec_token_indx)
                        g_non_spec = g.index_select(1, non_spec_token_indx)
                        beta_non_spec = beta.index_select(1, non_spec_token_indx)
                else:
                    g_spec = None
                    beta_spec = None
                    g_non_spec = g
                    beta_non_spec = beta

                # 2.1: Process the multi-query part
                if spec_sequence_masks is not None:
                    core_attn_out_spec, last_recurrent_state = fused_recurrent_gated_delta_rule(
                        q=query_spec,
                        k=key_spec,
                        v=value_spec,
                        g=g_spec,
                        beta=beta_spec,
                        initial_state=ssm_state,
                        inplace_final_state=True,
                        cu_seqlens=spec_query_start_loc[: attn_metadata.num_spec_decodes + 1],
                        ssm_state_indices=spec_state_indices_tensor,
                        num_accepted_tokens=num_accepted_tokens,
                        use_qk_l2norm_in_kernel=True,
                    )
                else:
                    core_attn_out_spec, last_recurrent_state = None, None

                # 2.2: Process the remaining part
                if is_all_mode and attn_metadata.num_prefills > 0:
                    # All-mode: Qwen3.5 pool [K,V] == kernel [K,V] → no transpose
                    initial_state = _build_initial_state(
                        ssm_state,
                        attn_metadata,
                        attn_metadata.num_decodes,
                        attn_metadata.num_prefills,
                        transpose_state=False,
                    )
                    (core_attn_out_non_spec, last_recurrent_state, chunk_history) = chunk_gated_delta_rule(
                        q=query_non_spec,
                        k=key_non_spec,
                        v=value_non_spec,
                        g=g_non_spec,
                        beta=beta_non_spec,
                        initial_state=initial_state,
                        output_final_state=True,
                        cu_seqlens=non_spec_query_start_loc,
                        prebuilt_meta=get_non_spec_chunked_prefill_meta(attn_metadata),
                        head_first=False,
                        use_qk_l2norm_in_kernel=True,
                        return_intermediate_states=True,
                        state_dtype=ssm_state.dtype,
                    )
                    _write_final_states(
                        ssm_state,
                        last_recurrent_state,
                        attn_metadata,
                        attn_metadata.num_decodes,
                        transpose_state=False,
                    )
                    if chunk_history is not None:
                        _scatter_intermediate_states(
                            ssm_state,
                            chunk_history.squeeze(0),
                            attn_metadata,
                            attn_metadata.num_decodes,
                            transpose_state=False,
                        )
                elif is_all_mode and attn_metadata.num_decodes > 0:
                    # All-mode decode-only: pre-copy SOURCE → DEST, then
                    # in-place fused_recurrent reads/writes via DEST slots.
                    # Qwen3.5: pool [K,V], no transpose for pre-copy
                    src_slots = attn_metadata.block_state_indices
                    dst_slots = non_spec_state_indices_tensor
                    _copy_slots_if_needed(ssm_state, src_slots, dst_slots)
                    core_attn_out_non_spec, last_recurrent_state = fused_recurrent_gated_delta_rule(
                        q=query_non_spec,
                        k=key_non_spec,
                        v=value_non_spec,
                        g=g_non_spec,
                        beta=beta_non_spec,
                        initial_state=ssm_state,
                        inplace_final_state=True,
                        cu_seqlens=non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                        ssm_state_indices=non_spec_state_indices_tensor,
                        use_qk_l2norm_in_kernel=True,
                    )
                elif attn_metadata.num_prefills > 0:
                    initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
                    clear_ssm_states(initial_state, has_initial_state)
                    (core_attn_out_non_spec, last_recurrent_state, _) = chunk_gated_delta_rule(
                        q=query_non_spec,
                        k=key_non_spec,
                        v=value_non_spec,
                        g=g_non_spec,
                        beta=beta_non_spec,
                        initial_state=initial_state,
                        output_final_state=True,
                        cu_seqlens=non_spec_query_start_loc,
                        prebuilt_meta=get_non_spec_chunked_prefill_meta(attn_metadata),
                        head_first=False,
                        use_qk_l2norm_in_kernel=True,
                    )
                    ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(ssm_state.dtype)
                elif attn_metadata.num_decodes > 0:
                    core_attn_out_non_spec, last_recurrent_state = fused_recurrent_gated_delta_rule(
                        q=query_non_spec,
                        k=key_non_spec,
                        v=value_non_spec,
                        g=g_non_spec,
                        beta=beta_non_spec,
                        initial_state=ssm_state,
                        inplace_final_state=True,
                        cu_seqlens=non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                        ssm_state_indices=non_spec_state_indices_tensor,
                        use_qk_l2norm_in_kernel=True,
                    )
                else:
                    core_attn_out_non_spec, last_recurrent_state = None, None
            elif attn_metadata.num_decodes > 0:
                core_attn_out_spec = None
                # All-mode: pre-copy SOURCE → DEST before in-place kernel
                if is_all_mode:
                    src_slots = attn_metadata.block_state_indices
                    dst_slots = non_spec_state_indices_tensor
                    _copy_slots_if_needed(ssm_state, src_slots, dst_slots)
                core_attn_out_non_spec = fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log.contiguous(),
                    dt_bias=self.dt_bias.contiguous(),
                    q=query_non_spec.contiguous(),
                    k=key_non_spec.contiguous(),
                    v=value_non_spec.contiguous(),
                    a=a.contiguous(),
                    b=b.contiguous(),
                    initial_state_source=ssm_state,
                    initial_state_indices=non_spec_state_indices_tensor,
                    cu_seqlens=non_spec_query_start_loc,
                    use_qk_l2norm_in_kernel=True,
                    softplus_beta=1.0,
                    softplus_threshold=20.0,
                )
            else:
                core_attn_out_spec, core_attn_out_non_spec = None, None
            maybe_save_kv_layer_to_connector("", [])

        # 3. Merge core attention output
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            if not enable_sp():
                core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
            else:
                core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)[:num_actual_tokens]
        elif spec_sequence_masks is not None:
            if not enable_sp():
                core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
            else:
                core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)[:num_actual_tokens]
        else:
            if not enable_sp():
                core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)
            else:
                core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)[:num_actual_tokens]
