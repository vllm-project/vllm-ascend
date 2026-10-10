# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend backend for the Bailing/Ling V3 Kimi Delta Attention layer."""

import torch
from einops import rearrange
from fla_npu.ops.ascendc import causal_conv1d_fn, causal_conv1d_update
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.models.bailing_moe_v3 import BailingMoeV3KimiDeltaAttention
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionBackend
from vllm_ascend.ops.kda import run_chunk_kda, run_recurrent_kda
from vllm_ascend.ops.triton.fla.utils import clear_ssm_states
from vllm_ascend.ops.triton.kda.conv_state import copy_conv_state

_KDA_MAX_RECURRENT_TOKENS = 8


def _zero_padded_spec_output(output: torch.Tensor, query_start_loc: torch.Tensor) -> torch.Tensor:
    """Clear graph-padding rows that the recurrent kernel does not write."""
    token_indices = torch.arange(output.shape[1], dtype=query_start_loc.dtype, device=output.device)
    valid_tokens = token_indices < query_start_loc[-1]
    return torch.where(valid_tokens.view(1, -1, 1, 1), output, 0.0)


class AscendBailingMoeV3KimiDeltaAttention(BailingMoeV3KimiDeltaAttention):
    """Run Bailing/Ling V3 KDA with the current Ascend GDN metadata and ops."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # The device kernel skips sequences longer than MAX_MTP without a
        # host-side error, so reject an unsupported verification width here.
        if self.num_speculative_tokens + 1 > _KDA_MAX_RECURRENT_TOKENS:
            raise ValueError(
                "Bailing/Ling V3 Ascend KDA supports at most "
                f"{_KDA_MAX_RECURRENT_TOKENS - 1} speculative tokens, got {self.num_speculative_tokens}."
            )
        self._conv_state_dim_first = is_conv_state_dim_first()

    def get_attn_backend(self) -> type[AttentionBackend]:
        return AscendGDNAttentionBackend

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        del positions
        num_tokens = hidden_states.size(0)
        if self.separate_b_proj:
            assert self.qkv_proj is not None and self.b_proj is not None
            qkv = self.qkv_proj(hidden_states)[0]
            q, k, v = qkv.split(self.projection_size_per_partition, dim=-1)
            beta_logits = self.b_proj(hidden_states)[0]
        else:
            assert self.qkvb_proj is not None
            qkvb = self.qkvb_proj(hidden_states)[0]
            q, k, v, beta_logits = qkvb.split(
                [
                    self.projection_size_per_partition,
                    self.projection_size_per_partition,
                    self.projection_size_per_partition,
                    self.local_num_heads,
                ],
                dim=-1,
            )

        beta = beta_logits.float().sigmoid().unsqueeze(0)
        # AscendC performs the A_log/dt_bias gate conversion in-kernel. Passing
        # vLLM's CUDA-preprocessed gate here would apply that conversion twice.
        raw_gate = rearrange(self.f_proj(hidden_states)[0], "n (h d) -> 1 n h d", d=self.head_dim)
        output_gate = rearrange(
            self.g_proj(hidden_states)[0],
            "n (h d) -> n h d",
            d=self.head_dim,
        )

        core_attn_out = torch.zeros(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.vllm.bailing_v3_kda_attention(q, k, v, raw_gate, beta, core_attn_out, self.prefix)
        core_attn_out = self.o_norm(core_attn_out, output_gate)
        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        output[:] = self.o_proj(core_attn_out)[0]

    def _conv_weights_t(self, dtype: torch.dtype) -> torch.Tensor:
        weights = [
            conv.weight.view(conv.weight.size(0), conv.weight.size(2))
            for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d)
        ]
        return torch.cat(weights, dim=0).transpose(0, 1).to(dtype=dtype).contiguous()

    def _run_causal_conv1d(
        self,
        mixed_qkv: torch.Tensor,
        conv_weights_t: torch.Tensor,
        conv_state_storage: torch.Tensor,
        query_start_loc: torch.Tensor,
        cache_indices: torch.Tensor,
        initial_state_mode: torch.Tensor | None,
        *,
        run_mode: int,
        max_query_len: int = -1,
        num_accepted_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if run_mode not in (0, 1):
            raise ValueError(f"Unsupported causal_conv1d run_mode: {run_mode}")
        # Convolution uses one history buffer per request, including MTP.
        if cache_indices.dim() == 2:
            cache_indices = cache_indices[:, 0]
        elif cache_indices.dim() != 1:
            raise ValueError(f"KDA causal cache_indices must be 1D or 2D, got shape={tuple(cache_indices.shape)}")
        cache_indices = cache_indices.contiguous()
        if cache_indices.shape[0] == 0:
            return torch.zeros_like(mixed_qkv)

        conv_state = conv_state_storage.transpose(-1, -2) if self._conv_state_dim_first else conv_state_storage
        kernel_state = conv_state
        kernel_indices = cache_indices
        null_block_id = NULL_BLOCK_ID
        copy_state_back = not conv_state.is_contiguous() or conv_state.dtype != mixed_qkv.dtype
        if copy_state_back:
            # Pack only this batch's rows. The copy kernel handles DS/page
            # strides and casts on load/store without copying the cache pool.
            kernel_state = torch.empty(
                (cache_indices.shape[0], *conv_state.shape[1:]),
                dtype=mixed_qkv.dtype,
                device=conv_state.device,
            )
            kernel_indices = torch.empty_like(cache_indices, dtype=torch.int32)
            cache_indices = torch.where(cache_indices == NULL_BLOCK_ID, PAD_SLOT_ID, cache_indices)
            copy_conv_state(conv_state, kernel_state, cache_indices, query_start_loc, kernel_indices, write_back=False)
            # Packed row zero is valid. FLA only enables null-block filtering
            # for nonnegative IDs, so reserve the ID just past the packed rows.
            null_block_id = kernel_state.shape[0]

        # Unlike prefill, FLA update has no pad_slot_id argument. Route padding
        # through its null-block filter for both packed and direct state views.
        kernel_indices = torch.where(kernel_indices == PAD_SLOT_ID, null_block_id, kernel_indices)

        if run_mode == 0:
            output = causal_conv1d_fn(
                mixed_qkv,
                conv_weights_t,
                None,
                conv_states=kernel_state,
                query_start_loc=query_start_loc,
                cache_indices=kernel_indices,
                has_initial_state=initial_state_mode,
                activation="silu",
                pad_slot_id=PAD_SLOT_ID,
                null_block_id=null_block_id,
            )
        else:
            output = causal_conv1d_update(
                mixed_qkv,
                kernel_state,
                conv_weights_t,
                bias=None,
                activation="silu",
                conv_state_indices=kernel_indices,
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=query_start_loc,
                max_query_len=max_query_len,
                null_block_id=null_block_id,
                out=torch.zeros_like(mixed_qkv),
            )

        if copy_state_back:
            copy_conv_state(conv_state, kernel_state, cache_indices, query_start_loc, kernel_indices, write_back=True)
        return output

    def _run_recurrent(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        state_indices: torch.Tensor,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return run_recurrent_kda(
            q,
            k,
            v,
            raw_gate,
            beta,
            recurrent_state,
            cu_seqlens,
            state_indices,
            self.A_log,
            self.dt_bias,
            lower_bound=self.lower_bound if self.safe_gate else None,
            num_accepted_tokens=num_accepted_tokens,
        )

    def _run_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        prebuilt_metadata,
    ) -> torch.Tensor:
        cu_seqlens = (
            prebuilt_metadata.cu_seqlens_host
            if prebuilt_metadata.cu_seqlens_kern is None
            else prebuilt_metadata.cu_seqlens_kern
        )
        keep = prebuilt_metadata.keep_meta
        if keep is not None:
            state_indices = state_indices[keep]
            has_initial_state = has_initial_state[keep]

        initial_state = recurrent_state[state_indices].contiguous()
        clear_ssm_states(initial_state, has_initial_state)
        output, final_state = run_chunk_kda(
            q,
            k,
            v,
            raw_gate,
            beta,
            initial_state,
            cu_seqlens,
            prebuilt_metadata.chunk_indices_chunk64_host,
            self.A_log,
            self.dt_bias,
            lower_bound=self.lower_bound if self.safe_gate else None,
        )
        recurrent_state[state_indices] = final_state.to(recurrent_state.dtype)
        return output

    @eager_break_during_capture
    def _forward(
        self,
        q_proj_states: torch.Tensor,
        k_proj_states: torch.Tensor,
        v_proj_states: torch.Tensor,
        g1: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        attn_metadata_map = get_forward_context().attn_metadata
        if attn_metadata_map is None:
            core_attn_out.zero_()
            return

        assert isinstance(attn_metadata_map, dict)
        attn_metadata = attn_metadata_map.get(self.prefix)
        if attn_metadata is None:
            core_attn_out.zero_()
            return
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        num_actual_tokens = attn_metadata.num_actual_tokens
        mixed_qkv = torch.cat(
            (
                q_proj_states[:num_actual_tokens],
                k_proj_states[:num_actual_tokens],
                v_proj_states[:num_actual_tokens],
            ),
            dim=-1,
        )
        # Keep the upstream custom-op keyword name (g1), while treating the
        # tensor as the raw gate required by the AscendC kernels.
        raw_gate = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]

        conv_state, recurrent_state = self.kv_cache
        conv_weights_t = self._conv_weights_t(mixed_qkv.dtype)
        spec_masks = attn_metadata.spec_sequence_masks
        spec_token_indices = attn_metadata.spec_token_indx
        non_spec_token_indices = attn_metadata.non_spec_token_indx

        if spec_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_spec = mixed_qkv
                gate_spec = raw_gate
                beta_spec = beta
                mixed_non_spec = gate_non_spec = beta_non_spec = None
            else:
                assert spec_token_indices is not None and non_spec_token_indices is not None
                mixed_spec = mixed_qkv.index_select(0, spec_token_indices)
                gate_spec = raw_gate.index_select(1, spec_token_indices)
                beta_spec = beta.index_select(1, spec_token_indices)
                mixed_non_spec = mixed_qkv.index_select(0, non_spec_token_indices)
                gate_non_spec = raw_gate.index_select(1, non_spec_token_indices)
                beta_non_spec = beta.index_select(1, non_spec_token_indices)
        else:
            mixed_spec = gate_spec = beta_spec = None
            mixed_non_spec = mixed_qkv
            gate_non_spec = raw_gate
            beta_non_spec = beta

        core_spec = None
        if mixed_spec is not None:
            spec_meta = attn_metadata.spec_decode_metadata
            assert spec_meta is not None
            conv_meta = spec_meta.spec_causal_conv1d
            mixed_spec = self._run_causal_conv1d(
                mixed_spec,
                conv_weights_t,
                conv_state,
                conv_meta.query_start_loc,
                conv_meta.cache_indices,
                None,
                run_mode=1,
                max_query_len=self.num_speculative_tokens + 1,
                num_accepted_tokens=conv_meta.num_accepted_tokens,
            )
            q_spec, k_spec, v_spec = (
                rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim) for x in mixed_spec.chunk(3, dim=-1)
            )
            assert gate_spec is not None and beta_spec is not None
            assert attn_metadata.spec_query_start_loc is not None
            assert attn_metadata.spec_state_indices_tensor is not None
            core_spec = self._run_recurrent(
                q_spec,
                k_spec,
                v_spec,
                gate_spec,
                beta_spec,
                recurrent_state,
                attn_metadata.spec_query_start_loc,
                attn_metadata.spec_state_indices_tensor,
                num_accepted_tokens=conv_meta.num_accepted_tokens,
            )
            core_spec = _zero_padded_spec_output(core_spec, attn_metadata.spec_query_start_loc)

        core_non_spec = None
        if mixed_non_spec is not None and mixed_non_spec.shape[0] > 0:
            if attn_metadata.num_prefills > 0:
                prefill_meta = attn_metadata.non_spec_prefill_metadata
                assert prefill_meta is not None
                conv_meta = prefill_meta.causal_conv1d
                mixed_non_spec = self._run_causal_conv1d(
                    mixed_non_spec,
                    conv_weights_t,
                    conv_state,
                    conv_meta.query_start_loc,
                    conv_meta.cache_indices,
                    conv_meta.initial_state_mode,
                    run_mode=0,
                )
            elif attn_metadata.num_decodes > 0:
                decode_meta = attn_metadata.non_spec_decode_metadata
                assert decode_meta is not None
                conv_meta = decode_meta.causal_conv1d
                mixed_non_spec = self._run_causal_conv1d(
                    mixed_non_spec,
                    conv_weights_t,
                    conv_state,
                    conv_meta.query_start_loc,
                    conv_meta.cache_indices,
                    None,
                    run_mode=1,
                    max_query_len=1,
                )

            q_non_spec, k_non_spec, v_non_spec = (
                rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim) for x in mixed_non_spec.chunk(3, dim=-1)
            )
            assert gate_non_spec is not None and beta_non_spec is not None

            split_non_spec = spec_masks is None and attn_metadata.num_prefills > 0 and attn_metadata.num_decodes > 0
            num_decode_tokens = attn_metadata.num_decode_tokens
            core_decode = None
            if split_non_spec:
                assert attn_metadata.non_spec_query_start_loc is not None
                assert attn_metadata.non_spec_state_indices_tensor is not None
                core_decode = self._run_recurrent(
                    q_non_spec[:, :num_decode_tokens],
                    k_non_spec[:, :num_decode_tokens],
                    v_non_spec[:, :num_decode_tokens],
                    gate_non_spec[:, :num_decode_tokens],
                    beta_non_spec[:, :num_decode_tokens],
                    recurrent_state,
                    attn_metadata.non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                    attn_metadata.non_spec_state_indices_tensor[: attn_metadata.num_decodes],
                )

            if attn_metadata.num_prefills > 0:
                if split_non_spec:
                    q_non_spec = q_non_spec[:, num_decode_tokens:]
                    k_non_spec = k_non_spec[:, num_decode_tokens:]
                    v_non_spec = v_non_spec[:, num_decode_tokens:]
                    gate_non_spec = gate_non_spec[:, num_decode_tokens:]
                    beta_non_spec = beta_non_spec[:, num_decode_tokens:]
                assert attn_metadata.prefill_state_indices is not None
                assert attn_metadata.prefill_has_initial_state is not None
                prefill_meta = attn_metadata.non_spec_prefill_metadata
                assert prefill_meta is not None
                core_prefill = self._run_prefill(
                    q_non_spec,
                    k_non_spec,
                    v_non_spec,
                    gate_non_spec,
                    beta_non_spec,
                    recurrent_state,
                    attn_metadata.prefill_state_indices,
                    attn_metadata.prefill_has_initial_state,
                    prefill_meta.chunk,
                )
                core_non_spec = (
                    torch.cat((core_decode, core_prefill), dim=1) if core_decode is not None else core_prefill
                )
            elif attn_metadata.num_decodes > 0:
                assert attn_metadata.non_spec_query_start_loc is not None
                assert attn_metadata.non_spec_state_indices_tensor is not None
                core_non_spec = self._run_recurrent(
                    q_non_spec,
                    k_non_spec,
                    v_non_spec,
                    gate_non_spec,
                    beta_non_spec,
                    recurrent_state,
                    attn_metadata.non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                    attn_metadata.non_spec_state_indices_tensor,
                )

        if core_spec is None and core_non_spec is None:
            core_attn_out.zero_()
            return

        core_attn_out[:, :num_actual_tokens].zero_()
        if core_spec is not None and core_non_spec is not None:
            assert spec_token_indices is not None and non_spec_token_indices is not None
            core_attn_out[:, :num_actual_tokens].index_copy_(1, spec_token_indices, core_spec)
            core_attn_out[:, :num_actual_tokens].index_copy_(1, non_spec_token_indices, core_non_spec)
        elif core_spec is not None:
            core_attn_out[:, :num_actual_tokens] = core_spec
        elif core_non_spec is not None:
            core_attn_out[:, :num_actual_tokens] = core_non_spec
        core_attn_out[:, num_actual_tokens:].zero_()
