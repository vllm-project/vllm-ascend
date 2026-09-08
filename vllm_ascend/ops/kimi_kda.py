# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend backend for the vLLM 0.27 Kimi K3 delta-attention layer.

The projections, weight loading, and cache specification stay owned by
upstream vLLM.  Only the CUDA-specific convolution and KDA execution is
replaced here with the Ascend metadata builder and AscendC operators.
"""

from functools import wraps

import torch
import torch_npu
from einops import rearrange
from torch import nn
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
)
from vllm.model_executor.utils import replace_parameter
from vllm.models.kimi_k3.nvidia.kda import (
    KimiK3DeltaAttention,
)
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionBackend
from vllm_ascend.ops.triton.fla.utils import clear_ssm_states
from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import (
    AscendW4A8MXFPDynamicLinearMethod,
)
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import (
    AscendW8A8MXFP8DynamicLinearMethod,
)
from vllm_ascend.utils import npu_stream_switch

_KDA_CHUNK_SIZE = 64
_PACKED_CONV_WEIGHT_NAME = "ascend_conv1d_weight"
_F_PROJ_SHARD_ID = 1
_KDA_BFG_STREAM: torch.npu.Stream | None = None


def _kda_bfg_stream() -> torch.npu.Stream:
    global _KDA_BFG_STREAM
    if _KDA_BFG_STREAM is None:
        _KDA_BFG_STREAM = torch_npu.npu.Stream()
    return _KDA_BFG_STREAM


class _KDAFusedBFGLinear(MergedColumnParallelLinear):
    """Pack beta, an offline-composed F projection, and the output gate."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        tp_size: int,
        quant_config,
        prefix: str,
    ) -> None:
        projection_size = num_heads * head_dim
        super().__init__(
            input_size=hidden_size,
            output_sizes=[num_heads, projection_size, projection_size],
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )
        if self.tp_size != tp_size:
            raise ValueError(f"KDA fused BFG TP mismatch: layer={self.tp_size}, attention={tp_size}")
        local_projection_size = projection_size // tp_size
        self.f_a_weight = nn.Parameter(
            self.weight.new_empty((head_dim, hidden_size)),
            requires_grad=False,
        )
        self.f_b_weight = nn.Parameter(
            self.weight.new_empty((local_projection_size, head_dim)),
            requires_grad=False,
        )
        self.f_a_weight.weight_loader = self._load_f_a_weight
        self.f_b_weight.weight_loader = self._load_f_b_weight
        self._f_a_loaded = False
        self._f_b_loaded = False

    def _load_f_a_weight(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        del loaded_shard_id
        if param.shape != loaded_weight.shape:
            raise ValueError(
                "KDA f_a_proj checkpoint shape mismatch: "
                f"expected {tuple(param.shape)}, got {tuple(loaded_weight.shape)}"
            )
        param.data.copy_(loaded_weight)
        self._f_a_loaded = True
        self._maybe_fuse_f_proj()

    def _load_f_b_weight(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        del loaded_shard_id
        if loaded_weight.shape == param.shape:
            local_weight = loaded_weight
        else:
            expected_shape = (param.shape[0] * self.tp_size, param.shape[1])
            if loaded_weight.shape != expected_shape:
                raise ValueError(
                    "KDA f_b_proj checkpoint shape mismatch: "
                    f"expected {expected_shape} or {tuple(param.shape)}, "
                    f"got {tuple(loaded_weight.shape)}"
                )
            local_weight = loaded_weight.narrow(
                0,
                self.tp_rank * param.shape[0],
                param.shape[0],
            )
        param.data.copy_(local_weight)
        self._f_b_loaded = True
        self._maybe_fuse_f_proj()

    @torch.no_grad()
    def _maybe_fuse_f_proj(self) -> None:
        if not self._f_a_loaded or not self._f_b_loaded:
            return
        output_dim = getattr(self.weight, "output_dim", None)
        if output_dim is None:
            raise ValueError("KDA fused f_proj requires an output-sharded parameter")
        shard_offset = sum(self.output_sizes[:_F_PROJ_SHARD_ID]) // self.tp_size
        shard_size = self.output_sizes[_F_PROJ_SHARD_ID] // self.tp_size
        param_shard = self.weight.narrow(output_dim, shard_offset, shard_size)
        fused_weight = torch.matmul(
            self.f_b_weight.float(),
            self.f_a_weight.float(),
        ).to(dtype=param_shard.dtype)
        if fused_weight.shape != param_shard.shape:
            raise ValueError(
                "KDA composed f_proj shape mismatch: "
                f"expected {tuple(param_shard.shape)}, got {tuple(fused_weight.shape)}"
            )
        param_shard.copy_(fused_weight)


def _zero_padded_output(
    output: torch.Tensor,
    num_live_tokens: torch.Tensor,
) -> torch.Tensor:
    """Clear graph-padding rows using a device-side live-token count."""
    token_indices = torch.arange(
        output.shape[1],
        dtype=num_live_tokens.dtype,
        device=output.device,
    )
    valid_tokens = token_indices < num_live_tokens
    return torch.where(valid_tokens.view(1, -1, 1, 1), output, 0.0)


def _zero_padded_recurrent_output(
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    """Clear graph-padding rows skipped by recurrent KDA."""
    return _zero_padded_output(output, query_start_loc[-1])


def _prepare_beta(
    beta: torch.Tensor,
    num_actual_tokens: int,
    *,
    is_preprocessed: bool = False,
) -> torch.Tensor:
    """Slice beta and apply sigmoid unless the auxiliary stream already did."""
    beta = beta[:, :num_actual_tokens]
    return beta if is_preprocessed else beta.float().sigmoid()


class AscendKimiK3DeltaAttention(KimiK3DeltaAttention):
    """Kimi K3 KDA using AscendC prefill and recurrent kernels."""

    def __init__(self, config, vllm_config, prefix: str = "") -> None:
        quant_config = getattr(vllm_config, "quant_config", None)
        uses_mixed_projection = bool(
            quant_config is not None
            and getattr(
                quant_config,
                "uses_kimi_k3_mixed_kda_projection",
                lambda _prefix: False,
            )(f"{prefix}.in_proj_qkvgfab")
        )
        super().__init__(config, vllm_config, prefix)
        self.uses_mixed_projection = uses_mixed_projection
        if uses_mixed_projection:
            # vLLM 0.27 packs all KDA input projections into one linear.  A
            # QuaRot checkpoint instead stores q/k/v as W8A8 and keeps B/F/G
            # in floating point. Split those precision groups so DynamicQuant
            # can overlap the composed BFG projection.
            self.in_proj_qkvgfab = MergedColumnParallelLinear(
                self.hidden_size,
                [self.projection_size] * 3,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_qkv",
            )
            del self.f_b_proj
            self.fused_bfg_proj = _KDAFusedBFGLinear(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                tp_size=self.tp_size,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_gfab",
            )
            self._fused_bfg_output_sizes = (
                self.local_num_heads,
                self.local_projection_size,
                self.local_projection_size,
            )
        # Upstream's FusedRMSNormGated constructor defaults to 1e-5, while
        # Kimi K3 checkpoints use the model-configured RMS epsilon (1e-6 for
        # the production checkpoint). Preserve the checkpoint contract used
        # by the validated v0.26 implementation.
        self.o_norm.eps = config.rms_norm_eps
        # vLLM keeps the checkpoint-compatible FP32 [3C, 1, W] weight, while
        # npu_causal_conv1d_custom consumes an activation-dtype [W, 3C]
        # tensor. Materialize that kernel layout once after weight loading.
        self.register_parameter(
            _PACKED_CONV_WEIGHT_NAME,
            nn.Parameter(
                torch.empty(
                    self.conv_size,
                    3 * self.local_projection_size,
                    dtype=self.model_config.dtype,
                    device=self.conv1d.weight.device,
                ),
                requires_grad=False,
            ),
        )
        original_process_weights = self.conv1d.quant_method.process_weights_after_loading

        @wraps(original_process_weights)
        def process_weights_and_pack(*args, **kwargs):
            result = original_process_weights(*args, **kwargs)
            self._pack_conv_weights()
            return result

        self.conv1d.quant_method.process_weights_after_loading = process_weights_and_pack

    def get_attn_backend(self) -> type[AttentionBackend]:
        return AscendGDNAttentionBackend

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if self.uses_mixed_projection:
            num_tokens = hidden_states.size(0)
            mixed_qkv, beta, g1, g2 = self._run_overlapped_qkv_bfg(hidden_states)
            core_attn_out = torch.empty(
                (1, num_tokens, self.local_num_heads, self.head_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            self._forward(
                mixed_qkv=mixed_qkv,
                g1=g1,
                g2=g2,
                beta=beta,
                core_attn_out=core_attn_out,
                beta_is_preprocessed=True,
            )
            core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
            return self.o_proj(core_attn_out)[0]
        return super().forward(hidden_states, positions)

    def _run_overlapped_qkv_bfg(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fork BFG work to an auxiliary stream and join it back to main."""
        main_stream = torch.npu.current_stream()
        bfg_stream = _kda_bfg_stream()

        hidden_states_ready = main_stream.record_event()
        hidden_states.record_stream(bfg_stream)
        with npu_stream_switch(bfg_stream):
            bfg_stream.wait_event(hidden_states_ready)
            fused_bfg = self._project_bfg(hidden_states)
            bfg_projection_ready = bfg_stream.record_event()

        quantized_qkv = self._quantize_fused_qkv(hidden_states)
        quant_ready = main_stream.record_event()

        # Stage 1 join: DynamicQuant on main overlaps the BFG GEMM on the
        # auxiliary stream, but the two Cube matmuls remain serialized.
        main_stream.wait_event(bfg_projection_ready)
        mixed_qkv = self._matmul_fused_qkv(quantized_qkv)

        with npu_stream_switch(bfg_stream):
            # Stage 2: after both first-stage branches complete, overlap the
            # QKV Cube matmul with beta's FP32 conversion and sigmoid vector
            # work. Split and reshape the F/output gates here as well so all
            # BFG output handling occurs after the QKV matmul is enqueued.
            bfg_stream.wait_event(quant_ready)
            beta, g1, g2 = self._postprocess_bfg(fused_bfg)
            bfg_ready = bfg_stream.record_event()

        for tensor in (beta, g1, g2):
            tensor.record_stream(main_stream)
        # bfg_ready is the auxiliary stream tail. Joining that exact event is
        # required for multi-stream ACL graph capture as well as eager reuse.
        main_stream.wait_event(bfg_ready)
        return mixed_qkv, beta, g1, g2

    def _project_bfg(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.fused_bfg_proj(hidden_states)[0]

    def _postprocess_bfg(
        self,
        fused_bfg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        beta, raw_gate, output_gate = fused_bfg.split(
            self._fused_bfg_output_sizes,
            dim=-1,
        )
        beta = beta.float().sigmoid().unsqueeze(0)
        raw_gate = rearrange(raw_gate, "n (h d) -> 1 n h d", d=self.head_dim)
        output_gate = rearrange(output_gate, "n (h d) -> n h d", d=self.head_dim)
        return beta, raw_gate, output_gate

    def _quantize_fused_qkv(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        quant_method = self.in_proj_qkvgfab.quant_method
        inner_quant_method = getattr(quant_method, "quant_method", quant_method)
        if (
            isinstance(
                inner_quant_method,
                (
                    AscendW4A8MXFPDynamicLinearMethod,
                    AscendW8A8MXFP8DynamicLinearMethod,
                ),
            )
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.ndim == 2
        ):
            if isinstance(inner_quant_method, AscendW8A8MXFP8DynamicLinearMethod):
                return torch_npu.npu_dynamic_mx_quant(
                    hidden_states,
                    dst_type=torch.float8_e4m3fn,
                    scale_alg=inner_quant_method.dynamic_mx_quant_scale_alg,
                )
            return torch_npu.npu_dynamic_mx_quant(
                hidden_states,
                dst_type=torch.float8_e4m3fn,
            )
        return hidden_states

    def _matmul_fused_qkv(
        self,
        qkv_input: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        quant_method = self.in_proj_qkvgfab.quant_method
        if quant_method is None:
            raise RuntimeError("KDA fused QKV quantization method is not initialized")
        return quant_method.apply(
            self.in_proj_qkvgfab,
            qkv_input,
            bias=None,
        )

    @staticmethod
    def _run_causal_conv1d(
        mixed_qkv: torch.Tensor,
        conv_weights_t: torch.Tensor,
        conv_state: torch.Tensor,
        query_start_loc: torch.Tensor,
        cache_indices: torch.Tensor,
        initial_state_mode: torch.Tensor | None,
        *,
        run_mode: int,
        num_accepted_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = torch.empty_like(mixed_qkv)
        # Consume the operator's declared output alias. Returning ``output``
        # independently would let graph functionalization treat the custom-op
        # result as dead and expose the uninitialized allocation instead.
        return torch.ops._C_ascend.npu_causal_conv1d_custom(
            output,
            mixed_qkv,
            conv_weights_t,
            conv_state=conv_state,
            bias_opt=None,
            query_start_loc_opt=query_start_loc,
            cache_indices_opt=cache_indices,
            initial_state_mode_opt=initial_state_mode,
            num_accepted_tokens_opt=num_accepted_tokens,
            activation_mode=1,
            pad_slot_id=PAD_SLOT_ID,
            run_mode=run_mode,
        )

    @torch.no_grad()
    def _pack_conv_weights(self) -> None:
        if self.conv1d.weight.is_meta:
            return
        packed_param = self.get_parameter(_PACKED_CONV_WEIGHT_NAME)
        packed_weight = (
            self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
            .transpose(0, 1)
            .to(device=packed_param.device, dtype=packed_param.dtype)
            .contiguous()
        )
        replace_parameter(
            self,
            _PACKED_CONV_WEIGHT_NAME,
            packed_weight,
            prefer_copy=True,
        )

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
        return torch.ops._C_ascend.recurrent_kda(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            raw_gate.contiguous(),
            beta.contiguous(),
            recurrent_state,
            cu_seqlens,
            state_indices,
            self.A_log.reshape(-1).contiguous(),
            self.dt_bias.contiguous(),
            num_accepted_tokens=num_accepted_tokens,
            scale=self.head_dim**-0.5,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=False,
            allow_neg_eigval=False,
            safe_gate=self.gate_lower_bound is not None,
            lower_bound=(self.gate_lower_bound if self.gate_lower_bound is not None else -5.0),
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

        # The recurrent cache uses [H,V,K]. The fused prefill operator accepts
        # that state layout directly through state_v_first.
        initial_state_vk = recurrent_state[state_indices].contiguous()
        clear_ssm_states(initial_state_vk, has_initial_state)

        q = l2norm_fwd(q.contiguous())
        k = l2norm_fwd(k.contiguous())
        result = torch.ops._C_ascend.chunk_kda_fwd(
            q,
            k,
            v.contiguous(),
            raw_gate.contiguous(),
            beta.contiguous(),
            self.head_dim**-0.5,
            _KDA_CHUNK_SIZE,
            layout="BSND",
            initial_state=initial_state_vk,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            chunk_indices=prebuilt_metadata.chunk_indices_chunk64_host,
            safe_gate=self.gate_lower_bound is not None,
            lower_bound=self.gate_lower_bound if self.gate_lower_bound is not None else -5.0,
            use_gate_in_kernel=True,
            A_log=self.A_log.reshape(-1).contiguous(),
            dt_bias=self.dt_bias.contiguous(),
            disable_recompute=False,
            return_intermediate_states=False,
            state_v_first=True,
        )
        recurrent_state[state_indices] = result[1].to(recurrent_state.dtype)
        return result[0]

    @eager_break_during_capture
    def _forward(
        self,
        mixed_qkv: torch.Tensor,
        g1: torch.Tensor,
        g2: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
        *,
        beta_is_preprocessed: bool = False,
    ) -> None:
        """Dispatch speculative, prefill, and decode tokens through KDA kernels."""
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        if attn_metadata_raw is None:
            core_attn_out.zero_()
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        num_actual_tokens = attn_metadata.num_actual_tokens
        mixed_qkv = mixed_qkv[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        g2 = g2[:num_actual_tokens]
        beta = _prepare_beta(
            beta,
            num_actual_tokens,
            is_preprocessed=beta_is_preprocessed,
        )

        conv_state, recurrent_state = self.kv_cache
        conv_weights_t = self.get_parameter(_PACKED_CONV_WEIGHT_NAME)
        spec_masks = attn_metadata.spec_sequence_masks
        spec_token_indices = attn_metadata.spec_token_indx
        non_spec_token_indices = attn_metadata.non_spec_token_indx

        if spec_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_spec = mixed_qkv
                raw_gate_spec = g1
                beta_spec = beta
                mixed_non_spec = raw_gate_non_spec = beta_non_spec = None
            else:
                assert spec_token_indices is not None
                assert non_spec_token_indices is not None
                mixed_spec = mixed_qkv.index_select(0, spec_token_indices)
                raw_gate_spec = g1.index_select(1, spec_token_indices)
                beta_spec = beta.index_select(1, spec_token_indices)
                mixed_non_spec = mixed_qkv.index_select(0, non_spec_token_indices)
                raw_gate_non_spec = g1.index_select(1, non_spec_token_indices)
                beta_non_spec = beta.index_select(1, non_spec_token_indices)
        else:
            mixed_spec = raw_gate_spec = beta_spec = None
            mixed_non_spec = mixed_qkv
            raw_gate_non_spec = g1
            beta_non_spec = beta

        core_spec = None
        if mixed_spec is not None:
            spec_meta = attn_metadata.spec_decode_metadata
            assert spec_meta is not None
            spec_conv_meta = spec_meta.spec_causal_conv1d
            mixed_spec = self._run_causal_conv1d(
                mixed_spec,
                conv_weights_t,
                conv_state,
                spec_conv_meta.query_start_loc,
                spec_conv_meta.cache_indices,
                None,
                run_mode=1,
                num_accepted_tokens=spec_conv_meta.num_accepted_tokens,
            )
            q_spec, k_spec, v_spec = (
                rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim) for x in mixed_spec.chunk(3, dim=-1)
            )
            assert raw_gate_spec is not None and beta_spec is not None
            assert attn_metadata.spec_query_start_loc is not None
            assert attn_metadata.spec_state_indices_tensor is not None
            core_spec = self._run_recurrent(
                q_spec,
                k_spec,
                v_spec,
                raw_gate_spec,
                beta_spec,
                recurrent_state,
                attn_metadata.spec_query_start_loc,
                attn_metadata.spec_state_indices_tensor,
                num_accepted_tokens=spec_conv_meta.num_accepted_tokens,
            )
            core_spec = _zero_padded_recurrent_output(
                core_spec,
                attn_metadata.spec_query_start_loc,
            )

        core_non_spec = None
        if mixed_non_spec is not None and mixed_non_spec.shape[0] > 0:
            if attn_metadata.num_prefills > 0:
                prefill_meta = attn_metadata.non_spec_prefill_metadata
                assert prefill_meta is not None
                mixed_non_spec = self._run_causal_conv1d(
                    mixed_non_spec,
                    conv_weights_t,
                    conv_state,
                    prefill_meta.causal_conv1d.query_start_loc,
                    prefill_meta.causal_conv1d.cache_indices,
                    prefill_meta.causal_conv1d.initial_state_mode,
                    run_mode=0,
                )
            elif attn_metadata.num_decodes > 0:
                decode_meta = attn_metadata.non_spec_decode_metadata
                assert decode_meta is not None
                mixed_non_spec = self._run_causal_conv1d(
                    mixed_non_spec,
                    conv_weights_t,
                    conv_state,
                    decode_meta.causal_conv1d.query_start_loc,
                    decode_meta.causal_conv1d.cache_indices,
                    None,
                    run_mode=1,
                )

            q_non_spec, k_non_spec, v_non_spec = (
                rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim) for x in mixed_non_spec.chunk(3, dim=-1)
            )
            assert raw_gate_non_spec is not None
            assert beta_non_spec is not None

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
                    raw_gate_non_spec[:, :num_decode_tokens],
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
                    raw_gate_non_spec = raw_gate_non_spec[:, num_decode_tokens:]
                    beta_non_spec = beta_non_spec[:, num_decode_tokens:]

                assert attn_metadata.prefill_state_indices is not None
                assert attn_metadata.prefill_has_initial_state is not None
                prefill_meta = attn_metadata.non_spec_prefill_metadata
                assert prefill_meta is not None
                core_prefill = self._run_prefill(
                    q_non_spec,
                    k_non_spec,
                    v_non_spec,
                    raw_gate_non_spec,
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
                    raw_gate_non_spec,
                    beta_non_spec,
                    recurrent_state,
                    attn_metadata.non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                    attn_metadata.non_spec_state_indices_tensor,
                )

        if core_non_spec is not None:
            assert attn_metadata.non_spec_query_start_loc is not None
            core_non_spec = _zero_padded_recurrent_output(
                core_non_spec,
                attn_metadata.non_spec_query_start_loc,
            )

        if core_spec is None and core_non_spec is None:
            # Idle DP dummy runs carry graph-shaped metadata with no live work.
            # Do not feed a previous replay's output through the norm gate.
            core_attn_out.zero_()
            return

        num_live_tokens = None
        if core_spec is not None:
            assert attn_metadata.spec_query_start_loc is not None
            num_live_tokens = attn_metadata.spec_query_start_loc[-1]
        if core_non_spec is not None:
            assert attn_metadata.non_spec_query_start_loc is not None
            num_non_spec_tokens = attn_metadata.non_spec_query_start_loc[-1]
            num_live_tokens = num_non_spec_tokens if num_live_tokens is None else num_live_tokens + num_non_spec_tokens
        assert num_live_tokens is not None

        # Reuse the caller-owned result buffer. FULL graphs can leave rows
        # outside the live spec/non-spec index sets, so define them before the
        # two index copies rather than allocating a temporary merged tensor.
        core_attn_out[:, :num_actual_tokens].zero_()
        if core_spec is not None and core_non_spec is not None:
            assert spec_token_indices is not None
            assert non_spec_token_indices is not None
            assert spec_token_indices.numel() + non_spec_token_indices.numel() <= num_actual_tokens
            core_attn_out[:, :num_actual_tokens].index_copy_(1, spec_token_indices, core_spec)
            core_attn_out[:, :num_actual_tokens].index_copy_(1, non_spec_token_indices, core_non_spec)
        elif core_spec is not None:
            core_attn_out[:, :num_actual_tokens] = core_spec
        elif core_non_spec is not None:
            core_attn_out[:, :num_actual_tokens] = core_non_spec

        # The registered Ascend FusedRMSNormGated uses the fused norm-gate
        # kernel while preserving the upstream parameter/loading contract.
        normalized = self.o_norm(core_attn_out[:, :num_actual_tokens], g2)
        # Mask again after the norm gate: zero * sigmoid(NaN) is still NaN in
        # static padding rows whose captured gate values are not live.
        core_attn_out[:, :num_actual_tokens].copy_(_zero_padded_output(normalized, num_live_tokens))
        core_attn_out[:, num_actual_tokens:].zero_()
