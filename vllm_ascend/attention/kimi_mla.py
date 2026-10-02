# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
import torch.nn.functional as F
import torch_npu
from vllm.logger import logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_ascend.attention.mla_v1 import MLAPO_MAX_SUPPORTED_TOKENS, AscendMLAImpl, DecodeMLAPreprocessResult
from vllm_ascend.attention.utils import notify_kv_cache_written
from vllm_ascend.quantization.methods import AscendW8A8DynamicLinearMethod
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND, ACL_FORMAT_FRACTAL_NZ


def _interleave_mla_rope(x: torch.Tensor, nope_dim: int) -> torch.Tensor:
    """Undo CANN prolog's even/odd-to-half permutation in weights/scales."""
    nope, rope = x.split([nope_dim, x.shape[-1] - nope_dim], dim=-1)
    rope = rope.unflatten(-1, (2, -1)).transpose(-1, -2).flatten(-2)
    return torch.cat((nope, rope), dim=-1)


class KimiMLAProlog:
    """Add K3 no-RoPE prolog preprocessing to the selected MLA implementation."""

    def __init__(self, impl: AscendMLAImpl):
        self.impl = impl
        self.enabled = False
        self._original_process_weights = impl.process_weights_after_loading
        self._original_preprocess = impl._mla_preprocess
        # The adapter owns its fast path. Keep the generic prolog disabled so
        # its weight packing/freeing and dispatch remain unchanged for K3.
        impl.enable_mlapo = False
        impl.process_weights_after_loading = self.process_weights_after_loading
        impl._mla_preprocess = self.preprocess

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        self.enabled = False
        self._original_process_weights(act_dtype)
        impl = self.impl
        fused = impl.fused_qkv_a_proj
        if fused is None:
            return
        methods = (fused.quant_method, impl.q_proj.quant_method)
        self.uses_native_weights = all(isinstance(method, UnquantizedLinearMethod) for method in methods)
        schemes = tuple(getattr(method, "quant_method", None) for method in methods)
        uses_dynamic_int8 = all(
            isinstance(scheme, AscendW8A8DynamicLinearMethod) and scheme.act_quant_type == torch.int8
            for scheme in schemes
        )
        if act_dtype != torch.bfloat16 or not (self.uses_native_weights or uses_dynamic_int8):
            logger.warning_once(
                "Kimi K3 MLAPO requires BF16 or W8A8_DYNAMIC MLA projections; falling back to unfused MLA."
            )
            return
        if any(getattr(norm, "bias_loaded", False) for norm in (impl.q_a_layernorm, impl.kv_a_layernorm)):
            # MlaPrologV3 accepts gamma but cannot apply ModelSlim's norm bias.
            logger.warning_once("Kimi K3 MLAPO cannot fuse a loaded MLA RMSNorm bias; falling back to unfused MLA.")
            return
        self._prepare_weights(act_dtype)
        self.enabled = True
        logger.info_once("Using CANN MlaPrologV3 for Kimi K3 MLA decode.")

    def _prepare_weights(self, act_dtype: torch.dtype):
        impl = self.impl
        # Preserve the original projections for prefill, mixed batches and
        # fallback; only the prolog's copies change the raw channel ordering.
        fused_weight = torch_npu.npu_format_cast(impl.fused_qkv_a_proj.weight.data, ACL_FORMAT_FRACTAL_ND)
        uq_weight = torch_npu.npu_format_cast(impl.q_proj.weight.data, ACL_FORMAT_FRACTAL_ND)
        if self.uses_native_weights:
            fused_weight = fused_weight.T
            uq_weight = uq_weight.T

        self.weight_dq = torch_npu.npu_format_cast(
            fused_weight[..., : impl.q_lora_rank].contiguous(), ACL_FORMAT_FRACTAL_NZ
        )
        # Identity sin/cos still makes CANN deinterleave QR/KR. Pre-interleave
        # these copies to preserve K3's no-RoPE query/cache basis.
        self.weight_dkv_kr = torch_npu.npu_format_cast(
            _interleave_mla_rope(fused_weight[..., impl.q_lora_rank :], impl.kv_lora_rank), ACL_FORMAT_FRACTAL_NZ
        )
        uq_weight = _interleave_mla_rope(
            uq_weight.reshape(impl.q_lora_rank, impl.num_heads, impl.qk_head_dim), impl.qk_nope_head_dim
        )
        # Materialize the flattened matrix before NZ conversion. An NPU view
        # can retain the padded 3-D storage layout and produce incorrect Q-up
        # weights even when npu_format_cast reports format 29.
        self.weight_uq_qr = torch_npu.npu_format_cast(
            F.pad(uq_weight, (0, 0, 0, impl.head_padding)).flatten(1).clone(), ACL_FORMAT_FRACTAL_NZ
        )
        # Unfused post-load processing may have converted W_UK_T to NZ.
        weight_uk = torch_npu.npu_format_cast(impl.W_UK_T, ACL_FORMAT_FRACTAL_ND)
        self.weight_uk = F.pad(weight_uk, (0, 0, 0, 0, 0, impl.head_padding))
        self.padded_num_heads = impl.num_heads_padded
        self.weight_quant_mode = 0 if self.uses_native_weights else 2

        if not self.uses_native_weights:
            scale = impl.fused_qkv_a_proj.weight_scale.data.float().reshape(1, -1)
            self.dequant_scale_w_dq = scale[:, : impl.q_lora_rank].contiguous()
            self.dequant_scale_w_dkv_kr = _interleave_mla_rope(scale[:, impl.q_lora_rank :], impl.kv_lora_rank)
            uq_scale = impl.q_proj.weight_scale.data.float().reshape(1, impl.num_heads, impl.qk_head_dim)
            uq_scale = _interleave_mla_rope(uq_scale, impl.qk_nope_head_dim)
            self.dequant_scale_w_uq_qr = F.pad(uq_scale, (0, 0, 0, impl.head_padding), value=1).flatten(1)

        rope_shape = (MLAPO_MAX_SUPPORTED_TOKENS, impl.qk_rope_head_dim)
        self.cos = torch.ones(rope_shape, dtype=act_dtype, device=fused_weight.device)
        self.sin = torch.zeros_like(self.cos)

    def preprocess(self, layer_name, hidden_states, kv_cache, attn_metadata):
        if (
            self.enabled
            and not self.impl._decode_requires_current_kv(attn_metadata)
            and attn_metadata.num_decode_tokens <= MLAPO_MAX_SUPPORTED_TOKENS
            and attn_metadata.num_prefills == 0
        ):
            if self.impl.layerwise_kv_cache_hook is not None:
                self.impl.layerwise_kv_cache_hook.wait_for_layer(layer_name)
            result = self.preprocess_decode(hidden_states, kv_cache, attn_metadata)
            notify_kv_cache_written(layer_name)
            return result
        return self._original_preprocess(layer_name, hidden_states, kv_cache, attn_metadata)

    def preprocess_decode(self, hidden_states, kv_cache, attn_metadata):
        impl = self.impl
        num_tokens = attn_metadata.num_decode_tokens
        hidden_states = hidden_states[:num_tokens]
        if self.uses_native_weights:
            token_x = hidden_states
            dequant_scale_x = None
            dequant_scale_w_dq = dequant_scale_w_uq_qr = dequant_scale_w_dkv_kr = None
        else:
            token_x, dynamic_scale = torch_npu.npu_dynamic_quant(hidden_states)
            dequant_scale_x = dynamic_scale.view(-1, 1)
            dequant_scale_w_dq = self.dequant_scale_w_dq
            dequant_scale_w_uq_qr = self.dequant_scale_w_uq_qr
            dequant_scale_w_dkv_kr = self.dequant_scale_w_dkv_kr

        q_nope, q_pe, dequant_scale_q_nope, _, _ = torch_npu.npu_mla_prolog_v3(
            kv_cache=kv_cache[0],
            kr_cache=kv_cache[1],
            token_x=token_x,
            weight_dq=self.weight_dq,
            weight_uq_qr=self.weight_uq_qr,
            weight_uk=self.weight_uk,
            weight_dkv_kr=self.weight_dkv_kr,
            rmsnorm_gamma_cq=impl.q_a_layernorm.weight.data,
            rmsnorm_gamma_ckv=impl.kv_a_layernorm.weight.data,
            rmsnorm_epsilon_cq=impl.q_a_layernorm.variance_epsilon,
            rmsnorm_epsilon_ckv=impl.kv_a_layernorm.variance_epsilon,
            rope_sin=self.sin[:num_tokens],
            rope_cos=self.cos[:num_tokens],
            cache_index=attn_metadata.slot_mapping[:num_tokens].to(torch.int64),
            dequant_scale_x=dequant_scale_x,
            dequant_scale_w_dq=dequant_scale_w_dq,
            dequant_scale_w_uq_qr=dequant_scale_w_uq_qr,
            dequant_scale_w_dkv_kr=dequant_scale_w_dkv_kr,
            cache_mode="PA_NZ" if impl.enable_kv_nz else "PA_BSND",
            query_quant_mode=0,
            weight_quant_mode=self.weight_quant_mode,
            kv_cache_quant_mode=0,
            quant_scale_ckv=None,
        )
        # Crop before backend-specific processing, especially DCP head gather.
        q_nope = q_nope.view(num_tokens, self.padded_num_heads, impl.kv_lora_rank)[:, : impl.num_heads]
        q_pe = q_pe.view(num_tokens, self.padded_num_heads, -1)[:, : impl.num_heads]
        q_nope, q_pe = impl.reorg_decode_q(q_nope, q_pe)
        return DecodeMLAPreprocessResult(
            q_nope, q_pe, kv_cache[0], kv_cache[1], dequant_scale_q_nope=dequant_scale_q_nope
        ), None
