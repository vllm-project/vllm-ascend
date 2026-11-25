# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch_npu

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND, ACL_FORMAT_FRACTAL_NZ

from .attention import DecodeWeights
from .mtp import Q6_MAX_CONTEXT
from .single import MAX_PACKED_OUTPUT_REQUESTS, OUTPUT_PANEL_CHANNELS


def _nd(tensor):
    tensor = tensor.detach()
    if torch_npu.get_npu_format(tensor) != ACL_FORMAT_FRACTAL_ND:
        tensor = torch_npu.npu_format_cast(tensor, ACL_FORMAT_FRACTAL_ND)
    return tensor


def _output_major(tensor, out_features, in_features):
    tensor = _nd(tensor)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.shape == (in_features, out_features):
        tensor = tensor.T
    if tensor.shape != (out_features, in_features):
        raise ValueError(f"Expected a projection with {in_features} input and {out_features} output channels")
    return tensor


def _mx_weight(layer, out_features, in_features, *, pack_output=False, allow_native_output=False):
    weight = layer.weight.detach()
    native_output = (
        allow_native_output
        and weight.shape == (in_features, out_features)
        and weight.is_contiguous()
        and weight.storage_offset() == 0
        and weight.untyped_storage().nbytes() == weight.numel()
        and torch_npu.get_npu_format(weight) == ACL_FORMAT_FRACTAL_NZ
    )
    codes = weight if native_output else _output_major(weight, out_features, in_features)
    scales = _nd(layer.weight_scale)
    if scales.ndim == 4 and scales.shape[0] == 1:
        scales = scales[0]
    if (
        codes.dtype != torch.float8_e4m3fn
        or scales.element_size() != 1
        or scales.shape != (in_features // 64, out_features, 2)
    ):
        raise ValueError("Expected MXFP8 weights with two E8M0 scales per 64 input channels")
    scales = scales.contiguous().view(torch.int16).view(in_features // 64, out_features)
    if native_output:
        return codes, scales
    # Pack directly into the worker's only prepared weight and preserve a native ND alias.
    if pack_output and (
        not codes.is_contiguous() or codes.untyped_storage().data_ptr() != layer.weight.untyped_storage().data_ptr()
    ):
        groups = in_features // OUTPUT_PANEL_CHANNELS
        codes = codes.view(torch.uint8).view(out_features, groups, OUTPUT_PANEL_CHANNELS)
        codes = codes.transpose(0, 1).contiguous().view(torch.float8_e4m3fn)
        scales = scales.view(groups, OUTPUT_PANEL_CHANNELS // 64, out_features).transpose(1, 2).contiguous()
        return codes, scales
    return codes.contiguous(), scales


def prepare_decode_weights(impl) -> DecodeWeights:
    """Pack a loaded AscendSFAImpl once before capturing the TP1 NoPE decode graph."""
    if (
        impl.local_num_heads != 64
        or impl.qk_rope_head_dim != 0
        or not impl.runtime_has_indexer
        or impl.g_proj is not None
    ):
        raise ValueError("Expected a replicated 64-head GLM-5.3-Flash NoPE layer with a runtime KeyPool indexer")
    indexer = impl.indexer.impl
    qkv_a, qkv_a_scales = _mx_weight(impl.fused_qkv_a_proj, 2048, 4096)
    q_b, q_b_scales = _mx_weight(impl.q_proj, 16384, 1536)
    config = impl.vllm_config
    # At 1M the extra projection copies can evict reusable KV prefixes.
    native_layout = config.speculative_config is not None and config.model_config.max_model_len == Q6_MAX_CONTEXT
    pack_output = (
        impl.vllm_config.speculative_config is None
        and impl.vllm_config.scheduler_config.max_num_seqs <= MAX_PACKED_OUTPUT_REQUESTS
    )
    out, out_scales = _mx_weight(
        impl.o_proj,
        4096,
        16384,
        pack_output=pack_output,
        allow_native_output=native_layout,
    )
    key_heads = _nd(indexer.wk_weights_proj.weight)
    if key_heads.shape != (160, 4096) or indexer.head_dim != 128:
        raise ValueError("Expected a 128-channel index key and 32 index head weights")
    w_uk = _nd(impl.W_UK_T)
    w_uv = _nd(impl.W_UV)
    if not native_layout:
        w_uk = w_uk.transpose(1, 2)
        w_uv = w_uv.transpose(1, 2)
    return DecodeWeights(
        qkv_a,
        qkv_a_scales,
        q_b,
        q_b_scales,
        w_uk.contiguous(),
        w_uv.contiguous(),
        out,
        out_scales,
        _nd(impl.q_a_layernorm.weight).contiguous(),
        _nd(impl.kv_a_layernorm.weight).contiguous(),
        torch.cat((key_heads[:128], _nd(indexer.index_kpool_compress_gate).reshape(128, 4096))),
        _nd(indexer.k_norm.weight).float().contiguous(),
        _nd(indexer.k_norm.bias).float().contiguous(),
        _nd(indexer.index_kpool_compress_ape).float().contiguous(),
        _output_major(indexer.wq_b.weight, 4096, 1536).contiguous().unsqueeze(0),
        key_heads[128:].to(torch.bfloat16).contiguous(),
    )
