// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once
#include "../mla_prolog_v3/mla_prolog_v3_torch_adpt.h"

namespace vllm_ascend {
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_mla_prolog_dcp_c8(
    const at::Tensor &token_x,
    const at::Tensor &weight_dq,
    const at::Tensor &weight_uq_qr,
    const at::Tensor &weight_uk,
    const at::Tensor &weight_dkv_kr,
    const at::Tensor &rmsnorm_gamma_cq,
    const at::Tensor &rmsnorm_gamma_ckv,
    const at::Tensor &rope_sin,
    const at::Tensor &rope_cos,
    at::Tensor &kv_cache,
    at::Tensor &kr_cache,
    const at::Tensor &kv_descale,
    const c10::optional<at::Tensor> &cache_index,
    const c10::optional<at::Tensor> &dequant_scale_x,
    const c10::optional<at::Tensor> &dequant_scale_w_dq,
    const c10::optional<at::Tensor> &dequant_scale_w_uq_qr,
    const c10::optional<at::Tensor> &dequant_scale_w_dkv_kr,
    const c10::optional<at::Tensor> &quant_scale_ckv,
    const c10::optional<at::Tensor> &quant_scale_ckr,
    const c10::optional<at::Tensor> &smooth_scales_cq,
    const c10::optional<at::Tensor> &actual_seq_len,
    const c10::optional<at::Tensor> &k_nope_clip_alpha,
    double rmsnorm_epsilon_cq,
    double rmsnorm_epsilon_ckv,
    c10::string_view cache_mode,
    bool query_norm_flag,
    int64_t weight_quant_mode,
    int64_t kv_cache_quant_mode,
    int64_t query_quant_mode,
    int64_t ckvkr_repo_mode,
    int64_t quant_scale_repo_mode,
    int64_t tile_size,
    double qc_qr_scale,
    double kc_scale)
{
    // Required args; empty (numel==0) means RoPE off. Both must be empty or both non-empty.
    const bool rope_sin_empty = !rope_sin.defined() || rope_sin.numel() == 0;
    const bool rope_cos_empty = !rope_cos.defined() || rope_cos.numel() == 0;
    TORCH_CHECK(rope_sin_empty == rope_cos_empty,
                "rope_sin and rope_cos must both be empty or both non-empty");

    TORCH_CHECK(token_x.scalar_type() == at::kBFloat16 && token_x.size(-1) == 7168 &&
                rope_sin_empty && weight_quant_mode == 3 && kv_cache_quant_mode == 0 &&
                query_quant_mode == 0 && !query_norm_flag && cache_mode == "PA_BSND",
                "DCP8 C8 query fusion requires BF16 X, MXFP8 weights, BF16 PA_BSND cache and no RoPE");
    TORCH_CHECK(kv_descale.scalar_type() == at::kFloat && kv_descale.numel() == 1 &&
                kv_descale.device() == token_x.device(), "kv_descale must be a device FP32 scalar");
    auto outputs = ConstructMlaPrologV3Outputs(
        token_x, weight_dq, weight_uq_qr, weight_uk, rope_sin, query_norm_flag,
        weight_quant_mode, kv_cache_quant_mode);
    at::Tensor query = std::get<0>(outputs);
    at::Tensor query_rope = std::get<1>(outputs);
    at::Tensor dequant_scale_q_nope = std::get<2>(outputs);
    at::Tensor query_norm = std::get<3>(outputs);
    at::Tensor dequant_scale_q_norm = std::get<4>(outputs);

    std::string cache_mode_str = std::string(cache_mode);
    char *cache_mode_ptr = const_cast<char *>(cache_mode_str.c_str());

    // Pass undefined tensors to aclnn when empty; tiling infers RoPE off from null rope.
    at::Tensor rope_sin_aclnn = rope_sin_empty ? at::Tensor() : rope_sin;
    at::Tensor rope_cos_aclnn = rope_cos_empty ? at::Tensor() : rope_cos;

    TORCH_CHECK(token_x.numel() / token_x.size(-1) == 64 && weight_uk.size(0) == 96,
                "DCP8 C8 query fusion requires T64/H96");
    at::Tensor query_c8 = at::empty(query.sizes(), query.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor rope_c8 = at::empty(query_rope.sizes(), query_rope.options());
    at::Tensor scale_c8 = at::empty({64, 96, 1}, token_x.options().dtype(at::kFloat));
    EXEC_NPU_CMD(
        aclnnInnerMlaPrologDcpC8,
        token_x,
        weight_dq,
        weight_uq_qr,
        weight_uk,
        weight_dkv_kr,
        rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv,
        rope_sin_aclnn,
        rope_cos_aclnn,
        kv_cache,
        kr_cache,
        cache_index,
        dequant_scale_x,
        dequant_scale_w_dq,
        dequant_scale_w_uq_qr,
        dequant_scale_w_dkv_kr,
        quant_scale_ckv,
        quant_scale_ckr,
        smooth_scales_cq,
        actual_seq_len,
        k_nope_clip_alpha,
        kv_descale,
        rmsnorm_epsilon_cq,
        rmsnorm_epsilon_ckv,
        cache_mode_ptr,
        query_norm_flag,
        weight_quant_mode,
        kv_cache_quant_mode,
        query_quant_mode,
        ckvkr_repo_mode,
        quant_scale_repo_mode,
        tile_size,
        qc_qr_scale,
        kc_scale,
        query,
        query_rope,
        dequant_scale_q_nope,
        query_norm,
        dequant_scale_q_norm,
        query_c8,
        rope_c8,
        scale_c8);

    return {query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm, query_c8, rope_c8, scale_c8};
}

}  // namespace vllm_ascend
