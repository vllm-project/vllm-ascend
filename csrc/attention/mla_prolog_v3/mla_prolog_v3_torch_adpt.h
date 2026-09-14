/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MLA_PROLOG_V3_TORCH_ADPT_H
#define MLA_PROLOG_V3_TORCH_ADPT_H

namespace vllm_ascend {

namespace {

constexpr int64_t FP8_E4M3_BLOCK_SIZE = 32;
constexpr int64_t WEIGHT_QUANT_MODE_NO_QUANT = 0;
constexpr int64_t WEIGHT_QUANT_MODE_FULL_QUANT = 2;
constexpr int64_t WEIGHT_QUANT_MODE_MXFP8_FULL_QUANT = 3;
constexpr int64_t WEIGHT_QUANT_MODE_FULL_QUANT_FP8 = 4;
constexpr int64_t WEIGHT_QUANT_MODE_FULL_QUANT_HIF8 = 5;
constexpr int64_t KV_CACHE_QUANT_MODE_PER_TENSOR = 1;

bool NeedDequantScaleQNope(int64_t weight_quant_mode, int64_t kv_cache_quant_mode)
{
    return (weight_quant_mode == WEIGHT_QUANT_MODE_FULL_QUANT ||
            weight_quant_mode == WEIGHT_QUANT_MODE_MXFP8_FULL_QUANT ||
            weight_quant_mode == WEIGHT_QUANT_MODE_FULL_QUANT_FP8 ||
            weight_quant_mode == WEIGHT_QUANT_MODE_FULL_QUANT_HIF8) &&
           kv_cache_quant_mode == KV_CACHE_QUANT_MODE_PER_TENSOR;
}

at::ScalarType GetQueryDtype(const at::Tensor &rope_sin, int64_t weight_quant_mode,
                             int64_t kv_cache_quant_mode)
{
    if (weight_quant_mode == WEIGHT_QUANT_MODE_MXFP8_FULL_QUANT &&
        kv_cache_quant_mode == KV_CACHE_QUANT_MODE_PER_TENSOR) {
        return at::kFloat8_e4m3fn;
    }
    if (weight_quant_mode == WEIGHT_QUANT_MODE_FULL_QUANT &&
        kv_cache_quant_mode == KV_CACHE_QUANT_MODE_PER_TENSOR) {
        return at::kChar;
    }
    // Empty rope means RoPE off; default query dtype to BF16.
    if (!rope_sin.defined() || rope_sin.numel() == 0) {
        return at::kBFloat16;
    }
    return rope_sin.scalar_type();
}

at::ScalarType GetQueryNormDtype(const at::Tensor &weight_uq_qr, int64_t weight_quant_mode)
{
    if (weight_quant_mode == WEIGHT_QUANT_MODE_MXFP8_FULL_QUANT ||
        weight_quant_mode == WEIGHT_QUANT_MODE_FULL_QUANT_FP8) {
        return at::kFloat8_e4m3fn;
    }
    if (weight_quant_mode == WEIGHT_QUANT_MODE_NO_QUANT) {
        return at::kBFloat16;
    }
    return weight_uq_qr.scalar_type();
}

at::ScalarType GetDequantScaleQNormDtype(int64_t weight_quant_mode)
{
    if (weight_quant_mode == WEIGHT_QUANT_MODE_MXFP8_FULL_QUANT) {
        return at::kFloat8_e8m0fnu;
    }
    return at::kFloat;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
ConstructMlaPrologV3Outputs(const at::Tensor &token_x, const at::Tensor &weight_dq,
                            const at::Tensor &weight_uq_qr, const at::Tensor &weight_uk,
                            const at::Tensor &rope_sin, bool query_norm_flag,
                            int64_t weight_quant_mode, int64_t kv_cache_quant_mode)
{
    const int64_t token_x_dim = token_x.dim();
    TORCH_CHECK(token_x_dim == 2 || token_x_dim == 3,
                "token_x dim num should be 2 or 3, but got ", token_x_dim);
    TORCH_CHECK(weight_uk.dim() == 3,
                "weight_uk dim num should be 3, but got ", weight_uk.dim());

    std::vector<int64_t> query_shape;
    std::vector<int64_t> query_rope_shape;
    std::vector<int64_t> dequant_scale_q_nope_shape;
    std::vector<int64_t> query_norm_shape;
    std::vector<int64_t> dequant_scale_q_norm_shape;

    const bool rope_enabled = rope_sin.defined() && rope_sin.numel() > 0;
    constexpr int64_t kDefaultRopeDim = 64;

    if (token_x_dim == 3) {
        if (rope_enabled) {
            TORCH_CHECK(rope_sin.dim() == 3,
                        "when token_x dim num is 3, rope_sin dim num should be 3, but got ",
                        rope_sin.dim());
        }
        const int64_t rope_dim = rope_enabled ? rope_sin.size(2) : kDefaultRopeDim;
        query_shape = {token_x.size(0), token_x.size(1), weight_uk.size(0), weight_uk.size(2)};
        query_rope_shape = {token_x.size(0), token_x.size(1), weight_uk.size(0), rope_dim};
        dequant_scale_q_nope_shape = {token_x.size(0) * token_x.size(1), weight_uk.size(0), 1};
        query_norm_shape = {token_x.size(0), token_x.size(1), weight_dq.size(1)};
        dequant_scale_q_norm_shape = {token_x.size(0) * token_x.size(1)};
        if (weight_quant_mode == WEIGHT_QUANT_MODE_MXFP8_FULL_QUANT) {
            dequant_scale_q_norm_shape.push_back(weight_dq.size(1) / FP8_E4M3_BLOCK_SIZE);
        } else {
            dequant_scale_q_norm_shape.push_back(1);
        }
    } else {
        if (rope_enabled) {
            TORCH_CHECK(rope_sin.dim() == 2,
                        "when token_x dim num is 2, rope_sin dim num should be 2, but got ",
                        rope_sin.dim());
        }
        const int64_t rope_dim = rope_enabled ? rope_sin.size(1) : kDefaultRopeDim;
        query_shape = {token_x.size(0), weight_uk.size(0), weight_uk.size(2)};
        query_rope_shape = {token_x.size(0), weight_uk.size(0), rope_dim};
        dequant_scale_q_nope_shape = {token_x.size(0), weight_uk.size(0), 1};
        query_norm_shape = {token_x.size(0), weight_dq.size(1)};
        dequant_scale_q_norm_shape = {token_x.size(0)};
        if (weight_quant_mode == WEIGHT_QUANT_MODE_MXFP8_FULL_QUANT) {
            dequant_scale_q_norm_shape.push_back(weight_dq.size(1) / FP8_E4M3_BLOCK_SIZE);
        } else {
            dequant_scale_q_norm_shape.push_back(1);
        }
    }

    const auto device_opts = token_x.options();
    at::Tensor query = at::empty(
        query_shape, device_opts.dtype(GetQueryDtype(rope_sin, weight_quant_mode, kv_cache_quant_mode)));
    at::Tensor query_rope = at::empty(query_rope_shape, device_opts.dtype(at::kBFloat16));

    at::Tensor dequant_scale_q_nope;
    if (NeedDequantScaleQNope(weight_quant_mode, kv_cache_quant_mode)) {
        dequant_scale_q_nope = at::empty(dequant_scale_q_nope_shape, device_opts.dtype(at::kFloat));
    } else {
        dequant_scale_q_nope = at::empty({0}, device_opts.dtype(at::kFloat));
    }

    at::Tensor query_norm;
    at::Tensor dequant_scale_q_norm;
    if (query_norm_flag) {
        query_norm = at::empty(query_norm_shape,
                               device_opts.dtype(GetQueryNormDtype(weight_uq_qr, weight_quant_mode)));
        if (weight_quant_mode != WEIGHT_QUANT_MODE_NO_QUANT) {
            dequant_scale_q_norm = at::empty(
                dequant_scale_q_norm_shape,
                device_opts.dtype(GetDequantScaleQNormDtype(weight_quant_mode)));
        } else {
            dequant_scale_q_norm = at::empty(
                {0}, device_opts.dtype(GetDequantScaleQNormDtype(weight_quant_mode)));
        }
    } else {
        query_norm = at::empty({0}, device_opts.dtype(GetQueryNormDtype(weight_uq_qr, weight_quant_mode)));
        dequant_scale_q_norm = at::empty(
            {0}, device_opts.dtype(GetDequantScaleQNormDtype(weight_quant_mode)));
    }

    return {query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm};
}

}  // namespace

// Torch schema name is npu_mla_prolog_v3 (aligned with torch_npu); underlying aclnn op is MlaPrologV3.
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_mla_prolog_v3(
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

    auto outputs = ConstructMlaPrologV3Outputs(
        token_x, weight_dq, weight_uq_qr, weight_uk, rope_sin, query_norm_flag,
        weight_quant_mode, kv_cache_quant_mode);
    at::Tensor query = std::get<0>(outputs);
    at::Tensor query_rope = std::get<1>(outputs);
    at::Tensor dequant_scale_q_nope = std::get<2>(outputs);
    at::Tensor query_norm = std::get<3>(outputs);
    at::Tensor dequant_scale_q_norm = std::get<4>(outputs);

    // aclnnMlaPrologV3WeightNz derives queryNormFlag from whether optional outs are non-null.
    c10::optional<at::Tensor> dequant_scale_q_nope_opt =
        NeedDequantScaleQNope(weight_quant_mode, kv_cache_quant_mode)
            ? c10::optional<at::Tensor>(dequant_scale_q_nope)
            : c10::nullopt;
    c10::optional<at::Tensor> query_norm_opt =
        query_norm_flag ? c10::optional<at::Tensor>(query_norm) : c10::nullopt;
    c10::optional<at::Tensor> dequant_scale_q_norm_opt =
        (query_norm_flag && weight_quant_mode != WEIGHT_QUANT_MODE_NO_QUANT)
            ? c10::optional<at::Tensor>(dequant_scale_q_norm)
            : c10::nullopt;

    std::string cache_mode_str = std::string(cache_mode);
    char *cache_mode_ptr = const_cast<char *>(cache_mode_str.c_str());

    // Pass undefined tensors to aclnn when empty; tiling infers RoPE off from null rope.
    at::Tensor rope_sin_aclnn = rope_sin_empty ? at::Tensor() : rope_sin;
    at::Tensor rope_cos_aclnn = rope_cos_empty ? at::Tensor() : rope_cos;

    EXEC_NPU_CMD(
        aclnnMlaPrologV3WeightNz,
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
        rmsnorm_epsilon_cq,
        rmsnorm_epsilon_ckv,
        cache_mode_ptr,
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
        dequant_scale_q_nope_opt,
        query_norm_opt,
        dequant_scale_q_norm_opt);

    return {query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm};
}

}  // namespace vllm_ascend

#endif  // MLA_PROLOG_V3_TORCH_ADPT_H
