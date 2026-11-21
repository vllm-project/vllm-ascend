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

#ifndef MLA_PREPROCESS_TORCH_ADPT_H
#define MLA_PREPROCESS_TORCH_ADPT_H

#include "mla_preprocess_validation.h"

namespace vllm_ascend {
namespace mla_preprocess_detail {

inline int64_t ParseCacheMode(c10::optional<c10::string_view> cacheMode)
{
    const c10::string_view mode = cacheMode.value_or("krope_ctkv");
    if (mode == "kvcache") {
        return 0;
    }
    if (mode == "krope_ctkv") {
        return 1;
    }
    if (mode == "int8_nzcache") {
        return 2;
    }
    if (mode == "nzcache") {
        return 3;
    }
    TORCH_CHECK(false, "Unsupported cache_mode value: '", mode, "'");
    return 0;
}

// The kernel supports an enlarged dim0 stride for paged cache blocks. Every
// inner axis must still use the compact layout expected by the selected mode.
inline void ValidateCacheNonFirstAxisContiguous(const at::Tensor &cache, const char *tensorName)
{
    if (cache.dim() <= 1) {
        return;
    }
    const auto sizes = cache.sizes();
    const auto strides = cache.strides();
    int64_t expectedStride = 1;
    for (int64_t dim = cache.dim() - 1; dim >= 1; --dim) {
        if (sizes[dim] == 1) {
            continue;
        }
        TORCH_CHECK(strides[dim] == expectedStride,
                    tensorName, " dim", dim, " is non-contiguous: actual stride=", strides[dim],
                    ", expected contiguous stride=", expectedStride,
                    ". Only dim0/blockNum may be non-contiguous.");
        expectedStride *= sizes[dim];
    }
}

inline int64_t GetStride0(const at::Tensor &cache, const char *tensorName)
{
    TORCH_CHECK(cache.dim() >= 1, tensorName, " must have at least one dimension.");
    TORCH_CHECK(cache.stride(0) > 0, tensorName, " dim0 stride must be positive.");
    return cache.stride(0);
}

inline bool IsPhysicalNzCache(const at::Tensor &cache, int64_t cacheMode)
{
    return (cacheMode == 2 || cacheMode == 3) && cache.dim() == 4 && cache.size(2) != 1;
}

inline int64_t GetRopeHeadDim(const at::Tensor &kvCacheRope, int64_t cacheMode)
{
    TORCH_CHECK(kvCacheRope.dim() >= 1, "kv_cache_rope must not be a scalar.");
    if (IsPhysicalNzCache(kvCacheRope, cacheMode)) {
        return kvCacheRope.size(1) * kvCacheRope.size(3);
    }
    return kvCacheRope.size(-1);
}

}  // namespace mla_preprocess_detail

std::tuple<at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &> mla_preprocess(
    const at::Tensor &hiddenState, const at::Tensor &wdqkv,
    const c10::optional<at::Tensor> &descale0, const at::Tensor &gamma1,
    const c10::optional<at::Tensor> &beta1, const at::Tensor &wuq,
    const c10::optional<at::Tensor> &descale1, const at::Tensor &gamma2,
    const c10::optional<at::Tensor> &cos, const c10::optional<at::Tensor> &sin,
    const at::Tensor &wuk, const at::Tensor &kv_cache, const at::Tensor &kv_cache_rope,
    const at::Tensor &slotmapping, const c10::optional<at::Tensor> &quant_scale0,
    const c10::optional<at::Tensor> &quant_offset0, const c10::optional<at::Tensor> &bias0,
    const c10::optional<at::Tensor> &quant_scale1, const c10::optional<at::Tensor> &quant_offset1,
    const c10::optional<at::Tensor> &bias1, const c10::optional<at::Tensor> &ctkv_scale,
    const c10::optional<at::Tensor> &q_nope_scale, c10::optional<c10::string_view> cache_mode,
    c10::optional<c10::string_view> quant_mode, c10::optional<bool> enable_inner_out,
    at::Tensor &q_out0, at::Tensor &kv_cache_out0, at::Tensor &q_out1,
    at::Tensor &kv_cache_out1, at::Tensor &inner_out)
{
    mla_preprocess_detail::CheckRopePair(cos, sin);

    const int64_t cacheMode = mla_preprocess_detail::ParseCacheMode(cache_mode);
    const int64_t quantMode = mla_preprocess_detail::ParseQuantMode(quant_mode);
    const bool enableInnerOut = enable_inner_out.value_or(false);
    const bool enableRope = cos.has_value();
    const bool directOutput = false;
    const int64_t noWriteback = 0;
    const at::Tensor slotMapping =
        slotmapping.scalar_type() == at::kLong ? slotmapping.to(at::kInt) : slotmapping;

    mla_preprocess_detail::ValidateCacheNonFirstAxisContiguous(kv_cache, "kv_cache");
    mla_preprocess_detail::ValidateCacheNonFirstAxisContiguous(kv_cache_rope, "kv_cache_rope");
    const int64_t kvCacheStride0 = mla_preprocess_detail::GetStride0(kv_cache, "kv_cache");
    const int64_t kvCacheRopeStride0 =
        mla_preprocess_detail::GetStride0(kv_cache_rope, "kv_cache_rope");

    const int64_t kvLoraRank = wuk.size(-1);
    const int64_t qkRopeHeadDim = mla_preprocess_detail::GetRopeHeadDim(kv_cache_rope, cacheMode);
    mla_preprocess_detail::WarnUnsupportedDimensions(kvLoraRank, qkRopeHeadDim);

    EXEC_NPU_CMD(
        aclnnMlaPreprocess,
        hiddenState,
        quant_scale0,
        quant_offset0,
        wdqkv,
        bias0,
        gamma1,
        beta1,
        quant_scale1,
        quant_offset1,
        gamma2,
        sin,
        cos,
        sin,
        cos,
        kv_cache,
        slotMapping,
        wuq,
        bias1,
        wuk,
        descale0,
        descale1,
        ctkv_scale,
        q_nope_scale,
        cacheMode,
        quantMode,
        enableInnerOut,
        enableRope,
        kvCacheStride0,
        kvCacheRopeStride0,
        directOutput,
        noWriteback,
        noWriteback,
        noWriteback,
        q_out0,
        kv_cache_out0,
        q_out1,
        kv_cache_out1,
        inner_out);

    return std::forward_as_tuple(q_out0, kv_cache_out0, q_out1, kv_cache_out1, inner_out);
}

std::tuple<at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &> mla_preprocess_with_kv_offload(
    const at::Tensor &hiddenState, const at::Tensor &wdqkv,
    const c10::optional<at::Tensor> &descale0, const at::Tensor &gamma1,
    const c10::optional<at::Tensor> &beta1, const at::Tensor &wuq,
    const c10::optional<at::Tensor> &descale1, const at::Tensor &gamma2,
    const c10::optional<at::Tensor> &cos, const c10::optional<at::Tensor> &sin,
    const at::Tensor &wuk, const c10::optional<at::Tensor> &quant_scale0,
    const c10::optional<at::Tensor> &quant_offset0, const c10::optional<at::Tensor> &bias0,
    const c10::optional<at::Tensor> &quant_scale1, const c10::optional<at::Tensor> &quant_offset1,
    const c10::optional<at::Tensor> &bias1, const c10::optional<at::Tensor> &ctkv_scale,
    const c10::optional<at::Tensor> &q_nope_scale, c10::optional<c10::string_view> quant_mode,
    at::Tensor &q_out0, at::Tensor &q_out1, at::Tensor &inner_out,
    at::Tensor &current_k, at::Tensor &current_rope,
    const c10::optional<at::Tensor> &writeback_slots,
    int64_t writeback_k_base, int64_t writeback_rope_base, int64_t writeback_capacity)
{
    const int64_t quantMode = CheckDirectMlapo(hiddenState, gamma1, cos, sin, wuk, quant_mode,
                                               q_out0, q_out1, inner_out, current_k, current_rope);
    CheckDirectMlapoWriteback(hiddenState, current_k, current_rope, writeback_slots,
                              writeback_k_base, writeback_rope_base, writeback_capacity);
    TORCH_CHECK(hiddenState.device().type() == c10::DeviceType::PrivateUse1,
                "mla_preprocess_with_kv_offload requires NPU tensors");
    // Compute inputs use the original operator's format handling. Direct
    // outputs are raw contiguous rows, so verify their physical layout here.
    for (const at::Tensor *tensor : {&q_out0, &q_out1, &inner_out, &current_k, &current_rope}) {
        const auto format = at_npu::native::custom_ops::get_npu_format(*tensor);
        TORCH_CHECK(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW,
                    "mla_preprocess_with_kv_offload requires ND-compatible output storage");
    }
    if (writeback_slots.has_value()) {
        const auto format = at_npu::native::custom_ops::get_npu_format(writeback_slots.value());
        TORCH_CHECK(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW,
                    "MLAPO writeback_slots requires ND-compatible storage");
    }
    // Reuse main's ACLNN tiling/workspace lifecycle. The original operator
    // remains in its default paged-cache mode; this call selects Direct rows.
    const at::Tensor slots = writeback_slots.has_value()
        ? writeback_slots.value()
        : at::empty({hiddenState.size(0)}, hiddenState.options().dtype(at::kInt));
    const int64_t cacheMode = 1;
    const bool enableInnerOut = true;
    const bool enableRope = cos.has_value();
    const bool directOutput = true;
    const int64_t kvStride0 = current_k.stride(0);
    const int64_t ropeStride0 = current_rope.stride(0);
    EXEC_NPU_CMD(
        aclnnMlaPreprocess,
        hiddenState,
        quant_scale0,
        quant_offset0,
        wdqkv,
        bias0,
        gamma1,
        beta1,
        quant_scale1,
        quant_offset1,
        gamma2,
        sin,
        cos,
        sin,
        cos,
        current_k,
        slots,
        wuq,
        bias1,
        wuk,
        descale0,
        descale1,
        ctkv_scale,
        q_nope_scale,
        cacheMode,
        quantMode,
        enableInnerOut,
        enableRope,
        kvStride0,
        ropeStride0,
        directOutput,
        writeback_k_base,
        writeback_rope_base,
        writeback_capacity,
        q_out0,
        current_k,
        q_out1,
        current_rope,
        inner_out);
    return std::forward_as_tuple(q_out0, q_out1, inner_out, current_k, current_rope);
}

}  // namespace vllm_ascend

#endif  // MLA_PREPROCESS_TORCH_ADPT_H
