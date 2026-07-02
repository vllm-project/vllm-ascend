/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#ifndef RECOMPUTE_WU_FWD_TORCH_ADPT_H
#define RECOMPUTE_WU_FWD_TORCH_ADPT_H

#include <tuple>

namespace vllm_ascend {

inline bool is_recompute_wu_activation_dtype(c10::ScalarType dtype)
{
    return dtype == at::kHalf || dtype == at::kBFloat16;
}

inline bool is_recompute_wu_scale_dtype(c10::ScalarType dtype)
{
    return dtype == at::kHalf || dtype == at::kBFloat16 || dtype == at::kFloat;
}

std::tuple<at::Tensor, at::Tensor> npu_recompute_wu_fwd(
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& beta,
    const at::Tensor& a,
    const at::Tensor& g,
    c10::optional<at::IntArrayRef> cu_seqlens,
    c10::optional<at::IntArrayRef> chunk_indices,
    int64_t chunk_size = 64)
{
    TORCH_CHECK(k.dim() == 4, "k must be 4D [B, HK, T, K], got ", k.dim(), "D");
    TORCH_CHECK(v.dim() == 4, "v must be 4D [B, HV, T, V], got ", v.dim(), "D");
    TORCH_CHECK(a.dim() == 4, "a must be 4D [B, HV, T, chunk_size], got ", a.dim(), "D");
    TORCH_CHECK(beta.dim() == 3, "beta must be 3D [B, HV, T], got ", beta.dim(), "D");
    TORCH_CHECK(g.dim() == 3, "g must be 3D [B, HV, T], got ", g.dim(), "D");

    TORCH_CHECK(is_recompute_wu_activation_dtype(k.scalar_type()),
                "k must be float16 or bfloat16, got ", k.scalar_type());
    TORCH_CHECK(k.scalar_type() == v.scalar_type(),
                "k and v must have the same dtype, got ", k.scalar_type(), " and ", v.scalar_type());
    TORCH_CHECK(k.scalar_type() == a.scalar_type(),
                "k and a must have the same dtype, got ", k.scalar_type(), " and ", a.scalar_type());
    TORCH_CHECK(is_recompute_wu_scale_dtype(beta.scalar_type()),
                "beta must be float16, bfloat16, or float32, got ", beta.scalar_type());
    TORCH_CHECK(beta.scalar_type() == g.scalar_type(),
                "beta and g must have the same dtype, got ", beta.scalar_type(), " and ", g.scalar_type());

    const int64_t batch = k.size(0);
    const int64_t hk = k.size(1);
    const int64_t seq_len = k.size(2);
    const int64_t key_dim = k.size(3);
    const int64_t hv = v.size(1);
    const int64_t value_dim = v.size(3);

    TORCH_CHECK(batch > 0, "batch size must be positive, got ", batch);
    TORCH_CHECK(hk > 0, "HK must be positive, got ", hk);
    TORCH_CHECK(hv > 0, "HV must be positive, got ", hv);
    TORCH_CHECK(hv % hk == 0, "HV must be divisible by HK, got HV=", hv, ", HK=", hk);
    TORCH_CHECK(key_dim == 128, "k last dimension must be 128, got ", key_dim);
    TORCH_CHECK(value_dim == 128 || value_dim == 256,
                "v last dimension must be 128 or 256, got ", value_dim);
    TORCH_CHECK(chunk_size == 64 || chunk_size == 128,
                "chunk_size must be 64 or 128, got ", chunk_size);

    TORCH_CHECK(v.size(0) == batch && v.size(2) == seq_len,
                "v shape must match k batch and sequence, got k=", k.sizes(), ", v=", v.sizes());
    TORCH_CHECK(a.size(0) == batch && a.size(1) == hv && a.size(2) == seq_len && a.size(3) == chunk_size,
                "a shape must be [B, HV, T, chunk_size], got ", a.sizes(),
                " for B=", batch, ", HV=", hv, ", T=", seq_len, ", chunk_size=", chunk_size);
    TORCH_CHECK(beta.size(0) == batch && beta.size(1) == hv && beta.size(2) == seq_len,
                "beta shape must be [B, HV, T], got ", beta.sizes());
    TORCH_CHECK(g.sizes() == beta.sizes(), "g shape must match beta shape, got g=", g.sizes(), ", beta=", beta.sizes());
    TORCH_CHECK(cu_seqlens.has_value() == chunk_indices.has_value(),
                "cu_seqlens and chunk_indices must be provided together");
    if (cu_seqlens.has_value()) {
        TORCH_CHECK(batch == 1, "varlen mode expects batch size 1, got ", batch);
    }

    auto w = at::empty({batch, hv, seq_len, key_dim}, k.options());
    auto u = at::empty_like(v);
    const c10::optional<at::Tensor> gk = c10::nullopt;
    EXEC_NPU_CMD(aclnnRecomputeWUFwd, k, v, beta, a, g, gk, cu_seqlens, chunk_indices, chunk_size, w, u);
    return std::make_tuple(w, u);
}

} // namespace vllm_ascend

#endif // RECOMPUTE_WU_FWD_TORCH_ADPT_H
