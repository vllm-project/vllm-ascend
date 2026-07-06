/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#ifndef RECOMPUTE_WU_FWD_TORCH_ADPT_H
#define RECOMPUTE_WU_FWD_TORCH_ADPT_H

#include <tuple>
#include <algorithm>
#include <ATen/TensorIndexing.h>

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

    auto w = at::zeros({batch, hv, seq_len, key_dim}, k.options());
    auto u = at::zeros_like(v);

    if (!cu_seqlens.has_value()) {
        using at::indexing::Slice;

        const int64_t key_group_size = hv / hk;
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t h = 0; h < hv; ++h) {
                const int64_t hk_idx = h / key_group_size;
                for (int64_t start = 0; start < seq_len; start += chunk_size) {
                    const int64_t end = std::min(start + chunk_size, seq_len);
                    const int64_t block_len = end - start;

                    auto a_block = a.index({b, h, Slice(start, end), Slice(0, block_len)}).to(at::kFloat);
                    auto beta_block = beta.index({b, h, Slice(start, end)}).to(at::kFloat).unsqueeze(-1);
                    auto g_block = g.index({b, h, Slice(start, end)}).to(at::kFloat).unsqueeze(-1);
                    auto v_block = v.index({b, h, Slice(start, end), Slice()}).to(at::kFloat);
                    auto k_block = k.index({b, hk_idx, Slice(start, end), Slice()}).to(at::kFloat);

                    auto vb = v_block * beta_block;
                    auto kbg_exp = k_block * beta_block * at::exp(g_block);
                    u.index_put_({b, h, Slice(start, end), Slice()}, at::matmul(a_block, vb).to(v.scalar_type()));
                    w.index_put_({b, h, Slice(start, end), Slice()}, at::matmul(a_block, kbg_exp).to(k.scalar_type()));
                }
            }
        }
        return std::make_tuple(w, u);
    }

    using at::indexing::Slice;

    const auto cu = cu_seqlens.value();
    const auto chunks = chunk_indices.value();
    TORCH_CHECK(cu.size() >= 2, "cu_seqlens must contain at least start and end offsets, got ", cu.size());
    TORCH_CHECK(cu[0] == 0, "cu_seqlens must start from 0, got ", cu[0]);
    TORCH_CHECK(cu[cu.size() - 1] == seq_len,
                "last cu_seqlens entry must equal sequence length, got ", cu[cu.size() - 1],
                " and T=", seq_len);
    TORCH_CHECK(chunks.size() % 2 == 0,
                "chunk_indices must contain flattened (seq_id, chunk_id) pairs, got ", chunks.size(),
                " values");
    for (int64_t i = 1; i < static_cast<int64_t>(cu.size()); ++i) {
        TORCH_CHECK(cu[i] >= cu[i - 1], "cu_seqlens must be non-decreasing, got ", cu[i - 1], " then ", cu[i]);
        TORCH_CHECK(cu[i] <= seq_len, "cu_seqlens entry exceeds sequence length: ", cu[i], " > ", seq_len);
    }

    const int64_t key_group_size = hv / hk;
    for (int64_t flat_idx = 0; flat_idx < static_cast<int64_t>(chunks.size()); flat_idx += 2) {
        const int64_t seq_id = chunks[flat_idx];
        const int64_t chunk_id = chunks[flat_idx + 1];
        TORCH_CHECK(seq_id >= 0 && seq_id + 1 < static_cast<int64_t>(cu.size()),
                    "chunk_indices seq_id out of range: ", seq_id, " for cu_seqlens size ", cu.size());
        TORCH_CHECK(chunk_id >= 0, "chunk_indices chunk_id must be non-negative, got ", chunk_id);

        const int64_t seq_start = cu[seq_id];
        const int64_t seq_end = cu[seq_id + 1];
        const int64_t start = seq_start + chunk_id * chunk_size;
        TORCH_CHECK(start < seq_end,
                    "chunk_indices points past sequence end: seq_id=", seq_id,
                    ", chunk_id=", chunk_id, ", start=", start, ", seq_end=", seq_end);
        const int64_t end = std::min(start + chunk_size, seq_end);
        const int64_t block_len = end - start;

        for (int64_t h = 0; h < hv; ++h) {
            const int64_t hk_idx = h / key_group_size;
            auto a_block = a.index({0, h, Slice(start, end), Slice(0, block_len)}).to(at::kFloat);
            auto beta_block = beta.index({0, h, Slice(start, end)}).to(at::kFloat).unsqueeze(-1);
            auto g_block = g.index({0, h, Slice(start, end)}).to(at::kFloat).unsqueeze(-1);
            auto v_block = v.index({0, h, Slice(start, end), Slice()}).to(at::kFloat);
            auto k_block = k.index({0, hk_idx, Slice(start, end), Slice()}).to(at::kFloat);

            auto vb = v_block * beta_block;
            auto kbg_exp = k_block * beta_block * at::exp(g_block);
            u.index_put_({0, h, Slice(start, end), Slice()}, at::matmul(a_block, vb).to(v.scalar_type()));
            w.index_put_({0, h, Slice(start, end), Slice()}, at::matmul(a_block, kbg_exp).to(k.scalar_type()));
        }
    }
    return std::make_tuple(w, u);
}

} // namespace vllm_ascend

#endif // RECOMPUTE_WU_FWD_TORCH_ADPT_H
