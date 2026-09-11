/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */
#ifndef KDA_TORCH_ADPT_COMMON_H
#define KDA_TORCH_ADPT_COMMON_H

#include <vector>

namespace vllm_ascend {
namespace {

int64_t kda_ceil_div(int64_t x, int64_t y)
{
    return (x + y - 1) / y;
}

int64_t get_kda_seq_num(int64_t batch, const c10::optional<at::IntArrayRef> &cu_seqlens)
{
    if (!cu_seqlens.has_value()) {
        return batch;
    }
    return static_cast<int64_t>(cu_seqlens.value().size()) - 1;
}

void check_kda_cu_seqlens(const c10::optional<at::IntArrayRef> &cu_seqlens,
                          int64_t total_tokens,
                          const char *op_name)
{
    if (!cu_seqlens.has_value()) {
        return;
    }
    auto cu = cu_seqlens.value();
    TORCH_CHECK(cu.size() >= 2, op_name, ": cu_seqlens must contain at least [0, total_tokens].");
    TORCH_CHECK(cu[0] == 0, op_name, ": cu_seqlens[0] must be 0, but got ", cu[0], ".");
    TORCH_CHECK(cu[cu.size() - 1] == total_tokens,
                op_name, ": cu_seqlens[-1] must equal sequence length ",
                total_tokens, ", but got ", cu[cu.size() - 1], ".");
    for (size_t i = 0; i + 1 < cu.size(); ++i) {
        TORCH_CHECK(cu[i] <= cu[i + 1],
                    op_name, ": cu_seqlens must be nondecreasing, but cu_seqlens[",
                    i, "]=", cu[i], " > cu_seqlens[", i + 1, "]=", cu[i + 1], ".");
    }
}

void check_kda_chunk_indices(const c10::optional<at::IntArrayRef> &chunk_indices,
                             const c10::optional<at::IntArrayRef> &cu_seqlens,
                             int64_t chunk_size,
                             const char *op_name)
{
    if (!chunk_indices.has_value()) {
        return;
    }
    auto indices = chunk_indices.value();
    TORCH_CHECK(indices.size() % 2 == 0,
                op_name, ": chunk_indices must contain (seq_id, chunk_id) pairs, but got ",
                indices.size(), " elements.");
    TORCH_CHECK(cu_seqlens.has_value(), op_name, ": chunk_indices requires cu_seqlens.");
    auto cu = cu_seqlens.value();
    int64_t expected_chunks = 0;
    for (size_t seq = 0; seq + 1 < cu.size(); ++seq) {
        expected_chunks += kda_ceil_div(cu[seq + 1] - cu[seq], chunk_size);
    }
    TORCH_CHECK(static_cast<int64_t>(indices.size() / 2) == expected_chunks,
                op_name, ": chunk_indices must contain exactly one pair per chunk.");
    for (size_t idx = 0; idx < indices.size(); idx += 2) {
        int64_t seq = indices[idx];
        int64_t chunk = indices[idx + 1];
        TORCH_CHECK(seq >= 0 && seq + 1 < static_cast<int64_t>(cu.size()),
                    op_name, ": chunk_indices seq_id is out of range.");
        int64_t chunks = kda_ceil_div(cu[seq + 1] - cu[seq], chunk_size);
        TORCH_CHECK(chunk >= 0 && chunk < chunks,
                    op_name, ": chunk_indices chunk_id is out of range.");
    }
}

int64_t get_kda_total_chunks(int64_t batch,
                             int64_t seqlen,
                             int64_t chunk_size,
                             const c10::optional<at::IntArrayRef> &cu_seqlens,
                             const c10::optional<at::IntArrayRef> &chunk_indices)
{
    if (chunk_indices.has_value()) {
        return static_cast<int64_t>(chunk_indices.value().size()) / 2;
    }
    if (!cu_seqlens.has_value()) {
        return kda_ceil_div(seqlen, chunk_size);
    }
    (void)batch;
    int64_t total = 0;
    auto cu = cu_seqlens.value();
    for (size_t i = 0; i + 1 < cu.size(); ++i) {
        total += kda_ceil_div(cu[i + 1] - cu[i], chunk_size);
    }
    return total;
}

std::vector<int64_t> build_kda_chunk_indices(at::IntArrayRef cu_seqlens, int64_t chunk_size)
{
    std::vector<int64_t> indices;
    int64_t total_chunks = 0;
    for (size_t i = 0; i + 1 < cu_seqlens.size(); ++i) {
        total_chunks += kda_ceil_div(cu_seqlens[i + 1] - cu_seqlens[i], chunk_size);
    }
    indices.reserve(static_cast<size_t>(total_chunks * 2));
    for (size_t seq = 0; seq + 1 < cu_seqlens.size(); ++seq) {
        int64_t seq_len = cu_seqlens[seq + 1] - cu_seqlens[seq];
        int64_t chunks = kda_ceil_div(seq_len, chunk_size);
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            indices.push_back(static_cast<int64_t>(seq));
            indices.push_back(chunk);
        }
    }
    return indices;
}

} // namespace
} // namespace vllm_ascend

#endif
