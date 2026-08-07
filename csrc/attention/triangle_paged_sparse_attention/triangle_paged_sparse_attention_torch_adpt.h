/*
 * Copyright (c) 2026 TriangleMix contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#ifndef TRIANGLE_PAGED_SPARSE_ATTENTION_TORCH_ADPT_H
#define TRIANGLE_PAGED_SPARSE_ATTENTION_TORCH_ADPT_H

#include <ATen/MemoryOverlap.h>
#include <c10/core/DeviceGuard.h>

namespace vllm_ascend {

at::Tensor& npu_triangle_paged_sparse_attention(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& block_table,
    int64_t query_start,
    int64_t seq_len,
    int64_t prompt_len,
    double scale,
    at::Tensor& output)
{
    TORCH_CHECK(query.is_privateuseone(), "query must be an NPU tensor");
    TORCH_CHECK(
        key_cache.is_privateuseone() && value_cache.is_privateuseone() &&
            block_table.is_privateuseone() && output.is_privateuseone(),
        "key_cache, value_cache, block_table, and output must be NPU tensors");
    TORCH_CHECK(
        query.scalar_type() == at::ScalarType::BFloat16 &&
            key_cache.scalar_type() == at::ScalarType::BFloat16 &&
            value_cache.scalar_type() == at::ScalarType::BFloat16 &&
            output.scalar_type() == at::ScalarType::BFloat16,
        "TriangleMix query, KV cache, and output must be BF16");
    TORCH_CHECK(
        block_table.scalar_type() == at::ScalarType::Int,
        "TriangleMix block_table must be INT32");
    TORCH_CHECK(
        query.device() == key_cache.device() &&
            query.device() == value_cache.device() &&
            query.device() == block_table.device() &&
            query.device() == output.device(),
        "TriangleMix tensors must be on the same NPU device");
    TORCH_CHECK(
        query.is_contiguous() && key_cache.is_contiguous() &&
            value_cache.is_contiguous() && block_table.is_contiguous() &&
            output.is_contiguous(),
        "TriangleMix tensors must be contiguous");
    TORCH_CHECK(
        query.dim() == 3 && query.size(1) == 32 && query.size(2) == 128,
        "TriangleMix query must be [Tq, 32, 128]");
    TORCH_CHECK(
        key_cache.dim() == 4 && key_cache.size(1) == 128 &&
            key_cache.size(2) == 8 && key_cache.size(3) == 128,
        "TriangleMix key_cache must be [pages, 128, 8, 128]");
    TORCH_CHECK(
        value_cache.sizes() == key_cache.sizes(),
        "TriangleMix value_cache must match key_cache");
    TORCH_CHECK(
        block_table.dim() == 2 && block_table.size(0) == 1,
        "TriangleMix block_table must be [1, max_pages]");
    TORCH_CHECK(
        output.sizes() == query.sizes(),
        "TriangleMix output must match query");
    TORCH_CHECK(
        query_start >= 0 && seq_len > 0 &&
            query_start + query.size(0) == seq_len &&
            prompt_len >= seq_len,
        "invalid TriangleMix sequence coordinates");
    at::assert_no_internal_overlap(output);
    at::assert_no_overlap(output, query);
    at::assert_no_overlap(output, key_cache);
    at::assert_no_overlap(output, value_cache);
    at::assert_no_overlap(output, block_table);
    c10::OptionalDeviceGuard device_guard(query.device());

    constexpr int64_t query_tile = 32;
    constexpr int64_t page_size = 128;
    constexpr int64_t sink_tokens = 8;
    constexpr int64_t local_window = 512;
    constexpr int64_t dense_tail = 128;
    EXEC_NPU_CMD(
        aclnnTrianglePagedSparseAttention,
        query,
        key_cache,
        value_cache,
        block_table,
        query_start,
        seq_len,
        prompt_len,
        scale,
        query_tile,
        page_size,
        sink_tokens,
        local_window,
        dense_tail,
        output);
    return output;
}

}  // namespace vllm_ascend

#endif  // TRIANGLE_PAGED_SPARSE_ATTENTION_TORCH_ADPT_H
