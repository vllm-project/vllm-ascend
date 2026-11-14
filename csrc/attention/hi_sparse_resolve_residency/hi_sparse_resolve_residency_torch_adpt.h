// SPDX-License-Identifier: Apache-2.0
#ifndef HI_SPARSE_RESOLVE_RESIDENCY_TORCH_ADPT_H
#define HI_SPARSE_RESOLVE_RESIDENCY_TORCH_ADPT_H

#include <ATen/ATen.h>

#include "op_kernel/resolver_scratch.h"

namespace vllm_ascend::hisparse {

inline void validate_resolver_scratch(int64_t hot, int64_t union_rows)
{
    TORCH_CHECK(union_rows > 0 && union_rows <= hot && hot <= 32768,
                "HiSparse union must fit the hot cache and int16 LRU slots");
    const uint64_t minimum_bytes = resolver_scratch_bytes(hot, union_rows + 1);
    TORCH_CHECK(minimum_bytes <= kResolverUbBytes,
                "HiSparse hot_size=", hot, ", max_union_rows=", union_rows,
                " needs at least ", minimum_bytes,
                " UB bytes; the compiled target allows ", kResolverUbBytes,
                ". Lower device_buffer_size or speculative tokens.");
}

/**
 * HiSparse residency resolution. All outputs are caller-owned and mutated in
 * place; the adapter maps the torch op arguments onto the ACLNN wrapper
 * signature (inputs, attributes, outputs).
 */
inline void resolve_residency(
    const at::Tensor &logical, const at::Tensor &hot_table,
    const std::optional<at::Tensor> &requests, const std::optional<at::Tensor> &states,
    const std::optional<at::Tensor> &source, const std::optional<at::Tensor> &resident,
    const std::optional<at::Tensor> &valid_rows, const at::Tensor &dgi,
    const at::Tensor &lru, const at::Tensor &output, const at::Tensor &resolved,
    const at::Tensor &errors, const std::optional<at::Tensor> &attention,
    const std::optional<at::Tensor> &counts, const std::optional<at::Tensor> &swap_host,
    const std::optional<at::Tensor> &swap_device, const std::optional<at::Tensor> &swap_counts,
    const std::optional<at::Tensor> &miss_mask, const std::optional<at::Tensor> &stats,
    int64_t host_rows, int64_t hot_block_size, int64_t source_block_size,
    int64_t resident_block_size, int64_t resident_null_block,
    int64_t attention_block_stride, int64_t union_rows, int64_t attention_capacity)
{
    const int64_t topk = logical.size(1);
    // EXEC_NPU_CMD binds arguments to non-const lvalue references, so every
    // argument must be a named local instead of a temporary.
    const at::Tensor requests_t = requests.value_or(at::Tensor());
    const at::Tensor states_t = states.value_or(at::Tensor());
    const at::Tensor source_t = source.value_or(at::Tensor());
    const at::Tensor resident_t = resident.value_or(at::Tensor());
    const at::Tensor valid_rows_t = valid_rows.value_or(at::Tensor());
    const at::Tensor attention_t = attention.value_or(at::Tensor());
    const at::Tensor counts_t = counts.value_or(at::Tensor());
    const at::Tensor swap_host_t = swap_host.value_or(at::Tensor());
    const at::Tensor swap_device_t = swap_device.value_or(at::Tensor());
    const at::Tensor swap_counts_t = swap_counts.value_or(at::Tensor());
    const at::Tensor miss_mask_t = miss_mask.value_or(at::Tensor());
    const at::Tensor stats_t = stats.value_or(at::Tensor());
    const int64_t logical_stride = logical.stride(0);
    const int64_t source_stride = source ? source->stride(0) : 0;
    const int64_t resident_stride = resident ? resident->stride(0) : 0;
    const int64_t hot_table_stride = hot_table.stride(0);
    const int64_t attention_stride = attention ? attention->stride(0) : topk;
    const int64_t counts_stride = counts ? counts->stride(0) : 1;
    const int64_t miss_mask_bytes = static_cast<int64_t>(miss_mask.has_value() && miss_mask->element_size() == 1);
    EXEC_NPU_CMD(
        aclnnHiSparseResolveResidency,
        logical, hot_table,
        requests_t, states_t,
        source_t, resident_t,
        valid_rows_t,
        host_rows, hot_block_size, source_block_size, resident_block_size,
        resident_null_block, attention_block_stride, union_rows, attention_capacity,
        logical_stride, source_stride,
        resident_stride, hot_table_stride,
        attention_stride, counts_stride,
        miss_mask_bytes,
        dgi, lru, output, resolved, errors,
        attention_t, counts_t,
        swap_host_t, swap_device_t,
        swap_counts_t, miss_mask_t,
        stats_t);
}

}  // namespace vllm_ascend::hisparse

#endif  // HI_SPARSE_RESOLVE_RESIDENCY_TORCH_ADPT_H
