// SPDX-License-Identifier: Apache-2.0
#ifndef HI_SPARSE_RESOLVE_RESIDENCY_TORCH_ADPT_H
#define HI_SPARSE_RESOLVE_RESIDENCY_TORCH_ADPT_H

#include <ATen/ATen.h>

#include "op_kernel/resolver_scratch.h"
#ifdef VLLM_ASCEND_HISPARSE_ARCH22
#include <tiling/platform/platform_ascendc.h>
#include <ascendcl.h>
#endif

namespace vllm_ascend::hisparse {

inline void validate_resolver_scratch(int64_t hot, int64_t union_rows)
{
#ifdef VLLM_ASCEND_HISPARSE_ARCH22
    // Fail fast at workspace construction; the tiling stage enforces the same
    // arch22 UB budget again at launch time.
    uint64_t platform_ub_bytes = 0;
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance(aclrtGetSocName());
    TORCH_CHECK(platform != nullptr, "HiSparse failed to query the target UB capacity");
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, platform_ub_bytes);
    const uint64_t budget = resolver_arch22_ub_budget(platform_ub_bytes);
    TORCH_CHECK(union_rows > 0 && union_rows <= hot && hot <= 32768,
                "HiSparse union must fit the hot cache and int16 LRU slots");
    const uint64_t minimum_bytes = resolver_arch22_scratch_bytes(hot, union_rows + 1);
    TORCH_CHECK(minimum_bytes <= budget,
                "HiSparse hot_size=", hot, ", max_union_rows=", union_rows,
                " needs at least ", minimum_bytes,
                " UB bytes; the target allows ", budget,
                ". Lower device_buffer_size or speculative tokens.");
#else
    TORCH_CHECK(union_rows > 0 && union_rows <= hot && hot <= 32768,
                "HiSparse union must fit the hot cache and int16 LRU slots");
    const uint64_t minimum_bytes = resolver_scratch_bytes(hot, union_rows + 1);
    TORCH_CHECK(minimum_bytes <= kResolverMaxSimtUbBytes,
                "HiSparse hot_size=", hot, ", max_union_rows=", union_rows,
                " needs at least ", minimum_bytes,
                " UB bytes; the target SIMT UB budget allows ", kResolverMaxSimtUbBytes,
                ". Lower device_buffer_size or speculative tokens.");
#endif
}

/**
 * HiSparse residency resolution. All outputs are caller-owned and mutated in
 * place; the adapter maps the torch op arguments onto the single ACLNN
 * wrapper signature (inputs, attributes, outputs). The `mode` attribute
 * selects the launch flavor inside the one aclnn op:
 *   0 = resolve residency, 2 = accumulate stats.
 * Invalidation runs as the shared Triton kernel on every SoC, so it never
 * reaches this aclnn wrapper. The aclnn wrapper selects the arch22 scalar-UB
 * kernel or the A5 SIMT kernel through the registered SoC configs; arch22
 * additionally accumulates its per-row error counters through a second
 * ordered mode=2 launch.
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
    // Resolve mode (0) leaves the accumulate-only inputs, attrs and outputs
    // unbound; empty tensors convert to nullptr inside EXEC_NPU_CMD.
    const at::Tensor empty;
    const int64_t zero = 0;
    EXEC_NPU_CMD(
        aclnnHiSparseResolveResidency,
        logical, hot_table,
        requests_t, states_t,
        source_t, resident_t,
        valid_rows_t,
        empty,
        host_rows, hot_block_size, source_block_size, resident_block_size,
        resident_null_block, attention_block_stride, union_rows, attention_capacity,
        logical_stride, source_stride,
        resident_stride, hot_table_stride,
        attention_stride, counts_stride,
        miss_mask_bytes,
        zero,
        dgi, lru, output, resolved, errors,
        attention_t, counts_t,
        swap_host_t, swap_device_t,
        swap_counts_t, miss_mask_t,
        stats_t);
#ifdef VLLM_ASCEND_HISPARSE_ARCH22
    // The arch22 resolve kernel cannot issue the ordered int64 stats
    // accumulation as a secondary launch, so it runs through the same aclnn
    // op with mode=2.
    if (stats_t.numel()) {
        EXEC_NPU_CMD(aclnnHiSparseResolveResidency,
                     empty, empty, empty, empty, empty, empty, empty,
                     errors,
                     zero, zero, zero, zero, zero, zero, zero, zero,
                     zero, zero, zero, zero, zero, zero, zero,
                     zero,
                     empty, empty, empty, empty, empty, empty, empty,
                     empty, empty, empty, empty, stats_t);
    }
#endif
}

}  // namespace vllm_ascend::hisparse

#endif  // HI_SPARSE_RESOLVE_RESIDENCY_TORCH_ADPT_H
