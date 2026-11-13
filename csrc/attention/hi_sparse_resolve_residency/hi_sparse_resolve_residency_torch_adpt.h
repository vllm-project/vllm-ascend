// SPDX-License-Identifier: Apache-2.0
#pragma once
// Tensor metadata is decoded here; launches keep TorchNPU queue ordering.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <array>
#include <optional>

#include "op_kernel/resolver_scratch.h"

namespace vllm_ascend::hisparse {

// Argument slots of the resolver launch. The packed list mirrors the ACLNN
// wrapper layout: required inputs first, then caller-owned outputs in def
// registration order, with the optional valid_rows input pinned last so the
// EXEC_NPU_CMD below can reorder them into inputs / attributes / outputs.
enum ResolverSlot : size_t {
    kLogical = 0, kRequests, kStates, kSource, kResident, kHotTable,
    kTags, kLru, kPhysical, kAttention, kResolved, kCounts,
    kSwapHost, kSwapDevice, kSwapCounts, kMissMask, kStats, kErrors,
    kValidRows,
};

at::Tensor tensor_or_undefined(const std::optional<at::Tensor>& tensor) {
    return tensor.value_or(at::Tensor());
}

inline void validate_resolver_scratch(int64_t hot, int64_t union_rows) {
    TORCH_CHECK(union_rows > 0 && union_rows <= hot && hot <= 32768,
                "HiSparse union must fit the hot cache and int16 LRU slots");
    const uint64_t minimum_bytes = resolver_scratch_bytes(hot, union_rows + 1);
    TORCH_CHECK(minimum_bytes <= kResolverUbBytes,
                "HiSparse hot_size=", hot, ", max_union_rows=", union_rows,
                " needs at least ", minimum_bytes, " UB bytes; the compiled target allows ", kResolverUbBytes,
                ". Lower device_buffer_size or speculative tokens.");
}

void resolve_residency(
    const at::Tensor& logical,
    const std::optional<at::Tensor>& requests,
    const std::optional<at::Tensor>& states,
    const std::optional<at::Tensor>& source,
    const std::optional<at::Tensor>& resident,
    const at::Tensor& hot_table,
    const at::Tensor& dgi,
    const at::Tensor& lru,
    const at::Tensor& output,
    const std::optional<at::Tensor>& attention,
    const at::Tensor& resolved,
    const std::optional<at::Tensor>& counts,
    const std::optional<at::Tensor>& swap_host,
    const std::optional<at::Tensor>& swap_device,
    const std::optional<at::Tensor>& swap_counts,
    const std::optional<at::Tensor>& miss_mask,
    const std::optional<at::Tensor>& stats,
    const at::Tensor& errors,
    const std::optional<at::Tensor>& valid_rows,
    int64_t host_rows, int64_t hot_block_size, int64_t source_block_size,
    int64_t resident_block_size, int64_t resident_null_block,
    int64_t attention_block_stride, int64_t union_rows, int64_t attention_capacity) {
    TORCH_CHECK(logical.dim() == 2 && dgi.dim() == 2 && hot_table.dim() == 2,
                "HiSparse requires two-dimensional TopK, hot tags and block table");
    const auto rows = logical.size(0), topk = logical.size(1), hot = dgi.size(1);
    TORCH_CHECK(rows > 0 && topk > 0 && rows <= ((int64_t(1) << 30) - 1) / topk,
                "HiSparse first-reference positions exceed the int32 tag representation");
    TORCH_CHECK(topk <= hot, "HiSparse TopK must fit the hot cache");
    TORCH_CHECK(host_rows > 0 && host_rows < (int64_t(1) << 31), "Invalid HiSparse host row capacity");
    TORCH_CHECK(hot_block_size > 0 && attention_block_stride >= hot_block_size,
                "Invalid hot/attention block strides");
    TORCH_CHECK(!source || source_block_size > 0, "Invalid source block size");
    TORCH_CHECK(!resident || resident_block_size > 0, "Invalid resident block size");
    TORCH_CHECK(attention_capacity >= 0 && attention_capacity < (int64_t(1) << 31),
                "HiSparse compact Attention capacity must fit int32");
    TORCH_CHECK(attention_capacity == 0 || attention,
                "HiSparse compact Attention requires an output tensor");
    TORCH_CHECK(attention_capacity == 0 || !attention->is_alias_of(output),
                "HiSparse compact Attention must not overwrite virtual refill indices");
    TORCH_CHECK(!states || states->dim() == 1, "HiSparse state indices must be a vector");
    TORCH_CHECK(!source || source->dim() == 2, "HiSparse source table must be a matrix");
    TORCH_CHECK(!resident || resident->dim() == 2, "HiSparse resident table must be a matrix");
    TORCH_CHECK(lru.sizes() == dgi.sizes(), "HiSparse LRU/tag geometry mismatch");
    validate_resolver_scratch(hot, union_rows);

    const std::array<at::Tensor, 19> tensors = {
        logical, tensor_or_undefined(requests), tensor_or_undefined(states),
        tensor_or_undefined(source), tensor_or_undefined(resident), hot_table,
        dgi, lru, output, tensor_or_undefined(attention), resolved,
        tensor_or_undefined(counts), tensor_or_undefined(swap_host),
        tensor_or_undefined(swap_device), tensor_or_undefined(swap_counts),
        tensor_or_undefined(miss_mask), tensor_or_undefined(stats), errors,
        tensor_or_undefined(valid_rows)};
    c10_npu::NPUGuard guard(logical.device());
    auto npu_stream = c10_npu::getCurrentNPUStream();
    for (const ResolverSlot slot : {kLogical, kRequests, kStates, kSource, kResident, kHotTable,
                                    kTags, kLru, kPhysical, kAttention, kResolved, kCounts,
                                    kSwapHost, kSwapDevice, kSwapCounts, kMissMask, kStats,
                                    kErrors, kValidRows}) {
        const auto& tensor = tensors[slot];
        if (!tensor.defined()) continue;
        TORCH_CHECK(tensor.device() == logical.device(), "HiSparse resolver device mismatch");
        const at::ScalarType expected =
            slot == kLru ? at::kShort : (slot == kStats ? at::kLong : at::kInt);
        if (slot == kMissMask) {
            TORCH_CHECK(tensor.scalar_type() == at::kInt || tensor.scalar_type() == at::kByte ||
                            tensor.scalar_type() == at::kBool, "HiSparse miss mask must be int32, uint8 or bool");
        } else {
            TORCH_CHECK(tensor.scalar_type() == expected, "HiSparse resolver dtype mismatch at ", slot);
        }
        if (tensor.dim() == 2) {
            TORCH_CHECK(tensor.stride(1) == 1 && tensor.stride(0) >= tensor.size(1),
                        "HiSparse resolver matrix rows must be dense and non-overlapping");
        }
        const bool dense_topk = slot == kPhysical || slot == kResolved ||
                                slot == kSwapHost || slot == kSwapDevice;
        if (slot == kStates || slot == kTags || slot == kLru) {
            TORCH_CHECK(tensor.is_contiguous(), "HiSparse state/LRU/tag storage must be contiguous");
        }
        if (dense_topk) {
            TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) == rows && tensor.size(1) == topk && tensor.is_contiguous(),
                        "HiSparse resolver requires contiguous TopK-shaped output at ", slot);
        }
        if (slot == kAttention) {
            TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) == rows && tensor.size(1) == topk,
                        "Invalid HiSparse attention index shape");
        }
        if (slot == kRequests || slot == kSwapCounts) {
            TORCH_CHECK(tensor.dim() == 1 && tensor.numel() == rows && tensor.is_contiguous(),
                        "Invalid HiSparse per-row vector at ", slot);
        }
        if (slot == kCounts) {
            TORCH_CHECK((tensor.dim() == 1 || (tensor.dim() == 2 && tensor.size(1) == 1)) && tensor.size(0) == rows,
                        "Invalid HiSparse valid-count shape");
            TORCH_CHECK(tensor.stride(0) > 0, "HiSparse valid-count rows must not overlap");
        }
        if (slot == kMissMask) {
            TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) == rows && tensor.size(1) == topk && tensor.is_contiguous(),
                        "Invalid HiSparse miss-mask shape");
        }
        if (slot == kStats) TORCH_CHECK(tensor.numel() == 2 && tensor.is_contiguous(), "Invalid HiSparse stats shape");
        if (slot == kErrors) {
            TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) == rows && tensor.size(1) == 8 && tensor.is_contiguous(),
                        "Invalid HiSparse error scratch shape");
        }
        if (slot == kValidRows) TORCH_CHECK(tensor.numel() == 1, "Invalid HiSparse valid-row scalar");
        // Register the use before dispatch so allocator reuse follows queued
        // work on the resolver stream, even if the caller releases a view.
        c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), npu_stream);
    }
    TORCH_CHECK(bool(swap_host) == bool(swap_device) && bool(swap_host) == bool(swap_counts),
                "All compact HiSparse swap outputs are required together");
    TORCH_CHECK(hot_table.size(1) >= (hot + hot_block_size - 1) / hot_block_size,
                "HiSparse hot block table is too narrow");
    TORCH_CHECK(hot_table.size(0) >= (states ? states->numel() : dgi.size(0)),
                "HiSparse hot block table does not cover request slots");
    const int64_t logical_stride = logical.stride(0);
    const int64_t source_stride = source ? source->stride(0) : 0;
    const int64_t resident_stride = resident ? resident->stride(0) : 0;
    const int64_t hot_table_stride = hot_table.stride(0);
    const int64_t attention_stride = attention ? attention->stride(0) : topk;
    const int64_t counts_stride = counts ? counts->stride(0) : 1;
    const int64_t byte_miss_mask = miss_mask && miss_mask->element_size() == 1;
    // Generated ACLNN takes inputs, attributes, then caller-owned outputs.
    EXEC_NPU_CMD(
        aclnnHiSparseResolveResidency,
        tensors[kLogical], tensors[kRequests], tensors[kStates], tensors[kSource],
        tensors[kResident], tensors[kHotTable], tensors[kValidRows],
        host_rows, hot_block_size, source_block_size, resident_block_size, resident_null_block,
        attention_block_stride, union_rows, attention_capacity, logical_stride, source_stride,
        resident_stride, hot_table_stride, attention_stride, counts_stride, byte_miss_mask,
        tensors[kTags], tensors[kLru], tensors[kPhysical], tensors[kAttention],
        tensors[kResolved], tensors[kCounts], tensors[kSwapHost], tensors[kSwapDevice],
        tensors[kSwapCounts], tensors[kMissMask], tensors[kStats], tensors[kErrors]);
}

// The operator mutates caller-owned tensors and returns no new tensors.
void resolve_residency_meta(
    const at::Tensor&, const std::optional<at::Tensor>&, const std::optional<at::Tensor>&,
    const std::optional<at::Tensor>&, const std::optional<at::Tensor>&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const std::optional<at::Tensor>&,
    const at::Tensor&, const std::optional<at::Tensor>&, const std::optional<at::Tensor>&,
    const std::optional<at::Tensor>&, const std::optional<at::Tensor>&, const std::optional<at::Tensor>&,
    const std::optional<at::Tensor>&, const at::Tensor&, const std::optional<at::Tensor>&,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t) {}
}  // namespace vllm_ascend::hisparse
