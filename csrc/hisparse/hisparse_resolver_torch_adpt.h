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
#include <torch_npu/csrc/framework/OpCommand.h>
#include <tiling/platform/platform_ascendc.h>
#include <algorithm>
#include <vector>
#include <utility>

extern "C" void launch_hisparse_resolve_residency(void**, uint64_t*, int, void*);
extern "C" void launch_hisparse_invalidate_written(uint64_t, uint64_t, uint64_t, uint64_t,
    int, int, int, int, int, bool, int, void*);


namespace vllm_ascend::hisparse {
void invalidate_written(const at::Tensor& groups, const at::Tensor& states,
    const at::Tensor& requests, const at::Tensor& written, int64_t state_count, int64_t region) {
    TORCH_CHECK(groups.scalar_type() == at::kLong && states.scalar_type() == at::kInt &&
                requests.scalar_type() == at::kInt &&
                (written.scalar_type() == at::kInt || written.scalar_type() == at::kLong),
                "Invalid HiSparse invalidation dtypes");
    TORCH_CHECK(requests.numel() == written.numel() && written.numel() <= 64,
                "HiSparse batched invalidation supports at most 64 writes");
    TORCH_CHECK(state_count >= 0 && state_count <= INT32_MAX && region >= 0 && region <= 32768 &&
                states.numel() <= INT32_MAX && groups.numel() <= INT32_MAX / 16,
                "Invalid HiSparse invalidation geometry");
    std::array<at::Tensor, 4> tensors = {groups, states, requests, written};
    for (const auto& tensor : tensors) {
        TORCH_CHECK(tensor.device() == groups.device() && tensor.dim() == 1 && tensor.is_contiguous(),
                    "HiSparse invalidation requires contiguous vectors on one NPU");
    }
    if (!groups.numel() || !states.numel() || !written.numel() || !state_count || !region) return;
    c10_npu::NPUGuard guard(groups.device());
    auto npu_stream = c10_npu::getCurrentNPUStream();
    std::array<uint64_t, 4> pointers{};
    std::vector<at::Tensor> owners;
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto& tensor = tensors[i];
        pointers[i] = reinterpret_cast<uintptr_t>(tensor.data_ptr());
        owners.emplace_back(tensor.detach());
        c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), npu_stream);
    }
    const int group_count = groups.numel(), request_count = states.numel(), tokens = written.numel();
    const bool written64 = written.scalar_type() == at::kLong;
    int64_t vector_cores = 0;
    TORCH_CHECK(aclGetDeviceCapability(groups.device().index(), ACL_DEVICE_INFO_VECTOR_CORE_NUM,
                                     &vector_cores) == ACL_SUCCESS && vector_cores > 0,
                "HiSparse failed to query the vector core count");
    const int blocks = std::min<int64_t>(group_count * ((region + 2047) / 2048), vector_cores);
    auto stream = npu_stream.stream();
    at_npu::native::OpCommand command;
    command.Name("HiSparseInvalidateWritten");
    command.SetCustomHandler([=, owners = std::move(owners)]() mutable -> int {
        auto submission_owners = std::move(owners);
        launch_hisparse_invalidate_written(pointers[0], pointers[1], pointers[2], pointers[3],
            group_count, request_count, state_count, tokens, region, written64, blocks, stream);
        return 0;
    });
    command.Run();
}

void invalidate_written_meta(const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, int64_t, int64_t) {}

inline int64_t resolver_hash_size(int64_t hot, int64_t union_rows) {
    TORCH_CHECK(union_rows > 0 && union_rows <= hot && hot <= 32768,
                "HiSparse union must fit the hot cache and int16 LRU slots");
    int64_t hash_size = 1;
    while (hash_size <= union_rows) hash_size <<= 1;
    static const uint64_t budget = [] {
        auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance(aclrtGetSocName());
        TORCH_CHECK(platform != nullptr, "HiSparse failed to query the target UB capacity");
        uint64_t ub_bytes = 0;
        platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_bytes);
        constexpr uint64_t reserved_bytes = 16 * 1024;
        TORCH_CHECK(ub_bytes > reserved_bytes, "HiSparse target UB capacity is too small");
        return ub_bytes - reserved_bytes;
    }();
    // The arch22 kernel retains the complete Hot tag row in UB; scalar hash updates never
    // write GM until the complete union has passed the capacity check.
    const uint64_t fixed_words = 2 * hot + 512;
    const uint64_t minimum_bytes = (2 * (union_rows + 1) + fixed_words + 15) / 16 * 64 + 512;
    TORCH_CHECK(minimum_bytes <= budget,
                "HiSparse hot_size=", hot, ", max_union_rows=", union_rows,
                " needs at least ", minimum_bytes, " UB bytes; the compiled target allows ", budget,
                ". Lower device_buffer_size or speculative tokens.");
    const uint64_t available_words = (budget - 512) / 64 * 16;
    return std::min(hash_size, int64_t((available_words - fixed_words) / 2));
}

at::Tensor tensor_or_undefined(const std::optional<at::Tensor>& tensor) {
    return tensor.value_or(at::Tensor());
}

inline void validate_resolver_scratch_arch22(int64_t hot, int64_t union_rows) {
    (void)resolver_hash_size(hot, union_rows);
}

inline void resolve_residency_arch22(
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
    validate_resolver_scratch_arch22(hot, union_rows);

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
    std::array<void*, 19> pointers{};
    std::vector<at::Tensor> owners;
    owners.reserve(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto& tensor = tensors[i];
        if (!tensor.defined()) continue;
        owners.emplace_back(tensor.detach());
        pointers[i] = tensor.numel() ? tensor.data_ptr() : nullptr;
        TORCH_CHECK(tensor.device() == logical.device(), "HiSparse resolver device mismatch");
        const auto dtype = i == 7 ? at::kShort : (i == 16 ? at::kLong : at::kInt);
        if (i == 15) {
            TORCH_CHECK(tensor.scalar_type() == at::kInt || tensor.scalar_type() == at::kByte ||
                            tensor.scalar_type() == at::kBool, "HiSparse miss mask must be int32, uint8 or bool");
        } else {
            TORCH_CHECK(tensor.scalar_type() == dtype, "HiSparse resolver dtype mismatch at ", i);
        }
        if (tensor.dim() == 2) {
            TORCH_CHECK(tensor.stride(1) == 1 && tensor.stride(0) >= tensor.size(1),
                        "HiSparse resolver matrix rows must be dense and non-overlapping");
        }
        const bool dense_topk = i == 8 || i == 10 || i == 12 || i == 13;
        if (i == 2 || i == 6 || i == 7) {
            TORCH_CHECK(tensor.is_contiguous(), "HiSparse state/LRU/tag storage must be contiguous");
        }
        if (dense_topk) {
            TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) == rows && tensor.size(1) == topk && tensor.is_contiguous(),
                        "HiSparse resolver requires contiguous TopK-shaped output at ", i);
        }
        if (i == 9) {
            TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) == rows && tensor.size(1) == topk,
                        "Invalid HiSparse attention index shape");
        }
        if (i == 1 || i == 14) {
            TORCH_CHECK(tensor.dim() == 1 && tensor.numel() == rows && tensor.is_contiguous(),
                        "Invalid HiSparse per-row vector at ", i);
        }
        if (i == 11) {
            TORCH_CHECK((tensor.dim() == 1 || (tensor.dim() == 2 && tensor.size(1) == 1)) && tensor.size(0) == rows,
                        "Invalid HiSparse valid-count shape");
            TORCH_CHECK(tensor.stride(0) > 0, "HiSparse valid-count rows must not overlap");
        }
        if (i == 15) {
            TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) == rows && tensor.size(1) == topk && tensor.is_contiguous(),
                        "Invalid HiSparse miss-mask shape");
        }
        if (i == 16) TORCH_CHECK(tensor.numel() == 2 && tensor.is_contiguous(), "Invalid HiSparse stats shape");
        if (i == 17) {
            TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) == rows && tensor.size(1) == 8 && tensor.is_contiguous(),
                        "Invalid HiSparse error scratch shape");
        }
        if (i == 18) TORCH_CHECK(tensor.numel() == 1, "Invalid HiSparse valid-row scalar");
        // Register the use before dispatch so allocator reuse follows queued
        // work on the resolver stream, even if the caller releases a view.
        if (tensor.numel()) {
            c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), npu_stream);
        }
    }
    TORCH_CHECK(bool(swap_host) == bool(swap_device) && bool(swap_host) == bool(swap_counts),
                "All compact HiSparse swap outputs are required together");
    TORCH_CHECK(hot_table.size(1) >= (hot + hot_block_size - 1) / hot_block_size,
                "HiSparse hot block table is too narrow");
    TORCH_CHECK(hot_table.size(0) >= (states ? states->numel() : dgi.size(0)),
                "HiSparse hot block table does not cover request slots");
    std::array<uint64_t, 25> sizes = {
        uint64_t(rows), uint64_t(topk), uint64_t(hot), uint64_t(host_rows), uint64_t(logical.stride(0)),
        uint64_t(states ? states->numel() : 0), uint64_t(dgi.size(0)),
        uint64_t(source ? source->size(0) : 0), uint64_t(source ? source->size(1) : 0),
        uint64_t(source ? source->stride(0) : 0), uint64_t(source_block_size),
        uint64_t(resident ? resident->size(0) : 0), uint64_t(resident ? resident->size(1) : 0),
        uint64_t(resident ? resident->stride(0) : 0), uint64_t(resident_block_size), uint64_t(resident_null_block),
        uint64_t(hot_block_size), uint64_t(hot_table.stride(0)),
        uint64_t(attention ? attention->stride(0) : topk), uint64_t(attention_block_stride),
        uint64_t(counts ? counts->stride(0) : 1), uint64_t(union_rows), uint64_t(resolver_hash_size(hot, union_rows)),
        uint64_t(attention_capacity), uint64_t(miss_mask && miss_mask->element_size() == 1)};
    auto stream = npu_stream.stream();
    at_npu::native::OpCommand command;
    command.Name("HiSparseUnionResolve");
    command.SetCustomHandler([pointers, sizes, stream, owners = std::move(owners)]() mutable -> int {
        // Release queued tensor owners after submission, not on queue-slot reuse.
        auto submission_owners = std::move(owners);
        launch_hisparse_resolve_residency(pointers.data(), sizes.data(), int(sizes[0]), stream);
        return 0;
    });
    command.Run();
}

}  // namespace vllm_ascend::hisparse
