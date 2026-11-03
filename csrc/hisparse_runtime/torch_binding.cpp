// SPDX-License-Identifier: Apache-2.0
// Tensor metadata is decoded here; launches keep TorchNPU queue ordering.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <c10/core/impl/PyObjectSlot.h>
#include <array>
#include <optional>
#include <vector>
#include <utility>

extern "C" void launch_hisparse_union_resolve60103(void**, uint64_t*, int, void*);
extern "C" uint64_t hisparse_resolver_scratch_capacity();

namespace {
at::Tensor tensor_or_undefined(const std::optional<at::Tensor>& tensor) {
    return tensor.value_or(at::Tensor());
}

void resolve_tensors(
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
    int64_t attention_block_stride, int64_t union_rows) {
    TORCH_CHECK(logical.dim() == 2 && dgi.dim() == 2 && hot_table.dim() == 2,
                "HiSparse requires two-dimensional TopK, hot tags and block table");
    const auto rows = logical.size(0), topk = logical.size(1), hot = dgi.size(1);
    TORCH_CHECK(rows > 0 && topk > 0 && rows <= ((int64_t(1) << 30) - 1) / topk,
                "HiSparse first-reference positions exceed the int32 tag representation");
    TORCH_CHECK(union_rows > 0 && union_rows <= hot && hot <= 32768 && topk <= hot,
                "HiSparse union must fit the hot cache and int16 LRU slots");
    TORCH_CHECK(host_rows > 0 && host_rows < (int64_t(1) << 31), "Invalid HiSparse host row capacity");
    TORCH_CHECK(hot_block_size > 0 && attention_block_stride >= hot_block_size,
                "Invalid hot/attention block strides");
    TORCH_CHECK(!source || source_block_size > 0, "Invalid source block size");
    TORCH_CHECK(!resident || resident_block_size > 0, "Invalid resident block size");
    TORCH_CHECK(!states || states->dim() == 1, "HiSparse state indices must be a vector");
    TORCH_CHECK(!source || source->dim() == 2, "HiSparse source table must be a matrix");
    TORCH_CHECK(!resident || resident->dim() == 2, "HiSparse resident table must be a matrix");
    TORCH_CHECK(lru.sizes() == dgi.sizes(), "HiSparse LRU/tag geometry mismatch");
    int64_t hash_size = 1;
    while (hash_size <= union_rows) hash_size <<= 1;
    const uint64_t tiles = (hot + 31) / 32;
    const uint64_t scratch_ints = (2 * hash_size + hot + 2 * tiles + 64 + 15) / 16 * 16;
    TORCH_CHECK(scratch_ints * sizeof(int32_t) + 512 <= hisparse_resolver_scratch_capacity(),
                "HiSparse resolver scratch exceeds the compiled target's UB budget");

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
        TORCH_CHECK(tensor.device() == logical.device(), "HiSparse resolver device mismatch");
        const auto dtype = i == 7 ? at::kShort : (i == 16 ? at::kLong : at::kInt);
        TORCH_CHECK(tensor.scalar_type() == dtype, "HiSparse resolver dtype mismatch at ", i);
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
        // Separate TensorImpls retain storage until queue submission without
        // Python-object destruction on a queue/release thread.
        owners.emplace_back(tensor.detach());
        TORCH_CHECK(owners.back().unsafeGetTensorImpl()->pyobj_slot()->load_pyobj() == nullptr &&
                        owners.back().unsafeGetTensorImpl() != tensor.unsafeGetTensorImpl(),
                    "HiSparse queue owner must have an independent, Python-free TensorImpl");
        // Queue callback completion is not device completion. The allocator
        // must keep every storage unavailable for reuse until this stream's
        // work completes, including when graph capture is active.
        c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), npu_stream);
        pointers[i] = tensor.numel() ? tensor.data_ptr() : nullptr;
    }
    TORCH_CHECK(bool(swap_host) == bool(swap_device) && bool(swap_host) == bool(swap_counts),
                "All compact HiSparse swap outputs are required together");
    TORCH_CHECK(hot_table.size(1) >= (hot + hot_block_size - 1) / hot_block_size,
                "HiSparse hot block table is too narrow");
    TORCH_CHECK(hot_table.size(0) >= (states ? states->numel() : dgi.size(0)),
                "HiSparse hot block table does not cover request slots");
    std::array<uint64_t, 23> sizes = {
        uint64_t(rows), uint64_t(topk), uint64_t(hot), uint64_t(host_rows), uint64_t(logical.stride(0)),
        uint64_t(states ? states->numel() : 0), uint64_t(dgi.size(0)),
        uint64_t(source ? source->size(0) : 0), uint64_t(source ? source->size(1) : 0),
        uint64_t(source ? source->stride(0) : 0), uint64_t(source_block_size),
        uint64_t(resident ? resident->size(0) : 0), uint64_t(resident ? resident->size(1) : 0),
        uint64_t(resident ? resident->stride(0) : 0), uint64_t(resident_block_size), uint64_t(resident_null_block),
        uint64_t(hot_block_size), uint64_t(hot_table.stride(0)),
        uint64_t(attention ? attention->stride(0) : topk), uint64_t(attention_block_stride),
        uint64_t(counts ? counts->stride(0) : 1), uint64_t(union_rows), uint64_t(hash_size)};
    auto stream = npu_stream.stream();
    at_npu::native::OpCommand command;
    command.Name("HiSparseUnionResolve");
    command.SetCustomHandler([pointers, sizes, stream, owners = std::move(owners)]() mutable -> int {
        // Release queued tensor owners after submission, not on queue-slot reuse.
        auto submission_owners = std::move(owners);
        launch_hisparse_union_resolve60103(pointers.data(), sizes.data(), int(sizes[0]), stream);
        return 0;
    });
    command.Run();
}

// The operator mutates caller-owned tensors and returns no new tensors.
void resolve_meta(
    const at::Tensor&, const std::optional<at::Tensor>&, const std::optional<at::Tensor>&,
    const std::optional<at::Tensor>&, const std::optional<at::Tensor>&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const std::optional<at::Tensor>&,
    const at::Tensor&, const std::optional<at::Tensor>&, const std::optional<at::Tensor>&,
    const std::optional<at::Tensor>&, const std::optional<at::Tensor>&, const std::optional<at::Tensor>&,
    const std::optional<at::Tensor>&, const at::Tensor&, const std::optional<at::Tensor>&,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t) {}
}  // namespace

TORCH_LIBRARY(hisparse_fused60103, m) {
    m.def("resolve_tensors(Tensor logical, Tensor? requests, Tensor? states, Tensor? source, Tensor? resident, "
          "Tensor hot_table, Tensor(a!) dgi, Tensor(a!) lru, Tensor(a!) output, Tensor(a!)? attention, "
          "Tensor(a!) resolved, Tensor(a!)? counts, Tensor(a!)? swap_host, Tensor(a!)? swap_device, "
          "Tensor(a!)? swap_counts, Tensor(a!)? miss_mask, Tensor(a!)? stats, Tensor(a!) errors, Tensor? valid_rows, "
          "int host_rows, int hot_block_size, int source_block_size, int resident_block_size, int resident_null_block, "
          "int attention_block_stride, int union_rows) -> ()");
    m.def("scratch_capacity() -> int", []() -> int64_t { return hisparse_resolver_scratch_capacity(); });
}
TORCH_LIBRARY_IMPL(hisparse_fused60103, PrivateUse1, m) { m.impl("resolve_tensors", TORCH_FN(resolve_tensors)); }
TORCH_LIBRARY_IMPL(hisparse_fused60103, Meta, m) { m.impl("resolve_tensors", TORCH_FN(resolve_meta)); }
