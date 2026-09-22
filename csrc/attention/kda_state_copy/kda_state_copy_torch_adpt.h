// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace vllm_ascend {
inline void kda_state_copy_meta(const at::Tensor &state, const at::Tensor &packed,
    const at::Tensor &indices, const c10::optional<at::Tensor> &has_initial_state, bool to_cache)
{
    TORCH_CHECK(state.dim() == 4 && state.size(0) > 0 && packed.dim() == 4 &&
        packed.size(0) == indices.numel() && packed.size(1) == state.size(1) &&
        packed.size(2) == state.size(2) && packed.size(3) == state.size(3),
        "KdaStateCopy requires matching cache[N,H,V,K] and packed[selected,H,V,K]");
    TORCH_CHECK((state.scalar_type() == at::kFloat || state.scalar_type() == at::kBFloat16) &&
        packed.scalar_type() == state.scalar_type() && packed.is_contiguous(),
        "KdaStateCopy requires matching FP32/BF16 and a contiguous packed buffer");
    int64_t payload = 1;
    for (int dim = 3; dim >= 1; --dim) {
        TORCH_CHECK(state.size(dim) > 0 && (state.size(dim) == 1 || state.stride(dim) == payload),
            "KdaStateCopy supports dense inner [H,V,K] matrices");
        payload *= state.size(dim);
    }
    TORCH_CHECK(state.stride(0) >= payload, "KdaStateCopy cache pages must not overlap");
    TORCH_CHECK(indices.dim() == 1 && (indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong) &&
        indices.device() == state.device() && packed.device() == state.device(),
        "KdaStateCopy indices must be same-device INT32/INT64 vectors");
    if (has_initial_state.has_value()) {
        const auto &flags = *has_initial_state;
        TORCH_CHECK(flags.dim() == 1 && flags.numel() == indices.numel() && flags.scalar_type() == at::kBool &&
            flags.device() == state.device(), "KdaStateCopy initial-state flags must be matching same-device BOOL vectors");
    }
}
inline void kda_state_copy(const at::Tensor &state, const at::Tensor &packed, const at::Tensor &indices,
    const c10::optional<at::Tensor> &has_initial_state, bool to_cache)
{
    kda_state_copy_meta(state, packed, indices, has_initial_state, to_cache);
    TORCH_CHECK(state.device().type() == c10::DeviceType::PrivateUse1, "KdaStateCopy requires NPU tensors");
    if (indices.numel() == 0) return;
    const int64_t payloadElements = state.size(1) * state.size(2) * state.size(3);
    // Tensor data pointers already include the layer storage offset. A flat
    // physical view keeps page gaps addressable without copying the cache.
    const int64_t span = (state.size(0) - 1) * state.stride(0) + payloadElements;
    auto cacheStorage = state.as_strided({span}, {1});
    auto packedStorage = packed.view({packed.numel()});
    const auto &source = to_cache ? packedStorage : cacheStorage;
    const auto &destination = to_cache ? cacheStorage : packedStorage;
    const int64_t cacheRows = state.size(0), selectedRows = indices.numel();
    const int64_t cacheStrideBytes = state.stride(0) * state.element_size();
    const int64_t payloadBytes = payloadElements * state.element_size();
    EXEC_NPU_CMD(aclnnInnerKdaStateCopy, source, indices, has_initial_state,
        cacheRows, selectedRows, cacheStrideBytes, payloadBytes, to_cache, destination);
}
}
