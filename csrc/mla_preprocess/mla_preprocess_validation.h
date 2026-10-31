// Shared MLAPO configuration and Direct output/writeback checks.
#pragma once
#include <torch/extension.h>
#include <cstdint>
#include <limits>

namespace vllm_ascend {
namespace mla_preprocess_detail {
constexpr int64_t FULLY_SUPPORTED_KV_LORA_RANK = 512;
constexpr int64_t FULLY_SUPPORTED_QK_ROPE_HEAD_DIM = 64;

inline int64_t ParseQuantMode(c10::optional<c10::string_view> quantMode)
{
    const c10::string_view mode = quantMode.value_or("per_token_quant_symm");
    if (mode == "per_tensor_quant_asymm") {
        return 0;
    }
    if (mode == "per_token_quant_symm") {
        return 1;
    }
    if (mode == "per_token_quant_asymm") {
        return 2;
    }
    if (mode == "no_quant") {
        return 3;
    }
    TORCH_CHECK(false, "Unsupported quant_mode value: '", mode, "'");
    return 0;
}

inline void CheckRopePair(const c10::optional<at::Tensor> &cos,
                          const c10::optional<at::Tensor> &sin)
{
    TORCH_CHECK(cos.has_value() == sin.has_value(),
                "mla_preprocess requires cos and sin to both be tensors or both be None.");
}

inline void WarnUnsupportedDimensions(int64_t kvLoraRank, int64_t qkRopeHeadDim)
{
    if (kvLoraRank != FULLY_SUPPORTED_KV_LORA_RANK ||
        qkRopeHeadDim != FULLY_SUPPORTED_QK_ROPE_HEAD_DIM) {
        TORCH_WARN_ONCE(
            "mla_preprocess currently fully supports only kv_lora_rank=",
            FULLY_SUPPORTED_KV_LORA_RANK,
            " and qk_rope_head_dim=", FULLY_SUPPORTED_QK_ROPE_HEAD_DIM,
            ", but received kv_lora_rank=", kvLoraRank,
            " and qk_rope_head_dim=", qkRopeHeadDim,
            ". Inputs outside the fully supported configuration are allowed to continue, "
            "but may produce accuracy issues.");
    }
}
}  // namespace mla_preprocess_detail

inline void CheckDirectMlapoTensor(const at::Tensor &tensor, const char *name,
                                   at::IntArrayRef shape, at::ScalarType dtype,
                                   const at::Device &device)
{
    TORCH_CHECK(tensor.device() == device && tensor.scalar_type() == dtype &&
                tensor.sizes() == shape && tensor.is_contiguous(),
                "mla_preprocess_with_kv_offload: ", name,
                " must be contiguous on ", device, " with dtype ", dtype,
                " and shape ", shape, "; got ", tensor.device(), ", ", tensor.scalar_type(),
                ", ", tensor.sizes());
}

inline void CheckDirectMlapoWriteback(
    const at::Tensor &hiddenState, const at::Tensor &current_k, const at::Tensor &current_rope,
    const c10::optional<at::Tensor> &slots, int64_t k_base, int64_t rope_base, int64_t capacity)
{
    if (k_base == 0 && rope_base == 0 && capacity == 0) {
        return;
    }
    TORCH_CHECK(slots.has_value(), "MLAPO writeback addresses/capacity require writeback_slots");
    CheckDirectMlapoTensor(slots.value(), "writeback_slots", {hiddenState.size(0)},
                           at::kInt, hiddenState.device());
    TORCH_CHECK(k_base > 0 && rope_base > 0 && capacity > 0,
                "MLAPO writeback requires positive mapped CPU bases and capacity");
    TORCH_CHECK(k_base % 32 == 0 && rope_base % 32 == 0,
                "MLAPO writeback mapped CPU bases must be 32-byte aligned");
    const int64_t address_limit = std::numeric_limits<int64_t>::max();
    TORCH_CHECK(current_k.size(1) <= address_limit / current_k.element_size() &&
                current_rope.size(1) <= address_limit / current_rope.element_size(),
                "MLAPO writeback row byte count overflows int64");
    const int64_t k_bytes = current_k.size(1) * current_k.element_size();
    const int64_t rope_bytes = current_rope.size(1) * current_rope.element_size();
    TORCH_CHECK(capacity <= (address_limit - k_base) / k_bytes &&
                capacity <= (address_limit - rope_base) / rope_bytes,
                "MLAPO writeback mapped CPU address range overflows int64");
    TORCH_CHECK(k_base + capacity * k_bytes <= rope_base ||
                rope_base + capacity * rope_bytes <= k_base,
                "MLAPO writeback mapped CPU K and RoPE ranges must not overlap");
}

inline int64_t CheckDirectMlapo(
    const at::Tensor &hiddenState, const at::Tensor &gamma1,
    const c10::optional<at::Tensor> &cos, const c10::optional<at::Tensor> &sin,
    const at::Tensor &wuk, c10::optional<c10::string_view> quant_mode,
    const at::Tensor &q_out0, const at::Tensor &q_out1, const at::Tensor &inner_out,
    const at::Tensor &current_k, const at::Tensor &current_rope)
{
    // Common compute inputs retain the original ACLNN/Host checks. Only the
    // new contiguous outputs and CPU destinations need Direct-specific checks.
    mla_preprocess_detail::CheckRopePair(cos, sin);
    TORCH_CHECK(hiddenState.dim() == 2, "mla_preprocess_with_kv_offload: hiddenState must be [N,D]");
    const auto device = hiddenState.device();
    const auto dtype = hiddenState.scalar_type();
    TORCH_CHECK(wuk.dim() == 3 && gamma1.dim() >= 1 && current_rope.dim() == 2,
                "mla_preprocess requires rank-3 wuk, non-scalar gamma1 and rank-2 current_rope");
    const int64_t n = hiddenState.size(0);
    const int64_t h = wuk.size(0);
    const int64_t kv_rank = wuk.size(2);
    const int64_t q_rank = gamma1.size(0);
    const int64_t rope_dim = current_rope.size(1);
    TORCH_CHECK(h > 0 && kv_rank > 0 && q_rank > 0 && rope_dim > 0,
                "MLAPO output dimensions must be positive");
    const int64_t quantMode = mla_preprocess_detail::ParseQuantMode(quant_mode);
    mla_preprocess_detail::WarnUnsupportedDimensions(kv_rank, rope_dim);
    CheckDirectMlapoTensor(q_out0, "q_out0", {n, h, kv_rank}, dtype, device);
    CheckDirectMlapoTensor(q_out1, "q_out1", {n, h, rope_dim}, dtype, device);
    CheckDirectMlapoTensor(inner_out, "inner_out", {n, q_rank}, dtype, device);
    CheckDirectMlapoTensor(current_k, "current_k", {n, kv_rank}, dtype, device);
    CheckDirectMlapoTensor(current_rope, "current_rope", {n, rope_dim}, dtype, device);
    return quantMode;
}
}  // namespace vllm_ascend
