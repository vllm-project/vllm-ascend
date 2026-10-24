// Direct-output MLAPO has a deliberately narrow, validated W8A8 BF16 contract.
#pragma once
#include <torch/extension.h>
#include <cstdint>
#include <limits>

namespace vllm_ascend {
inline void CheckDirectMlapoTensor(const at::Tensor &tensor, const char *name,
                                  at::IntArrayRef shape, at::ScalarType dtype,
                                  const at::Device &device)
{
    TORCH_CHECK(tensor.device() == device && tensor.scalar_type() == dtype &&
                tensor.sizes() == shape && tensor.is_contiguous(),
                "mla_preprocess_without_kv_cache_write: ", name,
                " must be contiguous on ", device, " with dtype ", dtype,
                " and shape ", shape, "; got ", tensor.device(), ", ", tensor.scalar_type(),
                ", ", tensor.sizes());
}

inline void CheckDirectMlapoOptional(const c10::optional<at::Tensor> &tensor, const char *name,
                                    int64_t numel, at::ScalarType dtype, const at::Device &device)
{
    TORCH_CHECK(tensor.has_value(), "mla_preprocess_without_kv_cache_write: ", name, " is required");
    TORCH_CHECK(tensor->device() == device && tensor->scalar_type() == dtype &&
                tensor->numel() == numel && tensor->is_contiguous(),
                "mla_preprocess_without_kv_cache_write: ", name,
                " has invalid device, dtype, element count or strides");
}

inline void CheckDirectMlapoWriteback(
    const at::Tensor &hiddenState, const c10::optional<at::Tensor> &slots,
    int64_t k_base, int64_t rope_base, int64_t capacity)
{
    if (!slots.has_value()) {
        TORCH_CHECK(k_base == 0 && rope_base == 0 && capacity == 0,
                    "MLAPO writeback addresses/capacity require writeback_slots");
        return;
    }
    CheckDirectMlapoTensor(slots.value(), "writeback_slots", {hiddenState.size(0)},
                          at::kInt, hiddenState.device());
    TORCH_CHECK(k_base > 0 && rope_base > 0 && capacity > 0,
                "MLAPO writeback requires positive mapped CPU bases and capacity");
    TORCH_CHECK(k_base % 32 == 0 && rope_base % 32 == 0,
                "MLAPO writeback mapped CPU bases must be 32-byte aligned");
    constexpr int64_t k_bytes = 512 * 2;
    constexpr int64_t rope_bytes = 64 * 2;
    const int64_t address_limit = std::numeric_limits<int64_t>::max();
    TORCH_CHECK(capacity <= (address_limit - k_base) / k_bytes &&
                capacity <= (address_limit - rope_base) / rope_bytes,
                "MLAPO writeback mapped CPU address range overflows int64");
    TORCH_CHECK(k_base + capacity * k_bytes <= rope_base ||
                rope_base + capacity * rope_bytes <= k_base,
                "MLAPO writeback mapped CPU K and RoPE ranges must not overlap");
}

inline void CheckDirectMlapo(
    const at::Tensor &hiddenState, const at::Tensor &wdqkv,
    const c10::optional<at::Tensor> &descale0, const at::Tensor &gamma1,
    const c10::optional<at::Tensor> &beta1, const at::Tensor &wuq,
    const c10::optional<at::Tensor> &descale1, const at::Tensor &gamma2,
    const c10::optional<at::Tensor> &cos, const c10::optional<at::Tensor> &sin,
    const at::Tensor &wuk, const c10::optional<at::Tensor> &quant_scale0,
    const c10::optional<at::Tensor> &quant_offset0, const c10::optional<at::Tensor> &bias0,
    const c10::optional<at::Tensor> &quant_scale1, const c10::optional<at::Tensor> &quant_offset1,
    const c10::optional<at::Tensor> &bias1, c10::optional<c10::string_view> quant_mode,
    const at::Tensor &q_out0, const at::Tensor &q_out1, const at::Tensor &inner_out,
    const at::Tensor &current_k, const at::Tensor &current_rope)
{
    TORCH_CHECK(hiddenState.dim() == 2 && hiddenState.size(0) >= 1 && hiddenState.size(0) <= 1024,
                "mla_preprocess_without_kv_cache_write: hiddenState must be [N,6144], N in [1,1024]");
    const auto device = hiddenState.device();
    const int64_t n = hiddenState.size(0);
    TORCH_CHECK(wuk.dim() == 3 && wuk.size(0) > 0,
                "mla_preprocess_without_kv_cache_write: wuk must have a positive head dimension");
    const int64_t h = wuk.size(0);
    TORCH_CHECK(h <= std::numeric_limits<uint32_t>::max() / 512 &&
                h <= std::numeric_limits<uint32_t>::max() / n,
                "mla_preprocess_without_kv_cache_write: head count exceeds tiling dimensions");
    const int64_t q_width = h * 256;
    TORCH_CHECK(quant_mode.has_value() && quant_mode.value() == "per_tensor_quant_asymm",
                "mla_preprocess_without_kv_cache_write supports only per_tensor_quant_asymm");
    CheckDirectMlapoTensor(hiddenState, "hiddenState", {n, 6144}, at::kBFloat16, device);
    CheckDirectMlapoTensor(wdqkv, "wdqkv", {1, 192, 2624, 32}, at::kChar, device);
    CheckDirectMlapoTensor(wuq, "wuq", {1, 64, q_width, 32}, at::kChar, device);
    CheckDirectMlapoTensor(wuk, "wuk", {h, 192, 512}, at::kBFloat16, device);
    CheckDirectMlapoTensor(gamma1, "gamma1", {2048}, at::kBFloat16, device);
    CheckDirectMlapoTensor(gamma2, "gamma2", {512}, at::kBFloat16, device);
    CheckDirectMlapoOptional(beta1, "beta1", 2048, at::kBFloat16, device);
    CheckDirectMlapoOptional(descale0, "descale0", 2624, at::kFloat, device);
    CheckDirectMlapoOptional(descale1, "descale1", q_width, at::kFloat, device);
    CheckDirectMlapoOptional(quant_scale0, "quant_scale0", 1, at::kBFloat16, device);
    CheckDirectMlapoOptional(quant_scale1, "quant_scale1", 1, at::kBFloat16, device);
    CheckDirectMlapoOptional(quant_offset0, "quant_offset0", 1, at::kChar, device);
    CheckDirectMlapoOptional(quant_offset1, "quant_offset1", 1, at::kChar, device);
    CheckDirectMlapoOptional(bias0, "bias0", 2624, at::kInt, device);
    CheckDirectMlapoOptional(bias1, "bias1", q_width, at::kInt, device);
    TORCH_CHECK(cos.has_value() && sin.has_value(),
                "mla_preprocess_without_kv_cache_write requires cos and sin");
    CheckDirectMlapoTensor(cos.value(), "cos", {n, 64}, at::kBFloat16, device);
    CheckDirectMlapoTensor(sin.value(), "sin", {n, 64}, at::kBFloat16, device);
    CheckDirectMlapoTensor(q_out0, "q_out0", {n, h, 512}, at::kBFloat16, device);
    CheckDirectMlapoTensor(q_out1, "q_out1", {n, h, 64}, at::kBFloat16, device);
    CheckDirectMlapoTensor(inner_out, "inner_out", {n, 2048}, at::kBFloat16, device);
    CheckDirectMlapoTensor(current_k, "current_k", {n, 512}, at::kBFloat16, device);
    CheckDirectMlapoTensor(current_rope, "current_rope", {n, 64}, at::kBFloat16, device);
}
}  // namespace vllm_ascend
