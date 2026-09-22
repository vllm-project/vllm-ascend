// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cmath>
#include <limits>

namespace vllm_ascend {
inline at::Tensor kda_rms_norm_gated_meta(const at::Tensor &x, const at::Tensor &gate,
    const at::Tensor &weight, at::Tensor &out, double eps, bool sigmoidOnly)
{
    (void)sigmoidOnly;
    const auto valid_layout = [](const at::Tensor &tensor) {
        return (tensor.dim() == 3 || (tensor.dim() == 4 && tensor.size(0) == 1)) &&
            tensor.size(-1) == 128 && tensor.stride(-1) == 1 && tensor.stride(-2) == 128 &&
            tensor.stride(-3) >= tensor.size(-2) * 128;
    };
    TORCH_CHECK(valid_layout(x) && valid_layout(gate) &&
        x.scalar_type() == at::kBFloat16 && gate.scalar_type() == at::kBFloat16 &&
        x.size(-3) == gate.size(-3) && x.size(-2) == gate.size(-2) &&
        x.size(-2) > 0 && x.size(-2) <= 128,
        "KdaRmsNormGated requires BF16 [T,H,128] or [1,T,H,128] with dense heads");
    TORCH_CHECK(weight.dim() == 1 && weight.numel() == 128 && weight.is_contiguous() &&
        (weight.scalar_type() == at::kFloat || weight.scalar_type() == at::kBFloat16),
        "KdaRmsNormGated requires one contiguous BF16/FP32 [128] weight");
    TORCH_CHECK(gate.device() == x.device() && weight.device() == x.device(),
        "KdaRmsNormGated tensors must share one device");
    TORCH_CHECK(out.sizes() == x.sizes() && out.scalar_type() == x.scalar_type() &&
        out.device() == x.device() && out.is_contiguous(),
        "KdaRmsNormGated output must match x and be contiguous");
    TORCH_CHECK(std::isfinite(eps) && eps > 0 && std::isfinite(static_cast<float>(eps)) &&
        static_cast<float>(eps) > 0, "KdaRmsNormGated epsilon must be positive finite FP32");
    for (const auto *tensor : {&x, &gate}) {
        TORCH_CHECK(tensor->stride(-3) - tensor->size(-2) * 128 <=
            std::numeric_limits<uint32_t>::max() / 2,
            "KdaRmsNormGated token DMA stride exceeds uint32 bytes");
    }
    return out;
}

inline at::Tensor kda_rms_norm_gated(const at::Tensor &x, const at::Tensor &gate,
    const at::Tensor &weight, at::Tensor &out, double eps, bool sigmoidOnly)
{
    kda_rms_norm_gated_meta(x, gate, weight, out, eps, sigmoidOnly);
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1 &&
        gate.device() == x.device() && weight.device() == x.device(),
        "KdaRmsNormGated tensors must share one NPU");
    if (x.numel() == 0) return out;
    const int64_t tokens = x.size(-3), heads = x.size(-2);
    const int64_t x_token_stride = x.stride(-3), gate_token_stride = gate.stride(-3);
    EXEC_NPU_CMD(aclnnInnerKdaRmsNormGated, x, gate, weight,
        tokens, heads, x_token_stride, gate_token_stride, eps, sigmoidOnly, out);
    return out;
}
}
