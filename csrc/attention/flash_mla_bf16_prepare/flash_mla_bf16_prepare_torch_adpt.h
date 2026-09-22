// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace vllm_ascend {
inline std::tuple<at::Tensor, at::Tensor> flash_mla_bf16_prepare_meta(
    const at::Tensor &key_nope, const at::Tensor &value, const at::Tensor &key_rope)
{
    TORCH_CHECK(key_nope.dim() == 3 && key_nope.size(2) == 128 &&
        key_nope.size(1) > 0 && key_nope.size(1) <= 128 && value.sizes() == key_nope.sizes() &&
        key_rope.dim() == 3 && key_rope.size(0) == key_nope.size(0) && key_rope.size(2) == 64 &&
        (key_rope.size(1) == 1 || key_rope.size(1) == key_nope.size(1)),
        "FlashMlaBf16Prepare requires K/V[T,H,128] and RoPE[T,1|H,64], 1<=H<=128");
    for (const auto *input : {&key_nope, &value, &key_rope}) {
        TORCH_CHECK(input->scalar_type() == at::kBFloat16 && input->stride(2) == 1 &&
            input->stride(1) >= input->size(2) && input->stride(0) >= input->size(1) * input->stride(1),
            "FlashMlaBf16Prepare requires BF16 dense, nonoverlapping head rows");
    }
    return {at::empty({key_nope.size(0), key_nope.size(1), 192}, key_nope.options()),
            at::empty({value.size(0), value.size(1), 128}, value.options())};
}

inline std::tuple<at::Tensor, at::Tensor> flash_mla_bf16_prepare(
    const at::Tensor &key_nope, const at::Tensor &value, const at::Tensor &key_rope)
{
    auto output = flash_mla_bf16_prepare_meta(key_nope, value, key_rope);
    TORCH_CHECK(key_nope.device().type() == c10::DeviceType::PrivateUse1 &&
        value.device() == key_nope.device() && key_rope.device() == key_nope.device(),
        "FlashMlaBf16Prepare requires tensors on one NPU");
    if (key_nope.size(0) == 0) return output;
    const int64_t kt = key_nope.stride(0), kh = key_nope.stride(1);
    const int64_t vt = value.stride(0), vh = value.stride(1);
    const int64_t rt = key_rope.stride(0), rh = key_rope.stride(1);
    EXEC_NPU_CMD(aclnnInnerFlashMlaBf16Prepare, key_nope, value, key_rope,
        kt, kh, vt, vh, rt, rh, std::get<0>(output), std::get<1>(output));
    return output;
}
} // namespace vllm_ascend
