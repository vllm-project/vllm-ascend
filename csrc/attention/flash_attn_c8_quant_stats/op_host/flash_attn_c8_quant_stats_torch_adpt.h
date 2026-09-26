// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "../op_kernel/flash_attn_c8_quant_common.h"
namespace vllm_ascend {
inline void check_c8_prepare_view(const at::Tensor &tensor, int64_t width)
{
    TORCH_CHECK(tensor.dim() == 3 && tensor.size(0) > 0 && tensor.size(1) > 0 &&
        tensor.size(1) <= 256 && tensor.size(2) == width && tensor.stride(2) == 1 &&
        tensor.stride(0) >= width && tensor.stride(1) >= width && tensor.scalar_type() == at::kBFloat16,
        "C8 preparation requires a nonempty BF16 TND view with dense final dimension");
}
inline at::Tensor c8_prepare_storage_view(const at::Tensor &tensor)
{
    const int64_t span = (tensor.size(0) - 1) * tensor.stride(0) +
        (tensor.size(1) - 1) * tensor.stride(1) + tensor.size(2);
    return tensor.as_strided({span}, {1});
}
inline at::Tensor flash_attn_c8_quant_stats_meta(const at::Tensor &key, const at::Tensor &value)
{
    check_c8_prepare_view(key, 128);
    check_c8_prepare_view(value, 128);
    TORCH_CHECK(key.sizes() == value.sizes() && key.device() == value.device(),
        "C8 key/value shape and device must match");
    return at::empty({(key.size(0) + FlashAttnC8Config::STATS_TOKENS - 1) / FlashAttnC8Config::STATS_TOKENS,
        key.size(1), 16}, key.options().dtype(at::kFloat));
}
inline at::Tensor flash_attn_c8_quant_stats(const at::Tensor &key, const at::Tensor &value)
{
    auto partial = flash_attn_c8_quant_stats_meta(key, value);
    TORCH_CHECK(key.device().type() == c10::DeviceType::PrivateUse1, "C8 preparation requires NPU tensors");
    auto keyStorage = c8_prepare_storage_view(key);
    auto valueStorage = c8_prepare_storage_view(value);
    const int64_t tokens = key.size(0), heads = key.size(1);
    const int64_t kt = key.stride(0), kh = key.stride(1), vt = value.stride(0), vh = value.stride(1);
    EXEC_NPU_CMD(aclnnInnerFlashAttnC8QuantStats, keyStorage, valueStorage, tokens, heads, kt, kh, vt, vh, partial);
    return partial;
}
}
