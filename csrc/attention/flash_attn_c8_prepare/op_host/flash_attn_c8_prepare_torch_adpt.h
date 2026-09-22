// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "../../flash_attn_c8_quant_stats/op_host/flash_attn_c8_quant_stats_torch_adpt.h"
namespace vllm_ascend {
using FlashAttnC8Prepared = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
inline FlashAttnC8Prepared flash_attn_c8_prepare_meta(const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &value, const at::Tensor &rope, const at::Tensor &partial, bool fake_quant)
{
    check_c8_prepare_view(query, 192);
    check_c8_prepare_view(key, 128);
    check_c8_prepare_view(value, 128);
    check_c8_prepare_view(rope, 64);
    const int64_t tq = query.size(0), tk = key.size(0), h = query.size(1);
    TORCH_CHECK(key.size(1) == h && value.sizes() == key.sizes() && rope.size(0) == tk &&
        (rope.size(1) == 1 || rope.size(1) == h), "C8 preparation incompatible Q/K/V/RoPE shapes");
    TORCH_CHECK(partial.dim() == 3 &&
        partial.size(0) == (tk + FlashAttnC8Config::STATS_TOKENS - 1) / FlashAttnC8Config::STATS_TOKENS &&
        partial.size(1) == h && partial.size(2) == 16 && partial.scalar_type() == at::kFloat &&
        partial.is_contiguous(), "C8 preparation requires matching contiguous quant statistics");
    for (const auto *tensor : {&key, &value, &rope, &partial}) {
        TORCH_CHECK(tensor->device() == query.device(), "C8 preparation tensors must share one device");
    }
    auto options = query.options().dtype(fake_quant ? at::kBFloat16 : at::ScalarType::Float8_e4m3fn);
    return {at::empty({tq, h, fake_quant ? 192 : 128}, options),
        at::empty({tk, h, fake_quant ? 192 : 128}, options), at::empty({tk, h, 128}, options),
        at::empty({tq, h, 64}, query.options()), at::empty({tk, h, 64}, query.options()),
        at::empty({tq, h}, query.options().dtype(at::kFloat)),
        at::empty({h}, query.options().dtype(at::kFloat)), at::empty({h}, query.options().dtype(at::kFloat))};
}
inline FlashAttnC8Prepared flash_attn_c8_prepare(const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &value, const at::Tensor &rope, const at::Tensor &partial, bool fake_quant)
{
    auto outputs = flash_attn_c8_prepare_meta(query, key, value, rope, partial, fake_quant);
    TORCH_CHECK(query.device().type() == c10::DeviceType::PrivateUse1, "C8 preparation requires NPU tensors");
    auto q = c8_prepare_storage_view(query), k = c8_prepare_storage_view(key);
    auto v = c8_prepare_storage_view(value), r = c8_prepare_storage_view(rope);
    const int64_t tq = query.size(0), tk = key.size(0), h = query.size(1);
    const int64_t qt = query.stride(0), qh = query.stride(1), kt = key.stride(0), kh = key.stride(1);
    const int64_t vt = value.stride(0), vh = value.stride(1), rt = rope.stride(0);
    const int64_t rh = rope.size(1) == 1 ? 0 : rope.stride(1);
    EXEC_NPU_CMD(aclnnInnerFlashAttnC8Prepare, q, k, v, r, partial, tq, tk, h, qt, qh, kt, kh,
        vt, vh, rt, rh, fake_quant, std::get<0>(outputs), std::get<1>(outputs), std::get<2>(outputs),
        std::get<3>(outputs), std::get<4>(outputs), std::get<5>(outputs), std::get<6>(outputs), std::get<7>(outputs));
    return outputs;
}
}
