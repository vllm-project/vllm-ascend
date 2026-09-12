// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vllm-ascend project

#ifndef REARRANGE_QKV_TORCH_ADPT_H
#define REARRANGE_QKV_TORCH_ADPT_H

#include <cstdint>

namespace vllm_ascend {

at::Tensor npu_rearrange_qkv(
    const at::Tensor& x, int64_t q_dim, int64_t k_dim, int64_t v_dim)
{
    constexpr int64_t elements_per_datablock = 16;
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf,
                "npu_rearrange_qkv requires BF16 or FP16 input");
    TORCH_CHECK(q_dim > 0 && k_dim > 0 && v_dim > 0,
                "npu_rearrange_qkv requires positive Q/K/V dimensions");
    TORCH_CHECK(q_dim % elements_per_datablock == 0 &&
                    k_dim % elements_per_datablock == 0 &&
                    v_dim % elements_per_datablock == 0,
                "npu_rearrange_qkv requires 32-byte-aligned Q/K/V widths");
    TORCH_CHECK(x.dim() == 2,
                "npu_rearrange_qkv requires [T, Q + K + V] input");

    const int64_t row_dim = x.size(1);
    TORCH_CHECK(q_dim <= row_dim && k_dim <= row_dim - q_dim &&
                    v_dim == row_dim - q_dim - k_dim,
                "npu_rearrange_qkv requires [T, Q + K + V] input");
    TORCH_CHECK(x.is_contiguous(),
                "npu_rearrange_qkv requires contiguous input");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(x.data_ptr()) % 32 == 0,
                "npu_rearrange_qkv requires a 32-byte-aligned input address");

    const c10_npu::OptionalNPUGuard guard(x.device());
    at::Tensor output = at::empty({x.numel()}, x.options());
    if (x.numel() == 0) {
        return output;
    }
    EXEC_NPU_CMD(aclnnRearrangeQkvDma, x, q_dim, k_dim, v_dim, output);
    return output;
}

}  // namespace vllm_ascend

#endif  // REARRANGE_QKV_TORCH_ADPT_H
