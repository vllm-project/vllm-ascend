/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#ifndef FUSED_GDN_GATING_TORCH_ADPT_H
#define FUSED_GDN_GATING_TORCH_ADPT_H

#include <tuple>

namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor> npu_fused_gdn_gating(
    const at::Tensor& A_log,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& dt_bias,
    double beta = 1.0,
    double threshold = 20.0)
{
    TORCH_CHECK(A_log.dim() == 1, "A_log should be 1-D [num_heads], got ", A_log.dim(), "D");
    TORCH_CHECK(dt_bias.dim() == 1, "dt_bias should be 1-D [num_heads], got ", dt_bias.dim(), "D");
    TORCH_CHECK(a.dim() == 2, "a should be 2-D [batch, num_heads], got ", a.dim(), "D");
    TORCH_CHECK(b.dim() == 2, "b should be 2-D [batch, num_heads], got ", b.dim(), "D");
    TORCH_CHECK(b.size(0) == a.size(0) && b.size(1) == a.size(1),
                "a and b must have the same shape, got a=", a.sizes(), " b=", b.sizes());
    TORCH_CHECK(a.scalar_type() == b.scalar_type(),
                "a and b must have the same dtype, got a=", a.scalar_type(),
                " b=", b.scalar_type());
    TORCH_CHECK(A_log.scalar_type() == dt_bias.scalar_type(),
                "A_log and dt_bias must have the same dtype, got A_log=",
                A_log.scalar_type(), " dt_bias=", dt_bias.scalar_type());
    TORCH_CHECK(a.size(1) == A_log.size(0),
                "a second dim (num_heads) must equal A_log first dim, got a.size(1)=",
                a.size(1), " A_log.size(0)=", A_log.size(0));

    int64_t batch = a.size(0);
    int64_t num_heads = a.size(1);

    at::Tensor g = at::empty({1, batch, num_heads},
                             a.options().dtype(c10::kFloat));
    at::Tensor beta_output = at::empty({1, batch, num_heads}, b.options());

    float beta_val = static_cast<float>(beta);
    float threshold_val = static_cast<float>(threshold);

    EXEC_NPU_CMD(aclnnFusedGdnGating,
                 A_log, a, b, dt_bias,
                 beta_val,
                 threshold_val,
                 g, beta_output);

    return std::make_tuple(g, beta_output);
}

} // namespace vllm_ascend

#endif // FUSED_GDN_GATING_TORCH_ADPT_H
