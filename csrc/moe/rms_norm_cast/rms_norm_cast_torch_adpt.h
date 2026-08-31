/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */
#ifndef VLLM_ASCEND_RMS_NORM_CAST_TORCH_ADPT_H
#define VLLM_ASCEND_RMS_NORM_CAST_TORCH_ADPT_H

namespace vllm_ascend {
std::tuple<at::Tensor, at::Tensor> npu_rms_norm_cast(
    const at::Tensor& x, const at::Tensor& gamma, double epsilon)
{
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "npu_rms_norm_cast only supports float16 and bfloat16 input");
    TORCH_CHECK(x.scalar_type() == gamma.scalar_type(),
                "x and gamma must have the same dtype");
    TORCH_CHECK(gamma.dim() == 1 && x.dim() >= 1 &&
                    x.size(-1) == gamma.size(0),
                "gamma must be 1D and match the last dimension of x");
    at::Tensor y = at::empty(x.sizes(), x.options());
    at::Tensor y_fp32 = at::empty(x.sizes(), x.options().dtype(at::kFloat));
    EXEC_NPU_CMD(aclnnRmsNormCast, x, gamma, epsilon, y, y_fp32);
    return {y, y_fp32};
}
}  // namespace vllm_ascend
#endif
