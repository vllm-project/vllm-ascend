/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Fused grouped matmul (fake-A8W8) + SiTU + per-token INT8 quant, single launch
// (GroupedMatmulSituQuantV2). Vendored from the A3 fused-operator delivery and
// adapted to the vllm-ascend extension build.
#ifndef VLLM_ASCEND_GMSQ2_SITU_QUANT_H_
#define VLLM_ASCEND_GMSQ2_SITU_QUANT_H_

#include <optional>
#include <tuple>
#include <vector>

#include <ATen/ATen.h>

namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor> grouped_matmul_situ_quant_v2(
    const at::Tensor &x, at::TensorList weight, at::TensorList weight_scale,
    const at::Tensor &x_scale, const at::Tensor &group_list, at::TensorList weight_assist_matrix,
    double beta, std::optional<double> linear_beta, int64_t group_list_type);

}  // namespace vllm_ascend

#endif  // VLLM_ASCEND_GMSQ2_SITU_QUANT_H_
