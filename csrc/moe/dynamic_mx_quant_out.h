/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace vllm_ascend {

// MXFP8 E4M3, per-32 RINT quantization into caller-owned communication storage.
// Keep raw scale bytes at the graph boundary: Inductor cannot decompose
// auto_functionalized_v2 nodes with E8M0 inputs. Reinterpret them only for ACLNN.
inline void dynamic_mx_quant_out(const at::Tensor& x, at::Tensor out, at::Tensor scale)
{
    if (x.numel() == 0) {
        return;
    }
    constexpr int64_t axis = -1;
    constexpr int64_t dst_type = ACL_FLOAT8_E4M3FN;
    constexpr int64_t block_size = 32;
    constexpr int64_t scale_algorithm = 0;
    char round_mode[] = "rint";
    char* mode = round_mode;
    auto mx_scale = scale.view(at::kFloat8_e8m0fnu);
    EXEC_NPU_CMD(aclnnDynamicMxQuant, x, axis, mode, dst_type,
                 block_size, scale_algorithm, out, mx_scale);
}

} // namespace vllm_ascend
