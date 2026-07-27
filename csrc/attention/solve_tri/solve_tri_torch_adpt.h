/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#ifndef SOLVE_TRI_TORCH_ADPT_H
#define SOLVE_TRI_TORCH_ADPT_H

#include <string>

namespace vllm_ascend {

at::Tensor npu_solve_tri(
    const at::Tensor& x,
    c10::optional<at::IntArrayRef> cu_seqlens,
    c10::optional<at::IntArrayRef> chunk_indices,
    c10::string_view layout = "bhtd")
{
    std::string layout_str(layout);
    TORCH_CHECK(layout_str == "bhtd" || layout_str == "bsnd" || layout_str == "tnd",
                "layout must be one of bhtd, bsnd, or tnd, got ", layout_str);
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "x must be float16 or bfloat16, got ", x.scalar_type());
    TORCH_CHECK(x.dim() == 3 || x.dim() == 4,
                "x must be 3D [total_T, H, BT] or 4D [B, H/T, T/H, BT], got ",
                x.dim(), "D");

    if (layout_str == "tnd") {
        TORCH_CHECK(x.dim() == 3, "tnd layout expects 3D x [total_T, H, BT], got ", x.dim(), "D");
        TORCH_CHECK(cu_seqlens.has_value(), "cu_seqlens is required when layout is tnd");
        TORCH_CHECK(chunk_indices.has_value(), "chunk_indices is required when layout is tnd");
    } else {
        TORCH_CHECK(x.dim() == 4, layout_str, " layout expects 4D x, got ", x.dim(), "D");
    }

    const int64_t chunk_size = x.size(x.dim() - 1);
    TORCH_CHECK(chunk_size == 16 || chunk_size == 32 || chunk_size == 64 || chunk_size == 128,
                "x last dimension must be one of 16, 32, 64, or 128, got ", chunk_size);

    at::Tensor output = at::empty_like(x);
    const char* layout_cstr = layout_str.c_str();
    EXEC_NPU_CMD(aclnnSolveTri, x, cu_seqlens, chunk_indices, layout_cstr, output);
    return output;
}

} // namespace vllm_ascend

#endif // SOLVE_TRI_TORCH_ADPT_H
