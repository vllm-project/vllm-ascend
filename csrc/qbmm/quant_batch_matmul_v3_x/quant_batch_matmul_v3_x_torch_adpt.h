/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef QUANT_BATCH_MATMUL_V3_X_TORCH_ADPT_H
#define QUANT_BATCH_MATMUL_V3_X_TORCH_ADPT_H
namespace vllm_ascend {

at::Tensor quant_batch_matmul_v3_x(
    const at::Tensor &x1,
    const at::Tensor &x2,
    const at::Tensor &scale,
    const c10::optional<at::Tensor> &offset,
    const c10::optional<at::Tensor> &pertoken_scale,
    const c10::optional<at::Tensor> &bias,
    bool transpose_x1,
    bool transpose_x2,
    int64_t group_size)
{
    TORCH_CHECK(x1.dim() >= 2 && x2.dim() >= 2,
                "quant_batch_matmul_v3_x: x1 and x2 must have rank >= 2");
    // Output shape matches the op's InferShape: batch dims from x1 + [M, N].
    const int64_t m = transpose_x1 ? x1.size(-1) : x1.size(-2);
    const int64_t n = transpose_x2 ? x2.size(-2) : x2.size(-1);
    std::vector<int64_t> out_shape(x1.sizes().begin(), x1.sizes().end() - 2);
    out_shape.push_back(m);
    out_shape.push_back(n);
    // Kernel output is FP16 (op_def declares DT_FLOAT16 for all variants).
    at::Tensor out = at::empty(out_shape, x1.options().dtype(at::kHalf));

    // The generated aclnn wrapper selects the FRACTAL_NZ / ND kernel variant
    // from x2's storage format, so no format dispatch is needed here.
    // dtype is the output dtype code: 1 = FP16. It must be an lvalue --
    // EXEC_NPU_CMD forwards arguments as non-const lvalue references.
    const int64_t dtype = 1;
    EXEC_NPU_CMD(aclnnQuantBatchMatmulV3X,
                 x1,
                 x2,
                 scale,
                 offset,
                 bias,
                 pertoken_scale,
                 dtype,
                 transpose_x1,
                 transpose_x2,
                 group_size,
                 out);
    return out;
}

}
#endif
