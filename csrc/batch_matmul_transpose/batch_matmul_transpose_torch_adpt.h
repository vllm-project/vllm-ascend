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
#ifndef BATCH_MATMUL_TRANSPOSE_TORCH_ADPT_H
#define BATCH_MATMUL_TRANSPOSE_TORCH_ADPT_H
#include "op_host/batch_matmul_transpose.h"

namespace vllm_ascend {

void batch_matmul_transpose(const at::Tensor &tensor_a, const at::Tensor &tensor_b, at::Tensor &tensor_c,
                                    c10::optional<c10::string_view> format_mode,
                                    c10::optional<c10::string_view> quant_mode)
{
    auto [tiling_tensor, block_dim] = bmm_trans::batch_matmul_transpose_tiling(
        tensor_a,
        tensor_b,
        tensor_c,
        format_mode,
        quant_mode
    );

    void *gm_a = tensor_a.data_ptr();
    void *gm_b = tensor_b.data_ptr();
    void *gm_c = tensor_c.data_ptr();
    void *gm_tiling_data = tiling_tensor.data_ptr();

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("batch_matmul_transpose");

    cmd.SetCustomHandler([stream, gm_a, gm_b, gm_c, gm_tiling_data,
                          block_dim]() -> int {
        batch_matmul_transpose_impl(stream, gm_a, gm_b, gm_c, gm_tiling_data,
                            block_dim);
        return 0;
    });
    cmd.Run();
    return;
}

}
#endif