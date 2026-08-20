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

namespace vllm_ascend {

void batch_matmul_transpose(const at::Tensor &tensor_a, const at::Tensor &tensor_b, at::Tensor &tensor_c,
                                    c10::optional<c10::string_view> format_mode,
                                    c10::optional<c10::string_view> quant_mode)
{
    const char *format_mode_ptr = format_mode.has_value() ? format_mode->data() : "ND";
    const char *quant_mode_ptr = quant_mode.has_value() ? quant_mode->data() : "per_channel_symm";
    EXEC_NPU_CMD(aclnnBatchMatmulTranspose, tensor_a, tensor_b, format_mode_ptr, quant_mode_ptr, tensor_c);
}

}
#endif
