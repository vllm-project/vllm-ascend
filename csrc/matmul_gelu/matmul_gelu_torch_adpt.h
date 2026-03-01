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

#ifndef MATMUL_GELU_TORCH_ADPT_H
#define MATMUL_GELU_TORCH_ADPT_H
namespace vllm_ascend {

at::Tensor matmul_gelu(const at::Tensor &x, const at::Tensor &weight, const at::Tensor &bias)
{
    TORCH_CHECK(x.dim() == 2, "The x should be 2D");
    TORCH_CHECK(weight.dim() == 2, "The weight should be 2D");
    TORCH_CHECK(bias.dim() == 1, "The bias should be 1D");
    TORCH_CHECK(weight.sizes()[0] == bias.sizes()[0] , "The weight first dim should be same as bias first dim");
    TORCH_CHECK(x.sizes()[1] == weight.sizes()[1] , "The x second dim should be same as weight second dim");
    TORCH_CHECK(
        x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat,
        "float16 or float32 tensor expected but got a tensor with dtype: ",
        x.scalar_type());
     TORCH_CHECK(
        x.scalar_type() == weight.scalar_type() && x.scalar_type() == bias.scalar_type(),
        "The dtype of x, weight and bias should be same");

	int m = x.sizes()[0];
	int n = bias.sizes()[0];
	auto options = x.options();

    at::Tensor gelu_output = at::empty({m,n}, x.options());
    EXEC_NPU_CMD(
        aclnnMatmulGelu,
        x,
        weight,
        bias,
		gelu_output);
    return gelu_output;
}

}
#endif