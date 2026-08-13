# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc

import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


@torch.inference_mode()
def test_grouped_matmul_swiglu_quant_v2_count_matches_cumsum():
    """Count and cumulative group lists must describe the same grouping."""
    num_tokens, hidden_size, num_experts, intermediate_size = 8, 7168, 4, 4096

    torch.manual_seed(0)
    hidden_states = torch.randint(-128, 127, (num_tokens, hidden_size), dtype=torch.int8).npu()
    weights = [
        torch_npu.npu_format_cast(
            torch.randint(-128, 127, (hidden_size, intermediate_size), dtype=torch.int8).npu(),
            29,
        )
        for _ in range(num_experts)
    ]
    weight_scales = [(torch.rand(intermediate_size, dtype=torch.float32) * 0.9 + 0.1).npu() for _ in range(num_experts)]
    token_scales = (torch.rand(num_tokens, dtype=torch.float32) * 0.9 + 0.1).npu()

    # Include an empty expert to exercise the boundary where count and
    # cumulative encodings differ most visibly.
    count_group_list = torch.tensor([2, 0, 3, 3], dtype=torch.int64, device="npu")
    cumsum_group_list = count_group_list.cumsum(dim=0)

    cumsum_output, cumsum_scale = torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2(
        x=hidden_states,
        weight=weights,
        weight_scale=weight_scales,
        x_scale=token_scales,
        group_list=cumsum_group_list,
        group_list_type=0,
    )
    count_output, count_scale = torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2(
        x=hidden_states,
        weight=weights,
        weight_scale=weight_scales,
        x_scale=token_scales,
        group_list=count_group_list,
        group_list_type=1,
    )

    torch.testing.assert_close(count_output.cpu(), cumsum_output.cpu(), atol=1, rtol=2**-13)
    torch.testing.assert_close(count_scale.cpu(), cumsum_scale.cpu(), atol=1e-9, rtol=1e-6)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
