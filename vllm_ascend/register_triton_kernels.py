#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#
from vllm.model_executor.triton_dispatcher import register_kernel


@register_kernel("vllm.v1.sample.rejection_sampler.expand_kernel")
def expand_kernel_ascend(
    output_ptr,
    input_ptr,
    cu_num_tokens_ptr,
    replace_from,
    replace_to,
    MAX_NUM_TOKENS,
    grid=None,
):
    """
    Ascend implementation of expand_kernel.

    This kernel expands a [batch_size] tensor to [num_tokens] tensor.
    The Ascend implementation uses a different grid/block strategy for
    better performance on NPU hardware.

    Args:
        output_ptr: Output tensor [num_tokens]
        input_ptr: Input tensor [batch_size]
        cu_num_tokens_ptr: Cumulative number of tokens [batch_size]
        replace_from: Value to replace
        replace_to: Replacement value
        MAX_NUM_TOKENS: Maximum number of tokens (tl.constexpr)
        grid: Grid configuration from dispatcher (batch_size)
    """
    from vllm_ascend.ops.triton.reject_sample import (
        cal_grid_and_block_size,
        expand_kernel,
    )

    batch_size = grid if isinstance(grid, int) else grid[0]
    grid_size, block_size = cal_grid_and_block_size(batch_size)
    vec_len = batch_size

    expand_kernel[(grid_size,)](
        output_ptr,
        input_ptr,
        cu_num_tokens_ptr,
        replace_from,
        replace_to,
        vec_len,
        MAX_NUM_TOKENS=MAX_NUM_TOKENS,
        BLOCK_SIZE=block_size,
    )
