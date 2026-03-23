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

from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

from vllm_ascend.ascend_forward_context import _EXTRA_CTX


def tensor_parallel_wrap(func):
    def wrap(*args, **kwargs):
        deepstack_input_embeds = func(*args, **kwargs)
        if deepstack_input_embeds is None:
            return deepstack_input_embeds
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        try:
            flash_comm_v1_enabled = _EXTRA_CTX.flash_comm_v1_enabled
        except (AssertionError, AttributeError):
            flash_comm_v1_enabled = False
        if flash_comm_v1_enabled:
            deepstack_input_embeds.tensors = {
                k: v.chunk(tp_size)[tp_rank] for k, v in deepstack_input_embeds.tensors.items()
            }
        return deepstack_input_embeds

    return wrap


from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration  # noqa: E402

Qwen3VLForConditionalGeneration._get_deepstack_input_embeds = tensor_parallel_wrap(
    Qwen3VLForConditionalGeneration._get_deepstack_input_embeds
)
