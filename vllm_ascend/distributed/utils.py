#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

from vllm.distributed.parallel_state import (
    get_dp_group, get_tensor_model_parallel_world_size)

from vllm_ascend.distributed.parallel_state import get_lmhead_group


def is_lmhead_tp():
    # We only activate optimization of lmhead communication
    # when tp_size == 1, dp_size > 1 and lmhead_tp_size > 1.

    try:
        get_lmhead_group()
    except AssertionError:
        return False

    tp_size = get_tensor_model_parallel_world_size()
    dp_size = get_dp_group().world_size
    lmhead_tp_size = get_lmhead_group().world_size

    return tp_size == 1 and dp_size > 1 and lmhead_tp_size > 1