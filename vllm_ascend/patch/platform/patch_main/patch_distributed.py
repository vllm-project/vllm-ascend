#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
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
# Adapted from vllm/model_executor/models/qwen2_vl.py
# This file is a part of the vllm-ascend project.

import vllm
import vllm.distributed
from vllm.config import ParallelConfig
from vllm_ascend.patch.platform.patch_0_8_4.patch_distributed import ascend_destroy_model_parallel, ascend_stateless_init_torch_distributed_process_group, parallel_config_get_dp_port

# All details of those patch please refer to vllm_ascend/patch/platform/patch_0_8_4/patch_distributed.py
vllm.distributed.parallel_state.destroy_model_parallel = ascend_destroy_model_parallel
vllm.distributed.stateless_init_torch_distributed_process_group = ascend_stateless_init_torch_distributed_process_group
ParallelConfig.get_next_dp_init_port = parallel_config_get_dp_port
