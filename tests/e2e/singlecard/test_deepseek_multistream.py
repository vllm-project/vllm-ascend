#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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

from tests.conftest import VllmRunner


def test_models_distributed_DeepSeek_multistream_moe(example_prompts):
    dtype = "half"
    max_tokens = 5
    with VllmRunner(
            "vllm-ascend/DeepSeek-V3-Pruning",
            dtype=dtype,
            additional_config={
                "torchair_graph_config": {
                    "enabled": True,
                    "enable_multistream_moe": True,
                },
                "ascend_scheduler_config": {
                    "enabled": True,
                },
                "refresh": True,
            },
            enforce_eager=False,
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
