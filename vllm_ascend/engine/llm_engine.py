# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm/tests/kernels/test_moe.py
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

def apply_custom_engine_patch():
    from vllm.engine.llm_engine import LLMEngine
    from vllm.config import ParallelConfig  
    from vllm.engine.arg_utils import EngineArgs  # noqa
    from vllm.config import VllmConfig as vllm_config  # noqa

    original_init = LLMEngine.__init__

    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.dp_enabled = self.parallel_config.data_parallel_size > 1  # noqa
        self.should_execute_dummy_batch = False
        if self.dp_enabled:
            self.dp_group = self.parallel_config.stateless_init_dp_group()

    def new_has_unfinished_requests(self) -> bool:
        has_unfinished = any(scheduler.has_unfinished_seqs()
                             for scheduler in self.scheduler)
        if not self.dp_enabled:
            return has_unfinished
        return new_has_unfinished_requests_dp(self, has_unfinished)

    def new_has_unfinished_requests_dp(self, has_unfinished: bool) -> bool:
        aggregated_has_unfinished = ParallelConfig.has_unfinished_dp(
            self.dp_group, has_unfinished)
        if not has_unfinished and aggregated_has_unfinished:
            self.should_execute_dummy_batch = True
        return aggregated_has_unfinished

    LLMEngine.__init__ = new_init
    LLMEngine.has_unfinished_requests = new_has_unfinished_requests
    LLMEngine.has_unfinished_requests_dp = new_has_unfinished_requests_dp
    print("[Success] Custom LLMEngine patch applied!")


apply_custom_engine_patch()
