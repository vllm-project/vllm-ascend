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
# Adapted from vllm-project/vllm/vllm/worker/gpu_model_runner.py
# isort: skip_file
import torch.nn as nn
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class XliteModelRunner(NPUModelRunner):

    def get_model(self) -> nn.Module:
        return self.model.unwrap()

    def load_model(self) -> None:
        super().load_model()
        from vllm_ascend.xlite.xlite import XliteWrapper
        self.model = XliteWrapper(self.model, self.vllm_config)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        super().initialize_kv_cache(kv_cache_config)
        self.model.register_kv_caches(self.kv_caches)
