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
from vllm.config import CUDAGraphMode
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

    def _should_build_dummy_attn_metadata(
        self,
        force_attention: bool = False,
        is_profile: bool = False,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
    ) -> bool:
        """
        Override to build attention metadata during dummy_run when xlite is enable.
        For xlite, we need to build metadata during DP dummy_run to ensure all ranks
        have consistent metadata, even when some ranks have no requests.
        """
        base_condition = super()._should_build_dummy_attn_metadata(force_attention, is_profile, cudagraph_runtime_mode)
        xlite_condition = self.ascend_config.xlite_graph_config.enabled and not is_profile
        return base_condition or xlite_condition
