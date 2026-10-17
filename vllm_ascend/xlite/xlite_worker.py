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
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_ascend.worker.worker import NPUWorker
from vllm_ascend.xlite.xlite_model_runner import XliteModelRunner


class XliteWorker(NPUWorker):
    """Xlite worker bases on NPUWorker. Only xlite specified code should be added in this class."""

    def init_device(self):
        self.device = self._init_device()
        num_ubatches = 1
        init_workspace_manager(self.device, num_ubatches)
        self.xlite_runner = self.model_runner = XliteModelRunner(self.vllm_config, self.device)  # type: ignore[assignment]

    def compile_or_warm_up_model(self):
        # Currently, the xlite runner does not need explicit compile or warmup
        if not hasattr(self, "xlite_runner"):
            return super().compile_or_warm_up_model()
        with self.xlite_runner._bypass_xlite_wrapper():
            return super().compile_or_warm_up_model()
