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
from vllm_ascend.worker.worker_v1 import NPUWorker
from vllm_ascend.xlite.xlite_model_runner import XliteModelRunner


class XliteWorker(NPUWorker):
    """Xlite worker bases on NPUWorker. Only xlite specified code should be added in this class."""

    def init_device(self):
        """Override init_device to init xlite model runner"""
        self.device = self._init_device()
        self.model_runner = XliteModelRunner(self.vllm_config, self.device)