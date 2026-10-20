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
#
"""xlite worker subclass for `vllm-ascend`"""

from vllm.logger import logger
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_ascend.worker.worker import NPUWorker
from vllm_ascend.xlite.xlite_model_runner import XliteModelRunner, XliteModelRunnerV2


class XliteWorker(NPUWorker):
    """Xlite worker bases on NPUWorker. Only xlite specified code should be added in this class."""

    def init_device(self):
        self.device = self._init_device()
        # Initialize workspace manager
        num_ubatches = 1
        init_workspace_manager(self.device, num_ubatches)
        # Init ModelRunner here, so that we have access to self.device.
        if self.use_v2_model_runner:
            logger.warning_once("xlite: npu model runner v2 is in developing, some features doesn't work for now.")
        runner_cls = XliteModelRunnerV2 if self.vllm_config.use_v2_model_runner else XliteModelRunner
        logger.info_once("XliteModelRunner: using xlite model runner class `%s`", runner_cls.__name__)
        self.xlite_runner = self.model_runner = runner_cls(self.vllm_config, self.device)  # type: ignore[assignment]
        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

    def compile_or_warm_up_model(self):
        # Currently, the xlite runner does not need explicit compilation or warmup
        if not hasattr(self, "xlite_runner"):
            return super().compile_or_warm_up_model()
        with self.xlite_runner._bypass_xlite_wrapper():
            return super().compile_or_warm_up_model()
