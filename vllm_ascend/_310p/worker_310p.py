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

import torch_npu
from vllm.logger import logger

from vllm_ascend._310p.model_runner_310p import NPUModelRunner310
from vllm_ascend.worker.worker import NPUWorker, init_workspace_manager


class NPUWorker310(NPUWorker):
    def init_device(self):
        self.device = self._init_device()
        torch_npu.npu.set_compile_mode(jit_compile=False)

        init_workspace_manager(self.device, num_ubatches=1)

        self.model_runner = NPUModelRunner310(self.vllm_config, self.device)

    def _warm_up_atb(self):
        # 310p device do not support torch_npu._npu_matmul_add_fp32 atb ops
        logger.info("Skip warm-up atb ops for 310P device.")
