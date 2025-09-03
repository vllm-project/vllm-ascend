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
import torch
from vllm.logger import logger

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.torchair.torchair_model_runner import NPUTorchairModelRunner
from vllm_ascend.torchair.utils import (check_kv_cache_bytes_cache_exist,
                                        delete_torchair_cache_file,
                                        read_kv_cache_bytes_from_file)
from vllm_ascend.worker.worker_v1 import NPUWorker


class NPUTorchairWorker(NPUWorker):
    """Torchair worker bases on NPUWorker. Only torchair specified code should be added in this class."""

    def determine_available_memory(self) -> int:
        """Override determine_available_memory to use cached torchair kv_cache_bytes."""

        available_kv_cache_memory = super().determine_available_memory()

        if get_ascend_config(
        ).torchair_graph_config.use_cached_kv_cache_bytes and check_kv_cache_bytes_cache_exist(
        ):
            old_kv_cache_bytes = read_kv_cache_bytes_from_file(
                torch.distributed.get_rank())
            if 0 < old_kv_cache_bytes <= available_kv_cache_memory:
                logger.info(
                    f"Use cached torchair kv_cache_bytes: {old_kv_cache_bytes}"
                )
                self.model_runner.new_kv_cache_bytes = old_kv_cache_bytes
                return old_kv_cache_bytes
            else:
                logger.info(
                    "Cached torchair kv_cache_bytes is too big, invalidate old torchair_cache"
                )
                delete_torchair_cache_file()
        bytes_floating_tolerance = 1024 * 1024 * envs_ascend.VLLM_ASCEND_KV_CACHE_MEGABYTES_FLOATING_TOLERANCE
        available_kv_cache_memory -= bytes_floating_tolerance
        logger.info(f"Use new kv_cache_bytes: {available_kv_cache_memory}")
        self.model_runner.new_kv_cache_bytes = available_kv_cache_memory

        return available_kv_cache_memory

    def init_device(self):
        """Override init_device to init torchair model runner"""
        device = self._init_device()
        # Init ModelRunner here, so that we have access to self.device.
        self.model_runner = NPUTorchairModelRunner(self.vllm_config, device)
