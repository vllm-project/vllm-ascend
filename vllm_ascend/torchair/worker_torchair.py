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
# Adapted from vllm-project/vllm/vllm/worker/gpu_worker.py
#

import torch
import torch_npu
from vllm.logger import logger

import vllm_ascend.envs as envs_ascend
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.torchair.model_runner_torchair import NPUTorchairModelRunner
from vllm_ascend.torchair.utils import (check_kv_cache_bytes_cache_exist,
                                        check_torchair_cache_exist,
                                        delete_torchair_cache_file,
                                        read_kv_cache_bytes_from_file)
from vllm_ascend.worker.worker_v1 import NPUWorker


class NPUTorchairWorker(NPUWorker):

    def init_device(self):
        device = torch.device(f"npu:{self.local_rank}")
        NPUPlatform.set_device(device)
        NPUPlatform.empty_cache()
        self.init_npu_memory = NPUPlatform.mem_get_info()[0]

        # Initialize the distributed environment.
        self._init_worker_distributed_environment()
        # Set random seed.
        NPUPlatform.seed_everything(self.model_config.seed)

        # Init ModelRunner here, so that we have access to self.device.
        self.model_runner = NPUTorchairModelRunner(self.vllm_config, device)

    def determine_available_memory(self) -> int:
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        NPUPlatform.clear_npu_memory()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        _, total_npu_memory = NPUPlatform.mem_get_info()
        self.model_runner.profile_run()

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        free_npu_memory, _ = NPUPlatform.mem_get_info()
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        assert self.init_npu_memory > free_npu_memory, (
            "Error in memory profiling. "
            f"Initial free memory {self.init_npu_memory}, current free memory"
            f" {free_npu_memory}. This happens when the NPU memory was "
            "not properly cleaned up before initializing the vLLM instance.")

        # Get the peak memory allocation recorded by torch
        peak_memory = torch_npu.npu.memory_stats()["allocated_bytes.all.peak"]
        # TODO: don`t need impl this func after empty_cache in
        # Worker.determine_num_available_blocks() unified`
        NPUPlatform.empty_cache()
        torch_allocated_bytes = torch_npu.npu.memory_stats(
        )["allocated_bytes.all.current"]
        total_allocated_bytes = torch_npu.npu.mem_get_info(
        )[1] - torch_npu.npu.mem_get_info()[0]
        non_torch_allocations = total_allocated_bytes - torch_allocated_bytes
        if non_torch_allocations > 0:
            peak_memory += non_torch_allocations
        available_kv_cache_memory = int(
            total_npu_memory * self.cache_config.gpu_memory_utilization -
            peak_memory)
        available_kv_cache_memory = int(max(available_kv_cache_memory, 0))
        logger.info(
            f"Available memory: {available_kv_cache_memory}, total memory: {total_npu_memory}"
        )

        if check_torchair_cache_exist() and check_kv_cache_bytes_cache_exist():
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

    def execute_dummy_batch(self) -> None:
        runner = self.model_runner
        max_num_tokens = 1
        with_prefill = False
        if runner.dp_size > 1:
            max_num_tokens, with_prefill = runner._get_forward_metadata_across_dp(
                max_num_tokens, with_prefill)
        if not with_prefill:
            max_num_tokens = runner.select_torchair_padded_batch_size(
                max_num_tokens)
        runner._dummy_run(max_num_tokens,
                          is_compile=False,
                          with_prefill=with_prefill)
