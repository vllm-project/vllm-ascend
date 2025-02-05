#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

import os
from typing import Optional, Tuple

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    print("Failed to import torch_npu.")

from vllm.config import VllmConfig
from vllm.platforms import Platform, PlatformEnum

os.environ["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"] = "1"


def _device_id_to_physical_device_id(device_id: int) -> int:
    if "ASCEND_RT_VISIBLE_DEVICES" in os.environ:
        device_ids = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
        if device_ids == [""]:
            raise RuntimeError("ASCEND_RT_VISIBLE_DEVICES is set to empty"
                               "string, which means Ascend NPU support is"
                               "disabled.")
        physical_device_id = device_ids[device_id]
        return int(physical_device_id)
    else:
        return device_id


class NPUPlatform(Platform):

    _enum = PlatformEnum.OOT
    device_name: str = "npu"
    device_type: str = "npu"
    simple_compile_backend: str = "npu"
    ray_device_key: str = "NPU"
    device_control_env_var: str = "ASCEND_RT_VISIBLE_DEVICES"

    @classmethod
    def get_device_capability(cls, device_id: int = 0):
        return None

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        physical_device_id = _device_id_to_physical_device_id(device_id)
        return torch.npu.get_device_name(physical_device_id)

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return True

    @classmethod
    def inference_mode(cls):
        return torch.inference_mode()

    @classmethod
    def set_device(cls, device: torch.device):
        torch.npu.set_device(device)

    @classmethod
    def empty_cache(cls):
        torch.npu.empty_cache()

    @classmethod
    def synchronize(cls):
        torch.npu.synchronize()

    @classmethod
    def mem_get_info(cls) -> Tuple[int, int]:
        return torch.npu.mem_get_info()

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        # Register ops when setup.
        from vllm_ascend import ops  # noqa: F401

        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm_ascend.worker.NPUWorker"
        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 16

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, head_size, dtype,
                             kv_cache_dtype, block_size, use_v1, use_mla):
        return "vllm_ascend.attention.AscendAttentionBackend"

    @classmethod
    def get_current_memory_usage(cls,
                                 device: Optional[torch.types.Device] = None
                                 ) -> float:
        torch.npu.reset_peak_memory_stats(device)
        return torch.npu.max_memory_allocated(device)

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm_ascend.communicator.NPUCommunicator"
