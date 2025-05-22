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

import logging
import os
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch_npu  # noqa: F401
import vllm.envs as envs
from vllm.config import CompilationLevel, VllmConfig
from vllm.logger import logger

try:
    # register custom ops into torch_library here
    import vllm_ascend.vllm_ascend_C  # type: ignore  # noqa: F401

except ImportError as e:
    if not str(
            e
    ) == "dynamic module does not define module export function (PyInit_vllm_ascend_C)":
        logging.warning(
            "Warning: Failed to register custom ops, all custom ops will be disabled"
        )

from vllm.platforms import Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = None

os.environ["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"] = "1"


class NPUPlatform(Platform):

    _enum = PlatformEnum.OOT
    device_name: str = "npu"
    device_type: str = "npu"
    simple_compile_backend: str = "eager"  # Disable torch.compile()
    ray_device_key: str = "NPU"
    device_control_env_var: str = "ASCEND_RT_VISIBLE_DEVICES"
    dispatch_key: str = "PrivateUse1"

    supported_quantization: list[str] = ["ascend"]

    @classmethod
    def pre_register_and_update(cls,
                                parser: Optional[FlexibleArgumentParser] = None
                                ) -> None:
        import vllm_ascend.patch_model_config  # noqa: F401
        import vllm_ascend.patch_outputs  # noqa: F401
        import vllm_ascend.patch_vllm_config  # noqa: F401
        from vllm_ascend.quantization.quant_config import \
            AscendQuantConfig  # noqa: F401

    @classmethod
    def get_device_capability(cls, device_id: int = 0):
        return None

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.npu.get_device_name(device_id)

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
        # RayWorkerWrapper monkey patch when setup
        from vllm_ascend.patch import ray_patch  # noqa: F401

        compilation_config = vllm_config.compilation_config
        if compilation_config and compilation_config.level != CompilationLevel.NO_COMPILATION:
            logger.warning(
                "Compilation level %s is not supported on NPU now, forcing compilation level to NO_COMPILATION",
                compilation_config.level)
            compilation_config.level = CompilationLevel.NO_COMPILATION

        parallel_config = vllm_config.parallel_config
        if parallel_config and parallel_config.worker_cls == "auto":
            if envs.VLLM_USE_V1:
                parallel_config.worker_cls = "vllm_ascend.worker.worker_v1.NPUWorker"
            elif vllm_config.speculative_config:
                parallel_config.worker_cls = "vllm.spec_decode.spec_decode_worker.create_spec_worker"
                parallel_config.sd_worker_cls = "vllm_ascend.worker.worker.NPUWorker"
            elif vllm_config.scheduler_config.is_multi_step:
                parallel_config.worker_cls = "vllm_ascend.worker.multi_step_worker.MultiStepWorker"
            else:
                parallel_config.worker_cls = "vllm_ascend.worker.worker.NPUWorker"

        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 128
        if cache_config and not hasattr(cache_config, "enable_prefix_caching"):
            setattr(cache_config, "enable_prefix_caching", False)
        if cache_config and cache_config.enable_prefix_caching and cache_config.block_size != 128:
            raise ValueError(
                "If prefix caching is enabled, block size must be set to 128.")
        if vllm_config.quant_config is not None and \
            'fa_quant_type' in vllm_config.quant_config.quant_description.keys() and \
            vllm_config.quant_config.quant_description['fa_quant_type'] is not None:
            # Ascend attention quant uses int8 dtype.
            cache_config.cache_dtype = 'int8'

        if envs.VLLM_USE_V1 and cache_config and cache_config.enable_prefix_caching:
            logger.warning(
                "Prefix caching is now supported for V1 on NPU, "
                "but it is still experimental and there may be issues with accuracy."
            )

        if envs.VLLM_USE_V1:
            # Activate custom ops for v1.
            vllm_config.compilation_config.custom_ops = ["all"]
            additional_config = vllm_config.additional_config
            if additional_config and additional_config.get(
                    "ascend_scheduler_config", None) is not None:
                additional_scheduler_config = additional_config.get(
                    "ascend_scheduler_config")
                from vllm_ascend.core.schedule_config import \
                    AscendSchedulerConfig
                ascend_scheduler_config = AscendSchedulerConfig.initialize_from_config(
                    vllm_config.scheduler_config, additional_scheduler_config)
                vllm_config.scheduler_config = ascend_scheduler_config
                from vllm.v1.engine.core import EngineCore

                from vllm_ascend.core.v1_engine_core_init import \
                    engine_core_init_with_ascend_scheduler
                EngineCore.__init__ = engine_core_init_with_ascend_scheduler

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, head_size, dtype,
                             kv_cache_dtype, block_size, use_v1, use_mla):
        if use_v1:
            return "vllm_ascend.attention.attention_v1.AscendAttentionBackend"
        elif use_mla:
            return "vllm_ascend.attention.attention.AscendMLAAttentionBackend"
        return "vllm_ascend.attention.attention.AscendAttentionBackend"

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm_ascend.lora.punica_wrapper.punica_npu.PunicaWrapperNPU"

    @classmethod
    def get_current_memory_usage(cls,
                                 device: Optional[torch.types.Device] = None
                                 ) -> float:
        torch.npu.reset_peak_memory_stats(device)
        return torch.npu.max_memory_allocated(device)

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm_ascend.distributed.communicator.NPUCommunicator"

    @classmethod
    def is_pin_memory_available(cls):
        return True
