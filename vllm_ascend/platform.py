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

import logging
import os
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import vllm.envs as envs
from vllm.logger import logger
from vllm.platforms import Platform, PlatformEnum
from vllm.utils import supports_dynamo

CUSTOM_OP_ENABLED = False
try:
    # register custom ops into torch_library here
    import vllm_ascend.vllm_ascend_C  # type: ignore  # noqa: F401

except ImportError:
    logging.warning(
        "Warning: Failed to register custom ops, all custom ops will be disabled"
    )
else:
    CUSTOM_OP_ENABLED = True

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
    from vllm.utils import FlexibleArgumentParser
else:
    ModelConfig = None
    VllmConfig = None
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

    def is_sleep_mode_available(self) -> bool:
        return True

    @classmethod
    def pre_register_and_update(cls,
                                parser: Optional[FlexibleArgumentParser] = None
                                ) -> None:
        # Adapt the global patch here.
        from vllm_ascend.utils import adapt_patch
        adapt_patch(is_global_patch=True)

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
        from vllm.config import CompilationLevel  # noqa: E402
        compilation_config = vllm_config.compilation_config

        if vllm_config.model_config is None:
            logger.warning("Model config is missing. This may indicate "
                           "that we are running a test case")
            enforce_eager = False
        else:
            enforce_eager = getattr(vllm_config.model_config, "enforce_eager",
                                    False)

        # TODO(Yizhou): Override the value of enforce_eager to True before
        # the CANN and torch_npu support NPU compilation.
        enforce_eager = True
        logger.warning(
            "NPU compilation support pending. Will be available in future CANN and "
            "torch_npu releases. NPU graph mode is currently experimental and disabled "
            "by default. You can just adopt additional_config={'enable_graph_mode': True} "
            "to serve deepseek models with NPU graph mode on vllm-ascend with V0 engine. "
        )

        if enforce_eager or compilation_config.level == CompilationLevel.NO_COMPILATION:
            logger.info("Compilation disabled, using eager mode by default")
            compilation_config.level = CompilationLevel.NO_COMPILATION
        elif compilation_config.level != CompilationLevel.PIECEWISE:
            logger.warning(
                "NPU does not support %s compilation level. Setting level to NO_COMPILATION",
                compilation_config.level)
            compilation_config.level = CompilationLevel.NO_COMPILATION
        else:
            logger.info(
                "PIECEWISE compilation enabled on NPU. use_inductor not supported - "
                "using only ACL Graph mode")
            compilation_config.use_inductor = False
            compilation_config.splitting_ops.extend(
                ["vllm.unified_ascend_attention_with_output"])

        if vllm_config.additional_config is not None:
            enable_graph_mode = vllm_config.additional_config.get(
                "enable_graph_mode", False)
            if enable_graph_mode and not supports_dynamo():
                logger.warning(
                    "enable_graph_mode is not supported because the version of torch is too low, forcing close enable_graph_mode"
                )
                vllm_config.additional_config["enable_graph_mode"] = False
            if enable_graph_mode and envs.VLLM_USE_V1:
                logger.warning(
                    "NPU graph mode is still experimental and not supported for V1 currently, "
                    "it has been disabled automatically.")
                vllm_config.additional_config["enable_graph_mode"] = False

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
        if cache_config:
            if cache_config.block_size is None:
                cache_config.block_size = 128
            if cache_config.enable_prefix_caching and cache_config.block_size != 128:
                logger.warning(
                    "If prefix caching is enabled, block size must be set to 128."
                )
                cache_config.block_size = 128

        if envs.VLLM_USE_V1:
            # Activate custom ops for v1.
            vllm_config.compilation_config.custom_ops = ["all"]
            # If ascend_scheduler_config exists in additional_config,
            # extents original scheduler_config to use AscendScheduler.

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

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, head_size, dtype,
                             kv_cache_dtype, block_size, use_v1, use_mla):
        if use_v1 and use_mla:
            return "vllm_ascend.attention.mla_v1.AscendMLABackend"
        if use_v1:
            return "vllm_ascend.attention.attention_v1.AscendAttentionBackend"
        if use_mla:
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

    @classmethod
    def supports_v1(cls, model_config: ModelConfig) -> bool:
        """Returns whether the current platform can support v1 for the supplied
        model configuration.
        """
        return True
