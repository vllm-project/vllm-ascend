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
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch_npu  # noqa: F401
import vllm.envs as envs
from vllm.config import CompilationLevel
from vllm.logger import init_logger
from vllm.platforms import Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
    from vllm.utils import FlexibleArgumentParser
else:
    ModelConfig = None
    VllmConfig = None
    FlexibleArgumentParser = None

os.environ["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"] = "1"

logger = init_logger(__name__)


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
        if cache_config:
            if cache_config.block_size is None:
                cache_config.block_size = 128
            if envs.VLLM_USE_V1 and cache_config.enable_prefix_caching:
                logger.warning(
                    "Prefix caching is not supported for V1 now, disable prefix caching"
                )
                cache_config.enable_prefix_caching = False

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, head_size, dtype,
                             kv_cache_dtype, block_size, use_v1, use_mla):
        if use_v1:
            return "vllm_ascend.attention.attention_v1.AscendAttentionBackend"
        if use_mla:
            return "vllm_ascend.attention.attention.AscendMLAAttentionBackend"
        return "vllm_ascend.attention.attention.AscendAttentionBackend"

    @classmethod
    def get_current_memory_usage(cls,
                                 device: Optional[torch.types.Device] = None
                                 ) -> float:
        torch.npu.reset_peak_memory_stats(device)
        return torch.npu.max_memory_allocated(device)

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm_ascend.communicator.NPUCommunicator"

    @classmethod
    def is_pin_memory_available(cls):
        return True

    @classmethod
    def supports_v1(cls, model_config: ModelConfig) -> bool:
        """Returns whether the current platform can support v1 for the supplied
        model configuration.
        """
        return True

    @classmethod
    def destroy_platform_model_parallel(cls) -> None:
        from vllm_ascend.distributed.parallel_state import _EP, _ETP
        if _EP:
            _EP.destroy()
        _EP = None

        if _ETP:
            _ETP.destroy()
        _ETP = None

    @classmethod
    def platform_has_backend_register(cls) -> bool:
        return True

    @classmethod
    def platform_register_backend(cls, pg, prefix_store, group_rank, group_size,
                                         backend_options, timeout) -> None:
        from torch.distributed import ProcessGroup, is_hccl_available
        assert is_hccl_available()
        import torch_npu  # noqa
        from torch_npu._C._distributed_c10d import ProcessGroupHCCL
        backend_options = ProcessGroupHCCL.Options()
        backend_options._timeout = timeout
        backend_class = ProcessGroupHCCL(prefix_store, group_rank, group_size,
                                         backend_options)
        device = torch.device("npu")
        backend_class._set_sequence_number_for_group()
        backend_type = ProcessGroup.BackendType.CUSTOM
        pg._register_backend(device, backend_type, backend_class)
