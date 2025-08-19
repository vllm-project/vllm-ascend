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

import gc
from datetime import timedelta
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import vllm.envs as envs_vllm
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import PrefixStore
from vllm.logger import logger
from vllm.platforms import Platform, PlatformEnum

from vllm_ascend.ascend_config import (check_ascend_config, get_ascend_config,
                                       init_ascend_config)
from vllm_ascend.utils import (ASCEND_QUATIZATION_METHOD, is_310p,
                               update_aclgraph_sizes)

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
    from vllm.utils import FlexibleArgumentParser
else:
    ModelConfig = None
    VllmConfig = None
    FlexibleArgumentParser = None


class NPUPlatform(Platform):

    _enum = PlatformEnum.OOT
    device_name: str = "npu"
    device_type: str = "npu"
    simple_compile_backend: str = "eager"  # Disable torch.compile()
    ray_device_key: str = "NPU"
    device_control_env_var: str = "ASCEND_RT_VISIBLE_DEVICES"
    dispatch_key: str = "PrivateUse1"

    supported_quantization: list[str] = [ASCEND_QUATIZATION_METHOD]

    def is_sleep_mode_available(self) -> bool:
        return True

    @classmethod
    def pre_register_and_update(cls,
                                parser: Optional[FlexibleArgumentParser] = None
                                ) -> None:
        # Adapt the global patch here.
        from vllm_ascend.utils import adapt_patch
        adapt_patch(is_global_patch=True)

        # For online serving, "ascend" quantization method is not a choice natively,
        # so we need to add "ascend" quantization method to quantization methods list
        # and the user can enable quantization using "vllm serve --quantization ascend".
        if parser is not None:
            quant_action = parser._option_string_actions.get('--quantization')
            if quant_action and hasattr(quant_action,
                                        'choices') and quant_action.choices:
                if ASCEND_QUATIZATION_METHOD not in quant_action.choices:
                    quant_action.choices.append(ASCEND_QUATIZATION_METHOD)

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
    def clear_npu_memory(cls):
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        if not envs_vllm.VLLM_USE_V1:
            raise ValueError("vLLM Ascend does not support V0 engine.")
        # initialize ascend config from vllm additional_config
        ascend_config = init_ascend_config(vllm_config)

        from vllm.config import CompilationLevel  # noqa: E402
        compilation_config = vllm_config.compilation_config
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        cache_config = vllm_config.cache_config
        kv_cache_dtype = vllm_config.additional_config.get(
            "kv_cache_dtype", None)
        if kv_cache_dtype is not None:
            vllm_config.cache_config.cache_dtype = kv_cache_dtype

        if model_config is None:
            logger.warning("Model config is missing. This may indicate "
                           "that we are running a test case")
            enforce_eager = False
        else:
            enforce_eager = getattr(model_config, "enforce_eager", False)

        check_ascend_config(vllm_config, enforce_eager)

        if enforce_eager or compilation_config.level == CompilationLevel.NO_COMPILATION:
            logger.info("Compilation disabled, using eager mode by default")
            compilation_config.level = CompilationLevel.NO_COMPILATION
        elif compilation_config.level != CompilationLevel.PIECEWISE:
            logger.warning(
                "NPU does not support %s compilation level. Setting level to NO_COMPILATION",
                compilation_config.level)
            compilation_config.level = CompilationLevel.NO_COMPILATION
        elif ascend_config.torchair_graph_config.enabled:
            logger.info(
                "Torchair compilation enabled on NPU. Setting level to NO_COMPILATION"
            )
            compilation_config.level = CompilationLevel.NO_COMPILATION
        elif parallel_config.distributed_executor_backend == "ray":
            logger.warning(
                "Ray distributed executor backend is not compatible with ACL Graph mode "
                "right now. Setting level to NO_COMPILATION")
            compilation_config.level = CompilationLevel.NO_COMPILATION
        else:
            logger.info(
                "PIECEWISE compilation enabled on NPU. use_inductor not supported - "
                "using only ACL Graph mode")
            compilation_config.use_inductor = False
            compilation_config.splitting_ops.extend(
                ["vllm.unified_ascend_attention_with_output"])
            update_aclgraph_sizes(vllm_config)

        if parallel_config and parallel_config.worker_cls == "auto":
            if ascend_config.torchair_graph_config.enabled:
                parallel_config.worker_cls = "vllm_ascend.torchair.torchair_worker.NPUTorchairWorker"
            else:
                parallel_config.worker_cls = "vllm_ascend.worker.worker_v1.NPUWorker"

        if cache_config:
            if cache_config.block_size is None:
                cache_config.block_size = 128
            if cache_config.enable_prefix_caching and cache_config.block_size != 128:
                logger.warning(
                    "If prefix caching is enabled, block size must be set to 128."
                )
                cache_config.block_size = 128

        # Activate custom ops for v1, except on 310P
        if not is_310p():
            compilation_config.custom_ops = ["all"]

        # If ascend_scheduler_config is enabled,
        # extents original scheduler_config to use AscendScheduler.
        if ascend_config.ascend_scheduler_config.enabled:
            from vllm_ascend.core.schedule_config import AscendSchedulerConfig
            ascend_scheduler_config = AscendSchedulerConfig.initialize_from_config(
                vllm_config.scheduler_config,
                ascend_config.ascend_scheduler_config)
            vllm_config.scheduler_config = ascend_scheduler_config

        if compilation_config.pass_config.enable_sequence_parallelism:
            if not parallel_config.enable_expert_parallel or vllm_config.model_config.hf_config.model_type != "qwen3_moe":
                raise NotImplementedError(
                    "For better performance in Qwen3 MoE, SP only works exclusively with MC2, AllToAll, and AllToAllV."
                )

    @classmethod
    def get_attn_backend_cls(cls,
                             selected_backend,
                             head_size,
                             dtype,
                             kv_cache_dtype,
                             block_size,
                             use_v1,
                             use_mla,
                             has_sink=False):
        if not use_v1:
            raise ValueError("vLLM Ascend does not support V0 engine.")

        use_torchair = get_ascend_config().torchair_graph_config.enabled
        if use_mla:
            return "vllm_ascend.attention.mla_v1.AscendMLABackend"
        elif use_torchair:
            return "vllm_ascend.torchair.torchair_attention.AscendAttentionTorchairBackend"
        else:
            return "vllm_ascend.attention.attention_v1.AscendAttentionBackend"

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

    @classmethod
    def get_piecewise_backend_cls(cls) -> str:
        """
        Get piecewise backend class for piecewise graph.
        """
        return "vllm_ascend.compilation.piecewise_backend.NPUPiecewiseBackend"  # noqa

    @classmethod
    def stateless_init_device_torch_dist_pg(
        cls,
        backend: str,
        prefix_store: PrefixStore,
        group_rank: int,
        group_size: int,
        timeout: timedelta,
    ) -> ProcessGroup:
        from torch.distributed import is_hccl_available
        from torch_npu._C._distributed_c10d import ProcessGroupHCCL

        assert is_hccl_available()

        pg: ProcessGroup = ProcessGroup(
            prefix_store,
            group_rank,
            group_size,
        )

        backend_options = ProcessGroupHCCL.Options()
        backend_options._timeout = timeout

        backend_class = ProcessGroupHCCL(prefix_store, group_rank, group_size,
                                         backend_options)
        device = torch.device("npu")
        # TODO(Yizhou): Like we mentioned above, _set_default_backend is not
        # implemented in the 2.5.1 version of PyTorch. But we need to set it
        # after the latest version is released.
        # pg._set_default_backend(backend_type)
        backend_class._set_sequence_number_for_group()
        backend_type = ProcessGroup.BackendType.CUSTOM

        pg._register_backend(device, backend_type, backend_class)
        return pg
