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
import os
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import vllm.envs as envs_vllm
from vllm.logger import logger
from vllm.platforms import Platform, PlatformEnum

# todo: please remove it when solve cuda hard code in vllm
os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "True"

from vllm_ascend.ascend_config import (check_ascend_config, get_ascend_config,
                                       init_ascend_config)
from vllm_ascend.compilation.utils import (update_compilation_config,
                                           update_compilation_config_v0_11_0)
from vllm_ascend.utils import (ASCEND_QUANTIZATION_METHOD,
                               prefill_context_parallel_enable,
                               vllm_version_is)

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

    supported_quantization: list[str] = [ASCEND_QUANTIZATION_METHOD]

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
                if ASCEND_QUANTIZATION_METHOD not in quant_action.choices:
                    quant_action.choices.append(ASCEND_QUANTIZATION_METHOD)

        from vllm_ascend.quantization.quant_config import \
            AscendQuantConfig  # noqa: F401

    @classmethod
    def get_device_capability(cls, device_id: int = 0):
        return None

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.npu.get_device_name(device_id)

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
        # 1. v0 engine is not supported
        if not envs_vllm.VLLM_USE_V1:
            raise ValueError("vLLM Ascend does not support V0 engine.")
        # 2. initialize ascend config from vllm additional_config
        ascend_config = init_ascend_config(vllm_config)

        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        cache_config = vllm_config.cache_config
        scheduler_config = vllm_config.scheduler_config
        ascend_scheduler_config = ascend_config.ascend_scheduler_config
        structured_outputs_config = vllm_config.structured_outputs_config

        # 3. update cache dtype from additional_config if necessary
        kv_cache_dtype = vllm_config.additional_config.get(
            "kv_cache_dtype", None)
        if kv_cache_dtype is not None:
            vllm_config.cache_config.cache_dtype = kv_cache_dtype
        elif model_config and hasattr(model_config.hf_config, "index_topk"):
            vllm_config.cache_config.cache_dtype = str(
                model_config.dtype).replace("torch.", "")

        # 4. init enforce_eager from model_config
        if model_config is None:
            logger.warning("Model config is missing. This may indicate "
                           "that we are running a test case")
            enforce_eager = False
        else:
            enforce_eager = getattr(model_config, "enforce_eager", False)

        # 5. check ascend config validity
        check_ascend_config(vllm_config, enforce_eager)

        # 6. update compilation config
        if vllm_version_is("0.11.0"):
            update_compilation_config_v0_11_0(vllm_config, ascend_config,
                                              enforce_eager)
        else:
            update_compilation_config(vllm_config, ascend_config,
                                      enforce_eager)

        # 7. update worker cls
        if parallel_config and parallel_config.worker_cls == "auto":
            # TODO: this is a tricky way to disable `use_sequence_parallel_moe` in vllm.
            os.environ["VLLM_ALL2ALL_BACKEND"] = "flashinfer_all2allv"
            if ascend_config.torchair_graph_config.enabled or ascend_config.enable_shared_expert_dp:
                parallel_config.worker_cls = "vllm_ascend.torchair.torchair_worker.NPUTorchairWorker"
            else:
                parallel_config.worker_cls = "vllm_ascend.worker.worker_v1.NPUWorker"

        # 8. set default block size to 128 if not specified
        if cache_config:
            if cache_config.block_size is None:
                cache_config.block_size = 128

            if cache_config.enable_prefix_caching and cache_config.block_size != 128:
                logger.warning(
                    "If prefix caching is enabled, block size must be set to 128."
                )
                cache_config.block_size = 128

        # 9. switch scheduler to ascend scheduler if necessary
        if (model_config is not None and not model_config.use_mla
                and not scheduler_config.async_scheduling
                and model_config.runner_type != "pooling"
                and not prefill_context_parallel_enable()):
            logger.info(
                "Non-MLA LLMs forcibly disable the chunked prefill feature,"
                "as the performance of operators supporting this feature "
                "functionality is currently suboptimal.")
            if not model_config.is_multimodal_model and \
            structured_outputs_config.backend == "auto" and \
            not getattr(scheduler_config, "send_delta_data", False) and \
            not getattr(scheduler_config, "scheduler_delay_factor", 0) > 0 and \
            scheduler_config.policy == "fcfs":
                ascend_scheduler_config.enabled = True
                chunked_prefill_enabled_in_ascend_scheduler = getattr(
                    ascend_scheduler_config, "enable_chunked_prefill", False)
                if chunked_prefill_enabled_in_ascend_scheduler:
                    logger.warning(
                        "Chunked prefill feature is enabled in ascend_scheduler,"
                        "but note that the operator supporting this feature "
                        "would lead to performance degradation.")
                # In this situation, max_num_batched_tokens would have been rewritten.
                # So we must make sure max_num_batched_tokens is not smaller than max_model_len.
                if (scheduler_config.max_num_batched_tokens
                        < scheduler_config.max_model_len
                        and not chunked_prefill_enabled_in_ascend_scheduler):
                    scheduler_config.max_num_batched_tokens = scheduler_config.max_model_len

        # 10. select scheduler, we support ascend scheduler, recompute scheduler and batch scheduler
        if ascend_config.ascend_scheduler_config.enabled:
            from vllm_ascend.core.schedule_config import AscendSchedulerConfig
            ascend_scheduler_config = AscendSchedulerConfig.initialize_from_config(
                vllm_config.scheduler_config,
                ascend_config.ascend_scheduler_config)
            vllm_config.scheduler_config = ascend_scheduler_config
        elif ascend_config.recompute_scheduler_enable:
            from vllm_ascend.core.recompute_schedule_config import \
                RecomputeSchedulerConfig
            recompute_scheduler_config = RecomputeSchedulerConfig.initialize_from_config(
                vllm_config.scheduler_config)
            vllm_config.scheduler_config = recompute_scheduler_config
        # TODO: refactor this part to keep the init logic like ascend scheduler
        if ascend_config.SLO_limits_for_dynamic_batch != -1:
            vllm_config.scheduler_config.scheduler_cls = (
                "vllm_ascend.core.scheduler_dynamic_batch.SchedulerDynamicBatch"
            )
            vllm_config.scheduler_config.chunked_prefill_enabled = True
            vllm_config.scheduler_config.SLO_limits_for_dynamic_batch = ascend_config.SLO_limits_for_dynamic_batch

        # 11. pcp and dcp config check
        if vllm_config.kv_transfer_config is not None and \
            prefill_context_parallel_enable() and \
            cache_config.block_size != parallel_config.cp_kv_cache_interleave_size and \
            parallel_config.decode_context_parallel_size * parallel_config.prefill_context_parallel_size > 1:
            raise AssertionError(
                f"cp_kv_cache_interleave_size({parallel_config.cp_kv_cache_interleave_size}) "
                f"and block_size({cache_config.block_size}) "
                "needs to be equal if use cp or dcp > 1 in P/D disaggregate scenario."
            )

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend,
        head_size,
        dtype,
        kv_cache_dtype,
        block_size,
        use_v1,
        use_mla,
        has_sink=False,
        use_sparse=False,
    ):
        if not use_v1:
            raise ValueError("vLLM Ascend does not support V0 engine.")

        ascend_config = get_ascend_config()

        if use_mla and ascend_config.enable_shared_expert_dp:
            if use_mla and not use_sparse:
                return "vllm_ascend.torchair.torchair_mla.AscendMLATorchairBackend"
            if use_mla and use_sparse:
                return "vllm_ascend.torchair.torchair_sfa.AscendSFATorchairBackend"

        use_torchair = ascend_config.torchair_graph_config.enabled
        # choose attention backend based on use_mla and use_torchair
        backend_map = {
            (True, False, True):
            "vllm_ascend.torchair.torchair_mla.AscendMLATorchairBackend",
            (True, False, False):
            "vllm_ascend.attention.mla_v1.AscendMLABackend",
            (False, False, True):
            "vllm_ascend.torchair.torchair_attention.AscendAttentionTorchairBackend",
            (False, False, False):
            "vllm_ascend.attention.attention_v1.AscendAttentionBackend",
            (True, True, False):
            "vllm_ascend.attention.sfa_v1.AscendSFABackend",
            (True, True, True):
            "vllm_ascend.torchair.torchair_sfa.AscendSFATorchairBackend",
        }
        return backend_map[(use_mla, use_sparse, use_torchair)]

    @classmethod
    def get_punica_wrapper(cls) -> str:
        if vllm_version_is("0.11.0"):
            return "vllm_ascend.lora.punica_npu.PunicaWrapperNPU0110"
        else:
            return "vllm_ascend.lora.punica_npu.PunicaWrapperNPU"

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
    def opaque_attention_op(cls) -> bool:
        return True

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        """
        Get piecewise backend class for piecewise graph.
        """
        return "vllm_ascend.compilation.acl_graph.ACLGraphWrapper"  # noqa

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        return True
