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
from vllm.logger import logger
from vllm.platforms import Platform, PlatformEnum

# todo: please remove it when solve cuda hard code in vllm
os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "1"

from vllm_ascend.ascend_config import (check_ascend_config, get_ascend_config,
                                       init_ascend_config)
from vllm_ascend.torchair.utils import (check_torchair_cache_exist,
                                        delete_torchair_cache_file)

# isort: off
from vllm_ascend.utils import (
    ASCEND_QUANTIZATION_METHOD, COMPRESSED_TENSORS_METHOD, AscendDeviceType,
    enable_sp, get_ascend_device_type, is_vl_model,
    prefill_context_parallel_enable, update_aclgraph_sizes,
    update_cudagraph_capture_sizes, update_default_aclgraph_sizes)

# set custom ops path
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
CUSTOM_OPP_PATH = os.path.join(CUR_DIR, "vllm_ascend", "_cann_ops_custom",
                               "vendors", "customize")
CUSTOM_LIB_PATH = os.path.join(CUSTOM_OPP_PATH, "op_api", "lib")

if os.path.exists(CUSTOM_OPP_PATH):
    current_cust_opp_path = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    if current_cust_opp_path:
        os.environ[
            "ASCEND_CUSTOM_OPP_PATH"] = f"{CUSTOM_OPP_PATH}:{current_cust_opp_path}"
    else:
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = CUSTOM_OPP_PATH

if os.path.exists(CUSTOM_LIB_PATH):
    current_lib_path = os.environ.get("LD_LIBRARY_PATH", "")
    if current_lib_path:
        os.environ["LD_LIBRARY_PATH"] = f"{CUSTOM_LIB_PATH}:{current_lib_path}"
    else:
        os.environ["LD_LIBRARY_PATH"] = CUSTOM_LIB_PATH

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

    supported_quantization: list[str] = [
        ASCEND_QUANTIZATION_METHOD, COMPRESSED_TENSORS_METHOD
    ]

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

        from vllm_ascend.quantization.compressed_tensors.compressed_tensors import \
            AscendCompressedTensorsConfig  # noqa: F401
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
        # initialize ascend config from vllm additional_config
        ascend_config = init_ascend_config(vllm_config)

        from vllm.config import CompilationMode  # noqa: E402

        compilation_config = vllm_config.compilation_config
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        cache_config = vllm_config.cache_config
        ascend_scheduler_config = ascend_config.ascend_scheduler_config

        kv_cache_dtype = vllm_config.additional_config.get(
            "kv_cache_dtype", None)
        if kv_cache_dtype is not None:
            vllm_config.cache_config.cache_dtype = kv_cache_dtype
        elif model_config and hasattr(model_config.hf_config, "index_topk"):
            vllm_config.cache_config.cache_dtype = str(
                model_config.dtype).replace("torch.", "")
        if model_config is None:
            logger.warning("Model config is missing. This may indicate "
                           "that we are running a test case")
            enforce_eager = False
        else:
            enforce_eager = getattr(model_config, "enforce_eager", False)

        check_ascend_config(vllm_config, enforce_eager)
        from vllm.config.compilation import CUDAGraphMode
        if enforce_eager:
            logger.info("Compilation disabled, using eager mode by default")
            compilation_config.mode = CompilationMode.NONE
            if compilation_config.splitting_ops is None:
                compilation_config.splitting_ops = []

        compilation_config.cudagraph_num_of_warmups = 1

        if compilation_config.mode not in [
                CompilationMode.NONE, CompilationMode.VLLM_COMPILE
        ]:
            logger.warning(
                "NPU does not support %s compilation mode. Setting CUDAGraphMode to NONE",
                compilation_config.mode)
            compilation_config.cudagraph_mode = CUDAGraphMode.NONE

        # set CUDAGraphMode to None when torchair is enabled, no mather what compilation_config.level is.
        if ascend_config.torchair_graph_config.enabled:
            logger.info(
                "Torchair compilation enabled on NPU. Setting CUDAGraphMode to NONE"
            )
            compilation_config.cudagraph_mode = CUDAGraphMode.NONE
            # Note: We delete the torchair cache folder here to prevent runtime issues caused by dimension
            # mismatches or configuration inconsistencies when users reuse cached computation graphs. Though
            # this will increase graph compilation duration, it significantly enhances robustness and decreases
            # graph launching time during inference.
            if check_torchair_cache_exist(
            ) and not ascend_config.torchair_graph_config.use_cached_kv_cache_bytes:
                logger.warning(
                    "Torchair cache folder is deleted here to prevent runtime issues caused by dimension "
                    "mismatches or configuration inconsistencies when users reuse cached computation graphs. "
                    "In order to decrease torchair graph compilation time, users can enable both use_cached_graph "
                    "and use_cached_kv_cache_bytes in torchair_graph_config.")
                delete_torchair_cache_file()

        # set cudaprah sizes before extending `compilation_config.splitting_ops`
        vllm_config._set_cudagraph_sizes()
        # There are cases where default cudagraph_capture_sizes are not friendly
        # to ascend ops && hardwares. We update these sizes here to improve
        # default performance.
        update_default_aclgraph_sizes(vllm_config)
        # TODO delete graph size update here when compilation_config.pass_config.enable_sequence_parallelism
        # is supported by vllm-ascend.
        if vllm_config.parallel_config.tensor_parallel_size > 1 and not vllm_config.model_config.enforce_eager and \
                enable_sp(vllm_config):
            original_sizes = compilation_config.cudagraph_capture_sizes
            sp_aclgraph_sizes = \
                vllm_config.update_sizes_for_sequence_parallelism(original_sizes)
            assert sp_aclgraph_sizes, (
                f"cudagraph_capture_sizes {original_sizes} does not contain"
                f"values that are multiples of tp_size "
                f"{vllm_config.parallel_config.tensor_parallel_size}")
            if len(sp_aclgraph_sizes) != len(original_sizes):
                compilation_config.cudagraph_capture_sizes = sp_aclgraph_sizes
                update_cudagraph_capture_sizes(vllm_config, sp_aclgraph_sizes)

        # TODO: Full graph is fully supported later, and the default value will be set to full graph.
        if compilation_config.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE:
            compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE

        if compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
            compilation_config.mode = CompilationMode.NONE
        elif compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE:
            logger.info(
                "PIECEWISE compilation enabled on NPU. use_inductor not supported - "
                "using only ACL Graph mode")
            assert compilation_config.mode == CompilationMode.VLLM_COMPILE, \
                "When enabling VLLM_COMPILE aclgraph, please make sure compilation_config.mode == CompilationMode.VLLM_COMPILE and compilation_config.cudagraph_mode == CUDAGraphMode.VLLM_COMPILE"
            compilation_config.set_splitting_ops_for_v1()
            compilation_config.use_inductor = False
            compilation_config.splitting_ops.extend(["vllm::mla_forward"])
            update_aclgraph_sizes(vllm_config)
        elif compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY or\
            compilation_config.cudagraph_mode == CUDAGraphMode.FULL:
            logger.info(
                "FULL_DECODE_ONLY compilation enabled on NPU. use_inductor not supported - "
                "using only ACL Graph mode")
            compilation_config.use_inductor = False
            warning_message = """\033[91m
            **********************************************************************************
            * WARNING: You have enabled the *full graph* feature.
            * This is an early experimental stage and may involve various unknown issues.
            * A known problem is that capturing too many batch sizes can lead to OOM
            * (Out of Memory) errors or inference hangs. If you encounter such issues,
            * consider reducing `gpu_memory_utilization` or manually specifying a smaller
            * batch size for graph capture.
            * For more details, please refer to:
            * https://docs.vllm.ai/en/stable/configuration/conserving_memory.html#reduce-cuda-graphs
            **********************************************************************************\033[0m
            """
            logger.warning(warning_message)
        else:
            logger.info(
                "%s cudagraph_mode is not support on NPU. falling back to NONE",
                compilation_config.cudagraph_mode)
            compilation_config.cudagraph_mode = CUDAGraphMode.NONE
            compilation_config.mode = CompilationMode.NONE

        # TODO: Remove this check when ACL Graph supports ASCEND_LAUNCH_BLOCKING=1
        # Then, we will have to discuss the error handling strategy and user experience
        if compilation_config.cudagraph_mode != CUDAGraphMode.NONE and \
            os.environ.get("ASCEND_LAUNCH_BLOCKING", "0") == "1":
            raise ValueError(
                "ACL graph is incompatible with ASCEND_LAUNCH_BLOCKING=1. "
                "Please unset ASCEND_LAUNCH_BLOCKING or set it to 0. If you "
                "need ASCEND_LAUNCH_BLOCKING for debugging, consider other methods — "
                "for example, check the plog files (default: $HOME/ascend/log/debug) "
                "for more information about runtime errors.")

        if parallel_config and parallel_config.worker_cls == "auto":
            # TODO: this is a tricky way to disable `use_sequence_parallel_moe` in vllm.
            parallel_config.all2all_backend = "flashinfer_all2allv"
            if ascend_config.torchair_graph_config.enabled or ascend_config.enable_shared_expert_dp:
                parallel_config.worker_cls = "vllm_ascend.torchair.torchair_worker.NPUTorchairWorker"
            else:
                parallel_config.worker_cls = "vllm_ascend.worker.worker_v1.NPUWorker"

        if cache_config:
            if cache_config.block_size is None:
                cache_config.block_size = 128

            if cache_config.enable_prefix_caching or \
                not ascend_scheduler_config.enabled or \
                getattr(ascend_scheduler_config, "enable_chunked_prefill", False):
                logger.warning(
                    "If chunked prefill or prefix caching is enabled, block size must be set to 128."
                )
                origin_block_size = cache_config.block_size
                cache_config.block_size = 128
                # TODO(MengqingCao): Remove the model_type check, after resolving the hidden error in get_kv_cache_groups.
                if model_config and model_config.hf_config.model_type == "qwen3_next":
                    logger.warning(
                        "When running qwen3-next model, block_size needs to be restored to its original value."
                    )
                    cache_config.block_size = origin_block_size

        # Activate custom ops for v1, except on 310P
        if get_ascend_device_type() != AscendDeviceType._310P:
            compilation_config.custom_ops = ["all"]

        # If ascend_scheduler_config is enabled,
        # extents original scheduler_config to use AscendScheduler.
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

        # Extend original scheduler_config to use SchedulerDynamicBatch.
        if ascend_config.SLO_limits_for_dynamic_batch != -1:
            vllm_config.scheduler_config.scheduler_cls = (
                "vllm_ascend.core.scheduler_dynamic_batch.SchedulerDynamicBatch"
            )
            vllm_config.scheduler_config.chunked_prefill_enabled = True
            vllm_config.scheduler_config.SLO_limits_for_dynamic_batch = ascend_config.SLO_limits_for_dynamic_batch

        if vllm_config.kv_transfer_config is not None and \
            prefill_context_parallel_enable() and \
            cache_config.block_size != parallel_config.cp_kv_cache_interleave_size and \
            parallel_config.decode_context_parallel_size * parallel_config.prefill_context_parallel_size > 1:
            raise AssertionError(
                f"cp_kv_cache_interleave_size({parallel_config.cp_kv_cache_interleave_size}) "
                f"and block_size({cache_config.block_size}) "
                "needs to be equal if use cp or dcp > 1 in P/D disaggregate scenario."
            )

        if is_vl_model(vllm_config):
            if bool(int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM", '0'))) or \
               bool(int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", '0'))):
                raise ValueError(
                    "Currently, VL models doesn't support "
                    "FLASHCOMM in vllm-ascend. We will fix this in the future. "
                    "Please set VLLM_ASCEND_ENABLE_FLASHCOMM1=0.")

    @classmethod
    def import_kernels(cls) -> None:
        # Directly importing vllm_ascend_C prevents ASCEND_RT_VISIBLE_DEVICES
        # from being applied during runtime initialization, which causes bugs
        # in the RL module. Therefore, we currently use lazy initialization
        # to avoid this issue. See https://github.com/vllm-project/vllm-ascend/pull/884.
        # TODO: when the above issue is fixed, we can uncomment the following lines.
        # from vllm_ascend.utils import enable_custom_op
        # enable_custom_op()
        pass

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend,
        head_size,
        dtype,
        kv_cache_dtype,
        block_size,
        use_mla,
        has_sink=False,
        use_sparse=False,
        attn_type: str | None = None,
    ):
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
