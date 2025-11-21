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

import logging
import os
from typing import Any, Callable, List, Optional, Tuple, Type

import torch
import vllm.envs as envs
from omni.adaptors.vllm.utils import SUPPORTED_QUANTIZATION_METHODS
from torch.library import Library
from vllm import utils
from vllm.logger import logger
from vllm.platforms import Platform, PlatformEnum
from vllm.utils import FlexibleArgumentParser, supports_dynamo, vllm_lib

CUSTOM_OP_ENABLED = False  # Custom operations not enabled for Omni inference

#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
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
# Adapted from vllm/model_executor/models/qwen2_vl.py
# This file is a part of the vllm-ascend project.

import torch
import vllm
import vllm.distributed
from torch.distributed import ProcessGroup

origin_pg_init = ProcessGroup.__init__
origin_destroy_model_parallel = None
origin_stateless_init_dp_group = None

def ascend_destroy_model_parallel():
    """Set the groups to none and destroy them."""

    origin_destroy_model_parallel()
    from omni.adaptors.vllm.distributed.parallel_state import \
        destory_ascend_model_parallel
    destory_ascend_model_parallel()

def pg_patched_init(self, *args, **kwargs):
    options = ProcessGroup.Options(backend="gloo")
    origin_pg_init(self, *args, options)

def noops(*args):
    pass

def init_dp_group(self) -> ProcessGroup:
    from vllm.config import ParallelConfig
    ProcessGroup.__init__ = pg_patched_init
    ProcessGroup._set_default_backend = noops
    pg = origin_stateless_init_dp_group(self)
    ProcessGroup.__init__ = origin_pg_init
    return pg

def update_parallel_state():
    global origin_destroy_model_parallel
    if origin_destroy_model_parallel == None:
        origin_destroy_model_parallel = vllm.distributed.parallel_state.destroy_model_parallel

    vllm.distributed.parallel_state.destroy_model_parallel = ascend_destroy_model_parallel

    if torch.__version__ >= '2.5.1' and torch.__version__ < "2.6":
        from vllm.config import ParallelConfig
        global origin_stateless_init_dp_group
        if origin_stateless_init_dp_group == None:
            origin_stateless_init_dp_group = ParallelConfig.stateless_init_dp_group

        ParallelConfig.stateless_init_dp_group = init_dp_group


def ascend_direct_register_custom_op(
        op_name: str,
        op_func: Callable,
        mutates_args: list[str],
        fake_impl: Optional[Callable] = None,
        target_lib: Optional[Library] = None,
        dispatch_key: str = "CUDA",
        tags: Tuple[torch.Tag, ...] = (),
):
    # In pytorch 2.5.1, torch.library.infer_schema require the input function to
    # have annotations supported by typing library. But in pytorch 2.7.0 which
    # vllm using, torch.library.infer_schema require the python builtin type. In
    # this case, we should revert built type to typing type for 2.5.1 backward
    # compatibility.
    for k, v in op_func.__annotations__.items():
        if v == list[int]:
            op_func.__annotations__[k] = List[int]
        if v == Optional[list[int]]:
            op_func.__annotations__[k] = Optional[List[int]]
    import torch.library
    schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
    my_lib = target_lib or vllm_lib
    my_lib.define(op_name + schema_str, tags=tags)
    my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)


def update_utils_custom_op():
    utils.direct_register_custom_op = ascend_direct_register_custom_op


def enable_overwrite_request_id():
    """Patch OpenAIServing to use random UUIDs for request IDs."""
    from fastapi import Request
    from vllm.entrypoints.openai.serving_engine import OpenAIServing
    from vllm.utils import random_uuid
    @staticmethod
    def _base_request_id(raw_request: Optional[Request], 
                         default: Optional[str] = None) -> Optional[str]:
        return default or random_uuid()

    OpenAIServing._base_request_id = _base_request_id
    logging.info("Applied patch: overwrite_request_id")


def apply_constant_list_patch():
      import vllm.v1.utils as vllm_utils
      cls = vllm_utils.ConstantList
      def __hash__(self):
            return hash(self._x)
      def __eq__(self, other):
            return isinstance(other, cls) and self._x == other._x
      cls.__hash__ = __hash__
      cls.__eq__ = __eq__
      cls._is_patched = True


def apply_kv_cache_patch():
    import vllm.v1.core.kv_cache_utils as kv_cache_utils
    from vllm.v1.core.kv_cache_utils import BlockHashType
    logger.info("apply_kv_cache_hash_patch")
    NONE_HASH = kv_cache_utils.NONE_HASH               
    def patched(hash_function,
                    parent_block_hash,
                    curr_block_token_ids,
                    extra_keys):
        if not parent_block_hash:
                parent_block_hash = NONE_HASH             
        curr_block_token_ids_tuple = tuple(curr_block_token_ids)
        h = hash_function((parent_block_hash, curr_block_token_ids_tuple, 0))
        return BlockHashType(
                h,                                                      
                curr_block_token_ids_tuple,
                extra_keys
        )
    kv_cache_utils.hash_block_tokens = patched
    patched.is_patched = True

    
def register() -> str:
    """Register the NPU platform for vLLM.

    Returns:
        str: The module path to the NPUPlatform class.
    """
    return "omni.adaptors.vllm.platform.NPUPlatform"


def ensure_v1_engine() -> None:
    """Ensure the V1 engine is used, raising an error otherwise."""
    if not envs.VLLM_USE_V1:
        raise RuntimeError("Omni inference requires the V1 engine.")


ensure_v1_engine()


class EnvironmentSetup:
    """Handles environment variable setup for NPU platform compatibility."""

    @staticmethod
    def configure_visible_devices() -> None:
        """Configure ASCEND_RT_VISIBLE_DEVICES for NPU compatibility."""
        os.environ["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"] = "1"
        if "ASCEND_RT_VISIBLE_DEVICES_BK" not in os.environ:
            visible_devices = os.getenv("ASCEND_RT_VISIBLE_DEVICES")
            if visible_devices is not None:
                os.environ["ASCEND_RT_VISIBLE_DEVICES_BK"] = visible_devices
            else:
                num_devices = torch.npu.device_count()
                os.environ["ASCEND_RT_VISIBLE_DEVICES_BK"] = ",".join(str(i) for i in range(num_devices))


class ConfigUpdater:
    """Handles configuration validation and updates for the NPU platform."""

    @staticmethod
    def update_parser(parser: Optional[FlexibleArgumentParser]) -> None:
        """Update the argument parser to include NPU-specific quantization options.

        Args:
            parser: The argument parser to update, if provided.
        """
        if parser is None:
            return

    @classmethod
    def update_vllm_config(cls, vllm_config: 'VllmConfig') -> None:
        """Update the vLLM configuration for NPU compatibility.

        Args:
            vllm_config: The vLLM configuration to update.
        """
        from omni.adaptors.vllm.compilation.compile_config import \
            NPUCompilationConfig
        additional_config = vllm_config.additional_config
        vllm_config.npu_compilation_config = NPUCompilationConfig()
        if additional_config:
            graph_model_compile_config = additional_config.get("graph_model_compile_config", {})
            vllm_config.npu_compilation_config.build_from_cli(graph_model_compile_config, vllm_config)
            logger.debug(f"Graph model compile config: {graph_model_compile_config}")
            cls._handle_graph_mode(vllm_config)

        cls._update_parallel_config(vllm_config)
        cls._update_cache_config(vllm_config)
        cls._enable_custom_ops(vllm_config)
        cls._may_enable_omni_attn(vllm_config)

    @staticmethod
    def _handle_graph_mode(vllm_config: 'VllmConfig') -> None:
        from vllm.config import CompilationLevel
        """Handle graph mode configuration for NPU."""
        enable_graph_mode = (vllm_config.npu_compilation_config.level != CompilationLevel.NO_COMPILATION)
        if not enable_graph_mode:
            return

        if not supports_dynamo():
            logger.warning("Graph mode unsupported due to low torch version. Disabling.")
            vllm_config.npu_compilation_config.level = CompilationLevel.NO_COMPILATION
        if envs.VLLM_USE_V1 and envs.VLLM_MLA_DISABLE:
            logger.warning("Graph mode not supported for V1 without MLA. Disabling.")
            vllm_config.npu_compilation_config.level = CompilationLevel.NO_COMPILATION
        if int(os.getenv("RANDOM_MODE", default='0')) or int(os.getenv("CAPTURE_MODE", default='0')) or int(
                os.getenv("REPLAY_MODE", default='0')):
            logger.warning("Graph mode not supported for MOCK mode. Disabling.")
            vllm_config.npu_compilation_config.level = CompilationLevel.NO_COMPILATION

    @staticmethod
    def _update_parallel_config(vllm_config: 'VllmConfig') -> None:
        """Update parallel configuration for NPU worker compatibility."""
        parallel_config = vllm_config.parallel_config
        if parallel_config and parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "omni.adaptors.vllm.worker.npu_worker.NPUWorker"

    @staticmethod
    def _update_cache_config(vllm_config: 'VllmConfig') -> None:
        """Update cache configuration for NPU compatibility."""
        cache_config = vllm_config.cache_config
        if not cache_config:
            return
        if cache_config.block_size is None:
            cache_config.block_size = 128
        if cache_config.enable_prefix_caching and cache_config.block_size != 128:
            logger.warning("Prefix caching requires block size 128. Setting block size to 128.")
            cache_config.block_size = 128

    @staticmethod
    def _enable_custom_ops(vllm_config: 'VllmConfig') -> None:
        vllm_config.compilation_config.custom_ops = ["all"]

    @staticmethod
    def _may_enable_omni_attn(vllm_config: 'VllmConfig') -> None:
        if vllm_config.additional_config is None:
            return
        from omni.accelerators.cache import (apply_omni_attn_patch,
                                             check_omni_attn_cmd_arg)
        enable_omni_attn = check_omni_attn_cmd_arg(vllm_config.additional_config)
        kv_transfer_config = vllm_config.kv_transfer_config
        is_kv_consumer = kv_transfer_config is None or kv_transfer_config.kv_role == 'kv_consumer'
        omni_attn_config = vllm_config.additional_config.get("omni_attn_config", None)
        apply_omni_attn_patch(enable=enable_omni_attn, is_kv_consumer=is_kv_consumer, config=omni_attn_config)


class NPUPlatform(Platform):
    """Platform implementation for NPU devices in vLLM."""

    _enum = PlatformEnum.OOT
    device_name: str = "npu"
    device_type: str = "npu"
    simple_compile_backend: str = "eager"
    ray_device_key: str = "NPU"
    device_control_env_var: str = "ASCEND_RT_VISIBLE_DEVICES"
    dispatch_key: str = "PrivateUse1"
    supported_quantization: list[str] = SUPPORTED_QUANTIZATION_METHODS

    def __init__(self):
        """Initialize the NPU platform and configure environment."""
        EnvironmentSetup.configure_visible_devices()
        update_utils_custom_op()
        super().__init__()

    @classmethod
    def is_sleep_mode_available(cls) -> bool:
        """Check if sleep mode is available for NPU.

        Returns:
            bool: Always True for NPU.
        """
        import omni.adaptors.vllm.envs as envs_ascend
        if not envs_ascend.VLLM_ENABLE_SLEEP_MODE:
            return False

        return True

    @classmethod
    def pre_register_and_update(cls, parser: Optional[FlexibleArgumentParser] = None) -> None:
        """Perform pre-registration tasks and update parser.

        Args:
            parser: Optional argument parser to update with NPU-specific options.
        """
        ConfigUpdater.update_parser(parser)
        update_parallel_state()
        if os.getenv("ENABLE_APC_EVENT", "0") == "1":
            apply_kv_cache_patch()
            apply_constant_list_patch()
        if os.getenv("ENABLE_OVERWRITE_REQ_IDS", "0") == "1":
            enable_overwrite_request_id()
        import omni.quantization  # noqa: F401
        from omni.adaptors.vllm.ems.ems_env import EmsEnv
        if EmsEnv.enable_vllm_ems:
            from omni.adaptors.vllm.patches import ems_patch

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> None:
        """Get device capability for the specified NPU.

        Args:
            device_id: The ID of the NPU device.

        Returns:
            None: Capability information not available for NPU.
        """
        return None

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """Get the name of the specified NPU device.

        Args:
            device_id: The ID of the NPU device.

        Returns:
            str: The name of the device.
        """
        return torch.npu.get_device_name(device_id)

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        """Check if asynchronous output is supported.

        Args:
            enforce_eager: Whether eager mode is enforced.

        Returns:
            bool: Always True for NPU.
        """
        return True

    @classmethod
    def inference_mode(cls) -> torch.inference_mode:
        """Get the inference mode context manager for NPU.

        Returns:
            torch.inference_mode: The inference mode context.
        """
        return torch.inference_mode()

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """Set the active NPU device.

        Args:
            device: The torch device to set.
        """
        visible_device_backup = os.getenv("ASCEND_RT_VISIBLE_DEVICES_BK")
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = visible_device_backup
        torch.npu.set_device(device)

    @classmethod
    def empty_cache(cls) -> None:
        """Clear the NPU memory cache."""
        torch.npu.empty_cache()

    @classmethod
    def synchronize(cls) -> None:
        """Synchronize all NPU operations."""
        torch.npu.synchronize()

    @classmethod
    def mem_get_info(cls) -> Tuple[int, int]:
        """Get memory information for the current NPU.

        Returns:
            Tuple[int, int]: Free and total memory in bytes.
        """
        return torch.npu.mem_get_info()

    @classmethod
    def check_and_update_config(cls, vllm_config: 'VllmConfig') -> None:
        """Update the vLLM configuration for NPU compatibility.

        Args:
            vllm_config: The vLLM configuration to update.
        """
        additional_config = vllm_config.additional_config
        if additional_config and vllm_config.additional_config.get("enable_hybrid_graph_mode", False):
            from omni.adaptors.vllm.worker.npu_schedule import \
                HybridSchedulerConfig
            ascend_scheduler_config = HybridSchedulerConfig.initialize_from_config(
                vllm_config.scheduler_config)
            vllm_config.scheduler_config = ascend_scheduler_config
            logger.info("--------enbale hybrid graph mode----------------")
            
        ConfigUpdater.update_vllm_config(vllm_config)

    @classmethod
    def get_attn_backend_cls(cls, selected_backend: str, head_size: int, dtype: torch.dtype,
                             kv_cache_dtype: torch.dtype, block_size: int, use_v1: bool,
                             use_mla: bool) -> str:
        """Get the attention backend class for the NPU.

        Args:
            selected_backend: The selected backend name.
            head_size: Size of the attention head.
            dtype: Data type for the model.
            kv_cache_dtype: Data type for the KV cache.
            block_size: Block size for attention.
            use_v1: Whether V1 engine is used.
            use_mla: Whether multi-layer attention is used.

        Returns:
            str: The module path to the attention backend class.
        """
        ensure_v1_engine()
        return ("omni.models.common.layers.attention.backend.mla.AscendMLABackend" if use_mla
                else "omni.models.common.layers.attention.backend.attention.AscendAttentionBackend")

    @classmethod
    def get_punica_wrapper(cls) -> str:
        """Get the Punica wrapper for LoRA support.

        Returns:
            str: The module path to the Punica wrapper.
        """
        return "vllm.lora.punica_wrapper.punica_cpu.PunicaWrapperCPU"

    @classmethod
    def get_current_memory_usage(cls, device: Optional[torch.types.Device] = None) -> float:
        """Get the current memory usage for the specified NPU.

        Args:
            device: Optional device to query.

        Returns:
            float: Maximum memory allocated in bytes.
        """
        torch.npu.reset_peak_memory_stats(device)
        return torch.npu.max_memory_allocated(device)

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        """Get the device communicator class for NPU.

        Returns:
            str: The module path to the communicator class.
        """
        return "omni.adaptors.vllm.distributed.communicator.NPUCommunicator"

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        """Check if pinned memory is available.

        Returns:
            bool: Always True for NPU unless mock (no NPU).
        """
        return not int(os.getenv("NO_NPU_MOCK", "0"))

    @classmethod
    def supports_v1(cls, model_config: 'ModelConfig') -> bool:
        """Check if the V1 engine is supported.

        Args:
            model_config: The model configuration.

        Returns:
            bool: Always True for NPU.
        """
        return True
