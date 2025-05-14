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
# Adapted from vllm/config.py
# This file is a part of the vllm-ascend project.

import torch
import vllm.envs as envs
from vllm.config import CompilationConfig, CompilationLevel, VllmConfig
from vllm.logger import init_logger
from vllm.utils import random_uuid

logger = init_logger(__name__)


def __post_init__(self):
    """Verify configs are valid & consistent with each other.
    """
    if self.model_config is not None:
        self.model_config.verify_async_output_proc(self.parallel_config,
                                                   self.speculative_config,
                                                   self.device_config)
        self.model_config.verify_with_parallel_config(self.parallel_config)
        self.model_config.verify_dual_chunk_attention_config(self.load_config)

    if self.cache_config is not None:
        self.cache_config.verify_with_parallel_config(self.parallel_config)

    if self.lora_config:
        self.lora_config.verify_with_cache_config(self.cache_config)
        self.lora_config.verify_with_model_config(self.model_config)
        self.lora_config.verify_lora_support()
    if self.prompt_adapter_config:
        self.prompt_adapter_config.verify_with_model_config(self.model_config)

    if self.quant_config is None and \
        self.model_config is not None and self.load_config is not None:
        self.quant_config = VllmConfig._get_quantization_config(
            self.model_config, self.load_config)

    from vllm.platforms import current_platform
    if self.scheduler_config is not None and \
        self.model_config is not None and \
        self.scheduler_config.chunked_prefill_enabled and \
        self.model_config.dtype == torch.float32 and \
        current_platform.get_device_capability() == (7, 5):
        logger.warning_once(
            "Turing devices tensor cores do not support float32 matmul. "
            "To workaround this limitation, vLLM will set 'ieee' input "
            "precision for chunked prefill triton kernels.")

    if self.compilation_config is None:
        self.compilation_config = CompilationConfig()
    if self.compilation_config.pass_config.enable_sequence_parallelism:
        self.compilation_config.custom_ops.append("+rms_norm")
    if envs.VLLM_USE_V1 and self.model_config is not None and \
        not self.model_config.enforce_eager:
        # NOTE(woosuk): Currently, we use inductor because the piecewise
        # CUDA graphs do not work properly with the custom CUDA kernels.
        # FIXME(woosuk): Disable inductor to reduce the compilation time
        # and avoid any potential issues with the inductor.
        # FIXME(rob): Add function to set all of these.
        if not self.compilation_config.custom_ops:
            self.compilation_config.custom_ops = ["none"]
        self.compilation_config.use_cudagraph = True
        self.compilation_config.use_inductor = True
        self.compilation_config.cudagraph_num_of_warmups = 1
        self.compilation_config.pass_config.enable_fusion = False
        self.compilation_config.pass_config.enable_noop = False
        self.compilation_config.level = CompilationLevel.PIECEWISE
        self.compilation_config.set_splitting_ops_for_v1()

    if self.parallel_config is not None and \
        self.parallel_config.tensor_parallel_size > 1 and \
        self.parallel_config.pipeline_parallel_size > 1 and \
        self.compilation_config is not None and \
            self.compilation_config.pass_config is not None and \
        self.compilation_config.pass_config.enable_sequence_parallelism:
        logger.warning_once(
            "Sequence parallelism is not supported with pipeline "
            "parallelism. Disabling sequence parallelism.")
        self.compilation_config.pass_config.\
            enable_sequence_parallelism = False

    self._set_cudagraph_sizes()

    if self.cache_config is not None and \
        self.cache_config.cpu_offload_gb > 0 and \
        self.compilation_config.level != CompilationLevel.NO_COMPILATION \
            and not envs.VLLM_USE_V1:
        logger.warning(
            "CPU offload is not supported with `torch.compile` in v0 yet."
            " Disabling `torch.compile`.")
        self.compilation_config.level = CompilationLevel.NO_COMPILATION

    if ((not envs.VLLM_USE_V1) and self.lora_config is not None and
            self.compilation_config.level != CompilationLevel.NO_COMPILATION):
        logger.warning(
            "LoRA for V0 is not supported with `torch.compile` yet. "
            "Disabling `torch.compile`.")
        self.compilation_config.level = CompilationLevel.NO_COMPILATION

    if self.compilation_config.full_cuda_graph and \
        not self.model_config.disable_cascade_attn:
        logger.warning_once("full_cuda_graph is not supported with "
                            "cascade attention. Disabling cascade attention.")
        self.model_config.disable_cascade_attn = True

    if self.model_config and self.model_config.use_mla and \
        not (current_platform.is_cuda() or current_platform.is_rocm()):
        logger.info(
            "MLA is enabled on a non-GPU and NPU platform; just forcing "
            "prefix caching to be disabled.")

        if self.cache_config is not None:
            self.cache_config.enable_prefix_caching = False

    if (self.kv_events_config and self.kv_events_config.enable_kv_cache_events
            and not self.cache_config.enable_prefix_caching):
        logger.warning(
            "KV cache events are on, but prefix caching is not enabled."
            "Use --enable-prefix-caching to enable.")
    if (self.kv_events_config and self.kv_events_config.publisher != "null"
            and not self.kv_events_config.enable_kv_cache_events):
        logger.warning("KV cache events are disabled,"
                       "but the scheduler is configured to publish them."
                       "Modify KVEventsConfig.enable_kv_cache_events"
                       "to True to enable.")
    current_platform.check_and_update_config(self)

    if not self.instance_id:
        self.instance_id = random_uuid()[:5]


VllmConfig.__post_init__ = __post_init__
