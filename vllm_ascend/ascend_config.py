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
from typing import Optional

import vllm.envs as envs
from vllm.logger import logger


class AscendConfig:
    """
    Configuration Object for additional_config from vllm.configs.
    """

    def __init__(self, vllm_config):
        additional_config = vllm_config.additional_config if vllm_config.additional_config is not None else {}

        torchair_graph_config = additional_config.get("torchair_graph_config",
                                                      {})
        self.torchair_graph_config = TorchairGraphConfig(
            torchair_graph_config
        )  # The config options for torchair graph mode.

        ascend_scheduler_config = additional_config.get(
            "ascend_scheduler_config", {})
        self.ascend_scheduler_config = AscendSchedulerConfig(
            ascend_scheduler_config
        )  # The config options for ascend scheduler.

        self.expert_map_path = additional_config.get(
            "expert_map_path", None
        )  # When using expert load balancing for the MOE model, an expert map path needs to be passed in
        self.dynamic_eplb = additional_config.get(
            "dynamic_eplb",
            False)  # Whether to enable dynamic expert load balancing
        self.num_iterations_eplb_update = additional_config.get(
            "num_iterations_eplb_update", 400
        )  # Frenquency (in iterations) at which EPLB recalculates and redistributes expert loads
        self.gate_eplb = additional_config.get(
            "gate_eplb", False
        )  # If set to True, the EPLB process will run only once; If False, it will execute periodically
        self.num_wait_worker_iterations = additional_config.get(
            "num_wait_worker_iterations", 30
        )  # Number of iterations to wait before applying a redistribution plan
        self.enable_weight_nz_layout = additional_config.get(
            "enable_weight_nz_layout", False
        )  # Whether to convert quantized weights to NZ format to accelerate matrix multiplication
        self.enable_prefill_optimizations = additional_config.get(
            "enable_prefill_optimizations",
            False)  # Whether to enable DeepSeek models' prefill optimizations
        self.enable_cpu_binding = additional_config.get(  # Whether to enable the cpu binding
            "enable_cpu_binding", False)
        self.lmhead_tp_size = additional_config.get("lmhead_tp_size", -1)


class TorchairGraphConfig:
    """
    Configuration Object for torchair_graph_config from additional_config
    """

    def __init__(self, torchair_graph_config):
        self.enabled = torchair_graph_config.get(
            "enabled", False
        )  # Whether to enable torchair graph mode. Currently only DeepSeek series models and PanguProMoE are supported to use torchair graph mode
        self.use_cached_graph = torchair_graph_config.get(
            "use_cached_graph", False)  # Whether to use cached graph
        self.use_cached_kv_cache_bytes = torchair_graph_config.get(
            "use_cached_kv_cache_bytes", False
        )  # Whether to use cached kv_caches' memory, this option can only be enabled with use_cached_graph
        self.graph_batch_sizes = torchair_graph_config.get(
            "graph_batch_sizes", [])  # The batch size for torchair graph cache
        self.graph_batch_sizes_init = torchair_graph_config.get(
            "graph_batch_sizes_init", False
        )  # Init graph batch size dynamically if graph_batch_sizes is empty
        self.enable_multistream_mla = torchair_graph_config.get(
            "enable_multistream_mla", False
        )  # Whether to put vector ops of MLA to another stream. This option only takes effects on models using MLA(e.g., DeepSeek)
        self.enable_multistream_moe = torchair_graph_config.get(
            "enable_multistream_moe", False
        )  # Whether to enable multistream shared expert. This option only takes effects on DeepSeek moe models
        self.enable_view_optimize = torchair_graph_config.get(
            "enable_view_optimize",
            True)  # Whether to enable torchair view optimization
        self.enable_kv_nz = torchair_graph_config.get(
            "enable_kv_nz", False
        )  # Whether to enable kvcache NZ layout. This option only takes effects on models using MLA(e.g., DeepSeek)
        self.enable_super_kernel = torchair_graph_config.get(
            "enable_super_kernel", False)  # Whether to enable super kernel

        if not isinstance(self.graph_batch_sizes, list):
            raise TypeError("graph_batch_sizes must be list[int]")
        if self.graph_batch_sizes_init and len(self.graph_batch_sizes) > 0:
            raise ValueError(
                "graph_batch_sizes_init is only valid when graph_batch_sizes is empty"
            )
        if not self.enabled:
            if self.use_cached_graph:
                raise RuntimeError(
                    "use_cached_graph is valid only when Torchair graph mode is enabled"
                )
            if self.use_cached_kv_cache_bytes:
                raise RuntimeError(
                    "use_cached_kv_cache_bytes is valid only when Torchair graph mode is enabled"
                )
            if self.graph_batch_sizes:
                raise RuntimeError(
                    "graph_batch_sizes is valid only when Torchair graph mode is enabled"
                )
            if self.graph_batch_sizes_init:
                raise RuntimeError(
                    "graph_batch_sizes_init is valid only when Torchair graph mode is enabled"
                )
            if self.enable_multistream_mla:
                raise RuntimeError(
                    "enable_multistream_mla is valid only when Torchair graph mode is enabled"
                )
            if self.enable_multistream_moe:
                raise RuntimeError(
                    "enable_multistream_moe is valid only when Torchair graph mode is enabled"
                )
            if self.enable_kv_nz:
                raise RuntimeError(
                    "enable_kv_nz is valid only when Torchair graph mode is enabled"
                )
            if self.enable_super_kernel:
                raise RuntimeError(
                    "enable_super_kernel is valid only when Torchair graph mode and enable_multistream_moe is enabled"
                )
        if not self.enable_multistream_moe:
            if self.enable_super_kernel:
                raise RuntimeError(
                    "enable_super_kernel is valid only when Torchair graph mode and enable_multistream_moe are enabled"
                )
        if self.use_cached_kv_cache_bytes and not self.use_cached_graph:
            raise RuntimeError(
                "use_cached_kv_cache_bytes is valid only when Torchair graph mode and use_cached_graph are enabled"
            )


class AscendSchedulerConfig:
    """
    Configuration Object for ascend_scheduler_config from additional_config
    """

    def __init__(self, ascend_scheduler_config: dict):
        self.enabled = ascend_scheduler_config.get("enabled", False)
        # Ascend scheduler is based on vllm v0 scheduler, so we should support
        # all vllm v0 scheduler configs as well.
        for k, v in ascend_scheduler_config.items():
            if not hasattr(self, k):
                setattr(self, k, v)


_ASCEND_CONFIG: Optional[AscendConfig] = None


def init_ascend_config(vllm_config):
    additional_config = vllm_config.additional_config if vllm_config.additional_config is not None else {}
    refresh = additional_config.get("refresh",
                                    False) if additional_config else False
    global _ASCEND_CONFIG
    if _ASCEND_CONFIG is not None and not refresh:
        return _ASCEND_CONFIG
    _ASCEND_CONFIG = AscendConfig(vllm_config)
    return _ASCEND_CONFIG


def clear_ascend_config():
    global _ASCEND_CONFIG
    _ASCEND_CONFIG = None


def get_ascend_config():
    global _ASCEND_CONFIG
    if _ASCEND_CONFIG is None:
        raise RuntimeError(
            "Ascend config is not initialized. Please call init_ascend_config first."
        )
    return _ASCEND_CONFIG


def check_ascend_config(vllm_config, enforce_eager):
    ascend_config = get_ascend_config()

    # for v0 engine
    if not envs.VLLM_USE_V1:
        if ascend_config.torchair_graph_config.enabled:
            raise NotImplementedError(
                "Torchair graph mode is only supported for V1 Engine.")
        if ascend_config.ascend_scheduler_config.enabled:
            raise NotImplementedError(
                "Ascend scheduler is only supported for V1 Engine.")
    # for v1 engine
    else:
        # for eager mode
        if enforce_eager:
            # torchair_graph cannot be enabled with eager mode.
            if ascend_config.torchair_graph_config.enabled:
                raise RuntimeError(
                    "Can't enable graph mode and eager mode at the same time. Please set `enforce_eager=False` if you attempt to enable NPU graph mode."
                )
        # for graph mode
        else:
            # torchair_graph case
            if ascend_config.torchair_graph_config.enabled:
                # torchair_graph is not supported for V1 without mla currently.
                if envs.VLLM_MLA_DISABLE:
                    logger.warning(
                        "Torchair graph mode is still experimental and not supported for V1 without mla currently, "
                        "it has been disabled automatically.")
                    ascend_config.torchair_graph_config.enabled = False
                # torchair_graph is supported for deepseek model only currently.
                if vllm_config.model_config:
                    model_type = vllm_config.model_config.hf_config.model_type
                    if "deepseek" not in model_type:
                        raise NotImplementedError(
                            "Torchair graph mode only works with deepseek model."
                        )
            # aclgraph case
            else:
                # aclgraph doesn't work with deepseek model and only qwen model is well tested.
                if vllm_config.model_config:
                    model_type = vllm_config.model_config.hf_config.model_type
                    if "deepseek" in model_type:
                        raise NotImplementedError(
                            "ACL Graph does not support deepseek. Please "
                            "try torchair graph mode to serve deepseek models on vllm-ascend."
                            " Or set `enforce_eager=True` to use eager mode.")
                    if "qwen" not in model_type:
                        logger.warning(
                            "ACL Graph is currently experimental. Please "
                            "raise an issue on https://github.com/vllm-project/vllm-ascend/issues"
                            " if you encourage any Error")
