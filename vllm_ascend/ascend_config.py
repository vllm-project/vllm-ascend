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

from vllm.logger import logger

TORCHAIR_MODEL_LIST = ["deepseek", "pangu", "kimi_k2", "qwen"]


def _check_torchair_supported(model_type: str):
    for supported_model in TORCHAIR_MODEL_LIST:
        if supported_model in model_type.lower():
            return True
    return False


class AscendConfig:
    """
    Configuration Object for additional_config from vllm.configs.
    """

    def __init__(self, vllm_config):
        additional_config = vllm_config.additional_config if vllm_config.additional_config is not None else {}

        torchair_graph_config = additional_config.get("torchair_graph_config",
                                                      {})
        self.torchair_graph_config = TorchairGraphConfig(torchair_graph_config)

        ascend_scheduler_config = additional_config.get(
            "ascend_scheduler_config", {})
        self.ascend_scheduler_config = AscendSchedulerConfig(
            ascend_scheduler_config)

        self.expert_map_path = additional_config.get("expert_map_path", None)
        self.chunked_prefill_for_mla = additional_config.get(
            "chunked_prefill_for_mla", False)
        self.enable_shared_expert_dp = additional_config.get(
            "enable_shared_expert_dp", False
        ) and not self.torchair_graph_config.enabled and vllm_config.parallel_config.enable_expert_parallel \
          and not vllm_config.parallel_config.enable_sequence_parallel
        self.enable_prefetch = additional_config.get("enable_prefetch", False)
        self.lmhead_tensor_parallel_size = additional_config.get(
            "lmhead_tensor_parallel_size", None)
        if self.lmhead_tensor_parallel_size is not None:
            logger.info(
                f"Enable lmhead_tensor_parallel_size={self.lmhead_tensor_parallel_size} in pure DP scenario"
            )
            if vllm_config.parallel_config.tensor_parallel_size != 1:
                raise AssertionError(
                    "lmhead_tensor_parallel_size is only supported in the pure DP scenario"
                )
        self.oproj_tensor_parallel_size = additional_config.get(
            "oproj_tensor_parallel_size", None)
        if self.oproj_tensor_parallel_size is not None:
            logger.info(
                f"Enable oproj_tensor_parallel_size={self.oproj_tensor_parallel_size} in pure DP scenario"
            )
            if vllm_config.parallel_config.tensor_parallel_size != 1:
                raise AssertionError(
                    "oproj_tensor_parallel_size is only supported in the pure DP scenario"
                )
            if not self.torchair_graph_config.enabled:
                raise AssertionError(
                    "oproj_tensor_parallel_size is only supported in graph mode"
                )
            if vllm_config.kv_transfer_config is None or not vllm_config.kv_transfer_config.is_kv_consumer:
                raise AssertionError(
                    "oproj_tensor_parallel_size is only supported in pd scenario and can only be used in D node."
                )


class TorchairGraphConfig:
    """
    Configuration Object for torchair_graph_config from additional_config
    """

    def __init__(self, torchair_graph_config):
        self.enabled = torchair_graph_config.get("enabled", False)
        self.mode = torchair_graph_config.get("mode", '')
        self.use_cached_graph = torchair_graph_config.get(
            "use_cached_graph", False)
        self.use_cached_kv_cache_bytes = torchair_graph_config.get(
            "use_cached_kv_cache_bytes", False)
        self.graph_batch_sizes = torchair_graph_config.get(
            "graph_batch_sizes", [])
        self.graph_batch_sizes_init = torchair_graph_config.get(
            "graph_batch_sizes_init", False)
        self.enable_multistream_mla = torchair_graph_config.get(
            "enable_multistream_mla", False)
        self.enable_multistream_moe = torchair_graph_config.get(
            "enable_multistream_moe", False)
        self.enable_view_optimize = torchair_graph_config.get(
            "enable_view_optimize", True)
        self.enable_kv_nz = torchair_graph_config.get("enable_kv_nz", False)

        if not isinstance(self.graph_batch_sizes, list):
            raise TypeError("graph_batch_sizes must be list[int]")
        if self.graph_batch_sizes_init and len(self.graph_batch_sizes) > 0:
            raise ValueError(
                "graph_batch_sizes_init is only valid when graph_batch_sizes is empty"
            )
        if not self.enabled:
            if self.mode:
                raise RuntimeError(
                    "mode is valid only when Torchair graph mode is enabled")
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
            # torchair_graph is supported for deepseek/pangu/qwen model only.
            if vllm_config.model_config:
                model_type = vllm_config.model_config.hf_config.model_type
                if not _check_torchair_supported(model_type):
                    raise NotImplementedError(
                        "Torchair graph mode only works with following model types:"
                        f"{TORCHAIR_MODEL_LIST}.")
            if ascend_config.enable_shared_expert_dp:
                logger.warning(
                    "enable_shared_expert_dp is not supported for torchair graph mode currently, "
                    "it has been disabled automatically.")
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
