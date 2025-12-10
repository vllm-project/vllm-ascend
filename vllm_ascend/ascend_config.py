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
from uuid import uuid4

from vllm.logger import logger


def check_kv_extra_config(vllm_config):

    def _check(name: str, config: dict):
        tp_key = "tp_size"
        dp_key = "dp_size"
        if tp_key in config:
            config_tp = config[tp_key]
            vllm_tp = vllm_config.parallel_config.tensor_parallel_size
            if config_tp != vllm_tp:
                raise ValueError(
                    f"KV transfer '{name}' config has a conflicting tensor parallel size. "
                    f"Expected {vllm_tp}, but got {config_tp}.")
        if dp_key in config:
            config_dp = config[dp_key]
            vllm_dp = vllm_config.parallel_config.data_parallel_size
            if config_dp != vllm_dp:
                raise ValueError(
                    f"KV transfer '{name}' config has a conflicting data parallel size. "
                    f"Expected {vllm_dp}, but got {config_dp}.")

    if vllm_config.kv_transfer_config.is_kv_producer:
        _check(
            "prefill",
            vllm_config.kv_transfer_config.get_from_extra_config(
                "prefill", {}))
    if vllm_config.kv_transfer_config.is_kv_consumer:
        _check(
            "decode",
            vllm_config.kv_transfer_config.get_from_extra_config("decode", {}))


class AscendConfig:
    """
    Configuration Object for additional_config from vllm.configs.
    """

    def __init__(self, vllm_config):
        additional_config = vllm_config.additional_config if vllm_config.additional_config is not None else {}

        xlite_graph_config = additional_config.get("xlite_graph_config", {})
        self.xlite_graph_config = XliteGraphConfig(xlite_graph_config,
                                                   vllm_config)

        ascend_compilation_config = additional_config.get(
            "ascend_compilation_config", {})
        self.ascend_compilation_config = AscendCompilationConfig(
            **ascend_compilation_config)

        # Dump / PrecisionDebugger configuration
        dump_config_path = additional_config.get("dump_config", None)
        self.dump_config = DumpConfig(dump_config_path)

        weight_prefetch_config = additional_config.get(
            "weight_prefetch_config", {})
        self.weight_prefetch_config = WeightPrefetchConfig(
            weight_prefetch_config)

        # Todo: Once https://github.com/vllm-project/vllm/issues/22246 is merged in vllm. Remove this config
        self.expert_map_path = additional_config.get("expert_map_path", None)
        self.eplb_policy_type = additional_config.get("eplb_policy_type", 1)
        self.expert_map_record_path = additional_config.get(
            "expert_map_record_path",
            None)  # Provide path to export expert map
        self.init_redundancy_expert = additional_config.get(
            "init_redundancy_expert", 0)
        self.dynamic_eplb = additional_config.get("dynamic_eplb", False)
        self.num_iterations_eplb_update = additional_config.get(
            "num_iterations_eplb_update", 400)
        self.gate_eplb = additional_config.get("gate_eplb", False)
        self.num_wait_worker_iterations = additional_config.get(
            "num_wait_worker_iterations", 30)
        self.chunked_prefill_for_mla = additional_config.get(
            "chunked_prefill_for_mla", False)
        self.enable_shared_expert_dp = additional_config.get(
            "enable_shared_expert_dp",
            False) and vllm_config.parallel_config.enable_expert_parallel
        if self.enable_shared_expert_dp:
            from vllm_ascend.utils import enable_sp
            assert enable_sp(vllm_config=vllm_config,
                             enable_shared_expert_dp=True)
        self.multistream_overlap_shared_expert = additional_config.get(
            "multistream_overlap_shared_expert", False)
        self.recompute_scheduler_enable = additional_config.get(
            "recompute_scheduler_enable", False)
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
            if vllm_config.model_config.enforce_eager is True:
                raise AssertionError(
                    "oproj_tensor_parallel_size is only supported in graph mode"
                )
            if vllm_config.kv_transfer_config is None or not vllm_config.kv_transfer_config.is_kv_consumer:
                raise AssertionError(
                    "oproj_tensor_parallel_size is only supported in pd scenario and can only be used in D node."
                )
        self.enable_cpu_binding = additional_config.get(
            "enable_cpu_binding", False)

        if vllm_config.kv_transfer_config is not None:
            check_kv_extra_config(vllm_config)

        self.pd_tp_ratio = 1
        self.pd_head_ratio = 1
        self.num_head_replica = 1
        if vllm_config.kv_transfer_config is not None and not vllm_config.model_config.is_deepseek_mla:
            prefill_tp_size = vllm_config.kv_transfer_config.get_from_extra_config(
                "prefill", {"tp_size": 1})["tp_size"]
            decode_tp_size = vllm_config.kv_transfer_config.get_from_extra_config(
                "decode", {"tp_size": 1})["tp_size"]
            assert prefill_tp_size % decode_tp_size == 0, "Prefill TP size must be divisible by Decode TP size."
            self.pd_tp_ratio = prefill_tp_size // decode_tp_size
            if self.pd_tp_ratio > 1:
                try:
                    # only support Qwen model now
                    # TODO: use a more robust method to get kv_head_num
                    num_kv_head = vllm_config.model_config.hf_config.num_key_value_heads
                    self.num_head_replica = prefill_tp_size // num_kv_head if prefill_tp_size >= num_kv_head else 1
                    prefill_tp_size = min(prefill_tp_size, num_kv_head)
                    decode_tp_size = min(decode_tp_size, num_kv_head)
                    self.pd_head_ratio = prefill_tp_size // decode_tp_size
                except Exception:
                    raise AssertionError(
                        "Can not get num_key_value_heads from model_config")

            if self.pd_tp_ratio == 0:
                raise AssertionError(
                    "Only support P node tp size lagger then D node tp size")
        self.SLO_limits_for_dynamic_batch = additional_config.get(
            "SLO_limits_for_dynamic_batch", -1)
        from vllm_ascend.utils import \
            get_flashcomm2_oproj_tp_size_and_validate_config
        self.flashcomm2_oproj_tensor_parallel_size = get_flashcomm2_oproj_tp_size_and_validate_config(
            self, vllm_config)
        kv_cfg = vllm_config.kv_transfer_config
        if kv_cfg is not None and not getattr(kv_cfg, "_engine_id_patched",
                                              False):
            kv_cfg.engine_id = f"{kv_cfg.engine_id}-{uuid4().hex}"
            kv_cfg._engine_id_patched = True


class AscendCompilationConfig:
    """
    Configuration for controlling the behavior of Ascend graph optimization.

    This class provides a way to configure graph fusion optimizations.
    These configurations directly impact the performance and behavior of models
    deployed on Ascend platforms.
    """

    def __init__(self, enable_quantization_fusion: bool = True, **kwargs):
        """
        Initialize the configuration.
        
        Args:
            enable_quantization_fusion (bool): Whether to enable quantization fusion optimization.
                When set to True, the system will optimize quantization-related operations,
                reducing the number of quantization/dequantization nodes.
                Default: True
                
            **kwargs: Additional optional parameters for forward compatibility and configuration extension.
        """
        self.enable_quantization_fusion = enable_quantization_fusion
        # Add more compilation related configs here as needed


class XliteGraphConfig:
    """
    Configuration Object for xlite_graph_config from additional_config
    """

    def __init__(self, xlite_graph_config, vllm_config):
        self.enabled = xlite_graph_config.get("enabled", False)
        self.full_mode = xlite_graph_config.get("full_mode", False)
        if self.enabled:
            if bool(vllm_config.speculative_config):
                raise RuntimeError(
                    "Xlite graph mode is not compatible with speculative decoding. Please disable speculative decoding."
                )
            if vllm_config.parallel_config.pipeline_parallel_size > 1:
                raise RuntimeError(
                    "Xlite graph mode is not compatible with pipeline parallelism. Please set pipeline_parallel_size to 1."
                )
            if vllm_config.cache_config.block_size != 128:
                raise RuntimeError(
                    "Xlite graph mode is only compatible with block_size of 128. Please set block_size to 128."
                )


class DumpConfig:
    """
    Configuration object for dump/PrecisionDebugger settings.
    """

    def __init__(self, dump_config_path: Optional[str] = None):
        # enable_dump is True when dump_cfg exists and config_path is not empty
        self.enable_dump: bool = bool(dump_config_path)
        # Path to msprobe config json; may be None.
        self.config_path: Optional[str] = dump_config_path


class WeightPrefetchConfig:
    """
    Configuration Object for weight_prefetch_config from additional_config
    """

    prefetch_ratio: dict = {
        "attn": {
            "qkv": 1.0,
            "o": 1.0,
        },
        "moe": {
            "gate_up": 0.8
        }
    }

    def __init__(self, weight_prefetch_config: dict):
        self.enabled = weight_prefetch_config.get("enabled", False)
        self.prefetch_ratio = weight_prefetch_config.get(
            "prefetch_ratio", self.prefetch_ratio)


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

    if ascend_config.ascend_compilation_config.enable_quantization_fusion:
        logger.info(
            "Quantization fusion enabled! op fusion on quantization are expected. "
        )
