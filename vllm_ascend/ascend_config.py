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
from typing import TYPE_CHECKING, Optional

from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class AscendConfig:
    """
    Configuration Object for additional_config from vllm.configs.
    """

    def __init__(self, vllm_config: "VllmConfig"):
        additional_config = vllm_config.additional_config if vllm_config.additional_config is not None else {}

        xlite_graph_config = additional_config.get("xlite_graph_config", {})
        self.xlite_graph_config = XliteGraphConfig(xlite_graph_config,
                                                   vllm_config)

        ascend_compilation_config = additional_config.get(
            "ascend_compilation_config", {})
        self.ascend_compilation_config = AscendCompilationConfig(
            **ascend_compilation_config)

        finegrained_tp_config = additional_config.get("finegrained_tp_config",
                                                      {})
        self.finegrained_tp_config = FinegrainedTPConfig(
            finegrained_tp_config, vllm_config)

        # Dump / PrecisionDebugger configuration
        self.dump_config_path = additional_config.get("dump_config_path", None)

        weight_prefetch_config = additional_config.get(
            "weight_prefetch_config", {})
        self.weight_prefetch_config = WeightPrefetchConfig(
            weight_prefetch_config)
        self.layer_sharding = additional_config.get("layer_sharding", None)
        logger.info_once(
            f"Linear layer sharding enabled with config: {self.layer_sharding}. "
            "Note: This feature works optimally with FLASHCOMM2 and DSA-CP enabled; "
            "using it without these features may result in significant performance degradation."
        )

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
        self.enable_shared_expert_dp = additional_config.get(
            "enable_shared_expert_dp",
            False) and vllm_config.parallel_config.enable_expert_parallel
        if self.enable_shared_expert_dp:
            from vllm_ascend.utils import enable_sp
            assert enable_sp(vllm_config=vllm_config,
                             enable_shared_expert_dp=True)
        self.multistream_overlap_shared_expert = additional_config.get(
            "multistream_overlap_shared_expert", False)
        self.multistream_overlap_gate = additional_config.get(
            "multistream_overlap_gate", False)
        self.recompute_scheduler_enable = additional_config.get(
            "recompute_scheduler_enable", False)
        self.enable_cpu_binding = additional_config.get(
            "enable_cpu_binding", False)

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
                    num_kv_head = vllm_config.model_config.hf_text_config.num_key_value_heads
                    self.num_head_replica = prefill_tp_size // num_kv_head if prefill_tp_size >= num_kv_head else 1
                    prefill_tp_size = min(prefill_tp_size, num_kv_head)
                    decode_tp_size = min(decode_tp_size, num_kv_head)
                    self.pd_head_ratio = prefill_tp_size // decode_tp_size
                except Exception:
                    raise ValueError(
                        "The text_config extracted from the model config does not have "
                        "`num_key_value_heads` attribute. This indicates a mismatch "
                        "between the model config and vLLM's expectations. Please "
                        "ensure that the model config is compatible with vLLM."
                    )

            if self.pd_tp_ratio == 0:
                raise AssertionError(
                    "Only support P node tp size lagger then D node tp size")
        self.SLO_limits_for_dynamic_batch = additional_config.get(
            "SLO_limits_for_dynamic_batch", -1)
        from vllm_ascend.utils import get_flashcomm2_config_and_validate
        self.flashcomm2_oproj_tensor_parallel_size = get_flashcomm2_config_and_validate(
            self, vllm_config)
        self.enable_npugraph_ex = additional_config.get(
            "enable_npugraph_ex", False)
        # We find that _npu_paged_attention still performs better than
        # npu_fused_infer_attention_score in some cases. We allow to execute
        # _npu_paged_attention in this cases. This should be removed once
        # npu_fused_infer_attention_score performs better on all scenarios.
        self.pa_shape_list = additional_config.get("pa_shape_list", [])

        self.enable_async_exponential = bool(
            additional_config.get("enable_async_exponential", False))

        self.enable_kv_nz = additional_config.get("enable_kv_nz", False)
        if self.enable_kv_nz:
            use_sparse = hasattr(vllm_config.model_config.hf_text_config,
                                 "index_topk")
            if not vllm_config.model_config.is_deepseek_mla or use_sparse:
                raise RuntimeError(
                    "enable_kv_nz is only supported for mla currently.")
            if vllm_config.kv_transfer_config is None \
                or not vllm_config.kv_transfer_config.is_kv_consumer:
                raise NotImplementedError(
                    "enable_kv_nz is only supported in pd scenario and can "
                    "only be used in D node.")


class FinegrainedTPConfig:
    """
    Configuration Object for finegrained_tp_config from additional_config
    """

    def __init__(self, finegrained_tp_config: dict, vllm_config):
        self.oproj_tensor_parallel_size = finegrained_tp_config.get(
            "oproj_tensor_parallel_size", 0)
        self.lmhead_tensor_parallel_size = finegrained_tp_config.get(
            "lmhead_tensor_parallel_size", 0)
        self.embedding_tensor_parallel_size = finegrained_tp_config.get(
            "embedding_tensor_parallel_size", 0)
        self.mlp_tensor_parallel_size = finegrained_tp_config.get(
            "mlp_tensor_parallel_size", 0)

        enabled_configs = []
        if self.oproj_tensor_parallel_size > 0:
            enabled_configs.append(
                f"oproj_tensor_parallel_size={self.oproj_tensor_parallel_size}"
            )
            # dummy_run does not run the entire attention module in eager mode,, so the o_proj tp split can only be used in graph mode.
            if vllm_config.model_config.enforce_eager is True:
                raise AssertionError(
                    "oproj_tensor_parallel_size is only supported in graph mode"
                )
            if vllm_config.kv_transfer_config is None or not vllm_config.kv_transfer_config.is_kv_consumer:
                raise AssertionError(
                    "oproj_tensor_parallel_size is only supported in pd scenario and can only be used in D node."
                )
        if self.lmhead_tensor_parallel_size > 0:
            enabled_configs.append(
                f"lmhead_tensor_parallel_size={self.lmhead_tensor_parallel_size}"
            )
        if self.embedding_tensor_parallel_size > 0:
            enabled_configs.append(
                f"embedding_tensor_parallel_size={self.embedding_tensor_parallel_size}"
            )
        if self.mlp_tensor_parallel_size > 0:
            enabled_configs.append(
                f"mlp_tensor_parallel_size={self.mlp_tensor_parallel_size}")
        module_tp_sizes = [
            self.oproj_tensor_parallel_size,
            self.lmhead_tensor_parallel_size,
            self.embedding_tensor_parallel_size,
            self.mlp_tensor_parallel_size,
        ]
        for module_tp_size in module_tp_sizes:
            if module_tp_size > 0 and vllm_config.parallel_config.data_parallel_size % module_tp_size != 0:
                raise AssertionError(
                    "module tp sizes must divide data_parallel_size")
        if any(size > 0 for size in module_tp_sizes) and enabled_configs:
            logger.info(
                f"finegrained_tp_config enabled: {', '.join(enabled_configs)}")


class AscendCompilationConfig:
    """
    Configuration for controlling the behavior of Ascend graph optimization.

    This class provides a way to configure graph fusion optimizations.
    These configurations directly impact the performance and behavior of models
    deployed on Ascend platforms.
    """

    def __init__(self,
                 fuse_norm_quant: bool = True,
                 fuse_qknorm_rope: bool = False,
                 **kwargs):
        """
        Initialize the configuration.
        
        Args:
            fuse_norm_quant (bool): Whether to enable norm and quant fusion optimization.
                When set to True, the system will optimize norm and quant operations.
                Default: True
            fuse_qknorm_rope (bool): Whether to enable qknorm and rope fusion optimization.
                Default: False
            **kwargs: Additional optional parameters for forward compatibility and configuration extension.
        """
        self.fuse_norm_quant = fuse_norm_quant
        self.fuse_qknorm_rope = HAS_TRITON or fuse_qknorm_rope


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
