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
import os
import warnings
from typing import TYPE_CHECKING

from vllm.logger import logger
from vllm.utils.math_utils import cdiv

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class AscendConfig:
    """
    Configuration Object for additional_config from vllm.configs.
    """

    def __init__(self, vllm_config: "VllmConfig"):
        self.vllm_config = vllm_config
        additional_config = vllm_config.additional_config if vllm_config.additional_config is not None else {}

        xlite_graph_config = additional_config.get("xlite_graph_config", {})
        self.xlite_graph_config = XliteGraphConfig(xlite_graph_config, vllm_config)

        ascend_compilation_config = additional_config.get("ascend_compilation_config", {})
        self.ascend_compilation_config = AscendCompilationConfig(**ascend_compilation_config)

        ascend_fusion_config = additional_config.get("ascend_fusion_config", {})
        self.ascend_fusion_config = AscendFusionConfig(**ascend_fusion_config)

        finegrained_tp_config = additional_config.get("finegrained_tp_config", {})
        self.finegrained_tp_config = FinegrainedTPConfig(finegrained_tp_config, vllm_config)

        eplb_config = additional_config.get("eplb_config", {})
        self.eplb_config = EplbConfig(eplb_config)

        # Dump / PrecisionDebugger configuration
        self.dump_config_path = additional_config.get("dump_config_path", None)
        self._construct_weight_prefetch_config(additional_config)
        self.layer_sharding = additional_config.get("layer_sharding", None)
        logger.info_once(
            f"Linear layer sharding enabled with config: {self.layer_sharding}. "
            "Note: This feature works optimally with FLASHCOMM2 and DSA-CP enabled; "
            "using it without these features may result in significant performance degradation."
        )

        self.enable_shared_expert_dp = (
            additional_config.get("enable_shared_expert_dp", False)
            and vllm_config.parallel_config.enable_expert_parallel
            and vllm_config.parallel_config.tensor_parallel_size > 1
        )
        from vllm_ascend.utils import enable_sp

        if self.enable_shared_expert_dp:
            assert enable_sp(vllm_config=vllm_config, enable_shared_expert_dp=True)

        if vllm_config.parallel_config.prefill_context_parallel_size > 1 and enable_sp(vllm_config=vllm_config):
            tp_pcp_size = (
                vllm_config.parallel_config.tensor_parallel_size
                * vllm_config.parallel_config.prefill_context_parallel_size
            )
            if vllm_config.scheduler_config.max_num_batched_tokens % tp_pcp_size != 0:
                vllm_config.scheduler_config.max_num_batched_tokens = (
                    cdiv(vllm_config.scheduler_config.max_num_batched_tokens, tp_pcp_size) * tp_pcp_size
                )
                logger.warning_once(
                    f"When using FLASHCOMM1, the max_num_batched_tokens should be divisible"
                    f"by tp_size * pcp_size ({tp_pcp_size}). It has been adjusted to"
                    f"{vllm_config.scheduler_config.max_num_batched_tokens}."
                )
        self.multistream_overlap_shared_expert = additional_config.get("multistream_overlap_shared_expert", False)
        self.multistream_overlap_gate = additional_config.get("multistream_overlap_gate", False)
        self.recompute_scheduler_enable = additional_config.get("recompute_scheduler_enable", False)
        self.enable_cpu_binding = additional_config.get("enable_cpu_binding", True)

        self.pd_tp_ratio = 1
        self.pd_head_ratio = 1
        self.num_head_replica = 1
        if vllm_config.kv_transfer_config is not None and not vllm_config.model_config.is_deepseek_mla:
            prefill_tp_size = vllm_config.kv_transfer_config.get_from_extra_config("prefill", {"tp_size": 1})["tp_size"]
            decode_tp_size = vllm_config.kv_transfer_config.get_from_extra_config("decode", {"tp_size": 1})["tp_size"]
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
                raise AssertionError("Only support P node tp size lagger then D node tp size")
        self.SLO_limits_for_dynamic_batch = additional_config.get("SLO_limits_for_dynamic_batch", -1)
        from vllm_ascend.utils import get_flashcomm2_config_and_validate

        self.flashcomm2_oproj_tensor_parallel_size = get_flashcomm2_config_and_validate(self, vllm_config)
        npugraph_ex_config = additional_config.get("npugraph_ex_config", {})
        self.npugraph_ex_config = NpugraphExConfig(**npugraph_ex_config)
        # We find that _npu_paged_attention still performs better than
        # npu_fused_infer_attention_score in some cases. We allow to execute
        # _npu_paged_attention in this cases. This should be removed once
        # npu_fused_infer_attention_score performs better on all scenarios.
        self.pa_shape_list = additional_config.get("pa_shape_list", [])

        self.enable_async_exponential = bool(additional_config.get("enable_async_exponential", False))

        self.enable_kv_nz = additional_config.get("enable_kv_nz", False)
        if self.enable_kv_nz:
            use_sparse = hasattr(vllm_config.model_config.hf_text_config, "index_topk")
            if not vllm_config.model_config.is_deepseek_mla or use_sparse:
                raise RuntimeError("enable_kv_nz is only supported for mla currently.")
            if vllm_config.kv_transfer_config is None or not vllm_config.kv_transfer_config.is_kv_consumer:
                raise NotImplementedError(
                    "enable_kv_nz is only supported in pd scenario and can only be used in D node."
                )

    def _construct_weight_prefetch_config(self, additional_config):
        weight_prefetch_config = additional_config.get("weight_prefetch_config", {})
        self.weight_prefetch_config = WeightPrefetchConfig(weight_prefetch_config)
        # Deprecated env var handling for backward compatibility
        if os.getenv("VLLM_ASCEND_ENABLE_PREFETCH_MLP", "0") == "1":
            MAX_PREFETCH_WEIGHT_SIZE: int = 18 * 1024 * 1024
            gate_up_prefetch_size = int(os.getenv("VLLM_ASCEND_MLP_GATE_UP_PREFETCH_SIZE", MAX_PREFETCH_WEIGHT_SIZE))
            down_prefetch_size = int(os.getenv("VLLM_ASCEND_MLP_DOWN_PREFETCH_SIZE", MAX_PREFETCH_WEIGHT_SIZE))
            self.weight_prefetch_config.set_mlp_pre_version_compatibale_config(
                gate_up_prefetch_size, down_prefetch_size
            )
            logger.info_once(
                f"MLP weight prefetch enabled from env variable VLLM_ASCEND_ENABLE_PREFETCH_MLP."
                f"gate_up_prefetch_size={gate_up_prefetch_size}, "
                f"down_prefetch_size={down_prefetch_size}."
            )
            warnings.warn(
                "VLLM_ASCEND_ENABLE_PREFETCH_MLP is deprecated and will be removed in a v0.16.0 version. "
                "Please use weight_prefetch_config in additional-config for now instead.",
                DeprecationWarning,
                stacklevel=2,
            )

    def update_compile_ranges_split_points(self):
        vllm_config = self.vllm_config
        if self.npugraph_ex_config.enable:
            if self.npugraph_ex_config.fuse_allreduce_rms:
                from vllm_ascend.compilation.passes.allreduce_rmsnorm_fusion_pass import ALLREDUCE_NORM_FUSE_THRESHOLD

                new_compile_ranges_split_points = vllm_config.compilation_config.compile_ranges_split_points
                new_compile_ranges_split_points.append(ALLREDUCE_NORM_FUSE_THRESHOLD)
                new_compile_ranges_split_points = sorted(new_compile_ranges_split_points)
                vllm_config.compilation_config.compile_ranges_split_points = new_compile_ranges_split_points
                logger.debug(
                    "set compile_ranges_split_points to "
                    "{new_compile_ranges_split_points} for matmul and allreduce fusion"
                )

        else:
            new_compile_ranges_split_points = vllm_config.compilation_config.compile_ranges_split_points
            if vllm_config.additional_config.get("ascend_compilation_config", {}).get("fuse_allreduce_rms", True):
                from vllm_ascend.compilation.passes.allreduce_rmsnorm_fusion_pass import ALLREDUCE_NORM_FUSE_THRESHOLD

                new_compile_ranges_split_points = vllm_config.compilation_config.compile_ranges_split_points
                new_compile_ranges_split_points.append(ALLREDUCE_NORM_FUSE_THRESHOLD)
                new_compile_ranges_split_points = sorted(new_compile_ranges_split_points)
                vllm_config.compilation_config.compile_ranges_split_points = new_compile_ranges_split_points
                logger.debug(
                    "set compile_ranges_split_points to "
                    "{new_compile_ranges_split_points} for matmul and allreduce fusion"
                )

            from vllm_ascend.utils import is_moe_model

            if vllm_config.compilation_config.pass_config.enable_sp and not is_moe_model(vllm_config):
                from vllm_ascend.compilation.passes.sequence_parallelism import get_sp_threshold

                sp_threshold = get_sp_threshold(vllm_config)
                new_compile_ranges_split_points.append(sp_threshold)
                logger.debug(f"add {sp_threshold} to compile_ranges_split_points for sequence parallelism")
            if len(new_compile_ranges_split_points) > len(vllm_config.compilation_config.compile_ranges_split_points):
                new_compile_ranges_split_points = sorted(new_compile_ranges_split_points)
                vllm_config.compilation_config.compile_ranges_split_points = new_compile_ranges_split_points


class FinegrainedTPConfig:
    """
    Configuration Object for finegrained_tp_config from additional_config
    """

    def __init__(self, finegrained_tp_config: dict, vllm_config):
        self.oproj_tensor_parallel_size = finegrained_tp_config.get("oproj_tensor_parallel_size", 0)
        self.lmhead_tensor_parallel_size = finegrained_tp_config.get("lmhead_tensor_parallel_size", 0)
        self.embedding_tensor_parallel_size = finegrained_tp_config.get("embedding_tensor_parallel_size", 0)
        self.mlp_tensor_parallel_size = finegrained_tp_config.get("mlp_tensor_parallel_size", 0)

        enabled_configs = []
        if self.oproj_tensor_parallel_size > 0:
            enabled_configs.append(f"oproj_tensor_parallel_size={self.oproj_tensor_parallel_size}")
            # dummy_run does not run the entire attention module in eager mode,
            # so the o_proj tp split can only be used in graph mode.
            if vllm_config.model_config.enforce_eager is True:
                raise AssertionError("oproj_tensor_parallel_size is only supported in graph mode")
            if vllm_config.kv_transfer_config is None or not vllm_config.kv_transfer_config.is_kv_consumer:
                raise AssertionError(
                    "oproj_tensor_parallel_size is only supported in pd scenario and can only be used in D node."
                )
        if self.lmhead_tensor_parallel_size > 0:
            enabled_configs.append(f"lmhead_tensor_parallel_size={self.lmhead_tensor_parallel_size}")
        if self.embedding_tensor_parallel_size > 0:
            enabled_configs.append(f"embedding_tensor_parallel_size={self.embedding_tensor_parallel_size}")
        if self.mlp_tensor_parallel_size > 0:
            enabled_configs.append(f"mlp_tensor_parallel_size={self.mlp_tensor_parallel_size}")
        module_tp_sizes = [
            self.oproj_tensor_parallel_size,
            self.lmhead_tensor_parallel_size,
            self.embedding_tensor_parallel_size,
            self.mlp_tensor_parallel_size,
        ]
        for module_tp_size in module_tp_sizes:
            if module_tp_size > 0 and vllm_config.parallel_config.data_parallel_size % module_tp_size != 0:
                raise AssertionError("module tp sizes must divide data_parallel_size")
        if any(size > 0 for size in module_tp_sizes) and enabled_configs:
            logger.info(f"finegrained_tp_config enabled: {', '.join(enabled_configs)}")


class AscendCompilationConfig:
    """
    Configuration for controlling the behavior of Ascend graph optimization.

    This class provides a way to configure graph fusion optimizations.
    These configurations directly impact the performance and behavior of models
    deployed on Ascend platforms.
    """

    def __init__(
        self, fuse_norm_quant: bool = True, fuse_qknorm_rope: bool = True, fuse_allreduce_rms: bool = False, **kwargs
    ):
        """
        Initialize the configuration.

        Args:
            fuse_norm_quant (bool): Whether to enable norm and quant fusion optimization.
                When set to True, the system will optimize norm and quant operations.
                Default: True
            fuse_qknorm_rope (bool): Whether to enable qknorm and rope fusion optimization.
                Default: True
            fuse_allreduce_rms (bool): Whether to enable allreduce and addrmsnorm fusion optimization.
                Default: False
            **kwargs: Additional optional parameters for forward compatibility and configuration extension.
        """
        self.fuse_norm_quant = fuse_norm_quant
        self.fuse_qknorm_rope = fuse_qknorm_rope
        self.fuse_allreduce_rms = fuse_allreduce_rms


class AscendFusionConfig:
    """
    Configuration for controlling whether to use a fused operator gmmswigluquant.
    """

    def __init__(self, fusion_ops_gmmswigluquant: bool = True, **kwargs):
        """
        Initialize the configuration.

        Args:
            fusion_ops_gmmswigluquant (bool): Whether to use a fused operator gmmswigluquant.
                When set to True, the system will use a fused operator gmmswigluquant.
                Default: True
            **kwargs: Additional optional parameters for forward compatibility and configuration extension.
        """
        self.fusion_ops_gmmswigluquant = fusion_ops_gmmswigluquant


class NpugraphExConfig:
    """
    Configuration for controlling the behavior of npugraph_ex backend.

    This class provides a way to configure whether to use the npugraph_ex backend and static kernel.
    These configurations can directly impact the performance and behavior of models deployed on Ascend platforms.
    """

    def __init__(
        self,
        enable: bool = True,
        enable_static_kernel: bool = False,
        fuse_norm_quant: bool = True,
        fuse_qknorm_rope: bool = True,
        fuse_allreduce_rms: bool = False,
        **kwargs,
    ):
        """
        Initialize the configuration.

        Args:
            enable (bool): Whether to enable npugraph_ex backend.
                When set to True, the Fx graph generated by Dymano will be
                optimized and compiled by the npugraph_ex backend.
                Default: True
            enable_static_kernel (bool): Whether to enable static kernel.
                Static kernel is suitable for scenarios with purely static shapes
                or minimal shape changes, and can improve network performance.
                When set to True, when during graph capture, it will compile operator
                binary files with the corresponding shapes based on the current batch_size,
                which usually takes some time.
                Default: False
            fuse_norm_quant (bool): Whether to enable norm and quant fusion optimization.
                When set to True, the system will optimize norm and quant operations.
                Default: True
            fuse_qknorm_rope (bool): Whether to enable qknorm and rope fusion optimization.
                Default: True
            fuse_allreduce_rms (bool): Whether to enable allreduce and addrmsnorm fusion optimization.
                Default: False
            **kwargs: Additional optional parameters for forward compatibility and configuration extension.
        """
        self.enable = enable
        self.enable_static_kernel = enable_static_kernel
        self.fuse_norm_quant = fuse_norm_quant
        self.fuse_qknorm_rope = fuse_qknorm_rope
        self.fuse_allreduce_rms = fuse_allreduce_rms


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
                    "Xlite graph mode is not compatible with pipeline parallelism. "
                    "Please set pipeline_parallel_size to 1."
                )
            if vllm_config.cache_config.block_size != 128:
                raise RuntimeError(
                    "Xlite graph mode is only compatible with block_size of 128. Please set block_size to 128."
                )


class WeightPrefetchConfig:
    """
    Configuration Object for weight_prefetch_config from additional_config
    """

    mlp_pre_version_compatibale_config: dict = {}

    prefetch_ratio: dict = {
        "attn": {
            "qkv": 1.0,
            "o": 1.0,
        },
        "moe": {"gate_up": 0.8},
        "mlp": {"gate_up": 1, "down": 1.0},
    }

    def __init__(self, weight_prefetch_config: dict):
        self.enabled = weight_prefetch_config.get("enabled", False)
        self.prefetch_ratio = weight_prefetch_config.get("prefetch_ratio", self.prefetch_ratio)

    def set_mlp_pre_version_compatibale_config(self, gate_up_prefetch_size: int, down_prefetch_size: int):
        config = {
            "gate_up": gate_up_prefetch_size,
            "down": down_prefetch_size,
        }
        self.mlp_pre_version_compatibale_config = config


class EplbConfig:
    """
    Configuration Object for xlite_graph_config from additional_config
    """

    _defaults = {
        "dynamic_eplb": False,
        "expert_map_path": None,
        "expert_heat_collection_interval": 400,
        "algorithm_execution_interval": 30,
        "expert_map_record_path": None,
        "num_redundant_experts": 0,
        "eplb_policy_type": 1,
    }

    def __init__(self, user_config: dict | None = None):
        if user_config is None:
            user_config = {}
        self.config = self._defaults.copy()
        if user_config and isinstance(user_config, dict):
            for key, value in user_config.items():
                if key in self.config:
                    self.config[key] = value
                else:
                    raise ValueError(f"Config has no attribute '{key}'")

        self._validate_config()

    def __getattr__(self, key):
        if key in self.config:
            return self.config[key]
        raise AttributeError(f"Config has no attribute '{key}'")

    def _validate_config(self):
        if self.expert_map_path is not None:
            logger.info(f"The expert_map is {self.config['dynamic_eplb']}")
            if self.expert_map_path[-5:] != ".json":
                raise TypeError("The expert_map is not json.")
            if not os.path.exists(self.expert_map_path):
                raise ValueError("The expert_map is not exist.")
        if self.expert_map_record_path is not None:
            self.config["dynamic_eplb"] = True
            if self.expert_map_record_path[-5:] != ".json":
                raise TypeError("The expert_map_record_path is not json.")
            dirname = os.path.dirname(self.expert_map_record_path)
            os.makedirs(dirname, exist_ok=True)
        for key in ["expert_heat_collection_interval", "algorithm_execution_interval", "num_redundant_experts"]:
            if not isinstance(self.config[key], int):
                raise TypeError(f"{key} must be an integer")
            if self.config[key] < 0:  # type: ignore
                raise ValueError(f"{key} must greater than 0; got {self.config[key]} instead")
        if self.eplb_policy_type not in [0, 1, 2, 3]:
            raise ValueError("eplb_policy_type must in [0, 1, 2, 3]")
        if self.config["dynamic_eplb"]:
            assert (
                os.getenv("DYNAMIC_EPLB", "false").lower() in ("true", "1")
                or os.getenv("EXPERT_MAP_RECORD", "false") == "true"
            ), "The environment variable DYNAMIC_EPLB or EXPERT_MAP_RECORD of the ePLB must be set to true."

        logger.info(f"Dynamic EPLB is {self.config['dynamic_eplb']}")
        logger.info(f"The number of redundant experts is {self.config['num_redundant_experts']}")


_ASCEND_CONFIG: AscendConfig | None = None


def init_ascend_config(vllm_config):
    additional_config = vllm_config.additional_config if vllm_config.additional_config is not None else {}
    refresh = additional_config.get("refresh", False) if additional_config else False
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
        raise RuntimeError("Ascend config is not initialized. Please call init_ascend_config first.")
    return _ASCEND_CONFIG
