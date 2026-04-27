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
from typing import TYPE_CHECKING, Any

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

        weight_prefetch_config = additional_config.get("weight_prefetch_config", {})
        self.weight_prefetch_config = WeightPrefetchConfig(weight_prefetch_config)

        profiling_chunk_config = additional_config.get("profiling_chunk_config", {})
        self.profiling_chunk_config = ProfilingChunkConfig(profiling_chunk_config)
        if self.profiling_chunk_config.enabled and vllm_config.parallel_config.pipeline_parallel_size <= 1:
            raise ValueError(
                "profiling_chunk_config requires pipeline parallelism (pp > 1). "
                "Please set --pipeline-parallel-size to a value greater than 1, "
                "or disable profiling_chunk_config."
            )

        from vllm_ascend import envs as ascend_envs

        if self.profiling_chunk_config.enabled and ascend_envs.VLLM_ASCEND_BALANCE_SCHEDULING:
            raise ValueError(
                "profiling_chunk_config and balance scheduling (VLLM_ASCEND_BALANCE_SCHEDULING) "
                "cannot be enabled at the same time. Please disable one of them."
            )

        # Dump / PrecisionDebugger configuration
        self.dump_config_path = additional_config.get("dump_config_path", None)
        self.layer_sharding = additional_config.get("layer_sharding", None)
        if self.layer_sharding:
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
        # PD-disaggregated only (kv_producer/kv_consumer); invalid in PD-mixed (kv_both / no kv_transfer_config).
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
                # Total KV heads from vLLM's resolved architecture (ModelArchConfigConvertor).
                num_kv_head = vllm_config.model_config.get_total_num_kv_heads()
                if not num_kv_head or num_kv_head < 1:
                    raise ValueError(
                        "Could not determine a positive total KV head count for PD "
                        "disaggregation (pd_tp_ratio > 1). Check that the model config "
                        "is compatible with vLLM."
                    )
                self.num_head_replica = prefill_tp_size // num_kv_head if prefill_tp_size >= num_kv_head else 1
                prefill_tp_size = min(prefill_tp_size, num_kv_head)
                decode_tp_size = min(decode_tp_size, num_kv_head)
                self.pd_head_ratio = prefill_tp_size // decode_tp_size

            if self.pd_tp_ratio == 0:
                raise AssertionError("Only support P node tp size lagger then D node tp size")
        self.SLO_limits_for_dynamic_batch = additional_config.get("SLO_limits_for_dynamic_batch", -1)
        from vllm_ascend.utils import get_flashcomm2_config_and_validate

        self.flashcomm2_oproj_tensor_parallel_size = get_flashcomm2_config_and_validate(self, vllm_config)
        # We find that _npu_paged_attention still performs better than
        # npu_fused_infer_attention_score in some cases. We allow to execute
        # _npu_paged_attention in this cases. This should be removed once
        # npu_fused_infer_attention_score performs better on all scenarios.
        self.pa_shape_list = additional_config.get("pa_shape_list", [])

        # when enable_async_exponential is True, AscendSampler will be different from vllm Sampler,
        # which make batch_invariant mode not working.
        # so we disable async exponential when batch_invariant mode is enabled.
        import vllm.envs as envs

        self.enable_async_exponential = (
            bool(additional_config.get("enable_async_exponential", False)) and not envs.VLLM_BATCH_INVARIANT
        )

        use_sparse = hasattr(vllm_config.model_config, "hf_text_config") and hasattr(
            vllm_config.model_config.hf_text_config, "index_topk"
        )

        self.enable_kv_nz = additional_config.get("enable_kv_nz", False)
        if self.enable_kv_nz:
            if not vllm_config.model_config.is_deepseek_mla or use_sparse:
                raise RuntimeError("enable_kv_nz is only supported for mla currently.")
            if vllm_config.kv_transfer_config is None or not vllm_config.kv_transfer_config.is_kv_consumer:
                raise NotImplementedError(
                    "enable_kv_nz is only supported in pd scenario and can only be used in D node."
                )

        from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

        # Disable Sparse C8 for A5
        # A5 has not been fully validated for this path and may carry hidden risks.
        # TODO(rjg-lyh): Enable A5 support after sufficient validation.
        self.enable_sparse_c8 = (
            additional_config.get("enable_sparse_c8", False)
            and use_sparse
            and get_ascend_device_type() != AscendDeviceType.A5
        )
        quant_config = getattr(vllm_config, "quant_config", None)
        self._sparse_c8_layer_ids, self._sparse_c8_layer_names = self._parse_sparse_c8_layers_from_quant_config(
            quant_config
        )
        self._sparse_c8_layer_filter_enabled = self._has_sparse_c8_layer_config(quant_config)
        self.enable_sp_by_pass = (
            vllm_config.model_config is not None
            and not vllm_config.model_config.enforce_eager
            and vllm_config.compilation_config.pass_config.enable_sp
        )

        # Enable dispatch/combine op inter-node communication by ROCE
        self.enable_mc2_hierarchy_comm = additional_config.get("enable_mc2_hierarchy_comm", False)

        self.mix_placement = additional_config.get("mix_placement", False)
        self._check_mix_placement()

    def _check_mix_placement(self):
        if self.mix_placement:
            if self.enable_shared_expert_dp or self.multistream_overlap_shared_expert:
                raise ValueError("Mix placement is not supported with shared expert DP or multistream overlap.")

    @staticmethod
    def _has_sparse_c8_layer_config(quant_config: Any) -> bool:
        quant_description = getattr(quant_config, "quant_description", None)
        if not isinstance(quant_description, dict):
            return False
        return any(isinstance(key, str) and key.endswith(".indexer.quant_type") for key in quant_description)

    @classmethod
    def _parse_sparse_c8_layers_from_quant_config(cls, quant_config: Any) -> tuple[set[int], set[str]]:
        quant_description = getattr(quant_config, "quant_description", None)
        if not isinstance(quant_description, dict):
            return set(), set()

        layer_ids: set[int] = set()
        layer_names: set[str] = set()
        suffix = ".indexer.quant_type"
        from vllm.model_executor.models.utils import extract_layer_index

        for key, value in quant_description.items():
            if not isinstance(key, str) or not key.endswith(suffix):
                continue
            if value != "INT8_DYNAMIC":
                continue
            layer_name = key[: -len(suffix)].rstrip(".")
            if not layer_name:
                continue
            layer_names.add(layer_name)
            layer_ids.update({extract_layer_index(layer_name)})
        return layer_ids, layer_names

    def is_sparse_c8_layer(self, layer_name: str | None) -> bool:
        if not self.enable_sparse_c8:
            return False
        if not self._sparse_c8_layer_filter_enabled:
            return True
        if layer_name is None:
            return False

        normalized_layer_name = layer_name.rstrip(".")
        if any(
            normalized_layer_name == candidate or normalized_layer_name.startswith(f"{candidate}.")
            for candidate in self._sparse_c8_layer_names
        ):
            return True
        from vllm.model_executor.models.utils import extract_layer_index

        layer_ids = {extract_layer_index(normalized_layer_name)}
        return any(layer_id in self._sparse_c8_layer_ids for layer_id in layer_ids)

    @staticmethod
    def _get_compile_ranges(compilation_config):
        return compilation_config.compile_ranges_endpoints or []

    @staticmethod
    def _set_compile_ranges(compilation_config, value):
        compilation_config.compile_ranges_endpoints = value

    def update_compile_ranges_split_points(self):
        vllm_config = self.vllm_config
        if self.ascend_compilation_config.enable_npugraph_ex:
            if self.ascend_compilation_config.fuse_allreduce_rms:
                from vllm_ascend.compilation.passes.allreduce_rmsnorm_fusion_pass import ALLREDUCE_NORM_FUSE_THRESHOLD

                new_compile_ranges_split_points = self._get_compile_ranges(vllm_config.compilation_config)
                new_compile_ranges_split_points.append(ALLREDUCE_NORM_FUSE_THRESHOLD)
                new_compile_ranges_split_points = sorted(new_compile_ranges_split_points)
                self._set_compile_ranges(vllm_config.compilation_config, new_compile_ranges_split_points)
                logger.debug(
                    "set compile_ranges_split_points to "
                    "{new_compile_ranges_split_points} for matmul and allreduce fusion"
                )

        else:
            new_compile_ranges_split_points = self._get_compile_ranges(vllm_config.compilation_config)
            if vllm_config.additional_config.get("ascend_compilation_config", {}).get("fuse_allreduce_rms", True):
                from vllm_ascend.compilation.passes.allreduce_rmsnorm_fusion_pass import ALLREDUCE_NORM_FUSE_THRESHOLD

                new_compile_ranges_split_points.append(ALLREDUCE_NORM_FUSE_THRESHOLD)
                new_compile_ranges_split_points = sorted(new_compile_ranges_split_points)
                self._set_compile_ranges(vllm_config.compilation_config, new_compile_ranges_split_points)
                logger.debug(
                    "set compile_ranges_split_points to "
                    "{new_compile_ranges_split_points} for matmul and allreduce fusion"
                )

            if len(new_compile_ranges_split_points) > len(self._get_compile_ranges(vllm_config.compilation_config)):
                new_compile_ranges_split_points = sorted(new_compile_ranges_split_points)
                self._set_compile_ranges(vllm_config.compilation_config, new_compile_ranges_split_points)


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
        self,
        enable_npugraph_ex: bool = True,
        enable_static_kernel: bool = False,
        fuse_norm_quant: bool = True,
        fuse_qknorm_rope: bool = True,
        fuse_allreduce_rms: bool = False,
        **kwargs,
    ):
        """
        Initialize the configuration.

        Args:
            enable_npugraph_ex (bool): Whether to enable npugraph_ex backend.
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
        self.fuse_norm_quant = fuse_norm_quant
        self.fuse_qknorm_rope = fuse_qknorm_rope
        self.fuse_allreduce_rms = fuse_allreduce_rms
        self.enable_npugraph_ex = enable_npugraph_ex
        self.enable_static_kernel = enable_static_kernel
        self.fuse_muls_add = kwargs.get("fuse_muls_add", True)
        if self.enable_static_kernel:
            assert self.enable_npugraph_ex, "Static kernel generation requires npugraph_ex to be enabled."


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
                logger.warning(
                    f"Current cache block size is {vllm_config.cache_config.block_size}, which may not be optimal or "
                    f"compatible with xlite graph mode. The recommended block size for xlite graph mode is 128."
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
        "moe": {"gate_up": 0.8},
        "mlp": {"gate_up": 1.0, "down": 1.0},
    }

    def __init__(self, weight_prefetch_config: dict):
        self.enabled = weight_prefetch_config.get("enabled", False)
        self.prefetch_ratio = weight_prefetch_config.get("prefetch_ratio", self.prefetch_ratio)


class ProfilingChunkConfig:
    """Configuration for profiling-based dynamic chunk sizing.

    When enabled, the scheduler profiles prefill latency during initialization
    and uses a quadratic model to predict optimal chunk sizes at runtime.

    Usage (online)::

        vllm serve <model> --additional-config '{"profiling_chunk_config": {"enabled": true}}'

    Usage (offline)::

        llm = LLM(model, additional_config={"profiling_chunk_config": {"enabled": true}})
    """

    def __init__(self, config: dict | None = None):
        if config is None:
            config = {}
        self.enabled: bool = config.get("enabled", False)
        self.smooth_factor: float = float(config.get("smooth_factor", 0.8))
        self.min_chunk: int = int(config.get("min_chunk", 4096))
        self._validate()

    def _validate(self):
        if not (0 < self.smooth_factor <= 1.0):
            raise ValueError(f"profiling_chunk_config.smooth_factor must be in (0, 1], got {self.smooth_factor}")
        if self.min_chunk <= 0:
            raise ValueError(f"profiling_chunk_config.min_chunk must be positive, got {self.min_chunk}")


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
            logger.info(f"The expert_map is {self.expert_map_path}")
            if self.expert_map_path[-5:] != ".json":
                raise TypeError("The expert_map is not json.")
            if not (os.path.exists(self.expert_map_path) and os.access(self.expert_map_path, os.R_OK)):
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
            ), "The environment variable DYNAMIC_EPLB or EXPERT_MAP_RECORD of the EPLB must be set to true."

        logger.info(f"Dynamic EPLB is {self.config['dynamic_eplb']}")
        logger.info(f"The number of redundant experts is {self.config['num_redundant_experts']}")


_ASCEND_CONFIG: AscendConfig | None = None


def _is_ascend_config_initialized(config: AscendConfig | None) -> bool:
    """Check whether a config object has essential initialized fields.

    Some unit tests monkeypatch ``AscendConfig.__init__`` to bypass heavy
    initialization. In that case, the singleton cache can be polluted with a
    partially initialized instance. This guard prevents reusing such instances
    across tests.
    """
    if config is None:
        return False
    return hasattr(config, "ascend_compilation_config") and hasattr(config, "eplb_config")


def init_ascend_config(vllm_config):
    additional_config = vllm_config.additional_config if vllm_config.additional_config is not None else {}
    refresh = additional_config.get("refresh", False) if additional_config else False
    global _ASCEND_CONFIG
    if _ASCEND_CONFIG is not None and not refresh and _is_ascend_config_initialized(_ASCEND_CONFIG):
        return _ASCEND_CONFIG
    new_config = AscendConfig(vllm_config)
    if _is_ascend_config_initialized(new_config):
        _ASCEND_CONFIG = new_config
    else:
        logger.warning("Ascend config instance is not fully initialized; skip singleton cache update.")
    return new_config


def clear_ascend_config():
    global _ASCEND_CONFIG
    _ASCEND_CONFIG = None


def get_ascend_config():
    global _ASCEND_CONFIG
    if _ASCEND_CONFIG is None or not _is_ascend_config_initialized(_ASCEND_CONFIG):
        raise RuntimeError("Ascend config is not initialized. Please call init_ascend_config first.")
    return _ASCEND_CONFIG
