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
from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import ConfigDict, TypeAdapter, model_validator
from vllm.logger import logger
from vllm.utils.math_utils import cdiv

from vllm_ascend.config_utils import config

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_MEGA_MOE_SUPPORTED = None


def is_mega_moe_supported() -> bool:
    """Whether the megamoe op is available at runtime.

    Always read _MEGA_MOE_SUPPORTED through this accessor instead of
    ``from ascend_config import _MEGA_MOE_SUPPORTED``: the global is rebound
    during config init (AscendConfig._validate_user_input_ranges rolls back
    megamoe), and a direct import binds a stale snapshot for the bool.
    """
    global _MEGA_MOE_SUPPORTED
    if _MEGA_MOE_SUPPORTED is None:
        _MEGA_MOE_SUPPORTED = importlib.util.find_spec("cann_ops_transformer") is not None
    return _MEGA_MOE_SUPPORTED


def validate_additional_config_bool(value: Any, path: str) -> bool:
    """Apply the same pydantic bool rules to values read before config init."""
    try:
        return TypeAdapter(bool).validate_python(value)
    except ValueError as exc:
        raise ValueError(f"{path} must be a boolean, got {value!r}.") from exc


@config
class AscendCompilationConfig:
    """Configuration for controlling the behavior of Ascend graph optimization.

    Migrated to ``@config`` (pydantic dataclass). Hardware-profile runtime
    downgrades (disable npugraph_ex / static_kernel) and the
    static_kernel→npugraph_ex dependency check are applied in an ``after``
    model_validator.
    """

    enable_npugraph_ex: bool = True
    enable_static_kernel: bool = False
    fuse_norm_quant: bool = True
    fuse_qknorm_rope: bool = True
    fuse_muls_add: bool = True

    @model_validator(mode="after")
    def _apply_unsupported_hardware_downgrade_and_static_kernel_check(self):
        from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

        if not get_current_hardware_profile().supports(HardwareCapability.NPUGRAPH_EX):
            if self.enable_npugraph_ex:
                logger.warning("npugraph_ex is not supported by the current hardware profile. Disabling it.")
            if self.enable_static_kernel:
                logger.warning(
                    "static kernel requires npugraph_ex, which is not supported by the current hardware profile. "
                    "Disabling it."
                )
            self.enable_npugraph_ex = False
            self.enable_static_kernel = False
        if self.enable_static_kernel:
            assert self.enable_npugraph_ex, "Static kernel generation requires npugraph_ex to be enabled."
        return self


@config
class AscendFusionConfig:
    """Configuration for controlling whether to use a fused operator gmmswigluquant.

    Migrated to ``@config`` (pydantic dataclass): bool field gets lax coercion
    (``"false"``→False) and unknown keys are forbidden (``extra="forbid"``),
    fixing the ``bool("false")`` pitfall and surfacing typos.
    """

    fusion_ops_gmmswigluquant: bool = True


@config
class EplbConfig:
    """Configuration Object for ``additional_config["eplb_config"]``.

    Migrated to ``@config`` (pydantic dataclass). Unknown-key detection is now
    handled by ``extra="forbid"`` (replaces the hand-written ``unknown`` check);
    int/range/enum checks moved to an ``after`` model_validator. The
    ``__getattr__`` proxy over the internal ``self.config`` dict is removed —
    fields are accessed directly (``self.dynamic_eplb`` etc.).
    """

    dynamic_eplb: bool = False
    expert_map_path: str | None = None
    expert_heat_collection_interval: int = 600
    algorithm_execution_interval: int = 50
    expert_map_record_path: str | None = None
    num_redundant_experts: int = 0
    eplb_policy_type: int = 2
    eplb_heat_collection_stage: str = "all"
    # Model Runner V2 only. Restricts which batch phase contributes to the
    # upstream EPLB expert-load window; any prefill request marks the batch
    # as prefill.
    load_collection_phase: str = "all"

    @model_validator(mode="after")
    def _validate_config(self):
        if self.expert_map_path is not None:
            logger.info("The expert_map is %s", self.expert_map_path)
            if self.expert_map_path[-5:] != ".json":
                raise TypeError("The expert_map is not json.")
            if not (os.path.exists(self.expert_map_path) and os.access(self.expert_map_path, os.R_OK)):
                raise ValueError("The expert_map is not exist.")
        if self.expert_map_record_path is not None:
            self.dynamic_eplb = True
            if self.expert_map_record_path[-5:] != ".json":
                raise TypeError("The expert_map_record_path is not json.")
            dirname = os.path.dirname(self.expert_map_record_path)
            os.makedirs(dirname, exist_ok=True)
        for key in ["expert_heat_collection_interval", "algorithm_execution_interval", "num_redundant_experts"]:
            value = getattr(self, key)
            if not isinstance(value, int):
                raise TypeError(f"{key} must be an integer")
            if value < 0:
                raise ValueError(f"{key} must greater than 0; got {value} instead")
        if self.eplb_policy_type not in [0, 1, 2, 3]:
            raise ValueError("eplb_policy_type must in [0, 1, 2, 3]")
        if self.dynamic_eplb:
            assert (
                os.getenv("DYNAMIC_EPLB", "false").lower() in ("true", "1")
                or os.getenv("EXPERT_MAP_RECORD", "false") == "true"
            ), "The environment variable DYNAMIC_EPLB or EXPERT_MAP_RECORD of the EPLB must be set to true."
        if self.eplb_heat_collection_stage not in ["all", "prefill", "decode"]:
            raise ValueError('eplb_heat_collection_stage must be one of ["all", "prefill", "decode"]')
        if self.load_collection_phase not in ["all", "prefill", "decode"]:
            raise ValueError('load_collection_phase must be one of ["all", "prefill", "decode"]')

        logger.info("Dynamic EPLB is %s", self.dynamic_eplb)
        logger.info("The number of redundant experts is %s", self.num_redundant_experts)
        return self


@config
class RejectionSamplerConfig:
    """Configuration for Block Verify and Entropy Verify in Rejection Sampler.

    Migrated to ``@config`` (pydantic dataclass). Type checks (bool/float) are
    now handled by pydantic field types; range checks moved to an ``after``
    model_validator.
    """

    enable_block_verify: bool = False
    enable_entropy_verify: bool = False
    posterior_threshold: float = 0.95
    posterior_alpha: float = 0.4

    @model_validator(mode="after")
    def _validate(self):
        if not (0 < self.posterior_threshold <= 1):
            raise ValueError(
                f"rejection_sampler_config.posterior_threshold must be in (0, 1], got {self.posterior_threshold}"
            )
        if self.posterior_alpha < 0:
            raise ValueError(f"rejection_sampler_config.posterior_alpha must be >= 0, got {self.posterior_alpha}")
        return self


@config
class RlConfig:
    """Unified defaults for reinforcement-learning workloads.

    Migrated to ``@config`` so bool values use the same pydantic lax coercion
    and unknown-key rejection as other vLLM-independent sub-configs.
    """

    enabled: bool = False
    sleep_mode_extra_cleanup: bool = False
    enable_training_consistency: bool = False
    enable_batch_invariant: bool = False

    def apply(self, ascend_config: AscendConfig) -> None:
        if not self.enabled:
            return

        if ascend_config.weight_nz_mode != 0:
            logger.warning(
                "RL config requires weight_nz_mode=0; overriding AscendConfig.weight_nz_mode from %s to 0.",
                ascend_config.weight_nz_mode,
            )
        ascend_config.weight_nz_mode = 0

        from vllm_ascend.platform import _disable_expandable_segments

        _disable_expandable_segments()

        if self.enable_batch_invariant:
            os.environ["VLLM_BATCH_INVARIANT"] = "1"

        os.environ["VLLM_SERVER_DEV_MODE"] = "1"


@config
class AscendConfig:
    """Configuration Object for additional_config from vllm.configs.

    Migrated to ``@config`` (pydantic dataclass). User-input switches are now
    typed fields with lax bool/int coercion (``"false"``→False, ``"2"``→2),
    fixing the ``bool("false")``/``"2"==2`` pitfalls. Unknown keys are
    forbidden (``extra="forbid"``). Cross-config derivations, downgrades and
    mutex checks that need ``vllm_config`` run in
    ``derive_and_validate()``, a plain method invoked explicitly by
    ``init_ascend_config`` (not a pydantic validator) — preserving original
    ordering and error messages.

    ``vllm_config`` is NOT a member of AscendConfig (neither a declared
    pydantic field nor a plain instance attribute). Pydantic handles only
    type/range/enum validation here; the factory passes ``vllm_config``
    explicitly to ``derive_and_validate()``. This keeps AscendConfig a pure
    Ascend-configuration container, free of the heavy upstream VllmConfig
    graph (and drops ``arbitrary_types_allowed``, which existed only for the
    former ``vllm_config`` field).

    Example: pass this dict via ``--additional-config '<json>'``. All keys
    are optional; the values below show every supported key with its default
    value (see the mutex/validation notes below before copying).

        {
            "refresh": false,
            "enable_cpu_binding": true,
            "multistream_dsv4_dsa_overlap": true,
            "enable_prefill_mc2": false,
            "multistream_overlap_shared_expert": false,
            "enable_kv_nz": false,
            "enable_mc2_hierarchy_comm": false,
            "enable_reduce_sample": false,
            "enable_dsa_cp": false,
            "enable_force_eplb": false,
            "draft_window_size": null,
            "mix_placement": false,
            "pa_shape_list": [],
            "mega_moe_max_tokens": 65536,
            "ascend_log_path": "~/ascend/log/vllm_ascend",
            "c8_enable_reshape_optim": false,
            "enable_fused_mc2": 0,
            "enable_mlapo": true,
            "mlapo_keep_prefill_weights": false,
            "msmonitor_use_daemon": false,
            "enable_transpose_kv_cache_by_block": true,
            "weight_nz_mode": 1,
            "enable_shared_expert_dp": false,
            "enable_sparse_sfa_c8": false,
            "enable_sparse_li_c8": false,
            "ascend_compilation_config": {
                "enable_npugraph_ex": true,
                "enable_static_kernel": false,
                "fuse_norm_quant": true,
                "fuse_qknorm_rope": true,
                "fuse_muls_add": true
            },
            "ascend_fusion_config": {
                "fusion_ops_gmmswigluquant": true
            },
            "eplb_config": {
                "dynamic_eplb": false,
                "expert_map_path": null,
                "expert_heat_collection_interval": 600,
                "algorithm_execution_interval": 50,
                "expert_map_record_path": null,
                "num_redundant_experts": 0,
                "eplb_policy_type": 2,
                "eplb_heat_collection_stage": "all",
                "load_collection_phase": "all"
            },
            "rejection_sampler_config": {
                "enable_block_verify": false,
                "enable_entropy_verify": false,
                "posterior_threshold": 0.95,
                "posterior_alpha": 0.4
            },
            "rl_config": {
                "enabled": false,
                "sleep_mode_extra_cleanup": false,
                "enable_training_consistency": false,
                "enable_batch_invariant": false
            },
            "xlite_graph_config": {
                "enabled": false,
                "full_mode": false
            },
            "finegrained_tp_config": {
                "oproj_tensor_parallel_size": 0,
                "lmhead_tensor_parallel_size": 0,
                "embedding_tensor_parallel_size": 0,
                "mlp_tensor_parallel_size": 0,
                "olora_tensor_parallel_size": 0
            },
            "scheduler_config": {
                "enable_balance_scheduling": false,
                "recompute_scheduler_enable": false,
                "short_request_first_config": {
                    "enabled": false,
                    "threshold": 256,
                    "long_max_wait_ms": 0.0
                },
                "profiling_chunk_config": {
                    "enabled": false,
                    "smooth_factor": 1.0,
                    "min_chunk": 4096,
                    "need_timing": null,
                    "max_fit_chunk": 30
                },
                "batch_job_sched_config": {
                    "enabled": false,
                    "max_jobs": 20,
                    "reserve_margin_blocks": 2,
                    "reserve_max_blocks": 8,
                    "low_available_tokens_threshold": 4096,
                    "short_decode_token_threshold": 32
                },
                "dyntra_lb_config": {
                    "enabled": false,
                    "enable_diagnostics": false,
                    "mode": "dynamic",
                    "start_step": 250,
                    "end_step": -1,
                    "bubble_threshold": 5.0,
                    "long_req_block_threshold": 700,
                    "dynamic_max_step": 256
                }
            },
            "dynamic_spec_config": {
                "method": null,
                "method_params": {}
            },
            "sparse_kv_offload_config": {
                "enabled": false,
                "topk_buffer_size": 4096,
                "dram_size_per_dp_GB": 128,
                "keep_device_kv_cache": false
            }
        }

    Additional notes:

    - Unknown keys are rejected (``extra="forbid"``); a typo fails fast.
    - ``dump_config_path`` (str path) / ``dump_config`` (inline dict, mutually
      exclusive with ``dump_config_path``; materialized by the factory to
      ``.vllm_ascend/msprobe/msprobe_dump_config.json``) are msprobe dump
      options consumed by the factory before schema validation.
    - ``refresh`` forces reconstruction of the cached singleton config.
    - Pure-derived fields computed automatically and not user-settable:
      ``enable_sp_by_pass``, ``pd_tp_ratio``, ``pd_head_ratio``,
      ``num_head_replica``.
    - Deprecated top-level scheduler keys (``enable_balance_scheduling``,
      ``recompute_scheduler_enable``, ``short_request_first_config``,
      ``profiling_chunk_config``, ``batch_job_sched_config``) still resolve,
      but emit deprecation warnings; prefer nesting under
      ``scheduler_config`` as shown above.
    """

    model_config = ConfigDict(extra="forbid")

    # ---- user-input switches: bool/int/list/str, auto type validation ----
    enable_cpu_binding: bool = True
    multistream_dsv4_dsa_overlap: bool = True
    enable_prefill_mc2: bool = False
    multistream_overlap_shared_expert: bool = False
    enable_kv_nz: bool = False
    enable_mc2_hierarchy_comm: bool = False  # deprecated, will be replaced by mc2_comm_alg = "hierarchy"
    enable_reduce_sample: bool = False
    enable_dsa_cp: bool = False
    enable_force_eplb: bool = False
    draft_window_size: int | None = None
    mix_placement: bool = False
    pa_shape_list: list[Any] = dataclasses.field(default_factory=list)
    # Per-rank token capacity after dispatch in the fused MC2/MegaMoe path.
    # The same value is passed as dispatch_ffn_combine's max_output_size
    # and CANN MegaMoe buffer's max_recv_token_num.
    # This is a reference value: if the actual per-rank received token
    # count exceeds it, tokens may be truncated, causing precision
    # degradation. Do not set it too large because workspace memory scales
    # linearly with this value. Default 65536.
    mega_moe_max_tokens: int = 65536
    ascend_log_path: str = dataclasses.field(
        default_factory=lambda: os.path.join(os.path.expanduser("~"), "ascend", "log", "vllm_ascend")
    )
    dump_config_path: str | None = None
    c8_enable_reshape_optim: bool = False
    mc2_comm_alg: Literal["", "fullmesh", "hierarchy", "fullmesh_v2"] = ""

    # ---- A-family (envs fallback): default = envs module value, before-validator injects ----
    enable_fused_mc2: int = 0
    enable_mlapo: bool = True
    # When True, keep MLAPO prefill weights on NPU instead of freeing them
    # on kv_consumer D nodes. Trades NPU memory for stability — D nodes have
    # normal local-prefill paths (recompute / fallback / preempt) that crash
    # when the weights are freed (issue #11882). Default False (preserve
    # existing memory-saving behavior).
    mlapo_keep_prefill_weights: bool = False
    msmonitor_use_daemon: bool = False
    enable_transpose_kv_cache_by_block: bool = True
    weight_nz_mode: int = 1

    # ---- sub-configs (no vllm_config dep): pydantic dict→dataclass coercion ----
    ascend_compilation_config: AscendCompilationConfig = dataclasses.field(default_factory=AscendCompilationConfig)
    ascend_fusion_config: AscendFusionConfig = dataclasses.field(default_factory=AscendFusionConfig)
    eplb_config: EplbConfig = dataclasses.field(default_factory=EplbConfig)
    rejection_sampler_config: RejectionSamplerConfig = dataclasses.field(default_factory=RejectionSamplerConfig)
    rl_config: RlConfig = dataclasses.field(default_factory=RlConfig)

    # ---- sub-configs declared later in this module ----
    # Lambdas defer class lookup until construction, after module initialization.
    xlite_graph_config: XliteGraphConfig = dataclasses.field(default_factory=lambda: XliteGraphConfig())
    finegrained_tp_config: FinegrainedTPConfig = dataclasses.field(default_factory=lambda: FinegrainedTPConfig())
    scheduler_config: SchedulerConfig = dataclasses.field(default_factory=lambda: SchedulerConfig())
    dynamic_spec_config: DynamicSpecConfig = dataclasses.field(default_factory=lambda: DynamicSpecConfig())
    # Still factory-injected: construction depends on vllm_config.
    sparse_kv_offload_config: Any = dataclasses.field(kw_only=True)

    # ---- derived fields: sentinel default, after-validator overwrites ----
    enable_shared_expert_dp: bool = False
    enable_sp_by_pass: bool = False
    enable_sparse_sfa_c8: bool = False
    enable_sparse_li_c8: bool = False
    pd_tp_ratio: int = 1
    pd_head_ratio: int = 1
    num_head_replica: int = 1

    # ---- private derived state (init=False) ----
    _sparse_li_c8_layer_ids: set[int] = dataclasses.field(default_factory=set, init=False, repr=False)
    _sparse_li_c8_layer_names: set[str] = dataclasses.field(default_factory=set, init=False, repr=False)
    _sparse_li_c8_layer_filter_enabled: bool = dataclasses.field(default=False, init=False, repr=False)

    @model_validator(mode="after")
    def _validate_user_input_ranges(self):
        if self.weight_nz_mode not in (0, 1, 2):
            raise ValueError(f"weight_nz_mode must be one of 0, 1, or 2; got {self.weight_nz_mode}")
        # TODO(zzzzwwjj): remove it after deprecating `enable_mc2_hierarchy_comm`.
        if self.enable_mc2_hierarchy_comm:
            self.mc2_comm_alg = "hierarchy"
        # TODO(zzzzwwjj): Currently, there are many problems with the megamoe op.
        # We will first roll back the megamoe internally and keep `enable_fused_mc2=2`
        # to enable the megamoe for testing capabilities.
        # These codes will be removed after megamoe is ready.
        global _MEGA_MOE_SUPPORTED
        if self.enable_fused_mc2 in (0, 1):
            # When enable_fused_mc2=1, roll back to dispatch_ffn_combine.
            _MEGA_MOE_SUPPORTED = False
        elif self.enable_fused_mc2 == 2:
            _MEGA_MOE_SUPPORTED = importlib.util.find_spec("cann_ops_transformer") is not None
            self.enable_fused_mc2 = 1
        return self

    # ---- derivations + cross-config downgrades/mutex ----
    # Business validation: invoked explicitly by init_ascend_config (NOT a
    # pydantic after-validator). Preserves the original __init__ ordering —
    # multi-step downgrades are order-dependent (e.g. profiling_chunk reads
    # the max_num_batched_tokens that sequence-parallel writeback corrected).
    def derive_and_validate(self, vllm_config: VllmConfig) -> AscendConfig:
        vc = vllm_config
        if (
            self.enable_force_eplb
            and self.eplb_config.dynamic_eplb
            and vc.model_config is not None
            and vc.model_config.is_moe
        ):
            raise ValueError("enable_force_eplb cannot be mixed with dynamic_eplb.")
        self._check_mooncake_c8_kv_cache_quant(vc)

        # profiling_chunk vs min_chunk clamp
        if self.scheduler_config.profiling_chunk_config.enabled:
            max_batched = vc.scheduler_config.max_num_batched_tokens
            if max_batched < self.scheduler_config.profiling_chunk_config.min_chunk:
                logger.warning(
                    "max_num_batched_tokens is smaller than profiling_chunk_config.min_chunk. "
                    "max_num_batched_tokens=%d, min_chunk=%d. "
                    "Clamping min_chunk to %d to avoid it being silently ignored.",
                    max_batched,
                    self.scheduler_config.profiling_chunk_config.min_chunk,
                    max_batched,
                )
                self.scheduler_config.profiling_chunk_config.min_chunk = max_batched
        if self.scheduler_config.profiling_chunk_config.enabled and vc.parallel_config.pipeline_parallel_size <= 1:
            raise ValueError(
                "profiling_chunk_config requires pipeline parallelism (pp > 1). "
                "Please set --pipeline-parallel-size to a value greater than 1, "
                "or disable profiling_chunk_config."
            )

        # profiling_chunk vs balance mutex
        if self.scheduler_config.profiling_chunk_config.enabled and self.scheduler_config.enable_balance_scheduling:
            raise ValueError(
                "profiling_chunk_config and balance scheduling (enable_balance_scheduling) "
                "cannot be enabled at the same time. Please disable one of them."
            )

        # enable_shared_expert_dp = val and ep and tp>1
        from vllm_ascend.utils import enable_sp

        self.enable_shared_expert_dp = (
            self.enable_shared_expert_dp
            and vc.parallel_config.enable_expert_parallel
            and vc.parallel_config.tensor_parallel_size > 1
        )

        # DSA CP is only applicable to models with an indexer (for example,
        # DeepSeek V3.2/V4). Resolve this while vllm_config is explicitly
        # available so runtime reads do not depend on vLLM's temporary config
        # context.
        has_indexer = hasattr(vc.model_config, "hf_text_config") and hasattr(
            vc.model_config.hf_text_config, "index_topk"
        )
        self.enable_dsa_cp = self.enable_dsa_cp and has_indexer

        # Sequence-parallel max_num_batched_tokens divisibility writeback
        if vc.parallel_config.prefill_context_parallel_size > 1 and enable_sp(vllm_config=vc):
            tp_pcp_size = vc.parallel_config.tensor_parallel_size * vc.parallel_config.prefill_context_parallel_size
            if vc.scheduler_config.max_num_batched_tokens % tp_pcp_size != 0:
                vc.scheduler_config.max_num_batched_tokens = (
                    cdiv(vc.scheduler_config.max_num_batched_tokens, tp_pcp_size) * tp_pcp_size
                )
                logger.warning_once(
                    "When using sequence parallelism, the max_num_batched_tokens should be divisible "
                    "by tp_size * pcp_size (%s). It has been adjusted to %s.",
                    str(tp_pcp_size),
                    str(vc.scheduler_config.max_num_batched_tokens),
                )

        # finegrained_tp requires recompute_scheduler
        if (
            self.finegrained_tp_config.oproj_tensor_parallel_size > 0
            or self.finegrained_tp_config.embedding_tensor_parallel_size > 0
        ) and not self.scheduler_config.recompute_scheduler_enable:
            raise AssertionError(
                "oproj_tensor_parallel_size / embedding_tensor_parallel_size "
                "require recompute_scheduler_enable=true: their cross-DP HCCL "
                "collectives need uniform num_tokens across DP ranks, which is "
                "only guaranteed when the recompute scheduler is enabled."
            )

        # enable_fused_mc2 enum + MiniMax mutex + multistream auto-disable
        assert self.enable_fused_mc2 in (0, 1), f"enable_fused_mc2 must be 0 or 1, got {self.enable_fused_mc2}"
        model_architectures = getattr(vc.model_config, "architectures", None) or []
        assert not (
            self.enable_fused_mc2 == 1
            and any(architecture.startswith("MiniMaxM3") for architecture in model_architectures)
        ), "MiniMax M3 does not support enable_fused_mc2=1. Please set additional_config.enable_fused_mc2 to 0."
        if self.enable_fused_mc2 == 1 and self.multistream_overlap_shared_expert:
            self.multistream_overlap_shared_expert = False
            logger.warning_once(
                "enable_fused_mc2 and multistream_overlap_shared_expert "
                "cannot be enabled at the same time. Setting multistream_overlap_shared_expert to False."
            )
        if self.enable_fused_mc2 == 1 and _MEGA_MOE_SUPPORTED and not self._is_megamoe_supported_by_config(vc):
            self.enable_fused_mc2 = 0
            logger.warning_once(
                "MegaMoe is not supported for this model config; additional_config.enable_fused_mc2 will be set to 0."
            )

        # mlapo_keep_prefill_weights preconditions: the prefill weights are only
        # freed by MLAPO in the MLA attention path, so the keep switch is only
        # meaningful under those same conditions. Fail fast on a no-op / mistaken
        # config instead of silently leaving the switch ineffective (issue #11882).
        if self.mlapo_keep_prefill_weights:
            if not self.enable_mlapo:
                raise ValueError(
                    "mlapo_keep_prefill_weights=True requires enable_mlapo=True. "
                    "The prefill weights are only freed when MLAPO is enabled."
                )
            if vc.model_config is None or not vc.model_config.is_deepseek_mla:
                raise ValueError(
                    "mlapo_keep_prefill_weights=True is only supported for MLA models "
                    "(e.g., DeepSeek). The prefill weights are only freed by MLAPO "
                    "in the MLA attention path."
                )

        # PD tp_ratio / head_ratio / num_head_replica derivation
        if vc.kv_transfer_config is not None and vc.model_config is not None and not vc.model_config.is_deepseek_mla:
            prefill_tp_size = vc.kv_transfer_config.get_from_extra_config("prefill", {"tp_size": 1})["tp_size"]
            decode_tp_size = vc.kv_transfer_config.get_from_extra_config("decode", {"tp_size": 1})["tp_size"]
            assert prefill_tp_size % decode_tp_size == 0, "Prefill TP size must be divisible by Decode TP size."
            self.pd_tp_ratio = prefill_tp_size // decode_tp_size
            if self.pd_tp_ratio > 1:
                num_kv_head = vc.model_config.get_total_num_kv_heads()
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

        # enable_kv_nz preconditions
        if self.enable_kv_nz:
            if vc.model_config is None:
                raise RuntimeError("enable_kv_nz requires a valid model_config.")
            from vllm_ascend.utils import model_uses_sfa_sparse

            use_sparse = model_uses_sfa_sparse(vc.model_config)
            if not vc.model_config.is_deepseek_mla or use_sparse:
                raise RuntimeError("enable_kv_nz is only supported for mla currently.")
            if vc.kv_transfer_config is None or not vc.kv_transfer_config.is_kv_consumer:
                raise NotImplementedError(
                    "enable_kv_nz is only supported in pd scenario and can only be used in D node."
                )

        # sparse c8 + reshape optim derivation
        from vllm_ascend.utils import model_uses_sfa_sparse

        use_sparse = model_uses_sfa_sparse(vc.model_config)
        self.enable_sparse_sfa_c8 = self.enable_sparse_sfa_c8 and use_sparse
        self.enable_sparse_li_c8 = self.enable_sparse_li_c8 and use_sparse
        # c8_enable_reshape_optim is a user input field now; keep the original
        # semantics: only meaningful when enable_sparse_li_c8 is true.
        self.c8_enable_reshape_optim = self.enable_sparse_li_c8 and self.c8_enable_reshape_optim
        quant_config = getattr(vc, "quant_config", None)
        (
            self._sparse_li_c8_layer_ids,
            self._sparse_li_c8_layer_names,
        ) = self._parse_sparse_li_c8_layers_from_quant_config(quant_config)
        self._sparse_li_c8_layer_filter_enabled = self._has_sparse_li_c8_layer_config(quant_config)
        self.enable_sp_by_pass = (
            vc.model_config is not None
            and not vc.model_config.enforce_eager
            and vc.compilation_config.pass_config.enable_sp
        )

        self._validate_mc2_comm_alg(vc)

        # mega_moe_max_tokens range
        if self.mega_moe_max_tokens <= 0:
            raise ValueError(f"mega_moe_max_tokens must be a positive integer, got {self.mega_moe_max_tokens}")

        # Enable optimized reduce sampling scheme. Preserve the safeguards
        # added on main while consuming the already-validated typed field.
        if self.enable_reduce_sample:
            logger.warning_once("enable_reduce_sample is an experimental feature. Use with caution.")
            if self.finegrained_tp_config.lmhead_tensor_parallel_size > 0:
                raise ValueError(
                    "enable_reduce_sample is incompatible with "
                    "finegrained_tp_config.lmhead_tensor_parallel_size. "
                    "Please disable one of them."
                )
            kv_transfer_config = getattr(vc, "kv_transfer_config", None)
            kv_role = getattr(kv_transfer_config, "kv_role", None)
            if kv_role == "kv_producer":
                raise ValueError(
                    "enable_reduce_sample is not supported on PD-disaggregated "
                    "scenarios. Please disable enable_reduce_sample."
                )

        # mix_placement mutex
        self._check_mix_placement()

        # sparse KV offload vs sparse SFA C8 main cache mutex
        self._validate_sparse_c8_kv_offload_compatibility()
        return self

    def _validate_mc2_comm_alg(self, vllm_config: VllmConfig) -> None:
        from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

        hardware_profile = get_current_hardware_profile()
        if self.mc2_comm_alg == "fullmesh_v2" and not hardware_profile.supports(
            HardwareCapability.MC2_FULLMESH_V2_COMM
        ):
            raise NotImplementedError("mc2_comm_alg == 'fullmesh_v2' is not supported by the current hardware profile.")

        if self.mc2_comm_alg != "hierarchy":
            return

        if not hardware_profile.supports(HardwareCapability.MC2_HIERARCHY_COMM):
            raise NotImplementedError("mc2_comm_alg == 'hierarchy' is not supported by the current hardware profile.")

        num_logical_experts = vllm_config.model_config.get_num_experts()
        num_redundant_experts = self.eplb_config.num_redundant_experts if self.eplb_config.dynamic_eplb else 0
        num_experts = num_logical_experts + num_redundant_experts
        if num_experts > 512:
            raise ValueError(
                "mc2_comm_alg == 'hierarchy' supports at most 512 experts, "
                f"but got {num_experts} experts "
                f"({num_logical_experts} logical experts + {num_redundant_experts} EPLB redundant experts)."
            )

        # Fused MC2 and hierarchy communication are mutually exclusive.
        if self.enable_fused_mc2:
            raise ValueError(
                "fused mc2 op cannot be used with hierarchy communication. "
                "Please set additional_config.enable_fused_mc2 to 0."
            )

    def _validate_sparse_c8_kv_offload_compatibility(self) -> None:
        if self.sparse_kv_offload_config.enabled and self.enable_sparse_sfa_c8:
            raise NotImplementedError(
                "Sparse KV offload does not support the sparse SFA C8 main "
                "cache. Disable enable_sparse_sfa_c8; enable_sparse_li_c8 is "
                "supported because the indexer cache remains device-resident."
            )

    @classmethod
    def _check_mooncake_c8_kv_cache_quant(cls, vllm_config: VllmConfig) -> None:
        kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
        if kv_transfer_config is None:
            return

        quant_config = getattr(vllm_config, "quant_config", None)
        enable_c8_quant = getattr(quant_config, "enable_c8_quant", False)
        if enable_c8_quant is not True:
            return

        from vllm_ascend.utils import is_gqa_backend, uses_mooncake_connector

        if not is_gqa_backend(vllm_config):
            return

        if not uses_mooncake_connector(kv_transfer_config):
            return

        raise ValueError(
            "MooncakeConnector does not support C8 KV cache quantization on GQA models. "
            "The producer keeps KV cache in bf16 while the consumer allocates int8 KV cache, so raw "
            "Mooncake transfer would reinterpret bf16 bytes as int8. Please disable C8 KV cache quantization "
            "or use MooncakeLayerwiseConnector, which quantizes KV cache before transfer."
        )

    def _check_mix_placement(self):
        if self.mix_placement:
            if self.enable_shared_expert_dp or self.multistream_overlap_shared_expert:
                raise ValueError("Mix placement is not supported with shared expert DP or multistream overlap.")

    @staticmethod
    def _is_megamoe_supported_by_config(vllm_config: VllmConfig) -> bool:
        hf_text_config = vllm_config.model_config.hf_text_config
        hidden_size = getattr(hf_text_config, "hidden_size", None)
        if hidden_size is None and hasattr(vllm_config.model_config, "get_hidden_size"):
            hidden_size = vllm_config.model_config.get_hidden_size()
        if hidden_size is None:
            return False
        hidden_size = int(hidden_size)
        if hidden_size < 1024 or hidden_size > 8192 or hidden_size % 512 != 0:
            return False

        moe_intermediate_size = getattr(hf_text_config, "moe_intermediate_size", None)
        if moe_intermediate_size is None:
            moe_intermediate_size = getattr(hf_text_config, "intermediate_size", None)
        if moe_intermediate_size is None:
            return False
        if moe_intermediate_size < 1024 or moe_intermediate_size > 3072 or moe_intermediate_size % 512 != 0:
            return False

        quant_type = getattr(hf_text_config, "moe_quantize", getattr(hf_text_config, "quantize", None))
        if quant_type is None:
            return True
        quant_name = str(getattr(quant_type, "name", quant_type)).lower()
        supported_quant_names = {
            "w8a8",
            "w4a8",
            "w8a8_dynamic",
            "w4a8_dynamic",
            "quanttype.w8a8",
            "quanttype.w4a8",
        }
        return quant_name in supported_quant_names

    @staticmethod
    def _materialize_dump_config_to_file(dump_config: dict[str, Any]) -> str:
        dump_config_dir = os.path.join(os.getcwd(), ".vllm_ascend", "msprobe")
        os.makedirs(dump_config_dir, exist_ok=True)
        dump_config_file_path = os.path.join(dump_config_dir, "msprobe_dump_config.json")
        with open(dump_config_file_path, "w", encoding="utf-8") as file:
            json.dump(dump_config, file, ensure_ascii=False, indent=2)
        logger.info("Materialized additional_config.dump_config to file: %s", dump_config_file_path)
        return dump_config_file_path

    @classmethod
    def _resolve_dump_config_path(cls, additional_config: dict[str, Any]) -> str | None:
        dump_config_path = additional_config.get("dump_config_path")
        dump_config = additional_config.get("dump_config")
        if dump_config_path is not None and dump_config is not None:
            raise ValueError(
                "Only one of additional_config.dump_config_path or additional_config.dump_config can be set."
            )
        if dump_config is not None:
            if not isinstance(dump_config, dict):
                raise ValueError(f"additional_config.dump_config must be a dict, got {type(dump_config).__name__}.")
            return cls._materialize_dump_config_to_file(dump_config)
        if dump_config_path is not None and not isinstance(dump_config_path, str):
            raise ValueError(
                f"additional_config.dump_config_path must be a string, got {type(dump_config_path).__name__}."
            )
        return dump_config_path

    @staticmethod
    def _has_sparse_li_c8_layer_config(quant_config: Any) -> bool:
        quant_description = getattr(quant_config, "quant_description", None)
        if not isinstance(quant_description, dict):
            return False
        quant_suffixes = (".indexer.quant_type", ".indexer.wq_b_weight")
        return any(isinstance(key, str) and key.endswith(quant_suffixes) for key in quant_description)

    @classmethod
    def _parse_sparse_li_c8_layers_from_quant_config(cls, quant_config: Any) -> tuple[set[int], set[str]]:
        quant_description = getattr(quant_config, "quant_description", None)
        if not isinstance(quant_description, dict):
            return set(), set()

        QUANT_SUFFIXES = (".indexer.quant_type", ".indexer.wq_b_weight")
        VALID_QUANT_TYPES = ("INT8_DYNAMIC", "W8A8_MXFP8")

        layer_ids: set[int] = set()
        layer_names: set[str] = set()
        from vllm.model_executor.models.utils import extract_layer_index

        for key, value in quant_description.items():
            if not isinstance(key, str):
                continue
            matched_suffix = next((s for s in QUANT_SUFFIXES if key.endswith(s)), None)
            if matched_suffix is None or value not in VALID_QUANT_TYPES:
                continue
            layer_name = key[: -len(matched_suffix)].rstrip(".")
            if not layer_name:
                continue
            layer_names.add(layer_name)
            layer_ids.add(extract_layer_index(layer_name))
        return layer_ids, layer_names

    def is_sparse_li_c8_layer(self, layer_name: str | None) -> bool:
        if not self.enable_sparse_li_c8:
            return False
        if not self._sparse_li_c8_layer_filter_enabled:
            return True
        if layer_name is None:
            return False

        normalized_layer_name = layer_name.rstrip(".")
        if any(
            normalized_layer_name == candidate or normalized_layer_name.startswith(f"{candidate}.")
            for candidate in self._sparse_li_c8_layer_names
        ):
            return True
        from vllm.model_executor.models.utils import extract_layer_index

        layer_ids = {extract_layer_index(normalized_layer_name)}
        return any(layer_id in self._sparse_li_c8_layer_ids for layer_id in layer_ids)

    @staticmethod
    def _get_compile_ranges(compilation_config):
        return compilation_config.compile_ranges_endpoints or []

    @staticmethod
    def _set_compile_ranges(compilation_config, value):
        compilation_config.compile_ranges_endpoints = value

    def update_compile_ranges_split_points(self):
        return

    def get_mc2_comm_alg(self) -> str:
        from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

        # When A3 and comm_alg == "fullmesh", dispatch/combine op need pass in "fullmesh_v1" instead of "fullmesh"
        # TODO(zzzzwwjj): Remove it when op's param is uniformed between A2/A3/A5.
        if self.mc2_comm_alg == "fullmesh" and get_current_hardware_profile().supports(
            HardwareCapability.MC2_FULLMESH_V2_COMM
        ):
            return "fullmesh_v1"
        return self.mc2_comm_alg


@config
class DynamicSpecConfig:
    """
    Configuration Object for dynamic_spec_config from additional_config
    """

    # Dynamic speculative-length methods. "dspark" relies on the DSpark
    # confidence head; models without such a head need another method.
    SUPPORTED_METHODS: ClassVar[tuple[str, ...]] = ("dspark", "dflash")

    # None disables the dynamic speculative-length path.
    method: str | None = None
    # Custom parameters of the selected dynamic method; the expected keys
    # depend on `method` (e.g. dspark accepts
    # initial_verify_budget_per_req, budget_update_interval and
    # budget_threshold). Empty by default, in which case each method
    # falls back to its own built-in defaults.
    method_params: dict[str, Any] = dataclasses.field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self):
        if self.method is not None and self.method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"dynamic_spec_config.method must be one of {self.SUPPORTED_METHODS} or None, got {self.method!r}"
            )
        return self


@config
class FinegrainedTPConfig:
    """Configuration Object for ``additional_config["finegrained_tp_config"]``.

    Migrated to ``@config`` (pydantic dataclass). 5 int fields get lax coercion
    ('2'→2). vllm_config-dependent preconditions (TP/eager/kv_consumer/is_moe/
    data_parallel divisibility) are validated in ``_validate_preconditions()``,
    a plain method invoked explicitly by ``init_ascend_config`` (Plan B:
    business validation stays out of pydantic). ``vllm_config`` is no longer a
    member field.
    """

    oproj_tensor_parallel_size: int = 0
    lmhead_tensor_parallel_size: int = 0
    embedding_tensor_parallel_size: int = 0
    mlp_tensor_parallel_size: int = 0
    olora_tensor_parallel_size: int = 0

    @model_validator(mode="after")
    def _validate_sizes(self):
        size_fields = (
            "oproj_tensor_parallel_size",
            "lmhead_tensor_parallel_size",
            "embedding_tensor_parallel_size",
            "mlp_tensor_parallel_size",
            "olora_tensor_parallel_size",
        )
        for field_name in size_fields:
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"finegrained_tp_config.{field_name} must be non-negative, got {value}")
        return self

    def _validate_preconditions(self, vllm_config: Any):
        vc = vllm_config
        enabled_configs = []
        if self.oproj_tensor_parallel_size > 0:
            enabled_configs.append(f"oproj_tensor_parallel_size={self.oproj_tensor_parallel_size}")
            # wo_a/wo_b are sharded solely by the OTP group (which splits DP,
            # orthogonal to the standard TP group), but _forward_o_proj reshapes
            # the attention output with n_local_groups = n_groups // tp_size
            # (standard TP). When tp_size > 1 the weight-shard and input-shard
            # operate on different axes of the rank grid and no longer align,
            # so oproj TP currently requires standard tp_size == 1.
            if vc.parallel_config.tensor_parallel_size > 1:
                raise AssertionError(
                    "oproj_tensor_parallel_size currently requires "
                    "tensor_parallel_size == 1, got "
                    f"{vc.parallel_config.tensor_parallel_size}."
                )
            # The static all_to_all / reduce_scatter exchange buffers used by
            # _forward_o_proj are sized for graph replay and require ACL graph
            # capture; dummy_run does not run the entire attention module in
            # eager mode, so o_proj tp split can only be used in graph mode.
            if vc.model_config and vc.model_config.enforce_eager:
                raise AssertionError("oproj_tensor_parallel_size is only supported in graph mode")
            if vc.kv_transfer_config is None or not vc.kv_transfer_config.is_kv_consumer:
                raise AssertionError(
                    "oproj_tensor_parallel_size is only supported in pd scenario and can only be used in D node."
                )
        if self.olora_tensor_parallel_size > 0:
            enabled_configs.append(f"olora_tensor_parallel_size={self.olora_tensor_parallel_size}")
            # dummy_run does not run the entire attention module in eager mode,
            # so the o_lora tp split can only be used in graph mode.
            if vc.model_config and vc.model_config.enforce_eager:
                raise AssertionError("olora_tensor_parallel_size is only supported in graph mode")
            if vc.kv_transfer_config is None or not vc.kv_transfer_config.is_kv_consumer:
                raise AssertionError(
                    "olora_tensor_parallel_size is only supported in pd scenario and can only be used in D node."
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
            self.olora_tensor_parallel_size,
        ]
        for module_tp_size in module_tp_sizes:
            # If it is a dense model, then expert parallel is not needed,
            # and data parallel is also not needed. If the data parallel size is set
            # to greater than 1 in the model launch configuration, its value will be changed to 1 later.
            # This will cause an issue when finegrained tp is enabled, as it
            # cannot be split into the data parallel communication group, leading to an error.
            if module_tp_size > 0 and not vc.model_config.is_moe:
                raise AssertionError("The finegrained tp sizes can be enabled only for MOE models.")
            if module_tp_size > 0 and vc.parallel_config.data_parallel_size % module_tp_size != 0:
                raise AssertionError("finegrained tp sizes must divide by data_parallel_size.")
        if any(size > 0 for size in module_tp_sizes) and enabled_configs:
            logger.info("finegrained_tp_config enabled: %s", ", ".join(enabled_configs))


@config
class XliteGraphConfig:
    """Configuration Object for ``additional_config["xlite_graph_config"]``.

    Migrated to ``@config`` (pydantic dataclass). The vllm_config-dependent
    preconditions (speculative decoding / pipeline parallelism / cache block
    size) are validated in ``_validate_preconditions()``, a plain method
    invoked explicitly by ``init_ascend_config`` (Plan B: business validation
    stays out of pydantic). ``vllm_config`` is no longer a member field.
    """

    enabled: bool = False
    full_mode: bool = False

    def _validate_preconditions(self, vllm_config: Any):
        if self.enabled:
            vc = vllm_config
            if bool(vc.speculative_config) and vc.speculative_config.num_speculative_tokens != 1:
                raise RuntimeError("Xlite graph mode only support speculative decoding with num_speculative_tokens=1.")
            if vc.parallel_config.pipeline_parallel_size > 1:
                raise RuntimeError(
                    "Xlite graph mode is not compatible with pipeline parallelism. "
                    "Please set pipeline_parallel_size to 1."
                )
            if vc.cache_config.block_size != 128:
                logger.warning(
                    "Current cache block size may not be optimal for xlite graph mode. "
                    "current_block_size=%d, recommended_block_size=128.",
                    vc.cache_config.block_size,
                )


@config
class ProfilingChunkConfig:
    """Configuration for profiling-based dynamic chunk sizing.

    Migrated to ``@config`` (pydantic dataclass). Range/positivity checks
    moved to an ``after`` model_validator. ``need_timing`` uses ``None`` as a
    "not provided" sentinel so the after-validator can distinguish "user did
    not set it" (→ default to ``enabled``) from "user explicitly set False"
    (→ keep False), mirroring the original ``config.get("need_timing", self.enabled)``.
    """

    enabled: bool = False
    smooth_factor: float = 1.0
    min_chunk: int = 4096
    need_timing: bool | None = None
    max_fit_chunk: int = 30

    @model_validator(mode="after")
    def _validate_and_link_need_timing(self):
        # need_timing defaults to enabled when not explicitly set by the user.
        # (Original: config.get("need_timing", self.enabled).) Using None as
        # sentinel distinguishes "not provided" from "explicitly False".
        if self.need_timing is None:
            self.need_timing = self.enabled
        if not self.enabled and self.need_timing:
            logger.warning(
                "profiling_chunk_config.need_timing=True is ignored because "
                "profiling_chunk_config.enabled=False. Setting need_timing to False."
            )
            self.need_timing = False
        if not (0 < self.smooth_factor <= 1.0):
            raise ValueError(f"profiling_chunk_config.smooth_factor must be in (0, 1], got {self.smooth_factor}")
        if self.min_chunk <= 0:
            raise ValueError(f"profiling_chunk_config.min_chunk must be positive, got {self.min_chunk}")
        if self.max_fit_chunk <= 5:
            raise ValueError(f"Recommend to use at least 30 data points for fitting, got {self.max_fit_chunk}")
        return self


@config
class BatchJobSchedConfig:
    """Configuration for batch-job-aware scheduler.

    Migrated to ``@config`` (pydantic dataclass). Range checks moved to an
    ``after`` model_validator.
    """

    enabled: bool = False
    max_jobs: int = 20
    reserve_margin_blocks: int = 2
    reserve_max_blocks: int = 8
    low_available_tokens_threshold: int = 4096
    short_decode_token_threshold: int = 32

    @model_validator(mode="after")
    def _validate(self):
        if self.max_jobs < 0:
            raise ValueError(f"batch_job_sched_config.max_jobs must be non-negative, got {self.max_jobs}")
        if self.reserve_margin_blocks < 0:
            raise ValueError(
                f"batch_job_sched_config.reserve_margin_blocks must be non-negative, got {self.reserve_margin_blocks}"
            )
        if self.reserve_max_blocks <= 0:
            raise ValueError(
                f"batch_job_sched_config.reserve_max_blocks must be positive, got {self.reserve_max_blocks}"
            )
        if self.low_available_tokens_threshold <= 0:
            raise ValueError(
                f"batch_job_sched_config.low_available_tokens_threshold must be positive, "
                f"got {self.low_available_tokens_threshold}"
            )
        if self.short_decode_token_threshold <= 0:
            raise ValueError(
                f"batch_job_sched_config.short_decode_token_threshold must be positive, "
                f"got {self.short_decode_token_threshold}"
            )
        return self


@config
class ShortRequestFirstConfig:
    """Configuration object for ``additional_config["scheduler_config"]["short_request_first_config"]``.

    Migrated to ``@config`` (pydantic dataclass). Unknown-key detection is now
    handled by ``extra="forbid"`` (replaces the hand-written ``unknown`` set
    check); range checks moved to an ``after`` model_validator.
    """

    enabled: bool = False
    threshold: int = 256
    long_max_wait_ms: float = 0.0

    @model_validator(mode="after")
    def _validate_config(self):
        if self.threshold < 0:
            raise ValueError(f"short_request_first_config.threshold must be a non-negative int; got {self.threshold}")
        if self.long_max_wait_ms < 0:
            raise ValueError(f"short_request_first_config.long_max_wait_ms must be >= 0; got {self.long_max_wait_ms}")
        return self


@config
class DyntraLBConfig:
    """Configuration object for ``scheduler_config.dyntra_lb_config``."""

    enabled: bool = False
    enable_diagnostics: bool = False
    mode: str = "dynamic"
    start_step: int = 250
    end_step: int = -1
    bubble_threshold: float = 5.0
    long_req_block_threshold: int = 700
    dynamic_max_step: int = 256

    _valid_modes: ClassVar[set[str]] = {"static", "dynamic"}

    @model_validator(mode="after")
    def _validate_config(self) -> DyntraLBConfig:
        if self.mode not in self._valid_modes:
            raise ValueError(f"dyntra_lb_config.mode must be one of {sorted(self._valid_modes)}, got {self.mode!r}.")
        if self.start_step < 0:
            raise ValueError(f"dyntra_lb_config.start_step must be >= 0, got {self.start_step}.")
        if self.end_step < -1:
            raise ValueError(f"dyntra_lb_config.end_step must be -1 or >= 0, got {self.end_step}.")
        if self.end_step != -1 and self.end_step <= self.start_step:
            raise ValueError(
                "dyntra_lb_config.end_step must be greater than start_step when it is set, "
                f"got start_step={self.start_step}, end_step={self.end_step}."
            )
        if self.bubble_threshold <= 0:
            raise ValueError(f"dyntra_lb_config.bubble_threshold must be > 0, got {self.bubble_threshold}.")
        if self.long_req_block_threshold <= 0:
            raise ValueError(
                f"dyntra_lb_config.long_req_block_threshold must be > 0, got {self.long_req_block_threshold}."
            )
        if self.dynamic_max_step <= 0:
            raise ValueError(f"dyntra_lb_config.dynamic_max_step must be > 0, got {self.dynamic_max_step}.")
        return self


@config
class SchedulerConfig:
    """Configuration object for ``additional_config["scheduler_config"]``.

    Migrated to ``@config`` (pydantic dataclass). ``from_additional_config``
    resolves the precedence (nested scheduler_config > top-level legacy >
    default), preserving the original deprecation warnings, and then constructs
    this class from final configuration values. Sub-configs
    (ShortRequestFirstConfig / ProfilingChunkConfig / BatchJobSchedConfig) are
    typed fields that pydantic coerces from nested dicts.
    """

    enable_balance_scheduling: bool = False
    recompute_scheduler_enable: bool = False
    short_request_first_config: ShortRequestFirstConfig = dataclasses.field(default_factory=ShortRequestFirstConfig)
    profiling_chunk_config: ProfilingChunkConfig = dataclasses.field(default_factory=ProfilingChunkConfig)
    batch_job_sched_config: BatchJobSchedConfig = dataclasses.field(default_factory=BatchJobSchedConfig)
    dyntra_lb_config: DyntraLBConfig = dataclasses.field(default_factory=DyntraLBConfig)

    @classmethod
    def from_additional_config(cls, additional_config: dict[str, Any]) -> SchedulerConfig:
        """Resolve legacy fallbacks and construct the final config."""
        scheduler_config = additional_config.get("scheduler_config")
        if scheduler_config is None:
            scheduler_config = {}
        elif not isinstance(scheduler_config, dict):
            raise ValueError(
                f"additional_config.scheduler_config must be a dict, got {type(scheduler_config).__name__}."
            )

        def _resolve(config_key: str, default: Any) -> Any:
            if config_key in scheduler_config:
                if config_key in additional_config:
                    logger.warning_once(
                        "additional_config.%s is deprecated and ignored because "
                        "additional_config.scheduler_config.%s is set.",
                        config_key,
                        config_key,
                    )
                return scheduler_config[config_key]
            if config_key in additional_config:
                logger.warning_once(
                    "additional_config.%s is deprecated; use additional_config.scheduler_config.%s instead.",
                    config_key,
                    config_key,
                )
                return additional_config[config_key]
            return default

        resolved = {
            # Balance scheduling is configured only through additional_config;
            # the legacy environment fallback has been removed.
            "enable_balance_scheduling": _resolve("enable_balance_scheduling", False),
            "recompute_scheduler_enable": _resolve("recompute_scheduler_enable", False),
            # Let pydantic coerce the resolved dicts into typed sub-configs.
            "short_request_first_config": _resolve("short_request_first_config", {}),
            "profiling_chunk_config": _resolve("profiling_chunk_config", {}),
            "batch_job_sched_config": _resolve("batch_job_sched_config", {}),
            "dyntra_lb_config": scheduler_config.get("dyntra_lb_config", {}),
        }
        # Forward nested unknown keys to pydantic so extra="forbid" reports
        # typos instead of the resolver silently dropping them.
        resolved.update({key: value for key, value in scheduler_config.items() if key not in resolved})
        return cls(**resolved)  # type: ignore[arg-type]


@config
class SparseKVOffloadConfig:
    """
    Configuration for the Sparse KV cache offloading.
    """

    enabled: bool = False
    topk_buffer_size: int = 4096
    dram_size_per_dp_GB: int = 128
    keep_device_kv_cache: bool = False
    topk: int = dataclasses.field(default=0, init=False)

    @model_validator(mode="after")
    def _validate_values(self):
        if self.topk_buffer_size <= 0:
            raise ValueError("sparse_kv_offload_config.topk_buffer_size must be positive")
        if self.dram_size_per_dp_GB <= 0:
            raise ValueError("sparse_kv_offload_config.dram_size_per_dp_GB must be positive")
        return self

    @classmethod
    def from_additional_config(cls, vllm_config: VllmConfig, user_config: Any) -> SparseKVOffloadConfig:
        if not isinstance(user_config, dict):
            raise ValueError(
                f"additional_config.sparse_kv_offload_config must be a dict, got {type(user_config).__name__}."
            )
        config = cls(**user_config)  # type: ignore[call-arg]
        config._validate_preconditions(vllm_config)
        return config

    def _validate_preconditions(self, vllm_config: VllmConfig) -> None:
        if not self.enabled:
            return

        if hasattr(vllm_config.model_config.hf_text_config, "compress_ratios"):
            raise ValueError("Sparse KV offload don't support compress now.")
        if not hasattr(vllm_config.model_config.hf_text_config, "index_topk"):
            raise ValueError("Sparse KV offload only support sparse attention model.")
        parallel_config = vllm_config.parallel_config
        if parallel_config.prefill_context_parallel_size * parallel_config.decode_context_parallel_size > 1:
            raise ValueError("Sparse KV offload don't support context parallel now.")
        if parallel_config.pipeline_parallel_size > 1:
            raise ValueError("Sparse KV offload don't support pipeline parallel now.")
        if self.keep_device_kv_cache:
            logger.warning_once(
                "Init sparse KV offload with keep_device_kv_cache enabled, "
                "in this case we will still allocate device kv cache "
                "and can not improve sequence length or batch_size. "
                "You should only use it for debugging in PD colocate scenario."
            )
        else:
            if vllm_config.kv_transfer_config is None or not vllm_config.kv_transfer_config.is_kv_consumer:
                raise AssertionError(
                    "Sparse KV offload is only supported in PD disaggregate scenario "
                    "and can only be used in D node. For debugging in PD colocate scenario, "
                    "you can enable keep_device_kv_cache."
                )
        if vllm_config.use_v2_model_runner:
            if vllm_config.speculative_config is not None:
                method = getattr(vllm_config.speculative_config, "method", None)
                if method not in ("mtp", "deepseek_mtp"):
                    raise ValueError("Sparse KV offload with model_runner_v2 only supports MTP speculative decoding.")

        self.topk = vllm_config.model_config.hf_text_config.index_topk
        if self.topk_buffer_size < self.topk:
            raise ValueError(
                "sparse_kv_offload_config.topk_buffer_size must be >= topk, "
                f"got topk_buffer_size={self.topk_buffer_size}, topk={self.topk}"
            )


_ASCEND_CONFIG: AscendConfig | None = None
# Identity key for the singleton cache: the vllm_config that initialized it.
# Private module state (not a field on AscendConfig) — replaces the former
# ``getattr(_ASCEND_CONFIG, "vllm_config", None) is vllm_config`` check, now
# that vllm_config is no longer a member of AscendConfig.
_INIT_VLLM_CONFIG: Any = None


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
    if "enable_flashcomm1" in additional_config or os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM1") is not None:
        logger.warning(
            "FlashComm is deprecated; remove enable_flashcomm1 and "
            "VLLM_ASCEND_ENABLE_FLASHCOMM1 from the configuration. Use upstream configuration instead"
        )
    refresh = validate_additional_config_bool(additional_config.get("refresh", False), "additional_config.refresh")
    raw_rl_config = additional_config.get("rl_config", {})
    if isinstance(raw_rl_config, dict):
        refresh = refresh or validate_additional_config_bool(
            raw_rl_config.get("enabled", False), "additional_config.rl_config.enabled"
        )
    elif "rl_config" in additional_config:
        # Do not reuse a cached config: let AscendConfig's normal nested
        # pydantic validation report the invalid sub-config input below.
        refresh = True
    global _ASCEND_CONFIG, _INIT_VLLM_CONFIG
    if (
        _ASCEND_CONFIG is not None
        and not refresh
        and _is_ascend_config_initialized(_ASCEND_CONFIG)
        and _INIT_VLLM_CONFIG is vllm_config
    ):
        return _ASCEND_CONFIG

    # Pre-construct sub-configs that need precedence resolution or vllm_config.
    sched = SchedulerConfig.from_additional_config(additional_config)
    sparse_kv = SparseKVOffloadConfig.from_additional_config(
        vllm_config, additional_config.get("sparse_kv_offload_config", {})
    )
    # dump_config: keep the mutual-exclusion / materialize logic as a factory
    # pre-step; the resolved path is passed as the dump_config_path field.
    dump_config_path = AscendConfig._resolve_dump_config_path(additional_config)

    # Keys that must NOT flow from additional_config into AscendConfig.
    # These are stripped so that only user-configurable keys reach pydantic,
    # where extra="forbid" can reject unknown options.
    _NON_USER_INPUT_KEYS = {
        # control-flow flag (singleton/cache refresh), not a configuration field
        "refresh",
        # Removed upstream option: warn above, but do not pass it into the
        # strict AscendConfig schema where it would be reported as a typo.
        "enable_flashcomm1",
        # injected fields (factory passes explicitly; a copy in additional_config would conflict)
        "scheduler_config",
        "sparse_kv_offload_config",
        # Factory-only input: materialized by _resolve_dump_config_path and
        # replaced with the validated dump_config_path field below.
        "dump_config",
        "dump_config_path",
        # pure-derived fields (derive_and_validate computes them; user input would residualize)
        # NOTE: enable_shared_expert_dp/enable_sparse_sfa_c8/enable_sparse_li_c8/
        # c8_enable_reshape_optim are NOT here — they are user-input fields that
        # derive_and_validate *augments* (self.x = self.x and condition), so the user
        # must be able to pass them. Only pure-derived fields (no user input) are stripped.
        "enable_sp_by_pass",
        "pd_tp_ratio",
        "pd_head_ratio",
        "num_head_replica",
        # private derived state (init=False, but listed for safety)
        "_sparse_li_c8_layer_ids",
        "_sparse_li_c8_layer_names",
        "_sparse_li_c8_layer_filter_enabled",
        # SchedulerConfig-internal top-level legacy keys (resolved internally,
        # then replaced by the typed scheduler_config passed above).
        "enable_balance_scheduling",
        "recompute_scheduler_enable",
        "short_request_first_config",
        "profiling_chunk_config",
        "batch_job_sched_config",
    }
    kwargs = {k: v for k, v in additional_config.items() if k not in _NON_USER_INPUT_KEYS}
    unknown_keys = sorted(set(kwargs) - AscendConfig.__dataclass_fields__.keys())
    # vLLM-Omni shares this mapping with the platform plugin. Preserve its
    # extension keys on VllmConfig while excluding them from Ascend validation.
    if unknown_keys and importlib.util.find_spec("vllm_omni") is not None:
        logger.warning(
            "The following additional_config keys are invalid for vLLM-Ascend: %s. "
            "They may be used by vLLM-Omni or another project. "
            "Please remove them if they are not needed for your use case.",
            unknown_keys,
        )
        kwargs = {k: v for k, v in kwargs.items() if k not in unknown_keys}

    new_config = AscendConfig(  # type: ignore[call-arg]
        scheduler_config=sched,
        sparse_kv_offload_config=sparse_kv,
        dump_config_path=dump_config_path,
        **kwargs,
    )
    # Business validation (Plan B): pydantic did type/range/enum checks during
    # construction; the cross-config derivations and mutex checks that need
    # vllm_config run here, explicitly, before the instance is usable. This is
    # the single legitimate entry point — bypassing the factory leaves derived
    # fields at their sentinel defaults.
    new_config.derive_and_validate(vllm_config)
    new_config.rl_config.apply(new_config)
    new_config.finegrained_tp_config._validate_preconditions(vllm_config)
    new_config.xlite_graph_config._validate_preconditions(vllm_config)
    if _is_ascend_config_initialized(new_config):
        _ASCEND_CONFIG = new_config
        _INIT_VLLM_CONFIG = vllm_config
        # Publish the fully validated singleton before invalidating derived
        # process caches. The next runtime read rebuilds them from new_config;
        # failed construction leaves the previous singleton/cache untouched.
        from vllm_ascend.utils import clear_enable_sp

        clear_enable_sp()
    else:
        logger.warning("Ascend config instance is not fully initialized. action: skip singleton cache update. ")
    return new_config


def clear_ascend_config():
    global _ASCEND_CONFIG, _INIT_VLLM_CONFIG
    _ASCEND_CONFIG = None
    _INIT_VLLM_CONFIG = None
    from vllm_ascend.utils import clear_enable_sp

    clear_enable_sp()


def get_ascend_config():
    global _ASCEND_CONFIG
    if _ASCEND_CONFIG is None or not _is_ascend_config_initialized(_ASCEND_CONFIG):
        raise RuntimeError("Ascend config is not initialized. Please call init_ascend_config first.")
    return _ASCEND_CONFIG
