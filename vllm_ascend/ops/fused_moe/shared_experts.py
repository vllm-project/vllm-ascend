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
#
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from functools import wraps
from typing import Any, TypeAlias

import torch
import torch.nn.functional as F
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed import tensor_model_parallel_all_gather, tensor_model_parallel_reduce_scatter
from vllm.logger import logger
from vllm.model_executor.layers.activation import (
    SiluAndMul,
    SiluAndMulWithClamp,
    SituAndMul,
    SwigluOAIAndMul,
)
from vllm.model_executor.layers.fused_moe import FusedMoEConfig, FusedMoEMethodBase

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import (
    _EXTRA_CTX,
    is_draft_vllm_config,
    use_cann_megamoe,
)
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.lora.fused_moe import has_lora
from vllm_ascend.ops.activation import AscendSiluAndMul
from vllm_ascend.ops.fused_moe import moe_utils
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MegamoeSharedExpertFusion, MegamoeSharedWeights
from vllm_ascend.ops.fused_moe.dataclass.shared_experts import (
    PreparedSharedExpertInput,
    RoutedMoEMilestones,
)
from vllm_ascend.ops.fused_moe.moe_utils import _pad_tokens_with_cat
from vllm_ascend.quantization.prepared_linear_input import PreparedLinearInput
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import (
    ACL_FORMAT_FRACTAL_ND,
    ACL_FORMAT_FRACTAL_NZ,
    npu_stream_switch,
    shared_experts_calculation_stream,
)

# CANN uses 36 to select FP8 E4M3FN output for situ_mx_quant.
SITU_MX_DST_TYPE_E4M3FN = 36
MXFP8_ACTIVATION_GROUP_SIZE = 32
# SwiGLU halves the input width; MX scales are stored in pairs of groups.
SWIGLU_MX_INPUT_ALIGNMENT = 4 * MXFP8_ACTIVATION_GROUP_SIZE
SWIGLU_MX_QUANT_MODE = 2

# CANN Arch22 tiling dimension contracts (mega_moe_tiling_arch22.cpp).
_A3_MIN_HIDDEN_DIM = 1024
_A3_MAX_HIDDEN_DIM = 8192
_A3_HIDDEN_DIM_ALIGN = 512
_A3_MIN_SHARED_INTER_DIM = 512
_A3_SHARED_INTER_DIM_ALIGN = 256


def _is_npu_runtime_available() -> bool:
    """Return True when an NPU runtime is available for device-residency checks."""
    is_avail = getattr(getattr(torch, "npu", None), "is_available", None)
    try:
        return bool(is_avail()) if callable(is_avail) else False
    except Exception:
        return False


def _has_generic_cpu_weight_offload(vllm_config: Any) -> bool:
    """Return True when generic CPU weight offload (UVA or prefetch) is active."""
    offload = getattr(vllm_config, "offload_config", None)
    if offload is None:
        return False
    return any(
        isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
        for v in (
            getattr(getattr(offload, "uva", None), "cpu_offload_gb", 0),
            getattr(getattr(offload, "prefetch", None), "offload_group_size", 0),
        )
    )


@dataclass(frozen=True, slots=True)
class AscendMegaMoESharedSpec:
    """Physical storage contract of a shared-expert linear quantization scheme."""

    carrier_dtype: torch.dtype
    post_load_transposed: bool
    k_pack_factor: int = 1
    scale_kind: str = "none"
    requires_symmetric_zero_offset: bool = False


@dataclass(frozen=True, slots=True)
class AscendMegaMoEFormat:
    """MegaMoe layout contract for a ``(routed_quant_type, shared_quant_type)`` pair."""

    requires_mxfp_capability: bool
    kernel_transposed: bool
    w1_acl_format: int
    w2_acl_format: int
    w2_mxfp4_c0_32_cast: bool = False
    allow_independent_shared_intermediate: bool = False


@dataclass(frozen=True, slots=True)
class _StagedSharedLinear:
    """Per-slot staging record so w1/w2 validation & TP collectives run in deterministic order."""

    ok: bool
    reason: str = ""
    linear: torch.nn.Module | None = None
    shared_quant_type: QuantType | None = None
    spec: AscendMegaMoESharedSpec | None = None
    fmt: AscendMegaMoEFormat | None = None
    shard_tp_size: int = 1
    local_in_packed: int = 0
    local_out_dim: int = 0
    in_packed: int = 0
    out_dim: int = 0
    local_scale_fp32: torch.Tensor | None = None


_ASCEND_MEGAMOE_SHARED_SPECS: dict[QuantType, AscendMegaMoESharedSpec] = {
    QuantType.NONE: AscendMegaMoESharedSpec(torch.bfloat16, post_load_transposed=False),
    QuantType.W8A8: AscendMegaMoESharedSpec(
        torch.int8,
        post_load_transposed=True,
        scale_kind="int64_low32",
        requires_symmetric_zero_offset=True,
    ),
    QuantType.W8A8MXFP: AscendMegaMoESharedSpec(
        torch.float8_e4m3fn,
        post_load_transposed=True,
        scale_kind="mxfp_e8m0",
    ),
    QuantType.W4A8MXFP: AscendMegaMoESharedSpec(
        torch.uint8,
        post_load_transposed=True,
        k_pack_factor=2,
        scale_kind="mxfp_e8m0",
    ),
    QuantType.W4A4MXFP: AscendMegaMoESharedSpec(
        torch.uint8,
        post_load_transposed=True,
        k_pack_factor=2,
        scale_kind="mxfp_e8m0",
    ),
}

_ASCEND_MEGAMOE_FORMATS: dict[tuple[QuantType, QuantType], AscendMegaMoEFormat] = {
    # Ascend A3 (arch22): (in, out) storage; (W4A8|W8A8, W8A8) in FRACTAL_NZ and (NONE, NONE) BF16 in ND.
    **{
        (routed_qt, QuantType.W8A8): AscendMegaMoEFormat(
            requires_mxfp_capability=False,
            kernel_transposed=True,
            w1_acl_format=ACL_FORMAT_FRACTAL_NZ,
            w2_acl_format=ACL_FORMAT_FRACTAL_NZ,
            allow_independent_shared_intermediate=True,
        )
        for routed_qt in (QuantType.W4A8, QuantType.W8A8)
    },
    (QuantType.NONE, QuantType.NONE): AscendMegaMoEFormat(
        requires_mxfp_capability=False,
        kernel_transposed=True,
        w1_acl_format=ACL_FORMAT_FRACTAL_ND,
        w2_acl_format=ACL_FORMAT_FRACTAL_ND,
        allow_independent_shared_intermediate=True,
    ),
    # Ascend A5 (arch35): (out, in) storage; MXFP8 in ND, packed MXFP4 in FRACTAL_NZ / FRACTAL_NZ_C0_32.
    **{
        (routed_qt, shared_qt): AscendMegaMoEFormat(
            requires_mxfp_capability=True,
            kernel_transposed=False,
            w1_acl_format=(
                ACL_FORMAT_FRACTAL_NZ
                if shared_qt in (QuantType.W4A4MXFP, QuantType.W4A8MXFP)
                else ACL_FORMAT_FRACTAL_ND
            ),
            w2_acl_format=(
                ACL_FORMAT_FRACTAL_NZ
                if shared_qt in (QuantType.W4A4MXFP, QuantType.W4A8MXFP)
                else ACL_FORMAT_FRACTAL_ND
            ),
            w2_mxfp4_c0_32_cast=shared_qt in (QuantType.W4A4MXFP, QuantType.W4A8MXFP),
        )
        for routed_qt in (QuantType.W8A8MXFP, QuantType.W4A8MXFP, QuantType.W4A4MXFP)
        for shared_qt in (QuantType.W8A8MXFP, QuantType.W4A8MXFP, QuantType.W4A4MXFP)
    },
}

_CARRIER_DTYPE_QUANT_MAP: dict[tuple[torch.dtype, bool], QuantType] = {
    (torch.bfloat16, False): QuantType.NONE,
    (torch.int8, True): QuantType.W8A8,
    (torch.uint8, True): QuantType.W4A4MXFP,
    (torch.float8_e4m3fn, True): QuantType.W8A8MXFP,
}


def _get_allow_internal_format() -> bool:
    """Read the active ``ALLOW_INTERNAL_FORMAT`` option across ``torch_npu`` builds."""
    get_option = getattr(getattr(torch_npu, "_C", None), "_npu_getOption", None)
    if callable(get_option):
        try:
            raw = get_option("ALLOW_INTERNAL_FORMAT")
        except Exception:
            raw = None
        if isinstance(raw, (bytes, str)):
            decoded = (raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else raw).strip().lower()
            if decoded:
                return decoded in ("enable", "1", "true")
        elif isinstance(raw, (bool, int)):
            return bool(raw)
    npu_cfg = getattr(getattr(torch, "npu", None), "config", None)
    try:
        flag = getattr(npu_cfg, "allow_internal_format", None) if npu_cfg is not None else None
    except Exception:
        flag = None
    return bool(flag) if isinstance(flag, (bool, int)) else True


def _set_allow_internal_format(enabled: bool) -> None:
    """Set ``ALLOW_INTERNAL_FORMAT`` without gating on ``hasattr`` (``_npuConfig`` is write-only)."""
    from types import ModuleType

    flag = bool(enabled)
    npu_cfg = getattr(getattr(torch, "npu", None), "config", None)
    if npu_cfg is not None and not isinstance(npu_cfg, ModuleType):
        try:
            npu_cfg.allow_internal_format = flag
            return
        except (AttributeError, TypeError):
            pass
    set_option = getattr(getattr(torch_npu, "_C", None), "_npu_setOption", None)
    if callable(set_option):
        set_option({"ALLOW_INTERNAL_FORMAT": "enable" if flag else "disable"})


@contextmanager
def _preserve_allow_internal_format():
    prev_flag = _get_allow_internal_format()
    try:
        yield
    finally:
        _set_allow_internal_format(prev_flag)


def _get_tensor_acl_format(tensor: torch.Tensor) -> int | None:
    if getattr(tensor, "device", None) is None or tensor.device.type != "npu":
        return None
    get_fmt = getattr(torch_npu, "get_npu_format", None)
    try:
        return int(get_fmt(tensor)) if callable(get_fmt) else None
    except Exception:
        return None


def _resolve_shared_linear_quant_type(linear: torch.nn.Module) -> QuantType | None:
    """Resolve the dynamic quantization scheme of a shared-expert linear layer."""
    if hasattr(linear, "input_scale") or hasattr(linear, "aclnn_input_scale"):
        return None
    linear_method = getattr(linear, "quant_method", None)
    for candidate in (getattr(linear_method, "quant_method", linear_method), linear_method):
        declared = getattr(candidate, "quant_type", None)
        if isinstance(declared, QuantType) and declared != QuantType.NONE:
            return declared
    weight = getattr(linear, "weight", None)
    weight_data = getattr(weight, "data", weight)
    if not isinstance(weight_data, torch.Tensor):
        return None
    has_scale = hasattr(linear, "weight_scale") or hasattr(linear, "weight_scale_fp32")
    return _CARRIER_DTYPE_QUANT_MAP.get((weight_data.dtype, has_scale))


class SharedExpertParallelMode(Enum):
    """Effective activation and weight layout for a shared-expert forward."""

    TENSOR_PARALLEL = auto()  # Full activations, TP-sharded weights.
    SHARED_EXPERT_DATA_PARALLEL_ONLY = auto()  # Full activations, replicated weights (DP only).
    SEQUENCE_PARALLEL_ONLY = auto()  # Sharded activations, TP-sharded weights (SP only).
    SEQUENCE_PARALLEL_SEDP = auto()  # Sharded activations, replicated weights (SP + DP).


class SharedExpertMLPPath(Enum):
    """Shared MLP implementation selected for a quantization scenario."""

    A8_INT_FUSED = auto()  # W8A8/W4A8: explicit A8 quant + fused activation quant.
    A8_MXFP_FUSED = auto()  # W4A8MXFP: explicit MXFP8 activation pipeline.
    W8A8_MXFP_LINEAR = auto()  # Preserve linears; fuse activation quant when supported.
    LINEAR_WRAPPER = auto()  # Dense, other quant schemes, or any active LoRA.


LinearInput: TypeAlias = torch.Tensor | PreparedLinearInput | tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class PreparedSharedExpertExecution:
    """Shared MLP input preparation submitted before the router matmul.

    ``gate_up_input`` may contain scheme-owned activation quantization that is
    already queued on the shared-expert stream. Gate-Up consumes it immediately
    after ``router_output_ready``; activation and Down are appended after the
    routed path exposes its stage milestones.
    """

    hidden_states: torch.Tensor
    gate_up_input: LinearInput
    local_dp_metadata: tuple[int, int] | None


@dataclass(frozen=True)
class SubmittedSharedExpertGateUp:
    """Gate-Up work already submitted on the shared-expert stream."""

    hidden_states: torch.Tensor
    gate_up: torch.Tensor
    gate_up_input_scale: torch.Tensor | None
    local_dp_metadata: tuple[int, int] | None
    path: SharedExpertMLPPath


class AscendSharedExperts:
    """Ascend-owned shared expert executor.

    Keep the original shared expert module registered on ``AscendMoERunner``
    for checkpoint compatibility while moving split/overlap execution details
    out of the runner.
    """

    def __init__(
        self,
        layer: torch.nn.Module,
        moe_config: FusedMoEConfig,
        quant_type: QuantType,
        quant_method: FusedMoEMethodBase,
    ):
        """`quant_type` is the ROUTED experts' quant type: it gates whether
        the routed MegaMoe path supports stacking the shared-expert fusion.
        The shared expert's own quant type is read from its linear scheme at
        weight-collection time and may differ (mixed quantization)."""
        self.layer = layer
        self.moe_config = moe_config
        self.hidden_size = moe_config.hidden_dim
        self.shared_expert_input_size = getattr(
            layer.gate_up_proj,
            "input_size",
            self.hidden_size,
        )
        self.in_dtype = moe_config.in_dtype
        self.swiglu_limit = 0.0 if moe_config.swiglu_limit is None else moe_config.swiglu_limit
        self.swiglu_alpha = 1.0 if moe_config.swiglu_alpha is None else moe_config.swiglu_alpha
        self.swiglu_beta = 0.0 if moe_config.swiglu_beta is None else moe_config.swiglu_beta
        self.is_sequence_parallel = moe_config.is_sequence_parallel
        self.situ_activation = layer.act_fn if isinstance(layer.act_fn, SituAndMul) else None
        self.quant_type = quant_type
        self.lora_context = None
        ascend_config = get_ascend_config()
        self.multistream_overlap = ascend_config.multistream_overlap_shared_expert
        self.weights_replicated = ascend_config.enable_shared_expert_dp

        # Shared-expert fusion into the CANN MegaMoe operator (A3 and A5):
        # disabled until the setup below proves the configuration is supported,
        # then armed once both shared linears have been processed into the
        # MegaMoe layout.
        self.megamoe_shared_weights_ready = False
        self._megamoe_shared_fusion_disabled = False
        self._megamoe_tp_gather_shards = False
        self._megamoe_local_slot_staging: dict[str, _StagedSharedLinear] = {}
        self._megamoe_shared_parts: dict[str, torch.Tensor] = {}
        self._megamoe_shared_quant_type = QuantType.NONE
        # Handed out by megamoe_shared_fusion() and filled in place once the
        # shared linears are rebuilt; see MegamoeSharedExpertFusion.
        self._megamoe_shared_fusion = MegamoeSharedExpertFusion()

        if self.multistream_overlap:
            # Wrap the quant_method's process_weights_after_loading to validate that
            # splitting shared expert computation (gate_up projection, activation,
            # then down projection) yields identical results to integrated
            # computation after weight loading.
            original_process_weights = quant_method.process_weights_after_loading

            @wraps(original_process_weights)
            def wrapped_process_weights(*args, **kwargs):
                result = original_process_weights(*args, **kwargs)
                self.validate_consistency()
                return result

            quant_method.process_weights_after_loading = wrapped_process_weights  # type: ignore

        self._setup_megamoe_shared_fusion()

    def _setup_megamoe_shared_fusion(self) -> None:
        """Fuse the shared expert into the CANN MegaMoe operator when supported.

        When active, each shared linear's ``process_weights_after_loading`` is
        wrapped to rebuild or zero-copy alias the weight into the declarative
        ``AscendMegaMoEFormat`` layout; the payload is exposed through
        ``megamoe_shared_fusion()`` and installed on the routed-expert layer by
        ``AscendMoERunner``, then forwarded to the operator on FUSED_MC2 steps.
        Any unsupported configuration keeps standalone shared-expert execution.
        """
        vllm_config = get_current_vllm_config()
        if is_draft_vllm_config(vllm_config) or not use_cann_megamoe(vllm_config):
            return
        # Only enable_fused_mc2=3 opts into fusing the shared expert into the
        # MegaMoe operator; =2 keeps the separate shared-expert forward.
        if not get_ascend_config().megamoe_shared_expert_fusion:
            return
        if _has_generic_cpu_weight_offload(vllm_config):
            self._disable_megamoe_shared_fusion(
                "CPU weight offload (--cpu-offload-gb / prefetch offload) is incompatible "
                "with MegaMoe shared-expert fusion"
            )
            return
        if self.in_dtype is not None and self.in_dtype != torch.bfloat16:
            self._disable_megamoe_shared_fusion(
                f"MegaMoe shared-expert fusion requires bfloat16 activations, got {self.in_dtype}"
            )
            return
        hardware_profile = get_current_hardware_profile()
        if not hardware_profile.supports(HardwareCapability.CANN_MEGAMOE_SHARED_EXPERT_FUSION):
            return
        supports_mxfp = hardware_profile.supports(HardwareCapability.CANN_MEGAMOE_MXFP)
        routed_supported = any(
            routed_qt == self.quant_type and fmt.requires_mxfp_capability == supports_mxfp
            for (routed_qt, _), fmt in _ASCEND_MEGAMOE_FORMATS.items()
        )
        if not routed_supported:
            logger.info_once(
                "MegaMoe shared-expert fusion is not implemented for routed quant type %s on this hardware; "
                "keeping the separate shared-expert forward.",
                self.quant_type,
            )
            self._megamoe_shared_fusion_disabled = True
            return
        if not supports_mxfp:
            hidden_dim = self.moe_config.hidden_dim
            if (
                hidden_dim < _A3_MIN_HIDDEN_DIM
                or hidden_dim > _A3_MAX_HIDDEN_DIM
                or hidden_dim % _A3_HIDDEN_DIM_ALIGN != 0
            ):
                self._disable_megamoe_shared_fusion(
                    f"A3 MegaMoe shared-expert fusion requires hidden_dim in "
                    f"[{_A3_MIN_HIDDEN_DIM}, {_A3_MAX_HIDDEN_DIM}] aligned to "
                    f"{_A3_HIDDEN_DIM_ALIGN}, got {hidden_dim}"
                )
                return
        # Output gating (e.g. Qwen3-Next expert_gate) happens outside the
        # operator and cannot be fused.
        if getattr(self.layer, "expert_gate", None) is not None:
            logger.info_once(
                "MegaMoe shared-expert fusion is not supported with expert gating; "
                "keeping the separate shared-expert forward."
            )
            self._megamoe_shared_fusion_disabled = True
            return
        # The operator consumes no shared bias.
        if (
            getattr(self.layer.gate_up_proj, "bias", None) is not None
            or getattr(self.layer.down_proj, "bias", None) is not None
        ):
            logger.info_once(
                "MegaMoe shared-expert fusion is not supported with shared-expert bias; "
                "keeping the separate shared-expert forward."
            )
            self._megamoe_shared_fusion_disabled = True
            return
        # The operator exposes a single global activation/activation_params
        # pair that it applies to the shared experts too, so fusion is only
        # valid when the shared activation is the *same function* as the
        # routed one: equal activation type AND equal activation params
        # (e.g. SituAndMul beta/linear_beta, SwiGLU-OAI alpha/beta/limit).
        # Parameterized activations of the same class but different params
        # (or an unbounded shared SiluAndMul under a clamped routed swiglu)
        # must keep the separate shared-expert forward.
        # TODO(ZT-AIA): The MegaMoe operator currently executes one activation
        # for both routed and shared experts, so it cannot represent configs
        # where the two differ. Remove this activation-signature equality check
        # once the operator supports distinct routed/shared activations.
        routed_activation = self._routed_megamoe_activation_signature()
        shared_activation = self._shared_megamoe_activation_signature()
        if shared_activation is None or shared_activation != routed_activation:
            self._disable_megamoe_shared_fusion(
                f"shared activation {type(self.layer.act_fn).__name__} {shared_activation} "
                f"does not match the routed activation {routed_activation} executed by the MegaMoe operator"
            )
            return
        # The operator accumulates the shared-expert output into the local
        # token accumulator with no cross-rank reduction, so MegaMoe needs full
        # shared weights on every rank. When enable_shared_expert_dp=true (or
        # tp_size == 1), weights are already full. On Ascend A3 under
        # TP-sharded shared weights (both TENSOR_PARALLEL and non-transform
        # SEQUENCE_PARALLEL_ONLY with enable_shared_expert_dp=false), we
        # assemble full weights once at post-load time via TP all_gather into
        # _megamoe_shared_parts while keeping the original 1/TP linear shards
        # intact for ALLTOALL/MC2 fallback execution.
        tp_world_size = self.moe_config.tp_group.world_size
        if tp_world_size > 1 and not self.weights_replicated:
            if supports_mxfp:
                self._disable_megamoe_shared_fusion(
                    "shared-expert weights are sharded across the TP group; set "
                    "enable_shared_expert_dp=true to enable MegaMoe shared-expert fusion"
                )
                return
            if getattr(self.layer.gate_up_proj, "gather_output", False) or getattr(
                self.layer.down_proj, "reduce_results", False
            ):
                self._disable_megamoe_shared_fusion(
                    "TP-sharded shared linears with internal gather_output/reduce_results "
                    "cannot be combined with MegaMoe shared-expert fusion"
                )
                return
            self._megamoe_tp_gather_shards = True

        for linear, slot in (
            (self.layer.gate_up_proj, "w1"),
            (self.layer.down_proj, "w2"),
        ):
            self._wrap_linear_for_megamoe_fusion(linear, slot)

    def _routed_megamoe_activation_signature(self) -> tuple[str, dict[str, float]] | None:
        """Canonical ``(name, params)`` of the activation the operator executes.

        Mirrors :func:`select_mega_moe_activation_kwargs`: whatever kwargs this
        builds are the ones actually bound into the CANN MegaMoe call, so this
        — not the raw ``FusedMoEConfig.activation`` tag — is the function the
        shared experts must equal. ``None`` marks an activation the operator
        cannot represent (GELU variants, swiglustep, ...), which always keeps
        the separate shared-expert forward.
        """
        moe_config = self.moe_config
        activation = getattr(moe_config, "activation", "silu")
        name = str(getattr(activation, "value", activation)).lower()
        clamp = moe_config.swiglu_limit if (moe_config.swiglu_limit or 0.0) > 0 else None
        if name == "situ":
            params = {"beta": 1.0 if moe_config.activation_situ_beta is None else moe_config.activation_situ_beta}
            if moe_config.activation_situ_linear_beta is not None:
                params["linear_beta"] = moe_config.activation_situ_linear_beta
            return ("situglu", params)
        if name in ("swigluoai", "swigluoai_uninterleave"):
            params = {
                "alpha": 1.0 if moe_config.swiglu_alpha is None else moe_config.swiglu_alpha,
                "beta": 0.0 if moe_config.swiglu_beta is None else moe_config.swiglu_beta,
            }
            if clamp is not None:
                params["clamp"] = clamp
            return ("swigluoai", params)
        if name == "silu":
            params = {"alpha": 1.0, "beta": 0.0}
            if clamp is not None:
                params["clamp"] = clamp
            return ("silu", params)
        return None

    def _shared_megamoe_activation_signature(self) -> tuple[str, dict[str, float]] | None:
        """Canonical ``(name, params)`` of the shared expert's activation module.

        ``None`` when the activation has no MegaMoe counterpart (GELU variants,
        unknown custom activations) or its standalone Ascend semantics cannot
        be reproduced by the operator's single global activation.
        """
        act_fn = self.layer.act_fn
        if isinstance(act_fn, SituAndMul):
            params = {"beta": act_fn.beta}
            if act_fn.linear_beta is not None:
                params["linear_beta"] = act_fn.linear_beta
            return ("situglu", params)
        if isinstance(act_fn, SiluAndMulWithClamp):
            params = {"alpha": getattr(act_fn, "alpha", 1.0), "beta": getattr(act_fn, "beta", 0.0)}
            limit = getattr(act_fn, "swiglu_limit", None)
            if limit is not None and limit > 0:
                params["clamp"] = limit
            return ("silu", params)
        if isinstance(act_fn, SiluAndMul):
            return ("silu", {"alpha": 1.0, "beta": 0.0})
        # vLLM's SwigluOAIAndMul hardcodes (up + 1); model-private look-alike
        # classes (e.g. MiniMaxM3SwiGLUOAI) carry alpha/beta/limit as attributes.
        if isinstance(act_fn, SwigluOAIAndMul) or (hasattr(act_fn, "alpha") and hasattr(act_fn, "limit")):
            params = {
                "alpha": act_fn.alpha,
                "beta": getattr(act_fn, "beta", 1.0),
            }
            limit = getattr(act_fn, "limit", None)
            if limit is not None and limit > 0:
                params["clamp"] = limit
            return ("swigluoai", params)
        return None

    def _wrap_linear_for_megamoe_fusion(self, linear: torch.nn.Module, slot: str) -> None:
        """Rebuild the shared weight right after its linear is processed.

        The wrapper is attached to the linear's own quant-method instance and
        ignores calls for other layers, so a shared quant-method instance
        stays safe.
        """
        quant_method = getattr(linear, "quant_method", None)
        original = getattr(quant_method, "process_weights_after_loading", None)
        if original is None:
            self._disable_megamoe_shared_fusion(
                f"shared linear {type(linear).__name__} has no process_weights_after_loading"
            )
            return

        @wraps(original)
        def wrapped_megamoe_collect(*args, **kwargs):
            result = original(*args, **kwargs)
            if args and args[0] is linear:
                self._collect_megamoe_shared_weight(slot, linear)
            return result

        quant_method.process_weights_after_loading = wrapped_megamoe_collect  # type: ignore

    def _collect_megamoe_shared_weight(self, slot: str, linear: torch.nn.Module) -> None:
        """Validate and stage one shared linear according to ``AscendMegaMoESharedSpec`` & ``AscendMegaMoEFormat``."""
        if self._megamoe_shared_fusion_disabled:
            return

        tp_gather_mode = bool(getattr(self, "_megamoe_tp_gather_shards", False))
        staging = self._megamoe_local_slot_staging

        def _reject_slot(reason: str) -> None:
            if tp_gather_mode:
                staging[slot] = _StagedSharedLinear(ok=False, reason=reason, linear=linear)
                self._publish_megamoe_shared_fusion()
            else:
                self._disable_megamoe_shared_fusion(reason)

        padding = getattr(linear, "mxfp8_tp_padding", getattr(linear, "mxfp4_tp_padding", (0, 0)))
        if any(padding):
            _reject_slot(f"padded shared-expert weights {padding} are unsupported by MegaMoe fusion")
            return

        raw_weight = linear.weight.data
        if _is_npu_runtime_available() and raw_weight.device.type != "npu":
            _reject_slot(f"shared-expert {slot} weight resides on {raw_weight.device.type} instead of npu")
            return
        if raw_weight.ndim != 2:
            _reject_slot(f"shared-expert {slot} weight must be 2D, got {tuple(raw_weight.shape)}")
            return

        shared_quant_type = _resolve_shared_linear_quant_type(linear)
        spec = _ASCEND_MEGAMOE_SHARED_SPECS.get(shared_quant_type) if shared_quant_type is not None else None
        fmt = _ASCEND_MEGAMOE_FORMATS.get((self.quant_type, shared_quant_type)) if shared_quant_type else None
        supports_mxfp = get_current_hardware_profile().supports(HardwareCapability.CANN_MEGAMOE_MXFP)
        if spec is None or fmt is None or fmt.requires_mxfp_capability != supports_mxfp:
            _reject_slot(
                f"(routed={self.quant_type}, shared={shared_quant_type}) is not registered "
                "for MegaMoe shared-expert fusion on the current hardware profile"
            )
            return
        if raw_weight.dtype != spec.carrier_dtype:
            _reject_slot(f"shared {slot} expects carrier {spec.carrier_dtype}, got {raw_weight.dtype}")
            return

        tp_world_size = self.moe_config.tp_group.world_size
        linear_tp_size = int(getattr(linear, "tp_size", tp_world_size))
        if tp_gather_mode and linear_tp_size not in (1, tp_world_size):
            _reject_slot(f"shared-expert {slot} tp_size={linear_tp_size} != expected TP size {tp_world_size}")
            return
        shard_tp_size = tp_world_size if tp_gather_mode and linear_tp_size > 1 else 1

        local_in_packed, local_out_dim = (
            (int(raw_weight.shape[0]), int(raw_weight.shape[1]))
            if spec.post_load_transposed
            else (int(raw_weight.shape[1]), int(raw_weight.shape[0]))
        )
        if shard_tp_size > 1 and slot == "w1":
            output_sizes = getattr(linear, "output_sizes", None)
            if local_out_dim % 2 != 0 or (
                isinstance(output_sizes, (list, tuple))
                and (len(output_sizes) != 2 or output_sizes[0] != output_sizes[1])
            ):
                _reject_slot(f"TP-sharded shared gate_up_proj requires equal gate/up partitions, got {output_sizes}")
                return
            in_packed, out_dim = local_in_packed, local_out_dim * shard_tp_size
        elif shard_tp_size > 1:
            in_packed, out_dim = local_in_packed * shard_tp_size, local_out_dim
        else:
            in_packed, out_dim = local_in_packed, local_out_dim

        if slot == "w1":
            min_gu, align_gu = 2 * _A3_MIN_SHARED_INTER_DIM, 2 * _A3_SHARED_INTER_DIM_ALIGN
            expected_in = self.moe_config.hidden_dim // spec.k_pack_factor
            shape_ok = in_packed == expected_in and (
                (out_dim >= min_gu and out_dim % align_gu == 0)
                if fmt.allow_independent_shared_intermediate
                else out_dim == 2 * self.moe_config.intermediate_size_per_partition
            )
        else:
            shared_inter = in_packed * spec.k_pack_factor
            shape_ok = out_dim == self.moe_config.hidden_dim and (
                (shared_inter >= _A3_MIN_SHARED_INTER_DIM and shared_inter % _A3_SHARED_INTER_DIM_ALIGN == 0)
                if fmt.allow_independent_shared_intermediate
                else in_packed == self.moe_config.intermediate_size_per_partition // spec.k_pack_factor
            )
        if not shape_ok:
            _reject_slot(f"shared-expert {slot} logical dims (in={in_packed}, out={out_dim}) violate MegaMoe contract")
            return

        if spec.requires_symmetric_zero_offset:
            offset_attr = getattr(linear, "weight_offset", None)
            offset_data = getattr(offset_attr, "data", offset_attr)
            if (
                isinstance(offset_data, torch.Tensor)
                and offset_data.numel() > 0
                and bool((offset_data != 0).any().item())
            ):
                _reject_slot(f"shared-expert {slot} carries non-zero asymmetric weight_offset")
                return

        local_scale_fp32: torch.Tensor | None = None
        if spec.scale_kind == "int64_low32":
            scale_attr = getattr(linear, "weight_scale_fp32", getattr(linear, "weight_scale", None))
            scale_raw = getattr(scale_attr, "data", scale_attr)
            expected_len = local_out_dim if (shard_tp_size > 1 and slot == "w1") else out_dim
            if not isinstance(scale_raw, torch.Tensor):
                _reject_slot(f"shared-expert {slot} is missing weight_scale for {shared_quant_type}")
                return
            with _preserve_allow_internal_format():
                _set_allow_internal_format(False)
                local_scale_fp32 = scale_raw.detach().to(torch.float32).reshape(-1).contiguous()
            if (
                local_scale_fp32.numel() != expected_len
                or not bool(torch.isfinite(local_scale_fp32).all().item())
                or not bool((local_scale_fp32 > 0).all().item())
            ):
                _reject_slot(f"shared-expert {slot} weight_scale must be finite positive of length {expected_len}")
                return
        elif spec.scale_kind == "mxfp_e8m0":
            if not isinstance(getattr(getattr(linear, "weight_scale", None), "data", None), torch.Tensor):
                _reject_slot(f"shared-expert {slot} is missing MXFP weight_scale")
                return

        staging[slot] = _StagedSharedLinear(
            ok=True,
            linear=linear,
            shared_quant_type=shared_quant_type,
            spec=spec,
            fmt=fmt,
            shard_tp_size=shard_tp_size,
            local_in_packed=local_in_packed,
            local_out_dim=local_out_dim,
            in_packed=in_packed,
            out_dim=out_dim,
            local_scale_fp32=local_scale_fp32,
        )
        self._publish_megamoe_shared_fusion()

    def megamoe_shared_fusion(self) -> MegamoeSharedExpertFusion:
        """Payload the routed layer needs to run the shared expert in MegaMoe.

        Returned once, right after construction, so ``AscendMoERunner`` can
        install it explicitly on the routed-expert layer. Weight loading runs
        later, hence the returned object is live: this class fills its fields
        as the shared linears are rebuilt, and consumers treat ``weights is
        None`` as "fusion inactive".
        """
        return self._megamoe_shared_fusion

    def _publish_megamoe_shared_fusion(self) -> None:
        """Materialize ``w1`` and ``w2`` in deterministic order once both slots are staged."""
        if self._megamoe_shared_fusion_disabled:
            return
        staging = self._megamoe_local_slot_staging
        if "w1" not in staging or "w2" not in staging:
            return
        s1, s2 = staging["w1"], staging["w2"]
        staging.clear()

        local_ok = bool(s1.ok and s2.ok)
        reason = s1.reason or s2.reason
        npu_runtime = _is_npu_runtime_available()
        if local_ok:
            assert s1.spec is not None and s2.spec is not None
            if npu_runtime and (
                s1.linear is None
                or s2.linear is None
                or s1.linear.weight.device.type != "npu"
                or s2.linear.weight.device.type != "npu"
            ):
                local_ok, reason = False, "shared-expert weights are not resident on npu"
            elif s1.shared_quant_type != s2.shared_quant_type:
                local_ok = False
                reason = f"shared quant types differ ({s1.shared_quant_type} vs {s2.shared_quant_type})"
            elif s1.shard_tp_size != s2.shard_tp_size:
                local_ok = False
                reason = f"shared TP shard factors differ ({s1.shard_tp_size} vs {s2.shard_tp_size})"
            elif s1.out_dim != 2 * (s2.in_packed * s2.spec.k_pack_factor):
                local_ok = False
                down_in = s2.in_packed * s2.spec.k_pack_factor
                reason = f"shared gate_up out_dim ({s1.out_dim}) != 2 * down in_dim ({down_in})"

        tp_group = self.moe_config.tp_group
        if self._megamoe_tp_gather_shards:
            dev = (
                torch.device(f"npu:{torch.npu.current_device()}")
                if npu_runtime
                else getattr(getattr(s1.linear or s2.linear, "weight", None), "device", torch.device("cpu"))
            )
            status_t = torch.tensor([1 if local_ok else 0], dtype=torch.int32, device=dev)
            if not bool((tp_group.all_gather(status_t, dim=0) == 1).all().item()):
                self._disable_megamoe_shared_fusion(reason or "peer TP rank rejected shared-expert fusion")
                return
        elif not local_ok:
            self._disable_megamoe_shared_fusion(reason)
            return

        assert s1.linear is not None and s2.linear is not None
        assert s1.spec is not None and s1.fmt is not None and s1.shared_quant_type is not None
        spec, fmt, shard_tp_size = s1.spec, s1.fmt, s1.shard_tp_size
        self._megamoe_shared_quant_type = s1.shared_quant_type

        with _preserve_allow_internal_format():
            if spec.scale_kind == "int64_low32":
                assert s1.local_scale_fp32 is not None and s2.local_scale_fp32 is not None
                _set_allow_internal_format(False)
                w1_s = (
                    tp_group.all_gather(s1.local_scale_fp32, dim=0)
                    .reshape(shard_tp_size, 2, s1.local_out_dim // 2)
                    .permute(1, 0, 2)
                    .reshape(s1.out_dim)
                    .contiguous()
                    if shard_tp_size > 1
                    else s1.local_scale_fp32
                )
                for key, s_fp32 in (("w1_scale", w1_s), ("w2_scale", s2.local_scale_fp32)):
                    # CANN Arch22 reads the low 32 bits of each ACL_UINT64 slot as IEEE-754 float32.
                    packed = (s_fp32.view(torch.int32).to(torch.int64) & 0xFFFFFFFF).contiguous()
                    self._megamoe_shared_parts[key] = packed

            for slot_key, st, target_fmt in (("w1", s1, fmt.w1_acl_format), ("w2", s2, fmt.w2_acl_format)):
                assert st.linear is not None
                raw_w = st.linear.weight.data
                if shard_tp_size > 1:
                    _set_allow_internal_format(False)
                    local_nd = torch_npu.npu_format_cast(raw_w, ACL_FORMAT_FRACTAL_ND)
                    local_in_out = (
                        local_nd.contiguous() if spec.post_load_transposed else local_nd.transpose(0, 1).contiguous()
                    )
                    if slot_key == "w1":
                        gathered = tp_group.all_gather(local_in_out.transpose(0, 1).contiguous(), dim=0)
                        full_out_in = (
                            gathered.reshape(shard_tp_size, 2, st.local_out_dim // 2, st.local_in_packed)
                            .permute(1, 0, 2, 3)
                            .reshape(st.out_dim, st.local_in_packed)
                        )
                        w_nd = (
                            full_out_in.transpose(0, 1).contiguous()
                            if fmt.kernel_transposed
                            else full_out_in.contiguous()
                        )
                    else:
                        gathered = tp_group.all_gather(local_in_out, dim=0)
                        w_nd = gathered.contiguous() if fmt.kernel_transposed else gathered.transpose(0, 1).contiguous()
                    if target_fmt != ACL_FORMAT_FRACTAL_ND:
                        _set_allow_internal_format(True)
                        weight = torch_npu.npu_format_cast(w_nd, target_fmt)
                    else:
                        weight = w_nd
                else:
                    # Orientation is governed algebraically by (post_load_transposed != kernel_transposed),
                    # avoiding shape heuristics on square matrices and reusing existing FRACTAL_NZ zero-copy.
                    needs_t = spec.post_load_transposed != fmt.kernel_transposed
                    needs_c0_32 = slot_key == "w2" and fmt.w2_mxfp4_c0_32_cast
                    cur_fmt = _get_tensor_acl_format(raw_w)
                    if not needs_t and not needs_c0_32:
                        if target_fmt == ACL_FORMAT_FRACTAL_NZ and cur_fmt == ACL_FORMAT_FRACTAL_NZ:
                            weight = raw_w.detach()
                        elif target_fmt == ACL_FORMAT_FRACTAL_ND and cur_fmt in (None, 0, ACL_FORMAT_FRACTAL_ND):
                            _set_allow_internal_format(False)
                            weight = raw_w.detach().contiguous()
                        else:
                            _set_allow_internal_format(target_fmt != ACL_FORMAT_FRACTAL_ND)
                            weight = torch_npu.npu_format_cast(raw_w.contiguous(), target_fmt)
                    else:
                        _set_allow_internal_format(False)
                        w_nd = torch_npu.npu_format_cast(raw_w, ACL_FORMAT_FRACTAL_ND)
                        w_nd = w_nd.transpose(0, 1).contiguous() if needs_t else w_nd.contiguous()
                        if needs_c0_32:
                            _set_allow_internal_format(True)
                            weight = torch_npu.npu_format_cast(
                                w_nd,
                                ACL_FORMAT_FRACTAL_NZ,
                                customize_dtype=torch.float8_e4m3fn,
                                input_dtype=torch_npu.float4_e2m1fn_x2,
                            )
                        elif target_fmt != ACL_FORMAT_FRACTAL_ND:
                            _set_allow_internal_format(True)
                            weight = torch_npu.npu_format_cast(w_nd, target_fmt)
                        else:
                            weight = w_nd
                self._megamoe_shared_parts[slot_key] = weight
                if spec.scale_kind == "mxfp_e8m0":
                    _set_allow_internal_format(False)
                    self._megamoe_shared_parts[f"{slot_key}_scale"] = (
                        st.linear.weight_scale.data.transpose(0, 1).contiguous().view(torch.float8_e8m0fnu)
                    )

        has_scale = spec.scale_kind != "none"
        _, quant_out_dtype, weight_type = moe_utils._get_cann_mega_moe_quant_settings(self._megamoe_shared_quant_type)
        fusion = self._megamoe_shared_fusion
        fusion.weights = MegamoeSharedWeights(
            w1=[self._megamoe_shared_parts["w1"]],
            w2=[self._megamoe_shared_parts["w2"]],
            w1_scale=[self._megamoe_shared_parts["w1_scale"]] if has_scale else None,
            w2_scale=[self._megamoe_shared_parts["w2_scale"]] if has_scale else None,
        )
        fusion.quant_out_dtype = quant_out_dtype
        fusion.weight_type = weight_type
        if not self.megamoe_shared_weights_ready:
            self.megamoe_shared_weights_ready = True
            logger.info_once(
                "Armed MegaMoe shared-expert fusion (routed=%s, shared=%s).",
                self.quant_type,
                self._megamoe_shared_quant_type,
            )

    def _disable_megamoe_shared_fusion(self, reason: str) -> None:
        if not self._megamoe_shared_fusion_disabled:
            logger.info_once(
                "Disabling MegaMoe shared-expert fusion: %s; keeping the separate shared-expert forward.",
                reason,
            )
        self._megamoe_shared_fusion_disabled = True
        self.megamoe_shared_weights_ready = False
        staging = getattr(self, "_megamoe_local_slot_staging", None)
        if staging is not None:
            staging.clear()
        self._megamoe_shared_parts.clear()
        self._megamoe_shared_fusion.weights = None
        self._megamoe_shared_fusion.quant_out_dtype = None
        self._megamoe_shared_fusion.weight_type = None

    def set_lora_context(self, lora_context) -> None:
        self.lora_context = lora_context

    def validate_consistency(self):
        """Validate that split shared expert computation matches integrated computation."""
        test_input = (
            torch.rand(
                10,
                self.shared_expert_input_size,
                device="npu",
                dtype=self.in_dtype,
            )
            * 2
            - 1
        )  # Random input for testing, scoped to [-1, 1]

        integrated_out = self.layer(test_input)
        part1_out = self.part1(test_input)
        if self._select_mlp_path() is SharedExpertMLPPath.W8A8_MXFP_LINEAR:
            down_input = self._prepare_w8a8_mxfp_down_input(part1_out)
        else:
            down_input = self.apply_activation(part1_out)
        split_out = self.part2(test_input, down_input)

        if not torch.allclose(integrated_out, split_out):
            diff = (integrated_out - split_out).abs()
            logger.error(
                "[fused_moe/layer] Shared expert split computation validation failed."
                " The split-path computation does not match the integrated-path result."
                " max_abs_diff=%s, integrated_sum=%s, integrated_norm=%s,"
                " split_sum=%s, split_norm=%s, hidden_size=%s, dtype=%s.",
                diff.max().item(),
                integrated_out.sum().item(),
                integrated_out.norm().item(),
                split_out.sum().item(),
                split_out.norm().item(),
                self.shared_expert_input_size,
                self.in_dtype,
            )
            raise ValueError("FusedMoE shared experts split computation does not match the integrated computation.")
        logger.info_once(
            "[fused_moe/layer] Shared expert split computation validation passed."
            " Integrated and split-path results are consistent."
        )

    def part1(
        self,
        hidden_states: LinearInput,
    ):
        shared_gate_up, _ = self.layer.gate_up_proj(hidden_states)  # type: ignore
        return shared_gate_up

    def apply_activation(self, shared_gate_up: torch.Tensor):
        return self.layer.act_fn(shared_gate_up)  # type: ignore

    def part2(
        self,
        hidden_states: torch.Tensor,
        down_input: LinearInput,
    ):
        shared_out, _ = self.layer.down_proj(down_input)  # type: ignore
        return self._apply_output_gate(hidden_states, shared_out)

    def _apply_output_gate(self, hidden_states: torch.Tensor, shared_out: torch.Tensor) -> torch.Tensor:
        # Qwen3-Next specific gating mechanism
        if hasattr(self.layer, "expert_gate") and self.layer.expert_gate is not None:
            gate_out, _ = self.layer.expert_gate(hidden_states)  # type: ignore
            shared_out = F.sigmoid(gate_out) * shared_out
        return shared_out

    def parallel_mode(self) -> SharedExpertParallelMode:
        """Resolve the effective activation/weight layout for this forward."""
        # EP rewrites FusedMoEParallelConfig.tp_size to 1 because each routed
        # expert is local. Shared-expert linears still span the physical TP
        # group, so their layout must be derived from that group instead.
        tp_size = self.moe_config.tp_group.world_size
        if tp_size <= 1:
            return SharedExpertParallelMode.TENSOR_PARALLEL

        if self.moe_config.is_sequence_parallel:
            # SP has already sharded the token dimension before entering the
            # runner. Replicated weights (SP+DP) compute directly on the
            # shard; TP-sharded weights (SP-only) gather the shard first and
            # reduce-scatter the output back.
            if self.weights_replicated:
                return SharedExpertParallelMode.SEQUENCE_PARALLEL_SEDP
            return SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY
        if self.weights_replicated:
            return SharedExpertParallelMode.SHARED_EXPERT_DATA_PARALLEL_ONLY
        return SharedExpertParallelMode.TENSOR_PARALLEL

    def _prepare_local_dp_input(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        original_num_tokens = hidden_states.shape[0]
        # See parallel_mode(): moe_config.tp_size describes routed experts in
        # EP, while this token split follows the shared-expert TP group.
        tp_group = self.moe_config.tp_group
        tp_size = tp_group.world_size
        pad_size = (tp_size - original_num_tokens % tp_size) % tp_size
        if pad_size > 0:
            hidden_states = _pad_tokens_with_cat(hidden_states, original_num_tokens + pad_size)
        hidden_states = torch.tensor_split(
            hidden_states,
            tp_size,
            dim=0,
        )[tp_group.rank_in_group]
        return hidden_states, (original_num_tokens, pad_size)

    def _finalize_local_dp_output(
        self,
        shared_out: torch.Tensor,
        metadata: tuple[int, int],
    ) -> torch.Tensor:
        original_num_tokens, pad_size = metadata
        shared_out = self.moe_config.tp_group.all_gather(shared_out, dim=0)
        if pad_size > 0:
            shared_out = shared_out[:original_num_tokens]
        return shared_out

    def _gather_sp_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Gather SP-sharded activations to the full sequence via TP all-gather + unpad.

        SP shards the token dimension across the TP group as [ceil(T/TP), H]
        per rank (``sequence_parallel_chunk`` pads T to a multiple of TP).
        TP-sharded shared-expert weights need the full activations, so
        all-gather the shards back to [T_padded, H] and drop the padding rows.
        """
        gathered = tensor_model_parallel_all_gather(hidden_states, dim=0)
        return gathered[: _EXTRA_CTX.num_tokens]

    def _pad_and_reduce_scatter(self, shared_out: torch.Tensor) -> torch.Tensor:
        """Pad the full output to a TP multiple, then reduce-scatter to the SP shard.

        Called on exit of the SP-only path (TP-sharded weights) to convert the
        full [T, H] output back to this rank's SP shard [ceil(T/TP), H]. The
        reduce-scatter also sums the TP-partial down-projection results
        (down_proj is built with reduce_results=False), replacing the usual
        TP all-reduce.
        """
        tp_size = self.moe_config.tp_group.world_size
        original_num_tokens = shared_out.shape[0]
        pad_size = (tp_size - original_num_tokens % tp_size) % tp_size
        if pad_size > 0:
            shared_out = F.pad(shared_out, (0, 0, 0, pad_size))
        return tensor_model_parallel_reduce_scatter(shared_out, dim=0)

    def prepare_input_async(
        self,
        hidden_states: torch.Tensor,
    ) -> PreparedSharedExpertInput:
        """Start the SP-only input all-gather on the shared-expert stream.

        The caller must wait for ``ready_event`` before enqueueing routed-path
        collectives.  This method is intended to overlap the gather only with
        a routed input transform such as Kimi-K3's latent projection.
        """
        if not (self.multistream_overlap and self.parallel_mode() is SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY):
            return PreparedSharedExpertInput(hidden_states=hidden_states)

        input_ready = torch.npu.current_stream().record_event()
        with npu_stream_switch(shared_experts_calculation_stream(), enabled=True):
            torch.npu.current_stream().wait_event(input_ready)
            # Keep TP padding across the custom-op boundary; the shared MLP
            # trims it before computation.
            hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)
            all_gather_done = torch.npu.current_stream().record_event()
        return PreparedSharedExpertInput(
            hidden_states=hidden_states,
            is_gathered=True,
            ready_event=all_gather_done,
        )

    def prepare_input_before_routed(
        self,
        hidden_states: torch.Tensor,
    ) -> PreparedSharedExpertInput:
        """Prepare SP-only input on the calling stream before routed MoE.

        The TP all-gather deliberately stays on the default/routed stream.
        Running it on the shared-expert stream while routed EP collectives run
        on the default stream can deadlock HCCL when ranks make different
        progress across the two process groups.
        """
        if not (self.multistream_overlap and self.parallel_mode() is SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY):
            return PreparedSharedExpertInput(hidden_states=hidden_states)

        # Keep TP padding across the custom-op boundary; the shared MLP trims
        # it before computation. The caller records shared_input_ready after
        # this returns, handing the gathered tensor to the auxiliary stream.
        hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)
        return PreparedSharedExpertInput(
            hidden_states=hidden_states,
            is_gathered=True,
        )

    def _wait_for_milestone(
        self,
        event: torch.npu.Event | None,
        name: str,
    ) -> None:
        """Wait for a required routed milestone when multistream is enabled."""
        if not self.multistream_overlap:
            return
        if event is None:
            raise RuntimeError(f"Missing {name} event while shared-expert multistream is enabled.")
        torch.npu.current_stream().wait_event(event)

    def _wait_for_routed_stage(
        self,
        milestones: RoutedMoEMilestones,
        event: torch.npu.Event | None,
        name: str,
    ) -> None:
        """Wait when the routed implementation exposes stage boundaries.

        Fused routed implementations such as FusedMC2 execute their stages in
        one operator and therefore cannot provide intermediate events. Shared
        experts remain independent and run as one concurrent auxiliary-stream
        workload in that case.
        """
        if milestones.has_fine_grained_stage_events:
            self._wait_for_milestone(event, name)

    def _prepare_execution_input(
        self,
        prepared_input: PreparedSharedExpertInput,
        milestones: RoutedMoEMilestones,
        mode: SharedExpertParallelMode,
    ) -> tuple[torch.Tensor, tuple[int, int] | None]:
        self._wait_for_milestone(
            milestones.shared_input_ready,
            "shared_input_ready",
        )
        hidden_states = prepared_input.hidden_states
        local_dp_metadata = None
        if mode is SharedExpertParallelMode.SHARED_EXPERT_DATA_PARALLEL_ONLY:
            # Full activations + replicated weights: shard tokens locally,
            # run the MLP, then gather its complete output.
            hidden_states, local_dp_metadata = self._prepare_local_dp_input(hidden_states)
        elif mode is SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY and not prepared_input.is_gathered:
            # TP-sharded weights require full activations. Multistream starts
            # this gather before the routed path; the serial path does it here.
            hidden_states = self._gather_sp_input(hidden_states)
        elif mode is SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY:
            # An early gather keeps padding across the custom-op boundary so
            # tracing sees the explicit gathered dependency. The MLP consumes
            # only the original tokens.
            hidden_states = hidden_states[: _EXTRA_CTX.num_tokens]
        return hidden_states, local_dp_metadata

    def prepare_for_router_overlap(
        self,
        prepared_input: PreparedSharedExpertInput,
        shared_input_ready: torch.npu.Event,
    ) -> PreparedSharedExpertExecution | None:
        """Queue Gate-Up input preparation before the router matmul.

        Recording an event before the router is insufficient by itself: if the
        auxiliary-stream work is submitted only after routed MoE, device-side
        execution still starts late. This method is called immediately after
        ``shared_input_ready`` is recorded, so Vector/AIV activation
        preparation can run concurrently with the Cube-heavy router matmul.

        Unsupported wrappers return the unmodified tensor as ``gate_up_input``;
        their complete registered linear is still ordered after
        ``router_output_ready`` and is submitted before routed TopK/dispatch.
        """
        if not self.multistream_overlap:
            return None

        mode = self.parallel_mode()
        milestones = RoutedMoEMilestones(shared_input_ready=shared_input_ready)
        with npu_stream_switch(shared_experts_calculation_stream(), enabled=True):
            hidden_states, local_dp_metadata = self._prepare_execution_input(
                prepared_input,
                milestones,
                mode,
            )
            gate_up_input = self._prepare_gate_up_input(hidden_states)
        return PreparedSharedExpertExecution(
            hidden_states=hidden_states,
            gate_up_input=gate_up_input,
            local_dp_metadata=local_dp_metadata,
        )

    def _prepare_gate_up_input(self, hidden_states: torch.Tensor) -> LinearInput:
        """Prepare the selected Gate-Up path without running its matmul."""
        path = self._select_mlp_path()
        if path is SharedExpertMLPPath.A8_INT_FUSED:
            return torch_npu.npu_dynamic_quant(hidden_states)
        if path is SharedExpertMLPPath.A8_MXFP_FUSED:
            return torch_npu.npu_dynamic_mx_quant(
                hidden_states,
                dst_type=torch.float8_e4m3fn,
            )
        return self._prepare_linear_input_for_overlap(self.layer.gate_up_proj, hidden_states)

    def _prepare_linear_input_for_overlap(
        self,
        layer: torch.nn.Module,
        hidden_states: LinearInput,
    ) -> LinearInput:
        """Run scheme-owned Vector/AIV preparation before a stage barrier."""
        if not self.multistream_overlap or has_lora(self.lora_context):
            return hidden_states
        if not isinstance(hidden_states, torch.Tensor):
            # Activation may have already quantized the Down input and scale.
            return hidden_states

        linear_method = getattr(layer, "quant_method", None)
        quant_scheme = getattr(linear_method, "quant_method", linear_method)
        prepare_input = getattr(quant_scheme, "prepare_input_for_overlap", None)
        if prepare_input is None:
            return hidden_states

        prepared_input = prepare_input(layer, hidden_states)
        return hidden_states if prepared_input is None else prepared_input

    def enqueue_gate_up_after_router(
        self,
        prepared_execution: PreparedSharedExpertExecution,
        router_output_ready: torch.npu.Event | None,
    ) -> SubmittedSharedExpertGateUp:
        """Submit Gate-Up before routed TopK/dispatch are enqueued.

        Gate-Up remains ordered after the router through ``router_output_ready``
        but is placed on the auxiliary stream immediately, allowing its Cube
        work to cover routed TopK and dispatch instead of missing that window
        because of host-side submission order.
        """
        path = self._select_mlp_path()
        gate_up_input_scale = None
        with npu_stream_switch(shared_experts_calculation_stream(), enabled=self.multistream_overlap):
            self._wait_for_milestone(router_output_ready, "router_output_ready")
            if path is SharedExpertMLPPath.A8_INT_FUSED:
                # W8A8/W4A8: INT8 activations with INT8 or INT4 weights.
                assert isinstance(prepared_execution.gate_up_input, tuple)
                quantized_x, gate_up_input_scale = prepared_execution.gate_up_input
                gate_up = torch_npu.npu_quant_matmul(
                    quantized_x,
                    self.layer.gate_up_proj.weight,
                    self.layer.gate_up_proj.weight_scale,
                    pertoken_scale=None,
                    bias=None,
                    output_dtype=torch.int32,
                )
            elif path is SharedExpertMLPPath.A8_MXFP_FUSED:
                # W4A8MXFP only: MXFP8 activations with MXFP4 weights.
                # W8A8MXFP remains on the registered linear-wrapper path.
                assert isinstance(prepared_execution.gate_up_input, tuple)
                quantized_x, pertoken_scale = prepared_execution.gate_up_input
                gate_up = self.layer.gate_up_proj((quantized_x, pertoken_scale))[0]
            else:
                gate_up = self.part1(prepared_execution.gate_up_input)

        return SubmittedSharedExpertGateUp(
            hidden_states=prepared_execution.hidden_states,
            gate_up=gate_up,
            gate_up_input_scale=gate_up_input_scale,
            local_dp_metadata=prepared_execution.local_dp_metadata,
            path=path,
        )

    def _run_a8_int_mlp(
        self,
        submitted_gate_up: SubmittedSharedExpertGateUp,
        milestones: RoutedMoEMilestones,
    ) -> torch.Tensor:
        """Finish the fused W8A8/W4A8 INT-activation shared MLP.

        Both quantization modes use INT8 activations. W8A8 supplies INT8
        weights, while W4A8 supplies INT4 weights. Gate-Up has already been
        submitted before routed TopK/dispatch; this stage performs fused
        activation/requantization and the Down projection.
        """
        pertoken_scale = submitted_gate_up.gate_up_input_scale
        assert pertoken_scale is not None
        self._wait_for_routed_stage(
            milestones,
            milestones.shared_activation_overlap_start,
            "shared_activation_overlap_start",
        )
        if self.situ_activation is not None:
            quantized_x, swiglu_out_scale = torch.ops._C_ascend.dequant_situ_quant(
                x=submitted_gate_up.gate_up,
                weight_scale=self.layer.gate_up_proj.weight_scale_fp32,
                activation_scale=pertoken_scale,
                bias=None,
                quant_scale=None,
                quant_offset=None,
                group_index=None,
                beta=self.situ_activation.beta,
                linear_beta=self.situ_activation.linear_beta or 0.0,
                activate_left=True,
                quant_mode="dynamic",
            )
        else:
            quantized_x, swiglu_out_scale = torch.ops._C_ascend.npu_dequant_swiglu_quant(
                x=submitted_gate_up.gate_up,
                weight_scale=self.layer.gate_up_proj.weight_scale_fp32,
                activation_scale=pertoken_scale,
                bias=None,
                quant_scale=None,
                quant_offset=None,
                group_index=None,
                activate_left=True,
                quant_mode=1,
                swiglu_mode=1,
                clamp_limit=self.swiglu_limit,
                **(
                    {}
                    if not get_current_hardware_profile().supports(HardwareCapability.FUSED_SWIGLU_TUNING_ARGS)
                    else {"glu_alpha": self.swiglu_alpha, "glu_bias": self.swiglu_beta}
                ),
            )
        self._wait_for_routed_stage(
            milestones,
            milestones.routed_combine_start,
            "routed_combine_start",
        )
        shared_out = torch_npu.npu_quant_matmul(
            quantized_x,
            self.layer.down_proj.weight,
            self.layer.down_proj.weight_scale,
            pertoken_scale=swiglu_out_scale,
            bias=None,
            output_dtype=submitted_gate_up.hidden_states.dtype,
        )
        return self._apply_output_gate(submitted_gate_up.hidden_states, shared_out)

    def _run_a8_mxfp_mlp(
        self,
        submitted_gate_up: SubmittedSharedExpertGateUp,
        milestones: RoutedMoEMilestones,
    ) -> torch.Tensor:
        """Finish the fused W4A8MXFP shared MLP.

        This path means MXFP4 weights with MXFP8 activations. W8A8MXFP does
        not enter here; it keeps the registered linear wrapper so its scheme-
        owned padding, scale algorithm, and output dtype remain intact.
        Gate-Up has already been submitted before routed TopK/dispatch.
        """
        self._wait_for_routed_stage(
            milestones,
            milestones.shared_activation_overlap_start,
            "shared_activation_overlap_start",
        )
        if self.situ_activation is not None:
            quantized_x, swiglu_out_scale = torch.ops._C_ascend.situ_mx_quant(
                x=submitted_gate_up.gate_up,
                beta=self.situ_activation.beta,
                linear_beta=self.situ_activation.linear_beta or 0.0,
                activate_left=True,
                dst_type=SITU_MX_DST_TYPE_E4M3FN,
            )
        else:
            quantized_x, swiglu_out_scale, _ = torch.ops._C_ascend.npu_swiglu_group_quant(
                submitted_gate_up.gate_up,
                topk_weight=None,
                group_index=None,
                dst_type=torch.float8_e4m3fn,
                quant_mode=2,
                clamp_value=self.swiglu_limit,
            )
        self._wait_for_routed_stage(
            milestones,
            milestones.routed_combine_start,
            "routed_combine_start",
        )
        return self.part2(submitted_gate_up.hidden_states, (quantized_x, swiglu_out_scale))

    def _can_fuse_w8a8_mxfp_activation(self, gate_up: torch.Tensor) -> bool:
        if self.quant_type != QuantType.W8A8MXFP or has_lora(self.lora_context):
            return False

        # Lazy import: the quantization module imports routed/shared MoE code.
        from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DynamicLinearMethod

        down_proj = self.layer.down_proj
        scheme = getattr(getattr(down_proj, "quant_method", None), "quant_method", None)
        # Prepared input bypasses the scheme's input padding and quantization.
        # Limit fusion to the existing kernel's BF16/default-MX contract.
        return (
            type(scheme) is AscendW8A8MXFP8DynamicLinearMethod
            and type(self.layer.act_fn) is AscendSiluAndMul
            and self.swiglu_limit == 0.0
            and scheme.dynamic_mx_quant_scale_alg == 0
            and scheme.group_size == MXFP8_ACTIVATION_GROUP_SIZE
            and not any(getattr(down_proj, "mxfp8_tp_padding", (0, 0)))
            and getattr(down_proj, "input_is_parallel", False)
            and gate_up.dtype == torch.bfloat16
            and gate_up.ndim == 2
            and gate_up.shape[-1] > 0
            and gate_up.shape[-1] % SWIGLU_MX_INPUT_ALIGNMENT == 0
            and hasattr(torch.ops._C_ascend, "npu_swiglu_group_quant")
        )

    def _prepare_w8a8_mxfp_down_input(self, gate_up: torch.Tensor) -> LinearInput:
        """Try fused SwiGLU/MX quant, otherwise use the scheme's preparation."""
        if self._can_fuse_w8a8_mxfp_activation(gate_up):
            quantized_x, scale, _ = torch.ops._C_ascend.npu_swiglu_group_quant(
                gate_up,
                topk_weight=None,
                group_index=None,
                dst_type=torch.float8_e4m3fn,
                quant_mode=SWIGLU_MX_QUANT_MODE,
                clamp_value=0.0,
            )
            return PreparedLinearInput(quantized_x, scale, gate_up.dtype)

        shared_act = self.apply_activation(gate_up)
        return self._prepare_linear_input_for_overlap(self.layer.down_proj, shared_act)

    def _finish_linear_wrapped_mlp(
        self,
        submitted_gate_up: SubmittedSharedExpertGateUp,
        milestones: RoutedMoEMilestones,
    ) -> torch.Tensor:
        self._wait_for_routed_stage(
            milestones,
            milestones.shared_activation_overlap_start,
            "shared_activation_overlap_start",
        )
        # All staged backends place activation/quantization alongside routed
        # GMM2. The Down matmul remains behind combine.
        if submitted_gate_up.path is SharedExpertMLPPath.W8A8_MXFP_LINEAR:
            down_input = self._prepare_w8a8_mxfp_down_input(submitted_gate_up.gate_up)
        else:
            shared_act = self.apply_activation(submitted_gate_up.gate_up)
            down_input = self._prepare_linear_input_for_overlap(self.layer.down_proj, shared_act)
        self._wait_for_routed_stage(
            milestones,
            milestones.routed_combine_start,
            "routed_combine_start",
        )
        return self.part2(submitted_gate_up.hidden_states, down_input)

    def _finish_shared_mlp(
        self,
        submitted_gate_up: SubmittedSharedExpertGateUp,
        milestones: RoutedMoEMilestones,
    ) -> torch.Tensor:
        if submitted_gate_up.path is SharedExpertMLPPath.A8_INT_FUSED:
            return self._run_a8_int_mlp(submitted_gate_up, milestones)
        if submitted_gate_up.path is SharedExpertMLPPath.A8_MXFP_FUSED:
            return self._run_a8_mxfp_mlp(submitted_gate_up, milestones)
        return self._finish_linear_wrapped_mlp(submitted_gate_up, milestones)

    def _select_mlp_path(self) -> SharedExpertMLPPath:
        """Select a path without changing the quantization scheme's math.

        Only schemes with a proven split activation-quant pipeline bypass the
        registered linear wrappers. W8A8MXFP keeps both linears and selects
        a path that checks activation-fusion compatibility after Gate-Up.
        W8A8FP, W4A4MXFP and other schemes use the generic wrapper path.
        Active LoRA always uses that generic path to preserve its adapters.
        """
        has_quantized_shared_without_lora = (
            not has_lora(self.lora_context)
            and hasattr(self.layer.gate_up_proj, "weight_scale")
            and hasattr(self.layer.down_proj, "weight_scale")
        )
        # W8A8 and W4A8 share an INT8-activation implementation; A8 names
        # the activation width and does not imply that weights are INT8.
        if has_quantized_shared_without_lora and self.quant_type in (QuantType.W8A8, QuantType.W4A8):
            return SharedExpertMLPPath.A8_INT_FUSED
        # This MXFP path currently represents W4A8MXFP only (MXFP4 weights
        # with MXFP8 activations).
        if has_quantized_shared_without_lora and self.quant_type == QuantType.W4A8MXFP:
            return SharedExpertMLPPath.A8_MXFP_FUSED
        # Selecting this path does not guarantee fusion: tensor metadata and
        # the Down scheme are checked after the registered Gate-Up linear.
        if not has_lora(self.lora_context) and self.quant_type == QuantType.W8A8MXFP:
            return SharedExpertMLPPath.W8A8_MXFP_LINEAR
        return SharedExpertMLPPath.LINEAR_WRAPPER

    def wait_for_output(self) -> None:
        """Join a deferred SP shared-output collective on the current stream."""
        # This join is intentionally stream-based. The shared forward executes
        # inside an opaque custom op for ACL Graph, so a Python-side Event
        # assigned inside that op is not available while its fake path traces.
        torch.npu.current_stream().wait_stream(shared_experts_calculation_stream())

    def forward(
        self,
        prepared_input: PreparedSharedExpertInput,
        milestones: RoutedMoEMilestones,
        defer_output_wait: bool = False,
        submitted_gate_up: SubmittedSharedExpertGateUp | None = None,
    ) -> torch.Tensor:
        mode = self.parallel_mode()
        if submitted_gate_up is None:
            with npu_stream_switch(shared_experts_calculation_stream(), enabled=self.multistream_overlap):
                hidden_states, local_dp_metadata = self._prepare_execution_input(
                    prepared_input,
                    milestones,
                    mode,
                )
                gate_up_input = self._prepare_gate_up_input(hidden_states)
            prepared_execution = PreparedSharedExpertExecution(
                hidden_states=hidden_states,
                gate_up_input=gate_up_input,
                local_dp_metadata=local_dp_metadata,
            )
            submitted_gate_up = self.enqueue_gate_up_after_router(
                prepared_execution,
                milestones.router_output_ready,
            )

        local_dp_metadata = submitted_gate_up.local_dp_metadata
        with npu_stream_switch(shared_experts_calculation_stream(), enabled=self.multistream_overlap):
            shared_out = self._finish_shared_mlp(submitted_gate_up, milestones)
            if self.multistream_overlap and mode is SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY:
                self._wait_for_milestone(
                    milestones.routed_finalize_done,
                    "routed_finalize_done",
                )
                shared_out = self._pad_and_reduce_scatter(shared_out)

        if self.multistream_overlap and (
            mode is not SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY or not defer_output_wait
        ):
            self.wait_for_output()

        if mode is SharedExpertParallelMode.SHARED_EXPERT_DATA_PARALLEL_ONLY:
            assert local_dp_metadata is not None
            shared_out = self._finalize_local_dp_output(shared_out, local_dp_metadata)
        elif mode is SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY and not self.multistream_overlap:
            shared_out = self._pad_and_reduce_scatter(shared_out)
        return shared_out
