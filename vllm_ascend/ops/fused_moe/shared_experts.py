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
from types import ModuleType
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
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, is_draft_vllm_config, use_cann_megamoe
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.distributed.parallel_state import get_shared_expert_tp_group
from vllm_ascend.lora.fused_moe import has_lora
from vllm_ascend.ops.activation import AscendSiluAndMul
from vllm_ascend.ops.fused_moe import moe_utils
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MegamoeSharedExpertFusion, MegamoeSharedWeights
from vllm_ascend.ops.fused_moe.dataclass.shared_experts import (
    PreparedSharedExpertInput,
    RoutedMoEMilestones,
)
from vllm_ascend.ops.fused_moe.moe_utils import _pad_tokens_with_cat
from vllm_ascend.ops.linear_op import SharedExpertColumnParallelOp, SharedExpertRowParallelOp
from vllm_ascend.quantization.prepared_linear_input import PreparedLinearInput
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import (
    ACL_FORMAT_FRACTAL_ND,
    ACL_FORMAT_FRACTAL_NZ,
    is_internal_npu_format,
    npu_stream_switch,
    shared_expert_tp_enable,
    shared_experts_calculation_stream,
)

# CANN uses 36 to select FP8 E4M3FN output for situ_mx_quant.
SITU_MX_DST_TYPE_E4M3FN = 36
MXFP8_ACTIVATION_GROUP_SIZE = 32
# SwiGLU halves the input width; MX scales are stored in pairs of groups.
SWIGLU_MX_INPUT_ALIGNMENT = 4 * MXFP8_ACTIVATION_GROUP_SIZE
SWIGLU_MX_QUANT_MODE = 2

# CANN arch22 shared-expert tiling limits.
_A3_MIN_HIDDEN_DIM = 1024
_A3_MAX_HIDDEN_DIM = 8192
_A3_HIDDEN_ALIGNMENT = 512
_A3_MIN_SHARED_INTERMEDIATE = 512
_A3_INTERMEDIATE_ALIGNMENT = 256


@dataclass(frozen=True, slots=True)
class _MegamoeSharedFormat:
    carrier: torch.dtype
    acl_format: int
    mxfp: bool = False
    packed_fp4: bool = False


_INT8_FORMAT = _MegamoeSharedFormat(torch.int8, ACL_FORMAT_FRACTAL_NZ)
_MXFP8_FORMAT = _MegamoeSharedFormat(torch.float8_e4m3fn, ACL_FORMAT_FRACTAL_ND, mxfp=True)
_MXFP4_FORMAT = _MegamoeSharedFormat(torch.uint8, ACL_FORMAT_FRACTAL_NZ, mxfp=True, packed_fp4=True)
_A3_SHARED_FORMATS = {
    (QuantType.NONE, QuantType.NONE): _MegamoeSharedFormat(torch.bfloat16, ACL_FORMAT_FRACTAL_ND),
    (QuantType.W8A8, QuantType.W8A8): _INT8_FORMAT,
    (QuantType.W4A8, QuantType.W8A8): _INT8_FORMAT,
}
_A5_SHARED_FORMATS = {
    (routed, shared): fmt
    for routed in (QuantType.W8A8MXFP, QuantType.W4A8MXFP, QuantType.W4A4MXFP)
    for shared, fmt in (
        (QuantType.W8A8MXFP, _MXFP8_FORMAT),
        (QuantType.W4A8MXFP, _MXFP4_FORMAT),
        (QuantType.W4A4MXFP, _MXFP4_FORMAT),
    )
}
_SHARED_CARRIER_QUANT_TYPES = {
    (torch.bfloat16, False): QuantType.NONE,
    (torch.int8, True): QuantType.W8A8,
    (torch.uint8, True): QuantType.W4A4MXFP,
    (torch.float8_e4m3fn, True): QuantType.W8A8MXFP,
}


@dataclass(frozen=True, slots=True)
class _SharedProjection:
    weight: torch.Tensor
    scale: torch.Tensor | None
    input_size: int
    output_size: int


def _get_tensor_acl_format(tensor: torch.Tensor) -> int | None:
    return int(torch_npu.get_npu_format(tensor)) if tensor.device.type == "npu" else None


def _tensor_layout(tensor: torch.Tensor) -> tuple[torch.Size, torch.dtype, torch.device, int | None]:
    return tensor.shape, tensor.dtype, tensor.device, _get_tensor_acl_format(tensor)


def _set_allow_internal_format(enabled: bool) -> None:
    # torch_npu's option can be write-only on torch.npu.config.
    config = torch.npu.config
    if isinstance(config, ModuleType):
        torch_npu._C._npu_setOption({"ALLOW_INTERNAL_FORMAT": "enable" if enabled else "disable"})
    else:
        config.allow_internal_format = enabled


@contextmanager
def _without_internal_format(device: torch.device):
    if device.type != "npu":
        yield
        return
    option = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    try:
        _set_allow_internal_format(False)
        yield
    finally:
        _set_allow_internal_format(option in (None, b"enable", "enable"))


def _resolve_shared_linear_quant_type(linear: torch.nn.Module) -> QuantType | None:
    # Static activation quantization is incompatible with MegaMoe's A8 path.
    if hasattr(linear, "input_scale") or hasattr(linear, "aclnn_input_scale"):
        return None
    method = getattr(linear, "quant_method", None)
    scheme = getattr(method, "quant_method", method)
    declared = getattr(scheme, "quant_type", None)
    if isinstance(declared, QuantType) and declared is not QuantType.NONE:
        return declared
    return _SHARED_CARRIER_QUANT_TYPES.get((linear.weight.dtype, hasattr(linear, "weight_scale")))


def _pack_shared_scale(scale: torch.Tensor) -> torch.Tensor:
    # CANN reads the low 32 bits of each UINT64 slot as float32.
    return scale.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF


class SharedExpertParallelMode(Enum):
    """Effective activation and weight layout for a shared-expert forward."""

    TENSOR_PARALLEL = auto()  # Full activations, TP-sharded weights.
    SHARED_EXPERT_DATA_PARALLEL_ONLY = auto()  # Full activations, replicated weights (DP only).
    SEQUENCE_PARALLEL_ONLY = auto()  # Sharded activations, TP-sharded weights (SP only).
    SEQUENCE_PARALLEL_SEDP = auto()  # Sharded activations, replicated weights (SP + DP).
    FINE_GRAINED_TENSOR_PARALLEL = auto()  # DP-local tokens, DP-axis weight shards.


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
        *,
        allow_megamoe_fusion: bool = True,
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
        self.finegrained_tp_group = None
        if shared_expert_tp_enable():
            if not isinstance(
                getattr(layer.gate_up_proj, "custom_op", None), SharedExpertColumnParallelOp
            ) or not isinstance(getattr(layer.down_proj, "custom_op", None), SharedExpertRowParallelOp):
                raise ValueError("Shared-expert fine-grained TP requires sharded gate_up_proj and down_proj linears.")
            if layer.gate_up_proj.gather_output or not layer.down_proj.input_is_parallel:
                raise ValueError(
                    "Shared-expert fine-grained TP requires local gate/up output and partitioned down input."
                )
            self.finegrained_tp_group = get_shared_expert_tp_group()

        self._megamoe_formats: dict[tuple[QuantType, QuantType], _MegamoeSharedFormat] | None = None
        self._megamoe_shared_fusion = MegamoeSharedExpertFusion()
        self._megamoe_weight_aliases = (False, False)
        self._megamoe_projection_layouts: tuple[tuple, tuple] | None = None

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

        if allow_megamoe_fusion:
            self._setup_megamoe_shared_fusion()

    def _setup_megamoe_shared_fusion(self) -> None:
        """Fuse the shared expert into the MegaMoe operator when supported.

        When active, each shared linear's ``process_weights_after_loading`` is
        wrapped to rebuild the weight into the MegaMoe layout; the rebuilt
        payload is exposed through ``megamoe_shared_fusion()`` and installed on
        the routed-expert layer by ``AscendMoERunner``, then forwarded to the
        operator as ``shared_l1_weights``/``shared_l2_weights``. The runner
        then skips the separate shared-expert forward whenever the step
        dispatches to MegaMoe. Any unsupported configuration keeps the
        standalone shared-expert execution unchanged.
        """
        # MegaMoe's fused shared ABI does not own the DP-axis AG/RS.
        if getattr(self, "finegrained_tp_group", None) is not None:
            return
        vllm_config = get_current_vllm_config()
        if is_draft_vllm_config(vllm_config) or not use_cann_megamoe(vllm_config):
            return
        if not get_ascend_config().megamoe_shared_expert_fusion:
            return
        profile = get_current_hardware_profile()
        if not profile.supports(HardwareCapability.CANN_MEGAMOE_SHARED_EXPERT_FUSION):
            return
        mxfp = profile.supports(HardwareCapability.CANN_MEGAMOE_MXFP)
        formats = _A5_SHARED_FORMATS if mxfp else _A3_SHARED_FORMATS
        if not any(routed == self.quant_type for routed, _ in formats):
            return
        if self.quant_type == QuantType.W4A8 and not moe_utils.supports_cann_mega_moe_mixed_int_shared():
            self._disable_megamoe_shared_fusion(
                "the CANN MegaMoe interface cannot describe W4A8 routed and W8A8 shared experts independently"
            )
            return
        if self.in_dtype is not None and self.in_dtype != torch.bfloat16:
            return
        offload = getattr(vllm_config, "offload_config", None)
        if offload is not None and (offload.uva.cpu_offload_gb > 0 or offload.prefetch.offload_group_size > 0):
            return
        if not mxfp and (
            not _A3_MIN_HIDDEN_DIM <= self.hidden_size <= _A3_MAX_HIDDEN_DIM or self.hidden_size % _A3_HIDDEN_ALIGNMENT
        ):
            return
        gate_up, down = self.layer.gate_up_proj, self.layer.down_proj
        # The operator has one activation and supports neither a shared bias
        # nor an input-dependent shared output gate.
        if getattr(self.layer, "expert_gate", None) is not None or gate_up.bias is not None or down.bias is not None:
            return
        shared_activation = self._shared_megamoe_activation_signature()
        if shared_activation is None or shared_activation != self._routed_megamoe_activation_signature():
            self._disable_megamoe_shared_fusion("shared and routed activations differ")
            return
        if (
            self.moe_config.tp_group.world_size > 1
            and not self.weights_replicated
            and (mxfp or getattr(gate_up, "gather_output", False) or getattr(down, "reduce_results", False))
        ):
            self._disable_megamoe_shared_fusion("unsupported sharded shared-expert layout")
            return
        self._megamoe_formats = formats

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

    @property
    def megamoe_shared_weights_ready(self) -> bool:
        return self._megamoe_shared_fusion.weights is not None

    def megamoe_shared_fusion(self) -> MegamoeSharedExpertFusion:
        """Return the live handle installed on the routed-expert layer."""
        return self._megamoe_shared_fusion

    def _disable_megamoe_shared_fusion(self, reason: str) -> None:
        if self.megamoe_shared_weights_ready:
            raise RuntimeError(f"Incompatible MegaMoe shared-expert weight reload: {reason}")
        self._megamoe_formats = None
        logger.info_once("Using standalone shared experts: %s.", reason)

    def _agree_megamoe_layout(self, signature: tuple, reason: str | None) -> str | None:
        if self.megamoe_shared_weights_ready:
            group = None if self.weights_replicated else self.moe_config.tp_group
        else:
            # Admission can change the combine algorithm across the MC2 group.
            group = getattr(self.moe_config, "mc2_group", None) or self.moe_config.tp_group
        if group is None or group.world_size <= 1:
            return reason
        layouts: list[Any] = [None] * group.world_size
        torch.distributed.all_gather_object(layouts, (signature, reason), group=group.cpu_group)
        for _, peer_reason in layouts:
            if peer_reason:
                return peer_reason
        if any(peer_signature != signature for peer_signature, _ in layouts):
            return "parallel ranks disagree on the shared-expert layout"
        return None

    @torch.inference_mode()
    def refresh_megamoe_shared_weights(self) -> None:
        """Build or refill the MegaMoe shared-expert weight payload in place."""
        formats = self._megamoe_formats
        if formats is None:
            return

        gate_up, down = self.layer.gate_up_proj, self.layer.down_proj
        device = gate_up.weight.device
        current = self._megamoe_shared_fusion.weights
        if current is not None and device.type == "npu":
            # Reload runs while inference is paused. Drain outstanding readers
            # before accessing or refilling the captured storage.
            torch.npu.synchronize(device)

        reason = None
        layout: tuple[Any, ...] = ()
        w1 = w2 = None
        fmt = None
        shared_quant = None
        shards = 1
        try:
            layouts = self._megamoe_projection_layouts
            if layouts is not None and layouts != (_tensor_layout(gate_up.weight), _tensor_layout(down.weight)):
                raise ValueError("registered shared projection layout changed after fusion was armed")
            shared_quant = _resolve_shared_linear_quant_type(gate_up)
            if shared_quant is None or shared_quant != _resolve_shared_linear_quant_type(down):
                raise ValueError("unsupported or mismatched shared-expert quantization")
            fmt = formats.get((self.quant_type, shared_quant))
            if fmt is None:
                raise ValueError(f"unsupported MegaMoe format for {self.quant_type.name}/{shared_quant.name}")

            shards = 1 if self.weights_replicated else self.moe_config.tp_group.world_size
            with _without_internal_format(device):
                w1 = self._read_shared_projection(gate_up, fmt, shards)
                w2 = self._read_shared_projection(down, fmt, shards)
            if w2.weight.device != device:
                raise ValueError("shared projections must be on the same device")
            if (
                w1.input_size != self.hidden_size
                or w2.output_size != self.hidden_size
                or w1.output_size != 2 * w2.input_size
            ):
                raise ValueError("shared projection dimensions do not match")
            intermediate = w2.input_size * shards
            if fmt.mxfp:
                if intermediate != self.moe_config.intermediate_size_per_partition:
                    raise ValueError("shared and routed MXFP intermediate dimensions differ")
            elif intermediate < _A3_MIN_SHARED_INTERMEDIATE or intermediate % _A3_INTERMEDIATE_ALIGNMENT:
                raise ValueError(f"unsupported A3 shared-expert intermediate size {intermediate}")
            gate_up_parts = getattr(gate_up, "output_sizes", None)
            if (
                shards > 1
                and gate_up_parts is not None
                and (len(gate_up_parts) != 2 or gate_up_parts[0] != gate_up_parts[1])
            ):
                raise ValueError("TP-sharded Gate-Up requires equal gate and up partitions")
            layout = (
                shared_quant,
                shards,
                w1.weight.device.type,
                w1.input_size,
                w1.output_size,
                w2.input_size,
                w2.output_size,
            )
        except ValueError as exc:
            reason = str(exc)

        if reason := self._agree_megamoe_layout(layout, reason):
            self._disable_megamoe_shared_fusion(reason)
            return

        assert w1 is not None and w2 is not None and fmt is not None and shared_quant is not None
        scale1, scale2 = w1.scale, w2.scale
        quant_out_dtype = weight_type = None
        with _without_internal_format(device):
            reason = None
            try:
                if shards > 1 and scale1 is not None:
                    assert scale2 is not None
                    scales = self.moe_config.tp_group.all_gather(torch.cat((scale1, scale2)), dim=0).view(shards, -1)
                    gate_scales, down_scales = scales.split((w1.output_size, w2.output_size), dim=1)
                    # Down is sharded along K. MegaMoe applies one scale to the
                    # accumulated output, so every TP shard must use that scale.
                    if not torch.equal(down_scales, down_scales[:1].expand_as(down_scales)):
                        raise ValueError("TP shards disagree on down_proj weight_scale")
                    scale1 = self._merge_gate_up_shards(gate_scales.flatten(), shards)
                    scale2 = down_scales[0]
                _, quant_out_dtype, weight_type = moe_utils._get_cann_mega_moe_quant_settings(shared_quant)
            except ValueError as exc:
                reason = str(exc)

            if reason := self._agree_megamoe_layout((), reason):
                self._disable_megamoe_shared_fusion(reason)
                return

            candidate = self._assemble_megamoe_weights(
                w1,
                w2,
                fmt,
                shards,
                gathered_scales=(scale1, scale2),
            )
            if current is not None:
                try:
                    self._check_megamoe_refill(candidate, quant_out_dtype, weight_type)
                except ValueError as exc:
                    self._disable_megamoe_shared_fusion(str(exc))
                    return
                # Copy E8M0 scales through their byte carrier. The kernel's
                # typed view remains bound to the same allocation.
                pointers = tuple(tensor.data_ptr() for tensor in current.operands())
                for old, new in zip(current.operands(), candidate.operands(), strict=True):
                    if old.data_ptr() == new.data_ptr():
                        continue
                    if old.dtype == torch.float8_e8m0fnu:
                        old.view(torch.uint8).copy_(new.view(torch.uint8))
                    else:
                        old.copy_(new)
                assert pointers == tuple(tensor.data_ptr() for tensor in current.operands())
            else:
                fusion = self._megamoe_shared_fusion
                fusion.quant_out_dtype = quant_out_dtype
                fusion.weight_type = weight_type
                fusion.weights = candidate
                self._megamoe_weight_aliases = (
                    w1.weight.data_ptr() == candidate.w1[0].data_ptr(),
                    w2.weight.data_ptr() == candidate.w2[0].data_ptr(),
                )
                self._megamoe_projection_layouts = (
                    _tensor_layout(gate_up.weight),
                    _tensor_layout(down.weight),
                )
                logger.info_once("Armed MegaMoe shared experts (routed=%s, shared=%s).", self.quant_type, shared_quant)

    @staticmethod
    def _read_shared_projection(linear: torch.nn.Module, fmt: _MegamoeSharedFormat, shards: int) -> _SharedProjection:
        if getattr(linear, "tp_size", shards) != shards:
            raise ValueError("shared projection TP size does not match the configured layout")
        padding = getattr(linear, "mxfp8_tp_padding", getattr(linear, "mxfp4_tp_padding", (0, 0)))
        if any(padding):
            raise ValueError(f"shared projection has unsupported TP padding {padding}")

        weight = linear.weight.detach()
        if weight.ndim != 2 or weight.dtype != fmt.carrier or weight.device.type == "meta":
            raise ValueError(f"expected a committed 2D {fmt.carrier} shared weight, got {weight.shape} {weight.dtype}")
        if fmt.carrier == torch.bfloat16:
            # Dense linears store [out, in]; normalize quantized and dense
            # linears to the same logical [in, out] view for TP assembly.
            weight = AscendSharedExperts._as_nd(weight).T
        input_size, output_size = weight.shape
        if fmt.packed_fp4:
            input_size *= 2

        scale = None
        if fmt.carrier != torch.bfloat16:
            scale = getattr(linear, "weight_scale", None)
            if scale is None or scale.device != weight.device:
                raise ValueError("shared projection lacks a committed weight_scale on the weight device")
            scale = scale.detach()
        if fmt.carrier == torch.int8:
            assert scale is not None
            offset = getattr(linear, "weight_offset", None)
            if offset is not None and bool((offset != 0).any()):
                raise ValueError("shared INT8 weight_offset must be zero")
            if scale.numel() != output_size:
                raise ValueError("shared INT8 weight_scale must have one value per output channel")
            scale = scale.reshape(-1).contiguous()
            if scale.dtype == torch.int64:
                scale = scale.to(torch.int32).view(torch.float32)
            elif scale.dtype in (torch.bfloat16, torch.float16, torch.float32):
                scale = scale.float()
            else:
                raise ValueError(f"unsupported shared INT8 weight_scale dtype {scale.dtype}")
            if not bool((torch.isfinite(scale) & (scale > 0)).all()):
                raise ValueError("shared INT8 weight_scale must be finite and positive")
        elif fmt.mxfp:
            assert scale is not None
            group_width = 2 * MXFP8_ACTIVATION_GROUP_SIZE
            scale_groups = (input_size + group_width - 1) // group_width
            if scale.shape != (scale_groups, output_size, 2) or scale.dtype not in (torch.uint8, torch.float8_e8m0fnu):
                raise ValueError(f"unsupported shared MXFP weight_scale layout {scale.shape} {scale.dtype}")
        return _SharedProjection(weight, scale, input_size, output_size)

    def _assemble_megamoe_weights(
        self,
        gate_up: _SharedProjection,
        down: _SharedProjection,
        fmt: _MegamoeSharedFormat,
        shards: int,
        *,
        gathered_scales: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
    ) -> MegamoeSharedWeights:
        weight1, weight2 = gate_up.weight, down.weight
        if gathered_scales is not None:
            scale1, scale2 = gathered_scales
        else:
            scale1, scale2 = gate_up.scale, down.scale
        if shards > 1:
            tp_group = self.moe_config.tp_group
            if gathered_scales is None and scale1 is not None:
                assert scale2 is not None
                scales = tp_group.all_gather(torch.cat((scale1, scale2)), dim=0).view(shards, -1)
                gate_scales, down_scales = scales.split((gate_up.output_size, down.output_size), dim=1)
                # Down is sharded along K. MegaMoe applies one scale to the
                # accumulated output, so every TP shard must use that scale.
                if not torch.equal(down_scales, down_scales[:1].expand_as(down_scales)):
                    raise ValueError("TP shards disagree on down_proj weight_scale")
                scale1 = self._merge_gate_up_shards(gate_scales.flatten(), shards)
                scale2 = down_scales[0]
            gate_rows = self._as_nd(weight1).T.contiguous()
            weight1 = self._merge_gate_up_shards(tp_group.all_gather(gate_rows, dim=0), shards).T
            weight2 = tp_group.all_gather(self._as_nd(weight2).contiguous(), dim=0)
        if fmt.carrier == torch.int8:
            assert scale1 is not None and scale2 is not None
            scale1, scale2 = _pack_shared_scale(scale1), _pack_shared_scale(scale2)
        elif fmt.mxfp:
            assert scale1 is not None and scale2 is not None
            scale1 = scale1.view(torch.uint8).transpose(0, 1).contiguous().view(torch.float8_e8m0fnu)
            scale2 = scale2.view(torch.uint8).transpose(0, 1).contiguous().view(torch.float8_e8m0fnu)
        weight1 = self._format_megamoe_weight(weight1, fmt, is_gate_up=True)
        weight2 = self._format_megamoe_weight(weight2, fmt, is_gate_up=False)
        return MegamoeSharedWeights(
            w1=[weight1],
            w2=[weight2],
            w1_scale=[scale1] if scale1 is not None else None,
            w2_scale=[scale2] if scale2 is not None else None,
        )

    @staticmethod
    def _merge_gate_up_shards(gathered: torch.Tensor, shards: int) -> torch.Tensor:
        # [gate_0, up_0, gate_1, up_1, ...] -> [all gates, all ups].
        return gathered.unflatten(0, (shards, 2, -1)).transpose(0, 1).flatten(0, 2)

    @staticmethod
    def _as_nd(weight: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_ND) if is_internal_npu_format(weight) else weight

    def _format_megamoe_weight(
        self, weight: torch.Tensor, fmt: _MegamoeSharedFormat, *, is_gate_up: bool
    ) -> torch.Tensor:
        if fmt.carrier == torch.int8 and _get_tensor_acl_format(weight) == fmt.acl_format:
            # A3 replicated/TP1: alias the committed registered NZ weight.
            return weight
        if fmt.mxfp:
            # A5 consumes [out, in]. Some packed linear schemes transpose
            # their internal NZ storage by view; undo that view before casting.
            weight = self._as_nd(weight.T) if fmt.packed_fp4 else self._as_nd(weight).T
        else:
            weight = self._as_nd(weight)
        weight = weight.contiguous()
        if weight.device.type != "npu" or fmt.acl_format == ACL_FORMAT_FRACTAL_ND:
            return weight
        _set_allow_internal_format(True)
        try:
            if fmt.packed_fp4 and not is_gate_up:
                return torch_npu.npu_format_cast(
                    weight, fmt.acl_format, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
                )
            return torch_npu.npu_format_cast(weight, fmt.acl_format)
        finally:
            _set_allow_internal_format(False)

    def _check_megamoe_refill(
        self, candidate: MegamoeSharedWeights, quant_out_dtype: int | None, weight_type: int | None
    ) -> None:
        fusion = self._megamoe_shared_fusion
        assert fusion.weights is not None
        old_operands, new_operands = fusion.weights.operands(), candidate.operands()
        if (
            quant_out_dtype != fusion.quant_out_dtype
            or weight_type != fusion.weight_type
            or len(old_operands) != len(new_operands)
        ):
            raise ValueError("shared-expert quantization changed")
        for name, old, new in zip(("w1", "w2", "w1_scale", "w2_scale"), old_operands, new_operands):
            old_layout, new_layout = _tensor_layout(old), _tensor_layout(new)
            if old_layout != new_layout:
                raise ValueError(f"{name} layout changed from {old_layout} to {new_layout}")
        for alias, old, new in zip(self._megamoe_weight_aliases, old_operands[:2], new_operands[:2], strict=True):
            if alias and old.data_ptr() != new.data_ptr():
                raise ValueError("registered shared weight storage changed after fusion was armed")

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
        if getattr(self, "finegrained_tp_group", None) is not None:
            return SharedExpertParallelMode.FINE_GRAINED_TENSOR_PARALLEL
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
        finegrained_tp_group = getattr(self, "finegrained_tp_group", None)
        if finegrained_tp_group is not None:
            # These are different DP requests, not SP slices of num_tokens.
            return finegrained_tp_group.all_gather(hidden_states, dim=0)
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
        finegrained_tp_group = getattr(self, "finegrained_tp_group", None)
        if finegrained_tp_group is not None:
            return finegrained_tp_group.reduce_scatter(shared_out, dim=0)
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
        if not (
            self.multistream_overlap
            and self.parallel_mode()
            in (SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY, SharedExpertParallelMode.FINE_GRAINED_TENSOR_PARALLEL)
        ):
            return PreparedSharedExpertInput(hidden_states=hidden_states)

        input_ready = torch.npu.current_stream().record_event()
        with npu_stream_switch(shared_experts_calculation_stream(), enabled=True):
            torch.npu.current_stream().wait_event(input_ready)
            # Keep TP padding across the custom-op boundary; the shared MLP
            # trims it before computation.
            if self.parallel_mode() is SharedExpertParallelMode.FINE_GRAINED_TENSOR_PARALLEL:
                hidden_states = self._gather_sp_input(hidden_states)
            else:
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
        if not (
            self.multistream_overlap
            and self.parallel_mode()
            in (SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY, SharedExpertParallelMode.FINE_GRAINED_TENSOR_PARALLEL)
        ):
            return PreparedSharedExpertInput(hidden_states=hidden_states)

        # Keep TP padding across the custom-op boundary; the shared MLP trims
        # it before computation. The caller records shared_input_ready after
        # this returns, handing the gathered tensor to the auxiliary stream.
        hidden_states = (
            self._gather_sp_input(hidden_states)
            if self.parallel_mode() is SharedExpertParallelMode.FINE_GRAINED_TENSOR_PARALLEL
            else tensor_model_parallel_all_gather(hidden_states, dim=0)
        )
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
        elif (
            mode
            in (SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY, SharedExpertParallelMode.FINE_GRAINED_TENSOR_PARALLEL)
            and not prepared_input.is_gathered
        ):
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
            if self.multistream_overlap and mode in (
                SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY,
                SharedExpertParallelMode.FINE_GRAINED_TENSOR_PARALLEL,
            ):
                self._wait_for_milestone(
                    milestones.routed_finalize_done,
                    "routed_finalize_done",
                )
                shared_out = self._pad_and_reduce_scatter(shared_out)

        if self.multistream_overlap and (
            mode
            not in (
                SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY,
                SharedExpertParallelMode.FINE_GRAINED_TENSOR_PARALLEL,
            )
            or not defer_output_wait
        ):
            self.wait_for_output()

        if mode is SharedExpertParallelMode.SHARED_EXPERT_DATA_PARALLEL_ONLY:
            assert local_dp_metadata is not None
            shared_out = self._finalize_local_dp_output(shared_out, local_dp_metadata)
        elif (
            mode
            in (SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY, SharedExpertParallelMode.FINE_GRAINED_TENSOR_PARALLEL)
            and not self.multistream_overlap
        ):
            shared_out = self._pad_and_reduce_scatter(shared_out)
        return shared_out
