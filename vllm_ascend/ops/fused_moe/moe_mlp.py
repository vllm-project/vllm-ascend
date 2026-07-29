# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.


import torch
import torch_npu
from torch.nn.functional import pad
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.device.mxfp_compat import (
    FLOAT8_E8M0FNU_DTYPE,
    QUANT_DTYPES,
    ensure_mxfp8_moe_available,
)
from vllm_ascend.ops.activation import AscendSwigluOAIAndMul, AscendSwigluStepAndMul
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEMlpComputeInput,
    MoEQuantParams,
    MoEWeights,
)
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import (
    dispose_tensor,
    enable_custom_op,
    get_ascend_device_type,
)

ASCEND_DEVICE_TYPE = get_ascend_device_type()


def _custom_gmm_swiglu_enabled(fusion, dynamic_eplb, activation=None):
    return (
        fusion
        and dynamic_eplb
        and getattr(activation, "value", activation) != "swigluoai_uninterleave"
        and enable_custom_op()
    )


def _gmm_swiglu_quant_fusion_enabled(use_mxfp_quant, fusion, dynamic_eplb, activation=None):
    return (use_mxfp_quant or (fusion and not dynamic_eplb)) and (
        getattr(activation, "value", activation) != "swigluoai_uninterleave"
    )


def cumsum_group_list(
    group_list: torch.Tensor, src_list_type: int, dst_list_type: int, active_num: int = 0, expert_num: int = 0
) -> torch.Tensor:
    if src_list_type not in [0, 1, 2]:
        raise ValueError(f"group_list_type should be in [0, 1, 2], but received {src_list_type}")

    if src_list_type == dst_list_type:
        return group_list
    if src_list_type == 1 and dst_list_type == 0:
        return group_list.cumsum(dim=0)
    if src_list_type == 0 and dst_list_type == 1:
        group_diff = torch.diff(group_list)
        new_group = torch.cat([group_list[0].unsqueeze(0), group_diff], dim=0)
        return new_group
    if src_list_type == 2 and dst_list_type == 0:
        experts = pad(group_list[:, 0], (1, 0))
        tokens = pad(group_list[:, 1].cumsum(dim=0), (1, 0))
        cumsum_group_list = torch.full(
            size=(expert_num,), fill_value=active_num, dtype=group_list.dtype, device=group_list.device
        )

        for i, (start, end) in enumerate(zip(experts[:-1], experts[1:])):
            if end > start:
                cumsum_group_list[start:end] = tokens[i]

        return cumsum_group_list
    raise NotImplementedError(
        f"Conversion from src_list_type={src_list_type} to dst_list_type={dst_list_type} is not implemented yet. "
        "This feature is under development."
    )


def _require_single_tensor_for_swiglu_quant(
    tensor_or_list: list[torch.Tensor] | torch.Tensor, *, name: str
) -> torch.Tensor:
    if isinstance(tensor_or_list, list):
        if len(tensor_or_list) != 1:
            raise ValueError(f"{name} must be a tensor or a single-element list, but got {len(tensor_or_list)}.")
        return tensor_or_list[0]
    return tensor_or_list


def _prepare_activation_quant(
    hidden_states: torch.Tensor,
    quant: MoEQuantParams,
    dynamic_scale: torch.Tensor | None,
    weights: MoEWeights,
) -> tuple[torch.Tensor, torch.Tensor | None, bool]:
    """Pre-quantize activations and produce the per-token scale for GMM1.

    Returns ``(hidden_states, pertoken_scale, dispose_after_gmm1)`` where the
    last element is True when the returned ``hidden_states`` were quantized
    upstream (``dynamic_scale`` provided) and may be freed once GMM1 consumes
    them. Weight-only schemes skip activation quantization entirely.
    """
    # Weight-only schemes keep activations in bf16/fp16.
    if quant.quant_type == QuantType.W4A16:
        return hidden_states, None, True
    if quant.quant_type == QuantType.W4A16MXFP:
        return hidden_states, None, False

    if quant.is_mxfp:
        ensure_mxfp8_moe_available("MXFP MoE MLP path")
        if weights.w1_scale_bias is not None or weights.w2_scale_bias is not None:
            raise NotImplementedError("MXFP path does not support scale_bias yet.")
        if weights.w1_offset is not None or weights.w2_offset is not None:
            raise NotImplementedError("MXFP path does not support antiquant offset yet.")

    if dynamic_scale is None:
        original = hidden_states
        hidden_states, pertoken_scale = DeviceOperator.npu_dynamic_quant(
            hidden_states=hidden_states,
            dynamic_scale=None,
            act_quant_type=quant.act_quant_type,
            use_mxfp_quant=quant.is_mxfp,
        )
        dispose_tensor(original)
        return hidden_states, pertoken_scale, False

    pertoken_scale = (
        DeviceOperator.maybe_normalize_mxfp_scale_layout(dynamic_scale) if quant.is_mxfp else dynamic_scale
    )
    return hidden_states, pertoken_scale, True


def _w2_scale_dtype(weights: MoEWeights) -> torch.dtype:
    """Output dtype derived from the w2 weight scale (common to all quant paths)."""
    w2s = weights.w2_scale
    return w2s[0].dtype if isinstance(w2s, list) else w2s.dtype


def _apply_activation_no_requant(
    hidden_states: torch.Tensor,
    activation: str | None,
    is_gelu: bool,
    swiglu_limit: float,
    is_swigluoai_uninterleave: bool,
    swiglu_alpha: float,
    swiglu_beta: float,
) -> torch.Tensor:
    """Activation for the antiquant path: no re-quantization follows."""
    if activation == MoEActivation.SWIGLUSTEP:
        return AscendSwigluStepAndMul.swiglustep_forward(hidden_states, limit=swiglu_limit or 7.0)
    if is_gelu:
        gate, up = hidden_states.chunk(2, dim=-1)
        approximate = "tanh" if activation == MoEActivation.GELU_TANH else "none"
        return torch.nn.functional.gelu(gate, approximate=approximate) * up
    if is_swigluoai_uninterleave:
        return torch_npu.npu_clipped_swiglu(
            hidden_states,
            interleaved=False,
            alpha=swiglu_alpha,
            limit=swiglu_limit,
            bias=swiglu_beta,
        )
    return torch_npu.npu_swiglu(hidden_states)


def _apply_w4a16(
    *,
    hidden_states: torch.Tensor,
    weights: MoEWeights,
    group_list: torch.Tensor,
    group_list_type: int,
    activation: str | None,
    is_gelu: bool,
    swiglu_limit: float,
    is_swigluoai_uninterleave: bool,
    swiglu_alpha: float,
    swiglu_beta: float,
) -> tuple[torch.Tensor, object]:
    """W4A16 antiquant path: antiquant GMM1 -> activation -> antiquant GMM2.

    W4A16 has no scale_bias (it uses antiquant offset, not scale_bias), so the
    output dtype follows the w2 antiquant scale dtype.
    """
    output_dtype = _w2_scale_dtype(weights)
    # gmm1: gate_up_proj (antiquant)
    gmm1_input = hidden_states
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[weights.w1],
        antiquant_scale=[weights.w1_scale],
        antiquant_offset=[weights.w1_offset],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
        output_dtype=output_dtype,
    )[0]
    dispose_tensor(gmm1_input)
    # act_fn: swiglu (no re-quantization for weight-only antiquant)
    hidden_states = _apply_activation_no_requant(
        hidden_states,
        activation,
        is_gelu,
        swiglu_limit,
        is_swigluoai_uninterleave,
        swiglu_alpha,
        swiglu_beta,
    )
    before_gmm2_evt = torch.npu.current_stream().record_event()
    # gmm2: down_proj (antiquant)
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[weights.w2],
        antiquant_scale=[weights.w2_scale],
        antiquant_offset=[weights.w2_offset],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
        output_dtype=output_dtype,
    )[0]
    return hidden_states, before_gmm2_evt


def _dequant_gmm1_scale(
    w1_scale: list[torch.Tensor] | torch.Tensor, w2_scale: list[torch.Tensor] | torch.Tensor
) -> list[torch.Tensor]:
    """Scale list for the dequant GMM1 call, cast to the w2 scale dtype."""
    if isinstance(w1_scale, list):
        return [w1_scale[0].to(w2_scale[0].dtype)]
    return [w1_scale]


def _gmm1_dequant_requant(
    *,
    x: torch.Tensor,
    weights: MoEWeights,
    pertoken_scale: torch.Tensor | None,
    bias1: torch.Tensor | None,
    group_list: torch.Tensor,
    group_list_type: int,
    activation: str | None,
    is_gelu: bool,
    is_swigluoai_uninterleave: bool,
    swiglu_alpha: float,
    swiglu_beta: float,
    swiglu_limit: float,
    output_dtype: torch.dtype,
    quant: MoEQuantParams,
    dispose_after_gmm1: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Unfused dequant path: dequant GMM1 -> activation -> re-quant.

    GELU and SWIGLUSTEP use a dequant GMM1 (scale + per-token scale) followed by
    the activation and a separate re-quant. silu / swigluoai_uninterleave use an
    int32 GMM1 followed by the fused ``npu_dequant_swiglu_quant`` (dequant +
    swiglu + quant in one op, with ``swiglu_mode`` for swigluoai_uninterleave).
    """
    if is_gelu or activation == MoEActivation.SWIGLUSTEP:
        gmm1_kwargs = {
            "x": [x],
            "weight": weights.w1 if isinstance(weights.w1, list) else [weights.w1],
            "scale": _dequant_gmm1_scale(weights.w1_scale, weights.w2_scale),
            "bias": bias1,
            "per_token_scale": [pertoken_scale],
            "split_item": 2,
            "group_type": 0,
            "group_list": group_list,
            "group_list_type": group_list_type,
            "output_dtype": output_dtype,
        }
        if quant.is_mxfp:
            gmm1_kwargs.update(
                {
                    "scale_dtype": FLOAT8_E8M0FNU_DTYPE,
                    "per_token_scale_dtype": FLOAT8_E8M0FNU_DTYPE,
                    "output_dtype": torch.bfloat16,
                }
            )
        # gmm1: gate_up_proj
        hidden_states = torch_npu.npu_grouped_matmul(**gmm1_kwargs)[0]
        if dispose_after_gmm1:
            dispose_tensor(x)
        # act_fn + re-quant
        if activation == MoEActivation.SWIGLUSTEP:
            # Step3.5/3.7: out = silu(gate).clamp(max=limit) * up.clamp(-limit, limit)
            hidden_states = AscendSwigluStepAndMul.swiglustep_forward(hidden_states, limit=swiglu_limit or 7.0)
            hidden_states, swiglu_out_scale = DeviceOperator.npu_dynamic_quant(
                hidden_states, act_quant_type=quant.act_quant_type, use_mxfp_quant=quant.is_mxfp
            )
        else:  # GELU
            gate, up = hidden_states.chunk(2, dim=-1)
            approximate = "tanh" if activation == MoEActivation.GELU_TANH else "none"
            hidden_states = torch.nn.functional.gelu(gate, approximate=approximate) * up
            hidden_states, swiglu_out_scale = torch_npu.npu_dynamic_quant(hidden_states)
        return hidden_states, swiglu_out_scale

    # swigluoai_uninterleave: int32 GMM1 -> fused dequant + swiglu + quant.
    assert is_swigluoai_uninterleave
    w1_scale = weights.w1_scale
    if w1_scale[0].dtype != torch.float32:
        w1_scale[0] = w1_scale[0].to(torch.float32)
    # gmm1: gate_up_proj (raw int32; dequant is fused into npu_dequant_swiglu_quant)
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[x],
        weight=weights.w1,
        split_item=3,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
        output_dtype=torch.int32,
    )[0]
    if dispose_after_gmm1:
        dispose_tensor(x)
    dequant_swiglu_kwargs = {
        "x": hidden_states,
        "weight_scale": w1_scale[0],
        "activation_scale": pertoken_scale,
        "bias": None,
        "quant_scale": None,
        "quant_offset": None,
        "group_index": cumsum_group_list(group_list, group_list_type, 1),
        "activate_left": True,
        "quant_mode": 1,
        "swiglu_mode": 1,
        "clamp_limit": swiglu_limit,
        "glu_alpha": swiglu_alpha,
        "glu_bias": swiglu_beta,
    }
    hidden_states, swiglu_out_scale = torch.ops._C_ascend.npu_dequant_swiglu_quant(**dequant_swiglu_kwargs)
    return hidden_states, swiglu_out_scale


def _apply_int_fp8(
    *,
    hidden_states: torch.Tensor,
    weights: MoEWeights,
    quant: MoEQuantParams,
    pertoken_scale: torch.Tensor | None,
    dispose_after_gmm1: bool,
    group_list: torch.Tensor,
    group_list_type: int,
    activation: str | None,
    fusion: bool,
    dynamic_eplb: bool,
    swiglu_limit: float,
    is_gelu: bool,
    is_swigluoai_uninterleave: bool,
    swiglu_alpha: float,
    swiglu_beta: float,
    input_hidden_dtype: torch.dtype,
) -> tuple[torch.Tensor, object]:
    """W4A8 / W8A8 / W8A8FP: per-channel fused | weight_nz fused | swiglu_quant fused | dequant.

    Only W4A8 takes the per-channel fused op (``grouped_matmul_swiglu_quant_v2``);
    otherwise all three share the same fusion/unfused dispatch. The unfused
    dequant path fuses dequant+swiglu+quant via ``npu_dequant_swiglu_quant`` for
    silu/swigluoai_uninterleave, and uses a dequant GMM1 + activation + re-quant
    for GELU/SWIGLUSTEP.

    W4A8 and W8A8 may carry a ``scale_bias`` that is passed as ``bias`` to GMM1/GMM2.
    When present it forces a bf16 output and a ``group_list_type`` 0 -> 1 conversion.
    W8A8FP never carries scale_bias.
    """
    bias1 = weights.w1_scale_bias
    bias2 = weights.w2_scale_bias
    output_dtype = _w2_scale_dtype(weights)
    if bias1 is not None:
        if group_list_type == 0:
            group_list = torch.cat([group_list[:1], torch.diff(group_list, dim=0)])
            group_list_type = 1
        output_dtype = torch.bfloat16
    x = hidden_states  # GMM1 input; freed once consumed
    if (
        quant.quant_type == QuantType.W4A8
        and enable_custom_op()
        and activation != MoEActivation.SWIGLUSTEP
        and not is_gelu
        and not is_swigluoai_uninterleave
    ):
        # fused GMM1 + swiglu + re-quant.
        hidden_states, swiglu_out_scale = torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2(
            x=x,
            weight=weights.w1,
            weight_scale=weights.w1_scale if isinstance(weights.w1_scale, list) else [weights.w1_scale],
            x_scale=pertoken_scale,
            group_list=group_list,
            weight_assist_matrix=bias1,
            dequant_mode=0,
            group_list_type=group_list_type,
            swiglu_limit=swiglu_limit,
        )
    elif _custom_gmm_swiglu_enabled(fusion, dynamic_eplb, activation) and not is_gelu:
        # fused GMM1 + swiglu + re-quant (weight-NZ tensor-list variant).
        hidden_states, swiglu_out_scale, _ = torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz_tensor_list(
            x=x,
            weight=weights.w1,
            weight_scale=weights.w1_scale,
            x_scale=pertoken_scale,
            group_list=cumsum_group_list(group_list, group_list_type, 0),
            bias=bias1,
            swiglu_limit=swiglu_limit,
        )
    elif (
        _gmm_swiglu_quant_fusion_enabled(False, fusion, dynamic_eplb, activation)
        and activation != MoEActivation.SWIGLUSTEP
        and not is_gelu
    ):
        hidden_states, swiglu_out_scale, _ = DeviceOperator.npu_grouped_matmul_swiglu_quant(
            x=x,
            weight=_require_single_tensor_for_swiglu_quant(weights.w1, name="w1"),
            group_list=cumsum_group_list(group_list, group_list_type, 0),
            weight_scale=_require_single_tensor_for_swiglu_quant(weights.w1_scale, name="w1_scale"),
            x_scale=pertoken_scale,
            bias=bias1,
            act_quant_type=quant.act_quant_type,
            weight_quant_type=quant.weight_quant_type,
            swiglu_limit=swiglu_limit,
        )
        if dispose_after_gmm1:
            dispose_tensor(x)
    else:
        hidden_states, swiglu_out_scale = _gmm1_dequant_requant(
            x=x,
            weights=weights,
            pertoken_scale=pertoken_scale,
            bias1=bias1,
            group_list=group_list,
            group_list_type=group_list_type,
            activation=activation,
            is_gelu=is_gelu,
            is_swigluoai_uninterleave=is_swigluoai_uninterleave,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            output_dtype=output_dtype,
            quant=quant,
            dispose_after_gmm1=dispose_after_gmm1,
        )
    before_gmm2_evt = torch.npu.current_stream().record_event()
    hidden_states = DeviceOperator.npu_grouped_matmul_gmm2(
        hidden_states=hidden_states,
        weight=weights.w2,
        weight_scale=weights.w2_scale,
        per_token_scale=swiglu_out_scale,
        group_list=group_list,
        group_list_type=group_list_type,
        input_dtype=input_hidden_dtype,
        act_quant_type=quant.act_quant_type,
        weight_quant_type=quant.weight_quant_type,
        scale_type=quant.scale_type,
        per_token_scale_type=quant.per_token_scale_type,
        use_bf16=quant.use_bf16(input_hidden_dtype),
        bias=bias2,
        fallback_output_dtype=output_dtype,
    )
    return hidden_states, before_gmm2_evt


def _apply_mxfp(
    *,
    hidden_states: torch.Tensor,
    weights: MoEWeights,
    quant: MoEQuantParams,
    pertoken_scale: torch.Tensor | None,
    dispose_after_gmm1: bool,
    group_list: torch.Tensor,
    group_list_type: int,
    activation: str | None,
    fusion: bool,
    dynamic_eplb: bool,
    swiglu_limit: float,
    is_gelu: bool,
    is_swigluoai_uninterleave: bool,
    swiglu_alpha: float,
    swiglu_beta: float,
    input_hidden_dtype: torch.dtype,
) -> tuple[torch.Tensor, object]:
    """MXFP (W8A8MXFP / W4A4MXFP / W4A8MXFP / W4A16MXFP): swiglu_quant fused | dequant.

    ``is_mxfp`` is True, so the swiglu_quant fusion is always enabled when the
    activation allows it. W4A16MXFP is weight-only: ``pertoken_scale`` is None
    and pre-quant is skipped upstream. No per-channel or weight_nz path (both
    require non-mxfp). The unfused dequant path fuses dequant+swiglu+quant via
    ``npu_dequant_swiglu_quant`` for silu/swigluoai_uninterleave, and uses a
    dequant GMM1 + activation + re-quant for GELU/SWIGLUSTEP.

    MXFP never carries scale_bias (validated in ``_prepare_activation_quant``),
    so bias1/bias2 are always None. The output dtype follows the w2 scale dtype
    and is only used by the dequant fallback (GMM2 computes its own).
    """
    output_dtype = _w2_scale_dtype(weights)
    x = hidden_states  # GMM1 input; freed once consumed
    if (
        _gmm_swiglu_quant_fusion_enabled(True, fusion, dynamic_eplb, activation)
        and activation != MoEActivation.SWIGLUSTEP
        and not is_gelu
    ):
        weight = _require_single_tensor_for_swiglu_quant(weights.w1, name="w1")
        weight_scale = _require_single_tensor_for_swiglu_quant(weights.w1_scale, name="w1_scale")
        gmm1_group_list = cumsum_group_list(group_list, group_list_type, 0)
        if quant.quant_type == QuantType.W4A8MXFP:
            # W4A8MXFP: antiquant matmul (MXFP4 weight, MXFP8 act) + swiglu+quant.
            hidden_states = torch_npu.npu_grouped_matmul(
                x=[x],
                weight=[weight],
                scale=None,
                antiquant_scale=[weight_scale],
                scale_dtype=None,
                per_token_scale=[pertoken_scale],
                per_token_scale_dtype=torch.float8_e8m0fnu,
                split_item=2,
                group_type=0,
                group_list=gmm1_group_list,
                x_dtype=torch.float8_e4m3fn,
                weight_dtype=torch_npu.float4_e2m1fn_x2,
                output_dtype=torch.bfloat16,
            )[0]
            # DSV4 needs swiglu_limit input.
            hidden_states, swiglu_out_scale, _ = torch.ops._C_ascend.npu_swiglu_group_quant(
                hidden_states,
                topk_weight=None,
                group_index=None,
                dst_type=torch.float8_e4m3fn,
                quant_mode=2,
                clamp_value=swiglu_limit,
            )
            swiglu_out_scale = DeviceOperator.maybe_normalize_mxfp_scale_layout(swiglu_out_scale)
        elif quant.quant_type == QuantType.W4A16MXFP:
            # W4A16MXFP: antiquant matmul (MXFP4 weight, bf16 act) + swiglu (no re-quant).
            hidden_states = torch_npu.npu_grouped_matmul(
                x=[x],
                weight=[weight],
                antiquant_scale=[weight_scale],
                group_list=gmm1_group_list,
                split_item=3,
                group_type=0,
                output_dtype=x.dtype,
            )[0]
            hidden_states = torch_npu.npu_swiglu(hidden_states)
            swiglu_out_scale = None
        else:  # W8A8MXFP, W4A4MXFP: fused GMM1 + swiglu + re-quant.
            hidden_states, swiglu_out_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
                x=x,
                weight=[weight],
                group_list=gmm1_group_list,
                weight_scale=[weight_scale],
                x_scale=pertoken_scale,
                dequant_mode=2,
                quant_mode=2,
                dequant_dtype=torch.float32,
                quant_dtype=quant.act_quant_type,
                x_dtype=quant.act_quant_type if quant.act_quant_type in QUANT_DTYPES else None,
                weight_dtype=quant.weight_quant_type if quant.weight_quant_type in QUANT_DTYPES else None,
                weight_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                x_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
            )
            swiglu_out_scale = DeviceOperator.maybe_normalize_mxfp_scale_layout(swiglu_out_scale)
        if dispose_after_gmm1:
            dispose_tensor(x)
    else:
        hidden_states, swiglu_out_scale = _gmm1_dequant_requant(
            x=x,
            weights=weights,
            pertoken_scale=pertoken_scale,
            bias1=None,
            group_list=group_list,
            group_list_type=group_list_type,
            activation=activation,
            is_gelu=is_gelu,
            is_swigluoai_uninterleave=is_swigluoai_uninterleave,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            output_dtype=output_dtype,
            quant=quant,
            dispose_after_gmm1=dispose_after_gmm1,
        )
    before_gmm2_evt = torch.npu.current_stream().record_event()
    # GMM2 (down_proj), explicit per quant type.
    w2 = weights.w2
    w2_scale = weights.w2_scale
    if isinstance(w2, list):
        if len(w2) != 1:
            raise ValueError(f"w2 must have a single tensor in MXFP path, but got {len(w2)}.")
        w2 = w2[0]
    if isinstance(w2_scale, list):
        if len(w2_scale) != 1:
            raise ValueError(f"w2_scale must have a single tensor in MXFP path, but got {len(w2_scale)}.")
        w2_scale = w2_scale[0]
    gmm2_weight = [w2]
    gmm2_scale = [w2_scale]
    gmm2_kwargs = DeviceOperator.get_quant_gmm2_kwargs(
        input_dtype=input_hidden_dtype,
        act_quant_type=quant.act_quant_type,
        weight_quant_type=quant.weight_quant_type,
        scale_type=quant.scale_type if quant.quant_type != QuantType.W4A8MXFP else None,
        per_token_scale_type=quant.per_token_scale_type,
        use_bf16=quant.use_bf16(input_hidden_dtype),
        use_mxfp_quant=True,
    )
    gmm2_output_dtype = gmm2_kwargs.pop("output_dtype")
    if quant.quant_type == QuantType.W4A16MXFP:
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=gmm2_weight,
            antiquant_scale=gmm2_scale,
            bias=None,
            split_item=3,
            group_type=0,
            group_list_type=group_list_type,
            group_list=group_list,
            output_dtype=gmm2_output_dtype,
        )[0]
    else:
        if quant.quant_type == QuantType.W4A8MXFP:
            gmm2_scale = None  # type: ignore[assignment]
            gmm2_kwargs.update({"antiquant_scale": [w2_scale]})
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=gmm2_weight,
            scale=gmm2_scale,
            bias=None,
            per_token_scale=[swiglu_out_scale],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=gmm2_output_dtype,
            **gmm2_kwargs,
        )[0]
    return hidden_states, before_gmm2_evt


def quant_apply_mlp(
    *,
    hidden_states: torch.Tensor,
    weights: MoEWeights,
    quant: MoEQuantParams,
    group_list: torch.Tensor,
    group_list_type: int = 1,
    dynamic_scale: torch.Tensor | None = None,
    activation: str | None = None,
    fusion: bool = False,
    dynamic_eplb: bool = False,
    swiglu_limit: float = 0.0,
    swiglu_alpha: float = 1.0,
    swiglu_beta: float = 0.0,
) -> tuple[torch.Tensor, object]:
    """Apply the quantized MoE MLP: GMM1 (gate_up) -> activation -> GMM2 (down).

    Dispatch is driven by ``quant.quant_type`` into per-family handlers.
    ``quant`` is the single source of truth for quantization config;
    weight tensors live on ``weights``.
    """
    input_hidden_dtype = hidden_states.dtype
    act_name = getattr(activation, "value", activation)
    # GELU can't use the fused SwiGLU+quant ops below; fall back to the
    # non-fused GMM -> GELU -> (re)quant -> GMM2 path for GELU activations.
    is_gelu = activation in (MoEActivation.GELU, MoEActivation.GELU_TANH)
    is_swigluoai_uninterleave = act_name == "swigluoai_uninterleave"

    hidden_states, pertoken_scale, dispose_after_gmm1 = _prepare_activation_quant(
        hidden_states, quant, dynamic_scale, weights
    )

    if quant.quant_type == QuantType.W4A16:
        return _apply_w4a16(
            hidden_states=hidden_states,
            weights=weights,
            group_list=group_list,
            group_list_type=group_list_type,
            activation=activation,
            is_gelu=is_gelu,
            swiglu_limit=swiglu_limit,
            is_swigluoai_uninterleave=is_swigluoai_uninterleave,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
        )

    full_quant_kwargs = dict(
        hidden_states=hidden_states,
        weights=weights,
        quant=quant,
        pertoken_scale=pertoken_scale,
        dispose_after_gmm1=dispose_after_gmm1,
        group_list=group_list,
        group_list_type=group_list_type,
        activation=activation,
        fusion=fusion,
        dynamic_eplb=dynamic_eplb,
        swiglu_limit=swiglu_limit,
        is_gelu=is_gelu,
        is_swigluoai_uninterleave=is_swigluoai_uninterleave,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        input_hidden_dtype=input_hidden_dtype,
    )
    if quant.quant_type in (QuantType.W4A8, QuantType.W8A8, QuantType.W8A8FP):
        return _apply_int_fp8(**full_quant_kwargs)
    # MXFP family: W8A8MXFP, W4A4MXFP, W4A8MXFP, W4A16MXFP
    return _apply_mxfp(**full_quant_kwargs)


def unquant_apply_mlp(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    group_list: torch.Tensor,
    w1_bias: torch.Tensor = None,
    w2_bias: torch.Tensor = None,
    activation: str | None = None,
    group_list_type: int = 1,
    topk_scales: torch.Tensor | None = None,
    need_trans: bool = True,
    swiglu_limit: float = 0.0,
    swiglu_alpha: float = 1.0,
    swiglu_beta: float = 0.0,
    lora_context=None,
    expanded_row_idx: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if need_trans:
        w1 = w1.transpose(1, 2)
        w2 = w2.transpose(1, 2)

    gate_up_out = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w1],
        bias=[w1_bias.to(dtype=torch.float32)] if w1_bias is not None else None,
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]

    # MoE LoRA: only attempt injection when an adapter wraps this layer and
    # the comm method provided routing metadata in lora_context.
    # Two paths are supported:
    #   - AllGather: expanded_row_idx + topk_ids from npu_moe_init_routing
    #   - AlltoAll:  lora_context.exchanged_lora_indices + group_list after all_to_all
    lora_routing = None
    if lora_context is not None:  # LoRA applied
        from vllm_ascend.lora.fused_moe import (
            _recover_moe_lora_routing_all2all,
            _recover_moe_lora_routing_allgather,
            moe_lora_apply_w2,
            moe_lora_apply_w13,
        )

        if expanded_row_idx is not None and topk_ids is not None:
            # AllGather path: use npu_moe_init_routing's expanded_row_idx.
            lora_routing = _recover_moe_lora_routing_allgather(lora_context, expanded_row_idx, topk_ids)
        elif getattr(lora_context, "exchanged_lora_indices", None) is not None:
            # AlltoAll path: tokens already sorted by expert after exchange.
            # Build per-row (expert_id, lora_id) directly from group_list.
            lora_routing = _recover_moe_lora_routing_all2all(
                lora_context,
                group_list=group_list,
            )
        else:
            raise AssertionError(
                "MoE LoRA requires either expanded_row_idx+topk_ids "
                "(AllGather) or lora_context.exchanged_lora_indices "
                "(AlltoAll). Neither was provided."
            )

        moe_lora_apply_w13(
            lora_context,
            gate_up_out=gate_up_out,
            hidden_states=hidden_states,
            lora_routing=lora_routing,
        )

    act_name = getattr(activation, "value", activation)
    if activation == MoEActivation.SWIGLUOAI:
        num_experts, _, hidden_size = w1.shape
        gate_up_out = AscendSwigluOAIAndMul.swiglu_oai_forward(gate_up_out.view(-1, hidden_size))
    elif act_name == "swigluoai_uninterleave":
        gate_up_out = torch_npu.npu_clipped_swiglu(
            gate_up_out,
            interleaved=False,
            alpha=swiglu_alpha,
            limit=swiglu_limit,
            bias=swiglu_beta,
        )
    elif activation == MoEActivation.SWIGLUSTEP:
        gate_up_out = AscendSwigluStepAndMul.swiglustep_forward(gate_up_out, limit=swiglu_limit or 7.0)
    elif activation == MoEActivation.GELU:
        gate, up = gate_up_out.chunk(2, dim=-1)
        gate_up_out = torch.nn.functional.gelu(gate) * up
    elif activation == MoEActivation.GELU_TANH:
        gate, up = gate_up_out.chunk(2, dim=-1)
        gate_up_out = torch.nn.functional.gelu(gate, approximate="tanh") * up
    else:
        if swiglu_limit > 0:
            gate, up = gate_up_out.chunk(2, dim=-1)
            gate.clamp_(max=swiglu_limit)
            up.clamp_(min=-swiglu_limit, max=swiglu_limit)
        gate_up_out = torch_npu.npu_swiglu(gate_up_out)

    if topk_scales is not None:
        gate_up_out *= topk_scales

    hidden_states = torch_npu.npu_grouped_matmul(
        x=[gate_up_out],
        weight=[w2],
        bias=[w2_bias.to(dtype=torch.float32)] if w2_bias is not None else None,
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]

    # LoRA w2 delta: applied to the down-proj output, with the activation output
    # as the lora_a input. Reuses the per-row routing computed for w13.
    if lora_routing is not None:
        moe_lora_apply_w2(
            lora_context,
            down_out=hidden_states,
            silu_out=gate_up_out,
            lora_routing=lora_routing,
        )
    return hidden_states, None


def unified_apply_mlp(*, mlp_compute_input: MoEMlpComputeInput) -> torch.Tensor:
    """Unified MoE MLP entry.

    The unquant path is dispatched directly; the quant path delegates to
    ``quant_apply_mlp`` with ``MoEQuantParams`` as the single source of truth.
    """
    weights = mlp_compute_input.weights
    quant = mlp_compute_input.quant
    hidden_states = mlp_compute_input.hidden_states
    group_list = mlp_compute_input.group_list
    group_list_type = mlp_compute_input.group_list_type
    swiglu_alpha = mlp_compute_input.swiglu_alpha
    swiglu_beta = mlp_compute_input.swiglu_beta

    if not quant.is_quant:
        return unquant_apply_mlp(
            hidden_states=hidden_states,
            w1=weights.w1,
            w2=weights.w2,
            w1_bias=weights.w1_bias,
            w2_bias=weights.w2_bias,
            activation=mlp_compute_input.activation,
            group_list=group_list,
            group_list_type=group_list_type,
            topk_scales=mlp_compute_input.topk_scales,
            need_trans=mlp_compute_input.need_trans,
            swiglu_limit=mlp_compute_input.swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            lora_context=mlp_compute_input.lora_context,
            expanded_row_idx=mlp_compute_input.expanded_row_idx,
            topk_ids=mlp_compute_input.topk_ids,
        )

    assert weights.w1_scale is not None and weights.w2_scale is not None
    return quant_apply_mlp(
        hidden_states=hidden_states,
        weights=weights,
        quant=quant,
        group_list=group_list,
        group_list_type=group_list_type,
        dynamic_scale=mlp_compute_input.dynamic_scale,
        activation=mlp_compute_input.activation,
        fusion=mlp_compute_input.fusion,
        dynamic_eplb=mlp_compute_input.dynamic_eplb,
        swiglu_limit=mlp_compute_input.swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
    )
