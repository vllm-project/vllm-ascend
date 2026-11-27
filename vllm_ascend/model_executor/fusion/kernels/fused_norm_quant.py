from collections.abc import Callable

import torch
from vllm.logger import logger
from vllm import envs
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt8StaticTensorSym,
    kInt8DynamicTokenSym,
)

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile


def _add_rms_norm_quant(
    norm: torch.nn.Module,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    linear: LinearBase,
):
    output = torch.ops.npu.npu_add_rms_norm_quant(
        hidden_states,
        residual,
        norm.weight,
        linear.aclnn_input_scale,
        linear.aclnn_input_offset,
        epsilon=norm.variance_epsilon,
        beta=norm.bias,
    )
    quantized_output = output[0]
    out1 = output[2]
    logger.info_once("norm and quant fused!!!")
    return QuantizedActivation(
        data=quantized_output,
        scale=linear.aclnn_input_scale,
        orig_dtype=hidden_states.dtype,
        orig_shape=hidden_states.shape,
        quant_key=kInt8StaticTensorSym,
    ), out1


def _add_gemma_rms_norm_dynamic_quant(
    norm: torch.nn.Module,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    linear: LinearBase,
):

    output = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
        hidden_states, residual, 1+norm.weight, epsilon=norm.variance_epsilon, output_mask=[True, False]
    )
    logger.info_once("norm and quant fused!!!")
    return QuantizedActivation(
        data=output[0],
        scale=output[3],
        orig_dtype=hidden_states.dtype,
        orig_shape=hidden_states.shape,
        quant_key=kInt8DynamicTokenSym,
    ), output[0], output[2]
    


_FUSED_NORM_QUANT: dict[tuple[str, QuantKey], Callable] = {}


if  get_current_hardware_profile().supports(
    HardwareCapability.GRAPH_NORM_QUANT_FUSION
):
    _FUSED_NORM_QUANT[("AscendGemmaRMSNorm", kInt8DynamicTokenSym)] = _add_gemma_rms_norm_dynamic_quant
    
    if not envs.VLLM_BATCH_INVARIANT:
        _FUSED_NORM_QUANT[("AscendRMSNorm", kInt8StaticTensorSym)] = _add_rms_norm_quant



def fused_norm_quant(
    key: QuantKey,
    norm: torch.nn.Module,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    linear: LinearBase,
) -> tuple[QuantizedActivation | None, QuantizedActivation | torch.Tensor, torch.Tensor]:
    registry_key = ((type(norm).__name__), key)
    producer = _FUSED_NORM_QUANT.get(registry_key)
    if producer is not None:
        return producer(norm, hidden_states, residual, linear)
    return None, norm(hidden_states, residual)
