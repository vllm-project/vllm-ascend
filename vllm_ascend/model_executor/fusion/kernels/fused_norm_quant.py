from collections.abc import Callable

import torch
from vllm.logger import logger
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    get_input_quant_key,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt8StaticTensorSym,
)

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.layernorm import AscendRMSNorm
from vllm_ascend.ops.linear import AscendQKVParallelLinear


def _add_rms_norm_quant(
    norm: AscendRMSNorm,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    linear: AscendQKVParallelLinear,
):
    
    output = torch.ops.npu.npu_add_rms_norm_quant(
        hidden_states,
        residual,
        norm.weight,
        linear.aclnn_input_scale,
        linear.aclnn_input_offset,
        epsilon=norm.variance_epsilon,
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


_FUSED_NORM_QUANT: dict[tuple[type, QuantKey], Callable] = {}


ascend_compilation_config = get_ascend_config().ascend_compilation_config
profile = get_current_hardware_profile()

if ascend_compilation_config.fuse_norm_quant and profile.supports(
    HardwareCapability.GRAPH_NORM_QUANT_FUSION
):
    _FUSED_NORM_QUANT[(AscendRMSNorm, kInt8StaticTensorSym)] = _add_rms_norm_quant


def maybe_fused_norm_quant(
    norm: AscendRMSNorm,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    linear: AscendQKVParallelLinear,
) -> tuple[QuantizedActivation, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:

    key = get_input_quant_key(linear)
    if key is not None:
        registry_key = (type(norm), key)
        producer = _FUSED_NORM_QUANT.get(registry_key)
        if producer is not None:
            return producer(norm, hidden_states, residual, linear)
    return norm(hidden_states, residual)
    


