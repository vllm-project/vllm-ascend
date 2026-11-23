# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve captured MXFP4 storage when layerwise reload copies NZ weights.

Remove when upstream layerwise reload exposes a backend tensor-copy hook.
"""

import torch
from vllm.model_executor.model_loader.reload import layerwise

from vllm_ascend.quantization.online_mx import AscendOnlineMXLinearMethod, AscendOnlineMXMoEMethod
from vllm_ascend.quantization.quant_type import QuantType


def _copy_mxfp4_tensor(destination: torch.Tensor, source: torch.Tensor) -> None:
    # Native FP4 schemes expose a transpose of contiguous NZ storage. Ascend950
    # copy_ requires metadata matching that storage, so undo both transposes.
    # This copies on device without replacing either captured storage or layout.
    if (
        destination.ndim >= 2
        and not destination.is_contiguous()
        and destination.transpose(-1, -2).is_contiguous()
        and source.transpose(-1, -2).is_contiguous()
    ):
        destination.transpose(-1, -2).copy_(source.transpose(-1, -2))
    else:
        destination.copy_(source)


def _copy_and_restore_kernel_tensors(layer, info):
    method = getattr(layer, "quant_method", None)
    if not (
        isinstance(method, (AscendOnlineMXLinearMethod, AscendOnlineMXMoEMethod))
        and method.quant_method.quant_type == QuantType.W4A8MXFP
    ):
        return _original_copy_and_restore(layer, info)

    assert info.kernel_tensors is not None
    parameters, buffers = info.kernel_tensors
    loaded_tensor_names = {name for name, _ in info.loaded_weights}
    for name, param in parameters.items():
        _copy_mxfp4_tensor(param.data, getattr(layer, name))
    for name, buffer in buffers.items():
        if name not in layer._buffers:
            continue
        if name in info.kernel_non_persistent_buffers and name not in loaded_tensor_names:
            continue
        _copy_mxfp4_tensor(buffer.data, getattr(layer, name))
    layerwise._place_kernel_tensors(layer, info)


_original_copy_and_restore = layerwise._copy_and_restore_kernel_tensors
layerwise._copy_and_restore_kernel_tensors = _copy_and_restore_kernel_tensors
