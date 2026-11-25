# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol

import torch
import torch_npu

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ


class _TensorCopyStrategy(Protocol):
    def supports(self, tensor: torch.Tensor) -> bool: ...

    def copy(self, dst: torch.Tensor, cpu_tensor: torch.Tensor) -> None: ...


class _W4A8V1NZPackedCopyStrategy:
    """Copy the msModelSlim W4A8_DYNAMIC v1.0.0 packed layout."""

    def supports(self, tensor: torch.Tensor) -> bool:
        # Match the destination storage layout handled by this strategy.
        return (
            tensor.dtype == torch.int32
            and tensor.device.type == "npu"
            and int(torch_npu.get_npu_format(tensor)) == ACL_FORMAT_FRACTAL_NZ
        )

    def copy(self, dst: torch.Tensor, cpu_tensor: torch.Tensor) -> None:
        if cpu_tensor.dtype != torch.int32:
            raise RuntimeError(f"W4A8 v1 NZ weight restore expects int32 cpu tensor, got {cpu_tensor.dtype}")
        if cpu_tensor.shape != dst.shape:
            raise RuntimeError(
                f"W4A8 v1 NZ weight shape mismatch: cpu {tuple(cpu_tensor.shape)} vs dst {tuple(dst.shape)}"
            )

        # The checkpoint stores the int32 view in ND layout. Rebuild the
        # original int8 FRACTAL_NZ storage before copying it into dst.
        cpu_i8 = cpu_tensor.contiguous().view(torch.int8)
        tmp = torch.empty(cpu_i8.shape, dtype=torch.int8, device=dst.device)
        tmp.copy_(cpu_i8)
        tmp = torch_npu.npu_format_cast(tmp, ACL_FORMAT_FRACTAL_NZ)
        dst.view(torch.int8).copy_(tmp)


class _Int8NZCopyStrategy:
    """Stage INT8 weights on the NPU before converting to the destination layout."""

    def supports(self, tensor: torch.Tensor) -> bool:
        return (
            tensor.dtype == torch.int8
            and tensor.device.type == "npu"
            and int(torch_npu.get_npu_format(tensor)) == ACL_FORMAT_FRACTAL_NZ
        )

    def copy(self, dst: torch.Tensor, cpu_tensor: torch.Tensor) -> None:
        if cpu_tensor.dtype != torch.int8 or cpu_tensor.shape != dst.shape:
            # Preserve copy_ conversion and broadcasting for non-matching inputs.
            dst.copy_(cpu_tensor)
            return

        nd = torch.empty(cpu_tensor.shape, dtype=torch.int8, device=dst.device)
        nd.copy_(cpu_tensor.contiguous())
        nz = torch_npu.npu_format_cast(nd, ACL_FORMAT_FRACTAL_NZ)
        dst.copy_(nz)


class _DirectCopyStrategy:
    """Copy tensors that do not require layout reconstruction."""

    def supports(self, tensor: torch.Tensor) -> bool:
        return True

    def copy(self, dst: torch.Tensor, cpu_tensor: torch.Tensor) -> None:
        dst.copy_(cpu_tensor)


# Dedicated restore strategies for msModelSlim W4A8_DYNAMIC with
# quant_description["version"] != "1.0.0" and W4A8_MXFP are not implemented.
_TENSOR_COPY_STRATEGIES: tuple[_TensorCopyStrategy, ...] = (
    _W4A8V1NZPackedCopyStrategy(),
    _Int8NZCopyStrategy(),
    _DirectCopyStrategy(),
)


def copy_checkpoint_tensor(dst: torch.Tensor, cpu_tensor: torch.Tensor) -> None:
    for strategy in _TENSOR_COPY_STRATEGIES:
        if strategy.supports(dst):
            strategy.copy(dst, cpu_tensor)
            return
    raise RuntimeError(f"No copy strategy for tensor with dtype={dst.dtype}, device={dst.device}")
