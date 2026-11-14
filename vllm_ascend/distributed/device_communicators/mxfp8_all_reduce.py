# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 AllToAll/AllGather SUM with BF16 staging on Atlas A5.

Uses the existing process group and ordinary collectives, without a native
QuantAllReduce operator or a separate HCCL domain. Routing depends on tensor
metadata and payload size, including for mixed prefill/decode batches.
"""

import logging
from typing import Any

import torch
import torch.distributed as dist
import torch_npu

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)

_SUPPORTED_WORLD_SIZES = (2, 4, 8)
_MX_BLOCK_SIZE = 32
_SCALE_PAIR_SIZE = 2
_HIDDEN_ALIGNMENT = _MX_BLOCK_SIZE * _SCALE_PAIR_SIZE
_DTYPE_BF16 = 15
_DTYPE_E4M3FN = 292
_SCALE_ALG_CEIL = 1


class Mxfp8AllReduce:
    """BF16 -> MXFP8 AllToAll -> BF16 local SUM -> BF16 AllGather."""

    def __init__(
        self,
        group: "dist.ProcessGroup",
        min_numel: int,
        group_name: str = "",
    ) -> None:
        self.group = group
        self.group_name = group_name
        self.min_numel = min_numel
        self.world_size = dist.get_world_size(group)
        device_type = get_ascend_device_type()
        if device_type != AscendDeviceType.A5:
            raise RuntimeError("MXFP8 AllToAll quantized all-reduce requires Atlas A5.")
        self.supported = self.world_size in _SUPPORTED_WORLD_SIZES
        self.quantize_op: Any = getattr(torch_npu, "npu_dynamic_mx_quant", None)
        self.dequantize_op: Any = getattr(torch_npu, "npu_anti_mx_quant", None)
        if self.supported and (not callable(self.quantize_op) or not callable(self.dequantize_op)):
            raise RuntimeError(
                "MXFP8 AllToAll quantized all-reduce requires npu_dynamic_mx_quant and npu_anti_mx_quant. "
                "Install matching torch_npu/CANN operators or select quant_dtype='int8'."
            )

    def _skip_reason(self, input_: torch.Tensor) -> str | None:
        if not self.supported:
            return "unsupported_group"
        if input_.dtype != torch.bfloat16:
            return "dtype"
        if input_.ndim < 2 or input_.shape[-1] < _HIDDEN_ALIGNMENT:
            return "degenerate_shape"
        if input_.numel() < self.min_numel:
            return "below_threshold"
        hidden = input_.shape[-1]
        if hidden % _HIDDEN_ALIGNMENT:
            return "hidden_alignment"
        if (input_.numel() // hidden) % self.world_size:
            return "row_sharding"
        return None

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        reason = self._skip_reason(input_)
        if reason is not None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[MXFP8-AR-SKIP] reason=%s group=%s world_size=%d shape=%s dtype=%s numel=%d min_numel=%d",
                    reason,
                    self.group_name,
                    self.world_size,
                    tuple(input_.shape),
                    input_.dtype,
                    input_.numel(),
                    self.min_numel,
                )
            dist.all_reduce(input_, group=self.group)
            return input_

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[MXFP8-AR-HIT] group=%s world_size=%d shape=%s numel=%d payload_bytes=%d min_numel=%d",
                self.group_name,
                self.world_size,
                tuple(input_.shape),
                input_.numel(),
                input_.numel() * input_.element_size(),
                self.min_numel,
            )

        return self._all_reduce_full(input_)

    def _pack(self, flat: torch.Tensor) -> tuple[torch.Tensor, torch.dtype]:
        rows, hidden = flat.shape
        # Round the E8M0 scale up after mapping amax to the E4M3FN range.
        # The default exponent-only algorithm can leave scaled values above 448.
        values, scales = self.quantize_op(
            flat,
            axis=-1,
            dst_type=torch.float8_e4m3fn,
            block_size=_MX_BLOCK_SIZE,
            round_mode="rint",
            scale_alg=_SCALE_ALG_CEIL,
        )
        if (
            values.dtype != torch.float8_e4m3fn
            or tuple(values.shape) != (rows, hidden)
            or scales.element_size() != 1
            or scales.numel() != flat.numel() // _MX_BLOCK_SIZE
        ):
            raise RuntimeError("Unexpected MXFP8 quantization dtype or layout; cannot form the AllToAll payload.")

        # Preserve E8M0 byte encodings. A numeric cast would corrupt scales.
        scale_dtype = scales.dtype
        scale_bytes = scales.contiguous().view(torch.uint8).reshape(rows, -1)
        send = torch.cat((values.contiguous().view(torch.uint8), scale_bytes), dim=-1)
        return send, scale_dtype

    def _partial(self, received: torch.Tensor, scale_dtype: torch.dtype, hidden: int) -> torch.Tensor:
        rows = received.shape[0]
        values = received[:, :hidden].contiguous().view(torch.float8_e4m3fn)
        scales = received[:, hidden:].contiguous().view(scale_dtype)
        scales = scales.reshape(rows, hidden // _HIDDEN_ALIGNMENT, _SCALE_PAIR_SIZE)
        dequantized = self.dequantize_op(values, scales, axis=-1, dst_type=_DTYPE_BF16, src_type=_DTYPE_E4M3FN)
        if dequantized.dtype != torch.bfloat16 or tuple(dequantized.shape) != (rows, hidden):
            raise RuntimeError("MXFP8 dequantization must produce a BF16 tensor matching the received payload.")
        # Every source contributes the owner's row shard. Both dequantization
        # storage and SUM output are BF16; accumulation/workspace is backend-defined.
        return dequantized.reshape(self.world_size, rows // self.world_size, hidden).sum(dim=0, dtype=torch.bfloat16)

    def _all_reduce_full(self, input_: torch.Tensor) -> torch.Tensor:
        hidden = input_.shape[-1]
        rows = input_.numel() // hidden
        send, scale_dtype = self._pack(input_.reshape(rows, hidden))
        received = torch.empty_like(send)
        dist.all_to_all_single(received, send, group=self.group)
        del send
        partial = self._partial(received, scale_dtype, hidden)
        del received

        # Return a separate result, following the GroupCoordinator contract.
        output = torch.empty_like(input_, memory_format=torch.contiguous_format)
        dist.all_gather_into_tensor(output.view(rows, hidden), partial, group=self.group)
        return output
