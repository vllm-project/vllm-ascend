# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 AllToAll/AllGather SUM with BF16 staging on Atlas A5.

Uses the existing process group and ordinary collectives, without a native
QuantAllReduce operator or a separate HCCL domain. Routing depends on tensor
metadata and payload size, including for mixed prefill/decode batches.
"""

import logging
from collections import deque
from typing import Any, NamedTuple

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
_PIPELINE_CHUNKS = 2


def _chunk_ranges(rows: int, world_size: int) -> list[tuple[int, int]]:
    """Split rows into two nonempty chunks in group-sized units."""
    if rows % world_size or rows // world_size < _PIPELINE_CHUNKS:
        raise ValueError("MXFP8 chunks require at least one group-sized row unit per chunk.")
    units, remainder = divmod(rows // world_size, _PIPELINE_CHUNKS)
    split = (units + remainder) * world_size
    return [(0, split), (split, rows)]


class _Exchange(NamedTuple):
    start: int
    end: int
    send: torch.Tensor
    received: torch.Tensor
    scale_dtype: torch.dtype
    work: dist.Work


class Mxfp8AllReduce:
    """BF16 -> MXFP8 AllToAll -> BF16 local SUM -> BF16 AllGather."""

    def __init__(
        self,
        group: "dist.ProcessGroup",
        min_numel: int,
        group_name: str = "",
        *,
        enable_pipeline: bool = False,
        pipeline_min_numel: int | None = None,
    ) -> None:
        if not isinstance(enable_pipeline, bool):
            raise ValueError("MXFP8 enable_pipeline must be a bool.")
        if pipeline_min_numel is not None and pipeline_min_numel < 1:
            raise ValueError("MXFP8 pipeline_min_numel must be >= 1.")
        self.group = group
        self.group_name = group_name
        self.min_numel = min_numel
        self.enable_pipeline = enable_pipeline
        self.pipeline_min_numel = min_numel if pipeline_min_numel is None else pipeline_min_numel
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

    def _pipeline_skip_reason(self, input_: torch.Tensor) -> str | None:
        if not self.enable_pipeline:
            return "disabled"
        if input_.numel() < self.pipeline_min_numel:
            return "below_pipeline_threshold"
        rows = input_.numel() // input_.shape[-1]
        if rows // self.world_size < _PIPELINE_CHUNKS:
            return "insufficient_rows"
        # Async Work joins are validated in Eager only. Preserve the existing
        # full-tensor path during capture, including for unusually large graphs.
        if torch.npu.is_current_stream_capturing():
            return "graph_capture"
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

        pipeline_reason = self._pipeline_skip_reason(input_)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[MXFP8-AR-HIT] group=%s world_size=%d shape=%s numel=%d payload_bytes=%d min_numel=%d "
                "mode=%s chunks=%d pipeline_min_numel=%d pipeline_reason=%s",
                self.group_name,
                self.world_size,
                tuple(input_.shape),
                input_.numel(),
                input_.numel() * input_.element_size(),
                self.min_numel,
                "pipeline" if pipeline_reason is None else "full",
                _PIPELINE_CHUNKS if pipeline_reason is None else 1,
                self.pipeline_min_numel,
                pipeline_reason or "eligible",
            )

        if pipeline_reason is None:
            return self._all_reduce_pipelined(input_)
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

    def _all_reduce_pipelined(self, input_: torch.Tensor) -> torch.Tensor:
        hidden = input_.shape[-1]
        rows = input_.numel() // hidden
        flat = input_.reshape(rows, hidden)
        output = torch.empty_like(input_, memory_format=torch.contiguous_format)
        flat_output = output.view(rows, hidden)
        pending: deque[_Exchange] = deque()
        # Submit both AllToAll calls before waiting, in the same order on all ranks.
        for start, end in _chunk_ranges(rows, self.world_size):
            send, scale_dtype = self._pack(flat[start:end])
            received = torch.empty_like(send)
            work = dist.all_to_all_single(received, send, group=self.group, async_op=True)
            pending.append(_Exchange(start, end, send, received, scale_dtype, work))
        del send, received, work

        gathers = []
        while pending:
            exchange = pending.popleft()
            exchange.work.wait()
            partial = self._partial(exchange.received, exchange.scale_dtype, hidden)
            target = flat_output[exchange.start : exchange.end]
            work = dist.all_gather_into_tensor(target, partial, group=self.group, async_op=True)
            # Retain async inputs, output views and work handles through their
            # joins; do not recycle a shard while HCCL may still access it.
            gathers.append((partial, target, work))
            del exchange
        for partial, target, work in gathers:
            work.wait()
        # Work.wait joins every output onto the caller's stream. No device
        # synchronize, new process group or persistent payload cache is needed.
        return output
