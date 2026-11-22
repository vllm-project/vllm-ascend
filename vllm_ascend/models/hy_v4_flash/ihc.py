# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from functools import cache

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.models.hy_v4.nvidia.hc import HYV4HCHeadLayer, HYV4HCLayer, HYV4HCPreLayer
from vllm.triton_utils import HAS_TRITON, tl, triton

from vllm_ascend.device.device_config import is_950

from .attention import get_flash_attention_metadata

logger = init_logger(__name__)

_PRECISE_DECODE_MEAN_GROUPS = {6: 4, 12: 8, 18: 16, 24: 8, 48: 16}


@triton.jit
def _ihc_post_kernel(
    hidden,
    residual,
    gates,
    output,
    HIDDEN_SIZE: tl.constexpr,
    CHANNELS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    channel = tl.program_id(1)
    col = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    mask = col < HIDDEN_SIZE
    lane = token * CHANNELS + channel
    x = tl.load(hidden + token * HIDDEN_SIZE + col, mask, 0.0).to(tl.float32)
    r = tl.load(residual + lane * HIDDEN_SIZE + col, mask, 0.0).to(tl.float32)
    p = tl.load(gates + lane)
    tl.store(output + lane * HIDDEN_SIZE + col, r + p * x, mask)


def fused_ihc_post(hidden, residual, gates):
    tokens, channels, hidden_size = residual.shape
    output = torch.empty_like(residual)
    block_size = 1024
    _ihc_post_kernel[(tokens, channels, triton.cdiv(hidden_size, block_size))](
        hidden,
        residual,
        gates,
        output,
        hidden_size,
        channels,
        block_size,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return output


@cache
def _decode_mean_workers(device_index: int) -> tuple[torch.npu.Stream, ...]:
    return tuple(torch.npu.Stream(device=device_index) for _ in range(3))


def _parallel_precise_mean(squares, means, rows, workers):
    """Keep each original single-row CANN mean, using independent output rows."""
    if means is None:
        means = torch.empty((squares.shape[0], 1), dtype=squares.dtype, device=squares.device)
    count = squares.shape[0]
    complete_rows = rows == list(range(count))
    group = _PRECISE_DECODE_MEAN_GROUPS.get(count) if complete_rows and squares.is_contiguous() else None
    if group is not None:
        logger.info_once("HYV4 Decode groups precise iHC row reductions.")
        logger.info_once("HYV4 Decode uses larger exact iHC mean groups on one stream.")
        for start in range(0, count, group):
            stop = min(start + group, count)
            torch.mean(squares[start:stop], dim=-1, keepdim=True, out=means[start:stop])
        return means

    main = torch.npu.current_stream()
    workers = workers[:1] if len(rows) < 18 else workers
    for stream in workers:
        stream.wait_stream(main)
    streams = (main, *workers)
    # These complete-row batches use the same CANN FP32 reduction order as one row.
    group = 4 if count in (6, 12, 18, 24, 30, 48) and complete_rows else 1
    if group > 1:
        logger.info_once("HYV4 Decode groups precise iHC row reductions.")
    starts = range(0, count, group) if group > 1 else rows
    for offset, stream in enumerate(streams):
        with torch.npu.stream(stream):
            for start in starts[offset :: len(streams)]:
                stop = min(start + group, count)
                torch.mean(squares[start:stop], dim=-1, keepdim=True, out=means[start:stop])
    for stream in workers:
        main.wait_stream(stream)
    return means


def _inverse_rms(
    x: torch.Tensor,
    eps: float,
    row_offset: int = 0,
    mean_workers: tuple[torch.npu.Stream, ...] = (),
) -> torch.Tensor:
    squares = x.square()
    if x.shape[0] <= 1:
        return (squares.mean(-1, keepdim=True) + eps).rsqrt()
    batch = get_flash_attention_metadata()
    if batch is None or not any(batch.precise_decode_reqs):
        return (squares.mean(-1, keepdim=True) + eps).rsqrt()

    # CANN's multi-row ReduceMean can use a different reduction order than
    # the single-row decode. Reduce decode rows separately and keep the
    # original batch reduction for any prefill rows.
    row_end = row_offset + x.shape[0]
    has_padding = row_end > batch.query_start_loc[-1]
    default = None if all(batch.precise_decode_reqs) and not has_padding else squares.mean(-1, keepdim=True)
    if (
        mean_workers
        and x.device.type == "npu"
        and x.dtype == torch.float32
        and x.ndim == 2
        and x.shape[1] == 12288
        and batch.query_start_loc[0] <= row_offset
        and len(batch.query_start_loc) == len(batch.precise_decode_reqs) + 1
    ):
        precise_rows = [
            row
            for start, stop, precise in zip(batch.query_start_loc, batch.query_start_loc[1:], batch.precise_decode_reqs)
            if precise
            for row in range(max(0, start - row_offset), min(x.shape[0], stop - row_offset))
        ]
        if len(precise_rows) >= 6:
            logger.info_once("HYV4 Decode runs precise iHC row means on parallel NPU streams")
            means = _parallel_precise_mean(squares, default, precise_rows, mean_workers)
            return (means + eps).rsqrt()

    means = []
    for start, stop, precise in zip(batch.query_start_loc, batch.query_start_loc[1:], batch.precise_decode_reqs):
        start = max(0, start - row_offset)
        stop = min(x.shape[0], stop - row_offset)
        if start >= stop:
            continue
        if precise:
            means.extend(row.mean(-1, keepdim=True) for row in squares[start:stop].split(1, dim=0))
        else:
            means.append(default[start:stop])
    if has_padding:
        means.append(default[max(0, batch.query_start_loc[-1] - row_offset) :])
    return (torch.cat(means, dim=0) + eps).rsqrt()


class FlashHCPreLayer(HYV4HCPreLayer):
    sequence_parallel_input = False
    use_cann_inverse_rms = False
    cann_pre_op = None
    inverse_rms_workers: tuple[torch.npu.Stream, ...] = ()

    def forward(self, x):
        if (
            self.cann_pre_op is not None
            and not self.sequence_parallel_input
            and self.magnitude == 2.0
            and x.device.type == "npu"
            and x.dtype == torch.bfloat16
            and x.ndim == 3
            and x.is_contiguous()
        ):
            tokens, channels, width = x.shape
            weight, scale, bias = self.hc_fn.weight, self.hc_scale, self.hc_base
            batch = get_flash_attention_metadata()
            if (
                tokens >= 128
                and channels == self.hc_mult
                and channels in (4, 6, 8)
                and 0 < width <= 16384
                and width % 16 == 0
                and weight.shape == (2 * channels, channels * width)
                and scale.shape == (2,)
                and bias.shape == (2 * channels,)
                and weight.dtype == scale.dtype == bias.dtype == torch.float32
                and weight.device == scale.device == bias.device == x.device
                and weight.is_contiguous()
                and scale.is_contiguous()
                and bias.is_contiguous()
                and batch is not None
                and batch.prefill_reqs
                and all(batch.prefill_reqs)
            ):
                hidden, post, *_ = self.cann_pre_op(
                    x,
                    weight,
                    scale,
                    bias,
                    norm_eps=self.layernorm_epsilon,
                    hc_eps=self.hc_eps,
                    out_flag=0,
                    inner_precise=0,
                )
                return hidden, post
        return super().forward(x)

    def _inverse_rms(self, x: torch.Tensor) -> torch.Tensor:
        row_offset = get_tensor_model_parallel_rank() * x.shape[0] if self.sequence_parallel_input else 0
        if (
            self.use_cann_inverse_rms
            and not self.sequence_parallel_input
            and x.device.type == "npu"
            and x.dtype == torch.float32
            and x.ndim == 2
            and x.shape[0] > 1
            and x.is_contiguous()
            and 0 < x.shape[1] <= 16384
            and x.shape[1] % 128 == 0
        ):
            batch = get_flash_attention_metadata()
            weight = self.hc_fn.weight
            if (
                batch is not None
                and not any(batch.precise_decode_reqs)
                and weight.device == x.device
                and weight.dtype == x.dtype
                and weight.ndim == 2
                and weight.shape[1] == x.shape[1]
                and weight.is_contiguous()
            ):
                # CANN computes rstd independently of gamma. Reuse one existing
                # row for its unused normalized output.
                _, inverse = torch_npu.npu_rms_norm(x, weight[0], self.layernorm_epsilon)
                return inverse
        return _inverse_rms(x, self.layernorm_epsilon, row_offset, self.inverse_rms_workers)


class FlashHCLayer(HYV4HCLayer):
    pre_layer_cls = FlashHCPreLayer
    use_cann_prefill = False
    cann_post_op = None

    def configure_prefill_post(self):
        self.use_cann_prefill = True
        if self.enable_ihc:
            self.hc_pre.use_cann_inverse_rms = is_950() and hasattr(torch_npu, "npu_rms_norm")
            if is_950() and get_current_vllm_config().model_config.enforce_eager:
                self.hc_pre.cann_pre_op = getattr(torch_npu, "npu_mhc_pre", None)
        if os.getenv("HY4FLASH_CANN_IHC_POST") == "1":
            self.cann_post_op = getattr(torch_npu, "npu_mhc_post", None)

    def configure_decode_post(self):
        if self.enable_ihc and is_950():
            self.cann_post_op = getattr(torch_npu, "npu_mhc_post", None)
            self.hc_pre.inverse_rms_workers = _decode_mean_workers(torch.npu.current_device())

    def post(self, output_with_bias, residual, post_gates):
        if (
            (self.use_cann_prefill or HAS_TRITON)
            and self.enable_ihc
            and output_with_bias.device.type == "npu"
            and output_with_bias.dtype == residual.dtype == torch.bfloat16
            and post_gates is not None
            and post_gates.dtype == torch.float32
            and output_with_bias.shape[0] > 0
            and output_with_bias.is_contiguous()
            and residual.is_contiguous()
            and post_gates.is_contiguous()
        ):
            if self.cann_post_op is not None and is_950():
                tokens, width = output_with_bias.shape if output_with_bias.ndim == 2 else (0, 0)
                if (
                    tokens > 0
                    and 0 < width <= 100_000
                    and width % 128 == 0
                    and residual.shape == (tokens, 4, width)
                    and post_gates.shape == (tokens, 4)
                    and residual.device == post_gates.device == output_with_bias.device
                ):
                    return self.cann_post_op(residual, None, output_with_bias, post_gates)
            if self.use_cann_prefill:
                scaled = output_with_bias.float().unsqueeze(1) * post_gates.unsqueeze(-1)
                return (residual.float() + scaled).to(residual.dtype)
            return fused_ihc_post(output_with_bias, residual, post_gates)
        return super().post(output_with_bias, residual, post_gates)


class FlashHCHeadLayer(HYV4HCHeadLayer):
    def _inverse_rms(self, x: torch.Tensor) -> torch.Tensor:
        return _inverse_rms(x, self.config.rms_norm_eps)
