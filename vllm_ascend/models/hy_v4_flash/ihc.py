# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.models.hy_v4.nvidia.hc import HYV4HCHeadLayer, HYV4HCLayer, HYV4HCPreLayer
from vllm.triton_utils import HAS_TRITON, tl, triton

from .attention import get_flash_attention_metadata


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


def _inverse_rms(x: torch.Tensor, eps: float, row_offset: int = 0) -> torch.Tensor:
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

    def _inverse_rms(self, x: torch.Tensor) -> torch.Tensor:
        row_offset = get_tensor_model_parallel_rank() * x.shape[0] if self.sequence_parallel_input else 0
        return _inverse_rms(x, self.layernorm_epsilon, row_offset)


class FlashHCLayer(HYV4HCLayer):
    pre_layer_cls = FlashHCPreLayer

    def post(self, output_with_bias, residual, post_gates):
        if (
            HAS_TRITON
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
            return fused_ihc_post(output_with_bias, residual, post_gates)
        return super().post(output_with_bias, residual, post_gates)


class FlashHCHeadLayer(HYV4HCHeadLayer):
    def _inverse_rms(self, x: torch.Tensor) -> torch.Tensor:
        return _inverse_rms(x, self.config.rms_norm_eps)
