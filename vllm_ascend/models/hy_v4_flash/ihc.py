# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import cache

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.models.hy_v4.nvidia.hc import HYV4HCHeadLayer, HYV4HCLayer, HYV4HCPreLayer
from vllm.triton_utils import HAS_TRITON, tl, triton

from vllm_ascend.device.device_config import is_950

from .attention import get_flash_attention_metadata
from .ihc_first import first_ihc_post, first_ihc_pre

logger = init_logger(__name__)

_PRECISE_DECODE_MEAN_GROUPS = {6: 4, 12: 8, 18: 16, 24: 8, 48: 16}
_FUSED_PRE_CHANNELS = 4
_FUSED_PRE_HIDDEN_SIZE = 3072
_FUSED_PRE_MAX_TOKENS = 48
_FUSED_PRE_QUERY_COUNT = 6
_FUSED_PRE_GATE_BLOCK = 128
_FUSED_PRE_WEIGHT_BLOCK = 1024


@triton.jit
def _ihc_pre_gate_logits_kernel(
    mixes,
    inverse_rms,
    scales,
    bias,
    pre_logits,
    post_logits,
    TOKENS: tl.constexpr,
    CHANNELS: tl.constexpr,
    MIX_ROW_STRIDE: tl.constexpr,
    MIX_COL_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = offset < TOKENS * CHANNELS
    token = offset // CHANNELS
    channel = offset % CHANNELS
    inverse = tl.load(inverse_rms + token, live, 0.0)
    pre = tl.load(mixes + token * MIX_ROW_STRIDE + channel * MIX_COL_STRIDE, live, 0.0)
    post = tl.load(mixes + token * MIX_ROW_STRIDE + (CHANNELS + channel) * MIX_COL_STRIDE, live, 0.0)
    pre_scale = tl.load(scales)
    post_scale = tl.load(scales + 1)
    pre_bias = tl.load(bias + channel, live, 0.0)
    post_bias = tl.load(bias + CHANNELS + channel, live, 0.0)
    # Keep the eager FP32 operation order; launches disable FMA contraction.
    pre = (pre * inverse) * pre_scale + pre_bias
    post = (post * inverse) * post_scale + post_bias
    tl.store(pre_logits + offset, pre, live)
    tl.store(post_logits + offset, post, live)


@triton.jit
def _ihc_pre_weighted_channels_kernel(
    hidden,
    pre_probability,
    post_probability,
    weighted,
    post_gates,
    TOKENS: tl.constexpr,
    CHANNELS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    MAGNITUDE: tl.constexpr,
    EPSILON: tl.constexpr,
    GATE_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = offset < TOKENS * CHANNELS * HIDDEN_SIZE
    channel_row = offset // HIDDEN_SIZE
    probability = tl.load(pre_probability + channel_row, live, 0.0)
    value = tl.load(hidden + offset, live, 0.0).to(tl.float32)
    tl.store(weighted + offset, (probability + EPSILON) * value, live)
    # One program writes each post gate once, outside the hidden-width grid.
    if tl.program_id(0) == 0:
        gate = tl.arange(0, GATE_BLOCK)
        gate_live = gate < TOKENS * CHANNELS
        post_gate_probability = tl.load(post_probability + gate, gate_live, 0.0)
        tl.store(post_gates + gate, MAGNITUDE * post_gate_probability + EPSILON, gate_live)


def fused_ihc_pre_tail(hidden, mixes, inverse_rms, scales, bias, magnitude, epsilon):
    """Fuse decode gate arithmetic while keeping CANN sigmoid and channel sum."""
    tokens, channels, width = hidden.shape
    pre_logits = torch.empty((tokens, channels), dtype=torch.float32, device=hidden.device)
    post_logits = torch.empty_like(pre_logits)
    _ihc_pre_gate_logits_kernel[(triton.cdiv(tokens * channels, _FUSED_PRE_GATE_BLOCK),)](
        mixes,
        inverse_rms,
        scales,
        bias,
        pre_logits,
        post_logits,
        tokens,
        channels,
        mixes.stride(0),
        mixes.stride(1),
        _FUSED_PRE_GATE_BLOCK,
        num_warps=1,
        enable_fp_fusion=False,
    )
    # Preserve the original [tokens, channels] native sigmoid calls. Folding
    # sigmoid or ReduceSum into Triton requires separate numerical validation.
    pre_probability = pre_logits.sigmoid()
    post_probability = post_logits.sigmoid()
    weighted = torch.empty(hidden.shape, dtype=torch.float32, device=hidden.device)
    post_gates = torch.empty_like(post_probability)
    _ihc_pre_weighted_channels_kernel[(triton.cdiv(hidden.numel(), _FUSED_PRE_WEIGHT_BLOCK),)](
        hidden,
        pre_probability,
        post_probability,
        weighted,
        post_gates,
        tokens,
        channels,
        width,
        magnitude,
        epsilon,
        triton.next_power_of_2(tokens * channels),
        _FUSED_PRE_WEIGHT_BLOCK,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return torch.sum(weighted, dim=1).to(hidden.dtype), post_gates


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

    fuse_decode_tail = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.broadcast_gate_buffers: dict[int, str] = {}

    def _can_fuse_decode_tail(self, x: torch.Tensor) -> bool:
        if (
            not self.fuse_decode_tail
            or not HAS_TRITON
            or not is_950()
            or self.sequence_parallel_input
            or x.device.type != "npu"
            or x.dtype != torch.bfloat16
            or x.ndim != 3
            or not x.is_contiguous()
            or x.shape[1:] != (_FUSED_PRE_CHANNELS, _FUSED_PRE_HIDDEN_SIZE)
            or not 0 < x.shape[0] <= _FUSED_PRE_MAX_TOKENS
            or x.shape[0] % _FUSED_PRE_QUERY_COUNT
            or self.hc_mult != _FUSED_PRE_CHANNELS
        ):
            return False
        weight, scales, bias = self.hc_fn.weight, self.hc_scale, self.hc_base
        if (
            weight.shape != (2 * _FUSED_PRE_CHANNELS, _FUSED_PRE_CHANNELS * _FUSED_PRE_HIDDEN_SIZE)
            or scales.shape != (2,)
            or bias.shape != (2 * _FUSED_PRE_CHANNELS,)
            or weight.dtype != scales.dtype
            or scales.dtype != bias.dtype
            or bias.dtype != torch.float32
            or weight.device != x.device
            or scales.device != x.device
            or bias.device != x.device
            or not weight.is_contiguous()
            or not scales.is_contiguous()
            or not bias.is_contiguous()
            or getattr(self.hc_fn, "bias", None) is not None
        ):
            return False
        batch = get_flash_attention_metadata()
        return (
            batch is not None
            and bool(batch.prefill_reqs)
            and not any(batch.prefill_reqs)
            and bool(batch.precise_decode_reqs)
            and all(batch.precise_decode_reqs)
            and bool(batch.query_start_loc)
            and batch.query_start_loc[0] == 0
            and batch.query_start_loc[-1] == x.shape[0]
        )

    def forward(self, x):
        if x.ndim == 2 and x.shape[0] in self.broadcast_gate_buffers:
            token_ids = get_forward_context().additional_kwargs["hy4_first_ihc_token_ids"]
            gates = getattr(self, self.broadcast_gate_buffers[x.shape[0]])
            return first_ihc_pre(x, token_ids, gates)
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
        if self._can_fuse_decode_tail(x):
            flat = x.flatten(1).float()
            inverse = self._inverse_rms(flat)
            mixes = self.hc_fn(flat)[0]
            return fused_ihc_pre_tail(x, mixes, inverse, self.hc_scale, self.hc_base, self.magnitude, self.hc_eps)
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
    fuse_decode_post = True

    def configure_prefill_post(self):
        self.use_cann_prefill = True
        if self.enable_ihc:
            self.hc_pre.use_cann_inverse_rms = is_950() and hasattr(torch_npu, "npu_rms_norm")
            if is_950() and get_current_vllm_config().model_config.enforce_eager:
                self.hc_pre.cann_pre_op = getattr(torch_npu, "npu_mhc_pre", None)
        if self.enable_ihc and is_950():
            self.cann_post_op = getattr(torch_npu, "npu_mhc_post", None)

    def configure_prefill_broadcast(self, embedding, token_budget):
        if not self.enable_ihc or self.hc_pre.cann_pre_op is None or self.hc_pre.magnitude != 2.0:
            return
        pre = self.hc_pre
        width, channels = self.config.hidden_size, pre.hc_mult
        weight = pre.hc_fn.weight
        if (
            channels != 4 or not 0 < width <= 4096 or width % 16
            or weight.shape != (2 * channels, channels * width) or weight.dtype != torch.float32
            or pre.hc_scale.shape != (2,) or pre.hc_scale.dtype != torch.float32
            or pre.hc_base.shape != (2 * channels,) or pre.hc_base.dtype != torch.float32
            or embedding.hy4_prefill_vocab_parameters() is None
            or embedding.weight.shape[1] != width or embedding.weight.dtype != torch.bfloat16
        ):
            return
        org_start, org_end, added_start, added_end, added_local, _ = embedding.hy4_prefill_vocab_parameters()
        vocab_size = embedding.num_embeddings
        ranges = [(org_start, org_end, 0), (added_start, added_end, added_local)]
        # CANN changes FP32 accumulation across batch shapes. Use the exact
        # native shape for each table and keep its original path for other
        # shapes. The scheduler budget and its halves cover full and
        # partially filled batches without depending on input lengths.
        rows = token_budget
        while rows > 4096:
            rows //= 2
        with torch.inference_mode():
            while 128 <= rows <= 4096:
                table = torch.zeros((vocab_size + 1, 2 * channels), dtype=torch.float32, device=weight.device)
                scratch = embedding.weight.new_zeros((rows, width))
                for first, last, local in ranges:
                    for start in range(first, last, rows):
                        count = min(rows, last - start)
                        offset = local + start - first
                        scratch[:count].copy_(embedding.weight[offset:offset + count])
                        self._write_broadcast_gates(pre, scratch, table[start:start + count])
                if embedding.tp_rank == 0:
                    scratch.zero_()
                    self._write_broadcast_gates(pre, scratch, table[vocab_size:vocab_size + 1])
                table = embedding._reduce_standard_embedding(table)
                name = f"_broadcast_gates_{rows}"
                pre.register_buffer(name, table, persistent=False)
                pre.broadcast_gate_buffers[rows] = name
                rows //= 2
        if pre.broadcast_gate_buffers:
            logger.info_once("HY4 Prefill first iHC uses native vocab gates for row counts %s",
                             tuple(pre.broadcast_gate_buffers))

    @staticmethod
    def _write_broadcast_gates(pre, scratch, destination):
        repeated = scratch.unsqueeze(1).repeat(1, pre.hc_mult, 1)
        _, post, _, _, _, gates = pre.cann_pre_op(
            repeated, pre.hc_fn.weight, pre.hc_scale, pre.hc_base,
            norm_eps=pre.layernorm_epsilon, hc_eps=pre.hc_eps, out_flag=1, inner_precise=0,
        )
        count = destination.shape[0]
        destination[:, :pre.hc_mult].copy_(gates[:count])
        destination[:, pre.hc_mult:].copy_(post[:count])

    def prepare_input(self, hidden_states):
        if (
            self.enable_ihc
            and hidden_states.ndim == 2
            and hidden_states.shape[0] in self.hc_pre.broadcast_gate_buffers
            and hidden_states.shape[-1] == self.config.hidden_size
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
            and is_forward_context_available()
        ):
            batch = get_flash_attention_metadata()
            token_ids = get_forward_context().additional_kwargs.get("hy4_first_ihc_token_ids")
            if (
                token_ids is not None and token_ids.ndim == 1 and token_ids.is_contiguous()
                and token_ids.device == hidden_states.device and token_ids.numel() == hidden_states.shape[0]
                and batch is not None and batch.prefill_reqs and all(batch.prefill_reqs)
            ):
                return hidden_states
        return super().prepare_input(hidden_states)

    def configure_decode_post(self):
        if not self.enable_ihc:
            return
        enabled = get_current_vllm_config().additional_config.get("hy4flash_fused_ihc_pre_tail", False)
        if not isinstance(enabled, bool):
            raise ValueError("hy4flash_fused_ihc_pre_tail must be a boolean")
        self.hc_pre.fuse_decode_tail = enabled
        if self.enable_ihc and is_950():
            self.fuse_decode_post = get_current_vllm_config().additional_config.get("hy4flash_fused_ihc_post", True)
            if not isinstance(self.fuse_decode_post, bool):
                raise ValueError("hy4flash_fused_ihc_post must be a boolean")
            self.cann_post_op = getattr(torch_npu, "npu_mhc_post", None)
            self.hc_pre.inverse_rms_workers = _decode_mean_workers(torch.npu.current_device())

    def post(self, output_with_bias, residual, post_gates):
        if self.enable_ihc and residual.ndim == 2 and residual.shape[0] in self.hc_pre.broadcast_gate_buffers:
            return first_ihc_post(output_with_bias, residual, post_gates)
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
            if not self.fuse_decode_post:
                scaled = output_with_bias.float().unsqueeze(1) * post_gates.unsqueeze(-1)
                return (residual.float() + scaled).to(residual.dtype)
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
        if not is_forward_context_available():
            # Deferred eager Prefill runs the head on selected rows in sampling.
            return super()._inverse_rms(x)
        return _inverse_rms(x, self.config.rms_norm_eps)
