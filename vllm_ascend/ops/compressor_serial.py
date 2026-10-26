# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""A5 serial compressor: AscendC cast, torch.matmul, then AscendC vectors.

Prepare once before graph capture. A plan owns its scratch buffers and output;
use a separate plan for each concurrent stream. Weights must remain unchanged
after preparation. Device metadata tensors may be updated in place for replay.
"""

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class CompressorSerialConfig:
    """Launch choices, exposed for reproducible performance experiments."""

    cast_blocks: int = 48
    pool_blocks: int = 48
    final_blocks: int = 48
    pool_threads: int = 256
    weight_layout: str = "nk"

    def __post_init__(self):
        if self.weight_layout not in ("kn", "nk"):
            raise ValueError("weight_layout must be kn or nk")
        for blocks in (self.cast_blocks, self.pool_blocks, self.final_blocks):
            if not 1 <= blocks <= 72:
                raise ValueError("block counts must be in [1, 72]")
        if self.pool_threads not in (128, 256, 512):
            raise ValueError("pool_threads must be 128, 256 or 512")


def _ensure_extension_loaded():
    if hasattr(torch.ops._C_ascend, "compressor_serial_cast"):
        return
    library = Path(__file__).resolve().parents[1] / "libcompressor_serial_ops.so"
    if not library.is_file():
        raise RuntimeError("Build the A5 extension with tools/build_compressor_serial.sh and install it first")
    torch.ops.load_library(str(library))


class CompressorSerialPlan:
    """Prepared, allocation-free forward path for torch.npu.NPUGraph.

    The public compressor argument order is preserved by ``prepare``. The
    matmul output stays FP32, including the values persisted to state_cache.
    The caller controls torch.npu.matmul.allow_hf32; this module never changes
    process-wide precision flags.
    """

    def __init__(self, tensors, params, weight, config, x):
        self.tensors = tensors
        self.params = params
        self.weight = weight
        self.config = config
        self.shape = x.shape
        self.dtype = x.dtype
        self.device = x.device
        self.x_float = torch.empty_like(x, dtype=torch.float32)
        self.projection = torch.empty(
            (x.numel() // x.shape[-1], weight.shape[1]),
            dtype=torch.float32,
            device=x.device,
        )
        self.output = torch.empty(
            (*tensors[3].shape[:-1], tensors[2].numel()),
            dtype=x.dtype,
            device=x.device,
        )
        self.pooled = torch.empty_like(self.output, dtype=torch.float32)

    def cast(self, x):
        torch.ops._C_ascend.compressor_serial_cast(x, self.x_float, self.config.cast_blocks)

    def matmul(self):
        torch.matmul(
            self.x_float.view(-1, self.shape[-1]),
            self.weight,
            out=self.projection,
        )

    def vector(self, stage):
        blocks = self.config.pool_blocks if stage == 0 else self.config.final_blocks
        torch.ops._C_ascend.compressor_serial_vector(
            self.projection,
            *self.tensors,
            self.pooled,
            self.output,
            self.params,
            stage,
            blocks,
            self.config.pool_threads,
        )

    def __call__(self, x):
        if x.shape != self.shape or x.dtype != self.dtype or x.device != self.device:
            raise ValueError("x must match the shape, dtype and device used by prepare")
        if not x.is_contiguous():
            raise ValueError("x must be contiguous")
        self.cast(x)
        self.matmul()
        # A task boundary keeps history reads ahead of all cache writes.
        self.vector(0)
        self.vector(1)
        return self.output

    @classmethod
    def prepare(
        cls,
        x,
        wkv,
        wgate,
        state_cache,
        ape,
        norm_weight,
        rope_sin,
        rope_cos,
        state_block_table=None,
        cu_seqlens=None,
        seqused=None,
        start_pos=None,
        rope_head_dim=64,
        cmp_ratio=1,
        coff=1,
        norm_eps=1e-6,
        rotary_mode=2,
        cache_mode=1,
        *,
        config=None,
    ):
        _ensure_extension_loaded()
        # Validate host-visible shape metadata only. No device reads or item().
        if config is None:
            decode = x.numel() // x.shape[-1] < 32
            config = CompressorSerialConfig(4 if decode else 64, 64, 16 if decode else 64, 128 if decode else 512)
        if x.device.type not in ("npu", "privateuseone"):
            raise ValueError("A5 NPU tensors are required")
        if x.ndim not in (2, 3) or x.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("x must be BF16/FP16 [T,H] or [B,S,H]")
        if coff not in (1, 2) or cache_mode not in (1, 2):
            raise ValueError("coff and cache_mode must be 1 or 2")
        if cmp_ratio not in (2, 4, 8, 16, 32, 64, 128):
            raise ValueError("unsupported compression ratio")
        if rotary_mode not in (1, 2) or rope_head_dim != 64:
            raise ValueError("only 64-wide half/interleave RoPE is supported")
        if not math.isfinite(norm_eps) or norm_eps <= 0:
            raise ValueError("norm_eps must be positive and finite")
        hidden, dim = x.shape[-1], norm_weight.numel()
        if dim not in (128, 512) or hidden < 1024 or hidden > 10240 or hidden % 512:
            raise ValueError("unsupported hidden or head dimension")
        if wkv.shape != (coff * dim, hidden) or wgate.shape != wkv.shape:
            raise ValueError("weights must have shape [coff*D,H]")
        if wkv.dtype != x.dtype or wgate.dtype != x.dtype:
            raise ValueError("weights and x must have the same dtype")
        if norm_weight.shape != (dim,) or ape.shape != (cmp_ratio, coff * dim):
            raise ValueError("invalid norm_weight or ape shape")
        if state_cache.ndim != 3 or state_cache.shape[-1] != 2 * coff * dim:
            raise ValueError("state_cache must have shape [blocks,block_size,2*coff*D]")
        if state_cache.stride(-1) != 1 or state_cache.stride(-2) != 2 * coff * dim:
            raise ValueError("only cache stride(0) padding is supported")
        if state_cache.stride(0) < state_cache.shape[1] * state_cache.shape[2]:
            raise ValueError("overlapping cache blocks are unsupported")
        if x.ndim == 2:
            if cu_seqlens is None or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
                raise ValueError("TND requires cu_seqlens[B+1]")
            batch, sequence = cu_seqlens.numel() - 1, 0
            if rope_sin.ndim != 2:
                raise ValueError("TND requires rope_sin[output_rows,64]")
        else:
            batch, sequence = x.shape[:2]
            if rope_sin.shape != (batch, (sequence + cmp_ratio - 1) // cmp_ratio, 64):
                raise ValueError("invalid BSH RoPE shape")
            if cu_seqlens is not None:
                raise ValueError("cu_seqlens is only supported for TND")
        if rope_sin.shape[-1] != 64 or rope_cos.shape != rope_sin.shape:
            raise ValueError("invalid RoPE shape")
        tensors = (x, wkv, wgate, state_cache, ape, norm_weight, rope_sin, rope_cos)
        for tensor in tensors:
            if tensor.device != x.device:
                raise ValueError("all tensors must be on the same NPU")
            if tensor is not state_cache and not tensor.is_contiguous():
                raise ValueError("inputs other than state_cache must be contiguous")
        for tensor in tensors[3:]:
            if tensor.dtype != torch.float32:
                raise ValueError("cache, bias, norm and RoPE tensors must be FP32")
        optionals = (state_block_table, cu_seqlens, seqused, start_pos)
        flags = sum(1 << i for i, tensor in enumerate(optionals) if tensor is not None)
        for tensor in optionals:
            if tensor is not None and (
                tensor.dtype != torch.int32 or tensor.device != x.device or not tensor.is_contiguous()
            ):
                raise ValueError("metadata must be contiguous INT32 on the same NPU")
        for tensor in (seqused, start_pos):
            if tensor is not None and tensor.shape != (batch,):
                raise ValueError("seqused/start_pos must have shape [B]")
        table_width = state_block_table.shape[-1] if state_block_table is not None else 1
        if state_block_table is not None:
            if cache_mode == 1 and (state_block_table.ndim != 2 or state_block_table.shape[0] != batch):
                raise ValueError("paged mode requires block_table[B,pages]")
            if cache_mode == 2 and state_block_table.shape != (batch,):
                raise ValueError("ring mode requires block_table[B]")
        if state_cache.shape[0] == 0 or state_cache.shape[1] == 0:
            raise ValueError("cache dimensions must be positive")
        if cache_mode == 2 and sequence and state_cache.shape[1] < coff * cmp_ratio + sequence - 1:
            raise ValueError("ring buffer is too small for the sequence")
        # Use 32-bit device indices only when every buffer offset fits.
        max_index = 2**31 - 1
        if (
            max(
                state_cache.shape[0] * state_cache.stride(0),
                x.numel() // hidden * (2 * coff * dim),
                rope_sin.numel() // 64 * dim,
            )
            > max_index
        ):
            raise ValueError("serial A5 buffer offsets must fit signed 32-bit indices")
        dummy = torch.empty(0, device=x.device, dtype=torch.int32)
        metadata = tuple(dummy if tensor is None else tensor for tensor in optionals)
        params = torch.tensor(
            [
                batch,
                sequence,
                x.numel() // hidden,
                dim,
                cmp_ratio,
                coff,
                cache_mode,
                state_cache.shape[1],
                state_cache.stride(0),
                state_cache.shape[0],
                table_width,
                rope_sin.numel() // 64,
                x.ndim,
                rotary_mode,
                flags,
                struct.unpack("I", struct.pack("f", norm_eps))[0],
                2 * coff * dim,
                coff * dim,
            ]
            + [0] * 6
            + [4 * ((j ^ 1) if rotary_mode == 2 else (j + 32) % 64) for j in range(64)]
            + [
                struct.unpack("i", struct.pack("f", -1.0 if (j % 2 == 0 if rotary_mode == 2 else j < 32) else 1.0))[0]
                for j in range(64)
            ],
            dtype=torch.int32,
            device=x.device,
        )
        # Static preparation is outside capture; per-replay X casting is timed.
        weight = torch.cat((wkv, wgate), dim=0).float().t()
        if config.weight_layout == "kn":
            weight = weight.contiguous()
        args = (state_cache, ape, norm_weight, rope_sin, rope_cos, *metadata)
        return cls(args, params, weight, config, x)
