# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-head INT8 query quantization with the FP16 scales consumed by QLI."""

from dataclasses import dataclass
from typing import Any

import torch
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
)
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.ascend_jit_warmup import AscendTritonJitKernel
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit(do_not_specialize=["num_rows", "blocks_per_core"])
def _quantize_indexer_query_kernel(
    query_ptr,
    quantized_ptr,
    scale_ptr,
    num_rows,
    blocks_per_core,
    BLOCK_ROWS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    QUANT_MAX: tl.constexpr,
    MIN_SCALE: tl.constexpr,
):
    first_block = tl.program_id(0) * blocks_per_core
    columns = tl.arange(0, HEAD_DIM)
    for block in range(blocks_per_core):
        rows = (first_block + block) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        offsets = rows[:, None] * HEAD_DIM + columns[None, :]
        query = tl.load(query_ptr + offsets, rows[:, None] < num_rows, other=0).to(tl.float32)
        abs_max = tl.max(tl.abs(query), axis=1)
        # Quantize with the rounded FP16 scale, not the original FP32 value.
        scale = tl.div_rn(abs_max, QUANT_MAX).to(tl.float16).to(tl.float32)
        scale = tl.maximum(scale, MIN_SCALE)
        normalized = tl.div_rn(query, scale[:, None])
        quantized = tl.extra.cann.libdevice.nearbyint(normalized)
        quantized = tl.minimum(tl.maximum(quantized, -QUANT_MAX), QUANT_MAX).to(tl.int8)
        tl.store(quantized_ptr + offsets, quantized, rows[:, None] < num_rows)
        tl.store(scale_ptr + rows, scale, rows < num_rows)


class QuantizeIndexerQueryKernel(AscendTritonJitKernel["QuantizeIndexerQueryKernel.CompileKey"]):
    BLOCK_ROWS = 16
    kernel = _quantize_indexer_query_kernel

    @dataclass(frozen=True)
    class CompileKey:
        dtype: torch.dtype
        head_dim: int

    def dispatch(self, *, dtype: torch.dtype, head_dim: int) -> "QuantizeIndexerQueryKernel.CompileKey":
        return self.CompileKey(dtype=dtype, head_dim=head_dim)

    def get_warmup_keys(self, vllm_config: Any) -> list["QuantizeIndexerQueryKernel.CompileKey"]:
        # BLOCK_ROWS / QUANT_MAX / MIN_SCALE are fixed; num_rows and
        # blocks_per_core are do_not_specialize, so the model dtype and the
        # indexer head dim fully determine the JIT key.
        head_dim = int(getattr(vllm_config.model_config.hf_text_config, "index_head_dim", 128))
        return self._trace_dispatch(self.dispatch)(
            dtype=vllm_config.model_config.dtype,
            head_dim=head_dim,
        )

    def warmup_inputs(self, compile_key: "QuantizeIndexerQueryKernel.CompileKey") -> dict[str, Any]:
        shape = (1, 1, compile_key.head_dim)
        strides = (compile_key.head_dim, compile_key.head_dim, 1)
        return dict(
            query=TritonWarmupTensor(compile_key.dtype, shape=shape, strides=strides),
            quantized=TritonWarmupTensor(torch.int8, shape=shape, strides=strides),
            scale=TritonWarmupTensor(torch.float16, shape=(1, 1)),
            num_rows=1,
            blocks_per_core=1,
            grid_size=1,
        )

    @kernel_launcher
    def __call__(
        self,
        query: torch.Tensor,
        quantized: torch.Tensor,
        scale: torch.Tensor,
        *,
        num_rows: int,
        blocks_per_core: int,
        grid_size: int,
    ) -> LaunchSpec:
        return (grid_size,), dict(
            BLOCK_ROWS=self.BLOCK_ROWS,
            HEAD_DIM=query.shape[-1],
            QUANT_MAX=127.0,
            MIN_SCALE=2.0**-24,
        )


_QUANTIZE_INDEXER_QUERY_KERNEL = QuantizeIndexerQueryKernel()


def quantize_indexer_query(query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize [tokens, heads, 128] queries to INT8 and FP16 per-head scales.

    Match the indexer's round-to-even and symmetric [-127, 127] range. Zero
    heads use the smallest positive FP16 subnormal so division stays defined.
    """
    assert query.ndim == 3 and query.shape[-1] == 128
    query = query.contiguous()
    quantized = torch.empty_like(query, dtype=torch.int8)
    scale = torch.empty(query.shape[:-1], dtype=torch.float16, device=query.device)
    num_rows = query.numel() // query.shape[-1]
    if num_rows == 0:
        return quantized, scale

    # Each core writes whole 32-byte groups of FP16 scales. Contiguous blocks
    # keep neighboring cores from racing on a partial output cache line.
    block_rows = 16
    num_blocks = triton.cdiv(num_rows, block_rows)
    grid = min(num_blocks, get_vectorcore_num())
    _QUANTIZE_INDEXER_QUERY_KERNEL(
        query=query,
        quantized=quantized,
        scale=scale,
        num_rows=num_rows,
        blocks_per_core=triton.cdiv(num_blocks, grid),
        grid_size=grid,
    )
    return quantized, scale
