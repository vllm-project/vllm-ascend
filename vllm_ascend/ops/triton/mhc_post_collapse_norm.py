# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import torch
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_triton_helper import LaunchSpec, TritonWarmupTensor
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.ascend_jit_warmup import AscendTritonJitKernel

MHC_FUSION_HIDDEN_SIZE = 5120
_HC_MULT = 4
_POST_TILE_SIZE = 2048


@triton.jit
def _post_collapse_tile(
    sublayer_out_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    pre_mix_ptr,
    updated_ptr,
    columns,
    HIDDEN_SIZE: tl.constexpr,
    HC_MULT: tl.constexpr,
):
    z = tl.load(sublayer_out_ptr + columns).to(tl.float32)
    r0 = tl.load(residual_ptr + columns).to(tl.float32)
    r1 = tl.load(residual_ptr + HIDDEN_SIZE + columns).to(tl.float32)
    r2 = tl.load(residual_ptr + 2 * HIDDEN_SIZE + columns).to(tl.float32)
    r3 = tl.load(residual_ptr + 3 * HIDDEN_SIZE + columns).to(tl.float32)
    merged = tl.full(columns.shape, 0.0, tl.float32)
    for j in tl.static_range(HC_MULT):
        # HcPost starts from z * post[j] and adds each residual in order.
        value = z * tl.load(post_ptr + j)
        value += r0 * tl.load(comb_ptr + j)
        value += r1 * tl.load(comb_ptr + HC_MULT + j)
        value += r2 * tl.load(comb_ptr + 2 * HC_MULT + j)
        value += r3 * tl.load(comb_ptr + 3 * HC_MULT + j)
        rounded = value.to(tl.bfloat16)
        tl.store(updated_ptr + j * HIDDEN_SIZE + columns, rounded)
        mix = tl.load(pre_mix_ptr + j)
        if j == 0:
            merged = rounded.to(tl.float32) * mix
        else:
            merged = tl.fma(rounded.to(tl.float32), mix, merged)
    # Match the BF16 input that the original HcPre writes for RMSNorm.
    return merged.to(tl.bfloat16)


@triton.jit
def _store_normalized(row, weight_ptr, normalized_ptr, normalized_fp32_ptr, columns, rstd, WRITE_FP32: tl.constexpr):
    gamma = tl.load(weight_ptr + columns).to(tl.float32)
    result = ((row * rstd) * gamma).to(tl.bfloat16)
    tl.store(normalized_ptr + columns, result)
    if WRITE_FP32:
        tl.store(normalized_fp32_ptr + columns, result.to(tl.float32))


@triton.jit
def _mhc_post_collapse_norm_kernel(
    sublayer_out_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    pre_mix_ptr,
    weight_ptr,
    updated_ptr,
    normalized_ptr,
    normalized_fp32_ptr,
    epsilon,
    HIDDEN_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HC_MULT: tl.constexpr,
    WRITE_FP32: tl.constexpr,
):
    token = tl.program_id(0)
    sublayer_out_ptr += token * HIDDEN_SIZE
    residual_ptr += token * HC_MULT * HIDDEN_SIZE
    updated_ptr += token * HC_MULT * HIDDEN_SIZE
    post_ptr += token * HC_MULT
    comb_ptr += token * (HC_MULT * HC_MULT)
    pre_mix_ptr += token * HC_MULT
    normalized_ptr += token * HIDDEN_SIZE
    normalized_fp32_ptr += token * HIDDEN_SIZE

    # Cover width 5120 with three power-of-two tiles and no masked tail.
    col0 = tl.arange(0, TILE_SIZE)
    col1 = TILE_SIZE + tl.arange(0, TILE_SIZE)
    col2 = 2 * TILE_SIZE + tl.arange(0, HIDDEN_SIZE - 2 * TILE_SIZE)
    y0 = _post_collapse_tile(
        sublayer_out_ptr, residual_ptr, post_ptr, comb_ptr, pre_mix_ptr, updated_ptr, col0, HIDDEN_SIZE, HC_MULT
    ).to(tl.float32)
    y1 = _post_collapse_tile(
        sublayer_out_ptr, residual_ptr, post_ptr, comb_ptr, pre_mix_ptr, updated_ptr, col1, HIDDEN_SIZE, HC_MULT
    ).to(tl.float32)
    y2 = _post_collapse_tile(
        sublayer_out_ptr, residual_ptr, post_ptr, comb_ptr, pre_mix_ptr, updated_ptr, col2, HIDDEN_SIZE, HC_MULT
    ).to(tl.float32)
    sum_sq = tl.sum(y0 * y0, axis=0) + tl.sum(y1 * y1, axis=0) + tl.sum(y2 * y2, axis=0)
    rstd = tl.rsqrt(sum_sq / HIDDEN_SIZE + epsilon)
    _store_normalized(y0, weight_ptr, normalized_ptr, normalized_fp32_ptr, col0, rstd, WRITE_FP32)
    _store_normalized(y1, weight_ptr, normalized_ptr, normalized_fp32_ptr, col1, rstd, WRITE_FP32)
    _store_normalized(y2, weight_ptr, normalized_ptr, normalized_fp32_ptr, col2, rstd, WRITE_FP32)


class MhcPostCollapseNormKernel(AscendTritonJitKernel[tuple[torch.dtype, bool]]):
    kernel = _mhc_post_collapse_norm_kernel

    def get_warmup_keys(self, weight_dtype: torch.dtype, write_fp32: bool) -> list[tuple[torch.dtype, bool]]:
        return [(weight_dtype, write_fp32)]

    def warmup_inputs(self, compile_key: tuple[torch.dtype, bool]) -> dict:
        weight_dtype, write_fp32 = compile_key
        hidden_size = MHC_FUSION_HIDDEN_SIZE
        return dict(
            sublayer_out=TritonWarmupTensor(torch.bfloat16, shape=(1, hidden_size)),
            residual=TritonWarmupTensor(torch.bfloat16, shape=(1, _HC_MULT, hidden_size)),
            post=TritonWarmupTensor(torch.float32, shape=(1, _HC_MULT)),
            comb=TritonWarmupTensor(torch.float32, shape=(1, _HC_MULT, _HC_MULT)),
            pre_mix=TritonWarmupTensor(torch.float32, shape=(1, _HC_MULT)),
            weight=TritonWarmupTensor(weight_dtype, shape=(hidden_size,)),
            updated=TritonWarmupTensor(torch.bfloat16, shape=(1, _HC_MULT, hidden_size)),
            normalized=TritonWarmupTensor(torch.bfloat16, shape=(1, hidden_size)),
            normalized_fp32=TritonWarmupTensor(torch.float32 if write_fp32 else torch.bfloat16, shape=(1, hidden_size)),
            epsilon=1e-6,
            write_fp32=write_fp32,
        )

    @kernel_launcher
    def __call__(
        self,
        sublayer_out: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        pre_mix: torch.Tensor,
        weight: torch.Tensor,
        updated: torch.Tensor,
        normalized: torch.Tensor,
        normalized_fp32: torch.Tensor,
        epsilon: float,
        write_fp32: bool,
    ) -> LaunchSpec:
        return (residual.shape[0],), dict(
            HIDDEN_SIZE=MHC_FUSION_HIDDEN_SIZE,
            TILE_SIZE=_POST_TILE_SIZE,
            HC_MULT=_HC_MULT,
            WRITE_FP32=write_fp32,
            # Request separate multiply and add for HcPost.
            enable_fp_fusion=False,
        )


MHC_POST_COLLAPSE_NORM = MhcPostCollapseNormKernel()


def mhc_post_collapse_norm(
    sublayer_out: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre_mix: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    *,
    write_fp32: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare a DeepSeek V4.1 mHC boundary without modifying its inputs."""
    updated = torch.empty_like(residual)
    normalized = torch.empty_like(sublayer_out, memory_format=torch.contiguous_format)
    routing_input = normalized.new_empty(normalized.shape, dtype=torch.float32) if write_fp32 else normalized
    MHC_POST_COLLAPSE_NORM(
        sublayer_out.contiguous(),
        residual,
        post,
        comb,
        pre_mix,
        weight,
        updated,
        normalized,
        routing_input,
        epsilon,
        write_fp32,
    )
    return updated, normalized, routing_input
