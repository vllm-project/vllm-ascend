# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import torch
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_triton_helper import LaunchSpec, TritonWarmupTensor
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.ascend_jit_warmup import AscendTritonJitKernel

MHC_FUSION_HIDDEN_SIZE = 5120
_POST_TILE_SIZE = 2048


@triton.jit
def _post_collapse_tile(
    sublayer_out,
    residual,
    post,
    comb,
    pre_mix,
    updated,
    columns,
    HIDDEN_SIZE: tl.constexpr,
):
    z = tl.load(sublayer_out + columns).to(tl.float32)
    r0 = tl.load(residual + columns).to(tl.float32)
    r1 = tl.load(residual + HIDDEN_SIZE + columns).to(tl.float32)
    r2 = tl.load(residual + 2 * HIDDEN_SIZE + columns).to(tl.float32)
    r3 = tl.load(residual + 3 * HIDDEN_SIZE + columns).to(tl.float32)
    merged = tl.full(columns.shape, 0.0, tl.float32)
    for j in tl.static_range(4):
        # HcPost starts from z * post[j] and adds each residual in order.
        value = z * tl.load(post + j)
        value += r0 * tl.load(comb + j)
        value += r1 * tl.load(comb + 4 + j)
        value += r2 * tl.load(comb + 8 + j)
        value += r3 * tl.load(comb + 12 + j)
        rounded = value.to(tl.bfloat16)
        tl.store(updated + j * HIDDEN_SIZE + columns, rounded)
        mix = tl.load(pre_mix + j)
        if j == 0:
            merged = rounded.to(tl.float32) * mix
        else:
            merged = tl.fma(rounded.to(tl.float32), mix, merged)
    # Match the BF16 input that the original HcPre writes for RMSNorm.
    return merged.to(tl.bfloat16)


@triton.jit
def _store_normalized(row, weight, normalized, normalized_fp32, columns, rstd, WRITE_FP32: tl.constexpr):
    gamma = tl.load(weight + columns).to(tl.float32)
    result = ((row * rstd) * gamma).to(tl.bfloat16)
    tl.store(normalized + columns, result)
    if WRITE_FP32:
        tl.store(normalized_fp32 + columns, result.to(tl.float32))


@triton.jit
def _mhc_post_collapse_norm_kernel(
    sublayer_out,
    residual,
    post,
    comb,
    pre_mix,
    weight,
    updated,
    normalized,
    normalized_fp32,
    epsilon,
    HIDDEN_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    WRITE_FP32: tl.constexpr,
):
    token = tl.program_id(0)
    sublayer_out += token * HIDDEN_SIZE
    residual += token * 4 * HIDDEN_SIZE
    updated += token * 4 * HIDDEN_SIZE
    post += token * 4
    comb += token * 16
    pre_mix += token * 4
    normalized += token * HIDDEN_SIZE
    normalized_fp32 += token * HIDDEN_SIZE

    col0 = tl.arange(0, TILE_SIZE)
    col1 = TILE_SIZE + tl.arange(0, TILE_SIZE)
    col2 = 2 * TILE_SIZE + tl.arange(0, HIDDEN_SIZE - 2 * TILE_SIZE)
    y0 = _post_collapse_tile(sublayer_out, residual, post, comb, pre_mix, updated, col0, HIDDEN_SIZE).to(tl.float32)
    y1 = _post_collapse_tile(sublayer_out, residual, post, comb, pre_mix, updated, col1, HIDDEN_SIZE).to(tl.float32)
    y2 = _post_collapse_tile(sublayer_out, residual, post, comb, pre_mix, updated, col2, HIDDEN_SIZE).to(tl.float32)
    sum_sq = tl.sum(y0 * y0, axis=0) + tl.sum(y1 * y1, axis=0) + tl.sum(y2 * y2, axis=0)
    rstd = tl.rsqrt(sum_sq / HIDDEN_SIZE + epsilon)
    _store_normalized(y0, weight, normalized, normalized_fp32, col0, rstd, WRITE_FP32)
    _store_normalized(y1, weight, normalized, normalized_fp32, col1, rstd, WRITE_FP32)
    _store_normalized(y2, weight, normalized, normalized_fp32, col2, rstd, WRITE_FP32)


class MhcPostCollapseNormKernel(AscendTritonJitKernel[tuple[torch.dtype, bool]]):
    kernel = _mhc_post_collapse_norm_kernel

    def get_warmup_keys(self, weight_dtype: torch.dtype, write_fp32: bool) -> list[tuple[torch.dtype, bool]]:
        return [(weight_dtype, write_fp32)]

    def warmup_inputs(self, compile_key: tuple[torch.dtype, bool]) -> dict:
        weight_dtype, write_fp32 = compile_key
        hidden_size = MHC_FUSION_HIDDEN_SIZE
        return dict(
            sublayer_out=TritonWarmupTensor(torch.bfloat16, shape=(1, hidden_size)),
            residual=TritonWarmupTensor(torch.bfloat16, shape=(1, 4, hidden_size)),
            post=TritonWarmupTensor(torch.float32, shape=(1, 4)),
            comb=TritonWarmupTensor(torch.float32, shape=(1, 4, 4)),
            pre_mix=TritonWarmupTensor(torch.float32, shape=(1, 4)),
            weight=TritonWarmupTensor(weight_dtype, shape=(hidden_size,)),
            updated=TritonWarmupTensor(torch.bfloat16, shape=(1, 4, hidden_size)),
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
            WRITE_FP32=write_fp32,
            enable_fp_fusion=False,
        )


MHC_POST_COLLAPSE_NORM = MhcPostCollapseNormKernel()
