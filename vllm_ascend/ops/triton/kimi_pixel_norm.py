# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPU-reference-exact pixel lookup on A3 and A5 vector cores."""

from dataclasses import dataclass

import torch
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_triton_helper import LaunchSpec, TritonWarmupTensor
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.triton.ascend_jit_warmup import AscendTritonJitKernel

PATCH_ELEMENTS = 588
SMALL_ELEMENTS = 196608
MEDIUM_ELEMENTS = 2408448


@triton.jit(do_not_specialize=["count"])
def _kimi_pixel_norm_kernel(
    pixels,
    lut,
    output,
    count,
    BLOCK: tl.constexpr,
):
    table_offsets = tl.arange(0, 256)
    table = tl.load(lut + table_offsets)
    lanes = tl.arange(0, BLOCK)
    for tile in range(tl.program_id(0), tl.cdiv(count, BLOCK), tl.num_programs(0)):
        offsets = tile * BLOCK + lanes
        valid = offsets < count
        # Preserve a supported vector Gather index path on the A5 backend.
        # All uint8 values are represented exactly by FP16 before widening.
        values = tl.load(pixels + offsets, mask=valid, other=0).to(tl.float16).to(tl.int32)
        result = tl.gather(table, values, axis=0)
        tl.store(output + offsets, result, mask=valid)


class KimiPixelNormKernel(AscendTritonJitKernel["KimiPixelNormKernel.CompileKey"]):
    kernel = _kimi_pixel_norm_kernel

    @dataclass(frozen=True)
    class CompileKey:
        dtype: torch.dtype
        block: int

    def get_warmup_keys(self, norm) -> list["KimiPixelNormKernel.CompileKey"]:
        # Expand after weight loading, using the module's final device/dtype.
        norm._refresh_table()
        return [self.CompileKey(norm.lut.dtype, block) for block in (1024, 4096, 8192)]

    def warmup_inputs(self, compile_key: "KimiPixelNormKernel.CompileKey") -> dict:
        return dict(
            pixels=TritonWarmupTensor(torch.uint8),
            lut=TritonWarmupTensor(compile_key.dtype, shape=(256,)),
            output=TritonWarmupTensor(compile_key.dtype),
            count=2,
            BLOCK=compile_key.block,
            programs=1,
        )

    @kernel_launcher
    def __call__(self, pixels, lut, output, count: int, BLOCK: int, programs: int) -> LaunchSpec:
        return (programs,), {}


_KIMI_PIXEL_NORM_KERNEL = KimiPixelNormKernel()


def kimi_pixel_norm_lut(pixels: torch.Tensor, lut: torch.Tensor, vector_cores: int) -> torch.Tensor:
    if pixels.device.type != "npu" or pixels.dtype != torch.uint8:
        raise ValueError("kimi_pixel_norm_lut requires uint8 NPU pixels")
    valid_shape = (pixels.ndim == 2 and pixels.shape[1] == PATCH_ELEMENTS) or (
        pixels.ndim == 4 and tuple(pixels.shape[1:]) == (3, 14, 14)
    )
    if not valid_shape or not pixels.is_contiguous():
        raise ValueError("expected contiguous [P,588] or [P,3,14,14] patches")
    if tuple(lut.shape) != (256,):
        raise ValueError("expected a [256] Kimi lookup table")
    if lut.device != pixels.device or not lut.is_contiguous():
        raise ValueError("lut must be contiguous on the pixels device")
    if lut.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("lut must be BF16 or FP16")
    count = pixels.numel()
    if count >= 2**31 - 1 or vector_cores <= 0:
        raise ValueError("invalid pixel count or vector core count")
    output = torch.empty_like(pixels, dtype=lut.dtype)
    if count == 0:
        return output
    block = 1024 if count <= SMALL_ELEMENTS else 4096 if count <= MEDIUM_ELEMENTS else 8192
    programs = min(triton.cdiv(count, block), vector_cores)
    with torch.npu.device(pixels.device):
        _KIMI_PIXEL_NORM_KERNEL(pixels=pixels, lut=lut, output=output, count=count, BLOCK=block, programs=programs)
    return output


def _kimi_pixel_norm_lut_fake(pixels: torch.Tensor, lut: torch.Tensor, vector_cores: int) -> torch.Tensor:
    return torch.empty_like(pixels, dtype=lut.dtype)


direct_register_custom_op(
    op_name="kimi_pixel_norm_lut",
    op_func=kimi_pixel_norm_lut,
    fake_impl=_kimi_pixel_norm_lut_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
