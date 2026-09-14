import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


@triton.jit
def _engram_int8_dequant_kernel(codes_ptr, scales_ptr, output_ptr, rows, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    if row >= rows:
        return
    offsets = tl.arange(0, WIDTH)
    codes = tl.load(codes_ptr + row * WIDTH + offsets).to(tl.float32)
    scales = tl.load(scales_ptr + row * (WIDTH // 32) + offsets // 32)
    tl.store(output_ptr + row * WIDTH + offsets, (codes * scales).to(tl.bfloat16))


@triton.jit
def _engram_int8_gather_dequant_kernel(weight_ptr, scale_ptr, ids_ptr, output_ptr, rows, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    if row >= rows:
        return
    offsets = tl.arange(0, WIDTH)
    source_row = tl.load(ids_ptr + row)
    codes = tl.load(weight_ptr + source_row * WIDTH + offsets).to(tl.float32)
    scales = tl.load(scale_ptr + source_row * (WIDTH // 32) + offsets // 32)
    tl.store(output_ptr + row * WIDTH + offsets, (codes * scales).to(tl.bfloat16))


def dequantize_engram_int8(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if codes.device.type != "npu" or codes.dtype != torch.int8 or scales.dtype != torch.float32:
        raise ValueError("Engram Triton INT8 dequant expects NPU int8 codes and FP32 scales")
    if codes.ndim != 2 or codes.shape[1] != 256 or scales.shape != (codes.shape[0], 8):
        raise ValueError("Engram Triton INT8 dequant expects [rows, 256] and [rows, 8]")
    init_device_properties_triton()
    output = torch.empty(codes.shape, dtype=torch.bfloat16, device=codes.device)
    _engram_int8_dequant_kernel[(codes.shape[0],)](codes, scales, output, codes.shape[0], WIDTH=256, num_warps=4)
    return output


def gather_dequantize_engram_int8(weight: torch.Tensor, scales: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    if (
        weight.device.type != "npu"
        or weight.dtype != torch.int8
        or scales.dtype != torch.float32
        or ids.device.type != "npu"
        or ids.dtype != torch.int64
    ):
        raise ValueError("Engram fused INT8 gather expects NPU int8/FP32/int64 tensors")
    if weight.ndim != 2 or weight.shape[1] != 256 or scales.shape != (weight.shape[0], 8):
        raise ValueError("Engram fused INT8 gather expects table [rows, 256] and [rows, 8] scales")
    if ids.ndim != 1:
        raise ValueError("Engram fused INT8 gather expects flat IDs")
    if ids.numel() == 0:
        return torch.empty((0, 256), dtype=torch.bfloat16, device=weight.device)
    init_device_properties_triton()
    output = torch.empty((ids.shape[0], 256), dtype=torch.bfloat16, device=weight.device)
    _engram_int8_gather_dequant_kernel[(ids.shape[0],)](
        weight, scales, ids, output, ids.shape[0], WIDTH=256, num_warps=4
    )
    return output
