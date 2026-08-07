# SPDX-License-Identifier: Apache-2.0
"""Load DeepSeek V4 block-FP8 linears through Ascend 310P W8A8."""

from __future__ import annotations

from typing import Any

import torch
from vllm.utils.math_utils import cdiv

from vllm_ascend._310p.quantization.methods.w8a8_dynamic import (
    AscendW8A8DynamicLinearMethod310,
)
from vllm_ascend.utils import maybe_trans_nz


def _build_e4m3fn_table() -> tuple[float, ...]:
    values: list[float] = []
    for code in range(256):
        sign = -1.0 if code & 0x80 else 1.0
        exponent = (code >> 3) & 0x0F
        mantissa = code & 0x07
        if exponent == 0:
            value = sign * mantissa * (2.0**-9)
        elif exponent == 0x0F and mantissa == 0x07:
            value = float("nan")
        else:
            value = sign * (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
        values.append(value)
    return tuple(values)


_E4M3FN_VALUES = _build_e4m3fn_table()
_E4M3FN_TABLE_CACHE: dict[tuple[str, int | None], torch.Tensor] = {}


def decode_e4m3fn(weight: torch.Tensor) -> torch.Tensor:
    """Decode E4M3FN bytes without invoking an unsupported 310P FP8 cast."""
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError(f"Expected float8_e4m3fn weight, got {weight.dtype}.")
    codes = weight.view(torch.uint8).to(torch.long)
    key = (weight.device.type, weight.device.index)
    table = _E4M3FN_TABLE_CACHE.get(key)
    if table is None:
        table = torch.tensor(_E4M3FN_VALUES, dtype=torch.float32, device=weight.device)
        _E4M3FN_TABLE_CACHE[key] = table
    # The checkpoint is trusted and validated by safetensors. Avoid a per-chunk
    # ``isnan().any()`` host synchronization, which can trip the 310P AICPU
    # timeout after a long queued conversion sequence.
    return table[codes]


def requantize_block_fp8_to_int8(
    weight: torch.Tensor,
    block_scale: torch.Tensor,
    *,
    block_shape: tuple[int, int] = (128, 128),
    rows_per_chunk: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert block-scaled FP8 weights to symmetric per-row INT8."""
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2D linear weight, got {tuple(weight.shape)}.")
    if block_scale.ndim != 2:
        raise ValueError(f"Expected a 2D block scale, got {tuple(block_scale.shape)}.")
    if rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be positive.")

    output_size, input_size = weight.shape
    block_rows, block_cols = block_shape
    expected_scale_shape = (cdiv(output_size, block_rows), cdiv(input_size, block_cols))
    if tuple(block_scale.shape) != expected_scale_shape:
        raise ValueError(f"Expected FP8 block scale shape {expected_scale_shape}, got {tuple(block_scale.shape)}.")

    quantized = torch.empty((output_size, input_size), dtype=torch.int8, device=weight.device)
    row_scales = torch.empty((output_size,), dtype=torch.float32, device=weight.device)

    for start in range(0, output_size, rows_per_chunk):
        end = min(start + rows_per_chunk, output_size)
        row_block_indices = torch.arange(start, end, device=weight.device) // block_rows
        chunk_block_scales = block_scale[row_block_indices].to(torch.float32)
        expanded_scales = chunk_block_scales.repeat_interleave(block_cols, dim=-1)[..., :input_size]
        dequantized = decode_e4m3fn(weight[start:end]) * expanded_scales

        max_abs = dequantized.abs().amax(dim=-1, keepdim=True)
        scale = max_abs / 127.0
        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        qweight = torch.round(dequantized / safe_scale).clamp_(-127, 127).to(torch.int8)
        quantized[start:end].copy_(qweight)
        row_scales[start:end].copy_(safe_scale.squeeze(-1))

    return quantized, row_scales


class AscendFP8ToW8A8DynamicLinearMethod310(AscendW8A8DynamicLinearMethod310):
    """Checkpoint-compatible DeepSeek FP8 linear executed as 310P W8A8."""

    def __init__(self, quant_config: dict[str, Any]):
        super().__init__()
        block_shape = quant_config.get("weight_block_size", [128, 128])
        if len(block_shape) != 2:
            raise ValueError(f"Expected two FP8 block dimensions, got {block_shape}.")
        self.block_shape = (int(block_shape[0]), int(block_shape[1]))

    def get_weight(
        self,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype = torch.float16,
    ) -> dict[str, Any]:
        return {"weight": torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn)}

    def get_perchannel_param(self, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        return {}

    def get_pergroup_param(
        self,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        layer_type: str | None = None,
    ) -> dict[str, Any]:
        block_rows, block_cols = self.block_shape
        return {
            "weight_scale": torch.empty(
                cdiv(output_size, block_rows),
                cdiv(input_size, block_cols),
                dtype=torch.float32,
            ),
            "_packed_dim": 0,
            "_packed_factor": block_rows,
        }

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_data = layer.weight.data
        scale_data = layer.weight_scale.data
        singleton_input_major = weight_data.ndim == 3 and weight_data.shape[0] == 1
        if singleton_input_major:
            # DeepSeek V4 O-LoRA and related packed linears store a singleton
            # shard as [1, input_size, output_size], while W8A8 conversion
            # expects the canonical [output_size, input_size] matrix.
            weight_data = weight_data.squeeze(0).transpose(0, 1).contiguous()
            if scale_data.ndim == 3 and scale_data.shape[0] == 1:
                scale_data = scale_data.squeeze(0)

        if weight_data.ndim == 2 and scale_data.ndim == 2:
            block_rows, block_cols = self.block_shape
            expected_scale_shape = (
                cdiv(weight_data.shape[0], block_rows),
                cdiv(weight_data.shape[1], block_cols),
            )
            if tuple(scale_data.shape) == expected_scale_shape[::-1]:
                # Packed DeepSeek V4 linears may store block scales in
                # input-block-major order. Normalize to output-block-major.
                scale_data = scale_data.transpose(0, 1).contiguous()

        weight, scale = requantize_block_fp8_to_int8(
            weight_data,
            scale_data,
            block_shape=self.block_shape,
        )
        # Match the native 310P W8A8 orientation and NZ representation.
        layer.weight.data = maybe_trans_nz(weight).transpose(0, 1)
        layer.weight_scale.data = scale
        layer.weight_scale_fp32 = scale
