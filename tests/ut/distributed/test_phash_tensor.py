# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import math
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.distributed.ec_transfer.phash import PHASH_NUM_BANDS, _dct_matrix, bands, compute_phash
from vllm_ascend.distributed.ec_transfer.ssim import _gaussian_kernel, ssim_score


def _reference_phash(resized: torch.Tensor, grid: tuple[int, int, int]) -> int:
    """Original float32 CPU algorithm, including its bit order."""
    t, gh, gw = grid
    x = resized.float().reshape(t, gh // 2, gw // 2, 2, 2, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(t, gh, gw, -1)
    gray = x.mean(dim=-1).mean(dim=0, keepdim=True)
    img = F.interpolate(gray.unsqueeze(0), size=(32, 32), mode="bilinear", align_corners=False)[0, 0]
    k = torch.arange(32, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(32, dtype=torch.float32).unsqueeze(0)
    matrix = torch.cos(math.pi / 32 * (i + 0.5) * k)
    matrix[0] *= 1.0 / math.sqrt(2.0)
    low = (matrix @ img @ matrix.T)[:8, :8].flatten()
    median = low.median()
    bits = (low > median) & ((low - median).abs() > 0.05 * low.std())
    return int("".join("1" if bit else "0" for bit in bits.tolist()), 2)


@pytest.mark.parametrize("grid", [(1, 4, 4), (1, 8, 12), (2, 6, 8)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_phash_matches_original_float32_definition(grid, dtype):
    generator = torch.Generator().manual_seed(123)
    resized = torch.randn(math.prod(grid), 12, generator=generator).to(dtype)
    assert compute_phash(resized, grid) == _reference_phash(resized, grid)


def test_phash_returns_all_64_bits_and_only_materializes_eight_bytes():
    resized = torch.ones(16, 12)
    original_tolist = torch.Tensor.tolist
    materialized_sizes = []

    def record_tolist(tensor):
        materialized_sizes.append(tensor.numel())
        return original_tolist(tensor)

    with patch.object(torch.Tensor, "tolist", record_tolist):
        result = compute_phash(resized, (1, 4, 4))
    assert result == 1 << 63
    assert materialized_sizes == [8]
    assert compute_phash(torch.zeros_like(resized), (1, 4, 4)) == 0


def test_phash_constants_follow_device_and_are_reused():
    cpu = torch.device("cpu")
    meta = torch.device("meta")
    assert _dct_matrix(cpu).device == cpu
    assert _dct_matrix(meta).device == meta
    assert _dct_matrix(cpu) is _dct_matrix(cpu)
    assert _gaussian_kernel(7, cpu).device == cpu
    assert _gaussian_kernel(7, meta).device == meta
    assert _gaussian_kernel(7, cpu) is _gaussian_kernel(7, cpu)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_ssim_normalizes_calculation_dtype(dtype):
    plane = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8).to(dtype)
    assert ssim_score(plane, plane) == pytest.approx(1.0)


def _has_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available() is True


@pytest.mark.skipif(not _has_npu(), reason="requires real Ascend NPU")
def test_npu_phash_and_ssim_on_non_default_stream():
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        resized = torch.ones((16, 12), device="npu")
        assert compute_phash(resized, (1, 4, 4)) == 1 << 63
        assert compute_phash(torch.zeros_like(resized), (1, 4, 4)) == 0
        plane = torch.ones((1, 1, 8, 8), device=resized.device)
        assert ssim_score(plane, plane) == pytest.approx(1.0)



def test_phash_default_bands_are_four_sixteen_bit_segments():
    assert PHASH_NUM_BANDS == 4
    assert bands(0x0123456789ABCDEF) == [(0, 0xCDEF), (1, 0x89AB), (2, 0x4567), (3, 0x0123)]


@pytest.mark.parametrize("num_bands", [1, 2, 4, 8, 16, 32, 64])
def test_phash_band_counts_preserve_all_bits(num_bands):
    value = 0x8123456789ABCDEF
    pieces = bands(value, num_bands=num_bands)
    assert len(pieces) == num_bands
    assert sum(part << (index * (64 // num_bands)) for index, part in pieces) == value


@pytest.mark.parametrize("num_bands", [0, -1, 3, 5, 65, True, 2.5])
def test_phash_rejects_invalid_band_counts(num_bands):
    with pytest.raises(ValueError, match="num_bands"):
        bands(0, num_bands=num_bands)
