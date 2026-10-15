# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the FP8 payload and CANN-compatible scale planes."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm_ascend.ops.triton import glm5_next_kpool_tail_compress as compress_module
from vllm_ascend.ops.triton import glm5_next_lightning_indexer as score_module


def test_fp8_decode_matches_all_256_torch_e4m3fn_encodings():
    # Execute the actual JIT helper body with NumPy lanes. This checks the
    # bit arithmetic on CPU without compiling a kernel or acquiring an NPU.
    source = ast.parse(Path(score_module.__file__).read_text(encoding="utf-8"))
    function = next(
        node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "_decode_fp8_e4m3fn"
    )
    function.decorator_list = []

    class Lanes(np.ndarray):
        def to(self, dtype, bitcast=False):
            return (self.view(dtype) if bitcast else self.astype(dtype)).view(Lanes)

    scope: dict[str, Any] = {
        "tl": SimpleNamespace(float32=np.float32, where=lambda condition, x, y: np.where(condition, x, y).view(Lanes))
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), score_module.__file__, "exec"), scope)
    values = np.arange(256, dtype=np.int32).view(Lanes)
    actual = scope[function.name](values)
    expected = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).float().numpy()
    np.testing.assert_equal(actual, expected)
    np.testing.assert_equal(np.signbit(actual[[0, 128]]), np.signbit(expected[[0, 128]]))


@pytest.mark.parametrize("packing", [False, True])
def test_fp8_score_passes_separate_scale_strides_and_mode(monkeypatch, packing):
    launcher = MagicMock()
    monkeypatch.setattr(score_module, "_glm5_next_lightning_indexer_score_kernel", launcher)
    gather = MagicMock()
    monkeypatch.setattr(score_module, "_gather_pool_cache", gather)
    monkeypatch.setattr(score_module, "_expand_pool_indices", MagicMock())
    count = 40 if packing else 2
    # Padded kernel pages share a backing allocation with scale planes.
    raw = torch.zeros(2 * 1536, dtype=torch.uint8)
    cache = torch.as_strided(raw.view(torch.float8_e4m3fn), (2, 8, 1, 128), (1536, 128, 128, 1))
    scales = torch.as_strided(raw.view(torch.float32), (2, 8, 1, 1), (384, 1, 1, 1), storage_offset=256)
    score_module.glm5_next_lightning_indexer_triton(
        torch.ones(count, 2, 128),
        cache,
        torch.ones(count, 2),
        torch.tensor([count], dtype=torch.int32),
        torch.tensor([8]),
        torch.zeros(1, 1, dtype=torch.int32),
        torch.full((count,), 31),
        index_topk=4,
        index_kpool=4,
        max_pool_seq_len=8,
        allow_cache_packing=packing,
        indexer_scale_cache=scales,
    )
    call = launcher.__getitem__.return_value.call_args.args
    assert call[-2:] == (2, True)  # FP8, planar scales
    assert call[3] is (gather.__getitem__.return_value.call_args.args[7] if packing else scales)
    assert gather.__getitem__.return_value.call_count == int(packing)


def test_fp8_compressor_keeps_key_and_scale_planes_separate(monkeypatch):
    launcher = MagicMock()
    monkeypatch.setattr(compress_module, "_glm5_next_kpool_tail_compress_kernel", launcher)
    monkeypatch.setattr(compress_module, "_store_kpool_tail_kernel", MagicMock())
    cache = torch.zeros(2, 8, 1, 128, dtype=torch.float8_e4m3fn)
    scales = torch.ones(2, 8, 1, 1)
    compress_module.glm5_next_kpool_tail_compress_and_write_cache_triton(
        torch.zeros(1, 2, 4, 128),
        cache,
        torch.zeros(4, 128),
        torch.zeros(4, 128),
        torch.zeros(4, 128),
        torch.arange(4),
        torch.tensor([4]),
        torch.tensor([4]),
        torch.arange(4),
        torch.tensor([[0]]),
        torch.tensor([-1, -1, -1, 0]),
        4,
        indexer_scale_cache=scales,
    )
    args = launcher.__getitem__.return_value.call_args.args
    assert args[1] is cache
    assert args[3] is scales
    assert args[-3:] == (2, 448.0, True)


@pytest.mark.parametrize("bad_scales", [None, torch.ones(2, 8, 1, 2), torch.ones(2, 8, 1, 1, dtype=torch.bfloat16)])
def test_fp8_planar_cache_rejects_missing_or_incompatible_scales(bad_scales):
    cache = torch.zeros(2, 8, 1, 128, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError):
        score_module.glm5_next_lightning_indexer_triton(
            torch.ones(2, 2, 128),
            cache,
            torch.ones(2, 2),
            torch.tensor([2]),
            torch.tensor([8]),
            torch.zeros(1, 1, dtype=torch.int32),
            torch.tensor([30, 31]),
            index_topk=4,
            index_kpool=4,
            max_pool_seq_len=8,
            indexer_scale_cache=bad_scales,
        )
