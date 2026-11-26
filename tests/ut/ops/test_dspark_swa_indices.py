# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the unified DSpark SWA Triton wrapper and helpers."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ops.triton import dspark_swa_indices as ds


def _npu_meta(t: torch.Tensor, idx: int = 0) -> Any:
    kw = dict(device=torch.device("npu", idx), dtype=t.dtype, ndim=t.ndim, shape=t.shape)
    ns = SimpleNamespace(**kw, stride=t.stride, is_contiguous=t.is_contiguous)
    ns.contiguous = lambda: ns
    return ns


def test_normalize_output_mode_and_availability():
    assert ds.normalize_dspark_swa_output_mode() == ds.OUTPUT_MODE_PHYSICAL
    assert ds.normalize_dspark_swa_output_mode(use_logical_indices=True) == ds.OUTPUT_MODE_LOGICAL
    assert ds.normalize_dspark_swa_output_mode("logical", use_logical_indices=True) == ds.OUTPUT_MODE_LOGICAL
    for kw, msg in (
        ({"output_mode": "bad"}, "Unsupported DSpark SWA output_mode"),
        ({"output_mode": "physical", "use_logical_indices": True}, "Conflicting DSpark SWA mode arguments"),
    ):
        with pytest.raises(ValueError, match=msg):
            ds.normalize_dspark_swa_output_mode(**kw)
    with patch.object(ds, "HAS_TRITON", False):
        assert ds.dspark_swa_indices_supported(128, 16) is False
    if HAS_TRITON:
        assert ds.dspark_swa_indices_supported(96, 16384, output_mode=ds.OUTPUT_MODE_PHYSICAL) is True
        assert ds.dspark_swa_indices_supported(0, 16, output_mode=ds.OUTPUT_MODE_PHYSICAL) is False
        assert ds.dspark_swa_indices_supported(0, 0, output_mode=ds.OUTPUT_MODE_LOGICAL) is True


@pytest.mark.skipif(not HAS_TRITON, reason="triton is not installed")
@pytest.mark.parametrize("bw,bs", [(8, 1), (8, 96), (1024, 12), (1024, 128), (1024, 2048)])
def test_bt_win_pow2_bounds_any_window_and_table_width(bw, bs):
    bt_win = ds._compute_bt_win_pow2(bw, bs, False)
    assert bt_win >= 2 and (bt_win & (bt_win - 1)) == 0
    for pos in (0, bs - 1, bs, 15000 * bs - 1):
        assert 0 <= min((pos + bw - 1) // bs, 19_999) - min(pos // bs, 19_999) < bt_win


def test_triton_wrapper_validation():
    qsl, seq_lens = torch.tensor([0, 4], dtype=torch.int32), torch.tensor([8], dtype=torch.int32)
    bt = torch.ones((1, 4), dtype=torch.int32)
    with patch.object(ds, "HAS_TRITON", False), pytest.raises(RuntimeError, match="Triton is required"):
        ds.build_dspark_swa_indices_triton(bt, 3, 16, 128, qsl, seq_lens)
    with patch.object(ds, "HAS_TRITON", True):
        with pytest.raises(ValueError, match="requires an accelerator device"):
            ds.build_dspark_swa_indices_triton(bt, 3, 16, 96, qsl, seq_lens)
        with pytest.raises(ValueError, match="same device"):
            ds.build_dspark_swa_indices_triton(_npu_meta(bt, 1), 3, 16, 96, _npu_meta(qsl), _npu_meta(seq_lens), 4)
        for bad_dtype in (torch.float32, torch.bool):
            for field, mode in (("query_start_loc", "physical"), ("seq_lens", "logical"), ("block_table", "physical")):
                q = _npu_meta(torch.tensor([0, 4], dtype=bad_dtype if field == "query_start_loc" else torch.int32))
                s = _npu_meta(torch.tensor([8], dtype=bad_dtype if field == "seq_lens" else torch.int32))
                b_t = torch.ones((1, 4), dtype=bad_dtype if field == "block_table" else torch.int32)
                b = None if mode == "logical" else _npu_meta(b_t)
                bs = 1 if mode == "logical" else 96
                with pytest.raises(ValueError, match="int32 or int64"):
                    ds.build_dspark_swa_indices_triton(b, 3, 16, bs, q, s, num_decode_tokens=4, output_mode=mode)
        base: dict[str, Any] = {"block_table": _npu_meta(bt), "num_speculative_tokens": 3, "window_size": 16}
        base |= {"block_size": 96, "query_start_loc": _npu_meta(qsl), "seq_lens": _npu_meta(seq_lens)}
        bad: list[tuple[str, int]] = [("num_speculative_tokens", -1), ("window_size", -1), ("block_w", 0)]
        bad += [("max_num_reqs", -1), ("index_width", 0), ("num_decode_tokens", -1), ("num_query_per_req", 0)]
        for key, val in bad:
            with pytest.raises(ValueError, match=key):
                ds.build_dspark_swa_indices_triton(**(base | {"num_decode_tokens": 4, key: val}))
