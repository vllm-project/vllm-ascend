# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Nightly on-device equivalence runs for the fused mamba "align" block-table gather.

Per the repo test organization, ``tests/ut`` runs on CPU runners, so the
on-device kernel tests live here: the parametrized builders are reused from
``tests/ut/ops/test_mamba_align_block_table.py`` and the Triton kernel is
exercised on a real NPU with bit-exact comparisons (``atol=0, rtol=0``)
against the seven-op torch expression it replaced.
"""

import pytest
import torch
import torch_npu  # noqa: F401

from tests.ut.ops.test_mamba_align_block_table import _make, _reference, _spec
from vllm_ascend.ops.triton.mamba.align_block_table import (
    mamba_get_block_table_tensor,
)


@pytest.mark.parametrize("nreq", [1, 8, 64, 512])
@pytest.mark.parametrize("block_size", [16, 64, 256])
@pytest.mark.parametrize("nspec", [0, 3, 5])
def test_nightly_matches_reference(nreq, block_size, nspec):
    bt, sl = _make(nreq, block_size, nspec, "npu")
    got = mamba_get_block_table_tensor(bt, sl, _spec(block_size, nspec), "align")
    want = _reference(bt, sl, block_size, nspec)
    assert got.shape == want.shape
    assert got.dtype == want.dtype
    torch.testing.assert_close(got, want, atol=0, rtol=0)


@pytest.mark.parametrize("nspec", [3, 7])
def test_nightly_non_contiguous_block_table(nspec):
    """A column stride != 1 must be honoured, not assumed away."""
    bt, sl = _make(8, 64, nspec, "npu", noncontig=True)
    got = mamba_get_block_table_tensor(bt, sl, _spec(64, nspec), "align")
    torch.testing.assert_close(got, _reference(bt, sl, 64, nspec), atol=0, rtol=0)


def test_nightly_row_sliced_view():
    """Callers pass ``block_table[:num_reqs]`` slices; the shifted base pointer
    and the shifted alignment must both be handled (this is the case the
    alignment de-specialization exists for)."""
    bt_full, _ = _make(64, 64, 3, "npu")
    bt = bt_full[3:11]
    torch.manual_seed(1)
    sl = torch.randint(0, 60 * 64, (8,), dtype=torch.int32, device="npu")
    got = mamba_get_block_table_tensor(bt, sl, _spec(64, 3), "align")
    torch.testing.assert_close(got, _reference(bt, sl, 64, 3), atol=0, rtol=0)


def test_nightly_zero_seq_len_clamps_to_first_row():
    """Graph padding rows carry seq_len 0; the old code clamped to row 0."""
    bt = torch.randint(0, 5000, (4, 64), dtype=torch.int32, device="npu")
    sl = torch.zeros(4, dtype=torch.int32, device="npu")
    got = mamba_get_block_table_tensor(bt, sl, _spec(64, 3), "align")
    torch.testing.assert_close(got, _reference(bt, sl, 64, 3), atol=0, rtol=0)
    torch.testing.assert_close(got, bt[:, :4], atol=0, rtol=0)


def test_nightly_zero_requests_returns_empty():
    """An empty batch must not launch a (0,) grid."""
    bt = torch.zeros((0, 64), dtype=torch.int32, device="npu")
    sl = torch.zeros(0, dtype=torch.int32, device="npu")
    got = mamba_get_block_table_tensor(bt, sl, _spec(64, 3), "align")
    assert got.shape == (0, 4)
    assert got.dtype == bt.dtype
