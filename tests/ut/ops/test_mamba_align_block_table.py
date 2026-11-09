# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Equivalence tests for the Ascend fused mamba "align" block-table gather.

Port of vLLM's ``tests/v1/attention/test_mamba_align_block_table.py`` for
``vllm_ascend.ops.triton.mamba.align_block_table``: the "align" branch used to
be five PyTorch statements that expanded to seven kernel launches over a
``(num_reqs,)`` tensor; it is now one Triton kernel. These tests pin the fused
kernel to the exact semantics of the expression it replaced, including cases a
single-request decode never exercises but other callers can hit.

``block_table`` is deliberately tested non-contiguous: the original
``torch.gather(block_table, 1, ...)`` honours any layout, and this is a shared
utility, so a column stride of 1 must not be assumed.
"""

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend.ops.triton.mamba.align_block_table import (
    mamba_get_block_table_tensor,
)

WIDTH = 64


def _real_npu_available() -> bool:
    # The UT conftest installs a MagicMock torch_npu/torch.npu when no NPU is
    # present, which makes torch.npu.is_available() truthy on CPU runners, so
    # a skipif on it never skips there. Probe the device itself: allocating a
    # one-element NPU tensor only succeeds on real hardware.
    try:
        torch.empty(1, device="npu")
    except Exception:  # noqa: BLE001 - any failure means there is no real NPU
        return False
    return True


requires_npu = pytest.mark.skipif(not _real_npu_available(), reason="needs a real NPU")


def _spec(block_size: int, num_speculative_blocks: int) -> MambaSpec:
    # Only block_size and num_speculative_blocks are read by the align branch;
    # shapes/dtypes just have to be well formed for the frozen dataclass.
    return MambaSpec(
        block_size=block_size,
        shapes=((1,),),
        dtypes=(torch.float32,),
        num_speculative_blocks=num_speculative_blocks,
    )


def _reference(block_table, seq_lens, block_size, num_spec_blocks):
    """The align branch exactly as it was before the fusion."""
    start_indices = (seq_lens - 1) // block_size
    start_indices = start_indices.clamp(min=0)
    offsets = torch.arange(1 + num_spec_blocks, device=block_table.device, dtype=torch.int32)
    indices_to_gather = (start_indices.unsqueeze(1) + offsets).to(torch.int64)
    return torch.gather(block_table, 1, indices_to_gather)


def _make(nreq, block_size, nspec, device, noncontig=False):
    torch.manual_seed(0)
    width = WIDTH * 2 if noncontig else WIDTH
    bt = torch.randint(0, 5000, (nreq, width), dtype=torch.int32, device=device)
    if noncontig:
        # Every other column: stride(1) == 2, which a kernel assuming a unit
        # column stride would read straight through, silently returning the
        # wrong blocks.
        bt = bt[:, ::2]
        assert not bt.is_contiguous()
    # Keep start + nspec inside the row, which the gather already requires.
    hi = max(1, (WIDTH - 1 - nspec) * block_size)
    sl = torch.randint(0, hi, (nreq,), dtype=torch.int32, device=device)
    return bt, sl


@requires_npu
@pytest.mark.parametrize("nreq", [1, 2, 8, 64])
@pytest.mark.parametrize("block_size", [16, 64, 256])
@pytest.mark.parametrize("nspec", [0, 1, 2, 3, 4, 7])
def test_matches_reference(nreq, block_size, nspec):
    bt, sl = _make(nreq, block_size, nspec, "npu")
    got = mamba_get_block_table_tensor(bt, sl, _spec(block_size, nspec), "align")
    want = _reference(bt, sl, block_size, nspec)
    assert got.shape == want.shape
    assert got.dtype == want.dtype
    torch.testing.assert_close(got, want, atol=0, rtol=0)


@requires_npu
@pytest.mark.parametrize("nspec", [0, 3, 7])
def test_non_contiguous_block_table(nspec):
    """A column stride != 1 must be honoured, not assumed away."""
    bt, sl = _make(8, 64, nspec, "npu", noncontig=True)
    got = mamba_get_block_table_tensor(bt, sl, _spec(64, nspec), "align")
    want = _reference(bt, sl, 64, nspec)
    torch.testing.assert_close(got, want, atol=0, rtol=0)


@requires_npu
def test_row_sliced_view_matches_reference():
    """Callers pass ``block_table[:num_reqs]`` slices; the kernel must honour
    the shifted base pointer (this is the case the alignment
    de-specialization exists for)."""
    bt_full, _ = _make(64, 64, 3, "npu")
    bt = bt_full[3:11]
    torch.manual_seed(1)
    sl = torch.randint(0, 60 * 64, (8,), dtype=torch.int32, device="npu")
    got = mamba_get_block_table_tensor(bt, sl, _spec(64, 3), "align")
    want = _reference(bt, sl, 64, 3)
    torch.testing.assert_close(got, want, atol=0, rtol=0)


@requires_npu
def test_zero_seq_len_clamps_to_first_row():
    """Graph padding rows carry seq_len 0; the old code clamped to row 0."""
    bt = torch.randint(0, 5000, (4, WIDTH), dtype=torch.int32, device="npu")
    sl = torch.zeros(4, dtype=torch.int32, device="npu")
    got = mamba_get_block_table_tensor(bt, sl, _spec(64, 3), "align")
    torch.testing.assert_close(got, _reference(bt, sl, 64, 3), atol=0, rtol=0)
    # start clamps to 0, so this is just the first 1 + nspec columns.
    torch.testing.assert_close(got, bt[:, :4], atol=0, rtol=0)


@requires_npu
def test_zero_requests_returns_empty():
    """An empty batch must not launch a (0,) grid."""
    bt = torch.zeros((0, WIDTH), dtype=torch.int32, device="npu")
    sl = torch.zeros(0, dtype=torch.int32, device="npu")
    got = mamba_get_block_table_tensor(bt, sl, _spec(64, 3), "align")
    assert got.shape == (0, 4)
    assert got.dtype == bt.dtype


@requires_npu
def test_none_mode_returns_the_input_unchanged():
    """``MambaCacheMode`` is Literal["align", "none"]; only "none" passes
    through. Anything that is not "none" takes the align branch and returns a
    NEW tensor, so asserting identity for it would fail -- correctly."""
    bt = torch.randint(0, 5000, (4, WIDTH), dtype=torch.int32, device="npu")
    sl = torch.randint(0, 1000, (4,), dtype=torch.int32, device="npu")
    assert mamba_get_block_table_tensor(bt, sl, _spec(64, 3), "none") is bt


# ---------------------------------------------------------------------------
# CPU tensors. The utility is SHARED and its callers legitimately pass CPU
# tensors -- the GDN and linear-attention metadata builders are unit-tested
# with DEVICE = torch.device("cpu"). A Triton kernel cannot run on those, so
# the torch expression must stay for the CPU path. These run without an NPU.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nspec", [0, 1, 3])
@pytest.mark.parametrize("block_size", [1, 16, 64])
def test_cpu_tensors_match_the_reference(nspec, block_size):
    """The align branch must work on CPU, not just on the accelerator."""
    width = 2 + nspec + 8
    bt = torch.arange(5 * width, dtype=torch.int32).reshape(5, width)
    sl = torch.tensor([0, 1, block_size, block_size * 3 + 1, 7], dtype=torch.int32)
    got = mamba_get_block_table_tensor(bt, sl, _spec(block_size, nspec), "align")
    want = _reference(bt, sl, block_size, nspec)
    assert torch.equal(got, want), f"cpu mismatch: {got.tolist()} != {want.tolist()}"
    assert got.device == bt.device
    assert got.dtype == bt.dtype


def test_cpu_none_mode_passes_through():
    bt = torch.arange(12, dtype=torch.int32).reshape(3, 4)
    sl = torch.tensor([1, 2, 3], dtype=torch.int32)
    assert mamba_get_block_table_tensor(bt, sl, _spec(16, 3), "none") is bt
