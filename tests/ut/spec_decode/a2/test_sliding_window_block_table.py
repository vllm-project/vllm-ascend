# SPDX-License-Identifier: Apache-2.0
# Unit tests for the sliding-window draft block-table helper and its proposer
# wiring. Focus: ACL graph (FULL mode) pointer stability -- the Ascend attention
# backends capture block_table by address and rebind it by reference on replay
# without copying, so block_table_tensor must keep an identical data_ptr across
# dummy_run (capture) and _propose (replay).
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
from vllm_ascend.spec_decode.utils import compute_sliding_window_block_table


def _reference_window_block_table(full_block_table, full_seq_lens, K, W, B, max_blocks):
    """Slow per-row oracle reproducing the ORIGINAL _apply_sliding_window loop.

    Used only inside tests (``.item()`` is fine here -- it is NOT a hot path).
    Assumes the realistic invariant ``start_block_index < full_cols`` (the full
    block table is always large enough for the actual sequences).
    """
    device = full_block_table.device
    num_reqs = full_seq_lens.shape[0]
    final_seq_lens = full_seq_lens + K
    start_tokens = (final_seq_lens - W).clamp(min=0)
    start_blocks = (start_tokens // B) * B
    start_block_indices = (start_blocks // B).to(torch.int64)
    tokens_to_cover = final_seq_lens - start_blocks
    needed_blocks_per_req = ((tokens_to_cover + B - 1) // B).to(torch.int64)
    out = torch.zeros((num_reqs, max_blocks), dtype=full_block_table.dtype, device=device)
    full_cols = full_block_table.shape[1]
    for i in range(num_reqs):
        start_idx = int(start_block_indices[i].item())
        needed = min(int(needed_blocks_per_req[i].item()), max_blocks)
        end_idx = start_idx + needed
        if start_idx >= full_cols:
            # All needed blocks lie beyond the full table -> nothing to copy
            # (leave zeros). The vectorized helper masks these to zero too; the
            # original loop would crash here, but this degenerate case never
            # occurs in production (the block table is always sized for the
            # actual sequences, so start_block_index < full_cols).
            continue
        if end_idx <= full_cols:
            out[i, :needed] = full_block_table[i, start_idx:end_idx]
        else:
            valid = full_cols - start_idx
            out[i, :valid] = full_block_table[i, start_idx:]
    return out


# fmt: off
class TestComputeSlidingWindowBlockTable:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.device = torch.device("npu")
        yield

    @pytest.mark.parametrize("K,W,B", [
        (2, 128, 16),   # small window, small block
        (3, 512, 128),  # the documented boundary-misalignment scale
        (1, 64, 32),    # minimal K
        (2, 256, 16),   # window spanning several blocks
    ])
    def test_correctness_in_range(self, K, W, B):
        """Gathered window table matches the per-row reference when every request
        fits inside the full block table (the realistic case)."""
        device = self.device
        window_blocks = (W + B - 1) // B
        max_window_blocks = window_blocks + 1
        num_reqs = 6
        full_cols = 64  # large enough that no request overruns the table
        full_block_table = torch.randint(
            1, 10000, (num_reqs, full_cols), dtype=torch.int32, device=device
        )
        seq_lens = torch.tensor(
            [10, W - 1, W, W + 3, 2 * W, 3 * W + 7],
            dtype=torch.int32, device=device,
        )

        # ``out`` larger than num_reqs, poisoned, so we can also assert no writes
        # spill past the active rows into the stable buffer.
        out = torch.full(
            (num_reqs + 3, max_window_blocks), 7, dtype=torch.int32, device=device
        )

        compute_sliding_window_block_table(
            full_block_table,
            full_seq_lens=seq_lens,
            num_speculative_tokens=K,
            window_size=W,
            block_size=B,
            max_window_blocks=max_window_blocks,
            out=out,
        )

        ref = _reference_window_block_table(full_block_table, seq_lens, K, W, B, max_window_blocks)
        assert torch.equal(out[:num_reqs], ref), "window block table differs from reference"
        assert torch.equal(
            out[num_reqs:], torch.full_like(out[num_reqs:], 7)
        ), "helper wrote past num_reqs into the stable buffer"

    def test_correctness_tail_truncation(self):
        """When a request's window extends past the full block table tail
        (``start_idx < full_cols < start_idx + needed``), the out-of-range
        columns are zeroed -- matching the per-row oracle."""
        device = self.device
        K, W, B = 2, 128, 16
        window_blocks = (W + B - 1) // B
        max_window_blocks = window_blocks + 1  # 9
        num_reqs = 3
        full_cols = 10  # deliberately small so seq=206 overruns the tail
        full_block_table = torch.randint(
            1, 10000, (num_reqs, full_cols), dtype=torch.int32, device=device
        )
        # seq=206: start_block_index=5, needed=8 -> end=13 > full_cols=10 (tail truncation)
        # seq=50 : fully in-range
        seq_lens = torch.tensor([206, 50, 206], dtype=torch.int32, device=device)

        out = torch.zeros((num_reqs, max_window_blocks), dtype=torch.int32, device=device)
        compute_sliding_window_block_table(
            full_block_table,
            full_seq_lens=seq_lens,
            num_speculative_tokens=K,
            window_size=W,
            block_size=B,
            max_window_blocks=max_window_blocks,
            out=out,
        )

        ref = _reference_window_block_table(full_block_table, seq_lens, K, W, B, max_window_blocks)
        assert torch.equal(out[:num_reqs], ref)

    def test_buffer_reuse_pointer_stable(self):
        """The helper writes into the caller-provided ``out`` in place: its
        data_ptr never changes, and the active slice shares the buffer's
        storage -- the property that keeps block_table stable across ACL graph
        capture and replay."""
        device = self.device
        K, W, B = 2, 128, 16
        max_window_blocks = (W + B - 1) // B + 1
        num_reqs = 4
        full_cols = 32
        out = torch.zeros((num_reqs, max_window_blocks), dtype=torch.int32, device=device)
        ptr_before = out.data_ptr()

        fbt1 = torch.randint(1, 1000, (num_reqs, full_cols), dtype=torch.int32, device=device)
        sl1 = torch.tensor([10, 100, 200, 300], dtype=torch.int32, device=device)
        compute_sliding_window_block_table(
            fbt1, full_seq_lens=sl1, num_speculative_tokens=K, window_size=W,
            block_size=B, max_window_blocks=max_window_blocks, out=out,
        )
        assert out.data_ptr() == ptr_before
        assert out[:num_reqs].data_ptr() == out.data_ptr()  # offset-0 view
        snapshot = out[:num_reqs].clone()

        # second call with different inputs -- still the same buffer, data updated
        fbt2 = torch.randint(1, 1000, (num_reqs, full_cols), dtype=torch.int32, device=device)
        sl2 = torch.tensor([500, 600, 700, 800], dtype=torch.int32, device=device)
        compute_sliding_window_block_table(
            fbt2, full_seq_lens=sl2, num_speculative_tokens=K, window_size=W,
            block_size=B, max_window_blocks=max_window_blocks, out=out,
        )
        assert out.data_ptr() == ptr_before
        ref2 = _reference_window_block_table(fbt2, sl2, K, W, B, max_window_blocks)
        assert torch.equal(out[:num_reqs], ref2)
        assert not torch.equal(out[:num_reqs], snapshot), "second call did not overwrite the buffer in place"

    def test_dtype_int32_preserved(self):
        device = self.device
        K, W, B = 2, 64, 16
        max_window_blocks = (W + B - 1) // B + 1
        full_block_table = torch.randint(1, 1000, (2, 8), dtype=torch.int32, device=device)
        seq_lens = torch.tensor([10, 70], dtype=torch.int32, device=device)
        out = torch.zeros((2, max_window_blocks), dtype=torch.int32, device=device)
        compute_sliding_window_block_table(
            full_block_table, full_seq_lens=seq_lens, num_speculative_tokens=K,
            window_size=W, block_size=B, max_window_blocks=max_window_blocks, out=out,
        )
        assert out.dtype == torch.int32


# fmt: off
class TestApplySlidingWindowGraphStable:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.device = torch.device("npu")
        yield

    def _new_proposer(self, K, W, B, max_num_reqs=8):
        # Bypass __init__; set only what _apply_sliding_window touches.
        proposer = object.__new__(AscendSpecDecodeBaseProposer)
        proposer.draft_window_size = W
        proposer.num_speculative_tokens = K
        proposer.block_size = B
        proposer.window_blocks = (W + B - 1) // B
        proposer.max_window_blocks = proposer.window_blocks + 1
        proposer._sliding_window_full_block_table = None
        proposer._sliding_window_start_block_indices = None
        proposer._sliding_window_block_table_clone = torch.zeros(
            (max_num_reqs, proposer.max_window_blocks), dtype=torch.int32, device=self.device
        )
        proposer.seq_lens_group = [
            torch.zeros(max_num_reqs, dtype=torch.int32, device=self.device) for _ in range(K)
        ]
        return proposer

    def test_block_table_pointer_stable_across_calls(self):
        """Core graph-mode guarantee: after _apply_sliding_window, the metadata's
        block_table_tensor points at the pre-allocated clone (offset-0 view), and
        its data_ptr is identical across two calls with different inputs."""
        K, W, B = 2, 128, 16
        max_num_reqs = 8
        proposer = self._new_proposer(K, W, B, max_num_reqs)
        clone_ptr = proposer._sliding_window_block_table_clone.data_ptr()

        full_cols = 32
        fbt1 = torch.randint(1, 1000, (max_num_reqs, full_cols), dtype=torch.int32, device=self.device)
        cad1 = SimpleNamespace(block_table_tensor=fbt1)
        sl1 = torch.tensor([10, 100, 200, 300, 400, 500, 600, 700], dtype=torch.int32, device=self.device)
        proposer._apply_sliding_window(cad1, sl1)
        assert cad1.block_table_tensor.data_ptr() == clone_ptr

        fbt2 = torch.randint(1, 1000, (max_num_reqs, full_cols), dtype=torch.int32, device=self.device)
        cad2 = SimpleNamespace(block_table_tensor=fbt2)
        sl2 = torch.tensor([20, 120, 220, 320, 420, 520, 620, 720], dtype=torch.int32, device=self.device)
        proposer._apply_sliding_window(cad2, sl2)
        assert cad2.block_table_tensor.data_ptr() == clone_ptr
        assert cad2.block_table_tensor.data_ptr() == cad1.block_table_tensor.data_ptr()

        # content written into the stable clone matches the reference
        ref = _reference_window_block_table(fbt2, sl2, K, W, B, proposer.max_window_blocks)
        assert torch.equal(proposer._sliding_window_block_table_clone[:max_num_reqs], ref)

    def test_seq_lens_clamped_and_group_prefilled(self):
        K, W, B = 3, 64, 16
        max_num_reqs = 5
        proposer = self._new_proposer(K, W, B, max_num_reqs)
        full_cols = 16
        fbt = torch.randint(1, 1000, (max_num_reqs, full_cols), dtype=torch.int32, device=self.device)
        cad = SimpleNamespace(block_table_tensor=fbt)
        sl = torch.tensor([10, 60, 64, 100, 200], dtype=torch.int32, device=self.device)
        proposer._apply_sliding_window(cad, sl)

        # seq_lens clamped to the window
        assert torch.equal(cad.seq_lens, torch.min(sl, torch.full_like(sl, W)))
        # seq_lens_group prefilled per step: min(seq + step, W)
        for step in range(K):
            expect = torch.min(sl + step, torch.full_like(sl, W))
            assert torch.equal(proposer.seq_lens_group[step][:max_num_reqs], expect)
            assert torch.equal(
                proposer.seq_lens_group[step][max_num_reqs:],
                torch.zeros_like(proposer.seq_lens_group[step][max_num_reqs:]),
            )

    def test_noop_when_window_disabled(self):
        proposer = self._new_proposer(2, 128, 16)
        proposer.draft_window_size = None  # sliding window disabled
        fbt = torch.randint(1, 1000, (4, 8), dtype=torch.int32, device=self.device)
        cad = SimpleNamespace(block_table_tensor=fbt)
        sl = torch.tensor([10, 20, 30, 40], dtype=torch.int32, device=self.device)
        proposer._apply_sliding_window(cad, sl)
        # metadata untouched
        assert cad.block_table_tensor is fbt
        assert not hasattr(cad, "seq_lens")
# fmt: on
