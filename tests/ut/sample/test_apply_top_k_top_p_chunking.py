import math

import pytest
import torch

from vllm_ascend.worker.v2.sample import apply_top_k_top_p as topp_mod


def _kept_indices(row: torch.Tensor) -> list[int]:
    return [i for i, v in enumerate(row.tolist()) if v != -float("inf")]


@pytest.fixture
def small_tile(monkeypatch: pytest.MonkeyPatch):
    """Force tiny input tiles (72 bytes -> 3 rows of 6 float32) to cover chunk boundaries."""
    monkeypatch.setattr(topp_mod, "TOP_K_TOP_P_INPUT_TILE_BYTES", 72)


@pytest.fixture
def single_row_tile(monkeypatch: pytest.MonkeyPatch):
    """Force 1-row tiles (24 bytes < one 6-float32 row) so k/p row slices split every row."""
    monkeypatch.setattr(topp_mod, "TOP_K_TOP_P_INPUT_TILE_BYTES", 24)


class TestApplyTopKTopPChunked:
    def test_mixed_rows_chunk_boundaries_and_ties(self, small_tile):
        logits = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5],
                [4, 4, 0, -1, -2, -3],
                [0, 1, 2, 3, 4, 5],
                [math.log(0.05), math.log(0.10), math.log(0.15), math.log(0.20), math.log(0.25), math.log(0.25)],
                [0, 0, 0, 0, 0, 0],
                [-100, -90, -80, -70, -60, 0],
                [3, -1, -1, -1, -1, -1],
            ],
            dtype=torch.float32,
        )
        expected_keep = [
            [5],
            [0, 1],
            [0, 1, 2, 3, 4, 5],
            [3, 4, 5],
            [0, 1, 2, 3, 4, 5],
            [5],
            [0],
        ]
        k = torch.tensor([1, 1, 6, 6, 2, 2, 6], dtype=torch.int64)
        p = torch.tensor([1.0, 1.0, 1.0, 0.55, 1.0, 0.9, 0.9], dtype=torch.float32)

        src = logits.clone()
        out = topp_mod._apply_top_k_top_p_pytorch(logits.clone(), k, p)

        for row, expected in zip(out, expected_keep):
            assert _kept_indices(row) == expected
        # kept logits values must be unchanged (in-place masking only)
        for i, expected in enumerate(expected_keep):
            for j in expected:
                assert out[i, j].item() == src[i, j].item()

    def test_k_only(self, single_row_tile):
        logits = torch.tensor(
            [
                [1.0, 5.0, 2.0, 2.0, 0.5, 0.1],
                [0.1, 0.2, 3.0, 1.0, 0.0, -1.0],
                [2.0, 2.0, 2.0, 1.0, 0.0, -2.0],
            ],
            dtype=torch.float32,
        )
        k = torch.tensor([2, 3, 3], dtype=torch.int64)
        out = topp_mod._apply_top_k_top_p_pytorch(logits.clone(), k, None)
        # k-th thresholds: 2.0 / 0.2 / 2.0; strict < keeps every token tied at the cutoff
        assert _kept_indices(out[0]) == [1, 2, 3]
        assert _kept_indices(out[1]) == [1, 2, 3]
        assert sorted(_kept_indices(out[2])) == [0, 1, 2]

    def test_p_only(self, single_row_tile):
        logits = torch.tensor(
            [
                [math.log(0.4), math.log(0.35), math.log(0.15), math.log(0.05), math.log(0.03), math.log(0.02)],
                [math.log(0.2), math.log(0.2), math.log(0.2), math.log(0.2), math.log(0.1), math.log(0.1)],
            ],
            dtype=torch.float32,
        )
        p = torch.tensor([0.89, 0.5], dtype=torch.float32)
        out = topp_mod._apply_top_k_top_p_pytorch(logits.clone(), None, p)
        # row0: lower-tail limit 0.11, first 3 tail masses sum 0.10 -> cutoff 0.15
        assert _kept_indices(out[0]) == [0, 1, 2]
        # row1: lower-tail first 3 sum 0.4 -> cutoff 0.2, ties keep all four 0.2s
        assert _kept_indices(out[1]) == [0, 1, 2, 3]

    def test_no_filtration_early_return(self):
        logits = torch.tensor(
            [
                [0.5, -1.0, 2.0, 0.0, -3.0, 1.5],
                [math.log(0.3), math.log(0.3), math.log(0.2), math.log(0.1), math.log(0.05), math.log(0.05)],
            ],
            dtype=torch.float32,
        )
        out = topp_mod._apply_top_k_top_p_pytorch(logits, None, None)
        assert out is logits
        # identity must not be the only assertion: every value is preserved bit-for-bit
        assert torch.equal(out, logits)
        assert not torch.isinf(out).any()
