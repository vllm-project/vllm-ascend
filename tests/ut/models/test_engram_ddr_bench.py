# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""CPU tests of the DDR benchmark policies at the Host UVA boundary."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm.models.deepseek_v41")

from vllm_ascend import envs
from vllm_ascend.models.deepseek_v41.engram import embedding
from vllm_ascend.models.deepseek_v41.engram.hash_state import DEAD_ID


@pytest.mark.parametrize(
    ("mode", "expected_reads"),
    [
        ("ddr-dedup", [6, 7, 10]),
        ("ddr-no-dedup", [10, 6, 7, 10, 6, 6]),
        ("normal", [10, 6, 7, 10, 6, 6]),
    ],
)
@pytest.mark.parametrize("audit", [False, True])
def test_ddr_lookup_reads_policy_rows_and_restores_shard_order(monkeypatch, mode, expected_reads, audit):
    monkeypatch.setitem(envs.env_variables, "VLLM_ASCEND_ENGRAM_BENCH_READ_MODE", lambda: mode)
    table = object.__new__(embedding.AscendParallelEngramEmbedding)
    torch.nn.Module.__init__(table)
    table.dim = 32
    table.head_start, table.part_n_hash_cols = 1, 2
    table.vocab_start_idx, table.vocab_end_idx = 6, 11
    # Simulate BF16 results as returned at the UVA boundary.
    table._codes_uva = SimpleNamespace(tensor=torch.tensor([16, 17, 18, 19, 20]).bfloat16().repeat(32, 1).T)
    table._scales_uva = None
    table._ddr_bench_row_counts = torch.zeros(2, dtype=torch.int64) if audit else None
    reads = []

    def host_gather(
        codes,
        scales,
        ids,
        *,
        head_start=0,
        local_heads=1,
        pad_heads=None,
        vocab_start=0,
        vocab_end=None,
        output=None,
        skip_invalid=False,
    ):
        columns = ids[:, head_start : head_start + local_heads]
        valid = (columns >= vocab_start) & (columns < vocab_end)
        reads.extend(columns[valid].tolist())
        assert ids.shape[0] * local_heads == 10
        assert skip_invalid == (mode != "normal")
        safe_rows = (columns - vocab_start).masked_fill(~valid, 0)
        result = codes.tensor[safe_rows].masked_fill(~valid.unsqueeze(-1), 0).reshape(-1, 32)
        if output is None:
            return result
        output.copy_(result)
        return output

    monkeypatch.setattr(embedding, "gather_dequantize_host_uva", host_gather)
    # A strided layer slice with unowned columns, a repeated valid ID, padding
    # and invalid IDs on both sides of the shard's vocabulary.
    ids = torch.tensor([[8, 10, 6], [8, 7, 11], [8, DEAD_ID, 10], [8, 5, 6], [8, 6, DEAD_ID]], dtype=torch.int32)
    ids = ids.unsqueeze(1).repeat(1, 2, 1)[:, 1]
    out = torch.full((5, 2, 32), float("nan"), dtype=torch.bfloat16)
    table.lookup(ids, out)
    expected = torch.tensor([[20, 16], [17, 0], [0, 20], [0, 16], [16, 0]], dtype=torch.bfloat16)
    torch.testing.assert_close(out, expected.unsqueeze(-1).expand_as(out), rtol=0, atol=0)
    assert reads == expected_reads
    if audit:
        assert table._ddr_bench_row_counts.tolist() == ([0, 0] if mode == "normal" else [6, len(expected_reads)])
        table.lookup(ids.flip(0), out)
        assert table._ddr_bench_row_counts.tolist() == ([0, 0] if mode == "normal" else [12, 2 * len(expected_reads)])
