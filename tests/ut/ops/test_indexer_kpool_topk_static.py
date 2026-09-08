# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equivalence tests for the static-shape kpool top-k scorer.

``indexer_kpool_topk_static`` is the graph-capture variant of
``indexer_kpool_topk_pytorch`` (sort-based top-k, because aclnnTopk caps
k at 256). Both scorers run on identical inputs; the selected pool-id
sets must match per row, allowing bf16 near-ties at the k-boundary.
"""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")  # noqa: F401  (registers the NPU backend)

from vllm_ascend.ops.indexer_kpool_topk import (  # noqa: E402
    indexer_kpool_topk_pytorch,
    indexer_kpool_topk_static,
)

H, D, BLOCK, POOL = 16, 128, 128, 4


def _run_both_scorers(
    T: int, seq_pool_lens: list[int], positions_list: list[int], sparse: int = 512
):
    """Run both scorers on the same inputs; return (reference, got)."""
    nreq = len(seq_pool_lens)
    max_pools = max(seq_pool_lens)
    nblocks = (max_pools + BLOCK - 1) // BLOCK
    gen = torch.Generator(device="cpu").manual_seed(19)

    # Paged cache [blocks, 128, 1, D]; give each request its own blocks.
    cache = (
        torch.randn(nblocks * nreq, BLOCK, 1, D, generator=gen).to("npu", torch.bfloat16) * 0.3
    )
    block_table = torch.zeros(nreq, nblocks, dtype=torch.int32, device="npu")
    for r in range(nreq):
        block_table[r] = torch.arange(
            r * nblocks, (r + 1) * nblocks, dtype=torch.int32, device="npu"
        )
    q = torch.randn(T, H, D, generator=gen).to("npu", torch.bfloat16)
    w = (torch.rand(T, H, generator=gen) + 0.1).abs().to("npu")
    positions = torch.tensor(positions_list, dtype=torch.int64, device="npu")

    # Split the T rows across requests: row t belongs to request
    # min(t * nreq // T, nreq - 1).
    owner = [min(t * nreq // max(1, T - 1), nreq - 1) if T > 1 else 0 for t in range(T)]
    cu_list = []
    running = 0
    for r in range(nreq):
        running += owner.count(r)
        cu_list.append(running)
    cu = torch.tensor(cu_list, dtype=torch.int32, device="npu")
    seq_lens = torch.tensor(seq_pool_lens, dtype=torch.int32, device="npu")

    reference = indexer_kpool_topk_pytorch(
        q,
        cache,
        w,
        cu,
        seq_lens,
        block_table,
        positions,
        sparse_count=sparse,
        pool_size=POOL,
        max_key_seq_len=max_pools,
    )
    got = indexer_kpool_topk_static(
        q,
        cache,
        w,
        cu,
        seq_lens,
        block_table,
        positions,
        sparse_count=sparse,
        pool_size=POOL,
    )
    return reference, got


def _assert_rows_match(reference: torch.Tensor, got: torch.Tensor, tolerance: int = 2):
    """Selected pool-id sets must match, allowing bf16 near-ties."""
    for t in range(reference.shape[0]):
        ref_set = {x for x in reference[t].tolist() if x >= 0}
        got_set = {x for x in got[t].tolist() if x >= 0}
        assert ref_set == got_set or len(ref_set & got_set) >= len(ref_set) - tolerance, (
            f"row {t}: reference pools {sorted(ref_set)}, static pools {sorted(got_set)}"
        )


@pytest.mark.parametrize(
    "T,seq_pool_lens,positions_list",
    [
        pytest.param(3, [750, 32, 500], [2999, 127, 1999], id="decode-3req-mixed"),
        pytest.param(1, [192], [767], id="single-req-short"),
        pytest.param(2, [1, 0], [3, 5], id="tiny-and-new-seqs"),
        pytest.param(8, [300] * 8, [1199] * 8, id="8req-capture-width"),
    ],
)
def test_static_topk_matches_pytorch(
    T: int, seq_pool_lens: list[int], positions_list: list[int]
):
    reference, got = _run_both_scorers(T, seq_pool_lens, positions_list)
    assert got.shape == reference.shape
    _assert_rows_match(reference, got)
