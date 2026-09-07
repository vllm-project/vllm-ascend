# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the GLM-5.3-Flash kpool indexer's scoring and selection path."""

import sys
import types

import pytest
import torch

from vllm_ascend.models.glm5next.ops.kpool_indexer import (
    LIGHTNING_INDEXER_POOLS_ALIGNMENT,
    SPARSE_INDEX_PAD,
    compact_selection,
    pa_bsnd_keys,
    request_index_per_row,
    row_seq_lens,
    rows_as_batch_entries,
    select_token_ids,
    tail_width_for,
)

HEAD_DIM = 128
NUM_HEADS = 16
POOL_SIZE = 4
POOLS_PER_BLOCK = 32
TOPK_TOKENS = 32


@pytest.fixture
def fake_operator(monkeypatch):
    """Stand in for ``npu_lightning_indexer`` with the CPU equivalent.

    The operator itself is verified against a reference by
    ``tools/probe_ascend_indexer_ops.py`` on hardware. What cannot be checked
    there is the composition around it -- the key length it is handed, and the
    expansion of what it returns -- which is what these tests cover.
    """
    calls: list[dict] = []

    def npu_lightning_indexer(query, key, weights, **kwargs):
        calls.append(kwargs)
        assert kwargs["layout_query"] == "TND"
        assert kwargs["layout_key"] == "PA_BSND"
        # A mask would remove pools the shared boundary already made legal.
        assert kwargs["sparse_mode"] == 0

        num_tokens = query.shape[0]
        budget = kwargs["sparse_count"]
        selectable = kwargs["actual_seq_lengths_key"]
        cumulative = kwargs["actual_seq_lengths_query"]
        block_table = kwargs["block_table"]
        pools_per_block = key.shape[1]

        # The real operator's fill for unused budget slots is not specified, so
        # stand in for it with a pool id no request could own rather than the
        # -1 the expansion used to rely on.
        out = torch.full((num_tokens, 1, budget), 999_999, dtype=torch.int32)
        start = 0
        for request in range(cumulative.shape[0]):
            end = int(cumulative[request])
            offered = int(selectable[request])
            gathered = torch.cat(
                [key[int(block)].reshape(pools_per_block, HEAD_DIM) for block in block_table[request]]
            )[:offered]
            scores = (
                torch.relu(torch.matmul(query[start:end].float(), gathered.t().float()))
                * weights[start:end].float().unsqueeze(-1)
            ).sum(dim=1)
            order = torch.argsort(scores, dim=-1, descending=True, stable=True)
            keep = min(budget, offered)
            out[start:end, 0, :keep] = order[:, :keep].to(torch.int32)
            start = end
        return out, None

    module = types.ModuleType("torch_npu")
    module.npu_lightning_indexer = npu_lightning_indexer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch_npu", module)
    return calls


def _cache(num_blocks: int) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(num_blocks, POOLS_PER_BLOCK, 1, HEAD_DIM, dtype=torch.bfloat16)


def _run(query_lens: list[int], seq_lens: list[int], fake_operator) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    num_tokens = sum(query_lens)
    num_requests = len(query_lens)
    blocks_per_request = 2
    return select_token_ids(
        torch.randn(num_tokens, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16),
        torch.randn(num_tokens, NUM_HEADS, dtype=torch.bfloat16),
        _cache(num_requests * blocks_per_request),
        block_table=torch.arange(num_requests * blocks_per_request, dtype=torch.int32).reshape(
            num_requests, blocks_per_request
        ),
        query_lens=torch.tensor(query_lens, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        pool_size=POOL_SIZE,
        pools_per_block=POOLS_PER_BLOCK,
        topk_tokens=TOPK_TOKENS,
        max_query_len=max(query_lens),
    )


def test_selection_is_offered_the_prefix_every_row_may_see(fake_operator):
    """The key length handed over is the earliest row's complete pool count."""
    _run(query_lens=[1, 4], seq_lens=[21, 33], fake_operator=fake_operator)

    (kwargs,) = fake_operator
    # Request 0 is a plain decode row at seq_len 21 -> 5 complete pools.
    # Request 1's earliest row is at seq_len 30 -> 7, not the 8 its last row has.
    torch.testing.assert_close(kwargs["actual_seq_lengths_key"], torch.tensor([5, 7], dtype=torch.int32))
    torch.testing.assert_close(kwargs["actual_seq_lengths_query"], torch.tensor([1, 5], dtype=torch.int32))
    assert kwargs["sparse_count"] == TOPK_TOKENS // POOL_SIZE


@pytest.mark.parametrize("query_lens", [[1, 1, 1], [4], [1, 4], [3, 2, 5]])
def test_selected_ids_are_causal_and_cover_the_recent_run(query_lens, fake_operator):
    """Nothing may name a future token, and nothing recent may go missing."""
    seq_lens = [200 + 7 * i for i in range(len(query_lens))]
    selected, lens = _run(query_lens, seq_lens, fake_operator)

    per_row = row_seq_lens(
        torch.tensor(seq_lens, dtype=torch.int32),
        torch.tensor(query_lens, dtype=torch.int32),
    )
    assert selected.shape == (
        sum(query_lens),
        TOPK_TOKENS + tail_width_for(POOL_SIZE, max(query_lens)),
    )
    assert lens.shape == (sum(query_lens),)

    # Every token from the request's shared boundary onwards must be named:
    # those pools were never offered to the operator, so if the expansion does
    # not list them the model simply cannot see its newest tokens.
    row = 0
    for request, query_len in enumerate(query_lens):
        boundary = ((seq_lens[request] - query_len + 1) // POOL_SIZE) * POOL_SIZE
        for _ in range(query_len):
            # Only the prefix the operator will read counts as named.
            named = selected[row, : int(lens[row])]
            assert torch.all(named >= 0) and torch.all(named < int(per_row[row]))
            assert torch.all(selected[row, int(lens[row]) :] == SPARSE_INDEX_PAD)
            assert set(range(boundary, int(per_row[row]))) <= {int(v) for v in named}
            row += 1


def test_row_sequence_lengths_count_back_from_the_step():
    per_row = row_seq_lens(torch.tensor([21, 33]), torch.tensor([1, 4]))
    torch.testing.assert_close(per_row, torch.tensor([21, 30, 31, 32, 33]))


def test_compaction_closes_the_gap_between_the_history_and_the_tail():
    """The operator reads a prefix of the list, so the tail cannot sit at the end.

    A row that fills 5 of 512 budget slots has its 20 oldest tokens in columns
    0-19 and its newest two in columns 2048-2049. Handed that layout with a key
    length of 22 the operator stops at column 20, never reaches the tail, and so
    cannot see the newest tokens at all.
    """
    budget_width = 2048
    ids = torch.full((1, budget_width + 3), -1, dtype=torch.int32)
    ids[0, :20] = torch.arange(20, dtype=torch.int32)
    ids[0, budget_width : budget_width + 2] = torch.tensor([20, 21], dtype=torch.int32)

    ordered, lens = compact_selection(ids)

    assert int(lens[0]) == 22
    torch.testing.assert_close(ordered[0, :22], torch.arange(22, dtype=torch.int32))
    assert torch.all(ordered[0, 22:] == SPARSE_INDEX_PAD)


def test_compaction_counts_each_row_on_its_own():
    ids = torch.tensor([[7, -1, 3, -1], [-1, -1, -1, -1], [4, 3, 2, 1]], dtype=torch.int32)

    ordered, lens = compact_selection(ids)

    torch.testing.assert_close(lens, torch.tensor([2, 0, 4], dtype=torch.int32))
    torch.testing.assert_close(ordered[0, :2], torch.tensor([3, 7], dtype=torch.int32))
    assert torch.all(ordered[1] == SPARSE_INDEX_PAD)
    torch.testing.assert_close(ordered[2], torch.tensor([1, 2, 3, 4], dtype=torch.int32))


@pytest.mark.parametrize("query_lens", [[1, 1, 1], [4], [1, 4], [3, 2, 5], [2, 0, 3]])
def test_request_index_per_row_matches_the_one_argument_spelling(query_lens):
    """The supported overload has to agree with the one it replaces."""
    lens = torch.tensor(query_lens, dtype=torch.int32)
    rows = request_index_per_row(lens, sum(query_lens))
    torch.testing.assert_close(rows, torch.repeat_interleave(lens.to(torch.int64)))
    assert rows.dtype == torch.int64


def test_kpool_ops_avoid_the_repeat_interleave_overload_without_an_npu_kernel():
    """``repeat_interleave(repeats)`` falls back to the CPU and stalls the step.

    The one-argument spelling picks ``aten::repeat_interleave.Tensor``, which
    the NPU backend does not implement. Naming the source tensor picks the
    two-argument overload, which it does.
    """
    import ast
    from pathlib import Path

    ops = Path(__file__).resolve().parents[3] / "vllm_ascend" / "models" / "glm5next" / "ops"
    offenders = []
    for path in sorted(ops.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if getattr(func, "attr", None) != "repeat_interleave":
                continue
            # A bare integer repeat count is a different, supported overload.
            if len(node.args) == 1 and not isinstance(node.args[0], ast.Constant):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "these calls select aten::repeat_interleave.Tensor, which has no NPU "
        f"kernel; pass the source tensor as well: {offenders}"
    )


def test_the_selection_chain_never_copies_a_length_onto_the_device():
    """An H2D copy in this chain cannot survive ACL-graph capture.

    Capture records the host address a copy reads, so a replay picks up
    whatever occupies it by then rather than the current step's lengths. The
    attention metadata does keep ``seq_lens`` on the host, which is why taking a
    device from it is called out separately: the positions are the device-side
    source to derive both lengths from.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[3] / "vllm_ascend"
    paths = sorted((root / "models" / "glm5next" / "ops").glob("*.py"))
    paths.append(root / "attention" / "kpool_mla_v1.py")

    def names_a_device(node) -> str:
        """The tensor a ``<tensor>.device`` expression reads, if it is one."""
        if not isinstance(node, ast.Attribute) or node.attr != "device":
            return ""
        name = getattr(node.value, "attr", None) or getattr(node.value, "id", None)
        return name if isinstance(name, str) else ""

    offenders = []
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            # `.to(dtype)` is free; `.to(other.device)` is the copy at issue.
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "to":
                if len(node.args) == 1 and names_a_device(node.args[0]):
                    offenders.append(f"{path.name}:{node.lineno} copies across devices")
            # Comparing or reporting a device is fine; placing a new tensor by
            # one the metadata keeps on the host is what strands it there.
            elif isinstance(node, ast.keyword) and node.arg == "device":
                source = names_a_device(node.value)
                if source.endswith("seq_lens"):
                    offenders.append(f"{path.name}:{node.lineno} places a tensor by the host-side {source}")
    assert not offenders, offenders


def test_pa_bsnd_keeps_the_pool_axis():
    keys = torch.zeros(2, 160, 1, HEAD_DIM, dtype=torch.bfloat16)
    viewed, stride = pa_bsnd_keys(keys, HEAD_DIM, 160)
    assert viewed.shape == (2, 160, 1, HEAD_DIM)
    assert stride == 1

    squeezed = keys.squeeze(2)
    assert squeezed.shape == (2, 160, HEAD_DIM)
    assert pa_bsnd_keys(squeezed, HEAD_DIM, 160)[0].shape == (2, 160, 1, HEAD_DIM)


def test_a_page_padded_to_the_mla_width_splits_into_addressable_blocks():
    """Unifying page sizes pads this cache; the operator caps a page at 1024.

    A block's pools sit at the front of the padded page, so splitting the page
    into block-sized ones and striding the block table names the same bytes
    through a page the operator accepts.
    """
    padded = torch.zeros(2, 2560, 1, HEAD_DIM, dtype=torch.bfloat16)
    viewed, stride = pa_bsnd_keys(padded, HEAD_DIM, 160)
    assert viewed.shape == (32, 160, 1, HEAD_DIM)
    assert stride == 16

    # Block 1 still starts where the second padded page does.
    assert viewed[1 * stride].data_ptr() == padded[1].data_ptr()


def test_block_table_stride_keeps_the_unallocated_sentinel(monkeypatch):
    """Padding splits a page; the operator still wants -1, not -stride."""
    captured: dict[str, torch.Tensor] = {}

    def fake_score(query, weights, index_cache, **kwargs):
        captured["block_table"] = kwargs["block_table"]
        return torch.zeros(query.shape[0], 1, TOPK_TOKENS // POOL_SIZE, dtype=torch.int32)

    monkeypatch.setattr(
        "vllm_ascend.models.glm5next.ops.kpool_indexer.score_and_select_pools",
        fake_score,
    )

    select_token_ids(
        torch.zeros(2, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16),
        torch.zeros(2, NUM_HEADS, dtype=torch.bfloat16),
        torch.zeros(2, 2560, 1, HEAD_DIM, dtype=torch.bfloat16),
        block_table=torch.tensor([[0, -1], [1, -1]], dtype=torch.int32),
        query_lens=torch.tensor([1, 1], dtype=torch.int32),
        seq_lens=torch.tensor([4, 4], dtype=torch.int32),
        pool_size=POOL_SIZE,
        pools_per_block=160,
        topk_tokens=TOPK_TOKENS,
        max_query_len=1,
    )
    torch.testing.assert_close(
        captured["block_table"],
        torch.tensor([[0, -1], [16, -1]], dtype=torch.int32),
    )


def test_pa_bsnd_rejects_a_page_that_is_not_whole_blocks():
    ragged = torch.zeros(2, 165, 1, HEAD_DIM, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="not a whole number of the 160"):
        pa_bsnd_keys(ragged, HEAD_DIM, 160)


def test_pa_bsnd_does_not_invent_pools_from_an_fp8_wide_page():
    """160 cells of width 132, viewed as 128, is 165 -- not a multiple of 16."""
    packed = torch.zeros(2, 160, 1, 132, dtype=torch.bfloat16)
    assert packed.numel() // (2 * HEAD_DIM) == 165
    assert 165 % LIGHTNING_INDEXER_POOLS_ALIGNMENT != 0

    with pytest.raises(RuntimeError, match="content width is 132"):
        pa_bsnd_keys(packed, HEAD_DIM, 160)


def test_indexer_cache_is_constructed_at_the_logical_head_dim():
    """Do not pass the FP8-plus-scale width (head_dim + 4) into the cache."""
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parents[3] / "vllm_ascend" / "models" / "glm5next" / "attention.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    widened = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name != "Glm5NextIndexerCache":
            continue
        for keyword in node.keywords:
            if keyword.arg != "head_dim":
                continue
            widened = isinstance(keyword.value, ast.BinOp)
    assert not widened, (
        "Glm5NextIndexerCache must be constructed with the logical head_dim; "
        "widening it by the FP8 scale bytes made npu_lightning_indexer see "
        "165 pools on a 160-pool page."
    )


NUM_BLOCKS = 5


@pytest.fixture
def block_table():
    return torch.arange(16 * NUM_BLOCKS, dtype=torch.int32).reshape(16, NUM_BLOCKS)


def test_a_verify_step_gets_one_batch_entry_per_row(block_table):
    """Drafts give a request several rows, and each states its own key length.

    The operator sizes its batch from the query lengths and then demands one key
    length per entry, so counting requests there while `compact_selection`
    counts rows is what raised

        the shape size of actualSeqLengths should be equal to batch size 16
    """
    rows_per_request, num_requests = 8, 16
    num_tokens = rows_per_request * num_requests

    per_row, query_lens = rows_as_batch_entries(block_table, num_tokens, num_requests)

    assert query_lens.shape == (num_tokens,)
    assert per_row.shape == (num_tokens, NUM_BLOCKS)
    # Which is what lets compact_selection's per-row counts be passed as they are.
    assert query_lens.shape == compact_selection(torch.zeros(num_tokens, 4, dtype=torch.int32))[1].shape


def test_every_row_reads_its_own_request_blocks(block_table):
    rows_per_request = 8
    per_row, _ = rows_as_batch_entries(block_table, rows_per_request * 16, 16)

    for row in range(rows_per_request * 16):
        assert torch.equal(per_row[row], block_table[row // rows_per_request]), (
            f"row {row} was pointed at another request's pages"
        )


def test_one_query_row_per_entry_so_the_causal_crop_spans_nothing(block_table):
    """Cumulated one row at a time, every entry's query length is 1.

    The rows of a request may not see each other's tokens, which the selection
    already enforced per row; an entry holding a single row leaves the
    operator's RightDownCausal crop nothing to cover.
    """
    _, query_lens = rows_as_batch_entries(block_table, 8 * 16, 16)

    assert query_lens[0] == 1
    assert torch.equal(query_lens.diff(), torch.ones(8 * 16 - 1, dtype=query_lens.dtype))


def test_a_plain_decode_batch_is_left_as_it_was(block_table):
    """One row per request is the case where rows and requests coincide."""
    per_row, query_lens = rows_as_batch_entries(block_table, 16, 16)

    assert torch.equal(per_row, block_table)
    assert torch.equal(query_lens, torch.arange(1, 17, dtype=torch.int32))


def test_an_empty_decode_batch_does_not_divide_by_zero(block_table):
    """A prefill-only step reports no decode rows and no decode requests."""
    per_row, query_lens = rows_as_batch_entries(block_table, 0, 0)

    assert per_row.shape == (0, NUM_BLOCKS)
    assert query_lens.shape == (0,)


def test_the_batch_is_described_without_leaving_the_device(block_table):
    """A host-built length is recorded by a capture and read stale on replay."""
    per_row, query_lens = rows_as_batch_entries(block_table, 8 * 16, 16)

    assert per_row.device == block_table.device
    assert query_lens.device == block_table.device
    assert query_lens.dtype == torch.int32
