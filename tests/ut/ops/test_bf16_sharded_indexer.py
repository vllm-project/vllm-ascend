# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.ut.ops.test_dcp_lse_batch_cpu import Pointer, TorchLanguage, load_function
from vllm_ascend.ops import bf16_sharded_indexer


class _RestoreLanguage(TorchLanguage):
    """Model Ascend's byte page loads and Triton's i1 store-mask requirement."""

    def program_id(self, axis):
        return torch.tensor(self.index[axis], dtype=torch.int32)

    @staticmethod
    def minimum(left, right):
        return torch.minimum(torch.as_tensor(left), torch.as_tensor(right))

    @staticmethod
    def maximum(left, right):
        return torch.maximum(torch.as_tensor(left), torch.as_tensor(right))

    def load(self, pointer, mask=None, other=0):
        if mask is None:
            mask = torch.ones_like(torch.as_tensor(pointer.offset), dtype=torch.bool)
        result = super().load(pointer, mask, other)
        return result.to(torch.uint8) if result.dtype == torch.bool else result

    def store(self, pointer, value, mask=None):
        if mask is None:
            mask = torch.ones_like(torch.as_tensor(pointer.offset), dtype=torch.bool)
        assert mask.dtype == torch.bool, "Mask must have boolean scalar type"
        super().store(pointer, torch.as_tensor(value), mask)


def install_li_v2_cpu_kernels(monkeypatch):
    """Execute the production LI V2 launchers and kernels against CPU tensors."""
    language = _RestoreLanguage(programs=1)
    triton = SimpleNamespace(cdiv=lambda x, y: (x + y - 1) // y, next_power_of_2=lambda x: 1 << (x - 1).bit_length())
    module = ModuleType("vllm_ascend.ops.triton.sharded_indexer")

    class Kernel:
        def __init__(self, name):
            self.function = load_function("../sharded_indexer.py", name, {"tl": language})

        def __getitem__(self, grid):
            def launch(*args):
                args = [Pointer(x) if isinstance(x, torch.Tensor) else x for x in args]
                for index in itertools.product(*(range(dim) for dim in grid)):
                    language.index = index
                    self.function(*args)

            return launch

    for name in ("build_li_v2_batched_inputs", "pack_li_v2_window_queries", "restore_li_v2_scores"):
        scope = {"torch": torch, "triton": triton, "_" + name: Kernel("_" + name)}
        setattr(module, name, load_function("../sharded_indexer.py", name, scope))
    monkeypatch.setitem(sys.modules, module.__name__, module)


@pytest.mark.parametrize("cols,windows,pages_per_window", [(1, 1, 1), (133, 3, 45)])
@pytest.mark.parametrize("for_dcp_merge,weight_dtype", [(False, torch.bfloat16), (True, torch.float32)])
def test_batched_li_v2_calls_native_once_and_selects_earliest_tied_ids(
    monkeypatch, cols, windows, pages_per_window, for_dcp_merge, weight_dtype
):
    from vllm_ascend.ops import lightning_indexer_v2 as li

    install_li_v2_cpu_kernels(monkeypatch)
    rows, page_size, topk = 5, 128, 2048
    width = pages_per_window * page_size
    capacity = cols * page_size
    table = torch.empty((2, 2 * cols), dtype=torch.int32)[:, ::2]
    table[0] = torch.arange(cols) % 5 + 2
    table[1] = 3
    if cols > 1:
        table[0, [0, 44, 89, 130]] = torch.tensor([0, -1, 1, 8], dtype=torch.int32)
    owners = torch.tensor([0, 1, 1, 1, -1], dtype=torch.int32)
    lengths = torch.tensor([capacity + 7, capacity - 5, 3, 0, capacity], dtype=torch.int32)
    query = torch.arange(rows * 4, dtype=torch.bfloat16).view(rows, 2, 2)[:, :, 1:]
    weights = torch.ones(rows, 4, dtype=weight_dtype)[:, ::2]
    weights[1].zero_()
    key = torch.arange(8, dtype=torch.bfloat16).view(8, 1, 1, 1).expand(-1, page_size, -1, -1)
    key[0, 0] = 100  # A hole redirected to page zero would otherwise win.
    metadata = Mock(return_value=torch.zeros(1024, dtype=torch.int32))
    monkeypatch.setattr(torch.ops._C_ascend, "npu_lightning_indexer_v2_metadata", metadata, raising=False)
    plan = li.build_lightning_indexer_v2_plan(table, owners, lengths, 8, page_size, 2, 2, metadata_size=1024)
    assert (plan.windows, plan.capacity) == (windows, capacity)
    metadata.assert_called_once_with(plan.cu_query, plan.visible_lengths, 2, width, metadata_size=1024)
    torch.testing.assert_close(plan.cu_query, torch.arange(rows * windows + 1, dtype=torch.int32))
    if cols == 133:
        assert not plan.valid_pages[-rows:, -2:].any()
        assert plan.visible_lengths[-rows].item() == 43 * page_size

    def native(q, k, w, cu, visible, pages, tiling, native_topk):
        assert cu is plan.cu_query and pages is plan.block_table and visible is plan.visible_lengths
        assert k is key and tiling is metadata.return_value and native_topk == width
        torch.testing.assert_close(q, query.repeat(windows, 1, 1))
        torch.testing.assert_close(w, weights.float().repeat(windows, 1))
        assert q.is_contiguous() and w.is_contiguous()
        # The native order is arbitrary, including ties. Poison its padding.
        ids = torch.arange(width - 1, -1, -1, dtype=torch.int32).expand(rows * windows, width)
        scores = key[pages.gather(1, ids.long() // page_size).long(), 0, 0, 0].float()
        scores *= (q[:, :, 0].float() * w).sum(dim=1, keepdim=True)
        invalid = ids >= visible[:, None]
        return ids.masked_fill(invalid, -1).unsqueeze(1), scores.masked_fill(invalid, torch.inf).unsqueeze(1)

    scorer = Mock(side_effect=native)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_lightning_indexer_v2", scorer, raising=False)
    ids, scores = li.lightning_indexer_v2_candidates(query, weights, key, plan, topk, for_dcp_merge=for_dcp_merge)
    scorer.assert_called_once()
    assert ids.dtype == (torch.int64 if for_dcp_merge else torch.int32)
    assert scores.dtype == torch.float32
    for row in range(rows):
        candidates = []
        if 0 <= owners[row] < table.shape[0]:
            # The reference walks logical IDs directly, without splitting windows.
            scale = (query[row, :, 0].float() * weights[row].float()).sum().item()
            page_ids = table[owners[row]].tolist()
            for token in range(min(lengths[row].item(), capacity)):
                page = page_ids[token // page_size]
                if 2 <= page < key.shape[0]:
                    candidates.append((scale * page, token))
        expected = sorted(candidates, key=lambda pair: (-pair[0], pair[1]))[:topk]
        count = len(expected)
        assert ids[row, 0, :count].tolist() == [token for _, token in expected]
        assert scores[row, 0, :count].tolist() == [score for score, _ in expected]
        assert torch.isneginf(scores[row, 0, count:]).all()
        if not for_dcp_merge:
            assert (ids[row, 0, count:] == -1).all()


@pytest.mark.parametrize("rows,cols", [(0, 133), (3, 0)])
def test_empty_li_v2_plan_skips_native_metadata_and_scorer(monkeypatch, rows, cols):
    from vllm_ascend.ops import lightning_indexer_v2 as li

    install_li_v2_cpu_kernels(monkeypatch)
    forbidden = Mock(side_effect=AssertionError("empty plan launched native op"))
    for name in ("npu_lightning_indexer_v2", "npu_lightning_indexer_v2_metadata"):
        monkeypatch.setattr(torch.ops._C_ascend, name, forbidden, raising=False)
    owners = torch.zeros(rows, dtype=torch.int32)
    table = torch.empty(1, cols, dtype=torch.int32)
    plan = li.build_lightning_indexer_v2_plan(table, owners, owners, 1, 128, 1, 2)
    q = torch.empty(rows, 2, 1, dtype=torch.bfloat16)
    ids, scores = li.lightning_indexer_v2_candidates(q, q[..., 0], q.new_empty(1, 128, 1, 1), plan, 2048)
    assert ids.shape == scores.shape == (rows, 1, 2048)
    assert (ids == -1).all() and torch.isneginf(scores).all()
    forbidden.assert_not_called()


@pytest.mark.parametrize("width", [128, 384, 8192])
def test_restore_li_v2_scores_with_byte_page_flags(width):
    rows, page_size, offset, block = 3, 128, 7, 256
    pages = width // page_size
    ids = torch.arange(width, dtype=torch.int32).flip(0).repeat(rows, 1)
    ids[0, :3] = torch.tensor([-1, width, width + 3])
    scores = torch.arange(rows * width, dtype=torch.float32).reshape(rows, width) - width
    # Views of per-row windows have a nonzero storage offset and full-table stride.
    flags = torch.zeros((rows, pages + 3), dtype=torch.uint8)[:, 1 : 1 + pages]
    flags[:, ::2] = 1
    lengths = torch.tensor([width, width - 17, 0], dtype=torch.int32)
    dense = torch.full((rows, offset + width + 3), -torch.inf)
    expected = dense.clone()
    for row in range(rows):
        token_ids = ids[row]
        valid = (token_ids >= 0) & (token_ids < lengths[row])
        positions = valid.nonzero().flatten()
        positions = positions[flags[row, token_ids[positions] // page_size].bool()]
        expected[row, offset + token_ids[positions]] = scores[row, positions]

    language = _RestoreLanguage(programs=rows)
    kernel = load_function("../sharded_indexer.py", "_restore_li_v2_scores", {"tl": language})
    for row in range(rows):
        for tile in range((width + block - 1) // block):
            language.index = row, tile
            kernel(
                Pointer(ids),
                Pointer(scores),
                Pointer(flags),
                Pointer(lengths),
                Pointer(dense),
                offset,
                dense.stride(0),
                flags.stride(0),
                width,
                page_size,
                block,
            )
    torch.testing.assert_close(dense, expected)


@pytest.mark.parametrize("topk", [2, 20])
@pytest.mark.parametrize("full_graph", [False, True])
@pytest.mark.parametrize("use_shared_rows", [False, True])
def test_bf16_paged_candidates_match_independent_causal_reference(monkeypatch, topk, full_graph, use_shared_rows):
    monkeypatch.setattr(bf16_sharded_indexer, "_KEY_TOKENS_PER_TILE", 4)
    generator = torch.Generator().manual_seed(23)
    query = torch.randn((8, 3, 6), generator=generator).bfloat16()
    weights = torch.randn((8, 3), generator=generator).bfloat16()
    query[2].zero_()  # Tied scores must be ordered by actual token ID.
    key = torch.randn((10, 4, 1, 6), generator=generator).bfloat16()
    key[:2].fill_(999)
    # Pages 0 and 1 belong to the physical null block. Page 10 is out of range.
    table = torch.tensor([[4, 2, 3], [2, 3, 4], [7, 1, 5], [0, 6, 10]], dtype=torch.int32)
    ends = torch.tensor([2, 2, 5, 7], dtype=torch.int32)
    visible = torch.tensor([[1], [7], [5], [6], [9], [0], [10], [12]], dtype=torch.int32)
    request_rows = None
    if use_shared_rows:
        # The fused metadata producer skips the empty request and maps graph
        # padding to the last table row with an empty visible prefix.
        request_rows = torch.tensor([0, 0, 2, 2, 2, 3, 3, 3], dtype=torch.int32)
        visible[-1] = 0
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual_ids, actual_scores = bf16_sharded_indexer.bf16_paged_indexer_candidates(
            query,
            weights,
            key,
            table,
            ends,
            visible,
            topk,
            request_rows=request_rows,
            first_valid_page=2,
            full_graph=full_graph,
            query_rows_per_tile=2,
        )

    expected_ids = torch.full_like(actual_ids, -1)
    expected_scores = torch.full_like(actual_scores, -torch.inf)
    for row in range(8):
        request = next((req for req, end in enumerate(ends) if row < end), None)
        if request is None:
            continue
        candidates = []
        for token in range(int(visible[row, 0])):
            page, offset = divmod(token, 4)
            physical = int(table[request, page])
            if not 2 <= physical < key.shape[0]:
                continue
            dots = (query[row].float() * key[physical, offset, 0].float()).sum(dim=-1)
            score = (dots.relu() * weights[row].float()).sum().item()
            candidates.append((score, token))
        for index, (score, token) in enumerate(sorted(candidates, key=lambda pair: (-pair[0], pair[1]))[:topk]):
            expected_ids[row, 0, index] = token
            expected_scores[row, 0, index] = score

    torch.testing.assert_close(actual_ids, expected_ids)
    torch.testing.assert_close(actual_scores, expected_scores)


@pytest.mark.parametrize("full_graph", [False, True])
def test_16_row_tile_preserves_exact_scores_and_ties_across_tiles(monkeypatch, full_graph):
    monkeypatch.setattr(bf16_sharded_indexer, "_KEY_TOKENS_PER_TILE", 4)
    generator = torch.Generator().manual_seed(51)
    query = torch.randint(-2, 3, (37, 3, 8), generator=generator).bfloat16()
    query[15:18].zero_()
    weights = torch.ones((37, 3), dtype=torch.bfloat16)
    key = torch.randint(-2, 3, (5, 4, 1, 8), generator=generator).bfloat16()
    args = (query, weights, key, torch.tensor([[4, 2, 3]]), torch.tensor([37]), torch.full((37,), 12), 7)
    reference = bf16_sharded_indexer.bf16_paged_indexer_candidates(*args, full_graph=full_graph)
    ids, scores = bf16_sharded_indexer.bf16_paged_indexer_candidates(
        *args, full_graph=full_graph, query_rows_per_tile=16
    )
    torch.testing.assert_close(ids, reference[0], rtol=0, atol=0)
    torch.testing.assert_close(scores, reference[1], rtol=0, atol=0)
    assert ids[15:18, 0].tolist() == [list(range(7))] * 3


def test_bf16_paged_candidates_handle_ranks_with_no_visible_pages():
    query = torch.ones((2, 1, 4), dtype=torch.bfloat16)
    ids, scores = bf16_sharded_indexer.bf16_paged_indexer_candidates(
        query,
        query[..., 0],
        torch.ones((2, 4, 1, 4), dtype=query.dtype),
        torch.empty((1, 0), dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        3,
    )
    torch.testing.assert_close(ids, torch.full((2, 1, 3), -1, dtype=torch.int32))
    torch.testing.assert_close(scores, torch.full((2, 1, 3), -torch.inf))


@pytest.mark.parametrize("full_graph", [False, True])
@pytest.mark.parametrize("topk", [4, 6])
def test_paged_candidates_preserve_nonfinite_cutoff_and_finite_ties(monkeypatch, full_graph, topk):
    monkeypatch.setattr(bf16_sharded_indexer, "_KEY_TOKENS_PER_TILE", 2)
    query = torch.ones((2, 1, 1), dtype=torch.bfloat16)
    weights = torch.tensor([[1], [-1]], dtype=torch.bfloat16)
    key = torch.tensor([0, 0, torch.nan, 2, 4, torch.inf, 4, 3], dtype=torch.bfloat16).view(4, 2, 1, 1)
    ids, scores = bf16_sharded_indexer.bf16_paged_indexer_candidates(
        query,
        weights,
        key,
        torch.tensor([[1, 2, 3]]),
        torch.tensor([2]),
        torch.tensor([6, 6]),
        topk,
        full_graph=full_graph,
    )
    expected_ids = torch.tensor([[0, 3, 2, 4, 5, 1], [0, 1, 5, 2, 4, -1]], dtype=torch.int32)
    expected_scores = torch.tensor([[torch.nan, torch.inf, 4, 4, 3, 2], [torch.nan, -2, -3, -4, -4, -torch.inf]])
    torch.testing.assert_close(ids[:, 0], expected_ids[:, :topk])
    torch.testing.assert_close(scores[:, 0], expected_scores[:, :topk], rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("full_graph", [False, True])
def test_paged_candidates_follow_replaced_table_and_shorter_causal_lengths(monkeypatch, full_graph):
    # A captured step can change both its page mapping and active rows while
    # keeping the input addresses and logical page-table extent fixed.
    from torch.fx.experimental.proxy_tensor import make_fx

    monkeypatch.setattr(bf16_sharded_indexer, "_KEY_TOKENS_PER_TILE", 4)
    query = torch.ones((3, 1, 1), dtype=torch.bfloat16)
    weights = query[..., 0]
    key = torch.arange(8, dtype=torch.bfloat16).view(4, 2, 1, 1)
    table = torch.tensor([[2, 1, 3], [3, 2, 1]], dtype=torch.int32)
    ends = torch.tensor([2, 3], dtype=torch.int32)
    visible = torch.full((3,), 6, dtype=torch.int32)

    def select(table, ends, visible):
        return bf16_sharded_indexer.bf16_paged_indexer_candidates(
            query, weights, key, table, ends, visible, 8, full_graph=full_graph, query_rows_per_tile=2
        )

    captured = make_fx(select)(table, ends, visible)
    table.copy_(torch.tensor([[0, 3, 1], [1, 2, 3]]))
    ends.copy_(torch.tensor([1, 2]))
    visible.copy_(torch.tensor([5, 3, 6]))
    ids, scores = captured(table, ends, visible)
    assert ids[:, 0].tolist() == [[3, 2, 4] + [-1] * 5, [2, 1, 0] + [-1] * 5, [-1] * 8]
    assert scores[:, 0].tolist() == [[7, 6, 2] + [-torch.inf] * 5, [4, 3, 2] + [-torch.inf] * 5, [-torch.inf] * 8]
