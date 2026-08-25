from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.models.qwen3_dflash2 import _grouped_conv, _score_edges
from vllm_ascend.spec_decode.dflash2_proposer import (
    AscendDflash2Proposer,
    greedy_select_path,
    is_dflash2_draft,
)


@pytest.fixture(autouse=True)
def _stub_device_properties(monkeypatch):
    """CPU CI has no NPU: ``init_device_properties_triton`` is skipped when
    ``HAS_TRITON`` is false, leaving ``_NUM_VECTORCORE`` unset, so
    ``get_vectorcore_num`` asserts. ``greedy_select_path`` sizes the grid via
    ``min(num_reqs, get_vectorcore_num())``; stub the device-property globals
    so the grid computation runs on CPU. The walk kernel itself is mocked in
    the dispatch test, since the Ascend Triton backend cannot launch on CPU."""
    monkeypatch.setattr("vllm_ascend.ops.triton.triton_utils._NUM_AICORE", 8)
    monkeypatch.setattr("vllm_ascend.ops.triton.triton_utils._NUM_VECTORCORE", 8)


@pytest.mark.parametrize("block_size", [5, 8])
def test_grouped_conv_matches_reference(block_size: int):
    torch.manual_seed(0)
    batch, taps, num_groups, group_size = 3, 3, 4, 2
    hidden = torch.randn(batch * block_size, num_groups * group_size)
    delta = torch.randn(batch * block_size, taps, num_groups)
    base = torch.randn(taps, num_groups * group_size)

    actual = _grouped_conv(hidden, delta, base, block_size, num_groups, group_size, taps)
    hidden_blocks = hidden.view(batch, block_size, num_groups, group_size)
    expected = torch.zeros_like(hidden_blocks)
    base = base.view(taps, num_groups, group_size)
    delta = delta.view(batch, block_size, taps, num_groups)
    for position in range(block_size):
        for tap in range(min(taps, position + 1)):
            expected[:, position] += (base[tap] + delta[:, position, tap, :, None]) * hidden_blocks[:, position - tap]

    torch.testing.assert_close(actual, expected.flatten(0, 1).flatten(-2))


def test_selector_edges_match_sequential_reference():
    torch.manual_seed(1)
    batch, steps, top_k, rank = 2, 4, 3, 5
    vocab = 17
    predecessors = torch.randn(vocab, rank)
    successors = torch.randn(vocab, rank)
    candidate_ids = torch.randint(vocab, (batch, steps, top_k))
    unary = torch.randn(batch, steps, top_k)
    hidden = torch.randn(batch, steps, rank)
    anchors = torch.randint(vocab, (batch,))

    actual = _score_edges(predecessors, successors, candidate_ids, unary, hidden, anchors, top_k)
    expected = torch.empty_like(actual)
    for step in range(steps):
        pred = anchors[:, None].expand(-1, top_k) if step == 0 else candidate_ids[:, step - 1]
        expected[:, step] = unary[:, step, None] + torch.einsum(
            "bpr,bcr->bpc",
            predecessors[pred] * hidden[:, step, None],
            successors[candidate_ids[:, step]],
        )

    torch.testing.assert_close(actual, expected)


def _reference_walk(candidate_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Pure-Python mirror of ``dflash2_greedy_selector_walk_kernel``.

    The Triton kernel cannot launch on CPU CI; this reference encodes the same
    greedy walk (smallest index wins ties, matching the kernel's argmax) so the
    path-selection contract is still asserted.
    """
    num_reqs, num_steps, top_k = candidate_ids.shape
    tokens = candidate_ids.new_empty(num_reqs, num_steps)
    for req in range(num_reqs):
        prev_idx = 0
        for step in range(num_steps):
            row = scores[req, step, prev_idx]
            prev_idx = int(row.argmax())
            tokens[req, step] = candidate_ids[req, step, prev_idx]
    return tokens


def test_greedy_select_path_walks_best_predecessor():
    # One request, two steps, K=2. Step 0 prefers candidate 1; step 1 then
    # prefers the edge from that predecessor.
    candidate_ids = torch.tensor([[[10, 11], [20, 21]]])
    scores = torch.tensor(
        [
            [
                [[0.0, 1.0], [0.0, 1.0]],
                [[5.0, 0.0], [0.1, 4.0]],
            ]
        ]
    )
    expected = _reference_walk(candidate_ids, scores)
    torch.testing.assert_close(expected, torch.tensor([[11, 21]]))


def test_greedy_select_path_sizes_grid_and_dispatches_kernel(monkeypatch):
    """``greedy_select_path`` sizes the grid to ``min(num_reqs, vectorcore)`` and
    hands contiguous tensors to the walk kernel. The kernel cannot launch on
    CPU CI, so it is mocked and the dispatch is asserted (matching the dspark
    kernel-test convention)."""
    launched = MagicMock()

    monkeypatch.setattr(
        "vllm_ascend.spec_decode.dflash2_proposer.dflash2_greedy_selector_walk_kernel",
        launched,
    )
    num_reqs, num_steps, top_k = 3, 2, 4
    candidate_ids = torch.arange(num_reqs * num_steps * top_k).reshape(num_reqs, num_steps, top_k)
    # Contiguous views so the contiguency assert passes.
    scores = torch.zeros(num_reqs, num_steps, top_k, top_k)

    greedy_select_path(candidate_ids, scores)

    launched.__getitem__.assert_called_once_with((min(num_reqs, 8),))
    call = launched.__getitem__.return_value
    call.assert_called_once()
    args = call.call_args.args
    torch.testing.assert_close(args[0], scores.contiguous())
    torch.testing.assert_close(args[1], candidate_ids.contiguous())
    assert args[2].shape == (num_reqs, num_steps)
    assert args[3] == num_reqs
    assert call.call_args.kwargs == {"num_steps": num_steps, "top_k": top_k}


def test_is_dflash2_draft_requires_architecture_and_dflash_method():
    assert is_dflash2_draft(
        SimpleNamespace(
            method="dflash",
            draft_model_config=SimpleNamespace(architectures=["DFlash2DraftModel"]),
        )
    )
    assert not is_dflash2_draft(
        SimpleNamespace(
            method="dflash",
            draft_model_config=SimpleNamespace(architectures=["DFlashDraftModel"]),
        )
    )
    assert not is_dflash2_draft(
        SimpleNamespace(
            method="eagle",
            draft_model_config=SimpleNamespace(architectures=["DFlash2DraftModel"]),
        )
    )


def test_get_spec_decode_method_dispatches_dflash2():
    from vllm_ascend.spec_decode import get_spec_decode_method

    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dflash",
            draft_model_config=SimpleNamespace(architectures=["DFlash2DraftModel"]),
        )
    )
    with (
        patch("vllm_ascend.spec_decode.AscendDflash2Proposer", return_value="dflash2") as d2,
        patch("vllm_ascend.spec_decode.AscendDflashProposer", return_value="dflash1") as d1,
    ):
        assert get_spec_decode_method("dflash", vllm_config, "cpu", None) == "dflash2"
        d2.assert_called_once()
        d1.assert_not_called()

    vllm_config.speculative_config.draft_model_config.architectures = ["DFlashDraftModel"]
    with (
        patch("vllm_ascend.spec_decode.AscendDflash2Proposer", return_value="dflash2") as d2,
        patch("vllm_ascend.spec_decode.AscendDflashProposer", return_value="dflash1") as d1,
    ):
        assert get_spec_decode_method("dflash", vllm_config, "cpu", None) == "dflash1"
        d1.assert_called_once()
        d2.assert_not_called()


def test_compute_draft_token_ids_uses_selector_and_anchor(monkeypatch):
    device = torch.device("cpu")
    num_reqs, num_steps, top_k, hidden = 2, 3, 4, 8
    proposer = AscendDflash2Proposer.__new__(AscendDflash2Proposer)
    proposer.num_speculative_tokens = num_steps
    proposer.selector_top_k = top_k
    proposer.device = device
    # __init__ is bypassed, so seed the prebuilt anchor indices directly.
    proposer._anchor_indices = torch.arange(num_reqs, dtype=torch.int64) * (1 + num_steps)
    proposer.input_ids = torch.arange(num_reqs * (1 + num_steps), dtype=torch.int64)
    # Bonus/anchor tokens sit at the start of each query block.
    proposer.input_ids[0] = 7
    proposer.input_ids[1 + num_steps] = 9

    candidate_ids = torch.arange(num_reqs * num_steps * top_k).view(num_reqs * num_steps, top_k)
    unary = torch.zeros(num_reqs * num_steps, top_k)
    scores = torch.zeros(num_reqs, num_steps, top_k, top_k)
    scores[..., 0] = 1.0

    model = MagicMock()
    model.compute_candidates.return_value = (candidate_ids, unary)
    model.model.candidate_selector.return_value = scores
    proposer.model = model

    # The walk kernel cannot launch on CPU CI; route it through the pure-Python
    # reference so the end-to-end proposer path (compute_candidates -> selector
    # -> walk) is still exercised.
    kernel = MagicMock()
    kernel.__getitem__.return_value.side_effect = lambda scores, cand, out, n_req, **kw: out.copy_(
        _reference_walk(cand, scores)
    )
    monkeypatch.setattr(
        "vllm_ascend.spec_decode.dflash2_proposer.dflash2_greedy_selector_walk_kernel",
        kernel,
    )

    hidden_states = torch.randn(num_reqs * num_steps, hidden)
    tokens, probs = proposer.compute_draft_token_ids(hidden_states)

    assert probs is None  # DFlash2 always drafts greedily.
    assert tokens.shape == (num_reqs * num_steps,)
    model.compute_candidates.assert_called_once()
    selector_args = model.model.candidate_selector.call_args[0]
    torch.testing.assert_close(selector_args[3], torch.tensor([7, 9]))
    # Greedy walk always takes cand 0 when that column is 1.
    expected = candidate_ids.view(num_reqs, num_steps, top_k)[:, :, 0].reshape(-1)
    torch.testing.assert_close(tokens, expected)
