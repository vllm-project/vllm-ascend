"""Unit tests for lmhead TP support in the Ascend V2 model runner.

Pure-mock tests (CPU tensors, no NPU): they lock the runner-side pad/trim
contract of sample()/_dummy_run and guard the copied dispatch tail with a
canary that compares it call-by-call against upstream GPUModelRunner.sample.
Collective behavior of the LM head itself is validated on real hardware.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec, patch

import pytest
import torch
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker

from vllm_ascend.worker.v2.model_runner import NPUModelRunner


def _make_runner(max_num_reqs=8, decode_query_len=2, vocab=6):
    """Bare instance bypassing __init__ (no NPU required).

    Dispatch components are autospecced against the real upstream classes so
    a call with drifted arguments fails loudly instead of being swallowed by
    a bare MagicMock.
    """
    runner = object.__new__(NPUModelRunner)
    runner.max_num_reqs = max_num_reqs
    runner.decode_query_len = decode_query_len
    runner.batch_sharder = None
    runner.model = MagicMock()
    runner.model.compute_logits.side_effect = lambda x: torch.zeros(x.shape[0], vocab)
    runner.sampler = create_autospec(Sampler, instance=True)
    runner.rejection_sampler = create_autospec(RejectionSampler, instance=True)
    runner.speculator = MagicMock()
    runner.structured_outputs_worker = create_autospec(StructuredOutputsWorker, instance=True)
    return runner


def _make_input_batch(logits_indices):
    return SimpleNamespace(
        logits_indices=logits_indices,
        num_reqs=len(logits_indices),
        num_draft_tokens=0,
    )


def test_passthrough_when_lmhead_tp_disabled():
    runner = _make_runner()
    hidden_states = torch.randn(10, 4)
    input_batch = _make_input_batch(torch.tensor([0, 3, 5]))
    grammar_output = MagicMock()

    with (
        patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_enable", return_value=False),
        patch.object(NPUModelRunner.__bases__[0], "sample") as super_sample,
    ):
        super_sample.return_value = "upstream-result"
        result = runner.sample(hidden_states, input_batch, grammar_output)

    assert result == "upstream-result"
    super_sample.assert_called_once_with(hidden_states, input_batch, grammar_output)
    runner.model.compute_logits.assert_not_called()
    runner.sampler.assert_not_called()


@pytest.mark.parametrize("at_capacity", [False, True])
def test_lmhead_tp_pads_to_capacity_then_trims(at_capacity):
    if at_capacity:
        runner = _make_runner(max_num_reqs=4, decode_query_len=2)  # capacity 8
        indices = torch.arange(8)
    else:
        runner = _make_runner(max_num_reqs=8, decode_query_len=2)  # capacity 16
        indices = torch.tensor([0, 3, 5])
    capacity = runner._lmhead_tp_max_num_logits()
    num_logits = indices.shape[0]
    hidden_dim = 4
    hidden_states = torch.randn(10, hidden_dim)
    input_batch = _make_input_batch(indices)

    with patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_enable", return_value=True):
        result = runner.sample(hidden_states, input_batch, None)

    compute_input = runner.model.compute_logits.call_args.args[0]
    # compute_logits sees the group-agreed capacity, not the real row count
    assert compute_input.shape == (capacity, hidden_dim)
    # real rows are the indexed hidden states, padding rows are zero
    torch.testing.assert_close(compute_input[:num_logits], hidden_states[indices])
    assert torch.all(compute_input[num_logits:] == 0)
    # the sampler only sees the trimmed real rows
    sampled_logits = runner.sampler.call_args.args[0]
    assert sampled_logits.shape[0] == num_logits
    # return contract mirrors upstream sample()
    sampler_output = runner.sampler.return_value
    assert result[0] is sampler_output
    assert result[1] is sampler_output.num_sampled
    assert result[2] is sampler_output.num_rejected


def test_lmhead_tp_raises_when_logits_exceed_capacity():
    runner = _make_runner(max_num_reqs=8, decode_query_len=2)  # capacity 16
    input_batch = _make_input_batch(torch.arange(17))

    with (
        patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_enable", return_value=True),
        pytest.raises(AssertionError, match="group-agreed capacity"),
    ):
        runner.sample(torch.randn(20, 4), input_batch, None)

    runner.model.compute_logits.assert_not_called()


def _canary_tail_calls(parent):
    """Dispatch-tail calls recorded on one parent mock, in global order.

    Tensor arguments are normalized to (shape, values) so calls from the two
    runs can be compared for equality.
    """
    calls = []
    for call in parent.mock_calls:
        name = call[0]
        args = tuple((a.shape, tuple(a.flatten().tolist())) if isinstance(a, torch.Tensor) else a for a in call[1])
        kwargs = {
            k: (v.shape, tuple(v.flatten().tolist())) if isinstance(v, torch.Tensor) else v for k, v in call[2].items()
        }
        calls.append((name, args, kwargs))
    return calls


@pytest.mark.parametrize(
    "with_grammar, with_draft",
    [
        (False, False),  # plain sampler branch
        (True, False),  # grammar bitmask + sampler
        (False, True),  # rejection sampler branch
    ],
)
def test_dispatch_tail_canary_matches_upstream_sample(with_grammar, with_draft):
    """Main2main canary: with lmhead TP on, the override must drive the
    dispatch tail (grammar bitmask / sampler / rejection sampler) exactly like
    upstream GPUModelRunner.sample — same calls, same order, same arguments
    (upstream's logits are bitwise identical to the override's trimmed
    logits). If upstream sample() gains a dispatch branch or changes its
    calling contract, this comparison fails and the copied tail in the
    override must be refreshed.
    """
    runner = _make_runner(max_num_reqs=8, decode_query_len=2)  # capacity 16
    # One parent mock holds the three dispatch components so calls are
    # recorded in global order across them.
    parent = MagicMock()
    runner.sampler = parent.sampler
    runner.rejection_sampler = parent.rejection_sampler
    runner.structured_outputs_worker = parent.structured_outputs_worker
    # Row-projective compute_logits: the override's trimmed logits are then
    # bitwise identical to upstream's (padding only appends zero rows).
    hidden_dim = vocab = 6
    runner.model.compute_logits.side_effect = lambda x: x[:, :vocab]

    hidden_states = torch.randn(10, hidden_dim)
    input_batch = _make_input_batch(torch.tensor([0, 3, 5]))
    if with_draft:
        input_batch.num_draft_tokens = 5
    grammar_output = MagicMock() if with_grammar else None

    GPUModelRunner.sample(runner, hidden_states, input_batch, grammar_output)
    upstream_calls = _canary_tail_calls(parent)
    parent.reset_mock()

    with patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_enable", return_value=True):
        runner.sample(hidden_states, input_batch, grammar_output)
    override_calls = _canary_tail_calls(parent)

    # Sanity floor: the canary must have actually exercised the tail.
    expected_calls = 2 if with_grammar else 1
    assert len(upstream_calls) == expected_calls
    assert override_calls == upstream_calls


def test_dummy_run_joins_lmhead_collectives_at_capacity():
    """Idle DP ranks must join the LM-head collectives on every dummy batch.

    Regression test for the PD-disaggregation hang: with lmhead TP the LM-head
    all_gather spans the whole group, but the V2 dummy path only runs the
    model forward, so a real sample() on the rank owning requests waited
    forever. The override must call compute_logits exactly once with
    zero-indexed rows at the same capacity the sample() override pads to.
    """
    runner = _make_runner(max_num_reqs=8, decode_query_len=2)  # capacity 16
    hidden_states = torch.randn(10, 6)
    sample_hidden = torch.randn(3, 6)

    with (
        patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_enable", return_value=True),
        patch.object(NPUModelRunner.__bases__[0], "_dummy_run") as super_dummy,
    ):
        super_dummy.return_value = (hidden_states, sample_hidden)
        result = runner._dummy_run(4, uniform_decode=True)

    super_dummy.assert_called_once()
    assert runner.model.compute_logits.call_count == 1
    dummy_input = runner.model.compute_logits.call_args.args[0]
    # zero-indexed rows gathered up to the group-agreed capacity
    assert dummy_input.shape == (16, 6)
    torch.testing.assert_close(dummy_input, hidden_states[torch.zeros(16, dtype=torch.long)])
    # return contract is a pure passthrough of the parent's values
    assert result == (hidden_states, sample_hidden)


def test_dummy_run_lmhead_disabled_or_profile_skips_collectives():
    """Feature off, profiling runs, and non-last PP ranks must not add dummy
    compute_logits calls (the profile dummy sampler already runs
    compute_logits on every rank; non-last PP ranks never produce logits)."""
    runner = _make_runner()
    hidden_states = torch.randn(10, 6)

    with (
        patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_enable", return_value=False),
        patch.object(NPUModelRunner.__bases__[0], "_dummy_run") as super_dummy,
    ):
        super_dummy.return_value = (hidden_states, None)
        runner._dummy_run(4)
    runner.model.compute_logits.assert_not_called()

    with (
        patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_enable", return_value=True),
        patch.object(NPUModelRunner.__bases__[0], "_dummy_run") as super_dummy,
    ):
        super_dummy.return_value = (hidden_states, None)
        runner._dummy_run(4, is_profile=True)
    runner.model.compute_logits.assert_not_called()

    with (
        patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_enable", return_value=True),
        patch.object(NPUModelRunner.__bases__[0], "_dummy_run") as super_dummy,
    ):
        super_dummy.return_value = (None, None)
        runner._dummy_run(4)
    runner.model.compute_logits.assert_not_called()
