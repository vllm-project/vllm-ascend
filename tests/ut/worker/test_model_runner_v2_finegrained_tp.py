"""Unit tests for lmhead TP support in the Ascend V2 model runner.

Pure-mock tests (CPU tensors, no NPU): they lock the runner-side pad/trim
contract of sample()/execute_model(dummy) and guard the copied dispatch tail
with a canary that compares it call-by-call against upstream
GPUModelRunner.sample. They also lock the draft-side alignment: Ascend
speculators pad draft sampling rows to the same group-agreed capacity inside
sample_draft (V1 parity) and reject unsupported combinations (probabilistic
draft sampling, DSpark-style paths that bypass sample_draft, local argmax
reduction, prompt logprobs) instead of hanging.
Collective behavior of the LM head itself is validated on real hardware.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec, patch

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker

from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import AscendAutoRegressiveSpeculator


def _make_runner(max_num_reqs=8, decode_query_len=2, vocab=6):
    """Bare instance bypassing __init__ (no NPU required).

    Dispatch components are autospecced against the real upstream classes so
    a call with drifted arguments fails loudly instead of being swallowed by
    a bare MagicMock.
    """
    runner = object.__new__(NPUModelRunner)
    runner.max_num_reqs = max_num_reqs
    runner.decode_query_len = decode_query_len
    runner.device = torch.device("cpu")
    runner.is_last_pp_rank = True
    runner.execute_model_state = None
    runner.ascend_config = SimpleNamespace(scheduler_config=SimpleNamespace(profiling_chunk_config=None))
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
        pytest.raises(ValueError, match="group-agreed capacity"),
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


def _run_execute_model(runner, hidden_states, dummy_run=True, is_profile=False):
    """Drive NPUModelRunner.execute_model with the parent mocked out.

    The parent stub mimics the upstream dummy path: it publishes
    execute_model_state (as the real forward does) so the lmhead TP hook can
    read the hidden states. Timing helpers are stubbed so no profiling config
    is consulted.
    """

    def super_execute(scheduler_output, **kwargs):
        runner.execute_model_state = SimpleNamespace(hidden_states=hidden_states)
        return "upstream-output"

    with (
        patch("vllm_ascend.worker.v2.model_runner.vllm_version_is", return_value=True),
        patch("vllm_ascend.worker.v2.model_runner._start_profiling_chunk_timing", return_value=None),
        patch("vllm_ascend.worker.v2.model_runner._finish_profiling_chunk_timing", return_value=None),
        patch.object(NPUModelRunner.__bases__[0], "execute_model", side_effect=super_execute),
    ):
        return runner.execute_model(MagicMock(), dummy_run=dummy_run, is_profile=is_profile)


def test_dummy_execute_model_joins_lmhead_collectives_at_capacity():
    """Idle DP ranks must join the LM-head collectives on every dummy run.

    Regression test for the PD-disaggregation hang: with lmhead TP the LM-head
    all_gather spans the whole group, but the V2 dummy path only runs the
    model forward, so a real sample() on the rank owning requests waited
    forever. The hook must call compute_logits exactly once with zero-indexed
    rows at the same capacity the sample() override pads to, before the
    parent _dummy_run proceeds to the speculator dummy propose (busy rank
    ordering: target head first, then draft head).
    """
    runner = _make_runner(max_num_reqs=8, decode_query_len=2)  # capacity 16
    hidden_states = torch.randn(10, 6)

    with patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_enable", return_value=True):
        output = _run_execute_model(runner, hidden_states, dummy_run=True)

    assert output == "upstream-output"  # parent return value is untouched
    assert runner.model.compute_logits.call_count == 1
    dummy_input = runner.model.compute_logits.call_args.args[0]
    # zero-indexed rows gathered up to the group-agreed capacity
    assert dummy_input.shape == (16, 6)
    torch.testing.assert_close(dummy_input, hidden_states[torch.zeros(16, dtype=torch.long)])


def test_dummy_execute_model_skips_lmhead_collectives_when_gated_off():
    """Real (non-dummy) runs, profiling runs, feature-off runs, and non-last
    PP ranks must not add dummy compute_logits calls (busy ranks join the
    collectives from sample(); the profile dummy sampler already runs
    compute_logits on every rank; non-last PP ranks never produce logits)."""
    runner = _make_runner()
    hidden_states = torch.randn(10, 6)

    # Real execute_model of a busy rank: the target collectives are joined by
    # sample(), never here.
    with patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_enable", return_value=True):
        _run_execute_model(runner, hidden_states, dummy_run=False)
    runner.model.compute_logits.assert_not_called()

    with patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_enable", return_value=False):
        _run_execute_model(runner, hidden_states, dummy_run=True)
    runner.model.compute_logits.assert_not_called()

    with patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_enable", return_value=True):
        _run_execute_model(runner, hidden_states, dummy_run=True, is_profile=True)
    runner.model.compute_logits.assert_not_called()

    runner.is_last_pp_rank = False
    with patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_enable", return_value=True):
        _run_execute_model(runner, hidden_states, dummy_run=True)
    runner.model.compute_logits.assert_not_called()


class _ConcreteSpeculator(AscendAutoRegressiveSpeculator):
    """Concrete stub: the base class keeps load_draft_model abstract."""

    def load_draft_model(self, *args, **kwargs):
        raise NotImplementedError


def _make_speculator(max_num_reqs=8, num_speculative_steps=1):
    """Bare speculator bypassing __init__ (no NPU)."""
    spec = object.__new__(_ConcreteSpeculator)
    spec.max_num_reqs = max_num_reqs
    spec.num_speculative_steps = num_speculative_steps
    spec.model = MagicMock()
    spec.use_local_argmax_reduction = False
    return spec


def _one_hot_argmax_logits(hidden_states):
    """Row-projective greedy logits: argmax of row i is i % 4."""
    num_rows = hidden_states.shape[0]
    return torch.nn.functional.one_hot(torch.arange(num_rows) % 4, num_classes=4).float()


def test_sample_draft_pads_to_capacity_then_trims():
    """With lmhead TP every draft sampling call must feed the draft LM head
    the group-agreed capacity rows (busy real propose and idle dummy propose
    alike) and hand the caller back only the real rows."""
    spec = _make_speculator(max_num_reqs=8, num_speculative_steps=1)  # capacity 16
    spec.model.compute_logits.side_effect = _one_hot_argmax_logits
    hidden_states = torch.randn(3, 5)

    with patch(
        "vllm_ascend.worker.v2.spec_decode.autoregressive.speculator.lmhead_tp_enable",
        return_value=True,
    ):
        draft_tokens = spec.sample_draft(
            hidden_states, MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), None
        )

    compute_input = spec.model.compute_logits.call_args.args[0]
    assert compute_input.shape == (16, 5)  # padded to capacity
    torch.testing.assert_close(compute_input[:3], hidden_states)
    assert torch.all(compute_input[3:] == 0)  # zero padding rows
    assert draft_tokens.shape[0] == 3  # trimmed back to real rows
    torch.testing.assert_close(draft_tokens, (torch.arange(3) % 4).to(draft_tokens.dtype))


def test_sample_draft_passthrough_when_lmhead_disabled():
    spec = _make_speculator()
    spec.model.compute_logits.side_effect = _one_hot_argmax_logits
    hidden_states = torch.randn(3, 5)

    with patch(
        "vllm_ascend.worker.v2.spec_decode.autoregressive.speculator.lmhead_tp_enable",
        return_value=False,
    ):
        draft_tokens = spec.sample_draft(
            hidden_states, MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), None
        )

    spec.model.compute_logits.assert_called_once_with(hidden_states)
    assert draft_tokens.shape[0] == 3


def test_sample_draft_at_capacity_skips_padding():
    spec = _make_speculator(max_num_reqs=4, num_speculative_steps=1)  # capacity 8
    spec.model.compute_logits.side_effect = _one_hot_argmax_logits
    hidden_states = torch.randn(8, 5)

    with patch(
        "vllm_ascend.worker.v2.spec_decode.autoregressive.speculator.lmhead_tp_enable",
        return_value=True,
    ):
        draft_tokens = spec.sample_draft(
            hidden_states, MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), None
        )

    compute_input = spec.model.compute_logits.call_args.args[0]
    assert compute_input is hidden_states  # untouched, no copy
    assert draft_tokens.shape[0] == 8


def test_sample_draft_raises_when_rows_exceed_capacity():
    spec = _make_speculator(max_num_reqs=4, num_speculative_steps=1)  # capacity 8
    hidden_states = torch.randn(9, 5)

    with (
        patch(
            "vllm_ascend.worker.v2.spec_decode.autoregressive.speculator.lmhead_tp_enable",
            return_value=True,
        ),
        pytest.raises(ValueError, match="group-agreed"),
    ):
        spec.sample_draft(hidden_states, MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), None)

    spec.model.compute_logits.assert_not_called()


def test_sample_draft_rejects_probabilistic_draft_sampling():
    """The gumbel path writes into fixed-size draft buffers that cannot hold
    the padding rows; the combination must fail fast instead of hanging the
    lmhead-TP collectives."""
    spec = _make_speculator()
    hidden_states = torch.randn(3, 5)

    with (
        patch(
            "vllm_ascend.worker.v2.spec_decode.autoregressive.speculator.lmhead_tp_enable",
            return_value=True,
        ),
        pytest.raises(NotImplementedError, match="probabilistic"),
    ):
        spec.sample_draft(
            hidden_states,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            torch.zeros(4, 5, 5),  # draft_logits buffer: probabilistic path
        )

    spec.model.compute_logits.assert_not_called()


def test_speculator_init_rejects_probabilistic_with_lmhead():
    """The unsupported combination must fail at construction time, not at the
    first sampling step inside a running engine."""
    spec = _make_speculator()
    spec.speculative_config = SimpleNamespace(draft_sample_method="probabilistic")

    with (
        patch(
            "vllm_ascend.worker.v2.spec_decode.autoregressive.speculator.lmhead_tp_enable",
            return_value=True,
        ),
        pytest.raises(NotImplementedError, match="probabilistic"),
    ):
        spec._lmhead_tp_validate_draft_sampling()


def test_speculator_init_allows_greedy_with_lmhead():
    spec = _make_speculator()
    spec.speculative_config = SimpleNamespace(draft_sample_method="greedy")

    with patch(
        "vllm_ascend.worker.v2.spec_decode.autoregressive.speculator.lmhead_tp_enable",
        return_value=True,
    ):
        spec._lmhead_tp_validate_draft_sampling()  # must not raise


def test_speculator_init_rejects_sampling_paths_that_bypass_sample_draft():
    """DSpark-style speculators (flag off) must be rejected at construction:
    their draft sampling calls compute_draft_logits directly, so the mixin's
    row alignment never runs and the lmhead-TP collectives would desync."""
    spec = _make_speculator()
    spec._lmhead_tp_sample_draft_supported = False
    spec.speculative_config = SimpleNamespace(draft_sample_method="greedy")

    with (
        patch(
            "vllm_ascend.worker.v2.spec_decode.autoregressive.speculator.lmhead_tp_enable",
            return_value=True,
        ),
        pytest.raises(NotImplementedError, match="sample_draft"),
    ):
        spec._lmhead_tp_validate_draft_sampling()


def test_speculator_init_rejects_local_argmax_reduction_with_lmhead():
    """get_top_tokens reduces over the local vocab shard only (the gather is
    a no-op under pure DP), silently producing wrong tokens; fail at
    construction instead."""
    spec = _make_speculator()
    spec.speculative_config = SimpleNamespace(draft_sample_method="greedy")
    spec.use_local_argmax_reduction = True

    with (
        patch(
            "vllm_ascend.worker.v2.spec_decode.autoregressive.speculator.lmhead_tp_enable",
            return_value=True,
        ),
        pytest.raises(NotImplementedError, match="use_local_argmax_reduction"),
    ):
        spec._lmhead_tp_validate_draft_sampling()


def test_sample_tokens_rejects_prompt_logprobs_with_lmhead():
    """The prompt-logprobs worker issues a second unpadded compute_logits
    that desyncs the LM-head collectives; fail at the first affected step
    instead of hanging."""
    runner = _make_runner()
    runner.use_spec_pp = False
    runner.prompt_logprobs_worker = SimpleNamespace(uses_prompt_logprobs=np.array([False, True, False]))

    with (
        patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_enable", return_value=True),
        pytest.raises(NotImplementedError, match="prompt_logprobs"),
    ):
        runner.sample_tokens(None)


def test_sample_tokens_passthrough_when_no_prompt_logprobs():
    runner = _make_runner()
    runner.use_spec_pp = False
    runner.prompt_logprobs_worker = SimpleNamespace(uses_prompt_logprobs=np.zeros(8, dtype=bool))

    with (
        patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_enable", return_value=True),
        patch.object(
            NPUModelRunner.__bases__[0], "sample_tokens", return_value="upstream-result"
        ) as super_sample_tokens,
    ):
        result = runner.sample_tokens(None)

    assert result == "upstream-result"
    super_sample_tokens.assert_called_once_with(None)


def test_draft_capacity_formula_matches_runner():
    """The speculator and the runner must derive the identical group-agreed
    capacity from the same config, otherwise the draft and target LM-head
    collectives desync and hang."""
    spec = _make_speculator(max_num_reqs=8, num_speculative_steps=3)
    runner = _make_runner(max_num_reqs=8, decode_query_len=4)
    assert spec._lmhead_tp_max_num_logits() == 32
    assert runner._lmhead_tp_max_num_logits() == 32
