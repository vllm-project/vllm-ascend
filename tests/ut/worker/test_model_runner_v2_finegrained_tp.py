"""Unit tests for fine-grained TP support in the Ascend V2 model runner.

Pure-mock tests (CPU tensors, no NPU): they lock the runner-side pad/trim
contract of sample()/_dummy_run (lmhead TP), guard the copied dispatch tail
with a canary that compares it call-by-call against upstream
GPUModelRunner.sample, and pin the o_proj TP graph-mode guard that turns an
eagerly dispatched step into an explicit error. Collective behavior itself
is validated on real hardware.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec, patch

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker

from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.worker.v2.spec_decode.lmhead_tp_utils import LmheadTPDraftSamplingMixin


def _make_runner(max_num_reqs=8, decode_query_len=2, vocab=6):
    """Bare instance bypassing __init__ (no NPU required).

    Dispatch components are autospecced against the real upstream classes so
    a call with drifted arguments fails loudly instead of being swallowed by
    a bare MagicMock.
    """
    runner = object.__new__(NPUModelRunner)
    runner.adaptive_verification = None
    runner.max_num_reqs = max_num_reqs
    runner.decode_query_len = decode_query_len
    runner.model = MagicMock()
    runner.model.compute_logits.side_effect = lambda x: torch.zeros(x.shape[0], vocab)
    runner.sampler = create_autospec(Sampler, instance=True)
    runner.rejection_sampler = create_autospec(RejectionSampler, instance=True)
    runner.speculator = MagicMock()
    # vLLM #50465 added batch-sharded sampling to GPUModelRunner.sample().
    # The production initializer always defines this field; mirror that
    # contract in this bare CPU-only fixture.
    runner.batch_sharder = None
    runner.structured_outputs_worker = create_autospec(StructuredOutputsWorker, instance=True)
    return runner


def _make_input_batch(logits_indices):
    return SimpleNamespace(
        logits_indices=logits_indices,
        num_draft_tokens=0,
        # vLLM #50465 lets a sharded rank own no requests and checks this
        # before dispatching to the sampler.
        num_reqs=1,
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


@pytest.mark.parametrize(
    "lmhead_enabled,is_profile,last_pp_rank,skip_eplb",
    [
        (True, False, True, False),
        (False, False, True, False),
        (True, True, True, False),
        (True, False, False, False),
        (True, False, True, True),
    ],
)
def test_idle_target_head_precedes_draft_and_eplb(lmhead_enabled, is_profile, last_pp_rank, skip_eplb):
    """Exercise the wrapper called by the parent before its draft propose."""
    runner = _make_runner(max_num_reqs=8, decode_query_len=2)
    runner.device = torch.device("cpu")
    runner.is_last_pp_rank = last_pp_rank
    runner.model_state = SimpleNamespace(kvpp_is_dummy_run=False)
    runner.kvpp = MagicMock()
    runner.ascend_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(profiling_chunk_config=None),
        xlite_graph_config=SimpleNamespace(enabled=False),
    )
    runner.eplb = MagicMock()
    events = []
    hidden_states = torch.randn(10, 6)

    def parent_execute(self, *args, **kwargs):
        events.append("forward")
        self.execute_model_state = SimpleNamespace(hidden_states=hidden_states)
        return "forward-output"

    def parent_dummy(self, *args, **kwargs):
        assert kwargs["skip_eplb"] is True
        self.execute_model(MagicMock(), dummy_run=True, is_profile=kwargs["is_profile"])
        events.append("draft")
        return hidden_states, hidden_states

    def compute_logits(inputs):
        events.append("target")
        return torch.zeros(inputs.shape[0], 6)

    runner.model.compute_logits.side_effect = compute_logits
    runner.eplb.step.side_effect = lambda **kwargs: events.append("eplb")
    with (
        patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_enable", return_value=lmhead_enabled),
        patch("vllm_ascend.worker.v2.model_runner.has_kv_transfer_group", return_value=False),
        patch("vllm_ascend.worker.v2.model_runner._start_profiling_chunk_timing", return_value=None),
        patch("vllm_ascend.worker.v2.model_runner._finish_profiling_chunk_timing", return_value=None),
        patch.object(GPUModelRunner, "execute_model", parent_execute),
        patch.object(GPUModelRunner, "_dummy_run", parent_dummy),
    ):
        runner._dummy_run(4, uniform_decode=True, is_profile=is_profile, skip_eplb=skip_eplb)

    expected = ["forward"]
    if lmhead_enabled and not is_profile and last_pp_rank:
        expected.append("target")
        inputs = runner.model.compute_logits.call_args.args[0]
        assert inputs.shape == (16, 6)
        torch.testing.assert_close(inputs, hidden_states[0].expand(16, 6))
        runner.model.compute_logits.assert_called_once()
    else:
        runner.model.compute_logits.assert_not_called()
    expected.append("draft")
    if not skip_eplb:
        expected.append("eplb")
    assert events == expected


def test_finegrained_tp_guard_contract():
    runner = object.__new__(NPUModelRunner)
    runner._finegrained_tp_requires_graph = False
    NPUModelRunner._check_finegrained_tp_graph_step(runner, CUDAGraphMode.NONE)
    runner._finegrained_tp_requires_graph = True
    with pytest.raises(RuntimeError, match="captured graph"):
        NPUModelRunner._check_finegrained_tp_graph_step(runner, CUDAGraphMode.NONE)
    NPUModelRunner._check_finegrained_tp_graph_step(runner, CUDAGraphMode.FULL_DECODE_ONLY)
    NPUModelRunner._check_finegrained_tp_graph_step(runner, CUDAGraphMode.FULL)


class _GreedyDraftSampler:
    def sample_draft(self, hidden_states, *args):
        return self.model.compute_logits(hidden_states).argmax(dim=-1)


class _DraftSampler(LmheadTPDraftSamplingMixin, _GreedyDraftSampler):
    pass


def _make_draft_sampler(max_num_reqs=8, num_speculative_steps=1):
    spec = _DraftSampler()
    spec.max_num_reqs = max_num_reqs
    spec.num_speculative_steps = num_speculative_steps
    spec.speculative_config = SimpleNamespace(draft_sample_method="greedy")
    spec.use_local_argmax_reduction = False
    spec.enable_adaptive_verification = False
    spec.model = MagicMock()
    spec.model.compute_logits.side_effect = lambda hs: torch.nn.functional.one_hot(
        torch.arange(hs.shape[0]) % 4, 4
    ).float()
    return spec


@pytest.mark.parametrize(
    "enabled,num_rows,num_steps",
    [(True, 1, 1), (True, 2, 1), (True, 3, 1), (True, 16, 1), (True, 0, 1), (True, 3, 3), (False, 3, 1)],
)
def test_draft_head_aligns_rank_rows_and_trims_tokens(enabled, num_rows, num_steps):
    spec = _make_draft_sampler(num_speculative_steps=num_steps)
    hidden_states = torch.randn(num_rows, 5)
    with patch("vllm_ascend.worker.v2.spec_decode.lmhead_tp_utils.lmhead_tp_enable", return_value=enabled):
        tokens = spec.sample_draft(hidden_states, None, None, None, None, None, None)
    inputs = spec.model.compute_logits.call_args.args[0]
    capacity = 8 * (num_steps + 1)
    assert inputs.shape == (capacity if enabled else num_rows, 5)
    torch.testing.assert_close(inputs[:num_rows], hidden_states)
    if enabled:
        assert torch.all(inputs[num_rows:] == 0)
    assert tokens.shape == (num_rows,)
    torch.testing.assert_close(tokens, torch.arange(num_rows) % 4)


def test_draft_head_rejects_overflow_before_collective():
    spec = _make_draft_sampler(max_num_reqs=4)
    with (
        patch("vllm_ascend.worker.v2.spec_decode.lmhead_tp_utils.lmhead_tp_enable", return_value=True),
        pytest.raises(ValueError, match="exceed capacity"),
    ):
        spec.sample_draft(torch.randn(9, 5), None, None, None, None, None, None)
    spec.model.compute_logits.assert_not_called()


@pytest.mark.parametrize(
    "option,match",
    [
        ("greedy", None),
        ("probabilistic", "probabilistic"),
        ("use_local_argmax_reduction", "use_local_argmax_reduction"),
        ("enable_adaptive_verification", "enable_adaptive_verification"),
    ],
)
def test_draft_sampling_rejects_unsupported_combinations(option, match):
    spec = _make_draft_sampler()
    if option == "probabilistic":
        spec.speculative_config.draft_sample_method = option
    elif option != "greedy":
        setattr(spec, option, True)
    with (
        patch("vllm_ascend.worker.v2.spec_decode.lmhead_tp_utils.lmhead_tp_enable", return_value=True),
        nullcontext() if match is None else pytest.raises(NotImplementedError, match=match),
    ):
        spec._lmhead_tp_validate_draft_sampling()


def test_mtp_uses_group_aligned_draft_sampling():
    from vllm_ascend.worker.v2.spec_decode.mtp.speculator import AscendMTPSpeculator

    assert issubclass(AscendMTPSpeculator, LmheadTPDraftSamplingMixin)
    assert AscendMTPSpeculator.sample_draft is LmheadTPDraftSamplingMixin.sample_draft
