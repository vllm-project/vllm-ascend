"""Pure-mock UTs (CPU, no NPU) for lmhead TP in the Ascend V2 model runner.

Lock the runner-side pad/trim contract and dispatch-tail canary, the
draft-side sample_draft row alignment and draft runtime config build (both
V1 parity), and init-time rejection of unsupported combinations
(probabilistic, DSpark, local argmax, prompt_logprobs). LM-head collective
behavior itself is validated on real hardware.
"""

from contextlib import nullcontext
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
from vllm_ascend.worker.v2.spec_decode.lmhead_tp_utils import LmheadTPDraftSamplingMixin


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
    runner.pcp_manager = None
    runner.ascend_config = SimpleNamespace(scheduler_config=SimpleNamespace(profiling_chunk_config=None))
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
        (True, True),  # grammar bitmask + rejection sampler
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
        patch("vllm_ascend.worker.v2.model_runner._start_profiling_chunk_timing", return_value=None),
        patch("vllm_ascend.worker.v2.model_runner._finish_profiling_chunk_timing", return_value=None),
        patch.object(NPUModelRunner.__bases__[0], "execute_model", side_effect=super_execute),
    ):
        return runner.execute_model(MagicMock(), dummy_run=dummy_run, is_profile=is_profile)


def test_dummy_execute_model_joins_lmhead_collectives_at_capacity():
    """Idle DP ranks must join the LM-head collectives on every dummy run.

    Regression: with lmhead TP the LM-head all_gather spans the whole group,
    but the V2 dummy path only runs the model forward, so a real sample() on
    the rank owning requests waited forever. The hook must call
    compute_logits exactly once with zero-indexed rows at the same capacity
    the sample() override pads to. On the real dummy path the hook runs
    inside execute_model, ahead of the parent _dummy_run's speculator dummy
    propose (busy-rank ordering: target head first, then draft head); the
    ordering itself is not driven here.
    """
    runner = _make_runner(max_num_reqs=8, decode_query_len=2)  # capacity 16
    hidden_states = torch.randn(10, 6)

    with patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_configured", return_value=True):
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
    with patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_configured", return_value=True):
        _run_execute_model(runner, hidden_states, dummy_run=False)
    runner.model.compute_logits.assert_not_called()

    with patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_configured", return_value=False):
        _run_execute_model(runner, hidden_states, dummy_run=True)
    runner.model.compute_logits.assert_not_called()

    with patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_configured", return_value=True):
        _run_execute_model(runner, hidden_states, dummy_run=True, is_profile=True)
    runner.model.compute_logits.assert_not_called()

    runner.is_last_pp_rank = False
    with patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_configured", return_value=True):
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


@pytest.mark.parametrize("max_num_reqs, num_rows", [(8, 3), (4, 8)])
def test_sample_draft_pads_to_capacity_then_trims(max_num_reqs, num_rows):
    """With lmhead TP every draft sampling call feeds the draft LM head the
    group-agreed capacity rows (busy real propose and idle dummy propose
    alike) and hands the caller back only the real rows. At capacity the
    hidden states pass through untouched."""
    spec = _make_speculator(max_num_reqs=max_num_reqs, num_speculative_steps=1)
    spec.model.compute_logits.side_effect = _one_hot_argmax_logits
    hidden_states = torch.randn(num_rows, 5)
    capacity = spec._lmhead_tp_max_num_logits()

    with patch(
        "vllm_ascend.worker.v2.spec_decode.lmhead_tp_utils.lmhead_tp_configured",
        return_value=True,
    ):
        draft_tokens = spec.sample_draft(
            hidden_states, MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), None
        )

    compute_input = spec.model.compute_logits.call_args.args[0]
    if num_rows == capacity:
        assert compute_input is hidden_states  # untouched, no copy
    else:
        assert compute_input.shape == (capacity, 5)  # padded to capacity
        torch.testing.assert_close(compute_input[:num_rows], hidden_states)
        assert torch.all(compute_input[num_rows:] == 0)  # zero padding rows
    assert draft_tokens.shape[0] == num_rows  # trimmed back to real rows
    torch.testing.assert_close(draft_tokens, (torch.arange(num_rows) % 4).to(draft_tokens.dtype))


def test_sample_draft_passthrough_when_lmhead_disabled():
    spec = _make_speculator()
    spec.model.compute_logits.side_effect = _one_hot_argmax_logits
    hidden_states = torch.randn(3, 5)

    with patch(
        "vllm_ascend.worker.v2.spec_decode.lmhead_tp_utils.lmhead_tp_configured",
        return_value=False,
    ):
        draft_tokens = spec.sample_draft(
            hidden_states, MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), None
        )

    spec.model.compute_logits.assert_called_once_with(hidden_states)
    assert draft_tokens.shape[0] == 3


def test_sample_draft_raises_when_rows_exceed_capacity():
    spec = _make_speculator(max_num_reqs=4, num_speculative_steps=1)  # capacity 8
    hidden_states = torch.randn(9, 5)

    with (
        patch(
            "vllm_ascend.worker.v2.spec_decode.lmhead_tp_utils.lmhead_tp_configured",
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
            "vllm_ascend.worker.v2.spec_decode.lmhead_tp_utils.lmhead_tp_configured",
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


@pytest.mark.parametrize(
    "draft_method, bypass_sample_draft, local_argmax, match",
    [
        ("probabilistic", False, False, "probabilistic"),
        # greedy is the default draft sampling method and stays supported
        ("greedy", False, False, None),
        # DSpark-style speculators (flag off) bypass sample_draft entirely
        ("greedy", True, False, "sample_draft"),
        ("greedy", False, True, "use_local_argmax_reduction"),
    ],
)
def test_speculator_init_validates_unsupported_draft_sampling(draft_method, bypass_sample_draft, local_argmax, match):
    """Unsupported combinations must fail at construction time, not at the
    first sampling step inside a running engine: probabilistic (gumbel writes
    fixed-size buffers), DSpark-style paths that bypass sample_draft (the row
    alignment never runs), and local argmax (reduces over the local vocab
    shard only, silently wrong under pure DP). Greedy stays allowed."""
    spec = _make_speculator()
    spec.speculative_config = SimpleNamespace(draft_sample_method=draft_method)
    spec._lmhead_tp_sample_draft_supported = not bypass_sample_draft
    spec.use_local_argmax_reduction = local_argmax

    with (
        nullcontext() if match is None else pytest.raises(NotImplementedError, match=match),
        patch(
            "vllm_ascend.worker.v2.spec_decode.lmhead_tp_utils.lmhead_tp_configured",
            return_value=True,
        ),
    ):
        spec._lmhead_tp_validate_draft_sampling()


def test_speculator_validation_without_engine_config_is_noop():
    """Construction-time validation also runs inside lightweight harnesses
    (bare speculators with the parent __init__ mocked); with no engine
    config the feature is not configured on, so validation must no-op
    instead of failing on the config read."""
    spec = _make_speculator()
    spec.speculative_config = SimpleNamespace(draft_sample_method="probabilistic")

    # No config initialization, no lmhead patch: the harshest combination
    # must not raise, proving the no-op comes from the tolerant config read.
    spec._lmhead_tp_validate_draft_sampling()


def test_production_speculators_carry_lmhead_sampling_mixin():
    """Every production speculator family must keep the mixin (directly or
    via AscendAutoRegressiveSpeculator): losing it to a re-parent during an
    upstream spec refactor silently drops the draft-side row alignment and
    hangs the lmhead-TP collectives. DSpark must keep its opt-out flag."""
    from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
    from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator
    from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator
    from vllm_ascend.worker.v2.spec_decode.mtp.speculator import AscendMTPSpeculator

    for cls in (AscendEagleSpeculator, AscendMTPSpeculator, AscendDFlashSpeculator, AscendDSparkSpeculator):
        assert issubclass(cls, LmheadTPDraftSamplingMixin)
    assert AscendDSparkSpeculator._lmhead_tp_sample_draft_supported is False


@pytest.mark.parametrize("any_prompt_logprobs", [True, False])
def test_sample_tokens_prompt_logprobs_with_lmhead(any_prompt_logprobs):
    """The prompt-logprobs worker issues a second unpadded compute_logits that
    desyncs the LM-head collectives; fail at the first affected step instead
    of hanging. Without lmhead TP the guard passes through."""
    runner = _make_runner()
    runner.use_spec_pp = False
    uses_prompt_logprobs = np.array([False, True, False]) if any_prompt_logprobs else np.zeros(8, dtype=bool)
    runner.prompt_logprobs_worker = SimpleNamespace(uses_prompt_logprobs=uses_prompt_logprobs)

    with (
        patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_configured", return_value=True),
        patch.object(
            NPUModelRunner.__bases__[0], "sample_tokens", return_value="upstream-result"
        ) as super_sample_tokens,
    ):
        if any_prompt_logprobs:
            with pytest.raises(NotImplementedError, match="prompt_logprobs"):
                runner.sample_tokens(None)
        else:
            result = runner.sample_tokens(None)

    if any_prompt_logprobs:
        super_sample_tokens.assert_not_called()
    else:
        assert result == "upstream-result"
        super_sample_tokens.assert_called_once_with(None)


def test_sample_tokens_bare_runner_passthrough_without_engine_config():
    """sample_tokens also runs on bare runners in lightweight harnesses that
    never call init_ascend_config and lack lmhead-only attributes; the guard
    must read the config tolerantly and pass through."""
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.pcp_manager = None
    runner.is_last_pp_rank = True
    runner.speculator = None
    runner.use_spec_pp = False
    runner.execute_model_state = None

    with patch.object(
        NPUModelRunner.__bases__[0], "sample_tokens", return_value="upstream-result"
    ) as super_sample_tokens:
        result = runner.sample_tokens(None)

    assert result == "upstream-result"
    super_sample_tokens.assert_called_once_with(None)


def test_draft_capacity_formula_matches_runner():
    """Lock the per-request row convention each side feeds the shared
    helper: the runner passes ``decode_query_len``, the speculator
    ``num_speculative_steps + 1``. The draft and target collectives are
    independent, so equality is convention, not requirement — but any
    change to either convention must surface here for the docstrings to
    catch up."""
    spec = _make_speculator(max_num_reqs=8, num_speculative_steps=3)
    runner = _make_runner(max_num_reqs=8, decode_query_len=4)
    assert spec._lmhead_tp_max_num_logits() == 32
    assert runner._lmhead_tp_max_num_logits() == 32


def test_draft_vllm_config_does_not_revalidate_draft_model_config():
    """The draft runtime config must be built the way V1 builds it.

    EAGLE/DFlash draft heads are dense even when the target is MoE. Rebuilding
    the draft config with ``replace(..., model_config=draft_model_config)``
    re-runs VllmConfig validation, which checks the draft head against the
    target-side fine-grained TP layout that ``ascend_config`` allows only for
    MoE models, so construction dies before any request can run. The draft
    model config still has to reach the draft config for the draft graph.
    """
    from vllm_ascend.worker.v2.spec_decode.autoregressive import speculator as autoreg_module

    spec = object.__new__(_ConcreteSpeculator)
    spec.vllm_config = MagicMock(name="target_vllm_config")
    spec.draft_model_config = MagicMock(name="draft_model_config")

    calls = []

    def _fake_replace(config, **kwargs):
        calls.append((config, kwargs))
        return SimpleNamespace(**kwargs)

    with patch.object(autoreg_module, "replace", side_effect=_fake_replace):
        draft_vllm_config = spec._create_draft_vllm_config()

    assert calls[0][0] is spec.vllm_config.parallel_config
    assert calls[0][1] == {"pipeline_parallel_size": 1}
    assert len(calls) == 2
    # Only the target-derived config is validated; the draft model config is
    # swapped in afterwards.
    assert "model_config" not in calls[1][1]
    assert draft_vllm_config.model_config is spec.draft_model_config


def test_eagle_draft_vllm_config_disables_expert_parallel():
    """The EAGLE draft keeps the target-derived config but must not run as an
    expert model: the dense drafter has no experts, so EP/EPLB stay off."""
    from vllm_ascend.worker.v2.spec_decode.autoregressive import speculator as autoreg_module
    from vllm_ascend.worker.v2.spec_decode.eagle import speculator as eagle_module
    from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator

    spec = object.__new__(AscendEagleSpeculator)
    spec.vllm_config = MagicMock(name="target_vllm_config")
    spec.draft_model_config = MagicMock(name="draft_model_config")

    calls = []

    def _fake_replace(config, **kwargs):
        calls.append((config, kwargs))
        return SimpleNamespace(**kwargs)

    with (
        patch.object(autoreg_module, "replace", side_effect=_fake_replace),
        patch.object(eagle_module, "replace", side_effect=_fake_replace),
    ):
        draft_vllm_config = spec._create_draft_vllm_config()

    assert calls[0][1] == {"pipeline_parallel_size": 1}
    assert calls[-1][1] == {"enable_expert_parallel": False, "enable_eplb": False}
    assert "model_config" not in calls[1][1]
    assert draft_vllm_config.model_config is spec.draft_model_config
