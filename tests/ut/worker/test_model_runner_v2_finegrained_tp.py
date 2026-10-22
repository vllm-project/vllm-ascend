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
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import AscendAutoRegressiveSpeculator
from vllm_ascend.worker.v2.spec_decode.lmhead_tp_utils import LmheadTPDraftSamplingMixin


def _make_runner(max_num_reqs=8, decode_query_len=2, vocab=6):
    """Bare instance bypassing __init__ (no NPU required).

    Dispatch components are autospecced against the real upstream classes so
    a call with drifted arguments fails loudly instead of being swallowed by
    a bare MagicMock.
    """
    runner = object.__new__(NPUModelRunner)
    runner.input_buffers = SimpleNamespace(dummy_num_tokens=None)
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)
    runner.vllm_config = MagicMock()
    runner.adaptive_verification = None
    runner.max_num_reqs = max_num_reqs
    runner.decode_query_len = decode_query_len
    # Dump contract from the production initializer; helpers no-op on None.
    runner.debugger = None
    runner._debugger_started = False
    # Extra fields for the execute_model-path tests below.
    runner.device = torch.device("cpu")
    runner.is_last_pp_rank = True
    runner.execute_model_state = None
    runner.pcp_manager = None
    runner.model_state = SimpleNamespace(kvpp_is_dummy_run=False)
    runner.kvpp = MagicMock()
    runner.ascend_config = SimpleNamespace(scheduler_config=SimpleNamespace(profiling_chunk_config=None))
    runner.model = MagicMock()
    runner.model.compute_logits.side_effect = lambda x: torch.zeros(x.shape[0], vocab)
    # The head carries the dynamic capacity captured by the lm-head forward;
    # None means "fall back to the static bound".
    runner.model.lm_head._lmhead_tp_dynamic_capacity = None
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
    capacity = runner.max_num_reqs * runner.decode_query_len
    num_logits = indices.shape[0]
    hidden_dim = 4
    hidden_states = torch.randn(10, hidden_dim)
    input_batch = _make_input_batch(indices)

    with patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_enable", return_value=True):
        result = runner.sample(hidden_states, input_batch, None)

    compute_input = runner.model.compute_logits.call_args.args[0]
    # compute_logits sees the group-agreed capacity, not the real row count
    assert compute_input.shape == (capacity, hidden_dim)
    # real rows are the indexed hidden states; padding rows are row-0 gathers
    # (the V1-style index pad pads the index copy with zeros) and are trimmed off
    torch.testing.assert_close(compute_input[:num_logits], hidden_states[indices])
    torch.testing.assert_close(compute_input[num_logits:], hidden_states[0].expand(capacity - num_logits, hidden_dim))
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


@pytest.mark.parametrize(
    "lmhead_enabled,is_profile,skip_eplb",
    [
        (True, False, False),
        (False, False, False),
        (True, True, False),
        (True, False, True),
    ],
)
def test_dummy_lmhead_collective_precedes_eplb(lmhead_enabled, is_profile, skip_eplb):
    """An idle rank must join LM-head TP before its EPLB step can block it:
    forward, LM-head, EPLB. The join lives in the ``execute_model`` wrapper
    (ahead of the parent ``_dummy_run``'s dummy propose), so the parent mock
    routes through it."""
    runner = _make_runner(max_num_reqs=8, decode_query_len=2)  # capacity 16
    hidden_states = torch.randn(10, 6)
    runner.eplb = MagicMock()
    events = []
    runner.eplb.step.side_effect = lambda **kwargs: events.append("eplb")

    def compute_logits(inputs):
        events.append("lmhead")
        return torch.zeros(inputs.shape[0], 6)

    runner.model.compute_logits.side_effect = compute_logits

    def parent_dummy_run(self, num_tokens, *args, **kwargs):
        assert kwargs["skip_eplb"] is True
        events.append("forward")
        # The real parent runs the forward through execute_model first.
        _run_execute_model(self, hidden_states, dummy_run=True, is_profile=kwargs["is_profile"])
        return hidden_states, hidden_states

    with (
        patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_enable", return_value=lmhead_enabled),
        patch.object(GPUModelRunner, "_dummy_run", parent_dummy_run),
    ):
        runner._dummy_run(4, uniform_decode=True, is_profile=is_profile, skip_eplb=skip_eplb)

    # _dummy_run itself never joins: only forward (parent) and eplb (decorator).
    expected_events = ["forward"]
    if lmhead_enabled and not is_profile:
        expected_events.append("lmhead")
        # one join at the group-agreed capacity with zero-indexed rows
        dummy_input = runner.model.compute_logits.call_args.args[0]
        assert dummy_input.shape == (16, 6)
        torch.testing.assert_close(dummy_input, hidden_states[torch.zeros(16, dtype=torch.long)])
    if not skip_eplb:
        expected_events.append("eplb")
        runner.eplb.step.assert_called_once_with(is_dummy=True, is_profile=is_profile)
    else:
        runner.eplb.step.assert_not_called()
    assert events == expected_events


def _run_execute_model(runner, hidden_states, dummy_run=True, is_profile=False):
    """Drive execute_model with the parent mocked out; the stub publishes
    execute_model_state (as the real forward does) for the hook to read."""

    def super_execute(scheduler_output, **kwargs):
        runner.execute_model_state = SimpleNamespace(hidden_states=hidden_states)
        return "upstream-output"

    with (
        patch("vllm_ascend.worker.v2.model_runner._start_profiling_chunk_timing", return_value=None),
        patch("vllm_ascend.worker.v2.model_runner._finish_profiling_chunk_timing", return_value=None),
        patch("vllm_ascend.worker.v2.model_runner.should_skip_allreduce_across_dp_group", return_value=False),
        patch.object(GPUModelRunner, "execute_model", side_effect=super_execute),
    ):
        return runner.execute_model(MagicMock(), dummy_run=dummy_run, is_profile=is_profile)


def test_execute_model_dummy_joins_target_lmhead_collectives():
    """Idle DP ranks join the target LM-head collectives at the end of a dummy
    ``execute_model`` (the parent's dummy forward already ran the draft
    collectives once, so only the target side is joined here). With no dynamic
    value cached by the parent forward, the join falls back to the static
    group-agreed bound."""
    runner = _make_runner(max_num_reqs=8, decode_query_len=2)  # static cap 16
    hidden_states = torch.randn(20, 6)
    runner.execute_model_state = SimpleNamespace(hidden_states=hidden_states)
    runner.kvpp = MagicMock()
    runner.model_state = SimpleNamespace(kvpp_is_dummy_run=False)
    runner.ascend_config = SimpleNamespace(scheduler_config=SimpleNamespace(profiling_chunk_config=None))

    def super_execute(scheduler_output, intermediate_tensors=None, **kwargs):
        return "upstream-output"

    with (
        patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_enable", return_value=True),
        patch("vllm_ascend.worker.v2.model_runner._start_profiling_chunk_timing", return_value=None),
        patch("vllm_ascend.worker.v2.model_runner._finish_profiling_chunk_timing", return_value=None),
        patch("vllm_ascend.worker.v2.model_runner.should_skip_allreduce_across_dp_group", return_value=False),
        patch.object(NPUModelRunner.__bases__[0], "execute_model", side_effect=super_execute),
    ):
        output = runner.execute_model(MagicMock(), dummy_run=True)

    assert output == "upstream-output"
    assert runner.model.compute_logits.call_count == 1
    dummy_input = runner.model.compute_logits.call_args.args[0]
    # zero-indexed rows at the static group-agreed target capacity
    assert dummy_input.shape == (16, 6)
    torch.testing.assert_close(dummy_input, hidden_states[torch.zeros(16, dtype=torch.long)])


def test_execute_model_dummy_join_uses_dynamic_capacity():
    """When ``_publish_lmhead_tp_step_capacity`` captured the per-step capacity
    on the runner, the idle join must use it instead of the static bound so it
    stays aligned with the busy ranks' dynamic capacity."""
    runner = _make_runner(max_num_reqs=8, decode_query_len=2)  # static cap 16
    runner._lmhead_tp_step_capacity_value = 12  # DP-synced capacity this step
    hidden_states = torch.randn(20, 6)
    runner.execute_model_state = SimpleNamespace(hidden_states=hidden_states)
    runner.kvpp = MagicMock()
    runner.model_state = SimpleNamespace(kvpp_is_dummy_run=False)
    runner.ascend_config = SimpleNamespace(scheduler_config=SimpleNamespace(profiling_chunk_config=None))

    def super_execute(scheduler_output, intermediate_tensors=None, **kwargs):
        return "upstream-output"

    with (
        patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_enable", return_value=True),
        patch("vllm_ascend.worker.v2.model_runner._start_profiling_chunk_timing", return_value=None),
        patch("vllm_ascend.worker.v2.model_runner._finish_profiling_chunk_timing", return_value=None),
        patch("vllm_ascend.worker.v2.model_runner.should_skip_allreduce_across_dp_group", return_value=False),
        patch.object(NPUModelRunner.__bases__[0], "execute_model", side_effect=super_execute),
    ):
        output = runner.execute_model(MagicMock(), dummy_run=True)

    assert output == "upstream-output"
    assert runner.model.compute_logits.call_count == 1
    dummy_input = runner.model.compute_logits.call_args.args[0]
    assert dummy_input.shape == (12, 6)
    torch.testing.assert_close(dummy_input, hidden_states[torch.zeros(12, dtype=torch.long)])


def test_execute_model_dummy_skips_join_when_gated_off():
    """Real runs, profiling runs, feature-off runs, and non-last PP ranks must
    not add the dummy target join."""
    runner = _make_runner()
    hidden_states = torch.randn(20, 6)
    runner.execute_model_state = SimpleNamespace(hidden_states=hidden_states)
    runner.kvpp = MagicMock()
    runner.model_state = SimpleNamespace(kvpp_is_dummy_run=False)
    runner.ascend_config = SimpleNamespace(scheduler_config=SimpleNamespace(profiling_chunk_config=None))

    def super_execute(scheduler_output, intermediate_tensors=None, **kwargs):
        return "upstream-output"

    def _run(dummy_run, is_profile=False, lmhead=True, last_pp=True):
        runner.is_last_pp_rank = last_pp
        with (
            patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_enable", return_value=lmhead),
            patch("vllm_ascend.worker.v2.model_runner._start_profiling_chunk_timing", return_value=None),
            patch("vllm_ascend.worker.v2.model_runner._finish_profiling_chunk_timing", return_value=None),
            patch("vllm_ascend.worker.v2.model_runner.should_skip_allreduce_across_dp_group", return_value=False),
            patch.object(NPUModelRunner.__bases__[0], "execute_model", side_effect=super_execute),
        ):
            runner.execute_model(MagicMock(), dummy_run=dummy_run, is_profile=is_profile)
        runner.model.compute_logits.assert_not_called()
        runner.model.compute_logits.reset_mock()

    _run(dummy_run=False)  # real run
    _run(dummy_run=True, is_profile=True)  # profile run
    _run(dummy_run=True, lmhead=False)  # feature off
    _run(dummy_run=True, last_pp=False)  # non-last PP


def test_finegrained_tp_guard_contract():
    runner = object.__new__(NPUModelRunner)
    runner._finegrained_tp_requires_graph = False
    NPUModelRunner._check_finegrained_tp_graph_step(runner, CUDAGraphMode.NONE)
    runner._finegrained_tp_requires_graph = True
    with pytest.raises(RuntimeError, match="captured graph"):
        NPUModelRunner._check_finegrained_tp_graph_step(runner, CUDAGraphMode.NONE)
    NPUModelRunner._check_finegrained_tp_graph_step(runner, CUDAGraphMode.FULL_DECODE_ONLY)
    NPUModelRunner._check_finegrained_tp_graph_step(runner, CUDAGraphMode.FULL)


class _ConcreteSpeculator(AscendAutoRegressiveSpeculator):
    # Concrete stub: the base class keeps load_draft_model abstract.

    def load_draft_model(self, *args, **kwargs):
        raise NotImplementedError


def _make_speculator(max_num_reqs=8, num_speculative_steps=1):
    """Bare speculator bypassing __init__ (no NPU)."""
    spec = object.__new__(_ConcreteSpeculator)
    spec.max_num_reqs = max_num_reqs
    spec.num_speculative_steps = num_speculative_steps
    spec.replicated_pcp = False
    spec.model = MagicMock()
    spec.use_local_argmax_reduction = False
    spec.enable_adaptive_verification = False  # parent __init__ fields
    spec.acceptance_estimator = None
    return spec


def _spec_lmhead(enabled):
    return patch("vllm_ascend.worker.v2.spec_decode.lmhead_tp_utils.lmhead_tp_enable", return_value=enabled)


@pytest.mark.parametrize(
    "draft_method, bypass, argmax, adaptive, match",
    [
        ("probabilistic", False, False, False, "probabilistic"),
        ("greedy", False, False, False, None),  # default stays supported
        ("greedy", True, False, False, "sample_draft"),  # a hypothetical family bypassing sample_draft
        ("greedy", False, True, False, "use_local_argmax_reduction"),
        ("greedy", False, False, True, "enable_adaptive_verification"),
    ],
)
def test_speculator_init_validates(draft_method, bypass, argmax, adaptive, match):
    """Unsupported combinations must fail at construction, not at the first
    sampling step in a running engine (probabilistic gumbel buffers, paths
    that bypass sample_draft, local argmax under pure DP, adaptive
    verification's row-count-indexed estimator buffers)."""
    spec = _make_speculator()
    spec.speculative_config = SimpleNamespace(draft_sample_method=draft_method)
    spec._lmhead_tp_sample_draft_supported = not bypass
    spec.use_local_argmax_reduction = argmax
    spec.enable_adaptive_verification = adaptive

    with (
        nullcontext() if match is None else pytest.raises(NotImplementedError, match=match),
        _spec_lmhead(True),
    ):
        spec._lmhead_tp_validate_draft_sampling()


def test_production_speculators_carry_lmhead_sampling_mixin():
    """Every family must keep the mixin (a re-parent silently drops the row
    alignment and hangs the collectives); no current family opts out -- the
    flag stays as the guard for a future family whose draft collectives
    cannot be fed group-uniform rows (DFlash2's candidate merge gathers over
    the lmhead-TP group on uniformly padded draft-batch rows)."""
    from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
    from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import AscendDFlash2Speculator
    from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator
    from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator
    from vllm_ascend.worker.v2.spec_decode.mtp.speculator import AscendMTPSpeculator

    for cls in (
        AscendEagleSpeculator,
        AscendMTPSpeculator,
        AscendDFlashSpeculator,
        AscendDSparkSpeculator,
        AscendDFlash2Speculator,
    ):
        assert issubclass(cls, LmheadTPDraftSamplingMixin)
    assert AscendDSparkSpeculator._lmhead_tp_sample_draft_supported is True
    assert AscendDFlash2Speculator._lmhead_tp_sample_draft_supported is True


@patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_enable", return_value=True)
def test_publish_lmhead_tp_step_capacity(_lmhead_tp_enable):
    """``_publish_lmhead_tp_step_capacity`` must derive the per-step target
    capacity from ``execute_model_state.dp_sync.num_reqs`` (the group-agreed
    request count) times ``decode_query_len``, and refresh it every step."""
    runner = _make_runner(max_num_reqs=8, decode_query_len=2)  # static cap 16
    runner._lmhead_tp_step_capacity_value = 99  # stale from a previous step

    # Fresh DP sync: num_reqs=3 -> capacity 3 * 2 = 6 (replaces stale 99).
    runner.execute_model_state = SimpleNamespace(
        dp_sync=SimpleNamespace(num_reqs=3),
    )
    runner._publish_lmhead_tp_step_capacity()
    assert runner._lmhead_tp_step_capacity() == 6

    # A next step with a different req count must refresh, not keep 6.
    runner.execute_model_state = SimpleNamespace(
        dp_sync=SimpleNamespace(num_reqs=2),
    )
    runner._publish_lmhead_tp_step_capacity()
    assert runner._lmhead_tp_step_capacity() == 4

    # No DP sync this step: keep the previously published value (4), do not
    # overwrite it or fall back to static.
    runner.execute_model_state = SimpleNamespace(dp_sync=None)
    runner._publish_lmhead_tp_step_capacity()
    assert runner._lmhead_tp_step_capacity() == 4

    # No DP sync and nothing ever published -> static bound.
    runner2 = _make_runner(max_num_reqs=8, decode_query_len=2)  # static cap 16
    runner2.execute_model_state = SimpleNamespace(dp_sync=None)
    runner2._publish_lmhead_tp_step_capacity()
    assert runner2._lmhead_tp_step_capacity() == 16


def test_sample_uses_step_capacity():
    """``sample()`` must use the runner's per-step capacity (dynamic when the
    DP sync published one, static otherwise) so the padded rows match the
    lm-head forward."""
    runner = _make_runner(max_num_reqs=8, decode_query_len=2)  # static cap 16
    hidden_states = torch.randn(10, 4)
    indices = torch.tensor([0, 3, 5])
    input_batch = _make_input_batch(indices)
    runner._lmhead_tp_step_capacity_value = 12

    with patch("vllm_ascend.worker.v2.model_runner.lmhead_tp_enable", return_value=True):
        runner.sample(hidden_states, input_batch, None)

    compute_input = runner.model.compute_logits.call_args.args[0]
    assert compute_input.shape == (12, 4)
    torch.testing.assert_close(compute_input[:3], hidden_states[indices])
    # The index pad gathers row 0 for the padding rows (V1-style contract).
    torch.testing.assert_close(compute_input[3:], hidden_states[0].expand(12 - 3, 4))


def _patch_super_method(spec_cls, method_name):
    """Patch the nearest MRO ancestor that defines ``method_name``.

    Families resolve ``super()`` differently (EAGLE to the Ascend
    autoregressive base, DSpark/DFlash to their upstream parents), and
    ``__bases__[0]`` can be the sampling mixin, which never defines these
    methods — patching it would raise AttributeError.
    """
    for cls in spec_cls.__mro__[1:]:
        if method_name in cls.__dict__:
            return patch.object(cls, method_name)
    raise AssertionError(f"no ancestor of {spec_cls.__name__} defines {method_name}")


@patch("vllm_ascend.worker.v2.spec_decode.dspark.speculator.lmhead_tp_enable", return_value=True)
def test_dspark_run_model_publishes_draft_capacity(_lmhead_tp_enable):
    """``AscendDSparkSpeculator._run_model`` must publish the draft-head
    capacity derived from the DP-synced query rows: query rows are
    ``max_reqs_across_dp * num_query_per_req``, and the draft head emits
    ``num_speculative_steps`` rows per request."""
    from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator

    spec = object.__new__(AscendDSparkSpeculator)
    spec.num_query_per_req = 4  # sample_from_anchor=False layout
    spec.num_speculative_steps = 4
    spec.replicated_pcp = False
    spec.model = MagicMock()
    spec.model.lm_head = MagicMock()

    def fake_run_model(num_tokens, attn_metadata, slot_mappings, num_tokens_across_dp, cudagraph_runtime_mode):
        return "hidden"

    with _patch_super_method(AscendDSparkSpeculator, "_run_model") as super_run_model:
        super_run_model.side_effect = fake_run_model
        # query rows = 3 reqs * 4 = 12; draft capacity = 3 * 4 = 12
        spec._run_model(12, None, None, torch.tensor([12, 8], dtype=torch.int32))
    assert spec.model.lm_head._lmhead_tp_dynamic_capacity == 12

    # No sync value this round -> the previous round's dynamic capacity is
    # dropped so the static lmhead_tp_capacity stays authoritative.
    with _patch_super_method(AscendDSparkSpeculator, "_run_model") as super_run_model:
        super_run_model.side_effect = fake_run_model
        spec._run_model(4, None, None, None)
    assert not hasattr(spec.model.lm_head, "_lmhead_tp_dynamic_capacity")

    # Non-divisible query rows fail fast.
    with (
        _patch_super_method(AscendDSparkSpeculator, "_run_model") as super_run_model,
        pytest.raises(ValueError, match="num_query_per_req"),
    ):
        super_run_model.side_effect = fake_run_model
        spec._run_model(5, None, None, torch.tensor([5], dtype=torch.int32))


def test_lmhead_tp_inject_static_capacity():
    """``_lmhead_tp_inject_static_capacity`` must attach the config-derived
    static bound (``max_num_reqs * (num_speculative_steps + 1)`` — the
    mixin-era coefficient) to a draft model's lm_head, gated on lmhead TP:
    generic draft families (EAGLE, DFlash) build their heads in shared model
    files that know nothing about the capacity kwarg."""
    spec = _make_speculator(max_num_reqs=8, num_speculative_steps=1)  # static 16
    model = SimpleNamespace(lm_head=SimpleNamespace())

    with _spec_lmhead(True):
        spec._lmhead_tp_inject_static_capacity(model)
    assert model.lm_head.lmhead_tp_capacity == 16

    # Feature off: nothing attached — the head keeps its passthrough behavior.
    model_off = SimpleNamespace(lm_head=SimpleNamespace())
    with _spec_lmhead(False):
        spec._lmhead_tp_inject_static_capacity(model_off)
    assert not hasattr(model_off.lm_head, "lmhead_tp_capacity")


def test_lmhead_tp_inject_static_capacity_requires_lm_head():
    """A draft model without an lm_head must fail loudly at load time, not
    silently run unaligned collectives."""
    spec = _make_speculator()
    with (
        _spec_lmhead(True),
        pytest.raises(ValueError, match="lm_head"),
    ):
        spec._lmhead_tp_inject_static_capacity(SimpleNamespace())


def test_lmhead_tp_publish_draft_capacity_gated_off():
    """The per-step publish is gated on lmhead TP: with the feature off it
    must not touch the head (and must not pay the DP-sync ``.item()``)."""
    spec = _make_speculator(max_num_reqs=8, num_speculative_steps=1)
    spec.model = SimpleNamespace(lm_head=SimpleNamespace())

    with _spec_lmhead(False):
        spec._lmhead_tp_publish_draft_capacity(torch.tensor([3, 2], dtype=torch.int32))
    assert not hasattr(spec.model.lm_head, "_lmhead_tp_dynamic_capacity")


def test_eagle_run_model_publishes_draft_capacity():
    """``AscendEagleSpeculator._run_model`` must publish the draft-head
    capacity derived from the DP sync: EAGLE drafts one query token per
    request per step and the head runs on exactly those rows, so the
    synced max query rows are the tightest group-uniform capacity."""
    from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator

    spec = object.__new__(AscendEagleSpeculator)
    spec.num_speculative_steps = 2
    spec.model = MagicMock()
    spec.model.lm_head = MagicMock()

    def fake_run_model(
        num_tokens, attn_metadata, slot_mappings, num_tokens_across_dp, cudagraph_runtime_mode, mm_inputs
    ):
        return ("hidden", "sample_hidden")

    with (
        _spec_lmhead(True),
        _patch_super_method(AscendEagleSpeculator, "_run_model") as super_run_model,
    ):
        super_run_model.side_effect = fake_run_model
        # one query token per req: the synced max query rows are the capacity
        spec._run_model(3, None, None, torch.tensor([3, 2], dtype=torch.int32))
    assert spec.model.lm_head._lmhead_tp_dynamic_capacity == 3

    # No sync value this round -> the previous dynamic capacity is dropped
    # so the static lmhead_tp_capacity stays authoritative.
    with (
        _spec_lmhead(True),
        _patch_super_method(AscendEagleSpeculator, "_run_model") as super_run_model,
    ):
        super_run_model.side_effect = fake_run_model
        spec._run_model(3, None, None, None)
    assert not hasattr(spec.model.lm_head, "_lmhead_tp_dynamic_capacity")


def test_eagle_load_draft_model_injects_static_capacity():
    """EAGLE's load_draft_model must hand the loaded draft model to the
    static-capacity injection (generic EAGLE model files never pass
    lmhead_tp_capacity themselves) and return it unchanged."""
    from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator

    spec = object.__new__(AscendEagleSpeculator)
    spec.max_num_reqs = 8
    spec.num_speculative_steps = 1
    draft = SimpleNamespace(lm_head=SimpleNamespace())

    with (
        _spec_lmhead(True),
        _patch_super_method(AscendEagleSpeculator, "load_draft_model") as super_load,
    ):
        super_load.return_value = draft
        loaded = spec.load_draft_model(MagicMock(), set())

    assert loaded is draft
    assert draft.lm_head.lmhead_tp_capacity == 16  # 8 * (1 + 1)


def test_dflash_run_model_publishes_draft_capacity():
    """``AscendDFlashSpeculator._run_model`` must publish the draft-head
    capacity derived from the DP sync: DFlash drafts ``num_query_per_req``
    query tokens per request and the head samples ``num_speculative_steps``
    of them, so the synced max query rows already cover the head rows and
    are the capacity."""
    from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator

    spec = object.__new__(AscendDFlashSpeculator)
    spec.num_query_per_req = 4
    spec.num_speculative_steps = 2
    spec.model = MagicMock()
    spec.model.lm_head = MagicMock()

    def fake_run_model(num_tokens, attn_metadata, slot_mappings, num_tokens_across_dp, cudagraph_runtime_mode):
        return "hidden"

    with (
        _spec_lmhead(True),
        _patch_super_method(AscendDFlashSpeculator, "_run_model") as super_run_model,
    ):
        super_run_model.side_effect = fake_run_model
        # capacity = the synced max query rows = 12
        spec._run_model(12, None, None, torch.tensor([12, 8], dtype=torch.int32))
    assert spec.model.lm_head._lmhead_tp_dynamic_capacity == 12

    # No sync value this round -> the previous dynamic capacity is dropped
    # so the static lmhead_tp_capacity stays authoritative.
    with (
        _spec_lmhead(True),
        _patch_super_method(AscendDFlashSpeculator, "_run_model") as super_run_model,
    ):
        super_run_model.side_effect = fake_run_model
        spec._run_model(4, None, None, None)
    assert not hasattr(spec.model.lm_head, "_lmhead_tp_dynamic_capacity")

    # Non-divisible query rows fail fast.
    with (
        _spec_lmhead(True),
        _patch_super_method(AscendDFlashSpeculator, "_run_model") as super_run_model,
        pytest.raises(ValueError, match="tokens_per_req"),
    ):
        super_run_model.side_effect = fake_run_model
        spec._run_model(5, None, None, torch.tensor([5], dtype=torch.int32))


def test_dflash_load_draft_model_injects_static_capacity(monkeypatch):
    """DFlash's load_draft_model must hand the loaded draft model to the
    static-capacity injection (generic DFlash draft models never pass
    lmhead_tp_capacity themselves) and return it unchanged."""
    import vllm_ascend.worker.v2.spec_decode.dflash.speculator as dflash_module
    from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator

    spec = object.__new__(AscendDFlashSpeculator)
    spec.max_num_reqs = 8
    spec.num_speculative_steps = 1
    spec.vllm_config = MagicMock()
    draft = SimpleNamespace(lm_head=SimpleNamespace())
    monkeypatch.setattr(dflash_module, "disable_profiling_chunk_for_draft", lambda _: nullcontext())

    with (
        _spec_lmhead(True),
        _patch_super_method(AscendDFlashSpeculator, "load_draft_model") as super_load,
    ):
        super_load.return_value = draft
        loaded = spec.load_draft_model(MagicMock(), set())

    assert loaded is draft
    assert draft.lm_head.lmhead_tp_capacity == 16  # 8 * (1 + 1)


def test_dflash_load_draft_model_decouples_shared_head(monkeypatch):
    """A draft that mounts the target's lm_head object (the DFlash/DFlash2
    drafter checkpoints ship no lm_head weights, so the loader shares the
    target's) must run on a decoupled copy: same weight tensors, separate
    object. The draft-side capacity lands on the copy; the target's head
    object stays clean (the target's sample() must never resolve the
    draft-scale dyn -- rows > dyn -> ValueError, the Q6 crash)."""
    import vllm_ascend.worker.v2.spec_decode.dflash.speculator as dflash_module
    from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator

    spec = object.__new__(AscendDFlashSpeculator)
    spec.max_num_reqs = 8
    spec.num_speculative_steps = 1
    spec.vllm_config = MagicMock()
    shared_head = SimpleNamespace(weight=object())
    draft = SimpleNamespace(lm_head=shared_head)
    # modules() mirrors nn.Module; the loader mounts the target's (possibly
    # nested, e.g. language_model.lm_head for VL targets) head onto the draft.
    target = SimpleNamespace(lm_head=shared_head, modules=lambda: [shared_head])
    monkeypatch.setattr(dflash_module, "disable_profiling_chunk_for_draft", lambda _: nullcontext())

    with (
        _spec_lmhead(True),
        _patch_super_method(AscendDFlashSpeculator, "load_draft_model") as super_load,
    ):
        super_load.return_value = draft
        loaded = spec.load_draft_model(target, set())

    assert loaded is draft
    assert spec._lmhead_tp_head_is_shared is True
    assert loaded.lm_head is not shared_head  # decoupled object
    assert loaded.lm_head.weight is shared_head.weight  # same weight tensors
    assert not hasattr(shared_head, "lmhead_tp_capacity")  # target head clean
    assert loaded.lm_head.lmhead_tp_capacity == 16  # capacity on the copy


def test_dflash_load_draft_model_detects_nested_shared_head(monkeypatch):
    """Regression (NPU crash): VL targets keep lm_head nested
    (language_model.lm_head) with no top-level attribute -- the detection
    must find the sharing via the module walk (not the wrapper attribute)
    and decouple the draft onto a copy so the capacity never lands on the
    target's nested head object."""
    import vllm_ascend.worker.v2.spec_decode.dflash.speculator as dflash_module
    from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator

    spec = object.__new__(AscendDFlashSpeculator)
    spec.max_num_reqs = 8
    spec.num_speculative_steps = 1
    spec.vllm_config = MagicMock()
    shared_head = SimpleNamespace(weight=object())
    draft = SimpleNamespace(lm_head=shared_head)
    # VL wrapper: no top-level lm_head; the head only appears in modules().
    target = SimpleNamespace(modules=lambda: [shared_head])
    monkeypatch.setattr(dflash_module, "disable_profiling_chunk_for_draft", lambda _: nullcontext())

    with (
        _spec_lmhead(True),
        _patch_super_method(AscendDFlashSpeculator, "load_draft_model") as super_load,
    ):
        super_load.return_value = draft
        loaded = spec.load_draft_model(target, set())

    assert loaded is draft
    assert spec._lmhead_tp_head_is_shared is True
    assert loaded.lm_head is not shared_head  # decoupled object
    assert not hasattr(shared_head, "lmhead_tp_capacity")  # target head clean
    assert loaded.lm_head.lmhead_tp_capacity == 16  # capacity on the copy


def test_dflash2_lmhead_tp_supported_at_construction():
    """DFlash2 passes the lmhead-TP construction validation under greedy
    drafting: its head call is collective-free and its candidate merge
    gathers over the lmhead-TP group on uniformly padded draft-batch rows."""
    from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import AscendDFlash2Speculator

    spec = object.__new__(AscendDFlash2Speculator)
    spec.speculative_config = SimpleNamespace(draft_sample_method="greedy")
    spec.use_local_argmax_reduction = False
    spec.enable_adaptive_verification = False

    with _spec_lmhead(True):
        spec._lmhead_tp_validate_draft_sampling()  # no raise


def test_dflash2_compute_candidates_merges_top_k_over_lmhead_group():
    """Under pure-DP lmhead TP the shared target head is vocab-sharded over
    the lmhead-TP group while the global TP world size is 1: the per-shard
    candidate top-k must gather over the lmhead-TP group, or every rank
    drafts from its own vocab shard only (silently wrong candidates)."""
    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

    import vllm_ascend.models.qwen3_dflash2 as dflash2_module
    from vllm_ascend.models.qwen3_dflash2 import DFlash2Qwen3ForCausalLM

    # Local shard logits [3 rows, 4 shard vocab]: local top-2 of row 0 is
    # values [5, 4] at shard-local ids [0, 1] -> global ids 100, 101.
    logits = torch.zeros(3, 4)
    logits[0, 0], logits[0, 1] = 5.0, 4.0
    logits[1, 2], logits[1, 3] = 7.0, 6.0
    logits[2, 0] = 1.0

    model = object.__new__(DFlash2Qwen3ForCausalLM)
    model.lm_head = SimpleNamespace(
        quant_method=MagicMock(spec=UnquantizedEmbeddingMethod),
        shard_indices=SimpleNamespace(num_org_vocab_padding=0, org_vocab_start_index=100),
    )
    model.lm_head.quant_method.apply.return_value = logits
    model.model = SimpleNamespace(candidate_selector=SimpleNamespace(top_k=2))
    model.output_multiplier = 1.0
    model.final_logit_softcapping = None

    hidden = torch.randn(3, 5)

    # The other lmhead-TP rank contributes values [6, 3] at global ids
    # [200, 201] per row: the merged global top-2 of row 0 must pick the
    # remote 6.0 (id 200) alongside the local 5.0 (id 100).
    remote_values = torch.tensor([[6.0, 3.0]] * 3)
    remote_ids = torch.tensor([[200, 201]] * 3)

    class _FakeLmheadGroup:
        world_size = 2

        def all_gather(self, x, dim):
            assert dim == -1
            remote = remote_values if x.dtype.is_floating_point else remote_ids
            return torch.cat([x, remote], dim=-1)

    with (
        patch.object(dflash2_module, "lmhead_tp_enable", return_value=True),
        patch.object(dflash2_module, "get_lmhead_tp_group", return_value=_FakeLmheadGroup()),
        patch.object(dflash2_module, "get_tp_group"),  # tripwire: must not be consulted
    ):
        ids, values = model.compute_candidates(hidden)

    assert ids.shape == (3, 2)
    # Row 0: global top-2 = remote 6.0 (id 200) + local 5.0 (id 100).
    assert ids[0].tolist() == [200, 100]
    assert values[0].tolist() == [6.0, 5.0]
    # Row 1: the local shard wins: 7.0 (id 102) + 6.0 (id 103).
    assert ids[1].tolist() == [102, 103]
    assert values[1].tolist() == [7.0, 6.0]

    # lmhead TP off + global TP of 1: no merge, local-shard candidates only.
    with (
        patch.object(dflash2_module, "lmhead_tp_enable", return_value=False),
        patch.object(dflash2_module, "get_tp_group", return_value=SimpleNamespace(world_size=1)),
    ):
        ids, _ = model.compute_candidates(hidden)
    assert ids[0].tolist() == [100, 101]


def test_publish_converts_stashed_request_count_on_bootstrap():
    """The autoregressive bootstrap reuses the target's verify sync, which
    carries (k+1) query rows per request, so the raw synced max overshoots
    the one-row-per-request draft head by the verify basis (the observed
    16-vs-4 bootstrap overshoot): with the round's request count stashed,
    the publish must convert the synced rows to the head basis."""
    spec = _make_speculator()
    spec.model = SimpleNamespace(lm_head=SimpleNamespace())
    spec._lmhead_tp_uniform_num_reqs = 4  # dp_sync.num_reqs of this round

    sync = torch.tensor([16, 16], dtype=torch.int32)
    with _spec_lmhead(True):
        spec._lmhead_tp_publish_draft_capacity(sync)
    assert spec.model.lm_head._lmhead_tp_dynamic_capacity == 4


def test_publish_stashed_request_count_keeps_continuation_unchanged():
    """The continuations rebuild the sync on a one-row basis, so the stashed
    conversion must not change the published capacity there: 4 requests ->
    4 synced rows -> capacity 4, byte-identical to the raw path (zero
    regression on the steady-state draft loop)."""
    spec = _make_speculator()
    spec.model = SimpleNamespace(lm_head=SimpleNamespace())
    spec._lmhead_tp_uniform_num_reqs = 4

    sync = torch.tensor([4, 4], dtype=torch.int32)
    with _spec_lmhead(True):
        spec._lmhead_tp_publish_draft_capacity(sync)
    assert spec.model.lm_head._lmhead_tp_dynamic_capacity == 4


def test_publish_stashed_request_count_tightens_true_prefill():
    """A real prompt prefill mixes irregular prompt lengths into the synced
    max, which wildly overshoots the head rows (one per request): the
    stashed request count must clamp the capacity to the request basis."""
    spec = _make_speculator()
    spec.model = SimpleNamespace(lm_head=SimpleNamespace())
    spec._lmhead_tp_uniform_num_reqs = 2

    sync = torch.tensor([1600, 1200], dtype=torch.int32)
    with _spec_lmhead(True):
        spec._lmhead_tp_publish_draft_capacity(sync)
    assert spec.model.lm_head._lmhead_tp_dynamic_capacity == 2


def test_publish_ignores_stale_stashed_request_count():
    """The draft syncs are request-scaled, so a valid stash can never exceed
    the synced rows; a stash above them is stale (e.g. a leftover from a
    larger round) and must be ignored, publishing the raw synced max
    exactly as before so a desynced capacity can never be published."""
    spec = _make_speculator()
    spec.model = SimpleNamespace(lm_head=SimpleNamespace())
    spec._lmhead_tp_uniform_num_reqs = 99  # stale: above the synced rows

    sync = torch.tensor([4, 4], dtype=torch.int32)
    with _spec_lmhead(True):
        spec._lmhead_tp_publish_draft_capacity(sync)
    assert spec.model.lm_head._lmhead_tp_dynamic_capacity == 4


def _patch_ar_context(name):
    """Neutralize an autoregressive-module context manager for
    bare-instance tests: the real ones need speculator state that an
    ``object.__new__`` fixture never has."""
    import vllm_ascend.worker.v2.spec_decode.autoregressive.speculator as ar_module

    return patch.object(ar_module, name, lambda *args: nullcontext())


def test_propose_stashes_dp_agreed_request_count():
    """The propose override must stash the round's DP-agreed request count
    (``dp_sync.num_reqs`` -- identical on every rank, padded to the bucket
    request count in FULL mode) behind the request-scaled draft syncs, and
    must leave the stash unset for replicated drafts (global-token syncs)
    and sync-less rounds, whose publish keeps the raw-sync behavior."""
    spec = _make_speculator()
    spec._lmhead_tp_uniform_num_reqs = None
    propose_args = ("batch",) + (None,) * 10  # through seeds
    super_patch = _patch_super_method(AscendAutoRegressiveSpeculator, "propose")

    def _run_propose(dp_sync):
        with (
            super_patch as super_propose,
            _patch_ar_context("disable_target_pcp_for_replicated_draft"),
            _patch_ar_context("build_attn_metadata_wrapper"),
            _patch_ar_context("torch_gather_wrapper"),
        ):
            super_propose.return_value = "proposals"
            return spec.propose(*propose_args, dp_sync)

    # Non-replicated with a DP sync: stash the agreed request count.
    assert _run_propose(SimpleNamespace(num_reqs=4)) == "proposals"
    assert spec.input_batch == "batch"
    assert spec._lmhead_tp_uniform_num_reqs == 4

    # Replicated drafts sync global tokens: no stash.
    spec.replicated_pcp = True
    _run_propose(SimpleNamespace(num_reqs=4))
    assert spec._lmhead_tp_uniform_num_reqs is None

    # No DP sync this round: no stash.
    spec.replicated_pcp = False
    _run_propose(None)
    assert spec._lmhead_tp_uniform_num_reqs is None


def test_capture_clears_stashed_request_count():
    """Capture invokes the phase entries directly (no propose round), so a
    stash left over from the last dummy run would be stale against the
    capture buckets; capture() must clear it so the in-capture capacity
    publish falls back to the raw synced max / static bound."""
    spec = _make_speculator(num_speculative_steps=1)  # early-return path
    spec.last_token_indices = torch.zeros(8, dtype=torch.int64)
    spec.prefill_cudagraph_manager = MagicMock()
    spec.prefill_cudagraph_manager.use_breakable_cg = False
    spec.model_state = MagicMock()
    spec.target_input_buffers = MagicMock()
    spec.block_tables = MagicMock()
    # draft_prefill_attn_groups is a read-only property resolving to
    # target_attn_groups when replicated_pcp is False.
    spec.target_attn_groups = MagicMock()
    spec.kv_cache_config = MagicMock()
    spec._lmhead_tp_uniform_num_reqs = 4  # stale from the last dummy run

    with (
        _patch_ar_context("disable_target_pcp_for_replicated_draft"),
        _patch_ar_context("build_attn_metadata_wrapper"),
    ):
        spec.capture()

    assert spec._lmhead_tp_uniform_num_reqs is None
    spec.prefill_cudagraph_manager.capture.assert_called_once()


def test_mtp_load_draft_model_decouples_shared_layer_heads():
    """Upstream load_eagle_model ties every MTP layer's shared_head.head
    (and a top-level lm_head alias) to the target's lm_head OBJECT when the
    MTP checkpoint ships no own head weights: the per-layer capacity
    publish in _run_model would then poison the target's head with
    draft-scale values (the Q6/DFlash crash), so load_draft_model must
    decouple every shared head onto a shallow copy with the static
    capacity re-attached, leaving the target's head object clean."""
    from vllm_ascend.worker.v2.spec_decode.mtp.speculator import AscendMTPSpeculator

    spec = object.__new__(AscendMTPSpeculator)
    spec.max_num_reqs = 8
    spec.num_speculative_steps = 1
    target_head = SimpleNamespace(weight=object())
    # Two MTP layers, both heads tied to the target's head object.
    layers = {str(i): SimpleNamespace(shared_head=SimpleNamespace(head=target_head)) for i in range(2)}
    draft = SimpleNamespace(
        model=SimpleNamespace(layers=layers),
        lm_head=target_head,  # the loader's top-level alias
    )
    target = SimpleNamespace(lm_head=target_head, modules=lambda: [target_head])

    import vllm_ascend.worker.v2.spec_decode.mtp.speculator as mtp_module

    with (
        # The mtp module imports lmhead_tp_enable into its own namespace.
        patch.object(mtp_module, "lmhead_tp_enable", return_value=True),
        _patch_super_method(AscendMTPSpeculator, "load_draft_model") as super_load,
    ):
        super_load.return_value = draft
        loaded = spec.load_draft_model(target, set())

    assert loaded is draft
    assert spec._lmhead_tp_head_is_shared is True
    for layer in layers.values():
        head = layer.shared_head.head
        assert head is not target_head  # decoupled object
        assert head.weight is target_head.weight  # same weight tensors
        assert head.lmhead_tp_capacity == 16  # static bound re-attached
    assert loaded.lm_head is not target_head  # top-level alias decoupled
    assert not hasattr(target_head, "lmhead_tp_capacity")  # target head clean


def test_mtp_load_draft_model_keeps_owned_layer_heads():
    """A draft that owns its layer head weights (some MTP families ship
    them) must keep its own head objects: nothing to decouple, and the
    constructor's static capacity stays exactly as loaded."""
    from vllm_ascend.worker.v2.spec_decode.mtp.speculator import AscendMTPSpeculator

    spec = object.__new__(AscendMTPSpeculator)
    spec.max_num_reqs = 8
    spec.num_speculative_steps = 1
    own_head = SimpleNamespace(weight=object(), lmhead_tp_capacity=24)
    layers = {"0": SimpleNamespace(shared_head=SimpleNamespace(head=own_head))}
    draft = SimpleNamespace(model=SimpleNamespace(layers=layers))
    target = SimpleNamespace(lm_head=SimpleNamespace(weight=object()), modules=lambda: [])

    import vllm_ascend.worker.v2.spec_decode.mtp.speculator as mtp_module

    with (
        # The mtp module imports lmhead_tp_enable into its own namespace.
        patch.object(mtp_module, "lmhead_tp_enable", return_value=True),
        _patch_super_method(AscendMTPSpeculator, "load_draft_model") as super_load,
    ):
        super_load.return_value = draft
        loaded = spec.load_draft_model(target, set())

    assert loaded is draft
    assert spec._lmhead_tp_head_is_shared is False
    assert layers["0"].shared_head.head is own_head  # untouched
    assert own_head.lmhead_tp_capacity == 24  # constructor value kept
