"""Unit tests for fine-grained TP support in the Ascend V2 model runner.

Pure-mock tests (CPU tensors, no NPU): they lock the runner-side pad/trim
contract of sample()/_dummy_run and guard the copied dispatch tail with a
canary that compares it call-by-call against upstream GPUModelRunner.sample,
plus the eager DP padding contract of o_proj TP. Collective behavior of the
LM head and the OTP exchange itself is validated on real hardware.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec, patch

import numpy as np
import pytest
import torch
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.model_runner import BatchReqState, GPUModelRunner
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker

from vllm_ascend.worker.v2.eager_dp_padding import make_dp_padded_dummy_output, sync_dp_group_max_tokens
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


def _make_dp_padding_runner(enabled=True, aligned_tokens=0):
    """Bare runner carrying only the eager-DP-padding state."""
    runner = object.__new__(NPUModelRunner)
    runner._dp_padding_enabled = enabled
    runner._dp_padding_aligned_tokens = aligned_tokens
    runner._dp_padding_original_tokens = 0
    runner.dp_size = 8
    runner.dp_rank = 3
    runner.decode_query_len = 2
    runner.max_num_reqs = 8
    # need_timing=False keeps both profiling helpers pure no-ops (no NPU sync).
    runner.ascend_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(profiling_chunk_config=SimpleNamespace(need_timing=False))
    )
    return runner


def _make_scheduler_output(total, num_scheduled_tokens=None):
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=MagicMock(),
        num_scheduled_tokens=dict(num_scheduled_tokens or {}),
        total_num_scheduled_tokens=total,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids={"done-r0"},
        free_encoder_mm_hashes=[],
    )


def _make_batch_req_state(num_tokens):
    return BatchReqState(
        req_ids=["r0"],
        num_scheduled_tokens=np.array([num_tokens], dtype=np.int32),
        num_tokens=num_tokens,
        idx_mapping_np=np.array([0], dtype=np.intp),
        prefill_len_np=np.array([0], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0], dtype=np.int32),
        is_prefilling_np=np.array([False]),
        has_prefill=False,
    )


@pytest.mark.parametrize(
    "enabled,dummy_run,is_profile,total,expected_calls",
    [
        (True, False, False, 5, 1),  # real step with work: agree the group max
        # Zero-token steps return before dispatch without any DP collective; an
        # all_reduce here would desync the gloo op stream against busy/dummy steps.
        (True, False, False, 0, 0),
        (True, True, False, 0, 1),  # dummy step (idle rank): report zero
        (True, False, True, 5, 0),  # profile run
        (False, False, False, 5, 0),  # feature off
    ],
)
def test_dp_padding_execute_model_gate(enabled, dummy_run, is_profile, total, expected_calls):
    runner = _make_dp_padding_runner(enabled=enabled)
    scheduler_output = _make_scheduler_output(total, {"r0": total} if total else {})

    with (
        patch("vllm_ascend.worker.v2.model_runner.sync_dp_group_max_tokens", return_value=0) as sync,
        patch("vllm_ascend.worker.v2.model_runner.make_dp_padded_dummy_output") as make_dummy,
        patch.object(NPUModelRunner.__bases__[0], "execute_model", return_value="out") as super_execute,
    ):
        assert runner.execute_model(scheduler_output, dummy_run=dummy_run, is_profile=is_profile) == "out"

    assert sync.call_count == expected_calls
    if expected_calls:
        # dummy ranks report zero, so only real work raises the group max
        assert sync.call_args.args == (0 if dummy_run else total, 8, 3)
    make_dummy.assert_not_called()
    assert super_execute.call_args.args[0] is scheduler_output


def test_dp_padding_dummy_step_forwards_group_max():
    """An idle rank must run its dummy batch at the group's agreed size, and the
    rewrite must leave the rest of the synthetic output intact."""
    runner = _make_dp_padding_runner()
    scheduler_output = _make_scheduler_output(0)

    with (
        patch("vllm_ascend.worker.v2.model_runner.sync_dp_group_max_tokens", return_value=4),
        patch.object(NPUModelRunner.__bases__[0], "execute_model", return_value="out") as super_execute,
    ):
        runner.execute_model(scheduler_output, dummy_run=True)

    forwarded = super_execute.call_args.args[0]
    assert forwarded.total_num_scheduled_tokens == 4
    # decode_query_len-sized requests keep the dummy decode-graph matchable
    assert list(forwarded.num_scheduled_tokens.values()) == [2, 2]
    assert forwarded.finished_req_ids == {"done-r0"}


def test_dp_padded_dummy_output_keeps_decode_shape():
    """The rewrite must stay uniform-decode shaped (graph match) with an exact total."""
    out = make_dp_padded_dummy_output(_make_scheduler_output(0), group_max=4, decode_query_len=2, max_num_reqs=8)
    assert list(out.num_scheduled_tokens.values()) == [2, 2]
    assert out.total_num_scheduled_tokens == 4

    # A remainder keeps the total exact; the batch is non-uniform so no decode graph matches.
    out = make_dp_padded_dummy_output(_make_scheduler_output(0), group_max=5, decode_query_len=2, max_num_reqs=8)
    assert list(out.num_scheduled_tokens.values()) == [2, 2, 1]
    assert out.total_num_scheduled_tokens == 5

    # Past max_num_reqs the decode graphs cannot match either; use _dummy_run's bounded even split.
    out = make_dp_padded_dummy_output(_make_scheduler_output(0), group_max=9, decode_query_len=2, max_num_reqs=3)
    assert list(out.num_scheduled_tokens.values()) == [3, 3, 3]
    assert out.total_num_scheduled_tokens == 9

    # The fallback also covers a non-divisible total.
    out = make_dp_padded_dummy_output(_make_scheduler_output(0), group_max=10, decode_query_len=2, max_num_reqs=3)
    assert list(out.num_scheduled_tokens.values()) == [3, 3, 4]
    assert out.total_num_scheduled_tokens == 10

    # decode_query_len == 1: one request per token, still decode-shaped.
    out = make_dp_padded_dummy_output(_make_scheduler_output(0), group_max=4, decode_query_len=1, max_num_reqs=8)
    assert list(out.num_scheduled_tokens.values()) == [1, 1, 1, 1]
    assert out.total_num_scheduled_tokens == 4


def test_prepare_inputs_restores_real_extent_wiring():
    """The restore branch in prepare_inputs must read the recorded real extent.

    Full behavioral coverage lands with the UT rework; this pins the wiring so a
    refactor cannot silently drop it.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(NPUModelRunner.prepare_inputs)))
    restore = [
        n for n in ast.walk(tree) if isinstance(n, ast.If) and "_dp_padding_aligned_tokens" in ast.unparse(n.test)
    ]
    assert len(restore) == 1
    assert "_dp_padding_original_tokens" in ast.unparse(restore[0].body)
    assert "batch_req_state.num_tokens" in ast.unparse(restore[0].orelse)
    # the padded row count still derives from the restored value
    assert "max(num_tokens, batch_desc.num_tokens)" in ast.unparse(tree)
    fills = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Subscript)
        and ast.unparse(n.targets[0]).startswith("query_start_loc_np[num_reqs")
        and isinstance(n.value, ast.Name)
        and n.value.id == "num_tokens"
    ]
    assert fills, "the trailing query_start_loc fill must use the restored num_tokens"


@pytest.mark.parametrize(
    "aligned,parent_tokens,expected_reported,expected_original",
    [
        (8, 5, 8, 5),  # dispatch sees the group max, the real extent stays recoverable
        (8, 8, 8, 8),  # this rank already is the group max
        (0, 5, 5, 5),  # padding inactive
        (8, None, None, 0),  # dummy step: no parent state to pad or record
    ],
)
def test_dp_padding_gather_reports_group_max(aligned, parent_tokens, expected_reported, expected_original):
    runner = _make_dp_padding_runner(aligned_tokens=aligned)
    dummy_run = parent_tokens is None
    parent = (None, 4) if dummy_run else (_make_batch_req_state(parent_tokens), None)

    with patch.object(NPUModelRunner.__bases__[0], "gather_batch_req_state", return_value=parent):
        # the scheduler output still reports the untrimmed total
        batch_req_state, uniform_tok_count = runner.gather_batch_req_state(_make_scheduler_output(8), dummy_run)

    assert (None if batch_req_state is None else batch_req_state.num_tokens) == expected_reported
    assert runner._dp_padding_original_tokens == expected_original
    assert uniform_tok_count == (4 if dummy_run else None)


def test_dp_padding_sync_is_one_hot_then_max():
    """The agreement mirrors dp_utils.sync_cudagraph_and_dp_padding: only this
    rank's slot carries a count, and the returned value is the vector max."""
    seen = {}

    def fake_all_reduce(tensor, group=None):
        seen["sent"] = tensor.clone()
        seen["group"] = group
        tensor[2] = 5  # some rank in the group reported 5 tokens

    with (
        patch("vllm_ascend.worker.v2.eager_dp_padding.get_dp_group", return_value=SimpleNamespace(cpu_group="dp-cpu")),
        patch("vllm_ascend.worker.v2.eager_dp_padding.dist.all_reduce", fake_all_reduce),
    ):
        group_max = sync_dp_group_max_tokens(3, dp_size=4, dp_rank=1)

    assert seen["sent"].tolist() == [0, 3, 0, 0]
    assert seen["group"] == "dp-cpu"
    assert group_max == 5


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
