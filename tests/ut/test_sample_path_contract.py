"""Sample-path DFX hook contract UTs.

Locks in the ordering / gating contract of the 8 sample-path DFX hooks
orchestrated by ``NPUModelRunner.sample_tokens`` +
``DfxProcessor.run_sample_phase``. Scenario IDs (S1/S2/S4/S6/S16/S17)
trace back to the design doc's timing-constraint table.

Hook numbering:

    1. ``check_before_sample``           runner, before ``apply_grammar_bitmask``
    2. ``ensure_logprobs_for_detection`` ``run_sample_phase``, before ``sample_fn``
    3. ``finalize_dump_data``
    4. ``note_kv_block_writes``
    5. ``mark_finished``
    6. ``check_after_spec``              spec + ``should_check_after_spec``
    7. ``record_sample_waves``           always; async: before handoff
    8. ``check_after_sample``            sync path only

Timing invariants enforced here:

    - 1 → 2 → 3 → 4 → 5 → [6 if spec] → 7 → [8 if sync]
    - ``finalize_dump_data`` (3) before ``note_kv_block_writes`` (4)
    - ``mark_finished`` (5) before ``record_sample_waves`` (7)
    - async: ``record_sample_waves`` (7) before ``AscendAsyncGPUModelRunnerOutput``
      handoff (else races the next ``sync_for_step``)
    - async: ``check_after_sample`` (8) deferred (runs on the async thread)
    - spec + ``need_accepted_tokens``: ``num_accepted_tokens_event.synchronize``
      before ``check_after_spec`` (6)
    - PP non-final rank (``execute_model_state=None``): no hooks at all
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.dfx.processor import DfxProcessor


# ---------- fixtures ----------

SAMPLE_HOOKS = [
    "check_before_sample",
    "ensure_logprobs_for_detection",
    "finalize_dump_data",
    "note_kv_block_writes",
    "mark_finished",
    "check_after_spec",
    "record_sample_waves",
    "check_after_sample",
]


@pytest.fixture
def instrumented_dfx():
    """Bare DfxProcessor with the 8 sample-path hooks wrapped to record calls.

    Uses the ``DfxProcessor.__new__`` pattern (matches test_dfx_processor.py)
    to avoid heavy ``__init__``. All hooks are MagicMock so call order/args
    are inspectable via mock_calls and a flat call log.
    """
    proc = DfxProcessor.__new__(DfxProcessor)
    proc.runner = MagicMock(tp_rank=0)
    proc.dfx_config = MagicMock()
    proc.dumper = MagicMock()
    proc.report_writer = MagicMock()
    proc.detectors = MagicMock()
    proc.save_sample_param = MagicMock()
    proc._get_report_tokenizer = MagicMock(return_value=None)

    for name in SAMPLE_HOOKS:
        setattr(proc, name, MagicMock(name=name))

    proc._recordings = []

    def make_recorder(n, mock):
        def w(*a, **kw):
            proc._recordings.append(n)
            return mock.return_value
        return w

    for name in SAMPLE_HOOKS:
        m = getattr(proc, name)
        m.side_effect = make_recorder(name, m)

    return proc


def _hook_order(proc) -> list[str]:
    """Return the flat list of hook names in call order."""
    return list(proc._recordings)


def _build_minimal_runner_mock(instrumented_dfx, *, use_async=False, spec=False):
    """MagicMock runner with ``sample_tokens`` bound from the real class.

    The test exercises the actual runner code path; heavy helpers
    (``_sample``, ``_bookkeeping_sync``, draft proposal, …) are mocked so the
    test focuses on DFX hook ordering, not sampling logic.
    """
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner as _RunnerCls

    runner = MagicMock(spec=_RunnerCls, tp_rank=0)
    runner.sample_tokens = _RunnerCls.sample_tokens.__get__(runner)

    runner.dfx = instrumented_dfx
    runner.use_async_scheduling = use_async
    runner.speculative_config = MagicMock() if spec else None
    runner.need_accepted_tokens = False
    runner.sampling_done_event = MagicMock() if spec else None
    runner.valid_sampled_token_count_gpu = None
    runner.input_batch = MagicMock()
    runner.input_batch.sampling_metadata = MagicMock()
    runner.device = MagicMock()
    runner.logits_indices = None
    runner.ascend_config = MagicMock()
    runner.ascend_config.scheduler_config.profiling_chunk_config = MagicMock(enabled=False)
    runner.kv_connector_output = None
    runner.execute_model_state = None  # set per-scenario
    runner.dynamic_eplb = False
    runner.routed_experts_initialized = False
    runner.propose_draft_token_ids = MagicMock(return_value=None)
    runner._copy_draft_token_ids_to_cpu = MagicMock()
    runner._update_states_after_model_execute = MagicMock()
    runner._reap_finished_requests = MagicMock()
    # Hook 2 is owned by run_sample_phase (before sample_fn); _sample must
    # stay a pure runner call.
    def _fake_sample(logits, spec_decode_metadata):
        return _make_sampler_output()
    runner._sample = MagicMock(side_effect=_fake_sample)
    runner._bookkeeping_sync = MagicMock(return_value=(
        None,                       # logprobs_lists
        [[1, 2, 3]],                # valid_sampled_token_ids
        {},                         # prompt_logprobs_dict
        ["r1"],                     # req_ids_output_copy
        {"r1": 0},                  # req_id_to_index_output_copy
        [],                         # invalid_req_indices
    ))
    runner.finalize_kv_connector = MagicMock()
    runner.supports_mm_inputs = False
    runner._sync_device = MagicMock()

    return runner


def _make_sampler_output():
    """Minimal sampler output object for sample_tokens."""
    so = MagicMock()
    so.sampled_token_ids = [[1, 2, 3]]
    so.logprobs_tensors = None
    return so


def _make_execute_model_state():
    """Build the 12-tuple execute_model_state matching sample_tokens unpack."""
    so = MagicMock()
    so.total_num_scheduled_tokens = 4
    so.finished_req_ids = None
    so.num_scheduled_tokens = {"r1": 4}
    so.num_reqs = 1
    logits = MagicMock()
    logits.dtype = MagicMock()
    positions = MagicMock()
    return (
        so,                           # scheduler_output
        logits,                       # logits
        None,                         # spec_decode_metadata
        MagicMock(),                  # spec_decode_common_attn_metadata
        MagicMock(),                  # hidden_states
        MagicMock(),                  # sample_hidden_states
        MagicMock(),                  # aux_hidden_states
        MagicMock(),                  # attn_metadata
        positions,                    # positions
        None,                         # ec_connector_output
        MagicMock(),                  # cudagraph_stats
        MagicMock(),                  # batch_desc
    )


@contextmanager
def _patch_sample_helpers():
    """Patch module-level helpers that sample_tokens calls."""
    with (
        patch("vllm_ascend.worker.model_runner_v1.get_pp_group") as pp,
        patch("vllm_ascend.worker.model_runner_v1.record_function_or_nullcontext"),
        patch("vllm_ascend.worker.model_runner_v1.apply_grammar_bitmask"),
    ):
        pp.return_value = MagicMock(world_size=1)
        yield


# ---------- contract tests ----------

class TestSamplePathContract:
    """Long-term regression guard for sample-path DFX hook orchestration."""

    def test_golden_path_sync_hook_order(self, instrumented_dfx):
        """S1: sync + non-spec + no finished → 1→2→3→4→5(None)→7→8."""
        runner = _build_minimal_runner_mock(instrumented_dfx, use_async=False, spec=False)
        runner.execute_model_state = _make_execute_model_state()

        with _patch_sample_helpers():
            runner.sample_tokens(grammar_output=None)

        # hook 6 (check_after_spec) skipped — speculative_config is None
        # hook 5 mark_finished receives finished_req_ids=None
        expected = [
            "check_before_sample",            # 1
            "ensure_logprobs_for_detection",  # 2
            "finalize_dump_data",             # 3
            "note_kv_block_writes",           # 4
            "mark_finished",                  # 5
            "record_sample_waves",            # 7
            "check_after_sample",             # 8
        ]
        assert _hook_order(instrumented_dfx) == expected
        instrumented_dfx.mark_finished.assert_called_once_with(None)

    def test_spec_event_synchronize_before_check_after_spec(self, instrumented_dfx):
        """S2: spec + need_accepted_tokens → 1→2→3→4→5→6→7→8 with event sync before 6.

        ``num_accepted_tokens_event.synchronize()`` must complete before
        ``check_after_spec`` reads ``num_accepted_tokens_cpu``, and only
        fire once per step.
        """
        runner = _build_minimal_runner_mock(instrumented_dfx, spec=True)
        runner.execute_model_state = _make_execute_model_state()
        runner.need_accepted_tokens = True
        runner.num_accepted_tokens_event = MagicMock()
        runner.input_batch.num_accepted_tokens_cpu = [2, 2, 2]
        sync_calls = []
        runner.num_accepted_tokens_event.synchronize.side_effect = (
            lambda: sync_calls.append(len(instrumented_dfx._recordings))
        )

        with _patch_sample_helpers():
            runner.sample_tokens(grammar_output=None)

        expected = [
            "check_before_sample", "ensure_logprobs_for_detection",
            "finalize_dump_data", "note_kv_block_writes",
            "mark_finished", "check_after_spec",
            "record_sample_waves", "check_after_sample",
        ]
        assert _hook_order(instrumented_dfx) == expected
        # event.synchronize fired exactly once, before hook 6 — i.e. after
        # 5 hooks were recorded (1,2,3,4,5), before check_after_spec.
        assert len(sync_calls) == 1
        assert sync_calls[0] == 5

    def test_async_defers_hook8_and_stamps_wave_before_handoff(self, instrumented_dfx):
        """S4: async → 1→2→3→4→5→7; hook 8 deferred; wave stamp before handoff.

        ``record_sample_waves`` (7) must run on the main thread BEFORE
        ``AscendAsyncGPUModelRunnerOutput`` construction hands the step off
        to the async copy thread, otherwise the wave stamp races the next
        ``sync_for_step``.
        """
        runner = _build_minimal_runner_mock(instrumented_dfx, use_async=True, spec=False)
        runner.execute_model_state = _make_execute_model_state()
        runner.async_output_copy_stream = MagicMock()
        runner.input_batch.vocab_size = 1000
        runner.input_batch.set_async_sampled_token_ids = MagicMock()

        construct_at = []
        def _track_construct(*a, **kw):
            construct_at.append(len(instrumented_dfx._recordings))
            return MagicMock()
        with _patch_sample_helpers():
            with patch(
                "vllm_ascend.worker.model_runner_v1.AscendAsyncGPUModelRunnerOutput",
                side_effect=_track_construct,
            ):
                runner.sample_tokens(grammar_output=None)

        # hook 6 skipped (spec=False); hook 8 deferred to the async thread
        expected = [
            "check_before_sample", "ensure_logprobs_for_detection",
            "finalize_dump_data", "note_kv_block_writes",
            "mark_finished", "record_sample_waves",
        ]
        assert _hook_order(instrumented_dfx) == expected
        assert len(construct_at) == 1
        # handoff fired after all 6 hooks (incl. record_sample_waves at index 5)
        assert construct_at[0] == 6

    def test_pp_non_final_rank_calls_no_hooks(self, instrumented_dfx):
        """S6: execute_model_state=None + no kv_connector_output → no hooks, return None."""
        runner = _build_minimal_runner_mock(instrumented_dfx)
        runner.execute_model_state = None
        runner.kv_connector_output = None

        with _patch_sample_helpers():
            result = runner.sample_tokens(grammar_output=None)

        assert result is None
        assert _hook_order(instrumented_dfx) == []

    def test_spec_async_need_accepted_combined(self, instrumented_dfx):
        """S16: spec + need_accepted + async → 1→2→3→4→5→6→7; 8 deferred; sync before 6.

        Strictest combo: the event-synchronize constraint (S2) and the
        wave-stamp-before-handoff constraint (S4) must both hold at once.
        """
        runner = _build_minimal_runner_mock(instrumented_dfx, use_async=True, spec=True)
        runner.execute_model_state = _make_execute_model_state()
        runner.need_accepted_tokens = True
        runner.num_accepted_tokens_event = MagicMock()
        runner.async_output_copy_stream = MagicMock()
        runner.input_batch.vocab_size = 1000
        runner.input_batch.num_accepted_tokens_cpu = [2, 2, 2]
        runner.input_batch.set_async_sampled_token_ids = MagicMock()

        sync_calls = []
        runner.num_accepted_tokens_event.synchronize.side_effect = (
            lambda: sync_calls.append(len(instrumented_dfx._recordings))
        )
        construct_at = []
        def _track_construct(*a, **kw):
            construct_at.append(len(instrumented_dfx._recordings))
            return MagicMock()

        with _patch_sample_helpers():
            with patch(
                "vllm_ascend.worker.model_runner_v1.AscendAsyncGPUModelRunnerOutput",
                side_effect=_track_construct,
            ):
                runner.sample_tokens(grammar_output=None)

        # hook 6 called (spec); hook 8 deferred (async)
        expected = [
            "check_before_sample", "ensure_logprobs_for_detection",
            "finalize_dump_data", "note_kv_block_writes",
            "mark_finished", "check_after_spec", "record_sample_waves",
        ]
        assert _hook_order(instrumented_dfx) == expected
        # event.synchronize before check_after_spec (hook 6, index 5)
        assert len(sync_calls) == 1
        assert sync_calls[0] == 5
        # handoff after record_sample_waves (hook 7, index 6)
        assert construct_at[0] == 7

    def test_spec_async_no_need_accepted(self, instrumented_dfx):
        """S17: spec + non-need_accepted + async → 6 uses accepted_token_counts.

        Contrast to S16: no event.synchronize on this path; hook 6's
        ``accepted_token_nums`` comes from the CPU-side
        ``accepted_token_counts`` helper instead of the async event copy.
        """
        runner = _build_minimal_runner_mock(instrumented_dfx, use_async=True, spec=True)
        runner.execute_model_state = _make_execute_model_state()
        runner.need_accepted_tokens = False
        runner.async_output_copy_stream = MagicMock()
        runner.input_batch.vocab_size = 1000
        runner.input_batch.set_async_sampled_token_ids = MagicMock()

        construct_at = []
        def _track_construct(*a, **kw):
            construct_at.append(len(instrumented_dfx._recordings))
            return MagicMock()

        with _patch_sample_helpers():
            with patch(
                "vllm_ascend.worker.model_runner_v1.AscendAsyncGPUModelRunnerOutput",
                side_effect=_track_construct,
            ):
                with patch(
                    "vllm_ascend.worker.model_runner_v1.accepted_token_counts",
                    return_value=[1, 2, 3],
                ):
                    runner.sample_tokens(grammar_output=None)

        expected = [
            "check_before_sample", "ensure_logprobs_for_detection",
            "finalize_dump_data", "note_kv_block_writes",
            "mark_finished", "check_after_spec", "record_sample_waves",
        ]
        assert _hook_order(instrumented_dfx) == expected
        _, kwargs = instrumented_dfx.check_after_spec.call_args
        assert kwargs["accepted_token_nums"] == [1, 2, 3]
        assert construct_at[0] == 7
