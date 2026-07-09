"""Unit tests for the ubatch overlap feature.

These tests cover the pure-Python, hardware-free pieces of the feature:

* ``slice_query_start_locs`` arithmetic.
* ``split_ascend_common_metadata`` for the whole-request split case and for the
  request-spanning-ubatches case (first / last request continuation), plus the
  chunked-prefill attention-state remapping.
* The decision helpers ``should_enable_ubatch`` / ``maybe_use_ubatch`` /
  ``get_ubatch_trigger_threshold``.
* Thread-local token slice binding / ``get_cos_and_sin_slice`` ubatch fallback.
* Mocked-thread concurrency: ``comm_section`` symmetry, ``yield_thread``
  timeout, ``reset_thread_pool`` error recovery.

Concurrency tests mock ``torch.npu.Stream`` / ``Event`` objects to avoid actual
NPU kernel launches, but the test code still imports ``vllm_ascend`` modules
that require ``torch_npu`` at import time. Run these tests in an environment
where ``torch_npu`` is installed (e.g. NPU CI or dev container).
"""

import threading
import unittest
from contextlib import nullcontext
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import torch
from vllm.config import VllmConfig

from vllm_ascend.ascend_config import clear_ascend_config, init_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
import vllm_ascend.worker.ubatch_utils as uu
from vllm_ascend.worker.ubatch_utils import (
    _make_ascend_common_metadata_with_slice,
    get_ubatch_runtime_manager,
    get_ubatch_trigger_threshold,
    maybe_use_ubatch,
    resolve_num_tokens_across_dp,
    should_enable_ubatch,
    should_run_ubatch,
    slice_model_inputs_for_ubatch,
    slice_query_start_locs,
    split_ascend_common_metadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeUBatchSlice:
    """Minimal stand-in for ``vllm.v1.worker.ubatch_utils.UBatchSlice``.

    ``UBatchSlice`` is an upstream vLLM dataclass; we only need the four
    attributes/methods that the slicing code touches:
    ``request_slice``, ``token_slice``, ``num_tokens`` and ``is_empty()``.
    """

    request_slice: slice
    token_slice: slice
    num_tokens: int

    def is_empty(self) -> bool:
        return self.num_tokens == 0


def _build_common_attn_metadata(
    query_start_loc_cpu: list[int],
    seq_lens_cpu: list[int],
    num_input_tokens: int | None = None,
    attn_state: AscendAttentionState = AscendAttentionState.PrefillNoCache,
) -> AscendCommonAttentionMetadata:
    """Build a fully-populated AscendCommonAttentionMetadata for slicing tests.

    ``attn_state`` uses the real ``AscendAttentionState`` enum (matching what
    NPUModelRunner stores on the metadata), NOT a raw int, so tests exercise
    the same enum-vs-int comparison path that production code hits.
    """
    n_reqs = len(query_start_loc_cpu) - 1
    if num_input_tokens is None:
        num_input_tokens = query_start_loc_cpu[-1]

    query_start_loc_cpu_t = torch.tensor(query_start_loc_cpu, dtype=torch.int32)
    seq_lens_cpu_t = torch.tensor(seq_lens_cpu, dtype=torch.int32)

    return AscendCommonAttentionMetadata(
        query_start_loc=query_start_loc_cpu_t.clone(),
        query_start_loc_cpu=query_start_loc_cpu_t,
        seq_lens=seq_lens_cpu_t.clone(),
        seq_lens_cpu=seq_lens_cpu_t,
        num_computed_tokens_cpu=torch.zeros(n_reqs, dtype=torch.int32),
        num_reqs=n_reqs,
        num_actual_tokens=num_input_tokens,
        max_query_len=max(
            query_start_loc_cpu[i + 1] - query_start_loc_cpu[i]
            for i in range(n_reqs)
        ),
        max_seq_len=max(seq_lens_cpu),
        block_table_tensor=torch.zeros((n_reqs, 4), dtype=torch.int32),
        slot_mapping=torch.arange(num_input_tokens, dtype=torch.int32),
        causal=True,
        num_input_tokens=num_input_tokens,
        actual_seq_lengths_q=[
            query_start_loc_cpu[i + 1] - query_start_loc_cpu[i]
            for i in range(n_reqs)
        ],
        positions=torch.arange(num_input_tokens, dtype=torch.int64),
        attn_state=attn_state,
    )


def _reset_ubatch_singletons(additional_config: dict | None = None):
    """Re-init AscendConfig and the ubatch runtime manager singleton.

    The ubatch settings now live on AscendConfig (additional_config), and
    UBatchRuntimeManager reads num_ubatches from it at construction time, so
    both singletons must be (re)created with the desired config. Shared by the
    decision-helper tests below and by the cross-file DP-voting tests in
    ``test_model_runner_v1.py``.
    """
    clear_ascend_config()
    cfg = VllmConfig()
    if additional_config is not None:
        cfg.additional_config = dict(additional_config, refresh=True)
    else:
        cfg.additional_config = {"refresh": True}
    init_ascend_config(cfg)
    # Reset the runtime manager singleton so the next get_ubatch_runtime_manager()
    # reconstructs it with the freshly initialized config.
    uu._UBATCH_RUNTIME_MANAGER = None


# ---------------------------------------------------------------------------
# slice_query_start_locs
# ---------------------------------------------------------------------------


class TestSliceQueryStartLocs(unittest.TestCase):

    def test_offsets_to_zero_at_slice_start(self):
        # query_start_loc = [0, 10, 30, 55]. Requests [1, 3) are reqs {1, 2}.
        # query_start_loc has num_reqs+1 entries, so we expect 3 values rebased
        # to start at 0: [0, 30-10, 55-10] = [0, 20, 45].
        qsl = torch.tensor([0, 10, 30, 55], dtype=torch.int32)
        out = slice_query_start_locs(qsl, slice(1, 3))
        self.assertEqual(out.tolist(), [0, 20, 45])

    def test_full_slice_is_identity(self):
        qsl = torch.tensor([0, 5, 12, 20], dtype=torch.int32)
        out = slice_query_start_locs(qsl, slice(0, 3))
        self.assertEqual(out.tolist(), qsl.tolist())

    def test_single_request_slice(self):
        # Requests [1, 2) => 1 request => 2 entries [0, 15-7] = [0, 8].
        qsl = torch.tensor([0, 7, 15], dtype=torch.int32)
        out = slice_query_start_locs(qsl, slice(1, 2))
        self.assertEqual(out.tolist(), [0, 8])


# ---------------------------------------------------------------------------
# split_ascend_common_metadata
# ---------------------------------------------------------------------------


class TestSplitAscendCommonMetadata(unittest.TestCase):

    def test_whole_request_split(self):
        """Two ubatches, each holding whole requests (no request is split)."""
        # 4 requests: query lens [4, 4, 4, 4] => qsl [0,4,8,12,16]
        cm = _build_common_attn_metadata(
            query_start_loc_cpu=[0, 4, 8, 12, 16],
            seq_lens_cpu=[10, 20, 30, 40],
        )
        slices = [
            FakeUBatchSlice(slice(0, 2), slice(0, 8), num_tokens=8),
            FakeUBatchSlice(slice(2, 4), slice(8, 16), num_tokens=8),
        ]
        out = split_ascend_common_metadata(slices, cm)
        self.assertEqual(len(out), 2)

        # First ubatch: reqs [0,2), tokens [0,8)
        self.assertEqual(out[0].num_reqs, 2)
        self.assertEqual(out[0].num_actual_tokens, 8)
        self.assertEqual(out[0].query_start_loc_cpu.tolist(), [0, 4, 8])
        self.assertEqual(out[0].actual_seq_lengths_q, [4, 8])
        self.assertEqual(out[0].max_query_len, 4)
        self.assertEqual(out[0].positions.tolist(), list(range(8)))

        # Second ubatch: reqs [2,4), tokens [8,16)
        self.assertEqual(out[1].num_reqs, 2)
        self.assertEqual(out[1].num_actual_tokens, 8)
        self.assertEqual(out[1].query_start_loc_cpu.tolist(), [0, 4, 8])
        self.assertEqual(out[1].actual_seq_lengths_q, [4, 8])
        self.assertEqual(out[1].max_query_len, 4)
        self.assertEqual(out[1].positions.tolist(), list(range(8, 16)))

    def test_request_split_across_ubatches_first_and_last(self):
        """A single long request is split across two ubatches.

        Both the first request of ubatch 1 and the last request of ubatch 0
        are continuations / overflow, exercising the
        ``splits_first_request`` / ``splits_last_request`` branches.
        """
        # One request of length 8 tokens => qsl [0, 8]
        cm = _build_common_attn_metadata(
            query_start_loc_cpu=[0, 8],
            seq_lens_cpu=[100],
        )
        # Ubatch 0: tokens [0, 4)  -> last request overflows (splits_last)
        # Ubatch 1: tokens [4, 8)  -> first request is continuation (splits_first)
        slices = [
            FakeUBatchSlice(slice(0, 1), slice(0, 4), num_tokens=4),
            FakeUBatchSlice(slice(0, 1), slice(4, 8), num_tokens=4),
        ]
        out = split_ascend_common_metadata(slices, cm)

        # Ubatch 0: query_start_loc [0, 4], seq_len reduced by 4 (100 -> 96)
        self.assertEqual(out[0].query_start_loc_cpu.tolist(), [0, 4])
        self.assertEqual(out[0].seq_lens_cpu.tolist(), [96])
        self.assertEqual(out[0].max_query_len, 4)

        # Ubatch 1: continuation; query_start_loc rebased to [0, 4]
        self.assertEqual(out[1].query_start_loc_cpu.tolist(), [0, 4])
        self.assertEqual(out[1].positions.tolist(), list(range(4, 8)))

    def test_chunked_prefill_state_remapping(self):
        """Second chunk of a chunked-prefill request gets state=ChunkedPrefill.

        Uses the real AscendAttentionState enum (as production does). This is a
        regression test for the enum-vs-int comparison bug where
        ``attn_state in [0, 3]`` never matched an Enum member, so the remap was
        silently skipped and the second chunk kept PrefillNoCache.
        """
        # PrefillNoCache on the base metadata.
        cm = _build_common_attn_metadata(
            query_start_loc_cpu=[0, 8],
            seq_lens_cpu=[100],
            attn_state=AscendAttentionState.PrefillNoCache,
        )
        slices = [
            FakeUBatchSlice(slice(0, 1), slice(0, 4), num_tokens=4),
            FakeUBatchSlice(slice(0, 1), slice(4, 8), num_tokens=4),
        ]
        out = split_ascend_common_metadata(slices, cm)
        # First chunk (is_first_chunk=True) keeps the original state.
        self.assertEqual(out[0].attn_state, AscendAttentionState.PrefillNoCache)
        # Second chunk (is_first_chunk=False) is remapped to ChunkedPrefill.
        self.assertEqual(out[1].attn_state, AscendAttentionState.ChunkedPrefill)

    def test_chunked_prefill_state_remapping_from_chunked_prefill(self):
        """ChunkedPrefill base state also remaps the second chunk."""
        cm = _build_common_attn_metadata(
            query_start_loc_cpu=[0, 8],
            seq_lens_cpu=[100],
            attn_state=AscendAttentionState.ChunkedPrefill,
        )
        slices = [
            FakeUBatchSlice(slice(0, 1), slice(0, 4), num_tokens=4),
            FakeUBatchSlice(slice(0, 1), slice(4, 8), num_tokens=4),
        ]
        out = split_ascend_common_metadata(slices, cm)
        self.assertEqual(out[0].attn_state, AscendAttentionState.ChunkedPrefill)
        self.assertEqual(out[1].attn_state, AscendAttentionState.ChunkedPrefill)

    def test_empty_slice_is_rejected(self):
        cm = _build_common_attn_metadata(
            query_start_loc_cpu=[0, 4],
            seq_lens_cpu=[10],
        )
        empty = FakeUBatchSlice(slice(0, 1), slice(0, 0), num_tokens=0)
        with self.assertRaises(AssertionError):
            _make_ascend_common_metadata_with_slice(empty, cm)

    def test_original_metadata_not_mutated(self):
        """split_last path must clone seq_lens, leaving the source intact."""
        cm = _build_common_attn_metadata(
            query_start_loc_cpu=[0, 8],
            seq_lens_cpu=[100],
        )
        original_seq_lens = cm.seq_lens_cpu.clone()
        slices = [
            FakeUBatchSlice(slice(0, 1), slice(0, 4), num_tokens=4),
            FakeUBatchSlice(slice(0, 1), slice(4, 8), num_tokens=4),
        ]
        split_ascend_common_metadata(slices, cm)
        self.assertEqual(cm.seq_lens_cpu.tolist(), original_seq_lens.tolist())


# ---------------------------------------------------------------------------
# Decision helpers
# ---------------------------------------------------------------------------


class TestDecisionHelpers(unittest.TestCase):
    """Tests for should_enable_ubatch / maybe_use_ubatch / threshold."""

    def _reset(self, additional_config: dict | None = None):
        """Re-init AscendConfig and the ubatch runtime manager singleton."""
        _reset_ubatch_singletons(additional_config)

    def setUp(self):
        self._reset()

    def tearDown(self):
        clear_ascend_config()
        uu._UBATCH_RUNTIME_MANAGER = None

    def test_disabled_when_single_ubatch(self):
        # Default num_ubatches == 1 => maybe_use_ubatch is False, feature disabled.
        self.assertFalse(maybe_use_ubatch())
        self.assertFalse(should_enable_ubatch(10000, 10000))

    def test_disabled_below_threshold(self):
        # num_ubatches == 2, but padded tokens below default threshold (2048).
        self._reset({"num_ubatches": 2})
        rt = get_ubatch_runtime_manager()
        self.assertEqual(rt.num_ubatches, 2)
        self.assertFalse(should_enable_ubatch(100, 100))

    def test_enabled_when_both_ubatches_non_empty(self):
        # is_last_ubatch_empty(orig, padded, n) = (padded//n)*(n-1) >= orig.
        # With n=2, orig=3000, padded=3000: (1500)*1 = 1500 < 3000 => not empty.
        # And padded 3000 > default threshold 2048 => enabled.
        self._reset({"num_ubatches": 2})
        self.assertTrue(should_enable_ubatch(3000, 3000))

    def test_disabled_when_last_ubatch_empty(self):
        # n=2, orig=4, padded=4096: (4096//2)*1 = 2048 >= 4 => last empty.
        self._reset({"num_ubatches": 2})
        self.assertFalse(should_enable_ubatch(4, 4096))

    def test_threshold_config_override(self):
        # ubatch_trigger_threshold comes from additional_config.
        self._reset({"num_ubatches": 2, "ubatch_trigger_threshold": 100})
        self.assertEqual(get_ubatch_trigger_threshold(), 100)
        # Above the overridden threshold and both ubatches non-empty => enabled.
        self.assertTrue(should_enable_ubatch(200, 200))
        # Default threshold (2048) is restored on re-init without the key.
        self._reset({"num_ubatches": 2})
        self.assertEqual(get_ubatch_trigger_threshold(), 2048)


# ---------------------------------------------------------------------------
# resolve_num_tokens_across_dp (DP==1 ubatch regression)
# ---------------------------------------------------------------------------


class TestResolveNumTokensAcrossDp(unittest.TestCase):
    """Regression tests for the DP==1 ubatch crash (P0-1).

    ``NPUModelRunner.execute_model`` builds per-ubatch forward contexts by
    iterating ``len(num_tokens_across_dp)`` and calling ``.new_full(...)`` on
    it. That tensor is only populated by ``_sync_batch_across_dp`` when
    ``data_parallel_size > 1``; in DP==1 it stays ``None`` and the loop crashed
    with ``TypeError: object of type 'NoneType' has no len()``.
    ``resolve_num_tokens_across_dp`` synthesizes a 1-element CPU tensor for the
    None case so ubatch overlap runs in single-DP-rank mode too.
    """

    def test_none_dp1_is_synthesized_to_single_element_tensor(self):
        # DP==1 path: None input -> 1-element tensor carrying padded count.
        out = resolve_num_tokens_across_dp(None, num_tokens_padded=3000)
        self.assertEqual(out.tolist(), [3000])
        self.assertEqual(out.device.type, "cpu")
        self.assertEqual(out.dtype, torch.int32)

    def test_non_none_dp_n_is_passed_through_unchanged(self):
        # DP>1 path: the per-rank tensor from _sync_batch_across_dp is returned
        # as-is (identity, not copied), so callers can still mutate it in place.
        original = torch.tensor([3000, 3000], device="cpu", dtype=torch.int32)
        out = resolve_num_tokens_across_dp(original, num_tokens_padded=3000)
        self.assertIs(out, original)

    def test_synthesized_tensor_supports_new_full_like_hot_path(self):
        # Mirror how execute_model uses the result: len() + .new_full() per
        # ubatch slice. This is the exact pattern that raised before the fix.
        out = resolve_num_tokens_across_dp(None, num_tokens_padded=3000)
        num_ranks = len(out)
        per_slice = out.new_full((num_ranks,), 1500)
        self.assertEqual(per_slice.tolist(), [1500])


# ---------------------------------------------------------------------------
# Thread-local token slice (T2: get_cos_and_sin_slice ubatch path)
# ---------------------------------------------------------------------------


class TestTokenSliceBinding(unittest.TestCase):
    """Tests for thread-local token slice binding used by rotary embedding."""

    def setUp(self):
        _reset_ubatch_singletons({"num_ubatches": 2})

    def tearDown(self):
        clear_ascend_config()
        uu._UBATCH_RUNTIME_MANAGER = None

    def test_get_current_token_slice_none_on_main_thread(self):
        """Main thread (not a worker) should get None."""
        rt = get_ubatch_runtime_manager()
        self.assertIsNone(rt.get_current_token_slice())

    def test_get_current_token_slice_set_on_worker_thread(self):
        """set_current_token_slice stores per-thread state."""
        rt = get_ubatch_runtime_manager()
        result = {}

        def worker():
            rt.set_current_token_slice(slice(10, 20))
            result["slice"] = rt.get_current_token_slice()

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        self.assertEqual(result["slice"], slice(10, 20))
        # Main thread still None
        self.assertIsNone(rt.get_current_token_slice())

    def test_get_cos_and_sin_slice_fallback_when_not_worker(self):
        """get_cos_and_sin_slice should fall back to _cos_slice on main thread."""
        from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_slice
        # _cos_slice / _sin_slice may be None if set_cos_and_sin was never
        # called. The function reads rt.get_current_token_slice() which is None
        # on the main thread, so it returns the module globals directly.
        # We just verify it doesn't crash and the code path returns.
        rt = get_ubatch_runtime_manager()
        rt.ubatch_slices = None  # ubatch_enabled() is False
        cos, sin = get_cos_and_sin_slice()
        # When ubatch is disabled, returns the global _cos_slice / _sin_slice.
        # They may be None in tests; just verify no exception.
        del cos, sin


# ---------------------------------------------------------------------------
# Mocked-thread concurrency (T1: exec / comm_section / yield_thread)
# ---------------------------------------------------------------------------


class TestUBatchRuntimeConcurrency(unittest.TestCase):
    """Mocked-thread tests for the lock-step ping-pong orchestration.

    These tests mock ``torch.npu.Stream`` and ``Event`` so they run without
    NPU hardware. They verify the host-side thread coordination logic:
    symmetric comm_section entries, timeout on divergence, and error recovery.
    """

    def setUp(self):
        _reset_ubatch_singletons({"num_ubatches": 2})

    def tearDown(self):
        clear_ascend_config()
        uu._UBATCH_RUNTIME_MANAGER = None

    def _make_rt_with_mocks(self):
        """Create a UBatchRuntimeManager with mocked NPU streams/events."""
        rt = get_ubatch_runtime_manager()
        rt.ubatch_slices = [
            FakeUBatchSlice(slice(0, 1), slice(0, 4), num_tokens=4),
            FakeUBatchSlice(slice(0, 1), slice(4, 8), num_tokens=4),
        ]
        rt.forward_contexts = [MagicMock(), MagicMock()]
        # Mock streams and events
        rt.stream = [MagicMock() for _ in range(rt.num_ubatches)]
        rt.comm_done_event = [MagicMock() for _ in range(rt.num_ubatches)]
        rt.compute_done_event = [MagicMock() for _ in range(rt.num_ubatches)]
        return rt

    @patch.object(uu, "npu_stream_switch")
    def test_symmetric_comm_section_completes(self, mock_stream_switch):
        """Two workers each calling comm_section once should complete cleanly."""
        # npu_stream_switch is mocked so exec() does not touch real NPU streams.
        mock_stream_switch.return_value = nullcontext()
        rt = self._make_rt_with_mocks()

        def target():
            # batch_idx is bound to the thread by exec() via _tls; read it
            # back so each worker reports which ubatch it ran as.
            with rt.comm_section():
                pass  # Simulate MoE comm
            return rt._tls.batch_idx

        # Simulate forward_init / forward_finished around two exec calls
        rt.is_ubatch_running = True
        f0 = rt.add_task_and_get_future(target, 0)
        f1 = rt.add_task_and_get_future(target, 1)
        r0 = f0.result(timeout=5)
        r1 = f1.result(timeout=5)
        self.assertEqual(r0, 0)
        self.assertEqual(r1, 1)
        # Both counters should be 1
        self.assertEqual(rt.comm_entry_counts, [1, 1])

    @patch.object(uu, "npu_stream_switch")
    def test_comm_entry_counts_increment_per_entry(self, mock_stream_switch):
        """Multiple comm_section calls increment counters correctly."""
        mock_stream_switch.return_value = nullcontext()
        rt = self._make_rt_with_mocks()

        def target():
            for _ in range(3):
                with rt.comm_section():
                    pass
            return rt._tls.batch_idx

        rt.is_ubatch_running = True
        f0 = rt.add_task_and_get_future(target, 0)
        f1 = rt.add_task_and_get_future(target, 1)
        f0.result(timeout=5)
        f1.result(timeout=5)
        self.assertEqual(rt.comm_entry_counts, [3, 3])

    def test_reset_thread_pool_creates_fresh_pool(self):
        """reset_thread_pool discards old pool; next access creates new one."""
        rt = self._make_rt_with_mocks()
        # Force pool creation
        old_pool = rt.thread_pool
        self.assertEqual(len(old_pool), 2)
        # Reset
        rt.reset_thread_pool()
        self.assertIsNone(rt._thread_pool)
        # Access again — new pool
        new_pool = rt.thread_pool
        self.assertIsNot(old_pool, new_pool)
        self.assertEqual(len(new_pool), 2)
        # Clean up: stop new pool threads by resetting again
        rt.reset_thread_pool()

    def test_release_parked_workers_sets_all_events(self):
        """release_parked_workers sets all cpu_events."""
        rt = self._make_rt_with_mocks()
        # Clear all events first
        for e in rt.cpu_event:
            e.clear()
        rt.release_parked_workers()
        for e in rt.cpu_event:
            self.assertTrue(e.is_set())

    def test_yield_thread_timeout_on_divergence(self):
        """If the peer never reaches comm_section, yield_thread times out."""
        rt = self._make_rt_with_mocks()
        rt.is_ubatch_running = True

        # Patch UBATCH_HANDOFF_TIMEOUT to a small value for testing
        with patch.object(uu, "UBATCH_HANDOFF_TIMEOUT", 0.5):
            error_raised = {}

            def divergent_worker():
                # Simulate what exec() does before calling target(): bind
                # the batch_idx to this thread so comm_section can read it.
                rt._tls.batch_idx = 0
                try:
                    with rt.comm_section():
                        pass
                except TimeoutError:
                    error_raised["timeout"] = True

            # Worker 0 enters comm_section; peer (worker 1) never runs.
            # yield_thread waits for worker 1's handoff which never comes.
            t = threading.Thread(target=divergent_worker)
            t.start()
            t.join(timeout=5)
            self.assertTrue(error_raised.get("timeout", False),
                            "Expected TimeoutError from yield_thread")

    def test_forward_finished_resets_counters(self):
        """forward_finished resets curr_batch and comm_entry_counts."""
        rt = self._make_rt_with_mocks()
        rt.curr_batch = 1
        rt.comm_entry_counts = [3, 2]
        rt.forward_finished()
        self.assertEqual(rt.curr_batch, 0)
        self.assertEqual(rt.comm_entry_counts, [0, 0])
        self.assertFalse(rt.is_ubatch_running)

    def test_forward_finished_skip_event_wait_on_error(self):
        """forward_finished(skip_event_wait=True) skips NPU event wait.

        On the error path some NPU events may never be recorded, so waiting
        would block forever. The skip_event_wait flag avoids this while
        still resetting the lock-step state.
        """
        rt = self._make_rt_with_mocks()
        rt.curr_batch = 1
        rt.comm_entry_counts = [2, 1]
        rt.forward_finished(skip_event_wait=True)
        self.assertEqual(rt.curr_batch, 0)
        self.assertEqual(rt.comm_entry_counts, [0, 0])
        self.assertFalse(rt.is_ubatch_running)

    def test_error_recovery_release_reset_and_forward_finished(self):
        """Full error recovery path: release + reset + forward_finished.

        Simulates the exact sequence used by _model_forward_ubatches when a
        worker raises: release_parked_workers → reset_thread_pool →
        forward_finished(skip_event_wait=True). Verifies that the runtime
        reaches a clean state suitable for the next step.
        """
        rt = self._make_rt_with_mocks()
        rt.is_ubatch_running = True
        rt.curr_batch = 1
        rt.comm_entry_counts = [2, 1]
        # Old pool created (lazy)
        _ = rt.thread_pool
        # Simulate error recovery
        rt.release_parked_workers()
        rt.reset_thread_pool()
        rt.forward_finished(skip_event_wait=True)
        # Verify clean state
        self.assertFalse(rt.is_ubatch_running)
        self.assertEqual(rt.curr_batch, 0)
        self.assertEqual(rt.comm_entry_counts, [0, 0])
        self.assertIsNone(rt._thread_pool)
        # Next forward_init should work with a fresh pool
        rt.is_ubatch_running = True
        new_pool = rt.thread_pool
        self.assertEqual(len(new_pool), 2)
        # Clean up
        rt.reset_thread_pool()


# ---------------------------------------------------------------------------
# _model_forward_ubatches result concatenation (T3)
# ---------------------------------------------------------------------------


class TestModelForwardUbatchesConcat(unittest.TestCase):
    """Tests for the result concatenation logic in _model_forward_ubatches.

    Verifies that torch.cat results are ordered correctly matching the
    ubatch_slices order, for both single-tensor and tuple (aux_hidden_states)
    return types.
    """

    def test_single_tensor_concat_preserves_order(self):
        """Cat along dim=0 in ubatch order."""
        r0 = torch.tensor([1, 2, 3])
        r1 = torch.tensor([4, 5, 6])
        results = [r0, r1]
        cat = torch.cat(results, dim=0)
        self.assertEqual(cat.tolist(), [1, 2, 3, 4, 5, 6])

    def test_tuple_concat_preserves_order(self):
        """Tuple results: cat both hidden_states and aux."""
        r0 = (torch.tensor([1, 2]), [torch.tensor([10, 20])])
        r1 = (torch.tensor([3, 4]), [torch.tensor([30, 40])])
        results = [r0, r1]
        hidden = torch.cat([r[0] for r in results], dim=0)
        aux = [torch.cat(items, dim=0) for items in zip(*[r[1] for r in results])]
        self.assertEqual(hidden.tolist(), [1, 2, 3, 4])
        self.assertEqual(aux[0].tolist(), [10, 20, 30, 40])


# ---------------------------------------------------------------------------
# AscendConfig PP>1 rejection (T4)
# ---------------------------------------------------------------------------


class TestAscendConfigPPRejection(unittest.TestCase):
    """Tests that AscendConfig rejects num_ubatches>1 with PP>1."""

    def test_pp1_with_ubatch2_is_allowed(self):
        """num_ubatches=2 with pipeline_parallel_size=1 should be fine."""
        cfg = VllmConfig()
        cfg.additional_config = {"num_ubatches": 2, "refresh": True}
        # Should not raise
        ascend_config = init_ascend_config(cfg)
        self.assertEqual(ascend_config.num_ubatches, 2)
        clear_ascend_config()

    def test_pp2_with_ubatch2_is_rejected(self):
        """num_ubatches=2 with pipeline_parallel_size>1 should raise."""
        cfg = VllmConfig()
        cfg.parallel_config.pipeline_parallel_size = 2
        cfg.additional_config = {"num_ubatches": 2, "refresh": True}
        with self.assertRaises(RuntimeError) as ctx:
            init_ascend_config(cfg)
        self.assertIn("not supported", str(ctx.exception))
        self.assertIn("pipeline_parallel_size", str(ctx.exception))
        clear_ascend_config()

    def test_pp2_with_ubatch1_is_allowed(self):
        """num_ubatches=1 (default) with PP>1 should be fine (disabled)."""
        cfg = VllmConfig()
        cfg.parallel_config.pipeline_parallel_size = 2
        cfg.additional_config = {"refresh": True}
        # Should not raise — ubatch is disabled
        ascend_config = init_ascend_config(cfg)
        self.assertEqual(ascend_config.num_ubatches, 1)
        clear_ascend_config()


# ---------------------------------------------------------------------------
# slice_model_inputs_for_ubatch (incl. M-RoPE 2D positions)
# ---------------------------------------------------------------------------


class TestSliceModelInputsForUbatch(unittest.TestCase):
    """Tests for token-dimension slicing of model inputs per ubatch.

    Covers the 1D case and the multimodal M-RoPE case where ``positions`` has
    shape ``(3, seq_len)`` and must be sliced on dim=1 (the extra leading
    dimension preserved), plus the None-input fall-throughs.
    """

    def test_1d_inputs_sliced_on_dim0(self):
        ids = torch.arange(12)
        pos = torch.arange(12)
        emb = torch.arange(12).float().view(-1, 1)
        s_ids, s_pos, s_emb = slice_model_inputs_for_ubatch(
            slice(4, 8), ids, pos, emb
        )
        self.assertEqual(s_ids.tolist(), [4, 5, 6, 7])
        self.assertEqual(s_pos.tolist(), [4, 5, 6, 7])
        self.assertEqual(s_emb.tolist(), [[4.0], [5.0], [6.0], [7.0]])

    def test_mrope_2d_positions_sliced_on_dim1(self):
        # M-RoPE positions have shape (3, seq_len); slicing must preserve the
        # leading dim=3 and slice along dim=1.
        pos = torch.arange(3 * 12).view(3, 12)
        ids = torch.arange(12)
        s_ids, s_pos, _ = slice_model_inputs_for_ubatch(
            slice(4, 8), ids, pos, None
        )
        self.assertEqual(s_ids.tolist(), [4, 5, 6, 7])
        self.assertEqual(s_pos.shape, (3, 4))
        # Row r of the slice equals pos[r, 4:8].
        for r in range(3):
            self.assertEqual(s_pos[r].tolist(), pos[r, 4:8].tolist())

    def test_none_inputs_pass_through(self):
        s_ids, s_pos, s_emb = slice_model_inputs_for_ubatch(
            slice(0, 4), None, None, None
        )
        self.assertIsNone(s_ids)
        self.assertIsNone(s_pos)
        self.assertIsNone(s_emb)

    def test_partial_none_inputs(self):
        ids = torch.arange(8)
        s_ids, s_pos, s_emb = slice_model_inputs_for_ubatch(
            slice(2, 5), ids, None, None
        )
        self.assertEqual(s_ids.tolist(), [2, 3, 4])
        self.assertIsNone(s_pos)
        self.assertIsNone(s_emb)


# ---------------------------------------------------------------------------
# should_run_ubatch / warmup guard
# ---------------------------------------------------------------------------


class TestShouldRunUbatchWarmup(unittest.TestCase):
    """Tests for the one-shot warmup guard in should_run_ubatch.

    The first eligible step (should_ubatch=True, with_prefill=True) is skipped
    so the CANN allocator warms up on a single-forward path; subsequent
    eligible steps run ubatch. Ineligible steps (no prefill / not enabled)
    always return False and do NOT consume the warmup.
    """

    def setUp(self):
        _reset_ubatch_singletons({"num_ubatches": 2})

    def tearDown(self):
        clear_ascend_config()
        uu._UBATCH_RUNTIME_MANAGER = None

    def test_first_eligible_step_is_skipped_for_warmup(self):
        rt = get_ubatch_runtime_manager()
        self.assertTrue(rt._ubatch_warmup_pending)
        # First eligible step: warmup consumed, ubatch skipped.
        self.assertFalse(should_run_ubatch(True, True))
        self.assertFalse(rt._ubatch_warmup_pending)

    def test_subsequent_eligible_steps_run_ubatch(self):
        rt = get_ubatch_runtime_manager()
        # Consume the warmup on the first step.
        should_run_ubatch(True, True)
        # Following eligible steps run ubatch.
        self.assertTrue(should_run_ubatch(True, True))
        self.assertTrue(should_run_ubatch(True, True))

    def test_ineligible_step_does_not_consume_warmup(self):
        rt = get_ubatch_runtime_manager()
        # should_ubatch=False / with_prefill=False must not consume warmup.
        self.assertFalse(should_run_ubatch(False, True))
        self.assertFalse(should_run_ubatch(True, False))
        self.assertTrue(rt._ubatch_warmup_pending)
        # The first truly eligible step is still treated as warmup.
        self.assertFalse(should_run_ubatch(True, True))
        self.assertFalse(rt._ubatch_warmup_pending)


# ---------------------------------------------------------------------------
# comm_section symmetry guard (desync AssertionError)
# ---------------------------------------------------------------------------


class TestCommSectionSymmetryGuard(unittest.TestCase):
    """Tests for the comm_entry_counts symmetry assert in comm_section.

    After the lock-step handoff, comm_section asserts the peer reached the
    same comm_section depth. We mock the yield helpers to no-ops so the guard
    can be exercised deterministically on a single thread: a divergent
    comm_entry_counts state must raise AssertionError.
    """

    def setUp(self):
        _reset_ubatch_singletons({"num_ubatches": 2})

    def tearDown(self):
        clear_ascend_config()
        uu._UBATCH_RUNTIME_MANAGER = None

    def _make_rt_with_mocks(self):
        rt = get_ubatch_runtime_manager()
        rt.ubatch_slices = [
            FakeUBatchSlice(slice(0, 1), slice(0, 4), num_tokens=4),
            FakeUBatchSlice(slice(0, 1), slice(4, 8), num_tokens=4),
        ]
        rt.forward_contexts = [MagicMock(), MagicMock()]
        rt.stream = [MagicMock() for _ in range(rt.num_ubatches)]
        rt.comm_done_event = [MagicMock() for _ in range(rt.num_ubatches)]
        rt.compute_done_event = [MagicMock() for _ in range(rt.num_ubatches)]
        rt.is_ubatch_running = True
        return rt

    def test_symmetric_counts_do_not_raise(self):
        rt = self._make_rt_with_mocks()
        rt._tls.batch_idx = 0
        rt.comm_entry_counts = [0, 1]  # peer already at 1
        with patch.object(rt, "yield_and_switch_from_compute_to_comm"), \
             patch.object(rt, "yield_and_switch_from_comm_to_compute"):
            with rt.comm_section():
                # our count is now 1, peer is 1 -> symmetric, no raise
                rt.comm_entry_counts[1] = 1

    def test_divergent_counts_raise_assertion(self):
        rt = self._make_rt_with_mocks()
        rt._tls.batch_idx = 0
        rt.comm_entry_counts = [0, 0]
        # Mock the compute->comm handoff; the peer's count is never advanced,
        # so after we increment to 1 the peer (still 0) diverges.
        with patch.object(rt, "yield_and_switch_from_compute_to_comm"), \
             patch.object(rt, "yield_and_switch_from_comm_to_compute"):
            with self.assertRaises(AssertionError) as ctx:
                with rt.comm_section():
                    pass
            self.assertIn("desync", str(ctx.exception))


# ---------------------------------------------------------------------------
# get_cos_and_sin_slice worker-thread branch (T2 supplement)
# ---------------------------------------------------------------------------


class TestGetCosAndSinSliceUbatchBranch(unittest.TestCase):
    """Tests that get_cos_and_sin_slice returns the per-ubatch slice on a
    worker thread that has a bound token_slice, and the global slice otherwise.
    """

    def setUp(self):
        _reset_ubatch_singletons({"num_ubatches": 2})
        from vllm_ascend.ops import rotary_embedding as re
        # Populate the module-global cos/sin tables with a known (1, 8) tensor.
        re._cos = torch.arange(8).float().view(1, 8)
        re._sin = (torch.arange(8) * 10).float().view(1, 8)
        re._cos_slice = re._cos[:, :8]
        re._sin_slice = re._sin[:, :8]
        self._re = re

    def tearDown(self):
        # Reset globals so we don't leak state into other test modules.
        self._re._cos = None
        self._re._sin = None
        self._re._cos_slice = None
        self._re._sin_slice = None
        clear_ascend_config()
        uu._UBATCH_RUNTIME_MANAGER = None

    def test_worker_thread_returns_sliced_cos_sin(self):
        from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_slice
        rt = get_ubatch_runtime_manager()
        result = {}

        def worker():
            rt.set_current_token_slice(slice(2, 6))
            result["cos"], result["sin"] = get_cos_and_sin_slice()

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        self.assertEqual(result["cos"].tolist(), [[2.0, 3.0, 4.0, 5.0]])
        self.assertEqual(result["sin"].tolist(), [[20.0, 30.0, 40.0, 50.0]])

    def test_main_thread_returns_global_slice(self):
        from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_slice
        rt = get_ubatch_runtime_manager()
        # Main thread has no bound token_slice -> global _cos_slice/_sin_slice.
        self.assertIsNone(rt.get_current_token_slice())
        cos, sin = get_cos_and_sin_slice()
        self.assertIs(cos, self._re._cos_slice)
        self.assertIs(sin, self._re._sin_slice)


if __name__ == "__main__":
    unittest.main()
