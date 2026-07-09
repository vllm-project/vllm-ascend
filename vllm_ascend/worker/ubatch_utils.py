#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from vllm-project/vllm/vllm/worker/gpu_model_runner.py
#

from concurrent.futures import Future
from contextlib import contextmanager
import threading
import queue

import torch
from vllm import forward_context as global_forward_context
from vllm.v1.worker.ubatch_utils import UBatchSlice, is_last_ubatch_empty
from vllm.logger import logger
from vllm.distributed.parallel_state import is_local_first_rank

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.utils import npu_stream_switch
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

# Maximum time (seconds) to wait for the peer ubatch worker to reach the next
# lock-step handoff point. If exceeded, the ping-pong has desynchronized
# (e.g. the two forwards executed a different number of MoE layers) and
# continuing would deadlock. Raise a clear error instead of hanging forever.
UBATCH_HANDOFF_TIMEOUT = 120

# Rate-limit the ubatch-enablement log to every N enabled steps to avoid
# flooding the log with identical-rate lines on high-frequency prefill steps.
UBATCH_LOG_INTERVAL = 20


def slice_query_start_locs(
    query_start_loc: torch.Tensor,
    request_slice: slice,
) -> torch.Tensor:
    return (
        query_start_loc[request_slice.start : request_slice.stop + 1]
        - query_start_loc[request_slice.start]
    )


def _make_ascend_common_metadata_with_slice(
    ubatch_slice: UBatchSlice,
    common_attn_metadata: AscendCommonAttentionMetadata,
    is_first_chunk: bool = True,
) -> AscendCommonAttentionMetadata:
    assert not ubatch_slice.is_empty(), f"Ubatch slice {ubatch_slice} is empty"

    request_slice = ubatch_slice.request_slice
    token_slice = ubatch_slice.token_slice

    start_locs = common_attn_metadata.query_start_loc_cpu
    first_req = request_slice.start
    first_tok = token_slice.start
    last_req = request_slice.stop - 1
    last_tok = token_slice.stop - 1

    assert start_locs[first_req] <= first_tok < start_locs[first_req + 1], (
        "Token slice start outside of first request"
    )

    # Determine if requests are split across ubatches
    # splits_first_request: The first request in this slice is the continuation of
    #                       a request that started in a previous slice.
    # splits_last_request:  The last request in this slice continues into the
    #                       next slice.
    splits_first_request = first_tok > start_locs[first_req]
    splits_last_request = last_tok < start_locs[last_req + 1] - 1

    query_start_loc_cpu = slice_query_start_locs(start_locs, request_slice)
    query_start_loc = slice_query_start_locs(
        common_attn_metadata.query_start_loc, request_slice
    )

    assert len(query_start_loc) >= 2, (
        f"query_start_loc must have at least 2 elements, got {len(query_start_loc)}"
    )

    if splits_first_request:
        tokens_skipped = first_tok - start_locs[first_req]
        query_start_loc[1:] -= tokens_skipped
        query_start_loc_cpu[1:] -= tokens_skipped

    seq_lens = common_attn_metadata.seq_lens[request_slice]
    seq_lens_cpu = common_attn_metadata.seq_lens_cpu[request_slice]

    if splits_last_request:
        tokens_skipped = start_locs[last_req + 1] - token_slice.stop
        query_start_loc[-1] -= tokens_skipped
        query_start_loc_cpu[-1] -= tokens_skipped

        # Clone to avoid modifying original tensors (not cudagraph compatible)
        seq_lens = seq_lens.clone()
        seq_lens_cpu = seq_lens_cpu.clone()
        seq_lens[-1] -= tokens_skipped
        seq_lens_cpu[-1] -= tokens_skipped

    max_query_len = int(
        (query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]).max().item()
    )
    max_seq_len = int(seq_lens_cpu.max().item())

    # Handle positions tensor
    positions = common_attn_metadata.positions
    if positions is not None:
        positions = positions[token_slice]

    # Handle attn_state - for chunked prefill, second chunk needs special state
    attn_state = common_attn_metadata.attn_state
    from vllm_ascend.attention.attention_v1 import AscendAttentionState

    if attn_state in (
        AscendAttentionState.PrefillNoCache,
        AscendAttentionState.ChunkedPrefill,
    ) and not is_first_chunk:
        attn_state = AscendAttentionState.ChunkedPrefill

    # Handle num_computed_tokens_cpu
    num_computed_tokens_cpu = None
    if common_attn_metadata.num_computed_tokens_cpu is not None:
        num_computed_tokens_cpu = common_attn_metadata.num_computed_tokens_cpu[request_slice]

    num_requests = request_slice.stop - request_slice.start
    num_actual_tokens = token_slice.stop - token_slice.start

    # Handle max_query_len edge case for dummy runs
    if max_query_len == 0:
        max_query_len = common_attn_metadata.max_query_len

    return AscendCommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=num_requests,
        num_actual_tokens=num_actual_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=common_attn_metadata.block_table_tensor[request_slice],
        slot_mapping=common_attn_metadata.slot_mapping[token_slice],
        causal=common_attn_metadata.causal,
        num_input_tokens=token_slice.stop - token_slice.start,
        actual_seq_lengths_q=query_start_loc_cpu[1:].tolist(),
        positions=positions,
        attn_state=attn_state,
        decode_token_per_req=common_attn_metadata.decode_token_per_req,
        prefill_context_parallel_metadata=None,  # PCP needs special handling
    )


def split_ascend_common_metadata(
    ubatch_slices: list[UBatchSlice],
    common_attn_metadata: "AscendCommonAttentionMetadata",
) -> list["AscendCommonAttentionMetadata"]:
    results = []
    for i, ubatch_slice in enumerate(ubatch_slices):
        is_first_chunk = (i == 0)
        results.append(
            _make_ascend_common_metadata_with_slice(
                ubatch_slice, common_attn_metadata, is_first_chunk
            )
        )
    return results


def slice_model_inputs_for_ubatch(
    token_slice: slice,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    # Slice model inputs for a single ubatch along the token dimension.
    # Mirrors upstream vllm.v1.worker.gpu_ubatch_wrapper._slice_model_inputs.
    # Handles M-RoPE positions whose shape is (3, seq_len) — the extra
    # leading dimension must be preserved by slicing on dim=1.
    sliced_input_ids = input_ids[token_slice] if input_ids is not None else None
    if positions is not None:
        if positions.ndim == 2:
            sliced_positions = positions[:, token_slice]
        else:
            sliced_positions = positions[token_slice]
    else:
        sliced_positions = None
    sliced_inputs_embeds = (
        inputs_embeds[token_slice] if inputs_embeds is not None else None
    )
    return sliced_input_ids, sliced_positions, sliced_inputs_embeds


class UBatchRuntimeManager:
    def __init__(self):
        self.num_ubatches = get_ascend_config().num_ubatches
        self.is_ubatch_running = False
        self.curr_batch = 0
        self.ubatch_slices = None
        self.forward_contexts = None
        self.stream: list[torch.npu.Stream] | None = None
        self.compute_done_event: list[torch.npu.Event] | None = None
        self.comm_done_event: list[torch.npu.Event] | None = None
        self.cpu_event = [threading.Event() for _ in range(self.num_ubatches)]
        self.count_enabled = 0
        self.count_disabled = 0
        self.num_tokens_enabled = 0
        self.num_tokens_disabled = 0
        # Warmup guard: skip ubatch overlap on the FIRST step that would
        # otherwise enable it, so the CANN caching allocator and any
        # lazy-initialized buffers get established via a single-forward
        # path before two concurrent forwards contend for HBM. Once the
        # warmup step completes this flips to False and never resets.
        self._ubatch_warmup_pending = True
        # Log the ubatch-enablement rate at most every UBATCH_LOG_INTERVAL
        # enabled steps instead of on every single one.
        self.log_interval = UBATCH_LOG_INTERVAL
        self._steps_since_log = 0
        # Worker thread pool is created lazily on first use. This keeps module
        # import cheap (no daemon threads) and avoids creating threads when the
        # feature is disabled (num_ubatches == 1).
        self._thread_pool: list["UBatchRuntimeManager._PersistentThread"] | None = None
        # Per-worker-thread token slice. Each _PersistentThread runs on its own
        # OS thread, so a threading.local gives each worker a private slot
        # holding the token_slice of the ubatch it is currently executing. This
        # is the correct source of truth for ops (e.g. rotary embedding) that
        # must index per-ubatch tables: unlike the shared ``curr_batch`` cursor,
        # it cannot be mutated by the peer worker thread mid-forward.
        # Filled by exec() on entry and read via get_current_token_slice().
        self._tls = threading.local()
        # Per-ubatch comm_section entry counters. Used by the symmetry guard
        # in comm_section: after a lock-step handoff, we verify the peer has
        # reached the same comm_section depth. Model-level divergence (where
        # one forward executes fewer MoE layers) is primarily caught by the
        # timeout in yield_thread(); this counter additionally catches runtime
        # bugs where switch_curr_batch is called outside comm_section.
        self.comm_entry_counts = [0] * self.num_ubatches

    def consume_warmup_if_pending(self) -> bool:
        if self._ubatch_warmup_pending:
            self._ubatch_warmup_pending = False
            return True
        return False

    class _PersistentThread(threading.Thread):
        def __init__(self, rt, batch_idx):
            super().__init__(daemon=True)
            self.task_queue = queue.Queue()
            self.rt = rt
            self.batch_idx = batch_idx

        def add_task(self, target, *args, **kwargs):
            future: Future = Future()
            self.task_queue.put((target, future, args, kwargs))
            return future

        def run(self) -> None:
            while True:
                task = self.task_queue.get()
                target, future, args, kwargs = task
                try:
                    output = self.rt.exec(target, self.batch_idx, *args, **kwargs)
                except Exception as exc:
                    # Surface failures on the Future so callers see them instead
                    # of hanging forever on future.result().
                    logger.exception(
                        "ubatch worker thread %s failed", self.batch_idx
                    )
                    future.set_exception(exc)
                    continue
                future.set_result(output)

    @property
    def thread_pool(self) -> list["UBatchRuntimeManager._PersistentThread"]:
        if self._thread_pool is None:
            self._thread_pool = [
                UBatchRuntimeManager._PersistentThread(self, ubid)
                for ubid in range(self.num_ubatches)
            ]
            for thread in self._thread_pool:
                thread.start()
        return self._thread_pool

    @torch.inference_mode()
    def exec(self, target, batch_idx, *args, **kwargs):
        if batch_idx != 0:
            if not self.cpu_event[batch_idx].wait(timeout=UBATCH_HANDOFF_TIMEOUT):
                raise TimeoutError(
                    f"ubatch worker {batch_idx} timed out after "
                    f"{UBATCH_HANDOFF_TIMEOUT}s waiting for initial handoff "
                    "from worker 0."
                )
            self.cpu_event[batch_idx].clear()
        assert batch_idx == self.curr_batch
        self._tls.batch_idx = batch_idx
        self.set_current_token_slice(self.ubatch_slices[batch_idx].token_slice)
        with npu_stream_switch(self.stream[batch_idx]):
            self.compute_begin()
            try:
                ret = target(*args, **kwargs)
            finally:
                self.compute_end()
                self.switch_curr_batch()
                # Clear the thread-local bindings so stale state is never
                # observed by a subsequent non-ubatch task on the same worker.
                self.set_current_token_slice(None)
                self._tls.batch_idx = None
        return ret

    def add_task_and_get_future(self, target, batch_idx, *args, **kwargs):
        return self.thread_pool[batch_idx].add_task(target, *args, **kwargs)

    def set_current_token_slice(self, token_slice: slice) -> None:
        """Bind the executing ubatch's token slice to the current thread.

        Called by exec() on entry so that custom ops running inside the
        forward pass can read the correct per-ubatch slice via
        get_current_token_slice() instead of the shared curr_batch cursor.
        """
        self._tls.token_slice = token_slice

    def get_current_token_slice(self) -> slice | None:
        """Return the token slice bound to the current thread, or None.

        Returns None when ubatch is disabled or the caller is not on a ubatch
        worker thread, so callers can fall back to the whole-table slice.
        """
        return getattr(self._tls, "token_slice", None)

    def compute_begin(self):
        curr_batch = self.curr_batch
        last_batch = (curr_batch - 1 + self.num_ubatches) % self.num_ubatches
        self.stream[curr_batch].wait_event(self.compute_done_event[last_batch])

    def compute_end(self):
        curr_batch = self.curr_batch
        self.compute_done_event[curr_batch].record(self.stream[curr_batch])

    def comm_begin(self):
        curr_batch = self.curr_batch
        last_batch = (curr_batch - 1 + self.num_ubatches) % self.num_ubatches
        self.stream[curr_batch].wait_event(self.comm_done_event[last_batch])

    def comm_end(self):
        curr_batch = self.curr_batch
        self.comm_done_event[curr_batch].record(self.stream[curr_batch])

    def switch_curr_batch(self):
        self.curr_batch = (self.curr_batch + 1) % self.num_ubatches
        global_forward_context._forward_context = self.forward_contexts[self.curr_batch]
        self.cpu_event[self.curr_batch].set()

    def yield_thread(self):
        curr_batch = self.curr_batch
        self.switch_curr_batch()
        if not self.cpu_event[curr_batch].wait(timeout=UBATCH_HANDOFF_TIMEOUT):
            raise TimeoutError(
                f"ubatch worker {curr_batch} timed out after "
                f"{UBATCH_HANDOFF_TIMEOUT}s waiting for peer handoff. "
                "The two ubatch forwards likely diverged (executed a "
                "different number of comm_section entries)."
            )
        self.cpu_event[curr_batch].clear()

    def yield_and_switch_from_compute_to_comm(self):
        self.compute_end()
        self.yield_thread()
        self.comm_begin()

    def yield_and_switch_from_comm_to_compute(self):
        self.comm_end()
        self.yield_thread()
        self.compute_begin()

    @contextmanager
    def comm_section(self):
        """Bracket a MoE collective-communication block.

        While ubatch overlap is running, this yields the current thread to the
        peer ubatch worker (so the peer resumes its dense compute on its own
        NPU stream while our comm runs on ours) on enter, and switches back to
        compute on exit. When ubatch is disabled it is a no-op, so callers can
        wrap any MoE comm call unconditionally instead of repeating the
        ``if rt.is_ubatch_running`` guard on both sides.
        """
        if not self.is_ubatch_running:
            yield
            return
        # Symmetry guard: the lock-step handoff in yield_thread() requires
        # every ubatch worker to enter comm_section the same number of times
        # in the same order. If two forwards diverge (e.g. one skips a MoE
        # layer), yield_thread()'s timeout will catch the deadlock and raise
        # a TimeoutError. Additionally, count entries per ubatch and assert
        # the peer's count matches ours after the handoff — this catches
        # runtime bugs (e.g. switch_curr_batch called outside comm_section)
        # that could desynchronize the ping-pong without blocking.
        my_idx = self._tls.batch_idx
        self.comm_entry_counts[my_idx] += 1
        my_count = self.comm_entry_counts[my_idx]
        self.yield_and_switch_from_compute_to_comm()
        # In a 2-worker setup the peer is always (my_idx + 1) % num_ubatches.
        # Verify it reached the same comm_section depth as us.
        peer_idx = (my_idx + 1) % self.num_ubatches
        assert self.comm_entry_counts[peer_idx] == my_count, (
            f"ubatch comm_section desync: worker {my_idx} entered #{my_count} "
            f"but peer {peer_idx} is at #{self.comm_entry_counts[peer_idx]}. "
            "Both ubatches must call comm_section the same number of times."
        )
        try:
            yield
        finally:
            self.yield_and_switch_from_comm_to_compute()

    def log(self):
        # Guard the denominators: callers increment count_enabled before
        # invoking forward_init()->log(), but do not rely on that implicit
        # ordering. max(1, ...) keeps this robust to reordering.
        total_steps = max(1, self.count_disabled + self.count_enabled)
        total_tokens = max(1, self.num_tokens_disabled + self.num_tokens_enabled)
        ubatch_rate = self.count_enabled / total_steps
        tokens_ubatch_rate = self.num_tokens_enabled / total_tokens
        logger.info(f'running ubatch, ubatch_rate: {ubatch_rate * 100}%, tokens_ubatch_rate: {tokens_ubatch_rate * 100}%')

    def forward_init(self):
        self.is_ubatch_running = True
        if is_local_first_rank():
            # Sample the rate log rather than emitting it on every enabled
            # step; high-frequency prefill otherwise floods the log.
            if self._steps_since_log >= self.log_interval:
                self.log()
                self._steps_since_log = 0
            self._steps_since_log += 1
        stream = torch.npu.current_stream()
        if self.stream is None:
            self.stream = [torch.npu.Stream() for _ in range(self.num_ubatches)]
            self.comm_done_event = [torch.npu.Event() for _ in range(self.num_ubatches)]
            self.compute_done_event = [torch.npu.Event() for _ in range(self.num_ubatches)]

        for i in range(self.num_ubatches):
            self.comm_done_event[i].record(stream)
            self.compute_done_event[i].record(stream)
            self.cpu_event[i].clear()

        global_forward_context._forward_context = self.forward_contexts[self.curr_batch]

    def forward_finished(self, *, skip_event_wait: bool = False):
        self.is_ubatch_running = False
        if not skip_event_wait:
            stream = torch.npu.current_stream()
            for i in range(self.num_ubatches):
                self.comm_done_event[i].wait(stream)
                self.compute_done_event[i].wait(stream)
        # Reset lock-step state so the next ubatch step starts clean.
        # On the error path (skip_event_wait=True), the caller has already
        # called release_parked_workers() and reset_thread_pool(), so the old
        # pool is discarded and some NPU events may never be recorded. Waiting
        # on them would block forever, so we skip the wait.
        # On the success path, all Futures have been drained and all workers
        # have exited exec(), so the event wait completes immediately.
        self.curr_batch = 0
        self.comm_entry_counts = [0] * self.num_ubatches

    def ubatch_enabled(self):
        return self.ubatch_slices is not None

    def release_parked_workers(self):
        """Set all cpu_events to release workers blocked in yield_thread().

        Called by the error path in _model_forward_ubatches so that a peer
        worker parked in cpu_event.wait() can wake up and exit its task.
        """
        for i in range(self.num_ubatches):
            self.cpu_event[i].set()

    def reset_thread_pool(self):
        """Destroy and recreate the worker thread pool.

        Called on the error path after a worker task raised. The failing
        worker's exec() finally already advanced curr_batch and the peer may
        be in an inconsistent state (e.g. still inside comm_section). Rather
        than trying to drain Futures with a timeout (which blocks the error
        propagation path for up to 120s), we simply discard the old pool and
        create a fresh one.

        Thread lifecycle of the discarded workers:
          - Each old daemon thread finishes its current task or hits the
            yield_thread timeout (which raises TimeoutError in exec(), caught
            by _PersistentThread.run which logs and continues).
          - It then blocks forever on task_queue.get() since no new tasks will
            be submitted to the old pool. As a daemon thread it will not
            prevent process exit, but it remains alive, consuming one OS
            thread slot. In a long-running service, repeated error-path
            triggers will accumulate leaked threads. This is accepted as a
            trade-off: the alternative (draining with timeout) would block
            the main thread for 120s on every error, which is worse for
            availability. The leak only occurs on the error path, not the
            normal hot path.
        """
        self._thread_pool = None


# Process-wide mutable singleton holding the ubatch runtime state (streams,
# events, daemon worker-thread pool). This is intentional: ubatch overlap is
# inherently a per-process resource (one set of NPU streams per device) and
# cannot be passed through the call stack without threading it through every
# custom op. It is created lazily so that simply importing this module does not
# spawn daemon worker threads, keeping imports cheap and making the module
# testable in isolation. Once created it is read-only from the hot path; the
# only mutation points are forward_init/forward_finished (single-threaded,
# called from the model runner) and the worker threads (each owns a distinct
# batch_idx slot).
_UBATCH_RUNTIME_MANAGER: UBatchRuntimeManager | None = None


def get_ubatch_runtime_manager() -> UBatchRuntimeManager:
    global _UBATCH_RUNTIME_MANAGER
    if _UBATCH_RUNTIME_MANAGER is None:
        _UBATCH_RUNTIME_MANAGER = UBatchRuntimeManager()
    return _UBATCH_RUNTIME_MANAGER


def should_enable_ubatch(num_tokens, num_tokens_padded):
    if not maybe_use_ubatch() or num_tokens_padded <= get_ubatch_trigger_threshold():
        return False
    return not is_last_ubatch_empty(num_tokens, num_tokens_padded, get_ubatch_runtime_manager().num_ubatches)


def should_run_ubatch(should_ubatch: bool, with_prefill: bool) -> bool:
    if not (should_ubatch and with_prefill):
        return False
    return not get_ubatch_runtime_manager().consume_warmup_if_pending()


def maybe_use_ubatch():
    return get_ubatch_runtime_manager().num_ubatches > 1


def get_ubatch_trigger_threshold():
    return get_ascend_config().ubatch_trigger_threshold


def resolve_num_tokens_across_dp(
    num_tokens_across_dp: torch.Tensor | None,
    num_tokens_padded: int,
) -> torch.Tensor:
    """Return a non-None per-DP-rank token-count tensor for ubatch build.

    ``NPUModelRunner._determine_batch_execution_and_padding`` only fills
    ``num_tokens_across_dp`` when ``data_parallel_size > 1``; in DP==1 it stays
    ``None`` because ``_sync_batch_across_dp`` short-circuits. ubatch overlap is
    still valid for a single DP rank (it overlaps TP/EP MoE comm with compute on
    local NPU streams), so synthesize a 1-element CPU tensor carrying the single
    rank's padded token count to keep the per-ubatch forward-context
    construction loop uniform and avoid a TypeError on ``len(None)``.
    """
    if num_tokens_across_dp is None:
        return torch.tensor([num_tokens_padded], device="cpu", dtype=torch.int32)
    return num_tokens_across_dp
