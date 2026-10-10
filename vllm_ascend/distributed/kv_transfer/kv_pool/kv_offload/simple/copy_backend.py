"""DMA copy backend for NPU<->CPU block transfers.

Mirrors :class:`vllm.v1.simple_kv_offload.copy_backend.DmaCopyBackend`
but routes batched memcpy through ``torch.ops._C_ascend.swap_blocks_batch``
and uses ``torch.npu`` streams/events.
"""

from __future__ import annotations

import queue
import threading

import torch
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.simple.npu_mem_ops import (
    DIRECTION_D2H,
    DIRECTION_H2D,
    BatchMemcpyParams,
    build_params,
    copy_blocks,
)


class NPUDmaCopyBackend:
    """``aclrtMemcpyBatchAsync`` copy backend running on a worker thread.

    Two pre-built ``BatchMemcpyParams`` are cached (load=H2D, store=D2H).
    Submitted jobs are dispatched in FIFO order to a single worker
    thread; each job issues its copies on a dedicated NPU stream and
    records an Event the main thread can poll without synchronizing
    the device.
    """

    def __init__(self) -> None:
        self._store_params: BatchMemcpyParams | None = None
        self._load_params: BatchMemcpyParams | None = None
        self._load_stream: torch.npu.Stream | None = None
        self._store_stream: torch.npu.Stream | None = None
        self._device: torch.device | None = None
        self._queue: queue.SimpleQueue | None = None
        self._thread: threading.Thread | None = None
        self._shutdown: bool = False
        # Destination NPU blocks of loads whose copy raised. The worker drains
        # these so vLLM can apply ``kv_load_failure_policy`` to the affected
        # blocks instead of attending over KV that was never written.
        self._load_error_blocks: set[int] = set()
        self._load_error_lock = threading.Lock()
        # Distinguishes "the loop returned on the shutdown sentinel" from "the
        # thread is gone for another reason", which a bare join() cannot tell
        # apart once the thread has already exited.
        self._loop_exited_cleanly: bool = False

    def init(
        self,
        npu_caches: dict[str, torch.Tensor],
        cpu_caches: dict[str, torch.Tensor],
        device: torch.device,
        load_stream: torch.npu.Stream,
        store_stream: torch.npu.Stream,
    ) -> None:
        self._load_stream = load_stream
        self._store_stream = store_stream
        self._device = device
        # Stores go NPU->CPU (D2H), loads go CPU->NPU (H2D).
        self._store_params = build_params(npu_caches, cpu_caches, DIRECTION_D2H)
        self._load_params = build_params(cpu_caches, npu_caches, DIRECTION_H2D)

        self._queue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._copy_loop,
            name="npu-kv-offload-copy",
            daemon=True,
        )
        self._thread.start()

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.npu.Event]],
        wait_event: torch.npu.Event | None = None,
    ) -> None:
        params = self._store_params if is_store else self._load_params
        assert params is not None and self._queue is not None
        self._queue.put((src_blocks, dst_blocks, params, is_store, event_idx, events_list, wait_event))

    def drain_load_errors(self) -> set[int]:
        """Return and clear the NPU blocks whose load copy failed."""
        with self._load_error_lock:
            failed = self._load_error_blocks
            self._load_error_blocks = set()
        return failed

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        if self._queue is not None:
            self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("NPU KV-offload copy thread did not exit within 5s of shutdown.")
            elif not self._loop_exited_cleanly:
                logger.warning(
                    "NPU KV-offload copy thread was already gone at shutdown; "
                    "queued transfers after its exit were never executed."
                )

    # ------------------------------------------------------------------
    # Worker thread main loop
    # ------------------------------------------------------------------
    def _copy_loop(self) -> None:
        # Store jobs carry a compute-done event and wait on it before reading
        # live NPU KV-cache blocks. Loads read stable pinned host memory and
        # can be submitted immediately, matching the upstream DMA backend's
        # ordering model.
        assert self._device is not None
        assert self._queue is not None
        assert self._load_stream is not None
        assert self._store_stream is not None
        torch.npu.set_device(self._device)

        while True:
            item = self._queue.get()
            if item is None:
                self._loop_exited_cleanly = True
                return
            # One failed job must not take the thread down with it: the
            # completion protocol is an ordered event list plus a high-water
            # mark, so an unraised job would strand every later transfer behind
            # its missing event, and nothing supervises or restarts this thread.
            try:
                self._run_job(item)
            except Exception:
                logger.exception("NPU KV-offload copy job failed; reporting it and continuing")

    def _run_job(self, item: tuple) -> None:
        (
            src_blocks,
            dst_blocks,
            params,
            is_store,
            event_idx,
            events_list,
            wait_event,
        ) = item

        stream = self._store_stream if is_store else self._load_stream
        # init() set both streams before the thread that runs this was started.
        assert stream is not None
        try:
            with torch.npu.stream(stream):
                if wait_event is not None:
                    stream.wait_event(wait_event)
                copy_blocks(src_blocks, dst_blocks, params)
        except Exception:
            if is_store:
                # Nothing in HBM is wrong: the blocks simply never reached the
                # host, so a later lookup misses and the tokens are recomputed.
                logger.exception("NPU KV-offload store failed for %d block(s); they stay uncached", len(src_blocks))
            else:
                # The destination blocks were left unwritten. vLLM requires the
                # request to still be reported as finished receiving, with the
                # failed blocks surfaced through get_block_ids_with_load_errors
                # no later than that same pass, so record them here.
                logger.exception(
                    "NPU KV-offload load failed for %d block(s); reporting them as load errors",
                    len(dst_blocks),
                )
                with self._load_error_lock:
                    self._load_error_blocks.update(dst_blocks)
        # Recorded even on failure, so the watermark advances and the requests
        # behind this index are not stranded. With the copy abandoned the stream
        # holds no work from this job, so the event completes immediately.
        event = torch.npu.Event()
        event.record(stream)
        events_list.append((event_idx, event))
