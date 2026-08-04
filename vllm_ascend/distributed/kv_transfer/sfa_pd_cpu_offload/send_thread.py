# SPDX-License-Identifier: Apache-2.0
"""P-side memfabric pull sending thread for SFA PD CPU offload."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any

import msgspec
import torch
import zmq
from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_path

from vllm_ascend import envs
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.protocol import (
    MF_META,
    READ_DONE,
    READ_FAILED,
    READ_READY_BATCH,
    LayerMetadata,
    SendTask,
    get_external_request_id,
)

THREAD_SHUTDOWN_TIMEOUT_SECONDS = 5.0


@dataclass
class ProducerSendState:
    last_layer_idx: int
    layer_metadata: dict[str, LayerMetadata]
    p_session: str
    main_group_idx: int
    indexer_group_idx: int
    block_sizes: tuple[int, ...]
    layer_storage_slots: dict[int, tuple[int, ...]]


class MembPullSendingThread(threading.Thread):
    """P-side sending thread for memfabric pull mode.

    Does NOT push (no batch_transfer_sync_write). Instead, after each layer's
    KV is ready in P HBM, notifies D via READ_READY_BATCH (ZMQ), and drains
    READ_DONE / READ_FAILED replies. First call sends MF_META (P session +
    layer addresses) to D.
    """

    def __init__(
        self,
        *,
        ready_event: threading.Event,
        state: ProducerSendState,
    ) -> None:
        super().__init__(daemon=True, name="SfaPDMembPullSendingThread")
        self.timeout = 10.0
        self._mf_meta_sent_paths: set[str] = set()
        self._state = state
        self.last_layer_idx = state.last_layer_idx
        self.ready_event = ready_event
        self.send_queue: queue.Queue[SendTask] = queue.Queue()
        self._persist_ctx = zmq.Context()
        self._dealers: dict[str, Any] = {}
        self._stopped = False
        self.startup_error: BaseException | None = None
        self._pending_reads_by_layer: dict[int, int] = {}
        num_storage_slots = max(
            (slot for slots in state.layer_storage_slots.values() for slot in slots),
            default=-1,
        ) + 1
        self.storage_send_done_events: list[threading.Event] = []
        for _ in range(num_storage_slots):
            event = threading.Event()
            event.set()
            self.storage_send_done_events.append(event)
        self._storage_read_errors: dict[int, str] = {}
        # Per-layer fresh compute-stream events recorded by the producer in
        # save_kv_layer right after KV scatter.
        self._p_save_events: dict[int, Any] = {}

    def _ensure_dealer(self, path: str):
        if path not in self._dealers:
            dealer = self._persist_ctx.socket(zmq.DEALER)
            dealer.setsockopt(zmq.LINGER, 0)
            dealer.setsockopt(zmq.SNDHWM, 0)
            dealer.setsockopt(zmq.RCVHWM, 0)
            dealer.connect(path)
            self._dealers[path] = dealer
        return self._dealers[path]

    def run(self) -> None:
        try:
            from vllm.distributed import get_world_group

            local_rank = get_world_group().local_rank
            torch.npu.set_device(torch.device(f"npu:{local_rank}"))
        except BaseException as error:
            self.startup_error = error
            logger.error("MembPull send thread failed to initialize its NPU device: %s", error)
            self.ready_event.set()
            self._persist_ctx.destroy(linger=0)
            return
        self.ready_event.set()

        encoder = msgspec.msgpack.Encoder()
        decoder = msgspec.msgpack.Decoder(type=tuple)
        thread_error = None
        try:
            while not self._stopped:
                try:
                    send_task = self.send_queue.get(timeout=0.001)
                    try:
                        self._process_send_task(send_task, encoder)
                    except Exception as e:
                        layer_idx = getattr(send_task, "layer_idx", -1)
                        logger.error(
                            "MembPull send task failed (layer=%s): %s: %s",
                            layer_idx,
                            type(e).__name__,
                            e,
                        )
                        self._fail_layer(layer_idx, str(e))
                except queue.Empty:
                    pass
                try:
                    if self._dealers:
                        self._drain_read_replies(decoder)
                except Exception as e:
                    logger.error("MembPull read-reply drain error: %s: %s", type(e).__name__, e)
        except BaseException as error:
            thread_error = error
            logger.error("MembPull send thread crashed: %s: %s", type(error).__name__, error)
        finally:
            pending_error = (
                str(thread_error)
                if thread_error is not None
                else "send thread stopped before D completed all pending reads"
            )
            for layer_idx in list(self._pending_reads_by_layer):
                self._record_layer_error(layer_idx, pending_error)
                self._pending_reads_by_layer.pop(layer_idx, None)
                self._signal_layer_done(layer_idx)
            if self._dealers:
                try:
                    self._drain_read_replies(decoder)
                except Exception as error:
                    logger.warning("MembPull final read-reply drain failed: %s", error)
                for dealer in self._dealers.values():
                    dealer.close(linger=0)
                self._dealers.clear()
            self._persist_ctx.destroy(linger=0)

    def stop(self, timeout: float = THREAD_SHUTDOWN_TIMEOUT_SECONDS) -> None:
        self._stopped = True
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=timeout)
        if self.is_alive():
            logger.warning("MembPull send thread did not stop within %.1f seconds", timeout)

    def record_p_save_event(self, layer_idx: int) -> None:
        evt = torch.npu.Event()
        evt.record()
        self._p_save_events[layer_idx] = evt

    def mark_layer_pending(self, layer_idx: int) -> None:
        """Close all gates protecting storage written by ``layer_idx``."""
        for slot_id in self._state.layer_storage_slots.get(layer_idx, ()):
            self.storage_send_done_events[slot_id].clear()
            self._storage_read_errors.pop(slot_id, None)

    def _process_send_task(self, send_task: SendTask, encoder: msgspec.msgpack.Encoder) -> None:
        layer_idx = send_task.layer_idx
        p_save_event = self._p_save_events.pop(layer_idx, None)
        if p_save_event is not None:
            p_save_event.synchronize()
        elif send_task.wait_event is not None:
            send_task.wait_event.synchronize()
        layer_name = send_task.layer_name

        # Group this layer's notifications by D-side endpoint so requests for
        # the same ``(remote_host, remote_port)`` share one READ_READY_BATCH.
        # Each value is ``(read_reqs, done_ext_ids)``; a read request contains
        # ``(external_req_id, main_block_ids, indexer_block_ids,
        # main_start_block, indexer_start_block)``.
        endpoint_payloads: dict[
            tuple[str, int],
            tuple[list[tuple[str, list[int], list[int], int, int]], list[str]],
        ] = {}
        layer_meta = self._state.layer_metadata[layer_name]
        layer_has_indexer = layer_meta.has_indexer

        def _blocks_for_chunk(rm, group_idx: int) -> tuple[list[int], int]:
            all_block_ids = (
                rm.local_block_ids[group_idx]
                if len(rm.local_block_ids) > group_idx
                else []
            )
            block_size = self._state.block_sizes[group_idx]
            transferred_tokens = max(
                int(getattr(rm, "remote_cache_tokens", 0) or 0),
                int(rm.local_transed_tokens or 0),
                0,
            )
            start_block = transferred_tokens // block_size
            computed_tokens = max(int(rm.local_computed_tokens or 0), 0)
            if rm.chunk_finish:
                end_block = (computed_tokens + block_size - 1) // block_size
            else:
                end_block = computed_tokens // block_size
            if end_block > len(all_block_ids):
                raise RuntimeError(
                    f"MembPull P chunk block range exceeds allocation for group {group_idx}: "
                    f"range=[{start_block}, {end_block}), allocated={len(all_block_ids)}"
                )
            end_block = max(end_block, start_block)
            return list(all_block_ids[start_block:end_block]), start_block

        for req_id, rm in send_task.send_request.items():
            p_main_block_ids, main_start_block = _blocks_for_chunk(
                rm, self._state.main_group_idx
            )
            if layer_has_indexer:
                p_indexer_block_ids, indexer_start_block = _blocks_for_chunk(
                    rm, self._state.indexer_group_idx
                )
            else:
                p_indexer_block_ids, indexer_start_block = [], 0
            ext_id = get_external_request_id(req_id)
            has_endpoint = bool(rm.remote_host) and bool(rm.remote_port)
            chunk_done = layer_idx == self.last_layer_idx and rm.chunk_finish and has_endpoint
            if (p_main_block_ids or p_indexer_block_ids or chunk_done) and has_endpoint:
                endpoint = (rm.remote_host, rm.remote_port)
                read_reqs, done_ext_ids = endpoint_payloads.setdefault(endpoint, ([], []))
                if p_main_block_ids or p_indexer_block_ids:
                    read_reqs.append(
                        (
                            ext_id,
                            p_main_block_ids,
                            p_indexer_block_ids,
                            main_start_block,
                            indexer_start_block,
                        )
                    )
                if chunk_done:
                    done_ext_ids.append(ext_id)
            if envs.VLLM_ASCEND_SFA_DEBUG:
                logger.info(
                    "MembPull P add READ_READY_BATCH item: layer=%d (%s), req=%s, "
                    "main_blocks=%d, indexer_blocks=%d, done=%s",
                    layer_idx,
                    layer_name,
                    ext_id,
                    len(p_main_block_ids),
                    len(p_indexer_block_ids),
                    chunk_done,
                )

        if endpoint_payloads:
            self.mark_layer_pending(layer_idx)
            self._pending_reads_by_layer[layer_idx] = len(endpoint_payloads)
            for (remote_host, remote_port), (read_reqs, done_ext_ids) in endpoint_payloads.items():
                path = make_zmq_path("tcp", remote_host, remote_port)
                dealer = self._ensure_dealer(path)
                if path not in self._mf_meta_sent_paths:
                    self._send_mf_meta(path, dealer, encoder)
                dealer.send(
                    encoder.encode(
                        (READ_READY_BATCH, layer_idx, layer_name, read_reqs, done_ext_ids)
                    )
                )
                if envs.VLLM_ASCEND_SFA_DEBUG:
                    logger.info(
                        "MembPull P send READ_READY_BATCH: layer=%d (%s), "
                        "endpoint=%s:%d, reqs=%d, done_reqs=%d",
                        layer_idx,
                        layer_name,
                        remote_host,
                        remote_port,
                        len(read_reqs),
                        len(done_ext_ids),
                    )
        else:
            self._signal_layer_done(layer_idx)

    def _send_mf_meta(self, path: str, dealer, encoder: msgspec.msgpack.Encoder) -> None:
        p_meta_dict = {}
        for ln, meta in self._state.layer_metadata.items():
            p_meta_dict[ln] = {
                "base_addrs": list(meta.kv_caches_base_addr),
                "block_len": list(meta.block_len),
                "block_size_scale": list(meta.block_size_scale),
                "tensor_group_idx": list(meta.tensor_group_idx),
                "main_tensor_count": meta.main_tensor_count,
                "has_indexer": meta.has_indexer,
            }
        dealer.send(encoder.encode((MF_META, self._state.p_session, encoder.encode(p_meta_dict))))
        if dealer.poll(timeout=int(self.timeout * 1000)):
            frames = dealer.recv_multipart()
            payload = [f for f in frames if f != b""]
            if payload != [b"ACK"]:
                raise RuntimeError(f"MembPull P MF_META got unexpected reply: {payload!r}")
            self._mf_meta_sent_paths.add(path)
            logger.info(
                "MembPull P sent MF_META: session=%s, layers=%d",
                self._state.p_session,
                len(p_meta_dict),
            )
        else:
            raise RuntimeError("MembPull P MF_META timed out (no reply from D)")

    def _drain_read_replies(self, decoder: msgspec.msgpack.Decoder) -> None:
        for dealer in self._dealers.values():
            while True:
                try:
                    if not dealer.poll(timeout=0):
                        break
                    frames = dealer.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                payload = [f for f in frames if f != b""]
                if len(payload) != 1:
                    continue
                try:
                    msg = decoder.decode(payload[0])
                except Exception:
                    continue
                if len(msg) >= 2 and msg[0] == READ_DONE:
                    self._signal_layer_done(msg[1])
                elif len(msg) >= 2 and msg[0] == READ_FAILED:
                    layer_idx = msg[1]
                    error = msg[2] if len(msg) > 2 else ""
                    logger.error(
                        "MembPull P received READ_FAILED: layer=%s, error=%s",
                        layer_idx,
                        error,
                    )
                    self._record_layer_error(layer_idx, str(error))
                    self._signal_layer_done(layer_idx)

    def _signal_layer_done(self, layer_idx: int) -> None:
        pending = self._pending_reads_by_layer.get(layer_idx)
        if pending is not None:
            if pending > 1:
                self._pending_reads_by_layer[layer_idx] = pending - 1
                return
            self._pending_reads_by_layer.pop(layer_idx, None)
        for slot_id in self._state.layer_storage_slots.get(layer_idx, ()):
            self.storage_send_done_events[slot_id].set()
        if envs.VLLM_ASCEND_SFA_DEBUG:
            logger.info("MembPull P layer send complete: layer=%d", layer_idx)

    def _fail_layer(self, layer_idx: int, error: str) -> None:
        self._pending_reads_by_layer.pop(layer_idx, None)
        self._record_layer_error(layer_idx, error)
        self._signal_layer_done(layer_idx)

    def _record_layer_error(self, layer_idx: int, error: str) -> None:
        for slot_id in self._state.layer_storage_slots.get(layer_idx, ()):
            self._storage_read_errors[slot_id] = error

    def get_storage_send_event(self, slot_id: int) -> threading.Event | None:
        if 0 <= slot_id < len(self.storage_send_done_events):
            return self.storage_send_done_events[slot_id]
        return None

    def get_storage_error(self, slot_id: int) -> str | None:
        return self._storage_read_errors.get(slot_id)
