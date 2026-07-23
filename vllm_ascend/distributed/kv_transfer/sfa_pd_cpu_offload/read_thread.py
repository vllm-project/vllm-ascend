# SPDX-License-Identifier: Apache-2.0
"""D-side memfabric pull read thread for SFA PD CPU offload."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import msgspec
import numpy as np
import zmq
from vllm.logger import logger
from vllm.utils.network_utils import get_ip

from vllm_ascend import envs
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.protocol import (
    MF_META,
    READ_DONE,
    READ_FAILED,
    READ_READY_BATCH,
)

READ_THREAD_POLL_TIMEOUT_MS = 100
THREAD_SHUTDOWN_TIMEOUT_SECONDS = 5.0


@dataclass
class ConsumerReadState:
    num_blocks: int
    tp_size: int
    layer_metadata: dict[str, Any]
    main_name_to_idx: dict[str, int]
    cpu_pools: list[tuple[Any, Any] | None]
    main_gva_bases: list[tuple[int, int]]
    main_block_lens: list[tuple[int, int]]
    indexer_tensors: list[Any | None]
    indexer_scale_tensors: list[Any | None]
    dest_blocks_by_req: dict[str, tuple[list[int], list[int]]]
    get_offload_layer_id: Callable[[str], int]


def _coalesce_desc(
    peer: np.ndarray,
    local: np.ndarray,
    length: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = peer.shape[0]
    if n <= 1:
        return peer, local, length
    contiguous = (peer[1:] == peer[:-1] + length[:-1]) & (local[1:] == local[:-1] + length[:-1])
    if contiguous.all():
        return peer[:1], local[:1], np.array([int(length.sum())], dtype=np.int64)
    run_start = np.concatenate(([0], np.nonzero(~contiguous)[0] + 1))
    run_end = np.append(run_start[1:] - 1, n - 1)
    cum = np.cumsum(length)
    merged_len = cum[run_end] - cum[run_start] + length[run_start]
    return peer[run_start], local[run_start], merged_len


def _tp_block_range(
    num_blocks: int,
    tp_rank: int,
    tp_size: int,
    start_block: int = 0,
) -> tuple[int, int]:
    # Rotate the first owner by the chunk's global start so consecutive small
    # chunks do not all fall on TP0.
    logical_rank = (tp_rank - start_block % tp_size) % tp_size
    blocks_per_rank, remainder = divmod(num_blocks, tp_size)
    start = logical_rank * blocks_per_rank + min(logical_rank, remainder)
    count = blocks_per_rank + int(logical_rank < remainder)
    return start, start + count


class MembPullReadThread(threading.Thread):
    """D-side thread for memfabric pull.

    Receives READ_READY_BATCH from P, reads KV from P's HBM via
    batch_transfer_sync_read, then replies with READ_DONE or READ_FAILED.
    """

    def __init__(
        self,
        tp_rank: int,
        side_channel_port: int,
        engine: Any,
        state: ConsumerReadState,
    ):
        super().__init__(daemon=True, name=f"MembPullReadThread-TP{tp_rank}")
        self.tp_rank = tp_rank
        self.side_channel_port = side_channel_port
        self.engine = engine
        self._state = state
        self.ready_event = threading.Event()
        self._p_session: str | None = None
        self._p_layer_meta: dict[str, Any] = {}
        self._p_sessions: dict[bytes, str] = {}
        self._p_layer_metas: dict[bytes, dict[str, Any]] = {}
        self._done_requests: set[str] = set()
        self._failed_requests: set[str] = set()
        self._lock = threading.Lock()
        self._host = get_ip()
        self._stop_event = threading.Event()
        self.startup_error: BaseException | None = None

    def get_and_clear_done(self) -> set[str]:
        with self._lock:
            d = self._done_requests
            self._done_requests = set()
            return d

    def get_and_clear_failed(self) -> set[str]:
        with self._lock:
            failed = self._failed_requests
            self._failed_requests = set()
            return failed

    def run(self):
        from vllm.utils.network_utils import make_zmq_path, make_zmq_socket

        handshake_port = self.side_channel_port + self.tp_rank
        path = make_zmq_path("tcp", self._host, handshake_port)
        logger.info("MembPull read thread listening on: %s", path)
        ctx = zmq.Context()
        sock = None
        try:
            try:
                sock = make_zmq_socket(ctx=ctx, path=path, socket_type=zmq.ROUTER, bind=True)
                sock.setsockopt(zmq.RCVTIMEO, READ_THREAD_POLL_TIMEOUT_MS)
            except BaseException as error:
                self.startup_error = error
                logger.error("MembPull read thread failed to start on %s: %s", path, error)
                return
            finally:
                self.ready_event.set()
            decoder = msgspec.msgpack.Decoder(type=tuple)
            encoder = msgspec.msgpack.Encoder()
            while not self._stop_event.is_set():
                try:
                    frames = sock.recv_multipart()
                    if len(frames) < 2:
                        continue
                    identity = frames[0]
                    payload = [f for f in frames[1:] if f != b""]
                    if len(payload) != 1:
                        continue
                    msg = decoder.decode(payload[0])
                    msg_type = msg[0]

                    if msg_type == MF_META:
                        self._p_session = msg[1]
                        self._p_layer_meta = msgspec.msgpack.decode(msg[2])
                        self._p_sessions[identity] = self._p_session
                        self._p_layer_metas[identity] = self._p_layer_meta
                        logger.info(
                            "Received MF_META: P session=%s, %d layers", self._p_session, len(self._p_layer_meta)
                        )
                        for layer_name, layer_meta in self._p_layer_meta.items():
                            if envs.VLLM_ASCEND_SFA_DEBUG:
                                logger.info(
                                    "MembPull D recv MF_META layer=%s: base_addrs=%s, "
                                    "block_len=%s, block_size_scale=%s",
                                    layer_name,
                                    layer_meta.get("base_addrs"),
                                    layer_meta.get("block_len"),
                                    layer_meta.get("block_size_scale"),
                                )
                        sock.send_multipart((identity, b"", b"ACK"))

                    elif msg_type == READ_READY_BATCH:
                        layer_idx = msg[1]
                        layer_name = msg[2]
                        # Normalize each request to:
                        # (external_req_id, p_main_block_ids,
                        #  p_indexer_block_ids, main_start_block,
                        #  indexer_start_block). The block-id lists identify
                        # source blocks in P's HBM; the start offsets identify
                        # the first corresponding destination block on D.
                        # Legacy two-field entries contain one shared block-id
                        # list, so use it for both components and default both
                        # destination offsets to zero.
                        read_reqs = [
                            (
                                entry[0],
                                list(entry[1]),
                                list(entry[2] if len(entry) > 2 else entry[1]),
                                int(entry[3]) if len(entry) > 3 else 0,
                                int(entry[4]) if len(entry) > 4 else 0,
                            )
                            for entry in msg[3]
                        ]
                        done_ext_ids = list(msg[4]) if len(msg) > 4 else []
                        if envs.VLLM_ASCEND_SFA_DEBUG:
                            logger.info(
                                "MembPull D recv READ_READY_BATCH: layer=%d (%s), reqs=%d, done_reqs=%d",
                                layer_idx,
                                layer_name,
                                len(read_reqs),
                                len(done_ext_ids),
                            )
                        succeeded = False
                        try:
                            p_session = self._p_sessions.get(identity)
                            p_layer_meta = self._p_layer_metas.get(identity)
                            if p_session is None or p_layer_meta is None:
                                raise RuntimeError(
                                    "MF_META was not received from this P connection "
                                    "before READ_READY_BATCH"
                                )
                            if read_reqs:
                                self._do_read_batch(
                                    layer_name,
                                    read_reqs,
                                    p_session=p_session,
                                    p_layer_meta=p_layer_meta,
                                )
                            sock.send_multipart((identity, b"", encoder.encode((READ_DONE, layer_idx))))
                            succeeded = True
                            if envs.VLLM_ASCEND_SFA_DEBUG:
                                logger.info(
                                    "MembPull D sent READ_DONE: layer=%d (%s), reqs=%d, done_reqs=%d",
                                    layer_idx,
                                    layer_name,
                                    len(read_reqs),
                                    len(done_ext_ids),
                                )
                        except Exception as e:
                            logger.error(
                                "MembPull batch read failed for layer %d (%s), reqs=%d: %s",
                                layer_idx,
                                layer_name,
                                len(read_reqs),
                                e,
                            )
                            payload = encoder.encode((READ_FAILED, layer_idx, str(e)))
                            sock.send_multipart((identity, b"", payload))
                            with self._lock:
                                self._failed_requests.update(entry[0] for entry in read_reqs)
                        if succeeded and done_ext_ids:
                            with self._lock:
                                self._done_requests.update(done_ext_ids)

                    else:
                        logger.error("MembPull got unexpected message %s", msg)
                except zmq.Again:
                    continue
                except Exception as e:
                    logger.error("MembPull exception: %s: %s", type(e), e)
        finally:
            self.ready_event.set()
            if sock is not None:
                sock.close(linger=0)
            ctx.destroy(linger=0)

    def stop(self, timeout: float = THREAD_SHUTDOWN_TIMEOUT_SECONDS) -> None:
        self._stop_event.set()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=timeout)
        if self.is_alive():
            logger.warning("MembPull read thread did not stop within %.1f seconds", timeout)

    def _resolve_read_layer(
        self,
        layer_name: str,
        p_layer_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        state = self._state
        if p_layer_meta is None:
            p_layer_meta = self._p_layer_meta
        pool_idx = state.main_name_to_idx.get(layer_name)
        if pool_idx is None:
            logger.warning("MembPull _do_read: layer %s not in main names, skip", layer_name)
            return None
        offload_id = state.get_offload_layer_id(layer_name)
        if pool_idx != offload_id and envs.VLLM_ASCEND_SFA_DEBUG:
            logger.warning(
                "MembPull _do_read: layer-order mismatch for %s -- pull _main_names "
                "idx=%d != resident offload_id=%d. Using offload_id.",
                layer_name,
                pool_idx,
                offload_id,
            )
        p_meta = p_layer_meta.get(layer_name)
        if p_meta is None:
            logger.warning(
                "MembPull _do_read: layer %s not in P layer_meta (MF_META not received? have %d layers), skip",
                layer_name,
                len(p_layer_meta),
            )
            return None
        p_base_addrs = p_meta["base_addrs"]
        p_block_len = p_meta["block_len"]
        p_block_size_scale = p_meta.get("block_size_scale", [1] * len(p_base_addrs))
        main_tensor_count = int(p_meta.get("main_tensor_count", 2))
        if main_tensor_count != 2 or len(p_base_addrs) < main_tensor_count:
            raise RuntimeError(
                f"MembPull P metadata for {layer_name} must expose two main K/V "
                f"tensors, got main_tensor_count={main_tensor_count}, "
                f"total={len(p_base_addrs)}"
            )
        p_has_indexer = bool(
            p_meta.get("has_indexer", len(p_base_addrs) > main_tensor_count)
        )

        try:
            k_cpu_ptr, v_cpu_ptr = state.main_gva_bases[offload_id]
            d_k_len, d_v_len = state.main_block_lens[offload_id]
        except IndexError as error:
            raise RuntimeError(
                f"MembPull shared CPU pool metadata is missing for {layer_name}"
            ) from error
        p_k_len = p_block_len[0] * p_block_size_scale[0]
        p_v_len = p_block_len[1] * p_block_size_scale[1]
        if (p_k_len, p_v_len) != (d_k_len, d_v_len):
            raise RuntimeError(
                f"MembPull main KV layout mismatch for {layer_name}: "
                f"P=({p_k_len}, {p_v_len}), D=({d_k_len}, {d_v_len})"
            )
        d_indexer = state.indexer_tensors[pool_idx]
        if p_has_indexer != (d_indexer is not None):
            raise RuntimeError(
                f"MembPull indexer presence mismatch for {layer_name}: "
                f"P={p_has_indexer}, D={d_indexer is not None}"
            )
        indexer = None
        if p_has_indexer:
            indexer_pos = main_tensor_count
            if len(p_base_addrs) <= indexer_pos:
                raise RuntimeError(
                    f"MembPull P metadata for {layer_name} marks an indexer but "
                    "does not expose its tensor"
                )
            p_dsa_len = p_block_len[indexer_pos] * p_block_size_scale[indexer_pos]
            d_dsa_row_len = d_indexer.element_size() * math.prod(d_indexer.shape[1:])
            d_dsa_len = d_dsa_row_len * (d_indexer.shape[0] // state.num_blocks)
            if p_dsa_len != d_dsa_len:
                raise RuntimeError(
                    f"MembPull indexer layout mismatch for {layer_name}: "
                    f"P external block bytes={p_dsa_len}, D={d_dsa_len}"
                )
            indexer = {
                "p_dsa_base": p_base_addrs[indexer_pos],
                "block_len": p_dsa_len,
                "d_base": d_indexer.data_ptr(),
                "shape": tuple(d_indexer.shape),
            }

        scale = None
        scale_tensor = state.indexer_scale_tensors[pool_idx] if pool_idx < len(state.indexer_scale_tensors) else None
        if scale_tensor is not None:
            scale_pos = main_tensor_count + 1
            if len(p_base_addrs) <= scale_pos:
                scale = {"error": "p_addr_mismatch", "p_n": len(p_base_addrs)}
            else:
                p_scale_len = p_block_len[scale_pos] * p_block_size_scale[scale_pos]
                d_scale_row_len = scale_tensor.element_size() * math.prod(scale_tensor.shape[1:])
                d_scale_factor = scale_tensor.shape[0] // state.num_blocks
                d_scale_len = d_scale_row_len * d_scale_factor
                if p_scale_len != d_scale_len:
                    scale = {
                        "error": "layout_mismatch",
                        "p_scale_len": p_scale_len,
                        "d_scale_len": d_scale_len,
                    }
                else:
                    scale = {
                        "p_scale_base": p_base_addrs[scale_pos],
                        "block_len": p_scale_len,
                        "d_scale_base": scale_tensor.data_ptr(),
                    }

        return {
            "layer_name": layer_name,
            "pool_idx": pool_idx,
            "offload_id": offload_id,
            "p_k_base": p_base_addrs[0],
            "p_v_base": p_base_addrs[1],
            "p_k_len": p_block_len[0] * p_block_size_scale[0],
            "p_v_len": p_block_len[1] * p_block_size_scale[1],
            "k_cpu_ptr": k_cpu_ptr,
            "v_cpu_ptr": v_cpu_ptr,
            "indexer": indexer,
            "scale": scale,
        }

    def _build_req_descriptors(
        self,
        layer: dict[str, Any],
        ext_req_id: str,
        p_main_block_ids: list[int],
        p_indexer_block_ids: list[int],
        want_info: bool,
        main_start_block: int = 0,
        indexer_start_block: int = 0,
    ) -> tuple[list[int], list[int], list[int], dict[str, Any] | None]:
        state = self._state
        layer_name = layer["layer_name"]

        dest = state.dest_blocks_by_req.get(ext_req_id)
        if dest is None:
            raise RuntimeError(
                f"MembPull has no destination blocks on D for req {ext_req_id} "
                f"(layer {layer_name})"
            )
        all_d_main_ids, all_d_indexer_ids = dest
        main_end_block = main_start_block + len(p_main_block_ids)
        if main_end_block > len(all_d_main_ids):
            raise RuntimeError(
                f"MembPull main destination range is incomplete for req {ext_req_id}: "
                f"range=[{main_start_block}, {main_end_block}), "
                f"allocated={len(all_d_main_ids)}"
            )
        d_main_ids = all_d_main_ids[main_start_block:main_end_block]
        indexer_end_block = indexer_start_block + len(p_indexer_block_ids)
        if indexer_end_block > len(all_d_indexer_ids):
            raise RuntimeError(
                f"MembPull indexer destination range is incomplete for req {ext_req_id}: "
                f"range=[{indexer_start_block}, {indexer_end_block}), "
                f"allocated={len(all_d_indexer_ids)}"
            )
        d_indexer_ids = all_d_indexer_ids[indexer_start_block:indexer_end_block]
        if envs.VLLM_ASCEND_SFA_DEBUG:
            logger.info(
                "MembPull D resolve dest: layer=%s, req=%s, p_main_ids=%s, "
                "p_indexer_ids=%s, d_main_cpu_ids=%s, d_indexer_hbm_ids=%s",
                layer_name,
                ext_req_id,
                p_main_block_ids,
                p_indexer_block_ids,
                d_main_ids,
                d_indexer_ids,
            )
        if not p_main_block_ids and not p_indexer_block_ids:
            raise RuntimeError(
                f"MembPull source block ids are empty for {layer_name}"
            )

        p_k_base, p_v_base = layer["p_k_base"], layer["p_v_base"]
        p_k_len, p_v_len = layer["p_k_len"], layer["p_v_len"]

        peer_chunks: list[np.ndarray] = []
        local_chunks: list[np.ndarray] = []
        length_chunks: list[np.ndarray] = []
        n_main = 0
        n_indexer = 0

        has_cpu_destination = layer["k_cpu_ptr"] is not None and layer["v_cpu_ptr"] is not None
        if has_cpu_destination and len(p_main_block_ids) != len(d_main_ids):
            raise RuntimeError(
                f"MembPull main block count mismatch for req {ext_req_id}: "
                f"P={len(p_main_block_ids)}, D={len(d_main_ids)}"
            )
        if has_cpu_destination:
            owned_start, owned_end = _tp_block_range(
                len(d_main_ids),
                self.tp_rank,
                state.tp_size,
                main_start_block,
            )
            p_main_block_ids = p_main_block_ids[owned_start:owned_end]
            d_main_ids = d_main_ids[owned_start:owned_end]
        n_main = len(d_main_ids) if has_cpu_destination else 0
        if n_main:
            p_main = np.array(p_main_block_ids, dtype=np.int64)
            d_main = np.array(d_main_ids, dtype=np.int64)
            len_k = np.full(n_main, p_k_len, dtype=np.int64)
            len_v = np.full(n_main, p_v_len, dtype=np.int64)
            cp, cl, clen = _coalesce_desc(p_k_base + p_main * p_k_len, layer["k_cpu_ptr"] + d_main * p_k_len, len_k)
            peer_chunks.append(cp)
            local_chunks.append(cl)
            length_chunks.append(clen)
            cp, cl, clen = _coalesce_desc(p_v_base + p_main * p_v_len, layer["v_cpu_ptr"] + d_main * p_v_len, len_v)
            peer_chunks.append(cp)
            local_chunks.append(cl)
            length_chunks.append(clen)

        idx = layer["indexer"]
        if idx is not None and len(p_indexer_block_ids) != len(d_indexer_ids):
            raise RuntimeError(
                f"MembPull indexer block count mismatch for req {ext_req_id}: "
                f"P={len(p_indexer_block_ids)}, D={len(d_indexer_ids)}"
            )
        if idx is not None and d_indexer_ids:
            p_dsa_base, d_base = idx["p_dsa_base"], idx["d_base"]
            block_len = idx["block_len"]
            d_idx_arr = np.array(d_indexer_ids, dtype=np.int64)
            p_idx = np.array(p_indexer_block_ids, dtype=np.int64)
            logger.debug(
                "MembPull indexer %s: block_len=%d, D dsa_k shape=%s, blocks=%d",
                layer_name,
                block_len,
                idx["shape"],
                len(p_indexer_block_ids),
            )
            cp, cl, clen = _coalesce_desc(
                p_dsa_base + p_idx * block_len,
                d_base + d_idx_arr * block_len,
                np.full(len(p_idx), block_len, dtype=np.int64),
            )
            peer_chunks.append(cp)
            local_chunks.append(cl)
            length_chunks.append(clen)
            n_indexer = len(p_idx)

        scale = layer.get("scale")
        if scale is not None and d_indexer_ids:
            if "error" in scale:
                if scale["error"] == "p_addr_mismatch":
                    raise RuntimeError(
                        f"MembPull indexer scale {layer_name}: D is LIC8 (has scale "
                        f"tensor) but P exposed only {scale['p_n']} base addrs "
                        f"(no scale leg) -- P/D LIC8 config mismatch."
                    )
                raise RuntimeError(
                    f"MembPull indexer scale {layer_name}: D scale block_len="
                    f"{scale['d_scale_len']} not a multiple of P={scale['p_scale_len']} -- "
                    f"scale layout mismatch; refusing to transfer to avoid silent "
                    f"stale-scale corruption."
                )
            p_scale_base = scale["p_scale_base"]
            block_len = scale["block_len"]
            d_scale_base = scale["d_scale_base"]
            if d_indexer_ids:
                p_scale_arr = np.array(p_indexer_block_ids, dtype=np.int64)
                d_scale_arr = np.array(d_indexer_ids, dtype=np.int64)
                cp, cl, clen = _coalesce_desc(
                    p_scale_base + p_scale_arr * block_len,
                    d_scale_base + d_scale_arr * block_len,
                    np.full(len(p_scale_arr), block_len, dtype=np.int64),
                )
                peer_chunks.append(cp)
                local_chunks.append(cl)
                length_chunks.append(clen)

        if not peer_chunks:
            logger.debug(
                "MembPull _do_read: this rank owns no requested transfer for %s "
                "(main=%d, indexer=%d)",
                layer_name,
                n_main,
                n_indexer,
            )
            return [], [], [], None

        peer_ptrs = np.concatenate(peer_chunks).tolist()
        local_ptrs = np.concatenate(local_chunks).tolist()
        lengths = np.concatenate(length_chunks).tolist()

        info = None
        if want_info:
            info = {
                "layer_name": layer_name,
                "ext_req_id": ext_req_id,
                "pool_idx": layer["pool_idx"],
                "offload_id": layer["offload_id"],
                "d_main_ids": d_main_ids,
                "d_indexer_ids": d_indexer_ids,
                "n_main": n_main,
                "n_indexer": n_indexer,
                "num_transfers": len(local_ptrs),
                "atomic_transfers": 2 * n_main + n_indexer,
            }
        return local_ptrs, peer_ptrs, lengths, info

    def _log_read_result(self, read_info: dict[str, Any]) -> None:
        state = self._state
        layer_name = read_info["layer_name"]
        ext_req_id = read_info["ext_req_id"]
        pool_idx = read_info["pool_idx"]
        offload_id = read_info["offload_id"]
        d_main_ids = read_info["d_main_ids"]
        d_indexer_ids = read_info["d_indexer_ids"]
        if envs.VLLM_ASCEND_MF_VERIFY:
            try:
                cpu_pool = state.cpu_pools[offload_id]
                if cpu_pool is None:
                    mk = mv = "n/a"
                else:
                    k_cpu, v_cpu = cpu_pool
                    mk = "%.6f" % (
                        k_cpu[d_main_ids].float().sum().item() if d_main_ids else 0.0
                    )
                    mv = "%.6f" % (
                        v_cpu[d_main_ids].float().sum().item() if d_main_ids else 0.0
                    )
                mi = state.indexer_tensors[pool_idx][d_indexer_ids].float().sum().item() if d_indexer_ids else 0.0
                logger.info(
                    "MFV D layer %s req %s main_k=%s main_v=%s idx_post=%.6f",
                    layer_name,
                    ext_req_id,
                    mk,
                    mv,
                    mi,
                )
            except Exception as ve:
                logger.warning("MFV D checksum failed for %s: %s", layer_name, ve)
        if envs.VLLM_ASCEND_SFA_DEBUG:
            logger.info(
                "MembPull D finished read: layer=%s, req=%s, pool_idx=%d, "
                "main=%d/%d blocks, indexer=%d/%d blocks (scale split), transfers=%d",
                layer_name,
                ext_req_id,
                pool_idx,
                read_info["n_main"],
                len(d_main_ids),
                read_info["n_indexer"],
                len(d_indexer_ids),
                read_info["num_transfers"],
            )

    def _do_read_batch(
        self,
        layer_name: str,
        read_reqs: list[tuple[str, list[int], list[int], int, int]],
        p_session: str | None = None,
        p_layer_meta: dict[str, Any] | None = None,
    ) -> None:
        if p_session is None:
            p_session = self._p_session
        if p_session is None:
            raise RuntimeError("MF_META not received before READ_READY_BATCH")

        layer = self._resolve_read_layer(layer_name, p_layer_meta)
        if layer is None:
            raise RuntimeError(f"MembPull cannot resolve P/D layout for {layer_name}")

        want_info = bool(envs.VLLM_ASCEND_MF_VERIFY or envs.VLLM_ASCEND_SFA_DEBUG)
        all_local_ptrs: list[int] = []
        all_peer_ptrs: list[int] = []
        all_lengths: list[int] = []
        read_infos: list[dict[str, Any]] = []
        for (
            ext_req_id,
            p_main_block_ids,
            p_indexer_block_ids,
            main_start_block,
            indexer_start_block,
        ) in read_reqs:
            local_ptrs, peer_ptrs, lengths, read_info = self._build_req_descriptors(
                layer,
                ext_req_id,
                p_main_block_ids,
                p_indexer_block_ids,
                want_info,
                main_start_block,
                indexer_start_block,
            )
            if not local_ptrs:
                owned_main_start, owned_main_end = _tp_block_range(
                    len(p_main_block_ids),
                    self.tp_rank,
                    self._state.tp_size,
                    main_start_block,
                )
                owns_requested_main = (
                    layer["k_cpu_ptr"] is not None
                    and layer["v_cpu_ptr"] is not None
                    and owned_main_end > owned_main_start
                )
                owns_requested_indexer = layer["indexer"] is not None and bool(
                    p_indexer_block_ids
                )
                if owns_requested_main or owns_requested_indexer:
                    raise RuntimeError(
                        f"MembPull built no transfer descriptors for layer "
                        f"{layer_name}, req {ext_req_id}"
                    )
                if envs.VLLM_ASCEND_SFA_DEBUG:
                    logger.info(
                        "MembPull D skips local no-op: layer=%s, req=%s, "
                        "P main blocks=%d, P indexer blocks=%d",
                        layer_name,
                        ext_req_id,
                        len(p_main_block_ids),
                        len(p_indexer_block_ids),
                    )
                continue
            all_local_ptrs.extend(local_ptrs)
            all_peer_ptrs.extend(peer_ptrs)
            all_lengths.extend(lengths)
            if want_info:
                read_infos.append(read_info)

        if not all_local_ptrs:
            # This rank may own no main block in a small chunk and the layer may
            # have no rank-local indexer transfer. It still acknowledges
            # READ_DONE so its corresponding P rank can reuse the source.
            return

        if envs.VLLM_ASCEND_SFA_DEBUG:
            atomic_total = sum(r.get("atomic_transfers", 0) for r in read_infos)
            logger.info(
                "MembPull D start batched memfabric read: layer=%s, reqs=%d, "
                "p_session=%s, transfers=%d (coalesced from %d)",
                layer_name,
                len(read_infos),
                p_session,
                len(all_local_ptrs),
                atomic_total,
            )
        ret = self.engine.batch_transfer_sync_read(p_session, all_local_ptrs, all_peer_ptrs, all_lengths)
        if ret != 0:
            raise RuntimeError(f"memfabric batch read failed for layer {layer_name}, ret={ret}")
        for read_info in read_infos:
            self._log_read_result(read_info)
