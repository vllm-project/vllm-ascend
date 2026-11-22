from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import vllm.envs as envs
import zmq
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LookupResult
from vllm_ascend.utils import vllm_version_is

LOOKUP_REPLAY_MSG = b"lookup_replay"
_TOKEN_COUNT_BYTES = 4

if vllm_version_is("0.33.0"):
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.protocol import (  # type: ignore[import-not-found]
        LOOKUP_MSG,
        RESET_MSG,
        RESP_ERR,
        RESP_OK,
    )
else:
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.protocol import (
        LOOKUP_MSG,
        RESET_MSG,
        RESP_ERR,
        RESP_OK,
    )

if TYPE_CHECKING:
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import BaseKVPoolWorker


def get_zmq_rpc_path_lookup(vllm_config: VllmConfig) -> str:
    # NOTE: This isolates lookup IPC paths only. MooncakeConnector (V1) worker
    # handshake ports and MooncakeConnectorV2 scheduler control ports may still
    # overlap ports reserved by other workers or components. DP indices do not
    # guarantee disjoint TCP port ranges.
    # Dense engines reset their compute DP rank, but retain their serving index.
    dp_index = vllm_config.parallel_config.data_parallel_index
    base_url = envs.VLLM_RPC_BASE_PATH
    rpc_port = 0
    extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
    if "lookup_rpc_port" in extra_config:
        rpc_port = extra_config["lookup_rpc_port"]
    elif "mooncake_rpc_port" in extra_config:
        rpc_port = extra_config["mooncake_rpc_port"]
        logger.warning(
            "It is recommended to use the lookup_rpc_port, as the mooncake_rpc_port will be removed in the future."
        )
    logger.debug("Base URL: %s, RPC Port: %s", base_url, rpc_port)
    return f"ipc://{base_url}/lookup_rpc_port_{rpc_port}_dp_rank{dp_index}"


class LookupKeyClient:
    def __init__(self, vllm_config: VllmConfig):
        self.encoder = MsgpackEncoder()
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lookup(vllm_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REQ,  # type: ignore[attr-defined]
            bind=False,
        )
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # type: ignore[attr-defined]
        self._closed = False

    def lookup(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        kv_cache_group_ids: list[int] | None = None,
        hbm_hit_tokens: int = 0,
    ) -> int:
        resp = self._lookup(token_len, block_hashes, kv_cache_group_ids, hbm_hit_tokens)
        return int.from_bytes(resp, "big") if resp is not None else 0

    def lookup_with_replay(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        kv_cache_group_ids: list[int],
        *,
        replay_token_limit: int,
        hbm_hit_tokens: int = 0,
    ) -> LookupResult:
        resp = self._lookup(token_len, block_hashes, kv_cache_group_ids, hbm_hit_tokens, replay_token_limit)
        if resp is None:
            return LookupResult(0, 0)
        if len(resp) == 2 * _TOKEN_COUNT_BYTES:
            raw_hit = int.from_bytes(resp[:_TOKEN_COUNT_BYTES], "big")
            replay_hit = int.from_bytes(resp[_TOKEN_COUNT_BYTES:], "big")
            if 0 <= replay_hit <= min(raw_hit, replay_token_limit) and raw_hit <= token_len:
                return LookupResult(raw_hit, replay_hit)
        logger.error("Invalid KV pool replay lookup response; reporting no external hit")
        self.close()
        return LookupResult(0, 0)

    def _lookup(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        kv_cache_group_ids: list[int] | None,
        hbm_hit_tokens: int,
        replay_token_limit: int | None = None,
    ) -> bytes | None:
        if self._closed:
            return None

        kv_cache_group_ids = kv_cache_group_ids or [0]
        hash_strs = [h.hex() for h in block_hashes]
        hash_frames = self.encoder.encode(hash_strs)
        kv_group_frames = self.encoder.encode(kv_cache_group_ids)
        all_frames = [
            LOOKUP_MSG if replay_token_limit is None else LOOKUP_REPLAY_MSG,
            token_len.to_bytes(4, byteorder="big"),
            *kv_group_frames,
            hbm_hit_tokens.to_bytes(4, byteorder="big"),
            *hash_frames,
        ]
        if replay_token_limit is not None:
            all_frames.insert(4, replay_token_limit.to_bytes(_TOKEN_COUNT_BYTES, "big"))
        try:
            self.socket.send_multipart(all_frames, copy=False)
            resp = self.socket.recv()
        except zmq.ZMQError:  # type: ignore[attr-defined]
            logger.exception("KV pool lookup failed; reporting 0 hit tokens")
            self.close()
            return None

        return bytes(resp)

    def reset(self) -> bool:
        """Ask rank 0 to drain queued puts, then ``remove_all(force=True)``."""
        self.socket.send(RESET_MSG)
        resp = self.socket.recv()
        return bytes(resp) == RESP_OK

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.socket.close(linger=0)
        finally:
            # Each client owns a private context. Close it too so repeated
            # lookup failures do not leak ZeroMQ IO threads.
            self.ctx.term()

    @property
    def closed(self) -> bool:
        return self._closed


class LookupKeyServer:
    def __init__(
        self,
        pool_worker: BaseKVPoolWorker,
        vllm_config: VllmConfig,
    ):
        self.decoder = MsgpackDecoder()
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lookup(vllm_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REP,  # type: ignore[attr-defined]
            bind=True,
        )

        self.pool_worker = pool_worker
        self.running = True
        self._closed = False

        def process_request():
            poller = zmq.Poller()  # type: ignore[attr-defined]
            poller.register(self.socket, zmq.POLLIN)  # type: ignore[attr-defined]
            try:
                while self.running:
                    if not poller.poll(timeout=100):
                        continue

                    # A failed receive leaves a REP socket in an unknown
                    # protocol state. Do not try to send a synthetic reply.
                    try:
                        all_frames = self.socket.recv_multipart(copy=False)
                    except zmq.ZMQError:  # type: ignore[attr-defined]
                        if self.running:
                            logger.exception("Failed to receive KV pool lookup request")
                        break

                    try:
                        msg_type = bytes(all_frames[0])
                        if msg_type in (LOOKUP_MSG, LOOKUP_REPLAY_MSG):
                            token_len = int.from_bytes(all_frames[1], byteorder="big")
                            kv_group_ids = self.decoder.decode([all_frames[2]])
                            hbm_hit_tokens = int.from_bytes(all_frames[3], byteorder="big")
                            if msg_type == LOOKUP_REPLAY_MSG:
                                replay_token_limit = int.from_bytes(all_frames[4], byteorder="big")
                                hashes_str = self.decoder.decode(all_frames[5:])
                                hits = self.pool_worker.lookup_scheduler_with_replay(
                                    token_len,
                                    hashes_str,
                                    kv_group_ids,
                                    replay_token_limit=replay_token_limit,
                                    hbm_hit_tokens=hbm_hit_tokens,
                                )
                                response = hits.raw_hit_tokens.to_bytes(
                                    _TOKEN_COUNT_BYTES, "big"
                                ) + hits.replay_hit_tokens.to_bytes(_TOKEN_COUNT_BYTES, "big")
                            else:
                                hashes_str = self.decoder.decode(all_frames[4:])
                                result = self.pool_worker.lookup_scheduler(
                                    token_len,
                                    hashes_str,
                                    kv_group_ids,
                                    use_layerwise=False,
                                    hbm_hit_tokens=hbm_hit_tokens,
                                )
                                logger.debug(
                                    "KV pool lookup response token_len=%d groups=%s hit_tokens=%d",
                                    token_len,
                                    kv_group_ids,
                                    result,
                                )
                                response = result.to_bytes(4, "big")
                        elif msg_type == RESET_MSG:
                            self.pool_worker.reset_store()
                            logger.info("AscendStore reset via remove_all succeeded.")
                            response = RESP_OK
                        else:
                            logger.warning("LookupKeyServer received unknown msg_type: %r", msg_type)
                            response = RESP_ERR
                    except Exception as e:
                        logger.error("Error handling lookup request: %s", e)
                        response = RESP_ERR

                    try:
                        self.socket.send(response)
                    except zmq.ZMQError:  # type: ignore[attr-defined]
                        if self.running:
                            logger.exception("Failed to send KV pool lookup response")
                        break
            finally:
                self.socket.close(linger=0)
                self.ctx.term()

        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.running = False
        self.thread.join()
