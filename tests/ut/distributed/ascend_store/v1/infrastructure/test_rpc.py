"""Byte-frame RPC works without Lookup messages or cache-miss semantics."""

from tempfile import TemporaryDirectory

import pytest
from zmq.error import Again

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.rpc import (
    RpcClient,
    RpcServer,
    deadline_after,
)


def test_rpc_exchanges_arbitrary_frames_and_reset_allows_reconnection() -> None:
    with TemporaryDirectory(prefix="kvrpc-", dir="/tmp") as directory:
        address = f"ipc://{directory}/rpc"
        server = RpcServer(lambda frames: b"|".join(frames), address)
        client = RpcClient(address)
        try:
            assert client.request([b"first", memoryview(b"second")], deadline=deadline_after(2000)) == b"first|second"
            client.reset_connection()
            assert client.request([bytearray(b"next")], deadline=deadline_after(2000)) == b"next"
            client.close()
            with pytest.raises(RuntimeError, match="RPC client is closed"):
                client.request([b"closed"], deadline=deadline_after(2000))
        finally:
            client.close()
            server.close()


def test_rpc_timeout_raises_and_next_request_can_connect() -> None:
    with TemporaryDirectory(prefix="kvrpc-", dir="/tmp") as directory:
        address = f"ipc://{directory}/rpc"
        client = RpcClient(address)
        server = None
        try:
            with pytest.raises(Again):
                client.request([b"unavailable"], deadline=deadline_after(0))
            server = RpcServer(lambda frames: b"|".join(frames), address)
            assert client.request([b"recovered"], deadline=deadline_after(2000)) == b"recovered"
        finally:
            client.close()
            if server is not None:
                server.close()
