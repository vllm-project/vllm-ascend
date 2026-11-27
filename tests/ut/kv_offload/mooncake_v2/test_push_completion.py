# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import msgspec
import pytest
from zmq.constants import SocketOption, SocketType
from zmq.error import Again
from zmq.sugar.context import Context

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import push_worker


@pytest.fixture
def producer():
    with Context() as context:
        worker = push_worker.MooncakePushConnectorWorker.__new__(push_worker.MooncakePushConnectorWorker)
        worker._zmq_context = context
        worker.pp_size = worker.pcp_size = worker.tp_size = 1
        yield worker


@pytest.fixture
def done_params():
    return {
        "completion_address": "tcp://127.0.0.1:1",
        "request_id": "d-request",
        "producer_engine_id": "p",
        "producer_rank": [0, 0, 0],
        "consumer_engine_id": "d",
        "consumer_rank": 0,
    }


@pytest.mark.parametrize("error", [None, "WRITE failed"])
def test_done_retries_on_fresh_socket_after_lost_ack(producer, done_params, monkeypatch, error):
    monkeypatch.setattr(push_worker.envs, "VLLM_MOONCAKE_CONNECTOR_TIMEOUT", 0.1)
    with Context() as context, context.socket(SocketType.ROUTER) as listener, ThreadPoolExecutor(1) as executor:
        listener.setsockopt(SocketOption.RCVTIMEO, 2000)
        listener.setsockopt(SocketOption.SNDTIMEO, 2000)
        listener.bind("tcp://127.0.0.1:*")
        done_params["completion_address"] = listener.getsockopt(SocketOption.LAST_ENDPOINT).decode()
        completion = executor.submit(producer._send_done, done_params, error)

        first_id, first_delimiter, first_payload = listener.recv_multipart()
        # Drop the first ACK and require a replay of the entire DONE message.
        second_id, second_delimiter, second_payload = listener.recv_multipart()
        assert first_id != second_id
        assert first_delimiter == second_delimiter == b""
        assert first_payload == second_payload
        assert msgspec.msgpack.decode(second_payload) == {
            **{key: value for key, value in done_params.items() if key != "completion_address"},
            "kind": "done",
            "status": "error" if error else "ok",
            "producer_sizes": [1, 1, 1],
            **({"error": error} if error else {}),
        }

        ack = msgspec.msgpack.encode({"kind": "done_ack", "request_id": done_params["request_id"]})
        listener.send_multipart([second_id, second_delimiter, ack])
        assert completion.result(timeout=2)


@pytest.mark.parametrize("request_id", ["d-request", "wrong-request"])
def test_done_accepts_only_matching_ack_without_retry(producer, done_params, monkeypatch, request_id):
    socket = Mock()
    socket.recv.return_value = msgspec.msgpack.encode({"kind": "done_ack", "request_id": request_id})
    make_socket = Mock(return_value=socket)
    monkeypatch.setattr(push_worker, "make_zmq_socket", make_socket)

    assert producer._send_done(done_params, None) == (request_id == done_params["request_id"])
    make_socket.assert_called_once()
    socket.send.assert_called_once()
    socket.close.assert_called_once_with(linger=0)


def test_done_stops_after_bounded_attempts(producer, done_params, monkeypatch):
    sockets = [Mock() for _ in range(push_worker._DONE_ACK_MAX_ATTEMPTS)]
    for socket in sockets:
        socket.recv.side_effect = Again()
    make_socket = Mock(side_effect=sockets)
    monkeypatch.setattr(push_worker, "make_zmq_socket", make_socket)

    assert not producer._send_done(done_params, None)
    assert make_socket.call_count == len(sockets)
    for socket in sockets:
        socket.send.assert_called_once()
        socket.close.assert_called_once_with(linger=0)
