import multiprocessing as mp
import queue
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import zmq

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.rpc import (
    AffinityExecutor,
    BoundedThreadPoolExecutor,
    InlineExecutor,
    MPClient,
    MPClientClosedError,
    MPProtocolError,
    MPRemoteError,
    MPRequestTimeoutError,
    MPServer,
    MPServerAbortedError,
    MPServerBusyError,
    MPServerUnavailableError,
    Route,
    SystemMethod,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.rpc.client import _ClientState
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.rpc.protocol import (
    ResponseStatus,
    decode_request,
    decode_response,
    encode_request,
    encode_response,
)

UPPERCASE_METHOD = "TEST_UPPERCASE"
INVALID_RESPONSE_METHOD = "TEST_INVALID_RESPONSE"
AFFINITY_METHOD = "TEST_AFFINITY"
DEFAULT_AFFINITY_METHOD = "TEST_DEFAULT_AFFINITY"
BLOCKING_METHOD = "TEST_BLOCKING"
RPC_CLIENT_MODULE = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.rpc.client"

_WORKER_COUNT = 4
_REQUESTS_PER_WORKER = 16


def _start_server(target):
    context = mp.get_context("fork")
    parent_conn, child_conn = context.Pipe()
    process = context.Process(target=target, args=(child_conn,))
    process.start()
    child_conn.close()

    try:
        assert parent_conn.poll(5), "Server did not start in time"
        endpoint = parent_conn.recv()
    except Exception:
        parent_conn.close()
        if process.is_alive():
            process.terminate()
        process.join()
        raise

    return process, parent_conn, endpoint


def _cleanup(client: MPClient | None, conn, process) -> None:
    if client is not None:
        client.close()

    conn.close()

    if process.is_alive():
        process.terminate()
    process.join()


def _send_ok_response(router: zmq.Socket, request: list[bytes], responses: tuple[bytes, ...] | None = None) -> None:  # type: ignore[name-defined]
    identity, *request_frames = request
    request_id, method, _, payloads = decode_request(request_frames)
    response_payloads = payloads if responses is None else responses
    response = [identity, *encode_response(request_id, method, ResponseStatus.OK, response_payloads)]
    router.send_multipart(response)


def _run_server(conn) -> None:
    server = MPServer("tcp://127.0.0.1:*")

    try:
        conn.send(server.endpoint)
        conn.close()
        server.run()
    finally:
        server.close()


def _run_client_worker(endpoint: str, worker_id: int, start_event, conn) -> None:
    try:
        with MPClient(endpoint) as client:
            client.wait_until_connected()
            conn.send(("ready", worker_id))

            if not start_event.wait(5):
                raise TimeoutError("Timed out waiting for other workers")

            responses = []
            for request_id in range(_REQUESTS_PER_WORKER):
                payload = f"worker-{worker_id}-request-{request_id}".encode()
                responses.append(client.echo(payload))

            conn.send(("result", worker_id, responses))
    except Exception as exc:
        conn.send(("error", worker_id, f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def _uppercase_handler(payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
    if len(payloads) != 1:
        raise ValueError(f"{UPPERCASE_METHOD} expects 1 payload, got {len(payloads)}")
    return (payloads[0].upper(),)


def _invalid_response_handler(_payloads: tuple[bytes, ...]):
    return None


def _affinity_handler(payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
    if len(payloads) != 1:
        raise ValueError(f"{AFFINITY_METHOD} expects 1 payload, got {len(payloads)}")
    return (str(threading.get_ident()).encode(),)


def _integer_affinity_key(_identity: bytes, payloads: tuple[bytes, ...]) -> int:
    return int(payloads[0])


def _client_identity_key(identity: bytes, _payloads: tuple[bytes, ...]) -> bytes:
    return identity


def _run_server_with_injected_handlers(conn) -> None:
    parallel_executor = BoundedThreadPoolExecutor(2, 64, "test-server-parallel")
    affinity_executor = AffinityExecutor(2, 64, "test-server-affinity")
    server = MPServer(
        "tcp://127.0.0.1:*",
        routes=(
            Route(UPPERCASE_METHOD, _uppercase_handler, parallel_executor),
            Route(INVALID_RESPONSE_METHOD, _invalid_response_handler, parallel_executor),
            Route(AFFINITY_METHOD, _affinity_handler, affinity_executor, _integer_affinity_key),
            Route(DEFAULT_AFFINITY_METHOD, _affinity_handler, affinity_executor, _client_identity_key),
        ),
    )

    try:
        conn.send(server.endpoint)
        conn.close()
        server.run()
    finally:
        server.close()


def _run_bounded_server(conn) -> None:
    def blocking_handler(payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        conn.send("handler_started")
        if conn.recv() != "release_handler":
            raise ValueError("Unexpected bounded server command")
        return payloads

    executor = BoundedThreadPoolExecutor(1, 0, "test-bounded-server")
    server = MPServer("tcp://127.0.0.1:*", routes=(Route(BLOCKING_METHOD, blocking_handler, executor),))

    try:
        conn.send(server.endpoint)
        server.run()
    finally:
        server.close()
        conn.close()


def _run_reordering_server(conn) -> None:
    context = zmq.Context()  # type: ignore[attr-defined]
    router = context.socket(zmq.ROUTER)  # type: ignore[attr-defined]
    port = router.bind_to_random_port("tcp://127.0.0.1")

    try:
        conn.send(f"tcp://127.0.0.1:{port}")
        requests = [router.recv_multipart() for _ in range(3)]

        for request in reversed(requests):
            _send_ok_response(router, request)

        if conn.recv() != "stop":
            raise ValueError("Unexpected reordering server command")
    finally:
        router.close(linger=0)
        context.term()
        conn.close()


def _run_delayed_response_server(conn) -> None:
    context = zmq.Context()  # type: ignore[attr-defined]
    router = context.socket(zmq.ROUTER)  # type: ignore[attr-defined]
    port = router.bind_to_random_port("tcp://127.0.0.1")

    try:
        conn.send(f"tcp://127.0.0.1:{port}")

        delayed_request = router.recv_multipart()
        conn.send("request_received")

        if conn.recv() != "send_late_response":
            raise ValueError("Unexpected delayed response server command")

        _send_ok_response(router, delayed_request)
        conn.send("late_response_sent")

        ping_request = router.recv_multipart()
        _send_ok_response(router, ping_request, (b"OK",))
        conn.send("completed")

        if conn.recv() != "stop":
            raise ValueError("Unexpected delayed response server stop command")
    finally:
        router.close(linger=0)
        context.term()
        conn.close()


def _run_hanging_server(conn) -> None:
    context = zmq.Context()  # type: ignore[attr-defined]
    router = context.socket(zmq.ROUTER)  # type: ignore[attr-defined]
    port = router.bind_to_random_port("tcp://127.0.0.1")

    try:
        conn.send(f"tcp://127.0.0.1:{port}")
        router.recv_multipart()
        conn.send("request_received")

        if conn.recv() != "stop":
            raise ValueError("Unexpected hanging server command")
    finally:
        router.close(linger=0)
        context.term()
        conn.close()


def test_protocol_round_trip():
    deadline_ns = time.monotonic_ns() + 1_000_000_000
    request_frames = encode_request(b"request-1", SystemMethod.ECHO, (b"payload",), deadline_ns)
    assert decode_request(request_frames) == (b"request-1", "ECHO", deadline_ns, (b"payload",))

    response_frames = encode_response(b"request-1", SystemMethod.ECHO, ResponseStatus.OK, (b"response",))
    assert decode_response(response_frames) == (
        b"request-1",
        "ECHO",
        ResponseStatus.OK,
        (b"response",),
    )


def test_protocol_rejects_invalid_request_deadline() -> None:
    with pytest.raises(MPProtocolError, match="Invalid request deadline"):
        decode_request((b"request-1", b"ECHO", b"invalid"))


def test_protocol_rejects_boolean_request_deadline() -> None:
    with pytest.raises(TypeError, match="deadline_ns must be an integer, got bool"):
        encode_request(b"request-1", SystemMethod.ECHO, deadline_ns=True)


def test_protocol_rejects_invalid_response_status() -> None:
    with pytest.raises(TypeError, match="status must be ResponseStatus, got str"):
        encode_response(b"request-1", SystemMethod.ECHO, "OK")


def test_client_server_round_trip():
    process, parent_conn, endpoint = _start_server(_run_server)
    client = None

    try:
        client = MPClient(endpoint)
        client.wait_until_connected()

        assert client.is_transport_connected
        assert client._lifecycle_state is _ClientState.CONNECTED
        assert client.ping() == "OK"
        assert client.echo(b"hello ascend store") == b"hello ascend store"

        with pytest.raises(MPRemoteError, match="ECHO expects 1 payload"):
            client.request(SystemMethod.ECHO, [])

        assert process.is_alive()
        assert client.ping() == "OK"
        client.close()
        assert client._lifecycle_state is _ClientState.CLOSED
    finally:
        _cleanup(client, parent_conn, process)


def test_client_close_wakes_thread_waiting_for_connection() -> None:
    endpoint = f"ipc:///tmp/ascend-store-mp-{uuid.uuid4().hex}"
    client = MPClient(endpoint)
    wait_started = threading.Event()

    def wait_until_connected() -> None:
        wait_started.set()
        client.wait_until_connected(timeout_ms=5000)

    try:
        assert client._lifecycle_state is _ClientState.DISCONNECTED
        with pytest.raises(MPServerUnavailableError, match="MP server is unavailable"):
            client.submit_request(SystemMethod.PING)

        with ThreadPoolExecutor(max_workers=1) as executor:
            wait_future = executor.submit(wait_until_connected)
            assert wait_started.wait(1)
            client.close()

            with pytest.raises(MPClientClosedError, match="MP client is closed"):
                wait_future.result(timeout=1)

        assert client._lifecycle_state is _ClientState.CLOSED
    finally:
        client.close()


def test_client_constructor_releases_resources_after_io_start_failure() -> None:
    context = MagicMock()
    context.socket.side_effect = RuntimeError("socket failed")

    with (
        patch(f"{RPC_CLIENT_MODULE}.zmq.Context", return_value=context),
        pytest.raises(RuntimeError, match="Failed to start MP client I/O thread") as exc_info,
    ):
        MPClient("tcp://127.0.0.1:12345")

    assert str(exc_info.value.__cause__) == "socket failed"
    context.term.assert_called_once_with()


def test_server_request_stop_wakes_run_loop() -> None:
    server = MPServer("tcp://127.0.0.1:*")
    server_thread = threading.Thread(target=server.run)

    try:
        server_thread.start()
        server.request_stop()
        server_thread.join(timeout=5)

        assert not server_thread.is_alive()
        assert server._socket.closed
        with pytest.raises(RuntimeError, match="can only be called once"):
            server.run()
    finally:
        server.close()


def test_server_run_honors_stop_requested_before_start() -> None:
    server = MPServer("tcp://127.0.0.1:*")
    server_thread = threading.Thread(target=server.run)

    try:
        assert server.request_stop()
        server_thread.start()
        server_thread.join(timeout=5)

        assert not server_thread.is_alive()
    finally:
        server.close()


def test_server_close_before_run_is_idempotent() -> None:
    server = MPServer("tcp://127.0.0.1:*")

    assert server.close()
    assert server.close()
    assert server._socket.closed
    with pytest.raises(RuntimeError, match="MPServer is closed"):
        server.run()


def test_server_abort_wins_after_drain_before_close() -> None:
    server = MPServer("tcp://127.0.0.1:*")
    wait_for_drain = server.wait_for_drain

    def abort_after_drain() -> bool:
        assert wait_for_drain()
        server.abort()
        return True

    server.wait_for_drain = abort_after_drain

    assert not server.close()
    assert server._socket.closed


def test_server_constructor_closes_executor_after_route_validation_failure() -> None:
    executor = MagicMock()
    routes = (
        Route(UPPERCASE_METHOD, _uppercase_handler, executor),
        Route(UPPERCASE_METHOD, _uppercase_handler, executor),
    )

    with pytest.raises(ValueError, match="Duplicate RPC method"):
        MPServer("tcp://127.0.0.1:*", routes)

    executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)


def test_server_constructor_closes_valid_executor_when_another_route_is_malformed() -> None:
    executor = MagicMock()
    routes = (Route(UPPERCASE_METHOD, _uppercase_handler, executor), object())

    with pytest.raises(TypeError, match="routes must contain Route instances"):
        MPServer("tcp://127.0.0.1:*", routes)

    executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)


def test_server_rejects_requests_after_stop_is_requested() -> None:
    handler = MagicMock(return_value=(b"response",))
    server = MPServer("tcp://127.0.0.1:*", routes=(Route(UPPERCASE_METHOD, handler, InlineExecutor()),))

    try:
        server.request_stop()
        server._dispatch_request(b"client", b"request-0", UPPERCASE_METHOD, (b"request",))

        handler.assert_not_called()
        identity, *response_frames = server._completed_response_queue.get_nowait().frames
        _, method, status, responses = decode_response(response_frames)
        assert identity == b"client"
        assert method == UPPERCASE_METHOD
        assert status is ResponseStatus.BUSY
        assert responses == (b"MPServerBusyError: MP server is stopping",)
    finally:
        server.close()


def test_server_reports_duplicate_in_flight_request() -> None:
    handler = MagicMock(return_value=(b"response",))
    server = MPServer("tcp://127.0.0.1:*", routes=(Route(UPPERCASE_METHOD, handler, InlineExecutor()),))

    try:
        server._dispatch_request(b"client", b"request-0", UPPERCASE_METHOD, ())
        server._dispatch_request(b"client", b"request-0", UPPERCASE_METHOD, ())

        server._completed_response_queue.get_nowait()
        identity, *response_frames = server._completed_response_queue.get_nowait().frames
        _, method, status, responses = decode_response(response_frames)
        assert handler.call_count == 1
        assert identity == b"client"
        assert method == UPPERCASE_METHOD
        assert status is ResponseStatus.BUSY
        assert responses == (b"MPServerBusyError: Duplicate in-flight MP request",)
    finally:
        server.abort()


def test_server_returns_handler_base_exception_without_leaking_request() -> None:
    def fail(_payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        raise KeyboardInterrupt("handler interrupted")

    server = MPServer("tcp://127.0.0.1:*", routes=(Route(UPPERCASE_METHOD, fail, InlineExecutor()),))

    try:
        server._dispatch_request(b"client", b"request-0", UPPERCASE_METHOD, ())

        response = server._completed_response_queue.get_nowait()
        identity, *response_frames = response.frames
        _, method, status, responses = decode_response(response_frames)
        assert identity == b"client"
        assert method == UPPERCASE_METHOD
        assert status is ResponseStatus.ERROR
        assert responses[0] == b"KeyboardInterrupt: handler interrupted"
        assert b"Traceback (most recent call last)" in responses[1]
        assert b'raise KeyboardInterrupt("handler interrupted")' in responses[1]

        server._send_backlog.append(response)
        socket = server._socket
        server._socket = MagicMock()
        server._send_responses()
        server._socket = socket
        assert server._accepted_requests == {}
    finally:
        server.close()


def test_server_returns_executor_failure() -> None:
    failed_future: Future[list[bytes]] = Future()
    failed_future.set_exception(RuntimeError("executor failed"))
    executor = MagicMock()
    executor.submit.return_value = failed_future
    server = MPServer("tcp://127.0.0.1:*", routes=(Route(UPPERCASE_METHOD, _uppercase_handler, executor),))

    try:
        server._dispatch_request(b"client", b"request-0", UPPERCASE_METHOD, (b"request",))

        _, *response_frames = server._completed_response_queue.get_nowait().frames
        _, method, status, responses = decode_response(response_frames)
        assert method == UPPERCASE_METHOD
        assert status is ResponseStatus.ERROR
        assert responses[0] == b"RuntimeError: executor failed"
        assert b"Traceback (most recent call last)" in responses[1]
        assert b"future.result()" in responses[1]
    finally:
        server.abort()


def test_server_close_drains_running_handler_and_returns_response() -> None:
    handler_started = threading.Event()
    queued_request_submitted = threading.Event()
    release_handler = threading.Event()

    def blocking_handler(payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        handler_started.set()
        if not release_handler.wait(5):
            raise TimeoutError("Timed out waiting to release the handler")
        return payloads

    executor = BoundedThreadPoolExecutor(1, 1, "test-server-close")
    original_submit = executor.submit
    submitted_requests = 0

    def tracking_submit(*args, **kwargs):
        nonlocal submitted_requests
        future = original_submit(*args, **kwargs)
        submitted_requests += 1
        if submitted_requests == 2:
            queued_request_submitted.set()
        return future

    executor.submit = tracking_submit
    server = MPServer("tcp://127.0.0.1:*", routes=(Route(BLOCKING_METHOD, blocking_handler, executor),))
    server_thread = threading.Thread(target=server.run)
    close_thread = threading.Thread(target=server.close)
    client = MPClient(server.endpoint)

    try:
        server_thread.start()
        client.wait_until_connected()
        running_future = client.submit_request(BLOCKING_METHOD, [b"running"], timeout_ms=5000)
        assert handler_started.wait(5), "Handler did not start in time"
        queued_future = client.submit_request(BLOCKING_METHOD, [b"queued"], timeout_ms=5000)
        assert queued_request_submitted.wait(5), "Queued request was not submitted in time"

        close_thread.start()
        assert close_thread.is_alive()
        assert not running_future.done()
        assert not queued_future.done()
        assert not server.wait_until_stopped(timeout=0.1)

        release_handler.set()
        assert running_future.result(timeout=5) == [b"running"]
        assert queued_future.result(timeout=5) == [b"queued"]
    finally:
        release_handler.set()
        if close_thread.ident is None:
            close_thread.start()
        close_thread.join(timeout=5)
        server_thread.join(timeout=5)
        client.close()
        server.close()

    assert not close_thread.is_alive()
    assert not server_thread.is_alive()


def test_server_abort_cancels_queued_requests_without_waiting_for_running_handler() -> None:
    handler_started = threading.Event()
    queued_request_submitted = threading.Event()
    release_handler = threading.Event()

    def blocking_handler(payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        handler_started.set()
        if not release_handler.wait(5):
            raise TimeoutError("Timed out waiting to release the handler")
        return payloads

    executor = BoundedThreadPoolExecutor(1, 1, "test-server-abort")
    original_submit = executor.submit
    submitted_requests = 0

    def tracking_submit(*args, **kwargs):
        nonlocal submitted_requests
        future = original_submit(*args, **kwargs)
        submitted_requests += 1
        if submitted_requests == 2:
            queued_request_submitted.set()
        return future

    executor.submit = tracking_submit
    server = MPServer("tcp://127.0.0.1:*", routes=(Route(BLOCKING_METHOD, blocking_handler, executor),))
    server_thread = threading.Thread(target=server.run)
    client = MPClient(server.endpoint)

    try:
        server_thread.start()
        client.wait_until_connected()
        running_future = client.submit_request(BLOCKING_METHOD, [b"running"])
        assert handler_started.wait(5), "Handler did not start in time"
        queued_future = client.submit_request(BLOCKING_METHOD, [b"queued"])
        assert queued_request_submitted.wait(5), "Queued request was not submitted in time"

        assert not server.request_stop()
        assert client.ping(timeout_ms=2000) == "OK"
        server.abort()
        server_thread.join(timeout=5)

        assert not server_thread.is_alive()
        assert not release_handler.is_set()
        with pytest.raises(MPServerAbortedError, match="force-aborted"):
            running_future.result(timeout=5)
        with pytest.raises(MPServerAbortedError, match="force-aborted"):
            queued_future.result(timeout=5)
    finally:
        release_handler.set()
        server_thread.join(timeout=5)
        client.close()
        server.abort()


def test_client_backpressure_does_not_leave_an_unsent_request_pending() -> None:
    client = MPClient.__new__(MPClient)
    client._outbound_queue = queue.Queue()
    client._pending_requests = {}
    future: Future[list[bytes]] = Future()
    client._outbound_queue.put(
        SimpleNamespace(
            request_id=b"request-0",
            method="TEST",
            frames=(b"request-0", b"TEST"),
            future=future,
            deadline_ns=None,
        )
    )
    zmq_socket = MagicMock()
    zmq_socket.send_multipart.side_effect = zmq.Again()  # type: ignore[attr-defined]

    client._process_outbound(zmq_socket)

    with pytest.raises(MPServerBusyError, match="outbound transport is busy"):
        future.result()
    assert client._pending_requests == {}


def test_client_transport_send_failure_completes_current_request() -> None:
    client = MPClient.__new__(MPClient)
    client._outbound_queue = queue.Queue()
    client._pending_requests = {}
    future: Future[list[bytes]] = Future()
    client._outbound_queue.put(
        SimpleNamespace(
            request_id=b"request-0",
            method="TEST",
            frames=(b"request-0", b"TEST"),
            future=future,
            deadline_ns=None,
        )
    )
    transport_error = RuntimeError("transport send failed")
    zmq_socket = MagicMock()
    zmq_socket.send_multipart.side_effect = transport_error

    with pytest.raises(MPServerUnavailableError, match="I/O thread is unavailable") as exc_info:
        client._process_outbound(zmq_socket)

    assert future.exception() is exc_info.value
    assert exc_info.value.__cause__ is transport_error
    assert client._outbound_queue.empty()
    assert client._pending_requests == {}


def test_server_backpressure_retains_response_without_blocking() -> None:
    server = MPServer.__new__(MPServer)
    response = (b"client", b"request-0", b"TEST")
    response_envelope = SimpleNamespace(frames=response, request_key=None)
    server._send_backlog = deque((response_envelope,))
    server._socket = MagicMock()
    server._socket.send_multipart.side_effect = zmq.Again()  # type: ignore[attr-defined]

    server._send_responses()

    assert server._send_backlog == deque((response_envelope,))


def test_server_close_rejects_request_without_deadline() -> None:
    handler_started = threading.Event()
    release_handler = threading.Event()

    def blocking_handler(payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        handler_started.set()
        release_handler.wait()
        return payloads

    executor = BoundedThreadPoolExecutor(1, 0, "test-server-indefinite-close")
    server = MPServer("tcp://127.0.0.1:*", routes=(Route(BLOCKING_METHOD, blocking_handler, executor),))
    server_thread = threading.Thread(target=server.run)
    client = MPClient(server.endpoint)

    try:
        server_thread.start()
        client.wait_until_connected()
        request_future = client.submit_request(BLOCKING_METHOD, [b"request"])
        assert handler_started.wait(5), "Handler did not start in time"

        assert not server.close()
        assert client.ping(timeout_ms=2000) == "OK"
        assert not request_future.done()
    finally:
        server.abort()
        release_handler.set()
        server_thread.join(timeout=5)
        client.close()


def test_server_close_returns_false_after_request_deadline() -> None:
    handler_started = threading.Event()
    release_handler = threading.Event()

    def blocking_handler(payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        handler_started.set()
        release_handler.wait()
        return payloads

    executor = BoundedThreadPoolExecutor(1, 0, "test-server-expired-close")
    server = MPServer("tcp://127.0.0.1:*", routes=(Route(BLOCKING_METHOD, blocking_handler, executor),))
    server_thread = threading.Thread(target=server.run)
    client = MPClient(server.endpoint)

    try:
        server_thread.start()
        client.wait_until_connected()
        request_future = client.submit_request(BLOCKING_METHOD, [b"request"], timeout_ms=500)
        assert handler_started.wait(5), "Handler did not start in time"

        assert not server.close()
        with pytest.raises(MPServerBusyError, match="MP server is stopping"):
            client.ping(timeout_ms=2000)
        with pytest.raises(MPRequestTimeoutError):
            request_future.result(timeout=5)
    finally:
        server.abort()
        release_handler.set()
        server_thread.join(timeout=5)
        client.close()


def test_multiple_worker_processes_receive_their_own_responses():
    server_process, server_conn, endpoint = _start_server(_run_server)
    context = mp.get_context("fork")
    start_event = context.Event()
    worker_processes = []
    worker_conns = []

    try:
        for worker_id in range(_WORKER_COUNT):
            parent_conn, child_conn = context.Pipe()
            process = context.Process(
                target=_run_client_worker,
                args=(endpoint, worker_id, start_event, child_conn),
            )
            process.start()
            child_conn.close()
            worker_processes.append(process)
            worker_conns.append(parent_conn)

        for worker_id, conn in enumerate(worker_conns):
            assert conn.poll(5), f"Worker {worker_id} did not connect in time"
            assert conn.recv() == ("ready", worker_id)

        start_event.set()

        for worker_id, conn in enumerate(worker_conns):
            assert conn.poll(10), f"Worker {worker_id} did not finish in time"
            message = conn.recv()
            assert message[0] == "result", message

            _, returned_worker_id, responses = message
            expected = [
                f"worker-{worker_id}-request-{request_id}".encode() for request_id in range(_REQUESTS_PER_WORKER)
            ]
            assert returned_worker_id == worker_id
            assert responses == expected

        for process in worker_processes:
            process.join(timeout=5)
            assert not process.is_alive()
            assert process.exitcode == 0
    finally:
        start_event.set()

        for conn in worker_conns:
            conn.close()

        for process in worker_processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)

        _cleanup(None, server_conn, server_process)


def test_server_uses_injected_handlers():
    process, parent_conn, endpoint = _start_server(_run_server_with_injected_handlers)
    client = None

    try:
        client = MPClient(endpoint)
        client.wait_until_connected()

        assert client.request(UPPERCASE_METHOD, [b"hello ascend store"]) == [b"HELLO ASCEND STORE"]

        with pytest.raises(MPRemoteError, match="Payloads must be an iterable of bytes") as exc_info:
            client.request(INVALID_RESPONSE_METHOD)
        assert exc_info.value.remote_method == INVALID_RESPONSE_METHOD
        assert exc_info.value.remote_request_id is not None
        assert exc_info.value.remote_traceback is not None
        assert "_execute_handler" in exc_info.value.remote_traceback
        assert "_normalize_payloads" in exc_info.value.remote_traceback
        assert "Remote traceback:" in str(exc_info.value)

        first_worker = client.request(AFFINITY_METHOD, [b"0"])
        assert client.request(AFFINITY_METHOD, [b"0"]) == first_worker
        assert client.request(AFFINITY_METHOD, [b"1"]) != first_worker

        default_worker = client.request(DEFAULT_AFFINITY_METHOD, [b"0"])
        assert client.request(DEFAULT_AFFINITY_METHOD, [b"1"]) == default_worker

        assert process.is_alive()
        assert client.ping() == "OK"
    finally:
        _cleanup(client, parent_conn, process)


def test_server_returns_busy_when_executor_is_at_capacity():
    process, parent_conn, endpoint = _start_server(_run_bounded_server)
    client = None
    handler_released = False

    try:
        client = MPClient(endpoint)
        client.wait_until_connected()
        first_future = client.submit_request(BLOCKING_METHOD, [b"first"])

        assert parent_conn.poll(5), "Handler did not start in time"
        assert parent_conn.recv() == "handler_started"

        with pytest.raises(MPServerBusyError, match="Parallel executor is at capacity"):
            client.request(BLOCKING_METHOD, [b"second"])

        parent_conn.send("release_handler")
        handler_released = True
        assert first_future.result(timeout=5) == [b"first"]
    finally:
        if process.is_alive() and not handler_released:
            parent_conn.send("release_handler")
        _cleanup(client, parent_conn, process)


def test_async_out_of_order_responses():
    process, parent_conn, endpoint = _start_server(_run_reordering_server)
    client = None

    try:
        client = MPClient(endpoint)
        client.wait_until_connected()

        futures = [
            client.submit_request(SystemMethod.ECHO, [b"0"]),
            client.submit_request(SystemMethod.ECHO, [b"1"]),
            client.submit_request(SystemMethod.ECHO, [b"2"]),
        ]

        assert [future.result(timeout=5) for future in futures] == [[b"0"], [b"1"], [b"2"]]

        parent_conn.send("stop")
        process.join(timeout=5)
        assert not process.is_alive()
        assert process.exitcode == 0
    finally:
        _cleanup(client, parent_conn, process)


def test_request_timeout_discards_late_response():
    process, parent_conn, endpoint = _start_server(_run_delayed_response_server)
    client = None

    try:
        client = MPClient(endpoint)
        client.wait_until_connected()

        future = client.submit_request(SystemMethod.ECHO, [b"late"], timeout_ms=500)

        assert parent_conn.poll(5), "Server did not receive request in time"
        assert parent_conn.recv() == "request_received"

        with pytest.raises(MPRequestTimeoutError, match="Timed out waiting for response"):
            future.result(timeout=5)

        parent_conn.send("send_late_response")
        assert parent_conn.poll(5), "Server did not send late response in time"
        assert parent_conn.recv() == "late_response_sent"

        assert client.ping(timeout_ms=2000) == "OK"

        with pytest.raises(MPRequestTimeoutError, match="Timed out waiting for response"):
            future.result()

        assert parent_conn.poll(5), "Server did not finish requests in time"
        assert parent_conn.recv() == "completed"

        parent_conn.send("stop")
        process.join(timeout=5)
        assert not process.is_alive()
        assert process.exitcode == 0
    finally:
        _cleanup(client, parent_conn, process)


def test_dispatched_request_cannot_be_cancelled():
    process, parent_conn, endpoint = _start_server(_run_hanging_server)
    client = None

    try:
        client = MPClient(endpoint)
        client.wait_until_connected()

        future = client.submit_request(SystemMethod.ECHO, [b"never-return"])

        assert parent_conn.poll(5), "Server did not receive request in time"
        assert parent_conn.recv() == "request_received"
        assert future.running()
        assert not future.cancel()

        client.close()

        with pytest.raises(MPClientClosedError, match="MP client was closed"):
            future.result()

        with pytest.raises(MPClientClosedError, match="MP client is closed"):
            client.submit_request(SystemMethod.PING)

        parent_conn.send("stop")
        process.join(timeout=5)
        assert not process.is_alive()
        assert process.exitcode == 0
    finally:
        _cleanup(client, parent_conn, process)


def test_bounded_thread_pool_rejects_excess_work():
    started = threading.Event()
    release = threading.Event()
    executor = BoundedThreadPoolExecutor(1, 0, "test-bounded-pool")

    def wait_for_release() -> None:
        started.set()
        if not release.wait(5):
            raise TimeoutError("Timed out waiting to release the executor")

    try:
        first_future = executor.submit(wait_for_release)
        assert started.wait(5)

        with pytest.raises(MPServerBusyError, match="Parallel executor is at capacity"):
            executor.submit(wait_for_release)

        release.set()
        first_future.result(timeout=5)
    finally:
        release.set()
        executor.shutdown(wait=True, cancel_futures=True)


def test_affinity_executor_serializes_same_key_on_same_thread():
    started = threading.Event()
    release = threading.Event()
    execution_order = []
    worker_threads = []
    executor = AffinityExecutor(2, 2, "test-affinity")

    def record_execution(index: int) -> int:
        if index == 0:
            started.set()
            if not release.wait(5):
                raise TimeoutError("Timed out waiting to release the affinity executor")
        execution_order.append(index)
        worker_threads.append(threading.get_ident())
        return index

    try:
        futures = [executor.submit(partial(record_execution, 0), "engine-0")]
        assert started.wait(5)
        futures.extend(executor.submit(partial(record_execution, index), "engine-0") for index in (1, 2))

        release.set()
        assert [future.result(timeout=5) for future in futures] == [0, 1, 2]
        assert execution_order == [0, 1, 2]
        assert len(set(worker_threads)) == 1
    finally:
        release.set()
        executor.shutdown(wait=True, cancel_futures=True)


def test_affinity_executor_runs_different_keys_in_parallel():
    barrier = threading.Barrier(2)
    executor = AffinityExecutor(2, 0, "test-affinity")

    def wait_for_peer() -> int:
        barrier.wait(timeout=5)
        return threading.get_ident()

    try:
        first_future = executor.submit(wait_for_peer, 0)
        second_future = executor.submit(wait_for_peer, 1)
        assert first_future.result(timeout=5) != second_future.result(timeout=5)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def test_affinity_executor_can_wait_for_capacity() -> None:
    started = threading.Event()
    release = threading.Event()
    submit_started = threading.Event()
    executor = AffinityExecutor(1, 0, "test-affinity")

    def wait_for_release() -> None:
        started.set()
        assert release.wait(5), "Timed out waiting to release the affinity executor"

    def submit_waiting_task():
        submit_started.set()
        return executor.submit(lambda: None, "engine-0", block=True)

    try:
        first_future = executor.submit(wait_for_release, "engine-0")
        assert started.wait(5)
        with ThreadPoolExecutor(max_workers=1) as submitter:
            waiting_submission = submitter.submit(submit_waiting_task)
            assert submit_started.wait(5)
            with pytest.raises(TimeoutError):
                waiting_submission.result(timeout=0.1)
            release.set()
            waiting_submission.result(timeout=5).result(timeout=5)
        first_future.result(timeout=5)
    finally:
        release.set()
        executor.shutdown(wait=True, cancel_futures=True)


def test_affinity_executor_shutdown_does_not_wait_for_blocked_submission() -> None:
    started = threading.Event()
    release = threading.Event()
    submit_started = threading.Event()
    executor = AffinityExecutor(1, 0, "test-affinity-shutdown")

    def wait_for_release() -> None:
        started.set()
        release.wait()

    def submit_waiting_task() -> None:
        submit_started.set()
        executor.submit(lambda: None, "engine-0", block=True)

    try:
        first_future = executor.submit(wait_for_release, "engine-0")
        assert started.wait(5)
        with ThreadPoolExecutor(max_workers=1) as submitter:
            waiting_submission = submitter.submit(submit_waiting_task)
            assert submit_started.wait(5)

            shutdown_thread = threading.Thread(target=executor.shutdown, kwargs={"wait": False})
            shutdown_thread.start()
            shutdown_thread.join(timeout=1)
            assert not shutdown_thread.is_alive()

            release.set()
            first_future.result(timeout=5)
            with pytest.raises(RuntimeError, match="Affinity executor is closed"):
                waiting_submission.result(timeout=5)
    finally:
        release.set()
        executor.shutdown(wait=True, cancel_futures=True)


def test_inline_executor_runs_task_in_submitting_thread() -> None:
    executor = InlineExecutor()
    submitting_thread = threading.get_ident()

    future = executor.submit(threading.get_ident)

    assert future.result() == submitting_thread


def test_inline_executor_captures_task_failure() -> None:
    executor = InlineExecutor()

    def fail() -> None:
        raise RuntimeError("inline task failed")

    future = executor.submit(fail)

    with pytest.raises(RuntimeError, match="inline task failed"):
        future.result()


def test_affinity_executor_requires_affinity_key() -> None:
    executor = AffinityExecutor(1, 0, "test-affinity")

    try:
        with pytest.raises(ValueError, match="must define an affinity key"):
            executor.submit(lambda: None)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
