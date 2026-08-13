# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from unittest import mock

import pytest
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

import vllm_ascend.patch.platform.patch_shm_broadcast as patch


@pytest.mark.parametrize("should_warn", [False, True])
def test_reader_timeout_caps_indefinite_waits(monkeypatch, should_warn):
    monkeypatch.setattr(patch, "SHM_READER_RECHECK_INTERVAL_MS", 7)
    timeout = MessageQueue.ReadTimeoutWithWarnings(timeout=None, should_warn=should_warn)
    assert timeout.timeout_ms() == 7


def test_reader_rechecks_shm_after_lost_notify(monkeypatch):
    monkeypatch.setattr(patch, "SHM_READER_RECHECK_INTERVAL_MS", 50)
    writer = MessageQueue(
        n_reader=1,
        n_local_reader=1,
        max_chunk_bytes=1024 * 1024,
        max_chunks=1,
    )
    reader = MessageQueue.create_from_handle(writer.export_handle(), rank=0)
    poll_started = threading.Event()
    allow_timeout = threading.Event()
    result = {}

    def acquire_read():
        try:
            with reader.acquire_read(indefinite=True) as buf:
                result["value"] = buf[0]
        except Exception as exc:
            result["exception"] = exc

    def poll_timeout(*, timeout: int | None = None):
        poll_started.set()
        assert allow_timeout.wait(timeout=5)
        return []

    try:
        writer.wait_until_ready()
        reader.wait_until_ready()
        reader._spin_condition.last_read = 0
        reader._spin_condition.busy_loop_s = 0

        with mock.patch.object(
            reader._spin_condition.poller,
            "poll",
            side_effect=poll_timeout,
        ) as poll:
            read_thread = threading.Thread(target=acquire_read, daemon=True)
            read_thread.start()
            assert poll_started.wait(timeout=5)
            with writer.acquire_write(timeout=0.1) as buf:
                buf[0] = 123
            allow_timeout.set()
            read_thread.join(timeout=5)

            assert not read_thread.is_alive()
            poll.assert_called_once_with(timeout=50)

        if exception := result.get("exception"):
            raise exception
        assert result["value"] == 123
    finally:
        writer.shutdown()
        reader.shutdown()
        for socket in (
            writer.local_socket,
            writer._spin_condition.local_notify_socket,
            reader.local_socket,
            reader._spin_condition.local_notify_socket,
            reader._spin_condition.read_cancel_socket,
            reader._spin_condition.write_cancel_socket,
        ):
            socket.close(linger=0)


def test_acquire_read_releases_slot_when_reader_raises():
    writer = MessageQueue(
        n_reader=1,
        n_local_reader=1,
        max_chunk_bytes=1024 * 1024,
        max_chunks=1,
    )
    reader = MessageQueue.create_from_handle(writer.export_handle(), rank=0)
    try:
        writer.wait_until_ready()
        reader.wait_until_ready()
        writer.enqueue({"payload": "first"})

        with (
            pytest.raises(RuntimeError, match="reader failed"),
            reader.acquire_read(timeout=0.1),
        ):
            raise RuntimeError("reader failed")

        with writer.buffer.get_metadata(0) as metadata_buffer:
            assert metadata_buffer[0] == 1
            assert metadata_buffer[1] == 1

        with writer.acquire_write(timeout=0.1) as buf:
            buf[0] = 0
    finally:
        writer.shutdown()
        reader.shutdown()
        for socket in (
            writer.local_socket,
            writer._spin_condition.local_notify_socket,
            reader.local_socket,
            reader._spin_condition.local_notify_socket,
            reader._spin_condition.read_cancel_socket,
            reader._spin_condition.write_cancel_socket,
        ):
            socket.close(linger=0)
