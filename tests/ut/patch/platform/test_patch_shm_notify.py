# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import zmq

from vllm_ascend.patch.platform import patch_shm_notify


def test_fast_reads_service_notifications_without_polling_every_read():
    sock = MagicMock()
    reader = SimpleNamespace(local_notify_socket=sock)
    with patch.object(patch_shm_notify.time, "monotonic", side_effect=[10.0, 10.0, 10.5, 11.0, 11.0]):
        patch_shm_notify.record_read(reader)
        patch_shm_notify.record_read(reader)
        assert reader.last_read == 10.5
        assert sock.poll.call_count == 1
        patch_shm_notify.record_read(reader)
    assert sock.poll.call_count == 2
    assert sock.recv.call_count == 2
    sock.poll.assert_called_with(timeout=0, flags=zmq.POLLIN)
    sock.recv.assert_called_with(flags=zmq.NOBLOCK, copy=False)


def test_empty_notification_socket_does_not_block_shm_reads():
    sock = MagicMock()
    sock.poll.return_value = 0
    reader = SimpleNamespace(local_notify_socket=sock)
    with patch.object(patch_shm_notify.time, "monotonic", return_value=10.0):
        patch_shm_notify.record_read(reader)
    sock.recv.assert_not_called()
    assert reader.last_read == 10.0


def test_notification_again_preserves_read_progress():
    sock = MagicMock()
    sock.recv.side_effect = zmq.Again()
    reader = SimpleNamespace(local_notify_socket=sock)
    with patch.object(patch_shm_notify.time, "monotonic", return_value=10.0):
        patch_shm_notify.record_read(reader)
    assert reader.last_read == 10.0
    assert reader._ascend_last_notify_poll == 10.0


def test_each_reader_services_its_own_notification_socket():
    first = SimpleNamespace(local_notify_socket=MagicMock())
    second = SimpleNamespace(local_notify_socket=MagicMock())
    with patch.object(patch_shm_notify.time, "monotonic", return_value=10.0):
        patch_shm_notify.record_read(first)
        patch_shm_notify.record_read(second)
    first.local_notify_socket.poll.assert_called_once()
    second.local_notify_socket.poll.assert_called_once()


def test_unexpected_socket_failure_is_not_hidden():
    sock = MagicMock()
    sock.recv.side_effect = zmq.ZMQError(zmq.ETERM)
    reader = SimpleNamespace(local_notify_socket=sock)
    with (
        patch.object(patch_shm_notify.time, "monotonic", return_value=10.0),
        pytest.raises(zmq.ZMQError) as error,
    ):
        patch_shm_notify.record_read(reader)
    assert error.value.errno == zmq.ETERM
