# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from zmq.constants import Flag, PollEvent

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
    sock.poll.assert_called_with(timeout=0, flags=PollEvent.POLLIN)
    sock.recv.assert_called_with(flags=Flag.DONTWAIT, copy=False)


def test_empty_notification_socket_does_not_block_shm_reads():
    sock = MagicMock()
    sock.poll.return_value = 0
    reader = SimpleNamespace(local_notify_socket=sock)
    with patch.object(patch_shm_notify.time, "monotonic", return_value=10.0):
        patch_shm_notify.record_read(reader)
    sock.recv.assert_not_called()
    assert reader.last_read == 10.0
