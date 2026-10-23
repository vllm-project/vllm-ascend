# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from zmq.constants import Flag

from vllm_ascend.patch.platform import patch_shm_notify


def test_fast_reads_service_notifications_without_polling_every_read():
    sock = MagicMock()
    reader = SimpleNamespace(local_notify_socket=sock, last_read=10.0)
    context = MagicMock()
    with patch.object(patch_shm_notify, "_original_init") as original_init:
        patch_shm_notify.init(reader, True, context, "ipc://notify")
        original_init.assert_called_once_with(reader, True, context, "ipc://notify", 1)
    assert reader.last_notify_poll == 10.0
    with patch.object(patch_shm_notify.time, "monotonic", side_effect=[11.0, 12.0, 13.0, 14.0]):
        patch_shm_notify.record_read(reader)
        assert reader.last_read == 11.0
        sock.poll.assert_not_called()
        patch_shm_notify.record_read(reader)
        assert reader.last_notify_poll == 12.0
        patch_shm_notify.record_read(reader)
        assert sock.poll.call_count == 1
        patch_shm_notify.record_read(reader)
    assert reader.last_notify_poll == 14.0
    assert sock.poll.call_count == 2
    assert sock.recv.call_count == 2
    sock.poll.assert_called_with(timeout=0)
    sock.recv.assert_called_with(flags=Flag.DONTWAIT, copy=False)


def test_empty_notification_socket_does_not_block_shm_reads():
    sock = MagicMock()
    sock.poll.return_value = 0
    reader = SimpleNamespace(local_notify_socket=sock, last_notify_poll=8.0)
    with patch.object(patch_shm_notify.time, "monotonic", return_value=10.0):
        patch_shm_notify.record_read(reader)
    sock.poll.assert_called_once_with(timeout=0)
    sock.recv.assert_not_called()
    assert reader.last_read == reader.last_notify_poll == 10.0
