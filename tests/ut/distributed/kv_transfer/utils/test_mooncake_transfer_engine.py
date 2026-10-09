# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, call

from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import GlobalTE


def test_unregister_and_reregister_registered_buffers():
    global_te = GlobalTE()
    global_te.transfer_engine = MagicMock()
    global_te.transfer_engine.register_memory.return_value = 0
    global_te.transfer_engine.unregister_memory.return_value = 0
    global_te.transfer_engine.disconnect_all_peers.return_value = 0

    global_te.register_buffer([0x1000, 0x2000], [128, 256])
    global_te.unregister_buffer()

    assert not global_te.is_register_buffer
    assert global_te.transfer_engine.mock_calls[2:5] == [
        call.disconnect_all_peers(),
        call.unregister_memory(0x1000),
        call.unregister_memory(0x2000),
    ]
    assert global_te.transfer_engine.unregister_memory.call_count == 2

    global_te.reregister_buffer()

    assert global_te.is_register_buffer
    assert global_te.transfer_engine.register_memory.call_count == 4


def test_unregister_failure_rolls_back_completed_buffers():
    global_te = GlobalTE()
    global_te.transfer_engine = MagicMock()
    global_te.transfer_engine.register_memory.return_value = 0
    global_te.transfer_engine.unregister_memory.side_effect = [0, -1]
    global_te.transfer_engine.disconnect_all_peers.return_value = 0
    global_te.registered_buffers = [(0x1000, 128), (0x2000, 256)]
    global_te.is_register_buffer = True

    try:
        global_te.unregister_buffer()
    except RuntimeError as error:
        assert "ptr=0x2000" in str(error)
    else:
        raise AssertionError("unregister_buffer should fail")

    global_te.transfer_engine.register_memory.assert_called_once_with(0x1000, 128)
    assert global_te.is_register_buffer


def test_disconnect_failure_stops_memory_unregistration():
    global_te = GlobalTE()
    global_te.transfer_engine = MagicMock()
    global_te.transfer_engine.disconnect_all_peers.return_value = -1
    global_te.registered_buffers = [(0x1000, 128)]
    global_te.is_register_buffer = True

    try:
        global_te.unregister_buffer()
    except RuntimeError as error:
        assert "peer disconnection failed" in str(error)
    else:
        raise AssertionError("unregister_buffer should fail")

    global_te.transfer_engine.unregister_memory.assert_not_called()
    assert global_te.is_register_buffer


def test_disconnect_all_peers_is_noop_before_engine_initialization():
    GlobalTE().disconnect_all_peers()
