# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.simple.copy_backend import NPUDmaCopyBackend


def test_uninitialized_backend_rejects_submission_and_shutdown_is_idempotent():
    backend = NPUDmaCopyBackend()
    with pytest.raises(AssertionError):
        backend.launch_copy([0], [0], True, 1, [])
    backend.shutdown()
    backend.shutdown()
    assert backend._shutdown
    assert backend._thread is None


def test_fifo_store_and_load_copy_events_and_barriers(monkeypatch):
    swap = MagicMock()
    monkeypatch.setattr(torch.ops._C_ascend, "swap_blocks_batch", swap, raising=False)
    backend = NPUDmaCopyBackend()
    device = torch.device("cpu")
    load_stream, store_stream, barrier = MagicMock(), MagicMock(), MagicMock()
    source, destination = {"k": torch.zeros(3, 4)}, {"k": torch.zeros(3, 4)}
    events = []
    backend.init(source, destination, device, load_stream, store_stream)
    try:
        backend.launch_copy([0], [1], True, 11, events, barrier)
        backend.launch_copy([1], [2], False, 12, events)
    finally:
        backend.shutdown()
    assert not backend._thread.is_alive()
    assert [idx for idx, event in events] == [11, 12]
    assert [call.args[-1] for call in swap.call_args_list] == [1, 0]
    assert swap.call_args_list[0].args[0].tolist() == [source["k"].data_ptr()]
    assert swap.call_args_list[1].args[0].tolist() == [destination["k"].data_ptr() + 16]
    store_stream.wait_event.assert_called_once_with(barrier)
    load_stream.wait_event.assert_not_called()
    assert torch.npu.Event.return_value.record.call_args_list[0].args == (store_stream,)
    assert torch.npu.Event.return_value.record.call_args_list[1].args == (load_stream,)
    backend.shutdown()
