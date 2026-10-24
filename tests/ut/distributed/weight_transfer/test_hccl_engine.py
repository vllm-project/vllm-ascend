# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLWeightTransferEngine, HCCLWeightTransferInitInfo


def _make_engine():
    engine = object.__new__(HCCLWeightTransferEngine)
    engine.model = MagicMock()
    engine.model_config = MagicMock()
    engine.device = torch.device("cpu")
    return engine


def test_start_weight_update_initializes_layerwise_reload():
    engine = _make_engine()

    with patch("vllm.model_executor.model_loader.reload.initialize_layerwise_reload") as initialize:
        engine.start_weight_update()

    initialize.assert_called_once_with(engine.model)


def test_finish_weight_update_finalizes_layerwise_reload():
    engine = _make_engine()

    with patch("vllm.model_executor.model_loader.reload.finalize_layerwise_reload") as finalize:
        engine.finish_weight_update()

    finalize.assert_called_once_with(engine.model, engine.model_config)


def test_shutdown_closes_session_once():
    engine = _make_engine()
    group = MagicMock()
    engine.model_update_group = group

    engine.shutdown()
    engine.shutdown()

    group.close.assert_called_once_with()
    assert engine.model_update_group is None


@patch("torch.accelerator.current_device_index", return_value=0)
def test_reinit_closes_previous_session_before_creating_next(_mock_device):
    engine = _make_engine()
    engine.parallel_config = SimpleNamespace(data_parallel_index=0, world_size=1, rank=0)
    old_group = MagicMock()
    engine.model_update_group = old_group
    init_info = HCCLWeightTransferInitInfo(master_address="127.0.0.1", master_port=12345, rank_offset=1, world_size=2)

    def create_group(*args, **kwargs):
        old_group.close.assert_called_once_with()
        assert engine.model_update_group is None
        return MagicMock()

    with patch.object(HCCLWeightTransferEngine, "_stateless_init_process_group", side_effect=create_group) as init:
        engine.init_transfer_engine(init_info)

    init.assert_called_once_with("127.0.0.1", 12345, 1, 2, device=engine.device)
    assert engine.model_update_group is not old_group


@pytest.mark.parametrize("receive_fails", [False, True])
def test_update_binds_assigned_device_through_receive_and_final_sync(receive_fails):
    engine = _make_engine()
    events = []
    payload = {"weights": True}
    typed = object()
    engine.parse_update_info = MagicMock(return_value=typed)

    @contextmanager
    def device_context(device):
        assert device == engine.device
        events.append("bound")
        try:
            yield
        finally:
            events.append("restored")

    def receive(info):
        assert info is typed
        assert events == ["bound"]
        events.append("receive")
        if receive_fails:
            raise RuntimeError("receive failed")

    def synchronize():
        assert events == ["bound", "receive"]
        events.append("sync")

    engine.receive_weights = receive
    with (
        patch.object(torch, "npu", SimpleNamespace(device=device_context), create=True),
        patch.object(torch.accelerator, "synchronize", side_effect=synchronize),
    ):
        if receive_fails:
            with pytest.raises(RuntimeError, match="receive failed"):
                engine.update_weights(payload)
        else:
            engine.update_weights(payload)
    engine.parse_update_info.assert_called_once_with(payload)
    assert events == (["bound", "receive", "restored"] if receive_fails else ["bound", "receive", "sync", "restored"])
