# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.distributed.weight_transfer.hccl_engine import (
    HCCLTrainerWeightTransferEngine,
    HCCLWeightTransferEngine,
    HCCLWeightTransferInitInfo,
)
from vllm_ascend.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    DEFAULT_PACKED_NUM_BUFFERS,
)


def _make_engine():
    engine = object.__new__(HCCLWeightTransferEngine)
    engine.model = MagicMock()
    engine.model_config = MagicMock()
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

    init.assert_called_once_with("127.0.0.1", 12345, 1, 2, device=0)
    assert engine.model_update_group is not old_group


@patch("torch.accelerator.current_device_index", return_value=0)
def test_reinit_restores_defaults_for_omitted_parameters(_mock_device):
    engine = _make_engine()
    engine.parallel_config = SimpleNamespace(data_parallel_index=0, world_size=1, rank=0)
    engine.model_update_group = MagicMock()

    init_infos = [
        (
            HCCLWeightTransferInitInfo(
                master_address="127.0.0.1",
                master_port=12345,
                rank_offset=1,
                world_size=2,
                packed=True,
                packed_buffer_size_bytes=64,
                packed_num_buffers=3,
            ),
            (True, 64, 3, True, True, True),
        ),
        (
            HCCLWeightTransferInitInfo(
                master_address="127.0.0.1",
                master_port=12346,
                rank_offset=1,
                world_size=2,
            ),
            (
                False,
                DEFAULT_PACKED_BUFFER_SIZE_BYTES,
                DEFAULT_PACKED_NUM_BUFFERS,
                False,
                False,
                False,
            ),
        ),
        (
            HCCLWeightTransferInitInfo(
                master_address="127.0.0.1",
                master_port=12347,
                rank_offset=1,
                world_size=2,
                packed=True,
                packed_num_buffers=4,
            ),
            (
                True,
                DEFAULT_PACKED_BUFFER_SIZE_BYTES,
                4,
                True,
                False,
                True,
            ),
        ),
    ]

    with patch.object(
        HCCLWeightTransferEngine,
        "_stateless_init_process_group",
        return_value=MagicMock(),
    ):
        for init_info, expected in init_infos:
            engine.init_transfer_engine(init_info)
            assert (
                engine.packed,
                engine.packed_buffer_size_bytes,
                engine.packed_num_buffers,
                engine._init_packed_explicit,
                engine._init_buffer_size_explicit,
                engine._init_num_buffers_explicit,
            ) == expected


@pytest.mark.parametrize(
    ("packed", "tensor"),
    [
        (False, None),
        (False, SimpleNamespace(dtype=torch.float32, shape=(3,))),
        (True, None),
        (True, SimpleNamespace(dtype=torch.float32, shape=(3,))),
    ],
)
def test_non_sender_drains_source_without_sender_validation(packed, tensor):
    metadata = [SimpleNamespace(name="weight", dtype=torch.float16, shape=(2,))]

    class RecordingSource:
        def __init__(self):
            self.seen = []

        def metadata(self):
            return metadata

        def __iter__(self):
            self.seen.append("weight")
            yield "weight", tensor

    engine = _make_trainer(is_sender=False)
    source = RecordingSource()
    engine.source = source
    engine.packed = packed
    fake_npu = SimpleNamespace(device=MagicMock(return_value=nullcontext()))

    with patch.object(torch, "npu", fake_npu, create=True):
        HCCLTrainerWeightTransferEngine._broadcast(engine, metadata)

    assert source.seen == ["weight"]
    engine.client.start_weight_update.assert_not_called()


def _make_trainer(*, is_sender: bool):
    engine = object.__new__(HCCLTrainerWeightTransferEngine)
    engine.client = MagicMock()
    engine.source = MagicMock()
    engine.source.metadata.return_value = []
    engine.is_sender = is_sender
    engine.packed = False
    engine.packed_buffer_size_bytes = 1024
    engine.packed_num_buffers = 2
    engine.device = torch.device("npu:0")
    engine.group = MagicMock() if is_sender else None
    engine._broadcast = MagicMock()
    engine._post_send_sync = MagicMock()
    return engine


def test_non_sender_synchronizes_source_before_returning():
    engine = _make_trainer(is_sender=False)

    engine.send_weights()

    engine._broadcast.assert_called_once_with([])
    engine._post_send_sync.assert_called_once_with()
    engine.client.start_weight_update.assert_not_called()


def test_post_send_sync_synchronizes_trainer_device_stream():
    engine = _make_trainer(is_sender=False)
    stream = MagicMock()
    fake_npu = SimpleNamespace(
        device=MagicMock(return_value=nullcontext()),
        current_stream=MagicMock(return_value=stream),
    )

    with patch.object(torch, "npu", fake_npu, create=True):
        HCCLTrainerWeightTransferEngine._post_send_sync(engine)

    fake_npu.device.assert_called_once_with(engine.device)
    stream.synchronize.assert_called_once_with()


def test_failed_trainer_rejects_retry_before_any_rpc():
    engine = _make_trainer(is_sender=True)
    group = engine.group
    engine._broadcast.side_effect = RuntimeError("broadcast failed")

    with pytest.raises(RuntimeError, match="broadcast failed"):
        engine.send_weights()

    group.close.assert_called_once_with()
    engine.client.start_weight_update.assert_called_once_with()
    with pytest.raises(RuntimeError, match="failed state"):
        engine.send_weights()
    engine.client.start_weight_update.assert_called_once_with()


def test_update_rpc_failure_marks_trainer_failed():
    engine = _make_trainer(is_sender=True)
    engine.client.update_weights.side_effect = RuntimeError("update rejected")

    with pytest.raises(RuntimeError, match="update rejected"):
        engine.send_weights()

    with pytest.raises(RuntimeError, match="failed state"):
        engine.send_weights()
    engine.client.start_weight_update.assert_called_once_with()


def test_shutdown_is_idempotent_and_rejects_future_send():
    engine = _make_trainer(is_sender=True)
    group = engine.group

    engine.shutdown()
    engine.shutdown()

    group.close.assert_called_once_with()
    with pytest.raises(RuntimeError, match="closed"):
        engine.send_weights()
    engine.client.start_weight_update.assert_not_called()


def test_trainer_does_not_expose_worker_update_method():
    assert "update_weights" not in HCCLTrainerWeightTransferEngine.__dict__
