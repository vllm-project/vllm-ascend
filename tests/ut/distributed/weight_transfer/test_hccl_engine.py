# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerWeightTransferEngine,
    NCCLWeightTransferEngine,
)

from vllm_ascend.distributed.weight_transfer.hccl_engine import (
    HCCLTrainerInitInfo,
    HCCLTrainerWeightTransferEngine,
    HCCLWeightTransferEngine,
    HCCLWeightTransferInitInfo,
    HCCLWeightTransferUpdateInfo,
)


@pytest.fixture(autouse=True)
def _npu_context():
    with (
        patch.object(
            torch, "npu", SimpleNamespace(device=lambda _: nullcontext(), current_stream=MagicMock()), create=True
        ),
        patch("torch.accelerator.current_device_index", return_value=0),
    ):
        yield


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

    init.assert_called_once_with("127.0.0.1", 12345, 1, 2, device=0)
    assert engine.model_update_group is not old_group


def test_inherits_upstream_orchestration():
    assert issubclass(HCCLWeightTransferEngine, NCCLWeightTransferEngine)
    assert issubclass(HCCLTrainerWeightTransferEngine, NCCLTrainerWeightTransferEngine)
    for name in ("start_weight_update", "finish_weight_update"):
        assert getattr(HCCLWeightTransferEngine, name) is getattr(NCCLWeightTransferEngine, name)
    assert HCCLTrainerWeightTransferEngine._checked_iter is NCCLTrainerWeightTransferEngine._checked_iter


@pytest.mark.parametrize("name,value", [("packed", True), ("packed_buffer_size_bytes", 64), ("packed_num_buffers", 3)])
def test_wire_params_preserve_legacy_updates_and_reject_init_conflicts(name, value):
    engine = _make_engine()
    engine.parallel_config = SimpleNamespace(data_parallel_index=0, world_size=1, rank=0)
    engine.model_update_group = None
    args: dict[str, Any] = dict(master_address="host", master_port=1234, rank_offset=1, world_size=2)
    update = HCCLWeightTransferUpdateInfo(names=[], dtype_names=[], shapes=[], **{name: value})
    with patch.object(engine, "_stateless_init_process_group", return_value=MagicMock()):
        engine.init_transfer_engine(HCCLWeightTransferInitInfo(**args))
        default = getattr(engine, name)
        assert engine._wire_params(update)[name] == value
        engine.init_transfer_engine(HCCLWeightTransferInitInfo(**args, **{name: value}))
        assert getattr(engine, name) == value
        assert engine._wire_params(update)[name] == value
        other = False if name == "packed" else value + 1
        with pytest.raises(ValueError, match=name):
            engine._wire_params(HCCLWeightTransferUpdateInfo(names=[], dtype_names=[], shapes=[], **{name: other}))
        engine.init_transfer_engine(HCCLWeightTransferInitInfo(**args))
        assert getattr(engine, name) == default
        assert engine._wire_params(HCCLWeightTransferUpdateInfo(names=[], dtype_names=[], shapes=[]))[name] != value


@pytest.mark.parametrize("name,value", [("packed", "yes"), ("packed_buffer_size_bytes", 0), ("packed_num_buffers", -1)])
def test_rejects_invalid_wire_params(name, value):
    with pytest.raises(ValueError, match=name):
        HCCLWeightTransferUpdateInfo(names=[], dtype_names=[], shapes=[], **{name: value})


def _trainer(sender=True):
    return HCCLTrainerWeightTransferEngine(
        client=MagicMock(), source=MagicMock(), is_sender=sender, device=torch.device("cpu")
    )


def test_trainer_defaults_to_packed_like_upstream():
    info = HCCLTrainerInitInfo(rank=0, master_address="host", master_port=1234, world_size=2)
    assert info.packed is True
    assert _trainer().packed is True


@pytest.mark.parametrize("sender", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_trainer_init_builds_rank_zero_handshake(sender, packed):
    info = HCCLTrainerInitInfo(
        rank=0 if sender else 1, master_address="host", master_port=1234, world_size=3, packed=packed
    )
    client = MagicMock()
    with (
        patch.dict(WeightTransferTrainerFactory._registry, {"hccl": lambda: HCCLTrainerWeightTransferEngine}),
        patch("torch.accelerator.current_device_index", return_value=0),
        patch("vllm_ascend.distributed.weight_transfer.hccl_common.trainer_init", return_value=MagicMock()) as init,
    ):
        engine = WeightTransferTrainerFactory.trainer_init(info, client=client, source=MagicMock())
    assert engine.is_sender is sender
    if sender:
        payload = client.init_weight_transfer_engine.call_args.args[0]
        assert payload["rank_offset"] == 1
        assert payload["packed"] is packed
        assert init.call_args.kwargs["rank"] == 0
    else:
        client.init_weight_transfer_engine.assert_not_called()
        init.assert_not_called()


@pytest.mark.parametrize("packed", [False, True])
def test_trainer_normalizes_source_for_both_modes(packed):
    engine = _trainer()
    engine.device = torch.device("cpu")  # CPU stand-in for the assigned NPU.
    engine.model_update_group = MagicMock()
    engine.packed = packed
    tensor = MagicMock()
    tensor.detach.return_value.to.return_value.contiguous.return_value = torch.ones(1)
    pairs = [("w", tensor)]
    namespace = "vllm_ascend.distributed.weight_transfer.hccl_engine"

    def pack(**kwargs):
        item = next(kwargs["iterator"])
        assert torch.equal(kwargs["post_iter_func"](item), torch.ones(1))

    with (
        patch.object(
            torch, "npu", SimpleNamespace(device=lambda _: nullcontext(), current_stream=MagicMock()), create=True
        ) as npu,
        patch.object(engine, "_checked_iter", return_value=iter(pairs)),
        patch(f"{namespace}.packed_broadcast_producer", side_effect=pack),
    ):
        engine._broadcast(pairs, [])
    tensor.detach.return_value.to.assert_called_once_with(engine.device)
    if not packed:
        assert torch.equal(engine.model_update_group.broadcast.call_args.args[0], torch.ones(1))
        npu.current_stream().synchronize.assert_called_once_with()


@pytest.mark.parametrize("available", [False, True])
def test_post_send_sync_only_when_npu_available(available):
    engine = _trainer()
    stream = MagicMock()
    device_context = MagicMock(return_value=nullcontext())
    with patch.object(
        torch,
        "npu",
        SimpleNamespace(
            is_available=lambda: available,
            device=device_context,
            current_stream=lambda: stream,
        ),
        create=True,
    ):
        engine._post_send_sync()
    if available:
        device_context.assert_called_once_with(engine.device)
        stream.synchronize.assert_called_once_with()
    else:
        device_context.assert_not_called()
        stream.synchronize.assert_not_called()


def test_non_sender_drains_source_without_validating_placeholder_tensors():
    engine = _trainer(False)
    seen = []

    def source():
        seen.append(True)
        yield "w", None

    with patch.object(torch, "npu", SimpleNamespace(device=lambda _: nullcontext()), create=True):
        engine._broadcast(source(), [])
    assert seen == [True]
    engine.client.start_weight_update.assert_not_called()


@pytest.mark.parametrize("failure", [False, True])
def test_trainer_wraps_upstream_send_and_guards_failure_or_shutdown(failure):
    engine = _trainer()
    group = engine.model_update_group = MagicMock()
    error = RuntimeError("send failed")
    with patch.object(NCCLTrainerWeightTransferEngine, "send_weights", side_effect=error if failure else None) as send:
        if failure:
            with pytest.raises(RuntimeError, match="send failed") as caught:
                engine.send_weights()
            assert caught.value is error
        else:
            engine.send_weights()
            engine.shutdown()
        with pytest.raises(RuntimeError):
            engine.send_weights()
        send.assert_called_once_with()
    engine.shutdown()
    group.close.assert_called_once_with()


def test_worker_update_binds_assigned_device_for_base_sync():
    engine = _make_engine()
    engine.device = torch.device("cpu")
    context = MagicMock(return_value=nullcontext())
    with (
        patch.object(torch, "npu", SimpleNamespace(device=context), create=True),
        patch.object(NCCLWeightTransferEngine, "update_weights") as update,
    ):
        engine.update_weights({})
    context.assert_called_once_with(engine.device)
    update.assert_called_once_with({})


@pytest.mark.parametrize("sender", [False, True])
def test_send_reuses_real_upstream_round_trip(sender):
    engine = _trainer(sender)
    engine.source.metadata.return_value = []
    with patch.object(engine, "_broadcast") as broadcast, patch.object(engine, "_post_send_sync") as sync:
        engine.send_weights()
    broadcast.assert_called_once_with(engine.source, [])
    sync.assert_called_once_with()
    if sender:
        assert [call[0] for call in engine.client.method_calls] == [
            "start_weight_update",
            "update_weights",
            "finish_weight_update",
        ]
    else:
        assert engine.client.method_calls == []


@pytest.mark.parametrize("packed", [False, True])
def test_worker_loads_on_assigned_device_in_both_modes(packed):
    engine = _make_engine()
    engine._init_info = HCCLWeightTransferInitInfo(master_address="host", master_port=1234, rank_offset=1, world_size=2)
    engine.model_update_group = MagicMock()
    info = HCCLWeightTransferUpdateInfo(names=["w"], dtype_names=["float32"], shapes=[[1]], packed=packed)

    def receive(**kwargs):
        assert list(kwargs["iterator"]) == [("w", ([1], torch.float32))]
        assert "device" not in kwargs  # Keep the existing packed helper interface.
        kwargs["post_unpack_func"]([("w", torch.ones(1))])

    with patch("vllm_ascend.distributed.weight_transfer.hccl_engine.packed_broadcast_consumer", side_effect=receive):
        engine.receive_weights(info)
    assert engine.model.load_weights.call_args.args[0][0][0] == "w"
