# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import cached_property
from threading import Barrier
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
import torch
from vllm.distributed.weight_transfer.base import ParamMeta, WeightSource
from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory
from vllm.distributed.weight_transfer.nccl_common import NCCLWeightTransferInitInfo, worker_init_payload
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerWeightTransferEngine,
    NCCLWeightTransferEngine,
)

from vllm_ascend.distributed.weight_transfer.hccl_common import HCCLWeightTransferInitInfo
from vllm_ascend.distributed.weight_transfer.hccl_engine import (
    HCCLTrainerInitInfo,
    HCCLTrainerWeightTransferEngine,
    HCCLWeightTransferEngine,
    HCCLWeightTransferUpdateInfo,
)

MULTI_SOURCE = hasattr(NCCLTrainerWeightTransferEngine, "_broadcast_plan")
requires_multi_source = pytest.mark.skipif(not MULTI_SOURCE, reason="requires vLLM multi-source broadcast support")


class _Source(WeightSource):
    """Exercise both supported upstream source contracts without device mocks."""

    def __init__(self, pairs, metadata=None):
        self.pairs = pairs
        self.metas = metadata if metadata is not None else [ParamMeta(n, t.dtype, tuple(t.shape)) for n, t in pairs]
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        yield from self.pairs

    def iter_full(self):
        yield from self

    if MULTI_SOURCE:

        @cached_property
        def metadata(self):
            return self.metas

    else:

        def metadata(self):
            return self.metas


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
    engine._last_hccl_update_info = {}
    engine.rank_offset = 1
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


def test_init_info_preserves_upstream_payload_contract():
    init_info = HCCLWeightTransferInitInfo(master_address="host", master_port=1234, rank_offset=1, world_size=2)

    assert isinstance(init_info, NCCLWeightTransferInitInfo)
    assert HCCLWeightTransferEngine.init_info_cls is HCCLWeightTransferInitInfo
    assert worker_init_payload(init_info) == dict(master_address="host", master_port=1234, rank_offset=1, world_size=2)
    restored = HCCLWeightTransferInitInfo(**worker_init_payload(init_info))
    assert restored == init_info
    assert restored.packed is None
    assert restored.packed_buffer_size_bytes is None
    assert restored.packed_num_buffers is None


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"master_address": None}, "master_address"),
        ({"master_port": None}, "master_port"),
        ({"nccl_unique_id_b64": "unsupported"}, "TCP rendezvous"),
    ],
)
def test_init_info_requires_tcp_rendezvous(overrides, match):
    args = dict(master_address="host", master_port=1234, rank_offset=1, world_size=2)
    with pytest.raises(ValueError, match=match):
        HCCLWeightTransferInitInfo(**(args | overrides))


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
        HCCLWeightTransferInitInfo(
            master_address="host", master_port=1234, rank_offset=1, world_size=2, **{name: value}
        )
    with pytest.raises(ValueError, match=name):
        HCCLWeightTransferUpdateInfo(names=[], dtype_names=[], shapes=[], **{name: value})


def _trainer(sender=True):
    return HCCLTrainerWeightTransferEngine(
        client=MagicMock(),
        source=_Source([("w", torch.ones(1))]),
        is_sender=sender,
        rank=0 if sender else 1,
        device=torch.device("cpu"),
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
        engine = WeightTransferTrainerFactory.trainer_init(info, client=client, source=_Source([("w", torch.ones(1))]))
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
    engine.model_update_group.rank = 0
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
        engine._send_tensors(engine.source, [], src=0)
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


@pytest.mark.skipif(MULTI_SOURCE, reason="legacy source contract")
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
    engine.model_update_group = SimpleNamespace(rank=0, broadcast=MagicMock())
    engine.packed = False
    with patch.object(engine, "_post_send_sync") as sync:
        engine.send_weights()
        engine.send_weights()
    assert sync.call_count == 2
    assert engine.source.iterations == 2
    if sender:
        assert [call[0] for call in engine.client.method_calls] == [
            "start_weight_update",
            "update_weights",
            "finish_weight_update",
        ] * 2
        payloads = [c.args[0] for c in engine.client.update_weights.call_args_list]
        assert payloads[0]["names"] == ["w"]
        assert "src" not in payloads[0]
        assert payloads[1]["names"] == ([] if hasattr(engine, "_sent_meta") else ["w"])
        assert engine.model_update_group.broadcast.call_count == 2
    else:
        assert engine.client.method_calls == []
        engine.model_update_group.broadcast.assert_not_called()


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


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("reuse_metadata", [False, True])
def test_worker_receives_two_updates(packed, reuse_metadata):
    engine = _make_engine()
    engine._init_info = HCCLWeightTransferInitInfo(
        master_address="host", master_port=1234, rank_offset=1, world_size=2, packed=packed
    )
    engine.model_update_group = MagicMock()
    full = HCCLWeightTransferUpdateInfo(names=["w"], dtype_names=["float32"], shapes=[[1]])
    empty = HCCLWeightTransferUpdateInfo()
    received = []

    def receive(**kwargs):
        received.extend(list(kwargs["iterator"]))

    with patch("vllm_ascend.distributed.weight_transfer.hccl_engine.packed_broadcast_consumer", side_effect=receive):
        engine.receive_weights(full)
        engine.receive_weights(empty if reuse_metadata else full)

    if packed:
        assert received == [("w", ([1], torch.float32))] * 2
    else:
        assert engine.model_update_group.broadcast.call_count == 2
        assert engine.model.load_weights.call_count == 2


def test_worker_metadata_is_per_model_and_cleared_on_reinit():
    engine = _make_engine()
    engine.parallel_config = SimpleNamespace(data_parallel_index=0, world_size=1, rank=0)
    init = HCCLWeightTransferInitInfo(master_address="host", master_port=1234, rank_offset=1, world_size=2)
    full = HCCLWeightTransferUpdateInfo(names=["w"], dtype_names=["float32"], shapes=[[1]])
    empty = HCCLWeightTransferUpdateInfo(names=[], dtype_names=[], shapes=[])
    with patch.object(engine, "_stateless_init_process_group", return_value=MagicMock()):
        engine.init_transfer_engine(init)
        original_model = engine.model
        engine.receive_weights(full)
        engine.model = MagicMock()
        engine.receive_weights(empty)
        engine.model.load_weights.assert_not_called()
        engine.model = original_model
        engine.receive_weights(empty)
        assert original_model.load_weights.call_count == 2
        engine.init_transfer_engine(init)
        engine.receive_weights(empty)
        assert original_model.load_weights.call_count == 2
        engine.model_update_group.broadcast.reset_mock()
        engine.receive_weights(full)
        engine.model_update_group.broadcast.assert_called_once()


@requires_multi_source
@pytest.mark.parametrize("rank,group_rank", [(0, 0), (2, None), (4, 1), (6, None)])
@pytest.mark.parametrize("rank_zero_owns", [False, True])
def test_trainer_initializes_compressed_group_ranks(rank, group_rank, rank_zero_owns):
    metas = [ParamMeta("b", torch.float32, (2,), owners=(4, 6))]
    if rank_zero_owns:
        metas.insert(0, ParamMeta("a", torch.float32, (1,), owners=(0, 2)))
    source = _Source([], metas)
    client = MagicMock()
    info = HCCLTrainerInitInfo(rank=rank, master_address="host", master_port=1234, world_size=4)
    with patch("vllm_ascend.distributed.weight_transfer.hccl_common.trainer_init") as init:
        engine = HCCLTrainerWeightTransferEngine.trainer_init(info, client=client, source=source)
    assert engine.rank == rank
    if group_rank is None:
        init.assert_not_called()
        assert engine.model_update_group is None
    else:
        init.assert_called_once_with(info, rank=group_rank)
    if rank == 0:
        payload = client.init_weight_transfer_engine.call_args.args[0]
        assert payload["rank_offset"] == 2
        assert payload["world_size"] == 4
        assert "nccl_unique_id_b64" not in payload
    else:
        client.init_weight_transfer_engine.assert_not_called()


@requires_multi_source
@pytest.mark.parametrize(
    "owners,world_size,match",
    [
        ([(0, 2), (4, 2)], 4, "owns parameters broadcast"),
        ([(0,), None, (4,)], 4, "has no owners"),
        ([(4, 0)], 4, "rank 0"),
        ([(0,), (4,)], 2, "no inference workers"),
    ],
)
def test_invalid_ownership_fails_before_hccl_rendezvous(owners, world_size, match):
    metas = [ParamMeta(str(i), torch.float32, (1,), owners=value) for i, value in enumerate(owners)]
    source = _Source([], metas)
    with patch("vllm_ascend.distributed.weight_transfer.hccl_common.trainer_init") as init:
        for rank in (0, 2, 4):
            info = HCCLTrainerInitInfo(rank=rank, master_address="host", master_port=1234, world_size=world_size)
            with pytest.raises(ValueError, match=match):
                HCCLTrainerWeightTransferEngine.trainer_init(info, client=MagicMock(), source=source)
    init.assert_not_called()


@requires_multi_source
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("group_rank,src", [(0, 1), (1, 0), (1, 1)])
def test_broadcast_nonzero_source_and_discard_other_sources(packed, group_rank, src):
    engine = _trainer(group_rank == 0)
    engine.packed = packed
    engine.model_update_group = MagicMock(rank=group_rank)
    meta = [ParamMeta("w", torch.float32, (1,))]
    namespace = "vllm_ascend.distributed.weight_transfer.hccl_engine"
    consumed = []

    def receive(**kwargs):
        assert kwargs["src"] == src
        consumed.extend(kwargs["iterator"])
        kwargs["post_unpack_func"]([("w", torch.ones(1))])

    with (
        patch.object(engine, "_send_tensors") as send,
        patch(f"{namespace}.packed_broadcast_consumer", side_effect=receive),
    ):
        engine._broadcast(meta, src)
    if group_rank == src:
        send.assert_called_once_with(engine.source, meta, src)
    else:
        send.assert_not_called()
        if packed:
            assert consumed == [("w", ([1], torch.float32))]
        else:
            call_args = engine.model_update_group.broadcast.call_args
            assert call_args.kwargs["src"] == src
            assert tuple(call_args.args[0].shape) == (1,)
            torch.npu.current_stream().synchronize.assert_called_once_with()


@requires_multi_source
def test_non_broadcaster_still_drives_owned_parameter_gathers():
    meta = [ParamMeta("a", torch.float32, (1,), owners=(0, 2)), ParamMeta("b", torch.float32, (2,), owners=(4, 6))]
    source = _Source([("a", torch.ones(1))], meta)
    info = HCCLTrainerInitInfo(rank=2, master_address="host", master_port=1234, world_size=4)
    engine = HCCLTrainerWeightTransferEngine.trainer_init(info, client=MagicMock(), source=source)
    with patch.object(engine, "_post_send_sync"):
        engine.send_weights()
    assert source.iterations == 1
    assert engine.model_update_group is None
    assert engine.client.method_calls == []


@requires_multi_source
def test_rank_zero_sends_each_source_then_reuses_metadata():
    engine = _trainer()
    metas = [ParamMeta("a", torch.float32, (1,), owners=(0,)), ParamMeta("b", torch.float32, (2,), owners=(4,))]
    engine.source = _Source([("a", torch.ones(1))], metas)
    with patch.object(engine, "_broadcast") as broadcast, patch.object(engine, "_post_send_sync"):
        engine.send_weights()
        engine.send_weights()
    assert broadcast.call_args_list == [call([metas[0]], 0), call([metas[1]], 1)] * 2
    payloads = [c.args[0] for c in engine.client.update_weights.call_args_list]
    assert payloads == [
        {"names": ["a"], "dtype_names": ["float32"], "shapes": [[1]]},
        {"names": ["b"], "dtype_names": ["float32"], "shapes": [[2]], "src": 1},
        {"names": [], "dtype_names": [], "shapes": []},
        {"names": [], "dtype_names": [], "shapes": [], "src": 1},
    ]
    assert engine.client.start_weight_update.call_count == 2
    assert engine.client.finish_weight_update.call_count == 2


@pytest.mark.parametrize("packed", [False, True])
def test_worker_metadata_is_per_source_and_current_wire_params_win(packed):
    engine = _make_engine()
    engine.rank_offset = 2
    engine._init_info = HCCLWeightTransferInitInfo(master_address="host", master_port=1234, rank_offset=2, world_size=4)
    engine.model_update_group = MagicMock()
    updates = [
        HCCLWeightTransferUpdateInfo(names=["a"], dtype_names=["float32"], shapes=[[1]], src=0),
        HCCLWeightTransferUpdateInfo(names=["b"], dtype_names=["float16"], shapes=[[2]], src=1),
    ]
    # Prime the two caches using unpacked, then independently switch this call's
    # legacy wire mode while reusing only the source's parameter metadata.
    for update in updates:
        engine.receive_weights(update)
    engine.model.load_weights.reset_mock()
    engine.model_update_group.broadcast.reset_mock()
    received = []

    def receive(**kwargs):
        received.append((kwargs["src"], list(kwargs["iterator"])))

    with patch("vllm_ascend.distributed.weight_transfer.hccl_engine.packed_broadcast_consumer", side_effect=receive):
        for src in (1, 0, 1):
            engine.receive_weights(HCCLWeightTransferUpdateInfo(src=src, packed=packed))
    if packed:
        assert received == [
            (1, [("b", ([2], torch.float16))]),
            (0, [("a", ([1], torch.float32))]),
            (1, [("b", ([2], torch.float16))]),
        ]
    else:
        assert [c.kwargs["src"] for c in engine.model_update_group.broadcast.call_args_list] == [1, 0, 1]
        assert [c.args[0][0][0] for c in engine.model.load_weights.call_args_list] == ["b", "a", "b"]
        assert [tuple(c.args[0][0][1].shape) for c in engine.model.load_weights.call_args_list] == [(2,), (1,), (2,)]


@pytest.mark.parametrize("src", [-1, True, False, 1.0, "1", None])
def test_rejects_non_integer_or_negative_source(src):
    with pytest.raises(ValueError, match="src"):
        HCCLWeightTransferUpdateInfo(names=[], dtype_names=[], shapes=[], src=src)


def test_worker_rejects_source_outside_trainer_rank_interval():
    engine = _make_engine()
    engine.model_update_group = MagicMock()
    with pytest.raises(ValueError, match="sender rank"):
        engine.receive_weights(HCCLWeightTransferUpdateInfo(names=["w"], dtype_names=["float32"], shapes=[[1]], src=1))
    engine.model_update_group.broadcast.assert_not_called()


@pytest.mark.parametrize("rank_offset,world_size", [(0, 2), (2, 2), (3, 2), (True, 2), (1, 0), (1, 2.0)])
def test_rejects_invalid_transfer_rank_layout(rank_offset, world_size):
    with pytest.raises(ValueError):
        HCCLWeightTransferInitInfo(
            master_address="host", master_port=1234, rank_offset=rank_offset, world_size=world_size
        )


def test_worker_group_rank_includes_data_parallel_offset():
    engine = _make_engine()
    engine.parallel_config = SimpleNamespace(data_parallel_index=1, world_size=2, rank=1)
    info = HCCLWeightTransferInitInfo(master_address="host", master_port=1234, rank_offset=2, world_size=6)
    with patch.object(engine, "_stateless_init_process_group") as init:
        engine.init_transfer_engine(info)
    init.assert_called_once_with("host", 1234, 5, 6, device=0)


def test_worker_rejects_group_rank_outside_world_before_rendezvous():
    engine = _make_engine()
    engine.parallel_config = SimpleNamespace(data_parallel_index=1, world_size=2, rank=1)
    info = HCCLWeightTransferInitInfo(master_address="host", master_port=1234, rank_offset=2, world_size=5)
    with (
        patch.object(engine, "_stateless_init_process_group") as init,
        pytest.raises(ValueError, match="worker group rank"),
    ):
        engine.init_transfer_engine(info)
    init.assert_not_called()


@requires_multi_source
@pytest.mark.parametrize("packed", [False, True])
def test_multi_source_round_trip_moves_changed_values_with_real_packers(packed, monkeypatch):
    """Two trainers and two workers join every CPU-simulated collective.

    The real upstream send loop, HCCL receive/discard adapters and NPU packing
    code run together. Only device/stream execution and native HCCL are faked.
    Round two changes values while reusing metadata, so a no-op cannot pass.
    """
    barrier = Barrier(4, timeout=10)
    pending = {}
    roots = [[] for _ in range(4)]

    def communicator(rank):
        def broadcast(tensor, src, stream=None):
            pending[rank] = (tensor, src)
            barrier.wait()
            if rank == 0:
                assert {root for _, root in pending.values()} == {src}
                payload = pending[src][0].clone()
                for buffer, _ in pending.values():
                    assert buffer.shape == payload.shape and buffer.dtype == payload.dtype
                    buffer.copy_(payload)
            barrier.wait()
            roots[rank].append(src)

        return SimpleNamespace(rank=rank, broadcast=broadcast, close=MagicMock())

    stream = SimpleNamespace(synchronize=lambda: None)
    monkeypatch.setattr(torch.npu, "Stream", lambda: stream, raising=False)
    monkeypatch.setattr(torch.npu, "stream", lambda _: nullcontext(), raising=False)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: stream)
    original_empty = torch.empty

    def cpu_empty(*args, **kwargs):
        if kwargs.get("device") == "npu":
            kwargs["device"] = "cpu"
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)
    pairs = [
        ("a", torch.ones(1)),
        ("b", torch.full((4,), 2.0)),
        ("c", torch.full((2,), 3.0)),
        ("d", torch.full((2,), 4.0, dtype=torch.bfloat16)),
        ("e", torch.full((6,), 5.0)),
        ("f", torch.tensor(6.0)),
    ]
    meta = [ParamMeta(n, t.dtype, tuple(t.shape), owners=(0 if i < 3 else 4,)) for i, (n, t) in enumerate(pairs)]
    workers = []
    received = [{}, {}]
    for index in range(2):
        worker = _make_engine()
        worker.rank_offset = 2
        worker._init_info = HCCLWeightTransferInitInfo(
            master_address="host",
            master_port=1234,
            rank_offset=2,
            world_size=4,
            packed=packed,
            packed_buffer_size_bytes=8,
            packed_num_buffers=2,
        )
        worker.model_update_group = communicator(index + 2)

        def load(weights, target=received[index]):
            target.update((name, value.clone()) for name, value in weights)

        worker.model.load_weights.side_effect = load
        workers.append(worker)
    client = MagicMock()

    def update(payload):
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(worker.update_weights, payload) for worker in workers]
            for future in futures:
                future.result(timeout=15)

    client.update_weights.side_effect = update
    trainers = [
        HCCLTrainerWeightTransferEngine(
            client=client,
            source=_Source(pairs[i * 3 : (i + 1) * 3], meta),
            is_sender=i == 0,
            rank=0 if i == 0 else 4,
            device=torch.device("cpu"),
            packed=packed,
            packed_buffer_size_bytes=8,
            packed_num_buffers=2,
        )
        for i in range(2)
    ]
    for index, trainer in enumerate(trainers):
        trainer.model_update_group = communicator(index)
        monkeypatch.setattr(trainer, "_post_send_sync", lambda: None)
    for update_round in range(2):
        if update_round:
            for _, tensor in pairs:
                tensor.add_(10)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(trainer.send_weights) for trainer in trainers]
            for future in futures:
                future.result(timeout=20)
        for values in received:
            assert set(values) == {name for name, _ in pairs}
            for name, tensor in pairs:
                torch.testing.assert_close(values[name], tensor)
    per_source = 2 if packed else 3
    assert roots == [([0] * per_source + [1] * per_source) * 2] * 4
    assert client.start_weight_update.call_count == client.finish_weight_update.call_count == 2
