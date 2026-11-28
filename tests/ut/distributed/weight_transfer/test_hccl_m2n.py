# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import cached_property
from queue import Queue
from threading import Condition
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, call

import pytest
import torch
from vllm.distributed.utils import StatelessProcessGroup
from vllm.distributed.weight_transfer.base import ParamMeta, Shard, WeightSource
from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory, WeightTransferTrainerFactory
from vllm.distributed.weight_transfer.m2n_common import M2NLayout, M2NMesh, M2NParamMeta
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_ascend.distributed.device_communicators.pyhccl import HcclSendRecvType, PyHcclCommunicator
from vllm_ascend.distributed.device_communicators.pyhccl_wrapper import HCCLLibrary, hcclDataTypeEnum
from vllm_ascend.distributed.weight_transfer import hccl_m2n_engine, register_engine
from vllm_ascend.distributed.weight_transfer.hccl_m2n_engine import (
    HCCLM2NTrainerInitInfo,
    HCCLM2NTrainerWeightTransferEngine,
    HCCLM2NWeightTransferEngine,
    HCCLM2NWeightTransferInitInfo,
    M2NWeightTransferUpdateInfo,
    _TransferSession,
    _validate_hccl_transfer_plan,
)
from vllm_ascend.distributed.weight_transfer.hccl_m2n_transport import plan_rank_transfers, transfer_parameter


class _MemoryP2P:
    available = True
    disabled = False

    def __init__(self, rank, queues):
        self.rank = rank
        self.queues = queues
        self.device = torch.device("cpu")

    def batch_send_recv(self, buffers, *, send, stream):
        assert len({peer for peer, _ in buffers}) == len(buffers)
        for peer, buffer in buffers:
            assert buffer.is_contiguous()
            if send:
                self.queues[self.rank, peer].put(buffer.clone())
            else:
                data = self.queues[peer, self.rank].get(timeout=5)
                assert data.shape == buffer.shape
                buffer.copy_(data)


class _MemoryStore:
    """Small thread-safe store for the upstream metadata protocol."""

    def __init__(self):
        self.condition = Condition()
        self.data = {}

    def set(self, key, value):
        with self.condition:
            self.data[key] = value
            self.condition.notify_all()

    def get(self, key):
        with self.condition:
            if not self.condition.wait_for(lambda: key in self.data, timeout=5):
                raise TimeoutError(key)
            return self.data[key]


@pytest.mark.parametrize(
    "src_dim,dst_layout,dst_dim",
    [
        (0, M2NLayout(M2NMesh((1, 4), 2), (-1, 0)), 0),
        (0, M2NLayout(M2NMesh((1, 4), 2), (-1, 1)), 1),
        (1, M2NLayout(M2NMesh((1, 4), 2), (-1, 0)), 0),
        (0, M2NLayout(M2NMesh((4, 1), 2), None), None),
    ],
)
def test_reshard_moves_each_piece_to_its_destination(src_dim, dst_layout, dst_dim):
    tensor = torch.arange(64).reshape(8, 8)
    source = M2NLayout(M2NMesh((1, 2), 0), (-1, src_dim))
    queues: dict[tuple[int, int], Queue[torch.Tensor]] = {
        (src, dst): Queue() for src in range(2) for dst in range(2, 6)
    }
    stream = Mock()
    for rank, shard in enumerate(tensor.chunk(2, dim=src_dim)):
        pieces = plan_rank_transfers(tuple(tensor.shape), source, dst_layout, rank)
        transfer_parameter(cast(Any, _MemoryP2P(rank, queues)), stream, shard, pieces, send=True)

    expected_shards = [tensor] * 4 if dst_dim is None else tensor.chunk(4, dim=dst_dim)
    for rank, expected in enumerate(expected_shards, start=2):
        pieces = plan_rank_transfers(tuple(tensor.shape), source, dst_layout, rank)
        received = torch.full(expected.shape, -1, dtype=expected.dtype)
        transfer_parameter(cast(Any, _MemoryP2P(rank, queues)), stream, received, pieces, send=False)
        torch.testing.assert_close(received, expected)
    assert all(queue.empty() for queue in queues.values())


def test_replicated_source_selects_one_sender_per_target():
    src = M2NLayout(M2NMesh((2, 1), 0), None)
    dst = M2NLayout(M2NMesh((2, 2), 2), (-1, 1))
    sending = [plan_rank_transfers((4, 8), src, dst, rank) for rank in range(2)]
    assert [len(plan) for plan in sending] == [2, 2]
    for rank in range(2, 6):
        (receive,) = plan_rank_transfers((4, 8), src, dst, rank)
        (send,) = [piece for piece in sending[receive.peer] if piece.peer == rank]
        assert send.local_slice[1] == slice((rank % 2) * 4, (rank % 2 + 1) * 4)
        assert receive.local_slice == (slice(0, 4), slice(0, 4))


def test_hccl_batch_uses_element_counts_and_peer_ranks():
    native_batch = Mock(return_value=0)
    library = HCCLLibrary.__new__(HCCLLibrary)
    library.lib = SimpleNamespace(HcclBatchSendRecv=native_batch)
    library._funcs = {}
    group = PyHcclCommunicator.__new__(PyHcclCommunicator)
    group.available = True
    group.disabled = False
    group.rank = 0
    group.world_size = 3
    group.device = torch.device("cpu")
    group.comm = ctypes.c_void_p()
    group.hccl = library

    data = torch.zeros((3, 2), dtype=torch.bfloat16)
    group.batch_send_recv([(2, data)], send=True, stream=SimpleNamespace(npu_stream=42))
    items, count, _, stream = native_batch.call_args.args
    assert count == 1
    assert stream.value == 42
    assert items[0].buffer == data.data_ptr()
    assert items[0].count == 6
    assert items[0].remote_rank == 2
    assert items[0].datatype == hcclDataTypeEnum.hcclBfloat16
    assert items[0].send_recv_type == HcclSendRecvType.SEND


def test_session_rejects_inconsistent_destination_before_hccl(monkeypatch):
    group = StatelessProcessGroup(rank=0, world_size=1, store=_MemoryStore())
    group.broadcast_obj = lambda obj, src: [(-1, 0)]
    native_group = Mock()
    monkeypatch.setattr(hccl_m2n_engine, "PyHcclCommunicator", native_group)
    with pytest.raises(RuntimeError, match="different hccl_m2n destination layouts"):
        hccl_m2n_engine._open_session(
            group, SimpleNamespace(type="npu", index=0), [], "test-digest", M2NMesh((1, 1), 0), [None]
        )
    native_group.assert_not_called()


class _LocalSource(WeightSource):
    def __init__(self, shard):
        self.shard = shard

    @cached_property
    def metadata(self):
        return [ParamMeta("weight", torch.float32, (4, 4), owners=(0, 1), mesh_shape=(2,), placements=(Shard(0),))]

    def iter_local(self):
        yield "weight", self.shard

    def iter_full(self):
        raise AssertionError("HCCL M2N must send local shards")


@pytest.mark.parametrize(
    "options,precast_fp32,nd_format,uses_direct",
    [
        ({}, False, True, True),
        ({"direct_write": False}, False, True, False),
        ({}, True, True, False),
        ({}, False, False, False),
    ],
)
def test_automatic_direct_destinations(monkeypatch, options, precast_fp32, nd_format, uses_direct):
    trainer_info = HCCLM2NTrainerInitInfo(master_address="127.0.0.1", master_port=5000, rank=0, world_size=2, **options)
    trainer = HCCLM2NTrainerWeightTransferEngine(client=Mock(), source=Mock(), is_sender=True)
    trainer._wire_params = []
    trainer._source_digest = hccl_m2n_engine.source_plan_digest([])
    worker_info = HCCLM2NWeightTransferInitInfo(**trainer._worker_init_info(trainer_info))

    model = torch.nn.Linear(2, 2, bias=False)
    model.precast_fp32_weight = precast_fp32
    config = SimpleNamespace(
        pipeline_parallel_size=1,
        data_parallel_size=1,
        world_size=1,
        tensor_parallel_size=1,
        enable_expert_parallel=False,
        enable_eplb=False,
    )
    model_config = SimpleNamespace(is_moe=False, quantization=None, enable_sleep_mode=False)
    worker = HCCLM2NWeightTransferEngine(
        Mock(), SimpleNamespace(parallel_config=config, model_config=model_config), torch.device("cpu"), model
    )
    worker._metas = [SimpleNamespace(name="weight", dtype=torch.float32, shape=(2, 2))]
    model.weight.weight_loader = default_weight_loader
    monkeypatch.setattr(hccl_m2n_engine, "_is_nd", lambda _: nd_format)

    worker._prepare_hccl_destinations(worker_info)

    assert worker._parameter_destinations[0].direct is uses_direct
    assert worker._uses_load_weights is not uses_direct
    if uses_direct:
        assert worker._parameter_destinations[0].tensor.data_ptr() == model.weight.data_ptr()


def test_two_update_rounds_reconstruct_latest_weight(monkeypatch):
    from vllm.model_executor.model_loader import reload

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(device=lambda _: nullcontext(), synchronize=Mock(), current_stream=Mock()),
        raising=False,
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
    monkeypatch.setattr(hccl_m2n_engine, "_require_nd", Mock())
    start = Mock()
    finish = Mock()
    monkeypatch.setattr(reload, "start_reload", start)
    monkeypatch.setattr(reload, "finish_reload", finish)
    monkeypatch.setattr(hccl_m2n_engine, "HCCLLibrary", Mock())

    source_layout = M2NLayout(M2NMesh((2, 1), 0), (0, -1))
    dst_layout = M2NLayout(M2NMesh((1, 1), 2), None)
    meta = M2NParamMeta("weight", torch.float32, (4, 4), source_layout)
    queues: dict[tuple[int, int], Queue[torch.Tensor]] = {(src, 2): Queue() for src in range(2)}
    store = _MemoryStore()
    metadata = [StatelessProcessGroup(rank=rank, world_size=3, store=store) for rank in range(3)]
    monkeypatch.setattr(hccl_m2n_engine, "PyHcclCommunicator", lambda group, device: _MemoryP2P(group.rank, queues))
    device = SimpleNamespace(type="npu", index=0)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(
                hccl_m2n_engine._open_session,
                group,
                device,
                [meta],
                "source-digest",
                dst_layout.mesh,
                [None] if rank == 2 else None,
            )
            for rank, group in enumerate(metadata)
        ]
        sessions = [future.result(timeout=10) for future in futures]

    worker = HCCLM2NWeightTransferEngine(
        Mock(),
        SimpleNamespace(parallel_config=Mock(), model_config=Mock()),
        torch.device("cpu"),
        Mock(),
    )
    worker._metas = [meta]
    worker._parameter_destinations = [hccl_m2n_engine.M2NDestination(meta.name, None, None)]
    worker._uses_load_weights = True
    worker._handle = sessions[2]

    client = SimpleNamespace(
        start_weight_update=worker.start_weight_update,
        update_weights=worker.update_weights,
        finish_weight_update=worker.finish_weight_update,
    )
    tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    trainers = []
    for rank, shard in enumerate(tensor.chunk(2)):
        trainer = HCCLM2NTrainerWeightTransferEngine(
            client=client, source=_LocalSource(shard), is_sender=rank == 0, rank=rank
        )
        trainer._prepare_source_plan(trainer.source, 2)
        trainer._num_trainer_ranks = 2
        trainer._metadata_group = metadata[rank]
        trainer.group = sessions[rank].group
        trainer._handle = sessions[rank]
        trainers.append(trainer)

    loaded = {}
    worker.model.load_weights.side_effect = lambda weights: loaded.update(weights)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for _ in range(2):
            send_futures = [pool.submit(trainer.send_weights) for trainer in trainers]
            for send_future in send_futures:
                send_future.result(timeout=10)
            torch.testing.assert_close(loaded["weight"], tensor)
            tensor.add_(100)
    assert start.call_args_list == [call(worker.model)] * 2
    assert finish.call_args_list == [call(worker.model, worker.model_config, worker.device)] * 2


def test_failed_trainer_session_cannot_send_again():
    client = Mock()
    client.start_weight_update.side_effect = RuntimeError("update failed")
    trainer = HCCLM2NTrainerWeightTransferEngine(client=client, source=Mock(), is_sender=True)
    trainer._handle = _TransferSession(Mock(), [])
    trainer.group = trainer._handle.group
    with pytest.raises(RuntimeError, match="update failed"):
        trainer.send_weights()
    with pytest.raises(RuntimeError, match="session is not ready"):
        trainer.send_weights()
    client.start_weight_update.assert_called_once()


def test_registered_worker_and_trainer_dispatch(monkeypatch):
    monkeypatch.setattr(WeightTransferEngineFactory, "_registry", {})
    monkeypatch.setattr(WeightTransferTrainerFactory, "_registry", {})
    register_engine()
    assert WeightTransferEngineFactory._registry["hccl_m2n"]() is HCCLM2NWeightTransferEngine
    assert (
        WeightTransferTrainerFactory._registry[HCCLM2NTrainerInitInfo.backend]() is HCCLM2NTrainerWeightTransferEngine
    )


def test_replicated_source_distributes_load_across_params():
    src = M2NLayout(M2NMesh((4, 1), 0), None)
    dst = M2NLayout(M2NMesh((1, 1), 4), None)
    selected_senders = [plan_rank_transfers((4, 4), src, dst, rank=4, param_index=index)[0].peer for index in range(8)]
    assert selected_senders == [0, 1, 2, 3, 0, 1, 2, 3]
    for index, sender in enumerate(selected_senders):
        for rank in range(4):
            pieces = plan_rank_transfers((4, 4), src, dst, rank=rank, param_index=index)
            assert len(pieces) == (1 if rank == sender else 0)


class _FakeStream:
    def __init__(self, device=None):
        self.device = device
        self.synced = 0

    def synchronize(self):
        self.synced += 1


def test_per_module_nd_and_precast_fallback_preserves_other_direct_modules(monkeypatch):
    class _TwoModuleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.good = torch.nn.Linear(2, 2, bias=False)
            self.bad_nd = torch.nn.Linear(2, 2, bias=False)
            self.bad_precast = torch.nn.Linear(2, 2, bias=False)
            self.bad_precast.precast_fp32_weight = True
            for mod in (self.good, self.bad_nd, self.bad_precast):
                mod.weight.weight_loader = default_weight_loader

    model = _TwoModuleModel()
    trainer_info = HCCLM2NTrainerInitInfo(master_address="127.0.0.1", master_port=5000, rank=0, world_size=2)
    trainer = HCCLM2NTrainerWeightTransferEngine(client=Mock(), source=Mock(), is_sender=True)
    trainer._wire_params = []
    trainer._source_digest = hccl_m2n_engine.source_plan_digest([])
    worker_info = HCCLM2NWeightTransferInitInfo(**trainer._worker_init_info(trainer_info))

    config = SimpleNamespace(
        pipeline_parallel_size=1,
        data_parallel_size=1,
        world_size=1,
        tensor_parallel_size=1,
        enable_expert_parallel=False,
        enable_eplb=False,
    )
    model_config = SimpleNamespace(is_moe=False, quantization=None, enable_sleep_mode=False)
    worker = HCCLM2NWeightTransferEngine(
        Mock(), SimpleNamespace(parallel_config=config, model_config=model_config), torch.device("cpu"), model
    )
    worker._metas = [
        SimpleNamespace(name="good.weight", dtype=torch.float32, shape=(2, 2)),
        SimpleNamespace(name="bad_nd.weight", dtype=torch.float32, shape=(2, 2)),
        SimpleNamespace(name="bad_precast.weight", dtype=torch.float32, shape=(2, 2)),
    ]
    monkeypatch.setattr(hccl_m2n_engine, "_is_nd", lambda tensor: tensor.data_ptr() != model.bad_nd.weight.data_ptr())

    worker._prepare_hccl_destinations(worker_info)

    assert [d.direct for d in worker._parameter_destinations] == [True, False, False]
    assert worker._uses_load_weights is True


def test_non_dense_or_sleep_mode_falls_back_to_loader():
    model = torch.nn.Linear(2, 2, bias=False)
    model.weight.weight_loader = default_weight_loader
    trainer_info = HCCLM2NTrainerInitInfo(master_address="127.0.0.1", master_port=5000, rank=0, world_size=2)
    trainer = HCCLM2NTrainerWeightTransferEngine(client=Mock(), source=Mock(), is_sender=True)
    trainer._wire_params = []
    trainer._source_digest = hccl_m2n_engine.source_plan_digest([])
    worker_info = HCCLM2NWeightTransferInitInfo(**trainer._worker_init_info(trainer_info))

    config = SimpleNamespace(
        pipeline_parallel_size=1,
        data_parallel_size=1,
        world_size=1,
        tensor_parallel_size=1,
        enable_expert_parallel=False,
        enable_eplb=False,
    )
    model_config = SimpleNamespace(is_moe=True, quantization=None, enable_sleep_mode=True)
    worker = HCCLM2NWeightTransferEngine(
        Mock(), SimpleNamespace(parallel_config=config, model_config=model_config), torch.device("cpu"), model
    )
    worker._metas = [SimpleNamespace(name="weight", dtype=torch.float32, shape=(2, 2))]

    worker._prepare_hccl_destinations(worker_info)

    assert worker._parameter_destinations[0].direct is False
    assert worker._uses_load_weights is True


def test_validate_hccl_transfer_plan_allows_more_than_16_source_shards():
    src_layout = M2NLayout(M2NMesh((1, 32), 0), (-1, 0))
    dst_mesh = M2NMesh((1, 1), 32)
    meta = M2NParamMeta("weight", torch.bfloat16, (64, 4), src_layout)
    _validate_hccl_transfer_plan([meta], dst_mesh, [None])


def test_shutdown_resets_failed_state_for_reinit():
    group = Mock()
    trainer = HCCLM2NTrainerWeightTransferEngine(client=Mock(), source=Mock(), is_sender=True)
    trainer._handle = _TransferSession(group, [])
    trainer.group = group
    trainer._failed = True
    trainer.shutdown()
    assert trainer._failed is False
    assert trainer._handle is None
    group.close.assert_called_once()

    worker_group = Mock()
    worker = HCCLM2NWeightTransferEngine(
        Mock(), SimpleNamespace(parallel_config=Mock(), model_config=Mock()), torch.device("cpu"), Mock()
    )
    worker._handle = _TransferSession(worker_group, [])
    worker._failed = True
    worker._received = True
    worker.shutdown()
    assert worker._failed is False
    assert worker._received is False
    assert worker._handle is None
    assert worker._index == {}
    worker_group.close.assert_called_once()


def test_receive_weights_supports_chunked_subset_updates(monkeypatch):
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(device=lambda _: nullcontext(), synchronize=Mock(), current_stream=lambda _: _FakeStream()),
        raising=False,
    )
    src = M2NLayout(M2NMesh((1, 1), 0), None)
    dst = M2NLayout(M2NMesh((1, 1), 1), None)
    w1 = torch.arange(4, 8, dtype=torch.float32).reshape(2, 2)
    metas = [
        M2NParamMeta("w0", torch.float32, (2, 2), src),
        M2NParamMeta("w1", torch.float32, (2, 2), src),
    ]
    queues: dict[tuple[int, int], Queue[torch.Tensor]] = {(0, 1): Queue()}
    sender = _MemoryP2P(0, queues)
    receiver = _MemoryP2P(1, queues)
    send_pieces = [plan_rank_transfers((2, 2), src, dst, 0, param_index=i) for i in range(2)]
    recv_pieces = [plan_rank_transfers((2, 2), src, dst, 1, param_index=i) for i in range(2)]

    # Send only the second parameter first.
    transfer_parameter(cast(Any, sender), cast(Any, _FakeStream()), w1, send_pieces[1], send=True)

    worker = HCCLM2NWeightTransferEngine(
        Mock(), SimpleNamespace(parallel_config=Mock(), model_config=Mock()), torch.device("cpu"), Mock()
    )
    worker._metas = metas
    worker._parameter_destinations = [hccl_m2n_engine.M2NDestination(m.name, None, None) for m in metas]
    worker._index = {"w0": 0, "w1": 1}
    worker._uses_load_weights = True
    worker._handle = _TransferSession(cast(Any, receiver), recv_pieces)
    loaded = {}
    worker.model.load_weights.side_effect = lambda weights: loaded.update(weights)

    worker.receive_weights(M2NWeightTransferUpdateInfo(names=["w1"]))
    torch.testing.assert_close(loaded["w1"], w1)
    assert "w0" not in loaded
    assert worker._received is True
