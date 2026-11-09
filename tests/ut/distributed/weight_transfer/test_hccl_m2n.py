# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import cached_property
from queue import Queue
from threading import Condition
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.distributed.utils import StatelessProcessGroup
from vllm.distributed.weight_transfer.base import ParamMeta, Shard, WeightSource
from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory, WeightTransferTrainerFactory
from vllm.distributed.weight_transfer.m2n_common import M2NLayout, M2NMesh, M2NParamMeta

from vllm_ascend.distributed.device_communicators.pyhccl import HcclSendRecvType, PyHcclCommunicator
from vllm_ascend.distributed.device_communicators.pyhccl_wrapper import HCCLLibrary, hcclDataTypeEnum
from vllm_ascend.distributed.weight_transfer import hccl_m2n_engine, register_engine
from vllm_ascend.distributed.weight_transfer.hccl_m2n_engine import (
    HCCLM2NTrainerInitInfo,
    HCCLM2NTrainerWeightTransferEngine,
    HCCLM2NWeightTransferEngine,
    _TransferSession,
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
    queues = {(src, dst): Queue() for src in range(2) for dst in range(2, 6)}
    stream = Mock()
    for rank, shard in enumerate(tensor.chunk(2, dim=src_dim)):
        pieces = plan_rank_transfers(tuple(tensor.shape), source, dst_layout, rank)
        transfer_parameter(_MemoryP2P(rank, queues), stream, shard, pieces, send=True)

    expected_shards = [tensor] * 4 if dst_dim is None else tensor.chunk(4, dim=dst_dim)
    for rank, expected in enumerate(expected_shards, start=2):
        pieces = plan_rank_transfers(tuple(tensor.shape), source, dst_layout, rank)
        received = torch.full(expected.shape, -1, dtype=expected.dtype)
        transfer_parameter(_MemoryP2P(rank, queues), stream, received, pieces, send=False)
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
    group.comm = None
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
    monkeypatch.setattr(reload, "initialize_layerwise_reload", Mock())
    monkeypatch.setattr(reload, "finalize_layerwise_reload", Mock())
    monkeypatch.setattr(hccl_m2n_engine, "HCCLLibrary", Mock())

    source_layout = M2NLayout(M2NMesh((2, 1), 0), (0, -1))
    dst_layout = M2NLayout(M2NMesh((1, 1), 2), None)
    meta = M2NParamMeta("weight", torch.float32, (4, 4), source_layout)
    queues = {(src, 2): Queue() for src in range(2)}
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
            futures = [pool.submit(trainer.send_weights) for trainer in trainers]
            for future in futures:
                future.result(timeout=10)
            torch.testing.assert_close(loaded["weight"], tensor)
            tensor.add_(100)
    assert reload.initialize_layerwise_reload.call_count == 2
    assert reload.finalize_layerwise_reload.call_count == 2


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
