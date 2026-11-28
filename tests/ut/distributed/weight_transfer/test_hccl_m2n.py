# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
from queue import Queue
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch
from vllm.distributed.weight_transfer.m2n_common import M2NLayout, M2NMesh

from vllm_ascend.distributed.device_communicators.pyhccl import HcclSendRecvType, PyHcclCommunicator
from vllm_ascend.distributed.device_communicators.pyhccl_wrapper import HCCLLibrary, hcclDataTypeEnum
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


def test_replicated_source_distributes_load_across_params():
    src = M2NLayout(M2NMesh((4, 1), 0), None)
    dst = M2NLayout(M2NMesh((1, 1), 4), None)
    selected_senders = [plan_rank_transfers((4, 4), src, dst, rank=4, param_index=index)[0].peer for index in range(8)]
    assert selected_senders == [0, 1, 2, 3, 0, 1, 2, 3]
    for index, sender in enumerate(selected_senders):
        for rank in range(4):
            pieces = plan_rank_transfers((4, 4), src, dst, rank=rank, param_index=index)
            assert len(pieces) == (1 if rank == sender else 0)


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
