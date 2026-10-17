# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse HCCL must preserve the upstream contract without entering CUDA."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.distributed.weight_transfer.sparse_nccl_engine import (
    SparseNCCLTrainerWeightTransferEngine,
    SparseNCCLWeightTransferUpdateInfo,
)

from vllm_ascend.distributed.weight_transfer import sparse_hccl_engine as sparse
from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLWeightTransferEngine, HCCLWeightTransferInitInfo


@pytest.fixture
def npu_context(monkeypatch):
    stream = MagicMock()
    context_devices, stream_devices = [], []
    monkeypatch.setattr(torch.npu, "device", lambda device: context_devices.append(device) or nullcontext())
    monkeypatch.setattr(torch.npu, "current_stream", lambda device=None: stream_devices.append(device) or stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **kw: pytest.fail("HCCL must not use CUDA streams"))
    return stream, context_devices, stream_devices


class ShardedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(4.0).reshape(2, 2))

    def load_weights(self, weights):
        for name, source in weights:
            self.weight.copy_(source[:2])
        return {"weight"}


def test_reuses_upstream_contract_and_lifecycle():
    assert issubclass(sparse.SparseHCCLWeightTransferUpdateInfo, SparseNCCLWeightTransferUpdateInfo)
    engine = object.__new__(sparse.SparseHCCLWeightTransferEngine)
    engine.start_weight_update()
    engine.finish_weight_update()
    for method in ("send_weights", "send_weight_chunk", "_validate_patch"):
        assert getattr(sparse.SparseHCCLTrainerWeightTransferEngine, method) is getattr(
            SparseNCCLTrainerWeightTransferEngine, method
        )


def test_receive_uses_communicator_device_and_upstream_loader(npu_context):
    stream, contexts, streams = npu_context
    engine = object.__new__(sparse.SparseHCCLWeightTransferEngine)
    engine.model = ShardedModel()
    # Ambient/self.device differs: every receive must use the communicator.
    engine.device = torch.device("meta")
    payloads = iter([torch.tensor([1, 7], dtype=torch.int32), torch.tensor([99.0, 88.0])])
    calls = []

    def broadcast(tensor, src, stream):
        calls.append((tensor.dtype, src, stream))
        tensor.copy_(next(payloads))

    engine.model_update_group = SimpleNamespace(device=torch.device("cpu"), broadcast=broadcast)
    info = sparse.SparseHCCLWeightTransferUpdateInfo(["weight"], ["float32"], [[4, 2]], [2])
    engine.receive_weights(info)
    assert torch.equal(engine.model.weight, torch.tensor([[0.0, 99.0], [2.0, 3.0]]))
    assert contexts == streams == [torch.device("cpu")]
    assert calls == [(torch.int32, 0, stream), (torch.float32, 0, stream)]


@pytest.mark.parametrize("indices,values", [([0, 0], [1.0, 2.0]), ([0, 8], [1.0, 2.0]), ([0, 1], [1.0, float("nan")])])
def test_receive_invalid_patch_never_mutates_model(npu_context, indices, values):
    engine = object.__new__(sparse.SparseHCCLWeightTransferEngine)
    engine.model = ShardedModel()
    baseline = engine.model.weight.detach().clone()
    payloads = iter([torch.tensor(indices, dtype=torch.int32), torch.tensor(values)])
    engine.model_update_group = SimpleNamespace(
        device=torch.device("cpu"), broadcast=lambda tensor, **kw: tensor.copy_(next(payloads))
    )
    with pytest.raises(ValueError, match="Sparse checkpoint patches contain"):
        engine.receive_weights(sparse.SparseHCCLWeightTransferUpdateInfo(["weight"], ["float32"], [[4, 2]], [2]))
    assert torch.equal(engine.model.weight, baseline)


def test_chunk_and_one_shot_control_plane_and_buffer_sync(npu_context):
    stream, _, _ = npu_context
    client = MagicMock()
    engine = sparse.SparseHCCLTrainerWeightTransferEngine(client=client)
    group = SimpleNamespace(device=torch.device("cpu"), broadcast=MagicMock(), close=MagicMock())
    engine.model_update_group = group
    patch = sparse.SparseWeightPatch("weight", torch.tensor([1], dtype=torch.int32), torch.tensor([4.0]), (4, 2))
    engine.send_weight_chunk([patch])
    client.start_weight_update.assert_not_called()
    client.finish_weight_update.assert_not_called()
    engine.send_weights([patch])
    client.start_weight_update.assert_called_once_with()
    client.finish_weight_update.assert_called_once_with()
    sent = client.update_weights.call_args.args[0]
    assert {key: sent[key] for key in ("names", "dtype_names", "shapes", "num_updates_list")} == {
        "names": ["weight"],
        "dtype_names": ["float32"],
        "shapes": [[4, 2]],
        "num_updates_list": [1],
    }
    assert group.broadcast.call_count == 4
    assert stream.synchronize.call_count == 2
    engine.shutdown()
    engine.shutdown()
    group.close.assert_called_once_with()


def test_non_sender_and_empty_updates_never_touch_client():
    client = MagicMock()
    engine = sparse.SparseHCCLTrainerWeightTransferEngine(client=client, is_sender=False)
    engine.send_weights([object()])
    engine.send_weight_chunk([object()])
    client.assert_not_called()
    assert not client.mock_calls
    sender = sparse.SparseHCCLTrainerWeightTransferEngine(client=client)
    sender.send_weights([])
    assert not client.mock_calls


def test_source_rejected_before_initialization():
    with pytest.raises(ValueError, match="takes no WeightSource"):
        sparse.SparseHCCLTrainerWeightTransferEngine(client=MagicMock(), source=object())


def test_trainer_factory_and_init_payload(monkeypatch):
    client = MagicMock()
    group = MagicMock()
    monkeypatch.setattr(HCCLWeightTransferEngine, "trainer_init", lambda info: group)
    info = sparse.SparseHCCLTrainerInitInfo("127.0.0.1", 12345, 3, rank=0)
    engine = sparse.SparseHCCLTrainerWeightTransferEngine.trainer_init(info, client=client)
    assert engine.model_update_group is group
    client.init_weight_transfer_engine.assert_called_once_with(
        {"master_address": "127.0.0.1", "master_port": 12345, "rank_offset": 1, "world_size": 3, "dense_seed": False}
    )
    client.reset_mock()
    info.rank = 1
    engine = sparse.SparseHCCLTrainerWeightTransferEngine.trainer_init(info, client=client)
    assert engine.model_update_group is None and not client.mock_calls


def test_worker_init_accounts_for_dp_and_releases_old_group(monkeypatch):
    engine = object.__new__(sparse.SparseHCCLWeightTransferEngine)
    engine.parallel_config = SimpleNamespace(data_parallel_index=2, world_size=4, rank=3)
    previous = MagicMock()
    engine.model_update_group = previous
    initializer = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 7)
    monkeypatch.setattr(HCCLWeightTransferEngine, "_stateless_init_process_group", initializer)
    engine.init_transfer_engine(HCCLWeightTransferInitInfo("localhost", 12345, 1, 13))
    initializer.assert_called_once_with("localhost", 12345, 12, 13, device=7)
    previous.close.assert_called_once_with()


def test_noncontiguous_buffers_rejected_before_receive_rpc(npu_context):
    client = MagicMock()
    engine = sparse.SparseHCCLTrainerWeightTransferEngine(client=client)
    engine.model_update_group = SimpleNamespace(device=torch.device("cpu"))
    patch = sparse.SparseWeightPatch("weight", torch.arange(4, dtype=torch.int32)[::2], torch.ones(2), (4,))
    with pytest.raises(ValueError, match="contiguous"):
        engine.send_weight_chunk([patch])
    with pytest.raises(ValueError, match="contiguous"):
        engine.send_weights([patch])
    assert not client.mock_calls


def test_update_keeps_device_context_through_parent_synchronization(monkeypatch):
    engine = object.__new__(sparse.SparseHCCLWeightTransferEngine)
    engine.model = ShardedModel()
    engine.device = torch.device("meta")
    device = torch.device("cpu")
    payloads = iter([torch.tensor([1], dtype=torch.int32), torch.tensor([99.0])])
    engine.model_update_group = SimpleNamespace(
        device=device, broadcast=lambda tensor, **kwargs: tensor.copy_(next(payloads))
    )
    active = []
    synchronized = []

    @contextmanager
    def device_context(selected):
        active.append(selected)
        try:
            yield
        finally:
            active.pop()

    def synchronize():
        assert active == [device]
        synchronized.append(device)

    monkeypatch.setattr(torch.npu, "device", device_context)
    monkeypatch.setattr(torch.npu, "current_stream", lambda device: MagicMock())
    monkeypatch.setattr(torch.accelerator, "synchronize", synchronize)
    engine.update_weights(
        {"names": ["weight"], "dtype_names": ["float32"], "shapes": [[4, 2]], "num_updates_list": [1]}
    )
    assert synchronized == [device] and not active
    assert engine.model.weight[0, 1] == 99
