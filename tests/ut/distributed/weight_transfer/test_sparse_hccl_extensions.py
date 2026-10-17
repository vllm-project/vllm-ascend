# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.distributed.weight_transfer.sparse_hccl_engine import (
    SparseHCCLTrainerInitInfo,
    SparseHCCLTrainerWeightTransferEngine,
    SparseHCCLWeightTransferEngine,
    SparseHCCLWeightTransferUpdateInfo,
    _verify_dense_load,
)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([10.0, 20.0, 30.0, 40.0]))

    def load_weights(self, weights):
        for name, tensor in weights:
            if name == "weight":
                self.weight.data.copy_(tensor)


class EPLocalModel(DummyModel):
    def load_weights(self, weights):
        for name, tensor in weights:
            if name == "model.layers.0.mlp.experts.0.gate_proj.weight":
                self.weight.data.copy_(tensor)


class PipelineLocalModel(DummyModel):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(start_layer=24, end_layer=48)

    def load_weights(self, weights):
        for name, tensor in weights:
            if name == "model.layers.24.self_attn.q_proj.weight":
                self.weight.data.copy_(tensor)


def make_info(**overrides):
    values = {
        "names": ["weight"],
        "dtype_names": ["float32"],
        "shapes": [[4]],
        "num_updates_list": [2],
    }
    values.update(overrides)
    return SparseHCCLWeightTransferUpdateInfo(**values)


def make_engine(model=None):
    engine = object.__new__(SparseHCCLWeightTransferEngine)
    engine.model = model or DummyModel()
    return engine


def test_dense_verify_skips_experts_not_owned_by_local_ep_rank():
    model = EPLocalModel()
    current = model.weight.detach().clone()
    _verify_dense_load(
        model,
        [
            ("model.layers.0.mlp.experts.0.gate_proj.weight", current),
            ("model.layers.0.mlp.experts.32.gate_proj.weight", current),
        ],
    )


def test_dense_verify_keeps_non_expert_noop_strict():
    model = EPLocalModel()
    with pytest.raises(RuntimeError, match="did not touch any weight"):
        _verify_dense_load(model, [("model.layers.0.self_attn.bad.weight", model.weight)])


def test_dense_verify_skips_layers_owned_by_another_pipeline_stage():
    model = PipelineLocalModel()
    current = model.weight.detach().clone()
    _verify_dense_load(
        model,
        [
            ("model.layers.0.self_attn.q_proj.weight", current),
            ("model.layers.24.self_attn.q_proj.weight", current),
        ],
    )


def test_update_info_accepts_zero_updates():
    assert make_info(num_updates_list=[0]).num_updates_list == [0]


def test_update_info_accepts_dense_seed():
    info = make_info(encoding="dense", packed=True, num_updates_list=[4])
    assert info.encoding == "dense"


def test_update_info_rejects_unknown_encoding():
    with pytest.raises(ValueError, match="encoding"):
        make_info(encoding="xor")


def test_update_info_rejects_unknown_checksum_algorithm():
    with pytest.raises(ValueError, match="checksum algorithm"):
        make_info(checksum_algorithm="unknown")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dtype_names": []}, "`dtype_names` should be"),
        ({"shapes": []}, "`shapes` should be"),
        ({"num_updates_list": []}, "cannot be empty"),
        ({"num_updates_list": [1, 2]}, "`num_updates_list` should be"),
        ({"num_updates_list": [-1]}, "must be non-negative"),
    ],
)
def test_update_info_rejects_invalid_metadata(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_info(**overrides)


def test_start_allows_pipeline_parallelism():
    engine = make_engine()
    engine.parallel_config = SimpleNamespace(world_size=2, pipeline_parallel_size=2)
    engine.start_weight_update()


@pytest.mark.parametrize("world_size", [1, 4])
def test_start_and_finish_allow_tp(world_size):
    engine = make_engine()
    engine.parallel_config = SimpleNamespace(world_size=world_size, pipeline_parallel_size=1)
    engine.start_weight_update()
    engine.finish_weight_update()


def test_sparse_engine_does_not_support_draft_update():
    assert SparseHCCLWeightTransferEngine.supports_draft_weight_update is False


def test_receive_requires_initialized_group():
    engine = make_engine()
    engine.model_update_group = None
    with pytest.raises(RuntimeError, match="not initialized"):
        engine.receive_weights(make_info())


def test_shutdown_clears_group():
    engine = make_engine()
    engine.model_update_group = MagicMock()
    engine.shutdown()
    assert engine.model_update_group is None


def test_dense_seed_uses_reload_once_then_sparse_lifecycle(monkeypatch):
    from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLWeightTransferEngine

    engine = make_engine()
    engine._dense_seed_pending = True
    engine._dense_received = False
    start, finish = MagicMock(), MagicMock()
    monkeypatch.setattr(HCCLWeightTransferEngine, "start_weight_update", start)
    monkeypatch.setattr(HCCLWeightTransferEngine, "finish_weight_update", finish)
    engine.start_weight_update()
    engine._dense_received = True
    engine.finish_weight_update()
    engine.start_weight_update()
    engine.finish_weight_update()
    start.assert_called_once_with(engine)
    finish.assert_called_once_with(engine)
    assert not engine._dense_seed_pending


def test_packed_sender_uses_shared_init_info_and_preserves_dtype_runs(monkeypatch):
    from vllm.distributed.weight_transfer.sparse_nccl_engine import SparseWeightPatch

    assert SparseHCCLTrainerWeightTransferEngine.init_info_cls is SparseHCCLTrainerInitInfo
    engine = SparseHCCLTrainerWeightTransferEngine(client=MagicMock())
    engine._packed = True
    engine.model_update_group = SimpleNamespace(device=torch.device("cpu"))
    captured = []
    monkeypatch.setattr(
        engine, "send_packed_chunk", lambda info, indices, values: captured.append((info, indices, values))
    )
    patches = [
        SparseWeightPatch("weight", torch.tensor([i], dtype=torch.int32), torch.tensor([float(i)], dtype=dtype), (4,))
        for i, dtype in enumerate((torch.float32, torch.bfloat16, torch.float32))
    ]
    engine.send_weight_chunk(patches)
    assert [info.dtype_names for info, _, _ in captured] == [["float32"], ["bfloat16"], ["float32"]]
    assert [indices.tolist() for _, indices, _ in captured] == [[0], [1], [2]]
    assert all(info.packed for info, _, _ in captured)
    assert not engine.client.mock_calls


def test_preassembled_sender_checks_lengths_before_rpc(monkeypatch):
    client = MagicMock()
    engine = SparseHCCLTrainerWeightTransferEngine(client=client)
    engine.model_update_group = SimpleNamespace(device=torch.device("cpu"))
    with pytest.raises(ValueError, match="disagree with update metadata"):
        engine.send_packed_chunk(make_info(packed=True), torch.tensor([0], dtype=torch.int32), torch.ones(2))
    assert not client.mock_calls


def test_packed_receiver_repeated_names_verifies_full_live_state(monkeypatch):
    engine = make_engine()
    payloads = iter([torch.tensor([0, 3], dtype=torch.int32), torch.tensor([11.0, 44.0])])
    engine.model_update_group = SimpleNamespace(
        device=torch.device("cpu"), broadcast=lambda tensor, **kwargs: tensor.copy_(next(payloads))
    )
    monkeypatch.setattr(torch.npu, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.npu, "current_stream", lambda device=None: MagicMock())
    info = SparseHCCLWeightTransferUpdateInfo(
        names=["weight", "weight"],
        dtype_names=["float32", "float32"],
        shapes=[[4], [4]],
        num_updates_list=[1, 1],
        packed=True,
        verify=True,
    )
    engine.receive_weights(info)
    assert torch.equal(engine.model.weight, torch.tensor([11.0, 20.0, 30.0, 44.0]))
