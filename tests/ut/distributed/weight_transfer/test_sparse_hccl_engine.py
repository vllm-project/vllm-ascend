# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse HCCL validation, rank calculation, and lifecycle tests."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLWeightTransferEngine, HCCLWeightTransferInitInfo
from vllm_ascend.distributed.weight_transfer.sparse_hccl_engine import (
    SparseHCCLTrainerWeightTransferEngine,
    SparseHCCLWeightTransferEngine,
    SparseHCCLWeightTransferUpdateInfo,
)
from vllm_ascend.distributed.weight_transfer.sparse_weight_patch import SparseWeightPatch, validate_sparse_patch


def make_patch(**kwargs):
    fields = dict(
        name="checkpoint.weight",
        full_shape=(4, 4),
        indices=torch.tensor([0, 15], dtype=torch.int32),
        values=torch.tensor([2.0, 3.0]),
    )
    return SparseWeightPatch(**(fields | kwargs))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": ""},
        {"full_shape": None},
        {"full_shape": (-1,)},
        {"full_shape": (1.5,)},
        {"indices": torch.tensor([0], dtype=torch.int64)},
        {"indices": torch.tensor([-1, 15], dtype=torch.int32)},
        {"indices": torch.tensor([0, 16], dtype=torch.int32)},
        {"indices": torch.tensor([0, 0], dtype=torch.int32)},
        {"indices": torch.tensor([[0, 15]], dtype=torch.int32)},
        {"values": torch.tensor([float("nan"), 3.0])},
        {"values": torch.tensor([2, 3])},
        {"values": torch.tensor([2.0])},
    ],
)
def test_bad_patch_rejected_before_start(kwargs):
    engine = SparseHCCLTrainerWeightTransferEngine(client=MagicMock())
    engine.model_update_group = MagicMock(device=torch.device("cpu"))
    with pytest.raises(ValueError):
        engine.send_weights([make_patch(**kwargs)])
    engine.client.start_weight_update.assert_not_called()
    engine.model_update_group.broadcast.assert_not_called()


def test_valid_and_empty_patch():
    validate_sparse_patch(make_patch())
    validate_sparse_patch(make_patch(indices=torch.empty(0, dtype=torch.int32), values=torch.empty(0)))


def test_large_checkpoint_bounds_do_not_wrap_int32():
    validate_sparse_patch(
        make_patch(
            full_shape=(65536, 32768),
            indices=torch.tensor([0, 2147483647], dtype=torch.int32),
        )
    )
    with pytest.raises(ValueError, match="int64"):
        validate_sparse_patch(make_patch(full_shape=(1 << 63,)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dtype_names": []},
        {"shapes": []},
        {"num_updates_list": []},
        {"num_updates_list": [-1]},
        {"num_updates_list": [1.5]},
        {"dtype_names": ["int32"]},
        {"dtype_names": ["invalid"]},
        {"shapes": [[-1]]},
        {"names": [""]},
    ],
)
def test_invalid_metadata(kwargs):
    fields = dict(names=["checkpoint.weight"], dtype_names=["float32"], shapes=[[4, 4]], num_updates_list=[2])
    with pytest.raises(ValueError):
        SparseHCCLWeightTransferUpdateInfo(**(fields | kwargs))


def make_worker(tp=4, dp=1, dp_rank=0, rank=0):
    engine = object.__new__(SparseHCCLWeightTransferEngine)
    engine.parallel_config = SimpleNamespace(
        world_size=tp,
        data_parallel_size=dp,
        data_parallel_index=dp_rank,
        rank=rank,
        pipeline_parallel_size=1,
        enable_eplb=False,
    )
    engine.model_config = SimpleNamespace(quantization=None)
    engine.model = torch.nn.Module()
    engine.model_update_group = None
    return engine


@pytest.mark.parametrize("dp,dp_rank,rank,expected", [(1, 0, 3, 4), (2, 0, 3, 4), (2, 1, 0, 5), (2, 1, 3, 8)])
def test_tp_dp_ranks(dp, dp_rank, rank, expected):
    engine = make_worker(dp=dp, dp_rank=dp_rank, rank=rank)
    info = HCCLWeightTransferInitInfo("127.0.0.1", 12345, 1, 4 * dp + 1)
    with (
        patch("torch.accelerator.current_device_index", return_value=0),
        patch.object(HCCLWeightTransferEngine, "_stateless_init_process_group") as create,
    ):
        engine.init_transfer_engine(info)
    create.assert_called_once_with("127.0.0.1", 12345, expected, 4 * dp + 1, device=0)


def test_invalid_world_size_rejected_before_rendezvous():
    engine = make_worker(dp=2)
    with patch.object(HCCLWeightTransferEngine, "_stateless_init_process_group") as create, pytest.raises(ValueError):
        engine.init_transfer_engine(HCCLWeightTransferInitInfo("127.0.0.1", 12345, 1, 5))
    create.assert_not_called()


@pytest.mark.parametrize("field,value", [("pipeline_parallel_size", 2), ("enable_eplb", True), ("quantization", "awq")])
def test_unsupported_configuration(field, value):
    engine = make_worker()
    setattr(engine.model_config if field == "quantization" else engine.parallel_config, field, value)
    with pytest.raises(NotImplementedError):
        engine.start_weight_update()


def test_caller_owned_chunks_and_one_shot_lifecycle():
    client = MagicMock()
    engine = SparseHCCLTrainerWeightTransferEngine(client=client)
    with (
        patch.object(engine, "_prepare_patches", side_effect=lambda p: list(p)),
        patch.object(engine, "_broadcast_chunk") as send,
    ):
        engine.send_weight_chunk([make_patch()])
        client.start_weight_update.assert_not_called()
        client.finish_weight_update.assert_not_called()
        engine.send_weights([make_patch()])
    assert send.call_count == 2
    client.start_weight_update.assert_called_once()
    client.finish_weight_update.assert_called_once()


def test_empty_and_non_sender_noop():
    for sender in (True, False):
        engine = SparseHCCLTrainerWeightTransferEngine(client=MagicMock(), is_sender=sender)
        engine.send_weights([])
        engine.send_weight_chunk([])
        engine.client.start_weight_update.assert_not_called()


def test_failure_does_not_finish_partial_update():
    engine = SparseHCCLTrainerWeightTransferEngine(client=MagicMock())
    with (
        patch.object(engine, "_prepare_patches", return_value=[make_patch()]),
        patch.object(engine, "_broadcast_chunk", side_effect=RuntimeError("failed")),
        pytest.raises(RuntimeError),
    ):
        engine.send_weights([make_patch()])
    engine.client.finish_weight_update.assert_not_called()


def test_receive_on_communicator_device():
    engine = make_worker()
    engine.model = MagicMock()
    engine.model_update_group = MagicMock(device=torch.device("cpu"))
    info = SparseHCCLWeightTransferUpdateInfo(["checkpoint.weight"], ["float32"], [[4, 4]], [0])
    with (
        patch("torch.npu.device", return_value=nullcontext()),
        patch("torch.npu.current_stream"),
        patch("vllm_ascend.distributed.weight_transfer.sparse_hccl_engine.load_checkpoint_weight_patches") as load,
    ):
        engine.receive_weights(info)
    payload = load.call_args.args[1][0]
    assert payload.name == "checkpoint.weight"
    assert payload.shape == (4, 4)
    assert payload.values.device.type == "cpu"
    engine.model_update_group.broadcast.assert_not_called()


def test_shutdown_closes_once():
    engine = make_worker()
    group = MagicMock()
    engine.model_update_group = group
    engine.shutdown()
    engine.shutdown()
    group.close.assert_called_once()


def test_prepare_moves_cpu_views_to_communicator_device():
    engine = SparseHCCLTrainerWeightTransferEngine(client=MagicMock())
    engine.model_update_group = MagicMock(device=torch.device("cpu"))
    indices = torch.tensor([0, 1, 15, 2], dtype=torch.int32)[::2]
    values = torch.tensor([2.0, 0.0, 3.0, 0.0])[::2]
    assert not values.is_contiguous()
    with patch("torch.npu.device", return_value=nullcontext()):
        prepared = engine._prepare_patches([make_patch(indices=indices, values=values)])
    assert prepared[0].indices.is_contiguous()
    assert prepared[0].values.is_contiguous()
    assert torch.equal(prepared[0].values, values)


def test_source_is_rejected():
    with pytest.raises(ValueError, match="WeightSource"):
        SparseHCCLTrainerWeightTransferEngine(client=MagicMock(), source=MagicMock())
