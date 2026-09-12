# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
import torch

from vllm_ascend.distributed.weight_transfer.hccl_engine import (
    HCCLTrainerSendWeightsArgs,
)
from vllm_ascend.distributed.weight_transfer.sparse_hccl_engine import (
    SparseHCCLWeightTransferEngine,
    SparseHCCLWeightTransferUpdateInfo,
    SparseWeightPatch,
)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor([10.0, 20.0, 30.0, 40.0])
        )


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


def test_update_info_accepts_zero_updates():
    assert make_info(num_updates_list=[0]).num_updates_list == [0]


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


def test_start_requires_tp1_pp1():
    engine = make_engine()
    engine.parallel_config = SimpleNamespace(world_size=2)
    with pytest.raises(NotImplementedError, match="TP=1 and PP=1"):
        engine.start_weight_update()


def test_start_and_finish_are_noops_for_world_size_one():
    engine = make_engine()
    engine.parallel_config = SimpleNamespace(world_size=1)
    engine.start_weight_update()
    engine.finish_weight_update()


def test_sparse_engine_does_not_support_draft_update():
    assert SparseHCCLWeightTransferEngine.supports_draft_weight_update is False


def test_receive_requires_initialized_group():
    engine = make_engine()
    engine.model_update_group = None
    with pytest.raises(RuntimeError, match="not initialized"):
        engine.receive_weights(make_info())


def test_trainer_send_rejects_packed():
    args = HCCLTrainerSendWeightsArgs(group=MagicMock(), packed=True)
    with pytest.raises(ValueError, match="cannot be combined"):
        SparseHCCLWeightTransferEngine.trainer_send_weights(iter(()), args)


def test_trainer_send_broadcasts_indices_then_values():
    group = MagicMock()
    stream = MagicMock()
    args = HCCLTrainerSendWeightsArgs(group=group, stream=stream)
    patch = SparseWeightPatch(
        "weight",
        torch.tensor([1, 3], dtype=torch.int32),
        torch.tensor([2.0, 4.0]),
    )

    SparseHCCLWeightTransferEngine.trainer_send_weights(iter([patch]), args)

    assert group.broadcast.call_args_list == [
        call(patch.indices, src=0, stream=stream),
        call(patch.values, src=0, stream=stream),
    ]


def test_shutdown_clears_group():
    engine = make_engine()
    engine.model_update_group = MagicMock()
    engine.shutdown()
    assert engine.model_update_group is None
