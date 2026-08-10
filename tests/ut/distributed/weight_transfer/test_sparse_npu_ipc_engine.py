# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.distributed.weight_transfer.sparse_weight_patch import SparseWeightPatch
from vllm_ascend.distributed.weight_transfer.sparse_npu_ipc_engine import (
    SparseNPUIPCWeightTransferEngine,
    SparseNPUIPCWeightTransferUpdateInfo,
)

_MODULE = "vllm_ascend.distributed.weight_transfer.sparse_npu_ipc_engine"
UUID = "host-0"
REBUILD_ARGS = (None, None, None, None, None, None, 99, None)


def make_info(**overrides):
    values = {
        "names": ["weight"],
        "dtype_names": ["float32"],
        "shapes": [[4]],
        "num_updates_list": [2],
        "indices_ipc_handles": [{UUID: REBUILD_ARGS}],
        "values_ipc_handles": [{UUID: REBUILD_ARGS}],
    }
    values.update(overrides)
    return SparseNPUIPCWeightTransferUpdateInfo(**values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dtype_names": []}, "dtype_names"),
        ({"shapes": []}, "shapes"),
        ({"num_updates_list": []}, "cannot be empty"),
        ({"indices_ipc_handles": []}, "indices_ipc_handles"),
        ({"values_ipc_handles": []}, "values_ipc_handles"),
        ({"num_updates_list": [-1]}, "non-negative"),
        ({"packed": True}, "packed=True"),
    ],
)
def test_update_info_rejects_invalid_metadata(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_info(**overrides)


def test_receive_rebuilds_paired_tensors_and_applies_patch():
    indices = torch.tensor([1, 3], dtype=torch.int32)
    values = torch.tensor([200.0, 400.0])
    rebuilt = iter((indices, values))
    seen_args = []

    def fake_rebuild(*args):
        seen_args.append(args)
        return next(rebuilt)

    reductions = types.ModuleType("torch_npu.multiprocessing.reductions")
    reductions.rebuild_npu_tensor = fake_rebuild
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.tensor([10.0, 20.0, 30.0, 40.0]))
    engine = object.__new__(SparseNPUIPCWeightTransferEngine)
    engine.model = model
    engine.device = torch.device("npu:0")

    with (
        patch.dict(
            sys.modules,
            {
                "torch_npu.multiprocessing": types.ModuleType(
                    "torch_npu.multiprocessing"
                ),
                "torch_npu.multiprocessing.reductions": reductions,
            },
        ),
        patch(f"{_MODULE}.npu_generate_uuid", return_value=UUID),
    ):
        engine.receive_weights(make_info())

    assert seen_args[0][6] == 0
    assert seen_args[1][6] == 0
    assert torch.equal(model.weight, torch.tensor([10.0, 200.0, 30.0, 400.0]))


def test_receive_rejects_missing_physical_npu_handle():
    engine = object.__new__(SparseNPUIPCWeightTransferEngine)
    engine.model = MagicMock()
    engine.device = torch.device("npu:0")
    with (
        patch(f"{_MODULE}.npu_generate_uuid", return_value="missing"),
        pytest.raises(ValueError, match="same physical NPU"),
    ):
        engine.receive_weights(make_info())


def test_receive_rejects_declared_dtype_mismatch():
    indices = torch.tensor([1, 3], dtype=torch.int32)
    values = torch.tensor([2.0, 4.0])
    with (
        patch(
            f"{_MODULE}._rebuild_sparse_ipc_tensor",
            side_effect=[indices, values],
        ),
        pytest.raises(ValueError, match="declared dtype"),
    ):
        engine = object.__new__(SparseNPUIPCWeightTransferEngine)
        engine.model = MagicMock()
        engine.device = torch.device("npu:0")
        engine.receive_weights(make_info(dtype_names=["bfloat16"]))


def test_start_requires_tp1_pp1_and_draft_is_unsupported():
    engine = object.__new__(SparseNPUIPCWeightTransferEngine)
    engine.parallel_config = SimpleNamespace(world_size=2)
    with pytest.raises(NotImplementedError, match="TP=1 and PP=1"):
        engine.start_weight_update()
    assert SparseNPUIPCWeightTransferEngine.supports_draft_weight_update is False


def test_trainer_send_creates_paired_args_only_handles():
    captured = {}
    args = MagicMock()
    args.packed = False
    args.parameter_shapes = {"weight": [4]}
    args.send_mode = lambda info: captured.setdefault("info", info)
    args.llm_handle = None
    args.url = None
    reduce = MagicMock(
        side_effect=[("index_func", REBUILD_ARGS), ("value_func", REBUILD_ARGS)]
    )
    patch_value = SparseWeightPatch(
        "weight",
        torch.tensor([1, 3], dtype=torch.int32),
        torch.tensor([2.0, 4.0]),
    )

    with (
        patch(f"{_MODULE}.reduce_tensor", reduce),
        patch(f"{_MODULE}.npu_generate_uuid", return_value=UUID),
        patch.object(
            SparseNPUIPCWeightTransferEngine,
            "_do_send",
            wraps=SparseNPUIPCWeightTransferEngine._do_send,
        ),
        patch(
            f"{_MODULE}.all_gather_and_merge_handles",
            side_effect=lambda handles: handles,
        ),
        patch(
            f"{_MODULE}.is_rank_zero",
            return_value=True,
        ),
        patch(f"{_MODULE}.post_send_sync"),
    ):
        SparseNPUIPCWeightTransferEngine.trainer_send_weights(
            iter([patch_value]), args
        )

    info = captured["info"]
    assert reduce.call_count == 2
    assert info.indices_ipc_handles == [{UUID: REBUILD_ARGS}]
    assert info.values_ipc_handles == [{UUID: REBUILD_ARGS}]
    assert info.num_updates_list == [2]
    assert info.shapes == [[4]]


def test_trainer_send_requires_shapes_and_rejects_packed():
    args = MagicMock(parameter_shapes=None, packed=False)
    with pytest.raises(ValueError, match="parameter_shapes"):
        SparseNPUIPCWeightTransferEngine.trainer_send_weights(iter(()), args)

    args.packed = True
    with pytest.raises(ValueError, match="packed=True"):
        SparseNPUIPCWeightTransferEngine.trainer_send_weights(iter(()), args)
