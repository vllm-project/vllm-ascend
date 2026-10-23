# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend-specific sparse HCCL configuration and cleanup checks."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.distributed.weight_transfer.sparse_hccl_engine import (
    SparseHCCLTrainerWeightTransferEngine,
    SparseHCCLWeightTransferEngine,
    SparseWeightPatch,
    checkpoint_loader_views,
)


@pytest.mark.parametrize(
    "target,field,value",
    [
        ("parallel_config", "pipeline_parallel_size", 2),
        ("parallel_config", "enable_eplb", True),
        ("model_config", "quantization", "awq"),
    ],
)
def test_unsupported_configuration(target, field, value):
    engine = object.__new__(SparseHCCLWeightTransferEngine)
    engine.parallel_config = SimpleNamespace(pipeline_parallel_size=1, enable_eplb=False)
    engine.model_config = SimpleNamespace(quantization=None)
    engine.model = torch.nn.Module()
    setattr(getattr(engine, target), field, value)
    with (
        patch(
            "vllm_ascend.distributed.weight_transfer.sparse_hccl_engine.HCCLWeightTransferEngine.init_transfer_engine"
        ) as initialize,
        pytest.raises(NotImplementedError),
    ):
        engine.init_transfer_engine(MagicMock())
    initialize.assert_not_called()


@pytest.mark.parametrize("engine_class", [SparseHCCLWeightTransferEngine, SparseHCCLTrainerWeightTransferEngine])
def test_shutdown_closes_once(engine_class):
    engine = object.__new__(engine_class)
    group = MagicMock()
    engine.model_update_group = group
    engine.shutdown()
    engine.shutdown()
    group.close.assert_called_once()
    assert engine.model_update_group is None


def make_patch(**kwargs):
    fields = dict(
        name="checkpoint.weight",
        full_shape=(4, 4),
        indices=torch.tensor([0, 15], dtype=torch.int32),
        values=torch.tensor([2.0, 3.0]),
    )
    return SparseWeightPatch(**(fields | kwargs))


@pytest.mark.parametrize("empty_only", [False, True])
def test_upstream_broadcast_filters_empty_patches_and_syncs_npu(empty_only):
    engine = SparseHCCLTrainerWeightTransferEngine(client=MagicMock())
    engine.model_update_group = MagicMock(device=torch.device("cpu"))
    empty = make_patch(indices=torch.empty(0, dtype=torch.int32), values=torch.empty(0))
    full = make_patch()
    stream = MagicMock()
    with (
        patch("vllm_ascend.worker.v2.utils.torch_cuda_wrapper", return_value=nullcontext()) as wrapper,
        patch("torch.cuda.current_stream", return_value=stream),
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.npu.is_available", return_value=True),
        patch("torch.npu.current_stream", return_value=stream),
    ):
        engine._broadcast_chunk([empty] if empty_only else [empty, full])
    if empty_only:
        wrapper.assert_not_called()
        engine.client.update_weights.assert_not_called()
        engine.model_update_group.broadcast.assert_not_called()
        stream.synchronize.assert_not_called()
    else:
        wrapper.assert_called_once()
        engine.client.update_weights.assert_called_once_with(
            dict(names=[full.name], dtype_names=["float32"], shapes=[[4, 4]], num_updates_list=[2])
        )
        assert engine.model_update_group.broadcast.call_count == 2
        stream.synchronize.assert_called_once()


@pytest.mark.parametrize("failure", [None, "entry", "loading"])
def test_checkpoint_loader_views_unwind_all_layers(failure):
    events = []

    @contextmanager
    def view(layer):
        if failure == "entry" and layer.name == "second":
            raise RuntimeError("entry")
        events.append(("enter", layer.name))
        try:
            yield
        finally:
            events.append(("exit", layer.name))

    model = torch.nn.Module()
    for name in ("first", "second"):
        layer = torch.nn.Module()
        layer.name = name
        layer.quant_method = SimpleNamespace(checkpoint_weight_loader_view=view)
        model.add_module(name, layer)
    expected = [("enter", "first")]
    if failure != "entry":
        expected.append(("enter", "second"))
    with pytest.raises(RuntimeError, match=failure) if failure else nullcontext(), checkpoint_loader_views(model):
        assert events == expected
        if failure == "loading":
            raise RuntimeError("loading")
    expected += [("exit", name) for _, name in reversed(expected)]
    assert events == expected
