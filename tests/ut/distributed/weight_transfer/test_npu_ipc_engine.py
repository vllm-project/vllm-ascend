#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""NPU device adapter tests over the shared upstream IPC lifecycle."""

import os
import sys
import types
import weakref
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.distributed.weight_transfer.base import WeightTransferEngine
from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerWeightTransferEngine, IPCWeightTransferEngine

from vllm_ascend.distributed.weight_transfer import npu_ipc_engine
from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
    NPUIPCTrainerWeightTransferEngine,
    NPUIPCWeightTransferEngine,
)

_MODULE = "vllm_ascend.distributed.weight_transfer.npu_ipc_engine"


def test_npu_completion_waits_after_the_trainer_barrier():
    """NPU export completion follows the inherited trainer-rank barrier."""
    events = []
    with (
        patch.object(torch.distributed, "is_initialized", return_value=True),
        patch.object(torch.distributed, "get_world_size", return_value=2),
        patch.object(torch.distributed, "barrier", side_effect=lambda: events.append("barrier")),
        patch.object(torch.npu, "synchronize", side_effect=lambda: events.append("npu")),
    ):
        NPUIPCTrainerWeightTransferEngine._post_send_sync()
    assert events == ["barrier", "npu"]


def _patch_rebuild_npu_tensor(rebuild_func):
    """Install a fake ``torch_npu.multiprocessing.reductions`` module.

    The engine imports ``rebuild_npu_tensor`` lazily from ``torch_npu``,
    which is only a stub on CPU CI runners, so provide a fake submodule.
    """
    fake_mod = types.ModuleType("torch_npu.multiprocessing.reductions")
    fake_mod.rebuild_npu_tensor = rebuild_func
    return patch.dict(
        sys.modules,
        {
            "torch_npu.multiprocessing": types.ModuleType("torch_npu.multiprocessing"),
            "torch_npu.multiprocessing.reductions": fake_mod,
        },
    )


def _patch_reload_module(*, initialize=None, finalize=None):
    """Provide the lazy-imported reload helpers without importing vLLM models."""
    fake_mod = types.ModuleType("vllm.model_executor.model_loader.reload")
    fake_mod.initialize_layerwise_reload = initialize or MagicMock()
    fake_mod.finalize_layerwise_reload = finalize or MagicMock()
    fake_mod.start_reload = fake_mod.initialize_layerwise_reload
    fake_mod.finish_reload = fake_mod.finalize_layerwise_reload
    return patch.dict(sys.modules, {"vllm.model_executor.model_loader.reload": fake_mod})


def test_receive_weights_rebuilds_with_rebuild_npu_tensor():
    """Bug 2 (consumer): receive_weights rebuilds via ``rebuild_npu_tensor``.

    Verifies the args-only handle is consumed without unpacking errors and
    that the receiver's device index is written into the rebuild args.
    """
    npu_uuid = "node-0"
    device_index = 0
    rebuilt_weight = torch.tensor([1.0, 2.0, 3.0])
    seen = {}

    def fake_rebuild(*args):
        seen["args"] = args
        return rebuilt_weight

    rebuild_args = (None, None, None, None, None, None, 999, None)
    kwargs = dict(names=["model.weight"], dtype_names=["float32"], shapes=[[3]], ipc_handles=[{npu_uuid: rebuild_args}])
    update_info = NPUIPCWeightTransferEngine.update_info_cls(**kwargs)
    engine = object.__new__(NPUIPCWeightTransferEngine)
    received: dict[str, list[tuple[str, torch.Tensor]]] = {}
    engine.model = MagicMock()
    engine.device = MagicMock(index=device_index)
    engine.packed = False
    engine.model.load_weights.side_effect = lambda weights: received.update(weights=weights)
    with (
        _patch_rebuild_npu_tensor(fake_rebuild),
        patch(f"{_MODULE}.npu_generate_uuid", return_value=npu_uuid) as mock_uuid,
    ):
        engine.receive_weights(update_info)
    mock_uuid.assert_called_once_with(device_index)
    engine.model.load_weights.assert_called_once()
    assert received["weights"][0][0] == "model.weight"
    assert torch.equal(received["weights"][0][1], rebuilt_weight)
    assert seen["args"][6] == device_index


@pytest.mark.parametrize("element_count", [0, 3])
def test_receive_packed_weights_loads_model(element_count):
    weights = torch.arange(1.0, element_count + 1)
    model = MagicMock()
    engine = _shared_packed_worker(model)
    info = engine.update_info_cls(
        names=["model.weight"],
        dtype_names=["float32"],
        shapes=[[element_count]],
        ipc_handles={"node-0": (None,) * 6 + (999,)} if element_count else {},
        chunk_capacity=16,
        last=True,
    )

    def rebuild(*args):
        assert element_count, "An empty-only transfer imported an IPC buffer"
        return weights.view(torch.uint8)

    with _patch_rebuild_npu_tensor(rebuild):
        engine.receive_weights(info)
    loaded = model.load_weights.call_args.args[0]
    assert loaded[0][0] == "model.weight"
    assert torch.equal(loaded[0][1], weights)
    engine.shutdown()
    return


@pytest.mark.parametrize("packed", [False, True])
def test_trainer_init_uses_assigned_npu_without_cuda(packed):
    client = MagicMock()
    source = [("w", torch.ones(1))]
    accelerator = types.SimpleNamespace(current_device_index=lambda: 2)
    with (
        patch.object(torch, "accelerator", accelerator, create=True),
        patch(f"{_MODULE}.npu_generate_uuid", return_value="node-7") as uuid,
        patch.object(torch.cuda, "get_device_properties", side_effect=AssertionError("CUDA used")),
    ):
        engine = NPUIPCTrainerWeightTransferEngine(
            client=client, source=source, packed=packed, packed_buffer_size_bytes=16
        )
    assert engine.source is source
    assert engine.device_index == 2
    assert engine.gpu_uuid == "node-7"
    assert engine.packed is packed
    assert engine.packed_buffer_size_bytes == 16
    uuid.assert_called_once_with(2)


def test_uuid_uses_assigned_device_with_visibility_mapping():
    npu_ipc_engine.npu_generate_uuid.cache_clear()
    try:
        with (
            patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "3,7"}),
            patch(f"{_MODULE}.get_ip", return_value="host"),
            patch.object(torch, "accelerator", types.SimpleNamespace(current_device_index=lambda: 0), create=True),
        ):
            assert npu_ipc_engine.npu_generate_uuid(1) == "host-7"
    finally:
        npu_ipc_engine.npu_generate_uuid.cache_clear()


@pytest.mark.parametrize("cleanup", ["finish_weight_update", "shutdown"])
def test_packed_export_imported_once_with_fresh_payload_and_release(cleanup):
    """Multiple chunks share an import, but load fresh independent weights."""
    _check_shared_packed_mapping_lifecycle(cleanup)
    return


def _shared_packed_worker(model):
    """Construct the real worker with external reload/device services replaced."""

    def initialize(self, config, vllm_config, device, model):
        self.device, self.model = (device, model)
        self.model_config = None

    with patch.object(WeightTransferEngine, "__init__", initialize):
        engine = NPUIPCWeightTransferEngine(None, None, torch.device("cpu:0"), model)
    engine.init_transfer_engine(engine.init_info_cls(packed=True))
    with _patch_reload_module(), patch(f"{_MODULE}.npu_generate_uuid", return_value="node-0"):
        engine.start_weight_update()
    return engine


def _check_shared_packed_mapping_lifecycle(cleanup):
    model = MagicMock()
    engine = _shared_packed_worker(model)
    staging = torch.tensor([1.0, 2.0, 3.0, 4.0])
    imports = []
    loaded = []

    def rebuild(*args):
        assert args[6] == 0
        assert args[-1] == f"export-{len(imports) + 1}"
        mapping = staging.view(torch.uint8)
        imports.append(weakref.ref(mapping))
        return mapping

    model.load_weights.side_effect = lambda weights: loaded.append(weights[0][1])
    first = engine.update_info_cls(
        names=["w"],
        dtype_names=["float32"],
        shapes=[[4]],
        ipc_handles={"node-0": (None,) * 6 + (999, "export-1")},
        chunk_capacity=16,
    )
    rest = engine.update_info_cls(names=["v"], dtype_names=["float32"], shapes=[[4]], ipc_handles={}, last=True)
    with (
        _patch_rebuild_npu_tensor(rebuild),
        _patch_reload_module(),
        patch(f"{_MODULE}.npu_generate_uuid", return_value="node-0"),
    ):
        engine.receive_weights(first)
        staging.add_(4)
        engine.receive_weights(rest)
        assert len(imports) == 1
        assert torch.equal(loaded[0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
        assert torch.equal(loaded[1], torch.tensor([5.0, 6.0, 7.0, 8.0]))
        getattr(engine, cleanup)()
        assert imports[0]() is None
        getattr(engine, cleanup)()
        engine.start_weight_update()
        engine.receive_weights(replace(first, ipc_handles={"node-0": (None,) * 6 + (999, "export-2")}, last=True))
        assert len(imports) == 2
        engine.shutdown()
        assert imports[1]() is None


def test_shared_ipc_uses_upstream_lifecycle_and_wire_schema():
    """Only the NPU extension points are maintained here; common methods inherit."""
    assert NPUIPCWeightTransferEngine.__bases__ == (IPCWeightTransferEngine,)
    assert NPUIPCTrainerWeightTransferEngine.__bases__ == (IPCTrainerWeightTransferEngine,)
    for engine, expected in (
        (NPUIPCWeightTransferEngine, {"_get_device_key", "_rebuild_ipc_tensor"}),
        (NPUIPCTrainerWeightTransferEngine, {"_get_device_key", "_post_send_sync"}),
    ):
        methods = {
            name
            for name, member in engine.__dict__.items()
            if isinstance(member, (types.FunctionType, classmethod, staticmethod))
        }
        assert methods == expected
    assert NPUIPCTrainerWeightTransferEngine.worker_init_info_cls is NPUIPCWeightTransferEngine.init_info_cls
    assert NPUIPCTrainerWeightTransferEngine.update_info_cls is NPUIPCWeightTransferEngine.update_info_cls


def test_incomplete_native_arguments_do_not_reach_the_model():
    """Malformed native handles fail before rebuilding or loading a weight."""
    engine = object.__new__(NPUIPCWeightTransferEngine)
    engine.device = torch.device("npu:0")
    engine.model = MagicMock()
    engine.packed = False
    info = engine.update_info_cls(
        names=["w"], dtype_names=["float32"], shapes=[[1]], ipc_handles=[{"node-0": (None,) * 6}]
    )
    rebuild = MagicMock()
    with (
        patch(f"{_MODULE}.npu_generate_uuid", return_value="node-0"),
        _patch_rebuild_npu_tensor(rebuild),
        pytest.raises(ValueError, match="do not contain a device index"),
    ):
        engine.receive_weights(info)
    rebuild.assert_not_called()
    engine.model.load_weights.assert_not_called()
