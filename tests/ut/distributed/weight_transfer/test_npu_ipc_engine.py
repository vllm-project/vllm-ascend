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

import inspect
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
from vllm.distributed.weight_transfer.packed_tensor import PackedIpcChunk

from vllm_ascend.distributed.weight_transfer import npu_ipc_engine
from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
    NPUIPCTrainerWeightTransferEngine,
    NPUIPCWeightTransferEngine,
)

_MODULE = "vllm_ascend.distributed.weight_transfer.npu_ipc_engine"


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


def test_init_accepts_model_argument():
    """Bug 1: __init__ must accept the optional ``model`` argument."""
    params = inspect.signature(NPUIPCWeightTransferEngine.__init__).parameters
    assert "model" in params


def test_init_passes_model_to_super():
    captured: dict = {}

    def fake_init_v1(self, config, vllm_config, device, model):
        captured["args"] = (config, vllm_config, device, model)

    with patch.object(WeightTransferEngine, "__init__", fake_init_v1):
        device = torch.device("npu:0")
        NPUIPCWeightTransferEngine("config", "vllm_config", device, "model")
    assert captured["args"] == ("config", "vllm_config", device, "model")


def test_unpacked_send_stores_reduce_tensor_args_only():
    """Bug 2 (producer): the handle stores only the ``reduce_tensor`` args.

    This matches upstream vLLM's CUDA IPC engine, which drops the rebuild
    func and relies on the consumer using the well-known rebuild function.
    """
    rebuild_args = (None, None, None, None, None, None, 999, None)
    fake_reduce = MagicMock(return_value=("rebuild_func_sentinel", rebuild_args))
    reduce_target = "torch.multiprocessing.reductions.reduce_tensor"
    with patch(reduce_target, fake_reduce):
        from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import NPUIPCTrainerWeightTransferEngine

        engine = object.__new__(NPUIPCTrainerWeightTransferEngine)
        engine.client = MagicMock()
        engine.is_sender = True
        engine.gpu_uuid = "node-0"
        source = MagicMock()
        source.__iter__.return_value = iter([("model.weight", torch.zeros(3))])
        source.iter_full.return_value = iter([("model.weight", torch.zeros(3))])
        with patch.object(engine, "_all_gather_and_merge_handles", side_effect=lambda handles: handles):
            engine._send_unpacked(source)
        stored = engine.client.update_weights.call_args.args[0]["ipc_handles"][0]["node-0"]
        assert stored == rebuild_args


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


def test_start_weight_update():
    engine = object.__new__(NPUIPCWeightTransferEngine)
    engine.model = MagicMock()
    engine.packed = False
    mock_init = MagicMock()
    with _patch_reload_module(initialize=mock_init):
        engine.start_weight_update()
    mock_init.assert_called_once_with(engine.model)


def test_finish_weight_update():
    engine = object.__new__(NPUIPCWeightTransferEngine)
    engine.model = MagicMock()
    engine.model_config = MagicMock()
    engine.device = torch.device("cpu")
    engine._packed_importer = MagicMock()
    engine._packed_consumer = MagicMock()
    consumer = engine._packed_consumer
    mock_finalize = MagicMock()
    with _patch_reload_module(finalize=mock_finalize):
        engine.finish_weight_update()
    mock_finalize.assert_called_once_with(engine.model, engine.model_config, engine.device)
    consumer.close.assert_called_once_with()
    assert engine._packed_consumer is None


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


@pytest.mark.parametrize("is_sender", [False, True])
def test_packed_sender_waits_before_reusing_each_chunk(is_sender):
    engine = object.__new__(NPUIPCTrainerWeightTransferEngine)
    engine.client = MagicMock()
    engine.is_sender = is_sender
    engine.packed = True
    engine.gpu_uuid = "node-0"
    engine.device = torch.device("cpu")
    engine.device_index = 0
    engine.packed_buffer_size_bytes = 16
    engine.source = MagicMock()
    events: list[str] = []

    def chunks(**kwargs):
        assert kwargs["gpu_uuid"] == "node-0"
        assert kwargs["buffer_size_bytes"] == 16
        for i in range(2):
            yield PackedIpcChunk(
                names=[f"w{i}"],
                dtype_names=["float32"],
                shapes=[[4]],
                ipc_handle={"node-0": (i,)} if i == 0 else None,
                chunk_capacity=16 if i == 0 else None,
                last=i == 1,
            )
            assert events[-1] == "sync"
            events.append("reuse")

    engine.client.update_weights.side_effect = lambda payload: events.append("update")
    producer_target = "vllm.distributed.weight_transfer.ipc_engine.packed_ipc_producer"
    sync_target = "_barrier"
    with (
        patch.object(engine, "_all_gather_and_merge_handles", side_effect=lambda handles: handles),
        patch.object(engine, sync_target, side_effect=lambda: events.append("sync")),
        patch(producer_target, side_effect=chunks),
    ):
        engine._send_packed(engine.source)
    expected_chunk = ["update", "sync", "reuse"] if is_sender else ["sync", "reuse"]
    expected = expected_chunk + ["sync"] + expected_chunk
    assert events == expected
    assert engine.client.update_weights.call_count == (2 if is_sender else 0)
    for i, call in enumerate(engine.client.update_weights.call_args_list):
        payload = call.args[0]
        assert isinstance(payload, dict)
        assert payload["names"] == [f"w{i}"]
        assert payload["dtype_names"] == ["float32"]
        assert payload["shapes"] == [[4]]
        assert payload["last"] is (i == 1)
        assert payload.get("chunk_capacity") == (16 if i == 0 else None)


@pytest.mark.parametrize("is_sender", [False, True])
def test_multi_rank_handles_are_merged_and_synchronized_on_npu(is_sender):
    engine = object.__new__(NPUIPCTrainerWeightTransferEngine)
    engine.is_sender = is_sender
    local = [{"node-0": (0,)}, {"node-0": (1,)}]
    remote = [{"node-1": (2,)}, {"node-1": (3,)}]
    events = []

    def gather(output, handles):
        assert handles is local
        output[:] = [local, remote]
        events.append("gather")

    with (
        patch.object(torch.distributed, "is_initialized", return_value=True),
        patch.object(torch.distributed, "get_world_size", return_value=2),
        patch.object(torch.distributed, "all_gather_object", side_effect=gather),
        patch.object(torch.distributed, "barrier", side_effect=lambda: events.append("barrier")),
        patch.object(torch.accelerator, "synchronize", side_effect=lambda: events.append("npu")),
        patch.object(torch.cuda, "synchronize", side_effect=AssertionError("CUDA used")),
    ):
        result = engine._all_gather_and_merge_handles(local)
    assert events == ["gather", "barrier", "npu"]
    assert result == ([{**local[i], **remote[i]} for i in range(2)] if is_sender else [{}, {}])


def test_packed_receive_rejects_missing_layout():
    engine = _shared_packed_worker(MagicMock())
    info = engine.update_info_cls(names=["w"], dtype_names=["float32"], shapes=[[1]], ipc_handles={}, last=True)
    with pytest.raises(AssertionError):
        engine.receive_weights(info)
    engine.shutdown()
    return


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
    """Modern NPU support supplies device operations, not duplicate transfer loops."""
    for method in (
        "__init__",
        "init_transfer_engine",
        "start_weight_update",
        "update_weights",
        "receive_weights",
        "finish_weight_update",
        "shutdown",
    ):
        assert inspect.getattr_static(NPUIPCWeightTransferEngine, method) is inspect.getattr_static(
            IPCWeightTransferEngine, method
        ), method
    for method in (
        "__init__",
        "trainer_init",
        "send_weights",
        "_send",
        "_send_unpacked",
        "_send_packed",
        "_all_gather_and_merge_handles",
        "_do_send",
        "_export_ipc_tensor",
        "_synchronize_device",
    ):
        assert inspect.getattr_static(NPUIPCTrainerWeightTransferEngine, method) is inspect.getattr_static(
            IPCTrainerWeightTransferEngine, method
        ), method
    info = NPUIPCWeightTransferEngine.update_info_cls(
        names=[], dtype_names=[], shapes=[], ipc_handles={}, chunk_capacity=16, last=True
    )
    assert not hasattr(info, "tensor_sizes")
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


def test_packed_split_tensor_requires_last_chunk_before_finish():
    """The native importer cannot turn an incomplete modern transfer into success."""
    model = MagicMock()
    engine = _shared_packed_worker(model)
    info = engine.update_info_cls(
        names=["wide"],
        dtype_names=["float32"],
        shapes=[[8]],
        ipc_handles={"node-0": (None,) * 6 + (999,)},
        chunk_capacity=16,
    )
    with _patch_rebuild_npu_tensor(lambda *args: torch.arange(4.0).view(torch.uint8)):
        engine.receive_weights(info)
    model.load_weights.assert_not_called()
    with _patch_reload_module(), pytest.raises(RuntimeError, match="before its last chunk"):
        engine.finish_weight_update()
    assert engine._packed_consumer is None
