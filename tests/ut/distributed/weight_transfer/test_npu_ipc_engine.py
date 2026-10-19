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
"""Regression tests for the NPU IPC weight transfer engine.

These cover the bugs that broke ``examples/rl/rlhf_http_npu_ipc.py``:

1. ``NPUIPCWeightTransferEngine.__init__`` did not accept the ``model``
   argument that ``WeightTransferEngineFactory.create_engine`` passes,
   raising ``TypeError: __init__() takes 3 positional arguments but 4
   were given`` at engine construction.
2. ``receive_weights`` / ``packed_npu_ipc_consumer`` unpacked the stored
   IPC handle as ``func, args`` even though the producer stored only the
   ``reduce_tensor`` *args*, raising ``ValueError: too many values to
   unpack (expected 2)``. Aligned with upstream vLLM's CUDA IPC engine:
   the producer stores args only and the consumer rebuilds with the
   well-known ``rebuild_npu_tensor``.
3. The NPU IPC worker skipped the layerwise reload START/FINISH lifecycle,
   so runtime-formatted weights were not restored before loading and graph-
   visible storage was not preserved after loading.
4. The packed receive path decoded tensors but never passed them to
   ``model.load_weights``.
"""

import inspect
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.distributed.weight_transfer.base import WeightTransferEngine

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
    fake_mod.rebuild_npu_tensor = rebuild_func  # type: ignore[attr-defined]
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
    fake_mod.initialize_layerwise_reload = initialize or MagicMock()  # type: ignore[attr-defined]
    fake_mod.finalize_layerwise_reload = finalize or MagicMock()  # type: ignore[attr-defined]
    return patch.dict(
        sys.modules,
        {"vllm.model_executor.model_loader.reload": fake_mod},
    )


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

    with patch("vllm.distributed.weight_transfer.ipc_engine.reduce_tensor", fake_reduce):
        from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
            NPUIPCTrainerWeightTransferEngine,
        )

        engine = object.__new__(NPUIPCTrainerWeightTransferEngine)
        engine.client = MagicMock()
        engine.is_sender = True
        engine.npu_uuid = "node-0"
        engine.gpu_uuid = engine.npu_uuid
        source = iter([("model.weight", torch.zeros(3))])
        with patch.object(engine, "_all_gather_and_merge_handles", side_effect=lambda handles: handles):
            engine._send_unpacked(source)

        stored = engine.client.update_weights.call_args.args[0]["ipc_handles"][0]["node-0"]

        # Only the args tuple is stored, not a (func, args) pair.
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

    # Sender stores 999 at index 6; the receiver must overwrite it.
    rebuild_args = (None, None, None, None, None, None, 999, None)

    kwargs = dict(
        names=["model.weight"],
        dtype_names=["float32"],
        shapes=[[3]],
        ipc_handles=[{npu_uuid: rebuild_args}],
    )

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
    # Index 6 (device index) overwritten with the receiver's device.
    assert seen["args"][6] == device_index


def test_start_weight_update():
    engine = object.__new__(NPUIPCWeightTransferEngine)
    engine.model = MagicMock()
    mock_init = MagicMock()

    with _patch_reload_module(initialize=mock_init):
        engine.start_weight_update()

    mock_init.assert_called_once_with(engine.model)


def test_finish_weight_update():
    engine = object.__new__(NPUIPCWeightTransferEngine)
    engine.model = MagicMock()
    engine.model_config = MagicMock()
    engine._packed_importer = MagicMock()
    mock_finalize = MagicMock()

    with _patch_reload_module(finalize=mock_finalize):
        engine.finish_weight_update()

    mock_finalize.assert_called_once_with(engine.model, engine.model_config)
    engine._packed_importer.close.assert_called_once_with()


def test_receive_packed_weights_loads_model():
    packed_weights = [("model.weight", torch.tensor([1.0, 2.0, 3.0]))]
    update_info = MagicMock(
        tensor_sizes=[12],
        ipc_handles={"node-0": ("packed-handle",)},
        names=["model.weight"],
        shapes=[[3]],
        dtype_names=["float32"],
    )

    engine = object.__new__(NPUIPCWeightTransferEngine)
    engine.model = MagicMock()
    engine.device = MagicMock(index=0)
    engine.packed = True

    with (
        patch(f"{_MODULE}.npu_generate_uuid", return_value="node-0"),
        patch(
            f"{_MODULE}.packed_npu_ipc_consumer",
            return_value=packed_weights,
        ) as mock_consumer,
    ):
        engine.receive_weights(update_info)

    mock_consumer.assert_called_once_with(
        ipc_handle=update_info.ipc_handles,
        physical_npu_id="node-0",
        names=update_info.names,
        shapes=update_info.shapes,
        dtype_names=update_info.dtype_names,
        tensor_sizes=update_info.tensor_sizes,
        device_index=0,
    )
    engine.model.load_weights.assert_called_once_with(packed_weights)


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
            client=client,
            source=source,
            packed=packed,
            packed_buffer_size_bytes=16,
        )
    assert engine.source is source
    assert engine.device_index == 2
    assert engine.gpu_uuid == engine.npu_uuid == "node-7"
    assert engine.packed is packed
    assert engine.packed_buffer_size_bytes == 16
    uuid.assert_called_once_with(2)


@pytest.mark.parametrize("is_sender", [False, True])
def test_packed_sender_waits_before_reusing_each_chunk(is_sender):
    engine = object.__new__(NPUIPCTrainerWeightTransferEngine)
    engine.client = MagicMock()
    engine.is_sender = is_sender
    engine.packed = True
    engine.npu_uuid = "node-0"
    engine.packed_buffer_size_bytes = 16
    engine.source = []
    events: list[str] = []

    def chunks(**kwargs):
        assert kwargs["npu_uuid"] == "node-0"
        assert kwargs["buffer_size_bytes"] == 16
        for i in range(2):
            yield dict(
                names=[f"w{i}"], dtype_names=["float32"], shapes=[[4]], tensor_sizes=[16], ipc_handle={"node-0": (i,)}
            )
            assert events[-1] == "sync"
            events.append("reuse")

    engine.client.update_weights.side_effect = lambda payload: events.append("update")
    with (
        patch.object(engine, "_all_gather_and_merge_handles", side_effect=lambda handles: handles),
        patch.object(engine, "_post_send_sync", side_effect=lambda: events.append("sync")),
        patch(f"{_MODULE}.packed_npu_ipc_producer", side_effect=chunks),
    ):
        engine._send_packed(engine.source)
    expected_chunk = ["update", "sync", "reuse"] if is_sender else ["sync", "reuse"]
    assert events == expected_chunk * 2
    assert engine.client.update_weights.call_count == (2 if is_sender else 0)


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
        patch.object(torch, "npu", types.SimpleNamespace(synchronize=lambda: events.append("npu")), create=True),
        patch.object(torch.cuda, "synchronize", side_effect=AssertionError("CUDA used")),
    ):
        result = engine._all_gather_and_merge_handles(local)
    assert events == ["gather", "barrier", "npu"]
    assert result == ([{**local[i], **remote[i]} for i in range(2)] if is_sender else [{}, {}])


def test_packed_receive_rejects_missing_sizes():
    engine = object.__new__(NPUIPCWeightTransferEngine)
    engine.device = types.SimpleNamespace(index=1)
    engine.packed = True
    engine.model = MagicMock()
    with (
        patch(f"{_MODULE}.npu_generate_uuid", return_value="node-1"),
        pytest.raises(ValueError, match="tensor_sizes.*required"),
    ):
        engine.receive_weights(types.SimpleNamespace(tensor_sizes=None))
    engine.model.load_weights.assert_not_called()


def test_receive_scopes_mtp_completeness_check_to_loading():
    from vllm.model_executor.model_loader.mtp_validation import is_mtp_completeness_check_enabled

    engine = object.__new__(NPUIPCWeightTransferEngine)
    engine.device = types.SimpleNamespace(index=1)
    engine.packed = False
    engine.model = MagicMock()
    info = engine.parse_update_info(
        dict(names=["w"], dtype_names=["float32"], shapes=[[1]], ipc_handles=[{"node-1": (None,) * 6 + (9,)}])
    )

    def load(weights):
        assert not is_mtp_completeness_check_enabled()
        raise RuntimeError("load failed")

    engine.model.load_weights.side_effect = load
    with (
        _patch_rebuild_npu_tensor(lambda *args: torch.ones(1)),
        patch(f"{_MODULE}.npu_generate_uuid", return_value="node-1"),
        pytest.raises(RuntimeError, match="load failed"),
    ):
        engine.receive_weights(info)
    assert is_mtp_completeness_check_enabled()


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
