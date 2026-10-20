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
import sys
import types
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.distributed.weight_transfer import npu_ipc_engine
from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
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

    with patch.object(npu_ipc_engine.WeightTransferEngine, "__init__", fake_init_v1):
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

    captured = {}

    with patch(f"{_MODULE}.reduce_tensor", fake_reduce):
        from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
            NPUIPCTrainerWeightTransferEngine,
        )

        engine = object.__new__(NPUIPCTrainerWeightTransferEngine)
        engine.client = MagicMock()
        engine.is_sender = True
        engine.npu_uuid = "node-0"
        engine._do_send = lambda **kw: captured.update(kw)
        engine._all_gather_and_merge_handles = lambda x: x
        engine._post_send_sync = MagicMock()

        source = iter([("model.weight", torch.zeros(3))])
        engine._send_unpacked(source)

        stored = captured["ipc_handles"][0]["node-0"]

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

    mock_uuid.assert_called_once_with()
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
    mock_finalize = MagicMock()

    with _patch_reload_module(finalize=mock_finalize):
        engine.finish_weight_update()

    mock_finalize.assert_called_once_with(engine.model, engine.model_config)


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
