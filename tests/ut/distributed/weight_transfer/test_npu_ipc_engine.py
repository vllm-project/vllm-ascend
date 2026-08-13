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

These cover two bugs that broke ``examples/rl/rlhf_http_npu_ipc.py``:

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
from vllm_ascend.utils import vllm_version_is

_MODULE = "vllm_ascend.distributed.weight_transfer.npu_ipc_engine"

IS_VLLM_026 = vllm_version_is("0.26.0")


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

    v0.26.0: static _send_unpacked on NPUIPCWeightTransferEngine.
    post-0.26.0: instance _send_unpacked on NPUIPCTrainerWeightTransferEngine.
    """
    rebuild_args = (None, None, None, None, None, None, 999, None)
    fake_reduce = MagicMock(return_value=("rebuild_func_sentinel", rebuild_args))

    captured = {}

    with patch(f"{_MODULE}.reduce_tensor", fake_reduce):
        if IS_VLLM_026:

            def send_mode(update_info):
                captured["update_info"] = update_info

            trainer_args = MagicMock()
            trainer_args.send_mode = send_mode
            trainer_args.packed = False

            iterator = iter([("model.weight", torch.zeros(3))])
            NPUIPCWeightTransferEngine._send_unpacked(iterator, trainer_args, "node-0")

            update_info = captured["update_info"]
            assert isinstance(update_info.ipc_handles, list)
            stored = update_info.ipc_handles[0]["node-0"]
        else:
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

    v0.26.0: ``packed`` in update_info, ``npu_generate_uuid(device_index)``.
    post-0.26.0: ``packed`` on engine instance, ``npu_generate_uuid()``.
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
    if IS_VLLM_026:
        kwargs["packed"] = False

    update_info = NPUIPCWeightTransferEngine.update_info_cls(**kwargs)

    engine = object.__new__(NPUIPCWeightTransferEngine)
    received: dict[str, list[tuple[str, torch.Tensor]]] = {}
    engine.model = MagicMock()
    engine.device = MagicMock(index=device_index)
    if not IS_VLLM_026:
        engine.packed = False
    engine.model.load_weights.side_effect = lambda weights: received.update(weights=weights)

    with (
        _patch_rebuild_npu_tensor(fake_rebuild),
        patch(f"{_MODULE}.npu_generate_uuid", return_value=npu_uuid) as mock_uuid,
    ):
        engine.receive_weights(update_info)

    if IS_VLLM_026:
        mock_uuid.assert_called_once_with(device_index)
    else:
        mock_uuid.assert_called_once_with()
    assert received["weights"][0][0] == "model.weight"
    assert torch.equal(received["weights"][0][1], rebuilt_weight)
    # Index 6 (device index) overwritten with the receiver's device.
    assert seen["args"][6] == device_index


def test_start_weight_update():
    """v0.26.0: calls initialize_layerwise_reload. post-0.26.0: no-op."""
    engine = object.__new__(NPUIPCWeightTransferEngine)
    engine.model = MagicMock()

    with patch("vllm.model_executor.model_loader.reload.initialize_layerwise_reload") as mock_init:
        engine.start_weight_update()

    if IS_VLLM_026:
        mock_init.assert_called_once_with(engine.model)
    else:
        mock_init.assert_not_called()


def test_finish_weight_update():
    """v0.26.0: calls finalize_layerwise_reload. post-0.26.0: no-op."""
    engine = object.__new__(NPUIPCWeightTransferEngine)
    engine.model = MagicMock()
    engine.model_config = MagicMock()

    with patch("vllm.model_executor.model_loader.reload.finalize_layerwise_reload") as mock_finalize:
        engine.finish_weight_update()

    if IS_VLLM_026:
        mock_finalize.assert_called_once_with(engine.model, engine.model_config)
    else:
        mock_finalize.assert_not_called()
