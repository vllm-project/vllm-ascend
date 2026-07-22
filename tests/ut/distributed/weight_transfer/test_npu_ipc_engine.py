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
    npu_uuid = "node-0"

    rebuild_args = (None, None, None, None, None, None, 999, None)
    fake_reduce = MagicMock(return_value=("rebuild_func_sentinel", rebuild_args))

    captured = {}

    def send_mode(update_info):
        captured["update_info"] = update_info

    trainer_args = MagicMock()
    trainer_args.send_mode = send_mode
    trainer_args.packed = False

    iterator = iter([("model.weight", torch.zeros(3))])

    with patch(f"{_MODULE}.reduce_tensor", fake_reduce):
        NPUIPCWeightTransferEngine._send_unpacked(iterator, trainer_args, npu_uuid)

    update_info = captured["update_info"]
    assert isinstance(update_info.ipc_handles, list)
    stored = update_info.ipc_handles[0][npu_uuid]
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

    update_info = NPUIPCWeightTransferEngine.update_info_cls(
        names=["model.weight"],
        dtype_names=["float32"],
        shapes=[[3]],
        ipc_handles=[{npu_uuid: rebuild_args}],
        packed=False,
    )

    engine = object.__new__(NPUIPCWeightTransferEngine)
    received = {}

    def load_weights(weights):
        received["weights"] = weights

    with (
        _patch_rebuild_npu_tensor(fake_rebuild),
        patch(f"{_MODULE}.npu_generate_uuid", return_value=npu_uuid),
        patch("torch.accelerator.current_device_index", return_value=device_index),
    ):
        engine.receive_weights(update_info, load_weights)

    assert received["weights"][0][0] == "model.weight"
    assert torch.equal(received["weights"][0][1], rebuilt_weight)
    # Index 6 (device index) overwritten with the receiver's device.
    assert seen["args"][6] == device_index
