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
"""Native cross-process IPC regression, without a model checkpoint.

The real trainer, worker, reducer and reload lifecycle exchange exact tensors.
This is transport coverage; it does not claim model-inference or LoRA coverage.
Only the matching upstream device-hook protocol is supported.
"""

import gc
import multiprocessing
import traceback
from multiprocessing.connection import Connection

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.config import DeviceConfig, VllmConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer import ModuleSource

from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
    NPUIPCTrainerInitInfo,
    NPUIPCTrainerWeightTransferEngine,
    NPUIPCWeightTransferEngine,
)

_TIMEOUT_SECONDS = 120


class _TensorSink(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights: dict[str, torch.Tensor] = {}

    def load_weights(self, weights):
        self.weights.update(weights)


def _snapshot(weights):
    return {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "values": tensor.detach().cpu().reshape(-1).tolist(),
            "storage_bytes": tensor.untyped_storage().nbytes(),
        }
        for name, tensor in weights.items()
    }


def _worker(connection: Connection, packed: bool):
    """ACK only after the real public update method has completed device work."""
    try:
        torch.npu.set_device(0)
        model = _TensorSink()
        worker = NPUIPCWeightTransferEngine(
            WeightTransferConfig(backend="npu_ipc"),
            VllmConfig(device_config=DeviceConfig(device="npu")),
            torch.device("npu:0"),
            model,
        )
        retained = None
        connection.send({"ok": True})
        while True:
            method, payload = connection.recv()
            result = None
            if method == "init":
                worker.init_transfer_engine(worker.parse_init_info(payload))
            elif method == "start":
                model.weights.clear()
                worker.start_weight_update()
            elif method == "update":
                worker.update_weights(payload)
            elif method == "finish":
                worker.finish_weight_update()
                torch.npu.synchronize()
                if packed and retained is None:
                    retained = dict(model.weights)
                result = {
                    "current": _snapshot(model.weights),
                    "retained": _snapshot(retained) if packed else None,
                }
            elif method == "shutdown":
                model.weights.clear()
                retained = None
                worker.shutdown()
                gc.collect()
                torch.npu.synchronize()
                connection.send({"ok": True})
                break
            else:
                raise AssertionError(f"Unknown IPC test command: {method}")
            connection.send({"ok": True, "result": result})
    except BaseException:
        connection.send({"ok": False, "traceback": traceback.format_exc()})
        raise
    finally:
        connection.close()


class _PipeClient:
    def __init__(self, connection):
        self.connection = connection
        self.rounds = []
        self.snapshots = []

    def _call(self, method, payload=None):
        self.connection.send((method, payload))
        assert self.connection.poll(_TIMEOUT_SECONDS), f"Worker timed out during {method}"
        response = self.connection.recv()
        assert response["ok"], response.get("traceback")
        return response.get("result")

    def init_weight_transfer_engine(self, info):
        self._call("init", info)

    def start_weight_update(self):
        self.rounds.append([])
        self._call("start")

    def update_weights(self, info):
        self.rounds[-1].append(info)
        self._call("update", info)

    def finish_weight_update(self, weight_version=None):
        self.snapshots.append(self._call("finish"))


def _source_module():
    device = torch.device("npu:0")
    module = torch.nn.Module()
    tensors = {
        "wide": torch.arange(37, dtype=torch.float32, device=device),
        "bytes": torch.tensor([1, 3, 5], dtype=torch.int8, device=device),
        "bf16": torch.arange(5, dtype=torch.bfloat16, device=device),
        "strided": torch.arange(12, dtype=torch.float16, device=device).reshape(3, 4).T,
        "empty": torch.empty((0, 2), dtype=torch.float32, device=device),
    }
    for name, tensor in tensors.items():
        module.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))
    assert not module.strided.is_contiguous()
    return module


def _assert_snapshot(actual, expected, *, owns_storage):
    assert actual.keys() == expected.keys()
    for name, tensor in expected.items():
        result = actual[name]
        assert result["shape"] == list(tensor.shape)
        assert result["dtype"] == str(tensor.dtype)
        assert result["values"] == tensor.reshape(-1).tolist(), name
        if owns_storage:
            assert result["storage_bytes"] == tensor.nbytes, name


@pytest.mark.parametrize("packed", [False, True])
def test_native_npu_ipc_inherits_transport_across_processes(packed):
    """Two rounds preserve exact values, source tensors and owned packed outputs."""
    torch.npu.set_device(0)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_worker, args=(child, packed))
    process.start()
    child.close()
    trainer = None
    module = None
    try:
        assert parent.poll(_TIMEOUT_SECONDS), "IPC worker did not initialize"
        response = parent.recv()
        assert response["ok"], response.get("traceback")
        client = _PipeClient(parent)
        module = _source_module()
        trainer = NPUIPCTrainerWeightTransferEngine.trainer_init(
            NPUIPCTrainerInitInfo(
                rank=0,
                packed=packed,
                packed_buffer_size_bytes=32,
            ),
            client=client,
            source=ModuleSource(module),
        )
        first = {name: tensor.detach().cpu().clone() for name, tensor in module.named_parameters()}
        for revision in range(2):
            expected = {name: tensor.detach().cpu().clone() for name, tensor in module.named_parameters()}
            trainer.send_weights()
            _assert_snapshot(client.snapshots[-1]["current"], expected, owns_storage=packed)
            for name, tensor in module.named_parameters():
                assert torch.equal(tensor.detach().cpu(), expected[name]), name
            if packed:
                _assert_snapshot(client.snapshots[-1]["retained"], first, owns_storage=True)
            if revision == 0:
                with torch.no_grad():
                    for tensor in module.parameters():
                        tensor.add_(17)
        if packed:
            for updates in client.rounds:
                assert len(updates) > 1
                assert updates[0]["ipc_handles"]
                assert updates[0]["chunk_capacity"] == 32
                assert all(not update["ipc_handles"] for update in updates[1:])
                assert [update["last"] for update in updates] == [False] * (len(updates) - 1) + [True]
                assert all("tensor_sizes" not in update for update in updates)
        client._call("shutdown")
        process.join(_TIMEOUT_SECONDS)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(_TIMEOUT_SECONDS)
        if trainer is not None:
            trainer.shutdown()
        trainer = None
        module = None
        gc.collect()
        torch.npu.synchronize()
        parent.close()
