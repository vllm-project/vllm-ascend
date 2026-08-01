# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys

from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory

from vllm_ascend.distributed.weight_transfer.registry import register_ascend_weight_transfer_engines


_BACKEND_MODULES = {
    "hccl": "vllm_ascend.distributed.weight_transfer.hccl_engine",
    "npu_ipc": "vllm_ascend.distributed.weight_transfer.npu_ipc_engine",
    "nccl": "vllm_ascend.distributed.weight_transfer.hccl_engine",
    "ipc": "vllm_ascend.distributed.weight_transfer.npu_ipc_engine",
}


def test_register_ascend_weight_transfer_engines_registers_backends(monkeypatch):
    registry = {}
    monkeypatch.setattr(WeightTransferEngineFactory, "_registry", registry)

    register_ascend_weight_transfer_engines()
    register_ascend_weight_transfer_engines()

    assert set(registry) == {"hccl", "npu_ipc", "nccl", "ipc"}
    for backend, module_name in _BACKEND_MODULES.items():
        closure_values = {cell.cell_contents for cell in registry[backend].__closure__}
        assert module_name in closure_values


def test_register_ascend_weight_transfer_engines_can_skip_aliases(monkeypatch):
    registry = {}
    monkeypatch.setattr(WeightTransferEngineFactory, "_registry", registry)

    register_ascend_weight_transfer_engines(include_upstream_aliases=False)

    assert set(registry) == {"hccl", "npu_ipc"}


def test_register_ascend_weight_transfer_engines_respects_existing_entries(monkeypatch):
    existing_loader = object()
    registry = {"nccl": existing_loader}
    monkeypatch.setattr(WeightTransferEngineFactory, "_registry", registry)

    register_ascend_weight_transfer_engines(override_existing=False)

    assert registry["nccl"] is existing_loader
    assert "ipc" in registry


def test_register_ascend_weight_transfer_engines_does_not_import_backends(monkeypatch):
    registry = {}
    monkeypatch.setattr(WeightTransferEngineFactory, "_registry", registry)
    for module_name in _BACKEND_MODULES.values():
        sys.modules.pop(module_name, None)

    register_ascend_weight_transfer_engines()

    for module_name in _BACKEND_MODULES.values():
        assert module_name not in sys.modules
