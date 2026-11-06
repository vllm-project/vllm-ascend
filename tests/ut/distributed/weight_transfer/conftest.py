# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU import boundary for HIXL protocol tests on the supported older Ray."""

import importlib
import sys
from unittest.mock import patch


def _import_trainer_without_nixl_transport() -> None:
    import ray

    module = "vllm.distributed.weight_transfer.sharded_rdt_trainer"
    try:
        importlib.import_module(module)
        return
    except ValueError as error:
        if "Invalid tensor transport NIXL" not in str(error):
            raise

    original_method = ray.method

    def cpu_method(*args, **kwargs):
        # These tests call the HIXL override, never Ray's NIXL tensor return.
        # Keep all other method options and restore Ray immediately after import.
        if str(kwargs.get("tensor_transport", "")).upper() == "NIXL":
            kwargs = {key: value for key, value in kwargs.items() if key != "tensor_transport"}
        return original_method(*args, **kwargs)

    with patch.object(ray, "method", cpu_method):
        importlib.import_module(module)


def pytest_collectstart(collector):
    if collector.path.name not in {
        "test_sharded_rdt_hixl_producer.py",
        "test_sharded_rdt_hixl_trainer.py",
    }:
        return
    # tests/ut/conftest.py installs a loader-less torch_npu module on CPU.
    # Real NPU runs must use their actual Ray/NIXL compatibility requirements.
    npu = sys.modules.get("torch_npu")
    if npu is not None and npu.__spec__ is not None and npu.__spec__.loader is None:
        _import_trainer_without_nixl_transport()
