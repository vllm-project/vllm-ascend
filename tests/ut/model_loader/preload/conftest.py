# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import inspect

import pytest
from vllm.model_executor.model_loader.weight_cache import daemon
from vllm.model_executor.model_loader.weight_cache.ipc_loader import IpcModelLoader
from vllm.v1.worker.utils import request_memory


@pytest.fixture
def require_preload_daemon_api():
    missing = []
    if not hasattr(daemon, "exported_weight_bytes"):
        missing.append("daemon.exported_weight_bytes")
    if not hasattr(daemon.WeightCacheDaemon, "release_distributed"):
        missing.append("WeightCacheDaemon.release_distributed")
    if missing:
        pytest.skip("Installed vLLM lacks Ascend preload interfaces: " + ", ".join(missing))


@pytest.fixture
def require_daemon_memory_query():
    if not hasattr(IpcModelLoader, "_daemon_memory"):
        pytest.skip("Installed vLLM lacks IpcModelLoader._daemon_memory")


@pytest.fixture
def require_external_weight_budget():
    if "external_weight_memory" not in inspect.signature(request_memory).parameters:
        pytest.skip("Installed vLLM request_memory lacks external_weight_memory")
