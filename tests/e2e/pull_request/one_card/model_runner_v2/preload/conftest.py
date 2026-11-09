# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Shared fixtures and CI selection for preload regression tests."""

import json
from contextlib import ExitStack
from typing import Any

import pytest

from .helpers import daemon, run_engine


@pytest.fixture(scope="module")
def npu_environment():
    # Initialize the backend only when executing a test, not during collection.
    import torch
    import torch_npu  # noqa: F401

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
        torch.npu.set_device(0)
        yield


@pytest.fixture(scope="module")
def preload_environment(npu_environment):
    from vllm.model_executor.model_loader.weight_cache import daemon
    from vllm.model_executor.model_loader.weight_cache.ipc_loader import IpcModelLoader

    required = (
        hasattr(daemon, "exported_weight_bytes"),
        hasattr(daemon.WeightCacheDaemon, "release_distributed"),
        hasattr(IpcModelLoader, "_daemon_memory"),
    )
    if not all(required):
        pytest.skip("Installed vLLM lacks the preload memory accounting and release interfaces")


@pytest.fixture(scope="module")
def preload_model(request):
    return request.config.getoption("preload_model", default="Qwen/Qwen3-0.6B")


@pytest.fixture(scope="module")
def preload_baseline(preload_environment, preload_model, tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("preload-baselines")
    baselines: dict[tuple[str, bool, str], dict[str, Any]] = {}

    def load(*, dtype="bfloat16", graph=False, hf_overrides=None):
        key = (dtype, graph, json.dumps(hf_overrides, sort_keys=True))
        if key not in baselines:
            baselines[key] = run_engine(
                output_dir,
                preload_model,
                f"baseline-{len(baselines)}",
                dtype=dtype,
                graph=graph,
                hf_overrides=hf_overrides,
            )
        return baselines[key]

    return load


@pytest.fixture
def preload_run(preload_environment, preload_model, tmp_path):
    def run(name, **kwargs):
        return run_engine(tmp_path, preload_model, name, **kwargs)

    return run


@pytest.fixture
def preload_daemon(preload_environment, preload_model, tmp_path):
    # ExitStack also stops every producer when an assertion or a consumer fails.
    with ExitStack() as stack:

        def start(name="preload", dtype="bfloat16", **kwargs):
            return stack.enter_context(daemon(tmp_path, preload_model, name, dtype, **kwargs))

        yield start


@pytest.fixture
def yarn_overrides():
    return {
        "rope_parameters": {
            "rope_type": "yarn",
            "factor": 2.0,
            "original_max_position_embeddings": 4096,
            "rope_theta": 1000000.0,
        }
    }


def pytest_collection_modifyitems(config, items):
    # File/directory targets run the PR suite. Nightly CI requests its class
    # explicitly so the existing workflows need no additional pytest flags.
    if any("::TestPreloadNightly" in target for target in config.args):
        return
    deselected = [item for item in items if getattr(item.cls, "__name__", None) == "TestPreloadNightly"]
    if deselected:
        items[:] = [item for item in items if item not in deselected]
        config.hook.pytest_deselected(items=deselected)
