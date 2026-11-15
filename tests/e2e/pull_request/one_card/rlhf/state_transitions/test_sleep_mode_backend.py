# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import os

import pytest
from vllm import LLM

from tests.e2e.common.single_node.sleep_backend import (
    check_sleep_backup_rollback,
    check_sleep_request_boundary,
    run_sleep_backend_case,
    shutdown_sleep_backend_engine,
)


@pytest.mark.parametrize("backend", ["ascend_legacy", "ascend"])
@pytest.mark.parametrize("eager", [True, False])
def test_sleep_mode_backend(backend, eager, monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    llm = LLM(
        model=os.environ.get("VLLM_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        dtype="bfloat16",
        enable_sleep_mode=True,
        sleep_mode_backend=backend,
        enforce_eager=eager,
        max_model_len=1024,
        max_num_seqs=8,
        kv_cache_memory_bytes=512 * 2**20,
        enable_prefix_caching=False,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8]},
        additional_config={"rl_config": {"enabled": True, "sleep_mode_extra_cleanup": True}},
        worker_extension_cls="tests.e2e.common.single_node.sleep_backend.SleepBackendProbe",
    )
    try:
        run_sleep_backend_case(llm, cycles=2, eager=eager, new_backend=backend == "ascend")
        if backend == "ascend":
            check_sleep_backup_rollback(llm)
            check_sleep_request_boundary(llm)
    finally:
        shutdown_sleep_backend_engine(llm)
