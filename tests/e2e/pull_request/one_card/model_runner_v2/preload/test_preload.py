# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Preload regression tests, grouped into PR and nightly CI targets."""

import gc
import json
import multiprocessing
import tempfile

import pytest
import torch
import torch_npu

from vllm_ascend.model_loader.preload.transport import NPUTensorEntry

from .helpers import (
    MEMORY_DRIFT_LIMIT_BYTES,
    ZERO_COPY_MAX_WEIGHT_FRACTION,
    cached_bytes,
    check_preload_result,
    check_rope_state,
    check_serve_api,
    check_tokens,
    expect_engine_failure,
    free_device_memory,
    ipc_alias_consumer,
    ipc_consumer,
    run_engine,
)

RESTART_COUNT = 5


class TestPreloadPR:
    """Basic IPC, model loading, failure handling and public serve API."""

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("view", ["contiguous", "transpose", "offset"])
    def test_nd_ipc_views_and_repeated_consumers(self, dtype, view):
        torch.npu.set_device(0)
        torch.npu.config.allow_internal_format = True
        base = torch.arange(256, dtype=torch.float32).to(dtype).npu().reshape(16, 16)
        tensor = base if view == "contiguous" else base.T if view == "transpose" else base[2:10, 1:9]
        context = multiprocessing.get_context("spawn")
        for _ in range(3):
            expected = tensor.cpu().tolist()
            entry = NPUTensorEntry.from_tensor(tensor, "param")
            parent, child = context.Pipe()
            process = context.Process(target=ipc_consumer, args=(child,))
            process.start()
            child.close()
            try:
                parent.send((entry, expected, tensor.stride(), tensor.storage_offset()))
                assert parent.poll(120), "NPU IPC consumer timed out"
                assert parent.recv() == "released"
                process.join(30)
                assert process.exitcode == 0
            finally:
                parent.close()
                if process.is_alive():
                    process.terminate()
                    process.join(30)
            torch.npu.synchronize()
            assert bool(torch.all(tensor == 7).item())
        del tensor, base
        gc.collect()
        torch.npu.empty_cache()

    def test_nz_ipc_fails_explicitly(
        self,
    ):
        torch.npu.set_device(0)
        torch.npu.config.allow_internal_format = True
        tensor = torch_npu.npu_format_cast(torch.ones(32, 32, dtype=torch.bfloat16, device="npu"), 29)
        with pytest.raises(ValueError, match="NZ IPC"):
            NPUTensorEntry.from_tensor(tensor, "param")

    def test_npu_ipc_preserves_tied_parameters_and_shared_views(
        self,
    ):
        from vllm.model_executor.model_loader.weight_cache.protocol import WeightCacheState

        torch.npu.set_device(0)
        torch.npu.config.allow_internal_format = True
        tensor = torch.zeros(16, 16, dtype=torch.bfloat16, device="npu")
        state = WeightCacheState(
            entries={
                "weight": NPUTensorEntry.from_tensor(tensor, "param"),
                "view": NPUTensorEntry.from_tensor(tensor[2:10, 1:9], "buffer"),
            },
            aliases={"tied": "weight"},
            attrs={},
        )
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=ipc_alias_consumer, args=(child,))
        process.start()
        child.close()
        try:
            parent.send(state)
            assert parent.poll(120), "NPU IPC alias consumer timed out"
            assert parent.recv() == "released"
            process.join(30)
            assert process.exitcode == 0
        finally:
            parent.close()
            if process.is_alive():
                process.terminate()
                process.join(30)
        torch.npu.synchronize()
        torch.testing.assert_close(tensor[2:10, 1:9], torch.full_like(tensor[2:10, 1:9], 7))
        del tensor
        gc.collect()
        torch.npu.empty_cache()

    @pytest.mark.parametrize("mode", ["zero_copy", "copy"])
    def test_preload_loading(self, preload_baseline, preload_daemon, preload_run, mode):
        baseline = preload_baseline()
        socket_dir = preload_daemon()
        before_free = free_device_memory()
        before_bytes = cached_bytes(socket_dir)
        actual = preload_run(mode, mode=mode, socket_dir=socket_dir)
        check_preload_result(actual, baseline, mode, socket_dir, before_free, before_bytes)

    @pytest.mark.parametrize("scenario", ["missing_daemon", "dtype_mismatch"])
    @pytest.mark.parametrize("fallback", [False, True], ids=["strict", "fallback"])
    def test_preload_failure_modes(
        self,
        preload_baseline,
        preload_daemon,
        preload_run,
        preload_model,
        tmp_path,
        scenario,
        fallback,
    ):
        dtype = "float16" if scenario == "dtype_mismatch" else "bfloat16"
        baseline = preload_baseline(dtype=dtype) if fallback else None
        # Keep socket paths below AF_UNIX's limit, even with pytest's long node IDs.
        with tempfile.TemporaryDirectory(prefix="ap-", dir="/tmp") as unavailable:
            socket_dir = preload_daemon() if scenario == "dtype_mismatch" else unavailable
            kwargs = dict(mode="zero_copy", dtype=dtype, socket_dir=socket_dir, fallback=fallback)
            if fallback:
                check_tokens(preload_run(scenario, **kwargs), baseline)
            else:
                expected = "WeightCacheKey mismatch" if scenario == "dtype_mismatch" else "did not become ready within"
                expect_engine_failure(tmp_path, preload_model, scenario, expected, **kwargs)

    def test_preload_serve(self, preload_baseline, preload_daemon, preload_model, tmp_path):
        baseline = preload_baseline()
        socket_dir = preload_daemon()
        check_serve_api(tmp_path, preload_model, baseline, socket_dir)


class TestPreloadNightly:
    """Representative graph/YaRN combinations, restarts and NZ compatibility."""

    @pytest.mark.parametrize(
        "dtype,graph,mode",
        [
            pytest.param("bfloat16", True, "zero_copy", id="bf16-graph-zero_copy"),
            pytest.param("bfloat16", True, "copy", id="bf16-graph-copy"),
            pytest.param("float16", False, "copy", id="fp16-eager-copy"),
            pytest.param("float16", True, "zero_copy", id="fp16-graph-zero_copy"),
        ],
    )
    def test_preload_matrix(self, preload_baseline, preload_daemon, preload_run, dtype, graph, mode):
        baseline = preload_baseline(dtype=dtype, graph=graph)
        socket_dir = preload_daemon(dtype=dtype)
        before_free = free_device_memory()
        before_bytes = cached_bytes(socket_dir)
        actual = preload_run(mode, mode=mode, dtype=dtype, graph=graph, socket_dir=socket_dir)
        check_preload_result(actual, baseline, mode, socket_dir, before_free, before_bytes)

    def test_preload_repeated_restarts(self, preload_baseline, preload_daemon, preload_run, tmp_path):
        baseline = preload_baseline()
        socket_dir = preload_daemon()
        free_after_restart = []
        for iteration in range(RESTART_COUNT):
            actual = preload_run(f"restart-{iteration}", mode="zero_copy", socket_dir=socket_dir)
            check_tokens(actual, baseline)
            assert actual["worker_memory"][0]["model_memory_usage"] < (
                baseline["worker_memory"][0]["model_memory_usage"] * ZERO_COPY_MAX_WEIGHT_FRACTION
            )
            free_after_restart.append(free_device_memory())
        (tmp_path / "restart-memory.json").write_text(json.dumps(free_after_restart) + "\n")
        assert max(free_after_restart) - min(free_after_restart) <= MEMORY_DRIFT_LIMIT_BYTES, free_after_restart

    def test_disk_nz_loading(self, npu_environment, preload_model, tmp_path):
        run_engine(tmp_path, preload_model, "disk-nz", nz=2)

    @pytest.mark.parametrize(
        "dtype,graph,mode",
        [
            pytest.param("bfloat16", False, "zero_copy", id="bf16-eager-zero_copy"),
            pytest.param("float16", True, "copy", id="fp16-graph-copy"),
        ],
    )
    def test_preload_yarn(self, preload_baseline, preload_daemon, preload_run, yarn_overrides, dtype, graph, mode):
        common = dict(dtype=dtype, graph=graph, hf_overrides=yarn_overrides)
        baseline = preload_baseline(**common)
        check_rope_state(baseline)
        socket_dir = preload_daemon(dtype=dtype, hf_overrides=yarn_overrides)
        actual = preload_run(mode, mode=mode, socket_dir=socket_dir, **common)
        check_rope_state(actual)
        check_tokens(actual, baseline)
        if mode == "copy":
            assert cached_bytes(socket_dir) == 0
