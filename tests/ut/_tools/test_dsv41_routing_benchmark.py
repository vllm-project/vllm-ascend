# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
from contextlib import contextmanager
from importlib.machinery import ModuleSpec, SourceFileLoader
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch


def test_distributed_setup_and_benchmark_keep_current_config(monkeypatch):
    active: SimpleNamespace | None = None
    calls: list[str | tuple[str, int]] = []

    @contextmanager
    def set_current_config(config):
        nonlocal active
        active = config
        yield
        active = None

    def init_world(**kwargs):
        assert active is not None
        assert active.parallel_config.tensor_parallel_size == 8
        assert kwargs == {"world_size": 8, "rank": 7, "local_rank": 7, "backend": "hccl"}
        calls.append("world")

    def init_tp(tensor_model_parallel_size):
        assert active is not None
        assert active.parallel_config.tensor_parallel_size == tensor_model_parallel_size == 8
        calls.append("tp")

    def run_benchmark(args, rank, world, progress):
        assert active is not None
        assert active.parallel_config.tensor_parallel_size == world == 8
        assert rank == 7
        assert args.num_tokens == 12
        calls.append("benchmark")

    config_module = ModuleType("vllm.config")
    config_module.__dict__.update(
        ParallelConfig=SimpleNamespace,
        VllmConfig=SimpleNamespace,
        set_current_vllm_config=set_current_config,
    )
    distributed_module = ModuleType("vllm.distributed")
    distributed_module.__dict__.update(
        init_distributed_environment=init_world,
        initialize_model_parallel=init_tp,
    )
    vllm_module = ModuleType("vllm")
    vllm_module.__path__ = []
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.config", config_module)
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed_module)
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(set_device=lambda device: calls.append(("device", device))), raising=False
    )
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("LOCAL_RANK", "7")
    monkeypatch.setenv("WORLD_SIZE", "8")

    path = Path(__file__).resolve().parents[3] / "benchmarks/dsv41_routing_input_ids.py"
    loader = SourceFileLoader("dsv41_routing_benchmark", str(path))
    module = importlib.util.module_from_spec(ModuleSpec(loader.name, loader))
    loader.exec_module(module)
    monkeypatch.setattr(module, "run_benchmark", run_benchmark)
    monkeypatch.setattr(sys, "argv", [str(path), "--num-tokens", "12"])
    module.main()

    assert calls == [("device", 7), "world", "tp", "benchmark"]
    assert active is None
