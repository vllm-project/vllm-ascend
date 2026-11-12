# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the real loader with four ranks and bounded collective rendezvous."""

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from .rfork_test_support import _load_module


@pytest.fixture
def cluster(request, monkeypatch):
    """Run four loader ranks in threads with bounded collective rendezvous."""
    _load_module(monkeypatch, "vllm_ascend.model_loader.rfork.local_weights", "local_weights.py")
    module = request.getfixturevalue("rfork_helpers").loader
    size = 4
    local = threading.local()
    events = []
    errors: dict[tuple[int, str], Exception] = {}
    values = [torch.zeros(1, dtype=torch.int32) for _ in range(size)]
    rendezvous = threading.Barrier(size, timeout=3)
    construction = threading.Barrier(size, timeout=3)
    cpu_group = object()
    world = SimpleNamespace(cpu_group=cpu_group, world_size=size)
    monkeypatch.setattr(sys.modules["vllm.distributed.parallel_state"], "get_world_group", lambda: world, raising=False)

    def reduce(tensor, op, group):
        assert group is world.cpu_group
        values[local.rank] = tensor
        rendezvous.wait()
        numbers = [int(value.item()) for value in values]
        result = sum(numbers) if op == torch.distributed.ReduceOp.SUM else max(numbers)
        rendezvous.wait()
        tensor.fill_(result)
        rendezvous.wait()

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    monkeypatch.setattr(module, "get_ascend_config", lambda: SimpleNamespace(eplb_config=SimpleNamespace()))
    monkeypatch.setattr(module, "_is_dynamic_eplb_enabled", lambda vc: False)
    monkeypatch.setattr(module, "_snapshot_process_global_model_state", lambda vc: None)
    monkeypatch.setattr(module, "_reset_process_global_model_state", lambda *a: None)
    monkeypatch.setattr(module, "_prepare_rfork_model_state", lambda *a: fail("prepare"))
    monkeypatch.setattr(module, "load_local_weights", lambda *a: fail("host_load"))
    monkeypatch.setattr(module, "process_weights_after_loading", lambda *a: fail("layout"))
    monkeypatch.setattr(module, "record_loaded_local_weights", lambda *a: None)
    monkeypatch.setattr(module, "_refresh_rfork_flatquant_state", lambda *a: None)

    def validate_shared(*args):
        events.append((local.rank, "finalization"))
        fail("shared_validation")

    monkeypatch.setattr(module, "_validate_rfork_shared_experts", validate_shared)
    monkeypatch.setattr(module, "gc", SimpleNamespace(collect=lambda: None))
    monkeypatch.setattr(
        module.torch, "npu", SimpleNamespace(empty_cache=lambda: None, synchronize=lambda: None), raising=False
    )

    def initialize(**kwargs):
        events.append((local.rank, "construct"))
        construction.wait()
        fail("construct")
        return torch.nn.Module()

    def release(model):
        fail("host_cleanup")
        events.append((local.rank, "released"))

    def fallback(*args):
        assert sum(event == "released" for _, event in events) == size, "reconstruction before peer cleanup"
        initialize()
        events.append((local.rank, "fallback"))
        return torch.nn.Module()

    def fail(stage):
        error = errors.get((local.rank, stage))
        if error is not None:
            raise error

    class Session:
        loading = staticmethod(nullcontext)

        def __init__(self, rank):
            self.identity = SimpleNamespace(tp_rank=rank % 2, is_draft_model=False)

        def register_receiver_weights(self, *args):
            fail("register")
            return True

        def acquire_seed_lease(self):
            return (local.rank, "seed_miss") not in errors

        def read_weights_from_seed(self, *args):
            fail("transfer")
            events.append((local.rank, "transfer"))
            return True

        def validate_transferred_model_layout(self, *args):
            fail("finalize")

        def prepare_for_fallback(self):
            fail("cleanup")
            return True

    monkeypatch.setattr(module, "initialize_model", initialize)
    monkeypatch.setattr(module, "release_local_weights", release)
    monkeypatch.setattr(module, "_load_with_default_loader", fallback)
    monkeypatch.setattr(module, "_schedule_seed_publication", lambda *a, **kw: events.append((local.rank, "publish")))

    def run():
        def worker(rank):
            local.rank = rank
            loader = object.__new__(module.RForkModelLoader)
            loader.load_config = SimpleNamespace(device=None)

            def session(*args):
                fail("session")
                return Session(rank)

            loader._get_session = session
            config = SimpleNamespace(
                device_config=SimpleNamespace(device="cpu"), engram_config=SimpleNamespace(cpu_offload=True)
            )
            try:
                result = loader.load_model(config, SimpleNamespace(dtype=torch.bfloat16))
                events.append((rank, "return"))
                return result
            except Exception as exc:
                return exc

        with ThreadPoolExecutor(max_workers=size) as pool:
            return list(pool.map(worker, range(size)))

    return SimpleNamespace(run=run, errors=errors, events=events)


@pytest.mark.parametrize(
    "stage", ["prepare", "host_load", "layout", "seed_miss", "register", "transfer", "shared_validation", "finalize"]
)
def test_one_rank_failure_reconstructs_all_ranks_after_cleanup(cluster, stage):
    cluster.errors[2, stage] = RuntimeError("injected failure")
    results = cluster.run()
    assert all(isinstance(result, torch.nn.Module) for result in results), results
    assert sum(event == "fallback" for _, event in cluster.events) == 4
    assert sum(event == "publish" for _, event in cluster.events) == (3 if stage == "prepare" else 4)
    if stage in ("layout", "seed_miss", "register", "transfer"):
        assert not any(event == "finalization" for _, event in cluster.events)
    first_publish = next(i for i, (_, event) in enumerate(cluster.events) if event == "publish")
    assert all(event != "transfer" for _, event in cluster.events[first_publish:])


def test_all_success_never_reconstructs(cluster):
    results = cluster.run()
    assert all(isinstance(result, torch.nn.Module) for result in results), results
    assert sum(event == "construct" for _, event in cluster.events) == 4
    assert not any(event == "released" for _, event in cluster.events)


@pytest.mark.parametrize("stage", ["cleanup", "host_cleanup"])
def test_one_rank_cleanup_failure_stops_every_rank_before_reconstruction(cluster, stage):
    cluster.errors[2, "seed_miss"] = RuntimeError("injected seed miss")
    cluster.errors[1, stage] = RuntimeError("injected cleanup failure")
    results = cluster.run()
    assert all(isinstance(result, RuntimeError) for result in results), results
    assert sum(event == "construct" for _, event in cluster.events) == 4
    assert not any(event in ("publish", "return") for _, event in cluster.events)


@pytest.mark.parametrize("stage", ["construct", "transfer"])
def test_unrecoverable_error_aborts_all_ranks(cluster, stage):
    cluster.errors[2, stage] = TypeError("non-recoverable error")
    results = cluster.run()
    assert all(isinstance(result, RuntimeError) for result in results), results
    assert not any(event in ("fallback", "publish", "return") for _, event in cluster.events)


@pytest.mark.parametrize("engram,world_size", [(None, 4), (object(), 1)])
def test_no_coordination_without_engram_or_with_one_rank(request, monkeypatch, engram, world_size):
    _load_module(monkeypatch, "vllm_ascend.model_loader.rfork.local_weights", "local_weights.py")
    module = request.getfixturevalue("rfork_helpers").loader

    def world():
        assert engram is not None, "Non-Engram loading must not access distributed groups"
        return SimpleNamespace(world_size=world_size, cpu_group=object())

    monkeypatch.setattr(sys.modules["vllm.distributed.parallel_state"], "get_world_group", world, raising=False)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **kw: pytest.fail("Unexpected collective"))
    coordinator = module._RForkLoadCoordinator(SimpleNamespace(engram_config=engram))
    original = RuntimeError("original exception")
    with pytest.raises(RuntimeError) as caught, coordinator.stage("transfer"):
        raise original
    assert caught.value is original
