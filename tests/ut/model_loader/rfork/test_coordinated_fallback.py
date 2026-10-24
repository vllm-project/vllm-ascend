# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the real loader with four ranks and bounded collective rendezvous."""

import multiprocessing
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch

from .rfork_test_support import _load_module


@pytest.fixture
def cluster(request, monkeypatch):
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
    monkeypatch.setattr(module, "record_loaded_local_weights", lambda *a: None)
    monkeypatch.setattr(module, "_refresh_rfork_flatquant_state", lambda *a: None)
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
        def __init__(self, rank):
            self.identity = SimpleNamespace(tp_rank=rank % 2, is_draft_model=False)

        def register_destination(self, *args):
            fail("register")
            return True

        def acquire_seed(self):
            return (local.rank, "seed_miss") not in errors

        def transfer_from_seed(self, *args):
            fail("transfer")
            events.append((local.rank, "transfer"))
            return True

        def log_transferred_model_layout(self, *args):
            fail("finalize")

        def prepare_for_fallback(self):
            fail("cleanup")
            return SimpleNamespace(can_schedule_seed=True)

        def can_reuse_shared_weights(self, *args):
            return local.rank in reuse

    reuse: set[int] = set()
    monkeypatch.setattr(module, "initialize_model", initialize)
    monkeypatch.setattr(module, "release_local_weights", release)
    monkeypatch.setattr(module, "_load_with_default_loader", fallback)
    monkeypatch.setattr(module, "_start_rfork_seed_service", lambda *a, **kw: events.append((local.rank, "publish")))

    def run():
        def worker(rank):
            local.rank = rank
            loader = object.__new__(module.RForkModelLoader)
            loader.load_config = SimpleNamespace(device=None)
            loader._requires_processed_layout_transfer = lambda mc: True
            loader._get_target_registered_blocks = lambda *args: [(1, 1)] if reuse else []

            def session(*args):
                fail("session")
                return Session(rank)

            loader._ensure_rfork_session = session
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

    return SimpleNamespace(run=run, errors=errors, events=events, reuse=reuse)


@pytest.mark.parametrize("stage", ["prepare", "host_load", "seed_miss", "register", "transfer", "finalize"])
def test_one_rank_failure_reconstructs_all_ranks_after_cleanup(cluster, stage):
    cluster.errors[2, stage] = RuntimeError("injected failure")
    results = cluster.run()
    assert all(isinstance(result, torch.nn.Module) for result in results), results
    assert sum(event == "fallback" for _, event in cluster.events) == 4
    assert sum(event == "publish" for _, event in cluster.events) == (3 if stage == "prepare" else 4)
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


def test_mixed_draft_reuse_aborts_without_mutating_shared_target(cluster):
    cluster.reuse.add(2)
    results = cluster.run()
    assert all(isinstance(result, RuntimeError) for result in results), results
    assert not any(event in ("transfer", "fallback", "publish", "return") for _, event in cluster.events)


def test_unanimous_draft_reuse_returns_without_transfer(cluster):
    cluster.reuse.update(range(4))
    results = cluster.run()
    assert all(isinstance(result, torch.nn.Module) for result in results), results
    assert not any(event in ("transfer", "fallback", "publish") for _, event in cluster.events)


@pytest.mark.skipif(sys.platform != "linux", reason="Bounded Gloo test uses Linux fork")
@pytest.mark.parametrize("mode", ["transfer", "seed_miss", "cleanup_failure"])
def test_real_gloo_ranks_agree_before_reconstruction(request, monkeypatch, tmp_path, mode):
    _load_module(monkeypatch, "vllm_ascend.model_loader.rfork.local_weights", "local_weights.py")
    module = request.getfixturevalue("rfork_helpers").loader
    dist = torch.distributed
    if not dist.is_gloo_available():
        pytest.skip("Gloo is required")
    ctx = multiprocessing.get_context("fork")

    def worker(rank, connection):
        try:
            dist.init_process_group(
                "gloo",
                init_method=f"file://{tmp_path}/rendezvous",
                rank=rank,
                world_size=2,
                timeout=timedelta(seconds=5),
            )
            monkeypatch.setattr(
                sys.modules["vllm.distributed.parallel_state"],
                "get_world_group",
                lambda: SimpleNamespace(world_size=2, cpu_group=dist.group.WORLD),
                raising=False,
            )
            coordinator = module._RForkLoadCoordinator(SimpleNamespace(engram_config=object()))
            try:
                with coordinator.stage("weight transfer"):
                    if mode != "transfer" and rank == 1:
                        raise module._RForkSeedUnavailable("test seed miss")
            except module.RFORK_FALLBACK_EXCEPTIONS:
                with coordinator.stage("fallback cleanup", fallback=False):
                    if mode == "cleanup_failure" and rank == 0:
                        raise RuntimeError("test cleanup failure")
                # Reproduce a constructor collective after coordinated cleanup.
                tensor = torch.tensor([rank], dtype=torch.int32)
                dist.broadcast(tensor, src=0)
                assert tensor.item() == 0
                connection.send("local")
            else:
                connection.send("transfer")
        except module._RForkLoadAborted:
            connection.send("aborted")
        except Exception as exc:
            connection.send(repr(exc))
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
            connection.close()

    children = []
    pipes = []
    try:
        for rank in range(2):
            parent, child = ctx.Pipe(duplex=False)
            process = ctx.Process(target=worker, args=(rank, child))
            process.start()
            child.close()
            children.append(process)
            pipes.append(parent)
        for process in children:
            process.join(timeout=15)
            assert process.exitcode == 0, "Gloo worker failed or hung"
        results = [pipe.recv() for pipe in pipes]
        expected = {"transfer": "transfer", "seed_miss": "local", "cleanup_failure": "aborted"}[mode]
        assert results == [expected, expected]
    finally:
        for process in children:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for pipe in pipes:
            pipe.close()


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
    assert coordinator.reuse_shared_weights(True)
    assert not coordinator.reuse_shared_weights(False)
