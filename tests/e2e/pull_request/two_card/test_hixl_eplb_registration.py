# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Real HIXL registration/READ regression; no model weights are required."""

import importlib.util
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401
from torch.multiprocessing.spawn import ProcessExitedException
from vllm.utils.network_utils import get_open_port
from vllm.v1.executor.multiproc_executor import WorkerProc

from vllm_ascend.distributed.eplb import eplb_communicator, hixl_compat
from vllm_ascend.distributed.eplb.eplb_state import AscendEplbState

_MIB = 1024 * 1024


def _migrate(communicator, buffers, rank: int) -> None:
    communicator.set_stream(None)
    communicator.set_transfer_context(np.arange(4), layer_idx=0)
    communicator.add_recv([tensor[0] for tensor in buffers], src_rank=1 - rank, expert_id=3 - 2 * rank)
    communicator.execute()


def _registration_worker(rank: int, port: int, binding: str) -> None:
    torch.npu.set_device(rank)
    if binding == "ctypes":
        eplb_communicator._resolve_hixl_module = lambda: hixl_compat
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=300),
    )
    timings = []
    for _ in range(3):
        # Same start, different lengths: the failing log registered 84 MiB
        # and then 86 MiB at the same address. Both views share live storage.
        allocation = torch.full((86 * _MIB // 4,), rank + 1, dtype=torch.float32, device=f"npu:{rank}")
        weights = [allocation[: 84 * _MIB // 4].view(2, -1), allocation.view(2, -1)]
        buffers = [torch.empty_like(tensor) for tensor in weights]
        communicator = eplb_communicator.AscendHixlEplbCommunicator(dist.group.WORLD, [weights], buffers)
        session = eplb_communicator._HixlSession(dist.group.WORLD)
        session.initialize({"model": communicator})
        registered = tuple(session._registered_handles)
        # Grow the allocator, then let it reclaim unused mapped pages while
        # the original expert registrations and connections remain alive.
        extra = torch.empty(128 * _MIB, dtype=torch.uint8, device=f"npu:{rank}")
        del extra
        torch.npu.empty_cache()
        assert tuple(session._registered_handles) == registered

        with ThreadPoolExecutor(max_workers=1) as executor:
            for _ in range(4):
                executor.submit(_migrate, communicator, buffers, rank).result(timeout=300)
                for tensor in buffers:
                    assert torch.all(tensor[0] == 2 - rank).item()
        timings.extend(communicator._eplb_hixl_phase_timings)
        session.close()
        session.close()
        assert not session._storage_refs
        del session, communicator, buffers, weights, allocation
        torch.npu.empty_cache()
    # Exercise the 4000-region boundary on the real backend, including one
    # additional region for the receive buffer.
    region_count = 3999
    # A small allocator prefix can make a 2 MiB view cover two aligned
    # pages; leave another page between views so merging keeps them apart.
    region_spacing = 6 * _MIB
    allocation = torch.empty(region_count * region_spacing // 4, dtype=torch.float32, device=f"npu:{rank}")
    weights = [
        [allocation[offset : offset + 2 * _MIB // 4].view(2, -1)]
        for offset in range(0, allocation.numel(), region_spacing // 4)
    ]
    buffers = [torch.empty_like(weights[0][0])]
    communicator = eplb_communicator.AscendHixlEplbCommunicator(dist.group.WORLD, weights, buffers)
    session = eplb_communicator._HixlSession(dist.group.WORLD)
    session.initialize({"model": communicator})
    assert len(session._registered_handles) == 4000
    session.close()
    del session, communicator, buffers, weights, allocation
    torch.npu.empty_cache()
    if rank == 0:
        print(
            {
                "binding": binding,
                "requests_per_migration": 1,
                "mean_transfer_ms": statistics.mean(t.transfer_ms for t in timings),
            }
        )
    dist.destroy_process_group()


@pytest.mark.parametrize("binding", ["ctypes", "official"])
def test_expandable_registration_growth_reclaim_and_rebuild(monkeypatch, binding):
    if torch.npu.device_count() < 2:
        pytest.skip("requires two NPU devices")
    if binding == "official" and importlib.util.find_spec("hixl") is None:
        pytest.skip("CANN does not provide the official Python binding")
    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    mp.spawn(_registration_worker, args=(get_open_port(), binding), nprocs=2, join=True)


class _FailedDeregistration:
    def __init__(self, engine, rank):
        self.engine = engine
        self.rank = rank

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def deregister_mem(self, handle):
        return hixl_compat.PARAM_INVALID if self.rank == 0 else self.engine.deregister_mem(handle)


def _failed_close_worker(rank: int, port: int) -> None:
    torch.npu.set_device(rank)
    eplb_communicator._resolve_hixl_module = lambda: hixl_compat
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=300),
    )
    weights = torch.full((2, 1024), rank + 1, dtype=torch.float32, device=f"npu:{rank}")
    buffers = [torch.empty_like(weights)]
    communicator = eplb_communicator.AscendHixlEplbCommunicator(dist.group.WORLD, [[weights]], buffers)
    session = eplb_communicator._HixlSession(dist.group.WORLD)
    session.initialize({"model": communicator})
    _migrate(communicator, buffers, rank)
    session._engine = _FailedDeregistration(session._engine, rank)
    state: Any = AscendEplbState.__new__(AscendEplbState)
    state._close_error = None
    state._hixl_session = session
    state.async_worker = None
    state.model_states = {"model": SimpleNamespace(communicator=communicator)}
    try:
        state.close()
    except SystemExit:
        # The owner remains intact until the actual subprocess exits. This
        # failure must bypass the upstream RPC loop's `except Exception`.
        assert session._engine is not None
        assert session._storage_refs
        assert state._close_error is not None
        raise
    raise AssertionError("failed deregistration must terminate the worker")


def test_failed_deregistration_terminates_workers(monkeypatch):
    if torch.npu.device_count() < 2:
        pytest.skip("requires two NPU devices")
    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    with pytest.raises(ProcessExitedException) as error:
        mp.spawn(_failed_close_worker, args=(get_open_port(),), nprocs=2, join=True)
    assert error.value.exit_code == 1


def _failed_initialization_worker(rank: int, port: int) -> None:
    torch.npu.set_device(rank)
    eplb_communicator._resolve_hixl_module = lambda: hixl_compat
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    weights = torch.full((2, 1024), rank + 1, dtype=torch.float32, device=f"npu:{rank}")
    buffers = [torch.empty_like(weights)]
    register = eplb_communicator._HixlSession._register_tensor_segments

    def fail_after_registration(communicator, tensors):
        register(communicator, tensors)
        if rank == 0:
            communicator._engine = _FailedDeregistration(communicator._engine, rank)
            raise RuntimeError("injected initialization failure after real registration")

    eplb_communicator._HixlSession._register_tensor_segments = fail_after_registration

    def rebuild():
        communicator = eplb_communicator.AscendHixlEplbCommunicator(dist.group.WORLD, [[weights]], buffers)
        session = eplb_communicator._HixlSession(dist.group.WORLD)
        session.initialize({"model": communicator})

    def unexpected_response(_output):
        raise AssertionError("failed rollback must bypass ordinary RPC error handling")

    rpc = SimpleNamespace(worker=SimpleNamespace(rebuild=rebuild), rank=rank, handle_output=unexpected_response)
    try:
        WorkerProc._execute_worker_rpc(rpc, ("rebuild", (), {}, None))
    except SystemExit as error:
        cause = error.__cause__
        assert cause is not None and hasattr(cause, "hixl_session")
        owner = cause.hixl_session
        assert owner._engine is not None
        assert owner._storage_refs
        if rank == 0:
            assert owner._registered_handles
        raise
    raise AssertionError("failed initialization rollback must terminate the worker")


def test_failed_initialization_rollback_terminates_workers_through_rpc(monkeypatch):
    if torch.npu.device_count() < 2:
        pytest.skip("requires two NPU devices")
    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    with pytest.raises(ProcessExitedException) as error:
        mp.spawn(_failed_initialization_worker, args=(get_open_port(),), nprocs=2, join=True)
    assert error.value.exit_code == 1


class _TrackingEngine:
    def __init__(self, engine):
        self.engine = engine
        self.regions = []
        self.connections = 0
        self.finalizations = 0

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def register_mem(self, descriptor, mem_type):
        address, size = descriptor.addr, descriptor.len
        assert all(address + size <= start or start + length <= address for start, length in self.regions)
        status, handle = self.engine.register_mem(descriptor, mem_type)
        if status == 0:
            self.regions.append((address, size))
        return status, handle

    def connect(self, *args):
        self.connections += 1
        return self.engine.connect(*args)

    def finalize(self):
        self.finalizations += 1
        return self.engine.finalize()


class _TrackingBinding:
    def __init__(self, binding):
        self.binding = binding
        self.engines = []

    def __getattr__(self, name):
        return getattr(self.binding, name)

    def Hixl(self):
        engine = _TrackingEngine(self.binding.Hixl())
        self.engines.append(engine)
        return engine


def _shared_registration_worker(rank: int, port: int, binding: str) -> None:
    torch.npu.set_device(rank)
    tracked = _TrackingBinding(hixl_compat if binding == "ctypes" else eplb_communicator._resolve_hixl_module())
    eplb_communicator._resolve_hixl_module = lambda: tracked
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2, timeout=timedelta(seconds=300)
    )
    # Use distinct ordinary allocator storages and select two whose outward
    # registration ranges overlap. Neither tensor must start on a 2 MiB boundary.
    candidates = [torch.full((2, 16 * 1024), rank + 1, device=f"npu:{rank}") for _ in range(32)]
    alignment = 2 * _MIB
    pairs = [
        (left, right)
        for index, left in enumerate(candidates)
        for right in candidates[index + 1 :]
        if left.data_ptr() // alignment == right.data_ptr() // alignment
        and left.data_ptr() % alignment
        and right.data_ptr() % alignment
    ]
    assert pairs, "allocator did not produce the overlapping aligned ranges required by this regression"
    main_weight, draft_weight = pairs[0]
    assert main_weight.untyped_storage().data_ptr() != draft_weight.untyped_storage().data_ptr()
    draft_weight.fill_(rank + 11)
    buffers = {
        model: [torch.full_like(weight, -7)] for model, weight in (("draft", draft_weight), ("main", main_weight))
    }
    communicators = {
        model: eplb_communicator.AscendHixlEplbCommunicator(dist.group.WORLD, [[weight]], buffers[model])
        for model, weight in (("draft", draft_weight), ("main", main_weight))
    }
    session = eplb_communicator._HixlSession(dist.group.WORLD)
    state: Any = AscendEplbState.__new__(AscendEplbState)
    state._hixl_session = session
    state._close_error = None
    state.async_worker = None
    state.model_states = {key: SimpleNamespace(communicator=value) for key, value in communicators.items()}
    session.initialize(communicators)
    assert len(tracked.engines) == 1
    engine = tracked.engines[0]
    assert engine.connections == 1
    shared_page = main_weight.data_ptr() // alignment * alignment
    assert sum(start <= shared_page < start + length for start, length in engine.regions) == 1
    handles = tuple(session._registered_handles)
    with ThreadPoolExecutor(max_workers=1) as executor:
        for _ in range(8):
            extra = torch.empty(128 * _MIB, dtype=torch.uint8, device=f"npu:{rank}")
            del extra
            torch.npu.empty_cache()
            for model in ("draft", "main"):
                executor.submit(_migrate, communicators[model], buffers[model], rank).result(timeout=300)
                expected = (12 if model == "draft" else 2) - rank
                assert torch.all(buffers[model][0][0] == expected).item()
                assert torch.all(buffers[model][0][1] == -7).item()
            assert tuple(session._registered_handles) == handles
            assert torch.all(main_weight == rank + 1).item()
            assert torch.all(draft_weight == rank + 11).item()
    for tensor in candidates:
        if tensor is not draft_weight:
            assert torch.all(tensor == rank + 1).item()
    timings = [timing for c in communicators.values() for timing in c._eplb_hixl_phase_timings]
    state.close()
    state.close()
    assert engine.finalizations == 1
    assert not session._storage_refs
    if rank == 0:
        print(
            {
                "binding": binding,
                "models": 2,
                "registered_regions": len(engine.regions),
                "registered_bytes": sum(size for _, size in engine.regions),
                "mean_transfer_ms": statistics.mean(t.transfer_ms for t in timings),
            }
        )
    dist.destroy_process_group()


@pytest.mark.parametrize("binding", ["ctypes", "official"])
def test_main_and_draft_share_aligned_registration_after_empty_cache(monkeypatch, binding):
    if torch.npu.device_count() < 2:
        pytest.skip("requires two NPU devices")
    if binding == "official" and importlib.util.find_spec("hixl") is None:
        pytest.skip("CANN does not provide the official Python binding")
    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    mp.spawn(_shared_registration_worker, args=(get_open_port(), binding), nprocs=2, join=True)
