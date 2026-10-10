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
        registered = tuple(communicator._registered_regions)
        # Grow the allocator, then let it reclaim unused mapped pages while
        # the original expert registrations and connections remain alive.
        extra = torch.empty(128 * _MIB, dtype=torch.uint8, device=f"npu:{rank}")
        del extra
        torch.npu.empty_cache()
        assert tuple(communicator._registered_regions) == registered

        with ThreadPoolExecutor(max_workers=1) as executor:
            for _ in range(4):
                executor.submit(_migrate, communicator, buffers, rank).result(timeout=300)
                for tensor in buffers:
                    assert torch.all(tensor[0] == 2 - rank).item()
        timings.extend(communicator._eplb_hixl_phase_timings)
        communicator.close()
        communicator.close()
        assert not communicator._storage_refs
        del communicator, buffers, weights, allocation
        torch.npu.empty_cache()
    # Validate the increased region count against the real backend rather
    # than imposing the old, unverified 256-region limit.
    region_count = 320
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
    assert len(communicator._registered_regions) > 256
    communicator.close()
    del communicator, buffers, weights, allocation
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
    _migrate(communicator, buffers, rank)
    communicator._engine = _FailedDeregistration(communicator._engine, rank)
    state: Any = AscendEplbState.__new__(AscendEplbState)
    state._close_error = None
    state.async_worker = None
    state.model_states = {"model": SimpleNamespace(communicator=communicator)}
    try:
        state.close()
    except SystemExit:
        # The owner remains intact until the actual subprocess exits. This
        # failure must bypass the upstream RPC loop's `except Exception`.
        assert communicator._engine is not None
        assert communicator._storage_refs
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
    register = eplb_communicator.AscendHixlEplbCommunicator._register_tensor_segments

    def fail_after_registration(communicator, tensors):
        register(communicator, tensors)
        if rank == 0:
            communicator._engine = _FailedDeregistration(communicator._engine, rank)
            raise RuntimeError("injected initialization failure after real registration")

    eplb_communicator.AscendHixlEplbCommunicator._register_tensor_segments = fail_after_registration

    def rebuild():
        eplb_communicator.AscendHixlEplbCommunicator(dist.group.WORLD, [[weights]], buffers)

    def unexpected_response(_output):
        raise AssertionError("failed rollback must bypass ordinary RPC error handling")

    rpc = SimpleNamespace(worker=SimpleNamespace(rebuild=rebuild), rank=rank, handle_output=unexpected_response)
    try:
        WorkerProc._execute_worker_rpc(rpc, ("rebuild", (), {}, None))
    except SystemExit as error:
        owner = error.__cause__.hixl_communicator
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
