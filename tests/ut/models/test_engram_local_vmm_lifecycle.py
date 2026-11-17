# SPDX-License-Identifier: Apache-2.0
"""A missing consumer must not cause the owner to release its backing."""

import ctypes
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def load_module(monkeypatch):
    if importlib.util.find_spec("vllm") is None:
        vllm = types.ModuleType("vllm")
        vllm.__path__ = []
        logger = types.ModuleType("vllm.logger")
        vars(logger)["logger"] = Mock()
        monkeypatch.setitem(sys.modules, "vllm", vllm)
        monkeypatch.setitem(sys.modules, "vllm.logger", logger)
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/engram"
    package = types.ModuleType("local_vmm_test_package")
    package.__path__ = [str(path)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    for name in ("local_vmm_host", "local_shared_vmm"):
        spec = importlib.util.spec_from_file_location(f"{package.__name__}.{name}", path / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
    return module


def test_backend_probe_is_cached_without_extra_collective(monkeypatch):
    module = load_module(monkeypatch)
    monkeypatch.setattr(module, "_library", Mock())
    monkeypatch.setattr(
        module,
        "torch",
        types.SimpleNamespace(
            int8=object(), device=lambda *args: object(), npu=types.SimpleNamespace(current_device=lambda: 0)
        ),
    )
    constructor = Mock()
    monkeypatch.setattr(module, "LocalSharedVmmBuffer", constructor)
    gather = Mock(side_effect=AssertionError("unexpected extra collective"))
    monkeypatch.setattr(module.dist, "all_gather_object", gather)
    group = types.SimpleNamespace(cpu_group=object(), world_size=2)
    assert module.local_vmm_available(group) is True
    assert module.local_vmm_available(group) is True
    constructor.assert_called_once()
    constructor.return_value.close_collective.assert_called_once()
    gather.assert_not_called()


def test_runtime_probe_fallback_requires_successful_collective_cleanup(monkeypatch):
    module = load_module(monkeypatch)
    monkeypatch.setattr(module, "_library", Mock())
    monkeypatch.setattr(
        module,
        "torch",
        types.SimpleNamespace(
            int8=object(), device=lambda *args: object(), npu=types.SimpleNamespace(current_device=lambda: 0)
        ),
    )
    monkeypatch.setattr(
        module.dist, "all_gather_object", lambda out, value, group: out.__setitem__(slice(None), [None, None])
    )
    group = types.SimpleNamespace(cpu_group=object(), world_size=2)
    monkeypatch.setattr(module, "LocalSharedVmmBuffer", Mock(side_effect=module.EngramSharedInitError("probe failed")))
    assert module.local_vmm_available(group) is False
    monkeypatch.setattr(
        module, "LocalSharedVmmBuffer", Mock(side_effect=module.EngramSharedCleanupError("cleanup failed"))
    )
    group = types.SimpleNamespace(cpu_group=object(), world_size=2)
    with pytest.raises(module.EngramSharedCleanupError):
        module.local_vmm_available(group)


def test_combined_codes_and_scales_share_one_allocation(monkeypatch, tmp_path):
    module = load_module(monkeypatch)
    backing = (ctypes.c_ubyte * 24)()
    address = ctypes.addressof(backing)
    mapping = Mock()
    mapping.allocation = types.SimpleNamespace(host_ptr=address, device_ptr=address)
    mapping.view.return_value = types.SimpleNamespace(tensor=object(), ptrs=object())
    allocate = Mock(return_value=mapping)
    monkeypatch.setattr(module, "MappedHostBuffer", allocate)
    monkeypatch.setattr(module.tempfile, "mkdtemp", lambda **_kwargs: str(tmp_path))
    monkeypatch.setattr(module.dist, "get_global_rank", lambda _group, rank: rank)
    monkeypatch.setattr(module.dist, "broadcast_object_list", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module.dist, "all_gather_object", lambda out, value, group: out.__setitem__(0, value))
    group = types.SimpleNamespace(rank_in_group=0, cpu_group=object(), world_size=1)
    buf = module.LocalSharedVmmBuffer((2, 8), torch.int8, "cpu", group, scales_shape=(2, 1), scale_dtype=torch.float32)
    assert allocate.call_count == 1
    assert allocate.call_args.args[1:3] == (3, 8)
    buf.scales.publish(0, torch.tensor([[1.5], [2.5]]))
    assert bytes(backing[:16]) == bytes(16)
    assert list((ctypes.c_float * 2).from_buffer(backing, 16)) == [1.5, 2.5]
    assert buf.scales.ptrs.tolist() == [address + 16]


def _missing_library_worker(rank, rendezvous, directory, missing_rank):
    with pytest.MonkeyPatch.context() as monkeypatch:
        module = load_module(monkeypatch)
        dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
        monkeypatch.setattr(module.tempfile, "mkdtemp", lambda **_kwargs: directory)
        monkeypatch.setattr(module, "retry_pending_rollbacks", lambda: None)

        class Mapping:
            def __init__(self, *_args, **_kwargs):
                pass

            def view(self, *_args):
                return types.SimpleNamespace(tensor=object(), ptrs=object(), close=lambda: None)

            def close(self):
                pass

        monkeypatch.setattr(module, "MappedHostBuffer", Mapping)
        group = types.SimpleNamespace(rank_in_group=rank, cpu_group=dist.group.WORLD, world_size=2)
        try:
            with pytest.raises(module.EngramSharedInitError, match="missing library"):
                module.LocalSharedVmmBuffer(
                    (8, 256),
                    None,
                    "npu:0",
                    group,
                    library_error="missing library" if rank == missing_rank else None,
                )
        finally:
            dist.destroy_process_group()


@pytest.mark.parametrize("missing_rank", [0, 1])
def test_missing_library_falls_back_on_both_ranks(tmp_path, missing_rank):
    directory = tmp_path / "capability"
    directory.mkdir()
    mp.spawn(
        _missing_library_worker,
        args=(str(tmp_path / "rendezvous"), str(directory), missing_rank),
        nprocs=2,
        join=True,
    )
    assert not directory.exists() if missing_rank == 1 else directory.exists()


def _rollback_worker(rank, rendezvous, directory, log_path, fail_consumer_close, retry_after=False):
    with pytest.MonkeyPatch.context() as monkeypatch:
        module = load_module(monkeypatch)
        dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
        monkeypatch.setattr(module.tempfile, "mkdtemp", lambda **_kwargs: directory)

        fail_close = [fail_consumer_close]

        class Mapping:
            def __init__(self, *_args, **_kwargs):
                pass

            def view(self, *_args):
                if rank == 1:
                    raise RuntimeError("injected view failure")
                return types.SimpleNamespace(tensor=object(), ptrs=object(), close=lambda: None)

            def close(self):
                with open(log_path, "a") as log:
                    log.write(f"close {rank}\n")
                if rank == 1 and fail_close[0]:
                    raise RuntimeError("injected cleanup failure")

        monkeypatch.setattr(module, "MappedHostBuffer", Mapping)
        monkeypatch.setattr(module, "retry_pending_rollbacks", lambda: None)
        group = types.SimpleNamespace(rank_in_group=rank, cpu_group=dist.group.WORLD, world_size=2)
        try:
            error_type = module.EngramSharedCleanupError if fail_consumer_close else module.EngramSharedInitError
            with pytest.raises(error_type, match="injected view failure"):
                module.LocalSharedVmmBuffer((8, 256), None, "npu:0", group)
            if retry_after:
                import gc

                gc.collect()
                retained = module.pending_failed_initializations()
                assert len(retained) == 1
                fail_close[0] = False
                retained[0].close_collective(retry_rollbacks=True)
                assert module.pending_failed_initializations() == ()
        finally:
            dist.destroy_process_group()


@pytest.mark.parametrize("fail_consumer_close", [False, True])
def test_constructor_rollback_orders_real_process_group(tmp_path, fail_consumer_close):
    directory = tmp_path / "capability"
    directory.mkdir()
    log = tmp_path / "cleanup.log"
    mp.spawn(
        _rollback_worker,
        args=(str(tmp_path / "rendezvous"), str(directory), str(log), fail_consumer_close),
        nprocs=2,
        join=True,
    )
    assert log.read_text().splitlines() == (["close 1"] if fail_consumer_close else ["close 1", "close 0"])
    assert directory.exists() == fail_consumer_close


def test_failed_constructor_retries_collectively_after_peer_recovers(tmp_path):
    directory = tmp_path / "capability"
    directory.mkdir()
    log = tmp_path / "cleanup.log"
    mp.spawn(
        _rollback_worker,
        args=(str(tmp_path / "rendezvous"), str(directory), str(log), True, True),
        nprocs=2,
        join=True,
    )
    assert log.read_text().splitlines() == ["close 1", "close 1", "close 0"]
    assert not directory.exists()


@pytest.mark.parametrize("failed_collective", ["broadcast", "gather"])
def test_unsynchronized_constructor_collective_failure_retains_local_mapping(monkeypatch, tmp_path, failed_collective):
    import gc

    module = load_module(monkeypatch)
    directory = tmp_path / "capability"
    directory.mkdir()
    mapping = Mock()
    mapping.view.return_value = types.SimpleNamespace(tensor=object(), ptrs=object(), close=Mock())
    monkeypatch.setattr(module.tempfile, "mkdtemp", lambda **_kwargs: str(directory))
    monkeypatch.setattr(module, "MappedHostBuffer", Mock(return_value=mapping))
    monkeypatch.setattr(module.dist, "get_global_rank", lambda _group, rank: rank)
    failure = RuntimeError(f"injected {failed_collective} failure")
    broadcast = Mock(side_effect=failure if failed_collective == "broadcast" else None)
    gather = Mock(side_effect=failure if failed_collective == "gather" else None)
    barrier = Mock()
    monkeypatch.setattr(module.dist, "broadcast_object_list", broadcast)
    monkeypatch.setattr(module.dist, "all_gather_object", gather)
    monkeypatch.setattr(module.dist, "monitored_barrier", barrier)
    group = types.SimpleNamespace(rank_in_group=0, cpu_group=object(), world_size=2)
    with pytest.raises(RuntimeError, match=f"injected {failed_collective} failure"):
        module.LocalSharedVmmBuffer((8, 256), None, "npu:0", group)
    gc.collect()
    pending = module.pending_failed_initializations()
    assert len(pending) == 1 and pending[0].mapping is mapping
    mapping.close.assert_not_called()
    barrier.assert_not_called()
    broadcast.assert_called_once()
    if failed_collective == "broadcast":
        gather.assert_not_called()
    else:
        gather.assert_called_once()
