# SPDX-License-Identifier: Apache-2.0
"""A missing consumer must not cause the owner to release its backing."""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest
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
    for name in ("shared_host", "local_shared_vmm"):
        spec = importlib.util.spec_from_file_location(f"{package.__name__}.{name}", path / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
    return module


def test_owner_is_retained_if_consumer_does_not_reach_barrier(monkeypatch):
    module = load_module(monkeypatch)
    obj = module.LocalSharedVmmBuffer.__new__(module.LocalSharedVmmBuffer)
    obj.owner = True
    obj.group = types.SimpleNamespace(cpu_group=object(), world_size=2)
    obj.close = Mock()
    monkeypatch.setattr(module.dist, "monitored_barrier", Mock(side_effect=RuntimeError("peer unavailable")))
    with pytest.raises(RuntimeError, match="peer unavailable"):
        obj.close_collective()
    obj.close.assert_not_called()


def test_owner_is_retained_if_consumer_release_fails(monkeypatch):
    module = load_module(monkeypatch)
    obj = module.LocalSharedVmmBuffer.__new__(module.LocalSharedVmmBuffer)
    obj.owner = True
    obj.group = types.SimpleNamespace(cpu_group=object(), world_size=2)
    obj.close = Mock()
    monkeypatch.setattr(module.dist, "monitored_barrier", Mock())

    def gather(output, value, group):
        output[:] = [None, "consumer release error"]

    monkeypatch.setattr(module.dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="consumer release error"):
        obj.close_collective()
    obj.close.assert_not_called()


def test_synchronized_constructor_failure_has_distinct_type_and_cleans_owner(monkeypatch, tmp_path):
    module = load_module(monkeypatch)
    directory = tmp_path / "capability"
    directory.mkdir()
    monkeypatch.setattr(module.tempfile, "mkdtemp", lambda **_kwargs: str(directory))
    monkeypatch.setattr(module, "MappedHostBuffer", Mock(side_effect=RuntimeError("allocation failed")))
    monkeypatch.setattr(module, "retry_pending_rollbacks", Mock())
    monkeypatch.setattr(module.dist, "get_global_rank", lambda _group, rank: rank)
    monkeypatch.setattr(module.dist, "broadcast_object_list", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module.dist, "monitored_barrier", Mock())
    monkeypatch.setattr(module.dist, "all_gather_object", lambda out, value, group: out.__setitem__(0, value))
    group = types.SimpleNamespace(rank_in_group=0, cpu_group=object(), world_size=2)
    with pytest.raises(module.EngramSharedInitError, match="rank 0: ERROR: RuntimeError: allocation failed"):
        module.LocalSharedVmmBuffer((8, 256), None, "npu:0", group)
    assert not directory.exists()


def test_constructor_retains_owner_when_consumer_rollback_fails(monkeypatch, tmp_path):
    module = load_module(monkeypatch)
    directory = tmp_path / "capability"
    directory.mkdir()
    mapping = Mock()
    mapping.view.return_value = types.SimpleNamespace(tensor=object(), ptrs=object())
    monkeypatch.setattr(module.tempfile, "mkdtemp", lambda **_kwargs: str(directory))
    monkeypatch.setattr(module, "MappedHostBuffer", Mock(return_value=mapping))
    retry = Mock()
    monkeypatch.setattr(module, "retry_pending_rollbacks", retry)
    monkeypatch.setattr(module.dist, "get_global_rank", lambda _group, rank: rank)
    monkeypatch.setattr(module.dist, "broadcast_object_list", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module.dist, "monitored_barrier", Mock())
    calls = iter(([None, "import failed"], [None, "consumer cleanup failed"]))
    monkeypatch.setattr(
        module.dist, "all_gather_object", lambda out, value, group: out.__setitem__(slice(None), next(calls))
    )
    group = types.SimpleNamespace(rank_in_group=0, cpu_group=object(), world_size=2)
    with pytest.raises(module.EngramSharedCleanupError, match="consumer cleanup failed"):
        module.LocalSharedVmmBuffer((8, 256), None, "npu:0", group)
    mapping.close.assert_not_called()
    retry.assert_not_called()
    assert directory.exists()


def _rollback_worker(rank, rendezvous, directory, log_path, fail_consumer_close, retry_after=False):
    with pytest.MonkeyPatch.context() as monkeypatch:
        module = load_module(monkeypatch)
        dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
        monkeypatch.setattr(module.tempfile, "mkdtemp", lambda **_kwargs: directory)

        fail_close = [fail_consumer_close]

        class Mapping:
            def __init__(self, *_args, **_kwargs):
                pass

            def view(self):
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


def test_failed_constructor_cleanup_preserves_mapping_for_retry(monkeypatch, tmp_path):
    import gc
    import weakref

    module = load_module(monkeypatch)
    directory = tmp_path / "capability"
    directory.mkdir()
    mapping = Mock()
    mapping.view.return_value = types.SimpleNamespace(tensor=object(), ptrs=object(), close=Mock())
    monkeypatch.setattr(module.tempfile, "mkdtemp", lambda **_kwargs: str(directory))
    monkeypatch.setattr(module, "MappedHostBuffer", Mock(return_value=mapping))
    monkeypatch.setattr(module, "retry_pending_rollbacks", Mock())
    monkeypatch.setattr(module.dist, "get_global_rank", lambda _group, rank: rank)
    monkeypatch.setattr(module.dist, "broadcast_object_list", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module.dist, "monitored_barrier", Mock())
    calls = iter(([None, "import failed"], [None, "consumer cleanup failed"]))
    monkeypatch.setattr(
        module.dist, "all_gather_object", lambda out, value, group: out.__setitem__(slice(None), next(calls))
    )
    refs = []
    cls = module.LocalSharedVmmBuffer

    def allocate(cls, *_args):
        obj = object.__new__(cls)
        refs.append(weakref.ref(obj))
        return obj

    monkeypatch.setattr(cls, "__new__", staticmethod(allocate))
    group = types.SimpleNamespace(rank_in_group=0, cpu_group=object(), world_size=2)
    with pytest.raises(module.EngramSharedCleanupError):
        cls((8, 256), None, "npu:0", group)
    gc.collect()
    retained = refs[0]()
    assert retained is not None and retained.mapping is mapping
    assert module.pending_failed_initializations() == (retained,)
    monkeypatch.setattr(
        module.dist, "all_gather_object", lambda out, value, group: out.__setitem__(slice(None), [None, None])
    )
    retained.close_collective(retry_rollbacks=True)
    mapping.close.assert_called_once()
    assert not directory.exists()
    assert module.pending_failed_initializations() == ()


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
