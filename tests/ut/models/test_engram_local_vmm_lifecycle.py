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
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/engram/local_shared_vmm.py"
    spec = importlib.util.spec_from_file_location("local_shared_vmm_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_library", Mock(return_value=object()))
    return module


def test_failed_probe_cleanup_does_not_fallback(monkeypatch):
    module = load_module(monkeypatch)
    monkeypatch.setattr(
        module,
        "torch",
        types.SimpleNamespace(
            int8=object(), device=lambda *_: object(), npu=types.SimpleNamespace(current_device=lambda: 0)
        ),
    )
    monkeypatch.setattr(
        module, "LocalSharedVmmBuffer", Mock(side_effect=module.EngramSharedCleanupError("cleanup failed"))
    )
    with pytest.raises(module.EngramSharedCleanupError):
        module.local_vmm_available(types.SimpleNamespace(cpu_group=object(), world_size=2))


def test_combined_codes_and_scales_share_one_allocation(monkeypatch, tmp_path):
    module = load_module(monkeypatch)
    backing = (ctypes.c_ubyte * 24)()
    address = ctypes.addressof(backing)
    mapping = Mock()
    mapping.allocation = types.SimpleNamespace(host_ptr=address, device_ptr=address)
    mapping.row_bytes = 8
    allocate = Mock(return_value=mapping)
    host_buffer_cls = module.MappedHostBuffer
    monkeypatch.setattr(module, "MappedHostBuffer", allocate)
    monkeypatch.setattr(module.tempfile, "mkdtemp", lambda **_kwargs: str(tmp_path))
    monkeypatch.setattr(module.dist, "get_global_rank", lambda _group, rank: rank)
    monkeypatch.setattr(module.dist, "broadcast_object_list", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module.dist, "all_gather_object", lambda out, value, group: out.__setitem__(0, value))
    group = types.SimpleNamespace(rank_in_group=0, cpu_group=object(), world_size=1)
    buf = module.LocalSharedVmmBuffer((2, 8), torch.int8, "cpu", group, scales_shape=(2, 1), scale_dtype=torch.float32)
    assert allocate.call_count == 1
    module._library.assert_called_once()
    assert allocate.call_args.kwargs["lib"] is module._library.return_value
    assert allocate.call_args.args[1:3] == (3, 8)
    buf.scales.publish(0, torch.tensor([[1.5], [2.5]]))
    assert bytes(backing[:16]) == bytes(16)
    assert list((ctypes.c_float * 2).from_buffer(backing, 16)) == [1.5, 2.5]
    assert buf.scales.ptrs.tolist() == [address + 16]
    codes = host_buffer_cls.__new__(host_buffer_cls)
    codes.rows, codes.width, codes.dtype = 2, 8, torch.int8
    codes.row_bytes, codes.owner, codes.closed = 8, True, False
    codes.allocation = types.SimpleNamespace(host_ptr=address)
    codes.publish(0, torch.ones((2, 8), dtype=torch.int8))
    assert bytes(backing[:16]) == bytes([1] * 16)
    assert list((ctypes.c_float * 2).from_buffer(backing, 16)) == [1.5, 2.5]
    codes.owner = False
    with pytest.raises(RuntimeError, match="owner"):
        codes.publish(0, torch.zeros((1, 8), dtype=torch.int8))


def _group_worker(rank, rendezvous, directory, log_path, mode):
    with pytest.MonkeyPatch.context() as monkeypatch:
        module = load_module(monkeypatch)
        dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
        monkeypatch.setattr(module.tempfile, "mkdtemp", lambda **_kwargs: directory)
        missing = mode.startswith("missing") and rank == int(mode[-1])
        loader = Mock(side_effect=OSError("missing library") if missing else None, return_value=object())
        monkeypatch.setattr(module, "_library", loader)
        retry = Mock()
        monkeypatch.setattr(module, "retry_pending_rollbacks", retry)
        fail_close = [mode == "retry"]

        class Mapping:
            def __init__(self, *_args, **_kwargs):
                pass

            def close(self):
                with open(log_path, "a") as log:
                    log.write(f"close {rank}\n")
                if rank == 1 and fail_close[0]:
                    raise RuntimeError("injected cleanup failure")

        monkeypatch.setattr(module, "MappedHostBuffer", Mapping)

        def view(*_args):
            if rank == 1 and not mode.startswith("missing"):
                raise RuntimeError("injected view failure")
            return object(), object()

        monkeypatch.setattr(module, "_codes_view", view)
        group = types.SimpleNamespace(rank_in_group=rank, cpu_group=dist.group.WORLD, world_size=2)
        try:
            if mode.startswith("missing"):
                monkeypatch.setattr(
                    module,
                    "torch",
                    types.SimpleNamespace(
                        int8=torch.int8,
                        float32=torch.float32,
                        empty=torch.empty,
                        tensor=torch.tensor,
                        device=lambda *_args: "npu:0",
                        npu=types.SimpleNamespace(current_device=lambda: 0),
                    ),
                )
                assert module.local_vmm_available(group) is False
                assert module.local_vmm_available(group) is False
                loader.assert_called_once()
                assert retry.called is not missing
            else:
                error = module.EngramSharedCleanupError if mode == "retry" else module.EngramSharedInitError
                with pytest.raises(error, match="injected view failure"):
                    module.LocalSharedVmmBuffer((8, 256), None, "npu:0", group)
                if mode == "retry":
                    retained = tuple(module._failed_initializations)
                    assert len(retained) == 1
                    fail_close[0] = False
                    retained[0].close_collective(retry_rollbacks=True)
                    assert tuple(module._failed_initializations) == ()
        finally:
            dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["missing0", "missing1", "cleanup", "retry"])
def test_group_fallback_and_consumer_first_retry(tmp_path, mode):
    directory = tmp_path / "capability"
    directory.mkdir()
    log = tmp_path / "cleanup.log"
    mp.spawn(_group_worker, args=(str(tmp_path / "rendezvous"), str(directory), str(log), mode), nprocs=2, join=True)
    if mode in ("cleanup", "retry"):
        assert log.read_text().splitlines() == (
            ["close 1", "close 1", "close 0"] if mode == "retry" else ["close 1", "close 0"]
        )
        assert not directory.exists()
