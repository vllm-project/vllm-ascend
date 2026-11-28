# SPDX-License-Identifier: Apache-2.0
"""Real shared mappings with simulated CANN initialization/cleanup failures."""

import ctypes
import gc
import os
import weakref
from multiprocessing import shared_memory
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.models.deepseek_v41.engram import npu


@pytest.fixture
def shared_buffer_factory(monkeypatch):
    real_shm = shared_memory.SharedMemory
    names, owners = [], []
    lib = SimpleNamespace(
        aclrtHostRegisterV2=Mock(return_value=0),
        aclrtHostUnregister=Mock(return_value=0),
    )

    def device_pointer(pointer, output, flags):
        output._obj.value = pointer.value
        return 0

    lib.aclrtHostGetDevicePointer = Mock(side_effect=device_pointer)

    def make_shm(*args, **kwargs):
        shm = real_shm(*args, **kwargs)
        names.append(shm.name)
        return shm

    monkeypatch.setattr(npu, "_host_library", lambda: lib)
    monkeypatch.setattr(npu.DeviceOperator, "host_register_flags", lambda: 0)
    monkeypatch.setattr(npu.shared_memory, "SharedMemory", make_shm)
    monkeypatch.setattr(npu.dist, "get_global_rank", lambda group, rank: 0)
    monkeypatch.setattr(npu.dist, "broadcast_object_list", lambda *a, **kw: None)
    monkeypatch.setattr(npu.dist, "barrier", lambda **kw: None)
    monkeypatch.setattr(npu.dist, "all_gather_object", lambda errors, error, **kw: errors.__setitem__(0, error))
    group = SimpleNamespace(cpu_group=None, rank_in_group=0, world_size=2)

    def construct():
        try:
            owner = npu.SharedUvaBuffer((8, 32), torch.bfloat16, "cpu", group, "shared", torch.nn.Module())
        except npu.HostUvaBufferCleanupError as exc:
            owners.append(weakref.ref(exc.host_buffer))
            raise
        owners.append(weakref.ref(owner))
        return owner

    yield construct, lib, names, real_shm
    lib.aclrtHostUnregister.return_value = 0
    for reference in owners:
        if (owner := reference()) is not None:
            owner.close()
    for name in names:
        try:
            shm = real_shm(name=name)
        except FileNotFoundError:
            continue
        shm.unlink()
        shm.close()


@pytest.mark.parametrize("failure", ["register", "device_pointer", "peer"])
def test_failed_initialization_unlinks_shared_name(shared_buffer_factory, monkeypatch, failure):
    construct, lib, names, real_shm = shared_buffer_factory
    if failure == "register":
        lib.aclrtHostRegisterV2.return_value = 1
    elif failure == "device_pointer":
        lib.aclrtHostGetDevicePointer.side_effect = lambda *args: 1
    else:
        monkeypatch.setattr(
            npu.dist, "all_gather_object", lambda errors, error, **kw: errors.__setitem__(1, "peer failed")
        )
    with pytest.raises(RuntimeError, match="initialization failed"):
        construct()
    with pytest.raises(FileNotFoundError):
        real_shm(name=names[0])
    assert lib.aclrtHostUnregister.call_count == int(failure != "register")


def test_failed_unregister_retains_mapping_for_retry(shared_buffer_factory):
    construct, lib, names, real_shm = shared_buffer_factory
    lib.aclrtHostGetDevicePointer.side_effect = lambda *args: 1
    lib.aclrtHostUnregister.return_value = 1
    with pytest.raises(npu.HostUvaBufferCleanupError, match="cleanup failed") as caught:
        construct()
    owner = caught.value.host_buffer
    reference = weakref.ref(owner)
    del caught
    gc.collect()
    assert reference() is owner
    assert owner.pointer.value and owner.shm is not None
    with pytest.raises(FileNotFoundError):
        real_shm(name=names[0])
    lib.aclrtHostUnregister.return_value = 0
    owner.close()
    owner.close()
    assert owner.shm is None and not owner.pointer.value
    assert lib.aclrtHostUnregister.call_count == 2


def test_aligned_shared_size_rounds_up_to_two_mib():
    align = npu.SHARED_MAP_ALIGN
    assert npu._aligned_shared_size(0) == 0
    assert npu._aligned_shared_size(1) == align
    assert npu._aligned_shared_size(align) == align
    assert npu._aligned_shared_size(align + 1) == 2 * align


@pytest.mark.skipif(not os.path.isdir("/dev/shm"), reason="POSIX shared memory is required")
def test_map_shared_hugepage_is_two_mib_aligned():
    size = npu._aligned_shared_size((3 << 20) + 1)
    shm = shared_memory.SharedMemory(create=True, size=size)
    try:
        address = npu.map_shared_hugepage(shm.name, size)
        try:
            assert address % npu.SHARED_MAP_ALIGN == 0
            # The device gathers from this mapping, so it must alias the segment.
            ctypes.memset(ctypes.c_void_p(address), 0x5A, 4096)
            assert shm.buf[0] == 0x5A
        finally:
            npu.unmap_shared_hugepage(address, size)
    finally:
        shm.close()
        shm.unlink()


@pytest.mark.skipif(not os.path.isdir("/dev/shm"), reason="POSIX shared memory is required")
def test_open_failure_reserves_nothing(monkeypatch):
    calls = []
    real = npu._libc()

    def record_mmap(*args):
        calls.append(args[1])
        return real.mmap(*args)

    fake = SimpleNamespace(mmap=record_mmap, munmap=real.munmap, madvise=real.madvise)
    monkeypatch.setattr(npu, "_libc", lambda: fake)

    def fail_open(name):
        raise OSError("too many open files")

    monkeypatch.setattr(npu, "_open_shared", fail_open)
    with pytest.raises(OSError):
        npu.map_shared_hugepage("unused", npu.SHARED_MAP_ALIGN)
    assert calls == []


def test_madvise_failure_logs_warning(monkeypatch, caplog):
    real = npu._libc()
    fake = SimpleNamespace(mmap=real.mmap, munmap=real.munmap, madvise=lambda *args: -1)
    monkeypatch.setattr(npu, "_libc", lambda: fake)
    size = npu.SHARED_MAP_ALIGN
    shm = shared_memory.SharedMemory(create=True, size=size)
    try:
        with caplog.at_level("WARNING"):
            address = npu.map_shared_hugepage(shm.name, size)
        npu.unmap_shared_hugepage(address, size)
        assert "madvise(MADV_HUGEPAGE) failed" in caplog.text
    finally:
        shm.close()
        shm.unlink()


def _spy_unmap(monkeypatch):
    """Record released ranges while still performing the real unmap."""
    released = []
    real_unmap = npu.unmap_shared_hugepage

    def unmap(address, size):
        released.append((address, size))
        real_unmap(address, size)

    monkeypatch.setattr(npu, "unmap_shared_hugepage", unmap)
    return released


def test_failed_registration_releases_aligned_mapping(shared_buffer_factory, monkeypatch):
    construct, lib, names, real_shm = shared_buffer_factory
    released = _spy_unmap(monkeypatch)
    lib.aclrtHostRegisterV2.return_value = 1
    with pytest.raises(RuntimeError, match="initialization failed"):
        construct()
    assert len(released) == 1
    address, size = released[0]
    assert address % npu.SHARED_MAP_ALIGN == 0
    assert size == npu._aligned_shared_size(8 * 32 * 2)


def test_close_releases_aligned_mapping_once(shared_buffer_factory, monkeypatch):
    construct, lib, names, real_shm = shared_buffer_factory
    released = _spy_unmap(monkeypatch)
    owner = construct()
    assert owner.map_address is not None
    assert owner.map_address % npu.SHARED_MAP_ALIGN == 0
    assert owner.size_padded == npu._aligned_shared_size(8 * 32 * 2)
    owner.close()
    owner.close()
    assert len(released) == 1
    assert owner.map_address is None
