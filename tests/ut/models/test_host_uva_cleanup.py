# SPDX-License-Identifier: Apache-2.0
"""Real shared mappings with simulated CANN initialization/cleanup failures."""

import gc
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
            owner = npu.SharedUvaBuffer((8, 32), torch.bfloat16, "cpu", group)
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
