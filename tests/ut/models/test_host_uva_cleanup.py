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

    def construct(buffer_name="shared", checkpoint_layer_id=None):
        try:
            owner = npu.SharedUvaBuffer(
                (8, 32),
                torch.bfloat16,
                "cpu",
                group,
                buffer_name,
                torch.nn.Module(),
                checkpoint_layer_id=checkpoint_layer_id,
            )
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


@pytest.mark.parametrize(("layer", "table", "rc"), [(1, "codes", 0), (14, "scales", 0), (14, "codes", 207001)])
def test_shared_engram_registration_events(shared_buffer_factory, monkeypatch, layer, table, rc):
    construct, lib, names, real_shm = shared_buffer_factory
    events = []
    clock = Mock(side_effect=[10.0, 11.25])
    monkeypatch.setattr(npu, "monotonic", clock)
    monkeypatch.setattr(npu.dist, "get_global_rank", lambda group, rank: 16)

    def log(fmt, *args):
        events.append(fmt % args)
        assert clock.call_count == (0 if len(events) == 1 else 2)

    def register(pointer, size, flags):
        assert len(events) == 1 and "registration start" in events[0]
        assert clock.call_count == 1
        return rc

    monkeypatch.setattr(npu.logger, "info", log)
    lib.aclrtHostRegisterV2.side_effect = register
    if rc:
        with pytest.raises(RuntimeError, match=f"aclrtHostRegisterV2 failed: rc={rc}"):
            construct(f"_{table}_uva", layer)
        lib.aclrtHostGetDevicePointer.assert_not_called()
        lib.aclrtHostUnregister.assert_not_called()
    else:
        construct(f"_{table}_uva", layer).close()
        lib.aclrtHostUnregister.assert_called_once()
    key = f"rank=16 layer={layer} table={table} bytes=512 shm={names[0]}"
    assert events == [
        f"Shared Engram registration start {key}",
        f"Shared Engram registration end {key} elapsed=1.250000s rc={rc}",
    ]
    with pytest.raises(FileNotFoundError):
        real_shm(name=names[0])


def test_completion_log_failure_releases_registered_range(shared_buffer_factory, monkeypatch):
    construct, lib, names, real_shm = shared_buffer_factory
    monkeypatch.setattr(npu.logger, "info", Mock(side_effect=[None, RuntimeError("log failed")]))
    with pytest.raises(RuntimeError, match="log failed"):
        construct("_codes_uva", 14)
    lib.aclrtHostGetDevicePointer.assert_not_called()
    lib.aclrtHostUnregister.assert_called_once()
    with pytest.raises(FileNotFoundError):
        real_shm(name=names[0])


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
