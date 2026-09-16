"""Real CANN lifecycle tests; no model restart or invalid device kernels."""

import ctypes
import errno
import gc
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from engram_vmm import mapping

torch.npu.set_device(0)
owner = SimpleNamespace(rank=0)
peer = SimpleNamespace(rank=1)
root = Path(tempfile.mkdtemp(prefix="engram-lifecycle-", dir="/dev/shm"))
lib = mapping.library()
assert lib.host_vmm_live_mappings() == 0

# Truncated descriptors must return an error, never accidental success (-0).
(root / "truncated").write_bytes(b"bad")
(root / "truncated.ready").touch()
result = mapping.Allocation()
rc = lib.host_shared_registered_alloc(mapping.GIB, 0, os.fsencode(root / "truncated"), 0, ctypes.byref(result))
assert rc == -errno.EBADMSG, (rc, lib.host_shared_last_error())
assert not result.device_ptr and lib.host_vmm_live_mappings() == 0
(root / "truncated.ready").unlink()
(root / "truncated").unlink()
print("PASS truncated descriptor", flush=True)

for cycle in range(3):
    m = mapping.VmmMapping(root / "reuse", 17, 256, torch.int8, owner)
    ref = torch.arange(17 * 256, dtype=torch.int32).reshape(17, 256).to(torch.int8)
    m.publish(0, ref)
    torch.npu.synchronize()
    descriptor = (root / "reuse").read_bytes()
    try:
        mapping.VmmMapping(root / "reuse", 17, 256, torch.int8, owner)
        raise AssertionError("duplicate owner accepted")
    except RuntimeError as exc:
        assert "exists" in str(exc) or "claim" in str(exc)
    assert (root / "reuse").read_bytes() == descriptor
    alias = m.tensor[1:3]
    try:
        m.close()
        raise AssertionError("live tensor alias was unmapped")
    except RuntimeError as exc:
        assert "aliases still alive" in str(exc)
    assert torch.equal(alias.clone().cpu(), ref[1:3])
    assert lib.host_vmm_live_mappings() == 1
    del alias
    gc.collect()
    m.close()
    m.close()
    assert lib.host_vmm_live_mappings() == 0
    assert not (root / "reuse").exists() and not (root / "reuse.ready").exists()
print("PASS aliases, idempotent close, descriptor reuse", flush=True)

with patch.object(mapping, "tensor_at", side_effect=RuntimeError("injected wrapping failure")):
    try:
        mapping.VmmMapping(root / "wrap", 17, 256, torch.int8, owner)
        raise AssertionError("wrapping failure not propagated")
    except RuntimeError as exc:
        assert "injected wrapping" in str(exc)
assert lib.host_vmm_live_mappings() == 0 and not (root / "wrap").exists()
print("PASS wrapping rollback", flush=True)

# An already-imported consumer must survive owner close, with no descriptor.
m = mapping.VmmMapping(root / "peer", 17, 256, torch.int8, owner)
n = mapping.VmmMapping(root / "peer", 17, 256, torch.int8, peer)
m.publish(0, ref)
torch.npu.synchronize()
m.close()
assert torch.equal(n.tensor.clone().cpu(), ref)
n.close()
assert lib.host_vmm_live_mappings() == 0
print("PASS imported mapping outlives owner", flush=True)

if os.environ.get("ENGRAM_TEST_RELEASE_FAILURES") == "1":
    shim = ctypes.CDLL(None)
    shim.engram_fail_release.argtypes = [ctypes.c_void_p, ctypes.c_int]
    for stage in (1, 2, 3):
        m = mapping.VmmMapping(root / f"fail{stage}", 17, 256, torch.int8, owner)
        shim.engram_fail_release(m.allocation.device_ptr, stage)
        try:
            m.close()
            raise AssertionError("injected release failure ignored")
        except RuntimeError as exc:
            assert "VMM free:" in str(exc)
        assert lib.host_vmm_live_mappings() == 1
        other = mapping.VmmMapping(root / f"other{stage}", 17, 256, torch.int8, owner)
        assert other.allocation.token != m.allocation.token
        other.publish(0, ref)
        m.close()
        assert torch.equal(other.tensor.clone().cpu(), ref)
        other.close()
        assert lib.host_vmm_live_mappings() == 0
    print("PASS release errors propagated and retried", flush=True)
    shim.engram_fail_allocation_rollback()
    try:
        mapping.VmmMapping(root / "rollback", 17, 256, torch.int8, owner)
        raise AssertionError("injected allocation failure ignored")
    except RuntimeError as exc:
        assert "rollback failed" in str(exc), str(exc)
    assert lib.host_vmm_live_mappings() == 1
    assert lib.host_vmm_retry_rollbacks() == 0
    assert lib.host_vmm_live_mappings() == 0
    print("PASS retained allocation rollback retried", flush=True)

    def failed_wrap(ptr, *args):
        shim.engram_fail_release(ptr, 1)
        raise RuntimeError("injected wrapping plus release failure")

    with patch.object(mapping, "tensor_at", side_effect=failed_wrap):
        try:
            mapping.VmmMapping(root / "wrap_release", 17, 256, torch.int8, owner)
            raise AssertionError("combined failure ignored")
        except RuntimeError as exc:
            assert "deferred rollback" in str(exc)
    assert lib.host_vmm_live_mappings() == 1
    assert lib.host_vmm_retry_rollbacks() == 0
    assert lib.host_vmm_live_mappings() == 0
    print("PASS wrapping plus release failure retained/retried", flush=True)

assert not list(root.iterdir()), list(root.iterdir())
root.rmdir()
print("LIFECYCLE_PASS", flush=True)
