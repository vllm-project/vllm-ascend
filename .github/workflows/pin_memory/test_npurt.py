#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""npurt host-register return-code check on Ascend NPU.

Run in the cann env (source Ascend/ascend-toolkit/set_env.sh first if the shell
does not source it from .bashrc):
    ~/miniconda3/envs/cann/bin/python test_npurt.py

Expected values (baseline from the 9/15 environment: CANN 8.5.0 + torch_npu
2.10.0.post5+git04c0562, driver 25.5.1, 910B4):
  npurt()                                  -> callable, raises nothing
  npuHostRegister(aligned, MAPPED=0x2)     -> rc=0
  npuHostRegister(same addr, MAPPED again) -> rc=507910 (already registered)
  npuHostRegister(flag=0)                  -> rc=107000 (invalid flag)
  npuHostRegister(unaligned addr, MAPPED)  -> rc=507899 (address not 4K-aligned)
  npuHostUnregister(addr)                  -> rc=0
  torch.is_pinned(registered buffer)       -> True on 9/15; observed False on
                                              the current box -- this flag is
                                              what routes copies to the fast
                                              DMA path, so it is the key check.

The aclrtPointerGetAttributes section classifies pointers via the CANN runtime
itself (the same API torch_npu's is_pinned uses) to show which layer breaks the
recognition chain when torch.is_pinned returns False.
"""

import ctypes

import torch
import torch_npu  # noqa: F401

MAPPED = 0x2
PINNED = 0x10000000
ALIGN = 4096

# aclrtMemLocationType (acl_rt.h, CANN 9.0.0): note HOST=0 is documented as
# "reserved enum, current version not support"; torch_npu checks type == HOST.
LOCATION_NAMES = {0: "HOST", 1: "DEVICE", 2: "UNREGISTERED", 4: "HOST_NUMA"}


class aclrtMemLocation(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint32), ("type", ctypes.c_int)]


class aclrtPtrAttributes(ctypes.Structure):
    _fields_ = [
        ("location", aclrtMemLocation),
        ("pageSize", ctypes.c_uint32),
        ("rsv", ctypes.c_uint32 * 4),
    ]


def aligned_buf(nbytes, value=0xAB):
    # Returns the ctypes array alongside the aligned address: the address must
    # be registered while the allocation is still alive. Letting the array be
    # garbage-collected first unmaps 64 MiB-class blocks (mmap arena) and the
    # stale address then fails registration with 107017 (invalid handle).
    raw = (ctypes.c_ubyte * (nbytes + ALIGN))()
    addr = ctypes.addressof(raw)
    offset = (-addr) % ALIGN
    aligned = addr + offset
    ctypes.memset(aligned, value, nbytes)
    return raw, aligned, nbytes


def section(title):
    print(f"\n=== {title} ===")


def check(name, actual, expected):
    mark = "OK" if actual == expected else "!! UNEXPECTED"
    print(f"[{mark}] {name}: got {actual}, expected {expected}")
    return actual == expected


def main():
    device = torch.device("npu:0")
    torch.npu.set_device(device)

    # --- 1. npurt availability -------------------------------------------
    section("npurt availability")
    has_npurt = hasattr(torch.npu, "npurt")
    print(f"hasattr(torch.npu, 'npurt'): {has_npurt}")
    if not has_npurt:
        print("npurt missing -- cannot continue")
        return
    npurt = torch.npu.npurt()
    print(f"npurt(): {npurt}")

    # --- 2. register/unregister return codes -----------------------------
    section("npuHostRegister / npuHostUnregister return codes")
    raw, aligned, nbytes = aligned_buf(64 * 1024 * 1024)
    print(f"aligned buffer: addr=0x{aligned:x}, size={nbytes}")

    rc = npurt.npuHostRegister(aligned, nbytes, MAPPED)
    check("register MAPPED", rc, 0)

    rc = npurt.npuHostRegister(aligned, nbytes, MAPPED)
    check("re-register same addr MAPPED", rc, 507910)

    rc = npurt.npuHostUnregister(aligned)
    check("unregister", rc, 0)

    rc = npurt.npuHostRegister(aligned, nbytes, 0)
    check("register flag=0", rc, 107000)

    unaligned = aligned + 1
    rc = npurt.npuHostRegister(unaligned, nbytes - 1, MAPPED)
    check("register unaligned addr", rc, 507899)

    rc = npurt.npuHostRegister(aligned, nbytes, PINNED)
    check("register PINNED", rc, 0)
    rc = npurt.npuHostUnregister(aligned)
    check("unregister PINNED", rc, 0)

    rc = npurt.npuHostRegister(aligned, nbytes, MAPPED | PINNED)
    check("register MAPPED|PINNED", rc, 0)
    rc = npurt.npuHostUnregister(aligned)
    check("unregister MAPPED|PINNED", rc, 0)

    # --- 3. torch.is_pinned recognition of a registered range ------------
    section("torch.is_pinned on a registered range")
    # op-independent: frombuffer view over the same buffer DeepSpeed's native
    # backend registers (posix_memalign + mlock + npurt MAPPED).
    rc = npurt.npuHostRegister(aligned, nbytes, MAPPED)
    check("register before is_pinned", rc, 0)
    view = torch.frombuffer((ctypes.c_ubyte * nbytes).from_address(aligned), dtype=torch.uint8)
    check("torch.is_pinned(registered view)", view.is_pinned(), True)
    rc = npurt.npuHostUnregister(aligned)
    check("unregister after is_pinned", rc, 0)
    check("torch.is_pinned after unregister", view.is_pinned(), False)

    # --- 4. runtime pointer classification (what is_pinned queries) ------
    section("aclrtPointerGetAttributes classification")
    lib = ctypes.CDLL("libascendcl.so")
    get_attrs = getattr(lib, "aclrtPointerGetAttributes", None)
    if get_attrs is None:
        print("aclrtPointerGetAttributes symbol NOT found in libascendcl.so")
        return
    get_attrs.restype = ctypes.c_int
    get_attrs.argtypes = [ctypes.c_void_p, ctypes.POINTER(aclrtPtrAttributes)]

    def classify(addr, label):
        attrs = aclrtPtrAttributes()
        rc = get_attrs(ctypes.c_void_p(addr), ctypes.byref(attrs))
        loc_name = LOCATION_NAMES.get(attrs.location.type, f"?({attrs.location.type})")
        print(f"  {label}: rc={rc} location.type={attrs.location.type} ({loc_name}), id={attrs.location.id}")
        return rc, attrs.location.type

    # plain malloc memory (never registered)
    plain_raw, plain_addr, _ = aligned_buf(1024 * 1024)
    classify(plain_addr, "plain malloc")

    # torch pinned allocator memory (aclrtMallocHost behind the scenes)
    torch_pinned = torch.empty(4096, dtype=torch.float32, pin_memory=True)
    print(f"  torch pin allocator buffer: torch.is_pinned={torch_pinned.is_pinned()}")
    classify(torch_pinned.data_ptr(), "torch pin allocator")

    # npurt-registered range
    rc = npurt.npuHostRegister(aligned, nbytes, MAPPED)
    check("register before classification", rc, 0)
    classify(aligned, "npurt-registered")
    rc = npurt.npuHostUnregister(aligned)
    check("unregister after classification", rc, 0)

    # --- 5. summary -------------------------------------------------------
    section("summary")
    print("Expected return codes: MAPPED=0, re-register=507910, flag0=107000, unaligned=507899, unregister=0")
    print(
        "Key observation: torch.is_pinned(registered view) -- True means the "
        "runtime recognizes the registration and copies take the DMA fast path."
    )


if __name__ == "__main__":
    main()
