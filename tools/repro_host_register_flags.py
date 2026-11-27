# SPDX-License-Identifier: Apache-2.0
"""Manual A2/A3 diagnostic for ``aclrtHostRegisterV2`` registration flags.

This script needs a real Ascend device and a CANN install. It does not start
vLLM: it allocates host memory with ``aclrtMallocHost``, registers it with
``MAPPED`` alone and with ``MAPPED|PINNED``, and asks CANN for the device
pointer. On runtimes that reject the redundant ``PINNED`` hint, the combined
flags return ``rc=107000`` (``ACL_ERROR_RT_PARAM_INVALID``) even for a small
allocation, while ``MAPPED`` alone succeeds for the full Engram table.

Make CANN's ``lib64`` visible before running, for example::

    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
    python tools/repro_host_register_flags.py --device-id 0

The default large case registers 12 GiB, matching a single Engram table. Use
``--skip-large`` on machines without enough host memory. ``--library`` selects
an explicit ``libascendcl.so`` path when the loader cannot find it.

This CANN version has no ``aclrtInit`` symbol, so initialization uses
``aclInit``.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util

ACL_HOST_REG_MAPPED = 0x2
ACL_HOST_REG_PINNED = 0x10000000
ACL_ERROR_RT_PARAM_INVALID = 107000
MIB = 1024 * 1024
GIB = 1024 * MIB


def _load_library(explicit_path: str | None):
    path = explicit_path or ctypes.util.find_library("ascendcl")
    if not path:
        raise SystemExit(
            "libascendcl.so not found; add the CANN lib64 directory to "
            "LD_LIBRARY_PATH or pass --library /path/to/libascendcl.so"
        )
    lib = ctypes.CDLL(path)
    lib.aclInit.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
    lib.aclInit.restype = ctypes.c_int
    lib.aclrtSetDevice.argtypes = [ctypes.c_int32]
    lib.aclrtSetDevice.restype = ctypes.c_int
    lib.aclrtMallocHost.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint32]
    lib.aclrtMallocHost.restype = ctypes.c_int
    lib.aclrtHostRegisterV2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32]
    lib.aclrtHostRegisterV2.restype = ctypes.c_int
    lib.aclrtHostGetDevicePointer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32]
    lib.aclrtHostGetDevicePointer.restype = ctypes.c_int
    lib.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
    lib.aclrtHostUnregister.restype = ctypes.c_int
    lib.aclrtFreeHost.argtypes = [ctypes.c_void_p]
    lib.aclrtFreeHost.restype = ctypes.c_int
    return lib


def _probe_register(lib, size: int, flags: int) -> tuple[int, int | None, bool]:
    pointer = ctypes.c_void_p()
    rc_malloc = lib.aclrtMallocHost(ctypes.byref(pointer), size, 0)
    if rc_malloc or not pointer.value:
        return rc_malloc, None, False

    rc_register = lib.aclrtHostRegisterV2(pointer, size, flags)
    device_pointer = ctypes.c_void_p()
    rc_get_pointer = None
    if rc_register == 0:
        rc_get_pointer = lib.aclrtHostGetDevicePointer(pointer, ctypes.byref(device_pointer), 0)
        lib.aclrtHostUnregister(pointer)
    lib.aclrtFreeHost(pointer)
    return rc_register, rc_get_pointer, bool(device_pointer.value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", help="explicit path to libascendcl.so")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--small-mib", type=int, default=128)
    parser.add_argument("--large-gib", type=int, default=12)
    parser.add_argument("--skip-large", action="store_true")
    args = parser.parse_args()

    lib = _load_library(args.library)
    rc_init = lib.aclInit(None, 0)
    rc_device = lib.aclrtSetDevice(args.device_id)
    print(f"aclInit={rc_init} aclrtSetDevice={rc_device} device={args.device_id}")
    if rc_init or rc_device:
        return 1

    cases = [
        ("small MAPPED", args.small_mib * MIB, ACL_HOST_REG_MAPPED),
        ("small MAPPED|PINNED", args.small_mib * MIB, ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED),
    ]
    if not args.skip_large:
        cases.append(("large MAPPED", args.large_gib * GIB, ACL_HOST_REG_MAPPED))

    failed = False
    for name, size, flags in cases:
        rc_register, rc_get_pointer, device_pointer_ok = _probe_register(lib, size, flags)
        print(
            f"{name}: size={size} flags={hex(flags)} register={rc_register} "
            f"getptr={rc_get_pointer} devptr_ok={device_pointer_ok}"
        )
        if rc_register == ACL_ERROR_RT_PARAM_INVALID:
            print("  runtime rejected the flags with ACL_ERROR_RT_PARAM_INVALID (107000)")
        if flags == ACL_HOST_REG_MAPPED and (rc_register != 0 or rc_get_pointer != 0 or not device_pointer_ok):
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
