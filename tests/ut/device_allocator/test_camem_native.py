# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Exercise the actual C++ allocator with a fault-injectable host ACL shim.

This verifies native ownership/error handling, not NPU graph correctness.
Run independently with --confcutdir=tests/ut/device_allocator if no Torch exists.
"""

import ctypes
import importlib.util
import pathlib
import shutil
import subprocess
import sys
import sysconfig

import pytest

FAKE_ACL = r"""
#pragma once
typedef int aclError;
typedef void* aclrtContext;
typedef void* aclrtStream;
typedef void* aclrtDrvMemHandle;
struct aclrtPhysicalMemProp {
  int handleType, allocationType, memAttr;
  struct { int id, type; } location;
  int reserve;
};
enum { ACL_SUCCESS=0, ACL_MEM_HANDLE_TYPE_NONE=0,
       ACL_MEM_ALLOCATION_TYPE_PINNED=0, ACL_HBM_MEM_HUGE=0,
       ACL_MEM_LOCATION_TYPE_DEVICE=0, ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM=0,
       ACL_ERROR_RT_MEMORY_ALLOCATION=100 };
static int fake_allocated=0, fake_mapped=0, fake_reserved=0;
static int fake_fail_alloc=0, fake_fail_map=0, fake_fail_unmap=0, fake_fail_free=0;
void fake_fail(int op) {
  if(op==1) fake_fail_alloc=1;
  if(op==2) fake_fail_map=1;
  if(op==3) fake_fail_unmap=1;
  if(op==4) fake_fail_free=1;
}
int fake_balance(int kind) {
  return kind==1 ? fake_allocated : kind==2 ? fake_mapped : fake_reserved;
}
aclError aclrtGetCurrentContext(aclrtContext* p) { *p=(void*)1; return 0; }
aclError aclrtCreateContext(aclrtContext* p, int d) { *p=(void*)1; return 0; }
aclError aclrtSetCurrentContext(aclrtContext p) { return 0; }
aclError aclrtMemGetAllocationGranularity(aclrtPhysicalMemProp*, int, size_t* n) {
  *n=4096; return 0;
}
aclError aclrtReserveMemAddress(void** p, size_t n, size_t, void*, int) {
  *p=malloc(n); ++fake_reserved; return 0;
}
aclError aclrtReleaseMemAddress(void* p) { free(p); --fake_reserved; return 0; }
aclError aclrtMallocPhysical(aclrtDrvMemHandle* p, size_t, aclrtPhysicalMemProp*, int) {
  if(fake_fail_alloc) { fake_fail_alloc=0; return 100; }
  *p=malloc(1); ++fake_allocated; return 0;
}
aclError aclrtMapMem(void*, size_t, size_t, aclrtDrvMemHandle, int) {
  if(fake_fail_map) { fake_fail_map=0; return 101; }
  ++fake_mapped; return 0;
}
aclError aclrtUnmapMem(void*) {
  if(fake_fail_unmap) { fake_fail_unmap=0; return 102; }
  if(fake_mapped<=0) return 104;
  --fake_mapped; return 0;
}
aclError aclrtFreePhysical(aclrtDrvMemHandle p) {
  if(fake_fail_free) { fake_fail_free=0; return 103; }
  free(p); --fake_allocated; return 0;
}
"""


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    compiler = shutil.which("g++")
    if not compiler or sys.platform != "linux":
        pytest.skip("native allocator shim needs Linux and g++")
    build = tmp_path_factory.mktemp("camem-native")
    (build / "acl").mkdir()
    (build / "acl/acl.h").write_text(FAKE_ACL)
    source = pathlib.Path(__file__).resolve().parents[3] / "csrc/camem_allocator.cpp"
    extension = build / ("vllm_ascend_C" + sysconfig.get_config_var("EXT_SUFFIX"))
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-pthread",
            "-I" + str(build),
            "-I" + sysconfig.get_path("include"),
            str(source),
            "-o",
            str(extension),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    spec = importlib.util.spec_from_file_location("vllm_ascend_C", extension)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lib = ctypes.PyDLL(str(extension))
    lib.my_malloc.argtypes = [ctypes.c_ssize_t, ctypes.c_int, ctypes.c_void_p]
    lib.my_malloc.restype = ctypes.c_void_p
    lib.my_free.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_int, ctypes.c_void_p]
    lib.my_free.restype = None
    allocations = {}

    def remember(handle):
        allocations[handle[2]] = handle

    def forget(ptr):
        return allocations.pop(ptr)

    module.init_module(remember, forget)
    yield module, lib, allocations
    assert allocations == {}
    assert [lib.fake_balance(i) for i in (1, 2, 3)] == [0, 0, 0]


@pytest.mark.parametrize("asleep", [False, True])
def test_free_after_repeated_sleep_wake(native, asleep):
    module, lib, allocations = native
    ptr = lib.my_malloc(100, 0, None)
    handle = allocations[ptr]
    assert handle[:3] == (0, 4096, ptr)
    for _ in range(3):
        module.python_unmap_and_release(*handle)
        module.python_unmap_and_release(*handle)
        assert [lib.fake_balance(i) for i in (1, 2)] == [0, 0]
        module.python_create_and_map(*handle)
        module.python_create_and_map(*handle)
        assert [lib.fake_balance(i) for i in (1, 2)] == [1, 1]
    if asleep:
        module.python_unmap_and_release(*handle)
    lib.my_free(ptr, 100, 0, None)
    assert [lib.fake_balance(i) for i in (1, 2, 3)] == [0, 0, 0]


@pytest.mark.parametrize("operation", [1, 2])
def test_failed_restore_can_retry_without_physical_leak(native, operation):
    module, lib, allocations = native
    ptr = lib.my_malloc(100, 0, None)
    handle = allocations[ptr]
    module.python_unmap_and_release(*handle)
    lib.fake_fail(operation)
    with pytest.raises(RuntimeError, match="aclrtMallocPhysical|aclrtMapMem"):
        module.python_create_and_map(*handle)
    assert [lib.fake_balance(i) for i in (1, 2)] == [0, 0]
    module.python_create_and_map(*handle)
    lib.my_free(ptr, 100, 0, None)


@pytest.mark.parametrize("operation", [3, 4])
def test_partial_release_can_retry_without_double_unmap(native, operation):
    module, lib, allocations = native
    ptr = lib.my_malloc(100, 0, None)
    handle = allocations[ptr]
    lib.fake_fail(operation)
    with pytest.raises(RuntimeError, match="aclrtUnmapMem|aclrtFreePhysical"):
        module.python_unmap_and_release(*handle)
    assert lib.fake_balance(1) == 1
    assert lib.fake_balance(2) == (1 if operation == 3 else 0)
    module.python_unmap_and_release(*handle)
    lib.my_free(ptr, 100, 0, None)
