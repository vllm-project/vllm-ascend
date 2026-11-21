# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Compile the real allocator against a host ACL shim; isolate native crashes.

This tests physical/virtual ownership at tensor deletion, not NPU correctness.
"""

import pathlib
import shutil
import subprocess
import sys
import sysconfig

import pytest

ACL_SHIM = r"""
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
enum { ACL_MEM_HANDLE_TYPE_NONE=0, ACL_MEM_ALLOCATION_TYPE_PINNED=0,
       ACL_HBM_MEM_HUGE=0, ACL_MEM_LOCATION_TYPE_DEVICE=0,
       ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM=0,
       ACL_ERROR_RT_MEMORY_ALLOCATION=100 };
static int physical=0, mapped=0, reserved=0;
int fake_balance(int kind) { return kind==0 ? physical : kind==1 ? mapped : reserved; }
aclError aclrtGetCurrentContext(aclrtContext* p) { *p=(void*)1; return 0; }
aclError aclrtCreateContext(aclrtContext* p, int) { *p=(void*)1; return 0; }
aclError aclrtSetCurrentContext(aclrtContext) { return 0; }
aclError aclrtMemGetAllocationGranularity(aclrtPhysicalMemProp*, int, size_t* n) {
  *n=4096; return 0;
}
aclError aclrtReserveMemAddress(void** p, size_t n, size_t, void*, int) {
  *p=malloc(n); ++reserved; return 0;
}
aclError aclrtReleaseMemAddress(void* p) { free(p); --reserved; return 0; }
aclError aclrtMallocPhysical(aclrtDrvMemHandle* p, size_t, aclrtPhysicalMemProp*, int) {
  *p=malloc(1); ++physical; return 0;
}
aclError aclrtMapMem(void*, size_t, size_t, aclrtDrvMemHandle, int) { ++mapped; return 0; }
aclError aclrtUnmapMem(void*) { if(mapped<=0) return 104; --mapped; return 0; }
aclError aclrtFreePhysical(aclrtDrvMemHandle p) { free(p); --physical; return 0; }
"""

CHILD = r"""
import ctypes, importlib.util, resource, sys
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
extension, case = sys.argv[1:]
spec = importlib.util.spec_from_file_location('vllm_ascend_C', extension)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
lib = ctypes.PyDLL(extension)
lib.my_malloc.argtypes = [ctypes.c_ssize_t, ctypes.c_int, ctypes.c_void_p]
lib.my_malloc.restype = ctypes.c_void_p
lib.my_free.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_int, ctypes.c_void_p]
lib.my_free.restype = None
allocations = {}
def remember(handle):
    assert len(handle) == 4
    allocations[handle[2]] = handle
def forget(pointer):
    return allocations.pop(pointer)
module.init_module(remember, forget)
pointer = lib.my_malloc(4096, 0, None)
handle = allocations[pointer]
assert [lib.fake_balance(i) for i in range(3)] == [1, 1, 1]
if case != 'awake':
    for _ in range(10 if case == 'cycles' else 1):
        module.python_unmap_and_release(*handle)
        assert [lib.fake_balance(i) for i in range(3)] == [0, 0, 1]
        module.python_unmap_and_release(*handle)
        assert [lib.fake_balance(i) for i in range(3)] == [0, 0, 1]
        if case == 'cycles':
            module.python_create_and_map(*handle)
            assert allocations[pointer] == handle
            assert [lib.fake_balance(i) for i in range(3)] == [1, 1, 1]
    if case == 'cycles':
        module.python_unmap_and_release(*handle)
lib.my_free(pointer, 4096, 0, None)
assert not allocations
assert [lib.fake_balance(i) for i in range(3)] == [0, 0, 0]
"""


@pytest.fixture(scope="module")
def allocator_extension(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None or sys.platform != "linux":
        pytest.skip("native allocator ownership test requires Linux and g++")
    assert compiler is not None
    build = tmp_path_factory.mktemp("camem-deallocation")
    (build / "acl").mkdir()
    (build / "acl/acl.h").write_text(ACL_SHIM)
    source = pathlib.Path(__file__).resolve().parents[3] / "csrc/camem_allocator.cpp"
    extension = build / ("vllm_ascend_C" + sysconfig.get_config_var("EXT_SUFFIX"))
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-shared",
            "-fPIC",
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
    return extension


@pytest.mark.parametrize("case", ["awake", "asleep", "cycles"])
def test_tensor_deletion_releases_each_resource_once(allocator_extension, case):
    result = subprocess.run(
        [sys.executable, "-c", CHILD, str(allocator_extension), case], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr
