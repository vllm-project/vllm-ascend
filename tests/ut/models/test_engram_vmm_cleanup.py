# SPDX-License-Identifier: Apache-2.0
"""Exercise the native cleanup code with injected ACL failures, without an NPU."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_native_cleanup_continues_and_retries(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    root = Path(__file__).resolve().parents[3]
    acl = tmp_path / "acl"
    acl.mkdir()
    (acl / "acl.h").write_text(r"""
#include <cstdint>
using aclError = int;
using aclrtDrvMemHandle = void*;
struct Location { int type = 0; int id = 0; };
struct aclrtPhysicalMemProp { int handleType, allocationType, memAttr; Location location; };
struct aclrtPtrAttributes { Location location; };
struct aclrtMemFabricHandle { uint8_t data[128]; };
constexpr int ACL_ERROR_NONE = 0, ACL_MEM_HANDLE_TYPE_NONE = 0, ACL_MEM_ALLOCATION_TYPE_PINNED = 0;
constexpr int ACL_MEM_P2P_HUGE1G = 1, ACL_MEM_P2P_HUGE = 2, ACL_MEM_LOCATION_TYPE_HOST_NUMA = 3;
constexpr int ACL_MEM_LOCATION_TYPE_HOST = 4, ACL_RT_VMM_EXPORT_FLAG_DISABLE_PID_VALIDATION = 0;
constexpr int ACL_MEM_SHARE_HANDLE_TYPE_FABRIC = 0;
inline bool fail_devices = true;
inline bool fail_unmap = false;
inline int free_calls = 0, release_address_calls = 0;
inline const char* aclGetRecentErrMsg() { return "mock failure"; }
inline int aclrtSetDevice(int d) { return fail_devices && (d == 1 || d == 3) ? 100 + d : 0; }
template<class... T> int aclrtUnmapMem(T...) { return fail_unmap ? 77 : 0; }
template<class... T> int aclrtReleaseMemAddress(T...) { ++release_address_calls; return 0; }
template<class... T> int aclrtFreePhysical(T...) { ++free_calls; return 0; }
template<class... T> int aclrtGetPhyDevIdByLogicDevId(T...) { return 0; }
template<class... T> int aclrtMallocPhysical(T...) { return 0; }
template<class... T> int aclrtMapMem(T...) { return 0; }
template<class... T> int aclrtMemExportToShareableHandleV2(T...) { return 0; }
template<class... T> int aclrtMemImportFromShareableHandleV2(T...) { return 0; }
template<class... T> int aclrtPointerGetAttributes(T...) { return 0; }
template<class... T> int aclrtReserveMemAddress(T...) { return 0; }
""")
    source = tmp_path / "test.cpp"
    source.write_text(r"""
#include "csrc/engram_vmm/native.cpp"
#include <cassert>
int main(int argc, char** argv) {
  LocalMapping a, b, c, d;
  a.logical_device = 1; b.logical_device = 2; c.logical_device = 3; d.logical_device = 4;
  g_mappings.emplace(1, a); g_mappings.emplace(2, b);
  g_pending.push_back(c); g_pending.push_back(d);
  assert(host_vmm_close_all() == 101);
  assert(g_error.find("acl=101") != std::string::npos);
  assert(g_mappings.size() == 1 && g_mappings.count(1));
  assert(g_pending.size() == 1 && g_pending[0].logical_device == 3);
  assert(g_free_calls == 1 && host_vmm_live_mappings() == 2);
  g_pending.push_back(d);
  assert(host_vmm_retry_rollbacks() == 103);
  assert(g_pending.size() == 1 && g_pending[0].logical_device == 3);
  fail_devices = false;
  assert(host_vmm_retry_rollbacks() == 0 && g_pending.empty());
  assert(host_vmm_close_all() == 0);
  assert(host_vmm_live_mappings() == 0 && g_free_calls == 2);
  // A failed unmap invalidates publication and closes the fd, while the
  // exclusive descriptor claim and dependent device resources survive retry.
  std::string path = std::string(argv[1]) + "/descriptor";
  LocalMapping mapping;
  mapping.logical_device = 2;
  mapping.descriptor_path = path;
  mapping.descriptor_fd = open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
  assert(mapping.descriptor_fd >= 0);
  int original_fd = mapping.descriptor_fd;
  int marker = open((path + ".ready").c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
  assert(marker >= 0); close(marker); mapping.owns_ready = true;
  mapping.imported_va = reinterpret_cast<void*>(1);
  mapping.imported_pa = reinterpret_cast<void*>(2);
  mapping.imported_mapped = true;
  fail_unmap = true;
  assert(ReleaseLocal(mapping) == 77);
  assert(mapping.descriptor_fd == -1 && fcntl(original_fd, F_GETFD) == -1);
  assert(access((path + ".ready").c_str(), F_OK) == -1);
  assert(access(path.c_str(), F_OK) == 0 && mapping.imported_mapped);
  assert(free_calls == 0 && release_address_calls == 0);
  fail_unmap = false;
  assert(ReleaseLocal(mapping) == 0);
  assert(access(path.c_str(), F_OK) == -1);
  assert(free_calls == 1 && release_address_calls == 1);
}
""")
    binary = tmp_path / "test"
    subprocess.run(
        [compiler, "-std=c++17", "-pthread", "-I", str(tmp_path), "-I", str(root), str(source), "-o", str(binary)],
        check=True,
    )
    subprocess.run([str(binary), str(tmp_path)], check=True)
