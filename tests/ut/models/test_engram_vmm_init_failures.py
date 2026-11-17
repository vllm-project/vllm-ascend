# SPDX-License-Identifier: Apache-2.0
"""Check native map rollback and retry using injected ACL failures."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_native_allocation_failure_rollbacks(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    assert compiler is not None
    root = Path(__file__).resolve().parents[3]
    acl = tmp_path / "acl"
    acl.mkdir()
    (acl / "acl.h").write_text(r"""
#include <cstdint>
#include <cstddef>
#include <set>
using aclError = int;
using aclrtDrvMemHandle = void*;
struct Location { int type = 0; int id = 0; };
struct aclrtPhysicalMemProp { int handleType, allocationType, memAttr; Location location; };
struct aclrtMemFabricHandle { uint8_t data[128]; };
constexpr int ACL_ERROR_NONE = 0, ACL_MEM_HANDLE_TYPE_NONE = 0, ACL_MEM_ALLOCATION_TYPE_PINNED = 0;
constexpr int ACL_MEM_P2P_HUGE1G = 1, ACL_MEM_P2P_HUGE = 2, ACL_MEM_LOCATION_TYPE_HOST_NUMA = 3;
constexpr int ACL_MEM_LOCATION_TYPE_HOST = 4, ACL_RT_VMM_EXPORT_FLAG_DISABLE_PID_VALIDATION = 0;
constexpr int ACL_MEM_SHARE_HANDLE_TYPE_FABRIC = 0;
inline bool fail_map = false;
inline bool fail_release = false;
inline uintptr_t serial = 0x1000;
inline std::set<void*> physical, reserved, mapped;
inline void* next_ptr() { serial += 0x1000; return reinterpret_cast<void*>(serial); }
inline const char* aclGetRecentErrMsg() { return "injected"; }
inline int aclrtSetDevice(int) { return 0; }
inline int aclrtMallocPhysical(void** out, size_t, aclrtPhysicalMemProp*, int) {
  *out = next_ptr(); physical.insert(*out); return 0;
}
inline int aclrtReserveMemAddress(void** out, size_t, size_t, void*, int) {
  *out = next_ptr(); reserved.insert(*out); return 0;
}
inline int aclrtMapMem(void* va, size_t, size_t, void*, int) {
  if (fail_map) return 93;
  mapped.insert(va); return 0;
}
inline int aclrtMemExportToShareableHandleV2(void*, int, int, aclrtMemFabricHandle*) {
  return 0;
}
inline int aclrtMemImportFromShareableHandleV2(aclrtMemFabricHandle*, int, int, void** out) {
  *out = next_ptr(); physical.insert(*out); return 0;
}
inline int aclrtUnmapMem(void* va) { return mapped.erase(va) == 1 ? 0 : 96; }
inline int aclrtReleaseMemAddress(void* va) {
  if (fail_release) return 97;
  return reserved.erase(va) == 1 ? 0 : 98;
}
inline int aclrtFreePhysical(void* handle) { return physical.erase(handle) == 1 ? 0 : 99; }
""")
    source = tmp_path / "init_failures.cpp"
    source.write_text(r"""
#include "csrc/engram_vmm/native.cpp"
#include <cassert>
#include <string>
void check_empty(const std::string& path) {
  assert(physical.empty() && reserved.empty() && mapped.empty());
  assert(g_mappings.empty() && g_pending.empty());
  assert(access(path.c_str(), F_OK) != 0 && access((path + ".ready").c_str(), F_OK) != 0);
}
int main(int argc, char** argv) {
  const std::string root(argv[1]);
  // Failed mapping must release the owner allocation and descriptor.
  std::string first = root + "/map";
  HostSharedRegisteredResult failed{};
  fail_map = true;
  assert(host_shared_registered_alloc(4096, 0, first.c_str(), 1, &failed) == 93);
  assert(failed.device_ptr == nullptr && failed.token == 0);
  check_empty(first);
  // Failed rollback keeps its descriptor claim and native pending state;
  // once release is available, retry removes both resources and claim.
  std::string path = root + "/retry";
  HostSharedRegisteredResult result{};
  fail_map = true; fail_release = true;
  assert(host_shared_registered_alloc(4096, 0, path.c_str(), 1, &result) == 93);
  assert(g_mappings.size() + g_pending.size() == 1 && access(path.c_str(), F_OK) == 0);
  assert(!reserved.empty() && !physical.empty());
  fail_release = false; fail_map = false;
  assert(host_vmm_retry_rollbacks() == 0);
  check_empty(path);
}
""")
    binary = tmp_path / "init_failures"
    subprocess.run(
        [compiler, "-std=c++17", "-pthread", "-I", str(tmp_path), "-I", str(root), str(source), "-o", str(binary)],
        check=True,
    )
    subprocess.run([str(binary), str(tmp_path)], check=True)
