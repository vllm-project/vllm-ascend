# SPDX-License-Identifier: Apache-2.0
"""Inject allocation, import and map failures into the real native allocator."""

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
struct aclrtPtrAttributes { Location location; };
struct aclrtMemFabricHandle { uint8_t data[128]; };
constexpr int ACL_ERROR_NONE = 0, ACL_MEM_HANDLE_TYPE_NONE = 0, ACL_MEM_ALLOCATION_TYPE_PINNED = 0;
constexpr int ACL_MEM_P2P_HUGE1G = 1, ACL_MEM_P2P_HUGE = 2, ACL_MEM_LOCATION_TYPE_HOST_NUMA = 3;
constexpr int ACL_MEM_LOCATION_TYPE_HOST = 4, ACL_RT_VMM_EXPORT_FLAG_DISABLE_PID_VALIDATION = 0;
constexpr int ACL_MEM_SHARE_HANDLE_TYPE_FABRIC = 0;
inline int fail_kind = 0, fail_occurrence = 1, reserve_calls = 0, map_calls = 0;
inline bool fail_release = false;
inline uintptr_t serial = 0x1000;
inline std::set<void*> physical, reserved, mapped;
inline void* next_ptr() { serial += 0x1000; return reinterpret_cast<void*>(serial); }
inline const char* aclGetRecentErrMsg() { return "injected"; }
inline int aclrtSetDevice(int) { return 0; }
inline int aclrtGetPhyDevIdByLogicDevId(int, int32_t* out) { *out = 0; return 0; }
inline int aclrtMallocPhysical(void** out, size_t, aclrtPhysicalMemProp*, int) {
  if (fail_kind == 1) return 91;
  *out = next_ptr(); physical.insert(*out); return 0;
}
inline int aclrtReserveMemAddress(void** out, size_t, size_t, void*, int) {
  if (++reserve_calls == fail_occurrence && fail_kind == 2) return 92;
  *out = next_ptr(); reserved.insert(*out); return 0;
}
inline int aclrtMapMem(void* va, size_t, size_t, void*, int) {
  if (++map_calls == fail_occurrence && fail_kind == 3) return 93;
  mapped.insert(va); return 0;
}
inline int aclrtMemExportToShareableHandleV2(void*, int, int, aclrtMemFabricHandle*) {
  return fail_kind == 4 ? 94 : 0;
}
inline int aclrtMemImportFromShareableHandleV2(aclrtMemFabricHandle*, int, int, void** out) {
  if (fail_kind == 5) return 95;
  *out = next_ptr(); physical.insert(*out); return 0;
}
inline int aclrtUnmapMem(void* va) { return mapped.erase(va) == 1 ? 0 : 96; }
inline int aclrtReleaseMemAddress(void* va) {
  if (fail_release) return 97;
  return reserved.erase(va) == 1 ? 0 : 98;
}
inline int aclrtFreePhysical(void* handle) { return physical.erase(handle) == 1 ? 0 : 99; }
inline int aclrtPointerGetAttributes(void*, aclrtPtrAttributes*) { return -1; }
""")
    source = tmp_path / "init_failures.cpp"
    source.write_text(r"""
#include "csrc/engram_vmm/native.cpp"
#include <cassert>
#include <string>
void reset_fault(int kind, int occurrence = 1) {
  fail_kind = kind; fail_occurrence = occurrence; reserve_calls = 0; map_calls = 0;
}
void check_empty(const std::string& path) {
  assert(physical.empty() && reserved.empty() && mapped.empty());
  assert(host_vmm_live_mappings() == 0);
  assert(access(path.c_str(), F_OK) != 0 && access((path + ".ready").c_str(), F_OK) != 0);
}
int main(int argc, char** argv) {
  const std::string root(argv[1]);
  // All physical-allocation fallbacks, owner VA reserve/map, export,
  // self-import, and self-import VA reserve/map must unwind prior resources.
  const int kinds[] = {1, 2, 3, 4, 5, 2, 3};
  const int occurrences[] = {1, 1, 1, 1, 1, 2, 2};
  for (int i = 0; i < 7; ++i) {
    std::string path = root + "/owner-" + std::to_string(i);
    HostSharedRegisteredResult result{};
    reset_fault(kinds[i], occurrences[i]);
    assert(host_shared_registered_alloc(4096, 0, path.c_str(), 1, &result) != 0);
    assert(result.device_ptr == nullptr && result.token == 0);
    check_empty(path);
  }
  // A consumer's failed import/reserve/map must leave the owner's live
  // capability and mappings untouched, then allow the owner to close.
  for (int kind : {5, 2, 3}) {
    std::string path = root + "/consumer-" + std::to_string(kind);
    HostSharedRegisteredResult owner{}, consumer{};
    reset_fault(0);
    assert(host_shared_registered_alloc(4096, 0, path.c_str(), 1, &owner) == 0);
    auto owner_physical = physical; auto owner_reserved = reserved; auto owner_mapped = mapped;
    reset_fault(kind);
    assert(host_shared_registered_alloc(4096, 0, path.c_str(), 0, &consumer) != 0);
    assert(physical == owner_physical && reserved == owner_reserved && mapped == owner_mapped);
    assert(host_vmm_live_mappings() == 1 && access((path + ".ready").c_str(), F_OK) == 0);
    reset_fault(0);
    assert(host_shared_registered_free(&owner) == 0);
    check_empty(path);
  }
  // Failed rollback keeps its descriptor claim and native pending state;
  // once release is available, retry removes both resources and claim.
  std::string path = root + "/retry";
  HostSharedRegisteredResult result{};
  reset_fault(3); fail_release = true;
  assert(host_shared_registered_alloc(4096, 0, path.c_str(), 1, &result) == 93);
  assert(host_vmm_live_mappings() == 1 && access(path.c_str(), F_OK) == 0);
  assert(!reserved.empty() && !physical.empty());
  fail_release = false; reset_fault(0);
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
