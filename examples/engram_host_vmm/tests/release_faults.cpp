// Test-only interception: fail exactly one release for a specified mapping.
// Never sent to the device and never installed into a serving process.
#include <dlfcn.h>
#include <unordered_map>
#include <mutex>
#include "acl/acl.h"
static void* target = nullptr;
static int stage = 0;
static aclrtDrvMemHandle target_pa = nullptr;
static std::unordered_map<void*, aclrtDrvMemHandle> handles;
static std::mutex lock;
static bool fail_allocation = false;
extern "C" void engram_fail_allocation_rollback() { fail_allocation = true; }
extern "C" void engram_fail_release(void* ptr, int which) {
  std::lock_guard<std::mutex> guard(lock);
  target = ptr;
  stage = which;
  target_pa = handles.at(ptr);
}
extern "C" aclError aclrtMapMem(void* ptr, size_t size, size_t offset, aclrtDrvMemHandle pa, uint64_t flags) {
  if (fail_allocation) {
    fail_allocation = false;
    target_pa = pa;
    stage = 3;
    return 100000;
  }
  static auto real = reinterpret_cast<aclError (*)(void*, size_t, size_t, aclrtDrvMemHandle, uint64_t)>(
      dlsym(RTLD_NEXT, "aclrtMapMem"));
  auto rc = real(ptr, size, offset, pa, flags);
  if (!rc) {
    std::lock_guard<std::mutex> guard(lock);
    handles[ptr] = pa;
  }
  return rc;
}
extern "C" aclError aclrtFreePhysical(aclrtDrvMemHandle pa) {
  {
    std::lock_guard<std::mutex> guard(lock);
    if (pa == target_pa && stage == 3) {
      stage = 0;
      return 100000;
    }
  }
  static auto real = reinterpret_cast<aclError (*)(aclrtDrvMemHandle)>(dlsym(RTLD_NEXT, "aclrtFreePhysical"));
  return real(pa);
}
extern "C" aclError aclrtUnmapMem(void* ptr) {
  {
    std::lock_guard<std::mutex> guard(lock);
    if (ptr == target && stage == 1) {
      stage = 0;
      return 100000;
    }
  }
  static auto real = reinterpret_cast<aclError (*)(void*)>(dlsym(RTLD_NEXT, "aclrtUnmapMem"));
  return real(ptr);
}
extern "C" aclError aclrtReleaseMemAddress(void* ptr) {
  {
    std::lock_guard<std::mutex> guard(lock);
    if (ptr == target && stage == 2) {
      stage = 0;
      return 100000;
    }
  }
  static auto real = reinterpret_cast<aclError (*)(void*)>(dlsym(RTLD_NEXT, "aclrtReleaseMemAddress"));
  return real(ptr);
}
