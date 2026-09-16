#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <mutex>
#include <vector>

#include "acl/acl.h"

namespace {
thread_local std::string g_error;

struct SharedDescriptor {
  uint64_t magic;
  uint64_t size;
  int32_t owner_physical_device;
  int32_t owner_numa;
  std::array<uint8_t, 128> fabric_handle;
};

struct LocalMapping {
  void* imported_va = nullptr;
  aclrtDrvMemHandle imported_pa = nullptr;
  void* owner_va = nullptr;
  aclrtDrvMemHandle owner_pa = nullptr;
  bool imported_mapped = false;
  bool owner_mapped = false;
  int logical_device = 0;
  int descriptor_fd = -1;
  std::string descriptor_path;
  bool owns_ready = false;
};

constexpr uint64_t kMagic = 0x4d37534841524544ULL;  // M7SHARED
std::unordered_map<uint64_t, LocalMapping> g_mappings;
uint64_t g_next_token = 0;
std::vector<LocalMapping> g_pending;
std::mutex g_mutex;
uint64_t g_alloc_calls = 0;
uint64_t g_free_calls = 0;

void SetError(const std::string& message) { g_error = message; }

void SetAclError(const std::string& stage, aclError ret) {
  const char* recent = aclGetRecentErrMsg();
  SetError(stage + " acl=" + std::to_string(static_cast<int>(ret)) + " " + (recent == nullptr ? "" : recent));
}

bool WaitForFile(const std::string& path, int timeout_seconds) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds);
  while (std::chrono::steady_clock::now() < deadline) {
    if (access(path.c_str(), F_OK) == 0) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  SetError("timeout waiting for " + path);
  return false;
}

bool WriteAll(int fd, const void* data, size_t size) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  while (size != 0) {
    const ssize_t written = write(fd, bytes, size);
    if (written < 0 && errno == EINTR) continue;
    if (written == 0) errno = EIO;
    if (written <= 0) return false;
    bytes += written;
    size -= static_cast<size_t>(written);
  }
  return true;
}

bool ReadAll(int fd, void* data, size_t size) {
  auto* bytes = static_cast<uint8_t*>(data);
  while (size != 0) {
    const ssize_t got = read(fd, bytes, size);
    if (got < 0 && errno == EINTR) continue;
    if (got == 0) errno = EBADMSG;
    if (got <= 0) return false;
    bytes += got;
    size -= static_cast<size_t>(got);
  }
  return true;
}

int ReleaseLocal(LocalMapping& mapping) {
  auto check = [](aclError ret, const char* stage) {
    if (ret != ACL_ERROR_NONE) SetAclError(stage, ret);
    return static_cast<int>(ret);
  };
  int rc = check(aclrtSetDevice(mapping.logical_device), "release set device");
  if (rc) return rc;
  auto release = [&](void*& va, aclrtDrvMemHandle& pa, bool& mapped) {
    if (mapped) {
      int ret = check(aclrtUnmapMem(va), "unmap");
      if (ret) return ret;
      mapped = false;
    }
    if (va) {
      int ret = check(aclrtReleaseMemAddress(va), "release address");
      if (ret) return ret;
      va = nullptr;
    }
    if (pa) {
      int ret = check(aclrtFreePhysical(pa), "free physical");
      if (ret) return ret;
      pa = nullptr;
    }
    return 0;
  };
  rc = release(mapping.imported_va, mapping.imported_pa, mapping.imported_mapped);
  if (rc) return rc;
  rc = release(mapping.owner_va, mapping.owner_pa, mapping.owner_mapped);
  if (rc) return rc;
  if (mapping.descriptor_fd >= 0) {
    close(mapping.descriptor_fd);
    mapping.descriptor_fd = -1;
  }
  if (mapping.owns_ready) {
    if (unlink((mapping.descriptor_path + ".ready").c_str()) && errno != ENOENT) {
      int err = errno;
      SetError("unlink owned ready marker failed");
      return -err;
    }
    mapping.owns_ready = false;
  }
  if (!mapping.descriptor_path.empty()) {
    if (unlink(mapping.descriptor_path.c_str()) && errno != ENOENT) {
      int err = errno;
      SetError("unlink owned descriptor failed");
      return -err;
    }
    mapping.descriptor_path.clear();
  }
  return 0;
}

int Rollback(LocalMapping& mapping, int original_ret) {
  const std::string original_error = g_error;
  if (ReleaseLocal(mapping) != 0) {
    g_pending.push_back(mapping);  // Retain failed releases for explicit retry.
    SetError(original_error + "; rollback failed: " + g_error);
  } else {
    SetError(original_error);
  }
  return original_ret;
}
}  // namespace

extern "C" {

int host_vmm_abi_version() { return 2; }

uint64_t host_vmm_lifecycle_counts() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return (g_alloc_calls << 32) | g_free_calls;
}

uint64_t host_vmm_live_mappings() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_mappings.size() + g_pending.size();
}

int host_vmm_retry_rollbacks() {
  std::lock_guard<std::mutex> lock(g_mutex);
  while (!g_pending.empty()) {
    int rc = ReleaseLocal(g_pending.back());
    if (rc) return rc;
    g_pending.pop_back();
  }
  return 0;
}

// Only after every user has stopped; also retries failed allocation rollbacks.
int host_vmm_close_all() {
  std::lock_guard<std::mutex> lock(g_mutex);
  for (auto it = g_mappings.begin(); it != g_mappings.end();) {
    int rc = ReleaseLocal(it->second);
    if (rc) return rc;
    ++g_free_calls;
    it = g_mappings.erase(it);
  }
  while (!g_pending.empty()) {
    int rc = ReleaseLocal(g_pending.back());
    if (rc) return rc;
    g_pending.pop_back();
  }
  return 0;
}

struct HostSharedRegisteredResult {
  void* host_ptr;
  void* device_ptr;
  size_t size;
  int32_t physical_device;
  int32_t owner;
  int32_t host_observed_location_type;
  int32_t host_observed_location_id;
  int32_t device_observed_location_type;
  int32_t device_observed_location_id;
  uint64_t token;  // Stable even if a partially released VA gets reused.
};

const char* host_shared_last_error() {
  if (!g_error.empty()) return g_error.c_str();
  const char* acl_error = aclGetRecentErrMsg();
  return acl_error == nullptr ? "" : acl_error;
}

int host_shared_registered_alloc(size_t size, int32_t logical_device, const char* path, int32_t owner,
                                 HostSharedRegisteredResult* result) {
  std::lock_guard<std::mutex> lock(g_mutex);
  ++g_alloc_calls;
  if (size == 0 || path == nullptr || path[0] == '\0' || result == nullptr) {
    SetError("invalid host_shared_registered_alloc argument");
    return -1;
  }
  g_error.clear();
  *result = {};
  result->size = size;
  result->owner = owner;
  const std::string descriptor_path(path);
  const std::string ready_path = descriptor_path + ".ready";

  aclError ret = aclrtSetDevice(logical_device);
  if (ret != ACL_ERROR_NONE) {
    SetAclError("aclrtSetDevice", ret);
    return ret;
  }
  ret = aclrtGetPhyDevIdByLogicDevId(logical_device, &result->physical_device);
  if (ret != ACL_ERROR_NONE) {
    SetAclError("aclrtGetPhyDevIdByLogicDevId", ret);
    return ret;
  }

  SharedDescriptor descriptor{};
  descriptor.magic = kMagic;
  descriptor.size = size;
  LocalMapping mapping{};
  mapping.logical_device = logical_device;

  if (owner) {
    // Never replace another live/stale run's capability descriptor.
    if (access(ready_path.c_str(), F_OK) == 0) {
      SetError("ready marker already exists");
      return -EEXIST;
    }
    mapping.descriptor_fd = open(path, O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600);
    if (mapping.descriptor_fd < 0) {
      int err = errno;
      SetError("descriptor claim failed: " + std::string(strerror(err)));
      return -err;
    }
    mapping.descriptor_path = descriptor_path;

    aclrtPhysicalMemProp prop{};
    prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    prop.memAttr = ACL_MEM_P2P_HUGE1G;
    const bool allow_huge2m_fallback = prop.memAttr == ACL_MEM_P2P_HUGE1G;
    prop.location.type = ACL_MEM_LOCATION_TYPE_HOST_NUMA;
    prop.location.id = result->physical_device / 2;
    ret = aclrtMallocPhysical(&mapping.owner_pa, size, &prop, 0);
    if (ret != ACL_ERROR_NONE) {
      prop.location.type = ACL_MEM_LOCATION_TYPE_HOST;
      prop.location.id = 0;
      ret = aclrtMallocPhysical(&mapping.owner_pa, size, &prop, 0);
    }
    if (ret != ACL_ERROR_NONE && allow_huge2m_fallback) {
      prop.memAttr = ACL_MEM_P2P_HUGE;
      ret = aclrtMallocPhysical(&mapping.owner_pa, size, &prop, 0);
    }
    if (ret != ACL_ERROR_NONE) {
      SetAclError("aclrtMallocPhysical(HOST)", ret);
      return Rollback(mapping, ret);
    }
    descriptor.owner_physical_device = result->physical_device;
    descriptor.owner_numa = prop.location.id;
    std::fprintf(stdout, "HOST_VMM_PHYSICAL bytes=%zu mem_attr=%d location=%d:%d\n", size,
                 static_cast<int>(prop.memAttr), static_cast<int>(prop.location.type), prop.location.id);
    std::fflush(stdout);

    ret = aclrtReserveMemAddress(&mapping.owner_va, size, 0, nullptr, 1);
    if (ret == ACL_ERROR_NONE) {
      ret = aclrtMapMem(mapping.owner_va, size, 0, mapping.owner_pa, 0);
      mapping.owner_mapped = ret == ACL_ERROR_NONE;
    }
    if (ret != ACL_ERROR_NONE) {
      SetAclError("owner reserve/map", ret);
      return Rollback(mapping, ret);
    }

    aclrtMemFabricHandle handle{};
    ret = aclrtMemExportToShareableHandleV2(mapping.owner_pa, ACL_RT_VMM_EXPORT_FLAG_DISABLE_PID_VALIDATION,
                                            ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, &handle);
    if (ret != ACL_ERROR_NONE) {
      SetAclError("aclrtMemExportToShareableHandleV2", ret);
      return Rollback(mapping, ret);
    }
    std::memcpy(descriptor.fabric_handle.data(), handle.data, descriptor.fabric_handle.size());

    if (!WriteAll(mapping.descriptor_fd, &descriptor, sizeof(descriptor))) {
      const int err = errno;
      SetError("descriptor write failed: " + std::string(strerror(err)));
      return Rollback(mapping, -err);
    }
    close(mapping.descriptor_fd);
    mapping.descriptor_fd = -1;
  } else {
    if (!WaitForFile(ready_path, 120)) return -ETIMEDOUT;
    const int fd = open(descriptor_path.c_str(), O_RDONLY);
    if (fd < 0 || !ReadAll(fd, &descriptor, sizeof(descriptor))) {
      const int err = errno;
      if (fd >= 0) close(fd);
      SetError("descriptor read failed: " + std::string(strerror(err)));
      return -err;
    }
    close(fd);
    if (descriptor.magic != kMagic || descriptor.size != size) {
      SetError("shared VMM descriptor mismatch");
      return -EINVAL;
    }
  }

  aclrtMemFabricHandle imported_handle{};
  std::memcpy(imported_handle.data, descriptor.fabric_handle.data(), descriptor.fabric_handle.size());
  ret =
      aclrtMemImportFromShareableHandleV2(&imported_handle, ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, 0, &mapping.imported_pa);
  if (ret == ACL_ERROR_NONE) {
    ret = aclrtReserveMemAddress(&mapping.imported_va, size, 0, nullptr, 1);
  }
  if (ret == ACL_ERROR_NONE) {
    ret = aclrtMapMem(mapping.imported_va, size, 0, mapping.imported_pa, 0);
    mapping.imported_mapped = ret == ACL_ERROR_NONE;
  }
  if (ret != ACL_ERROR_NONE) {
    SetAclError("self/cross import reserve/map", ret);
    return Rollback(mapping, ret);
  }

  if (owner) {
    const int marker = open(ready_path.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
    if (marker < 0) {
      const int err = errno;
      SetError("ready marker create failed: " + std::string(strerror(err)));
      return Rollback(mapping, -err);
    }
    close(marker);
    mapping.owns_ready = true;
  }

  result->host_ptr = owner ? mapping.owner_va : mapping.imported_va;
  result->device_ptr = mapping.imported_va;
  aclrtPtrAttributes attributes{};
  if (owner && aclrtPointerGetAttributes(mapping.owner_va, &attributes) == ACL_ERROR_NONE) {
    result->host_observed_location_type = attributes.location.type;
    result->host_observed_location_id = attributes.location.id;
  } else {
    result->host_observed_location_type = ACL_MEM_LOCATION_TYPE_HOST_NUMA;
    result->host_observed_location_id = descriptor.owner_numa;
  }
  attributes = {};
  if (aclrtPointerGetAttributes(mapping.imported_va, &attributes) == ACL_ERROR_NONE) {
    result->device_observed_location_type = attributes.location.type;
    result->device_observed_location_id = attributes.location.id;
  }
  result->token = ++g_next_token;
  if (!result->token || !g_mappings.emplace(result->token, mapping).second) {
    *result = {};
    SetError("VMM allocation token collision");
    return Rollback(mapping, -EOVERFLOW);
  }
  return ACL_ERROR_NONE;
}

int host_shared_registered_free(HostSharedRegisteredResult* result) {
  std::lock_guard<std::mutex> lock(g_mutex);
  ++g_free_calls;
  if (result == nullptr || result->device_ptr == nullptr || result->size == 0) {
    SetError("invalid host_shared_registered_free argument");
    return -1;
  }
  auto found = g_mappings.find(result->token);
  if (found == g_mappings.end()) {
    SetError("shared VMM mapping not found");
    return -ENOENT;
  }
  int rc = ReleaseLocal(found->second);
  if (rc) return rc;
  g_mappings.erase(found);
  result->host_ptr = nullptr;
  result->device_ptr = nullptr;
  result->token = 0;
  return ACL_ERROR_NONE;
}

// Construction never returned a tensor to a caller, but rollback failed.
// Transfer ownership to the native retry list rather than orphaning a token.
int host_shared_registered_defer_free(HostSharedRegisteredResult* result) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!result) return -EINVAL;
  auto found = g_mappings.find(result->token);
  if (found == g_mappings.end()) return -ENOENT;
  g_pending.push_back(std::move(found->second));
  g_mappings.erase(found);
  *result = {};
  return 0;
}

}  // extern "C"
