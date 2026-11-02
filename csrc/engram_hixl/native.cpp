// SPDX-License-Identifier: Apache-2.0
// C ABI built against the installed HIXL headers, independent of Python's ABI.
#include <acl/acl.h>
#include <hixl/hixl.h>
#include <cstdlib>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;
struct Engine {
  hixl::Hixl engine;
  int device;
  std::mutex mutex;
  std::vector<hixl::MemHandle> regions;
  std::vector<std::string> peers;
};
int status(uint32_t rc, const char* operation) {
  if (rc) last_error = std::string(operation) + " rc=" + std::to_string(rc);
  return static_cast<int>(rc);
}
}  // namespace

extern "C" {
const char* engram_hixl_error() { return last_error.c_str(); }

int engram_hixl_create(const char* endpoint, int device, void** result) {
  *result = nullptr;
  auto* state = new Engine{};
  state->device = device;
  int rc = status(aclrtSetDevice(device), "set device");
  if (!rc) {
    std::map<hixl::AscendString, hixl::AscendString> options;
    options["BufferPool"] = "0:0";
    const char* port = std::getenv("VLLM_ASCEND_ENGRAM_HIXL_LISTEN_PORT");
    std::string resource_config;
    if (port != nullptr && *port != '\0') {
      char* end = nullptr;
      long value = std::strtol(port, &end, 10);
      if (*end != '\0' || value < 1 || value > 65535) {
        last_error = "invalid VLLM_ASCEND_ENGRAM_HIXL_LISTEN_PORT";
        delete state;
        return -1;
      }
      resource_config = "{\"comm_resource_config.listen_port\":" + std::to_string(value) + "}";
      options[hixl::OPTION_GLOBAL_RESOURCE_CONFIG] = resource_config.c_str();
    }
    rc = status(state->engine.Initialize(endpoint, options), "HIXL Initialize");
  }
  if (rc) { delete state; return rc; }
  *result = state;
  return 0;
}

int engram_hixl_register(void* instance, uintptr_t address, size_t bytes, int host, void** handle) {
  auto& state = *static_cast<Engine*>(instance);
  std::lock_guard<std::mutex> lock(state.mutex);
  int rc = status(aclrtSetDevice(state.device), "set device");
  if (rc) return rc;
  hixl::MemDesc region{};
  region.addr = address;
  region.len = bytes;
  rc = status(state.engine.RegisterMem(region, host ? hixl::MEM_HOST : hixl::MEM_DEVICE, *handle), "HIXL RegisterMem");
  if (!rc) state.regions.push_back(*handle);
  return rc;
}

int engram_hixl_deregister(void* instance, void* handle) {
  auto& state = *static_cast<Engine*>(instance);
  std::lock_guard<std::mutex> lock(state.mutex);
  int rc = status(aclrtSetDevice(state.device), "set device");
  if (rc) return rc;
  rc = status(state.engine.DeregisterMem(handle), "HIXL DeregisterMem");
  if (!rc) state.regions.erase(std::remove(state.regions.begin(), state.regions.end(), handle), state.regions.end());
  return rc;
}

int engram_hixl_connect(void* instance, const char* peer) {
  auto& state = *static_cast<Engine*>(instance);
  std::lock_guard<std::mutex> lock(state.mutex);
  int rc = status(aclrtSetDevice(state.device), "set device");
  if (!rc) rc = status(state.engine.Connect(peer, 60000), "HIXL Connect");
  if (!rc) state.peers.emplace_back(peer);
  return rc;
}

int engram_hixl_read(void* instance, const char* peer, uintptr_t local, uintptr_t remote, size_t bytes) {
  auto& state = *static_cast<Engine*>(instance);
  int rc = status(aclrtSetDevice(state.device), "set device");
  if (rc) return rc;
  const hixl::TransferOpDesc request{local, remote, bytes};
  return status(state.engine.TransferSync(peer, hixl::READ, {request}, 10000), "HIXL READ");
}

int engram_hixl_close(void* instance) {
  auto* state = static_cast<Engine*>(instance);
  // Caller must join NPU consumers before closing. On failure it retains ownership.
  int rc = status(aclrtSetDevice(state->device), "set device");
  if (rc) return rc;
  while (!state->peers.empty()) {
    rc = status(state->engine.Disconnect(state->peers.back().c_str(), 10000), "HIXL Disconnect");
    if (rc) return rc;
    state->peers.pop_back();
  }
  while (!state->regions.empty()) {
    rc = status(state->engine.DeregisterMem(state->regions.back()), "HIXL DeregisterMem");
    if (rc) return rc;
    state->regions.pop_back();
  }
  state->engine.Finalize();
  delete state;
  return 0;
}

int engram_pack_rows(uintptr_t destination, uintptr_t source, size_t stride,
                     const int64_t* ids, size_t count, size_t table_rows) {
  if (!stride || table_rows > std::numeric_limits<size_t>::max() / stride ||
      count > std::numeric_limits<size_t>::max() / stride) {
    last_error = "invalid packed row dimensions";
    return -1;
  }
  // Validate the complete request before modifying the reusable staging buffer.
  for (size_t i = 0; i < count; ++i) {
    if (ids[i] < -1 || (ids[i] >= 0 && static_cast<uint64_t>(ids[i]) >= table_rows)) {
      last_error = "row id outside immutable table";
      return -1;
    }
  }
  auto* dst = reinterpret_cast<uint8_t*>(destination);
  const auto* src = reinterpret_cast<const uint8_t*>(source);
  for (size_t i = 0; i < count; ++i) {
    if (ids[i] == -1) std::memset(dst + i * stride, 0, stride);
    else std::memcpy(dst + i * stride, src + ids[i] * stride, stride);
  }
  return 0;
}
}
