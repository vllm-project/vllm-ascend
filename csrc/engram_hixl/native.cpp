// SPDX-License-Identifier: Apache-2.0
// C ABI built against the installed HIXL headers, independent of Python's ABI.
#include <acl/acl.h>
#include <hixl/hixl.h>
#include <cstdlib>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;
struct Engine {
  hixl::Hixl engine;
  int device;
  aclrtContext context = nullptr;
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

int engram_hixl_create_v2(const char* endpoint, int device, const char* port, const char* local_comm, void** result) {
  *result = nullptr;
  auto* state = new Engine{};
  state->device = device;
  int rc = status(aclrtSetDevice(device), "set device");
  if (!rc) {
    std::map<hixl::AscendString, hixl::AscendString> options;
    options[hixl::OPTION_BUFFER_POOL] = "0:0";
    std::string resource_config;
    std::string local_comm_res;
    const bool has_port = port != nullptr && *port != '\0';
    const bool has_local_comm = local_comm != nullptr && *local_comm != '\0';
    if (has_port && has_local_comm) {
      last_error = "listen_port and local_comm_res are mutually exclusive";
      delete state;
      return -1;
    }
    if (has_port) {
      char* end = nullptr;
      long value = std::strtol(port, &end, 10);
      if (*end != '\0' || value < 1 || value > 65535) {
        last_error = "invalid listen_port";
        delete state;
        return -1;
      }
      resource_config = "{\"comm_resource_config.listen_port\":" + std::to_string(value) + "}";
      options[hixl::OPTION_GLOBAL_RESOURCE_CONFIG] = resource_config.c_str();
    }
    if (has_local_comm) {
      // Endpoint resources are deployment input, generated for the target device.
      std::string resource_path(local_comm);
      const std::string marker = "{physical}";
      const auto marker_pos = resource_path.find(marker);
      if (marker_pos != std::string::npos) {
        int32_t physical = -1;
        rc = status(aclrtGetPhyDevIdByUserDevId(device, &physical), "get physical device");
        if (rc) { delete state; return rc; }
        resource_path.replace(marker_pos, marker.size(), std::to_string(physical));
        if (resource_path.find(marker) != std::string::npos) {
          last_error = "local_comm_res has more than one {physical} marker";
          delete state;
          return -1;
        }
      }
      std::ifstream input(resource_path, std::ios::binary);
      if (!input) {
        last_error = "cannot read local_comm_res";
        delete state;
        return -1;
      }
      local_comm_res.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
      if (!input && !input.eof()) {
        last_error = "cannot read local_comm_res";
        delete state;
        return -1;
      }
      if (local_comm_res.empty() || local_comm_res.size() > (1u << 20)) {
        last_error = "local_comm_res must be 1 byte to 1 MiB";
        delete state;
        return -1;
      }
      options[hixl::OPTION_LOCAL_COMM_RES] = local_comm_res.c_str();
    }
    rc = status(state->engine.Initialize(endpoint, options), "HIXL Initialize");
    if (!rc) {
      rc = status(aclrtGetCurrentContext(&state->context), "get HIXL context");
      if (rc) state->engine.Finalize();
    }
  }
  if (rc) { delete state; return rc; }
  *result = state;
  return 0;
}

int engram_hixl_register(void* instance, uintptr_t address, size_t bytes, int host, void** handle) {
  auto& state = *static_cast<Engine*>(instance);
  std::lock_guard<std::mutex> lock(state.mutex);
  int rc = status(aclrtSetCurrentContext(state.context), "set HIXL context");
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
  int rc = status(aclrtSetCurrentContext(state.context), "set HIXL context");
  if (rc) return rc;
  rc = status(state.engine.DeregisterMem(handle), "HIXL DeregisterMem");
  if (!rc) state.regions.erase(std::remove(state.regions.begin(), state.regions.end(), handle), state.regions.end());
  return rc;
}

int engram_hixl_connect(void* instance, const char* peer) {
  auto& state = *static_cast<Engine*>(instance);
  std::lock_guard<std::mutex> lock(state.mutex);
  int rc = status(aclrtSetCurrentContext(state.context), "set HIXL context");
  if (!rc) rc = status(state.engine.Connect(peer, 60000), "HIXL Connect");
  if (!rc) state.peers.emplace_back(peer);
  return rc;
}

int engram_hixl_read(void* instance, const char* peer, uintptr_t local, uintptr_t remote, size_t bytes) {
  auto& state = *static_cast<Engine*>(instance);
  int rc = status(aclrtSetCurrentContext(state.context), "set HIXL context");
  if (rc) return rc;
  const hixl::TransferOpDesc request{local, remote, bytes};
  return status(state.engine.TransferSync(peer, hixl::READ, {request}, 10000), "HIXL READ");
}

int engram_hixl_close(void* instance) {
  auto* state = static_cast<Engine*>(instance);
  // Caller must join NPU consumers before closing. On failure it retains ownership.
  int rc = status(aclrtSetCurrentContext(state->context), "set HIXL context");
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
