#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace hixl {
class AscendString {
 public:
  AscendString() = default;
  AscendString(const char* value) : value_(value) {}
  AscendString& operator=(const char* value) {
    value_ = value;
    return *this;
  }
  const char* c_str() const { return value_.c_str(); }
  bool operator<(const AscendString& other) const { return value_ < other.value_; }

 private:
  std::string value_;
};

inline const AscendString OPTION_BUFFER_POOL("buffer_pool");
inline const AscendString OPTION_GLOBAL_RESOURCE_CONFIG("global_resource_config");
inline const AscendString OPTION_LOCAL_COMM_RES("local_comm_res");
inline constexpr int MEM_HOST = 0;
inline constexpr int MEM_DEVICE = 1;
inline constexpr int READ = 0;

using MemHandle = void*;
struct MemDesc {
  uintptr_t addr = 0;
  size_t len = 0;
};
struct TransferOpDesc {
  uintptr_t local;
  uintptr_t remote;
  size_t bytes;
};

namespace cpu_test {
inline int initialize_calls = 0;
inline std::string local_comm_res;
}  // namespace cpu_test

class Hixl {
 public:
  int Initialize(const char*, const std::map<AscendString, AscendString>& options) {
    ++cpu_test::initialize_calls;
    auto found = options.find(OPTION_LOCAL_COMM_RES);
    if (found != options.end()) cpu_test::local_comm_res = found->second.c_str();
    return 0;
  }
  int Finalize() { return 0; }
  int RegisterMem(const MemDesc&, int, MemHandle& handle) {
    handle = nullptr;
    return 0;
  }
  int DeregisterMem(MemHandle) { return 0; }
  int Connect(const char*, int) { return 0; }
  int Disconnect(const char*, int) { return 0; }
  int TransferSync(const char*, int, const std::vector<TransferOpDesc>&, int) { return 0; }
};
}  // namespace hixl
