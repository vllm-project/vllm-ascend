#pragma once

#include <cstdint>

using aclError = int;
using aclrtContext = void*;

namespace engram_hixl_cpu_test {
inline int acl_calls = 0;
}

inline aclError aclrtSetDevice(int) {
  ++engram_hixl_cpu_test::acl_calls;
  return 0;
}
inline aclError aclrtGetPhyDevIdByUserDevId(int, int32_t* physical) {
  ++engram_hixl_cpu_test::acl_calls;
  *physical = 0;
  return 0;
}
inline aclError aclrtGetCurrentContext(aclrtContext* context) {
  ++engram_hixl_cpu_test::acl_calls;
  *context = nullptr;
  return 0;
}
inline aclError aclrtSetCurrentContext(aclrtContext) {
  ++engram_hixl_cpu_test::acl_calls;
  return 0;
}
