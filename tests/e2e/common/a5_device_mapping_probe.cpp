// SPDX-License-Identifier: Apache-2.0
// Built against the runner's own SDK headers, not hand-written ctypes ABIs.
#include <acl/acl.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

template <typename F>
aclError Call(const char* name, int32_t input, F function) {
  std::cout << "{\"event\":\"start\",\"api\":" << std::quoted(name) << ",\"input\":" << input << "}" << std::endl;
  const auto result = function();
  std::cout << "{\"event\":\"return\",\"api\":" << std::quoted(name) << ",\"rc\":" << result << "}" << std::endl;
  return result;
}

void Libraries() {
  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    const auto path = line.find('/');
    if (path != std::string::npos && line.find(".so", path) != std::string::npos &&
        (line.find("ascend", path) != std::string::npos || line.find("runtime", path) != std::string::npos ||
         line.find("drv", path) != std::string::npos)) {
      std::cout << "{\"event\":\"library\",\"path\":" << std::quoted(line.substr(path)) << "}" << std::endl;
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 2 || (std::string(argv[1]) != "0" && std::string(argv[1]) != "1")) {
    return 2;
  }
  const int32_t user = std::atoi(argv[1]);
  bool ok = true;
  const auto init = Call("aclInit", -1, [] { return aclInit(nullptr); });
  Libraries();
  if (init != ACL_SUCCESS) {
    return 1;
  }
  uint32_t count = 0;
  ok &= Call("aclrtGetDeviceCount", -1, [&] { return aclrtGetDeviceCount(&count); }) == ACL_SUCCESS;
  const auto selected = Call("aclrtSetDevice", user, [&] { return aclrtSetDevice(user); });
  ok &= selected == ACL_SUCCESS;
  int32_t current = -1, logical = -1, physical = -1;
  int32_t user_from_physical = -1, user_from_logical = -1;
  if (selected == ACL_SUCCESS) {
    ok &= Call("aclrtGetDevice", -1, [&] { return aclrtGetDevice(&current); }) == ACL_SUCCESS;
    ok &= Call("aclrtGetLogicDevIdByUserDevId", user, [&] { return aclrtGetLogicDevIdByUserDevId(user, &logical); }) ==
          ACL_SUCCESS;
    ok &= Call("aclrtGetPhyDevIdByUserDevId", user, [&] { return aclrtGetPhyDevIdByUserDevId(user, &physical); }) ==
          ACL_SUCCESS;
    if (physical >= 0) {
      ok &= Call("aclrtGetUserDevIdByPhyDevId", physical,
                 [&] { return aclrtGetUserDevIdByPhyDevId(physical, &user_from_physical); }) == ACL_SUCCESS;
    }
    if (logical >= 0) {
      ok &= Call("aclrtGetUserDevIdByLogicDevId", logical,
                 [&] { return aclrtGetUserDevIdByLogicDevId(logical, &user_from_logical); }) == ACL_SUCCESS;
    }
  }
  std::cout << "{\"event\":\"mapping\",\"user\":" << user << ",\"count\":" << count << ",\"current\":" << current
            << ",\"logical\":" << logical << ",\"physical\":" << physical
            << ",\"user_from_physical\":" << user_from_physical << ",\"user_from_logical\":" << user_from_logical << "}"
            << std::endl;
  Libraries();
  if (selected == ACL_SUCCESS) {
    ok &= Call("aclrtResetDevice", user, [&] { return aclrtResetDevice(user); }) == ACL_SUCCESS;
  }
  ok &= Call("aclFinalize", -1, [] { return aclFinalize(); }) == ACL_SUCCESS;
  return ok ? 0 : 1;
}
