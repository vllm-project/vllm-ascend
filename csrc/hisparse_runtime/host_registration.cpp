// SPDX-License-Identifier: Apache-2.0
// An existing CPU pool needs an alias-returning registration interface. Keep
// this minimal compatibility boundary under a PTA guard. Installed PTA headers
// declare these wrappers but libtorch_npu does not export them; evidence2361
// includes the failed load and dynamic symbol inventory. No Python CANN ABI.
#include <torch/library.h>
#include <ATen/ATen.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <acl/acl.h>
#include <limits>
#include <tuple>

namespace {
std::tuple<int64_t, int64_t> register_host(const at::Tensor& host, int64_t device) {
  TORCH_CHECK(host.device().is_cpu() && host.is_contiguous() && host.numel() > 0,
              "Host registration requires a nonempty contiguous CPU owner");
  TORCH_CHECK(device >= 0 && device <= std::numeric_limits<c10::DeviceIndex>::max(),
              "Invalid NPU device index");
  const c10_npu::NPUGuard guard(c10::Device(c10::DeviceType::PrivateUse1,
                                         static_cast<c10::DeviceIndex>(device)));
  aclrtHostMemMapCapability capability{};
  const auto capability_status = aclrtHostMemMapCapabilities(
      static_cast<uint32_t>(device), static_cast<aclrtHacType>(3), &capability);
  TORCH_CHECK(capability_status == ACL_SUCCESS && static_cast<int>(capability) != 0,
              "AIV mapped Host memory is unavailable: ", capability_status);
  void* mapped = nullptr;
  const auto status = aclrtHostRegister(
      host.data_ptr(), host.numel() * host.element_size(),
      static_cast<aclrtHostRegisterType>(0), &mapped);
  // Return the actual alias, never assume CPU and NPU virtual addresses match.
  // Return status too, so the Python owner can retain an uncertain mapping.
  return {static_cast<int64_t>(status),
          static_cast<int64_t>(reinterpret_cast<uintptr_t>(mapped))};
}

int64_t unregister_host(const at::Tensor& host, int64_t device) {
  TORCH_CHECK(host.device().is_cpu() && host.is_contiguous(), "Invalid CPU mapping owner");
  TORCH_CHECK(device >= 0 && device <= std::numeric_limits<c10::DeviceIndex>::max(),
              "Invalid NPU device index");
  const c10_npu::NPUGuard guard(c10::Device(c10::DeviceType::PrivateUse1,
                                         static_cast<c10::DeviceIndex>(device)));
  const auto status = aclrtHostUnregister(host.data_ptr());
  return static_cast<int64_t>(status);
}
}  // namespace

TORCH_LIBRARY(hisparse_pta_host2511, m) {
  m.def("register_host(Tensor host, int device) -> (int, int)");
  m.def("unregister_host(Tensor host, int device) -> int");
}
TORCH_LIBRARY_IMPL(hisparse_pta_host2511, CPU, m) {
  m.impl("register_host", &register_host);
  m.impl("unregister_host", &unregister_host);
}
