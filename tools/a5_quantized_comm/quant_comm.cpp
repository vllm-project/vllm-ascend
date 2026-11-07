// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <dlfcn.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>

#include "acl/acl.h"
#include "aclnn_quant_all_reduce.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace {
constexpr int64_t kMxBlockSize = 32;
constexpr int64_t kScalesPerPair = 2;
constexpr int64_t kHiddenAlignment = 128;
constexpr int64_t kMinHidden = 1024;
constexpr int64_t kMaxHidden = 8192;

struct TensorDeleter {
  void operator()(aclTensor* tensor) const {
    if (tensor != nullptr) {
      aclDestroyTensor(tensor);
    }
  }
};
using AclTensorPtr = std::unique_ptr<aclTensor, TensorDeleter>;

const char* workspace_api_library() {
  Dl_info info{};
  if (dladdr(reinterpret_cast<const void*>(&aclnnQuantAllReduceGetWorkspaceSize), &info) != 0 &&
      info.dli_fname != nullptr) {
    return info.dli_fname;
  }
  return "<unknown>";
}

AclTensorPtr make_descriptor(const at::Tensor& tensor, aclDataType dtype) {
  // All tensors are contiguous ND. data_ptr() already includes storage_offset.
  // In particular, a three-dimensional scale tensor must NOT become NCL.
  AclTensorPtr result(aclCreateTensor(tensor.sizes().data(), tensor.dim(), dtype, tensor.strides().data(), 0,
                                      ACL_FORMAT_ND, tensor.sizes().data(), tensor.dim(), tensor.data_ptr()));
  TORCH_CHECK(result != nullptr, "aclCreateTensor failed");
  return result;
}

at::Tensor quant_all_reduce(const at::Tensor& values, const at::Tensor& scale_bytes, const std::string& group,
                            bool output_fp32) {
  TORCH_CHECK(values.device().type() == c10::DeviceType::PrivateUse1, "values must be on NPU");
  TORCH_CHECK(scale_bytes.device() == values.device(), "scales and values must be on the same NPU");
  TORCH_CHECK(values.scalar_type() == at::ScalarType::Float8_e4m3fn, "values must have dtype float8_e4m3fn");
  TORCH_CHECK(scale_bytes.scalar_type() == at::kByte, "scales must be raw E8M0 bytes (torch.uint8), not cast values");
  TORCH_CHECK(values.dim() == 2 && values.size(0) > 0, "values must have nonempty shape [T, H]");
  const int64_t hidden = values.size(1);
  TORCH_CHECK(hidden >= kMinHidden && hidden <= kMaxHidden && hidden % kHiddenAlignment == 0,
              "H must be in [1024, 8192] and aligned to 128");
  TORCH_CHECK(scale_bytes.dim() == 3 && scale_bytes.size(0) == values.size(0) &&
                  scale_bytes.size(1) == hidden / (kMxBlockSize * kScalesPerPair) &&
                  scale_bytes.size(2) == kScalesPerPair,
              "scale bytes must have shape [T, H/64, 2]");
  TORCH_CHECK(values.is_contiguous() && scale_bytes.is_contiguous(), "inputs must be contiguous ND tensors");
  TORCH_CHECK(!group.empty(), "group must be a dedicated HCCL communicator name");

  const c10::DeviceGuard guard(values.device());
  auto stream = c10_npu::getCurrentNPUStream(values.get_device());
  // A no-op/stub API must leave a visible failure, not allocator-dependent zeros.
  auto output = at::full(values.sizes(), std::numeric_limits<float>::quiet_NaN(),
                         values.options().dtype(output_fp32 ? at::kFloat : at::kBFloat16));
  // This diagnostic wrapper is intentionally synchronous and eager-only.
  // Drain the host task queue and complete the sentinel fill before direct ACLNN.
  stream.synchronize();
  const auto acl_stream = stream.stream();
  auto x_desc = make_descriptor(values, ACL_FLOAT8_E4M3FN);
  auto scale_desc = make_descriptor(scale_bytes, ACL_FLOAT8_E8M0);
  auto output_desc = make_descriptor(output, output_fp32 ? ACL_FLOAT : ACL_BF16);
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  auto status = aclnnQuantAllReduceGetWorkspaceSize(x_desc.get(), scale_desc.get(), group.c_str(), "sum",
                                                    output_desc.get(), &workspace_size, &executor);
  TORCH_CHECK(status == ACL_SUCCESS, "aclnnQuantAllReduceGetWorkspaceSize failed: ", status);
  TORCH_CHECK(executor != nullptr,
              "aclnnQuantAllReduceGetWorkspaceSize returned ACL_SUCCESS but left executor=nullptr; workspace_size=",
              workspace_size, "; symbol_library=", workspace_api_library(),
              ". The native execution plan was not created. Check for stub/no-op op-api exports and which shared "
              "library the loader resolved. Symbol presence and return code 0 alone do not establish support.");
  TORCH_CHECK(workspace_size <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()), "workspace too large");
  auto workspace = at::empty({static_cast<int64_t>(workspace_size)}, values.options().dtype(at::kByte));
  status =
      aclnnQuantAllReduce(workspace_size == 0 ? nullptr : workspace.data_ptr(), workspace_size, executor, acl_stream);
  // Keep workspace, descriptors, inputs and output alive until device execution
  // completes, including when the launch reports an error.
  const auto sync_status = aclrtSynchronizeStream(acl_stream);
  TORCH_CHECK(status == ACL_SUCCESS, "aclnnQuantAllReduce launch failed: ", status);
  TORCH_CHECK(sync_status == ACL_SUCCESS, "aclnnQuantAllReduce stream synchronization failed: ", sync_status);
  return output;
}
}  // namespace

TORCH_LIBRARY(a5_quant_comm_probe, m) {
  m.def("quant_all_reduce(Tensor values, Tensor scale_bytes, str group, bool output_fp32=False) -> Tensor");
}

TORCH_LIBRARY_IMPL(a5_quant_comm_probe, PrivateUse1, m) { m.impl("quant_all_reduce", TORCH_FN(quant_all_reduce)); }
