// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <dlfcn.h>
#include <initializer_list>
#include <stdexcept>
#include <string>

// Include after op_api_common.h. Keep official QLI's exceptional tensor
// representation and library selection out of the shared ACLNN adapter.
struct CannQliTensor {
  const at::Tensor& tensor;
};

inline aclTensor* ConvertType(const CannQliTensor& input) {
  const auto& tensor = input.tensor;
  if (tensor.scalar_type() != at::ScalarType::Float4_e2m1fn_x2) {
    return ConvertType(tensor);
  }
  TORCH_CHECK(IsOpInputBaseFormat(tensor) && tensor.dim() > 0 && tensor.stride(-1) == 1,
              "Official QLI FP4 tensors require base format and a packed contiguous last dimension");
  auto shape = tensor.sizes().vec();
  auto strides = tensor.strides().vec();
  constexpr int64_t values_per_byte = 2;
  shape.back() *= values_per_byte;
  for (size_t axis = 0; axis + 1 < strides.size(); ++axis) {
    strides[axis] *= values_per_byte;
  }
  int64_t storage_size = tensor.storage().nbytes() * values_per_byte;
  static const auto create = GET_OP_API_FUNC(aclCreateTensor);
  TORCH_CHECK(create != nullptr, "aclCreateTensor is unavailable");
  return create(shape.data(), shape.size(), ACL_FLOAT4_E2M1, strides.data(),
                tensor.storage_offset() * values_per_byte, ACL_FORMAT_ND,
                &storage_size, 1, const_cast<void*>(tensor.storage().data()));
}

namespace vllm_ascend {

inline void* OpenOfficialQliLibrary(std::initializer_list<const char*> symbols) {
  std::string errors;
  for (const char* library : {"libopapi_transformer.so", "libopapi.so"}) {
    void* handle = dlopen(library, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      const char* error = dlerror();
      errors += std::string(library) + ": " + (error ? error : "cannot load") + "; ";
      continue;
    }
    bool complete = true;
    for (const char* symbol : symbols) {
      if (dlsym(handle, symbol) == nullptr) {
        errors += std::string(library) + ": missing " + symbol + "; ";
        complete = false;
      }
    }
    if (complete) {
      return handle;  // Keep the provider loaded while pointers or graphs live.
    }
    dlclose(handle);
  }
  throw std::runtime_error("No compatible official CANN operator library. " + errors);
}

inline void* GetOfficialQliFuncAddr(const char* symbol) {
  static void* handle = OpenOfficialQliLibrary({
      "aclnnQuantLightningIndexerV2GetWorkspaceSize",
      "aclnnQuantLightningIndexerV2",
      "aclnnQuantLightningIndexerV2MetadataGetWorkspaceSize",
      "aclnnQuantLightningIndexerV2Metadata",
  });
  void* address = dlsym(handle, symbol);
  if (address == nullptr) {
    throw std::runtime_error(std::string("Missing official CANN symbol: ") + symbol);
  }
  return address;
}

}  // namespace vllm_ascend

// The shared EXEC_NPU_CMD intentionally searches custom operators first.
// Official QLI must resolve both stages from one CANN library. Only this
// operator uses the otherwise identical execution sequence below.
#define EXEC_OFFICIAL_QLI_CMD(aclnn_api, ...)                                 \
  do {                                                                        \
    static const auto getWorkspaceSizeFuncAddr =                              \
        vllm_ascend::GetOfficialQliFuncAddr(#aclnn_api "GetWorkspaceSize");    \
    static const auto opApiFuncAddr =                                          \
        vllm_ascend::GetOfficialQliFuncAddr(#aclnn_api);                      \
    static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal"); \
    static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal"); \
    static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");    \
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);           \
    uint64_t workspace_size = 0;                                              \
    uint64_t *workspace_size_addr = &workspace_size;                          \
    aclOpExecutor *executor = nullptr;                                        \
    aclOpExecutor **executor_addr = &executor;                                \
    InitHugeMemThreadLocal initMemFunc =                                      \
        reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);                \
    UnInitHugeMemThreadLocal unInitMemFunc =                                  \
        reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);            \
    if (initMemFunc) {                                                        \
      initMemFunc(nullptr, false);                                            \
    }                                                                         \
    auto converted_params =                                                   \
        ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);        \
    static auto getWorkspaceSizeFunc =                                        \
        ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);       \
    auto workspace_status = call(getWorkspaceSizeFunc, converted_params);     \
    TORCH_CHECK(workspace_status == 0,                                        \
                "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg()); \
    void *workspace_addr = nullptr;                                           \
    if (workspace_size != 0) {                                                \
      at::TensorOptions options =                                             \
          at::TensorOptions(torch_npu::utils::get_npu_device_type());         \
      auto workspace_tensor = at::empty({workspace_size}, options.dtype(kByte)); \
      workspace_addr = const_cast<void *>(workspace_tensor.storage().data()); \
    }                                                                         \
    auto acl_call = [converted_params, workspace_addr, workspace_size,        \
                     acl_stream, executor]() -> int {                         \
      typedef int (*OpApiFunc)(void *, uint64_t, aclOpExecutor *,             \
                               const aclrtStream);                            \
      OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);        \
      auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream); \
      TORCH_CHECK(api_ret == 0, "call " #aclnn_api " failed, detail:",       \
                  aclGetRecentErrMsg());                                      \
      ReleaseConvertTypes(converted_params);                                  \
      ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr); \
      if (releaseMemFunc) {                                                   \
        releaseMemFunc(nullptr, false);                                       \
      }                                                                       \
      return api_ret;                                                         \
    };                                                                        \
    at_npu::native::OpCommand cmd;                                            \
    cmd.Name(#aclnn_api);                                                     \
    cmd.SetCustomHandler(acl_call);                                           \
    cmd.Run();                                                                \
    if (unInitMemFunc) {                                                      \
      unInitMemFunc(nullptr, false);                                          \
    }                                                                         \
  } while (false)
