// Ordered submission ported from the original A5 implementation.
#ifndef LIM_QUANT_ARCH35_ORDERED_SUBMISSION_H
#define LIM_QUANT_ARCH35_ORDERED_SUBMISSION_H

#include <dlfcn.h>
#include <tuple>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "../../aclnn_torch_adapter/op_api_common.h"

namespace lim_quant_arch35_submission {
using AclUseStreamResFunc = aclError (*)(aclrtStream);

inline AclUseStreamResFunc GetUseStreamResFunc() {
  static void *libacl_handle = dlopen("libascendcl.so", RTLD_NOW);
  static auto use_stream_res =
      libacl_handle == nullptr
          ? nullptr
          : reinterpret_cast<AclUseStreamResFunc>(
                dlsym(libacl_handle, "aclrtUseStreamResInCurrentThread"));
  return use_stream_res;
}

inline void UseStreamResIfNeeded(aclrtStream stream, bool enqueue) {
  const bool needed = enqueue ? c10_npu::check_enqueue_need_use(stream)
                              : c10_npu::check_dequeue_need_use(stream);
  if (!needed) {
    return;
  }
  auto use_stream_res = GetUseStreamResFunc();
  TORCH_CHECK(
      use_stream_res != nullptr,
      "aclrtUseStreamResInCurrentThread is unavailable.");
  const aclError ret = use_stream_res(stream);
  TORCH_CHECK(
      ret == ACL_SUCCESS,
      "aclrtUseStreamResInCurrentThread failed, ret=", ret,
      ", detail:", aclGetRecentErrMsg());
}
} // namespace lim_quant_arch35_submission

// Queue-safe aclnn launch for stateful GLM decode operators.
#define LIM_QUANT_EXEC_NPU_CMD_ORDERED(aclnn_api, tensor_keepalive, ...)                \
  do {                                                                        \
    static const auto getWorkspaceSizeFuncAddr =                              \
        GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");                      \
    static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);           \
    static const auto initMemAddr =                                           \
        GetOpApiFuncAddr("InitHugeMemThreadLocal");                           \
    static const auto unInitMemAddr =                                         \
        GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                         \
    static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");    \
    TORCH_CHECK(                                                              \
        getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,      \
        #aclnn_api, " or ", #aclnn_api "GetWorkspaceSize", " not in ",        \
        GetOpApiLibName(), ", or ", GetOpApiLibName(), "not found.");         \
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);           \
    lim_quant_arch35_submission::UseStreamResIfNeeded(acl_stream, true);                                   \
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
    at::Tensor workspace_tensor;                                              \
    void *workspace_addr = nullptr;                                           \
    if (workspace_size != 0) {                                                \
      at::TensorOptions options =                                             \
          at::TensorOptions(torch_npu::utils::get_npu_device_type());         \
      workspace_tensor = at::empty(                                           \
          {static_cast<int64_t>(workspace_size)}, options.dtype(at::kByte));  \
      workspace_addr =                                                        \
          const_cast<void *>(workspace_tensor.storage().data());              \
    }                                                                         \
    auto ordered_tensor_keepalive = tensor_keepalive;                         \
    auto acl_call = [converted_params, workspace_tensor, workspace_addr,      \
                     workspace_size, acl_stream, executor,                    \
                     ordered_tensor_keepalive]() -> int {                     \
      (void)workspace_tensor;                                                 \
      (void)ordered_tensor_keepalive;                                         \
      lim_quant_arch35_submission::UseStreamResIfNeeded(acl_stream, false);                                \
      typedef int (*OpApiFunc)(void *, uint64_t, aclOpExecutor *,             \
                               const aclrtStream);                            \
      OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);       \
      auto api_ret =                                                          \
          opApiFunc(workspace_addr, workspace_size, executor, acl_stream);    \
      TORCH_CHECK(api_ret == 0, "call " #aclnn_api " failed, detail:",        \
                  aclGetRecentErrMsg());                                      \
      ReleaseConvertTypes(converted_params);                                  \
      ReleaseHugeMem releaseMemFunc =                                         \
          reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);                   \
      if (releaseMemFunc) {                                                   \
        releaseMemFunc(nullptr, false);                                       \
      }                                                                       \
      return api_ret;                                                         \
    };                                                                        \
    at_npu::native::OpCommand::RunOpApiV2(#aclnn_api, acl_call);              \
    if (unInitMemFunc) {                                                      \
      unInitMemFunc(nullptr, false);                                          \
    }                                                                         \
  } while (false)

#endif
