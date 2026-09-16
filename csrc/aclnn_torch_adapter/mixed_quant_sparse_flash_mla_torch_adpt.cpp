/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Adapted from ops-transformer c6240b268a6818ff343c721b8508217d4cfb9812.
#include "mixed_quant_sparse_flash_mla_torch_adpt.h"

#include <c10/core/DeviceGuard.h>
#include "op_api_common.h"

namespace vllm_ascend::mqsmla {

// Kept local so generic operators retain their normal Torch-to-ACL dtype mapping.
// ConvertTypes discovers this overload by ADL on Fp8TensorWrapper.
struct Fp8TensorWrapper {
    const at::Tensor &tensor;
    aclDataType dtype;
};

inline Fp8TensorWrapper MakeFp8Wrapper(const at::Tensor &tensor, aclDataType dtype)
{
    return {tensor, dtype};
}

inline aclTensor *ConvertType(const Fp8TensorWrapper &wrapper)
{
    static const auto createTensor = GET_OP_API_FUNC(aclCreateTensor);
    TORCH_CHECK(createTensor != nullptr, "aclCreateTensor was not found");
    const auto &tensor = wrapper.tensor;
    if (!tensor.defined()) {
        return nullptr;
    }
    TORCH_CHECK(wrapper.dtype == ACL_FLOAT8_E4M3FN, "MQSMLA KV descriptor must be FP8 E4M3");
    TORCH_CHECK(tensor.itemsize() > 0, "tensor item size must be positive");
    c10::SmallVector<int64_t, 8> storageDims;
    aclFormat format = ACL_FORMAT_ND;
    if (!IsOpInputBaseFormat(tensor)) {
        const auto *storage = vllm_ascend::NPUBridge::GetNpuStorageImpl(tensor);
        format = storage->npu_desc_.npu_format_;
        storageDims.assign(storage->npu_desc_.storage_sizes_.begin(), storage->npu_desc_.storage_sizes_.end());
    } else {
        switch (tensor.dim()) {
            case 3: format = ACL_FORMAT_NCL; break;
            case 4: format = ACL_FORMAT_NCHW; break;
            case 5: format = ACL_FORMAT_NCDHW; break;
            default: break;
        }
        storageDims.push_back(tensor.storage().nbytes() / tensor.itemsize());
    }
    // Use the original bytes, strides and offset. No cast/view/copy is performed.
    return createTensor(tensor.sizes().data(), tensor.dim(), wrapper.dtype,
                        tensor.strides().data(), tensor.storage_offset(), format,
                        storageDims.data(), storageDims.size(), const_cast<void *>(tensor.storage().data()));
}

static_assert(std::is_same_v<decltype(::ConvertTypes(std::declval<Fp8TensorWrapper &>())),
                             std::tuple<aclTensor *>>,
              "MQSMLA FP8 wrapper must convert to an ACL tensor descriptor");

inline c10::optional<at::Tensor> GetValidMetadataTensor(
    const c10::optional<at::Tensor> &tensor, const at::Device &device)
{
    return tensor.has_value() ? tensor : at::empty({0}, at::TensorOptions().dtype(at::kInt).device(device));
}

// Keep workspace alive until OpCommand has submitted the ACLNN call.
#define MQSMLA_EXEC_NPU_CMD(aclnn_api, ...)                                          \
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
    at::Tensor workspace_tensor;                                           \
    void *workspace_addr = nullptr;                                           \
    if (workspace_size != 0) {                                                \
      at::TensorOptions options =                                             \
          at::TensorOptions(torch_npu::utils::get_npu_device_type());         \
      workspace_tensor =                                                 \
          at::empty({workspace_size}, options.dtype(kByte));                  \
      workspace_addr = const_cast<void *>(workspace_tensor.storage().data()); \
    }                                                                         \
    auto acl_call = [converted_params, workspace_addr, workspace_size,        \
                     acl_stream, executor]() -> int {                         \
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
    at_npu::native::OpCommand cmd;                                            \
    cmd.Name(#aclnn_api);                                                     \
    cmd.SetCustomHandler(acl_call);                                           \
    cmd.Run();                                                                \
    if (unInitMemFunc) {                                                      \
      unInitMemFunc(nullptr, false);                                          \
    }                                                                         \
  } while (false)


at::Tensor MixedQuantSparseFlashMlaMetadata(
    int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim, int64_t quantMode,
    const c10::optional<at::Tensor> &cuSeqlensQ, const c10::optional<at::Tensor> &cuSeqlensOriKv,
    const c10::optional<at::Tensor> &cuSeqlensCmpKv, const c10::optional<at::Tensor> &sequsedQ,
    const c10::optional<at::Tensor> &sequsedOriKv, const c10::optional<at::Tensor> &sequsedCmpKv,
    const c10::optional<at::Tensor> &cmpResidualKv, const c10::optional<at::Tensor> &oriTopkLength,
    const c10::optional<at::Tensor> &cmpTopkLength, c10::optional<int64_t> batchSize, c10::optional<int64_t> maxSeqlenQ, c10::optional<int64_t> maxSeqlenOriKv,
    c10::optional<int64_t> maxSeqlenCmpKv, c10::optional<int64_t> oriTopk, c10::optional<int64_t> cmpTopk, c10::optional<int64_t> ropeHeadDim, c10::optional<int64_t> cmpRatio,
    c10::optional<int64_t> oriMaskMode, c10::optional<int64_t> cmpMaskMode, c10::optional<int64_t> oriWinLeft, c10::optional<int64_t> oriWinRight, c10::optional<c10::string_view> layoutQ,
    c10::optional<c10::string_view> layoutKv, c10::optional<bool> hasOriKv, c10::optional<bool> hasCmpKv)
{
    at::Device outputDevice = at::Device(c10::DeviceType::PrivateUse1);
    if (cuSeqlensQ.has_value()) {
        outputDevice = cuSeqlensQ.value().device();
    } else if (cuSeqlensOriKv.has_value()) {
        outputDevice = cuSeqlensOriKv.value().device();
    } else if (cuSeqlensCmpKv.has_value()) {
        outputDevice = cuSeqlensCmpKv.value().device();
    } else if (sequsedQ.has_value()) {
        outputDevice = sequsedQ.value().device();
    } else if (sequsedOriKv.has_value()) {
        outputDevice = sequsedOriKv.value().device();
    } else if (sequsedCmpKv.has_value()) {
        outputDevice = sequsedCmpKv.value().device();
    } else if (cmpResidualKv.has_value()) {
        outputDevice = cmpResidualKv.value().device();
    } else if (oriTopkLength.has_value()) {
        outputDevice = oriTopkLength.value().device();
    } else if (cmpTopkLength.has_value()) {
        outputDevice = cmpTopkLength.value().device();
    }

    const c10::OptionalDeviceGuard deviceGuard(outputDevice);
    at::Tensor output = torch::empty({MQSMLA_METADATA_SIZE}, torch::dtype(torch::kInt32).device(outputDevice));
    auto cuSeqlensQVal = GetValidMetadataTensor(cuSeqlensQ, outputDevice);
    auto cuSeqlensOriKvVal = GetValidMetadataTensor(cuSeqlensOriKv, outputDevice);
    auto cuSeqlensCmpKvVal = GetValidMetadataTensor(cuSeqlensCmpKv, outputDevice);
    auto sequsedQVal = GetValidMetadataTensor(sequsedQ, outputDevice);
    auto sequsedOriKvVal = GetValidMetadataTensor(sequsedOriKv, outputDevice);
    auto sequsedCmpKvVal = GetValidMetadataTensor(sequsedCmpKv, outputDevice);
    auto cmpResidualKvVal = GetValidMetadataTensor(cmpResidualKv, outputDevice);
    auto oriTopkLengthVal = GetValidMetadataTensor(oriTopkLength, outputDevice);
    auto cmpTopkLengthVal = GetValidMetadataTensor(cmpTopkLength, outputDevice);

    // convert str
    std::string layoutQStr = std::string(layoutQ.value_or("BSND"));
    std::string layoutKvStr = std::string(layoutKv.value_or("BSND"));
    char *layoutQPtr = const_cast<char *>(layoutQStr.c_str());
    char *layoutKvPtr = const_cast<char *>(layoutKvStr.c_str());

    auto batchSizeValue = batchSize.value_or(0);
    auto maxSeqlenQValue = maxSeqlenQ.value_or(0);
    auto maxSeqlenOriKvValue = maxSeqlenOriKv.value_or(0);
    auto maxSeqlenCmpKvValue = maxSeqlenCmpKv.value_or(0);
    auto oriTopkValue = oriTopk.value_or(0);
    auto cmpTopkValue = cmpTopk.value_or(0);
    auto ropeHeadDimValue = ropeHeadDim.value_or(64);
    auto cmpRatioValue = cmpRatio.value_or(1);
    auto oriMaskModeValue = oriMaskMode.value_or(0);
    auto cmpMaskModeValue = cmpMaskMode.value_or(0);
    auto oriWinLeftValue = oriWinLeft.value_or(-1);
    auto oriWinRightValue = oriWinRight.value_or(-1);
    auto hasOriKvValue = hasOriKv.value_or(true);
    auto hasCmpKvValue = hasCmpKv.value_or(true);
    MQSMLA_EXEC_NPU_CMD(aclnnMixedQuantSparseFlashMlaMetadata, cuSeqlensQVal, cuSeqlensOriKvVal, cuSeqlensCmpKvVal, sequsedQVal,
              sequsedOriKvVal, sequsedCmpKvVal, cmpResidualKvVal, oriTopkLengthVal, cmpTopkLengthVal, numHeadsQ,
              numHeadsKv, headDim, quantMode, batchSizeValue, maxSeqlenQValue, maxSeqlenOriKvValue, maxSeqlenCmpKvValue, oriTopkValue, cmpTopkValue,
              ropeHeadDimValue, cmpRatioValue, oriMaskModeValue, cmpMaskModeValue, oriWinLeftValue, oriWinRightValue, layoutQPtr, layoutKvPtr,
              hasOriKvValue, hasCmpKvValue, output);
    return output;
}

std::tuple<at::Tensor, at::Tensor> MixedQuantSparseFlashMla(
    const at::Tensor &q, const c10::optional<at::Tensor> &oriKv, const c10::optional<at::Tensor> &cmpKv,
    const c10::optional<at::Tensor> &oriSparseIndices, const c10::optional<at::Tensor> &cmpSparseIndices,
    const c10::optional<at::Tensor> &oriBlockTable, const c10::optional<at::Tensor> &cmpBlockTable,
    const c10::optional<at::Tensor> &cuSeqlensQ, const c10::optional<at::Tensor> &cuSeqlensOriKv,
    const c10::optional<at::Tensor> &cuSeqlensCmpKv, const c10::optional<at::Tensor> &sequsedQ,
    const c10::optional<at::Tensor> &sequsedOriKv, const c10::optional<at::Tensor> &sequsedCmpKv,
    const c10::optional<at::Tensor> &cmpResidualKv, const c10::optional<at::Tensor> &oriTopkLength,
    const c10::optional<at::Tensor> &cmpTopkLength, const c10::optional<at::Tensor> &sinks,
    const c10::optional<at::Tensor> &metadata, int64_t quantMode, int64_t ropeHeadDim, double softmaxScale,
    int64_t cmpRatio, int64_t oriMaskMode, int64_t cmpMaskMode, int64_t oriWinLeft, int64_t oriWinRight,
    c10::string_view layoutQ, c10::string_view layoutKv, int64_t topkValueMode, bool returnSoftmaxLse,
    c10::optional<int64_t> keyDtype, c10::optional<int64_t> valueDtype)
{
    const c10::OptionalDeviceGuard deviceGuard(q.device());
    TORCH_CHECK(q.numel() > 0, "Tensor query is empty.");

    std::string layoutQStr = std::string(layoutQ);
    std::string layoutKvStr = std::string(layoutKv);
    // convert str
    char *layoutQPtr = const_cast<char *>(layoutQStr.c_str());
    char *layoutKvPtr = const_cast<char *>(layoutKvStr.c_str());

    // construct the atten_out tensor
    std::tuple<at::Tensor, at::Tensor> mixedQuantSparseFlashMlaAttenOut =
        ConstructOutputs(q, oriKv, layoutQStr, layoutKvStr, returnSoftmaxLse);
    at::Tensor attenOut = std::get<0>(mixedQuantSparseFlashMlaAttenOut);
    at::Tensor softmaxLse = std::get<1>(mixedQuantSparseFlashMlaAttenOut);

    // keyDtype/valueDtype are retained for schema compatibility; c624 always
    // describes KV bytes as FP8 E4M3, including uint8 carriers.
    (void)keyDtype;
    (void)valueDtype;
    at::Tensor nullTensor;
    auto oriKvValue = oriKv.has_value() ? oriKv.value() : nullTensor;
    auto cmpKvValue = cmpKv.has_value() ? cmpKv.value() : nullTensor;
    Fp8TensorWrapper oriKvWrapper = MakeFp8Wrapper(oriKvValue, ACL_FLOAT8_E4M3FN);
    Fp8TensorWrapper cmpKvWrapper = MakeFp8Wrapper(cmpKvValue, ACL_FLOAT8_E4M3FN);
    MQSMLA_EXEC_NPU_CMD(aclnnMixedQuantSparseFlashMla, q, oriKvWrapper, cmpKvWrapper, oriSparseIndices, cmpSparseIndices,
              oriBlockTable, cmpBlockTable, cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, sequsedQ, sequsedOriKv,
              sequsedCmpKv, cmpResidualKv, oriTopkLength, cmpTopkLength, sinks, metadata, quantMode, ropeHeadDim,
              softmaxScale, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight, layoutQPtr, layoutKvPtr,
              topkValueMode, returnSoftmaxLse, attenOut, softmaxLse);
    return std::tuple<at::Tensor, at::Tensor>(attenOut, softmaxLse);
}

#undef MQSMLA_EXEC_NPU_CMD

}  // namespace vllm_ascend::mqsmla
