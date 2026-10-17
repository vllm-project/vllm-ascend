/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_sparse_flash_attention.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_sparse_flash_attention.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true :(false))

#define LOG_PRINT(message, ...)     \
  do {                              \
    (void)printf(message, ##__VA_ARGS__); \
  } while (0)
 
int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int32_t Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclInit failed. ERROR: %d\n", ret); 
    return ret;
  }
  ret = aclrtSetDevice(deviceId);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); 
    return ret;
  }
  ret = aclrtCreateStream(stream);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); 
    return ret;
  }
  return 0;
}

template <typename T>
int32_t CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); 
    return ret;
  }

  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (!CHECK_RET(ret == ACL_SUCCESS)) { 
    LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); 
    return ret;
  }
 
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
 
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

struct TensorResources {
    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* valueDeviceAddr = nullptr;
    void* sparseIndicesDeviceAddr = nullptr;
    void* attentionOutDeviceAddr = nullptr;
    void* softmaxMaxDeviceAddr = nullptr;
    void* softmaxSumDeviceAddr = nullptr;
    void* queryRopeDeviceAddr = nullptr;
    void* keyRopeDeviceAddr = nullptr;

    aclTensor* queryTensor = nullptr;
    aclTensor* keyTensor = nullptr;
    aclTensor* valueTensor = nullptr;
    aclTensor* sparseIndicesTensor = nullptr;
    aclTensor* attentionOutTensor = nullptr;
    aclTensor* softmaxMaxTensor = nullptr;
    aclTensor* softmaxSumTensor = nullptr;
    aclTensor* queryRopeTensor = nullptr;
    aclTensor* keyRopeTensor = nullptr; 
};

int32_t InitializeTensors(TensorResources& resources) {
    std::vector<int64_t> queryShape = {1, 2, 1, 512};
    std::vector<int64_t> keyShape = {1, 2, 1, 512};
    std::vector<int64_t> valueShape = {1, 2, 1, 512};
    std::vector<int64_t> sparseIndicesShape = {1, 2, 1, 2};
    std::vector<int64_t> attentionOutShape = {1, 2, 1, 512};
    std::vector<int64_t> softmaxMaxShape = {1, 2, 1, 16};
    std::vector<int64_t> softmaxSumShape = {1, 2, 1, 16};
    std::vector<int64_t> queryRopeShape = {1, 2, 1, 64};
    std::vector<int64_t> keyRopeShape = {1, 2, 1, 64};

    int64_t queryShapeSize = GetShapeSize(queryShape);
    int64_t keyShapeSize = GetShapeSize(keyShape);
    int64_t valueShapeSize = GetShapeSize(valueShape);
    int64_t sparseIndicesShapeSize =  GetShapeSize(sparseIndicesShape);
    int64_t attentionOutShapeSize = GetShapeSize(attentionOutShape);
    int64_t softmaxMaxShapeSize = GetShapeSize(softmaxMaxShape);
    int64_t softmaxSumShapeSize = GetShapeSize(softmaxSumShape);
    int64_t queryRopeShapeSize = GetShapeSize(queryRopeShape);
    int64_t keyRopeShapeSize = GetShapeSize(keyRopeShape);

    std::vector<float> queryHostData(queryShapeSize, 1);
    std::vector<float> keyHostData(keyShapeSize, 1);
    std::vector<float> valueHostData(valueShapeSize, 1);
    std::vector<int32_t> sparseIndicesHostData(sparseIndicesShapeSize, 1);
    std::vector<float> attentionOutHostData(attentionOutShapeSize, 1);
    std::vector<float> softmaxMaxHostData(softmaxMaxShapeSize, 1);
    std::vector<float> softmaxSumHostData(softmaxSumShapeSize, 1);
    std::vector<float> queryRopeHostData(queryRopeShapeSize, 1);
    std::vector<float> keyRopeHostData(keyRopeShapeSize, 1);

    // Create query aclTensor.
    int32_t ret = CreateAclTensor(queryHostData, queryShape, &resources.queryDeviceAddr, 
                             aclDataType::ACL_FLOAT16, &resources.queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create key aclTensor.
    ret = CreateAclTensor(keyHostData, keyShape, &resources.keyDeviceAddr, 
                         aclDataType::ACL_FLOAT16, &resources.keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create value aclTensor.
    ret = CreateAclTensor(valueHostData, valueShape, &resources.valueDeviceAddr, 
                         aclDataType::ACL_FLOAT16, &resources.valueTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create sparseIndices aclTensor.
    ret = CreateAclTensor(sparseIndicesHostData, sparseIndicesShape, &resources.sparseIndicesDeviceAddr, 
                         aclDataType::ACL_INT32, &resources.sparseIndicesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create queryRope aclTensor.
    ret = CreateAclTensor(queryRopeHostData, queryRopeShape, &resources.queryRopeDeviceAddr, 
                         aclDataType::ACL_FLOAT16, &resources.queryRopeTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create keyRope aclTensor.
    ret = CreateAclTensor(keyRopeHostData, keyRopeShape, &resources.keyRopeDeviceAddr, 
                         aclDataType::ACL_FLOAT16, &resources.keyRopeTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create attention_out aclTensor.
    ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &resources.attentionOutDeviceAddr, 
                         aclDataType::ACL_FLOAT16, &resources.attentionOutTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create softmax_max aclTensor.
    ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &resources.softmaxMaxDeviceAddr, 
                         aclDataType::ACL_FLOAT, &resources.softmaxMaxTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create softmax_sum aclTensor.
    ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &resources.softmaxSumDeviceAddr, 
                         aclDataType::ACL_FLOAT, &resources.softmaxSumTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    return ACL_SUCCESS;
}

int32_t ExecuteSparseFlashAttention(TensorResources& resources, aclrtStream stream, 
                              void** workspaceAddr, uint64_t* workspaceSize) {
    int64_t d = 2;
    double scaleValue = 1 / sqrt(d);
    int64_t sparseBlockSize = 64;
    constexpr const char layerOutStr[] = "BSND";
    constexpr size_t layerOutLen = sizeof(layerOutStr);
    char layoutQuery[layerOutLen];
    char layoutKv[layerOutLen];
    errno_t memcpyRet = memcpy_s(layoutQuery, sizeof(layoutQuery), layerOutStr, layerOutLen);
    if (memcpyRet != 0) {
        LOG_PRINT("memcpy_s layoutQuery failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    memcpyRet = memcpy_s(layoutKv, sizeof(layoutKv), layerOutStr, layerOutLen);
    if (memcpyRet != 0) {
        LOG_PRINT("memcpy_s layoutKv failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    int64_t sparseMode = 3;
    int64_t preTokens = 9223372036854775807;
    int64_t nextTokens = 9223372036854775807;
    int64_t attentionMode = 2;
    bool returnSoftmaxLse = false;
    aclOpExecutor* executor;

    int32_t ret = aclnnSparseFlashAttentionGetWorkspaceSize(resources.queryTensor, resources.keyTensor, resources.valueTensor, resources.sparseIndicesTensor, nullptr, nullptr, nullptr, resources.queryRopeTensor, resources.keyRopeTensor,
                                                    scaleValue, sparseBlockSize, layoutQuery, layoutKv, sparseMode, preTokens,
                                                    nextTokens, attentionMode, returnSoftmaxLse, resources.attentionOutTensor, resources.softmaxMaxTensor, resources.softmaxSumTensor, workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnSparseFlashAttentionGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnSparseFlashAttention(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnSparseFlashAttention failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

int32_t PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<aclFloat16> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
  }
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
  }
  return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr, 
                     aclrtStream stream, int32_t deviceId) {
    if (resources.queryTensor) {
      aclDestroyTensor(resources.queryTensor);
    }
    if (resources.keyTensor) {
      aclDestroyTensor(resources.keyTensor);
    }
    if (resources.valueTensor) {
      aclDestroyTensor(resources.valueTensor);
    }
    if (resources.sparseIndicesTensor) {
      aclDestroyTensor(resources.sparseIndicesTensor);
    }
    if (resources.attentionOutTensor) {
      aclDestroyTensor(resources.attentionOutTensor);
    }
    if (resources.softmaxMaxTensor) {
      aclDestroyTensor(resources.softmaxMaxTensor);
    }
    if (resources.softmaxSumTensor) {
      aclDestroyTensor(resources.softmaxSumTensor);
    }
    if (resources.queryRopeTensor) {
      aclDestroyTensor(resources.queryRopeTensor);
    }
    if (resources.keyRopeTensor) {
      aclDestroyTensor(resources.keyRopeTensor);
    }

    if (resources.queryDeviceAddr) {
      aclrtFree(resources.queryDeviceAddr);
    }
    if (resources.keyDeviceAddr) {
      aclrtFree(resources.keyDeviceAddr);
    }
    if (resources.valueDeviceAddr) {
      aclrtFree(resources.valueDeviceAddr);
    }
    if (resources.sparseIndicesDeviceAddr) {
      aclrtFree(resources.sparseIndicesDeviceAddr);
    }
    if (resources.attentionOutDeviceAddr) {
      aclrtFree(resources.attentionOutDeviceAddr);
    }
    if (resources.softmaxMaxDeviceAddr) {
      aclrtFree(resources.softmaxMaxDeviceAddr);
    }
    if (resources.softmaxSumDeviceAddr) {
      aclrtFree(resources.softmaxSumDeviceAddr);
    }
    if (resources.queryRopeDeviceAddr) {
      aclrtFree(resources.queryRopeDeviceAddr);
    }
    
    if (resources.keyRopeDeviceAddr) {
      aclrtFree(resources.keyRopeDeviceAddr);
    }

    if (workspaceAddr) {
      aclrtFree(workspaceAddr);
    }
    if (stream) {
      aclrtDestroyStream(stream);
    }
    
    aclrtResetDevice(deviceId);
    aclFinalize();
}

} // namespace

int32_t main() 
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    TensorResources resources = {};
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<int64_t> attentionOutShape = {1, 2, 1, 16};
    std::vector<int64_t> softmaxMaxShape = {1, 2, 1, 16};
    std::vector<int64_t> softmaxSumShape = {1, 2, 1, 16}; 
    int32_t ret = ACL_SUCCESS;

    // 1. Initialize device and stream
    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    // 2. Initialize tensors
    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 3. Execute the operation
    ret = ExecuteSparseFlashAttention(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 4. Synchronize stream
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 5. Process results
    printf("-----------attentionOut输出-----------\n");
    PrintOutResult(attentionOutShape, &resources.attentionOutDeviceAddr);
    printf("-----------softmaxMax输出-----------\n");
    PrintOutResult(softmaxMaxShape, &resources.softmaxMaxDeviceAddr);
    printf("-----------softmaxSum输出-----------\n");
    PrintOutResult(softmaxSumShape, &resources.softmaxSumDeviceAddr);
    // 6. Cleanup resources
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}
