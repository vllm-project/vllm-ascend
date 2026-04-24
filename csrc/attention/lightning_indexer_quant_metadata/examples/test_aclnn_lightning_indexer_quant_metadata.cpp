/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#include <stdio.h>
#include <iostream>
#include <tuple>
#include <vector>
#include "acl/acl.h"
#include "../op_kernel_aicpu/lightning_indexer_quant_metadata.h"
#include "../op_api/aclnn_lightning_indexer_quant_metadata.h"

static const uint32_t batchSize = 4;
static const uint32_t seqSizeK = 10240;
static const uint32_t seqSizeQ = 3;
static const uint32_t numHeadsQ = 128;
static const uint32_t numHeadsK = 1;
static const uint32_t sparseMode = 3;
static const uint32_t preToken = 3;
static const uint32_t nextToken = 3;
static const uint32_t cmpRatio = 3;
static const bool isFD = false;
static std::string layoutQuery = "BSND";
static std::string layoutKV = "PA_ND";
static std::string socVersion = "ascend910B";
static const uint32_t aicCoreNum = 24;
static const uint32_t aivCoreNum = 48;

static const bool enableActLenQuery = true;
static const bool enableActLenKey = true;
static const std::vector<int32_t> actualSeqLengthsQuery = {3, 6, 9, 12};
static const std::vector<int32_t> actualSeqLengthsKey = {10240, 10240, 10240, 10240};
static const std::vector<int64_t> actualSeqLengthsQueryShape = {batchSize};
static const std::vector<int64_t> actualSeqLengthsKeyShape = {batchSize};
static const std::vector<int64_t> metadataShape = {optiling::LIQ_META_SIZE};
static const std::vector<int64_t> actualSeqLengthsQueryStride = {1};
static const std::vector<int64_t> actualSeqLengthsKeyStride = {1};
static const std::vector<int64_t> metadataStride = {1};
static const std::string dumpFile = "./dump";

std::tuple<aclTensor*, void*> CreateTensor(size_t size,  // in bytes
                                           std::vector<int64_t> shape,
                                           std::vector<int64_t> stride,
                                           aclDataType dType,
                                           const void* hostData = nullptr) {
  void* devicePtr = nullptr;
  auto ret = aclrtMalloc(&devicePtr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    printf("aclrtMalloc %d\n", ret);
    return {nullptr, nullptr};
  }

  aclTensor* tensor = aclCreateTensor(&shape[0], shape.size(), dType,
                                      &stride[0], 0, aclFormat::ACL_FORMAT_ND,
                                      &shape[0], shape.size(), devicePtr);
  if (tensor == nullptr) {
    aclrtFree(devicePtr);
    return {nullptr, nullptr};
  }

  if (hostData != nullptr) {
    aclrtMemcpy(devicePtr, size, hostData, size, ACL_MEMCPY_HOST_TO_DEVICE);
  }
  return {tensor, devicePtr};
}

static void DumpMeta(void *data) {
  optiling::detail::LiqMetaData *metaDataPtr =
      (optiling::detail::LiqMetaData *)data;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  aclError ret = 0;
  aclTensor* qSeqLenTensor = nullptr;
  void* qSeqLenDevPtr = nullptr;
  aclTensor* kvSeqLenTensor = nullptr;
  void* kvSeqLenDevPtr = nullptr;
  aclTensor* metadataTensor = nullptr;
  void* metadataDevPtr = nullptr;
  aclOpExecutor* executor = nullptr;
  uint64_t workspaceSize = 0;
  void* workspace = nullptr;

  ret = aclInit(nullptr);
  if (ret != ACL_SUCCESS) {
    printf("aclInit %d\n", ret);
    return -1;
  }

  ret = aclrtSetDevice(deviceId);
  if (ret != ACL_SUCCESS) {
    printf("aclrtSetDevice %d\n", ret);
    return -1;
  }

  ret = aclrtCreateStream(&stream);
  if (ret != ACL_SUCCESS) {
    printf("aclrtCreateStream %d\n", ret);
    return -1;
  }

  if (enableActLenQuery) {
    std::tie(qSeqLenTensor, qSeqLenDevPtr) = CreateTensor(
        actualSeqLengthsQuery.size() * sizeof(actualSeqLengthsQuery[0]), actualSeqLengthsQueryShape,
        actualSeqLengthsQueryStride, aclDataType::ACL_INT32, &actualSeqLengthsQuery[0]);
    if (qSeqLenTensor == nullptr) {
      return -1;
    }
  }

  if (enableActLenKey) {
    std::tie(kvSeqLenTensor, kvSeqLenDevPtr) = CreateTensor(
        actualSeqLengthsKey.size() * sizeof(actualSeqLengthsKey[0]), actualSeqLengthsKeyShape,
        actualSeqLengthsKeyStride, aclDataType::ACL_INT32, &actualSeqLengthsKey[0]);
    if (kvSeqLenTensor == nullptr) {
      return -1;
    }
  }


  std::tie(metadataTensor, metadataDevPtr) =
      CreateTensor(sizeof(int32_t) * optiling::LIQ_META_SIZE, metadataShape,
                   metadataStride, aclDataType::ACL_INT32);
  if (metadataTensor == nullptr) {
    return -1;
  }

  ret = aclnnLightningIndexerQuantMetadataGetWorkspaceSize(
        qSeqLenTensor, kvSeqLenTensor, aicCoreNum, aivCoreNum, batchSize,
        seqSizeQ, numHeadsQ, seqSizeK, numHeadsK, &layoutQuery[0], &layoutKV[0],
        sparseMode, &socVersion[0], isFD, preToken, nextToken, cmpRatio, metadataTensor, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    printf("aclnnLightningIndexerQuantMetadataGetWorkspaceSize %d\n",
           ret);
    return -1;
  }

  ret = aclnnLightningIndexerQuantMetadata(workspace, workspaceSize,
                                                 executor, stream);
  if (ret != ACL_SUCCESS) {
    printf("aclnnLightningIndexerQuantMetadata %d\n", ret);
    return -1;
  }

  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    printf("aclrtSynchronizeStream %d\n", ret);
    return -1;
  }

  std::vector<int32_t> metdataHost(optiling::LIQ_META_SIZE);
  ret = aclrtMemcpy(metdataHost.data(),
                    metdataHost.size() * sizeof(metdataHost[0]), metadataDevPtr,
                    optiling::LIQ_META_SIZE * sizeof(int32_t),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != ACL_SUCCESS) {
    printf("aclrtMemcpy %d\n", ret);
    return -1;
  }

  DumpMeta(&metdataHost[0]);

  aclDestroyTensor(qSeqLenTensor);
  aclDestroyTensor(kvSeqLenTensor);
  aclDestroyTensor(metadataTensor);

  aclrtFree(qSeqLenDevPtr);
  aclrtFree(kvSeqLenDevPtr);
  aclrtFree(metadataDevPtr);
  aclrtFree(workspace);

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
