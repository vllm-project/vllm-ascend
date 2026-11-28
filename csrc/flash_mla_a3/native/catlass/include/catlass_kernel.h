/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHARED_LIB_CATLASS_KERNEL_H
#define SHARED_LIB_CATLASS_KERNEL_H

#include <acl/acl.h>
#include <vector>
#include <string>

namespace CatlassKernel {

struct MLAKernelInfo {
    std::vector<uint8_t *> inputAddr;
    std::vector<uint8_t *> outputAddr;
    int32_t batch;
    int32_t numHeads;
    int32_t embeddingSize;
    int32_t embeddingSizeRope;
    int32_t numTokens;
    int32_t kvHeads;
    int32_t numBlocks;
    int32_t blockSize;
    int32_t maxKvSeqlen;
    int32_t *kvSeqLen{nullptr};
    int32_t *qSeqLen{nullptr};
    int32_t dTypeKey;
    float softmaxScale;
    uint32_t maskMode = 0;
    uint32_t inputStride = 0;
};

void LaunchMLA(uint32_t blockNum, aclrtStream stream, MLAKernelInfo mlaKernelInfo);
std::vector<uint64_t> GetKVSplitKernel(uint32_t blockNum, aclrtStream stream, MLAKernelInfo mlaKernelInfo);
}

#endif // SHARED_LIB_CATLASS_KERNEL_H
