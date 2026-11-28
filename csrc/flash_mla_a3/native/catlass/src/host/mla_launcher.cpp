/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <acl/acl.h>
#include "catlass_kernel.h"
#include "helper.hpp"
#include "mla_tiling.cpp"

#include "kernel/mla_kernel.cpp"
#include "kernel/mla_kernel_tp1_spec.cpp"

namespace CatlassKernel {

const uint32_t TILINGKEY_FP16 = 0;
const uint32_t TILINGKEY_BFP16 = 1;
const uint32_t TILINGKEY_TP1_SPEC_FP16 = 4;
const uint32_t TILINGKEY_TP1_SPEC_BFP16 = 5;


MLATiling::MLAInfo CreateMlaInfo(MLAKernelInfo& mlaKernelInfo){
    MLATiling::MLAInfo mlaInfo;

    mlaInfo.numTokens = mlaKernelInfo.numTokens;
    mlaInfo.numHeads = mlaKernelInfo.numHeads;
    mlaInfo.embeddingSize = mlaKernelInfo.embeddingSize;
    mlaInfo.embeddingSizeRope = mlaKernelInfo.embeddingSizeRope;
    mlaInfo.numBlocks = mlaKernelInfo.numBlocks;
    mlaInfo.blockSize = mlaKernelInfo.blockSize;
    mlaInfo.maxKvSeqlen = mlaKernelInfo.maxKvSeqlen;
    mlaInfo.kvHeads = mlaKernelInfo.kvHeads;
    mlaInfo.batch = mlaKernelInfo.batch;
    mlaInfo.qSeqLen = mlaKernelInfo.qSeqLen;
    mlaInfo.kvSeqLen = mlaKernelInfo.kvSeqLen;
    mlaInfo.maskType = static_cast<MLATiling::MaskType>(mlaKernelInfo.maskMode);
    mlaInfo.inputStride = mlaKernelInfo.inputStride;

    return mlaInfo;
}


struct InputTensors{
    uint8_t *q, *qRope, *k, *kRope, *blockTable, *s, *p;
};


struct OutputTensors{
    uint8_t *o, *oTmp, *globalO, *l, *oCoreTmp;
};


void LaunchMLA(uint32_t blockNum, aclrtStream stream, MLAKernelInfo mlaKernelInfo)
{
    MLATiling::MLAInfo mlaInfo = CreateMlaInfo(mlaKernelInfo);

    // get MLA tiling on host
    int32_t specStraKey = (mlaInfo.numHeads == MLATiling::NUM128) ? 1 : 0;
    uint32_t tilingSize = (MLATiling::TILING_HEAD_SIZE + mlaInfo.batch * MLATiling::TILING_PARA_SIZE) * sizeof(int32_t);
    if (specStraKey) {
        tilingSize = (MLATiling::TILING_HEAD_SIZE + mlaInfo.numTokens * MLATiling::TILING_PARA_SIZE) * sizeof(int32_t);
    }

    void *tilingHost = nullptr;
    ACL_CHECK(aclrtMallocHost(&tilingHost, tilingSize));
    MLATiling::GetMLATilingParam(mlaInfo, blockNum, (uint32_t *)tilingHost, mlaKernelInfo.softmaxScale);

    // copy tiling info to device
    uint8_t *tilingDevice;
    ACL_CHECK(aclrtMalloc((void **)(&tilingDevice), tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // prepare for device kernel launch
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    // 3 bits for tilingKey(specStraKey : 1, dTypeKey : 2)
    int32_t tilingKey = (specStraKey << MLATiling::NUM2) + mlaKernelInfo.dTypeKey;

    std::vector<uint8_t *> inAddrs = mlaKernelInfo.inputAddr;
    InputTensors inputs{inAddrs[0], inAddrs[1], inAddrs[2], inAddrs[3], inAddrs[4], inAddrs[5], inAddrs[6]};

    std::vector<uint8_t *> outAddrs = mlaKernelInfo.outputAddr;
    OutputTensors outputs{outAddrs[0], outAddrs[1], outAddrs[2], outAddrs[3], outAddrs[4]};

    // use Tp1Spec kernel to get better performance when numHeads = 128
    switch (tilingKey) {
        case TILINGKEY_FP16:
            MLAFp16<<<blockNum, nullptr, stream>>>(fftsAddr, inputs.q, inputs.qRope, inputs.k, inputs.kRope,
                                                inputs.blockTable, outputs.o, inputs.s, inputs.p, outputs.oTmp,
                                                outputs.globalO, outputs.oCoreTmp, outputs.l, tilingDevice);
            break;
        case TILINGKEY_BFP16:
            MLABf16<<<blockNum, nullptr, stream>>>(fftsAddr, inputs.q, inputs.qRope, inputs.k, inputs.kRope,
                                                inputs.blockTable, outputs.o, inputs.s, inputs.p, outputs.oTmp,
                                                outputs.globalO, outputs.oCoreTmp, outputs.l, tilingDevice);
            break;
        case TILINGKEY_TP1_SPEC_FP16:
            MLATp1SpecFp16<<<blockNum, nullptr, stream>>>(fftsAddr, inputs.q, inputs.qRope, inputs.k, inputs.kRope,
                                                inputs.blockTable, outputs.o, inputs.s, inputs.p, outputs.oTmp,
                                                outputs.globalO, outputs.oCoreTmp, outputs.l, tilingDevice);
            break;
        case TILINGKEY_TP1_SPEC_BFP16:
            MLATp1SpecBf16<<<blockNum, nullptr, stream>>>(fftsAddr, inputs.q, inputs.qRope, inputs.k, inputs.kRope,
                                                inputs.blockTable, outputs.o, inputs.s, inputs.p, outputs.oTmp,
                                                outputs.globalO, outputs.oCoreTmp, outputs.l, tilingDevice);
            break;
        default: break;
    }
    // memory clean-up
    aclrtFreeHost(tilingHost);
    aclrtFree(tilingDevice);
}

std::vector<uint64_t> GetKVSplitKernel(uint32_t blockNum, aclrtStream stream, MLAKernelInfo mlaKernelInfo)
{
    MLATiling::MLAInfo mlaInfo = CreateMlaInfo(mlaKernelInfo);

    // for convenience later
    int32_t batch = mlaInfo.batch;
    int32_t numTokens = mlaInfo.numTokens;
    int32_t numHeads = mlaInfo.numHeads;
    int32_t embeddingSize = mlaInfo.embeddingSize;

    // get MLA tiling on host
    int32_t specStraKey = (numHeads == MLATiling::NUM128) ? 1 : 0;
    uint32_t tilingSize = (MLATiling::TILING_HEAD_SIZE + batch * MLATiling::TILING_PARA_SIZE) * sizeof(int32_t);
    if (specStraKey) {
        tilingSize = (MLATiling::TILING_HEAD_SIZE + numTokens * MLATiling::TILING_PARA_SIZE) * sizeof(int32_t);
    }

    void *tilingHost = nullptr;
    ACL_CHECK(aclrtMallocHost(&tilingHost, tilingSize));
    MLATiling::GetMLATilingParam(mlaInfo, blockNum, (uint32_t *)tilingHost, 0.0);

    uint64_t aicCoreNum = (uint64_t)blockNum;
    uint64_t kvSplitCoreNum = (uint64_t)(*((uint32_t *)tilingHost + MLATiling::TILING_KVCORENUM));
    uint64_t oFdSize = embeddingSize * numHeads * numTokens * kvSplitCoreNum * sizeof(float);
    uint64_t lSize = numTokens * numHeads * kvSplitCoreNum * sizeof(float);

    aclrtFreeHost(tilingHost);

    return {aicCoreNum, kvSplitCoreNum, oFdSize, lSize};
}

} // namespace CatlassKernel

// Enqueue the prepared BF16 kernel on the caller's stream.
extern "C" __attribute__((visibility("default"))) int flash_mla_a3_catlass_launch(
    uint32_t grid, void *stream, void *q, void *qr, void *k, void *kr,
    void *table, void *out, void *s, void *p, void *otmp, void *oupdate,
    void *osplit, void *l, void *tiling)
{
    uint64_t fftsAddr = 0;
    uint32_t fftsLen = 0;
    rtError_t status = rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    if (status != RT_ERROR_NONE) {
        return static_cast<int>(status);
    }
    MLABf16<<<grid, nullptr, static_cast<aclrtStream>(stream)>>>(
        fftsAddr, static_cast<uint8_t *>(q), static_cast<uint8_t *>(qr),
        static_cast<uint8_t *>(k), static_cast<uint8_t *>(kr),
        static_cast<uint8_t *>(table), static_cast<uint8_t *>(out),
        static_cast<uint8_t *>(s), static_cast<uint8_t *>(p),
        static_cast<uint8_t *>(otmp), static_cast<uint8_t *>(oupdate),
        static_cast<uint8_t *>(osplit), static_cast<uint8_t *>(l),
        static_cast<uint8_t *>(tiling));
    return static_cast<int>(RT_ERROR_NONE);
}

extern "C" __attribute__((visibility("default"))) int flash_mla_a3_catlass_graph_launch(
    uint32_t grid, void *stream, void *q, void *qr, void *k, void *kr,
    void *table, void *out, void *s, void *p, void *otmp, void *oupdate,
    void *osplit, void *l, void *tiling, void *lengths)
{
    uint64_t fftsAddr = 0;
    uint32_t fftsLen = 0;
    rtError_t status = rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    if (status != RT_ERROR_NONE) {
        return static_cast<int>(status);
    }
    MLAGraphBf16<<<grid, nullptr, static_cast<aclrtStream>(stream)>>>(
        fftsAddr, static_cast<uint8_t *>(q), static_cast<uint8_t *>(qr),
        static_cast<uint8_t *>(k), static_cast<uint8_t *>(kr),
        static_cast<uint8_t *>(table), static_cast<uint8_t *>(out),
        static_cast<uint8_t *>(s), static_cast<uint8_t *>(p),
        static_cast<uint8_t *>(otmp), static_cast<uint8_t *>(oupdate),
        static_cast<uint8_t *>(osplit), static_cast<uint8_t *>(l),
        static_cast<uint8_t *>(tiling), static_cast<uint8_t *>(lengths));
    return static_cast<int>(RT_ERROR_NONE);
}

extern "C" __attribute__((visibility("default"))) int flash_mla_a3_catlass_balanced_graph_launch(
    uint32_t grid, void *stream, void *q, void *qr, void *k, void *kr,
    void *table, void *out, void *s, void *p, void *otmp, void *oupdate,
    void *osplit, void *l, void *tiling, void *lengths, void *schedule)
{
    uint64_t fftsAddr = 0;
    uint32_t fftsLen = 0;
    rtError_t status = rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    if (status != RT_ERROR_NONE) {
        return static_cast<int>(status);
    }
    MLABalancedGraphBf16<<<grid, nullptr, static_cast<aclrtStream>(stream)>>>(
        fftsAddr, static_cast<uint8_t *>(q), static_cast<uint8_t *>(qr),
        static_cast<uint8_t *>(k), static_cast<uint8_t *>(kr),
        static_cast<uint8_t *>(table), static_cast<uint8_t *>(out),
        static_cast<uint8_t *>(s), static_cast<uint8_t *>(p),
        static_cast<uint8_t *>(otmp), static_cast<uint8_t *>(oupdate),
        static_cast<uint8_t *>(osplit), static_cast<uint8_t *>(l),
        static_cast<uint8_t *>(tiling), static_cast<uint8_t *>(lengths), static_cast<uint8_t *>(schedule));
    return static_cast<int>(RT_ERROR_NONE);
}
