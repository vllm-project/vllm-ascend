/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ifa_public_define.h
 * \brief
 */
#ifndef FIA_PUBLIC_DEFINE_H
#define FIA_PUBLIC_DEFINE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

using namespace AscendC;
using AscendC::AIC;
using AscendC::AIV;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::SetFlag;
using AscendC::ShapeInfo;
using AscendC::SoftmaxConfig;
using AscendC::WaitFlag;

namespace AttentionCommon {
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

// 将isCheckTiling设置为false
constexpr SoftmaxConfig FIA_SOFTMAX_FLASHV2_CFG = {false};
// 将isCheckTiling设置为false, 输入输出的max&sum&exp的shape为(m, 1)
constexpr SoftmaxConfig FIA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC = {false, 0, 0, SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};

enum class FIA_LAYOUT : uint32_t
{
    BSH = 0,
    BSND = 0,
    BNSD = 1,
    NZ = 2,
    TND = 3,
    NBSD = 4,
    NTD = 5
};

template <typename Q_T, typename KV_T, typename OUT_T, typename ORIGIN_T, const bool PAGE_ATTENTION = false,
          const bool FLASH_DECODE = false, FIA_LAYOUT LAYOUT_T = FIA_LAYOUT::BSH, const uint8_t ANTIQUANT_MODE = 0,
          const bool SHARED_PREFIX = false, FIA_LAYOUT KV_LAYOUT_T = FIA_LAYOUT::BSH,
          const bool SOFTMAX_WITH_BRC = false, const bool ENABLE_TREE = false, typename... Args>
struct FIAType {
    using queryType = Q_T;
    using kvType = KV_T;
    using outputType = OUT_T;
    using orginalType = ORIGIN_T;
    static constexpr bool pageAttention = PAGE_ATTENTION;
    static constexpr bool flashDecode = FLASH_DECODE;
    static constexpr FIA_LAYOUT layout = LAYOUT_T;
    static constexpr uint8_t antiquantMode = ANTIQUANT_MODE;
    static constexpr bool sharedPrefix = SHARED_PREFIX;
    static constexpr FIA_LAYOUT kvLayout = KV_LAYOUT_T;
    static constexpr bool softmaxWithBrc = SOFTMAX_WITH_BRC;
    static constexpr bool enableTree = ENABLE_TREE;
};

struct FDparams {
    uint32_t *bN2IdxOfFdHead;
    uint32_t *gS1IdxOfFdHead;
    uint32_t *s2SplitNumOfFdHead;
    uint32_t *gS1SplitNumOfFdHead;
    uint32_t *gS1LastPartSizeOfFdHead;
    uint32_t *gS1IdxEndOfFdHead;
    uint32_t *gS1IdxEndOfFdHeadSplit;
    uint32_t usedVecNumOfFd;
    uint32_t gS1BaseSizeOfFd;
};

struct RunInfo {
    bool isValid = false;
    bool isChangeBatch = false;
    bool isFirstSInnerLoop = false;
    bool isLastS2Loop = false;
    uint32_t loop = 0;

    uint32_t bIdx = 0;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx = 0;
    uint32_t s2Idx = 0;
    uint64_t actS1Size = 1; // 当前处理head的S1轴实际大小
    uint64_t actS2Size = 1; // 当前处理head的S2轴实际大小

    uint32_t actMBaseSize = 0;
    uint32_t actualSingleProcessSInnerSize = 0;
    uint32_t actualSingleProcessSInnerSizeAlign = 0;

    uint32_t curSInnerLoopTimes = 0;
    uint32_t bn2IdxInCurCore = 0;
    uint32_t s2BatchOffset = 0;
    uint32_t tndIsS2SplitCore = 0;
    uint32_t tndCoreStartKVSplitPos = 0;

    uint64_t tensorAOffset = 0;
    uint64_t tensorBOffset = 0;
    uint64_t tensorARopeOffset = 0;
    uint64_t tensorBRopeOffset = 0;
    uint64_t attenOutOffset = 0;
    uint64_t attenMaskOffset = 0;

    int64_t preTokensPerBatch = 0;
    int64_t nextTokensPerBatch = 0;
    uint64_t accumTmpOutNum = 0;

    uint64_t qPaddingBeginOffset = 0;
    uint64_t kvPaddingBeginOffset = 0;
};

struct ConstInfo {
    // CUBE与VEC核间同步的模式
    static constexpr uint32_t FIA_SYNC_MODE2 = 2;
    // BUFFER的字节数
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    static constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64;
    static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512;
    static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
    static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
    // FP32的0值和极大值
    static constexpr float FLOAT_ZERO = 0;
    static constexpr float FLOAT_MAX = 3.402823466e+38F;
    static constexpr float FLOAT_INF = 3e+99;
    // 整个AICORE的任务信息, 左闭右开区间[ (bN2Start, gS1Start, s2Start), (bN2End, gS1End, s2End) )
    uint32_t bN2Start = 0U;
    uint32_t gS1Start = 0U;
    uint32_t s2Start = 0U;
    uint32_t bN2End = 0U;
    uint32_t gS1End = 0U;
    uint32_t s2End = 0U;

    // preLoad的总次数
    uint32_t preLoadNum = 0U;
    uint32_t nBufferMBaseSize = 0U;
    // CUBE和VEC的核间同步EventID
    uint32_t syncV1NupdateC2 = 0U;
    uint32_t syncV0C1 = 0U;
    uint32_t syncC1V1 = 0U;
    uint32_t syncV1C2 = 0U;
    uint32_t syncC2V2 = 0U;
    uint32_t syncC2V1 = 0U;

    float scaleValue = 0;
    uint32_t mmResUbSize = 0U;   // Matmul1输出结果GM上的大小
    uint32_t vec1ResUbSize = 0U; // Vector1输出结果GM上的大小
    uint32_t bmm2ResUbSize = 0U; // Matmul2输出结果GM上的大小
    uint64_t batchSize = 0ULL;
    uint64_t gSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kvHeadNum = 0ULL;
    uint64_t headDim = 0;
    uint64_t headDimRope = 0;
    uint64_t headDimAlign = 0;
    uint64_t kvSeqSize = 0ULL;        // kv最大S长度
    uint64_t qSeqSize = 1ULL;         // q最大S长度
    int64_t preToken = 0;
    int64_t nextToken = 0;
    uint64_t systemPrefixMaxLen = 0;
    uint64_t attenMaskBatchStride = 0ULL;
    uint64_t qLeftPaddingSize = 0;
    uint64_t kvLeftPaddingSize = 0;
    uint32_t kvCacheBlockSize = 0;    // PA场景的block size
    uint32_t maxBlockNumPerBatch = 0; // PA场景的最大单batch block number
    uint32_t splitKVNum = 0U;         // S2核间切分的切分份数
    FIA_LAYOUT outputLayout;          // 输出的Transpose格式
    uint32_t systemPrefixLen = 0;
    uint32_t subBlockNum = 2; // AI Core上 AIC与AIV的数量默认为1:2

    uint32_t pseShiftS1 = 0U;
    uint32_t pseShiftS2 = 0U;
    uint32_t attenMaskStride = 0ULL;
    uint32_t sparseMode = 0;
    uint32_t tndFDCoreArrLen = 0U;     // TNDFlashDecoding相关分核信息array的长度
    uint32_t coreStartKVSplitPos = 0U; // TNDFlashDecoding kv起始位置

    uint32_t actualLenQDims = 0U; // query的actualSeqLength 的维度
    uint32_t actualLenDims = 0U;  // KV 的actualSeqLength 的维度
    uint32_t mBaseSize = 1ULL;
    uint32_t s2BaseSize = 1ULL;
    uint32_t l2CacheOffFlag = 0;

    bool attenMaskFlag = false;
    bool accumQSeqFlag = false;
    bool accumKVSeqFlag = false;
    bool needInit = false;
    bool isRowInvalid = false;  // 是否使能行无效
    bool isExistRowInvalid = false;  // 实际是否存在行无效

    bool batchContinuous = true;
    bool ropeSplitMode = false;

    bool pseShiftFlag = false;
    bool pseShiftByBatch = false;

    bool softmaxLseFlag = false;
    bool isLegacyIfa = false;

    bool isQHasLeftPadding = false;
    bool isKVHasLeftPadding = false;

    bool headS2Split = false;
    bool tailS2Split = false;
    bool systemPrefixFlag = false;

    bool isPostQuantPerChn = false;
    bool isPostQuantTypeBf16 = false;

    uint64_t keyBnStride = 0;
    uint64_t keyN2Stride = 0;
    uint64_t valueBnStride = 0;
    uint64_t valueN2Stride = 0;
    uint64_t keyRopeBnStride = 0;
    uint64_t keyRopeN2Stride = 0;
};

struct FusedTransposeInfo {
    // 以下是FlashDecode分支区分的信息
    uint32_t n2Idx = 0;
    uint32_t bIdx = 0;

    // 以下是需要用公式计算的信息
    uint32_t s1StartIdx = 0;
    uint32_t s1EndIdx = 0;
    uint32_t s1Count = 0;
    uint32_t gStartIdx = 0;
    uint32_t gEndIdx = 0;
    uint32_t gCount = 0;
};

struct MSplitInfo {
    uint32_t nBufferIdx = 0U;
    uint32_t nBufferStartM = 0U;
    uint32_t nBufferDealM = 0U;
    uint32_t vecStartM = 0U;
    uint32_t vecDealM = 0U;
};

template <FIA_LAYOUT LAYOUT_T>
__aicore__ inline void GetGS1Idx(uint32_t gS1Idx, uint32_t &gIdx, uint32_t &s1Idx, AttentionCommon::ConstInfo &constInfo)
{
    // GS1
    if constexpr (LAYOUT_T == FIA_LAYOUT::BNSD || LAYOUT_T == FIA_LAYOUT::NBSD || LAYOUT_T == FIA_LAYOUT::NTD) {
        gIdx = gS1Idx / constInfo.qSeqSize;
        s1Idx = gS1Idx % constInfo.qSeqSize;
    } else {
        // S1G
        s1Idx = gS1Idx / constInfo.gSize;
        gIdx = gS1Idx % constInfo.gSize;
    }
}

__aicore__ inline int64_t ClipSInnerToken(int64_t sInnerToken, int64_t minValue, int64_t maxValue)
{
    sInnerToken = sInnerToken > minValue ? sInnerToken : minValue;
    sInnerToken = sInnerToken < maxValue ? sInnerToken : maxValue;
    return sInnerToken;
}
} // namespace

#endif // FIA_PUBLIC_DEFINE_H
