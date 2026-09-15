/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_lightning_indexer_v2_common_arch35.h
 * \brief
 */
#ifndef QUANT_LIGHTNING_INDEXER_V2_COMMON_H
#define QUANT_LIGHTNING_INDEXER_V2_COMMON_H
using namespace AscendC;
namespace QLIV2Common {
using FP8E4M3 = fp8_e4m3fn_t;
using FP4E2M1 = fp4x2_e2m1_t;
using FP8E8M0 = fp8_e8m0_t;

constexpr uint32_t MX_SCALE_GROUP_SIZE = 32; // 每32个D维度元素一个scale
constexpr uint32_t FP8_TWO = 2;              // 2个FP8E8M0打包成1个BF16
constexpr uint32_t FP4_PACK_NUM = 2;         // 2个FP4E2M1打包成1个字节存储

// 与tiling的layout保持一致
enum class LI_LAYOUT : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BBND = 2
};

template <typename Q_T, typename K_T, typename QK_T, typename SCORE_T, typename OUT_T,
          const bool PAGE_ATTENTION = false, LI_LAYOUT Q_LAYOUT_T = LI_LAYOUT::BSND,
          LI_LAYOUT K_LAYOUT_T = LI_LAYOUT::PA_BBND, typename SCALE_T = float, typename WEIGHT_T = float,
          typename... Args>
struct QLIV2Type {
    static_assert((std::is_same_v<QK_T, float> &&
                   (std::is_same_v<SCORE_T, uint32_t> || std::is_same_v<SCORE_T, uint16_t>)) ||
                      (std::is_same_v<QK_T, bfloat16_t> && std::is_same_v<SCORE_T, uint16_t>) ||
                      (std::is_same_v<QK_T, int32_t> && std::is_same_v<SCORE_T, uint16_t>),
                  "Invalid combination of QK_T and SCORE_T");
    using rawQueryType = Q_T;
    using rawKeyType = K_T;

    static constexpr bool isMxFp8 = std::is_same_v<rawQueryType, FP8E4M3> && std::is_same_v<rawKeyType, FP8E4M3> &&
                                    std::is_same_v<SCALE_T, FP8E8M0>;
    static constexpr bool isMxFp4 = std::is_same_v<rawQueryType, FP4E2M1> && std::is_same_v<rawKeyType, FP4E2M1> &&
                                    std::is_same_v<SCALE_T, FP8E8M0>;
    static constexpr bool isMx = isMxFp8 || isMxFp4;

    using queryType = std::conditional_t<isMxFp4, uint8_t, rawQueryType>;
    using keyType = std::conditional_t<isMxFp4, uint8_t, rawKeyType>;
    using queryKeyType = QK_T;
    using scoreType = SCORE_T;
    using outputType = OUT_T;
    using scaleType = SCALE_T;
    using weightType = WEIGHT_T;

    static constexpr bool pageAttention = PAGE_ATTENTION;
    static constexpr LI_LAYOUT layout = Q_LAYOUT_T;
    static constexpr LI_LAYOUT keyLayout = K_LAYOUT_T;
    static constexpr bool isWeightFP16 = std::is_same_v<WEIGHT_T, half>;
};

struct RunInfo {
    uint32_t loop;
    uint32_t bN2Idx;
    uint32_t bIdx;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx;
    uint32_t s2Idx;
    uint32_t s2Start;
    uint32_t s2LoopEnd;
    uint32_t validS2Len;
    uint32_t qScaleLoop;
    uint32_t kScaleLoop;

    uint32_t actS1Size = 1;
    uint32_t actS2Size = 1;
    uint32_t actS2SizeOrig = 1;
    uint32_t actMBaseSize;
    uint32_t actualSingleProcessSInnerSize;
    uint32_t actualSingleProcessSInnerSizeAlign;
    uint32_t curCuSeqlensQ;
    uint32_t curCuSeqlensK;
    uint32_t curSequsedQ;
    uint32_t curSequsedK;
    uint64_t tensorQueryOffset;
    uint64_t tensorKeyOffset;
    uint64_t tensorQScaleOffset; // MX场景qScale在Cube核使用的偏移
    uint64_t tensorKeyScaleOffset;
    uint64_t tensorWeightsOffset;
    uint64_t indiceOutOffset;
    uint64_t valueOutOffset;
    uint64_t outputIdxCoreOffset;

    bool isFirstS2InnerLoop;
    bool isLastS2InnerLoop;
    bool isAllLoopEnd = false;
    bool isValid = false;
    bool isNeedLD = false;
    bool isOutputIdxOffsetValid = false;
    bool needTndPadding = false;
    uint32_t saveWorkSpaceIdx = 0;
};

struct ConstInfo {
    // CUBE与VEC核间同步的模式
    static constexpr uint32_t QLIV2_SYNC_MODE4 = 4;
    static constexpr uint32_t AIV0_AIV1_OFFSET = 16;
    static constexpr uint32_t CROSS_VC_EVENT = 0;
    static constexpr uint32_t CROSS_CV_EVENT = 2;
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
    // 无效索引
    static constexpr int INVALID_IDX = -1;
    static constexpr uint16_t NEG_INF_BFLOAT = 0xFF80;

    // CUBE和VEC的核间同步EventID
    uint32_t syncC1V1 = 0U;
    uint32_t syncC1V0 = 2U;
    uint32_t syncV1C1 = 0U;
    uint32_t syncV0C1 = 1U;

    // 基本块大小
    uint32_t mBaseSize = 1ULL;
    uint32_t mBaseSizeMax = 1ULL;
    uint32_t s1BaseSize = 1ULL;
    uint32_t s2BaseSize = 1ULL;

    uint64_t batchSize = 0ULL;
    uint64_t gSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kHeadNum;
    uint64_t headDim;
    uint64_t sparseCount;             // topK选取大小
    uint64_t kSeqSize = 0ULL;         // kv最大S长度
    uint64_t qSeqSize = 1ULL;         // q最大S长度
    uint32_t kCacheBlockSize = 0;     // PA场景的block size
    uint32_t maxBlockNumPerBatch = 0; // PA场景的最大单batch block number
    LI_LAYOUT outputLayout;           // 输出的格式
    bool attenMaskFlag = false;
    uint32_t cmpRatio = 1;
    uint32_t keyStride0 = 0;
    uint32_t keyDequantScaleStride0 = 0;
    int32_t maxSeqlenQ = -1;
    uint32_t quantMode = 1; // quant模式，默认为1

    uint32_t actualLenQDims = 0U;     // query的actualSeqLength 的维度
    uint32_t actualLenDims = 0U;      // KV 的actualSeqLength 的维度
    uint32_t cmpResiduaKLenDims = 0U; // cmpResidualK的维度
    bool isAccumSeqS1 = false;        // 是否累加模式
    bool isAccumSeqS2 = false;        // 是否累加模式
    bool isLDOpen = false;
    bool returnValue = false;
};

struct LdSplitCoreInfo {
    bool isLdCoreEnable = false;    // 当前核是否参与规约任务
    uint32_t saveWorkSpaceIdx = 0U; // 存放LD参数的地址
    uint32_t bn2Idx = 0U;           // 归约任务
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t mIdx = 0U;
    uint32_t workspaceIdx = 0U; // 当前AIV核上规约任务的索引
    uint32_t workspaceNum = 0U; // 当前AIV核上规约任务的S2切分数量
    uint32_t mStart = 0U;
    uint32_t mNum = 0U;
    uint64_t indiceOutCoreOffset = 0U; // 最终输出索引搬出Topk的初始偏移地址
};

struct SplitCoreInfo {
    uint32_t s2Start = 0U; // S2的起始位置
    uint32_t s2End = 0U;   // S2循环index上限
    uint32_t bN2Start = 0U;
    uint32_t bN2End = 0U;
    uint32_t gS1Start = 0U;
    uint32_t gS1End = 0U;
    bool isLD = false; // 当前核是否需要进行Decode归约任务
    bool isCoreEnable = false;
};

template <typename T>
__aicore__ inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a > b) ? (a) : (b);
}

template <typename T>
__aicore__ inline T CeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)));
}
} // namespace QLIV2Common

// bank冲突优化
// david 256KB bank layout
// shape  (             bank_depth  (            banks  bank_groups  block))  (512  (  2   8  32))
// stride (banks*bank_groups*block  (bank_groups*block        block      1))  (512  (256  32   1))
#define UB_BLOCK 32 // 32B
#define UB_BANK_GROUPS 8
#define UB_BANKS 2
#define UB_BANK_DEPTH 512

#define UB_BANK_GROUP_STRIDE UB_BLOCK                               // 32B
#define UB_BANK_STRIDE (UB_BANK_GROUPS * UB_BLOCK)                  // 256B
#define UB_BANK_DEPTH_STRIDE (UB_BANKS * UB_BANK_GROUPS * UB_BLOCK) // 512B

#endif // QUANT_LIGHTNING_INDEXER_V2_COMMON_H
