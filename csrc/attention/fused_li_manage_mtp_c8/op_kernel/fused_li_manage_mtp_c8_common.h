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
 * \file fused_li_manage_mtp_c8_common.h
 * \brief
 */
#ifndef FUSED_LI_MANAGE_MTP_C8_COMMON_H
#define FUSED_LI_MANAGE_MTP_C8_COMMON_H

namespace LIMtpC8Common {

// 与tiling的layout保持一致
enum class LI_LAYOUT {
    BSND = 0,
    TND = 1,
    PA_BSND = 2
};

template <typename Q_T, typename K_T, typename W_T, typename OUT_T, const bool PAGE_ATTENTION = false,
          LI_LAYOUT LAYOUT_T = LI_LAYOUT::BSND, LI_LAYOUT K_LAYOUT_T = LI_LAYOUT::PA_BSND, typename... Args>
struct LIType {
    using queryType = Q_T;
    using keyType = K_T;
    using weightType = W_T; // C8: query/key int8, weights stay bf16
    using outputType = OUT_T;
    static constexpr bool pageAttention = PAGE_ATTENTION;
    static constexpr LI_LAYOUT layout = LAYOUT_T;
    static constexpr LI_LAYOUT keyLayout = K_LAYOUT_T;
};

struct RunInfo {
    uint32_t loop;
    uint32_t bN2Idx;
    uint32_t bIdx;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx;
    uint32_t s2Idx;

    uint32_t actS1Size = 1;
    uint32_t actS2Size = 1;
    uint32_t actMBaseSize;
    uint32_t actualSingleProcessSInnerSize;
    uint32_t actualSingleProcessSInnerSizeAlign;

    uint64_t tensorQueryOffset;
    uint64_t tensorKeyOffset;
    uint64_t tensorWeightsOffset;
    uint64_t indiceOutOffset;

    bool isFirstS2InnerLoop;
    bool isLastS2InnerLoop;
    bool isAllLoopEnd = false;
    bool causal = false;
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
    // 无效索引
    static constexpr int INVALID_IDX = -1;

    // CUBE和VEC的核间同步EventID。V0C1(1)/C1V0(2) 与 fused_li_manage_c8 一致。
    // C1V1 与 V1C1 必须拆成不同 flag (fused_li_manage_c8 双向共用 0 是被其
    // 对称负载掩盖的隐患): ffts 到达按 flagId 汇池, 双向共用 0 时 AIV 的
    // wait(C1V1) 可能消费兄弟 AIV 的 V1C1 到达而被提前放行, 在 cube Fixpipe
    // 写完成前读 mm1Res 得到撕裂数据 (q=7 双 AIV 满载时窗口打开, b2 偶数
    // chunk 全量污染的根因)。拆开后 C1V1(3) 池内只有本 AIC 的到达,
    // N set : 每 AIV N wait 逐 chunk 严格配对, 无法提前消费。
    uint32_t syncC1V1 = 3U;
    uint32_t syncV1C1 = 0U;
    uint32_t syncV0C1 = 1U;
    uint32_t syncC1V0 = 2U;

    // 基本块大小
    uint32_t mBaseSize = 1ULL;
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
    int64_t preTokens = INT64_MAX;
    int64_t nextTokens = INT64_MAX;
    bool returnValue = false;

    uint32_t actualLenQDims = 0U; // query的actualSeqLength 的维度
    uint32_t actualLenDims = 0U;  // KV 的actualSeqLength 的维度
    bool isAccumSeqS1 = false;    // 是否累加模式
    bool isAccumSeqS2 = false;    // 是否累加模式
};

struct SplitCoreInfo {
    uint32_t s2Start = 0U; // S2的起始位置
    uint32_t s2End = 0U;   // S2循环index上限
    uint32_t bN2Start = 0U;
    uint32_t bN2End = 0U;
    uint32_t gS1Start = 0U;
    uint32_t gS1End = 0U;
    bool isLD = false;     // 当前核是否需要进行Decode归约任务
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
} // namespace LIMtpC8Common

#endif // FUSED_LI_MANAGE_MTP_C8_COMMON_H
