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
 * \file quant_lightning_indexer_common.h
 * \brief
 */
#ifndef VLLM_FAST_QUANT_LIGHTNING_INDEXER_COMMON_H
#define VLLM_FAST_QUANT_LIGHTNING_INDEXER_COMMON_H
using namespace AscendC;
namespace VllmFastQLICommon {

// =====================================================================================
// [QuantLI vs LI 差异点 #3] LAYOUT 枚举
// =====================================================================================
// LI (lightning_indexer_common.h L21-25): LI_LAYOUT 同样的三个值, 无差异
// =====================================================================================

// 与tiling的layout保持一致
enum class LI_LAYOUT : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BSND = 2
};

// =====================================================================================
// [QuantLI vs LI 差异点 #4] 类型系统 QLIType vs LIType   ★核心差异★
// =====================================================================================
// LI 的 LIType (lightning_indexer_common.h L27-38):
//   template <typename Q_T, typename K_T, typename OUT_T,
//             bool PAGE_ATTENTION, LI_LAYOUT LAYOUT_T, LI_LAYOUT K_LAYOUT_T,
//             bool DT_W_FLAG, typename... Args>
//   7个模板参数: Q_K_OUT类型 + 2个layout + 1个flag
//   只定义了: queryType, keyType, outputType, weightsTypeFlag, pageAttention, layout, keyLayout
//   SCORE_T 在 kernel.h 里固定为 uint32_t (L68)
//   W_T 通过 LightningIndexerTypeTraits 推导 (service_vector.h L36-43)
//
// QuantLI 的 QLIType (本文件):
//   10个模板参数: 多了 W_T(weight类型), SCALE_T(scale类型), QK_T(Cube累加类型), SCORE_T(score存储类型)
//   每个dtype组合(如fp8_e4m3fn + half_weight + uint16_score)都显式指定, 不依赖traits推导
//   原因: 量化路径下精度的选择更多 - QK可以是int32/float32, score可以是uint16/uint32,
//         scale可以是bf16/fp16/fp32, weight可以是bf16/fp16/fp32, 必须编译期确定
// =====================================================================================

template <typename Q_T, typename K_T, typename OUT_T, const bool PAGE_ATTENTION = false,
          LI_LAYOUT Q_LAYOUT_T = LI_LAYOUT::BSND, LI_LAYOUT K_LAYOUT_T = LI_LAYOUT::PA_BSND,
          typename W_T = half, typename SCALE_T = half, typename QK_T = int32_t, typename SCORE_T = uint16_t,
          typename... Args>
struct QLIType {
    using queryType = Q_T;
    using keyType = K_T;
    using qkType = QK_T;         // ← QuantLI 独有: Cube Matmul 输出类型 (int32/float32)
    using outputType = OUT_T;
    static constexpr bool pageAttention = PAGE_ATTENTION;
    static constexpr LI_LAYOUT layout = Q_LAYOUT_T;
    static constexpr LI_LAYOUT keyLayout = K_LAYOUT_T;
    using scoreType = SCORE_T;   // ← QuantLI 独有: score 存储类型 (uint16/uint32)
    using weightType = W_T;      // ← QuantLI 独有: weight 精度 (half/bf16/fp32)
    using scaleType = SCALE_T;   // ← QuantLI 独有: 反量化 scale 精度 (half/bf16/fp32)
};

// =====================================================================================
// RunInfo — 每个循环迭代的运行时信息
// =====================================================================================
// 与 lidu (LICommon::RunInfo) 同构，C8 额外保留:
//   s2Start          — S2 循环起始位置（kScale 窗口索引 %16 用）
//   validS2Len       — attenMask 下有效 S2 长度（kScale 预取长度用）
//   kScaleLoop       — kScale 16 块窗口 pingpong 索引（大 source 长度下活跃）
//   tensorKeyScaleOffset — keyScale GM 偏移
//   cacheRowIdx / cacheTokenCount — fused 缓存管理（与 lidu 一致）
// （重构已删除死代码 qScaleLoop；lidu 的 valueOutOffset 因 C8 无 value 输出而不存在）
// =====================================================================================
struct RunInfo {
    uint32_t loop;
    uint32_t bN2Idx;
    uint32_t bIdx;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx;
    uint32_t s2Idx;
    uint32_t s2Start;           // S2 起始位置
    uint32_t validS2Len;        // attenMask 下有效 S2 长度
    uint32_t kScaleLoop;        // kScale pingpong
    uint32_t cacheRowIdx = 0;       // fused C8: req_pool_entries[bIdx] 指向的缓存行
    uint32_t cacheTokenCount = 0;   // fused C8: cache_tokens[bIdx] 预算 C
    uint32_t offloadedLen = 0;      // durable prefix; selected tail is a resident hit
    // First private-workspace row of this request: B*4 for legacy Q1-4
    // launches, or the compact query start for a launch containing Q5-14.
    // The vector service derives the current route from the public row.
    uint32_t workspaceRow = 0;

    uint32_t actS1Size = 1;
    uint32_t actS2Size = 1;
    uint32_t actS2SizeOrig = 1;
    uint32_t actMBaseSize;
    uint32_t actualSingleProcessSInnerSize;
    uint32_t actualSingleProcessSInnerSizeAlign;

    uint64_t tensorQueryOffset;
    uint64_t tensorKeyOffset;
    uint64_t tensorKeyScaleOffset; // ← QuantLI 独有: keyScale GM偏移
    uint64_t tensorWeightsOffset;
    uint64_t tensorQueryScaleOffset;
    uint64_t indiceOutOffset;

    bool isFirstS2InnerLoop;
    bool isLastS2InnerLoop;
    bool isAllLoopEnd = false;
    bool isValid = false;
    // All routes scan the final KV length and clip TopK independently to
    // their causal visible length. DENSE publishes plain token IDs without
    // consulting cache_slots; SPARSE classifies prefix/tail in the payload.
    bool causalClip = false;
    bool denseDirectPublish = false;
};

// =====================================================================================
// [QuantLI vs LI 差异点 #6] ConstInfo — Tiling层传递的运行时常量
// =====================================================================================
// LI (lightning_indexer_common.h L67-124): 约30个字段
// QuantLI (本文件): 约26个字段
//
// QuantLI 有而 LI 没有的:
//   tSize                  — TND layout下的总token数
//   keyStride0             — PA场景 key 跨block stride
//   keyDequantScaleStride0 — PA场景 keyScale 跨block stride
//   setL2DisableFlag       — tSize较小时关闭L2 cache改善性能
//   (注: QLI_SYNC_MODE4 vs LI_SYNC_MODE4 — 同步模式常量名不同, 值相同=4)
//
// LI 有而 QuantLI 没有的:
//   mBaseSizeAlign   — M方向对齐
//   INVALID_VAL      — 无效value标记(bf16=0xFF80/fp16=0xFC00)
//   preTokens/nextTokens    — attention mask窗口
//   returnValueFlag / splitMFlag / isSparseCountOver2K
//   returnValue
// =====================================================================================
struct ConstInfo {
    // CUBE与VEC核间同步的模式
    static constexpr uint32_t FIA_SYNC_MODE2 = 2;
    static constexpr uint32_t QLI_SYNC_MODE4 = 4;    // 值与LI_SYNC_MODE4相同(=4), 仅命名不同
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

    // CUBE和VEC的核间同步EventID
    uint32_t syncC1V1 = 0U;
    uint32_t syncC1V0 = 2U;
    uint32_t syncV1C1 = 0U;
    uint32_t syncV0C1 = 1U;

    // ========== 基本块大小 ==========
    // QuantLI: mBaseSize = S1_BASE_SIZE(4或2) * gSize = 128 或 64 (取决于tSize)
    // LI:      mBaseSize = S1_BASE_SIZE(4或2) * gSize = 256 或 128 (取决于sparseCount)
    uint32_t mBaseSize = 1ULL;
    uint32_t s1BaseSize = 1ULL;
    uint32_t s2BaseSize = 1ULL;

    uint64_t batchSize = 0ULL;
    uint64_t tSize = 0ULL;             // ← QuantLI 独有: TND的T维度总大小
    uint64_t gSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kHeadNum;
    uint64_t headDim;
    uint64_t keyStride0;               // ← QuantLI 独有: PA key GM stride
    uint64_t keyDequantScaleStride0;   // ← QuantLI 独有: PA keyScale GM stride
    uint64_t sparseCount;              // topK选取大小
    uint64_t kSeqSize = 0ULL;          // kv最大S长度
    uint64_t qSeqSize = 1ULL;          // q最大S长度
    uint32_t kCacheBlockSize = 0;      // PA场景的block size
    uint32_t maxBlockNumPerBatch = 0;  // PA场景的最大单batch block number
    LI_LAYOUT outputLayout;            // 输出的格式
    bool attenMaskFlag = false;
    bool setL2DisableFlag = false;     // ← QuantLI 独有: tSize≤s1BaseSize时关闭L2

    uint32_t actualLenQDims = 0U;  // query的actualSeqLength 的维度
    uint32_t actualLenDims = 0U;   // KV 的actualSeqLength 的维度
    bool isAccumSeqS1 = false;     // 是否累加模式
    bool isAccumSeqS2 = false;     // 是否累加模式
    bool isLDOpen = false;
    uint64_t poolSize = 0ULL;          // fused C8: request-pool 行数
    uint64_t cacheSlotsSize = 0ULL;    // fused C8: 缓存行容量 (<= 2^18)
};

struct SplitCoreInfo {
    uint32_t s2Start = 0U;  // S2的起始位置
    uint32_t s2End = 0U;    // S2循环index上限
    uint32_t bN2Start = 0U;
    uint32_t bN2End = 0U;
    uint32_t gS1Start = 0U;
    uint32_t gS1End = 0U;
    bool isLD = false;  // 当前核是否需要进行Decode归约任务
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
}  // namespace VllmFastQLICommon

// bank冲突优化
// david 256KB bank layout
// shape  (             bank_depth  (            banks  bank_groups  block))  (512  (  2   8  32))
// stride (banks*bank_groups*block  (bank_groups*block        block      1))  (512  (256  32   1))
#ifndef UB_BLOCK
#define UB_BLOCK              32   // 32B
#define UB_BANK_GROUPS        8
#define UB_BANKS              2
#define UB_BANK_DEPTH         512

#define UB_BANK_GROUP_STRIDE  UB_BLOCK                                   // 32B
#define UB_BANK_STRIDE        (UB_BANK_GROUPS * UB_BLOCK)               // 256B
#define UB_BANK_DEPTH_STRIDE  (UB_BANKS * UB_BANK_GROUPS * UB_BLOCK)    // 512B
#endif

#endif  // VLLM_FAST_QUANT_LIGHTNING_INDEXER_COMMON_H
