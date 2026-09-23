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
 * \file pivot_lightning_indexer_common.h
 * \brief
 */
#ifndef PIVOT_LIGHTNING_INDEXER_COMMON_H
#define PIVOT_LIGHTNING_INDEXER_COMMON_H
using namespace AscendC;
namespace LICommon {

// 与tiling的layout保持一致
enum class LI_LAYOUT {
    BSND = 0,
    TND = 1,
    PA_BSND = 2
};

template <typename Q_T, typename K_T, typename OUT_T, const bool PAGE_ATTENTION = false,
          LI_LAYOUT LAYOUT_T = LI_LAYOUT::BSND, LI_LAYOUT K_LAYOUT_T = LI_LAYOUT::PA_BSND,
          bool DT_W_FLAG = false, typename... Args>
struct LIType {
    static constexpr bool pivotReuse = false;
    static constexpr bool weightsTypeFlag = DT_W_FLAG;   // weight的dtype是否为FP32
    using queryType = Q_T;
    using keyType = K_T;
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
    uint32_t actS2SizeOrig = 1;
    uint32_t actMBaseSize;
    uint32_t actualSingleProcessSInnerSize;
    uint32_t actualSingleProcessSInnerSizeAlign;

    uint64_t tensorQueryOffset;
    uint64_t tensorKeyOffset;
    uint64_t tensorWeightsOffset;
    uint64_t indiceOutOffset;
    uint64_t valueOutOffset;

    bool isFirstS2InnerLoop;
    bool isLastS2InnerLoop;
    bool isAllLoopEnd = false;
    bool isValid = false;
};

struct ConstInfo {
    uint32_t pivotInputRows = 0;
    uint32_t pivotDenseRows = 0;
    // CUBE与VEC核间同步的模式
    static constexpr uint32_t FIA_SYNC_MODE2 = 2;
    static constexpr uint32_t QLI_SYNC_MODE4 = 4;
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
    uint16_t INVALID_VAL = 0;
    // CUBE和VEC的核间同步EventID
    uint32_t syncC1V1 = 0U;
    uint32_t syncC1V0 = 2U;
    uint32_t syncV1C1 = 0U;
    uint32_t syncV0C1 = 1U;

    // 基本块大小
    uint32_t mBaseSize = 1ULL;
    uint32_t mBaseSizeAlign = 1ULL;
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
    bool isSparseCountOver2K = false; //sparseCount小于等于2048为false
    bool isLDOpen = false;
    bool returnValueFlag = false;
    bool splitMFlag = false;
};

__aicore__ inline uint32_t PivotSourceRow(const ConstInfo &info, uint32_t row)
{
    return row < info.pivotDenseRows ? row : info.pivotDenseRows + 4 * (row - info.pivotDenseRows);
}

__aicore__ inline uint32_t PivotRowCopies(const ConstInfo &info, uint32_t row)
{
    uint32_t remaining = info.pivotInputRows - PivotSourceRow(info, row);
    return row < info.pivotDenseRows ? 1 : (remaining < 4 ? remaining : 4);
}

// Widest trailing window any one row of a group can need forced into its top-k.
constexpr uint32_t PIVOT_LOCAL_WINDOW = 4;
// Stride between the rows' window slots in the scratch buffer. DataCopyPad
// requires its UB source to start on a 32-byte boundary and a row needs at most
// PIVOT_LOCAL_WINDOW - 1 int32, so a packed layout would misalign odd rows.
constexpr uint32_t PIVOT_LOCAL_WINDOW_STRIDE = 8;

// How many of row `repeat`'s trailing causal keys the group's shared scan cannot
// have seen.
//
// PIVOT-Reuse scores one proxy query per group and replicates its top-k to every
// row. The shared scan covers the keys of the group's FIRST row, [0, L0-1]. Under
// a causal mask each following row is one position later, so row r reaches keys
// up to L0+r-1: exactly r of them lie past the end of the scan and were never
// scored by anyone. Row 0 is the row the scan was computed for and loses nothing.
//
// That is what makes appending them cheap and unconditional -- no membership
// test, no re-scoring, and nothing to evict, because a key the shared top-k
// cannot contain cannot be a duplicate of it. The row's older candidates were
// scored by the proxy and are left to its judgement.
//
// Widths are row-dependent, so the row is emitted as a head of sparseCount - r
// shared ids plus an r-wide tail; row 0 takes the untouched one-copy path.
//
// Without a causal mask every row of the group scans the same range and none is
// excluded, so this is 0 for all of them and PIVOT-Reuse has no defect there.
// Not Min(): the AscendC Min visible here is the element-wise vector intrinsic
// over LocalTensor and returns void. This header is included before the scalar
// overloads exist, so the clamp is written out.
__aicore__ inline int64_t PivotWindowWidth(int64_t repeat, bool perRowCausal)
{
    constexpr int64_t widest = static_cast<int64_t>(PIVOT_LOCAL_WINDOW) - 1;
    if (!perRowCausal || repeat <= 0) {
        return 0;
    }
    return repeat < widest ? repeat : widest;
}

// Write those ids -- row r's are [firstRowLen, firstRowLen + r - 1] -- into the
// scratch buffer, one row per `repeat`. Returns the widest width it wrote, i.e.
// the number of rows that splice minus one, or 0 when no row splices and the
// caller must copy every row verbatim.
//
// `firstRowLen` is the causal key count of the group's FIRST row -- exactly the
// `cuRealAcSeq` the emitter already computed for that row. Under sparseMode 3
// each following row is one position later, so its window starts at the same
// absolute position; without the mask pass perRowCausal = false and nothing is
// written. Row r's last id is firstRowLen + r - 1, which is within its own causal
// bound by construction, so no length guard is needed.
//
// The ids go to a small scratch buffer rather than into the top-k buffer the
// vector pipe just produced. That keeps the scalar stores independent of that
// pipe -- no V_S drain -- at the cost of splitting each row's copy-out in two.
// Every row is filled before a single S_MTE3, so the barrier is one flag per
// group, not per row. This header is included from pivot_lightning_indexer.cpp before
// the AscendC intrinsics are declared, so it cannot issue that flag itself; the
// caller owns it and issues it once, after this returns positive.
//
// Note: only indices are rewritten. The returned scores for those slots still
// hold whatever the row carried there; they are consumed only when returnValue
// is set, which the PIVOT experiments do not do.
//
// Returns 0 -- meaning "emit every row verbatim" -- unless the caller is writing
// the whole row at once. A sparseCount above 4096 is emitted in halves, and the
// tail of the first half is not the tail of the top-k, so splicing a window
// there would land in the wrong place. PIVOT-Reuse is gated on sparseCount ==
// 2048 and never takes that path, but the check keeps the helper safe to call.
__aicore__ inline int64_t PivotFillLocalWindows(const ConstInfo &constInfo, int64_t copies,
    int64_t firstRowLen, bool perRowCausal, int64_t copyLen,
    const LocalTensor<int32_t> &localIds)
{
    if (copies <= 0 || copyLen != static_cast<int64_t>(constInfo.sparseCount)) {
        return 0;
    }
    int64_t widest = PivotWindowWidth(copies - 1, perRowCausal);
    for (int64_t repeat = 0; repeat < copies; ++repeat) {
        int64_t width = PivotWindowWidth(repeat, perRowCausal);
        for (int64_t slot = 0; slot < width; ++slot) {
            localIds.SetValue(static_cast<int32_t>(repeat * PIVOT_LOCAL_WINDOW_STRIDE + slot),
                static_cast<int32_t>(firstRowLen + slot));
        }
    }
    return widest;
}

struct SplitCoreInfo {
    uint32_t s2Start = 0U; // S2的起始位置
    uint32_t s2End = 0U;   // S2循环index上限
    uint32_t bN2Start = 0U;
    uint32_t bN2End = 0U;
    uint32_t gS1Start = 0U;
    uint32_t gS1End = 0U;
    bool isLD = false;     // 当前核是否需要进行Decode归约任务
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
} // namespace LICommon

// bank冲突优化
// david 256KB bank layout
// shape  (             bank_depth  (            banks  bank_groups  block))  (512  (  2   8  32))
// stride (banks*bank_groups*block  (bank_groups*block        block      1))  (512  (256  32   1))
#define UB_BLOCK              32   // 32B
#define UB_BANK_GROUPS        8
#define UB_BANKS              2
#define UB_BANK_DEPTH         512

#define UB_BANK_GROUP_STRIDE  UB_BLOCK                                   // 32B
#define UB_BANK_STRIDE        (UB_BANK_GROUPS * UB_BLOCK)               // 256B
#define UB_BANK_DEPTH_STRIDE  (UB_BANKS * UB_BANK_GROUPS * UB_BLOCK)    // 512B

#endif // PIVOT_LIGHTNING_INDEXER_COMMON_H
