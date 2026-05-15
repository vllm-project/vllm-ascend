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
 * \file sparse_attn_sharedkv_common.h
 * \brief
 */

#ifndef SPARSE_ATTN_SHAREDKV_COMMON_H
#define SPARSE_ATTN_SHAREDKV_COMMON_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

namespace SASKernel {
using namespace AscendC;
// 将isCheckTiling设置为false, 输入输出的max&sum&exp的shape为(m, 1)
constexpr SoftmaxConfig SAS_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC = {false, 0, 0, SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};

enum class SAS_RUN_MODE {
    SWA_MODE = 0,
    SCFA_MODE = 1,
    CFA_MODE = 2,
};

enum class SAS_LAYOUT {
    BSND = 0,
    TND = 1,
    PA_ND = 2
};

template <typename Q_T, typename KV_T, typename OUT_T, const bool FLASH_DECODE = false,
          SAS_LAYOUT LAYOUT_T = SAS_LAYOUT::BSND, SAS_LAYOUT KV_LAYOUT_T = SAS_LAYOUT::PA_ND, int TEMPLATE_MODE = 0,
          typename... Args>
struct SASType {
    using queryType = Q_T;
    using kvType = KV_T;
    using outputType = OUT_T;
    static constexpr bool flashDecode = FLASH_DECODE;
    static constexpr SAS_LAYOUT layout = LAYOUT_T;
    static constexpr SAS_LAYOUT kvLayout = KV_LAYOUT_T;
    static constexpr bool pageAttention = (KV_LAYOUT_T == SAS_LAYOUT::PA_ND);
    static constexpr int templateMode = TEMPLATE_MODE;
};

// ================================Util functions==================================
template <typename T1, typename T2>
__aicore__ inline T1 SASAlign(T1 num, T2 rnd)
{
    return (rnd == 0) ? 0 : ((num + rnd - 1) / rnd * rnd);
}

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 num, T2 rnd)
{
    return (rnd == 0) ? 0 : ((num + rnd - 1) / rnd);
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? b : a;
}

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a > b) ? a : b;
}

template <typename T>
__aicore__ inline size_t BlockAlign(size_t s)
{
    if constexpr (IsSameType<T, int4b_t>::value) {
        return (s + 63) / 64 * 64;
    }
    size_t n = (32 / sizeof(T));
    return (s + n - 1) / n * n;
}

struct PAShape {
    uint32_t blockSize;
    uint32_t headNum;             // 一般为kv的head num，对应n2
    uint32_t headDim;             // 512 对应d
    uint32_t maxblockNumPerBatch; // block table 每一行的最大个数
    uint32_t actHeadDim;          // 实际拷贝col大小,考虑到N切块   s*d, 对应d
    uint32_t copyRowNum;          // 总共要拷贝的行数
    uint32_t copyRowNumAlign;
};

struct Position {
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t s2Idx;
    uint32_t dIdx;
};

// 场景：query、key、value GM to L1
// GM按ND格式存储
// L1按NZ格式存储
// GM的行、列、列的stride
template <typename T>
__aicore__ inline void DataCopyGmNDToL1(LocalTensor<T> &l1Tensor, GlobalTensor<T> &gmTensor, uint32_t rowAct,
                                        uint32_t rowAlign,
                                        uint32_t col,       // D
                                        uint32_t colStride) // D or N*D
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = rowAct; // nd矩阵的行数
    // T为int4场景下，dValue = col / 2，srcDValue = colStride / 2
    nd2nzPara.dValue = col;          // nd矩阵的列数
    nd2nzPara.srcDValue = colStride; // 同一nd矩阵相邻行起始地址间的偏移
    nd2nzPara.dstNzC0Stride = rowAlign;
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmTensor, nd2nzPara);
}

/*
    适用PA数据从GM拷贝到L1，支持ND、NZ数据；
    PA的layout分 BNBD（blockNum,N,blockSize,D） BBH（blockNum,blockSize,N*D
    BSH\BSND\TND 为BBH
    shape.copyRowNumAlign 需要16字节对齐，如拷贝k矩阵，一次拷贝128*512，遇到尾块 10*512 需对齐到16*512
*/
template <typename T, SAS_LAYOUT SRC_LAYOUT>
__aicore__ inline void DataCopyPA(LocalTensor<T> &dstTensor,  // l1
                                  GlobalTensor<T> &srcTensor, // gm
                                  GlobalTensor<int32_t> &blockTableGm,
                                  const PAShape &shape,     // blockSize, headNum, headDim
                                  const Position &startPos) // bacthIdx nIdx curSeqIdx
{
    uint32_t copyFinishRowCnt = 0;
    uint64_t blockTableBaseOffset = startPos.bIdx * shape.maxblockNumPerBatch;
    uint32_t curS2Idx = startPos.s2Idx;
    uint32_t blockElementCnt = 32 / sizeof(T);
    while (copyFinishRowCnt < shape.copyRowNum) {
        uint64_t blockIdOffset = curS2Idx / shape.blockSize; // 获取block table上的索引
        uint64_t reaminRowCnt = curS2Idx % shape.blockSize;  // 获取在单个块上超出的行数
        uint64_t idInBlockTable =
            blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset); // 从block table上的获取编号
        uint32_t copyRowCnt = shape.blockSize - reaminRowCnt;            // 一次只能处理一个Block
        if (copyFinishRowCnt + copyRowCnt > shape.copyRowNum) {
            copyRowCnt = shape.copyRowNum - copyFinishRowCnt; // 一个block未拷满
        }
        uint64_t offset = idInBlockTable * shape.blockSize * shape.headNum * shape.headDim; // PA的偏移

        uint64_t dStride = shape.headDim;
        if constexpr (SRC_LAYOUT == SAS_LAYOUT::BSND || SRC_LAYOUT == SAS_LAYOUT::TND) {
            offset += (uint64_t)(startPos.n2Idx * shape.headDim) + reaminRowCnt * shape.headDim * shape.headNum +
                      startPos.dIdx;
            dStride = shape.headDim * shape.headNum;
        } else {
            offset += (uint64_t)(startPos.n2Idx * shape.headDim * shape.blockSize) + reaminRowCnt * shape.headDim +
                      startPos.dIdx;
        }

        uint32_t dValue = shape.actHeadDim;
        uint32_t srcDValue = dStride;
        LocalTensor<T> tmpDstTensor = dstTensor[copyFinishRowCnt * blockElementCnt];
        GlobalTensor<T> tmpSrcTensor = srcTensor[offset];

        DataCopyGmNDToL1<T>(tmpDstTensor, tmpSrcTensor, copyRowCnt, shape.copyRowNumAlign, dValue, srcDValue);
        copyFinishRowCnt += copyRowCnt;
        curS2Idx += copyRowCnt;
    }
}

struct RunInfo {
    uint32_t loop = 0;
    uint32_t cmpLoop = 0; // 用于判断取 用于merge的4块GM 中的哪一块
    uint32_t bIdx = 0;
    uint32_t gIdx = 0;
    uint32_t s1Idx = 0;
    uint32_t s2Idx = 0;
    uint32_t relativeS2Idx = 0;
    uint32_t bn2IdxInCurCore = 0;
    uint32_t curSInnerLoopTimes = 0;
    uint64_t tndBIdxOffsetForQ = 0;
    uint64_t tndBIdxOffsetForKV = 0;
    uint64_t tensorAOffset = 0;
    uint64_t tensorBOffset = 0;
    uint64_t tensorCmpBOffset = 0;
    uint64_t attenOutOffset = 0;
    uint64_t attenMaskOffset = 0;
    uint64_t topKBaseOffset = 0;
    uint32_t actualSingleProcessSInnerSize = 0;
    uint32_t actualSingleProcessSInnerSizeAlign = 0;
    bool isFirstSInnerLoop = false;
    uint32_t s2BatchOffset = 0;
    uint32_t gSize = 0;
    uint32_t s1Size = 0;
    uint32_t s2Size = 0;
    uint32_t mSize = 0;
    uint32_t mSizeV = 0;
    uint32_t mSizeVStart = 0;
    uint32_t tndIsS2SplitCore = 0;
    uint32_t tndCoreStartKVSplitPos = 0;
    bool isBmm2Output = false;
    bool isValid = false;

    static constexpr uint32_t n2Idx = 0;
    uint64_t actS1Size = 1;
    uint64_t actS2SizeOri = 0ULL;
    uint32_t gS1Idx = 0;
    uint64_t actS2Size = 1;
    uint64_t actOriS2Size = 1;
    uint32_t actMBaseSize = 0;
    bool isLastS2Loop = 0;
    int32_t nextTokensPerBatch = 0;
    int64_t threshold = 0;
    uint32_t curTopKIdx = 0;
    uint64_t curOffsetInSparseBlock = 0;
    bool isOri = true; // 判断当前块是在Ori部分还是Cmp部分
    uint64_t s2StartPoint = 0;
    int64_t cmpS2IdLimit = 0;
    int32_t v0S2DealSize = 0;
    int32_t v0S2Start = 0;
};

struct ConstInfo {
    // CUBE与VEC核间同步的模式
    static constexpr uint32_t SAS_SYNC_MODE2 = 2;
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

    // preLoad的总次数
    uint32_t preLoadNum = 0U;
    uint32_t nBufferMBaseSize = 0U;
    // CUBE和VEC的核间同步EventID
    uint32_t syncV0C1 = 0U;
    uint32_t syncC1V1 = 0U;
    uint32_t syncV1C2 = 0U;
    uint32_t syncC2V2 = 0U;

    uint32_t mmResUbSize = 0U;   // Matmul1输出结果GM上的大小
    uint32_t vec1ResUbSize = 0U; // Vector1输出结果GM上的大小
    uint32_t bmm2ResUbSize = 0U; // Matmul2输出结果GM上的大小
    uint64_t batchSize = 0ULL;
    uint64_t gSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kvHeadNum = 0;
    uint64_t headDim = 0;
    uint64_t kvSeqSize = 0ULL;    // kv最大S长度
    uint64_t qSeqSize = 1ULL;     // q最大S长度
    int64_t kvCacheBlockSize = 0; // PA场景的block size
    uint64_t paCmpBlockSize = 0;
    uint64_t paOriBlockSize = 0;
    int64_t orikvCacheBlockSize = 0;
    int64_t cmpkvCacheBlockSize = 0;
    uint32_t oriMaxBlockNumPerBatch = 0; // PA场景的最大单batch block number
    uint32_t cmpMaxBlockNumPerBatch = 0;
    uint32_t splitKVNum = 0U; // S2核间切分的切分份数
    SAS_LAYOUT outputLayout;  // 输出的Transpose格式
    uint32_t oriMaskMode = 0;
    uint32_t cmpMaskMode = 0;
    bool needInit = false;
    uint32_t templateMode = 0;

    // FlashDecoding
    uint32_t actualCombineLoopSize = 0U; // FlashDecoding场景, S2在核间切分的最大份数
    uint64_t combineLseOffset = 0ULL;
    uint64_t combineAccumOutOffset = 0ULL;

    uint32_t actualLenDimsQ = 0U;  // query的actualSeqLength 的维度
    uint32_t actualLenDimsKV = 0U; // KV 的actualSeqLength 的维度

    // TND
    uint32_t s2Start = 0U; // TND场景下，S2的起始位置
    uint32_t s2End = 0U;   // 单核TND场景下S2循环index上限

    uint32_t bN2Start = 0U;
    uint32_t bN2End = 0U;
    uint32_t gS1Start = 0U;
    uint32_t gS1End = 0U;

    uint32_t tndFDCoreArrLen = 0U;     // TNDFlashDecoding相关分核信息array的长度
    uint32_t coreStartKVSplitPos = 0U; // TNDFlashDecoding kv起始位置

    uint32_t mBaseSize = 1ULL;
    uint32_t s2BaseSize = 1ULL;

    // sparse attr
    int64_t sparseBlockSize = 0;
    uint32_t sparseBlockCount = 0;

    // cmp attr
    int64_t cmpRatio = 0;

    // win
    int32_t oriWinRight = 0;
    int32_t oriWinLeft = 128;
};

struct MSplitInfo {
    uint32_t nBufferIdx = 0U;
    uint32_t nBufferStartM = 0U;
    uint32_t nBufferDealM = 0U;
    uint32_t vecStartM = 0U;
    uint32_t vecDealM = 0U;
};
} // namespace SASKernel
#endif // SPARSE_ATTN_SHAREDKV_COMMON_H