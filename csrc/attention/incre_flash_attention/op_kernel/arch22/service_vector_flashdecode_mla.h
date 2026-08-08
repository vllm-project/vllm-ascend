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
 * \file service_vector_flashdecode_mla.h
 * \brief
 */
#ifndef SERVICE_VECTOR_FLASHDECODE_MLA_H
#define SERVICE_VECTOR_FLASHDECODE_MLA_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../ifa_public_define.h"

namespace AttentionCommonFlashDecode {
    constexpr SoftmaxConfig FIA_SOFTMAX_FLASHV2_CFG = {false};
    constexpr SoftmaxConfig FIA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC = {false, 0, 0, SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};
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
    struct ConstInfoFD {
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

        // preLoad的总次数
        uint32_t preLoadNum = 0U;
        uint64_t batchSize = 0UL;
        uint64_t gSize = 0UL;
        uint64_t qHeadNum = 0UL;
        uint64_t kvHeadNum;
        uint64_t headDim;

        uint64_t headDimAlign;
        uint64_t qSeqSize = 1UL;         // q最大S长度
        LAYOUT outputLayout;             // 输出的Transpose格式

        uint32_t mBaseSize = 1UL;
    };
    struct FusedTransposeInfoFD {
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
    template <LAYOUT LAYOUT_T>
    __aicore__ inline void GetGS1Idx(uint32_t gS1Idx, uint32_t &gIdx, uint32_t &s1Idx, AttentionCommonFlashDecode::ConstInfoFD &constInfo)
    {
        // GS1
        if constexpr (LAYOUT_T == LAYOUT::BNSD) {
            gIdx = gS1Idx / constInfo.qSeqSize;
            s1Idx = gS1Idx % constInfo.qSeqSize;
        } else {
            // S1G
            s1Idx = gS1Idx / constInfo.gSize;
            gIdx = gS1Idx % constInfo.gSize;
        }
    }
}

using namespace AttentionCommonFlashDecode;
using namespace AscendC;
using AscendC::AIC;
using AscendC::AIV;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::SetFlag;
using AscendC::ShapeInfo;
using AscendC::SoftmaxConfig;
using AscendC::WaitFlag;
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

struct TaskInfo {
    uint32_t bIdx;
    uint32_t gS1Idx;
    uint32_t actualCombineLoopSize;
};

namespace FaBaseVectorFlashDecode {
// BLOCK和REPEAT的字节数
constexpr uint64_t BYTE_BLOCK = 32UL;
constexpr uint32_t REPEAT_BLOCK_BYTE = 256U;
// BLOCK和REPEAT的FP32元素数
constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);
constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);
// repeat stride不能超过256
constexpr uint32_t REPEATE_STRIDE_UP_BOUND = 256U;
static constexpr uint64_t HEAD_DIM = 512ULL;
template <typename T>
__aicore__ inline void RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                               uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t repeatElementNum = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockElementNum = FP32_BLOCK_ELEMENT_NUM;

    if constexpr (std::is_same<T, half>::value) {
        // 此限制由于每个repeat至多连续读取256B数据
        repeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2; // 2:此场景元素个数翻倍
        blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;   // 2:此场景元素个数翻倍
    }

    // 每次只能连续读取256B的数据进行计算，故每次只能处理256B/sizeof(dType)
    // 列方向分dLoop次，每次处理8列数据
    uint32_t dLoop = actualColumnCount / repeatElementNum;
    uint32_t dRemain = actualColumnCount % repeatElementNum;
    // REPEATE_STRIDE_UP_BOUND=256， 此限制由于src0RepStride数据类型为uint8之多256个datablock间距
    if (columnCount < REPEATE_STRIDE_UP_BOUND * blockElementNum) {
        BinaryRepeatParams repeatParams;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0RepStride = columnCount / blockElementNum;
        repeatParams.src1RepStride = 1;
        repeatParams.dstRepStride = columnCount / blockElementNum;

        // 如果以列为repeat所处理的次数小于行处理次数，则以列方式处理。反之则以行进行repeat处理
        if (dLoop <= dealRowCount) {
            int64_t offset = 0;
            for (uint32_t i = 0; i < dLoop; i++) {
                Mul(dstUb[offset], src0Ub[offset], src1Ub, repeatElementNum, dealRowCount, repeatParams);
                offset += repeatElementNum;
            }
        } else {
            BinaryRepeatParams columnRepeatParams;
            columnRepeatParams.src0BlkStride = 1;
            columnRepeatParams.src1BlkStride = 0;
            columnRepeatParams.dstBlkStride = 1;
            columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            columnRepeatParams.src1RepStride = 0;
            columnRepeatParams.dstRepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            for (uint32_t i = 0; i < dealRowCount; i++) {
                Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum], repeatElementNum,
                    dLoop, columnRepeatParams);
            }
        }

        // 最后一次完成[dealRowCount, dRemain] * [dealRowCount, blockElementNum] 只计算有效部分
        if (dRemain > 0) {
            Mul(dstUb[dLoop * repeatElementNum], src0Ub[dLoop * repeatElementNum], src1Ub, dRemain, dealRowCount,
                repeatParams);
        }
    } else {
        BinaryRepeatParams repeatParams;
        repeatParams.src0RepStride = 8; // 每个repeat为256B数据，正好8个datablock
        repeatParams.src0BlkStride = 1;
        repeatParams.src1RepStride = 0;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = 8; // 每个repeat为256B数据，正好8个datablock
        repeatParams.dstBlkStride = 1;
        // 每次计算一行，共计算dealRowCount行
        for (uint32_t i = 0; i < dealRowCount; i++) {
            // 计算一行中的dLoop个repeat, 每个repeat计算256/block_size 个data_block
            Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum], repeatElementNum, dLoop,
                repeatParams);
            //  计算一行中的尾块
            if (dRemain > 0) {
                Mul(dstUb[i * columnCount + dLoop * repeatElementNum],
                    src0Ub[i * columnCount + dLoop * repeatElementNum], src1Ub[i * blockElementNum], dRemain, 1,
                    repeatParams);
            }
        }
    }
}

__aicore__ inline void MatDivsVecFd(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                               uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsDiv;
    repeatParamsDiv.src0BlkStride = 1;
    repeatParamsDiv.src1BlkStride = 1;
    repeatParamsDiv.dstBlkStride = 1;
    repeatParamsDiv.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsDiv.src1RepStride = 0;
    repeatParamsDiv.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t columnRepeatCount = dLoop;
    int64_t offset = 0;
    for (uint32_t i = 0; i < dLoop; i++) {
        Div(dstUb[offset], src0Ub[offset], src1Ub[offset], dtypeMask, dealRowCount, repeatParamsDiv);
        offset += dtypeMask;
    }

    if (dRemain > 0) {
        Div(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub[dLoop * dtypeMask], dRemain, dealRowCount, repeatParamsDiv);
    }
}

__aicore__ inline void RowSubFd(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                               uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsSub;
    repeatParamsSub.src0BlkStride = 1;
    repeatParamsSub.src1BlkStride = 1;
    repeatParamsSub.dstBlkStride = 1;
    repeatParamsSub.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsSub.src1RepStride = 0;
    repeatParamsSub.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t columnRepeatCount = dLoop;
    int64_t offset = 0;
    for (uint32_t i = 0; i < dLoop; i++) {
        Sub(dstUb[offset], src0Ub[offset], src1Ub[offset], dtypeMask, dealRowCount, repeatParamsSub);
        offset += dtypeMask;
    }

    if (dRemain > 0) {
        Sub(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub[dLoop * dtypeMask], dRemain, dealRowCount, repeatParamsSub);
    }
}

__aicore__ inline void ColMaxFd(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                               uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsMax;
    repeatParamsMax.src0BlkStride = 1;
    repeatParamsMax.src1BlkStride = 1;
    repeatParamsMax.dstBlkStride = 1;
    repeatParamsMax.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsMax.src1RepStride = 0;
    repeatParamsMax.dstRepStride = 0;
    uint32_t columnRepeatCount = dLoop;
    int64_t offset = 0;
    for (uint32_t i = 0; i < dLoop; i++) {
        Max(dstUb[offset], src0Ub[offset], src1Ub[offset], dtypeMask, dealRowCount, repeatParamsMax);
        offset += dtypeMask;
    }

    if (dRemain > 0) {
        Max(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub[dLoop * dtypeMask], dRemain, dealRowCount, repeatParamsMax);
    }
}

__aicore__ inline void ColAddFd(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                               uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsAdd;
    repeatParamsAdd.src0BlkStride = 1;
    repeatParamsAdd.src1BlkStride = 1;
    repeatParamsAdd.dstBlkStride = 1;
    repeatParamsAdd.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsAdd.src1RepStride = 0;
    repeatParamsAdd.dstRepStride = 0;
    uint32_t columnRepeatCount = dLoop;
    int64_t offset = 0;
    for (uint32_t i = 0; i < dLoop; i++) {
        Add(dstUb[offset], src0Ub[offset], src1Ub[offset], dtypeMask, dealRowCount, repeatParamsAdd);
        offset += dtypeMask;
    }

    if (dRemain > 0) {
        Add(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub[dLoop * dtypeMask], dRemain, dealRowCount, repeatParamsAdd);
    }
}

template <LAYOUT LAYOUT_T, typename OUT_T>
__aicore__ inline void Bmm2DataCopyOutNBSDMTilingFd(LocalTensor<OUT_T> &attenOutUb, const FusedTransposeInfoFD &transInfo,
                                                  const AttentionCommonFlashDecode::ConstInfoFD &constInfo,
                                                  GlobalTensor<uint64_t> &actualSeqLengthsGmQ,
                                                  GlobalTensor<OUT_T> &attentionOutGm)
{
    uint32_t tSize = constInfo.batchSize * constInfo.qSeqSize;
    uint32_t tBase = transInfo.bIdx * constInfo.qSeqSize;
    if constexpr (LAYOUT_T == LAYOUT::TND) {
        tSize = actualSeqLengthsGmQ.GetValue(constInfo.batchSize - 1);
        tBase = transInfo.bIdx == 0 ? 0 : actualSeqLengthsGmQ.GetValue(transInfo.bIdx - 1);
    }

    uint32_t s1Idx = transInfo.s1StartIdx;
    int64_t attenOutUbOffset = 0;
    for (int i = 0; i < transInfo.s1Count; i++) {
        uint32_t gIdx = 0; // 中间块
        uint32_t gCountOneS1 = constInfo.gSize;
        if (i == 0) { // 首块
            gIdx = transInfo.gStartIdx;
            gCountOneS1 = (constInfo.gSize - transInfo.gStartIdx) < transInfo.gCount ?
                              (constInfo.gSize - transInfo.gStartIdx) :
                              transInfo.gCount;
        } else if (i == transInfo.s1Count - 1) { // 尾块
            gIdx = 0;
            gCountOneS1 = transInfo.gEndIdx + 1;
        }
        int64_t attenOutOffset = transInfo.n2Idx * constInfo.gSize * tSize * HEAD_DIM + // N2轴的偏移
                                  gIdx * tSize * HEAD_DIM +                              // G轴的偏移
                                  tBase * HEAD_DIM +                                     // B轴的偏移
                                  s1Idx * HEAD_DIM;                                      // S1轴的偏移
        bool dstStrideFlag = ((tSize - 1) * HEAD_DIM * sizeof(OUT_T) / 32U) > UINT16_MAX ? 1 : 0;
        if (dstStrideFlag) {
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = gCountOneS1;
            dataCopyParams.blockLen = HEAD_DIM * sizeof(OUT_T);                // 一个D的大小
            dataCopyParams.srcStride = 0;                                     // 连读
            dataCopyParams.dstStride = (tSize - 1) * HEAD_DIM * sizeof(OUT_T); // 跳写
            DataCopyPad(attentionOutGm[attenOutOffset], attenOutUb[attenOutUbOffset], dataCopyParams);
        } else {
            DataCopyParams dataCopyParams;
            dataCopyParams.blockCount = gCountOneS1;
            dataCopyParams.blockLen = HEAD_DIM * sizeof(OUT_T) / 32U;                // 一个D的大小
            dataCopyParams.srcStride = 0;                                           // 连读
            dataCopyParams.dstStride = (tSize - 1) * HEAD_DIM * sizeof(OUT_T) / 32U; // 跳写
            DataCopy(attentionOutGm[attenOutOffset], attenOutUb[attenOutUbOffset], dataCopyParams);
        }
        s1Idx++;
        attenOutUbOffset += gCountOneS1 * HEAD_DIM;
    }
}

template <typename OUT_T>
__aicore__ inline void Bmm2DataCopyOutNBSDGTilingFd(LocalTensor<OUT_T> &attenOutUb, const FusedTransposeInfoFD &transInfo,
                                                  const AttentionCommonFlashDecode::ConstInfoFD &constInfo, GlobalTensor<OUT_T> &attentionOutGm)
{
    bool hasHeadBlock = transInfo.s1StartIdx != 0;
    bool hasTailBlock = (transInfo.s1EndIdx + 1) != constInfo.qSeqSize;
    int64_t attenOutUbOffset = 0;
    if (hasHeadBlock) { // 头块单独一条DataCopy指令
        DataCopyParams dataCopyParamsHead;
        dataCopyParamsHead.blockCount = 1;
        dataCopyParamsHead.blockLen = (constInfo.qSeqSize - transInfo.s1StartIdx) * HEAD_DIM * sizeof(OUT_T) / 32U;
        dataCopyParamsHead.srcStride = 0;
        dataCopyParamsHead.dstStride = 0; // blockCount = 1 无所谓跳写
        int64_t attenOutOffset =
            transInfo.n2Idx * constInfo.gSize * constInfo.batchSize * constInfo.qSeqSize * HEAD_DIM + // N2轴的偏移
            transInfo.gStartIdx * constInfo.batchSize * constInfo.qSeqSize * HEAD_DIM +               // G轴的偏移
            transInfo.bIdx * constInfo.qSeqSize * HEAD_DIM +                                          // B轴的偏移
            transInfo.s1StartIdx * HEAD_DIM;                                                          // S1轴的偏移
        DataCopy(attentionOutGm[attenOutOffset], attenOutUb, dataCopyParamsHead);
        attenOutUbOffset += (constInfo.qSeqSize - transInfo.s1StartIdx) * HEAD_DIM;
    }
    // 中间块DataCopy指令
    int64_t attenOutOffset =
        transInfo.n2Idx * constInfo.gSize * constInfo.batchSize * constInfo.qSeqSize * HEAD_DIM + // N2轴的偏移
        (transInfo.gStartIdx + static_cast<uint32_t>(hasHeadBlock)) * constInfo.batchSize * constInfo.qSeqSize *
            HEAD_DIM +                                  // G轴的偏移
        transInfo.bIdx * constInfo.qSeqSize * HEAD_DIM; // B轴的偏移
    bool dstStrideFlag =
        ((constInfo.batchSize * constInfo.qSeqSize - constInfo.qSeqSize) * HEAD_DIM * sizeof(OUT_T) / 32U) > UINT16_MAX ?
            1 :
            0;
    if (dstStrideFlag) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount =
            transInfo.gCount - static_cast<uint32_t>(hasHeadBlock) - static_cast<uint32_t>(hasTailBlock); // 处理多少个G
        dataCopyParams.blockLen = constInfo.qSeqSize * HEAD_DIM * sizeof(OUT_T); // 一个S1*D的大小
        dataCopyParams.srcStride = 0;                                           // 连读
        dataCopyParams.dstStride =
            (constInfo.batchSize * constInfo.qSeqSize - constInfo.qSeqSize) * HEAD_DIM * sizeof(OUT_T); // 跳写
        DataCopyPad(attentionOutGm[attenOutOffset], attenOutUb[attenOutUbOffset], dataCopyParams);
        attenOutUbOffset += dataCopyParams.blockCount * (constInfo.qSeqSize * HEAD_DIM);
    } else {
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount =
            transInfo.gCount - static_cast<uint32_t>(hasHeadBlock) - static_cast<uint32_t>(hasTailBlock); // 处理多少个G
        dataCopyParams.blockLen = constInfo.qSeqSize * HEAD_DIM * sizeof(OUT_T) / 32U; // 一个S1*D的大小
        dataCopyParams.srcStride = 0;                                                 // 连读
        dataCopyParams.dstStride =
            (constInfo.batchSize * constInfo.qSeqSize - constInfo.qSeqSize) * HEAD_DIM * sizeof(OUT_T) / 32U; // 跳写
        DataCopy(attentionOutGm[attenOutOffset], attenOutUb[attenOutUbOffset], dataCopyParams);
        attenOutUbOffset += dataCopyParams.blockCount * (constInfo.qSeqSize * HEAD_DIM);
    }
    if (hasTailBlock) { // 尾块单独一条DataCopy指令
        DataCopyParams dataCopyParamsTail;
        dataCopyParamsTail.blockCount = 1;
        dataCopyParamsTail.blockLen = (transInfo.s1EndIdx + 1) * HEAD_DIM * sizeof(OUT_T) / 32U;
        dataCopyParamsTail.srcStride = 0;
        dataCopyParamsTail.dstStride = 0; // blockCount = 1 无所谓跳写
        int64_t attenOutOffset =
            transInfo.n2Idx * constInfo.gSize * constInfo.batchSize * constInfo.qSeqSize * HEAD_DIM + // N2轴的偏移
            (transInfo.gStartIdx + transInfo.gCount - 1) * constInfo.batchSize * constInfo.qSeqSize *
                HEAD_DIM +                                  // G轴的偏移
            transInfo.bIdx * constInfo.qSeqSize * HEAD_DIM; // B轴的偏移
        DataCopy(attentionOutGm[attenOutOffset], attenOutUb[attenOutUbOffset], dataCopyParamsTail);
    }
}
}

// ServiceFlashDecode类
template <typename IFAT>
class ServiceFlashDecode {
public:
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using OUT_T = typename IFAT::outputType;
    static constexpr LAYOUT LAYOUT_T = IFAT::layout;

    __aicore__ inline void InitGlobalTensor(GlobalTensor<T> lseMaxFdGm, GlobalTensor<T> lseSumFdGm, GlobalTensor<T> accumOutGm,
            GlobalTensor<OUT_T> attentionOutGm, GlobalTensor<uint64_t> actualSeqLengthsGmQ);
    __aicore__ inline void InitParams(const AttentionCommonFlashDecode::ConstInfoFD &constInfo);
    __aicore__ inline void InitDecodeParams();
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void FlashDecode(FDparams &fd);
protected:
    __aicore__ inline void CopyAccumOutIn(LocalTensor<T> &accumOutLocal, uint32_t splitKVIndex, uint32_t startRow,
                                          uint32_t dealRowCount);
    __aicore__ inline void CopyLseIn(uint32_t startRow, uint32_t dealRowCount, int64_t baseOffset, uint32_t cntM);
    __aicore__ inline void ComputeScaleValue(LocalTensor<T> &lseExp, uint32_t startRow, uint32_t dealRowCount,
                                             uint32_t cntM);
     __aicore__ inline void Bmm2DataCopyOut(int64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb, uint32_t startRow,
                                           uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void ReduceFinalRes(LocalTensor<T> &reduceOut, LocalTensor<T> &mm2Res, LocalTensor<T> &lseLocal,
                                          uint32_t cntKV, uint32_t dealRowCount);
    __aicore__ inline void CopyFinalResOut(LocalTensor<T> &accumOutLocal, uint32_t startRow, uint32_t dealRowCount, int64_t attenOutOffset);
private:
// =================================常量区=================================
    static constexpr uint64_t SYNC_LSE_SUM_BUF1_FLAG = 6;
    static constexpr uint64_t SYNC_LSE_SUM_BUF2_FLAG = 7;
    static constexpr uint64_t SYNC_LSE_MAX_BUF1_FLAG = 8;
    static constexpr uint64_t SYNC_LSE_MAX_BUF2_FLAG = 9;
    static constexpr uint64_t SYNC_MM2RES_BUF1_FLAG = 10;
    static constexpr uint64_t SYNC_MM2RES_BUF2_FLAG = 11;
    static constexpr uint64_t SYNC_FDOUTPUT_BUF_FLAG = 12;

    static constexpr uint32_t BLOCK_ELEMENT_NUM = FaBaseVectorFlashDecode::BYTE_BLOCK / sizeof(T);

protected:
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;
    GlobalTensor<T> accumOutGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<uint64_t> actualSeqLengthsGmQ;

    // ================================类成员变量====================================
    // aic、aiv核信息
    uint32_t blockIdx = 0U;
    AttentionCommonFlashDecode::ConstInfoFD constInfo{};
    TaskInfo taskInfo{};
private:
    // ================================FD Local Buffer区====================================
    TBuf<> fdSumBuf1;
    TBuf<> fdSumBuf2;
    TBuf<> fdMaxBuf1;
    TBuf<> fdMaxBuf2;
    TBuf<> fdLseExpBuf;
    TBuf<> fdMm2ResBuf1;
    TBuf<> fdMm2ResBuf2;
    TBuf<> fdReduceBuf;
    TBuf<> fdOutputBuf;

    TBuf<> fdLseMaxUbBuf;
    TBuf<> fdLseSumUbBuf;
};

template <typename IFAT> __aicore__ inline
void ServiceFlashDecode<IFAT>::InitGlobalTensor(GlobalTensor<T> lseMaxFdGm,
                                                        GlobalTensor<T> lseSumFdGm,
                                                        GlobalTensor<T> accumOutGm,
                                                        GlobalTensor<OUT_T> attentionOutGm,
                                                        GlobalTensor<uint64_t> actualSeqLengthsGmQ)
{
   this->lseMaxFdGm = lseMaxFdGm;
   this->lseSumFdGm = lseSumFdGm;
   this->accumOutGm = accumOutGm;
   this->attentionOutGm = attentionOutGm;
   this->actualSeqLengthsGmQ = actualSeqLengthsGmQ;
}

template <typename IFAT> __aicore__ inline
void ServiceFlashDecode<IFAT>::InitParams(const AttentionCommonFlashDecode::ConstInfoFD &constInfo)
{
   this->constInfo = constInfo;
}


template <typename IFAT>__aicore__ inline
void ServiceFlashDecode<IFAT>::InitDecodeParams()
{
    this->blockIdx = GetBlockIdx();
}

template <typename IFAT> __aicore__ inline
void ServiceFlashDecode<IFAT>::InitBuffers(TPipe *pipe)
{
    if ASCEND_IS_AIV {
        pipe->Reset();
        pipe->InitBuffer(fdSumBuf1, AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_4K + AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_2K);
        pipe->InitBuffer(fdSumBuf2, AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_4K + AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_2K);
        pipe->InitBuffer(fdMaxBuf1, AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_4K + AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_2K);
        pipe->InitBuffer(fdMaxBuf2, AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_4K + AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_2K);
        pipe->InitBuffer(fdLseExpBuf, AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_4K + AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_2K);
        pipe->InitBuffer(fdMm2ResBuf1, AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_16K);
        pipe->InitBuffer(fdMm2ResBuf2, AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_16K);
        pipe->InitBuffer(fdReduceBuf, AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_16K);
        pipe->InitBuffer(fdOutputBuf, AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_16K);
        pipe->InitBuffer(fdLseMaxUbBuf, AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_256B);
        pipe->InitBuffer(fdLseSumUbBuf, AttentionCommonFlashDecode::ConstInfoFD::BUFFER_SIZE_BYTE_256B);
    }
}

template <typename IFAT> __aicore__ inline
void ServiceFlashDecode<IFAT>::AllocEventID()
{
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_SUM_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_SUM_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
}

template <typename IFAT> __aicore__ inline
void ServiceFlashDecode<IFAT>::FreeEventID()
{
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_SUM_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_SUM_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
}

template <typename IFAT> __aicore__ inline
void ServiceFlashDecode<IFAT>::CopyAccumOutIn(LocalTensor<T> &accumOutLocal, uint32_t splitKVIndex,
    uint32_t startRow, uint32_t dealRowCount)
{
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<T> copyInPadParams;
    copyInParams.blockCount = dealRowCount;
    copyInParams.blockLen = constInfo.headDim * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = (constInfo.headDimAlign - constInfo.headDim) / BLOCK_ELEMENT_NUM;

    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = (constInfo.headDimAlign - constInfo.headDim) % BLOCK_ELEMENT_NUM;
    copyInPadParams.paddingValue = 0;
    int64_t combineAccumOutOffset = startRow * constInfo.headDim +                          // taskoffset + g轴offset
                                      splitKVIndex * constInfo.mBaseSize * constInfo.headDim; // 份数offset
    DataCopyPad(accumOutLocal, accumOutGm[combineAccumOutOffset], copyInParams, copyInPadParams);
}

template <typename IFAT> __aicore__ inline
void ServiceFlashDecode<IFAT>::CopyLseIn(uint32_t startRow,
    uint32_t dealRowCount, int64_t baseOffset, uint32_t cntM)
{
    // 双缓冲区域
    LocalTensor<T> lseSum = cntM % 2 == 0 ? fdSumBuf1.Get<T>() : fdSumBuf2.Get<T>();
    LocalTensor<T> lseMax = cntM % 2 == 0 ? fdMaxBuf1.Get<T>() : fdMaxBuf2.Get<T>();

    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_SUM_BUF1_FLAG + cntM % 2);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_BUF1_FLAG + cntM % 2);

    int64_t combineLseOffset = (baseOffset + startRow) * FaBaseVectorFlashDecode::FP32_BLOCK_ELEMENT_NUM;
    int64_t combineLoopOffset = constInfo.mBaseSize * FaBaseVectorFlashDecode::FP32_BLOCK_ELEMENT_NUM;
    uint64_t dealRowCountAlign = dealRowCount * FaBaseVectorFlashDecode::FP32_BLOCK_ELEMENT_NUM;
    for (uint32_t i = 0; i < taskInfo.actualCombineLoopSize; i++) {
        DataCopy(lseSum[i * dealRowCountAlign], lseSumFdGm[combineLseOffset + i * combineLoopOffset],
                 dealRowCountAlign); // 份数offset
        DataCopy(lseMax[i * dealRowCountAlign], lseMaxFdGm[combineLseOffset + i * combineLoopOffset],
                 dealRowCountAlign);
    }
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_SUM_BUF1_FLAG + cntM % 2);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_MAX_BUF1_FLAG + cntM % 2);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_SUM_BUF1_FLAG + cntM % 2);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_MAX_BUF1_FLAG + cntM % 2);
}

template <typename IFAT> __aicore__ inline void
ServiceFlashDecode<IFAT>::ComputeScaleValue(LocalTensor<T> &lseExp,
                                                    uint32_t startRow,
                                                    uint32_t dealRowCount,
                                                    uint32_t cntM)
{
    LocalTensor<T> lseSum = cntM % 2 == 0 ? fdSumBuf1.Get<T>() : fdSumBuf2.Get<T>();
    LocalTensor<T> lseMax = cntM % 2 == 0 ? fdMaxBuf1.Get<T>() : fdMaxBuf2.Get<T>();

    LocalTensor<T> lseMaxUb = fdLseMaxUbBuf.Get<T>();
    LocalTensor<T> lseSumUb = fdLseSumUbBuf.Get<T>();
    uint64_t dealRowCountAlign = dealRowCount * FaBaseVectorFlashDecode::FP32_BLOCK_ELEMENT_NUM;
    Duplicate(lseMaxUb, -AttentionCommonFlashDecode::ConstInfoFD::FLOAT_MAX, dealRowCountAlign);
    Duplicate(lseSumUb, AttentionCommonFlashDecode::ConstInfoFD::FLOAT_ZERO, dealRowCountAlign);
    pipe_barrier(PIPE_V);

    FaBaseVectorFlashDecode::ColMaxFd(lseMaxUb, lseMax, lseMaxUb, taskInfo.actualCombineLoopSize, dealRowCountAlign, dealRowCountAlign);
    pipe_barrier(PIPE_V);

    FaBaseVectorFlashDecode::RowSubFd(lseExp, lseMax, lseMaxUb, taskInfo.actualCombineLoopSize, dealRowCountAlign, dealRowCountAlign);
    pipe_barrier(PIPE_V);

    Exp(lseExp, lseExp, taskInfo.actualCombineLoopSize * dealRowCountAlign);
    pipe_barrier(PIPE_V);

    Mul(lseExp, lseSum, lseExp, taskInfo.actualCombineLoopSize * dealRowCountAlign);
    pipe_barrier(PIPE_V);

    FaBaseVectorFlashDecode::ColAddFd(lseSumUb, lseExp, lseSumUb, taskInfo.actualCombineLoopSize, dealRowCountAlign, dealRowCountAlign);
    pipe_barrier(PIPE_V);

    FaBaseVectorFlashDecode::MatDivsVecFd(lseExp, lseExp, lseSumUb, taskInfo.actualCombineLoopSize, dealRowCountAlign, dealRowCountAlign);
    pipe_barrier(PIPE_V);

    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_SUM_BUF1_FLAG + cntM % 2);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_BUF1_FLAG + cntM % 2);
}


template <typename IFAT>__aicore__ inline
void ServiceFlashDecode<IFAT>::Bmm2DataCopyOut(int64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb,
                                                               uint32_t startRow, uint32_t dealRowCount,
                                                               uint32_t columnCount, uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (FaBaseVectorFlashDecode::BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[attenOutOffset + startRow * actualColumnCount], attenOutUb,
                dataCopyParams);
}

template <typename IFAT>__aicore__ inline
void ServiceFlashDecode<IFAT>::ReduceFinalRes(LocalTensor<T> &reduceOut,
                                                      LocalTensor<T> &mm2Res,
                                                      LocalTensor<T> &lseLocal,
                                                      uint32_t cntKV,
                                                      uint32_t dealRowCount)
{
    uint32_t dealRowCountAlign = dealRowCount * FaBaseVectorFlashDecode::FP32_BLOCK_ELEMENT_NUM;
    LocalTensor<T> tmpRst =
        cntKV == 0 ? reduceOut : mm2Res; // 第一次mul结果直接写入reduceOut，否则在mm2Res原地进行mul，再加到reduceOut

    FaBaseVectorFlashDecode::RowMuls(tmpRst, mm2Res, lseLocal[cntKV * dealRowCountAlign], dealRowCount, constInfo.headDimAlign, constInfo.headDim);

    if (cntKV != 0) {
        pipe_barrier(PIPE_V);
        Add(reduceOut, reduceOut, tmpRst, dealRowCount * constInfo.headDimAlign);
        pipe_barrier(PIPE_V);
    }
}

template <typename IFAT> __aicore__ inline
void ServiceFlashDecode<IFAT>::CopyFinalResOut(LocalTensor<T> &accumOutLocal,
                                                       uint32_t startRow,
                                                       uint32_t dealRowCount,
                                                       int64_t attenOutOffset)
{
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = fdOutputBuf.Get<OUT_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
    uint32_t shapeArray[] = {dealRowCount, (uint32_t)constInfo.headDim};
    tmpBmm2ResCastTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
        Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_RINT, dealRowCount * constInfo.headDimAlign);
    } else {
        Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_ROUND, dealRowCount * constInfo.headDimAlign);
    }

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
    if (constInfo.outputLayout == LAYOUT::NBSD || constInfo.outputLayout == LAYOUT::NTD) {
        FusedTransposeInfoFD transInfo;
        transInfo.n2Idx = 0;
        transInfo.bIdx = taskInfo.bIdx;
        auto gS1StartIdx = taskInfo.gS1Idx + startRow;
        auto gS1EndIdx = gS1StartIdx + dealRowCount - 1;
        GetGS1Idx<LAYOUT_T>(gS1StartIdx, transInfo.gStartIdx, transInfo.s1StartIdx, constInfo);
        GetGS1Idx<LAYOUT_T>(gS1EndIdx, transInfo.gEndIdx, transInfo.s1EndIdx, constInfo);
        if constexpr (LAYOUT_T == LAYOUT::BNSD) {
            transInfo.gCount = transInfo.gEndIdx - transInfo.gStartIdx + 1;
            FaBaseVectorFlashDecode::Bmm2DataCopyOutNBSDGTilingFd(tmpBmm2ResCastTensor, transInfo, constInfo, attentionOutGm);
        } else {
            transInfo.s1Count = transInfo.s1EndIdx - transInfo.s1StartIdx + 1;
            transInfo.gCount = dealRowCount;
            FaBaseVectorFlashDecode::Bmm2DataCopyOutNBSDMTilingFd<LAYOUT_T, OUT_T>(tmpBmm2ResCastTensor, transInfo, constInfo,
                                                                        actualSeqLengthsGmQ, attentionOutGm);
        }
    } else {
        Bmm2DataCopyOut(attenOutOffset, tmpBmm2ResCastTensor, startRow, dealRowCount, constInfo.headDimAlign, constInfo.headDim);
    }
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
}


template <typename IFAT> __aicore__ inline void
ServiceFlashDecode<IFAT>::FlashDecode(FDparams &fd)
{
    if (blockIdx >= fd.usedVecNumOfFd) {
        return;
    }
    uint32_t fdTaskPrevEnd = (blockIdx > 0) ? fd.gS1IdxEndOfFdHead[blockIdx - 1] : 0U; // 上一个核末尾是第几个规约
    uint32_t fdS1gOuterMPrevEnd =
        (blockIdx > 0) ? fd.gS1IdxEndOfFdHeadSplit[blockIdx - 1] : 0U; //上一个核末尾是该规约的第几个base行
    uint32_t fdTaskEnd = fd.gS1IdxEndOfFdHead[blockIdx];                 // 当前核的末尾是第几个规约任务
    uint32_t fdS1gOuterMEnd = fd.gS1IdxEndOfFdHeadSplit[blockIdx]; // 当前核的末尾是该规约的第几个base行
    uint32_t tmpFdS1gOuterMStart = (blockIdx > 0) ? fdS1gOuterMPrevEnd + 1 : 0U; // 当前核从第几个base行开始
    uint32_t tmpFdS1gOuterMEnd = 0U;
    uint32_t reduceGlobalLoop = 0U;
    uint32_t reduceMLoop = 0U;

    // 并不是每个核都会处理同一规约任务的不同行，以下是一个泛化性说明，自己手动推一下即可
    // 首先针对每个核遍历一下它需要处理的规约任务
    for (uint32_t fdTaskId = fdTaskPrevEnd; fdTaskId <= fdTaskEnd; fdTaskId++) {
        tmpFdS1gOuterMEnd = (fdTaskId == fdTaskEnd) ? fdS1gOuterMEnd : (fd.gS1SplitNumOfFdHead[fdTaskId] - 1); //针对每个核当前处理的规约任务的尾块索引
        taskInfo.bIdx = fd.bN2IdxOfFdHead[fdTaskId]; //针对每个核当前处理的规约任务的batch索引
        taskInfo.gS1Idx = fd.gS1IdxOfFdHead[fdTaskId] * constInfo.mBaseSize; //针对每个核当前处理的规约任务的S1G索引
        taskInfo.actualCombineLoopSize = fd.s2SplitNumOfFdHead[fdTaskId]; // 当前规约任务kv方向有几份

        //统计当前处理的规约任务之前所有规约任务的块数，单位是块！！
        uint64_t combineTaskPrefixSum = 0UL;
        for (int i = 0; i < fdTaskId; i++) {
            // 计算此前规约数据的累计份数，每一份的数据大小为 kvHeadNum * constInfo.tndSgBasicSize
            // |Task0-0|Task0-1|Task0-3|Task1-0|Task1-2|...|
            combineTaskPrefixSum += fd.s2SplitNumOfFdHead[i];
        }

        int64_t curAttenOutOffset = 0;

        if constexpr (LAYOUT_T == LAYOUT::BNSD) {
            curAttenOutOffset =
                taskInfo.bIdx * constInfo.qHeadNum * constInfo.qSeqSize * constInfo.headDim +
                    taskInfo.gS1Idx * constInfo.headDim;
        } else {
            uint64_t actualSeqQPrefixSum;
            if constexpr (LAYOUT_T == LAYOUT::TND) {
                actualSeqQPrefixSum =
                    (taskInfo.bIdx <= 0) ? 0 : actualSeqLengthsGmQ.GetValue(taskInfo.bIdx - 1);
            } else { // BSND
                actualSeqQPrefixSum = (taskInfo.bIdx <= 0) ? 0 : taskInfo.bIdx * constInfo.qSeqSize;
            }
            int64_t batchOutOffset = actualSeqQPrefixSum * constInfo.qHeadNum * constInfo.headDim;
            curAttenOutOffset = batchOutOffset + taskInfo.gS1Idx * constInfo.headDim;
        }
        // 计算一个偏移，计算真实偏移量，分块大小（这里的分块是指workspace阶段沿着S1G切分）的实际行数*块数，计算当前规约任务之前，所有规约任务的偏移
        int64_t taskOffset = combineTaskPrefixSum * constInfo.kvHeadNum * constInfo.mBaseSize;
        // 第二级索引，针对每个核的正在处理的规约任务来说，遍历它的行基本块（这里是指除完8之后的）
        for (uint32_t fdS1gOuterMIdx = tmpFdS1gOuterMStart; fdS1gOuterMIdx <= tmpFdS1gOuterMEnd;
             fdS1gOuterMIdx++) { // 左闭右闭

            uint32_t actualGSplitSize = fd.gS1BaseSizeOfFd;
            // 如果处理到每一个规约任务的尾块，要考虑到尾块并不都是8的情况，所以actualGSplitSize被重新赋值为尾块里的实际行数
            if (fdS1gOuterMIdx == fd.gS1SplitNumOfFdHead[fdTaskId] - 1) {
                actualGSplitSize = fd.gS1LastPartSizeOfFdHead[fdTaskId];
            }
            // startRow被定义为8*行基本块（索引）,可以把它理解成正在处理的当前规约任务的行基本块，之前有多少总行数
            uint32_t startRow = fdS1gOuterMIdx * fd.gS1BaseSizeOfFd;

            LocalTensor<T> lseExp = fdLseExpBuf.Get<T>();
            LocalTensor<T> reduceOut = fdReduceBuf.Get<T>();
            CopyLseIn(startRow, actualGSplitSize, taskOffset, reduceMLoop);

            LocalTensor<T> mm2Res;
            for (uint32_t preLoadIdx = 0; preLoadIdx < constInfo.preLoadNum; preLoadIdx++) {
                mm2Res = (reduceGlobalLoop + preLoadIdx) % 2 == 0 ? fdMm2ResBuf1.Get<T>() : fdMm2ResBuf2.Get<T>();
                WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobalLoop + preLoadIdx) % 2);
                CopyAccumOutIn(mm2Res, preLoadIdx, taskOffset + startRow, actualGSplitSize);
                SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobalLoop + preLoadIdx) % 2);
            }

            ComputeScaleValue(lseExp, startRow, actualGSplitSize, reduceMLoop);

            for (uint32_t i = 0; i < taskInfo.actualCombineLoopSize; i++) {
                mm2Res = reduceGlobalLoop % 2 == 0 ? fdMm2ResBuf1.Get<T>() : fdMm2ResBuf2.Get<T>();
                if (i >= constInfo.preLoadNum) {
                    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + reduceGlobalLoop % 2);
                    CopyAccumOutIn(mm2Res, i, taskOffset + startRow, actualGSplitSize);
                    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + reduceGlobalLoop % 2);
                }

                WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + reduceGlobalLoop % 2);
                ReduceFinalRes(reduceOut, mm2Res, lseExp, i, actualGSplitSize);
                SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + reduceGlobalLoop % 2);
                reduceGlobalLoop += 1;
            }
            CopyFinalResOut(reduceOut, startRow, actualGSplitSize, curAttenOutOffset);
            reduceMLoop += 1;
        }
        tmpFdS1gOuterMStart = 0;
    }
}
#endif