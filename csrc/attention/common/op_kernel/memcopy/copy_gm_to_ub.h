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
 * \file copy_gm_to_ub.h
 * \brief
 */
#ifndef COPY_GM_TO_UB_H
#define COPY_GM_TO_UB_H

#include "gm_layout.h"
#include "gm_coord.h"
#include "fa_ub_tensor.h"
#include "offset_calculator_v2.h"
#include "../const_def.h"

// GM->UB
/*
 * dealRowCount: 需要拷贝的行数
 * actDataLen: 一行需要拷贝的元素数
 * srcRowStride: gm上两行数据起始位置之间间隔元素数
 * dstRowStride: ub上两行数据起始位置之间间隔元素数
 * enableLargeStride默认为false, 当srcStrideOfDataCopy超过datacopypad范围时开启
 */
template <typename T, bool enableLargeStride = false>
__aicore__ inline void CopySingleMatrixNDToND(LocalTensor<T> ubTensor, const GlobalTensor<T> gmTensor,
                                              uint32_t dealRowCount, uint32_t actDataLen, uint64_t srcRowStride,
                                              uint64_t dstRowStride)
{
    constexpr uint64_t UINT16_MAX_VALUE = 65535u;
    constexpr uint64_t UINT32_MAX_VALUE = 4294967295u;
    uint32_t blockElemNum = 32UL / sizeof(T);
    if constexpr (IsSameType<T, int4b_t>::value) {
        constexpr uint32_t HALF_SIZE_DIVISOR = 2;
        blockElemNum = blockElemNum * HALF_SIZE_DIVISOR;
        actDataLen = actDataLen / HALF_SIZE_DIVISOR;
        srcRowStride = srcRowStride / HALF_SIZE_DIVISOR;
    }
    if constexpr (enableLargeStride) {
        uint64_t srcStrideOfDataCopyPad = (srcRowStride - actDataLen) * sizeof(T);
        if (unlikely(srcStrideOfDataCopyPad > UINT32_MAX_VALUE)) {
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = 1;
            dataCopyParams.blockLen = actDataLen * sizeof(T);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;

            DataCopyPadExtParams<T> dataCopyPadParams;
            dataCopyPadParams.isPad = true;
            dataCopyPadParams.leftPadding = 0;
            dataCopyPadParams.rightPadding = (blockElemNum - (actDataLen % blockElemNum)) % blockElemNum;
            dataCopyPadParams.paddingValue = 0;

            for (uint32_t i = 0; i < dealRowCount; ++i) {
                DataCopyPad(ubTensor[i * dstRowStride], gmTensor[i * srcRowStride], dataCopyParams, dataCopyPadParams);
            }
            return;
        }
    }
    bool isPad = ((actDataLen % blockElemNum) != 0 || (srcRowStride % blockElemNum) != 0 ||
                  (dstRowStride % blockElemNum) != 0); // 判断是否32字节对齐，确定是否走datacopypad
    uint64_t srcStrideOfDataCopy = (srcRowStride - actDataLen) / blockElemNum;
    // 在有pad或srcStrideOfDataCopy不符合datacopy范围时，使用datacopypad拷贝完成
    if (unlikely(isPad || (srcStrideOfDataCopy > UINT16_MAX_VALUE))) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = static_cast<uint16_t>(dealRowCount); // 外部传入
        dataCopyParams.blockLen = actDataLen * sizeof(T);
        dataCopyParams.srcStride = (srcRowStride - actDataLen) * sizeof(T);
        dataCopyParams.dstStride = (dstRowStride - actDataLen) / blockElemNum; // 外部传入

        DataCopyPadExtParams<T> dataCopyPadParams;
        dataCopyPadParams.isPad = true;
        dataCopyPadParams.leftPadding = 0;
        dataCopyPadParams.rightPadding = (blockElemNum - (actDataLen % blockElemNum)) % blockElemNum;
        dataCopyPadParams.paddingValue = 0;
        DataCopyPad(ubTensor, gmTensor, dataCopyParams, dataCopyPadParams);
    } else { // 其他情况使用datacopy拷贝完成
        DataCopyParams repeatParams;
        repeatParams.blockCount = static_cast<uint16_t>(dealRowCount);
        repeatParams.blockLen = actDataLen / blockElemNum;
        repeatParams.srcStride = (srcRowStride - actDataLen) / blockElemNum;
        repeatParams.dstStride = (dstRowStride - actDataLen) / blockElemNum;
        DataCopy(ubTensor, gmTensor, repeatParams);
    }
}

// antiquant
//  ----------------------------------------------CopyAntiquantGmToUb--------------------------------
struct AntiqGmCoord {
    uint32_t bIdx = 0;
    uint32_t n2Idx = 0;
    uint32_t s2Idx = 0;

    uint32_t s2DealSize = 0; // actualSingleProcessSInnerSize //实际s2长度
};

template <typename T, GmFormat GM_FORMAT>
class CopyAntiquantGmToUb {
public:
    __aicore__ inline void operator()(FaUbTensor<T> &dstTensor, FaGmTensor<T, GM_FORMAT> &srcTensor,
                                      AntiqGmCoord &antiqGmCoord)
    {
        // per tensor场景在接口外部直接getvalue
        // per channel / per token
        if constexpr ((GM_FORMAT == GmFormat::ND) || (GM_FORMAT == GmFormat::BS2) || (GM_FORMAT == GmFormat::BNS2)) {
            ProcessAntiqPerChannelOrPerToken(dstTensor, srcTensor, antiqGmCoord);
        }
        // per token + PA
        else if constexpr ((GM_FORMAT == GmFormat::PA_BnBs) || (GM_FORMAT == GmFormat::PA_BnNBs) ||
                           (GM_FORMAT == GmFormat::PA_BnNBs_KS)) {
            ProcessAntiqPA(dstTensor, srcTensor, antiqGmCoord);
        }
    }

private:
    __aicore__ inline void ProcessAntiqPerChannelOrPerToken(FaUbTensor<T> &dstTensor,
                                                            FaGmTensor<T, GM_FORMAT> &srcTensor,
                                                            AntiqGmCoord &antiqGmCoord)
    {
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint64_t offset = offsetCalculator.GetOffset(antiqGmCoord.bIdx, antiqGmCoord.n2Idx, antiqGmCoord.s2Idx, 0);
        if constexpr (GM_FORMAT == GmFormat::ND) {
            CopySingleMatrixNDToND<T>(dstTensor.tensor, srcTensor.gmTensor[offset], 1, offsetCalculator.GetDimD(),
                                      offsetCalculator.GetDimD(), dstTensor.colCount);
        } else if constexpr (GM_FORMAT == GmFormat::BS2 || GM_FORMAT == GmFormat::BNS2) {
            CopySingleMatrixNDToND<T>(dstTensor.tensor, srcTensor.gmTensor[offset], 1, antiqGmCoord.s2DealSize,
                                      antiqGmCoord.s2DealSize, dstTensor.colCount);
        }
    }

    __aicore__ inline void ProcessAntiqPA(FaUbTensor<T> &dstTensor, FaGmTensor<T, GM_FORMAT> &srcTensor,
                                          AntiqGmCoord &antiqGmCoord)
    {
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;

        uint64_t dstOffset = 0;
        uint32_t copyFinishElmeCnt = 0;
        uint32_t curS2Idx = antiqGmCoord.s2Idx;

        while (copyFinishElmeCnt < antiqGmCoord.s2DealSize) {
            uint32_t copyElemCnt = offsetCalculator.GetDimBlockSize() -
                                   curS2Idx % offsetCalculator.GetDimBlockSize(); // 一次只能处理一个block
            if (copyFinishElmeCnt + copyElemCnt > antiqGmCoord.s2DealSize) {
                copyElemCnt = antiqGmCoord.s2DealSize - copyFinishElmeCnt; // 一个block未拷满
            }

            uint64_t srcOffset = offsetCalculator.GetOffset(antiqGmCoord.bIdx, antiqGmCoord.n2Idx, curS2Idx);
            CopySingleMatrixNDToND<T>(dstTensor.tensor[dstOffset], srcTensor.gmTensor[srcOffset], 1, copyElemCnt,
                                      copyElemCnt, copyElemCnt);

            dstOffset += copyElemCnt;
            copyFinishElmeCnt += copyElemCnt;
            curS2Idx += copyElemCnt;
        }
    }
};

// ----------------------------------------------CopyQueryScaleGmToUb--------------------------------
template <typename T, GmFormat GM_FORMAT>
class CopyQueryScaleGmToUb {
public:
    __aicore__ inline void operator()(FaUbTensor<T> &dstTensor, FaGmTensor<T, GM_FORMAT> &srcTensor, GmCoord &gmCoord)
    {
        if constexpr (GM_FORMAT == GmFormat::NGT) {
            ProcessGS1(dstTensor, srcTensor, gmCoord);
        }
    }

private:
__aicore__ inline void ProcessGS1(FaUbTensor<T> &dstTensor, FaGmTensor<T, GM_FORMAT> &srcTensor, GmCoord &gmCoord)
    {
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint64_t s1Size = 0;
        s1Size = offsetCalculator.actualSeqLensQParser.GetActualSeqLength(gmCoord.bIdx);
        uint32_t gIdxStart = gmCoord.gS1Idx / s1Size;
        uint32_t s1IdxStart = gmCoord.gS1Idx % s1Size;

        uint64_t queryGmbaseOffset =
            offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, gIdxStart, 0);

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1; // 外部传入
        dataCopyParams.blockLen = static_cast<uint16_t>(gmCoord.gS1DealSize) * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0; // 外部传入

        DataCopyPadExtParams<T> dataCopyPadParams;
        DataCopyPad(dstTensor.tensor, srcTensor.gmTensor[queryGmbaseOffset + s1IdxStart],
                    dataCopyParams, dataCopyPadParams);
    }
};

// ----------------------------------------------CopyKeyScaleGmToUb--------------------------------
template <typename T, GmFormat GM_FORMAT>
class CopyKeyScaleGmToUb {
public:
    __aicore__ inline void operator()(FaUbTensor<T> &dstTensor, FaGmTensor<T, GM_FORMAT> &srcTensor,
                                      AntiqGmCoord &antiqGmCoord)
    {
        if constexpr (GM_FORMAT == GmFormat::PA_BnNBs_KS) {
            ProcessAntiqPA(dstTensor, srcTensor, antiqGmCoord);
        }
    }

private:
    __aicore__ inline void ProcessAntiqPA(FaUbTensor<T> &dstTensor, FaGmTensor<T, GM_FORMAT> &srcTensor,
                                          AntiqGmCoord &antiqGmCoord)
    {
        OffsetCalculator<GM_FORMAT> &offsetCal = srcTensor.offsetCalculator;

        uint64_t dstOffset = 0;
        uint32_t copyFinishElmeCnt = 0;
        uint32_t curS2Idx = antiqGmCoord.s2Idx;

        while (copyFinishElmeCnt < antiqGmCoord.s2DealSize) {
            uint32_t copyElemCnt = offsetCal.GetDimBlockSize() -
                                   curS2Idx % offsetCal.GetDimBlockSize(); // 一次只能处理一个block
            if (copyFinishElmeCnt + copyElemCnt > antiqGmCoord.s2DealSize) {
                copyElemCnt = antiqGmCoord.s2DealSize - copyFinishElmeCnt; // 一个block未拷满
            }

            uint64_t srcOffset = offsetCal.GetOffset(antiqGmCoord.bIdx, antiqGmCoord.n2Idx, curS2Idx);
            CopyKeyScaleNDToND(dstTensor.tensor[dstOffset], srcTensor.gmTensor[srcOffset], 1, copyElemCnt,
                               copyElemCnt, copyElemCnt);

            dstOffset += copyElemCnt;
            copyFinishElmeCnt += copyElemCnt;
            curS2Idx += copyElemCnt;
        }
    }
    __aicore__ inline void CopyKeyScaleNDToND(LocalTensor<T> ubTensor, const GlobalTensor<T> gmTensor,
        uint32_t dealRowCount, uint32_t actDataLen, uint64_t srcRowStride, uint64_t dstRowStride)
    {
        constexpr uint64_t UINT32_MAX_VALUE = 4294967295u;
        constexpr uint64_t UINT16_MAX_VALUE = 65535u;
        uint32_t blockElemNum = 32UL / sizeof(T);
        if constexpr (IsSameType<T, int4b_t>::value) {
            constexpr uint32_t HALF_SIZE_DIVISOR = 2;
            blockElemNum = blockElemNum * HALF_SIZE_DIVISOR;
            actDataLen = actDataLen / HALF_SIZE_DIVISOR;
            srcRowStride = srcRowStride / HALF_SIZE_DIVISOR;
        }
        bool isPad = ((actDataLen % blockElemNum) != 0 || (srcRowStride % blockElemNum) != 0 ||
                    (dstRowStride % blockElemNum) != 0); // 判断是否32字节对齐，确定是否走datacopypad
        uint64_t srcStrideOfDataCopy = (srcRowStride - actDataLen) / blockElemNum;
        // 在有pad或srcStrideOfDataCopy不符合datacopy范围时，使用datacopypad拷贝完成
        if (unlikely(isPad || (srcStrideOfDataCopy > UINT16_MAX_VALUE))) {
            DataCopyExtParams ubCopyParams;
            ubCopyParams.blockCount = static_cast<uint16_t>(dealRowCount); // 外部传入
            ubCopyParams.blockLen = actDataLen * sizeof(T);
            ubCopyParams.srcStride = (srcRowStride - actDataLen) * sizeof(T);
            ubCopyParams.dstStride = (dstRowStride - actDataLen) / blockElemNum; // 外部传入

            DataCopyPadExtParams<T> dataCopyPadParams;
            dataCopyPadParams.isPad = true;
            dataCopyPadParams.leftPadding = 0;
            dataCopyPadParams.rightPadding = (blockElemNum - (actDataLen % blockElemNum)) % blockElemNum;
            dataCopyPadParams.paddingValue = 1;
            DataCopyPad(ubTensor, gmTensor, ubCopyParams, dataCopyPadParams);
        } else { // 其他情况使用datacopy拷贝完成
            DataCopyParams repeatParams;
            repeatParams.blockLen = actDataLen / blockElemNum;
            repeatParams.blockCount = static_cast<uint16_t>(dealRowCount);
            repeatParams.srcStride = (srcRowStride - actDataLen) / blockElemNum;
            repeatParams.dstStride = (dstRowStride - actDataLen) / blockElemNum;
            DataCopy(ubTensor, gmTensor, repeatParams);
        }
    }
};

// ----------------------------------------------CopyQueryGmToUb--------------------------------

template <typename T, GmFormat GM_FORMAT>
class CopyQueryGmToUb {
public:
    __aicore__ inline void operator()(FaUbTensor<T> &dstTensor, FaGmTensor<T, GM_FORMAT> &srcTensor, GmCoord &gmCoord)
    {
        if constexpr ((GM_FORMAT == GmFormat::BSNGD) || (GM_FORMAT == GmFormat::TNGD)) {
            ProcessS1G(dstTensor, srcTensor, gmCoord);
        } else if constexpr (GM_FORMAT == GmFormat::NGTD) {
            ProcessGS1(dstTensor, srcTensor, gmCoord);
        } else if constexpr (GM_FORMAT == GmFormat::BNGSD) {
            OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
            if (offsetCalculator.actualSeqLensQParser.GetActualLenDims() != 0) {
                ProcessGS1(dstTensor, srcTensor, gmCoord);
            } else {
                ProcessContinuous(dstTensor, srcTensor, gmCoord);
            }
        }
    }

private:
    __aicore__ inline void ProcessGS1(FaUbTensor<T> &dstTensor, FaGmTensor<T, GM_FORMAT> &srcTensor, GmCoord &gmCoord)
    {
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint64_t s1Size = 0;
        if constexpr (GmLayoutParams<GM_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND) {
            s1Size = offsetCalculator.actualSeqLensQParser.GetActualSeqLength(gmCoord.bIdx);
        } else {
            if (offsetCalculator.actualSeqLensQParser.GetActualLenDims() == 0) {
                s1Size = offsetCalculator.GetDimS1();
            } else {
                s1Size = offsetCalculator.actualSeqLensQParser.GetActualSeqLength(gmCoord.bIdx);
            }
        }
        uint32_t gIdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) / s1Size;
        uint32_t s1IdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) % s1Size;
        uint32_t gIdxStart = gmCoord.gS1Idx / s1Size;
        uint32_t s1IdxStart = gmCoord.gS1Idx % s1Size;

        uint64_t queryGmbaseOffset =
            offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, gIdxStart, 0, gmCoord.dIdx);

        // 处理 首行
        uint32_t headS1 = 0;
        if (gIdxStart == gIdxEnd) {
            headS1 = s1IdxEnd - s1IdxStart;
        } else {
            headS1 = s1Size - s1IdxStart;
        }

        CopySingleMatrixNDToND<T>(dstTensor.tensor,
                                  srcTensor.gmTensor[queryGmbaseOffset + s1IdxStart * offsetCalculator.GetDimD()],
                                  headS1, gmCoord.dDealSize, offsetCalculator.GetStrideS1(), dstTensor.colCount);

        if (gIdxEnd - gIdxStart >= 1) {
            // 处理中间块
            uint64_t gmOffset = queryGmbaseOffset + offsetCalculator.GetStrideG();
            uint32_t ubOffset = headS1 * dstTensor.colCount;
            //
            for (uint32_t i = gIdxStart + 1; i < gIdxEnd; i++) {
                CopySingleMatrixNDToND<T>(dstTensor.tensor[ubOffset], srcTensor.gmTensor[gmOffset], s1Size,
                                          gmCoord.dDealSize, offsetCalculator.GetStrideS1(), dstTensor.colCount);
                gmOffset += offsetCalculator.GetStrideG();
                ubOffset += s1Size * dstTensor.colCount;
            }

            // 处理尾块
            if (s1IdxEnd > 0) {
                CopySingleMatrixNDToND<T>(dstTensor.tensor[ubOffset], srcTensor.gmTensor[gmOffset], s1IdxEnd,
                                          gmCoord.dDealSize, offsetCalculator.GetStrideS1(), dstTensor.colCount);
            }
        }
    }

    __aicore__ inline void ProcessContinuous(FaUbTensor<T> &dstTensor, FaGmTensor<T, GM_FORMAT> &srcTensor,
                                             GmCoord &gmCoord)
    {
        // B*N2*GS1*D
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint32_t gIdxStart = gmCoord.gS1Idx / offsetCalculator.GetDimS1();
        uint32_t s1IdxStart = gmCoord.gS1Idx % offsetCalculator.GetDimS1();

        uint64_t offset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, gIdxStart, s1IdxStart, gmCoord.dIdx);
        CopySingleMatrixNDToND<T>(dstTensor.tensor, srcTensor.gmTensor[offset], gmCoord.gS1DealSize, gmCoord.dDealSize,
                                  offsetCalculator.GetStrideS1(), dstTensor.colCount);
    }

    __aicore__ inline void ProcessS1G(FaUbTensor<T> &dstTensor, FaGmTensor<T, GM_FORMAT> &srcTensor, GmCoord &gmCoord)
    {
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint32_t s1IdxStart = gmCoord.gS1Idx / offsetCalculator.GetDimG();
        uint32_t gIdxStart = gmCoord.gS1Idx % offsetCalculator.GetDimG();
        uint32_t s1IdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) / offsetCalculator.GetDimG();
        uint32_t gIdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) % offsetCalculator.GetDimG();

        uint64_t queryGmbaseOffset =
            offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, 0, s1IdxStart, gmCoord.dIdx);
        // 处理第一个g
        uint32_t headSize = 0;
        if (s1IdxStart == s1IdxEnd) {
            headSize = gIdxEnd - gIdxStart;
        } else {
            headSize = offsetCalculator.GetDimG() - gIdxStart;
        }

        CopySingleMatrixNDToND<T>(dstTensor.tensor,
                                  srcTensor.gmTensor[queryGmbaseOffset + gIdxStart * offsetCalculator.GetDimD()],
                                  headSize, gmCoord.dDealSize, offsetCalculator.GetStrideG(), dstTensor.colCount);

        if (s1IdxEnd - s1IdxStart >= 1) {
            uint64_t gmOffset = queryGmbaseOffset + offsetCalculator.GetStrideS1();
            uint32_t ubOffset = headSize * dstTensor.colCount;
            // 处理中间块
            for (uint32_t i = s1IdxStart + 1; i < s1IdxEnd; i++) {
                CopySingleMatrixNDToND<T>(dstTensor.tensor[ubOffset], srcTensor.gmTensor[gmOffset],
                                          offsetCalculator.GetDimG(), gmCoord.dDealSize, offsetCalculator.GetStrideG(),
                                          dstTensor.colCount);
                gmOffset += offsetCalculator.GetStrideS1();
                ubOffset += offsetCalculator.GetDimG() * dstTensor.colCount;
            }

            // 处理尾块
            if (gIdxEnd > 0) {
                CopySingleMatrixNDToND<T>(dstTensor.tensor[ubOffset], srcTensor.gmTensor[gmOffset], gIdxEnd,
                                          gmCoord.dDealSize, offsetCalculator.GetStrideG(), dstTensor.colCount);
            }
        }
    }
};

// ----------------------------------------------CopyKvGmToUb--------------------------------
template <typename KV_T, GmFormat GM_FORMAT>
class CopyKvGmToUb {
public:
    __aicore__ inline void operator()(FaUbTensor<KV_T> &dstTensor, FaGmTensor<KV_T, GM_FORMAT> &srcTensor,
                                      GmCoord &gmCoord)
    {
        if constexpr (GM_FORMAT == GmFormat::BNSD || GM_FORMAT == GmFormat::BSND || GM_FORMAT == GmFormat::NTD ||
                      GM_FORMAT == GmFormat::TND) {
            ProcessContinuousOrTensorlist(dstTensor, srcTensor, gmCoord);
        } else if constexpr (GM_FORMAT == GmFormat::PA_BnBsND || GM_FORMAT == GmFormat::PA_BnNBsD ||
                             GM_FORMAT == GmFormat::PA_NZ) {
            ProcessPageAttention(dstTensor, srcTensor, gmCoord);
        }
    }

private:
    __aicore__ inline void ProcessContinuousOrTensorlist(FaUbTensor<KV_T> &dstTensor,
                                                         FaGmTensor<KV_T, GM_FORMAT> &srcTensor, GmKvCoord &gmCoord)
    {
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint64_t offset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, gmCoord.s2Idx, gmCoord.dIdx);
        CopySingleMatrixNDToND<KV_T>(dstTensor.tensor, srcTensor.gmTensor[offset], gmCoord.s2DealSize,
                                     offsetCalculator.GetStrideS2(), gmCoord.dDealSize, dstTensor.colCount);
    }

    __aicore__ inline void ProcessPageAttention(FaUbTensor<KV_T> &dstTensor, FaGmTensor<KV_T, GM_FORMAT> &srcTensor,
                                                GmKvCoord &gmCoord)
    {
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint32_t curS2Idx = gmCoord.s2Idx;
        uint32_t copyFinishRowCnt = 0;
        uint32_t blockElementCnt = AttentionCommon::BYTE_BLOCK / sizeof(KV_T);

        if constexpr (GM_FORMAT == GmFormat::PA_NZ) {
            while (copyFinishRowCnt < gmCoord.s2DealSize) {
                // 获取需要拷贝的行数
                uint32_t copyRowCnt = offsetCalculator.GetBlockSize() - curS2Idx % offsetCalculator.GetBlockSize();
                if (copyFinishRowCnt + copyRowCnt > gmCoord.s2DealSize) {
                    copyRowCnt =
                        gmCoord.s2DealSize - copyFinishRowCnt; // block table中当前batch表项的尾块，一个block未拷满
                }

                // 计算offset
                uint64_t gmOffset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, curS2Idx, gmCoord.dIdx);
                uint64_t l1Offset = copyFinishRowCnt * blockElementCnt;

                // 拷贝数据
                // DataCopy
                DataCopyParams repeatParams;
                repeatParams.blockCount = gmCoord.dDealSize / blockElementCnt;         // D可切出多少个32B
                repeatParams.blockLen = copyRowCnt;                                    // 单位32B
                repeatParams.srcStride = offsetCalculator.GetBlockSize() - copyRowCnt; // 单位32B
                repeatParams.dstStride = dstTensor.rowCount - copyRowCnt;              // 单位32B
                DataCopy(dstTensor.tensor[l1Offset], srcTensor.gmTensor[gmOffset], repeatParams);

                // 更新完成拷贝的行数和s2Idx
                copyFinishRowCnt += copyRowCnt;
                curS2Idx += copyRowCnt;
            }
        } else { // BBH BNBD
            while (copyFinishRowCnt < gmCoord.s2DealSize) {
                // 获取需要拷贝的行数
                uint32_t copyRowCnt = offsetCalculator.GetBlockSize() - curS2Idx % offsetCalculator.GetBlockSize();
                if (copyFinishRowCnt + copyRowCnt > gmCoord.s2DealSize) {
                    copyRowCnt =
                        gmCoord.s2DealSize - copyFinishRowCnt; // block table中当前batch表项的尾块，一个block未拷满
                }

                // 计算offset
                uint64_t gmOffset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, curS2Idx, gmCoord.dIdx);
                uint64_t l1Offset = copyFinishRowCnt * blockElementCnt;

                // DataCopyPad
                CopySingleMatrixNDToND<KV_T>(dstTensor.tensor[l1Offset], srcTensor.gmTensor[gmOffset], copyRowCnt,
                                             gmCoord.dDealSize, offsetCalculator.GetStrideBlockSize(),
                                             dstTensor.rowCount);

                // 更新完成拷贝的行数和s2Idx
                copyFinishRowCnt += copyRowCnt;
                curS2Idx += copyRowCnt;
            }
        }
    }
};

// ---------------------------------------------CopyPSEGmToUb--------------------------------------
struct GmPseCoord {
    uint32_t bIdx = 0;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx = 0;
    uint32_t s2Idx = 0;
    uint32_t gS1DealSize = 0;
    uint32_t s2DealSize = 0;
    uint64_t s1LeftPaddingSize = 0;
    uint64_t s2LeftPaddingSize = 0;
    uint64_t actualBIdx = 0;
};

// 对齐暂不考虑TND
template <typename PSE_T, GmFormat GM_FORMAT, UbFormat UB_FORMAT>
class CopyPSEGmToUb {
public:
    __aicore__ inline void operator()(FaUbTensor<PSE_T> &dstTensor, FaGmTensor<PSE_T, GM_FORMAT> &srcTensor,
                                      GmPseCoord &gmPseCoord,
                                      bool qsEqualOne = false) // qsEqualOne用于适配qs = 1时，pseshifts1 > qs的场景
    {
        if constexpr (UB_FORMAT == UbFormat::GS1) {
            // 连续，单次拷贝
            OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
            uint64_t s1Size = 0;
            if (offsetCalculator.actualSeqLensQParser.GetActualLenDims() != 0) {
                s1Size = offsetCalculator.actualSeqLensQParser.GetActualSeqLength(gmPseCoord.actualBIdx);
            } else if (qsEqualOne) {
                s1Size = 1U;
            } else {
                s1Size = offsetCalculator.GetDimS1();
            }
            uint32_t gIdxStart = gmPseCoord.gS1Idx / s1Size;
            uint32_t s1IdxStart = gmPseCoord.gS1Idx % s1Size;
            uint64_t pseOffset = offsetCalculator.GetOffset(gmPseCoord.bIdx, gmPseCoord.n2Idx, gIdxStart,
                                                            gmPseCoord.s1LeftPaddingSize + s1IdxStart,
                                                            gmPseCoord.s2LeftPaddingSize + gmPseCoord.s2Idx);
            // 统一的接口
            if (qsEqualOne) {
                CopySingleMatrixNDToND<PSE_T>(dstTensor.tensor, srcTensor.gmTensor[pseOffset], gmPseCoord.gS1DealSize,
                                              gmPseCoord.s2DealSize, offsetCalculator.GetStrideG(), dstTensor.colCount);
            } else {
                CopySingleMatrixNDToND<PSE_T>(dstTensor.tensor, srcTensor.gmTensor[pseOffset], gmPseCoord.gS1DealSize,
                                              gmPseCoord.s2DealSize, offsetCalculator.GetStrideS1(),
                                              dstTensor.colCount);
            }
        } else if constexpr (UB_FORMAT == UbFormat::S1G) {
            // 不连续，需要分3次拷贝
            OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
            uint32_t s1IdxStart = gmPseCoord.gS1Idx / offsetCalculator.GetDimG();
            uint32_t gIdxStart = gmPseCoord.gS1Idx % offsetCalculator.GetDimG();
            uint32_t s1IdxEnd = (gmPseCoord.gS1Idx + gmPseCoord.gS1DealSize) / offsetCalculator.GetDimG();
            uint32_t gIdxEnd = (gmPseCoord.gS1Idx + gmPseCoord.gS1DealSize) % offsetCalculator.GetDimG();
            uint64_t gmOffset = offsetCalculator.GetOffset(
                gmPseCoord.bIdx, gmPseCoord.n2Idx, gIdxStart, gmPseCoord.s1LeftPaddingSize + s1IdxStart,
                gmPseCoord.s2LeftPaddingSize + gmPseCoord.s2Idx); // GM上为GS1

            // 处理第一个S
            uint32_t headSize = 0;
            if (s1IdxStart == s1IdxEnd) {
                headSize = gIdxEnd - gIdxStart;
            } else {
                headSize = offsetCalculator.GetDimG() - gIdxStart;
            }

            CopySingleMatrixNDToND<PSE_T>(dstTensor.tensor, srcTensor.gmTensor[gmOffset], headSize,
                                          gmPseCoord.s2DealSize, offsetCalculator.GetStrideG(), dstTensor.colCount);
            if (s1IdxEnd - s1IdxStart >= 1) {
                uint64_t ubOffset = ((uint64_t)headSize) * ((uint64_t)dstTensor.colCount);
                // 处理中间块
                gmOffset = offsetCalculator.GetOffset(gmPseCoord.bIdx, gmPseCoord.n2Idx, 0,
                                                      gmPseCoord.s1LeftPaddingSize + s1IdxStart + 1,
                                                      gmPseCoord.s2LeftPaddingSize + gmPseCoord.s2Idx); // GM上为GS1
                // 处理中间块
                for (uint32_t i = s1IdxStart + 1; i < s1IdxEnd; i++) {
                    CopySingleMatrixNDToND<PSE_T>(dstTensor.tensor[ubOffset], srcTensor.gmTensor[gmOffset],
                                                  offsetCalculator.GetDimG(), gmPseCoord.s2DealSize,
                                                  offsetCalculator.GetStrideG(), dstTensor.colCount);
                    ubOffset += offsetCalculator.GetDimG() * dstTensor.colCount;
                    gmOffset += offsetCalculator.GetStrideS1();
                }

                // 处理尾块
                if (gIdxEnd > 0) {
                    CopySingleMatrixNDToND<PSE_T>(dstTensor.tensor[ubOffset], srcTensor.gmTensor[gmOffset], gIdxEnd,
                                                  gmPseCoord.s2DealSize, offsetCalculator.GetStrideG(),
                                                  dstTensor.colCount);
                }
            }
        }
    }
};

#endif