/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file copy_gm_to_l1.h
 * \brief
 */
#ifndef COPY_GM_TO_L1_H
#define COPY_GM_TO_L1_H

#include "gm_layout.h"
#include "fa_l1_tensor.h"
#include "gm_coord.h"
#include "offset_calculator_v2.h"

// ----------------------------------------------CopyQueryGmToL1--------------------------------
template <typename T>
__aicore__ inline void CopySingleMatrixNDToNZ(LocalTensor<T> l1Tensor,
                                              const GlobalTensor<T> gmTensor,
                                              uint32_t nValue,
                                              uint32_t dValue,
                                              uint32_t srcDValue,
                                              uint32_t dstNzC0Stride)
{
    constexpr uint32_t SRCDVALUE_LIMIT = 65536;
    Nd2NzParams nd2nzPara;
    if (unlikely(srcDValue < SRCDVALUE_LIMIT)) {
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = nValue; // nd矩阵的行数
        if constexpr (IsSameType<T, int4b_t>::value) {
            constexpr uint32_t HALF_SIZE_DIVISOR = 2;
            nd2nzPara.dValue = dValue / HALF_SIZE_DIVISOR;
            nd2nzPara.srcDValue = srcDValue / HALF_SIZE_DIVISOR;
        } else {
            nd2nzPara.dValue = dValue; // nd矩阵的列数
            nd2nzPara.srcDValue = srcDValue; // 同一nd矩阵相邻行起始地址间的偏移
        }
        nd2nzPara.dstNzC0Stride = dstNzC0Stride;
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = 0;
        nd2nzPara.dstNzMatrixStride = 0;
        DataCopy(l1Tensor, gmTensor, nd2nzPara);
    } else {
        uint64_t l1Offset = 0;
        uint64_t gmOffset = 0;
        uint32_t srcNdMatrixNum = nValue;
        uint32_t srcNdMatrixStride = srcDValue;
        uint32_t dstNzMatrixStride = 16;
        nValue = 1;
        srcDValue = dValue;
        for (uint32_t i = 0; i < srcNdMatrixNum; i++) {
            nd2nzPara.ndNum = 1;
            nd2nzPara.nValue = nValue; // nd矩阵的行数
            if constexpr (IsSameType<T, int4b_t>::value) {
                constexpr uint32_t HALF_SIZE_DIVISOR = 2;
                nd2nzPara.dValue = dValue / HALF_SIZE_DIVISOR;
                nd2nzPara.srcDValue = srcDValue / HALF_SIZE_DIVISOR;
            } else {
                nd2nzPara.dValue = dValue; // nd矩阵的列数
                nd2nzPara.srcDValue = srcDValue; // 同一nd矩阵相邻行起始地址间的偏移
            }
            nd2nzPara.dstNzC0Stride = dstNzC0Stride;
            nd2nzPara.dstNzNStride = 1;
            nd2nzPara.srcNdMatrixStride = 0;
            nd2nzPara.dstNzMatrixStride = 0;
            DataCopy(l1Tensor[l1Offset], gmTensor[gmOffset], nd2nzPara);

            gmOffset += srcNdMatrixStride;
            l1Offset += dstNzMatrixStride;
        }
    }
}

template <typename T>
__aicore__ inline void CopyMultiMatrixNDToNZ(LocalTensor<T> l1Tensor,
                                             const GlobalTensor<T> gmTensor,
                                             uint32_t srcNdMatrixNum,
                                             uint32_t srcNdMatrixStride,
                                             uint32_t dstNzMatrixStride,
                                             uint32_t nValue,
                                             uint32_t dValue,
                                             uint32_t srcDValue,
                                             uint32_t dstNzC0Stride)
{
    constexpr uint32_t ND_MATRIX_STRIDE_LIMIT =
        65536; // Mutil ND2NZ搬运时，Nd2NzParams支持的srcNdMatrixStride的取值范围为[0, 65536]，单位为元素
    if (unlikely(srcNdMatrixStride > ND_MATRIX_STRIDE_LIMIT)) {
        uint64_t l1Offset = 0;
        uint64_t gmOffset = 0;
        for (uint32_t i = 0; i < srcNdMatrixNum; i++) {
            CopySingleMatrixNDToNZ(l1Tensor[l1Offset], gmTensor[gmOffset], nValue, dValue, srcDValue, dstNzC0Stride);
            gmOffset += srcNdMatrixStride;
            l1Offset += dstNzMatrixStride;
        }
    } else {
        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = srcNdMatrixNum;
        nd2nzPara.nValue = nValue; // nd矩阵的行数
        if constexpr (IsSameType<T, int4b_t>::value) {
            constexpr uint32_t HALF_SIZE_DIVISOR = 2;
            nd2nzPara.dValue = dValue / HALF_SIZE_DIVISOR;
            nd2nzPara.srcDValue = srcDValue / HALF_SIZE_DIVISOR;
        } else {
            nd2nzPara.dValue = dValue; // nd矩阵的列数
            nd2nzPara.srcDValue = srcDValue; // 同一nd矩阵相邻行起始地址间的偏移
        }
        nd2nzPara.dstNzC0Stride = dstNzC0Stride;
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = srcNdMatrixStride;
        nd2nzPara.dstNzMatrixStride = dstNzMatrixStride;
        DataCopy(l1Tensor, gmTensor, nd2nzPara);
    }
}

template <typename Q_T, GmFormat GM_FORMAT, L1Format L1_FORMAT = L1Format::NZ> class CopyQueryGmToL1 {
  public:
    __aicore__ inline void
    operator()(FaL1Tensor<Q_T, L1_FORMAT> &dstTensor, FaGmTensor<Q_T, GM_FORMAT> &srcTensor, GmCoord &gmCoord)
    {
        if constexpr ((GM_FORMAT == GmFormat::BSNGD) || (GM_FORMAT == GmFormat::TNGD)) {
            ProcessS1G(dstTensor, srcTensor, gmCoord);
        } else if constexpr (GM_FORMAT == GmFormat::BNGSD) {
            OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
            if (offsetCalculator.actualSeqLensQParser.GetActualLenDims() != 0) {
                ProcessGS1(dstTensor, srcTensor, gmCoord);
            } else {
                ProcessContinuous(dstTensor, srcTensor, gmCoord);
            }
        } else if constexpr (GM_FORMAT == GmFormat::NGTD) {
            ProcessGS1(dstTensor, srcTensor, gmCoord);
        }
    }

  private:
    __aicore__ inline void
    ProcessS1G(FaL1Tensor<Q_T, L1_FORMAT> &dstTensor, FaGmTensor<Q_T, GM_FORMAT> &srcTensor, GmCoord &gmCoord)
    {
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint32_t s1IdxStart = gmCoord.gS1Idx / offsetCalculator.GetDimG();
        uint32_t gIdxStart = gmCoord.gS1Idx % offsetCalculator.GetDimG();
        uint32_t s1IdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) / offsetCalculator.GetDimG();
        uint32_t gIdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) % offsetCalculator.GetDimG();

        uint64_t queryGmbaseOffset =
            offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, 0, s1IdxStart, gmCoord.dIdx);

        if (offsetCalculator.GetDimG() == 1) {
            CopySingleMatrixNDToNZ(dstTensor.tensor,
                                   srcTensor.gmTensor[queryGmbaseOffset],
                                   s1IdxEnd - s1IdxStart,
                                   gmCoord.dDealSize,
                                   offsetCalculator.GetStrideS1(),
                                   dstTensor.rowCount);
            return;
        }

        // 处理第一个S
        uint32_t headSize = 0;
        if (s1IdxStart == s1IdxEnd) {
            headSize = gIdxEnd - gIdxStart;
        } else {
            headSize = offsetCalculator.GetDimG() - gIdxStart;
        }

        uint64_t offset = queryGmbaseOffset + gIdxStart * offsetCalculator.GetDimD();
        CopySingleMatrixNDToNZ(dstTensor.tensor,
                               srcTensor.gmTensor[offset],
                               headSize,
                               gmCoord.dDealSize,
                               offsetCalculator.GetStrideG(),
                               dstTensor.rowCount);

        if (s1IdxEnd - s1IdxStart >= 1) {
            // 处理中间块
            uint64_t gmOffset = queryGmbaseOffset + offsetCalculator.GetStrideS1();
            uint64_t l1Offset = headSize * 16U;
            if (s1IdxEnd - s1IdxStart > 1) {
                CopyMultiMatrixNDToNZ(dstTensor.tensor[l1Offset],
                                      srcTensor.gmTensor[gmOffset],
                                      s1IdxEnd - s1IdxStart - 1,
                                      offsetCalculator.GetStrideS1(),
                                      offsetCalculator.GetDimG() * 16U,
                                      offsetCalculator.GetDimG(),
                                      gmCoord.dDealSize,
                                      offsetCalculator.GetStrideG(),
                                      dstTensor.rowCount);
                gmOffset += (s1IdxEnd - s1IdxStart - 1) * offsetCalculator.GetStrideS1();
                l1Offset += (s1IdxEnd - s1IdxStart - 1) * offsetCalculator.GetDimG() * 16U;
            }

            // 处理尾块
            if (gIdxEnd > 0) {
                CopySingleMatrixNDToNZ(dstTensor.tensor[l1Offset],
                                       srcTensor.gmTensor[gmOffset],
                                       gIdxEnd,
                                       gmCoord.dDealSize,
                                       offsetCalculator.GetStrideG(),
                                       dstTensor.rowCount);
            }
        }
    }

    __aicore__ inline void
    ProcessContinuous(FaL1Tensor<Q_T, L1_FORMAT> &dstTensor, FaGmTensor<Q_T, GM_FORMAT> &srcTensor, GmCoord &gmCoord)
    {
        // B*N2*GS1*D
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint32_t gIdxStart = gmCoord.gS1Idx / offsetCalculator.GetDimS1();
        uint32_t s1IdxStart = gmCoord.gS1Idx % offsetCalculator.GetDimS1();

        uint64_t offset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, gIdxStart, s1IdxStart, gmCoord.dIdx);
        CopySingleMatrixNDToNZ(dstTensor.tensor,
                               srcTensor.gmTensor[offset],
                               gmCoord.gS1DealSize,
                               gmCoord.dDealSize,
                               offsetCalculator.GetDimD(),
                               dstTensor.rowCount);
    }

    __aicore__ inline void
    ProcessGS1(FaL1Tensor<Q_T, L1_FORMAT> &dstTensor, FaGmTensor<Q_T, GM_FORMAT> &srcTensor, GmCoord &gmCoord)
    {
        // N2*G*T(BS1)*D
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint64_t s1Size = 0;
        if constexpr (GmLayoutParams<GM_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND) {
            s1Size = offsetCalculator.actualSeqLensQParser.GetActualSeqLength(gmCoord.bIdx);
        } else {
            if (offsetCalculator.actualSeqLensQParser.GetActualLenDims() != 0) {
                s1Size = offsetCalculator.actualSeqLensQParser.GetActualSeqLength(gmCoord.bIdx);
            } else {
                s1Size = offsetCalculator.GetDimS1();
            }
        }

        uint32_t gIdxStart = gmCoord.gS1Idx / s1Size;
        uint32_t s1IdxStart = gmCoord.gS1Idx % s1Size;
        uint32_t gIdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) / s1Size;
        uint32_t s1IdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) % s1Size;

        uint64_t queryGmbaseOffset =
            offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, gIdxStart, 0, gmCoord.dIdx);

        // 处理第一个S
        uint32_t headSize = 0;
        if (gIdxStart == gIdxEnd) {
            headSize = s1IdxEnd - s1IdxStart;
        } else {
            headSize = s1Size - s1IdxStart;
        }

        uint64_t offset = queryGmbaseOffset + s1IdxStart * offsetCalculator.GetDimD();
        CopySingleMatrixNDToNZ(dstTensor.tensor,
                               srcTensor.gmTensor[offset],
                               headSize,
                               gmCoord.dDealSize,
                               offsetCalculator.GetStrideS1(),
                               dstTensor.rowCount);

        if (gIdxEnd - gIdxStart >= 1) {
            // 处理中间块
            uint64_t gmOffset = queryGmbaseOffset + offsetCalculator.GetStrideG();
            uint64_t l1Offset = headSize * 16U;

            if (gIdxEnd - gIdxStart > 1) {
                CopyMultiMatrixNDToNZ(dstTensor.tensor[l1Offset],
                                      srcTensor.gmTensor[gmOffset],
                                      gIdxEnd - gIdxStart - 1,
                                      offsetCalculator.GetStrideG(),
                                      s1Size * 16U,
                                      s1Size,
                                      gmCoord.dDealSize,
                                      offsetCalculator.GetStrideS1(),
                                      dstTensor.rowCount);
                gmOffset += (gIdxEnd - gIdxStart - 1) * offsetCalculator.GetStrideG();
                l1Offset += (gIdxEnd - gIdxStart - 1) * s1Size * 16U;
            }

            // 处理尾块
            if (s1IdxEnd > 0) {
                CopySingleMatrixNDToNZ(dstTensor.tensor[l1Offset],
                                       srcTensor.gmTensor[gmOffset],
                                       s1IdxEnd,
                                       gmCoord.dDealSize,
                                       offsetCalculator.GetStrideS1(),
                                       dstTensor.rowCount);
            }
        }
    }
};

// ----------------------------------------------CopyKvGmToL1--------------------------------
struct GmKvCoord {
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t s2Idx;
    uint32_t dIdx;
    uint32_t s2DealSize;
    uint32_t dDealSize;
};

template <typename KV_T, GmFormat GM_FORMAT, L1Format L1_FORMAT = L1Format::NZ> class CopyKvGmToL1 {
  public:
    __aicore__ inline void
    operator()(FaL1Tensor<KV_T, L1_FORMAT> &dstTensor, FaGmTensor<KV_T, GM_FORMAT> &srcTensor, GmKvCoord &gmCoord)
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
    __aicore__ inline void ProcessContinuousOrTensorlist(FaL1Tensor<KV_T, L1_FORMAT> &dstTensor,
                                                         FaGmTensor<KV_T, GM_FORMAT> &srcTensor,
                                                         GmKvCoord &gmCoord)
    {
        // B*N2*GS1*D
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint64_t offset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, gmCoord.s2Idx, gmCoord.dIdx);
        CopySingleMatrixNDToNZ(dstTensor.tensor,
                               srcTensor.gmTensor[offset],
                               gmCoord.s2DealSize,
                               gmCoord.dDealSize,
                               offsetCalculator.GetStrideS2(),
                               dstTensor.rowCount);
    }

    __aicore__ inline void ProcessPageAttention(FaL1Tensor<KV_T, L1_FORMAT> &dstTensor,
                                                FaGmTensor<KV_T, GM_FORMAT> &srcTensor,
                                                GmKvCoord &gmCoord)
    {
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint32_t curS2Idx = gmCoord.s2Idx;
        uint32_t copyFinishRowCnt = 0;
        uint32_t blockElementCnt = 32 / sizeof(KV_T);
        if constexpr (IsSameType<KV_T, int4b_t>::value) {
            blockElementCnt = 64; // int4b时32B可以存64个元素
        }
        while (copyFinishRowCnt < gmCoord.s2DealSize) {
            // 获取需要拷贝的行数
            uint32_t copyRowCnt = offsetCalculator.GetBlockSize() - curS2Idx % offsetCalculator.GetBlockSize();
            if (copyFinishRowCnt + copyRowCnt > gmCoord.s2DealSize) {
                copyRowCnt = gmCoord.s2DealSize - copyFinishRowCnt; // 一个block未拷满
            }

            // 计算offset
            uint64_t gmOffset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, curS2Idx, gmCoord.dIdx);
            uint64_t l1Offset = copyFinishRowCnt * blockElementCnt;

            // 拷贝数据
            if constexpr (GM_FORMAT == GmFormat::PA_NZ) {
                DataCopyParams intriParams;
                intriParams.blockCount = gmCoord.dDealSize / blockElementCnt;
                intriParams.blockLen = copyRowCnt;
                intriParams.dstStride = dstTensor.rowCount - copyRowCnt;
                intriParams.srcStride = offsetCalculator.GetBlockSize() - copyRowCnt;
                DataCopy(dstTensor.tensor[l1Offset], srcTensor.gmTensor[gmOffset], intriParams);
            } else {
                CopySingleMatrixNDToNZ(dstTensor.tensor[l1Offset],
                                       srcTensor.gmTensor[gmOffset],
                                       copyRowCnt,
                                       gmCoord.dDealSize,
                                       offsetCalculator.GetStrideBlockSize(),
                                       dstTensor.rowCount);
            }

            // 更新完成拷贝的行数和s2Idx
            copyFinishRowCnt += copyRowCnt;
            curS2Idx += copyRowCnt;
        }
    }
};

template <typename KV_T, GmFormat GM_FORMAT, L1Format L1_FORMAT = L1Format::NZ> class CopyKKropePAGmToL1 {
  public:
    __aicore__ inline void operator()(FaL1Tensor<KV_T, L1_FORMAT> &dstTensorK,
                                      FaL1Tensor<KV_T, L1_FORMAT> &dstTensorKrope,
                                      FaGmTensor<KV_T, GM_FORMAT> &srcTensorK,
                                      FaGmTensor<KV_T, GM_FORMAT> &srcTensorKrope,
                                      GmKvCoord &gmCoordK,
                                      GmKvCoord &gmCoordKrope)
    {
        if constexpr (GM_FORMAT == GmFormat::PA_NZ || GM_FORMAT == GmFormat::PA_BnNBsD ||
                      GM_FORMAT == GmFormat::PA_BnBsND) {
            ProcessPageAttention(dstTensorK, dstTensorKrope, srcTensorK, srcTensorKrope, gmCoordK, gmCoordKrope);
        }
    }

  private:
    __aicore__ inline uint64_t
    GetBlockIdx(FaGmTensor<KV_T, GM_FORMAT> &srcKTensor, uint64_t blockIdxInBatch, uint32_t bIdx)
    {
        return srcKTensor.offsetCalculator.blockTableParser.GetBlockIdx(bIdx, blockIdxInBatch);
    }

    __aicore__ inline uint64_t
    GetOffset(FaGmTensor<KV_T, GM_FORMAT> &srcTensor, int32_t blockIdx, uint32_t n2Idx, uint32_t s2Idx, uint32_t dIdx)
    {
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint64_t bsIdx = s2Idx % offsetCalculator.GetBlockSize();
        uint64_t offset = 0;
        if constexpr (GM_FORMAT == GmFormat::PA_NZ) {
            uint32_t d1Idx = dIdx / offsetCalculator.GetD0();
            uint32_t d0Idx = dIdx % offsetCalculator.GetD0();
            offset = blockIdx * offsetCalculator.GetStrideBlockNum() + n2Idx * offsetCalculator.GetStrideN2() +
                     d1Idx * offsetCalculator.GetStrideD1() + bsIdx * offsetCalculator.GetStrideBlockSize() +
                     d0Idx * offsetCalculator.GetStrideD0();
        } else {
            offset = blockIdx * offsetCalculator.GetStrideBlockNum() + n2Idx * offsetCalculator.GetStrideN2() +
                     bsIdx * offsetCalculator.GetStrideBlockSize() + dIdx * offsetCalculator.GetStrideD();
        }

        return offset;
    }

    __aicore__ inline void ProcessPageAttention(FaL1Tensor<KV_T, L1_FORMAT> &dstTensorK,
                                                FaL1Tensor<KV_T, L1_FORMAT> &dstTensorKrope,
                                                FaGmTensor<KV_T, GM_FORMAT> &srcTensorK,
                                                FaGmTensor<KV_T, GM_FORMAT> &srcTensorKrope,
                                                GmKvCoord &gmCoordK,
                                                GmKvCoord &gmCoordKrope)
    {
        OffsetCalculator<GM_FORMAT> &offsetCalculatorK = srcTensorK.offsetCalculator;
        OffsetCalculator<GM_FORMAT> &offsetCalculatorKrope = srcTensorKrope.offsetCalculator;
        uint32_t curS2Idx = gmCoordK.s2Idx;
        uint32_t copyFinishRowCnt = 0;
        uint32_t blockElementCnt = 32 / sizeof(KV_T);
        if constexpr (IsSameType<KV_T, int4b_t>::value) {
            blockElementCnt = 64; // int4b时32B可以存64个元素
        }

        while (copyFinishRowCnt < gmCoordK.s2DealSize) {
            // 获取需要拷贝的行数
            uint32_t copyRowCnt = offsetCalculatorK.GetBlockSize() - curS2Idx % offsetCalculatorK.GetBlockSize();
            if (copyFinishRowCnt + copyRowCnt > gmCoordK.s2DealSize) {
                copyRowCnt = gmCoordK.s2DealSize - copyFinishRowCnt; // 一个block未拷满
            }

            // 计算offset
            uint64_t blockIdxInBatch = curS2Idx / offsetCalculatorK.GetBlockSize(); // 获取block table上的索引
            uint32_t blockIdx = GetBlockIdx(srcTensorK, blockIdxInBatch, gmCoordK.bIdx);
            uint64_t gmOffsetK = GetOffset(srcTensorK, blockIdx, gmCoordK.n2Idx, curS2Idx, gmCoordK.dIdx);
            uint64_t gmOffsetKrope =
                GetOffset(srcTensorKrope, blockIdx, gmCoordKrope.n2Idx, curS2Idx, gmCoordKrope.dIdx);
            uint64_t l1Offset = copyFinishRowCnt * blockElementCnt;

            // 拷贝数据
            if constexpr (GM_FORMAT == GmFormat::PA_NZ) {
                DataCopyParams intriParamsK;
                intriParamsK.blockCount = gmCoordK.dDealSize / blockElementCnt;
                intriParamsK.blockLen = copyRowCnt;
                intriParamsK.dstStride = dstTensorK.rowCount - copyRowCnt;
                intriParamsK.srcStride = offsetCalculatorK.GetBlockSize() - copyRowCnt;
                DataCopy(dstTensorK.tensor[l1Offset], srcTensorK.gmTensor[gmOffsetK], intriParamsK);

                DataCopyParams intriParamsKrope;
                intriParamsKrope.blockCount = gmCoordKrope.dDealSize / blockElementCnt;
                intriParamsKrope.blockLen = copyRowCnt;
                intriParamsKrope.dstStride = dstTensorKrope.rowCount - copyRowCnt;
                intriParamsKrope.srcStride = offsetCalculatorKrope.GetBlockSize() - copyRowCnt;
                DataCopy(dstTensorKrope.tensor[l1Offset], srcTensorKrope.gmTensor[gmOffsetKrope], intriParamsKrope);
            } else {
                CopySingleMatrixNDToNZ(dstTensorK.tensor[l1Offset],
                                       srcTensorK.gmTensor[gmOffsetK],
                                       copyRowCnt,
                                       gmCoordK.dDealSize,
                                       offsetCalculatorK.GetStrideBlockSize(),
                                       dstTensorK.rowCount);
                CopySingleMatrixNDToNZ(dstTensorKrope.tensor[l1Offset],
                                       srcTensorKrope.gmTensor[gmOffsetKrope],
                                       copyRowCnt,
                                       gmCoordKrope.dDealSize,
                                       offsetCalculatorKrope.GetStrideBlockSize(),
                                       dstTensorKrope.rowCount);
            }

            // 更新完成拷贝的行数和s2Idx
            copyFinishRowCnt += copyRowCnt;
            curS2Idx += copyRowCnt;
        }
    }
};

#endif