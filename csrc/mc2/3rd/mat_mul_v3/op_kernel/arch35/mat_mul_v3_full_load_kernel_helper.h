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
 * \file mat_mul_v3_full_load_kernel_helper.h
 * \brief
 */
#ifndef __OP_KERNEL_MAT_MUL_V3_FULL_LOAD_KERNEL_HELPER_H__
#define __OP_KERNEL_MAT_MUL_V3_FULL_LOAD_KERNEL_HELPER_H__

#include "../mat_mul_v3_common.h"

namespace Mc2MatmulV3Advanced {
using namespace AscendC;
using namespace matmul;

template <TPosition POSITION, CubeFormat FORMAT, typename TYPE, bool ISTRANS = false,
          LayoutMode LAYOUT = LayoutMode::NONE, bool IBSHARE = false>
struct MatmulL1GmType : MatmulType<POSITION, FORMAT, TYPE, ISTRANS, LAYOUT, IBSHARE> {
    constexpr static TPosition srcPos = TPosition::GM;
};

template <class A_TYPE, class A_T, class BLOCK_TYPE>
__aicore__ inline void AswAL1FullLoadKernelCopyInA1(BLOCK_TYPE &block, const Mc2MatMulV3TilingData *matmulv3TilingData,
                                                    bool isMMultiCore, GlobalTensor<A_T> &aGlobal,
                                                    TQue<QuePosition::A1, 1> &InQueueAL1, LocalTensor<A_T> &al1Local)
{
    uint64_t innerAlignedBlock = BLOCK_BYTE_SIZE / sizeof(A_T);
    uint64_t mAligned = MMV3CeilAlign(matmulv3TilingData->tCubeTiling.singleCoreM, BLOCK_SIZE);
    uint64_t kAligned = MMV3CeilAlign(matmulv3TilingData->tCubeTiling.Ka, innerAlignedBlock);
    if constexpr (A_TYPE::isTrans) {
        mAligned = MMV3CeilAlign(matmulv3TilingData->tCubeTiling.singleCoreM, innerAlignedBlock);
        kAligned = MMV3CeilAlign(matmulv3TilingData->tCubeTiling.Ka, BLOCK_SIZE);
    }
    GetTPipePtr()->InitBuffer(InQueueAL1, 1, mAligned * kAligned * sizeof(A_T));
    al1Local = InQueueAL1.AllocTensor<A_T>();
    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    int32_t shapeM = matmulv3TilingData->tCubeTiling.M;
    int32_t shapeK = matmulv3TilingData->tCubeTiling.Ka;
    uint64_t instrM = isMMultiCore && block.params_.mCntIndex == block.params_.mCnt - 1 ?
                          block.params_.mBaseTail :
                          static_cast<uint64_t>(matmulv3TilingData->tCubeTiling.singleCoreM);
    uint64_t nDim = static_cast<uint64_t>(instrM);
    uint64_t dDim = static_cast<uint64_t>(shapeK);
    uint64_t offsetA = block.params_.mCntIndex * matmulv3TilingData->tCubeTiling.singleCoreM * shapeK;
    if constexpr (A_TYPE::isTrans) {
        nDim = static_cast<uint64_t>(shapeK);
        dDim = static_cast<uint64_t>(instrM);
        offsetA = block.params_.mCntIndex * matmulv3TilingData->tCubeTiling.singleCoreM;
    }
    nd2nzParams.nValue = nDim;
    nd2nzParams.dValue = dDim;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.srcDValue = static_cast<uint64_t>(A_TYPE::isTrans ? shapeM : shapeK);
    nd2nzParams.dstNzC0Stride = MMV3CeilAlign(nDim, static_cast<uint64_t>(BLOCK_SIZE));
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;
    DataCopy(al1Local, aGlobal[offsetA], nd2nzParams);
    InQueueAL1.EnQue(al1Local);
    al1Local = InQueueAL1.DeQue<A_T>();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2AswAL1FullLoadKernelMainLoop(MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG> &mm, BLOCK_TYPE &block,
                                const Mc2MatMulV3TilingData *matmulv3TilingData,
                                GlobalTensor<typename B_TYPE::T> &bGlobal, GlobalTensor<typename C_TYPE::T> &cGlobal,
                                GlobalTensor<typename BIAS_TYPE::T> &biasGlobal, TQue<QuePosition::A1, 1> &InQueueAL1,
                                LocalTensor<typename A_TYPE::T> &al1Local, uint8_t enAtomic)
{
    mm.SetSubBlockIdx(0);
    mm.Init(&matmulv3TilingData->tCubeTiling, GetTPipePtr());
    SetAtomicNone();
    mm.SetHF32(matmulv3TilingData->isHf32, 1);

    for (uint64_t j = 0; j < block.params_.round; j++) {
        uint64_t newBlockIdx =
            (j == block.params_.round - 1) ? (GetBlockIdx() / block.params_.totalSplitCnt) : GetBlockIdx();
        block.UpdateBasicIndex(j, newBlockIdx); // 使能错位分核更新Index
        if (block.params_.index < block.params_.totalCnt) {
            block.template UpdateBlockParams<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(j);
            if (block.params_.singleCoreM > 0 && block.params_.singleCoreN > 0) {
                block.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>();

                mm.SetTensorA(al1Local, A_TYPE::isTrans);
                mm.SetTensorB(bGlobal[block.offset_.offsetB], B_TYPE::isTrans);
                if (matmulv3TilingData->tCubeTiling.isBias) {
                    mm.SetBias(biasGlobal[block.offset_.offsetBias]);
                }
                mm.SetSingleShape(block.params_.singleCoreM, block.params_.singleCoreN,
                                  matmulv3TilingData->tCubeTiling.singleCoreK);
                // MDL模板，L1输入场景默认al1M=M，分核全载需要通过设置SetOrgShape指定al1M=singleCoreM
                mm.SetOrgShape(block.params_.singleCoreM, matmulv3TilingData->tCubeTiling.N,
                               matmulv3TilingData->tCubeTiling.Ka);
                mm.IterateAll(cGlobal[block.offset_.offsetC], enAtomic);
            }
        }
    }
    mm.SetHF32(false, 0);
    InQueueAL1.FreeTensor(al1Local);
}

template <class B_TYPE>
__aicore__ inline void CalCopyBL1Nd2NzParams(const Mc2MatMulV3TilingData *matmulv3TilingData, Nd2NzParams &nd2nzParams,
                                             uint64_t instrN)
{
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = B_TYPE::isTrans ? instrN : static_cast<uint64_t>(matmulv3TilingData->tCubeTiling.Kb);
    nd2nzParams.dValue = B_TYPE::isTrans ? static_cast<uint64_t>(matmulv3TilingData->tCubeTiling.Kb) : instrN;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.srcDValue =
        static_cast<uint64_t>(B_TYPE::isTrans ? matmulv3TilingData->tCubeTiling.Kb : matmulv3TilingData->tCubeTiling.N);
    nd2nzParams.dstNzC0Stride = MMV3CeilAlign(nd2nzParams.nValue, BLOCK_SIZE);
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;
}

template <class B_TYPE, class B_T, class BLOCK_TYPE>
__aicore__ inline void CalCopyBL1Nz2NzParams(const BLOCK_TYPE &block, const Mc2MatMulV3TilingData *matmulv3TilingData,
                                             bool isNMultiCore, DataCopyParams &dataCopyParams, uint64_t instrN)
{
    if (B_TYPE::isTrans && isNMultiCore) {
        dataCopyParams.blockCount = MMV3DivCeil(matmulv3TilingData->tCubeTiling.Kb, block.params_.kbAlignSize);
        dataCopyParams.blockLen =
            block.params_.kbAlignSize * MMV3CeilAlign(instrN, block.params_.nAlignSize) * sizeof(B_T) / BLOCK_BYTE_SIZE;
        dataCopyParams.srcStride = block.params_.kbAlignSize *
                                   (MMV3CeilAlign(matmulv3TilingData->tCubeTiling.N, block.params_.nAlignSize) -
                                    MMV3CeilAlign(instrN, block.params_.nAlignSize)) *
                                   sizeof(B_T) / BLOCK_BYTE_SIZE;
        dataCopyParams.dstStride = 0;

    } else {
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = MMV3CeilAlign(matmulv3TilingData->tCubeTiling.Kb, block.params_.kbAlignSize) *
                                  MMV3CeilAlign(instrN, block.params_.nAlignSize) * sizeof(B_T) / BLOCK_BYTE_SIZE;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
    }
}

template <class B_TYPE, class B_T, class BLOCK_TYPE>
__aicore__ inline void AswBL1FullLoadKernelCopyInB1(BLOCK_TYPE &block, const Mc2MatMulV3TilingData *matmulv3TilingData,
                                                    bool isNMultiCore, GlobalTensor<B_T> &bGlobal,
                                                    TQue<QuePosition::B1, 1> &InQueueBL1, LocalTensor<B_T> &bl1Local)
{
    uint64_t kAligned = MMV3CeilAlign(matmulv3TilingData->tCubeTiling.Kb, block.params_.kbAlignSize);
    uint64_t nAligned = MMV3CeilAlign(matmulv3TilingData->tCubeTiling.singleCoreN, block.params_.nAlignSize);


    GetTPipePtr()->InitBuffer(InQueueBL1, 1, kAligned * nAligned * sizeof(B_T));
    bl1Local = InQueueBL1.AllocTensor<B_T>();

    uint64_t instrN = isNMultiCore && block.params_.nCntIndex == block.params_.nCnt - 1 ?
                          block.params_.nBaseTail :
                          static_cast<uint64_t>(matmulv3TilingData->tCubeTiling.singleCoreN);
    uint64_t offsetB;
    if constexpr (B_TYPE::format == CubeFormat::ND) {
        Nd2NzParams nd2nzParams;
        CalCopyBL1Nd2NzParams<B_TYPE>(matmulv3TilingData, nd2nzParams, instrN);
        offsetB = B_TYPE::isTrans ? block.params_.nCntIndex * matmulv3TilingData->tCubeTiling.singleCoreN *
                                        matmulv3TilingData->tCubeTiling.Kb :
                                    block.params_.nCntIndex * matmulv3TilingData->tCubeTiling.singleCoreN;
        DataCopy(bl1Local, bGlobal[offsetB], nd2nzParams);
    } else {
        DataCopyParams dataCopyParams;
        CalCopyBL1Nz2NzParams<B_TYPE, B_T, BLOCK_TYPE>(block, matmulv3TilingData, isNMultiCore, dataCopyParams, instrN);
        if (B_TYPE::isTrans && isNMultiCore) {
            offsetB = block.params_.nCntIndex *
                      MMV3CeilAlign(matmulv3TilingData->tCubeTiling.singleCoreN, block.params_.nAlignSize) *
                      block.params_.kbAlignSize;
        } else {
            offsetB = block.params_.nCntIndex *
                      MMV3CeilAlign(matmulv3TilingData->tCubeTiling.singleCoreN, block.params_.nAlignSize) *
                      MMV3CeilAlign(matmulv3TilingData->tCubeTiling.Kb, block.params_.kbAlignSize);
        }
        DataCopy(bl1Local, bGlobal[offsetB], dataCopyParams);
    }
    InQueueBL1.EnQue(bl1Local);
    bl1Local = InQueueBL1.DeQue<B_T>();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
AswBL1FullLoadKernelMainLoop(MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG> &mm, BLOCK_TYPE &block,
                             const Mc2MatMulV3TilingData *matmulv3TilingData, GlobalTensor<typename A_TYPE::T> &aGlobal,
                             GlobalTensor<typename C_TYPE::T> &cGlobal, GlobalTensor<typename BIAS_TYPE::T> &biasGlobal,
                             TQue<QuePosition::B1, 1> &InQueueBL1, LocalTensor<typename B_TYPE::T> &bl1Local,
                             uint8_t enAtomic)
{
    mm.SetSubBlockIdx(0);
    mm.SetHF32(matmulv3TilingData->isHf32, 1);
    mm.Init(&matmulv3TilingData->tCubeTiling, GetTPipePtr());
    SetMMLayoutTransform(true); // fixp使用n搬出，达到cube和fixp并行的效果
    SetAtomicNone();
    for (uint64_t j = 0; j < block.params_.round; j++) {
        uint64_t newBlockIdx =
            (j == block.params_.round - 1) ? (GetBlockIdx() / block.params_.totalSplitCnt) : GetBlockIdx();
        block.UpdateBasicIndex(j, newBlockIdx); // 使能错位分核更新Index
        if (block.params_.index < block.params_.totalCnt) {
            block.template UpdateBlockParams<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(j);
            if (block.params_.singleCoreM > 0 && block.params_.singleCoreN > 0) {
                block.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>();
                mm.SetTensorA(aGlobal[block.offset_.offsetA], A_TYPE::isTrans);
                mm.SetTensorB(bl1Local, B_TYPE::isTrans);
                if (matmulv3TilingData->tCubeTiling.isBias) {
                    mm.SetBias(biasGlobal[block.offset_.offsetBias]);
                }
                mm.SetSingleShape(block.params_.singleCoreM, block.params_.singleCoreN,
                                  matmulv3TilingData->tCubeTiling.singleCoreK);
                // MDL模板，L1输入场景默认bl1N=N，分核全载需要通过设置SetOrgShape指定al1N=singleCoreN,
                // N为内轴，L0C->GM需重置shape
                mm.SetOrgShape(matmulv3TilingData->tCubeTiling.M, block.params_.singleCoreN,
                               matmulv3TilingData->tCubeTiling.Kb, matmulv3TilingData->tCubeTiling.Kb,
                               matmulv3TilingData->tCubeTiling.N);
                mm.Iterate();
                mm.GetTensorC(cGlobal[block.offset_.offsetC], enAtomic);
            }
        }
    }
    InQueueBL1.FreeTensor(bl1Local);
    mm.SetHF32(false, 0);
}

} // namespace Mc2MatmulV3Advanced
#endif // __OP_KERNEL_MAT_MUL_V3_FULL_LOAD_KERNEL_HELPER_H__
