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
 * \file mat_mul_sc_splitk_block.h
 * \brief
 */
#ifndef __OP_KERNEL_MATMUL_V3_SC_SPLITK_BLOCK_H__
#define __OP_KERNEL_MATMUL_V3_SC_SPLITK_BLOCK_H__

#include "mat_mul_v3_common.h"

using namespace AscendC;
using namespace matmul;


struct SingleCoreSplitKBaseBlockArgs {
    bool isTransposeA;
    bool isTransposeB;
    uint32_t isHf32;

    uint64_t mCnt;
    uint64_t nCnt;
    uint64_t kCnt;
    uint64_t mCoreTail;
    uint64_t nCoreTail;
    uint64_t kCoreTail;
    uint64_t loopK;

    uint64_t kTileL2;

    uint64_t innerBlockM;
    uint64_t innerBlockN;

    uint64_t innerLoopM;
    uint64_t innerLoopN;
    bool atomicAddFlag;

    uint64_t index;
    // M和N方向绑多核, 按照3*3个128*128的基本块
    uint64_t totalCnt;
    uint64_t round;
    uint64_t realRound;
    uint64_t preCoreNum;

    uint64_t mCoreUse;
    uint64_t nCoreUse;
    uint64_t kCoreUse;

    uint64_t mIndex;
    uint64_t nIndex;

    uint64_t innerSingleCoreM; // 增加
    uint64_t innerSingleCoreN; // 增加

    uint64_t outNAlign;
    uint64_t c0Size;
    uint64_t alignedOriM;
    uint64_t alignedOriN;
    uint64_t alignedKaSize;
    uint64_t alignedKbSize;
    uint32_t rowOrder; // 用于区分是否进入n轴错峰
};

class Mc2MatmulSingleCoreSplitKBaseBlock {
public:
    __aicore__ inline Mc2MatmulSingleCoreSplitKBaseBlock()
    {
    }
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void Init(const Mc2MatmulV3TilingData *matmulTilingData);
    __aicore__ inline void UpdateBlockCnt();
    __aicore__ inline void InitBlockIndex();
    __aicore__ inline void UpdateBlockParamsMk(uint64_t innerMIndex, uint64_t kIndex);
    __aicore__ inline void UpdateBlockParamsNk(uint64_t innerNIndex, uint64_t kIndex);
    __aicore__ inline void UpdateBlockIndex();
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void CalcGMOffset(uint64_t innerMIndex, uint64_t kIndex, uint64_t innerNIndex, bool isNKM);

public:
    BlockOffset offset_;
    SingleCoreSplitKBaseBlockArgs params_;
    const Mc2MatmulV3TilingData *matmulTilingData_;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void Mc2MatmulSingleCoreSplitKBaseBlock::Init(const Mc2MatmulV3TilingData *matmulTilingData)
{
    matmulTilingData_ = matmulTilingData;
    const Mc2L2cacheTilePara &tilingL2 = matmulTilingData_->tileL2cacheTiling;
    params_.rowOrder = tilingL2.calOrder;
    params_.isTransposeA = matmulTilingData_->matmulRunInfo.transA;
    params_.isTransposeB = matmulTilingData_->matmulRunInfo.transB;
    params_.isHf32 = matmulTilingData_->matmulRunInfo.isHf32;

    params_.mCnt = MMV3DivCeil(matmulTilingData_->matmulTiling.M, matmulTilingData_->matmulTiling.singleCoreM);
    params_.nCnt = MMV3DivCeil(matmulTilingData_->matmulTiling.N, matmulTilingData_->matmulTiling.singleCoreN);
    params_.kCnt = MMV3DivCeil(matmulTilingData_->matmulTiling.Ka, matmulTilingData_->matmulTiling.singleCoreK);
    params_.mCoreTail = static_cast<uint64_t>(matmulTilingData_->matmulTiling.M) -
                        (params_.mCnt - 1) * matmulTilingData_->matmulTiling.singleCoreM;
    params_.nCoreTail = static_cast<uint64_t>(matmulTilingData_->matmulTiling.N) -
                        (params_.nCnt - 1) * matmulTilingData_->matmulTiling.singleCoreN;
    params_.kCoreTail = static_cast<uint64_t>(matmulTilingData_->matmulTiling.Ka) -
                        (params_.kCnt - 1) * matmulTilingData_->matmulTiling.singleCoreK;
    params_.loopK = MMV3DivCeil(matmulTilingData_->matmulTiling.Ka, matmulTilingData_->matmulTiling.singleCoreK);
    params_.innerBlockM =
        static_cast<uint64_t>(matmulTilingData_->matmulTiling.stepM) * matmulTilingData_->matmulTiling.baseM;
    params_.innerBlockN =
        static_cast<uint64_t>(matmulTilingData_->matmulTiling.stepN) * matmulTilingData_->matmulTiling.baseN;
    params_.atomicAddFlag = false;
    // 记录3*3的基本块的位置
    params_.index = 0;
    // M和N方向绑多核, 按照3*3个128*128的基本块
    params_.totalCnt = params_.mCnt * params_.nCnt;
    params_.round = DivCeil(params_.totalCnt, matmulTilingData_->matmulTiling.usedCoreNum);
    params_.realRound = 0;
    params_.preCoreNum = params_.totalCnt % matmulTilingData_->matmulTiling.usedCoreNum;
    if (params_.preCoreNum == 0) {
        params_.preCoreNum = matmulTilingData_->matmulTiling.usedCoreNum;
    }

    params_.mCoreUse = matmulTilingData_->matmulTiling.singleCoreM;
    params_.nCoreUse = matmulTilingData_->matmulTiling.singleCoreN;
    params_.kCoreUse = matmulTilingData_->matmulTiling.singleCoreK;

    uint64_t cTypeSize = 64; // 64 means 256Byte
    params_.outNAlign = MMV3DivCeil(matmulTilingData_->matmulTiling.N, cTypeSize) * cTypeSize;
    using A_T = typename A_TYPE::T;
    if constexpr (sizeof(A_T) == sizeof(float)) {
        params_.outNAlign = matmulTilingData_->matmulTiling.N;
    }

    GetSizeC0<A_T>(params_.c0Size);
    params_.alignedOriM = MMV3DivCeil(matmulTilingData_->matmulTiling.M, ALIGNED_H) * ALIGNED_H;
    params_.alignedOriN = MMV3DivCeil(matmulTilingData_->matmulTiling.N, params_.c0Size) * params_.c0Size;
    params_.alignedKaSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Ka, params_.c0Size) * params_.c0Size;
    params_.alignedKbSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Kb, ALIGNED_H) * ALIGNED_H;
    // A B矩阵都是对齐矩阵
    if (params_.isTransposeA) {
        params_.alignedOriM = MMV3DivCeil(matmulTilingData_->matmulTiling.M, params_.c0Size) * params_.c0Size;
        params_.alignedKaSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Ka, ALIGNED_H) * ALIGNED_H;
    }
    if (params_.isTransposeB) {
        params_.alignedOriN = MMV3DivCeil(matmulTilingData_->matmulTiling.N, ALIGNED_H) * ALIGNED_H;
        params_.alignedKbSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Kb, params_.c0Size) * params_.c0Size;
    }
}

__aicore__ inline void Mc2MatmulSingleCoreSplitKBaseBlock::UpdateBlockCnt()
{
    params_.mIndex = params_.index % params_.mCnt;
    params_.nIndex = params_.index / params_.mCnt;
    if (params_.index == (params_.totalCnt - 1)) {
        // 最后一块是尾块
        params_.mCoreUse = params_.mCoreTail;
        params_.nCoreUse = params_.nCoreTail;
    } else if (params_.mIndex == (params_.mCnt - 1)) {
        // m方向尾块
        params_.mCoreUse = params_.mCoreTail;
        params_.nCoreUse = matmulTilingData_->matmulTiling.singleCoreN;
    } else if (params_.nIndex == (params_.nCnt - 1)) {
        // n方向尾块
        params_.mCoreUse = matmulTilingData_->matmulTiling.singleCoreM;
        params_.nCoreUse = params_.nCoreTail;
    } else {
        // 对齐整块
        params_.mCoreUse = matmulTilingData_->matmulTiling.singleCoreM;
        params_.nCoreUse = matmulTilingData_->matmulTiling.singleCoreN;
    }
    params_.innerLoopM = MMV3DivCeil(matmulTilingData_->matmulTiling.singleCoreM, params_.innerBlockM);
    params_.innerLoopN = MMV3DivCeil(matmulTilingData_->matmulTiling.singleCoreN, params_.innerBlockN);
    if (params_.mIndex == params_.mCnt - 1) {
        params_.innerLoopM = DivCeil(params_.mCoreTail, params_.innerBlockM);
    }
    if (params_.nIndex == params_.nCnt - 1) {
        params_.innerLoopN = DivCeil(params_.nCoreTail, params_.innerBlockN);
    }
    if (params_.rowOrder == 0) { // 单核切k中 l2IterateOrder 为默认值0时走原kernel
        params_.innerLoopN = 1;
    } else { // 走NKM模板
        params_.innerLoopM = 1;
    }
}

__aicore__ inline void Mc2MatmulSingleCoreSplitKBaseBlock::InitBlockIndex()
{
    if (GetCurrentBlockIdx() < params_.preCoreNum) {
        params_.index = GetCurrentBlockIdx() * params_.round;
        params_.realRound = params_.round;
    } else {
        params_.index = GetCurrentBlockIdx() * (params_.round - 1) + params_.preCoreNum;
        params_.realRound = params_.round - 1;
    }
}

__aicore__ inline void Mc2MatmulSingleCoreSplitKBaseBlock::UpdateBlockParamsMk(uint64_t innerMIndex, uint64_t kIndex)
{
    params_.innerSingleCoreM = params_.innerBlockM;
    if (innerMIndex == params_.innerLoopM - 1) {
        params_.innerSingleCoreM = params_.mCoreUse - (params_.innerLoopM - 1) * params_.innerBlockM;
    }
    if (kIndex == params_.loopK - 1) {
        params_.kCoreUse = params_.kCoreTail;
    } else {
        params_.kCoreUse = matmulTilingData_->matmulTiling.singleCoreK;
    }
}

__aicore__ inline void Mc2MatmulSingleCoreSplitKBaseBlock::UpdateBlockParamsNk(uint64_t innerNIndex, uint64_t kIndex)
{
    params_.innerSingleCoreN = params_.innerBlockN;
    if (innerNIndex == params_.innerLoopN - 1) {
        params_.innerSingleCoreN = params_.nCoreUse - (params_.innerLoopN - 1) * params_.innerBlockN;
    }
    if (kIndex == params_.loopK - 1) {
        params_.kCoreUse = params_.kCoreTail;
    } else {
        params_.kCoreUse = matmulTilingData_->matmulTiling.singleCoreK;
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void Mc2MatmulSingleCoreSplitKBaseBlock::CalcGMOffset(uint64_t innerMIndex, uint64_t kIndex,
                                                                        uint64_t innerNIndex, bool isNKM)
{
    uint64_t innerNShiftIndex = (GetCurrentBlockIdx() + innerNIndex) % params_.innerLoopN;
    params_.innerSingleCoreN = params_.innerBlockN;
    if (innerNShiftIndex == params_.innerLoopN - 1) {
        params_.innerSingleCoreN = params_.nCoreUse - innerNShiftIndex * params_.innerBlockN;
    }
    if (isNKM) {
        uint64_t innerMShiftIndex = (GetCurrentBlockIdx() + innerMIndex) % params_.innerLoopM;
        params_.innerSingleCoreM = params_.innerBlockM;
        if (innerMShiftIndex == params_.innerLoopM - 1) {
            params_.innerSingleCoreM = params_.mCoreUse - innerMShiftIndex * params_.innerBlockM;
        }
        innerMIndex = innerMShiftIndex;
        innerNShiftIndex = innerNIndex;
    }

    if constexpr (A_TYPE::format == CubeFormat::ND) {
        if (params_.isTransposeA) {
            offset_.offsetA =
                (params_.mIndex * matmulTilingData_->matmulTiling.singleCoreM + innerMIndex * params_.innerBlockM +
                 kIndex * matmulTilingData_->matmulTiling.singleCoreK * matmulTilingData_->matmulTiling.M);
        } else {
            offset_.offsetA =
                ((params_.mIndex * matmulTilingData_->matmulTiling.singleCoreM + innerMIndex * params_.innerBlockM) *
                     matmulTilingData_->matmulTiling.Ka +
                 kIndex * matmulTilingData_->matmulTiling.singleCoreK);
        }
    } else if constexpr (A_TYPE::format == CubeFormat::NZ) {
        if (params_.isTransposeA) {
            offset_.offsetA =
                ((params_.mIndex * matmulTilingData_->matmulTiling.singleCoreM + innerMIndex * params_.innerBlockM) *
                     params_.alignedKaSize +
                 kIndex * matmulTilingData_->matmulTiling.singleCoreK * params_.c0Size);
        } else {
            offset_.offsetA =
                ((params_.mIndex * matmulTilingData_->matmulTiling.singleCoreM + innerMIndex * params_.innerBlockM) *
                     params_.c0Size +
                 kIndex * matmulTilingData_->matmulTiling.singleCoreK * params_.alignedOriM);
        }
    }
    if constexpr (B_TYPE::format == CubeFormat::ND) {
        if (params_.isTransposeB) {
            offset_.offsetB = (kIndex * matmulTilingData_->matmulTiling.singleCoreK +
                               (params_.nIndex * matmulTilingData_->matmulTiling.singleCoreN +
                                innerNShiftIndex * params_.innerBlockN) *
                                   matmulTilingData_->matmulTiling.Kb);
        } else {
            offset_.offsetB =
                (params_.nIndex * matmulTilingData_->matmulTiling.singleCoreN + innerNShiftIndex * params_.innerBlockN +
                 kIndex * matmulTilingData_->matmulTiling.singleCoreK * matmulTilingData_->matmulTiling.N);
        }
    } else if constexpr (B_TYPE::format == CubeFormat::NZ) {
        if (params_.isTransposeB) {
            offset_.offsetB = (kIndex * matmulTilingData_->matmulTiling.singleCoreK * params_.alignedOriN +
                               (params_.nIndex * matmulTilingData_->matmulTiling.singleCoreN +
                                innerNShiftIndex * params_.innerBlockN) *
                                   params_.c0Size);
        } else {
            offset_.offsetB = ((params_.nIndex * matmulTilingData_->matmulTiling.singleCoreN +
                                innerNShiftIndex * params_.innerBlockN) *
                                   params_.alignedKbSize +
                               kIndex * matmulTilingData_->matmulTiling.singleCoreK * params_.c0Size);
        }
    }
    offset_.offsetC =
        ((params_.mIndex * matmulTilingData_->matmulTiling.singleCoreM + innerMIndex * params_.innerBlockM) *
             params_.outNAlign +
         params_.nIndex * matmulTilingData_->matmulTiling.singleCoreN + innerNShiftIndex * params_.innerBlockN);
    if (A_TYPE::format == CubeFormat::ND && B_TYPE::format == CubeFormat::ND &&
        (matmulTilingData_->matmulTiling.N % ALIGN_128_BYTE == 0)) {
        offset_.offsetC =
            ((params_.mIndex * matmulTilingData_->matmulTiling.singleCoreM + innerMIndex * params_.innerBlockM) *
                 matmulTilingData_->matmulTiling.N +
             params_.nIndex * matmulTilingData_->matmulTiling.singleCoreN + innerNShiftIndex * params_.innerBlockN);
    }
    offset_.offsetBias = params_.nIndex * matmulTilingData_->matmulTiling.singleCoreN;
}

__aicore__ inline void Mc2MatmulSingleCoreSplitKBaseBlock::UpdateBlockIndex()
{
    params_.index += 1;
}

#endif // MMV3_MATMUL_BLOCK_H
