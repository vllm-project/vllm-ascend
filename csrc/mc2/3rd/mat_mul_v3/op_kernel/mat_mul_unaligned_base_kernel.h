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
 * \file mat_mul_unaligned_base_kernel.h
 * \brief
 */
#ifndef __OP_KERNEL_MATMUL_V3_UNALIGNED_BASE_KERNEL_H__
#define __OP_KERNEL_MATMUL_V3_UNALIGNED_BASE_KERNEL_H__

#include "mat_mul_base_block.h"
#include "mat_mul_base_kernel.h"
#include "mat_mul_nd2nz.h"


using namespace AscendC;
using namespace matmul;

const uint64_t CV_FLAG = 4;

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = Mc2MatmulBaseBlock,
          const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD>
class Mc2MatmulBaseUnAlignedKernel {
    struct BaseUnAlignedKernelParams {
        bool isTransposeA;
        bool isTransposeB;
        int nd2nzFlag; // 2表示B矩阵做nd2nz，1表示A矩阵做nd2nz
        GM_ADDR aGMNZ;
        GM_ADDR bGMNZ;
        GM_ADDR workspaceGMNZ;
        GM_ADDR workspaceGMabNZ;
        uint64_t baseAN;
        uint64_t baseAD;
        uint64_t baseBN;
        uint64_t baseBD;
    };

public:
    __aicore__ inline Mc2MatmulBaseUnAlignedKernel()
    {
    }

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                GM_ADDR workspaceGM, const Mc2MatmulV3TilingData *matmulTilingData, TPipe *pipe);

    __aicore__ inline void UpdateGlobalTensor(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                              GM_ADDR workspaceGM);

    __aicore__ inline void Process(uint64_t index = 0, uint8_t enAtomic = 0);

    __aicore__ inline void End()
    {
        if (innerParams_.nd2nzFlag == static_cast<int32_t>(ND2NZ_SELECT::ONLY_B)) {
            mmb_.End();
        } else if (innerParams_.nd2nzFlag == static_cast<int32_t>(ND2NZ_SELECT::ONLY_A)) {
            mma_.End();
        } else if (innerParams_.nd2nzFlag == static_cast<int32_t>(ND2NZ_SELECT::BOTH_AB)) {
            mmab_.End();
        }
    }

protected:
    __aicore__ inline void ProcessNDtoNZ();
    __aicore__ inline void CalculateabGM(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                         GM_ADDR workspaceGM);
    using aType = MatmulType<A_TYPE::pos, CubeFormat::NZ, typename A_TYPE::T, A_TYPE::isTrans>;
    using bType = MatmulType<B_TYPE::pos, CubeFormat::NZ, typename B_TYPE::T, B_TYPE::isTrans>;
    using a_T = typename aType::T;
    using b_T = typename bType::T;
    Mc2MatmulBaseKernel<aType, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG> mma_;
    Mc2MatmulBaseKernel<A_TYPE, bType, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG> mmb_;
    Mc2MatmulBaseKernel<aType, bType, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG> mmab_;
    BaseUnAlignedKernelParams innerParams_;
    TPipe *pipe_;
    TBuf<> ubBuf_;
    const Mc2MatmulV3TilingData *matmulTilingData_;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void Mc2MatmulBaseUnAlignedKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const Mc2MatmulV3TilingData *matmulTilingData, TPipe *pipe)
{
    pipe_ = pipe;
    pipe_->InitBuffer(ubBuf_, TOTAL_UB_SIZE);
    matmulTilingData_ = matmulTilingData;
    innerParams_.isTransposeA = matmulTilingData_->matmulRunInfo.transA;
    innerParams_.isTransposeB = matmulTilingData_->matmulRunInfo.transB;
    bool nd2nzA = matmulTilingData_->matmulRunInfo.nd2nzA;
    bool nd2nzB = matmulTilingData_->matmulRunInfo.nd2nzB;
    innerParams_.baseAN = matmulTilingData->baseAN;
    innerParams_.baseAD = matmulTilingData->baseAD;
    innerParams_.baseBN = matmulTilingData->baseBN;
    innerParams_.baseBD = matmulTilingData->baseBD;

    if (nd2nzA) {
        innerParams_.nd2nzFlag = static_cast<int32_t>(ND2NZ_SELECT::ONLY_A);
    }
    if (nd2nzB) {
        innerParams_.nd2nzFlag = static_cast<int32_t>(ND2NZ_SELECT::ONLY_B);
    }
    if (nd2nzA && nd2nzB) {
        innerParams_.nd2nzFlag = static_cast<int32_t>(ND2NZ_SELECT::BOTH_AB);
    }

    CalculateabGM(aGM, bGM, cGM, biasGM, offsetWGM, workspaceGM);
    if (innerParams_.nd2nzFlag == static_cast<int32_t>(ND2NZ_SELECT::ONLY_B)) {
        mmb_.Init(aGM, innerParams_.workspaceGMNZ, cGM, biasGM, offsetWGM, workspaceGM, matmulTilingData_, pipe_);
    } else if (innerParams_.nd2nzFlag == static_cast<int32_t>(ND2NZ_SELECT::ONLY_A)) {
        mma_.Init(innerParams_.workspaceGMNZ, bGM, cGM, biasGM, offsetWGM, workspaceGM, matmulTilingData_, pipe_);
    } else if (innerParams_.nd2nzFlag == static_cast<int32_t>(ND2NZ_SELECT::BOTH_AB)) {
        mmab_.Init(innerParams_.workspaceGMNZ, innerParams_.workspaceGMabNZ, cGM, biasGM, offsetWGM, workspaceGM,
                   matmulTilingData_, pipe_);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatmulBaseUnAlignedKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::UpdateGlobalTensor(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM)
{
    CalculateabGM(aGM, bGM, cGM, biasGM, offsetWGM, workspaceGM);
    if (innerParams_.nd2nzFlag == static_cast<int32_t>(ND2NZ_SELECT::ONLY_B)) {
        mmb_.UpdateGlobalTensor(aGM, innerParams_.workspaceGMNZ, cGM, biasGM, offsetWGM, workspaceGM);
    } else if (innerParams_.nd2nzFlag == static_cast<int32_t>(ND2NZ_SELECT::ONLY_A)) {
        mma_.UpdateGlobalTensor(innerParams_.workspaceGMNZ, bGM, cGM, biasGM, offsetWGM, workspaceGM);
    } else if (innerParams_.nd2nzFlag == static_cast<int32_t>(ND2NZ_SELECT::BOTH_AB)) {
        mmab_.UpdateGlobalTensor(innerParams_.workspaceGMNZ, innerParams_.workspaceGMabNZ, cGM, biasGM, offsetWGM,
                                 workspaceGM);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatmulBaseUnAlignedKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::CalculateabGM(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM)
{
    innerParams_.aGMNZ = aGM;
    innerParams_.bGMNZ = bGM;
    using A_T = typename A_TYPE::T;
    uint64_t c0Size;
    GetSizeC0<A_T>(c0Size);
    auto alignedMSize = MMV3DivCeil(matmulTilingData_->matmulTiling.M, ALIGNED_H) * ALIGNED_H; // N轴转换成分型
    auto alignedKSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Ka, c0Size) * c0Size;      // K轴转换成分型
    if (innerParams_.isTransposeA) {
        alignedMSize = MMV3DivCeil(matmulTilingData_->matmulTiling.M, c0Size) * c0Size;        // N轴转换成分型
        alignedKSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Ka, ALIGNED_H) * ALIGNED_H; // K轴转换成分型
    }
    uint64_t inputDtypeSize = sizeof(typename A_TYPE::T);
    innerParams_.workspaceGMNZ = workspaceGM;
    innerParams_.workspaceGMabNZ = workspaceGM + alignedMSize * alignedKSize * inputDtypeSize;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatmulBaseUnAlignedKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::ProcessNDtoNZ()
{
    // ND2NZ
    if (innerParams_.nd2nzFlag == static_cast<int32_t>(ND2NZ_SELECT::ONLY_B)) {
        MatrixBtoNZV2<typename B_TYPE::T>(innerParams_.workspaceGMNZ, innerParams_.bGMNZ,
                                          matmulTilingData_->matmulTiling, innerParams_.isTransposeB, ubBuf_,
                                          innerParams_.baseBN, innerParams_.baseBD);
    } else if (innerParams_.nd2nzFlag == static_cast<int32_t>(ND2NZ_SELECT::ONLY_A)) {
        MatrixAtoNZV2<typename A_TYPE::T>(innerParams_.workspaceGMNZ, innerParams_.aGMNZ,
                                          matmulTilingData_->matmulTiling, innerParams_.isTransposeA, ubBuf_,
                                          innerParams_.baseAN, innerParams_.baseAD);
    } else if (innerParams_.nd2nzFlag == static_cast<int32_t>(ND2NZ_SELECT::BOTH_AB)) {
        MatrixAtoNZV2<typename A_TYPE::T>(innerParams_.workspaceGMNZ, innerParams_.aGMNZ,
                                          matmulTilingData_->matmulTiling, innerParams_.isTransposeA, ubBuf_,
                                          innerParams_.baseAN, innerParams_.baseAD);
        event_t eventMTE3MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2);
        MatrixBtoNZV2<typename B_TYPE::T>(innerParams_.workspaceGMabNZ, innerParams_.bGMNZ,
                                          matmulTilingData_->matmulTiling, innerParams_.isTransposeB, ubBuf_,
                                          innerParams_.baseBN, innerParams_.baseBD);
    }
    SyncAll();
    // CV SYNC
    if ASCEND_IS_AIV {
        NotifyEvent<PIPE_MTE3>(CV_FLAG);
    }
    if ASCEND_IS_AIC {
        WaitEvent(CV_FLAG);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatmulBaseUnAlignedKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Process(uint64_t index,
                                                                                             uint8_t enAtomic)
{
    ProcessNDtoNZ();
    if (innerParams_.nd2nzFlag == static_cast<int32_t>(ND2NZ_SELECT::ONLY_B)) {
        mmb_.Process(index, enAtomic);
    } else if (innerParams_.nd2nzFlag == static_cast<int32_t>(ND2NZ_SELECT::ONLY_A)) {
        mma_.Process(index, enAtomic);
    } else if (innerParams_.nd2nzFlag == static_cast<int32_t>(ND2NZ_SELECT::BOTH_AB)) {
        mmab_.Process(index, enAtomic);
    }
}


#endif // MMV3_MATMUL_KERNEL_H
