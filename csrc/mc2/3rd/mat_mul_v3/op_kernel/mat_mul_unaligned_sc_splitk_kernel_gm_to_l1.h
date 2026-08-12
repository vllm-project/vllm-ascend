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
 * \file mat_mul_unaligned_sc_splitk_kernel_gm_to_l1.h
 * \brief
 */
#ifndef __OP_KERNEL_MATMUL_V3_UNALIGNED_SC_SPLITK_KERNEL_GM_TO_L1_H__
#define __OP_KERNEL_MATMUL_V3_UNALIGNED_SC_SPLITK_KERNEL_GM_TO_L1_H__

#include "mat_mul_v3_common.h"
#include "mat_mul_sc_splitk_block.h"
#include "mat_mul_nd2nz.h"

using namespace AscendC;
using namespace matmul;

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE,
          class BLOCK_TYPE = Mc2MatmulSingleCoreSplitKBaseBlock, const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD>
class Mc2MatMulUnAlignedSingleCoreSplitKKernelGmToL1 {
    struct UnAlignedSingleCoreSplitKParams {
        int nd2nzFlag;
        GM_ADDR alignedworkspaceGM;
        GM_ADDR castAddr;
        uint64_t vIndex;
        uint64_t alignedN;
        uint64_t coreSizeNum;
        uint64_t offset;
        uint64_t alignedOriM;
        uint64_t alignedOriN;
        uint64_t alignedKaSize;
        uint64_t alignedKbSize;
        bool isTransposeAIn;
        bool isTransposeBIn;
        bool nd2nzA;
        bool nd2nzB;
        uint64_t inputDtypeSize;
        GM_ADDR aGM;
        GM_ADDR bGM;
        GM_ADDR cGM;
        uint64_t baseAN;
        uint64_t baseAD;
        uint64_t baseBN;
        uint64_t baseBD;
        // A B矩阵都是对齐矩阵
    };

public:
    __aicore__ inline Mc2MatMulUnAlignedSingleCoreSplitKKernelGmToL1()
    {
    }
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                GM_ADDR workspaceGM, const Mc2MatmulV3TilingData *matmulTilingData, TPipe *pipe);
    __aicore__ inline void UpdateGlobalTensor(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                              GM_ADDR workspaceGM);
    __aicore__ inline void Process();

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
    using aType = MatmulType<A_TYPE::pos, CubeFormat::NZ, typename A_TYPE::T, A_TYPE::isTrans>;
    using bType = MatmulType<B_TYPE::pos, CubeFormat::NZ, typename B_TYPE::T, B_TYPE::isTrans>;
    using cType = MatmulType<C_TYPE::pos, C_TYPE::format, float, C_TYPE::isTrans>;
    Mc2MatMulBaseKernelSingleCoreSplitKGmToL1<aType, B_TYPE, cType, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG> mma_;
    Mc2MatMulBaseKernelSingleCoreSplitKGmToL1<A_TYPE, bType, cType, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG> mmb_;
    Mc2MatMulBaseKernelSingleCoreSplitKGmToL1<aType, bType, cType, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG> mmab_;
    GlobalTensor<float> matmulOutput_;
    GlobalTensor<typename C_TYPE::T> castCGm_;
    TPipe *pipe_;
    TBuf<> ubBuf_;
    UnAlignedSingleCoreSplitKParams innerParams_;
    const Mc2MatmulV3TilingData *matmulTilingData_;

private:
    __aicore__ inline void ProcessNDtoNZ();
    __aicore__ inline void InitInputs(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                      GM_ADDR workspaceGM);
    __aicore__ inline void ProcessRemovePaddingImpl();
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatMulUnAlignedSingleCoreSplitKKernelGmToL1<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const Mc2MatmulV3TilingData *matmulTilingData, TPipe *pipe)
{
    matmulTilingData_ = matmulTilingData;
    innerParams_.isTransposeAIn = matmulTilingData_->matmulRunInfo.transA;
    innerParams_.isTransposeBIn = matmulTilingData_->matmulRunInfo.transB;
    innerParams_.nd2nzA = matmulTilingData_->matmulRunInfo.nd2nzA;
    innerParams_.nd2nzB = matmulTilingData_->matmulRunInfo.nd2nzB;
    innerParams_.baseAN = matmulTilingData->baseAN;
    innerParams_.baseAD = matmulTilingData->baseAD;
    innerParams_.baseBN = matmulTilingData->baseBN;
    innerParams_.baseBD = matmulTilingData->baseBD;

    InitInputs(aGM, bGM, cGM, biasGM, offsetWGM, workspaceGM);

    pipe_ = pipe;
    pipe_->InitBuffer(ubBuf_, TOTAL_UB_SIZE);

    if ASCEND_IS_AIC {
        if (GetBlockIdx() >= matmulTilingData_->matmulTiling.usedCoreNum) {
            return;
        }

        if constexpr (sizeof(typename C_TYPE::T) == sizeof(float)) {
            innerParams_.castAddr = innerParams_.cGM;
        }

        if (innerParams_.nd2nzFlag == static_cast<int32_t>(ND2NZ_SELECT::ONLY_B)) {
            mmb_.UnAlignedInit(innerParams_.aGM, innerParams_.alignedworkspaceGM, innerParams_.castAddr, biasGM,
                               offsetWGM, workspaceGM, matmulTilingData, pipe_);
        } else if (innerParams_.nd2nzFlag == static_cast<int32_t>(ND2NZ_SELECT::ONLY_A)) {
            mma_.UnAlignedInit(innerParams_.alignedworkspaceGM, innerParams_.bGM, innerParams_.castAddr, biasGM,
                               offsetWGM, workspaceGM, matmulTilingData, pipe_);
        } else if (innerParams_.nd2nzFlag == static_cast<int32_t>(ND2NZ_SELECT::BOTH_AB)) {
            mmab_.UnAlignedInit(innerParams_.alignedworkspaceGM,
                                innerParams_.alignedworkspaceGM +
                                    innerParams_.alignedOriM * innerParams_.alignedKaSize * innerParams_.inputDtypeSize,
                                innerParams_.castAddr, biasGM, offsetWGM, workspaceGM, matmulTilingData, pipe_);
        }
        return;
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatMulUnAlignedSingleCoreSplitKKernelGmToL1<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::InitInputs(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM)
{
    innerParams_.alignedworkspaceGM = reinterpret_cast<GM_ADDR>(
        ((reinterpret_cast<uint64_t>(workspaceGM + MAX_BLOCK_NUM * DEFAULT_BLOCK_LEN * sizeof(int32_t)) + 511) / 512) *
        512);
    innerParams_.aGM = aGM;
    innerParams_.bGM = bGM;
    innerParams_.cGM = cGM;

    using A_T = typename A_TYPE::T;
    innerParams_.inputDtypeSize = sizeof(A_T);
    uint64_t c0Size;
    GetSizeC0<A_T>(c0Size);
    innerParams_.alignedOriM = MMV3DivCeil(matmulTilingData_->matmulTiling.M, ALIGNED_H) * ALIGNED_H;
    innerParams_.alignedOriN = MMV3DivCeil(matmulTilingData_->matmulTiling.N, c0Size) * c0Size;
    innerParams_.alignedKaSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Ka, c0Size) * c0Size;
    innerParams_.alignedKbSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Kb, ALIGNED_H) * ALIGNED_H;
    // A B矩阵都是对齐矩阵
    if (innerParams_.isTransposeAIn) {
        innerParams_.alignedOriM = MMV3DivCeil(matmulTilingData_->matmulTiling.M, c0Size) * c0Size;
        innerParams_.alignedKaSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Ka, ALIGNED_H) * ALIGNED_H;
    }
    if (innerParams_.isTransposeBIn) {
        innerParams_.alignedOriN = MMV3DivCeil(matmulTilingData_->matmulTiling.N, ALIGNED_H) * ALIGNED_H;
        innerParams_.alignedKbSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Kb, c0Size) * c0Size;
    }

    innerParams_.castAddr = innerParams_.alignedworkspaceGM;
    if (innerParams_.nd2nzA) {
        innerParams_.nd2nzFlag = static_cast<int32_t>(ND2NZ_SELECT::ONLY_A);
        innerParams_.castAddr = innerParams_.alignedworkspaceGM +
                                innerParams_.alignedOriM * innerParams_.alignedKaSize * innerParams_.inputDtypeSize;
    }
    if (innerParams_.nd2nzB) {
        innerParams_.nd2nzFlag = static_cast<int32_t>(ND2NZ_SELECT::ONLY_B);
        innerParams_.castAddr = innerParams_.alignedworkspaceGM +
                                innerParams_.alignedKbSize * innerParams_.alignedOriN * innerParams_.inputDtypeSize;
    }
    if (innerParams_.nd2nzA && innerParams_.nd2nzB) {
        bool isAFullLoad = static_cast<uint64_t>(matmulTilingData_->matmulTiling.baseM) *
                               matmulTilingData_->matmulTiling.baseK * matmulTilingData_->matmulTiling.depthA1 >=
                           innerParams_.alignedOriM * innerParams_.alignedKaSize;
        bool isBFullLoad = static_cast<uint64_t>(matmulTilingData_->matmulTiling.baseN) *
                               matmulTilingData_->matmulTiling.baseK * matmulTilingData_->matmulTiling.depthB1 >=
                           innerParams_.alignedOriN * innerParams_.alignedKbSize;
        if (isAFullLoad) {
            innerParams_.nd2nzFlag = static_cast<int32_t>(ND2NZ_SELECT::ONLY_B);
            innerParams_.castAddr = innerParams_.alignedworkspaceGM +
                                    innerParams_.alignedOriN * innerParams_.alignedKbSize * innerParams_.inputDtypeSize;
        } else if (isBFullLoad) {
            innerParams_.nd2nzFlag = static_cast<int32_t>(ND2NZ_SELECT::ONLY_A);
            innerParams_.castAddr = innerParams_.alignedworkspaceGM +
                                    innerParams_.alignedOriM * innerParams_.alignedKaSize * innerParams_.inputDtypeSize;
        } else {
            innerParams_.nd2nzFlag = static_cast<int32_t>(ND2NZ_SELECT::BOTH_AB);
            innerParams_.castAddr =
                innerParams_.alignedworkspaceGM + (innerParams_.alignedOriM + innerParams_.alignedOriN) *
                                                      innerParams_.alignedKaSize * innerParams_.inputDtypeSize;
        }
    }
    if ASCEND_IS_AIV {
        // Clear gm
        innerParams_.vIndex = GetBlockIdx();
        if (innerParams_.vIndex >= (matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO)) {
            return;
        }
        if constexpr (sizeof(typename C_TYPE::T) == sizeof(float)) {
            return;
        }
        uint64_t totalSize =
            static_cast<uint64_t>(matmulTilingData_->matmulTiling.M) * matmulTilingData_->matmulTiling.N;
        uint64_t coreSize = totalSize / (matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO); // need to align
        innerParams_.coreSizeNum = coreSize;
        innerParams_.offset = innerParams_.vIndex * coreSize;
        if (innerParams_.vIndex == matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO - 1) {
            // 尾块数据量
            innerParams_.coreSizeNum =
                totalSize - (matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO - 1) * coreSize;
        }
        if (static_cast<uint64_t>(matmulTilingData_->matmulTiling.N) * DATA_SIZE_FP32 % ALIGN_BYTE != 0) {
            innerParams_.alignedN = MMV3DivCeil(matmulTilingData_->matmulTiling.N, 64) * 64;
            matmulOutput_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(innerParams_.castAddr),
                                          static_cast<uint64_t>(matmulTilingData_->matmulTiling.M) *
                                              innerParams_.alignedN);
            castCGm_.SetGlobalBuffer(reinterpret_cast<__gm__ typename C_TYPE::T *>(cGM),
                                     static_cast<uint64_t>(matmulTilingData_->matmulTiling.M) *
                                         matmulTilingData_->matmulTiling.N);
        }
        return;
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatMulUnAlignedSingleCoreSplitKKernelGmToL1<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE,
                                               MM_CFG>::UpdateGlobalTensor(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM,
                                                                           GM_ADDR biasGM, GM_ADDR offsetWGM,
                                                                           GM_ADDR workspaceGM)
{
    InitInputs(aGM, bGM, cGM, biasGM, offsetWGM, workspaceGM);

    if ASCEND_IS_AIC {
        if (GetBlockIdx() >= matmulTilingData_->matmulTiling.usedCoreNum) {
            return;
        }
        if constexpr (sizeof(typename A_TYPE::T) == sizeof(float)) {
            innerParams_.castAddr = innerParams_.cGM;
        }
        if (innerParams_.nd2nzFlag == static_cast<int32_t>(ND2NZ_SELECT::ONLY_B)) {
            mmb_.UpdateGlobalTensor(innerParams_.aGM, innerParams_.alignedworkspaceGM, innerParams_.castAddr, biasGM,
                                    offsetWGM, workspaceGM);
        } else if (innerParams_.nd2nzFlag == static_cast<int32_t>(ND2NZ_SELECT::ONLY_A)) {
            mma_.UpdateGlobalTensor(innerParams_.alignedworkspaceGM, innerParams_.bGM, innerParams_.castAddr, biasGM,
                                    offsetWGM, workspaceGM);
        } else if (innerParams_.nd2nzFlag == static_cast<int32_t>(ND2NZ_SELECT::BOTH_AB)) {
            mmab_.UpdateGlobalTensor(innerParams_.alignedworkspaceGM,
                                     innerParams_.alignedworkspaceGM + innerParams_.alignedOriM *
                                                                           innerParams_.alignedKaSize *
                                                                           innerParams_.inputDtypeSize,
                                     innerParams_.castAddr, biasGM, offsetWGM, workspaceGM);
        }
        return;
    }
}


template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatMulUnAlignedSingleCoreSplitKKernelGmToL1<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::ProcessNDtoNZ()
{
    // ND2NZ
    if (innerParams_.nd2nzFlag == static_cast<int32_t>(ND2NZ_SELECT::ONLY_B)) {
        MatrixBtoNZV2<typename B_TYPE::T>(innerParams_.alignedworkspaceGM, innerParams_.bGM,
                                          matmulTilingData_->matmulTiling, innerParams_.isTransposeBIn, ubBuf_,
                                          innerParams_.baseBN, innerParams_.baseBD);
    } else if (innerParams_.nd2nzFlag == static_cast<int32_t>(ND2NZ_SELECT::ONLY_A)) {
        MatrixAtoNZV2<typename A_TYPE::T>(innerParams_.alignedworkspaceGM, innerParams_.aGM,
                                          matmulTilingData_->matmulTiling, innerParams_.isTransposeAIn, ubBuf_,
                                          innerParams_.baseAN, innerParams_.baseAD);
    } else if (innerParams_.nd2nzFlag == static_cast<int32_t>(ND2NZ_SELECT::BOTH_AB)) {
        MatrixAtoNZV2<typename A_TYPE::T>(innerParams_.alignedworkspaceGM, innerParams_.aGM,
                                          matmulTilingData_->matmulTiling, innerParams_.isTransposeAIn, ubBuf_,
                                          innerParams_.baseAN, innerParams_.baseAD);
        event_t eventMTE3MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2);
        MatrixBtoNZV2<typename B_TYPE::T>(
            innerParams_.alignedworkspaceGM +
                innerParams_.alignedOriM * innerParams_.alignedKaSize * innerParams_.inputDtypeSize,
            innerParams_.bGM, matmulTilingData_->matmulTiling, innerParams_.isTransposeBIn, ubBuf_, innerParams_.baseBN,
            innerParams_.baseBD);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void Mc2MatMulUnAlignedSingleCoreSplitKKernelGmToL1<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE,
                                                                      MM_CFG>::ProcessRemovePaddingImpl()
{
    if (static_cast<uint64_t>(matmulTilingData_->matmulTiling.N) * DATA_SIZE_FP32 % ALIGN_BYTE != 0) {
        uint64_t splitM = matmulTilingData_->matmulTiling.M / (matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO);
        uint64_t coreMSize = splitM;
        if (matmulTilingData_->matmulTiling.M < (matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO)) {
            splitM = 1;
            if (innerParams_.vIndex * splitM >= matmulTilingData_->matmulTiling.M) {
                PipeBarrier<PIPE_ALL>();
                return;
            }
            coreMSize = splitM;
        } else {
            if (innerParams_.vIndex == matmulTilingData_->matmulTiling.usedCoreNum * 2 - 1) {
                coreMSize = matmulTilingData_->matmulTiling.M - coreMSize * innerParams_.vIndex;
            }
        }
        RemovePaddingImpl<float, typename C_TYPE::T>(
            castCGm_[innerParams_.vIndex * splitM * static_cast<uint64_t>(matmulTilingData_->matmulTiling.N)],
            matmulOutput_[innerParams_.vIndex * splitM * innerParams_.alignedN], coreMSize, innerParams_.alignedN,
            matmulTilingData_->matmulTiling.N, ubBuf_);
    } else {
        UnAlignedCast32to16V220(reinterpret_cast<__gm__ typename C_TYPE::T *>(innerParams_.cGM) + innerParams_.offset,
                                reinterpret_cast<__gm__ float *>(innerParams_.castAddr) + innerParams_.offset, 0,
                                innerParams_.coreSizeNum, ubBuf_);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatMulUnAlignedSingleCoreSplitKKernelGmToL1<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Process()
{
    using C_T = typename C_TYPE::T;
    if ASCEND_IS_AIV {
        if (GetBlockIdx() >= (matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO)) {
            NotifyEvent<PIPE_MTE3>(6);
            PipeBarrier<PIPE_ALL>();
            return;
        }
        ProcessNDtoNZ();

        SyncAll();
        NotifyEvent<PIPE_MTE3>(6);
        PipeBarrier<PIPE_ALL>();
        if constexpr (sizeof(C_T) == sizeof(float)) {
            return;
        }
        WaitFlagDevLocal(5);
        SyncAll();
        PipeBarrier<PIPE_ALL>();

        ProcessRemovePaddingImpl();

        PipeBarrier<PIPE_ALL>();
        return;
    }

    if ASCEND_IS_AIC {
        WaitFlagDevLocal(6);
        if (GetBlockIdx() >= matmulTilingData_->matmulTiling.usedCoreNum) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
            if constexpr (sizeof(C_T) != sizeof(float)) {
                NotifyEvent<PIPE_FIX>(5);
            }
#endif
            PipeBarrier<PIPE_ALL>();
            return;
        }

        if (innerParams_.nd2nzFlag == 2) {
            mmb_.UnAlignedProcess();
        } else if (innerParams_.nd2nzFlag == 1) {
            mma_.UnAlignedProcess();
        } else if (innerParams_.nd2nzFlag == 3) {
            mmab_.UnAlignedProcess();
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        if constexpr (sizeof(C_T) != sizeof(float)) {
            NotifyEvent<PIPE_FIX>(5);
        }
#endif
        PipeBarrier<PIPE_ALL>();
        return;
    }
}
#endif // MMV3_MATMUL_KERNEL_H
