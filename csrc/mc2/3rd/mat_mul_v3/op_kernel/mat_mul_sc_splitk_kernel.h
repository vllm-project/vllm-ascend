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
 * \file mat_mul_sc_splitk_kernel.h
 * \brief
 */
#ifndef __OP_KERNEL_MATMUL_V3_SC_SPLITK_KERNEL_H__
#define __OP_KERNEL_MATMUL_V3_SC_SPLITK_KERNEL_H__

#include "mat_mul_sc_splitk_block.h"

using namespace AscendC;
using namespace matmul;

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE,
          class BLOCK_TYPE = Mc2MatmulSingleCoreSplitKBaseBlock, const MatmulConfig &MM_CFG = MM_CFG_PRELOAD_MK,
          const bool IS_NKM = false>
class Mc2MatMulBaseKernelSingleCoreSplitK {
public:
    __aicore__ inline Mc2MatMulBaseKernelSingleCoreSplitK()
    {
    }

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                GM_ADDR workspaceGM, const Mc2MatmulV3TilingData *matmulTilingData, TPipe *pipe);

    __aicore__ inline void UnAlignedInit(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                         GM_ADDR workspaceGM, const Mc2MatmulV3TilingData *matmulTilingData,
                                         TPipe *pipe);

    __aicore__ inline void UpdateGlobalTensor(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                              GM_ADDR workspaceGM);

    __aicore__ inline void ProcessBlockMKN(GM_ADDR cGM, GM_ADDR srcAddr, TBuf<TPosition::VECCALC> &ubBuf);

    __aicore__ inline void ProcessBlockNKM(GM_ADDR cGM, GM_ADDR srcAddr, TBuf<TPosition::VECCALC> &ubBuf);

    __aicore__ inline void Process(GM_ADDR cGM, GM_ADDR srcAddr, TBuf<TPosition::VECCALC> &ubBuf);

    __aicore__ inline void UnAlignedProcess();

    __aicore__ inline void SetParamAndExec(int kIndex);

    __aicore__ inline void Exector();

    __aicore__ inline void End()
    {
        mm_.End();
    }

protected:
    BLOCK_TYPE block_;
    MatmulImpl<A_TYPE, B_TYPE, L0C_TYPE, BIAS_TYPE, MM_CFG> mm_;
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename L0C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    GlobalTensor<A_T> aGlobal_;
    GlobalTensor<B_T> bGlobal_;
    GlobalTensor<C_T> cGlobal_;
    GlobalTensor<BiasT> biasGlobal_;
    TPipe *pipe_;
    bool n128AlignFlag_ = false;

private:
    __aicore__ inline void InitInputs(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR workspaceGM);
    __aicore__ inline void SetOrgShape();
};

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
          const MatmulConfig &MM_CFG, const bool IS_NKM>
__aicore__ inline void
Mc2MatMulBaseKernelSingleCoreSplitK<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, IS_NKM>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const Mc2MatmulV3TilingData *matmulTilingData, TPipe *pipe)
{
    block_.template Init<A_TYPE, B_TYPE, L0C_TYPE, BIAS_TYPE>(matmulTilingData);
    n128AlignFlag_ = (block_.matmulTilingData_->matmulTiling.N % ALIGN_128_BYTE == 0);
    if ASCEND_IS_AIV {
        return;
    }
    SetAtomicNone();
    if (GetCurrentBlockIdx() >= block_.matmulTilingData_->matmulTiling.usedCoreNum) {
        return;
    }
    pipe_ = pipe;
    InitInputs(aGM, bGM, cGM, biasGM, workspaceGM);

    mm_.SetSubBlockIdx(0);
    PRELOAD(4);
    mm_.Init(&block_.matmulTilingData_->matmulTiling, pipe_);
    SetOrgShape();
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
          const MatmulConfig &MM_CFG, const bool IS_NKM>
__aicore__ inline void
Mc2MatMulBaseKernelSingleCoreSplitK<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
                                    IS_NKM>::UnAlignedInit(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM,
                                                           GM_ADDR offsetWGM, GM_ADDR workspaceGM,
                                                           const Mc2MatmulV3TilingData *matmulTilingData, TPipe *pipe)
{
    if ASCEND_IS_AIV {
        return;
    }
    SetAtomicNone();
    block_.template Init<A_TYPE, B_TYPE, L0C_TYPE, BIAS_TYPE>(matmulTilingData);
    if (GetCurrentBlockIdx() >= block_.matmulTilingData_->matmulTiling.usedCoreNum) {
        return;
    }
    pipe_ = pipe;
    InitInputs(aGM, bGM, cGM, biasGM, workspaceGM);

    mm_.SetSubBlockIdx(0);
    PRELOAD(4);
    mm_.Init(&block_.matmulTilingData_->matmulTiling, pipe_);
    SetOrgShape();
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
          const MatmulConfig &MM_CFG, const bool IS_NKM>
__aicore__ inline void Mc2MatMulBaseKernelSingleCoreSplitK<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE, BLOCK_TYPE,
                                                           MM_CFG, IS_NKM>::InitInputs(GM_ADDR aGM, GM_ADDR bGM,
                                                                                       GM_ADDR cGM, GM_ADDR biasGM,
                                                                                       GM_ADDR workspaceGM)
{
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename L0C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(aGM),
                             static_cast<uint64_t>(block_.matmulTilingData_->matmulTiling.M) *
                                 block_.matmulTilingData_->matmulTiling.Ka);
    bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM),
                             static_cast<uint64_t>(block_.matmulTilingData_->matmulTiling.Kb) *
                                 block_.matmulTilingData_->matmulTiling.N);
    cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(cGM),
                             static_cast<uint64_t>(block_.matmulTilingData_->matmulTiling.M) *
                                 block_.matmulTilingData_->matmulTiling.N);
    biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT *>(biasGM), block_.matmulTilingData_->matmulTiling.N);
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
          const MatmulConfig &MM_CFG, const bool IS_NKM>
__aicore__ inline void Mc2MatMulBaseKernelSingleCoreSplitK<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE, BLOCK_TYPE,
                                                           MM_CFG, IS_NKM>::SetOrgShape()
{
    if constexpr (A_TYPE::format == CubeFormat::NZ && B_TYPE::format == CubeFormat::NZ) {
        mm_.SetOrgShape(block_.params_.alignedOriM, block_.params_.alignedOriN, block_.params_.alignedKaSize,
                        block_.params_.alignedKbSize, block_.params_.outNAlign);
    } else if constexpr (A_TYPE::format == CubeFormat::NZ) {
        mm_.SetOrgShape(block_.params_.alignedOriM, block_.matmulTilingData_->matmulTiling.N,
                        block_.params_.alignedKaSize, block_.matmulTilingData_->matmulTiling.Kb,
                        block_.params_.outNAlign);
    } else if constexpr (B_TYPE::format == CubeFormat::NZ) {
        mm_.SetOrgShape(block_.matmulTilingData_->matmulTiling.M, block_.params_.alignedOriN,
                        block_.matmulTilingData_->matmulTiling.Ka, block_.params_.alignedKbSize,
                        block_.params_.outNAlign);
    } else {
        if (n128AlignFlag_) {
            mm_.SetOrgShape(block_.matmulTilingData_->matmulTiling.M, block_.matmulTilingData_->matmulTiling.N,
                            block_.matmulTilingData_->matmulTiling.Ka, block_.matmulTilingData_->matmulTiling.Kb,
                            block_.matmulTilingData_->matmulTiling.N);
        } else {
            mm_.SetOrgShape(block_.matmulTilingData_->matmulTiling.M, block_.matmulTilingData_->matmulTiling.N,
                            block_.matmulTilingData_->matmulTiling.Ka, block_.matmulTilingData_->matmulTiling.Kb,
                            block_.params_.outNAlign);
        }
    }
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
          const MatmulConfig &MM_CFG, const bool IS_NKM>
__aicore__ inline void
Mc2MatMulBaseKernelSingleCoreSplitK<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
                                    IS_NKM>::UpdateGlobalTensor(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM,
                                                                GM_ADDR offsetWGM, GM_ADDR workspaceGM)
{
    if (GetCurrentBlockIdx() >= block_.matmulTilingData_->matmulTiling.usedCoreNum) {
        return;
    }

    InitInputs(aGM, bGM, cGM, biasGM, workspaceGM);
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
          const MatmulConfig &MM_CFG, const bool IS_NKM>
__aicore__ inline void
Mc2MatMulBaseKernelSingleCoreSplitK<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
                                    IS_NKM>::ProcessBlockMKN(GM_ADDR cGM, GM_ADDR srcAddr,
                                                             TBuf<TPosition::VECCALC> &ubBuf)
{
    for (uint64_t innerMIndex = 0; innerMIndex < block_.params_.innerLoopM; ++innerMIndex) {
        if ASCEND_IS_AIV {
            // Cast f322f16
            WaitFlagDevLocal(AIC_SYNC_AIV_FLAG);
            // do_cast C：innerSingleCoreM * nCoreUse
            block_.UpdateBlockParamsMk(innerMIndex, 0);
            uint64_t singleMOffset = block_.params_.mIndex * block_.matmulTilingData_->matmulTiling.singleCoreM;
            uint64_t innerMOffset = innerMIndex * block_.params_.innerBlockM;
            uint64_t offset = (singleMOffset + innerMOffset) * block_.matmulTilingData_->matmulTiling.N +
                              block_.params_.nIndex * block_.matmulTilingData_->matmulTiling.singleCoreN;
            uint64_t vMOffset = MMV3DivCeil(block_.params_.innerSingleCoreM, NUM_TWO);
            if (GetBlockIdx() % NUM_TWO == 1) { // 一个C核对应两个V核中的第二个V核的计算处理
                offset = offset + vMOffset * block_.matmulTilingData_->matmulTiling.N;
                vMOffset = block_.params_.innerSingleCoreM - vMOffset;
            }
            uint64_t singleSize = vMOffset * block_.params_.nCoreUse;
            Cast32to16V220(reinterpret_cast<__gm__ typename OUTPUT_TYPE::T *>(cGM) + offset,
                           reinterpret_cast<__gm__ float *>(srcAddr) + offset, singleSize, block_.params_.nCoreUse,
                           block_.matmulTilingData_->matmulTiling.N, ubBuf);
            PipeBarrier<PIPE_ALL>();
        }
        if ASCEND_IS_AIC {
            for (uint64_t kIndex = 0; kIndex < block_.params_.loopK; ++kIndex) {
                block_.UpdateBlockParamsMk(innerMIndex, kIndex);
                for (uint64_t innerNIndex = 0; innerNIndex < block_.params_.innerLoopN; ++innerNIndex) {
                    block_.template CalcGMOffset<A_TYPE, B_TYPE, L0C_TYPE, BIAS_TYPE>(innerMIndex, kIndex, innerNIndex,
                                                                                      false);
                    SetParamAndExec(kIndex);
                }
            }
            // c侧做完才能做v侧
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
            NotifyEvent<PIPE_FIX>(AIC_SYNC_AIV_FLAG);
#endif
            PipeBarrier<PIPE_ALL>();
        }
    }
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
          const MatmulConfig &MM_CFG, const bool IS_NKM>
__aicore__ inline void
Mc2MatMulBaseKernelSingleCoreSplitK<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
                                    IS_NKM>::ProcessBlockNKM(GM_ADDR cGM, GM_ADDR srcAddr,
                                                             TBuf<TPosition::VECCALC> &ubBuf)
{
    for (uint64_t innerNIndex = 0; innerNIndex < block_.params_.innerLoopN; ++innerNIndex) {
        if ASCEND_IS_AIV {
            // Cast f322f16
            WaitFlagDevLocal(AIC_SYNC_AIV_FLAG);
            // do_cast C：mCoreUse * innerSingleCoreN
            block_.UpdateBlockParamsNk(innerNIndex, 0);
            uint64_t singleNOffset = block_.params_.nIndex * block_.matmulTilingData_->matmulTiling.singleCoreN;
            uint64_t innerNOffset = innerNIndex * block_.params_.innerBlockN;
            uint64_t offset = (singleNOffset + innerNOffset) + block_.params_.mIndex *
                                                                   block_.matmulTilingData_->matmulTiling.singleCoreM *
                                                                   block_.matmulTilingData_->matmulTiling.N;
            uint64_t vMOffset = MMV3DivCeil(block_.params_.mCoreUse, NUM_TWO);
            if (GetBlockIdx() % NUM_TWO == 1) { // 一个C核对应两个V核中的第二个V核的计算处理
                offset = offset + vMOffset * block_.matmulTilingData_->matmulTiling.N;
                vMOffset = block_.params_.innerSingleCoreM - vMOffset;
            }
            uint64_t singleSize = vMOffset * block_.params_.innerSingleCoreN;
            Cast32to16V220(reinterpret_cast<__gm__ typename OUTPUT_TYPE::T *>(cGM) + offset,
                           reinterpret_cast<__gm__ float *>(srcAddr) + offset, singleSize,
                           block_.params_.innerSingleCoreN, block_.matmulTilingData_->matmulTiling.N, ubBuf);
            PipeBarrier<PIPE_ALL>();
        }

        if ASCEND_IS_AIC {
            for (uint64_t kIndex = 0; kIndex < block_.params_.loopK; ++kIndex) {
                block_.UpdateBlockParamsNk(innerNIndex, kIndex);
                for (uint64_t innerMIndex = 0; innerMIndex < block_.params_.innerLoopM; ++innerMIndex) {
                    block_.template CalcGMOffset<A_TYPE, B_TYPE, L0C_TYPE, BIAS_TYPE>(innerMIndex, kIndex, innerNIndex,
                                                                                      true);
                    SetParamAndExec(kIndex);
                }
            }
            // c侧做完才能做v侧
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
            NotifyEvent<PIPE_FIX>(AIC_SYNC_AIV_FLAG);
#endif
            PipeBarrier<PIPE_ALL>();
        }
    }
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
          const MatmulConfig &MM_CFG, const bool IS_NKM>
__aicore__ inline void Mc2MatMulBaseKernelSingleCoreSplitK<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE, BLOCK_TYPE,
                                                           MM_CFG, IS_NKM>::Process(GM_ADDR cGM, GM_ADDR srcAddr,
                                                                                    TBuf<TPosition::VECCALC> &ubBuf)
{
    block_.InitBlockIndex();
    if ASCEND_IS_AIC {
        mm_.SetHF32(block_.params_.isHf32, 1); // 1: round mode is round to the nearest tie away from zero
        //  SetL2CacheHint 的逻辑: 左、右矩阵仅搬运一次的情形
        if (IS_NKM && (block_.matmulTilingData_->matmulTiling.stepN * block_.matmulTilingData_->matmulTiling.baseN ==
                       block_.matmulTilingData_->matmulTiling.N)) {
            aGlobal_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        } else if (!IS_NKM &&
                   (block_.matmulTilingData_->matmulTiling.stepM * block_.matmulTilingData_->matmulTiling.baseM ==
                    block_.matmulTilingData_->matmulTiling.M)) {
            bGlobal_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
    }
    for (uint64_t j = 0; j < block_.params_.realRound; ++j) {
        block_.UpdateBlockCnt();
        if constexpr (!IS_NKM) {
            ProcessBlockMKN(cGM, srcAddr, ubBuf);
        } else {
            ProcessBlockNKM(cGM, srcAddr, ubBuf);
        }

        block_.UpdateBlockIndex();
    }
    PipeBarrier<PIPE_ALL>();
    SetAtomicNone();
    if ASCEND_IS_AIC {
        mm_.SetHF32(false, 0);
    }
    return;
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
          const MatmulConfig &MM_CFG, const bool IS_NKM>
__aicore__ inline void Mc2MatMulBaseKernelSingleCoreSplitK<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE, BLOCK_TYPE,
                                                           MM_CFG, IS_NKM>::Exector()
{
    if constexpr (!IS_NKM) {
        for (uint64_t innerMIndex = 0; innerMIndex < block_.params_.innerLoopM; ++innerMIndex) {
            for (int kIndex = 0; kIndex < block_.params_.loopK; ++kIndex) {
                block_.UpdateBlockParamsMk(innerMIndex, kIndex);
                for (uint64_t innerNIndex = 0; innerNIndex < block_.params_.innerLoopN; ++innerNIndex) {
                    block_.template CalcGMOffset<A_TYPE, B_TYPE, L0C_TYPE, BIAS_TYPE>(innerMIndex, kIndex, innerNIndex,
                                                                                      IS_NKM);
                    SetParamAndExec(kIndex);
                }
            }
        }
    } else {
        for (uint64_t innerNIndex = 0; innerNIndex < block_.params_.innerLoopN; ++innerNIndex) {
            for (int kIndex = 0; kIndex < block_.params_.loopK; ++kIndex) {
                block_.UpdateBlockParamsNk(innerNIndex, kIndex);
                for (uint64_t innerMIndex = 0; innerMIndex < block_.params_.innerLoopM; ++innerMIndex) {
                    block_.template CalcGMOffset<A_TYPE, B_TYPE, L0C_TYPE, BIAS_TYPE>(innerMIndex, kIndex, innerNIndex,
                                                                                      IS_NKM);
                    SetParamAndExec(kIndex);
                }
            }
        }
    }
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
          const MatmulConfig &MM_CFG, const bool IS_NKM>
__aicore__ inline void Mc2MatMulBaseKernelSingleCoreSplitK<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE, BLOCK_TYPE,
                                                           MM_CFG, IS_NKM>::SetParamAndExec(int kIndex)
{
    mm_.SetSingleShape(block_.params_.innerSingleCoreM, block_.params_.innerSingleCoreN, block_.params_.kCoreUse);
    mm_.SetTensorA(aGlobal_[block_.offset_.offsetA], block_.params_.isTransposeA);
    mm_.SetTensorB(bGlobal_[block_.offset_.offsetB], block_.params_.isTransposeB);
    if (kIndex == 0) {
        block_.params_.atomicAddFlag = false;
        if (block_.matmulTilingData_->matmulTiling.isBias) {
            mm_.SetBias(biasGlobal_[block_.offset_.offsetBias]);
        }
    } else {
        block_.params_.atomicAddFlag = true;
    }
    mm_.IterateAll(cGlobal_[block_.offset_.offsetC], block_.params_.atomicAddFlag);
    mm_.ClearBias();
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
          const MatmulConfig &MM_CFG, const bool IS_NKM>
__aicore__ inline void Mc2MatMulBaseKernelSingleCoreSplitK<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE, BLOCK_TYPE,
                                                           MM_CFG, IS_NKM>::UnAlignedProcess()
{
    if ASCEND_IS_AIV {
        return;
    }

    if (GetCurrentBlockIdx() >= block_.matmulTilingData_->matmulTiling.usedCoreNum) {
        return;
    }
    mm_.SetHF32(false, 0);
    if (block_.params_.isHf32) {
        mm_.SetHF32(true, 1);
    }

    block_.InitBlockIndex();
    for (uint64_t j = 0; j < block_.params_.realRound; ++j) {
        block_.UpdateBlockCnt();
        Exector();
        block_.UpdateBlockIndex();
    }
    PipeBarrier<PIPE_ALL>();
    SetAtomicNone();
    mm_.SetHF32(false, 0);
    return;
}


template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE,
          class BLOCK_TYPE = Mc2MatmulSingleCoreSplitKBaseBlock, const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD,
          const bool IS_NKM = false>
class Mc2MatMulSingleCoreSplitKKernel {
    struct SingleCoreSplitKParams {
        GM_ADDR alignedworkspaceGM;
        uint64_t vIndex;
        uint64_t alignedN;
        uint64_t coreSizeNum;
        uint64_t offset;
        GM_ADDR cGM;
        bool n128Align = false;
    };

public:
    __aicore__ inline Mc2MatMulSingleCoreSplitKKernel()
    {
    }
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                GM_ADDR workspaceGM, const Mc2MatmulV3TilingData *matmulTilingData, TPipe *pipe);
    __aicore__ inline void UpdateGlobalTensor(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                              GM_ADDR workspaceGM);
    __aicore__ inline void Process();
    __aicore__ inline void NNot128AlignProcess();
    __aicore__ inline void End()
    {
        mmcBaseKernel_.End();
    }

protected:
    using cType = MatmulType<C_TYPE::pos, C_TYPE::format, float, C_TYPE::isTrans>;
    Mc2MatMulBaseKernelSingleCoreSplitK<A_TYPE, B_TYPE, cType, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, IS_NKM>
        mmcBaseKernel_;

    TPipe *pipe_;
    TBuf<> ubBuf_;
    const Mc2MatmulV3TilingData *matmulTilingData_;
    SingleCoreSplitKParams innerParams_;
    GlobalTensor<float> cTmpGlobal_;
    GlobalTensor<float> matmulOutput_;
    GlobalTensor<typename C_TYPE::T> castCGm_;

private:
    __aicore__ inline void ProcessRemovePaddingImpl();
    __aicore__ inline void InitInputs(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                      GM_ADDR workspaceGM);
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          const bool IS_NKM>
__aicore__ inline void
Mc2MatMulSingleCoreSplitKKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, IS_NKM>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const Mc2MatmulV3TilingData *matmulTilingData, TPipe *pipe)
{
    pipe_ = pipe;
    matmulTilingData_ = matmulTilingData;
    innerParams_.n128Align = (matmulTilingData_->matmulTiling.N % ALIGN_128_BYTE == 0);

    InitInputs(aGM, bGM, cGM, biasGM, offsetWGM, workspaceGM);
    if ASCEND_IS_AIC {
        if (GetBlockIdx() >= matmulTilingData_->matmulTiling.usedCoreNum) {
            return;
        }
        using C_T = typename C_TYPE::T;
        if constexpr (sizeof(C_T) == sizeof(float)) {
            innerParams_.alignedworkspaceGM = innerParams_.cGM;
        }
        if (!innerParams_.n128Align) {
            mmcBaseKernel_.UnAlignedInit(aGM, bGM, innerParams_.alignedworkspaceGM, biasGM, offsetWGM, workspaceGM,
                                         matmulTilingData, pipe_);
            return;
        }
    }
    mmcBaseKernel_.Init(aGM, bGM, innerParams_.alignedworkspaceGM, biasGM, offsetWGM, workspaceGM, matmulTilingData,
                        pipe_);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          const bool IS_NKM>
__aicore__ inline void
Mc2MatMulSingleCoreSplitKKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, IS_NKM>::InitInputs(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM)
{
    innerParams_.alignedworkspaceGM = reinterpret_cast<GM_ADDR>(
        ((reinterpret_cast<uint64_t>(workspaceGM + MAX_BLOCK_NUM * DEFAULT_BLOCK_LEN * sizeof(int32_t)) + 511) / 512) *
        512);
    innerParams_.cGM = cGM;

    if ASCEND_IS_AIV {
        using C_T = typename C_TYPE::T;
        if constexpr (sizeof(C_T) == sizeof(float)) {
            return;
        }
        // Clear gm
        innerParams_.vIndex = GetBlockIdx();
        if (innerParams_.vIndex >= (matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO)) {
            return;
        }
        uint64_t totalSize = static_cast<uint64_t>(matmulTilingData_->matmulTiling.M) *
                             static_cast<uint64_t>(matmulTilingData_->matmulTiling.N);
        uint64_t coreSize = totalSize / (matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO); // need to align
        innerParams_.coreSizeNum = coreSize;
        innerParams_.offset = innerParams_.vIndex * coreSize;
        if (innerParams_.vIndex == matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO - 1) {
            // 尾块数据量
            innerParams_.coreSizeNum =
                totalSize - (matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO - 1) * coreSize;
        }
        cTmpGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(innerParams_.alignedworkspaceGM), totalSize);
        pipe_->InitBuffer(ubBuf_, TOTAL_UB_SIZE);
        if (matmulTilingData_->matmulTiling.N * DATA_SIZE_FP32 % ALIGN_BYTE != 0) {
            innerParams_.alignedN = MMV3DivCeil(matmulTilingData_->matmulTiling.N, 64) * 64;
            matmulOutput_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(innerParams_.alignedworkspaceGM),
                                          matmulTilingData_->matmulTiling.M * innerParams_.alignedN);
            castCGm_.SetGlobalBuffer(reinterpret_cast<__gm__ typename C_TYPE::T *>(cGM),
                                     matmulTilingData_->matmulTiling.M * matmulTilingData_->matmulTiling.N);
        }
        return;
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          const bool IS_NKM>
__aicore__ inline void
Mc2MatMulSingleCoreSplitKKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, IS_NKM>::UpdateGlobalTensor(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM)
{
    InitInputs(aGM, bGM, cGM, biasGM, offsetWGM, workspaceGM);

    if ASCEND_IS_AIC {
        if (GetBlockIdx() >= matmulTilingData_->matmulTiling.usedCoreNum) {
            return;
        }
        using C_T = typename C_TYPE::T;
        if constexpr (sizeof(C_T) == sizeof(float)) {
            innerParams_.alignedworkspaceGM = innerParams_.cGM;
        }
        mmcBaseKernel_.UpdateGlobalTensor(aGM, bGM, innerParams_.alignedworkspaceGM, biasGM, offsetWGM, workspaceGM);
        return;
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          const bool IS_NKM>
__aicore__ inline void Mc2MatMulSingleCoreSplitKKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
                                                       IS_NKM>::ProcessRemovePaddingImpl()
{
    if (matmulTilingData_->matmulTiling.N * DATA_SIZE_FP32 % ALIGN_BYTE != 0) {
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
            castCGm_[innerParams_.vIndex * splitM * matmulTilingData_->matmulTiling.N],
            matmulOutput_[innerParams_.vIndex * splitM * innerParams_.alignedN], coreMSize, innerParams_.alignedN,
            matmulTilingData_->matmulTiling.N, ubBuf_);
    } else {
        UnAlignedCast32to16V220(reinterpret_cast<__gm__ typename C_TYPE::T *>(innerParams_.cGM) + innerParams_.offset,
                                reinterpret_cast<__gm__ float *>(innerParams_.alignedworkspaceGM) + innerParams_.offset,
                                0, innerParams_.coreSizeNum, ubBuf_);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          const bool IS_NKM>
__aicore__ inline void
Mc2MatMulSingleCoreSplitKKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, IS_NKM>::Process()
{
    if (!innerParams_.n128Align) {
        NNot128AlignProcess();
        return;
    }
    using C_T = typename C_TYPE::T;
    if ASCEND_IS_AIV {
        if constexpr (sizeof(C_T) == sizeof(float)) {
            return;
        }
        if (GetBlockIdx() >= matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO) {
            return;
        }
        PipeBarrier<PIPE_ALL>();
    }
    if constexpr (sizeof(C_T) == sizeof(float)) {
        // fp32不需要vector核
        mmcBaseKernel_.UnAlignedProcess();
        return;
    }
    if ASCEND_IS_AIC {
        if (GetBlockIdx() >= matmulTilingData_->matmulTiling.usedCoreNum) {
            return;
        }
    }
    mmcBaseKernel_.Process(innerParams_.cGM, innerParams_.alignedworkspaceGM, ubBuf_);
    return;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          const bool IS_NKM>
__aicore__ inline void
Mc2MatMulSingleCoreSplitKKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, IS_NKM>::NNot128AlignProcess()
{
    using C_T = typename C_TYPE::T;
    if ASCEND_IS_AIV {
        if constexpr (sizeof(C_T) == sizeof(float)) {
            return;
        }
        if (GetBlockIdx() >= matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO) {
            NotifyEvent<PIPE_MTE3>(6);
            PipeBarrier<PIPE_ALL>();
            return;
        }
        SyncAll();
        NotifyEvent<PIPE_MTE3>(6);
        PipeBarrier<PIPE_ALL>();
        // Cast f322f16
        WaitFlagDevLocal(5);
        SyncAll();
        PipeBarrier<PIPE_ALL>();

        ProcessRemovePaddingImpl();

        PipeBarrier<PIPE_ALL>();
        return;
    }
    if constexpr (sizeof(C_T) == sizeof(float)) {
        // fp32不需要vector核
        mmcBaseKernel_.UnAlignedProcess();
        return;
    }
    if ASCEND_IS_AIC {
        WaitFlagDevLocal(6);
        if (GetBlockIdx() >= matmulTilingData_->matmulTiling.usedCoreNum) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
            NotifyEvent<PIPE_FIX>(5);
#endif
            PipeBarrier<PIPE_ALL>();
            return;
        }
        mmcBaseKernel_.UnAlignedProcess();
        // c侧做完才能做v侧
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        NotifyEvent<PIPE_FIX>(5);
#endif
        PipeBarrier<PIPE_ALL>();
        return;
    }
}

#endif // MMV3_MATMUL_KERNEL_H
