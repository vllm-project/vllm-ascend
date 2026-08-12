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
 * \file mat_mul_sc_splitk_kernel_gm_to_l1.h
 * \brief
 */
#ifndef __OP_KERNEL_MATMUL_V3_SC_SPLITK_KERNEL_GM_TO_L1_H__
#define __OP_KERNEL_MATMUL_V3_SC_SPLITK_KERNEL_GM_TO_L1_H__

#include "mat_mul_sc_splitk_block.h"

using namespace AscendC;
using namespace matmul;

namespace MatMulBaseKernelSingleCoreSplitKGmToL1Constant {
constexpr uint32_t L1_MAX_SIZE_910B_NO_BIAS = 512 * 1024;
constexpr uint32_t L1_MAX_SIZE_910B_HAS_BIAS = 511 * 1024;
constexpr uint32_t L1_CUT_BLOCK_NUM_24 = 64 * 1024;
constexpr uint32_t L1_CUT_BLOCK_NUM_33 = 48 * 1024;
constexpr uint32_t LOC_MAX_SIZE_910B = 128 * 1024;
constexpr uint32_t M_0 = 0;
constexpr uint32_t M_1 = 1;
constexpr uint32_t M_2 = 2;
constexpr uint32_t M0 = 128;
constexpr uint32_t N0 = 128;
constexpr uint32_t K_INDEX_0 = 0;
constexpr uint32_t N_INDEX_0 = 0;
} // namespace MatMulBaseKernelSingleCoreSplitKGmToL1Constant
using namespace MatMulBaseKernelSingleCoreSplitKGmToL1Constant;

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE,
          class BLOCK_TYPE = Mc2MatmulSingleCoreSplitKBaseBlock, const MatmulConfig &MM_CFG = MM_CFG_MDL>
class Mc2MatMulBaseKernelSingleCoreSplitKGmToL1 {
public:
    __aicore__ inline Mc2MatMulBaseKernelSingleCoreSplitKGmToL1()
    {
    }

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                GM_ADDR workspaceGM, const Mc2MatmulV3TilingData *matmulTilingData, TPipe *pipe);

    __aicore__ inline void UnAlignedInit(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                         GM_ADDR workspaceGM, const Mc2MatmulV3TilingData *matmulTilingData,
                                         TPipe *pipe);

    __aicore__ inline void UpdateGlobalTensor(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                              GM_ADDR workspaceGM);

    __aicore__ inline void Process(GM_ADDR cGM, GM_ADDR srcAddr, TBuf<TPosition::VECCALC> &ubBuf);

    __aicore__ inline void UnAlignedProcess();

    __aicore__ inline void End()
    {
        mm_.End();
    }

protected:
    BLOCK_TYPE block_;
    using AL1 = MatmulType<AscendC::TPosition::A1, CubeFormat::NZ, DTYPE_X1, A_TYPE::isTrans>;
    using BL1 = MatmulType<AscendC::TPosition::B1, CubeFormat::NZ, DTYPE_X1, B_TYPE::isTrans>;
    using CL0 = MatmulType<AscendC::TPosition::CO1, CubeFormat::NZ, float>;
    MatmulImpl<AL1, BL1, CL0, BIAS_TYPE, MM_CFG_MDL> mm_;
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
    TQue<TPosition::CO1, 1> C01Ping_;
    TQue<TPosition::CO1, 1> C01Pong_;
    LocalTensor<float> L0cPing_;
    LocalTensor<float> L0cPong_;
    TBuf<TPosition::A1> l1TBuf_;
    LocalTensor<A_T> aL1Ping;
    LocalTensor<A_T> aL1Pong;
    LocalTensor<A_T> aL1Peng;
    LocalTensor<B_T> bL1Ping;
    LocalTensor<B_T> bL1Pong;
    LocalTensor<B_T> bL1;
    LocalTensor<B_T> bL2;
    LocalTensor<float> l0c_;
    event_t eventIdBPingMte1Mte2;
    event_t eventIdBPongMte1Mte2;
    event_t eventIdBPingMte2Mte1;
    event_t eventIdBPongMte2Mte1;
    event_t eventIdAPingMte1Mte2;
    event_t eventIdAPongMte1Mte2;
    event_t eventIdAPingMte2Mte1;
    event_t eventIdAPongMte2Mte1;
    event_t eventIdAPengMte1Mte2;
    event_t eventIdAPengMte2Mte1;
    int32_t pingMReal;
    int32_t pongMReal;
    int32_t pengMReal;
    uint32_t l0cFixMAndMFIX;
    bool bpingflag = true;
    bool l0cpingflag = true;
    uint32_t nmloop_ = 1;
    uint32_t nkloop_ = 1;
    uint32_t nnloop_ = 1;
    uint32_t k0 = 512;

private:
    __aicore__ inline void InitInputs(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR workspaceGM);
    __aicore__ inline void CopyInAl1(uint32_t mloop, int32_t realK, int32_t kLoop, uint32_t k0);
    __aicore__ inline void CopyInAl1ND(uint32_t mloop, int32_t realK, int32_t kLoop, uint32_t k0);
    __aicore__ inline void CopyInAl1NZ(uint32_t mloop, int32_t realK, int32_t kLoop, uint32_t k0);
    __aicore__ inline void CopyInBl1(bool bpingflag, int32_t realK, int32_t kLoop, int32_t realN, int32_t nLoop);
    __aicore__ inline void CopyInBl1ND(bool bpingflag, int32_t realK, int32_t kLoop, int32_t realN, int32_t nLoop);
    __aicore__ inline void CopyInBl1NZ(bool bpingflag, int32_t realK, int32_t kLoop, int32_t realN, int32_t nLoop);
    __aicore__ inline void AMatMulB(bool bpingflag, int32_t nloop, int32_t k_real, int32_t n_real, uint64_t kIndex,
                                    LocalTensor<A_T> aL1, event_t eventAMte1WaitMte2, uint32_t mloop);
    __aicore__ inline void FreeEvent();
    __aicore__ inline void GetEvent();
    __aicore__ inline void VectorProcess(uint64_t innerMIndex, GM_ADDR cGM, GM_ADDR srcAddr,
                                         TBuf<TPosition::VECCALC> &ubBuf);
    __aicore__ inline void ProcessBaseMNK(bool lastK, int32_t realK, int32_t nextRealK, uint64_t kIndex);
    __aicore__ inline void UnAlignedProcessBaseMNK(bool lastK, int32_t realK, int32_t realKb, int32_t nextRealK,
                                                   uint64_t kIndex);
    __aicore__ inline void SetEventFlag();
    __aicore__ inline void WaitEventFlag();
    __aicore__ inline void SetOrgShape();
    __aicore__ inline void InitL1Buffer();
};

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
          const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatMulBaseKernelSingleCoreSplitKGmToL1<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Init(
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
    GetEvent();
    InitL1Buffer();
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
          const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatMulBaseKernelSingleCoreSplitKGmToL1<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE, BLOCK_TYPE,
                                          MM_CFG>::UnAlignedInit(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM,
                                                                 GM_ADDR offsetWGM, GM_ADDR workspaceGM,
                                                                 const Mc2MatmulV3TilingData *matmulTilingData,
                                                                 TPipe *pipe)
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
    GetEvent();
    InitL1Buffer();
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
          const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatMulBaseKernelSingleCoreSplitKGmToL1<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE, BLOCK_TYPE,
                                          MM_CFG>::InitInputs(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM,
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
          const MatmulConfig &MM_CFG>
__aicore__ inline void Mc2MatMulBaseKernelSingleCoreSplitKGmToL1<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE,
                                                                 BLOCK_TYPE, MM_CFG>::SetOrgShape()
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
          const MatmulConfig &MM_CFG>
__aicore__ inline void Mc2MatMulBaseKernelSingleCoreSplitKGmToL1<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE,
                                                                 BLOCK_TYPE, MM_CFG>::InitL1Buffer()
{
    if (block_.matmulTilingData_->matmulTiling.isBias) {
        pipe_->InitBuffer(l1TBuf_, L1_MAX_SIZE_910B_HAS_BIAS);
    } else {
        pipe_->InitBuffer(l1TBuf_, L1_MAX_SIZE_910B_NO_BIAS);
    }
    if (block_.matmulTilingData_->matmulTiling.stepM == 2 || block_.matmulTilingData_->matmulTiling.stepM == 1) {
        // 2*4算法
        bL1Ping = l1TBuf_.Get<B_T>()[0];
        bL1Pong = l1TBuf_.Get<B_T>()[1 * L1_CUT_BLOCK_NUM_24];
        aL1Ping = l1TBuf_.Get<A_T>()[2 * L1_CUT_BLOCK_NUM_24];
        aL1Pong = l1TBuf_.Get<A_T>()[3 * L1_CUT_BLOCK_NUM_24];
    } else {
        // 3*3算法
        bL1Ping = l1TBuf_.Get<B_T>()[0];
        bL1Pong = l1TBuf_.Get<B_T>()[1 * L1_CUT_BLOCK_NUM_33];
        aL1Ping = l1TBuf_.Get<A_T>()[2 * L1_CUT_BLOCK_NUM_33];
        aL1Pong = l1TBuf_.Get<A_T>()[3 * L1_CUT_BLOCK_NUM_33];
        aL1Peng = l1TBuf_.Get<A_T>()[4 * L1_CUT_BLOCK_NUM_33];
    }
    pipe_->InitBuffer(C01Ping_, 1, LOC_MAX_SIZE_910B / 2);
    pipe_->InitBuffer(C01Pong_, 1, LOC_MAX_SIZE_910B / 2);
    L0cPing_ = C01Ping_.template AllocTensor<float>();
    L0cPong_ = C01Pong_.template AllocTensor<float>();
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
          const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatMulBaseKernelSingleCoreSplitKGmToL1<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE, BLOCK_TYPE,
                                          MM_CFG>::UpdateGlobalTensor(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM,
                                                                      GM_ADDR biasGM, GM_ADDR offsetWGM,
                                                                      GM_ADDR workspaceGM)
{
    if (GetCurrentBlockIdx() >= block_.matmulTilingData_->matmulTiling.usedCoreNum) {
        return;
    }

    InitInputs(aGM, bGM, cGM, biasGM, workspaceGM);
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
          const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatMulBaseKernelSingleCoreSplitKGmToL1<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE, BLOCK_TYPE,
                                          MM_CFG>::Process(GM_ADDR cGM, GM_ADDR srcAddr,
                                                           TBuf<TPosition::VECCALC> &ubBuf)
{
    block_.InitBlockIndex();
    if ASCEND_IS_AIC {
        mm_.SetHF32(block_.params_.isHf32, 1); // 1: round mode is round to the nearest tie away from zero
    }
    if ASCEND_IS_AIC {
        SetEventFlag();
    }
    for (uint64_t j = 0; j < block_.params_.realRound; ++j) {
        block_.UpdateBlockCnt();
        for (uint64_t innerMIndex = 0; innerMIndex < block_.params_.innerLoopM; ++innerMIndex) {
            if ASCEND_IS_AIV {
                VectorProcess(innerMIndex, cGM, srcAddr, ubBuf);
            }
            if ASCEND_IS_AIC {
                block_.UpdateBlockParamsMk(innerMIndex, K_INDEX_0);
                block_.template CalcGMOffset<A_TYPE, B_TYPE, L0C_TYPE, BIAS_TYPE>(innerMIndex, K_INDEX_0, N_INDEX_0,
                                                                                  false);
                nmloop_ = MMV3DivCeil(block_.params_.innerSingleCoreM, M0);
                nnloop_ = MMV3DivCeil(block_.params_.innerSingleCoreN, N0);
                CopyInAl1ND(M_0, k0, K_INDEX_0, k0);
                if (nmloop_ - 1 >= M_1) {
                    CopyInAl1ND(M_1, k0, K_INDEX_0, k0);
                }
                int32_t nPingReal = N_INDEX_0 == nnloop_ - 1 ? block_.params_.innerSingleCoreN : N0;
                CopyInBl1ND(bpingflag, k0, K_INDEX_0, nPingReal, N_INDEX_0);
                for (uint64_t kIndex = 0; kIndex < block_.params_.loopK; ++kIndex) {
                    bool lastK = kIndex == block_.params_.loopK - 1;
                    int32_t realK = lastK ? block_.matmulTilingData_->matmulTiling.Ka - kIndex * k0 : k0;
                    int32_t nextRealK = kIndex == block_.params_.loopK - 2 ?
                                            block_.matmulTilingData_->matmulTiling.Ka - (kIndex + 1) * k0 :
                                            k0;
                    block_.UpdateBlockParamsMk(innerMIndex, kIndex);
                    if (kIndex != 0) {
                        SetAtomicAdd<float>();
                    } else {
                        SetAtomicNone();
                    }
                    for (uint64_t innerNIndex = 0; innerNIndex < block_.params_.innerLoopN; ++innerNIndex) {
                        block_.template CalcGMOffset<A_TYPE, B_TYPE, L0C_TYPE, BIAS_TYPE>(innerMIndex, kIndex,
                                                                                          innerNIndex, false);
                        nmloop_ = MMV3DivCeil(block_.params_.innerSingleCoreM, M0);
                        nkloop_ = MMV3DivCeil(block_.params_.kCoreUse, k0);
                        nnloop_ = MMV3DivCeil(block_.params_.innerSingleCoreN, N0);
                        if (nmloop_ - 1 == M_2) {
                            CopyInAl1ND(M_2, realK, K_INDEX_0, k0);
                        }
                        if (nnloop_ > 1) {
                            int32_t nPongReal = N_INDEX_0 + 1 == nnloop_ - 1 ?
                                                    block_.params_.innerSingleCoreN - (N_INDEX_0 + 1) * N0 :
                                                    N0;
                            CopyInBl1ND(!bpingflag, realK, K_INDEX_0, nPongReal, N_INDEX_0 + 1);
                        }
                        ProcessBaseMNK(lastK, realK, nextRealK, kIndex);
                    }
                }
                // c侧做完才能做v侧
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
                NotifyEvent<PIPE_FIX>(5);
#endif
                PipeBarrier<PIPE_ALL>();
            }
        }
        block_.UpdateBlockIndex();
    }
    if ASCEND_IS_AIC {
        WaitEventFlag();
    }
    PipeBarrier<PIPE_ALL>();
    SetAtomicNone();
    if ASCEND_IS_AIC {
        mm_.SetHF32(false, 0);
        FreeEvent();
    }
    return;
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
          const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatMulBaseKernelSingleCoreSplitKGmToL1<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE, BLOCK_TYPE,
                                          MM_CFG>::VectorProcess(uint64_t innerMIndex, GM_ADDR cGM, GM_ADDR srcAddr,
                                                                 TBuf<TPosition::VECCALC> &ubBuf)
{
    // Cast f322f16
    WaitFlagDevLocal(5);
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

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
          const MatmulConfig &MM_CFG>
__aicore__ inline void Mc2MatMulBaseKernelSingleCoreSplitKGmToL1<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE,
                                                                 BLOCK_TYPE, MM_CFG>::SetEventFlag()
{
    SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(eventIdBPingMte1Mte2));
    SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(eventIdAPingMte1Mte2));
    if (block_.matmulTilingData_->matmulTiling.stepM != 1) {
        SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(eventIdAPongMte1Mte2));
    }
    SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(eventIdBPongMte1Mte2));
    if (block_.matmulTilingData_->matmulTiling.stepM == 3) {
        SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(eventIdAPengMte1Mte2));
        k0 = 384;
    }
    SetFlag<HardEvent::FIX_M>(static_cast<event_t>(eventIdAPingMte2Mte1));
    SetFlag<HardEvent::FIX_M>(static_cast<event_t>(eventIdAPongMte2Mte1));
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
          const MatmulConfig &MM_CFG>
__aicore__ inline void Mc2MatMulBaseKernelSingleCoreSplitKGmToL1<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE,
                                                                 BLOCK_TYPE, MM_CFG>::WaitEventFlag()
{
    WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(eventIdBPingMte1Mte2));
    WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(eventIdAPingMte1Mte2));
    if (block_.matmulTilingData_->matmulTiling.stepM != 1) {
        WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(eventIdAPongMte1Mte2));
    }
    WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(eventIdBPongMte1Mte2));
    if (block_.matmulTilingData_->matmulTiling.stepM == 3) {
        WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(eventIdAPengMte1Mte2));
    }
    WaitFlag<HardEvent::FIX_M>(static_cast<event_t>(eventIdAPingMte2Mte1));
    WaitFlag<HardEvent::FIX_M>(static_cast<event_t>(eventIdAPongMte2Mte1));
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
          const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatMulBaseKernelSingleCoreSplitKGmToL1<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE, BLOCK_TYPE,
                                          MM_CFG>::CopyInAl1(uint32_t mloop, int32_t realK, int32_t kLoop, uint32_t k0)
{
    if constexpr (A_TYPE::format == CubeFormat::ND) {
        CopyInAl1ND(mloop, realK, kLoop, k0);
    } else {
        CopyInAl1NZ(mloop, realK, kLoop, k0);
    }
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
          const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatMulBaseKernelSingleCoreSplitKGmToL1<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE, BLOCK_TYPE,
                                          MM_CFG>::CopyInAl1ND(uint32_t mloop, int32_t realK, int32_t kLoop,
                                                               uint32_t k0)
{
    if ASCEND_IS_AIV {
        return;
    }
    // [m, k]  [k ,m]
    int32_t realM;
    uint64_t aOffset;
    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;
    realM = mloop == nmloop_ - 1 ? block_.params_.innerSingleCoreM - mloop * M0 : M0;
    if (block_.params_.isTransposeA) {
        aOffset = block_.offset_.offsetA + mloop * M0 + kLoop * k0 * block_.matmulTilingData_->matmulTiling.M;
        nd2nzParams.srcDValue = block_.matmulTilingData_->matmulTiling.M;
        nd2nzParams.dstNzC0Stride = MMV3CeilAlign(realK, ALIGNED_H);
        nd2nzParams.nValue = realK;
        nd2nzParams.dValue = realM;
    } else {
        aOffset = block_.offset_.offsetA + kLoop * k0 + mloop * M0 * block_.matmulTilingData_->matmulTiling.Ka;
        nd2nzParams.srcDValue = block_.matmulTilingData_->matmulTiling.Ka;
        nd2nzParams.dstNzC0Stride = MMV3CeilAlign(realM, ALIGNED_H);
        nd2nzParams.nValue = realM;
        nd2nzParams.dValue = realK;
    }
    if (mloop == 0) {
        pingMReal = realM;
        WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(eventIdAPingMte1Mte2));
        DataCopy(aL1Ping, aGlobal_[aOffset], nd2nzParams);
        SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(eventIdAPingMte2Mte1));
    } else if (mloop == 1) {
        pongMReal = realM;
        WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(eventIdAPongMte1Mte2));
        DataCopy(aL1Pong, aGlobal_[aOffset], nd2nzParams);
        SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(eventIdAPongMte2Mte1));
    } else if (mloop == 2) {
        pengMReal = realM;
        WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(eventIdAPengMte1Mte2));
        DataCopy(aL1Peng, aGlobal_[aOffset], nd2nzParams);
        SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(eventIdAPengMte2Mte1));
    }

    return;
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
          const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatMulBaseKernelSingleCoreSplitKGmToL1<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE, BLOCK_TYPE,
                                          MM_CFG>::CopyInAl1NZ(uint32_t mloop, int32_t realK, int32_t kLoop,
                                                               uint32_t k0)
{
    if ASCEND_IS_AIV {
        return;
    }
    // [m, k]  [k ,m]
    int32_t realM;
    int32_t alignM;
    uint64_t aOffset;
    DataCopyParams aNzParams;
    realM = mloop == nmloop_ - 1 ? block_.params_.innerSingleCoreM - mloop * M0 : M0;
    alignM = MMV3CeilAlign(realM, ALIGNED_H);
    if (block_.params_.isTransposeA) {
        aOffset = block_.offset_.offsetA + mloop * M0 * block_.params_.alignedKaSize +
                  kLoop * k0 * block_.params_.c0Size; // k, m
    } else {
        aOffset = block_.offset_.offsetA + kLoop * k0 * block_.params_.alignedOriM +
                  mloop * M0 * block_.params_.c0Size; // m, k
    }
    aNzParams.blockCount =
        block_.params_.isTransposeA ? MMV3DivCeil(realM, ALIGNED_H) : MMV3DivCeil(realK, block_.params_.c0Size);
    aNzParams.blockLen = block_.params_.isTransposeA ? realK : alignM;
    aNzParams.srcStride =
        block_.params_.isTransposeA ? block_.params_.alignedKaSize - realK : block_.params_.alignedOriM - alignM;
    if (mloop == 0) {
        pingMReal = block_.params_.isTransposeA ? realM : alignM;
        WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(eventIdAPingMte1Mte2));
        DataCopy(aL1Ping, aGlobal_[aOffset], aNzParams);
        SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(eventIdAPingMte2Mte1));
    } else if (mloop == 1) {
        pongMReal = block_.params_.isTransposeA ? realM : alignM;
        WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(eventIdAPongMte1Mte2));
        DataCopy(aL1Pong, aGlobal_[aOffset], aNzParams);
        SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(eventIdAPongMte2Mte1));
    } else if (mloop == 2) {
        pengMReal = block_.params_.isTransposeA ? realM : alignM;
        WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(eventIdAPengMte1Mte2));
        DataCopy(aL1Peng, aGlobal_[aOffset], aNzParams);
        SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(eventIdAPengMte2Mte1));
    }
    return;
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
          const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatMulBaseKernelSingleCoreSplitKGmToL1<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE, BLOCK_TYPE,
                                          MM_CFG>::CopyInBl1(bool bpingflag, int32_t realK, int32_t kLoop,
                                                             int32_t realN, int32_t nLoop)
{
    if constexpr (B_TYPE::format == CubeFormat::ND) {
        CopyInBl1ND(bpingflag, realK, kLoop, realN, nLoop);
    } else {
        CopyInBl1NZ(bpingflag, realK, kLoop, realN, nLoop);
    }
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
          const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatMulBaseKernelSingleCoreSplitKGmToL1<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE, BLOCK_TYPE,
                                          MM_CFG>::CopyInBl1ND(bool bpingflag, int32_t realK, int32_t kLoop,
                                                               int32_t realN, int32_t nLoop)
{
    if ASCEND_IS_AIV {
        return;
    }
    uint32_t eidMte1Mte2 = bpingflag ? eventIdBPingMte1Mte2 : eventIdBPongMte1Mte2;
    uint32_t eidMte2Mte1 = bpingflag ? eventIdBPingMte2Mte1 : eventIdBPongMte2Mte1;
    bL1 = bpingflag ? bL1Ping : bL1Pong;
    // [k, n] [n, k]
    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;
    nd2nzParams.dstNzC0Stride = MMV3CeilAlign(block_.params_.isTransposeB ? realN : realK, block_.params_.c0Size);
    uint64_t bOffset = block_.offset_.offsetB;
    if (block_.params_.isTransposeB) {
        bOffset += kLoop * k0 + nLoop * N0 * block_.matmulTilingData_->matmulTiling.Kb; // n, k
        nd2nzParams.srcDValue = block_.matmulTilingData_->matmulTiling.Kb;
        nd2nzParams.dstNzC0Stride = MMV3CeilAlign(realN, block_.params_.c0Size);
        nd2nzParams.nValue = realN;
        nd2nzParams.dValue = realK;
    } else {
        bOffset += nLoop * N0 + kLoop * k0 * block_.matmulTilingData_->matmulTiling.N; // k, n
        nd2nzParams.srcDValue = block_.matmulTilingData_->matmulTiling.N;
        nd2nzParams.dstNzC0Stride = MMV3CeilAlign(realK, block_.params_.c0Size);
        nd2nzParams.nValue = realK;
        nd2nzParams.dValue = realN;
    }
    WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(eidMte1Mte2));
    DataCopy(bL1, bGlobal_[bOffset], nd2nzParams);
    SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(eidMte2Mte1));
    return;
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
          const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatMulBaseKernelSingleCoreSplitKGmToL1<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE, BLOCK_TYPE,
                                          MM_CFG>::CopyInBl1NZ(bool bpingflag, int32_t realK, int32_t kLoop,
                                                               int32_t realN, int32_t nLoop)
{
    if ASCEND_IS_AIV {
        return;
    }
    uint32_t eidMte1Mte2 = bpingflag ? eventIdBPingMte1Mte2 : eventIdBPongMte1Mte2;
    uint32_t eidMte2Mte1 = bpingflag ? eventIdBPingMte2Mte1 : eventIdBPongMte2Mte1;
    bL1 = bpingflag ? bL1Ping : bL1Pong;
    // [k, n] [n, k]
    DataCopyParams bNzParams;
    uint64_t bOffset = block_.offset_.offsetB;
    if (block_.params_.isTransposeB) {
        bOffset += kLoop * k0 * block_.params_.alignedOriN + nLoop * N0 * block_.params_.c0Size; // n, k
        bNzParams.blockCount = MMV3DivCeil(realK, block_.params_.c0Size);
        bNzParams.blockLen = realN;
        bNzParams.srcStride = block_.params_.alignedOriN - realN;
    } else {
        bOffset += nLoop * N0 * block_.params_.alignedKbSize + kLoop * k0 * block_.params_.c0Size; // k, n
        bNzParams.blockCount = MMV3DivCeil(realN, ALIGNED_H);
        bNzParams.blockLen = realK;
        bNzParams.srcStride = block_.params_.alignedKbSize - realK;
    }
    WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(eidMte1Mte2));
    DataCopy(bL1, bGlobal_[bOffset], bNzParams);
    SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(eidMte2Mte1));
    return;
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
          const MatmulConfig &MM_CFG>
__aicore__ inline void Mc2MatMulBaseKernelSingleCoreSplitKGmToL1<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE,
                                                                 BLOCK_TYPE, MM_CFG>::GetEvent()
{
    eventIdBPingMte1Mte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>());
    eventIdBPongMte1Mte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>());
    eventIdBPingMte2Mte1 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>());
    eventIdBPongMte2Mte1 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>());
    eventIdAPingMte1Mte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>());
    eventIdAPongMte1Mte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>());
    eventIdAPingMte2Mte1 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>());
    eventIdAPongMte2Mte1 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>());
    if (block_.matmulTilingData_->matmulTiling.stepM == 3) {
        eventIdAPengMte1Mte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>());
        eventIdAPengMte2Mte1 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>());
    }
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
          const MatmulConfig &MM_CFG>
__aicore__ inline void Mc2MatMulBaseKernelSingleCoreSplitKGmToL1<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE,
                                                                 BLOCK_TYPE, MM_CFG>::FreeEvent()
{
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eventIdBPingMte1Mte2);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eventIdBPongMte1Mte2);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eventIdAPingMte1Mte2);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eventIdAPongMte1Mte2);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE1>(eventIdBPingMte2Mte1);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE1>(eventIdBPongMte2Mte1);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE1>(eventIdAPingMte2Mte1);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE1>(eventIdAPongMte2Mte1);
    if (block_.matmulTilingData_->matmulTiling.stepM == 3) {
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eventIdAPengMte1Mte2);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE1>(eventIdAPengMte2Mte1);
    }
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
          const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatMulBaseKernelSingleCoreSplitKGmToL1<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE, BLOCK_TYPE,
                                          MM_CFG>::ProcessBaseMNK(bool lastK, int32_t realK, int32_t nextRealK,
                                                                  uint64_t kIndex)
{
    for (int32_t nloop = 0; nloop < nnloop_; nloop++) {
        bool lastN = nloop == nnloop_ - 1;
        bool firstN = nloop == 0;
        int32_t realN = lastN ? block_.params_.innerSingleCoreN - nloop * N0 : N0;
        if (!lastN && !firstN) {
            int32_t nextRealN = nloop == nnloop_ - 2 ? block_.params_.innerSingleCoreN - (nloop + 1) * N0 : N0;
            CopyInBl1ND(!bpingflag, realK, K_INDEX_0, nextRealN, nloop + 1);
        }
        if (lastN && !lastK) {
            CopyInBl1ND(!bpingflag, nextRealK, K_INDEX_0 + 1, N0, N_INDEX_0); // B nextk
        }
        if (nmloop_ - 1 >= M_0) {
            AMatMulB(bpingflag, nloop, realK, realN, kIndex, aL1Ping, eventIdAPingMte2Mte1, M_0);
            if (lastN) {
                SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(eventIdAPingMte1Mte2));
                if (!lastK) {
                    CopyInAl1ND(M_0, nextRealK, K_INDEX_0 + 1, k0);
                }
            }
        }
        if (nmloop_ - 1 >= M_1) {
            AMatMulB(bpingflag, nloop, realK, realN, kIndex, aL1Pong, eventIdAPongMte2Mte1, M_1);
            if (lastN) {
                SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(eventIdAPongMte1Mte2));
                if (!lastK) {
                    CopyInAl1ND(M_1, nextRealK, K_INDEX_0 + 1, k0); // A1 nextk
                }
            }
        }
        if (nmloop_ - 1 == M_2) {
            AMatMulB(bpingflag, nloop, realK, realN, kIndex, aL1Peng, eventIdAPengMte2Mte1, M_2);
            if (lastN) {
                SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(eventIdAPengMte1Mte2));
            }
        }
        uint32_t eid = bpingflag ? eventIdBPingMte1Mte2 : eventIdBPongMte1Mte2;
        SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(eid));
        bpingflag = !bpingflag;
    }
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
          const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatMulBaseKernelSingleCoreSplitKGmToL1<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE, BLOCK_TYPE,
                                          MM_CFG>::AMatMulB(bool bpingflag, int32_t nloop, int32_t k_real,
                                                            int32_t n_real, uint64_t kIndex, LocalTensor<A_T> aL1,
                                                            event_t eventAMte1WaitMte2, uint32_t mloop)
{
    FixpipeParamsV220 params;
    params.ndNum = 1;
    params.nSize = n_real;
    params.dstStride = block_.params_.outNAlign;
    params.mSize = pingMReal;
    if (A_TYPE::format == CubeFormat::ND && B_TYPE::format == CubeFormat::ND &&
        (block_.matmulTilingData_->matmulTiling.N % ALIGN_128_BYTE == 0)) {
        params.dstStride = block_.matmulTilingData_->matmulTiling.N;
    }
    uint32_t eidMte2Mte1 = bpingflag ? eventIdBPingMte2Mte1 : eventIdBPongMte2Mte1;
    bL1 = bpingflag ? bL1Ping : bL1Pong;
    l0c_ = l0cpingflag ? L0cPing_ : L0cPong_;
    l0cFixMAndMFIX = l0cpingflag ? eventIdAPingMte2Mte1 : eventIdAPongMte2Mte1;
    mm_.SetSingleShape(pingMReal, n_real, k_real);
    if (mloop == M_1) {
        params.mSize = pongMReal;
        mm_.SetSingleShape(pongMReal, n_real, k_real);
    } else if (mloop == M_2) {
        params.mSize = pengMReal;
        mm_.SetSingleShape(pengMReal, n_real, k_real);
    }
    params.srcStride = (params.mSize + BLOCK_CUBE - 1) / BLOCK_CUBE * BLOCK_CUBE;
    mm_.SetTensorA(aL1, block_.params_.isTransposeA);
    mm_.SetTensorB(bL1, block_.params_.isTransposeB);
    if (block_.matmulTilingData_->matmulTiling.isBias && kIndex == 0) {
        mm_.SetBias(biasGlobal_[block_.offset_.offsetBias + nloop * N0]);
    }
    if (nloop == 0) {
        WaitFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(eventAMte1WaitMte2));
    }
    if (mloop == 0) {
        WaitFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(eidMte2Mte1));
    }
    WaitFlag<HardEvent::FIX_M>(static_cast<event_t>(l0cFixMAndMFIX));
    mm_.Iterate(false, l0c_);
    SetFlag<HardEvent::M_FIX>(static_cast<event_t>(l0cFixMAndMFIX));
    WaitFlag<HardEvent::M_FIX>(static_cast<event_t>(l0cFixMAndMFIX));
    Fixpipe(cGlobal_[block_.offset_.offsetC + mloop * params.dstStride * M0 + nloop * N0], l0c_, params);
    SetFlag<HardEvent::FIX_M>(static_cast<event_t>(l0cFixMAndMFIX));
    mm_.ClearBias();
    mm_.End();
    l0cpingflag = !l0cpingflag;
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
          const MatmulConfig &MM_CFG>
__aicore__ inline void Mc2MatMulBaseKernelSingleCoreSplitKGmToL1<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE,
                                                                 BLOCK_TYPE, MM_CFG>::UnAlignedProcess()
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
    SetEventFlag();
    for (uint64_t j = 0; j < block_.params_.realRound; ++j) {
        block_.UpdateBlockCnt();
        for (uint64_t innerMIndex = 0; innerMIndex < block_.params_.innerLoopM; ++innerMIndex) {
            block_.UpdateBlockParamsMk(innerMIndex, K_INDEX_0);
            block_.template CalcGMOffset<A_TYPE, B_TYPE, L0C_TYPE, BIAS_TYPE>(innerMIndex, K_INDEX_0, N_INDEX_0, false);
            nmloop_ = MMV3DivCeil(block_.params_.innerSingleCoreM, M0);
            nnloop_ = MMV3DivCeil(block_.params_.innerSingleCoreN, N0);
            CopyInAl1(M_0, k0, K_INDEX_0, k0);
            if (nmloop_ - 1 >= M_1) {
                CopyInAl1(M_1, k0, K_INDEX_0, k0);
            }
            int32_t nPingReal =
                N_INDEX_0 == nnloop_ - 1 ? MMV3CeilAlign(block_.params_.innerSingleCoreN, ALIGNED_H) : N0;
            CopyInBl1(bpingflag, k0, K_INDEX_0, nPingReal, N_INDEX_0);
            for (int kIndex = 0; kIndex < block_.params_.loopK; ++kIndex) {
                bool lastK = kIndex == block_.params_.loopK - 1;
                int32_t realK = lastK ? MMV3CeilAlign(block_.matmulTilingData_->matmulTiling.Ka - kIndex * k0,
                                                      block_.params_.c0Size) :
                                        k0;
                int32_t realKb = realK;
                if (B_TYPE::format == CubeFormat::ND && lastK) {
                    realKb = block_.matmulTilingData_->matmulTiling.Ka - kIndex * k0;
                }
                int32_t nextRealK = kIndex == block_.params_.loopK - 2 ?
                                        MMV3CeilAlign(block_.matmulTilingData_->matmulTiling.Ka - (kIndex + 1) * k0,
                                                      block_.params_.c0Size) :
                                        k0;
                block_.UpdateBlockParamsMk(innerMIndex, kIndex);
                if (kIndex != 0) {
                    SetAtomicAdd<float>();
                } else {
                    SetAtomicNone();
                }
                for (uint64_t innerNIndex = 0; innerNIndex < block_.params_.innerLoopN; ++innerNIndex) {
                    block_.template CalcGMOffset<A_TYPE, B_TYPE, L0C_TYPE, BIAS_TYPE>(innerMIndex, kIndex, innerNIndex,
                                                                                      false);
                    nmloop_ = MMV3DivCeil(block_.params_.innerSingleCoreM, M0);
                    nkloop_ = MMV3DivCeil(block_.params_.kCoreUse, k0);
                    nnloop_ = MMV3DivCeil(block_.params_.innerSingleCoreN, N0);
                    if (nmloop_ - 1 == M_2) {
                        CopyInAl1(M_2, realK, K_INDEX_0, k0);
                    }
                    if (nnloop_ > 1) {
                        int32_t nPongReal =
                            N_INDEX_0 + 1 == nnloop_ - 1 ?
                                MMV3CeilAlign(block_.params_.innerSingleCoreN - (N_INDEX_0 + 1) * N0, ALIGNED_H) :
                                N0;
                        CopyInBl1(!bpingflag, realKb, K_INDEX_0, nPongReal, N_INDEX_0 + 1);
                    }
                    UnAlignedProcessBaseMNK(lastK, realK, realKb, nextRealK, kIndex);
                }
            }
        }
        block_.UpdateBlockIndex();
    }
    WaitEventFlag();
    PipeBarrier<PIPE_ALL>();
    SetAtomicNone();
    mm_.SetHF32(false, 0);
    FreeEvent();
    return;
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class BLOCK_TYPE,
          const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatMulBaseKernelSingleCoreSplitKGmToL1<A_TYPE, B_TYPE, L0C_TYPE, OUTPUT_TYPE, BIAS_TYPE, BLOCK_TYPE,
                                          MM_CFG>::UnAlignedProcessBaseMNK(bool lastK, int32_t realK, int32_t realKb,
                                                                           int32_t nextRealK, uint64_t kIndex)
{
    int32_t nextRealKb = nextRealK;
    if (B_TYPE::format == CubeFormat::ND && kIndex == block_.params_.loopK - 2) {
        nextRealKb = block_.matmulTilingData_->matmulTiling.Ka - (kIndex + 1) * k0;
    }
    for (int32_t nloop = 0; nloop < nnloop_; nloop++) {
        bool lastN = nloop == nnloop_ - 1;
        bool firstN = nloop == 0;
        int32_t realN = lastN ? MMV3CeilAlign(block_.params_.innerSingleCoreN - nloop * N0, ALIGNED_H) : N0;
        if (!lastN && !firstN) {
            int32_t nextRealN = nloop == nnloop_ - 2 ?
                                    MMV3CeilAlign(block_.params_.innerSingleCoreN - (nloop + 1) * N0, ALIGNED_H) :
                                    N0;
            CopyInBl1(!bpingflag, realKb, K_INDEX_0, nextRealN, nloop + 1);
        }
        if (lastN && !lastK) {
            CopyInBl1(!bpingflag, nextRealKb, K_INDEX_0 + 1, N0, N_INDEX_0);
        }
        if (nmloop_ - 1 >= M_0) {
            AMatMulB(bpingflag, nloop, realK, realN, kIndex, aL1Ping, eventIdAPingMte2Mte1, M_0);
            if (lastN) {
                SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(eventIdAPingMte1Mte2));
                if (!lastK) {
                    CopyInAl1(M_0, nextRealK, K_INDEX_0 + 1, k0);
                }
            }
        }
        if (nmloop_ - 1 >= M_1) {
            AMatMulB(bpingflag, nloop, realK, realN, kIndex, aL1Pong, eventIdAPongMte2Mte1, M_1);
            if (lastN) {
                SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(eventIdAPongMte1Mte2));
                if (!lastK) {
                    CopyInAl1(M_1, nextRealK, K_INDEX_0 + 1, k0);
                }
            }
        }
        if (nmloop_ - 1 == M_2) {
            AMatMulB(bpingflag, nloop, realK, realN, kIndex, aL1Peng, eventIdAPengMte2Mte1, M_2);
            if (lastN) {
                SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(eventIdAPengMte1Mte2));
            }
        }
        uint32_t eid = bpingflag ? eventIdBPingMte1Mte2 : eventIdBPongMte1Mte2;
        SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(eid));
        bpingflag = !bpingflag;
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE,
          class BLOCK_TYPE = Mc2MatmulSingleCoreSplitKBaseBlock, const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD>
class MatMulSingleCoreSplitKKernelGmToL1 {
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
    __aicore__ inline MatMulSingleCoreSplitKKernelGmToL1()
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
    Mc2MatMulBaseKernelSingleCoreSplitKGmToL1<A_TYPE, B_TYPE, cType, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>
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

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MatMulSingleCoreSplitKKernelGmToL1<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Init(
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

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
MatMulSingleCoreSplitKKernelGmToL1<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::InitInputs(
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

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
MatMulSingleCoreSplitKKernelGmToL1<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::UpdateGlobalTensor(
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

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
MatMulSingleCoreSplitKKernelGmToL1<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::ProcessRemovePaddingImpl()
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

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
MatMulSingleCoreSplitKKernelGmToL1<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Process()
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

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
MatMulSingleCoreSplitKKernelGmToL1<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::NNot128AlignProcess()
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
