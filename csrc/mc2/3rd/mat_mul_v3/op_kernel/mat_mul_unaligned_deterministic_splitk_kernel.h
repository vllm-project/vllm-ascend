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
 * \file mat_mul_unaligned_deterministic_splitk_kernel.h
 * \brief
 */
#ifndef __OP_KERNEL_MATMUL_V3_UNALIGNED_DETERMINISTIC_SPLITK_KERNEL_H__
#define __OP_KERNEL_MATMUL_V3_UNALIGNED_DETERMINISTIC_SPLITK_KERNEL_H__

#include "mat_mul_nd2nz.h"
#include "mat_mul_deterministic_splitk_kernel.h"

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE,
          FIXPIPE_OPT_SELECT FIXPIPE_OPT = FIXPIPE_OPT_SELECT::BASE>
__aicore__ inline void
Mc2MatMulUnAlignedKernelDeterministicSplitK(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM,
                                            const Mc2MatmulV3TilingData &matmulTilingData, GM_ADDR workspaceGM)
{
    const TCubeTiling &tiling = matmulTilingData.matmulTiling;
    using A_T = typename A_TYPE::T;
    uint64_t c0Size = 8; // initial c0size as fp32's c0size
    GetSizeC0<A_T>(c0Size);
    uint64_t alignedOriM = MMV3CeilAlign(tiling.M, ALIGNED_H);
    uint64_t alignedOriN = MMV3CeilAlign(tiling.N, c0Size);
    uint64_t alignedKaSize = MMV3CeilAlign(tiling.Ka, c0Size);
    uint64_t alignedKbSize = MMV3CeilAlign(tiling.Kb, ALIGNED_H);
    uint64_t originM = tiling.M;

    TPipe que;
    bool orderNMFlag = tiling.iterateOrder;
    bool orderFlag = !tiling.iterateOrder;
    bool isL2cacheSplit = orderFlag ? tiling.M != tiling.singleCoreM : tiling.N != tiling.singleCoreN;
    uint64_t singleSize = 0;
    uint64_t coreSize = 0;
    uint64_t mCnt = 0;
    uint64_t nCnt = 0;
    uint64_t cnt = 0;

    uint64_t alignedSingleCoreM = MMV3CeilAlign(tiling.singleCoreM, 16); // 384
    uint64_t alignedM = MMV3CeilAlign(tiling.M, 16);
    uint64_t alignedN = MMV3CeilAlign(tiling.N, 16);
    alignedM =
        alignedM > static_cast<uint64_t>(tiling.singleCoreM) ? alignedM : static_cast<uint64_t>(tiling.singleCoreM);
    alignedN =
        alignedN > static_cast<uint64_t>(tiling.singleCoreN) ? alignedN : static_cast<uint64_t>(tiling.singleCoreN);

    mCnt = MMV3DivCeil(tiling.M, tiling.singleCoreM);
    nCnt = MMV3DivCeil(tiling.N, tiling.singleCoreN);
    singleSize = alignedM * alignedN;
    if (isL2cacheSplit) {
        if (FIXPIPE_OPT == FIXPIPE_OPT_SELECT::BASE) {
            singleSize = static_cast<uint64_t>(tiling.singleCoreM) * static_cast<uint64_t>(tiling.singleCoreN);
        }
        coreSize = MMV3DivCeil(tiling.singleCoreM, static_cast<uint64_t>(tiling.usedCoreNum) * NUM_TWO) *
                   tiling.singleCoreN; // 无论MK还是NK都按照M方向进行分AIV核
    } else {                           // 不切L2cache
        if (orderFlag) {
            if (FIXPIPE_OPT == FIXPIPE_OPT_SELECT::BASE) {
                singleSize = static_cast<uint64_t>(tiling.singleCoreN) * static_cast<uint64_t>(tiling.M);
            }
            coreSize = MMV3DivCeil(tiling.M, static_cast<uint64_t>(tiling.usedCoreNum) * NUM_TWO) * tiling.singleCoreN;
            cnt = nCnt;
        } else {
            if (FIXPIPE_OPT == FIXPIPE_OPT_SELECT::BASE) {
                singleSize = static_cast<uint64_t>(tiling.singleCoreM) * static_cast<uint64_t>(tiling.N);
            }
            coreSize = MMV3DivCeil(singleSize, static_cast<uint64_t>(tiling.usedCoreNum) * NUM_TWO);
            cnt = mCnt;
        }
    }

    GM_ADDR mmGM = workspaceGM;
    GM_ADDR mmOffsetGM = reinterpret_cast<GM_ADDR>(mmGM + GetBlockIdx() * singleSize * NUM_TWO * sizeof(float));

    if (matmulTilingData.matmulRunInfo.transA) {
        alignedOriM = MMV3CeilAlign(tiling.M, c0Size);
        alignedKaSize = MMV3CeilAlign(tiling.Ka, ALIGNED_H);
    }
    if (matmulTilingData.matmulRunInfo.transB) {
        alignedOriN = MMV3CeilAlign(tiling.N, ALIGNED_H);
        alignedKbSize = MMV3CeilAlign(tiling.Kb, c0Size);
    }
    GM_ADDR alignedworkspaceGM =
        reinterpret_cast<GM_ADDR>(mmGM + tiling.usedCoreNum * singleSize * NUM_TWO * sizeof(float)); // NUM_TWO for DB
    if ASCEND_IS_AIV {
        if (GetBlockIdx() >= (tiling.usedCoreNum * NUM_TWO)) {
            NotifyEvent<PIPE_MTE3>(ND2NZ_AIV_SYNC_AIC_FLAG);
            PipeBarrier<PIPE_ALL>();
            return;
        }
        uint64_t totalSize = singleSize * static_cast<uint64_t>(tiling.usedCoreNum);
        uint64_t outSize = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.N);
        TBuf<TPosition::VECCALC> tmpBuf;
        que.InitBuffer(tmpBuf, TOTAL_UB_SIZE);
        PipeBarrier<PIPE_ALL>();
        // ND2NZ
        GM_ADDR workspaceGMInUsing = alignedworkspaceGM;
        if (matmulTilingData.matmulRunInfo.nd2nzA) {
            MatrixAtoNZV2<typename A_TYPE::T>(workspaceGMInUsing, aGM, tiling, matmulTilingData.matmulRunInfo.transA,
                                              tmpBuf, matmulTilingData.baseAN, matmulTilingData.baseAD);
            workspaceGMInUsing =
                reinterpret_cast<GM_ADDR>(workspaceGMInUsing + alignedOriM * alignedKaSize * sizeof(A_T));
            originM = alignedOriM;
        }
        event_t event_mte3_mte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(event_mte3_mte2);
        WaitFlag<HardEvent::MTE3_MTE2>(event_mte3_mte2);
        if (matmulTilingData.matmulRunInfo.nd2nzB) {
            MatrixBtoNZV2<typename B_TYPE::T>(workspaceGMInUsing, bGM, tiling, matmulTilingData.matmulRunInfo.transB,
                                              tmpBuf, matmulTilingData.baseBN, matmulTilingData.baseBD);
        }

        SyncAll();
        NotifyEvent<PIPE_MTE3>(ND2NZ_AIV_SYNC_AIC_FLAG);
        PipeBarrier<PIPE_ALL>();
        if (isL2cacheSplit) {
            if constexpr (FIXPIPE_OPT == FIXPIPE_OPT_SELECT::VEC_NZ2ND_UNALIGNOUT) {
                ReduceKInUbNzL2cache<C_TYPE>(cGM, mmGM, coreSize, singleSize, totalSize, outSize, mCnt, nCnt,
                                             tiling.singleCoreN, tiling.N, tmpBuf, orderNMFlag, tiling, originM);
            } else {
                ReduceKInUbL2cache<C_TYPE>(cGM, mmGM, coreSize, singleSize, totalSize, outSize, mCnt, nCnt,
                                           tiling.singleCoreN, tiling.N, tmpBuf, orderNMFlag, tiling);
            }
        } else {
            if constexpr (FIXPIPE_OPT == FIXPIPE_OPT_SELECT::VEC_NZ2ND_UNALIGNOUT) {
                ReduceKNzInUb<C_TYPE>(cGM, mmGM, coreSize, singleSize, totalSize, outSize, cnt, tiling.singleCoreN,
                                      tiling.N, tmpBuf, orderFlag, tiling, mCnt, nCnt, originM);
            } else {
                ReduceKInUb<C_TYPE>(cGM, mmGM, coreSize, singleSize, totalSize, outSize, cnt, tiling.singleCoreN,
                                    tiling.N, tmpBuf, orderFlag, tiling);
            }
        }
        PipeBarrier<PIPE_ALL>();
        return;
    }

    if ASCEND_IS_AIC {
        WaitEvent(ND2NZ_AIV_SYNC_AIC_FLAG);
        if (GetBlockIdx() >= tiling.usedCoreNum) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
            NotifyEvent<PIPE_FIX>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX);
            NotifyEvent<PIPE_FIX>(AIC_SYNC_AIV_FLAG);
#elif defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
            NotifyEvent<PIPE_FIX>(AIC_SYNC_AIV_FLAG);
#endif
            PipeBarrier<PIPE_ALL>();
            return;
        }
        using cType = MatmulType<C_TYPE::pos, C_TYPE::format, float, C_TYPE::isTrans>;
        if (isL2cacheSplit) {
            if (matmulTilingData.matmulRunInfo.nd2nzA && !matmulTilingData.matmulRunInfo.nd2nzB) {
                using aType = MatmulType<A_TYPE::pos, CubeFormat::NZ, typename A_TYPE::T, A_TYPE::isTrans>;
                if (matmulTilingData.matmulRunInfo.isNzB) {
                    using bType = MatmulType<B_TYPE::pos, CubeFormat::NZ, typename B_TYPE::T, B_TYPE::isTrans>;
                    Mc2MatMulMultiCoreSplitKDivideL2cache<aType, bType, cType, BIAS_TYPE>(
                        alignedworkspaceGM, bGM, biasGM, mmOffsetGM, singleSize, matmulTilingData.matmulRunInfo.isHf32,
                        &que, tiling, tiling.isBias);
                } else {
                    Mc2MatMulMultiCoreSplitKDivideL2cache<aType, B_TYPE, cType, BIAS_TYPE>(
                        alignedworkspaceGM, bGM, biasGM, mmOffsetGM, singleSize, matmulTilingData.matmulRunInfo.isHf32,
                        &que, tiling, tiling.isBias);
                }
            } else if (!matmulTilingData.matmulRunInfo.nd2nzA && matmulTilingData.matmulRunInfo.nd2nzB) {
                using bType = MatmulType<B_TYPE::pos, CubeFormat::NZ, typename B_TYPE::T, B_TYPE::isTrans>;
                if (matmulTilingData.matmulRunInfo.isNzA) {
                    using aType = MatmulType<A_TYPE::pos, CubeFormat::NZ, typename A_TYPE::T, A_TYPE::isTrans>;
                    Mc2MatMulMultiCoreSplitKDivideL2cache<aType, bType, cType, BIAS_TYPE>(
                        aGM, alignedworkspaceGM, biasGM, mmOffsetGM, singleSize, matmulTilingData.matmulRunInfo.isHf32,
                        &que, tiling, tiling.isBias);
                } else {
                    Mc2MatMulMultiCoreSplitKDivideL2cache<A_TYPE, bType, cType, BIAS_TYPE>(
                        aGM, alignedworkspaceGM, biasGM, mmOffsetGM, singleSize, matmulTilingData.matmulRunInfo.isHf32,
                        &que, tiling, tiling.isBias);
                }
            } else if (matmulTilingData.matmulRunInfo.nd2nzA && matmulTilingData.matmulRunInfo.nd2nzB) {
                using aType = MatmulType<A_TYPE::pos, CubeFormat::NZ, typename A_TYPE::T, A_TYPE::isTrans>;
                using bType = MatmulType<B_TYPE::pos, CubeFormat::NZ, typename B_TYPE::T, B_TYPE::isTrans>;

                Mc2MatMulMultiCoreSplitKDivideL2cache<aType, bType, cType, BIAS_TYPE>(
                    alignedworkspaceGM, alignedworkspaceGM + alignedOriM * alignedKaSize * sizeof(A_T), biasGM,
                    mmOffsetGM, singleSize, matmulTilingData.matmulRunInfo.isHf32, &que, tiling, tiling.isBias);
            }
        } else {
            if (matmulTilingData.matmulRunInfo.nd2nzA && !matmulTilingData.matmulRunInfo.nd2nzB) {
                using aType = MatmulType<A_TYPE::pos, CubeFormat::NZ, typename A_TYPE::T, A_TYPE::isTrans>;
                if (matmulTilingData.matmulRunInfo.isNzB) {
                    using bType = MatmulType<B_TYPE::pos, CubeFormat::NZ, typename B_TYPE::T, B_TYPE::isTrans>;
                    Mc2MatMulMultiCoreSplitKDivide<aType, bType, cType, BIAS_TYPE>(
                        alignedworkspaceGM, bGM, biasGM, mmOffsetGM, singleSize, matmulTilingData.matmulRunInfo.isHf32,
                        &que, tiling, tiling.isBias);
                } else {
                    Mc2MatMulMultiCoreSplitKDivide<aType, B_TYPE, cType, BIAS_TYPE>(
                        alignedworkspaceGM, bGM, biasGM, mmOffsetGM, singleSize, matmulTilingData.matmulRunInfo.isHf32,
                        &que, tiling, tiling.isBias);
                }
            } else if (!matmulTilingData.matmulRunInfo.nd2nzA && matmulTilingData.matmulRunInfo.nd2nzB) {
                using bType = MatmulType<B_TYPE::pos, CubeFormat::NZ, typename B_TYPE::T, B_TYPE::isTrans>;
                if (matmulTilingData.matmulRunInfo.isNzA) {
                    using aType = MatmulType<A_TYPE::pos, CubeFormat::NZ, typename A_TYPE::T, A_TYPE::isTrans>;
                    Mc2MatMulMultiCoreSplitKDivide<aType, bType, cType, BIAS_TYPE>(
                        aGM, alignedworkspaceGM, biasGM, mmOffsetGM, singleSize, matmulTilingData.matmulRunInfo.isHf32,
                        &que, tiling, tiling.isBias);
                } else {
                    Mc2MatMulMultiCoreSplitKDivide<A_TYPE, bType, cType, BIAS_TYPE>(
                        aGM, alignedworkspaceGM, biasGM, mmOffsetGM, singleSize, matmulTilingData.matmulRunInfo.isHf32,
                        &que, tiling, tiling.isBias);
                }
            } else if (matmulTilingData.matmulRunInfo.nd2nzA && matmulTilingData.matmulRunInfo.nd2nzB) {
                using aType = MatmulType<A_TYPE::pos, CubeFormat::NZ, typename A_TYPE::T, A_TYPE::isTrans>;
                using bType = MatmulType<B_TYPE::pos, CubeFormat::NZ, typename B_TYPE::T, B_TYPE::isTrans>;
                Mc2MatMulMultiCoreSplitKDivide<aType, bType, cType, BIAS_TYPE>(
                    alignedworkspaceGM, alignedworkspaceGM + alignedOriM * alignedKaSize * sizeof(A_T), biasGM,
                    mmOffsetGM, singleSize, matmulTilingData.matmulRunInfo.isHf32, &que, tiling, tiling.isBias);
            }
        }
        return;
    }
}
#endif // __OP_KERNEL_MATMUL_V3_H__
