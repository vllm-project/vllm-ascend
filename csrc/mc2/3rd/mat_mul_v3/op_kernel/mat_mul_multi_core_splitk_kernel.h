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
 * \file mat_mul_multi_core_splitk_kernel.h
 * \brief
 */
#ifndef __OP_KERNEL_MATMUL_V3_MULTI_CORE_SPLITK_KERNEL_H__
#define __OP_KERNEL_MATMUL_V3_MULTI_CORE_SPLITK_KERNEL_H__

#include "mat_mul_deterministic_splitk_kernel.h"

namespace Mc2MatmulV3 {

template <class C_T>
__aicore__ inline void ClearOutput(GlobalTensor<C_T> &cGlobal, const TCubeTiling &tiling, TPipe &pipe)
{
    // Init output
    uint64_t c0Size = 32 / sizeof(C_T); // 清零数据32B对齐
    uint64_t outSize = MMV3CeilAlign(tiling.M * tiling.N, c0Size);
    uint64_t clearSizePerCore = MMV3CeilAlign(MMV3DivCeil(outSize, GetBlockNum()), c0Size);
    uint64_t dstOffset = clearSizePerCore * GetBlockIdx();
    // Init L1 buffer
    TBuf<TPosition::TSCM> localBuffer;
    pipe.InitBuffer(localBuffer, clearSizePerCore * sizeof(C_T));
    LocalTensor<C_T> tmpBuf = localBuffer.Get<C_T>();
    // 确保当前核仍需清零
    if (outSize > dstOffset) {
        uint64_t realClearSize = clearSizePerCore;
        uint16_t blockLen = MMV3DivCeil(realClearSize, c0Size);
        if (outSize - dstOffset < clearSizePerCore) {
            realClearSize = outSize - dstOffset;
            blockLen = MMV3DivCeil(realClearSize, c0Size);
            dstOffset = outSize - blockLen * c0Size;
        }
        InitConstValue(tmpBuf, {1, blockLen, 0, 0U});
        DataCopyParams dataCopyParams(1, blockLen, 0, 0);
        DataCopy(cGlobal[dstOffset], tmpBuf, dataCopyParams);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void Mc2MatMulBlockMultiCoreSplitK(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM,
                                                     const TCubeTiling &tiling)
{
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    GlobalTensor<A_T> aGlobal;
    GlobalTensor<B_T> bGlobal;
    GlobalTensor<C_T> cGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(aGM), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM), tiling.Kb * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(cGM), tiling.M * tiling.N);
    TPipe pipe;
    ClearOutput<C_T>(cGlobal, tiling, pipe);

    SetAtomicNone();
    if (GetBlockIdx() >= tiling.usedCoreNum) {
        CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_AIC_FLAG);
        CrossCoreWaitFlag(SYNC_AIC_FLAG);
        return;
    }
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG_NO_PRELOAD> mm;
    mm.SetSubBlockIdx(0);
    mm.Init(&tiling, &pipe);
    mm.SetOrgShape(tiling.M, tiling.N, tiling.Ka, tiling.Kb, tiling.N);

    uint64_t mCnt = MMV3DivCeil(tiling.M, tiling.singleCoreM);
    uint64_t nCnt = MMV3DivCeil(tiling.N, tiling.singleCoreN);
    uint64_t kCnt = MMV3DivCeil(tiling.Ka, tiling.singleCoreK);
    uint64_t mCoreTail = tiling.M - (mCnt - 1) * tiling.singleCoreM;
    uint64_t nCoreTail = tiling.N - (nCnt - 1) * tiling.singleCoreN;
    uint64_t kCoreTail = tiling.Ka - (kCnt - 1) * tiling.singleCoreK;

    uint64_t mIndex = GetBlockIdx() % mCnt;
    uint64_t nIndex = (GetBlockIdx() / mCnt) % nCnt;
    uint64_t kIndex = GetBlockIdx() / (mCnt * nCnt);
    uint64_t mCoreUse = mIndex == mCnt - 1 ? mCoreTail : tiling.singleCoreM;
    uint64_t nCoreUse = nIndex == nCnt - 1 ? nCoreTail : tiling.singleCoreN;
    uint64_t kCoreUse = kIndex == kCnt - 1 ? kCoreTail : tiling.singleCoreK;
    mm.SetSingleShape(mCoreUse, nCoreUse, kCoreUse);

    uint64_t offsetA = kIndex * tiling.singleCoreK + mIndex * tiling.singleCoreM * tiling.Ka;
    uint64_t offsetB = kIndex * tiling.singleCoreK + nIndex * tiling.singleCoreN * tiling.Kb;
    uint64_t offsetC = nIndex * tiling.singleCoreN + mIndex * tiling.singleCoreM * tiling.N;

    mm.SetTensorA(aGlobal[offsetA], A_TYPE::isTrans);
    mm.SetTensorB(bGlobal[offsetB], B_TYPE::isTrans);
    mm.Iterate();
    // 同步所有核
    CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_AIC_FLAG);
    CrossCoreWaitFlag(SYNC_AIC_FLAG);
    mm.GetTensorC(cGlobal[offsetC], 1);

    SetAtomicNone();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, FIXPIPE_OPT_SELECT FIXPIPE_OPT>
__aicore__ inline void Mc2MatMulMultiCoreSplitK(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM,
                                                const Mc2MatmulV3TilingData &matmulTilingData, GM_ADDR workspaceGM)
{
    if ASCEND_IS_AIV {
        return;
    }
    const TCubeTiling &tiling = matmulTilingData.matmulTiling;
    if ASCEND_IS_AIC {
        Mc2MatMulBlockMultiCoreSplitK<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(aGM, bGM, cGM, biasGM, tiling);
        return;
    }
}
} // namespace Mc2MatmulV3

#endif // __OP_KERNEL_MATMUL_V3_MULTI_CORE_SPLITK_KERNEL_H__
