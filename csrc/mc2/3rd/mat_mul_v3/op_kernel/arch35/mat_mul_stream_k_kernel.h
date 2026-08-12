/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mat_mul_stream_k_kernel.h
 * \brief
 */
#ifndef MMV3_MATMUL_STREAM_K_KERNEL_H
#define MMV3_MATMUL_STREAM_K_KERNEL_H
#ifndef DTYPE_Y
#define DTYPE_Y half
#endif
#include "mat_mul_stream_k_block.h"

namespace Mc2MatmulV3Advanced {

using namespace AscendC;
using namespace matmul;

static __aicore__ inline void CustomDataCopyOut(const __gm__ void *gm, const LocalTensor<int8_t> &co1Local,
                                                const void *dataCopyOutParams, const uint64_t tilingPtr,
                                                const uint64_t dataPtr)
{
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    auto params = static_cast<const DataCopyOutParams *>(dataCopyOutParams);
    fixpipeParams.nSize = params->oriNSize;
    fixpipeParams.dstStride = params->dstStride;
    fixpipeParams.mSize = static_cast<uint16_t>(params->burstLen * ONE_BLK_SIZE / BLOCK_CUBE / sizeof(float));
    fixpipeParams.srcStride = (fixpipeParams.mSize + BLOCK_CUBE - 1) / BLOCK_CUBE * BLOCK_CUBE;
    if (params->enUnitFlag) {
        fixpipeParams.unitFlag = MM_FIX_PIPE_UNIT_FLAG; // 使能unitflag的参数 3U
    }
    GlobalTensor<float> dst;
    dst.SetGlobalBuffer((__gm__ float *)(gm));
    if (dataPtr == 1) {
        if constexpr (IsSameType<DTYPE_Y, half>::value) {
            fixpipeParams.quantPre = QuantMode_t::F322F16;
        } else if constexpr (IsSameType<DTYPE_Y, bfloat16_t>::value) {
            fixpipeParams.quantPre = QuantMode_t::F322BF16;
        }
        Fixpipe<DTYPE_Y, float, CFG_ROW_MAJOR>(dst.template ReinterpretCast<DTYPE_Y>(),
                                               co1Local.template ReinterpretCast<float>(), fixpipeParams);
    } else {
        fixpipeParams.quantPre = QuantMode_t::NoQuant;
        if (dataPtr == NUM_TWO) {
            // N轴非对齐情况下将N轴的stride对齐到32*4B=128B搬出，而L0C只能保证16*4B=64B的清零，
            // 当N轴余32不大于16时需要另外对workspace清零
            fixpipeParams.nSize = MMV3CeilAlign(params->oriNSize, BLOCK_SIZE);
            fixpipeParams.dstStride = MMV3CeilAlign(params->dstStride, BLOCK_BYTE_SIZE);
        }
        Fixpipe<float, float, CFG_ROW_MAJOR>(dst, co1Local.template ReinterpretCast<float>(), fixpipeParams);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = Mc2MatmulStreamKBlock,
          const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD, const bool ALIGN_FLAG = false>
class Mc2MatmulStreamKKernel {
public:
    __aicore__ inline Mc2MatmulStreamKKernel()
    {
    }
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                GM_ADDR workspaceGM, const void *tilingData, TPipe *pipe);
    __aicore__ inline void InitInputs(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR workspaceGM);
    __aicore__ inline void Process();
    __aicore__ inline void StreamKAicProcess(uint64_t roundIdx);
    __aicore__ inline void StreamKAivProcess(uint64_t roundIdx);
    __aicore__ inline void CheckNeedClearWorkSpace(uint64_t roundIdx);
    __aicore__ inline void End()
    {
        mm_.End();
    }
    __aicore__ inline const BLOCK_TYPE GetBlock()
    {
        return block_;
    }

protected:
    BLOCK_TYPE block_;
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;

    using WorkspaceType = MatmulType<C_TYPE::pos, C_TYPE::format, float>;
    MatmulImpl<A_TYPE, B_TYPE, WorkspaceType, BIAS_TYPE, MM_CFG,
               MatmulCallBackFunc<CustomDataCopyOut, nullptr, nullptr>>
        mm_;
    GlobalTensor<A_T> aGlobal_;
    GlobalTensor<B_T> bGlobal_;
    GlobalTensor<float> cGlobal_;
    GlobalTensor<float> workspaceGlobal_;
    GlobalTensor<BiasT> biasGlobal_;
    TPipe *pipe_;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          const bool ALIGN_FLAG>
__aicore__ inline void Mc2MatmulStreamKKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, ALIGN_FLAG>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const void *tilingData, TPipe *pipe)
{
    pipe_ = pipe;
    block_.template Init<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(tilingData);
    InitInputs(aGM, bGM, cGM, biasGM, workspaceGM);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          const bool ALIGN_FLAG>
__aicore__ inline void
Mc2MatmulStreamKKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, ALIGN_FLAG>::InitInputs(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR workspaceGM)
{
    workspaceGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspaceGM),
                                     block_.aicParams_.lastLoopTotalCnt * BLOCK_BASE_M * BLOCK_BASE_N);
    cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(cGM),
                             static_cast<uint64_t>(block_.matmulTilingData_->tCubeTiling.M) *
                                     block_.matmulTilingData_->tCubeTiling.N * sizeof(C_T) / sizeof(float) +
                                 1UL);
    if ASCEND_IS_AIC {
        aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(aGM),
                                 static_cast<uint64_t>(block_.matmulTilingData_->tCubeTiling.M) *
                                     block_.matmulTilingData_->tCubeTiling.Ka);
        bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM),
                                 static_cast<uint64_t>(block_.matmulTilingData_->tCubeTiling.Kb) *
                                     block_.matmulTilingData_->tCubeTiling.N);
        if (block_.matmulTilingData_->tCubeTiling.isBias) {
            biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT *>(biasGM),
                                        block_.matmulTilingData_->tCubeTiling.N);
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          const bool ALIGN_FLAG>
__aicore__ inline void
Mc2MatmulStreamKKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, ALIGN_FLAG>::Process()
{
    if ASCEND_IS_AIV {
        if (GetBlockIdx() >= block_.aicParams_.lastLoopTotalCnt * GetTaskRation()) {
            CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIC_SYNC_AIV_FLAG);
            SyncAll();
            return;
        }
        ICachePreLoad(2);
        StreamKAivProcess(block_.params_.round - 1);
        return;
    }

    if ASCEND_IS_AIC {
        if (GetBlockIdx() >= block_.params_.totalCnt) {
            CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + VEC0_FLAG_ID_OFFSET);
            CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + VEC1_FLAG_ID_OFFSET);
            return;
        }
        SetAtomicNone();
        mm_.SetSubBlockIdx(0);
        mm_.Init(&block_.matmulTilingData_->tCubeTiling, pipe_);
        mm_.SetHF32(block_.matmulTilingData_->isHf32, 1);
        SetMMLayoutTransform(true); // fixp使用n搬出，达到cube和fixp并行的效果
        for (uint64_t j = 0; j < block_.params_.round; j++) {
            StreamKAicProcess(block_.UpdateLoopIndex(j));
        }
        mm_.SetHF32(false, 0);
        SetMMLayoutTransform(false);
        SetAtomicNone();
        return;
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          const bool ALIGN_FLAG>
__aicore__ inline void
Mc2MatmulStreamKKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, ALIGN_FLAG>::StreamKAicProcess(
    uint64_t roundIdx)
{
    block_.UpdateBasicIndex(roundIdx); // 使能ASWT更新Index
    if (block_.params_.index < block_.params_.totalCnt) {
        block_.template UpdateBlockParams<ALIGN_FLAG>(roundIdx);
        if (block_.aicParams_.singleCoreK > 0) {
            block_.template CalcGMOffset<A_TYPE, B_TYPE>(roundIdx);
            mm_.SetSingleShape(block_.params_.singleCoreM, block_.params_.singleCoreN, block_.aicParams_.singleCoreK);
            mm_.SetTensorA(aGlobal_[block_.offset_.offsetA], A_TYPE::isTrans);
            mm_.SetTensorB(bGlobal_[block_.offset_.offsetB], B_TYPE::isTrans);
            if (block_.matmulTilingData_->tCubeTiling.isBias) {
                if (block_.aicParams_.kCntIndex == 0) {
                    mm_.SetBias(biasGlobal_[block_.offset_.offsetBias]);
                } else {
                    mm_.ClearBias();
                }
            }
            mm_.Iterate();
            // dp场景直接输出到C矩阵，其他场景还是走workspace
            if (roundIdx != block_.params_.round - 1) {
                constexpr uint64_t dataPtr = static_cast<uint64_t>(sizeof(C_T) == sizeof(half));
                mm_.SetSelfDefineData(dataPtr);
                mm_.GetTensorC(cGlobal_[block_.offset_.offsetC * sizeof(C_T) / sizeof(float)], 0);
            } else {
                constexpr uint64_t dataPtr = ALIGN_FLAG ? 2 : 0;
                mm_.SetSelfDefineData(dataPtr);
                CheckNeedClearWorkSpace(roundIdx);
                // true表示输出按照baseM,baseN连续输出
                mm_.GetTensorC(workspaceGlobal_[block_.aicParams_.offsetCWorkspace], 0, true);
                CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + VEC0_FLAG_ID_OFFSET);
                CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + VEC1_FLAG_ID_OFFSET);
            }
        } else {
            CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + VEC0_FLAG_ID_OFFSET);
            CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + VEC1_FLAG_ID_OFFSET);
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          const bool ALIGN_FLAG>
__aicore__ inline void
Mc2MatmulStreamKKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, ALIGN_FLAG>::StreamKAivProcess(
    uint64_t roundIdx)
{
    block_.UpdateBasicIndex(roundIdx);
    block_.template UpdateBlockParams<ALIGN_FLAG>(roundIdx);
    block_.CalcAivBaseParams();
    CheckNeedClearWorkSpace(roundIdx);
    for (uint64_t index = 0; index < block_.aivParams_.aivMte2Num; ++index) {
        block_.UpdateAivParams(index, roundIdx);
        // DP+SK场景，需要降低AIV带宽，所以进行同步保证串行
        if (block_.params_.round >= 2) {
            TPipeSetWaitFlag<HardEvent::MTE3_MTE2>();
        }
        TBuf<TPosition::VECIN> ubAddBuf;
        pipe_->InitBuffer(ubAddBuf,
                          block_.aivParams_.copyGm2UbBurstLen * block_.aivParams_.copyGm2UbKCnt * sizeof(float));
        LocalTensor<float> ubAddTensor =
            ubAddBuf.Get<float>(block_.aivParams_.copyGm2UbBurstLen * block_.aivParams_.copyGm2UbKCnt);
        DataCopyExtParams dataCopyExtParams{static_cast<uint16_t>(block_.aivParams_.copyGm2UbKCnt),
                                            static_cast<uint32_t>(block_.aivParams_.copyGm2UbBurstLen * sizeof(float)),
                                            static_cast<uint32_t>(block_.aivParams_.copyGm2UbSrcGap * sizeof(float)), 0,
                                            0};
        if (index == 0) {
            CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIC_SYNC_AIV_FLAG);
            SyncAll();
        }
        if (block_.aivParams_.copyGm2UbMBurst == 0) {
            return;
        }
        DataCopyPad<float>(ubAddTensor, workspaceGlobal_[block_.aivParams_.offsetWorkspaceSrc], dataCopyExtParams,
                           {false, 0, 0, 0});
        TPipeSetWaitFlag<HardEvent::MTE2_V>();
        for (uint64_t i = 1; i < block_.aivParams_.copyGm2UbKCnt; ++i) {
            Add(ubAddTensor, ubAddTensor, ubAddTensor[i * block_.aivParams_.copyGm2UbBurstLen],
                block_.aivParams_.copyGm2UbBurstLen);
        }
        DataCopyExtParams ub2gmExtParams{static_cast<uint16_t>(block_.aivParams_.copyUb2GmMBurst),
                                         static_cast<uint32_t>(block_.aivParams_.copyUb2GmBurstLen * sizeof(C_T)),
                                         static_cast<uint32_t>(block_.aivParams_.copyUb2GmSrcGap * sizeof(C_T) / 32),
                                         static_cast<uint32_t>(block_.aivParams_.copyUb2GmDstGap * sizeof(C_T)), 0};
        if constexpr (sizeof(C_T) == sizeof(half)) {
            LocalTensor<C_T> ubCastDst = ubAddTensor.template ReinterpretCast<C_T>();
            Cast(ubCastDst, ubAddTensor, RoundMode::CAST_RINT, block_.aivParams_.copyGm2UbBurstLen);
            TPipeSetWaitFlag<HardEvent::V_MTE3>();
            DataCopyPad<C_T, ALIGN_FLAG ? PaddingMode::Normal : PaddingMode::Compact>(
                cGlobal_.template ReinterpretCast<C_T>()[block_.aivParams_.offsetCGm], ubCastDst, ub2gmExtParams);
        } else {
            TPipeSetWaitFlag<HardEvent::V_MTE3>();
            DataCopyPad<C_T, ALIGN_FLAG ? PaddingMode::Normal : PaddingMode::Compact>(
                cGlobal_.template ReinterpretCast<C_T>()[block_.aivParams_.offsetCGm], ubAddTensor, ub2gmExtParams);
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          const bool ALIGN_FLAG>
__aicore__ inline void
Mc2MatmulStreamKKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, ALIGN_FLAG>::CheckNeedClearWorkSpace(
    uint64_t roundIdx)
{
    if constexpr (ALIGN_FLAG) {
        if ASCEND_IS_AIV {
            // 当N轴余32不大于16时需要对workspace清零
            if (block_.params_.singleCoreN % BLOCK_BYTE_SIZE > BLOCK_SIZE ||
                block_.params_.singleCoreN % BLOCK_BYTE_SIZE == 0 || GetSubBlockIdx() == 1) {
                return;
            }
            // 只用AIC对应的一个AIV进行清零，清零大小就是singleCoreM*16的范围
            uint64_t offsetWorkspaceSrc = (block_.params_.index * block_.matmulTilingData_->kTailCnt +
                                           block_.aicParams_.kCntIndex / GetTaskRation()) *
                                              BLOCK_BASE_M * BLOCK_BASE_N +
                                          block_.aicParams_.alignSingleCoreN - BLOCK_SIZE;
            uint64_t calCount = block_.params_.singleCoreM * BLOCK_SIZE;
            TBuf<TPosition::VECOUT> ubZeroBuf;
            pipe_->InitBuffer(ubZeroBuf, calCount * sizeof(float));
            LocalTensor<float> ubZeroTensor = ubZeroBuf.Get<float>(calCount);
            Duplicate(ubZeroTensor, (float)0, calCount);
            DataCopyExtParams clearWorkspaceExtParams{
                static_cast<uint16_t>(block_.params_.singleCoreM), static_cast<uint32_t>(BLOCK_SIZE * sizeof(float)), 0,
                static_cast<uint32_t>((block_.aicParams_.alignSingleCoreN - BLOCK_SIZE) * sizeof(float)), 0};
            workspaceGlobal_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            DataCopyPad<float>(workspaceGlobal_[offsetWorkspaceSrc], ubZeroTensor, clearWorkspaceExtParams);
            CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG);
        }
        if ASCEND_IS_AIC {
            if (block_.params_.singleCoreN % BLOCK_BYTE_SIZE <= BLOCK_SIZE &&
                block_.params_.singleCoreN % BLOCK_BYTE_SIZE != 0) {
                CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + VEC0_FLAG_ID_OFFSET);
            }
        }
    }
}
} // namespace Mc2MatmulV3Advanced

#endif // MMV3_MATMUL_STREAM_K_KERNEL_H
