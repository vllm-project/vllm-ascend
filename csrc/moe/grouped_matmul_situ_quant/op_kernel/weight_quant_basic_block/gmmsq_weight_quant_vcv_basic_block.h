/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gmmsq_weight_quant_vcv_basic_block.h
 * \brief GMMSQ MxA8W4 VCV BasicBlock header file
 */
#ifndef GMMSQ_WEIGHT_QUANT_VCV_BASIC_BLOCK_H
#define GMMSQ_WEIGHT_QUANT_VCV_BASIC_BLOCK_H

#include "kernel_cube_intf.h"
#include "gmmsq_math_common.h"
#include "basic_block_config.h"
#include "basic_block_vf_mx.h"
#include "weight_quant_tool.h"
#include "gmmsq_weight_quant_cube_compute.h"
#include "gmmsq_weight_quant_vec_compute.h"
#include "gmmsq_weight_quant_cube_compute_tools.h"

using AscendC::GetSubBlockIdx;
using AscendC::LocalTensor;
using AscendC::TPosition;
using namespace WeightQuantBatchMatmulV2::Arch35;

namespace GMMSQWeightQuant {

#define GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM \
    template <typename xType, typename wType, typename weightScaleType, typename xScaleType, typename yType, \
              typename yScaleType, const WqmmConfig& wqmmConfig, const VecAntiQuantConfig& vecConfig, typename actType>

// 类声明处带默认实参（默认 SwiGLU）；成员类外定义只能使用不含默认值的版本
#define GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM_DEFAULT \
    template <typename xType, typename wType, typename weightScaleType, typename xScaleType, typename yType, \
              typename yScaleType, const WqmmConfig& wqmmConfig, const VecAntiQuantConfig& vecConfig, \
              typename actType = GmmVecActivationSwiGLU>

#define GMMSQ_WQ_VCV_BASIC_BLOCK_CLASS \
    GMMSQWeightQuantVcvBasicBlock<xType, wType, weightScaleType, xScaleType, yType, yScaleType, wqmmConfig, vecConfig, \
                                  actType>

GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM_DEFAULT
class GMMSQWeightQuantVcvBasicBlock {
public:
    __aicore__ inline GMMSQWeightQuantVcvBasicBlock() = default;
    __aicore__ inline void Init();
    __aicore__ inline void UpdateGlobalAddr(__gm__ xType* x, __gm__ wType* weight, __gm__ weightScaleType* weightScale,
                                            __gm__ xScaleType* xScale, const bool weightL2Cacheable);
    __aicore__ inline void ComputeBasicBlock(const BasicBlockOffsetParam& curOffsetParam,
                                             const BasicBlockOffsetParam& previousOffsetParam);
    __aicore__ inline void SetActParams(float beta, float invBeta, float linearBeta, float invLinearBeta);
    __aicore__ inline void ZeroTailOutput(__gm__ yType* y, __gm__ yScaleType* yScale, uint64_t rowStart,
                                          uint64_t rowCount, uint64_t nOut, uint64_t nScalePerRow);
    __aicore__ inline void End(const BasicBlockOffsetParam& curOffsetParam);

protected:
    __aicore__ inline void IterateNzNkWithAiv(const BasicBlockOffsetParam& curOffsetParam,
                                              const BasicBlockOffsetParam& previousOffsetParam);
    __aicore__ inline void IterateNzNkWithKAic(const BasicBlockOffsetParam& curOffsetParam,
                                               const BasicBlockOffsetParam& previousOffsetParam);
    __aicore__ inline void VecComputeNzNkWithStartLimit(uint64_t& kMte2Offset, uint64_t kMte2Limit,
                                                        const BasicBlockOffsetParam& curOffsetParam);
    __aicore__ inline void UpdateYF32Buffer(const BasicBlockOffsetParam& firstParam,
                                            const BasicBlockOffsetParam& secondParam);

    template <const pipe_t pipe>
    __aicore__ inline void SetAivToAic(uint64_t syncFlag)
    {
#ifndef __CCE_KT_TEST__
        CrossCoreSetFlag<SYNC_MODE4, pipe>(syncFlag);
#endif
    };

    template <const pipe_t pipe>
    __aicore__ inline void WaitAivToAic(uint64_t syncFlag)
    {
#ifndef __CCE_KT_TEST__
        CrossCoreWaitFlag<SYNC_MODE4, pipe>(syncFlag + FLAG_ID_MAX);
        CrossCoreWaitFlag<SYNC_MODE4, pipe>(syncFlag);
#endif
    };

    template <const pipe_t pipe>
    __aicore__ inline void SetAicToAiv(uint64_t syncFlag)
    {
#ifndef __CCE_KT_TEST__
        CrossCoreSetFlag<SYNC_MODE4, pipe>(syncFlag + FLAG_ID_MAX);
        CrossCoreSetFlag<SYNC_MODE4, pipe>(syncFlag);
#endif
    };

    template <const pipe_t pipe>
    __aicore__ inline void WaitAicToAiv(uint64_t syncFlag)
    {
#ifndef __CCE_KT_TEST__
        CrossCoreWaitFlag<SYNC_MODE4, pipe>(syncFlag);
#endif
    };

    GMMSQWeightQuantVecCompute<xType, wType, yType, yScaleType, wqmmConfig, vecConfig, actType> vecCompute_;
    GMMSQWeightQuantCubeCompute<xType, weightScaleType, xScaleType, float, wqmmConfig> cubeCompute_;

    uint64_t cvLoopIdx_ = 0;
    uint64_t weightL1DbOffset_ = 0;
    bool yF32Independent_ = false;

    LocalTensor<xType> weightL1_;
    LocalTensor<float> yF32Buffer_;
};

GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VCV_BASIC_BLOCK_CLASS::Init()
{
    weightL1_ = LocalTensor<xType>(TPosition::TSCM, 0, L1_SIZE_BYTE / sizeof(xType));

    static constexpr uint64_t MXA8W4_WEIGHT_SIZE = 256 * 256;
    weightL1DbOffset_ = L1_SIZE * GetKBUnit<xType>() - MXA8W4_WEIGHT_SIZE;

    uint64_t l1RemainSize = L1_SIZE_BYTE - MXA8W4_WEIGHT_SIZE * DOUBLE_BUFFER_NUM;
    uint64_t l1StartSize = MXA8W4_WEIGHT_SIZE;

    constexpr uint64_t ubOffset = GetGMMSQMxA8W4BufferInfo<vecConfig>().weightLowbitTotalSize;
    constexpr uint64_t highBitSize = GetGMMSQMxA8W4BufferInfo<vecConfig>().weightHighBitTotalSize;
    yF32Buffer_ = LocalTensor<float>(TPosition::LCM, ubOffset, highBitSize / sizeof(float));

    if ASCEND_IS_AIC {
        cubeCompute_.MxA8W4Init(l1RemainSize, l1StartSize);
        SetAicToAiv<PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
        SetAicToAiv<PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
    } else {
        vecCompute_.Init();
    }
    cvLoopIdx_ = 0;
}

GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VCV_BASIC_BLOCK_CLASS::UpdateGlobalAddr(__gm__ xType* x, __gm__ wType* weight,
                                                                        __gm__ weightScaleType* weightScale,
                                                                        __gm__ xScaleType* xScale,
                                                                        const bool weightL2Cacheable)
{
    if ASCEND_IS_AIC {
        cubeCompute_.UpdateGlobalAddr(x, weightScale, xScale);
    } else {
        vecCompute_.UpdateGlobalAddr(weight, weightL2Cacheable);
    }
}

GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VCV_BASIC_BLOCK_CLASS::ComputeBasicBlock(
    const BasicBlockOffsetParam& curOffsetParam, const BasicBlockOffsetParam& previousOffsetParam)
{
    // yF32 承载的可能是本 block 或上一个 block 的 cube 输出，按两者中较大的尺寸选地址，
    // 避免首块（prev 参数尚未填充）时按 0 尺寸误判为「装得下独立区」而越界。
    UpdateYF32Buffer(curOffsetParam, previousOffsetParam);
    if ASCEND_IS_AIV {
        IterateNzNkWithAiv(curOffsetParam, previousOffsetParam);
    } else {
        IterateNzNkWithKAic(curOffsetParam, previousOffsetParam);
    }
}

GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VCV_BASIC_BLOCK_CLASS::UpdateYF32Buffer(
    const BasicBlockOffsetParam& firstParam, const BasicBlockOffsetParam& secondParam)
{
    // weightHighBit 环降为 2 深后只占该区低 64KB，高 64KB 空闲：只要 yF32 装得下就放这里，
    // 与反量化的高 8bit 权重区完全分离，AIC 写 y 时不会覆盖 AIV 在途的反量化结果。
    // mL1 对齐后超过 64 行时（yF32 需要 64~128KB）退回与整个 weightHighBit 区别名，与原实现一致。
    constexpr uint64_t ubOffset = GetGMMSQMxA8W4BufferInfo<vecConfig>().weightLowbitTotalSize;
    constexpr uint64_t highBitSize = GetGMMSQMxA8W4BufferInfo<vecConfig>().weightHighBitTotalSize;
    constexpr uint64_t independentSize = highBitSize / DOUBLE_BUFFER_NUM;

    const uint64_t mL1Size = firstParam.mL1Size > secondParam.mL1Size ? firstParam.mL1Size : secondParam.mL1Size;
    const uint64_t nL1Size = firstParam.nL1Size > secondParam.nL1Size ? firstParam.nL1Size : secondParam.nL1Size;
    uint64_t yF32Size = CeilAlign(mL1Size, static_cast<uint64_t>(BLOCK_CUBE)) *
                        CeilAlign(nL1Size, static_cast<uint64_t>(BLOCK_CUBE)) * sizeof(float);
    yF32Independent_ = yF32Size <= independentSize;
    yF32Buffer_ = yF32Independent_ ?
                      LocalTensor<float>(TPosition::LCM, ubOffset + independentSize,
                                         independentSize / sizeof(float)) :
                      LocalTensor<float>(TPosition::LCM, ubOffset, highBitSize / sizeof(float));
}

GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VCV_BASIC_BLOCK_CLASS::SetActParams(float beta, float invBeta, float linearBeta,
                                                                    float invLinearBeta)
{
    vecCompute_.SetActParams(beta, invBeta, linearBeta, invLinearBeta);
}

GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VCV_BASIC_BLOCK_CLASS::ZeroTailOutput(__gm__ yType* y, __gm__ yScaleType* yScale,
                                                                     uint64_t rowStart, uint64_t rowCount,
                                                                     uint64_t nOut, uint64_t nScalePerRow)
{
    if ASCEND_IS_AIV {
        vecCompute_.ZeroTailOutput(y, yScale, rowStart, rowCount, nOut, nScalePerRow);
    }
}

GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VCV_BASIC_BLOCK_CLASS::IterateNzNkWithAiv(
    const BasicBlockOffsetParam& curOffsetParam, const BasicBlockOffsetParam& previousOffsetParam)
{
    uint64_t kMte2Offset = 0;
    uint64_t curCvLoopIdx = cvLoopIdx_;

    // 只有 yF32 真正落在独立区（不与 weightHighBit 别名）时才能提前放行 AIC 的 fixpipe：
    // 此时 AIC 写上一 block 的 yF32 不会覆盖本 block 在途的反量化结果（上一 block 的 epilogue 已结束）。
    // 别名回退时权重预取会写进同一块 UB，必须保持原来的「预取完成后再放行」顺序。
    bool earlyFixRelease = curCvLoopIdx > 0 && yF32Independent_;
    if (earlyFixRelease) {
        SetAivToAic<PIPE_MTE3>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
    }

    VecComputeNzNkWithStartLimit(kMte2Offset, Min(curOffsetParam.kSize, DOUBLE_BUFFER_NUM * curOffsetParam.kbL1Size),
                                 curOffsetParam);

    if (curCvLoopIdx > 0) {
        if (!earlyFixRelease) {
            SetAivToAic<PIPE_MTE3>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
        }
        WaitAicToAiv<PIPE_V>(SYNC_AIC_FIX_AIV_VF_FLAG);

        // 上一块 epilogue 的 y/yScale 落盘是异步 MTE3，源 buffer 到这才被覆写：
        // 排空点放在这里（而不是上块 epilogue 末尾），中间的本块预取可以与之重叠。
        vecCompute_.WaitQuantOutputMte3();
        // 本块块首预取已发出前几个 K 块的 MTE3 权重拷贝，而下面的 epilogue 会写
        // gluRes_/quantScaleOutput_/halfScale_（与 weightHighBit 环别名）：先排空。
        vecCompute_.WaitWeightHighBitMte3();
        vecCompute_.Activation(yF32Buffer_, previousOffsetParam);
        vecCompute_.ComputeMaxExp(yF32Buffer_, previousOffsetParam);
        vecCompute_.ComputeScale(previousOffsetParam);
        vecCompute_.ComputeDataForQuantTargetFp8(yF32Buffer_, previousOffsetParam);
        vecCompute_.CopyOutputFromUb2Gm(previousOffsetParam,
                                        reinterpret_cast<__gm__ yType*>(previousOffsetParam.yGmAddr));
        vecCompute_.TransMxScaleLayout(previousOffsetParam);
        vecCompute_.CopyScaleFromUb2Gm(previousOffsetParam,
                                       reinterpret_cast<__gm__ yScaleType*>(previousOffsetParam.yScaleGmAddr));
    }

    VecComputeNzNkWithStartLimit(kMte2Offset, curOffsetParam.kSize, curOffsetParam);
}

GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VCV_BASIC_BLOCK_CLASS::VecComputeNzNkWithStartLimit(
    uint64_t& kMte2Offset, uint64_t kMte2Limit, const BasicBlockOffsetParam& curOffsetParam)
{
    uint64_t kMte2BaseSize = curOffsetParam.kbL1Size / DOUBLE_BUFFER_NUM;
    uint64_t nL1AlignSize = CeilAlign(curOffsetParam.nL1Size, static_cast<uint64_t>(BLOCK_CUBE));

    for (; kMte2Offset < kMte2Limit; kMte2Offset += curOffsetParam.kbL1Size, cvLoopIdx_++) {
        vecCompute_.WaitVToMTE2();
        uint64_t kL1Offset = GetSubBlockIdx() * kMte2BaseSize;
        uint64_t mte2RealK = kMte2Offset + kL1Offset >= kMte2Limit ? 0 :
                             kMte2Offset + kL1Offset + kMte2BaseSize >= kMte2Limit ?
                                                                     kMte2Limit - kMte2Offset - kL1Offset :
                                                                     kMte2BaseSize;

        vecCompute_.CopyGmToUb(mte2RealK, kMte2Offset, kL1Offset, curOffsetParam);

        WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);

        vecCompute_.WeightAntiQuantComputeNzNk(
            mte2RealK, weightL1_[(cvLoopIdx_ & 1) * weightL1DbOffset_ + nL1AlignSize * kL1Offset], curOffsetParam);
        SetAivToAic<PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
        vecCompute_.SetVToMTE2();
    }
}

GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VCV_BASIC_BLOCK_CLASS::IterateNzNkWithKAic(
    const BasicBlockOffsetParam& curOffsetParam, const BasicBlockOffsetParam& previousOffsetParam)
{
    if (cvLoopIdx_ > 0) {
        WaitAivToAic<PIPE_FIX>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
        cubeCompute_.GetTensorC(yF32Buffer_, previousOffsetParam);
        SetAicToAiv<PIPE_FIX>(SYNC_AIC_FIX_AIV_VF_FLAG);
    }

    cubeCompute_.SetWeightMTE1ToMTE2(curOffsetParam);
    uint64_t kMutilLoadL1Size = MX_SCALE_K_L1_SIZE;
    for (uint64_t kbL1Offset = 0; kbL1Offset < curOffsetParam.kSize;
         kbL1Offset += curOffsetParam.kbL1Size, cvLoopIdx_++) {
        uint64_t kbL1RealSize = (kbL1Offset + curOffsetParam.kbL1Size) >= curOffsetParam.kSize ?
                                    curOffsetParam.kSize - kbL1Offset :
                                    curOffsetParam.kbL1Size;
        if (kbL1Offset % MX_SCALE_K_L1_SIZE == 0) {
            kMutilLoadL1Size = (kbL1Offset + MX_SCALE_K_L1_SIZE) > curOffsetParam.kSize ?
                                   CeilAlign(curOffsetParam.kSize - kbL1Offset, K_ALIGNMENT64) :
                                   MX_SCALE_K_L1_SIZE;
        }
        cubeCompute_.WaitScaleMTE1ToMTE2(kbL1Offset);
        cubeCompute_.CopyMxScaleGmToL1(curOffsetParam, kbL1Offset);
        cubeCompute_.WaitMTE1ToMTE2(kbL1Offset, curOffsetParam);
        cubeCompute_.CopyAGmToL1(curOffsetParam, kbL1Offset);
        cubeCompute_.WaitMTE1ToMTE2(kbL1Offset, curOffsetParam, kbL1RealSize);
        cubeCompute_.PadL1IfNotAlign(weightL1_, weightL1DbOffset_, cvLoopIdx_, kbL1RealSize, curOffsetParam.nL1Size);
        cubeCompute_.SetMTE1ToMTE2(kbL1Offset, curOffsetParam, kbL1RealSize);
        WaitAivToAic<PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
        cubeCompute_.LaunchMatmul(weightL1_[(cvLoopIdx_ & 1) * weightL1DbOffset_], kbL1Offset, kbL1RealSize,
                                  kMutilLoadL1Size, curOffsetParam);
        cubeCompute_.SetMTE1ToMTE2(kbL1Offset, curOffsetParam);
        cubeCompute_.SetScaleMTE1ToMTE2(kbL1Offset, curOffsetParam);
        SetAicToAiv<PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
    }
}

GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VCV_BASIC_BLOCK_CLASS::End(const BasicBlockOffsetParam& previousOffsetParam)
{
    if ASCEND_IS_AIC {
        if (cvLoopIdx_ > 0) {
            // End 是「最后一个 block」的 epilogue，按它的真实尺寸重选一次 yF32 地址，
            // 与 AIV 侧保持一致（上一轮 ComputeBasicBlock 里的 prev 参数可能仍是首块时的全 0）。
            UpdateYF32Buffer(previousOffsetParam, previousOffsetParam);
            WaitAivToAic<PIPE_FIX>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
            cubeCompute_.GetTensorC(yF32Buffer_, previousOffsetParam);
            SetAicToAiv<PIPE_FIX>(SYNC_AIC_FIX_AIV_VF_FLAG);
        }
        cubeCompute_.EndSync();
    } else {
        if (cvLoopIdx_ > 0) {
            UpdateYF32Buffer(previousOffsetParam, previousOffsetParam);
            SetAivToAic<PIPE_MTE3>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
            WaitAicToAiv<PIPE_V>(SYNC_AIC_FIX_AIV_VF_FLAG);

            vecCompute_.WaitQuantOutputMte3();
            vecCompute_.WaitWeightHighBitMte3();
            vecCompute_.Activation(yF32Buffer_, previousOffsetParam);
            vecCompute_.ComputeMaxExp(yF32Buffer_, previousOffsetParam);
            vecCompute_.ComputeScale(previousOffsetParam);
            vecCompute_.ComputeDataForQuantTargetFp8(yF32Buffer_, previousOffsetParam);
            vecCompute_.CopyOutputFromUb2Gm(previousOffsetParam,
                                            reinterpret_cast<__gm__ yType*>(previousOffsetParam.yGmAddr));
            vecCompute_.TransMxScaleLayout(previousOffsetParam);
            vecCompute_.CopyScaleFromUb2Gm(previousOffsetParam,
                                           reinterpret_cast<__gm__ yScaleType*>(previousOffsetParam.yScaleGmAddr));
        }
        WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
        WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
        vecCompute_.End();
    }
}

} // namespace GMMSQWeightQuant

#endif // GMMSQ_WEIGHT_QUANT_VCV_BASIC_BLOCK_H
