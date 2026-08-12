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
 * \file weight_quant_batch_matmul_v2_cube_compute.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCHMATMUL_V2_CUBE_COMPUTE_H
#define WEIGHT_QUANT_BATCHMATMUL_V2_CUBE_COMPUTE_H

#include "basic_block_config.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"

using AscendC::BLOCK_CUBE;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::PipeBarrier;
using AscendC::SetFlag;
using AscendC::TBuf;
using AscendC::TPosition;
using AscendC::WaitFlag;
using matmul::MatmulImpl;
using matmul::MatmulType;

namespace Mc2WeightQuantBatchMatmulV2::Arch35 {

#define WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM                                                                              \
    template <typename xType, typename biasType, typename yType, const WqmmConfig &wqmmConfig, typename MatmulImplType>

#define WQBMM_CUBE_COMPUTE_CLASS WeightQuantBatchMatmulV2CubeCompute<xType, biasType, yType, wqmmConfig, MatmulImplType>

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
class WeightQuantBatchMatmulV2CubeCompute {
public:
    __aicore__ inline WeightQuantBatchMatmulV2CubeCompute(){};
    __aicore__ inline void UpdateGlobalAddr(__gm__ xType *x, __gm__ yType *y, __gm__ biasType *bias,
                                            __gm__ uint64_t *quantScale, const bool isBias);
    __aicore__ inline void Init(TBuf<TPosition::TSCM> &l1Tbuf, uint64_t weightL1Space, uint64_t aPreloadSize,
                                const TCubeTiling *__restrict matmulTiling, AscendC::TPipe *tPipe);
    __aicore__ inline void LaunchMatmul(const LocalTensor<xType> &weightF16L1, int64_t kbOffset, uint64_t kbL1RealSize,
                                        const BasicBlockOffsetParam &param, uint64_t cvLoopIdx);
    __aicore__ inline void WaitMTE1ToMTE2(uint64_t cvLoopIdx);
    __aicore__ inline void SetMTE1ToMTE2(uint64_t cvLoopIdx);
    __aicore__ inline void CopyAAndBiasGmToL1(const BasicBlockOffsetParam &param, int64_t kaGmOffset,
                                              int64_t kbL1RealSize, int64_t biasRealN, uint64_t cvLoopIdx);
    __aicore__ inline void GetTensorC(const BasicBlockOffsetParam &param);
    __aicore__ inline void GetTensorC(LocalTensor<yType> &yUb);
    __aicore__ inline void EndSync(uint64_t cvLoopIdx);
    __aicore__ inline void ClearAFullLoadFlag();

private:
    __aicore__ inline void PreloadA(uint64_t aPreloadSize, const LocalTensor<xType> &perloadBuffer,
                                    const TCubeTiling *__restrict matmulTiling);
    __aicore__ inline void InitSync();
    __aicore__ inline uint64_t CheckMaxSpace(const BasicBlockOffsetParam &param);
    __aicore__ inline void CopyAGmToL1SingleBuffer(const BasicBlockOffsetParam &param, int64_t kaGmOffset,
                                                   int64_t kbL1RealSize, int64_t biasRealN, uint64_t cvLoopIdx,
                                                   int64_t aGmOffset);

    int8_t al1DbNum_;
    bool isBias_;
    uint64_t quantScaleValue_;
    static constexpr uint32_t KB_UNIT = GetKBUnit<xType>();

    MatmulImplType mmObj_;

    uint64_t aL1Count_;
    uint64_t aL1MaxHalfCount_;

    AscendC::TEventID cubeEventIdsMte1ToMte2_[DOUBLE_BUFFER_NUM];
    AscendC::TEventID cubeEventIdMte2ToMte1_;
    GlobalTensor<xType> xGlobal_;
    GlobalTensor<biasType> biasGlobal_;
    GlobalTensor<uint64_t> quantScaleGlobal_;
    GlobalTensor<yType> yGlobal_;

    LocalTensor<xType> aF16L1_;
    uint64_t aF16L1DbOffset_;

    LocalTensor<biasType> biasL1_;
    uint64_t biasL1DbOffset_;
};

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline uint64_t WQBMM_CUBE_COMPUTE_CLASS::CheckMaxSpace(const BasicBlockOffsetParam &param)
{
    uint64_t maxSpace = aL1MaxHalfCount_ * param.kbL1Size * CeilAlign(param.mL1Size, static_cast<uint64_t>(BLOCK_CUBE));
    if (param.kbL1Size > 0 && param.kSize % param.kbL1Size == 0 && !wqmmConfig.aTrans && maxSpace <= aF16L1DbOffset_) {
        return maxSpace;
    }
    return 0;
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::LaunchMatmul(const LocalTensor<xType> &weightF16L1, int64_t kbOffset,
                                                              uint64_t kbL1RealSize, const BasicBlockOffsetParam &param,
                                                              uint64_t cvLoopIdx)
{
    mmObj_.SetOrgShape(CeilAlign(param.mL1Size, static_cast<uint64_t>(BLOCK_CUBE)),
                       CeilAlign(param.nL1Size, static_cast<uint64_t>(BLOCK_CUBE)),
                       CeilAlign(kbL1RealSize, static_cast<uint64_t>(BLOCK_CUBE)),
                       CeilAlign(kbL1RealSize, static_cast<uint64_t>(BLOCK_CUBE)), param.nSize);
    if (al1DbNum_ == SINGLE_BUFFER_NUM) {
        uint64_t maxSpace = CheckMaxSpace(param);
        if (maxSpace > 0) {
            // block = kbOffset / param.kbL1Size 计算是在第几块
            // blockOffset = block / 2 确定从A0还是A1读取数据后，在块内的偏移，单位是块
            // k = blockOffset * param.kbL1Size 当前块内的偏移量kOffset，单位是元素
            // 将block和blockOffset带入，计算k
            // k = (kbOffset / param.kbL1Size) / 2 * param.kbL1Size
            // 块内偏移量 = m * k
            // 举例：
            // L1A: |0|2|4|      |1|3|
            //      |A0:0~128KB  |A1:128KB~256KB|
            // 第5块（block = 5），在A1中偏移为3（blockOffset = 3），块内偏移量为m * (3 * k)
            mmObj_.SetTensorA(aF16L1_[(cvLoopIdx & 1) * aF16L1DbOffset_ +
                                      CeilAlign(param.mL1Size, static_cast<uint64_t>(BLOCK_CUBE)) *
                                          (static_cast<uint64_t>(kbOffset) / (param.kbL1Size * 2) * param.kbL1Size)],
                              wqmmConfig.aTrans);
        } else {
            mmObj_.SetTensorA(aF16L1_[CeilAlign(param.mL1Size, static_cast<uint64_t>(BLOCK_CUBE)) * kbOffset],
                              wqmmConfig.aTrans);
        }
    } else {
        mmObj_.SetTensorA(aF16L1_[(cvLoopIdx & 1) * aF16L1DbOffset_], wqmmConfig.aTrans);
    }

    mmObj_.SetTensorB(weightF16L1, wqmmConfig.bTrans);

    if (isBias_) {
        mmObj_.SetBias(biasL1_[(cvLoopIdx & 1) * biasL1DbOffset_]);
    }

    mmObj_.SetTail(param.mL1Size, param.nL1Size, kbL1RealSize);

    if constexpr (IsSameType<yType, int8_t>::value) {
        if constexpr (wqmmConfig.quantType == Mc2QuantType::PER_TENSOR) {
            mmObj_.SetQuantScalar(quantScaleValue_);
        } else {
            mmObj_.SetQuantVector(quantScaleGlobal_[param.nOffset]);
        }
    }

    mmObj_.Iterate(kbOffset != 0);
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::WaitMTE1ToMTE2(uint64_t cvLoopIdx)
{
    // 编译器对成员变量数组访问优化能力较弱，会引入大量scalar，此处抽取局部变量，规避编译器优化问题
    AscendC::TEventID tempEventIdsMte1ToMte2[DOUBLE_BUFFER_NUM] = {cubeEventIdsMte1ToMte2_[0],
                                                                   cubeEventIdsMte1ToMte2_[1]};
    // 单buffer时保证了A一次全载不需要Wait，Double buffer时首次使用不需要Wait
    if (al1DbNum_ > SINGLE_BUFFER_NUM && cvLoopIdx >= DOUBLE_BUFFER_NUM) {
        WaitFlag<HardEvent::MTE1_MTE2>(tempEventIdsMte1ToMte2[cvLoopIdx & 1]);
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::SetMTE1ToMTE2(uint64_t cvLoopIdx)
{
    // 编译器对成员变量数组访问优化能力较弱，会引入大量scalar，此处抽取局部变量，规避编译器优化问题
    AscendC::TEventID tempEventIdsMte1ToMte2[DOUBLE_BUFFER_NUM] = {cubeEventIdsMte1ToMte2_[0],
                                                                   cubeEventIdsMte1ToMte2_[1]};
    if (al1DbNum_ > SINGLE_BUFFER_NUM) {
        SetFlag<HardEvent::MTE1_MTE2>(tempEventIdsMte1ToMte2[cvLoopIdx & 1]);
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::CopyAGmToL1SingleBuffer(const BasicBlockOffsetParam &param,
                                                                         int64_t kaGmOffset, int64_t kbL1RealSize,
                                                                         int64_t biasRealN, uint64_t cvLoopIdx,
                                                                         int64_t aGmOffset)
{
    AscendC::Nd2NzParams nd2nzParams;
    uint64_t maxSpace = CheckMaxSpace(param);
    if (maxSpace > 0) {
        nd2nzParams.ndNum = aL1MaxHalfCount_;
        nd2nzParams.nValue = param.mL1Size;
        nd2nzParams.dValue = param.kbL1Size;
        nd2nzParams.srcDValue = param.kSize;
        nd2nzParams.srcNdMatrixStride = 2 * nd2nzParams.dValue;
        nd2nzParams.dstNzC0Stride = CeilAlign(nd2nzParams.nValue, static_cast<uint16_t>(BLOCK_CUBE));
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride =
            nd2nzParams.dstNzC0Stride * CeilAlign(nd2nzParams.dValue, static_cast<uint32_t>(BLOCK_CUBE));
        DataCopy(aF16L1_[(cvLoopIdx & 1) * aF16L1DbOffset_], xGlobal_[aGmOffset], nd2nzParams);

        nd2nzParams.ndNum = aL1Count_ - aL1MaxHalfCount_;
        DataCopy(aF16L1_[((cvLoopIdx + 1) & 1) * aF16L1DbOffset_], xGlobal_[aGmOffset + nd2nzParams.dValue],
                 nd2nzParams);
    } else {
        nd2nzParams.ndNum = 1;
        if constexpr (wqmmConfig.aTrans) {
            nd2nzParams.nValue = param.kSize;
            nd2nzParams.dValue = param.mL1Size;
            nd2nzParams.srcDValue = param.mSize;
        } else {
            nd2nzParams.nValue = param.mL1Size;
            nd2nzParams.dValue = param.kSize;
            nd2nzParams.srcDValue = param.kSize;
        }
        nd2nzParams.srcNdMatrixStride = 0;
        nd2nzParams.dstNzC0Stride = CeilAlign(nd2nzParams.nValue, static_cast<uint16_t>(BLOCK_CUBE));
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 0;

        DataCopy(aF16L1_, xGlobal_[aGmOffset], nd2nzParams);
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::CopyAAndBiasGmToL1(const BasicBlockOffsetParam &param,
                                                                    int64_t kaGmOffset, int64_t kbL1RealSize,
                                                                    int64_t biasRealN, uint64_t cvLoopIdx)
{
    int64_t aGmOffset;
    if constexpr (!wqmmConfig.aTrans) {
        aGmOffset = param.mOffset * param.kSize + kaGmOffset;
    } else {
        aGmOffset = kaGmOffset * param.mSize + param.mOffset;
    }

    if (al1DbNum_ > SINGLE_BUFFER_NUM) {
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        if constexpr (wqmmConfig.aTrans) {
            nd2nzParams.nValue = kbL1RealSize;
            nd2nzParams.dValue = param.mL1Size;
            nd2nzParams.srcDValue = param.mSize;
        } else {
            nd2nzParams.nValue = param.mL1Size;
            nd2nzParams.dValue = kbL1RealSize;
            nd2nzParams.srcDValue = param.kSize;
        }
        nd2nzParams.srcNdMatrixStride = 0;
        nd2nzParams.dstNzC0Stride = CeilAlign(nd2nzParams.nValue, static_cast<uint16_t>(BLOCK_CUBE));
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 0;

        DataCopy(aF16L1_[(cvLoopIdx & 1) * aF16L1DbOffset_], xGlobal_[aGmOffset], nd2nzParams);
    } else if (al1DbNum_ == SINGLE_BUFFER_NUM && kaGmOffset == 0) {
        CopyAGmToL1SingleBuffer(param, kaGmOffset, kbL1RealSize, biasRealN, cvLoopIdx, aGmOffset);
    }

    // bias仅与n有关，与k无关，所以只需要拷贝一次
    if (isBias_ && kaGmOffset == 0) {
        DataCopy(biasL1_[(cvLoopIdx & 1) * biasL1DbOffset_], biasGlobal_[param.nOffset],
                 CeilAlign(biasRealN, static_cast<int64_t>(BLOCK_CUBE)));
    }

    SetFlag<HardEvent::MTE2_MTE1>(cubeEventIdMte2ToMte1_);
    WaitFlag<HardEvent::MTE2_MTE1>(cubeEventIdMte2ToMte1_);
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::EndSync(uint64_t cvLoopIdx)
{
    AscendC::TEventID tempEventIdsMte1ToMte2[DOUBLE_BUFFER_NUM] = {cubeEventIdsMte1ToMte2_[0],
                                                                   cubeEventIdsMte1ToMte2_[1]};

    // 考虑到只循环一次时， 只需要同步wait第0块缓存。 不止1次时， 2个同步块都需要wait
    if (cvLoopIdx > 1 && al1DbNum_ > SINGLE_BUFFER_NUM) {
        WaitFlag<HardEvent::MTE1_MTE2>(tempEventIdsMte1ToMte2[cvLoopIdx & 1]);
    }

    if (cvLoopIdx > 0 && al1DbNum_ > SINGLE_BUFFER_NUM) {
        WaitFlag<HardEvent::MTE1_MTE2>(tempEventIdsMte1ToMte2[(cvLoopIdx + 1) & 1]);
    }

    GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(tempEventIdsMte1ToMte2[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(tempEventIdsMte1ToMte2[1]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE1>(cubeEventIdMte2ToMte1_);
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::ClearAFullLoadFlag()
{
    if (al1DbNum_ == SINGLE_BUFFER_NUM) {
        SetFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMte1ToMte2_[0]);
        WaitFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMte1ToMte2_[0]);
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::InitSync()
{
    cubeEventIdsMte1ToMte2_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
    cubeEventIdsMte1ToMte2_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
    cubeEventIdMte2ToMte1_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>();
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::UpdateGlobalAddr(__gm__ xType *x, __gm__ yType *y,
                                                                  __gm__ biasType *bias, __gm__ uint64_t *quantScale,
                                                                  const bool isBias)
{
    isBias_ = isBias;
    xGlobal_.SetGlobalBuffer(x);
    yGlobal_.SetGlobalBuffer(y);
    if (isBias_) {
        biasGlobal_.SetGlobalBuffer(bias);
    }
    if constexpr (IsSameType<yType, int8_t>::value) {
        quantScaleGlobal_.SetGlobalBuffer(quantScale);
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::PreloadA(uint64_t aPreloadSize,
                                                          const LocalTensor<xType> &perloadBuffer,
                                                          const TCubeTiling *__restrict matmulTiling)
{
    uint64_t xOffset = GetBlockIdx() * aPreloadSize;
    uint64_t xSizeLimit = matmulTiling->M * matmulTiling->Ka;
    if (aPreloadSize == 0 || xOffset >= xSizeLimit) {
        return;
    }
    DataCopyPadExtParams<xType> extParams;
    DataCopyExtParams param;
    param.blockCount = 1;
    param.blockLen = (xOffset + aPreloadSize > xSizeLimit ? xSizeLimit - xOffset : aPreloadSize) * sizeof(xType);
    param.srcStride = 0;
    param.dstStride = 0;
    DataCopyPad(perloadBuffer, xGlobal_[xOffset], param, extParams);
    PipeBarrier<PIPE_MTE2>();
}

// 场景1： 使能a preload。必须先更新地址再init
// 场景2： gm地址变化需要实时获取场景，必须先init再更新地址
WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::Init(TBuf<TPosition::TSCM> &l1Tbuf, uint64_t weightL1Space,
                                                      uint64_t aPreloadSize, const TCubeTiling *__restrict matmulTiling,
                                                      AscendC::TPipe *tPipe)
{
    // (1) y 数据类型为 int8 时，quantScale需要预留一份空间
    //  ① 有bias
    //  L1 (0~512KB): WeightL1_P0(128KB) | Bias_P0(4KB) | AL1_P0(120KB) | AL1_P1(120KB) | Bias_P1(4KB) |
    //  WeightL1_P1(128KB) | quantScale(8KB) ② 无bias L1 (0~512KB): WeightL1_P0(128KB) | AL1_P0(124KB) | AL1_P1(124KB) |
    //  WeightL1_P1(128KB) | quantScale(8KB)
    // (2) L1 上有bias时:
    //  L1 (0~512KB): WeightL1_P0(128KB) | Bias_P0(4KB) | AL1_P0(124KB) | AL1_P1(124KB) | Bias_P1(4KB) |
    //  WeightL1_P1(128KB)
    // (3) 其他场景时
    //  L1 (0~512KB): WeightL1_P0(128KB) | AL1_P0(128KB) | AL1_P1(128KB) | WeightL1_P1(128KB)

    uint64_t biasL1Space = matmulTiling->isBias ? 4 * KB_UNIT : 0; // bias单块分配4K空间
    uint64_t aF16L1Offset = weightL1Space + biasL1Space;           // A要跳过WeightL1_P0 + Bias_P0
    if constexpr (IsSameType<yType, int8_t>::value) {
        uint64_t aF16L1Space = 504 * KB_UNIT - DOUBLE_BUFFER_NUM * aF16L1Offset; // L1上A可占据剩余空间
        aF16L1DbOffset_ = aF16L1Space >> 1;
    } else if (matmulTiling->isBias) {
        uint64_t aF16L1Space = L1_SIZE * KB_UNIT - DOUBLE_BUFFER_NUM * aF16L1Offset; // L1上A可占据剩余空间
        aF16L1DbOffset_ = aF16L1Space >> 1;
        if constexpr (IsSameType<biasType, float>::value) {
            biasL1_ = l1Tbuf.Get<biasType>()[weightL1Space >> 1];
            biasL1DbOffset_ = (aF16L1Space + biasL1Space) >> 1;
        } else {
            biasL1_ = l1Tbuf.Get<biasType>()[weightL1Space];
            biasL1DbOffset_ = aF16L1Space + biasL1Space;
        }
    } else {
        aF16L1DbOffset_ = 256 * KB_UNIT - weightL1Space;
    }
    aF16L1_ = l1Tbuf.Get<xType>()[aF16L1Offset];
    aL1Count_ = matmulTiling->Ka / (matmulTiling->baseK * matmulTiling->stepKb);
    aL1MaxHalfCount_ = CeilDiv(aL1Count_, static_cast<uint64_t>(DOUBLE_BUFFER_NUM));

    PreloadA(aPreloadSize, aF16L1_, matmulTiling);
    uint64_t kaL1Size = matmulTiling->stepKa * matmulTiling->baseK;
    // 当前tiling策略的细分场景：
    // 1. stepKa <= stepKb 当前限制baseM的最大值，因此该场景下在L1上A矩阵大小<=128k。可以固定走db分支，保证a矩阵的db载入
    // 2. stepKa > stepKb 当前在m小k大的情况下才会出现该场景，走全载分支
    // 3. gmm场景，不知道真实的m值，tiling采取保守策略，恒定走db分支
    if (matmulTiling->stepKa > matmulTiling->stepKb) {
        al1DbNum_ = SINGLE_BUFFER_NUM;
    } else {
        al1DbNum_ = DOUBLE_BUFFER_NUM;
    }
    mmObj_.SetSubBlockIdx(0);
    mmObj_.Init(matmulTiling, tPipe);
    InitSync();

    if constexpr (IsSameType<yType, int8_t>::value && wqmmConfig.quantType == Mc2QuantType::PER_TENSOR) {
        quantScaleValue_ = this->quantScaleGlobal_.GetValue(0);
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::GetTensorC(const BasicBlockOffsetParam &param)
{
    uint64_t outOffset = param.mOffset * param.nSize + param.nOffset;
#ifndef __CCE_KT_TEST__
    mmObj_.GetTensorC(yGlobal_[outOffset]);
#endif
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::GetTensorC(LocalTensor<yType> &yUb)
{
#ifndef __CCE_KT_TEST__
    mmObj_.GetTensorC(yUb, 0, true);
#endif
}
} // namespace Mc2WeightQuantBatchMatmulV2::Arch35
#endif
