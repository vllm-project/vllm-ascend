/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file weight_quant_cube_compute.h
 * \brief
 */
#ifndef GROUPED_MATMUL_WEIGHT_QUANT_CUBE_COMPUTE_H
#define GROUPED_MATMUL_WEIGHT_QUANT_CUBE_COMPUTE_H

#include "basic_block_config.h"
#include "custom_policy/wqbmm_custom_policy.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "lib/matmul_intf.h"
#include "weight_quant_tool.h"

#include "basic_api/weight_quant_basic_api_v1.h"

using AscendC::Dn2NzParams;
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

namespace WeightQuantBatchMatmulV2::Arch35 {

#define WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM                                                                 \
    template <typename xType, typename biasType, typename antiQuantScaleType, typename perTokenScaleType, \
              typename yType, const WqmmConfig &wqmmConfig, typename MatmulImplType>

#define WQBMM_CUBE_COMPUTE_CLASS                                                                                   \
    WeightQuantBatchMatmulV2CubeCompute<xType, biasType, antiQuantScaleType, perTokenScaleType, yType, wqmmConfig, \
                                        MatmulImplType>

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
class WeightQuantBatchMatmulV2CubeCompute {
public:
    __aicore__ inline WeightQuantBatchMatmulV2CubeCompute(){};
    __aicore__ inline void UpdateGlobalAddr(__gm__ xType *x, __gm__ yType *y, __gm__ biasType *bias,
                                            __gm__ antiQuantScaleType *antiquantScale, __gm__ uint64_t *quantScale,
                                            __gm__ perTokenScaleType *perTokenScale, const bool isBias);
    __aicore__ inline void Init(uint64_t totalSize, uint64_t weightL1Space, uint64_t aPrefetchSize,
                                const TCubeTiling *__restrict matmulTiling, AscendC::TPipe *tPipe,
                                uint64_t mxBiasL1DbOffset);
    __aicore__ inline void MxA8W4Init(uint64_t l1RemainSize, uint64_t l1StartSize, uint64_t mxBiasL1DbOffset,
                                      const TCubeTiling *__restrict matmulTiling, const LocalTensor<biasType> &biasL1,
                                      uint64_t mxScaleBL1DbOffset,
                                      const LocalTensor<fp8_e8m0_t> &mxScaleBL1);
    __aicore__ inline void LaunchMatmul(const LocalTensor<xType> &weightL1, int64_t kbOffset, uint64_t kbL1RealSize,
                                        uint64_t cvLoopIdx, const BasicBlockOffsetParam &param);
    __aicore__ inline void WaitMTE1ToMTE2(uint64_t kaGmOffset, const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void SetMTE1ToMTE2(uint64_t kaGmOffset, const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void WaitScaleMTE1ToMTE2(uint64_t kbGmOffset);
    __aicore__ inline void SetScaleMTE1ToMTE2(uint64_t kbGmOffset, const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void CopyAAndBiasGmToL1(const BasicBlockOffsetParam &param, int64_t kaGmOffset,
                                              uint64_t cvLoopIdx);
    __aicore__ inline void CopyMxScaleGmToL1(const BasicBlockOffsetParam &param, uint64_t kbL1Offset);
    __aicore__ inline void GetTensorC(const BasicBlockOffsetParam &param);
    __aicore__ inline void GetTensorC(LocalTensor<yType> &yUb);
    __aicore__ inline void EndSync();
    __aicore__ inline void ClearAFullLoadFlag();
    __aicore__ inline void PrefetchA(uint64_t aPrefetchSize, uint64_t xSizeLimit);
private:
    __aicore__ inline void PrefetchA(uint64_t aPrefetchSize, uint64_t aGmSize, const LocalTensor<xType> &perloadBuffer);
    __aicore__ inline void InitSync();
    __aicore__ inline uint64_t CheckMaxSpace(const BasicBlockOffsetParam &param);
    __aicore__ inline void CopyAGmToL1SingleBuffer(const BasicBlockOffsetParam &param, int64_t kaGmOffset,
                                                   int64_t kbL1RealSize, int64_t biasRealN, int64_t aGmOffset);
    __aicore__ inline void ConfigScaleDn2NzParams(uint64_t rowNum, uint64_t scaleKGmSize, uint64_t scaleKL1RealSize,
                                                  Dn2NzParams &dn2NzParams);

    int8_t aL1DbNum_;
    bool isBias_;
    static constexpr uint32_t KB_UNIT = GetKBUnit<xType>();
    static constexpr uint64_t MX_SCALE_L1_SIZE =
        MX_SCALE_L1_SIZE_KB * GetKBUnit<xType>() * sizeof(xType); // scaleA/B单块分配空间

    MatmulImplType mmObj_;

    uint64_t aL1Count_;
    uint64_t aL1MaxHalfCount_;
    uint64_t mxScaleBufIdx_ = 0;
    uint64_t aL1BufIdx_ = 0;

    AscendC::TEventID cubeEventIdsMxScaleMte1ToMte2_[DOUBLE_BUFFER_NUM];
    AscendC::TEventID cubeEventIdsMte1ToMte2_[DOUBLE_BUFFER_NUM];
    AscendC::TEventID cubeEventIdMte2ToMte1_;
    GlobalTensor<xType> xGlobal_;
    GlobalTensor<biasType> biasGlobal_;
    GlobalTensor<fp8_e8m0_t> mxScaleAGlobal_;
    GlobalTensor<yType> yGlobal_;

    LocalTensor<xType> aL1_;
    uint64_t aL1DbOffset_;

    LocalTensor<biasType> biasL1_;
    uint64_t biasL1DbOffset_;

    LocalTensor<fp8_e8m0_t> mxScaleAL1_;
    uint64_t mxScaleAL1DbOffset_;

    LocalTensor<fp8_e8m0_t> mxScaleBL1_;
    uint64_t mxScaleBL1DbOffset_;
};

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline uint64_t WQBMM_CUBE_COMPUTE_CLASS::CheckMaxSpace(const BasicBlockOffsetParam &param)
{
    uint64_t maxSpace = aL1MaxHalfCount_ * param.kbL1Size * CeilAlign(param.mL1Size, static_cast<uint64_t>(BLOCK_CUBE));
    if (param.kbL1Size > 0 && param.kSize % param.kbL1Size == 0 && !wqmmConfig.aTrans && maxSpace <= aL1DbOffset_) {
        return maxSpace;
    }
    return 0;
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::LaunchMatmul(const LocalTensor<xType> &weightL1, int64_t kbOffset,
                                                              uint64_t kbL1RealSize, uint64_t cvLoopIdx,
                                                              const BasicBlockOffsetParam &param)
{
    SetFlag<HardEvent::MTE2_MTE1>(cubeEventIdMte2ToMte1_);
    WaitFlag<HardEvent::MTE2_MTE1>(cubeEventIdMte2ToMte1_);
    uint64_t aL1Offset = 0;
    if (aL1DbNum_ == SINGLE_BUFFER_NUM) {
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
            aL1Offset = (aL1BufIdx_ & 1) * aL1DbOffset_ +
                        CeilAlign(param.mL1Size, static_cast<uint64_t>(BLOCK_CUBE)) *
                            (static_cast<uint64_t>(kbOffset) / (param.kbL1Size * 2) * param.kbL1Size);
        } else {
            aL1Offset = CeilAlign(param.mL1Size, static_cast<uint64_t>(BLOCK_CUBE)) * kbOffset;
        }
    } else {
        aL1Offset = (aL1BufIdx_ & 1) * aL1DbOffset_;
    }
    if constexpr (!IsMxA8W4<xType, wqmmConfig.antiQuantType>()) {
        mmObj_.SetOrgShape(CeilAlign(param.mL1Size, static_cast<uint64_t>(BLOCK_CUBE)),
                        CeilAlign(param.nL1Size, static_cast<uint64_t>(BLOCK_CUBE)),
                        CeilAlign(kbL1RealSize, static_cast<uint64_t>(BLOCK_CUBE)),
                        CeilAlign(kbL1RealSize, static_cast<uint64_t>(BLOCK_CUBE)), param.nSize);
        mmObj_.SetTensorA(aL1_[aL1Offset], wqmmConfig.aTrans);
        mmObj_.SetTensorB(weightL1, wqmmConfig.bTrans);

        if (isBias_) {
            mmObj_.SetBias(biasL1_[(cvLoopIdx & 1) * biasL1DbOffset_]);
        }
        mmObj_.SetTail(param.mL1Size, param.nL1Size, kbL1RealSize);
        mmObj_.Iterate(kbOffset != 0);
    } else {
        BasicApiParamsV1 basicApiParams;
        basicApiParams.l1KbSize = kbL1RealSize;
        basicApiParams.l1KaSize = param.kaL1Size;
        basicApiParams.l0NSize = param.nL1Size;
        basicApiParams.l0MSize = param.mL1Size;
        basicApiParams.isBias = isBias_;
        basicApiParams.gmKOffset = kbOffset;
        mmObj_.Iterate(kbOffset + kbL1RealSize >= param.kSize, kbOffset == 0, aL1_[aL1Offset],
                       mxScaleAL1_[(mxScaleBufIdx_ & 1) * mxScaleAL1DbOffset_], weightL1,
                       mxScaleBL1_[(mxScaleBufIdx_ & 1) * mxScaleBL1DbOffset_],
                       biasL1_[(cvLoopIdx & 1) * biasL1DbOffset_], basicApiParams);
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::WaitMTE1ToMTE2(uint64_t kaGmOffset,
                                                                const BasicBlockOffsetParam &offsetParam)
{
    if (aL1DbNum_ > SINGLE_BUFFER_NUM && kaGmOffset % offsetParam.kaL1Size == 0) {
        WaitFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMte1ToMte2_[aL1BufIdx_ & 1]);
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::WaitScaleMTE1ToMTE2(uint64_t kbGmOffset)
{
    if (kbGmOffset % MX_SCALE_K_L1_SIZE == 0) {
        WaitFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMxScaleMte1ToMte2_[mxScaleBufIdx_ & 1]);
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::SetScaleMTE1ToMTE2(uint64_t kbGmOffset,
                                                                    const BasicBlockOffsetParam &offsetParam)
{
    if ((kbGmOffset + offsetParam.kbL1Size) % MX_SCALE_K_L1_SIZE == 0 ||
        kbGmOffset + offsetParam.kbL1Size >= offsetParam.kSize) {
        SetFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMxScaleMte1ToMte2_[mxScaleBufIdx_ & 1]);
        mxScaleBufIdx_++;
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::SetMTE1ToMTE2(uint64_t kaGmOffset,
                                                               const BasicBlockOffsetParam &offsetParam)
{
    if (aL1DbNum_ > SINGLE_BUFFER_NUM && ((kaGmOffset + offsetParam.kbL1Size) % offsetParam.kaL1Size == 0 ||
                                          kaGmOffset + offsetParam.kbL1Size >= offsetParam.kSize)) {
        SetFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMte1ToMte2_[aL1BufIdx_ & 1]);
        aL1BufIdx_++;
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::CopyAGmToL1SingleBuffer(const BasicBlockOffsetParam &param,
                                                                         int64_t kaGmOffset, int64_t kbL1RealSize,
                                                                         int64_t biasRealN, int64_t aGmOffset)
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
        DataCopy(aL1_[(aL1BufIdx_ & 1) * aL1DbOffset_], xGlobal_[aGmOffset], nd2nzParams);

        nd2nzParams.ndNum = aL1Count_ - aL1MaxHalfCount_;
        DataCopy(aL1_[((aL1BufIdx_ + 1) & 1) * aL1DbOffset_], xGlobal_[aGmOffset + nd2nzParams.dValue], nd2nzParams);
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

        DataCopy(aL1_, xGlobal_[aGmOffset], nd2nzParams);
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::CopyAAndBiasGmToL1(const BasicBlockOffsetParam &param,
                                                                    int64_t kaGmOffset, uint64_t cvLoopIdx)
{
    if (kaGmOffset % param.kaL1Size != 0) {
        return;
    }
    int64_t kaL1RealSize = (kaGmOffset + param.kaL1Size) >= param.kSize ? param.kSize - kaGmOffset : param.kaL1Size;
    int64_t aGmOffset;
    if constexpr (!wqmmConfig.aTrans) {
        aGmOffset = param.mOffset * param.kSize + kaGmOffset;
    } else {
        aGmOffset = kaGmOffset * param.mSize + param.mOffset;
    }

    if (aL1DbNum_ > SINGLE_BUFFER_NUM) {
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        if constexpr (wqmmConfig.aTrans) {
            nd2nzParams.nValue = kaL1RealSize;
            nd2nzParams.dValue = param.mL1Size;
            nd2nzParams.srcDValue = param.mSize;
        } else {
            nd2nzParams.nValue = param.mL1Size;
            nd2nzParams.dValue = kaL1RealSize;
            nd2nzParams.srcDValue = param.kSize;
        }
        nd2nzParams.srcNdMatrixStride = 0;
        nd2nzParams.dstNzC0Stride = CeilAlign(nd2nzParams.nValue, static_cast<uint16_t>(BLOCK_CUBE));
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 0;

        DataCopy(aL1_[(aL1BufIdx_ & 1) * aL1DbOffset_], xGlobal_[aGmOffset], nd2nzParams);
    } else if (aL1DbNum_ == SINGLE_BUFFER_NUM && kaGmOffset == 0) {
        CopyAGmToL1SingleBuffer(param, kaGmOffset, kaL1RealSize, param.nL1Size, aGmOffset);
    }

    // bias仅与n有关，与k无关，所以只需要拷贝一次
    if constexpr (!IsMxA8W4<xType, wqmmConfig.antiQuantType>()) {
        if (isBias_ && kaGmOffset == 0) {
            DataCopyPad2D(biasL1_[(cvLoopIdx & 1) * biasL1DbOffset_], biasGlobal_[param.nOffset], 1, param.nL1Size,
                          param.nL1Size, param.nL1Size);
        }
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::CopyMxScaleGmToL1(const BasicBlockOffsetParam &param,
                                                                   uint64_t kbL1Offset)
{
    if (kbL1Offset % MX_SCALE_K_L1_SIZE != 0) {
        return;
    }
    uint64_t scaleKGmSize = param.kSize / MX_GROUPSIZE;
    // 当前scaleFactor为1，暂不考虑scaleFactor相关计算
    uint64_t scaleKL1StandardLen = MX_SCALE_K_L1_SIZE / MX_GROUPSIZE;
    uint64_t scaleKL1RealSize = (kbL1Offset + MX_SCALE_K_L1_SIZE) > param.kSize ?
                                    (param.kSize - kbL1Offset) / MX_GROUPSIZE :
                                    scaleKL1StandardLen;

    // copy mxScaleA
    Dn2NzParams scaleAdn2NzParams;
    ConfigScaleDn2NzParams(param.mL1Size, scaleKGmSize, scaleKL1RealSize, scaleAdn2NzParams);

    int64_t scaleAGmOffset = param.mOffset * scaleKGmSize + kbL1Offset / MX_GROUPSIZE;
    GlobalTensor<half> f16ScaleAGlobal;
    f16ScaleAGlobal.SetGlobalBuffer((__gm__ half *)mxScaleAGlobal_[scaleAGmOffset].GetPhyAddr(),
                                    (param.mL1Size * scaleKL1RealSize) >> 1);
    auto f16ScaleALocal = mxScaleAL1_[(mxScaleBufIdx_ & 1) * mxScaleAL1DbOffset_].template ReinterpretCast<half>();

    DataCopy(f16ScaleALocal, f16ScaleAGlobal, scaleAdn2NzParams);
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::ConfigScaleDn2NzParams(uint64_t rowNum, uint64_t scaleKGmSize,
                                                                        uint64_t scaleKL1RealSize,
                                                                        Dn2NzParams &dn2NzParams)
{
    dn2NzParams.dnNum = 1;
    dn2NzParams.dValue = rowNum;  // 矩阵的行数，即待搬运的mxScaleA的m或mxScaleB的n
    dn2NzParams.nValue = CeilDivide(scaleKL1RealSize, SCALE_COPY_GROUP_SIZE);  // 矩阵的列数，使用B16搬B8需要除以2向上取整
    dn2NzParams.srcDnMatrixStride = SCALE_COPY_DEFAULT_STRIDE;
    dn2NzParams.srcDValue = CeilDivide(scaleKGmSize, SCALE_COPY_GROUP_SIZE);  // 源矩阵一行所含B16元素个数
    // 目标矩阵行方向两个相邻分形起始地址之间的间隔，单位32B
    dn2NzParams.dstNzC0Stride = CeilDivide(MX_SCALE_K_L1_SIZE, MX_GROUPSIZE * SCALE_COPY_GROUP_SIZE);
    // 目标矩阵列方向两个相邻分形起始地址之间的间隔，单位32B
    dn2NzParams.dstNzNStride = SCALE_COPY_DEFAULT_N_STRIDE;
    dn2NzParams.dstNzMatrixStride = SCALE_COPY_DEFAULT_STRIDE;
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::EndSync()
{
    if constexpr (IsMxA8W4<xType, wqmmConfig.antiQuantType>()) {
        mmObj_.End();
    }

    for (uint64_t i = 0; i < DOUBLE_BUFFER_NUM; i++) {
        WaitFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMxScaleMte1ToMte2_[i]);
        if (aL1DbNum_ > SINGLE_BUFFER_NUM) {
            WaitFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMte1ToMte2_[i]);
        }
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(cubeEventIdsMte1ToMte2_[i]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(cubeEventIdsMxScaleMte1ToMte2_[i]);
    }
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE1>(cubeEventIdMte2ToMte1_);
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::ClearAFullLoadFlag()
{
    if (aL1DbNum_ == SINGLE_BUFFER_NUM) {
        SetFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMte1ToMte2_[0]);
        WaitFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMte1ToMte2_[0]);
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::InitSync()
{
    for (uint64_t i = 0; i < DOUBLE_BUFFER_NUM; i++) {
        cubeEventIdsMte1ToMte2_[i] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
        cubeEventIdsMxScaleMte1ToMte2_[i] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
        SetFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMxScaleMte1ToMte2_[i]);
        if (aL1DbNum_ > SINGLE_BUFFER_NUM) {
            SetFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMte1ToMte2_[i]);
        }
    }
    cubeEventIdMte2ToMte1_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>();
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::UpdateGlobalAddr(
    __gm__ xType *x, __gm__ yType *y, __gm__ biasType *bias, __gm__ antiQuantScaleType *antiquantScale,
    __gm__ uint64_t *quantScale, __gm__ perTokenScaleType *perTokenScale, const bool isBias)
{
    isBias_ = isBias;
    xGlobal_.SetGlobalBuffer(x);
    yGlobal_.SetGlobalBuffer(y);
    if (isBias_) {
        biasGlobal_.SetGlobalBuffer(bias);
    }

    if constexpr (IsMxA8W4<xType, wqmmConfig.antiQuantType>()) {
        mxScaleAGlobal_.SetGlobalBuffer(perTokenScale);
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::PrefetchA(uint64_t aPrefetchSize, uint64_t xSizeLimit,
                                                           const LocalTensor<xType> &perloadBuffer)
{
    uint64_t xOffset = GetBlockIdx() * aPrefetchSize;
    if (aPrefetchSize == 0 || xOffset >= xSizeLimit) {
        return;
    }
    DataCopyPadExtParams<xType> extParams;
    DataCopyExtParams param;
    param.blockCount = 1;
    param.blockLen = (xOffset + aPrefetchSize > xSizeLimit ? xSizeLimit - xOffset : aPrefetchSize) * sizeof(xType);
    param.srcStride = 0;
    param.dstStride = 0;
    DataCopyPad(perloadBuffer, xGlobal_[xOffset], param, extParams);
    PipeBarrier<PIPE_MTE2>();
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::PrefetchA(uint64_t aPrefetchSize, uint64_t xSizeLimit)
{
    uint64_t xOffset = GetBlockIdx() * aPrefetchSize;
    if (aPrefetchSize == 0 || xOffset >= xSizeLimit) {
        return;
    }
    DataCopyExtParams param;
    param.blockCount = 1;
    param.blockLen = (xOffset + aPrefetchSize > xSizeLimit ? xSizeLimit - xOffset : aPrefetchSize) * sizeof(xType);
    param.srcStride = 0;
    param.dstStride = 0;

    event_t eventIdMTE1ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE1_MTE2>());
    SetFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2);
    WaitFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2);

    DataCopyPadExtParams<xType> extParams;
    DataCopyPad(aL1_, xGlobal_[xOffset], param, extParams);
    PipeBarrier<PIPE_MTE2>();
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::MxA8W4Init(uint64_t l1RemainSize, uint64_t l1StartSize,
                                                            uint64_t mxBiasL1DbOffset,
                                                            const TCubeTiling *__restrict matmulTiling,
                                                            const LocalTensor<biasType> &biasL1,
                                                            uint64_t mxScaleBL1DbOffset,
                                                            const LocalTensor<fp8_e8m0_t> &mxScaleBL1)
{
    //  MxA8W4场景空间分配: 其中 weight\Bias为全局分配，此处不感知
    //  L1 (0~512KB): WeightL1_P0(64KB) |    Bias_P0(4KB)   | ScaleAL1_P0(32KB) | ScaleBL1_P0(32KB) | AL1_P0(80KB) |
    //              | AL1_P1(80KB)     | ScaleBL1_P1(32KB) | ScaleAL1_P1(32KB) | Bias_P1(4KB)      | WeightL1_P1(64KB)
    biasL1_ = biasL1; // 预计大小4Kb
    biasL1DbOffset_ = mxBiasL1DbOffset;
    aL1Count_ = 0;        // GMM 动态场景暂时关闭A分段全载策略
    aL1MaxHalfCount_ = 0; // GMM 动态场景暂时关闭A分段全载策略
    aL1DbNum_ = DOUBLE_BUFFER_NUM;

    mxScaleAL1_ = LocalTensor<fp8_e8m0_t>(TPosition::TSCM, l1StartSize, l1RemainSize / sizeof(fp8_e8m0_t));
    mxScaleAL1DbOffset_ = l1RemainSize - MX_SCALE_L1_SIZE;
    l1RemainSize -= DOUBLE_BUFFER_NUM * MX_SCALE_L1_SIZE;
    l1StartSize += MX_SCALE_L1_SIZE;

    mxScaleBL1_ = mxScaleBL1;
    mxScaleBL1DbOffset_ = mxScaleBL1DbOffset;
    l1RemainSize -= DOUBLE_BUFFER_NUM * MX_SCALE_L1_SIZE;
    l1StartSize += MX_SCALE_L1_SIZE;

    aL1_ = LocalTensor<xType>(TPosition::TSCM, l1StartSize, l1RemainSize / sizeof(xType));
    aL1DbOffset_ = l1RemainSize - MX_A8W4_A_L1_RESERVED_KB * 1024;

    mmObj_.Init();
    InitSync();
}

// 场景1： 使能a prefetch。必须先更新地址再init
// 场景2： gm地址变化需要实时获取场景，必须先init再更新地址
WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::Init(uint64_t totalSize, uint64_t weightL1Space,
                                                      uint64_t aPrefetchSize,
                                                      const TCubeTiling *__restrict matmulTiling, AscendC::TPipe *tPipe,
                                                      uint64_t mxBiasL1DbOffset)
{
    // (1) L1 上有bias时:
    //  L1 (0~512KB): WeightL1_P0(128KB) | Bias_P0(4KB) | AL1_P0(124KB) | AL1_P1(124KB) | Bias_P1(4KB) |
    //  WeightL1_P1(128KB)
    // (2) 其他场景时
    //  L1 (0~512KB): WeightL1_P0(128KB) | AL1_P0(128KB) | AL1_P1(128KB) | WeightL1_P1(128KB)
    uint64_t biasL1Space = matmulTiling->isBias ? BIAS_L1_SIZE * KB_UNIT : 0; // bias单块分配4K空间
    uint64_t aL1Offset = weightL1Space + biasL1Space;                         // A要跳过WeightL1_P0 + Bias_P0
    if (matmulTiling->isBias) {
        uint64_t aL1Space = L1_SIZE * KB_UNIT - DOUBLE_BUFFER_NUM * aL1Offset; // L1上A可占据剩余空间
        aL1DbOffset_ = aL1Space >> 1;
        if constexpr (IsSameType<biasType, float>::value) {
            biasL1_ = LocalTensor<biasType>(TPosition::TSCM, (weightL1Space >> 1) * sizeof(biasType),
                                            totalSize / sizeof(biasType) - (weightL1Space >> 1));
            biasL1DbOffset_ = (aL1Space + biasL1Space) >> 1;
        } else {
            biasL1_ = LocalTensor<biasType>(TPosition::TSCM, weightL1Space * sizeof(biasType),
                                            totalSize / sizeof(biasType) - weightL1Space);
            biasL1DbOffset_ = aL1Space + biasL1Space;
        }
    } else {
        aL1DbOffset_ = L1_HALF_SIZE * KB_UNIT - weightL1Space;
    }
    aL1_ = LocalTensor<xType>(TPosition::TSCM, aL1Offset * sizeof(xType), totalSize / sizeof(xType) - aL1Offset);
    aL1Count_ = matmulTiling->Ka / (matmulTiling->baseK * matmulTiling->stepKb);
    aL1MaxHalfCount_ = CeilDivide(aL1Count_, static_cast<uint64_t>(DOUBLE_BUFFER_NUM));
    PrefetchA(aPrefetchSize, matmulTiling->M * matmulTiling->Ka, aL1_);

    // 当前tiling策略的细分场景：
    // 1. stepKa <= stepKb 当前限制baseM的最大值，因此该场景下在L1上A矩阵大小<=128k。可以固定走db分支，保证a矩阵的db载入
    // 2. stepKa > stepKb 当前在m小k大的情况下才会出现该场景，走全载分支
    // 3. gmm场景，不知道真实的m值，tiling采取保守策略，恒定走db分支
    if (matmulTiling->stepKa > matmulTiling->stepKb) {
        aL1DbNum_ = SINGLE_BUFFER_NUM;
    } else {
        aL1DbNum_ = DOUBLE_BUFFER_NUM;
    }
    mmObj_.SetSubBlockIdx(0);
    mmObj_.Init(matmulTiling, tPipe);
    InitSync();
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::GetTensorC(const BasicBlockOffsetParam &param)
{
    uint64_t outOffset = param.mOffset * param.nSize + param.nOffset;
#ifndef __CCE_KT_TEST__
    if constexpr (!IsMxA8W4<xType, wqmmConfig.antiQuantType>()) {
        mmObj_.GetTensorC(yGlobal_[outOffset]);
    } else {
        mmObj_.GetTensorC(param.mL1Size, param.nL1Size, param.nSize, yGlobal_[outOffset]);
    }
#endif
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::GetTensorC(LocalTensor<yType> &yUb)
{
#ifndef __CCE_KT_TEST__
    mmObj_.GetTensorC(yUb, 0, true);
#endif
}
}  // namespace WeightQuantBatchMatmulV2::Arch35
#endif
