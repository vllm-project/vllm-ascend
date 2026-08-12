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
 * \file weight_quant_batch_matmul_v2_common.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCHMATMUL_V2_COMMON_H
#define WEIGHT_QUANT_BATCHMATMUL_V2_COMMON_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "weight_quant_batch_matmul_v2_constant.h"
#include "anti_quant.h"

using AscendC::AIC;
using AscendC::AIV;
using AscendC::BinaryRepeatParams;
using AscendC::BLOCK_CUBE;
using AscendC::BrcbRepeatParams;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPadParams;
using AscendC::DataCopyParams;
using AscendC::DataFormat;
using AscendC::DEFAULT_BLK_STRIDE;
using AscendC::DEFAULT_REPEAT_STRIDE;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::int4b_t;
using AscendC::LocalTensor;
using AscendC::MAX_REPEAT_TIMES;
using AscendC::ONE_BLK_FLOAT_NUM;
using AscendC::ONE_BLK_SIZE;
using AscendC::ONE_FOURTH_DEFAULT_REPEAT_STRIDE;
using AscendC::QuePosition;
using AscendC::RoundMode;
using AscendC::ShapeInfo;
using AscendC::SyncAll;
using AscendC::TBuf;
using AscendC::ToFloat;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TQue;
using AscendC::UnaryRepeatParams;

using matmul::MatmulImpl;
using matmul::MatmulType;

namespace Mc2WeightQuantBatchMatmulV2 {
struct AntiquantLoopParams {
    uint8_t repeatRemain = 0;
    uint8_t maskRemain = 0;
    uint16_t repeatLoop = 0;
    uint16_t maskLoop = 0;

    // 保留字段，group size较大的情况下，repeatStride可能超出限制，此时一次只能作一行，规避repeatStride的限制
    uint16_t repeatStrideLoop = 0;
};

struct WeightSplitInfo {
    uint16_t vecSingleN = 0;
    uint16_t vecSingleK = 0;
    uint32_t splitNOffset = 0;
    uint32_t originNOffset = 0;
    uint32_t kOffset = 0;
    uint32_t vecNzSingleN = 0;
    uint32_t vecNzSingleK = 0;
    uint32_t vecNzNRealSize = 0;
};

template <class T>
struct Mc2AntiQuantCalType {
    using type = float;
};

template <>
struct Mc2AntiQuantCalType<half> {
    using type = half;
};

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
class Mc2WeightQuantBatchMatmulV2Common {
public:
    __aicore__ inline Mc2WeightQuantBatchMatmulV2Common(){};

protected:
    using antiQuantCalType = typename Mc2AntiQuantCalType<biasType>::type;
    __aicore__ inline void BaseInit(const Mc2WeightQuantBatchMatmulV2TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void InitInput(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                     GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y);
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void InitBuffer();
    __aicore__ inline int64_t ComputeAntiquantShape();
    __aicore__ inline void WeightCopyOut(WeightSplitInfo &weightSplitInfo);
    __aicore__ inline void CopyInAntiquantParams(WeightSplitInfo &weightSplitInfo);
    __aicore__ inline void SetAntiquantCopyInParams(uint64_t &antiquantOffset, WeightSplitInfo &weightSplitInfo);
    __aicore__ inline void SetAntiquantTensorShape(WeightSplitInfo &weightSplitInfo);
    __aicore__ inline void SetAntiquantTensorShapeWithGroup(WeightSplitInfo &weightSplitInfo);
    __aicore__ inline void BroadCastAntiquantParams(LocalTensor<antiQuantCalType> &dstTensor,
                                                    LocalTensor<xType> &srcTensor);
    __aicore__ inline void AntiQuantCompute(WeightSplitInfo &weightSplitInfo);
    __aicore__ inline void WeightCopyInAndCast(WeightSplitInfo &weightSplitInfo);
    __aicore__ inline void AntiquantWeight(uint64_t cubeNLoopIdx, uint64_t nBaseOffset, uint64_t nRealSize,
                                           uint64_t nLoopLimit);
    __aicore__ inline void InitWorkSpace(GM_ADDR workspace);
    __aicore__ inline void AntiQuantAdd(const LocalTensor<antiQuantCalType> &srcTensor,
                                        const LocalTensor<antiQuantCalType> &offsetComputeBuf,
                                        const AntiquantLoopParams &antiquantLoopParams,
                                        const BinaryRepeatParams &repeatParams);
    __aicore__ inline void AntiQuantAddAtMask(const LocalTensor<antiQuantCalType> &srcTensor,
                                              const LocalTensor<antiQuantCalType> &offsetComputeBuf, int32_t repeat,
                                              const AntiquantLoopParams &antiquantLoopParams,
                                              const BinaryRepeatParams &repeatParams);
    __aicore__ inline void AntiQuantMul(const LocalTensor<antiQuantCalType> &srcTensor,
                                        const LocalTensor<antiQuantCalType> &scaleComputeBuf,
                                        const AntiquantLoopParams &antiquantLoopParams,
                                        const BinaryRepeatParams &repeatParams);
    __aicore__ inline void AntiQuantMulAtMask(const LocalTensor<antiQuantCalType> &srcTensor,
                                              const LocalTensor<antiQuantCalType> &scaleComputeBuf, int32_t repeat,
                                              const AntiquantLoopParams &antiquantLoopParams,
                                              const BinaryRepeatParams &repeatParams);
    __aicore__ inline void NotifyCube()
    {
        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
    }
    __aicore__ inline void NotifyVector()
    {
        CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_AIV_FLAG);
    }
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilDiv(T1 a, T2 b)
    {
        if (b == 0) {
            return 0;
        }
        return (a + b - 1) / b;
    };
    __aicore__ inline void WaitFlagDevLocal(int64_t flagID)
    {
        CrossCoreWaitFlag(flagID);
    }
    __aicore__ inline void WaitForVector()
    {
        WaitFlagDevLocal(SYNC_AIV_AIC_FLAG);
    }
    __aicore__ inline void WaitForCube()
    {
        WaitFlagDevLocal(SYNC_AIC_AIV_FLAG);
    }

protected:
    static constexpr uint64_t SYNC_MODE0 = 0;
    static constexpr uint64_t SYNC_MODE2 = 2;
    static constexpr uint64_t SYNC_AIV_ONLY_ALL_FLAG = 6;
    static constexpr uint64_t SYNC_AIC_ONLY_ALL_FLAG = 7;
    static constexpr uint64_t SYNC_AIV_AIC_FLAG = 8;
    static constexpr uint64_t SYNC_AIC_AIV_FLAG = 9;

    static constexpr int32_t DOUBLE_BUFFER_NUM = 2;
    static constexpr int32_t SINGLE_BUFFER_NUM = 1;
    static constexpr int32_t WEIGHT_CACHE_COUNT = 3;
    static constexpr int32_t FP32_MASKMAX = 256 / sizeof(float);
    static constexpr int32_t FP16_MASKMAX = 256 / sizeof(half);
    static constexpr uint32_t FP32_ALIGN_SIZE = (ONE_BLK_SIZE / sizeof(float));
    static constexpr uint32_t FP16_ALIGN_SIZE = (ONE_BLK_SIZE / sizeof(half));
    static constexpr uint32_t FRACTAL_SIZE_F16 = 256;
    TPipe *pipe_;
    const Mc2WeightQuantBatchMatmulV2TilingData *tiling_;

    bool biasFlag_ = false;

    int32_t curBlockIdx_;
    int32_t cubeNDimIdx_;
    int32_t cubeMDimIdx_;
    int32_t vecKDimIdx_;
    int32_t vecNDimIdx_;
    int32_t groupNum_;
    int32_t maskMax_;
    int32_t oneBlockAlignSize_;
    int32_t weightInnerAxisAlignSize_;
    uint64_t weightCacheSizeAlign_;
    uint64_t weightCacheIdx_;
    uint64_t kCacheAlignSize_;
    uint64_t cubeBaseN_;
    int32_t broadCastFactor_ = 1;
    int32_t wFormat_ = static_cast<int32_t>(CubeFormat::ND);
    antiQuantCalType scaleValue_ = 0;
    antiQuantCalType offsetValue_ = 0;

    BrcbRepeatParams brcbParams_;
    DataCopyParams antiquantCopyinParams_;
    DataCopyPadParams antiquantCopyinPadParams_;

    GlobalTensor<xType> xGlobal_;
    GlobalTensor<xType> weightCache_;
    GlobalTensor<wType> wGlobal_;
    GlobalTensor<xType> offsetGlobal_;
    GlobalTensor<xType> scaleGlobal_;
    GlobalTensor<biasType> biasGlobal_;
    GlobalTensor<yType> yGlobal_;
    GlobalTensor<uint64_t> quantScaleGlobal_;

    LocalTensor<antiQuantCalType> offsetComputeTensor_;
    LocalTensor<antiQuantCalType> scaleComputeTensor_;

    Mc2AntiQuantTensorShape tensorShape_;

    /*
    A矩阵类型关联的buffer分配方式(包括dtype, 以w8为例):
        +-------------------------+----------+-----------+
        |                         | bf16     | fp16      |
        +------------------------------------------------+
        |offsetQueue_             | 2*n*Gc   | 2*n*Gc    |
        +------------------------------------------------+
        |scaleQueue_              | 2*n*Gc   | 2*n*Gc    |
        +------------------------------------------------+
        |originWeightQueue_       | n*k      | n*k       |
        +------------------------------------------------+
        |weightOutputQueue_       | 2*2*n*k  | 2*2*n*k   |
        +------------------------------------------------+
        |antiquantParamsFp32TBuf_ | 4*n*Gc   | 0         |
        +------------------------------------------------+
        |offsetComputeTbuf_       | 4*n*Gc*8 | 2*n*Gc*16 |
        +------------------------------------------------+
        |scaleComputeTbuf_        | 4*n*Gc*8 | 2*n*Gc*16 |
        +------------------------------------------------+
        |weight16Tbuf_            | 2*n*k    | 2*n*k     |
        +------------------------------------------------+
        |weight32Tbuf_            | 4*n*k    | 0         |
        +-------------------------+----------+-----------+
    其中，
    Gc = CeilDiv(k, groupSize)，per tensor/ per channel场景下 Gc = 1
    w4场景下，仅originWeightQueue_的分配空间变小
    */
    TQue<QuePosition::VECIN, SINGLE_BUFFER_NUM> offsetQueue_;
    TQue<QuePosition::VECIN, SINGLE_BUFFER_NUM> scaleQueue_;
    TQue<QuePosition::VECIN, SINGLE_BUFFER_NUM> originWeightQueue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> weightOutputQueue_;

    TBuf<> antiquantParamsFp32TBuf_;
    TBuf<> offsetComputeTbuf_;
    TBuf<> scaleComputeTbuf_;
    TBuf<> weight16Tbuf_;
    TBuf<> weight32Tbuf_;
    TBuf<> sharedTmpBuffer_;
};

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2Common<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
                                  quantType>::BaseInit(const Mc2WeightQuantBatchMatmulV2TilingData *tilingData,
                                                       TPipe *tPipe)
{
    tiling_ = tilingData;
    pipe_ = tPipe;
    curBlockIdx_ = GetBlockIdx();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2Common<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
                                  quantType>::InitWorkSpace(GM_ADDR workspace)
{
    weightCache_.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(workspace));
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2Common<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                         hasAntiQuantOffset, quantType>::ComputeConstexpr()
{
    cubeNDimIdx_ = curBlockIdx_ % tiling_->cubeBlockDimN;
    cubeMDimIdx_ = curBlockIdx_ / tiling_->cubeBlockDimN;
    vecKDimIdx_ = curBlockIdx_ / tiling_->vecBlockDimN;
    vecNDimIdx_ = curBlockIdx_ % tiling_->vecBlockDimN;

    cubeBaseN_ = tiling_->matmulTiling.singleCoreN * tiling_->cubeBlockDimN;
    kCacheAlignSize_ = tiling_->matmulTiling.Kb;
    uint64_t weightCacheSize =
        static_cast<uint64_t>(tiling_->matmulTiling.singleCoreN) * tiling_->cubeBlockDimN * tiling_->kAlign;
    if constexpr (bTrans) {
        weightCacheSize = tiling_->matmulTiling.singleCoreN * tiling_->cubeBlockDimN * kCacheAlignSize_;
    }
    // 向256对齐，可以保证workspace起始地址保证512B对齐，提升mte3性能
    weightCacheSizeAlign_ = CeilDiv(weightCacheSize, 256) * 256;

    // brcb接口只支持按照Block处理源数据。antiquant操作在目的操作上不需要跳写，不同block间地址步长为1个block
    brcbParams_.dstBlkStride = 1;
    brcbParams_.dstRepStride = 8;

    if constexpr (IsSameType<xType, bfloat16_t>::value) {
        oneBlockAlignSize_ = FP32_ALIGN_SIZE;
        maskMax_ = FP32_MASKMAX;
    } else {
        oneBlockAlignSize_ = FP16_ALIGN_SIZE;
        maskMax_ = FP16_MASKMAX;
    }

    if constexpr (IsSameType<wType, int4b_t>::value) {
        // int4场景, 内轴shape按照64对齐
        weightInnerAxisAlignSize_ = ONE_BLK_SIZE * 2;
    } else {
        weightInnerAxisAlignSize_ = ONE_BLK_SIZE / sizeof(wType);
    }

    // antiquant参数搬运进ub时不需要跳写
    antiquantCopyinParams_.dstStride = 0;
    // per channel/per tensor场景，可以近似等价为groupNum_为1的per group场景，因此可以赋默认值1
    groupNum_ = 1;

    if constexpr (antiQuantType == Mc2QuantType::PER_GROUP) {
        groupNum_ = CeilDiv(tiling_->kSize, tiling_->groupSize);
        if constexpr (bTrans) {
            if (tiling_->vecSingleK >= tiling_->kSize) {
                // 全载场景，所有有效数据视为一行，src尾块在最后
                antiquantCopyinParams_.blockCount = 1;
            }
        }
    } else {
        // antiquant参数每次搬运进UB都是连续数据，因此blockCount为1, blockLen为实际的长度
        antiquantCopyinParams_.blockCount = 1;
        antiquantCopyinParams_.srcStride = 0;
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2Common<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
                                  quantType>::InitInput(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale,
                                                        GM_ADDR antiquantOffset, GM_ADDR quantScale,
                                                        GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y)
{
    xGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(x), tiling_->matmulTiling.M * tiling_->matmulTiling.Ka);
    wGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ wType *>(weight),
                             tiling_->matmulTiling.Kb * tiling_->matmulTiling.N);
    yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ yType *>(y), tiling_->matmulTiling.M * tiling_->matmulTiling.N);
    biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ biasType *>(bias), tiling_->matmulTiling.N);
    biasFlag_ = static_cast<bool>(tiling_->matmulTiling.isBias);
    offsetGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(antiquantOffset), tiling_->matmulTiling.N);
    scaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(antiquantScale), tiling_->matmulTiling.N);
    quantScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(quantScale), tiling_->matmulTiling.N);

    if constexpr (antiQuantType == Mc2QuantType::PER_TENSOR) {
        if constexpr (IsSameType<xType, bfloat16_t>::value) {
            scaleValue_ = ToFloat(scaleGlobal_.GetValue(0));
            if constexpr (hasAntiQuantOffset) {
                offsetValue_ = ToFloat(offsetGlobal_.GetValue(0));
            }
        } else {
            scaleValue_ = scaleGlobal_.GetValue(0);
            if constexpr (hasAntiQuantOffset) {
                offsetValue_ = offsetGlobal_.GetValue(0);
            }
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2Common<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                         hasAntiQuantOffset, quantType>::InitBuffer()
{
    // offset/scale的shape需要按照32Byte对齐看
    int64_t antiquantShape = ComputeAntiquantShape();

    // gm读取的是xType类型
    int64_t antiquantOriSize = antiquantShape * sizeof(xType);
    pipe_->InitBuffer(offsetQueue_, SINGLE_BUFFER_NUM, antiquantOriSize);
    pipe_->InitBuffer(scaleQueue_, SINGLE_BUFFER_NUM, antiquantOriSize);

    int64_t antiquantBroadCastSize = antiquantShape * sizeof(antiQuantCalType);
    // 需要broadcast的场景：
    // (1) ND的转置场景[包括per-group和per-channel] (2) Nz的per-channel场景 (3) Nz的per-group转置场景
    if ((bTrans && wFormat_ == static_cast<int32_t>(CubeFormat::ND)) ||
        (wFormat_ == static_cast<int32_t>(CubeFormat::NZ) && (antiQuantType == Mc2QuantType::PER_CHANNEL || bTrans))) {
        // 尾轴broadCast至32Byte对齐；Nz场景16元素对齐
        antiquantBroadCastSize = antiquantShape * ONE_BLK_SIZE * broadCastFactor_;
    }
    pipe_->InitBuffer(offsetComputeTbuf_, antiquantBroadCastSize);
    pipe_->InitBuffer(scaleComputeTbuf_, antiquantBroadCastSize);

    // weight输入的shape是n*k
    int64_t weightShape;
    if constexpr (bTrans) {
        weightShape =
            tiling_->vecSingleN * CeilDiv(tiling_->vecSingleK, weightInnerAxisAlignSize_) * weightInnerAxisAlignSize_;
    } else {
        weightShape =
            tiling_->vecSingleK * CeilDiv(tiling_->vecSingleN, weightInnerAxisAlignSize_) * weightInnerAxisAlignSize_;
    }
    int64_t originWeightSize = weightShape * sizeof(wType);
    if constexpr (IsSameType<wType, int4b_t>::value) {
        originWeightSize = originWeightSize >> 1;
    }
    int64_t weight16Size = weightShape * sizeof(xType);

    pipe_->InitBuffer(originWeightQueue_, SINGLE_BUFFER_NUM, originWeightSize);
    pipe_->InitBuffer(weightOutputQueue_, DOUBLE_BUFFER_NUM, weight16Size);
    pipe_->InitBuffer(weight16Tbuf_, weight16Size);

    if constexpr (IsSameType<xType, bfloat16_t>::value) {
        if constexpr (bTrans) {
            // bf16+weight转置场景下，运算需要Fp32格式的中间结果
            int64_t antiquantFp32Size = antiquantShape * sizeof(float);
            pipe_->InitBuffer(antiquantParamsFp32TBuf_, antiquantFp32Size);
        }

        int64_t weight32Size = weightShape * sizeof(float);
        pipe_->InitBuffer(weight32Tbuf_, weight32Size);
    }

    offsetComputeTensor_ = offsetComputeTbuf_.Get<antiQuantCalType>();
    scaleComputeTensor_ = scaleComputeTbuf_.Get<antiQuantCalType>();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline int64_t
Mc2WeightQuantBatchMatmulV2Common<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
                                  quantType>::ComputeAntiquantShape()
{
    if constexpr (antiQuantType != Mc2QuantType::PER_GROUP) {
        return CeilDiv(tiling_->vecSingleN * groupNum_, ONE_BLK_SIZE / sizeof(xType)) * (ONE_BLK_SIZE / sizeof(xType));
    } else if constexpr (bTrans) {
        // 全载场景
        if (tiling_->vecSingleK >= tiling_->kSize) {
            return CeilDiv(tiling_->vecSingleN * groupNum_, ONE_BLK_SIZE / sizeof(xType)) *
                   (ONE_BLK_SIZE / sizeof(xType));
        } else {
            // 不全载场景
            return tiling_->vecSingleN *
                   CeilDiv(CeilDiv(tiling_->vecSingleK, tiling_->groupSize), ONE_BLK_SIZE / sizeof(xType)) *
                   (ONE_BLK_SIZE / sizeof(xType));
        }
    } else {
        // weigh不转置场景，不区分k是否全载
        return CeilDiv(tiling_->vecSingleK, tiling_->groupSize) *
               CeilDiv(tiling_->vecSingleN, ONE_BLK_SIZE / sizeof(xType)) * (ONE_BLK_SIZE / sizeof(xType));
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2Common<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
                                  quantType>::AntiquantWeight(uint64_t cubeNLoopIdx, uint64_t nBaseOffset,
                                                              uint64_t nRealSize, uint64_t nLoopLimit)
{
    if (vecKDimIdx_ >= tiling_->vecBlockDimK) {
        // 当前核不需要处理数据
        return;
    }

    // 求解当次循环n方向的起点和终点
    uint64_t nSingleCoreLoop = CeilDiv(nLoopLimit, tiling_->vecBlockDimN);
    uint64_t nSingleCoreLoopStart = vecNDimIdx_ * nSingleCoreLoop;
    uint64_t nSingleCoreLoopLimit = nSingleCoreLoopStart + nSingleCoreLoop;
    if (nSingleCoreLoopLimit > nLoopLimit) {
        nSingleCoreLoopLimit = nLoopLimit;
    }

    // 为节约workspace空间，weight缓存空间整体分成WEIGHT_CACHE_COUNT块轮询使用，需要根据cubeNLoopIdx确定当前使用哪块地址
    weightCacheIdx_ = cubeNLoopIdx % WEIGHT_CACHE_COUNT;
    WeightSplitInfo weightSplitInfo;
    weightSplitInfo.vecSingleN = tiling_->vecSingleN;
    for (uint64_t nLoopIdx = nSingleCoreLoopStart; nLoopIdx < nSingleCoreLoopLimit; nLoopIdx++) {
        weightSplitInfo.splitNOffset = nLoopIdx * tiling_->vecSingleN;
        weightSplitInfo.originNOffset = nBaseOffset + weightSplitInfo.splitNOffset;
        if (unlikely(nLoopIdx == nLoopLimit - 1)) {
            // 当计算到最后一块时，需要重新计算尾块实际的n是多少
            weightSplitInfo.vecSingleN = nRealSize - weightSplitInfo.splitNOffset;
        }

        if constexpr (antiQuantType == Mc2QuantType::PER_CHANNEL) {
            CopyInAntiquantParams(weightSplitInfo);
        }

        // k方向按照分核数量直接排列到上限即可
        weightSplitInfo.vecSingleK = tiling_->vecSingleK;
        for (uint64_t vecKLoopIdx = vecKDimIdx_; vecKLoopIdx < tiling_->vecSingleKLoop;
             vecKLoopIdx += tiling_->vecBlockDimK) {
            weightSplitInfo.kOffset = vecKLoopIdx * tiling_->vecSingleK;
            // 当计算到最后一块时，需要重新计算尾块实际的k是多少
            if (unlikely(vecKLoopIdx == tiling_->vecSingleKLoop - 1)) {
                weightSplitInfo.vecSingleK = tiling_->kSize - weightSplitInfo.kOffset;
            }

            if constexpr (antiQuantType == Mc2QuantType::PER_GROUP) {
                CopyInAntiquantParams(weightSplitInfo);
            }

            WeightCopyInAndCast(weightSplitInfo);
            AntiQuantCompute(weightSplitInfo);
            WeightCopyOut(weightSplitInfo);
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2Common<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
                                  quantType>::CopyInAntiquantParams(WeightSplitInfo &weightSplitInfo)
{
    if constexpr (antiQuantType == Mc2QuantType::PER_TENSOR) {
        // pertensor场景直接return
        return;
    }
    uint64_t antiquantOffset = weightSplitInfo.originNOffset;
    SetAntiquantCopyInParams(antiquantOffset, weightSplitInfo);

    if constexpr (hasAntiQuantOffset) {
        LocalTensor<xType> offsetInput = offsetQueue_.AllocTensor<xType>();
        DataCopyPad(offsetInput, offsetGlobal_[antiquantOffset], antiquantCopyinParams_, antiquantCopyinPadParams_);
        offsetQueue_.EnQue(offsetInput);
        offsetInput = offsetQueue_.DeQue<xType>();
        BroadCastAntiquantParams(offsetComputeTensor_, offsetInput);
        offsetQueue_.FreeTensor(offsetInput);
    }

    LocalTensor<xType> scaleInput = scaleQueue_.AllocTensor<xType>();
    DataCopyPad(scaleInput, scaleGlobal_[antiquantOffset], antiquantCopyinParams_, antiquantCopyinPadParams_);
    scaleQueue_.EnQue(scaleInput);
    scaleInput = scaleQueue_.DeQue<xType>();
    BroadCastAntiquantParams(scaleComputeTensor_, scaleInput);
    scaleQueue_.FreeTensor(scaleInput);

    SetAntiquantTensorShape(weightSplitInfo);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2Common<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
                                  quantType>::SetAntiquantCopyInParams(uint64_t &antiquantOffset,
                                                                       WeightSplitInfo &weightSplitInfo)
{
    if constexpr (antiQuantType != Mc2QuantType::PER_GROUP) {
        // 非group场景，每次搬运n个数
        antiquantCopyinParams_.blockLen = weightSplitInfo.vecSingleN * sizeof(xType);
    } else {
        int64_t curGroupNum = weightSplitInfo.kOffset / tiling_->groupSize;
        int64_t dstGroupNum = CeilDiv(weightSplitInfo.kOffset + weightSplitInfo.vecSingleK, tiling_->groupSize);
        int64_t vecBaseGroupNum = dstGroupNum - curGroupNum;
        if constexpr (bTrans) {
            // 全载场景
            if (tiling_->vecSingleK >= tiling_->kSize) {
                antiquantCopyinParams_.blockLen = weightSplitInfo.vecSingleN * vecBaseGroupNum * sizeof(xType);
            } else {
                // 不全载场景
                antiquantCopyinParams_.blockCount = weightSplitInfo.vecSingleN;
                antiquantCopyinParams_.blockLen = vecBaseGroupNum * sizeof(xType);
            }
            antiquantCopyinParams_.srcStride = (groupNum_ - vecBaseGroupNum) * sizeof(xType);
            antiquantOffset = weightSplitInfo.originNOffset * groupNum_ + curGroupNum;
        } else {
            // weight不转置场景，不区分k是否全载
            antiquantCopyinParams_.blockCount = vecBaseGroupNum;
            antiquantCopyinParams_.blockLen = weightSplitInfo.vecSingleN * sizeof(xType);
            antiquantCopyinParams_.srcStride = (tiling_->nSize - weightSplitInfo.vecSingleN) * sizeof(xType);
            antiquantOffset = curGroupNum * tiling_->nSize + weightSplitInfo.originNOffset;
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2Common<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
                                  quantType>::SetAntiquantTensorShape(WeightSplitInfo &weightSplitInfo)
{
    if constexpr (antiQuantType == Mc2QuantType::PER_GROUP) {
        SetAntiquantTensorShapeWithGroup(weightSplitInfo);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2Common<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
                                  quantType>::SetAntiquantTensorShapeWithGroup(WeightSplitInfo &weightSplitInfo)
{
    int64_t curGroupNum = weightSplitInfo.kOffset / tiling_->groupSize;
    int64_t dstGroupNum = CeilDiv(weightSplitInfo.kOffset + weightSplitInfo.vecSingleK, tiling_->groupSize);
    int64_t vecBaseGroupNum = dstGroupNum - curGroupNum;
    if constexpr (bTrans) {
        // 全载场景
        if (tiling_->vecSingleK >= tiling_->kSize) {
            tensorShape_.scaleKFull = true;
            tensorShape_.scaleK = static_cast<uint32_t>(
                CeilDiv(weightSplitInfo.vecSingleN * groupNum_, FP32_ALIGN_SIZE) * FP32_ALIGN_SIZE);
            tensorShape_.scaleN = static_cast<uint32_t>(oneBlockAlignSize_);
            tensorShape_.scaleOrigK = static_cast<uint32_t>(weightSplitInfo.vecSingleN * groupNum_);
            tensorShape_.scaleOrigN = tensorShape_.scaleN;
        } else {
            tensorShape_.scaleKFull = false;
            tensorShape_.scaleK = static_cast<uint32_t>(weightSplitInfo.vecSingleN);
            tensorShape_.scaleN = static_cast<uint32_t>(CeilDiv(vecBaseGroupNum, FP16_ALIGN_SIZE) * FP16_ALIGN_SIZE);
            tensorShape_.scaleOrigK = static_cast<uint32_t>(weightSplitInfo.vecSingleN);
            tensorShape_.scaleOrigN = static_cast<uint32_t>(vecBaseGroupNum);
        }
    } else {
        // weight不转置场景，不区分k是否全载
        tensorShape_.scaleK = static_cast<uint32_t>(vecBaseGroupNum);
        tensorShape_.scaleN =
            static_cast<uint32_t>(CeilDiv(weightSplitInfo.vecSingleN, FP16_ALIGN_SIZE) * FP16_ALIGN_SIZE);
        tensorShape_.scaleOrigK = static_cast<uint32_t>(vecBaseGroupNum);
        tensorShape_.scaleOrigN = weightSplitInfo.vecSingleN;
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2Common<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
                                  quantType>::BroadCastAntiquantParams(LocalTensor<antiQuantCalType> &dstTensor,
                                                                       LocalTensor<xType> &srcTensor)
{
    if constexpr (antiQuantType == Mc2QuantType::PER_TENSOR) {
        return;
    }
    if constexpr (bTrans) {
        // weight转置场景需要作BroadCast与处理
        if constexpr (IsSameType<xType, bfloat16_t>::value) {
            // bf16场景，需要将tensor转换成float再运算
            LocalTensor<float> fp32Tensor = antiquantParamsFp32TBuf_.Get<float>();
            Cast(fp32Tensor, srcTensor, RoundMode::CAST_NONE, srcTensor.GetSize());
            PipeBarrier<PIPE_V>();
            Brcb(dstTensor, fp32Tensor, srcTensor.GetSize() / (ONE_BLK_FLOAT_NUM), brcbParams_);
        } else {
            Brcb(dstTensor, srcTensor, srcTensor.GetSize() / (ONE_BLK_FLOAT_NUM), brcbParams_);
        }
    } else {
        if (wFormat_ == static_cast<int32_t>(CubeFormat::NZ) && antiQuantType == Mc2QuantType::PER_CHANNEL) {
            UnaryRepeatParams unaryRepeatParams;
            unaryRepeatParams.dstBlkStride = 1;
            unaryRepeatParams.srcBlkStride = 1;
            unaryRepeatParams.dstRepStride = 32;
            unaryRepeatParams.srcRepStride = 1;

            Duplicate(dstTensor, static_cast<antiQuantCalType>(0), dstTensor.GetSize());
            if constexpr (IsSameType<xType, bfloat16_t>::value) {
                Cast(dstTensor, srcTensor, RoundMode::CAST_NONE, BLOCK_CUBE, srcTensor.GetSize() / BLOCK_CUBE,
                     unaryRepeatParams);
                PipeBarrier<PIPE_V>();
                for (uint32_t i = 0; i < dstTensor.GetSize() / FRACTAL_SIZE_F16; i++) {
                    Adds(dstTensor[i * FRACTAL_SIZE_F16], dstTensor[i * FRACTAL_SIZE_F16],
                         static_cast<antiQuantCalType>(0), BLOCK_CUBE, BLOCK_CUBE, {1, 1, 2, 0});
                }

            } else {
                for (uint32_t i = 0; i < dstTensor.GetSize() / FRACTAL_SIZE_F16; i++) {
                    Adds(dstTensor[i * FRACTAL_SIZE_F16], srcTensor[i * BLOCK_CUBE], static_cast<antiQuantCalType>(0),
                         BLOCK_CUBE, BLOCK_CUBE, {1, 1, 1, 0});
                }
            }
        } else {
            // weight不转置场景不需要预处理
            if constexpr (IsSameType<xType, bfloat16_t>::value) {
                // bf16场景，需要将tensor转换成float再运算
                Cast(dstTensor, srcTensor, RoundMode::CAST_NONE, srcTensor.GetSize());
            } else {
                DataCopy(dstTensor, srcTensor, srcTensor.GetSize());
            }
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2Common<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
                                  quantType>::WeightCopyInAndCast(WeightSplitInfo &weightSplitInfo)
{
    uint64_t wSrcOffset = weightSplitInfo.originNOffset;
    DataCopyParams copyinParams;
    DataCopyPadParams copyInPadParams;
    if constexpr (bTrans) {
        uint64_t vecSingleKAlign =
            CeilDiv(weightSplitInfo.vecSingleK, weightInnerAxisAlignSize_) * weightInnerAxisAlignSize_;
        wSrcOffset = wSrcOffset * tiling_->kSize + weightSplitInfo.kOffset;

        // weight非连续数据，存在跳读的情况，因此BlockCount为实际需要搬运的n值。b转至的场景下两行相差k
        copyinParams.blockCount = weightSplitInfo.vecSingleN;
        copyinParams.blockLen = weightSplitInfo.vecSingleK * sizeof(wType);
        copyinParams.srcStride = (tiling_->kSize * sizeof(wType) - copyinParams.blockLen);
        copyinParams.dstStride = (vecSingleKAlign - weightSplitInfo.vecSingleK) * sizeof(wType) / ONE_BLK_SIZE;
    } else {
        uint64_t vecSingleNAlign =
            CeilDiv(weightSplitInfo.vecSingleN, weightInnerAxisAlignSize_) * weightInnerAxisAlignSize_;
        wSrcOffset = weightSplitInfo.kOffset * tiling_->nSize + wSrcOffset;
        copyinParams.blockCount = weightSplitInfo.vecSingleK;
        copyinParams.blockLen = weightSplitInfo.vecSingleN * sizeof(wType);
        copyinParams.srcStride = (tiling_->nSize * sizeof(wType) - copyinParams.blockLen);
        copyinParams.dstStride = (vecSingleNAlign - weightSplitInfo.vecSingleN) * sizeof(wType) / ONE_BLK_SIZE;
    }

    if constexpr (IsSameType<wType, int4b_t>::value) {
        // int4场景下，跳转的步长、数据长度等需要除2
        copyinParams.blockLen = copyinParams.blockLen >> 1;
        copyinParams.srcStride = copyinParams.srcStride >> 1;
        copyinParams.dstStride = copyinParams.dstStride >> 1;
    }

    LocalTensor<wType> originWeight = originWeightQueue_.AllocTensor<wType>();
    DataCopyPad(originWeight, wGlobal_[wSrcOffset], copyinParams, copyInPadParams);
    originWeightQueue_.EnQue(originWeight);
    originWeight = originWeightQueue_.DeQue<wType>();

    LocalTensor<half> weight16 = weight16Tbuf_.Get<half>();
    Cast(weight16, originWeight, RoundMode::CAST_NONE, originWeight.GetSize());

    originWeightQueue_.FreeTensor(originWeight);

    PipeBarrier<PIPE_V>();
    if constexpr (IsSameType<xType, bfloat16_t>::value) {
        // bf16场景，需要转换成fp32计算
        LocalTensor<float> weight32 = weight32Tbuf_.Get<float>();
        Cast(weight32, weight16, RoundMode::CAST_NONE, weight16.GetSize());
        PipeBarrier<PIPE_V>();
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2Common<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
                                  quantType>::AntiQuantCompute(WeightSplitInfo &weightSplitInfo)
{
    LocalTensor<antiQuantCalType> antiquantWeightTensor;
    if constexpr (IsSameType<xType, bfloat16_t>::value) {
        antiquantWeightTensor = weight32Tbuf_.Get<float>();
    } else {
        antiquantWeightTensor = weight16Tbuf_.Get<half>();
    }

    if constexpr (bTrans) {
        uint64_t vecSingleKAlign =
            CeilDiv(weightSplitInfo.vecSingleK, weightInnerAxisAlignSize_) * weightInnerAxisAlignSize_;

        if (tiling_->groupSize > 0 && tiling_->vecSingleK >= tiling_->kSize &&
            tiling_->vecSingleK >= tiling_->repeatAxisMax) {
            // group k全载，且stride超过限制的情况，需要将shape reshape后传入,
            // 此时tiling保证kAlign是groupsize倍数。不会出现尾块
            tensorShape_.srcK = static_cast<uint32_t>(weightSplitInfo.vecSingleN * this->groupNum_);
            tensorShape_.srcN = static_cast<uint32_t>(tiling_->groupSize);
            tensorShape_.srcOrigK = static_cast<uint32_t>(weightSplitInfo.vecSingleN * this->groupNum_);
            tensorShape_.srcOrigN = static_cast<uint32_t>(tiling_->groupSize);
        } else {
            tensorShape_.srcK = static_cast<uint32_t>(weightSplitInfo.vecSingleN);
            tensorShape_.srcN = static_cast<uint32_t>(vecSingleKAlign);
            tensorShape_.srcOrigK = static_cast<uint32_t>(weightSplitInfo.vecSingleN);
            tensorShape_.srcOrigN = static_cast<uint32_t>(weightSplitInfo.vecSingleK);
        }
    } else {
        uint64_t vecSingleNAlign =
            CeilDiv(weightSplitInfo.vecSingleN, weightInnerAxisAlignSize_) * weightInnerAxisAlignSize_;
        tensorShape_.srcK = static_cast<uint32_t>(weightSplitInfo.vecSingleK);
        tensorShape_.srcN = static_cast<uint32_t>(vecSingleNAlign);
        tensorShape_.srcOrigK = static_cast<uint32_t>(weightSplitInfo.vecSingleK);
        tensorShape_.srcOrigN = static_cast<uint32_t>(weightSplitInfo.vecSingleN);
    }
    if constexpr (antiQuantType == Mc2QuantType::PER_TENSOR) {
        Mc2AntiQuant<antiQuantCalType, antiQuantCalType, antiQuantCalType, bTrans, hasAntiQuantOffset>(
            antiquantWeightTensor, antiquantWeightTensor, scaleValue_, offsetValue_, sharedTmpBuffer_);
    } else {
        Mc2AntiQuant<antiQuantCalType, antiQuantCalType, antiQuantCalType, bTrans, hasAntiQuantOffset>(
            antiquantWeightTensor, antiquantWeightTensor, scaleComputeTensor_, offsetComputeTensor_, tensorShape_,
            sharedTmpBuffer_, tiling_->groupSize);
    }
    PipeBarrier<PIPE_V>();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2Common<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
                                  quantType>::AntiQuantAdd(const LocalTensor<antiQuantCalType> &srcTensor,
                                                           const LocalTensor<antiQuantCalType> &offsetComputeBuf,
                                                           const AntiquantLoopParams &antiquantLoopParams,
                                                           const BinaryRepeatParams &repeatParams)
{
    uint64_t srcOffset = 0;
    uint64_t antiQuantOffset = 0;
    for (uint64_t repeatLoopIdx = 0; repeatLoopIdx < antiquantLoopParams.repeatLoop; repeatLoopIdx++) {
        AntiQuantAddAtMask(srcTensor[srcOffset], offsetComputeBuf[antiQuantOffset], MAX_REPEAT_TIMES,
                           antiquantLoopParams, repeatParams);
        srcOffset += (MAX_REPEAT_TIMES * tiling_->groupSize);

        // antiquantOffset每次repeat只跳一个block，总共跳MAX_REPEAT_TIMES次
        antiQuantOffset += (MAX_REPEAT_TIMES * oneBlockAlignSize_);
    }

    if (antiquantLoopParams.repeatRemain > 0) {
        AntiQuantAddAtMask(srcTensor[srcOffset], offsetComputeBuf[antiQuantOffset], antiquantLoopParams.repeatRemain,
                           antiquantLoopParams, repeatParams);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2Common<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
                                  quantType>::AntiQuantAddAtMask(const LocalTensor<antiQuantCalType> &srcTensor,
                                                                 const LocalTensor<antiQuantCalType> &offsetComputeBuf,
                                                                 int32_t repeat,
                                                                 const AntiquantLoopParams &antiquantLoopParams,
                                                                 const BinaryRepeatParams &repeatParams)
{
    uint64_t srcOffset = 0;
    for (uint64_t maskLoopIdx = 0; maskLoopIdx < antiquantLoopParams.maskLoop; maskLoopIdx++) {
        Add(srcTensor[srcOffset], offsetComputeBuf, srcTensor[srcOffset], maskMax_, repeat, repeatParams);
        srcOffset += maskMax_;
    }

    if (antiquantLoopParams.maskRemain > 0) {
        Add(srcTensor[srcOffset], offsetComputeBuf, srcTensor[srcOffset], antiquantLoopParams.maskRemain, repeat,
            repeatParams);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2Common<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
                                  quantType>::AntiQuantMul(const LocalTensor<antiQuantCalType> &srcTensor,
                                                           const LocalTensor<antiQuantCalType> &scaleComputeBuf,
                                                           const AntiquantLoopParams &antiquantLoopParams,
                                                           const BinaryRepeatParams &repeatParams)
{
    uint64_t srcOffset = 0;
    uint64_t antiQuantOffset = 0;
    for (uint64_t repeatLoopIdx = 0; repeatLoopIdx < antiquantLoopParams.repeatLoop; repeatLoopIdx++) {
        AntiQuantMulAtMask(srcTensor[srcOffset], scaleComputeBuf[antiQuantOffset], MAX_REPEAT_TIMES,
                           antiquantLoopParams, repeatParams);
        srcOffset += (MAX_REPEAT_TIMES * tiling_->groupSize);

        // antiquantOffset每次repeat只跳一个block，总共跳MAX_REPEAT_TIMES次
        antiQuantOffset += (MAX_REPEAT_TIMES * oneBlockAlignSize_);
    }

    if (antiquantLoopParams.repeatRemain > 0) {
        AntiQuantMulAtMask(srcTensor[srcOffset], scaleComputeBuf[antiQuantOffset], antiquantLoopParams.repeatRemain,
                           antiquantLoopParams, repeatParams);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2Common<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
                                  quantType>::AntiQuantMulAtMask(const LocalTensor<antiQuantCalType> &srcTensor,
                                                                 const LocalTensor<antiQuantCalType> &scaleComputeBuf,
                                                                 int32_t repeat,
                                                                 const AntiquantLoopParams &antiquantLoopParams,
                                                                 const BinaryRepeatParams &repeatParams)
{
    uint64_t srcOffset = 0;
    for (uint64_t maskLoopIdx = 0; maskLoopIdx < antiquantLoopParams.maskLoop; maskLoopIdx++) {
        Mul(srcTensor[srcOffset], scaleComputeBuf, srcTensor[srcOffset], maskMax_, repeat, repeatParams);
        srcOffset += maskMax_;
    }

    if (antiquantLoopParams.maskRemain > 0) {
        Mul(srcTensor[srcOffset], scaleComputeBuf, srcTensor[srcOffset], antiquantLoopParams.maskRemain, repeat,
            repeatParams);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2Common<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
                                  quantType>::WeightCopyOut(WeightSplitInfo &weightSplitInfo)
{
    LocalTensor<xType> weightOutput = weightOutputQueue_.AllocTensor<xType>();
    PipeBarrier<PIPE_V>();
    if constexpr (IsSameType<xType, bfloat16_t>::value) {
        LocalTensor<float> weight32 = weight32Tbuf_.Get<float>();
        Cast(weightOutput, weight32, RoundMode::CAST_RINT, weight32.GetSize());
    } else {
        LocalTensor<half> weight16 = weight16Tbuf_.Get<half>();
        DataCopy(weightOutput, weight16, weight16.GetSize());
    }

    // 计算好的weight搬运回workspace上涉及跳写
    DataCopyExtParams copyoutParams;

    weightOutputQueue_.EnQue(weightOutput);
    weightOutput = weightOutputQueue_.DeQue<xType>();
    uint64_t wDstOffset = weightCacheIdx_ * weightCacheSizeAlign_;
    if constexpr (bTrans) {
        wDstOffset += weightSplitInfo.splitNOffset * kCacheAlignSize_ + weightSplitInfo.kOffset;

        uint64_t vecSingleKAlign =
            CeilDiv(weightSplitInfo.vecSingleK, weightInnerAxisAlignSize_) * weightInnerAxisAlignSize_;
        copyoutParams.blockCount = weightSplitInfo.vecSingleN;
        copyoutParams.blockLen = weightSplitInfo.vecSingleK * sizeof(xType);
        copyoutParams.dstStride = (kCacheAlignSize_ - weightSplitInfo.vecSingleK) * sizeof(xType);
        copyoutParams.srcStride = (vecSingleKAlign - weightSplitInfo.vecSingleK) * sizeof(xType) / ONE_BLK_SIZE;
    } else {
        wDstOffset += cubeBaseN_ * weightSplitInfo.kOffset + weightSplitInfo.splitNOffset;
        uint64_t vecSingleNAlign =
            CeilDiv(weightSplitInfo.vecSingleN, weightInnerAxisAlignSize_) * weightInnerAxisAlignSize_;
        copyoutParams.blockCount = weightSplitInfo.vecSingleK;
        copyoutParams.blockLen = weightSplitInfo.vecSingleN * sizeof(xType);
        copyoutParams.dstStride = (cubeBaseN_ - weightSplitInfo.vecSingleN) * sizeof(xType);
        copyoutParams.srcStride = (vecSingleNAlign - weightSplitInfo.vecSingleN) * sizeof(xType) / ONE_BLK_SIZE;
    }
    DataCopyPad(weightCache_[wDstOffset], weightOutput, copyoutParams);
    weightOutputQueue_.FreeTensor(weightOutput);
}
} // namespace Mc2WeightQuantBatchMatmulV2
#endif // WEIGHT_QUANT_BATCHMATMUL_V2_COMMON_H
