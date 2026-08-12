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
 * \file weight_quant_batch_matmul_v2_reg_base_common.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCHMATMUL_V2_REG_BASE_COMMON_H
#define WEIGHT_QUANT_BATCHMATMUL_V2_REG_BASE_COMMON_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "tool_arch35.h"
#include "../weight_quant_batch_matmul_v2_constant.h"
#include "weight_quant_batch_matmul_v2_vf.h"
#include "n_first/basic_block_vf_nz.h"
#include "n_first/basic_block_vf_mx.h"

using AscendC::BLOCK_CUBE;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPadExtParams;
using AscendC::DataCopyParams;
using AscendC::GetBlockIdx;
using AscendC::GetTaskRation;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::int4b_t;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;
using AscendC::QuePosition;
using AscendC::RoundMode;
using AscendC::SetFlag;
using AscendC::SupportEnum;
using AscendC::SupportType;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::VECTOR_REG_WIDTH;
using AscendC::WaitFlag;
namespace MicroAPI = AscendC::MicroAPI;
using AscendC::MicroAPI::GetRound;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::TypeGet;
using matmul::MatmulImpl;
using matmul::MatmulType;

namespace Mc2WeightQuantBatchMatmulV2::Arch35 {

template <typename DtypeWeight>
constexpr bool Is4BitWeight()
{
    return (IsSameType<DtypeWeight, int4b_t>::value || IsSameType<DtypeWeight, fp4x2_e2m1_t>::value ||
            IsSameType<DtypeWeight, fp4x2_e1m2_t>::value);
}

template <typename DtypeWeight, bool weightNz>
constexpr int32_t GetMaxAL1BufNum()
{
    if (Is4BitWeight<DtypeWeight>() && weightNz) {
        // w4 weightNz场景AL1 buffer数量最大为4
        return 4;
    } else {
        // w8/w4 weightNd场景AL1 buffer数量最大为2
        return 2;
    }
}

template <typename DtypeWeight, bool transposeWeight, bool weightNz, Mc2QuantType antiQuantType>
constexpr int32_t GetMaxBubOutBufNum()
{
    if (Is4BitWeight<DtypeWeight>()) {
        if (weightNz) {
            // w4 weightNz场景ub buffer数量最大为4
            return 4;
        }
        // w4 weightNd场景ub buffer数量最大为2
        return 2;
    }
    if (IsSameType<DtypeWeight, int8_t>::value && antiQuantType == Mc2QuantType::PER_GROUP) {
        // A16W8 ND per-Group场景，BL1最大buffer数为2
        return 2;
    }
    // int8_t
    if (transposeWeight) {
        // w8 weight转置场景ub buffer数量最大为2
        return 2;
    }
    // w8 weight非转置场景ub buffer数量最大为4
    return 4;
}

constexpr int32_t MAX_AL1_BUF_NUM = GetMaxAL1BufNum<DTYPE_WEIGHT, true>();
constexpr uint64_t DOUBLE_BUFFER = 2;

template <typename dtype>
constexpr int32_t ElemsInBlock()
{
    if (Is4BitWeight<dtype>()) {
        // 当数据类型是int4时，单个block（32B）中的元素个数是64个
        return 64;
    }
    return ONE_BLK_SIZE / sizeof(dtype);
}

template <typename T>
__aicore__ inline T Max(T a, T b)
{
    return (a > b) ? (a) : (b);
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return (a > b) ? (b) : (a);
}

__aicore__ inline int64_t CeilDiv(int64_t x, int64_t y)
{
#ifdef __CCE_KT_TEST__
    ASCENDC_ASSERT(y != 0, { KERNEL_LOG(KERNEL_ERROR, "divide 0: x(%ld) y(%ld)", x, y); });
#endif
    return (x + y - 1) / y;
}

__aicore__ inline int64_t FloorAlign(int64_t x, int64_t y)
{
#ifdef __CCE_KT_TEST__
    ASCENDC_ASSERT(y != 0, { KERNEL_LOG(KERNEL_ERROR, "divide 0: x(%ld) y(%ld)", x, y); });
#endif
    return x / y * y;
}

template <Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline int64_t GetVecGn(int32_t bubKLen, uint64_t groupSize)
{
    if constexpr (antiQuantType == Mc2QuantType::PER_GROUP) {
        return CeilDiv(bubKLen, groupSize);
    } else {
        return 1;
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz = false>
class Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel {
    using VregType = typename TypeGet<xType>::T;

public:
    __aicore__ inline Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel(){};

    __aicore__ inline void UpdateGlobalAddr(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                            GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y,
                                            GM_ADDR workspace);
    __aicore__ inline void InitBiasL1Buffer(uint64_t offsetPing, uint64_t offsetPong);
    __aicore__ inline void InitL1Buffer();
    __aicore__ inline void InitUbBuffer();
    __aicore__ inline void InitScaleOffsetBuffer(int8_t bufNum, int32_t elemSize);
    __aicore__ inline void InitTilingData();
    __aicore__ inline void InitL1Params();
    __aicore__ inline void InitUB();
    __aicore__ inline bool IterMatmulOutKNotFullloadNoReuse();
    __aicore__ inline bool IterMatmulOut();
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                                const Mc2WeightQuantBatchMatmulV2RegBaseTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void CopyInWeight(int64_t bubKOffset, int64_t bubNOffset, int32_t bubKLen, int32_t bubNLen);
    __aicore__ inline void CopyInScaleOffset(int64_t bubNOffset, int32_t bubNLen, int32_t bubNLoopIdx,
                                             int32_t bubKLoopIdx, int64_t bubKOffset, int64_t bubKFactor);
    __aicore__ inline void AntiQuantCompute(int32_t bubKOffset, int32_t bubNLen, int32_t bubKLen);
    __aicore__ inline void AntiQuantComputeKNGroupWeightNz(int32_t bubKOffset, int32_t bubNLen, int32_t bubKLen);
    __aicore__ inline void AntiquantComputeW4NKPerGroup(int32_t bubNLen, int32_t bubKLen);
    __aicore__ inline void AntiquantComputeW8NKPerGroup(int32_t bubNLen, int32_t bubKLen);
    __aicore__ inline void AntiquantComputeKNPerGroup(int32_t bubNLen, int32_t bubKLen);
    __aicore__ inline void CopyVecOut2L1(int64_t l1Offset, LocalTensor<xType> ubLocal, int32_t bubKLen,
                                         int32_t bubNLen);
    __aicore__ inline void VectorRegCompute(int64_t bubKOffset, int32_t bubKLen, int32_t bubNLen);
    __aicore__ inline void BL1Process(uint64_t curBL1BufIdx, int64_t nBL1Offset, int64_t kBL1Offset, int32_t kL1Len,
                                      int32_t nL0Len);
    __aicore__ inline void BL1ProcessNK1Vs1(uint64_t curBL1BufIdx, int64_t nBL1Offset, int64_t kBL1Offset,
                                            int32_t kL1Len, int32_t nL0Len);
    __aicore__ inline void BL1ProcessNK1VsN(uint64_t curBL1BufIdx, int64_t nBL1Offset, int64_t kBL1Offset,
                                            int32_t kL1Len, int32_t nL0Len);
    __aicore__ inline void BL1ProcessNK(uint64_t curBL1BufIdx, int64_t nBL1Offset, int64_t kBL1Offset, int32_t kL1Len,
                                        int32_t nL0Len);
    __aicore__ inline void BL1ProcessKN1Vs1(uint64_t curBL1BufIdx, int64_t nBL1Offset, int64_t kBL1Offset);
    __aicore__ inline void BL1ProcessKN(uint64_t curBL1BufIdx, int64_t nBL1Offset, int64_t kBL1Offset, int32_t kL1Len,
                                        int32_t nL0Len);
    __aicore__ inline void CopyND2NZ(int al1PongFlag);

    __aicore__ inline void GetAL1KNotFullloadNoReuse(int64_t kFactorIdx,
                                                     AscendC::TEventID eventIdsMte1ToMte2[MAX_AL1_BUF_NUM]);
    __aicore__ inline void GetAL1(int64_t kFactorIdx, AscendC::TEventID eventIdsMte1ToMte2[MAX_AL1_BUF_NUM]);
    __aicore__ inline void GetBL1KNotFullloadNoReuse(int64_t kFactorIdx);
    __aicore__ inline void GetBL1(int64_t kFactorIdx);
    __aicore__ inline void GetBiasL1(int64_t kFactorIdx, AscendC::TEventID biasEventIdsMte1ToMte2[DOUBLE_BUFFER]);
    __aicore__ inline void IterateMatmulKNotFullloadNoReuse(int64_t kFactorIdx);
    __aicore__ inline void SetOrgShape();
    __aicore__ inline void SetTensorA(int64_t kFactorIdx);
    __aicore__ inline void SetTensorB(int64_t kFactorIdx);
    __aicore__ inline void GetTensorC();
    __aicore__ inline void IterateMatmul(int64_t kFactorIdx);
    __aicore__ inline void CopyUb2L1();
    __aicore__ inline void VectorProcess();
    __aicore__ inline void InitSync(AscendC::TEventID eventIdsMte1ToMte2[MAX_AL1_BUF_NUM],
                                    AscendC::TEventID biasEventIdsMte1ToMte2[DOUBLE_BUFFER]);
    __aicore__ inline void EndSync(AscendC::TEventID eventIdsMte1ToMte2[MAX_AL1_BUF_NUM],
                                   AscendC::TEventID biasEventIdsMte1ToMte2[DOUBLE_BUFFER]);
    __aicore__ inline void PostProcess(int32_t kFactorIdx, AscendC::TEventID eventIdsMte1ToMte2[MAX_AL1_BUF_NUM],
                                       AscendC::TEventID biasEventIdsMte1ToMte2[DOUBLE_BUFFER]);

    __aicore__ inline void NotifyCube()
    {
#ifndef __CCE_KT_TEST__
        CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE3>(1);
#endif
    }

    __aicore__ inline void WaitForVector(uint64_t bL1BufIdx)
    {
        if constexpr (IS_4BIT_WEIGHT ||
                      (IsSameType<wType, int8_t>::value && (!bTrans || antiQuantType == Mc2QuantType::PER_GROUP))) {
#ifndef __CCE_KT_TEST__
            CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(1 + FLAG_ID_MAX);
            CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(1);
#endif
            return;
        }

        if (unlikely((bl1pingpong_ == 1) && (tiling_->vecCoreParallel == 0))) {
#ifndef __CCE_KT_TEST__
            CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(1);
#endif
            return;
        }
        if (unlikely((bl1pingpong_ == 1) && (tiling_->vecCoreParallel == 1))) {
#ifndef __CCE_KT_TEST__
            CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(1 + FLAG_ID_MAX);
            CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(1);
#endif
            return;
        }

        if (likely(bl1pingpong_ == DOUBLE_BUFFER)) {
            if (bL1BufIdx == 1) {
#ifndef __CCE_KT_TEST__
                CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(1 + FLAG_ID_MAX);
#endif
            } else {
#ifndef __CCE_KT_TEST__
                CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(1);
#endif
            }
        }
    }

    __aicore__ inline void NotifyVector(uint64_t bL1BufIdx)
    {
        if constexpr (IS_4BIT_WEIGHT ||
                      (IsSameType<wType, int8_t>::value && (!bTrans || antiQuantType == Mc2QuantType::PER_GROUP))) {
#ifndef __CCE_KT_TEST__
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG + FLAG_ID_MAX);
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
#endif
            return;
        }

        if (unlikely((bl1pingpong_ == 1) && (tiling_->vecCoreParallel == 0))) {
#ifndef __CCE_KT_TEST__
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
#endif
            return;
        }

        if (unlikely((bl1pingpong_ == 1) && (tiling_->vecCoreParallel == 1))) {
#ifndef __CCE_KT_TEST__
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG + FLAG_ID_MAX);
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
#endif
            return;
        }

        if (likely(bl1pingpong_ == DOUBLE_BUFFER)) {
            if (bL1BufIdx == 1) {
#ifndef __CCE_KT_TEST__
                CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG + FLAG_ID_MAX);
#endif
            } else {
#ifndef __CCE_KT_TEST__
                CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
#endif
            }
        }
    }

    __aicore__ inline void WaitForCube()
    {
#ifndef __CCE_KT_TEST__
        CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
#endif
    }

    __aicore__ inline void EndWaitForCube()
    {
        if constexpr (IS_4BIT_WEIGHT ||
                      (IsSameType<wType, int8_t>::value && (!bTrans || antiQuantType == Mc2QuantType::PER_GROUP))) {
            if (bl1pingpong_ == QUADRUPLE_BUFFER) {
                WaitForCube();
                WaitForCube();
                WaitForCube();
                WaitForCube();
            } else if (bl1pingpong_ == DOUBLE_BUFFER) {
                WaitForCube();
                WaitForCube();
            } else {
                WaitForCube();
            }
            return;
        }
        if (unlikely(bl1pingpong_ == 1)) {
            if (tiling_->vecCoreParallel == 0) {
                if (AscendC::GetSubBlockIdx() == 0) {
                    WaitForCube();
                }
            } else {
                WaitForCube();
            }
            return;
        }
        if (likely(bl1pingpong_ == DOUBLE_BUFFER)) {
            WaitForCube();
        }
    }

    using inputXType = MatmulL1GmType<TPosition::TSCM, CubeFormat::NZ, xType, aTrans>;
    using inputWType = MatmulL1GmType<TPosition::TSCM, CubeFormat::NZ, xType, bTrans>;
    using outputYType = MatmulType<TPosition::GM, CubeFormat::ND, yType>;
    using inputBiasType = MatmulType<TPosition::TSCM, CubeFormat::ND, biasType>;
    MatmulImpl<inputXType, inputWType, outputYType, inputBiasType, CFG_MDL> mmObj_;

    TPipe *pipe_;
    const Mc2WeightQuantBatchMatmulV2RegBaseTilingData *tiling_;

    xType scaleValue_;
    xType offsetValue_;

    int8_t bubpingpong_;
    int8_t bl1pingpong_;
    int8_t al1pingpong_;
    int8_t biasPingPong_;

    int32_t curBlockIdx_;
    int32_t nDimIdx_;
    int32_t mDimIdx_;

    int64_t kAL1Size_;
    int64_t kBL1Size_;
    int64_t aL1DataSize_;
    int64_t bL1DataSize_;
    int64_t biasL1DataSize_;

    int64_t kSingleCoreIterNum_;

    int64_t mSingleCoreSize_;
    int64_t kSingleCoreSize_;
    int64_t nSingleCoreSize_;

    int64_t tailM_;
    int64_t tailN_;

    int64_t tailL1M_;
    int64_t tailL1N_;

    int64_t mBlockOffset_;
    int64_t nBlockOffset_;

    int64_t kAL1Offset_;
    int64_t kBL1Offset_;
    int64_t kAL1Len_;
    int64_t kBL1Len_;
    int64_t vecKBL1Len_;
    int64_t nBL1Offset_;
    int64_t aGmOffset_;
    int64_t bL1Offset_;
    int64_t aL1Offset_;

    int64_t curML0Idx_;
    int64_t curNL0Idx_;
    int64_t curML1Idx_;
    int64_t curNL1Idx_;
    int64_t mIter_;
    int64_t nIter_;
    int64_t baseUseM_;
    int64_t baseUseN_;
    int64_t mAL1Len_;
    int64_t nBL1Len_;
    int64_t vecNBL1Len_;

    uint16_t repeatTimes_;
    int32_t vsstbConfig_;
    int64_t outDimOffset_;
    uint32_t groupNumBub_;
    uint32_t groupNum_;

    uint32_t predictTailVcvt_;
    uint32_t predictTailCalcFirst_;
    uint32_t predictTailCalcSecond_;
    uint32_t calCount_;

    uint64_t ubScalePongFlag_;
    uint64_t vecWeightInSize_;
    uint64_t vecWeightOutSize_;
    uint64_t vecScaleOffsetSize_;
    int64_t kAl1Factor_;
    int64_t kBl1Factor_;
    // ubBufIdx_ 的计算对于 A16W4 和 A16W8 不兼容, A16W4 目前可以保证 UB buffer 的数量
    // 与 BL1 buffer 数量一致, 但是 A16W8 有例外情况

    uint64_t ubInBufIdx_ = 0;
    uint64_t ubOutBufIdx_ = 0;
    uint8_t biasIdx_ = 0;

    int64_t idx_ = -1;
    static constexpr uint64_t QUADRUPLE_BUFFER = 4;
    uint8_t vecPingpong_ = SINGLE_BUFFER;
    bool isFirstIter_ = true;
    uint8_t curAL1BufIdx_ = 0;
    uint8_t curBL1BufIdx_ = 0;
    bool twoVectorCoreSplitN_ = false;
    bool twoVectorCoreSplitK_ = false;
    bool fullloadKaIn1Buf_ = false;
    bool reserved_ = false;

    GlobalTensor<xType> xGlobal_;
    GlobalTensor<wType> wGlobal_;
    GlobalTensor<xType> addGlobal_;
    GlobalTensor<xType> mulGlobal_;
    GlobalTensor<biasType> biasGlobal_;
    GlobalTensor<yType> yGlobal_;

    LocalTensor<wType> weightInUb_;
    LocalTensor<xType> scaleInUb_;
    LocalTensor<xType> offsetInUb_;
    LocalTensor<xType> weightOutUb_;
    LocalTensor<xType> aL1LocalBuf0_;
    LocalTensor<xType> aL1LocalBuf1_;
    LocalTensor<xType> aL1LocalBuf2_;
    LocalTensor<xType> aL1LocalBuf3_;
    LocalTensor<xType> bL1LocalBuf0_;
    LocalTensor<xType> bL1LocalBuf1_;
    LocalTensor<xType> bL1LocalBuf2_;
    LocalTensor<xType> bL1LocalBuf3_;
    LocalTensor<xType> l1Local_;
    LocalTensor<biasType> biasL1LocalBuf0_;
    LocalTensor<biasType> biasL1LocalBuf1_;

    __local_mem__ int8_t *weightInUbBaseAddr_;
    __local_mem__ xType *scaleBaseAddr1_;
    __local_mem__ xType *offsetBaseAddr1_;
    __local_mem__ xType *weightOutUbAddr_;
    __local_mem__ xType *weightOutUbAddr1_;

protected:
    static constexpr uint64_t FLAG_ID_MAX = 16;
    static constexpr uint64_t L1_BUFFER_SIZE = 512 * 1024;
    static constexpr uint64_t L1_BUFFER_HALF_SIZE = 256 * 1024;
    static constexpr int32_t REPEAT_STRIDE_UINT = 5; // 位移位数, 32
    static constexpr uint64_t SINGLE_BUFFER = 1;
    static constexpr uint64_t SYNC_AIV_AIC_FLAG = 2;
    static constexpr uint8_t SYNC_MODE4 = 4;
    static constexpr uint64_t INT4_DTYPE_PARAM = 1; // 位移位数, 2
    static constexpr uint64_t VCVT_PARAM = 1;       // 位移位数, 2
    static constexpr int32_t VEC_MAX_ELEM_B16 = VECTOR_REG_WIDTH / sizeof(xType);
    static constexpr int64_t ONE_BLK_ELEM_B16 = ONE_BLK_SIZE / sizeof(xType);
    static constexpr int32_t BLOCK_CUBE_INT4 = BLOCK_CUBE >> INT4_DTYPE_PARAM;
    static constexpr int32_t BLOCK_NUM_REG = VECTOR_REG_WIDTH / ONE_BLK_SIZE;
    static constexpr int32_t MAX_BL1_BUF_NUM = GetMaxBubOutBufNum<DTYPE_WEIGHT, bTrans, weightNz, antiQuantType>();
    static constexpr bool USE_VSSTB =
        IsSameType<wType, int8_t>::value || (IsSameType<wType, int4b_t>::value && !weightNz);
    static constexpr bool IS_4BIT_WEIGHT = Is4BitWeight<wType>();
    static constexpr bool L1_4BUFFER = Is4BitWeight<wType>() && weightNz;
    static constexpr int8_t ONE_BLK_X_NUM = ElemsInBlock<xType>();
    static constexpr int8_t ONE_BLK_W_NUM = ElemsInBlock<wType>();

    TBuf<QuePosition::TSCM> l1Tbuf_;
    TBuf<QuePosition::VECIN> vecQueWeight_;
    TBuf<QuePosition::VECIN> vecQueScale_;
    TBuf<QuePosition::VECIN> vecQueOffset_;
    TBuf<QuePosition::VECOUT> vecQueWeightOut_;
};

enum class IterateOrder {
    ORDER_M = 0,
    ORDER_N,
    UNDEF
};

/*
 *  功能：申请各输入的 global buffer，A 矩阵与 B 矩阵在 L1 上的空间，B 矩阵、scale、offset 在 UB 上的空间
 */
template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::UpdateGlobalAddr(GM_ADDR x, GM_ADDR weight,
                                                                                          GM_ADDR antiquantScale,
                                                                                          GM_ADDR antiquantOffset,
                                                                                          GM_ADDR quantScale,
                                                                                          GM_ADDR quantOffset,
                                                                                          GM_ADDR bias, GM_ADDR y,
                                                                                          GM_ADDR workspace)
{
    // 申请 Global Buffer
    if ASCEND_IS_AIC {
        xGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(x));
        yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ yType *>(y));
        biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ biasType *>(bias));
    } else {
        wGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ wType *>(weight));
        mulGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(antiquantScale));
        addGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(antiquantOffset));
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::InitL1Params()
{
    kAL1Size_ = tiling_->matmulTiling.stepKa * tiling_->matmulTiling.baseK;
    kBL1Size_ = tiling_->matmulTiling.stepKb * tiling_->matmulTiling.baseK;
    int64_t maxKL1 = Max(kAL1Size_, kBL1Size_);
    kAl1Factor_ = CeilDiv(maxKL1, kBL1Size_);
    kBl1Factor_ = CeilDiv(maxKL1, kAL1Size_);

    aL1DataSize_ = tiling_->matmulTiling.baseM * kAL1Size_;
    bL1DataSize_ = tiling_->matmulTiling.baseN * kBL1Size_;
    biasL1DataSize_ = tiling_->matmulTiling.baseN;

    uint64_t singleM = CeilAlign(CeilDiv(tiling_->mSize, tiling_->cubeBlockDimM), 16);
    uint64_t singleN = CeilAlign(CeilDiv(tiling_->nSize, tiling_->cubeBlockDimN), 16);
    kSingleCoreIterNum_ = CeilDiv(tiling_->kSize, Min(kAL1Size_, kBL1Size_));

    int64_t mTailCoreSize_ = tiling_->mSize - (tiling_->cubeBlockDimM - 1) * singleM; // 尾核 m 大小
    int64_t nTailCoreSize_ = tiling_->nSize - (tiling_->cubeBlockDimN - 1) * singleN; // 尾核 n 大小

    mSingleCoreSize_ = mDimIdx_ == tiling_->cubeBlockDimM - 1 ? mTailCoreSize_ : singleM; // 单核内 m 方向大小
    nSingleCoreSize_ = nDimIdx_ == tiling_->cubeBlockDimN - 1 ? nTailCoreSize_ : singleN; // 单核内 n 方向大小

    tailL1M_ = mSingleCoreSize_ % tiling_->matmulTiling.baseM; // 单核内l1上m方向尾块
    if (tailL1M_ == 0) {
        tailL1M_ = tiling_->matmulTiling.baseM;
    }

    tailL1N_ = nSingleCoreSize_ % tiling_->matmulTiling.baseN; // 单核内l1上n方向尾块
    if (tailL1N_ == 0) {
        tailL1N_ = tiling_->matmulTiling.baseN;
    }

    mBlockOffset_ = mDimIdx_ * singleM; // m 方向核间偏移
    nBlockOffset_ = nDimIdx_ * singleN; // n 方向核间偏移
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::InitTilingData()
{
    InitL1Params();

    mIter_ = CeilDiv(mSingleCoreSize_, tiling_->matmulTiling.baseM); // 单核内一共有basem块数量
    nIter_ = CeilDiv(nSingleCoreSize_, tiling_->matmulTiling.baseN); // 单核内一共有basen块数量
    biasPingPong_ = Min(nIter_, static_cast<int64_t>(DOUBLE_BUFFER));

    tailM_ = mSingleCoreSize_ % tiling_->matmulTiling.baseM; // 单核内l0上m方向尾块
    if (tailM_ == 0) {
        tailM_ = tiling_->matmulTiling.baseM;
    }

    tailN_ = nSingleCoreSize_ % tiling_->matmulTiling.baseN; // 单核内l0上n方向尾块
    if (tailN_ == 0) {
        tailN_ = tiling_->matmulTiling.baseN;
    }

    fullloadKaIn1Buf_ = kAL1Size_ >= tiling_->kSize && al1pingpong_ == 1;
    if (antiQuantType == Mc2QuantType::PER_GROUP && bTrans) {
        // 计算antiquantOffset/Scale gm的内轴大小
        groupNum_ = CeilDiv(tiling_->kSize, tiling_->groupSize);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<
    xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
    weightNz>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale,
                    GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                    const Mc2WeightQuantBatchMatmulV2RegBaseTilingData *tilingData, TPipe *tPipe)
{
    tiling_ = tilingData;
    pipe_ = tPipe;
    if constexpr (L1_4BUFFER) {
        al1pingpong_ = tiling_->AL1Pingpong;
        bl1pingpong_ = tiling_->BL1Pingpong;
        bubpingpong_ = tiling_->BL1Pingpong;
    } else {
        al1pingpong_ = Min(tiling_->AL1Pingpong, static_cast<uint16_t>(2));
        bl1pingpong_ = Min(tiling_->BL1Pingpong, static_cast<uint16_t>(2));
        bubpingpong_ = tiling_->BL1Pingpong;
    }

    if ASCEND_IS_AIV {
        curBlockIdx_ = GetBlockIdx() / GetTaskRation();
    } else {
        curBlockIdx_ = GetBlockIdx();
    }
    nDimIdx_ = curBlockIdx_ / tiling_->cubeBlockDimM;
    mDimIdx_ = curBlockIdx_ % tiling_->cubeBlockDimM;

    InitTilingData();

    UpdateGlobalAddr(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, workspace);
    InitL1Buffer();
    InitUbBuffer();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::InitBiasL1Buffer(uint64_t offsetPing,
                                                                                          uint64_t offsetPong)
{
    if (tiling_->matmulTiling.isBias) {
        biasL1LocalBuf0_ = l1Tbuf_.template GetWithOffset<biasType>(biasL1DataSize_, offsetPing);
        if (biasPingPong_ > 1) {
            biasL1LocalBuf1_ = l1Tbuf_.template GetWithOffset<biasType>(biasL1DataSize_, offsetPong);
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::InitL1Buffer()
{
    // 申请 L1 Buffer 以及 B 矩阵在 UB 上的输入与输出 buffer 空间
    pipe_->InitBuffer(l1Tbuf_,
                      L1_BUFFER_SIZE - tiling_->matmulTiling.isBias * tiling_->matmulTiling.baseN * sizeof(biasType));
    l1Local_ = l1Tbuf_.template Get<xType>();

    if ASCEND_IS_NOT_AIC {
        return;
    }

    // A16W4
    //   BL1Pingpong 1 2 4
    //   AL1Pingpong 1 2 4
    if constexpr (L1_4BUFFER) {
        if (bl1pingpong_ == QUADRUPLE_BUFFER) {
            // L1 布局如下：
            // 1) AL1 使能 4 buffer 时
            //    L1 (0~256KB):   BL1_P0 | BL1_P2 | AL1_P1 | AL1_P3 | BIAS_P1
            //    L1 (257~512KB): BL1_P1 | BL1_P3 | AL1_P0 | AL1_P2 | BIAS_P0
            // 2) AL1 使能 2 buffer 时
            //    L1 (0~256KB):   BL1_P0 | BL1_P2 | AL1_P1 | BIAS_P1
            //    L1 (257~512KB): BL1_P1 | BL1_P3 | AL1_P0 | BIAS_P0
            // 3) AL1 使能 1 buffer 时
            //    L1 (0~512KB):   BL1_P0 | BL1_P2 | AL1_P0 | BL1_P1 | BL1_P3 | BIAS_P0 | BIAS_P1
            int32_t bL1DataSizeTotal = DOUBLE_BUFFER * bL1DataSize_ * sizeof(xType);
            if (al1pingpong_ == QUADRUPLE_BUFFER) {
                bL1LocalBuf0_ = l1Tbuf_.template GetWithOffset<xType>(bL1DataSize_, 0);
                bL1LocalBuf2_ = l1Tbuf_.template GetWithOffset<xType>(bL1DataSize_, bL1DataSize_ * sizeof(xType));
                aL1LocalBuf0_ =
                    l1Tbuf_.template GetWithOffset<xType>(aL1DataSize_, L1_BUFFER_HALF_SIZE + bL1DataSizeTotal);
                aL1LocalBuf2_ = l1Tbuf_.template GetWithOffset<xType>(
                    aL1DataSize_, L1_BUFFER_HALF_SIZE + bL1DataSizeTotal + aL1DataSize_ * sizeof(xType));
                aL1LocalBuf1_ = l1Tbuf_.template GetWithOffset<xType>(aL1DataSize_, bL1DataSizeTotal);
                aL1LocalBuf3_ = l1Tbuf_.template GetWithOffset<xType>(aL1DataSize_,
                                                                      bL1DataSizeTotal + aL1DataSize_ * sizeof(xType));

                bL1LocalBuf1_ = l1Tbuf_.template GetWithOffset<xType>(bL1DataSize_, L1_BUFFER_HALF_SIZE);
                bL1LocalBuf3_ = l1Tbuf_.template GetWithOffset<xType>(bL1DataSize_, L1_BUFFER_HALF_SIZE +
                                                                                        bL1DataSize_ * sizeof(xType));
                InitBiasL1Buffer(L1_BUFFER_HALF_SIZE + bL1DataSizeTotal + DOUBLE_BUFFER * aL1DataSize_ * sizeof(xType),
                                 bL1DataSizeTotal + DOUBLE_BUFFER * aL1DataSize_ * sizeof(xType));
            } else if (al1pingpong_ == 2) {
                bL1LocalBuf0_ = l1Tbuf_.template GetWithOffset<xType>(bL1DataSize_, 0);
                bL1LocalBuf2_ = l1Tbuf_.template GetWithOffset<xType>(bL1DataSize_, bL1DataSize_ * sizeof(xType));
                aL1LocalBuf0_ =
                    l1Tbuf_.template GetWithOffset<xType>(aL1DataSize_, L1_BUFFER_HALF_SIZE + bL1DataSizeTotal);
                aL1LocalBuf1_ = l1Tbuf_.template GetWithOffset<xType>(aL1DataSize_, bL1DataSizeTotal);

                bL1LocalBuf1_ = l1Tbuf_.template GetWithOffset<xType>(bL1DataSize_, L1_BUFFER_HALF_SIZE);
                bL1LocalBuf3_ = l1Tbuf_.template GetWithOffset<xType>(bL1DataSize_, L1_BUFFER_HALF_SIZE +
                                                                                        bL1DataSize_ * sizeof(xType));
                InitBiasL1Buffer(L1_BUFFER_HALF_SIZE + bL1DataSizeTotal + aL1DataSize_ * sizeof(xType),
                                 bL1DataSizeTotal + aL1DataSize_ * sizeof(xType));
            } else { // al1pingpong_ == 1
                bL1LocalBuf0_ = l1Tbuf_.template GetWithOffset<xType>(bL1DataSize_, 0);
                bL1LocalBuf2_ = l1Tbuf_.template GetWithOffset<xType>(bL1DataSize_, bL1DataSize_ * sizeof(xType));
                aL1LocalBuf0_ = l1Tbuf_.template GetWithOffset<xType>(aL1DataSize_, bL1DataSizeTotal);
                bL1LocalBuf1_ = l1Tbuf_.template GetWithOffset<xType>(
                    bL1DataSize_, Max(L1_BUFFER_HALF_SIZE, bL1DataSizeTotal + aL1DataSize_ * sizeof(xType)));
                bL1LocalBuf3_ = l1Tbuf_.template GetWithOffset<xType>(
                    bL1DataSize_, Max(L1_BUFFER_HALF_SIZE, bL1DataSizeTotal + aL1DataSize_ * sizeof(xType)) +
                                      bL1DataSize_ * sizeof(xType));
                InitBiasL1Buffer(Max(L1_BUFFER_HALF_SIZE, bL1DataSizeTotal + aL1DataSize_ * sizeof(xType)) +
                                     bL1DataSizeTotal,
                                 Max(L1_BUFFER_HALF_SIZE, bL1DataSizeTotal + aL1DataSize_ * sizeof(xType)) +
                                     bL1DataSizeTotal + biasL1DataSize_ * sizeof(biasType));
            }
        } else {
            // BL1 为 single buffer 或 double buffer (未考虑 L1 bank 冲突)
            bL1LocalBuf0_ = l1Tbuf_.template GetWithOffset<xType>(bL1DataSize_, 0);
            if (bl1pingpong_ != 1) {
                bL1LocalBuf1_ = l1Tbuf_.template GetWithOffset<xType>(bL1DataSize_, bL1DataSize_ * sizeof(xType));
            }

            aL1LocalBuf0_ =
                l1Tbuf_.template GetWithOffset<xType>(aL1DataSize_, bl1pingpong_ * bL1DataSize_ * sizeof(xType));
            if (al1pingpong_ == 4) {
                aL1LocalBuf1_ = l1Tbuf_.template GetWithOffset<xType>(
                    aL1DataSize_, (aL1DataSize_ + bl1pingpong_ * bL1DataSize_) * sizeof(xType));
                aL1LocalBuf2_ = l1Tbuf_.template GetWithOffset<xType>(
                    aL1DataSize_, (aL1DataSize_ * 2 + bl1pingpong_ * bL1DataSize_) * sizeof(xType));
                aL1LocalBuf3_ = l1Tbuf_.template GetWithOffset<xType>(
                    aL1DataSize_, (aL1DataSize_ * 3 + bl1pingpong_ * bL1DataSize_) * sizeof(xType));
            } else if (al1pingpong_ == 2) {
                aL1LocalBuf1_ = l1Tbuf_.template GetWithOffset<xType>(
                    aL1DataSize_, (aL1DataSize_ + bl1pingpong_ * bL1DataSize_) * sizeof(xType));
            }
            InitBiasL1Buffer((al1pingpong_ * aL1DataSize_ + bl1pingpong_ * bL1DataSize_) * sizeof(xType),
                             (al1pingpong_ * aL1DataSize_ + bl1pingpong_ * bL1DataSize_) * sizeof(xType) +
                                 biasL1DataSize_ * sizeof(biasType));
        }
    } else {
        // A16W8
        //   BL1Pingpong 1 2
        //   AL1Pingpong 1 2
        // BL1 为 single buffer 或 double buffer (未考虑 L1 bank 冲突)
        bL1LocalBuf0_ = l1Tbuf_.template GetWithOffset<xType>(bL1DataSize_, 0);
        if (bl1pingpong_ != 1) {
            bL1LocalBuf1_ = l1Tbuf_.template GetWithOffset<xType>(bL1DataSize_, bL1DataSize_ * sizeof(xType));
        }

        aL1LocalBuf0_ =
            l1Tbuf_.template GetWithOffset<xType>(aL1DataSize_, bl1pingpong_ * bL1DataSize_ * sizeof(xType));
        if (al1pingpong_ != 1) {
            aL1LocalBuf1_ = l1Tbuf_.template GetWithOffset<xType>(
                aL1DataSize_, (aL1DataSize_ + bl1pingpong_ * bL1DataSize_) * sizeof(xType));
        }
        InitBiasL1Buffer((al1pingpong_ * aL1DataSize_ + bl1pingpong_ * bL1DataSize_) * sizeof(xType),
                         (al1pingpong_ * aL1DataSize_ + bl1pingpong_ * bL1DataSize_) * sizeof(xType) +
                             biasL1DataSize_ * sizeof(biasType));
    }
    mmObj_.SetSubBlockIdx(0);
    mmObj_.Init(&tiling_->matmulTiling, pipe_);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::InitUbBuffer()
{
    if ASCEND_IS_NOT_AIV {
        return;
    }

    if constexpr (IS_4BIT_WEIGHT && weightNz) {
        if constexpr (bTrans) {
            ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "not supported yet"); });
        } else {
            // 目前仅考虑 per-group 且 B 矩阵非转置场景
            vecWeightInSize_ = tiling_->nBubSize * tiling_->kBubSize;
            vecWeightOutSize_ = vecWeightInSize_;
            pipe_->InitBuffer(vecQueWeightOut_, bubpingpong_ * vecWeightOutSize_ * sizeof(xType));
            pipe_->InitBuffer(vecQueWeight_, bubpingpong_ * vecWeightInSize_ >> INT4_DTYPE_PARAM);
        }
    } else if constexpr (IsSameType<wType, int4b_t>::value && !weightNz) {
        if constexpr (bTrans) {
            vecWeightInSize_ = tiling_->nBubSize * CeilAlign(tiling_->kBubSize, ONE_BLK_W_NUM);
            vecWeightOutSize_ = (CeilAlign(tiling_->nBubSize, BLOCK_CUBE) + 1) * CeilAlign(tiling_->kBubSize, 32);
        } else {
            vecWeightInSize_ = tiling_->kBubSize * tiling_->nBubSize;
            vecWeightOutSize_ = (CeilAlign(tiling_->kBubSize, BLOCK_CUBE) + 1) * tiling_->nBubSize;
        }
        pipe_->InitBuffer(vecQueWeightOut_, bl1pingpong_ * vecWeightOutSize_ * sizeof(xType));
        pipe_->InitBuffer(vecQueWeight_, bubpingpong_ * vecWeightInSize_ >> INT4_DTYPE_PARAM);
    } else if constexpr (IsSameType<wType, int8_t>::value && !weightNz && antiQuantType == Mc2QuantType::PER_GROUP) {
        vecWeightInSize_ = tiling_->kBubSize * tiling_->nBubSize;
        if constexpr (bTrans) {
            vecWeightOutSize_ = (CeilAlign(tiling_->nBubSize, BLOCK_CUBE) + 1) * tiling_->kBubSize;
        } else {
            vecWeightOutSize_ = (CeilAlign(tiling_->kBubSize, BLOCK_CUBE) + 1) * tiling_->nBubSize;
        }
        pipe_->InitBuffer(vecQueWeightOut_, bl1pingpong_ * vecWeightOutSize_ * sizeof(xType));
        pipe_->InitBuffer(vecQueWeight_, bubpingpong_ * vecWeightInSize_);
    } else { // A16W8
        if constexpr (bTrans) {
            // (k1, n1, n0, k0)
            vecPingpong_ = (tiling_->nBubSize < tiling_->matmulTiling.baseN || tiling_->kBubSize < kBL1Size_) ?
                               DOUBLE_BUFFER :
                               SINGLE_BUFFER;
            vecWeightInSize_ = tiling_->nBubSize * CeilAlign(tiling_->kBubSize, ONE_BLK_W_NUM);
            vecWeightOutSize_ =
                (CeilAlign(tiling_->nBubSize, BLOCK_CUBE) + 1) * CeilAlign(tiling_->kBubSize, ONE_BLK_X_NUM);
            pipe_->InitBuffer(vecQueWeightOut_, vecPingpong_ * vecWeightOutSize_ * sizeof(xType));
            pipe_->InitBuffer(vecQueWeight_, vecPingpong_ * vecWeightInSize_ * sizeof(wType));
        } else {
            vecWeightInSize_ = tiling_->kBubSize * CeilAlign(tiling_->nBubSize, ONE_BLK_W_NUM);
            // (n1, k1, k0, n0) 由于vsstb指令每次处理32B数据，为了保证下一个32B不在同一个bank,需要多移动一个32B
            vecWeightOutSize_ =
                (CeilAlign(tiling_->kBubSize, BLOCK_CUBE) + 1) * CeilAlign(tiling_->nBubSize, ONE_BLK_X_NUM);
            pipe_->InitBuffer(vecQueWeightOut_, bubpingpong_ * vecWeightOutSize_ * sizeof(xType));
            pipe_->InitBuffer(vecQueWeight_, bubpingpong_ * vecWeightInSize_ * sizeof(wType));
        }
    }

    // 申请 scale 和 offset 在 UB 上的 buffer 空间
    if constexpr (antiQuantType == Mc2QuantType::PER_GROUP) {
        // 目前仅考虑 A16W4 场景
        if constexpr (bTrans) {
            vecScaleOffsetSize_ =
                tiling_->nBubSize * CeilAlign(CeilDiv(tiling_->kBubSize, tiling_->groupSize), ONE_BLK_X_NUM);
        } else {
            // 申请 bL1BufNum * groupNumBub * nBubSize 大小的空间
            vecScaleOffsetSize_ = CeilDiv(tiling_->kBubSize, tiling_->groupSize) * tiling_->nBubSize;
        }
        InitScaleOffsetBuffer(bubpingpong_, vecScaleOffsetSize_);
    }

    weightInUb_ = vecQueWeight_.template Get<wType>();
    weightOutUb_ = vecQueWeightOut_.template Get<xType>();
    if constexpr (antiQuantType != Mc2QuantType::PER_TENSOR) {
        scaleInUb_ = vecQueScale_.template Get<xType>();
        if constexpr (hasAntiQuantOffset) {
            offsetInUb_ = vecQueOffset_.template Get<xType>();
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::InitScaleOffsetBuffer(int8_t bufNum,
                                                                                               int32_t elemSize)
{
    pipe_->InitBuffer(vecQueScale_, bufNum * elemSize * sizeof(xType));
    if constexpr (hasAntiQuantOffset) {
        pipe_->InitBuffer(vecQueOffset_, bufNum * elemSize * sizeof(xType));
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::CopyND2NZ(int al1PongFlag)
{
    AscendC::Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    if constexpr (aTrans) {
        nd2nzParams.nValue = kAL1Len_;
        nd2nzParams.dValue = mAL1Len_;
        nd2nzParams.srcDValue = tiling_->matmulTiling.M;
    } else {
        nd2nzParams.nValue = mAL1Len_;
        nd2nzParams.dValue = kAL1Len_;
        nd2nzParams.srcDValue = tiling_->matmulTiling.Ka;
    }
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dstNzC0Stride = CeilAlign(nd2nzParams.nValue, BLOCK_CUBE);
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;

    if constexpr (L1_4BUFFER) {
        if (al1PongFlag == 0) {
            DataCopy(aL1LocalBuf0_, xGlobal_[aGmOffset_], nd2nzParams);
        } else if (al1PongFlag == 1) {
            DataCopy(aL1LocalBuf1_, xGlobal_[aGmOffset_], nd2nzParams);
        } else if (al1PongFlag == 2) {
            DataCopy(aL1LocalBuf2_, xGlobal_[aGmOffset_], nd2nzParams);
        } else {
            DataCopy(aL1LocalBuf3_, xGlobal_[aGmOffset_], nd2nzParams);
        }
    } else {
        if (al1PongFlag == 0) {
            DataCopy(aL1LocalBuf0_, xGlobal_[aGmOffset_], nd2nzParams);
        } else {
            DataCopy(aL1LocalBuf1_, xGlobal_[aGmOffset_], nd2nzParams);
        }
    }

    SetFlag<HardEvent::MTE2_MTE1>(al1PongFlag);
    WaitFlag<HardEvent::MTE2_MTE1>(al1PongFlag);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::CopyInWeight(int64_t bubKOffset,
                                                                                      int64_t bubNOffset,
                                                                                      int32_t bubKLen, int32_t bubNLen)
{
    DataCopyExtParams intriParams;
    intriParams.dstStride = 0;
    DataCopyPadExtParams<wType> padParams;
    int64_t gmOffset; // GM 上 B 矩阵的地址偏移（以元素个数为单位）
    if constexpr (bTrans) {
        if constexpr (IsSameType<wType, int8_t>::value) {
            intriParams.blockCount = bubNLen;
            intriParams.blockLen = bubKLen * sizeof(int8_t);
            intriParams.srcStride = (tiling_->kSize - bubKLen) * sizeof(int8_t);
            gmOffset = bubNOffset * tiling_->kSize + bubKOffset;
        } else if constexpr (IsSameType<wType, int4b_t>::value && !weightNz) {
            // (n, k)
            intriParams.blockCount = bubNLen;
            intriParams.blockLen = bubKLen >> INT4_DTYPE_PARAM;
            intriParams.srcStride = (tiling_->kSize - bubKLen) >> INT4_DTYPE_PARAM;
            gmOffset = bubNOffset * tiling_->kSize + bubKOffset;
#ifdef __CCE_KT_TEST__
        } else {
            ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "not support yet"); });
#endif
        }
    } else {                                              // B 矩阵非转置
        if constexpr (IsSameType<wType, int8_t>::value) { // A16W8, ND 格式输入, (k, n)
            intriParams.blockCount = bubKLen;
            intriParams.blockLen = bubNLen * sizeof(int8_t);
            intriParams.srcStride = (tiling_->nSize - bubNLen) * sizeof(int8_t);
            gmOffset = bubKOffset * tiling_->nSize + bubNOffset;
        } else if constexpr (IS_4BIT_WEIGHT && weightNz) {    // (n1, k1, k0, n0)
            intriParams.blockCount = bubNLen / BLOCK_CUBE;    // n1
            intriParams.blockLen = bubKLen * BLOCK_CUBE_INT4; // k1 * k0 * n0
            int64_t kAlignSize = CeilAlign(tiling_->kSize, BLOCK_CUBE);
            intriParams.srcStride = (kAlignSize - bubKLen) * BLOCK_CUBE_INT4;
            gmOffset = (bubNOffset * kAlignSize + bubKOffset * BLOCK_CUBE);
        } else if constexpr (IsSameType<wType, int4b_t>::value && !weightNz) {
            intriParams.blockCount = bubKLen;
            intriParams.blockLen = bubNLen >> INT4_DTYPE_PARAM;
            intriParams.srcStride = (tiling_->nSize - bubNLen) >> INT4_DTYPE_PARAM;
            gmOffset = bubKOffset * tiling_->nSize + bubNOffset;
        }
    }
    // UB 上存储反量化前 B 矩阵的空间的地址偏移（以元素个数为单位）
    DataCopyPad(weightInUb_[ubInBufIdx_ * vecWeightInSize_], wGlobal_[gmOffset], intriParams, padParams);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::CopyInScaleOffset(int64_t bubNOffset,
                                                                                           int32_t bubNLen,
                                                                                           int32_t bubNLoopIdx,
                                                                                           int32_t bubKLoopIdx,
                                                                                           int64_t bubKOffset,
                                                                                           int64_t bubKFactor)
{
    if constexpr (antiQuantType == Mc2QuantType::PER_TENSOR) {
        return;
    }

    DataCopyExtParams intriParams;
    intriParams.dstStride = 0;
    DataCopyPadExtParams<xType> padParams;
    int64_t gmOffset; // GM 上 scale 和 offset 的地址偏移（以元素个数为单位）
    int64_t ubOffset; // UB 上 scale 和 offset 的地址偏移（以元素个数为单位）

    if constexpr (bTrans) { // (n, k)
        if (bubKLoopIdx != 0) {
            return;
        }
        if constexpr (antiQuantType == Mc2QuantType::PER_GROUP) {
            intriParams.blockCount = bubNLen;
            intriParams.blockLen = groupNumBub_ * sizeof(xType);
            intriParams.srcStride = (groupNum_ - groupNumBub_) * sizeof(xType);
            gmOffset = bubNOffset * CeilDiv(tiling_->kSize, tiling_->groupSize) + bubKOffset / tiling_->groupSize;
            ubOffset = ubInBufIdx_ * vecScaleOffsetSize_;
        }
    } else { // B 矩阵非转置
        intriParams.blockLen = bubNLen * sizeof(xType);
        intriParams.srcStride = (tiling_->nSize - bubNLen) * sizeof(xType);
        if constexpr (antiQuantType == Mc2QuantType::PER_GROUP) {
            // k_offset + n_offset
            gmOffset = bubKOffset / tiling_->groupSize * tiling_->nSize + bubNOffset;
            ubOffset = ubInBufIdx_ * vecScaleOffsetSize_;
            intriParams.blockCount = groupNumBub_;
        }
    }
    DataCopyPad(scaleInUb_[ubOffset], mulGlobal_[gmOffset], intriParams, padParams);
    if constexpr (hasAntiQuantOffset) {
        DataCopyPad(offsetInUb_[ubOffset], addGlobal_[gmOffset], intriParams, padParams);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<
    xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
    weightNz>::AntiQuantComputeKNGroupWeightNz(int32_t bubKOffset, int32_t bubNLen, int32_t bubKLen)
{
    // 处理一块 (kBub, nBub), 数据格式为 (nBub1, kBub1, kBub0, nBub0)
    // 每次处理 128 个数，按照 128 个数 (256B) 输出, 对于每个 256B, 间隔 1024B 存放, 避免 bank 冲突

    uint16_t nbub1 = CeilDiv(bubNLen, BLOCK_CUBE);
    uint16_t mainGroupNum = bubKLen / tiling_->groupSize; // 一个 kBubSize 中 group 的个数
    uint16_t mainInnerNum; // 对于 kBubSize 的一个 group 中, 需要处理 innerNum 次 128 个数（对应 256B）, 目前只支持常数
    int32_t overlap_part = tiling_->groupSize - (bubKOffset - FloorAlign(bubKOffset, tiling_->groupSize));
    if ((bubKOffset % tiling_->groupSize == 0 && bubKLen % tiling_->groupSize == 0) ||
        bubKOffset + bubKLen >= tiling_->kSize) {
        mainInnerNum = tiling_->groupSize * BLOCK_CUBE / VEC_MAX_ELEM_B16;
    } else if (overlap_part >= bubKLen) {
        mainInnerNum = bubKLen * BLOCK_CUBE / VEC_MAX_ELEM_B16;
#ifdef __CCE_KT_TEST__
    } else {
        ASCENDC_ASSERT(false, {
            KERNEL_LOG(
                KERNEL_ERROR,
                "innerNum is not a constant, not supported yet!(bubKOffset %d groupSize %d bubKLen %d overlap_part %d)",
                bubKOffset, tiling_->groupSize, bubKLen, overlap_part);
        });
#endif
    }

    // extend以byte为单位
    constexpr uint32_t innerSrcExtend = VECTOR_REG_WIDTH >> 2; // 单次处理后 src 偏移为 128 个数（64B），偏移1/4
    // 单次处理后 dst 偏移为 128 * buffer数量，当buffer数量为4时，偏移1024B
    uint32_t innerDstExtend = VEC_MAX_ELEM_B16 * bubpingpong_;

    // 向VF中传入局部变量
    __local_mem__ xType *scaleBaseAddr = scaleBaseAddr1_;
    __local_mem__ xType *offsetBaseAddr = offsetBaseAddr1_;
    __local_mem__ int8_t *weightInUbBaseAddr = weightInUbBaseAddr_;
    __local_mem__ xType *weightOutUbBaseAddr = weightOutUbAddr_;

    int64_t mainGroupSize = mainGroupNum * tiling_->groupSize;
    if (unlikely(bubKLen > mainGroupSize)) {
        uint16_t tailInnerNum = CeilAlign(bubKLen - mainGroupSize, BLOCK_CUBE) * BLOCK_CUBE / VEC_MAX_ELEM_B16;
        uint32_t mainGroupNumSrcExtend = mainInnerNum * innerSrcExtend;
        uint32_t tailGroupNumSrcExtend = tailInnerNum * innerSrcExtend;
        uint32_t n1SrcExtend = CeilAlign(bubKLen, BLOCK_CUBE) * BLOCK_CUBE_INT4;
        uint32_t mainGroupNumDstExtend = mainInnerNum * innerDstExtend;
        uint32_t tailGroupNumDstExtend = tailInnerNum * innerDstExtend;
        uint32_t n1DstExtend =
            CeilAlign((CeilAlign(bubKLen, BLOCK_CUBE) * BLOCK_CUBE) * sizeof(xType), VECTOR_REG_WIDTH) / sizeof(xType) *
            bubpingpong_;
        __local_mem__ xType *tailScaleBaseAddr = scaleBaseAddr1_ + mainGroupNum * bubNLen;
        __local_mem__ xType *tailOffsetBaseAddr = offsetBaseAddr1_ + mainGroupNum * bubNLen;
        __local_mem__ int8_t *tailWeightInUbBaseAddr = weightInUbBaseAddr_ + mainGroupSize * BLOCK_CUBE_INT4;
        __local_mem__ xType *tailWeightOutUbBaseAddr =
            weightOutUbAddr_ +
            CeilAlign(mainGroupSize * BLOCK_CUBE * sizeof(xType), VECTOR_REG_WIDTH) / sizeof(xType) * bubpingpong_;
#ifndef __CCE_KT_TEST__
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<xType> wNzF16Part0, scale, offset;
            MicroAPI::RegTensor<typename RegTensorActualT<wType>::T> wNz4BitPart0;
            MaskReg preg = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
            // 对一个 kBubSize 中 group 的个数迭代
            for (uint16_t n1Idx = 0; n1Idx < (uint16_t)nbub1; ++n1Idx) { // 对 nBub1 迭代
                // 对一个 kBubSize 中 group 的个数迭代
                for (uint16_t groupIdx = 0; groupIdx < (uint16_t)mainGroupNum; ++groupIdx) {
                    MicroAPI::AddrReg aregScale = MicroAPI::CreateAddrReg<xType>(n1Idx, BLOCK_CUBE, groupIdx, bubNLen);
                    // 每次处理 128 个数, scale broadcast 为 128 个数 (256B)
                    MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BLK>(scale, scaleBaseAddr, aregScale);

                    if constexpr (hasAntiQuantOffset) {
                        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BLK>(offset, offsetBaseAddr, aregScale);
                    }
                    for (uint16_t innerIdx = 0; innerIdx < (uint16_t)mainInnerNum; ++innerIdx) { // 按 128 个数迭代
                        // UNPK4_B8 表示按照如下形式载入：
                        // Vn 1 2 3 4 5 6 7 8 9 a b c d e f g
                        // Vd 1 2 0 0 0 0 0 0 3 4 0 0 0 0 0 0
                        MicroAPI::DataCopy<typename RegTensorActualT<wType>::T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                            wNz4BitPart0,
                            (__local_mem__ typename RegTensorActualT<wType>::T *)(weightInUbBaseAddr +
                                                                                  n1Idx * n1SrcExtend +
                                                                                  groupIdx * mainGroupNumSrcExtend +
                                                                                  innerIdx * innerSrcExtend));
                        if constexpr (IsSameType<wType, fp4x2_e2m1_t>::value ||
                                      IsSameType<wType, fp4x2_e1m2_t>::value) {
                            if constexpr (!IsSameType<xType, half>::value) {
                                MicroAPI::Cast<xType, wType, CAST_FP4_TO_BF16_TRAIT>(wNzF16Part0, wNz4BitPart0, preg);
                            } else {
                                // FP4-->FP16指令不支持，需要转换为BF16再转换为FP16
                                RegTensor<bfloat16_t> weightBF16Vreg0;
                                MicroAPI::Cast<bfloat16_t, wType, CAST_FP4_TO_BF16_TRAIT>(weightBF16Vreg0, wNz4BitPart0,
                                                                                          preg);
                                MicroAPI::Cast<xType, bfloat16_t, CAST_BF16_TO_FP16_TRAIT>(wNzF16Part0, weightBF16Vreg0,
                                                                                           preg);
                            }
                        } else {
                            MicroAPI::Cast<xType, int4x2_t, CAST_S4_TO_F16_TRAIT>(wNzF16Part0, wNz4BitPart0, preg);
                        }
                        if constexpr (hasAntiQuantOffset) {
                            MicroAPI::Add(wNzF16Part0, wNzF16Part0, offset, preg);
                        }
                        MicroAPI::Mul(wNzF16Part0, wNzF16Part0, scale, preg);
                        MicroAPI::DataCopy<xType, MicroAPI::StoreDist::DIST_NORM_B16>(
                            weightOutUbBaseAddr + n1Idx * n1DstExtend + groupIdx * mainGroupNumDstExtend +
                                innerIdx * innerDstExtend,
                            wNzF16Part0, preg);
                    }
                }
                // 每次处理 128 个数, scale broadcast 为 128 个数 (256B)
                MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BLK>(scale, tailScaleBaseAddr + n1Idx * BLOCK_CUBE);
                if constexpr (hasAntiQuantOffset) {
                    MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BLK>(offset,
                                                                            tailOffsetBaseAddr + n1Idx * BLOCK_CUBE);
                }
                for (uint16_t innerIdx = 0; innerIdx < (uint16_t)tailInnerNum; ++innerIdx) { // 按 128 个数迭代
                    // UNPK4_B8 表示按照如下形式载入：
                    // Vn 1 2 3 4 5 6 7 8 9 a b c d e f g
                    // Vd 1 2 0 0 0 0 0 0 3 4 0 0 0 0 0 0
                    MicroAPI::DataCopy<typename RegTensorActualT<wType>::T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                        wNz4BitPart0, (__local_mem__ typename RegTensorActualT<wType>::T *)(tailWeightInUbBaseAddr +
                                                                                            n1Idx * n1SrcExtend +
                                                                                            innerIdx * innerSrcExtend));
                    if constexpr (IsSameType<wType, fp4x2_e2m1_t>::value || IsSameType<wType, fp4x2_e1m2_t>::value) {
                        if constexpr (!IsSameType<xType, half>::value) {
                            MicroAPI::Cast<xType, wType, CAST_FP4_TO_BF16_TRAIT>(wNzF16Part0, wNz4BitPart0, preg);
                        } else {
                            RegTensor<bfloat16_t> weightBF16Vreg0;
                            MicroAPI::Cast<bfloat16_t, wType, CAST_FP4_TO_BF16_TRAIT>(weightBF16Vreg0, wNz4BitPart0,
                                                                                      preg);
                            MicroAPI::Cast<xType, bfloat16_t, CAST_BF16_TO_FP16_TRAIT>(wNzF16Part0, weightBF16Vreg0,
                                                                                       preg);
                        }
                    } else {
                        MicroAPI::Cast<xType, int4x2_t, CAST_S4_TO_F16_TRAIT>(wNzF16Part0, wNz4BitPart0, preg);
                    }
                    if constexpr (hasAntiQuantOffset) {
                        MicroAPI::Add(wNzF16Part0, wNzF16Part0, offset, preg);
                    }
                    MicroAPI::Mul(wNzF16Part0, wNzF16Part0, scale, preg);
                    MicroAPI::DataCopy<xType, MicroAPI::StoreDist::DIST_NORM_B16>(
                        tailWeightOutUbBaseAddr + n1Idx * n1DstExtend + innerIdx * innerDstExtend, wNzF16Part0, preg);
                }
            }
        }
#endif
    } else {
        uint32_t groupNumSrcExtend = mainInnerNum * innerSrcExtend;
        uint32_t n1SrcExtend = mainGroupNum * groupNumSrcExtend;
        uint32_t groupNumDstExtend = mainInnerNum * innerDstExtend;
        uint32_t n1DstExtend = mainGroupNum * groupNumDstExtend;
#ifndef __CCE_KT_TEST__
        __VEC_SCOPE__
        {
            // (n1, k1, k0, n0)
            // nbub1, gn * mainInnerNum / 2, 16, 16
            RegTensor<xType> wNzF16;
            RegTensor<xType> scale;
            RegTensor<xType> offset;
            MaskReg preg = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
            // 对一个 kBubSize 中 group 的个数迭代
            for (uint16_t n1Idx = 0; n1Idx < (uint16_t)nbub1; ++n1Idx) { // 对 nBub1 迭代
                // 对一个 kBubSize 中 group 的个数迭代
                for (uint16_t groupIdx = 0; groupIdx < (uint16_t)mainGroupNum; ++groupIdx) {
                    MicroAPI::AddrReg aregScale = MicroAPI::CreateAddrReg<xType>(n1Idx, BLOCK_CUBE, groupIdx, bubNLen);
                    // 每次处理 128 个数, scale broadcast 为 128 个数 (256B)
                    MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BLK>(scale, scaleBaseAddr, aregScale);
                    if constexpr (hasAntiQuantOffset) {
                        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BLK>(offset, offsetBaseAddr, aregScale);
                    }
                    for (uint16_t innerIdx = 0; innerIdx < (uint16_t)mainInnerNum; ++innerIdx) { // 按 128 个数迭代
                        // UNPK4_B8 表示按照如下形式载入：
                        // Vn 1 2 3 4 5 6 7 8 9 a b c d e f g
                        // Vd 1 2 0 0 0 0 0 0 3 4 0 0 0 0 0 0
                        MicroAPI::AddrReg aregWeightIn = MicroAPI::CreateAddrReg<typename RegTensorActualT<wType>::T>(
                            n1Idx, n1SrcExtend, groupIdx, groupNumSrcExtend, innerIdx, innerSrcExtend);
                        MicroAPI::AddrReg aregWeightOut = MicroAPI::CreateAddrReg<xType>(
                            n1Idx, n1DstExtend, groupIdx, groupNumDstExtend, innerIdx, innerDstExtend);
                        RegTensor<typename RegTensorActualT<wType>::T> wNz4Bit;
                        MicroAPI::DataCopy<typename RegTensorActualT<wType>::T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                            wNz4Bit, (__local_mem__ typename RegTensorActualT<wType>::T *)(weightInUbBaseAddr),
                            aregWeightIn);
                        if constexpr (IsSameType<wType, fp4x2_e2m1_t>::value ||
                                      IsSameType<wType, fp4x2_e1m2_t>::value) {
                            if constexpr (!IsSameType<xType, half>::value) {
                                MicroAPI::Cast<xType, wType, CAST_FP4_TO_BF16_TRAIT>(wNzF16, wNz4Bit, preg);
                            } else {
                                RegTensor<bfloat16_t> weightBF16Vreg0;
                                MicroAPI::Cast<bfloat16_t, wType, CAST_FP4_TO_BF16_TRAIT>(weightBF16Vreg0, wNz4Bit,
                                                                                          preg);
                                MicroAPI::Cast<xType, bfloat16_t, CAST_BF16_TO_FP16_TRAIT>(wNzF16, weightBF16Vreg0,
                                                                                           preg);
                            }
                        } else {
                            MicroAPI::Cast<xType, int4x2_t, CAST_S4_TO_F16_TRAIT>(wNzF16, wNz4Bit, preg);
                        }
                        if constexpr (hasAntiQuantOffset) {
                            MicroAPI::Add(wNzF16, wNzF16, offset, preg);
                        }
                        MicroAPI::Mul(wNzF16, wNzF16, scale, preg);
                        MicroAPI::DataCopy<xType, MicroAPI::StoreDist::DIST_NORM_B16>(weightOutUbBaseAddr, wNzF16,
                                                                                      aregWeightOut, preg);
                    }
                }
            }
        }
#endif
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::AntiquantComputeKNPerGroup(int32_t bubNLen,
                                                                                                    int32_t bubKLen)
{
#ifdef __CCE_KT_TEST__
    ASCENDC_ASSERT(tiling_->groupSize > 0, { KERNEL_LOG(KERNEL_ERROR, "Invalid groupSize."); });
#endif
    ParamsKN<xType> param;
    param.kBub = bubKLen; // 实际长度
    param.nBub = bubNLen; // 实际长度
    param.nBubXTypeAlign = CeilAlign(param.nBub, BLOCK_CUBE);
    param.nBubWTypeAlign = CeilAlign(param.nBub, ONE_BLK_W_NUM);
    param.vfElemB16 = VEC_MAX_ELEM_B16;
    param.nBubTail = param.nBubWTypeAlign % VEC_MAX_ELEM_B16;
    param.nLoop = CeilDiv(param.nBubWTypeAlign, VEC_MAX_ELEM_B16);
    param.groupSize = tiling_->groupSize;
    param.groupNum = CeilDiv(param.kBub, param.groupSize) - 1;
    param.groupTail = param.kBub % param.groupSize > 0 ? param.kBub % param.groupSize : param.groupSize;
    param.dataBlockStride = CeilAlign(param.kBub, BLOCK_CUBE) + 1;
    param.weightOutStride = param.dataBlockStride * (param.vfElemB16 - BLOCK_CUBE) + BLOCK_CUBE;
    param.repeatStride = 1; // unit: 32B

    param.offsetBaseAddr = offsetBaseAddr1_;
    param.scaleBaseAddr = scaleBaseAddr1_;
    param.weightInBaseAddr = weightInUbBaseAddr_;
    param.weightOutBaseAddr = weightOutUbAddr_;
    param.offsetTailAddr = param.offsetBaseAddr + param.groupNum * param.nBubXTypeAlign;
    param.scaleTailAddr = param.scaleBaseAddr + param.groupNum * param.nBubXTypeAlign;
    if constexpr (IsSameType<wType, int8_t>::value) {
        param.wNStride = VEC_MAX_ELEM_B16;
        param.wKStride = param.nBubWTypeAlign;
        param.wGroupStride = param.groupSize * param.nBubWTypeAlign;
        param.weightInGroupTailAddr = param.weightInBaseAddr + param.groupNum * param.groupSize * param.nBubWTypeAlign;
    } else {
        param.wNStride = VEC_MAX_ELEM_B16 >> INT4_DTYPE_PARAM;
        param.wKStride = param.nBubWTypeAlign >> INT4_DTYPE_PARAM;
        param.wGroupStride = param.groupSize * param.nBubWTypeAlign >> INT4_DTYPE_PARAM;
        param.weightInGroupTailAddr =
            param.weightInBaseAddr +
            (param.groupNum * param.groupSize * param.nBubWTypeAlign) / 2; // 2: int4按int8_t计数
    }

    AscendC::VF_CALL<AntiquantPerGroupKN<xType, wType, hasAntiQuantOffset>>(param);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::AntiquantComputeW4NKPerGroup(int32_t bubNLen,
                                                                                                      int32_t bubKLen)
{
    if (tiling_->groupSize == 32) {
        ParamsGroupSize32<xType, wType> params;

        params.dataBlockStride = CeilAlign(bubNLen, ONE_BLK_X_NUM) + 1;
        params.repeatStride = 1; // unit: 32B

        params.outerExtend = bubNLen;
        params.outerStrideScale = CeilAlign(groupNumBub_, ONE_BLK_X_NUM);
        params.outerStrideWeight = CeilAlign(bubKLen, ONE_BLK_W_NUM) >> INT4_DTYPE_PARAM;

        params.innerExtend = CeilDiv(bubKLen, 2 * VEC_MAX_ELEM_B16);

        params.outDimOffset = 2 * VEC_MAX_ELEM_B16 * (CeilAlign(bubNLen, ONE_BLK_X_NUM) + 1) - bubNLen * ONE_BLK_X_NUM;

        params.offsetBaseAddr0 = offsetBaseAddr1_;
        params.scaleBaseAddr0 = scaleBaseAddr1_;
        params.weightInBaseAddr = weightInUbBaseAddr_;
        params.weightOutBaseAddr0 = weightOutUbAddr_;
        params.weightOutBaseAddr1 = weightOutUbAddr_ + params.dataBlockStride * VEC_MAX_ELEM_B16;

        params.maskWeight =
            Min(bubKLen - FloorAlign(bubKLen, 2 * VEC_MAX_ELEM_B16), static_cast<int64_t>(VEC_MAX_ELEM_B16)) +
            bubKLen / (2 * VEC_MAX_ELEM_B16) * VEC_MAX_ELEM_B16;
        params.maskWeight1 =
            Max(bubKLen - FloorAlign(bubKLen, 2 * VEC_MAX_ELEM_B16) - VEC_MAX_ELEM_B16, static_cast<int64_t>(0)) +
            bubKLen / (2 * VEC_MAX_ELEM_B16) * VEC_MAX_ELEM_B16;
        if (params.innerExtend == 1 || params.outerExtend == 1) {
            AscendC::VF_CALL<AntiquantW4Pergroup32NK<xType, wType, hasAntiQuantOffset, false>>(params);
        } else {
            AscendC::VF_CALL<AntiquantW4Pergroup32NK<xType, wType, hasAntiQuantOffset, true>>(params);
        }
    } else if (tiling_->groupSize == 64) {
        ParamsGroupSize64<xType, wType> params;
        params.outerExtend = bubNLen;
        params.innerExtend = CeilDiv(bubKLen, VEC_MAX_ELEM_B16);

        params.dataBlockStride = CeilAlign(bubNLen, BLOCK_CUBE) + 1; // uint: 32B
        params.repeatStride = 1;                                     // unit: 32B
        params.outerStride = VEC_MAX_ELEM_B16 * (CeilAlign(bubNLen, BLOCK_CUBE) + 1) - bubNLen * ONE_BLK_X_NUM;
        params.outerStrideScale = CeilAlign(groupNumBub_, ONE_BLK_X_NUM);
        params.outerStrideWeight = CeilAlign(bubKLen, ONE_BLK_W_NUM) >> 1; // unit: sizeof(uint8_t)

        params.offsetBaseAddr00 = offsetBaseAddr1_ + 0;
        params.offsetBaseAddr01 = offsetBaseAddr1_ + 1;

        params.scaleBaseAddr00 = scaleBaseAddr1_ + 0;
        params.scaleBaseAddr01 = scaleBaseAddr1_ + 1;

        params.weightInBaseAddr0 = weightInUbBaseAddr_;
        params.weightOutBaseAddr0 = weightOutUbAddr_;

        params.maskWeight = bubKLen;
        if (params.innerExtend == 1 || params.outerExtend == 1) {
            AscendC::VF_CALL<AntiquantW4Pergroup64NK<xType, wType, hasAntiQuantOffset, false>>(params);
        } else {
            AscendC::VF_CALL<AntiquantW4Pergroup64NK<xType, wType, hasAntiQuantOffset, true>>(params);
        }
    } else if (tiling_->groupSize == 128) {
        ParamsGroupSize128And256<xType, wType> params;

        params.dataBlockStride = CeilAlign(bubNLen, ONE_BLK_X_NUM) + 1; // unit: 32B
        params.repeatStride = 1;                                        // unit: 32B
        params.weightOutAddrOffset =
            VEC_MAX_ELEM_B16 * (CeilAlign(bubNLen, ONE_BLK_X_NUM) + 1) - bubNLen * ONE_BLK_X_NUM;

        params.bubNLen = bubNLen;

        params.groupNum = CeilDiv(bubKLen, tiling_->groupSize);
        params.outerStrideScale = CeilAlign(groupNumBub_, ONE_BLK_X_NUM);
        params.outerStrideWeight = CeilAlign(bubKLen, ONE_BLK_W_NUM) >> 1; // unit: sizeof(uint8_t)
        params.innerStrideWeight = 64;
        params.weightInBaseAddr = weightInUbBaseAddr_;
        params.weightOutBaseAddr = weightOutUbAddr_;
        params.scaleBaseAddr = scaleBaseAddr1_;
        params.offsetBaseAddr = offsetBaseAddr1_;

        params.maskWeight = bubKLen;
        if (params.groupNum == 1 || params.bubNLen == 1) {
            AscendC::VF_CALL<AntiquantW4Pergroup128NK<xType, wType, hasAntiQuantOffset, false>>(params);
        } else {
            AscendC::VF_CALL<AntiquantW4Pergroup128NK<xType, wType, hasAntiQuantOffset, true>>(params);
        }
    } else if (tiling_->groupSize == 256) {
        ParamsGroupSize128And256<xType, wType> params;

        params.dataBlockStride = CeilAlign(bubNLen, ONE_BLK_X_NUM) + 1; // unit: 32B
        params.repeatStride = 1;                                        // unit: 32B
        params.weightOutAddrOffset =
            2 * VEC_MAX_ELEM_B16 * (CeilAlign(bubNLen, ONE_BLK_X_NUM) + 1) - bubNLen * ONE_BLK_X_NUM;

        params.bubNLen = bubNLen;

        params.groupNum = CeilDiv(bubKLen, tiling_->groupSize);
        params.outerStrideScale = CeilAlign(groupNumBub_, ONE_BLK_X_NUM);
        params.outerStrideWeight = CeilAlign(bubKLen, ONE_BLK_W_NUM) >> 1; // unit: sizeof(uint8_t)
        params.innerStrideWeight = 2 * 64;
        params.weightInBaseAddr = weightInUbBaseAddr_;
        params.weightInBaseAddr1 = weightInUbBaseAddr_ + 64;
        params.weightOutBaseAddr = weightOutUbAddr_;
        params.weightOutBaseAddr1 = weightOutUbAddr_ + VEC_MAX_ELEM_B16 * (CeilAlign(bubNLen, ONE_BLK_X_NUM) + 1);
        params.scaleBaseAddr = scaleBaseAddr1_;
        params.offsetBaseAddr = offsetBaseAddr1_;

        params.maskWeight = Min(bubKLen - bubKLen / 256 * 256, 128) + bubKLen / 256 * 128;
        params.maskWeight1 = Max(bubKLen - bubKLen / 256 * 256 - 128, 0) + bubKLen / 256 * 128;
        if (params.groupNum == 1 || params.bubNLen == 1) {
            AscendC::VF_CALL<AntiquantW4Pergroup256NK<xType, wType, hasAntiQuantOffset, false>>(params);
        } else {
            AscendC::VF_CALL<AntiquantW4Pergroup256NK<xType, wType, hasAntiQuantOffset, true>>(params);
        }
        // group非64对齐场景
    } else if (tiling_->groupSize % 64 > 0) {
        // border部分的计算长度为96
        constexpr uint32_t CROSS_LEN = 96;
        ParamsGroupSize32OddNK<xType> params;
        // loop extent
        // 以2*gs粒度进行处理
        params.groupPairNum = CeilDiv(bubKLen, tiling_->groupSize * 2);
        params.bubNLen = bubNLen;
        params.oddGroupVLNum = CeilDiv(Min(bubKLen, static_cast<int32_t>(tiling_->groupSize)), VEC_MAX_ELEM_B16);
        // 1) bubKLen <= gs, 无border及evenGroup
        // 2）bubKLen > gs, tiling保证bubKLen为gs的偶数倍, 此时even group部分已计算的96长度需要扣除
        params.evenGroupVLNum =
            bubKLen > tiling_->groupSize ? CeilDiv(tiling_->groupSize - CROSS_LEN, VEC_MAX_ELEM_B16) : 0;

        // loop stride
        params.scaleNStride = CeilAlign(groupNumBub_, ONE_BLK_X_NUM);
        params.weightNStride = CeilAlign(bubKLen, ONE_BLK_W_NUM) >> INT4_DTYPE_PARAM;
        // 以2*group为粒度处理
        params.scaleGroupPairStride = 2;
        params.weightGroupPairStride = params.scaleGroupPairStride * tiling_->groupSize >> INT4_DTYPE_PARAM;
        params.weightVLStride = VEC_MAX_ELEM_B16 >> INT4_DTYPE_PARAM;

        // vsstb
        params.dataBlockStride = CeilAlign(bubNLen, BLOCK_CUBE) + 1; // unit 32B
        params.repeatStride = (VEC_MAX_ELEM_B16 / ONE_BLK_X_NUM) * params.dataBlockStride;
        params.offsetNWeightOutOdd = params.oddGroupVLNum * params.repeatStride * ONE_BLK_X_NUM - ONE_BLK_X_NUM;
        params.offsetNWeightOutEven = params.evenGroupVLNum * params.repeatStride * ONE_BLK_X_NUM - ONE_BLK_X_NUM;
        params.offsetNWeightOutBorder = params.repeatStride * ONE_BLK_X_NUM - ONE_BLK_X_NUM;
        params.offsetKWeightOut =
            params.scaleGroupPairStride * tiling_->groupSize * (CeilAlign(bubNLen, BLOCK_CUBE) + 1) -
            ONE_BLK_X_NUM * bubNLen;

        // addr init
        params.offsetOddAddr = offsetBaseAddr1_;
        params.scaleOddAddr = scaleBaseAddr1_;
        params.weightInOddAddr = weightInUbBaseAddr_;
        params.weightOutOddAddr = weightOutUbAddr_;
        params.weightInBorderAddr =
            weightInUbBaseAddr_ + (FloorAlign(tiling_->groupSize, ONE_BLK_W_NUM) >> INT4_DTYPE_PARAM);
        params.weightOutBorderAddr =
            weightOutUbAddr_ + FloorAlign(tiling_->groupSize, ONE_BLK_W_NUM) * (CeilAlign(bubNLen, BLOCK_CUBE) + 1);

        params.offsetEvenAddr = offsetBaseAddr1_ + 1;
        params.scaleEvenAddr = scaleBaseAddr1_ + 1;
        params.weightInEvenAddr =
            weightInUbBaseAddr_ + ((CeilAlign(tiling_->groupSize, ONE_BLK_W_NUM) + ONE_BLK_W_NUM) >> INT4_DTYPE_PARAM);
        params.weightOutEvenAddr = weightOutUbAddr_ + (CeilAlign(tiling_->groupSize, ONE_BLK_W_NUM) + ONE_BLK_W_NUM) *
                                                          (CeilAlign(bubNLen, BLOCK_CUBE) + 1);

        // mask
        params.maskWeightOdd = Min(bubKLen, static_cast<int32_t>(tiling_->groupSize));
        params.maskWeightEven = tiling_->groupSize - CROSS_LEN;
        if ((tiling_->groupSize - CROSS_LEN) % VEC_MAX_ELEM_B16 == 0) {
            if (bubKLen > tiling_->groupSize) {
                if (params.evenGroupVLNum == 1) {
                    AscendC::VF_CALL<AntiquantW4Pergroup32OddNK<xType, wType, hasAntiQuantOffset, false, true, false>>(
                        params);
                } else {
                    AscendC::VF_CALL<AntiquantW4Pergroup32OddNK<xType, wType, hasAntiQuantOffset, false, true, true>>(
                        params);
                }
            } else {
                AscendC::VF_CALL<AntiquantW4Pergroup32OddNK<xType, wType, hasAntiQuantOffset, false, false, true>>(
                    params);
            }
        } else {
            if (bubKLen > tiling_->groupSize) {
                if (params.evenGroupVLNum == 1) {
                    AscendC::VF_CALL<AntiquantW4Pergroup32OddNK<xType, wType, hasAntiQuantOffset, true, true, false>>(
                        params);
                } else {
                    AscendC::VF_CALL<AntiquantW4Pergroup32OddNK<xType, wType, hasAntiQuantOffset, true, true, true>>(
                        params);
                }
            } else {
                AscendC::VF_CALL<AntiquantW4Pergroup32OddNK<xType, wType, hasAntiQuantOffset, true, false, true>>(
                    params);
            }
        }
    } else { // group size > 128 && group size != 256，功能模板
        ParamsGroupSizeGt128<xType, wType> params;
        params.innerExtend = CeilAlign(bubKLen, 64) >> INT4_DTYPE_PARAM;
        params.bubNLen = bubNLen;
        params.vlB4SizeInByte = VEC_MAX_ELEM_B16 >> INT4_DTYPE_PARAM;
        params.dataBlockStride = CeilAlign(bubNLen, BLOCK_CUBE) + 1;
        params.repeatStride0 = params.dataBlockStride * VEC_MAX_ELEM_B16 / BLOCK_CUBE;
        params.offsetBaseAddr = offsetBaseAddr1_;
        params.scaleBaseAddr = scaleBaseAddr1_;
        params.weightInBaseAddr = weightInUbBaseAddr_;
        params.weightOutBaseAddr0 = weightOutUbAddr_;
        params.oriWeightOutBaseAddr0 = weightOutUbAddr_;

        if (tiling_->kBubSize >= tiling_->groupSize) {
            params.groupSizeInByte = tiling_->groupSize >> INT4_DTYPE_PARAM;
            params.groupSize = tiling_->groupSize;
            params.oriGroupSize = tiling_->groupSize;
            params.groupNumBub = CeilAlign(groupNumBub_, BLOCK_CUBE);
            params.numVLInGroup = tiling_->groupSize / VEC_MAX_ELEM_B16;
            params.resGrpMod128 = tiling_->groupSize % VEC_MAX_ELEM_B16;
            params.tailVLInGroup = params.resGrpMod128 == 0 ? 0 : 1;

            params.tailKLen = bubKLen % tiling_->groupSize; // bubKLen 相对于 group size 的尾块大小
            params.numVLInRemainGroup =
                params.tailKLen / VEC_MAX_ELEM_B16; // bubKLen 相对于 group_size 的尾块中 VL 的数量
            params.resRemainGrpMod128 = params.tailKLen % VEC_MAX_ELEM_B16; // 一个不完整的 group 相对于 128 的尾块大小
            params.tailGroupInBubKLen = params.tailKLen == 0 ? 0 : 1;
            params.tailVLInTailGroup = params.resRemainGrpMod128 == 0 ? 0 : 1;

            params.mainGroupNum = bubKLen / tiling_->groupSize; // 一个 kBubSize 中 group 的个数
            params.mainGroupSize = params.mainGroupNum * tiling_->groupSize;

            params.weightOutBaseAddr1 =
                weightOutUbAddr_ + params.dataBlockStride * params.groupSize * params.mainGroupNum;
            params.oriWeightOutBaseAddr1 = params.weightOutBaseAddr1;
            params.repeatStride1 = params.dataBlockStride * CeilAlign(params.resGrpMod128, BLOCK_CUBE) / BLOCK_CUBE;
            // kBubSize >= group_size > 128, group_size 非 128、256, 功能模板
            if (params.tailVLInGroup && params.tailVLInTailGroup) {
                AscendC::VF_CALL<AntiquantW4PergroupGt128NKCase1<xType, wType, hasAntiQuantOffset, true, true>>(params);
            } else if (params.tailVLInGroup) {
                AscendC::VF_CALL<AntiquantW4PergroupGt128NKCase1<xType, wType, hasAntiQuantOffset, true, false>>(
                    params);
            } else if (params.tailVLInTailGroup) {
                AscendC::VF_CALL<AntiquantW4PergroupGt128NKCase1<xType, wType, hasAntiQuantOffset, false, true>>(
                    params);
            } else {
                AscendC::VF_CALL<AntiquantW4PergroupGt128NKCase1<xType, wType, hasAntiQuantOffset, false, false>>(
                    params);
            }
        } else { // tiling_->kBubSize < params.groupSize
            params.numVLInKLen = bubKLen / VEC_MAX_ELEM_B16;
            params.tailKLen = bubKLen % VEC_MAX_ELEM_B16; // bubKLen 相对于 128 的尾块大小
            params.tailVLInKLen = params.tailKLen == 0 ? 0 : 1;
            AscendC::VF_CALL<AntiquantW4PergroupGt128NKCase2<xType, wType, hasAntiQuantOffset>>(params);
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::AntiquantComputeW8NKPerGroup(int32_t bubNLen,
                                                                                                      int32_t bubKLen)
{
#ifdef __CCE_KT_TEST__
    ASCENDC_ASSERT(tiling_->groupSize > 0, { KERNEL_LOG(KERNEL_ERROR, "Invalid groupSize."); });
#endif
    ParamsW8NK<xType> param;
    param.groupSize = tiling_->groupSize;
    param.groupNum = CeilDiv(bubKLen, param.groupSize) - 1;
    param.groupTail = bubKLen % param.groupSize > 0 ? bubKLen % param.groupSize : param.groupSize;
    param.vlNum = CeilDiv(param.groupSize, VEC_MAX_ELEM_B16);
    param.tailVlNum = CeilDiv(param.groupTail, VEC_MAX_ELEM_B16);
    param.nBub = bubNLen;
    param.kBubXTypeAlign = CeilAlign(bubKLen, ONE_BLK_X_NUM);
    param.kBubWTypeAlign = CeilAlign(bubKLen, ONE_BLK_W_NUM);
    param.vfElemB16 = VEC_MAX_ELEM_B16;
    param.scaleNStride = CeilAlign(CeilDiv(bubKLen, param.groupSize), ONE_BLK_X_NUM);

    param.dataBlockStride = CeilAlign(param.nBub, BLOCK_CUBE) + 1;
    // 8含义：reg width对应8个block
    param.repeatStride = param.dataBlockStride * Min(8, static_cast<int32_t>(param.groupSize / ONE_BLK_X_NUM));
    param.weightOutFixStride = ONE_BLK_X_NUM - param.vlNum * param.repeatStride * ONE_BLK_X_NUM;
    param.weightOutGroupFixStride = param.dataBlockStride * param.groupSize - param.nBub * ONE_BLK_X_NUM;
    param.weightOutTailFixStride = ONE_BLK_X_NUM - param.tailVlNum * param.repeatStride * ONE_BLK_X_NUM;

    param.offsetBaseAddr = offsetBaseAddr1_;
    param.scaleBaseAddr = scaleBaseAddr1_;
    param.weightInBaseAddr = weightInUbBaseAddr_;
    param.weightOutBaseAddr = weightOutUbAddr_;

    param.offsetTailAddr = param.offsetBaseAddr + param.groupNum;
    param.scaleTailAddr = param.scaleBaseAddr + param.groupNum;
    param.weightInTailAddr = param.weightInBaseAddr + param.groupNum * param.groupSize;
    param.weightOutTailAddr = param.weightOutBaseAddr + param.dataBlockStride * param.groupNum * param.groupSize;

    AscendC::VF_CALL<AntiquantW8PerGroupNK<xType, wType, hasAntiQuantOffset>>(param);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::AntiQuantCompute(int32_t bubKOffset,
                                                                                          int32_t bubNLen,
                                                                                          int32_t bubKLen)
{
    // 以下表征 offset 的变量单位均为元素个数
    uint64_t weightOutUbOffset;
    uint64_t weightInUbOffset; // B矩阵反量化前数据类型为 INT4 时, 在 UB 上的存储格式仍是 INT8
    uint64_t scaleUbOffset;

    weightInUbOffset = ubInBufIdx_ * vecWeightInSize_;
    scaleUbOffset = ubInBufIdx_ * vecScaleOffsetSize_;
    if constexpr (weightNz) {
        weightOutUbOffset = ubOutBufIdx_ * VEC_MAX_ELEM_B16; // 一次处理256B
    } else {
        weightOutUbOffset = ubOutBufIdx_ * vecWeightOutSize_;
    }

    weightInUbBaseAddr_ = (__local_mem__ int8_t *)weightInUb_[weightInUbOffset].GetPhyAddr();
    if constexpr (antiQuantType != Mc2QuantType::PER_TENSOR) {
        scaleBaseAddr1_ = (__local_mem__ xType *)scaleInUb_[scaleUbOffset].GetPhyAddr();
        offsetBaseAddr1_ = (__local_mem__ xType *)offsetInUb_[scaleUbOffset].GetPhyAddr();
    }
    weightOutUbAddr_ = (__local_mem__ xType *)weightOutUb_[weightOutUbOffset].GetPhyAddr();

    if constexpr (bTrans) {
        if constexpr (IsSameType<wType, int4b_t>::value && antiQuantType == Mc2QuantType::PER_GROUP && !weightNz) {
            AntiquantComputeW4NKPerGroup(bubNLen, bubKLen);
        } else if constexpr (IsSameType<wType, int8_t>::value && antiQuantType == Mc2QuantType::PER_GROUP &&
                             !weightNz) {
            AntiquantComputeW8NKPerGroup(bubNLen, bubKLen);
        } else {
            // this template only supports pergroup mode.
        }
    } else { // B 矩阵非转置
        if constexpr (IS_4BIT_WEIGHT && antiQuantType == Mc2QuantType::PER_GROUP && weightNz) {
            AntiQuantComputeKNGroupWeightNz(bubKOffset, bubNLen, bubKLen);
        } else if constexpr (antiQuantType == Mc2QuantType::PER_GROUP && !weightNz) {
            // A16W4 ND perGroup, A16W8 ND perGroup
            AntiquantComputeKNPerGroup(bubNLen, bubKLen);
        } else {
            // this template only supports pergroup mode.
        }
    }
}
template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::CopyVecOut2L1(int64_t l1Offset,
                                                                                       LocalTensor<xType> ubLocal,
                                                                                       int32_t bubKLen, int32_t bubNLen)
{
    DataCopyParams params;
    if constexpr (bTrans) {
        if constexpr (USE_VSSTB) {
            // (k1, n1, n0, k0)
            // 1. 固定1，bubNLen=8x-1时写到ub时会完全冲突
            // 2. nBL1Len_尾块会有精度问题
            params.blockLen = bubNLen;
            params.blockCount = CeilDiv(bubKLen, BLOCK_CUBE);
            params.srcStride = 1 + CeilAlign(bubNLen, BLOCK_CUBE) - bubNLen; // solve bank confilict
            params.dstStride = CeilAlign(nBL1Len_, BLOCK_CUBE) - bubNLen;
            DataCopy(l1Local_[l1Offset], ubLocal, params);
        } else {
            // "only support transpose_weight=True in s8"
        }
    } else { // B 矩阵非转置
        if constexpr (USE_VSSTB) {
            params.blockLen = bubKLen;
            params.blockCount = CeilDiv(bubNLen, BLOCK_CUBE);
            params.srcStride = 1 + CeilAlign(bubKLen, BLOCK_CUBE) - bubKLen; // solve bank confilict
            params.dstStride = twoVectorCoreSplitK_ ? (CeilAlign(kBL1Len_, BLOCK_CUBE) - bubKLen) : 0;
            DataCopy(l1Local_[l1Offset], ubLocal, params);
        } else { // A16W4, NZ 格式输入, (n1, k1, k0, n0)
            params.blockLen = BLOCK_NUM_REG;
            if (twoVectorCoreSplitK_) {
                // l1 (n1, 2*k1, k0, n0)
                // v0 (n1, k1, k0, n0)
                // v1 (n1, k1, k0, n0)
                if (bubKLen < bubNLen) {
                    params.blockCount = (BLOCK_CUBE >> 1) * bubNLen * sizeof(xType) / VECTOR_REG_WIDTH;
                    params.srcStride = (CeilDiv(vecKBL1Len_, BLOCK_CUBE) * 2 - 1) * BLOCK_NUM_REG;
                    params.dstStride = (kBL1Len_ * BLOCK_CUBE - VECTOR_REG_WIDTH) / ONE_BLK_SIZE;
                    for (int32_t idxK = 0; idxK < CeilDiv(bubKLen, BLOCK_CUBE >> 1); idxK++) {
                        DataCopy(l1Local_[l1Offset], ubLocal, params);
                        l1Offset += VECTOR_REG_WIDTH;
                    }
                } else {
                    params.blockCount = bubKLen * BLOCK_CUBE * sizeof(xType) / VECTOR_REG_WIDTH;
                    params.srcStride = (bubpingpong_ - 1) * BLOCK_NUM_REG;
                    params.dstStride = 0;
                    for (int32_t idxN = 0; idxN < CeilDiv(bubNLen, BLOCK_CUBE); idxN++) {
                        DataCopy(l1Local_[l1Offset], ubLocal, params);
                        l1Offset += (kBL1Len_ - bubKLen) * BLOCK_CUBE / ONE_BLK_SIZE;
                    }
                }
            } else {
                // 一块 bubNLen * bubKLen一共有多少256B
                params.blockCount = bubNLen * bubKLen * sizeof(xType) / VECTOR_REG_WIDTH;
                // 每256B跳1024B(4buffer) / 512B(2buffer)
                params.srcStride = (bubpingpong_ - 1) * BLOCK_NUM_REG;
                params.dstStride = 0; // dst地址连续

                DataCopy(l1Local_[l1Offset], ubLocal, params);
            }
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::VectorRegCompute(int64_t bubKOffset,
                                                                                          int32_t bubKLen,
                                                                                          int32_t bubNLen)
{
    WaitFlag<HardEvent::MTE3_V>(ubOutBufIdx_);
    WaitFlag<HardEvent::MTE2_V>(ubInBufIdx_);
    AntiQuantCompute(bubKOffset, bubNLen, bubKLen);
    SetFlag<HardEvent::V_MTE3>(ubOutBufIdx_);
}

/**
 * 只支持在n方向分成两个任务给AIV0，AIV1
 * 当只有一个任务时，AIV0和AIV1执行同样的任务
 */
template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::BL1ProcessNK1Vs1(uint64_t curBL1BufIdx,
                                                                                          int64_t nBL1Offset,
                                                                                          int64_t kBL1Offset,
                                                                                          int32_t kL1Len,
                                                                                          int32_t nL0Len)
{
    idx_ += 1;
    if constexpr (IsSameType<wType, int4b_t>::value && weightNz) {
        ubInBufIdx_ = idx_ % bubpingpong_;
        ubOutBufIdx_ = ubInBufIdx_;
    } else if constexpr (!weightNz &&
                         antiQuantType == Mc2QuantType::PER_GROUP) { // A16W8 ND-perGroup, A16W4-ND-perGroup
        ubInBufIdx_ = idx_ % bubpingpong_;
        ubOutBufIdx_ = idx_ % bl1pingpong_;
    }
    WaitFlag<HardEvent::V_MTE2>(ubInBufIdx_);

    int32_t bubKLen = Min(vecKBL1Len_, static_cast<int64_t>(tiling_->kBubSize));
    groupNumBub_ = GetVecGn<antiQuantType, weightNz>(bubKLen, tiling_->groupSize);
    int32_t bubNLen = Min(vecNBL1Len_, static_cast<int64_t>(tiling_->nBubSize));
    CopyInWeight(kBL1Offset, nBL1Offset, bubKLen, bubNLen);
    CopyInScaleOffset(nBL1Offset, bubNLen, 0, 0, kBL1Offset, 1);
    SetFlag<HardEvent::MTE2_V>(ubInBufIdx_);
    VectorRegCompute(kBL1Offset, bubKLen, bubNLen);

    SetFlag<HardEvent::V_MTE2>(ubInBufIdx_);
    WaitFlag<HardEvent::V_MTE3>(ubOutBufIdx_);

    uint32_t l1BufOffset = curBL1BufIdx * bL1DataSize_;
    int64_t l1Offset = l1BufOffset;
    if (AscendC::GetSubBlockIdx() == 1 && nBL1Len_ > tiling_->nBubSize) {
        l1Offset += AscendC::GetSubBlockIdx() * tiling_->nBubSize * BLOCK_CUBE;
    }

    CopyVecOut2L1(l1Offset, weightOutUb_[ubOutBufIdx_ * vecWeightOutSize_], bubKLen, bubNLen);
    SetFlag<HardEvent::MTE3_V>(ubOutBufIdx_);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::BL1ProcessNK1VsN(uint64_t curBL1BufIdx,
                                                                                          int64_t nBL1Offset,
                                                                                          int64_t kBL1Offset,
                                                                                          int32_t kL1Len,
                                                                                          int32_t nL0Len)
{
    int64_t bubNFactor = CeilDiv(vecNBL1Len_, tiling_->nBubSize);
    int64_t bubKFactor = CeilDiv(vecKBL1Len_, tiling_->kBubSize);
    for (int32_t bubNLoopIdx = 0; bubNLoopIdx < bubNFactor; bubNLoopIdx++) {
        int64_t vecNBL1Offset = bubNLoopIdx * tiling_->nBubSize;
        int64_t bubNOffset = nBL1Offset + vecNBL1Offset;
        int32_t bubNLen = Min(vecNBL1Len_ - vecNBL1Offset, static_cast<int64_t>(tiling_->nBubSize));
        for (int32_t bubKLoopIdx = 0; bubKLoopIdx < bubKFactor; bubKLoopIdx++) {
            int64_t vecKBL1Offset = bubKLoopIdx * tiling_->kBubSize;
            int64_t bubKOffset = kBL1Offset + vecKBL1Offset;

            idx_ += 1;
            if constexpr (IsSameType<wType, int4b_t>::value && weightNz) {
                ubInBufIdx_ = idx_ % bubpingpong_;
            } else if constexpr (IsSameType<wType, int4b_t>::value && !weightNz) {
                ubInBufIdx_ = idx_ % bubpingpong_;
            } else { // A16W8
                ubInBufIdx_ = idx_ % vecPingpong_;
            }
            ubOutBufIdx_ = ubInBufIdx_;
            WaitFlag<HardEvent::V_MTE2>(ubInBufIdx_);

            int32_t bubKLen = Min(vecKBL1Len_ - vecKBL1Offset, static_cast<int64_t>(tiling_->kBubSize));
            groupNumBub_ = GetVecGn<antiQuantType, weightNz>(bubKLen, tiling_->groupSize);
            CopyInWeight(bubKOffset, bubNOffset, bubKLen, bubNLen);
            CopyInScaleOffset(bubNOffset, bubNLen, bubNLoopIdx, bubKLoopIdx, bubKOffset, bubKFactor);
            SetFlag<HardEvent::MTE2_V>(ubInBufIdx_);
            VectorRegCompute(bubKOffset, bubKLen, bubNLen);

            SetFlag<HardEvent::V_MTE2>(ubInBufIdx_);
            WaitFlag<HardEvent::V_MTE3>(ubOutBufIdx_);
            if constexpr (IsSameType<wType, int4b_t>::value && weightNz) {
                ubOutBufIdx_ = ubInBufIdx_;
            } else if constexpr (IsSameType<wType, int4b_t>::value && !weightNz) {
                ubOutBufIdx_ = idx_ % bl1pingpong_;
            } else { // A16W8
                ubOutBufIdx_ = ubInBufIdx_;
            }

            int64_t kl1Offset = vecKBL1Offset;
            if (AscendC::GetSubBlockIdx() == 1 && tiling_->vecCoreParallel == 1) {
                kl1Offset += tiling_->kBubSize;
            }
            int64_t l1Offset = nBL1Len_ * kl1Offset + vecNBL1Offset * BLOCK_CUBE + curBL1BufIdx * bL1DataSize_;
            CopyVecOut2L1(l1Offset, weightOutUb_[ubOutBufIdx_ * vecWeightOutSize_], bubKLen, bubNLen);
            SetFlag<HardEvent::MTE3_V>(ubOutBufIdx_);
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::BL1ProcessNK(uint64_t curBL1BufIdx,
                                                                                      int64_t nBL1Offset,
                                                                                      int64_t kBL1Offset,
                                                                                      int32_t kL1Len, int32_t nL0Len)
{
    if constexpr (!weightNz && antiQuantType == Mc2QuantType::PER_GROUP) {
        // A16W4-ND-perGroup, A16W8-ND-perGroup
        BL1ProcessNK1Vs1(curBL1BufIdx, nBL1Offset, kBL1Offset, kL1Len, nL0Len);
    } else if constexpr (IsSameType<wType, int8_t>::value) {
        // A16W8-ND-perChannel
        BL1ProcessNK1VsN(curBL1BufIdx, nBL1Offset, kBL1Offset, kL1Len, nL0Len);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::BL1ProcessKN1Vs1(uint64_t curBL1BufIdx,
                                                                                          int64_t nBL1Offset,
                                                                                          int64_t kBL1Offset)
{
    // l1 space: bp0 bp1
    int64_t l1Offset = curBL1BufIdx * bL1DataSize_;
    if constexpr (IS_4BIT_WEIGHT && weightNz) {
        if constexpr (antiQuantType == Mc2QuantType::PER_GROUP && !bTrans) {
            if (bl1pingpong_ == QUADRUPLE_BUFFER) {
                l1Offset = (curBL1BufIdx & 0x1) *
                               Max(L1_BUFFER_HALF_SIZE / sizeof(xType), DOUBLE_BUFFER * bL1DataSize_ + aL1DataSize_) +
                           ((curBL1BufIdx & 0x2) > 1) * bL1DataSize_;
            }
        } else {
            // "not support this scenario"
        }
    }

    // AIV-0 / AIV-1 上一块 UB buffer 对应于一块 BL1 buffer 的搬运与计算过程
    int32_t bubKLen = Min(vecKBL1Len_, static_cast<int64_t>(tiling_->kBubSize));
    groupNumBub_ = GetVecGn<antiQuantType, weightNz>(bubKLen, tiling_->groupSize);

    idx_ += 1;
    if constexpr (antiQuantType == Mc2QuantType::PER_GROUP) { // A16W4 ND-perGroup, A16W4 NZ-perGroup, A16W8 ND-perGroup
        ubInBufIdx_ = idx_ % bubpingpong_;
        ubOutBufIdx_ = idx_ % bl1pingpong_;
    }

    WaitFlag<HardEvent::V_MTE2>(ubInBufIdx_);

    int32_t bubNLen = Min(vecNBL1Len_, static_cast<int64_t>(tiling_->nBubSize));
    CopyInWeight(kBL1Offset, nBL1Offset, bubKLen, bubNLen);
    CopyInScaleOffset(nBL1Offset, bubNLen, 0, 0, kBL1Offset, 1);
    SetFlag<HardEvent::MTE2_V>(ubInBufIdx_);
    VectorRegCompute(kBL1Offset, bubKLen, bubNLen);

    SetFlag<HardEvent::V_MTE2>(ubInBufIdx_);
    WaitFlag<HardEvent::V_MTE3>(ubOutBufIdx_);

    // (n1, k1, k0, n0)
    if constexpr (antiQuantType == Mc2QuantType::PER_GROUP) {
        int64_t nl1Offset = 0;
        int64_t kl1Offset = 0;
        if (AscendC::GetSubBlockIdx() == 1 && twoVectorCoreSplitK_ && kBL1Len_ > bubKLen) {
            // 存在尾块kBL1Len_= bubKLen的情况，此时两个vec核处理同一块BL1buffer,往同一个地址搬运
            kl1Offset += CeilDiv(CeilDiv(kBL1Len_, tiling_->kBubSize), 2) * tiling_->kBubSize;
        } else if (AscendC::GetSubBlockIdx() == 1 && twoVectorCoreSplitN_ && nBL1Len_ > bubNLen) {
            nl1Offset += CeilDiv(CeilDiv(nBL1Len_, tiling_->nBubSize), 2) * tiling_->nBubSize;
        }
        l1Offset += nl1Offset * CeilAlign(kBL1Len_, BLOCK_CUBE) + kl1Offset * BLOCK_CUBE;
    } else {
        // not support this scenario
    }

    if constexpr (IS_4BIT_WEIGHT && antiQuantType == Mc2QuantType::PER_GROUP && weightNz) {
        CopyVecOut2L1(l1Offset, weightOutUb_[ubOutBufIdx_ * VEC_MAX_ELEM_B16], bubKLen, bubNLen);
    } else {
        CopyVecOut2L1(l1Offset, weightOutUb_[ubOutBufIdx_ * vecWeightOutSize_], bubKLen, bubNLen);
    }
    SetFlag<HardEvent::MTE3_V>(ubOutBufIdx_);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::BL1ProcessKN(uint64_t curBL1BufIdx,
                                                                                      int64_t nBL1Offset,
                                                                                      int64_t kBL1Offset,
                                                                                      int32_t kL1Len, int32_t nL0Len)
{
    BL1ProcessKN1Vs1(curBL1BufIdx, nBL1Offset, kBL1Offset);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::BL1Process(uint64_t curBL1BufIdx,
                                                                                    int64_t nBL1Offset,
                                                                                    int64_t kBL1Offset, int32_t kL1Len,
                                                                                    int32_t nL0Len)
{
    if constexpr (bTrans) {
        BL1ProcessNK(curBL1BufIdx, nBL1Offset, kBL1Offset, kL1Len, nL0Len);
    } else {
        BL1ProcessKN(curBL1BufIdx, nBL1Offset, kBL1Offset, kL1Len, nL0Len);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline bool
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::IterMatmulOutKNotFullloadNoReuse()
{
    if (unlikely(isFirstIter_)) {
        curML0Idx_ = 0;
        curNL0Idx_ = 0;
        isFirstIter_ = false;
    } else if (likely(tiling_->matmulTiling.iterateOrder == static_cast<int>(IterateOrder::ORDER_N))) {
        if (++curML0Idx_ >= mIter_) {
            curML0Idx_ = 0;
            if (++curNL0Idx_ >= nIter_) {
                return false;
            }
        }
    } else {
        if (++curNL0Idx_ >= nIter_) {
            curNL0Idx_ = 0;
            if (++curML0Idx_ >= mIter_) {
                return false;
            }
        }
    }

    baseUseM_ = (curML0Idx_ + 1 == mIter_) ? tailM_ : tiling_->matmulTiling.baseM;
    baseUseN_ = (curNL0Idx_ + 1 == nIter_) ? tailN_ : tiling_->matmulTiling.baseN;
    mAL1Len_ = baseUseM_;
    nBL1Len_ = baseUseN_;
    curML1Idx_ = curML0Idx_;
    curNL1Idx_ = curNL0Idx_;
    return true;
}

/*
 * 功能：该函数作用为通过每次移动一个 baseM 或 baseN，并更新当前L0和L1的index以及对应当前的使用大小（包含尾块）
 */
template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline bool
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::IterMatmulOut()
{
    return IterMatmulOutKNotFullloadNoReuse();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<
    xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
    weightNz>::GetAL1KNotFullloadNoReuse(int64_t kFactorIdx, AscendC::TEventID eventIdsMte1ToMte2[MAX_AL1_BUF_NUM])
{
    if ASCEND_IS_AIC {
        if (fullloadKaIn1Buf_) {
            if (kFactorIdx != 0 || curNL0Idx_ != 0) {
                return;
            }
        } else if (kFactorIdx % kAl1Factor_ != 0) {
            return;
        }
        int64_t kAl1Idx = kFactorIdx / kAl1Factor_;
        // 等待目标buffer可用
        WaitFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2[curAL1BufIdx_]);

        kAL1Offset_ = kAl1Idx * kAL1Size_;
        kAL1Len_ = Min(tiling_->kSize - kAL1Offset_, static_cast<uint64_t>(kAL1Size_));
        aGmOffset_ = mBlockOffset_ + curML0Idx_ * tiling_->matmulTiling.baseM;
        if constexpr (!aTrans) {
            aGmOffset_ *= tiling_->matmulTiling.Ka;
            aGmOffset_ += kAL1Offset_;
        } else {
            aGmOffset_ += kAl1Idx * tiling_->mSize;
        }
        // CopyND2NZ 函数内部包含 SetFlag<MTE2_MTE1> / WaitFlag<MTE2_MTE1> 等同步指令
        CopyND2NZ(curAL1BufIdx_);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<
    xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
    weightNz>::GetAL1(int64_t kFactorIdx, AscendC::TEventID eventIdsMte1ToMte2[MAX_AL1_BUF_NUM])
{
    GetAL1KNotFullloadNoReuse(kFactorIdx, eventIdsMte1ToMte2);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::GetBL1KNotFullloadNoReuse(int64_t kFactorIdx)
{
    if (kFactorIdx % kBl1Factor_ != 0) {
        return;
    }

    int64_t kbl1Idx = kFactorIdx / kBl1Factor_;
    kBL1Offset_ = kbl1Idx * kBL1Size_;
    kBL1Len_ = Min(tiling_->kSize - kBL1Offset_, static_cast<uint64_t>(kBL1Size_));
    nBL1Offset_ = nBlockOffset_ + curNL1Idx_ * tiling_->matmulTiling.baseN;

    if ASCEND_IS_AIV {
        // WaitForCube在CopyUb2L1内
        CopyUb2L1();
        // NotifyCube在CopyUb2L1内
    } else {
        WaitForVector(curBL1BufIdx_);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::GetBL1(int64_t kFactorIdx)
{
    GetBL1KNotFullloadNoReuse(kFactorIdx);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<
    xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
    weightNz>::GetBiasL1(int64_t kFactorIdx, AscendC::TEventID biasEventIdsMte1ToMte2[DOUBLE_BUFFER])
{
    if ASCEND_IS_AIC {
        if (!tiling_->matmulTiling.isBias || kFactorIdx != 0) {
            return;
        }
        WaitFlag<HardEvent::MTE1_MTE2>(biasEventIdsMte1ToMte2[biasIdx_]);

        int64_t biasOffset = nBlockOffset_ + curNL0Idx_ * biasL1DataSize_;
        if (biasIdx_ == 0) {
            DataCopy(biasL1LocalBuf0_, biasGlobal_[biasOffset], baseUseN_);
        } else {
            DataCopy(biasL1LocalBuf1_, biasGlobal_[biasOffset], baseUseN_);
        }

        SetFlag<HardEvent::MTE2_MTE1>(biasIdx_ + al1pingpong_);
        WaitFlag<HardEvent::MTE2_MTE1>(biasIdx_ + al1pingpong_);
    }
}

/*
 * 功能：从 UB 上获取一块 BL1 buffer 的数据
 */
template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::CopyUb2L1()
{
    vecKBL1Len_ = kBL1Len_;
    vecNBL1Len_ = nBL1Len_;
    if constexpr (antiQuantType == Mc2QuantType::PER_GROUP && !weightNz && bTrans) {
        // A16W4-ND-perGroup, A16W8-ND-perGroup
        // 只在n方向切分，且只有2个或1个任务。一个任务时AIV0，AIV1执行相同的任务
        if (AscendC::GetSubBlockIdx() == 1) {
            nBL1Offset_ += nBL1Len_ > tiling_->nBubSize ? tiling_->nBubSize : 0;
            vecNBL1Len_ = nBL1Len_ > tiling_->nBubSize ? nBL1Len_ - tiling_->nBubSize : nBL1Len_;
        } else {
            vecNBL1Len_ = Min(tiling_->nBubSize, static_cast<uint64_t>(nBL1Len_));
        }
        VectorProcess();
    } else if constexpr (antiQuantType == Mc2QuantType::PER_GROUP && !weightNz && !bTrans) {
        // A16W4-ND-perGroup, A16W8-ND-perGroup
        // 只在k方向切分，且只有2个或1个任务。一个任务时AIV0，AIV1执行相同的任务
        twoVectorCoreSplitK_ = kBL1Len_ > tiling_->kBubSize;
        if (AscendC::GetSubBlockIdx() == 1) {
            kBL1Offset_ += twoVectorCoreSplitK_ ? tiling_->kBubSize : 0;
            vecKBL1Len_ = twoVectorCoreSplitK_ ? kBL1Len_ - tiling_->kBubSize : kBL1Len_;
        } else {
            vecKBL1Len_ = Min(tiling_->kBubSize, static_cast<uint64_t>(kBL1Len_));
        }
        VectorProcess();
    } else if constexpr (IS_4BIT_WEIGHT) {
        // A16W4-NZ-perGroup, A16Fp4-NZ-perGroup
        uint64_t nBubFactor = CeilDiv(nBL1Len_, tiling_->nBubSize);
        uint64_t kBubFactor = CeilDiv(kBL1Len_, tiling_->kBubSize);

        if (nBubFactor > 1) { // 优先在N方向切分
            twoVectorCoreSplitN_ = true;

            vecNBL1Len_ = CeilDiv(nBubFactor, 2) * tiling_->nBubSize;
            if (AscendC::GetSubBlockIdx() == 1) {
                nBL1Offset_ += vecNBL1Len_;
                vecNBL1Len_ = Min(tiling_->nSize - nBL1Offset_, static_cast<uint64_t>(nBL1Len_ - vecNBL1Len_));
            } else {
                vecNBL1Len_ = Min(tiling_->nSize - nBL1Offset_, static_cast<uint64_t>(vecNBL1Len_));
            }
        } else if (kBubFactor > 1) {
            twoVectorCoreSplitK_ = true;

            vecKBL1Len_ = CeilDiv(kBubFactor, 2) * tiling_->kBubSize;
            if (AscendC::GetSubBlockIdx() == 1) {
                kBL1Offset_ += vecKBL1Len_;
                vecKBL1Len_ = Min(tiling_->kSize - kBL1Offset_, static_cast<uint64_t>(kBL1Len_ - vecKBL1Len_));
            } else {
                vecKBL1Len_ = Min(tiling_->kSize - kBL1Offset_, static_cast<uint64_t>(vecKBL1Len_));
            }
        } else { // 仅使用 AIV-0 核
#ifdef __CCE_KT_TEST__
            ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "just use a single aiv core is not supported yet"); });
#endif
        }
        VectorProcess();
    } else if (bl1pingpong_ == DOUBLE_BUFFER) {
        // curBL1BufIdx_ == 0 && AscendC::GetSubBlockIdx() == 0 或者
        // curBL1BufIdx_ == 1 && AscendC::GetSubBlockIdx() == 1 时进入 VectorProcess 函数
        if (curBL1BufIdx_ == AscendC::GetSubBlockIdx()) {
            VectorProcess();
        }
    } else if (tiling_->vecCoreParallel == 1) {
        if (AscendC::GetSubBlockIdx() == 1) {
            kBL1Offset_ += tiling_->kBubSize;
            vecKBL1Len_ =
                Min(tiling_->kSize - kBL1Offset_, static_cast<uint64_t>(kBL1Size_)); // AIV-1 需要对应去搬运的 kBL1 大小
        } else {
            vecKBL1Len_ = Min(tiling_->kSize - kBL1Offset_,
                              static_cast<uint64_t>(tiling_->kBubSize)); // AIV-0 需要对应去搬运的 kBL1 大小
        }
        VectorProcess();
    } else if (AscendC::GetSubBlockIdx() == 0) {
        VectorProcess();
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::SetOrgShape()
{
    if constexpr (aTrans) {
        if constexpr (bTrans) {
            mmObj_.SetOrgShape(baseUseM_, CeilAlign(baseUseN_, BLOCK_CUBE), CeilAlign(kAL1Len_, BLOCK_CUBE), kBL1Len_,
                               tiling_->nSize);
        } else {
            mmObj_.SetOrgShape(baseUseM_, baseUseN_, CeilAlign(kAL1Len_, BLOCK_CUBE), CeilAlign(kBL1Len_, BLOCK_CUBE),
                               tiling_->nSize);
        }
    } else {
        if constexpr (bTrans) {
            mmObj_.SetOrgShape(CeilAlign(baseUseM_, BLOCK_CUBE), CeilAlign(baseUseN_, BLOCK_CUBE), kAL1Len_, kBL1Len_,
                               tiling_->nSize);
        } else {
            mmObj_.SetOrgShape(CeilAlign(baseUseM_, BLOCK_CUBE), baseUseN_, kAL1Len_, CeilAlign(kBL1Len_, BLOCK_CUBE),
                               tiling_->nSize);
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::SetTensorA(int64_t kFactorIdx)
{
    // 一次载入gm到L1载入，多次使用，需要有偏移
    if constexpr (aTrans) {
        // (m1,k1,k0,m0), offsetK(kBL1Size_肯定是16的倍数，所以不需要对齐) + offsetM
        aL1Offset_ = (kFactorIdx % kAl1Factor_) * kBL1Size_ * BLOCK_CUBE;
    } else {
        // (k1,m1,m0,k0), offsetM + offsetK
        aL1Offset_ = (kFactorIdx % kAl1Factor_) * kBL1Size_ * CeilAlign(mAL1Len_, BLOCK_CUBE);
    }

    if constexpr (L1_4BUFFER) {
        if (curAL1BufIdx_ == 0) {
            mmObj_.SetTensorA(aL1LocalBuf0_[aL1Offset_], aTrans);
        } else if (curAL1BufIdx_ == 1) {
            mmObj_.SetTensorA(aL1LocalBuf1_[aL1Offset_], aTrans);
        } else if (curAL1BufIdx_ == 2) {
            mmObj_.SetTensorA(aL1LocalBuf2_[aL1Offset_], aTrans);
        } else {
            mmObj_.SetTensorA(aL1LocalBuf3_[aL1Offset_], aTrans);
        }
    } else {
        if (curAL1BufIdx_ == 0) {
            mmObj_.SetTensorA(aL1LocalBuf0_[aL1Offset_], aTrans);
        } else {
            mmObj_.SetTensorA(aL1LocalBuf1_[aL1Offset_], aTrans);
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::SetTensorB(int64_t kFactorIdx)
{
    if constexpr (bTrans) {
        // (k1,n1,n0,k0), offsetN + offsetK
        bL1Offset_ = (kFactorIdx % kBl1Factor_) * kAL1Size_ * CeilAlign(nBL1Len_, BLOCK_CUBE);
    } else {
        // (n1,k1,k0,n0), offsetK + offsetN
        bL1Offset_ = (kFactorIdx % kBl1Factor_) * kAL1Size_ * BLOCK_CUBE;
    }
    if constexpr (MAX_BL1_BUF_NUM == 4) {
        if (curBL1BufIdx_ == 0) {
            mmObj_.SetTensorB(bL1LocalBuf0_[bL1Offset_], bTrans);
        } else if (curBL1BufIdx_ == 1) {
            mmObj_.SetTensorB(bL1LocalBuf1_[bL1Offset_], bTrans);
        } else if (curBL1BufIdx_ == 2) {
            mmObj_.SetTensorB(bL1LocalBuf2_[bL1Offset_], bTrans);
        } else {
            mmObj_.SetTensorB(bL1LocalBuf3_[bL1Offset_], bTrans);
        }
    } else {
        if (curBL1BufIdx_ == 0) {
            mmObj_.SetTensorB(bL1LocalBuf0_[bL1Offset_], bTrans);
        } else {
            mmObj_.SetTensorB(bL1LocalBuf1_[bL1Offset_], bTrans);
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType,
                                               weightNz>::IterateMatmulKNotFullloadNoReuse(int64_t kFactorIdx)
{
    if ASCEND_IS_AIC {
        SetOrgShape();
        SetTensorA(kFactorIdx);
        SetTensorB(kFactorIdx);

        if (tiling_->matmulTiling.isBias) {
            if (biasIdx_ == 0) {
                mmObj_.SetBias(biasL1LocalBuf0_);
            } else {
                mmObj_.SetBias(biasL1LocalBuf1_);
            }
        }

        mmObj_.SetTail(baseUseM_, baseUseN_, Min(kAL1Len_, kBL1Len_));
#ifndef __CCE_KT_TEST__
        mmObj_.Iterate(kFactorIdx != 0);
#endif
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::GetTensorC()
{
    if ASCEND_IS_AIC {
        uint64_t outOffset = (mBlockOffset_ + curML0Idx_ * tiling_->matmulTiling.baseM) * tiling_->nSize +
                             nBlockOffset_ + curNL0Idx_ * tiling_->matmulTiling.baseN;
#ifndef __CCE_KT_TEST__
        mmObj_.GetTensorC(yGlobal_[outOffset]);
#endif
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::IterateMatmul(int64_t kFactorIdx)
{
    IterateMatmulKNotFullloadNoReuse(kFactorIdx);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                               antiQuantType, weightNz>::VectorProcess()
{
    WaitForCube();
    BL1Process(curBL1BufIdx_, nBL1Offset_, kBL1Offset_, kBL1Len_, baseUseN_);
    NotifyCube();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<
    xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
    weightNz>::InitSync(AscendC::TEventID eventIdsMte1ToMte2[MAX_AL1_BUF_NUM],
                        AscendC::TEventID biasEventIdsMte1ToMte2[DOUBLE_BUFFER])
{
    if ASCEND_IS_AIC {
        if constexpr (MAX_BL1_BUF_NUM == 4) {
            if (bl1pingpong_ == 4) {
                NotifyVector(0);
                NotifyVector(1);
                NotifyVector(2);
                NotifyVector(3);
            } else if (bl1pingpong_ == 2) {
                NotifyVector(0);
                NotifyVector(1);
            } else {
                NotifyVector(0);
            }
        } else {
            if (bl1pingpong_ >= 2) {
                NotifyVector(0);
                NotifyVector(1);
            } else {
                NotifyVector(0);
            }
        }

        if constexpr (MAX_AL1_BUF_NUM == 4) {
            eventIdsMte1ToMte2[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
            eventIdsMte1ToMte2[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
            eventIdsMte1ToMte2[2] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
            eventIdsMte1ToMte2[3] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
            SetFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2[0]);
            SetFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2[1]);
            SetFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2[2]);
            SetFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2[3]);
        } else {
            eventIdsMte1ToMte2[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
            eventIdsMte1ToMte2[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
            SetFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2[0]);
            SetFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2[1]);
        }

        if (tiling_->matmulTiling.isBias) {
            if (biasPingPong_ == 1) {
                biasEventIdsMte1ToMte2[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
                SetFlag<HardEvent::MTE1_MTE2>(biasEventIdsMte1ToMte2[0]);
            } else {
                biasEventIdsMte1ToMte2[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
                biasEventIdsMte1ToMte2[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
                SetFlag<HardEvent::MTE1_MTE2>(biasEventIdsMte1ToMte2[0]);
                SetFlag<HardEvent::MTE1_MTE2>(biasEventIdsMte1ToMte2[1]);
            }
        }
    } else {
        if constexpr (!weightNz && antiQuantType == Mc2QuantType::PER_GROUP) {
            for (int32_t idx = 0; idx < bubpingpong_; idx++) {
                SetFlag<HardEvent::V_MTE2>(idx);
            }
            for (int32_t idx = 0; idx < bl1pingpong_; idx++) {
                SetFlag<HardEvent::MTE3_V>(idx);
            }
        } else if constexpr (weightNz && antiQuantType == Mc2QuantType::PER_GROUP) {
            for (int32_t idx = 0; idx < bubpingpong_; idx++) {
                SetFlag<HardEvent::V_MTE2>(idx);
                SetFlag<HardEvent::MTE3_V>(idx);
            }
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<
    xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
    weightNz>::EndSync(AscendC::TEventID eventIdsMte1ToMte2[MAX_AL1_BUF_NUM],
                       AscendC::TEventID biasEventIdsMte1ToMte2[DOUBLE_BUFFER])
{
    if ASCEND_IS_AIV {
        if constexpr (IS_4BIT_WEIGHT && weightNz) {
            int32_t buffNum = Min(nIter_ * tiling_->kSize / tiling_->kBubSize, static_cast<uint64_t>(bubpingpong_));
            for (int32_t idx = 0; idx < buffNum; idx++) {
                WaitFlag<HardEvent::V_MTE2>(idx);
                WaitFlag<HardEvent::MTE3_V>(idx);
            }
        } else if constexpr (!weightNz && antiQuantType == Mc2QuantType::PER_GROUP) {
            for (int32_t idx = 0; idx < bubpingpong_; idx++) {
                WaitFlag<HardEvent::V_MTE2>(idx);
            }
            for (int32_t idx = 0; idx < bl1pingpong_; idx++) {
                WaitFlag<HardEvent::MTE3_V>(idx);
            }
        }
        // 消耗InitSync里多发的NotifyVector，防止遗留到下个算子
        EndWaitForCube();
    } else {
        if constexpr (MAX_AL1_BUF_NUM == 4) {
            WaitFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2[0]);
            WaitFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2[1]);
            WaitFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2[2]);
            WaitFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2[3]);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2[0]);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2[1]);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2[2]);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2[3]);
        } else {
            WaitFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2[0]);
            WaitFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2[1]);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2[0]);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2[1]);
        }

        if (tiling_->matmulTiling.isBias) {
            if (biasPingPong_ == 1) {
                WaitFlag<HardEvent::MTE1_MTE2>(biasEventIdsMte1ToMte2[0]);
                GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(biasEventIdsMte1ToMte2[0]);
            } else {
                WaitFlag<HardEvent::MTE1_MTE2>(biasEventIdsMte1ToMte2[0]);
                WaitFlag<HardEvent::MTE1_MTE2>(biasEventIdsMte1ToMte2[1]);
                GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(biasEventIdsMte1ToMte2[0]);
                GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(biasEventIdsMte1ToMte2[1]);
            }
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<
    xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
    weightNz>::PostProcess(int32_t kFactorIdx, AscendC::TEventID eventIdsMte1ToMte2[MAX_AL1_BUF_NUM],
                           AscendC::TEventID biasEventIdsMte1ToMte2[DOUBLE_BUFFER])
{
    if ASCEND_IS_AIC {
        if ((kFactorIdx + 1) % kBl1Factor_ == 0 || (kFactorIdx + 1) == kSingleCoreIterNum_) {
            NotifyVector(curBL1BufIdx_);
            curBL1BufIdx_ = (curBL1BufIdx_ + 1) % bl1pingpong_;
        }
        // AL1
        // k非全载时，每次跨kAl1Factor_前一拍做同步，考虑mk合轴后开db时kAL1可能存在尾块，因此在最后一次k循环也要加同步
        // AL1 k全载时，仅当完成一次n方向iter后做同步
        if ((!fullloadKaIn1Buf_ && ((kFactorIdx + 1) % kAl1Factor_ == 0 || (kFactorIdx + 1) == kSingleCoreIterNum_)) ||
            (fullloadKaIn1Buf_ && (kFactorIdx + 1) == kSingleCoreIterNum_ && (curNL0Idx_ + 1) == nIter_)) {
            SetFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2[curAL1BufIdx_]);
            curAL1BufIdx_ = (curAL1BufIdx_ + 1) % al1pingpong_;
        }

        if (tiling_->matmulTiling.isBias && kFactorIdx == kSingleCoreIterNum_ - 1) {
            SetFlag<HardEvent::MTE1_MTE2>(biasEventIdsMte1ToMte2[biasIdx_]);
            biasIdx_ = (biasIdx_ + 1) % biasPingPong_;
        }
    } else {
        if ((kFactorIdx + 1) % kBl1Factor_ == 0 || (kFactorIdx + 1) == kSingleCoreIterNum_) {
            curBL1BufIdx_ = (curBL1BufIdx_ + 1) % bl1pingpong_;
        }
    }
}
} // namespace Mc2WeightQuantBatchMatmulV2::Arch35

#endif // WEIGHT_QUANT_BATCHMATMUL_V2_REG_BASE_COMMON_H
