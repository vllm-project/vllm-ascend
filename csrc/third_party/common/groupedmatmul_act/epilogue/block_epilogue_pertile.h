/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_epilogue_pertile.h
 * \brief
 */

#ifndef EPILOGUE_BLOCK_EPILOGUE_PERTILE_H
#define EPILOGUE_BLOCK_EPILOGUE_PERTILE_H
#include "kernel_operator.h"
#include "../utils/common_utils.h"
#include "../utils/grouped_matmul_constant.h"
#include "../utils/layout_utils.h"
#include "../utils/tensor_utils.h"

namespace Act {
namespace Gemm {
namespace Block {
#define QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS                                                                         \
    template <typename L0TileShape_, typename DataTypeOut_, typename DataTypeIn_, typename DataTypeBias_,              \
              typename DataTypeX2Scale_, typename LayoutX1Scale_, typename DataTypeX1Scale_, typename LayoutX2Scale_>
#define QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS                                                                          \
    L0TileShape_, DataTypeOut_, DataTypeIn_, DataTypeBias_, DataTypeX2Scale_, LayoutX1Scale_, DataTypeX1Scale_,        \
        LayoutX2Scale_

using namespace Act::Gemm::GroupedMatmul;

struct PerBlockUBParam {
    bool CopyOutWithSplitN = false;
    uint16_t ndNum;
    uint64_t singleM;
    uint64_t singleN;
    uint64_t validM;
    uint32_t validN[UB_SUB_BANK_NUM];
    uint64_t offsetScaleM;
    uint64_t offsetScaleN[UB_SUB_BANK_NUM];
    uint64_t offsetY[UB_SUB_BANK_NUM];
};

namespace {
constexpr uint32_t Y_IDX = 0;
constexpr uint32_t X2SCALE_IDX = 1;
constexpr uint32_t X1SCALE_IDX = 2;
constexpr uint32_t BIAS_IDX = 3;
} // namespace

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
class BlockEpiloguePerTile {
public:
    __aicore__ inline BlockEpiloguePerTile()
    {
        if ASCEND_IS_AIV {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
            if constexpr (!AscendC::IsSameType<CType, YType>::value) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
            }
        }
    }

    __aicore__ inline ~BlockEpiloguePerTile()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
        if ASCEND_IS_AIV {
            if constexpr (!AscendC::IsSameType<CType, YType>::value) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
            }
        }
    }

    struct Arguments {
        GM_ADDR outGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t groupSizeM = 1U;
        uint32_t groupSizeN = 128U;
        uint32_t groupSizeK = 128U;
        Arguments() = default;
    };

    // params
    using Params = Arguments;

    using YType = DataTypeOut_;
    using CType = DataTypeIn_;
    using BiasType = DataTypeBias_;
    using X2ScaleType = DataTypeX2Scale_;
    using X1ScaleType = DataTypeX1Scale_;
    using LayoutX1Scale = LayoutX1Scale_;
    using LayoutX2Scale = LayoutX2Scale_;

    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>; // m,n,k
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;

    static constexpr bool transA = TagToTrans<LayoutX1Scale>::value;
    static constexpr bool transB = TagToTrans<LayoutX2Scale>::value;

public:
    __aicore__ inline void Init(const Params* params);
    __aicore__ inline void operator()(const TupleShape& actualSingleShape, const BlockCoord& blockCoord);
    __aicore__ inline void UpdateGlobalAddr(const BlockCoord& baseOffset);
    __aicore__ inline void UpdateParamsForNextProblem(const TupleShape& problemShape);
    __aicore__ inline auto GetL0c2UbPingTensor();
    __aicore__ inline auto GetL0c2UbPongTensor();

private:
    __aicore__ inline void ProcessAivSingleKPerTile(int64_t x1ScaleOffset,
                                                    __gm__ X2ScaleType* x2ScaleAddr[UB_SUB_BANK_NUM]);

    __aicore__ inline void ProcessAivSingleKPerBlock(int64_t x1ScaleOffset,
                                                     __gm__ X2ScaleType* x2ScaleAddr[UB_SUB_BANK_NUM]);
    template <class T>
    __aicore__ inline __ubuf__ T* CopyInX1Scale(uint64_t srcOffset, uint64_t m, uint64_t k);
    template <class T>
    __aicore__ inline T CopyInX1ScalePerblock(__gm__ T* src, uint64_t offset);
    template <class T>
    __aicore__ inline void CopyInX2Scale(T x2Scale[UB_SUB_BANK_NUM], __gm__ T* src[UB_SUB_BANK_NUM], uint64_t offset);
    __aicore__ inline int64_t CalcX1OffsetPerGroup();
    __aicore__ inline void CalcX2OffsetPerGroup(int64_t x2ScaleOffset[UB_SUB_BANK_NUM]);
    template <class T>
    __aicore__ inline __ubuf__ T* GetX1ScaleUbAddrPerGroup(int64_t x1ScaleOffset, uint64_t kOffset, uint64_t kElem);
    template <bool isFirstKLoop, uint32_t ndNum>
    __aicore__ inline void AivPerTensor(__ubuf__ CType* dst, __ubuf__ CType* l0cOut, __ubuf__ X1ScaleType* x1Scale,
                                        uint16_t mSize, uint32_t nSize0, uint32_t nSize1, uint16_t kSize,
                                        X2ScaleType x2Scale0, X2ScaleType x2Scale1, uint64_t x1ScaleKIdxInCache);
    template <bool isFirstKLoop, uint32_t ndNum>
    __aicore__ inline void AivPerTensor(__ubuf__ CType* dst, __ubuf__ CType* l0cOut, X1ScaleType x1Scale,
                                        uint16_t mSize, uint32_t nSize0, uint32_t nSize1, X2ScaleType x2Scale0,
                                        X2ScaleType x2Scale1);
    __aicore__ inline void AivPostProcess(const AscendC::LocalTensor<CType>& mmAddUb);
    __aicore__ inline void CopyOut(const AscendC::LocalTensor<YType>& ubRes, uint16_t eventId, uint16_t blkCount,
                                   uint32_t blkLen, uint32_t srcStride, uint32_t dstStride, uint64_t yOffset);
    __aicore__ inline void CastAndCopyOut(const AscendC::LocalTensor<CType>& mmAddUb);
    __aicore__ inline void UpdatePerBlockUBValidMN();
    __aicore__ inline void UpdatePerBlockUBParam();
    __aicore__ inline void WaitForCube(uint16_t crossPingPongID)
    {
        AscendC::CrossCoreWaitFlag<GMM_AIC_SYNC_AIV_MODE, PIPE_V>(GMM_AIV_SYNC_AIC_FLAG + crossPingPongID);
    }
    __aicore__ inline void NotifyCube(uint16_t crossPingPongID)
    {
        AscendC::CrossCoreSetFlag<GMM_AIC_SYNC_AIV_MODE, PIPE_V>(GMM_AIC_SYNC_AIV_FLAG + crossPingPongID);
    }

    AscendC::GlobalTensor<YType> cGlobal_;
    AscendC::GlobalTensor<X1ScaleType> x1ScaleGlobal_;
    __gm__ X1ScaleType* x1ScaleGlobalPerblock_;
    __gm__ X2ScaleType* x2ScaleGlobal_;
    AscendC::LocalTensor<CType> mmResPing_;
    AscendC::LocalTensor<CType> mmResPong_;
    AscendC::LocalTensor<YType> ubResPing_;
    AscendC::LocalTensor<YType> ubResPong_;
    AscendC::LocalTensor<CType> mmAddUb_;
    AscendC::LocalTensor<X1ScaleType> x1ScaleUbPing_;
    AscendC::LocalTensor<X1ScaleType> x1ScaleUbPong_;

private:
    const Params* params_;
    PerBlockUBParam ubParams_;
    TupleShape problemShape_{};
    TupleShape actualSingleShape_{};
    BlockCoord baseOffset_{0, 0, 0, 0};
    BlockCoord blockCoord_{0, 0, 0, 0};

    uint64_t scaleM_ = 0;
    uint64_t scaleN_ = 0;
    uint64_t scaleK_ = 0;
    uint32_t subBlockIdx_;
    uint16_t crossPingPongID_ = 0;
    uint16_t x1ScalePingPongID_ = 0;
    bool isPertile_ = false;
};

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::Init(const Params* params)
{
    if ASCEND_IS_AIC {
        return;
    }

    params_ = params;
    subBlockIdx_ = AscendC::GetSubBlockIdx();
    (void)GetL0c2UbPingTensor();
    (void)GetL0c2UbPongTensor();
    constexpr uint32_t elems = UB_TWO_BANK_ELEMS_B32 * PER_BLOCK_SIZE;
    constexpr uint32_t addUbOffset = elems * UB_SUB_BANK_NUM * sizeof(CType); // l0c res 128KB
    mmAddUb_ = AscendC::LocalTensor<CType>(AscendC::TPosition::VECCALC, addUbOffset, elems);
    constexpr uint32_t afterAddOffset = addUbOffset + elems * sizeof(CType); // ub add res 64KB
    if constexpr (!AscendC::IsSameType<CType, YType>::value) {
        ubResPing_ = AscendC::LocalTensor<YType>(AscendC::TPosition::VECCALC, afterAddOffset, elems);
        ubResPong_ = ubResPing_[elems / GMM_BUFFER_NUM];
    }

    isPertile_ = params_->groupSizeM == 1;
    if (isPertile_) {
        constexpr uint32_t x1ScaleUbOffset =
            (AscendC::IsSameType<CType, YType>::value) ? afterAddOffset : afterAddOffset + elems * sizeof(YType);
        x1ScaleUbPing_ = AscendC::LocalTensor<X1ScaleType>(AscendC::TPosition::VECCALC, x1ScaleUbOffset,
                                                           PER_BLOCK_SIZE * GMM_MAX_STEP_SCALEA_K);
        x1ScaleUbPong_ = AscendC::LocalTensor<X1ScaleType>(
            AscendC::TPosition::VECCALC, x1ScaleUbOffset + PER_BLOCK_SIZE * GMM_MAX_STEP_SCALEA_K * sizeof(X1ScaleType),
            PER_BLOCK_SIZE * GMM_MAX_STEP_SCALEA_K);
    }
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline auto BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::GetL0c2UbPingTensor()
{
    constexpr uint32_t elems = UB_TWO_BANK_ELEMS_B32 * PER_BLOCK_SIZE;
    mmResPing_ = AscendC::LocalTensor<CType>(AscendC::TPosition::VECCALC, 0, elems * UB_SUB_BANK_NUM);
    return mmResPing_;
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline auto BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::GetL0c2UbPongTensor()
{
    constexpr uint32_t elems = UB_TWO_BANK_ELEMS_B32 * PER_BLOCK_SIZE;
    mmResPong_ = AscendC::LocalTensor<CType>(AscendC::TPosition::VECCALC, UB_SUB_BANK_ELEMS_B32 * sizeof(CType),
                                             elems * UB_SUB_BANK_NUM - UB_SUB_BANK_ELEMS_B32); // 64 stride for fp32 DB
    return mmResPong_;
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::UpdateParamsForNextProblem(const TupleShape& problemShape)
{
    problemShape_ = problemShape;

    scaleM_ = Act::Gemm::CeilDiv(Get<MNK_M>(problemShape_), params_->groupSizeM);
    scaleN_ = Act::Gemm::CeilDiv(Get<MNK_N>(problemShape_), params_->groupSizeN);
    scaleK_ = Act::Gemm::CeilDiv(Get<MNK_K>(problemShape_), params_->groupSizeK);
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::UpdateGlobalAddr(const BlockCoord& baseOffset)
{
    if ASCEND_IS_AIV {
        x1ScaleGlobal_.SetGlobalBuffer((__gm__ X1ScaleType*)params_->x1ScaleGmAddr + Get<X1SCALE_IDX>(baseOffset));
        x1ScaleGlobalPerblock_ = (__gm__ X1ScaleType*)params_->x1ScaleGmAddr + Get<X1SCALE_IDX>(baseOffset);
        x2ScaleGlobal_ = (__gm__ X2ScaleType*)params_->x2ScaleGmAddr + Get<X2SCALE_IDX>(baseOffset);
        cGlobal_.SetGlobalBuffer((__gm__ YType*)params_->outGmAddr + Get<Y_IDX>(baseOffset));
    }
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline int64_t BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::CalcX1OffsetPerGroup()
{
    int64_t x1ScaleOffset = Get<X1SCALE_IDX>(blockCoord_);
    if (subBlockIdx_ == 1) {
        if constexpr (transA) {
            x1ScaleOffset += ubParams_.offsetScaleM;
        } else {
            x1ScaleOffset += (ubParams_.offsetScaleM * scaleK_);
        }
    }
    return x1ScaleOffset;
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::CalcX2OffsetPerGroup(
    int64_t x2ScaleOffset[UB_SUB_BANK_NUM])
{
    if constexpr (transB) {
        x2ScaleOffset[0] = Get<X2SCALE_IDX>(blockCoord_) + (ubParams_.offsetScaleN[0] * scaleK_);
        x2ScaleOffset[1] = Get<X2SCALE_IDX>(blockCoord_) + (ubParams_.offsetScaleN[1] * scaleK_);
    } else {
        x2ScaleOffset[0] = Get<X2SCALE_IDX>(blockCoord_) + ubParams_.offsetScaleN[0];
        x2ScaleOffset[1] = Get<X2SCALE_IDX>(blockCoord_) + ubParams_.offsetScaleN[1];
    }
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
template <class T>
__aicore__ inline __ubuf__ T*
BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::CopyInX1Scale(uint64_t srcOffset, uint64_t m, uint64_t k)
{
    AscendC::DataCopyExtParams x1ScaleGm2UbParams;
    AscendC::DataCopyPadExtParams<X1ScaleType> padParams;
    if constexpr (transA) {
        x1ScaleGm2UbParams.blockCount = k;
        x1ScaleGm2UbParams.blockLen = m * sizeof(T);
        x1ScaleGm2UbParams.srcStride = (scaleM_ - m) * sizeof(T);
    } else {
        x1ScaleGm2UbParams.blockCount = m;
        x1ScaleGm2UbParams.blockLen = k * sizeof(T);
        x1ScaleGm2UbParams.srcStride = (scaleK_ - k) * sizeof(T);
    }
    auto x1ScaleUb = x1ScalePingPongID_ == 0 ? &x1ScaleUbPing_ : &x1ScaleUbPong_;
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(x1ScalePingPongID_);
    AscendC::DataCopyPad(*x1ScaleUb, x1ScaleGlobal_[srcOffset], x1ScaleGm2UbParams, padParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(x1ScalePingPongID_);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(x1ScalePingPongID_);
    return reinterpret_cast<__ubuf__ T*>(x1ScaleUb->GetPhyAddr());
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
template <class T>
__aicore__ inline T BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::CopyInX1ScalePerblock(__gm__ T* src,
                                                                                                       uint64_t offset)
{
    if constexpr (transA) {
        return src[offset * scaleM_];
    } else {
        return src[offset];
    }
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
template <class T>
__aicore__ inline void BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::CopyInX2Scale(
    T x2Scale[UB_SUB_BANK_NUM], __gm__ T* src[UB_SUB_BANK_NUM], uint64_t offset)
{
    if constexpr (transB) {
        x2Scale[0] = src[0][offset];
        x2Scale[1] = src[1][offset];
    } else {
        x2Scale[0] = src[0][offset * scaleN_];
        x2Scale[1] = src[1][offset * scaleN_];
    }
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::UpdatePerBlockUBValidMN()
{
    int64_t actualN = Get<MNK_N>(actualSingleShape_);
    if (ubParams_.CopyOutWithSplitN) {
        ubParams_.validM = ubParams_.singleM;
        uint64_t subBlockIdxOffset = AscendC::GetSubBlockIdx() * ubParams_.singleN;
        uint64_t ndNumN = 2 * ubParams_.singleN + subBlockIdxOffset; // 2: the nSize of 2 ND, base is 2 ND
        ubParams_.validN[0] = actualN < subBlockIdxOffset ? 0 : Min(ubParams_.singleN, actualN - subBlockIdxOffset);
        ubParams_.validN[1] = actualN < ndNumN ? 0 : Min(ubParams_.singleN, actualN - ndNumN);
    } else {
        if (AscendC::GetSubBlockIdx() == 0) {
            ubParams_.validM = ubParams_.singleM;
        } else {
            ubParams_.validM = Get<MNK_M>(actualSingleShape_) - ubParams_.singleM;
        }
        ubParams_.validN[0] = Min(ubParams_.singleN, static_cast<uint64_t>(actualN));
        ubParams_.validN[1] = actualN < ubParams_.singleN ? 0 : Min(ubParams_.singleN, actualN - ubParams_.singleN);
    }
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::UpdatePerBlockUBParam()
{
    ubParams_.CopyOutWithSplitN =
        Get<MNK_N>(actualSingleShape_) > params_->groupSizeN || Get<MNK_M>(actualSingleShape_) == 1;
    uint32_t fixpipeN = 0;
    if (ubParams_.CopyOutWithSplitN) {
        // (m * n/2) is written to 2 UB, n must be multiples of 32
        // | AIV0 singleN | AIV1 singleN | AIV0 singleN | AIV1 singleN |, max(singleN) = 64
        ubParams_.ndNum = Get<MNK_N>(actualSingleShape_) > UB_TWO_BANK_ELEMS_B32 ? 2 : 1; // 2: 2 ND
        int64_t alignedNBase =
            Get<MNK_N>(actualSingleShape_) > PER_BLOCK_SIZE ? PER_BLOCK_SIZE : AscendC::ONE_BLK_SIZE * ubParams_.ndNum;
        fixpipeN = Align(Get<MNK_N>(actualSingleShape_), static_cast<uint64_t>(alignedNBase)) / ubParams_.ndNum;
        ubParams_.singleN = fixpipeN / static_cast<uint32_t>(AscendC::GetTaskRation());
        ubParams_.singleM = Get<MNK_M>(actualSingleShape_);
    } else {
        // (m/2 * n) is written to 2 UB, m must be multiples of 2
        // | AIV0 singleN | AIV0 singleN |
        // | AIV1 singleN | AIV1 singleN |
        ubParams_.ndNum = Get<MNK_N>(actualSingleShape_) > UB_SUB_BANK_ELEMS_B32 ? 2 : 1; // 2: 2 ND
        fixpipeN = Align(Get<MNK_N>(actualSingleShape_), static_cast<uint64_t>(AscendC::BLOCK_CUBE) * ubParams_.ndNum) /
                   ubParams_.ndNum;
        ubParams_.singleN = fixpipeN;
        ubParams_.singleM = CeilDiv(Get<MNK_M>(actualSingleShape_), AscendC::GetTaskRation());
    }
    UpdatePerBlockUBValidMN();
    int64_t offsetM = 0;
    int64_t offsetN0 = 0;
    int64_t offsetN1 = 0;
    if (ubParams_.CopyOutWithSplitN) {
        offsetN0 = ubParams_.validN[0] == 0 ? 0 : AscendC::GetSubBlockIdx() * ubParams_.singleN;
        offsetN1 = ubParams_.validN[1] == 0 ? offsetN0 : offsetN0 + UB_SUB_BANK_NUM * ubParams_.singleN;
    } else {
        if (AscendC::GetSubBlockIdx() == 1) {
            offsetM += ubParams_.singleM;
        }
        offsetN1 = ubParams_.validN[1] == 0 ? 0 : ubParams_.singleN;
    }
    ubParams_.offsetScaleM = offsetM / params_->groupSizeM;
    ubParams_.offsetScaleN[0] = offsetN0 / params_->groupSizeN;
    ubParams_.offsetScaleN[1] = offsetN1 / params_->groupSizeN;
    ubParams_.offsetY[0] = Get<Y_IDX>(blockCoord_) + offsetM * Get<MNK_N>(problemShape_) + offsetN0;
    ubParams_.offsetY[1] = Get<Y_IDX>(blockCoord_) + offsetM * Get<MNK_N>(problemShape_) + offsetN1;
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::operator()(const TupleShape& actualSingleShape,
                                                                        const BlockCoord& blockCoord)
{
    actualSingleShape_ = actualSingleShape;
    blockCoord_ = blockCoord;
    UpdatePerBlockUBParam();
    int64_t x1ScaleOffset = CalcX1OffsetPerGroup(); // one block same x1Scale
    int64_t x2ScaleOffset[UB_SUB_BANK_NUM] = {0};   // maybe 2 different x2Scale
    CalcX2OffsetPerGroup(x2ScaleOffset);
    __gm__ X2ScaleType* x2ScaleAddr[UB_SUB_BANK_NUM] = {x2ScaleGlobal_ + x2ScaleOffset[0],
                                                        x2ScaleGlobal_ + x2ScaleOffset[1]};

    if (isPertile_) {
        ProcessAivSingleKPerTile(x1ScaleOffset, x2ScaleAddr);
    } else {
        ProcessAivSingleKPerBlock(x1ScaleOffset, x2ScaleAddr);
    }
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
template <class T>
__aicore__ inline __ubuf__ T*
BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::GetX1ScaleUbAddrPerGroup(int64_t x1ScaleOffset,
                                                                                      uint64_t kOffset, uint64_t kElem)
{
    uint64_t scaleX1GmOffset;
    if constexpr (transA) {
        scaleX1GmOffset = x1ScaleOffset + kOffset * scaleM_;
    } else {
        scaleX1GmOffset = x1ScaleOffset + kOffset;
    }
    return CopyInX1Scale<X1ScaleType>(scaleX1GmOffset, ubParams_.validM, kElem);
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::ProcessAivSingleKPerTile(
    int64_t x1ScaleOffset, __gm__ X2ScaleType* x2ScaleAddr[UB_SUB_BANK_NUM])
{
    auto mmAddUbAddr = reinterpret_cast<__ubuf__ CType*>(mmAddUb_.GetPhyAddr());
    const uint16_t x1ScaleKElem = Min(GMM_MAX_STEP_SCALEA_K, scaleK_);
    uint64_t kElem;
    __ubuf__ X1ScaleType* x1ScaleUbAddr;
    X2ScaleType x2Scale[UB_SUB_BANK_NUM];
    for (uint64_t kb = 0, kOffset = 0; kb < Get<MNK_K>(problemShape_); kb += params_->baseK, kOffset++) {
        CopyInX2Scale<X2ScaleType>(x2Scale, x2ScaleAddr, kOffset);
        uint64_t x1ScaleKRem = kOffset % x1ScaleKElem;
        if (x1ScaleKRem == 0) {
            kElem = Min(static_cast<uint64_t>(x1ScaleKElem), scaleK_ - kOffset);
            x1ScaleUbAddr = GetX1ScaleUbAddrPerGroup<X1ScaleType>(x1ScaleOffset, kOffset, kElem);
        }

        WaitForCube(crossPingPongID_);
        auto mmUbInputAddr = crossPingPongID_ == 0 ? reinterpret_cast<uint64_t>(mmResPing_.GetPhyAddr()) :
                                                     reinterpret_cast<uint64_t>(mmResPong_.GetPhyAddr());
        if (kb == 0) {
            if (ubParams_.ndNum == 1) {
                AivPerTensor<true, 1U>((__ubuf__ CType*)mmAddUbAddr, (__ubuf__ CType*)mmUbInputAddr, x1ScaleUbAddr,
                                       ubParams_.validM, ubParams_.validN[0], ubParams_.validN[1], kElem, x2Scale[0],
                                       x2Scale[1], x1ScaleKRem);
            } else {
                AivPerTensor<true, 2U>((__ubuf__ CType*)mmAddUbAddr, (__ubuf__ CType*)mmUbInputAddr, x1ScaleUbAddr,
                                       ubParams_.validM, ubParams_.validN[0], ubParams_.validN[1], kElem, x2Scale[0],
                                       x2Scale[1], x1ScaleKRem);
            }
        } else {
            if (ubParams_.ndNum == 1) {
                AivPerTensor<false, 1U>((__ubuf__ CType*)mmAddUbAddr, (__ubuf__ CType*)mmUbInputAddr, x1ScaleUbAddr,
                                        ubParams_.validM, ubParams_.validN[0], ubParams_.validN[1], kElem, x2Scale[0],
                                        x2Scale[1], x1ScaleKRem);
            } else {
                AivPerTensor<false, 2U>((__ubuf__ CType*)mmAddUbAddr, (__ubuf__ CType*)mmUbInputAddr, x1ScaleUbAddr,
                                        ubParams_.validM, ubParams_.validN[0], ubParams_.validN[1], kElem, x2Scale[0],
                                        x2Scale[1], x1ScaleKRem);
            }
        }
        if (x1ScaleKRem == x1ScaleKElem - 1 || kOffset == scaleK_ - 1) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(x1ScalePingPongID_);
            x1ScalePingPongID_ = (x1ScalePingPongID_ + 1) & 1;
        }
        NotifyCube(crossPingPongID_);
        crossPingPongID_ = (crossPingPongID_ + 1) & 1;
    }
    AivPostProcess(mmAddUb_);
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::ProcessAivSingleKPerBlock(
    int64_t x1ScaleOffset, __gm__ X2ScaleType* x2ScaleAddr[UB_SUB_BANK_NUM])
{
    auto mmAddUbAddr = reinterpret_cast<__ubuf__ CType*>(mmAddUb_.GetPhyAddr());
    auto x1ScaleAddr = x1ScaleGlobalPerblock_ + x1ScaleOffset;
    X2ScaleType x2Scale[UB_SUB_BANK_NUM];
    for (uint64_t kb = 0, kOffset = 0; kb < Get<MNK_K>(problemShape_); kb += params_->baseK, kOffset++) {
        CopyInX2Scale<X2ScaleType>(x2Scale, x2ScaleAddr, kOffset);
        X1ScaleType x1Scale = CopyInX1ScalePerblock(x1ScaleAddr, kOffset);
        WaitForCube(crossPingPongID_);
        auto mmUbInputAddr = crossPingPongID_ == 0 ? reinterpret_cast<uint64_t>(mmResPing_.GetPhyAddr()) :
                                                     reinterpret_cast<uint64_t>(mmResPong_.GetPhyAddr());
        if (kb == 0) {
            if (ubParams_.ndNum == 1) {
                AivPerTensor<true, 1U>((__ubuf__ CType*)mmAddUbAddr, (__ubuf__ CType*)mmUbInputAddr, x1Scale,
                                       ubParams_.validM, ubParams_.validN[0], ubParams_.validN[1], x2Scale[0],
                                       x2Scale[1]);
            } else {
                AivPerTensor<true, 2U>((__ubuf__ CType*)mmAddUbAddr, (__ubuf__ CType*)mmUbInputAddr, x1Scale,
                                       ubParams_.validM, ubParams_.validN[0], ubParams_.validN[1], x2Scale[0],
                                       x2Scale[1]);
            }
        } else {
            if (ubParams_.ndNum == 1) {
                AivPerTensor<false, 1U>((__ubuf__ CType*)mmAddUbAddr, (__ubuf__ CType*)mmUbInputAddr, x1Scale,
                                        ubParams_.validM, ubParams_.validN[0], ubParams_.validN[1], x2Scale[0],
                                        x2Scale[1]);
            } else {
                AivPerTensor<false, 2U>((__ubuf__ CType*)mmAddUbAddr, (__ubuf__ CType*)mmUbInputAddr, x1Scale,
                                        ubParams_.validM, ubParams_.validN[0], ubParams_.validN[1], x2Scale[0],
                                        x2Scale[1]);
            }
        }
        NotifyCube(crossPingPongID_);
        crossPingPongID_ = (crossPingPongID_ + 1) & 1;
    }
    AivPostProcess(mmAddUb_);
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
template <bool isFirstKLoop, uint32_t ndNum>
__aicore__ inline void BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::AivPerTensor(
    __ubuf__ CType* dst, __ubuf__ CType* l0cOut, __ubuf__ X1ScaleType* x1Scale, uint16_t mSize, uint32_t nSize0,
    uint32_t nSize1, uint16_t kSize, X2ScaleType x2Scale0, X2ScaleType x2Scale1, uint64_t x1ScaleKIdxInCache)
{
    uint16_t alignM = Align(mSize, GMM_UB_ALIGN_SIZE / sizeof(X1ScaleType));
    uint16_t alignK = Align(kSize, GMM_UB_ALIGN_SIZE / sizeof(X1ScaleType));
    __VEC_SCOPE__
    {
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
            AscendC::MicroAPI::RegTensor<X1ScaleType> x1ScaleReg, muledScaleReg;
            if constexpr (transA) {
                AscendC::MicroAPI::DataCopy<X1ScaleType, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                    x1ScaleReg, x1Scale + x1ScaleKIdxInCache * alignM + mIdx);
            } else {
                AscendC::MicroAPI::DataCopy<X1ScaleType, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                    x1ScaleReg, x1Scale + mIdx * alignK + x1ScaleKIdxInCache);
            }
            // 1 ND
            uint32_t elementNum = nSize0;
            AscendC::MicroAPI::MaskReg maskN = AscendC::MicroAPI::UpdateMask<CType>(elementNum);
            AscendC::MicroAPI::RegTensor<CType> l0cOutReg;
            AscendC::MicroAPI::RegTensor<CType> addReg;
            AscendC::MicroAPI::RegTensor<CType> ResReg, mulScaleOutReg;
            // copy input from ub to register, addr of ub should align to 32B
            uint32_t offset = mIdx * UB_TWO_BANK_ELEMS_B32;
            uint32_t l0cOutOffset = offset;
            AscendC::MicroAPI::DataCopy(l0cOutReg, l0cOut + offset);
            // l0c_out * scale
            AscendC::MicroAPI::Muls(muledScaleReg, x1ScaleReg, x2Scale0, maskN);
            AscendC::MicroAPI::Mul(mulScaleOutReg, l0cOutReg, muledScaleReg, maskN);
            uint32_t dstUbOffset = offset;
            if constexpr (isFirstKLoop) {
                AscendC::MicroAPI::DataCopy<CType, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(dst + dstUbOffset,
                                                                                                mulScaleOutReg, maskN);
            } else {
                AscendC::MicroAPI::DataCopy(addReg, dst + dstUbOffset);
                AscendC::MicroAPI::Add(ResReg, mulScaleOutReg, addReg, maskN);
                // copy out from register to ub
                AscendC::MicroAPI::DataCopy<CType, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(dst + dstUbOffset,
                                                                                                ResReg, maskN);
            }
            // 2 ND
            if constexpr (ndNum == 1) {
                continue;
            }
            elementNum = nSize1;
            maskN = AscendC::MicroAPI::UpdateMask<CType>(elementNum);
            // copy input from ub to register, addr of ub should align to 32B
            l0cOutOffset = offset + UB_TWO_BANK_ELEMS_B32 * PER_BLOCK_SIZE; // + distance of 2 NDs
            AscendC::MicroAPI::DataCopy(l0cOutReg, l0cOut + l0cOutOffset);
            // l0c_out * scale
            AscendC::MicroAPI::Muls(muledScaleReg, x1ScaleReg, x2Scale1, maskN);
            AscendC::MicroAPI::Mul(mulScaleOutReg, l0cOutReg, muledScaleReg, maskN);
            dstUbOffset = offset + nSize0;
            if constexpr (isFirstKLoop) {
                AscendC::MicroAPI::DataCopy<CType, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(dst + dstUbOffset,
                                                                                                mulScaleOutReg, maskN);
            } else {
                AscendC::MicroAPI::DataCopy(addReg, dst + dstUbOffset);
                AscendC::MicroAPI::Add(ResReg, mulScaleOutReg, addReg, maskN);
                // copy out from register to ub
                AscendC::MicroAPI::DataCopy<CType, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(dst + dstUbOffset,
                                                                                                ResReg, maskN);
            }
        }
    }
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
template <bool isFirstKLoop, uint32_t ndNum>
__aicore__ inline void BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::AivPerTensor(
    __ubuf__ CType* dst, __ubuf__ CType* l0cOut, X1ScaleType x1Scale, uint16_t mSize, uint32_t nSize0, uint32_t nSize1,
    X2ScaleType x2Scale0, X2ScaleType x2Scale1)
{
    __VEC_SCOPE__
    {
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
            // 1 ND
            X2ScaleType scaleMul = x1Scale * x2Scale0;
            uint32_t elementNum = nSize0;
            AscendC::MicroAPI::MaskReg maskN = AscendC::MicroAPI::UpdateMask<CType>(elementNum);
            AscendC::MicroAPI::RegTensor<CType> l0cOutReg;
            AscendC::MicroAPI::RegTensor<CType> addReg;
            AscendC::MicroAPI::RegTensor<CType> ResReg, mulScaleOutReg;
            // copy input from ub to register, addr of ub should align to 32B
            uint32_t offset = mIdx * UB_TWO_BANK_ELEMS_B32;
            uint32_t l0cOutOffset = offset;
            AscendC::MicroAPI::DataCopy(l0cOutReg, l0cOut + offset);
            // l0c_out * scale
            AscendC::MicroAPI::Muls(mulScaleOutReg, l0cOutReg, scaleMul, maskN);
            uint32_t dstUbOffset = offset;
            if constexpr (isFirstKLoop) {
                AscendC::MicroAPI::DataCopy<CType, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(dst + dstUbOffset,
                                                                                                mulScaleOutReg, maskN);
            } else {
                AscendC::MicroAPI::DataCopy(addReg, dst + dstUbOffset);
                AscendC::MicroAPI::Add(ResReg, mulScaleOutReg, addReg, maskN);
                // copy out from register to ub
                AscendC::MicroAPI::DataCopy<CType, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(dst + dstUbOffset,
                                                                                                ResReg, maskN);
            }
            // 2 ND
            if constexpr (ndNum == 1) {
                continue;
            }
            scaleMul = x1Scale * x2Scale1;
            elementNum = nSize1;
            maskN = AscendC::MicroAPI::UpdateMask<CType>(elementNum);
            // copy input from ub to register, addr of ub should align to 32B
            l0cOutOffset = offset + UB_TWO_BANK_ELEMS_B32 * PER_BLOCK_SIZE; // + distance of 2 NDs
            AscendC::MicroAPI::DataCopy(l0cOutReg, l0cOut + l0cOutOffset);
            // l0c_out * scale
            AscendC::MicroAPI::Muls(mulScaleOutReg, l0cOutReg, scaleMul, maskN);
            dstUbOffset = offset + nSize0;
            if constexpr (isFirstKLoop) {
                AscendC::MicroAPI::DataCopy<CType, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(dst + dstUbOffset,
                                                                                                mulScaleOutReg, maskN);
            } else {
                AscendC::MicroAPI::DataCopy(addReg, dst + dstUbOffset);
                AscendC::MicroAPI::Add(ResReg, mulScaleOutReg, addReg, maskN);
                // copy out from register to ub
                AscendC::MicroAPI::DataCopy<CType, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(dst + dstUbOffset,
                                                                                                ResReg, maskN);
            }
        }
    }
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::AivPostProcess(const AscendC::LocalTensor<CType>& mmAddUb)
{
    if (ubParams_.validM == 0) {
        return;
    }
    if constexpr (AscendC::IsSameType<YType, CType>::value) {
        // mov optimize in splitM, 0~63 + 64 ~127 -> 0~127
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        if (ubParams_.ndNum == 2 && !ubParams_.CopyOutWithSplitN) { // 2: 2ND, opt branch with splitM
            uint32_t sumN = ubParams_.validN[0] + ubParams_.validN[1];
            CopyOut(mmAddUb, 0, ubParams_.validM, sumN, UB_TWO_BANK_ELEMS_B32 - sumN, Get<MNK_N>(problemShape_) - sumN,
                    ubParams_.offsetY[0]);
        } else {
            for (uint64_t ndIdx = 0; ndIdx < ubParams_.ndNum; ndIdx++) {
                if (ubParams_.validN[ndIdx] > 0) {
                    CopyOut(mmAddUb[ndIdx * ubParams_.validN[0]], 0, ubParams_.validM, ubParams_.validN[ndIdx],
                            UB_TWO_BANK_ELEMS_B32 - ubParams_.validN[ndIdx],
                            Get<MNK_N>(problemShape_) - ubParams_.validN[ndIdx], ubParams_.offsetY[ndIdx]);
                }
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    } else {
        AscendC::PipeBarrier<PIPE_V>();
        CastAndCopyOut(mmAddUb);
    }
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::CopyOut(
    const AscendC::LocalTensor<YType> &ubRes, uint16_t eventId, uint16_t blkCount, uint32_t blkLen, uint32_t srcStride,
    uint32_t dstStride, uint64_t yOffset)
{
    AscendC::DataCopyExtParams copyParams{blkCount, static_cast<uint32_t>(blkLen * sizeof(YType)),
                                          static_cast<uint32_t>(srcStride * sizeof(YType) / AscendC::ONE_BLK_SIZE),
                                          static_cast<uint32_t>(dstStride * sizeof(YType)), 0};
    AscendC::DataCopyPad<YType>(cGlobal_[yOffset], ubRes, copyParams);
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::CastAndCopyOut(const AscendC::LocalTensor<CType> &mmAddUb)
{
    if (ubParams_.ndNum == 2 && !ubParams_.CopyOutWithSplitN) { // 2: 2ND, opt branch with splitM
        uint32_t sumN = ubParams_.validN[0] + ubParams_.validN[1];
        uint32_t mSizePing = CeilDiv(ubParams_.validM, static_cast<uint64_t>(GMM_BUFFER_NUM));
        uint32_t mSize[GMM_BUFFER_NUM] = {mSizePing, static_cast<uint32_t>(ubParams_.validM - mSizePing)};
        for (uint32_t mDbIdx = 0; mDbIdx < GMM_BUFFER_NUM; ++mDbIdx) {
            if (mSize[mDbIdx] > 0 && sumN > 0) {
                auto ubRes = mDbIdx == 0 ? &ubResPing_ : &ubResPong_;
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mDbIdx);
                AscendC::Cast(*ubRes, mmAddUb[mDbIdx * mSizePing * UB_TWO_BANK_ELEMS_B32],
                              AscendC::RoundMode::CAST_RINT, mSize[mDbIdx] * UB_TWO_BANK_ELEMS_B32);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(mDbIdx);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(mDbIdx);
                CopyOut(*ubRes, mDbIdx, mSize[mDbIdx], sumN, UB_TWO_BANK_ELEMS_B32 - sumN,
                        Get<MNK_N>(problemShape_) - sumN,
                        ubParams_.offsetY[0] + mDbIdx * mSizePing * Get<MNK_N>(problemShape_));
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mDbIdx);
            }
        }
    } else {
        for (uint64_t ndIdx = 0; ndIdx < ubParams_.ndNum; ndIdx++) {
            auto ubRes = ndIdx == 0 ? &ubResPing_ : &ubResPong_;
            if (ubParams_.validN[ndIdx] > 0) {
                AscendC::UnaryRepeatParams repeatParam;
                repeatParam.srcBlkStride = 1;
                repeatParam.dstBlkStride = 1;
                // write continuously
                repeatParam.dstRepStride = CeilDiv(ubParams_.singleN, AscendC::ONE_BLK_SIZE / sizeof(YType));
                // srcStride is 16(512B / 32B), subBank0 256B one repeat
                repeatParam.srcRepStride = GMM_BMM_BLOCK_NUM;
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ndIdx);
                AscendC::Cast(*ubRes, mmAddUb[ubParams_.validN[0] * ndIdx], AscendC::RoundMode::CAST_RINT,
                              ubParams_.validN[ndIdx], ubParams_.validM, repeatParam);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ndIdx);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ndIdx);
                CopyOut(*ubRes, ndIdx, ubParams_.validM, ubParams_.validN[ndIdx],
                        ubParams_.singleN - ubParams_.validN[ndIdx],
                        Get<MNK_N>(problemShape_) - ubParams_.validN[ndIdx], ubParams_.offsetY[ndIdx]);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ndIdx);
            }
        }
    }
}
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif // EPILOGUE_BLOCK_EPILOGUE_PERTILE_H
