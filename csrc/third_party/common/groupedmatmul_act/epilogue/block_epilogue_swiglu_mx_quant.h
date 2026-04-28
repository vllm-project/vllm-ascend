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
 * \file block_epilogue_swiglu_mx_quant.h
 * \brief
 */

#ifndef EPILOGUE_BLOCK_EPILOGUE_SWIGLU_MX_QUANT_H
#define EPILOGUE_BLOCK_EPILOGUE_SWIGLU_MX_QUANT_H
#if defined(__DAV_C310__)
#include "kernel_operator.h"
#include "../utils/common_utils.h"
#include "../utils/device_utils.h"
#include "../utils/status_utils.h"
#include "../utils/tensor_utils.h"
#include "../tile/tile_copy_policy.h"

namespace Act {
namespace Gemm {
namespace Block {

namespace Gmmsg {
enum class QuantMode : uint32_t {
    DEFAULT = 0x0U,
    PERTENSOR_MODE = 0x1U,
    PERCHANNEL_MODE = 0x1U << 1,
    PERTOKEN_MODE = 0x1U << 2,
    MX_PERGROUP_MODE = 0x1U << 3,
    PERBLOCK_MODE = 0x1U << 4,
};

enum class QuantDtype : uint8_t {
    DEFAULT = 0x0U,
    FP8_E4M3FN = 0x1U,
    FP8_E5M2 = 0x1U << 1,
};
} // namespace Gmmsg

namespace {
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr uint32_t Y_IDX = 0;
constexpr uint32_t Y_SCALE_IDX = 1;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t MAX_SINGLE_MN = 64 * 256;
constexpr uint32_t PADDED_QUANT_SCALE_SIZE = 64 * 32;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400; // elem_emax右移7位(BF16E8M7)
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;
constexpr uint16_t FP4_E2M1_MAX_EXP = 0x0100;
constexpr uint16_t FP4_E1M2_MAX_EXP = 0x0000;
} // namespace

constexpr AscendC::MicroAPI::CastTrait ctInt322Fp32 = {
    AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

constexpr AscendC::MicroAPI::CastTrait ctFp322Half = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

constexpr AscendC::MicroAPI::CastTrait ctHalf2Fp32Zero = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

constexpr AscendC::MicroAPI::CastTrait ctHalf2Fp32One = {
    AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

static constexpr AscendC::MicroAPI::DivSpecificMode DIV_MODE = {
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    true,
};
static constexpr AscendC::MicroAPI::CastTrait CAST_BF16_FP16_TO_FP32 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_BF16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
#define QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS                                                             \
    template <typename L0TileShape_, typename DataTypeOut_, typename DataTypeIn_, typename DataTypeX2Scale_,           \
              typename DataTypeX1Scale_, bool IsTensorList_>
#define QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS                                                                   \
    L0TileShape_, DataTypeOut_, DataTypeIn_, DataTypeX2Scale_, DataTypeX1Scale_, IsTensorList_

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
class BlockEpilogueSwigluQuant {
public:
    __aicore__ inline BlockEpilogueSwigluQuant()
    {
    }

    struct Arguments {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR yScaleGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        uint32_t baseM;
        uint32_t baseN;
        Arguments() = default;
    };

    // params
    using Params = Arguments;

    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    using DataTypeX1Scale = DataTypeX1Scale_;
    using DataTypeX2Scale = DataTypeX2Scale_;

    // shape
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BaseOffset = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t>; // y, yScale, x2Scale, x1Scale, bias
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

public:
    __aicore__ inline void Init(Params const &params);
    __aicore__ inline auto GetFirstL0c2UbTensor();
    __aicore__ inline auto GetSecondL0c2UbTensor();
    __aicore__ inline void operator()(const BlockShape &blockShape, const BlockCoord &blockCoord);
    __aicore__ inline void UpdateGlobalAddr(const BlockCoord &baseOffset);
    __aicore__ inline void UpdateNextProblem(const ProblemShape &problemShape);
    // static init
    __host_aicore__ static Params InitParams(Arguments const &args)
    {
        Params params = {args.yGmAddr,    args.yScaleGmAddr, args.x2ScaleGmAddr, args.x1ScaleGmAddr,
                         args.biasGmAddr, args.baseM,        args.baseN};
        return params;
    }

private:
    __aicore__ inline void VFDoSwigluForMX(uint16_t mSize);
    __aicore__ inline void TransMxScaleLayout(uint16_t mSize, uint16_t scaleBlockN);

    template <Gmmsg::QuantMode quantMode>
    __aicore__ inline void VFDoSwigluAndQuantForMX(__ubuf__ int8_t *outputDst, __ubuf__ uint16_t *scaleDst,
                                                   __ubuf__ DataTypeIn *firstSrc, __ubuf__ DataTypeIn *secondSrc,
                                                   uint16_t mSize, uint16_t nSize);

    __aicore__ inline void ComputeScale(__ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr,
                                        __ubuf__ uint16_t *halfScaleLocalAddr, uint32_t totalScaleInUB,
                                        uint16_t loopNumScale);

    __aicore__ inline void ComputeMaxExp(__ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *maxExpAddr,
                                         uint32_t totalCountInUB, uint16_t loopNum);

    __aicore__ inline void ComputeDataForQuantTargetFp8(__ubuf__ bfloat16_t *srcAddr,
                                                        __ubuf__ uint16_t *halfScaleLocalAddr,
                                                        __ubuf__ int8_t *outLocalAddr, uint32_t totalCountInUB,
                                                        uint16_t loopNum);

    __aicore__ inline void ComputeDataForQuantTargetFp4(__ubuf__ bfloat16_t *srcAddr,
                                                        __ubuf__ uint16_t *halfScaleLocalAddr,
                                                        __ubuf__ int8_t *outLocalAddr, uint32_t totalCountInUB,
                                                        uint16_t loopNum);

    __aicore__ inline void CopyOutputFromUb2Gm(uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src);

    __aicore__ inline void CopyScaleFromUb2Gm(uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src);
    // GM ADDR
    AscendC::GlobalTensor<int8_t> quantOutputGlobal_;
    AscendC::GlobalTensor<int8_t> quantScaleGlobal_;

    // UB ADDR
    AscendC::LocalTensor<DataTypeIn> l0cOutUbFirst_{AscendC::TPosition::VECIN, 0, MAX_SINGLE_MN};
    AscendC::LocalTensor<DataTypeIn> l0cOutUbSecond_{AscendC::TPosition::VECIN, MAX_SINGLE_MN * sizeof(DataTypeIn),
                                                     MAX_SINGLE_MN};
    AscendC::LocalTensor<int8_t> quantOutput_;
    AscendC::LocalTensor<int8_t> quantScaleOutput_;
    AscendC::LocalTensor<int8_t> quantScaleBlockOutput_;
    AscendC::LocalTensor<bfloat16_t> gluRes_;
    AscendC::LocalTensor<uint16_t> maxExp_;
    AscendC::LocalTensor<uint16_t> halfScale_;

    const Params *params_;

    int64_t n_;
    int64_t scaleN_;
    int64_t scaleBlockN_;
    uint32_t subBlockIdx_ = AscendC::GetSubBlockIdx();
    uint32_t singleM_; // cur singleShapeM
    uint32_t singleN_;
    bool isBiasEpilogue_ = false;

    int64_t UBBlockSize_ = 0;
    uint32_t vlForHalfNumber_ = 0;
    uint16_t elementAfterReduce_ = 0;
    uint16_t fpEmax_ = 0;

    BlockCoord blockCoord_{0, 0, 0, 0, 0};
};

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::Init(Params const &params)
{
    if ASCEND_IS_AIC {
        return;
    }
    params_ = &params;
    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value) {
        fpEmax_ = FP8_E4M3_MAX_EXP;
    } else if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        fpEmax_ = FP8_E5M2_MAX_EXP;
    } else if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value) {
        fpEmax_ = FP4_E2M1_MAX_EXP;
    } else {
        fpEmax_ = FP4_E1M2_MAX_EXP;
    }

    // out
    constexpr uint32_t afterIn = 2 * MAX_SINGLE_MN * sizeof(DataTypeIn); // 2: 2 block in
    quantOutput_ = AscendC::LocalTensor<int8_t>(AscendC::TPosition::VECOUT, afterIn, MAX_SINGLE_MN);
    quantScaleOutput_ = AscendC::LocalTensor<int8_t>(
        AscendC::TPosition::VECOUT, afterIn + MAX_SINGLE_MN * sizeof(int8_t), MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE);
    constexpr uint32_t afterIO =
        afterIn + MAX_SINGLE_MN * sizeof(int8_t) + MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE * sizeof(int8_t);
    // swiglu res
    gluRes_ = AscendC::LocalTensor<bfloat16_t>(AscendC::TPosition::VECCALC, afterIO, MAX_SINGLE_MN);
    constexpr uint32_t afterIOAndGlu = afterIO + MAX_SINGLE_MN * sizeof(bfloat16_t);
    // sharedExp
    maxExp_ = AscendC::LocalTensor<uint16_t>(AscendC::TPosition::VECCALC, afterIOAndGlu,
                                             MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE);
    constexpr uint32_t afterIOAndGluExp = afterIOAndGlu + MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE * sizeof(uint16_t);
    halfScale_ = AscendC::LocalTensor<uint16_t>(AscendC::TPosition::VECCALC, afterIOAndGluExp,
                                                MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE);
    constexpr uint32_t realScaleBlockOffset =
        afterIOAndGluExp + MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE * sizeof(uint16_t);
    quantScaleBlockOutput_ =
        AscendC::LocalTensor<int8_t>(AscendC::TPosition::VECOUT, realScaleBlockOffset,
                                     params_->baseM / AscendC::GetTaskRation() * AscendC::ONE_BLK_SIZE);
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateGlobalAddr(const BlockCoord &baseOffset)
{
    if ASCEND_IS_AIV {
        quantOutputGlobal_.SetGlobalBuffer((__gm__ int8_t *)params_->yGmAddr + Get<Y_IDX>(baseOffset));
        quantScaleGlobal_.SetGlobalBuffer((__gm__ int8_t *)params_->yScaleGmAddr + Get<Y_SCALE_IDX>(baseOffset));
    }
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateNextProblem(
    const ProblemShape &problemShape)
{
    n_ = Get<MNK_N>(problemShape);
    scaleN_ = CeilDiv(static_cast<uint64_t>(n_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyOutputFromUb2Gm(
    uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src)
{
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockCount = blockCount;
    ub2GmParams.blockLen = singleN_ * sizeof(int8_t);
    ub2GmParams.dstStride = (n_ - singleN_) * sizeof(int8_t);
    if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp4x2_e1m2_t>::value) {
        ub2GmParams.blockLen = ub2GmParams.blockLen >> 1;
        ub2GmParams.dstStride = ub2GmParams.dstStride >> 1;
        offset = offset >> 1;
    }
    AscendC::DataCopyPad(quantOutputGlobal_[offset], src, ub2GmParams);
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyScaleFromUb2Gm(
    uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src)
{
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    auto blockScaleN =
        CeilDiv(static_cast<uint64_t>(singleN_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    ub2GmParams.blockLen = blockScaleN * sizeof(int8_t);
    ub2GmParams.blockCount = blockCount;
    ub2GmParams.dstStride = (scaleN_ - blockScaleN) * sizeof(int8_t);
    AscendC::DataCopyPad(quantScaleGlobal_[offset], src, ub2GmParams);
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeMaxExp(
    __ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *maxExpAddr, uint32_t totalCountInUB, uint16_t loopNum)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0BF16;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1BF16;
        AscendC::MicroAPI::RegTensor<uint16_t> vdExpExtract0;
        AscendC::MicroAPI::RegTensor<uint16_t> vdExpExtract1;

        AscendC::MicroAPI::RegTensor<uint16_t> expMaskBF16;
        AscendC::MicroAPI::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);

        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
        AscendC::MicroAPI::MaskReg scaleMask1;
        AscendC::MicroAPI::MaskReg scaleMask2;
        AscendC::MicroAPI::UnalignReg u1;
        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
            AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_TRUNC};
        for (uint16_t i = 0; i < loopNum; i++) {
            scaleMask1 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(totalCountInUB);
            scaleMask2 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(totalCountInUB);
            AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr,
                vlForHalfNumber_ * 2); // copy two chunks from srcAddr to regbase
            if constexpr (AscendC::IsSameType<bfloat16_t, half>::value) {
                AscendC::MicroAPI::Cast<bfloat16_t, bfloat16_t, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, scaleMask1);
                AscendC::MicroAPI::Cast<bfloat16_t, bfloat16_t, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, scaleMask1);
                AscendC::MicroAPI::And(vdExpExtract0, (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp0BF16, expMaskBF16,
                                       scaleMask1);
                AscendC::MicroAPI::And(vdExpExtract1, (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp1BF16, expMaskBF16,
                                       scaleMask1);
            } else {
                AscendC::MicroAPI::And(vdExpExtract0, (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp0, expMaskBF16,
                                       scaleMask1);
                AscendC::MicroAPI::And(vdExpExtract1, (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp1, expMaskBF16,
                                       scaleMask1);
            }

            AscendC::MicroAPI::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
            AscendC::MicroAPI::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);

            AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, vdMaxExp, u1, elementAfterReduce_);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(maxExpAddr, u1, 0);
    }
    return;
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeScale(
    __ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr, __ubuf__ uint16_t *halfScaleLocalAddr,
    uint32_t totalScaleInUB, uint16_t loopNumScale)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> expMask, sharedExp, scaleValue, scaleBias, halfScale, fp8NanRegTensor;
        AscendC::MicroAPI::Duplicate(expMask, MAX_EXP_FOR_BF16);
        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0, vdExp1;
        AscendC::MicroAPI::MaskReg cmpResult, zeroMask, cmpResultSub, preMaskScale;
        AscendC::MicroAPI::RegTensor<uint16_t> maxExpValue, zeroRegTensor, nanRegTensor, specialExpRegTensor;
        AscendC::MicroAPI::Duplicate(maxExpValue, fpEmax_);
        AscendC::MicroAPI::Duplicate(scaleBias, BF16_EXP_BIAS);
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);
        AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        AscendC::MicroAPI::MaskReg invalidDataMask, specialDataMask;
        AscendC::MicroAPI::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);
        for (uint16_t i = 0; i < loopNumScale; i++) {
            preMaskScale = AscendC::MicroAPI::UpdateMask<uint16_t>(totalScaleInUB);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                vdMaxExp, maxExpAddr, vlForHalfNumber_);
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(cmpResult, vdMaxExp, expMask,
                                                                       preMaskScale); // INF/NAN
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue,
                                                                       preMaskScale);
            AscendC::MicroAPI::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);
            AscendC::MicroAPI::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
            AscendC::MicroAPI::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);

            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr, scaleValue, vlForHalfNumber_ >> 1, preMaskScale);

            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias,
                                                                       preMaskScale);
            AscendC::MicroAPI::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                halfScaleLocalAddr, halfScale, vlForHalfNumber_, preMaskScale);
        }
    }
    return;
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeDataForQuantTargetFp8(
    __ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *halfScaleLocalAddr, __ubuf__ int8_t *outLocalAddr,
    uint32_t totalCountInUB, uint16_t loopNum)
{
    uint32_t totalCountInUB2 = totalCountInUB * 2;
    using T = bfloat16_t;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg dataMask1, dataMask2, dataMask3, dataMask4;
        AscendC::MicroAPI::MaskReg maskAll =
            AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
        AscendC::MicroAPI::RegTensor<float> floatScaleForMul;
        AscendC::MicroAPI::RegTensor<T> vdExp0, vdExp1, vdExp0Convert, vdExp1Convert;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0BF16, vdExp1BF16;
        AscendC::MicroAPI::RegTensor<float> vdExp0FP32Zero, vdExp0FP32One, vdExp1FP32Zero, vdExp1FP32One;
        AscendC::MicroAPI::RegTensor<DataTypeOut> vdExp0FP8Zero, vdExp0FP8One, vdExp1FP8Zero, vdExp1FP8One;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdBf16Exp0FP4, vdBf16Exp1FP4;

        static constexpr AscendC::MicroAPI::CastTrait castTraitZero = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait castTraitOne = {
            AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait castTrait32to8 = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        for (uint16_t i = 0; i < loopNum; i++) {
            dataMask1 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            dataMask2 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            dataMask3 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB2);
            dataMask4 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB2);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr,
                vlForHalfNumber_ * 2); // copy two chunks from srcAddr to regbase
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr,
                                                                                   elementAfterReduce_);

            AscendC::MicroAPI::Mul(vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);
            AscendC::MicroAPI::Mul(vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);
            AscendC::MicroAPI::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
            AscendC::MicroAPI::Cast<float, T, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
            AscendC::MicroAPI::Cast<float, T, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
            AscendC::MicroAPI::Interleave(vdExp0FP32Zero, vdExp0FP32One, vdExp0FP32Zero, vdExp0FP32One);
            AscendC::MicroAPI::Cast<DataTypeOut, float, castTrait32to8>(vdExp0FP8Zero, vdExp0FP32Zero, dataMask3);
            AscendC::MicroAPI::Cast<DataTypeOut, float, castTrait32to8>(vdExp0FP8One, vdExp0FP32One, dataMask3);
            AscendC::MicroAPI::Cast<float, T, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask2);
            AscendC::MicroAPI::Cast<float, T, castTraitOne>(vdExp1FP32One, vdExp1, dataMask2);
            AscendC::MicroAPI::Interleave(vdExp1FP32Zero, vdExp1FP32One, vdExp1FP32Zero, vdExp1FP32One);
            AscendC::MicroAPI::Cast<DataTypeOut, float, castTrait32to8>(vdExp1FP8Zero, vdExp1FP32Zero, dataMask4);
            AscendC::MicroAPI::Cast<DataTypeOut, float, castTrait32to8>(vdExp1FP8One, vdExp1FP32One, dataMask4);
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp0FP8Zero, OUT_ELE_NUM_ONE_BLK, dataMask3);
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp0FP8One, OUT_ELE_NUM_ONE_BLK, dataMask3);
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp1FP8Zero, OUT_ELE_NUM_ONE_BLK, dataMask4);
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp1FP8One, OUT_ELE_NUM_ONE_BLK, dataMask4);
        }
    }
    return;
}


QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeDataForQuantTargetFp4(
    __ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *halfScaleLocalAddr, __ubuf__ int8_t *outLocalAddr,
    uint32_t totalCountInUB, uint16_t loopNum)
{
    using T = bfloat16_t;
    using U = DataTypeOut;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg dataMask1;
        AscendC::MicroAPI::MaskReg dataMask2;
        AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
        AscendC::MicroAPI::RegTensor<T> vdExp0;
        AscendC::MicroAPI::RegTensor<T> vdExp1;
        AscendC::MicroAPI::RegTensor<T> vdExp0Convert;
        AscendC::MicroAPI::RegTensor<T> vdExp1Convert;

        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0BF16;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1BF16;

        AscendC::MicroAPI::RegTensor<U> vdExp0FP4;
        AscendC::MicroAPI::RegTensor<U> vdExp1FP4;

        AscendC::MicroAPI::RegTensor<bfloat16_t> vdBf16Exp0FP4;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdBf16Exp1FP4;
        static constexpr AscendC::MicroAPI::CastTrait castTrait = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        for (uint16_t i = 0; i < loopNum; i++) {
            dataMask1 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            dataMask2 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr,
                vlForHalfNumber_ * 2); // copy two chunks from srcAddr to regbase
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr,
                                                                                   elementAfterReduce_);

            AscendC::MicroAPI::Mul(vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);
            AscendC::MicroAPI::Mul(vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);
            AscendC::MicroAPI::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
            AscendC::MicroAPI::Cast<U, T, castTrait>(vdExp0FP4, vdExp0, dataMask1);
            AscendC::MicroAPI::Cast<U, T, castTrait>(vdExp1FP4, vdExp1, dataMask2);

            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp0FP4, OUT_ELE_NUM_ONE_BLK, dataMask1);
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp1FP4, OUT_ELE_NUM_ONE_BLK, dataMask2);
        }
    }
    return;
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
template <Gmmsg::QuantMode quantMode>
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoSwigluAndQuantForMX(
    __ubuf__ int8_t *outputDst, __ubuf__ uint16_t *scaleDst, __ubuf__ DataTypeIn *firstSrc,
    __ubuf__ DataTypeIn *secondSrc, uint16_t mSize, uint16_t nSize)
{
    constexpr uint16_t sizePerRepeat = AscendC::VECTOR_REG_WIDTH / sizeof(DataTypeIn);
    uint16_t OneRowRepeatTimes =
        CeilDiv(static_cast<uint64_t>(nSize), static_cast<uint64_t>(sizePerRepeat)); // 一行需要循环的次数，等于n/64
    uint32_t nSrcUbAligned = CeilAlign(nSize, AscendC::ONE_BLK_SIZE / sizeof(DataTypeIn));
    uint32_t nDstUbAligned = CeilAlign(nSize, AscendC::ONE_BLK_SIZE); // out 1B
    const float scalarOne = 1.0;
    __ubuf__ bfloat16_t *gluResAddr = (__ubuf__ bfloat16_t *)gluRes_.GetPhyAddr();

    // swiglu
    __VEC_SCOPE__
    {
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) { // 需要计算m次
            uint32_t elementNum = nSize;
            AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::UpdateMask<DataTypeIn>(elementNum);
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < OneRowRepeatTimes; vfBlockIdx++) { // 每次计算m=1, n=64的数据大小
                AscendC::MicroAPI::RegTensor<bfloat16_t> verg7;
                AscendC::MicroAPI::RegTensor<float> swishInput, gateInput;
                AscendC::MicroAPI::RegTensor<float> verg1, verg2, verg3, verg4, verg5, verg6, swishOutput;

                uint32_t l0cOutOffset = mIdx * nSrcUbAligned + vfBlockIdx * sizePerRepeat;
                AscendC::MicroAPI::DataCopy(swishInput, firstSrc + l0cOutOffset);
                // swish
                AscendC::MicroAPI::Muls(verg2, swishInput, -(scalarOne), mask);
                AscendC::MicroAPI::Exp(verg3, verg2, mask);
                AscendC::MicroAPI::Adds(verg4, verg3, scalarOne, mask);
                AscendC::MicroAPI::Div<float, &DIV_MODE>(swishOutput, swishInput, verg4, mask);

                // load gate data to regbase
                AscendC::MicroAPI::DataCopy(gateInput, secondSrc + l0cOutOffset);

                // swish * gate
                AscendC::MicroAPI::Mul(verg6, swishOutput, gateInput, mask);

                AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(verg7, verg6, mask);
                uint32_t dstUbOffset = mIdx * nDstUbAligned + vfBlockIdx * sizePerRepeat;
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    gluResAddr + dstUbOffset, verg7, mask);
            }
        }
    }

    // quant
    uint32_t totalDataInUb = mSize * nDstUbAligned; // nDstUbAligned aligned by 64
    uint32_t totalScaleInUb = totalDataInUb / AscendC::ONE_BLK_SIZE;
    uint16_t loopDataNum = (totalDataInUb + vlForHalfNumber_ * 2 - 1) / (vlForHalfNumber_ * 2);
    uint16_t loopScaleNum = (totalScaleInUb + vlForHalfNumber_ - 1) / vlForHalfNumber_;
    __ubuf__ uint16_t *maxExpAddr = (__ubuf__ uint16_t *)maxExp_.GetPhyAddr();
    ComputeMaxExp(gluResAddr, maxExpAddr, totalDataInUb, loopDataNum);
    __ubuf__ uint16_t *halfScaleLocalAddr = (__ubuf__ uint16_t *)halfScale_.GetPhyAddr();
    ComputeScale(maxExpAddr, scaleDst, halfScaleLocalAddr, totalScaleInUb, loopScaleNum);
    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        ComputeDataForQuantTargetFp8(gluResAddr, halfScaleLocalAddr, outputDst, totalDataInUb, loopDataNum);
    }
    if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp4x2_e1m2_t>::value) {
        ComputeDataForQuantTargetFp4(gluResAddr, halfScaleLocalAddr, outputDst, totalDataInUb, loopDataNum);
    }
    return;
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoSwigluForMX(uint16_t mSize)
{
    __ubuf__ int8_t *quantOutputInUbAddr = (__ubuf__ int8_t *)quantOutput_.GetPhyAddr();
    __ubuf__ uint16_t *quantScaleOutputInUbAddr = (__ubuf__ uint16_t *)quantScaleOutput_.GetPhyAddr();
    __ubuf__ DataTypeIn *l0cOutUbFirstAddr = (__ubuf__ DataTypeIn *)l0cOutUbFirst_.GetPhyAddr();
    __ubuf__ DataTypeIn *l0cOutUbSecondAddr = (__ubuf__ DataTypeIn *)l0cOutUbSecond_.GetPhyAddr();
    VFDoSwigluAndQuantForMX<Gmmsg::QuantMode::MX_PERGROUP_MODE>(quantOutputInUbAddr, quantScaleOutputInUbAddr,
                                                                l0cOutUbFirstAddr, l0cOutUbSecondAddr, mSize, singleN_);
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::TransMxScaleLayout(uint16_t mSize,
                                                                                           uint16_t scaleBlockN)
{
    __ubuf__ int8_t *quantScaleOutputInUbAddr = (__ubuf__ int8_t *)quantScaleOutput_.GetPhyAddr();
    __ubuf__ int8_t *quantScaleBlockOutputInUbAddr = (__ubuf__ int8_t *)quantScaleBlockOutput_.GetPhyAddr();
    // scale layout: (mSize*8) -> (mSize,32)
    __VEC_SCOPE__
    {
        for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) {
            uint32_t elemNum = scaleBlockN;
            AscendC::MicroAPI::MaskReg maskScaleN = AscendC::MicroAPI::UpdateMask<int8_t>(elemNum);
            AscendC::MicroAPI::RegTensor<int8_t> vreg0;
            AscendC::MicroAPI::UnalignReg u0, u1;
            auto srcUb = quantScaleOutputInUbAddr + mIdx * scaleBlockN;
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, srcUb);
            AscendC::MicroAPI::DataCopyUnAlign(vreg0, u0, srcUb);
            auto dstUb = quantScaleBlockOutputInUbAddr + mIdx * AscendC::ONE_BLK_SIZE;
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::StoreDist::DIST_NORM_B8>(dstUb, vreg0, maskScaleN);
        }
    }
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline auto BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::GetFirstL0c2UbTensor()
{
    return l0cOutUbFirst_;
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline auto BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::GetSecondL0c2UbTensor()
{
    return l0cOutUbSecond_;
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::operator()(const BlockShape &blockShape,
                                                                                   const BlockCoord &blockCoord)
{
    singleM_ = Get<MNK_M>(blockShape);
    singleN_ = Get<MNK_N>(blockShape);
    scaleBlockN_ =
        CeilDiv(static_cast<uint64_t>(singleN_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    blockCoord_ = blockCoord;
    auto halfSingleM = CeilDiv(static_cast<uint64_t>(singleM_), static_cast<uint64_t>(AscendC::GetTaskRation()));
    uint64_t singleMInVec = subBlockIdx_ == 1 ? singleM_ - halfSingleM : halfSingleM;
    if (singleMInVec == 0) {
        return;
    }
    uint64_t mOffset = subBlockIdx_ * halfSingleM;

    vlForHalfNumber_ = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t);
    UBBlockSize_ = BLOCK_SIZE;
    elementAfterReduce_ = AscendC::VECTOR_REG_WIDTH / UBBlockSize_;

    VFDoSwigluForMX(singleMInVec);
    uint64_t yOffset = Get<Y_IDX>(blockCoord) + subBlockIdx_ * halfSingleM * n_;
    uint64_t yScaleOffset = Get<Y_SCALE_IDX>(blockCoord) + subBlockIdx_ * halfSingleM * scaleN_;
    TransMxScaleLayout(singleMInVec, scaleBlockN_);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    CopyOutputFromUb2Gm(singleMInVec, yOffset, quantOutput_);
    CopyScaleFromUb2Gm(singleMInVec, yScaleOffset, quantScaleBlockOutput_);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    return;
}
} // namespace Block
} // namespace Gemm
} // namespace Act

#endif // EPILOGUE_BLOCK_EPILOGUE_SWIGLU_QUANT_H
#endif