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
 * \file block_epilogue_swiglu_mx_quant.h
 * \brief
 */

#ifndef BLOCK_EPILOGUE_SWIGLU_MX_QUANT_H
#define BLOCK_EPILOGUE_SWIGLU_MX_QUANT_H

#if defined(__DAV_C310__)
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "mega_moe_base.h"

namespace SwigluQuantMsg {
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
} // namespace SwigluQuantMsg

namespace {
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr uint32_t Y_IDX = 0;
constexpr uint32_t Y_SCALE_IDX = 1;
constexpr uint32_t GROUP_FLAG_IDX = 2;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t MAX_SINGLE_MN = 256 * 256;
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
constexpr uint32_t FLAG_VALUE_ONE = 1;
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
#define BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS                                                             \
    template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeX2Scale_,                              \
              typename DataTypeX1Scale_, bool IsTensorList_>
#define BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS                                                                   \
    DataTypeOut_, DataTypeIn_, DataTypeX2Scale_, DataTypeX1Scale_, IsTensorList_

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
class BlockEpilogueSwigluMxQuant {
public:
    __aicore__ inline BlockEpilogueSwigluMxQuant()
    {
    }

    struct Arguments {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR yScaleGmAddr{nullptr};
        GM_ADDR groupFlagListGmAddr{nullptr};
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
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    // y, yScale, x2Scale, x1Scale, bias
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

public:
    __aicore__ inline void Init(Params const &params);
    __aicore__ inline auto GetFirstL0c2UbTensor();
    __aicore__ inline auto GetSecondL0c2UbTensor();
    __aicore__ inline void operator()(const BlockShape &blockShape, const BlockCoord &blockCoord);
    __aicore__ inline void UpdateGlobalAddr(const BlockCoord &baseOffset);
    __aicore__ inline void UpdateNextProblem(const ProblemShape &problemShape);

private:
    __aicore__ inline void VFDoSwigluForMX(uint16_t mSize);
    __aicore__ inline void TransMxScaleLayout(uint16_t mSize, uint16_t scaleBlockN);

    template <SwigluQuantMsg::QuantMode quantMode>
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

    __aicore__ inline void AddFlagFromUb2Gm(AscendC::GlobalTensor<uint32_t> &dst, uint32_t flag);
    // GM ADDR
    AscendC::GlobalTensor<int8_t> quantOutputGlobal_;
    AscendC::GlobalTensor<int8_t> quantScaleGlobal_;
    __gm__ int32_t* groupFlagListGmAddr_;

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

    BlockCoord blockCoord_{0, 0, 0, 0, 0, 0};
};

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::Init(Params const &params)
{
    if constexpr(g_coreType == AscendC::AIC) {
        return;
    }
    params_ = &params;
    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value) {
        fpEmax_ = FP8_E4M3_MAX_EXP;
    } else if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        fpEmax_ = FP8_E5M2_MAX_EXP; // this
    } else if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value) {
        fpEmax_ = FP4_E2M1_MAX_EXP;
    } else {
        fpEmax_ = FP4_E1M2_MAX_EXP;
    }

    // 重构UB内存
    constexpr uint32_t gluResOffset = 0;
    gluRes_ = AscendC::LocalTensor<bfloat16_t>(AscendC::TPosition::VECCALC, gluResOffset, MAX_SINGLE_MN);
    constexpr uint32_t quantOutputOffset = gluResOffset + MAX_SINGLE_MN * sizeof(bfloat16_t);
    quantOutput_ = AscendC::LocalTensor<int8_t>(AscendC::TPosition::VECOUT, quantOutputOffset, MAX_SINGLE_MN);
    constexpr uint32_t quantScaleOffset = quantOutputOffset + MAX_SINGLE_MN * sizeof(int8_t);
    quantScaleOutput_ = AscendC::LocalTensor<int8_t>(
        AscendC::TPosition::VECOUT, quantScaleOffset, MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE);
    constexpr uint32_t maxExpOffset = quantScaleOffset + MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE * sizeof(int8_t);
    maxExp_ = AscendC::LocalTensor<uint16_t>(
        AscendC::TPosition::VECCALC, maxExpOffset, MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE);
    constexpr uint32_t halfScaleOffset = maxExpOffset + MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE * sizeof(uint16_t);
    halfScale_ = AscendC::LocalTensor<uint16_t>(
        AscendC::TPosition::VECCALC, halfScaleOffset, MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE);
    constexpr uint32_t quantScaleBlockOffset = halfScaleOffset +
        MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE * sizeof(uint16_t);
    quantScaleBlockOutput_ = AscendC::LocalTensor<int8_t>(
        AscendC::TPosition::VECOUT, quantScaleBlockOffset, params_->baseM * AscendC::ONE_BLK_SIZE);
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateGlobalAddr(const BlockCoord &baseOffset)
{
    if constexpr(g_coreType == AscendC::AIV) {
        quantOutputGlobal_.SetGlobalBuffer((__gm__ int8_t *)params_->yGmAddr + Get<Y_IDX>(baseOffset));
        quantScaleGlobal_.SetGlobalBuffer((__gm__ int8_t *)params_->yScaleGmAddr + Get<Y_SCALE_IDX>(baseOffset));
        groupFlagListGmAddr_ = (__gm__ int32_t *)params_->groupFlagListGmAddr +
        Get<GROUP_FLAG_IDX>(baseOffset) * INT_CACHELINE;
    }
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateNextProblem(
    const ProblemShape &problemShape)
{
    n_ = Get<N_VALUE>(problemShape); // n/2
    scaleN_ = Ops::Base::CeilDiv(static_cast<uint64_t>(n_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE))
        * MXFP_MULTI_BASE_SIZE;
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyOutputFromUb2Gm(
    uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src)
{
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockCount = blockCount; // 128
    ub2GmParams.blockLen = singleN_ * sizeof(int8_t); // 256
    ub2GmParams.dstStride = (n_ - singleN_) * sizeof(int8_t);
    if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp4x2_e1m2_t>::value) {
        ub2GmParams.blockLen = ub2GmParams.blockLen >> 1;
        ub2GmParams.dstStride = ub2GmParams.dstStride >> 1;
        offset = offset >> 1;
    }
    AscendC::DataCopyPad(quantOutputGlobal_[offset], src, ub2GmParams);
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyScaleFromUb2Gm(
    uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src)
{
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    auto blockScaleN = Ops::Base::CeilDiv(static_cast<uint64_t>(singleN_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE))
        * MXFP_MULTI_BASE_SIZE; // 256 / 32 = 8
    ub2GmParams.blockLen = blockScaleN * sizeof(int8_t); // 8
    ub2GmParams.blockCount = blockCount; // 128
    ub2GmParams.dstStride = (scaleN_ - blockScaleN) * sizeof(int8_t);
    AscendC::DataCopyPad(quantScaleGlobal_[offset], src, ub2GmParams);
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::AddFlagFromUb2Gm(
    AscendC::GlobalTensor<uint32_t> &dst, uint32_t flag)
{
    AscendC::AtomicAdd((__gm__ uint32_t*)dst.GetPhyAddr(), flag);
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeMaxExp(
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
            AscendC::MicroAPI::And(vdExpExtract0, (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp0, expMaskBF16,
                                       scaleMask1);
            AscendC::MicroAPI::And(vdExpExtract1, (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp1, expMaskBF16,
                                       scaleMask1);
            AscendC::MicroAPI::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
            AscendC::MicroAPI::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);

            AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, vdMaxExp, u1, elementAfterReduce_);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(maxExpAddr, u1, 0);
    }
    return;
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeScale(
    __ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr, __ubuf__ uint16_t *halfScaleLocalAddr,
    uint32_t totalScaleInUB, uint16_t loopNumScale) // 128*8  8
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> expMask, sharedExp, scaleValue, scaleBias, halfScale, fp8NanRegTensor;
        AscendC::MicroAPI::Duplicate(expMask, MAX_EXP_FOR_BF16); // MAX_EXP_FOR_BF16表示bf16正无穷 大小：128
        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0, vdExp1;
        AscendC::MicroAPI::MaskReg cmpResult, zeroMask, cmpResultSub, preMaskScale;
        AscendC::MicroAPI::RegTensor<uint16_t> maxExpValue, zeroRegTensor, nanRegTensor, specialExpRegTensor;
        AscendC::MicroAPI::Duplicate(maxExpValue, fpEmax_); // 0x0780 大小：128 对应bf16指数位后四位
        AscendC::MicroAPI::Duplicate(scaleBias, BF16_EXP_BIAS); // 0x7f00 大小：128
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8); // 0x00ff 大小：128
        AscendC::MicroAPI::Duplicate(zeroRegTensor, 0); // 0 大小：128
        AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION); // 0x7f81 大小：128
        AscendC::MicroAPI::MaskReg invalidDataMask, specialDataMask;
        AscendC::MicroAPI::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD); // 0x0040 大小：128
        for (uint16_t i = 0; i < loopNumScale; i++) { // 8
            preMaskScale = AscendC::MicroAPI::UpdateMask<uint16_t>(totalScaleInUB); // 128*8
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                vdMaxExp, maxExpAddr, vlForHalfNumber_); // 每次搬运128个数到vdMaxExp
            // 得到不等于INF的结果掩码 cmpResult
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(cmpResult, vdMaxExp, expMask,
                                                                       preMaskScale); // INF/NAN
            // 得到不等于0的结果掩码 zeroMask
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
            // 得到小于或等于0x0780的结果掩码 invalidDataMask
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue,
                                                                       preMaskScale);
            // 将vdMaxExp中小于或等于0x0780的结果替换成0x0780
            AscendC::MicroAPI::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);
            AscendC::MicroAPI::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale); // sharedExp = vdMaxExp - 0x0780
            AscendC::MicroAPI::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
            // 逻辑右移7位 当前指数位在减去0x0780后，已移至最低位
            // 将scaleValue中INF的结果替换成0x00ff
            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
            // 将scaleValue中原来是0的结果替换成0
            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);

            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr, scaleValue, vlForHalfNumber_ >> 1, preMaskScale);
            // 将scaleValue中数取低半部分，搬运到mxScaleLocalAddr uint16--int8

            // 得到sharedExp等于0x7f00的结果掩码 specialDataMask
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias,
                                                                       preMaskScale);
            AscendC::MicroAPI::Sub(halfScale, scaleBias, sharedExp, preMaskScale); // halfScale = 0x7f00 - sharedExp
            // 将halfScale中原等于INF的数值替换成0x7f81
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            // 将halfScale中原等于0的数值替换成0
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            // 将halfScale中原等于0x7f00的数值替换成0x0040
            AscendC::MicroAPI::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                halfScaleLocalAddr, halfScale, vlForHalfNumber_, preMaskScale);
            // 将128个数搬运到halfScaleLocalAddr uint16--uint16
        }
    }
    return;
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeDataForQuantTargetFp8(
    __ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *halfScaleLocalAddr, __ubuf__ int8_t *outLocalAddr,
    uint32_t totalCountInUB, uint16_t loopNum)
{
    uint32_t totalCountInUB2 = totalCountInUB * 2; // 128*256*2
    using T = bfloat16_t;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg dataMask1, dataMask2, dataMask3, dataMask4;
        AscendC::MicroAPI::MaskReg maskAll =
            AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>(); // 全1
        AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
        AscendC::MicroAPI::RegTensor<float> floatScaleForMul;
        AscendC::MicroAPI::RegTensor<T> vdExp0, vdExp1, vdExp0Convert, vdExp1Convert;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0BF16, vdExp1BF16;
        AscendC::MicroAPI::RegTensor<float> vdExp0FP32Zero, vdExp0FP32One, vdExp1FP32Zero, vdExp1FP32One;
        AscendC::MicroAPI::RegTensor<DataTypeOut> vdExp0FP8Zero, vdExp0FP8One, vdExp1FP8Zero, vdExp1FP8One;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdBf16Exp0FP4, vdBf16Exp1FP4;

        static constexpr AscendC::MicroAPI::CastTrait castTraitZero = { // src与dst类型位宽不同时,写入索引0的位置，其余位置置零
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait castTraitOne = { // src与dst类型位宽不同时,写入索引1的位置，其余位置置零
            AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait castTrait32to8 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    // 当结果超出范围时，截断到最大值或最小值
    // 返回最接近参数的整数，如果有2个整数同样接近，则会返回偶数的那个
        for (uint16_t i = 0; i < loopNum; i++) { // 128
            dataMask1 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB); // 128*256
            dataMask2 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB); // 128*256
            dataMask3 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB2); // 128*256*2
            dataMask4 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB2); // 128*256*2
            // DIST_DINTLV_B16:双搬入模式，基于元素的交错搬运，从src中读取2*VL长度数据，将偶数索引的元素存入dst0，将奇数索引的元素存入dst1
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr,
                vlForHalfNumber_ * 2); // copy two chunks from srcAddr to regbase 128*256个数
            // 将halfScale中的8个数uint16广播到halfScaleForMul中，halfScale[0]*16 halfScale[1]*16...
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr,
                                                                                   elementAfterReduce_);
            // vdExp0*halfScaleForMul
            AscendC::MicroAPI::Mul(vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);
            // vdExp1*halfScaleForMul
            AscendC::MicroAPI::Mul(vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);
            // 交叉vdExp0、vdExp1数据，并将前后分别存入vdExp0、vdExp1
            AscendC::MicroAPI::Interleave(vdExp0, vdExp1, vdExp0, vdExp1); // 交叉数据
            AscendC::MicroAPI::Cast<float, T, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1); // vdExp0转float,值在前半部分
            AscendC::MicroAPI::Cast<float, T, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1); // vdExp0转float，值在后半部分
            // 交叉vdExp0FP32Zero、vdExp0FP32One数据，并将前后分别存入vdExp0FP32Zero、vdExp0FP32One
            AscendC::MicroAPI::Interleave(vdExp0FP32Zero, vdExp0FP32One, vdExp0FP32Zero, vdExp0FP32One);
            AscendC::MicroAPI::Cast<DataTypeOut, float, castTrait32to8>(vdExp0FP8Zero, vdExp0FP32Zero, dataMask3);
            // vdExp0FP32Zero转int8
            AscendC::MicroAPI::Cast<DataTypeOut, float, castTrait32to8>(vdExp0FP8One, vdExp0FP32One, dataMask3);
            // vdExp0FP32One转int8
            AscendC::MicroAPI::Cast<float, T, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask2); // vdExp1转float，前半部分值
            AscendC::MicroAPI::Cast<float, T, castTraitOne>(vdExp1FP32One, vdExp1, dataMask2); // vdExp1转float，后半部分值
            // 交叉vdExp1FP32Zero、vdExp1FP32One数据，并将前后分别存入vdExp1FP32Zero、vdExp1FP32One
            AscendC::MicroAPI::Interleave(vdExp1FP32Zero, vdExp1FP32One, vdExp1FP32Zero, vdExp1FP32One);
            AscendC::MicroAPI::Cast<DataTypeOut, float, castTrait32to8>(vdExp1FP8Zero, vdExp1FP32Zero, dataMask4);
            // vdExp1FP32Zero转int8
            AscendC::MicroAPI::Cast<DataTypeOut, float, castTrait32to8>(vdExp1FP8One, vdExp1FP32One, dataMask4);
            // vdExp1FP32One转int8
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                // 将src中有效元素的低8bit（四分之一）数据连续存储于dst中
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp0FP8Zero, OUT_ELE_NUM_ONE_BLK, dataMask3);
            // 64个数
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


BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeDataForQuantTargetFp4(
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

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
template <SwigluQuantMsg::QuantMode quantMode>
__aicore__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoSwigluAndQuantForMX(
    __ubuf__ int8_t *outputDst, __ubuf__ uint16_t *scaleDst, __ubuf__ DataTypeIn *firstSrc,
    __ubuf__ DataTypeIn *secondSrc, uint16_t mSize, uint16_t nSize) // mSize=128 nSize=256
{
    constexpr uint16_t sizePerRepeat = AscendC::VECTOR_REG_WIDTH / sizeof(DataTypeIn); // 128
    uint16_t OneRowRepeatTimes =
        Ops::Base::CeilDiv(static_cast<uint64_t>(nSize), static_cast<uint64_t>(sizePerRepeat));
    uint32_t nSrcUbAligned = Ops::Base::CeilAlign(static_cast<uint32_t>(nSize),
        static_cast<uint32_t>(AscendC::ONE_BLK_SIZE / sizeof(DataTypeIn)));
    uint32_t nDstUbAligned = Ops::Base::CeilAlign(static_cast<uint32_t>(nSize),
        static_cast<uint32_t>(AscendC::ONE_BLK_SIZE));
    const float scalarOne = 1.0;
    __ubuf__ bfloat16_t *gluResAddr = (__ubuf__ bfloat16_t *)gluRes_.GetPhyAddr();

    // swiglu
    __VEC_SCOPE__
    {
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) { // 需要计算m次
            uint32_t elementNum = nSize;
            AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::UpdateMask<DataTypeIn>(elementNum);
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < OneRowRepeatTimes; vfBlockIdx++) { // 每次计算m=1, n=64的数据大小
                AscendC::MicroAPI::RegTensor<DataTypeIn> swishInput0, gateInput0, swishInput1, gateInput1;
                AscendC::MicroAPI::RegTensor<float> swishData0, swishData1, gateData0, gateData1;

                uint32_t l0cOutOffset = mIdx * nSrcUbAligned + vfBlockIdx * sizePerRepeat; // 计算的数据的偏移量
                AscendC::MicroAPI::DataCopy(swishInput0, firstSrc + l0cOutOffset); // swishInput0:x bf16
                AscendC::MicroAPI::DataCopy(gateInput0, secondSrc + l0cOutOffset); // gateInput0:y  bf16
                // 1.交错数据，为cast做准备
                AscendC::MicroAPI::Interleave(swishInput0, swishInput1, swishInput0, swishInput1);
                AscendC::MicroAPI::Interleave(gateInput0, gateInput1, gateInput0, gateInput1);
                // 2.数据类型转换
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_BF16_FP16_TO_FP32>(swishData0, swishInput0, mask);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_BF16_FP16_TO_FP32>(swishData1, swishInput1, mask);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_BF16_FP16_TO_FP32>(gateData0, gateInput0, mask);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_BF16_FP16_TO_FP32>(gateData1, gateInput1, mask);
                // 计算
                uint16_t calBlockNum = sizeof(float) / sizeof(DataTypeIn); // 2
                uint16_t calBlockSize = sizePerRepeat / calBlockNum; // 128/2 = 64
                for (uint16_t i = 0; i < calBlockNum; i++) {
                    AscendC::MicroAPI::RegTensor<bfloat16_t> output; // output:swish(x)*y
                    AscendC::MicroAPI::RegTensor<float> swishData = i == 0 ? swishData0 : swishData1;
                    AscendC::MicroAPI::RegTensor<float> gateData = i == 0 ? gateData0 : gateData1;
                    AscendC::MicroAPI::RegTensor<float> verg2, verg3, verg4, verg5, verg6, swishOutput;
                    
                    // swish
                    AscendC::MicroAPI::Muls(verg2, swishData, -(scalarOne), mask); // verg2:-x
                    AscendC::MicroAPI::Exp(verg3, verg2, mask);                     // verg3:exp(-x)
                    AscendC::MicroAPI::Adds(verg4, verg3, scalarOne, mask);         // verg4:exp(-x)+1
                    AscendC::MicroAPI::Div<float, &DIV_MODE>(swishOutput, swishData, verg4,
                                                            mask); // swishOutput:swish(x)=x/(exp(-x)+1)
                    // swish * gate
                    AscendC::MicroAPI::Mul(verg6, swishOutput, gateData, mask); // verg6:swish(x)*y

                    AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(output, verg6,
                                                                                mask); // verg7:verg6转为bfloat16
                    // 计算的数据的偏移量
                    uint32_t dstUbOffset = mIdx * nDstUbAligned + vfBlockIdx * sizePerRepeat + i * calBlockSize;
                    AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                        gluResAddr + dstUbOffset, output, mask); // gluRes:swish(x)*y 搬运到目标地址
                }
            }
        }
    }

    // quant
    uint32_t totalDataInUb = mSize * nDstUbAligned; // 128*256
    uint32_t totalScaleInUb = totalDataInUb / AscendC::ONE_BLK_SIZE; // 128*256 / 32 = 128 * 8
    uint16_t loopDataNum = (totalDataInUb + vlForHalfNumber_ * 2 - 1) / (vlForHalfNumber_ * 2); // 128
    uint16_t loopScaleNum = (totalScaleInUb + vlForHalfNumber_ - 1) / vlForHalfNumber_; // 8
    __ubuf__ uint16_t *maxExpAddr = (__ubuf__ uint16_t *)maxExp_.GetPhyAddr();
    ComputeMaxExp(gluResAddr, maxExpAddr, totalDataInUb, loopDataNum); // 获取最大值
    __ubuf__ uint16_t *halfScaleLocalAddr = (__ubuf__ uint16_t *)halfScale_.GetPhyAddr();
    ComputeScale(maxExpAddr, scaleDst, halfScaleLocalAddr, totalScaleInUb, loopScaleNum); // 计算scale和halfScale
    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        ComputeDataForQuantTargetFp8(gluResAddr, halfScaleLocalAddr, outputDst, totalDataInUb, loopDataNum); // 计算量化后的值
    }
    if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp4x2_e1m2_t>::value) {
        ComputeDataForQuantTargetFp4(gluResAddr, halfScaleLocalAddr, outputDst, totalDataInUb, loopDataNum);
    }
    return;
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoSwigluForMX(uint16_t mSize)
{
    __ubuf__ int8_t *quantOutputInUbAddr = (__ubuf__ int8_t *)quantOutput_.GetPhyAddr();
    __ubuf__ uint16_t *quantScaleOutputInUbAddr = (__ubuf__ uint16_t *)quantScaleOutput_.GetPhyAddr();
    __ubuf__ DataTypeIn *l0cOutUbFirstAddr = (__ubuf__ DataTypeIn *)l0cOutUbFirst_.GetPhyAddr();
    __ubuf__ DataTypeIn *l0cOutUbSecondAddr = (__ubuf__ DataTypeIn *)l0cOutUbSecond_.GetPhyAddr();
    VFDoSwigluAndQuantForMX<SwigluQuantMsg::QuantMode::MX_PERGROUP_MODE>(quantOutputInUbAddr, quantScaleOutputInUbAddr,
                                                                l0cOutUbFirstAddr, l0cOutUbSecondAddr, mSize, singleN_);
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::TransMxScaleLayout(uint16_t mSize,
                                                                                           uint16_t scaleBlockN)
{
    __ubuf__ int8_t *quantScaleOutputInUbAddr = (__ubuf__ int8_t *)quantScaleOutput_.GetPhyAddr();
    __ubuf__ int8_t *quantScaleBlockOutputInUbAddr = (__ubuf__ int8_t *)quantScaleBlockOutput_.GetPhyAddr();
    // scale layout: (mSize*8) -> (mSize,32)
    __VEC_SCOPE__
    {
        for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) { // 128
            uint32_t elemNum = scaleBlockN; // 8
            AscendC::MicroAPI::MaskReg maskScaleN = AscendC::MicroAPI::UpdateMask<int8_t>(elemNum);
            AscendC::MicroAPI::RegTensor<int8_t> vreg0;
            AscendC::MicroAPI::UnalignReg u0, u1;
            auto srcUb = quantScaleOutputInUbAddr + mIdx * scaleBlockN;
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, srcUb); // 初始化
            AscendC::MicroAPI::DataCopyUnAlign(vreg0, u0, srcUb); // ?
            auto dstUb = quantScaleBlockOutputInUbAddr + mIdx * AscendC::ONE_BLK_SIZE; // mId * 32
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::StoreDist::DIST_NORM_B8>(dstUb, vreg0, maskScaleN);
        }
    }
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline auto BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::GetFirstL0c2UbTensor()
{
    return l0cOutUbFirst_;
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline auto BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::GetSecondL0c2UbTensor()
{
    return l0cOutUbSecond_;
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::operator()(const BlockShape &blockShape,
                                                                                   const BlockCoord &blockCoord)
{
    singleM_ = Get<M_VALUE>(blockShape); // 128
    singleN_ = Get<N_VALUE>(blockShape); // 256
    scaleBlockN_ = Ops::Base::CeilDiv(static_cast<uint64_t>(singleN_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE))
        * MXFP_MULTI_BASE_SIZE;
    blockCoord_ = blockCoord;

    if (singleM_ == 0) {
        return;
    }

    vlForHalfNumber_ = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t); // 256 / 2 = 128
    UBBlockSize_ = BLOCK_SIZE; // 32
    elementAfterReduce_ = AscendC::VECTOR_REG_WIDTH / UBBlockSize_; // 256 / 32 = 8

    uint64_t yOffset = Get<Y_IDX>(blockCoord);
    uint64_t yScaleOffset = Get<Y_SCALE_IDX>(blockCoord);
    VFDoSwigluForMX(singleM_); // switch(x)*y 计算quant quantScale
    TransMxScaleLayout(singleM_, scaleBlockN_); // 重排scale
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    CopyOutputFromUb2Gm(singleM_, yOffset, quantOutput_);
    CopyScaleFromUb2Gm(singleM_, yScaleOffset, quantScaleBlockOutput_);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
    AscendC::AtomicAdd(groupFlagListGmAddr_, 1);
    return;
}

#endif // BLOCK_EPILOGUE_SWIGLU_QUANT_H
#endif