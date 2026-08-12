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

enum class ActMode : uint8_t {
    SWIGLU = 0,
    SITU = 1,
};

enum class ActSubMode : uint8_t {
    DEFAULT = 0,
    LINEAR = 1,
};

namespace {
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr uint32_t Y_IDX = 0;
constexpr uint32_t Y_SCALE_IDX = 1;
constexpr uint32_t GROUP_FLAG_IDX = 2;
constexpr uint32_t M_LOC_IDX = 4;
constexpr uint32_t BLOCK_SIZE = 32;
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
constexpr int64_t QUANT_ONCE_NUM = 256;
constexpr int64_t QUANT_ONCE_NUM_FP4 = 128;
constexpr int64_t SCALE_ONCE_NUM = 8;
constexpr int64_t CONST_64 = 64;
constexpr uint32_t VF_LEN_FP32 = AscendC::VECTOR_REG_WIDTH / sizeof(float);
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
static constexpr AscendC::MicroAPI::CastTrait CAST_ZERO = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::MicroAPI::CastTrait CAST_ONE = {
    AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_FP16_BF16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_80 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_81 = {
    AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_82 = {
    AscendC::MicroAPI::RegLayout::TWO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_83 = {
    AscendC::MicroAPI::RegLayout::THREE, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
#define BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS \
    template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeX2Scale_, typename DataTypeX1Scale_, \
              bool IsTensorList_, uint32_t TileM, uint32_t TileN, bool TopkWeightsPrefetch, bool IsInterleaved_, \
              bool IsWaveFlagGrained_>
#define BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS \
    DataTypeOut_, DataTypeIn_, DataTypeX2Scale_, DataTypeX1Scale_, IsTensorList_, TileM, TileN, TopkWeightsPrefetch, \
        IsInterleaved_, IsWaveFlagGrained_

// 前向声明：激活分块上下文结构体（仅含标量与 UB 指针，可跨函数传递）
// 完整定义见 BlockEpilogueSwigluMxQuant 类定义之后
template <typename DataTypeInT>
struct ActivationContext;

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeX2Scale_, typename DataTypeX1Scale_,
          bool IsTensorList_, uint32_t TileM = 256, uint32_t TileN = 256, bool TopkWeightsPrefetch = false,
          bool IsInterleaved_ = false, bool IsWaveFlagGrained_ = false>
class BlockEpilogueSwigluMxQuant {
public:
    __aicore__ inline BlockEpilogueSwigluMxQuant() {}

    static constexpr uint32_t MAX_TILE_M = TileM;
    static constexpr uint32_t MAX_SINGLE_MN = TileM * TileN;

    struct Arguments {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR yScaleGmAddr{nullptr};
        GM_ADDR groupFlagListGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR metaInfoGmAddr{nullptr};
        float clampLimit{0.0f};
        uint8_t actMode{0};
        uint8_t actSubMode{0};
        float activationAlpha{1.0f};
        float activationBeta{1.0f};
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
    // Computes and writes one activation/quant tile without notifying GMM2.
    __aicore__ inline void operator()(const BlockShape &blockShape, const BlockCoord &blockCoord,
                                      uint16_t pingpongIdx = 0);
    __aicore__ inline void UpdateGlobalAddr(const BlockCoord &baseOffset, uint32_t flagElementOffset = 0);
    // Pipeline controllers call this immediately after each completed tile.
    __aicore__ inline void NotifyCompletedTile(uint32_t flagSlotIdx = 0);
    __aicore__ inline void UpdateNextProblem(const ProblemShape &problemShape);

    // 权重预加载相关：由 ComputeSwigluTileGeneric 提前 DataCopy，VFDoSwigluAndQuantForMX 直接消费
    AscendC::LocalTensor<float> weightUb_;
    AscendC::GlobalTensor<float> metaInfoGm_;

private:
    __aicore__ inline void VFDoSwigluForMX(uint16_t mSize, uint16_t pingpongIdx = 0, uint64_t mLoc = 0);

    template <SwigluQuantMsg::QuantMode quantMode, bool IsInterleavedSrc = false, ActMode Activation = ActMode::SWIGLU,
              ActSubMode ActSub = ActSubMode::DEFAULT>
    __aicore__ inline void VFDoSwigluAndQuantForMX(__ubuf__ int8_t *outputDst, __ubuf__ uint16_t *scaleDst,
                                                   __ubuf__ DataTypeIn *firstSrc, __ubuf__ DataTypeIn *secondSrc,
                                                   __ubuf__ bfloat16_t *gluResAddr, __ubuf__ uint16_t *maxExpAddr,
                                                   __ubuf__ uint16_t *halfScaleLocalAddr, uint16_t mSize,
                                                   uint16_t nSize, uint64_t mLoc = 0);

    // 激活流程独立函数：每种激活一个完整函数，自含全部寄存器声明与 __VEC_SCOPE__
    // （RegTensor 不能跨函数传递，故各激活函数内独立声明寄存器）
    // 由 VFDoSwigluAndQuantForMX 按模板参数 Activation 分派调用
    template <bool IsInterleaved>
    __aicore__ inline void ActivationFlowSwiGLU(const ActivationContext<DataTypeIn> &ctx);

    template <ActSubMode ActSub, bool IsInterleaved>
    __aicore__ inline void ActivationFlowSiTU(const ActivationContext<DataTypeIn> &ctx);

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

    __aicore__ inline void CopyScaleFromUb2GmCompact(uint64_t blockCount, uint64_t offset,
                                                     AscendC::LocalTensor<int8_t> &src);
    // GM ADDR
    AscendC::GlobalTensor<int8_t> quantOutputGlobal_;
    AscendC::GlobalTensor<int8_t> quantScaleGlobal_;
    __gm__ int32_t *groupFlagListGmAddr_;
    GM_ADDR yGmAddr_{nullptr};
    GM_ADDR yScaleGmAddr_{nullptr};
    GM_ADDR groupFlagListGmBaseAddr_{nullptr};
    GM_ADDR metaInfoGmAddr_{nullptr};

    // UB ADDR
    AscendC::LocalTensor<DataTypeIn> l0cOutUbFirst_{AscendC::TPosition::VECIN, 0, MAX_SINGLE_MN};
    static constexpr uint32_t kUbSecondOffset =
        (MAX_SINGLE_MN * sizeof(DataTypeIn) * 2U <= 256U * 1024U) ? (MAX_SINGLE_MN * sizeof(DataTypeIn)) : 0U;
    AscendC::LocalTensor<DataTypeIn> l0cOutUbSecond_{AscendC::TPosition::VECIN, kUbSecondOffset, MAX_SINGLE_MN};
    AscendC::LocalTensor<int8_t> quantOutput_;
    AscendC::LocalTensor<int8_t> quantScaleOutput_;
    AscendC::LocalTensor<bfloat16_t> gluRes_;
    AscendC::LocalTensor<uint16_t> maxExp_;
    AscendC::LocalTensor<uint16_t> halfScale_;

    int64_t n_;
    int64_t scaleN_;
    uint32_t subBlockIdx_ = AscendC::GetSubBlockIdx();
    uint32_t singleM_; // cur singleShapeM
    uint32_t singleN_;
    bool isBiasEpilogue_ = false;
    int64_t UBBlockSize_ = 0;
    uint32_t vlForHalfNumber_ = 0;
    uint16_t elementAfterReduce_ = 0;
    uint16_t fpEmax_ = 0;

    BlockCoord blockCoord_{0, 0, 0, 0, 0, 0};
    float clampLimit_{0.0f};
    uint8_t actMode_{0};
    uint8_t actSubMode_{0};
    float activationAlpha_{1.0f};
    float activationBeta_{1.0f};
};

// ============================================================================
// ActivationContext: 激活分块上下文
// 仅含标量与 UB 指针（不含 RegTensor），可安全跨函数传递给各 ActivationFlow*
// 由 VFDoSwigluAndQuantForMX 填充后分派给具体激活函数
// ============================================================================
template <typename DataTypeInT>
struct ActivationContext {
    // 源数据地址
    __ubuf__ DataTypeInT *firstSrc = nullptr;
    __ubuf__ DataTypeInT *secondSrc = nullptr;
    __ubuf__ bfloat16_t *gluResAddr = nullptr;

    // 对齐参数
    uint32_t nSrcUbAligned = 0;
    uint32_t nDstUbAligned = 0;

    // 分块循环控制
    uint16_t dim0VfTimes = 0;
    uint16_t dim1VfTimes = 0;
    uint32_t dim1Tail = 0;
    uint16_t dim1TailTimes = 0;
    uint16_t dim1Tail2 = 0;

    // 尾循环 mask
    uint32_t mask1Num = 0;
    uint32_t mask2Num = 0;
    uint32_t mask3Num = 0;

    // 尾循环地址
    __ubuf__ DataTypeInT *firstTailAddr = nullptr;
    __ubuf__ DataTypeInT *secondTailAddr = nullptr;
    __ubuf__ bfloat16_t *swigluTailAddr1 = nullptr;
    __ubuf__ bfloat16_t *swigluTailAddr2 = nullptr;

    // 权重预取地址（nullptr 表示不启用 TopkWeightsPrefetch）
    __ubuf__ float *weightUbAddr = nullptr;

    // 激活参数（beta/alpha 默认 1.0，倒数同步 1.0，兜底为恒等变换）
    float clampLimit = 0.0f;
    float beta = 1.0f;
    float invBeta = 1.0f;
    float alpha = 1.0f;
    float invAlpha = 1.0f;
};

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::Init(Params const &params)
{
    if constexpr (g_coreType == AscendC::AIC) {
        return;
    }
    // Init is called with a temporary Arguments object. Keep the GM bases owned by this
    // epilogue so UpdateGlobalAddr never dereferences a dangling parameter reference.
    yGmAddr_ = params.yGmAddr;
    yScaleGmAddr_ = params.yScaleGmAddr;
    groupFlagListGmBaseAddr_ = params.groupFlagListGmAddr;
    clampLimit_ = params.clampLimit;
    actMode_ = params.actMode;
    actSubMode_ = params.actSubMode;
    activationAlpha_ = params.activationAlpha;
    activationBeta_ = params.activationBeta;
    metaInfoGmAddr_ = params.metaInfoGmAddr;
    metaInfoGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(metaInfoGmAddr_));
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
    // 当前master默认非interleaved，保留完整256x256空间；interleaved预留按half tile + pingpong偏移使用。
    constexpr uint32_t MAX_SINGLE_MN_ALIAS =
        IsInterleaved_ ? MAX_SINGLE_MN / MegaMoeImpl::SWIGLU_N_HALF : MAX_SINGLE_MN;
    constexpr uint32_t gluResOffset = 0;
    gluRes_ = AscendC::LocalTensor<bfloat16_t>(AscendC::TPosition::VECCALC, gluResOffset, MAX_SINGLE_MN_ALIAS);
    constexpr uint32_t quantOutputOffset = gluResOffset + MAX_SINGLE_MN_ALIAS * sizeof(bfloat16_t);
    quantOutput_ = AscendC::LocalTensor<int8_t>(AscendC::TPosition::VECOUT, quantOutputOffset, MAX_SINGLE_MN_ALIAS);
    constexpr uint32_t quantScaleOffset = quantOutputOffset + MAX_SINGLE_MN_ALIAS * sizeof(int8_t);
    quantScaleOutput_ = AscendC::LocalTensor<int8_t>(AscendC::TPosition::VECOUT, quantScaleOffset,
                                                     MAX_SINGLE_MN_ALIAS / AscendC::ONE_BLK_SIZE);
    constexpr uint32_t maxExpOffset = quantScaleOffset + MAX_SINGLE_MN_ALIAS / AscendC::ONE_BLK_SIZE * sizeof(int8_t);
    maxExp_ = AscendC::LocalTensor<uint16_t>(AscendC::TPosition::VECCALC, maxExpOffset,
                                             MAX_SINGLE_MN_ALIAS / AscendC::ONE_BLK_SIZE);
    constexpr uint32_t halfScaleOffset = maxExpOffset + MAX_SINGLE_MN_ALIAS / AscendC::ONE_BLK_SIZE * sizeof(uint16_t);
    halfScale_ = AscendC::LocalTensor<uint16_t>(AscendC::TPosition::VECCALC, halfScaleOffset,
                                                MAX_SINGLE_MN_ALIAS / AscendC::ONE_BLK_SIZE);

    // weight UB: 放在 VECIN 区之后的安全区域，仅 mode=1 分配
    if constexpr (TopkWeightsPrefetch) {
        constexpr uint32_t vecinEnd = MAX_SINGLE_MN * sizeof(DataTypeIn) * 2U;
        weightUb_ = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, vecinEnd, MAX_TILE_M * INT32_PER_256B);
    }
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateGlobalAddr(
    const BlockCoord &baseOffset, uint32_t flagElementOffset)
{
    if constexpr (g_coreType == AscendC::AIV) {
        quantOutputGlobal_.SetGlobalBuffer((__gm__ int8_t *)yGmAddr_ + Get<Y_IDX>(baseOffset));
        quantScaleGlobal_.SetGlobalBuffer((__gm__ int8_t *)yScaleGmAddr_ + Get<Y_SCALE_IDX>(baseOffset));
        if (groupFlagListGmBaseAddr_ != nullptr) {
            groupFlagListGmAddr_ = (__gm__ int32_t *)groupFlagListGmBaseAddr_ +
                                   Get<GROUP_FLAG_IDX>(baseOffset) * INT_CACHELINE + flagElementOffset;
        } else {
            groupFlagListGmAddr_ = nullptr;
        }
    }
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateNextProblem(
    const ProblemShape &problemShape)
{
    n_ = Get<N_VALUE>(problemShape); // n/2
    scaleN_ =
        Ops::Base::CeilDiv(static_cast<uint64_t>(n_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyOutputFromUb2Gm(
    uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src)
{
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockCount = blockCount; // 128
    if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp4x2_e1m2_t>::value) {
        ub2GmParams.blockLen = singleN_ >> 1;
        ub2GmParams.dstStride = (n_ - singleN_) >> 1;
        offset = offset >> 1;
    } else {
        uint64_t nDstUbAligned =
            Ops::Base::CeilAlign(static_cast<uint64_t>(singleN_), static_cast<uint64_t>(AscendC::ONE_BLK_SIZE));
        ub2GmParams.blockLen = singleN_; // 256
        ub2GmParams.srcStride = (nDstUbAligned - singleN_) / AscendC::ONE_BLK_SIZE;
        ub2GmParams.dstStride = n_ - singleN_;
    }
    AscendC::DataCopyPad(quantOutputGlobal_[offset], src, ub2GmParams);
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyScaleFromUb2GmCompact(
    uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src)
{
    AscendC::DataCopyExtParams ub2GmParams{0, 0, 0, 0, 0};
    auto blockScaleN = Ops::Base::CeilDiv(static_cast<uint64_t>(singleN_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) *
                       MXFP_MULTI_BASE_SIZE; // 256 / 32 = 8
    // scale layout in UB is already compact: (mSize, blockScaleN). Compact copy avoids (mSize*8)->(mSize,32).
    ub2GmParams.blockCount = blockCount; // 128
    ub2GmParams.blockLen = blockScaleN;  // 8
    ub2GmParams.srcStride = 0;
    ub2GmParams.dstStride = scaleN_ - blockScaleN;
    AscendC::DataCopyPad<int8_t, AscendC::PaddingMode::Compact>(quantScaleGlobal_[offset], src, ub2GmParams);
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeMaxExp(
    __ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *maxExpAddr, uint32_t totalCountInUB, uint16_t loopNum)
{
    int64_t onceNum = QUANT_ONCE_NUM;
    int64_t scaleNum = SCALE_ONCE_NUM;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0, vdExp1;
        AscendC::MicroAPI::RegTensor<uint16_t> vdExpExtract0, vdExpExtract1;
        AscendC::MicroAPI::RegTensor<uint16_t> expMaskBF16, vdMaxExp;
        AscendC::MicroAPI::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);
        AscendC::MicroAPI::MaskReg scaleMask1, scaleMask2;
        AscendC::MicroAPI::UnalignReg u1;
        for (uint16_t i = 0; i < loopNum; i++) {
            scaleMask1 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(totalCountInUB);
            scaleMask2 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(totalCountInUB);
            AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, onceNum); // copy two chunks from srcAddr to regbase
            AscendC::MicroAPI::And(vdExpExtract0, (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp0, expMaskBF16,
                                   scaleMask1);
            AscendC::MicroAPI::And(vdExpExtract1, (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp1, expMaskBF16,
                                   scaleMask1);
            AscendC::MicroAPI::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
            AscendC::MicroAPI::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);
            AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, vdMaxExp, u1, scaleNum);
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
    int64_t onceNum = QUANT_ONCE_NUM_FP4;
    int64_t onceNumMxScale = CONST_64;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> expMask, vdMaxExp;
        AscendC::MicroAPI::Duplicate(expMask, MAX_EXP_FOR_BF16); // MAX_EXP_FOR_BF16表示bf16正无穷 大小：128
        AscendC::MicroAPI::MaskReg cmpResult, zeroMask, cmpResultSub, preMaskScale;
        AscendC::MicroAPI::RegTensor<uint16_t> maxExpValue, sharedExp, scaleValue, scaleBias, halfScale;
        AscendC::MicroAPI::Duplicate(maxExpValue, fpEmax_);     // 0x0780 大小：128 对应bf16指数位后四位
        AscendC::MicroAPI::Duplicate(scaleBias, BF16_EXP_BIAS); // 0x7f00 大小：128
        AscendC::MicroAPI::RegTensor<uint16_t> fp8NanRegTensor, zeroRegTensor, nanRegTensor;
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8); // 0x00ff 大小：128
        AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);                 // 0 大小：128
        AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);  // 0x7f81 大小：128
        AscendC::MicroAPI::MaskReg invalidDataMask, specialDataMask;
        AscendC::MicroAPI::RegTensor<uint16_t> specialExpRegTensor;
        AscendC::MicroAPI::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);   // 0x0040 大小：128
        for (uint16_t i = 0; i < loopNumScale; i++) {                               // 8
            preMaskScale = AscendC::MicroAPI::UpdateMask<uint16_t>(totalScaleInUB); // 128*8
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                vdMaxExp, maxExpAddr, onceNum); // 每次搬运128个数到vdMaxExp
            // 得到不等于INF的结果掩码 cmpResult
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale);
            // 得到不等于0的结果掩码 zeroMask
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
            // 得到小于或等于0x0780的结果掩码 invalidDataMask
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue,
                                                                       preMaskScale);
            // 将vdMaxExp中小于或等于0x0780的结果替换成0x0780
            AscendC::MicroAPI::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);
            AscendC::MicroAPI::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale); // sharedExp = vdMaxExp - 0x0780
            // 逻辑右移7位 当前指数位在减去0x0780后，已移至最低位
            AscendC::MicroAPI::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
            // 将scaleValue中INF的结果替换成0x00ff
            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
            // 将scaleValue中原来是0的结果替换成0
            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);
            // 将scaleValue中数取低半部分，搬运到mxScaleLocalAddr uint16--int8
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(mxScaleLocalAddr, scaleValue,
                                                                                     onceNumMxScale, preMaskScale);
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
            // 将128个数搬运到halfScaleLocalAddr uint16--uint16
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                halfScaleLocalAddr, halfScale, onceNum, preMaskScale);
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
    using T = bfloat16_t;
    using U = DataTypeOut;
    (void)totalCountInUB;
    int64_t elementAfterReduce = SCALE_ONCE_NUM;
    int64_t onceXNum = QUANT_ONCE_NUM;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
        AscendC::MicroAPI::RegTensor<T> vdExp0, vdExp1;
        AscendC::MicroAPI::RegTensor<float> vdExp0FP32Zero, vdExp0FP32One;
        AscendC::MicroAPI::RegTensor<float> vdExp1FP32Zero, vdExp1FP32One;
        AscendC::MicroAPI::RegTensor<U> vdExp0FP8Zero, vdExp0FP8One;
        AscendC::MicroAPI::RegTensor<U> vdExp1FP8Zero, vdExp1FP8One;
        AscendC::MicroAPI::MaskReg maskAll =
            AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg maskAllB8 =
            AscendC::MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        for (uint16_t i = 0; i < loopNum; i++) {
            // DIST_DINTLV_B16:双搬入模式，读取2*VL长度数据，将偶数索引的元素存入dst0，奇数索引的元素存入dst1
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr,
                                                                                      onceXNum);
            // 将halfScale中的8个数uint16广播到halfScaleForMul中，halfScale[0]*16 halfScale[1]*16...
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr,
                                                                                   elementAfterReduce);
            // vdExp0/vdExp1乘以广播后的halfScale，得到量化前缩放值
            AscendC::MicroAPI::Mul(vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, maskAll);
            AscendC::MicroAPI::Mul(vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, maskAll);
            AscendC::MicroAPI::Cast<float, T, CAST_ZERO>(vdExp0FP32Zero, vdExp0, maskAll);
            AscendC::MicroAPI::Cast<float, T, CAST_ONE>(vdExp0FP32One, vdExp0, maskAll);
            AscendC::MicroAPI::Cast<float, T, CAST_ZERO>(vdExp1FP32Zero, vdExp1, maskAll);
            AscendC::MicroAPI::Cast<float, T, CAST_ONE>(vdExp1FP32One, vdExp1, maskAll);
            // CAST_32_TO_80/82/81/83把4路fp32 lane cast到fp8 lane，后续按uint8合并成连续fp8输出
            AscendC::MicroAPI::Cast<U, float, CAST_32_TO_80>(vdExp0FP8Zero, vdExp0FP32Zero, maskAll);
            AscendC::MicroAPI::Cast<U, float, CAST_32_TO_82>(vdExp0FP8One, vdExp0FP32One, maskAll);
            AscendC::MicroAPI::Cast<U, float, CAST_32_TO_81>(vdExp1FP8Zero, vdExp1FP32Zero, maskAll);
            AscendC::MicroAPI::Cast<U, float, CAST_32_TO_83>(vdExp1FP8One, vdExp1FP32One, maskAll);
            AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8One, maskAllB8);
            AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp1FP8Zero, maskAllB8);
            AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp1FP8One, maskAllB8);
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_NORM_B8>(
                // 将src中有效元素的低8bit数据连续存储于dst中
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp0FP8Zero, onceXNum, maskAllB8);
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
    int64_t elementAfterReduce = SCALE_ONCE_NUM;
    int64_t onceXNum = QUANT_ONCE_NUM;
    int64_t onceYNum = OUT_ELE_NUM_ONE_BLK;
    static constexpr AscendC::MicroAPI::CastTrait castTrait = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg dataMask1;
        AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
        AscendC::MicroAPI::RegTensor<T> vdExp0, vdExp1;
        AscendC::MicroAPI::RegTensor<U> vdExp0FP4, vdExp1FP4;
        for (uint16_t i = 0; i < loopNum; i++) {
            dataMask1 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr,
                                                                                      onceXNum);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr,
                                                                                   elementAfterReduce);
            AscendC::MicroAPI::Mul(vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);
            AscendC::MicroAPI::Mul(vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);
            AscendC::MicroAPI::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
            AscendC::MicroAPI::Cast<U, T, castTrait>(vdExp0FP4, vdExp0, dataMask1);
            AscendC::MicroAPI::Cast<U, T, castTrait>(vdExp1FP4, vdExp1, dataMask1);
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp0FP4, onceYNum, dataMask1);
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp1FP4, onceYNum, dataMask1);
        }
    }
    return;
}

// ============================================================================
// COMPUTE_TANH_TWOPATH: 两段式 tanh 宏（多项式路 + sigmoid 分解路）
//   多项式路(|x|<0.6): x*(1 + c1*x^2 + c2*x^4 + c3*x^6 + c4*x^8), Horner + FusedMulDstAdd
//   sigmoid路(|x|>=0.6): 2/(1+e^{-2x}) - 1, 天然保号无相消
//   系数来源: tanh_dag.h / situ_mx_quant_common.h
//   c1=-0.333327681, c2=0.133152977, c3=-0.0523039624, c4=0.0157396831
// 用法: COMPUTE_TANH_TWOPATH(result, zReg, msk, oneReg, absReg, sqrReg,
//                           polyReg, tmpReg, expReg, c1Reg, c2Reg, cmpMask);
// 注意: 宏在函数内展开，所用寄存器须在调用点函数作用域内声明；
//       宏会覆盖 sqrReg/polyReg/tmpReg/expReg/absReg/cmpMask，结果写入 result
// ============================================================================
#define COMPUTE_TANH_TWOPATH(result, zReg, msk, oneReg, absReg, sqrReg, \
                             polyReg, tmpReg, expReg, c1Reg, c2Reg, cmpMask) \
    do { \
        AscendC::MicroAPI::Mul(sqrReg, zReg, zReg, msk); \
        AscendC::MicroAPI::Muls(polyReg, sqrReg, 0.0157396831f, msk); \
        AscendC::MicroAPI::Adds(polyReg, polyReg, -0.0523039624f, msk); \
        AscendC::MicroAPI::FusedMulDstAdd(polyReg, sqrReg, c2Reg, msk); \
        AscendC::MicroAPI::FusedMulDstAdd(polyReg, sqrReg, c1Reg, msk); \
        AscendC::MicroAPI::Mul(polyReg, polyReg, sqrReg, msk); \
        AscendC::MicroAPI::FusedMulDstAdd(polyReg, zReg, zReg, msk); \
        AscendC::MicroAPI::Muls(expReg, zReg, -2.0f, msk); \
        AscendC::MicroAPI::Exp(expReg, expReg, msk); \
        AscendC::MicroAPI::Adds(expReg, expReg, 1.0f, msk); \
        AscendC::MicroAPI::Div(tmpReg, oneReg, expReg, msk); \
        AscendC::MicroAPI::Muls(tmpReg, tmpReg, 2.0f, msk); \
        AscendC::MicroAPI::Adds(tmpReg, tmpReg, -1.0f, msk); \
        AscendC::MicroAPI::Abs(absReg, zReg, msk); \
        AscendC::MicroAPI::CompareScalar<float, AscendC::CMPMODE::GE>(cmpMask, absReg, \
                                                                      0.60000002384185791016f, msk); \
        AscendC::MicroAPI::Select(result, tmpReg, polyReg, cmpMask); \
    } while (0)

// ============================================================================
// ActivationFlowSwiGLU: SwiGLU 激活完整计算流
//   swish(x1) * x2, 其中 swish(x) = x / (exp(-x) + 1) = x * sigmoid(x)
//   含主循环 + 尾循环 + pad 填零，所有寄存器在本函数内声明
// ============================================================================
BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
template <bool IsInterleaved>
__aicore__ inline void
BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ActivationFlowSwiGLU(
    const ActivationContext<DataTypeIn> &ctx)
{
    const float scalarOne = 1.0f;
    const float negScalarOne = -1.0f;
    bfloat16_t numZero = 0;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<DataTypeIn> vregX1;
        AscendC::MicroAPI::RegTensor<DataTypeIn> vregX2;
        AscendC::MicroAPI::RegTensor<float> vregX1F;
        AscendC::MicroAPI::RegTensor<float> vregX2F;
        AscendC::MicroAPI::RegTensor<float> negReg;
        AscendC::MicroAPI::RegTensor<float> expReg;
        AscendC::MicroAPI::RegTensor<float> addsReg;
        AscendC::MicroAPI::RegTensor<float> sigmoidReg;
        AscendC::MicroAPI::RegTensor<float> outFReg;
        AscendC::MicroAPI::RegTensor<float> weightReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> outTReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> zeroReg;
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        uint32_t mask1Num = ctx.mask1Num;
        uint32_t mask2Num = ctx.mask2Num;
        uint32_t mask3Num = ctx.mask3Num;
        AscendC::MicroAPI::MaskReg mask1 = AscendC::MicroAPI::UpdateMask<float>(mask1Num);
        AscendC::MicroAPI::MaskReg mask2 = AscendC::MicroAPI::UpdateMask<float>(mask2Num);
        AscendC::MicroAPI::MaskReg mask3 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(mask3Num);
        for (uint16_t dim0vfLoopIdx = 0; dim0vfLoopIdx < ctx.dim0VfTimes; dim0vfLoopIdx++) {
            if constexpr (TopkWeightsPrefetch) {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                    weightReg, ctx.weightUbAddr + dim0vfLoopIdx * INT32_PER_256B + WEIGHT_INDEX);
            }
            for (uint16_t dim1vfLoopIdx = 0; dim1vfLoopIdx < ctx.dim1VfTimes; dim1vfLoopIdx++) {
                AscendC::MicroAPI::AddrReg srcIdxOffset = AscendC::MicroAPI::CreateAddrReg<DataTypeIn>(
                    dim0vfLoopIdx, ctx.nSrcUbAligned, dim1vfLoopIdx, VF_LEN_FP32);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX1, ctx.firstSrc, srcIdxOffset);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX2, ctx.secondSrc, srcIdxOffset);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX1F, vregX1, mask);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX2F, vregX2, mask);

                AscendC::MicroAPI::Mins(vregX1F, vregX1F, ctx.clampLimit, mask);
                AscendC::MicroAPI::Mins(vregX2F, vregX2F, ctx.clampLimit, mask);
                AscendC::MicroAPI::Maxs(vregX2F, vregX2F, -ctx.clampLimit, mask);

                // === SwiGLU: swish(x1) * x2 ===
                AscendC::MicroAPI::Muls(negReg, vregX1F, negScalarOne, mask); // -x
                AscendC::MicroAPI::Exp(expReg, negReg, mask);                 // exp(-x)
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask);    // exp(-x)+1
                AscendC::MicroAPI::Div(sigmoidReg, vregX1F, addsReg, mask);   // swish(x)=x/(exp(-x)+1)
                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, vregX2F, mask);   // swish(x)*y
                // === 激活结束 ===

                if constexpr (TopkWeightsPrefetch) {
                    AscendC::MicroAPI::Mul(outFReg, outFReg, weightReg, mask);
                }

                AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask);
                AscendC::MicroAPI::AddrReg outOffset = AscendC::MicroAPI::CreateAddrReg<bfloat16_t>(
                    dim0vfLoopIdx, ctx.nDstUbAligned, dim1vfLoopIdx, VF_LEN_FP32);
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    ctx.gluResAddr, outTReg, outOffset, mask);
            }
            AscendC::MicroAPI::AddrReg srcIdxOffset1 =
                AscendC::MicroAPI::CreateAddrReg<DataTypeIn>(dim0vfLoopIdx, ctx.nSrcUbAligned);
            AscendC::MicroAPI::AddrReg outOffset1 =
                AscendC::MicroAPI::CreateAddrReg<bfloat16_t>(dim0vfLoopIdx, ctx.nDstUbAligned);
            for (uint16_t aa = 0; aa < ctx.dim1TailTimes; aa++) {
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX1, ctx.firstTailAddr, srcIdxOffset1);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX2, ctx.secondTailAddr, srcIdxOffset1);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX1F, vregX1, mask1);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX2F, vregX2, mask1);

                AscendC::MicroAPI::Mins(vregX1F, vregX1F, ctx.clampLimit, mask1);
                AscendC::MicroAPI::Mins(vregX2F, vregX2F, ctx.clampLimit, mask1);
                AscendC::MicroAPI::Maxs(vregX2F, vregX2F, -ctx.clampLimit, mask1);

                // === SwiGLU: swish(x1) * x2 (tail, mask1) ===
                AscendC::MicroAPI::Muls(negReg, vregX1F, negScalarOne, mask1);
                AscendC::MicroAPI::Exp(expReg, negReg, mask1);
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask1);
                AscendC::MicroAPI::Div(sigmoidReg, vregX1F, addsReg, mask1);
                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, vregX2F, mask1);
                // === 激活结束 ===

                if constexpr (TopkWeightsPrefetch) {
                    AscendC::MicroAPI::Mul(outFReg, outFReg, weightReg, mask1);
                }

                AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask1);
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    ctx.swigluTailAddr1, outTReg, outOffset1, mask2);
            }
            for (uint16_t cc = 0; cc < ctx.dim1Tail2; cc++) {
                AscendC::MicroAPI::Duplicate(zeroReg, numZero);
                AscendC::MicroAPI::DataCopy<bfloat16_t>(ctx.swigluTailAddr2, zeroReg, outOffset1, mask3);
            }
        }
    }
}

// ============================================================================
// ActivationFlowSiTU: SiTU 激活完整计算流
//   situ_a = beta * tanh(gate/beta) * sigmoid(gate)
//   ActSub==LINEAR: out = situ_a * (alpha * tanh(up/alpha))
//   ActSub==DEFAULT: out = situ_a * up
//   tanh 采用两段式（多项式路 + sigmoid 分解路），保号无相消
//   参照 situ_mx_quant_common.h::ComputeVfSitu，含主循环 + 尾循环 + pad 填零
// ============================================================================
BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
template <ActSubMode ActSub, bool IsInterleaved>
__aicore__ inline void
BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ActivationFlowSiTU(
    const ActivationContext<DataTypeIn> &ctx)
{
    // 标量常量
    const float tanhC1 = -0.333327681f;
    const float tanhC2 = 0.133152977f;
    const float scalarOne = 1.0f;
    const float negScalarOne = -1.0f;
    bfloat16_t numZero = 0;

    __VEC_SCOPE__
    {
        // 通用寄存器（搬入/clamp/store）
        AscendC::MicroAPI::RegTensor<DataTypeIn> vregX1;
        AscendC::MicroAPI::RegTensor<DataTypeIn> vregX2;
        AscendC::MicroAPI::RegTensor<float> vregX1F;
        AscendC::MicroAPI::RegTensor<float> vregX2F;
        AscendC::MicroAPI::RegTensor<float> outFReg;
        AscendC::MicroAPI::RegTensor<float> weightReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> outTReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> zeroReg;

        // SiTU 专用寄存器
        AscendC::MicroAPI::RegTensor<float> negReg;
        AscendC::MicroAPI::RegTensor<float> expReg;
        AscendC::MicroAPI::RegTensor<float> addsReg;
        AscendC::MicroAPI::RegTensor<float> sigmoidReg;
        AscendC::MicroAPI::RegTensor<float> zReg;
        AscendC::MicroAPI::RegTensor<float> betaReg;
        AscendC::MicroAPI::RegTensor<float> oneReg;

        // tanh 两段式专用寄存器
        AscendC::MicroAPI::RegTensor<float> absReg;
        AscendC::MicroAPI::RegTensor<float> sqrReg;
        AscendC::MicroAPI::RegTensor<float> polyReg;
        AscendC::MicroAPI::RegTensor<float> tanhTmpReg;
        AscendC::MicroAPI::RegTensor<float> c1Reg;
        AscendC::MicroAPI::RegTensor<float> c2Reg;
        AscendC::MicroAPI::MaskReg cmpMaskReg;

        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        uint32_t mask1Num = ctx.mask1Num;
        uint32_t mask2Num = ctx.mask2Num;
        uint32_t mask3Num = ctx.mask3Num;
        AscendC::MicroAPI::MaskReg mask1 = AscendC::MicroAPI::UpdateMask<float>(mask1Num);
        AscendC::MicroAPI::MaskReg mask2 = AscendC::MicroAPI::UpdateMask<float>(mask2Num);
        AscendC::MicroAPI::MaskReg mask3 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(mask3Num);

        // 循环前常量初始化（全量广播，不依赖 mask）
        AscendC::MicroAPI::Duplicate(oneReg, scalarOne);
        AscendC::MicroAPI::Duplicate(c1Reg, tanhC1);
        AscendC::MicroAPI::Duplicate(c2Reg, tanhC2);

        for (uint16_t dim0vfLoopIdx = 0; dim0vfLoopIdx < ctx.dim0VfTimes; dim0vfLoopIdx++) {
            if constexpr (TopkWeightsPrefetch) {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                    weightReg, ctx.weightUbAddr + dim0vfLoopIdx * INT32_PER_256B + WEIGHT_INDEX);
            }
            for (uint16_t dim1vfLoopIdx = 0; dim1vfLoopIdx < ctx.dim1VfTimes; dim1vfLoopIdx++) {
                AscendC::MicroAPI::AddrReg srcIdxOffset = AscendC::MicroAPI::CreateAddrReg<DataTypeIn>(
                    dim0vfLoopIdx, ctx.nSrcUbAligned, dim1vfLoopIdx, VF_LEN_FP32);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX1, ctx.firstSrc, srcIdxOffset);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX2, ctx.secondSrc, srcIdxOffset);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX1F, vregX1, mask);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX2F, vregX2, mask);

                AscendC::MicroAPI::Mins(vregX1F, vregX1F, ctx.clampLimit, mask);
                AscendC::MicroAPI::Mins(vregX2F, vregX2F, ctx.clampLimit, mask);
                AscendC::MicroAPI::Maxs(vregX2F, vregX2F, -ctx.clampLimit, mask);

                // === SiTU 激活（主循环, mask）===
                // gate路: z = gate / beta
                AscendC::MicroAPI::Muls(zReg, vregX1F, ctx.invBeta, mask);
                // tanh(gate/beta) — 两段式, 结果存 sigmoidReg
                COMPUTE_TANH_TWOPATH(sigmoidReg, zReg, mask, oneReg, absReg, sqrReg, polyReg, tanhTmpReg, expReg,
                                     c1Reg, c2Reg, cmpMaskReg);
                // beta * sigmoid(gate) = beta / (exp(-gate)+1)
                AscendC::MicroAPI::Muls(negReg, vregX1F, negScalarOne, mask); // -gate
                AscendC::MicroAPI::Exp(expReg, negReg, mask);                 // exp(-gate)
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask);    // exp(-gate)+1
                AscendC::MicroAPI::Duplicate(betaReg, ctx.beta, mask);        // betaReg = beta
                AscendC::MicroAPI::Div(expReg, betaReg, addsReg, mask);       // beta/(exp(-gate)+1)
                // situ_a = tanh(gate/beta) * beta * sigmoid(gate)
                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, expReg, mask); // outFReg = situ_a

                if constexpr (ActSub == ActSubMode::LINEAR) {
                    // up路: alpha * tanh(up/alpha), 用 vregX1F 暂存 gate_result
                    AscendC::MicroAPI::Muls(vregX1F, outFReg, scalarOne, mask); // vregX1F = gate_result (保存)
                    AscendC::MicroAPI::Muls(zReg, vregX2F, ctx.invAlpha, mask); // zReg = up/alpha
                    COMPUTE_TANH_TWOPATH(sigmoidReg, zReg, mask, oneReg, absReg, sqrReg, polyReg, tanhTmpReg, expReg,
                                         c1Reg, c2Reg, cmpMaskReg);
                    // alpha * tanh(up/alpha)
                    AscendC::MicroAPI::Muls(sigmoidReg, sigmoidReg, ctx.alpha, mask);
                    // out = gate_result * up_transformed
                    AscendC::MicroAPI::Mul(outFReg, vregX1F, sigmoidReg, mask);
                } else {
                    // SITU: out = situ_a * up
                    AscendC::MicroAPI::Mul(outFReg, outFReg, vregX2F, mask);
                }
                // === 激活结束 ===

                if constexpr (TopkWeightsPrefetch) {
                    AscendC::MicroAPI::Mul(outFReg, outFReg, weightReg, mask);
                }

                AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask);
                AscendC::MicroAPI::AddrReg outOffset = AscendC::MicroAPI::CreateAddrReg<bfloat16_t>(
                    dim0vfLoopIdx, ctx.nDstUbAligned, dim1vfLoopIdx, VF_LEN_FP32);
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    ctx.gluResAddr, outTReg, outOffset, mask);
            }
            AscendC::MicroAPI::AddrReg srcIdxOffset1 =
                AscendC::MicroAPI::CreateAddrReg<DataTypeIn>(dim0vfLoopIdx, ctx.nSrcUbAligned);
            AscendC::MicroAPI::AddrReg outOffset1 =
                AscendC::MicroAPI::CreateAddrReg<bfloat16_t>(dim0vfLoopIdx, ctx.nDstUbAligned);
            for (uint16_t aa = 0; aa < ctx.dim1TailTimes; aa++) {
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX1, ctx.firstTailAddr, srcIdxOffset1);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX2, ctx.secondTailAddr, srcIdxOffset1);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX1F, vregX1, mask1);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX2F, vregX2, mask1);

                AscendC::MicroAPI::Mins(vregX1F, vregX1F, ctx.clampLimit, mask1);
                AscendC::MicroAPI::Mins(vregX2F, vregX2F, ctx.clampLimit, mask1);
                AscendC::MicroAPI::Maxs(vregX2F, vregX2F, -ctx.clampLimit, mask1);

                // === SiTU 激活（尾循环, mask1）===
                AscendC::MicroAPI::Muls(zReg, vregX1F, ctx.invBeta, mask1);
                COMPUTE_TANH_TWOPATH(sigmoidReg, zReg, mask1, oneReg, absReg, sqrReg, polyReg, tanhTmpReg, expReg,
                                     c1Reg, c2Reg, cmpMaskReg);
                AscendC::MicroAPI::Muls(negReg, vregX1F, negScalarOne, mask1);
                AscendC::MicroAPI::Exp(expReg, negReg, mask1);
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask1);
                AscendC::MicroAPI::Duplicate(betaReg, ctx.beta, mask1);
                AscendC::MicroAPI::Div(expReg, betaReg, addsReg, mask1);
                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, expReg, mask1);

                if constexpr (ActSub == ActSubMode::LINEAR) {
                    AscendC::MicroAPI::Muls(vregX1F, outFReg, scalarOne, mask1);
                    AscendC::MicroAPI::Muls(zReg, vregX2F, ctx.invAlpha, mask1);
                    COMPUTE_TANH_TWOPATH(sigmoidReg, zReg, mask1, oneReg, absReg, sqrReg, polyReg, tanhTmpReg, expReg,
                                         c1Reg, c2Reg, cmpMaskReg);
                    AscendC::MicroAPI::Muls(sigmoidReg, sigmoidReg, ctx.alpha, mask1);
                    AscendC::MicroAPI::Mul(outFReg, vregX1F, sigmoidReg, mask1);
                } else {
                    AscendC::MicroAPI::Mul(outFReg, outFReg, vregX2F, mask1);
                }
                // === 激活结束 ===

                if constexpr (TopkWeightsPrefetch) {
                    AscendC::MicroAPI::Mul(outFReg, outFReg, weightReg, mask1);
                }

                AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask1);
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    ctx.swigluTailAddr1, outTReg, outOffset1, mask2);
            }
            for (uint16_t cc = 0; cc < ctx.dim1Tail2; cc++) {
                AscendC::MicroAPI::Duplicate(zeroReg, numZero);
                AscendC::MicroAPI::DataCopy<bfloat16_t>(ctx.swigluTailAddr2, zeroReg, outOffset1, mask3);
            }
        }
    }
}

// ============================================================================
// VFDoSwigluAndQuantForMX: 激活 + MX 量化主入口（重构后为分派器）
//   1. 标量分块计算（对齐、tail、mask）
//   2. 填充 ActivationContext
//   3. 按模板参数 Activation 分派到 ActivationFlowSwiGLU / ActivationFlowSiTU
//   4. MX 量化调度（ComputeMaxExp / ComputeScale / ComputeDataForQuantTargetFp8/Fp4）
// ============================================================================
BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
template <SwigluQuantMsg::QuantMode quantMode, bool IsInterleavedSrc, ActMode Activation, ActSubMode ActSub>
__aicore__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoSwigluAndQuantForMX(
    __ubuf__ int8_t *outputDst, __ubuf__ uint16_t *scaleDst, __ubuf__ DataTypeIn *firstSrc,
    __ubuf__ DataTypeIn *secondSrc, __ubuf__ bfloat16_t *gluResAddr, __ubuf__ uint16_t *maxExpAddr,
    __ubuf__ uint16_t *halfScaleLocalAddr, uint16_t mSize, uint16_t nSize, uint64_t mLoc)
{
    // ---- 标量分块计算（原样保留）----
    uint32_t nSrcUbAligned;
    if constexpr (IsInterleavedSrc) {
        // interleaved源布局为[x1, x2]连续存放在同一行，下一行stride是2*nSize
        nSrcUbAligned = static_cast<uint32_t>(nSize) * 2U;
    } else {
        // 非interleaved源布局为两块独立UB，按原master逻辑只需要对齐单个nSize
        nSrcUbAligned = Ops::Base::CeilAlign(static_cast<uint32_t>(nSize),
                                             static_cast<uint32_t>(AscendC::ONE_BLK_SIZE / sizeof(DataTypeIn)));
    }
    uint32_t nDstUbAligned =
        Ops::Base::CeilAlign(static_cast<uint32_t>(nSize), static_cast<uint32_t>(AscendC::ONE_BLK_SIZE));
    uint16_t dim0VfTimes = mSize;
    uint16_t dim1VfTimes = nSize / VF_LEN_FP32;
    uint32_t dim1Tail = nSize % VF_LEN_FP32;
    uint16_t dim1TailTimes = 0;
    uint16_t dim1Tail2 = 0;
    uint32_t mask1Num = 0;
    uint32_t mask2Num = 0;
    uint32_t mask3Num = 0;
    __ubuf__ DataTypeIn *firstTailAddr = firstSrc;
    __ubuf__ DataTypeIn *secondTailAddr = secondSrc;
    __ubuf__ bfloat16_t *swigluTailAddr1 = gluResAddr;
    __ubuf__ bfloat16_t *swigluTailAddr2 = gluResAddr;
    if (dim1Tail > 0) {
        mask1Num = dim1Tail;
        dim1TailTimes = 1;
        uint32_t padNum = nDstUbAligned - dim1VfTimes * VF_LEN_FP32;
        if (padNum <= VF_LEN_FP32) {
            mask2Num = padNum;
        } else {
            dim1Tail2 = 1;
            mask2Num = VF_LEN_FP32;
            mask3Num = padNum - VF_LEN_FP32;
        }
        uint32_t offsetAlign = dim1VfTimes * VF_LEN_FP32;
        firstTailAddr = firstSrc + offsetAlign;
        secondTailAddr = secondSrc + offsetAlign;
        swigluTailAddr1 = gluResAddr + offsetAlign;
        swigluTailAddr2 = gluResAddr + offsetAlign + dim1TailTimes * VF_LEN_FP32;
    }

    // ---- 填充分块上下文结构体 ----
    ActivationContext<DataTypeIn> ctx;
    ctx.firstSrc = firstSrc;
    ctx.secondSrc = secondSrc;
    ctx.gluResAddr = gluResAddr;
    ctx.nSrcUbAligned = nSrcUbAligned;
    ctx.nDstUbAligned = nDstUbAligned;
    ctx.dim0VfTimes = dim0VfTimes;
    ctx.dim1VfTimes = dim1VfTimes;
    ctx.dim1Tail = dim1Tail;
    ctx.dim1TailTimes = dim1TailTimes;
    ctx.dim1Tail2 = dim1Tail2;
    ctx.mask1Num = mask1Num;
    ctx.mask2Num = mask2Num;
    ctx.mask3Num = mask3Num;
    ctx.firstTailAddr = firstTailAddr;
    ctx.secondTailAddr = secondTailAddr;
    ctx.swigluTailAddr1 = swigluTailAddr1;
    ctx.swigluTailAddr2 = swigluTailAddr2;
    ctx.clampLimit = clampLimit_;
    ctx.beta = activationBeta_;
    ctx.invBeta = (activationBeta_ != 0.0f) ? (1.0f / activationBeta_) : 1.0f;
    ctx.alpha = activationAlpha_;
    ctx.invAlpha = (activationAlpha_ != 0.0f) ? (1.0f / activationAlpha_) : 1.0f;
    if constexpr (TopkWeightsPrefetch) {
        ctx.weightUbAddr = (__ubuf__ float *)weightUb_.GetPhyAddr();
    } else {
        ctx.weightUbAddr = nullptr;
    }

    // ---- 激活分派 ----
    if constexpr (Activation == ActMode::SWIGLU) {
        ActivationFlowSwiGLU<IsInterleavedSrc>(ctx);
    } else if constexpr (Activation == ActMode::SITU) {
        ActivationFlowSiTU<ActSub, IsInterleavedSrc>(ctx);
    }

    // ---- 量化（原样保留，不涉及激活寄存器）----
    uint32_t totalDataInUb = mSize * nDstUbAligned;
    uint32_t totalScaleInUb = totalDataInUb / AscendC::ONE_BLK_SIZE;
    uint16_t loopDataNum = (totalDataInUb + vlForHalfNumber_ * 2 - 1) / (vlForHalfNumber_ * 2); // 128
    uint16_t loopScaleNum = (totalScaleInUb + vlForHalfNumber_ - 1) / vlForHalfNumber_;         // 8
    ComputeMaxExp(gluResAddr, maxExpAddr, totalDataInUb, loopDataNum);                          // 获取最大值
    ComputeScale(maxExpAddr, scaleDst, halfScaleLocalAddr, totalScaleInUb, loopScaleNum);       // 计算scale和halfScale
    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        ComputeDataForQuantTargetFp8(gluResAddr, halfScaleLocalAddr, outputDst, totalDataInUb,
                                     loopDataNum); // 计算量化后的值
    }
    if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp4x2_e1m2_t>::value) {
        ComputeDataForQuantTargetFp4(gluResAddr, halfScaleLocalAddr, outputDst, totalDataInUb, loopDataNum);
    }
    return;
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoSwigluForMX(
    uint16_t mSize, uint16_t pingpongIdx, uint64_t mLoc)
{
    constexpr uint32_t pongElemOf_DataTypeIn = MAX_SINGLE_MN;
    constexpr uint32_t pongElemOf_bf16 = MAX_SINGLE_MN * sizeof(DataTypeIn) / sizeof(bfloat16_t);
    constexpr uint32_t pongElemOf_int8 = MAX_SINGLE_MN * sizeof(DataTypeIn);
    constexpr uint32_t pongElemOf_uint16 = MAX_SINGLE_MN * sizeof(DataTypeIn) / sizeof(uint16_t);
    // 当前master调用不传pingpongIdx，默认走非interleaved/idx0；interleaved接入后只在入口切当前half UB。
    const uint32_t pongMul = (IsInterleaved_ && pingpongIdx == 1U) ? 1U : 0U;

    __ubuf__ DataTypeIn *l0cOutUbBase =
        (__ubuf__ DataTypeIn *)l0cOutUbFirst_.GetPhyAddr() + pongMul * pongElemOf_DataTypeIn;
    __ubuf__ bfloat16_t *gluResAddr = (__ubuf__ bfloat16_t *)gluRes_.GetPhyAddr() + pongMul * pongElemOf_bf16;
    __ubuf__ int8_t *quantOutputInUbAddr = (__ubuf__ int8_t *)quantOutput_.GetPhyAddr() + pongMul * pongElemOf_int8;
    __ubuf__ uint16_t *quantScaleOutputInUbAddr =
        (__ubuf__ uint16_t *)quantScaleOutput_.GetPhyAddr() + pongMul * pongElemOf_uint16;
    __ubuf__ uint16_t *maxExpAddr = (__ubuf__ uint16_t *)maxExp_.GetPhyAddr() + pongMul * pongElemOf_uint16;
    __ubuf__ uint16_t *halfScaleAddr = (__ubuf__ uint16_t *)halfScale_.GetPhyAddr() + pongMul * pongElemOf_uint16;

    __ubuf__ DataTypeIn *l0cOutUbSecondAddr;
    if constexpr (IsInterleaved_) {
        l0cOutUbSecondAddr = l0cOutUbBase + singleN_;
    } else {
        l0cOutUbSecondAddr = (__ubuf__ DataTypeIn *)l0cOutUbSecond_.GetPhyAddr();
    }
    if (actMode_ == static_cast<uint8_t>(ActMode::SITU)) {
        if (actSubMode_ == static_cast<uint8_t>(ActSubMode::LINEAR)) {
            VFDoSwigluAndQuantForMX<SwigluQuantMsg::QuantMode::MX_PERGROUP_MODE, IsInterleaved_, ActMode::SITU,
                                    ActSubMode::LINEAR>(quantOutputInUbAddr, quantScaleOutputInUbAddr, l0cOutUbBase,
                                                        l0cOutUbSecondAddr, gluResAddr, maxExpAddr, halfScaleAddr,
                                                        mSize, singleN_, mLoc);
        } else {
            VFDoSwigluAndQuantForMX<SwigluQuantMsg::QuantMode::MX_PERGROUP_MODE, IsInterleaved_, ActMode::SITU,
                                    ActSubMode::DEFAULT>(quantOutputInUbAddr, quantScaleOutputInUbAddr, l0cOutUbBase,
                                                         l0cOutUbSecondAddr, gluResAddr, maxExpAddr, halfScaleAddr,
                                                         mSize, singleN_, mLoc);
        }
    } else {
        VFDoSwigluAndQuantForMX<SwigluQuantMsg::QuantMode::MX_PERGROUP_MODE, IsInterleaved_, ActMode::SWIGLU,
                                ActSubMode::DEFAULT>(quantOutputInUbAddr, quantScaleOutputInUbAddr, l0cOutUbBase,
                                                     l0cOutUbSecondAddr, gluResAddr, maxExpAddr, halfScaleAddr, mSize,
                                                     singleN_, mLoc);
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
__aicore__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::operator()(
    const BlockShape &blockShape, const BlockCoord &blockCoord, uint16_t pingpongIdx)
{
    singleM_ = Get<M_VALUE>(blockShape); // 128
    singleN_ = Get<N_VALUE>(blockShape); // 256
    blockCoord_ = blockCoord;

    if (singleM_ == 0) {
        return;
    }

    vlForHalfNumber_ = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t); // 256 / 2 = 128
    UBBlockSize_ = BLOCK_SIZE;                                         // 32
    elementAfterReduce_ = AscendC::VECTOR_REG_WIDTH / UBBlockSize_;    // 256 / 32 = 8

    uint64_t yOffset = Get<Y_IDX>(blockCoord);
    uint64_t yScaleOffset = Get<Y_SCALE_IDX>(blockCoord);
    uint64_t mLoc = Get<M_LOC_IDX>(blockCoord);
    VFDoSwigluForMX(singleM_, pingpongIdx, mLoc); // switch(x)*y 计算quant quantScale
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    // scale已按compact布局生成，直接copy到GM，省掉原先TransMxScaleLayout重排scale。
    if constexpr (IsInterleaved_) {
        constexpr uint32_t PONG_INT8_ELEMS = MAX_SINGLE_MN * sizeof(DataTypeIn);
        if (pingpongIdx == 1U) {
            AscendC::LocalTensor<int8_t> quantOutputPong = quantOutput_[PONG_INT8_ELEMS];
            AscendC::LocalTensor<int8_t> quantScalePong = quantScaleOutput_[PONG_INT8_ELEMS];
            CopyOutputFromUb2Gm(singleM_, yOffset, quantOutputPong);
            CopyScaleFromUb2GmCompact(singleM_, yScaleOffset, quantScalePong);
        } else {
            CopyOutputFromUb2Gm(singleM_, yOffset, quantOutput_);
            CopyScaleFromUb2GmCompact(singleM_, yScaleOffset, quantScaleOutput_);
        }
    } else {
        CopyOutputFromUb2Gm(singleM_, yOffset, quantOutput_);
        CopyScaleFromUb2GmCompact(singleM_, yScaleOffset, quantScaleOutput_);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
}

BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::NotifyCompletedTile(
    uint32_t flagSlotIdx)
{
    if (groupFlagListGmAddr_ != nullptr) {
        if constexpr (IsWaveFlagGrained_) {
            AscendC::AtomicAdd(groupFlagListGmAddr_ + static_cast<uint64_t>(flagSlotIdx) * INT_CACHELINE, 1);
        } else {
            AscendC::AtomicAdd(groupFlagListGmAddr_, 1);
        }
    }
}

#endif // defined(__DAV_C310__)
#endif // BLOCK_EPILOGUE_SWIGLU_MX_QUANT_H
