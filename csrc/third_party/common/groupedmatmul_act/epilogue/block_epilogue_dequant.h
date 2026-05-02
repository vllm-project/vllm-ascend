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
 * \file block_epilogue_dequant.h
 * \brief
 */

#ifndef EPILOGUE_BLOCK_EPILOGUE_DEQUANT_H
#define EPILOGUE_BLOCK_EPILOGUE_DEQUANT_H
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

namespace Qmm {
enum class QuantMode : uint32_t {
    DEFAULT = 0x0U,
    PERTENSOR_MODE = 0x1U,
    PERCHANNEL_MODE = 0x1U << 1,
    PERTOKEN_MODE = 0x1U << 2,
    MX_PERGROUP_MODE = 0x1U << 3,
    PERBLOCK_MODE = 0x1U << 4,
};

struct DequantTiling {
    uint32_t baseM;
    uint32_t baseN;
    QuantMode x1QuantMode = QuantMode::DEFAULT;
    QuantMode x2QuantMode = QuantMode::DEFAULT;
    uint32_t biasDtype = DT_FLOAT;
    bool isBiasEpilogue = false;
};
} // namespace Qmm

namespace {
constexpr uint32_t FP32_OUTPUT_TIMES = 4;
constexpr uint32_t Y_IDX = 0;
constexpr uint32_t X2SCALE_IDX = 1;
constexpr uint32_t X1SCALE_IDX = 2;
constexpr uint32_t BIAS_IDX = 3;
constexpr uint32_t M_IDX = 0;
constexpr uint32_t N_IDX = 1;
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

#define QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS                                                                  \
    template <typename L0TileShape_, typename DataTypeOut_, typename DataTypeIn_, typename DataTypeX2Scale_,           \
              typename DataTypeX1Scale_, typename DataTypeBias_, bool IsTensorList_>
#define QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS                                                                   \
    L0TileShape_, DataTypeOut_, DataTypeIn_, DataTypeX2Scale_, DataTypeX1Scale_, DataTypeBias_, IsTensorList_

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
class BlockEpilogueDequant {
public:
    __aicore__ inline BlockEpilogueDequant() {}

    struct Arguments {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        const Qmm::DequantTiling dequantTiling;
    };

    struct Params {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        const Qmm::DequantTiling dequantTiling;
    };

    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    using DataTypeX1Scale = DataTypeX1Scale_;
    using DataTypeX2Scale = DataTypeX2Scale_;
    using DataTypeBias = DataTypeBias_;

    static constexpr int64_t l0M = GetIntegralConstant<MNK_M, L0TileShape_>();
    static constexpr int64_t l0N = GetIntegralConstant<MNK_N, L0TileShape_>();
    // shape
    using BlockShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using BaseOffset = Coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = Coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = Shape<int64_t, int64_t, int64_t, int64_t>;

public:
    __aicore__ inline void Init(Params const& params, ProblemShape& problemShape);
    __aicore__ inline void UpdateGlobalBuffer(Params const& params);
    __aicore__ inline void UpdateGroupedParams(Params const& params, BaseOffset const& offset, uint32_t groupIdx);
    __aicore__ inline void UpdateBatchParams(Params const& params, BaseOffset const& offset, uint32_t bacthIdx);
    __aicore__ inline void Run(BlockShape& blockShape, BlockCoord& blockCoord);
    __aicore__ inline auto GetL0c2UbTensor();
    __aicore__ inline void SetOutL2CacheHint();
    __aicore__ inline void operator()(BlockShape& blockShape, BlockCoord& blockCoord);
    // static init
    __host_aicore__ static Params InitParams(Arguments const& args)
    {
        Params params = {args.yGmAddr, args.x2ScaleGmAddr, args.x1ScaleGmAddr, args.biasGmAddr, args.dequantTiling};
        return params;
    }

    __host_aicore__ static size_t GetWorkspaceSize(int64_t blockNum, int64_t l1M, int64_t l1N)
    {
        return 0;
    }

    __host_aicore__ static Status CanImplement(Arguments const& args)
    {
        if (l0M * l0N * sizeof(DataTypeIn_) > TOTAL_UB_SIZE) {
            return Status::l1L0ErrorExceedsLimit;
        }
        return Status::success;
    }

private:
    __aicore__ inline void UpdateTensorListGlobalBuffer(Params const& params);
    __aicore__ inline void UpdateTensorGlobalBuffer(Params const& params);
    __aicore__ inline void CopyDataFromGm2Ub();
    __aicore__ inline void CopyX1ScaleFromGm2Ub(AscendC::LocalTensor<DataTypeX1Scale>& dst, uint64_t blockLen,
                                                uint64_t offset);
    __aicore__ inline void CopyX2ScaleFromGm2Ub(AscendC::LocalTensor<DataTypeX2Scale>& dst);
    template <class BiasDtype>
    __aicore__ inline void CopyBiasFromGm2Ub(AscendC::LocalTensor<BiasDtype>& dst);
    __aicore__ inline void CopyDequantResFromUb2Gm(uint64_t blockCount, uint64_t offset,
                                                   AscendC::LocalTensor<DataTypeOut>& src);
    __aicore__ inline void FreeUbTensor();
    __aicore__ inline void VFDoDequantWithX1Pertoken(__ubuf__ DataTypeOut* dequantOutInUbAddr,
                                                     __ubuf__ DataTypeIn* l0cOutUbAddr, uint64_t offsetPtScale,
                                                     uint16_t mSize);
    __aicore__ inline void VFDoDequantWithX1Pertensor(__ubuf__ DataTypeOut* dequantOutInUbAddr,
                                                      __ubuf__ DataTypeIn* l0cOutUbAddr, uint16_t mSize);
    __aicore__ inline void VFDoDequantWithoutX1Scale(__ubuf__ DataTypeOut* dequantOutInUbAddr,
                                                     __ubuf__ DataTypeIn* l0cOutUbAddr, uint16_t mSize);
    template <bool isPertensor, Qmm::QuantMode x1QuantMode, bool isBiasEpilogue, class BiasDtype>
    __aicore__ inline void VFDoDequant(__ubuf__ DataTypeOut* dst, __ubuf__ DataTypeIn* l0cOut,
                                       __ubuf__ DataTypeX2Scale* scale2, __ubuf__ DataTypeX1Scale* x1Scale,
                                       __ubuf__ BiasDtype* bias, uint16_t mSize, uint16_t nSize);

    // GM ADDR
    AscendC::GlobalTensor<DataTypeOut> yGlobal_;
    AscendC::GlobalTensor<float> biasGlobalFloat_;
    AscendC::GlobalTensor<bfloat16_t> biasGlobalB16_;
    AscendC::GlobalTensor<DataTypeX2Scale> x2ScaleGlobal_;
    AscendC::GlobalTensor<DataTypeX1Scale> x1ScaleGlobal_;
    // UB Tensor
    AscendC::LocalTensor<DataTypeIn> l0cOutUb_;
    AscendC::LocalTensor<DataTypeX2Scale> x2ScaleUb_;
    AscendC::LocalTensor<DataTypeX1Scale> x1ScaleUb_;
    AscendC::LocalTensor<bfloat16_t> biasUbB16_;
    AscendC::LocalTensor<float> biasUbFloat_;
    float x2ScaleScalar_;
    float x1ScaleScalar_;
    // define the que
    TQue<QuePosition::VECIN, 1> vecQueMMRes_;
    TQue<QuePosition::VECIN, 1> vecQueX2Scale_;
    TQue<QuePosition::VECIN, 1> vecQueX1Scale_;
    TQue<QuePosition::VECIN, 1> vecQueBias_;
    TQue<QuePosition::VECOUT, 1> vecQueOut_;
    const Qmm::DequantTiling* dequantTiling_;
    ProblemShape problemShape_;
    uint32_t biasDtype_ = DT_FLOAT;
    uint32_t groupIdx_ = 0;
    uint32_t subBlockIdx_ = AscendC::GetSubBlockIdx();
    uint32_t singleM_; // cur singleShapeM
    uint32_t singleN_;
    bool isBiasEpilogue_ = false;
    BaseOffset baseOffset_{0, 0, 0, 0}; // order in Params and for groupedMM or BatchMM
    BlockCoord blockCoord_{0, 0, 0, 0}; // order in Params
};

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueDequant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateTensorListGlobalBuffer(Params const& params)
{
    if (dequantTiling_->x2QuantMode == Qmm::QuantMode::PERTENSOR_MODE) {
        DataTypeX2Scale x2ScaleValue =
            *((__gm__ DataTypeX2Scale*)(GetTensorAddr<DataTypeX2Scale>(0, params.x2ScaleGmAddr) + groupIdx_));
        if constexpr (AscendC::IsSameType<DataTypeX2Scale, bfloat16_t>::value) {
            x2ScaleScalar_ = AscendC::ToFloat(x2ScaleValue);
        } else {
            x2ScaleScalar_ = x2ScaleValue;
        }
    } else {
        x2ScaleGlobal_.SetGlobalBuffer(GetTensorAddr<DataTypeX2Scale>(0, params.x2ScaleGmAddr) +
                                       Get<X2SCALE_IDX>(baseOffset_));
    }
    if (dequantTiling_->x1QuantMode == Qmm::QuantMode::PERTENSOR_MODE) {
        x1ScaleScalar_ = *((__gm__ DataTypeX1Scale*)params.x1ScaleGmAddr + groupIdx_);
    } else if (dequantTiling_->x1QuantMode == Qmm::QuantMode::PERTOKEN_MODE) {
        x1ScaleGlobal_.SetGlobalBuffer((__gm__ DataTypeX1Scale*)params.x1ScaleGmAddr + Get<X1SCALE_IDX>(baseOffset_));
    }
    if (isBiasEpilogue_) {
        if (biasDtype_ == DT_FLOAT) {
            biasGlobalFloat_.SetGlobalBuffer(GetTensorAddr<float>(0, params.biasGmAddr) + Get<BIAS_IDX>(baseOffset_));
        } else {
            biasGlobalB16_.SetGlobalBuffer(GetTensorAddr<bfloat16_t>(0, params.biasGmAddr) +
                                           Get<BIAS_IDX>(baseOffset_));
        }
    }
    yGlobal_.SetGlobalBuffer(GetTensorAddr<DataTypeOut>(0, params.yGmAddr) + Get<Y_IDX>(baseOffset_));
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueDequant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateTensorGlobalBuffer(Params const& params)
{
    if (dequantTiling_->x2QuantMode == Qmm::QuantMode::PERTENSOR_MODE) {
        DataTypeX2Scale x2ScaleValue = *((__gm__ DataTypeX2Scale*)params.x2ScaleGmAddr + groupIdx_);
        if constexpr (AscendC::IsSameType<DataTypeX2Scale, bfloat16_t>::value) {
            x2ScaleScalar_ = AscendC::ToFloat(x2ScaleValue);
        } else {
            x2ScaleScalar_ = x2ScaleValue;
        }
    } else {
        x2ScaleGlobal_.SetGlobalBuffer((__gm__ DataTypeX2Scale*)params.x2ScaleGmAddr + Get<X2SCALE_IDX>(baseOffset_));
    }
    if (dequantTiling_->x1QuantMode == Qmm::QuantMode::PERTENSOR_MODE) {
        x1ScaleScalar_ = *((__gm__ DataTypeX1Scale*)params.x1ScaleGmAddr + groupIdx_);
    } else if (dequantTiling_->x1QuantMode == Qmm::QuantMode::PERTOKEN_MODE) {
        x1ScaleGlobal_.SetGlobalBuffer((__gm__ DataTypeX1Scale*)params.x1ScaleGmAddr + Get<X1SCALE_IDX>(baseOffset_));
    }
    // ub res + biasAdd
    if (isBiasEpilogue_) {
        if (biasDtype_ == DT_FLOAT) {
            biasGlobalFloat_.SetGlobalBuffer((__gm__ float*)params.biasGmAddr + Get<BIAS_IDX>(baseOffset_));
        } else {
            biasGlobalB16_.SetGlobalBuffer((__gm__ bfloat16_t*)params.biasGmAddr + Get<BIAS_IDX>(baseOffset_));
        }
    }
    yGlobal_.SetGlobalBuffer((__gm__ DataTypeOut*)params.yGmAddr + Get<Y_IDX>(baseOffset_));
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueDequant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateGlobalBuffer(Params const& params)
{
    if constexpr (IsTensorList_) {
        UpdateTensorListGlobalBuffer(params);
    } else {
        UpdateTensorGlobalBuffer(params);
    }
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS __aicore__ inline void
BlockEpilogueDequant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::Init(Params const& params,
                                                                         ProblemShape& problemShape)
{
    dequantTiling_ = &params.dequantTiling;
    uint64_t mForSingleVec = CeilDiv(dequantTiling_->baseM, AscendC::GetTaskRation());
    GetTPipePtr()->InitBuffer(vecQueMMRes_, 1, mForSingleVec * dequantTiling_->baseN * sizeof(DataTypeIn));
    l0cOutUb_ = vecQueMMRes_.AllocTensor<DataTypeIn>();
    if ASCEND_IS_AIV {
        UpdateGlobalBuffer(params);
        isBiasEpilogue_ = dequantTiling_->isBiasEpilogue && params.biasGmAddr != nullptr;
        biasDtype_ = dequantTiling_->biasDtype;
        if (dequantTiling_->x2QuantMode == Qmm::QuantMode::PERCHANNEL_MODE) {
            GetTPipePtr()->InitBuffer(vecQueX2Scale_, 1, dequantTiling_->baseN * sizeof(DataTypeX2Scale));
        }
        if (dequantTiling_->x1QuantMode == Qmm::QuantMode::PERTOKEN_MODE) {
            GetTPipePtr()->InitBuffer(
                vecQueX1Scale_, 1,
                Align(mForSingleVec * sizeof(DataTypeX1Scale), static_cast<uint64_t>(UB_ALIGN_SIZE)));
        }
        if (isBiasEpilogue_) {
            if (biasDtype_ == DT_FLOAT) {
                GetTPipePtr()->InitBuffer(vecQueBias_, 1, dequantTiling_->baseN * sizeof(float));
            } else {
                GetTPipePtr()->InitBuffer(vecQueBias_, 1, dequantTiling_->baseN * sizeof(bfloat16_t));
            }
        }
        GetTPipePtr()->InitBuffer(vecQueOut_, DOUBLE_BUFFER_COUNT,
                                  CeilDiv(mForSingleVec, FP32_OUTPUT_TIMES) * dequantTiling_->baseN *
                                      sizeof(DataTypeOut));
    }
    problemShape_ = problemShape;
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueDequant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateGroupedParams(
    Params const& params, BaseOffset const& offset, uint32_t groupIdx)
{
    baseOffset_ = offset;
    groupIdx_ = groupIdx;
    UpdateGlobalBuffer(params);
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueDequant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateBatchParams(
    Params const& params, BaseOffset const& offset, uint32_t bacthIdx)
{
    UpdateGroupedParams(params, offset, bacthIdx);
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueDequant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyDataFromGm2Ub()
{
    auto halfSingleM = CeilDiv(singleM_, AscendC::GetTaskRation());
    auto singleMInVec = subBlockIdx_ == 1 ? singleM_ - halfSingleM : halfSingleM;
    // scale2: GM -> UB
    if (dequantTiling_->x2QuantMode == Qmm::QuantMode::PERCHANNEL_MODE) {
        x2ScaleUb_ = vecQueX2Scale_.AllocTensor<DataTypeX2Scale>();
        CopyX2ScaleFromGm2Ub(x2ScaleUb_);
        vecQueX2Scale_.EnQue<DataTypeX2Scale>(x2ScaleUb_);
        x2ScaleUb_ = vecQueX2Scale_.DeQue<DataTypeX2Scale>();
    }

    uint64_t mOffset = subBlockIdx_ * halfSingleM;
    // x1Scale: GM -> UB
    if (dequantTiling_->x1QuantMode == Qmm::QuantMode::PERTOKEN_MODE) {
        x1ScaleUb_ = vecQueX1Scale_.AllocTensor<DataTypeX1Scale>();
        CopyX1ScaleFromGm2Ub(x1ScaleUb_, singleMInVec * sizeof(DataTypeX1Scale),
                             Get<X1SCALE_IDX>(blockCoord_) + mOffset);
        vecQueX1Scale_.EnQue<DataTypeX1Scale>(x1ScaleUb_);
        x1ScaleUb_ = vecQueX1Scale_.DeQue<DataTypeX1Scale>();
    }
    if (isBiasEpilogue_) {
        if (biasDtype_ == DT_FLOAT) {
            biasUbFloat_ = vecQueBias_.AllocTensor<float>();
            CopyBiasFromGm2Ub<float>(biasUbFloat_);
            vecQueBias_.EnQue<float>(biasUbFloat_);
            biasUbFloat_ = vecQueBias_.DeQue<float>();
        } else {
            biasUbB16_ = vecQueBias_.AllocTensor<bfloat16_t>();
            CopyBiasFromGm2Ub<bfloat16_t>(biasUbB16_);
            vecQueBias_.EnQue<bfloat16_t>(biasUbB16_);
            biasUbB16_ = vecQueBias_.DeQue<bfloat16_t>();
        }
    }
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueDequant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyX1ScaleFromGm2Ub(
    AscendC::LocalTensor<DataTypeX1Scale>& dst, uint64_t blockLen, uint64_t offset)
{
    DataCopyParams ptScale2UbParams{1, 0, 0, 0};
    DataCopyPadParams padParams;
    ptScale2UbParams.blockLen = blockLen;
    AscendC::DataCopyPad(dst, x1ScaleGlobal_[offset], ptScale2UbParams, padParams);
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueDequant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyX2ScaleFromGm2Ub(
    AscendC::LocalTensor<DataTypeX2Scale>& dst)
{
    DataCopyParams scale2UbParams{1, 0, 0, 0};
    DataCopyPadParams padParams;
    scale2UbParams.blockLen = singleN_ * sizeof(DataTypeX2Scale);
    AscendC::DataCopyPad(dst, x2ScaleGlobal_[Get<X2SCALE_IDX>(blockCoord_)], scale2UbParams, padParams);
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
template <class BiasDtype>
__aicore__ inline void BlockEpilogueDequant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyBiasFromGm2Ub(
    AscendC::LocalTensor<BiasDtype>& dst)
{
    DataCopyParams bias2UbParams{1, 0, 0, 0};
    DataCopyPadParams padParams;
    if constexpr (IsSameType<BiasDtype, float>::value) {
        bias2UbParams.blockLen = singleN_ * sizeof(float);
        AscendC::DataCopyPad(dst, biasGlobalFloat_[Get<BIAS_IDX>(blockCoord_)], bias2UbParams, padParams);
    } else {
        bias2UbParams.blockLen = singleN_ * sizeof(bfloat16_t);
        AscendC::DataCopyPad(dst, biasGlobalB16_[Get<BIAS_IDX>(blockCoord_)], bias2UbParams, padParams);
    }
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueDequant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyDequantResFromUb2Gm(
    uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<DataTypeOut>& src)
{
    DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockLen = singleN_ * sizeof(DataTypeOut);
    ub2GmParams.blockCount = blockCount;
    ub2GmParams.dstStride = (Get<1>(problemShape_) - singleN_) * sizeof(DataTypeOut);
    AscendC::DataCopyPad(yGlobal_[Get<Y_IDX>(blockCoord_) + offset], src, ub2GmParams);
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueDequant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::FreeUbTensor()
{
    if (dequantTiling_->x2QuantMode == Qmm::QuantMode::PERCHANNEL_MODE) {
        vecQueX2Scale_.FreeTensor(x2ScaleUb_);
    }

    if (dequantTiling_->x1QuantMode == Qmm::QuantMode::PERTOKEN_MODE) {
        vecQueX1Scale_.FreeTensor(x1ScaleUb_);
    }

    if (isBiasEpilogue_) {
        if (biasDtype_ == DT_FLOAT) {
            vecQueBias_.FreeTensor(biasUbFloat_);
        } else {
            vecQueBias_.FreeTensor(biasUbB16_);
        }
    }
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueDequant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoDequantWithX1Pertoken(
    __ubuf__ DataTypeOut* dequantOutInUbAddr, __ubuf__ DataTypeIn* l0cOutUbAddr, uint64_t offsetPtScale, uint16_t mSize)
{
    __ubuf__ DataTypeX1Scale* ptScaleUbAddr = (__ubuf__ DataTypeX1Scale*)x1ScaleUb_.GetPhyAddr();
    ptScaleUbAddr = ptScaleUbAddr + offsetPtScale;
    if (!isBiasEpilogue_) {
        if (dequantTiling_->x2QuantMode == Qmm::QuantMode::PERTENSOR_MODE) {
            VFDoDequant<true, Qmm::QuantMode::PERTOKEN_MODE, false, float>(dequantOutInUbAddr, l0cOutUbAddr, nullptr,
                                                                           ptScaleUbAddr, nullptr, mSize, singleN_);
        } else {
            VFDoDequant<false, Qmm::QuantMode::PERTOKEN_MODE, false, float>(
                dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ DataTypeX2Scale*)x2ScaleUb_.GetPhyAddr(), ptScaleUbAddr,
                nullptr, mSize, singleN_);
        }
    } else {
        if (biasDtype_ == DT_FLOAT) {
            if (dequantTiling_->x2QuantMode == Qmm::QuantMode::PERTENSOR_MODE) {
                VFDoDequant<true, Qmm::QuantMode::PERTOKEN_MODE, true, float>(
                    dequantOutInUbAddr, l0cOutUbAddr, nullptr, ptScaleUbAddr,
                    (__ubuf__ float*)biasUbFloat_.GetPhyAddr(), mSize, singleN_);
            } else {
                VFDoDequant<false, Qmm::QuantMode::PERTOKEN_MODE, true, float>(
                    dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ DataTypeX2Scale*)x2ScaleUb_.GetPhyAddr(), ptScaleUbAddr,
                    (__ubuf__ float*)biasUbFloat_.GetPhyAddr(), mSize, singleN_);
            }
        } else if (biasDtype_ == DT_BF16) {
            if (dequantTiling_->x2QuantMode == Qmm::QuantMode::PERTENSOR_MODE) {
                VFDoDequant<true, Qmm::QuantMode::PERTOKEN_MODE, true, bfloat16_t>(
                    dequantOutInUbAddr, l0cOutUbAddr, nullptr, ptScaleUbAddr,
                    (__ubuf__ bfloat16_t*)biasUbB16_.GetPhyAddr(), mSize, singleN_);
            } else {
                VFDoDequant<false, Qmm::QuantMode::PERTOKEN_MODE, true, bfloat16_t>(
                    dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ DataTypeX2Scale*)x2ScaleUb_.GetPhyAddr(), ptScaleUbAddr,
                    (__ubuf__ bfloat16_t*)biasUbB16_.GetPhyAddr(), mSize, singleN_);
            }
        } else if (biasDtype_ == DT_FLOAT16) {
            if (dequantTiling_->x2QuantMode == Qmm::QuantMode::PERTENSOR_MODE) {
                VFDoDequant<true, Qmm::QuantMode::PERTOKEN_MODE, true, half>(
                    dequantOutInUbAddr, l0cOutUbAddr, nullptr, ptScaleUbAddr, (__ubuf__ half*)biasUbB16_.GetPhyAddr(),
                    mSize, singleN_);
            } else {
                VFDoDequant<false, Qmm::QuantMode::PERTOKEN_MODE, true, half>(
                    dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ DataTypeX2Scale*)x2ScaleUb_.GetPhyAddr(), ptScaleUbAddr,
                    (__ubuf__ half*)biasUbB16_.GetPhyAddr(), mSize, singleN_);
            }
        }
    }
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueDequant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoDequantWithX1Pertensor(
    __ubuf__ DataTypeOut* dequantOutInUbAddr, __ubuf__ DataTypeIn* l0cOutUbAddr, uint16_t mSize)
{
    VFDoDequant<false, Qmm::QuantMode::PERTENSOR_MODE, false, float>(dequantOutInUbAddr, l0cOutUbAddr,
                                                                     (__ubuf__ DataTypeX2Scale*)x2ScaleUb_.GetPhyAddr(),
                                                                     nullptr, nullptr, mSize, singleN_);
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueDequant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoDequantWithoutX1Scale(
    __ubuf__ DataTypeOut* dequantOutInUbAddr, __ubuf__ DataTypeIn* l0cOutUbAddr, uint16_t mSize)
{
    if (!isBiasEpilogue_) {
        VFDoDequant<false, Qmm::QuantMode::DEFAULT, false, float>(dequantOutInUbAddr, l0cOutUbAddr,
                                                                  (__ubuf__ DataTypeX2Scale*)x2ScaleUb_.GetPhyAddr(),
                                                                  nullptr, nullptr, mSize, singleN_);
    } else {
        if (biasDtype_ == DT_FLOAT) {
            if (dequantTiling_->x2QuantMode == Qmm::QuantMode::PERTENSOR_MODE) {
                VFDoDequant<true, Qmm::QuantMode::DEFAULT, true, float>(
                    dequantOutInUbAddr, l0cOutUbAddr, nullptr, nullptr, (__ubuf__ float*)biasUbFloat_.GetPhyAddr(),
                    mSize, singleN_);
            } else {
                VFDoDequant<false, Qmm::QuantMode::DEFAULT, true, float>(
                    dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ DataTypeX2Scale*)x2ScaleUb_.GetPhyAddr(), nullptr,
                    (__ubuf__ float*)biasUbFloat_.GetPhyAddr(), mSize, singleN_);
            }
        } else if (biasDtype_ == DT_BF16) {
            if (dequantTiling_->x2QuantMode == Qmm::QuantMode::PERTENSOR_MODE) {
                VFDoDequant<true, Qmm::QuantMode::DEFAULT, true, bfloat16_t>(
                    dequantOutInUbAddr, l0cOutUbAddr, nullptr, nullptr, (__ubuf__ bfloat16_t*)biasUbB16_.GetPhyAddr(),
                    mSize, singleN_);
            } else {
                VFDoDequant<false, Qmm::QuantMode::DEFAULT, true, bfloat16_t>(
                    dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ DataTypeX2Scale*)x2ScaleUb_.GetPhyAddr(), nullptr,
                    (__ubuf__ bfloat16_t*)biasUbB16_.GetPhyAddr(), mSize, singleN_);
            }
        }
    }
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
template <bool isPertensor, Qmm::QuantMode x1QuantMode, bool isBiasEpilogue, class BiasDtype>
__aicore__ inline void BlockEpilogueDequant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoDequant(
    __ubuf__ DataTypeOut* dst, __ubuf__ DataTypeIn* l0cOut, __ubuf__ DataTypeX2Scale* scale2,
    __ubuf__ DataTypeX1Scale* x1Scale, __ubuf__ BiasDtype* bias, uint16_t mSize, uint16_t nSize)
{
    uint32_t eleNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(DataTypeIn);
    uint32_t nSrcUbAligned = Align(nSize, static_cast<uint16_t>(UB_ALIGN_SIZE / sizeof(DataTypeIn)));
    uint32_t nDstUbAligned = Align(nSize, static_cast<uint16_t>(UB_ALIGN_SIZE / sizeof(DataTypeOut)));
    uint16_t nLoopCnt = (nSize + eleNumPerVf - 1) / eleNumPerVf;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg maskN4B16 =
            AscendC::MicroAPI::CreateMask<bfloat16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
            uint32_t elementNum = nSize;
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
                AscendC::MicroAPI::RegTensor<DataTypeIn> l0cOutReg;
                AscendC::MicroAPI::RegTensor<DataTypeX2Scale> scaleReg;
                AscendC::MicroAPI::RegTensor<DataTypeX1Scale> perTokenScaleReg;
                AscendC::MicroAPI::RegTensor<BiasDtype> biasReg;
                AscendC::MicroAPI::RegTensor<float> castSrcOutReg, castScaleReg, castScaleOneReg, mulScaleOutReg,
                    mulPtScaleOutReg, castBiasReg, castBiasOneReg, addBiasOutReg;
                AscendC::MicroAPI::RegTensor<DataTypeOut> castResultOutReg;
                AscendC::MicroAPI::MaskReg maskN = AscendC::MicroAPI::UpdateMask<DataTypeIn>(elementNum);
                // copy input from ub to register, addr of ub should align to 32B
                uint32_t l0cOutOffset = mIdx * nSrcUbAligned + vfBlockIdx * eleNumPerVf;
                AscendC::MicroAPI::DataCopy(l0cOutReg, l0cOut + l0cOutOffset);
                // cast l0cOut from int32 to float
                if constexpr (IsSameType<DataTypeIn, int32_t>::value) {
                    AscendC::MicroAPI::Cast<float, DataTypeIn, ctInt322Fp32>(castSrcOutReg, l0cOutReg, maskN);
                } else {
                    castSrcOutReg = l0cOutReg;
                }
                // l0c_out * scale2
                if constexpr (isPertensor) {
                    AscendC::MicroAPI::Muls(mulScaleOutReg, castSrcOutReg, x2ScaleScalar_, maskN);
                } else {
                    AscendC::MicroAPI::DataCopy(scaleReg, scale2 + vfBlockIdx * eleNumPerVf);
                    if constexpr (!IsSameType<DataTypeX2Scale, float>::value) { // cast scale2 from bf16 to float
                        AscendC::MicroAPI::Cast<float, DataTypeX2Scale, ctHalf2Fp32Zero>(castScaleReg, scaleReg, maskN);
                        AscendC::MicroAPI::Cast<float, DataTypeX2Scale, ctHalf2Fp32One>(castScaleOneReg, scaleReg,
                                                                                        maskN4B16);
                        AscendC::MicroAPI::Interleave(castScaleReg, castScaleOneReg, castScaleReg, castScaleOneReg);
                    } else {
                        castScaleReg = scaleReg;
                    }
                    AscendC::MicroAPI::Mul(mulScaleOutReg, castSrcOutReg, castScaleReg, maskN);
                }
                // out * x1Scale
                if constexpr (x1QuantMode == Qmm::QuantMode::PERTENSOR_MODE) {
                    AscendC::MicroAPI::Muls(mulPtScaleOutReg, mulScaleOutReg, x1ScaleScalar_, maskN);
                } else if constexpr (x1QuantMode == Qmm::QuantMode::PERTOKEN_MODE) {
                    AscendC::MicroAPI::DataCopy<DataTypeX1Scale, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        perTokenScaleReg, x1Scale + mIdx);
                    AscendC::MicroAPI::Mul(mulPtScaleOutReg, mulScaleOutReg, perTokenScaleReg, maskN);
                } else {
                    mulPtScaleOutReg = mulScaleOutReg;
                }
                // out + bias
                if constexpr (isBiasEpilogue) {
                    AscendC::MicroAPI::DataCopy(biasReg, bias + vfBlockIdx * eleNumPerVf);
                    // cast bias from bf16/fp16 to float
                    if constexpr (IsSameType<BiasDtype, bfloat16_t>::value || IsSameType<BiasDtype, half>::value) {
                        AscendC::MicroAPI::Cast<float, BiasDtype, ctHalf2Fp32Zero>(castBiasReg, biasReg, maskN);
                        AscendC::MicroAPI::Cast<float, BiasDtype, ctHalf2Fp32One>(castBiasOneReg, biasReg, maskN4B16);
                        AscendC::MicroAPI::Interleave(castBiasReg, castBiasOneReg, castBiasReg, castBiasOneReg);
                    } else {
                        castBiasReg = biasReg;
                    }
                    AscendC::MicroAPI::Add(addBiasOutReg, mulPtScaleOutReg, castBiasReg, maskN);
                } else {
                    addBiasOutReg = mulPtScaleOutReg;
                }
                // cast dequant result from float to fp16/bf16
                if constexpr (!IsSameType<DataTypeOut, float>::value) {
                    AscendC::MicroAPI::Cast<DataTypeOut, float, ctFp322Half>(castResultOutReg, addBiasOutReg, maskN);
                } else {
                    castResultOutReg = addBiasOutReg;
                }
                // copy out from register to ub
                uint32_t dstUbOffset = mIdx * nDstUbAligned + vfBlockIdx * eleNumPerVf;
                if constexpr (IsSameType<DataTypeOut, float>::value) {
                    AscendC::MicroAPI::DataCopy<DataTypeOut, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(
                        dst + dstUbOffset, castResultOutReg, maskN);
                } else {
                    AscendC::MicroAPI::DataCopy<DataTypeOut, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                        dst + dstUbOffset, castResultOutReg, maskN);
                }
            }
        }
    }
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueDequant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::Run(BlockShape& blockShape,
                                                                                               BlockCoord& blockCoord)
{
    singleM_ = Get<M_IDX>(blockShape);
    singleN_ = Get<N_IDX>(blockShape);
    blockCoord_ = blockCoord;
    auto halfSingleM = CeilDiv(singleM_, AscendC::GetTaskRation());
    uint64_t singleMInVec = subBlockIdx_ == 1 ? singleM_ - halfSingleM : halfSingleM;
    if (singleMInVec == 0) {
        return;
    }
    uint64_t mOffset = subBlockIdx_ * halfSingleM;
    CopyDataFromGm2Ub();
    // 4 times out because of ub size
    uint16_t splitNumOfOut = singleMInVec >= 4 ? 4 : singleMInVec;
    auto mSizeForOnce = CeilDiv(singleMInVec, static_cast<uint64_t>(splitNumOfOut));
    for (uint16_t i = 0; i < splitNumOfOut; i++) {
        // do dequant in vector
        uint64_t offsetL0c =
            i * mSizeForOnce * Align(singleN_, static_cast<uint64_t>(UB_ALIGN_SIZE / sizeof(DataTypeIn)));
        if (i * mSizeForOnce >= singleMInVec) {
            break;
        }
        auto mSize = singleMInVec - i * mSizeForOnce >= mSizeForOnce ? mSizeForOnce : singleMInVec - i * mSizeForOnce;
        AscendC::LocalTensor<DataTypeOut> dequantOutInUB = vecQueOut_.AllocTensor<DataTypeOut>();

        __ubuf__ DataTypeOut* dequantOutInUbAddr = (__ubuf__ DataTypeOut*)dequantOutInUB.GetPhyAddr();
        __ubuf__ DataTypeIn* l0cOutUbAddr = (__ubuf__ DataTypeIn*)l0cOutUb_.GetPhyAddr();
        l0cOutUbAddr = l0cOutUbAddr + offsetL0c;

        switch (dequantTiling_->x1QuantMode) {
            case (Qmm::QuantMode::PERTOKEN_MODE): {
                uint64_t offsetPtScale = i * mSizeForOnce;
                VFDoDequantWithX1Pertoken(dequantOutInUbAddr, l0cOutUbAddr, offsetPtScale, mSize);
                break;
            }
            case (Qmm::QuantMode::PERTENSOR_MODE): {
                VFDoDequantWithX1Pertensor(dequantOutInUbAddr, l0cOutUbAddr, mSize);
                break;
            }
            default: {
                VFDoDequantWithoutX1Scale(dequantOutInUbAddr, l0cOutUbAddr, mSize);
            }
        }
        vecQueOut_.EnQue<DataTypeOut>(dequantOutInUB);
        // mmDequant result: UB -> GM
        dequantOutInUB = vecQueOut_.DeQue<DataTypeOut>();
        CopyDequantResFromUb2Gm(mSize, (mOffset + i * mSizeForOnce) * Get<N_IDX>(problemShape_), dequantOutInUB);
        vecQueOut_.FreeTensor(dequantOutInUB);
    }
    FreeUbTensor();
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
__aicore__ inline auto BlockEpilogueDequant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::GetL0c2UbTensor()
{
    return l0cOutUb_;
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueDequant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::SetOutL2CacheHint()
{
    yGlobal_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
}

QMM_BLOCK_EPILOGUE_DEQUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueDequant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::operator()(BlockShape& blockShape,
                                                                               BlockCoord& blockCoord)
{
    Run(blockShape, blockCoord);
    return;
}
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif // EPILOGUE_BLOCK_EPILOGUE_DEQUANT_H
#endif