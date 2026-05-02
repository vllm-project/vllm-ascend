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
 * \file block_prologue_b_cast_scsc.h
 * \brief
 */
#ifndef ACT_INCLUDE_PROLOGUE_BLOCK_PROLOGUE_B_CAST_SCSC_H
#define ACT_INCLUDE_PROLOGUE_BLOCK_PROLOGUE_B_CAST_SCSC_H

#include "../utils/common_utils.h"
#include "../utils/integral_constant.h"
#include "../utils/tuple_utils.h"
#include "block_prologue.h"
#include "dispatch_policy.h"

namespace Act::Prologue {

using Act::Gemm::_;
using Act::Gemm::_16;
using Act::Gemm::_32;
using Act::Gemm::Get;
using Act::Gemm::GetTile;
using Act::Gemm::Min;
using AscendC::BLOCK_CUBE;
using AscendC::CeilAlign;
using AscendC::CeilDiv;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::GetSubBlockIdx;
using AscendC::HardEvent;
using AscendC::IsSameType;
using AscendC::MakeCoord;
using AscendC::MakeShape;
using AscendC::ONE_BLK_SIZE;
using AscendC::ONE_BLOCK_SIZE;
using AscendC::SetFlag;
using AscendC::SupportType;
using AscendC::VECTOR_REG_WIDTH;
using AscendC::WaitFlag;
namespace MicroAPI = AscendC::MicroAPI;

template <class InType, class OutType, class TileShapeL1>
class BlockPrologue<BCastScsc, InType, OutType, TileShapeL1> {
public:
    using ElementIn = typename InType::Element;
    using LayoutIn = typename InType::Layout;
    using ElementOut = typename OutType::Element;
    using LayoutOut = typename OutType::Layout;

    struct Arguments {
        GM_ADDR ptrB;
        LayoutIn layoutB;
    };

    struct Params {
        GM_ADDR ptrB;
        TileShapeL1 tileShapeL1;
        LayoutIn layoutB;
        int64_t l1BufNum;
        int32_t nUbSize;
        int32_t kUbSize;
    };

    template <class TensorB, class ActualBlockShape>
    __aicore__ inline void operator()(const TensorB &bGlobal, const ActualBlockShape &actualBlockShape,
                                      const Params &params)
    {
        nL1Len_ = Get<1>(actualBlockShape);
        uint64_t kTileCount = CeilDiv(kSize_, Get<2>(params.tileShapeL1));
        for (uint64_t kLoopIdx = 0; kLoopIdx < kTileCount; kLoopIdx++) {
            kGmOffset_ = kLoopIdx * kL1Size_;
            kL1Len_ = Min(kSize_ - kGmOffset_, kL1Size_);
            TensorB tensorBlockB;
            if constexpr (weightNz) {
                auto tileShape = MakeShape(MakeShape(_16{}, static_cast<uint64_t>(CeilDiv(nL1Len_, 16UL))),
                                           MakeShape(_32{}, static_cast<uint64_t>(CeilDiv(kL1Len_, 32UL))));
                tensorBlockB =
                    GetTile(bGlobal, MakeCoord(MakeCoord(_, _), MakeCoord(_, CeilDiv(kGmOffset_, 32U))), tileShape);
            } else {
                tensorBlockB = GetTile(bGlobal, MakeCoord(0, kGmOffset_),
                                       MakeShape(static_cast<uint64_t>(nL1Len_), static_cast<uint64_t>(kL1Len_)));
            }
            nUbLen_ = nL1Len_;
            kUbLen_ = kL1Len_;
            if constexpr (weightNz) {  // weightNz 只有4buffer 1buffer，1buffer直接使用AIV0
                if (likely(l1BufNum_ == QUADRUPLE_BUFFER)) {
                    ComputeUbParamsByL1Size();
                    VectorProcess(tensorBlockB);
                } else if (GetSubBlockIdx() == 0) {
                    VectorProcess(tensorBlockB);
                }
            } else {  // ND格式有4 buffer 2 buffer 1 buffer场景
                if (likely(l1BufNum_ == QUADRUPLE_BUFFER)) {
                    ComputeUbParamsByL1Size();
                    VectorProcess(tensorBlockB);
                } else if (l1BufNum_ == DOUBLE_BUFFER) {
                    if (l1BufIdx_ == GetSubBlockIdx()) {
                        VectorProcess(tensorBlockB);
                    }
                } else if (GetSubBlockIdx() == 0) {
                    VectorProcess(tensorBlockB);
                }
            }
            l1BufIdx_ = (l1BufIdx_ + 1) % l1BufNum_;
        }
    }

    __aicore__ inline BlockPrologue(const Params &params)
    {
        l1BufNum_ = params.l1BufNum;
        nUbSize_ = params.nUbSize;
        kUbSize_ = params.kUbSize;
        if constexpr (weightNz) {
            nSize_ = Get<0>(Get<0>(params.layoutB.GetShape())) * Get<1>(Get<0>(params.layoutB.GetShape()));
            kSize_ = Get<0>(Get<1>(params.layoutB.GetShape())) * Get<1>(Get<1>(params.layoutB.GetShape()));
        } else {
            nSize_ = Get<0>(params.layoutB.GetShape());
            kSize_ = Get<1>(params.layoutB.GetShape());
        }
        kL1Size_ = Get<3>(params.tileShapeL1);  // 3 in order to obtain k
        bL1Size_ = Get<1>(params.tileShapeL1) * kL1Size_;
        aL1Size_ = Get<0>(params.tileShapeL1) * Get<2>(params.tileShapeL1);  // 2 in order to obtain k
        if (likely(l1BufNum_ == QUADRUPLE_BUFFER)) {
            if constexpr (weightNz) {
                vecWeightInLen_ = (l1BufNum_ * nUbSize_ * kUbSize_) >> INT4_DTYPE_PARAM;
                vecWeightOutLen_ = l1BufNum_ * nUbSize_ * kUbSize_ * sizeof(ElementOut);
            } else {
                vecWeightInLen_ = (l1BufNum_ * (nUbSize_ * CeilAlign(kUbSize_, OFFSET_64))) >> INT4_DTYPE_PARAM;
                vecWeightOutLen_ =
                    l1BufNum_ * (CeilAlign(nUbSize_, BLOCK_CUBE) + 1) * CeilAlign(kUbSize_, ONE_BLK_SIZE);
            }
        } else {
            if constexpr (weightNz) {
                vecWeightInLen_ = (nUbSize_ * kUbSize_) >> INT4_DTYPE_PARAM;
                vecWeightOutLen_ = nUbSize_ * kUbSize_ * sizeof(ElementOut);
            } else {
                vecBufNum_ = Min(l1BufNum_, DOUBLE_BUFFER);
                vecWeightInLen_ = (vecBufNum_ * (nUbSize_ * CeilAlign(kUbSize_, OFFSET_64))) >> INT4_DTYPE_PARAM;
                vecWeightOutLen_ =
                    vecBufNum_ * (CeilAlign(nUbSize_, BLOCK_CUBE) + 1) * CeilAlign(kUbSize_, ONE_BLK_SIZE);
            }
        }
        weightOutUb_ = AscendC::LocalTensor<ElementOut>(AscendC::TPosition::VECCALC, 0, vecWeightOutLen_);
        weightInUb_ = AscendC::LocalTensor<ElementIn>(AscendC::TPosition::VECCALC, vecWeightOutLen_, vecWeightInLen_);
        l1Local_ = AscendC::LocalTensor<ElementOut>(AscendC::TPosition::B1, 0, L1_BUFFER_SIZE);
    }

    __aicore__ inline ~BlockPrologue()
    {
        if (l1BufNum_ == QUADRUPLE_BUFFER) {
            int64_t buffNum = Min(idx_ + 1, static_cast<int64_t>(l1BufNum_));
            for (int64_t index = 0; index < buffNum; index++) {
                WaitFlag<HardEvent::V_MTE2>(index);
                WaitFlag<HardEvent::MTE3_V>(index);
            }
        } else {
            if (idx_ > 0) {
                for (uint8_t index = 0; index < vecBufNum_; index++) {
                    WaitFlag<HardEvent::V_MTE2>(index);
                    WaitFlag<HardEvent::MTE3_V>(index);
                }
            } else if (idx_ == 0) {
                WaitFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::MTE3_V>(0);
            }
        }
        if (l1BufNum_ == QUADRUPLE_BUFFER) {
#pragma unroll
            for (int8_t index = 0; index < QUADRUPLE_BUFFER; index++) {
                WaitForCube();
            }
            return;
        }
        if (l1BufNum_ == vecBufNum_) {
            if (GetSubBlockIdx() == 0) {
                WaitForCube();
            }
            return;
        }
        if (l1BufNum_ == DOUBLE_BUFFER) {
            for (int8_t i = 0; i < DOUBLE_BUFFER; i++) {
                WaitForCube();
            }
            return;
        }
    }

private:
    __aicore__ inline void WaitForCube() { CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIV_AIC_FLAG); }

    __aicore__ inline void NotifyCube() { CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE3>(1); }

    template <class TensorB>
    __aicore__ inline void VectorProcess(const TensorB &tensorBlockB)
    {
        WaitForCube();
        ProcessL1(tensorBlockB);
        NotifyCube();
    }

    __aicore__ inline void ComputeUbParamsByL1Size()
    {
        if constexpr (weightNz) {
            if (kL1Len_ > kUbSize_) {
                kUbLen_ = kUbSize_;
                if (GetSubBlockIdx() == 1) {
                    kL1Aiv1Offset_ = kUbLen_;
                    kUbLen_ = kL1Len_ - kUbLen_;
                }
            } else {
                kL1Aiv1Offset_ = 0;
            }
        } else {
            if (nL1Len_ > nUbSize_) {
                nUbLen_ = nUbSize_;
                if (GetSubBlockIdx() == 1) {
                    nL1Aiv1Offset_ = nUbLen_;
                    nUbLen_ = nL1Len_ - nUbLen_;
                }
            } else {
                nL1Aiv1Offset_ = 0;
            }
        }
    }

    template <class TensorB>
    __aicore__ inline void ProcessL1(const TensorB &bGlobal)
    {
        int64_t l1Offset = (l1BufIdx_ & 0x1) * Act::Gemm::Max(L1_BUFFER_HALF_SIZE / sizeof(ElementOut),
                                                              DOUBLE_BUFFER * bL1Size_ + aL1Size_) +
                           ((l1BufIdx_ & 0x2) > 1) * bL1Size_;
        idx_ += 1;
        ubBufIdx_ = idx_ % l1BufNum_;
        if (idx_ >= l1BufNum_) {
            WaitFlag<HardEvent::V_MTE2>(ubBufIdx_);
        }
        CopyInTensorWeight(bGlobal);
        SetFlag<HardEvent::MTE2_V>(ubBufIdx_);
        if (idx_ >= l1BufNum_) {
            WaitFlag<HardEvent::MTE3_V>(ubBufIdx_);
        }
        WaitFlag<HardEvent::MTE2_V>(ubBufIdx_);
        AntiQuantCompute();
        SetFlag<HardEvent::V_MTE3>(ubBufIdx_);
        SetFlag<HardEvent::V_MTE2>(ubBufIdx_);
        WaitFlag<HardEvent::V_MTE3>(ubBufIdx_);
        int64_t nl1Offset = 0;
        int64_t kl1Offset = 0;
        if constexpr (weightNz) {
            if (GetSubBlockIdx() == 1 && kL1Len_ > kUbSize_) {
                kl1Offset += kUbSize_;
            }
        } else {
            if (GetSubBlockIdx() == 1 && nL1Len_ > nUbSize_) {
                nl1Offset += nUbSize_;
            }
        }
        l1Offset += nl1Offset * ONE_BLK_SIZE + kl1Offset * CeilAlign(nL1Len_, BLOCK_CUBE);
        if constexpr (weightNz) {
            CopyVecOut2L1(l1Offset, weightOutUb_[ubBufIdx_ * VEC_MAX_ELEM_B8]);
        } else {
            uint64_t weightOutUbOffset = ubBufIdx_ * (vecWeightOutLen_ / sizeof(ElementOut) / l1BufNum_);
            CopyVecOut2L1(l1Offset, weightOutUb_[weightOutUbOffset]);
        }
        SetFlag<HardEvent::MTE3_V>(ubBufIdx_);
    }

    template <class TensorB>
    __aicore__ inline void CopyInTensorWeight(const TensorB &bGlobal)
    {
        AscendC::DataCopyExtParams intriParams;
        intriParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<ElementIn> padParams;
        if constexpr (weightNz) {
            intriParams.blockCount = kUbLen_ / C0_SIZE_B8;
            intriParams.blockLen = nUbLen_ * BLOCK_CUBE;
            int64_t nAlignSize = CeilAlign(nSize_, BLOCK_CUBE);
            intriParams.srcStride = (nAlignSize - nUbLen_) * BLOCK_CUBE;
        } else {
            intriParams.blockCount = nUbLen_;
            intriParams.blockLen = kUbLen_ >> INT4_DTYPE_PARAM;
            intriParams.srcStride = (kSize_ - kUbLen_) >> INT4_DTYPE_PARAM;
        }
        uint64_t weightInOffset = ubBufIdx_ * (vecWeightInLen_ << INT4_DTYPE_PARAM) / l1BufNum_;
        AscendC::GlobalTensor<ElementIn> srcTensor;
        srcTensor.SetGlobalBuffer(bGlobal.address_);
        if constexpr (weightNz) {
            DataCopyPad(weightInUb_[weightInOffset], srcTensor[kL1Aiv1Offset_ * nSize_], intriParams, padParams);
        } else {
            DataCopyPad(weightInUb_[weightInOffset], srcTensor[nL1Aiv1Offset_ * kSize_], intriParams, padParams);
        }
    }

    __aicore__ inline void CopyVecOut2L1(int64_t l1Offset, const AscendC::LocalTensor<ElementOut> &ubLocal)
    {
        AscendC::DataCopyParams params;
        if constexpr (weightNz) {
            params.blockLen = BLOCK_NUM_REG;
            params.blockCount = nUbLen_ * kUbLen_ * sizeof(ElementOut) / VECTOR_REG_WIDTH;
            params.srcStride = (l1BufNum_ - 1) * BLOCK_NUM_REG;
            params.dstStride = 0;
            DataCopy(l1Local_[l1Offset], ubLocal, params);
        } else {
            params.blockLen = nUbLen_;
            params.blockCount = CeilDiv(kUbLen_, GROUP_SIZE);
            params.srcStride = 1 + CeilAlign(nUbLen_, BLOCK_CUBE) - nUbLen_;
            params.dstStride = CeilAlign(nL1Len_, BLOCK_CUBE) - nUbLen_;
            DataCopy(l1Local_[l1Offset], ubLocal, params);
        }
    }

    __aicore__ inline void AntiQuantComputeNormal()
    {
        uint16_t outExtend = static_cast<uint16_t>(nUbLen_);
        uint16_t innerExtend = CeilDiv(CeilAlign(kUbLen_, UB_ALIGN_SIZE_FOR_4BITS), VECTOR_REG_WIDTH_FOR_4BITS);
        uint32_t dataBlockStride = CeilAlign(nUbLen_, BLOCK_CUBE) + 1;
        uint32_t repeatStride = dataBlockStride * BLOCK_CUBE;
        int32_t outDimOffset = ONE_BLOCK_SIZE - innerExtend * repeatStride * ONE_BLOCK_SIZE;
        uint32_t maskB8Tail0 = Min(kUbLen_ % VECTOR_REG_WIDTH_FOR_4BITS, static_cast<int32_t>(VECTOR_REG_WIDTH)) +
                               kUbLen_ / VECTOR_REG_WIDTH_FOR_4BITS * VECTOR_REG_WIDTH;
        uint32_t maskB8Tail1 =
            Act::Gemm::Max(kUbLen_ % VECTOR_REG_WIDTH_FOR_4BITS - static_cast<int32_t>(VECTOR_REG_WIDTH), 0) +
            kUbLen_ / VECTOR_REG_WIDTH_FOR_4BITS * VECTOR_REG_WIDTH;
        RegCompute(outExtend, innerExtend, dataBlockStride, repeatStride, outDimOffset, maskB8Tail0, maskB8Tail1);
    }

    __aicore__ inline void RegCompute(uint16_t outExtend, uint16_t innerExtend, uint32_t dataBlockStride,
                                      uint32_t repeatStride, int32_t outDimOffset, uint32_t maskB8Tail0,
                                      uint32_t maskB8Tail1)
    {
        __VEC_SCOPE__
        {
            __local_mem__ int8_t *weightInUbBaseAddr = weightInUbBaseAddr_;
            __local_mem__ ElementOut *weightOutUbAddr = weightOutUbAddr_;
            __local_mem__ ElementOut *weightOutUbAddr1 = weightOutUbAddr1_;
            MicroAPI::RegTensor<uint8_t> wDIntlv0, wDIntlv1, wLoad0, sAnd0, sAnd1, wShr, wShl, s1, wOr0, wOr1, wdup1,
                wdup4;
            MicroAPI::RegTensor<int8_t> wdup0, wdup2, wdup3;
            MicroAPI::MaskReg preg = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
            MicroAPI::Duplicate<int8_t, MicroAPI::MaskMergeMode::ZEROING>(wdup0, DUP_CONFIG_2, preg);
            MicroAPI::Duplicate<uint8_t, MicroAPI::MaskMergeMode::ZEROING>(wdup1, DUP_CONFIG_MODE_1C, preg);
            MicroAPI::Duplicate<int8_t, MicroAPI::MaskMergeMode::ZEROING>(wdup2, DUP_CONFIG_2, preg);
            MicroAPI::Duplicate<int8_t, MicroAPI::MaskMergeMode::ZEROING>(wdup3, DUP_CONFIG_4, preg);
            MicroAPI::Duplicate<uint8_t, MicroAPI::MaskMergeMode::ZEROING>(wdup4, DUP_FLAG_80, preg);
            // 一次处理一个N轴
            for (uint16_t outIdx = 0; outIdx < outExtend; ++outIdx) {
                uint32_t maskWeight0Tmp = maskB8Tail0;
                uint32_t maskWeight1Tmp = maskB8Tail1;
                for (uint16_t repeatIdx = 0; repeatIdx < innerExtend; ++repeatIdx) {
                    MicroAPI::MaskReg MaskRegB8Tail0 = MicroAPI::UpdateMask<uint8_t>(maskWeight0Tmp);
                    MicroAPI::MaskReg MaskRegB8Tail1 = MicroAPI::UpdateMask<uint8_t>(maskWeight1Tmp);
                    MicroAPI::AddrReg aregWeightB8 =
                        MicroAPI::CreateAddrReg<uint8_t>(outIdx, kUbLen_ >> 1, repeatIdx, VEC_MAX_ELEM_B8);
                    MicroAPI::DataCopy(wLoad0, (__local_mem__ uint8_t *&)weightInUbBaseAddr, aregWeightB8);
                    // 提取E/M
                    MicroAPI::ShiftRight(wShr, wLoad0, wdup0, preg); //vr1
                    MicroAPI::And(wShr, wShr, wdup1, preg);          //vr1
                    MicroAPI::ShiftLeft(wShl, wLoad0, wdup2, preg);  //vr2
                    MicroAPI::And(wShl, wShl, wdup1, preg);          //vr2
                    // 提取S
                    MicroAPI::ShiftLeft(s1, wLoad0, wdup3, preg);    //vr3
                    MicroAPI::And(sAnd0, s1, wdup4, preg);           //vr3
                    MicroAPI::And(sAnd1, wLoad0, wdup4, preg);       //vr4
                    // 合并S/E/M
                    MicroAPI::Or(wOr0, wShr, sAnd1, preg);           //odd
                    MicroAPI::Or(wOr1, wShl, sAnd0, preg);           //even
                    MicroAPI::Interleave(wDIntlv0, wDIntlv1, wOr1, wOr0);
                    MicroAPI::DataCopy<uint8_t, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                       MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        (__local_mem__ uint8_t *&)weightOutUbAddr, wDIntlv0, dataBlockStride, repeatStride,
                        MaskRegB8Tail0);
                    MicroAPI::DataCopy<uint8_t, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                       MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        (__local_mem__ uint8_t *&)weightOutUbAddr1, wDIntlv1, dataBlockStride, repeatStride,
                        MaskRegB8Tail1);
                }
                weightOutUbAddr += outDimOffset;
                weightOutUbAddr1 += outDimOffset;
            }
        }
    }

    __aicore__ inline void AntiQuantCompute()
    {
        uint64_t weightOutUbOffset;
        uint64_t weightInUbOffset;
        if constexpr (weightNz) {
            weightOutUbOffset = ubBufIdx_ * VEC_MAX_ELEM_B8;
        } else {
            weightOutUbOffset = ubBufIdx_ * (vecWeightOutLen_ / sizeof(ElementOut) / l1BufNum_);
        }
        weightInUbOffset = ubBufIdx_ * (vecWeightInLen_ << INT4_DTYPE_PARAM) / l1BufNum_;
        weightInUbBaseAddr_ = (__local_mem__ int8_t *)weightInUb_[weightInUbOffset].GetPhyAddr();
        weightOutUbAddr_ = (__local_mem__ ElementOut *)weightOutUb_[weightOutUbOffset].GetPhyAddr();
        if constexpr (!weightNz) {
            uint16_t blockStride = CeilAlign(nUbLen_, BLOCK_CUBE) + 1;
            weightOutUbAddr1_ = weightOutUbAddr_ + VEC_MAX_ELEM_B8 * blockStride;
            AntiQuantComputeNormal();
        } else {
            AntiQuantComputeNKMxNz();
        }
    }

    __aicore__ inline void AntiQuantComputeNKMxNz()
    {
        static_assert(SupportType<ElementIn, fp4x2_e2m1_t, fp4x2_e1m2_t>(),
                      "only support fp4x2_e2m1_t and fp4x2_e1m2_t");
        uint32_t shiftLeftSize =
            IsSameType<ElementIn, fp4x2_e2m1_t>::value ? E2M1_SHIFT_LEFT_SIZE : E1M2_SHIFT_LEFT_SIZE;
        uint32_t andMask = IsSameType<ElementIn, fp4x2_e2m1_t>::value ? E2M1_AND_MASK : E1M2_AND_MASK;
        uint16_t innerExtend = CeilDiv(kUbLen_ * nUbLen_, VECTOR_REG_WIDTH);
        uint32_t innerDstExtend = VECTOR_REG_WIDTH * l1BufNum_;
        uint32_t innerSrcExtend = VECTOR_REG_WIDTH >> 1;
        __local_mem__ int8_t *weightInUbBaseAddr = weightInUbBaseAddr_;
        __local_mem__ ElementOut *weightOutUbAddr = weightOutUbAddr_;
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<int8_t> wdup0, wdup1, wdup2, wLoad0, wShl, wShr0, wShr1, wSel0, sAnd0;
            MicroAPI::MaskReg preg = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg pregVsel = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
            MicroAPI::Duplicate<int8_t, MicroAPI::MaskMergeMode::ZEROING>(wdup0, shiftLeftSize, preg);
            MicroAPI::Duplicate<int8_t, MicroAPI::MaskMergeMode::ZEROING>(wdup1, SHIFT_RIGHT_SIZE, preg);
            MicroAPI::Duplicate<int8_t, MicroAPI::MaskMergeMode::ZEROING>(wdup2, andMask, preg);
            for (uint16_t repeatIdx = 0; repeatIdx < innerExtend; ++repeatIdx) {
                MicroAPI::AddrReg aregWeightB8In = MicroAPI::CreateAddrReg<uint8_t>(repeatIdx, innerSrcExtend);
                MicroAPI::AddrReg aregWeightB8Out = MicroAPI::CreateAddrReg<uint8_t>(repeatIdx, innerDstExtend);
                MicroAPI::DataCopy<uint8_t, MicroAPI::LoadDist::DIST_US_B8>(
                    (MicroAPI::RegTensor<uint8_t> &)wLoad0, (__local_mem__ uint8_t *&)weightInUbBaseAddr,
                    aregWeightB8In);
                MicroAPI::ShiftRight(wShr0, wLoad0, wdup0, preg);
                MicroAPI::ShiftLeft(wShl, wLoad0, wdup1, preg);
                MicroAPI::ShiftRight(wShr1, wShl, wdup0, preg);
                MicroAPI::Select(wSel0, wShr1, wShr0, pregVsel);
                MicroAPI::And(sAnd0, wSel0, wdup2, preg);
                MicroAPI::DataCopy<uint8_t, MicroAPI::StoreDist::DIST_NORM_B8>(
                    (__local_mem__ uint8_t *&)weightOutUbAddr, (MicroAPI::RegTensor<uint8_t> &)sAnd0, aregWeightB8Out,
                    preg);
            }
        }
    }

    static constexpr int64_t DOUBLE_BUFFER = 2;
    static constexpr int64_t QUADRUPLE_BUFFER = 4;
    static constexpr uint64_t SYNC_MODE4 = 4;
    static constexpr uint64_t L1_BUFFER_SIZE = 512 * 1024;
    static constexpr uint64_t L1_BUFFER_HALF_SIZE = 256 * 1024;
    static constexpr uint64_t INT4_DTYPE_PARAM = 1;
    static constexpr uint64_t BLOCK_NUM_REG = VECTOR_REG_WIDTH / ONE_BLK_SIZE;
    static constexpr uint64_t SYNC_AIV_AIC_FLAG = 2;
    static constexpr uint64_t SINGLE_BUFFER = 1;
    static constexpr uint64_t GROUP_SIZE = 32;
    static constexpr int32_t C0_SIZE_B8 = 32;
    static constexpr int32_t UB_ALIGN_SIZE_FOR_4BITS = 64;
    static constexpr uint32_t DUP_CONFIG_2 = 0x2;
    static constexpr uint32_t DUP_CONFIG_MODE_1C = 0x1C;
    static constexpr uint32_t DUP_CONFIG_4 = 0x4;
    static constexpr uint32_t DUP_FLAG_80 = 0x80;
    static constexpr uint32_t E1M2_SHIFT_LEFT_SIZE = 0x3;
    static constexpr uint32_t E1M2_AND_MASK = 0x8E;
    static constexpr uint32_t E2M1_SHIFT_LEFT_SIZE = 0x2;
    static constexpr uint32_t E2M1_AND_MASK = 0x9C;
    static constexpr uint32_t SHIFT_RIGHT_SIZE = 0x4;
    static constexpr int32_t VEC_MAX_ELEM_B8 = VECTOR_REG_WIDTH / sizeof(ElementOut);
    static constexpr int32_t VECTOR_REG_WIDTH_FOR_4BITS = 512;
    static constexpr int32_t OFFSET_64 = 64;

    uint64_t nSize_;
    uint64_t kSize_;
    int32_t nUbSize_;
    int32_t kUbSize_;
    int32_t nUbLen_;
    int32_t kUbLen_;
    int64_t l1BufNum_;
    uint64_t kL1Size_;
    uint64_t kGmOffset_;
    int32_t nL1Len_;
    int32_t kL1Len_;
    uint64_t aL1Size_;
    uint64_t bL1Size_;
    uint64_t vecWeightOutLen_;
    uint64_t vecWeightInLen_;
    uint64_t ubBufIdx_;
    int64_t l1BufIdx_ = 0;
    int64_t idx_ = -1;
    uint64_t nL1Aiv1Offset_ = 0;
    uint64_t kL1Aiv1Offset_ = 0;
    uint8_t vecBufNum_ = SINGLE_BUFFER;
    uint8_t occupied_ = 0; //unused
    __local_mem__ ElementOut *weightOutUbAddr_;
    __local_mem__ ElementOut *weightOutUbAddr1_;
    __local_mem__ int8_t *weightInUbBaseAddr_;
    AscendC::LocalTensor<ElementIn> weightInUb_;
    AscendC::LocalTensor<ElementOut> weightOutUb_;
    AscendC::LocalTensor<ElementOut> l1Local_;
    static constexpr bool weightNz = Gemm::is_2d_nz_c0_32<decltype(LayoutIn{}.GetStride())>::value;
};
}  // namespace Act::Prologue

#endif