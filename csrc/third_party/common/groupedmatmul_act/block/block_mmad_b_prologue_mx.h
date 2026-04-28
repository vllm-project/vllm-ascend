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
 * \file block_mmad_b_prologue_mx.h
 * \brief
 */

#ifndef ACT_INCLUDE_MATMUL_BLOCK_BLOCK_MMAD_B_PROLOGUE_MX_H
#define ACT_INCLUDE_MATMUL_BLOCK_BLOCK_MMAD_B_PROLOGUE_MX_H

#include "../utils/common_utils.h"
#include "../utils/gemm_type.h"
#include "../utils/tensor_utils.h"
#include "../utils/tuple_utils.h"
#include "../policy/dispatch_policy.h"
#include "block_mmad.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace Act::Gemm::Block {
using AscendC::BLOCK_CUBE;
using AscendC::CeilAlign;
using AscendC::CeilDiv;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::MakeCoord;
using AscendC::MakeShape;
using AscendC::PipeBarrier;
using AscendC::QuePosition;
using AscendC::SetFlag;
using AscendC::SYNC_AIC_AIV_FLAG;
using AscendC::TEventID;
using AscendC::TPosition;
using AscendC::WaitFlag;

template <class L1TileShape_, class L0TileShape_, class ATypeTuple_, class BType_, class CType_, class BiasType_,
          class TileCopy_, class TileMmad_>
class BlockMmad<Act::Gemm::UbAntiquantWithScSc, L1TileShape_, L0TileShape_, ATypeTuple_, BType_, CType_, BiasType_,
                TileCopy_, TileMmad_> {
public:
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using AType = typename AscendC::Std::tuple_element<0, ATypeTuple_>::type;
    using ScaleType = typename AscendC::Std::tuple_element<1, ATypeTuple_>::type;
    using BiasType = BiasType_;
    using BType = BType_;
    using CType = CType_;
    using TileCopy = TileCopy_;
    using TileMmad = TileMmad_;
    using ElementA = typename AType::Element;
    using LayoutA = typename AType::Layout;
    using ElementC = typename CType::Element;
    using LayoutC = typename CType::Layout;
    using ElementB = typename BType::Element;
    using LayoutB = typename BType::Layout;
    using ElementBias = typename BiasType::Element;
    using LayoutBias = typename BiasType::Layout;
    using ElementScale = typename ScaleType::Element;
    using LayoutScale = typename ScaleType::Layout;
    using L0DataType = typename AscendC::GetL0DataType<ElementA, true>::Type;
    static_assert(AscendC::IsSameTypeV<ElementA, fp8_e4m3fn_t>);
    static_assert(AscendC::IsSameTypeV<L0DataType, AscendC::mx_fp8_e4m3_t>);

    struct Arguments {
        GM_ADDR ptrA = nullptr;
        GM_ADDR ptrC = nullptr;
        GM_ADDR ptrAScale = nullptr;
        GM_ADDR ptrBScale = nullptr;
        GM_ADDR ptrBias = nullptr;
        LayoutA layoutA;
        LayoutC layoutC;
        LayoutScale layoutScale;
        LayoutBias layoutBias;
    };

    struct Params {
        GM_ADDR ptrA = nullptr;
        GM_ADDR ptrC = nullptr;
        GM_ADDR ptrAScale = nullptr;
        GM_ADDR ptrBScale = nullptr;
        GM_ADDR ptrBias = nullptr;
        LayoutA layoutA;
        LayoutC layoutC;
        LayoutScale layoutScale;
        LayoutBias layoutBias;
        L1TileShape tileShapeL1;
        L0TileShape tileShapeL0;
        int64_t scaleFactor;
        uint64_t aL1BufNum;
        bool isBias;
    };

    template <class TensorA, class TensorAScale, class TensorBScale, class TensorBias, class TensorC, class Shape>
    __aicore__ inline void operator()(const TensorA &tensorA, const TensorAScale &scaleA, const TensorBScale &scaleB,
                                      const TensorBias &tensorBias, const TensorC &tensorC, const Shape &actualShape,
                                      [[maybe_unused]] const Params &params)
    {
        mL1Len_ = Get<0>(actualShape);
        nL1Len_ = Get<1>(actualShape);
        mL0Len_ = mL1Len_;
        nL0Len_ = nL1Len_;
        for (uint64_t kLoopIdx = 0; kLoopIdx < kTileCount_; kLoopIdx++) {
            kL1Offset_ = kLoopIdx * kL1Size_;
            kL1Len_ = Act::Gemm::Min(kSize_ - kL1Offset_, kL1Size_);
            WaitFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2_[l1BufIdx_]);
            auto tensorBlockA =
                GetTile(tensorA, MakeCoord(0, kL1Offset_), MakeShape(mL1Len_, static_cast<uint64_t>(kL1Len_)));
            GetAL1(kLoopIdx, scaleA, tensorBlockA);
            GetBL1(kLoopIdx, scaleB, tensorBias);
            SetFlag<HardEvent::MTE2_MTE1>(0);
            WaitFlag<HardEvent::MTE2_MTE1>(0);
            IterateMatmul(kLoopIdx);
            PostProcess(kLoopIdx);
        }
        GetTensorC(tensorC);
    }

    __aicore__ inline BlockMmad() = delete;
    __aicore__ inline BlockMmad(const Params &params)
    {
        hasBias_ = params.isBias;
        l1BufNum_ = params.aL1BufNum;
        scaleFactor_ = params.scaleFactor;
        kSize_ = static_cast<int64_t>(Get<1>(params.layoutA.GetShape()));
        nSize_ = Get<1>(params.layoutC.GetShape());
        kL1Size_ = Get<2>(params.tileShapeL1);  // 2 in order to obtain k
        auto nBL1Size = Get<1>(params.tileShapeL1);
        kL0Size_ = Get<2>(params.tileShapeL0);                    // 2 in order to obtain k
        bL1Size_ = nBL1Size * Get<3>(params.tileShapeL1);         // 3 in order to obtain kb
        aL1Size_ = Get<0>(params.tileShapeL1) * kL1Size_;         // 0 in order to obtain m
        scaleAL1Size_ = aL1Size_ * scaleFactor_ / GROUP_SIZE_32;  // aL1Size is an integer multiple of 32
        scaleBL1Size_ = bL1Size_ * scaleFactor_ / GROUP_SIZE_32;
        cL0Tensor_ = LocalTensor<float>(AscendC::TPosition::CO1, 0, L0C_BUFFER_SIZE);
        aL0Tensor_ = LocalTensor<L0DataType>(AscendC::TPosition::A2, 0, L0A_BUFFER_SIZE);
        bL0Tensor_ = LocalTensor<L0DataType>(AscendC::TPosition::B2, 0, L0B_BUFFER_SIZE);
        biasBtTensor_ = LocalTensor<float>(AscendC::TPosition::C2, 0, BIAS_TABLE_SIZE);
        InitBuf(nBL1Size);
        for (uint16_t index = 0; index < l1BufNum_; index++) {
            NotifyVector(index);
            SetFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2_[index]);
        }
        SetFlag<HardEvent::MTE1_MTE2>(eventIdsScaleAMte1ToMte2_[0]);
        SetFlag<HardEvent::MTE1_MTE2>(eventIdsScaleAMte1ToMte2_[1]);
        SetFlag<HardEvent::MTE1_MTE2>(eventIdsScaleBMte1ToMte2_[0]);
        SetFlag<HardEvent::MTE1_MTE2>(eventIdsScaleBMte1ToMte2_[1]);
        SetFlag<HardEvent::M_MTE1>(eventIdsMToMte1_[0]);
        SetFlag<HardEvent::M_MTE1>(eventIdsMToMte1_[1]);
        SetFlag<HardEvent::FIX_M>(eventIdsFixToM_[0]);
        SetFlag<HardEvent::FIX_M>(eventIdsFixToM_[1]);
        kTileCount_ = CeilDiv(kSize_, Get<2>(params.tileShapeL1));  // 2 in order to obtain k
    }

    __aicore__ inline ~BlockMmad()
    {
        WaitFlag<HardEvent::MTE1_MTE2>(eventIdsScaleAMte1ToMte2_[0]);
        WaitFlag<HardEvent::MTE1_MTE2>(eventIdsScaleAMte1ToMte2_[1]);
        WaitFlag<HardEvent::MTE1_MTE2>(eventIdsScaleBMte1ToMte2_[0]);
        WaitFlag<HardEvent::MTE1_MTE2>(eventIdsScaleBMte1ToMte2_[1]);
        WaitFlag<HardEvent::M_MTE1>(eventIdsMToMte1_[0]);
        WaitFlag<HardEvent::M_MTE1>(eventIdsMToMte1_[1]);
        WaitFlag<HardEvent::FIX_M>(eventIdsFixToM_[0]);
        WaitFlag<HardEvent::FIX_M>(eventIdsFixToM_[1]);
        for (uint16_t index = 0; index < l1BufNum_; index++) {
            WaitFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2_[index]);
        }
    }

private:
    __aicore__ inline void InitBuf(uint64_t nBL1Size)
    {
        uint64_t l1BufOffsetPart1 = 0;
        uint64_t l1BufOffsetPart2 = L1_BUFFER_HALF_SIZE;
        bL1LocalBuf0_ = LocalTensor<ElementA>(AscendC::TPosition::B1, l1BufOffsetPart1, bL1Size_);
        l1BufOffsetPart1 += bL1Size_ * sizeof(ElementA);
        if (l1BufNum_ == DOUBLE_BUFFER) {
            bL1LocalBuf1_ = LocalTensor<ElementA>(AscendC::TPosition::B1, l1BufOffsetPart2, bL1Size_);
            l1BufOffsetPart2 += bL1Size_ * sizeof(ElementA);
        } else if (l1BufNum_ == QUADRUPLE_BUFFER) {
            bL1LocalBuf1_ = LocalTensor<ElementA>(AscendC::TPosition::B1, l1BufOffsetPart2, bL1Size_);
            l1BufOffsetPart2 += bL1Size_ * sizeof(ElementA);
            bL1LocalBuf2_ = LocalTensor<ElementA>(AscendC::TPosition::B1, l1BufOffsetPart1, bL1Size_);
            l1BufOffsetPart1 += bL1Size_ * sizeof(ElementA);
            bL1LocalBuf3_ = LocalTensor<ElementA>(AscendC::TPosition::B1, l1BufOffsetPart2, bL1Size_);
            l1BufOffsetPart2 += bL1Size_ * sizeof(ElementA);
        }
        aL1LocalBuf0_ = LocalTensor<ElementA>(AscendC::TPosition::B1, l1BufOffsetPart1, aL1Size_);
        l1BufOffsetPart1 += aL1Size_ * sizeof(ElementA);
        if (l1BufNum_ == DOUBLE_BUFFER) {
            aL1LocalBuf1_ = LocalTensor<ElementA>(AscendC::TPosition::B1, l1BufOffsetPart2, aL1Size_);
            l1BufOffsetPart2 += aL1Size_ * sizeof(ElementA);
        } else if (l1BufNum_ == QUADRUPLE_BUFFER) {
            aL1LocalBuf1_ = LocalTensor<ElementA>(AscendC::TPosition::B1, l1BufOffsetPart2, aL1Size_);
            l1BufOffsetPart2 += aL1Size_ * sizeof(ElementA);
            aL1LocalBuf2_ = LocalTensor<ElementA>(AscendC::TPosition::B1, l1BufOffsetPart1, aL1Size_);
            l1BufOffsetPart1 += aL1Size_ * sizeof(ElementA);
            aL1LocalBuf3_ = LocalTensor<ElementA>(AscendC::TPosition::B1, l1BufOffsetPart2, aL1Size_);
            l1BufOffsetPart2 += aL1Size_ * sizeof(ElementA);
        }
        scaleBL1Buf0_ = LocalTensor<ElementScale>(AscendC::TPosition::B1, l1BufOffsetPart1, scaleBL1Size_);
        l1BufOffsetPart1 += scaleBL1Size_ * sizeof(ElementScale);
        scaleBL1Buf1_ = LocalTensor<ElementScale>(AscendC::TPosition::B1, l1BufOffsetPart2, scaleBL1Size_);
        l1BufOffsetPart2 += scaleBL1Size_ * sizeof(ElementScale);
        scaleAL1Buf0_ = LocalTensor<ElementScale>(AscendC::TPosition::A1, l1BufOffsetPart1, scaleAL1Size_);
        l1BufOffsetPart1 += scaleAL1Size_ * sizeof(ElementScale);
        scaleAL1Buf1_ = LocalTensor<ElementScale>(AscendC::TPosition::A1, l1BufOffsetPart2, scaleAL1Size_);
        l1BufOffsetPart2 += scaleAL1Size_ * sizeof(ElementScale);
        if (hasBias_) {
            biasL1Buf0_ = LocalTensor<ElementBias>(AscendC::TPosition::B1, l1BufOffsetPart1, nBL1Size);
            l1BufOffsetPart1 += nBL1Size * sizeof(ElementBias);
            if (biasBufNum_ > 1) {
                biasL1Buf1_ = LocalTensor<ElementBias>(AscendC::TPosition::B1, l1BufOffsetPart2, nBL1Size);
                l1BufOffsetPart2 += nBL1Size * sizeof(ElementBias);
            }
        }
    }

    __aicore__ inline void WaitForVector(uint64_t bL1BufIdx)
    {
        if (likely(l1BufNum_ == QUADRUPLE_BUFFER)) {
            CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(1 + FLAG_ID_MAX);
            CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(1);
            return;
        }
        if (likely(l1BufNum_ == DOUBLE_BUFFER)) {
            if (bL1BufIdx == 1) {
                CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(1 + FLAG_ID_MAX);
            } else {
                CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(1);
            }
        }
        if (unlikely(l1BufNum_ == 1)) {
            CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(1);
            return;
        }
    }

    __aicore__ inline void NotifyVector(uint64_t bL1BufIdx)
    {
        if (likely(l1BufNum_ == QUADRUPLE_BUFFER)) {
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG + FLAG_ID_MAX);
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
            return;
        }
        if (likely(l1BufNum_ == DOUBLE_BUFFER)) {
            if (bL1BufIdx == IDX_1) {
                CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG + FLAG_ID_MAX);
            } else {
                CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
            }
        }
        if (unlikely(l1BufNum_ == 1)) {
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
            return;
        }
    }

    template <class TensorAScale, class TensorA>
    __aicore__ inline void GetAL1(int64_t kLoopIdx, const TensorAScale &scaleA, const TensorA &tensorA)
    {
        if (kLoopIdx % scaleFactor_ == 0) {
            WaitFlag<HardEvent::MTE1_MTE2>(eventIdsScaleAMte1ToMte2_[scaleABufIdx_]);
            int64_t scaleKAL1Size = kL1Size_ * scaleFactor_;
            if (kL1Offset_ + scaleKAL1Size > kSize_) {
                scaleKAL1Size = kSize_ - kL1Offset_;
            }
            auto tensorBlockScaleA =
                GetTile(scaleA, MakeCoord(0, kL1Offset_ / GROUP_SIZE_32),
                        MakeShape(mL1Len_, static_cast<uint64_t>(CeilDiv(kL1Len_, GROUP_SIZE_32))));
            if (scaleABufIdx_ == 0) {
                CopyScaleDn2Nz(scaleAL1Buf0_, tensorBlockScaleA, 0, kL1Size_ * scaleFactor_ / GROUP_SIZE_32, mL1Len_,
                               scaleKAL1Size / GROUP_SIZE_32, kSize_ / GROUP_SIZE_32);
            } else {
                CopyScaleDn2Nz(scaleAL1Buf1_, tensorBlockScaleA, 0, kL1Size_ * scaleFactor_ / GROUP_SIZE_32, mL1Len_,
                               scaleKAL1Size / GROUP_SIZE_32, kSize_ / GROUP_SIZE_32);
            }
        }
        CopyND2NZ(l1BufIdx_, tensorA);
    }

    template <class TensorBScale, class TensorBias>
    __aicore__ inline void GetBL1(int64_t kLoopIdx, const TensorBScale &scaleB, const TensorBias &tensorBias)
    {
        if (hasBias_ && kLoopIdx == 0) {
            CopyBias2L1(tensorBias);
        }
        if (kLoopIdx % scaleFactor_ == 0) {
            auto tensorBlockScaleB =
                GetTile(scaleB, MakeCoord(0, kL1Offset_ / GROUP_SIZE_32),
                        MakeShape(mL1Len_, static_cast<uint64_t>(CeilDiv(kL1Len_, GROUP_SIZE_32))));
            WaitFlag<HardEvent::MTE1_MTE2>(eventIdsScaleBMte1ToMte2_[scaleBBufIdx_]);
            CopyScaleB2L1(tensorBlockScaleB);
        }
        WaitForVector(l1BufIdx_);
    }

    template <class TensorC>
    __aicore__ inline void GetTensorC(const TensorC &tensorC)
    {
        LocalTensor<float> cL0Tensor = cL0Tensor_[(madLoopIdx_ & 1) << BUFFER_HALF_SHL];
        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixParams;
        fixParams.nSize = nL0Len_;
        fixParams.mSize = mL0Len_;
        fixParams.srcStride = CeilAlign(mL0Len_, static_cast<int64_t>(BLOCK_CUBE));
        fixParams.dstStride = nSize_;
        if constexpr (IsSameType<ElementC, bfloat16_t>::value) {
            fixParams.quantPre = QuantMode_t::F322BF16;
        } else {
            fixParams.quantPre = QuantMode_t::F322F16;
        }
        fixParams.params.ndNum = 1;
        SetFlag<HardEvent::M_FIX>(0);
        WaitFlag<HardEvent::M_FIX>(0);
        GlobalTensor<ElementC> tmpTensorC;
        tmpTensorC.address_ = tensorC.address_;
        AscendC::Fixpipe<ElementC, float, AscendC::CFG_ROW_MAJOR>(tmpTensorC, cL0Tensor, fixParams);
        SetFlag<HardEvent::FIX_M>(eventIdsFixToM_[madLoopIdx_ & 1]);
        PostUpdateParams();
    }

    __aicore__ inline void IterateMatmul(int64_t kLoopIdx)
    {
        mL1AlignLen_ = CeilAlign(mL1Len_, BLOCK_CUBE);
        nL1AlignLen_ = CeilAlign(nL1Len_, BLOCK_CUBE);
        if (l1BufIdx_ == IDX_0) {
            aL1Tensor_ = aL1LocalBuf0_;
        } else if (l1BufIdx_ == IDX_1) {
            aL1Tensor_ = aL1LocalBuf1_;
        } else if (l1BufIdx_ == IDX_2) {
            aL1Tensor_ = aL1LocalBuf2_;
        } else {
            aL1Tensor_ = aL1LocalBuf3_;
        }
        if (l1BufIdx_ == IDX_0) {
            bL1Tensor_ = bL1LocalBuf0_;
        } else if (l1BufIdx_ == IDX_1) {
            bL1Tensor_ = bL1LocalBuf1_;
        } else if (l1BufIdx_ == IDX_2) {
            bL1Tensor_ = bL1LocalBuf2_;
        } else {
            bL1Tensor_ = bL1LocalBuf3_;
        }
        if (scaleABufIdx_ == IDX_0) {
            scaleAL1Tensor_ =
                scaleAL1Buf0_[(kL1Offset_ - (scaleFactor_ * kL1Size_) * (kLoopIdx / scaleFactor_)) / C0_SIZE_MX_B8];
        } else {
            scaleAL1Tensor_ =
                scaleAL1Buf1_[(kL1Offset_ - (scaleFactor_ * kL1Size_) * (kLoopIdx / scaleFactor_)) / C0_SIZE_MX_B8];
        }
        if (scaleBBufIdx_ == IDX_0) {
            scaleBL1Tensor_ =
                scaleBL1Buf0_[(kL1Offset_ - (scaleFactor_ * kL1Size_) * (kLoopIdx / scaleFactor_)) / C0_SIZE_MX_B8];
        } else {
            scaleBL1Tensor_ =
                scaleBL1Buf1_[(kL1Offset_ - (scaleFactor_ * kL1Size_) * (kLoopIdx / scaleFactor_)) / C0_SIZE_MX_B8];
        }
        if (hasBias_) {
            if (biasBufIdx_ == IDX_0) {
                biasL1Tensor_ = biasL1Buf0_;
            } else {
                biasL1Tensor_ = biasL1Buf1_;
            }
        }
        Iterate(kLoopIdx != 0);
    }

    __aicore__ inline void Iterate(bool enPartialSum)
    {
        LocalTensor<float> cL0Tensor = cL0Tensor_[(madLoopIdx_ & 1) << BUFFER_HALF_SHL];
        LocalTensor<L0DataType> aL0Tensor;
        LocalTensor<L0DataType> bL0Tensor;
        LocalTensor<float> biasBtTensor;
        AscendC::MmadParams mmadParams;
        mmadParams.m = CeilAlign(mL0Len_, static_cast<int64_t>(BLOCK_CUBE));
        mmadParams.n = CeilAlign(nL0Len_, static_cast<int64_t>(BLOCK_CUBE));
        mmadParams.cmatrixInitVal = !enPartialSum;
        mmadParams.cmatrixSource = false;
        int32_t kFractalIdx = totalKLoopIdx_ * kL1Size_ / kL0Size_;
        int32_t stepK = CeilDiv(kL1Len_, static_cast<int64_t>(kL0Size_));
        bool needPipeM = mmadParams.m * mmadParams.n < 2560;
        for (int32_t kL0Idx = 0; kL0Idx < stepK; kL0Idx++) {
            int32_t baseK = (kL0Idx == stepK - 1) ? kL1Len_ - kL0Idx * kL0Size_ : kL0Size_;
            kFractalIdx += kL0Idx;
            aL0Tensor = aL0Tensor_[(kFractalIdx & 1) << BUFFER_HALF_SHL];
            bL0Tensor = bL0Tensor_[(kFractalIdx & 1) << BUFFER_HALF_SHL];
            WaitFlag<HardEvent::M_MTE1>(eventIdsMToMte1_[kFractalIdx & 1]);
            CopyAMatrixL1ToL0A(aL0Tensor, kL0Idx, baseK);
            CopyBMatrixL1ToL0B(bL0Tensor, kL0Idx, baseK);
            if (kFractalIdx == 0) {
                WaitFlag<HardEvent::FIX_M>(eventIdsFixToM_[madLoopIdx_ & 1]);
            }
            mmadParams.k = baseK;
            if (kL0Idx > 0) {
                mmadParams.cmatrixInitVal = false;
            }
            if (hasBias_) {
                biasBtTensor = biasBtTensor_[(kFractalIdx & 1) * 512];  // bias table size is 512 elements
                CopyBiasL1ToBT(biasBtTensor);
            }
            SetFlag<HardEvent::MTE1_M>(eventIdsMte1ToM_[kFractalIdx & 1]);
            WaitFlag<HardEvent::MTE1_M>(eventIdsMte1ToM_[kFractalIdx & 1]);
            if (hasBias_ && kFractalIdx == 0) {
                mmadParams.cmatrixInitVal = false;
                AscendC::Mmad(cL0Tensor, aL0Tensor, bL0Tensor, biasBtTensor, mmadParams);
            } else {
                AscendC::Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
            }
            if (needPipeM) {
                PipeBarrier<PIPE_M>();
            }
            SetFlag<HardEvent::M_MTE1>(eventIdsMToMte1_[kFractalIdx & 1]);
        }
        totalKLoopIdx_ += 1;
    }

    __aicore__ inline void PostProcess(uint64_t kLoopIdx)
    {
        NotifyVector(l1BufIdx_);
        SetFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2_[l1BufIdx_]);
        l1BufIdx_ = (l1BufIdx_ + 1) % l1BufNum_;
        if (hasBias_ && kLoopIdx == kTileCount_ - 1) {
            biasBufIdx_ = (biasBufIdx_ + 1) % biasBufNum_;
        }
        if ((kLoopIdx + 1) % scaleFactor_ == 0 || kLoopIdx == (kTileCount_ - 1)) {
            SetFlag<HardEvent::MTE1_MTE2>(eventIdsScaleAMte1ToMte2_[scaleABufIdx_]);
            scaleABufIdx_ = (scaleABufIdx_ + 1) % DOUBLE_BUFFER;
        }
        if ((kLoopIdx + 1) % scaleFactor_ == 0 || kLoopIdx == (kTileCount_ - 1)) {
            SetFlag<HardEvent::MTE1_MTE2>(eventIdsScaleBMte1ToMte2_[scaleBBufIdx_]);
            scaleBBufIdx_ = (scaleBBufIdx_ + 1) % DOUBLE_BUFFER;
        }
    }

    __aicore__ inline void CopyAMatrixL1ToL0A(const LocalTensor<L0DataType> &aL0Tensor, int32_t kL0Idx, int32_t baseK)
    {
        AscendC::LoadData2DParamsV2 aL0Load2dParams;
        aL0Load2dParams.mStartPosition = 0;
        aL0Load2dParams.kStartPosition =
            CeilDiv(static_cast<uint64_t>(kL0Idx * kL0Size_ * sizeof(ElementA)), C0_SIZE_B8);
        aL0Load2dParams.mStep = CeilDiv(mL0Len_, static_cast<uint64_t>(BLOCK_CUBE));
        aL0Load2dParams.kStep = CeilDiv(baseK, C0_SIZE_B8);
        aL0Load2dParams.srcStride = CeilDiv(mL1AlignLen_ * C0_SIZE_B8, FRACTAL_SIZE);
        aL0Load2dParams.dstStride = aL0Load2dParams.srcStride;
        AscendC::LoadData2DMxParams aL0Load2dMx;
        aL0Load2dMx.xStartPosition = 0;
        aL0Load2dMx.yStartPosition = CeilDiv(kL0Idx * kL0Size_, GROUP_SIZE_32 * C0_SIZE_MX_B8);
        aL0Load2dMx.xStep = CeilDiv(mL0Len_, static_cast<uint64_t>(BLOCK_CUBE));
        aL0Load2dMx.yStep = CeilDiv(baseK, GROUP_SIZE_32 * C0_SIZE_MX_B8);
        aL0Load2dMx.srcStride = CeilDiv(kL1Size_ * scaleFactor_, GROUP_SIZE_32 * C0_SIZE_MX_B8);
        aL0Load2dMx.dstStride = CeilDiv(baseK, GROUP_SIZE_32 * C0_SIZE_MX_B8);
        AscendC::LoadData(aL0Tensor, aL1Tensor_, scaleAL1Tensor_, aL0Load2dParams, aL0Load2dMx);
    }

    __aicore__ inline void CopyBMatrixL1ToL0B(LocalTensor<L0DataType> bL0Tensor, int32_t kL0Idx, int32_t baseK)
    {
        AscendC::LoadData2DParamsV2 bL0Load2dParams;
        bL0Load2dParams.mStartPosition = 0;
        bL0Load2dParams.kStartPosition =
            CeilDiv(static_cast<uint64_t>(kL0Idx * kL0Size_ * sizeof(ElementA)), C0_SIZE_B8);
        bL0Load2dParams.mStep = CeilDiv(nL0Len_, static_cast<uint64_t>(BLOCK_CUBE));
        bL0Load2dParams.kStep = CeilDiv(baseK, C0_SIZE_B8);
        bL0Load2dParams.srcStride = CeilDiv(nL1AlignLen_ * C0_SIZE_B8, FRACTAL_SIZE);
        bL0Load2dParams.dstStride = bL0Load2dParams.srcStride;
        AscendC::LoadData2DMxParams bL0Load2dMx;
        bL0Load2dMx.xStartPosition = 0;
        bL0Load2dMx.yStartPosition = CeilDiv(kL0Idx * kL0Size_, GROUP_SIZE_32 * C0_SIZE_MX_B8);
        bL0Load2dMx.xStep = CeilDiv(nL0Len_, static_cast<uint64_t>(BLOCK_CUBE));
        bL0Load2dMx.yStep = CeilDiv(baseK, GROUP_SIZE_32 * C0_SIZE_MX_B8);
        bL0Load2dMx.srcStride = CeilDiv(kL1Size_ * scaleFactor_, GROUP_SIZE_32 * C0_SIZE_MX_B8);
        bL0Load2dMx.dstStride = CeilDiv(baseK, GROUP_SIZE_32 * C0_SIZE_MX_B8);
        AscendC::LoadData(bL0Tensor, bL1Tensor_, scaleBL1Tensor_, bL0Load2dParams, bL0Load2dMx);
    }

    __aicore__ inline void CopyBiasL1ToBT(LocalTensor<float> biasBtTensor)
    {
        constexpr auto biasDataType = IsSameType<float, ElementBias>::value ? 2 : 1;
        uint16_t lenBurst = CeilDiv(nL0Len_ * biasDataType * C0_SIZE_MX_B8, GROUP_SIZE_32);
        if constexpr (IsSameType<float, ElementBias>::value) {
            lenBurst = CeilAlign(lenBurst, static_cast<uint16_t>(C0_SIZE_MX_B8));
        }
        AscendC::DataCopy(biasBtTensor, biasL1Tensor_, {1, lenBurst, 0, 0});
    }

    template <class TensorBias>
    __aicore__ inline void CopyBias2L1(const TensorBias &tensorBias)
    {
        AscendC::DataCopyParams dataCopyParams = {1, static_cast<uint16_t>(nL0Len_ * sizeof(ElementBias)), 0, 0};
        AscendC::DataCopyPadParams dataCopyPadParams;
        AscendC::GlobalTensor<ElementBias> srcTensor;
        srcTensor.SetGlobalBuffer(tensorBias.address_);
        if (biasBufIdx_ == IDX_0) {
            AscendC::DataCopyPad(biasL1Buf0_, srcTensor, dataCopyParams, dataCopyPadParams);
        } else {
            AscendC::DataCopyPad(biasL1Buf1_, srcTensor, dataCopyParams, dataCopyPadParams);
        }
    }

    template <class TensorBScale>
    __aicore__ inline void CopyScaleB2L1(const TensorBScale &scaleB)
    {
        int64_t scaleKBL1Size = kL1Size_ * scaleFactor_;
        if (kL1Offset_ + scaleKBL1Size > kSize_) {
            scaleKBL1Size = kSize_ - kL1Offset_;
        }
        if (scaleBBufIdx_ == IDX_0) {
            CopyScaleDn2Nz(scaleBL1Buf0_, scaleB, 0, kL1Size_ * scaleFactor_ / GROUP_SIZE_32, nL1Len_,
                           scaleKBL1Size / GROUP_SIZE_32, kSize_ / GROUP_SIZE_32);
        } else {
            CopyScaleDn2Nz(scaleBL1Buf1_, scaleB, 0, kL1Size_ * scaleFactor_ / GROUP_SIZE_32, nL1Len_,
                           scaleKBL1Size / GROUP_SIZE_32, kSize_ / GROUP_SIZE_32);
        }
    }

    template <class TensorScale>
    __aicore__ inline void CopyScaleDn2Nz(const LocalTensor<ElementScale> &dst, const TensorScale &scale,
                                          [[maybe_unused]] const int row, const int col, const int height,
                                          const int width, const int gScaleCol)
    {
        AscendC::Dn2NzParams dn2NzParams;
        dn2NzParams.dnNum = 1;
        dn2NzParams.dValue = height;
        dn2NzParams.nValue = CeilDiv(static_cast<uint64_t>(width), static_cast<uint64_t>(SCALE_COPY_GROUP_SIZE));
        dn2NzParams.srcDnMatrixStride = SCALE_COPY_DEFAULT_STRIDE;
        dn2NzParams.srcDValue = gScaleCol / SCALE_COPY_GROUP_SIZE;
        dn2NzParams.dstNzC0Stride = CeilDiv(static_cast<uint64_t>(col), static_cast<uint64_t>(SCALE_COPY_GROUP_SIZE));
        dn2NzParams.dstNzNStride = SCALE_COPY_DEFAULT_NS_STRIDE;
        dn2NzParams.dstNzMatrixStride = SCALE_COPY_DEFAULT_STRIDE;
        AscendC::GlobalTensor<half> srcScale;
        srcScale.SetGlobalBuffer((__gm__ half *)scale.address_);
        auto bf16ScaleLocal = dst.template ReinterpretCast<half>();
        DataCopy(bf16ScaleLocal, srcScale, dn2NzParams);
    }

    template <class TensorA>
    __aicore__ inline void CopyND2NZ(int aL1Idx, const TensorA &tensorA)
    {
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        nd2nzParams.nValue = mL1Len_;
        nd2nzParams.dValue = kL1Len_;
        nd2nzParams.srcDValue = kSize_;
        nd2nzParams.srcNdMatrixStride = 0;
        nd2nzParams.dstNzC0Stride = CeilAlign(nd2nzParams.nValue, BLOCK_CUBE);
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 0;
        AscendC::GlobalTensor<ElementA> srcTensor;
        srcTensor.SetGlobalBuffer(tensorA.address_);
        if (aL1Idx == IDX_0) {
            DataCopy(aL1LocalBuf0_, srcTensor, nd2nzParams);
        } else if (aL1Idx == IDX_1) {
            DataCopy(aL1LocalBuf1_, srcTensor, nd2nzParams);
        } else if (aL1Idx == IDX_2) {
            DataCopy(aL1LocalBuf2_, srcTensor, nd2nzParams);
        } else if (aL1Idx == IDX_3) {
            DataCopy(aL1LocalBuf3_, srcTensor, nd2nzParams);
        }
    }

    __aicore__ inline void PostUpdateParams()
    {
        madLoopIdx_ += 1;
        totalKLoopIdx_ = 0;
    }

    LocalTensor<float> cL0Tensor_;
    LocalTensor<float> biasBtTensor_;
    LocalTensor<ElementA> aL1Tensor_;
    LocalTensor<ElementA> bL1Tensor_;
    LocalTensor<ElementA> aL1LocalBuf0_;
    LocalTensor<ElementA> aL1LocalBuf1_;
    LocalTensor<ElementA> aL1LocalBuf2_;
    LocalTensor<ElementA> aL1LocalBuf3_;
    LocalTensor<ElementA> bL1LocalBuf0_;
    LocalTensor<ElementA> bL1LocalBuf1_;
    LocalTensor<ElementA> bL1LocalBuf2_;
    LocalTensor<ElementA> bL1LocalBuf3_;
    LocalTensor<ElementScale> scaleAL1Buf0_;
    LocalTensor<ElementScale> scaleAL1Buf1_;
    LocalTensor<ElementScale> scaleBL1Buf0_;
    LocalTensor<ElementScale> scaleBL1Buf1_;
    LocalTensor<ElementScale> scaleAL1Tensor_;
    LocalTensor<ElementScale> scaleBL1Tensor_;
    LocalTensor<ElementBias> biasL1Buf1_;
    LocalTensor<ElementBias> biasL1Buf0_;
    LocalTensor<ElementBias> biasL1Tensor_;
    LocalTensor<L0DataType> aL0Tensor_;
    LocalTensor<L0DataType> bL0Tensor_;

    bool hasBias_;
    int8_t biasBufIdx_ = 0;
    int8_t biasBufNum_ = 2;
    uint8_t l1BufIdx_ = 0;
    int8_t occupied_ = 0;  // unused
    int64_t madLoopIdx_ = 0;
    int64_t totalKLoopIdx_ = 0;
    uint64_t scaleABufIdx_ = 0;
    uint64_t scaleBBufIdx_ = 0;
    int64_t kSize_;
    int64_t aL1Size_;
    int64_t bL1Size_;
    int64_t scaleAL1Size_;
    int64_t scaleBL1Size_;
    int64_t kL1Size_;
    int64_t kL1Len_;
    int64_t mL1AlignLen_;
    int64_t nL1AlignLen_;
    int64_t scaleFactor_;
    int64_t kL1Offset_;
    uint64_t nSize_;
    uint64_t mL0Len_;
    uint64_t nL0Len_;
    uint64_t kL0Size_;
    uint64_t mL1Len_;
    uint64_t nL1Len_;
    uint64_t kTileCount_;
    uint64_t l1BufNum_;

    static constexpr int32_t L0C_BUFFER_SIZE = 262144;
    static constexpr int32_t L0A_BUFFER_SIZE = 65536;
    static constexpr int32_t L0B_BUFFER_SIZE = 65536;
    static constexpr int32_t BIAS_TABLE_SIZE = 32 * 1024;
    static constexpr int32_t BUFFER_HALF_SHL = 15;
    static constexpr int32_t GROUP_SIZE_32 = 32;
    static constexpr int32_t SCALE_COPY_GROUP_SIZE = 2;
    static constexpr int32_t SCALE_COPY_DEFAULT_STRIDE = 0;
    static constexpr int32_t SCALE_COPY_DEFAULT_NS_STRIDE = 1;
    static constexpr uint64_t IDX_0 = 0;
    static constexpr uint64_t IDX_1 = 1;
    static constexpr uint64_t IDX_2 = 2;
    static constexpr uint64_t IDX_3 = 3;
    static constexpr uint64_t SYNC_MODE4 = 4;
    static constexpr uint64_t SYNC_AIV_AIC_FLAG = 2;
    static constexpr uint64_t QUADRUPLE_BUFFER = 4;
    static constexpr uint64_t FLAG_ID_MAX = 16;
    static constexpr uint64_t L1_BUFFER_HALF_SIZE = 256 * 1024;
    static constexpr uint64_t DOUBLE_BUFFER = 2;
    static constexpr uint64_t C0_SIZE_B8 = 32;
    static constexpr uint64_t FRACTAL_SIZE = 512;  // 16 * 32
    static constexpr uint64_t C0_SIZE_MX_B8 = 2;
    static constexpr uint8_t MAX_AL1_BUF_NUM = 4;
    static constexpr TEventID eventIdsMte1ToMte2_[MAX_AL1_BUF_NUM] = {0, 1, 2, 3};
    static constexpr TEventID eventIdsScaleAMte1ToMte2_[DOUBLE_BUFFER] = {4, 5};
    static constexpr TEventID eventIdsScaleBMte1ToMte2_[DOUBLE_BUFFER] = {6, 7};
    static constexpr TEventID eventIdsMToMte1_[2] = {0, 1};
    static constexpr TEventID eventIdsMte1ToM_[2] = {0, 1};
    static constexpr TEventID eventIdsFixToM_[2] = {0, 1};
};
}  // namespace Act::Gemm::Block
#endif