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
 * \file block_mmad_qbmm_mx.h
 * \brief
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad.h"
#include "tensor_api/tensor.h"
#include "blaze/gemm/tile/tile_trait.h"
#include "blaze/gemm/tile/pad_mx_kl1.h"
#include "apace/basic/fragment_tensor/fragment_tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <
    uint64_t A_FULL_LOAD_MODE, bool ATOMIC_ADD, class AType_, class LayoutA_, class BType_, class LayoutB_,
    class CType_, class LayoutC_, class BiasType_, class LayoutBias_>
class QmmMxBlockMmadFragment;

#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)

template <
    uint64_t A_FULL_LOAD_MODE, bool ATOMIC_ADD, class AType_, class LayoutA_, class BType_, class LayoutB_,
    class CType_, class LayoutC_, class BiasType_, class LayoutBias_>
class QmmMxBlockMmadFragment {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using BiasType = BiasType_;
    using DispatchPolicy = MatmulWithScaleMx<A_FULL_LOAD_MODE, ATOMIC_ADD>;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    uint64_t k_;
    uint64_t l1BufNum_{1};
    uint64_t kL1Iter_{0};
    uint64_t kL1_{1};
    uint64_t scaleKL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    bool isBias_{false};
    static constexpr bool weightNz = IsWeightNz<LayoutB>::value;
    static constexpr bool transA = IsTrans<LayoutA>::value;
    static constexpr bool transB = IsTrans<LayoutB>::value;

    constexpr static uint64_t HALF_L0_SIZE = AscendC::TOTAL_L0A_SIZE / DOUBLE_BUFFER_COUNT;
    constexpr static uint64_t HALF_L0C_SIZE = AscendC::TOTAL_L0C_SIZE / DOUBLE_BUFFER_COUNT;
    constexpr static uint64_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    constexpr static uint64_t SCALE_C0 = 2;
    constexpr static uint64_t SCALE_BUFFER_NUM = 2;

    using MakeLayoutAL1 = AscendC::Std::conditional_t<
        transA, AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>>;
    using MakeLayoutBL1 = AscendC::Std::conditional_t<
        transB, AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>>;

    uint64_t abL1LoopCnt_{0};
    uint64_t scaleLoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    uint64_t splitKNum_{1};
    bool enableL0cPingPong_{false};

    struct TileL1L0Param {
        uint64_t curM = 0;
        uint64_t curN = 0;
        uint64_t curGmAKL1 = 0;
        uint64_t curGmBKL1 = 0;
        uint64_t curPadAKL1 = 0; // pad to 64 align
        uint64_t curPadBKL1 = 0; // pad to 64 align
    };

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR scaleAGmAddr{nullptr};
        GM_ADDR scaleBGmAddr{nullptr};
    };

    struct L1Params {
        uint64_t kL1;
        uint64_t scaleKL1;
        uint64_t l1BufNum;
    };

    template <typename TSA, typename TSB>
    struct ScalePair {
        TSA scaleA;
        TSB scaleB;
    };

    __aicore__ inline QmmMxBlockMmadFragment()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_1);
        AscendC::SetMMLayoutTransform(true); // true means column first when fixpipe_l0c2out
    }

    __aicore__ inline ~QmmMxBlockMmadFragment()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_1);
        AscendC::SetMMLayoutTransform(false); // false means row first when fixpipe_l0c2out
    }

public:

    __aicore__ inline void Init(
        const ProblemShape& problemShape, const BlockShape& l0TileShape, const L1Params& l1Params, bool isBias,
        bool dbL0C, uint64_t splitKNum = 1)
    {
        k_ = AscendC::Te::Get<IDX_K_IDX>(problemShape);
        kL1_ = l1Params.kL1;
        scaleKL1_ = l1Params.scaleKL1;
        splitKNum_ = splitKNum;
        baseM_ = AscendC::Te::Get<IDX_M_IDX>(l0TileShape);
        baseN_ = AscendC::Te::Get<IDX_N_IDX>(l0TileShape);
        baseK_ = AscendC::Te::Get<IDX_K_IDX>(l0TileShape);
        isBias_ = isBias;
        l1BufNum_ = l1Params.l1BufNum;
        enableL0cPingPong_ = dbL0C;
        constexpr uint64_t sizeShift = IsFp4<AType>() ? 1 : 0;
        bL1OneBuffer_ = (baseN_ * kL1_) >> sizeShift;
        scaleBL1OneBuffer_ = baseN_ * Blaze::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        biasL1OneBuffer_ = isBias_ ? baseN_ * sizeof(BiasType) : 0;
        if constexpr (DispatchPolicy::FULL_LOAD_MODE == 0) {
            aL1OneBuffer_ = (baseM_ * Blaze::Gemm::CeilAlign(kL1_, MXFP_DIVISOR_SIZE)) >> sizeShift;
            scaleAL1OneBuffer_ = baseM_ * Blaze::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
            for (int32_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
                // 2 buffer: L1 space is : A0|B0|AScale0|BScale0|bias0|...|A1|B1|AScale1|BScale1|bias1|...
                // 4 buffer: L1 space is : A0A2|B0B2|AScale0|BScale0|bias0|...|A1A3|B1B3|AScale1|BScale1|bias1|...
                uint64_t l1Offset = (AscendC::TOTAL_L1_SIZE >> 1) * (bufferId & 1);
                l1BufferAOffset_[bufferId] = l1Offset + aL1OneBuffer_ * (bufferId >> 1);
                l1BufferBOffset_[bufferId] =
                    l1Offset + aL1OneBuffer_ * (l1BufNum_ >> 1) + bL1OneBuffer_ * (bufferId >> 1);
            }
            for (int32_t bufferId = 0; bufferId < SCALE_BUFFER_NUM; bufferId++) {
                l1BufferScaleAOffset_[bufferId] = l1BufferBOffset_[bufferId] + bL1OneBuffer_ * (l1BufNum_ >> 1);
                l1BufferScaleBOffset_[bufferId] = l1BufferScaleAOffset_[bufferId] + scaleAL1OneBuffer_;
                l1BufferBiasOffset_[bufferId] = l1BufferScaleBOffset_[bufferId] + scaleBL1OneBuffer_;
            }
        } else {
            uint64_t mAlign = Blaze::Gemm::CeilAlign(baseM_, transA ? C0_SIZE : BLOCK_CUBE);
            uint64_t kAlign = Blaze::Gemm::CeilAlign(k_, MXFP_DIVISOR_SIZE);
            aL1OneBuffer_ = (mAlign * kAlign) >> sizeShift;
            scaleAL1OneBuffer_ = baseM_ * Blaze::Gemm::CeilDiv(k_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
            // 2 buffer: L1 space is : B0|BScale0|bias0|A|AScale|...|B1|BScale1|bias1|
            // 4 buffer: L1 space is : B0B2|BScale0|bias0|A|AScale|...|B1B3|BScale1|bias1|...
            l1BufferAOffset_[0] = bL1OneBuffer_ * (l1BufNum_ >> 1) + scaleBL1OneBuffer_ + biasL1OneBuffer_;
            l1BufferScaleAOffset_[0] = l1BufferAOffset_[0] + aL1OneBuffer_;
            uint64_t b1Offset = l1BufferScaleAOffset_[0] + scaleAL1OneBuffer_ >= (AscendC::TOTAL_L1_SIZE >> 1) ?
                                    l1BufferScaleAOffset_[0] + scaleAL1OneBuffer_ :
                                    (AscendC::TOTAL_L1_SIZE >> 1);
            for (int32_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
                l1BufferBOffset_[bufferId] = b1Offset * (bufferId & 1) + bL1OneBuffer_ * (bufferId >> 1);
            }
            for (int32_t bufferId = 0; bufferId < SCALE_BUFFER_NUM; bufferId++) {
                l1BufferScaleBOffset_[bufferId] = l1BufferBOffset_[bufferId] + bL1OneBuffer_ * (l1BufNum_ >> 1);
                l1BufferBiasOffset_[bufferId] = l1BufferScaleBOffset_[bufferId] + scaleBL1OneBuffer_;
            }
        }
        kL1Iter_ = CeilDiv(k_, kL1_);
    }

    template <typename TensorScaleA, typename TensorScaleB>
    __aicore__ inline auto CopyScalesInL1(
        TensorScaleA const& gmScaleA, TensorScaleB const& gmScaleB, TileL1L0Param& tileL1L0Param, uint64_t scaleL1BufId,
        uint64_t iter0)
    {
        uint64_t kL1Offset = iter0 * kL1_;
        auto layoutScaleBL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::NNLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
            Blaze::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, tileL1L0Param.curN);
        auto tensorScaleBL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, fp8_e8m0_t>(
                l1BufferScaleBOffset_[scaleL1BufId]), layoutScaleBL1);
        auto CopyScaleGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        if constexpr (DispatchPolicy::FULL_LOAD_MODE == 0) {
            auto layoutScaleAL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
                tileL1L0Param.curM, Blaze::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE);
            auto tensorScaleAL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, fp8_e8m0_t>(
                    l1BufferScaleAOffset_[scaleL1BufId]), layoutScaleAL1);
            if (iter0 % (scaleKL1_ / kL1_) == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + scaleL1BufId);
                uint64_t curScaleKL1 = scaleKL1_;
                if (kL1Offset + curScaleKL1 > k_) {
                    curScaleKL1 = k_ - kL1Offset;
                }
                uint64_t scaleKStart = iter0 * Blaze::Gemm::CeilDiv(kL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
                uint64_t scaleKLength = Blaze::Gemm::CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
                auto blockScaleA = gmScaleA.Slice(
                    AscendC::Te::MakeCoord(0, scaleKStart),
                    AscendC::Te::MakeShape(tileL1L0Param.curM, scaleKLength));
                    FragmentSliceCopy(CopyScaleGM2L1, tensorScaleAL1, blockScaleA);

                auto gmBlockScaleB = gmScaleB.Slice(
                    AscendC::Te::MakeCoord(scaleKStart, 0),
                    AscendC::Te::MakeShape(scaleKLength, tileL1L0Param.curN));
                AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleBL1, gmBlockScaleB);
            }
            return ScalePair<decltype(tensorScaleAL1), decltype(tensorScaleBL1)>{
                tensorScaleAL1, tensorScaleBL1};
        } else {
            auto layoutScaleAL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
                tileL1L0Param.curM, Blaze::Gemm::CeilDiv(k_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE);
            auto tensorScaleAL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, fp8_e8m0_t>(
                    l1BufferScaleAOffset_[0]), layoutScaleAL1);
            if (iter0 % (scaleKL1_ / kL1_) == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + scaleL1BufId);
                uint64_t curScaleKL1 = scaleKL1_;
                if (kL1Offset + curScaleKL1 > k_) {
                    curScaleKL1 = k_ - kL1Offset;
                }
                uint64_t scaleKStart = iter0 * Blaze::Gemm::CeilDiv(kL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
                uint64_t scaleKLength = Blaze::Gemm::CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
                auto gmBlockScaleB = gmScaleB.Slice(
                    AscendC::Te::MakeCoord(scaleKStart, 0),
                    AscendC::Te::MakeShape(scaleKLength, tileL1L0Param.curN));
                AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleBL1, gmBlockScaleB);
            }
            if (abL1LoopCnt_ == 0) {
                    FragmentSliceCopy(CopyScaleGM2L1, tensorScaleAL1, gmScaleA);
            }
            return ScalePair<decltype(tensorScaleAL1), decltype(tensorScaleBL1)>{
                tensorScaleAL1, tensorScaleBL1};
        }
    }

    template <typename TensorA>
    __aicore__ inline auto CopyAInL1(TensorA const& gmA, TileL1L0Param& tileL1L0Param, uint64_t l1BufId, uint64_t iter0)
    {
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        if constexpr (DispatchPolicy::FULL_LOAD_MODE == 0) {
            auto layoutAL1 = MakeLayoutAL1{}(tileL1L0Param.curM, tileL1L0Param.curPadAKL1);
            auto tensorAL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(l1BufferAOffset_[l1BufId]), layoutAL1);
            auto blockA = gmA.Slice(
                AscendC::Te::MakeCoord(0, iter0 * kL1_),
                AscendC::Te::MakeShape(tileL1L0Param.curM, tileL1L0Param.curGmAKL1));
            Apace::Basic::PadMxKAL1Zero(tensorAL1, tileL1L0Param.curGmAKL1);
                FragmentSliceCopy(copyGM2L1, tensorAL1, blockA);
            return tensorAL1;
        } else {
            auto layoutAL1 = MakeLayoutAL1{}(tileL1L0Param.curM, Blaze::Gemm::CeilAlign(k_, MXFP_DIVISOR_SIZE));
            auto tensorTotalAL1 =
                AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(
                    l1BufferAOffset_[0]), layoutAL1);
            auto tensorAL1 = tensorTotalAL1.Slice(
                AscendC::Te::MakeCoord(0, iter0 * kL1_),
                AscendC::Te::MakeShape(tileL1L0Param.curM, tileL1L0Param.curPadAKL1));
            if (abL1LoopCnt_ < kL1Iter_) {
                auto blockA = gmA.Slice(
                    AscendC::Te::MakeCoord(0, iter0 * kL1_),
                    AscendC::Te::MakeShape(tileL1L0Param.curM, tileL1L0Param.curGmAKL1));
                Apace::Basic::PadMxKAL1Zero(tensorAL1, tileL1L0Param.curGmAKL1);
                    FragmentSliceCopy(copyGM2L1, tensorAL1, blockA);
            }
            return tensorAL1;
        }
    }

    template <
        typename TensorScaleAL1, typename TensorScaleBL1, typename TensorAL1, typename TensorBL1,
        typename TensorBiasL1, typename TensorL0C>
    __aicore__ inline void Iterate(
        TileL1L0Param& tileL1L0Param, uint64_t iter0, TensorScaleAL1& tensorScaleAL1,
        TensorScaleBL1& tensorScaleBL1, TensorAL1& tensorAL1, TensorBL1& tensorBL1,
        TensorBiasL1& tensorBiasL1, TensorL0C& tensorL0C, uint64_t splitKIdx)
    {
        // 从scaleKL1中切出kL1_对应的部分
        auto scaleKL1Len = Blaze::Gemm::CeilDiv(kL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        auto coordScaleKL1 = (iter0 % (scaleKL1_ / kL1_)) * scaleKL1Len;
        auto tensorBlockScaleBL1 = tensorScaleBL1.Slice(
            AscendC::Te::MakeCoord(coordScaleKL1, 0), AscendC::Te::MakeShape(scaleKL1Len, tileL1L0Param.curN));
        if constexpr (DispatchPolicy::FULL_LOAD_MODE != 0) {
            coordScaleKL1 = iter0 * scaleKL1Len;
        }
        auto tensorBlockScaleAL1 = tensorScaleAL1.Slice(
            AscendC::Te::MakeCoord(0, coordScaleKL1), AscendC::Te::MakeShape(tileL1L0Param.curM, scaleKL1Len));

        uint64_t kL0Iter = Blaze::Gemm::CeilDiv(tileL1L0Param.curGmBKL1, baseK_);
        for (uint16_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
            auto curKL0 = (iter1 * baseK_ + baseK_ > tileL1L0Param.curPadBKL1) ?
                              (tileL1L0Param.curPadBKL1 - iter1 * baseK_) :
                              baseK_;
            // Load data to L0 and open DB
            uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
            uint16_t mte1WaitMFlag = (l0PingPong_ & 0x1) + M_MTE1_FLAG_0;
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mte1WaitMFlag);

            // A, ScaleA L1->L0
            auto layoutAL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>(
                tileL1L0Param.curM, curKL0);
            auto tensorAL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, AType>(l0Offset), layoutAL0);
            auto tensorBlockAL1 = tensorAL1.Slice(
                AscendC::Te::MakeCoord(0, iter1 * baseK_), AscendC::Te::MakeShape(tileL1L0Param.curM, curKL0));
            auto CopyL12L0A = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{});
            AscendC::Te::Copy(CopyL12L0A, tensorAL0, tensorBlockAL1);

            auto CopyL12L0ScaleA = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0ScaleA{});
            auto layoutScaleAL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
                tileL1L0Param.curM, Blaze::Gemm::CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE);
            auto tensorScaleAL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0ScaleA, fp8_e8m0_t>(l0Offset >> 4), layoutScaleAL0);
            AscendC::Te::Copy(
                CopyL12L0ScaleA, tensorScaleAL0,
                tensorBlockScaleAL1.Slice(
                    AscendC::Te::MakeCoord(
                        0, iter1 * Blaze::Gemm::CeilDiv(baseK_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE),
                    AscendC::Te::MakeShape(
                        tileL1L0Param.curM, Blaze::Gemm::CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE)));

            // bias L1->BT
            auto layoutBt = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(
                1UL, Blaze::Gemm::CeilAlign(tileL1L0Param.curN, BLOCK_CUBE));
            auto tensorBt = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::BIAS, float>(baseN_ * biasBufId_ * sizeof(float)),
                layoutBt);
            if (NeedBias(iter0, iter1, splitKIdx)) {
                auto CopyL12BT = AscendC::Te::MakeCopy(AscendC::Te::CopyL12BT{});
                AscendC::Te::Copy(CopyL12BT, tensorBt, tensorBiasL1);
            }

            // B, scaleB L1->L0
            auto layoutBL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>(
                curKL0, tileL1L0Param.curN);
            auto tensorBL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, BType>(l0Offset), layoutBL0);
            auto tensorBlockBL1 = tensorBL1.Slice(
                AscendC::Te::MakeCoord(iter1 * baseK_, 0), AscendC::Te::MakeShape(curKL0, tileL1L0Param.curN));
            auto CopyL12L0B = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{});
            AscendC::Te::Copy(CopyL12L0B, tensorBL0, tensorBlockBL1);

            auto CopyL12L0ScaleB = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0ScaleB{});
            auto layoutScaleBL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::NNLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
                Blaze::Gemm::CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, tileL1L0Param.curN);
            auto tensorScaleBL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0ScaleB, fp8_e8m0_t>(l0Offset >> 4), layoutScaleBL0);
            AscendC::Te::Copy(
                CopyL12L0ScaleB, tensorScaleBL0,
                tensorBlockScaleBL1.Slice(
                    AscendC::Te::MakeCoord(
                        iter1 * Blaze::Gemm::CeilDiv(baseK_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, 0),
                    AscendC::Te::MakeShape(
                        Blaze::Gemm::CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, tileL1L0Param.curN)));

            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);

            Mmad(tileL1L0Param, iter0, iter1, kL0Iter, curKL0, tensorL0C, tensorAL0, tensorBL0, tensorBt, splitKIdx);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mte1WaitMFlag);
            l0PingPong_++;
        }
    }

    template <
        typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorBias,
        typename TensorC = int>
    __aicore__ inline void operator()(
        TensorA const& gmA, TensorB const& gmB, TensorScaleA const& gmScaleA, TensorScaleB const& gmScaleB,
        TensorBias const& gmBias, GM_ADDR* cFragAddrs, uint64_t mPerRank, uint64_t tileM,
        uint64_t tileCnt, uint64_t tailM, uint64_t remoteRankCnt, int64_t cRowStride,
        BlockShape const& singleShape, int64_t mPosGlobal, int64_t nPosGlobal,
        uint64_t splitKIdx = 0, TensorC const& gmC = 0)
    {
        TileL1L0Param tileL1L0Param;
        tileL1L0Param.curM = AscendC::Te::Get<IDX_M_TILEIDX>(singleShape);
        tileL1L0Param.curN = AscendC::Te::Get<IDX_N_TILEIDX>(singleShape);
        uint64_t l0cOffset = (l0cPingPong_ & 1) * HALF_L0C_SIZE;
        auto layoutL0C = AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE_L0C>>{}(
            tileL1L0Param.curM, tileL1L0Param.curN);
        auto tensorL0C = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(l0cOffset), layoutL0C);
        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            uint64_t l1BufId = abL1LoopCnt_ & (l1BufNum_ - 1);
            uint64_t scaleL1BufId = scaleLoopCnt_ & 1;

            // scaleA, scaleB GM->L1
            auto scalePair = CopyScalesInL1(gmScaleA, gmScaleB, tileL1L0Param, scaleL1BufId, iter0);
            auto& tensorScaleAL1 = scalePair.scaleA;
            auto& tensorScaleBL1 = scalePair.scaleB;

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            biasBufId_ = scaleL1BufId;
            tileL1L0Param.curGmBKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - iter0 * kL1_) : kL1_;
            tileL1L0Param.curPadBKL1 =
                Blaze::Gemm::CeilAlign(tileL1L0Param.curGmBKL1, MXFP_DIVISOR_SIZE); // pad to 64 align
            tileL1L0Param.curGmAKL1 = tileL1L0Param.curGmBKL1;
            tileL1L0Param.curPadAKL1 = tileL1L0Param.curPadBKL1;

            // A GM->L1
            auto tensorAL1 = CopyAInL1(gmA, tileL1L0Param, l1BufId, iter0);

            // bias GM->L1
            auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
            auto layoutBiasL1 = AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn>{}(
                1UL, Blaze::Gemm::CeilAlign(tileL1L0Param.curN, BLOCK_CUBE));
            auto tensorBiasL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BiasType>(
                    l1BufferBiasOffset_[biasBufId_]), layoutBiasL1);
            if (isBias_ && iter0 == 0 && splitKIdx == 0) {
                AscendC::Te::Copy(copyGM2L1, tensorBiasL1, gmBias);
            }

            // B GM->L1; 先slice再copy
            auto layoutBL1 = MakeLayoutBL1{}(tileL1L0Param.curPadBKL1, tileL1L0Param.curN);
            auto tensorBL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(l1BufferBOffset_[l1BufId]), layoutBL1);
            auto gmBlockB = gmB.Slice(
                AscendC::Te::MakeCoord(iter0 * kL1_, 0),
                AscendC::Te::MakeShape(tileL1L0Param.curGmBKL1, tileL1L0Param.curN));
            Blaze::Gemm::Tile::PadMxKBL1::PadZero(tensorBL1, gmBlockB);
            AscendC::Te::Copy(copyGM2L1, tensorBL1, gmBlockB);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

            Iterate(tileL1L0Param, iter0, tensorScaleAL1, tensorScaleBL1, tensorAL1,
                    tensorBL1, tensorBiasL1, tensorL0C, splitKIdx);

            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            if ((iter0 + 1) % (scaleKL1_ / kL1_) == 0 || iter0 == kL1Iter_ - 1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + scaleL1BufId);
                scaleLoopCnt_++;
            }
            abL1LoopCnt_++;
        }
        if (splitKIdx == splitKNum_ - 1) {
            ScatterL0C2GM(gmC, tensorL0C);
        }
    }

    template <typename TensorC, typename TensorL0C>
    __aicore__ inline void ScatterL0C2GM(TensorC const& gmC, TensorL0C& tensorL0C)
    {
        auto copyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
        FragmentSliceScatter(copyL0C2GM.with(AscendC::Te::FixpipeParams(3)), gmC, tensorL0C);
        if (enableL0cPingPong_) {
            l0cPingPong_++;
        }
    }

private:
    __aicore__ inline bool NeedBias(uint64_t kIter0, uint64_t kIter1, uint64_t kIter2)
    {
        return isBias_ && kIter0 == 0 && kIter1 == 0 && kIter2 == 0;
    }

    template <typename TensorL0CType, typename TensorAL0, typename TensorBL0, typename TensorBT>
    __aicore__ inline void Mmad(
        TileL1L0Param& tileL1L0Param, uint64_t iter0, uint64_t iter1, uint64_t kL0Iter, uint64_t curKL0,
        TensorL0CType& tensorL0C, TensorAL0 const& tensorAL0, TensorBL0 const& tensorBL0, TensorBT const& tensorBt,
        uint64_t splitKIdx)
    {
        AscendC::Te::MmadParams params;
        params.m = static_cast<uint16_t>(tileL1L0Param.curM);
        params.k = static_cast<uint16_t>(Blaze::Gemm::CeilAlign(curKL0, MXFP_DIVISOR_SIZE));
        params.n = static_cast<uint16_t>(tileL1L0Param.curN);
        params.unitFlag = (iter0 + 1 == kL1Iter_ && iter1 + 1 == kL0Iter && (splitKIdx == splitKNum_ - 1)) ?
                          FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
        params.cmatrixInitVal = (iter0 == 0 && iter1 == 0 && splitKIdx == 0 && !isBias_);
        if (NeedBias(iter0, iter1, splitKIdx)) {
            AscendC::Te::Mmad(
                AscendC::Te::MmadAtom<
                    AscendC::Te::MmadTraits<AscendC::Te::MmadOperation, Blaze::Gemm::Tile::MmadTraitMX>>{}
                    .with(params),
                tensorL0C, tensorAL0, tensorBL0, tensorBt);
        } else {
            AscendC::Te::Mmad(
                AscendC::Te::MmadAtom<
                    AscendC::Te::MmadTraits<AscendC::Te::MmadOperation, Blaze::Gemm::Tile::MmadTraitMX>>{}
                    .with(params),
                tensorL0C, tensorAL0, tensorBL0);
        }
    }

    constexpr static uint16_t SCALE_BUFFER_FLAG_0 = 4;
    constexpr static uint16_t SCALE_BUFFER_FLAG_1 = 5;
    constexpr static uint16_t M_MTE1_FLAG_0 = 4;
    constexpr static uint16_t M_MTE1_FLAG_1 = 5;

    uint16_t biasBufId_ = 0;
    uint64_t biasL1OneBuffer_ = 0UL;
    uint64_t aL1OneBuffer_ = 0UL;
    uint64_t bL1OneBuffer_ = 0UL;
    uint64_t scaleAL1OneBuffer_ = 0UL;
    uint64_t scaleBL1OneBuffer_ = 0UL;
    uint64_t l1BufferAOffset_[4] = {0UL};      // default 4 buffer
    uint64_t l1BufferBOffset_[4] = {0UL};      // default 4 buffer
    uint64_t l1BufferScaleAOffset_[2] = {0UL}; // default 2 buffer
    uint64_t l1BufferScaleBOffset_[2] = {0UL}; // default 2 buffer
    uint64_t l1BufferBiasOffset_[2] = {0UL};   // default 2 buffer
};
#endif
} // namespace Block
} // namespace Gemm
} // namespace Blaze
