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
 * \file block_mmad_a_prefetch_b_prologue.h
 * \brief
 */

#pragma once

#include "../utils/constant.h"
#include "../utils/gemm_type.h"
#include "../utils/tensor_utils.h"
#include "../policy/dispatch_policy.h"
#include "../utils/tuple_utils.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator_intf.h"
#endif
#include "../utils/underscore.h"
#include "../utils/tensor_traits.h"
#include "../utils/math_utils.h"
namespace Cmct::Gemm::Block {
using AscendC::BLOCK_CUBE;
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
using AscendC::TEventID;
using AscendC::TPosition;
using AscendC::WaitFlag;
using Cmct::Gemm::CeilAlign;
using Cmct::Gemm::CeilDiv;
using Cmct::Gemm::CreateTensor;
using Cmct::Gemm::Get;

template <TPosition Position, CubeFormat Format, typename Type, bool IsTrans = false,
          LayoutMode Layout = LayoutMode::NONE, bool IbShare = false>
struct MatmulL1GmType : AscendC::MatmulType<Position, Format, Type, IsTrans, Layout, IbShare> {
    constexpr static TPosition srcPos = TPosition::GM;
};

template <uint64_t AivNum, class L1TileShape_, class L0TileShape_, class AType_, class BType_, class CType_,
          class BiasType_, class TileCopy_, class TileMmad_>
class BlockMmad<MmadAPrefetchBAntiquantScmc<AivNum>, L1TileShape_, L0TileShape_, AType_, BType_, CType_, BiasType_,
                TileCopy_, TileMmad_> {
public:
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using AType = AType_;
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
    using L0DataType = typename AscendC::GetL0DataType<ElementA, true>::Type;

    struct Arguments {
        GM_ADDR ptrA = nullptr;
        GM_ADDR ptrC = nullptr;
        GM_ADDR ptrBias = nullptr;
        LayoutA layoutA;
        LayoutC layoutC;
        LayoutBias layoutBias;
    };

    struct Params {
        GM_ADDR ptrA = nullptr;
        GM_ADDR ptrC = nullptr;
        GM_ADDR ptrBias = nullptr;
        LayoutA layoutA;
        LayoutC layoutC;
        LayoutBias layoutBias;
        L1TileShape tileShapeL1;
        L0TileShape tileShapeL0;
        bool isBias;
        uint64_t aPreloadSize;
        uint64_t xSize;
        uint64_t kSize;
        const TCubeTiling* matmulTiling;
    };

    template <typename ProblemShape, typename Tiling>
    __aicore__ inline static Params ToUnderlyingArguments(const ProblemShape& problemShape, const Arguments& args,
                                                          Tiling const* tiling)
    {
        auto baseM = static_cast<uint32_t>(tiling->matmulTiling.baseM);
        auto baseN = static_cast<uint32_t>(tiling->matmulTiling.baseN);
        auto baseK = static_cast<uint32_t>(tiling->matmulTiling.baseK);
        auto stepKa = static_cast<uint32_t>(tiling->matmulTiling.stepKa);
        auto stepKb = static_cast<uint32_t>(tiling->matmulTiling.stepKb);
        return {.ptrA = args.ptrA,
                .ptrC = args.ptrC,
                .ptrBias = args.ptrBias,
                .layoutA = args.layoutA,
                .layoutC = args.layoutC,
                .layoutBias = args.layoutBias,
                .tileShapeL1 = AscendC::MakeShape(baseM, baseN, stepKa * baseK, stepKb * baseK),
                .tileShapeL0 = AscendC::MakeShape(baseM, baseN, baseK),
                .isBias = bool(tiling->matmulTiling.isBias),
                .aPreloadSize = tiling->aPreloadSize,
                .xSize = tiling->mSize * tiling->kSize,
                .kSize = tiling->kSize,
                .matmulTiling = &(tiling->matmulTiling)};
    }

    template <class TensorA, class TensorBias, class TensorC, class Shape>
    __aicore__ inline void operator()(const TensorA& tensorA, const TensorBias& tensorBias, const TensorC& tensorC,
                                      const Shape& actualShape, [[maybe_unused]] const Params& params)
    {
        mL1Len_ = Get<0>(actualShape);
        nL1Len_ = Get<1>(actualShape);
        for (uint64_t kLoopIdx = 0; kLoopIdx < kTileCount_; kLoopIdx++, cvLoopIdx_++) {
            uint64_t kbL1Offset = kLoopIdx * kbL1Size_;
            uint64_t kbL1RealSize = (kbL1Offset + kbL1Size_) >= kSize_ ? kSize_ - kbL1Offset : kbL1Size_;
            auto tensorATile = tensorA[tensorA.GetLayout()(AscendC::MakeCoord(_, kbL1Offset))];
            WaitMTE1ToMTE2(cvLoopIdx_);
            CopyAAndBiasGmToL1(tensorATile, tensorBias, kbL1Offset, kbL1RealSize, nL1Len_, cvLoopIdx_);
            WaitAivToAic();
            // mte1 mmad fixp流水
            LaunchMatmul(kbL1Offset, kbL1RealSize, Get<0>(tensorC.GetStride()));
            SetMTE1ToMTE2(cvLoopIdx_);
            SetAicToAiv();
        }
        GetTensorC(tensorC);
        ClearAFullLoadFlag();
    }

    __aicore__ inline BlockMmad() = delete;
    __aicore__ inline BlockMmad(const Params& params)
    {
        kSize_ = params.kSize;
        isBias_ = params.isBias;
        // (1) y 数据类型为 int8 时，quantScale需要预留一份空间
        //  ① 有bias
        //  L1 (0~512KB): WeightL1_P0(128KB) | Bias_P0(4KB) | AL1_P0(120KB) | AL1_P1(120KB) | Bias_P1(4KB) |
        //  WeightL1_P1(128KB) | quantScale(8KB) ② 无bias L1 (0~512KB): WeightL1_P0(128KB) | AL1_P0(124KB) |
        //  AL1_P1(124KB) | WeightL1_P1(128KB) | quantScale(8KB)
        // (2) L1 上有bias时:
        //  L1 (0~512KB): WeightL1_P0(128KB) | Bias_P0(4KB) | AL1_P0(124KB) | AL1_P1(124KB) | Bias_P1(4KB) |
        //  WeightL1_P1(128KB)
        // (3) 其他场景时
        //  L1 (0~512KB): WeightL1_P0(128KB) | AL1_P0(128KB) | AL1_P1(128KB) | WeightL1_P1(128KB)
        uint64_t biasL1Space = isBias_ ? 4 * KB_ELEM<ElementA> : 0; // bias单块分配4K空间
        uint64_t weightL1Space = Get<1>(params.tileShapeL1) * Get<3>(params.tileShapeL1);
        weightF16L1DbOffset_ = L1_SIZE * KB_ELEM<half> - weightL1Space;
        uint64_t aF16L1Offset = weightL1Space + biasL1Space; // A要跳过WeightL1_P0 + Bias_P0
        if (isBias_) {
            uint64_t aF16L1Space = L1_SIZE * KB_ELEM<ElementA> -
                                   DOUBLE_BUFFER_NUM * aF16L1Offset; // L1上A可占据剩余空间
            aF16L1DbOffset_ = aF16L1Space >> 1;
            if constexpr (IsSameType<ElementBias, float>::value) {
                biasL1_ = CreateTensor<BiasL1TensorTrait>((weightL1Space >> 1) * sizeof(ElementBias));
                biasL1DbOffset_ = (aF16L1Space + biasL1Space) >> 1;
            } else {
                biasL1_ = CreateTensor<BiasL1TensorTrait>(weightL1Space * sizeof(ElementBias));
                biasL1DbOffset_ = aF16L1Space + biasL1Space;
            }
        } else {
            // 256KB
            aF16L1DbOffset_ = 256 * KB_ELEM<ElementA> - weightL1Space;
        }
        // 3、kbL1
        aL1Count_ = CeilDiv(kSize_, static_cast<uint64_t>(Get<3>(params.tileShapeL1)));
        aL1MaxHalfCount_ = CeilDiv(aL1Count_, static_cast<uint64_t>(DOUBLE_BUFFER_NUM));
        aF16L1_ = CreateTensor<AL1TensorTrait>(aF16L1Offset * sizeof(ElementA));

        // 当前tiling策略的细分场景：
        // 1. stepKa <= stepKb
        // 当前限制baseM的最大值，因此该场景下在L1上A矩阵大小<=128k。可以固定走db分支，保证a矩阵的db载入
        // 2. stepKa > stepKb 当前在m小k大的情况下才会出现该场景，走全载分支
        // 3. gmm场景，不知道真实的m值，tiling采取保守策略，恒定走db分支
        // 2、kaL1,3、kbL1
        if (Get<2>(params.tileShapeL1) > Get<3>(params.tileShapeL1)) {
            al1DbNum_ = SINGLE_BUFFER_NUM;
        } else {
            al1DbNum_ = DOUBLE_BUFFER_NUM;
        }
        mmObj_.SetSubBlockIdx(0);
        mmObj_.Init(params.matmulTiling, GetTPipePtr());
        // 3 in order to obtain kb
        kTileCount_ = CeilDiv(kSize_, static_cast<uint64_t>(Get<3>(params.tileShapeL1)));
        // 3 in order to obtain kb
        kbL1Size_ = Get<3>(params.tileShapeL1);
        weightF16L1_ = CreateTensor<BL1TensorTrait>(0);
    }

    template <typename TensorA>
    __aicore__ inline static void PreloadA(const TensorA& tensorA, const Params& params)
    {
        uint64_t xOffset = AscendC::GetBlockIdx() * params.aPreloadSize;
        if (params.aPreloadSize == 0 || xOffset >= params.xSize) {
            return;
        }
        AscendC::DataCopyPadExtParams<ElementA> extParams;
        AscendC::DataCopyExtParams param;
        param.blockCount = 1;
        param.blockLen = (xOffset + params.aPreloadSize > params.xSize ? params.xSize - xOffset : params.aPreloadSize) *
                         sizeof(ElementA);
        param.srcStride = 0;
        param.dstStride = 0;

        uint64_t biasL1Space = params.isBias ? 4 * KB_ELEM<ElementA> : 0; // bias单块分配4K空间
        uint64_t weightL1Space = Get<1>(params.tileShapeL1) * Get<3>(params.tileShapeL1);
        uint64_t aF16L1Offset = weightL1Space + biasL1Space; // A要跳过WeightL1_P0 + Bias_P0
        auto aF16L1 = CreateTensor<AL1TensorTrait>(aF16L1Offset * sizeof(ElementA));
        // PS 需要api支持
        LocalTensor<ElementA> tmpAF16L1;
        tmpAF16L1.address_ = aF16L1.address_;
        GlobalTensor<ElementA> tmpTensorA;
        tmpTensorA.address_ = tensorA.address_;
        DataCopyPad(tmpAF16L1, tmpTensorA[xOffset], param, extParams);
        PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline ~BlockMmad()
    {
        // 考虑到只循环一次时， 只需要同步wait第0块缓存。 不止1次时， 2个同步块都需要wait
        if (cvLoopIdx_ > 1 && al1DbNum_ > SINGLE_BUFFER_NUM) {
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MTE1_TO_MTE2[cvLoopIdx_ & 1]);
        }

        if (cvLoopIdx_ > 0 && al1DbNum_ > SINGLE_BUFFER_NUM) {
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MTE1_TO_MTE2[(cvLoopIdx_ + 1) & 1]);
        }
    }

private:
    __aicore__ inline void SetMTE1ToMTE2(uint64_t cvLoopIdx)
    {
        if (al1DbNum_ > SINGLE_BUFFER_NUM) {
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MTE1_TO_MTE2[cvLoopIdx_ & 1]);
        }
    }

    __aicore__ inline uint64_t CheckMaxSpace()
    {
        uint64_t maxSpace = aL1MaxHalfCount_ * kbL1Size_ * CeilAlign(mL1Len_, static_cast<uint64_t>(BLOCK_CUBE));
        if (kbL1Size_ > 0 && kSize_ % kbL1Size_ == 0 && !A_TRANS && maxSpace <= aF16L1DbOffset_) {
            return maxSpace;
        }
        return 0;
    }

    template <typename TensorA, typename TensorBias>
    __aicore__ inline void CopyAAndBiasGmToL1(const TensorA& tensorA, const TensorBias& tensorBias, int64_t kaGmOffset,
                                              int64_t kbL1RealSize, int64_t biasRealN, uint64_t cvLoopIdx)
    {
        if (al1DbNum_ > SINGLE_BUFFER_NUM) {
            AscendC::Nd2NzParams nd2nzParams;
            nd2nzParams.ndNum = 1;
            if constexpr (A_TRANS) {
                // k, m
                nd2nzParams.nValue = kbL1RealSize;
                nd2nzParams.dValue = mL1Len_;
                nd2nzParams.srcDValue = Get<1>(tensorA.GetStride());
            } else {
                // m, k
                nd2nzParams.nValue = mL1Len_;
                nd2nzParams.dValue = kbL1RealSize;
                nd2nzParams.srcDValue = Get<0>(tensorA.GetStride());
            }
            nd2nzParams.srcNdMatrixStride = 0;
            nd2nzParams.dstNzC0Stride = CeilAlign(nd2nzParams.nValue, static_cast<uint16_t>(BLOCK_CUBE));
            nd2nzParams.dstNzNStride = 1;
            nd2nzParams.dstNzMatrixStride = 0;

            LocalTensor<ElementA> tmpAF16L1;
            tmpAF16L1.address_ = aF16L1_[(cvLoopIdx_ & 1) * aF16L1DbOffset_].address_;
            GlobalTensor<ElementA> tmpTensorA;
            tmpTensorA.address_ = tensorA.address_;
            DataCopy(tmpAF16L1, tmpTensorA, nd2nzParams);
        } else if (al1DbNum_ == SINGLE_BUFFER_NUM && kaGmOffset == 0) {
            CopyAGmToL1SingleBuffer(tensorA);
        }

        // bias仅与n有关，与k无关，所以只需要拷贝一次
        if (isBias_ && kaGmOffset == 0) {
            AscendC::DataCopyPadExtParams<ElementBias> extParams;
            AscendC::DataCopyExtParams param;
            param.blockCount = 1;
            param.blockLen = biasRealN * sizeof(ElementBias);
            param.srcStride = 0;
            param.dstStride = 0;
            LocalTensor<ElementBias> tmpBiasL1;
            tmpBiasL1.address_ = biasL1_[(cvLoopIdx_ & 1) * biasL1DbOffset_].address_;
            GlobalTensor<ElementBias> tmpTensorBias;
            tmpTensorBias.address_ = tensorBias.address_;
            DataCopyPad(tmpBiasL1, tmpTensorBias, param, extParams);
        }
        SetFlag<HardEvent::MTE2_MTE1>(0);
        WaitFlag<HardEvent::MTE2_MTE1>(0);
    }

    template <typename TensorA>
    __aicore__ inline void CopyAGmToL1SingleBuffer(const TensorA& tensorA)
    {
        AscendC::Nd2NzParams nd2nzParams;
        uint64_t maxSpace = CheckMaxSpace();

        if (maxSpace > 0) {
            nd2nzParams.ndNum = aL1MaxHalfCount_;
            nd2nzParams.nValue = mL1Len_;
            nd2nzParams.dValue = kbL1Size_;
            nd2nzParams.srcDValue = kSize_;
            nd2nzParams.srcNdMatrixStride = 2 * nd2nzParams.dValue;
            nd2nzParams.dstNzC0Stride = CeilAlign(nd2nzParams.nValue, static_cast<uint16_t>(BLOCK_CUBE));
            nd2nzParams.dstNzNStride = 1;
            nd2nzParams.dstNzMatrixStride = nd2nzParams.dstNzC0Stride *
                                            CeilAlign(nd2nzParams.dValue, static_cast<uint32_t>(BLOCK_CUBE));

            LocalTensor<ElementA> tmpAF16L1;
            tmpAF16L1.address_ = aF16L1_.address_;
            GlobalTensor<ElementA> tmpTensorA;
            tmpTensorA.address_ = tensorA.address_;
            DataCopy(tmpAF16L1[(cvLoopIdx_ & 1) * aF16L1DbOffset_], tmpTensorA, nd2nzParams);
            nd2nzParams.ndNum = aL1Count_ - aL1MaxHalfCount_;
            DataCopy(tmpAF16L1[((cvLoopIdx_ + 1) & 1) * aF16L1DbOffset_], tmpTensorA[nd2nzParams.dValue], nd2nzParams);
        } else {
            nd2nzParams.ndNum = 1;
            if constexpr (A_TRANS) {
                // k, m
                nd2nzParams.nValue = kSize_;
                nd2nzParams.dValue = mL1Len_;
                nd2nzParams.srcDValue = Get<1>(tensorA.GetStride());
            } else {
                // m, k
                nd2nzParams.nValue = mL1Len_;
                nd2nzParams.dValue = kSize_;
                nd2nzParams.srcDValue = Get<0>(tensorA.GetStride());
            }
            nd2nzParams.srcNdMatrixStride = 0;
            nd2nzParams.dstNzC0Stride = CeilAlign(nd2nzParams.nValue, static_cast<uint16_t>(BLOCK_CUBE));
            nd2nzParams.dstNzNStride = 1;
            nd2nzParams.dstNzMatrixStride = 0;

            LocalTensor<ElementA> tmpAF16L1;
            tmpAF16L1.address_ = aF16L1_.address_;
            GlobalTensor<ElementA> tmpTensorA;
            tmpTensorA.address_ = tensorA.address_;
            DataCopy(tmpAF16L1, tmpTensorA, nd2nzParams);
        }
    }

    __aicore__ inline void LaunchMatmul(int64_t kbOffset, uint64_t kbL1RealSize, uint64_t nSize)
    {
        mmObj_.SetOrgShape(CeilAlign(mL1Len_, static_cast<uint64_t>(BLOCK_CUBE)),
                           CeilAlign(nL1Len_, static_cast<uint64_t>(BLOCK_CUBE)),
                           CeilAlign(kbL1RealSize, static_cast<uint64_t>(BLOCK_CUBE)),
                           CeilAlign(kbL1RealSize, static_cast<uint64_t>(BLOCK_CUBE)), nSize);
        LocalTensor<ElementA> tmpAF16L1;
        tmpAF16L1.address_ = aF16L1_.address_;
        if (al1DbNum_ == SINGLE_BUFFER_NUM) {
            uint64_t maxSpace = CheckMaxSpace();
            if (maxSpace > 0) {
                // block = kbOffset / kbL1Size_ 计算是在第几块
                // blockOffset = block / 2 确定从A0还是A1读取数据后，在块内的偏移，单位是块
                // k = blockOffset * kbL1Size_ 当前块内的偏移量kOffset，单位是元素
                // 将block和blockOffset带入，计算k
                // k = (kbOffset / kbL1Size_) / 2 * kbL1Size_
                // 块内偏移量 = m * k
                // 举例：
                // L1A: |0|2|4|      |1|3|
                //      |A0:0~128KB  |A1:128KB~256KB|
                // 第5块（block = 5），在A1中偏移为3（blockOffset = 3），块内偏移量为m * (3 * k)
                mmObj_.SetTensorA(tmpAF16L1[(cvLoopIdx_ & 1) * aF16L1DbOffset_ +
                                            CeilAlign(mL1Len_, static_cast<uint64_t>(BLOCK_CUBE)) *
                                                // 2块block
                                                (static_cast<uint64_t>(kbOffset) / (kbL1Size_ * 2) * kbL1Size_)],
                                  A_TRANS);
            } else {
                mmObj_.SetTensorA(tmpAF16L1[CeilAlign(mL1Len_, static_cast<uint64_t>(BLOCK_CUBE)) * kbOffset], A_TRANS);
            }
        } else {
            mmObj_.SetTensorA(tmpAF16L1[(cvLoopIdx_ & 1) * aF16L1DbOffset_], A_TRANS);
        }

        LocalTensor<ElementA> tmpWeightF16L1;
        tmpWeightF16L1.address_ = weightF16L1_.address_;
        mmObj_.SetTensorB(tmpWeightF16L1[(cvLoopIdx_ & 1) * weightF16L1DbOffset_], B_TRANS);

        if (isBias_) {
            LocalTensor<ElementBias> tmpBiasL1;
            tmpBiasL1.address_ = biasL1_.address_;
            mmObj_.SetBias(tmpBiasL1[(cvLoopIdx_ & 1) * biasL1DbOffset_]);
        }

        mmObj_.SetTail(mL1Len_, nL1Len_, kbL1RealSize);
        mmObj_.Iterate(kbOffset != 0);
    }

    __aicore__ inline void WaitAivToAic()
    {
        if constexpr (AIV_NUM == 2) {
            CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
        }
        CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
    }

    __aicore__ inline void SetAicToAiv()
    {
        if constexpr (AIV_NUM == 2) {
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG + FLAG_ID_MAX);
        }
        CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
    }

    __aicore__ inline void ClearAFullLoadFlag()
    {
        if (al1DbNum_ == SINGLE_BUFFER_NUM) {
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MTE1_TO_MTE2[0]);
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MTE1_TO_MTE2[0]);
        }
    }

    template <class TensorC>
    __aicore__ inline void GetTensorC(const TensorC& tensorC)
    {
        GlobalTensor<ElementC> tmpTensorC;
        tmpTensorC.address_ = tensorC.address_;
        mmObj_.GetTensorC(tmpTensorC);
    }

    __aicore__ inline void WaitMTE1ToMTE2(uint64_t cvLoopIdx)
    {
        // 单buffer时保证了A一次全载不需要Wait，Double buffer时首次使用不需要Wait
        if (al1DbNum_ > SINGLE_BUFFER_NUM && cvLoopIdx_ >= DOUBLE_BUFFER_NUM) {
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MTE1_TO_MTE2[cvLoopIdx_ & 1]);
        }
    }

    static constexpr uint8_t AIV_NUM = AivNum;
    static constexpr int64_t L1_SIZE = 512;
    static constexpr uint64_t SINGLE_BUFFER_NUM = 1;
    static constexpr uint64_t DOUBLE_BUFFER_NUM = 2;
    static constexpr uint64_t SYNC_AIV_AIC_FLAG = 8;
    static constexpr uint64_t SYNC_AIC_AIV_FLAG = 9;
    static constexpr AscendC::TEventID EVENT_IDS_MTE1_TO_MTE2[DOUBLE_BUFFER_NUM] = {0, 1};
    static constexpr bool A_TRANS = IsColumnMajor2D<LayoutA>::value;
    static constexpr bool B_TRANS = IsRowMajor2D<LayoutB>::value || IsNz2D<LayoutB>::value;

    int8_t al1DbNum_;
    uint64_t mL1Len_;
    uint64_t nL1Len_;
    uint64_t kSize_;
    uint64_t kbL1Size_;
    uint64_t kTileCount_;
    uint64_t aF16L1DbOffset_;
    uint64_t biasL1DbOffset_;
    uint64_t aL1Count_;
    uint64_t aL1MaxHalfCount_;
    uint64_t weightF16L1DbOffset_;
    bool isBias_;
    uint64_t cvLoopIdx_ = 0;

    using MatmulImplType = AscendC::MatmulImpl<MatmulL1GmType<TPosition::TSCM, CubeFormat::NZ, ElementA, A_TRANS>,
                                               MatmulL1GmType<TPosition::TSCM, CubeFormat::NZ, ElementA, B_TRANS>,
                                               AscendC::MatmulType<TPosition::GM, CubeFormat::ND, ElementC>,
                                               AscendC::MatmulType<TPosition::TSCM, CubeFormat::ND, ElementBias>,
                                               CFG_MDL>;
    MatmulImplType mmObj_;
    // (n1,k1,k0,n0)
    using BL1TensorTrait = typename TensorTraitL1<IsZn2D<LayoutB>::value, ElementA, AscendC::TPosition::B1>::Type;
    AscendC::LocalTensor<BL1TensorTrait> weightF16L1_;
    using AL1TensorTrait = typename TensorTraitL1<!A_TRANS, ElementA, AscendC::TPosition::A1>::Type;
    LocalTensor<AL1TensorTrait> aF16L1_;
    using BiasL1TensorTrait = AscendC::TensorTrait<
        ElementBias, AscendC::TPosition::C1,
        AscendC::Layout<AscendC::Shape<uint64_t>, AscendC::Stride<Cmct::Gemm::_1>>>;
    AscendC::LocalTensor<BiasL1TensorTrait> biasL1_;
};
} // namespace Cmct::Gemm::Block
