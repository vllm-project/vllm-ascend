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
 * \file block_matmul_pingpong_without_que.h
 * \brief
 */

#pragma once

#include "../../../inc/macro.h"
#include "./block_mmad.h"
#include "../utils/common_utils.h"
#include "../utils/layout_utils.h"
#include "../utils/tuple_utils.h"
#include "../policy/dispatch_policy.h"

namespace Cmct {
namespace Gemm {
namespace Block {

template <class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class AType_, class BType_, class CType_,
          class BiasType_, class TileCopy_>
class BlockMmad<
    DispatchPolicy_, L1TileShape_, L0TileShape_, AType_, BType_, CType_, BiasType_, TileCopy_,
    AscendC::Std::enable_if_t<
        AscendC::Std::is_base_of_v<MatmulMultiBlockWithOutQue<>, DispatchPolicy_> ||
        AscendC::Std::is_base_of_v<MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, B_FULL_LOAD_MODE>,
                                   DispatchPolicy_> ||
        AscendC::Std::is_base_of_v<MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, A_FULL_LOAD_MODE>,
                                   DispatchPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, NONE_FULL_LOAD_MODE, OP_TYPE_ADD>,
            DispatchPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, A_FULL_LOAD_MODE, OP_TYPE_ADD>,
            DispatchPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, B_FULL_LOAD_MODE, OP_TYPE_ADD>,
            DispatchPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, NONE_FULL_LOAD_MODE, OP_TYPE_MUL>,
            DispatchPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, A_FULL_LOAD_MODE, OP_TYPE_MUL>,
            DispatchPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, B_FULL_LOAD_MODE, OP_TYPE_MUL>,
            DispatchPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, NONE_FULL_LOAD_MODE, OP_TYPE_RELU>,
            DispatchPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, B_FULL_LOAD_MODE, OP_TYPE_RELU>,
            DispatchPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, A_FULL_LOAD_MODE, OP_TYPE_RELU>,
            DispatchPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, NONE_FULL_LOAD_MODE, OP_TYPE_QUANT>,
            DispatchPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, A_FULL_LOAD_MODE, OP_TYPE_QUANT>,
            DispatchPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, B_FULL_LOAD_MODE, OP_TYPE_QUANT>,
            DispatchPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, NONE_FULL_LOAD_MODE, OP_TYPE_RELU_QUANT>,
            DispatchPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, A_FULL_LOAD_MODE, OP_TYPE_RELU_QUANT>,
            DispatchPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, B_FULL_LOAD_MODE, OP_TYPE_RELU_QUANT>,
            DispatchPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, NONE_FULL_LOAD_MODE, OP_TYPE_GELU_ERF>,
            DispatchPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, NONE_FULL_LOAD_MODE, OP_TYPE_GELU_TANH>,
            DispatchPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, NONE_FULL_LOAD_MODE, OP_TYPE_SILU>,
            DispatchPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, NONE_FULL_LOAD_MODE, OP_TYPE_SWIGLU_SINGLE>,
            DispatchPolicy_>>> {
public:
    using L0cType = typename GetL0CAndBtType::Type;
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = BiasType_;
    using A_T = typename AType::T;
    using B_T = typename BType::T;
    using C_T = typename CType::T;
    using Bias_T = typename BiasType::T;
    using DispatchPolicy = DispatchPolicy_;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using TupleL1L0Shape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
    uint64_t blkK_{1};
    uint64_t kAlign_{1};
    uint64_t l1BufNum_{1};
    uint64_t kL1Iter_{0};
    uint64_t mL1_{1};
    uint64_t nL1_{1};
    uint64_t kL1_{1};
    bool isSplitSingleK_{false};
    bool isFirstSplitK_{false};
    bool isEndSplitK_{false};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    // swiglu single padded-half 尾块：>0 时表示当前 tile 是尾块，值为原始半宽
    // （如 120），mmad/UB 帧宽已按 Align(原始宽,32) 补齐，B/bias 按此原始半宽装载。
    uint64_t swigluTailHalfRaw_{0};
    __aicore__ inline void SetSwigluTailHalfRaw(uint64_t v) { swigluTailHalfRaw_ = v; }
    bool isBias_{false};
    uint64_t sliceM_{1};
    uint64_t srcNdStride_{1};
    int64_t innerBatch_{1};
    constexpr static uint64_t BUFFER_NUM = 2;
    constexpr static uint64_t SPLIT_M_ALIGN = 2;
    constexpr static uint64_t HALF_L0_SIZE = L0A_SIZE / DOUBLE_BUFFER_COUNT / sizeof(A_T);
    constexpr static uint64_t HALF_L0C_SIZE = AscendC::TOTAL_L0C_SIZE / DOUBLE_BUFFER_COUNT / sizeof(L0cType);
    // C0_SIZE equals 8 in order to adapt to the fp32 matrix
    constexpr static int32_t C0_SIZE = AscendC::AuxGetC0Size<typename AType::T>();
    constexpr static int32_t BIAS_C0 = AscendC::AuxGetC0Size<typename BiasType::T>();
    constexpr static uint64_t halfL0Size_ = L0AUF_SIZE / BUFFER_NUM / sizeof(A_T);
    constexpr static uint64_t HALF_L1_SIZE = AscendC::TOTAL_L1_SIZE / DOUBLE_BUFFER_COUNT / sizeof(A_T);
    // Set unitflag state: 3 = final accumulation, 2 = non-final accumulation
    constexpr static uint32_t FINAL_ACCUMULATION = 3;
    constexpr static uint32_t NON_FINAL_ACCUMULATION = 2;
    constexpr static uint64_t BLOCK_BYTE_SIZE = 32UL;
    constexpr static uint64_t N_TAIL_ALIGN_THRESHOLD = 8UL;
    uint64_t abL1LoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    bool enableL0cPingPong_{false};
    bool splitM_{false};
    uint64_t fullLoadMode_{0};
#if __NPU_ARCH__ == 5102
    uint8_t shiftValue_{42};
    constexpr static uint8_t FIX_SHIFT_VAL_LEN_A16W16 = 58;
#endif
    uint64_t quantScalar_{0};

    __aicore__ inline BlockMmad()
    {
        // ASCEND_IS_NOT_AIV 等价于 (分离架构ASCEND_IS_AIC OR 耦合架构)
        if ASCEND_IS_NOT_AIV {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(FIRST_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SECOND_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(THIRD_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FIRST_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(SIXTH_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(SEVENTH_FLAG);
        }
    }

    __aicore__ inline ~BlockMmad()
    {
        // ASCEND_IS_NOT_AIV 等价于 (分离架构ASCEND_IS_AIC OR 耦合架构)
        if ASCEND_IS_NOT_AIV {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(FIRST_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SECOND_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(THIRD_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FIRST_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(SIXTH_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(SEVENTH_FLAG);
        }
    }

public:
    template <uint64_t FULL_LOAD_MODE_ = B_FULL_LOAD_MODE>
    __aicore__ inline void Init(const TupleShape& shape, const TupleShape& tileL1, const TupleShape& tileL0,
                                bool isBias, uint64_t l1BufNum, bool l0cDB,
                                const AscendC::Shape<int64_t, int64_t, int64_t>& nonContinuousParam,
                                bool isSplitSingleK = false, uint8_t shiftValue = 42)
    {
        m_ = Get<DIMENSION_M>(shape);
        n_ = Get<DIMENSION_N>(shape);
        k_ = Get<DIMENSION_K>(shape);
        blkK_ = Get<DIMENSION_K>(shape);
        mL1_ = Get<DIMENSION_M>(tileL1);
        nL1_ = Get<DIMENSION_N>(tileL1);
        kL1_ = Get<DIMENSION_K>(tileL1);
        isSplitSingleK_ = isSplitSingleK;
        baseM_ = Get<DIMENSION_M>(tileL0);
        baseN_ = Get<DIMENSION_N>(tileL0);
        baseK_ = Get<DIMENSION_K>(tileL0);
        kAlign_ = Cmct::Gemm::Align(blkK_, AscendC::BLOCK_CUBE);
        isBias_ = isBias;
        l1BufNum_ = l1BufNum;
        enableL0cPingPong_ = l0cDB;
#if __NPU_ARCH__ == 5102
        shiftValue_ = shiftValue;
#endif
        // init tensor
        if constexpr (FULL_LOAD_MODE_ == A_FULL_LOAD_MODE) {
            // A全载
            aL1OneBuffer_ = mL1_ * kAlign_;
        } else {
            // 非全载和B全载
            aL1OneBuffer_ = mL1_ * kL1_;
        }
        // 当前B全载后续未用到bL1OneBuffer_
        if constexpr (FULL_LOAD_MODE_ == B_FULL_LOAD_MODE) {
            bL1OneBuffer_ = nL1_ * kAlign_;
        } else {
            bL1OneBuffer_ = nL1_ * kL1_;
        }
        fullLoadMode_ = FULL_LOAD_MODE_;
        kL1Iter_ = CeilDiv(blkK_, kL1_);
        l0PingPong_ = 0;
        abL1LoopCnt_ = 0;
        l0cPingPong_ = 0;
        sliceM_ = Get<0>(nonContinuousParam);
        srcNdStride_ = Get<1>(nonContinuousParam);
        innerBatch_ = Get<2>(nonContinuousParam);
    }

    __aicore__ inline void SetDualParam(bool splitM) { splitM_ = splitM; }

    __aicore__ inline void CacheQuantScalar(uint64_t quantScalar) { quantScalar_ = quantScalar; }

    // For FP32: L1 copy needs no modification
    __aicore__ inline void CopyInA1(const AscendC::GlobalTensor<A_T>& aGlobal,
                                    const AscendC::LocalTensor<A_T>& al1Local, uint64_t curML1, uint64_t curKL1)
    {
        AscendC::Nd2NzParams nd2nzParams;
        if (srcNdStride_ != 1 && sliceM_ != 0) { // For Slice
            nd2nzParams.ndNum = curML1 / sliceM_;
            uint64_t nDim = sliceM_;
            uint64_t dDim = curKL1;
            nd2nzParams.nValue = nDim;
            nd2nzParams.dValue = dDim;
            nd2nzParams.srcNdMatrixStride = srcNdStride_;
            nd2nzParams.srcDValue = k_;
            nd2nzParams.dstNzC0Stride = Cmct::Gemm::Align(curML1, AscendC::BLOCK_CUBE);
            nd2nzParams.dstNzNStride = 1;
            nd2nzParams.dstNzMatrixStride = sliceM_ * C0_SIZE;
        } else if (innerBatch_ > 1) {
            nd2nzParams.ndNum = 1;
            uint64_t nDim = AType::isTrans ? curKL1 : curML1;
            uint64_t dDim = AType::isTrans ? curML1 : curKL1;
            nd2nzParams.nValue = nDim;
            nd2nzParams.dValue = dDim;
            nd2nzParams.srcNdMatrixStride = 1;
            nd2nzParams.srcDValue = innerBatch_ * (AType::isTrans ? m_ : k_);
            nd2nzParams.dstNzC0Stride = Cmct::Gemm::Align(nDim, AscendC::BLOCK_CUBE);
            nd2nzParams.dstNzNStride = 1;
            nd2nzParams.dstNzMatrixStride = 1;
        } else {
            nd2nzParams.ndNum = 1;
            uint64_t nDim = AType::isTrans ? curKL1 : curML1;
            uint64_t dDim = AType::isTrans ? curML1 : curKL1;
            nd2nzParams.nValue = nDim;
            nd2nzParams.dValue = dDim;
            nd2nzParams.srcNdMatrixStride = 1;
            nd2nzParams.srcDValue = AType::isTrans ? m_ : k_;
            nd2nzParams.dstNzC0Stride = Cmct::Gemm::Align(nDim, AscendC::BLOCK_CUBE);
            nd2nzParams.dstNzNStride = 1;
            nd2nzParams.dstNzMatrixStride = 1;
        }

        AscendC::DataCopy(al1Local, aGlobal, nd2nzParams);
    }

    template <CubeFormat LayoutB = CubeFormat::ND>
    __aicore__ inline void CopyInB1(const AscendC::GlobalTensor<B_T>& bGlobal,
                                    const AscendC::LocalTensor<B_T>& bl1Local, uint64_t curNL1, uint64_t curKL1)
    {
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        uint64_t nDim = BType::isTrans ? curNL1 : curKL1;
        uint64_t dDim = BType::isTrans ? curKL1 : curNL1;
        if constexpr (LayoutB == CubeFormat::ND) {
            if constexpr (DispatchPolicy::enableSwigluSingle) {
                // swiglu 单 matmul：半 tile 布局——gate 半块（curNL1/2 行）进 tile 前半，
                // up 半块进 tile 后半，使 L0C 一行 = [g(连续) | u(连续)]，epilogue 免 gather。
                // 调用点已把 bGlobal 指向本 tile 的 gate 半块起点（offsetB/2），
                // up 半块在 (n_/2)*k_ 元素之后。要求 L0 tile N == L1 tile N（host 保证）。
                static_assert(BType::isTrans, "swiglu single requires transB (linear [N,K] weight)");
                static_assert(sizeof(B_T) == 2, "swiglu single interleave assumes 16B C0 (bf16/fp16)");
                ASCENDC_ASSERT((curNL1 % 32) == 0, {
                    KERNEL_LOG(KERNEL_ERROR, , "swiglu single requires 32-aligned tile N");
                });
                AscendC::Nd2NzParams interleaveParams;
                interleaveParams.ndNum = 1;
                // 尾块（padded-half）：半块原始宽度 < curNL1/2，Nd2Nz 右补零到 C0 块；
                // 整块时原始半宽 == curNL1/2，行为不变。
                const uint64_t halfW = swigluTailHalfRaw_ > 0 ? swigluTailHalfRaw_ : curNL1 / 2;
                interleaveParams.nValue = static_cast<uint16_t>(halfW);
                interleaveParams.dValue = static_cast<uint32_t>(curKL1);
                interleaveParams.srcNdMatrixStride = 0;
                interleaveParams.srcDValue = k_;
                interleaveParams.dstNzC0Stride =
                    static_cast<uint16_t>(Cmct::Gemm::Align(curNL1, AscendC::BLOCK_CUBE));
                interleaveParams.dstNzNStride = 1;
                interleaveParams.dstNzMatrixStride = 0;
                // gate -> tile 前半；up -> tile 后半（行 curNL1/2 起，每行 16 元素/C0 块）
                AscendC::DataCopy(bl1Local, bGlobal, interleaveParams);
                AscendC::DataCopy(bl1Local[(curNL1 / 2) * 16], bGlobal[(n_ / 2) * k_], interleaveParams);
            } else {
                if (innerBatch_ > 1) {
                    nd2nzParams.nValue = nDim;
                    nd2nzParams.dValue = dDim;
                    nd2nzParams.srcNdMatrixStride = 1;
                    nd2nzParams.srcDValue = innerBatch_ * (BType::isTrans ? k_ : n_);
                    nd2nzParams.dstNzC0Stride = Cmct::Gemm::Align(nDim, AscendC::BLOCK_CUBE);
                    nd2nzParams.dstNzNStride = 1;
                    nd2nzParams.dstNzMatrixStride = 1;
                } else {
                    nd2nzParams.nValue = nDim;
                    nd2nzParams.dValue = dDim;
                    nd2nzParams.srcNdMatrixStride = 1;
                    nd2nzParams.srcDValue = BType::isTrans ? k_ : n_;
                    nd2nzParams.dstNzC0Stride = Cmct::Gemm::Align(nDim, AscendC::BLOCK_CUBE);
                    nd2nzParams.dstNzNStride = 1;
                    nd2nzParams.dstNzMatrixStride = 1;
                }
                AscendC::DataCopy(bl1Local, bGlobal, nd2nzParams);
            }
        } else {
            AscendC::DataCopyExtParams dataCopyParams;
            uint64_t nDim = BType::isTrans ? curNL1 : curKL1;
            uint64_t dDim = BType::isTrans ? curKL1 : curNL1;
            uint64_t nkDim = BType::isTrans ? n_ : k_;
            dataCopyParams.blockCount = Cmct::Gemm::CeilDiv(Cmct::Gemm::CeilAlign(dDim, C0_SIZE), C0_SIZE);
            dataCopyParams.blockLen = Cmct::Gemm::CeilAlign(nDim, AscendC::BLOCK_CUBE) * BLOCK_BYTE_SIZE;
            dataCopyParams.srcStride = (Cmct::Gemm::CeilAlign(nkDim, AscendC::BLOCK_CUBE) -
                                        Cmct::Gemm::CeilAlign(nDim, AscendC::BLOCK_CUBE)) *
                                       BLOCK_BYTE_SIZE;
            dataCopyParams.dstStride = 0;
            AscendC::DataCopyPadExtParams<B_T> padParams{false, 0, 0, 0};
            AscendC::DataCopyPad(bl1Local, bGlobal, dataCopyParams, padParams);
        }
    }

    // 重载函数，适用于A全载场景
    __aicore__ inline void CopyInA1(const AscendC::GlobalTensor<B_T>& aGlobal, uint64_t curML1, uint64_t curKL1)
    {
        // A全载-AL1搬入偏移位置：*AL1*-BL1Ping-BL1Pong-BiasPing-BiasPong
        CopyInA1(aGlobal, l1Local_, curML1, curKL1);
    }

    // 重载函数，适用于B全载场景
    template <CubeFormat LayoutB = CubeFormat::ND>
    __aicore__ inline void CopyInB1(const AscendC::GlobalTensor<B_T>& bGlobal, uint64_t curNL1, uint64_t curKL1)
    {
        // B全载-BL1搬入偏移位置：AL1Ping-AL1Pong-*BL1*-Bias
        CopyInB1<LayoutB>(bGlobal, l1Local_[aL1OneBuffer_ * l1BufNum_], curNL1, curKL1);
    }

    __aicore__ inline void CopyInC1(const AscendC::GlobalTensor<Bias_T>& biasGlobal, uint64_t curNL1)
    {
        if (isBias_) {
            // B全载-Bias搬入偏移位置：AL1Ping-AL1Pong-BL1-*Bias*
            AscendC::LocalTensor<Bias_T> biasL1Local = l1Local_[aL1OneBuffer_ * l1BufNum_ + bL1OneBuffer_]
                                                           .template ReinterpretCast<Bias_T>();
            CopyInC1(biasGlobal, biasL1Local, curNL1);
        }
    }

    __aicore__ inline void CopyInC1(const AscendC::GlobalTensor<Bias_T>& biasGlobal,
                                    const AscendC::LocalTensor<Bias_T>& cl1Local, uint64_t curNL1)
    {
        if constexpr (DispatchPolicy::enableSwigluSingle) {
            // swiglu 单 matmul：bias1=[gate_bias(H)|up_bias(H)]，调用点已传 offsetBias/2 指向
            // gate 半块起点。半 tile 布局：gate bias 进前半、up bias 进后半（各连续，免 stride）。
            ASCENDC_ASSERT((curNL1 % 32) == 0, {
                KERNEL_LOG(KERNEL_ERROR, , "swiglu single requires 32-aligned tile N");
            });
            const uint64_t halfW = swigluTailHalfRaw_ > 0 ? swigluTailHalfRaw_ : curNL1 / 2;
            const uint64_t halfPadded = curNL1 / 2;
            // rightPadding 补到 32B 对齐：DataCopyPad 的 rightPadding 上限 32B，
            // 超出会触发 aicore 507015。剩余 pad 区域保留 L1 旧值——pad 列仅进入
            // 补齐帧，epilogue 只写有效半宽，逐列运算不跨界，旧值不影响输出。
            const uint64_t alignElems = 32 / sizeof(Bias_T); // fp32: 8, bf16: 16
            const uint64_t rp = (alignElems - halfW % alignElems) % alignElems;
            AscendC::DataCopyPadParams padParams;
            padParams.isPad = (rp > 0);
            padParams.rightPadding = static_cast<uint8_t>(rp);
            AscendC::DataCopyParams biasParam{1, static_cast<uint16_t>(halfW * sizeof(Bias_T)), 0, 0};
            AscendC::DataCopyPad(cl1Local, biasGlobal, biasParam, padParams);                          // gate bias
            AscendC::DataCopyPad(cl1Local[curNL1 / 2], biasGlobal[n_ / 2], biasParam, padParams);      // up bias
        } else {
            AscendC::DataCopyPadParams padParams;
            // 单位为Byte
            AscendC::DataCopyParams biasParam{1, static_cast<uint16_t>(curNL1 * sizeof(Bias_T)), 0, 0};
            AscendC::DataCopyPad(cl1Local, biasGlobal, biasParam, padParams);
        }
    }

    __aicore__ inline void CopyInC2(const AscendC::LocalTensor<Bias_T>& biasL1Local,
                                    const AscendC::LocalTensor<L0cType>& biasBt, uint64_t nl1Align, bool needBias)
    {
        if (!needBias) {
            return;
        }
        // s32场景要对齐到2 因此是align(nl1Align / 8, 2)
        uint64_t btAlign = AscendC::BLOCK_CUBE / BIAS_C0;
        uint16_t bustLenth = Cmct::Gemm::Align(nl1Align / BIAS_C0, btAlign);
        AscendC::DataCopyParams biasParam{1, static_cast<uint16_t>(bustLenth), 0, 0};
#if __NPU_ARCH__ == 5102
        biasParam.fixShiftVal = FIX_SHIFT_VAL_LEN_A16W16 - shiftValue_;
#endif
        // 当dstlocal位于C2时，C2中至少为fp32*16
        AscendC::DataCopy(biasBt, biasL1Local, biasParam);
    }

    __aicore__ inline void CopyInA2(const AscendC::LocalTensor<A_T>& a2Local, const AscendC::LocalTensor<A_T>& al1Local,
                                    uint64_t curML1, uint64_t curKL1, uint64_t mL0, uint64_t kL0)
    {
        if constexpr (!AType::isTrans) {
            // (M, K) use LoadData2D
            AscendC::LoadData2DParamsV2 loadDataParams;
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = 0;
            loadDataParams.mStep = Cmct::Gemm::CeilDiv(mL0, AscendC::BLOCK_CUBE);
            if constexpr (AscendC::IsSameType<A_T, half>::value || AscendC::IsSameType<A_T, bfloat16_t>::value) {
                loadDataParams.kStep = Cmct::Gemm::CeilDiv(kL0, AscendC::BLOCK_CUBE);
            } else {
                loadDataParams.kStep = Cmct::Gemm::CeilDiv(kL0, C0_SIZE);
            }
            loadDataParams.srcStride = Cmct::Gemm::CeilDiv(curML1, AscendC::BLOCK_CUBE);
            loadDataParams.dstStride = loadDataParams.mStep;
            loadDataParams.ifTranspose = false;
            AscendC::LoadData<A_T>(a2Local, al1Local, loadDataParams);
        } else {
            // (K, M)
            AscendC::LoadData2DParamsV2 loadDataParams;
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = 0;
            loadDataParams.mStep = Cmct::Gemm::CeilDiv(kL0, AscendC::BLOCK_CUBE);
            if constexpr (AscendC::IsSameType<A_T, half>::value || AscendC::IsSameType<A_T, bfloat16_t>::value) {
                loadDataParams.kStep = Cmct::Gemm::CeilDiv(mL0, AscendC::BLOCK_CUBE);
                loadDataParams.dstStride = loadDataParams.kStep;
            } else {
                // actually div 8 then align to 2
                loadDataParams.kStep = Cmct::Gemm::CeilDiv(mL0, AscendC::BLOCK_CUBE) * TWO_ALIGN;
                loadDataParams.dstStride = loadDataParams.kStep >> 1;
            }
            loadDataParams.srcStride = Cmct::Gemm::CeilDiv(curKL1, AscendC::BLOCK_CUBE);
            loadDataParams.ifTranspose = true;
            AscendC::LoadData<A_T>(a2Local, al1Local, loadDataParams);
        }
    }

    __aicore__ inline void CopyInB2(const AscendC::LocalTensor<B_T>& b2Local, const AscendC::LocalTensor<B_T>& bl1Local,
                                    uint64_t curNL1, uint64_t curKL1, uint64_t nL0, uint64_t kL0)
    {
        if constexpr (BType::isTrans) {
            // (N, K) use LoadData2D
            AscendC::LoadData2DParamsV2 loadDataParams;
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = 0;
            loadDataParams.mStep = Cmct::Gemm::CeilDiv(nL0, AscendC::BLOCK_CUBE);
            if constexpr (AscendC::IsSameType<B_T, half>::value || AscendC::IsSameType<B_T, bfloat16_t>::value) {
                loadDataParams.kStep = Cmct::Gemm::CeilDiv(kL0, AscendC::BLOCK_CUBE);
            } else {
                loadDataParams.kStep = Cmct::Gemm::CeilDiv(kL0, C0_SIZE);
            }
            loadDataParams.srcStride = Cmct::Gemm::CeilDiv(curNL1, AscendC::BLOCK_CUBE);
            loadDataParams.dstStride = loadDataParams.mStep;
            loadDataParams.ifTranspose = false;
            if constexpr (AscendC::IsSameType<B_T, bfloat16_t>::value) {
                AscendC::LoadData(b2Local, bl1Local, loadDataParams);
            } else {
                AscendC::LoadData<B_T>(b2Local, bl1Local, loadDataParams);
            }
        } else {
            // (K, N) use LoadData2D
            AscendC::LoadData2DParamsV2 loadDataParams;
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = 0;
            loadDataParams.mStep = Cmct::Gemm::CeilDiv(kL0, AscendC::BLOCK_CUBE);
            if constexpr (AscendC::IsSameType<A_T, half>::value || AscendC::IsSameType<A_T, bfloat16_t>::value) {
                loadDataParams.kStep = Cmct::Gemm::CeilDiv(nL0, AscendC::BLOCK_CUBE);
                loadDataParams.dstStride = loadDataParams.kStep;
            } else {
                loadDataParams.kStep = Cmct::Gemm::CeilDiv(nL0, AscendC::BLOCK_CUBE) * TWO_ALIGN;
                loadDataParams.dstStride = loadDataParams.kStep >> 1;
            }
            loadDataParams.srcStride = Cmct::Gemm::CeilDiv(curKL1, AscendC::BLOCK_CUBE);
            loadDataParams.ifTranspose = true;
            if constexpr (AscendC::IsSameType<B_T, bfloat16_t>::value) {
                AscendC::LoadData(b2Local, bl1Local, loadDataParams);
            } else {
                AscendC::LoadData<B_T>(b2Local, bl1Local, loadDataParams);
            }
        }
    }

    __aicore__ inline void CopyOutForFixedPoint(const AscendC::GlobalTensor<C_T>& cGlobal,
                                                AscendC::LocalTensor<L0cType>& c1Local, uint64_t baseM, uint64_t baseN)
    {
        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = static_cast<uint16_t>(baseN);
        fixpipeParams.mSize = static_cast<uint16_t>(baseM);
        fixpipeParams.dstStride = n_;
        fixpipeParams.srcStride = CeilAlign(baseM, BLOCK_CUBE);
        fixpipeParams.params = {1, static_cast<uint16_t>(baseM), static_cast<uint16_t>(baseN)};
        if constexpr (DispatchPolicy::enableQuant) {
            fixpipeParams.quantPre = QuantMode_t::REQ8;
            fixpipeParams.deqScalar = quantScalar_;
        } else {
            fixpipeParams.quantPre = QuantMode_t::DEQF16;
        }
        fixpipeParams.unitFlag = enableL0cPingPong_ ? 0 : FINAL_ACCUMULATION; // 3 unitflag
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 1;
        fixpipeParams.params.dstNdStride = 1;
        if constexpr (DispatchPolicy::enableRelu) {
            fixpipeParams.reluEn = 1;
        } else {
            fixpipeParams.reluEn = 0;
        }
#if __NPU_ARCH__ == 5102
        fixpipeParams.fixShiftVal = FIX_SHIFT_VAL_LEN_A16W16 - shiftValue_;
#endif
        AscendC::Fixpipe<C_T, L0cType, AscendC::CFG_ROW_MAJOR>(cGlobal, c1Local, fixpipeParams);
    }

    __aicore__ inline void CopyOutForOther(const AscendC::GlobalTensor<C_T>& cGlobal,
                                           AscendC::LocalTensor<L0cType>& c1Local, uint64_t baseM, uint64_t baseN)
    {
        if (isSplitSingleK_) {
            PipeBarrier<PIPE_FIX>();
            if (!isFirstSplitK_) {
                AscendC::SetAtomicAdd<float>();
            }
        }
        AscendC::DataCopyCO12DstParams intriParams;
        intriParams.nSize = baseN;
        intriParams.mSize = baseM;
        intriParams.dstStride = n_;
        intriParams.srcStride = Cmct::Gemm::Align(baseM, AscendC::BLOCK_CUBE);
        // set mode according to dtype
        if constexpr (AscendC::IsSameType<C_T, bfloat16_t>::value) {
            intriParams.quantPre = QuantMode_t::F322BF16;
        } else if (AscendC::IsSameType<C_T, half>::value) {
            intriParams.quantPre = QuantMode_t::F322F16;
        } else if (AscendC::IsSameType<C_T, float>::value) {
            intriParams.quantPre = QuantMode_t::NoQuant;
        }
        if constexpr (DispatchPolicy::enableRelu) {
            intriParams.reluPre = 1;
        } else {
            intriParams.reluPre = 0;
        }
        intriParams.nz2ndEn = true;
        intriParams.unitFlag = enableL0cPingPong_ ? 0 : FINAL_ACCUMULATION; // 3 unitflag
        AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
        AscendC::DataCopy(cGlobal, c1Local, intriParams);
        if (isSplitSingleK_ && isEndSplitK_) {
            AscendC::DisableDmaAtomic();
        }
    }

    __aicore__ inline void CopyOut(const AscendC::GlobalTensor<C_T>& cGlobal, AscendC::LocalTensor<L0cType>& c1Local,
                                   uint64_t baseM, uint64_t baseN)
    {
#if __FIXED_POINT_ONLY_CUBE_TO_L0C__
        CopyOutForFixedPoint(cGlobal, c1Local, baseM, baseN);
#else
        CopyOutForOther(cGlobal, c1Local, baseM, baseN);
#endif
    }

    // fixpipe CopyOut实现c01拷贝到UB
    __aicore__ inline void CopyOut(const AscendC::LocalTensor<C_T>& dstLocal, AscendC::LocalTensor<L0cType>& c1Local,
                                   uint64_t baseM, uint64_t baseN)
    {
        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixpipeParams; // ROW_MAJOR默认使能NZ2ND
        uint64_t c0 = AscendC::AuxGetC0Size<C_T>();
        fixpipeParams.nSize = Cmct::Gemm::Align(baseN, c0);
        fixpipeParams.mSize = splitM_ ? Cmct::Gemm::Align(baseM, SPLIT_M_ALIGN) : baseM; // 切m需要m是2对齐
        fixpipeParams.dstStride = fixpipeParams.nSize;
        fixpipeParams.srcStride = Cmct::Gemm::Align(baseM, AscendC::BLOCK_CUBE); // 单位CO_SIZE (16*sizeof(C_T))

        if constexpr (AscendC::IsSameType<C_T, bfloat16_t>::value) {
            fixpipeParams.quantPre = QuantMode_t::F322BF16;
        } else if (AscendC::IsSameType<C_T, half>::value) {
            fixpipeParams.quantPre = QuantMode_t::F322F16;
        } else if (AscendC::IsSameType<C_T, float>::value) {
            fixpipeParams.quantPre = QuantMode_t::NoQuant;
        }
        // set cvRatio=1:2 默认splitM
        fixpipeParams.dualDstCtl = splitM_ ? static_cast<uint8_t>(AscendC::McgShfMode::DUAL_DST_SPLIT_M) : 0;
        fixpipeParams.unitFlag = enableL0cPingPong_ ? 0 : FINAL_ACCUMULATION; // 3 unitflag
        fixpipeParams.params.ndNum = 1;                                       // ndNum
        fixpipeParams.params.srcNdStride = 1;                                 // srcNdStride
        fixpipeParams.params.dstNdStride = 1;                                 // dstNdStride
        AscendC::Fixpipe<C_T, L0cType, AscendC::Impl::CFG_ROW_MAJOR_UB>(dstLocal, c1Local, fixpipeParams);
    }

    // 重载GlobalTensor
    __aicore__ inline void DoubleCopyOut(const AscendC::GlobalTensor<C_T>& cGlobal, uint64_t l0cOffset, uint64_t baseM,
                                         uint64_t baseN)
    {
        AscendC::LocalTensor<L0cType> c1Local = c1Local_[l0cOffset];
        return CopyOut(cGlobal, c1Local, baseM, baseN);
    }

    __aicore__ inline void DoubleCopyOutTwoFixpipe(const AscendC::LocalTensor<C_T>& dstLocal, uint64_t l0cOffset,
                                                   uint64_t baseM, uint64_t baseN)
    {
        AscendC::LocalTensor<L0cType> c1Local = c1Local_[l0cOffset];
        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixpipeParams; // ROW_MAJOR默认使能NZ2ND
        uint64_t c0 = AscendC::AuxGetC0Size<C_T>();
        uint64_t halfBaseM = Cmct::Gemm::CeilDiv(baseM, SPLIT_M_ALIGN);
        // 第一条Fixpipe指令
        fixpipeParams.nSize = Cmct::Gemm::Align(baseN, c0);
        fixpipeParams.mSize = halfBaseM;
        fixpipeParams.dstStride = fixpipeParams.nSize;
        fixpipeParams.srcStride = Cmct::Gemm::Align(baseM, AscendC::BLOCK_CUBE); // M方向stride
        if constexpr (AscendC::IsSameType<C_T, bfloat16_t>::value) {
            fixpipeParams.quantPre = QuantMode_t::F322BF16;
        } else if (AscendC::IsSameType<C_T, half>::value) {
            fixpipeParams.quantPre = QuantMode_t::F322F16;
        } else if (AscendC::IsSameType<C_T, float>::value) {
            fixpipeParams.quantPre = QuantMode_t::NoQuant;
        }
        fixpipeParams.dualDstCtl = 0;
        fixpipeParams.unitFlag = 0;           // no unitflag
        fixpipeParams.subBlockId = 0;         // aiv0
        fixpipeParams.params.ndNum = 1;       // ndNum
        fixpipeParams.params.srcNdStride = 1; // srcNdStride
        fixpipeParams.params.dstNdStride = 1; // dstNdStride
        AscendC::Fixpipe<C_T, L0cType, AscendC::Impl::CFG_ROW_MAJOR_UB>(dstLocal, c1Local, fixpipeParams);

        // 第二条Fixpipe指令
        if (baseM == 1) {
            return;
        }
        // LOC偏移[M/2*16]
        AscendC::LocalTensor<L0cType> c1LocalNext = c1Local_[l0cOffset + halfBaseM * AscendC::BLOCK_CUBE];
        fixpipeParams.mSize = baseM - halfBaseM; // baseM - baseM/2
        fixpipeParams.subBlockId = 1;            // aiv1
        AscendC::Fixpipe<C_T, L0cType, AscendC::Impl::CFG_ROW_MAJOR_UB>(dstLocal, c1LocalNext, fixpipeParams);
    }

    __aicore__ inline void DoubleCopyOutDualSplitM(const AscendC::LocalTensor<C_T>& dstLocal, uint64_t l0cOffset,
                                                   uint64_t baseM, uint64_t baseN)
    {
        AscendC::LocalTensor<L0cType> c1Local = c1Local_[l0cOffset];
        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixpipeParams;
        uint64_t c0 = AscendC::AuxGetC0Size<C_T>();
        fixpipeParams.nSize = Cmct::Gemm::Align(baseN, c0);
        fixpipeParams.mSize = Cmct::Gemm::Align(baseM, SPLIT_M_ALIGN);
        bool need16Align = false;
        if constexpr (DispatchPolicy::enableGelu || DispatchPolicy::enableSilu || DispatchPolicy::enableSwigluSingle) {
            // GELU/SILU/SWIGLU consumes float workspace through vector copy, whose row stride follows fp16 C0 alignment.
            need16Align = true;
        } else if constexpr (DispatchPolicy::enableHighPrecision &&
                             (DispatchPolicy::enableAdd || DispatchPolicy::enableMul)) {
            // High-precision Add/Mul (INNER_PRECISE=0) tail (baseN & 0xF) in [1, N_TAIL_ALIGN_THRESHOLD]
            // pads row stride to fp16 C0 alignment.
            need16Align = ((baseN & 0xF) > 0 && (baseN & 0xF) <= N_TAIL_ALIGN_THRESHOLD);
        }
        fixpipeParams.dstStride = need16Align ? Cmct::Gemm::Align(fixpipeParams.nSize, AscendC::AuxGetC0Size<half>()) :
                                                fixpipeParams.nSize;
        fixpipeParams.srcStride = Cmct::Gemm::Align(baseM, AscendC::BLOCK_CUBE);
        fixpipeParams.dualDstCtl = static_cast<uint8_t>(AscendC::McgShfMode::DUAL_DST_SPLIT_M);
        fixpipeParams.unitFlag = 0;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 1;
        fixpipeParams.params.dstNdStride = 1;
        AscendC::Fixpipe<C_T, L0cType, AscendC::Impl::CFG_ROW_MAJOR_UB>(dstLocal, c1Local, fixpipeParams);
    }

    // Fixpipe: copy L0C to UB of two aiv
    // 非随路quantPre走DUAL_DST_SPLIT_M单指令(性能优), 需要Quant走两条Fixpipe兼容
    __aicore__ inline void DoubleCopyOut(const AscendC::LocalTensor<C_T>& dstLocal, uint64_t l0cOffset, uint64_t baseM,
                                         uint64_t baseN)
    {
        if constexpr (AscendC::IsSameType<C_T, float>::value) {
            // splitM不支持quantPre
            DoubleCopyOutDualSplitM(dstLocal, l0cOffset, baseM, baseN);
        } else {
            DoubleCopyOutTwoFixpipe(dstLocal, l0cOffset, baseM, baseN);
        }
    }

    template <typename T, CubeFormat LayoutB = CubeFormat::ND>
    __aicore__ inline void operator()(T cTensor, AscendC::GlobalTensor<A_T> aGlobal, AscendC::GlobalTensor<B_T> bGlobal,
                                      AscendC::GlobalTensor<Bias_T> biasGlobal, TupleL1L0Shape tileShape,
                                      uint64_t mOffset, uint64_t nOffset, bool isFirstTile = false,
                                      bool isAllLoc2Ub = false, uint64_t blkK = 0, bool isFirstSplitK = false,
                                      bool isEndSplitK = false)
    {
        if (fullLoadMode_ == A_FULL_LOAD_MODE) {
            return DoAFullLoad<T, LayoutB>(cTensor, aGlobal, bGlobal, biasGlobal, tileShape, mOffset);
        } else if (isAllLoc2Ub) {
            return DoAllLoc2UbAswt(cTensor, aGlobal, bGlobal, biasGlobal, tileShape, nOffset);
        } else {
            return DoBFullLoadOrAswt<T, LayoutB>(cTensor, aGlobal, bGlobal, biasGlobal, tileShape, nOffset, isFirstTile,
                                                 blkK, isFirstSplitK, isEndSplitK);
        }
    }

    template <typename T>
    __aicore__ inline void DoAllLoc2UbAswt(T cTensor, AscendC::GlobalTensor<A_T> aGlobal,
                                           AscendC::GlobalTensor<B_T> bGlobal, AscendC::GlobalTensor<Bias_T> biasGlobal,
                                           TupleL1L0Shape tileShape, uint64_t nL1Offset)
    {
        uint64_t curML1 = Get<MNK_M>(tileShape);
        uint64_t curNL1 = Get<MNK_N>(tileShape);
        uint64_t curML0 = Get<MNK_M0>(tileShape);
        uint64_t curNL0 = Get<MNK_N0>(tileShape);
        uint64_t ml1Align = Cmct::Gemm::Align(curML1, AscendC::BLOCK_CUBE);
        uint64_t nl1Align = Cmct::Gemm::Align(curNL1, AscendC::BLOCK_CUBE);
        uint64_t kbL1Size = kL1_;
        AscendC::MmadParams mmadParams;
        mmadParams.m = curML0;
        mmadParams.n = curNL0;
        mmadParams.disableGemv = true;
        AscendC::LocalTensor<Bias_T> biasL1LocalInit;
        AscendC::LocalTensor<B_T> bl1Local;
        uint64_t kl1Offset = 0;
        uint64_t l0cOffset = (l0cPingPong_ & 0x1) * HALF_L0C_SIZE;
        if (enableL0cPingPong_) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cPingPong_ & 0x1);
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
        }
        AscendC::LocalTensor<Bias_T> biasL1Local;
        kL1_ = Min(k_, kL1_);
        uint64_t curKL1 = kL1_;
        uint64_t kL1OffsetLength = 0;
        uint64_t curKL1Iter = kL1Iter_;
        for (uint64_t iter0 = 0; iter0 < curKL1Iter; ++iter0) {
            curKL1 = (iter0 + 1 == curKL1Iter) ? (k_ - kL1OffsetLength) : kL1_;
            // A搬运数据到L1，开启4buffer
            uint64_t l1BufId = abL1LoopCnt_ & (l1BufNum_ - 1);
            uint64_t offsetA = AType::isTrans ? kL1OffsetLength * m_ : kL1OffsetLength;
            // 普通模板-2buffer-AL1搬入偏移位置：*AL1Ping*-BL1Ping-BiasPing|*AL1Pong*-BL1Pong-BiasPong
            // 普通模板-4buffer-AL1搬入偏移位置: *AL1Ping-AL1Pong*-BL1Ping-BL1Pong-BiasPing-BiasPong
            uint64_t offsetAl1 = (DispatchPolicy::fullLoadMode == 0 && l1BufNum_ == DOUBLE_BUFFER_COUNT) ?
                                     (HALF_L1_SIZE * l1BufId) :
                                     aL1OneBuffer_ * l1BufId;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            uint64_t biasBufId = abL1LoopCnt_ & 0x1;

            CopyInA1(aGlobal[offsetA], l1Local_[offsetAl1], curML1, curKL1);
            if constexpr (DispatchPolicy::fullLoadMode == 0) {
                if (isBias_ && iter0 == 0) {
                    // 普通模板-2buffer-Bias搬入偏移位置：AL1Ping-BL1Ping-*BiasPing*|AL1Pong-BL1Pong-*BiasPong*
                    // 普通模板-4buffer-Bias搬入偏移位置: AL1Ping-AL1Pong-BL1Ping-BL1Pong-*BiasPing-BiasPong*
                    biasL1LocalInit = l1Local_[(l1BufNum_ == DOUBLE_BUFFER_COUNT) ?
                                                   HALF_L1_SIZE * l1BufId + aL1OneBuffer_ + bL1OneBuffer_ :
                                                   aL1OneBuffer_ * l1BufNum_ + bL1OneBuffer_ * l1BufNum_]
                                          .template ReinterpretCast<Bias_T>();
                    biasL1Local = biasL1LocalInit[(l1BufNum_ == DOUBLE_BUFFER_COUNT) ? 0 : nL1_ * l1BufId];
                    CopyInC1(biasGlobal, biasL1Local, curNL1);
                }
                // B搬运数据到L1，开启4buffer
                // 普通模板-2buffer-BL1搬入偏移位置：AL1Ping-*BL1Ping*-BiasPing|AL1Pong-*BL1Pong*-BiasPong
                // 普通模板-4buffer-BL1搬入偏移位置: AL1Ping-AL1Pong-*BL1Ping-BL1Pong*-BiasPing-BiasPong
                uint64_t offsetBl1 = (l1BufNum_ == DOUBLE_BUFFER_COUNT) ?
                                         (HALF_L1_SIZE * l1BufId + aL1OneBuffer_) :
                                         (aL1OneBuffer_ * l1BufNum_ + bL1OneBuffer_ * l1BufId);
                bl1Local = l1Local_[offsetBl1];
                uint64_t offsetB = BType::isTrans ? kL1OffsetLength : kL1OffsetLength * n_;
                CopyInB1(bGlobal[offsetB], bl1Local, curNL1, curKL1);
                kbL1Size = curKL1;
            } else {
                // B全载-BL1搬入偏移位置：AL1Ping-AL1Pong-*BL1*-Bias
                bl1Local = l1Local_[aL1OneBuffer_ * l1BufNum_];
                kl1Offset = kL1OffsetLength;
                // B全载-Bias搬入偏移位置：AL1Ping-AL1Pong-BL1-*Bias*
                biasL1LocalInit = l1Local_[aL1OneBuffer_ * l1BufNum_ + bL1OneBuffer_]
                                      .template ReinterpretCast<Bias_T>();
                biasL1Local = biasL1LocalInit[nL1Offset];
                kbL1Size = kAlign_;
            }
            kL1OffsetLength += curKL1;
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

            uint64_t kL0Iter = (curKL1 + baseK_ - 1) / baseK_;
            for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                uint64_t curK0 = (iter1 + 1 == kL0Iter) ? (curKL1 - iter1 * baseK_) : baseK_;
                // 搬运数据到L0 开启DB
                uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
                uint64_t mte1Flag = ((l0PingPong_ & 0x1) + SIXTH_FLAG);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(static_cast<uint16_t>(mte1Flag));
                CopyInA2(l0aLocal_[l0Offset], l1Local_[offsetAl1], curML1, curKL1, curML0, curK0);
                offsetAl1 += AType::isTrans ? baseK_ * C0_SIZE : ml1Align * baseK_;
                // copy bias to bt
                CopyInC2(biasL1Local, biasBt_[baseN_ * biasBufId], Cmct::Gemm::Align(mmadParams.n, AscendC::BLOCK_CUBE),
                         NeedBias(iter0, iter1));
                uint64_t offsetBl1 = 0;
                if constexpr (BType::isTrans) {
                    offsetBl1 = nL1Offset * C0_SIZE + (kl1Offset + iter1 * baseK_) * nl1Align;
                } else {
                    offsetBl1 = nL1Offset * kAlign_ + (iter1 * baseK_ + kl1Offset) * C0_SIZE;
                }
                CopyInB2(l0bLocal_[l0Offset], bl1Local[offsetBl1], curNL1, kbL1Size, curNL0, curK0);

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(static_cast<uint16_t>(mte1Flag));
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(static_cast<uint16_t>(mte1Flag));
                mmadParams.k = curK0;
                mmadParams.unitFlag = 0; // no unitFlag
                mmadParams.cmatrixInitVal = (iter0 == 0 && iter1 == 0 && !isBias_);
                Mmad(mmadParams, l0cOffset, l0Offset, baseN_ * biasBufId, NeedBias(iter0, iter1));
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(static_cast<uint16_t>(mte1Flag));
                l0PingPong_++;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            abL1LoopCnt_++;
        }
        if (enableL0cPingPong_) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0cPingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0cPingPong_ & 0x1);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(ZERO_FLAG);
        }
        // 数据搬出到GM或者ub
        DoubleCopyOut(cTensor, l0cOffset, mmadParams.m, mmadParams.n);
        if (enableL0cPingPong_) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0cPingPong_ & 0x1);
            l0cPingPong_++;
        } else {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
        }
    }

    template <typename T, CubeFormat LayoutB>
    __aicore__ inline void DoBFullLoadOrAswt(T cTensor, AscendC::GlobalTensor<A_T> aGlobal,
                                             AscendC::GlobalTensor<B_T> bGlobal,
                                             AscendC::GlobalTensor<Bias_T> biasGlobal, TupleL1L0Shape tileShape,
                                             uint64_t nL1Offset, bool isFirstTile, uint64_t blkK = 0,
                                             bool isFirstSplitK = false, bool isEndSplitK = false)
    {
        if (isSplitSingleK_) {
            blkK_ = blkK;
            kAlign_ = Cmct::Gemm::Align(blkK, AscendC::BLOCK_CUBE);
            kL1Iter_ = CeilDiv(blkK, kL1_);
            isFirstSplitK_ = isFirstSplitK;
            isEndSplitK_ = isEndSplitK;
        }
        uint64_t curML1 = Get<MNK_M>(tileShape);
        uint64_t curNL1 = Get<MNK_N>(tileShape);
        uint64_t curML0 = Get<MNK_M0>(tileShape);
        uint64_t curNL0 = Get<MNK_N0>(tileShape);
        uint64_t ml1Align = Cmct::Gemm::Align(curML1, AscendC::BLOCK_CUBE);
        uint64_t nl1Align = Cmct::Gemm::Align(curNL1, AscendC::BLOCK_CUBE);
        uint64_t kbL1Size = kL1_;
        AscendC::MmadParams mmadParams;
        mmadParams.m = curML0;
        mmadParams.n = curNL0;
        mmadParams.disableGemv = true;
        AscendC::LocalTensor<Bias_T> biasL1LocalInit;
        AscendC::LocalTensor<B_T> bl1Local;
        uint64_t kl1Offset = 0;
        uint64_t l0cOffset = (l0cPingPong_ & 0x1) * HALF_L0C_SIZE;
        if (enableL0cPingPong_) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cPingPong_ & 0x1);
        }
        AscendC::LocalTensor<Bias_T> biasL1Local;
        kL1_ = Min(blkK_, kL1_);
        uint64_t curKL1 = kL1_;
        bool isFirstLoopKL1Half = false;
        uint64_t kL1OffsetLength = 0;
        uint64_t curKL1Iter = kL1Iter_;
        // 若stepK>=2,则开启首轮减半
        if (isFirstTile && kL1_ / baseK_ >= NUM_TWO) {
            isFirstLoopKL1Half = true;
            curKL1Iter++;
        }
        for (uint64_t iter0 = 0; iter0 < curKL1Iter; ++iter0) {
            curKL1 = (iter0 + 1 == curKL1Iter) ? (blkK_ - kL1OffsetLength) : kL1_;
            // 前两轮将搬运量减半，提前mmad计算
            if (isFirstLoopKL1Half) {
                if (iter0 == 0) {
                    curKL1 = CeilAlign(kL1_ / NUM_TWO, AscendC::BLOCK_CUBE);
                } else if (iter0 == 1) {
                    curKL1 = kL1_ - kL1OffsetLength;
                }
            }
            // A搬运数据到L1，开启4buffer
            uint64_t l1BufId = abL1LoopCnt_ & (l1BufNum_ - 1);
            uint64_t offsetA = AType::isTrans ? kL1OffsetLength * m_ : kL1OffsetLength;
            // 普通模板-2buffer-AL1搬入偏移位置：*AL1Ping*-BL1Ping-BiasPing|*AL1Pong*-BL1Pong-BiasPong
            // 普通模板-4buffer-AL1搬入偏移位置: *AL1Ping-AL1Pong*-BL1Ping-BL1Pong-BiasPing-BiasPong
            uint64_t offsetAl1 = (DispatchPolicy::fullLoadMode == 0 && l1BufNum_ == DOUBLE_BUFFER_COUNT) ?
                                     (HALF_L1_SIZE * l1BufId) :
                                     aL1OneBuffer_ * l1BufId;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            uint64_t biasBufId = abL1LoopCnt_ & 0x1;

            CopyInA1(aGlobal[offsetA], l1Local_[offsetAl1], curML1, curKL1);
            if constexpr (DispatchPolicy::fullLoadMode == 0) {
                if (isBias_ && iter0 == 0) {
                    // 普通模板-2buffer-Bias搬入偏移位置：AL1Ping-BL1Ping-*BiasPing*|AL1Pong-BL1Pong-*BiasPong*
                    // 普通模板-4buffer-Bias搬入偏移位置: AL1Ping-AL1Pong-BL1Ping-BL1Pong-*BiasPing-BiasPong*
                    biasL1LocalInit = l1Local_[(l1BufNum_ == DOUBLE_BUFFER_COUNT) ?
                                                   HALF_L1_SIZE * l1BufId + aL1OneBuffer_ + bL1OneBuffer_ :
                                                   aL1OneBuffer_ * l1BufNum_ + bL1OneBuffer_ * l1BufNum_]
                                          .template ReinterpretCast<Bias_T>();
                    biasL1Local = biasL1LocalInit[(l1BufNum_ == DOUBLE_BUFFER_COUNT) ? 0 : nL1_ * l1BufId];
                    CopyInC1(biasGlobal, biasL1Local, curNL1);
                }
                // B搬运数据到L1，开启4buffer
                // 普通模板-2buffer-BL1搬入偏移位置：AL1Ping-*BL1Ping*-BiasPing|AL1Pong-*BL1Pong*-BiasPong
                // 普通模板-4buffer-BL1搬入偏移位置: AL1Ping-AL1Pong-*BL1Ping-BL1Pong*-BiasPing-BiasPong
                uint64_t offsetBl1 = (l1BufNum_ == DOUBLE_BUFFER_COUNT) ?
                                         (HALF_L1_SIZE * l1BufId + aL1OneBuffer_) :
                                         (aL1OneBuffer_ * l1BufNum_ + bL1OneBuffer_ * l1BufId);
                bl1Local = l1Local_[offsetBl1];
                uint64_t offsetB = BType::isTrans ? kL1OffsetLength : kL1OffsetLength * n_;
                if constexpr (LayoutB == CubeFormat::NZ) {
                    if constexpr (BType::isTrans) {
                        offsetB = kL1OffsetLength * Cmct::Gemm::CeilAlign(n_, AscendC::BLOCK_CUBE);
                    } else {
                        offsetB = kL1OffsetLength * C0_SIZE;
                    }
                }
                CopyInB1<LayoutB>(bGlobal[offsetB], bl1Local, curNL1, curKL1);
                kbL1Size = curKL1;
            } else {
                // B全载-BL1搬入偏移位置：AL1Ping-AL1Pong-*BL1*-Bias
                bl1Local = l1Local_[aL1OneBuffer_ * l1BufNum_];
                kl1Offset = kL1OffsetLength;
                // B全载-Bias搬入偏移位置：AL1Ping-AL1Pong-BL1-*Bias*
                biasL1LocalInit = l1Local_[aL1OneBuffer_ * l1BufNum_ + bL1OneBuffer_]
                                      .template ReinterpretCast<Bias_T>();
                biasL1Local = biasL1LocalInit[nL1Offset];
                kbL1Size = kAlign_;
            }
            kL1OffsetLength += curKL1;
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

            uint64_t kL0Iter = (curKL1 + baseK_ - 1) / baseK_;
            for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                uint64_t curK0 = (iter1 + 1 == kL0Iter) ? (curKL1 - iter1 * baseK_) : baseK_;
                // 搬运数据到L0 开启DB
                uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
                uint64_t mte1Flag = ((l0PingPong_ & 0x1) + SIXTH_FLAG);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(static_cast<uint16_t>(mte1Flag));
                CopyInA2(l0aLocal_[l0Offset], l1Local_[offsetAl1], curML1, curKL1, curML0, curK0);
                offsetAl1 += AType::isTrans ? baseK_ * C0_SIZE : ml1Align * baseK_;
                // copy bias to bt
                CopyInC2(biasL1Local, biasBt_[baseN_ * biasBufId], Cmct::Gemm::Align(mmadParams.n, AscendC::BLOCK_CUBE),
                         NeedBias(iter0, iter1));
                uint64_t offsetBl1 = 0;
                if constexpr (BType::isTrans) {
                    offsetBl1 = nL1Offset * C0_SIZE + (kl1Offset + iter1 * baseK_) * nl1Align;
                } else {
                    offsetBl1 = nL1Offset * kAlign_ + (iter1 * baseK_ + kl1Offset) * C0_SIZE;
                }
                CopyInB2(l0bLocal_[l0Offset], bl1Local[offsetBl1], curNL1, kbL1Size, curNL0, curK0);

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(static_cast<uint16_t>(mte1Flag));
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(static_cast<uint16_t>(mte1Flag));
                mmadParams.k = curK0;
                mmadParams.unitFlag = enableL0cPingPong_ ?
                                          0 :
                                          ((iter0 + 1 == curKL1Iter && iter1 + 1 == kL0Iter) ? FINAL_ACCUMULATION :
                                                                                               NON_FINAL_ACCUMULATION);
                if (isSplitSingleK_) {
                    mmadParams.cmatrixInitVal = (iter0 == 0 && iter1 == 0 && !(isBias_ && isFirstSplitK_));
                } else {
                    mmadParams.cmatrixInitVal = (iter0 == 0 && iter1 == 0 && !isBias_);
                }
                Mmad(mmadParams, l0cOffset, l0Offset, baseN_ * biasBufId, NeedBias(iter0, iter1));
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(static_cast<uint16_t>(mte1Flag));
                l0PingPong_++;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            abL1LoopCnt_++;
        }
        AscendC::LocalTensor<L0cType> c1Local = c1Local_[l0cOffset];
        if (enableL0cPingPong_) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0cPingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0cPingPong_ & 0x1);
        }
        // 数据搬出到GM或者ub
        CopyOut(cTensor, c1Local, mmadParams.m, mmadParams.n);
        if (enableL0cPingPong_) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0cPingPong_ & 0x1);
            l0cPingPong_++;
        }
    }

    // A全载mmad
    template <typename T, CubeFormat LayoutB>
    __aicore__ inline void DoAFullLoad(T cTensor, AscendC::GlobalTensor<A_T> aGlobal,
                                       AscendC::GlobalTensor<B_T> bGlobal, AscendC::GlobalTensor<Bias_T> biasGlobal,
                                       TupleL1L0Shape tileShape, uint64_t mL1Offset)
    {
        uint64_t curML1 = Get<MNK_M>(tileShape);
        uint64_t curNL1 = Get<MNK_N>(tileShape);
        uint64_t curML0 = Get<MNK_M0>(tileShape);
        uint64_t curNL0 = Get<MNK_N0>(tileShape);
        uint64_t ml1Align = Cmct::Gemm::Align(curML1, AscendC::BLOCK_CUBE);
        uint64_t nl1Align = Cmct::Gemm::Align(curNL1, AscendC::BLOCK_CUBE);
        uint64_t kaL1Size = kL1_;
        AscendC::MmadParams mmadParams;
        mmadParams.m = curML0;
        mmadParams.n = curNL0;
        mmadParams.disableGemv = true;
        // A全载-Bias搬入偏移位置：AL1-BL1Ping-BL1Pong-*BiasPing-BiasPong*
        AscendC::LocalTensor<Bias_T> biasL1LocalInit = l1Local_[aL1OneBuffer_ + bL1OneBuffer_ * l1BufNum_]
                                                           .template ReinterpretCast<Bias_T>();
        AscendC::LocalTensor<A_T> aL1Local;

        uint64_t l0cOffset = (l0cPingPong_ & 0x1) * HALF_L0C_SIZE;
        if (enableL0cPingPong_) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cPingPong_ & 0x1);
        }

        uint64_t kL1Offset = 0;
        AscendC::LocalTensor<Bias_T> biasL1Local;
        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            uint64_t curKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - iter0 * kL1_) : kL1_;
            uint64_t l1BufId = abL1LoopCnt_ & (l1BufNum_ - 1);
            uint64_t offsetB = BType::isTrans ? iter0 * kL1_ : iter0 * kL1_ * n_;
            // A全载-BL1搬入偏移位置：AL1-*BL1Ping-BL1Pong*-BiasPing-BiasPong
            uint64_t offsetBl1 = aL1OneBuffer_ + bL1OneBuffer_ * l1BufId;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            // B -> L1
            if constexpr (LayoutB == CubeFormat::NZ) {
                if constexpr (BType::isTrans) {
                    offsetB = iter0 * kL1_ * CeilAlign(n_, AscendC::BLOCK_CUBE);
                } else {
                    offsetB = iter0 * kL1_ * C0_SIZE;
                }
            }
            CopyInB1<LayoutB>(bGlobal[offsetB], l1Local_[offsetBl1], curNL1, curKL1);

            if (isBias_ && iter0 == 0) {
                biasL1Local = biasL1LocalInit[nL1_ * l1BufId];
                CopyInC1(biasGlobal, biasL1Local, curNL1);
            }
            // A -> L1
            aL1Local = l1Local_; // biasL1 -> AL1 -> BL1
            kL1Offset = iter0 * kL1_;
            kaL1Size = kAlign_;

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

            uint64_t kL0Iter = (curKL1 + baseK_ - 1) / baseK_;
            for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                uint64_t curK0 = (iter1 + 1 == kL0Iter) ? (curKL1 - iter1 * baseK_) : baseK_;
                // 搬运数据到L0 开启DB
                uint64_t mte1Flag = ((l0PingPong_ & 0x1) + SIXTH_FLAG);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(static_cast<uint16_t>(mte1Flag));
                uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);

                CopyInB2(l0bLocal_[l0Offset], l1Local_[offsetBl1], curNL1, curKL1, curNL0, curK0);
                offsetBl1 += BType::isTrans ? nl1Align * baseK_ : baseK_ * C0_SIZE;

                // copy bias
                CopyInC2(biasL1Local, biasBt_[baseN_ * (abL1LoopCnt_ & 0x1)],
                         Cmct::Gemm::Align(mmadParams.n, AscendC::BLOCK_CUBE), NeedBias(iter0, iter1));

                uint64_t offsetAl1 = 0;
                if constexpr (AType::isTrans) {
                    offsetAl1 = mL1Offset * kAlign_ + (kL1Offset + iter1 * baseK_) * C0_SIZE;
                } else {
                    offsetAl1 = mL1Offset * C0_SIZE + ml1Align * (kL1Offset + iter1 * baseK_);
                }

                CopyInA2(l0aLocal_[l0Offset], aL1Local[offsetAl1], curML1, kaL1Size, curML0, curK0);

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(static_cast<uint16_t>(mte1Flag));
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(static_cast<uint16_t>(mte1Flag));

                mmadParams.k = curK0;
                // 进行mad计算, 设置unitflag状态，3表示最后一次累加，2表示非最后一次累加
                mmadParams.unitFlag = enableL0cPingPong_ ?
                                          0 :
                                          ((iter0 + 1 == kL1Iter_ && iter1 + 1 == kL0Iter) ? FINAL_ACCUMULATION :
                                                                                             NON_FINAL_ACCUMULATION);
                mmadParams.cmatrixInitVal = (iter0 == 0 && iter1 == 0 && !isBias_);
                // mmad
                Mmad(mmadParams, l0cOffset, l0Offset, baseN_ * (abL1LoopCnt_ & 0x1), NeedBias(iter0, iter1));

                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(static_cast<uint16_t>(mte1Flag));
                l0PingPong_++;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            abL1LoopCnt_++;
        }

        if (enableL0cPingPong_) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0cPingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0cPingPong_ & 0x1);
        }

        // 数据搬出到GM或者ub
        DoubleCopyOut(cTensor, l0cOffset, mmadParams.m, mmadParams.n);
        if (enableL0cPingPong_) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0cPingPong_ & 0x1);
            l0cPingPong_++;
        }
    }

private:
    __aicore__ inline bool NeedBias(uint64_t kIter0, uint64_t kIter1)
    {
        if (isSplitSingleK_) {
            return isBias_ && kIter0 == 0 && kIter1 == 0 && isFirstSplitK_;
        } else {
            return isBias_ && kIter0 == 0 && kIter1 == 0;
        }
    }

    __aicore__ inline void Mmad(AscendC::MmadParams& mmadParams, uint64_t l0cOffset, uint64_t l0abOffset,
                                uint64_t biasOffset, bool needBias)
    {
        mmadParams.cmatrixSource = needBias;
#if __NPU_ARCH__ == 5102
        mmadParams.fixShiftVal = shiftValue_;
#endif
        if (needBias) {
            AscendC::Mmad(c1Local_[l0cOffset], l0aLocal_[l0abOffset], l0bLocal_[l0abOffset], biasBt_[biasOffset],
                          mmadParams);
        } else {
            mmadParams.cmatrixSource = false;
            AscendC::Mmad(c1Local_[l0cOffset], l0aLocal_[l0abOffset], l0bLocal_[l0abOffset], mmadParams);
        }
    }

private:
    constexpr static uint16_t DIMENSION_M = 0;
    constexpr static uint16_t DIMENSION_N = 1;
    constexpr static uint16_t DIMENSION_K = 2;
    constexpr static uint16_t ZERO_FLAG = 0;
    constexpr static uint16_t FIRST_FLAG = 1;
    constexpr static uint16_t SECOND_FLAG = 2;
    constexpr static uint16_t THIRD_FLAG = 3;
    constexpr static uint16_t FOURTH_FLAG = 4;
    constexpr static uint16_t FIFTH_FLAG = 5;
    constexpr static uint16_t SIXTH_FLAG = 6;
    constexpr static uint16_t SEVENTH_FLAG = 7;
    constexpr static uint16_t M_ALIGN = 16;
    constexpr static uint16_t TWO_ALIGN = 2;
    constexpr static uint16_t NUM_TWO = 2;
    constexpr static int32_t BT_SIZE = 4096;
    uint64_t aL1OneBuffer_ = 0;
    uint64_t bL1OneBuffer_ = 0;
    AscendC::LocalTensor<A_T> l0aLocal_{AscendC::TPosition::A2, 0, L0A_SIZE / sizeof(A_T)};
    AscendC::LocalTensor<B_T> l0bLocal_{AscendC::TPosition::B2, 0, L0B_SIZE / sizeof(B_T)};
    AscendC::LocalTensor<L0cType> c1Local_{AscendC::TPosition::CO1, 0, AscendC::TOTAL_L0C_SIZE / sizeof(L0cType)};
    AscendC::LocalTensor<L0cType> biasBt_{AscendC::TPosition::C2, 0, BT_SIZE / sizeof(L0cType)};
    AscendC::LocalTensor<A_T> l1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE / sizeof(A_T)};
};
} // namespace Block
} // namespace Gemm
} // namespace Cmct
