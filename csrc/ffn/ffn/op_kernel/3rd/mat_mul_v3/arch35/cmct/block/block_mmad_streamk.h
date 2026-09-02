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
 * \file block_mmad_streamk.h
 * \brief
 */

#pragma once
#include "./block_mmad.h"
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
        AscendC::Std::is_base_of_v<MatmulMultiBlockWithStreamK<MatMulL0C2Out::ON_THE_FLY>, DispatchPolicy_> ||
        AscendC::Std::is_base_of_v<MatmulMultiBlockWithStreamK<MatMulL0C2Out::ND_FIXPIPE_1_2>, DispatchPolicy_> ||
        AscendC::Std::is_base_of_v<MatmulMultiBlockWithStreamK<MatMulL0C2Out::ON_THE_FLY, OP_TYPE_RELU>,
                                   DispatchPolicy_> ||
        AscendC::Std::is_base_of_v<MatmulMultiBlockWithStreamK<MatMulL0C2Out::ND_FIXPIPE_1_2, OP_TYPE_RELU>,
                                   DispatchPolicy_> ||
        AscendC::Std::is_base_of_v<MatmulMultiBlockWithStreamK<MatMulL0C2Out::ON_THE_FLY, OP_TYPE_ADD>,
                                   DispatchPolicy_> ||
        AscendC::Std::is_base_of_v<MatmulMultiBlockWithStreamK<MatMulL0C2Out::ND_FIXPIPE_1_2, OP_TYPE_ADD>,
                                   DispatchPolicy_> ||
        AscendC::Std::is_base_of_v<MatmulMultiBlockWithStreamK<MatMulL0C2Out::ON_THE_FLY, OP_TYPE_MUL>,
                                   DispatchPolicy_> ||
        AscendC::Std::is_base_of_v<MatmulMultiBlockWithStreamK<MatMulL0C2Out::ND_FIXPIPE_1_2, OP_TYPE_MUL>,
                                   DispatchPolicy_>>> {
public:
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
    uint64_t m_;
    uint64_t n_;
    uint64_t k_;
    uint64_t skKSingleCore_{1};
    uint64_t blkK_{1};
    uint64_t mL1_{1};
    uint64_t nL1_{1};
    uint64_t kL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};

    bool isBias_{false};
    bool isSplitSingleK_{false};
    bool isFirstSplitK_{false};
    bool isEndSplitK_{false};
    constexpr static uint64_t BUFFER_NUM = 2;
    constexpr static uint64_t HALF_L0_SIZE = L0A_SIZE / BUFFER_NUM / sizeof(A_T);
    // C0_SIZE equals 8 in order to adapt to the fp32 matrix
    constexpr static int32_t C0_SIZE = AscendC::AuxGetC0Size<typename AType::T>();
    constexpr static int32_t BIAS_C0 = AscendC::AuxGetC0Size<typename BiasType::T>();
    constexpr static uint64_t halfL0Size_ = L0AUF_SIZE / BUFFER_NUM / sizeof(A_T);
    // Set unitflag state: 3 = final accumulation, 2 = non-final accumulation
    constexpr static uint32_t FINAL_ACCUMULATION = 3;
    constexpr static uint32_t NON_FINAL_ACCUMULATION = 2;
    uint64_t abL1LoopCnt_{0};
    uint64_t l0PingPong_{0};

    __aicore__ inline BlockMmad()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(FIRST_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SECOND_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(THIRD_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(FOURTH_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(FIFTH_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SIXTH_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SEVENTH_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
    }

    __aicore__ inline ~BlockMmad()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(FIRST_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SECOND_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(THIRD_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(FOURTH_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(FIFTH_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SIXTH_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SEVENTH_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
    }

public:
    __aicore__ inline void Init(const TupleShape& shape, const TupleShape& tileL1, const TupleShape& tileL0,
                                bool isBias, bool isSplitSingleK = false)
    {
        m_ = Get<DIMENSION_M>(shape);
        n_ = Get<DIMENSION_N>(shape);
        k_ = Get<DIMENSION_K>(shape);

        mL1_ = Get<DIMENSION_M>(tileL1);
        nL1_ = Get<DIMENSION_N>(tileL1);
        kL1_ = Get<DIMENSION_K>(tileL1);

        baseM_ = Get<DIMENSION_M>(tileL0);
        baseN_ = Get<DIMENSION_N>(tileL0);
        baseK_ = Get<DIMENSION_K>(tileL0);
        isBias_ = isBias;
        isSplitSingleK_ = isSplitSingleK;
        // init tensor
        if (isBias_) {
            biasL1Offset_ = nL1_ * sizeof(Bias_T) * BUFFER_NUM;
        }
        aL1OneBuffer_ = mL1_ * kL1_;
        bL1Init_ = biasL1Offset_ + aL1OneBuffer_ * BUFFER_NUM;
        bL1OneBuffer_ = nL1_ * kL1_;
        l0PingPong_ = 0;
        abL1LoopCnt_ = 0;
    }
    // For FP32: L1 copy needs no modification
    __aicore__ inline void CopyInA1(const AscendC::GlobalTensor<A_T>& aGlobal,
                                    const AscendC::LocalTensor<A_T>& al1Local, uint64_t curML1, uint64_t curKL1)
    {
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        uint64_t nDim = AType::isTrans ? curKL1 : curML1;
        uint64_t dDim = AType::isTrans ? curML1 : curKL1;

        nd2nzParams.nValue = nDim;
        nd2nzParams.dValue = dDim;
        nd2nzParams.srcNdMatrixStride = 1;
        nd2nzParams.srcDValue = AType::isTrans ? m_ : k_;
        nd2nzParams.dstNzC0Stride = (nDim + AscendC::BLOCK_CUBE - 1) / AscendC::BLOCK_CUBE * AscendC::BLOCK_CUBE;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 1;
        AscendC::DataCopy(al1Local, aGlobal, nd2nzParams);
    }

    template <CubeFormat LayoutB = CubeFormat::ND>
    __aicore__ inline void CopyInB1(const AscendC::GlobalTensor<B_T>& bGlobal,
                                    const AscendC::LocalTensor<B_T>& bl1Local, uint64_t curNL1, uint64_t curKL1)
    {
        if constexpr (LayoutB == CubeFormat::ND) {
            AscendC::Nd2NzParams nd2nzParams;
            nd2nzParams.ndNum = 1;
            uint64_t nDim = BType::isTrans ? curNL1 : curKL1;
            uint64_t dDim = BType::isTrans ? curKL1 : curNL1;

            nd2nzParams.nValue = nDim;
            nd2nzParams.dValue = dDim;
            nd2nzParams.srcNdMatrixStride = 1;
            nd2nzParams.srcDValue = BType::isTrans ? k_ : n_;
            nd2nzParams.dstNzC0Stride = (nDim + AscendC::BLOCK_CUBE - 1) / AscendC::BLOCK_CUBE * AscendC::BLOCK_CUBE;
            nd2nzParams.dstNzNStride = 1;
            nd2nzParams.dstNzMatrixStride = 1;
            AscendC::DataCopy(bl1Local, bGlobal, nd2nzParams);
        } else {
            AscendC::DataCopyExtParams dataCopyParams;
            uint64_t nDim = BType::isTrans ? curNL1 : curKL1;
            uint64_t dDim = BType::isTrans ? curKL1 : curNL1;
            uint64_t nkDim = BType::isTrans ? n_ : k_;
            dataCopyParams.blockCount = Cmct::Gemm::CeilDiv(dDim, C0_SIZE);
            dataCopyParams.blockLen = Cmct::Gemm::Align(nDim, AscendC::BLOCK_CUBE) * C0_BYTE_SIZE;
            dataCopyParams.srcStride = (Cmct::Gemm::Align(nkDim, AscendC::BLOCK_CUBE) -
                                        Cmct::Gemm::Align(nDim, AscendC::BLOCK_CUBE)) *
                                       C0_BYTE_SIZE;
            dataCopyParams.dstStride = 0;
            AscendC::DataCopyPadExtParams<B_T> padParams{false, 0, 0, 0};
            AscendC::DataCopyPad(bl1Local, bGlobal, dataCopyParams, padParams);
        }
    }

    __aicore__ inline void CopyInC1(const AscendC::GlobalTensor<Bias_T>& biasGlobal,
                                    const AscendC::LocalTensor<Bias_T>& cl1Local, uint64_t curNL1)
    {
        AscendC::DataCopyPadParams padParams;
        AscendC::DataCopyParams biasParam{1, static_cast<uint16_t>(curNL1 * sizeof(Bias_T)), 0, 0};
        AscendC::DataCopyPad(cl1Local, biasGlobal, biasParam, padParams);
    }

    __aicore__ inline void CopyInC2(const AscendC::LocalTensor<Bias_T>& BiasL1Local,
                                    const AscendC::LocalTensor<float>& biasBt, uint64_t nl1Align, bool needBias)
    {
        if (!needBias) {
            return;
        }
        // fp32 need to two parts merge, so align to (nl1Align / 8, 2)
        uint64_t btAlign = AscendC::BLOCK_CUBE / BIAS_C0;
        uint16_t burstLength = Cmct::Gemm::Align(nl1Align / BIAS_C0, btAlign);
        AscendC::DataCopyParams biasParam{1, static_cast<uint16_t>(burstLength), 0, 0};
        // As dstlocal is located in C2, at least fp32 * 16 in C2
        AscendC::DataCopy(biasBt, BiasL1Local, biasParam);
    }

    __aicore__ inline void CopyInA2(const AscendC::LocalTensor<A_T>& a2Local, const AscendC::LocalTensor<A_T>& al1Local,
                                    uint64_t curML1, uint64_t curKL1, uint64_t kL0)
    {
        if constexpr (!AType::isTrans) {
            // (M, K) use LoadData2D
            AscendC::LoadData2DParamsV2 loadDataParams;
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = 0;
            loadDataParams.mStep = Cmct::Gemm::CeilDiv(curML1, AscendC::BLOCK_CUBE);
            if constexpr (AscendC::IsSameType<A_T, half>::value || AscendC::IsSameType<A_T, bfloat16_t>::value) {
                loadDataParams.kStep = Cmct::Gemm::CeilDiv(kL0, AscendC::BLOCK_CUBE);
            } else {
                loadDataParams.kStep = Cmct::Gemm::CeilDiv(kL0, C0_SIZE);
            }
            loadDataParams.srcStride = loadDataParams.mStep;
            loadDataParams.dstStride = loadDataParams.mStep;
            loadDataParams.ifTranspose = false;
            if constexpr (AscendC::IsSameType<A_T, bfloat16_t>::value) {
                AscendC::LoadData(a2Local, al1Local, loadDataParams);
            } else {
                AscendC::LoadData<A_T>(a2Local, al1Local, loadDataParams);
            }
        } else {
            // (K, M)
            AscendC::LoadData2DParamsV2 loadDataParams;
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = 0;
            loadDataParams.mStep = Cmct::Gemm::CeilDiv(kL0, AscendC::BLOCK_CUBE);
            if constexpr (AscendC::IsSameType<A_T, half>::value || AscendC::IsSameType<A_T, bfloat16_t>::value) {
                loadDataParams.kStep = Cmct::Gemm::CeilDiv(curML1, AscendC::BLOCK_CUBE);
                loadDataParams.dstStride = loadDataParams.kStep;
            } else {
                // actually div 8 then align to 2
                loadDataParams.kStep = Cmct::Gemm::CeilDiv(curML1, AscendC::BLOCK_CUBE) * TWO_ALIGN;
                loadDataParams.dstStride = loadDataParams.kStep >> 1;
            }
            loadDataParams.srcStride = Cmct::Gemm::CeilDiv(curKL1, AscendC::BLOCK_CUBE);
            loadDataParams.ifTranspose = true;
            if constexpr (AscendC::IsSameType<A_T, bfloat16_t>::value) {
                AscendC::LoadData(a2Local, al1Local, loadDataParams);
            } else {
                AscendC::LoadData<A_T>(a2Local, al1Local, loadDataParams);
            }
        }
    }

    __aicore__ inline void CopyInB2(const AscendC::LocalTensor<B_T>& b2Local, const AscendC::LocalTensor<B_T>& bl1Local,
                                    uint64_t curNL1, uint64_t curKL1, uint64_t kL0)
    {
        if constexpr (BType::isTrans) {
            // (N, K) use LoadData2D
            AscendC::LoadData2DParamsV2 loadDataParams;
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = 0;
            loadDataParams.mStep = Cmct::Gemm::CeilDiv(curNL1, AscendC::BLOCK_CUBE);
            if constexpr (AscendC::IsSameType<A_T, half>::value || AscendC::IsSameType<A_T, bfloat16_t>::value) {
                loadDataParams.kStep = Cmct::Gemm::CeilDiv(kL0, AscendC::BLOCK_CUBE);
            } else {
                loadDataParams.kStep = Cmct::Gemm::CeilDiv(kL0, C0_SIZE);
            }
            loadDataParams.srcStride = loadDataParams.mStep;
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
                loadDataParams.kStep = Cmct::Gemm::CeilDiv(curNL1, AscendC::BLOCK_CUBE);
                loadDataParams.dstStride = loadDataParams.kStep;
            } else {
                loadDataParams.kStep = Cmct::Gemm::CeilDiv(curNL1, AscendC::BLOCK_CUBE) * TWO_ALIGN;
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

    __aicore__ inline void CopyOut(const AscendC::GlobalTensor<C_T>& cGlobal,
                                   const AscendC::GlobalTensor<float>& workspaceGlobal,
                                   const AscendC::LocalTensor<float>& c1Local, uint64_t baseM, uint64_t baseN,
                                   bool checkIsSkScene)
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
        if (checkIsSkScene) {
            // set mode to float32, then cast in ub
            intriParams.quantPre = QuantMode_t::NoQuant;
            if constexpr (DispatchPolicy::fixpOpti_ == MatMulL0C2Out::ND_FIXPIPE_1_2) {
                intriParams.dstStride = Cmct::Gemm::Align(baseN, AscendC::ONE_BLK_SIZE);
            } else {
                intriParams.dstStride = baseN;
            }
        } else {
            // set mode according to dtype
            if constexpr (AscendC::IsSameType<C_T, bfloat16_t>::value) {
                intriParams.quantPre = QuantMode_t::F322BF16;
            } else if (AscendC::IsSameType<C_T, half>::value) {
                intriParams.quantPre = QuantMode_t::F322F16;
            } else if (AscendC::IsSameType<C_T, float>::value) {
                intriParams.quantPre = QuantMode_t::NoQuant;
            }
            intriParams.dstStride = n_;
        }
        intriParams.srcStride = Cmct::Gemm::Align(baseM, AscendC::BLOCK_CUBE);
        intriParams.nz2ndEn = true;
        intriParams.unitFlag = FINAL_ACCUMULATION;
        AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
        if (checkIsSkScene) {
            AscendC::DataCopy(workspaceGlobal, c1Local, intriParams);
        } else {
            AscendC::DataCopy(cGlobal, c1Local, intriParams);
        }
        if (isSplitSingleK_ && isEndSplitK_) {
            AscendC::DisableDmaAtomic();
        }
    }

    template <CubeFormat LayoutB = CubeFormat::ND>
    __aicore__ inline void operator()(AscendC::GlobalTensor<C_T> cGlobal, AscendC::GlobalTensor<A_T> aGlobal,
                                      AscendC::GlobalTensor<B_T> bGlobal, AscendC::GlobalTensor<Bias_T> biasGlobal,
                                      AscendC::GlobalTensor<float> workspaceGlobal, TupleShape tileShape,
                                      int64_t kCntIndex, bool checkIsSkScene, uint64_t blkK = 0,
                                      bool isFirstSplitK = false, bool isEndSplitK = false)
    {
        // mL1_ == ml0, nL1_ == nl0
        uint64_t curML1 = Get<0>(tileShape);
        uint64_t curNL1 = Get<1>(tileShape);
        skKSingleCore_ = Get<2>(tileShape);
        blkK_ = blkK;
        isFirstSplitK_ = isFirstSplitK;
        isEndSplitK_ = isEndSplitK;
        uint64_t curKL1Iter = (blkK_ + kL1_ - 1) / kL1_;
        uint64_t ml1Align = Cmct::Gemm::Align(curML1, AscendC::BLOCK_CUBE);
        uint64_t nl1Align = Cmct::Gemm::Align(curNL1, AscendC::BLOCK_CUBE);
        AscendC::MmadParams mmadParams;
        mmadParams.m = curML1;
        mmadParams.n = curNL1;
        mmadParams.disableGemv = true;
        AscendC::LocalTensor<Bias_T> BiasL1Local = l1Local_.template ReinterpretCast<Bias_T>();
        AscendC::LocalTensor<B_T> bl1Local;

        for (uint64_t iter0 = 0; iter0 < curKL1Iter; ++iter0) {
            uint64_t curKL1 = (iter0 + 1 == curKL1Iter) ? (blkK_ - iter0 * kL1_) : kL1_;
            // switch on pingpong, now only support double buffer in streamk
            uint64_t l1BufId = abL1LoopCnt_ & (BUFFER_NUM - 1);
            uint64_t offsetA = AType::isTrans ? iter0 * kL1_ * m_ : iter0 * kL1_;
            uint64_t offsetAl1 = biasL1Offset_ + aL1OneBuffer_ * l1BufId;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);

            uint64_t BiasBufId = abL1LoopCnt_ & 0x1;
            if (isBias_ && iter0 == 0 && kCntIndex == 0 && isFirstSplitK_) {
                CopyInC1(biasGlobal, BiasL1Local[nL1_ * l1BufId], curNL1);
            }

            CopyInA1(aGlobal[offsetA], l1Local_[offsetAl1], curML1, curKL1);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

            bl1Local = l1Local_[bL1Init_ + bL1OneBuffer_ * l1BufId];
            uint64_t offsetB = BType::isTrans ? iter0 * kL1_ : iter0 * kL1_ * n_;
            if constexpr (LayoutB == CubeFormat::NZ) {
                if constexpr (BType::isTrans) {
                    offsetB = iter0 * kL1_ * Cmct::Gemm::Align(n_, AscendC::BLOCK_CUBE);
                } else {
                    offsetB = iter0 * kL1_ * C0_SIZE;
                }
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId + L1_EVENT_ID_OFFSET);

            CopyInB1<LayoutB>(bGlobal[offsetB], bl1Local, curNL1, curKL1);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId + L1_EVENT_ID_OFFSET);

            uint64_t kL0Iter = (curKL1 + baseK_ - 1) / baseK_;
            for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                uint64_t curK0 = (iter1 + 1 == kL0Iter) ? (curKL1 - iter1 * baseK_) : baseK_;
                uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
                CopyInA2(l0aLocal_[l0Offset], l1Local_[offsetAl1], curML1, curKL1, curK0);
                offsetAl1 += AType::isTrans ? baseK_ * C0_SIZE : ml1Align * baseK_;
                if (iter1 == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId + L1_EVENT_ID_OFFSET);
                }

                // copy bias to biastable
                CopyInC2(BiasL1Local[nL1_ * l1BufId], biasBt_[nL1_ * BiasBufId], nl1Align,
                         NeedBias(iter0, iter1, kCntIndex));

                uint64_t offsetBl1 = BType::isTrans ? iter1 * baseK_ * nl1Align : iter1 * baseK_ * C0_SIZE;
                CopyInB2(l0bLocal_[l0Offset], bl1Local[offsetBl1], curNL1, curKL1, curK0);

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
                mmadParams.k = curK0;
                mmadParams.unitFlag = (iter0 + 1 == curKL1Iter && iter1 + 1 == kL0Iter) ? FINAL_ACCUMULATION :
                                                                                          NON_FINAL_ACCUMULATION;
                if (isSplitSingleK_) {
                    mmadParams.cmatrixInitVal = (iter0 == 0 && iter1 == 0 &&
                                                 (!isBias_ || (isBias_ && !(kCntIndex == 0 && isFirstSplitK_))));
                } else {
                    mmadParams.cmatrixInitVal = (iter0 == 0 && iter1 == 0 && (!isBias_ || (isBias_ && kCntIndex != 0)));
                }

                if (NeedBias(iter0, iter1, kCntIndex)) {
                    mmadParams.cmatrixSource = true;
                    AscendC::Mmad(c1Local_, l0aLocal_[l0Offset], l0bLocal_[l0Offset],
                                  biasBt_[nL1_ * (abL1LoopCnt_ & 0x1)], mmadParams);
                } else {
                    mmadParams.cmatrixSource = false;
                    AscendC::Mmad(c1Local_, l0aLocal_[l0Offset], l0bLocal_[l0Offset], mmadParams);
                }

                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
                l0PingPong_++;
            }
            if (iter0 + 1 == curKL1Iter) {
                CopyOut(cGlobal, workspaceGlobal, c1Local_, curML1, curNL1, checkIsSkScene); // Copy out to GM
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId + L1_EVENT_ID_OFFSET);
            abL1LoopCnt_++;
        }
    }

    __aicore__ inline bool NeedBias(uint64_t kIter0, uint64_t kIter1, int64_t kCntIndex)
    {
        if (isSplitSingleK_) {
            return isBias_ && kIter0 == 0 && kIter1 == 0 && kCntIndex == 0 && isFirstSplitK_;
        } else {
            return isBias_ && kIter0 == 0 && kIter1 == 0 && kCntIndex == 0;
        }
    }

private:
    constexpr static uint16_t L1_EVENT_ID_OFFSET = 4;
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
    constexpr static int32_t BT_SIZE = 4096;
    uint64_t biasL1Offset_ = 0;
    uint64_t bL1Init_ = 0;
    uint64_t aL1OneBuffer_ = 0;
    uint64_t bL1OneBuffer_ = 0;
    AscendC::LocalTensor<A_T> l0aLocal_{AscendC::TPosition::A2, 0, L0A_SIZE};
    AscendC::LocalTensor<B_T> l0bLocal_{AscendC::TPosition::B2, 0, L0B_SIZE};
    AscendC::LocalTensor<float> c1Local_{AscendC::TPosition::CO1, 0, AscendC::TOTAL_L0C_SIZE};
    AscendC::LocalTensor<float> biasBt_{AscendC::TPosition::C2, 0, BT_SIZE};
    AscendC::LocalTensor<A_T> l1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE};
};
} // namespace Block
} // namespace Gemm
} // namespace Cmct
