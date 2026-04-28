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
 * \file block_mmad_mx_quant.h
 * \brief
 */

#ifndef MATMUL_BLOCK_MMAD_MX_QUANT_H
#define MATMUL_BLOCK_MMAD_MX_QUANT_H
#include "../utils/layout_utils.h"
#include "../utils/common_utils.h"
#include "../utils/tuple_utils.h"
#include "../policy/dispatch_policy.h"
#include "../tile/tile_copy.h"

namespace Act {
namespace Gemm {
namespace Block {
using namespace AscendC;

template <class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class AType_, class LayoutA_, class BType_,
          class LayoutB_, class CType_, class LayoutC_, class BiasType_, class LayoutBias_, class TileCopy_,
          class Enable = void>
class BlockMmadMx<DispatchPolicy_, L1TileShape_, L0TileShape_, AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_,
                  BiasType_, LayoutBias_, TileCopy_,> {
    static_assert(AscendC::Std::always_false_v<DispatchPolicy_>, "Should not be here!");
};

template <class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class AType_, class LayoutA_, class BType_,
          class LayoutB_, class CType_, class LayoutC_, class BiasType_, class LayoutBias_, class TileCopy_>
class BlockMmadMx<DispatchPolicy_, L1TileShape_, L0TileShape_, AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_,
                BiasType_, LayoutBias_, TileCopy_,
                AscendC::Std::enable_if_t<
                    AscendC::Std::is_base_of_v<MatmulWithScale<>, DispatchPolicy_> ||
                    AscendC::Std::is_base_of_v<MatmulWithScale<AscendC::Shape<_0, _0, _0, _0>, A_FULL_LOAD_MODE>,
                                               DispatchPolicy_>>> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using MxL0AType = typename AscendC::Conditional<AscendC::IsSameType<AType, fp8_e4m3fn_t>::value, mx_fp8_e4m3_t,
                                                    mx_fp8_e5m2_t>::type;
    using MxL0BType = typename AscendC::Conditional<AscendC::IsSameType<BType, fp8_e4m3fn_t>::value, mx_fp8_e4m3_t,
                                                    mx_fp8_e5m2_t>::type;
    using BiasType = BiasType_;
    using DispatchPolicy = DispatchPolicy_;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using TupleL1L0Shape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    uint64_t m_;
    uint64_t n_;
    uint64_t k_;
    uint64_t kAlign_;
    uint64_t l1BufNum_{1};
    uint64_t kL1Iter_{0};
    uint64_t mL1_{1};
    uint64_t nL1_{1};
    uint64_t kL1_{1};
    uint64_t scaleKL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    bool isBias_{false};
    static constexpr bool transA = TagToTrans<LayoutA>::value;
    static constexpr bool transB = TagToTrans<LayoutB>::value;
    constexpr static uint64_t BUFFER_NUM = 2;
    constexpr static uint64_t HALF_L0_SIZE = L0A_SIZE / DOUBLE_BUFFER_COUNT / sizeof(AType);
    constexpr static uint64_t HALF_L0C_SIZE = AscendC::TOTAL_L0C_SIZE / DOUBLE_BUFFER_COUNT / sizeof(float);
    constexpr static int32_t C0_SIZE = AscendC::AuxGetC0Size<AType>();
    constexpr static int32_t BIAS_C0 = AscendC::AuxGetC0Size<BiasType>();
    constexpr static uint64_t halfL0Size_ = L0AUF_SIZE / BUFFER_NUM / sizeof(AType);
    constexpr static uint64_t BLOCK_CUBE = 16UL;
    constexpr static uint64_t BLOCK_REDUCE_CUBE = 32UL;
    constexpr static uint64_t MXFP_GROUP_SIZE = 32UL;
    constexpr static uint64_t MXFP_DIVISOR_SIZE = 64UL;
    constexpr static uint64_t MXFP_MULTI_BASE_SIZE = 2;
    constexpr static uint64_t IDX_M_TILE_IDX = 0UL;
    constexpr static uint64_t IDX_N_TILE_IDX = 1UL;
    constexpr static uint64_t IDX_M_TAIL_SPLIT_TILE_IDX = 2UL;
    constexpr static uint64_t IDX_N_TAIL_SPLIT_TILE_IDX = 3UL;
    constexpr static uint64_t IDX_SCALE_A_OFFSET_IDX = 2UL;
    constexpr static uint64_t IDX_SCALE_B_OFFSET_IDX = 3UL;
    // Set unitflag state: 3 = final accumulation, 2 = non-final accumulation
    constexpr static uint32_t FINAL_ACCUMULATION = 3;
    constexpr static uint32_t NON_FINAL_ACCUMULATION = 2;
    uint64_t abL1LoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    bool enableL0cPingPong_{false};

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR pertokenScaleGmAddr{nullptr};
        GM_ADDR scaleGmAddr{nullptr};
    };

    struct Arguments {
        uint64_t m;
        uint64_t n;
        uint64_t k;
        uint64_t kL1;
        uint64_t scaleKL1;
        uint64_t baseM;
        uint64_t baseN;
        uint64_t baseK;
        uint64_t mTailTile;
        uint64_t nTailTile;
        uint64_t isBias;
        uint64_t l1BufNum;
        uint64_t dbL0C;
    };

    __aicore__ inline BlockMmadMx()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(FIRST_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SECOND_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(THIRD_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FIRST_FLAG);
    }

    __aicore__ inline ~BlockMmadMx()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(FIRST_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SECOND_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(THIRD_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FIRST_FLAG);
    }

public:
    __aicore__ inline void Init(const Arguments &args)
    {
        m_ = args.m;
        n_ = args.n;
        k_ = args.k;
        kL1_ = args.kL1;
        scaleKL1_ = args.scaleKL1;
        baseM_ = args.baseM;
        baseN_ = args.baseN;
        baseK_ = args.baseK;
        mTailTile_ = args.mTailTile;
        nTailTile_ = args.nTailTile;
        kAlign_ = Act::Gemm::Align(k_, AscendC::BLOCK_CUBE);
        isBias_ = args.isBias == 1;
        l1BufNum_ = args.l1BufNum;
        enableL0cPingPong_ = args.dbL0C != 1;
        if constexpr (DispatchPolicy::fullLoadMode == 0) {
            aL1OneBuffer_ = baseM_ * kL1_;
            bL1Init_ = aL1OneBuffer_ * l1BufNum_;
            scaleAL1OneBuffer_ = baseM_ * Act::Gemm::CeilDiv(k_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        } else {
            uint64_t mAlign = Act::Gemm::Align(m_, transA ? BLOCK_REDUCE_CUBE : BLOCK_CUBE);
            uint64_t kAlign = Act::Gemm::Align(k_, MXFP_DIVISOR_SIZE);
            aL1OneBuffer_ = mAlign * kAlign;
            bL1Init_ = aL1OneBuffer_;
            scaleAL1OneBuffer_ = baseM_ * Act::Gemm::CeilDiv(kL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        }
        bL1OneBuffer_ = baseN_ * kL1_;
        scaleBL1OneBuffer_ = baseN_ * Act::Gemm::CeilDiv(kL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        kL1Iter_ = CeilDiv(k_, kL1_);
        l0PingPong_ = 0;
        abL1LoopCnt_ = 0;
        l0cPingPong_ = 0;
    }

    __aicore__ inline void CopyInA1(const AscendC::GlobalTensor<AType> &aGlobal,
        const AscendC::LocalTensor<AType> &al1Local, uint64_t curML1, uint64_t curKL1)
    {
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        uint64_t nDim = transA ? curKL1 : curML1;
        uint64_t dDim = transA ? curML1 : curKL1;

        nd2nzParams.nValue = nDim;
        nd2nzParams.dValue = dDim;
        nd2nzParams.srcNdMatrixStride = 1;
        nd2nzParams.srcDValue = transA ? m_ : k_;
        nd2nzParams.dstNzC0Stride = (nDim + AscendC::BLOCK_CUBE - 1) / AscendC::BLOCK_CUBE * AscendC::BLOCK_CUBE;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 1;
        AscendC::DataCopy(al1Local, aGlobal, nd2nzParams);
    }

    __aicore__ inline void CopyInB1(const AscendC::GlobalTensor<BType> &bGlobal,
        const AscendC::LocalTensor<BType> &bl1Local, uint64_t curNL1, uint64_t curKL1)
    {
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        uint64_t nDim = transB ? curNL1 : curKL1;
        uint64_t dDim = transB ? curKL1 : curNL1;

        nd2nzParams.nValue = nDim;
        nd2nzParams.dValue = dDim;
        nd2nzParams.srcNdMatrixStride = 1;
        nd2nzParams.srcDValue = transB ? k_ : n_;
        nd2nzParams.dstNzC0Stride = (nDim + AscendC::BLOCK_CUBE - 1) / AscendC::BLOCK_CUBE * AscendC::BLOCK_CUBE;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 1;
        AscendC::DataCopy(bl1Local, bGlobal, nd2nzParams);
    }

    __aicore__ inline void InitA1(const AscendC::LocalTensor<AType> &al1Local, uint64_t curML1, uint64_t curKL1)
    {
        if (curKL1 % MXFP_DIVISOR_SIZE == 0) {
            return;
        }
        AscendC::LocalTensor<half> al1LocalHalf = al1Local.template ReinterpretCast<half>();
        if constexpr (!transA) {
            uint64_t mAlign = Act::Gemm::Align(curML1, AscendC::BLOCK_CUBE);
            uint64_t kAlign = Act::Gemm::CeilDiv(curKL1, C0_SIZE) * AscendC::BLOCK_CUBE;
            uint64_t offset = mAlign * kAlign;
            AscendC::Duplicate<half>(al1LocalHalf[offset], 0.0, mAlign * AscendC::BLOCK_CUBE);
        } else {
            uint64_t m1 = Act::Gemm::CeilDiv(curML1, C0_SIZE);
            uint64_t kL1Aligned = Act::Gemm::Align(curKL1, AscendC::BLOCK_CUBE);
            uint64_t dumpCnt = (kL1Aligned - curKL1) * AscendC::BLOCK_CUBE;
            uint64_t offset = curKL1 * AscendC::BLOCK_CUBE;
            uint64_t offsetPer = kL1Aligned * AscendC::BLOCK_CUBE;
            for (uint64_t mIndex1 = 0; mIndex1 < m1; mIndex1++) {
                AscendC::Duplicate<half>(al1LocalHalf[offset], 0.0, dumpCnt);
                offset = offset + offsetPer;
            }
        }
    }

    __aicore__ inline void InitB1(const AscendC::LocalTensor<AType> &bl1Local, uint64_t curNL1, uint64_t curKL1)
    {
        if (curKL1 % MXFP_DIVISOR_SIZE == 0) {
            return;
        }
        AscendC::LocalTensor<half> bl1LocalHalf = bl1Local.template ReinterpretCast<half>();
        if constexpr (transB) {
            uint64_t nAlign = Act::Gemm::Align(curNL1, AscendC::BLOCK_CUBE);
            uint64_t kAlign = Act::Gemm::CeilDiv(curKL1, C0_SIZE) * AscendC::BLOCK_CUBE;
            uint64_t offset = nAlign * kAlign;
            AscendC::Duplicate<half>(bl1LocalHalf[offset], 0.0, nAlign * AscendC::BLOCK_CUBE);
        } else {
            uint64_t n1 = Act::Gemm::CeilDiv(curNL1, C0_SIZE);
            uint64_t kL1Aligned = Act::Gemm::Align(curKL1, AscendC::BLOCK_CUBE);
            uint64_t dumpCnt = (kL1Aligned - curKL1) * AscendC::BLOCK_CUBE;
            uint64_t offset = curKL1 * AscendC::BLOCK_CUBE;
            uint64_t offsetPer = kL1Aligned * AscendC::BLOCK_CUBE;
            for (uint64_t nIndex1 = 0; nIndex1 < n1; nIndex1++) {
                AscendC::Duplicate<half>(bl1LocalHalf[offset], 0.0, dumpCnt);
                offset = offset + offsetPer;
            }
        }
    }

    __aicore__ inline void CopyInBias(const AscendC::GlobalTensor<BiasType> &biasGlobal,
                                      const AscendC::LocalTensor<BiasType> &cl1Local, uint64_t curNL1)
    {
        AscendC::DataCopyPadParams padParams;
        // 单位为Byte
        AscendC::DataCopyParams biasParam{1, static_cast<uint16_t>(curNL1 * sizeof(BiasType)), 0, 0};
        AscendC::DataCopyPad(cl1Local, biasGlobal, biasParam, padParams);
    }

    __aicore__ inline void CopyInScaleA(const GlobalTensor<fp8_e8m0_t> &aScaleGlobal,
                                        LocalTensor<fp8_e8m0_t> &aScaleL1Local, uint64_t curML1, uint64_t kL1Offset,
                                        uint64_t offsetScaleA)
    {
        if (DispatchPolicy::fullLoadMode != 0 && kL1Offset != 0) {
            return;
        }
        uint64_t curScaleKL1 = scaleKL1_;
        if (kL1Offset + curScaleKL1 > k_) {
            curScaleKL1 = k_ - kL1Offset;
        }
        uint64_t nDim = transA ? curScaleKL1 / MXFP_GROUP_SIZE : curML1;
        uint64_t dDim = transA ? curML1 : curScaleKL1 / MXFP_GROUP_SIZE;

        GlobalTensor<half> aScaleGlobalB16;
        aScaleGlobalB16.SetGlobalBuffer(((__gm__ half*)(aScaleGlobal.GetPhyAddr())), (nDim * dDim) >> 1);
        auto aScaleL1LocalImpl = aScaleL1Local.template ReinterpretCast<half>();

        if (!transA) {
            AscendC::Dn2NzParams dn2nzParams;
            dn2nzParams.dnNum = 1;
            dn2nzParams.dValue = nDim;
            dn2nzParams.nValue = Act::Gemm::CeilDiv(dDim, 2); // Combine 2 float8_e8m0 data into 1 half type
            dn2nzParams.srcDnMatrixStride = 0;
            dn2nzParams.srcDValue = (k_ / MXFP_GROUP_SIZE) >> 1;
            dn2nzParams.dstNzC0Stride = Act::Gemm::CeilDiv(scaleKL1_ / MXFP_GROUP_SIZE, 2); // Combine 2 float8_e8m0 data into 1 half type
            dn2nzParams.dstNzNStride = 1;
            dn2nzParams.dstNzMatrixStride = 0;
            AscendC::DataCopy(aScaleL1LocalImpl, aScaleGlobalB16[offsetScaleA >> 1], dn2nzParams);
        } else {
            AscendC::Nd2NzParams nd2nzParams;
            nd2nzParams.ndNum = 1;
            nd2nzParams.nValue = Act::Gemm::CeilDiv(nDim, 2); // Combine 2 float8_e8m0 data into 1 half type
            nd2nzParams.dValue = dDim;
            nd2nzParams.srcNdMatrixStride = 0;
            nd2nzParams.srcDValue = m_;
            nd2nzParams.dstNzC0Stride = Act::Gemm::CeilDiv(nDim, 2); // Combine 2 float8_e8m0 data into 1 half type
            nd2nzParams.dstNzNStride = 1;
            nd2nzParams.dstNzMatrixStride = 0;
            AscendC::DataCopy(aScaleL1LocalImpl, aScaleGlobalB16[offsetScaleA >> 1], nd2nzParams);
        }
    }

    __aicore__ inline void CopyInScaleB(const GlobalTensor<fp8_e8m0_t> &bScaleGlobal,
                                        LocalTensor<fp8_e8m0_t> &bScaleL1Local, uint64_t curNL1, uint64_t kL1Offset,
                                        uint64_t offsetScaleB)
    {
        uint64_t curScaleKL1 = scaleKL1_;
        if (kL1Offset + curScaleKL1 > k_) {
            curScaleKL1 = k_ - kL1Offset;
        }
        uint64_t nDim = transB ? curNL1 : curScaleKL1 / MXFP_GROUP_SIZE;
        uint64_t dDim = transB ? curScaleKL1 / MXFP_GROUP_SIZE : curNL1;

        GlobalTensor<half> bScaleGlobalB16;
        bScaleGlobalB16.SetGlobalBuffer(((__gm__ half*)(bScaleGlobal.GetPhyAddr())), (nDim * dDim) >> 1);
        auto bScaleL1LocalImpl = bScaleL1Local.template ReinterpretCast<half>();

        if (transB) {
            AscendC::Dn2NzParams dn2nzParams;
            dn2nzParams.dnNum = 1;
            dn2nzParams.dValue = nDim;
            dn2nzParams.nValue = Act::Gemm::CeilDiv(dDim, 2); // Combine 2 float8_e8m0 data into 1 half type
            dn2nzParams.srcDnMatrixStride = 0;
            dn2nzParams.srcDValue = (k_ / MXFP_GROUP_SIZE) >> 1;
            dn2nzParams.dstNzC0Stride = Act::Gemm::CeilDiv(scaleKL1_ / MXFP_GROUP_SIZE, 2); // Combine 2 float8_e8m0 data into 1 half type
            dn2nzParams.dstNzNStride = 1;
            dn2nzParams.dstNzMatrixStride = 0;
            AscendC::DataCopy(bScaleL1LocalImpl, bScaleGlobalB16[offsetScaleB >> 1], dn2nzParams);
        } else {
            AscendC::Nd2NzParams nd2nzParams;
            nd2nzParams.ndNum = 1;
            nd2nzParams.nValue = Act::Gemm::CeilDiv(nDim, 2); // Combine 2 float8_e8m0 data into 1 half type
            nd2nzParams.dValue = dDim;
            nd2nzParams.srcNdMatrixStride = 0;
            nd2nzParams.srcDValue = n_;
            nd2nzParams.dstNzC0Stride = Act::Gemm::CeilDiv(nDim, 2); // Combine 2 float8_e8m0 data into 1 half type
            nd2nzParams.dstNzNStride = 1;
            nd2nzParams.dstNzMatrixStride = 0;
            AscendC::DataCopy(bScaleL1LocalImpl, bScaleGlobalB16[offsetScaleB >> 1], nd2nzParams);
        }
    }

    __aicore__ inline void CopyInC2(const AscendC::LocalTensor<BiasType> &biasL1Local,
                                    const AscendC::LocalTensor<float> &biasBt, uint64_t nl1Align, bool needBias)
    {
        if (!needBias) {
            return;
        }
        // s32场景要对齐到2 因此是align(nl1Align / 8, 2)
        uint64_t btAlign = AscendC::BLOCK_CUBE / BIAS_C0;
        uint16_t bustLenth = Act::Gemm::Align(nl1Align / BIAS_C0, btAlign);
        AscendC::DataCopyParams biasParam{1, static_cast<uint16_t>(bustLenth), 0, 0};
        // 当dstlocal位于C2时，C2中至少为fp32*16
        AscendC::DataCopy(biasBt, biasL1Local, biasParam);
    }

    __aicore__ inline void CopyInL0A(const AscendC::LocalTensor<MxL0AType> &l0aLocal,
                                     const AscendC::LocalTensor<AType> &al1Local,
                                     const AscendC::LocalTensor<fp8_e8m0_t> &scaleAl1Local, uint64_t iter,
                                     uint64_t curML0, uint64_t curKL0, uint64_t curKL1)
    {
        AscendC::LoadData2DParamsV2 loadDataParams;
        if constexpr (!transA) {
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = Act::Gemm::CeilDiv(iter * baseK_, C0_SIZE);
            loadDataParams.mStep = Act::Gemm::CeilDiv(curML0, AscendC::BLOCK_CUBE);
            loadDataParams.kStep = Act::Gemm::CeilDiv(curKL0, C0_SIZE);
            loadDataParams.srcStride = loadDataParams.mStep;
            loadDataParams.dstStride = loadDataParams.mStep;
            loadDataParams.ifTranspose = false;
        } else {
            loadDataParams.mStartPosition = Act::Gemm::CeilDiv(iter * baseK_, AscendC::BLOCK_CUBE);
            loadDataParams.kStartPosition = 0;
            loadDataParams.mStep = Act::Gemm::CeilDiv(curKL0, AscendC::BLOCK_CUBE);
            loadDataParams.kStep = Act::Gemm::CeilDiv(curML0, C0_SIZE);
            loadDataParams.srcStride = Act::Gemm::CeilDiv(curKL1, AscendC::BLOCK_CUBE);
            loadDataParams.dstStride = Act::Gemm::CeilDiv(curKL0, AscendC::BLOCK_CUBE);
            loadDataParams.ifTranspose = true;
        }
        AscendC::LoadData2DMxParams loadData2DMxParams;
        loadData2DMxParams.xStartPosition = 0;
        loadData2DMxParams.yStartPosition = Act::Gemm::CeilDiv(iter * baseK_, MXFP_DIVISOR_SIZE);
        loadData2DMxParams.xStep = Act::Gemm::CeilDiv(curML0, AscendC::BLOCK_CUBE);
        loadData2DMxParams.yStep = Act::Gemm::CeilDiv(curKL0, MXFP_DIVISOR_SIZE);
        loadData2DMxParams.srcStride = Act::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE);
        loadData2DMxParams.dstStride = Act::Gemm::CeilDiv(curKL0, MXFP_DIVISOR_SIZE);
        AscendC::LoadData(l0aLocal, al1Local, scaleAl1Local, loadDataParams, loadData2DMxParams);
    }

    __aicore__ inline void CopyInL0B(const AscendC::LocalTensor<MxL0BType> &l0bLocal,
                                     const AscendC::LocalTensor<BType> &bl1Local,
                                     const AscendC::LocalTensor<fp8_e8m0_t> &scaleBl1Local, uint64_t iter,
                                     uint64_t curNL0, uint64_t curKL0, uint64_t curKL1)
    {
        AscendC::LoadData2DParamsV2 loadDataParams;
        if constexpr (transB) {
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = Act::Gemm::CeilDiv(iter * baseK_, C0_SIZE);
            loadDataParams.mStep = Act::Gemm::CeilDiv(curNL0, AscendC::BLOCK_CUBE);
            loadDataParams.kStep = Act::Gemm::CeilDiv(curKL0, C0_SIZE);
            loadDataParams.srcStride = loadDataParams.mStep;
            loadDataParams.dstStride = loadDataParams.mStep;
            loadDataParams.ifTranspose = true;
        } else {
            loadDataParams.mStartPosition = Act::Gemm::CeilDiv(iter * baseK_, AscendC::BLOCK_CUBE);
            loadDataParams.kStartPosition = 0;
            loadDataParams.mStep = Act::Gemm::CeilDiv(curKL0, AscendC::BLOCK_CUBE);
            loadDataParams.kStep = Act::Gemm::CeilDiv(curNL0, C0_SIZE);
            loadDataParams.srcStride = Act::Gemm::CeilDiv(curKL1, AscendC::BLOCK_CUBE);
            loadDataParams.dstStride = Act::Gemm::CeilDiv(curKL0, AscendC::BLOCK_CUBE);
            loadDataParams.ifTranspose = false;
        }
        AscendC::LoadData2DMxParams loadData2DMxParams;
        loadData2DMxParams.xStartPosition = 0;
        loadData2DMxParams.yStartPosition = Act::Gemm::CeilDiv(iter * baseK_, MXFP_DIVISOR_SIZE);
        loadData2DMxParams.xStep = Act::Gemm::CeilDiv(curNL0, AscendC::BLOCK_CUBE);
        loadData2DMxParams.yStep = Act::Gemm::CeilDiv(curKL0, MXFP_DIVISOR_SIZE);
        loadData2DMxParams.srcStride = Act::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE);
        loadData2DMxParams.dstStride = Act::Gemm::CeilDiv(curKL0, MXFP_DIVISOR_SIZE);
        AscendC::LoadData(l0bLocal, bl1Local, scaleBl1Local, loadDataParams, loadData2DMxParams);
    }

    __aicore__ inline void CopyOut(const AscendC::GlobalTensor<CType> &cGlobal, AscendC::LocalTensor<float> &c1Local,
                                   uint64_t baseM, uint64_t baseN)
    {
        AscendC::DataCopyCO12DstParams intriParams;
        intriParams.nSize = baseN;
        intriParams.mSize = baseM;
        intriParams.dstStride = n_;
        intriParams.srcStride = Act::Gemm::Align(baseM, AscendC::BLOCK_CUBE);
        // set mode according to dtype
        if constexpr (AscendC::IsSameType<CType, bfloat16_t>::value) {
            intriParams.quantPre = QuantMode_t::F322BF16;
        } else if (AscendC::IsSameType<CType, half>::value) {
            intriParams.quantPre = QuantMode_t::F322F16;
        } else if (AscendC::IsSameType<CType, float>::value) {
            intriParams.quantPre = QuantMode_t::NoQuant;
        }
        intriParams.nz2ndEn = true;
        intriParams.unitFlag = enableL0cPingPong_ ? 0 : FINAL_ACCUMULATION;  // 3 unitflag
        AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
        AscendC::DataCopy(cGlobal, c1Local, intriParams);
    }

    __aicore__ inline void operator()(AscendC::GlobalTensor<CType> cGlobal, AscendC::GlobalTensor<AType> aGlobal,
                                      AscendC::GlobalTensor<BType> bGlobal, AscendC::GlobalTensor<BiasType> biasGlobal,
                                      AscendC::GlobalTensor<fp8_e8m0_t> scaleAGlobal,
                                      AscendC::GlobalTensor<fp8_e8m0_t> scaleBGlobal, BlockShape singleShape,
                                      BlockOffset blockOffset, uint64_t nL1Offset)
    {
        uint64_t curML1 = Get<IDX_M_TILE_IDX>(singleShape);
        uint64_t curNL1 = Get<IDX_N_TILE_IDX>(singleShape);
        uint64_t curML0 = curML1;
        uint64_t curNL0 = curNL1;
        uint64_t ml1Align = Act::Gemm::Align(curML1, AscendC::BLOCK_CUBE);
        uint64_t nl1Align = Act::Gemm::Align(curNL1, AscendC::BLOCK_CUBE);
        uint64_t kbL1Size = kL1_;
        AscendC::MmadParams mmadParams;
        mmadParams.m = curML0;
        mmadParams.n = curNL0;
        mmadParams.disableGemv = true;
        AscendC::LocalTensor<BiasType> biasL1Local = l1Local_.template ReinterpretCast<BiasType>();
        AscendC::LocalTensor<BType> bl1Local;
        uint64_t kL1Offset = 0;
        uint64_t l0cOffset = (l0cPingPong_ & 1) * HALF_L0C_SIZE;
        if (enableL0cPingPong_) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cPingPong_ & 1);
        }
        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            uint64_t curKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - iter0 * kL1_) : kL1_;
            // Load data to L1 and open DB
            uint64_t l1BufId = abL1LoopCnt_ & (l1BufNum_ - 1);
            uint64_t offsetA = transA ? iter0 * kL1_ * m_ : iter0 * kL1_;
            uint64_t offsetAl1 = aL1OneBuffer_ * l1BufId;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            uint64_t BiasBufId = abL1LoopCnt_ & 1;

            if constexpr (DispatchPolicy::fullLoadMode == 0) {
                InitA1(l1Local_[offsetAl1], curML1, curKL1);
                CopyInA1(aGlobal[offsetA], l1Local_, curML1, curKL1);
            } else if (iter0 == 0) {
                InitA1(l1Local_, curML1, k_);
                CopyInA1(aGlobal, l1Local_, curML1, k_);
            }
            if (isBias_ && iter0 == 0) {
                biasL1Local = biasL1Local[bL1Init_];
                CopyInBias(biasGlobal, biasL1Local, curNL1);
                biasL1Offset_ = curNL1 * sizeof(BiasType);
            }
            uint64_t offsetBl1 = bL1Init_ + biasL1Offset_ + bL1OneBuffer_ * l1BufId;
            bl1Local = l1Local_[offsetBl1];
            uint64_t offsetB = transB ? iter0 * kL1_ : iter0 * kL1_ * n_;
            InitB1(l1Local_[offsetB], curNL1, curKL1);
            CopyInB1(bGlobal[offsetB], bl1Local, curNL1, curKL1);
            kbL1Size = curKL1;
            kL1Offset = iter0 * kL1_;
            uint64_t offsetScale = bL1Init_ + biasL1Offset_ + bL1OneBuffer_ * l1BufNum_;
            uint64_t offsetScaleA = offsetScale + scaleAL1OneBuffer_ * l1BufId;
            uint64_t offsetScaleB = offsetScale + scaleAL1OneBuffer_ * l1BufNum_ + scaleBL1OneBuffer_ * l1BufId;
            if (iter0 % (scaleKL1_ / kL1_) == 0) {
                AscendC::LocalTensor<fp8_e8m0_t> scaleAL1Local = l1Local_[offsetScaleA];
                AscendC::LocalTensor<fp8_e8m0_t> scaleBL1Local = l1Local_[offsetScaleB];
                CopyInScaleA(scaleAGlobal, scaleAL1Local, curML1, kL1Offset, Get<IDX_SCALE_A_OFFSET_IDX>(blockOffset));
                CopyInScaleB(scaleBGlobal, scaleBL1Local, curNL1, kL1Offset, Get<IDX_SCALE_B_OFFSET_IDX>(blockOffset));
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

            uint64_t kL0Iter = (curKL1 + baseK_ - 1) / baseK_;
            for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                uint64_t curK0 = (iter1 + 1 == kL0Iter) ? (curKL1 - iter1 * baseK_) : baseK_;
                // Load data to L0 and open DB
                uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
                CopyInL0A(l0aLocal_[l0Offset], l1Local_[offsetAl1], l1Local_[offsetScaleA], iter1, curML0, curK0,
                          curKL1);
                offsetAl1 += transA ? baseK_ * C0_SIZE : ml1Align * baseK_;
                // copy bias to bt
                CopyInC2(biasL1Local, biasBt_[baseN_ * BiasBufId], Act::Gemm::Align(mmadParams.n, AscendC::BLOCK_CUBE),
                         NeedBias(iter0, iter1));
                CopyInL0B(l0bLocal_[l0Offset], l1Local_[offsetBl1], l1Local_[offsetScaleB], iter0, curNL0, curK0,
                          curKL1);
                offsetBl1 += transB ? nl1Align * baseK_: baseK_ * C0_SIZE;

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
                mmadParams.k = curK0;
                mmadParams.unitFlag = enableL0cPingPong_
                                          ? 0
                                          : ((iter0 + 1 == kL1Iter_ && iter1 + 1 == kL0Iter) ? FINAL_ACCUMULATION
                                                                                             : NON_FINAL_ACCUMULATION);
                mmadParams.cmatrixInitVal = (iter0 == 0 && iter1 == 0 && !isBias_);
                Mmad(mmadParams, l0cOffset, l0Offset, nL1_ * (abL1LoopCnt_ & 0x1), NeedBias(iter0, iter1));
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
                l0PingPong_++;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            abL1LoopCnt_++;
        }
        // Copy out to GM
        AscendC::LocalTensor<float> c1Local = c1Local_[l0cOffset];
        if (enableL0cPingPong_) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0cPingPong_ & 1);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0cPingPong_ & 1);
        }
        // 数据搬出到GM或ub
        CopyOut(cGlobal, c1Local, mmadParams.m, mmadParams.n);
        if (enableL0cPingPong_) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0cPingPong_ & 1);
            l0cPingPong_++;
        }
    }

private:
    __aicore__ inline bool NeedBias(uint64_t kIter0, uint64_t kIter1)
    {
        return isBias_ && kIter0 == 0 && kIter1 == 0;
    }

    __aicore__ inline void Mmad(
        AscendC::MmadParams &mmadParams, uint64_t l0cOffset, uint64_t l0abOffset, uint64_t biasOffset, bool needBias)
    {
        mmadParams.cmatrixSource = needBias;
        if (needBias) {
            AscendC::Mmad(
                c1Local_[l0cOffset], l0aLocal_[l0abOffset], l0bLocal_[l0abOffset], biasBt_[biasOffset], mmadParams);
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
    constexpr static uint16_t M_ALIGN = 16;
    constexpr static uint16_t TWO_ALIGN = 2;
    constexpr static int32_t BT_SIZE = 4096;
    uint64_t biasL1Offset_ = 0;
    uint64_t bL1Init_ = 0;
    uint64_t scaleAL1Init_ = 0;
    uint64_t scaleBL1Init_ = 0;
    uint64_t aL1OneBuffer_ = 0;
    uint64_t bL1OneBuffer_ = 0;
    uint64_t scaleAL1OneBuffer_ = 0;
    uint64_t scaleBL1OneBuffer_ = 0;
    int64_t mTailTile_ = 0;
    int64_t nTailTile_ = 0;
    AscendC::LocalTensor<MxL0AType> l0aLocal_{AscendC::TPosition::A2, 0, L0A_SIZE};
    AscendC::LocalTensor<MxL0BType> l0bLocal_{AscendC::TPosition::B2, 0, L0B_SIZE};
    AscendC::LocalTensor<float> c1Local_{AscendC::TPosition::CO1, 0, AscendC::TOTAL_L0C_SIZE};
    AscendC::LocalTensor<float> biasBt_{AscendC::TPosition::C2, 0, BT_SIZE};
    AscendC::LocalTensor<AType> l1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE};
};
}  // namespace Block
}  // namespace Gemm
}  // namespace Act
#endif