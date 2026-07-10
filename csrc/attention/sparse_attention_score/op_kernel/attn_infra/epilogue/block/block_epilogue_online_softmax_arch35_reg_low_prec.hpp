/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_HPP

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/resource.hpp"
#include "../../../attn_infra/epilogue/dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/tile_common/tile_copy.hpp"
#include "../../../attn_infra/gemm_coord.hpp"
#include "../../../attn_infra/matrix_coord.hpp"
#include "../../../tla/tensor.hpp"
#include "../../../tla/layout.hpp"

namespace NpuArch::Epilogue::Block {

enum class KvBaseTileRegSplitStages {
    ONE,
    TWO
};

template <
    class OutputType_,
    class LayoutS_>
class BlockEpilogue<
    EpilogueOnlineSoftmaxBsa,
    OutputType_,
    Gemm::GemmType<half, LayoutS_>>
{
public:
    using DispatchPolicy = EpilogueOnlineSoftmaxBsa;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element;
    using ElementInput = half;

    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = LayoutS_;

    static constexpr uint32_t BLOCK_SIZE_IN_BYTE = 32;
    static constexpr uint32_t REPEAT_SIZE_IN_BYTE = 256;
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t HALF_VECTOR_SIZE = 128;
    static constexpr uint32_t BLOCK_SIZE = 16;
    static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 32768;
    static constexpr uint32_t VECTOR_SIZE = 128;
    static constexpr uint32_t MAX_UB_S_ELEM_NUM = 16384;
    static constexpr uint32_t DM_UB_GLOBAL_ELEM_NUM = 64;
    static constexpr uint32_t ELE_NUM_PER_C0 = 16;
    static constexpr uint32_t ELE_NUM_PER_C0_FP8 = 32;

    static constexpr uint32_t REDUCE_UB_SIZE = 1024;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_32 = 32;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_8 = 8;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_4 = 4;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_2 = 2;
    static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;
    static constexpr int64_t UB_FLOAT_LINE_SIZE = 64;

    static constexpr uint32_t SPLIT_COL_IDX_2 = 2;
    static constexpr uint32_t SPLIT_COL_IDX_3 = 3;
    static constexpr ElementInput MIN_VALUE = -65504.0f;
    static constexpr uint32_t HALF_REP_SIZE = 128;
    static constexpr uint32_t FLOAT_REP_SIZE = 64;
    static constexpr uint32_t BLOCK_REP_SIZE = 8;
    static constexpr uint32_t REPEAT_STRIDE = 1;
    static constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
    static constexpr uint32_t SM_ROW_MAX_ELEM_NUM = 64;
    static constexpr uint32_t SM_COL_MAX_ELEM_NUM = 256;
    static constexpr uint32_t SM_VREG_SIZE = 256 / sizeof(ElementInput);

    static constexpr bool FULL_QUANT_FP8 = AscendC::IsSameType<ElementOutput, fp8_e4m3fn_t>::value;

    __aicore__ inline
    BlockEpilogue(Arch::Resource<ArchTag> &resource, float scaleValue_)
    {
        // Allocate UB space
        constexpr uint32_t LS_UB_TENSOR_OFFSET = 0;
        constexpr uint32_t LP_UB_TENSOR_OFFSET = 2 * UB_UINT8_BLOCK_SIZE;

        constexpr uint32_t LM_UB_TENSOR_OFFSET = 7 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t GM_UB_TENSOR_OFFSET = LM_UB_TENSOR_OFFSET + 64 * sizeof(float);
        constexpr uint32_t DM_UB_TENSOR_OFFSET = GM_UB_TENSOR_OFFSET + 64 * sizeof(float);
        constexpr uint32_t LL_UB_TENSOR_OFFSET = DM_UB_TENSOR_OFFSET + 3 * 64 * sizeof(float);
        constexpr uint32_t GL_UB_TENSOR_OFFSET = LL_UB_TENSOR_OFFSET +  64 * sizeof(float);

        subBlockIdx_ = AscendC::GetSubBlockIdx();

        scaleValue = static_cast<ElementInput>(scaleValue_);
        lsUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(LS_UB_TENSOR_OFFSET);
        lpUbTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(LP_UB_TENSOR_OFFSET);
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
        dmUbTensor = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
        lmUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(LM_UB_TENSOR_OFFSET);
        llUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(LL_UB_TENSOR_OFFSET);
    }

    __aicore__ inline
    ~BlockEpilogue()
    {
    }

    template <class TensorDst, class TensorSrc>
    __aicore__ inline
    void CopyPUbToPL1(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint32_t m)
    {
        const uint32_t blockCount = tla::get<1, 1>(srcTensor.shape());
        const uint32_t blockLen = tla::get<0, 0>(srcTensor.shape()) * tla::get<0, 1>(srcTensor.shape());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());

        AscendC::DataCopyParams repeatParams;

        uint32_t elementNumPerC0;
        if constexpr (FULL_QUANT_FP8) {
            elementNumPerC0 = ELE_NUM_PER_C0_FP8;
        } else {
            elementNumPerC0 = ELE_NUM_PER_C0;
        }
        repeatParams.blockCount = blockCount;
        repeatParams.blockLen = m;
        repeatParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / elementNumPerC0 - m;
        repeatParams.dstStride = tla::get<1, 1>(dstTensor.stride()) / elementNumPerC0 - m;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams);
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void SetCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        // in mode 4, AIC set for 2 AIVs seperately
        if constexpr (MODE == 4U) {
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void WaitCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        // in mode 4, AIC wait for 2 AIVs seperately
        if constexpr (MODE == 4U) {
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
        }
    }
    
    template <class TensorP>
    __aicore__ inline
    void operator()(TensorP &l1PTensorTla, GemmCoord actualBlockShape,
        uint32_t isFirstKvSTile, uint32_t ubSBufId, uint32_t l1PBufId,
         Arch::CrossCoreFlag mm1ToSmFlag, Arch::CrossCoreFlag smToMm2Flag)
    {
        uint32_t mCopyOffset = RoundUp(actualBlockShape.m(), 8) / 2;
        uint32_t m = actualBlockShape.m() < mCopyOffset ? actualBlockShape.m() : mCopyOffset;
        m = subBlockIdx_ == 0 ? m : actualBlockShape.m() - m;
        if (m == 0) {
            WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
            SetCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
            WaitCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
            SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
            return;
        }
        uint32_t n = actualBlockShape.n();
        uint16_t mRound = RoundUp(m, C0_NUM_PER_FRACTAL);
        uint16_t nRound = RoundUp(n, ELE_NUM_PER_C0);
        uint32_t blockStride = mRound;
        constexpr int16_t vlSize = static_cast<int16_t>(AscendC::GetVecLen() / sizeof(ElementInput));
        int16_t nLoops = AscendC::CeilDivision(n, vlSize) - 1;
        uint32_t tailN = (n - 1) % vlSize + 1;
        int16_t mLoops = AscendC::CeilDivision(m, vlSize) - 1;
        uint32_t tailM = (m - 1) % vlSize + 1;
        uint32_t nPadding = (tailN + BLOCK_SIZE_IN_BYTE - 1) / BLOCK_SIZE_IN_BYTE * BLOCK_SIZE_IN_BYTE;
        __ubuf__ ElementOutput *pAddr = (__ubuf__ ElementOutput*) lpUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM].GetPhyAddr();
        __ubuf__ ElementInput *sAddr = (__ubuf__ ElementInput*) lsUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM].GetPhyAddr();
        __ubuf__ float *lastMaxAddr = (__ubuf__ float *)gmUbTensor.GetPhyAddr();
        __ubuf__ float *lastSumAddr = (__ubuf__ float*) glUbTensor.GetPhyAddr();
        __ubuf__ ElementInput *nowMaxAddr = (__ubuf__ ElementInput*) lmUbTensor.GetPhyAddr();
        __ubuf__ ElementInput *nowSumAddr = (__ubuf__ ElementInput*) llUbTensor.GetPhyAddr();
        __ubuf__ float *expMaxUbAddr = (__ubuf__ float *)dmUbTensor[l1PBufId * DM_UB_GLOBAL_ELEM_NUM].GetPhyAddr();

        // wait QK Fixpipe finsh
        WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
        uint32_t kvBaseTileRegStages = CeilDiv(n, SM_VREG_SIZE);
        if (kvBaseTileRegStages == 1) {
            ComputeScaleAndMax<KvBaseTileRegSplitStages::ONE>(
                sAddr, nowMaxAddr, m, tailN, nPadding, scaleValue, nRound);
        } else if (kvBaseTileRegStages == 2) {
            ComputeScaleAndMax<KvBaseTileRegSplitStages::TWO>(
                sAddr, nowMaxAddr, m, tailN, nPadding, scaleValue, nRound);
        }
        if (!isFirstKvSTile) {
            UpdateMax(nowMaxAddr, lastMaxAddr, mLoops, tailM);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        if (kvBaseTileRegStages == 1) {
            if constexpr (FULL_QUANT_FP8) {
                ComputeExpSubSum16FP8<KvBaseTileRegSplitStages::ONE>(pAddr, sAddr, nowMaxAddr, nowSumAddr, m, tailN,
                                                                     blockStride, nRound);

            } else {
                ComputeExpSubSum16<KvBaseTileRegSplitStages::ONE>(pAddr, sAddr, nowMaxAddr, nowSumAddr, m, tailN,
                                                                  blockStride, nRound);
            }
        } else if (kvBaseTileRegStages == 2) {
            if constexpr (FULL_QUANT_FP8) {
                ComputeExpSubSum16FP8<KvBaseTileRegSplitStages::TWO>(pAddr, sAddr, nowMaxAddr, nowSumAddr, m, tailN,
                                                                  blockStride, nRound);

            } else {
                ComputeExpSubSum16<KvBaseTileRegSplitStages::TWO>(pAddr, sAddr, nowMaxAddr, nowSumAddr, m, tailN,
                                                                     blockStride, nRound);
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        SetCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);

        uint32_t curNRound = FULL_QUANT_FP8 ? RoundUp(n, ELE_NUM_PER_C0_FP8) : RoundUp(n, ELE_NUM_PER_C0);
        auto ubPLayoutTla = tla::MakeLayout<ElementOutput, LayoutOutput>(mRound, curNRound);
        auto ubPTensorTla = tla::MakeTensor(lpUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM],
            ubPLayoutTla, Arch::PositionUB{});
        auto ubPTensorTlaTile = GetTile(ubPTensorTla,
                tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        auto l1PTensorTlaTile = GetTile(l1PTensorTla,
                tla::MakeCoord(subBlockIdx_ * mCopyOffset, 0), tla::MakeShape(m, n));
        WaitCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);

        CopyPUbToPL1(l1PTensorTlaTile, ubPTensorTlaTile, m);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        // crossCoreSync after PIPE_MTE1 move
        SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
        mLoops = AscendC::CeilDivision(m, 64) - 1;
        tailM = (m - 1) % 64 + 1;
        if (isFirstKvSTile) {
            CastExpSumAndExpMax<ElementInput>(
                lastSumAddr, lastMaxAddr, nowSumAddr, nowMaxAddr, mLoops, tailM);
        } else {
            UpdateExpSumAndExpMax<ElementInput>(
                lastSumAddr, expMaxUbAddr, lastMaxAddr, nowSumAddr, nowMaxAddr, mLoops, tailM);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

private:
    ElementInput scaleValue;
    AscendC::LocalTensor<ElementInput> lsUbTensor;
    AscendC::LocalTensor<ElementOutput> lpUbTensor;
    AscendC::LocalTensor<float> gmUbTensor;
    AscendC::LocalTensor<float> glUbTensor;
    AscendC::LocalTensor<float> dmUbTensor;
    AscendC::LocalTensor<ElementInput> lmUbTensor;
    AscendC::LocalTensor<ElementInput> llUbTensor;
    uint32_t subBlockIdx_;

    template <KvBaseTileRegSplitStages kvBaseTileRegSplitStages>
    __simd_vf__ inline void ComputeScaleAndMax(__ubuf__ ElementInput *srcUb, __ubuf__ ElementInput *newMaxUb,
        uint16_t m, uint32_t tailN, uint32_t nPadding, ElementInput dScale, uint16_t S2BaseSize)
    {
        static_assert(kvBaseTileRegSplitStages == KvBaseTileRegSplitStages::ONE ||
                      kvBaseTileRegSplitStages == KvBaseTileRegSplitStages::TWO,
                      "ComputeScaleAndMax only supports ONE Or TWO stages, please use the specialized versions.");
    }
    
    template <>
    __simd_vf__ inline void ComputeScaleAndMax<KvBaseTileRegSplitStages::ONE>(
        __ubuf__ ElementInput *srcUb, __ubuf__ ElementInput *newMaxUb,
        uint16_t m, uint32_t tailN, uint32_t nPadding, ElementInput dScale, uint16_t S2BaseSize)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<ElementInput> minVreg;
        RegTensor<ElementInput> srcVreg;
        RegTensor<ElementInput> maxTmpVreg;
        UnalignReg maxUreg;
        MaskReg pregCompare;
        MaskReg pregFull = CreateMask<ElementInput, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<ElementInput>(tailN);

        Duplicate(minVreg, MIN_VALUE);
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign(srcVreg, srcUb + i * S2BaseSize);
            Muls(srcVreg, srcVreg, dScale, pregFull);
            Select(srcVreg, srcVreg, minVreg, pregTailN);
            StoreAlign<ElementInput, StoreDist::DIST_NORM_B16>(
                srcUb + i * S2BaseSize, srcVreg, pregTailN);
            ReduceMax(maxTmpVreg, srcVreg, pregFull);
            StoreUnAlign<ElementInput, PostLiteral::POST_MODE_UPDATE>(newMaxUb, maxTmpVreg, maxUreg, 1);
        }
        vstas(maxUreg, newMaxUb, 0, POST_UPDATE);
    }

    template <>
    __simd_vf__ inline void ComputeScaleAndMax<KvBaseTileRegSplitStages::TWO>(
        __ubuf__ ElementInput *srcUb, __ubuf__ ElementInput *newMaxUb,
        uint16_t m, uint32_t tailN, uint32_t nPadding, ElementInput dScale, uint16_t S2BaseSize)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<ElementInput> minVreg;
        RegTensor<ElementInput> srcVreg0;
        RegTensor<ElementInput> srcVreg1;
        RegTensor<ElementInput> maxTmpVreg;
        UnalignReg maxUreg;
        MaskReg pregCompare;
        MaskReg pregFull = CreateMask<ElementInput, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<ElementInput>(tailN);

        Duplicate(minVreg, MIN_VALUE);
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign(srcVreg0, srcUb + i * S2BaseSize);
            LoadAlign(srcVreg1, srcUb + i * S2BaseSize + HALF_REP_SIZE);
            Muls(srcVreg0, srcVreg0, dScale, pregFull);
            Muls(srcVreg1, srcVreg1, dScale, pregFull);
            StoreAlign<ElementInput, StoreDist::DIST_NORM_B16>(
                srcUb + i * S2BaseSize, srcVreg0, pregFull);
            StoreAlign<ElementInput, StoreDist::DIST_NORM_B16>(
                srcUb + i * S2BaseSize + HALF_REP_SIZE, srcVreg1, pregTailN);
            Max<ElementInput, MaskMergeMode::MERGING>(srcVreg0, srcVreg0, srcVreg1, pregTailN);
            ReduceMax(maxTmpVreg, srcVreg0, pregFull);
            StoreUnAlign<ElementInput, PostLiteral::POST_MODE_UPDATE>(newMaxUb, maxTmpVreg, maxUreg, 1);
        }
        vstas(maxUreg, newMaxUb, 0, POST_UPDATE);
    }

    __simd_vf__ inline void UpdateMax(
        __ubuf__ ElementInput *nowMaxUb, __ubuf__ float *lastMaxUb, uint16_t mLoops, uint32_t tailM)
    {
        using namespace AscendC::MicroAPI;
        constexpr static CastTrait castTraitZero = {
            RegLayout::ZERO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        constexpr static CastTrait castTraitOne = {
            RegLayout::ONE,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        RegTensor<ElementInput> nowMaxVreg;
        RegTensor<float> lastMaxFloatVreg0;
        RegTensor<float> lastMaxFloatVreg1;
        RegTensor<ElementInput> maxVreg;
        RegTensor<ElementInput> lastMaxVreg;
        RegTensor<ElementInput> lastMaxVreg0;
        RegTensor<ElementInput> lastMaxVreg1;

        MaskReg pregFull = CreateMask<ElementInput, MaskPattern::ALL>();
        MaskReg pregFloatFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailM = UpdateMask<ElementInput>(tailM);
        for (uint16_t i = 0; i < mLoops; ++i) {
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
                    lastMaxFloatVreg0, lastMaxFloatVreg1, lastMaxUb + i * HALF_REP_SIZE);
            Cast<ElementInput, float, castTraitZero>(lastMaxVreg0, lastMaxFloatVreg0, pregFloatFull);
            Cast<ElementInput, float, castTraitOne>(lastMaxVreg1, lastMaxFloatVreg1, pregFloatFull);
            Or((RegTensor<uint16_t>&)lastMaxVreg,
                (RegTensor<uint16_t>&)lastMaxVreg0, (RegTensor<uint16_t>&)lastMaxVreg1,
                pregFull);
            LoadAlign(nowMaxVreg, nowMaxUb + i * HALF_REP_SIZE);
            Max(maxVreg, nowMaxVreg, lastMaxVreg, pregFull);
            StoreAlign<ElementInput, StoreDist::DIST_NORM_B16>(nowMaxUb + i * HALF_REP_SIZE, maxVreg, pregFull);
        }
        if (tailM > 64) {
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
                    lastMaxFloatVreg0, lastMaxFloatVreg1, lastMaxUb + mLoops * HALF_REP_SIZE);
            Cast<ElementInput, float, castTraitZero>(lastMaxVreg0, lastMaxFloatVreg0, pregFloatFull);
            Cast<ElementInput, float, castTraitOne>(lastMaxVreg1, lastMaxFloatVreg1, pregFloatFull);
            Or((RegTensor<uint16_t>&)lastMaxVreg,
               (RegTensor<uint16_t>&)lastMaxVreg0, (RegTensor<uint16_t>&)lastMaxVreg1,
               pregFull);
            LoadAlign(nowMaxVreg, nowMaxUb + mLoops * HALF_REP_SIZE);
            Max(maxVreg, nowMaxVreg, lastMaxVreg, pregTailM);
            StoreAlign<ElementInput, StoreDist::DIST_NORM_B16>(nowMaxUb + mLoops * HALF_REP_SIZE, maxVreg, pregTailM);
        } else {
            LoadAlign(lastMaxFloatVreg0, lastMaxUb + mLoops * HALF_REP_SIZE);
            Cast<ElementInput, float, castTraitZero>(lastMaxVreg, lastMaxFloatVreg0, pregFloatFull);
            DeInterleave(lastMaxVreg0, lastMaxVreg1, lastMaxVreg, lastMaxVreg);
            LoadAlign(nowMaxVreg, nowMaxUb + mLoops * HALF_REP_SIZE);
            Max(maxVreg, nowMaxVreg, lastMaxVreg0, pregTailM);
            StoreAlign<ElementInput, StoreDist::DIST_NORM_B16>(nowMaxUb + mLoops * HALF_REP_SIZE, maxVreg, pregTailM);
        }
    }

    template <KvBaseTileRegSplitStages kvBaseTileRegSplitStages>
    __simd_vf__ inline void ComputeExpSubSum16(__ubuf__ ElementOutput *expUb, __ubuf__ ElementInput *srcUb,
                                               __ubuf__ ElementInput *nowMaxUb, __ubuf__ ElementInput *expSumUb,
                                               uint16_t m, uint32_t tailN, uint32_t blockStride, uint16_t S2BaseSize)
    {
        static_assert(kvBaseTileRegSplitStages == KvBaseTileRegSplitStages::ONE ||
                      kvBaseTileRegSplitStages == KvBaseTileRegSplitStages::TWO,
                      "ComputeExpSubSum16 only supports ONE Or TWO stages, please use the specialized versions.");
    }

    template <>
    __simd_vf__ inline void ComputeExpSubSum16<KvBaseTileRegSplitStages::ONE>(
        __ubuf__ ElementOutput *expUb, __ubuf__ ElementInput *srcUb,
        __ubuf__ ElementInput *nowMaxUb, __ubuf__ ElementInput *expSumUb,
        uint16_t m, uint32_t tailN, uint32_t blockStride, uint16_t S2BaseSize)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<ElementInput> expVreg;
        RegTensor<ElementInput> expSumVreg;
        RegTensor<ElementInput> maxVreg;

        RegTensor<ElementOutput> expDstVreg;

        UnalignReg expSumUreg;

        MaskReg pregTailN = UpdateMask<ElementInput>(tailN);
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<ElementInput, LoadDist::DIST_BRC_B16>(maxVreg, nowMaxUb + i);
            LoadAlign(expVreg, srcUb + i * S2BaseSize);
            Sub(expDstVreg, expVreg, maxVreg, pregTailN);
            Exp(expDstVreg, expDstVreg, pregTailN);
            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY>(
                expUb + i * ELE_NUM_PER_C0,
                expDstVreg, blockStride, pregTailN);
            ReduceSum(expSumVreg, expDstVreg, pregTailN);
            StoreUnAlign<ElementInput, PostLiteral::POST_MODE_UPDATE>(expSumUb, expSumVreg, expSumUreg, 1);
        }
        vstas(expSumUreg, expSumUb, 0, POST_UPDATE);
    }

    template <>
    __simd_vf__ inline void ComputeExpSubSum16<KvBaseTileRegSplitStages::TWO>(
        __ubuf__ ElementOutput *expUb, __ubuf__ ElementInput *srcUb,
        __ubuf__ ElementInput *nowMaxUb, __ubuf__ ElementInput *expSumUb,
        uint16_t m, uint32_t tailN, uint32_t blockStride, uint16_t S2BaseSize)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<ElementInput> expVreg0;
        RegTensor<ElementInput> expVreg1;
        RegTensor<ElementInput> expSumVreg;
        RegTensor<ElementInput> maxVreg;

        RegTensor<ElementOutput> expDstVreg0;
        RegTensor<ElementOutput> expDstVreg1;

        UnalignReg expSumUreg;

        MaskReg pregFull = CreateMask<ElementInput, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<ElementInput>(tailN);
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<ElementInput, LoadDist::DIST_BRC_B16>(maxVreg, nowMaxUb + i);
            // Duplicate(expSumVreg, 0);
            LoadAlign(expVreg0, srcUb + i * S2BaseSize);
            LoadAlign(expVreg1, srcUb + i * S2BaseSize + HALF_REP_SIZE);
            Sub(expDstVreg0, expVreg0, maxVreg, pregFull);
            Sub(expDstVreg1, expVreg1, maxVreg, pregTailN);
            Exp(expDstVreg0, expDstVreg0, pregFull);
            Exp(expDstVreg1, expDstVreg1, pregTailN);
            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY>(
                expUb + i * ELE_NUM_PER_C0,
                expDstVreg0, blockStride, pregFull);
            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY>(
                expUb + i * ELE_NUM_PER_C0 + blockStride * ELE_NUM_PER_C0 * BLOCK_REP_SIZE,
                expDstVreg1, blockStride, pregTailN);
            Add<ElementOutput, MaskMergeMode::MERGING>(expDstVreg0, expDstVreg1, expDstVreg0, pregTailN);

            ReduceSum(expSumVreg, expDstVreg0, pregFull);
            StoreUnAlign<ElementInput, PostLiteral::POST_MODE_UPDATE>(expSumUb, expSumVreg, expSumUreg, 1);
        }
        vstas(expSumUreg, expSumUb, 0, POST_UPDATE);
    }

    template <KvBaseTileRegSplitStages kvBaseTileRegSplitStages>
    __simd_vf__ inline void ComputeExpSubSum16FP8(__ubuf__ ElementOutput *expUb, __ubuf__ ElementInput *srcUb,
                                                  __ubuf__ ElementInput *nowMaxUb, __ubuf__ ElementInput *expSumUb,
                                                  uint16_t m, uint32_t tailN, uint32_t blockStride, uint16_t S2BaseSize)
    {
        static_assert(kvBaseTileRegSplitStages == KvBaseTileRegSplitStages::ONE ||
                      kvBaseTileRegSplitStages == KvBaseTileRegSplitStages::TWO,
                      "ComputeExpSubSum16FP8 only supports ONE Or TWO stages, please use the specialized versions.");
    }

    template <>
    __simd_vf__ inline void ComputeExpSubSum16FP8<KvBaseTileRegSplitStages::ONE>(
        __ubuf__ ElementOutput *expUb, __ubuf__ ElementInput *srcUb, __ubuf__ ElementInput *nowMaxUb,
        __ubuf__ ElementInput *expSumUb, uint16_t m, uint32_t tailN, uint32_t blockStride, uint16_t S2BaseSize)
    {
        using namespace AscendC::MicroAPI;

        RegTensor<ElementInput> expVreg;
        RegTensor<ElementInput> expSumVreg;
        RegTensor<ElementInput> maxVreg;

        RegTensor<ElementInput> expDstVreg;
        RegTensor<ElementInput> expDstVregResult;

        UnalignReg expSumUreg;

        constexpr static CastTrait castTraitZeroUNKNOWN = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        constexpr static CastTrait castTraitOneUNKNOWN = {
            RegLayout::ONE,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        constexpr static CastTrait castTraitZeroRINT = {
            RegLayout::ZERO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT,
        };

        constexpr static CastTrait castTraitOneRINT = {
            RegLayout::ONE,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT,
        };

        constexpr static CastTrait castTraitTwoRINT = {
            RegLayout::TWO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT,
        };

        constexpr static CastTrait castTraitThreeRINT = {
            RegLayout::THREE,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT,
        };

        RegTensor<float> floatExpVreg0;
        RegTensor<float> floatExpVreg1;
        RegTensor<float> deInterleaveVreg0;
        RegTensor<float> deInterleaveVreg1;
        RegTensor<float> deInterleaveVreg2;
        RegTensor<float> deInterleaveVreg3;
        RegTensor<ElementOutput> pVreg0;
        RegTensor<ElementOutput> pVreg1;
        RegTensor<ElementOutput> pVreg2;
        RegTensor<ElementOutput> pVreg3;

        MaskReg pRegUint8All = CreateMask<uint8_t, MaskPattern::ALL>();
        MaskReg pRegUint8VL128 = CreateMask<uint8_t, MaskPattern::VL128>();
        MaskReg pRegFp16All = CreateMask<ElementInput, MaskPattern::ALL>();
        MaskReg pregFp16TailN = UpdateMask<ElementInput>(tailN);
        MaskReg pRegFp32All = CreateMask<float, MaskPattern::ALL>();

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<ElementInput, LoadDist::DIST_BRC_B16>(maxVreg, nowMaxUb + i);
            LoadAlign(expVreg, srcUb + i * S2BaseSize);
            Sub(expDstVreg, expVreg, maxVreg, pregFp16TailN);
            Exp(expDstVregResult, expDstVreg, pregFp16TailN);
            Exp(expDstVreg, expDstVreg, pregFp16TailN);

            constexpr half maxValueFP8 = static_cast<half>(448.0f);
            Muls(expDstVreg, expDstVreg, maxValueFP8, pregFp16TailN);

            Cast<float, ElementInput, castTraitZeroUNKNOWN>(floatExpVreg0, expDstVreg, pRegFp16All);
            Cast<float, ElementInput, castTraitOneUNKNOWN>(floatExpVreg1, expDstVreg, pRegFp16All);

            DeInterleave(deInterleaveVreg0, deInterleaveVreg1, floatExpVreg0, floatExpVreg0);
            DeInterleave(deInterleaveVreg2, deInterleaveVreg3, floatExpVreg1, floatExpVreg1);

            Cast<ElementOutput, float, castTraitZeroRINT>(pVreg0, deInterleaveVreg0, pRegFp32All);
            Cast<ElementOutput, float, castTraitOneRINT>(pVreg1, deInterleaveVreg2, pRegFp32All);
            Cast<ElementOutput, float, castTraitTwoRINT>(pVreg2, deInterleaveVreg1, pRegFp32All);
            Cast<ElementOutput, float, castTraitThreeRINT>(pVreg3, deInterleaveVreg3, pRegFp32All);

            Or((RegTensor<uint8_t> &)pVreg0, (RegTensor<uint8_t> &)pVreg0, (RegTensor<uint8_t> &)pVreg1, pRegUint8All);
            Or((RegTensor<uint8_t> &)pVreg0, (RegTensor<uint8_t> &)pVreg0, (RegTensor<uint8_t> &)pVreg2, pRegUint8All);
            Or((RegTensor<uint8_t> &)pVreg0, (RegTensor<uint8_t> &)pVreg0, (RegTensor<uint8_t> &)pVreg3, pRegUint8All);

            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY>(expUb + i * ELE_NUM_PER_C0_FP8, pVreg0,
                                                                     blockStride, pRegUint8VL128);
            ReduceSum(expSumVreg, expDstVregResult, pregFp16TailN);
            StoreUnAlign<ElementInput, PostLiteral::POST_MODE_UPDATE>(expSumUb, expSumVreg, expSumUreg, 1);
        }
        vstas(expSumUreg, expSumUb, 0, POST_UPDATE);
    }

    template <>
    __simd_vf__ inline void ComputeExpSubSum16FP8<KvBaseTileRegSplitStages::TWO>(
        __ubuf__ ElementOutput *expUb, __ubuf__ ElementInput *srcUb, __ubuf__ ElementInput *nowMaxUb,
        __ubuf__ ElementInput *expSumUb, uint16_t m, uint32_t tailN, uint32_t blockStride, uint16_t S2BaseSize)
    {
        using namespace AscendC::MicroAPI;

        RegTensor<ElementInput> expVreg0;
        RegTensor<ElementInput> expVreg1;
        RegTensor<ElementInput> expSumVreg;
        RegTensor<ElementInput> maxVreg;

        RegTensor<ElementInput> expDstVreg0;
        RegTensor<ElementInput> expDstVreg1;
        RegTensor<ElementInput> expDstVregResult;

        UnalignReg expSumUreg;

        constexpr static CastTrait castTraitZeroUNKNOWN = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        constexpr static CastTrait castTraitOneUNKNOWN = {
            RegLayout::ONE,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        constexpr static CastTrait castTraitZeroRINT = {
            RegLayout::ZERO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT,
        };

        constexpr static CastTrait castTraitOneRINT = {
            RegLayout::ONE,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT,
        };

        constexpr static CastTrait castTraitTwoRINT = {
            RegLayout::TWO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT,
        };

        constexpr static CastTrait castTraitThreeRINT = {
            RegLayout::THREE,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT,
        };

        RegTensor<ElementInput> expDstVreg0Fp8;
        RegTensor<ElementInput> expDstVreg1Fp8;
        RegTensor<float> floatVreg0;
        RegTensor<float> floatVreg1;
        RegTensor<float> floatVreg2;
        RegTensor<float> floatVreg3;
        RegTensor<float> deInterleaveVreg0;
        RegTensor<float> deInterleaveVreg1;
        RegTensor<float> deInterleaveVreg2;
        RegTensor<float> deInterleaveVreg3;
        RegTensor<float> deInterleaveVreg4;
        RegTensor<float> deInterleaveVreg5;
        RegTensor<float> deInterleaveVreg6;
        RegTensor<float> deInterleaveVreg7;
        RegTensor<ElementOutput> pVreg0;
        RegTensor<ElementOutput> pVreg1;
        RegTensor<ElementOutput> pVreg2;
        RegTensor<ElementOutput> pVreg3;
        RegTensor<ElementOutput> pVreg4;
        RegTensor<ElementOutput> pVreg5;
        RegTensor<ElementOutput> pVreg6;
        RegTensor<ElementOutput> pVreg7;

        MaskReg pRegUint8All = CreateMask<uint8_t, MaskPattern::ALL>();
        MaskReg pRegUint8VL128 = CreateMask<uint8_t, MaskPattern::VL128>();
        MaskReg pRegFp16All = CreateMask<ElementInput, MaskPattern::ALL>();
        MaskReg pregFp16TailN = UpdateMask<ElementInput>(tailN);
        MaskReg pRegFp32All = CreateMask<float, MaskPattern::ALL>();

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<ElementInput, LoadDist::DIST_BRC_B16>(maxVreg, nowMaxUb + i);
            LoadAlign(expVreg0, srcUb + i * S2BaseSize);
            LoadAlign(expVreg1, srcUb + i * S2BaseSize + HALF_REP_SIZE);
            Sub(expDstVreg0, expVreg0, maxVreg, pRegFp16All);
            Sub(expDstVreg1, expVreg1, maxVreg, pregFp16TailN);
            Exp(expDstVreg0Fp8, expDstVreg0, pRegFp16All);
            Exp(expDstVreg1Fp8, expDstVreg1, pregFp16TailN);
            Exp(expDstVreg0, expDstVreg0, pRegFp16All);
            Exp(expDstVreg1, expDstVreg1, pregFp16TailN);

            constexpr half maxValueFP8 = static_cast<half>(448.0f);
            Muls(expDstVreg0Fp8, expDstVreg0Fp8, maxValueFP8, pRegFp16All);
            Muls(expDstVreg1Fp8, expDstVreg1Fp8, maxValueFP8, pregFp16TailN);

            Cast<float, ElementInput, castTraitZeroUNKNOWN>(floatVreg0, expDstVreg0Fp8, pRegFp16All);
            Cast<float, ElementInput, castTraitOneUNKNOWN>(floatVreg1, expDstVreg0Fp8, pRegFp16All);
            Cast<float, ElementInput, castTraitZeroUNKNOWN>(floatVreg2, expDstVreg1Fp8, pRegFp16All);
            Cast<float, ElementInput, castTraitOneUNKNOWN>(floatVreg3, expDstVreg1Fp8, pRegFp16All);

            DeInterleave(deInterleaveVreg0, deInterleaveVreg1, floatVreg0, floatVreg0);
            DeInterleave(deInterleaveVreg2, deInterleaveVreg3, floatVreg1, floatVreg1);
            DeInterleave(deInterleaveVreg4, deInterleaveVreg5, floatVreg2, floatVreg2);
            DeInterleave(deInterleaveVreg6, deInterleaveVreg7, floatVreg3, floatVreg3);

            Cast<ElementOutput, float, castTraitZeroRINT>(pVreg0, deInterleaveVreg0, pRegFp32All);
            Cast<ElementOutput, float, castTraitOneRINT>(pVreg1, deInterleaveVreg2, pRegFp32All);
            Cast<ElementOutput, float, castTraitTwoRINT>(pVreg2, deInterleaveVreg1, pRegFp32All);
            Cast<ElementOutput, float, castTraitThreeRINT>(pVreg3, deInterleaveVreg3, pRegFp32All);
            Cast<ElementOutput, float, castTraitZeroRINT>(pVreg4, deInterleaveVreg4, pRegFp32All);
            Cast<ElementOutput, float, castTraitOneRINT>(pVreg5, deInterleaveVreg6, pRegFp32All);
            Cast<ElementOutput, float, castTraitTwoRINT>(pVreg6, deInterleaveVreg5, pRegFp32All);
            Cast<ElementOutput, float, castTraitThreeRINT>(pVreg7, deInterleaveVreg7, pRegFp32All);

            Or((RegTensor<uint8_t> &)pVreg0, (RegTensor<uint8_t> &)pVreg0, (RegTensor<uint8_t> &)pVreg1, pRegUint8All);
            Or((RegTensor<uint8_t> &)pVreg0, (RegTensor<uint8_t> &)pVreg0, (RegTensor<uint8_t> &)pVreg2, pRegUint8All);
            Or((RegTensor<uint8_t> &)pVreg0, (RegTensor<uint8_t> &)pVreg0, (RegTensor<uint8_t> &)pVreg3, pRegUint8All);
            Or((RegTensor<uint8_t> &)pVreg4, (RegTensor<uint8_t> &)pVreg4, (RegTensor<uint8_t> &)pVreg5, pRegUint8All);
            Or((RegTensor<uint8_t> &)pVreg4, (RegTensor<uint8_t> &)pVreg4, (RegTensor<uint8_t> &)pVreg6, pRegUint8All);
            Or((RegTensor<uint8_t> &)pVreg4, (RegTensor<uint8_t> &)pVreg4, (RegTensor<uint8_t> &)pVreg7, pRegUint8All);

            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY>(expUb + i * ELE_NUM_PER_C0_FP8, pVreg0,
                                                                     blockStride, pRegUint8VL128);
            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY>(
                expUb + i * ELE_NUM_PER_C0_FP8 + blockStride * HALF_VECTOR_SIZE, pVreg4, blockStride, pRegUint8VL128);
            Add<ElementInput, MaskMergeMode::MERGING>(expDstVreg0, expDstVreg1, expDstVreg0, pregFp16TailN);
            ReduceSum(expSumVreg, expDstVreg0, pRegFp16All);
            StoreUnAlign<ElementInput, PostLiteral::POST_MODE_UPDATE>(expSumUb, expSumVreg, expSumUreg, 1);
        }
        vstas(expSumUreg, expSumUb, 0, POST_UPDATE);
    }

    template <typename ElementInput>
    __simd_vf__ inline void CastExpSumAndExpMax(__ubuf__ float *sumUb, __ubuf__ float *maxUb,
        __ubuf__ ElementInput *expSumUb, __ubuf__ ElementInput *nowMaxUb,
        uint16_t mLoops, uint32_t tailM)
    {
        using namespace AscendC::MicroAPI;

        constexpr static CastTrait castTrait = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        RegTensor<ElementInput> nowMaxVreg0;
        RegTensor<ElementInput> nowMaxVreg1;
        RegTensor<ElementInput> nowMaxTmpVreg;
        RegTensor<float> nowMaxFloatVreg;
        RegTensor<ElementInput> brcExpSumVreg0;
        RegTensor<ElementInput> brcExpSumVreg1;
        RegTensor<ElementInput> brcExpSumTmpVreg;
        RegTensor<float> brcExpSumFloatVreg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailM = UpdateMask<float>(tailM);
        MaskReg pregNowMaxFull = CreateMask<ElementInput, MaskPattern::ALL>();
        for (int16_t i = 0; i < mLoops; ++i) {
            LoadAlign(nowMaxTmpVreg, nowMaxUb + i * FLOAT_REP_SIZE);
            Interleave(nowMaxVreg0, nowMaxVreg1, nowMaxTmpVreg, nowMaxTmpVreg);
            Cast<float, ElementInput, castTrait>(nowMaxFloatVreg, nowMaxVreg0, pregNowMaxFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(maxUb + i * FLOAT_REP_SIZE, nowMaxFloatVreg, pregFull);

            LoadAlign(brcExpSumTmpVreg, expSumUb + i * FLOAT_REP_SIZE);
            Interleave(brcExpSumVreg0, brcExpSumVreg1, brcExpSumTmpVreg, brcExpSumTmpVreg);
            Cast<float, ElementInput, castTrait>(brcExpSumFloatVreg, brcExpSumVreg0, pregNowMaxFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(sumUb + i * FLOAT_REP_SIZE, brcExpSumFloatVreg, pregFull);
        }
        LoadAlign(nowMaxTmpVreg, nowMaxUb + mLoops * FLOAT_REP_SIZE);
        Interleave(nowMaxVreg0, nowMaxVreg1, nowMaxTmpVreg, nowMaxTmpVreg);
        Cast<float, ElementInput, castTrait>(nowMaxFloatVreg, nowMaxVreg0, pregNowMaxFull);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(maxUb + mLoops * FLOAT_REP_SIZE, nowMaxFloatVreg, pregTailM);

        LoadAlign(brcExpSumTmpVreg, expSumUb + mLoops * FLOAT_REP_SIZE);
        Interleave(brcExpSumVreg0, brcExpSumVreg1, brcExpSumTmpVreg, brcExpSumTmpVreg);
        Cast<float, ElementInput, castTrait>(brcExpSumFloatVreg, brcExpSumVreg0, pregNowMaxFull);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(sumUb + mLoops * FLOAT_REP_SIZE, brcExpSumFloatVreg, pregTailM);
    }

    template <typename ElementInput>
    __simd_vf__ inline void UpdateExpSumAndExpMax(__ubuf__ float *sumUb, __ubuf__ float *expMaxUb,
        __ubuf__ float *maxUb, __ubuf__ ElementInput *expSumUb, __ubuf__ ElementInput *nowMaxUb,
        uint16_t mLoops, uint32_t tailM)
    {
        using namespace AscendC::MicroAPI;

        constexpr static CastTrait castTrait = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        RegTensor<ElementInput> nowMaxVreg0;
        RegTensor<ElementInput> nowMaxVreg1;
        RegTensor<ElementInput> nowMaxTmpVreg;
        RegTensor<float> nowMaxFloatVreg;
        RegTensor<float> lastMaxVreg;
        RegTensor<float> expMaxVreg;
        RegTensor<float> lastExpSumVreg;
        RegTensor<ElementInput> brcExpSumVreg0;
        RegTensor<ElementInput> brcExpSumVreg1;
        RegTensor<ElementInput> brcExpSumTmpVreg;
        RegTensor<float> brcExpSumFloatVreg;
        RegTensor<float> updateExpSumVreg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailM = UpdateMask<float>(tailM);
        MaskReg pregNowMaxFull = CreateMask<ElementInput, MaskPattern::ALL>();
        for (int16_t i = 0; i < mLoops; ++i) {
            LoadAlign(lastMaxVreg, maxUb + i * FLOAT_REP_SIZE);
            LoadAlign(nowMaxTmpVreg, nowMaxUb + i * FLOAT_REP_SIZE);
            Interleave(nowMaxVreg0, nowMaxVreg1, nowMaxTmpVreg, nowMaxTmpVreg);
            Cast<float, ElementInput, castTrait>(nowMaxFloatVreg, nowMaxVreg0, pregNowMaxFull);
            FusedExpSub(expMaxVreg, lastMaxVreg, nowMaxFloatVreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(expMaxUb + i * FLOAT_REP_SIZE, expMaxVreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(maxUb + i * FLOAT_REP_SIZE, nowMaxFloatVreg, pregFull);

            LoadAlign(lastExpSumVreg, sumUb + i * FLOAT_REP_SIZE);
            LoadAlign(brcExpSumTmpVreg, expSumUb + i * FLOAT_REP_SIZE);
            Interleave(brcExpSumVreg0, brcExpSumVreg1, brcExpSumTmpVreg, brcExpSumTmpVreg);
            Cast<float, ElementInput, castTrait>(brcExpSumFloatVreg, brcExpSumVreg0, pregNowMaxFull);
            Mul(updateExpSumVreg, expMaxVreg, lastExpSumVreg, pregFull);
            Add(updateExpSumVreg, updateExpSumVreg, brcExpSumFloatVreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(sumUb + i * FLOAT_REP_SIZE, updateExpSumVreg, pregFull);
        }
        LoadAlign(lastMaxVreg, maxUb + mLoops * FLOAT_REP_SIZE);
        LoadAlign(nowMaxTmpVreg, nowMaxUb + mLoops * FLOAT_REP_SIZE);
        Interleave(nowMaxVreg0, nowMaxVreg1, nowMaxTmpVreg, nowMaxTmpVreg);
        Cast<float, ElementInput, castTrait>(nowMaxFloatVreg, nowMaxVreg0, pregNowMaxFull);
        FusedExpSub(expMaxVreg, lastMaxVreg, nowMaxFloatVreg, pregTailM);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(expMaxUb + mLoops * FLOAT_REP_SIZE, expMaxVreg, pregTailM);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(maxUb + mLoops * FLOAT_REP_SIZE, nowMaxFloatVreg, pregTailM);

        LoadAlign(lastExpSumVreg, sumUb + mLoops * FLOAT_REP_SIZE);
        LoadAlign(brcExpSumTmpVreg, expSumUb + mLoops * FLOAT_REP_SIZE);
        Interleave(brcExpSumVreg0, brcExpSumVreg1, brcExpSumTmpVreg, brcExpSumTmpVreg);
        Cast<float, ElementInput, castTrait>(brcExpSumFloatVreg, brcExpSumVreg0, pregNowMaxFull);
        Mul(updateExpSumVreg, expMaxVreg, lastExpSumVreg, pregTailM);
        Add(updateExpSumVreg, updateExpSumVreg, brcExpSumFloatVreg, pregTailM);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(sumUb + mLoops * FLOAT_REP_SIZE, updateExpSumVreg, pregTailM);
    }
};
}

#endif  // EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_LOW_PREC_HPP