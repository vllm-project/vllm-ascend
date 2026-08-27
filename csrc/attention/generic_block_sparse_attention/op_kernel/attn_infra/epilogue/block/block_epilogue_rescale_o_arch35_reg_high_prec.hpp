/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_ARCH35_REG_HIGH_PREC
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_ARCH35_REG_HIGH_PREC

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/resource.hpp"
#include "../../../attn_infra/epilogue/dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/tile_common/tile_copy.hpp"
#include "../../../attn_infra/gemm_coord.hpp"
#include "../../../attn_infra/matrix_coord.hpp"
#include "../../../tla/tensor.hpp"
#include "../../../tla/layout.hpp"

namespace NpuArch::Epilogue::Block {

enum class DRegSplitStages {
    ONE,
    TWO
};

template <
    class ElementO_,
    class ElementOTmp_,
    class ElementS_,
    class ElementKV_,
    class TileCopy_,
    class OTmpSrcPos_, // the src TPosition of pv res, viable configurations: GM/L0C
    LseMode LSE_MODE_,
    LseFormat LSE_FORMAT_>
class BlockEpilogue<
    EpilogueAtlasA5BsaRescaleO<LSE_MODE_, LSE_FORMAT_>,
    ElementO_,
    ElementOTmp_,
    ElementS_,
    ElementKV_,
    TileCopy_,
    OTmpSrcPos_>
{
public:
    using DispatchPolicy = EpilogueAtlasA5BsaRescaleO<LSE_MODE_, LSE_FORMAT_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementO = ElementO_;
    using ElementOTmp = ElementOTmp_;
    using ElementLse = float;
    using SMDtype = ElementS_;
    using ElementKV = ElementKV_;
    using TileCopy = TileCopy_;
    using OTmpSrcPos = OTmpSrcPos_;

    using CopyUbToGmO = typename TileCopy::CopyUbToGmO;

    static constexpr uint32_t UB_OTMP_BUF_STAGES = 2;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 32768;
    static constexpr uint32_t DM_UB_GLOBAL_ELEM_NUM = 64;
    static constexpr uint32_t RESCALE_ROW_MAX_ELEM_NUM = 64;
    static constexpr uint32_t RESCALE_COL_MAX_ELEM_NUM = 128;
    static constexpr uint32_t RESCALE_VREG_SIZE = 256 / sizeof(ElementOTmp);
    static constexpr float MAX_VALUE_RECIPROCAL_FP8 = 1.0f / 448.0f;
    static constexpr bool FULL_QUANT_FP8 = AscendC::IsSameType<ElementKV, fp8_e4m3fn_t>::value;
    __aicore__ inline
    BlockEpilogue(Arch::Resource<ArchTag> &resource)
    {
        constexpr uint32_t LO_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t GO_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t LM_UB_TENSOR_OFFSET = 7 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t GM_UB_TENSOR_OFFSET = LM_UB_TENSOR_OFFSET + 64 * sizeof(float);
        constexpr uint32_t DM_UB_TENSOR_OFFSET = GM_UB_TENSOR_OFFSET + 64 * sizeof(float);
        constexpr uint32_t LL_UB_TENSOR_OFFSET = DM_UB_TENSOR_OFFSET + 3 * 64 * sizeof(float);
        constexpr uint32_t GL_UB_TENSOR_OFFSET = LL_UB_TENSOR_OFFSET +  64 * sizeof(float);
        constexpr uint32_t LSE_UB_TENSOR_OFFSET = GL_UB_TENSOR_OFFSET +  64 * sizeof(float);

        for (uint32_t i = 0; i < UB_OTMP_BUF_STAGES; i++) {
            loUbTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(
                LO_UB_TENSOR_OFFSET + i * UB_UINT8_BLOCK_SIZE);
        }
        goUbTensor32 = resource.ubBuf.template GetBufferByByte<ElementOTmp>(GO_UB_TENSOR_OFFSET);
        goUbTensor16 = resource.ubBuf.template GetBufferByByte<ElementO>(GO_UB_TENSOR_OFFSET);
        glUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
        dmUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
        gmUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET);
        lseUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(LSE_UB_TENSOR_OFFSET);
    }

    __aicore__ inline
    ~BlockEpilogue()
    {
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void SetCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        // in mode 4, AIC set for 2 AIVs separately
        if constexpr (MODE == 4U) {
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void WaitCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        // in mode 4, AIC wait for 2 AIVs separately
        if constexpr (MODE == 4U) {
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    template <class TensorLseGm, class TensorLseUb>
    __aicore__ inline
    void CopyUbToGmLse(TensorLseGm const &gLseTensorTlaTile, TensorLseUb const &ubLseTensorTla)
    {
        AscendC::DataCopyExtParams repeatParams;
        if constexpr ((DispatchPolicy::LSE_FORMAT == LseFormat::TN1) ||
            (DispatchPolicy::LSE_FORMAT == LseFormat::BSN1)) {
            repeatParams.blockCount = tla::get<0>(ubLseTensorTla.shape());
            repeatParams.blockLen = sizeof(float);
            repeatParams.srcStride = 0;
            repeatParams.dstStride = (tla::get<0>(gLseTensorTlaTile.stride()) - 1) * sizeof(float);
        } else if constexpr (DispatchPolicy::LSE_FORMAT == LseFormat::BNS1) {
            repeatParams.blockCount = 1;
            repeatParams.blockLen = tla::get<0>(ubLseTensorTla.shape()) * sizeof(float);
            repeatParams.srcStride = 0;
            repeatParams.dstStride = 0;
        }
        auto dstOffset = gLseTensorTlaTile.layout()(gLseTensorTlaTile.coord());
        auto srcOffset = ubLseTensorTla.layout()(ubLseTensorTla.coord());
        AscendC::DataCopyPad(gLseTensorTlaTile.data()[dstOffset], ubLseTensorTla.data()[srcOffset], repeatParams);
    }

    template <bool IS_FD, class TensorO, class TensorLse>
    __aicore__ inline
    void SubCoreCompute(TensorO &gOTensorTlaTile,
                        TensorLse &gLseTensorTlaTile,
                        uint32_t curTileMod,
                        uint32_t ubOTmpBufId,
                        bool isFirstKvSTile,
                        bool isLastKvSTile,
                        uint32_t colStrideCurSubCore,
                        Arch::CrossCoreFlag mm2ToReFlag,
                        AscendC::GlobalTensor<float> &gPartialO,
                        AscendC::GlobalTensor<float> &gPartialLse,
                        uint32_t fdTaskId,
                        uint32_t groupRowOffset,
                        uint32_t groupSize,
                        uint32_t fdLseSubStride)
    {
        uint32_t rowNumCurSubCore = tla::get<0>(gOTensorTlaTile.shape());
        uint32_t colNumCurSubCore = tla::get<1>(gOTensorTlaTile.shape());
        uint32_t vlElemNum = AscendC::GetVecLen() / sizeof(ElementOTmp);
        uint32_t colFullLoop = CeilDiv(colNumCurSubCore, vlElemNum) - 1;
        uint32_t colTail = (colNumCurSubCore - 1) % vlElemNum + 1;
        uint32_t dStages = CeilDiv(colNumCurSubCore, RESCALE_VREG_SIZE);

        __ubuf__ ElementOTmp *goUb = (__ubuf__ ElementOTmp *) goUbTensor32.GetPhyAddr();
        __ubuf__ ElementOTmp *loUb = (__ubuf__ ElementOTmp *) loUbTensor[ubOTmpBufId].GetPhyAddr();
        __ubuf__ ElementOTmp *glUb = (__ubuf__ ElementOTmp *) glUbTensor32.GetPhyAddr();
        __ubuf__ ElementOTmp *dmUb =
            (__ubuf__ ElementOTmp *) dmUbTensor32[curTileMod * DM_UB_GLOBAL_ELEM_NUM].GetPhyAddr();
        __ubuf__ float *gmUb = (__ubuf__ float *) gmUbTensor32.GetPhyAddr();
        __ubuf__ float *lseUb = (__ubuf__ float *) lseUbTensor32.GetPhyAddr();
        
        WaitCrossCoreSync<4, PIPE_V>(mm2ToReFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        if (isFirstKvSTile) {
            if (!isLastKvSTile) {
                uint32_t totalCopyElems = rowNumCurSubCore * colStrideCurSubCore;
                AscendC::DataCopy(goUbTensor32, loUbTensor[ubOTmpBufId], totalCopyElems);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                if (dStages == 1) {
                    DivFuncLastAndFirst<DRegSplitStages::ONE>(
                        goUb, loUb, glUb, rowNumCurSubCore, colStrideCurSubCore, colTail, vlElemNum);
                } else if (dStages == 2) {
                    DivFuncLastAndFirst<DRegSplitStages::TWO>(
                        goUb, loUb, glUb, rowNumCurSubCore, colStrideCurSubCore, colTail, vlElemNum);
                }
            }
        } else if (!isLastKvSTile) {
            if (dStages == 1) {
                RescaleFunc<DRegSplitStages::ONE>(
                    goUb, loUb, dmUb, rowNumCurSubCore, colStrideCurSubCore, colTail, vlElemNum);
            } else if (dStages == 2) {
                RescaleFunc<DRegSplitStages::TWO>(
                    goUb, loUb, dmUb, rowNumCurSubCore, colStrideCurSubCore, colTail, vlElemNum);
            }
        } else {
            if (dStages == 1) {
                RescaleFuncLastNotFirst<DRegSplitStages::ONE>(
                goUb, loUb, dmUb, glUb, rowNumCurSubCore, colStrideCurSubCore, colTail, vlElemNum);
            } else if (dStages == 2) {
                RescaleFuncLastNotFirst<DRegSplitStages::TWO>(
                goUb, loUb, dmUb, glUb, rowNumCurSubCore, colStrideCurSubCore, colTail, vlElemNum);
            }
        }
        // release lo buf
        SetCrossCoreSync<4, PIPE_V>(mm2ToReFlag);
        if (isLastKvSTile) {
            if constexpr (IS_FD) {
                const uint32_t rowCountAlign = RoundUp(rowNumCurSubCore, 8);
                AscendC::Ln(lseUbTensor32, glUbTensor32, rowCountAlign);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(lseUbTensor32, lseUbTensor32, gmUbTensor32, rowCountAlign);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                const uint64_t oOffset = static_cast<uint64_t>(fdTaskId) * groupSize * colStrideCurSubCore +
                    static_cast<uint64_t>(groupRowOffset) * colStrideCurSubCore;
                AscendC::DataCopy(gPartialO[oOffset], goUbTensor32,
                                  rowNumCurSubCore * colStrideCurSubCore);
                const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
                const uint64_t lseOffset = static_cast<uint64_t>(fdTaskId) * 2 * fdLseSubStride +
                    subBlockIdx * fdLseSubStride;
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
                AscendC::DataCopyPad(gPartialLse[lseOffset], lseUbTensor32,
                    AscendC::DataCopyExtParams(1, rowNumCurSubCore * sizeof(float), 0, 0, 0));
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
                return;
            }
            if constexpr (DispatchPolicy::LSE_MODE == LseMode::OUT_ONLY) {
                uint32_t colNumLseUb = 0;
                uint32_t colStrideLseUb = 0;
                if constexpr ((DispatchPolicy::LSE_FORMAT == LseFormat::TN1) ||
                    (DispatchPolicy::LSE_FORMAT == LseFormat::BSN1)) {
                    LogSumExpFuncQSAxisIncontinuous(gmUb, glUb, lseUb, rowNumCurSubCore);
                    colNumLseUb = 8;
                    colStrideLseUb = 8;
                } else if constexpr (DispatchPolicy::LSE_FORMAT == LseFormat::BNS1) {
                    LogSumExpFuncQSAxisContinuous(gmUb, glUb, lseUb, rowNumCurSubCore);
                    colNumLseUb = 1;
                    colStrideLseUb = 1;
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
                auto ubLseLayoutTla = tla::MakeLayout(
                    tla::MakeShape(rowNumCurSubCore, colNumLseUb),
                    tla::MakeStride(colStrideLseUb, tla::Int<1>{}));
                auto ubLseTensorTla = tla::MakeTensor(lseUbTensor32, ubLseLayoutTla, Arch::PositionUB{});
                CopyUbToGmLse(gLseTensorTlaTile, ubLseTensorTla);
            }
            AscendC::PipeBarrier<PIPE_V>();

            if (std::is_same<ElementO, bfloat16_t>::value) {
                AscendC::Cast(
                    goUbTensor16, goUbTensor32,
                    AscendC::RoundMode::CAST_RINT,
                    rowNumCurSubCore * colStrideCurSubCore);
            } else {
                AscendC::Cast(
                    goUbTensor16, goUbTensor32,
                    AscendC::RoundMode::CAST_NONE,
                    rowNumCurSubCore * colStrideCurSubCore);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            auto ubOLayoutTla = tla::MakeLayout(
                tla::MakeShape(rowNumCurSubCore, colNumCurSubCore),
                tla::MakeStride(colStrideCurSubCore, tla::Int<1>{})
            );
            auto ubOTensorTla = tla::MakeTensor(goUbTensor16, ubOLayoutTla, Arch::PositionUB{});
            copyUbToGmO(gOTensorTlaTile, ubOTensorTla);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
    }
    
    template <DRegSplitStages dRegSplitStages>
    __simd_vf__ inline void RescaleFunc(
        __ubuf__ ElementOTmp *goUb, __ubuf__ ElementOTmp *loUb, __ubuf__ ElementOTmp *dmUb,
        uint32_t row, uint32_t colStride,
        uint32_t colTail, uint32_t vlElemNum)
    {
    }

    template <>
    __simd_vf__ inline void RescaleFunc<DRegSplitStages::ONE>(
        __ubuf__ ElementOTmp *goUb, __ubuf__ ElementOTmp *loUb, __ubuf__ ElementOTmp *dmUb,
        uint32_t row, uint32_t colStride,
        uint32_t colTail, uint32_t vlElemNum)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> dmVreg;
        RegTensor<float> goPreVreg;
        RegTensor<float> loVreg;
        RegTensor<float> mulVreg;
        RegTensor<float> goCurVreg;
        MaskReg pregTail = UpdateMask<float>(colTail);
        for (uint16_t i = 0; i < row; i++) {
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(dmVreg, dmUb + i);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(goPreVreg, goUb + i * colStride);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(loVreg, loUb + i * colStride);
            Mul(mulVreg, goPreVreg, dmVreg, pregTail);
            Add(goCurVreg, mulVreg, loVreg, pregTail);
            StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(
                goUb + i * colStride, goCurVreg, pregTail);
        }
    }

    template <>
    __simd_vf__ inline void RescaleFunc<DRegSplitStages::TWO>(
        __ubuf__ ElementOTmp *goUb, __ubuf__ ElementOTmp *loUb, __ubuf__ ElementOTmp *dmUb,
        uint32_t row, uint32_t colStride,
        uint32_t colTail, uint32_t vlElemNum)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> dmVreg;
        RegTensor<float> goPreVreg0;
        RegTensor<float> goPreVreg1;
        RegTensor<float> loVreg0;
        RegTensor<float> loVreg1;
        RegTensor<float> mulVreg0;
        RegTensor<float> mulVreg1;
        RegTensor<float> goCurVreg0;
        RegTensor<float> goCurVreg1;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTail = UpdateMask<float>(colTail);
        for (uint16_t i = 0; i < row; i++) {
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(dmVreg, dmUb + i);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(goPreVreg0, goUb + i * colStride);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(goPreVreg1, goUb + i * colStride + vlElemNum);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(loVreg0, loUb + i * colStride);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(loVreg1, loUb + i * colStride + vlElemNum);
            Mul(mulVreg0, goPreVreg0, dmVreg, pregFull);
            Mul(mulVreg1, goPreVreg1, dmVreg, pregTail);
            Add(goCurVreg0, mulVreg0, loVreg0, pregFull);
            Add(goCurVreg1, mulVreg1, loVreg1, pregTail);
            StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(
                goUb + i * colStride, goCurVreg0, pregFull);
            StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(
                goUb + i * colStride + vlElemNum, goCurVreg1, pregTail);
        }
    }

    template <DRegSplitStages dRegSplitStages>
    __simd_vf__ inline void RescaleFuncLastNotFirst(
        __ubuf__ ElementOTmp *goUb, __ubuf__ ElementOTmp *loUb,
        __ubuf__ ElementOTmp *dmUb, __ubuf__ ElementOTmp *glUb,
        uint32_t row, uint32_t colStride,
        uint32_t colTail, uint32_t vlElemNum)
    {
    }

    template <>
    __simd_vf__ inline void RescaleFuncLastNotFirst<DRegSplitStages::ONE>(
        __ubuf__ ElementOTmp *goUb, __ubuf__ ElementOTmp *loUb,
        __ubuf__ ElementOTmp *dmUb, __ubuf__ ElementOTmp *glUb,
        uint32_t row, uint32_t colStride,
        uint32_t colTail, uint32_t vlElemNum)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> dmVreg;
        RegTensor<float> goPreVreg;
        RegTensor<float> loVreg;
        RegTensor<float> mulVreg;
        RegTensor<float> goCurVreg;
        RegTensor<float> glVreg;
        RegTensor<float> divVreg;
        MaskReg pregTail = UpdateMask<float>(colTail);
        for (uint16_t i = 0; i < row; i++) {
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(dmVreg, dmUb + i);
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(glVreg, glUb + i);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(goPreVreg, goUb + i * colStride);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(loVreg, loUb + i * colStride);
            Mul(mulVreg, goPreVreg, dmVreg, pregTail);
            Add(goCurVreg, mulVreg, loVreg, pregTail);
            Div(divVreg, goCurVreg, glVreg, pregTail);
            if constexpr (FULL_QUANT_FP8) {
                Muls(divVreg, divVreg, MAX_VALUE_RECIPROCAL_FP8, pregTail);
            }
            StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(goUb + i * colStride, divVreg, pregTail);
        }
    }

    template <>
    __simd_vf__ inline void RescaleFuncLastNotFirst<DRegSplitStages::TWO>(
        __ubuf__ ElementOTmp *goUb, __ubuf__ ElementOTmp *loUb,
        __ubuf__ ElementOTmp *dmUb, __ubuf__ ElementOTmp *glUb,
        uint32_t row, uint32_t colStride,
        uint32_t colTail, uint32_t vlElemNum)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> dmVreg;
        RegTensor<float> goPreVreg0;
        RegTensor<float> goPreVreg1;
        RegTensor<float> loVreg0;
        RegTensor<float> loVreg1;
        RegTensor<float> mulVreg0;
        RegTensor<float> mulVreg1;
        RegTensor<float> goCurVreg0;
        RegTensor<float> goCurVreg1;
        RegTensor<float> glVreg;
        RegTensor<float> divVreg0;
        RegTensor<float> divVreg1;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTail = UpdateMask<float>(colTail);
        for (uint16_t i = 0; i < row; i++) {
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(dmVreg, dmUb + i);
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(glVreg, glUb + i);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(goPreVreg0, goUb + i * colStride);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(goPreVreg1, goUb + i * colStride + vlElemNum);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(loVreg0, loUb + i * colStride);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(loVreg1, loUb + i * colStride + vlElemNum);
            Mul(mulVreg0, goPreVreg0, dmVreg, pregFull);
            Mul(mulVreg1, goPreVreg1, dmVreg, pregTail);
            Add(goCurVreg0, mulVreg0, loVreg0, pregFull);
            Add(goCurVreg1, mulVreg1, loVreg1, pregTail);
            Div(divVreg0, goCurVreg0, glVreg, pregFull);
            Div(divVreg1, goCurVreg1, glVreg, pregTail);
            if constexpr (FULL_QUANT_FP8) {
                Muls(divVreg0, divVreg0, MAX_VALUE_RECIPROCAL_FP8, pregFull);
                Muls(divVreg1, divVreg1, MAX_VALUE_RECIPROCAL_FP8, pregTail);
            }
            StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(goUb + i * colStride, divVreg0, pregFull);
            StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(goUb + i * colStride + vlElemNum, divVreg1, pregTail);
        }
    }

    template <DRegSplitStages dRegSplitStages>
    __simd_vf__ inline void DivFuncLastAndFirst(
        __ubuf__ ElementOTmp *goUb, __ubuf__ ElementOTmp *loUb, __ubuf__ ElementOTmp *glUb,
        uint32_t row, uint32_t colStride,
        uint32_t colTail, uint32_t vlElemNum)
    {
    }

    template <>
    __simd_vf__ inline void DivFuncLastAndFirst<DRegSplitStages::ONE>(
        __ubuf__ ElementOTmp *goUb, __ubuf__ ElementOTmp *loUb, __ubuf__ ElementOTmp *glUb,
        uint32_t row, uint32_t colStride,
        uint32_t colTail, uint32_t vlElemNum)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> goCurVreg;
        RegTensor<float> glVreg;
        RegTensor<float> divVreg;
        MaskReg pregTail = UpdateMask<float>(colTail);
        for (uint16_t i = 0; i < row; i++) {
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(glVreg, glUb + i);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(goCurVreg, loUb + i * colStride);
            Div(divVreg, goCurVreg, glVreg, pregTail);
            if constexpr (FULL_QUANT_FP8) {
                Muls(divVreg, divVreg, MAX_VALUE_RECIPROCAL_FP8, pregTail);
            }
            StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(goUb + i * colStride, divVreg, pregTail);
        }
    }

    template <>
    __simd_vf__ inline void DivFuncLastAndFirst<DRegSplitStages::TWO>(
        __ubuf__ ElementOTmp *goUb, __ubuf__ ElementOTmp *loUb, __ubuf__ ElementOTmp *glUb,
        uint32_t row, uint32_t colStride,
        uint32_t colTail, uint32_t vlElemNum)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> goCurVreg0;
        RegTensor<float> goCurVreg1;
        RegTensor<float> glVreg;
        RegTensor<float> divVreg0;
        RegTensor<float> divVreg1;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTail = UpdateMask<float>(colTail);
        for (uint16_t i = 0; i < row; i++) {
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(glVreg, glUb + i);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(goCurVreg0, loUb + i * colStride);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(goCurVreg1, loUb + i * colStride + vlElemNum);
            Div(divVreg0, goCurVreg0, glVreg, pregFull);
            Div(divVreg1, goCurVreg1, glVreg, pregTail);
            if constexpr (FULL_QUANT_FP8) {
                Muls(divVreg0, divVreg0, MAX_VALUE_RECIPROCAL_FP8, pregFull);
                Muls(divVreg1, divVreg1, MAX_VALUE_RECIPROCAL_FP8, pregTail);
            }
            StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(goUb + i * colStride, divVreg0, pregFull);
            StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(goUb + i * colStride + vlElemNum, divVreg1, pregTail);
        }
    }

    // When qS can be copied from UB to GM continuously,
    // qS would be stored from reg to UB continuously.
    __simd_vf__ inline void LogSumExpFuncQSAxisContinuous(
        __ubuf__ float *gmUb, __ubuf__ float *glUb, __ubuf__ float *lseUb,
        uint32_t row)
    {
        // This vf works only when the rowNum in each AIV does not exceed 64
        using namespace AscendC::MicroAPI;
        RegTensor<float> gmVreg;
        RegTensor<float> glVreg;
        RegTensor<float> logGlVreg;
        RegTensor<float> lseVreg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTail = UpdateMask<float>(row);
        static constexpr LnSpecificMode mode = {MaskMergeMode::ZEROING, AscendC::LnAlgo::PRECISION_1ULP_FTZ_FALSE};

        LoadAlign<float, LoadDist::DIST_NORM>(glVreg, glUb);
        LoadAlign<float, LoadDist::DIST_NORM>(gmVreg, gmUb);
        Ln<float, &mode>(logGlVreg, glVreg, pregTail);
        Add(lseVreg, logGlVreg, gmVreg, pregTail);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(lseUb, lseVreg, pregTail);
    }

    // When qS cannot be copied from UB to GM continuously,
    // lse would be broadcasted and then stored from reg to UB continuously.
    __simd_vf__ inline void LogSumExpFuncQSAxisIncontinuous(
        __ubuf__ float *gmUb, __ubuf__ float *glUb, __ubuf__ float *lseUb,
        uint32_t row)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> gmRowwiseVreg0;
        RegTensor<float> gmRowwiseVreg1;
        RegTensor<float> glRowwiseVreg0;
        RegTensor<float> glRowwiseVreg1;
        RegTensor<float> logGlRowwiseVreg0;
        RegTensor<float> logGlRowwiseVreg1;
        RegTensor<float> lseRowwiseVreg0;
        RegTensor<float> lseRowwiseVreg1;
        UnalignReg rowwiseUreg0;
        UnalignReg rowwiseUreg1;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        static constexpr LnSpecificMode mode = {MaskMergeMode::ZEROING, AscendC::LnAlgo::PRECISION_1ULP_FTZ_FALSE};
        static constexpr uint32_t postUpdateStride = 32 / sizeof(float);

        for (uint16_t i = 0; i < row; i+=2) {
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(glRowwiseVreg0, glUb + i);
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(glRowwiseVreg1, glUb + (i + 1));
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(gmRowwiseVreg0, gmUb + i);
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(gmRowwiseVreg1, gmUb + (i + 1));
            Ln<float, &mode>(logGlRowwiseVreg0, glRowwiseVreg0, pregFull);
            Ln<float, &mode>(logGlRowwiseVreg1, glRowwiseVreg1, pregFull);
            Add(lseRowwiseVreg0, logGlRowwiseVreg0, gmRowwiseVreg0, pregFull);
            Add(lseRowwiseVreg1, logGlRowwiseVreg1, gmRowwiseVreg1, pregFull);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(lseUb, lseRowwiseVreg0, rowwiseUreg0, postUpdateStride);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(lseUb, lseRowwiseVreg1, rowwiseUreg1, postUpdateStride);
        }
        StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(lseUb, rowwiseUreg0, postUpdateStride);
        StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(lseUb, rowwiseUreg1, postUpdateStride);
    }

    template <class TensorO, class TensorLse>
    __aicore__ inline
    void operator()(TensorO &gOTensor,
                    TensorLse &gLseTensor,
                    GemmCoord actualOriShape,
                    uint32_t curTileMod,
                    uint32_t gatheredKvSTileIdx,
                    bool isFirstKvSTile,
                    bool isLastKvSTile,
                    Arch::CrossCoreFlag mm2ToReFlag)
    {
        uint32_t rowNumOri = actualOriShape[0];
        uint32_t colNumOri = actualOriShape[1];
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        uint32_t rowNumOriAligned8 = RoundUp(rowNumOri, 8);
        uint32_t colNumOriAligned8 = RoundUp(colNumOri, 8);

        uint32_t rowNumSplit = rowNumOriAligned8 / subBlockNum;
        rowNumSplit = (rowNumOri < rowNumSplit) ? rowNumOri : rowNumSplit;
        uint32_t rowNumCurSubCore = (subBlockIdx == 0) ? rowNumSplit : (rowNumOri - rowNumSplit);
        uint32_t rowOffsetCurSubCore = rowNumSplit * subBlockIdx;
        uint32_t colNumCurSubCore = colNumOri;
        uint32_t colStrideCurSubCore = colNumOriAligned8;

        auto gOTensorTlaTile = GetTile(gOTensor,
            tla::MakeCoord(rowOffsetCurSubCore, 0), tla::MakeShape(rowNumCurSubCore, colNumCurSubCore));
        auto gLseTensorTlaTile = GetTile(gLseTensor,
            tla::MakeCoord(rowOffsetCurSubCore, 0), tla::MakeShape(rowNumCurSubCore, 1));
        uint32_t ubOTmpBufId = gatheredKvSTileIdx % UB_OTMP_BUF_STAGES;
        AscendC::GlobalTensor<float> dummyPartialO;
        AscendC::GlobalTensor<float> dummyPartialLse;

        if (rowNumCurSubCore > 0) {
            SubCoreCompute<false>(
                gOTensorTlaTile,
                gLseTensorTlaTile,
                curTileMod,
                ubOTmpBufId,
                isFirstKvSTile,
                isLastKvSTile,
                colStrideCurSubCore,
                mm2ToReFlag, dummyPartialO, dummyPartialLse, 0U,
                rowOffsetCurSubCore, rowNumOri, 0U);
        } else {
            Arch::CrossCoreWaitFlag<4, PIPE_V>(mm2ToReFlag);
            Arch::CrossCoreSetFlag<4, PIPE_V>(mm2ToReFlag);
        }
    }

    template <class TensorO, class TensorLse>
    __aicore__ inline
    void ProcessPartial(TensorO &gOTensor,
                        TensorLse &gLseTensor,
                        GemmCoord actualOriShape,
                        uint32_t curTileMod,
                        uint32_t gatheredKvSTileIdx,
                        bool isFirstKvSTile,
                        bool isLastKvSTile,
                        Arch::CrossCoreFlag mm2ToReFlag,
                        AscendC::GlobalTensor<float> &gPartialO,
                        AscendC::GlobalTensor<float> &gPartialLse,
                        uint32_t fdTaskId,
                        uint32_t fdLseSubStride)
    {
        const uint32_t rowNumOri = actualOriShape[0];
        const uint32_t colNumOri = actualOriShape[1];
        const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t rowNumSplit = RoundUp(rowNumOri, 8) / AscendC::GetSubBlockNum();
        rowNumSplit = (rowNumOri < rowNumSplit) ? rowNumOri : rowNumSplit;
        const uint32_t rowNumCurSubCore = subBlockIdx == 0 ? rowNumSplit : rowNumOri - rowNumSplit;
        const uint32_t rowOffsetCurSubCore = rowNumSplit * subBlockIdx;
        const uint32_t colStrideCurSubCore = RoundUp(colNumOri, 8);
        auto gOTensorTlaTile = GetTile(gOTensor,
            tla::MakeCoord(rowOffsetCurSubCore, 0), tla::MakeShape(rowNumCurSubCore, colNumOri));
        auto gLseTensorTlaTile = GetTile(gLseTensor,
            tla::MakeCoord(rowOffsetCurSubCore, 0), tla::MakeShape(rowNumCurSubCore, 1));
        const uint32_t ubOTmpBufId = gatheredKvSTileIdx % UB_OTMP_BUF_STAGES;
        if (rowNumCurSubCore > 0) {
            SubCoreCompute<true>(gOTensorTlaTile, gLseTensorTlaTile, curTileMod, ubOTmpBufId,
                isFirstKvSTile, isLastKvSTile, colStrideCurSubCore, mm2ToReFlag,
                gPartialO, gPartialLse, fdTaskId, rowOffsetCurSubCore, rowNumOri, fdLseSubStride);
        } else {
            Arch::CrossCoreWaitFlag<4, PIPE_V>(mm2ToReFlag);
            Arch::CrossCoreSetFlag<4, PIPE_V>(mm2ToReFlag);
        }
    }

    __aicore__ inline
    void WriteNeutralPartial(AscendC::GlobalTensor<float> &gPartialO,
                             AscendC::GlobalTensor<float> &gPartialLse,
                             uint32_t fdTaskId,
                             uint32_t groupSize,
                             uint32_t colNum,
                             uint32_t fdLseSubStride)
    {
        const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t rowNumSplit = RoundUp(groupSize, 8) / AscendC::GetSubBlockNum();
        rowNumSplit = groupSize < rowNumSplit ? groupSize : rowNumSplit;
        const uint32_t rowNum = subBlockIdx == 0 ? rowNumSplit : groupSize - rowNumSplit;
        const uint32_t rowOffset = rowNumSplit * subBlockIdx;
        if (rowNum == 0) {
            return;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::Duplicate(goUbTensor32, 0.0f, rowNum * colNum);
        AscendC::Duplicate(lseUbTensor32, -3.402823466e+38F, RoundUp(rowNum, 8));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        const uint64_t oOffset = static_cast<uint64_t>(fdTaskId) * groupSize * colNum +
            static_cast<uint64_t>(rowOffset) * colNum;
        AscendC::DataCopy(gPartialO[oOffset], goUbTensor32, rowNum * colNum);
        const uint64_t lseOffset = static_cast<uint64_t>(fdTaskId) * 2 * fdLseSubStride +
            subBlockIdx * fdLseSubStride;
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        AscendC::DataCopyPad(gPartialLse[lseOffset], lseUbTensor32,
            AscendC::DataCopyExtParams(1, rowNum * sizeof(float), 0, 0, 0));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
    }

    template <class TensorO, class TensorLse>
    __aicore__ inline
    void WriteEmptyOutput(TensorO &gOTensor, TensorLse &gLseTensor, GemmCoord actualOriShape)
    {
        const uint32_t rowNumOri = actualOriShape[0];
        const uint32_t colNumOri = actualOriShape[1];
        const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t rowNumSplit = RoundUp(rowNumOri, 8) / AscendC::GetSubBlockNum();
        rowNumSplit = rowNumOri < rowNumSplit ? rowNumOri : rowNumSplit;
        const uint32_t rowNum = subBlockIdx == 0 ? rowNumSplit : rowNumOri - rowNumSplit;
        const uint32_t rowOffset = rowNumSplit * subBlockIdx;
        if (rowNum == 0) {
            return;
        }
        const uint32_t colStride = RoundUp(colNumOri, 8);
        auto gOTile = GetTile(gOTensor, tla::MakeCoord(rowOffset, 0), tla::MakeShape(rowNum, colNumOri));
        auto gLseTile = GetTile(gLseTensor, tla::MakeCoord(rowOffset, 0), tla::MakeShape(rowNum, 1));
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::Duplicate(goUbTensor16, static_cast<ElementO>(0), rowNum * colStride);
        if constexpr (DispatchPolicy::LSE_MODE == LseMode::OUT_ONLY) {
            // TN1/BSN1 LSE rows use an 8-float UB stride in CopyUbToGmLse.
            // Initialize the complete strided span, otherwise only head 0 is
            // deterministic and the remaining empty-head LSE values expose UB
            // residue.
            AscendC::Duplicate(lseUbTensor32, -3.402823466e+38F, rowNum * 8U);
        }
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (DispatchPolicy::LSE_MODE == LseMode::OUT_ONLY) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            auto ubLseLayout = tla::MakeLayout(tla::MakeShape(rowNum, 8U),
                                               tla::MakeStride(8U, tla::Int<1>{}));
            auto ubLseTensor = tla::MakeTensor(lseUbTensor32, ubLseLayout, Arch::PositionUB{});
            CopyUbToGmLse(gLseTile, ubLseTensor);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        auto ubOLayout = tla::MakeLayout(tla::MakeShape(rowNum, colNumOri),
                                         tla::MakeStride(colStride, tla::Int<1>{}));
        auto ubOTensor = tla::MakeTensor(goUbTensor16, ubOLayout, Arch::PositionUB{});
        copyUbToGmO(gOTile, ubOTensor);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
    }

private:
    AscendC::LocalTensor<ElementOTmp> loUbTensor[UB_OTMP_BUF_STAGES];
    AscendC::LocalTensor<SMDtype> dmUbTensor16;
    AscendC::LocalTensor<SMDtype> glUbTensor16;
    AscendC::LocalTensor<float> gmUbTensor32;
    AscendC::LocalTensor<float> dmUbTensor32;
    AscendC::LocalTensor<float> glUbTensor32;
    AscendC::LocalTensor<ElementO> goUbTensor16;
    AscendC::LocalTensor<ElementOTmp> goUbTensor32;
    AscendC::LocalTensor<float> lseUbTensor32;

    CopyUbToGmO copyUbToGmO;
};
}
#endif
