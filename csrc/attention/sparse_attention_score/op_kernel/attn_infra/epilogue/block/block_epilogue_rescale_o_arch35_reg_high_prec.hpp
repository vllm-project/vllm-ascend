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
    class TileCopy_,
    class OTmpSrcPos_ // the src TPosition of pv res, viable configurations: GM/L0C
>
class BlockEpilogue<
    EpilogueAtlasA5BsaRescaleO,
    ElementO_,
    ElementOTmp_,
    ElementS_,
    TileCopy_,
    OTmpSrcPos_>
{
public:
    using DispatchPolicy = EpilogueAtlasA5BsaRescaleO;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementO = ElementO_;
    using ElementOTmp = ElementOTmp_;
    using SMDtype = ElementS_;
    using TileCopy = TileCopy_;
    using OTmpSrcPos = OTmpSrcPos_;

    using CopyUbToGmO = typename TileCopy::CopyUbToGmO;

    static constexpr uint32_t UB_OTMP_BUF_STAGES = 2;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 32768;
    static constexpr uint32_t DM_UB_GLOBAL_ELEM_NUM = 64;
    static constexpr uint32_t RESCALE_ROW_MAX_ELEM_NUM = 64;
    static constexpr uint32_t RESCALE_COL_MAX_ELEM_NUM = 128;
    static constexpr uint32_t RESCALE_VREG_SIZE = 256 / sizeof(ElementOTmp);

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

        for (uint32_t i = 0; i < UB_OTMP_BUF_STAGES; i++) {
            loUbTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(
                LO_UB_TENSOR_OFFSET + i * UB_UINT8_BLOCK_SIZE);
        }
        goUbTensor32 = resource.ubBuf.template GetBufferByByte<ElementOTmp>(GO_UB_TENSOR_OFFSET);
        goUbTensor16 = resource.ubBuf.template GetBufferByByte<ElementO>(GO_UB_TENSOR_OFFSET);
        glUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
        dmUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
    }

    __aicore__ inline
    ~BlockEpilogue()
    {
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

    template <class TensorDst>
    __aicore__ inline
    void SubCoreCompute(TensorDst &gOTensorTlaTile,
                        uint32_t curTileMod,
                        uint32_t ubOTmpBufId,
                        bool isFirstKvSTile,
                        bool isLastKvSTile,
                        uint32_t colStrideCurSubCore,
                        Arch::CrossCoreFlag mm2ToReFlag,
                        bool isFullQuantFp8)
    {
        uint32_t rowNumCurSubCore = tla::get<0>(gOTensorTlaTile.shape());
        uint32_t colNumCurSubCore = tla::get<1>(gOTensorTlaTile.shape());
        uint32_t vlElemNum = AscendC::GetVecLen() / sizeof(ElementOTmp);
        uint32_t colFullLoop = CeilDiv(colNumCurSubCore, vlElemNum) - 1;
        uint32_t colTail = (colNumCurSubCore - 1) % vlElemNum + 1;
        uint32_t dStages = CeilDiv(colNumCurSubCore, RESCALE_VREG_SIZE);

        __ubuf__ ElementOTmp *goUb = (__ubuf__ ElementOTmp *) goUbTensor32.GetPhyAddr();
        __ubuf__ ElementOTmp *loUb = (__ubuf__ ElementOTmp *) loUbTensor[ubOTmpBufId].GetPhyAddr();
        __ubuf__ ElementOTmp *glUb = ( __ubuf__ ElementOTmp *) glUbTensor32.GetPhyAddr();
        __ubuf__ ElementOTmp *dmUb =
            (__ubuf__ ElementOTmp *) dmUbTensor32[curTileMod * DM_UB_GLOBAL_ELEM_NUM].GetPhyAddr();
        
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
                // DivFuncLastAndFirst<ElementOTmp>(
                //     goUb, loUb, glUb, rowNumCurSubCore, colStrideCurSubCore, colFullLoop, colTail, vlElemNum);
            }
        } else if (!isLastKvSTile) {
            if (dStages == 1) {
                RescaleFunc<DRegSplitStages::ONE>(
                    goUb, loUb, dmUb, rowNumCurSubCore, colStrideCurSubCore, colTail, vlElemNum);
            } else if (dStages == 2) {
                RescaleFunc<DRegSplitStages::TWO>(
                    goUb, loUb, dmUb, rowNumCurSubCore, colStrideCurSubCore, colTail, vlElemNum);
            }
            // RescaleFunc<ElementOTmp>(
            //     goUb, loUb, dmUb, rowNumCurSubCore, colStrideCurSubCore, colFullLoop, colTail, vlElemNum);
        } else {
            if (dStages == 1) {
                RescaleFuncLastNotFirst<DRegSplitStages::ONE>(
                goUb, loUb, dmUb, glUb, rowNumCurSubCore, colStrideCurSubCore, colTail, vlElemNum);
            } else if (dStages == 2) {
                RescaleFuncLastNotFirst<DRegSplitStages::TWO>(
                goUb, loUb, dmUb, glUb, rowNumCurSubCore, colStrideCurSubCore, colTail, vlElemNum);
            }
            // RescaleFuncLastNotFirst<ElementOTmp>(
            //     goUb, loUb, dmUb, glUb, rowNumCurSubCore, colStrideCurSubCore, colFullLoop, colTail, vlElemNum);
        }
        // release lo buf
        SetCrossCoreSync<4, PIPE_V>(mm2ToReFlag);
        if (isLastKvSTile) {
            AscendC::PipeBarrier<PIPE_V>();
            if (isFullQuantFp8) {
                if (dStages == 1) {
                    deQuantScaleO<DRegSplitStages::ONE>(goUb, rowNumCurSubCore, colStrideCurSubCore, colTail,
                                                        vlElemNum);
                } else if (dStages == 2) {
                    deQuantScaleO<DRegSplitStages::TWO>(goUb, rowNumCurSubCore, colStrideCurSubCore, colTail,
                                                        vlElemNum);
                }
                AscendC::PipeBarrier<PIPE_V>();
            }
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
            AscendC::PipeBarrier<PIPE_V>();
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
        for (uint32_t i = 0; i < row; i++) {
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
        for (uint32_t i = 0; i < row; i++) {
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
        for (uint32_t i = 0; i < row; i++) {
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(dmVreg, dmUb + i);
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(glVreg, glUb + i);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(goPreVreg, goUb + i * colStride);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(loVreg, loUb + i * colStride);
            Mul(mulVreg, goPreVreg, dmVreg, pregTail);
            Add(goCurVreg, mulVreg, loVreg, pregTail);
            Div(divVreg, goCurVreg, glVreg, pregTail);
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
        for (uint32_t i = 0; i < row; i++) {
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
        for (uint32_t i = 0; i < row; i++) {
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(glVreg, glUb + i);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(goCurVreg, loUb + i * colStride);
            Div(divVreg, goCurVreg, glVreg, pregTail);
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
        for (uint32_t i = 0; i < row; i++) {
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(glVreg, glUb + i);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(goCurVreg0, loUb + i * colStride);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(goCurVreg1, loUb + i * colStride + vlElemNum);
            Div(divVreg0, goCurVreg0, glVreg, pregFull);
            Div(divVreg1, goCurVreg1, glVreg, pregTail);
            StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(goUb + i * colStride, divVreg0, pregFull);
            StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(goUb + i * colStride + vlElemNum, divVreg1, pregTail);
        }
    }

    template <DRegSplitStages dRegSplitStages>
    __simd_vf__ inline void deQuantScaleO(__ubuf__ ElementOTmp *goUb, uint32_t row, uint32_t colStride,
                                          uint32_t colTail, uint32_t vlElemNum)
    {
    }

    template <>
    __simd_vf__ inline void deQuantScaleO<DRegSplitStages::ONE>(__ubuf__ ElementOTmp *goUb, uint32_t row,
                                                                uint32_t colStride, uint32_t colTail,
                                                                uint32_t vlElemNum)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<ElementOTmp> oVreg;
        MaskReg pregTail = UpdateMask<float>(colTail);

        for (uint16_t i = 0; i < row; ++i) {
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(oVreg, goUb + i * colStride);
            constexpr float maxValueReciprocal = 1.0f / 448.0f;
            Muls(oVreg, oVreg, maxValueReciprocal, pregTail);
            StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(goUb + i * colStride, oVreg, pregTail);
        }
    }

    template <>
    __simd_vf__ inline void deQuantScaleO<DRegSplitStages::TWO>(__ubuf__ ElementOTmp *goUb, uint32_t row,
                                                                uint32_t colStride, uint32_t colTail,
                                                                uint32_t vlElemNum)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<ElementOTmp> oVreg0;
        RegTensor<ElementOTmp> oVreg1;
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTail = UpdateMask<float>(colTail);

        for (uint16_t i = 0; i < row; ++i) {
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(oVreg0, goUb + i * colStride);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(oVreg1, goUb + i * colStride + vlElemNum);
            constexpr float maxValueReciprocal = 1.0f / 448.0f;
            Muls(oVreg0, oVreg0, maxValueReciprocal, pregAll);
            Muls(oVreg1, oVreg1, maxValueReciprocal, pregTail);
            StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(goUb + i * colStride, oVreg0, pregAll);
            StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(goUb + i * colStride + vlElemNum, oVreg1, pregTail);
        }
    }

    template <class TensorDst>
    __aicore__ inline
    void operator()(TensorDst &gOTensor,
                    GemmCoord actualOriShape,
                    uint32_t curTileMod,
                    uint32_t gatheredKvSTileIdx,
                    bool isFirstKvSTile,
                    bool isLastKvSTile,
                    Arch::CrossCoreFlag mm2ToReFlag,
                    bool isFullQuantFp8 = false)
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
        uint32_t ubOTmpBufId = gatheredKvSTileIdx % UB_OTMP_BUF_STAGES;

        if (rowNumCurSubCore > 0) {
            SubCoreCompute(
                gOTensorTlaTile,
                curTileMod,
                ubOTmpBufId,
                isFirstKvSTile,
                isLastKvSTile,
                colStrideCurSubCore,
                mm2ToReFlag,
                isFullQuantFp8);
        } else {
            Arch::CrossCoreWaitFlag<4, PIPE_V>(mm2ToReFlag);
            Arch::CrossCoreSetFlag<4, PIPE_V>(mm2ToReFlag);
        }
    }

private:
    AscendC::LocalTensor<ElementOTmp> loUbTensor[UB_OTMP_BUF_STAGES];
    AscendC::LocalTensor<SMDtype> dmUbTensor16;
    AscendC::LocalTensor<SMDtype> glUbTensor16;
    AscendC::LocalTensor<float> dmUbTensor32;
    AscendC::LocalTensor<float> glUbTensor32;
    AscendC::LocalTensor<ElementO> goUbTensor16;
    AscendC::LocalTensor<ElementOTmp> goUbTensor32;

    CopyUbToGmO copyUbToGmO;

};
}
#endif