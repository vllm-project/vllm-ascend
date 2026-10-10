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

#include "../../../attn_infra/gbsa_base_defs.hpp"
#include "../../../attn_infra/arch/gbsa_resource.hpp"
#include "../../../attn_infra/epilogue/gbsa_epilogue_dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/block/gbsa_block_epilogue_arch35_utils.hpp"
#include "../../../attn_infra/epilogue/tile_common/gbsa_epilogue_tile_copy.hpp"
#include "../../../attn_infra/gbsa_gemm_coord.hpp"
#include "../../../attn_infra/gbsa_matrix_coord.hpp"
#include "../../../tla/gbsa_tla_tensor.hpp"
#include "../../../tla/gbsa_tla_layout.hpp"

namespace NpuArch::Epilogue::Block {

template <class ElementO_, class ElementOTmp_, class ElementS_, class ElementKV_, class TileCopy_,
          class OTmpSrcPos_, // the src TPosition of pv res, viable configurations: GM/L0C
          LseMode LSE_MODE_, LseFormat LSE_FORMAT_>
class BlockEpilogue<EpilogueAtlasA5BsaRescaleO<LSE_MODE_, LSE_FORMAT_>, ElementO_, ElementOTmp_, ElementS_, ElementKV_,
                    TileCopy_, OTmpSrcPos_> {
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
    static constexpr uint32_t RESCALE_ROW_MAX_ELEM_NUM = 64;
    static constexpr uint32_t RESCALE_COL_MAX_ELEM_NUM = 128;
    static constexpr uint32_t RESCALE_VREG_SIZE = 256 / sizeof(ElementOTmp);
    static constexpr float MAX_VALUE_RECIPROCAL_FP8 = 1.0f / 448.0f;
    static constexpr bool FULL_QUANT_FP8 = AscendC::IsSameType<ElementKV, fp8_e4m3fn_t>::value;
    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag> &resource, UBufTileHelper &uBufTileHelper)
    {
        loUbBufNum = uBufTileHelper.loUbBufNum;
        loUbRowNum = uBufTileHelper.loUbRowNum;
        loGmTransit = uBufTileHelper.loGmTransit;
        for (uint32_t i = 0; i < UB_OTMP_BUF_STAGES; i++) {
            loUbTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(
                uBufTileHelper.loStartOffset +
                loUbRowNum * uBufTileHelper.embedPerSubCore * sizeof(ElementOTmp) *
                    (i % loUbBufNum));
        }
        goUbTensor32 = resource.ubBuf.template GetBufferByByte<ElementOTmp>(uBufTileHelper.goStartOffset);
        goUbTensor16 = resource.ubBuf.template GetBufferByByte<ElementO>(uBufTileHelper.goStartOffset);
        glUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(uBufTileHelper.glStartOffset);
        dmUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(uBufTileHelper.dmStartOffset);
        gmUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(uBufTileHelper.gmStartOffset);
        lseUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(uBufTileHelper.lseStartOffset);
    }

    __aicore__ inline ~BlockEpilogue() {}

    __aicore__ inline void SetLoWorkspace(AscendC::GlobalTensor<ElementOTmp> workspace, uint64_t tileElems)
    {
        gLo = workspace;
        loGmTileElems = tileElems;
    }

    __aicore__ inline void ComputeLoRows(uint32_t rowOffset, uint32_t rows, uint32_t colStride,
                                        uint32_t cols, uint32_t curTileMod, uint32_t loBufId,
                                        bool isFirst, bool isLast)
    {
        auto go = goUbTensor32[rowOffset * colStride];
        auto lo = loUbTensor[loBufId];
        auto *goPtr = reinterpret_cast<__ubuf__ ElementOTmp *>(go.GetPhyAddr());
        auto *loPtr = reinterpret_cast<__ubuf__ ElementOTmp *>(lo.GetPhyAddr());
        auto *glPtr = reinterpret_cast<__ubuf__ ElementOTmp *>(glUbTensor32[rowOffset].GetPhyAddr());
        auto *dmPtr = reinterpret_cast<__ubuf__ ElementOTmp *>(
            dmUbTensor32[curTileMod * RESCALE_ROW_MAX_ELEM_NUM + rowOffset].GetPhyAddr());
        const uint16_t vl = AscendC::GetVecLen() / sizeof(ElementOTmp);
        const uint32_t tail = (cols - 1U) % vl + 1U;
        if (isFirst && !isLast) {
            AscendC::DataCopy(go, lo, rows * colStride);
            AscendC::PipeBarrier<PIPE_V>();
        } else if (isFirst) {
            DivFuncLastAndFirst(goPtr, loPtr, glPtr, colStride, tail, vl, rows / 2U, cols / vl,
                                rows % 2U, cols % vl != 0U);
        } else if (!isLast) {
            RescaleFunc(goPtr, loPtr, dmPtr, colStride, tail, vl, rows / 2U, cols / vl,
                        rows % 2U, cols % vl != 0U);
        } else {
            RescaleFuncLastNotFirst(goPtr, loPtr, dmPtr, glPtr, colStride, tail, vl, rows / 2U, cols / vl,
                                    rows % 2U, cols % vl != 0U);
        }
    }

    __aicore__ inline void LoadLoRows(uint64_t gmOffset, uint32_t rows, uint32_t stride, uint32_t bufId)
    {
        const uint32_t eventId = 6U + bufId;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventId);
        AscendC::DataCopy(loUbTensor[bufId], gLo[gmOffset], rows * stride);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline void SetCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        // in mode 4, AIC set for 2 AIVs seperately
        if constexpr (MODE == 4U) {
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline void WaitCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        // in mode 4, AIC wait for 2 AIVs seperately
        if constexpr (MODE == 4U) {
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    template <class TensorLseGm, class TensorLseUb>
    __aicore__ inline void CopyUbToGmLse(TensorLseGm const &gLseTensorTlaTile, TensorLseUb const &ubLseTensorTla)
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
    __aicore__ inline void SubCoreCompute(TensorO &gOTensorTlaTile, TensorLse &gLseTensorTlaTile, uint32_t curTileMod,
                                          uint32_t ubOTmpBufId, bool isFirstKvSTile, bool isLastKvSTile,
                                          uint32_t colStrideCurSubCore, Arch::CrossCoreFlag mm2ToReFlag,
                                          AscendC::GlobalTensor<float> &gPartialO,
                                          AscendC::GlobalTensor<float> &gPartialLse, uint32_t fdTaskId,
                                          uint32_t groupRowOffset, uint32_t groupSize, uint32_t fdLseSubStride)
    {
        uint32_t rowNumCurSubCore = tla::get<0>(gOTensorTlaTile.shape());
        uint32_t colNumCurSubCore = tla::get<1>(gOTensorTlaTile.shape());
        uint32_t vlElemNum = AscendC::GetVecLen() / sizeof(ElementOTmp);
        uint32_t colTail = (colNumCurSubCore - 1) % vlElemNum + 1;
        uint16_t mLoop = rowNumCurSubCore / 2U;
        uint16_t nLoop = colNumCurSubCore / vlElemNum;
        uint16_t hasMTail = rowNumCurSubCore % 2U;
        uint16_t hasNTail = colNumCurSubCore % vlElemNum != 0U;

        __ubuf__ ElementOTmp *goUb = (__ubuf__ ElementOTmp *)goUbTensor32.GetPhyAddr();
        __ubuf__ ElementOTmp *loUb = (__ubuf__ ElementOTmp *)loUbTensor[ubOTmpBufId].GetPhyAddr();
        __ubuf__ ElementOTmp *glUb = (__ubuf__ ElementOTmp *)glUbTensor32.GetPhyAddr();
        __ubuf__ ElementOTmp *dmUb =
            (__ubuf__ ElementOTmp *)dmUbTensor32[curTileMod * RESCALE_ROW_MAX_ELEM_NUM].GetPhyAddr();
        __ubuf__ float *gmUb = (__ubuf__ float *)gmUbTensor32.GetPhyAddr();
        __ubuf__ float *lseUb = (__ubuf__ float *)lseUbTensor32.GetPhyAddr();

        if (loGmTransit) {
            WaitCrossCoreSync<4, PIPE_MTE2>(mm2ToReFlag);
        } else {
            WaitCrossCoreSync<4, PIPE_V>(mm2ToReFlag);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        if (loGmTransit) {
            const uint64_t gmOffset = ubOTmpBufId * loGmTileElems +
                                      static_cast<uint64_t>(groupRowOffset) * colStrideCurSubCore;
            const uint32_t chunks = CeilDiv(rowNumCurSubCore, loUbRowNum);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID7);
            LoadLoRows(gmOffset, min(loUbRowNum, rowNumCurSubCore), colStrideCurSubCore, 0U);
            for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
                const uint32_t bufId = chunk % UB_OTMP_BUF_STAGES;
                const uint32_t rowOffset = chunk * loUbRowNum;
                if (chunk + 1U < chunks) {
                    const uint32_t nextRow = rowOffset + loUbRowNum;
                    LoadLoRows(gmOffset + static_cast<uint64_t>(nextRow) * colStrideCurSubCore,
                               min(loUbRowNum, rowNumCurSubCore - nextRow), colStrideCurSubCore, 1U - bufId);
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(6U + bufId);
                ComputeLoRows(rowOffset, min(loUbRowNum, rowNumCurSubCore - rowOffset), colStrideCurSubCore,
                              colNumCurSubCore, curTileMod, bufId, isFirstKvSTile, isLastKvSTile);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(6U + bufId);
            }
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID7);
        } else if (isFirstKvSTile) {
            if (!isLastKvSTile) {
                uint32_t totalCopyElems = rowNumCurSubCore * colStrideCurSubCore;
                AscendC::DataCopy(goUbTensor32, loUbTensor[ubOTmpBufId], totalCopyElems);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                DivFuncLastAndFirst(goUb, loUb, glUb, colStrideCurSubCore, colTail, vlElemNum, mLoop, nLoop,
                                    hasMTail, hasNTail);
            }
        } else if (!isLastKvSTile) {
            RescaleFunc(goUb, loUb, dmUb, colStrideCurSubCore, colTail, vlElemNum, mLoop, nLoop, hasMTail,
                        hasNTail);
        } else {
            RescaleFuncLastNotFirst(goUb, loUb, dmUb, glUb, colStrideCurSubCore, colTail, vlElemNum, mLoop, nLoop,
                                    hasMTail, hasNTail);
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
                AscendC::DataCopy(gPartialO[oOffset], goUbTensor32, rowNumCurSubCore * colStrideCurSubCore);
                const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
                const uint64_t lseOffset =
                    static_cast<uint64_t>(fdTaskId) * 2 * fdLseSubStride + subBlockIdx * fdLseSubStride;
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
                auto ubLseLayoutTla = tla::MakeLayout(tla::MakeShape(rowNumCurSubCore, colNumLseUb),
                                                      tla::MakeStride(colStrideLseUb, tla::Int<1>{}));
                auto ubLseTensorTla = tla::MakeTensor(lseUbTensor32, ubLseLayoutTla, Arch::PositionUB{});
                CopyUbToGmLse(gLseTensorTlaTile, ubLseTensorTla);
            }
            AscendC::PipeBarrier<PIPE_V>();

            if (std::is_same<ElementO, bfloat16_t>::value) {
                AscendC::Cast(goUbTensor16, goUbTensor32, AscendC::RoundMode::CAST_RINT,
                              rowNumCurSubCore * colStrideCurSubCore);
            } else {
                AscendC::Cast(goUbTensor16, goUbTensor32, AscendC::RoundMode::CAST_NONE,
                              rowNumCurSubCore * colStrideCurSubCore);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            auto ubOLayoutTla = tla::MakeLayout(tla::MakeShape(rowNumCurSubCore, colNumCurSubCore),
                                                tla::MakeStride(colStrideCurSubCore, tla::Int<1>{}));
            auto ubOTensorTla = tla::MakeTensor(goUbTensor16, ubOLayoutTla, Arch::PositionUB{});
            copyUbToGmO(gOTensorTlaTile, ubOTensorTla);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
    }

    static __simd_vf__ inline void RescaleFunc(
        __ubuf__ ElementOTmp *goUb, __ubuf__ ElementOTmp *loUb, __ubuf__ ElementOTmp *dmUb, uint16_t colStride,
        uint32_t colTail, uint16_t vlElemNum, uint16_t mLoop, uint16_t nLoop, uint16_t hasMTail,
        uint16_t hasNTail)
    {
        using namespace AscendC::Reg;
        RegTensor<float> dmVreg0;
        RegTensor<float> dmVreg1;
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

        for (uint16_t i = 0; i < mLoop; i++) {
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(dmVreg0, dmUb + 2U * i);
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(dmVreg1, dmUb + 2U * i + 1U);
            for (uint16_t j = 0; j < nLoop; j++) {
                LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(goPreVreg0,
                                                             goUb + 2U * i * colStride + j * vlElemNum);
                LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(goPreVreg1,
                                                             goUb + (2U * i + 1U) * colStride + j * vlElemNum);
                LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(loVreg0,
                                                             loUb + 2U * i * colStride + j * vlElemNum);
                LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(loVreg1,
                                                             loUb + (2U * i + 1U) * colStride + j * vlElemNum);
                Mul(mulVreg0, goPreVreg0, dmVreg0, pregFull);
                Mul(mulVreg1, goPreVreg1, dmVreg1, pregFull);
                Add(goCurVreg0, mulVreg0, loVreg0, pregFull);
                Add(goCurVreg1, mulVreg1, loVreg1, pregFull);
                StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(
                    goUb + 2U * i * colStride + j * vlElemNum, goCurVreg0, pregFull);
                StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(
                    goUb + (2U * i + 1U) * colStride + j * vlElemNum, goCurVreg1, pregFull);
            }
            for (uint16_t j = 0; j < hasNTail; j++) {
                LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(
                    goPreVreg0, goUb + 2U * i * colStride + (nLoop + j) * vlElemNum);
                LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(
                    goPreVreg1, goUb + (2U * i + 1U) * colStride + (nLoop + j) * vlElemNum);
                LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(loVreg0,
                                                             loUb + 2U * i * colStride + (nLoop + j) * vlElemNum);
                LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(
                    loVreg1, loUb + (2U * i + 1U) * colStride + (nLoop + j) * vlElemNum);
                Mul(mulVreg0, goPreVreg0, dmVreg0, pregTail);
                Mul(mulVreg1, goPreVreg1, dmVreg1, pregTail);
                Add(goCurVreg0, mulVreg0, loVreg0, pregTail);
                Add(goCurVreg1, mulVreg1, loVreg1, pregTail);
                StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(
                    goUb + 2U * i * colStride + (nLoop + j) * vlElemNum, goCurVreg0, pregTail);
                StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(
                    goUb + (2U * i + 1U) * colStride + (nLoop + j) * vlElemNum, goCurVreg1, pregTail);
            }
        }

        for (uint16_t i = 0; i < hasMTail; i++) {
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(dmVreg0, dmUb + 2U * mLoop + i);
            for (uint16_t j = 0; j < nLoop; j++) {
                LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(
                    goPreVreg0, goUb + (2U * mLoop + i) * colStride + j * vlElemNum);
                LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(
                    loVreg0, loUb + (2U * mLoop + i) * colStride + j * vlElemNum);
                Mul(mulVreg0, goPreVreg0, dmVreg0, pregFull);
                Add(goCurVreg0, mulVreg0, loVreg0, pregFull);
                StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(
                    goUb + (2U * mLoop + i) * colStride + j * vlElemNum, goCurVreg0, pregFull);
            }
            for (uint16_t j = 0; j < hasNTail; j++) {
                LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(
                    goPreVreg0, goUb + (2U * mLoop + i) * colStride + (nLoop + j) * vlElemNum);
                LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(
                    loVreg0, loUb + (2U * mLoop + i) * colStride + (nLoop + j) * vlElemNum);
                Mul(mulVreg0, goPreVreg0, dmVreg0, pregTail);
                Add(goCurVreg0, mulVreg0, loVreg0, pregTail);
                StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(
                    goUb + (2U * mLoop + i) * colStride + (nLoop + j) * vlElemNum, goCurVreg0, pregTail);
            }
        }
    }

    static __simd_vf__ inline void RescaleFuncLastNotFirst(
        __ubuf__ ElementOTmp *goUb, __ubuf__ ElementOTmp *loUb, __ubuf__ ElementOTmp *dmUb,
        __ubuf__ ElementOTmp *glUb, uint16_t colStride, uint32_t colTail, uint16_t vlElemNum, uint16_t mLoop,
        uint16_t nLoop, uint16_t hasMTail, uint16_t hasNTail)
    {
        using namespace AscendC::Reg;
        RegTensor<float> dmVreg0;
        RegTensor<float> dmVreg1;
        RegTensor<float> glVreg0;
        RegTensor<float> glVreg1;
        RegTensor<float> goPreVreg0;
        RegTensor<float> goPreVreg1;
        RegTensor<float> loVreg0;
        RegTensor<float> loVreg1;
        RegTensor<float> mulVreg0;
        RegTensor<float> mulVreg1;
        RegTensor<float> goCurVreg0;
        RegTensor<float> goCurVreg1;
        RegTensor<float> divVreg0;
        RegTensor<float> divVreg1;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTail = UpdateMask<float>(colTail);

        for (uint16_t i = 0; i < mLoop; i++) {
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(dmVreg0, dmUb + 2U * i);
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(dmVreg1, dmUb + 2U * i + 1U);
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(glVreg0, glUb + 2U * i);
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(glVreg1, glUb + 2U * i + 1U);
            for (uint16_t j = 0; j < nLoop; j++) {
                LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(goPreVreg0,
                                                             goUb + 2U * i * colStride + j * vlElemNum);
                LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(goPreVreg1,
                                                             goUb + (2U * i + 1U) * colStride + j * vlElemNum);
                LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(loVreg0,
                                                             loUb + 2U * i * colStride + j * vlElemNum);
                LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(loVreg1,
                                                             loUb + (2U * i + 1U) * colStride + j * vlElemNum);
                Mul(mulVreg0, goPreVreg0, dmVreg0, pregFull);
                Mul(mulVreg1, goPreVreg1, dmVreg1, pregFull);
                Add(goCurVreg0, mulVreg0, loVreg0, pregFull);
                Add(goCurVreg1, mulVreg1, loVreg1, pregFull);
                Div(divVreg0, goCurVreg0, glVreg0, pregFull);
                Div(divVreg1, goCurVreg1, glVreg1, pregFull);
                if constexpr (FULL_QUANT_FP8) {
                    Muls(divVreg0, divVreg0, MAX_VALUE_RECIPROCAL_FP8, pregFull);
                    Muls(divVreg1, divVreg1, MAX_VALUE_RECIPROCAL_FP8, pregFull);
                }
                StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(
                    goUb + 2U * i * colStride + j * vlElemNum, divVreg0, pregFull);
                StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(
                    goUb + (2U * i + 1U) * colStride + j * vlElemNum, divVreg1, pregFull);
            }
            for (uint16_t j = 0; j < hasNTail; j++) {
                LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(
                    goPreVreg0, goUb + 2U * i * colStride + (nLoop + j) * vlElemNum);
                LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(
                    goPreVreg1, goUb + (2U * i + 1U) * colStride + (nLoop + j) * vlElemNum);
                LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(loVreg0,
                                                             loUb + 2U * i * colStride + (nLoop + j) * vlElemNum);
                LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(
                    loVreg1, loUb + (2U * i + 1U) * colStride + (nLoop + j) * vlElemNum);
                Mul(mulVreg0, goPreVreg0, dmVreg0, pregTail);
                Mul(mulVreg1, goPreVreg1, dmVreg1, pregTail);
                Add(goCurVreg0, mulVreg0, loVreg0, pregTail);
                Add(goCurVreg1, mulVreg1, loVreg1, pregTail);
                Div(divVreg0, goCurVreg0, glVreg0, pregTail);
                Div(divVreg1, goCurVreg1, glVreg1, pregTail);
                if constexpr (FULL_QUANT_FP8) {
                    Muls(divVreg0, divVreg0, MAX_VALUE_RECIPROCAL_FP8, pregTail);
                    Muls(divVreg1, divVreg1, MAX_VALUE_RECIPROCAL_FP8, pregTail);
                }
                StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(
                    goUb + 2U * i * colStride + (nLoop + j) * vlElemNum, divVreg0, pregTail);
                StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(
                    goUb + (2U * i + 1U) * colStride + (nLoop + j) * vlElemNum, divVreg1, pregTail);
            }
        }

        for (uint16_t i = 0; i < hasMTail; i++) {
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(dmVreg0, dmUb + 2U * mLoop + i);
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(glVreg0, glUb + 2U * mLoop + i);
            for (uint16_t j = 0; j < nLoop; j++) {
                LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(
                    goPreVreg0, goUb + (2U * mLoop + i) * colStride + j * vlElemNum);
                LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(
                    loVreg0, loUb + (2U * mLoop + i) * colStride + j * vlElemNum);
                Mul(mulVreg0, goPreVreg0, dmVreg0, pregFull);
                Add(goCurVreg0, mulVreg0, loVreg0, pregFull);
                Div(divVreg0, goCurVreg0, glVreg0, pregFull);
                if constexpr (FULL_QUANT_FP8) {
                    Muls(divVreg0, divVreg0, MAX_VALUE_RECIPROCAL_FP8, pregFull);
                }
                StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(
                    goUb + (2U * mLoop + i) * colStride + j * vlElemNum, divVreg0, pregFull);
            }
            for (uint16_t j = 0; j < hasNTail; j++) {
                LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(
                    goPreVreg0, goUb + (2U * mLoop + i) * colStride + (nLoop + j) * vlElemNum);
                LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(
                    loVreg0, loUb + (2U * mLoop + i) * colStride + (nLoop + j) * vlElemNum);
                Mul(mulVreg0, goPreVreg0, dmVreg0, pregTail);
                Add(goCurVreg0, mulVreg0, loVreg0, pregTail);
                Div(divVreg0, goCurVreg0, glVreg0, pregTail);
                if constexpr (FULL_QUANT_FP8) {
                    Muls(divVreg0, divVreg0, MAX_VALUE_RECIPROCAL_FP8, pregTail);
                }
                StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(
                    goUb + (2U * mLoop + i) * colStride + (nLoop + j) * vlElemNum, divVreg0, pregTail);
            }
        }
    }

    static __simd_vf__ inline void DivFuncLastAndFirst(
        __ubuf__ ElementOTmp *goUb, __ubuf__ ElementOTmp *loUb, __ubuf__ ElementOTmp *glUb, uint16_t colStride,
        uint32_t colTail, uint16_t vlElemNum, uint16_t mLoop, uint16_t nLoop, uint16_t hasMTail,
        uint16_t hasNTail)
    {
        using namespace AscendC::Reg;
        RegTensor<float> glVreg0;
        RegTensor<float> glVreg1;
        RegTensor<float> goCurVreg0;
        RegTensor<float> goCurVreg1;
        RegTensor<float> divVreg0;
        RegTensor<float> divVreg1;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTail = UpdateMask<float>(colTail);

        for (uint16_t i = 0; i < mLoop; i++) {
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(glVreg0, glUb + 2U * i);
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(glVreg1, glUb + 2U * i + 1U);
            for (uint16_t j = 0; j < nLoop; j++) {
                LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(goCurVreg0,
                                                             loUb + 2U * i * colStride + j * vlElemNum);
                LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(
                    goCurVreg1, loUb + (2U * i + 1U) * colStride + j * vlElemNum);
                Div(divVreg0, goCurVreg0, glVreg0, pregFull);
                Div(divVreg1, goCurVreg1, glVreg1, pregFull);
                if constexpr (FULL_QUANT_FP8) {
                    Muls(divVreg0, divVreg0, MAX_VALUE_RECIPROCAL_FP8, pregFull);
                    Muls(divVreg1, divVreg1, MAX_VALUE_RECIPROCAL_FP8, pregFull);
                }
                StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(
                    goUb + 2U * i * colStride + j * vlElemNum, divVreg0, pregFull);
                StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(
                    goUb + (2U * i + 1U) * colStride + j * vlElemNum, divVreg1, pregFull);
            }
            for (uint16_t j = 0; j < hasNTail; j++) {
                LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(
                    goCurVreg0, loUb + 2U * i * colStride + (nLoop + j) * vlElemNum);
                LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(
                    goCurVreg1, loUb + (2U * i + 1U) * colStride + (nLoop + j) * vlElemNum);
                Div(divVreg0, goCurVreg0, glVreg0, pregTail);
                Div(divVreg1, goCurVreg1, glVreg1, pregTail);
                if constexpr (FULL_QUANT_FP8) {
                    Muls(divVreg0, divVreg0, MAX_VALUE_RECIPROCAL_FP8, pregTail);
                    Muls(divVreg1, divVreg1, MAX_VALUE_RECIPROCAL_FP8, pregTail);
                }
                StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(
                    goUb + 2U * i * colStride + (nLoop + j) * vlElemNum, divVreg0, pregTail);
                StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(
                    goUb + (2U * i + 1U) * colStride + (nLoop + j) * vlElemNum, divVreg1, pregTail);
            }
        }

        for (uint16_t i = 0; i < hasMTail; i++) {
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(glVreg0, glUb + 2U * mLoop + i);
            for (uint16_t j = 0; j < nLoop; j++) {
                LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(
                    goCurVreg0, loUb + (2U * mLoop + i) * colStride + j * vlElemNum);
                Div(divVreg0, goCurVreg0, glVreg0, pregFull);
                if constexpr (FULL_QUANT_FP8) {
                    Muls(divVreg0, divVreg0, MAX_VALUE_RECIPROCAL_FP8, pregFull);
                }
                StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(
                    goUb + (2U * mLoop + i) * colStride + j * vlElemNum, divVreg0, pregFull);
            }
            for (uint16_t j = 0; j < hasNTail; j++) {
                LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(
                    goCurVreg0, loUb + (2U * mLoop + i) * colStride + (nLoop + j) * vlElemNum);
                Div(divVreg0, goCurVreg0, glVreg0, pregTail);
                if constexpr (FULL_QUANT_FP8) {
                    Muls(divVreg0, divVreg0, MAX_VALUE_RECIPROCAL_FP8, pregTail);
                }
                StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(
                    goUb + (2U * mLoop + i) * colStride + (nLoop + j) * vlElemNum, divVreg0, pregTail);
            }
        }
    }
    // When qS can be copied from UB to GM continuously,
    // qS would be stored from reg to UB continuously.
    __simd_vf__ inline void LogSumExpFuncQSAxisContinuous(__ubuf__ float *gmUb, __ubuf__ float *glUb,
                                                          __ubuf__ float *lseUb, uint32_t row)
    {
        // This vf works only when the rowNum in each AIV does not exceed 64
        using namespace AscendC::Reg;
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
    __simd_vf__ inline void LogSumExpFuncQSAxisIncontinuous(__ubuf__ float *gmUb, __ubuf__ float *glUb,
                                                            __ubuf__ float *lseUb, uint32_t row)
    {
        using namespace AscendC::Reg;
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

        for (uint16_t i = 0; i < row; i += 2) {
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
    __aicore__ inline void operator()(TensorO &gOTensor, TensorLse &gLseTensor, GemmCoord actualOriShape,
                                      uint32_t curTileMod, uint32_t gatheredKvSTileIdx, bool isFirstKvSTile,
                                      bool isLastKvSTile, Arch::CrossCoreFlag mm2ToReFlag)
    {
        uint32_t rowNumOri = actualOriShape[0];
        uint32_t colNumOri = actualOriShape[1];
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        uint32_t rowNumOriAligned2 = RoundUp(rowNumOri, 2);
        uint32_t colNumOriAligned16 = RoundUp(colNumOri, 16);

        uint32_t rowNumSplit = rowNumOriAligned2 / subBlockNum;
        rowNumSplit = (rowNumOri < rowNumSplit) ? rowNumOri : rowNumSplit;
        uint32_t rowNumCurSubCore = (subBlockIdx == 0) ? rowNumSplit : (rowNumOri - rowNumSplit);
        uint32_t rowOffsetCurSubCore = rowNumSplit * subBlockIdx;
        uint32_t colNumCurSubCore = colNumOri;
        uint32_t colStrideCurSubCore = colNumOriAligned16;

        auto gOTensorTlaTile = GetTile(gOTensor, tla::MakeCoord(rowOffsetCurSubCore, 0),
                                       tla::MakeShape(rowNumCurSubCore, colNumCurSubCore));
        auto gLseTensorTlaTile =
            GetTile(gLseTensor, tla::MakeCoord(rowOffsetCurSubCore, 0), tla::MakeShape(rowNumCurSubCore, 1));
        uint32_t ubOTmpBufId = gatheredKvSTileIdx % loUbBufNum;
        AscendC::GlobalTensor<float> dummyPartialO;
        AscendC::GlobalTensor<float> dummyPartialLse;

        if (rowNumCurSubCore > 0) {
            SubCoreCompute<false>(gOTensorTlaTile, gLseTensorTlaTile, curTileMod, ubOTmpBufId, isFirstKvSTile,
                                  isLastKvSTile, colStrideCurSubCore, mm2ToReFlag, dummyPartialO, dummyPartialLse, 0U,
                                  rowOffsetCurSubCore, rowNumOri, 0U);
        } else {
            Arch::CrossCoreWaitFlag<4, PIPE_V>(mm2ToReFlag);
            Arch::CrossCoreSetFlag<4, PIPE_V>(mm2ToReFlag);
        }
    }

    template <class TensorO, class TensorLse>
    __aicore__ inline void ProcessPartial(TensorO &gOTensor, TensorLse &gLseTensor, GemmCoord actualOriShape,
                                          uint32_t curTileMod, uint32_t gatheredKvSTileIdx, bool isFirstKvSTile,
                                          bool isLastKvSTile, Arch::CrossCoreFlag mm2ToReFlag,
                                          AscendC::GlobalTensor<float> &gPartialO,
                                          AscendC::GlobalTensor<float> &gPartialLse, uint32_t fdTaskId,
                                          uint32_t fdLseSubStride)
    {
        const uint32_t rowNumOri = actualOriShape[0];
        const uint32_t colNumOri = actualOriShape[1];
        const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t rowNumSplit = RoundUp(rowNumOri, 2) / AscendC::GetSubBlockNum();
        rowNumSplit = (rowNumOri < rowNumSplit) ? rowNumOri : rowNumSplit;
        const uint32_t rowNumCurSubCore = subBlockIdx == 0 ? rowNumSplit : rowNumOri - rowNumSplit;
        const uint32_t rowOffsetCurSubCore = rowNumSplit * subBlockIdx;
        const uint32_t colStrideCurSubCore = RoundUp(colNumOri, 16);
        auto gOTensorTlaTile =
            GetTile(gOTensor, tla::MakeCoord(rowOffsetCurSubCore, 0), tla::MakeShape(rowNumCurSubCore, colNumOri));
        auto gLseTensorTlaTile =
            GetTile(gLseTensor, tla::MakeCoord(rowOffsetCurSubCore, 0), tla::MakeShape(rowNumCurSubCore, 1));
        const uint32_t ubOTmpBufId = gatheredKvSTileIdx % loUbBufNum;
        if (rowNumCurSubCore > 0) {
            SubCoreCompute<true>(gOTensorTlaTile, gLseTensorTlaTile, curTileMod, ubOTmpBufId, isFirstKvSTile,
                                 isLastKvSTile, colStrideCurSubCore, mm2ToReFlag, gPartialO, gPartialLse, fdTaskId,
                                 rowOffsetCurSubCore, rowNumOri, fdLseSubStride);
        } else {
            Arch::CrossCoreWaitFlag<4, PIPE_V>(mm2ToReFlag);
            Arch::CrossCoreSetFlag<4, PIPE_V>(mm2ToReFlag);
        }
    }

    __aicore__ inline void WriteNeutralPartial(AscendC::GlobalTensor<float> &gPartialO,
                                               AscendC::GlobalTensor<float> &gPartialLse, uint32_t fdTaskId,
                                               uint32_t groupSize, uint32_t colNum, uint32_t fdLseSubStride)
    {
        const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t rowNumSplit = RoundUp(groupSize, 2) / AscendC::GetSubBlockNum();
        rowNumSplit = groupSize < rowNumSplit ? groupSize : rowNumSplit;
        const uint32_t rowNum = subBlockIdx == 0 ? rowNumSplit : groupSize - rowNumSplit;
        const uint32_t rowOffset = rowNumSplit * subBlockIdx;
        if (rowNum == 0) {
            return;
        }
        const uint32_t colStride = RoundUp(colNum, 16);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::Duplicate(goUbTensor32, 0.0f, rowNum * colStride);
        AscendC::Duplicate(lseUbTensor32, -3.402823466e+38F, RoundUp(rowNum, 8));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        const uint64_t oOffset =
            static_cast<uint64_t>(fdTaskId) * groupSize * colStride + static_cast<uint64_t>(rowOffset) * colStride;
        AscendC::DataCopy(gPartialO[oOffset], goUbTensor32, rowNum * colStride);
        const uint64_t lseOffset = static_cast<uint64_t>(fdTaskId) * 2 * fdLseSubStride + subBlockIdx * fdLseSubStride;
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        AscendC::DataCopyPad(gPartialLse[lseOffset], lseUbTensor32,
                             AscendC::DataCopyExtParams(1, rowNum * sizeof(float), 0, 0, 0));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
    }

    template <class TensorO, class TensorLse>
    __aicore__ inline void WriteEmptyOutput(TensorO &gOTensor, TensorLse &gLseTensor, GemmCoord actualOriShape)
    {
        const uint32_t rowNumOri = actualOriShape[0];
        const uint32_t colNumOri = actualOriShape[1];
        const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t rowNumSplit = RoundUp(rowNumOri, 2) / AscendC::GetSubBlockNum();
        rowNumSplit = rowNumOri < rowNumSplit ? rowNumOri : rowNumSplit;
        const uint32_t rowNum = subBlockIdx == 0 ? rowNumSplit : rowNumOri - rowNumSplit;
        const uint32_t rowOffset = rowNumSplit * subBlockIdx;
        if (rowNum == 0) {
            return;
        }
        // UB source rows consumed by MTE3 must keep a 32-byte aligned start address.
        // ElementO is 2 bytes in the A5 non-quant path, hence 16 elements per row.
        const uint32_t colStride = RoundUp(colNumOri, 16);
        auto gOTile = GetTile(gOTensor, tla::MakeCoord(rowOffset, 0), tla::MakeShape(rowNum, colNumOri));
        auto gLseTile = GetTile(gLseTensor, tla::MakeCoord(rowOffset, 0), tla::MakeShape(rowNum, 1));
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::Duplicate(goUbTensor16, static_cast<ElementO>(0), rowNum * colStride);
        if constexpr (DispatchPolicy::LSE_MODE == LseMode::OUT_ONLY) {
            AscendC::Duplicate(lseUbTensor32, -3.402823466e+38F, rowNum * 8U);
        }
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (DispatchPolicy::LSE_MODE == LseMode::OUT_ONLY) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            auto ubLseLayout = tla::MakeLayout(tla::MakeShape(rowNum, 8U), tla::MakeStride(8U, tla::Int<1>{}));
            auto ubLseTensor = tla::MakeTensor(lseUbTensor32, ubLseLayout, Arch::PositionUB{});
            CopyUbToGmLse(gLseTile, ubLseTensor);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        auto ubOLayout = tla::MakeLayout(tla::MakeShape(rowNum, colNumOri), tla::MakeStride(colStride, tla::Int<1>{}));
        auto ubOTensor = tla::MakeTensor(goUbTensor16, ubOLayout, Arch::PositionUB{});
        copyUbToGmO(gOTile, ubOTensor);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
    }

private:
    uint32_t loUbBufNum;
    uint32_t loUbRowNum;
    bool loGmTransit;
    uint64_t loGmTileElems;
    AscendC::GlobalTensor<ElementOTmp> gLo;
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
} // namespace NpuArch::Epilogue::Block
#endif
