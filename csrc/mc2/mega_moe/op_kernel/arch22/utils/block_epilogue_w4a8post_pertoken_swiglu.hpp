/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_W4A8POST_PER_TOKEN_SWIGLU_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_W4A8POST_PER_TOKEN_SWIGLU_HPP

#include <limits>

#include "../template_linear_algebra_v2/catlass.hpp"
#include "../template_linear_algebra_v2/arch/resource.hpp"
#include "../template_linear_algebra_v2/epilogue/dispatch_policy.hpp"
#include "../template_linear_algebra_v2/gemm_coord.hpp"
#include "../template_linear_algebra_v2/matrix_coord.hpp"
#include "../template_linear_algebra_v2/layout/layout.hpp"
#include "../template_linear_algebra_v2/detail/callback.hpp"
#include "gated_activation.hpp"
#include "get_tensor_addr.hpp"

namespace Catlass::Epilogue::Block {

template <uint32_t UB_STAGES_, class CType_, class LayoutPerTokenScale_, class DType_, class TileElemWiseMuls_,
          class TileCopy_>
class BlockEpilogue<EpilogueAtlasA2W4A8PostPerTokenDequantSwigluQuant<UB_STAGES_>, CType_,
                    Gemm::GemmType<float, LayoutPerTokenScale_>, DType_, TileElemWiseMuls_, TileCopy_> {
public:
    using DispatchPolicy = EpilogueAtlasA2W4A8PostPerTokenDequantSwigluQuant<UB_STAGES_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using Activation = typename DispatchPolicy::Activation;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    static constexpr uint32_t TILE_LENGTH = DispatchPolicy::TILE_LENGTH;
    static constexpr uint32_t SCALE_BUFFER_COUNT = 2;
    static constexpr uint32_t SCALE_BATCH_COUNT = 256;
    static constexpr uint32_t SCALE_BUFFER_BYTES =
        (SCALE_BATCH_COUNT + BYTE_PER_BLK / sizeof(float)) * sizeof(float);
    static constexpr uint32_t ROW_MAX_BYTES = BYTE_PER_BLK;

    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementPerTokenScale = float;
    using LayoutPerTokenScale = LayoutPerTokenScale_;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;

    static constexpr size_t BUFFER_SIZE =
        UB_STAGES * (4 * TILE_LENGTH * sizeof(ElementC) + 2 * TILE_LENGTH * sizeof(float) + TILE_LENGTH) +
        3 * TILE_LENGTH * sizeof(float) + TILE_LENGTH * sizeof(ElementD) + 128 * sizeof(int16_t) +
        SCALE_BUFFER_COUNT * SCALE_BUFFER_BYTES + ROW_MAX_BYTES;
    static_assert(TILE_LENGTH % 256 == 0, "The W4A8 activation tile must be aligned to 256 elements");
    static_assert(UB_STAGES >= 2, "The pipelined W4A8 activation epilogue requires double buffering");
    static_assert(SCALE_BATCH_COUNT % (BYTE_PER_BLK / sizeof(ElementPerTokenScale)) == 0,
                  "The scale batch must be block aligned");
    static_assert(BUFFER_SIZE + Activation::MIN_SHARED_TMP_BYTES <= ArchTag::UB_SIZE,
                  "The tiled W4A8 activation buffers and workspace exceed UB capacity");

    static_assert((std::is_same_v<ElementC, half> ||
                   std::is_same_v<ElementC, bfloat16_t>) &&
                      std::is_same_v<ElementD, int8_t>,
                  "The element type template parameters of BlockEpilogue are wrong");
    static_assert(std::is_same_v<LayoutC, layout::RowMajor> &&
                      std::is_same_v<LayoutPerTokenScale, layout::VectorLayout> &&
                      std::is_same_v<LayoutD, layout::RowMajor>,
                  "The layout template parameters of BlockEpilogue are wrong");

    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyUbToGmDequantScale =
        Epilogue::Tile::CopyUb2Gm<ArchTag, Gemm::GemmType<ElementPerTokenScale, LayoutPerTokenScale>>;

    struct Params {
        __gm__ ElementPerTokenScale *ptrPerTokenScale{nullptr};
        LayoutPerTokenScale layoutPerTokenScale{};
        __gm__ ElementD *ptrD{nullptr};
        LayoutD layoutD{};
        int32_t expertPerRank{0};

        CATLASS_DEVICE Params() {};

        CATLASS_DEVICE
        Params(__gm__ ElementPerTokenScale *ptrPerTokenScale_, LayoutPerTokenScale const &layoutPerTokenScale_,
               __gm__ ElementD *ptrD_, LayoutD const &layoutD_, int32_t expertPerRank_)
            : ptrPerTokenScale(ptrPerTokenScale_),
              layoutPerTokenScale(layoutPerTokenScale_),
              ptrD(ptrD_),
              layoutD(layoutD_),
              expertPerRank(expertPerRank_)
        {
        }
    };

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> const &resource, int32_t n, Params const &params = Params{})
        : params(params)
    {
        (void)n;
        ubOffset = 0;
        int32_t eventVMTE2 = 0;
        int32_t eventMTE2V = 0;
        int32_t eventMTE3V = 0;
        int32_t eventVMTE3 = 0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            // Two MSD rows, each containing gate and up tiles.
            ubCList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
            ubOffset += 4 * TILE_LENGTH * sizeof(ElementC);
            ubWeightAuxList[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += 2 * TILE_LENGTH * sizeof(float);
            xHighI4TensorList[i] = resource.ubBuf.template GetBufferByByte<int4b_t>(ubOffset);
            ubOffset += TILE_LENGTH / 2;
            xLowI4TensorList[i] = resource.ubBuf.template GetBufferByByte<int4b_t>(ubOffset);
            ubOffset += TILE_LENGTH / 2;

            eventUbCVMTE2List[i] = eventVMTE2++;
            eventUbCMTE2VList[i] = eventMTE2V++;
            eventUbWAVMTE2List[i] = eventVMTE2++;
            eventUbWAMTE2VList[i] = eventMTE2V++;
            eventxHighMTE3VList[i] = eventMTE3V++;
            eventxHighVMTE3List[i] = eventVMTE3++;
            eventxLowMTE3VList[i] = eventMTE3V++;
            eventxLowVMTE3List[i] = eventVMTE3++;
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbWAVMTE2List[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventxHighMTE3VList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventxLowMTE3VList[i]);
        }

        ubGateFp32 = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += TILE_LENGTH * sizeof(float);
        ubUpFp32 = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += TILE_LENGTH * sizeof(float);
        ubActivation = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += TILE_LENGTH * sizeof(float);
        ubD = resource.ubBuf.template GetBufferByByte<ElementD>(ubOffset);
        ubOffset += TILE_LENGTH * sizeof(ElementD);
        xLowI16Tensor = resource.ubBuf.template GetBufferByByte<int16_t>(ubOffset);
        ubOffset += 128 * sizeof(int16_t);
        for (uint32_t i = 0; i < SCALE_BUFFER_COUNT; ++i) {
            ubPerTokenScaleOutputList[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += SCALE_BUFFER_BYTES;
        }
        ubRowAbsMax = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += ROW_MAX_BYTES;
        sharedTmpBuffer = resource.ubBuf.template GetBufferByByte<uint8_t>(ubOffset);
    }

    CATLASS_DEVICE
    void Finalize()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbWAVMTE2List[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventxHighMTE3VList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventxLowMTE3VList[i]);
        }
    }

    CATLASS_DEVICE ~BlockEpilogue()
    {
    }

    CATLASS_DEVICE
    void UpdateParams(Params const &params_)
    {
        params = params_;
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementC> const &gmC, MatrixCoord const &shapeC,
                    AscendC::GlobalTensor<ElementPerTokenScale> const &gmPerTokenScale1, __gm__ float *gmWeightAux,
                    AscendC::GlobalTensor<ElementD> const &gmD, AscendC::GlobalTensor<int32_t> const &cumsumMM,
                    uint32_t MOffset, AscendC::GlobalTensor<ElementPerTokenScale> const &gmPerTokenScale2,
                    uint32_t expertPerRank, uint32_t EP, int32_t rank, int32_t listLen,
                    Arch::Resource<ArchTag> const &resource, uint32_t epilogueCoreNum = 40,
                    float activationClamp = 0.0f, uint32_t activationCode = 0,
                    float activationParams1 = SwigluOaiActivation::DEFAULT_ALPHA,
                    float activationParams2 = SituActivation::DEFAULT_BETA, uint32_t gmmOutPreRowStride = 1,
                    Callback &&callback = Callback{})
    {
        (void)rank;
        (void)resource;
        activationCode_ = activationCode;
        activationParams1_ = activationParams1;
        activationParams2_ = activationParams2;
        callback();
        uint32_t blockM = shapeC.row();
        uint32_t branchLength = shapeC.column() / 2;
        uint32_t loopStartIdx = 0;
        uint32_t tasksForIdx = 0;
        if (!GetTaskRange(blockM, epilogueCoreNum, loopStartIdx, tasksForIdx)) {
            return;
        }
        uint32_t scaleBatchStart = loopStartIdx;
        uint32_t scaleBatchCount = 0;
        uint32_t scaleBatchIndex = 0;
        Duplicate(xLowI16Tensor, static_cast<int16_t>(0x0F0F), 128);
        PipeBarrier<PIPE_V>();

        constexpr float DEFAULT_MUL_SCALE = 16.0f;
        for (uint32_t loopIdx = loopStartIdx; loopIdx < loopStartIdx + tasksForIdx; ++loopIdx) {
            auto gmTileC = gmC[loopIdx * gmmOutPreRowStride * 2];
            auto gmTileD = gmD[loopIdx * branchLength];
            uint32_t groupIdx = GetGroupIndex(cumsumMM, MOffset + loopIdx, expertPerRank, EP);
            int32_t arrayGroupIdx = listLen == 1 ? 0 : static_cast<int32_t>(groupIdx);
            AscendC::GlobalTensor<float> weightAux;
            weightAux.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
                GetTensorAddr<float>(arrayGroupIdx, reinterpret_cast<GM_ADDR>(gmWeightAux))));
            uint32_t weightAuxOffset = listLen == 1 ? groupIdx * shapeC.column() : 0;

            dcci(gmPerTokenScale1.GetPhyAddr(loopIdx), 0);
            ElementPerTokenScale perTokenScale = gmPerTokenScale1(loopIdx);
            uint32_t tileCount = TileCount(branchLength);
            uint32_t currentStage = 0;
            LoadTile(gmTileC, weightAux[weightAuxOffset], branchLength, 0, TileLength(branchLength, 0),
                     gmmOutPreRowStride, currentStage);
            ResetRowAbsMax();
            for (uint32_t tileIdx = 0; tileIdx < tileCount; ++tileIdx) {
                uint32_t tileOffset = tileIdx * TILE_LENGTH;
                uint32_t tileLength = TileLength(branchLength, tileOffset);
                uint32_t nextStage = NextStage(currentStage);
                uint32_t nextOffset = tileOffset + TILE_LENGTH;
                if (nextOffset < branchLength) {
                    LoadTile(gmTileC, weightAux[weightAuxOffset], branchLength, nextOffset,
                             TileLength(branchLength, nextOffset), gmmOutPreRowStride, nextStage);
                }
                ComputeLoadedTile(currentStage, tileLength, perTokenScale, activationClamp, DEFAULT_MUL_SCALE);
                AccumulateRowAbsMax(tileLength);
                currentStage = nextStage;
            }

            ElementPerTokenScale rowAbsMax = ReadRowAbsMax();
            uint32_t scaleStage = scaleBatchIndex % SCALE_BUFFER_COUNT;
            // Reuse waits only after both buffers have been launched, allowing scale MTE3 to overlap the next pass.
            if (scaleBatchCount == 0 && scaleBatchIndex >= SCALE_BUFFER_COUNT) {
                WaitScaleBuffer(scaleStage);
            }
            ubPerTokenScaleOutputList[scaleStage].SetValue(scaleBatchCount, rowAbsMax / 127.0f);
            ElementPerTokenScale quantMultiplier = rowAbsMax > 0.0f ? 127.0f / rowAbsMax : 0.0f;
            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);

            currentStage = 0;
            if (tileCount == 1) {
                // Single-tile fast path: ubActivationList[0] still holds the first-pass activation
                // output, so skip the second LoadTile + reactivation and quantize it directly.
                QuantizeAndCopyTile(0, gmTileD, branchLength, 0, TileLength(branchLength, 0), quantMultiplier);
            } else {
                LoadTile(gmTileC, weightAux[weightAuxOffset], branchLength, 0, TileLength(branchLength, 0),
                         gmmOutPreRowStride, currentStage);
                for (uint32_t tileIdx = 0; tileIdx < tileCount; ++tileIdx) {
                    uint32_t tileOffset = tileIdx * TILE_LENGTH;
                    uint32_t tileLength = TileLength(branchLength, tileOffset);
                    uint32_t nextStage = NextStage(currentStage);
                    uint32_t nextOffset = tileOffset + TILE_LENGTH;
                    if (nextOffset < branchLength) {
                        LoadTile(gmTileC, weightAux[weightAuxOffset], branchLength, nextOffset,
                                 TileLength(branchLength, nextOffset), gmmOutPreRowStride, nextStage);
                    }
                    ComputeLoadedTile(currentStage, tileLength, perTokenScale, activationClamp, DEFAULT_MUL_SCALE);
                    QuantizeAndCopyTile(currentStage, gmTileD, branchLength, tileOffset, tileLength, quantMultiplier);
                    currentStage = nextStage;
                }
            }
            ++scaleBatchCount;
            if (scaleBatchCount == SCALE_BATCH_COUNT) {
                FlushScaleBatch(gmPerTokenScale2, scaleBatchStart, scaleBatchCount, scaleStage);
                scaleBatchStart += scaleBatchCount;
                scaleBatchCount = 0;
                ++scaleBatchIndex;
            }
        }
        if (scaleBatchCount > 0) {
            FlushScaleBatch(gmPerTokenScale2, scaleBatchStart, scaleBatchCount,
                            scaleBatchIndex % SCALE_BUFFER_COUNT);
            ++scaleBatchIndex;
        }
        DrainScaleBuffers(scaleBatchIndex);
    }

private:
    CATLASS_DEVICE
    bool GetTaskRange(uint32_t blockM, uint32_t epilogueCoreNum, uint32_t &loopStartIdx, uint32_t &tasksForIdx)
    {
        uint32_t subblockIdx = get_block_idx() + get_subblockid() * get_block_num();
        uint32_t moveDataCoreNum = get_block_num() * 2 - epilogueCoreNum;
        if (subblockIdx < moveDataCoreNum) {
            return false;
        }
        uint32_t epilogueCoreIdx = subblockIdx - moveDataCoreNum;
        uint32_t perCoreData = blockM / epilogueCoreNum;
        uint32_t remainderData = blockM % epilogueCoreNum;
        tasksForIdx = epilogueCoreIdx < remainderData ? perCoreData + 1 : perCoreData;
        loopStartIdx =
            epilogueCoreIdx * perCoreData + (epilogueCoreIdx < remainderData ? epilogueCoreIdx : remainderData);
        return true;
    }

    CATLASS_DEVICE
    void FlushScaleBatch(AscendC::GlobalTensor<ElementPerTokenScale> const &gmPerTokenScale, uint32_t gmOffset,
                         uint32_t scaleCount, uint32_t scaleStage)
    {
        if (scaleCount == 0) {
            return;
        }
        LayoutPerTokenScale scaleLayout{scaleCount};
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(scaleStage);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(scaleStage);
        copyUbToGmDequantScale(gmPerTokenScale[gmOffset], ubPerTokenScaleOutputList[scaleStage], scaleLayout,
                               scaleLayout);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(scaleStage);
    }

    CATLASS_DEVICE
    void WaitScaleBuffer(uint32_t scaleStage)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(scaleStage);
    }

    CATLASS_DEVICE
    void DrainScaleBuffers(uint32_t scaleBatchCount)
    {
        uint32_t pendingCount = scaleBatchCount < SCALE_BUFFER_COUNT ? scaleBatchCount : SCALE_BUFFER_COUNT;
        uint32_t firstPendingBatch = scaleBatchCount - pendingCount;
        for (uint32_t i = 0; i < pendingCount; ++i) {
            WaitScaleBuffer((firstPendingBatch + i) % SCALE_BUFFER_COUNT);
        }
    }

    CATLASS_DEVICE
    static uint32_t TileCount(uint32_t branchLength)
    {
        return (branchLength + TILE_LENGTH - 1) / TILE_LENGTH;
    }

    CATLASS_DEVICE
    static uint32_t TileLength(uint32_t branchLength, uint32_t tileOffset)
    {
        uint32_t remaining = branchLength - tileOffset;
        return remaining < TILE_LENGTH ? remaining : TILE_LENGTH;
    }

    CATLASS_DEVICE
    static uint32_t NextStage(uint32_t stageId)
    {
        return stageId + 1 < UB_STAGES ? stageId + 1 : 0;
    }

    CATLASS_DEVICE
    static uint32_t GetGroupIndex(AscendC::GlobalTensor<int32_t> const &cumsumMM, uint32_t globalOffset,
                                  uint32_t expertPerRank, uint32_t EP)
    {
        uint32_t groupIdx = 0;
        uint32_t sum = cumsumMM((EP - 1) * expertPerRank);
        while (globalOffset >= sum && groupIdx < expertPerRank) {
            ++groupIdx;
            sum += cumsumMM((EP - 1) * expertPerRank + groupIdx);
        }
        return groupIdx;
    }

    CATLASS_DEVICE
    void LoadTile(AscendC::GlobalTensor<ElementC> const &gmTileC, AscendC::GlobalTensor<float> const &weightAux,
                  uint32_t branchLength, uint32_t tileOffset, uint32_t tileLength, uint32_t gmRowStride,
                  uint32_t stageId)
    {
        auto &ubC = ubCList[stageId];
        auto &ubWeightAux = ubWeightAuxList[stageId];

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[stageId]);
        LayoutC tileLayout{1, tileLength};
        LayoutC gmTileLayout{1, tileLength, gmRowStride};
        copyGmToUbC(ubC, gmTileC[tileOffset], tileLayout, gmTileLayout);
        copyGmToUbC(ubC[TILE_LENGTH], gmTileC[branchLength + tileOffset], tileLayout, gmTileLayout);
        copyGmToUbC(ubC[2 * TILE_LENGTH], gmTileC[gmRowStride + tileOffset], tileLayout, gmTileLayout);
        copyGmToUbC(ubC[3 * TILE_LENGTH], gmTileC[gmRowStride + branchLength + tileOffset], tileLayout, gmTileLayout);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[stageId]);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbWAVMTE2List[stageId]);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(tileLength * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(ubWeightAux, weightAux[tileOffset], copyParams, padParams);
        DataCopyPad(ubWeightAux[TILE_LENGTH], weightAux[branchLength + tileOffset], copyParams, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbWAMTE2VList[stageId]);
    }

    CATLASS_DEVICE
    void ComputeLoadedTile(uint32_t stageId, uint32_t tileLength, ElementPerTokenScale perTokenScale,
                           float activationClamp, float reconstructScale)
    {
        auto &ubC = ubCList[stageId];
        auto &ubWeightAux = ubWeightAuxList[stageId];

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[stageId]);
        AscendC::Cast(ubGateFp32, ubC, AscendC::RoundMode::CAST_NONE, tileLength);
        AscendC::Cast(ubActivation, ubC[2 * TILE_LENGTH], AscendC::RoundMode::CAST_NONE, tileLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(ubGateFp32, ubGateFp32, reconstructScale, tileLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(ubGateFp32, ubGateFp32, ubActivation, tileLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(ubUpFp32, ubC[TILE_LENGTH], AscendC::RoundMode::CAST_NONE, tileLength);
        AscendC::Cast(ubActivation, ubC[3 * TILE_LENGTH], AscendC::RoundMode::CAST_NONE, tileLength);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[stageId]);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(ubUpFp32, ubUpFp32, reconstructScale, tileLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(ubUpFp32, ubUpFp32, ubActivation, tileLength);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbWAMTE2VList[stageId]);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(ubGateFp32, ubGateFp32, ubWeightAux, tileLength);
        AscendC::Add(ubUpFp32, ubUpFp32, ubWeightAux[TILE_LENGTH], tileLength);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbWAVMTE2List[stageId]);

        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(ubGateFp32, ubGateFp32, perTokenScale, tileLength);
        AscendC::Muls(ubUpFp32, ubUpFp32, perTokenScale, tileLength);
        AscendC::PipeBarrier<PIPE_V>();
        Activation::Compute(ubActivation, ubGateFp32, ubUpFp32, sharedTmpBuffer, tileLength, activationClamp,
                            activationCode_, activationParams1_, activationParams2_);
    }

    CATLASS_DEVICE
    void QuantizeAndCopyTile(uint32_t stageId, AscendC::GlobalTensor<ElementD> const &gmTileD, uint32_t branchLength,
                             uint32_t tileOffset, uint32_t tileLength, ElementPerTokenScale quantMultiplier)
    {
        auto ubQuantS32 = ubGateFp32.template ReinterpretCast<int32_t>();
        auto ubQuantF16 = ubGateFp32.template ReinterpretCast<half>();
        AscendC::Muls(ubGateFp32, ubActivation, quantMultiplier, tileLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(ubQuantS32, ubGateFp32, AscendC::RoundMode::CAST_RINT, tileLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetDeqScale(static_cast<half>(1.0));
        AscendC::Cast(ubQuantF16, ubQuantS32, AscendC::RoundMode::CAST_RINT, tileLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(ubD, ubQuantF16, AscendC::RoundMode::CAST_RINT, tileLength);
        AscendC::PipeBarrier<PIPE_V>();

        auto &xHighI4Tensor = xHighI4TensorList[stageId];
        auto &xLowI4Tensor = xLowI4TensorList[stageId];
        Cast(ubQuantF16, ubD, AscendC::RoundMode::CAST_NONE, tileLength);
        PipeBarrier<PIPE_V>();
        Muls(ubQuantF16, ubQuantF16, static_cast<half>(0.0625f), tileLength);
        PipeBarrier<PIPE_V>();
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventxHighMTE3VList[stageId]);
        Cast(xHighI4Tensor, ubQuantF16, AscendC::RoundMode::CAST_FLOOR, tileLength);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventxHighVMTE3List[stageId]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventxHighVMTE3List[stageId]);
        DataCopy(gmTileD[tileOffset / 2], xHighI4Tensor.template ReinterpretCast<int8_t>(), tileLength / 2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventxHighMTE3VList[stageId]);

        auto xLowHalfTensor = ubActivation.template ReinterpretCast<half>();
        auto xLowHalfTensor2 = xLowHalfTensor[TILE_LENGTH];
        uint32_t lenVk = (tileLength / 2) / 128;
        uint32_t lastLenVk = (tileLength % 256) / 2;
        And(xLowHalfTensor.template ReinterpretCast<int16_t>(), ubD.template ReinterpretCast<int16_t>(), xLowI16Tensor,
            128, lenVk, {1, 1, 1, 8, 8, 0});
        if (lastLenVk > 0) {
            And(xLowHalfTensor[lenVk * 128].template ReinterpretCast<int16_t>(),
                ubD[lenVk * 256].template ReinterpretCast<int16_t>(), xLowI16Tensor, lastLenVk, 1, {1, 1, 1, 8, 8, 0});
        }
        PipeBarrier<PIPE_V>();
        Cast(xLowHalfTensor2, xLowHalfTensor.template ReinterpretCast<int8_t>(), AscendC::RoundMode::CAST_NONE,
             tileLength);
        PipeBarrier<PIPE_V>();
        Adds(ubQuantF16, xLowHalfTensor2, static_cast<half>(-8), tileLength);
        PipeBarrier<PIPE_V>();
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventxLowMTE3VList[stageId]);
        Cast(xLowI4Tensor, ubQuantF16, AscendC::RoundMode::CAST_NONE, tileLength);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventxLowVMTE3List[stageId]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventxLowVMTE3List[stageId]);
        DataCopy(gmTileD[branchLength / 2 + tileOffset / 2], xLowI4Tensor.template ReinterpretCast<int8_t>(),
                 tileLength / 2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventxLowMTE3VList[stageId]);
    }

    CATLASS_DEVICE
    void ResetRowAbsMax()
    {
        AscendC::Duplicate(ubRowAbsMax, 0.0f, BYTE_PER_BLK / sizeof(float));
        AscendC::PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE
    void AccumulateRowAbsMax(uint32_t tileLength)
    {
        AscendC::Abs(ubGateFp32, ubActivation, tileLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceMax<float>(ubUpFp32, ubGateFp32, ubUpFp32, tileLength, false);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Max(ubRowAbsMax, ubRowAbsMax, ubUpFp32, 1);
        AscendC::PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE
    ElementPerTokenScale ReadRowAbsMax()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        return ubRowAbsMax.GetValue(0);
    }

    Params params;
    AscendC::LocalTensor<ElementC> ubCList[UB_STAGES];
    AscendC::LocalTensor<float> ubWeightAuxList[UB_STAGES];
    AscendC::LocalTensor<int4b_t> xHighI4TensorList[UB_STAGES];
    AscendC::LocalTensor<int4b_t> xLowI4TensorList[UB_STAGES];
    AscendC::LocalTensor<float> ubGateFp32;
    AscendC::LocalTensor<float> ubUpFp32;
    AscendC::LocalTensor<float> ubActivation;
    AscendC::LocalTensor<ElementD> ubD;
    AscendC::LocalTensor<int16_t> xLowI16Tensor;
    AscendC::LocalTensor<float> ubPerTokenScaleOutputList[SCALE_BUFFER_COUNT];
    AscendC::LocalTensor<float> ubRowAbsMax;
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer;

    int32_t eventUbCVMTE2List[UB_STAGES];
    int32_t eventUbCMTE2VList[UB_STAGES];
    int32_t eventUbWAVMTE2List[UB_STAGES];
    int32_t eventUbWAMTE2VList[UB_STAGES];
    int32_t eventxHighMTE3VList[UB_STAGES];
    int32_t eventxHighVMTE3List[UB_STAGES];
    int32_t eventxLowMTE3VList[UB_STAGES];
    int32_t eventxLowVMTE3List[UB_STAGES];
    uint32_t activationCode_{0};
    float activationParams1_{SwigluOaiActivation::DEFAULT_ALPHA};
    float activationParams2_{SituActivation::DEFAULT_BETA};
    size_t ubOffset{0};

    CopyGmToUbC copyGmToUbC;
    CopyUbToGmDequantScale copyUbToGmDequantScale;
};

} // namespace Catlass::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_EPILOGUE_W4A8POST_PER_TOKEN_SWIGLU_HPP
