/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_SWIGLU_A2_BF16_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_SWIGLU_A2_BF16_HPP

#include <limits>

#include "../template_linear_algebra_v2/catlass.hpp"
#include "../template_linear_algebra_v2/arch/resource.hpp"
#include "../template_linear_algebra_v2/epilogue/dispatch_policy.hpp"
#include "../template_linear_algebra_v2/gemm_coord.hpp"
#include "../template_linear_algebra_v2/matrix_coord.hpp"
#include "../template_linear_algebra_v2/layout/layout.hpp"
#include "../template_linear_algebra_v2/detail/callback.hpp"
#include "../utils/gated_activation.hpp"

namespace Catlass::Epilogue::Block {

// A2 BF16/FP16 gated activation. The input row is split into fixed gate/up
// tiles, making the UB footprint independent of N.
template <uint32_t UB_STAGES_, class CType_, class LayoutPerTokenScale_, class DType_, class TileElemWiseMuls_,
          class TileCopy_>
class BlockEpilogue<EpilogueAtlasA2PerTokenDequantSwigluQuantBF16<UB_STAGES_>, CType_,
                    Gemm::GemmType<float, LayoutPerTokenScale_>, DType_, TileElemWiseMuls_, TileCopy_> {
public:
    using DispatchPolicy = EpilogueAtlasA2PerTokenDequantSwigluQuantBF16<UB_STAGES_>;
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
        UB_STAGES *
            (2 * TILE_LENGTH * sizeof(ElementC) + TILE_LENGTH * sizeof(ElementD) + 3 * TILE_LENGTH * sizeof(float)) +
        SCALE_BUFFER_COUNT * SCALE_BUFFER_BYTES + ROW_MAX_BYTES;
    static_assert(UB_STAGES >= 2, "The pipelined activation epilogue requires double buffering");
    static_assert(SCALE_BATCH_COUNT % (BYTE_PER_BLK / sizeof(ElementPerTokenScale)) == 0,
                  "The scale batch must be block aligned");
    static_assert(BUFFER_SIZE + Activation::MIN_SHARED_TMP_BYTES <= ArchTag::UB_SIZE,
                  "The tiled activation buffers and workspace exceed UB capacity");

    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;
    using CopyUbToGmDequantScale =
        Epilogue::Tile::CopyUb2Gm<ArchTag, Gemm::GemmType<ElementPerTokenScale, LayoutPerTokenScale>>;

    struct Params {
        CATLASS_DEVICE Params() {};
    };

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> const &resource, int32_t n, Params const &params = Params{})
    {
        (void)n;
        (void)params;
        size_t ubOffset = 0;
        int32_t eventVMTE2 = 0;
        int32_t eventMTE2V = 0;
        int32_t eventMTE3V = 0;
        int32_t eventVMTE3 = 0;

        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubCList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
            ubOffset += 2 * TILE_LENGTH * sizeof(ElementC);
            ubDList[i] = resource.ubBuf.template GetBufferByByte<ElementD>(ubOffset);
            ubOffset += TILE_LENGTH * sizeof(ElementD);
            ubGateFp32List[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += TILE_LENGTH * sizeof(float);
            ubUpFp32List[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += TILE_LENGTH * sizeof(float);
            ubActivationList[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += TILE_LENGTH * sizeof(float);

            eventUbCVMTE2List[i] = eventVMTE2++;
            eventUbCMTE2VList[i] = eventMTE2V++;
            eventUbDMTE3VList[i] = eventMTE3V++;
            eventUbDVMTE3List[i] = eventVMTE3++;
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }

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
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
    }

    CATLASS_DEVICE ~BlockEpilogue()
    {
    }

    // Per-token dequant + activation + per-token dynamic quant. This overload
    // is kept for interface compatibility with the unified kernel.
    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementC> const &gmC, MatrixCoord const &shapeC,
                    AscendC::GlobalTensor<ElementPerTokenScale> const &gmPerTokenScale1,
                    AscendC::GlobalTensor<ElementD> const &gmD,
                    AscendC::GlobalTensor<ElementPerTokenScale> const &gmPerTokenScale2, uint32_t epilogueCoreNum = 40,
                    float activationClamp = 0.0f, uint32_t activationCode = 0,
                    float activationParams1 = SwigluOaiActivation::DEFAULT_ALPHA,
                    float activationParams2 = SituActivation::DEFAULT_BETA, uint32_t gmmOutPreRowStride = 1,
                    Callback &&callback = Callback{})
    {
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

        for (uint32_t loopIdx = loopStartIdx; loopIdx < loopStartIdx + tasksForIdx; ++loopIdx) {
            auto gmTileC = gmC[loopIdx * gmmOutPreRowStride];
            auto gmTileD = gmD[loopIdx * branchLength];
            ElementPerTokenScale perTokenScale = gmPerTokenScale1(loopIdx);
            uint32_t tileCount = TileCount(branchLength);
            uint32_t currentStage = 0;
            LoadTile(gmTileC, branchLength, 0, TileLength(branchLength, 0), gmmOutPreRowStride, currentStage);
            ResetRowAbsMax();
            for (uint32_t tileIdx = 0; tileIdx < tileCount; ++tileIdx) {
                uint32_t tileOffset = tileIdx * TILE_LENGTH;
                uint32_t tileLength = TileLength(branchLength, tileOffset);
                uint32_t nextStage = NextStage(currentStage);
                uint32_t nextOffset = tileOffset + TILE_LENGTH;
                if (nextOffset < branchLength) {
                    LoadTile(gmTileC, branchLength, nextOffset, TileLength(branchLength, nextOffset),
                             gmmOutPreRowStride, nextStage);
                }
                ComputeLoadedTile(currentStage, tileLength, perTokenScale, true, activationClamp);
                AccumulateRowAbsMax(currentStage, tileLength);
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
                QuantizeAndCopyTile(0, gmTileD[0], TileLength(branchLength, 0), quantMultiplier);
            } else {
                LoadTile(gmTileC, branchLength, 0, TileLength(branchLength, 0), gmmOutPreRowStride, currentStage);
                for (uint32_t tileIdx = 0; tileIdx < tileCount; ++tileIdx) {
                    uint32_t tileOffset = tileIdx * TILE_LENGTH;
                    uint32_t tileLength = TileLength(branchLength, tileOffset);
                    uint32_t nextStage = NextStage(currentStage);
                    uint32_t nextOffset = tileOffset + TILE_LENGTH;
                    if (nextOffset < branchLength) {
                        LoadTile(gmTileC, branchLength, nextOffset, TileLength(branchLength, nextOffset),
                                 gmmOutPreRowStride, nextStage);
                    }
                    ComputeLoadedTile(currentStage, tileLength, perTokenScale, true, activationClamp);
                    QuantizeAndCopyTile(currentStage, gmTileD[tileOffset], tileLength, quantMultiplier);
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

    // Non-quantized BF16/FP16 path.
    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementC> const &gmC, MatrixCoord const &shapeC,
                    AscendC::GlobalTensor<ElementD> const &gmD, uint32_t epilogueCoreNum = 40,
                    float activationClamp = 0.0f, uint32_t activationCode = 0,
                    float activationParams1 = SwigluOaiActivation::DEFAULT_ALPHA,
                    float activationParams2 = SituActivation::DEFAULT_BETA, uint32_t gmmOutPreRowStride = 1,
                    Callback &&callback = Callback{})
    {
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

        for (uint32_t loopIdx = loopStartIdx; loopIdx < loopStartIdx + tasksForIdx; ++loopIdx) {
            auto gmTileC = gmC[loopIdx * gmmOutPreRowStride];
            auto gmTileD = gmD[loopIdx * branchLength];
            uint32_t tileCount = TileCount(branchLength);
            uint32_t currentStage = 0;
            LoadTile(gmTileC, branchLength, 0, TileLength(branchLength, 0), gmmOutPreRowStride, currentStage);
            for (uint32_t tileIdx = 0; tileIdx < tileCount; ++tileIdx) {
                uint32_t tileOffset = tileIdx * TILE_LENGTH;
                uint32_t tileLength = TileLength(branchLength, tileOffset);
                uint32_t nextStage = NextStage(currentStage);
                uint32_t nextOffset = tileOffset + TILE_LENGTH;
                if (nextOffset < branchLength) {
                    LoadTile(gmTileC, branchLength, nextOffset, TileLength(branchLength, nextOffset),
                             gmmOutPreRowStride, nextStage);
                }
                ComputeLoadedTile(currentStage, tileLength, 1.0f, false, activationClamp);
                CopyDirectTile(currentStage, gmTileD[tileOffset], tileLength);
                currentStage = nextStage;
            }
        }
    }

private:
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
    void LoadTile(AscendC::GlobalTensor<ElementC> const &gmTileC, uint32_t branchLength, uint32_t tileOffset,
                  uint32_t tileLength, uint32_t gmRowStride, uint32_t stageId)
    {
        auto &ubC = ubCList[stageId];
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[stageId]);
        LayoutC tileLayout{1, tileLength};
        LayoutC gmTileLayout{1, tileLength, gmRowStride};
        copyGmToUbC(ubC, gmTileC[tileOffset], tileLayout, gmTileLayout);
        copyGmToUbC(ubC[TILE_LENGTH], gmTileC[branchLength + tileOffset], tileLayout, gmTileLayout);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[stageId]);
    }

    CATLASS_DEVICE
    void ComputeLoadedTile(uint32_t stageId, uint32_t tileLength, ElementPerTokenScale perTokenScale,
                           bool applyPerTokenScale, float activationClamp)
    {
        auto &ubC = ubCList[stageId];
        auto &ubGate = ubGateFp32List[stageId];
        auto &ubUp = ubUpFp32List[stageId];

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[stageId]);
        Activation::PrepareBranches(ubGate, ubUp, ubC, TILE_LENGTH, tileLength);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[stageId]);

        if (applyPerTokenScale) {
            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(ubGate, ubGate, perTokenScale, tileLength);
            AscendC::Muls(ubUp, ubUp, perTokenScale, tileLength);
            AscendC::PipeBarrier<PIPE_V>();
        }
        Activation::Compute(ubActivationList[stageId], ubGate, ubUp, sharedTmpBuffer, tileLength, activationClamp,
                            activationCode_, activationParams1_, activationParams2_);
    }

    CATLASS_DEVICE
    void ResetRowAbsMax()
    {
        AscendC::Duplicate(ubRowAbsMax, 0.0f, BYTE_PER_BLK / sizeof(float));
        AscendC::PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE
    void AccumulateRowAbsMax(uint32_t stageId, uint32_t tileLength)
    {
        auto &ubAbs = ubGateFp32List[stageId];
        auto &ubTileMax = ubUpFp32List[stageId];
        AscendC::Abs(ubAbs, ubActivationList[stageId], tileLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceMax<float>(ubTileMax, ubAbs, ubTileMax, tileLength, false);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Max(ubRowAbsMax, ubRowAbsMax, ubTileMax, 1);
        AscendC::PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE
    ElementPerTokenScale ReadRowAbsMax()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        return ubRowAbsMax.GetValue(0);
    }

    CATLASS_DEVICE
    void QuantizeAndCopyTile(uint32_t stageId, AscendC::GlobalTensor<ElementD> const &gmTileD, uint32_t tileLength,
                             ElementPerTokenScale quantMultiplier)
    {
        auto &ubQuantTmp = ubGateFp32List[stageId];
        auto ubQuantS32 = ubQuantTmp.template ReinterpretCast<int32_t>();
        auto ubQuantF16 = ubQuantTmp.template ReinterpretCast<half>();
        AscendC::Muls(ubQuantTmp, ubActivationList[stageId], quantMultiplier, tileLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(ubQuantS32, ubQuantTmp, AscendC::RoundMode::CAST_RINT, tileLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetDeqScale(static_cast<half>(1.0));
        AscendC::Cast(ubQuantF16, ubQuantS32, AscendC::RoundMode::CAST_RINT, tileLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[stageId]);
        AscendC::Cast(ubDList[stageId], ubQuantF16, AscendC::RoundMode::CAST_RINT, tileLength);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[stageId]);
        LayoutD tileLayout{1, tileLength};
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[stageId]);
        copyUbToGmD(gmTileD, ubDList[stageId], tileLayout, tileLayout);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[stageId]);
    }

    CATLASS_DEVICE
    void CopyDirectTile(uint32_t stageId, AscendC::GlobalTensor<ElementD> const &gmTileD, uint32_t tileLength)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[stageId]);
        AscendC::Cast(ubDList[stageId], ubActivationList[stageId], AscendC::RoundMode::CAST_RINT, tileLength);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[stageId]);
        LayoutD tileLayout{1, tileLength};
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[stageId]);
        copyUbToGmD(gmTileD, ubDList[stageId], tileLayout, tileLayout);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[stageId]);
    }

    AscendC::LocalTensor<ElementC> ubCList[UB_STAGES];
    AscendC::LocalTensor<ElementD> ubDList[UB_STAGES];
    AscendC::LocalTensor<float> ubGateFp32List[UB_STAGES];
    AscendC::LocalTensor<float> ubUpFp32List[UB_STAGES];
    AscendC::LocalTensor<float> ubActivationList[UB_STAGES];
    AscendC::LocalTensor<float> ubPerTokenScaleOutputList[SCALE_BUFFER_COUNT];
    AscendC::LocalTensor<float> ubRowAbsMax;
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer;

    int32_t eventUbCVMTE2List[UB_STAGES];
    int32_t eventUbCMTE2VList[UB_STAGES];
    int32_t eventUbDMTE3VList[UB_STAGES];
    int32_t eventUbDVMTE3List[UB_STAGES];
    uint32_t activationCode_{0};
    float activationParams1_{SwigluOaiActivation::DEFAULT_ALPHA};
    float activationParams2_{SituActivation::DEFAULT_BETA};
    CopyGmToUbC copyGmToUbC;
    CopyUbToGmD copyUbToGmD;
    CopyUbToGmDequantScale copyUbToGmDequantScale;
};

} // namespace Catlass::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_SWIGLU_A2_BF16_HPP
