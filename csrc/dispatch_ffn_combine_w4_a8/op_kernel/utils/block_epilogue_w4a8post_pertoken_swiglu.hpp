/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_W4A8_POST_PER_TOKEN_SWIGLU_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_W4A8_POST_PER_TOKEN_SWIGLU_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/callback.hpp"
#include "get_tensor_addr.hpp"


namespace Catlass::Epilogue::Block {

// float scale, dequant per expert
template <uint32_t UB_STAGES_, class CType_, class LayoutPerTokenScale_, class DType_, class TileElemWiseMuls_,
          class TileCopy_>
class BlockEpilogue<EpilogueAtlasA2W4A8PostPerTokenDequantSwigluQuant<UB_STAGES_>, CType_,
                    Gemm::GemmType<float, LayoutPerTokenScale_>, DType_, TileElemWiseMuls_, TileCopy_> {
public:
    using DispatchPolicy = EpilogueAtlasA2W4A8PostPerTokenDequantSwigluQuant<UB_STAGES_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;

    // Data infos
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementPerTokenScale = float;
    using LayoutPerTokenScale = LayoutPerTokenScale_;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;

    // Check data infos
    static_assert(std::is_same_v<ElementC, half> &&
                      (std::is_same_v<ElementD, float> || std::is_same_v<ElementD, int8_t>),
                  "The element type template parameters of BlockEpilogue are wrong");
    static_assert(std::is_same_v<LayoutC, layout::RowMajor> &&
                      std::is_same_v<LayoutPerTokenScale, layout::VectorLayout> &&
                      std::is_same_v<LayoutD, layout::RowMajor>,
                  "The layout template parameters of BlockEpilogue are wrong");

    // Tile compute ops

    // Tile copy
    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;
    using CopyUbToGmDequantScale =
        Epilogue::Tile::CopyUb2Gm<ArchTag, Gemm::GemmType<ElementPerTokenScale, LayoutPerTokenScale>>;

    struct Params {
        __gm__ ElementPerTokenScale *ptrPerTokenScale{nullptr};
        LayoutPerTokenScale layoutPerTokenScale{};
        __gm__ ElementD *ptrD{nullptr};
        LayoutD layoutD{};
        int32_t expertPerRank{0};

        CATLASS_DEVICE
        Params(){};

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
    BlockEpilogue(Arch::Resource<ArchTag> const &resource, int32_t n, Params const &params = Params{}) : params(params)
    {
        size_t ubOffset = 0;
        int32_t eventVMTE2 = 0;
        int32_t eventMTE2V = 0;
        int32_t eventMTE3V = 0;
        int32_t eventVMTE3 = 0;
        uint32_t blockN = n;
        uint32_t ChunkTileLen = blockN / 2;
        uint32_t HalfChunkTileLen = ChunkTileLen / 2;

        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubCList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
            ubOffset += blockN * sizeof(ElementC) * 2;

            ubweighAuxList[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += blockN * sizeof(float);

            xHighI4TensorList[i] = resource.ubBuf.template GetBufferByByte<int4b_t>(ubOffset);
            ubOffset += ChunkTileLen / 2;
            xLowI4TensorList[i] = resource.ubBuf.template GetBufferByByte<int4b_t>(ubOffset);
            ubOffset += ChunkTileLen / 2;

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

#ifdef W4A8_DEBUG
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID5);
#endif

        ubD = resource.ubBuf.template GetBufferByByte<ElementD>(ubOffset);
        ubOffset += blockN * sizeof(ElementD);
        ubCFp32 = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += blockN * sizeof(float) * 2;
        ubCFp32ChunkN = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += ChunkTileLen * sizeof(float);
        ubCFp32ChunkNAbs = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += ChunkTileLen * sizeof(float);
        ubCFp32ChunkNMax = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += HalfChunkTileLen * sizeof(float);
        ubQuantS32 = ubCFp32ChunkNAbs.template ReinterpretCast<int32_t>();
        ubQuantF16 = ubCFp32ChunkNAbs.template ReinterpretCast<half>();

        xLowHalfTensor = resource.ubBuf.template GetBufferByByte<half>(ubOffset);
        ubOffset += ChunkTileLen * sizeof(half);
        xLowHalfTensor2 = resource.ubBuf.template GetBufferByByte<half>(ubOffset);
        ubOffset += ChunkTileLen * sizeof(half);
        xLowI16Tensor = resource.ubBuf.template GetBufferByByte<int16_t>(ubOffset);
        ubOffset += 128 * sizeof(int16_t);

        ubPerTokenScaleOutput = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        // ubOffset += blockN * sizeof(float);
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
#ifdef W4A8_DEBUG
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID5);
#endif
    }
    CATLASS_DEVICE
    ~BlockEpilogue() {}

    CATLASS_DEVICE
    void UpdateParams(Params const &params_)
    {
        params = params_;
    }

    // Each tile is 1*7168, each block represents all tokens for one expert: [group[i], 7168]
    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementC> const &gmC, MatrixCoord const &shapeC,
                    AscendC::GlobalTensor<ElementPerTokenScale> const &gmPerTokenScale1,
                    __gm__ float *gmWeightAux, AscendC::GlobalTensor<ElementD> const &gmD,
                    AscendC::GlobalTensor<int32_t> const &cumsumMM, uint32_t MOffset,
                    AscendC::GlobalTensor<ElementPerTokenScale> const &gmPerTokenScale2, uint32_t expertPerRank,
                    uint32_t EP, AscendC::GlobalTensor<float> const &gmGMM1, int32_t rank, int32_t listLen,
                    uint32_t epilogueCoreNum = 40, Callback &&callback = Callback{})
    {
        callback();
        uint32_t blockM = shapeC.row();
        uint32_t blockN = shapeC.column();

        uint32_t tileLoops = blockM;
        uint32_t subblockIdx = get_block_idx() + get_subblockid() * get_block_num();

        uint32_t subblockNum = get_block_num() * 2;
        uint32_t moveDataCoreNum = subblockNum - epilogueCoreNum;

        if (subblockIdx < moveDataCoreNum) {
            return;
        }
        uint32_t epilogueCoreIdx = subblockIdx - moveDataCoreNum;

        uint32_t perCoreData = blockM / epilogueCoreNum;
        uint32_t remainderData = blockM % epilogueCoreNum;

        uint32_t tasksForIdx = epilogueCoreIdx < remainderData ? perCoreData + 1 : perCoreData;
        uint32_t loopStartIdx =
            epilogueCoreIdx * perCoreData + (epilogueCoreIdx < remainderData ? epilogueCoreIdx : remainderData);

        uint32_t alignedPerCoreData = RoundUp<BYTE_PER_BLK / sizeof(ElementPerTokenScale)>(perCoreData + 1);

        uint32_t ChunkTileLen = blockN / 2;
        uint32_t HalfChunkTileLen = ChunkTileLen / 2;
        const size_t LEN_VK = (ChunkTileLen / 2) / 128;
        const size_t LAST_LEN_VK = (ChunkTileLen % 256) / 2;
        const half ONE_SIXTEENTH = static_cast<half>(0.0625f);
        constexpr size_t LEN_128 = 128;

        constexpr float DEFAULT_MUL_SCALE = 16.0f;
        for (uint32_t loopIdx = loopStartIdx; loopIdx < loopStartIdx + tasksForIdx; ++loopIdx) {
            auto gmTileC = gmC[loopIdx * blockN * 2];

            auto &ubC = ubCList[ubListId];
            auto &ubweighAux = ubweighAuxList[ubListId];
            auto &xHighI4Tensor = xHighI4TensorList[ubListId];
            auto &xLowI4Tensor = xLowI4TensorList[ubListId];
            constexpr int32_t MASK = 128;
            Duplicate(xLowI16Tensor, static_cast<int16_t>(0x0F0F), MASK);
            PipeBarrier<PIPE_V>();

            auto &ubAbs = ubCFp32ChunkNAbs;
            auto &ubMax = ubCFp32ChunkNMax;
            auto &ubReduceMax = ubCFp32ChunkNMax;
            auto &ubOutputTmp = ubAbs;
            auto &sharedUbTmpBuffer = ubReduceMax;

            auto gmTileD = gmD[loopIdx * ChunkTileLen];
#ifdef W4A8_DEBUG
            auto gmTileGMM1 = gmGMM1[loopIdx * blockN];
#endif
            LayoutC layoutUbC{2, blockN};

            // Copy data from GM workspace to UB
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
            copyGmToUbC(ubC, gmTileC, layoutUbC, layoutUbC);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbWAVMTE2List[ubListId]);
            uint32_t globalOffset = MOffset + loopIdx;
            if (loopIdx < loopStartIdx + UB_STAGES)  // Initialize auxiliary matrix for the first time
            {
                curGroupIdx[ubListId] = 0;
                curSum[ubListId] = cumsumMM((EP - 1) * expertPerRank);
                for (; globalOffset >= curSum[ubListId] && curGroupIdx[ubListId] < expertPerRank;
                     ++curGroupIdx[ubListId]) {
                    curSum[ubListId] += cumsumMM((EP - 1) * expertPerRank + curGroupIdx[ubListId] + 1);
                }
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(blockN * sizeof(float)), 0, 0, 0};
                DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
                AscendC::GlobalTensor<float> weightAux;
                if (listLen == 1) { // Large tensor
                    weightAux.SetGlobalBuffer(gmWeightAux);
                    DataCopyPad(ubweighAux, weightAux[curGroupIdx[ubListId] * blockN], copyParams, padParams);
                } else {
                    weightAux.SetGlobalBuffer(GetTensorAddr<float>(curGroupIdx[ubListId], reinterpret_cast<GM_ADDR>(gmWeightAux)));
                    DataCopyPad(ubweighAux, weightAux, copyParams, padParams); // groupid = curGroupIdx[ubListId]
                }
            } else {  // May need to update auxiliary matrix in subsequent iterations
                uint32_t lastGroupIdx = curGroupIdx[ubListId];
                for (; globalOffset >= curSum[ubListId] && curGroupIdx[ubListId] < expertPerRank;
                     ++curGroupIdx[ubListId]) {
                    curSum[ubListId] += cumsumMM((EP - 1) * expertPerRank + curGroupIdx[ubListId] + 1);
                }
                bool update_weightAux = (lastGroupIdx != curGroupIdx[ubListId]);
                if (update_weightAux) {  // Need to update auxiliary matrix
                    DataCopyExtParams copyParams{1, static_cast<uint32_t>(blockN * sizeof(float)), 0, 0, 0};
                    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
                    AscendC::GlobalTensor<float> weightAux;
                    if (listLen == 1) { // Large tensor
                        weightAux.SetGlobalBuffer(gmWeightAux);
                        DataCopyPad(ubweighAux, weightAux[curGroupIdx[ubListId] * blockN], copyParams, padParams);
                    } else {
                        weightAux.SetGlobalBuffer(GetTensorAddr<float>(curGroupIdx[ubListId], reinterpret_cast<GM_ADDR>(gmWeightAux)));
                        DataCopyPad(ubweighAux, weightAux, copyParams, padParams);
                    }
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbWAMTE2VList[ubListId]);

            // Cast C to FP32 in UB
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);
#ifdef W4A8_DEBUG
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID5);
#endif
            AscendC::Cast(ubCFp32, ubC, AscendC::RoundMode::CAST_NONE, blockN * 2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
            PipeBarrier<PIPE_V>();
            AscendC::Muls(ubCFp32, ubCFp32, DEFAULT_MUL_SCALE, blockN);
            PipeBarrier<PIPE_V>();
            AscendC::Add(ubCFp32, ubCFp32, ubCFp32[blockN], blockN);
            // Add W4A8 auxiliary matrix in UB
            PipeBarrier<PIPE_V>();

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbWAMTE2VList[ubListId]);
            AscendC::Add(ubCFp32, ubCFp32, ubweighAux, blockN);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbWAVMTE2List[ubListId]);

            // Get per-token scale value from the loopIdx-th row of gmPerTokenScale
            ElementPerTokenScale perTokenScale = gmPerTokenScale1(loopIdx);

            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);  // S is scalar operation, ensure line 320 synchronization
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
            // Multiply FP32 C with per-token scale value
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(ubCFp32, ubCFp32, perTokenScale, blockN);

#ifdef W4A8_DEBUG
            // PipeBarrier<PIPE_ALL>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
            DataCopy(gmTileGMM1, ubCFp32, blockN);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID5);
            // PipeBarrier<PIPE_ALL>();
#endif

            // SwiGLU computation process
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(ubCFp32ChunkN, ubCFp32, -1.0f, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(ubCFp32ChunkN, ubCFp32ChunkN, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(ubCFp32ChunkN, ubCFp32ChunkN, 1.0f, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();
            // TODO: Check if division affects subsequent data
            AscendC::Div(ubCFp32ChunkN, ubCFp32, ubCFp32ChunkN, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(ubCFp32ChunkN, ubCFp32ChunkN, ubCFp32[ChunkTileLen], ChunkTileLen);

            // Quantization
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Abs(ubAbs, ubCFp32ChunkN, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceMax<float>(ubReduceMax, ubAbs, sharedUbTmpBuffer, ChunkTileLen, false);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            // AscendC::PipeBarrier<PIPE_V>();

            ElementPerTokenScale GMubDequantScale = ubReduceMax.GetValue(0);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
            auto ubPerTokenScaleOutputOffset = loopIdx - loopStartIdx;
            ubPerTokenScaleOutput.SetValue(ubPerTokenScaleOutputOffset, GMubDequantScale / 127.f);

            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
            AscendC::Muls(ubOutputTmp, ubCFp32ChunkN, 127.f / GMubDequantScale, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Cast(ubQuantS32, ubOutputTmp, AscendC::RoundMode::CAST_RINT, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetDeqScale(static_cast<half>(1.0));
            AscendC::Cast(ubQuantF16, ubQuantS32, AscendC::RoundMode::CAST_RINT, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Cast(ubD, ubQuantF16, AscendC::RoundMode::CAST_RINT, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();
            // High 4-bit processing start
            Cast(ubQuantF16, ubD, AscendC::RoundMode::CAST_NONE, ChunkTileLen);
            PipeBarrier<PIPE_V>();
            Muls(ubQuantF16, ubQuantF16, ONE_SIXTEENTH, ChunkTileLen);
            PipeBarrier<PIPE_V>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventxHighMTE3VList[ubListId]);  // Event ID needs adjustment
            Cast(xHighI4Tensor, ubQuantF16, AscendC::RoundMode::CAST_FLOOR, ChunkTileLen);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventxHighVMTE3List[ubListId]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventxHighVMTE3List[ubListId]);
            DataCopy(gmTileD, xHighI4Tensor.template ReinterpretCast<int8_t>(), ChunkTileLen / 2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventxHighMTE3VList[ubListId]);
            // High 4-bit processing end

            // Low 4-bit processing start
            And(xLowHalfTensor.template ReinterpretCast<int16_t>(), ubD.template ReinterpretCast<int16_t>(),
                xLowI16Tensor, LEN_128, LEN_VK, {1, 1, 1, 8, 8, 0});
            if (LAST_LEN_VK > 0) {
                And(xLowHalfTensor[LEN_VK * LEN_128].template ReinterpretCast<int16_t>(),
                    ubD[LEN_VK * LEN_128 * 2].template ReinterpretCast<int16_t>(), xLowI16Tensor, LAST_LEN_VK, 1,
                    {1, 1, 1, 8, 8, 0});
            }
            PipeBarrier<PIPE_V>();
            Cast(xLowHalfTensor2.template ReinterpretCast<half>(), xLowHalfTensor.template ReinterpretCast<int8_t>(),
                 AscendC::RoundMode::CAST_NONE, ChunkTileLen);
            PipeBarrier<PIPE_V>();
            const half MINUS_EIGHT = static_cast<half>(-8);
            Adds(ubQuantF16, xLowHalfTensor2, MINUS_EIGHT, ChunkTileLen);
            PipeBarrier<PIPE_V>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventxLowMTE3VList[ubListId]);
            Cast(xLowI4Tensor, ubQuantF16.template ReinterpretCast<half>(), AscendC::RoundMode::CAST_NONE,
                 ChunkTileLen);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventxLowVMTE3List[ubListId]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventxLowVMTE3List[ubListId]);
            DataCopy(gmTileD[ChunkTileLen / 2], xLowI4Tensor.template ReinterpretCast<int8_t>(), ChunkTileLen / 2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventxLowMTE3VList[ubListId]);

            ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
        }

        if (tasksForIdx > 0) {
            LayoutPerTokenScale layoutGmPerTokenScale2{tasksForIdx};
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            copyUbToGmDequantScale(gmPerTokenScale2[loopStartIdx], ubPerTokenScaleOutput[0], layoutGmPerTokenScale2,
                                   layoutGmPerTokenScale2);
        }
    }

private:
    Params params;

    AscendC::LocalTensor<ElementC> ubCList[UB_STAGES];
    AscendC::LocalTensor<ElementD> ubD;
    AscendC::LocalTensor<float> ubweighAuxList[UB_STAGES];
    AscendC::LocalTensor<int4b_t> xHighI4TensorList[UB_STAGES];
    AscendC::LocalTensor<int4b_t> xLowI4TensorList[UB_STAGES];
    AscendC::LocalTensor<half> xHighHalfTensor;
    AscendC::LocalTensor<half> xLowHalfTensor;
    AscendC::LocalTensor<half> xLowHalfTensor2;
    AscendC::LocalTensor<int16_t> xLowI16Tensor;

    uint32_t curGroupIdx[UB_STAGES];
    uint32_t curSum[UB_STAGES];

    int32_t eventUbCVMTE2List[UB_STAGES];
    int32_t eventUbCMTE2VList[UB_STAGES];
    int32_t eventUbWAVMTE2List[UB_STAGES];
    int32_t eventUbWAMTE2VList[UB_STAGES];
    int32_t eventxHighMTE3VList[UB_STAGES];
    int32_t eventxHighVMTE3List[UB_STAGES];
    int32_t eventxLowMTE3VList[UB_STAGES];
    int32_t eventxLowVMTE3List[UB_STAGES];

    uint32_t ubListId{0};

    AscendC::LocalTensor<float> ubCFp32;
    AscendC::LocalTensor<float> ubCFp32ChunkN;
    AscendC::LocalTensor<float> ubCFp32ChunkNAbs;
    AscendC::LocalTensor<float> ubCFp32ChunkNMax;
    AscendC::LocalTensor<int32_t> ubQuantS32;
    AscendC::LocalTensor<half> ubQuantF16;
    AscendC::LocalTensor<float> ubPerTokenScaleOutput;

    CopyGmToUbC copyGmToUbC;
    CopyUbToGmD copyUbToGmD;
    CopyUbToGmDequantScale copyUbToGmDequantScale;
};

}  // namespace Catlass::Epilogue::Block

#endif  // CATLASS_EPILOGUE_BLOCK_EPILOGUE_W4A8_POST_PER_TOKEN_SWIGLU_HPP