/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_SWIGLU_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_SWIGLU_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/callback.hpp"

namespace Catlass::Epilogue::Block {

// float scale, dequant per expert
template <
    uint32_t UB_STAGES_,
    class CType_,
    class LayoutPerTokenScale_,
    class DType_,
    class TileElemWiseMuls_,
    class TileCopy_
>
class BlockEpilogue <
    EpilogueAtlasA2PerTokenDequantSwigluQuant<UB_STAGES_>,
    CType_,
    Gemm::GemmType<float, LayoutPerTokenScale_>,
    DType_,
    TileElemWiseMuls_,
    TileCopy_
> {
public:
    using DispatchPolicy = EpilogueAtlasA2PerTokenDequantSwigluQuant<UB_STAGES_>;
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
    static_assert(
        std::is_same_v<ElementC, half> && (std::is_same_v<ElementD, float> || std::is_same_v<ElementD, int8_t>),
        "The element type template parameters of BlockEpilogue are wrong"
    );
    static_assert(
        std::is_same_v<LayoutC, layout::RowMajor> && 
            std::is_same_v<LayoutPerTokenScale, layout::VectorLayout> && std::is_same_v<LayoutD, layout::RowMajor>,
        "The layout template parameters of BlockEpilogue are wrong"
    );

    // Tile copy
    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;
    using CopyUbToGmDequantScale = Epilogue::Tile::CopyUb2Gm<ArchTag, Gemm::GemmType<ElementPerTokenScale, LayoutPerTokenScale>>;

    struct Params {
        __gm__ ElementPerTokenScale *ptrPerTokenScale{nullptr};
        LayoutPerTokenScale layoutPerTokenScale{};
        __gm__ ElementD *ptrD{nullptr};
        LayoutD layoutD{};

        CATLASS_DEVICE
        Params() {};

        CATLASS_DEVICE
        Params(__gm__ ElementPerTokenScale *ptrPerTokenScale_, LayoutPerTokenScale const &layoutPerTokenScale_,
            __gm__ ElementD *ptrD_, LayoutD const &layoutD_
        ) : ptrPerTokenScale(ptrPerTokenScale_), layoutPerTokenScale(layoutPerTokenScale_),
            ptrD(ptrD_), layoutD(layoutD_) {}
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
            ubOffset += blockN * sizeof(ElementC);
            ubDList[i] = resource.ubBuf.template GetBufferByByte<ElementD>(ubOffset);
            ubOffset += blockN * sizeof(ElementD);
            ubCFp32List[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += blockN * sizeof(float);
            ubCFp32ChunkNList[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += ChunkTileLen * sizeof(float);
            ubCFp32ChunkNAbsList[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += ChunkTileLen * sizeof(float);
            ubCFp32ChunkNMaxList[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += HalfChunkTileLen * sizeof(float);
            ubQuantS32List[i] = ubCFp32ChunkNAbsList[i].template ReinterpretCast<int32_t>();
            ubQuantF16List[i] = ubCFp32ChunkNAbsList[i].template ReinterpretCast<half>();

            eventUbCVMTE2List[i] = eventVMTE2++;
            eventUbCMTE2VList[i] = eventMTE2V++;
            eventUbDMTE3VList[i] = eventMTE3V++;
            eventUbDVMTE3List[i] = eventVMTE3++;

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);  
        }

        ubPerTokenScaleOutput = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
    }
    CATLASS_DEVICE
    void Finalize()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
    }
    CATLASS_DEVICE
    ~BlockEpilogue()
    {
    }

    CATLASS_DEVICE
    void UpdateParams(Params const &params_)
    {
        params = params_;
    }
    // Each tile is 1x7168, and each block covers all tokens for one expert = [group[i], 7168]
    CATLASS_DEVICE
    void operator() (
        AscendC::GlobalTensor<ElementC> const &gmC,
        MatrixCoord const &shapeC,
        AscendC::GlobalTensor<ElementPerTokenScale> const &gmPerTokenScale1,
        AscendC::GlobalTensor<ElementD> const &gmD,
        AscendC::GlobalTensor<ElementPerTokenScale> const &gmPerTokenScale2,

        uint32_t epilogueCoreNum = 40,
        Callback &&callback = Callback{}
    )
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

        uint32_t perCoreData =  blockM / epilogueCoreNum;
        uint32_t remainderData = blockM % epilogueCoreNum;

        uint32_t tasksForIdx  = epilogueCoreIdx < remainderData ? perCoreData + 1 : perCoreData;
        uint32_t loopStartIdx = epilogueCoreIdx * perCoreData + (epilogueCoreIdx < remainderData? epilogueCoreIdx : remainderData);

        uint32_t alignedPerCoreData = RoundUp<BYTE_PER_BLK / sizeof(ElementPerTokenScale)>(perCoreData + 1);

        uint32_t ChunkTileLen = blockN / 2;
        uint32_t HalfChunkTileLen = ChunkTileLen / 2;


        for (uint32_t loopIdx = loopStartIdx; loopIdx < loopStartIdx + tasksForIdx; ++loopIdx) {

            auto gmTileC = gmC[loopIdx * blockN];

            auto &ubC = ubCList[ubListId];
            auto &ubD = ubDList[ubListId];

            auto &ubCFp32 = ubCFp32List[ubListId];
            auto &ubCFp32ChunkN = ubCFp32ChunkNList[ubListId]; 
            auto &ubAbs = ubCFp32ChunkNAbsList[ubListId];
            // auto &ubMax = ubCFp32ChunkNMaxList[ubListId];
            auto &ubReduceMax = ubCFp32ChunkNMaxList[ubListId];
            auto &ubOutputTmp = ubAbs;
            auto &sharedUbTmpBuffer = ubReduceMax;
            auto &ubQuantS32 = ubQuantS32List[ubListId];
            auto &ubQuantF16 = ubQuantF16List[ubListId];

            auto gmTileD = gmD[loopIdx * ChunkTileLen];
            LayoutC layoutUbC{1, blockN};

            // Move C from GM workspace to UB
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
            copyGmToUbC(ubC, gmTileC, layoutUbC, layoutUbC);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);

            // Cast C to FP32 in UB
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);
            AscendC::Cast(ubCFp32, ubC, AscendC::RoundMode::CAST_NONE, blockN);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);

            // Get per-token scale from row loopIdx of gmPerTokenScale
            ElementPerTokenScale perTokenScale = gmPerTokenScale1(loopIdx);

            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
            // Multiply FP32 C by the per-token scale
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(ubCFp32, ubCFp32, perTokenScale, blockN);
            AscendC::PipeBarrier<PIPE_V>();

            // Swiglu computation process
            AscendC::Muls(ubCFp32ChunkN, ubCFp32, -1.0f, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(ubCFp32ChunkN, ubCFp32ChunkN, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(ubCFp32ChunkN, ubCFp32ChunkN, 1.0f, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();
            // TODO: confirm whether the division impacts subsequent data
            AscendC::Div(ubCFp32ChunkN, ubCFp32, ubCFp32ChunkN, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(ubCFp32ChunkN, ubCFp32ChunkN, ubCFp32[ChunkTileLen], ChunkTileLen);
            
            // Quantization process; difference between the two approaches
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Abs(ubAbs, ubCFp32ChunkN, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::ReduceMax<float>(ubReduceMax, ubAbs, sharedUbTmpBuffer, ChunkTileLen, false);
            AscendC::PipeBarrier<PIPE_V>();
            
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);

            // TODO: compare the efficiency of the two calculation methods
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

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDVMTE3List[ubListId]);
            AscendC::Cast(ubD, ubQuantF16, AscendC::RoundMode::CAST_RINT, ChunkTileLen);
            // AscendC::Muls(ubD, ubCFp32ChunkN, 127.f / GMubDequantScale, ChunkTileLen);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDMTE3VList[ubListId]);         

            LayoutD layoutUbD{1, ChunkTileLen};
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
            copyUbToGmD(gmTileD, ubD, layoutUbD, layoutUbD);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
            ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
        }

        if(tasksForIdx > 0){
            LayoutPerTokenScale layoutGmPerTokenScale2{tasksForIdx};

            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);

            copyUbToGmDequantScale(gmPerTokenScale2[loopStartIdx], ubPerTokenScaleOutput[0], layoutGmPerTokenScale2, layoutGmPerTokenScale2);
        }
        

    }

private:
    Params params;

    AscendC::LocalTensor<ElementC> ubCList[UB_STAGES];
    AscendC::LocalTensor<ElementD> ubDList[UB_STAGES];

    int32_t eventUbCVMTE2List[UB_STAGES];
    int32_t eventUbCMTE2VList[UB_STAGES];
    int32_t eventUbDMTE3VList[UB_STAGES];
    int32_t eventUbDVMTE3List[UB_STAGES];

    uint32_t ubListId{0};

    AscendC::LocalTensor<float> ubCFp32List[UB_STAGES];
    AscendC::LocalTensor<float> ubCFp32ChunkNList[UB_STAGES];
    AscendC::LocalTensor<float> ubCFp32ChunkNAbsList[UB_STAGES];
    AscendC::LocalTensor<float> ubCFp32ChunkNMaxList[UB_STAGES];
    AscendC::LocalTensor<int32_t> ubQuantS32List[UB_STAGES];
    AscendC::LocalTensor<half> ubQuantF16List[UB_STAGES];
    AscendC::LocalTensor<float> ubPerTokenScaleOutput;


    CopyGmToUbC copyGmToUbC;
    CopyUbToGmD copyUbToGmD;
    CopyUbToGmDequantScale copyUbToGmDequantScale;
};

}  // namespace Catlass::Epilogue::Block

#endif  // CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_SWIGLU_HPP
