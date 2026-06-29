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

#include "../template_linear_algebra_v2/catlass.hpp"
#include "../template_linear_algebra_v2/arch/resource.hpp"
#include "../template_linear_algebra_v2/epilogue/dispatch_policy.hpp"
#include "../template_linear_algebra_v2/gemm_coord.hpp"
#include "../template_linear_algebra_v2/matrix_coord.hpp"
#include "../template_linear_algebra_v2/layout/layout.hpp"
#include "../template_linear_algebra_v2/detail/callback.hpp"

namespace Catlass::Epilogue::Block {

// A2 BF16/FP16 SwiGLU (non-quantized): SiLU(x_gate) * x_up, cast to output.
// 6-arg operator() matches the A2 kernel call pattern (no `resource` parameter).
// Uses CAST_RINT to work around CANN 9.1.0 AIV compiler limitations on
// float→bfloat16 CAST_ROUND.
template <
    uint32_t UB_STAGES_,
    class CType_,
    class LayoutPerTokenScale_,
    class DType_,
    class TileElemWiseMuls_,
    class TileCopy_
>
class BlockEpilogue <
    EpilogueAtlasA2PerTokenDequantSwigluQuantBF16<UB_STAGES_>,
    CType_,
    Gemm::GemmType<float, LayoutPerTokenScale_>,
    DType_,
    TileElemWiseMuls_,
    TileCopy_
> {
public:
    using DispatchPolicy = EpilogueAtlasA2PerTokenDequantSwigluQuantBF16<UB_STAGES_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;

    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementPerTokenScale = float;
    using LayoutPerTokenScale = LayoutPerTokenScale_;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;

    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;
    using CopyUbToGmDequantScale = Epilogue::Tile::CopyUb2Gm<ArchTag, Gemm::GemmType<ElementPerTokenScale, LayoutPerTokenScale>>;

    struct Params {
        CATLASS_DEVICE Params() {};
    };

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> const &resource, int32_t n, Params const &params = Params{})
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

    CATLASS_DEVICE ~BlockEpilogue() {}

    // 7-arg: per-token dequant + SwiGLU + quant → cast & copy to D.
    // Matches the unified kernel call pattern (receives gmPerTokenScale2
    // for interface uniformity; writes dequant scale back for INT8).
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

        uint32_t subblockIdx = get_block_idx() + get_subblockid() * get_block_num();
        uint32_t subblockNum = get_block_num() * 2;
        uint32_t moveDataCoreNum = subblockNum - epilogueCoreNum;

        if (subblockIdx < moveDataCoreNum) return;

        uint32_t epilogueCoreIdx = subblockIdx - moveDataCoreNum;
        uint32_t perCoreData = blockM / epilogueCoreNum;
        uint32_t remainderData = blockM % epilogueCoreNum;
        uint32_t tasksForIdx = epilogueCoreIdx < remainderData ? perCoreData + 1 : perCoreData;
        uint32_t loopStartIdx = epilogueCoreIdx * perCoreData +
            (epilogueCoreIdx < remainderData ? epilogueCoreIdx : remainderData);

        uint32_t ChunkTileLen = blockN / 2;

        for (uint32_t loopIdx = loopStartIdx; loopIdx < loopStartIdx + tasksForIdx; ++loopIdx) {
            auto gmTileC = gmC[loopIdx * blockN];
            auto &ubC = ubCList[ubListId];
            auto &ubD = ubDList[ubListId];
            auto &ubCFp32 = ubCFp32List[ubListId];
            auto &ubCFp32ChunkN = ubCFp32ChunkNList[ubListId];
            auto &ubAbs = ubCFp32ChunkNAbsList[ubListId];
            auto &ubReduceMax = ubCFp32ChunkNMaxList[ubListId];
            auto &ubOutputTmp = ubAbs;
            auto &sharedUbTmpBuffer = ubReduceMax;
            auto &ubQuantS32 = ubQuantS32List[ubListId];
            auto &ubQuantF16 = ubQuantF16List[ubListId];
            auto gmTileD = gmD[loopIdx * ChunkTileLen];
            LayoutC layoutUbC{1, blockN};

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
            copyGmToUbC(ubC, gmTileC, layoutUbC, layoutUbC);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);
            AscendC::Cast(ubCFp32, ubC, AscendC::RoundMode::CAST_NONE, blockN);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);

            ElementPerTokenScale perTokenScale = gmPerTokenScale1(loopIdx);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(ubCFp32, ubCFp32, perTokenScale, blockN);
            AscendC::PipeBarrier<PIPE_V>();

            // SiLU(x_gate) * x_up
            AscendC::Muls(ubCFp32ChunkN, ubCFp32, -1.0f, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(ubCFp32ChunkN, ubCFp32ChunkN, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(ubCFp32ChunkN, ubCFp32ChunkN, 1.0f, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(ubCFp32ChunkN, ubCFp32, ubCFp32ChunkN, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(ubCFp32ChunkN, ubCFp32ChunkN, ubCFp32[ChunkTileLen], ChunkTileLen);

            // Per-token dynamic quant
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Abs(ubAbs, ubCFp32ChunkN, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceMax<float>(ubReduceMax, ubAbs, sharedUbTmpBuffer, ChunkTileLen, false);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);

            ElementPerTokenScale dequantScale = ubReduceMax.GetValue(0);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
            auto ubPerTokenScaleOffset = loopIdx - loopStartIdx;
            ubPerTokenScaleOutput.SetValue(ubPerTokenScaleOffset, dequantScale / 127.f);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
            AscendC::Muls(ubOutputTmp, ubCFp32ChunkN, 127.f / dequantScale, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Cast(ubQuantS32, ubOutputTmp, AscendC::RoundMode::CAST_RINT, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetDeqScale(static_cast<half>(1.0));
            AscendC::Cast(ubQuantF16, ubQuantS32, AscendC::RoundMode::CAST_RINT, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
            AscendC::Cast(ubD, ubQuantF16, AscendC::RoundMode::CAST_RINT, ChunkTileLen);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);

            LayoutD layoutUbD{1, ChunkTileLen};
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
            copyUbToGmD(gmTileD, ubD, layoutUbD, layoutUbD);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);

            ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
        }

        if (tasksForIdx > 0) {
            LayoutPerTokenScale layoutGmPerTokenScale2{tasksForIdx};
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            copyUbToGmDequantScale(gmPerTokenScale2[loopStartIdx], ubPerTokenScaleOutput[0],
                                   layoutGmPerTokenScale2, layoutGmPerTokenScale2);
        }
    }

    // 5-arg: BF16-only SwiGLU without dequant, direct Cast to output.
    CATLASS_DEVICE
    void operator() (
        AscendC::GlobalTensor<ElementC> const &gmC,
        MatrixCoord const &shapeC,
        AscendC::GlobalTensor<ElementD> const &gmD,
        uint32_t epilogueCoreNum = 40,
        Callback &&callback = Callback{}
    )
    {
        callback();
        uint32_t blockM = shapeC.row();
        uint32_t blockN = shapeC.column();

        uint32_t subblockIdx = get_block_idx() + get_subblockid() * get_block_num();
        uint32_t subblockNum = get_block_num() * 2;
        uint32_t moveDataCoreNum = subblockNum - epilogueCoreNum;

        if (subblockIdx < moveDataCoreNum) return;

        uint32_t epilogueCoreIdx = subblockIdx - moveDataCoreNum;
        uint32_t perCoreData = blockM / epilogueCoreNum;
        uint32_t remainderData = blockM % epilogueCoreNum;
        uint32_t tasksForIdx = epilogueCoreIdx < remainderData ? perCoreData + 1 : perCoreData;
        uint32_t loopStartIdx = epilogueCoreIdx * perCoreData +
            (epilogueCoreIdx < remainderData ? epilogueCoreIdx : remainderData);

        uint32_t ChunkTileLen = blockN / 2;

        for (uint32_t loopIdx = loopStartIdx; loopIdx < loopStartIdx + tasksForIdx; ++loopIdx) {
            auto gmTileC = gmC[loopIdx * blockN];
            auto &ubC = ubCList[ubListId];
            auto &ubD = ubDList[ubListId];
            auto &ubCFp32 = ubCFp32List[ubListId];
            auto &ubCFp32ChunkN = ubCFp32ChunkNList[ubListId];
            auto gmTileD = gmD[loopIdx * ChunkTileLen];
            LayoutC layoutUbC{1, blockN};

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
            copyGmToUbC(ubC, gmTileC, layoutUbC, layoutUbC);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);
            AscendC::Cast(ubCFp32, ubC, AscendC::RoundMode::CAST_NONE, blockN);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);

            // SiLU(x_gate) * x_up
            AscendC::Muls(ubCFp32ChunkN, ubCFp32, -1.0f, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(ubCFp32ChunkN, ubCFp32ChunkN, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(ubCFp32ChunkN, ubCFp32ChunkN, 1.0f, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(ubCFp32ChunkN, ubCFp32, ubCFp32ChunkN, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(ubCFp32ChunkN, ubCFp32ChunkN, ubCFp32[ChunkTileLen], ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDVMTE3List[ubListId]);
            AscendC::Cast(ubD, ubCFp32ChunkN, AscendC::RoundMode::CAST_RINT, ChunkTileLen);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDMTE3VList[ubListId]);

            LayoutD layoutUbD{1, ChunkTileLen};
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
            copyUbToGmD(gmTileD, ubD, layoutUbD, layoutUbD);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);

            ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
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

#endif  // CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_SWIGLU_A2_BF16_HPP
