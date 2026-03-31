/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACT_EPILOGUE_BLOCK_EPILOGUE_BF16_FP16_HPP
#define ACT_EPILOGUE_BLOCK_EPILOGUE_BF16_FP16_HPP

#include "../../raw_distributed/cam_moe_distribute_combine.h"
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/detail/callback.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Epilogue::Block {

template <uint32_t UB_STAGES_, uint32_t EXEC_FLAG_,
          class CType_, class ScaleType_, class LayoutScale_, class LayoutPerTokenScale_, class DType_,
          class TileRowBroadcastMul_, class TileBroadcastOneBlk_, class TileOneBlkColumnBroadcastMul_,
          class TileCopy_, class EpilogueTileSwizzle_>
class BlockEpilogue<EpilogueAtlasA2Combine<UB_STAGES_, EXEC_FLAG_>,
          CType_, Gemm::GemmType<ScaleType_, LayoutScale_>, Gemm::GemmType<float, LayoutPerTokenScale_>, DType_,
          TileRowBroadcastMul_, TileBroadcastOneBlk_, TileOneBlkColumnBroadcastMul_,
          TileCopy_, EpilogueTileSwizzle_>
{
public:
    using DispatchPolicy = EpilogueAtlasA2Combine<UB_STAGES_, EXEC_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    static constexpr uint32_t EXEC_FLAG = EXEC_FLAG_;

    // Data infos
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementRawScale = ScaleType_;
    using ElementFp32Scale = float;
    using LayoutScale = LayoutScale_;
    using ElementPerTokenScale = float;
    using LayoutPerTokenScale = LayoutPerTokenScale_;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;

    // Check data infos
    static_assert(std::is_same_v<ElementC, float> &&
                      (std::is_same_v<ElementD, half> || std::is_same_v<ElementD, bfloat16_t>),
                  "The element type template parameters of BlockEpilogue are wrong");
    static_assert(std::is_same_v<LayoutC, layout::RowMajor> && std::is_same_v<LayoutScale, layout::VectorLayout> &&
                      std::is_same_v<LayoutPerTokenScale, layout::VectorLayout> &&
                      std::is_same_v<LayoutD, layout::RowMajor>,
                  "The layout template parameters of BlockEpilogue are wrong");

    // Tile compute ops
    using TileRowBroadcastMul = TileRowBroadcastMul_;
    using TileBroadcastOneBlk = TileBroadcastOneBlk_;
    using TileOneBlkColumnBroadcastMul = TileOneBlkColumnBroadcastMul_;

    // Tile copy
    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyGmToUbScale = typename TileCopy_::CopyGmToUbX;
    using CopyGmToUbPerTokenScale = typename TileCopy_::CopyGmToUbY;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;

    using EpilogueTileSwizzle = EpilogueTileSwizzle_;

    using TileShape = typename TileRowBroadcastMul::TileShape;

    static_assert(TileShape::ROW == TileBroadcastOneBlk::COMPUTE_LENGTH &&
                      std::is_same_v<TileShape, typename TileOneBlkColumnBroadcastMul::TileShape>,
                  "TileShape must be consistent for all tile compute ops");

    static_assert((UB_STAGES * (TileShape::COUNT * sizeof(ElementC) + TileShape::COUNT * sizeof(ElementD)) +
                   TileShape::ROW * BYTE_PER_BLK) <= ArchTag::UB_SIZE,
                  "TileShape is too large to fit in UB");

    struct Params {
        __gm__ ElementRawScale *ptrScale{nullptr};
        LayoutScale layoutScale{};
        __gm__ ElementPerTokenScale *ptrPerTokenScale{nullptr};
        LayoutPerTokenScale layoutPerTokenScale{};
        __gm__ ElementD *ptrD{nullptr};
        LayoutD layoutD{};

        CATLASS_DEVICE
        Params() {};

        CATLASS_DEVICE
        Params(__gm__ ElementRawScale *ptrScale_, LayoutScale const &layoutScale_,
               __gm__ ElementPerTokenScale *ptrPerTokenScale_, LayoutPerTokenScale const &layoutPerTokenScale_,
               __gm__ ElementD *ptrD_, LayoutD const &layoutD_)
            : ptrScale(ptrScale_),
              layoutScale(layoutScale_),
              ptrPerTokenScale(ptrPerTokenScale_),
              layoutPerTokenScale(layoutPerTokenScale_),
              ptrD(ptrD_),
              layoutD(layoutD_)
        {}
    };

    CATLASS_DEVICE void AlignUbOffset()
    {
        size_t ubMask = ubOffset & (MoeDistributeCombineImpl::UB_ALIGN - 1);
        if (ubMask != 0) {
            ubOffset += MoeDistributeCombineImpl::UB_ALIGN - ubMask;
        }
    }

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource, MoeDistributeCombineImpl::CombineCalcInfo &calcInfo,
                  Params const &params = Params{})
        : resource(resource), calcInfo(calcInfo), params(params)
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubCList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
            ubOffset += TileShape::COUNT * sizeof(ElementC);
            ubDList[i] = resource.ubBuf.template GetBufferByByte<ElementD>(ubOffset);
            ubOffset += TileShape::COUNT * sizeof(ElementD);

            eventUbCVMTE2List[i] = eventVMTE2++;
            eventUbCMTE2VList[i] = eventMTE2V++;
            eventUbDMTE3VList[i] = eventMTE3V++;
            eventUbDVMTE3List[i] = eventVMTE3++;

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }

        if constexpr (EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) {
            AlignUbOffset();
            epSendCountLocal_ = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset);
            ubOffset += calcInfo.moeSendNum_ * sizeof(int32_t);
            AlignUbOffset();
            AscendC::GlobalTensor<int32_t> epSendCountGM;
            epSendCountGM.SetGlobalBuffer((__gm__ int32_t *)calcInfo.epSendCount_);
            uint32_t epSendCountSize = calcInfo.isShardExpert_ ? calcInfo.epWorldSize_ : calcInfo.moeSendNum_;
            AscendC::DataCopyExtParams epSendCntParams = {1U, static_cast<uint32_t>(epSendCountSize * sizeof(uint32_t)),
                                                          0U, 0U, 0U};
            AscendC::DataCopyPadExtParams<int32_t> copyPadParams{false, 0U, 0U, 0U};
            AscendC::DataCopyPad(epSendCountLocal_, epSendCountGM, epSendCntParams, copyPadParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(eventMTE2S);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(eventMTE2S);
        }
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
    }

    CATLASS_DEVICE
    void UpdateParams(Params const &params_)
    {
        params = params_;
    }

    CATLASS_DEVICE GM_ADDR GetWinAddrByRankId(const int32_t rankId, const uint8_t expertLocalId = 0U)
    {
        return (GM_ADDR)((calcInfo.epRankId_ == rankId)
                             ? calcInfo.epWinContext_->localWindowsIn
                             : ((HcclRankRelationResV2 *)(calcInfo.epWinContext_->remoteRes[rankId].nextDevicePtr))
                                   ->windowsIn) +
               calcInfo.winDataSizeOffset_ + expertLocalId * calcInfo.expertPerSizeOnWin_ + rankId * OPT_RANK_OFFSET;
    }

    CATLASS_DEVICE void SetCombineSendEpRank(uint32_t epRank, uint32_t &remoteEpRank, uint32_t &localEpRank)
    {
        if ((calcInfo.isShardExpert_) && (epRank < calcInfo.sharedExpertRankNum_)) {
            remoteEpRank = calcInfo.epRankId_;
            localEpRank = epRank;
        } else {
            remoteEpRank = epRank;
            localEpRank = calcInfo.epRankId_;
        }
    }

    CATLASS_DEVICE void DoCombineSend(AscendC::LocalTensor<ElementD> &ubD, layout::RowMajor &layoutGmTileD,
                                  LayoutD &layoutUbD, int64_t groupOffsetD, uint32_t expertIdx, uint32_t tileOffsetD)
    {
        const uint32_t copyTokenLen = layoutGmTileD.shape(1) * sizeof(ElementD);
        const uint32_t copyTokenSrcStride =
            (layoutUbD.stride(0) - layoutUbD.shape(1)) / (BYTE_PER_C0 / sizeof(ElementD));
        const uint32_t copyTokenDstStride = (layoutGmTileD.stride(0) - layoutGmTileD.shape(1)) * sizeof(ElementD);

        int64_t offsetD = groupOffsetD + tileOffsetD;
        uint32_t startToken = offsetD / calcInfo.axisH_;
        uint32_t tokenOffset = offsetD - startToken * calcInfo.axisH_;
        uint32_t itToken = startToken;
        uint32_t endToken = startToken + layoutGmTileD.shape(0);
        constexpr uint32_t epRankStart = 0;
        uint32_t sendCount =
            expertIdx == 0 && epRankStart == 0 ? 0 : epSendCountLocal_.GetValue(expertOffset + epRankStart - 1);
        for (uint32_t epRank = epRankStart; epRank < calcInfo.epWorldSize_ && itToken < endToken; ++epRank) {
            uint32_t prevSendCount = sendCount;
            sendCount = epSendCountLocal_.GetValue(expertOffset + epRank);
            if (prevSendCount <= itToken && itToken < sendCount) {
                uint32_t copyTokenCount = (sendCount < endToken ? sendCount : endToken) - itToken;
                AscendC::DataCopyExtParams dataCopyParams(copyTokenCount, copyTokenLen, copyTokenSrcStride,
                                                          copyTokenDstStride, 0);
                uint32_t remoteEpRank;
                uint32_t localEpRank;
                SetCombineSendEpRank(epRank, remoteEpRank, localEpRank);
                GM_ADDR rankGM = GetWinAddrByRankId(remoteEpRank, expertIdx) +
                                 localEpRank * calcInfo.moeExpertPerRankNum_ * calcInfo.expertPerSizeOnWin_;
                AscendC::GlobalTensor<ElementD> rankWindow;
                rankWindow.SetGlobalBuffer((__gm__ ElementD *)rankGM);
                AscendC::DataCopyPad(rankWindow[(itToken - prevSendCount) * calcInfo.axisH_ + tokenOffset],
                                     ubD[(itToken - startToken) * layoutUbD.stride(0)], dataCopyParams);
                itToken += copyTokenCount;
            }
        }
    }

    CATLASS_DEVICE
    void operator()(int64_t groupOffsetD, uint32_t expertIdx, GemmCoord const &blockShapeMNK,
                    GemmCoord const &blockCoordMNK, GemmCoord const &actualBlockShapeMNK,
                    AscendC::GlobalTensor<ElementC> const &gmBlockC, LayoutC const &layoutBlockC,
                    Callback &&callback = Callback{})
    {
        if (actualBlockShapeMNK.k() == 0) {
            return;
        }

        if constexpr (EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) {
            expertOffset = expertIdx * calcInfo.epWorldSize_;
        }

        callback();
        // Calculate the offset of the current block
        MatrixCoord blockShape = blockShapeMNK.GetCoordMN();
        MatrixCoord blockCoord = blockCoordMNK.GetCoordMN();
        MatrixCoord actualBlockShape = actualBlockShapeMNK.GetCoordMN();
        MatrixCoord blockOffset = blockCoord * blockShape;

        AscendC::GlobalTensor<ElementRawScale> gmScale;
        gmScale.SetGlobalBuffer(params.ptrScale);
        AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale;
        gmPerTokenScale.SetGlobalBuffer(params.ptrPerTokenScale);
        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer(params.ptrD);

        auto ubTileStride = MakeCoord(static_cast<int64_t>(TileShape::COLUMN), 1L);
        auto tileShape = TileShape::ToCoord();
        EpilogueTileSwizzle epilogueTileSwizzle(actualBlockShape, tileShape);
        uint32_t tileLoops = epilogueTileSwizzle.GetLoops();
        uint32_t subblockIdx = AscendC::GetSubBlockIdx();
        uint32_t subblockNum = AscendC::GetSubBlockNum();
        for (uint32_t loopIdx = subblockIdx; loopIdx < tileLoops; loopIdx += subblockNum) {
            auto tileCoord = epilogueTileSwizzle.GetTileCoord(loopIdx);
            auto actualTileShape = epilogueTileSwizzle.GetActualTileShape(tileCoord);
            auto tileOffsetInBlock = tileCoord * tileShape;
            auto tileOffset = blockOffset + tileOffsetInBlock;

            auto gmTileC = gmBlockC[layoutBlockC.GetOffset(tileOffsetInBlock)];
            auto layoutGmTileC = layoutBlockC.GetTileLayout(actualTileShape);

            auto &ubC = ubCList[ubListId];
            LayoutC layoutUbC{actualTileShape, ubTileStride};

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
            copyGmToUbC(ubC, gmTileC, layoutUbC, layoutGmTileC);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);

            auto &ubD = ubDList[ubListId];
            LayoutD layoutUbD{actualTileShape, ubTileStride};

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
            AscendC::Cast(ubD, ubC, AscendC::RoundMode::CAST_RINT, TileShape::COUNT);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);

            auto tileOffsetD = params.layoutD.GetOffset(tileOffset);
            auto layoutGmTileD = params.layoutD.GetTileLayout(actualTileShape);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);

            if constexpr (EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) {
                DoCombineSend(ubD, layoutGmTileD, layoutUbD, groupOffsetD, expertIdx, tileOffsetD);
            } else {
                auto gmTileD = gmD[tileOffsetD];
                copyUbToGmD(gmTileD, ubD, layoutGmTileD, layoutUbD);
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);

            ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
        }
    }

private:
    Params params;
    Arch::Resource<ArchTag> &resource;
    MoeDistributeCombineImpl::CombineCalcInfo calcInfo;

    AscendC::LocalTensor<ElementC> ubCList[UB_STAGES];
    AscendC::LocalTensor<ElementD> ubDList[UB_STAGES];

    int32_t eventUbCVMTE2List[UB_STAGES];
    int32_t eventUbCMTE2VList[UB_STAGES];
    int32_t eventUbScaleVMTE2List[UB_STAGES];
    int32_t eventUbScaleMTE2VList[UB_STAGES];
    int32_t eventUbPerTokenScaleVMTE2List[UB_STAGES];
    int32_t eventUbPerTokenScaleMTE2VList[UB_STAGES];
    int32_t eventUbDMTE3VList[UB_STAGES];
    int32_t eventUbDVMTE3List[UB_STAGES];

    AscendC::LocalTensor<int32_t> epSendCountLocal_;

    size_t ubOffset{0};
    int32_t eventVMTE2{0};
    int32_t eventMTE2V{0};
    int32_t eventMTE3V{0};
    int32_t eventVMTE3{0};
    int32_t eventVS{0};
    int32_t eventMTE2S{0};

    uint32_t expertOffset;

    uint32_t ubListId{0};

    CopyGmToUbC copyGmToUbC;
    CopyUbToGmD copyUbToGmD;
};

}  // namespace Catlass::Epilogue::Block

#endif  // ACT_EPILOGUE_BLOCK_EPILOGUE_BF16_FP16_HPP
