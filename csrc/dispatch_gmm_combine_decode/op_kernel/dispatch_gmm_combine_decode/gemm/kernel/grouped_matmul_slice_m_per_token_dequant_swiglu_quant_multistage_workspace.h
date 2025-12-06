/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "catlass/catlass.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/detail/callback.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

#include "../../../dispatch_gmm_combine_decode_base.h"

constexpr uint32_t STATE_OFFSET = 512;
constexpr uint64_t WIN_STATE_OFFSET = 512 * 1024;
constexpr uint64_t STATE_WIN_OFFSET = 900 * 1024;
constexpr uint64_t GROUP_TOKEN_NUM_OFFSET = 932 * 1024;
constexpr uint64_t SOFT_SYNC_OFFSET = 964 * 1024;
constexpr uint32_t SELF_STATE_OFFSET = 256 * 1024;
constexpr uint32_t SUM_TMP_TENSOR_SIZE = 1024;
constexpr uint32_t UB_ALIGN = 32;
constexpr uint32_t TOKEN_EXTRA_SPACE = 512;
constexpr uint32_t INT32_COUNT_PER_BLOCK = 8;
constexpr uint32_t SOFT_SYNC_SPACE_SIZE = 512;
constexpr int64_t LOOP_TMP_SIZE = 4096;
constexpr int32_t SUB_AIV_NUM = 2;
constexpr int32_t ODD_EVEN_BASE = 2;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t GATHER_SECOND_NUM = 2;
constexpr uint32_t MAX_QUANT_ROW_ONCE = 8;
constexpr uint32_t QUANT_SPACE_FACTOR = 176 * 1024 / 11;  // up to 176KB for quant
#define OPT_RANK_OFFSET 512

#define CEIL_UP(x) ((x + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN)
#define CEIL(x, y) (((x) + (y - 1)) / (y))
#define UB_BLOCK_SIZE (32)
#define GET_WIND_STATE_ADDR_BY_RANK_ID(rankId)                                                                    \
    (((epRankId == rankId)                                                                                        \
          ? ((GM_ADDR)(winContext_->localWindowsExp))                                                             \
          : ((GM_ADDR)(((HcclRankRelationResV2 *)(winContext_->remoteRes[rankId].nextDevicePtr))->windowsExp))) + \
     dataState * WIN_STATE_OFFSET)
#define GET_WIND_ADDR_BY_RANK_ID(rankId)                                                                         \
    (((epRankId == rankId)                                                                                       \
          ? ((GM_ADDR)(winContext_->localWindowsIn))                                                             \
          : ((GM_ADDR)(((HcclRankRelationResV2 *)(winContext_->remoteRes[rankId].nextDevicePtr))->windowsIn))) + \
     winDataSizeOffset + rankId * OPT_RANK_OFFSET)
#define TOKEN_FLAG_1 (0x55555555)
#define TOKEN_FLAG_2 (0x33333333)
#define V_TO_C_FLAG_1 (0x03030303)
#define V_TO_C_FLAG_2 (0x05050505)
#define CV_FLAG_INDEX 0
#define GROUP_ID_INDEX 1
#define PRE_COUNT_INDEX 2
#define SELF_COUNT_INDEX 3
#define TOTAL_COUNT_INDEX 4
#define GROUP_TOKEN_COUNT 3  // equal to SELF_COUNT_INDEX
#define GROUP_INFO_SIZE 32

namespace Catlass::Gemm::Kernel {

template <class ArchTag>
class BlockQuant
{
public:
    using ElementInput = float;
    using LayoutInput = layout::RowMajor;
    using ElementDequantScale = float;
    using LayoutDequantScale = layout::VectorLayout;
    using ElementOutput = int8_t;
    using LayoutOutput = layout::RowMajor;

    using InputType = GemmType<ElementInput, LayoutInput>;
    using DequantScaleType = GemmType<ElementDequantScale, LayoutDequantScale>;
    using OutputType = GemmType<ElementOutput, LayoutOutput>;

    using EpilogueTileSwizzle = Epilogue::Tile::EpilogueHorizontalTileSwizzle;

    struct Params {
        __gm__ ElementInput *ptrInput{nullptr};
        LayoutInput layoutInput;
        __gm__ ElementDequantScale *ptrDequantScale{nullptr};
        LayoutDequantScale layoutDequantScale;
        __gm__ ElementOutput *ptrOutput{nullptr};
        LayoutOutput layoutOutput;
        uint32_t tileRow;
        uint32_t tileColumn;

        CATLASS_DEVICE
        Params() {};

        CATLASS_DEVICE
        Params(__gm__ ElementInput *ptrInput_, LayoutInput const &layoutInput_,
               __gm__ ElementDequantScale *ptrQuantScale_, LayoutDequantScale const &layoutQuantScale_,
               __gm__ ElementOutput *ptrOutput_, LayoutOutput const layoutOutput_, const uint32_t tileRow_,
               const uint32_t tileColumn_)
            : ptrInput(ptrInput_),
              layoutInput(layoutInput_),
              ptrDequantScale(ptrQuantScale_),
              layoutDequantScale(layoutQuantScale_),
              ptrOutput(ptrOutput_),
              layoutOutput(layoutOutput_),
              tileRow(tileRow_),
              tileColumn(tileColumn_)
        {}
    };

    CATLASS_DEVICE
    BlockQuant(Arch::Resource<ArchTag> const &resource, Params const &params_) : params(params_)
    {
        int64_t ubOffset = 0;
        tileRow = params_.tileRow;
        tileColumn = params_.tileColumn;
        tileCount = tileRow * tileColumn;
        halfTileColumn = tileColumn / 2;
        halfTileCount = tileRow * halfTileColumn;

        ubInput = resource.ubBuf.template GetBufferByByte<ElementInput>(ubOffset);
        ubOffset += tileCount * sizeof(ElementInput);
        ubDequantScale = resource.ubBuf.template GetBufferByByte<ElementDequantScale>(ubOffset);
        ubOffset += CEIL_UP(tileRow * sizeof(ElementDequantScale));
        ubOutput = resource.ubBuf.template GetBufferByByte<ElementOutput>(ubOffset);
        ubOffset += tileCount * sizeof(ElementOutput);

        ubAbs = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += tileCount * sizeof(float);
        ubMax = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += halfTileCount * sizeof(float);
        ubReduceMax = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += CEIL_UP(tileRow * sizeof(float));
        ubQuantScale = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += CEIL_UP(tileRow * sizeof(float));
        ubInputTmp = ubAbs;
        ubQuantF32 = ubAbs;
        ubQuantS32 = ubAbs.ReinterpretCast<int32_t>();
        ubQuantF16 = ubAbs.ReinterpretCast<half>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
    }

    CATLASS_DEVICE
    ~BlockQuant()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
    }

    CATLASS_DEVICE
    void operator()(MatrixCoord const &blockShape, MatrixCoord const &blockCoord, MatrixCoord const &actualBlockShape)
    {
        MatrixCoord blockOffset = blockCoord * blockShape;

        AscendC::GlobalTensor<ElementInput> gmInput;
        gmInput.SetGlobalBuffer(params.ptrInput);
        AscendC::GlobalTensor<ElementDequantScale> gmDequantScale;
        gmDequantScale.SetGlobalBuffer(params.ptrDequantScale);
        AscendC::GlobalTensor<ElementOutput> gmOutput;
        gmOutput.SetGlobalBuffer(params.ptrOutput);

        auto ubTileStride = MakeCoord(static_cast<int64_t>(tileColumn), 1L);
        auto ubHalfTileStride = MakeCoord(static_cast<int64_t>(halfTileColumn), 1L);
        auto tileShape = MakeCoord(tileRow, tileColumn);
        EpilogueTileSwizzle epilogueTileSwizzle(actualBlockShape, tileShape);
        uint32_t tileLoops = epilogueTileSwizzle.GetLoops();
        uint32_t subblockIdx = AscendC::GetSubBlockIdx();
        uint32_t subblockNum = AscendC::GetSubBlockNum();
        for (uint32_t loopIdx = subblockIdx; loopIdx < tileLoops; loopIdx += subblockNum) {
            auto tileCoord = epilogueTileSwizzle.GetTileCoord(loopIdx);
            auto actualTileShape = epilogueTileSwizzle.GetActualTileShape(tileCoord);
            auto tileOffsetInBlock = tileCoord * tileShape;
            auto tileOffset = blockOffset + tileOffsetInBlock;

            auto gmTileInput = gmInput[params.layoutInput.GetOffset(tileOffset)];
            auto layoutGmTileInput = params.layoutInput.GetTileLayout(actualTileShape);

            layout::RowMajor layoutUbInput{actualTileShape, ubTileStride};

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            copyGmToUbInput(ubInput, gmTileInput, layoutUbInput, layoutGmTileInput);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::Abs(ubAbs, ubInput, tileCount);
            AscendC::PipeBarrier<PIPE_V>();

            for (uint32_t rowIdx = 0; rowIdx < tileRow; ++rowIdx) {
                AscendC::Max(ubMax[rowIdx * halfTileColumn], ubAbs[rowIdx * tileColumn],
                             ubAbs[rowIdx * tileColumn + halfTileColumn], halfTileColumn);
            }

            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(ubInputTmp, ubInput, 127.f, tileCount);

            constexpr uint32_t elementPerBlk = BYTE_PER_BLK / sizeof(float);
            constexpr int32_t mask = 64;

            AscendC::BinaryRepeatParams maxParams;
            maxParams.dstBlkStride = halfTileColumn / elementPerBlk;
            maxParams.src0BlkStride = halfTileColumn / elementPerBlk;
            maxParams.src1BlkStride = halfTileColumn / elementPerBlk;
            maxParams.dstRepStride = 1;
            maxParams.src0RepStride = 1;
            maxParams.src1RepStride = 1;
            constexpr uint32_t colNumPerCompute = BYTE_PER_VECTOR_FRACTAL / sizeof(float);
            uint32_t reduceWidth = halfTileColumn;
            while (reduceWidth > (BLK_NUM_PER_VECTOR_FRACTAL * BYTE_PER_BLK / sizeof(float))) {
                reduceWidth >>= 1;
                AscendC::Max(ubMax, ubMax, ubMax[reduceWidth], mask, reduceWidth / elementPerBlk, maxParams);
                AscendC::PipeBarrier<PIPE_V>();
            }

            AscendC::WholeReduceMax(ubReduceMax, ubMax, mask, tileRow, 1, 1, halfTileColumn / elementPerBlk,
                                    AscendC::ReduceOrder::ORDER_ONLY_VALUE);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::Muls(ubDequantScale, ubReduceMax, 1.0f / 127.0f, tileRow);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);

            auto dequantScaleTileOffset = tileOffset.template GetCoordByAxis<0>();
            auto dequantScaleTileShape = actualTileShape.template GetCoordByAxis<0>();

            auto gmTileDequantScale = gmDequantScale[params.layoutDequantScale.GetOffset(dequantScaleTileOffset)];
            auto layoutGmTileDequantScale = params.layoutDequantScale.GetTileLayout(dequantScaleTileShape);

            auto layoutUbDequantScale =
                LayoutDequantScale::template MakeLayoutInUb<ElementDequantScale>(dequantScaleTileShape);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            copyUbToGmDequantScale(gmTileDequantScale, ubDequantScale, layoutGmTileDequantScale, layoutUbDequantScale);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);

            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            for (uint32_t rowIdx = 0; rowIdx < tileRow; ++rowIdx) {
                AscendC::Muls(ubQuantF32[rowIdx * tileColumn], ubInputTmp[rowIdx * tileColumn],
                              1.f / ubReduceMax.GetValue(rowIdx), tileColumn);
            }

            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(ubQuantS32, ubQuantF32, AscendC::RoundMode::CAST_RINT, tileCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetDeqScale(static_cast<half>(1.0));
            AscendC::Cast(ubQuantF16, ubQuantS32, AscendC::RoundMode::CAST_RINT, tileCount);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
            AscendC::Cast(ubOutput, ubQuantF16, AscendC::RoundMode::CAST_RINT, tileCount);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1);

            auto gmTileOutput = gmOutput[params.layoutOutput.GetOffset(tileOffset)];
            auto layoutGmTileOutput = params.layoutOutput.GetTileLayout(actualTileShape);

            LayoutOutput layoutUbOutput{actualTileShape, ubTileStride};

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1);
            copyUbToGmOutput(gmTileOutput, ubOutput, layoutGmTileOutput, layoutUbOutput);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
        }
    }

private:
    Params params;
    uint32_t tileRow;
    uint32_t tileColumn;
    uint32_t tileCount;
    uint32_t halfTileColumn;
    uint32_t halfTileCount;

    AscendC::LocalTensor<ElementInput> ubInput;
    AscendC::LocalTensor<ElementDequantScale> ubDequantScale;
    AscendC::LocalTensor<ElementOutput> ubOutput;

    AscendC::LocalTensor<float> ubAbs;
    AscendC::LocalTensor<float> ubMax;
    AscendC::LocalTensor<float> ubReduceMax;
    AscendC::LocalTensor<float> ubQuantScale;
    AscendC::LocalTensor<float> ubQuantScaleBrcb;
    AscendC::LocalTensor<float> ubInputTmp;
    AscendC::LocalTensor<float> ubQuantF32;
    AscendC::LocalTensor<int32_t> ubQuantS32;
    AscendC::LocalTensor<half> ubQuantF16;

    Epilogue::Tile::CopyGm2Ub<ArchTag, InputType> copyGmToUbInput;
    Epilogue::Tile::CopyUb2Gm<ArchTag, DequantScaleType> copyUbToGmDequantScale;
    Epilogue::Tile::CopyUb2Gm<ArchTag, OutputType> copyUbToGmOutput;
};

__aicore__ inline static void EncreaseSyncFlag(__gm__ uint8_t *flagAddr, uint8_t idx)
{
    // flag++, like set flag
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(flagAddr + idx * SOFT_SYNC_SPACE_SIZE);
    __asm__ __volatile__("");
    AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        global);
    __asm__ __volatile__("");
    uint8_t value = global.GetValue(0);
    global.SetValue(0, value + 1);
    __asm__ __volatile__("");
    AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        global);
    __asm__ __volatile__("");
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline static void CheckSyncFlag(__gm__ uint8_t *flagAddr, uint8_t idx, uint32_t target)
{
    //  check flag, like wait flag
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(flagAddr + idx * SOFT_SYNC_SPACE_SIZE);
    while (true) {
        __asm__ __volatile__("");
        AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(global);
        __asm__ __volatile__("");
        uint8_t value = global.GetValue(0);
        if (value >= target) {
            __asm__ __volatile__("");
            AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                              AscendC::DcciDst::CACHELINE_OUT>(global);
            __asm__ __volatile__("");
            break;
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline static void CalQuantRow(const uint32_t column, uint32_t &row)
{
    row = QUANT_SPACE_FACTOR / column;
    row = row < MAX_QUANT_ROW_ONCE ? row : MAX_QUANT_ROW_ONCE;
}

template <typename XType_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, uint32_t WORKSPACE_STAGES_,
          class ElementGroupList_>
class GroupedMatmulSliceMPerTokenDequantSwigluQuantMultiStageWorkspace
{
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using BlockEpilogue = BlockEpilogue_;
    using ElementScale = typename BlockEpilogue::ElementScale;
    using LayoutScale = typename BlockEpilogue::LayoutScale;
    using ElementPerTokenScale = typename BlockEpilogue::ElementPerTokenScale;
    using LayoutPerTokenScale = typename BlockEpilogue::LayoutPerTokenScale;
    using ElementD = typename BlockEpilogue::ElementD;
    using LayoutD = typename BlockEpilogue::LayoutD;
    using EpilogueParams = typename BlockEpilogue::Params;

    using ElementDequantScale = typename BlockQuant<ArchTag>::ElementDequantScale;
    using LayoutDequantScale = typename BlockQuant<ArchTag>::LayoutDequantScale;
    using ElementOutput = typename BlockQuant<ArchTag>::ElementOutput;
    using LayoutOutput = typename BlockQuant<ArchTag>::LayoutOutput;

    using BlockScheduler = BlockScheduler_;
    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;
    using ElementGroupList = ElementGroupList_;

    using XType = XType_;

    // Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        uint32_t problemCount;
        __gm__ ElementGroupList_ *ptrGroupList;
        __gm__ ElementA *ptrA;
        LayoutA layoutA;
        __gm__ ElementB *ptrB;
        LayoutB layoutB;
        __gm__ ElementScale *ptrScale;
        LayoutScale layoutScale;
        __gm__ ElementPerTokenScale *ptrPerTokenScale;
        LayoutPerTokenScale layoutPerTokenScale;
        __gm__ ElementOutput *ptrOutput;
        LayoutOutput layoutOutput;
        __gm__ ElementDequantScale *ptrDequantScale;
        LayoutDequantScale layoutDequantScale;
        GM_ADDR ptrWorkspace;
        GM_ADDR gmX;
        GM_ADDR debugGm;
        GM_ADDR gmexpertIds;

        GM_ADDR gmExpandIdx;
        GM_ADDR gmEpSendCount;
        GM_ADDR gmResvered;
        GM_ADDR gmOutputRecvCount;

        uint32_t epRankSize;
        uint32_t epRankId;
        uint32_t moeExpertNum;
        uint32_t moeExpertNumPerRank;
        uint32_t sharedExpertNum;
        uint32_t sharedExpertRankNum;
        uint32_t quantMode;
        uint32_t globalBs;
        uint32_t bs;
        uint32_t topK;
        uint32_t tokenLen;
        // Methods
        CATLASS_DEVICE
        Params() {}

        CATLASS_DEVICE
        Params(GemmCoord problemShape_, uint32_t problemCount_, GM_ADDR ptrGroupList_, GM_ADDR ptrA_,
               LayoutA const &layoutA_, GM_ADDR ptrB_, LayoutB const &layoutB_, GM_ADDR ptrScale_,
               LayoutScale const &layoutScale_, GM_ADDR ptrPerTokenScale_,
               LayoutPerTokenScale const &layoutPerTokenScale_, GM_ADDR ptrOutput_, LayoutOutput const &layoutOutput_,
               GM_ADDR ptrDequantScale_, LayoutDequantScale const &layoutDequantScale_, GM_ADDR ptrWorkspace_,
               GM_ADDR gmX_, GM_ADDR debugGm_, GM_ADDR gmexpertIds_, GM_ADDR gmExpandIdx_, GM_ADDR gmEpSendCount_,
               GM_ADDR gmResvered_, GM_ADDR gmOutputRecvCount_, uint32_t epRankSize_, uint32_t epRankId_,
               uint32_t moeExpertNum_, uint32_t moeExpertNumPerRank_, uint32_t sharedExpertNum_,
               uint32_t sharedExpertRankNum_, uint32_t quantMode_, uint32_t globalBs_, uint32_t bs_, uint32_t topK_,
               uint32_t h)
            : problemShape(problemShape_),
              problemCount(problemCount_),
              ptrGroupList(reinterpret_cast<__gm__ ElementGroupList *>(ptrGroupList_)),
              ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)),
              layoutA(layoutA_),
              ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)),
              layoutB(layoutB_),
              ptrScale(reinterpret_cast<__gm__ ElementScale *>(ptrScale_)),
              layoutScale(layoutScale_),
              ptrPerTokenScale(reinterpret_cast<__gm__ ElementPerTokenScale *>(ptrPerTokenScale_)),
              layoutPerTokenScale(layoutPerTokenScale_),
              ptrOutput(reinterpret_cast<__gm__ ElementOutput *>(ptrOutput_)),
              layoutOutput(layoutOutput_),
              ptrDequantScale(reinterpret_cast<__gm__ ElementDequantScale *>(ptrDequantScale_)),
              layoutDequantScale(layoutDequantScale_),
              ptrWorkspace(ptrWorkspace_),
              gmX(gmX_),
              debugGm(debugGm_),
              gmexpertIds(gmexpertIds_),
              gmExpandIdx(gmExpandIdx_),
              gmEpSendCount(gmEpSendCount_),
              gmOutputRecvCount(gmOutputRecvCount_),
              gmResvered(gmResvered_),
              epRankSize(epRankSize_),
              epRankId(epRankId_),
              moeExpertNum(moeExpertNum_),
              moeExpertNumPerRank(moeExpertNumPerRank_),
              sharedExpertNum(sharedExpertNum_),
              sharedExpertRankNum(sharedExpertRankNum_),
              quantMode(quantMode_),
              globalBs(globalBs_),
              bs(bs_),
              topK(topK_),
              tokenLen(h)
        {}
    };

    // Methods
    CATLASS_DEVICE
    GroupedMatmulSliceMPerTokenDequantSwigluQuantMultiStageWorkspace() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        aicIdx = AscendC::GetBlockIdx();
        subBlockNum = AscendC::GetSubBlockNum();
        aiCoreGroupNum = AscendC::GetBlockNum();
        aicNum = aiCoreGroupNum;
        aivNum = aiCoreGroupNum * SUB_AIV_NUM;
        aicStateGlobalCoreIdx = aivNum + aicIdx;
        moeExpertNumPerRank = params.moeExpertNumPerRank;
        isShareExpert = (params.epRankId < params.sharedExpertRankNum);
        localExpertNum = isShareExpert ? 1 : moeExpertNumPerRank;
        // when localExpertNum=1, all cores send token and recv token in sequence
        recvCoreNum = aivNum;
        // when localExpertNum>1, half of cores send token and another half recv token in parallel
        if (localExpertNum > 1) {
            recvCoreNum = aiCoreGroupNum;
        }
        uint32_t coreNumPerGroup = recvCoreNum / localExpertNum;
        winContext_ = (__gm__ HcclOpResParam *)AscendC::GetHcclContext<AscendC::HCCL_GROUP_ID_0>();

        // state of cv flag
        statusDataSpaceGm = (GM_ADDR)(winContext_->localWindowsExp);
        AscendC::GlobalTensor<int32_t> selfDataStatusTensor;
        selfDataStatusTensor.SetGlobalBuffer((__gm__ int32_t *)(statusDataSpaceGm + STATE_WIN_OFFSET));
        __asm__ __volatile__("");
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(
            selfDataStatusTensor[aicStateGlobalCoreIdx * UB_ALIGN]);
        __asm__ __volatile__("");
        cvDataState = selfDataStatusTensor(aicStateGlobalCoreIdx * UB_ALIGN);
        if (cvDataState == 0) {
            selfDataStatusTensor(aicStateGlobalCoreIdx * UB_ALIGN) = 1;
            vToCFlag = V_TO_C_FLAG_1;
        } else {
            selfDataStatusTensor(aicStateGlobalCoreIdx * UB_ALIGN) = 0;
            vToCFlag = V_TO_C_FLAG_2;
        }

        BlockScheduler blockScheduler;
        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(params.ptrB);

        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);

        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetB = 0;

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrWorkspace));
        auto layoutC = layout::RowMajor{L1TileShape::M * aicNum * WORKSPACE_STAGES, L1TileShape::N};

        uint32_t stageId = 0;
        uint32_t stageUsed = 0;
        uint32_t startCoreIdx = 0;
        AscendC::GlobalTensor<int32_t> groupTokenNumStateTensor;
        aicSetFunc1 = {statusDataSpaceGm + SOFT_SYNC_OFFSET,
                       static_cast<uint8_t>(aicNum + AscendC::GetBlockIdx())};  // AIV wait for flags in latter part
        uint32_t target = 1;
        for (uint32_t groupIdx = 0; groupIdx < localExpertNum; ++groupIdx) {
            groupTokenNumStateTensor.SetGlobalBuffer((__gm__ int32_t *)(statusDataSpaceGm + GROUP_TOKEN_NUM_OFFSET) +
                                                     groupIdx * GROUP_INFO_SIZE);
            // wait AIV recv needed tokens
            while (true) {
                __asm__ __volatile__("");
                AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                                  AscendC::DcciDst::CACHELINE_OUT>(groupTokenNumStateTensor);
                __asm__ __volatile__("");
                if (groupTokenNumStateTensor.GetValue(0) == coreNumPerGroup * vToCFlag) {
                    break;
                }
            }

            uint32_t currentM = groupTokenNumStateTensor.GetValue(GROUP_TOKEN_COUNT);
            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};

            LayoutA layoutA = params.layoutA.GetTileLayout(inGroupProblemShape.GetCoordMK());
            LayoutB layoutB = params.layoutB;

            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();

            // Determine the starting loopIdx of the current core under the current groupIdx
            uint32_t startLoopIdx = ((aicIdx < startCoreIdx) ? (aicIdx + aicNum) : aicIdx) - startCoreIdx;
            // Loop through the matmul of each groupIdx
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += aicNum) {
                // Compute block location
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);

                Callback callbackBeforeFixpipe{};
                if (stageUsed == WORKSPACE_STAGES) {
                    aicWaitFunc1 = {statusDataSpaceGm + SOFT_SYNC_OFFSET, static_cast<uint8_t>(AscendC::GetBlockIdx()),
                                    target};  // AIC wait for flags in former part
                    target += 1;
                    callbackBeforeFixpipe = MakeCallback(&aicWaitFunc1);
                } else {
                    ++stageUsed;
                }
                Callback callbackAfterFixpipe = MakeCallback(&aicSetFunc1);

                // Compute initial location in logical coordinates
                MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
                MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
                MatrixCoord offsetC{(stageId * aicNum + aicIdx) * L1TileShape::M, 0};
                int64_t gmOffsetA = layoutA.GetOffset(offsetA);
                int64_t gmOffsetB = layoutB.GetOffset(offsetB);
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);

                // Compute block-scoped matrix multiply-add
                if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                    blockMmad(gmA[gmGroupOffsetA + gmOffsetA], layoutA, gmB[gmGroupOffsetB + gmOffsetB], layoutB,
                              gmC[gmOffsetC], layoutC, actualBlockShape, callbackBeforeFixpipe, callbackAfterFixpipe);
                } else {
                    callbackBeforeFixpipe();
                    blockMmad(gmA[gmGroupOffsetA + gmOffsetA], layoutA, gmB[gmGroupOffsetB + gmOffsetB], layoutB,
                              gmC[gmOffsetC], layoutC, actualBlockShape);
                    callbackAfterFixpipe();
                }

                stageId = (stageId + 1 < WORKSPACE_STAGES) ? (stageId + 1) : 0;
            }

            gmGroupOffsetA += inGroupProblemShape.m() * inGroupProblemShape.k();
            gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % aicNum;
        }

        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
        }

        while (stageUsed > 0) {
            uint32_t aivComputeStageId =
                (stageId >= stageUsed) ? (stageId - stageUsed) : (stageId + WORKSPACE_STAGES - stageUsed);
            target += 1;
            --stageUsed;
        }
        AscendC::SyncAll<false>();
    }

    CATLASS_DEVICE
    void CalExpandxIdx(int32_t dstExpertId, uint32_t tokenIndex, int32_t &curExpertCnt, int64_t ubOffset)
    {
        // calculate index in remote
        int64_t subUbOffset = ubOffset;
        AscendC::LocalTensor<int32_t> dstExpIdTensor_ = (resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset));
        subUbOffset += LOOP_TMP_SIZE;
        AscendC::LocalTensor<int32_t> subExpIdTensor_ = (resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset));
        subUbOffset += LOOP_TMP_SIZE;
        AscendC::LocalTensor<float> workLocalTensor_ = (resource.ubBuf.template GetBufferByByte<float>(ubOffset));
        subUbOffset += LOOP_TMP_SIZE;
        AscendC::Duplicate<int32_t>(dstExpIdTensor_, dstExpertId, tokenIndex);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(subExpIdTensor_, expertIdsTensor_, dstExpIdTensor_, tokenIndex);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<float> tmpFp32 = subExpIdTensor_.ReinterpretCast<float>();
        AscendC::LocalTensor<float> tmpoutFp32 = dstExpIdTensor_.ReinterpretCast<float>();
        AscendC::Abs(tmpoutFp32, tmpFp32, tokenIndex);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mins(subExpIdTensor_, dstExpIdTensor_, 1, tokenIndex);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum<float>(tmpoutFp32, tmpFp32, workLocalTensor_, tokenIndex);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        int32_t curOtherExpertCnt = dstExpIdTensor_(0);
        if (tokenIndex > curOtherExpertCnt) {
            curExpertCnt = tokenIndex - curOtherExpertCnt;
        }
    }

    CATLASS_DEVICE
    void CalAndSendTokenCount()
    {
        uint32_t totalExpertNum = sharedExpertRankNum + moeExpertNum;
        uint32_t sendCountExpertNum = totalExpertNum / sendCoreNum;
        uint32_t remainderRankNum = totalExpertNum % sendCoreNum;
        uint32_t startExpertId = sendCountExpertNum * sendCoreIdx;
        if (sendCoreIdx < remainderRankNum) {
            sendCountExpertNum += 1;
            startExpertId += sendCoreIdx;
        } else {
            startExpertId += remainderRankNum;
        }
        uint32_t endExpertId = startExpertId + sendCountExpertNum;
        if (startExpertId >= totalExpertNum) {
            return;
        }

        AscendC::LocalTensor<int32_t> statusTensor_ = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset);
        ubOffset += CEIL_UP(CEIL(expertCntUp, INT32_COUNT_PER_BLOCK) * INT32_COUNT_PER_BLOCK * UB_BLOCK_SIZE);
        AscendC::Duplicate(statusTensor_, (int32_t)0,
                           expertCntUp * INT32_COUNT_PER_BLOCK);
        if (state == 0) {
            // set the first number of every 8 numbers as 0x3F800000(float 1.0)
            uint64_t mask[2] = {0x101010101010101, 0};
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate<int32_t>(statusTensor_, 0x3F800000, mask, CEIL(expertCntUp, 8), 1, 8);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);

        if (!isShareExpert) {
            for (uint32_t curSatatusExpId = 0; curSatatusExpId < sharedExpertRankNum; ++curSatatusExpId) {
                int32_t curExpertCnt = (curSatatusExpId + 1 + epRankId) * axisBS / sharedExpertRankNum -
                                       (curSatatusExpId + epRankId) * axisBS / sharedExpertRankNum;
                statusTensor_((curSatatusExpId)*INT32_COUNT_PER_BLOCK + 1) = curExpertCnt;
            }
        }

        for (uint32_t curExpertId = startExpertId; curExpertId < endExpertId; ++curExpertId) {
            if (curExpertId < sharedExpertRankNum) {
                continue;
            }
            int32_t curExpertCnt = 0;
            int32_t dstExpertId = curExpertId - sharedExpertRankNum;
            CalExpandxIdx(dstExpertId, expertIdsCnt, curExpertCnt, ubOffset);
            int32_t cntPosIndex = curExpertId * INT32_COUNT_PER_BLOCK + 1;
            statusTensor_(cntPosIndex) = curExpertCnt;
        }

        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);

        AscendC::GlobalTensor<int32_t> rankGMTensor;
        uint32_t offset = stateOffset * epRankId;
        for (uint32_t rankIndex = startExpertId; rankIndex < endExpertId; ++rankIndex) {
            uint32_t dstRankId = rankIndex;
            if (moeExpertNumPerRank > 1 && (rankIndex >= sharedExpertRankNum)) {
                dstRankId = ((rankIndex - sharedExpertRankNum) / moeExpertNumPerRank + sharedExpertRankNum);
                offset =
                    (epRankId + (rankIndex - sharedExpertRankNum) % moeExpertNumPerRank * epRankSize) * stateOffset;
            }
            GM_ADDR rankGM = (__gm__ uint8_t *)(GET_WIND_STATE_ADDR_BY_RANK_ID(dstRankId) + offset);
            rankGMTensor.SetGlobalBuffer((__gm__ int32_t *)rankGM);
            AscendC::DataCopy<int32_t>(rankGMTensor, statusTensor_[rankIndex * 8], 8UL);
        }
    }

    CATLASS_DEVICE
    void QuantToken(AscendC::LocalTensor<XType> &xInTensor, AscendC::LocalTensor<int8_t> &yInt8Tensor, int64_t ubOffset)
    {
        int64_t subUbOffset = ubOffset;
        AscendC::LocalTensor<float> xFp32TmpTensor = resource.ubBuf.template GetBufferByByte<float>(subUbOffset);
        subUbOffset += CEIL_UP(tokenLength * sizeof(float));
        AscendC::LocalTensor<float> xFp32AbsTensor = resource.ubBuf.template GetBufferByByte<float>(subUbOffset);
        subUbOffset += CEIL_UP(tokenLength * sizeof(float));
        AscendC::LocalTensor<float> xRowMaxTensor = resource.ubBuf.template GetBufferByByte<float>(subUbOffset);
        subUbOffset += CEIL_UP(UB_BLOCK_SIZE);
        AscendC::LocalTensor<int32_t> ytmpInt32Tensor = xFp32TmpTensor.template ReinterpretCast<int32_t>();
        AscendC::LocalTensor<half> yHalfTensor = xFp32TmpTensor.template ReinterpretCast<half>();
        AscendC::LocalTensor<float> yFp32Tensor = yInt8Tensor.template ReinterpretCast<float>();
        AscendC::LocalTensor<int32_t> yInt32Tensor = yInt8Tensor.template ReinterpretCast<int32_t>();

        AscendC::Cast(xFp32TmpTensor, xInTensor, AscendC::RoundMode::CAST_NONE, tokenLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Abs(xFp32AbsTensor, xFp32TmpTensor, tokenLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceMax(xRowMaxTensor, xFp32AbsTensor, xFp32AbsTensor, tokenLength, false);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        float dynamicQuantScale = float(127.0) / xRowMaxTensor.GetValue(0);
        yFp32Tensor.SetValue(tokenLength / sizeof(float), float(1.0) / dynamicQuantScale);
        yInt32Tensor.SetValue(tokenLength / sizeof(int32_t) + 1, tokenFlag);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);

        AscendC::Muls(xFp32TmpTensor, xFp32TmpTensor, dynamicQuantScale, tokenLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(ytmpInt32Tensor, xFp32TmpTensor, AscendC::RoundMode::CAST_RINT, tokenLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(yHalfTensor, ytmpInt32Tensor, AscendC::RoundMode::CAST_ROUND, tokenLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(yInt8Tensor, yHalfTensor, AscendC::RoundMode::CAST_TRUNC, tokenLength);
    }

    CATLASS_DEVICE
    void SendToShareExprt(GM_ADDR gmX, GM_ADDR gmX1, GM_ADDR gmX1Scale)
    {
        uint32_t newAivId = sendCoreIdx - sendToMoeAivNum;
        uint32_t sendTokenNum = axisBS / sendToShareAivNum;
        uint32_t remainderTokenNum = axisBS % sendToShareAivNum;
        uint32_t startTokenId = sendTokenNum * newAivId;
        if (newAivId < remainderTokenNum) {
            sendTokenNum += 1;
            startTokenId += newAivId;
        } else {
            startTokenId += remainderTokenNum;
        }
        uint32_t endTokenId = startTokenId + sendTokenNum;
        if (startTokenId >= axisBS) {
            return;
        }

        AscendC::LocalTensor<XType> xInTensor[BUFFER_NUM];
        AscendC::LocalTensor<int8_t> yInt8Tensor[BUFFER_NUM];
        AscendC::LocalTensor<float> yFp32Tensor[BUFFER_NUM];

        AscendC::GlobalTensor<XType> srcWinGMTensor;
        srcWinGMTensor.SetGlobalBuffer((__gm__ XType *)gmX);

        xInTensor[0] = resource.ubBuf.template GetBufferByByte<XType>(ubOffset);
        ubOffset += CEIL_UP(tokenLength * sizeof(XType));
        xInTensor[1] = resource.ubBuf.template GetBufferByByte<XType>(ubOffset);
        ubOffset += CEIL_UP(tokenLength * sizeof(XType));
        yInt8Tensor[0] = resource.ubBuf.template GetBufferByByte<int8_t>(ubOffset);
        yFp32Tensor[0] = yInt8Tensor[0].template ReinterpretCast<float>();
        ubOffset += CEIL_UP(axisHCommu * sizeof(int8_t));
        yInt8Tensor[1] = resource.ubBuf.template GetBufferByByte<int8_t>(ubOffset);
        yFp32Tensor[1] = yInt8Tensor[1].template ReinterpretCast<float>();
        ubOffset += CEIL_UP(axisHCommu * sizeof(int8_t));
        AscendC::GlobalTensor<int8_t> dstWinGMTensor;
        AscendC::GlobalTensor<int8_t> expandXOutGlobal;
        expandXOutGlobal.SetGlobalBuffer((__gm__ int8_t *)(gmX1));
        AscendC::GlobalTensor<float> dynamicScalesOutGMTensor_;
        dynamicScalesOutGMTensor_.SetGlobalBuffer((__gm__ float *)(gmX1Scale));

        // double buffer
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);

        for (uint32_t tokenIndex = startTokenId; tokenIndex < endTokenId; ++tokenIndex) {
            uint32_t index = (tokenIndex & 1) ? 0 : 1;
            int32_t eventId = (tokenIndex & 1) ? 0 : 1;
            uint32_t temp = (epRankId * axisBS) / sharedExpertRankNum;
            uint32_t moeOnShareRank = CEIL((tokenIndex + 1 + temp) * sharedExpertRankNum, axisBS) - 1 - epRankId;
            uint32_t preCnt = (moeOnShareRank + epRankId) * axisBS / sharedExpertRankNum -
                                epRankId * axisBS / sharedExpertRankNum;
            dstWinGMTensor.SetGlobalBuffer(
                (__gm__ int8_t *)(GET_WIND_ADDR_BY_RANK_ID(moeOnShareRank) + expertPerSizeOnWin * epRankId));

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            AscendC::DataCopy(xInTensor[index], srcWinGMTensor[tokenIndex * tokenLength], tokenLength);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
            QuantToken(xInTensor[index], yInt8Tensor[index], ubOffset);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
            if (isShareExpert) {
                AscendC::DataCopyExtParams dataCopyParamsFloat = {1U, sizeof(float), 0U, 0U, 0U};
                AscendC::DataCopy(expandXOutGlobal[tokenIndex * tokenLength], yInt8Tensor[index], tokenLength);
                AscendC::PipeBarrier<PIPE_MTE3>();
                AscendC::DataCopyPad(dynamicScalesOutGMTensor_[tokenIndex],
                                        yFp32Tensor[index][tokenLength / sizeof(float)], dataCopyParamsFloat);
            } else {
                AscendC::DataCopy(dstWinGMTensor[(tokenIndex - preCnt) * axisHCommu], yInt8Tensor[index],
                                    tokenLength);
                AscendC::PipeBarrier<PIPE_MTE3>();
                AscendC::DataCopy(dstWinGMTensor[(tokenIndex - preCnt) * axisHCommu + tokenLength],
                                    yInt8Tensor[index][tokenLength], scaleParamPad);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventId);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
    }

    CATLASS_DEVICE
    void SendToMoeExprt(GM_ADDR gmX, GM_ADDR gmExpandIdx)
    {
        uint32_t sendTokenNum = expertIdsCnt / sendToMoeAivNum;
        uint32_t remainderTokenNum = expertIdsCnt % sendToMoeAivNum;
        uint32_t startTokenId = sendTokenNum * sendCoreIdx;
        if (sendCoreIdx < remainderTokenNum) {
            sendTokenNum += 1;
            startTokenId += sendCoreIdx;
        } else {
            startTokenId += remainderTokenNum;
        }
        uint32_t endTokenId = startTokenId + sendTokenNum;
        if (startTokenId >= expertIdsCnt) {
            return;
        }
        AscendC::LocalTensor<int32_t> expertCountTensor = (resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset));
        ubOffset += CEIL_UP(expertIdsCnt * sizeof(int32_t));
        AscendC::Duplicate(expertCountTensor, (int32_t)0, expertIdsCnt);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(1);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(1);

        AscendC::LocalTensor<XType> xInTensor[BUFFER_NUM];
        AscendC::LocalTensor<int8_t> yInt8Tensor[BUFFER_NUM];
        AscendC::LocalTensor<float> yFp32Tensor[BUFFER_NUM];

        AscendC::GlobalTensor<XType> srcWinGMTensor;
        srcWinGMTensor.SetGlobalBuffer((__gm__ XType *)gmX);

        xInTensor[0] = resource.ubBuf.template GetBufferByByte<XType>(ubOffset);
        ubOffset += CEIL_UP(tokenLength * sizeof(XType));
        xInTensor[1] = resource.ubBuf.template GetBufferByByte<XType>(ubOffset);
        ubOffset += CEIL_UP(tokenLength * sizeof(XType));
        yInt8Tensor[0] = resource.ubBuf.template GetBufferByByte<int8_t>(ubOffset);
        ubOffset += CEIL_UP(axisHCommu * sizeof(int8_t));
        yInt8Tensor[1] = resource.ubBuf.template GetBufferByByte<int8_t>(ubOffset);
        ubOffset += CEIL_UP(axisHCommu * sizeof(int8_t));
        AscendC::GlobalTensor<int8_t> dstWinGMTensor;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
        uint32_t sendValidTokenIndex = 0;
        for (uint32_t sendGroupIndex = 0; sendGroupIndex < moeExpertNumPerRank; ++sendGroupIndex) {
            for (uint32_t tokenIndex = startTokenId; tokenIndex < endTokenId; ++tokenIndex) {
                int32_t dstExpertId = expertIdsTensor_(tokenIndex);
                if (dstExpertId < 0) {
                    continue;
                }
                // Send to preferentically to the specicied expert
                if ((dstExpertId % moeExpertNumPerRank) != sendGroupIndex) {
                    continue;
                }
                uint32_t index = (sendValidTokenIndex & 1) ? 0 : 1;
                int32_t eventId = (sendValidTokenIndex & 1) ? 0 : 1;
                sendValidTokenIndex += 1;
                int32_t curExpertCnt = 0;
                CalExpandxIdx(dstExpertId, tokenIndex, curExpertCnt, ubOffset);
                expertCountTensor(tokenIndex - startTokenId) = curExpertCnt;
                uint32_t tempRankId = dstExpertId / moeExpertNumPerRank + sharedExpertRankNum;
                GM_ADDR rankGM = (__gm__ uint8_t *)(GET_WIND_ADDR_BY_RANK_ID(tempRankId) +
                                                    (expertPerSizeOnWin * (epRankId * moeExpertNumPerRank +
                                                                           dstExpertId % moeExpertNumPerRank)) +
                                                    hCommuSize * curExpertCnt);
                dstWinGMTensor.SetGlobalBuffer((__gm__ int8_t *)rankGM);

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                AscendC::DataCopy(xInTensor[index], srcWinGMTensor[tokenIndex / axisK * tokenLength], tokenLength);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
                QuantToken(xInTensor[index], yInt8Tensor[index], ubOffset);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);

                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);

                AscendC::DataCopy(dstWinGMTensor, yInt8Tensor[index], tokenLength);
                AscendC::PipeBarrier<PIPE_MTE3>();
                AscendC::DataCopy(dstWinGMTensor[tokenLength], yInt8Tensor[index][tokenLength], scaleParamPad);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventId);
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);

        AscendC::GlobalTensor<int32_t> expandIdxGMTensor;
        expandIdxGMTensor.SetGlobalBuffer((__gm__ int32_t *)gmExpandIdx + startTokenId);
        AscendC::DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(sendTokenNum * sizeof(uint32_t)), 0U,
                                                         0U, 0U};
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::DataCopyPad(expandIdxGMTensor, expertCountTensor, expertIdsCntParams);
    }

    CATLASS_DEVICE void
    SendCoreFunc(GM_ADDR gmX, GM_ADDR gmExpertIds, GM_ADDR gmX1, GM_ADDR gmX1Scale, GM_ADDR gmExpandIdx)
    {
        ubOffset = 0;
        expertIdsCnt = axisBS * axisK;

        AscendC::GlobalTensor<int32_t> expertIdsGMTensor_;
        expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)gmExpertIds);
        expertIdsTensor_ = (resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset));
        ubOffset += CEIL_UP(expertIdsCnt * sizeof(int32_t));

        AscendC::DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(expertIdsCnt * sizeof(uint32_t)), 0U, 0U,
                                                        0U};
        AscendC::DataCopyPadExtParams<int32_t> copyPadParams{false, 0U, 0U, 0U};
        AscendC::DataCopyPad(expertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, copyPadParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

        CalAndSendTokenCount();
        AscendC::PipeBarrier<PIPE_ALL>();
        if (hasShareExpert) {
            sendToShareAivNum = sendCoreNum / (axisK + 1);
            if (sendToShareAivNum == 0) {
                sendToShareAivNum = 1;
            }
        }
        sendToMoeAivNum = sendCoreNum - sendToShareAivNum;

        AscendC::SetDeqScale((half)1.000000e+00f);
        if (hasShareExpert && sendCoreIdx >= sendToMoeAivNum) {
            SendToShareExprt(gmX, gmX1, gmX1Scale);
        } else {
            SendToMoeExprt(gmX, gmExpandIdx);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE
    void RecvCount(int64_t ubOffset)
    {
        uint32_t recStatusNumPerCore = isShareExpert ? epRankSize : expertCntUp;
        uint32_t startStatusIndex = 0;  // every wait for all token counts

        int64_t subUbOffset = ubOffset;
        AscendC::LocalTensor<int32_t> statusTensor_ = resource.ubBuf.template GetBufferByByte<int32_t>(subUbOffset);
        subUbOffset += CEIL_UP(expertCntUp * UB_BLOCK_SIZE);
        AscendC::LocalTensor<uint32_t> gatherTmpTensor = (resource.ubBuf.template GetBufferByByte<uint32_t>(subUbOffset));
        subUbOffset += CEIL_UP(UB_BLOCK_SIZE);
        AscendC::LocalTensor<float> gatherMaskOutTensor = resource.ubBuf.template GetBufferByByte<float>(subUbOffset);
        subUbOffset += CEIL_UP(expertCntUp * sizeof(float));
        AscendC::LocalTensor<float> statusFp32Tensor_ = statusTensor_.ReinterpretCast<float>();

        AscendC::LocalTensor<float> statusSumOutTensor = resource.ubBuf.template GetBufferByByte<float>(subUbOffset);
        subUbOffset += CEIL_UP(UB_BLOCK_SIZE);
        AscendC::LocalTensor<uint8_t> sumTmpTensor = resource.ubBuf.template GetBufferByByte<uint8_t>(subUbOffset);
        subUbOffset += CEIL_UP(SUM_TMP_TENSOR_SIZE);
        gatherTmpTensor.SetValue(0, 1);

        uint32_t mask = 1;
        uint64_t rsvdCnt = 0;
        AscendC::SumParams sumParams{1, recStatusNumPerCore, recStatusNumPerCore};
        float sumOfFlag = static_cast<float>(-1.0);
        float minTarget = (sumTarget * recStatusNumPerCore) - (float)0.5;
        float maxTarget = (sumTarget * recStatusNumPerCore) + (float)0.5;
        AscendC::DataCopyParams intriParams{static_cast<uint16_t>(recStatusNumPerCore), 1, static_cast<uint16_t>(15),
                                            0};
        AscendC::GlobalTensor<float> windowInstatusFp32Tensor_;
        windowInstatusFp32Tensor_.SetGlobalBuffer((__gm__ float *)GET_WIND_STATE_ADDR_BY_RANK_ID(epRankId));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);

        uint32_t preRecvTokenCount = 0;
        while ((sumOfFlag < minTarget) || (sumOfFlag > maxTarget)) {
            AscendC::DataCopy(statusFp32Tensor_, windowInstatusFp32Tensor_[startStatusIndex * stateOffset / sizeof(float)],
                            intriParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::GatherMask(gatherMaskOutTensor, statusFp32Tensor_, gatherTmpTensor, true, mask,
                                {1, (uint16_t)recStatusNumPerCore, 1, 0}, rsvdCnt);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sum(statusSumOutTensor, gatherMaskOutTensor, sumTmpTensor, sumParams);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            sumOfFlag = statusSumOutTensor.GetValue(0);
        }
    }

    CATLASS_DEVICE
    void GetCumSum(int32_t startRankId, int32_t recvExpertNum, int64_t ubOffset, GM_ADDR gmOutputRecvCount)
    {
        // calculate token index in output tensor
        int64_t subUbOffset = ubOffset;
        uint32_t recStatusNumPerCore = isShareExpert ? epRankSize : expertCntUp;
        AscendC::LocalTensor<int32_t> statusTensor_ = resource.ubBuf.template GetBufferByByte<int32_t>(subUbOffset);
        subUbOffset += CEIL_UP(expertCntUp * UB_BLOCK_SIZE);
        AscendC::LocalTensor<uint32_t> gatherTmpTensor = (resource.ubBuf.template GetBufferByByte<uint32_t>(subUbOffset));
        subUbOffset += CEIL_UP(UB_BLOCK_SIZE);
        AscendC::LocalTensor<float> gatherMaskOutTensor = resource.ubBuf.template GetBufferByByte<float>(subUbOffset);
        subUbOffset += CEIL_UP(expertCntUp * sizeof(float));
        AscendC::LocalTensor<float> statusFp32Tensor_ = statusTensor_.ReinterpretCast<float>();
        if (isShareExpert) {
            for (uint32_t curSatatusExpId = 0; curSatatusExpId < sharedExpertRankNum; ++curSatatusExpId) {
                int32_t curExpertCnt = (curSatatusExpId + 1 + epRankId) * axisBS / sharedExpertRankNum -
                                    (curSatatusExpId + epRankId) * axisBS / sharedExpertRankNum;
                statusTensor_((curSatatusExpId)*INT32_COUNT_PER_BLOCK + 1) = curExpertCnt;
            }
        }

        uint64_t rsvdCnt = 0;
        gatherTmpTensor.SetValue(0, GATHER_SECOND_NUM);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        AscendC::GatherMask(gatherMaskOutTensor, statusFp32Tensor_, gatherTmpTensor, true, GATHER_SECOND_NUM,
                            {1, (uint16_t)recStatusNumPerCore, 1, 0}, rsvdCnt);
        if (isRecvCore && recvCoreIdx == 0) {
            AscendC::GlobalTensor<int32_t> recvCountTensor;
            recvCountTensor.SetGlobalBuffer((__gm__ int32_t *)gmOutputRecvCount);
            AscendC::DataCopyExtParams dataCopyParams = {
                1U, static_cast<uint32_t>(localExpertNum * epRankSize * sizeof(int32_t)), 0U, 0U, 0U};
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::DataCopyPad(recvCountTensor, gatherMaskOutTensor.ReinterpretCast<int32_t>(), dataCopyParams);
        }
        AscendC::LocalTensor<float> workLocalTensor = resource.ubBuf.template GetBufferByByte<float>(subUbOffset);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum<float>(gatherMaskOutTensor, gatherMaskOutTensor, workLocalTensor,
                                (startRankId + 1) <= recvExpertNum ? (startRankId + 1) : recvExpertNum);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
    }

    CATLASS_DEVICE
    void RecvToken(GM_ADDR gmX1, GM_ADDR gmX1Scale, GM_ADDR gmEpSendCount, uint32_t &coreTokenCount, uint32_t startRankId,
                uint32_t endRankId, uint32_t recvRankNumPerCore, int64_t ubOffset)
    {
        int64_t subUbOffset = ubOffset;
        AscendC::LocalTensor<int32_t> statusTensor_ = resource.ubBuf.template GetBufferByByte<int32_t>(subUbOffset);
        subUbOffset += CEIL_UP(expertCntUp * UB_BLOCK_SIZE);
        AscendC::LocalTensor<uint32_t> gatherTmpTensor = (resource.ubBuf.template GetBufferByByte<uint32_t>(subUbOffset));
        subUbOffset += CEIL_UP(UB_BLOCK_SIZE);
        AscendC::LocalTensor<float> gatherMaskOutTensor = resource.ubBuf.template GetBufferByByte<float>(subUbOffset);
        subUbOffset += CEIL_UP(expertCntUp * sizeof(float));
        AscendC::LocalTensor<float> statusFp32Tensor_ = statusTensor_.ReinterpretCast<float>();

        AscendC::DataCopyExtParams dataCopyParamsFloat = {1U, sizeof(float), 0U, 0U, 0U};
        AscendC::LocalTensor<int8_t> xTmpTensor_ = resource.ubBuf.template GetBufferByByte<int8_t>(subUbOffset);
        subUbOffset += CEIL_UP(axisHCommu * sizeof(int8_t));
        AscendC::LocalTensor<float> xOutFp32Tensor_ = xTmpTensor_.template ReinterpretCast<float>();
        AscendC::LocalTensor<int32_t> tmpLocalTensor = resource.ubBuf.template GetBufferByByte<int32_t>(subUbOffset);
        subUbOffset += CEIL_UP(UB_BLOCK_SIZE);
        AscendC::LocalTensor<int32_t> gatherMaskOutCountTensor = (gatherMaskOutTensor.template ReinterpretCast<int32_t>());
        AscendC::GlobalTensor<int8_t> tokGlobal;
        AscendC::GlobalTensor<int32_t> tokGlobalInt32;
        AscendC::GlobalTensor<int8_t> expandXOutGlobal;
        AscendC::GlobalTensor<float> dynamicScalesOutGMTensor_;
        dynamicScalesOutGMTensor_.SetGlobalBuffer((__gm__ float *)(gmX1Scale));
        uint32_t beginIdx = 0;
        for (uint32_t index = startRankId; index < endRankId; index++) {
            uint32_t i = index - startRankId;
            if (i > 0) {
                gatherMaskOutCountTensor.SetValue(
                    i, gatherMaskOutCountTensor.GetValue(i - 1) + gatherMaskOutCountTensor.GetValue(index));
            }
            uint32_t count = statusTensor_.GetValue(index * INT32_COUNT_PER_BLOCK + 1);
            coreTokenCount += count;
            beginIdx = gatherMaskOutCountTensor.GetValue(i) - count;
            if (isShareExpert && index < sharedExpertRankNum) {
                beginIdx += count;
                continue;
            }
            uint32_t winOffset = index;
            if (!isShareExpert && moeExpertNumPerRank > 1) {
                // srcRank: index % epRankSize
                // localExpertId: index / epRankSize
                // Addr: (srcRank * moeExpertNumPerRank + localExpertId) * expertPerSizeOnWin
                winOffset = (index % epRankSize) * moeExpertNumPerRank + index / epRankSize;
            }
            GM_ADDR wAddr = (__gm__ uint8_t *)(GET_WIND_ADDR_BY_RANK_ID(epRankId)) + winOffset * expertPerSizeOnWin;
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            for (uint32_t j = 0; j < count; j++) {
                tokGlobal.SetGlobalBuffer((__gm__ int8_t *)(wAddr + j * hCommuSize));
                tokGlobalInt32.SetGlobalBuffer((__gm__ int32_t *)(wAddr + j * hCommuSize + hOutSize));
                expandXOutGlobal.SetGlobalBuffer((__gm__ int8_t *)(gmX1) + (beginIdx + j) * tokenLength, tokenLength);

                while (true) {
                    AscendC::DataCopy(tmpLocalTensor, tokGlobalInt32, INT32_COUNT_PER_BLOCK);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
                    if (tmpLocalTensor.GetValue(1) == tokenFlag) {
                        tokGlobalInt32.SetValue(1, 0);
                        __asm__ __volatile__("");
                        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                                        AscendC::DcciDst::CACHELINE_OUT>(tokGlobalInt32[1]);
                        __asm__ __volatile__("");
                        break;
                    }
                }
                AscendC::PipeBarrier<PIPE_ALL>();

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                AscendC::DataCopy(xTmpTensor_, tokGlobal, axisHCommu);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);
                AscendC::DataCopyPad(dynamicScalesOutGMTensor_[beginIdx + j], xOutFp32Tensor_[tokenLength / sizeof(float)],
                                    dataCopyParamsFloat);
                AscendC::DataCopy(expandXOutGlobal, xTmpTensor_, tokenLength);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            beginIdx += count;
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::DataCopyExtParams dataCopyOutParams = {1U, static_cast<uint32_t>(recvRankNumPerCore * sizeof(int32_t)), 0U,
                                                        0U, 0U};
        AscendC::GlobalTensor<int32_t> sendCountsGlobal;
        sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(gmEpSendCount));
        AscendC::DataCopyPad(sendCountsGlobal[startRankId], gatherMaskOutCountTensor, dataCopyOutParams);
    }

    CATLASS_DEVICE
    void RecvCoreFunc(GM_ADDR gmX1, GM_ADDR gmX1Scale, GM_ADDR gmEpSendCount, GM_ADDR gmOutputRecvCount)
    {
        ubOffset = 0;
        RecvCount(ubOffset);

        uint32_t recvExpertNum = isShareExpert ? epRankSize : expertCntUp;
        uint32_t recvCoreNumPerGroup = recvCoreNum / localExpertNum;
        uint32_t recvRankNumPerCore = epRankSize / recvCoreNumPerGroup;
        uint32_t remainderRankNum = epRankSize % recvCoreNumPerGroup;

        uint32_t groupId = recvCoreIdx / recvCoreNumPerGroup;
        uint32_t recvCoreIdxInGroup = recvCoreIdx % recvCoreNumPerGroup;
        uint32_t startRankIdInGroup = recvRankNumPerCore * recvCoreIdxInGroup;
        if (recvCoreIdxInGroup < remainderRankNum) {
            recvRankNumPerCore += 1;
            startRankIdInGroup += recvCoreIdxInGroup;
        } else {
            startRankIdInGroup += remainderRankNum;
        }
        uint32_t endRankIdInGroup = startRankIdInGroup + recvRankNumPerCore;
        uint32_t startRankId = epRankSize * groupId + startRankIdInGroup;
        uint32_t endRankId = epRankSize * groupId + endRankIdInGroup;

        uint32_t coreTokenCount = 0;

        if (startRankId < recvExpertNum) {
            // RecvCount, GetCumSum, RecvToken must use the same ubOffset to get right info
            GetCumSum(startRankId, recvExpertNum, ubOffset, gmOutputRecvCount);
            RecvToken(gmX1, gmX1Scale, gmEpSendCount, coreTokenCount, startRankId, endRankId, recvRankNumPerCore, ubOffset);
        }

        // recv finish, inform AIC
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::LocalTensor<int32_t> tmpLocalTensor = resource.ubBuf.template GetBufferByByte<int32_t>(0);
        ubOffset += CEIL_UP(UB_BLOCK_SIZE);
        tmpLocalTensor.SetValue(CV_FLAG_INDEX, vToCFlag);
        tmpLocalTensor.SetValue(GROUP_ID_INDEX, groupId);
        tmpLocalTensor.SetValue(SELF_COUNT_INDEX, coreTokenCount);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);

        AscendC::GlobalTensor<int32_t> groupTokenNumStateTensor;
        groupTokenNumStateTensor.SetGlobalBuffer((__gm__ int32_t *)(statusDataSpaceGm + GROUP_TOKEN_NUM_OFFSET));
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::SetAtomicAdd<int32_t>();
        AscendC::DataCopy(groupTokenNumStateTensor[groupId * GROUP_INFO_SIZE], tmpLocalTensor, INT32_COUNT_PER_BLOCK);
        AscendC::SetAtomicNone();
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE
    void CompCoreFunc(GM_ADDR gmCVSwapBuff, __gm__ ElementScale *gmScale, __gm__ ElementPerTokenScale *gmTokenScale,
                    __gm__ float *gmSwigluOutput, uint32_t n, uint32_t k, LayoutScale layoutScale,
                    LayoutPerTokenScale wholeLayoutPerTokenScale, LayoutOutput layoutOutput)
    {
        uint32_t nOut = n / 2;
        uint32_t coreNumPerGroup = recvCoreNum / localExpertNum;
        int64_t gmGroupOffsetScale = 0;
        int64_t gmGroupOffsetPerTokenScale = 0;
        int64_t gmGroupOffsetD = 0;

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(gmCVSwapBuff));
        auto layoutC = layout::RowMajor{L1TileShape::M * aiCoreGroupNum * WORKSPACE_STAGES, L1TileShape::N};
        {
            BlockScheduler blockScheduler;
            BlockEpilogue blockEpilogue(resource);

            uint32_t stageId = 0;
            uint32_t target = 1;
            uint32_t startCoreIdx = 0;

            AscendC::GlobalTensor<int32_t> groupTokenNumStateTensor;
            for (uint32_t groupIdx = 0; groupIdx < localExpertNum; ++groupIdx) {
                // just like AIC
                groupTokenNumStateTensor.SetGlobalBuffer((__gm__ int32_t *)(statusDataSpaceGm + GROUP_TOKEN_NUM_OFFSET) +
                                                        groupIdx * GROUP_INFO_SIZE);
                while (true) {
                    __asm__ __volatile__("");
                    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                                    AscendC::DcciDst::CACHELINE_OUT>(groupTokenNumStateTensor);
                    __asm__ __volatile__("");
                    if (groupTokenNumStateTensor.GetValue(0) == coreNumPerGroup * vToCFlag) {
                        break;
                    }
                }
                uint32_t currentM = groupTokenNumStateTensor.GetValue(GROUP_TOKEN_COUNT);
                GemmCoord inGroupProblemShape{currentM, n, k};
                LayoutPerTokenScale layoutPerTokenScale =
                    wholeLayoutPerTokenScale.GetTileLayout(inGroupProblemShape.template GetCoordByAxis<0>());
                LayoutD layoutD = layoutOutput.GetTileLayout(MakeCoord(currentM, nOut));
                EpilogueParams epilogueParams{gmScale + gmGroupOffsetScale,
                                            layoutScale,
                                            gmTokenScale + gmGroupOffsetPerTokenScale,
                                            layoutPerTokenScale,
                                            gmSwigluOutput + gmGroupOffsetD,
                                            layoutD};
                blockScheduler.Update(inGroupProblemShape, L1TileShape::ToCoordMN());
                blockEpilogue.UpdateParams(epilogueParams);
                uint32_t coreLoops = blockScheduler.GetCoreLoops();

                GemmCoord blockShapeMNK = L1TileShape::ToCoord();
                uint32_t startLoopIdx =
                    ((compCoreIdx < startCoreIdx) ? (compCoreIdx + aiCoreGroupNum) : compCoreIdx) - startCoreIdx;
                for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += aiCoreGroupNum) {
                    GemmCoord blockCoordMNK = blockScheduler.GetBlockCoord(loopIdx);
                    GemmCoord actualBlockShapeMNK = blockScheduler.GetActualBlockShape(blockCoordMNK);

                    MatrixCoord offsetC{(stageId * aiCoreGroupNum + aiCoreGroupIdx) * L1TileShape::M, 0};
                    int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                    auto gmBlockC = gmC[gmOffsetC];
                    auto layoutBlockC = layoutC.GetTileLayout(actualBlockShapeMNK.GetCoordMN());
                    CheckSyncFlag(statusDataSpaceGm + SOFT_SYNC_OFFSET,
                        static_cast<uint8_t>(compCoreNum + compCoreIdx), target);
                    target += 1;
                    blockEpilogue(blockShapeMNK, blockCoordMNK, actualBlockShapeMNK, gmBlockC, layoutBlockC);
                    EncreaseSyncFlag(statusDataSpaceGm + SOFT_SYNC_OFFSET, static_cast<uint8_t>(compCoreIdx));
                    stageId = (stageId + 1 < WORKSPACE_STAGES) ? (stageId + 1) : 0;
                }

                gmGroupOffsetScale += inGroupProblemShape.n();
                gmGroupOffsetPerTokenScale += inGroupProblemShape.m();
                gmGroupOffsetD += currentM * nOut;

                startCoreIdx = (startCoreIdx + coreLoops) % aiCoreGroupNum;
            }
        }
        // clean
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::GlobalTensor<int32_t> softSyncTensor;
        softSyncTensor.SetGlobalBuffer((__gm__ int32_t *)(statusDataSpaceGm + SOFT_SYNC_OFFSET));
        AscendC::LocalTensor<int32_t> tmpZeroLocalTensor = resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::Duplicate(tmpZeroLocalTensor, (int32_t)0, INT32_COUNT_PER_BLOCK);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::DataCopy(softSyncTensor[compCoreIdx * SOFT_SYNC_SPACE_SIZE / sizeof(int32_t)], tmpZeroLocalTensor,
                        INT32_COUNT_PER_BLOCK);
        AscendC::DataCopy(softSyncTensor[(compCoreIdx + compCoreNum) * SOFT_SYNC_SPACE_SIZE / sizeof(int32_t)],
                        tmpZeroLocalTensor, INT32_COUNT_PER_BLOCK);
    }

    CATLASS_DEVICE
    void AivInitParams(Params const &params)
    {
        aiCoreGroupNum = AscendC::GetBlockNum();
        subBlockNum = AscendC::GetSubBlockNum();  // 1C2V
        aicNum = aiCoreGroupNum;
        aivNum = aiCoreGroupNum * subBlockNum;
        aivIdx = AscendC::GetBlockIdx();
        aiCoreGroupIdx = aivIdx / subBlockNum;
        aivStateGlobalCoreIdx = aivNum + aicNum + aivIdx;

        isCompCore = (aivIdx % subBlockNum) == 0;
        compCoreNum = aiCoreGroupNum;
        compCoreIdx = aiCoreGroupIdx;
        // when localExpertNum=1, all cores send token and recv token in sequence
        isRecvCore = true;
        isSendCore = true;
        recvCoreIdx = aivIdx;
        sendCoreIdx = aivIdx;
        sendCoreNum = aivNum;
        recvCoreNum = aivNum;

        moeExpertNumPerRank = params.moeExpertNumPerRank;

        epRankSize = params.epRankSize;
        epRankId = params.epRankId;
        expertCntUp = epRankSize * moeExpertNumPerRank;
        sharedExpertRankNum = params.sharedExpertRankNum;
        hasShareExpert = (sharedExpertRankNum > 0);
        isShareExpert = (epRankId < sharedExpertRankNum);
        localExpertNum = isShareExpert ? 1 : moeExpertNumPerRank;
        moeExpertNum = params.moeExpertNum;
        tokenLength = params.tokenLen;

        // when localExpertNum>1, half of cores send token and another half recv token in parallel
        if (localExpertNum > 1) {
            isRecvCore = ((aivIdx % ODD_EVEN_BASE) == 0);
            isSendCore = ((aivIdx % ODD_EVEN_BASE) == 1);
            recvCoreIdx = aivIdx / subBlockNum;
            sendCoreIdx = aivIdx / subBlockNum;
            sendCoreNum = aiCoreGroupNum;
            recvCoreNum = aiCoreGroupNum;
        }

        hOutSize = tokenLength * sizeof(int8_t);
        scaleParamPad = TOKEN_EXTRA_SPACE;  // 512B for dynamic quant scale
        hCommuSize = hOutSize + scaleParamPad;
        axisHCommu = hCommuSize / sizeof(int8_t);
        axisBS = params.bs;
        axisK = params.topK;
        uint32_t maxAxisBs = params.globalBs / epRankSize;

        stateOffset = STATE_OFFSET;
        expertPerSizeOnWin = maxAxisBs * tokenLength * sizeof(XType);
        winContext_ = (__gm__ HcclOpResParam *)AscendC::GetHcclContext<AscendC::HCCL_GROUP_ID_0>();
        statusDataSpaceGm = (GM_ADDR)(winContext_->localWindowsExp);
    }

    CATLASS_DEVICE
    void AivInitState()
    {
        // state of data sapce
        AscendC::GlobalTensor<int32_t> selfDataStatusTensor;
        selfDataStatusTensor.SetGlobalBuffer((__gm__ int32_t *)(statusDataSpaceGm + STATE_WIN_OFFSET));
        __asm__ __volatile__("");
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
            selfDataStatusTensor[aivIdx * UB_ALIGN]);
        __asm__ __volatile__("");
        dataState = selfDataStatusTensor(aivIdx * UB_ALIGN);
        if (dataState == 0) {
            selfDataStatusTensor(aivIdx * UB_ALIGN) = 1;
        } else {
            selfDataStatusTensor(aivIdx * UB_ALIGN) = 0;
        }
        __asm__ __volatile__("");
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
            selfDataStatusTensor[aivIdx * UB_ALIGN]);
        __asm__ __volatile__("");

        // state of cv flag
        __asm__ __volatile__("");
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
            selfDataStatusTensor[aivStateGlobalCoreIdx * UB_ALIGN]);
        __asm__ __volatile__("");
        cvDataState = selfDataStatusTensor(aivStateGlobalCoreIdx * UB_ALIGN);
        if (cvDataState == 0) {
            selfDataStatusTensor(aivStateGlobalCoreIdx * UB_ALIGN) = 1;
            vToCFlag = V_TO_C_FLAG_1;
        } else {
            selfDataStatusTensor(aivStateGlobalCoreIdx * UB_ALIGN) = 0;
            vToCFlag = V_TO_C_FLAG_2;
        }
        __asm__ __volatile__("");
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
            selfDataStatusTensor[aivStateGlobalCoreIdx * UB_ALIGN]);
        __asm__ __volatile__("");

        AscendC::PipeBarrier<PIPE_ALL>();
        winDataSizeOffset = dataState * epRankSize * expertPerSizeOnWin * moeExpertNumPerRank;
        GM_ADDR statusSpaceGm_ = GET_WIND_STATE_ADDR_BY_RANK_ID(epRankId);
        AscendC::GlobalTensor<int32_t> selfStatusTensor;
        selfStatusTensor.SetGlobalBuffer((__gm__ int32_t *)(statusSpaceGm_ + SELF_STATE_OFFSET));
        __asm__ __volatile__("");
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
            selfStatusTensor[aivIdx * UB_ALIGN]);
        __asm__ __volatile__("");
        state = selfStatusTensor(aivIdx * UB_ALIGN);
        if (state == 0) {
            sumTarget = (float)1.0;
            tokenFlag = TOKEN_FLAG_1;
            selfStatusTensor(aivIdx * UB_ALIGN) = 0x3F800000;
        } else {
            sumTarget = 0.0;
            tokenFlag = TOKEN_FLAG_2;
            selfStatusTensor(aivIdx * UB_ALIGN) = 0;
        }
        __asm__ __volatile__("");
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
            selfStatusTensor[aivIdx * UB_ALIGN]);
        __asm__ __volatile__("");
    }

    CATLASS_DEVICE
    void UpdateAndCleanInfo(__gm__ ElementGroupList_ *ptrGroupList, GM_ADDR gmEpSendCount)
    {
        if (aivIdx == aiCoreGroupNum * subBlockNum - 1) {
            // clean
            AscendC::GlobalTensor<int32_t> groupTokenNumStateTensor;
            groupTokenNumStateTensor.SetGlobalBuffer((__gm__ int32_t *)(statusDataSpaceGm + GROUP_TOKEN_NUM_OFFSET));
            AscendC::LocalTensor<int32_t> tmpZeroLocalTensor = resource.ubBuf.template GetBufferByByte<int32_t>(0);
            AscendC::Duplicate(tmpZeroLocalTensor, (int32_t)0, GROUP_INFO_SIZE * localExpertNum);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::DataCopy(groupTokenNumStateTensor, tmpZeroLocalTensor, GROUP_INFO_SIZE * localExpertNum);
        }

        if (isRecvCore && recvCoreIdx == (recvCoreNum - 1)) {
            // record token count for each local expert
            AscendC::GlobalTensor<int64_t> expertTokenNumsOutGMTensor_;
            expertTokenNumsOutGMTensor_.SetGlobalBuffer((__gm__ int64_t *)(ptrGroupList));
            AscendC::GlobalTensor<int32_t> sendCountsGlobal;
            sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(gmEpSendCount));
            for (uint32_t localMoeIndex = 0; localMoeIndex < localExpertNum; ++localMoeIndex) {
                __asm__ __volatile__("");
                AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                                AscendC::DcciDst::CACHELINE_OUT>(
                    sendCountsGlobal[localMoeIndex * epRankSize + epRankSize - 1]);
                __asm__ __volatile__("");
                uint32_t tokenNum = sendCountsGlobal.GetValue(localMoeIndex * epRankSize + epRankSize - 1);
                expertTokenNumsOutGMTensor_.SetValue(localMoeIndex, tokenNum);
                __asm__ __volatile__("");
                AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                                AscendC::DcciDst::CACHELINE_OUT>(
                    expertTokenNumsOutGMTensor_[localMoeIndex]);
                __asm__ __volatile__("");
            }
        }
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
    {
        AivInitParams(params);
        AivInitState();
        if (isSendCore) {
            SendCoreFunc((GM_ADDR)params.gmX, (GM_ADDR)params.gmexpertIds, (GM_ADDR)params.ptrA,
                        (GM_ADDR)params.ptrPerTokenScale, (GM_ADDR)params.gmExpandIdx);
        }
        if (isRecvCore) {
            RecvCoreFunc((GM_ADDR)params.ptrA, (GM_ADDR)params.ptrPerTokenScale, (GM_ADDR)params.gmEpSendCount,
                        (GM_ADDR)params.gmOutputRecvCount);
        }

        auto gmSwigluOutput = reinterpret_cast<__gm__ float *>(
            params.ptrWorkspace + sizeof(int32_t) * (L1TileShape::M * aiCoreGroupNum * WORKSPACE_STAGES * L1TileShape::N));
        if (isCompCore) {
            CompCoreFunc(params.ptrWorkspace, params.ptrScale, params.ptrPerTokenScale, gmSwigluOutput,
                        params.problemShape.n(), params.problemShape.k(), params.layoutScale, params.layoutPerTokenScale,
                        params.layoutOutput);
        }

        icache_preload(8);
        AscendC::SyncAll<false>();
        AscendC::PipeBarrier<PIPE_ALL>();

        UpdateAndCleanInfo(params.ptrGroupList, params.gmEpSendCount);
        {
            // dynamic quant
            AscendC::GlobalTensor<int32_t> sendCountsGlobal;
            sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.gmEpSendCount));
            __asm__ __volatile__("");
            AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                            AscendC::DcciDst::CACHELINE_OUT>(sendCountsGlobal);
            __asm__ __volatile__("");
            totalTokenCount = sendCountsGlobal.GetValue(localExpertNum * epRankSize - 1);
            AscendC::PipeBarrier<PIPE_ALL>();
            uint32_t nOut = params.problemShape.n() / 2;
            uint32_t quantRowOnce = 0;
            CalQuantRow(nOut, quantRowOnce);
            typename BlockQuant<ArchTag>::Params quantParams{
                gmSwigluOutput,   params.layoutOutput, params.ptrDequantScale, params.layoutDequantScale,
                params.ptrOutput, params.layoutOutput, quantRowOnce,           nOut};

            BlockQuant<ArchTag> blockQuant(resource, quantParams);
            MatrixCoord quantShape(totalTokenCount, nOut);
            MatrixCoord quantBlockShape((uint16_t)(subBlockNum * quantRowOnce), nOut);
            Epilogue::Tile::EpilogueHorizontalTileSwizzle quantSwizzle(quantShape, quantBlockShape);
            for (uint32_t loopIdx = aiCoreGroupIdx; loopIdx < quantSwizzle.GetLoops(); loopIdx += aiCoreGroupNum) {
                auto blockCoord = quantSwizzle.GetTileCoord(loopIdx);
                auto actualBlockShape = quantSwizzle.GetActualTileShape(blockCoord);
                blockQuant(quantBlockShape, blockCoord, actualBlockShape);
            }
        }
    }

private:
    friend struct AicWaitFunc1;
    friend struct AicSetFunc1;

    struct AicWaitFunc1 {
        CATLASS_DEVICE
        AicWaitFunc1() = default;

        CATLASS_DEVICE
        void operator()() const
        {
            CheckSyncFlag(flagAddr, idx, target);
        }

        __gm__ uint8_t *flagAddr;
        uint8_t idx;
        uint32_t target;
    };

    struct AicSetFunc1 {
        CATLASS_DEVICE
        AicSetFunc1() = default;

        CATLASS_DEVICE
        void operator()() const
        {
            EncreaseSyncFlag(flagAddr, idx);
        }

        __gm__ uint8_t *flagAddr;
        uint8_t idx;
    };

    AicWaitFunc1 aicWaitFunc1;
    AicSetFunc1 aicSetFunc1;
    Arch::Resource<ArchTag> resource;

    AscendC::LocalTensor<int32_t> expertIdsTensor_;

    // rank and expert info
    uint32_t epRankSize{0};
    uint32_t epRankId{0};
    bool hasShareExpert{false};
    bool isShareExpert{false};
    uint32_t expertCntUp{0};
    uint32_t localExpertNum{0};
    uint32_t sharedExpertRankNum{0};
    uint32_t moeExpertNumPerRank{0};
    uint32_t moeExpertNum{0};

    // token info
    uint32_t hOutSize{0};
    uint32_t scaleParamPad{0};
    uint32_t hCommuSize{0};
    uint32_t axisHCommu{0};
    uint32_t axisBS{0};
    uint32_t axisK{0};
    uint32_t totalTokenCount{0};
    uint32_t expertIdsCnt{0};
    uint32_t tokenLength{0};

    // state info
    int32_t tokenFlag{0};    // token flag
    int32_t vToCFlag{0};     // cv flag, decided by cvDataState
    int32_t dataState{0};    // data space state
    int32_t cvDataState{0};  // cv flag state
    int32_t state{0};        // count flag state
    float sumTarget{0.0};

    // memory info
    __gm__ HcclOpResParam *winContext_;
    GM_ADDR statusDataSpaceGm;
    uint32_t stateOffset{0};
    uint64_t expertPerSizeOnWin{0};
    uint64_t winDataSizeOffset{0};

    int64_t ubOffset;

    // core info
    bool isSendCore{false};
    bool isRecvCore{false};
    bool isCompCore{false};  // calculate deq_swiglu
    uint32_t aiCoreGroupNum{0};
    uint32_t aiCoreGroupIdx{0};
    uint32_t subBlockNum{0};
    uint32_t aicNum{0};
    uint32_t aivNum{0};
    uint32_t sendCoreNum{0};
    uint32_t recvCoreNum{0};
    uint32_t compCoreNum{0};
    uint32_t aivIdx{0};
    uint32_t aicIdx{0};
    uint32_t sendCoreIdx{0};
    uint32_t recvCoreIdx{0};
    uint32_t compCoreIdx{0};
    uint32_t aivStateGlobalCoreIdx{0};
    uint32_t aicStateGlobalCoreIdx{0};
    uint32_t sendToMoeAivNum{0};
    uint32_t sendToShareAivNum{0};
};

}  // namespace Catlass::Gemm::Kernel

namespace Catlass::Gemm::Kernel {

template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, uint32_t WORKSPACE_STAGES_,
          class ElementGroupList_>
class GroupedMatmulSliceMPerTokenDequantSwigluQuantMultiStageWorkspaceWithShallowDispatch
{
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using BlockEpilogue = BlockEpilogue_;
    using ElementScale = typename BlockEpilogue::ElementScale;
    using LayoutScale = typename BlockEpilogue::LayoutScale;
    using ElementPerTokenScale = typename BlockEpilogue::ElementPerTokenScale;
    using LayoutPerTokenScale = typename BlockEpilogue::LayoutPerTokenScale;
    using ElementD = typename BlockEpilogue::ElementD;
    using LayoutD = typename BlockEpilogue::LayoutD;
    using EpilogueParams = typename BlockEpilogue::Params;

    using ElementDequantScale = typename BlockQuant<ArchTag>::ElementDequantScale;
    using LayoutDequantScale = typename BlockQuant<ArchTag>::LayoutDequantScale;
    using ElementOutput = typename BlockQuant<ArchTag>::ElementOutput;
    using LayoutOutput = typename BlockQuant<ArchTag>::LayoutOutput;

    using BlockScheduler = BlockScheduler_;
    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;
    using ElementGroupList = ElementGroupList_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        uint32_t problemCount;
        __gm__ ElementGroupList_ *ptrGroupList;
        __gm__ ElementA *ptrA;
        LayoutA layoutA;
        __gm__ ElementB *ptrB;
        LayoutB layoutB;
        __gm__ ElementScale *ptrScale;
        LayoutScale layoutScale;
        __gm__ ElementPerTokenScale *ptrPerTokenScale;
        LayoutPerTokenScale layoutPerTokenScale;
        __gm__ ElementOutput *ptrOutput;
        LayoutOutput layoutOutput;
        __gm__ ElementDequantScale *ptrDequantScale;
        LayoutDequantScale layoutDequantScale;
        GM_ADDR ptrWorkspace;

        // Methods
        CATLASS_DEVICE
        Params() {}

        CATLASS_DEVICE
        Params(GemmCoord problemShape_, uint32_t problemCount_, GM_ADDR ptrGroupList_, GM_ADDR ptrA_,
               LayoutA const &layoutA_, GM_ADDR ptrB_, LayoutB const &layoutB_, GM_ADDR ptrScale_,
               LayoutScale const &layoutScale_, GM_ADDR ptrPerTokenScale_,
               LayoutPerTokenScale const &layoutPerTokenScale_, GM_ADDR ptrOutput_, LayoutOutput const &layoutOutput_,
               GM_ADDR ptrDequantScale_, LayoutDequantScale const &layoutDequantScale_, GM_ADDR ptrWorkspace_)
            : problemShape(problemShape_),
              problemCount(problemCount_),
              ptrGroupList(reinterpret_cast<__gm__ ElementGroupList *>(ptrGroupList_)),
              ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)),
              layoutA(layoutA_),
              ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)),
              layoutB(layoutB_),
              ptrScale(reinterpret_cast<__gm__ ElementScale *>(ptrScale_)),
              layoutScale(layoutScale_),
              ptrPerTokenScale(reinterpret_cast<__gm__ ElementPerTokenScale *>(ptrPerTokenScale_)),
              layoutPerTokenScale(layoutPerTokenScale_),
              ptrOutput(reinterpret_cast<__gm__ ElementOutput *>(ptrOutput_)),
              layoutOutput(layoutOutput_),
              ptrDequantScale(reinterpret_cast<__gm__ ElementDequantScale *>(ptrDequantScale_)),
              layoutDequantScale(layoutDequantScale_),
              ptrWorkspace(ptrWorkspace_)
        {}
    };

    // Methods
    CATLASS_DEVICE
    GroupedMatmulSliceMPerTokenDequantSwigluQuantMultiStageWorkspaceWithShallowDispatch()
    {
        Arch::FlagID flagId = 0;
        for (uint32_t stageId = 0; stageId < WORKSPACE_STAGES; ++stageId) {
            flagAicFinishStoreList[stageId] = Arch::CrossCoreFlag(flagId++);
            flagAivFinishComputeList[stageId] = Arch::CrossCoreFlag(flagId++);
            aicWaitFuncList[stageId] = {this, stageId};
            aicSetFuncList[stageId] = {this, stageId};
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        BlockScheduler blockScheduler;
        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(params.ptrB);
        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetB = 0;

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrWorkspace));
        auto layoutC = layout::RowMajor{L1TileShape::M * coreNum * WORKSPACE_STAGES, L1TileShape::N};

        uint32_t stageId = 0;
        uint32_t stageUsed = 0;
        uint32_t startCoreIdx = 0;
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentM = (groupIdx == 0) ? groupList.GetValue(groupIdx)
                                                : (groupList.GetValue(groupIdx) - groupList.GetValue(groupIdx - 1));
            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};

            LayoutA layoutA = params.layoutA.GetTileLayout(inGroupProblemShape.GetCoordMK());
            LayoutB layoutB = params.layoutB;

            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();

            // Determine the starting loopIdx of the current core under the current groupIdx
            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;
            // Loop through the matmul of each groupIdx
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                // Compute block location
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);

                Callback callbackBeforeFixpipe{};
                if (stageUsed == WORKSPACE_STAGES) {
                    callbackBeforeFixpipe = MakeCallback(&aicWaitFuncList[stageId]);
                } else {
                    ++stageUsed;
                }
                Callback callbackAfterFixpipe = MakeCallback(&aicSetFuncList[stageId]);

                // Compute initial location in logical coordinates
                MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
                MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
                MatrixCoord offsetC{(stageId * coreNum + coreIdx) * L1TileShape::M, 0};
                int64_t gmOffsetA = layoutA.GetOffset(offsetA);
                int64_t gmOffsetB = layoutB.GetOffset(offsetB);
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);

                // Compute block-scoped matrix multiply-add
                if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                    blockMmad(gmA[gmGroupOffsetA + gmOffsetA], layoutA, gmB[gmGroupOffsetB + gmOffsetB], layoutB,
                              gmC[gmOffsetC], layoutC, actualBlockShape, callbackBeforeFixpipe, callbackAfterFixpipe);
                } else {
                    callbackBeforeFixpipe();
                    blockMmad(gmA[gmGroupOffsetA + gmOffsetA], layoutA, gmB[gmGroupOffsetB + gmOffsetB], layoutB,
                              gmC[gmOffsetC], layoutC, actualBlockShape);
                    callbackAfterFixpipe();
                }

                stageId = (stageId + 1 < WORKSPACE_STAGES) ? (stageId + 1) : 0;
            }

            gmGroupOffsetA += inGroupProblemShape.m() * inGroupProblemShape.k();
            gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }

        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
        }

        while (stageUsed > 0) {
            uint32_t aivComputeStageId =
                (stageId >= stageUsed) ? (stageId - stageUsed) : (stageId + WORKSPACE_STAGES - stageUsed);
            Arch::CrossCoreWaitFlag(flagAivFinishComputeList[aivComputeStageId]);
            --stageUsed;
        }
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
    {
        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();
        int64_t gmGroupOffsetScale = 0;
        int64_t gmGroupOffsetPerTokenScale = 0;
        int64_t gmGroupOffsetD = 0;

        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrWorkspace));
        auto layoutC = layout::RowMajor{L1TileShape::M * coreNum * WORKSPACE_STAGES, L1TileShape::N};

        auto ptrD = reinterpret_cast<__gm__ float *>(
            params.ptrWorkspace + sizeof(int32_t) * (L1TileShape::M * coreNum * WORKSPACE_STAGES * L1TileShape::N));

        uint32_t mActual = groupList.GetValue(params.problemCount - 1);
        uint32_t nOut = params.problemShape.n() / 2;

        {
            BlockScheduler blockScheduler;
            BlockEpilogue blockEpilogue(resource);

            uint32_t stageId = 0;
            uint32_t startCoreIdx = 0;
            for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
                uint32_t currentM = (groupIdx == 0) ? groupList.GetValue(groupIdx)
                                                    : (groupList.GetValue(groupIdx) - groupList.GetValue(groupIdx - 1));
                GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};

                LayoutScale layoutScale = params.layoutScale;
                LayoutPerTokenScale layoutPerTokenScale =
                    params.layoutPerTokenScale.GetTileLayout(inGroupProblemShape.template GetCoordByAxis<0>());
                LayoutD layoutD = params.layoutOutput.GetTileLayout(MakeCoord(currentM, nOut));

                EpilogueParams epilogueParams{params.ptrScale + gmGroupOffsetScale,
                                              layoutScale,
                                              params.ptrPerTokenScale + gmGroupOffsetPerTokenScale,
                                              layoutPerTokenScale,
                                              ptrD + gmGroupOffsetD,
                                              layoutD};

                blockScheduler.Update(inGroupProblemShape, L1TileShape::ToCoordMN());
                blockEpilogue.UpdateParams(epilogueParams);
                uint32_t coreLoops = blockScheduler.GetCoreLoops();

                GemmCoord blockShapeMNK = L1TileShape::ToCoord();
                uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;
                for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                    GemmCoord blockCoordMNK = blockScheduler.GetBlockCoord(loopIdx);
                    GemmCoord actualBlockShapeMNK = blockScheduler.GetActualBlockShape(blockCoordMNK);

                    MatrixCoord offsetC{(stageId * coreNum + coreIdx) * L1TileShape::M, 0};
                    int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                    auto gmBlockC = gmC[gmOffsetC];
                    auto layoutBlockC = layoutC.GetTileLayout(actualBlockShapeMNK.GetCoordMN());

                    Arch::CrossCoreWaitFlag(flagAicFinishStoreList[stageId]);
                    blockEpilogue(blockShapeMNK, blockCoordMNK, actualBlockShapeMNK, gmBlockC, layoutBlockC);
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishComputeList[stageId]);

                    stageId = (stageId + 1 < WORKSPACE_STAGES) ? (stageId + 1) : 0;
                }

                gmGroupOffsetScale += inGroupProblemShape.n();
                gmGroupOffsetPerTokenScale += inGroupProblemShape.m();
                gmGroupOffsetD += currentM * nOut;

                startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
            }
        }

        Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();

        {
            uint32_t quantRowOnce = 0;
            CalQuantRow(nOut, quantRowOnce);
            typename BlockQuant<ArchTag>::Params quantParams{ptrD,
                                                             params.layoutOutput,
                                                             params.ptrDequantScale,
                                                             params.layoutDequantScale,
                                                             params.ptrOutput,
                                                             params.layoutOutput,
                                                             quantRowOnce,
                                                             nOut};

            BlockQuant<ArchTag> blockQuant(resource, quantParams);
            MatrixCoord quantShape(mActual, nOut);
            MatrixCoord quantBlockShape((uint16_t)(AscendC::GetSubBlockNum() * quantRowOnce), nOut);
            Epilogue::Tile::EpilogueHorizontalTileSwizzle quantSwizzle(quantShape, quantBlockShape);
            for (uint32_t loopIdx = coreIdx; loopIdx < quantSwizzle.GetLoops(); loopIdx += coreNum) {
                auto blockCoord = quantSwizzle.GetTileCoord(loopIdx);
                auto actualBlockShape = quantSwizzle.GetActualTileShape(blockCoord);

                blockQuant(quantBlockShape, blockCoord, actualBlockShape);
            }
        }
    }

private:
    friend struct AicWaitFunc;
    friend struct AicSetFunc;

    struct AicWaitFunc {
        using MatmulKernel = GroupedMatmulSliceMPerTokenDequantSwigluQuantMultiStageWorkspaceWithShallowDispatch<
            BlockMmad, BlockEpilogue, BlockScheduler, WORKSPACE_STAGES, ElementGroupList>;

        CATLASS_DEVICE
        AicWaitFunc() = default;

        CATLASS_DEVICE
        void operator()() const
        {
            Arch::CrossCoreWaitFlag(ptr->flagAivFinishComputeList[stageId]);
        }

        MatmulKernel *ptr{nullptr};
        uint32_t stageId;
    };

    struct AicSetFunc {
        using MatmulKernel = GroupedMatmulSliceMPerTokenDequantSwigluQuantMultiStageWorkspaceWithShallowDispatch<
            BlockMmad, BlockEpilogue, BlockScheduler, WORKSPACE_STAGES, ElementGroupList>;

        CATLASS_DEVICE
        AicSetFunc() = default;

        CATLASS_DEVICE
        void operator()() const
        {
            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(ptr->flagAicFinishStoreList[stageId]);
        }

        MatmulKernel *ptr{nullptr};
        uint32_t stageId;
    };

    Arch::CrossCoreFlag flagAicFinishStoreList[WORKSPACE_STAGES];
    Arch::CrossCoreFlag flagAivFinishComputeList[WORKSPACE_STAGES];

    AicWaitFunc aicWaitFuncList[WORKSPACE_STAGES];
    AicSetFunc aicSetFuncList[WORKSPACE_STAGES];
    Arch::Resource<ArchTag> resource;
};

}  // namespace Catlass::Gemm::Kernel
