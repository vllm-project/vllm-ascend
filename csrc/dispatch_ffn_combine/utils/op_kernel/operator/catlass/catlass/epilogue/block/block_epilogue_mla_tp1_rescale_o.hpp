/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_MLA_TP1_RESCALE_O_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_MLA_TP1_RESCALE_O_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Epilogue::Block {

template <class OutputType_, class UpdateType_, class InputType_>
class BlockEpilogue<EpilogueAtlasA2MLATP1RescaleO, OutputType_, UpdateType_, InputType_> {
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasA2MLATP1RescaleO;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using ElementOutput = typename OutputType_::Element;
    using ElementUpdate = typename UpdateType_::Element;
    using ElementInput = typename InputType_::Element;

    using LayoutOutput = typename OutputType_::Layout;
    using LayoutUpdate = typename UpdateType_::Layout;
    using LayoutInput = typename InputType_::Layout;

    static constexpr uint32_t HALF_ELENUM_PER_BLK = 16;
    static constexpr uint32_t HALF_ELENUM_PER_VECCALC = 128;
    static constexpr uint32_t FLOAT_ELENUM_PER_VECCALC = 64;
    static constexpr uint32_t HALF_ELENUM_PER_LINE = 256;
    static constexpr uint32_t FLOAT_ELENUM_PER_LINE = 128;
    static constexpr uint32_t MULTIPLIER = 2;
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t UB_UINT8_LINE_SIZE = 512;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE_MLA = 16384;
    static constexpr uint32_t ROW_WISE_CYCLE_TILE = 8;
    static constexpr uint32_t HALF_DM_UB_SIZE = 64;
    static constexpr uint32_t HALF_LL_UB_SIZE = 256;
    static constexpr uint32_t VECTOR_SIZE = 128;
    static constexpr uint32_t NUM4 = 4;

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource, uint32_t kvSplitCoreNum_ = 1)
    {
        // Allocate UB space
        constexpr uint32_t LO_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE_MLA;
        constexpr uint32_t DM_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE_MLA + 6 * UB_UINT8_LINE_SIZE;
        constexpr uint32_t GL_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE_MLA + 16 * UB_UINT8_LINE_SIZE;
        constexpr uint32_t GO_UB_TENSOR_OFFSET = 8 * UB_UINT8_BLOCK_SIZE_MLA;
        constexpr uint32_t TV_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE_MLA;
        constexpr uint32_t HM_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE_MLA + 1 * UB_UINT8_LINE_SIZE;
        constexpr uint32_t GM_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE_MLA + 14 * UB_UINT8_LINE_SIZE;

        kvSplitCoreNum = kvSplitCoreNum_;
        loUbTensor = resource.ubBuf.template GetBufferByByte<float>(LO_UB_TENSOR_OFFSET);
        dmUbTensor = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
        tvUbTensor = resource.ubBuf.template GetBufferByByte<float>(TV_UB_TENSOR_OFFSET);
        goUbTensor16 = resource.ubBuf.template GetBufferByByte<ElementOutput>(GO_UB_TENSOR_OFFSET);
        goUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(GO_UB_TENSOR_OFFSET);
        hmUbTensor = resource.ubBuf.template GetBufferByByte<float>(HM_UB_TENSOR_OFFSET);
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET);
    }

    CATLASS_DEVICE
    ~BlockEpilogue() {}

    CATLASS_DEVICE
    void SetMask(int32_t len)
    {
        uint64_t mask = 0;
        uint64_t one = 1;
        uint64_t temp = len % FLOAT_VECTOR_SIZE;
        for (int64_t i = 0; i < temp; i++) {
            mask |= one << i;
        }

        if (len == VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        } else if (len >= FLOAT_VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>(mask, (uint64_t)-1);
        } else {
            AscendC::SetVectorMask<int8_t>(0x0, mask);
        }
    }

    CATLASS_DEVICE
    void SetkvSplitCoreNum(uint32_t kvSplitCoreNum_)
    {
        kvSplitCoreNum = kvSplitCoreNum_;
    }

    CATLASS_DEVICE
    void SubCoreCompute(AscendC::GlobalTensor<ElementInput> gInput, AscendC::GlobalTensor<ElementUpdate> gUpdate,
                        AscendC::GlobalTensor<ElementOutput> gOutput, AscendC::GlobalTensor<ElementUpdate> gOCoreTmp,
                        AscendC::GlobalTensor<ElementUpdate> gl, const LayoutInput &layoutInput,
                        const LayoutOutput &layoutOutput, const LayoutUpdate &layoutUpdate, uint32_t nIdx,
                        uint32_t isLastNTile, uint32_t needRowLoop, uint32_t rowLoopIdx, uint32_t rescaleOPingPongFlag,
                        uint32_t &glFlag)
    {
        uint32_t curRowNum = layoutInput.shape(0);
        uint32_t embed = layoutInput.shape(1);
        uint32_t embedRound = layoutInput.stride(0);
        uint32_t curRowNumRound = RoundUp<HALF_ELENUM_PER_BLK>(curRowNum);
        uint64_t dmUbOffsetCurCycle =
            (uint64_t)(rescaleOPingPongFlag * HALF_DM_UB_SIZE + rowLoopIdx * ROW_WISE_CYCLE_TILE);
        uint32_t oUbOffset = oPingPangFlag * ROW_WISE_CYCLE_TILE * embedRound;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(oPingPangFlag);
        if (nIdx != NUM4) {
            AscendC::DataCopy(loUbTensor[oUbOffset], gInput,
                              AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(oPingPangFlag + 4);
        if (nIdx != NUM4) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            AscendC::Brcb(tvUbTensor.ReinterpretCast<uint32_t>(),
                          dmUbTensor[dmUbOffsetCurCycle].ReinterpretCast<uint32_t>(), curRowNumRound / FLOAT_BLOCK_SIZE,
                          AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();
            if (needRowLoop) {
                AscendC::DataCopy(goUbTensor32[oUbOffset], gUpdate,
                                  AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            }
            // *** go = go * dm_block
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            for (uint32_t vmul_idx = 0; vmul_idx < embed / FLOAT_VECTOR_SIZE; ++vmul_idx) {
                AscendC::Mul<float, false>(goUbTensor32[oUbOffset + vmul_idx * FLOAT_VECTOR_SIZE],
                                           goUbTensor32[oUbOffset + vmul_idx * FLOAT_VECTOR_SIZE],
                                           tvUbTensor, (uint64_t)0,
                                           curRowNum,
                                           AscendC::BinaryRepeatParams(1, 1, 0, embedRound / FLOAT_BLOCK_SIZE,
                                                                       embedRound / FLOAT_BLOCK_SIZE, 1));
            }
            if (embed % FLOAT_VECTOR_SIZE > 0) {
                SetMask(embed % FLOAT_VECTOR_SIZE);
                AscendC::Mul<float, false>(goUbTensor32[oUbOffset + embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                           goUbTensor32[oUbOffset + embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                           tvUbTensor,
                                           (uint64_t)0, curRowNum,
                                           AscendC::BinaryRepeatParams(1, 1, 0, embedRound / FLOAT_BLOCK_SIZE,
                                                                       embedRound / FLOAT_BLOCK_SIZE, 1));
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
            AscendC::PipeBarrier<PIPE_V>();
            // *** go = lo + go
            AscendC::Add<float, false>(goUbTensor32[oUbOffset], goUbTensor32[oUbOffset],
                                       loUbTensor[oUbOffset], (uint64_t)0,
                                       (curRowNum * embedRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                       AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            // *** gl = ll
            AscendC::DataCopy(goUbTensor32[oUbOffset], gInput,
                              AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(oPingPangFlag);

        if (isLastNTile) {
            // *** gl_block = expand_to_block(gl), 存放于 tv
            AscendC::Brcb(tvUbTensor.ReinterpretCast<uint32_t>(),
                          glUbTensor.ReinterpretCast<uint32_t>()[rowLoopIdx * ROW_WISE_CYCLE_TILE],
                          curRowNumRound / FLOAT_BLOCK_SIZE,
                          AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // *** go = go / gl_block
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            for (uint32_t vdiv_idx = 0; vdiv_idx < embed / FLOAT_VECTOR_SIZE; ++vdiv_idx) {
                AscendC::Div<float, false>(goUbTensor32[oUbOffset + vdiv_idx * FLOAT_VECTOR_SIZE],
                                           goUbTensor32[oUbOffset + vdiv_idx * FLOAT_VECTOR_SIZE],
                                           tvUbTensor, (uint64_t)0,
                                           curRowNum,
                                           AscendC::BinaryRepeatParams(1, 1, 0, embedRound / FLOAT_BLOCK_SIZE,
                                                                       embedRound / FLOAT_BLOCK_SIZE, 1));
            }
            if (embed % FLOAT_VECTOR_SIZE > 0) {
                SetMask(embed % FLOAT_VECTOR_SIZE);
                AscendC::Div<float, false>(goUbTensor32[oUbOffset + embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                           goUbTensor32[oUbOffset + embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                           tvUbTensor,
                                           (uint64_t)0, curRowNum,
                                           AscendC::BinaryRepeatParams(1, 1, 0, embedRound / FLOAT_BLOCK_SIZE,
                                                                       embedRound / FLOAT_BLOCK_SIZE, 1));
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
            AscendC::PipeBarrier<PIPE_V>();

            if (kvSplitCoreNum != 1) {
                // log(l)
                AscendC::Ln<float, false>(
                    tvUbTensor,
                    tvUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::UnaryRepeatParams(1, 1, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Brcb(
                    hmUbTensor.ReinterpretCast<uint32_t>(),
                    gmUbTensor.ReinterpretCast<uint32_t>()[rowLoopIdx * ROW_WISE_CYCLE_TILE],
                    curRowNumRound / FLOAT_BLOCK_SIZE,
                    AscendC::BrcbRepeatParams(1, 8));
                AscendC::PipeBarrier<PIPE_V>();
                // logf(lse_sum) + lse_max
                AscendC::Add<float, false>(
                    tvUbTensor,
                    tvUbTensor,
                    hmUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
                AscendC::DataCopyPad(gl, tvUbTensor,
                    AscendC::DataCopyExtParams(curRowNum, 4, 0, (kvSplitCoreNum - 1) * 4, 0));

                if (glFlag == 0) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
                    glFlag = 1;
                }
                uint32_t srcGap = ((embed % 16 <= 8) && (embed % 16 > 0)) ? 1 : 0;
                AscendC::DataCopyPad(gOCoreTmp, goUbTensor32[oUbOffset],
                    AscendC::DataCopyExtParams(curRowNum, embed * 4, srcGap, (kvSplitCoreNum - 1) * embed * 4, 0));
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            } else {
                // *** go = castfp32to16(go)
                if (std::is_same<ElementOutput, bfloat16_t>::value) {
                    AscendC::Cast<ElementOutput, float, false>(
                        goUbTensor16[oUbOffset * 2], goUbTensor32[oUbOffset],
                        AscendC::RoundMode::CAST_RINT, (uint64_t)0,
                        (curRowNum * embedRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                        AscendC::UnaryRepeatParams(1, 1, 4, 8));
                } else {
                    AscendC::Cast<ElementOutput, float, false>(
                        goUbTensor16[oUbOffset * 2], goUbTensor32[oUbOffset],
                        AscendC::RoundMode::CAST_NONE, (uint64_t)0,
                        (curRowNum * embedRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                        AscendC::UnaryRepeatParams(1, 1, 4, 8));
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

                // ********************* move O to GM ************************
                AscendC::DataCopyPad(gOutput, goUbTensor16[oUbOffset * 2],
                                     AscendC::DataCopyExtParams(curRowNum, embed * 2, 0, 0, 0));
            }
        } else if (needRowLoop) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
            AscendC::DataCopy(gUpdate, goUbTensor32[oUbOffset],
                              AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(oPingPangFlag + 4);
        oPingPangFlag = 1 - oPingPangFlag;
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementInput> gInput, AscendC::GlobalTensor<ElementUpdate> gUpdate,
                    AscendC::GlobalTensor<ElementOutput> gOutput, AscendC::GlobalTensor<ElementUpdate> gOCoreTmp,
                    AscendC::GlobalTensor<ElementUpdate> gl, const LayoutInput &layoutInput,
                    const LayoutUpdate &layoutUpdate, const LayoutOutput &layoutOutput, GemmCoord actualBlockShape,
                    uint32_t nIdx, uint32_t isLastNTile, uint32_t rescaleOPingPongFlag, uint32_t &glFlag)
    {
        uint32_t embed = layoutInput.shape(1);
        uint32_t rowActual = actualBlockShape.m();
        uint32_t columnActual = actualBlockShape.n();

        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        uint32_t curRowSplitSubBlock = rowActual / subBlockNum;
        uint32_t rowActualThisSubBlock = (subBlockIdx == 0) ? curRowSplitSubBlock : (rowActual - curRowSplitSubBlock);
        uint32_t rowOffsetSubBlock = subBlockIdx * curRowSplitSubBlock;

        if (rowActualThisSubBlock > 0) {
            uint32_t rowLoop = (rowActualThisSubBlock + ROW_WISE_CYCLE_TILE - 1) / ROW_WISE_CYCLE_TILE;
            uint32_t needRowLoop = (rowLoop > 1) ? 1 : 0;
            for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoop; rowLoopIdx++) {
                uint32_t rowOffsetLoop = rowLoopIdx * ROW_WISE_CYCLE_TILE;
                uint32_t rowOffsetCurCycle = rowOffsetSubBlock + rowOffsetLoop;
                uint32_t rowActualCurCycle =
                    (rowLoopIdx == (rowLoop - 1)) ? rowActualThisSubBlock - rowOffsetLoop : ROW_WISE_CYCLE_TILE;
                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetCurCycle, 0));
                auto gInputThisCurCycle = gInput[offsetInput];
                auto layoutInputCurCycle = layoutInput.GetTileLayout(MatrixCoord(rowActualCurCycle, columnActual));

                int64_t offsetUpdate = layoutUpdate.GetOffset(MatrixCoord(rowOffsetCurCycle, 0));
                auto gUpdateCurCycle = gUpdate[offsetUpdate];
                auto layoutUpdateCurCycle = layoutUpdate.GetTileLayout(MatrixCoord(rowActualCurCycle, columnActual));

                int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetCurCycle, 0));
                auto gOutputCurCycle = gOutput[offsetOutput];
                auto layoutOutputCurCycle = layoutOutput.GetTileLayout(MatrixCoord(rowActualCurCycle, columnActual));

                SubCoreCompute(gInputThisCurCycle, gUpdateCurCycle, gOutputCurCycle,
                               gOCoreTmp[rowOffsetLoop * embed * kvSplitCoreNum],
                               gl[rowOffsetLoop * kvSplitCoreNum], layoutInputCurCycle,
                               layoutOutputCurCycle, layoutUpdateCurCycle, nIdx, isLastNTile, needRowLoop, rowLoopIdx,
                               rescaleOPingPongFlag, glFlag);
            }
        }
    }

private:
    uint32_t kvSplitCoreNum = 1;
    uint32_t oPingPangFlag = 0;
    AscendC::LocalTensor<float> loUbTensor;
    AscendC::LocalTensor<float> dmUbTensor;
    AscendC::LocalTensor<float> glUbTensor;
    AscendC::LocalTensor<float> tvUbTensor;
    AscendC::LocalTensor<ElementOutput> goUbTensor16;
    AscendC::LocalTensor<float> goUbTensor32;
    AscendC::LocalTensor<float> hmUbTensor;
    AscendC::LocalTensor<float> gmUbTensor;
};

} // namespace Catlass::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_MLA_TP1_RESCALE_O_HPP