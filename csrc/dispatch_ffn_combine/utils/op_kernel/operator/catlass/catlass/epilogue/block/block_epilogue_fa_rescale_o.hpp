/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FA_RESCALE_O_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FA_RESCALE_O_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

namespace Catlass::Epilogue::Block {

template <
    class OutputType_,
    class InputType_
>
class BlockEpilogue <
    EpilogueAtlasA2FARescaleO,
    OutputType_,
    InputType_
> {
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasA2FARescaleO;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element;
    using ElementInput = typename InputType_::Element;

    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = typename InputType_::Layout;

    using CopyGmToUbInput = Tile::CopyGm2Ub<ArchTag, InputType_>;
    using CopyUbToGmOutput = Tile::CopyUb2Gm<ArchTag, OutputType_>;

    static constexpr uint32_t HALF_ELENUM_PER_BLK = 16;
    static constexpr uint32_t FLOAT_ELENUM_PER_BLK = 8;
    static constexpr uint32_t HALF_ELENUM_PER_VECCALC = 128;
    static constexpr uint32_t FLOAT_ELENUM_PER_VECCALC = 64;
    static constexpr uint32_t UB_TILE_SIZE = 16384;  // 64 * 128 * 2B
    static constexpr uint32_t UB_LINE_SIZE = 512;   // 128 * 2 * 2B
    static constexpr uint32_t HALF_ELENUM_PER_LINE = 256;    // 128 * 2
    static constexpr uint32_t FLOAT_ELENUM_PER_LINE = 128;   // 128
    static constexpr uint32_t MULTIPLIER = 2;

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource)
    {
        constexpr uint32_t LO_UB_TENSOR_OFFSET = 5 * UB_TILE_SIZE;
        constexpr uint32_t DM_UB_TENSOR_OFFSET = 7 * UB_TILE_SIZE + 4 * UB_LINE_SIZE;
        constexpr uint32_t LL_UB_TENSOR_OFFSET = 7 * UB_TILE_SIZE + 6 * UB_LINE_SIZE;
        constexpr uint32_t GL_UB_TENSOR_OFFSET = 7 * UB_TILE_SIZE + 10 * UB_LINE_SIZE;
        constexpr uint32_t TV_UB_TENSOR_OFFSET = 7 * UB_TILE_SIZE + 11 * UB_LINE_SIZE;
        constexpr uint32_t GO_UB_TENSOR_OFFSET = 8 * UB_TILE_SIZE;

        loUbTensor = resource.ubBuf.template GetBufferByByte<float>(LO_UB_TENSOR_OFFSET);
        dmUbTensor = resource.ubBuf.template GetBufferByByte<half>(DM_UB_TENSOR_OFFSET);
        llUbTensor = resource.ubBuf.template GetBufferByByte<float>(LL_UB_TENSOR_OFFSET);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
        tvUbTensor = resource.ubBuf.template GetBufferByByte<float>(TV_UB_TENSOR_OFFSET);
        goUbTensor = resource.ubBuf.template GetBufferByByte<float>(GO_UB_TENSOR_OFFSET);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    CATLASS_DEVICE
    void SetMask(int32_t len)
    {
        const int32_t MAX_MASK_LEN = 128;
        const int32_t HALF_MASK_LEN = 64;
        if (len >= MAX_MASK_LEN) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
        }
        int32_t highMask = len - HALF_MASK_LEN > 0 ? len - HALF_MASK_LEN : 0;
        int32_t lowMask = len - HALF_MASK_LEN >= 0 ? HALF_MASK_LEN : len;
        if (len < HALF_MASK_LEN) {
            AscendC::SetVectorMask<int8_t>(0x0, ((uint64_t)1 << lowMask) - 1);
        } else {
            AscendC::SetVectorMask<int8_t>(((uint64_t)1 << highMask) - 1, 0xffffffffffffffff);
        }
    }

    CATLASS_DEVICE
    void subCoreCompute(
        AscendC::GlobalTensor<ElementOutput> gOutput,
        AscendC::GlobalTensor<ElementInput> gInput,
        const LayoutOutput &layoutOutput,
        const LayoutInput &layoutInput,
        uint32_t nIdx, uint32_t isLast)
    {
        uint32_t subM = layoutInput.shape(0);
        uint32_t k = layoutInput.shape(1);
        uint32_t kRound = layoutInput.stride(0);
        uint32_t strideQO = layoutOutput.stride(0);
        uint32_t subMAligned128 = (subM + HALF_ELENUM_PER_VECCALC - 1) / HALF_ELENUM_PER_VECCALC;
        uint32_t subMAligned64 = (subM + FLOAT_ELENUM_PER_VECCALC - 1) / FLOAT_ELENUM_PER_VECCALC;
        uint32_t subMRound = (subM + HALF_ELENUM_PER_BLK - 1) / HALF_ELENUM_PER_BLK * HALF_ELENUM_PER_BLK;

        // Get the layout on UB
        LayoutInput layoutInUb(subM, k, kRound);

        if (subM > 0) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            // Copy O
            copyGmToUbInput(loUbTensor, gInput, layoutInUb, layoutInput);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            // 更新 L 和 O
            if (nIdx != 0) {
                // dm32 = castfp16to32(dm)
                AscendC::Cast<float, half, false>(
                    tvUbTensor,
                    dmUbTensor[nIdx % MULTIPLIER * HALF_ELENUM_PER_LINE],
                    AscendC::RoundMode::CAST_NONE,
                    (uint64_t)0,
                    subMAligned64,
                    AscendC::UnaryRepeatParams(1, 1, 8, 4));
                AscendC::PipeBarrier<PIPE_V>();
                // dm32_block = brcb(dm32)
                AscendC::Brcb(
                    tvUbTensor.ReinterpretCast<uint32_t>()[HALF_ELENUM_PER_VECCALC],
                    tvUbTensor.ReinterpretCast<uint32_t>(),
                    subMRound / FLOAT_ELENUM_PER_BLK,
                    AscendC::BrcbRepeatParams(1, 8));
                AscendC::PipeBarrier<PIPE_V>();
                // dm32 = exp(dm32)
                AscendC::Exp<float, false>(
                    tvUbTensor,
                    tvUbTensor,
                    (uint64_t)0,
                    subMAligned64,
                    AscendC::UnaryRepeatParams(1, 1, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
                // gl = dm * gl
                AscendC::Mul<float, false>(
                    glUbTensor,
                    tvUbTensor,
                    glUbTensor,
                    (uint64_t)0,
                    subMAligned64,
                    AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
                // gl = ll + gl
                AscendC::Add<float, false>(
                    glUbTensor,
                    glUbTensor,
                    llUbTensor[nIdx % MULTIPLIER * FLOAT_ELENUM_PER_LINE],
                    (uint64_t)0,
                    subMAligned64,
                    AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
                // dm32_block = exp(dm32_block)
                AscendC::Exp<float, false>(
                    tvUbTensor[HALF_ELENUM_PER_VECCALC],
                    tvUbTensor[HALF_ELENUM_PER_VECCALC],
                    (uint64_t)0,
                    (subM * FLOAT_ELENUM_PER_BLK + FLOAT_ELENUM_PER_VECCALC - 1) / FLOAT_ELENUM_PER_VECCALC,
                    AscendC::UnaryRepeatParams(1, 1, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
                if (goFlag == 1) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                    goFlag = 0;
                }
                // go = go * dm32_block
                for (uint32_t vmulIdx = 0; vmulIdx < k / FLOAT_ELENUM_PER_VECCALC; vmulIdx++) {
                    AscendC::Mul<float, false>(
                        goUbTensor[vmulIdx * FLOAT_ELENUM_PER_VECCALC],
                        goUbTensor[vmulIdx * FLOAT_ELENUM_PER_VECCALC],
                        tvUbTensor[HALF_ELENUM_PER_VECCALC],
                        (uint64_t)0,
                        subM,
                        AscendC::BinaryRepeatParams(
                            1, 1, 0, kRound / FLOAT_ELENUM_PER_BLK, kRound / FLOAT_ELENUM_PER_BLK, 1));
                    AscendC::PipeBarrier<PIPE_V>();
                }
                if (k % FLOAT_ELENUM_PER_VECCALC > 0) {
                    SetMask(k % FLOAT_ELENUM_PER_VECCALC);
                    AscendC::Mul<float, false>(
                        goUbTensor[k / FLOAT_ELENUM_PER_VECCALC * FLOAT_ELENUM_PER_VECCALC],
                        goUbTensor[k / FLOAT_ELENUM_PER_VECCALC * FLOAT_ELENUM_PER_VECCALC],
                        tvUbTensor[HALF_ELENUM_PER_VECCALC],
                        (uint64_t)0,
                        subM,
                        AscendC::BinaryRepeatParams(
                            1, 1, 0, kRound / FLOAT_ELENUM_PER_BLK, kRound / FLOAT_ELENUM_PER_BLK, 1));
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                }
                // go = lo + go
                AscendC::Add<float, false>(
                    goUbTensor,
                    goUbTensor,
                    loUbTensor,
                    (uint64_t)0,
                    (subM * kRound + FLOAT_ELENUM_PER_VECCALC - 1) / FLOAT_ELENUM_PER_VECCALC,
                    AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                // gl = ll
                AscendC::DataCopy(glUbTensor, llUbTensor[nIdx % MULTIPLIER * FLOAT_ELENUM_PER_LINE],
                    AscendC::DataCopyParams(1, subMRound / FLOAT_ELENUM_PER_BLK, 0, 0));
                AscendC::PipeBarrier<PIPE_V>();
                if (goFlag == 1) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                    goFlag = 0;
                }
                AscendC::DataCopy(goUbTensor, loUbTensor,
                    AscendC::DataCopyParams(1, subM * kRound / FLOAT_ELENUM_PER_BLK, 0, 0));
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            if (isLast) {
                // gl = castfp32to16(gl)
                AscendC::Cast<half, float, false>(
                    glUbTensor.ReinterpretCast<half>(),
                    glUbTensor,
                    AscendC::RoundMode::CAST_NONE,
                    (uint64_t)0,
                    subMAligned64,
                    AscendC::UnaryRepeatParams(1, 1, 4, 8));
                AscendC::PipeBarrier<PIPE_V>();
                // go = castfp32to16(go)
                AscendC::Cast<half, float, false>(
                    goUbTensor.ReinterpretCast<half>(),
                    goUbTensor,
                    AscendC::RoundMode::CAST_NONE,
                    (uint64_t)0,
                    (subM * kRound + FLOAT_ELENUM_PER_VECCALC - 1) / FLOAT_ELENUM_PER_VECCALC,
                    AscendC::UnaryRepeatParams(1, 1, 4, 8));
                AscendC::PipeBarrier<PIPE_V>();
                // gl_block = brcb(gl)
                AscendC::Brcb(
                    tvUbTensor.ReinterpretCast<uint16_t>(),
                    glUbTensor.ReinterpretCast<uint16_t>(),
                    subMRound / FLOAT_ELENUM_PER_BLK,
                    AscendC::BrcbRepeatParams(1, 8));
                AscendC::PipeBarrier<PIPE_V>();
                // go = go / gl_block
                for (uint32_t vdivIdx = 0; vdivIdx < k / HALF_ELENUM_PER_VECCALC; vdivIdx++) {
                    AscendC::Div<half, false>(
                        goUbTensor.ReinterpretCast<half>()[vdivIdx * HALF_ELENUM_PER_VECCALC],
                        goUbTensor.ReinterpretCast<half>()[vdivIdx * HALF_ELENUM_PER_VECCALC],
                        tvUbTensor.ReinterpretCast<half>(),
                        (uint64_t)0,
                        subM,
                        AscendC::BinaryRepeatParams(
                            1, 1, 0, kRound / HALF_ELENUM_PER_BLK, kRound / HALF_ELENUM_PER_BLK, 1));
                }
                if (k % HALF_ELENUM_PER_VECCALC > 0) {
                    SetMask(k % HALF_ELENUM_PER_VECCALC);
                    AscendC::Div<half, false>(
                        goUbTensor.ReinterpretCast<half>()[k / HALF_ELENUM_PER_VECCALC * HALF_ELENUM_PER_VECCALC],
                        goUbTensor.ReinterpretCast<half>()[k / HALF_ELENUM_PER_VECCALC * HALF_ELENUM_PER_VECCALC],
                        tvUbTensor.ReinterpretCast<half>(),
                        (uint64_t)0,
                        subM,
                        AscendC::BinaryRepeatParams(
                            1, 1, 0, kRound / HALF_ELENUM_PER_BLK, kRound / HALF_ELENUM_PER_BLK, 1));
                    AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                }
                // copy O to GM
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
                copyUbToGmOutput(gOutput, goUbTensor.ReinterpretCast<half>(), layoutOutput, layoutInUb);
                if (goFlag == 0) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                    goFlag = 1;
                }
            }
        }
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementOutput> gOutput,
        AscendC::GlobalTensor<ElementInput> gInput,
        const LayoutOutput &layoutOutput,
        const LayoutInput &layoutInput,
        GemmCoord actualBlockShape,
        uint32_t nIdx, uint32_t isLast)
    {
        uint32_t mActual = actualBlockShape.m();
        uint32_t nActual = actualBlockShape.n();

        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        uint32_t mActualPerSubBlock = CeilDiv(mActual, subBlockNum);
        uint32_t mActualThisSubBlock = (subBlockIdx == 0) ? mActualPerSubBlock : (mActual - mActualPerSubBlock);
        uint32_t mOffset = subBlockIdx * mActualPerSubBlock;
        uint32_t nOffset = 0;

        int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(mOffset, nOffset));
        auto gOutputThisSubBlock = gOutput[offsetOutput];
        auto layoutOutputThisSubBlock = layoutOutput.GetTileLayout(MatrixCoord(mActualThisSubBlock, nActual));

        int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(mOffset, nOffset));
        auto gInputThisSubBlock = gInput[offsetInput];
        auto layoutInputThisSubBlock = layoutInput.GetTileLayout(MatrixCoord(mActualThisSubBlock, nActual));

        subCoreCompute(gOutputThisSubBlock, gInputThisSubBlock,
                       layoutOutputThisSubBlock, layoutInputThisSubBlock,
                       nIdx, isLast);
    }

private:
    uint32_t goFlag = 1;
    AscendC::LocalTensor<float> loUbTensor;
    AscendC::LocalTensor<half> dmUbTensor;
    AscendC::LocalTensor<float> llUbTensor;
    AscendC::LocalTensor<float> glUbTensor;
    AscendC::LocalTensor<float> tvUbTensor;
    AscendC::LocalTensor<float> goUbTensor;

    CopyGmToUbInput copyGmToUbInput;
    CopyUbToGmOutput copyUbToGmOutput;
};

}  // namespace Catlass::Epilogue::Block

#endif  // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FA_RESCALE_O_HPP