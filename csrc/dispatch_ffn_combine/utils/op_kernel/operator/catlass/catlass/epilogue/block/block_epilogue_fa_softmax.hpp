/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FA_SOFTMAX_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FA_SOFTMAX_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

namespace Catlass::Epilogue::Block {

template <
    class OutputType_,
    class InputType_,
    class MaskType_
>
class BlockEpilogue <
    EpilogueAtlasA2FASoftmax,
    OutputType_,
    InputType_,
    MaskType_
> {
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasA2FASoftmax;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element;
    using ElementInput = typename InputType_::Element;
    using ElementMask = typename MaskType_::Element;

    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = typename InputType_::Layout;
    using LayoutMask = typename MaskType_::Layout;

    using CopyGmToUbInput = Tile::CopyGm2Ub<ArchTag, InputType_>;
    using CopyGmToUbMask = Tile::CopyGm2Ub<ArchTag, MaskType_>;
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
    BlockEpilogue(Arch::Resource<ArchTag> &resource, half tor_)
    {
        constexpr uint32_t LS32_UB_TENSOR_OFFSET = 2 * UB_TILE_SIZE;
        constexpr uint32_t MASK_UB_TENSOR_OFFSET = 4 * UB_TILE_SIZE;
        constexpr uint32_t LM_UB_TENSOR_OFFSET = 7 * UB_TILE_SIZE;
        constexpr uint32_t HM_UB_TENSOR_OFFSET = 7 * UB_TILE_SIZE + 1 * UB_LINE_SIZE;
        constexpr uint32_t GM_UB_TENSOR_OFFSET = 7 * UB_TILE_SIZE + 2 * UB_LINE_SIZE;
        constexpr uint32_t DM_UB_TENSOR_OFFSET = 7 * UB_TILE_SIZE + 4 * UB_LINE_SIZE;
        constexpr uint32_t LL_UB_TENSOR_OFFSET = 7 * UB_TILE_SIZE + 6 * UB_LINE_SIZE;
        constexpr uint32_t TV_UB_TENSOR_OFFSET = 7 * UB_TILE_SIZE + 11 * UB_LINE_SIZE;

        tor = tor_;
        lsUbTensor = resource.ubBuf.template GetBufferByByte<half>(0);
        lpUbTensor = resource.ubBuf.template GetBufferByByte<half>(0);
        ls32UbTensor = resource.ubBuf.template GetBufferByByte<float>(LS32_UB_TENSOR_OFFSET);
        maskUbTensor = resource.ubBuf.template GetBufferByByte<half>(MASK_UB_TENSOR_OFFSET);
        lmUbTensor = resource.ubBuf.template GetBufferByByte<half>(LM_UB_TENSOR_OFFSET);
        hmUbTensor = resource.ubBuf.template GetBufferByByte<half>(HM_UB_TENSOR_OFFSET);
        gmUbTensor = resource.ubBuf.template GetBufferByByte<half>(GM_UB_TENSOR_OFFSET);
        dmUbTensor = resource.ubBuf.template GetBufferByByte<half>(DM_UB_TENSOR_OFFSET);
        llUbTensor = resource.ubBuf.template GetBufferByByte<float>(LL_UB_TENSOR_OFFSET);
        tvUbTensor = resource.ubBuf.template GetBufferByByte<float>(TV_UB_TENSOR_OFFSET);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
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
    void SetVcgMask(int32_t len)
    {
        const int32_t MAX_LEN = 16;
        if (len > MAX_LEN) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
        }
        uint64_t subMask = ((uint64_t)1 << len) - 1;
        uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << 16) + subMask;
        AscendC::SetVectorMask<int8_t>(maskValue, maskValue);
    }

    CATLASS_DEVICE
    void subCoreCompute(
        AscendC::GlobalTensor<ElementOutput> gOutput,
        AscendC::GlobalTensor<ElementInput> gInput,
        AscendC::GlobalTensor<ElementMask> gMask,
        const LayoutOutput &layoutOutput,
        const LayoutInput &layoutInput,
        const LayoutMask &layoutMask,
        uint32_t nIdx, Arch::CrossCoreFlag qkReady)
    {
        uint32_t subM = layoutInput.shape(0);
        uint32_t qkN = layoutInput.shape(1);
        uint32_t qkNRound = layoutInput.stride(0);
        uint32_t maxSeqlen = layoutMask.stride(0);
        uint32_t offset = pingpongFlag * UB_TILE_SIZE / sizeof(ElementInput);
        uint32_t subMAligned128 = (subM + HALF_ELENUM_PER_VECCALC - 1) / HALF_ELENUM_PER_VECCALC;
        uint32_t subMRound = (subM + HALF_ELENUM_PER_BLK - 1) / HALF_ELENUM_PER_BLK * HALF_ELENUM_PER_BLK;

        // Get the layout on UB
        auto layoutInUb = LayoutInput::template MakeLayoutInUb<ElementInput>(MatrixCoord{subM, qkN});

        if (subM > 0) {
            // Copy mask
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            copyGmToUbMask(maskUbTensor, gMask, layoutInUb, layoutMask);
        }
        Arch::CrossCoreWaitFlag(qkReady);
        if (subM > 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(pingpongFlag);
            // Copy QK
            copyGmToUbInput(lsUbTensor[offset], gInput, layoutInUb, layoutInput);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            // ls = tor * ls
            AscendC::Muls<half, false>(
                lsUbTensor[offset],
                lsUbTensor[offset],
                tor,
                (uint64_t)0,
                (subM * qkNRound + HALF_ELENUM_PER_VECCALC - 1) / HALF_ELENUM_PER_VECCALC,
                AscendC::UnaryRepeatParams(1, 1, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // ls = ls + mask
            AscendC::Add<half, false>(
                lsUbTensor[offset],
                lsUbTensor[offset],
                maskUbTensor,
                (uint64_t)0,
                (subM * qkNRound + HALF_ELENUM_PER_VECCALC - 1) / HALF_ELENUM_PER_VECCALC,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            // lm = rowmax(ls)
            if (qkN <= HALF_ELENUM_PER_VECCALC) {
                SetMask(qkN);
                AscendC::BlockReduceMax<half, false>(
                    tvUbTensor.ReinterpretCast<half>(),
                    lsUbTensor[offset],
                    subM,
                    0,
                    2,
                    1,
                    qkNRound / HALF_ELENUM_PER_BLK);
                AscendC::PipeBarrier<PIPE_V>();
                SetVcgMask(qkNRound / HALF_ELENUM_PER_BLK);
                AscendC::BlockReduceMax<half, false>(
                    lmUbTensor,
                    tvUbTensor.ReinterpretCast<half>(),
                    (subM * HALF_ELENUM_PER_BLK + HALF_ELENUM_PER_VECCALC - 1) / HALF_ELENUM_PER_VECCALC,
                    0,
                    1,
                    1,
                    8);
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
            AscendC::PipeBarrier<PIPE_V>();
            if (nIdx == 0) {
                // hm = lm
                AscendC::DataCopy(
                    hmUbTensor, lmUbTensor, AscendC::DataCopyParams(1, subMRound / HALF_ELENUM_PER_BLK, 0, 0));
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                // hm = vmax(lm, gm)
                AscendC::Max<half, false>(
                    hmUbTensor,
                    lmUbTensor,
                    gmUbTensor,
                    (uint64_t)0,
                    subMAligned128,
                    AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
                // dm = gm - hm
                AscendC::Sub<half, false>(
                    dmUbTensor[nIdx % MULTIPLIER * HALF_ELENUM_PER_LINE],
                    gmUbTensor,
                    hmUbTensor,
                    (uint64_t)0,
                    subMAligned128,
                    AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
            }
            // gm = hm
            AscendC::DataCopy(
                gmUbTensor, hmUbTensor, AscendC::DataCopyParams(1, subMRound / HALF_ELENUM_PER_BLK, 0, 0));
            AscendC::PipeBarrier<PIPE_V>();
            // hm_block = brcb(hm), 存放于tv
            AscendC::Brcb(
                tvUbTensor.ReinterpretCast<uint16_t>(),
                hmUbTensor.ReinterpretCast<uint16_t>(),
                subMRound / FLOAT_ELENUM_PER_BLK,
                AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // ls = ls - hm_block
            for (uint32_t vsubIdx = 0; vsubIdx < qkN / HALF_ELENUM_PER_VECCALC; vsubIdx++) {
                AscendC::Sub<half, false>(
                    lsUbTensor[offset + vsubIdx * HALF_ELENUM_PER_VECCALC],
                    lsUbTensor[offset + vsubIdx * HALF_ELENUM_PER_VECCALC],
                    tvUbTensor.ReinterpretCast<half>(),
                    (uint64_t)0,
                    subM,
                    AscendC::BinaryRepeatParams(
                        1, 1, 0, qkNRound / HALF_ELENUM_PER_BLK, qkNRound / HALF_ELENUM_PER_BLK, 1));
            }
            if (qkN % HALF_ELENUM_PER_VECCALC > 0) {
                SetMask(qkN % HALF_ELENUM_PER_VECCALC);
                AscendC::Sub<half, false>(
                    lsUbTensor[offset + qkN / HALF_ELENUM_PER_VECCALC * HALF_ELENUM_PER_VECCALC],
                    lsUbTensor[offset + qkN / HALF_ELENUM_PER_VECCALC * HALF_ELENUM_PER_VECCALC],
                    tvUbTensor.ReinterpretCast<half>(),
                    (uint64_t)0,
                    subM,
                    AscendC::BinaryRepeatParams(
                        1, 1, 0, qkNRound / HALF_ELENUM_PER_BLK, qkNRound / HALF_ELENUM_PER_BLK, 1));
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
            AscendC::PipeBarrier<PIPE_V>();
            // ls32 = castfp16to32(ls)
            AscendC::Cast<float, half, false>(
                ls32UbTensor,
                lsUbTensor[offset],
                AscendC::RoundMode::CAST_NONE,
                (uint64_t)0,
                (subM * qkNRound + FLOAT_ELENUM_PER_VECCALC - 1) / FLOAT_ELENUM_PER_VECCALC,
                AscendC::UnaryRepeatParams(1, 1, 8, 4));
            AscendC::PipeBarrier<PIPE_V>();
            // ls32 = exp(ls32)
            AscendC::Exp<float, false>(
                ls32UbTensor,
                ls32UbTensor,
                (uint64_t)0,
                (subM * qkNRound + FLOAT_ELENUM_PER_VECCALC - 1) / FLOAT_ELENUM_PER_VECCALC,
                AscendC::UnaryRepeatParams(1, 1, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // lp = castfp32to16(ls)
            AscendC::Cast<half, float, false>(
                lpUbTensor[offset],
                ls32UbTensor,
                AscendC::RoundMode::CAST_NONE,
                (uint64_t)0,
                (subM * qkNRound + FLOAT_ELENUM_PER_VECCALC - 1) / FLOAT_ELENUM_PER_VECCALC,
                AscendC::UnaryRepeatParams(1, 1, 4, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            // ll = rowsum(ls32)
            if (qkN <= FLOAT_ELENUM_PER_VECCALC) {
                SetMask(qkN);
                AscendC::RepeatReduceSum<float, false>(
                    llUbTensor[nIdx % MULTIPLIER * FLOAT_ELENUM_PER_LINE],
                    ls32UbTensor,
                    subM, 0, 0, 1, 1, qkNRound / FLOAT_ELENUM_PER_BLK);
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            } else {
                for (uint32_t vaddIdx = 1; vaddIdx < qkN / FLOAT_ELENUM_PER_VECCALC; vaddIdx++) {
                    AscendC::Add<float, false>(
                        ls32UbTensor,
                        ls32UbTensor,
                        ls32UbTensor[vaddIdx * FLOAT_ELENUM_PER_VECCALC],
                        (uint64_t)0,
                        subM,
                        AscendC::BinaryRepeatParams(
                            1, 1, 1,
                            qkNRound / FLOAT_ELENUM_PER_BLK,
                            qkNRound / FLOAT_ELENUM_PER_BLK,
                            qkNRound / FLOAT_ELENUM_PER_BLK));
                    AscendC::PipeBarrier<PIPE_V>();
                }
                if (qkN % FLOAT_ELENUM_PER_VECCALC > 0) {
                    SetMask(qkN % FLOAT_ELENUM_PER_VECCALC);
                        AscendC::Add<float, false>(
                            ls32UbTensor,
                            ls32UbTensor,
                            ls32UbTensor[qkN / FLOAT_ELENUM_PER_VECCALC * FLOAT_ELENUM_PER_VECCALC],
                            (uint64_t)0,
                            subM,
                            AscendC::BinaryRepeatParams(
                                1, 1, 1,
                                qkNRound / FLOAT_ELENUM_PER_BLK,
                                qkNRound / FLOAT_ELENUM_PER_BLK,
                                qkNRound / FLOAT_ELENUM_PER_BLK));
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                }
                AscendC::RepeatReduceSum<float, false>(
                    llUbTensor[nIdx % MULTIPLIER * FLOAT_ELENUM_PER_LINE],
                    ls32UbTensor,
                    subM, 0, 0, 1, 1, qkNRound / FLOAT_ELENUM_PER_BLK);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            copyUbToGmOutput(gOutput, lpUbTensor[offset], layoutOutput, layoutInUb);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(pingpongFlag);
            pingpongFlag = 1 - pingpongFlag;
        }
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementOutput> gOutput,
        AscendC::GlobalTensor<ElementInput> gInput,
        AscendC::GlobalTensor<ElementMask> gMask,
        const LayoutOutput &layoutOutput,
        const LayoutInput &layoutInput,
        const LayoutMask &layoutMask,
        GemmCoord actualBlockShape,
        uint32_t nIdx, Arch::CrossCoreFlag qkReady)
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

        int64_t offsetMask = layoutMask.GetOffset(MatrixCoord(mOffset, nOffset));
        auto gMaskThisSubBlock = gMask[offsetMask];
        auto layoutMaskThisSubBlock = layoutMask.GetTileLayout(MatrixCoord(mActualThisSubBlock, nActual));

        subCoreCompute(gOutputThisSubBlock, gInputThisSubBlock, gMaskThisSubBlock,
                       layoutOutputThisSubBlock, layoutInputThisSubBlock, layoutMaskThisSubBlock,
                       nIdx, qkReady);
    }

private:
    half tor;
    uint32_t pingpongFlag = 0;
    AscendC::LocalTensor<half> lsUbTensor;
    AscendC::LocalTensor<half> lpUbTensor;
    AscendC::LocalTensor<float> ls32UbTensor;
    AscendC::LocalTensor<half> maskUbTensor;
    AscendC::LocalTensor<half> lmUbTensor;
    AscendC::LocalTensor<half> hmUbTensor;
    AscendC::LocalTensor<half> gmUbTensor;
    AscendC::LocalTensor<half> dmUbTensor;
    AscendC::LocalTensor<float> llUbTensor;
    AscendC::LocalTensor<float> tvUbTensor;

    CopyGmToUbInput copyGmToUbInput;
    CopyGmToUbMask copyGmToUbMask;
    CopyUbToGmOutput copyUbToGmOutput;
};

}  // namespace Catlass::Epilogue::Block

#endif  // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FA_SOFTMAX_HPP