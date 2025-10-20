/*
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_MLA_FD_RESCALE_O_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_MLA_FD_RESCALE_O_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Epilogue::Block {

template <
    class OutputType_,
    class InputType_,
    uint32_t ComputeEleNum_>
class BlockEpilogue<
    EpilogueAtlasA2MLAFDRescaleO<ComputeEleNum_>,
    OutputType_,
    InputType_>
{
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasA2MLAFDRescaleO<ComputeEleNum_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element;
    using ElementInput = typename InputType_::Element;
    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = typename InputType_::Layout;

    static constexpr uint32_t KV_SPLIT_MAX = DispatchPolicy::KV_SPLIT_MAX;
    static constexpr uint32_t HEADS_PROCESS_MAX = DispatchPolicy::HEADS_PROCESS_MAX;
    static constexpr uint32_t COMPUTE_ELE_NUM = DispatchPolicy::COMPUTE_ELE_NUM;
    static constexpr uint32_t FLOAT_ELENUM_PER_VECCALC = 64;
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    static constexpr uint32_t STAGES = 2;

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource, uint32_t kvSplitCoreNum_)
    {
        kvSplitCoreNum = kvSplitCoreNum_;

        uint32_t ubOffset = 0;
        oIn[0] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += COMPUTE_ELE_NUM * sizeof(float);
        oIn[1] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += COMPUTE_ELE_NUM * sizeof(float);
        oTemp[0] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += COMPUTE_ELE_NUM * sizeof(float);
        oTemp[1] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += COMPUTE_ELE_NUM * sizeof(float);
        oSum = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += COMPUTE_ELE_NUM * sizeof(float);
        out = resource.ubBuf.template GetBufferByByte<ElementOutput>(ubOffset);
        ubOffset += COMPUTE_ELE_NUM * sizeof(ElementOutput);
        lIn = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += KV_SPLIT_MAX * HEADS_PROCESS_MAX * sizeof(float);
        lExp = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += KV_SPLIT_MAX * HEADS_PROCESS_MAX * sizeof(float);
        lMax = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += HEADS_PROCESS_MAX * sizeof(float);
        lSum = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += HEADS_PROCESS_MAX * sizeof(float);
        lBrcb[0] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += HEADS_PROCESS_MAX * FLOAT_BLOCK_SIZE * sizeof(float);
        lBrcb[1] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
    }
    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
    }
    
    CATLASS_DEVICE
    void SetMask(int32_t len)
    {
        constexpr int32_t MAX_MASK_LEN = 128;
        constexpr int32_t HALF_MASK_LEN = 64;
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
    void operator()(
        AscendC::GlobalTensor<ElementOutput> gOutput,
        AscendC::GlobalTensor<ElementInput> gOCoreTmp,
        AscendC::GlobalTensor<ElementInput> gl,
        uint32_t actualHeads, uint32_t headsProcess, uint32_t headSize)
    {
        uint32_t kvSplitRound = (kvSplitCoreNum + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE * FLOAT_BLOCK_SIZE;

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::DataCopyPad(
            lIn, gl,
            AscendC::DataCopyExtParams(
                actualHeads, kvSplitCoreNum * sizeof(ElementInput), 0,
                (KV_SPLIT_MAX - kvSplitCoreNum) / FLOAT_BLOCK_SIZE, 0),
            AscendC::DataCopyPadExtParams<ElementInput>(false, 0, 0, 0));

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);

        SetMask(kvSplitCoreNum);
        AscendC::WholeReduceMax<float, false>(
            lMax, lIn, (int32_t)0, actualHeads, 1, 1, 8,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t i = 0; i < kvSplitRound / FLOAT_BLOCK_SIZE; i++) {
            AscendC::Brcb(
                lExp[i * FLOAT_BLOCK_SIZE],
                lMax,
                (headsProcess + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE,
                AscendC::BrcbRepeatParams(KV_SPLIT_MAX / FLOAT_BLOCK_SIZE,
                                          8 * KV_SPLIT_MAX / FLOAT_BLOCK_SIZE));
        }
        AscendC::PipeBarrier<PIPE_V>();

        SetMask(kvSplitCoreNum);
        AscendC::Sub<float, false>(
            lExp,
            lIn,
            lExp,
            (uint64_t)0,
            actualHeads,
            AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Exp<float, false>(
            lExp,
            lExp,
            (uint64_t)0,
            actualHeads,
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::RepeatReduceSum<float, false>(lSum, lExp, actualHeads, 0, 0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Ln(lSum, lSum, (headsProcess + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE * FLOAT_BLOCK_SIZE);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Add(lSum, lSum, lMax, (headsProcess + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE * FLOAT_BLOCK_SIZE);
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t i = 0; i < kvSplitRound / FLOAT_BLOCK_SIZE; i++) {
            AscendC::Brcb(
                lExp[i * FLOAT_BLOCK_SIZE],
                lSum,
                (headsProcess + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE,
                AscendC::BrcbRepeatParams(KV_SPLIT_MAX / FLOAT_BLOCK_SIZE,
                                          8 * KV_SPLIT_MAX / FLOAT_BLOCK_SIZE));
        }
        AscendC::PipeBarrier<PIPE_V>();

        SetMask(kvSplitCoreNum);
        AscendC::Sub<float, false>(
            lExp,
            lIn,
            lExp,
            (uint64_t)0,
            actualHeads,
            AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);

        AscendC::Exp<float, false>(
            lExp,
            lExp,
            (uint64_t)0,
            actualHeads,
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();

        // preload
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::DataCopyPad(
            oIn[0], gOCoreTmp,
            AscendC::DataCopyExtParams(
                actualHeads, headSize * sizeof(ElementInput),
                (kvSplitCoreNum * headSize - headSize) * sizeof(ElementInput), 0, 0),
            AscendC::DataCopyPadExtParams<ElementInput>(false, 0, 0, 0));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);

        SetMask(FLOAT_ELENUM_PER_VECCALC);
        uint32_t bufferId = 0;
        for (uint32_t i = 0; i < kvSplitCoreNum; i++) {
            // load next o
            if (i < kvSplitCoreNum - 1) {
                uint32_t nextBufferId = 1 - bufferId;
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(oInEventList[nextBufferId]);
                AscendC::DataCopyPad(
                    oIn[nextBufferId], gOCoreTmp[(i + 1) * headSize],
                    AscendC::DataCopyExtParams(
                        actualHeads, headSize * sizeof(ElementInput),
                        (kvSplitCoreNum * headSize - headSize) * sizeof(ElementInput), 0, 0),
                    AscendC::DataCopyPadExtParams<ElementInput>(false, 0, 0, 0));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(oInEventList[nextBufferId]);
            }

            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t j = 0; j < actualHeads; j++) {
                float a = lExp[j * KV_SPLIT_MAX + i].GetValue(0);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(oTempEventList[bufferId]);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(oTempEventList[bufferId]);
                AscendC::Duplicate<float, false>(
                    lBrcb[bufferId][j * FLOAT_BLOCK_SIZE], a, uint64_t(0), 1, 0, 0);
            }
            AscendC::PipeBarrier<PIPE_V>();

            // caculate current o
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(oInEventList[bufferId]);
            uint32_t loops = (headSize + FLOAT_ELENUM_PER_VECCALC - 1) / FLOAT_ELENUM_PER_VECCALC;
            if (i > 0) {
                for (uint32_t j = 0; j < loops; j++) {
                    AscendC::Mul<float, false>(
                        oTemp[bufferId][j * FLOAT_ELENUM_PER_VECCALC], lBrcb[bufferId],
                        oIn[bufferId][j * FLOAT_ELENUM_PER_VECCALC], (uint64_t)0,
                        actualHeads,
                        AscendC::BinaryRepeatParams(
                            1, 0, 1, headSize / FLOAT_BLOCK_SIZE, 1,
                            headSize / FLOAT_BLOCK_SIZE));
                }
            } else {
                for (uint32_t j = 0; j < loops; j++) {
                    AscendC::Mul<float, false>(
                        oSum[j * FLOAT_ELENUM_PER_VECCALC], lBrcb[bufferId],
                        oIn[bufferId][j * FLOAT_ELENUM_PER_VECCALC], (uint64_t)0,
                        actualHeads,
                        AscendC::BinaryRepeatParams(
                            1, 0, 1, headSize / FLOAT_BLOCK_SIZE, 1,
                            headSize / FLOAT_BLOCK_SIZE));
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(oInEventList[bufferId]);

            if (i > 0) {
                AscendC::Add(oSum, oSum, oTemp[bufferId], actualHeads * headSize);
            }
            AscendC::PipeBarrier<PIPE_V>();
            bufferId = 1 - bufferId;
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        if (std::is_same<ElementOutput, bfloat16_t>::value) {
            AscendC::Cast(out, oSum, AscendC::RoundMode::CAST_RINT, actualHeads * headSize);
        } else {
            AscendC::Cast(out, oSum, AscendC::RoundMode::CAST_NONE, actualHeads * headSize);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(
            gOutput, out,
            AscendC::DataCopyExtParams(
                actualHeads, headSize * sizeof(ElementOutput), 0, 0, 0));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

private:
    uint32_t kvSplitCoreNum = 1;
    AscendC::LocalTensor<ElementOutput> out;
    AscendC::LocalTensor<float> oIn[STAGES];
    AscendC::LocalTensor<float> oTemp[STAGES];
    AscendC::LocalTensor<float> lBrcb[STAGES];
    AscendC::LocalTensor<float> oSum;
    AscendC::LocalTensor<float> lIn;
    AscendC::LocalTensor<float> lExp;
    AscendC::LocalTensor<float> lTrans;
    AscendC::LocalTensor<float> lMax;
    AscendC::LocalTensor<float> lSum;

    int32_t oTempEventList[STAGES] = {0, 1};
    int32_t oInEventList[STAGES] = {0, 1};
};
} // namespace Catlass::Epilogue::Block
#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_MLA_FD_RESCALE_O_HPP