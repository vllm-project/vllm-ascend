/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSED_BLOCK_EPILOGUE_ONLINE_SOFTMAX_SINK_INC_HPP
#define FUSED_BLOCK_EPILOGUE_ONLINE_SOFTMAX_SINK_INC_HPP

    __aicore__ inline
    float ConvertElementSinkToFloat(const ElementSink& rawSinkVal)
    {
        if constexpr (std::is_same_v<ElementSink, bfloat16_t>) {
            return AscendC::ToFloat(rawSinkVal);
        } else if constexpr (std::is_same_v<ElementSink, half>) {
            return static_cast<float>(rawSinkVal);
        }
    }

    __aicore__ inline
    void SetSinkVecMask(uint32_t elemStart, uint32_t elemEnd)
    {
        uint64_t mask = 0;
        uint64_t one = 1;
        if (elemStart < 0 || elemStart > elemEnd || elemEnd > FLOAT_VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>((uint64_t)0, (uint64_t)0);
            return;
        }

        for (uint32_t elemIdx = elemStart; elemIdx <= elemEnd; elemIdx++) {
            mask |= (one << elemIdx);
        }
        AscendC::SetVectorMask<int8_t>(0x0, mask);
    }

    __aicore__ inline
    void UpdateRowMaxWithSink(
        AscendC::GlobalTensor<ElementSink> gSink,
        uint32_t rowOffset,
        uint32_t dmUbOffsetCurCycle,
        SinkLoopParam &curLoop)
    {
        const uint32_t loopStart = curLoop.rowOffsetIoGm;
        const uint32_t loopEnd = curLoop.rowOffsetIoGm + curLoop.rowNumCurLoop - 1;
        const uint32_t qSBlockSize = curLoop.qSBlockSize;

        const uint32_t firstHeadId = loopStart / qSBlockSize;
        const uint32_t lastHeadId = loopEnd / qSBlockSize;

        SetVecMask(curLoop.rowNumCurLoop);
        float zeroNum = 0.0f;
        AscendC::Duplicate<float, false>(lmUbTensor[rowOffset], zeroNum, (uint64_t)0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);

        for (uint32_t headId = firstHeadId; headId <= lastHeadId; headId++) {
            uint32_t curHeadQsBlockStartGm = headId * qSBlockSize;
            uint32_t curHeadQsBlockEndGm = curHeadQsBlockStartGm + qSBlockSize - 1U;

            uint32_t headActualStartThisSubCore =
                AscendC::Std::max(curHeadQsBlockStartGm, loopStart) - curLoop.rowOffsetThisSubBlock - rowOffset;
            uint32_t headActualEndThisSubCore =
                AscendC::Std::min(curHeadQsBlockEndGm, loopEnd) - curLoop.rowOffsetThisSubBlock - rowOffset;

            float sinkValue = ConvertElementSinkToFloat(gSink.GetValue(headId));
            uint32_t headRowNumThisSubCore = headActualEndThisSubCore - headActualStartThisSubCore + 1;

            SetSinkVecMask(headActualStartThisSubCore, headActualEndThisSubCore);

            AscendC::UnaryRepeatParams maxsRepeatParams(1, 1, 8, 8);

            // hm = Maxs(hm, sink)
            AscendC::Maxs<float, false>(
                dmUbTensor[dmUbOffsetCurCycle],
                hmUbTensor[rowOffset],
                sinkValue,
                (uint64_t)0, 1,
                maxsRepeatParams);

            // sinkTensor
            AscendC::Adds<float, false>(
                lmUbTensor[rowOffset],
                lmUbTensor[rowOffset],
                sinkValue,
                (uint64_t)0, 1,
                maxsRepeatParams);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        uint64_t mask = static_cast<uint64_t>(curLoop.rowNumCurLoop);

        AscendC::CompareScalar(
            selMaskUbTensor,
            hmUbTensor[rowOffset],
            NEG_INF,
            AscendC::CMPMODE::EQ,
            mask,
            1,
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Select(
            hmUbTensor[rowOffset],
            selMaskUbTensor,
            hmUbTensor[rowOffset],
            dmUbTensor[dmUbOffsetCurCycle],
            AscendC::SELMODE::VSEL_CMPMASK_SPR,
            mask,
            1,
            AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline
    void UpdateRowSumWithSink(uint32_t rowOffset, uint32_t rowNumCurLoop)
    {
        SetVecMask(rowNumCurLoop);
        // sink = sink - hm
        AscendC::Sub<float, false>(
            lmUbTensor[rowOffset],
            lmUbTensor[rowOffset],
            hmUbTensor[rowOffset],
            (uint64_t)0, 1,
            AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();

        // exp(sink-hm)
        AscendC::Exp<float, false>(
            lmUbTensor[rowOffset],
            lmUbTensor[rowOffset],
            (uint64_t)0,
            1,
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();

        // gl+exp(sink -m)
        AscendC::Add<float, false>(
            glUbTensor[rowOffset],
            glUbTensor[rowOffset],
            lmUbTensor[rowOffset],
            (uint64_t)0, 1,
            AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }

#endif // FUSED_BLOCK_EPILOGUE_ONLINE_SOFTMAX_SINK_INC_HPP
