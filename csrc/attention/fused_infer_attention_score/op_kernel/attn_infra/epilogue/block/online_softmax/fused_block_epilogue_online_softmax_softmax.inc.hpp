/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSED_BLOCK_EPILOGUE_ONLINE_SOFTMAX_SOFTMAX_INC_HPP
#define FUSED_BLOCK_EPILOGUE_ONLINE_SOFTMAX_SOFTMAX_INC_HPP

    __aicore__ inline
    void CalcLocalRowMax(uint32_t sUbOffset, uint32_t rowNumCurLoopRound, uint32_t columnNum, uint32_t columnNumRound,
        uint32_t rowOffset)
    {
        if (columnNum == 512) {
            RowmaxSPECTILE512(
                lsUbTensor[sUbOffset],
                lmUbTensor[rowOffset],
                tvUbTensor,
                rowNumCurLoopRound,
                columnNum,
                columnNumRound);
        } else if (columnNum == 256) {
            RowmaxSPECTILE256(
                lsUbTensor[sUbOffset],
                lmUbTensor[rowOffset],
                tvUbTensor,
                rowNumCurLoopRound,
                columnNum,
                columnNumRound);
        } else {
            RowmaxTAILTILE(
                lsUbTensor[sUbOffset],
                lmUbTensor[rowOffset],
                tvUbTensor,
                rowNumCurLoopRound,
                columnNum,
                columnNumRound);
        }
    }

    __aicore__ inline
    void UpdateGlobalRowMax(uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound, uint32_t columnNum,
        uint32_t columnNumRound, uint32_t dmUbOffsetCurCycle, uint32_t rowOffset, uint32_t isFirstStackTile)
    {
        if (isFirstStackTile) {
            AscendC::DataCopy(
                hmUbTensor[rowOffset],
                lmUbTensor[rowOffset],
                AscendC::DataCopyParams(1, rowNumCurLoopRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            SetVecMask(rowNumCurLoop);
            // *** hm = vmax(lm, gm)
            AscendC::Max<float, false>(
                hmUbTensor[rowOffset],
                lmUbTensor[rowOffset],
                gmUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // *** dm = gm - hm
            AscendC::Sub<float, false>(
                dmUbTensor[dmUbOffsetCurCycle],
                gmUbTensor[rowOffset],
                hmUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // *** dm = exp(dm)
            AscendC::Exp<float, false>(
                dmUbTensor[dmUbOffsetCurCycle],
                dmUbTensor[dmUbOffsetCurCycle],
                (uint64_t)0,
                1,
                AscendC::UnaryRepeatParams(1, 1, 8, 8));
        }
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        AscendC::PipeBarrier<PIPE_V>();
        // *** gm = hm
        AscendC::DataCopy(
            gmUbTensor[rowOffset],
            hmUbTensor[rowOffset],
            AscendC::DataCopyParams(1, rowNumCurLoopRound / FLOAT_BLOCK_SIZE, 0, 0));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline
    void UpdateGlobalRowMax(
        AscendC::GlobalTensor<ElementSink> gSink,
        uint32_t rowNumCurLoop,
        uint32_t rowNumCurLoopRound,
        uint32_t columnNum,
        uint32_t columnNumRound,
        uint32_t dmUbOffsetCurCycle,
        uint32_t rowOffset,
        uint32_t isFirstStackTile,
        bool isLastStackTile,
        SinkLoopParam &curLoop)
    {
        if (isFirstStackTile) {
            AscendC::DataCopy(
                hmUbTensor[rowOffset],
                lmUbTensor[rowOffset],
                AscendC::DataCopyParams(1, rowNumCurLoopRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::PipeBarrier<PIPE_V>();
            // hm = Maxs(hm, sink)
            if constexpr (SINK_MODE == SinkMode::ENABLE) {
                if (isLastStackTile) {
                    UpdateRowMaxWithSink(gSink, rowOffset, dmUbOffsetCurCycle, curLoop);
                }
            }
        } else {
            SetVecMask(rowNumCurLoop);
            // *** hm = vmax(lm, gm)
            AscendC::Max<float, false>(
                hmUbTensor[rowOffset],
                lmUbTensor[rowOffset],
                gmUbTensor[rowOffset],
                (uint64_t)0, 1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();

            // hm = Maxs(hm, sink)
            if constexpr (SINK_MODE == SinkMode::ENABLE) {
                if (isLastStackTile) {
                    UpdateRowMaxWithSink(gSink, rowOffset, dmUbOffsetCurCycle, curLoop);
                    SetVecMask(rowNumCurLoop);
                }
            }

            // *** dm = gm - hm
            AscendC::Sub<float, false>(
                dmUbTensor[dmUbOffsetCurCycle],
                gmUbTensor[rowOffset],
                hmUbTensor[rowOffset],
                (uint64_t)0, 1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // *** dm = exp(dm)
            AscendC::Exp<float, false>(
                dmUbTensor[dmUbOffsetCurCycle],
                dmUbTensor[dmUbOffsetCurCycle],
                (uint64_t)0, 1,
                AscendC::UnaryRepeatParams(1, 1, 8, 8));
        }
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        AscendC::PipeBarrier<PIPE_V>();

        // *** copy hm to gm
        AscendC::DataCopy(
            gmUbTensor[rowOffset],
            hmUbTensor[rowOffset],
            AscendC::DataCopyParams(1, rowNumCurLoopRound / FLOAT_BLOCK_SIZE, 0, 0));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline
    void CalcExp(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound, uint32_t columnNum,
        uint32_t columnNumRound, uint32_t rowOffset)
    {
        // *** hm_block = expand_to_block(hm), 存放于 tv
        AscendC::Brcb(
            tvUbTensor.template ReinterpretCast<uint32_t>(),
            hmUbTensor[rowOffset].template ReinterpretCast<uint32_t>(),
            rowNumCurLoopRound / FLOAT_BLOCK_SIZE,
            AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();
        // *** ls = ls - hm_block
        for (uint32_t subIdx = 0; subIdx < columnNum / FLOAT_VECTOR_SIZE; ++subIdx) {
            AscendC::Sub<float, false>(
                lsUbTensor[sUbOffset][subIdx * FLOAT_VECTOR_SIZE],
                lsUbTensor[sUbOffset][subIdx * FLOAT_VECTOR_SIZE],
                tvUbTensor,
                (uint64_t)0,
                rowNumCurLoop,
                AscendC::BinaryRepeatParams(
                    1, 1, 0, columnNumRound / FLOAT_BLOCK_SIZE, columnNumRound / FLOAT_BLOCK_SIZE, 1));
        }
        if (columnNum % FLOAT_VECTOR_SIZE > 0) {
            SetVecMask(columnNum % FLOAT_VECTOR_SIZE);
            AscendC::Sub<float, false>(
                lsUbTensor[sUbOffset][columnNum / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                lsUbTensor[sUbOffset][columnNum / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                tvUbTensor,
                (uint64_t)0,
                rowNumCurLoop,
                AscendC::BinaryRepeatParams(
                    1, 1, 0, columnNumRound / FLOAT_BLOCK_SIZE, columnNumRound / FLOAT_BLOCK_SIZE, 1));
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        AscendC::PipeBarrier<PIPE_V>();
        // *** ls = exp(ls)
        AscendC::Exp<float, false>(
            lsUbTensor[sUbOffset],
            lsUbTensor[sUbOffset],
            (uint64_t)0,
            NpuArch::Detail::Alignment::CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE),
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline
    void CalcLocalRowSum(uint32_t sUbOffset, uint32_t rowNumCurLoopRound, uint32_t columnNum, uint32_t columnNumRound,
        uint32_t rowOffset)
    {
        // *** ll = rowsum(ls32)
        if (columnNum == 512) {
            RowsumSPECTILE512(
                lsUbTensor[sUbOffset],
                llUbTensor[rowOffset],
                tvUbTensor,
                rowNumCurLoopRound,
                columnNum,
                columnNumRound);
        } else if (columnNum == 256) {
            RowsumSPECTILE256(
                lsUbTensor[sUbOffset],
                llUbTensor[rowOffset],
                tvUbTensor,
                rowNumCurLoopRound,
                columnNum,
                columnNumRound);
        } else {
            RowsumTAILTILE(
                lsUbTensor[sUbOffset],
                llUbTensor[rowOffset],
                tvUbTensor,
                rowNumCurLoopRound,
                columnNum,
                columnNumRound);
        }
    }

    __aicore__ inline
    void UpdateGlobalRowSum(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound,
        uint32_t dmUbOffsetCurCycle, uint32_t rowOffset, uint32_t isFirstStackTile)
    {
        if (isFirstStackTile) {
            // *** copy ll to g1
            AscendC::DataCopy(
                glUbTensor[rowOffset],
                llUbTensor[rowOffset],
                AscendC::DataCopyParams(1, rowNumCurLoopRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            SetVecMask(rowNumCurLoop);
            // *** gl = dm * gl
            AscendC::Mul<float, false>(
                glUbTensor[rowOffset],
                dmUbTensor[dmUbOffsetCurCycle],
                glUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // *** gl = ll + gl
            AscendC::Add<float, false>(
                glUbTensor[rowOffset],
                glUbTensor[rowOffset],
                llUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetVectorMask<int8_t>((uint64_t)(-1), (uint64_t)(-1));
        }
    }

    __aicore__ inline
    void UpdateGlobalRowSum(
        AscendC::GlobalTensor<ElementSink> gSink,
        uint32_t sUbOffset,
        uint32_t rowNumCurLoop,
        uint32_t rowNumCurLoopRound,
        uint32_t dmUbOffsetCurCycle,
        uint32_t rowOffset,
        uint32_t isFirstStackTile,
        bool isLastStackTile,
        SinkLoopParam &curLoop)
    {
        if (isFirstStackTile) {
            // *** copy ll to g1
            AscendC::DataCopy(
                glUbTensor[rowOffset],
                llUbTensor[rowOffset],
                AscendC::DataCopyParams(1, rowNumCurLoopRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            SetVecMask(rowNumCurLoop);
            // *** gl *= dm
            AscendC::Mul<float, false>(
                glUbTensor[rowOffset],
                dmUbTensor[dmUbOffsetCurCycle],
                glUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();

            // *** gl += ll
            AscendC::Add<float, false>(
                glUbTensor[rowOffset],
                glUbTensor[rowOffset],
                llUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }

        // gl = gl + exp(sink-lm)
        if constexpr (SINK_MODE == SinkMode::ENABLE) {
            if (isLastStackTile) {
                UpdateRowSumWithSink(rowOffset, curLoop.rowNumCurLoop);
            }
        }
    }

    __aicore__ inline
    void DownCastP(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        // *** lp = castfp32to16(ls)
        if (std::is_same<ElementOutput, bfloat16_t>::value) {
            AscendC::Cast<ElementOutput, float, false>(
                lpUbTensor[sUbOffset],
                lsUbTensor[sUbOffset],
                AscendC::RoundMode::CAST_RINT,
                (uint64_t)0,
                NpuArch::Detail::Alignment::CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE),
                AscendC::UnaryRepeatParams(1, 1, 4, 8));
        } else {
            AscendC::Cast<ElementOutput, float, false>(
                lpUbTensor[sUbOffset],
                lsUbTensor[sUbOffset],
                AscendC::RoundMode::CAST_NONE,
                (uint64_t)0,
                NpuArch::Detail::Alignment::CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE),
                AscendC::UnaryRepeatParams(1, 1, 4, 8));
        }
    }

    __aicore__ inline
    void CopyPUbToGm(AscendC::GlobalTensor<ElementOutput> gOutput, uint32_t sUbOffset, uint32_t rowNumCurLoop,
        uint32_t columnNumRound, uint32_t columnNumPad)
    {
        AscendC::DataCopy(
            gOutput,
            lpUbTensor[sUbOffset],
            AscendC::DataCopyParams(
                rowNumCurLoop, columnNumRound / BLOCK_SIZE, 0, (columnNumPad - columnNumRound) / BLOCK_SIZE));
    }

#endif // FUSED_BLOCK_EPILOGUE_ONLINE_SOFTMAX_SOFTMAX_INC_HPP
