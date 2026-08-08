/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSED_BLOCK_EPILOGUE_ONLINE_SOFTMAX_OPERATOR_INC_HPP
#define FUSED_BLOCK_EPILOGUE_ONLINE_SOFTMAX_OPERATOR_INC_HPP

    __aicore__ inline
    void operator()(AscendC::GlobalTensor<ElementOutput> gOutputBase,
                    AscendC::GlobalTensor<ElementInput> gInputBase,
                    const LayoutOutput &layoutOutput,
                    const LayoutInput &layoutInput,
                    GemmCoord actualBlockShape,
                    uint32_t isFirstStackTile,
                    uint32_t isLastNoMaskStackTile,
                    uint32_t qSBlockSize,
                    uint32_t qNBlockSize,
                    uint32_t curStackTileMod,
                    uint32_t kvNBlockSize,
                    uint64_t gmOffsetSBase,
                    uint64_t gmOffsetPBase,
                    Arch::CrossCoreFlag qkReady,
                    Arch::CrossCoreFlag softmaxReady)
    {
        Arch::CrossCoreWaitFlag(qkReady);
        uint32_t rowNum = actualBlockShape.m();
        uint32_t columnNum = actualBlockShape.n();
        uint32_t columnNumRound = NpuArch::Detail::Alignment::RoundUp(columnNum, BLOCK_SIZE);
        uint32_t columnNumPad = layoutInput.stride(0);

        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        uint32_t kvNSplitSubBlock = kvNBlockSize / subBlockNum;
        uint32_t kvNThisSubBlock = (kvNBlockSize == 1U) ? 0
            : (subBlockIdx == 1U) ? (kvNBlockSize - kvNSplitSubBlock) : kvNSplitSubBlock;

        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        uint32_t qNThisSubBlock = (kvNBlockSize == 1U) ?
            ((qNBlockSize == 1U) ? 0
                                : (subBlockIdx == 1U) ? (qNBlockSize - qNSplitSubBlock)
                                                    : qNSplitSubBlock)
            : (kvNThisSubBlock * qNBlockSize);

        uint32_t rowSplitSubBlock = (kvNBlockSize == 1U) ?
            ((qNBlockSize == 1U) ? (qSBlockSize / subBlockNum) : (qSBlockSize * qNSplitSubBlock)) :
            (qSBlockSize * qNBlockSize * kvNSplitSubBlock);
        uint32_t rowActualThisSubBlock = (subBlockIdx == 1) ? (rowNum - rowSplitSubBlock) : rowSplitSubBlock;
        uint32_t rowOffsetThisSubBlock = subBlockIdx * rowSplitSubBlock;
        uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
        uint32_t rowNumTile = NpuArch::Detail::Alignment::RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);
        rowNumTile = AscendC::Std::min(rowNumTile, FLOAT_VECTOR_SIZE);
        uint32_t rowLoopNum = NpuArch::Detail::Alignment::CeilDiv(rowActualThisSubBlock, rowNumTile);
        uint32_t preLoad = 1;

        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum + preLoad; rowLoopIdx++) {
            if (rowLoopIdx < rowLoopNum) {
                uint32_t pingpongFlag = rowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop = (rowLoopIdx == rowLoopNum - 1) ?
                    (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gInputCurLoop = gInputBase[offsetInput];

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);

                CopySGmToUb(
                    gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound, columnNumPad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
            }
            if (rowLoopIdx >= preLoad) {
                uint32_t delayedRowLoopIdx = rowLoopIdx - preLoad;
                uint32_t pingpongFlag = delayedRowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = delayedRowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop =
                    (delayedRowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gOutputCurLoop = gOutputBase[offsetOutput];
                auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, columnNum));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);

                // add sink

                ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                SubCoreCompute<false>(
                    gOutputCurLoop,
                    layoutOutputCurLoop,
                    rowOffsetCurLoop,
                    isFirstStackTile,
                    isLastNoMaskStackTile,
                    (delayedRowLoopIdx == 0),
                    (delayedRowLoopIdx == rowLoopNum - 1),
                    columnNumRound,
                    pingpongFlag,
                    curStackTileMod
                    );
            }
        }
    }
    __aicore__ inline
    void operator()(
        AscendC::GlobalTensor<ElementOutput> gOutput,
        AscendC::GlobalTensor<ElementInput> gInput,
        AscendC::GlobalTensor<ElementSink> gSink,
        const LayoutOutput &layoutOutput,
        const LayoutInput &layoutInput,
        GemmCoord actualBlockShape,
        uint32_t isFirstStackTile,
        uint32_t isLastNoMaskStackTile,
        uint32_t qSBlockSize,
        uint32_t qNBlockSize,
        uint32_t curStackTileMod,
        bool isLastStackTile,
        bool isSplitKV = false,
        bool startsWithMaskTile = false,
        bool startsWithMaskThenNomaskFlag = false)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t columnNum = actualBlockShape.n();
        uint32_t columnNumRound = NpuArch::Detail::Alignment::RoundUp(columnNum, BLOCK_SIZE);
        uint32_t columnNumPad = layoutInput.stride(0);

        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        uint32_t qNThisSubBlock = (qNBlockSize == 1) ?
            0 : (subBlockIdx == 1) ? (qNBlockSize - qNSplitSubBlock) : qNSplitSubBlock;
        uint32_t rowSplitSubBlock = (qNBlockSize == 1) ?
            (qSBlockSize / 2) : (qSBlockSize * qNSplitSubBlock);
        uint32_t rowActualThisSubBlock = (subBlockIdx == 1) ? (rowNum - rowSplitSubBlock) : rowSplitSubBlock;
        uint32_t rowOffsetThisSubBlock = subBlockIdx * rowSplitSubBlock;
        uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
        uint32_t rowNumTile = NpuArch::Detail::Alignment::RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);
        rowNumTile = AscendC::Std::min(rowNumTile, FLOAT_VECTOR_SIZE);
        uint32_t rowLoopNum = NpuArch::Detail::Alignment::CeilDiv(rowActualThisSubBlock, rowNumTile);
        uint32_t preLoad = 1;

        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum + preLoad; rowLoopIdx++) {
            if (rowLoopIdx < rowLoopNum) {
                uint32_t pingpongFlag = rowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop = (rowLoopIdx == rowLoopNum - 1) ?
                    (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gInputCurLoop = gInput[offsetInput];

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);
                if (startsWithMaskTile && rowLoopIdx == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                }
                CopySGmToUb(
                    gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound, columnNumPad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
            }
            if (rowLoopIdx >= preLoad) {
                uint32_t delayedRowLoopIdx = rowLoopIdx - preLoad;
                uint32_t rowOffsetCurLoop = delayedRowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop =
                    (delayedRowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;
                uint32_t pingpongFlag = delayedRowLoopIdx % 2;

                int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gOutputCurLoop = gOutput[offsetOutput];
                auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, columnNum));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);

                // add sink
                SinkLoopParam curSinkLoop(rowOffsetIoGm, rowNumCurLoop, qSBlockSize, rowOffsetThisSubBlock);

                ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                SubCoreCompute<false>(
                    gOutputCurLoop,
                    gSink,
                    layoutOutputCurLoop,
                    rowOffsetCurLoop,
                    isFirstStackTile,
                    isLastNoMaskStackTile,
                    (delayedRowLoopIdx == 0),
                    (delayedRowLoopIdx == rowLoopNum - 1),
                    columnNumRound,
                    pingpongFlag,
                    curStackTileMod,
                    curSinkLoop,
                    isLastStackTile,
                    isSplitKV,
                    startsWithMaskThenNomaskFlag);
            }
        }
    }

    __aicore__ inline
    void operator()(
        AscendC::GlobalTensor<ElementOutput> gOutput,
        AscendC::GlobalTensor<ElementInput> gInput,
        AscendC::GlobalTensor<ElementSink> gSink,
        AscendC::GlobalTensor<ElementMask> gMask,
        const LayoutOutput &layoutOutput,
        const LayoutInput &layoutInput,
        const LayoutInput &layoutMask,
        GemmCoord actualBlockShape,
        uint32_t isFirstStackTile,
        uint32_t qSBlockSize,
        uint32_t qNBlockSize,
        uint32_t curStackTileMod,
        Arch::CrossCoreFlag qkReady,
        uint32_t triUp,
        uint32_t triDown,
        uint32_t kvSStartIdx,
        uint32_t kvSEndIdx,
        bool isLastStackTile,
        bool isSplitKV = false)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t columnNum = actualBlockShape.n();
        uint32_t columnNumRound = NpuArch::Detail::Alignment::RoundUp(columnNum, BLOCK_SIZE_IN_BYTE);
        uint32_t columnNumPad = layoutInput.stride(0);
        uint32_t maskStride = layoutMask.stride(0);
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        uint32_t qNThisSubBlock = (qNBlockSize == 1) ?
            0 : (subBlockIdx == 1) ? (qNBlockSize - qNSplitSubBlock) : qNSplitSubBlock;
        uint32_t rowSplitSubBlock = (qNBlockSize == 1) ?
            (qSBlockSize / 2) : (qSBlockSize * qNSplitSubBlock);
        uint32_t rowActualThisSubBlock = (subBlockIdx == 1) ?
            (rowNum - rowSplitSubBlock) : rowSplitSubBlock;
        uint32_t rowOffsetThisSubBlock = subBlockIdx * rowSplitSubBlock;

        uint32_t tokenNumPerHeadThisSubBlock = Min(qSBlockSize, rowActualThisSubBlock);
        uint32_t maskOffsetThisSubBlock = (qNBlockSize == 1) ?
            rowOffsetThisSubBlock : 0;

        // calc mask shift in gm
        uint32_t gmOffsetMaskRow;
        uint32_t gmOffsetMaskColumn;
        uint32_t maskColumn;
        uint32_t addMaskUbOffset;
        if (triUp >= kvSStartIdx) {
            uint32_t triUpRoundDown = NpuArch::Detail::Alignment::RoundDown(triUp, BLOCK_SIZE_IN_BYTE);
            gmOffsetMaskRow = triUp - triUpRoundDown;
            gmOffsetMaskColumn = 0;
            maskColumn = kvSEndIdx - triUpRoundDown;
            addMaskUbOffset = triUpRoundDown - kvSStartIdx;
        } else {
            gmOffsetMaskRow = 0;
            gmOffsetMaskColumn = kvSStartIdx - triUp;
            maskColumn = columnNum;
            addMaskUbOffset = 0;
        }
        uint32_t maskColumnRound = NpuArch::Detail::Alignment::RoundUp(maskColumn, BLOCK_SIZE_IN_BYTE);

        int64_t offsetMask =
            layoutMask.GetOffset(MatrixCoord(gmOffsetMaskRow + maskOffsetThisSubBlock, gmOffsetMaskColumn));
        auto gMaskThisSubBlock = gMask[offsetMask];
        auto layoutMaskThisSubBlock = layoutMask;

        uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
        uint32_t rowNumTile = NpuArch::Detail::Alignment::RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);
        rowNumTile = AscendC::Std::min(rowNumTile, FLOAT_VECTOR_SIZE);
        uint32_t rowLoopNum = NpuArch::Detail::Alignment::CeilDiv(rowActualThisSubBlock, rowNumTile);
        uint32_t preLoad = 1;

        if (rowActualThisSubBlock == 0) {
            Arch::CrossCoreWaitFlag(qkReady);
            return;
        }

        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum + preLoad; rowLoopIdx++) {
            if (rowLoopIdx < rowLoopNum) {
                uint32_t pingpongFlag = rowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop = (rowLoopIdx == rowLoopNum - 1) ?
                    (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;
                // loop 0 mask load before cross core sync
                if (rowLoopIdx == 0) {
                    // the token idx of the start token of the prologue part
                    uint32_t proTokenIdx = rowOffsetCurLoop % tokenNumPerHeadThisSubBlock;
                    // the token num of the prologue part
                    uint32_t proTokenNum =
                        Min(rowNumCurLoop, (tokenNumPerHeadThisSubBlock - proTokenIdx)) % tokenNumPerHeadThisSubBlock;
                    // the token num of the epilogue part
                    uint32_t integralHeadNum = (rowNumCurLoop - proTokenNum) / tokenNumPerHeadThisSubBlock;
                    // the number of integral heads within a cycle
                    uint32_t epiTokenNum = rowNumCurLoop - proTokenNum - integralHeadNum * tokenNumPerHeadThisSubBlock;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    CopyMaskGmToUb(
                        gMaskThisSubBlock,
                        maskColumn, maskColumnRound, maskStride,
                        tokenNumPerHeadThisSubBlock,
                        proTokenIdx, proTokenNum, integralHeadNum, epiTokenNum, false);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                    Arch::CrossCoreWaitFlag(qkReady);
                }
                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gInputCurLoop = gInput[offsetInput];
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);
                CopySGmToUb(
                    gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound, columnNumPad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
            }
            if (rowLoopIdx >= preLoad) {
                uint32_t delayedRowLoopIdx = rowLoopIdx - preLoad;
                uint32_t pingpongFlag = delayedRowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = delayedRowLoopIdx * rowNumTile;
                uint32_t rowNumCurLoop = (delayedRowLoopIdx == rowLoopNum - 1) ?
                    (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                UpCastMask<half, ElementMask>(maskUbTensor16, maskUbTensor, rowNumCurLoop, columnNumRound);
                UpCastMask<float, half>(maskUbTensor32, maskUbTensor16, rowNumCurLoop, columnNumRound);

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
                ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                ApplyMask(
                    (pingpongFlag * MAX_UB_S_ELEM_NUM),
                    rowNumCurLoop, columnNumRound,
                    maskColumnRound, addMaskUbOffset);
                // next loop mask load
                if (rowLoopIdx < rowLoopNum) {
                    uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                    uint32_t rowNumCurLoop =
                        (rowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;
                    // the token idx of the start token of the prologue part
                    uint32_t proTokenIndex = rowOffsetCurLoop % tokenNumPerHeadThisSubBlock;
                    // the token num of the prologue part
                    uint32_t proTokenNum =
                        Min(rowNumCurLoop, (tokenNumPerHeadThisSubBlock - proTokenIndex)) % tokenNumPerHeadThisSubBlock;
                    // the number of integral heads within a cycle
                    uint32_t integralHeadNum = (rowNumCurLoop - proTokenNum) / tokenNumPerHeadThisSubBlock;
                    // the token num of the epilogue part
                    uint32_t epiTokenNum = rowNumCurLoop - proTokenNum - integralHeadNum * tokenNumPerHeadThisSubBlock;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    CopyMaskGmToUb(
                        gMaskThisSubBlock,
                        maskColumn, maskColumnRound, maskStride,
                        tokenNumPerHeadThisSubBlock,
                        proTokenIndex, proTokenNum, integralHeadNum, epiTokenNum, false);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                }
                // online softmax vectorized compute

                // add sink
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                SinkLoopParam curSinkLoop(rowOffsetIoGm, rowNumCurLoop, qSBlockSize, rowOffsetThisSubBlock);

                int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gOutputCurLoop = gOutput[offsetOutput];
                auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, columnNum));
                SubCoreCompute<true>(
                    gOutputCurLoop,
                    gSink,
                    layoutOutputCurLoop,
                    rowOffsetCurLoop,
                    isFirstStackTile,
                    0,
                    (delayedRowLoopIdx == 0),
                    (delayedRowLoopIdx == rowLoopNum - 1),
                    columnNumRound,
                    pingpongFlag,
                    curStackTileMod,
                    curSinkLoop,
                    isLastStackTile,
                    isSplitKV,
                    false);
            }
        }
    }

    __aicore__ inline
    void operator()(
        AscendC::GlobalTensor<ElementOutput> gOutput,
        AscendC::GlobalTensor<ElementInput> gInput,
        AscendC::GlobalTensor<ElementSink> gSink,
        AscendC::GlobalTensor<ElementMask> gMask,
        const LayoutOutput &layoutOutput,
        const LayoutInput &layoutInput,
        const LayoutInput &layoutMask,
        GemmCoord actualBlockShape,
        uint32_t isFirstStackTile,
        uint32_t qSBlockSize,
        uint32_t qNBlockSize,
        uint32_t curStackTileMod,
        Arch::CrossCoreFlag qkReady,
        int32_t kvSStartIdx,
        bool doTriUPreMask,
        bool doTriUNextMask,
        int32_t preTokenStartLen,
        int32_t preTokenEndLen,
        int32_t nextTokenStartLen,
        int32_t nextTokenEndLen,
        bool isLastStackTile)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t columnNum = actualBlockShape.n();
        uint32_t columnNumRound = NpuArch::Detail::Alignment::RoundUp(columnNum, BLOCK_SIZE_IN_BYTE);
        uint32_t columnNumPad = layoutInput.stride(0);
        uint32_t maskStride = layoutMask.stride(0);
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();

        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        uint32_t qNThisSubBlock = (qNBlockSize == 1) ?
            0 : (subBlockIdx == 1) ? (qNBlockSize - qNSplitSubBlock) : qNSplitSubBlock;
        uint32_t rowSplitSubBlock = (qNBlockSize == 1) ?
            (qSBlockSize / 2) : (qSBlockSize * qNSplitSubBlock);
        uint32_t rowOffsetThisSubBlock = subBlockIdx * rowSplitSubBlock;
        uint32_t rowActualThisSubBlock = (subBlockIdx == 1) ?
            (rowNum - rowSplitSubBlock) : rowSplitSubBlock;

        uint32_t tokenNumPerHeadThisSubBlock = Min(qSBlockSize, rowActualThisSubBlock);
        uint32_t maskOffsetThisSubBlock = (qNBlockSize == 1) ?
            rowOffsetThisSubBlock : 0;

        uint32_t addMaskUbOffset = 0;

        uint32_t gmOffsetMaskRowPre;
        uint32_t gmOffsetMaskColumnPre;
        uint32_t maskColumnPre;

        uint32_t gmOffsetMaskRowNext;
        uint32_t gmOffsetMaskColumnNext;
        uint32_t maskColumnNext;
        if (doTriUPreMask) {
            if (preTokenStartLen > kvSStartIdx) {
                gmOffsetMaskRowPre = preTokenStartLen - kvSStartIdx - 1;
                gmOffsetMaskColumnPre = 0;
                maskColumnPre = columnNumRound;
            } else {
                gmOffsetMaskRowPre = 0;
                gmOffsetMaskColumnPre = kvSStartIdx - preTokenStartLen + 1;
                maskColumnPre = columnNumRound;
            }
        }
        if (doTriUNextMask) {
            if (nextTokenStartLen > kvSStartIdx) {
                gmOffsetMaskRowNext = nextTokenStartLen - kvSStartIdx;
                gmOffsetMaskColumnNext = 0;
                maskColumnNext = columnNumRound;
            } else {
                gmOffsetMaskRowNext = 0;
                gmOffsetMaskColumnNext = kvSStartIdx - nextTokenStartLen;
                maskColumnNext = columnNumRound;
            }
        }
        uint32_t columnNumRoundPre = NpuArch::Detail::Alignment::RoundUp(maskColumnPre, BLOCK_SIZE_IN_BYTE);
        int64_t offsetMaskPre = layoutMask.GetOffset(
            MatrixCoord(gmOffsetMaskRowPre + maskOffsetThisSubBlock, gmOffsetMaskColumnPre));
        auto gMaskThisSubBlockPre = gMask[offsetMaskPre];

        uint32_t columnNumRoundNext = NpuArch::Detail::Alignment::RoundUp(maskColumnNext, BLOCK_SIZE_IN_BYTE);
        int64_t offsetMaskNext = layoutMask.GetOffset(
            MatrixCoord(gmOffsetMaskRowNext + maskOffsetThisSubBlock, gmOffsetMaskColumnNext));
        auto gMaskThisSubBlockNext = gMask[offsetMaskNext];
        uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
        uint32_t rowNumTile = NpuArch::Detail::Alignment::RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);
        rowNumTile = AscendC::Std::min(rowNumTile, FLOAT_VECTOR_SIZE);
        uint32_t rowLoopNum = NpuArch::Detail::Alignment::CeilDiv(rowActualThisSubBlock, rowNumTile);

        if (rowActualThisSubBlock == 0) {
            Arch::CrossCoreWaitFlag(qkReady);
            return;
        }

        uint32_t preLoad = 1;
        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum + preLoad; rowLoopIdx++) {
            if (rowLoopIdx < rowLoopNum) {
                uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop = (rowLoopIdx == rowLoopNum - 1) ?
                    (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;
                uint32_t pingpongFlag = rowLoopIdx % 2;
                // loop 0 mask load before cross core sync
                if (rowLoopIdx == 0) {
                    // the token idx of the start token of the prologue part
                    uint32_t proTokenIdx = rowOffsetCurLoop % tokenNumPerHeadThisSubBlock;
                    // the token num of the prologue part
                    uint32_t proTokenNum =
                        Min(rowNumCurLoop, (tokenNumPerHeadThisSubBlock - proTokenIdx)) % tokenNumPerHeadThisSubBlock;
                    // the token num of the epilogue part
                    uint32_t integralHeadNum = (rowNumCurLoop - proTokenNum) / tokenNumPerHeadThisSubBlock;
                    // the number of integral heads within a cycle
                    uint32_t epiTokenNum = rowNumCurLoop - proTokenNum - integralHeadNum * tokenNumPerHeadThisSubBlock;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    if (doTriUPreMask && doTriUNextMask) {
                        CopyMaskGmToUb(
                            gMaskThisSubBlockPre,
                            maskColumnPre, columnNumRoundPre, maskStride,
                            tokenNumPerHeadThisSubBlock,
                            proTokenIdx, proTokenNum, integralHeadNum, epiTokenNum, false);
                    } else if (doTriUPreMask) {
                        CopyMaskGmToUb(
                            gMaskThisSubBlockPre,
                            maskColumnPre, columnNumRoundPre, maskStride,
                            tokenNumPerHeadThisSubBlock,
                            proTokenIdx, proTokenNum, integralHeadNum, epiTokenNum, false);
                    } else if (doTriUNextMask) {
                        CopyMaskGmToUb(
                            gMaskThisSubBlockNext,
                            maskColumnNext, columnNumRoundNext, maskStride,
                            tokenNumPerHeadThisSubBlock,
                            proTokenIdx, proTokenNum, integralHeadNum, epiTokenNum, true);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                    Arch::CrossCoreWaitFlag(qkReady);
                }
                int64_t offsetInputTemp = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gInputCurLoop = gInput[offsetInputTemp];
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);
                CopySGmToUb(
                    gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound, columnNumPad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
            }
            if (rowLoopIdx >= preLoad) {
                uint32_t delayedRowLoopIdx = rowLoopIdx - preLoad;
                uint32_t rowOffsetCurLoop = delayedRowLoopIdx * rowNumTile;
                uint32_t pingpongFlag = delayedRowLoopIdx % 2;
                uint32_t rowNumCurLoop = (delayedRowLoopIdx == rowLoopNum - 1) ?
                    (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                if (doTriUPreMask && doTriUNextMask)  {
                    OperatePreMaskUb(rowNumCurLoop, columnNumRound);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
                    ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                    ApplyMask(
                        (pingpongFlag * MAX_UB_S_ELEM_NUM),
                        rowNumCurLoop, columnNumRound,
                        columnNumRoundPre, addMaskUbOffset);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID6);
                    if (doTriUNextMask) {
                        uint32_t proTokenIdx = rowOffsetCurLoop % tokenNumPerHeadThisSubBlock;
                        // the token num of the prologue part
                        uint32_t proTokenNum = Min(
                            rowNumCurLoop,
                            (tokenNumPerHeadThisSubBlock - proTokenIdx)) % tokenNumPerHeadThisSubBlock;
                        // the token num of the epilogue part
                        uint32_t integralHeadNum = (rowNumCurLoop - proTokenNum) / tokenNumPerHeadThisSubBlock;
                        // the number of integral heads within a cycle
                        uint32_t epiTokenNum =
                            rowNumCurLoop - proTokenNum - integralHeadNum * tokenNumPerHeadThisSubBlock;
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID6);
                        CopyMaskGmToUb(
                            gMaskThisSubBlockNext,
                            maskColumnNext, columnNumRoundNext, maskStride,
                            tokenNumPerHeadThisSubBlock,
                            proTokenIdx, proTokenNum, integralHeadNum, epiTokenNum, true);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
                        OperateNextMaskUb(rowNumCurLoop, columnNumRound);
                        ApplyMask(
                            (pingpongFlag * MAX_UB_S_ELEM_NUM),
                            rowNumCurLoop, columnNumRound,
                            columnNumRoundNext, addMaskUbOffset);
                    }
                } else if (doTriUPreMask)  {
                    OperatePreMaskUb(rowNumCurLoop, columnNumRound);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
                    ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                    ApplyMask(
                        (pingpongFlag * MAX_UB_S_ELEM_NUM),
                        rowNumCurLoop, columnNumRound,
                        columnNumRoundPre, addMaskUbOffset);
                } else if (doTriUNextMask)  {
                    OperateNextMaskUb(rowNumCurLoop, columnNumRound);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
                    ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                    ApplyMask(
                        (pingpongFlag * MAX_UB_S_ELEM_NUM),
                        rowNumCurLoop, columnNumRound,
                        columnNumRoundNext, addMaskUbOffset);
                }
                // next loop mask load
                if (rowLoopIdx < rowLoopNum) {
                    uint32_t rowOffsetCurLoopTemp = rowLoopIdx * rowNumTile;
                    uint32_t rowNumCurLoop =
                        (rowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoopTemp) : rowNumTile;
                    // the token idx of the start token of the prologue part
                    uint32_t proTokenIdx = rowOffsetCurLoopTemp % tokenNumPerHeadThisSubBlock;
                    // the token num of the prologue part
                    uint32_t proTokenNum =
                        Min(rowNumCurLoop, (tokenNumPerHeadThisSubBlock - proTokenIdx)) % tokenNumPerHeadThisSubBlock;
                    // the number of integral heads within a cycle
                    uint32_t integralHeadNum = (rowNumCurLoop - proTokenNum) / tokenNumPerHeadThisSubBlock;
                    // the token num of the epilogue part
                    uint32_t epiTokenNumTemp = rowNumCurLoop - proTokenNum -
                        integralHeadNum * tokenNumPerHeadThisSubBlock;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    if (doTriUPreMask && doTriUNextMask) {
                        CopyMaskGmToUb(
                            gMaskThisSubBlockPre,
                            maskColumnPre, columnNumRoundPre, maskStride,
                            tokenNumPerHeadThisSubBlock,
                            proTokenIdx, proTokenNum, integralHeadNum, epiTokenNumTemp, false);
                    } else if (doTriUPreMask) {
                        CopyMaskGmToUb(
                            gMaskThisSubBlockPre,
                            maskColumnPre, columnNumRoundPre, maskStride,
                            tokenNumPerHeadThisSubBlock,
                            proTokenIdx, proTokenNum, integralHeadNum, epiTokenNumTemp, false);
                    } else if (doTriUNextMask) {
                        CopyMaskGmToUb(
                            gMaskThisSubBlockNext,
                            maskColumnNext, columnNumRoundNext, maskStride,
                            tokenNumPerHeadThisSubBlock,
                            proTokenIdx, proTokenNum, integralHeadNum, epiTokenNumTemp, true);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                }
                // online softmax vectorized compute
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                // add sink
                SinkLoopParam curSinkLoop(rowOffsetIoGm, rowNumCurLoop, qSBlockSize, rowOffsetThisSubBlock);
                int64_t offsetOutputSecond = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gOutputCurLoop = gOutput[offsetOutputSecond];
                auto layoutOutputCurLoopTemp = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, columnNum));
                SubCoreCompute<true>(
                    gOutputCurLoop,
                    gSink,
                    layoutOutputCurLoopTemp,
                    rowOffsetCurLoop,
                    isFirstStackTile,
                    0,
                    (delayedRowLoopIdx == 0),
                    (delayedRowLoopIdx == rowLoopNum - 1),
                    columnNumRound,
                    pingpongFlag,
                    curStackTileMod,
                    curSinkLoop,
                    isLastStackTile,
                    false,
                    false);
            }
        }
    }

    __aicore__ inline
    void operator()(
        AscendC::GlobalTensor<ElementOutput> gOutput,
        AscendC::GlobalTensor<ElementInput> gInput,
        AscendC::GlobalTensor<ElementSink> gSink,
        AscendC::GlobalTensor<ElementFull> gFull,
        const LayoutOutput &layoutOutput,
        const LayoutInput &layoutInput,
        const LayoutInput &layoutMask,
        GemmCoord actualBlockShape,
        uint32_t isFirstStackTile,
        uint32_t qSBlockSize,
        uint32_t qNBlockSize,
        uint32_t curStackTileMod,
        Arch::CrossCoreFlag qkReady,
        uint32_t rowOffer,
        uint32_t kvSStartIdx,
        uint32_t kvSEndIdx,
        uint32_t qNStartIdx,
        uint32_t BIdx,
        uint32_t qHeads,
        int64_t pseQ,
        int64_t pseKv,
        bool isLastStackTile)
    {
        uint32_t columnNum = actualBlockShape.n();  // fullmask块的总列数
        uint32_t columnNumRound = NpuArch::Detail::Alignment::RoundUp(columnNum, BLOCK_SIZE_IN_BYTE);  // 列数向上对齐到32B
        uint32_t rowNum = actualBlockShape.m();  // 当前fullmask块的总行数
        uint32_t columnNumPad = layoutInput.stride(0);  // 输入张量stride
        uint32_t maskStride = layoutMask.stride(0);  // 掩码张量
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();  // 当前向量核编号
        uint32_t subBlockNum = AscendC::GetSubBlockNum();  // 向量核总数
        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;  // 1 每核分到的头数
        uint32_t qNThisSubBlock = (qNBlockSize == 1) ?
            0 : (subBlockIdx == 1) ? (qNBlockSize - qNSplitSubBlock) : qNSplitSubBlock;  // 1 本核的头索引
        uint32_t rowSplitSubBlock = (qNBlockSize == 1) ?
            (qSBlockSize / 2) : (qSBlockSize * qNSplitSubBlock);  // 每核分到的行数
        uint32_t rowActualThisSubBlock = (subBlockIdx == 1) ?
            (rowNum - rowSplitSubBlock) : rowSplitSubBlock;  // 本核实际要处理的行数
        uint32_t rowOffsetThisSubBlock = subBlockIdx * rowSplitSubBlock;  // 本核子块在fullmask的行偏移
        uint32_t maskOffsetThisSubBlock = (qNBlockSize == 1) ? rowOffsetThisSubBlock : 0;
        uint32_t tokenNumPerHeadThisSubBlock = Min(qSBlockSize, rowActualThisSubBlock);
        // calc qNstartIdx per vec core
        uint32_t qNStartIdxVec = (qNBlockSize == 1) ?
            qNStartIdx : (subBlockIdx == 1) ? (qNStartIdx + qNSplitSubBlock) : qNStartIdx;
        // calc Full shift in gm
        int64_t offsetFull = 0;
        CalcGmFullShift(offsetFull, layoutMask, rowOffer, kvSStartIdx, maskOffsetThisSubBlock);
        uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
        uint32_t rowNumTile = NpuArch::Detail::Alignment::RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);
        rowNumTile = AscendC::Std::min(rowNumTile, FLOAT_VECTOR_SIZE);
        uint32_t rowLoopNum = NpuArch::Detail::Alignment::CeilDiv(rowActualThisSubBlock, rowNumTile);
        uint32_t preLoad = 1;

        if (rowActualThisSubBlock == 0) {
            Arch::CrossCoreWaitFlag(qkReady);
            return;
        }
        uint32_t proTokenIdx;
        uint32_t proTokenNum;
        uint32_t integralHeadNum;
        uint32_t epiTokenNum;
        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum + preLoad; rowLoopIdx++) {
            if (rowLoopIdx < rowLoopNum) {
                uint32_t pingpongFlag = rowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop = (rowLoopIdx == rowLoopNum - 1) ?
                    (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;
                // loop 0 mask load before cross core sync
                if (rowLoopIdx == 0) {
                    // the token idx of the start token of the prologue part
                    proTokenIdx = rowOffsetCurLoop % tokenNumPerHeadThisSubBlock;
                    // the token num of the prologue part
                    proTokenNum =
                        Min(rowNumCurLoop, (tokenNumPerHeadThisSubBlock - proTokenIdx)) % tokenNumPerHeadThisSubBlock;
                    // the token num of the epilogue part
                    integralHeadNum = (rowNumCurLoop - proTokenNum) / tokenNumPerHeadThisSubBlock;
                    // the number of integral heads within a cycle
                    epiTokenNum = rowNumCurLoop - proTokenNum - integralHeadNum * tokenNumPerHeadThisSubBlock;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    CopyFullGmToUb(
                        gFull,
                        columnNum, columnNumRound, maskStride,
                        tokenNumPerHeadThisSubBlock, proTokenIdx, proTokenNum, integralHeadNum, epiTokenNum,
                        qNStartIdxVec, qNThisSubBlock, rowNumCurLoop, BIdx, qHeads, offsetFull, pseQ, pseKv);
                    Arch::CrossCoreWaitFlag(qkReady);
                }
                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gInputCurLoop = gInput[offsetInput];
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);
                CopySGmToUb(
                    gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound, columnNumPad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
            }
            if (rowLoopIdx >= preLoad) {
                uint32_t delayedRowLoopIdx = rowLoopIdx - preLoad;
                uint32_t rowOffsetCurLoop = delayedRowLoopIdx * rowNumTile;
                uint32_t rowNumCurLoop = (delayedRowLoopIdx == rowLoopNum - 1) ?
                    (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;
                uint32_t pingpongFlag = delayedRowLoopIdx % 2;

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
                ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                Applyfull((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                // next loop mask load
                // online softmax vectorized compute

                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                SinkLoopParam curSinkLoop(rowOffsetIoGm, rowNumCurLoop, qSBlockSize, rowOffsetThisSubBlock);

                int64_t offsetOutputThird = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gOutputCurLoop = gOutput[offsetOutputThird];
                auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, columnNum));
                SubCoreCompute<true>(
                    gOutputCurLoop,
                    gSink,
                    layoutOutputCurLoop,
                    rowOffsetCurLoop,
                    isFirstStackTile,
                    0,
                    delayedRowLoopIdx == 0,
                    delayedRowLoopIdx == rowLoopNum - 1,
                    columnNumRound,
                    pingpongFlag,
                    curStackTileMod,
                    curSinkLoop,
                    isLastStackTile,
                    false,
                    false);
                if (rowLoopIdx < rowLoopNum) {
                    uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                    uint32_t rowNumCurLoop =
                        (rowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;
                    // the token idx of the start token of the prologue part
                    proTokenIdx = rowOffsetCurLoop % tokenNumPerHeadThisSubBlock;
                    // the token num of the prologue part
                    proTokenNum =
                        Min(rowNumCurLoop, (tokenNumPerHeadThisSubBlock - proTokenIdx)) % tokenNumPerHeadThisSubBlock;
                    if ((qNThisSubBlock >= HEAD_NUM_2) && (proTokenIdx == 0) && proTokenNum > 0) {
                        qNStartIdxVec++;
                    }
                    // the number of integral heads within a cycle
                    integralHeadNum = (rowNumCurLoop - proTokenNum) / tokenNumPerHeadThisSubBlock;
                    if ((qNThisSubBlock >= HEAD_NUM_2) && integralHeadNum > 0 && proTokenNum == 0) {
                        qNStartIdxVec++;
                    }
                    // the token num of the epilogue part
                    epiTokenNum = rowNumCurLoop - proTokenNum - integralHeadNum * tokenNumPerHeadThisSubBlock;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

                    CopyFullGmToUb(
                        gFull,
                        columnNum, columnNumRound, maskStride,
                        tokenNumPerHeadThisSubBlock, proTokenIdx, proTokenNum, integralHeadNum, epiTokenNum,
                        qNStartIdxVec, qNThisSubBlock, rowNumCurLoop, BIdx, qHeads, offsetFull, pseQ, pseKv);
                    // 当前循环要搬的行数
                }
            }
        }
    }

#endif // FUSED_BLOCK_EPILOGUE_ONLINE_SOFTMAX_OPERATOR_INC_HPP
