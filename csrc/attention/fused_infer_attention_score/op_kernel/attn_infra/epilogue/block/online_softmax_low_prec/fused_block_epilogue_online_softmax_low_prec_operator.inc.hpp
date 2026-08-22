/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSED_BLOCK_EPILOGUE_ONLINE_SOFTMAX_LOW_PREC_OPERATOR_INC_HPP
#define FUSED_BLOCK_EPILOGUE_ONLINE_SOFTMAX_LOW_PREC_OPERATOR_INC_HPP

    __aicore__ inline
    void operator()(
        AscendC::GlobalTensor<ElementOutput> gOutput,
        AscendC::GlobalTensor<half> gInput,
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
        uint32_t qNThisSubBlock = (qNBlockSize == 1U) ?
            0 : (subBlockIdx == 1U) ? (qNBlockSize - qNSplitSubBlock) : qNSplitSubBlock;
        uint32_t rowSplitSubBlock = (qNBlockSize == 1U) ? (qSBlockSize / 2U) : (qSBlockSize * qNSplitSubBlock);
        uint32_t rowActualThisSubBlock = (subBlockIdx == 1U) ? (rowNum - rowSplitSubBlock) : rowSplitSubBlock;
        uint32_t rowOffsetThisSubBlock = subBlockIdx * rowSplitSubBlock;
        uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
        uint32_t rowNumTile = NpuArch::Detail::Alignment::RoundDown(maxRowNumPerLoop, BLOCK_SIZE);
        rowNumTile = AscendC::Std::min(rowNumTile, HALF_VECTOR_SIZE);
        uint32_t rowLoopNum = NpuArch::Detail::Alignment::CeilDiv(rowActualThisSubBlock, rowNumTile);

        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum; rowLoopIdx++) {
            uint32_t pingpongFlag = rowLoopIdx % 2U;
            uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
            uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
            uint32_t rowNumCurLoop =
                (rowLoopIdx == rowLoopNum - 1U) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

            int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
            auto gInputCurLoop = gInput[offsetInput];

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            CopySGmToUb(
                gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound, columnNumPad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);

            int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
            auto gOutputCurLoop = gOutput[offsetOutput];
            auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, columnNum));
            SubCoreCompute(
                gOutputCurLoop,
                layoutOutputCurLoop,
                rowOffsetCurLoop,
                isFirstStackTile,
                (rowLoopIdx == 0U),
                columnNumRound,
                pingpongFlag,
                curStackTileMod,
                isSplitKV);
        }
    }

    __aicore__ inline
    void operator()(
        AscendC::GlobalTensor<ElementOutput> gOutput,
        AscendC::GlobalTensor<half> gInput,
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
        uint32_t columnNumRound = NpuArch::Detail::Alignment::RoundUp(columnNum, BLOCK_SIZE);
        uint32_t columnNumPad = layoutInput.stride(0);
        uint32_t maskStride = layoutMask.stride(0);
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        uint32_t qNThisSubBlock = (qNBlockSize == 1U) ?
            0 : (subBlockIdx == 1U) ? (qNBlockSize - qNSplitSubBlock) : qNSplitSubBlock;
        uint32_t rowSplitSubBlock = (qNBlockSize == 1U) ? (qSBlockSize / 2U) : (qSBlockSize * qNSplitSubBlock);
        uint32_t rowActualThisSubBlock = (subBlockIdx == 1U) ? (rowNum - rowSplitSubBlock) : rowSplitSubBlock;
        uint32_t rowOffsetThisSubBlock = subBlockIdx * rowSplitSubBlock;

        uint32_t tokenNumPerHeadThisSubBlock = AscendC::Std::min(qSBlockSize, rowActualThisSubBlock);

        uint32_t maskOffsetThisSubBlock = (qNBlockSize == 1U) ? rowOffsetThisSubBlock : 0;

        uint32_t gmOffsetMaskRow;
        uint32_t gmOffsetMaskColumn;
        uint32_t maskColumn;
        uint32_t addMaskUbOffset;
        if (triUp >= kvSStartIdx) {
            uint32_t triUpRoundDown = NpuArch::Detail::Alignment::RoundDown(triUp, BLOCK_SIZE);
            gmOffsetMaskRow = triUp - triUpRoundDown;
            gmOffsetMaskColumn = 0U;
            maskColumn = kvSEndIdx - triUpRoundDown;
            addMaskUbOffset = triUpRoundDown - kvSStartIdx;
        } else {
            gmOffsetMaskRow = 0U;
            gmOffsetMaskColumn = kvSStartIdx - triUp;
            maskColumn = columnNum;
            addMaskUbOffset = 0U;
        }
        uint32_t maskColumnRound = NpuArch::Detail::Alignment::RoundUp(maskColumn, BLOCK_SIZE);

        int64_t offsetMask =
            layoutMask.GetOffset(MatrixCoord(gmOffsetMaskRow + maskOffsetThisSubBlock, gmOffsetMaskColumn));
        auto gMaskThisSubBlock = gMask[offsetMask];
        auto layoutMaskThisSubBlock = layoutMask;

        uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
        uint32_t rowNumTile = NpuArch::Detail::Alignment::RoundDown(maxRowNumPerLoop, BLOCK_SIZE);
        rowNumTile = AscendC::Std::min(rowNumTile, HALF_VECTOR_SIZE);
        uint32_t rowLoopNum = NpuArch::Detail::Alignment::CeilDiv(rowActualThisSubBlock, rowNumTile);

        if (rowActualThisSubBlock == 0U) {
            Arch::CrossCoreWaitFlag(qkReady);
            return;
        }
        Arch::CrossCoreWaitFlag(qkReady);
        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum; rowLoopIdx++) {
            uint32_t pingpongFlag = rowLoopIdx % 2U;
            uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
            uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
            uint32_t rowNumCurLoop =
                (rowLoopIdx == rowLoopNum - 1U) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

            uint32_t proTokenIdx = rowOffsetCurLoop % tokenNumPerHeadThisSubBlock;
            uint32_t proTokenNum = AscendC::Std::min(rowNumCurLoop, (tokenNumPerHeadThisSubBlock - proTokenIdx)) %
                tokenNumPerHeadThisSubBlock;
            uint32_t integralHeadNum = (rowNumCurLoop - proTokenNum) / tokenNumPerHeadThisSubBlock;
            uint32_t epiTokenNum = rowNumCurLoop - proTokenNum - integralHeadNum * tokenNumPerHeadThisSubBlock;

            int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
            auto gInputCurLoop = gInput[offsetInput];
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            CopySGmToUb(
                gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound, columnNumPad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            CopyMaskGmToUb(
                gMaskThisSubBlock,
                maskColumn,
                maskColumnRound,
                maskStride,
                tokenNumPerHeadThisSubBlock,
                proTokenIdx,
                proTokenNum,
                integralHeadNum,
                epiTokenNum);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            UpCastMask<half, ElementMask>(maskUbTensor16, maskUbTensor, rowNumCurLoop, columnNumRound);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            ApplyMask(
                (pingpongFlag * MAX_UB_S_ELEM_NUM),
                rowNumCurLoop,
                columnNumRound,
                maskColumnRound,
                addMaskUbOffset);

            // online softmax vectorized compute
            int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
            auto gOutputCurLoop = gOutput[offsetOutput];
            auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, columnNum));
            SubCoreCompute(
                gOutputCurLoop,
                layoutOutputCurLoop,
                rowOffsetCurLoop,
                isFirstStackTile,
                (rowLoopIdx == 0),
                columnNumRound,
                pingpongFlag,
                curStackTileMod,
                isSplitKV);
        }
    }

#endif // FUSED_BLOCK_EPILOGUE_ONLINE_SOFTMAX_LOW_PREC_OPERATOR_INC_HPP
