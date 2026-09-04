/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSED_BLOCK_EPILOGUE_ONLINE_SOFTMAX_DATA_COPY_INC_HPP
#define FUSED_BLOCK_EPILOGUE_ONLINE_SOFTMAX_DATA_COPY_INC_HPP

    __aicore__ inline
    void CopySGmToUb(
        AscendC::GlobalTensor<ElementInput> gInput,
        uint32_t sUbOffset,
        uint32_t rowCurLoop,
        uint32_t columnRound,
        uint32_t columnPad)
    {
        AscendC::DataCopy(
            lsUbTensor[sUbOffset],
            gInput,
            AscendC::DataCopyParams(
                rowCurLoop, columnRound / FLOAT_BLOCK_SIZE,
                (columnPad - columnRound) / FLOAT_BLOCK_SIZE, 0));
    }

    __aicore__ inline void OperatePreMaskUb(uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        UpCastMask<half, ElementMask>(
            maskUbTensor16,
            maskUbTensor,
            rowNumCurLoop,
            columnNumRound);
        AscendC::CompareScalar(
            maskUbTensorUint8,
            maskUbTensor16,
            static_cast<half>(1.0),
            AscendC::CMPMODE::NE,
            REPEAT_SIZE_IN_BYTE / sizeof(half),
            (rowNumCurLoop * columnNumRound + HALF_VECTOR_SIZE - 1) / HALF_VECTOR_SIZE,
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate<half>(tempMaskTensor, static_cast<half>(1), rowNumCurLoop * columnNumRound);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(
            maskUbTensor16,
            maskUbTensorUint8,
            tempMaskTensor,
            static_cast<half>(0),
            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
            rowNumCurLoop * columnNumRound);
        AscendC::PipeBarrier<PIPE_V>();
        UpCastMask<float, half>(maskUbTensor32, maskUbTensor16, rowNumCurLoop, columnNumRound);
    }

    __aicore__ inline void OperateNextMaskUb(uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        UpCastMask<half, ElementMask>(
            maskUbTensor16,
            maskUbTensor[MAX_UB_S_ELEM_NUM],
            rowNumCurLoop,
            columnNumRound);
        UpCastMask<float, half>(
            maskUbTensor32,
            maskUbTensor16,
            rowNumCurLoop,
            columnNumRound);
    }


    __aicore__ inline
    void CopyMaskGmToUb(
        AscendC::GlobalTensor<ElementMask> gMask,
        uint32_t columnNum, uint32_t columnNumRound,
        uint32_t maskStride, uint32_t tokenNumPerHead,
        uint32_t proTokenIdx, uint32_t proTokenNum,
        uint32_t integralHeadNum, uint32_t epiTokenNum, bool isNextMask)
    {
        uint32_t innerUbRowOffset = isNextMask ? MAX_UB_S_ELEM_NUM : 0;
        if (proTokenNum != 0) {
            AscendC::DataCopyPad(
                maskUbTensor[innerUbRowOffset], gMask[proTokenIdx * maskStride],
                AscendC::DataCopyExtParams(
                    proTokenNum, columnNum * sizeof(ElementMask),
                    (maskStride - columnNum) * sizeof(ElementMask), 0, 0),
                AscendC::DataCopyPadExtParams<ElementMask>(false, 0, 0, 0));
            innerUbRowOffset += proTokenNum * columnNumRound;
        }
        for (uint32_t headIdx = 0; headIdx < integralHeadNum; headIdx++) {
            AscendC::DataCopyPad(
                maskUbTensor[innerUbRowOffset], gMask,
                AscendC::DataCopyExtParams(
                    tokenNumPerHead, columnNum * sizeof(ElementMask),
                    (maskStride - columnNum) * sizeof(ElementMask), 0, 0),
                AscendC::DataCopyPadExtParams<ElementMask>(false, 0, 0, 0));
            innerUbRowOffset += tokenNumPerHead * columnNumRound;
        }
        if (epiTokenNum != 0) {
            AscendC::DataCopyPad(
                maskUbTensor[innerUbRowOffset], gMask,
                AscendC::DataCopyExtParams(
                    epiTokenNum, columnNum * sizeof(ElementMask),
                    (maskStride - columnNum) * sizeof(ElementMask), 0, 0),
                AscendC::DataCopyPadExtParams<ElementMask>(false, 0, 0, 0));
        }
    }

    __aicore__ inline
    void CalcGmFullShift(
        int64_t &offsetFull,
        const LayoutInput &layoutMask,
        uint32_t rowOffer,
        uint32_t kvSStartIdx,
        uint32_t maskOffsetThisSubBlock)
    {
        uint32_t fullBlockStart = rowOffer;
        uint32_t gmOffsetFullRow = fullBlockStart + maskOffsetThisSubBlock;
        uint32_t gmOffsetFullColumn = kvSStartIdx;
        offsetFull = layoutMask.GetOffset(MatrixCoord(gmOffsetFullRow, gmOffsetFullColumn));
    }

    __aicore__ inline
    void CopyFullGmToUb(
        AscendC::GlobalTensor<ElementFull> gFull,
        uint32_t columnNum, uint32_t columnNumRound, uint32_t maskStride, uint32_t tokenNumPerHead,
        uint32_t proTokenIdx, uint32_t proTokenNum, uint32_t integralHeadNum, uint32_t epiTokenNum,
        uint32_t &qNStartIdxVec, uint32_t qNThisSubBlock, uint32_t rowNumCurLoop, uint32_t BIdx,
        uint32_t qHeads, int64_t offsetFull, int64_t pseQ, int64_t pseKv)
    {
        uint32_t innerUbRowOffset = 0;
        int64_t gFullOffset = 0;
        if (proTokenNum != 0) {
            gFullOffset = BIdx * qHeads * pseQ * pseKv + qNStartIdxVec * pseQ * pseKv + offsetFull;
            AscendC::DataCopyPad(
                fullUbTensor16[innerUbRowOffset], gFull[gFullOffset + proTokenIdx * maskStride],
                AscendC::DataCopyExtParams(
                    proTokenNum, columnNum * sizeof(ElementFull),
                    (maskStride - columnNum) * sizeof(ElementFull),
                    (columnNumRound - columnNum) * sizeof(ElementFull) / BLOCK_SIZE_IN_BYTE, 0),
                AscendC::DataCopyPadExtParams<ElementFull>(false, 0, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
            UpCastMask<float, ElementFull>(fullUbTensor32[innerUbRowOffset], fullUbTensor16[innerUbRowOffset],
                rowNumCurLoop, columnNumRound);
            innerUbRowOffset += proTokenNum * columnNumRound;
        }
        for (uint32_t headIdx = 0; headIdx < integralHeadNum; headIdx++) {
            if (qNThisSubBlock >= HEAD_NUM_2) {
                if (proTokenNum > 0 && headIdx == 0) {
                    qNStartIdxVec++;
                }
                if (headIdx > 0) {
                    qNStartIdxVec++;
                }
            }
            gFullOffset = BIdx * qHeads * pseQ * pseKv + qNStartIdxVec * pseQ * pseKv + offsetFull;
            AscendC::DataCopyPad(
                fullUbTensor16[innerUbRowOffset], gFull[gFullOffset],
                AscendC::DataCopyExtParams(
                    tokenNumPerHead, columnNum * sizeof(ElementFull),
                    (maskStride - columnNum) * sizeof(ElementFull),
                    (columnNumRound - columnNum) * sizeof(ElementFull) / BLOCK_SIZE_IN_BYTE, 0),
                AscendC::DataCopyPadExtParams<ElementFull>(false, 0, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
            UpCastMask<float, ElementFull>(fullUbTensor32[innerUbRowOffset], fullUbTensor16[innerUbRowOffset],
                rowNumCurLoop, columnNumRound);
            innerUbRowOffset += tokenNumPerHead * columnNumRound;
        }
        if (epiTokenNum != 0) {
            if (qNThisSubBlock >= HEAD_NUM_2) {
                qNStartIdxVec++;
            }
            gFullOffset = BIdx * qHeads * pseQ * pseKv + qNStartIdxVec * pseQ * pseKv + offsetFull;
            AscendC::DataCopyPad(
                fullUbTensor16[innerUbRowOffset], gFull[gFullOffset],
                AscendC::DataCopyExtParams(
                    epiTokenNum, columnNum * sizeof(ElementFull),
                    (maskStride - columnNum) * sizeof(ElementFull),
                    (columnNumRound - columnNum) * sizeof(ElementFull) / BLOCK_SIZE_IN_BYTE, 0),
                AscendC::DataCopyPadExtParams<ElementFull>(false, 0, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
            UpCastMask<float, ElementFull>(fullUbTensor32[innerUbRowOffset], fullUbTensor16[innerUbRowOffset],
                rowNumCurLoop, columnNumRound);
        }
    }

    __aicore__ inline
    void Applyfull(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        AscendC::Add<float, false>(
            lsUbTensor[sUbOffset],
            lsUbTensor[sUbOffset],
            fullUbTensor32,
            (uint64_t)0,
            NpuArch::Detail::Alignment::CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE),
            AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline
    void ScaleS(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        AscendC::Muls<float, false>(
            lsUbTensor[sUbOffset],
            lsUbTensor[sUbOffset],
            scaleValue,
            (uint64_t)0,
            NpuArch::Detail::Alignment::CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE),
            AscendC::UnaryRepeatParams(1, 1, 8, 8));

        AscendC::PipeBarrier<PIPE_V>();
    }

#endif // FUSED_BLOCK_EPILOGUE_ONLINE_SOFTMAX_DATA_COPY_INC_HPP
