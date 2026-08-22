/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSED_BLOCK_EPILOGUE_ONLINE_SOFTMAX_LOW_PREC_MASK_INC_HPP
#define FUSED_BLOCK_EPILOGUE_ONLINE_SOFTMAX_LOW_PREC_MASK_INC_HPP

    template<typename ElementMaskDst, typename ElementMaskSrc>
    __aicore__ inline
    void UpCastMask(
        const AscendC::LocalTensor<ElementMaskDst> &maskUbTensorDst,
        const AscendC::LocalTensor<ElementMaskSrc> &maskUbTensorSrc,
        uint32_t rowMaskNumCurLoop,
        uint32_t columnNumRound)
    {
        AscendC::Cast<ElementMaskDst, ElementMaskSrc, false>(
            maskUbTensorDst, maskUbTensorSrc, AscendC::RoundMode::CAST_NONE, (uint64_t)0,
            NpuArch::Detail::Alignment::CeilDiv(
                rowMaskNumCurLoop * columnNumRound, (uint32_t)(REPEAT_SIZE_IN_BYTE / sizeof(ElementMaskDst))),
            AscendC::UnaryRepeatParams(1, 1, 8, 4));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline
    void ApplyMask(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound, uint32_t maskColumnRound,
        uint32_t addMaskUbOffset)
    {
        AscendC::Muls<half, false>(
            maskUbTensor16,
            maskUbTensor16,
            (half)-6e4, // -65504
            (uint64_t)0,
            (rowNumCurLoop * maskColumnRound + HALF_VECTOR_SIZE - 1) / HALF_VECTOR_SIZE,
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
        if (maskColumnRound == columnNumRound) {
            AscendC::Add<half, false>(
                computeUbTensor,
                computeUbTensor,
                maskUbTensor16,
                (uint64_t)0,
                (rowNumCurLoop * maskColumnRound + HALF_VECTOR_SIZE - 1) / HALF_VECTOR_SIZE,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        } else {
            uint32_t loop = maskColumnRound / HALF_VECTOR_SIZE;
            for (uint32_t i = 0; i < loop; i++) {
                AscendC::Add<half, false>(
                    computeUbTensor[addMaskUbOffset + i * HALF_VECTOR_SIZE],
                    computeUbTensor[addMaskUbOffset + i * HALF_VECTOR_SIZE],
                    maskUbTensor16[i * HALF_VECTOR_SIZE],
                    (uint64_t)0,
                    rowNumCurLoop,
                    AscendC::BinaryRepeatParams(1,
                        1,
                        1,
                        columnNumRound / BLOCK_SIZE,
                        columnNumRound / BLOCK_SIZE,
                        maskColumnRound / BLOCK_SIZE));
            }
            if (maskColumnRound % HALF_VECTOR_SIZE > 0) {
                SetVecMask(maskColumnRound % HALF_VECTOR_SIZE);
                AscendC::Add<half, false>(
                    computeUbTensor[addMaskUbOffset + loop * HALF_VECTOR_SIZE],
                    computeUbTensor[addMaskUbOffset + loop * HALF_VECTOR_SIZE],
                    maskUbTensor16[loop * HALF_VECTOR_SIZE],
                    (uint64_t)0,
                    rowNumCurLoop,
                    AscendC::BinaryRepeatParams(1,
                        1,
                        1,
                        columnNumRound / BLOCK_SIZE,
                        columnNumRound / BLOCK_SIZE,
                        maskColumnRound / BLOCK_SIZE));
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

#endif // FUSED_BLOCK_EPILOGUE_ONLINE_SOFTMAX_LOW_PREC_MASK_INC_HPP
