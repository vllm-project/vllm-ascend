/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recompute_wu_fwd_common.h
 * \brief
 */

#ifndef RECOMPUTE_WU_FWD_COMMON_H
#define RECOMPUTE_WU_FWD_COMMON_H
constexpr uint64_t SYNC_AIV_AIC_FLAG_3 = 3;
constexpr uint64_t SYNC_AIC_AIV_FLAG_5 = 5;
constexpr uint64_t ONE_BLOCK_32 = 32;
constexpr uint32_t FP32_PER_BLOCK_8 = 8;
constexpr uint32_t FP32_PER_REPEAT_64 = 64;

__aicore__ void inline GetChunkOffset(GM_ADDR cu_seqlens, GM_ADDR chunk_indices, uint64_t B, uint64_t H, uint64_t T,
                                      uint64_t chunkSize, uint32_t loopIdx, uint32_t &bos, uint32_t &eos)
{
    if (cu_seqlens == nullptr) {
        uint32_t coreLoopsInB = (T + chunkSize - 1) / chunkSize;
        uint32_t chunkIdx = loopIdx % coreLoopsInB;
        uint32_t bIdx = loopIdx / coreLoopsInB;
        bos = chunkIdx * chunkSize;
        eos = bos + chunkSize > T ? T : bos + chunkSize;
        bos += (bIdx * H * T);
        eos += (bIdx * H * T);
    } else {
        AscendC::GlobalTensor<uint64_t> cuSeqlensTensor;
        AscendC::GlobalTensor<uint64_t> chunkIndicesTensor;
        cuSeqlensTensor.SetGlobalBuffer((__gm__ uint64_t *)cu_seqlens);
        chunkIndicesTensor.SetGlobalBuffer((__gm__ uint64_t *)chunk_indices);
        uint32_t seqIdx = chunkIndicesTensor.GetValue(2 * loopIdx);
        uint32_t chunkIdx = chunkIndicesTensor.GetValue(2 * loopIdx + 1);
        uint32_t curSeqBegin = cuSeqlensTensor.GetValue(seqIdx);
        uint32_t curSeqEnd = cuSeqlensTensor.GetValue(seqIdx + 1);
        bos = curSeqBegin + chunkIdx * chunkSize;
        eos = bos + chunkSize > curSeqEnd ? curSeqEnd : bos + chunkSize;
    }

    return;
}
#endif // RECOMPUTE_WU_FWD_COMMON_H
