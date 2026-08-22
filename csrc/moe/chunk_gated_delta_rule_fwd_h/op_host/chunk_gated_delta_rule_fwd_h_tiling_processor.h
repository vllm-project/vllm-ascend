/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_gated_delta_rule_fwd_h_tiling_processor.h
 * \brief Tiling computation decoupled from gert::TilingContext, reusable by both the
 *        aclnn tiling entry and the fast kernel launch C++ extension.
 *
 * The caller is responsible for resolving framework-specific information (shapes, dtypes,
 * platform core number, lib-api workspace size) into the plain context struct below. The
 * processor then fills the plain ChunkGatedDeltaRuleFwdHTilingData together with the block
 * dim and the total workspace size, mirroring exactly the original Tiling4ChunkGatedDeltaRuleFwdH.
 */

#ifndef CHUNK_GATED_DELTA_RULE_FWD_H_TILING_PROCESSOR_H
#define CHUNK_GATED_DELTA_RULE_FWD_H_TILING_PROCESSOR_H

#include <cstddef>
#include <cstdint>

#include "../op_kernel/chunk_gated_delta_rule_fwd_h_struct.h"

namespace optiling {

// dtype enum convention shared with the kernel: 0 - fp16, 1 - bf16, 2 - fp32
static constexpr int64_t GDN_FWD_H_DTYPE_FP16 = 0;
static constexpr int64_t GDN_FWD_H_DTYPE_BF16 = 1;
static constexpr int64_t GDN_FWD_H_DTYPE_FP32 = 2;

static constexpr size_t GDN_FWD_H_WORKSPACE_RSV_BYTE = 16 * 1024 * 1024;
static constexpr size_t GDN_FWD_H_GM_ALIGN = 512;
static constexpr int64_t GDN_FWD_H_PING_PONG_STAGES = 2;

// Plain, framework-agnostic inputs needed to compute the tiling.
struct ChunkGatedDeltaRuleFwdHTilingContext {
    // shapes
    int64_t seqlen;        // k.dim(2)
    int64_t kNumHead;      // k.dim(1)
    int64_t kHeadDim;      // k.dim(3)
    int64_t vNumHead;      // u.dim(1)
    int64_t vHeadDim;      // u.dim(3)
    int64_t shapeBatchDim; // k.dim(0)
    // variable length
    bool hasCuSeqlens;
    int64_t cuSeqlensDim0; // length of cu_seqlens (only used when hasCuSeqlens)
    // dtypes (use GDN_FWD_H_DTYPE_*)
    int64_t dataType;      // input (k/w/u) dtype: fp16 or bf16
    int64_t gDataType;     // g dtype
    bool useInitialState;
    int64_t stateDataType; // initial/final state dtype
    // attrs
    bool storeFinalState;
    int64_t chunkSize;
    // platform
    uint32_t aicCoreNum;
    size_t libApiWorkSpaceSize;
};

class ChunkGatedDeltaRuleFwdHTilingProcessor {
public:
    explicit ChunkGatedDeltaRuleFwdHTilingProcessor(const ChunkGatedDeltaRuleFwdHTilingContext &ctx) : ctx_(ctx) {}

    // Fills the plain tiling struct, the block dim and the total workspace size.
    void Process(::ChunkGatedDeltaRuleFwdHTilingData &tiling, uint32_t &blockDim, size_t &workspaceSize) const
    {
        int64_t isVariedLen;
        int64_t shapeBatch;
        int64_t tokenBatch;
        int64_t batch;

        if (!ctx_.hasCuSeqlens) {
            isVariedLen = 0;
            shapeBatch = ctx_.shapeBatchDim;
            tokenBatch = 1;
            batch = shapeBatch;
        } else {
            isVariedLen = 1;
            shapeBatch = 1;
            tokenBatch = ctx_.cuSeqlensDim0 - 1;
            batch = tokenBatch;
        }

        blockDim = ctx_.aicCoreNum;
        const int64_t aicCoreNum = static_cast<int64_t>(ctx_.aicCoreNum);
        const int64_t chunkSize = ctx_.chunkSize;
        const int64_t kHeadDim = ctx_.kHeadDim;
        const int64_t vHeadDim = ctx_.vHeadDim;

        size_t workspaceOffset = ctx_.libApiWorkSpaceSize;
        workspaceOffset += GDN_FWD_H_WORKSPACE_RSV_BYTE;

        tiling.vWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignUp(static_cast<size_t>(aicCoreNum * chunkSize * vHeadDim * static_cast<int64_t>(sizeof(float)) * GDN_FWD_H_PING_PONG_STAGES));

        tiling.vUpdateWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignUp(static_cast<size_t>(aicCoreNum * chunkSize * vHeadDim * static_cast<int64_t>(sizeof(float)) * GDN_FWD_H_PING_PONG_STAGES));

        tiling.hWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignUp(static_cast<size_t>(aicCoreNum * kHeadDim * vHeadDim * static_cast<int64_t>(sizeof(float)) * GDN_FWD_H_PING_PONG_STAGES));

        tiling.numSeqWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignUp(static_cast<size_t>((tokenBatch + 1) * static_cast<int64_t>(sizeof(int64_t))));

        tiling.numChunksWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignUp(static_cast<size_t>((tokenBatch + 1) * static_cast<int64_t>(sizeof(int64_t))));

        workspaceOffset += GDN_FWD_H_WORKSPACE_RSV_BYTE;
        workspaceSize = workspaceOffset;

        tiling.batch = batch;
        tiling.seqlen = ctx_.seqlen;
        tiling.kNumHead = ctx_.kNumHead;
        tiling.vNumHead = ctx_.vNumHead;
        tiling.kHeadDim = ctx_.kHeadDim;
        tiling.vHeadDim = ctx_.vHeadDim;
        tiling.chunkSize = chunkSize;
        tiling.useInitialState = ctx_.useInitialState;
        tiling.storeFinalState = ctx_.storeFinalState;
        tiling.dataType = ctx_.dataType;
        tiling.gDataType = ctx_.gDataType;
        tiling.stateDataType = ctx_.stateDataType;
        tiling.isVariedLen = isVariedLen;
        tiling.shapeBatch = shapeBatch;
        tiling.tokenBatch = tokenBatch;
    }

private:
    // Mirrors the original "(x + GM_ALIGN) / GM_ALIGN * GM_ALIGN" alignment.
    static size_t AlignUp(size_t x)
    {
        return (x + GDN_FWD_H_GM_ALIGN) / GDN_FWD_H_GM_ALIGN * GDN_FWD_H_GM_ALIGN;
    }

    const ChunkGatedDeltaRuleFwdHTilingContext &ctx_;
};

} // namespace optiling

#endif // CHUNK_GATED_DELTA_RULE_FWD_H_TILING_PROCESSOR_H
