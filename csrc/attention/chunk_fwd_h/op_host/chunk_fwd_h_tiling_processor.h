/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_fwd_h_tiling_processor.h
 * \brief Tiling computation decoupled from gert::TilingContext.
 *
 * The caller is responsible for resolving framework-specific information (shapes, dtypes,
 * platform core number, lib-api workspace size) into the plain context struct below. The
 * processor then fills ChunkFwdHPlainTilingData together with the block
 * dim and the total workspace size, mirroring exactly the original Tiling4ChunkFwdH.
 */

#ifndef CHUNK_FWD_H_TILING_PROCESSOR_H
#define CHUNK_FWD_H_TILING_PROCESSOR_H

#include <cstddef>
#include <cstdint>

#include "../op_kernel/chunk_fwd_h_struct.h"

namespace optiling {

static constexpr size_t CHUNK_FWD_H_WORKSPACE_RSV_BYTE = 16 * 1024 * 1024;
static constexpr size_t CHUNK_FWD_H_GM_ALIGN = 512;
static constexpr int64_t CHUNK_FWD_H_ROUND_HEAD_SLOTS = 4;
static constexpr int64_t CHUNK_FWD_H_CHUNK_SIZE = 64;
static constexpr int64_t CHUNK_FWD_H_K_DIM = 128;
static constexpr int64_t CHUNK_FWD_H_V_DIM = 128;

// Plain, framework-agnostic inputs needed to compute the tiling.
struct ChunkFwdHTilingContext {
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
    bool useInitialState;
    // attrs
    bool storeFinalState;
    int64_t chunkSize;
    // platform
    uint32_t aicCoreNum;
    size_t libApiWorkSpaceSize;
};

class ChunkFwdHTilingProcessor {
public:
    explicit ChunkFwdHTilingProcessor(const ChunkFwdHTilingContext &ctx) : ctx_(ctx) {}

    // Fills the plain tiling struct, the block dim and the total workspace size.
    void Process(::ChunkFwdHPlainTilingData &tiling,
                 uint32_t &blockDim, size_t &workspaceSize) const
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

        const uint64_t effectiveSequenceCount = static_cast<uint64_t>(batch);
        const uint64_t totalHeadTasks =
            effectiveSequenceCount * static_cast<uint64_t>(ctx_.vNumHead);
        const uint64_t physicalAicCores = static_cast<uint64_t>(ctx_.aicCoreNum);
        const uint64_t headsPerCore =
            (totalHeadTasks + physicalAicCores - 1) / physicalAicCores;
        blockDim = static_cast<uint32_t>(
            (totalHeadTasks + headsPerCore - 1) / headsPerCore);
        const int64_t activeAicCoreNum = static_cast<int64_t>(blockDim);
        constexpr int64_t chunkSize = CHUNK_FWD_H_CHUNK_SIZE;
        constexpr int64_t kHeadDim = CHUNK_FWD_H_K_DIM;
        constexpr int64_t vHeadDim = CHUNK_FWD_H_V_DIM;

        size_t workspaceOffset = ctx_.libApiWorkSpaceSize;
        workspaceOffset += CHUNK_FWD_H_WORKSPACE_RSV_BYTE;

        tiling.vWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignUp(static_cast<size_t>(activeAicCoreNum * CHUNK_FWD_H_ROUND_HEAD_SLOTS *
                                                        chunkSize * vHeadDim * sizeof(float)));

        tiling.vUpdateWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignUp(static_cast<size_t>(activeAicCoreNum * CHUNK_FWD_H_ROUND_HEAD_SLOTS *
                                                        chunkSize * vHeadDim * sizeof(uint16_t)));

        tiling.kDecayWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignUp(static_cast<size_t>(activeAicCoreNum * CHUNK_FWD_H_ROUND_HEAD_SLOTS *
                                                        kHeadDim * vHeadDim * sizeof(float)));

        tiling.hWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignUp(static_cast<size_t>(activeAicCoreNum * CHUNK_FWD_H_ROUND_HEAD_SLOTS *
                                                        kHeadDim * vHeadDim * sizeof(float)));

        workspaceOffset += CHUNK_FWD_H_WORKSPACE_RSV_BYTE;
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
        tiling.isVariedLen = isVariedLen;
        tiling.shapeBatch = shapeBatch;
        tiling.tokenBatch = tokenBatch;
    }

private:
    // Mirrors the original "(x + GM_ALIGN) / GM_ALIGN * GM_ALIGN" alignment.
    static size_t AlignUp(size_t x)
    {
        return (x + CHUNK_FWD_H_GM_ALIGN - 1) / CHUNK_FWD_H_GM_ALIGN * CHUNK_FWD_H_GM_ALIGN;
    }

    const ChunkFwdHTilingContext &ctx_;
};

} // namespace optiling

#endif // CHUNK_FWD_H_TILING_PROCESSOR_H
