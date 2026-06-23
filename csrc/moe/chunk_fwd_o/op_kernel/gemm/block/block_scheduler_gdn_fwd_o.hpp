/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CATLASS_GEMM_SCHEDULER_GDN_FWD_O_HPP
#define CATLASS_GEMM_SCHEDULER_GDN_FWD_O_HPP

// constexpr uint32_t PING_PONG_STAGES = 1;
constexpr uint32_t PING_PONG_STAGES = 2;
constexpr uint32_t BYTE_SIZE_16_BIT = 2;

template <typename T>
CATLASS_DEVICE T AlignUp(T a, T b) {
    return (b == 0) ? 0 : (a + b - 1) / b * b;
}

template <typename T>
CATLASS_DEVICE T Min(T a, T b) {
    return (a > b) ? b : a;
}

template <typename T>
CATLASS_DEVICE T Max(T a, T b) {
    return (a > b) ? a : b;
}

namespace Catlass::Gemm::Block {


struct GDNFwdOOffsets {
    uint32_t qOffset;
    uint32_t kOffset;
    uint32_t vOffset;
    uint32_t oOffset;
    uint32_t hOffset;
    uint32_t gOffset;
    uint32_t attnWorkOffset;
    uint32_t hvWorkOffset;
    bool isFinalState;
    uint32_t blockTokens;
    // for debug
    uint32_t batchIdx;
    uint32_t headIdx;
    uint32_t chunkIdx;

};

struct BlockSchedulerGdnFwdO {
    uint32_t shapeBatch;
    uint32_t seqlen;
    uint32_t kNumHead;
    uint32_t vNumHead;
    uint32_t kHeadDim;
    uint32_t vHeadDim;
    uint32_t chunkSize;
    uint32_t kStride0;
    uint32_t vStride0;
    uint32_t isVariedLen;
    uint32_t tokenBatch;
    uint32_t numChunks{0};
    uint32_t vBlockSize{128};

    uint32_t taskIdx;
    uint32_t cubeCoreIdx;
    uint32_t cubeCoreNum;
    uint32_t vLoops;
    uint32_t taskNum;
    uint32_t headGroups;

    bool isRunning;
    bool processNewTask {true};
    bool firstLoop {true};
    bool lastLoop {false};
    GDNFwdOOffsets offsets[PING_PONG_STAGES];
    int32_t currStage{PING_PONG_STAGES - 1};

    uint32_t vIdx;
    uint32_t batchIdx;
    uint32_t baseHeadIdx;
    uint32_t chunkIdx;
    uint32_t headInnerIdx;
    uint32_t vHeadIdx;
    uint32_t kHeadIdx;
    uint32_t shapeBatchIdx;
    uint32_t tokenBatchIdx;
    
    uint32_t batchChunkIdx;
    uint32_t batchChunkStartIdx;
    uint32_t tokenOffset;
    uint32_t batchChunks;
    uint32_t batchTokens;

    AscendC::GlobalTensor<int64_t> gmSeqlen;
    AscendC::GlobalTensor<int64_t> gmChunkOffsets;

    Arch::CrossCoreFlag cube1Done{3};
    Arch::CrossCoreFlag vec1Done{4};
    Arch::CrossCoreFlag cube2Done{5};
    Arch::CrossCoreFlag cube3Done{6};
    Arch::CrossCoreFlag vec2Done{7};

    CATLASS_DEVICE
    BlockSchedulerGdnFwdO() {}

    CATLASS_DEVICE
    void Init(GM_ADDR cu_seqlens, GM_ADDR chunk_offsets, GM_ADDR tiling, uint32_t coreIdx, uint32_t coreNum) {
        __gm__ ChunkFwdOTilingData *__restrict gdnFwdOTilingData = reinterpret_cast<__gm__ ChunkFwdOTilingData *__restrict>(tiling);
        shapeBatch = gdnFwdOTilingData->shapeBatch;
        seqlen = gdnFwdOTilingData->seqlen;
        kNumHead = gdnFwdOTilingData->kNumHead;
        vNumHead = gdnFwdOTilingData->vNumHead;
        kHeadDim = gdnFwdOTilingData->kHeadDim;
        vHeadDim = gdnFwdOTilingData->vHeadDim;
        chunkSize = gdnFwdOTilingData->chunkSize;
        kStride0 = gdnFwdOTilingData->kStride0;
        vStride0 = gdnFwdOTilingData->vStride0;
        isVariedLen = gdnFwdOTilingData->isVariedLen;
        tokenBatch = gdnFwdOTilingData->tokenBatch;

        gmSeqlen.SetGlobalBuffer((__gm__ int64_t *)cu_seqlens);
        gmChunkOffsets.SetGlobalBuffer((__gm__ int64_t *)chunk_offsets);

        if (isVariedLen) {
            for (uint32_t b = 1; b <= tokenBatch; b++) {
                numChunks += (gmSeqlen.GetValue(b) - gmSeqlen.GetValue(b - 1) + chunkSize - 1) / chunkSize;
            }
        } else {
            numChunks = (seqlen + chunkSize - 1) / chunkSize;
        }

        cubeCoreIdx = coreIdx;
        cubeCoreNum = coreNum;
        vLoops = vHeadDim / vBlockSize;
        taskNum = vLoops * shapeBatch * numChunks * vNumHead;
        headGroups = vNumHead / kNumHead;
        taskIdx = cubeCoreIdx * PING_PONG_STAGES;
        isRunning = taskIdx < taskNum;

    }

    CATLASS_DEVICE
    void InitTask() {
        if (processNewTask) {
            if (unlikely(taskIdx >= taskNum)) {
                isRunning = false;
            }
            vIdx = taskIdx / (shapeBatch * numChunks * vNumHead);
            shapeBatchIdx = (taskIdx - vIdx * shapeBatch * numChunks * vNumHead) / (numChunks * vNumHead);
            chunkIdx = (taskIdx - vIdx * shapeBatch * numChunks * vNumHead - shapeBatchIdx * numChunks * vNumHead) / vNumHead;
            baseHeadIdx = taskIdx % vNumHead;
            tokenBatchIdx = isVariedLen ? gmChunkOffsets.GetValue(2 * chunkIdx) : 0;
            batchChunkIdx = isVariedLen ? gmChunkOffsets.GetValue(2 * chunkIdx + 1) : chunkIdx;
            batchChunkStartIdx = chunkIdx - batchChunkIdx;
            tokenOffset = isVariedLen ? gmSeqlen.GetValue(tokenBatchIdx) : 0;
            batchTokens = isVariedLen ? (gmSeqlen.GetValue(tokenBatchIdx + 1) - tokenOffset) : seqlen;
            headInnerIdx = 0;
        } else {
            headInnerIdx = (headInnerIdx + 1) % PING_PONG_STAGES;
        }
        
        vHeadIdx = baseHeadIdx + headInnerIdx;
        kHeadIdx = vHeadIdx / headGroups;
        offsets[currStage].qOffset = (shapeBatchIdx * kNumHead * seqlen + kHeadIdx * seqlen + tokenOffset + batchChunkIdx * chunkSize) * kHeadDim;
        offsets[currStage].kOffset = shapeBatchIdx * kStride0 + kHeadIdx * seqlen * kHeadDim + tokenOffset * kHeadDim + batchChunkIdx * chunkSize * kHeadDim;
        offsets[currStage].vOffset = shapeBatchIdx * vStride0 + vHeadIdx * seqlen * vHeadDim + tokenOffset * vHeadDim + batchChunkIdx * chunkSize * vHeadDim;
        offsets[currStage].oOffset = (shapeBatchIdx * vNumHead * seqlen + vHeadIdx * seqlen + tokenOffset + batchChunkIdx * chunkSize) * vHeadDim;
        offsets[currStage].hOffset = (shapeBatchIdx * vNumHead * numChunks + vHeadIdx * numChunks + chunkIdx) * kHeadDim * vHeadDim;
        offsets[currStage].gOffset = shapeBatchIdx * vNumHead * seqlen + vHeadIdx * seqlen + tokenOffset + batchChunkIdx * chunkSize;
        offsets[currStage].attnWorkOffset = (cubeCoreIdx * PING_PONG_STAGES + currStage) * chunkSize * chunkSize;
        offsets[currStage].hvWorkOffset = (cubeCoreIdx * PING_PONG_STAGES + currStage) * chunkSize * vHeadDim;
        offsets[currStage].isFinalState = chunkIdx == (numChunks - 1) || (isVariedLen && gmChunkOffsets.GetValue(2 * chunkIdx + 3) == 0);
        offsets[currStage].blockTokens = offsets[currStage].isFinalState ? (batchTokens - batchChunkIdx * chunkSize) : chunkSize;
        offsets[currStage].batchIdx = batchIdx; 
        offsets[currStage].headIdx = vHeadIdx; 
        offsets[currStage].chunkIdx = chunkIdx; 

        processNewTask = headInnerIdx == PING_PONG_STAGES - 1;
        if (processNewTask) {
            taskIdx += PING_PONG_STAGES * cubeCoreNum;
        }
        
        currStage = (currStage + 1) % PING_PONG_STAGES;
    }


};

struct BlockSchedulerGdnFwdOCube : public BlockSchedulerGdnFwdO {
    CATLASS_DEVICE
    BlockSchedulerGdnFwdOCube() {}

    CATLASS_DEVICE
    void Init(GM_ADDR cu_seqlens, GM_ADDR chunk_offsets, GM_ADDR tiling) {
        BlockSchedulerGdnFwdO::Init(cu_seqlens, chunk_offsets, tiling, AscendC::GetBlockIdx(), AscendC::GetBlockNum());
    }

    CATLASS_DEVICE
    bool NeedProcessCube1() {
        return true;
    }

    CATLASS_DEVICE
    GDNFwdOOffsets& GetCube1Offsets() {
        return offsets[(currStage - 1) % PING_PONG_STAGES];
    }

    CATLASS_DEVICE
    GemmCoord GetCube1Shape() {
        GDNFwdOOffsets& cube1Offsets = GetCube1Offsets();
        return GemmCoord{cube1Offsets.blockTokens, cube1Offsets.blockTokens, kHeadDim};
    }

    CATLASS_DEVICE
    bool NeedProcessCube23() {
        if (unlikely(firstLoop)) {
            firstLoop = false;
            return false;
        }
        return true;
    }

    CATLASS_DEVICE
    GDNFwdOOffsets& GetCube23Offsets() {
        return offsets[(currStage - 2) % PING_PONG_STAGES];
    }

    CATLASS_DEVICE
    GemmCoord GetCube2Shape() {
        GDNFwdOOffsets& cube2Offsets = GetCube23Offsets();
        return GemmCoord{kHeadDim, vHeadDim, cube2Offsets.blockTokens};
    }

    CATLASS_DEVICE
    GemmCoord GetCube3Shape() {
        GDNFwdOOffsets& cube2Offsets = GetCube23Offsets();
        return GemmCoord{kHeadDim, vHeadDim, cube2Offsets.blockTokens};
    }

};

struct BlockSchedulerGdnFwdOVec : public BlockSchedulerGdnFwdO {
    CATLASS_DEVICE
    BlockSchedulerGdnFwdOVec() {}

    CATLASS_DEVICE
    void Init(GM_ADDR cu_seqlens, GM_ADDR chunk_offsets, GM_ADDR tiling) {
        BlockSchedulerGdnFwdO::Init(cu_seqlens, chunk_offsets, tiling, AscendC::GetBlockIdx() / AscendC::GetSubBlockNum(), AscendC::GetBlockNum());
    }

    CATLASS_DEVICE
    bool NeedProcessVec1() {
        return isRunning;
    }

    CATLASS_DEVICE
    bool NeedProcessVec2() {
        if (unlikely(firstLoop)) {
            firstLoop = false;
            return false;
        }
        return true;
    }

    CATLASS_DEVICE
    GDNFwdOOffsets& GetVec1Offsets() {
        return offsets[(currStage - 1) % PING_PONG_STAGES];
    }
    
    CATLASS_DEVICE
    GDNFwdOOffsets& GetVec2Offsets() {
        return offsets[(currStage - 2) % PING_PONG_STAGES];
    }

};

}  // namespace Catlass::Gemm::Block

#endif // CATLASS_GEMM_SCHEDULER_GDN_FWD_O_HPP