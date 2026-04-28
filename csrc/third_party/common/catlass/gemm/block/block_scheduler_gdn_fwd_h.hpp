#include "catlass/gemm_coord.hpp"
using namespace Catlass;

#ifndef CATLASS_GEMM_SCHEDULER_GDN_FWD_H_HPP
#define CATLASS_GEMM_SCHEDULER_GDN_FWD_H_HPP

// constexpr uint32_t PING_PONG_STAGES = 1;
constexpr uint32_t PING_PONG_STAGES = 2;

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

struct GDNFwdHOffsets {
    uint32_t hSrcOffset;
    uint32_t hDstOffset;
    uint32_t uvOffset;
    uint32_t wkOffset;
    uint32_t wOffset;
    uint32_t gOffset;
    uint32_t hWorkOffset;
    uint32_t vWorkOffset;
    uint32_t initialStateOffset;
    uint32_t finalStateOffset;
    bool isInitialState;
    bool isFinalState;
    uint32_t blockTokens;
    bool isDummyHead;
    // for debug
    uint32_t batchIdx;
    uint32_t headIdx;
    uint32_t chunkIdx;

};

struct BlockSchedulerGdnFwdH {
    uint32_t batch;
    uint32_t seqlen;
    uint32_t kNumHead;
    uint32_t vNumHead;
    uint32_t kHeadDim;
    uint32_t vHeadDim;
    uint32_t chunkSize;
    uint32_t vBlockSize{128};
    uint32_t isVariedLen;
    uint32_t shapeBatch;
    uint32_t tokenBatch;
    bool useInitialState;
    bool storeFinalState;

    uint32_t taskIdx;
    uint32_t taskLoops;
    uint32_t cubeCoreIdx;
    uint32_t cubeCoreNum;
    uint32_t vLoops;
    uint32_t taskNum;
    uint32_t headGroups;
    uint32_t totalChunks;
    uint32_t totalTokens;
    uint32_t headInnerLoop;

    uint32_t iterId {0};
    bool hasDummyHead;
    bool isRunning;
    bool processNewTask {true};
    bool firstLoop {true};
    bool lastLoop {false};
    GDNFwdHOffsets offsets[PING_PONG_STAGES];
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
    
    uint32_t chunkOffset;
    uint32_t tokenOffset;
    uint32_t batchChunks;
    uint32_t batchTokens;

    AscendC::GlobalTensor<int64_t> gmSeqlen;
    AscendC::GlobalTensor<int64_t> gmNumChunks;

    Arch::CrossCoreFlag cube1Done{0};
    Arch::CrossCoreFlag vec1Done{1};
    Arch::CrossCoreFlag cube2Done{2};
    Arch::CrossCoreFlag vec2Done{3};

    CATLASS_DEVICE
    BlockSchedulerGdnFwdH() {}

    CATLASS_DEVICE
    void Init(GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR tiling, uint32_t coreIdx, uint32_t coreNum) {
        __gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict gdnFwdHTilingData = reinterpret_cast<__gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict>(tiling);

        batch = gdnFwdHTilingData->batch;
        seqlen = gdnFwdHTilingData->seqlen;
        kNumHead = gdnFwdHTilingData->kNumHead;
        vNumHead = gdnFwdHTilingData->vNumHead;
        kHeadDim = gdnFwdHTilingData->kHeadDim;
        vHeadDim = gdnFwdHTilingData->vHeadDim;
        chunkSize = gdnFwdHTilingData->chunkSize;
        isVariedLen = gdnFwdHTilingData->isVariedLen;
        shapeBatch = gdnFwdHTilingData->shapeBatch;
        tokenBatch = gdnFwdHTilingData->tokenBatch;
        useInitialState = gdnFwdHTilingData->useInitialState;
        storeFinalState = gdnFwdHTilingData->storeFinalState;

        gmSeqlen.SetGlobalBuffer((__gm__ int64_t *)cu_seqlens);
        gmNumChunks.SetGlobalBuffer((__gm__ int64_t *)chunk_indices);

        cubeCoreIdx = coreIdx;
        cubeCoreNum = coreNum;
        vLoops = vHeadDim / vBlockSize;
        taskNum = vLoops * batch * vNumHead;
        headGroups = vNumHead / kNumHead;
        hasDummyHead = (taskNum % (PING_PONG_STAGES * cubeCoreNum) <= cubeCoreNum) && (taskNum % (PING_PONG_STAGES * cubeCoreNum) > 0);
        taskLoops = (taskNum + cubeCoreNum * PING_PONG_STAGES - 1) / (cubeCoreNum * PING_PONG_STAGES);
        headInnerLoop = taskNum > cubeCoreNum ? PING_PONG_STAGES : 1;
        taskIdx = cubeCoreIdx * headInnerLoop;
        isRunning = taskIdx < taskNum;

        if (isVariedLen) {
            for (uint32_t b = 1; b <= tokenBatch; b++) {
                int64_t batchChunk = (gmSeqlen.GetValue(b) - gmSeqlen.GetValue(b - 1) + chunkSize - 1) / chunkSize;
                gmNumChunks.SetValue(b, gmNumChunks.GetValue(b - 1) + batchChunk);
            }
            totalChunks = gmNumChunks.GetValue(tokenBatch);
            totalTokens = gmSeqlen.GetValue(tokenBatch);
        } else {
            totalChunks = (seqlen + chunkSize - 1) / chunkSize;
            totalTokens = seqlen;
        }
        
    }

    CATLASS_DEVICE
    void InitTask() {
        iterId++;
        currStage = (currStage + 1) % PING_PONG_STAGES;
        if (processNewTask) {
            if (taskIdx >= taskNum) {
                lastLoop = true;
                isRunning = false;
                return;
            }
            vIdx = taskIdx / (batch * vNumHead);
            batchIdx = (taskIdx - vIdx * batch * vNumHead) / vNumHead;
            baseHeadIdx = taskIdx % vNumHead;
            shapeBatchIdx = isVariedLen ? 0 : batchIdx;
            tokenBatchIdx = isVariedLen ? batchIdx : 0;
            chunkOffset = isVariedLen ? gmNumChunks.GetValue(tokenBatchIdx) : 0;
            batchChunks = isVariedLen ? (gmNumChunks.GetValue(tokenBatchIdx + 1) - chunkOffset) : totalChunks;
            tokenOffset = isVariedLen ? gmSeqlen.GetValue(tokenBatchIdx) : 0;
            batchTokens = isVariedLen ? (gmSeqlen.GetValue(tokenBatchIdx + 1) - tokenOffset) : totalTokens;
            chunkIdx = 0;
            headInnerIdx = 0;
        } else {
            chunkIdx = headInnerIdx == PING_PONG_STAGES - 1 ? chunkIdx + 1 : chunkIdx;
            headInnerIdx = (headInnerIdx + 1) % PING_PONG_STAGES;
        }
        
        vHeadIdx = baseHeadIdx + headInnerIdx;
        kHeadIdx = vHeadIdx / headGroups;
        offsets[currStage].isInitialState = chunkIdx == 0; 
        offsets[currStage].isFinalState = chunkIdx == (batchChunks - 1); 
        offsets[currStage].initialStateOffset = (batchIdx * vNumHead + vHeadIdx) * kHeadDim * vHeadDim; 
        offsets[currStage].finalStateOffset = (batchIdx * vNumHead + vHeadIdx) * kHeadDim * vHeadDim; 
        offsets[currStage].hSrcOffset = (shapeBatchIdx * vNumHead * totalChunks + vHeadIdx * totalChunks + chunkOffset + chunkIdx) * kHeadDim * vHeadDim;
        offsets[currStage].hDstOffset = offsets[currStage].hSrcOffset + kHeadDim * vHeadDim;
        offsets[currStage].uvOffset = (shapeBatchIdx * vNumHead * totalTokens + vHeadIdx * totalTokens + tokenOffset + chunkIdx * chunkSize) * vHeadDim;
        offsets[currStage].wkOffset = (shapeBatchIdx * kNumHead * totalTokens + kHeadIdx * totalTokens + tokenOffset + chunkIdx * chunkSize) * kHeadDim;
        offsets[currStage].wOffset = (shapeBatchIdx * vNumHead * totalTokens + vHeadIdx * totalTokens + tokenOffset + chunkIdx * chunkSize) * kHeadDim;
        offsets[currStage].gOffset = shapeBatchIdx * vNumHead * totalTokens + vHeadIdx * totalTokens + tokenOffset + chunkIdx * chunkSize;
        offsets[currStage].hWorkOffset = (cubeCoreIdx * PING_PONG_STAGES + currStage) * kHeadDim * vHeadDim;
        offsets[currStage].vWorkOffset = (cubeCoreIdx * PING_PONG_STAGES + currStage) * chunkSize * vHeadDim;
        offsets[currStage].blockTokens = offsets[currStage].isFinalState ? (batchTokens - chunkIdx * chunkSize) : chunkSize;
        offsets[currStage].isDummyHead = headInnerLoop < PING_PONG_STAGES && headInnerIdx >= headInnerLoop; 
        offsets[currStage].batchIdx = batchIdx; 
        offsets[currStage].headIdx = vHeadIdx; 
        offsets[currStage].chunkIdx = chunkIdx; 

        processNewTask = chunkIdx == batchChunks - 1 && headInnerIdx == PING_PONG_STAGES - 1;
        if (processNewTask) {
            uint32_t currLoopIdx = taskIdx / (PING_PONG_STAGES * cubeCoreNum);
            headInnerLoop = ((currLoopIdx + 2 == taskLoops) && hasDummyHead) ? 1 : PING_PONG_STAGES;
            taskIdx = (currLoopIdx + 1) * PING_PONG_STAGES * cubeCoreNum + headInnerLoop * cubeCoreIdx;
        }
    }

    CATLASS_DEVICE
    GDNFwdHOffsets& GetStage1Offsets() {
        return offsets[currStage];
    }
    
    CATLASS_DEVICE
    bool NeedProcessStage1() {
        GDNFwdHOffsets& stage1Offsets = GetStage1Offsets();
        return !(lastLoop || stage1Offsets.isDummyHead);
    }

    CATLASS_DEVICE
    GDNFwdHOffsets& GetStage2Offsets() {
        return offsets[(currStage - 1) % PING_PONG_STAGES];
    }

    CATLASS_DEVICE
    bool NeedProcessStage2() {
        GDNFwdHOffsets& stage2Offsets = GetStage2Offsets();
        return !(iterId == 1 || (!storeFinalState && stage2Offsets.isFinalState) || stage2Offsets.isDummyHead);
    }
};

struct BlockSchedulerGdnFwdHCube : public BlockSchedulerGdnFwdH {
    CATLASS_DEVICE
    BlockSchedulerGdnFwdHCube() {}

    CATLASS_DEVICE
    void Init(GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR tiling) {
        BlockSchedulerGdnFwdH::Init(cu_seqlens, chunk_indices, tiling, AscendC::GetBlockIdx(), AscendC::GetBlockNum());
    }

};

struct BlockSchedulerGdnFwdHVec : public BlockSchedulerGdnFwdH {
    CATLASS_DEVICE
    BlockSchedulerGdnFwdHVec() {}

    CATLASS_DEVICE
    void Init(GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR tiling) {
        BlockSchedulerGdnFwdH::Init(cu_seqlens, chunk_indices, tiling, AscendC::GetBlockIdx() / AscendC::GetSubBlockNum(), AscendC::GetBlockNum());
    }

};

}  // namespace Catlass::Gemm::Block

#endif  // CATLASS_GEMM_SCHEDULER_GDN_FWD_H_HPP