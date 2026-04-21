#include "kernel_utils.h"

constexpr int32_t ALIGN = 32;
using namespace AscendC;

#define YF_LOG(format, ...)                                                                                            \
    if (false) {                                                                                          \
        printf("CoreIdx: %d on CoreType %d, " format, GetBlockIdx(), g_coreType, ##__VA_ARGS__);                       \
    }


class ReshapeAndCacheBnsd {
public:
    __aicore__ inline ReshapeAndCacheBnsd(ReshapeAndCacheBNSDTilingData tilingData)
        : batchNum_(tilingData.batch), blockSize_(tilingData.blockSize),
        coreNum_(tilingData.numCore), headNum_(tilingData.numHeads), headDim_(tilingData.headDim)
        {}

    __aicore__ inline void Init(GM_ADDR keyIn, GM_ADDR keyCacheIn, GM_ADDR slotMapping, GM_ADDR seqLen,
        GM_ADDR keyCacheOut)
    {
        AscendC::TPipe pipe;
        pipe.InitBuffer(ubBuf_, RoundUp(blockSize_ * headDim_, ALIGN));
        tmpTensor_ = ubBuf_.Get<uint8_t>();
        keyInGm_.SetGlobalBuffer((__gm__ uint8_t *)keyIn);
        keyCacheInGm_.SetGlobalBuffer((__gm__ uint8_t *)keyCacheIn);
        slotMappingGm_.SetGlobalBuffer((__gm__ int32_t *)slotMapping);
        seqLenGm_.SetGlobalBuffer((__gm__ int32_t *)seqLen);
        keyCacheOutGm_.SetGlobalBuffer((__gm__ uint8_t *)keyCacheOut);
    }

    __aicore__ inline void Process()
    {
        // Calculate the total number of pages
        uint32_t totalBlockNum = 0;
        uint32_t offsetInSlotmapping = 0;
        for (uint32_t batchIdx = 0; batchIdx < batchNum_; batchIdx++) {
            uint32_t seqLen = seqLenGm_.GetValue(batchIdx);
            int32_t slotValue = slotMappingGm_.GetValue(offsetInSlotmapping);
            uint32_t offsetInBlock = slotValue % blockSize_;
            uint32_t leftTokenNum = blockSize_ - offsetInBlock;
            uint32_t blockNumForCurrBatch = seqLen < leftTokenNum ? 1 :
                                                                   (CeilDiv(seqLen - leftTokenNum, blockSize_) + 1);
            totalBlockNum += blockNumForCurrBatch;
            offsetInSlotmapping += seqLen;

            //YF_LOG("batchIdx: %d, totalBlockNum: %d, offsetInSlotmapping: %d\n", batchIdx, totalBlockNum, offsetInSlotmapping);
        }

        uint32_t blockIdx_ = GetBlockIdx();
        uint32_t actualCoreNum = totalBlockNum <= coreNum_ ? totalBlockNum : coreNum_;
        // How many pages each core transfers
        uint32_t blockNumPerCore = totalBlockNum / actualCoreNum;
        uint32_t leftBlockNum = totalBlockNum - blockNumPerCore * actualCoreNum;
        uint32_t blockNum = blockIdx_ < leftBlockNum ? blockNumPerCore + 1 : blockNumPerCore;
        uint32_t startBlockOffset_ = blockIdx_ < leftBlockNum ? (blockNumPerCore * blockIdx_ + blockIdx_) :
                                                       (blockNumPerCore * blockIdx_ + leftBlockNum);
        if (blockIdx_ >= actualCoreNum) {
            return;
        }

        // Position of keyIn and KeyCache corresponding to startBlockOffset_ for each core
        uint32_t startBatchIdx = 0;
        uint32_t accuBlockNum = 0;
        uint32_t startTokenOffsetInBatch = 0;
        offsetInSlotmapping = 0;
        bool copyFromBatchStart = true;
        for (uint32_t batchIdx = 0; batchIdx < batchNum_; batchIdx++) {
            uint32_t seqLen = seqLenGm_.GetValue(batchIdx);
            int32_t slotValue = slotMappingGm_.GetValue(offsetInSlotmapping);
            uint32_t offsetInBlock = slotValue % blockSize_;
            uint32_t leftTokenNum = blockSize_ - offsetInBlock;
            uint32_t blockNumForCurrBatch = seqLen < leftTokenNum ? 1 :
                                                                   (CeilDiv(seqLen - leftTokenNum, blockSize_) + 1);
            accuBlockNum += blockNumForCurrBatch;

            if (startBlockOffset_ == 0) {
                break;
            } else if (accuBlockNum == startBlockOffset_) {
                startBatchIdx = batchIdx + 1;
                startTokenOffsetInBatch = 0;
                copyFromBatchStart = true;
                offsetInSlotmapping = offsetInSlotmapping + seqLen;
                break;
            } else if (accuBlockNum > startBlockOffset_) {
                startBatchIdx = batchIdx;
                startTokenOffsetInBatch = (startBlockOffset_ - (accuBlockNum - blockNumForCurrBatch + 1)) *
                    blockSize_ +leftTokenNum;
                copyFromBatchStart = false;
                offsetInSlotmapping = offsetInSlotmapping + startTokenOffsetInBatch;
                break;
            }
            offsetInSlotmapping += seqLen;
        }

        uint32_t batchIdx = startBatchIdx;
        for (uint32_t blockIdx = 0; blockIdx < blockNum; blockIdx++) {
            uint32_t seqLen = seqLenGm_.GetValue(batchIdx);
            int32_t slotValue = slotMappingGm_.GetValue(offsetInSlotmapping);
            uint32_t blockId = static_cast<uint32_t>(slotValue) / blockSize_;
            uint32_t slotId = static_cast<uint32_t>(slotValue) % blockSize_;

            if (startTokenOffsetInBatch + blockSize_ - slotId > seqLen) {
                //YF_LOG("batchIdx: %d, true\n", batchIdx);
                uint32_t currCopyTokenNum = seqLen - startTokenOffsetInBatch;
                uint32_t copyBlocks = CeilDiv(currCopyTokenNum * headDim_, 32);
                //YF_LOG("batchIdx: %d, currCopyTokenNum: %d, copyBlocks: %d from %d\n", batchIdx, currCopyTokenNum, copyBlocks, currCopyTokenNum * headDim_);
                AscendC::DataCopyParams copyInParams = {1, static_cast<uint16_t>(copyBlocks), 0, 0};
                AscendC::DataCopyParams copyOutParams = {1, static_cast<uint16_t>(copyBlocks), 0, 0};
                int64_t dstOffset = blockId * headNum_ * blockSize_ * headDim_ + slotId * headDim_;
                int64_t srcOffset = (offsetInSlotmapping - startTokenOffsetInBatch) * headNum_ * headDim_ +
                    startTokenOffsetInBatch * headDim_;
                //YF_LOG("batchIdx: %d, srcOffset[%d] -> dstOffset[%d], size: %d\n", batchIdx, srcOffset, dstOffset, static_cast<uint16_t>(copyBlocks));
                
                for (uint32_t headId = 0; headId < headNum_; headId++) {
                    DataCopy(tmpTensor_, keyInGm_[srcOffset + headId * seqLen * headDim_], copyInParams);
                    SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                    DataCopy(keyCacheOutGm_[dstOffset + headId * blockSize_* headDim_], tmpTensor_, copyOutParams);
                    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                    //YF_LOG("batchIdx: %d, src[%d] -> dst[%d], size: %d\n", batchIdx, srcOffset + headId * seqLen * headDim_, dstOffset + headId * blockSize_* headDim_, static_cast<uint16_t>(copyBlocks));
                    
                }
                batchIdx += 1;
                startTokenOffsetInBatch = 0;
                offsetInSlotmapping += currCopyTokenNum;
            } else {
                uint32_t currCopyTokenNum = blockSize_ - slotId;
                uint32_t copyBlocks = currCopyTokenNum * headDim_ / ALIGN;
                //YF_LOG("batchIdx: %d, currCopyTokenNum: %d, currCopyTokenNum * headDim_: %d\n", batchIdx, currCopyTokenNum, currCopyTokenNum * headDim_);
                uint32_t leftBytes = currCopyTokenNum * headDim_ - copyBlocks * ALIGN;
                AscendC::DataCopyParams copyInParams = {1, static_cast<uint16_t>(copyBlocks), 0, 0};
                AscendC::DataCopyParams copyOutParams = {1, static_cast<uint16_t>(copyBlocks), 0, 0};
                int64_t dstOffset = blockId * headNum_ * blockSize_ * headDim_ + slotId * headDim_;
                int64_t srcOffset = (offsetInSlotmapping - startTokenOffsetInBatch) * headNum_ * headDim_ +
                    startTokenOffsetInBatch * headDim_;
                if (copyBlocks != 0) {
                    for (uint32_t headId = 0; headId < headNum_; headId++) {
                        DataCopy(tmpTensor_, keyInGm_[srcOffset + headId * seqLen * headDim_], copyInParams);
                        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                        DataCopy(keyCacheOutGm_[dstOffset + headId * blockSize_* headDim_], tmpTensor_, copyOutParams);
                        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                        //YF_LOG("batchIdx: %d, src[%d] -> dst[%d], size: %d\n", batchIdx, srcOffset + headId * seqLen * headDim_, dstOffset + headId * blockSize_* headDim_, static_cast<uint16_t>(copyBlocks));
                    }
                }
                
                if (currCopyTokenNum + startTokenOffsetInBatch == seqLen) {
                    batchIdx += 1;
                    startTokenOffsetInBatch = 0;
                    offsetInSlotmapping += currCopyTokenNum;
                } else {
                    startTokenOffsetInBatch += currCopyTokenNum;
                    offsetInSlotmapping += currCopyTokenNum;
                }
                if (leftBytes == 0) {
                    continue;
                }
                // If there is a tail block, process it; it is less than 32 bytes.
                for (uint32_t headId = 0; headId < headNum_; headId++) {
                    for (uint32_t dimId = 0; dimId < leftBytes; dimId++) {
                        uint8_t cacheValue = keyInGm_.GetValue(srcOffset + headId * seqLen * headDim_ +
                            copyBlocks * ALIGN + dimId);
                        keyCacheOutGm_.SetValue(dstOffset + headId * blockSize_ * headDim_ +
                            copyBlocks * ALIGN + dimId, cacheValue);
                    }
                }
                // TODO: Move DataCacheCleanAndInvalid outside the loop,
                // to resolve the issue where partial data cannot be read correctly.
                AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(keyCacheOutGm_);
            }
        }
    }

private:
    GlobalTensor<uint8_t> keyInGm_;
    GlobalTensor<uint8_t> keyCacheInGm_;
    GlobalTensor<int32_t> slotMappingGm_;
    GlobalTensor<int32_t> seqLenGm_;
    GlobalTensor<uint8_t> keyCacheOutGm_;
    TBuf<TPosition::VECCALC> ubBuf_;
    LocalTensor<uint8_t> tmpTensor_;
    LocalTensor<uint8_t> keyIn_;
    LocalTensor<uint8_t> keyCacheIn_;
    LocalTensor<int32_t> slotMapping_;
    LocalTensor<int32_t> seqLen_;
    LocalTensor<uint8_t> keyCacheOut_;

    uint32_t batchNum_{0};
    uint32_t blockSize_{0};
    uint32_t coreNum_{0};
    uint32_t headNum_{0};
    uint32_t headDim_{0};
};

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata,
                                      ReshapeAndCacheBNSDTilingData *tilingdata) {
    tilingdata->numTokens = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingdata->headDim = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    tilingdata->numBlocks = (*(const __gm__ uint32_t *)(p_tilingdata + 8));
    tilingdata->numHeads = (*(const __gm__ uint32_t *)(p_tilingdata + 12));
    tilingdata->blockSize = (*(const __gm__ uint32_t *)(p_tilingdata + 16));
    tilingdata->batchSeqLen = (*(const __gm__ uint32_t *)(p_tilingdata + 20));
    tilingdata->batch = (*(const __gm__ uint32_t *)(p_tilingdata + 24));
    tilingdata->numCore = (*(const __gm__ uint32_t *)(p_tilingdata + 28));

    //YF_LOG("numTokens: %d\n", tilingdata->numTokens);
    //YF_LOG("headDim: %d\n", tilingdata->headDim);
    //YF_LOG("numBlocks: %d\n", tilingdata->numBlocks);
    //YF_LOG("numHeads: %d\n", tilingdata->numHeads);
    //YF_LOG("blockSize: %d\n", tilingdata->blockSize);
    //YF_LOG("batchSeqLen: %d\n", tilingdata->batchSeqLen);
    //YF_LOG("batch: %d\n", tilingdata->batch);
    //YF_LOG("numCore: %d\n", tilingdata->numCore);
}

extern "C" __global__ __aicore__ void reshape_and_cache_bnsd(GM_ADDR keyIn, GM_ADDR keyCacheIn, GM_ADDR slotMapping, GM_ADDR seqLen, GM_ADDR keyCacheOut, GM_ADDR workspace, GM_ADDR tiling) {
    // ReshapeAndCacheBNSDTilingData tilingData;
    // InitTilingData(tiling, &tilingData);

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);

    ReshapeAndCacheBnsd op(tilingData);
    op.Init(keyIn, keyCacheIn, slotMapping, seqLen, keyCacheOut);
    op.Process();
}