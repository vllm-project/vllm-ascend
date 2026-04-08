#include "kernel_utils.h"

constexpr int32_t ALGIN = 32;
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
        pipe.InitBuffer(ubBuf_, RoundUp(blockSize_ * headDim_, ALGIN));
        tmpTensor_ = ubBuf_.Get<uint8_t>();
        keyInGm_.SetGlobalBuffer((__gm__ uint8_t *)keyIn);
        keyCacheInGm_.SetGlobalBuffer((__gm__ uint8_t *)keyCacheIn);
        slotMappingGm_.SetGlobalBuffer((__gm__ int32_t *)slotMapping);
        seqLenGm_.SetGlobalBuffer((__gm__ int32_t *)seqLen);
        keyCacheOutGm_.SetGlobalBuffer((__gm__ uint8_t *)keyCacheOut);
    }

    __aicore__ inline void Process()
    {
        // 计算一共有多少页
        uint32_t totalBlockNum = 0;
        uint32_t offsetInSlotmapping = 0; // 不同batch的第一个token在slotmapping这个tensor中的偏移位置
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
        // 每个核搬运多少页
        uint32_t blockNumPerCore = totalBlockNum / actualCoreNum;
        uint32_t leftBlockNum = totalBlockNum - blockNumPerCore * actualCoreNum; // 均分完，还剩下这些block
        uint32_t blockNum = blockIdx_ < leftBlockNum ? blockNumPerCore + 1 : blockNumPerCore;
        uint32_t startBlockOffset_ = blockIdx_ < leftBlockNum ? (blockNumPerCore * blockIdx_ + blockIdx_) :
                                                       (blockNumPerCore * blockIdx_ + leftBlockNum);
        if (blockIdx_ >= actualCoreNum) { // 总任务量过少，其余核直接返回，不往下执行
            return;
        }

        // 每个核的startBlockOffset_对应的keyIn和KeyCache位置；
        uint32_t startBatchIdx = 0;
        uint32_t accuBlockNum = 0;
        uint32_t startTokenOffsetInBatch = 0; // 在batch内的第几个token
        offsetInSlotmapping = 0;
        bool copyFromBatchStart = true; // 表示是否是从当前batch的第一个token开始
        for (uint32_t batchIdx = 0; batchIdx < batchNum_; batchIdx++) {
            uint32_t seqLen = seqLenGm_.GetValue(batchIdx);
            int32_t slotValue = slotMappingGm_.GetValue(offsetInSlotmapping);
            uint32_t offsetInBlock = slotValue % blockSize_;
            uint32_t leftTokenNum = blockSize_ - offsetInBlock;
            uint32_t blockNumForCurrBatch = seqLen < leftTokenNum ? 1 :
                                                                   (CeilDiv(seqLen - leftTokenNum, blockSize_) + 1);
            accuBlockNum += blockNumForCurrBatch;

            //YF_LOG("batchIdx: %d, offsetInBlock: %d, leftTokenNum: %d, blockNumForCurrBatch: %d, accuBlockNum: %d, startBlockOffset_: %d\n", batchIdx, offsetInBlock, leftTokenNum, blockNumForCurrBatch, accuBlockNum, startBlockOffset_);

            if (startBlockOffset_ == 0) { // 从第一块开始做，什么都不需要更新
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
                    blockSize_ +leftTokenNum; // 1是去掉第一个block
                copyFromBatchStart = false;
                offsetInSlotmapping = offsetInSlotmapping + startTokenOffsetInBatch;
                //YF_LOG("batchIdx: %d, startTokenOffsetInBatch: %d, offsetInSlotmapping: %d\n", batchIdx, startTokenOffsetInBatch, offsetInSlotmapping);
                break;
            }
            offsetInSlotmapping += seqLen;
        }
        // 有了起始位置，开始拷贝
        uint32_t batchIdx = startBatchIdx;
        for (uint32_t blockIdx = 0; blockIdx < blockNum; blockIdx++) {
            uint32_t seqLen = seqLenGm_.GetValue(batchIdx);
            int32_t slotValue = slotMappingGm_.GetValue(offsetInSlotmapping);
            uint32_t blockId = static_cast<uint32_t>(slotValue) / blockSize_;
            uint32_t slotId = static_cast<uint32_t>(slotValue) % blockSize_;
            //YF_LOG("batchIdx: %d, seqLen: %d, slotValue: %d, blockId: %d, slotId: %d\n", batchIdx, seqLen, slotValue, blockId, slotId);

            if (startTokenOffsetInBatch + blockSize_ - slotId > seqLen) { // 剩余空间足够大数据拷贝的时候32字节对齐向上拷贝
                // 这次能把这个batch的内容拷完
                //YF_LOG("batchIdx: %d, true\n", batchIdx);
                uint32_t currCopyTokenNum = seqLen - startTokenOffsetInBatch;
                uint32_t copyBlocks = CeilDiv(currCopyTokenNum * headDim_, 32); // 向上对齐
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
                // 更新下一个batch的信息
                batchIdx += 1;
                startTokenOffsetInBatch = 0;
                offsetInSlotmapping += currCopyTokenNum;
            } else {
                uint32_t currCopyTokenNum = blockSize_ - slotId;
                uint32_t copyBlocks = currCopyTokenNum * headDim_ / ALGIN;
                //YF_LOG("batchIdx: %d, currCopyTokenNum: %d, currCopyTokenNum * headDim_: %d\n", batchIdx, currCopyTokenNum, currCopyTokenNum * headDim_);
                uint32_t leftBytes = currCopyTokenNum * headDim_ - copyBlocks * ALGIN;
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
                
                // 更新token的位置
                if (currCopyTokenNum + startTokenOffsetInBatch == seqLen) { // 刚好写完要更新batchIdx等
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
                // 如果有尾块的话，处理一下尾块，不足32Bytes
                for (uint32_t headId = 0; headId < headNum_; headId++) {
                    for (uint32_t dimId = 0; dimId < leftBytes; dimId++) {
                        uint8_t cacheValue = keyInGm_.GetValue(srcOffset + headId * seqLen * headDim_ +
                            copyBlocks * ALGIN + dimId);
                        keyCacheOutGm_.SetValue(dstOffset + headId * blockSize_ * headDim_ +
                            copyBlocks * ALGIN + dimId, cacheValue);
                    }
                }
                AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(keyCacheOutGm_);
                // 否则部分数据无法正确刷出
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