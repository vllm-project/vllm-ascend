#include "common.h"

template <typename T, uint32_t DB, bool needHandleUnFactorSplit>
class TransposeKvCacheByBlockKernelGeneral {
 protected:
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> queBind_;
    GlobalTensor<T> kCacheGm_;
    GlobalTensor<T> vCacheGm_;
    GlobalTensor<int64_t> blockIDsGm_;

    GM_ADDR kCachePtr_;
    GM_ADDR vCachePtr_;

    // shape info
    uint32_t blockNum_;
    uint32_t blockSize_;
    uint32_t headNum_;
    uint32_t headDim_;
    uint32_t splitNum_;
    uint32_t layerNum_;
    uint32_t headNumSplited_;
    uint32_t blockSizeSplitNum_;

    // tiling info
    uint32_t useCoreNum_;
    uint32_t blockPerCore_;
    uint32_t tailCoreNum_;
    uint32_t calBlockNum_;

    uint32_t srcFactor_;
    uint32_t dstFactor_;
    uint32_t copyOutLength_;

    uint32_t blockSizePerTime_;
    uint32_t blockSizePerTimeTail_;

    uint32_t blockIdx_;
    uint32_t dataBlockSize_;
    bool needSync_;

    __aicore__ inline void CopyIn(GlobalTensor<T> &cacheGm, uint32_t offsetBlock, DataCopyParams &repeatParams) {
        LocalTensor<T> cacheLocal = queBind_.AllocTensor<T>();
        for (uint32_t i = 0; i < splitNum_; ++i) {
            DataCopy(cacheLocal[i * dstFactor_], cacheGm[i * srcFactor_ + offsetBlock], repeatParams);
        }
        queBind_.EnQue(cacheLocal);
    }

    __aicore__ inline void CopyOut(GlobalTensor<T> &cacheGm, uint32_t offsetBlock) {
        LocalTensor<T> cacheLocal = queBind_.DeQue<T>();
        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE2>(0x8);
        AscendC::CrossCoreWaitFlag(0x8);
        DataCopy(cacheGm[offsetBlock], cacheLocal, copyOutLength_);
        queBind_.FreeTensor(cacheLocal);
    }

    __aicore__ inline void SetGlobalBuffers(uint32_t layerId) {
        kCacheGm_.SetGlobalBuffer(GetTensorAddr<T>(layerId, kCachePtr_));
        vCacheGm_.SetGlobalBuffer(GetTensorAddr<T>(layerId, vCachePtr_));
    }

    __aicore__ inline void Caloffset(uint32_t &startBlock, uint32_t &endBlock, uint32_t &startLayer, uint32_t &endLayer) {

        uint32_t curBlockStart;
        uint32_t curBlocknum;
        uint32_t groupBlockIdx = blockIdx_ / blockSizeSplitNum_;
        if (groupBlockIdx < tailCoreNum_) {
            needSync_ = false;
            curBlockStart = groupBlockIdx * (blockPerCore_ + 1);
            curBlocknum = blockPerCore_ + 1;
        } else {
            needSync_ = true;
            curBlockStart = groupBlockIdx * blockPerCore_ + tailCoreNum_;
            curBlocknum = blockPerCore_;
        }
        uint32_t curBlockEnd = curBlockStart + curBlocknum;
        startBlock = curBlockStart / layerNum_;
        startLayer = curBlockStart % layerNum_;
        endBlock = (curBlockEnd + layerNum_ - 1) / layerNum_;
        endLayer = curBlockEnd % layerNum_;
        if (endLayer == 0) {
            endLayer = layerNum_;
        }
    }


 public:
    __aicore__ inline void Init(GM_ADDR KCache, GM_ADDR VCache, GM_ADDR blockIDs,
                                TransposeKvCacheByBlockTilingData* tilingData, TPipe* tPipe) {
        kCachePtr_ = KCache;
        vCachePtr_ = VCache;
        blockIDsGm_.SetGlobalBuffer((__gm__ int64_t*)blockIDs);
        blockIdx_ = GetBlockIdx();
        // shape info
        blockSize_ = tilingData->blockSize;
        headNum_ = tilingData->headNum;
        headDim_ = tilingData->headDim;
        splitNum_ = tilingData->splitNum;
        layerNum_ = tilingData->layerNum;
        // tiling info
        useCoreNum_ = tilingData->useCoreNum;
        blockPerCore_ = tilingData->blockPerCore;
        tailCoreNum_ = tilingData->tailCoreNum;
        calBlockNum_ = tilingData->calBlockNum;
        blockSizeSplitNum_ = tilingData->blockSizeSplitNum;
        blockSizePerTime_ = tilingData->blockSizePerTime;
        blockSizePerTimeTail_ = tilingData->blockSizePerTimeTail;
        headNumSplited_ = headNum_ / splitNum_;

        tPipe->InitBuffer(queBind_, DB, TOTAL_UB_SIZE / DB);
        srcFactor_ = blockSize_ * headNumSplited_ * headDim_;
        dstFactor_ = headNumSplited_ * headDim_;
        copyOutLength_ = blockSizePerTime_ * headNum_ * headDim_;
        dataBlockSize_ = static_cast<uint32_t>(AscendC::GetDataBlockSizeInBytes());
    }

    __aicore__ inline void Process() {
        DataCopyParams repeatParams;
        repeatParams.blockCount = blockSizePerTime_;
        repeatParams.blockLen = headNumSplited_ * headDim_ * sizeof(T) / dataBlockSize_;
        repeatParams.srcStride = 0;
        repeatParams.dstStride = (headNum_ * headDim_ - headNumSplited_ * headDim_) * sizeof(T) / dataBlockSize_;

        uint32_t startBlock;
        uint32_t endBlock;
        uint32_t startLayer;
        uint32_t endLayer;

        Caloffset(startBlock, endBlock, startLayer, endLayer);

        for (uint32_t i = startBlock; i < endBlock; ++i) {
            int64_t blockId = blockIDsGm_.GetValue(i);
            uint32_t offsetBlock = blockId * blockSize_ * headNum_ * headDim_;
            uint32_t realStartLayer;
            uint32_t realEndLayer;
            if (i == startBlock) {
                realStartLayer = startLayer;
            } else {
                realStartLayer = 0;
            }

            if (i == (endBlock - 1)) {
                realEndLayer = endLayer;
            } else {
                realEndLayer = layerNum_;
            }
            for (uint32_t layerId = realStartLayer; layerId < realEndLayer; ++layerId) {
                SetGlobalBuffers(layerId);
                uint32_t blockSizeIndex = blockIdx_ % blockSizeSplitNum_;
                uint32_t srcOffset;
                uint32_t dstOffset;
                if constexpr (needHandleUnFactorSplit) {
                    // handle tail
                    if (blockSizeIndex >= blockSizePerTimeTail_) {
                        repeatParams.blockCount = (blockSizePerTime_ - 1);
                        copyOutLength_ = (blockSizePerTime_ - 1) * headNum_ * headDim_;
                        srcOffset = (blockSizeIndex * blockSizePerTime_ - (blockSizeIndex - blockSizePerTimeTail_)) * headNumSplited_ * headDim_;
                        dstOffset = (blockSizeIndex * blockSizePerTime_ - (blockSizeIndex - blockSizePerTimeTail_)) * headNum_ * headDim_;    
                    } else {
                        repeatParams.blockCount = blockSizePerTime_;
                        copyOutLength_ = blockSizePerTime_ * headNum_ * headDim_;
                        srcOffset = blockSizeIndex * blockSizePerTime_ * headNumSplited_ * headDim_;
                        dstOffset = blockSizeIndex * blockSizePerTime_ * headNum_ * headDim_;        
                    }
                } else {
                    repeatParams.blockCount = blockSizePerTime_;
                    copyOutLength_ = blockSizePerTime_ * headNum_ * headDim_;
                    srcOffset = blockSizeIndex * blockSizePerTime_ * headNumSplited_ * headDim_;
                    dstOffset = blockSizeIndex * blockSizePerTime_ * headNum_ * headDim_;
                }

                CopyIn(kCacheGm_, offsetBlock + srcOffset, repeatParams);
                CopyOut(kCacheGm_, offsetBlock + dstOffset);

                CopyIn(vCacheGm_, offsetBlock + srcOffset, repeatParams);
                CopyOut(vCacheGm_, offsetBlock + dstOffset);
            }

        }

        if (needSync_) {
            AscendC::CrossCoreSetFlag<0x0, PIPE_MTE2>(0x8);
            AscendC::CrossCoreWaitFlag(0x8);

            AscendC::CrossCoreSetFlag<0x0, PIPE_MTE2>(0x8);
            AscendC::CrossCoreWaitFlag(0x8);
        }

    }
};