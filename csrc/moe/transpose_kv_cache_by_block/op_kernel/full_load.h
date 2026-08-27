#include "common.h"

template <typename T>
class TransposeKvCacheByBlockKernelFullLoad {
 protected:
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> vecInQueue_;
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
    // tiling info
    uint32_t useCoreNum_;
    uint32_t blockPerCore_;
    uint32_t tailCoreNum_;
    uint32_t calBlockNum_;

    uint32_t srcFactor_;
    uint32_t dstFactor_;
    uint32_t copyOutLength_;
    uint32_t dataBlockSize_;

    __aicore__ inline void CopyIn(GlobalTensor<T> &cacheGm, uint32_t offsetBlock, DataCopyParams &repeatParams) {
        LocalTensor<T> cacheLocal = vecInQueue_.AllocTensor<T>();
        for (uint32_t i = 0; i < splitNum_; ++i) {
            DataCopy(cacheLocal[i * dstFactor_], cacheGm[i * srcFactor_ + offsetBlock], repeatParams);
        }
        vecInQueue_.EnQue(cacheLocal);
    }

    __aicore__ inline void CopyOut(GlobalTensor<T> &cacheGm, uint32_t offsetBlock) {
        LocalTensor<T> cacheLocal = vecInQueue_.DeQue<T>();
        DataCopy(cacheGm[offsetBlock], cacheLocal, copyOutLength_);
        vecInQueue_.FreeTensor(cacheLocal);
    }

    __aicore__ inline void SetGlobalBuffers(uint32_t layerId) {
        kCacheGm_.SetGlobalBuffer(GetTensorAddr<T>(layerId, kCachePtr_));
        vCacheGm_.SetGlobalBuffer(GetTensorAddr<T>(layerId, vCachePtr_));
    }

    __aicore__ inline void Caloffset(uint32_t &startBlock, uint32_t &endBlock, uint32_t &startLayer, uint32_t &endLayer) {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t curBlockStart;
        uint32_t curBlocknum;

        if (blockIdx < tailCoreNum_) {
            curBlockStart = blockIdx * (blockPerCore_ + 1);
            curBlocknum = blockPerCore_ + 1;
        } else {
            curBlockStart = blockIdx * blockPerCore_ + tailCoreNum_;
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

        tPipe->InitBuffer(vecInQueue_, 1, TOTAL_UB_SIZE);
        srcFactor_ = blockSize_ * headNum_ / splitNum_ * headDim_;
        dstFactor_ = headNum_ / splitNum_ * headDim_;
        copyOutLength_ = blockSize_ * headNum_ * headDim_;
        dataBlockSize_ = static_cast<uint32_t>(AscendC::GetDataBlockSizeInBytes());
    }

    __aicore__ inline void Process() {
        DataCopyParams repeatParams;
        repeatParams.blockCount = blockSize_;
        repeatParams.blockLen = headNum_ / splitNum_ * headDim_ * sizeof(T) / dataBlockSize_;
        repeatParams.srcStride = 0;
        repeatParams.dstStride = (headNum_ * headDim_ - headNum_ / splitNum_ * headDim_) * sizeof(T) / dataBlockSize_;

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

                CopyIn(kCacheGm_, offsetBlock, repeatParams);
                CopyOut(kCacheGm_, offsetBlock);

                CopyIn(vCacheGm_, offsetBlock, repeatParams);
                CopyOut(vCacheGm_, offsetBlock);
            }

        }
    }
};