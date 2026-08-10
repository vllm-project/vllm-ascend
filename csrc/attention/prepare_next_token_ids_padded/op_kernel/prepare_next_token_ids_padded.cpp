#include "kernel_operator.h"
#include "prepare_next_token_ids_padded.h"


namespace{
    constexpr uint32_t BLOCK_BYTES = 32;
    constexpr uint32_t INT32_BYTES = sizeof(int32_t);
    constexpr uint32_t INT32_ELEMENTS_PER_BLOCK = BLOCK_BYTES / INT32_BYTES;

    __aicore__ inline uint32_t AlignUp(
        uint32_t value, uint32_t alignment)
    {
        return (value + alignment - 1) / alignment * alignment;
    }
}


class KernelPrepareNextTokenIdsPadded {
public:
    __aicore__ inline KernelPrepareNextTokenIdsPadded(){};

    __aicore__ inline void Init(
        GM_ADDR sampledTokenIds,
        GM_ADDR discardRequestMask,
        GM_ADDR backupNextTokenIds,
        GM_ADDR nextTokenIds,
        GM_ADDR validSampledTokensCount,
        GM_ADDR workspace,
        GM_ADDR tiling
    )
    {
        (void)workspace;

        REGISTER_TILING_DEFAULT(PrepareNextTokenIdsPaddedTilingInfo);

        GET_TILING_DATA_WITH_STRUCT(
            PrepareNextTokenIdsPaddedTilingInfo,
            tilingData,
            tiling);
        
        batchSize_ = tilingData.batchSize;
        sampledTokensPerRequest_ = tilingData.sampledTokensPerRequest;
        sampledTokensAligned_ = tilingData.sampledTokensAligned;

        usedCoreNum_ = tilingData.usedCoreNum;
        rowsPerCore_ = tilingData.rowsPerCore;
        extraRowCoreNum_ = tilingData.extraRowCoreNum;
        rowsPerTile_ = tilingData.rowsPerTile;
        vocabSize_ = tilingData.vocabSize;


        sampledTokenIdsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)sampledTokenIds,
            static_cast<uint64_t>(batchSize_) * 
                sampledTokensPerRequest_);
        
        discardRequestMaskGm_.SetGlobalBuffer(
            (__gm__ uint8_t *)discardRequestMask,
            batchSize_);
        
        backupNextTokenIdsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)backupNextTokenIds,
            batchSize_);
        
        nextTokenIdsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)nextTokenIds,
            batchSize_);
        
        validSampledTokensCountGm_.SetGlobalBuffer(
            (__gm__ int32_t *)validSampledTokensCount,
            batchSize_);
        
        uint32_t blockIdx = AscendC::GetBlockIdx();
        if(blockIdx >= usedCoreNum_){
            rowsForthisCore_ = 0;
            coreRowOffset_ = 0;
            return;
        }

        if(blockIdx < extraRowCoreNum_){
            rowsForthisCore_ = rowsPerCore_ + 1;
            coreRowOffset_ = blockIdx * rowsForthisCore_;
        } else {
            rowsForthisCore_ = rowsPerCore_;
            coreRowOffset_ = extraRowCoreNum_ * (rowsPerCore_ + 1) + 
                (blockIdx - extraRowCoreNum_) * rowsPerCore_;
        }

        uint32_t sampleBufferBytes = rowsPerTile_ * sampledTokensAligned_ * INT32_BYTES;

        uint32_t maskBufferBytes = AlignUp(rowsPerTile_, BLOCK_BYTES);

        uint32_t metadataBufferBytes = AlignUp(rowsPerTile_ * INT32_BYTES, BLOCK_BYTES);

        pipe_.InitBuffer(sampledTokenIdsQueue_, 1, sampleBufferBytes);
        pipe_.InitBuffer(discardRequestMaskQueue_, 1, maskBufferBytes);
        pipe_.InitBuffer(backupNextTokenIdsQueue_, 1, metadataBufferBytes);
        pipe_.InitBuffer(nextTokenIdsQueue_, 1, metadataBufferBytes);
        pipe_.InitBuffer(validSampledTokensCountQueue_, 1, metadataBufferBytes);
    }

    __aicore__ inline void Process()
    {
        if(rowsForthisCore_ == 0){
            return;
        }

        uint32_t processedRows = 0;

        while(processedRows < rowsForthisCore_){
            uint32_t remainingRows = rowsForthisCore_ - processedRows;

            uint32_t currentRows = remainingRows < rowsPerTile_ 
                                ? remainingRows 
                                : rowsPerTile_;

            uint32_t globalRowOffset = coreRowOffset_ + processedRows;

            CopyIn(globalRowOffset, currentRows);
            Compute(currentRows);
            CopyOut(globalRowOffset, currentRows);
            
            processedRows += currentRows;
        }
    }

private:
    __aicore__ inline void CopyIn(
        uint32_t globalRowOffset,
        uint32_t currentRows
    )
    {
        auto sampledLocal = sampledTokenIdsQueue_.AllocTensor<int32_t>();
        auto discardMaskLocal = discardRequestMaskQueue_.AllocTensor<uint8_t>();
        auto backupLocal = backupNextTokenIdsQueue_.AllocTensor<int32_t>();

        uint32_t sampledRightPadding = sampledTokensAligned_ - sampledTokensPerRequest_;

        AscendC::DataCopyExtParams sampledCopyParams{
            1,
            sampledTokensPerRequest_ * INT32_BYTES,
            0,
            0,
            0
        };

        AscendC::DataCopyPadExtParams<int32_t> sampledPadParams
        {
            true,
            0,
            static_cast<uint8_t>(sampledRightPadding),
            -1
        };

        for (uint32_t row = 0; row < currentRows; ++row){
            uint64_t gmOffset = 
                static_cast<uint64_t>(
                    globalRowOffset + row) * sampledTokensPerRequest_;
            uint32_t localOffset = row * sampledTokensAligned_;
            
            AscendC::DataCopyPad(
                sampledLocal[localOffset],
                sampledTokenIdsGm_[gmOffset],
                sampledCopyParams,
                sampledPadParams);
        }

        uint32_t maskAlignedBytes = AlignUp(currentRows, BLOCK_BYTES);
        uint32_t maskRightPadding = maskAlignedBytes - currentRows;

        AscendC::DataCopyExtParams maskCopyParams{
            1,
            currentRows,
            0,
            0,
            0
        };

        AscendC::DataCopyPadExtParams<uint8_t> 
            maskPadParams{
                true,
                0,
                static_cast<uint8_t>(maskRightPadding),
                0
            };
        AscendC::DataCopyPad(
            discardMaskLocal,
            discardRequestMaskGm_[globalRowOffset],
            maskCopyParams,
            maskPadParams);
        
        uint32_t backupAlignedElements = AlignUp(currentRows, INT32_ELEMENTS_PER_BLOCK);
        uint32_t backupRightPadding = backupAlignedElements - currentRows;
        
        AscendC::DataCopyExtParams backupCopyParams{
            1,
            currentRows * INT32_BYTES,
            0,
            0,
            0
        };

        AscendC::DataCopyPadExtParams<int32_t> 
            backupPadParams{
                true,
                0,
                static_cast<uint8_t>(backupRightPadding),
                0
            };
        
        AscendC::DataCopyPad(
            backupLocal,
            backupNextTokenIdsGm_[globalRowOffset],
            backupCopyParams,
            backupPadParams);
        
        sampledTokenIdsQueue_.EnQue(
            sampledLocal
        );

        discardRequestMaskQueue_.EnQue(
            discardMaskLocal
        );

        backupNextTokenIdsQueue_.EnQue(
            backupLocal
        );
    }

    __aicore__ inline void Compute(uint32_t currentRows)
    {
        auto sampledLocal = sampledTokenIdsQueue_
                            .DeQue<int32_t>();
        auto discardMaskLocal = discardRequestMaskQueue_
                                .DeQue<uint8_t>();
        auto backupLocal = backupNextTokenIdsQueue_
                                .DeQue<int32_t>();

        auto nextTokenIdsLocal = nextTokenIdsQueue_
                                .AllocTensor<int32_t>();
        auto validSampledTokensCountLocal = validSampledTokensCountQueue_
                                            .AllocTensor<int32_t>();

        for (uint32_t row = 0; row < currentRows; ++row){
            bool shouldDiscard = 
                discardMaskLocal.GetValue(row) != 0;
            
            int32_t validCount = 0;
            
            if (!shouldDiscard) {
                uint32_t rowOffset = 
                    row * sampledTokensAligned_;
                
                for (uint32_t column = 0;
                    column < sampledTokensPerRequest_;
                    ++column){
                    int32_t tokenId = sampledLocal.GetValue(
                                        rowOffset + column);
                    bool isValid = tokenId != -1 && tokenId < vocabSize_;
                        
                    if (isValid) {
                        ++validCount;
                    }
                }
            }

            int32_t nextTokenId = backupLocal.GetValue(row);
            if (validCount > 0){
                uint32_t selectedOffset = 
                    row * sampledTokensAligned_ + 
                    static_cast<uint32_t>(validCount - 1);

                nextTokenId = sampledLocal.GetValue(selectedOffset);                 
            }
            nextTokenIdsLocal.SetValue(row, nextTokenId);
            validSampledTokensCountLocal.SetValue(row, validCount);
        }

        nextTokenIdsQueue_.EnQue(
            nextTokenIdsLocal
        );

        validSampledTokensCountQueue_.EnQue(
            validSampledTokensCountLocal
        );

        sampledTokenIdsQueue_.FreeTensor(sampledLocal);
        discardRequestMaskQueue_.FreeTensor(discardMaskLocal);
        backupNextTokenIdsQueue_.FreeTensor(backupLocal);
    }

    __aicore__ inline void CopyOut(
        uint32_t globalRowOffset,
        uint32_t currentRows
    )
    {
        auto nextTokenIdsLocal = nextTokenIdsQueue_
                                .DeQue<int32_t>();
        auto validSampledTokensCountLocal = validSampledTokensCountQueue_
                                            .DeQue<int32_t>();

        AscendC::DataCopyExtParams outputCopyParams{
            1,
            currentRows * INT32_BYTES,
            0,
            0,
            0
        };

        AscendC::DataCopyPad(
            nextTokenIdsGm_[globalRowOffset],
            nextTokenIdsLocal,
            outputCopyParams);
        
        AscendC::DataCopyPad(
            validSampledTokensCountGm_[globalRowOffset],
            validSampledTokensCountLocal,
            outputCopyParams
        );

        nextTokenIdsQueue_.FreeTensor(nextTokenIdsLocal);
        validSampledTokensCountQueue_.FreeTensor(validSampledTokensCountLocal);
    }
    


private:
    AscendC::TPipe pipe_;

    AscendC::TQue<AscendC::QuePosition::VECIN,1> sampledTokenIdsQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN,1> discardRequestMaskQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN,1> backupNextTokenIdsQueue_;

    AscendC::TQue<AscendC::QuePosition::VECOUT,1> nextTokenIdsQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT,1> validSampledTokensCountQueue_;

    AscendC::GlobalTensor<int32_t> sampledTokenIdsGm_;
    AscendC::GlobalTensor<uint8_t> discardRequestMaskGm_;
    AscendC::GlobalTensor<int32_t> backupNextTokenIdsGm_;
    AscendC::GlobalTensor<int32_t> nextTokenIdsGm_;
    AscendC::GlobalTensor<int32_t> validSampledTokensCountGm_;

    uint32_t batchSize_ = 0;
    uint32_t sampledTokensPerRequest_ = 0;
    uint32_t sampledTokensAligned_ = 0;

    uint32_t usedCoreNum_ = 0;
    uint32_t rowsPerCore_ = 0;
    uint32_t extraRowCoreNum_ = 0;
    uint32_t rowsPerTile_ = 0;
    
    int32_t vocabSize_ = 0;

    uint32_t coreRowOffset_ = 0;
    uint32_t rowsForthisCore_ = 0;
};


extern "C" __global__ __aicore__
void prepare_next_token_ids_padded(
    GM_ADDR sampledTokenIds,
    GM_ADDR discardRequestMask,
    GM_ADDR backupNextTokenIds,
    GM_ADDR nextTokenIds,
    GM_ADDR validSampledTokensCount,
    GM_ADDR workspace,
    GM_ADDR tiling
)
{
    KernelPrepareNextTokenIdsPadded kernel;
    kernel.Init(
        sampledTokenIds,
        discardRequestMask,
        backupNextTokenIds,
        nextTokenIds,
        validSampledTokensCount,
        workspace,
        tiling
    );
    kernel.Process();
}
