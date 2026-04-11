#ifndef DISPATCH_LAYOUT_H
#define DISPATCH_LAYOUT_H

#include <climits>
#include "kernel_operator.h"

#include "../common/comm_args.h"
#include "../common/data_copy.h"
#include "../common/sync_collectives.h"
#include "../common/moe_distribute_base.h"
#include "dispatch_layout_tiling.h"

using namespace AscendC;
using namespace Moe;

constexpr uint32_t UB_32_ALIGN = 32U;
constexpr uint32_t AIV_NUM = 48;

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

template <typename T>
class DispatchLayout {

public:
    __aicore__ inline DispatchLayout() {};

    __aicore__ inline void Init(GM_ADDR topkIdx, GM_ADDR numTokensPerRank, GM_ADDR numTokensPerExpert, GM_ADDR isTokenInRank, 
                                GM_ADDR workspace, TPipe *pipe, const DispatchLayoutTilingData *tilingData)
    {
        numTokens_ = tilingData->dispatchLayoutInfo.numTokens;
        numRanks_ = tilingData->dispatchLayoutInfo.numRanks;
        numExperts_ = tilingData->dispatchLayoutInfo.numExperts;
        numTopk_ = tilingData->dispatchLayoutInfo.numTopk;
        tpipe_ = pipe;

        coreIdx_ = GetBlockIdx();
        uint32_t temp = numTokens_ / AIV_NUM;
        uint32_t restNum = numTokens_ % AIV_NUM;
        int64_t topkIdxOffset;
        int64_t isTokenOffset;
        tempTokens_ = temp;
        if (coreIdx_ < restNum) {
            tempTokens_++;
        }
        topkIdx32AlignIntLen_ = Ceil(tempTokens_ * numTopk_ * sizeof(int64_t), UB_32_ALIGN) * UB_32_ALIGN;
        numTokensPerRank32AlignIntLen_ = Ceil(numRanks_ * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN;
        numTokensPerExpert32AlignIntLen_ = Ceil(numExperts_ * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN;
        isTokenInRank32AlignIntLen_ = Ceil(tempTokens_ * numRanks_ * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN;

        if (coreIdx_ < restNum) {
            topkIdxOffset = coreIdx_ * topkIdx32AlignIntLen_;
            isTokenOffset = coreIdx_ * isTokenInRank32AlignIntLen_;
        } else {
            topkIdxOffset = restNum * Ceil((tempTokens_ + 1) * numTopk_ * sizeof(int64_t), UB_32_ALIGN) * UB_32_ALIGN
                            + (coreIdx_ - restNum) * topkIdx32AlignIntLen_;
            isTokenOffset = restNum * Ceil((tempTokens_ + 1) * numRanks_ * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN
                            + (coreIdx_ - restNum) * isTokenInRank32AlignIntLen_;  
        }

        topkIdxGM_.SetGlobalBuffer((__gm__ int64_t*)(topkIdx + topkIdxOffset));
        numTokensPerRankGM_.SetGlobalBuffer((__gm__ T*)numTokensPerRank);
        numTokensPerExpertGM_.SetGlobalBuffer((__gm__ T*)numTokensPerExpert);
        isTokenInRankGM_.SetGlobalBuffer((__gm__ T*)(isTokenInRank + isTokenOffset));

        
    }

    __aicore__ inline void Process()
    {
        tpipe_->Reset();
        tpipe_->InitBuffer(topkIdxBuf_, topkIdx32AlignIntLen_);
        tpipe_->InitBuffer(numTokensPerRankBuf_, numTokensPerRank32AlignIntLen_);
        tpipe_->InitBuffer(numTokensPerExpertBuf_, numTokensPerExpert32AlignIntLen_);
        tpipe_->InitBuffer(isTokenInRankBuf_, isTokenInRank32AlignIntLen_);
        tpipe_->InitBuffer(seenRankBuf_, numRanks_ * sizeof(T));

        LocalTensor<int64_t> topkIdxTensor = topkIdxBuf_.AllocTensor<int64_t>();
        const DataCopyExtParams dataCopyParams{1U, topkIdx32AlignIntLen_, 0U, 0U, 0U};
        const DataCopyPadExtParams<int64_t> padParams{false, 0U, 0U, 0U};
        DataCopyPad(topkIdxTensor, topkIdxGM_, dataCopyParams, padParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        LocalTensor<T> numTokensPerRankTensor = numTokensPerRankBuf_.AllocTensor<T>();
        LocalTensor<T> numTokensPerExpertTensor = numTokensPerExpertBuf_.AllocTensor<T>();
        LocalTensor<T> isTokenInRankTensor = isTokenInRankBuf_.AllocTensor<T>();
        LocalTensor<T> seenRankTensor = seenRankBuf_.AllocTensor<T>();
        Duplicate<T>(numTokensPerRankTensor, 0, numRanks_);
        Duplicate<T>(numTokensPerExpertTensor, 0, numExperts_);
        Duplicate<T>(isTokenInRankTensor, 0, tempTokens_ * numRanks_);
        SyncFunc<AscendC::HardEvent::V_S>();

        int experts_per_rank = numExperts_ / numRanks_;
        for (int i = 0; i < tempTokens_; ++i) {
            SyncFunc<AscendC::HardEvent::S_V>();
            Duplicate<T>(seenRankTensor, 0, numRanks_);
            SyncFunc<AscendC::HardEvent::V_S>();
            for (int j = 0; j < numTopk_; ++j) {
                int64_t expert_idx = topkIdxTensor.GetValue(i * numTopk_ + j);
                uint32_t per_expert_num = numTokensPerExpertTensor.GetValue(expert_idx) + 1;
                numTokensPerExpertTensor.SetValue(expert_idx, per_expert_num);
                int rank_id = expert_idx / experts_per_rank;
                if (!seenRankTensor.GetValue(rank_id)) {
                    uint32_t per_rank_num = numTokensPerRankTensor.GetValue(rank_id) + 1;
                    isTokenInRankTensor.SetValue(i * numRanks_ + rank_id, 1);
                    seenRankTensor.SetValue(rank_id, 1);
                    numTokensPerRankTensor.SetValue(rank_id, per_rank_num);
                }
            }
        }

        const DataCopyExtParams isTokenInRankDataCopyParams{1U, isTokenInRank32AlignIntLen_, 0U, 0U, 0U};
        DataCopyPad(isTokenInRankGM_, isTokenInRankTensor, isTokenInRankDataCopyParams);
        AscendC::SetAtomicAdd<T>();
        const DataCopyExtParams numTokensPerRankDataCopyParams{1U, numTokensPerRank32AlignIntLen_, 0U, 0U, 0U};
        DataCopyPad(numTokensPerRankGM_, numTokensPerRankTensor, numTokensPerRankDataCopyParams);
        const DataCopyExtParams numTokensPerExpertDataCopyParams{1U, numTokensPerExpert32AlignIntLen_, 0U, 0U, 0U};
        DataCopyPad(numTokensPerExpertGM_, numTokensPerExpertTensor, numTokensPerExpertDataCopyParams);
        AscendC::SetAtomicNone();
    }

private:
    GlobalTensor<int64_t> topkIdxGM_;
    GlobalTensor<T> numTokensPerRankGM_;
    GlobalTensor<T> numTokensPerExpertGM_;
    GlobalTensor<T> isTokenInRankGM_;

    TBuf<> topkIdxBuf_;
    TBuf<> numTokensPerRankBuf_;
    TBuf<> numTokensPerExpertBuf_;
    TBuf<> isTokenInRankBuf_;
    TBuf<> seenRankBuf_;

    TPipe *tpipe_{nullptr};
    uint32_t numTokens_{0};
    uint32_t numRanks_{0};
    uint32_t numExperts_{0};
    uint32_t numTopk_{0};
    uint32_t coreIdx_{0};
    uint32_t tempTokens_{0};

    uint32_t topkIdx32AlignIntLen_{0};
    uint32_t numTokensPerRank32AlignIntLen_{0};
    uint32_t numTokensPerExpert32AlignIntLen_{0};
    uint32_t isTokenInRank32AlignIntLen_{0};
};

#endif  // DISPATCH_LAYOUT_H
