#ifndef FUSED_QUANT_LI_MANAGE_REQUEST_CONTRACT_H
#define FUSED_QUANT_LI_MANAGE_REQUEST_CONTRACT_H

#include "kernel_operator.h"
#include "fused_quant_lightning_indexer_manage_tiling.h"
#include "lim_cache_budget.h"

namespace lim_request_contract {
using namespace AscendC;

// Only metadata is checked here. Sparse pool contents and distinct active
// pool rows are caller invariants, as in the A3 standard operator.
class Metadata {
public:
    __aicore__ inline void Init(
        GM_ADDR actualSeqLengthsQuery, GM_ADDR finalSeqLengthsKv,
        GM_ADDR offloadedLens, GM_ADDR numCacheTokens,
        GM_ADDR requestState, GM_ADDR reqPoolEntries,
        const FusedQuantLightningIndexerManageTilingData *tiling)
    {
        queryEnds_.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsQuery);
        finalLengths_.SetGlobalBuffer((__gm__ int32_t *)finalSeqLengthsKv);
        prefixes_.SetGlobalBuffer((__gm__ int32_t *)offloadedLens);
        budgets_.SetGlobalBuffer((__gm__ int32_t *)numCacheTokens);
        states_.SetGlobalBuffer((__gm__ int32_t *)requestState);
        entries_.SetGlobalBuffer((__gm__ int32_t *)reqPoolEntries);
        tiling_ = tiling;
    }

    __aicore__ inline bool Valid(uint32_t batch) const
    {
        return Valid(batch, PriorMaximum(batch));
    }

    // The LI scheduler computes the running maximum once per core and uses
    // this overload while caching request validity for repeated task walks.
    __aicore__ inline bool Valid(uint32_t batch, int32_t priorMaximum) const
    {
        const int32_t start = batch == 0U ? 0 : queryEnds_.GetValue(batch - 1U);
        const int32_t end = queryEnds_.GetValue(batch);
        if (start < priorMaximum || end <= start ||
            end > static_cast<int32_t>(tiling_->totalQueryRows) ||
            (batch + 1U == tiling_->batchSize &&
             end != static_cast<int32_t>(tiling_->totalQueryRows))) return false;
        const int32_t routes = end - start;
        if (routes > static_cast<int32_t>(lim_request_limits::MAX_ROUTES)) return false;

        const int32_t state = states_.GetValue(batch);
        if (state != lim_request_limits::IDENTITY_STATE &&
            state != lim_request_limits::FIRST_FILL_STATE &&
            state != lim_request_limits::STEADY_STATE) return false;
        const int32_t finalLen = finalLengths_.GetValue(batch);
        const int32_t row = entries_.GetValue(batch);
        if (finalLen < routes ||
            finalLen > static_cast<int32_t>(tiling_->sourceCapacity) ||
            row < 0 || row >= static_cast<int32_t>(tiling_->poolSize))
            return false;
        if (state == lim_request_limits::IDENTITY_STATE) return true;

        const int32_t prefix = prefixes_.GetValue(batch);
        const int32_t budget = budgets_.GetValue(batch);
        if (prefix < static_cast<int32_t>(lim_request_limits::TOPK) ||
            (prefix & static_cast<int32_t>(lim_request_limits::BLOCK_SIZE - 1U)) != 0 ||
            prefix > finalLen - (routes - 1) ||
            finalLen - prefix > static_cast<int32_t>(lim_request_limits::MAX_TAIL_TOKENS) ||
            budget <= 0) return false;
        return lim_cache_budget::Valid(
            static_cast<uint32_t>(budget), static_cast<uint32_t>(prefix),
            static_cast<uint32_t>(routes));
    }

    __aicore__ inline int32_t QueryEnd(uint32_t batch) const
    {
        return queryEnds_.GetValue(batch);
    }

    // Invalid cumulative boundaries can overlap an earlier valid request.
    // Only rows beyond every earlier end are safe for a failed publisher.
    __aicore__ inline void SafeRouteRange(
        uint32_t batch, uint32_t &start, uint32_t &end) const
    {
        int32_t lower = PriorMaximum(batch);
        const int32_t rawEnd = queryEnds_.GetValue(batch);
        const int32_t capacity = static_cast<int32_t>(tiling_->totalQueryRows);
        if (lower > capacity) lower = capacity;
        start = static_cast<uint32_t>(lower);
        end = rawEnd > lower && rawEnd <= capacity
            ? static_cast<uint32_t>(rawEnd) : start;
    }

private:
    __aicore__ inline int32_t PriorMaximum(uint32_t batch) const
    {
        int32_t maximum = 0;
        for (uint32_t previous = 0; previous < batch; ++previous) {
            const int32_t end = queryEnds_.GetValue(previous);
            if (end > maximum) maximum = end;
        }
        return maximum;
    }

    GlobalTensor<int32_t> queryEnds_;
    GlobalTensor<int32_t> finalLengths_;
    GlobalTensor<int32_t> prefixes_;
    GlobalTensor<int32_t> budgets_;
    GlobalTensor<int32_t> states_;
    GlobalTensor<int32_t> entries_;
    const FusedQuantLightningIndexerManageTilingData *tiling_ = nullptr;
};
} // namespace lim_request_contract

#endif
