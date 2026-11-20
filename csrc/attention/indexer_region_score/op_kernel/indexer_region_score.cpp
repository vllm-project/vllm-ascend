// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#include "indexer_region_score_common.h"
#include "indexer_region_score_paged.h"
#include "indexer_region_score_select.h"

using namespace IndexerRegionScorePaged;

template <bool PARTIAL_PAGES, bool FOUR_HEADS>
__aicore__ inline void ComputeScore(GM_ADDR query, GM_ADDR weights, GM_ADDR pagedKeySummary,
    GM_ADDR actualKvLengths, GM_ADDR blockTable, GM_ADDR cuSeqLens, GM_ADDR output,
    const PagedTilingData &data, AscendC::TPipe *pipe)
{
    ScalarPagedKeyResidentBringup<PARTIAL_PAGES> key;
    KeyResidentCompute<ScalarPagedKeyResidentBringup<PARTIAL_PAGES>, FOUR_HEADS> compute;
    compute.Init(query, weights, actualKvLengths, output, &data, pipe, &key);
    key.Init(pagedKeySummary, blockTable, cuSeqLens, &data, pipe);
    compute.Process();
    compute.Release();
    key.Release();
}

template <bool FOUR_HEADS>
__aicore__ inline void DispatchScore(GM_ADDR query, GM_ADDR weights, GM_ADDR pagedKeySummary,
    GM_ADDR actualKvLengths, GM_ADDR blockTable, GM_ADDR cuSeqLens, GM_ADDR output,
    const PagedTilingData &data, AscendC::TPipe *pipe)
{
    if (data.pageLoadMode == PAGE_LOAD_WITH_OFFSET) {
        ComputeScore<true, FOUR_HEADS>(query, weights, pagedKeySummary, actualKvLengths,
            blockTable, cuSeqLens, output, data, pipe);
    } else {
        ComputeScore<false, FOUR_HEADS>(query, weights, pagedKeySummary, actualKvLengths,
            blockTable, cuSeqLens, output, data, pipe);
    }
}

extern "C" __global__ __aicore__ void indexer_region_score(
    GM_ADDR query, GM_ADDR weights, GM_ADDR pagedKeySummary, GM_ADDR actualKvLengths,
    GM_ADDR blockTable, GM_ADDR cuSeqLens, GM_ADDR output, GM_ADDR selectedRegionIds,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(PagedTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (TILING_KEY_IS(13)) {
        PagedTilingData data{};
        auto local = reinterpret_cast<uint32_t *>(&data);
        auto global = reinterpret_cast<__gm__ uint32_t *>(tiling);
        for (uint32_t i = 0U; i < sizeof(data) / sizeof(uint32_t); ++i) {
            local[i] = global[i];
        }
        AscendC::TPipe pipe;
        if (data.batchTokens != 0U) {
            output = GetUserWorkspace(workspace) + data.scorePoolOffset;
        }
        do {
            data.batchCount = data.batchTokens == 0U ? data.totalTokens :
                PagedMin(data.batchTokens, data.totalTokens - data.batchBegin);
            if (data.headsPerGroup == PAGED_H) {
                DispatchScore<true>(query, weights, pagedKeySummary, actualKvLengths,
                    blockTable, cuSeqLens, output, data, &pipe);
            } else {
                DispatchScore<false>(query, weights, pagedKeySummary, actualKvLengths,
                    blockTable, cuSeqLens, output, data, &pipe);
            }
            AscendC::SyncAll<false>();
            if ASCEND_IS_AIV {
                pipe.Reset();
                RegionSelector select;
                select.Init(output, actualKvLengths, cuSeqLens, selectedRegionIds, data, &pipe);
                select.Process();
            }
            if (data.batchTokens == 0U) break;
            data.batchBegin += data.batchCount;
            if (data.batchBegin == data.totalTokens) break;
            AscendC::SyncAll<false>();
            pipe.Reset();
            if ASCEND_IS_AIC {
                // Reset preserves TPipe's three initial M_MTE1 hardware permits.
                // Reserve their IDs again; Destroy consumes them after the last batch.
                for (uint32_t i = 0U; i < 3U; ++i) {
                    (void)pipe.AllocEventID<HardEvent::M_MTE1>();
                }
            }
        } while (data.batchBegin < data.totalTokens);
    }
}
