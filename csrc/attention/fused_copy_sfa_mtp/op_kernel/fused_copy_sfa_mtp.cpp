#include "kernel_operator.h"
#define C_TEMPLATE 0
#define V_TEMPLATE 1

// OPC generates only this operator's tiling class.  The fused payload has the
// complete production SFA payload as its prefix, so expose it under the type
// name expected by the shared SFA implementation (the same pattern used by
// the non-MTP fused_copy_sfa kernel).
using CopySfaMtpAttentionTilingDataMla =
    FusedCopySfaMtpTilingData;

#include "sfa_impl/sparse_tail_attention_kernel_mla.h"

using namespace AscendC;
namespace {
__aicore__ inline bool IsBatchFirstFill(
    __gm__ uint8_t *cacheTokens,
    __gm__ uint8_t *missCounts,
    uint32_t batchSize,
    uint32_t missCap)
{
    GlobalTensor<int32_t> cacheTokensGm;
    GlobalTensor<int32_t> missCountsGm;
    cacheTokensGm.SetGlobalBuffer(
        (__gm__ int32_t *)cacheTokens, batchSize);
    missCountsGm.SetGlobalBuffer(
        (__gm__ int32_t *)missCounts, batchSize);
    bool firstFill = false;
    for (uint32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
        const int32_t cacheTokenCount = cacheTokensGm.GetValue(batchIdx);
        const int32_t missCount = missCountsGm.GetValue(batchIdx);
        ASSERT_MSG(cacheTokenCount >= 0,
                   "num_cache_tokens must be non-negative.");
        ASSERT_MSG(missCount >= 0 &&
                       static_cast<uint32_t>(missCount) <= missCap,
                   "miss_count exceeds the request-level input capacity.");
        if (missCount >= cacheTokenCount) {
            firstFill = true;
        }
    }
    return firstFill;
}

template <typename T, bool SOURCE_AWARE>
__aicore__ inline void RunFusedMtp(
    __gm__ uint8_t *query,
    __gm__ uint8_t *key,
    __gm__ uint8_t *value,
    __gm__ uint8_t *sparseIndices,
    __gm__ uint8_t *cacheTokens,
    __gm__ uint8_t *hbmBlockTable,
    __gm__ uint8_t *actualSeqLengthsQuery,
    __gm__ uint8_t *actualSeqLengthsKv,
    __gm__ uint8_t *queryRope,
    __gm__ uint8_t *hbmKeyRope,
    __gm__ uint8_t *dramKeyRope,
    __gm__ uint8_t *dramKvCache,
    __gm__ uint8_t *dramBlockTable,
    __gm__ uint8_t *topkSourceIds,
    __gm__ uint8_t *topkMissCounts,
    __gm__ uint8_t *missSourceIds,
    __gm__ uint8_t *missDstSlots,
    __gm__ uint8_t *missCounts,
    __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *attentionWorkspace,
    const FusedCopySfaMtpTilingData *fusedTiling,
    __gm__ uint8_t *tiling,
    TPipe *pipe)
{
    // Request-level payloads are consumed by the ordered conditional copy
    // launch. This kernel only reads missCounts for the batch path decision.
    (void)missSourceIds;
    (void)missDstSlots;
    using MtpType = SFAType<
        T, T, T, false, SFA_LAYOUT::TND, SFA_LAYOUT::PA_BSND,
        V_TEMPLATE, SFA_STAGE_NORMAL, SOURCE_AWARE, true>;
    CopySfaMtpAttentionMla<MtpType> attention;
    const auto *attentionTiling = reinterpret_cast<
        const CopySfaMtpAttentionTilingDataMla *>(fusedTiling);
    attention.Init(
        query, key, value, sparseIndices, cacheTokens, nullptr,
        actualSeqLengthsQuery, actualSeqLengthsKv, hbmBlockTable,
        queryRope, hbmKeyRope, attentionOut,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        attentionWorkspace, attentionTiling, tiling, pipe);
    if constexpr (SOURCE_AWARE) {
        attention.InitSourceAwareGather(
            dramKeyRope, dramKvCache, dramBlockTable, topkSourceIds,
            topkMissCounts, fusedTiling->copyCap,
            fusedTiling->dramMaxBlockNum);
    }
    attention.Process();
}

}  // namespace

extern "C" __global__ __aicore__ void fused_copy_sfa_mtp(
    __gm__ uint8_t *query,
    __gm__ uint8_t *key,
    __gm__ uint8_t *value,
    __gm__ uint8_t *sparseIndices,
    __gm__ uint8_t *cacheTokens,
    __gm__ uint8_t *hbmBlockTable,
    __gm__ uint8_t *actualSeqLengthsQuery,
    __gm__ uint8_t *actualSeqLengthsKv,
    __gm__ uint8_t *queryRope,
    __gm__ uint8_t *hbmKeyRope,
    __gm__ uint8_t *dramKeyRope,
    __gm__ uint8_t *dramKvCache,
    __gm__ uint8_t *dramBlockTable,
    __gm__ uint8_t *topkSourceIds,
    __gm__ uint8_t *topkMissCounts,
    __gm__ uint8_t *missSourceIds,
    __gm__ uint8_t *missDstSlots,
    __gm__ uint8_t *missCounts,
    __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(
        FusedCopySfaMtpTilingData,
        fusedTilingIn, tiling);
    const auto *fusedTiling = &fusedTilingIn;

    TPipe pipe;
    __gm__ uint8_t *attentionWorkspace = GetUserWorkspace(workspace);
    if (TILING_KEY_IS(1)) {
        const bool firstFill = IsBatchFirstFill(
            cacheTokens, missCounts, fusedTiling->baseParams.batchSize,
            fusedTiling->missCap);
        if constexpr (ORIG_DTYPE_QUERY == DT_FLOAT16) {
            if (firstFill) {
                RunFusedMtp<half, false>(
                    query, key, value, sparseIndices, cacheTokens,
                    hbmBlockTable, actualSeqLengthsQuery,
                    actualSeqLengthsKv, queryRope, hbmKeyRope,
                    dramKeyRope, dramKvCache, dramBlockTable,
                    topkSourceIds, topkMissCounts,
                    missSourceIds, missDstSlots, missCounts,
                    attentionOut, attentionWorkspace, fusedTiling,
                    tiling, &pipe);
            } else {
                RunFusedMtp<half, true>(
                    query, key, value, sparseIndices, cacheTokens,
                    hbmBlockTable, actualSeqLengthsQuery,
                    actualSeqLengthsKv, queryRope, hbmKeyRope,
                    dramKeyRope, dramKvCache, dramBlockTable,
                    topkSourceIds, topkMissCounts,
                    missSourceIds, missDstSlots, missCounts,
                    attentionOut, attentionWorkspace, fusedTiling,
                    tiling, &pipe);
            }
        } else {
            if (firstFill) {
                RunFusedMtp<bfloat16_t, false>(
                    query, key, value, sparseIndices, cacheTokens,
                    hbmBlockTable, actualSeqLengthsQuery,
                    actualSeqLengthsKv, queryRope, hbmKeyRope,
                    dramKeyRope, dramKvCache, dramBlockTable,
                    topkSourceIds, topkMissCounts,
                    missSourceIds, missDstSlots, missCounts,
                    attentionOut, attentionWorkspace, fusedTiling,
                    tiling, &pipe);
            } else {
                RunFusedMtp<bfloat16_t, true>(
                    query, key, value, sparseIndices, cacheTokens,
                    hbmBlockTable, actualSeqLengthsQuery,
                    actualSeqLengthsKv, queryRope, hbmKeyRope,
                    dramKeyRope, dramKvCache, dramBlockTable,
                    topkSourceIds, topkMissCounts,
                    missSourceIds, missDstSlots, missCounts,
                    attentionOut, attentionWorkspace, fusedTiling,
                    tiling, &pipe);
            }
        }
    }
}
