/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include "kernel_operator.h"
#include "fused_quant_lightning_indexer_manage_template_tiling_key.h"

#if (__CCE_AICORE__ == 310)
#include "arch35/fused_quant_lightning_indexer_manage_impl.h"
#elif (defined __DAV_310R6__) || (__CCE_AICORE__ == 200)
// These architectures do not implement this operator.
#else
#include "arch22/fused_quant_lightning_indexer_manage_impl.h"
#endif

template <int DT>
__global__ __aicore__ void fused_quant_lightning_indexer_manage(
    __gm__ uint8_t *weights, __gm__ uint8_t *queryDequantScale,
    __gm__ uint8_t *query, __gm__ uint8_t *keyDequantScale,
    __gm__ uint8_t *key, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *actualSeqLengthsQuery,
    __gm__ uint8_t *actualSeqLengthsKey,
    __gm__ uint8_t *offloadSeqLengthsKey,
    __gm__ uint8_t *cacheTokens, __gm__ uint8_t *requestState,
    __gm__ uint8_t *reqPoolEntries, __gm__ uint8_t *cacheSlots,
    __gm__ uint8_t *topkSourceIds, __gm__ uint8_t *topkSlots,
    __gm__ uint8_t *topkMissCounts, __gm__ uint8_t *missSrcIds,
    __gm__ uint8_t *missDstSlots, __gm__ uint8_t *missCount,
    __gm__ uint8_t *cacheSlotsOut, __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
#if (__CCE_AICORE__ == 310)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    RunFusedQuantLightningIndexerManageArch35(
        weights, queryDequantScale, query, keyDequantScale, key, blockTable,
        actualSeqLengthsQuery, actualSeqLengthsKey, offloadSeqLengthsKey,
        cacheTokens, requestState, reqPoolEntries, cacheSlots, topkSourceIds,
        topkSlots, topkMissCounts, missSrcIds, missDstSlots, missCount,
        cacheSlotsOut, workspace, tiling);
#elif (defined __DAV_310R6__) || (__CCE_AICORE__ == 200)
#else
    TPipe tPipe;
    (void)actualSeqLengthsKey;
    (void)requestState;
    (void)cacheSlotsOut;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    // C8 唯一实例化: query/key int8 + bf16 index weights (W_T), fp16 dequant scales
    INVOKE_LI_MTP_C8_TOPK(int8_t, int8_t, bfloat16_t);
#endif
}
