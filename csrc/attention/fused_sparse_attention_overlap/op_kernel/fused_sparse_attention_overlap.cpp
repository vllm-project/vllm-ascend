#include "kernel_operator.h"
#include "fused_sparse_attention_overlap_template_tiling_key.h"
#include "fused_sparse_attention_overlap_kernel_mla.h"

using namespace AscendC;

#define FSA_OVERLAP_SELECTION_UPDATE_OP_IMPL(templateClass, tilingdataClass, ...)                 \
    do {                                                                                          \
        templateClass<FusedSparseAttentionOverlapType<__VA_ARGS__>> op;                                                   \
        GET_TILING_DATA_WITH_STRUCT(tilingdataClass, tiling_data_in, tiling);                      \
        const tilingdataClass *__restrict tiling_data = &tiling_data_in;                           \
        op.Init(query, key, value, sparseIndices, actualSeqLengthsQuery, actualSeqLengthsKV,       \
            blocktable, queryRope, keyRope, attentionOut, user, tiling_data, tiling, &tPipe);       \
        op.InitSelectionUpdateGlobalTensor(selectionKRope, selectionKvCache,                       \
            selectionKvBlockTable, selectionKvBlockStatus, selectionMembershipMap,                 \
            selectionKvActualSeq, true);                                                           \
        op.Process();                                                                              \
    } while (0)

template<int FLASH_DECODE, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE>
__global__ __aicore__ void fused_sparse_attention_overlap(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *sparseIndices, __gm__ uint8_t *blocktable,
    __gm__ uint8_t *actualSeqLengthsQuery, __gm__ uint8_t *actualSeqLengthsKV,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *selectionKRope, __gm__ uint8_t *selectionKvCache,
    __gm__ uint8_t *selectionKvBlockTable, __gm__ uint8_t *selectionKvBlockStatus,
    __gm__ uint8_t *selectionMembershipMap,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *selectionKvActualSeq,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

    if constexpr (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_FLOAT16 &&
                  ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) {
        FSA_OVERLAP_SELECTION_UPDATE_OP_IMPL(FusedSparseAttentionOverlapMla, FusedSparseAttentionOverlapTilingDataMla,
            half, half, half, FLASH_DECODE, static_cast<FusedSparseAttentionOverlapLayout>(LAYOUT_T),
            static_cast<FusedSparseAttentionOverlapLayout>(KV_LAYOUT_T), TEMPLATE_MODE);
    } else {
        FSA_OVERLAP_SELECTION_UPDATE_OP_IMPL(FusedSparseAttentionOverlapMla, FusedSparseAttentionOverlapTilingDataMla,
            bfloat16_t, bfloat16_t, bfloat16_t, FLASH_DECODE, static_cast<FusedSparseAttentionOverlapLayout>(LAYOUT_T),
            static_cast<FusedSparseAttentionOverlapLayout>(KV_LAYOUT_T), TEMPLATE_MODE);
    }
}
