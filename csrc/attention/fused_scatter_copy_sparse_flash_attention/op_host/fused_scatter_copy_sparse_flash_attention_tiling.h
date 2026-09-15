/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#ifndef FUSED_SCATTER_COPY_SPARSE_FLASH_ATTENTION_TILING_H
#define FUSED_SCATTER_COPY_SPARSE_FLASH_ATTENTION_TILING_H

#include "fused_scatter_copy_sparse_flash_attention_tiling_impl.h"

namespace optiling {

// Keep the production sparse Attention payload as an exact prefix.  The
// kernel reinterprets that prefix as FusedScatterCopySparseFlashAttentionTilingDataMla
// and consumes the suffix for source-aware DRAM gather metadata.
BEGIN_TILING_DATA_DEF(FusedScatterCopySparseFlashAttentionTilingData)
TILING_DATA_FIELD_DEF_STRUCT(FusedScatterCopySparseFlashAttentionBaseParamsMla, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedScatterCopySparseFlashAttentionSplitKVParamsMla, splitKVParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedScatterCopySparseFlashAttentionSingleCoreParamsMla, singleCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedScatterCopySparseFlashAttentionSingleCoreTensorSizeMla, singleCoreTensorSize);
TILING_DATA_FIELD_DEF_STRUCT(FusedScatterCopySparseFlashAttentionInnerSplitParams, innerSplitParams);
TILING_DATA_FIELD_DEF(uint32_t, copyCap);
TILING_DATA_FIELD_DEF(uint32_t, missCap);
TILING_DATA_FIELD_DEF(uint32_t, hbmMaxBlockNum);
TILING_DATA_FIELD_DEF(uint32_t, dramMaxBlockNum);
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(
    FusedScatterCopySparseFlashAttention,
    FusedScatterCopySparseFlashAttentionTilingData)

struct FusedScatterCopySparseFlashAttentionCompileInfo {
};

}  // namespace optiling

#endif
