#ifndef FUSED_COPY_SFA_MTP_TILING_H
#define FUSED_COPY_SFA_MTP_TILING_H

#include "sfa_mtp_base_tiling.h"

namespace optiling {

// Keep the production sparse Attention payload as an exact prefix.  The
// kernel reinterprets that prefix as CopySfaMtpAttentionTilingDataMla
// and consumes the suffix for source-aware DRAM gather metadata.
BEGIN_TILING_DATA_DEF(FusedCopySfaMtpTilingData)
TILING_DATA_FIELD_DEF_STRUCT(CopySfaMtpAttentionBaseParamsMla, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(CopySfaMtpAttentionSplitKVParamsMla, splitKVParams);
TILING_DATA_FIELD_DEF_STRUCT(CopySfaMtpAttentionSingleCoreParamsMla, singleCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(CopySfaMtpAttentionSingleCoreTensorSizeMla, singleCoreTensorSize);
TILING_DATA_FIELD_DEF_STRUCT(CopySfaMtpAttentionInnerSplitParams, innerSplitParams);
TILING_DATA_FIELD_DEF(uint32_t, copyCap);
TILING_DATA_FIELD_DEF(uint32_t, missCap);
TILING_DATA_FIELD_DEF(uint32_t, hbmMaxBlockNum);
TILING_DATA_FIELD_DEF(uint32_t, dramMaxBlockNum);
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(
    FusedCopySfaMtp,
    FusedCopySfaMtpTilingData)

struct FusedCopySfaMtpCompileInfo {
};

}  // namespace optiling

#endif
