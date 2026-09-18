#ifndef CHUNK_KDA_FWD_FINALIZE_TILING_H
#define CHUNK_KDA_FWD_FINALIZE_TILING_H

#include <cstddef>
#include <cstdint>

#include "register/tilingdata_base.h"

namespace optiling {

// 字段顺序须与 op_kernel/chunk_kda_fwd_finalize_struct.h 保持一致。
BEGIN_TILING_DATA_DEF(ChunkKdaFwdFinalizeTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, seqNum);
TILING_DATA_FIELD_DEF(uint32_t, seqLen);
TILING_DATA_FIELD_DEF(uint32_t, valueHeadNum);
TILING_DATA_FIELD_DEF(uint32_t, totalChunks);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, headsPerPartition);
TILING_DATA_FIELD_DEF(bool, isVarLen);
TILING_DATA_FIELD_DEF(bool, outputSequenceMajor);
TILING_DATA_FIELD_DEF(bool, stateVFirst);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ChunkKdaFwdFinalize, ChunkKdaFwdFinalizeTilingData)

enum ChunkKdaFwdFinalizeInputIndex : size_t {
    FINALIZE_INPUT_QG_SCALED = 0,
    FINALIZE_INPUT_AQK,
    FINALIZE_INPUT_V_NEW,
    FINALIZE_INPUT_H,
    FINALIZE_INPUT_CU_SEQLENS,
    FINALIZE_INPUT_CHUNK_INDICES,
};

enum ChunkKdaFwdFinalizeAttrIndex : size_t {
    FINALIZE_ATTR_OUTPUT_LAYOUT = 0,
    FINALIZE_ATTR_STATE_V_FIRST,
};

struct ChunkKdaFwdFinalizeCompileInfo {};

} // namespace optiling

#endif // CHUNK_KDA_FWD_FINALIZE_TILING_H
