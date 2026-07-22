#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>

namespace optiling {

BEGIN_TILING_DATA_DEF(ChunkGatedDeltaRuleComputeWyTilingData)
TILING_DATA_FIELD_DEF(int64_t, batch);
TILING_DATA_FIELD_DEF(int64_t, seqlen);
TILING_DATA_FIELD_DEF(int64_t, kNumHead);
TILING_DATA_FIELD_DEF(int64_t, vNumHead);
TILING_DATA_FIELD_DEF(int64_t, kHeadDim);
TILING_DATA_FIELD_DEF(int64_t, vHeadDim);
TILING_DATA_FIELD_DEF(int64_t, chunkSize);
TILING_DATA_FIELD_DEF(int64_t, numChunks);
TILING_DATA_FIELD_DEF(int64_t, groupSize);
TILING_DATA_FIELD_DEF(int64_t, totalTasks);
TILING_DATA_FIELD_DEF(uint32_t, localWorkspaceSize);
TILING_DATA_FIELD_DEF(uint32_t, perCoreWorkspaceBytes);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, reserved0);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmAttn);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmSquare);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmApplyU);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmApplyW);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ChunkGatedDeltaRuleComputeWy, ChunkGatedDeltaRuleComputeWyTilingData)

struct ChunkGatedDeltaRuleComputeWyCompileInfo {};
} // namespace optiling
