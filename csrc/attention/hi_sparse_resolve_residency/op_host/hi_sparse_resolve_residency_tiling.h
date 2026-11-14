// SPDX-License-Identifier: Apache-2.0
#ifndef HI_SPARSE_RESOLVE_RESIDENCY_TILING_H
#define HI_SPARSE_RESOLVE_RESIDENCY_TILING_H

#include <cstdint>

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(HiSparseResolveResidencyTilingData)
TILING_DATA_FIELD_DEF(uint64_t, rows);
TILING_DATA_FIELD_DEF(uint64_t, topk);
TILING_DATA_FIELD_DEF(uint64_t, hotSlots);
TILING_DATA_FIELD_DEF(uint64_t, hostRows);
TILING_DATA_FIELD_DEF(uint64_t, logicalStride);
TILING_DATA_FIELD_DEF(uint64_t, stateIndexCount);
TILING_DATA_FIELD_DEF(uint64_t, tagRows);
TILING_DATA_FIELD_DEF(uint64_t, sourceRows);
TILING_DATA_FIELD_DEF(uint64_t, sourceColumns);
TILING_DATA_FIELD_DEF(uint64_t, sourceStride);
TILING_DATA_FIELD_DEF(uint64_t, sourceBlockSize);
TILING_DATA_FIELD_DEF(uint64_t, residentRows);
TILING_DATA_FIELD_DEF(uint64_t, residentColumns);
TILING_DATA_FIELD_DEF(uint64_t, residentStride);
TILING_DATA_FIELD_DEF(uint64_t, residentBlockSize);
TILING_DATA_FIELD_DEF(uint64_t, residentNullBlock);
TILING_DATA_FIELD_DEF(uint64_t, hotBlockSize);
TILING_DATA_FIELD_DEF(uint64_t, hotTableStride);
TILING_DATA_FIELD_DEF(uint64_t, attentionStride);
TILING_DATA_FIELD_DEF(uint64_t, attentionBlockStride);
TILING_DATA_FIELD_DEF(uint64_t, countsStride);
TILING_DATA_FIELD_DEF(uint64_t, maxUnionRows);
TILING_DATA_FIELD_DEF(uint64_t, hashSize);
TILING_DATA_FIELD_DEF(uint64_t, attentionCapacity);
TILING_DATA_FIELD_DEF(uint64_t, byteMissMask);
TILING_DATA_FIELD_DEF(uint32_t, ubBytes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(HiSparseResolveResidency, HiSparseResolveResidencyTilingData)

struct HiSparseResolveResidencyCompileInfo {};
}  // namespace optiling

#endif  // HI_SPARSE_RESOLVE_RESIDENCY_TILING_H
