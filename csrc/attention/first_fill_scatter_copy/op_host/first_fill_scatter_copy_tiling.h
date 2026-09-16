#ifndef FIRST_FILL_SCATTER_COPY_TILING_H
#define FIRST_FILL_SCATTER_COPY_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(FirstFillScatterCopyTilingData)
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, totalPairSlots);
TILING_DATA_FIELD_DEF(int64_t, batchSize);
TILING_DATA_FIELD_DEF(int64_t, copyCap);
TILING_DATA_FIELD_DEF(int64_t, hbmMaxBlockNum);
TILING_DATA_FIELD_DEF(int64_t, dramMaxBlockNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    FirstFillScatterCopy,
    FirstFillScatterCopyTilingData)

struct FirstFillScatterCopyCompileInfo {
};

}  // namespace optiling

#endif
