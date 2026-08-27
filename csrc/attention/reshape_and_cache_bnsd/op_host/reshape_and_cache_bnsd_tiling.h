
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ReshapeAndCacheBNSDTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, numTokens);
    TILING_DATA_FIELD_DEF(uint32_t, headDim);
    TILING_DATA_FIELD_DEF(uint32_t, numBlocks);
    TILING_DATA_FIELD_DEF(uint32_t, numHeads);
    TILING_DATA_FIELD_DEF(uint32_t, blockSize);
    TILING_DATA_FIELD_DEF(uint32_t, batchSeqLen);
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, numCore);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ReshapeAndCacheBnsd, ReshapeAndCacheBNSDTilingData)
}

struct reshapeAndCacheBnsdCompileInfo {};


