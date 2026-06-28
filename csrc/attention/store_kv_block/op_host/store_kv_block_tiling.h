#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(StoreKVBlockTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, blockTableSize);
    TILING_DATA_FIELD_DEF(uint32_t, typeByte);
    TILING_DATA_FIELD_DEF(uint32_t, tokenSize);
    TILING_DATA_FIELD_DEF(uint32_t, corePerNum);
    TILING_DATA_FIELD_DEF(uint32_t, coreTail);
    TILING_DATA_FIELD_DEF(uint32_t, numTokens);
    TILING_DATA_FIELD_DEF(uint32_t, numCache);
    TILING_DATA_FIELD_DEF(uint32_t, groupInfoLen);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(StoreKVBlock, StoreKVBlockTilingData)

struct StoreKVBlockCompileInfo {
    uint32_t coreNum;
    uint64_t ubSizePlatForm;
    uint32_t sysWorkspaceSize;
};

} // namespace optiling
