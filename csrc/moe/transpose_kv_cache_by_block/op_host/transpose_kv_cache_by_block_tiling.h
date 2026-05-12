#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TransposeKvCacheByBlockTilingData)
  // shape info
  // TILING_DATA_FIELD_DEF(uint32_t, blockNum);
  TILING_DATA_FIELD_DEF(uint32_t, blockSize);
  TILING_DATA_FIELD_DEF(uint32_t, headNum);
  TILING_DATA_FIELD_DEF(uint32_t, headDim);
  TILING_DATA_FIELD_DEF(uint32_t, splitNum);
  TILING_DATA_FIELD_DEF(uint32_t, layerNum);
  // tiling info
  TILING_DATA_FIELD_DEF(uint32_t, useCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, blockPerCore);
  TILING_DATA_FIELD_DEF(uint32_t, tailCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, calBlockNum);
  TILING_DATA_FIELD_DEF(uint32_t, blockSizePerTime);
  TILING_DATA_FIELD_DEF(uint32_t, blockSizePerTimeTail);
  TILING_DATA_FIELD_DEF(uint32_t, blockSizeSplitNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TransposeKvCacheByBlock, TransposeKvCacheByBlockTilingData)
}
