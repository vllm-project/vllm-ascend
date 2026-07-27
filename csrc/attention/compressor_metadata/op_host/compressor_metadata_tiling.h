#ifndef COMPRESSOR_METADATA_TILING_H
#define COMPRESSOR_METADATA_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CompressorMetadataTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, numRows);
    TILING_DATA_FIELD_DEF(uint32_t, numReqs);
    TILING_DATA_FIELD_DEF(uint32_t, actualNumReqs);
    TILING_DATA_FIELD_DEF(uint32_t, ropeRows);
    TILING_DATA_FIELD_DEF(uint32_t, ropeDim);
    TILING_DATA_FIELD_DEF(uint32_t, kvBlockTableStride);
    TILING_DATA_FIELD_DEF(uint32_t, kvBlockSize);
    TILING_DATA_FIELD_DEF(uint32_t, slotMappingFormat);
    TILING_DATA_FIELD_DEF(uint32_t, cmpRatio);
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, tileRows);
    TILING_DATA_FIELD_DEF(uint32_t, ropeRowBytes);
    TILING_DATA_FIELD_DEF(uint32_t, ropeRowBytesAligned);
    TILING_DATA_FIELD_DEF(uint32_t, slotCols);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CompressorMetadata, CompressorMetadataTilingData)

struct CompressorMetadataCompileInfo {
    uint32_t coreNum;
    uint64_t ubSizePlatForm;
};
}  // namespace optiling

#endif
