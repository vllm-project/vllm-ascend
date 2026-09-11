#ifndef CUSTOM_MULS_TILING_H
#define CUSTOM_MULS_TILING_H

#include <cstdint>

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CustomMulsTilingData)
    TILING_DATA_FIELD_DEF(int64_t, totalLength);
    TILING_DATA_FIELD_DEF(int64_t, blockFormer);
    TILING_DATA_FIELD_DEF(int64_t, blockTail);
    TILING_DATA_FIELD_DEF(uint32_t, blockNum);
    TILING_DATA_FIELD_DEF(uint32_t, ubFormer);
    TILING_DATA_FIELD_DEF(uint32_t, ubLoopFormer);
    TILING_DATA_FIELD_DEF(uint32_t, ubTailFormer);
    TILING_DATA_FIELD_DEF(uint32_t, ubLoopTail);
    TILING_DATA_FIELD_DEF(uint32_t, ubTailTail);
    TILING_DATA_FIELD_DEF(float, scalarValue);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(customMuls, CustomMulsTilingData)
} // namespace optiling

#endif // CUSTOM_MULS_TILING_H
