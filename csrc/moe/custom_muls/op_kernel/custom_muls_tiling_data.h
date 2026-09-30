#ifndef CUSTOM_MULS_TILING_DATA_H
#define CUSTOM_MULS_TILING_DATA_H

#include <cstdint>

namespace optiling {
struct alignas(8) CustomMulsTilingData {
    int64_t totalLength = 0;
    int64_t blockFormer = 0;
    int64_t blockTail = 0;
    uint32_t blockNum = 0;
    uint32_t ubFormer = 0;
    uint32_t ubLoopFormer = 0;
    uint32_t ubTailFormer = 0;
    uint32_t ubLoopTail = 0;
    uint32_t ubTailTail = 0;
    float scalarValue = 0.0F;
};
} // namespace optiling

#endif // CUSTOM_MULS_TILING_DATA_H
