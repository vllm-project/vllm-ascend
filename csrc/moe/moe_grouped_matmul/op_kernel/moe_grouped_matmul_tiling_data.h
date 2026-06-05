#ifndef MOE_GROUPED_MATMUL_TILING_DATA_H
#define MOE_GROUPED_MATMUL_TILING_DATA_H

#include <cstdint>

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(push, 8)
struct alignas(8) MoeGroupedMatmulTilingData {
    uint32_t group_num;
    uint32_t core_num;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t single_m;
    uint32_t single_n;
    TCubeTiling mm_tiling;
};
#pragma pack(pop)

#endif
