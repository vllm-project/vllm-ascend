#ifndef GROUPED_MATMUL_SITU_QUANT_TILING_H
#define GROUPED_MATMUL_SITU_QUANT_TILING_H

#include <stdint.h>

namespace gmm_situ {

struct SituTilingHeader {
    uint32_t coreNum;
    uint32_t activeCount;
    uint32_t kSize;
    uint32_t nSize;
    uint32_t baseM;
    uint32_t mainBlockSize;
    uint32_t firstTailBlockSize;
    uint32_t reserved;
    uint64_t mainBlockCount;
    uint64_t firstTailBlockCount;
    float beta;
    float invBeta;
    float linearBeta;
    float invLinearBeta;
};

static_assert(sizeof(SituTilingHeader) == 64, "GMM-SiTU tiling header must remain 64 bytes");

} // namespace gmm_situ

#endif // GROUPED_MATMUL_SITU_QUANT_TILING_H
