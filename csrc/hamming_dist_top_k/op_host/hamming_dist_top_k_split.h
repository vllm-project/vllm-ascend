#ifndef HAMMING_DIST_TOP_K_SPLIT_H
#define HAMMING_DIST_TOP_K_SPLIT_H

#include "hamming_dist_top_k.h"
#include "hamming_dist_top_k_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
class HammingDistTopKSplitSTiling : public HammingDistTopKTiling {
public:
    HammingDistTopKSplitSTiling(gert::TilingContext *context) : HammingDistTopKTiling(context) {}

    bool IsCapable();

    ge::graphStatus DoOpTiling();

    uint64_t GetTilingKey();

    void SetMatmulTiling();
    void SetMatmulTilingRope();

    void SetTopKTiling();
    
    void PrintTilingData();
    void PrintTilingDataRope();

    ge::graphStatus GetWorkspaceSize();

    float CORE_USE_RATIO = 0.8f;
    uint64_t WORKSIZE = 16 * 1024 * 1024;
    uint32_t COMPRESSED_RATE = 8;
    uint32_t TILE_N1 = 128;
    uint32_t TILE_N3 = 7 * 1024;
    uint32_t TOP_K_INNER_SIZE = 4 * 1024;
    uint32_t S_BLOCK_SIZE = 256;
    uint32_t L0B_BASE_SIZE = 512;
    uint32_t VECTOR_CUBE_RATIO = 2;
    uint64_t WORKSPACE_SCALE = 2;
    uint64_t MAX_BATCH = 16;
    uint64_t SUPER_LONG_SEQLEN = 26 * 1024;
    uint64_t MIN_SPLIT_S_SEQLEN = 8 * 1024;

    uint32_t KEY_ROPE_INPUT_INDEX = 7;
    uint32_t KEY_BLOCK_TABLE_INPUT_INDEX = 5;
};
}
#endif