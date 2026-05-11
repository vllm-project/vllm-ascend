#ifndef HAMMING_DIST_TOP_K_H
#define HAMMING_DIST_TOP_K_H


#include "hamming_dist_top_k_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
class HammingDistTopKTiling {
public:
    // from parent class
    gert::TilingContext *context_ = nullptr;
    std::unique_ptr<platform_ascendc::PlatformAscendC> ascendcPlatform_{nullptr};
    uint32_t blockDim_{0};
    uint64_t workspaceSize_{0};
    uint64_t tilingKey_{0};
    AiCoreParams aicoreParams_{0};

    // from child class
    HammingDistTopKMatmulInfo inputParams_;
    uint32_t libApiWorkSpaceSize_ = 0;
    uint32_t coreNum_ = 1;
    const char *opName_ = "";
    int32_t dtypeByte_ = 2; /* 2: size of float16 */
    HammingDistTopKTilingData tilingData_;
    bool compileInfoInit_ = false;
    bool continFlag_ = false;
    uint32_t seqLen_ = 1;

    HammingDistTopKTiling(gert::TilingContext *context) : context_(context) {
        InitAttrParam();
        uint32_t dimNum = GetOutShape(0).GetDimNum();
        maxK = GetOutShape(0).GetDim(dimNum - 1);
    }

    bool IsCapable();
    // 1. Obtain platform information such as CoreNum, UB/L1/L0C resource size
    ge::graphStatus GetPlatformInfo();
    // 2. Obtain INPUT/OUTPUT/ATTR information
    ge::graphStatus GetShapeAttrsInfo();
    // 3. Calculate data split TilingData
    ge::graphStatus DoOpTiling();
    // 4. Calculate TilingData for high-level API
    ge::graphStatus DoLibApiTiling();
    // 5. Calculate TilingKey
    uint64_t GetTilingKey();
    // 6. Calculate Workspace size
    ge::graphStatus GetWorkspaceSize();
    // 7. Save Tiling data
    ge::graphStatus PostTiling();

    void Reset();
    void SetMatmulTiling();
    void SetMatmulTilingRope();
    void SetTopKTiling();
    bool SetPlatformInfoForTiling();
    const gert::Shape GetShape(const size_t index);
    // Get input data
    const uint32_t GetInputAttrData(const size_t index);
    // output shape
    const gert::Shape GetOutShape(const size_t index);

    // Initialize sink and recent
    const void InitAttrParam() {
        uint32_t sink = GetInputAttrData(1);
        uint32_t recent = GetInputAttrData(2);
        uint32_t supportOffload = GetInputAttrData(3);
        tilingData_.params.set_sink(sink);
        tilingData_.params.set_recent(recent);
        tilingData_.params.set_supportOffload(supportOffload);
    }

    uint32_t maxK = 512;  // default maxK
    uint32_t TILE_N1 = 254;
    uint32_t TILE_N2 = 3328;
    uint32_t DIMENSION = 128;
    uint32_t SEQ_LEN_THRES = 512;
    uint64_t SUB_BLOCK_NUM_WITH_DB = 4;
    uint64_t WORKSIZE = 16 * 1024 * 1024;
    uint32_t TOP_K_ALIGN_NUM = 32;
    uint32_t KEY_ROPE_INPUT_INDEX = 7;
    uint32_t KEY_BLOCK_TABLE_INPUT_INDEX = 5;
    uint32_t COMPRESSED_RATE = 8;
};
}

#endif