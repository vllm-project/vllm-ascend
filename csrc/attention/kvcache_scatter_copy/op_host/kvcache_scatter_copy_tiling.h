/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */

#ifndef KVCACHE_SCATTER_COPY_TILING_H_
#define KVCACHE_SCATTER_COPY_TILING_H_

#include "error/ops_error.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(KvcacheScatterCopyTilingData)
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, totalPairSlots);
TILING_DATA_FIELD_DEF(int64_t, batchSize);
TILING_DATA_FIELD_DEF(int64_t, copyCap);
TILING_DATA_FIELD_DEF(int64_t, hbmMaxBlockNum);
TILING_DATA_FIELD_DEF(int64_t, dramMaxBlockNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(KvcacheScatterCopy, KvcacheScatterCopyTilingData)

struct KvcacheScatterCopyCompileInfo {
};

class KvcacheScatterCopyTiling {
public:
    explicit KvcacheScatterCopyTiling(gert::TilingContext* context) : context_(context) {}
    ~KvcacheScatterCopyTiling() {}
    ge::graphStatus RunTiling();

private:
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetShapeInfo();
    ge::graphStatus GetDtypeInfo();
    ge::graphStatus DoOpTiling();
    ge::graphStatus PostTiling();

private:
    KvcacheScatterCopyTilingData tilingData_;
    gert::TilingContext* context_ = nullptr;
    int64_t coreNum_ = 0;
    int64_t batchSize_ = 0;
};

} // namespace optiling

#endif // KVCACHE_SCATTER_COPY_TILING_H_
