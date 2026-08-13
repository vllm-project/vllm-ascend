/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#ifndef CATEGORICAL_SAMPLE_TILING_H
#define CATEGORICAL_SAMPLE_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CategoricalSampleTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, numRows);
    TILING_DATA_FIELD_DEF(uint32_t, vocabSize);
    TILING_DATA_FIELD_DEF(uint32_t, numRequests);
    TILING_DATA_FIELD_DEF(uint32_t, rowStride);
    TILING_DATA_FIELD_DEF(uint32_t, outputProcessedLogitsStride);
    TILING_DATA_FIELD_DEF(uint32_t, outputProcessedLogitsNumCols);
    TILING_DATA_FIELD_DEF(uint32_t, tileElements);
    TILING_DATA_FIELD_DEF(uint32_t, tileCount);
    TILING_DATA_FIELD_DEF(uint32_t, hasOutputProcessedLogits);
    TILING_DATA_FIELD_DEF(uint32_t, hasOutputProcessedLogitsCol);
    TILING_DATA_FIELD_DEF(uint32_t, outputProcessedLogitsColPerToken);
    TILING_DATA_FIELD_DEF(uint32_t, applyTemperature);
    TILING_DATA_FIELD_DEF(uint32_t, returnLse);
    TILING_DATA_FIELD_DEF(uint32_t, useFp64);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CategoricalSample, CategoricalSampleTilingData)

struct CategoricalSampleCompileInfo {};
}  // namespace optiling

#endif
