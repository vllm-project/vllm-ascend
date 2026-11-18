/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef QSA_UNION_INPUTS_TILING_DATA_H
#define QSA_UNION_INPUTS_TILING_DATA_H

#include <cstdint>

// Device-side mirror of optiling::QsaUnionInputsTilingData. Keep the field
// order and widths identical to the host tiling definition.
struct QsaUnionInputsTilingData {
    uint32_t rows{0};
    uint32_t groups{0};
    uint32_t input_groups{0};
    uint32_t topk{0};
    uint32_t width{0};
    uint32_t compact_row_stride{0};
    uint32_t sort_tmp_bytes{0};
    uint32_t groups_per_core{0};
};

#endif  // QSA_UNION_INPUTS_TILING_DATA_H
