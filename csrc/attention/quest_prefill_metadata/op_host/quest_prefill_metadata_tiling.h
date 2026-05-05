/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUEST_PREFILL_METADATA_TILING_H_
#define QUEST_PREFILL_METADATA_TILING_H_

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(QuestPrefillMetadataTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize)
TILING_DATA_FIELD_DEF(uint32_t, numKvHeads)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, headDim)
TILING_DATA_FIELD_DEF(uint32_t, maxKvBlocksPerRequest)
TILING_DATA_FIELD_DEF(uint32_t, maxMetadataBlocksPerRequest)
TILING_DATA_FIELD_DEF(uint32_t, dataType)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(QuestPrefillMetadata, QuestPrefillMetadataTilingData)
} // namespace optiling

#endif // QUEST_PREFILL_METADATA_TILING_H_
