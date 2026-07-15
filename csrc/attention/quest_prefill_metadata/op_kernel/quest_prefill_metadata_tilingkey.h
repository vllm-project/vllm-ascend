/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "kernel_tiling/kernel_tiling.h"

#define QUEST_PREFILL_METADATA_TILING 0ULL

#define QUEST_PREFILL_METADATA_DTYPE_FP16 0U
#define QUEST_PREFILL_METADATA_DTYPE_BF16 1U

#define QUEST_PREFILL_METADATA_COPY_TILING_DATA(tilingDataStruct, tiling)                   \
    GET_TILING_DATA_WITH_STRUCT(tilingDataStruct, tiling_data_in, tiling);                  \
    const tilingDataStruct *__restrict tiling_data = &tiling_data_in;
