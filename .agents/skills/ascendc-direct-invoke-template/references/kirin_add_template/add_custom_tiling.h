/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */
#ifndef ADD_CUSTOM_TILING_H
#define ADD_CUSTOM_TILING_H

constexpr uint32_t TILE_LENGTH = 1024;
constexpr int32_t  DOUBLE_BUFFER = 2;   

struct AddTilingData {
    uint32_t totalLength;
    uint32_t blockNum;
    uint32_t numPerCore;
    uint32_t tailNumLastCore;
};

#endif // ADD_CUSTOM_TILING_H