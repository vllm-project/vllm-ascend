/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef DGEMMA_APPLY_ROUTER_SCALE_TILING_DATA_H
#define DGEMMA_APPLY_ROUTER_SCALE_TILING_DATA_H
#include "kernel_tiling/kernel_tiling.h"
namespace DgemmaApplyRouterScale {
#pragma pack(push, 8)
struct alignas(8) DgemmaApplyRouterScaleTilingData {
    uint32_t numElems;
};
#pragma pack(pop)
}
#endif
