/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file comm_tiling_data.h
 * \brief Apace 统一的 CommTilingData 结构体定义
 *
 * 该结构体同时被 host 端切分算法和 kernel 端通信实现使用，
 * 提供简化的 5 字段格式，足以支撑集合通信操作。
 */

#pragma once

#ifndef __CCE_AICORE__
#include <cstdint>
#endif


/*!
 * \brief 统一的 CommTilingData 结构体（kernel 端简化格式）
 *
 * 使用元素个数作为单位，kernel 端内部转换为字节大小。
 *
 * 切分策略：
 * - 头块：splitAxisTileCnt 个大小为 splitAxisTileSize 的 tile
 * - 尾块：splitAxisTailCnt 个大小为 splitAxisTailSize 的 tile
 *
 * 使用示例：
 * - AllToAll：切分轴 = M 轴，非切分轴 = K 轴
 *   CommTilingData = {tileSize_M, tileCnt_M, tailSize_M, tailCnt_M, K}
 * - AllGather：切分轴 = M 轴，非切分轴 = K 轴
 *   CommTilingData = {tileSize_M, tileCnt_M, tailSize_M, tailCnt_M, K}
 *
 * 内存布局：
 * - 切分轴总元素数 = tileSize * tileCnt + tailSize * tailCnt
 * - 每个 tile 有 (tileSize * nonSplitAxisSize) 个元素
 * - 字节大小 = 元素个数 * sizeof(dtype)
 */
struct CommTilingData {
    uint64_t splitAxisTileSize{0};     // 切分轴头块大小（元素个数）
    uint64_t splitAxisTileCnt{0};      // 切分轴头块数量
    uint64_t splitAxisTailSize{0};     // 切分轴尾块大小（元素个数）
    uint64_t splitAxisTailCnt{0};      // 切分轴尾块数量
    uint64_t nonSplitAxisSize{0};      // 非切分轴数据块大小（元素个数，所有内轴乘积）
};