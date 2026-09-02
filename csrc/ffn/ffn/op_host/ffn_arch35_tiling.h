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
 * \file ffn_arch35_tiling.h
 * \brief FFN arch35 (ascend950) fused 路径的 tiling：布局识别 + 融合 tiling 计算。
 */

#ifndef OPS_TRANSFORMER_FFN_ARCH35_TILING_H_
#define OPS_TRANSFORMER_FFN_ARCH35_TILING_H_

#include <cstdint>
#include <cstddef>
#include <graph/types.h>
#include <exe_graph/runtime/tiling_context.h>
#include <exe_graph/runtime/storage_shape.h>
#include "ffn_tiling.h"
#include "platform/platform_info.h"

namespace optiling {

enum class ActiveType {
    FASTGELU = 0,
    RELU,
    SILU,
    GELU,
    GEGLU,
    SWIGLU,
    REGLU,
    INVALID_TYPE
};

constexpr size_t FFN_ATTR_INDEX_ACTIVATION = 0;
constexpr size_t FFN_ATTR_INDEX_INNER_PRECISE = 1;
constexpr size_t FFN_ATTR_INDEX_TOKENS_INDEX_FLAG = 3;

constexpr uint32_t BIAS1_INDEX = 4;
constexpr uint32_t BIAS2_INDEX = 5;

struct FFNCompileInfo {
    uint32_t blockDim;
    uint32_t coreNum;
    uint32_t aivCoreNum;
    uint32_t aicCoreNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0ASize;
    uint64_t l0BSize;
    uint64_t l0CSize;
    uint64_t sysWorkspaceSize;
    platform_ascendc::SocVersion socVersion;
    bool isRegbase = false;
};

// tiling 入口校验完成后传给 arch35 融合 tiling 的输入参数
struct FFNArch35TilingParams {
    uint32_t activeType = 0;
    uint32_t expertNum = 0;
    uint32_t bs = 0;
    uint32_t k1 = 0;
    uint32_t n1 = 0;
    uint32_t n2 = 0;
    bool isFfnTransB = false;
    ge::DataType xDataType = ge::DT_UNDEFINED;
    uint32_t xDataTypeSize = 0;
};

class FFNArch35Tiling {
public:
    // canonical/linear 布局识别
    // 返回 false 表示 w1 形状与 xK 完全不匹配。
    static bool DetectLayout(const gert::TilingContext *context,
                             const gert::StorageShape *weight1Shape,
                             const gert::StorageShape *weight2Shape,
                             int64_t xK, bool isSwiglu, bool &isLinear);

    static ge::graphStatus Tiling(gert::TilingContext *context,
                                  const FFNCompileInfo *compileInfoPtr,
                                  FFNTilingData &tilingData,
                                  const FFNArch35TilingParams &params);
};

} // namespace optiling

#endif // OPS_TRANSFORMER_FFN_ARCH35_TILING_H_
