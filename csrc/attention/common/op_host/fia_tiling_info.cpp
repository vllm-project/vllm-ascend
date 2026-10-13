/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fia_tiling_info.cpp
 * \brief
 */

#include "fia_tiling_info.h"


namespace optiling {

std::string LayoutToSerialString(FiaLayout layout)
{
    const std::map<FiaLayout, std::string> layout2Str = {
        { FiaLayout::BSND, "BSND" },
        { FiaLayout::BNSD, "BNSD" },
    };

    if (layout2Str.find(layout) != layout2Str.end()) {
        return layout2Str.at(layout);
    }
    return "UNKNOWN";
}

std::string AxisToSerialString(FiaAxis axis)
{
    switch (axis) {
        case FiaAxis::B:
            return "B";
        case FiaAxis::S:
            return "S";
        case FiaAxis::N:
            return "N";
        case FiaAxis::D:
            return "D";
        default:
            return "UNKNOWN";
    }
}


std::string QuantModeToSerialString(FiaQuantMode fiaQuantMode)
{
    switch (fiaQuantMode) {
        case FiaQuantMode::NO_QUANT:
            return "NO_QUANT";
        case FiaQuantMode::ANTI_QUANT:
            return "ANTI_QUANT";
        case FiaQuantMode::FULL_QUANT:
            return "FULL_QUANT";
        default:
            return "UNKNOWN";
    }
}

std::string SituationToSerialString(RopeMode ropeMode)
{
    if (ropeMode == RopeMode::ROPE_SPLIT) {
        return "qkHeadDim = vHeadDim and rope exist";
    } else {
        return "rope not exist";
    }
}
} // namespace optiling