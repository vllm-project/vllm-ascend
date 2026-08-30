/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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
        { FiaLayout::BSH, "BSH" },
        { FiaLayout::BSND, "BSND" },
        { FiaLayout::BNSD, "BNSD" },
        { FiaLayout::NZ, "NZ" },
        { FiaLayout::TND, "TND" },
        { FiaLayout::NBSD, "NBSD" },
        { FiaLayout::NTD, "NTD" },
        { FiaLayout::S1S2, "S1S2" },
        { FiaLayout::BS2, "BS2" },
        { FiaLayout::BnBsH, "BnBsH" },
        { FiaLayout::BnNBsD, "BnNBsD" },
        { FiaLayout::BnBsND, "BnBsND" },
        { FiaLayout::BNS1S2, "BNS1S2" },
        { FiaLayout::INS1S2, "1NS1S2" },
        { FiaLayout::BNS11, "BNS11" },
        { FiaLayout::TN1, "TN1" },
        { FiaLayout::BS1S2, "BS1S2" },
        { FiaLayout::B1S1S2, "B1S1S2" },
        { FiaLayout::IS1S2, "1S1S2" },
        { FiaLayout::I1S1S2, "11S1S2" }
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
        case FiaAxis::H:
            return "H";
        case FiaAxis::T:
            return "T";
        case FiaAxis::D1:
            return "D1";
        case FiaAxis::D0:
            return "D0";
        case FiaAxis::S1:
            return "S1";
        case FiaAxis::S2:
            return "S2";
        case FiaAxis::Bn:
            return "Bn";
        case FiaAxis::Bs:
            return "Bs";
        case FiaAxis::CONST:
            return "CONST";
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