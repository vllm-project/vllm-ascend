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
 * \file gm_coord.h
 * \brief
 */
#ifndef GM_COORD_H
#define GM_COORD_H

enum class InnerMLayout {
    GS1_MERGE_LAYOUT = 0,
    S1_ONLY_LAYOUT = 1,
};

template <InnerMLayout M_LAYOUT>
struct GmCoordT;

template <>
struct GmCoordT<InnerMLayout::GS1_MERGE_LAYOUT> {
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t gS1Idx;
    uint32_t dIdx;
    uint32_t gS1DealSize;
    uint32_t dDealSize;
};

template <>
struct GmCoordT<InnerMLayout::S1_ONLY_LAYOUT> {
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t gIdx;
    uint32_t s1Idx;
    uint32_t dIdx;
    uint32_t s1DealSize;
    uint32_t dDealSize;
};

using GmCoordGs1Merge = GmCoordT<InnerMLayout::GS1_MERGE_LAYOUT>;
using GmCoordS1Only = GmCoordT<InnerMLayout::S1_ONLY_LAYOUT>;
#endif
