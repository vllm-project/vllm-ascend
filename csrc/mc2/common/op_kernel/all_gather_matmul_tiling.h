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
 * \file all_gather_matmul_tiling.h
 * \brief
 */

#ifndef __ALL_GATHER_MATMUL_TILING_H__
#define __ALL_GATHER_MATMUL_TILING_H__

#pragma once
#include "kernel_tiling/kernel_tiling.h"
#include "mc2_tiling_struct.h"

struct AllGatherSoc {
    uint32_t commAlg;
    uint32_t isA3;
    uint32_t isStep;
    uint32_t isND2NZ;
};

class AllGatherMatmulTilingData {
public:
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    TCubeTiling tileTiling;
    TCubeTiling tailTiling;
    TCubeTiling localTiling;
    Mc2Tiling::TileL2Tiling tileL2Tiling;
    Mc2Tiling::TileL2Tiling tailL2Tiling;
    Mc2Tiling::TileL2Tiling localL2Tiling;
    Mc2Tiling::RCSTiling param;
    AllGatherSoc socParam;
};

#endif //__ALL_GATHER_MATMUL_TILING_H__
