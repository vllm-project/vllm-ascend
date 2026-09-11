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
 * \file situ_mx_quant_tiling_data.h
 * \brief Tiling data structure for Situ + MX quantization
 */

#ifndef SITU_MX_QUANT_TILING_DATA_H
#define SITU_MX_QUANT_TILING_DATA_H

struct SituMxQuantTilingData {
    // Basic parameters
    int64_t usedCoreNum;

    // 3D data shape: [inputDim0, inputDim1, inputDim2]
    //   inputDim0 = batch dim (=1 for 2D, product of leading dims for >2D)
    //   inputDim1 = row dim (M)
    //   inputDim2 = Situ output dim = last_dim / 2 (N)
    int64_t inputDim0;
    int64_t inputDim1;
    int64_t inputDim2;

    // Block distribution
    int64_t dimNBlockNum;       // CeilDiv(N, 256)

    // Memory allocation parameters
    int64_t maxBasicNumUbDim2;  // UB 内最大列方向 block 数
    int64_t maxBasicNumUbDim1;  // UB 内最大行数

    // Core grid distribution
    int64_t nCoreNum;           // cores in N direction
    int64_t mCorePerB;          // M-cores per batch

    // Inter-core split parameters
    int64_t frontCoreNum;
    int64_t tailCoreBasicNumDim1;

    // Attributes
    int64_t activateLeft;
    float beta;
    float linearBeta;
    int64_t hasLinearBeta;      // 0 or 1
    int64_t hasGroupList;       // 0 or 1
    int64_t groupListSize;      // per-expert count tensor length
};
#endif // SITU_MX_QUANT_TILING_DATA_H
