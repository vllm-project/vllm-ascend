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
 * \file weight_quant_batch_matmul_v2_checker_for_mmads8s4.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_CHECKER_FOR_MMADS8S4_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_CHECKER_FOR_MMADS8S4_H
#include "weight_quant_batch_matmul_v2_tiling.h"

namespace optiling {

class Mc2WeightQuantBatchMatmulV2Checker4MmadS8S4 {
public:
    explicit Mc2WeightQuantBatchMatmulV2Checker4MmadS8S4(gert::TilingContext *context) : context_(context)
    {
    }

    ~Mc2WeightQuantBatchMatmulV2Checker4MmadS8S4() = default;

    ge::graphStatus Check();

private:
    ge::graphStatus CheckContext();
    bool CheckDtype();
    bool CheckAttr();
    bool CheckShape();
    bool CheckInputShape(const gert::StorageShape *xShape, const gert::StorageShape *weightShape);
    bool CheckAntiQuantShape(const gert::StorageShape *antiQuantScaleShape,
                             const gert::StorageShape *antiQuantOffsetShape);

    gert::TilingContext *context_ = nullptr;
    Mc2WeightQuantBatchMatmulInfo inputParams_;
};
} // namespace optiling
#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_CHECKER_FOR_MMADS8S4_H
