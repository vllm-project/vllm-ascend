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
 * \file quant_batch_matmul_v3_checker.h
 * \brief
 */
#ifndef MC2_QUANT_BATCH_MATMUL_V3_CHECKER_H
#define MC2_QUANT_BATCH_MATMUL_V3_CHECKER_H
#include "op_host/tiling_base.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../quant_batch_matmul_v3_tiling.h"

namespace optiling {

class Mc2QuantBatchMatmulV3Checker {
public:
    Mc2QuantBatchMatmulV3Checker(gert::TilingContext *context, const Mc2QuantBatchMatmulInfo &inputParams)
        : context_(context), inputParams_(inputParams)
    {
    }

    ~Mc2QuantBatchMatmulV3Checker() = default;
    bool CheckDtype() const;
    bool CheckShape(const std::vector<gert::Shape *> &mandtoryShape, const gert::StorageShape *biasShape,
                    const gert::StorageShape *pertokenShape, const std::vector<int64_t> &DimValueOfMKN) const;

protected:
    bool CheckABDtypes() const;
    bool CheckScalesDtype() const;
    bool CheckScaleDtypeWithPertoken() const;
    bool CheckBiasDtype() const;
    bool CheckOutputDtype() const;
    bool CheckDtypesInRange() const;
    bool MxPertokenScaleShapeCheck(const gert::StorageShape *pertokenShape) const;
    bool MxScaleShapeCheck(const gert::Shape &scaleShape) const;
    bool CheckInputValidInPerblockMode(const gert::Shape &scaleShape, const gert::StorageShape *pertokenShape,
                                       const gert::Shape &x1Shape, const gert::Shape &x2Shape) const;
    bool CheckDimValidInPerblockMode(size_t x1ShapeLen, size_t x2ShapeLen, size_t pertokenShapeLen,
                                     size_t scaleShapeLen) const;
    bool CheckBatchValidInPerblockMode(const gert::Shape &scaleShape, const gert::Shape &pertoken,
                                       const gert::Shape &x1Shape, const gert::Shape &x2Shape) const;
    bool CheckInputValidInMxPerGroupMode(const gert::Shape &scaleShape, const gert::StorageShape *pertokenShape,
                                         const std::vector<int64_t> &dimValueOfMKN) const;
    bool CheckShapeValidInPerblockMode(const gert::Shape &scaleShape, const gert::Shape &pertoken,
                                       const gert::Shape &x1Shape, const gert::Shape &x2Shape) const;
    bool CheckGroupValidInPerblockMode() const;
    bool PerTokenDimValueCheck(const gert::Shape &scaleShape, const gert::StorageShape *pertokenShape) const;
    bool CheckShapeInRangeForOptionalInputs(const gert::Shape &scaleShape, const gert::StorageShape *biasShape,
                                            const gert::StorageShape *pertokenShape,
                                            const gert::StorageShape *offsetShape, size_t outDimNum) const;
    bool CheckDimValue(const gert::Shape &scaleShape, const gert::StorageShape *biasShape,
                       const gert::StorageShape *pertokenShape, const gert::StorageShape *offsetShape,
                       const std::vector<int64_t> &DimValueOfMKN) const;
    bool CheckShapeInBoundary(const gert::Shape &shape, uint32_t shapeIdx) const;
    bool BiasShapeCheck(const gert::Shape &biasShape, const gert::Shape &scaleShape,
                        const gert::StorageShape *pertokenShape) const;
    bool ExtraInputCheck() const;
    bool LogicXOR(bool cond1, bool cond2) const;

protected:
    gert::TilingContext *context_ = nullptr;
    Mc2QuantBatchMatmulInfo inputParams_;
};
} // namespace optiling
#endif // QUANT_BATCH_MATMUL_V3_CHECKER_H
