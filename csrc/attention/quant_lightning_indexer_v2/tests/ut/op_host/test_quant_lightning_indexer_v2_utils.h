/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_QUANT_LIGHTNING_INDEXER_V2_UTILS_H
#define TEST_QUANT_LIGHTNING_INDEXER_V2_UTILS_H

#include <cstdint>
#include <string>
#include <vector>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

namespace qliv2_ut {

constexpr uint64_t SKIP_TILING_KEY = UINT64_MAX;

// Default case matches a valid Ascend910B PA_BBND int8 success case.
struct CaseParam {
    std::string soc = "Ascend910B";
    uint64_t coreNum = 64;
    uint64_t ubSize = 262144;
    uint64_t l2Size = 16384;
    std::vector<int64_t> qShape = {2, 39, 64, 128};
    std::vector<int64_t> kShape = {2, 16, 1, 128}; // PA_BBND: block_num, block_size, N2, D
    std::vector<int64_t> wShape = {2, 39, 64};
    std::vector<int64_t> qScaleShape = {2, 39, 64};
    std::vector<int64_t> kScaleShape = {2, 16, 1};
    std::vector<int64_t> outShape = {2, 39, 1, 2048};
    std::vector<int64_t> valuesShape = {0};
    std::vector<int64_t> cuSeqQ; // empty means not provided
    std::vector<int64_t> cuSeqK;
    std::vector<int64_t> sequsedQ;
    std::vector<int64_t> sequsedK = {2};
    std::vector<int64_t> cmpResidual;
    std::vector<int64_t> blockTable = {2, 2};
    std::vector<int64_t> idxOffset;
    std::vector<int64_t> metadata = {1024};
    ge::DataType qType = ge::DT_INT8;
    ge::DataType kType = ge::DT_INT8;
    ge::DataType wType = ge::DT_FLOAT16;
    ge::DataType qScaleType = ge::DT_FLOAT16;
    ge::DataType kScaleType = ge::DT_FLOAT16;
    ge::DataType outType = ge::DT_INT32;
    ge::DataType valuesType = ge::DT_BF16;
    ge::DataType cuSeqQType = ge::DT_INT32;
    ge::DataType cuSeqKType = ge::DT_INT32;
    ge::DataType sequsedQType = ge::DT_INT32;
    ge::DataType sequsedKType = ge::DT_INT32;
    ge::DataType cmpResidualType = ge::DT_INT32;
    ge::DataType blockTableType = ge::DT_INT32;
    ge::DataType idxOffsetType = ge::DT_INT32;
    std::string layoutQ = "BSND";
    std::string layoutK = "PA_BBND";
    int64_t topk = 2048;
    int64_t quantMode = 2;
    int64_t maxSeqlenQ = -1;
    int64_t maskMode = 0;
    int64_t cmpRatio = 1;
    int64_t returnValue = 0;
};

inline gert::StorageShape ToStorageShape(const std::vector<int64_t> &dims)
{
    gert::StorageShape shape;
    if (dims.empty()) {
        return shape;
    }
    shape.MutableShape().SetDimNum(dims.size());
    shape.MutableStorageShape().SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); i++) {
        shape.MutableShape().SetDim(i, dims[i]);
        shape.MutableStorageShape().SetDim(i, dims[i]);
    }
    return shape;
}

inline gert::TilingContextPara::TensorDescription Desc(const std::vector<int64_t> &dims, ge::DataType dtype)
{
    return gert::TilingContextPara::TensorDescription(ToStorageShape(dims), dtype, ge::FORMAT_ND);
}

inline void RunTilingCase(const CaseParam &p, ge::graphStatus expect)
{
    struct QLIV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara para(
        "QuantLightningIndexerV2",
        {Desc(p.qShape, p.qType), Desc(p.kShape, p.kType), Desc(p.wShape, p.wType), Desc(p.qScaleShape, p.qScaleType),
         Desc(p.kScaleShape, p.kScaleType), Desc(p.cuSeqQ, p.cuSeqQType), Desc(p.cuSeqK, p.cuSeqKType),
         Desc(p.sequsedQ, p.sequsedQType), Desc(p.sequsedK, p.sequsedKType), Desc(p.cmpResidual, p.cmpResidualType),
         Desc(p.blockTable, p.blockTableType), Desc(p.idxOffset, p.idxOffsetType), Desc(p.metadata, ge::DT_INT32)},
        {Desc(p.outShape, p.outType), Desc(p.valuesShape, p.valuesType)},
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(p.topk)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(p.quantMode)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(p.maxSeqlenQ)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(p.layoutQ)},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>(p.layoutK)},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(p.maskMode)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(p.cmpRatio)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(p.returnValue)}},
        &compileInfo, p.soc, p.coreNum, p.ubSize, p.l2Size);
    ExecuteTestCase(para, expect, SKIP_TILING_KEY);
}

} // namespace qliv2_ut

#endif // TEST_QUANT_LIGHTNING_INDEXER_V2_UTILS_H
