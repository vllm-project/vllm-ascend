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
 * \file weight_quant_batch_matmul_v2_tiling.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_H

#include "weight_quant_batch_matmul_v2_tiling_tool.h"
#include "weight_quant_batch_matmul_v2_tiling_key.h"
#include "op_host/tiling_base.h"
#include "../../op_kernel/weight_quant_batch_matmul_v2_tiling_data.h"

using Ops::Transformer::OpTiling::TilingBaseClass;

namespace optiling {

enum class Mc2QuantType {
    NONE = 0,
    PER_TENSOR = 1,
    PER_CHANNEL = 2,
    PER_GROUP = 3,
    MX = 4,
};

enum class Mc2KernelTemplateType {
    SERIAL = 0,
    GENERAL_PARALLEL = 1,
    SPLIT_K = 2,
    CUSTOM_ANTIQUANT = 3,
    MSD_MULTI_CORE = 6,
    MSD_GROUP = 7,
    WEIGHT_NZ = 8,
    MIX_SPLIT_K = 9,
    ANTI_REG = 10,
};

enum class Mc2WeightFormat {
    ND = 0,
    FRACTAL_NZ = 1,
};

enum class Mc2KernelTemplateTypeExtra {
    MSD_GENERAL = 1,
    HIGH_PRECISION = 2,
};

// 对应6-9位TilingKey v100上nFirst模板自定义组合方式
enum class Mc2Mte2Configuration : uint32_t {
    MTE2_INNER_SIZE_512_BUF_NUM_2 = 0,
    MTE2_INNER_SIZE_512_BUF_NUM_4 = 1,
    MTE2_INNER_SIZE_1024_BUF_NUM_2 = 2,
    MTE2_INNER_SIZE_256_BUF_NUM_4 = 3,
    MTE2_INNER_SIZE_512_BUF_NUM_DEFAULT = 4, // w8 w4在非性能场景下复用一组设置
    MTE2_INNER_SIZE_256_BUF_NUM_2 = 5,
    MTE2_INNER_SIZE_1024_BUF_NUM_4 = 6,
};

struct Mc2WeightQuantBatchMatmulInfo {
    bool transA = false;
    bool transB = false;
    bool hasBias = false;
    bool hasAntiQuantOffset = false;
    uint64_t groupSize = 0L;
    uint64_t mSize = 0L;
    uint64_t kSize = 0L;
    uint64_t nSize = 0L;
    ge::DataType aDtype = ge::DT_FLOAT16;
    ge::DataType bDtype = ge::DT_INT8;
    ge::DataType cDtype = ge::DT_FLOAT16;
    ge::DataType biasDtype = ge::DT_FLOAT16;
    ge::DataType antiQuantScaleDtype = ge::DT_FLOAT16;
    Mc2QuantType antiQuantType = Mc2QuantType::NONE;
    Mc2QuantType quantType = Mc2QuantType::PER_TENSOR;
    // 整改Base类时统一换成使用opName_
    const char *opName;
    ge::Format aFormat = ge::FORMAT_ND;
    ge::Format bFormat = ge::FORMAT_ND;
    uint64_t innerPrecise = 0;

    uint64_t batchX0 = 1L;
    uint64_t batchX1 = 1L;
    uint64_t batchX2 = 1L;
    uint64_t batchX3 = 1L;
    uint64_t batchWeight0 = 1L;
    uint64_t batchWeight1 = 1L;
    uint64_t batchWeight2 = 1L;
    uint64_t batchWeight3 = 1L;
    uint64_t batchY0 = 1L;
    uint64_t batchY1 = 1L;
    uint64_t batchY2 = 1L;
    uint64_t batchY3 = 1L;
    bool biasWithBatch = false;
};

struct Mc2WeightQuantBatchMatmulV2CompileInfo {
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0cSize;
    uint64_t l0aSize;
    uint64_t l0bSize;
    uint32_t workspaceNum;
    uint32_t aivNum;
    uint32_t aicNum;
    platform_ascendc::SocVersion socVersion;
    bool supportMmadS8S4;
};

class Mc2WeightQuantBatchMatmulV2Tiling : public TilingBaseClass {
public:
    using TilingBaseClass::Reset;

    explicit Mc2WeightQuantBatchMatmulV2Tiling(gert::TilingContext *context) : TilingBaseClass(context)
    {
    }

    ~Mc2WeightQuantBatchMatmulV2Tiling() override = default;

protected:
    bool IsCapable() override
    {
        return true;
    }
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    void SetCommonTilingKeyElement(Mc2TilingKeyConfigure &tilingKeyConfigure) const;

    void Reset() {};
    void InitCompileInfo();
    // 算子名称
    const char *opName_;

    // 伪量化输入信息
    std::unique_ptr<Mc2WeightQuantBatchMatmulInfo> matmulInfoPtr_;

    // 平台相关信息
    std::unique_ptr<Mc2WeightQuantBatchMatmulV2CompileInfo> compileInfoPtr_;
};

ge::graphStatus Mc2CheckPara(gert::TilingContext *context, platform_ascendc::SocVersion socVersion);

bool CheckTempLimit(Mc2WeightQuantBatchMatmulInfo *inputParams);

void GetDtype(Mc2WeightQuantBatchMatmulInfo &matmulInfo, const gert::TilingContext *context);

void GetAttrs(Mc2WeightQuantBatchMatmulInfo &matmulInfo, const gert::TilingContext *context);

void GetInputs(Mc2WeightQuantBatchMatmulInfo &matmulInfo, const gert::TilingContext *context);

bool CheckInputShape(Mc2WeightQuantBatchMatmulInfo *inputParams, const gert::StorageShape *xShape,
                     const gert::StorageShape *weightShape);

bool CheckDtype(gert::TilingContext *context, Mc2WeightQuantBatchMatmulInfo *inputParams,
                platform_ascendc::SocVersion socVersion);

bool CheckInputDtype(gert::TilingContext *context, Mc2WeightQuantBatchMatmulInfo *inputParams,
                     platform_ascendc::SocVersion socVersion);

bool CheckAntiQuantDtype(gert::TilingContext *context, Mc2WeightQuantBatchMatmulInfo *inputParams,
                         platform_ascendc::SocVersion socVersion);

bool CheckQuantDtype(gert::TilingContext *context, Mc2WeightQuantBatchMatmulInfo *inputParams);

bool CheckShapeDims(Mc2WeightQuantBatchMatmulInfo *inputParams, platform_ascendc::SocVersion socVersion);

bool CheckBiasShape(Mc2WeightQuantBatchMatmulInfo *inputParams, const gert::StorageShape *biasShape);

bool CheckQuantShape(Mc2WeightQuantBatchMatmulInfo *inputParams, const gert::StorageShape *quantScaleShape,
                     const gert::StorageShape *quantOffsetShape);

bool CheckShape(gert::TilingContext *context, Mc2WeightQuantBatchMatmulInfo *inputParams,
                platform_ascendc::SocVersion socVersion);

bool CheckAntiQuantShape(Mc2WeightQuantBatchMatmulInfo *inputParams, const gert::StorageShape *antiQuantScaleShape,
                         const gert::StorageShape *antiQuantOffsetShape);

bool CheckAttr(gert::TilingContext *context, Mc2WeightQuantBatchMatmulInfo *inputParams);
} // namespace optiling
#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_H
