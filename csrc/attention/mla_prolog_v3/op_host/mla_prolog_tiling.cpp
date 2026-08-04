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
 * \file mla_prolog_tiling.cpp
 * \brief
 */

#include <numeric>
#include <functional>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "err/ops_err.h"
#include "register/op_def_registry.h"
#include "mla_prolog_tiling_check.h"
#include "mla_prolog_tiling.h"
using namespace ge;
namespace optiling {

const std::unordered_map<ge::DataType, uint32_t> DTYPE_TO_SIZE{
    {ge::DT_BF16, 2},        {ge::DT_FLOAT16, 2},  {ge::DT_INT8, 1},  {ge::DT_FLOAT8_E4M3FN, 1},
    {ge::DT_FLOAT8_E8M0, 1}, {ge::DT_HIFLOAT8, 1}, {ge::DT_INT32, 4}, {ge::DT_FLOAT, 4}};

const std::unordered_map<ge::DataType, matmul_tiling::DataType> GE_TO_MM_DTYPE{
    {ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16},
    {ge::DT_BF16, matmul_tiling::DataType::DT_BF16},
    {ge::DT_INT8, matmul_tiling::DataType::DT_INT8},
    {ge::DT_INT4, matmul_tiling::DataType::DT_INT4},
    {ge::DT_FLOAT, matmul_tiling::DataType::DT_FLOAT},
    {ge::DT_FLOAT8_E4M3FN, matmul_tiling::DataType::DT_FLOAT8_E4M3FN},
    {ge::DT_FLOAT8_E8M0, matmul_tiling::DataType::DT_FLOAT8_E8M0},
    {ge::DT_HIFLOAT8, matmul_tiling::DataType::DT_HIFLOAT8}};

template <typename T>
inline auto CeilDiv(T a, T b) -> T
{
    if (b == 0) {
        return b;
    }
    return (a + b - 1) / b;
}

template <typename T>
inline auto Align(T num, T rnd) -> T
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

NpuArch MlaPrologTiling::GetCurNpuArch() const
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->platformInfo);
    NpuArch npuArch = ascendcPlatform.GetCurNpuArch();
    return npuArch;
}

ge::graphStatus MlaPrologTiling::GetNpuInfo()
{
    OP_CHECK_IF(context_->platformInfo == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context_->opName, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->platformInfo);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize_);

    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();

    OP_CHECK_IF(aicNum_ == 0 || aivNum_ == 0,
                OPS_REPORT_VECTOR_INNER_ERR(context_->opName, "num of core obtained is 0."), return GRAPH_FAILED);

    OP_CHECK_IF((aicNum_ != aivNum_) && (aicNum_ * 2 != aivNum_),
                OPS_REPORT_VECTOR_INNER_ERR(context_->opName, "aicNum(%u):aivNum(%u) only support 1:1 or 1:2", aicNum_,
                                            aivNum_),
                return GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

QUANT_MODE MlaPrologTiling::GetQuantizationModeV3() const
{
    if (*(context_->weightQuantMode) == static_cast<int>(WEIGHT_QUANT_MODE::NO_QUANT)) {
        if (*(context_->kvQuantMode) == static_cast<int>(KV_QUANT_MODE::NO_QUANT)) {
            return QUANT_MODE::NO_QUANT;
        } else {
            OP_LOGE(context_->opName, "When weightQuantMode == 0, kvQuantMode must be within {0}, actually is %ld.",
                    *(context_->kvQuantMode));
        }
    } else if (*(context_->weightQuantMode) == static_cast<int>(WEIGHT_QUANT_MODE::PARTIAL_QUANT)) {
        if (*(context_->kvQuantMode) == static_cast<int>(KV_QUANT_MODE::NO_QUANT)) {
            return QUANT_MODE::PARTIAL_QUANT_KV_NO_QUANT;
        } else if (*(context_->kvQuantMode) == static_cast<int>(KV_QUANT_MODE::PER_CHANNEL)) {
            return QUANT_MODE::PARTIAL_QUANT_KV_QUANT_PER_CHANNEL;
        } else if (*(context_->kvQuantMode) == static_cast<int>(KV_QUANT_MODE::PER_TILE)) {
            return QUANT_MODE::PARTIAL_QUANT_KV_QUANT_PER_TILE;
        } else {
            OP_LOGE(context_->opName,
                    "When weightQuantMode == 1, kvQuantMode must be within {0, 2, 3}, actually is %ld.",
                    *(context_->kvQuantMode));
        }
    } else if (*(context_->weightQuantMode) == static_cast<int>(WEIGHT_QUANT_MODE::FULL_QUANT)) {
        if (*(context_->kvQuantMode) == static_cast<int>(KV_QUANT_MODE::NO_QUANT)) {
            return QUANT_MODE::FULL_QUANT_KV_NO_QUANT;
        } else if (*(context_->kvQuantMode) == static_cast<int>(KV_QUANT_MODE::PER_TENSOR)) {
            return QUANT_MODE::FULL_QUANT_KV_QUANT_PER_TENSOR;
        } else if (*(context_->kvQuantMode) == static_cast<int>(KV_QUANT_MODE::PER_TILE)) {
            return QUANT_MODE::FULL_QUANT_KV_QUANT_PER_TILE;
        } else {
            OP_LOGE(context_->opName,
                    "When weightQuantMode == 2, kvQuantMode must be within {0, 1, 3}, actually is %ld.",
                    *(context_->kvQuantMode));
        }
    } else {
        OP_LOGE(context_->opName, "WeightQuantMode must be within {0, 1, 2}, actually is %ld.",
                *(context_->weightQuantMode));
    }
    return QUANT_MODE::ERROR_MODE;
}

QUANT_MODE MlaPrologTiling::GetQuantizationModeV3Dav() const
{
    const int weightQuantMode = *(context_->weightQuantMode);
    const int kvQuantMode = *(context_->kvQuantMode);

    // 卫语句1：weightQuantMode 越界（外层哈希找不到）
    auto wqIt = QUANT_MODE_HASH_TABLE.find(weightQuantMode);
    if (wqIt == QUANT_MODE_HASH_TABLE.end()) {
        OP_LOGE_FOR_INVALID_VALUE(context_->opName, "weightQuantMode", std::to_string(weightQuantMode),
                                  "{0, 1, 2, 3, 4, 5}");
        return QUANT_MODE::ERROR_MODE;
    }

    // 卫语句2：kvQuantMode 越界或组合非法（内层哈希找不到）
    auto kvIt = wqIt->second.find(kvQuantMode);
    if (kvIt == wqIt->second.end()) {
        auto reasonIt = VALID_KV_REASON_TABLE.find(weightQuantMode);
        const char *reason = (reasonIt != VALID_KV_REASON_TABLE.end()) ? reasonIt->second : "invalid kvQuantMode";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->opName, "kvQuantMode", std::to_string(kvQuantMode), reason);
        return QUANT_MODE::ERROR_MODE;
    }

    // hash 命中，返回对应场景
    return kvIt->second;
}

QUANT_MODE MlaPrologTiling::GetQuantizationMode() const
{
    if (std::strncmp(context_->opType, V3_OP_NAME, OP_NAME_LEN) == 0) {
        if (GetCurNpuArch() == NpuArch::DAV_3510) {
            return GetQuantizationModeV3Dav();
        } else {
            return GetQuantizationModeV3();
        }
    } else {
        if (context_->tokenX.desc->GetDataType() == ge::DT_INT8) {
            if (context_->kvCache.desc->GetDataType() == ge::DT_INT8) {
                return QUANT_MODE::FULL_QUANT_KV_QUANT_PER_TENSOR;
            } else {
                return QUANT_MODE::FULL_QUANT_KV_NO_QUANT;
            }
        }
        if (context_->tokenX.desc->GetDataType() == ge::DT_BF16 &&
            context_->weightUqQr.desc->GetDataType() == ge::DT_INT8) {
            if (context_->kvCache.desc->GetDataType() == ge::DT_INT8) {
                return QUANT_MODE::PARTIAL_QUANT_KV_QUANT_PER_CHANNEL;
            } else {
                return QUANT_MODE::PARTIAL_QUANT_KV_NO_QUANT;
            }
        }
        return QUANT_MODE::NO_QUANT;
    }
    return QUANT_MODE::ERROR_MODE;
}

ge::graphStatus MlaPrologTiling::SetShapeInfo()
{
    if (context_->tokenX.shape->GetStorageShape().GetDimNum() == MLA_PROLOG_DIM_NUM_3) {
        baseShapeInfo_.bSize = context_->tokenX.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_0);
        baseShapeInfo_.s1Size = context_->tokenX.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_1);
        baseShapeInfo_.heSize = context_->tokenX.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_2);
        baseShapeInfo_.tSize = baseShapeInfo_.bSize * baseShapeInfo_.s1Size;
    } else {
        baseShapeInfo_.tSize = context_->tokenX.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_0);
        baseShapeInfo_.heSize = context_->tokenX.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_1);
    }
    if (context_->weightDq.shape->GetStorageShape().GetDimNum() == MLA_PROLOG_DIM_NUM_2) {
        baseShapeInfo_.hcqSize = context_->weightDq.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_1);
    } else {
        uint32_t weightDqAxisSize_ = 32U / ge::GetSizeByDataType(context_->weightDq.desc->GetDataType());
        // weightDq: [He, Hcq] -> [Hcq/16, He/16, 16, 16] || [Hcq/32, He/16, 16, 32]
        baseShapeInfo_.hcqSize =
            weightDqAxisSize_ * context_->weightDq.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_0);
    }
    baseShapeInfo_.nSize = context_->weightUk.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_0);
    if (context_->ropeCos.shape != nullptr) {
        baseShapeInfo_.drSize = context_->ropeCos.shape->GetStorageShape().GetDim(
            context_->ropeCos.shape->GetStorageShape().GetDimNum() - 1);
    } else {
        // RoPE off: infer Dr from kr_cache last dim (or default 64).
        OP_CHECK_IF(context_->krCache.shape == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(context_->opName, "krCache"), return ge::GRAPH_FAILED);
        const auto &krShape = context_->krCache.shape->GetStorageShape();
        baseShapeInfo_.drSize = static_cast<uint32_t>(krShape.GetDim(krShape.GetDimNum() - 1));
    }
    baseShapeInfo_.dSize = context_->weightUk.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_1);
    baseShapeInfo_.headSizeQc = baseShapeInfo_.dSize * baseShapeInfo_.nSize;
    baseShapeInfo_.headSizeQr = baseShapeInfo_.drSize * baseShapeInfo_.nSize;
    baseShapeInfo_.headSizeUqQr = baseShapeInfo_.headSizeQc + baseShapeInfo_.headSizeQr;
    if (context_->kvCache.shape->GetStorageShape().GetDimNum() == MLA_PROLOG_DIM_NUM_3) {
        baseShapeInfo_.blockNum = context_->kvCache.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_0);
        baseShapeInfo_.nkvSize = context_->kvCache.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_1);
        baseShapeInfo_.dtileSize = context_->kvCache.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_2);
    } else {
        baseShapeInfo_.blockNum = context_->kvCache.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_0);
        baseShapeInfo_.blockSize = context_->kvCache.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_1);
        baseShapeInfo_.nkvSize = context_->kvCache.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_2);
        baseShapeInfo_.dtileSize = context_->kvCache.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_3);
    }
    if (context_->weightDkvKr.shape->GetStorageShape().GetDimNum() == MLA_PROLOG_DIM_NUM_4) {
        baseShapeInfo_.hckvSize = context_->weightDkvKr.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_0) *
                                      context_->weightDkvKr.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_3) -
                                  baseShapeInfo_.drSize;
    } else {
        baseShapeInfo_.hckvSize =
            context_->weightDkvKr.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_1) - baseShapeInfo_.drSize;
    }
    baseShapeInfo_.s2Size = baseShapeInfo_.nkvSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaPrologTiling::SetScenarioInfo()
{
    scenarioInfo_.isV1Flag_ = (std::strncmp(context_->opType, V1_OP_NAME, OP_NAME_LEN) == 0);
    scenarioInfo_.batchSeqFusedFlag_ = context_->tokenX.shape->GetStorageShape().GetDimNum() == MLA_PROLOG_DIM_NUM_2;
    scenarioInfo_.quantMode_ = GetQuantizationMode();
    if (scenarioInfo_.quantMode_ == QUANT_MODE::ERROR_MODE) {
        return ge::GRAPH_FAILED;
    }

    // 由 quantMode_ 反向映射 wq/kvq（V1/V2 按 dtype 推断、V3 正向查表，均唯一可反推）
    const auto &wqKvq = QUANT_MODE_REVERSE_TABLE.at(scenarioInfo_.quantMode_);
    scenarioInfo_.weightQuantMode_ = wqKvq.first;
    scenarioInfo_.kvQuantMode_ = wqKvq.second;

    // cacheMode 字符串 → 枚举，hash 查表（CheckCacheMode 已保证 cacheMode 合法，必命中）
    scenarioInfo_.cacheMode_ = CACHE_MODE_HASH_TABLE.at(std::string(context_->cacheMode));

    if ((scenarioInfo_.batchSeqFusedFlag_ && baseShapeInfo_.tSize == 0U) ||
        (!scenarioInfo_.batchSeqFusedFlag_ && (baseShapeInfo_.bSize * baseShapeInfo_.s1Size == 0U))) {
        scenarioInfo_.emptyTensorMode_ = EMPTY_TENSOR_MODE::EMPTY_QUERY;
    } else if (baseShapeInfo_.blockNum == 0U) {
        scenarioInfo_.emptyTensorMode_ = EMPTY_TENSOR_MODE::EMPTY_CACHE;
    } else {
        scenarioInfo_.emptyTensorMode_ = EMPTY_TENSOR_MODE::NON_EMPTY;
    }

    if (scenarioInfo_.batchSeqFusedFlag_ &&
        (scenarioInfo_.cacheMode_ == CACHE_MODE::PA_BLK_BSND || scenarioInfo_.cacheMode_ == CACHE_MODE::PA_BLK_NZ)) {
        OP_CHECK_IF(context_->actualSeqLen.shape == nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->opName, "actualSeqLen", "null",
                                                             "When cacheMode in {PA_BLK_BSND, PA_BLK_NZ} and tokenX "
                                                             "shape dim num is 2, actualSeqLen should not be null"),
                    return GRAPH_FAILED);
        baseShapeInfo_.bSize = context_->actualSeqLen.shape->GetStorageShape().GetDim(MLA_PROLOG_DIM_INDEX_0);
        scenarioInfo_.actualSeqMode_ = ACTUAL_SEQ_MODE::EN_Q_LEN;
    } else {
        scenarioInfo_.actualSeqMode_ = ACTUAL_SEQ_MODE::DISABLED;
    }
    uint32_t cvRatio = aivNum_ / aicNum_;
    // 当前仅在BS>=8K且数据类型为MXFP8时路由到切M模板，其他情况均路由到切N模板
    scenarioInfo_.splitMFlag_ = 0U;
    if (scenarioInfo_.weightQuantMode_ == WEIGHT_QUANT_MODE::MXFP8_FULL_QUANT &&
        baseShapeInfo_.tSize >= 8192 && cvRatio == 2) { // 8192：BS >= 8K
        if ((baseShapeInfo_.heSize == HEAD_SIZE1 || baseShapeInfo_.heSize == HEAD_SIZE2) &&
            baseShapeInfo_.nSize == 128) { // 128：N为128时路由到切M模板
            scenarioInfo_.splitMFlag_ = 1U;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaPrologTiling::SetAttrInfo()
{
    reciprocalCq_ = 1.0f / baseShapeInfo_.hcqSize;
    epsilonCq_ = *(context_->rmsNormEspilonCq);
    reciprocalCkv_ = 1.0f / baseShapeInfo_.hckvSize;
    epsilonCkv_ = *(context_->rmsNormEspilonCkv);
    if (std::strncmp(context_->opType, V3_OP_NAME, OP_NAME_LEN) == 0) {
        queryNormFlag_ = *(context_->queryNormFlag);
        weightQuantMode_ = static_cast<WEIGHT_QUANT_MODE>(*(context_->weightQuantMode));
        kvQuantMode_ = static_cast<KV_QUANT_MODE>(*(context_->kvQuantMode));
        queryQuantMode_ = static_cast<QUERY_QUANT_MODE>(*(context_->queryQuantMode));
        ckvkrRepoMode_ = static_cast<CKVKR_REPO_MODE>(*(context_->ckvkrRepoMode));
        quantSacleRepoMode_ = static_cast<QUANT_SCALE_REPO_MODE>(*(context_->quantScaleRepoMode));
        tileSize_ = static_cast<uint32_t>(*(context_->tileSize));
        qcQrScale_ = *(context_->qcQrScale);
        kcScale_ = *(context_->kcScale);
    }

    enableRope_ = true;
    if (GetCurNpuArch() == NpuArch::DAV_3510 && context_->doRope != nullptr) {
        enableRope_ = *(context_->doRope);
    }
    return ge::GRAPH_SUCCESS;
}

bool MlaPrologTiling::GetMatmulType(ge::DataType getype, matmul_tiling::DataType *mmType)
{
    auto mmdt = GE_TO_MM_DTYPE.find(getype);
    if (mmdt != GE_TO_MM_DTYPE.end()) {
        *mmType = mmdt->second;
        return true;
    }
    return false;
}

uint32_t MlaPrologTiling::CalcSingleCoreN(uint32_t n, uint32_t coreNum, uint32_t alignNum) const
{
    return CeilDiv(n, alignNum * coreNum) * alignNum;
}

// mm1.m = stepBatchSize            // 32
// mm1.n = singlecoreHeadSizeCq     // 64
// mm1.k = headSizeX                // 7168
// mm1.baseM = stepBatchSize        // 32
// mm1.baseN = singlecoreHeadSizeCq // 64
// mm1.baseK = 256
ge::graphStatus MlaPrologTiling::FillMatmul1Tiling()
{
    if (scenarioInfo_.splitMFlag_ == 1U) {
        singlecoreHeadSizeCq_ = baseShapeInfo_.hcqSize;
        mm1BlockNum_ = aicNum_;
    } else {
        auto dataType = context_->weightDq.desc->GetDataType();
        singlecoreHeadSizeCq_ =
            CalcSingleCoreN(baseShapeInfo_.hcqSize, aicNum_, BLOCK_SIZE / DTYPE_TO_SIZE.at(dataType));
        singlecoreHeadSizeCq_ = std::max(singlecoreHeadSizeCq_, 64U); // 64：最大使用24核
        mm1BlockNum_ = CeilDiv(baseShapeInfo_.hcqSize, singlecoreHeadSizeCq_);
    }
    return ge::GRAPH_SUCCESS;
}

// singlecoreHeadSizeCkvKr =  HeadSizeCkvDr / mm2CoreNum // 576 / 9 == 64
// mm2.m = stepBatchSize
// mm2.n = singlecoreHeadSizeCkvKr
// mm2.k = headSizeX // size of He
// mm2.baseN = n
// mm2.baseK = 256
ge::graphStatus MlaPrologTiling::FillMatmul2Tiling()
{
    if (scenarioInfo_.emptyTensorMode_ == EMPTY_TENSOR_MODE::EMPTY_CACHE) {
        return ge::GRAPH_SUCCESS;
    }
    if (scenarioInfo_.splitMFlag_ == 1U) {
        singlecoreHeadSizeCkvKr_ = baseShapeInfo_.hckvSize + baseShapeInfo_.drSize;
        mm2BlockNum_ = aicNum_;
    } else if (aicNum_ >= 9U) { // 9是经验值
        uint32_t baseN = 64U;
        mm2BlockNum_ = (baseShapeInfo_.hckvSize + baseShapeInfo_.drSize) / baseN;
        singlecoreHeadSizeCkvKr_ = baseN;
    } else {
        auto dataType = context_->weightDkvKr.desc->GetDataType();
        singlecoreHeadSizeCkvKr_ = CalcSingleCoreN(baseShapeInfo_.hckvSize + baseShapeInfo_.drSize, aicNum_,
                                                   BLOCK_SIZE / DTYPE_TO_SIZE.at(dataType));
        mm2BlockNum_ = CeilDiv(baseShapeInfo_.hckvSize + baseShapeInfo_.drSize, singlecoreHeadSizeCkvKr_);
    }
    return ge::GRAPH_SUCCESS;
}

// singlecoreHeadSizeQcQr = headNum * (dimHeadSizeQc + dimHeadRope) / mm3CoreNum  = 32 * (128 + 64) / 24
// mm3.m = stepBatchSize
// mm3.n = singlecoreHeadSizeQcQr   // 256
// mm3.k = headSizeCq // size of Hcq   1536
// mm3.baseN = 64  //
// mm3.baseK = 256 //
ge::graphStatus MlaPrologTiling::FillMatmul3Tiling()
{
    auto dataType = context_->weightUqQr.desc->GetDataType();
    auto oriM = baseShapeInfo_.nSize * (baseShapeInfo_.dSize + baseShapeInfo_.drSize);
    if (enableGroupComputeOpt_) {
        // 算力分组场景下G=8，dimHeadSizeQc跨8核切，dimHeadSizeQr跨4核切；matmulQc和matmulQr的singleN都取128
        singlecoreHeadSizeQcQr_ = CalcSingleCoreN(baseShapeInfo_.nSize * baseShapeInfo_.dSize,
                                                  GROUP_COMPUTE_CUBE_NUM_PER_GROUP, baseShapeInfo_.dSize);
    } else if (enableDequantOpt_) {
        // dequant流水掩盖场景，dimHeadSizeQc + dimHeadRope不跨核
        singlecoreHeadSizeQcQr_ = CalcSingleCoreN(oriM, aicNum_, baseShapeInfo_.dSize + baseShapeInfo_.drSize);
    } else {
        // headnum * (dimHeadSizeQc + dimHeadRope) 合轴切
        singlecoreHeadSizeQcQr_ = CalcSingleCoreN(oriM, aicNum_, BLOCK_SIZE / DTYPE_TO_SIZE.at(dataType));
    }
    mm3BlockNum_ = CeilDiv(oriM, singlecoreHeadSizeQcQr_);

    if (scenarioInfo_.splitMFlag_ == 1U) {
        singlecoreHeadSizeQcQr_ = oriM;
        mm3BlockNum_ = aicNum_;
    }

    return ge::GRAPH_SUCCESS;
}

// mm4.m = stepBatchSize
// mm4.n = headSizeCkv  // 512
// mm4.k = dimHeadSizeQc // size of Qc  128
// mm4.baseN = 128  //
// mm4.baseK = 128 //
// mm4.Kstride = dimHeadSizeQc + dimHeadRope
ge::graphStatus MlaPrologTiling::FillMatmul4Tiling()
{
    if (scenarioInfo_.splitMFlag_ == 1U) {
        singlecoreNumHeadSize_ = baseShapeInfo_.nSize;
        mm4BlockNum_ = aicNum_;
    } else {
        singlecoreNumHeadSize_ = CeilDiv(baseShapeInfo_.nSize, aicNum_);
        mm4BlockNum_ = CeilDiv(baseShapeInfo_.nSize, singlecoreNumHeadSize_);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaPrologTiling::ProcessBaseInputs()
{
    stepBatchSize_ = std::min(128U, baseShapeInfo_.tSize);
    if (scenarioInfo_.splitMFlag_ == 1U && (stepBatchSize_ > 0U) && (aicNum_ > 0U) && (baseShapeInfo_.tSize > 0U)) {
        mSubSize_ = (baseShapeInfo_.tSize + aicNum_ - 1U) / aicNum_;
        // idx为[0, mSubCoreNum_]的核分到mSubSize_，其余核分到mSubSize_ - 1
        mSubCoreNum_ = baseShapeInfo_.tSize - (mSubSize_ - 1U) * aicNum_;
    }
    if (baseShapeInfo_.dSize == HIGH_THROUGHPUT__D_SIZE) {
        stepNumHeadDequant_ = std::min(64U, baseShapeInfo_.nSize);
    } else {
        stepNumHeadDequant_ = std::min(16U, baseShapeInfo_.nSize);
    }
    vectorBlockNum_ = std::min(stepBatchSize_, aivNum_);

    uint32_t cvRatio = aivNum_ / aicNum_;
    // 算力分组开关，仅当半量化场景，BS=1，G=8，可用核数大于等于16时进入分支
    // CV1:1时不支持分组计算场景
    if ((scenarioInfo_.quantMode_ == QUANT_MODE::PARTIAL_QUANT_KV_NO_QUANT ||
         scenarioInfo_.quantMode_ == QUANT_MODE::PARTIAL_QUANT_KV_QUANT_PER_CHANNEL) &&
        baseShapeInfo_.tSize == GROUP_COMPUTE_T_SIZE && baseShapeInfo_.nkvSize == GROUP_COMPUTE_N_SIZE &&
        aivNum_ >= GROUP_COMPUTE_MIN_AIV_NUM && aicNum_ >= GROUP_COMPUTE_MIN_AIC_NUM && cvRatio != 1) {
        enableGroupComputeOpt_ = true;
        aivNum_ = 32U;
        aicNum_ = 16U;
    } else if ((context_->weightUqQr.desc->GetDataType() == ge::DT_INT8 &&
                baseShapeInfo_.nSize >= GROUP_COMPUTE_N_SIZE) ||
               context_->weightUqQr.desc->GetDataType() == ge::DT_FLOAT8_E4M3FN ||
               context_->weightUqQr.desc->GetDataType() == ge::DT_HIFLOAT8) {
        // 场景1：INT8全量化且N大于等于8；场景2：MXFP8全量化场景
        // 通过切N处理MM3，MM4之后的操作例如Rope，DynamicQuant等会有性能收益
        enableDequantOpt_ = true;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaPrologTiling::FillTiling()
{
    baseParams_->tokenSize = baseShapeInfo_.tSize;
    baseParams_->seq1Size = baseShapeInfo_.s1Size;
    baseParams_->seq2Size = baseShapeInfo_.s2Size;
    baseParams_->headSizeX = baseShapeInfo_.heSize;
    baseParams_->headSizeCq = baseShapeInfo_.hcqSize;
    baseParams_->headSizeCkv = baseShapeInfo_.hckvSize;
    baseParams_->dtileSize = baseShapeInfo_.dtileSize;
    baseParams_->headSizeQc = baseShapeInfo_.headSizeQc;
    baseParams_->headSizeQr = baseShapeInfo_.headSizeQr;
    baseParams_->headSizeKr = baseShapeInfo_.drSize;
    baseParams_->numHeadSize = baseShapeInfo_.nSize;
    baseParams_->numHeadKvSize = baseShapeInfo_.nkvSize;
    baseParams_->dimHeadSizeQc = baseShapeInfo_.dSize;
    baseParams_->dimHeadRope = baseShapeInfo_.drSize;
    baseParams_->blockNum = baseShapeInfo_.blockNum;
    baseParams_->blockSize = baseShapeInfo_.blockSize;
    baseParams_->reciprocalCq = reciprocalCq_;
    baseParams_->epsilonCq = epsilonCq_;
    baseParams_->reciprocalCkv = reciprocalCkv_;
    baseParams_->epsilonCkv = epsilonCkv_;
    if (std::strncmp(context_->opType, V3_OP_NAME, OP_NAME_LEN) == 0) {
        baseParams_->queryNormFlag = queryNormFlag_ ? 1U : 0U;
        baseParams_->kvQuantMode = static_cast<uint32_t>(kvQuantMode_);
        baseParams_->ckvkrRepoMode = static_cast<uint32_t>(ckvkrRepoMode_);
        baseParams_->quantScaleRepoMode = static_cast<uint32_t>(quantSacleRepoMode_);
        baseParams_->tileSize = tileSize_;
        baseParams_->qcQrScale = qcQrScale_;
        baseParams_->kcScale = kcScale_;
        baseParams_->isQcQrScaleEnable =
            static_cast<uint16_t>(std::abs(qcQrScale_ - 1.0f) >= std::numeric_limits<float>::epsilon());
        baseParams_->isKcScaleEnable =
            static_cast<uint16_t>(std::abs(kcScale_ - 1.0f) >= std::numeric_limits<float>::epsilon());
    } else {
        baseParams_->queryNormFlag = 0U;
        baseParams_->kvQuantMode = 0U;
        baseParams_->ckvkrRepoMode = 0U;
        baseParams_->quantScaleRepoMode = 0U;
        baseParams_->tileSize = 128U;
        baseParams_->qcQrScale = 0;
        baseParams_->kcScale = 0;
        baseParams_->isQcQrScaleEnable = 0U;
        baseParams_->isKcScaleEnable = 0U;
    }
    FillTilingCoreParams(); // 分核相关baseParams
    return ge::GRAPH_SUCCESS;
}

void MlaPrologTiling::FillTilingCoreParams()
{
    baseParams_->batchSize = baseShapeInfo_.bSize;
    baseParams_->stepBatchSize = stepBatchSize_;
    baseParams_->stepNumHeadDequant = stepNumHeadDequant_;
    baseParams_->mSubSize = mSubSize_;
    baseParams_->mSubCoreNum = mSubCoreNum_;
    baseParams_->mm1BlockNum = mm1BlockNum_;
    baseParams_->mm2BlockNum = mm2BlockNum_;
    baseParams_->mm3BlockNum = mm3BlockNum_;
    baseParams_->mm4BlockNum = mm4BlockNum_;
    baseParams_->mm1SingleCoreN = singlecoreHeadSizeCq_;
    baseParams_->mm2SingleCoreN = singlecoreHeadSizeCkvKr_;
    baseParams_->mm3SingleCoreN = singlecoreHeadSizeQcQr_;
    baseParams_->mm4SingleCoreBatch = singlecoreNumHeadSize_;
    baseParams_->vectorBlockNum = vectorBlockNum_;
}

ge::graphStatus MlaPrologTiling::CalcWorkSpace()
{
    workspaceSize_ = libapiSize_;
    uint32_t mm1Mult = (scenarioInfo_.splitMFlag_ == 1U) ? mm1BlockNum_ : 1U;
    uint32_t mm2Mult = (scenarioInfo_.splitMFlag_ == 1U) ? mm2BlockNum_ : 1U;
    uint32_t mm3Mult = (scenarioInfo_.splitMFlag_ == 1U) ? mm3BlockNum_ : 1U;
    uint32_t mm4Mult = (scenarioInfo_.splitMFlag_ == 1U) ? mm4BlockNum_ : 1U;
    uint32_t dequantScaleMult = (scenarioInfo_.splitMFlag_ == 1U) ? static_cast<uint32_t>(aicNum_) : 1U;
    // 全量化场景：weightQuantMode 为 FULL/MXFP8/FP8/HIF8（即非 NO_QUANT 且非 PARTIAL）
    if (scenarioInfo_.weightQuantMode_ != WEIGHT_QUANT_MODE::NO_QUANT &&
        scenarioInfo_.weightQuantMode_ != WEIGHT_QUANT_MODE::PARTIAL_QUANT) {
        workspaceSize_ += static_cast<size_t>(stepBatchSize_) * static_cast<size_t>(baseShapeInfo_.hcqSize) * mm1Mult *
                          static_cast<size_t>(NUM_BYTES_INT32);
        workspaceSize_ += static_cast<size_t>(stepBatchSize_) * static_cast<size_t>(baseShapeInfo_.hcqSize) * mm1Mult *
                          static_cast<size_t>(NUM_BYTES_BF16);
        workspaceSize_ += static_cast<size_t>(stepBatchSize_) *
                          static_cast<size_t>(baseShapeInfo_.hckvSize + baseShapeInfo_.drSize) * mm2Mult *
                          static_cast<size_t>(NUM_BYTES_INT32);
        if (scenarioInfo_.kvQuantMode_ == KV_QUANT_MODE::PER_TENSOR) {
            // 全量化场景mmQnRes输出到workspace, B, S1, N, Hckv, BF16
            workspaceSize_ += static_cast<size_t>(stepBatchSize_) * static_cast<size_t>(baseShapeInfo_.nSize) *
                              static_cast<size_t>(baseShapeInfo_.hckvSize) * mm4Mult *
                              static_cast<size_t>(NUM_BYTES_BF16);
        }
    } else {
        workspaceSize_ += static_cast<size_t>(stepBatchSize_) * static_cast<size_t>(baseShapeInfo_.hcqSize) * mm1Mult *
                          static_cast<size_t>(NUM_BYTES_BF16) * static_cast<size_t>(2); // 2: double
        workspaceSize_ += static_cast<size_t>(stepBatchSize_) *
                          static_cast<size_t>(baseShapeInfo_.hckvSize + baseShapeInfo_.drSize) * mm2Mult *
                          static_cast<size_t>(NUM_BYTES_BF16);
    }
    workspaceSize_ += static_cast<size_t>(stepBatchSize_) *
                      static_cast<size_t>(baseShapeInfo_.headSizeQc + baseShapeInfo_.headSizeQr) * mm3Mult *
                      static_cast<size_t>(NUM_BYTES_INT32);
    workspaceSize_ += static_cast<size_t>(stepBatchSize_) * static_cast<size_t>(baseShapeInfo_.nSize) *
                      static_cast<size_t>(baseShapeInfo_.dSize) * mm3Mult * static_cast<size_t>(NUM_BYTES_BF16);
    // SplitM非pertile量化场景下 mmQcQrResDequant使用两份workspace
    if (scenarioInfo_.splitMFlag_ == 1U &&
        (scenarioInfo_.quantMode_ == QUANT_MODE::MXFP8_FULL_QUANT_KV_NO_QUANT ||
        scenarioInfo_.quantMode_ == QUANT_MODE::MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR)) {
        workspaceSize_ += static_cast<size_t>(stepBatchSize_) * static_cast<size_t>(baseShapeInfo_.nSize) *
                          static_cast<size_t>(baseShapeInfo_.dSize) * mm3Mult * static_cast<size_t>(NUM_BYTES_BF16);
    }

    if (enableGroupComputeOpt_ || enableDequantOpt_) {
        workspaceSize_ += static_cast<size_t>(stepBatchSize_) * dequantScaleMult * static_cast<size_t>(BLOCK_SIZE);
    }
    if (context_->workSpaces) {
        context_->workSpaces[0] = workspaceSize_;
    }
    OP_LOGI(context_->opName, "Tiling info: workspaceSize_ = %zu", workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaPrologTiling::GenTilingKey() const
{
    uint8_t typeValue = 0;
    uint8_t quantType = 0;
    if (scenarioInfo_.quantMode_ == QUANT_MODE::NO_QUANT) {
        typeValue = 1U;
    } else {
        typeValue = 2U;
        // kvCache量化场景，对应tiling key为1(半量化:0 + kv量化:1)或3(全量化:2 + kv量化:1)
        // 全量化场景，对应tiling key为2+0(全量化:2)或2+1（全量化:2+ kv量化:1）
        // 非量化和半量化场景，对应tiling key为0
        quantType = static_cast<uint8_t>(scenarioInfo_.quantMode_);
    }

    uint8_t cvMode = ASCENDC_TPL_MIX_AIC_1_2; // 默认cv 1:2模式
    if (aivNum_ == aicNum_) {
        cvMode = ASCENDC_TPL_MIX_AIC_1_1; // cv 1:1模式
    }

    if (cvMode == ASCENDC_TPL_MIX_AIC_1_1 &&
        (scenarioInfo_.quantMode_ != QUANT_MODE::NO_QUANT ||
         (scenarioInfo_.cacheMode_ != CACHE_MODE::PA_BSND && scenarioInfo_.cacheMode_ != CACHE_MODE::PA_NZ))) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context_->opName, "quantMode",
            std::to_string(static_cast<uint32_t>(scenarioInfo_.quantMode_)) + ", cacheMode is " +
                std::to_string(static_cast<uint32_t>(scenarioInfo_.cacheMode_)),
            "CV1:1 mode only support quantMode in {NO_QUANT} and cacheMode in {PA_BSND,PA_NZ}");
        return ge::GRAPH_FAILED;
    }

    if (scenarioInfo_.emptyTensorMode_ == EMPTY_TENSOR_MODE::EMPTY_QUERY) {
        context_->tilingKey = GET_TPL_TILING_KEY(0, 0, 0, false, false,
                                                 static_cast<uint8_t>(scenarioInfo_.emptyTensorMode_), 0, 0, cvMode,
                                                 true);
    } else {
        uint8_t cacheMode =
            scenarioInfo_.cacheMode_ == CACHE_MODE::TND ? 0 : static_cast<uint8_t>(scenarioInfo_.cacheMode_);
        context_->tilingKey = GET_TPL_TILING_KEY(
            static_cast<uint8_t>(cacheMode), typeValue, quantType, enableDequantOpt_, enableGroupComputeOpt_,
            static_cast<uint8_t>(scenarioInfo_.emptyTensorMode_), static_cast<uint8_t>(scenarioInfo_.actualSeqMode_),
            static_cast<uint8_t>(scenarioInfo_.splitMFlag_), cvMode, enableRope_);
        OP_LOGI(
            context_->opName,
            "MlaProlog tilingKey args: "
            "CACHE_MODE:%u, SCENARIO:%u, QUANT_MODE:%u, ENABLE_DEQUANT_OPTIONAL:%u, ENABLE_GROUP_COMPUTE_OPTIONAL:%u, "
            "EMPTY_TENSOR_MODE:%u, ACTUAL_SEQ_LEN_MODE:%u, SPLIT_M_MODE:%u, CV_MODE:%u, ENABLE_ROPE:%u",
            static_cast<uint8_t>(cacheMode), typeValue, quantType, enableDequantOpt_, enableGroupComputeOpt_,
            static_cast<uint8_t>(scenarioInfo_.emptyTensorMode_), static_cast<uint8_t>(scenarioInfo_.actualSeqMode_),
            static_cast<uint8_t>(scenarioInfo_.splitMFlag_), cvMode, static_cast<uint8_t>(enableRope_));
    }
    OP_LOGI(context_->opName, "MlaProlog tilingKey:%lu", context_->tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaPrologTiling::RunBigKernelTiling(MlaPrologContext &context, MlaPrologTilingData *tilingData)
{
    this->context_ = &context;
    this->baseParams_ = &tilingData->baseParams;
    MlaPrologTilingCheck tilingCheck_{*context_, baseShapeInfo_, scenarioInfo_};

    using StatusFunction = std::function<ge::graphStatus()>;
    std::vector<StatusFunction> requiredTilingFuncs{
        std::bind(&MlaPrologTiling::GetNpuInfo, this),
        std::bind(&MlaPrologTilingCheck::CheckAttrs, &tilingCheck_),
        std::bind(&MlaPrologTilingCheck::CheckSingleRequiredParam, &tilingCheck_),
        std::bind(&MlaPrologTilingCheck::CheckCacheMode, &tilingCheck_),
        std::bind(&MlaPrologTiling::SetShapeInfo, this),
        std::bind(&MlaPrologTiling::SetScenarioInfo, this),
        std::bind(&MlaPrologTilingCheck::CheckScenarParam, &tilingCheck_),
        std::bind(&MlaPrologTilingCheck::CheckQuantMode, &tilingCheck_),
        std::bind(&MlaPrologTilingCheck::CheckDims, &tilingCheck_),
        std::bind(&MlaPrologTilingCheck::CheckSpecialScenarioParamShape, &tilingCheck_),
        std::bind(&MlaPrologTilingCheck::CheckParamByScenario, &tilingCheck_),
        std::bind(&MlaPrologTiling::SetAttrInfo, this),
        std::bind(&MlaPrologTiling::ProcessBaseInputs, this),
    };
    for (const auto &func : requiredTilingFuncs) {
        if (func() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    if (scenarioInfo_.emptyTensorMode_ == EMPTY_TENSOR_MODE::EMPTY_QUERY) {
        FillTiling();
        if (context_->workSpaces) {
            context_->workSpaces[0] = libapiSize_;
        }
        GenTilingKey();
        context_->blockDim = 1U;
        return ge::GRAPH_SUCCESS;
    }

    std::vector<StatusFunction> optionalTilingFuncs{
        std::bind(&MlaPrologTiling::FillMatmul1Tiling, this), std::bind(&MlaPrologTiling::FillMatmul2Tiling, this),
        std::bind(&MlaPrologTiling::FillMatmul3Tiling, this), std::bind(&MlaPrologTiling::FillMatmul4Tiling, this),
        std::bind(&MlaPrologTiling::FillTiling, this),        std::bind(&MlaPrologTiling::CalcWorkSpace, this),
        std::bind(&MlaPrologTiling::GenTilingKey, this)};
    for (const auto &func : optionalTilingFuncs) {
        if (func() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    context_->blockDim = aicNum_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaPrologTiling::ConvertContext(gert::TilingContext &context, MlaPrologContext &mlaPrologContext)
{
    OP_CHECK_IF(context.GetNodeName() == nullptr, OP_LOGE_WITH_INVALID_INPUT(V1_OP_NAME, "OpName"),
                return ge::GRAPH_FAILED);

    mlaPrologContext.opName = context.GetNodeName();
    mlaPrologContext.opType = context.GetNodeType();
    mlaPrologContext.platformInfo = context.GetPlatformInfo();

    ConvertRequiredParams(context, mlaPrologContext);
    ConvertOptionalParams(context, mlaPrologContext);

    auto attrs = context.GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(context.GetNodeName(), "attrs"), return ge::GRAPH_FAILED);
    mlaPrologContext.rmsNormEspilonCq = attrs->GetAttrPointer<float>(RMS_NORM_EPSILON_CQ_ATTR_INDEX);
    mlaPrologContext.rmsNormEspilonCkv = attrs->GetAttrPointer<float>(RMS_NORM_EPSILON_CKV_ATTR_INDEX);
    mlaPrologContext.cacheMode = attrs->GetStr(CACHE_MODE_ATTR_INDEX);
    if (std::strncmp(mlaPrologContext.opType, V3_OP_NAME, OP_NAME_LEN) == 0) {
        mlaPrologContext.queryNormFlag = attrs->GetAttrPointer<bool>(QUERY_NORM_FLAG_ATTR_INDEX);
        mlaPrologContext.weightQuantMode = attrs->GetAttrPointer<int64_t>(WEIGHT_QUANT_MODE_ATTR_INDEX);
        mlaPrologContext.kvQuantMode = attrs->GetAttrPointer<int64_t>(KV_CACHE_QUANT_MODE_ATTR_INDEX);
        mlaPrologContext.queryQuantMode = attrs->GetAttrPointer<int64_t>(QUERY_QUANT_MODE_ATTR_INDEX);
        mlaPrologContext.ckvkrRepoMode = attrs->GetAttrPointer<int64_t>(CKVKR_REPO_MODE_ATTR_INDEX);
        mlaPrologContext.quantScaleRepoMode = attrs->GetAttrPointer<int64_t>(QUANT_SCALE_REPO_MODE_ATTR_INDEX);
        mlaPrologContext.tileSize = attrs->GetAttrPointer<int64_t>(TILE_SIZE_ATTR_INDEX);
        mlaPrologContext.qcQrScale = attrs->GetAttrPointer<float>(QC_QR_SCALE_ATTR_INDEX);
        mlaPrologContext.kcScale = attrs->GetAttrPointer<float>(KC_SCALE_ATTR_INDEX);
        // Infer RoPE switch from optional rope inputs (both null → off, both non-null → on).
        const bool ropeSinNull = (mlaPrologContext.ropeSin.shape == nullptr);
        const bool ropeCosNull = (mlaPrologContext.ropeCos.shape == nullptr);
        OP_CHECK_IF(ropeSinNull != ropeCosNull,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        mlaPrologContext.opName, "ropeSin / ropeCos", "(one null, one non-null)",
                        "ropeSin and ropeCos must both be null (RoPE off) or both non-null (RoPE on)"),
                    return ge::GRAPH_FAILED);
        mlaPrologContext.doRopeValue = !ropeSinNull;
        mlaPrologContext.doRope = &mlaPrologContext.doRopeValue;
    } else {
        mlaPrologContext.queryNormFlag = nullptr;
        mlaPrologContext.weightQuantMode = nullptr;
        mlaPrologContext.kvQuantMode = nullptr;
        mlaPrologContext.queryQuantMode = nullptr;
        mlaPrologContext.ckvkrRepoMode = nullptr;
        mlaPrologContext.quantScaleRepoMode = nullptr;
        mlaPrologContext.tileSize = nullptr;
        mlaPrologContext.qcQrScale = nullptr;
        mlaPrologContext.kcScale = nullptr;
        // 非 V3：RoPE 始终开启（rope 为必选输入）
        mlaPrologContext.doRopeValue = true;
        mlaPrologContext.doRope = nullptr;
    }

    OP_CHECK_IF(context.GetWorkspaceSizes(1) == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "workSpaceSize got from ge is nullptr"),
                return ge::GRAPH_FAILED);
    mlaPrologContext.workSpaces = context.GetWorkspaceSizes(1);
    return ge::GRAPH_SUCCESS;
}

void MlaPrologTiling::ConvertRequiredParams(gert::TilingContext &context, MlaPrologContext &mlaPrologContext)
{
    mlaPrologContext.tokenX.desc = context.GetRequiredInputDesc(TOKEN_X_INPUT_INDEX);
    mlaPrologContext.tokenX.shape = context.GetRequiredInputShape(TOKEN_X_INPUT_INDEX);
    mlaPrologContext.weightDq.desc = context.GetRequiredInputDesc(WEIGHT_DQ_INPUT_INDEX);
    mlaPrologContext.weightDq.shape = context.GetRequiredInputShape(WEIGHT_DQ_INPUT_INDEX);
    mlaPrologContext.weightUqQr.desc = context.GetRequiredInputDesc(WEIGHT_UQ_QR_INPUT_INDEX);
    mlaPrologContext.weightUqQr.shape = context.GetRequiredInputShape(WEIGHT_UQ_QR_INPUT_INDEX);
    mlaPrologContext.weightUk.desc = context.GetRequiredInputDesc(WEIGHT_UK_INPUT_INDEX);
    mlaPrologContext.weightUk.shape = context.GetRequiredInputShape(WEIGHT_UK_INPUT_INDEX);
    mlaPrologContext.weightDkvKr.desc = context.GetRequiredInputDesc(WEIGHT_DKV_KR_INPUT_INDEX);
    mlaPrologContext.weightDkvKr.shape = context.GetRequiredInputShape(WEIGHT_DKV_KR_INPUT_INDEX);
    mlaPrologContext.rmsnormGammaCq.desc = context.GetRequiredInputDesc(RMSNORM_GAMMA_CQ_INPUT_INDEX);
    mlaPrologContext.rmsnormGammaCq.shape = context.GetRequiredInputShape(RMSNORM_GAMMA_CQ_INPUT_INDEX);
    mlaPrologContext.rmsnormGammaCkv.desc = context.GetRequiredInputDesc(RMS_NORM_GAMMA_CKV_INPUT_INDEX);
    mlaPrologContext.rmsnormGammaCkv.shape = context.GetRequiredInputShape(RMS_NORM_GAMMA_CKV_INPUT_INDEX);
    // V3: rope is optional (null when do_rope=false). Non-V3 keeps required.
    if (std::strncmp(mlaPrologContext.opType, V3_OP_NAME, OP_NAME_LEN) == 0) {
        mlaPrologContext.ropeSin.desc = context.GetOptionalInputDesc(ROPE_SIN_INPUT_INDEX);
        mlaPrologContext.ropeSin.shape = context.GetOptionalInputShape(ROPE_SIN_INPUT_INDEX);
        mlaPrologContext.ropeCos.desc = context.GetOptionalInputDesc(ROPE_COS_INPUT_INDEX);
        mlaPrologContext.ropeCos.shape = context.GetOptionalInputShape(ROPE_COS_INPUT_INDEX);
    } else {
        mlaPrologContext.ropeSin.desc = context.GetRequiredInputDesc(ROPE_SIN_INPUT_INDEX);
        mlaPrologContext.ropeSin.shape = context.GetRequiredInputShape(ROPE_SIN_INPUT_INDEX);
        mlaPrologContext.ropeCos.desc = context.GetRequiredInputDesc(ROPE_COS_INPUT_INDEX);
        mlaPrologContext.ropeCos.shape = context.GetRequiredInputShape(ROPE_COS_INPUT_INDEX);
    }

    if (std::strncmp(mlaPrologContext.opType, V3_OP_NAME, OP_NAME_LEN) == 0) {
        mlaPrologContext.kvCache.desc = context.GetRequiredInputDesc(KV_CACHE_INPUT_INDEX_V3);
        mlaPrologContext.kvCache.shape = context.GetRequiredInputShape(KV_CACHE_INPUT_INDEX_V3);
        mlaPrologContext.krCache.desc = context.GetRequiredInputDesc(KR_CACHE_INPUT_INDEX_V3);
        mlaPrologContext.krCache.shape = context.GetRequiredInputShape(KR_CACHE_INPUT_INDEX_V3);
    } else {
        mlaPrologContext.cacheIndex.desc = context.GetRequiredInputDesc(CACHE_INDEX_INPUT_INDEX);
        mlaPrologContext.cacheIndex.shape = context.GetRequiredInputShape(CACHE_INDEX_INPUT_INDEX);
        mlaPrologContext.kvCache.desc = context.GetRequiredInputDesc(KV_CACHE_INPUT_INDEX);
        mlaPrologContext.kvCache.shape = context.GetRequiredInputShape(KV_CACHE_INPUT_INDEX);
        mlaPrologContext.krCache.desc = context.GetRequiredInputDesc(KR_CACHE_INPUT_INDEX);
        mlaPrologContext.krCache.shape = context.GetRequiredInputShape(KR_CACHE_INPUT_INDEX);
    }

    mlaPrologContext.query.desc = context.GetOutputDesc(QUERY_OUTPUT_INDEX);
    mlaPrologContext.query.shape = context.GetOutputShape(QUERY_OUTPUT_INDEX);
    mlaPrologContext.queryRope.desc = context.GetOutputDesc(QUERY_ROPE_OUTPUT_INDEX);
    mlaPrologContext.queryRope.shape = context.GetOutputShape(QUERY_ROPE_OUTPUT_INDEX);
    mlaPrologContext.kvCacheOut.desc = context.GetOutputDesc(KV_CACHE_OUT_OUTPUT_INDEX);
    mlaPrologContext.kvCacheOut.shape = context.GetOutputShape(KV_CACHE_OUT_OUTPUT_INDEX);
    mlaPrologContext.krCacheOut.desc = context.GetOutputDesc(KR_CACHE_OUT_OUTPUT_INDEX);
    mlaPrologContext.krCacheOut.shape = context.GetOutputShape(KR_CACHE_OUT_OUTPUT_INDEX);
}

void MlaPrologTiling::ConvertOptionalParams(gert::TilingContext &context, MlaPrologContext &mlaPrologContext)
{
    if (std::strncmp(mlaPrologContext.opType, V3_OP_NAME, OP_NAME_LEN) == 0) {
        mlaPrologContext.cacheIndex.desc = context.GetRequiredInputDesc(CACHE_INDEX_INPUT_INDEX_V3);
        mlaPrologContext.cacheIndex.shape = context.GetRequiredInputShape(CACHE_INDEX_INPUT_INDEX_V3);
    }
    mlaPrologContext.dequantScaleX.desc = context.GetOptionalInputDesc(DEQUANT_SCALE_X_INDEX);
    mlaPrologContext.dequantScaleX.shape = context.GetOptionalInputShape(DEQUANT_SCALE_X_INDEX);
    mlaPrologContext.dequantScaleWDq.desc = context.GetOptionalInputDesc(DEQUANT_SCALE_W_DQ_INDEX);
    mlaPrologContext.dequantScaleWDq.shape = context.GetOptionalInputShape(DEQUANT_SCALE_W_DQ_INDEX);
    mlaPrologContext.dequantScaleWUqQr.desc = context.GetOptionalInputDesc(DEQUANT_SCALE_W_UQ_QR_INDEX);
    mlaPrologContext.dequantScaleWUqQr.shape = context.GetOptionalInputShape(DEQUANT_SCALE_W_UQ_QR_INDEX);
    mlaPrologContext.dequantScaleWDkvKr.desc = context.GetOptionalInputDesc(DEQUANT_SCALE_W_DKV_KR_INDEX);
    mlaPrologContext.dequantScaleWDkvKr.shape = context.GetOptionalInputShape(DEQUANT_SCALE_W_DKV_KR_INDEX);
    mlaPrologContext.quantScaleCkv.desc = context.GetOptionalInputDesc(QUANT_SCALE_CKV_INDEX);
    mlaPrologContext.quantScaleCkv.shape = context.GetOptionalInputShape(QUANT_SCALE_CKV_INDEX);
    mlaPrologContext.quantScaleCkr.desc = context.GetOptionalInputDesc(QUANT_SCALE_CKR_INDEX);
    mlaPrologContext.quantScaleCkr.shape = context.GetOptionalInputShape(QUANT_SCALE_CKR_INDEX);
    mlaPrologContext.smoothScalesCq.desc = context.GetOptionalInputDesc(SMOOTH_SCALES_CQ_INDEX);
    mlaPrologContext.smoothScalesCq.shape = context.GetOptionalInputShape(SMOOTH_SCALES_CQ_INDEX);
    mlaPrologContext.actualSeqLen.desc = context.GetOptionalInputDesc(ACTUAL_SEQ_LEN_INDEX);
    mlaPrologContext.actualSeqLen.shape = context.GetOptionalInputShape(ACTUAL_SEQ_LEN_INDEX);
    mlaPrologContext.kNopeClipAlpha.desc = context.GetOptionalInputDesc(K_NOPE_CLIP_ALPHA_INDEX);
    mlaPrologContext.kNopeClipAlpha.shape = context.GetOptionalInputShape(K_NOPE_CLIP_ALPHA_INDEX);

    // only v1 does not support dequantScaleQNope
    if (std::strncmp(mlaPrologContext.opType, V1_OP_NAME, OP_NAME_LEN) == 0) {
        mlaPrologContext.dequantScaleQNope.desc = nullptr;
        mlaPrologContext.dequantScaleQNope.shape = nullptr;
    } else {
        mlaPrologContext.dequantScaleQNope.desc = context.GetOutputDesc(DEQUANT_SCALE_Q_NOPE_OUTPUT_INDEX);
        mlaPrologContext.dequantScaleQNope.shape = context.GetOutputShape(DEQUANT_SCALE_Q_NOPE_OUTPUT_INDEX);
    }

    if (std::strncmp(mlaPrologContext.opType, V3_OP_NAME, OP_NAME_LEN) == 0) {
        mlaPrologContext.queryNorm.desc = context.GetOutputDesc(QUERY_NORM_OUTPUT_INDEX);
        mlaPrologContext.queryNorm.shape = context.GetOutputShape(QUERY_NORM_OUTPUT_INDEX);
        mlaPrologContext.dequantScaleQNorm.desc = context.GetOutputDesc(DEQUANT_SCALE_Q_NORM_OUTPUT_INDEX);
        mlaPrologContext.dequantScaleQNorm.shape = context.GetOutputShape(DEQUANT_SCALE_Q_NORM_OUTPUT_INDEX);
    } else {
        mlaPrologContext.queryNorm.desc = nullptr;
        mlaPrologContext.queryNorm.shape = nullptr;
        mlaPrologContext.dequantScaleQNorm.desc = nullptr;
        mlaPrologContext.dequantScaleQNorm.shape = nullptr;
    }
}

MLA_EXTERN_C ge::graphStatus TilingMlaProlog(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR(V1_OP_NAME, "Context is nullptr."),
                return ge::GRAPH_FAILED);

    MlaPrologContext mlaPrologContext{};
    OP_CHECK_IF(MlaPrologTiling::ConvertContext(*context, mlaPrologContext) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                                       "Error occurred while converting tilingContext to MlaProlog context."),
                return ge::GRAPH_FAILED);

    MlaPrologTiling mlaPrologTiling;
    MlaPrologTilingData *tilingData = context->GetTilingData<MlaPrologTilingData>();
    OP_CHECK_IF(tilingData == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "TilingData is nullptr."),
                return ge::GRAPH_FAILED);
    if (mlaPrologTiling.RunBigKernelTiling(mlaPrologContext, tilingData) == ge::SUCCESS) {
        context->SetTilingKey(mlaPrologContext.tilingKey);
        context->SetBlockDim(mlaPrologContext.blockDim);
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}
} // namespace optiling
