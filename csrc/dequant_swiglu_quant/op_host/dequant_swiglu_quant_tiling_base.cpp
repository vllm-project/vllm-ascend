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
 * \file dequant_swiglu_quant_tiling_base.cpp
 * \brief
 */

#include <cmath>
#include <cstdint>
#include "tiling/tiling_api.h"
#include "swi_glu_tiling.h"
#include "../tiling_base/tiling_util.h"
#include "dequant_swiglu_quant_tiling.h"
#include "../tiling_base/tiling_templates_registry.h"

#define CHECK_FAIL(cont, cond, ...)                      \
    do {                                                 \
        if (cond) {                                      \
            OP_LOGE(cont->GetNodeName(), ##__VA_ARGS__); \
            return ge::GRAPH_FAILED;                     \
        }                                                \
    } while (0)

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;
constexpr uint32_t UB_RESERVED_BUFF = 0;          // reserve 0k
constexpr uint32_t PACK_UINT_IN_CACHE_512B = 512; // pack unit in cache 512B
constexpr uint32_t ALIGN_UINT_IN_CACHE_32B = 32;  // align unit in cache 32B
constexpr uint32_t ALIGN_UINT_IN_CACHE_64B = 64;  // align unit in cache 64B
constexpr uint32_t ALIGN_TYPE_INT32 = 8;          // int32 对齐32字节
constexpr uint32_t DEFAULT_BUFFER_NUM = 2;
constexpr uint32_t MAX_BLOCK_COUNT = 4095; // datacopy指令包含的连续传输数据块的最大个数
constexpr uint32_t MAX_BLOCK_LEN = 2097120; // 65535 * 32 datacopy指令每个连续传输数据块的最长长度为65535，单位为32bytes
constexpr uint32_t MAX_UINT32 = 4294967295;
constexpr uint32_t MAX_CORE_NUMBER = 64;
constexpr uint16_t DISCONTINE_COPY_MAX_BLOCKCNT = 4095;  // 非连续拷贝，blockCount最大值,AscendC接口限制
constexpr uint16_t DISCONTINE_COPY_MAX_BLOCKLEN = 65535; // 非连续拷贝，blockLen最大值,AscendC接口限制
constexpr uint16_t DISCONTINE_COPY_MAX_STRIDE = 65535; // 非连续拷贝，srcStride/dstStride最大值,AscendC接口限制

static const uint32_t DYNAMIC_BF16_TBUF_NUM_HALF = 11;
static const uint32_t DYNAMIC_BF16_INT16_TBUF_NUM_HALF = 6;
static const uint32_t STATIC_BF16_TBUF_NUM_HALF = 12;
static const uint32_t STATIC_BF16_INT16_TBUF_NUM_HALF = 7;
static const uint32_t DYNAMIC_INT16_TBUF_NUM_HALF = 2;

static const size_t INDEX_IN_WEIGHT_SCALE = 1;
static const size_t INDEX_IN_ACTIVATE_SCALE = 2;
static const size_t INDEX_IN_BIAS = 3;
static const size_t INDEX_IN_QUANT_SCALE = 4;
static const size_t INDEX_IN_QUANT_OFFSET = 5;
static const size_t NUMBER_OF_INPUT_SIZE = 10;
static const size_t USER_WORKSPACE = 16777216; // 16 * 1024 * 1024
constexpr uint32_t PERFORMANCE_COL_LEN = 1536;
constexpr uint32_t PERFORMANCE_ROW_LEN = 128;
constexpr uint32_t MIN_CORE = 12;
const int64_t DYNAMIC_INT_X_FLOAT32_BIAS_QUANT_D_PERFORMANCE = 30013;

// Tiling优选参数
struct GluSingleTilingOptParam {
    // Maximum amount of data that can be transferred by an operator UB at a time. Unit:element
    uint32_t maxTileLen = 0;
    uint32_t optBaseRowLen = 0;   // 最优的BaseRowLen
    uint32_t optBaseColLen = 0;   // 最优的BaseColLen
    uint64_t optTotalTileNum = 0; // 最优的分割后的数据块数量
    uint64_t optBaseSize = 0; // 最优的分割后的base shape数据块的大小， optBaseRowLen*optBaseColLen, Unit:element
    uint64_t optBaseTileNum = 0; // 最优的分割后的base shape数据块数量，不包含尾块

    uint32_t totalUsedCoreNum = 0; // 最终实际使用的核数
    uint64_t tileNumPerCore = 0;   // 每个核需要处理的TileNum，如果不均匀，按照多的计算
};

class DequantSwigluQuantTiling : public TilingBaseClass {
public:
    explicit DequantSwigluQuantTiling(gert::TilingContext* cont) : TilingBaseClass(cont)
    {
        Reset();
    }
    ~DequantSwigluQuantTiling() override = default;

    void Reset(gert::TilingContext* cont) override
    {
        TilingBaseClass::Reset(cont);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        if (Ops::NN::OpTiling::IsRegbaseSocVersion(context_)) {
            return false;
        }
        auto shapeGroupIndex = context_->GetOptionalInputShape(6);
        if (shapeGroupIndex == nullptr) {
            return true;
        }
        return false;
    }

    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;
    void Reset();

private:
    void ShowTilingData();

    ge::graphStatus checkInputShape(gert::TilingContext* context, ge::DataType xDataType);

    ge::graphStatus checkWeightBiasActivate(gert::TilingContext* context);

    ge::graphStatus SetTotalShape(gert::TilingContext* cont, const gert::Shape& inShape);

    bool SetAttr(const gert::RuntimeAttrs* attrs);

    bool CalcTiling(const uint32_t totalCores, const uint64_t ubSize, const platform_ascendc::SocVersion socVersion_);

    bool CalcOptTiling(const uint64_t ubSize, const int32_t dtype, GluSingleTilingOptParam& optTiling);

    bool CalcUbMaxTileLen(uint64_t ubSize, int32_t dtype, GluSingleTilingOptParam& optTiling);

    bool GetBufferNumAndDataLenPerUB(uint64_t ubSize, int32_t dtype, uint64_t& dataLenPerUB);

    bool CalcOptBaseShape(GluSingleTilingOptParam& optTiling, int32_t dtype);

    uint32_t getBaseColLenUpBound(GluSingleTilingOptParam& optTiling);

    void SaveOptBaseShape(uint32_t baseRowLen_, uint32_t baseColLen_, GluSingleTilingOptParam& optTiling);

    int64_t getTilingKeyDynamic(
        const int32_t inputDtype, const ge::DataType biasType, const int64_t scaleSize) const;

    bool isPerformanceBranch();

    int64_t getTilingKeyStatic(
        const int32_t inputDtype, const ge::DataType biasType, const int64_t scaleSize) const;

    ge::graphStatus GetShapeAttrsInfoInner();

    uint32_t inputDTypeLen = 2;
    uint32_t activateLeft = 0; // false <-> 0: activate right
    int32_t quantMode = 0;
    uint32_t maxTileLen = 0;
    uint32_t optBaseRowLen = 0;   // 最优的BaseRowLen
    uint32_t optBaseColLen = 0;   // 最优的BaseColLen
    uint64_t optTotalTileNum = 0; // 最优的分割后的数据块数量
    uint64_t optBaseSize = 0; // 最优的分割后的base shape数据块的大小， optBaseRowLen*optBaseColLen, Unit:element
    uint64_t optBaseTileNum = 0; // 最优的分割后的base shape数据块数量，不包含尾块
    uint32_t ubMinBlockLen = 0;
    uint32_t cacheLineLen = 0;
    uint32_t alignPackLen = 0;
    uint32_t totalAvailableCore = 0;
    uint32_t totalUsedCoreNum_ = 0;
    uint32_t totalUsedCoreNum = 0;
    uint32_t totalCore = 0;
    ge::DataType xInputDataType;

    bool isPerfBranch = false;

    ge::DataType biasDataType = ge::DT_FLOAT;
    uint64_t quantScaleShapeSize = 0;
    platform_ascendc::SocVersion curShortSocName_;

    const char* opName = "";
    SwiGluTilingData tilingData;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
};

void DequantSwigluQuantTiling::Reset()
{
    opName = nullptr;
    return;
}

ge::graphStatus DequantSwigluQuantTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(opName, "fail to get platform info"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    curShortSocName_ = ascendcPlatform.GetSocVersion();
    totalCore = ascendcPlatform.GetCoreNumAiv();
    aicoreParams_.numBlocks = totalCore;
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    aicoreParams_.ubSize = ubSizePlatForm;
    socVersion = ascendcPlatform.GetSocVersion();
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus DequantSwigluQuantTiling::SetTotalShape(gert::TilingContext* cont, const gert::Shape& inShape)
{
    int64_t shapeBefore = 1;
    int64_t shapeAfter = 1;
    int64_t dimNum = inShape.GetDimNum();
    CHECK_FAIL(cont, dimNum <= 1, "The shape dim of x can not be less than 2");

    int64_t splitDim = dimNum - 1; // inDim default -1
    for (int64_t i = 0; i < splitDim; i++) {
        shapeBefore *= inShape.GetDim(i);
    }
    shapeAfter = inShape.GetDim(splitDim);
    // 如果shape不是2的倍数,返回

    CHECK_FAIL(cont, shapeAfter % 2 != 0, "The shape dim of x dim must be even number");

    tilingData.set_rowLen(shapeBefore);
    // colLen为原shape除以2
    tilingData.set_colLen(shapeAfter / 2);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantTiling::checkWeightBiasActivate(gert::TilingContext* context)
{
    auto biasShapeShapePtr = context->GetOptionalInputShape(3);
    if (biasShapeShapePtr != nullptr) {
        auto biasInputDesc = context->GetOptionalInputDesc(3);
        OP_CHECK_NULL_WITH_CONTEXT(context, biasInputDesc);
        biasDataType = biasInputDesc->GetDataType();

        bool checkBiasRes = biasDataType != ge::DT_INT32 && biasDataType != ge::DT_FLOAT &&
                            biasDataType != ge::DT_FLOAT16 && biasDataType != ge::DT_BF16;
        OP_CHECK_IF(checkBiasRes,
            OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "bias",
                ge::TypeUtils::DataTypeToSerialString(biasDataType).c_str(), "int32, float, fp16 or bf16"),
            return ge::GRAPH_FAILED);

        uint64_t biasShapeSize = biasShapeShapePtr->GetStorageShape().GetShapeSize();
        OP_CHECK_IF(biasShapeSize != tilingData.get_colLen() * 2,
            OP_LOGE_FOR_INVALID_SHAPESIZE(context->GetNodeName(), "bias",
                std::to_string(biasShapeSize).c_str(),
                (std::to_string(tilingData.get_colLen() * 2)).c_str()),
            return ge::GRAPH_FAILED);
    }
    tilingData.set_biasIsEmpty(biasShapeShapePtr == nullptr);
    // int32时 weight_scale为必选项
    auto weightScaleShapePtr = context->GetOptionalInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScaleShapePtr);

    auto weightScaleInputDesc = context->GetOptionalInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScaleInputDesc);
    ge::DataType weightScaleDataType = weightScaleInputDesc->GetDataType();
    OP_CHECK_IF(weightScaleDataType != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "weight_scale",
            ge::TypeUtils::DataTypeToSerialString(weightScaleDataType).c_str(), "float32"),
        return ge::GRAPH_FAILED);

    uint64_t weightScaleShapeSize = weightScaleShapePtr->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(weightScaleShapeSize != tilingData.get_colLen() * 2,
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(context->GetNodeName(), "weight_scale",
            std::to_string(weightScaleShapeSize).c_str(),
            ("The shapesize of the weight scale is not equal to the last dimension of the xshape "
            + std::to_string(tilingData.get_colLen() * 2)).c_str()),
        return ge::GRAPH_FAILED);

    // int32时 activate_scale为可选项
    auto activateScaleShapePtr = context->GetOptionalInputShape(2);
    if (activateScaleShapePtr != nullptr) {
        auto activateScaleInputDesc = context->GetOptionalInputDesc(2);
        OP_CHECK_NULL_WITH_CONTEXT(context, activateScaleInputDesc);
        ge::DataType activateScaleDataType = activateScaleInputDesc->GetDataType();
        OP_CHECK_IF(activateScaleDataType != ge::DT_FLOAT,
            OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "activation_scale",
                ge::TypeUtils::DataTypeToSerialString(activateScaleDataType).c_str(), "float32"),
            return ge::GRAPH_FAILED);

        uint64_t activateScaleShapeSize = activateScaleShapePtr->GetStorageShape().GetShapeSize();
        OP_CHECK_IF(activateScaleShapeSize != tilingData.get_rowLen(),
            OP_LOGE_FOR_INVALID_SHAPESIZE(context->GetNodeName(), "activation_scale",
                std::to_string(activateScaleShapeSize).c_str(),
                ("equal to " + std::to_string(tilingData.get_rowLen())).c_str()),
            return ge::GRAPH_FAILED);
    }
    tilingData.set_activateScaleIsEmpty(activateScaleShapePtr == nullptr);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantTiling::checkInputShape(gert::TilingContext* context, ge::DataType xDataType)
{
    if (xDataType == ge::DT_INT32) {
        if (checkWeightBiasActivate(context) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    // quant_scale
    auto quantScaleShapePtr = context->GetOptionalInputShape(4); // 3: bias idx
    if (quantScaleShapePtr == nullptr) {
        tilingData.set_quantScaleIsEmpty(1);
        return ge::GRAPH_SUCCESS;
    }
    auto quantScaleInputDesc = context->GetOptionalInputDesc(4);
    OP_CHECK_NULL_WITH_CONTEXT(context, quantScaleInputDesc);
    ge::DataType quantScaleDataType = quantScaleInputDesc->GetDataType();
    OP_CHECK_IF(quantScaleDataType != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "quant_scale",
            ge::TypeUtils::DataTypeToSerialString(quantScaleDataType).c_str(), "float32"),
        return ge::GRAPH_FAILED);
    quantScaleShapeSize = quantScaleShapePtr->GetStorageShape().GetShapeSize();
    bool checkQuantScaleSize = (quantScaleShapeSize != tilingData.get_colLen()) && (quantScaleShapeSize != 1);
    OP_CHECK_IF(checkQuantScaleSize,
        OP_LOGE_FOR_INVALID_SHAPESIZE(context->GetNodeName(), "quant_scale",
            std::to_string(quantScaleShapeSize).c_str(),
            (std::to_string(tilingData.get_colLen()) + " or 1").c_str()),
        return ge::GRAPH_FAILED);
    if (quantMode == 0) {
        auto quantOffsetShapePtr = context->GetOptionalInputShape(5);
        auto quantOffsetInputDesc = context->GetOptionalInputDesc(5);
        OP_CHECK_NULL_WITH_CONTEXT(context, quantOffsetInputDesc);
        ge::DataType quantOffsetDataType = quantOffsetInputDesc->GetDataType();
        OP_CHECK_IF(quantOffsetDataType != ge::DT_FLOAT,
            OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "quant_offset",
                ge::TypeUtils::DataTypeToSerialString(quantOffsetDataType).c_str(), "float32"),
            return ge::GRAPH_FAILED);
        uint64_t quantOffsetShapeSize = quantOffsetShapePtr->GetStorageShape().GetShapeSize();
        bool checkQuantOffsetSize = (quantOffsetShapeSize != tilingData.get_colLen()) && (quantOffsetShapeSize != 1);
        OP_CHECK_IF(checkQuantOffsetSize,
            OP_LOGE_FOR_INVALID_SHAPESIZE(context->GetNodeName(), "quant_offset",
                std::to_string(quantOffsetShapeSize).c_str(),
                (std::to_string(tilingData.get_colLen()) + " or 1").c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

bool DequantSwigluQuantTiling::SetAttr(const gert::RuntimeAttrs* attrs)
{
    auto isActivateLeftAttr = *(attrs->GetBool(0));
    auto str = attrs->GetStr(1);
    std::string quantModeAttr{str};
    std::transform(quantModeAttr.begin(), quantModeAttr.end(), quantModeAttr.begin(), ::tolower);

    if ((quantModeAttr != "static") && (quantModeAttr != "dynamic")) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context_->GetNodeName(), "quant_mode",
            quantModeAttr.c_str(),
            "quant_mode should be static or dynamic with case insensitive");
        return false;
    }
    activateLeft = (isActivateLeftAttr ? 1 : 0);
    quantMode = ((quantModeAttr == "static") ? 0 : 1);
    tilingData.set_activateLeft(activateLeft);
    return true;
}

bool DequantSwigluQuantTiling::GetBufferNumAndDataLenPerUB(uint64_t ubSize, int32_t dtype, uint64_t& dataLenPerUB)
{
    uint32_t singleDataSize = 1;
    if (quantMode == 1) {
        if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
            singleDataSize = DYNAMIC_BF16_INT16_TBUF_NUM_HALF * static_cast<uint32_t>(sizeof(float)) +
                             static_cast<uint32_t>(sizeof(int8_t));
        } else if (dtype == ge::DT_INT32) {
            if ((biasDataType == ge::DT_INT32 || biasDataType == ge::DT_FLOAT)) {
                singleDataSize = DYNAMIC_BF16_TBUF_NUM_HALF * static_cast<uint32_t>(sizeof(float)) +
                                 static_cast<uint32_t>(sizeof(int8_t));
            } else {
                singleDataSize = DYNAMIC_BF16_TBUF_NUM_HALF * static_cast<uint32_t>(sizeof(float)) +
                                 DYNAMIC_INT16_TBUF_NUM_HALF * static_cast<uint32_t>(sizeof(int16_t)) +
                                 static_cast<uint32_t>(sizeof(int8_t));
            }
        }
    }
    if (quantMode == 0) {
        if (dtype == ge::DT_INT32) {
            if ((biasDataType == ge::DT_INT32 || biasDataType == ge::DT_FLOAT)) {
                singleDataSize = STATIC_BF16_TBUF_NUM_HALF * static_cast<uint32_t>(sizeof(float)) +
                                 static_cast<uint32_t>(sizeof(int8_t)); /* 11 -> float 块数量 */
            } else {
                singleDataSize = STATIC_BF16_TBUF_NUM_HALF * static_cast<uint32_t>(sizeof(float)) +
                                 DYNAMIC_INT16_TBUF_NUM_HALF * static_cast<uint32_t>(sizeof(int16_t)) +
                                 static_cast<uint32_t>(sizeof(int8_t)); /* 11 -> float 块数量 */
            }
        } else if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
            singleDataSize = STATIC_BF16_INT16_TBUF_NUM_HALF * static_cast<uint32_t>(sizeof(float)) +
                             static_cast<uint32_t>(sizeof(int8_t));
        }
    }
    dataLenPerUB = ubSize / singleDataSize;
    return true;
}

bool DequantSwigluQuantTiling::CalcUbMaxTileLen(uint64_t ubSize, int32_t dtype, GluSingleTilingOptParam& optTiling)
{
    // get buffernum and maxTileLen
    uint64_t maxTileLenPerUB = 1;
    if (!GetBufferNumAndDataLenPerUB(ubSize, dtype, maxTileLenPerUB)) {
        OP_LOGE("DequantSwigluQuant", "CalcTiling Get maxTileLenPerUB %lu failed", maxTileLenPerUB);
        return false;
    }
    optTiling.maxTileLen = AlignDown<uint64_t>(maxTileLenPerUB, ALIGN_UINT_IN_CACHE_32B); // 32个元素对齐
    OP_LOGI("DequantSwigluQuant", "CalcTiling ubSize:%lu, maxTileLenPerUB:%u", ubSize, optTiling.maxTileLen);
    return true;
}

uint32_t DequantSwigluQuantTiling::getBaseColLenUpBound(GluSingleTilingOptParam& optTiling)
{
    uint32_t upBound = std::min(tilingData.get_colLen(), static_cast<uint64_t>(optTiling.maxTileLen));
    if (tilingData.get_is32BAligned() == 1) {
        upBound = std::min(upBound, static_cast<uint32_t>(DISCONTINE_COPY_MAX_BLOCKLEN));
    } else {
        upBound = std::min(upBound, static_cast<uint32_t>(DISCONTINE_COPY_MAX_BLOCKLEN / sizeof(xInputDataType)));
    }

    if (upBound < tilingData.get_colLen() && upBound > cacheLineLen) {
        // 该种场景，每一个colLen至少被切割成2块，需要保证baseColLen为512B整数倍才高效
        return AlignDown<uint32_t>(upBound, cacheLineLen);
    } else {
        return upBound;
    }
}

void DequantSwigluQuantTiling::SaveOptBaseShape(
    uint32_t baseRowLen_, uint32_t baseColLen_, GluSingleTilingOptParam& optTiling)
{
    uint64_t totalTileNum =
        std::min(static_cast<uint64_t>(tilingData.get_rowLen()), static_cast<uint64_t>(totalAvailableCore));
    uint64_t baseSize = static_cast<uint64_t>(baseRowLen_ * baseColLen_);
    if (static_cast<int32_t>(baseRowLen_) == 0 || static_cast<int32_t>(baseColLen_) == 0) {
        OP_LOGI("SaveOptBaseShape", "baseRowLen_:%u or baseColLen:%u is zero.", baseRowLen_, baseColLen_);
        return;
    }
    uint64_t baseTileNum = (baseRowLen_ == 0 ? 0 : (tilingData.get_rowLen() / baseRowLen_)) *
                           (baseColLen_ == 0 ? 0 : (tilingData.get_colLen() / baseColLen_));
    totalUsedCoreNum_ = std::min(totalTileNum, static_cast<uint64_t>(totalAvailableCore));
    if(tilingData.get_colLen() < PERFORMANCE_COL_LEN
    && tilingData.get_rowLen() < PERFORMANCE_ROW_LEN) {
        totalUsedCoreNum_ = std::min(totalUsedCoreNum_, static_cast<uint32_t>(MIN_CORE));
    }
    optTiling.optBaseRowLen = baseRowLen_;
    optTiling.optBaseColLen = baseColLen_;
    optTiling.optTotalTileNum = totalTileNum;
    optTiling.optBaseSize = baseSize;
    optTiling.optBaseTileNum = baseTileNum;
    optTiling.totalUsedCoreNum = totalUsedCoreNum_;
    optTiling.tileNumPerCore = DivCeil<uint64_t>(totalTileNum, totalUsedCoreNum_);
}

bool DequantSwigluQuantTiling::CalcOptBaseShape(GluSingleTilingOptParam& optTiling, int32_t dtype)
{
    uint32_t baseColLen_ = getBaseColLenUpBound(optTiling);
    uint32_t baseRowlen_ = 1;
    if ((quantMode == 1) && (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16)) {
        baseRowlen_ = std::min(
            optTiling.maxTileLen / AlignUp<uint32_t>(baseColLen_, ALIGN_UINT_IN_CACHE_32B),
            static_cast<uint32_t>(tilingData.get_rowLen()));
        baseRowlen_ = std::min(DivCeil<uint32_t>(tilingData.get_rowLen(), totalAvailableCore), baseRowlen_);
    }
    SaveOptBaseShape(baseRowlen_, baseColLen_, optTiling);
    return true;
}

bool DequantSwigluQuantTiling::CalcOptTiling(
    const uint64_t ubSize, const int32_t dtype, GluSingleTilingOptParam& optTiling)
{
    // 计算maxTilingLen
    if (!CalcUbMaxTileLen(ubSize, dtype, optTiling)) {
        return false;
    }
    // 计算最优的base块形状
    if (!CalcOptBaseShape(optTiling, dtype)) {
        return false;
    }
    return true;
}

bool DequantSwigluQuantTiling::CalcTiling(
    const uint32_t totalCores, const uint64_t ubSize, const platform_ascendc::SocVersion socVersion_)
{
    totalAvailableCore = totalCores;
    if (!GetLengthByType(xInputDataType, inputDTypeLen)) {
        OP_LOGI("DequantSwigluQuant", "CalcTiling Unsupported input data type %d", xInputDataType);
        return false;
    }
    ubMinBlockLen = ALIGN_UINT_IN_CACHE_32B / inputDTypeLen; // min block size
    cacheLineLen = PACK_UINT_IN_CACHE_512B / inputDTypeLen;  // bandwidth max efficiency
    alignPackLen = cacheLineLen;                             // 默认512对齐，策略可调整
    OP_LOGI(
        "DequantSwigluQuant", "CalcTiling GetLengthByType:%u ubMinBlockLen:%u cacheLineLen:%u alignPackLen:%u",
        inputDTypeLen, ubMinBlockLen, cacheLineLen, alignPackLen);
    // Is 32-byte aligned for split colLen?
    tilingData.set_is32BAligned(tilingData.get_colLen() % ubMinBlockLen == 0);
    // 310p not support Non-64B
    const uint32_t blockSizeOf64B = ALIGN_UINT_IN_CACHE_64B / inputDTypeLen;
    if (((socVersion_ == platform_ascendc::SocVersion::ASCEND310P)) &&
        (tilingData.get_colLen() % blockSizeOf64B != 0)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x",
            std::to_string(tilingData.get_colLen()).c_str(),
            "colLen (the last dimension of x) must be 64B aligned on ASCEND310P");
        return false;
    }
    GluSingleTilingOptParam optTilingDb;
    if (!CalcOptTiling(ubSize, xInputDataType, optTilingDb)) {
        return false;
    }
    const GluSingleTilingOptParam* const optTiling = &optTilingDb;
    // 记录最优的结果
    tilingData.set_baseRowLen(optTiling->optBaseRowLen);
    tilingData.set_baseColLen(optTiling->optBaseColLen);
    totalUsedCoreNum = optTiling->totalUsedCoreNum;
    tilingData.set_usedCoreNum(totalUsedCoreNum);
    OP_LOGI(
        "DequantSwigluQuant", "CalcTilingRES baseRowLen:%u baseColLen:%u", optTiling->optBaseRowLen,
        optTiling->optBaseColLen);
    return true;
}

ge::graphStatus DequantSwigluQuantTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantTiling::GetShapeAttrsInfoInner()
{
    opName = context_->GetNodeName();
    // 获取输入shape
    auto xShapePtr = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    const gert::Shape xShape = xShapePtr->GetStorageShape();
    auto inputDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    xInputDataType = inputDesc->GetDataType();
    if (SetTotalShape(context_, xShape) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    // 获取输入属性
    const gert::RuntimeAttrs* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    if (!SetAttr(attrs)) {
        return ge::GRAPH_FAILED;
    }

    if (checkInputShape(context_, xInputDataType) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    auto yShapePtr = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    const gert::Shape yShape = yShapePtr->GetStorageShape();

    int32_t dimNum = xShape.GetDimNum();
    if(xShape.GetDimNum() != yShape.GetDimNum()){
        std::string incorrectDims = std::to_string(xShape.GetDimNum()) + " and " + std::to_string(yShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(opName, "x and y",
        incorrectDims.c_str(),
        "The shape of y must be equal to the shape of x");
    }

    if(xShape.GetDim(dimNum - 1) != yShape.GetDim(dimNum - 1) * 2){
         std::string incorrectDims = std::to_string(xShape.GetDimNum()) + " and " + std::to_string(yShape.GetDimNum());
         OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName, "x and y",
        incorrectDims.c_str(),
        "The last dimension of x must be twice the last dimension of y.");
    }

    auto scaleShapePtr = context_->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleShapePtr);
    const gert::Shape scaleShape = scaleShapePtr->GetStorageShape();

    if (static_cast<uint64_t>(scaleShape.GetShapeSize()) != tilingData.get_rowLen()) {
         std::string incorrectSize = std::to_string(static_cast<uint64_t>(scaleShape.GetShapeSize()));
         std::string reason =
             "scale's shapesize must be equal to row length" + std::to_string(tilingData.get_rowLen()) +
             "(row length is total number of elements of x across all dimensions except the last one.)";
         OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(opName, "scale", incorrectSize.c_str(), reason.c_str());
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantTiling::DoOpTiling()
{
    if (GetShapeAttrsInfoInner() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (!CalcTiling(totalCore, aicoreParams_.ubSize, curShortSocName_)) {
        return ge::GRAPH_FAILED;
    }
    isPerfBranch = isPerformanceBranch();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

int64_t DequantSwigluQuantTiling::getTilingKeyStatic(
    const int32_t inputDtype, const ge::DataType biasType, const int64_t scaleSize) const
{
    if (inputDtype != ge::DT_INT32) {
        if (scaleSize == 1) {
            if (inputDtype == ge::DT_FLOAT16) {
                return STATIC_FLOAT16_X;
            } else {
                return STATIC_BFLOAT16_X;
            }
        } else {
            if (inputDtype == ge::DT_FLOAT16) {
                return STATIC_FLOAT16_XD;
            } else {
                return STATIC_BFLOAT16_XD;
            }
        }
    }
    if (scaleSize == 1) {
        if (biasType == ge::DT_INT32) {
            return STATIC_INT_X_INT_BIAS_QUANT_ONE;
        } else if (biasType == ge::DT_FLOAT) {
            return STATIC_INT_X_FLOAT32_BIAS_QUANT_ONE;
        } else if (biasType == ge::DT_FLOAT16) {
            return STATIC_INT_X_FLOAT16_BIAS_QUANT_ONE;
        } else {
            return STATIC_INT_X_BFLOAT16_BIAS_QUANT_ONE;
        }
    } else {
        if (biasType == ge::DT_INT32) {
            return STATIC_INT_X_INT_BIAS_QUANT_D;
        } else if (biasType == ge::DT_FLOAT) {
            return STATIC_INT_X_FLOAT32_BIAS_QUANT_D;
        } else if (biasType == ge::DT_FLOAT16) {
            return STATIC_INT_X_FLOAT16_BIAS_QUANT_D;
        } else {
            return STATIC_INT_X_BFLOAT16_BIAS_QUANT_D;
        }
    }
}

int64_t DequantSwigluQuantTiling::getTilingKeyDynamic(
    const int32_t inputDtype, const ge::DataType biasType, const int64_t scaleSize) const
{
    if (inputDtype != ge::DT_INT32) {
        if (inputDtype == ge::DT_FLOAT16) {
            if (scaleSize == 1) {
                return DYNAMIC_FLOAT16_X;
            } else {
                return DYNAMIC_FLOAT16_XD;
            }
        } else {
            if (scaleSize == 1) {
                return DYNAMIC_BFLOAT16_X;
            } else {
                return DYNAMIC_BFLOAT16_XD;
            }
        }
    }
    if (scaleSize == 1) {
        if (biasType == ge::DT_INT32) {
            return DYNAMIC_INT_X_INT_BIAS_QUANT_ONE;
        } else if (biasType == ge::DT_FLOAT) {
            return DYNAMIC_INT_X_FLOAT32_BIAS_QUANT_ONE;
        } else if (biasType == ge::DT_FLOAT16) {
            return DYNAMIC_INT_X_FLOAT16_BIAS_QUANT_ONE;
        } else {
            return DYNAMIC_INT_X_BFLOAT16_BIAS_QUANT_ONE;
        }
    } else {
        if (biasType == ge::DT_INT32) {
            return DYNAMIC_INT_X_INT_BIAS_QUANT_D;
        } else if (biasType == ge::DT_FLOAT) {
            if(isPerfBranch) {
                return DYNAMIC_INT_X_FLOAT32_BIAS_QUANT_D_PERFORMANCE;
            }
            return DYNAMIC_INT_X_FLOAT32_BIAS_QUANT_D;
        } else if (biasType == ge::DT_FLOAT16) {
            return DYNAMIC_INT_X_FLOAT16_BIAS_QUANT_D;
        } else {
            return DYNAMIC_INT_X_BFLOAT16_BIAS_QUANT_D;
        }
    }
}

bool DequantSwigluQuantTiling::isPerformanceBranch() {
    if(tilingData.get_is32BAligned() == 1
    && tilingData.get_colLen() <= PERFORMANCE_COL_LEN
    && tilingData.get_baseRowLen() == 1
    && tilingData.get_baseColLen() == tilingData.get_colLen()
    && tilingData.get_biasIsEmpty() == 1
    && tilingData.get_activateScaleIsEmpty() == 0) {
        return true;
    }
    return false;
}

uint64_t DequantSwigluQuantTiling::GetTilingKey() const
{
    if (quantMode == 0) { // static
        return getTilingKeyStatic(xInputDataType, biasDataType, quantScaleShapeSize);
    } else { // dynamic
        return getTilingKeyDynamic(xInputDataType, biasDataType, quantScaleShapeSize);
    }
}

ge::graphStatus DequantSwigluQuantTiling::GetWorkspaceSize()
{
    // 计算workspace大小，无需workspace临时空间，不存在多核同步，预留固定大小即可
    workspaceSize_ = USER_WORKSPACE;
    if (quantMode == 1 && (tilingData.get_colLen() > tilingData.get_baseColLen())) {
        workspaceSize_ += (totalUsedCoreNum * tilingData.get_colLen() * sizeof(float));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluQuantTiling::PostTiling()
{
    context_->SetBlockDim(totalCore);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());

    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    context_->SetBlockDim(totalUsedCoreNum);
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("DequantSwigluQuant", DequantSwigluQuantTiling, 1);
} // namespace optiling
