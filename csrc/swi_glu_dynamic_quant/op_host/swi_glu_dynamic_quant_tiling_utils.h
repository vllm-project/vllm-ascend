/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file swi_glu_dynamic_quant_tiling_utils.h
 * \brief
 */


#ifndef SWI_GLU_DYNAMIC_QUANT_TILING_UTILS_H
#define SWI_GLU_DYNAMIC_QUANT_TILING_UTILS_H

#include "swi_glu_dynamic_quant_tiling.h"

namespace optiling {
constexpr uint32_t DQ_INPUT_X_INDEX = 0;
constexpr uint32_t DQ_INPUT_SMOOTH_SCALES_INDEX = 1;
constexpr uint32_t DQ_INPUT_OFFSETS_INDEX = 2;
constexpr uint32_t DQ_INPUT_GROUPIDX_INDEX = 3;
constexpr uint32_t DQ_OUTPUT_Y_INDEX = 0;
constexpr uint32_t DQ_OUTPUT_SCALE_INDEX = 1;

constexpr uint32_t DQ_EVEN_FACTOR = 2;
constexpr uint32_t DQ_CONST_ROUR = 4;
constexpr uint32_t DQ_MAX_EXPERT_NUM = 1024;

constexpr uint32_t DQ_ALIGEN_EIGHT = 8;

constexpr uint32_t DQ_ZERO = 0;
constexpr uint32_t DQ_ONE = 1;
constexpr uint32_t DQ_TWO = 2;
constexpr uint32_t DQ_GROUP_LIST_TYPE_INDEX = 2;
constexpr uint32_t DQ_DST_TYPE_INDEX = 3;
constexpr uint32_t DQ_COMPARE_INT = 255;

constexpr uint32_t DQ_TILING_KEY_VAL_FP16 = 100;
constexpr uint32_t DQ_TILING_KEY_VAL_BF16 = 200;
constexpr uint32_t DQ_TILING_KEY_VAL_FP = 300;

constexpr uint8_t DQ_PER_TENSOR_KEY_VAL = 4;
constexpr uint8_t DQ_PER_CHANEL_KEY_VAL = 5;
constexpr uint8_t DQ_DYNAMIC_KEY_VAL = 6;

const std::string DQ_QUANT_MODE_STATIC = "static";
const std::string DQ_QUANT_MODE_DYNAMIC = "dynamic";
const std::string DQ_QUANT_MODE_DYNAMIC_MSD = "dynamic_msd";


template <typename T>
inline auto DQAlignUp(T num, T div) -> decltype(num)
{
    return (div == 0) ? 0 : (num + div - 1) / div * div;
}

template <typename T>
inline auto DQAlignDown(T num, T div) -> decltype(num)
{
    return (div == 0) ? 0 : num / div * div;
}

template <typename T>
inline auto DQCeilDiv(T num, T div) -> decltype(num)
{
    return div == 0 ? 0 : (num + div - 1) / div;
}

template <typename T>
inline auto DQMin(T num, T div) -> decltype(num)
{
    return num < div ? num : div;
}


inline ge::graphStatus DQCheckInputDtype(const gert::TilingContext *context)
{
    auto xDtype = context->GetInputDesc(DQ_INPUT_X_INDEX)->GetDataType();
    if (xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16 && xDtype != ge::DT_FLOAT) {
        OP_LOGE(context->GetNodeName(), "input x dtype is only support fp16/bf16/fp32.");
        return ge::GRAPH_FAILED;
    }

    auto smoothScalesDesc = context->GetOptionalInputDesc(DQ_INPUT_SMOOTH_SCALES_INDEX);
    if (smoothScalesDesc != nullptr) {
        auto smoothScalesDtype = smoothScalesDesc->GetDataType();
        if (smoothScalesDtype != ge::DataType::DT_FLOAT) {
            OP_LOGE(context->GetNodeName(), "input smoothScales dtype is only support fp32.");
            return ge::GRAPH_FAILED;
        }
    }

    auto offsetsDesc = context->GetOptionalInputDesc(DQ_INPUT_OFFSETS_INDEX);
    if (offsetsDesc != nullptr) {
        auto offsetsDtype = offsetsDesc->GetDataType();
        if (offsetsDtype != ge::DataType::DT_FLOAT) {
            OP_LOGE(context->GetNodeName(), "input offsets dtype is only support fp32.");
            return ge::GRAPH_FAILED;
        }
    }

    auto groupDesc = context->GetOptionalInputDesc(DQ_INPUT_GROUPIDX_INDEX);
    if (groupDesc != nullptr) {
        auto groupDtype = groupDesc->GetDataType();
        if (groupDtype != ge::DataType::DT_INT32) {
            OP_LOGE(context->GetNodeName(), "group dtype is only support int32.");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}


inline ge::graphStatus DQCheckOutputDtype(const gert::TilingContext *context)
{
    auto yDtype = context->GetOutputDesc(DQ_OUTPUT_Y_INDEX)->GetDataType();
    if (yDtype != ge::DataType::DT_INT8 && yDtype != ge::DataType::DT_INT4) {
        OP_LOGE(context->GetNodeName(), "y dtype is support int8 or int4.");
        return ge::GRAPH_FAILED;
    }

    auto scaleDtype = context->GetOutputDesc(DQ_OUTPUT_SCALE_INDEX)->GetDataType();
    if (scaleDtype != ge::DataType::DT_FLOAT) {
        OP_LOGE(context->GetNodeName(), "output scale dtype is only support fp32.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}


inline ge::graphStatus DQCheckAttrs(const gert::TilingContext *context, SwiGluDynamicQuantCompileInfo &compileInfo)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs != nullptr) {
        std::string quantMode = attrs->GetAttrPointer<char>(1);
        if (quantMode != DQ_QUANT_MODE_STATIC && quantMode != DQ_QUANT_MODE_DYNAMIC && quantMode != DQ_QUANT_MODE_DYNAMIC_MSD) {
            OP_LOGE(context->GetNodeName(), "quantMode must be equal to static; dynamic; dynamic_msd.");
            return ge::GRAPH_FAILED;
        }
        compileInfo.curQuantMode = quantMode;

        auto isActivateLeftAttr = *(attrs->GetBool(0));
        compileInfo.activateLeft = (isActivateLeftAttr ? 1 : 0);

        auto groupListTypePtr = attrs->GetInt(DQ_GROUP_LIST_TYPE_INDEX);
        OP_CHECK_IF((groupListTypePtr == nullptr),
            OP_LOGE(context->GetNodeName(), "Get groupListType failed."),
            return ge::GRAPH_FAILED);
        int64_t groupListType = static_cast<int64_t>(*groupListTypePtr);
        OP_CHECK_IF((groupListType != 0 && groupListType != 1),
            OP_LOGE(context->GetNodeName(), "groupListType can only be 0 or 1."),
            return ge::GRAPH_FAILED);
        compileInfo.groupListType = groupListType;

        auto dstTypePtr = attrs->GetInt(DQ_DST_TYPE_INDEX);
        OP_CHECK_IF((dstTypePtr == nullptr),
            OP_LOGE(context->GetNodeName(), "Get dstType failed."),
            return ge::GRAPH_FAILED);
        int64_t dstType = static_cast<int64_t>(*dstTypePtr);
        OP_CHECK_IF((dstType != ge::DT_INT8 && dstType != ge::DT_INT4),
            OP_LOGE(context->GetNodeName(), "dstType can only be 2 or 29."),
            return ge::GRAPH_FAILED);
        compileInfo.dstType = dstType;
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus DQCheckMoeInputShpae(const gert::StorageShape *shape, uint32_t groupLen){
    if (groupLen > DQ_ONE) {
        uint32_t shapeDimFirst = shape->GetStorageShape().GetDim(0);
        if (groupLen != shapeDimFirst) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus DQCheckOpInputShape(const gert::TilingContext *context, SwiGluDynamicQuantCompileInfo &compileInfo)
{
    auto xShape = context->GetInputShape(DQ_INPUT_X_INDEX);
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    if (xDimNum < DQ_TWO) {
        OP_LOGE(context->GetNodeName(), "Input x shape dimension is less than 2.");
        return ge::GRAPH_FAILED;
    }
    int64_t xDimLast = xShape->GetStorageShape().GetDim(xDimNum - 1);
    auto yDtype = context->GetOutputDesc(DQ_OUTPUT_Y_INDEX)->GetDataType();
    if (yDtype == ge::DT_INT4 && xDimLast % static_cast<int32_t>(DQ_CONST_ROUR) != 0) {
            OP_LOGE(context->GetNodeName(), "if y datatype is int4, the last dim(%ld) of x should be divisible by 4.",
                xDimLast);
            return ge::GRAPH_FAILED;
    }
    auto smoothScalesShape = context->GetOptionalInputShape(DQ_INPUT_SMOOTH_SCALES_INDEX);
    auto offsetsShape = context->GetOptionalInputShape(DQ_INPUT_OFFSETS_INDEX);
    uint32_t groupLen = 1;
    int64_t hasGroup = 0;
    if (smoothScalesShape != nullptr) {
        auto scaleDimNum = smoothScalesShape->GetStorageShape().GetDimNum();
        compileInfo.isPerTensor = scaleDimNum == 1 ? true : false;
        auto groupShape = context->GetOptionalInputShape(DQ_INPUT_GROUPIDX_INDEX);
        if (groupShape != nullptr) {
            groupLen = static_cast<uint32_t>(
                groupShape->GetStorageShape().GetDim(groupShape->GetStorageShape().GetDimNum() - 1));
            hasGroup = 1;
            if (groupLen == DQ_ZERO || groupLen > DQ_MAX_EXPERT_NUM) {
                OP_LOGE(context->GetNodeName(), "group length must be more than 0 and less than the maximum value.");
                return ge::GRAPH_FAILED;
            }
        }
        if (DQCheckMoeInputShpae(smoothScalesShape, groupLen) != ge::GRAPH_SUCCESS) {
            OP_LOGE(context->GetNodeName(),
                    "moe expert and smooth_scales first dim is not equal! expert nums is :%u", groupLen);
            return ge::GRAPH_FAILED;
        }
    }
    if (offsetsShape != nullptr) {
        if (DQCheckMoeInputShpae(offsetsShape, groupLen) != ge::GRAPH_SUCCESS) {
            OP_LOGE(context->GetNodeName(),
                    "moe expert and offsets first dim is not equal! expert nums is :%u", groupLen);
            return ge::GRAPH_FAILED;
        }
    }
    compileInfo.groupLength = groupLen;
    compileInfo.hasGroup = hasGroup;
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus DQCheckOpDim(const gert::StorageShape *shape1, const gert::StorageShape *shape2, uint32_t shape1Dim,
    uint32_t shape2Dim)
{
    if (shape1Dim != shape2Dim) {
        return ge::GRAPH_FAILED;
    }
    for (uint32_t i = 0; i < shape1Dim; i++) {
        if (shape1->GetStorageShape().GetDim(i) != shape2->GetStorageShape().GetDim(i)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}


inline ge::graphStatus DQCheckOpOutputShape(const gert::TilingContext *context, const SwiGluDynamicQuantCompileInfo &compileInfo)
{
    auto xShape = context->GetInputShape(DQ_INPUT_X_INDEX);
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    auto yShape = context->GetOutputShape(DQ_OUTPUT_Y_INDEX);
    size_t yDimNum = yShape->GetStorageShape().GetDimNum();
    if (DQCheckOpDim(xShape, yShape, xDimNum - 1, yDimNum - 1) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "x and y shape is inconsistency.");
        return ge::GRAPH_FAILED;
    }
    if (compileInfo.dstType == ge::DT_INT8 && (xShape->GetStorageShape().GetDim(xDimNum - 1) != 2 * yShape->GetStorageShape().GetDim(yDimNum - 1))) {
        OP_LOGE(context->GetNodeName(), "CheckOpOutputShape x and y last dim error.");
        return ge::GRAPH_FAILED;
    }
    auto scaleShape = context->GetOutputShape(DQ_OUTPUT_SCALE_INDEX);
    size_t scaleDimNum = scaleShape->GetStorageShape().GetDimNum();
    if (DQCheckOpDim(xShape, scaleShape, xDimNum - 1, scaleDimNum) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "output_scale shape is wrong, must be same as before x exclude the last dim.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}


inline ge::graphStatus DQCheckOpShape(gert::TilingContext *context, SwiGluDynamicQuantCompileInfo &compileInfo)
{
    if (DQCheckOpInputShape(context, compileInfo) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "input shape check failed.");
        return ge::GRAPH_FAILED;
    }
    if (DQCheckOpOutputShape(context, compileInfo) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "output shape check failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}


inline ge::graphStatus DQCheckOpParams(gert::TilingContext *context, SwiGluDynamicQuantCompileInfo &compileInfo)
{
    if (DQCheckInputDtype(context) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "x or smooth_scales dtype is invalid.");
        return ge::GRAPH_FAILED;
    }
    if (DQCheckOutputDtype(context) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "op output dtype is invalid.");
        return ge::GRAPH_FAILED;
    }
    if (DQCheckAttrs(context, compileInfo) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "op attrs is invalid.");
        return ge::GRAPH_FAILED;
    }
    if (DQCheckOpShape(context, compileInfo) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "input or output shape is invalid.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

inline bool DQSetTotalShape(gert::Shape &inShape, const gert::TilingContext *context, SwiGluDynamicQuantTilingData &tilingData)
{
    int32_t shapeBefore = 1;
    int32_t shapeAfter = 1;
    int32_t dimNum = inShape.GetDimNum();
    int32_t inDim = -1;

    if (inDim < -dimNum) {
        OP_LOGE(context->GetNodeName(), "SetTotalShape Unsupported inDim %d", inDim);
        return false;
    }
    int32_t splitDim = inDim < 0 ? dimNum + inDim : inDim;
    for (int32_t i = 0; i < splitDim; i++) {
        shapeBefore *= inShape.GetDim(i);
    }
    for (int32_t i = splitDim; i < dimNum; i++) {
        shapeAfter *= inShape.GetDim(i);
    }
    if (shapeAfter % DQ_EVEN_FACTOR != 0) {
        OP_LOGE(context->GetNodeName(), "SetTotalShape Unsupported inDim %d, shapeAfter %d %% 2 != 0 ", inDim,
            shapeAfter);
        return false;
    }
    if (shapeAfter == 0) {
        OP_LOGE(context->GetNodeName(), "SetTotalShape Unsupported shapeAfter %d == 0 ", shapeAfter);
        return false;
    }
    tilingData.set_rowLen(shapeBefore);
    tilingData.set_colLen(shapeAfter / DQ_EVEN_FACTOR);
    return true;
}

inline bool DQCalculateMaxUbSizePerRow(const gert::TilingContext *context, SwiGluDynamicQuantCompileInfo &compileInfo,
    SwiGluDynamicQuantTilingParam &tilingParam, SwiGluDynamicQuantTilingData &tilingData)
{
    uint32_t colLen = tilingData.get_colLen();
    uint32_t alignedColLen = DQAlignUp<uint32_t>(colLen, compileInfo.block_num);
    if (alignedColLen == 0) {
        OP_LOGE(context->GetNodeName(), "CalculateMaxUbSizePerRow Unsupported alignedColLen %u == 0 ", alignedColLen);
        return false;
    }
    uint32_t ubAvail = compileInfo.dataNumSingleUb / alignedColLen;
    if (ubAvail == 0) {
        tilingParam.optBaseColLen = DQAlignDown(compileInfo.dataNumSingleUb, compileInfo.block_num);
        if (tilingParam.optBaseColLen > compileInfo.cacheLineLen) {
            tilingParam.optBaseColLen = DQAlignDown<uint32_t>(tilingParam.optBaseColLen, compileInfo.cacheLineLen);
        }
        ubAvail = DQ_ONE;
    } else {
        tilingParam.optBaseColLen = colLen;
        ubAvail = std::max(ubAvail, DQ_ONE);
    }

    tilingParam.optBaseRowLenHeadCore = std::min(std::min(ubAvail, tilingParam.rowLenPerHeadCore), DQ_COMPARE_INT);
    tilingParam.optBaseRowLenTailCore = std::min(std::min(ubAvail, tilingParam.rowLenPerTailCore), DQ_COMPARE_INT);
    return true;
}

inline SwiGluDynamicQuantQuantMode DQGetQuantMode(const SwiGluDynamicQuantCompileInfo &compileInfo)
{
    if (compileInfo.curQuantMode == DQ_QUANT_MODE_DYNAMIC) {
        return SwiGluDynamicQuantQuantMode::DYNAMIC;
    } else {
        return compileInfo.isPerTensor ? SwiGluDynamicQuantQuantMode::STATIC_PER_TENSOR : SwiGluDynamicQuantQuantMode::STATIC_PER_CHANNEL;
    }
}

inline uint32_t DQCalculateTilingKey(ge::DataType xDtype, const SwiGluDynamicQuantCompileInfo &compileInfo)
{
    uint32_t tilingKey = 0;
    std::unordered_map<ge::DataType, uint32_t> dTypeTilingKeyMap{
        {ge::DT_FLOAT16, DQ_TILING_KEY_VAL_FP16},
        {ge::DT_BF16, DQ_TILING_KEY_VAL_BF16},
        {ge::DT_FLOAT, DQ_TILING_KEY_VAL_FP}
    };
    std::unordered_map<SwiGluDynamicQuantQuantMode, uint8_t> quantModeTilingKeyMap{
        {SwiGluDynamicQuantQuantMode::STATIC_PER_TENSOR, DQ_PER_TENSOR_KEY_VAL},
        {SwiGluDynamicQuantQuantMode::STATIC_PER_CHANNEL, DQ_PER_CHANEL_KEY_VAL},
        {SwiGluDynamicQuantQuantMode::DYNAMIC, DQ_DYNAMIC_KEY_VAL}
    };
    SwiGluDynamicQuantQuantMode quantMode = DQGetQuantMode(compileInfo);
    tilingKey += dTypeTilingKeyMap[xDtype];
    tilingKey += quantModeTilingKeyMap[quantMode];
    return tilingKey;
}

inline void DQCalTilingData(gert::TilingContext *context, SwiGluDynamicQuantCompileInfo &compileInfo,
    SwiGluDynamicQuantTilingParam &tilingParam, SwiGluDynamicQuantTilingData &tilingData)
{
    uint32_t rowLen = tilingData.get_rowLen();
    tilingParam.coreNumUsed = std::max(std::min(compileInfo.totalCore, rowLen), DQ_ONE);
    tilingParam.headCoreNum = rowLen % tilingParam.coreNumUsed;
    tilingParam.rowLenPerHeadCore = (rowLen + tilingParam.coreNumUsed - 1) / tilingParam.coreNumUsed;
    tilingParam.rowLenPerTailCore = rowLen / tilingParam.coreNumUsed;
    DQCalculateMaxUbSizePerRow(context, compileInfo, tilingParam, tilingData);
}

inline void DQSetTilingData(SwiGluDynamicQuantCompileInfo &compileInfo, SwiGluDynamicQuantTilingParam &tilingParam,
    SwiGluDynamicQuantTilingData &tilingData)
{
    tilingData.set_headCoreNum(tilingParam.headCoreNum);
    tilingData.set_basicRowLenHeadCore(tilingParam.optBaseRowLenHeadCore);
    tilingData.set_basicRowLenTailCore(tilingParam.optBaseRowLenTailCore);
    tilingData.set_basicColLen(tilingParam.optBaseColLen);
    tilingData.set_rowLenPerHeadCore(tilingParam.rowLenPerHeadCore);
    tilingData.set_rowLenPerTailCore(tilingParam.rowLenPerTailCore);
    tilingData.set_realCoreNum(tilingParam.coreNumUsed);
    tilingData.set_activateLeft(compileInfo.activateLeft);
    tilingData.set_groupListType(compileInfo.groupListType);
    tilingData.set_dstType(compileInfo.dstType);
    tilingData.set_hasGroup(compileInfo.hasGroup);
}
}
#endif
