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
 * \file fused_infer_attention_score_tiling_check_single_para.cpp
 * \brief
 */

#include <set>
#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "fused_infer_attention_score_tiling_check.h"

using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
namespace optiling {
const std::map<std::string, std::vector<ge::DataType>> DTYPE_SUPPORT_MAP = {
    {QUERY_NAME,                  {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8}},
    {KEY_NAME,                    {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT4}},
    {VALUE_NAME,                  {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT4}},
    {PSE_SHIFT_NAME,              {ge::DT_FLOAT16, ge::DT_BF16}},
    {ATTEN_MASK_NAME,             {ge::DT_BOOL, ge::DT_INT8, ge::DT_UINT8}},
    {DEQUANT_SCALE1_NAME,         {ge::DT_UINT64, ge::DT_FLOAT}},
    {QUANT_SCALE1_NAME,           {ge::DT_FLOAT}},
    {DEQUANT_SCALE2_NAME,         {ge::DT_UINT64, ge::DT_FLOAT}},
    {QUANT_SCALE2_NAME,           {ge::DT_FLOAT, ge::DT_BF16}},
    {QUANT_OFFSET2_NAME,          {ge::DT_FLOAT, ge::DT_BF16}},
    {ANTIQUANT_SCALE_NAME,        {ge::DT_FLOAT16, ge::DT_BF16}},
    {ANTIQUANT_OFFSET_NAME,       {ge::DT_FLOAT16, ge::DT_BF16}},
    {KEY_ANTIQUANT_SCALE_NAME,    {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}},
    {KEY_ANTIQUANT_OFFSET_NAME,   {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}},
    {VALUE_ANTIQUANT_SCALE_NAME,  {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}},
    {VALUE_ANTIQUANT_OFFSET_NAME, {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}},
    {KEY_SHARED_PREFIX_NAME,      {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8}},
    {VALUE_SHARED_PREFIX_NAME,    {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8}},
    {QUERY_ROPE_NAME,             {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8}},
    {KEY_ROPE_NAME,               {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8}},
    {DEQUANT_SCALE_QUERY_NAME,    {ge::DT_FLOAT}},
    {ATTEN_OUT_NAME,              {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8}},
    {SOFTMAX_LSE_NAME,            {ge::DT_FLOAT}},
};

const std::map<std::string, std::vector<FiaLayout>> LAYOUT_SUPPORT_MAP = {
    {QUERY_NAME,      {FiaLayout::BSH, FiaLayout::BSND, FiaLayout::BNSD, FiaLayout::TND, FiaLayout::NTD}},
    {KEY_NAME,        {FiaLayout::BSH, FiaLayout::BSND, FiaLayout::BNSD, FiaLayout::TND, FiaLayout::NTD, FiaLayout::NZ, FiaLayout::BnBsH, FiaLayout::BnNBsD}},
    {VALUE_NAME,      {FiaLayout::BSH, FiaLayout::BSND, FiaLayout::BNSD, FiaLayout::TND, FiaLayout::NTD, FiaLayout::NZ, FiaLayout::BnBsH, FiaLayout::BnNBsD}},
    {ATTEN_OUT_NAME,  {FiaLayout::BSH, FiaLayout::BSND, FiaLayout::BNSD, FiaLayout::TND, FiaLayout::NTD, FiaLayout::NBSD}},
    {KEY_SHARED_PREFIX_NAME, {FiaLayout::BSH, FiaLayout::BSND, FiaLayout::BNSD}},
    {VALUE_SHARED_PREFIX_NAME, {FiaLayout::BSH, FiaLayout::BSND, FiaLayout::BNSD}},
};

const std::set<ge::Format> FORMAT_SUPPORT_SET = {
    ge::FORMAT_ND,
    ge::FORMAT_NCHW,
    ge::FORMAT_NHWC,
    ge::FORMAT_NCDHW
};

void FiaTilingCheck::LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList,
    const ge::DataType &actualDtype, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectDtypeList.size(); ++i) {
        oss << FusedDataTypeToSerialString(expectDtypeList[i]);
        if (i < expectDtypeList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE(opName_, "Tensor %s only supports dtype %s, but got %s",
        name.c_str(), oss.str().c_str(), FusedDataTypeToSerialString(actualDtype).c_str());
}

ge::graphStatus FiaTilingCheck::CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc,
    const std::string &name) const
{
    if (desc != nullptr) {
        const auto& it = DTYPE_SUPPORT_MAP.find(name);
        OP_CHECK_IF(it == DTYPE_SUPPORT_MAP.end(),
            OP_LOGE(opName_, "%s datatype support list should be specify in DTYPE_SUPPORT_MAP", name.c_str()),
            return ge::GRAPH_FAILED);
        auto &expectDtypeList = it->second;
        OP_CHECK_IF(std::find(
            expectDtypeList.begin(), expectDtypeList.end(), desc->GetDataType()) == expectDtypeList.end(),
            LogErrorDtypeSupport(expectDtypeList, desc->GetDataType(), name),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFormatSupport(const gert::CompileTimeTensorDesc *desc,
    const std::string &name) const
{
    if (desc != nullptr) {
        auto format = desc->GetOriginFormat();
        OP_CHECK_IF(
            (FORMAT_SUPPORT_SET.find(format) == FORMAT_SUPPORT_SET.end()),
            OP_LOGE(opName_, "%s format only supports ND/NCHW/NHWC/NCDHW, but got %d",
                name.c_str(), static_cast<int32_t>(format)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
void FiaTilingCheck::LogErrorNumberSupport(const std::vector<T> &expectNumberList,
    const T &actualValue, const std::string &name, const std::string subName) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectNumberList.size(); ++i) {
        oss << std::to_string(expectNumberList[i]);
        if (i < expectNumberList.size() - 1) {
            oss << ", ";
        }
    }

    OP_LOGE(opName_, "%s %s only supports %s, but got %s",
              name.c_str(), subName.c_str(), oss.str().c_str(), std::to_string(actualValue).c_str());
}

template <typename T>
void FiaTilingCheck::LogErrorDimNumSupport(const std::vector<T> &expectNumberList,
    const T &actualValue, const std::string &name) const
{
    LogErrorNumberSupport(expectNumberList, actualValue, name, "dim num");
}

template <typename T>
void FiaTilingCheck::LogErrorShapeNumSupport(const std::vector<T> &expectNumberList,
    const T &actualValue, const std::string &name) const
{
    LogErrorNumberSupport(expectNumberList, actualValue, name, "shape num");
}

template <typename T>
void FiaTilingCheck::LogErrorAttrValueSupport(const std::vector<T> &expectNumberList,
    const T &actualValue, const std::string &name) const
{
    LogErrorNumberSupport(expectNumberList, actualValue, name, "attr value");
}

ge::graphStatus FiaTilingCheck::CheckDimNumSupport(const gert::StorageShape *shape,
    const std::vector<size_t> &expectDimNumList, const std::string &name) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(expectDimNumList.begin(), expectDimNumList.end(),
        shape->GetStorageShape().GetDimNum()) == expectDimNumList.end()) {
        LogErrorDimNumSupport(expectDimNumList, shape->GetStorageShape().GetDimNum(), name);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckDimNumSupport(const gert::Tensor *tensor,
    const std::vector<size_t> &expectDimNumList, const std::string &name) const
{
    if (tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(expectDimNumList.begin(), expectDimNumList.end(),
        tensor->GetStorageShape().GetDimNum()) == expectDimNumList.end()) {
        LogErrorDimNumSupport(expectDimNumList, tensor->GetStorageShape().GetDimNum(), name);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckShapeSupport(const gert::Tensor *tensor,
    const std::vector<int64_t> &expectShapeList, const std::string &name) const
{
    if (tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(expectShapeList.begin(), expectShapeList.end(),
        tensor->GetShapeSize()) == expectShapeList.end()) {
        LogErrorShapeNumSupport(expectShapeList, tensor->GetShapeSize(), name);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus FiaTilingCheck::CheckAttrValueSupport(const T *attrValue,
    const std::vector<T> &expectAttrValList, const std::string &name) const
{
    if (attrValue == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(expectAttrValList.begin(), expectAttrValList.end(), *attrValue) == expectAttrValList.end()) {
        LogErrorAttrValueSupport(expectAttrValList, *attrValue, name);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void FiaTilingCheck::LogErrorLayoutSupport(const std::vector<FiaLayout> &expectLayoutList,
    const FiaLayout &actualLayout, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectLayoutList.size(); ++i) {
        oss << LayoutToSerialString(expectLayoutList[i]);
        if (i < expectLayoutList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE(opName_, "%s only supports layout %s, but got %s",
        name.c_str(), oss.str().c_str(), LayoutToSerialString(actualLayout).c_str());
}

ge::graphStatus FiaTilingCheck::CheckLayoutSupport(const FiaLayout &actualLayout, const std::string &name) const
{
    const auto& it = LAYOUT_SUPPORT_MAP.find(name);
    OP_CHECK_IF(it == LAYOUT_SUPPORT_MAP.end(),
        OP_LOGE(opName_, "%s layout support list should be specify in LAYOUT_SUPPORT_MAP", name.c_str()),
        return ge::GRAPH_FAILED);
    auto &expectLayoutList = it->second;
    OP_CHECK_IF(std::find(
        expectLayoutList.begin(), expectLayoutList.end(), actualLayout) == expectLayoutList.end(),
        LogErrorLayoutSupport(expectLayoutList, actualLayout, name),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaQuery() const
{
    const std::vector<size_t> queryDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.query.desc, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(qLayout_, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(opParamInfo_.query.shape, queryDimNumList, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckFormatSupport(opParamInfo_.query.desc, QUERY_NAME)) {
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(opParamInfo_.query.shape->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE(opName_, "%s shapesize is 0.", QUERY_NAME.c_str()), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaKey() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.key.desc, KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(kvLayout_, KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckFormatSupport(opParamInfo_.key.desc, KEY_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaValue() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.value.desc, VALUE_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(kvLayout_, VALUE_NAME) ||
        ge::GRAPH_SUCCESS != CheckFormatSupport(opParamInfo_.value.desc, VALUE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaPseShift() const
{
    const std::vector<size_t> pseShiftDimNumList = {DIM_NUM_FOUR};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.pseShift.desc, PSE_SHIFT_NAME) ||
 	    ge::GRAPH_SUCCESS != CheckFormatSupport(opParamInfo_.pseShift.desc, PSE_SHIFT_NAME) ||
 	    ge::GRAPH_SUCCESS != CheckDimNumSupport(opParamInfo_.pseShift.tensor, pseShiftDimNumList, PSE_SHIFT_NAME)) {
 	    return ge::GRAPH_FAILED;
 	}
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaAttenMask() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.attenMask.desc, ATTEN_MASK_NAME) ||
        ge::GRAPH_SUCCESS != CheckFormatSupport(opParamInfo_.attenMask.desc, ATTEN_MASK_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaActualSeqLengthsQ() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaActualSeqLengths() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaDeqScale1() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.deqScale1.desc, DEQUANT_SCALE1_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaQuantScale1() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.quantScale1.desc, QUANT_SCALE1_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaDeqScale2() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.deqScale2.desc, DEQUANT_SCALE2_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaQuantScale2() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.quantScale2.desc, QUANT_SCALE2_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaQuantOffset2() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.quantOffset2.desc, QUANT_OFFSET2_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaAntiquantScale() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.antiquantScale.desc, ANTIQUANT_SCALE_NAME) ||
        ge::GRAPH_SUCCESS != CheckFormatSupport(opParamInfo_.antiquantScale.desc, ANTIQUANT_SCALE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaAntiquantOffset() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.antiquantOffset.desc, ANTIQUANT_OFFSET_NAME) ||
        ge::GRAPH_SUCCESS != CheckFormatSupport(opParamInfo_.antiquantOffset.desc, ANTIQUANT_OFFSET_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaBlockTable() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaQueryPaddingSize() const
{
    if (fiaInfo_.qPaddingSizeFlag) {
 	    const std::vector<size_t> querypaddingsizeDimNumList = {DIM_NUM_ONE};
 	    const std::vector<int64_t> querypaddingsizeShapeNumList = {SHAPE_NUM_ONE};
 	    if (ge::GRAPH_SUCCESS != CheckShapeSupport(opParamInfo_.queryPaddingSize.tensor, querypaddingsizeShapeNumList, QUERY_PADDING_SIZE_NAME) ||
 	        ge::GRAPH_SUCCESS != CheckDimNumSupport(opParamInfo_.queryPaddingSize.tensor, querypaddingsizeDimNumList, QUERY_PADDING_SIZE_NAME)) {
 	        return ge::GRAPH_FAILED;
 	    }
 	}

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaKvPaddingSize() const
{
    if (ge::GRAPH_SUCCESS != CheckFormatSupport(opParamInfo_.kvPaddingSize.desc, KV_PADDING_SIZE_NAME)) {
        return ge::GRAPH_FAILED;
    }

    if (!fiaInfo_.isLegacyIfa) {
 	    const std::vector<size_t> kvpaddingsizeDimNumList = {DIM_NUM_ONE};
 	    const std::vector<int64_t> kvpaddingsizeShapeNumList = {SHAPE_NUM_ONE};
 	    if (ge::GRAPH_SUCCESS != CheckShapeSupport(opParamInfo_.kvPaddingSize.tensor, kvpaddingsizeShapeNumList, KV_PADDING_SIZE_NAME) ||
 	        ge::GRAPH_SUCCESS != CheckDimNumSupport(opParamInfo_.kvPaddingSize.tensor, kvpaddingsizeDimNumList, KV_PADDING_SIZE_NAME)) {
 	        return ge::GRAPH_FAILED;
 	    }
 	}

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaKeyAntiquantScale() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.keyAntiquantScale.desc, KEY_ANTIQUANT_SCALE_NAME) ||
        ge::GRAPH_SUCCESS != CheckFormatSupport(opParamInfo_.keyAntiquantScale.desc, KEY_ANTIQUANT_SCALE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaKeyAntiquantOffset() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.keyAntiquantOffset.desc, KEY_ANTIQUANT_OFFSET_NAME) ||
        ge::GRAPH_SUCCESS != CheckFormatSupport(opParamInfo_.keyAntiquantOffset.desc, KEY_ANTIQUANT_OFFSET_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaValueAntiquantScale() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.valueAntiquantScale.desc, VALUE_ANTIQUANT_SCALE_NAME) ||
        ge::GRAPH_SUCCESS != CheckFormatSupport(opParamInfo_.valueAntiquantScale.desc, VALUE_ANTIQUANT_SCALE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaValueAntiquantOffset() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.valueAntiquantOffset.desc, VALUE_ANTIQUANT_OFFSET_NAME) ||
        ge::GRAPH_SUCCESS != CheckFormatSupport(opParamInfo_.valueAntiquantOffset.desc, VALUE_ANTIQUANT_OFFSET_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaKeySharedPrefix() const
{
    if (!fiaInfo_.sysPrefixFlag) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.keySharedPrefix.desc, KEY_SHARED_PREFIX_NAME) ||
        ge::GRAPH_SUCCESS != CheckFormatSupport(opParamInfo_.keySharedPrefix.desc, KEY_SHARED_PREFIX_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(kvLayout_, KEY_SHARED_PREFIX_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaValueSharedPrefix() const
{
    if (!fiaInfo_.sysPrefixFlag) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.valueSharedPrefix.desc, VALUE_SHARED_PREFIX_NAME) ||
        ge::GRAPH_SUCCESS != CheckFormatSupport(opParamInfo_.valueSharedPrefix.desc, VALUE_SHARED_PREFIX_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(kvLayout_, VALUE_SHARED_PREFIX_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaQueryRope() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.queryRope.desc, QUERY_ROPE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaKeyRope() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.keyRope.desc, KEY_ROPE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaKeyRopeAntiquantScale() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaDequantScaleQuery() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.dequantScaleQuery.desc, DEQUANT_SCALE_QUERY_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaAttenOut() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.attenOut.desc, ATTEN_OUT_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(outLayout_, ATTEN_OUT_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaLseOut() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.lseOut.desc, SOFTMAX_LSE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaNumHeads() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaPreToken() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaNextToken() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaScaleValue() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaKvHeadNums() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaLayout() const
{
    const std::vector<std::string> inputLayoutList = {
        "BSH", "BSND", "BNSD", "TND", "NTD", "BSH_NBSD", "BSND_NBSD", "BNSD_NBSD", "TND_NTD", "NTD_TND", "BSH_BNSD", "BSND_BNSD", "BNSD_BSND"
    };
    std::string inputLayout = opParamInfo_.layOut;
    if (std::find(inputLayoutList.begin(), inputLayoutList.end(), inputLayout) == inputLayoutList.end()) {
        OP_LOGE(opName_,
            "input layout only supports BSH, BSND, BNSD, TND, NTD, BSH_NBSD, BSND_NBSD, BNSD_NBSD, TND_NTD, NTD_TND, BSH_BNSD, BSND_BNSD, BNSD_BSND, but got %s",
            inputLayout.c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaBlockSize() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaInnerPrecise() const
{
    const std::vector<int32_t> innerPreciseList = {
        INNER_PRECISE_HIGH_PRECISION,
        INNER_PRECISE_HIGH_PERFORMANCE,
        INNER_PRECISE_HIGH_PRECISION_ROW_INVALID,
        INNER_PRECISE_HIGH_PERFORMANCE_ROW_INVALID
    };
    if (ge::GRAPH_SUCCESS != CheckAttrValueSupport(opParamInfo_.innerPrecise, innerPreciseList, INNER_PRECISE_NAME)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaAntiquantMode() const
{
    const std::vector<int64_t> antiquantModeList = {
        ANTIQUANT_PER_CHANNEL_MODE,
        ANTIQUANT_PER_TOKEN_MODE
    };
    if (ge::GRAPH_SUCCESS != CheckAttrValueSupport(opParamInfo_.antiquantMode,
        antiquantModeList, ANTIQUANT_MODE_NAME)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaSoftmaxLseFlag() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaKeyAntiquantMode() const
{
    const std::vector<int64_t> keyAntiquantModeList = {
        ANTIQUANT_PER_CHANNEL_MODE,
        ANTIQUANT_PER_TOKEN_MODE,
        ANTIQUANT_PER_TENSOR_HEAD_MODE,
        ANTIQUANT_PER_TOKEN_HEAD_MODE,
        ANTIQUANT_PER_TOKEN_PA_MODE,
        ANTIQUANT_PER_TOKEN_HEAD_PA_MODE
    };
    if (ge::GRAPH_SUCCESS != CheckAttrValueSupport(opParamInfo_.keyAntiquantMode,
        keyAntiquantModeList, KEY_ANTIQUANT_MODE_NAME)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaValueAntiquantMode() const
{
    const std::vector<int64_t> valueAntiquantModeList = {
        ANTIQUANT_PER_CHANNEL_MODE,
        ANTIQUANT_PER_TOKEN_MODE,
        ANTIQUANT_PER_TENSOR_HEAD_MODE,
        ANTIQUANT_PER_TOKEN_HEAD_MODE,
        ANTIQUANT_PER_TOKEN_PA_MODE,
        ANTIQUANT_PER_TOKEN_HEAD_PA_MODE
    };
    if (ge::GRAPH_SUCCESS != CheckAttrValueSupport(opParamInfo_.valueAntiquantMode,
        valueAntiquantModeList, VALUE_ANTIQUANT_MODE_NAME)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaSparseMode() const
{
    const std::vector<int32_t> sparseModeList = {
        SPARSE_MODE_NO_MASK,
        SPARSE_MODE_ALL_MASK,
        SPARSE_MODE_LEFT_UP,
        SPARSE_MODE_RIGHT_DOWN,
        SPARSE_MODE_BAND,
        SPARSE_MODE_TREE
    };
    if (ge::GRAPH_SUCCESS != CheckAttrValueSupport(opParamInfo_.sparseMode, sparseModeList, SPARSE_MODE_NAME)) {
        OP_LOGE(opName_, "sparseMode only supports 0/1/2/3/4/9, but got %u", *opParamInfo_.sparseMode);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSingleParaQueryQuantMode() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckSinglePara() const
{
    if (ge::GRAPH_SUCCESS != CheckSingleParaQuery() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKey() ||
        ge::GRAPH_SUCCESS != CheckSingleParaValue() ||
        ge::GRAPH_SUCCESS != CheckSingleParaPseShift() ||
        ge::GRAPH_SUCCESS != CheckSingleParaAttenMask() ||
        ge::GRAPH_SUCCESS != CheckSingleParaActualSeqLengthsQ() ||
        ge::GRAPH_SUCCESS != CheckSingleParaActualSeqLengths() ||
        ge::GRAPH_SUCCESS != CheckSingleParaDeqScale1() ||
        ge::GRAPH_SUCCESS != CheckSingleParaQuantScale1() ||
        ge::GRAPH_SUCCESS != CheckSingleParaDeqScale2() ||
        ge::GRAPH_SUCCESS != CheckSingleParaQuantScale2() ||
        ge::GRAPH_SUCCESS != CheckSingleParaQuantOffset2() ||
        ge::GRAPH_SUCCESS != CheckSingleParaAntiquantScale() ||
        ge::GRAPH_SUCCESS != CheckSingleParaAntiquantOffset() ||
        ge::GRAPH_SUCCESS != CheckSingleParaBlockTable() ||
        ge::GRAPH_SUCCESS != CheckSingleParaQueryPaddingSize() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKvPaddingSize() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKeyAntiquantScale() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKeyAntiquantOffset() ||
        ge::GRAPH_SUCCESS != CheckSingleParaValueAntiquantScale() ||
        ge::GRAPH_SUCCESS != CheckSingleParaValueAntiquantOffset() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKeySharedPrefix() ||
        ge::GRAPH_SUCCESS != CheckSingleParaValueSharedPrefix() ||
        ge::GRAPH_SUCCESS != CheckSingleParaQueryRope() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKeyRope() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKeyRopeAntiquantScale() ||
        ge::GRAPH_SUCCESS != CheckSingleParaDequantScaleQuery() ||
        ge::GRAPH_SUCCESS != CheckSingleParaAttenOut() ||
        ge::GRAPH_SUCCESS != CheckSingleParaLseOut() ||
        ge::GRAPH_SUCCESS != CheckSingleParaNumHeads() ||
        ge::GRAPH_SUCCESS != CheckSingleParaPreToken() ||
        ge::GRAPH_SUCCESS != CheckSingleParaNextToken() ||
        ge::GRAPH_SUCCESS != CheckSingleParaScaleValue() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKvHeadNums() ||
        ge::GRAPH_SUCCESS != CheckSingleParaLayout() ||
        ge::GRAPH_SUCCESS != CheckSingleParaBlockSize() ||
        ge::GRAPH_SUCCESS != CheckSingleParaInnerPrecise() ||
        ge::GRAPH_SUCCESS != CheckSingleParaAntiquantMode() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSoftmaxLseFlag() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKeyAntiquantMode() ||
        ge::GRAPH_SUCCESS != CheckSingleParaValueAntiquantMode() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSparseMode() ||
        ge::GRAPH_SUCCESS != CheckSingleParaQueryQuantMode()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
