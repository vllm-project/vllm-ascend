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
 * \file smla_infershape_common.h
 * \brief sparse_flash_mla / mixed_quant_sparse_flash_mla / quant_sparse_flash_mla 三个算子
 *        InferShape / InferDataType 公共子函数。
 */

#ifndef MQSMLA_C624_PRIVATE_SMLA_INFERSHAPE_COMMON_H
#define MQSMLA_C624_PRIVATE_SMLA_INFERSHAPE_COMMON_H

#include <sstream>
#include <string>
#include <vector>
#include <graph/utils/type_utils.h>
#include <exe_graph/runtime/infer_shape_context.h>
#include <exe_graph/runtime/infer_datatype_context.h>
#include <register/op_impl_registry.h>
#include "../include/err/ops_err.h"
#include "smla_host_common_defs.h"

namespace optiling {

// 算子原型输入索引（三个算子 Q / ori_kv / cmp_kv 位置一致）
constexpr uint32_t SMLA_QUERY_INPUT_INDEX = 0;
constexpr uint32_t SMLA_ORI_KV_INPUT_INDEX = 1;
constexpr uint32_t SMLA_CMP_KV_INPUT_INDEX = 2;

inline std::vector<int64_t> MQSMLAC624ToVector(const gert::Shape *shape)
{
    size_t shapeSize = shape->GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);
    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape->GetDim(i);
    }
    return shapeVec;
}

inline std::string MQSMLAC624ToString(const gert::Shape *shape)
{
    std::ostringstream oss;
    auto v = MQSMLAC624ToVector(shape);
    if (v.size() > 0) {
        for (size_t i = 0; i < v.size() - 1; ++i) {
            oss << v[i] << ", ";
        }
        oss << v[v.size() - 1];
    }
    return oss.str();
}

inline int64_t MQSMLAC624GetKvHeadNum(const gert::Shape *kvShape, const std::string &layoutKv)
{
    if (layoutKv == "TND") {
        return kvShape->GetDim(DIM_IDX_ONE);
    }
    return kvShape->GetDim(DIM_IDX_TWO);
}

inline ge::graphStatus MQSMLAC624CheckKvHeadNum(const char *opName, const gert::Shape *kvShape, int64_t kvHeadNum)
{
    OP_CHECK_IF(kvHeadNum <= 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, "ori_kv or cmp_kv", MQSMLAC624ToString(kvShape).c_str(),
                    "The head num of ori_kv or cmp_kv should be greater than 0 but got " + std::to_string(kvHeadNum)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline void MQSMLAC624SetSoftmaxLseShape(gert::Shape *softmaxLseShape, const gert::Shape *queryShape, int64_t kvHeadNum,
                                   bool isTnd, bool returnSoftmaxLse)
{
    if (!returnSoftmaxLse) {
        softmaxLseShape->SetDimNum(DIM_NUM_ONE);
        softmaxLseShape->SetDim(DIM_IDX_ZERO, 0);
        return;
    }
    if (isTnd) {
        softmaxLseShape->SetDimNum(DIM_NUM_THREE);
        softmaxLseShape->SetDim(DIM_IDX_ZERO, kvHeadNum);
        softmaxLseShape->SetDim(DIM_IDX_ONE, queryShape->GetDim(DIM_IDX_ZERO));
        softmaxLseShape->SetDim(DIM_IDX_TWO, queryShape->GetDim(DIM_IDX_ONE) / kvHeadNum);
    } else {
        softmaxLseShape->SetDimNum(DIM_NUM_FOUR);
        softmaxLseShape->SetDim(DIM_IDX_ZERO, queryShape->GetDim(DIM_IDX_ZERO));
        softmaxLseShape->SetDim(DIM_IDX_ONE, kvHeadNum);
        softmaxLseShape->SetDim(DIM_IDX_TWO, queryShape->GetDim(DIM_IDX_ONE));
        softmaxLseShape->SetDim(DIM_IDX_THREE, queryShape->GetDim(DIM_IDX_TWO) / kvHeadNum);
    }
}

struct MQSMLAC624InferShapeBase {
    const gert::Shape *queryShape = nullptr;
    gert::Shape *attentionOutShape = nullptr;
    gert::Shape *softmaxLseShape = nullptr;
    const gert::RuntimeAttrs *attrs = nullptr;
    const bool *returnSoftmaxLsePtr = nullptr;
    const gert::Shape *kvShape = nullptr;
};

// 解析 query 输入、两个输出、attr 及 return_softmax_lse 指针
inline ge::graphStatus MQSMLAC624GetInferShapeBase(gert::InferShapeContext *context, uint32_t returnSoftmaxLseAttrIdx,
                                             MQSMLAC624InferShapeBase &base)
{
    base.queryShape = context->GetInputShape(SMLA_QUERY_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, base.queryShape);
    base.attentionOutShape = context->GetOutputShape(ATTN_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, base.attentionOutShape);
    *base.attentionOutShape = *base.queryShape;

    base.softmaxLseShape = context->GetOutputShape(SOFTMAX_LSE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, base.softmaxLseShape);
    base.attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, base.attrs);
    base.returnSoftmaxLsePtr = base.attrs->GetAttrPointer<bool>(returnSoftmaxLseAttrIdx);
    return ge::GRAPH_SUCCESS;
}

// 三个算子统一的 InferShape 主体：ori_kv / cmp_kv 均为可选输入，layout_q / layout_kv 为 String attr
inline ge::graphStatus MQSMLAC624InferShape(gert::InferShapeContext *context, const char *opName,
                                      uint32_t returnSoftmaxLseAttrIdx, uint32_t layoutQAttrIdx,
                                      uint32_t layoutKvAttrIdx)
{
    MQSMLAC624InferShapeBase base;
    if (MQSMLAC624GetInferShapeBase(context, returnSoftmaxLseAttrIdx, base) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *oriKvShape = context->GetOptionalInputShape(SMLA_ORI_KV_INPUT_INDEX);
    const gert::Shape *cmpKvShape = context->GetOptionalInputShape(SMLA_CMP_KV_INPUT_INDEX);
    base.kvShape = (oriKvShape != nullptr) ? oriKvShape : cmpKvShape;
    OP_CHECK_NULL_WITH_CONTEXT(context, base.kvShape);

    const char *layoutQ = base.attrs->GetStr(layoutQAttrIdx);
    const char *layoutKv = base.attrs->GetStr(layoutKvAttrIdx);
    std::string layoutQStr = (layoutQ != nullptr) ? std::string(layoutQ) : "BSND";
    std::string layoutKvStr = (layoutKv != nullptr) ? std::string(layoutKv) : "BSND";
    bool returnSoftmaxLse = (base.returnSoftmaxLsePtr != nullptr) ? *base.returnSoftmaxLsePtr : false;
    int64_t kvHeadNum = MQSMLAC624GetKvHeadNum(base.kvShape, layoutKvStr);
    if (MQSMLAC624CheckKvHeadNum(opName, base.kvShape, kvHeadNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    MQSMLAC624SetSoftmaxLseShape(base.softmaxLseShape, base.queryShape, kvHeadNum, layoutQStr == "TND", returnSoftmaxLse);
    return ge::GRAPH_SUCCESS;
}

// sparse_flash_mla / mixed_quant_sparse_flash_mla 的 InferDataType（输出 dtype 跟随 query 输入）
inline ge::graphStatus MQSMLAC624InferDataTypeByInput(gert::InferDataTypeContext *context, const char *opName)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(opName, "InferShapeContext is nullptr"), return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(SMLA_QUERY_INPUT_INDEX);
    context->SetOutputDataType(ATTN_OUT_INDEX, inputDataType);
    context->SetOutputDataType(SOFTMAX_LSE_INDEX, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

#endif // MQSMLA_C624_PRIVATE_SMLA_INFERSHAPE_COMMON_H
