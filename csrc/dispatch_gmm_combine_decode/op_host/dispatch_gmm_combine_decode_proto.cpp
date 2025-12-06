/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "log/ops_log.h"
#include "error/ops_error.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"

namespace ge {
constexpr uint32_t EXPAND_X_INDEX = 0;
constexpr uint32_t EXPERT_IDS_INDEX = 1;
constexpr uint32_t OUTPUT_X_INDEX = 0;
constexpr uint32_t OUTPUT_REC_COUNT_INDEX = 1;

constexpr uint32_t ATTR_GROUP_EP_INDEX = 0;
constexpr uint32_t ATTR_EP_RANK_SIZE_INDEX = 1;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2;
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 3;
constexpr uint32_t ATTR_SHARE_EXPERT_NUM_INDEX = 4;
constexpr uint32_t ATTR_SHARE_EXPERT_RANK_NUM_INDEX = 5;
constexpr uint32_t ATTR_QUANT_MODE_INDEX = 6;
constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 7;

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const char *nodeName = context->GetNodeName();
    // infer output shape
    const gert::Shape *expandXShape = context->GetInputShape(EXPAND_X_INDEX);
    const gert::Shape *expertIdsShape = context->GetInputShape(EXPERT_IDS_INDEX);
    gert::Shape *expandXOutShape = context->GetOutputShape(OUTPUT_X_INDEX);
    gert::Shape *recvCountOutShape = context->GetOutputShape(OUTPUT_REC_COUNT_INDEX);
    if (expandXShape == nullptr || expertIdsShape == nullptr || expandXOutShape == nullptr ||
        recvCountOutShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (expandXShape->GetDimNum() < 2 || expertIdsShape->GetDimNum() < 1) {
        return GRAPH_FAILED;
    }

    int bs = expertIdsShape->GetDim(0);
    int h = expandXShape->GetDim(1);

    expandXOutShape->SetDimNum(expandXShape->GetDimNum());
    expandXOutShape->SetDim(0, bs);
    expandXOutShape->SetDim(1, h);

    // infer recvCount shape
    auto attrs = context->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_LOG_E(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto epRankSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARE_EXPERT_RANK_NUM_INDEX);

    OPS_ERR_IF(epRankIdPtr == nullptr, OPS_LOG_E(nodeName, "epRankIdPtr is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(moeExpertNumPtr == nullptr, OPS_LOG_E(nodeName, "moeExpertNumPtr is nullptr."),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF(epRankSizePtr == nullptr, OPS_LOG_E(nodeName, "epRankSizePtr is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(sharedExpertRankNumPtr == nullptr, OPS_LOG_E(nodeName, "sharedExpertRankNumPtr is nullptr."),
                    return ge::GRAPH_FAILED);
    uint32_t epRankSize = static_cast<uint32_t>(*epRankSizePtr);
    uint32_t moeExpertNum = static_cast<uint32_t>(*moeExpertNumPtr);
    uint32_t epRankId = static_cast<uint32_t>(*epRankIdPtr);
    uint32_t sharedExpertRankNum = static_cast<uint32_t>(*sharedExpertRankNumPtr);

    recvCountOutShape->SetDimNum(1);
    bool isShareExpert = (epRankId < sharedExpertRankNum);
    if (isShareExpert) {
        recvCountOutShape->SetDim(0, epRankSize);
    } else {
        recvCountOutShape->SetDim(0, epRankSize * (moeExpertNum / (epRankSize - sharedExpertRankNum)));
    }

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto expandXDataType = context->GetInputDataType(EXPAND_X_INDEX);
    context->SetOutputDataType(OUTPUT_X_INDEX, expandXDataType);
    context->SetOutputDataType(OUTPUT_REC_COUNT_INDEX, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(DispatchGmmCombineDecode).InferShape(InferShape).InferDataType(InferDataType);
}  // namespace ge
