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
 * \file moe_init_routing_custom_infershape.cpp
 * \brief
 */
 
#include <sstream>
#include <string>
#include <vector>
#include "register/op_def_registry.h"
#include "log/ops_log.h"
#include "platform/platform_info.h"

#define unlikely(x) __builtin_expect((x), 0)
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                           \
    do {                                                                                                   \
        if (unlikely((ptr) == nullptr)) {                                                                  \
            const char* name = (unlikely(((context) == nullptr) || (context)->GetNodeName() == nullptr)) ? \
                                   "nil" :                                                                 \
                                   (context)->GetNodeName();                                               \
            OPS_LOG_E(name, "%s is nullptr!", #ptr);                                                         \
            return ge::GRAPH_FAILED;                                                                       \
        }                                                                                                  \
    } while (0)

using namespace ge;
namespace ops {
static constexpr size_t DIM_ONE = 1U;
static constexpr size_t DIM_TWO = 2U;
static constexpr size_t DIM_THREE = 3U;
static constexpr int64_t NEG_ONE = static_cast<int64_t>(-1);
static constexpr int64_t NEG_TWO = static_cast<int64_t>(-2);
static constexpr int64_t MOE_INIT_ROUTING_CUSTOM_INPUT_X = 0;
static constexpr int64_t MOE_INIT_ROUTING_CUSTOM_INPUT_EXPERT_IDX = 1;
static constexpr int64_t MOE_INIT_ROUTING_CUSTOM_INPUT_SCALE = 2;
static constexpr int64_t MOE_INIT_ROUTING_CUSTOM_INPUT_OFFSET = 3;
static constexpr int64_t MOE_INIT_ROUTING_CUSTOM_ATTR_ACTIVE_NUM = 0;
static constexpr int64_t MOE_INIT_ROUTING_CUSTOM_ATTR_EXPERT_CAPACITY = 1;
static constexpr int64_t MOE_INIT_ROUTING_CUSTOM_ATTR_EXPERT_NUM = 2;
static constexpr int64_t MOE_INIT_ROUTING_CUSTOM_ATTR_DROP_PAD_MODE = 3;
static constexpr int64_t MOE_INIT_ROUTING_CUSTOM_ATTR_EXPERT_TOKEN_NUM_TYPE = 4;
static constexpr int64_t MOE_INIT_ROUTING_CUSTOM_ATTR_EXPERT_TOKEN_NUM_FLAG = 5;
static constexpr int64_t MOE_INIT_ROUTING_CUSTOM_ATTR_QUANT_MODE = 6;
static constexpr int64_t MOE_INIT_ROUTING_CUSTOM_ATTR_ACTIVE_EXPERT_RANGE = 7;
static constexpr int64_t MOE_INIT_ROUTING_CUSTOM_ATTR_ROW_IDX_TYPE = 8;
static constexpr int64_t MOE_INIT_ROUTING_CUSTOM_OUTPUT_EXPANDED_X = 0;
static constexpr int64_t MOE_INIT_ROUTING_CUSTOM_OUTPUT_EXPANDED_ROW_IDX = 1;
static constexpr int64_t MOE_INIT_ROUTING_CUSTOM_OUTPUT_EXPERT_TOKEN_CUMSUM_OR_COUNT = 2;
static constexpr int64_t MOE_INIT_ROUTING_CUSTOM_OUTPUT_EXPANDED_SCALE = 3;
static constexpr int64_t MOE_INIT_ROUTING_CUSTOM_EXPERT_END_BOUND = 10240;
static constexpr int64_t KEY_VALUE_MODE_DIM0_NUM = 2;
enum DropPadMode : int8_t {
    NO_DROP_PAD = 0,
    DROP_PAD = 1,
};
enum QuantMode : int8_t {
    NON_QUANT = -1,
    STATIC_QUANT = 0,
    DYNAMIC_QUANT = 1
};
enum ExpertTokenNumType : int8_t {
    CUMSUM = 0,
    COUNT = 1,
    KEY_VALUE = 2
};

static bool isSameDim(int64_t dim1, int64_t dim2)
{
    if (dim1 <= NEG_ONE || dim2 <= NEG_ONE) {
        return true;
    }
    return dim1 == dim2;
}

static ge::graphStatus GetAndCheckAttrActiveExpertRange(const gert::RuntimeAttrs *attrs,
                                                        gert::InferShapeContext *context, int64_t &expertStart,
                                                        int64_t &expertEnd, int64_t &experNum)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do GetAndCheckAttrActiveExpertRange.");
    // Check if active_expert_range size is 2 and if expert_start < expert_end
    auto activeExpertRangePtr = attrs->GetListInt(MOE_INIT_ROUTING_CUSTOM_ATTR_ACTIVE_EXPERT_RANGE);
    if (nullptr == activeExpertRangePtr) {
        OPS_LOG_E(context->GetNodeName(), "The active_expert_range should be list int. But it is none.");
        return ge::GRAPH_FAILED;
    }
    int64_t activeExpertRangeSize = activeExpertRangePtr->GetSize();
    if (activeExpertRangePtr->GetSize() == DIM_TWO) {
        expertStart = activeExpertRangePtr->GetData()[0];
        expertEnd = activeExpertRangePtr->GetData()[1];
        if (expertStart >= expertEnd || expertStart < 0 || expertEnd > MOE_INIT_ROUTING_CUSTOM_EXPERT_END_BOUND) {
            OPS_LOG_E(context->GetNodeName(),
                    "The active_expert_range should be in [0, %ld), but the active_expert_range is [%ld, %ld).",
                    MOE_INIT_ROUTING_CUSTOM_EXPERT_END_BOUND, expertStart, expertEnd);
            return ge::GRAPH_FAILED;
        }
    } else if (activeExpertRangePtr->GetSize() == 0) {
        expertStart = 0;
        expertEnd = experNum;
    } else {
        OPS_LOG_E(context->GetNodeName(), "The active_expert_range size should be 2, but its size is %ld.", activeExpertRangeSize);
        return ge::GRAPH_FAILED;
    }

    OPS_LOG_D(context->GetNodeName(), "End to do GetAndCheckAttrActiveExpertRange.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAndCheckAttrActiveNum(const gert::RuntimeAttrs *attrs, gert::InferShapeContext *context,
                                                int64_t &activeNum, int64_t &dropPadMode)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do GetAndCheckAttrActiveNum.");
    const int64_t *activeNumPtr = attrs->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_CUSTOM_ATTR_ACTIVE_NUM);
    if (nullptr == activeNumPtr) {
        OPS_LOG_E(context->GetNodeName(), "The active_num should not be none.");
        return ge::GRAPH_FAILED;
    }
    activeNum = *activeNumPtr;
    if (dropPadMode == DropPadMode::NO_DROP_PAD && activeNum < -1) {
    	OPS_LOG_E(context->GetNodeName(), "The active_num should be greater than or equal to 0. But it is %ld.", activeNum);
        return ge::GRAPH_FAILED;
    }

    OPS_LOG_D(context->GetNodeName(), "End to do GetAndCheckAttrActiveNum.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAndCheckAttrExpertCapacity(const gert::RuntimeAttrs *attrs, gert::InferShapeContext *context,
                                                     const gert::Shape *xShape, int64_t &expertCapacity,
                                                     int64_t &dropPadMode)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do GetAndCheckAttrExpertCapacity.");
    const int64_t *expertCapacityPtr = attrs->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_CUSTOM_ATTR_EXPERT_CAPACITY);
    if (nullptr == expertCapacityPtr) {
        OPS_LOG_E(context->GetNodeName(), "The expert_capacity should not be none.");
        return ge::GRAPH_FAILED;
    }
    expertCapacity = *expertCapacityPtr;
    if (dropPadMode == DropPadMode::DROP_PAD && xShape->GetDim(0) > 0 && expertCapacity > xShape->GetDim(0)) {
            OPS_LOG_E(context->GetNodeName(), "The expert_capacity should be between 0 and n. But it is %ld.", expertCapacity);
            return ge::GRAPH_FAILED;
    }

    OPS_LOG_D(context->GetNodeName(), "End to do GetAndCheckAttrExpertCapacity.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAndCheckAttrExpertNum(const gert::RuntimeAttrs *attrs, gert::InferShapeContext *context,
                                                int64_t &experNum)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do GetAndCheckexperNum.");
    const int64_t *experNumPtr = attrs->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_CUSTOM_ATTR_EXPERT_NUM);
    if (nullptr == experNumPtr) {
        OPS_LOG_E(context->GetNodeName(), "The expert_num should not be none.");
        return ge::GRAPH_FAILED;
    }
    experNum = *experNumPtr;
    if (experNum <= 0 || experNum > MOE_INIT_ROUTING_CUSTOM_EXPERT_END_BOUND) {
        OPS_LOG_E(context->GetNodeName(), "The expert_num should be greater than 0. But it is %ld.", experNum);
        return ge::GRAPH_FAILED;
    }

    OPS_LOG_D(context->GetNodeName(), "End to do GetAndCheckAttrExpertNum.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAndCheckAttrDropPadMode(const gert::RuntimeAttrs *attrs, gert::InferShapeContext *context,
                                                  int64_t &dropPadMode)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do GetAndCheckAttrDropPadMode.");
    const int64_t *dropPadModePtr = attrs->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_CUSTOM_ATTR_DROP_PAD_MODE);
    if (nullptr == dropPadModePtr) {
        OPS_LOG_E(context->GetNodeName(), "The RuntimeAttrs for drop_pad_mode is none.");
        return ge::GRAPH_FAILED;
    }

    dropPadMode = *dropPadModePtr;
    if (dropPadMode < DropPadMode::NO_DROP_PAD || dropPadMode > DropPadMode::DROP_PAD) {
        OPS_LOG_E(context->GetNodeName(), "The drop_pad_mode should be %d or %d. But it is %ld.", DropPadMode::NO_DROP_PAD,
                DropPadMode::DROP_PAD, dropPadMode);
        return ge::GRAPH_FAILED;
    }

    OPS_LOG_D(context->GetNodeName(), "End to do GetAndCheckAttrDropPadMode.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAndCheckAttrExpertTokenNumType(const gert::RuntimeAttrs *attrs, gert::InferShapeContext* context,
                                                         int64_t &experTokenNumType)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do GetAndCheckexperTokenNumType.");
    const int64_t *experTokenNumTypePtr =
        attrs->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_CUSTOM_ATTR_EXPERT_TOKEN_NUM_TYPE);
    if (nullptr == experTokenNumTypePtr) {
        OPS_LOG_E(context->GetNodeName(), "The expert_token_num_type should not be none.");
        return ge::GRAPH_FAILED;
    }
    experTokenNumType = *experTokenNumTypePtr;
    if (experTokenNumType < ExpertTokenNumType::CUMSUM || experTokenNumType > ExpertTokenNumType::KEY_VALUE) {
        OPS_LOG_E(context->GetNodeName(), "The expert_token_num_type should be %d, %d or %d. But it is %ld.",
                  ExpertTokenNumType::CUMSUM, ExpertTokenNumType::COUNT, ExpertTokenNumType::KEY_VALUE,
                  experTokenNumType);
        return ge::GRAPH_FAILED;
    }

    OPS_LOG_D(context->GetNodeName(), "End to do GetAndCheckAttrExpertTokenNumType.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAndCheckAttrExpertTokenNumFlag(const gert::RuntimeAttrs *attrs,
                                                         gert::InferShapeContext *context, bool &experTokenNumFlag)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do GetAndCheckexperTokenNumType.");
    const bool *experTokenNumFlagPtr = attrs->GetAttrPointer<bool>(MOE_INIT_ROUTING_CUSTOM_ATTR_EXPERT_TOKEN_NUM_FLAG);
    if (nullptr == experTokenNumFlagPtr) {
        OPS_LOG_E(context->GetNodeName(), "The expert_token_num_flag should not be none.");
        return ge::GRAPH_FAILED;
    }
    experTokenNumFlag = *experTokenNumFlagPtr;
    OPS_LOG_D(context->GetNodeName(), "End to do GetAndCheckAttrExpertTokenNumType.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAndCheckAttrQuantMode(const gert::RuntimeAttrs *attrs, gert::InferShapeContext *context,
                                                int64_t &quantMode)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do GetAndCheckQuantMode.");
    if (nullptr == attrs) {
        OPS_LOG_E(context->GetNodeName(), "The RuntimeAttrs for quant_mode is none.");
        return ge::GRAPH_FAILED;
    }
    const int64_t *quantModePtr = attrs->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_CUSTOM_ATTR_QUANT_MODE);
    if (nullptr == quantModePtr) {
        OPS_LOG_E(context->GetNodeName(), "The quant_mode should be %d, %d or %d. But it is none.", QuantMode::NON_QUANT,
                QuantMode::STATIC_QUANT, QuantMode::DYNAMIC_QUANT);
        return ge::GRAPH_FAILED;
    }
    quantMode = *quantModePtr;
    if (quantMode < QuantMode::NON_QUANT || quantMode > QuantMode::DYNAMIC_QUANT) {
        OPS_LOG_E(context->GetNodeName(), "The quant_mode should be %d, %d or %d. But it is %ld.", QuantMode::NON_QUANT,
                QuantMode::STATIC_QUANT, QuantMode::DYNAMIC_QUANT, quantMode);
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_D(context->GetNodeName(), "End to do GetAndCheckQuantMode.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAndCheckAttrRowIdxType(const gert::RuntimeAttrs *attrs, gert::InferShapeContext *context,
                                                 int64_t &rowIdxType, int64_t &dropPadMode)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do GetAndCheckAttrRowIdxType.");
    if (nullptr == attrs) {
        OPS_LOG_E(context->GetNodeName(), "The RuntimeAttrs for row_Idx_type is none.");
        return ge::GRAPH_FAILED;
    }
    const int64_t *dropPadModePtr = attrs->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_CUSTOM_ATTR_DROP_PAD_MODE);
    dropPadMode = *dropPadModePtr;

    const int64_t *rowIdxTypePtr = attrs->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_CUSTOM_ATTR_ROW_IDX_TYPE);
    if (nullptr == rowIdxTypePtr) {
        OPS_LOG_E(context->GetNodeName(), "The row_Idx_type should be 0 or 1. But it is none.");
        return ge::GRAPH_FAILED;
    }
    rowIdxType = *rowIdxTypePtr;
    if (dropPadMode == DropPadMode::DROP_PAD && rowIdxType != 0) {
    	OPS_LOG_E(context->GetNodeName(), "The row_Idx_type should be 0 when dropPadMode is equal to 1 But it is %ld.", rowIdxType);
        return ge::GRAPH_FAILED;
    }

    if (rowIdxType < 0 || rowIdxType > 1) {
        OPS_LOG_E(context->GetNodeName(), "The row_Idx_type should be 0 or 1 But it is %ld.", rowIdxType);
        return ge::GRAPH_FAILED;
    }

    OPS_LOG_D(context->GetNodeName(), "End to do GetAndCheckAttrRowIdxType.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputScaleShape(gert::InferShapeContext *context, const gert::Shape *xShape,
                                            const gert::Shape *scaleShape, const int64_t expertStart,
                                            const int64_t expertEnd, const int64_t quantMode)
{
    // When quant_mode is STATIC_QUANT, scale cannot be none.
    OP_CHECK((nullptr == scaleShape && QuantMode::STATIC_QUANT == quantMode),
                OPS_LOG_E(context->GetNodeName(), "The scale cannot be none when quant_mode is %ld.", quantMode),
                return ge::GRAPH_FAILED);

    // When quant_mode is NON_QUANT or DYNAMIC_QUANT, scale can be none.
    OP_CHECK((nullptr == scaleShape && (QuantMode::NON_QUANT == quantMode || QuantMode::DYNAMIC_QUANT == quantMode)),
                OPS_LOG_I(context->GetNodeName(), "When quant_mode is NON_QUANT or DYNAMIC_QUANT, scale can be none."),
                return ge::GRAPH_SUCCESS);

    if (QuantMode::NON_QUANT == quantMode) {
        if (scaleShape->GetDimNum() == DIM_ONE) {
            OP_CHECK(scaleShape->GetDim(0) < 0 && scaleShape->GetDim(0) != NEG_ONE && scaleShape->GetDim(0) != NEG_TWO,
                     OPS_LOG_E(context->GetNodeName(),
                     "When quant_mode is %ld and use scale in dynamic graph, The shape of scale should be (-1) or (-2), current shape is (%s).",
                     quantMode, ops::Shape2String(*scaleShape).c_str()),
                     return ge::GRAPH_FAILED);
            OP_CHECK(scaleShape->GetDim(0) > 0 && !isSameDim(scaleShape->GetDim(0), xShape->GetDim(0)),
                     OPS_LOG_E(context->GetNodeName(),
                     "When quant_mode is %ld and use scale in static graph, The shape of scale should be (%ld,), current shape is (%s).",
                     quantMode, xShape->GetDim(0), ops::Shape2String(*scaleShape).c_str()),
                     return ge::GRAPH_FAILED);
        } else {
            OPS_LOG_E(context->GetNodeName(), "When quant_mode is %ld, The dimNum of scale should be 1, current shape is (%ld).", quantMode,
                      scaleShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
    } else if (QuantMode::STATIC_QUANT == quantMode) {
        if (scaleShape->GetDimNum() == DIM_ONE) {
            OP_CHECK(
                scaleShape->GetDim(0) != NEG_ONE && scaleShape->GetDim(0) != NEG_TWO &&
                    !isSameDim(scaleShape->GetDim(0), DIM_ONE),
                OPS_LOG_E(
                    context->GetNodeName(),
                    "When quant_mode is %ld, the shape of scale should be (-1) or (-2) or (1,), current shape is (%s).",
                    quantMode, ops::Shape2String(*scaleShape).c_str()),
                return ge::GRAPH_FAILED);
        } else {
            OPS_LOG_E(context->GetNodeName(), "When quant_mode is %ld, the dimNum of scale should be (1,), current shape is (%ld).",
                      quantMode, scaleShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
    } else if (QuantMode::DYNAMIC_QUANT == quantMode) {
        int64_t activeExpertRange = expertEnd - expertStart;
        if (scaleShape->GetDimNum() == DIM_ONE) {
            OP_CHECK(scaleShape->GetDim(0) != NEG_TWO,
                     OPS_LOG_E(context->GetNodeName(),
                     "When quant_mode is %ld and scale dim is 1 in dynamic graph, the first dim of scale should be -2, but "
                     "its shape is (%ld).",
                     quantMode, scaleShape->GetDim(0)),
                     return ge::GRAPH_FAILED);
        } else if (scaleShape->GetDimNum() == DIM_TWO) {
            if (scaleShape->GetDim(0) > 0) {
                OP_CHECK(
                    !isSameDim(scaleShape->GetDim(0), activeExpertRange) && !isSameDim(scaleShape->GetDim(0), DIM_ONE),
                    OPS_LOG_E(
                        context->GetNodeName(),
                        "When quant_mode is %ld in static graph, the first dim of scale should be 1 or %ld, but its shape is (%ld).",
                        quantMode, activeExpertRange, scaleShape->GetDim(0)),
                    return ge::GRAPH_FAILED);
                OP_CHECK(
                    !isSameDim(scaleShape->GetDim(1), xShape->GetDim(1)),
                    OPS_LOG_E(
                        context->GetNodeName(),
                        "When quant_mode is %ld in static graph, the second dim of scale should or %ld, but its shape is (%ld).",
                        quantMode, xShape->GetDim(1), scaleShape->GetDim(0)),
                    return ge::GRAPH_FAILED);
            } else {
                OP_CHECK(
                    scaleShape->GetDim(0) != NEG_ONE || (scaleShape->GetDim(1) != NEG_ONE && scaleShape->GetDim(1) != xShape->GetDim(1)),
                    OPS_LOG_E(context->GetNodeName(),
                            "When quant_mode is %ld and scale dim is 2 in dynamic graph, the shape of scale should be (-1, -1) or (-1, %d), but its shape is (%s).",
                            quantMode, xShape->GetDim(1), ops::Shape2String(*scaleShape).c_str()),
                    return ge::GRAPH_FAILED);
            }
        } else {
            OPS_LOG_E(
                context->GetNodeName(),
                "When quant_mode is %ld, the dimNum of scale should be 1(dynamic graph) or 2, but its shape is (%ld).",
                scaleShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputOffsetShape(gert::InferShapeContext *context, 
                                             const gert::Shape *offsetShape, const int64_t expertStart,
                                             const int64_t expertEnd, const int64_t quantMode)
{
    // The shape of offset can be none.
    if (quantMode != QuantMode::STATIC_QUANT) {
        return ge::GRAPH_SUCCESS;
    } else if (nullptr == offsetShape) {
        return ge::GRAPH_FAILED;
    }

    if (offsetShape->GetDimNum() != DIM_ONE) {
        OPS_LOG_E(context->GetNodeName(), "The dimNum of offset should be 1, current shape is (%ld).", offsetShape->GetDimNum());
        return ge::GRAPH_FAILED;
    }
    if (offsetShape->GetDim(0) != NEG_ONE && offsetShape->GetDim(0) != NEG_TWO && !isSameDim(offsetShape->GetDim(0), DIM_ONE)) {
        OPS_LOG_E(context->GetNodeName(),
                  "The shape of offset should be (1,) in static graph or (-2), (-1,) in dynamic graph, current shape is (%s).",
                  ops::Shape2String(*offsetShape).c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputShape(gert::InferShapeContext *context, const gert::Shape *xShape,
                                       const gert::Shape *expertIdxShape, const gert::Shape *scaleShape,
                                       const gert::Shape *offsetShape, const int64_t expertStart,
                                       const int64_t expertEnd, const int64_t quantMode)
{
    // Check the shape of input_x
    if (xShape->GetDimNum() == DIM_ONE) {
        if (xShape->GetDim(0) != ge::UNKNOWN_DIM_NUM) {
            OPS_LOG_E(context->GetNodeName(), "The dynamic dim of x should be -2, current shape is %s.",
                      ops::Shape2String(*xShape).c_str());
            return ge::GRAPH_FAILED;
        }
    } else if (xShape->GetDimNum() != DIM_TWO) {
        OPS_LOG_E(context->GetNodeName(), "The dim of x should be 2 or dynamic, current shape is %s.",
                  ops::Shape2String(*xShape).c_str());
        return ge::GRAPH_FAILED;
    }

    int64_t x_n = xShape->GetDimNum() == DIM_ONE ? NEG_ONE : xShape->GetDim(0);
    int64_t cols = xShape->GetDimNum() == DIM_ONE ? NEG_ONE : xShape->GetDim(1);
    if (x_n < NEG_ONE || cols < NEG_ONE) {
        OPS_LOG_E(context->GetNodeName(), "Invalid x shape, shape is %s.", ops::Shape2String(*xShape).c_str());
        return ge::GRAPH_FAILED;
    }

    // Check the shape of expert_idx
    if (expertIdxShape->GetDimNum() == DIM_ONE) {
        if (expertIdxShape->GetDim(0) != ge::UNKNOWN_DIM_NUM) {
            OPS_LOG_E(context->GetNodeName(), "The dynamic dim of expert_idx should be -2, current shape is %s.",
                      ops::Shape2String(*expertIdxShape).c_str());
            return ge::GRAPH_FAILED;
        }
    } else if (expertIdxShape->GetDimNum() != DIM_TWO) {
        OPS_LOG_E(context->GetNodeName(), "The dim of expert_idx should be 2 or dynamic, current shape is %s.",
                  ops::Shape2String(*expertIdxShape).c_str());
        return ge::GRAPH_FAILED;
    }

    int64_t expert_idx_n = expertIdxShape->GetDimNum() == DIM_ONE ? NEG_ONE : expertIdxShape->GetDim(0);
    int64_t expert_idx_k = expertIdxShape->GetDimNum() == DIM_ONE ? NEG_ONE : expertIdxShape->GetDim(1);
    if (expert_idx_n < NEG_ONE || expert_idx_k < NEG_ONE) {
        OPS_LOG_E(context->GetNodeName(), "Invalid expert_idx shape, shape is %s.",
                  ops::Shape2String(*expertIdxShape).c_str());
        return ge::GRAPH_FAILED;
    }

    if (!isSameDim(x_n, expert_idx_n)) {
        OPS_LOG_E(context->GetNodeName(), "The first dim of x and expert_idx should be same.");
        return ge::GRAPH_FAILED;
    }
    // Check the shape of scale
    if (CheckInputScaleShape(context, xShape, scaleShape, expertStart, expertEnd, quantMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Check the shape of offset
    if (CheckInputOffsetShape(context, offsetShape, expertStart, expertEnd, quantMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static void ShowInputShapeAndAttrInfo(gert::InferShapeContext *context, const gert::Shape *xShape,
                                      const gert::Shape *expertIdxShape, const gert::Shape *scaleShape,
                                      const gert::Shape *offsetShape, const int64_t expertStart,
                                      const int64_t expertEnd, const int64_t quantMode, const int64_t rowIdxType)
{
    // input_x and expert_idx are all required.
    OPS_LOG_D(context->GetNodeName(), "x shape is: %s.", ops::Shape2String(*xShape).c_str());
    OPS_LOG_D(context->GetNodeName(), "expert_idx shape is: %s.", ops::Shape2String(*expertIdxShape).c_str());

    // scale is optional and can be none.
    if (nullptr == scaleShape) {
        OPS_LOG_D(context->GetNodeName(), "scale_shape is: none.");
    } else {
        OPS_LOG_D(context->GetNodeName(), "scale_shape is: %s.", ops::Shape2String(*scaleShape).c_str());
    }

    // offset is optional and can be none.
    OPS_LOG_D(context->GetNodeName(), "Begin print offset_shape.");
    if (nullptr == offsetShape) {
        OPS_LOG_D(context->GetNodeName(), "offset_shape is: none.");
    } else {
        OPS_LOG_D(context->GetNodeName(), "offset_shape is: %s.", ops::Shape2String(*offsetShape).c_str());
    }
    OPS_LOG_D(context->GetNodeName(), "End print offset_shape.");

    // Attrs are all required.
    OPS_LOG_D(context->GetNodeName(), "active_expert_range is: [%ld, %ld).", expertStart, expertEnd);
    OPS_LOG_D(context->GetNodeName(), "quant_mode is: %ld.", quantMode);
    OPS_LOG_D(context->GetNodeName(), "row_Idx_type is: %ld.", rowIdxType);
}

static void ShowOutputShapeInfo(gert::InferShapeContext *context, const gert::Shape *expandedXShape,
                                const gert::Shape *expandedRowIdxShape,
                                const gert::Shape *expertTokenCumsumOrCountShape, const gert::Shape *expandedScaleShape)
{
    OPS_LOG_D(context->GetNodeName(), "expanded_x shape is: %s after infershape.",
	          ops::Shape2String(*expandedXShape).c_str());
    OPS_LOG_D(context->GetNodeName(), "expanded_row_idx shape is: %s after infershape.",
              ops::Shape2String(*expandedRowIdxShape).c_str());
    OPS_LOG_D(context->GetNodeName(), "expert_token_cumsum_or_count shape is: %s after infershape.",
              ops::Shape2String(*expertTokenCumsumOrCountShape).c_str());
    OPS_LOG_D(context->GetNodeName(), "expanded_scale shape is: %s after infershape.",
              ops::Shape2String(*expandedScaleShape).c_str());
}

static ge::graphStatus InferShape4MoeInitRoutingCustom(gert::InferShapeContext *context)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do MoeInitRoutingCustomInfershape.");
    // 1. Get and check input shape
    // 1.1 Get and check input_x
    const gert::Shape *xShape = context->GetInputShape(MOE_INIT_ROUTING_CUSTOM_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // 1.2 Get and check expert_idx
    const gert::Shape *expertIdxShape = context->GetInputShape(MOE_INIT_ROUTING_CUSTOM_INPUT_EXPERT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertIdxShape);

    // 1.3 Get scale shape without checking null, because scale is optional and can be none.
    const gert::Shape *scaleShape = context->GetOptionalInputShape(MOE_INIT_ROUTING_CUSTOM_INPUT_SCALE);

    // 1.4 Get offset shape without checking null, because offset is optional and can be none.
    const gert::Shape *offsetShape = context->GetOptionalInputShape(MOE_INIT_ROUTING_CUSTOM_INPUT_OFFSET);
    // 2. Get and check attrs
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    // 2.1 Get and check expert_num attr
    int64_t experNum = static_cast<int64_t>(-1);
    if (GetAndCheckAttrExpertNum(attrs, context, experNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 2.2 Get and check active_expert_range attr
    int64_t expertStart = static_cast<int64_t>(-1);
    int64_t expertEnd = static_cast<int64_t>(-1);
    if (GetAndCheckAttrActiveExpertRange(attrs, context, expertStart, expertEnd, experNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (nullptr == attrs) {
        OPS_LOG_E(context->GetNodeName(), "The attrs is none.");
        return ge::GRAPH_FAILED;
    }

    // 2.3 Get and check drop_pad_mode attr
    int64_t dropPadMode = static_cast<int64_t>(-1);
    if (GetAndCheckAttrDropPadMode(attrs, context, dropPadMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 2.4 Get and check active_num attr
    int64_t activeNum = static_cast<int64_t>(-1);
    if (GetAndCheckAttrActiveNum(attrs, context, activeNum, dropPadMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 2.5 Get and check expert_capacity attr
    int64_t expertCapacity = static_cast<int64_t>(-1);
    if (GetAndCheckAttrExpertCapacity(attrs, context, xShape, expertCapacity, dropPadMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 2.6 Get and check expert_token_num_type attr
    int64_t expertTokenNumType = static_cast<int64_t>(-1);
    if (GetAndCheckAttrExpertTokenNumType(attrs, context, expertTokenNumType) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 2.7 Get and check expert_token_num_type attr
    bool expertTokenNumFlag = false;
    if (GetAndCheckAttrExpertTokenNumFlag(attrs, context, expertTokenNumFlag) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 2.8 Get and check quant_mode attr
    int64_t quantMode = static_cast<int64_t>(-1);
    if (GetAndCheckAttrQuantMode(attrs, context, quantMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 2.9 Get and check row_Idx_type attr
    int64_t rowIdxType = static_cast<int64_t>(-1);
    if (GetAndCheckAttrRowIdxType(attrs, context, rowIdxType, dropPadMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Check input shape
    if (CheckInputShape(context, xShape, expertIdxShape, scaleShape, offsetShape, expertStart, expertEnd, quantMode) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 3. Infer output shape
    // 3.1 Prepare output shape
    gert::Shape *expandedXShape = context->GetOutputShape(MOE_INIT_ROUTING_CUSTOM_OUTPUT_EXPANDED_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, expandedXShape);
    gert::Shape *expandedRowIdxShape = context->GetOutputShape(MOE_INIT_ROUTING_CUSTOM_OUTPUT_EXPANDED_ROW_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, expandedRowIdxShape);
    gert::Shape *expertTokenCumsumOrCountShape =
        context->GetOutputShape(MOE_INIT_ROUTING_CUSTOM_OUTPUT_EXPERT_TOKEN_CUMSUM_OR_COUNT);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertTokenCumsumOrCountShape);
    gert::Shape *expandedScaleShape = context->GetOutputShape(MOE_INIT_ROUTING_CUSTOM_OUTPUT_EXPANDED_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context, expandedScaleShape);

    int64_t x_n = xShape->GetDimNum() == DIM_ONE ? NEG_ONE : xShape->GetDim(0);
    int64_t cols = xShape->GetDimNum() == DIM_ONE ? NEG_ONE : xShape->GetDim(1);

    int64_t expert_idx_n = expertIdxShape->GetDimNum() == DIM_ONE ? NEG_ONE : expertIdxShape->GetDim(0);
    int64_t k = expertIdxShape->GetDimNum() == DIM_ONE ? NEG_ONE : expertIdxShape->GetDim(1);
    int64_t n = x_n > expert_idx_n ? x_n : expert_idx_n;
    if (activeNum == 0 || activeNum == -1) {
        activeNum = n * k;
    } else {
        activeNum = std::min(activeNum, n * k);
    }

    int64_t xOutDimNum = activeNum < n * k ? activeNum : n * k;
    int64_t outNum = (n == NEG_ONE || k == NEG_ONE) ? NEG_ONE : n * k;
    int64_t xOutNum = (n == NEG_ONE || k == NEG_ONE) ? NEG_ONE : xOutDimNum;
    // 3.2 Set output expanded_x shape
    if (dropPadMode == DropPadMode::NO_DROP_PAD) {
        expandedXShape->SetDimNum(DIM_TWO);
        expandedXShape->SetDim(0U, xOutNum);
        expandedXShape->SetDim(DIM_ONE, cols);
    } else {
        expandedXShape->SetDimNum(DIM_THREE);
        expandedXShape->SetDim(0U, experNum);
        expandedXShape->SetDim(DIM_ONE, expertCapacity);
        expandedXShape->SetDim(DIM_TWO, cols);
    }

    // 3.3 Set output expanded_row_idx shape
    expandedRowIdxShape->SetDimNum(DIM_ONE);
    expandedRowIdxShape->SetDim(0U, outNum);

    // 3.4 Set output expert_token_cumsum_or_count shape
    if (expertTokenNumFlag) {
        if (expertTokenNumType == ExpertTokenNumType::KEY_VALUE) {
            expertTokenCumsumOrCountShape->SetDimNum(DIM_TWO);
            expertTokenCumsumOrCountShape->SetDim(0U, experNum);
            expertTokenCumsumOrCountShape->SetDim(DIM_ONE, KEY_VALUE_MODE_DIM0_NUM);
        } else {
            expertTokenCumsumOrCountShape->SetDimNum(DIM_ONE);
            expertTokenCumsumOrCountShape->SetDim(0U, expertEnd - expertStart);
        }
    }

    // 3.5 Set output expanded_scale shape
    // When scale_shape=(b*s) and non-quant, or it is dynamic quant mode, the shape of expanded_scale should be (b*s*k)
    if (QuantMode::NON_QUANT == quantMode || QuantMode::DYNAMIC_QUANT == quantMode) {
        expandedScaleShape->SetDimNum(DIM_ONE);
        if (dropPadMode == DropPadMode::NO_DROP_PAD) {
            expandedScaleShape->SetDim(0U, xOutNum);
        } else {
            expandedScaleShape->SetDim(0U, experNum * expertCapacity);
        }
    }

    ShowOutputShapeInfo(context, expandedXShape, expandedRowIdxShape, expertTokenCumsumOrCountShape,
                        expandedScaleShape);
    OPS_LOG_D(context->GetNodeName(), "End to do MoeInitRoutingCustomInfershape.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MoeInitRoutingCustom(gert::InferDataTypeContext *context)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do MoeInitRoutingCustomInferDataType.");

    // Get and check quant_mode attr
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    int64_t quantMode = static_cast<int64_t>(-1);
    const int64_t *quantModePtr = attrs->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_CUSTOM_ATTR_QUANT_MODE);
    if (nullptr == quantModePtr) {
        OPS_LOG_E(context->GetNodeName(), "The quant_mode should be %d, %d or %d. But it is none.", QuantMode::NON_QUANT,
                QuantMode::STATIC_QUANT, QuantMode::DYNAMIC_QUANT);
        return ge::GRAPH_FAILED;
    }
    quantMode = *quantModePtr;
    // Infer output dtype according quant_mode
    auto xDtype = context->GetInputDataType(MOE_INIT_ROUTING_CUSTOM_INPUT_X);
    if (QuantMode::NON_QUANT == quantMode) {
        context->SetOutputDataType(MOE_INIT_ROUTING_CUSTOM_OUTPUT_EXPANDED_X, xDtype);
    } else if (QuantMode::STATIC_QUANT == quantMode || QuantMode::DYNAMIC_QUANT == quantMode) {
        if (ge::DT_INT8 == xDtype) {
            OPS_LOG_E(context->GetNodeName(), "When quant_mode=%ld, xDtype cannot be int_8.", quantMode);
            return ge::GRAPH_FAILED;
        }
        context->SetOutputDataType(MOE_INIT_ROUTING_CUSTOM_OUTPUT_EXPANDED_X, ge::DT_INT8);
    }
    context->SetOutputDataType(MOE_INIT_ROUTING_CUSTOM_OUTPUT_EXPANDED_ROW_IDX, ge::DT_INT32);
    context->SetOutputDataType(MOE_INIT_ROUTING_CUSTOM_OUTPUT_EXPERT_TOKEN_CUMSUM_OR_COUNT, ge::DT_INT64);
    context->SetOutputDataType(MOE_INIT_ROUTING_CUSTOM_OUTPUT_EXPANDED_SCALE, ge::DT_FLOAT);
    OPS_LOG_D(context->GetNodeName(), "End to do MoeInitRoutingCustomInferDataType.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeRange4MoeInitRoutingCustom(gert::InferShapeRangeContext *context)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do MoeInitRoutingCustomInferRange.");

    // Get and check the pointers of all the outputs' shape range object
    auto expanded_x = context->GetOutputShapeRange(MOE_INIT_ROUTING_CUSTOM_OUTPUT_EXPANDED_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, expanded_x);
    auto expanded_row_idx = context->GetOutputShapeRange(MOE_INIT_ROUTING_CUSTOM_OUTPUT_EXPANDED_ROW_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, expanded_row_idx);
    auto count = context->GetOutputShapeRange(MOE_INIT_ROUTING_CUSTOM_OUTPUT_EXPERT_TOKEN_CUMSUM_OR_COUNT);
    OP_CHECK_NULL_WITH_CONTEXT(context, count);
    auto expanded_scale = context->GetOutputShapeRange(MOE_INIT_ROUTING_CUSTOM_OUTPUT_EXPANDED_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context, expanded_scale);

    // Print the shape ranges of the outputs before InferShapeRange
    OPS_LOG_D(context->GetNodeName(), "Before InferShapeRange, expanded_x->GetMin() = %s",
              ops::Shape2String(*(expanded_x->GetMin())).c_str());
    OPS_LOG_D(context->GetNodeName(), "Before InferShapeRange, expanded_x->GetMax() = %s",
              ops::Shape2String(*(expanded_x->GetMax())).c_str());

    OPS_LOG_D(context->GetNodeName(), "Before InferShapeRange, expanded_row_idx->GetMin() = %s",
              ops::Shape2String(*(expanded_row_idx->GetMin())).c_str());
    OPS_LOG_D(context->GetNodeName(), "Before InferShapeRange, expanded_row_idx->GetMax() = %s",
              ops::Shape2String(*(expanded_row_idx->GetMax())).c_str());

    OPS_LOG_D(context->GetNodeName(), "Before InferShapeRange, count->GetMin() = %s",
              ops::Shape2String(*(count->GetMin())).c_str());
    OPS_LOG_D(context->GetNodeName(), "Before InferShapeRange, count->GetMax() = %s",
              ops::Shape2String(*(count->GetMax())).c_str());

    OPS_LOG_D(context->GetNodeName(), "Before InferShapeRange, expanded_scale->GetMin() = %s",
              ops::Shape2String(*(expanded_scale->GetMin())).c_str());
    OPS_LOG_D(context->GetNodeName(), "Before InferShapeRange, expanded_scale->GetMax() = %s",
              ops::Shape2String(*(expanded_scale->GetMax())).c_str());

    // Set the dim num and dim of the outputs' shape range object
    if (expanded_x->GetMin() != nullptr && expanded_x->GetMax() != nullptr) {
        expanded_x->GetMin()->SetDimNum(DIM_TWO);
        expanded_x->GetMax()->SetDimNum(DIM_TWO);
        for (size_t i = 0; i < DIM_TWO; i++) {
            expanded_x->GetMin()->SetDim(i, 0);
            expanded_x->GetMax()->SetDim(i, -1);
        }
    }

    if (expanded_row_idx->GetMin() != nullptr && expanded_row_idx->GetMax() != nullptr) {
        expanded_row_idx->GetMin()->SetDimNum(DIM_ONE);
        expanded_row_idx->GetMax()->SetDimNum(DIM_ONE);
        expanded_row_idx->GetMin()->SetDim(0, 0);
        expanded_row_idx->GetMax()->SetDim(0, -1);
    }

    if (count->GetMin() != nullptr && count->GetMax() != nullptr) {
        count->GetMin()->SetDimNum(DIM_ONE);
        count->GetMax()->SetDimNum(DIM_ONE);
        count->GetMin()->SetDim(0, 0);
        count->GetMax()->SetDim(0, -1);
    }

    if (expanded_scale->GetMin() != nullptr && expanded_scale->GetMax() != nullptr) {
        expanded_scale->GetMin()->SetDimNum(DIM_ONE);
        expanded_scale->GetMax()->SetDimNum(DIM_ONE);
        expanded_scale->GetMin()->SetDim(0, 0);
        expanded_scale->GetMax()->SetDim(0, -1);
    }

    // Print the shape ranges of the outputs after InferShapeRange
    OPS_LOG_D(context->GetNodeName(), "After InferShapeRange, expanded_x->GetMin() = %s",
              ops::Shape2String(*(expanded_x->GetMin())).c_str());
    OPS_LOG_D(context->GetNodeName(), "After InferShapeRange, expanded_x->GetMax() = %s",
              ops::Shape2String(*(expanded_x->GetMax())).c_str());

    OPS_LOG_D(context->GetNodeName(), "After InferShapeRange, expanded_row_idx->GetMin() = %s",
              ops::Shape2String(*(expanded_row_idx->GetMin())).c_str());
    OPS_LOG_D(context->GetNodeName(), "After InferShapeRange, expanded_row_idx->GetMax() = %s",
              ops::Shape2String(*(expanded_row_idx->GetMax())).c_str());

    OPS_LOG_D(context->GetNodeName(), "After InferShapeRange, count->GetMin() = %s",
              ops::Shape2String(*(count->GetMin())).c_str());
    OPS_LOG_D(context->GetNodeName(), "After InferShapeRange, count->GetMax() = %s",
              ops::Shape2String(*(count->GetMax())).c_str());

    OPS_LOG_D(context->GetNodeName(), "After InferShapeRange, expanded_scale->GetMin() = %s",
              ops::Shape2String(*(expanded_scale->GetMin())).c_str());
    OPS_LOG_D(context->GetNodeName(), "After InferShapeRange, expanded_scale->GetMax() = %s",
              ops::Shape2String(*(expanded_scale->GetMax())).c_str());

    OPS_LOG_D(context->GetNodeName(), "End to do MoeInitRoutingCustomInferRange.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeInitRoutingCustom)
    .InferShape(InferShape4MoeInitRoutingCustom)
    .InferDataType(InferDataType4MoeInitRoutingCustom)
    .InferShapeRange(InferShapeRange4MoeInitRoutingCustom);
} // namespace ops