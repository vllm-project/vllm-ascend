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
 * \file scatter_pa_kv_cache_vllm_infershape.cpp
 * \brief
 */

#include <exe_graph/runtime/infer_shape_context.h>
#include <register/op_impl_registry.h>
#include "../../causal_conv1d/op_host/error_log.h"

using namespace ge;

namespace ops {

constexpr int64_t INPUT_KEY = 0;
constexpr int64_t INPUT_KEY_CACHE = 1;
constexpr int64_t INPUT_SLOT_MAPPING = 2;
constexpr int64_t INPUT_VALUE = 3;
constexpr int64_t INPUT_VALUE_CACHE = 4;
constexpr int64_t ATTR_CACHE_MODE_INDEX = 0;
constexpr int64_t INPUT_SCATTER_MODE_INDEX = 1;
constexpr int64_t OPTIONAL_INPUT_COMPRESS_LENS = 5;
constexpr int64_t OPTIONAL_INPUT_COMPRESS_SEQ_OFFSET = 6;
constexpr int64_t OPTIONAL_INPUT_SEQ_LENS = 7;

constexpr int64_t OUTPUT_KEY_CACHE = 0;
constexpr int64_t OUTPUT_VALUE_CACHE = 1;

ge::graphStatus InferShape4ScatterPaKvCacheVllm(gert::InferShapeContext *context)
{
    // inferShape key_cache
    OP_LOGD(context, "Begin to do InferShape4ScatterPaKvCacheVllm");
    auto inputKeyCacheShape = context->GetInputShape(INPUT_KEY_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputKeyCacheShape);
    auto outputKeyCacheShape = context->GetOutputShape(OUTPUT_KEY_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputKeyCacheShape);
    *outputKeyCacheShape = *inputKeyCacheShape;

    // inferShape value_cache
    auto inputValueCacheShape = context->GetInputShape(INPUT_VALUE_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputValueCacheShape);
    auto outputValueCacheShape = context->GetOutputShape(OUTPUT_VALUE_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputValueCacheShape);
    *outputValueCacheShape = *inputValueCacheShape;
    OP_LOGD(context, "End to do InferShape4ScatterPaKvCacheVllm");
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataType4ScatterPaKvCacheVllm(gert::InferDataTypeContext *context)
{
    OP_LOGD(context, "Begin to do InferDataType4ScatterPaKvCacheVllm");
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto cache_mode = attrs->GetAttrPointer<char>(ATTR_CACHE_MODE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, cache_mode);
    OP_CHECK_IF(strcmp(cache_mode, "") != 0 && strcmp(cache_mode, "Norm") != 0 && strcmp(cache_mode, "PA_NZ") != 0,
                OP_LOGE(context, "invalid cache mode : %s", cache_mode), return ge::GRAPH_FAILED);
    auto scatter_mode = attrs->GetAttrPointer<char>(INPUT_SCATTER_MODE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, scatter_mode);
    OP_CHECK_IF(strcmp(scatter_mode, "") != 0 && strcmp(scatter_mode, "None") != 0 &&
                    strcmp(scatter_mode, "Alibi") != 0 && strcmp(scatter_mode, "Rope") != 0 &&
                    strcmp(scatter_mode, "Omni") != 0 && strcmp(scatter_mode, "Nct") != 0,
                OP_LOGE(context, "invalid scatter mode : %s", scatter_mode), return ge::GRAPH_FAILED);

    context->SetOutputDataType(OUTPUT_KEY_CACHE, context->GetInputDataType(INPUT_KEY));
    context->SetOutputDataType(OUTPUT_VALUE_CACHE, context->GetInputDataType(INPUT_VALUE));
    OP_LOGD(context, "End to do InferDataType4ScatterPaKvCacheVllm");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ScatterPaKvCacheVllm)
    .InferShape(InferShape4ScatterPaKvCacheVllm)
    .InferDataType(InferDataType4ScatterPaKvCacheVllm);
} // namespace ops