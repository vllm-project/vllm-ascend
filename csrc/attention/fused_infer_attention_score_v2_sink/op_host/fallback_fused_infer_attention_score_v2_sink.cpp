/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include "log/ops_log.h"
#include "../../fused_infer_attention_score_v2_sink_common/omni_common/include/fallback/fallback_comm.h"
#include "../../fused_infer_attention_score_v2_sink_common/omni_common/include/fallback/fallback.h"

#ifdef __cplusplus
extern "C" {
#endif
namespace fallback {

using namespace ge;
using namespace gert;
static const size_t QUERY_INDEX = 0;
static const size_t KEY_INDEX = 1;
static const size_t VALUE_INDEX = 2;
static const size_t PSE_SHIFT_INDEX = 3;
static const size_t ATTEN_MASK_INDEX = 4;
static const size_t ACTUAL_SEQ_Q_INDEX = 5;
static const size_t ACTUAL_SEQ_KV_INDEX = 6;
static const size_t DEQUANT_SCALE1_INDEX = 7;
static const size_t QUANT_SCALE1_INDEX = 8;
static const size_t DEQUANT_SCALE2_INDEX = 9;
static const size_t QUANT_SCALE2_INDEX = 10;
static const size_t QUANT_OFFSET2_INDEX = 11;
static const size_t ANTIQUANT_SCALE_INDEX = 12;
static const size_t ANTIQUANT_OFFSET_INDEX = 13;
static const size_t BLOCK_TABLE_INDEX = 14;
static const size_t QUERY_PADDING_INDEX = 15;
static const size_t KV_PADDING_INDEX = 16;
static const size_t KEY_ANTIQUANT_SCALE_INDEX = 17;
static const size_t KEY_ANTIQUANT_OFFSET_INDEX = 18;
static const size_t VALUE_ANTIQUANT_SCALE_INDEX = 19;
static const size_t VALUE_ANTIQUANT_OFFSET_INDEX = 20;
static const size_t QUERY_ROPE_INDEX = 21;
static const size_t KEY_ROPE_INDEX = 22;
static const size_t KEY_ROPE_ANTIQUANT_SCALE_INDEX = 23;
static const size_t DEQUANT_SCALE_QUERY_INDEX = 24;
static const size_t LEARNABLE_SINK_INDEX = 25;
static const size_t KEY_SINK_INDEX = 26;
static const size_t KEY_ROPE_SINK_INDEX = 27;
static const size_t VALUE_SINK_INDEX = 28;

static const size_t ATTR_N_INDEX = 0;
static const size_t ATTR_SCALE_INDEX = 1;
static const size_t ATTR_PRE_TOKEN_INDEX = 2;
static const size_t ATTR_NEXT_TOKEN_INDEX = 3;
static const size_t ATTR_INPUT_LAYOUT_INDEX = 4;
static const size_t ATTR_NUM_KV_HEADS_INDEX = 5;
static const size_t ATTR_SPARSE_MODE_INDEX = 6;
static const size_t ATTR_INNER_PRECISE_INDEX = 7;
static const size_t ATTR_BLOCK_SIZE_INDEX = 8;
static const size_t ATTR_ANTIQUANT_MODE_INDEX = 9;
static const size_t ATTR_SOFTMAX_LSE_FLAG_INDEX = 10;
static const size_t ATTR_KEY_ANTIQUANT_MODE_INDEX = 11;
static const size_t ATTR_VALUE_ANTIQUANT_MODE_INDEX = 12;
static const size_t ATTR_QUERY_QUANT_MODE_INDEX = 13;
static const size_t ATTR_SINK_NUMBER_INDEX = 15;

static const size_t ATTENTION_OUT_INDEX = 0;
static const size_t SOFTMAX_LSE_INDEX = 1;

static constexpr int64_t SPARSE_MODE_INT_MAX = 2147483647;

struct FusedInferHostTensorParams {
    const gert::Tensor *query = nullptr;
    const gert::Tensor *key = nullptr;
    const gert::Tensor *value = nullptr;
    const gert::Tensor *output = nullptr;
    const gert::Tensor *softmaxLse = nullptr;
    const gert::Tensor *pseShiftGe = nullptr;
    const gert::Tensor *attenMaskGe = nullptr;
    const gert::Tensor *actualSeqLengthsGe = nullptr;
    const gert::Tensor *actualSeqLengthsGeKv = nullptr;
    const gert::Tensor *deqScale1 = nullptr;
    const gert::Tensor *quantScale1 = nullptr;
    const gert::Tensor *deqScale2 = nullptr;
    const gert::Tensor *quantScale2 = nullptr;
    const gert::Tensor *quantOffset2 = nullptr;
    const gert::Tensor *antiquantScaleGe = nullptr;
    const gert::Tensor *antiquantOffsetGe = nullptr;
    const gert::Tensor *blocktableGe = nullptr;
    const gert::Tensor *queryPaddingGe = nullptr;
    const gert::Tensor *kvPaddingGe = nullptr;
    const gert::Tensor *keyAntiquantScaleGe = nullptr;
    const gert::Tensor *keyAntiquantOffsetGe = nullptr;
    const gert::Tensor *valueAntiquantScaleGe = nullptr;
    const gert::Tensor *valueAntiquantOffsetGe = nullptr;
    const gert::Tensor *queryRopeGe = nullptr;
    const gert::Tensor *keyRopeGe = nullptr;
    const gert::Tensor *keyRopeAntiquantScaleGe = nullptr;
    const gert::Tensor *dequantScaleQueryGe = nullptr;
    const gert::Tensor *learnableSinkGe = nullptr;
    const gert::Tensor *keySinkGe = nullptr;
    const gert::Tensor *keyRopeSinkGe = nullptr;
    const gert::Tensor *valueSinkGe = nullptr;
};

static graphStatus FiaFillTensorParams(const OpExecuteContext *host_api_ctx, FusedInferHostTensorParams &fiaTensors)
{
    fiaTensors.query = host_api_ctx->GetInputTensor(QUERY_INDEX);
    OPS_CHECK(
        fiaTensors.query == nullptr, OPS_LOG_E(host_api_ctx->GetNodeName(), "query is null"), return GRAPH_FAILED);

    fiaTensors.key = host_api_ctx->GetDynamicInputTensor(KEY_INDEX, 0);
    OPS_CHECK(fiaTensors.key == nullptr, OPS_LOG_E(host_api_ctx->GetNodeName(), "key is null"), return GRAPH_FAILED);

    fiaTensors.value = host_api_ctx->GetDynamicInputTensor(VALUE_INDEX, 0);
    OPS_CHECK(
        fiaTensors.value == nullptr, OPS_LOG_E(host_api_ctx->GetNodeName(), "value is null"), return GRAPH_FAILED);

    fiaTensors.output = host_api_ctx->GetOutputTensor(ATTENTION_OUT_INDEX);
    OPS_CHECK(
        fiaTensors.output == nullptr, OPS_LOG_E(host_api_ctx->GetNodeName(), "output is null"), return GRAPH_FAILED);

    fiaTensors.softmaxLse = host_api_ctx->GetOutputTensor(SOFTMAX_LSE_INDEX);
    OPS_CHECK(fiaTensors.softmaxLse == nullptr,
              OPS_LOG_E(host_api_ctx->GetNodeName(), "softmaxLse is null"),
              return GRAPH_FAILED);

    fiaTensors.pseShiftGe = host_api_ctx->GetOptionalInputTensor(PSE_SHIFT_INDEX);
    fiaTensors.attenMaskGe = host_api_ctx->GetOptionalInputTensor(ATTEN_MASK_INDEX);
    fiaTensors.actualSeqLengthsGe = host_api_ctx->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    fiaTensors.actualSeqLengthsGeKv = host_api_ctx->GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    fiaTensors.deqScale1 = host_api_ctx->GetOptionalInputTensor(DEQUANT_SCALE1_INDEX);
    fiaTensors.quantScale1 = host_api_ctx->GetOptionalInputTensor(QUANT_SCALE1_INDEX);
    fiaTensors.deqScale2 = host_api_ctx->GetOptionalInputTensor(DEQUANT_SCALE2_INDEX);
    fiaTensors.quantScale2 = host_api_ctx->GetOptionalInputTensor(QUANT_SCALE2_INDEX);
    fiaTensors.quantOffset2 = host_api_ctx->GetOptionalInputTensor(QUANT_OFFSET2_INDEX);
    fiaTensors.antiquantScaleGe = host_api_ctx->GetOptionalInputTensor(ANTIQUANT_SCALE_INDEX);
    fiaTensors.antiquantOffsetGe = host_api_ctx->GetOptionalInputTensor(ANTIQUANT_OFFSET_INDEX);
    fiaTensors.blocktableGe = host_api_ctx->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    fiaTensors.queryPaddingGe = host_api_ctx->GetOptionalInputTensor(QUERY_PADDING_INDEX);
    fiaTensors.kvPaddingGe = host_api_ctx->GetOptionalInputTensor(KV_PADDING_INDEX);
    fiaTensors.keyAntiquantScaleGe = host_api_ctx->GetOptionalInputTensor(KEY_ANTIQUANT_SCALE_INDEX);
    fiaTensors.keyAntiquantOffsetGe = host_api_ctx->GetOptionalInputTensor(KEY_ANTIQUANT_OFFSET_INDEX);
    fiaTensors.valueAntiquantScaleGe = host_api_ctx->GetOptionalInputTensor(VALUE_ANTIQUANT_SCALE_INDEX);
    fiaTensors.valueAntiquantOffsetGe = host_api_ctx->GetOptionalInputTensor(VALUE_ANTIQUANT_OFFSET_INDEX);
    fiaTensors.queryRopeGe = host_api_ctx->GetOptionalInputTensor(QUERY_ROPE_INDEX);
    fiaTensors.keyRopeGe = host_api_ctx->GetOptionalInputTensor(KEY_ROPE_INDEX);
    fiaTensors.keyRopeAntiquantScaleGe = host_api_ctx->GetOptionalInputTensor(KEY_ROPE_ANTIQUANT_SCALE_INDEX);
    fiaTensors.dequantScaleQueryGe = host_api_ctx->GetOptionalInputTensor(DEQUANT_SCALE_QUERY_INDEX);
    fiaTensors.learnableSinkGe = host_api_ctx->GetOptionalInputTensor(LEARNABLE_SINK_INDEX);
    fiaTensors.keySinkGe = host_api_ctx->GetOptionalInputTensor(KEY_SINK_INDEX);
    fiaTensors.keyRopeSinkGe = host_api_ctx->GetOptionalInputTensor(KEY_ROPE_SINK_INDEX);
    fiaTensors.valueSinkGe = host_api_ctx->GetOptionalInputTensor(VALUE_SINK_INDEX);
    return GRAPH_SUCCESS;
}

struct ActualSeqInfo {
    std::vector<int64_t> actSeqArray;
    std::vector<int64_t> actSeqArrayKv;
};

static void FillActualSeqInfo(const FusedInferHostTensorParams &fiaTensors, ActualSeqInfo &actualSeqInfo)
{
    if (fiaTensors.actualSeqLengthsGe != nullptr) {
        const int64_t *actSeqData = fiaTensors.actualSeqLengthsGe->GetData<int64_t>();
        const size_t len = static_cast<size_t>(fiaTensors.actualSeqLengthsGe->GetShapeSize());
        for (size_t i = 0; i < len; i++) {
            actualSeqInfo.actSeqArray.push_back(actSeqData[i]);
        }
    }

    if (fiaTensors.actualSeqLengthsGeKv != nullptr) {
        const int64_t *actSeqData = fiaTensors.actualSeqLengthsGeKv->GetData<int64_t>();
        const size_t len = static_cast<size_t>(fiaTensors.actualSeqLengthsGeKv->GetShapeSize());
        for (size_t i = 0; i < len; i++) {
            actualSeqInfo.actSeqArrayKv.push_back(actSeqData[i]);
        }
    }
}

struct FusedInferHostAttrPtrs {
    const uint32_t *getNumHeads = nullptr;
    const float *scaleValue = nullptr;
    const int64_t *getPreTokens = nullptr;
    const int64_t *getNextTokens = nullptr;
    const char *layout = nullptr;
    const uint32_t *getKVHeadNum = nullptr;
    const uint32_t *getSparseMode = nullptr;
    const uint32_t *getInnerPrecise = nullptr;
    const uint32_t *getBlockSize = nullptr;
    const uint32_t *getAntiquantMode = nullptr;
    const bool *getSoftmaxLseFlag = nullptr;
    const uint32_t *getKeyAntiquantMode = nullptr;
    const uint32_t *getValueAntiquantMode = nullptr;
    const uint32_t *getQueryQuantMode = nullptr;
    const int64_t *getSinkNumber = nullptr;
};

static void FillAttrPointers(const gert::RuntimeAttrs *attrs, FusedInferHostAttrPtrs &attrPtrs)
{
    attrPtrs.getNumHeads = attrs->GetAttrPointer<uint32_t>(ATTR_N_INDEX);
    attrPtrs.scaleValue = attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX);
    attrPtrs.getPreTokens = attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKEN_INDEX);
    attrPtrs.getNextTokens = attrs->GetAttrPointer<int64_t>(ATTR_NEXT_TOKEN_INDEX);
    attrPtrs.layout = attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX);
    attrPtrs.getKVHeadNum = attrs->GetAttrPointer<uint32_t>(ATTR_NUM_KV_HEADS_INDEX);
    attrPtrs.getSparseMode = attrs->GetAttrPointer<uint32_t>(ATTR_SPARSE_MODE_INDEX);
    attrPtrs.getInnerPrecise = attrs->GetAttrPointer<uint32_t>(ATTR_INNER_PRECISE_INDEX);
    attrPtrs.getBlockSize = attrs->GetAttrPointer<uint32_t>(ATTR_BLOCK_SIZE_INDEX);
    attrPtrs.getAntiquantMode = attrs->GetAttrPointer<uint32_t>(ATTR_ANTIQUANT_MODE_INDEX);
    attrPtrs.getSoftmaxLseFlag = attrs->GetAttrPointer<bool>(ATTR_SOFTMAX_LSE_FLAG_INDEX);
    attrPtrs.getKeyAntiquantMode = attrs->GetAttrPointer<uint32_t>(ATTR_KEY_ANTIQUANT_MODE_INDEX);
    attrPtrs.getValueAntiquantMode = attrs->GetAttrPointer<uint32_t>(ATTR_VALUE_ANTIQUANT_MODE_INDEX);
    attrPtrs.getQueryQuantMode = attrs->GetAttrPointer<uint32_t>(ATTR_QUERY_QUANT_MODE_INDEX);
    attrPtrs.getSinkNumber = attrs->GetAttrPointer<int64_t>(ATTR_SINK_NUMBER_INDEX);
}

struct FusedInferHostScalarParams {
    int64_t numHeads = 1;
    double dScaleValue = 1.0;
    int64_t preTokens = SPARSE_MODE_INT_MAX;
    int64_t nextTokens = SPARSE_MODE_INT_MAX;
    int64_t kvHeadNum = 0;
    int64_t sparseMode = 0;
    int64_t innerPrecise = 1;
    int64_t blockSize = 0;
    int64_t antiquantMode = 0;
    bool softmaxLseFlag = false;
    int64_t keyAntiquantMode = 0;
    int64_t valueAntiquantMode = 0;
    int64_t queryQuantMode = 0;
    int64_t sinkNumber = 0;
};

static void GetFusedInferHostScalarParams(const FusedInferHostAttrPtrs &attrPointers,
                                          FusedInferHostScalarParams &params)
{
    params.numHeads = *(attrPointers.getNumHeads);
    params.dScaleValue = *(attrPointers.scaleValue);
    params.preTokens = *(attrPointers.getPreTokens);
    params.nextTokens = *(attrPointers.getNextTokens);
    params.kvHeadNum = *(attrPointers.getKVHeadNum);
    params.sparseMode = *(attrPointers.getSparseMode);
    params.innerPrecise = *(attrPointers.getInnerPrecise);
    params.blockSize = *(attrPointers.getBlockSize);
    params.antiquantMode = *(attrPointers.getAntiquantMode);
    params.softmaxLseFlag = *(attrPointers.getSoftmaxLseFlag);
    params.keyAntiquantMode = *(attrPointers.getKeyAntiquantMode);
    params.valueAntiquantMode = *(attrPointers.getValueAntiquantMode);
    params.queryQuantMode = *(attrPointers.getQueryQuantMode);
    params.sinkNumber = *(attrPointers.getSinkNumber);
}

static graphStatus FusedInferHostExecuteFunc(OpExecuteContext *host_api_ctx)
{
    OPS_CHECK(host_api_ctx == nullptr, OPS_LOG_E("aclnnfallback", "host_api_ctx is null"), return GRAPH_FAILED);
    FusedInferHostTensorParams fiaTensors{};
    auto apiRet = FiaFillTensorParams(host_api_ctx, fiaTensors);
    OPS_CHECK(apiRet != GRAPH_SUCCESS, OPS_LOG_E("aclnnfallback", "Input Tensors invalid"), return GRAPH_FAILED);

    std::vector<const gert::Tensor *> ge_tenserListKey;
    ge_tenserListKey.push_back(fiaTensors.key);

    std::vector<const gert::Tensor *> ge_tenserListValue;
    ge_tenserListValue.push_back(fiaTensors.value);

    ActualSeqInfo actualSeqInfo{};
    FillActualSeqInfo(fiaTensors, actualSeqInfo);

    FusedInferHostAttrPtrs attrPointers{};
    auto attrs = host_api_ctx->GetAttrs();
    OPS_CHECK(attrs == nullptr, OPS_LOG_E("aclnnfallback", "host_api_ctx Attrs is null"), return GRAPH_FAILED);
    FillAttrPointers(attrs, attrPointers);

    FusedInferHostScalarParams sclarParams{};
    GetFusedInferHostScalarParams(attrPointers, sclarParams);

    if (sclarParams.innerPrecise < 0 ||
        sclarParams.innerPrecise >
            3) { // innerPrecise=2,3 corresponds to rows with invalid high precision and high performance
        OPS_LOG_E(
            host_api_ctx->GetNodeName(), "invalid innerPrecise(%ld). Only support 0~3 now.", sclarParams.innerPrecise);
        return GRAPH_FAILED;
    }
    OPS_LOG_I(host_api_ctx->GetNodeName(),
              "FusedInferAttentionScoreV2Sink fallback begin, numHeads = %ld, dScaleValue = %lf",
              sclarParams.numHeads,
              sclarParams.dScaleValue);
    OPS_LOG_I(host_api_ctx->GetNodeName(),
              "preTokens = %ld, nextTokens = %ld, kvHeadNum = %ld, sparseMode = %ld, innerPrecise = %ld",
              sclarParams.preTokens,
              sclarParams.nextTokens,
              sclarParams.kvHeadNum,
              sclarParams.sparseMode,
              sclarParams.innerPrecise);

    if (sclarParams.sparseMode >= 10 && sclarParams.sparseMode <= 14) { // 10: min  14: max
        sclarParams.innerPrecise = 0;
        sclarParams.sparseMode -= 10; // subtract 10 to modify sparseMode
        OPS_LOG_I(host_api_ctx->GetNodeName(),
                  "because sparseMode in range [10, 14], after modification, sparseMode = %ld, innerPrecise = %ld.",
                  sclarParams.sparseMode,
                  sclarParams.innerPrecise);
    }

    apiRet = EXEC_OPAPI_CMD(aclnnFusedInferAttentionScoreV2Sink,
                            fiaTensors.query,
                            ge_tenserListKey,
                            ge_tenserListValue,
                            fiaTensors.pseShiftGe,
                            fiaTensors.attenMaskGe,
                            actualSeqInfo.actSeqArray,
                            actualSeqInfo.actSeqArrayKv,
                            fiaTensors.deqScale1,
                            fiaTensors.quantScale1,
                            fiaTensors.deqScale2,
                            fiaTensors.quantScale2,
                            fiaTensors.quantOffset2,
                            fiaTensors.antiquantScaleGe,
                            fiaTensors.antiquantOffsetGe,
                            fiaTensors.blocktableGe,
                            fiaTensors.queryPaddingGe,
                            fiaTensors.kvPaddingGe,
                            fiaTensors.keyAntiquantScaleGe,
                            fiaTensors.keyAntiquantOffsetGe,
                            fiaTensors.valueAntiquantScaleGe,
                            fiaTensors.valueAntiquantOffsetGe,
                            fiaTensors.queryRopeGe,
                            fiaTensors.keyRopeGe,
                            fiaTensors.keyRopeAntiquantScaleGe,
                            fiaTensors.dequantScaleQueryGe,
                            fiaTensors.keySinkGe,
                            fiaTensors.keyRopeSinkGe,
                            fiaTensors.valueSinkGe,
                            sclarParams.numHeads,
                            sclarParams.dScaleValue,
                            sclarParams.preTokens,
                            sclarParams.nextTokens,
                            attrPointers.layout,
                            sclarParams.kvHeadNum,
                            sclarParams.sparseMode,
                            sclarParams.innerPrecise,
                            sclarParams.blockSize,
                            sclarParams.antiquantMode,
                            sclarParams.softmaxLseFlag,
                            sclarParams.keyAntiquantMode,
                            sclarParams.valueAntiquantMode,
                            sclarParams.queryQuantMode,
                            sclarParams.sinkNumber,
                            fiaTensors.output,
                            fiaTensors.softmaxLse);

    OPS_CHECK(apiRet != GRAPH_SUCCESS,
              OPS_LOG_E(host_api_ctx->GetNodeName(), "apiRet failed:%u", apiRet),
              return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP(FusedInferAttentionScoreV2Sink)
    .OpExecuteFunc(FusedInferHostExecuteFunc)
    .HostInputs({
        ACTUAL_SEQ_Q_INDEX,
        ACTUAL_SEQ_KV_INDEX,
    });
} // namespace fallback

#ifdef __cplusplus
}
#endif
