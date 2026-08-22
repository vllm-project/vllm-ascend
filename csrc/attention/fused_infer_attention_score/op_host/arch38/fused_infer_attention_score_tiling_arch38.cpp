/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file fused_infer_attention_score_tiling_arch38.cpp
 * \brief
 */

#include "fused_infer_attention_score_tiling_arch38.h"
#include "../../../incre_flash_attention/op_host/incre_flash_attention_tiling_arch38.h"
#include "../../../prompt_flash_attention/op_host/prompt_flash_attention_tiling_arch38.h"
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "../../op_kernel/fused_infer_attention_score_tilingkey.h"

using namespace ge;
using namespace AscendC;
using namespace optiling::arch38;
namespace optiling {
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_10000000000000090, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000090, FlashAttentionScoreSimplifiedTilingData)

REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1002122000000021012, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1001121000000021012, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1002312000000021012, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1001311000000021012, FlashAttentionScoreSimplifiedTilingData)

REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1002312000040001212, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1002312000040021212, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1001311000000001212, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1001311000000021212, FlashAttentionScoreSimplifiedTilingData)

// Inputs Index
constexpr uint32_t QUERY_DIM_0 = 0;
constexpr uint32_t QUERY_DIM_1 = 1;
constexpr uint32_t QUERY_DIM_2 = 2;
constexpr uint32_t QUERY_DIM_3 = 3;
constexpr uint32_t QUERY_DIM_4 = 4;
constexpr uint32_t VALUE_DIM_2 = 2;
constexpr uint32_t VALUE_DIM_3 = 3;
constexpr uint32_t KV_DIM_0 = 0;
constexpr uint32_t KV_DIM_2 = 2;
constexpr uint32_t KV_DIM_3 = 3;
constexpr uint32_t OUT_DIM_1 = 1;
constexpr uint32_t OUT_DIM_2 = 2;
constexpr uint32_t OUT_DIM_3 = 3;
// Output Index
constexpr uint32_t DLIMIT = 512;
constexpr uint32_t BLIMIT = 65536;

constexpr uint32_t QK_D_PFA_MLA = 192;
constexpr uint32_t V_D_PFA_MLA = 128;

constexpr uint32_t SHAPE_INDEX_TWO = 2;

constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t ATTENTION_OUT_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t VALUE_INDEX = 2;
constexpr uint32_t PSE_SHIFT_INDEX = 3;
constexpr uint32_t ATTEN_MASK_INDEX = 4;
constexpr uint32_t FROM_FUSED_FLAG = 71;
constexpr uint32_t ACTUAL_SEQ_Q_INDEX = 5;
constexpr uint32_t ACTUAL_SEQ_KV_INDEX = 6;
constexpr uint32_t ANTIQUANT_SCALE_INDEX = 12;
constexpr uint32_t ANTIQUANT_OFFSET_INDEX = 13;
constexpr uint32_t QUANT_SCALE1_INDEX = 8;
constexpr uint32_t QUANT_SCALE2_INDEX = 10;
constexpr uint32_t QUANT_OFFSET2_INDEX = 11;
constexpr uint32_t ATTR_N_INDEX = 0;
constexpr uint32_t ATTR_SCALE_INDEX = 1;
constexpr uint32_t ATTR_PRE_TOKEN_INDEX = 2;
constexpr uint32_t ATTR_NEXT_TOKEN_INDEX = 3;
constexpr uint32_t ATTR_INPUT_LAYOUT_INDEX = 4;
constexpr uint32_t ATTR_NUM_KV_HEADS_INDEX = 5;

constexpr uint32_t DEQUANT_SCALE1_INDEX = 7;
constexpr uint32_t DEQUANT_SCALE2_INDEX = 9;
constexpr uint32_t BLOCK_TABLE_INDEX = 14;
constexpr uint32_t QUERY_PADDING_SIZE_INDEX = 15;
constexpr uint32_t KV_PADDING_SIZE_INDEX = 16;
constexpr uint32_t KEY_ANTIQUANT_SCALE_INDEX = 17;
constexpr uint32_t KEY_ANTIQUANT_OFFSET_INDEX = 18;
constexpr uint32_t VALUE_ANTIQUANT_SCALE_INDEX = 19;
constexpr uint32_t VALUE_ANTIQUANT_OFFSET_INDEX = 20;
constexpr uint32_t KEY_SHARED_PREFIX_INDEX = 21;
constexpr uint32_t VALUE_SHARED_PREFIX_INDEX = 22;
constexpr uint32_t ACTUAL_SHARED_PREFIX_LEN_INDEX = 23;
constexpr uint32_t QUERY_ROPE_INDEX = 24;
constexpr uint32_t KEY_ROPE_INDEX = 25;
constexpr uint32_t DEQUANT_SCALE_QUERY_INDEX = 27;
constexpr uint32_t LEARNABLE_SINK_INDEX = 28;
constexpr uint32_t Q_START_IDX_INDEX = 29;
constexpr uint32_t KV_START_IDX_INDEX = 30;

constexpr uint32_t ATTR_SPARSE_MODE_INDEX = 6;
constexpr uint32_t ATTR_INNER_PRECISE_INDEX = 7;
constexpr uint32_t ATTR_BLOCK_SIZE_INDEX = 8;
constexpr uint32_t ANTIQUANT_MODE_INDEX = 9;
constexpr uint32_t SOFTMAX_LSE_FLAG_INDEX = 10;
constexpr uint32_t KEY_ANTIQUANT_MODE_INDEX = 11;
constexpr uint32_t VALUE_ANTIQUANT_MODE_INDEX = 12;
constexpr uint32_t QUERY_QUANT_MODE_INDEX = 13;
constexpr uint32_t PSE_TYPE_INDEX = 14;

constexpr uint32_t SOFTMAX_LSE_INDEX = 1;

static bool CheckEmptyTensorList(ContextParamsForPFATiling &contextKeyParams, int64_t validBatchOfK)
{
    for (int64_t tmpIdx = 0; tmpIdx < validBatchOfK; ++tmpIdx) {
        if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetShapeSize() != 0) {
            return false;
        }
        if (contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetShapeSize() != 0) {
            return false;
        }
    }
    contextKeyParams.emptyTensor = 1;
    return true;
}

static bool GetKeyRopeSeqLenAndCheckBatch(gert::TilingContext *context,
    ContextParamsForPFATiling &contextKeyParams, int64_t validBatchOfK,
    int64_t keyRopeSDimIdx, int64_t &keyRopeS)
{
    keyRopeS = 0;
    if (contextKeyParams.keyRopeInputShape != nullptr) {
        keyRopeS = contextKeyParams.keyRopeInputShape->GetStorageShape().GetDim(keyRopeSDimIdx);
        OP_CHECK_IF(contextKeyParams.keyRopeInputShape->GetStorageShape().GetDim(0) != validBatchOfK,
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
            "Batch of Key(%ld) do NOT equal to Batch of KeyRope(%ld) under tensorlist mode!", validBatchOfK,
            contextKeyParams.keyRopeInputShape->GetStorageShape().GetDim(0)),
            return false);
    }
    return true;
}

static bool CheckNormalTensorListBSH(gert::TilingContext *context,
    ContextParamsForPFATiling &contextKeyParams, int64_t validBatchOfK)
{
    auto standardKH = contextKeyParams.kTensorList[0]->GetStorageShape().GetDim(KV_DIM_2);
    auto standardVH = contextKeyParams.vTensorList[0]->GetStorageShape().GetDim(KV_DIM_2);
    int64_t keyRopeS = 0;
    if (!GetKeyRopeSeqLenAndCheckBatch(context, contextKeyParams, validBatchOfK, 1, keyRopeS)) {
        return false;
    }

    for (int64_t tmpIdx = 0; tmpIdx < validBatchOfK; ++tmpIdx) {
        if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_2) != standardKH) {
            OP_LOGE(context->GetNodeName(),
                "H of Key(%ld) in the %ld-th batch is different from H of Key(%ld) in the first batch under "
                "tensorlist mode!",
                contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_2), tmpIdx + 1, standardKH);
            return false;
        }
        if (contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_2) != standardVH) {
            OP_LOGE(context->GetNodeName(),
                "H of Value(%ld) in the %ld-th batch is different from H of Value(%ld) in the first batch under "
                "tensorlist mode!",
                contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_2), tmpIdx + 1, standardVH);
            return false;
        }
        if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1) !=
            contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(1)) {
            OP_LOGE(context->GetNodeName(), "S for Key(%ld) and Value(%ld) does NOT equal!",
                contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1),
                contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(1));
            return false;
        }
        if (contextKeyParams.keyRopeInputShape != nullptr) {
            if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1) != keyRopeS) {
                OP_LOGE(context->GetNodeName(), "S for Key(%ld) and keyRope(%ld) does NOT equal but they should!",
                    contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1), keyRopeS);
                return false;
            }
        }
        contextKeyParams.maxKVs = std::max(contextKeyParams.maxKVs,
            uint32_t(contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1)));
    }
    return true;
}

static bool CheckNormalTensorListBNSD(gert::TilingContext *context,
    ContextParamsForPFATiling &contextKeyParams, int64_t validBatchOfK)
{
    auto standardN = contextKeyParams.kTensorList[0]->GetStorageShape().GetDim(1);
    auto standardKD = contextKeyParams.kTensorList[0]->GetStorageShape().GetDim(KV_DIM_3);
    auto standardVD = contextKeyParams.vTensorList[0]->GetStorageShape().GetDim(KV_DIM_3);
    int64_t tmpNKv = (*contextKeyParams.numKeyValueHeads != 0) ? *contextKeyParams.numKeyValueHeads :
                                                                     *contextKeyParams.headsNumber;
    int64_t keyRopeS = 0;
    if (!GetKeyRopeSeqLenAndCheckBatch(context, contextKeyParams, validBatchOfK, SHAPE_INDEX_TWO, keyRopeS)) {
        return false;
    }

    if (tmpNKv != standardN) {
        OP_LOGE(context->GetNodeName(), "N of Key(%ld) in the first batch is different from numKeyValueHeads(%ld)!",
            standardN, tmpNKv);
        return false;
    }

    for (int64_t tmpIdx = 0; tmpIdx < validBatchOfK; ++tmpIdx) {
        if ((contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1) != standardN) ||
            (contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(1) != standardN)) {
            OP_LOGE(context->GetNodeName(),
                "N of Key(%ld) and Value(%ld) in the %ld-th batch is different from numKeyValueHeads(%ld) under "
                "tensorlist mode!",
                contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1),
                contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(1), tmpIdx + 1, standardN);
            return false;
        }
        if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_3) != standardKD) {
            OP_LOGE(context->GetNodeName(),
                "D of Key(%ld) in the %ld-th batch is different from D of Key(%ld) in the first batch under "
                "tensorlist mode!",
                contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_3), tmpIdx + 1, standardKD);
            return false;
        }
        if (contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_3) != standardVD) {
            OP_LOGE(context->GetNodeName(),
                "D of Value(%ld) in the %ld-th batch is different from D of Value(%ld) in the first batch under "
                "tensorlist mode!",
                contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_3), tmpIdx + 1, standardVD);
            return false;
        }
        if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_2) !=
            contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_2)) {
            OP_LOGE(context->GetNodeName(), "S from Key(%ld) and Value(%ld) does NOT equal but they should!",
                contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_2),
                contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_2));
            return false;
        }
        if (contextKeyParams.keyRopeInputShape != nullptr) {
            if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_2) !=
                keyRopeS) {
                OP_LOGE(context->GetNodeName(), "S for Key(%ld) and keyRope(%ld) does NOT equal but they should!",
                    contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_2), keyRopeS);
                return false;
            }
        }
        contextKeyParams.maxKVs = std::max(contextKeyParams.maxKVs,
            uint32_t(contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_2)));
    }
    return true;
}

static bool CheckNormalTensorListBSND(gert::TilingContext *context,
    ContextParamsForPFATiling &contextKeyParams, int64_t validBatchOfK)
{
    auto standardN = contextKeyParams.kTensorList[0]->GetStorageShape().GetDim(KV_DIM_2);
    auto standardKD = contextKeyParams.kTensorList[0]->GetStorageShape().GetDim(KV_DIM_3);
    auto standardVD = contextKeyParams.vTensorList[0]->GetStorageShape().GetDim(KV_DIM_3);
    int64_t tmpNKv = (*contextKeyParams.numKeyValueHeads != 0) ? *contextKeyParams.numKeyValueHeads :
                                                                     *contextKeyParams.headsNumber;
    int64_t keyRopeS = 0;
    if (!GetKeyRopeSeqLenAndCheckBatch(context, contextKeyParams, validBatchOfK, 1, keyRopeS)) {
        return false;
    }

    if (tmpNKv != standardN) {
        OP_LOGE(context->GetNodeName(), "N of Key(%ld) in the first batch is different from numKeyValueHeads(%ld)!",
            standardN, tmpNKv);
        return false;
    }

    for (int64_t tmpIdx = 0; tmpIdx < validBatchOfK; ++tmpIdx) {
        if ((contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_2) !=
            standardN) ||
            (contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_2) !=
            standardN)) {
            OP_LOGE(context->GetNodeName(),
                "N of Key(%ld) and Value(%ld) in the %ld-th batch is different from numKeyValueHeads(%ld) under "
                "tensorlist mode!",
                contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_2),
                contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_2), tmpIdx + 1, standardN);
            return false;
        }
        if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_3) != standardKD) {
            OP_LOGE(context->GetNodeName(),
                "D of Key(%ld) in the %ld-th batch is different from D of Key(%ld) in the first batch under "
                "tensorlist mode!",
                contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_3), tmpIdx + 1, standardKD);
            return false;
        }
        if (contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_3) != standardVD) {
            OP_LOGE(context->GetNodeName(),
                "D of Value(%ld) in the %ld-th batch is different from D of Value(%ld) in the first batch under "
                "tensorlist mode!",
                contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_3), tmpIdx + 1, standardVD);
            return false;
        }
        if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1) !=
            contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(1)) {
            OP_LOGE(context->GetNodeName(), "S from Key(%ld) and Value(%ld) does NOT equal but they should!",
                contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1),
                contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(1));
            return false;
        }
        if (contextKeyParams.keyRopeInputShape != nullptr) {
            if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1) != keyRopeS) {
                OP_LOGE(context->GetNodeName(), "S for Key(%ld) and keyRope(%ld) does NOT equal but they should!",
                    contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1), keyRopeS);
                return false;
            }
        }
        contextKeyParams.maxKVs = std::max(contextKeyParams.maxKVs,
            uint32_t(contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1)));
    }
    return true;
}

static bool CheckNormalTensorList(gert::TilingContext *context, ContextParamsForPFATiling &contextKeyParams,
    const string layoutStr, int64_t validBatchOfK)
{
    bool ret = false;
    if (layoutStr == "BSH") {
        ret = CheckNormalTensorListBSH(context, contextKeyParams, validBatchOfK);
    } else if (layoutStr == "BNSD" || layoutStr == "BNSD_BSND") {
        ret = CheckNormalTensorListBNSD(context, contextKeyParams, validBatchOfK);
    } else {
        ret = CheckNormalTensorListBSND(context, contextKeyParams, validBatchOfK);
    }
    if (ret) {
        contextKeyParams.isKvContinuous = 0;
    }
    return ret;
}

static bool CheckTensorList(gert::TilingContext *context, ContextParamsForPFATiling &contextKeyParams,
    const string layoutStr, int64_t batchOfQ)
{
    int64_t validBatchOfK = 0;
    int64_t validBatchOfV = 0;
    contextKeyParams.kTensorList.resize(batchOfQ);
    contextKeyParams.vTensorList.resize(batchOfQ);
    while (context->GetDynamicInputShape(KEY_INDEX, validBatchOfK) != nullptr) {
        contextKeyParams.kTensorList[validBatchOfK] = context->GetDynamicInputShape(KEY_INDEX, validBatchOfK);
        validBatchOfK++;
    }

    while (context->GetDynamicInputShape(VALUE_INDEX, validBatchOfV) != nullptr) {
        contextKeyParams.vTensorList[validBatchOfV] = context->GetDynamicInputShape(VALUE_INDEX, validBatchOfV);
        validBatchOfV++;
    }

    OP_CHECK_IF((batchOfQ != validBatchOfK) || (validBatchOfK != validBatchOfV),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "Batch of Query(%ld) do NOT equal to Batch of Key(%ld) and Value(%ld) under tensorlist mode!", batchOfQ,
        validBatchOfK, validBatchOfV),
        return false);

    OP_CHECK_IF((batchOfQ > BLIMIT),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "Batch of Query(%ld) do NOT larger than 65535 under tensorlist mode!", batchOfQ),
        return false);

    if (CheckEmptyTensorList(contextKeyParams, validBatchOfK)) {
        return true;
    }

    for (int i = 0; i < validBatchOfK; i++) {
        OP_CHECK_IF(contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(0) != 1,
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
            "Batch value of Key(%ld) is NOT 1 but should be 1 under tensorlist mode!",
            contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(0)),
            return false);
    }

    for (int i = 0; i < validBatchOfV; i++) {
        OP_CHECK_IF(contextKeyParams.vTensorList[i]->GetStorageShape().GetDim(0) != 1,
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
            "Batch value of Value(%ld) is NOT 1 but should be 1 under tensorlist mode!",
            contextKeyParams.vTensorList[i]->GetStorageShape().GetDim(0)),
            return false);
    }

    if (!CheckNormalTensorList(context, contextKeyParams, layoutStr, validBatchOfK)) {
        return false;
    }
    return true;
}

static bool CheckKVPaddingCrossover(gert::TilingContext *context, ContextParamsForPFATiling &contextKeyParams)
{
    OP_CHECK_IF(((contextKeyParams.queryPaddingSize != nullptr) &&
        (contextKeyParams.queryPaddingSize->GetStorageShape().GetShapeSize() != 1 ||
        contextKeyParams.queryPaddingSize->GetStorageShape().GetDimNum() != 1)),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "Query PaddingSize(%zu) input is invalid, the shape size of query paddingsize is not 1 or dim number of query "
        "padding size is not 1!",
        contextKeyParams.queryPaddingSize->GetStorageShape().GetDimNum()),
        return false);
    OP_CHECK_IF(((contextKeyParams.kvPaddingSize != nullptr) &&
        (contextKeyParams.kvPaddingSize->GetStorageShape().GetShapeSize() != 1 ||
        contextKeyParams.kvPaddingSize->GetStorageShape().GetDimNum() != 1)),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "KV PaddingSize(%zu) input is invalid, the shape size of kv padding size is not 1 or dim number of kv padding "
        "size is not 1!",
        contextKeyParams.kvPaddingSize->GetStorageShape().GetDimNum()),
        return false);
    OP_CHECK_IF(((contextKeyParams.blockTable != nullptr) &&
        ((contextKeyParams.queryPaddingSize != nullptr) || (contextKeyParams.kvPaddingSize != nullptr))),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "When page attention is used, left padding is not supported!"),
        return false);
    OP_CHECK_IF(((contextKeyParams.queryPaddingSize != nullptr) && (contextKeyParams.actualSequenceLengthQ == nullptr)),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "if Query has leftpadding, actual_seq_lengths are required, but actual_seq_lengths is null!"),
        return false);
    OP_CHECK_IF(((contextKeyParams.kvPaddingSize != nullptr) && (contextKeyParams.actualSequenceLengthKV == nullptr)),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "if KV has leftpadding, actual_seq_lengths_kv are required, but actual_seq_lengths_kv is null!"),
        return false);
    return true;
}

// 0 不转置; 1 BNSD_BSND; 2 BSND_BNSD; 3 BSH_BNSD; 4 BNSD_NBSD; 5 BSND_NBSD; 6 BSH_NBSD; 7 NTD_TND; 8 TND_NTD
static uint32_t GetTransposeLayout(const std::string &layout)
{
    const std::map<std::string, uint32_t> transposeLayoutMp = { { "BNSD_BSND", 1 }, { "BSND_BNSD", 2 },
        { "BSH_BNSD", 3 },  { "BNSD_NBSD", 4 },
        { "BSND_NBSD", 5 }, { "BSH_NBSD", 6 },
        { "NTD_TND", 7 },   { "TND_NTD", 8 } };
    if (transposeLayoutMp.find(layout) != transposeLayoutMp.end()) {
        return transposeLayoutMp.at(layout);
    }
    return 0;
}

static ge::graphStatus ConvertQuantOptionalInputs(const gert::TilingContext *context,
    ContextParamsForPFATiling &contextKeyParams)
{
    contextKeyParams.deqScale1Shape = context->GetOptionalInputShape(DEQUANT_SCALE1_INDEX);
    contextKeyParams.scale1Shape = context->GetOptionalInputShape(QUANT_SCALE1_INDEX);
    contextKeyParams.deqScale2Shape = context->GetOptionalInputShape(DEQUANT_SCALE2_INDEX);
    contextKeyParams.scale2Shape = context->GetOptionalInputShape(QUANT_SCALE2_INDEX);
    contextKeyParams.offset2Shape = context->GetOptionalInputShape(QUANT_OFFSET2_INDEX);
    contextKeyParams.antiquantScaleShape = context->GetOptionalInputShape(ANTIQUANT_SCALE_INDEX);
    contextKeyParams.antiquantOffsetShape = context->GetOptionalInputShape(ANTIQUANT_OFFSET_INDEX);

    contextKeyParams.quantScale2Type = (context->GetOptionalInputDesc(QUANT_SCALE2_INDEX) != nullptr) ?
        context->GetOptionalInputDesc(QUANT_SCALE2_INDEX)->GetDataType() :
        ge::DT_FLOAT;
    contextKeyParams.quantOffset2Type = (context->GetOptionalInputDesc(QUANT_OFFSET2_INDEX) != nullptr) ?
        context->GetOptionalInputDesc(QUANT_OFFSET2_INDEX)->GetDataType() :
        ge::DT_FLOAT;
    contextKeyParams.dequantScaleQueryShape = context->GetOptionalInputShape(DEQUANT_SCALE_QUERY_INDEX);
    contextKeyParams.KeyAntiquantScaleShape = context->GetOptionalInputShape(KEY_ANTIQUANT_SCALE_INDEX);
    contextKeyParams.KeyAntiquantOffsetShape = context->GetOptionalInputShape(KEY_ANTIQUANT_OFFSET_INDEX);
    contextKeyParams.valueAntiquantScaleShape = context->GetOptionalInputShape(VALUE_ANTIQUANT_SCALE_INDEX);
    contextKeyParams.valueAntiquantOffsetShape = context->GetOptionalInputShape(VALUE_ANTIQUANT_OFFSET_INDEX);

    contextKeyParams.dequantScaleQueryType = (context->GetOptionalInputDesc(DEQUANT_SCALE_QUERY_INDEX) != nullptr) ?
        context->GetOptionalInputDesc(DEQUANT_SCALE_QUERY_INDEX)->GetDataType() :
        ge::DT_FLOAT;
    contextKeyParams.KeyAntiquantScaleType = (context->GetOptionalInputDesc(KEY_ANTIQUANT_SCALE_INDEX) != nullptr) ?
        context->GetOptionalInputDesc(KEY_ANTIQUANT_SCALE_INDEX)->GetDataType() :
        ge::DT_FLOAT16;
    contextKeyParams.valueAntiquantScaleType = (context->GetOptionalInputDesc(VALUE_ANTIQUANT_SCALE_INDEX) != nullptr) ?
        context->GetOptionalInputDesc(VALUE_ANTIQUANT_SCALE_INDEX)->GetDataType() :
        ge::DT_FLOAT16;

    contextKeyParams.dequantScaleQuery = context->GetOptionalInputTensor(DEQUANT_SCALE_QUERY_INDEX);
    contextKeyParams.keyAntiquantScale = context->GetOptionalInputTensor(KEY_ANTIQUANT_SCALE_INDEX);
    contextKeyParams.valueAntiquantScale = context->GetOptionalInputTensor(VALUE_ANTIQUANT_SCALE_INDEX);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ConvertContextToParamsPFA(gert::TilingContext *context,
    ContextParamsForPFATiling &contextKeyParams, bool isMaxWorkspace)
{
    contextKeyParams.opName = context->GetNodeName();

    contextKeyParams.isKvContinuous = 1;
    contextKeyParams.emptyTensor = 0;
    contextKeyParams.fromFused = FROM_FUSED_FLAG;
    contextKeyParams.maxKVs = 0;
    contextKeyParams.pseShift = context->GetOptionalInputTensor(PSE_SHIFT_INDEX);
    contextKeyParams.attentionMask = context->GetOptionalInputTensor(ATTEN_MASK_INDEX);
    OP_CHECK_IF((contextKeyParams.attentionMask != nullptr) &&
        (context->GetOptionalInputDesc(ATTEN_MASK_INDEX)->GetDataType() != ge::DT_BOOL) &&
        (context->GetOptionalInputDesc(ATTEN_MASK_INDEX)->GetDataType() != ge::DT_INT8) &&
        (context->GetOptionalInputDesc(ATTEN_MASK_INDEX)->GetDataType() != ge::DT_UINT8),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "Invalid attention mask datatype(%s)! Only support BOOL, INT8 and UINT8",
        arch38::GetPfaDataTypeStr(context->GetOptionalInputDesc(ATTEN_MASK_INDEX)->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);
    contextKeyParams.actualSequenceLengthQ = context->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    contextKeyParams.actualSequenceLengthKV = context->GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    contextKeyParams.antiquantScale = context->GetOptionalInputTensor(ANTIQUANT_SCALE_INDEX);
    contextKeyParams.antiquantOffset = context->GetOptionalInputTensor(ANTIQUANT_OFFSET_INDEX);
    contextKeyParams.queryPaddingSize = context->GetOptionalInputTensor(QUERY_PADDING_SIZE_INDEX);
    contextKeyParams.kvPaddingSize = context->GetOptionalInputTensor(KV_PADDING_SIZE_INDEX);
    contextKeyParams.blockTable = context->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    contextKeyParams.keySharedPrefix = context->GetOptionalInputTensor(KEY_SHARED_PREFIX_INDEX);
    contextKeyParams.valueSharedPrefix = context->GetOptionalInputTensor(VALUE_SHARED_PREFIX_INDEX);
    contextKeyParams.actualSharedPrefixLen = context->GetOptionalInputTensor(ACTUAL_SHARED_PREFIX_LEN_INDEX);
    contextKeyParams.learnableSink = context->GetOptionalInputTensor(LEARNABLE_SINK_INDEX);
    contextKeyParams.learnableSinkShape = context->GetOptionalInputShape(LEARNABLE_SINK_INDEX);
    contextKeyParams.hasLearnableSink =
        ((contextKeyParams.learnableSink != nullptr) && (contextKeyParams.learnableSinkShape != nullptr) &&
        (contextKeyParams.learnableSinkShape->GetStorageShape().GetShapeSize() != 0)) ?
        true :
        false;
    contextKeyParams.inputDataType = context->GetInputDesc(QUERY_INDEX)->GetDataType();
    contextKeyParams.kDataType = context->GetInputDesc(KEY_INDEX)->GetDataType();
    contextKeyParams.vDataType = context->GetInputDesc(VALUE_INDEX)->GetDataType();
    contextKeyParams.pseShiftDataType = (contextKeyParams.pseShift != nullptr) ?
        context->GetOptionalInputDesc(PSE_SHIFT_INDEX)->GetDataType() :
        contextKeyParams.inputDataType;
    contextKeyParams.maskDataType = (contextKeyParams.attentionMask != nullptr) ?
        context->GetOptionalInputDesc(ATTEN_MASK_INDEX)->GetDataType() :
        contextKeyParams.inputDataType;
    contextKeyParams.blockTableType = (context->GetOptionalInputDesc(BLOCK_TABLE_INDEX) != nullptr) ?
        context->GetOptionalInputDesc(BLOCK_TABLE_INDEX)->GetDataType() :
        ge::DT_INT32;
    contextKeyParams.outputDataType = context->GetOutputDesc(ATTENTION_OUT_INDEX)->GetDataType();
    contextKeyParams.queryInputShape = context->GetInputShape(QUERY_INDEX);
    contextKeyParams.keyInputShape = context->GetDynamicInputShape(KEY_INDEX, 0);
    contextKeyParams.valueInputShape = context->GetDynamicInputShape(VALUE_INDEX, 0);
    contextKeyParams.pseShiftShape = context->GetOptionalInputShape(PSE_SHIFT_INDEX);
    contextKeyParams.attentionMaskShape = context->GetOptionalInputShape(ATTEN_MASK_INDEX);
    contextKeyParams.blockTableShape = context->GetOptionalInputShape(BLOCK_TABLE_INDEX);
    contextKeyParams.outputShape = context->GetOutputShape(ATTENTION_OUT_INDEX);
    contextKeyParams.lseoutputShape = context->GetOutputShape(SOFTMAX_LSE_INDEX);
    contextKeyParams.keySharedPrefixDataType = (contextKeyParams.keySharedPrefix != nullptr) ?
        context->GetOptionalInputDesc(KEY_SHARED_PREFIX_INDEX)->GetDataType() :
        contextKeyParams.inputDataType;
    contextKeyParams.valueSharedPrefixDataType = (contextKeyParams.valueSharedPrefix != nullptr) ?
        context->GetOptionalInputDesc(VALUE_SHARED_PREFIX_INDEX)->GetDataType() :
        contextKeyParams.inputDataType;
    contextKeyParams.learnableSinkDataType = (contextKeyParams.learnableSink != nullptr) ?
        context->GetOptionalInputDesc(LEARNABLE_SINK_INDEX)->GetDataType() :
        contextKeyParams.inputDataType;

    auto convertQuantRet = ConvertQuantOptionalInputs(context, contextKeyParams);
    if (convertQuantRet != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Error occured while convert quant related tilingContext to PFA context!");
        return convertQuantRet;
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Attributes returned from context is a nullptr"),
        return ge::GRAPH_FAILED);
    contextKeyParams.queryQuantMode = attrs->GetAttrPointer<int64_t>(QUERY_QUANT_MODE_INDEX);
    contextKeyParams.keyAntiquantMode = attrs->GetAttrPointer<int64_t>(KEY_ANTIQUANT_MODE_INDEX);
    contextKeyParams.valueAntiquantMode = attrs->GetAttrPointer<int64_t>(VALUE_ANTIQUANT_MODE_INDEX);
    contextKeyParams.innerPrecisePtr = attrs->GetAttrPointer<int64_t>(ATTR_INNER_PRECISE_INDEX);
    contextKeyParams.headsNumber = attrs->GetAttrPointer<int64_t>(ATTR_N_INDEX);
    contextKeyParams.sparseMode = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_MODE_INDEX);
    contextKeyParams.preToken = attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKEN_INDEX);
    contextKeyParams.nextToken = attrs->GetAttrPointer<int64_t>(ATTR_NEXT_TOKEN_INDEX);
    contextKeyParams.scaleValue = attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX);
    contextKeyParams.layout = attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX);
    contextKeyParams.numKeyValueHeads = attrs->GetAttrPointer<int64_t>(ATTR_NUM_KV_HEADS_INDEX);
    contextKeyParams.blockSize = attrs->GetAttrPointer<int32_t>(ATTR_BLOCK_SIZE_INDEX);
    contextKeyParams.workspaceSize = context->GetWorkspaceSizes(1);
    contextKeyParams.isBSNDOut = (string(contextKeyParams.layout) == "BNSD_BSND") ? 1 : 0;
    contextKeyParams.transposeLayout = GetTransposeLayout(string(contextKeyParams.layout));
    contextKeyParams.softmaxLseFlag = attrs->GetAttrPointer<bool>(SOFTMAX_LSE_FLAG_INDEX);
    contextKeyParams.isSoftMaxLseEnable =
        (contextKeyParams.softmaxLseFlag == nullptr) ? false : *contextKeyParams.softmaxLseFlag;
    contextKeyParams.queryRopeInputShape = context->GetOptionalInputShape(QUERY_ROPE_INDEX);
    contextKeyParams.qRopeDataType = (contextKeyParams.queryRopeInputShape != nullptr) ?
        context->GetOptionalInputDesc(QUERY_ROPE_INDEX)->GetDataType() :
        contextKeyParams.inputDataType;
    contextKeyParams.keyRopeInputShape = context->GetOptionalInputShape(KEY_ROPE_INDEX);
    contextKeyParams.kRopeDataType = (contextKeyParams.keyRopeInputShape != nullptr) ?
        context->GetOptionalInputDesc(KEY_ROPE_INDEX)->GetDataType() :
        contextKeyParams.kDataType;
    contextKeyParams.qStartIdx = context->GetOptionalInputTensor(Q_START_IDX_INDEX);
    contextKeyParams.kvStartIdx = context->GetOptionalInputTensor(KV_START_IDX_INDEX);
    contextKeyParams.pseType = attrs->GetAttrPointer<int64_t>(PSE_TYPE_INDEX);

    const string layoutStr = string(contextKeyParams.layout);
    int64_t batchOfQ = 1;
    if (layoutStr != "NSD") {
        if (layoutStr != "TND" && layoutStr != "NTD" && layoutStr != "NTD_NTD" && layoutStr != "TND_NTD") {
            batchOfQ = contextKeyParams.queryInputShape->GetStorageShape().GetDim(QUERY_DIM_0);
        } else {
            if (!isMaxWorkspace) {
                const gert::Tensor *actSeqLenData = contextKeyParams.actualSequenceLengthQ;
                int64_t actSeqLenDims = (actSeqLenData != nullptr) ? actSeqLenData->GetShapeSize() : 0;
                OP_CHECK_IF(((actSeqLenData == nullptr) || (actSeqLenDims == 0) ||
                    (actSeqLenData->GetData<int64_t>() == nullptr)),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "When layout is TND/NTD/TND_NTD/NTD_TND, actualSequenceLengthQ is required"),
                    return ge::GRAPH_FAILED);
                const gert::Tensor *actSeqLenDataKV = contextKeyParams.actualSequenceLengthKV;
                int64_t actSeqLenKVDims = (actSeqLenDataKV != nullptr) ? actSeqLenDataKV->GetShapeSize() : 0;
                OP_CHECK_IF(((actSeqLenDataKV == nullptr) || (actSeqLenKVDims == 0) ||
                    (actSeqLenDataKV->GetData<int64_t>() == nullptr)),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "When layout is TND/NTD/TND_NTD/NTD_TND, actualSequenceLengthKV is required"),
                    return ge::GRAPH_FAILED);
                batchOfQ = actSeqLenDims;
            }
        }
    }
    // Obtain the actual number of K input elements and determine whether they belong to the tensorlist scene
    int64_t validBatchOfK = 0;
    while (context->GetDynamicInputShape(KEY_INDEX, validBatchOfK) != nullptr) {
        validBatchOfK++;
        // If there are more than 1, break. When the input is large, it saves time.
        // The tensorlist scene also needs to verify separately whether it is 1
        if (validBatchOfK > 1) {
            break;
        }
    }
    if (validBatchOfK > 1) { // k tensor 超过一个, 认为是tensorlist场景
        // PA和TND格式, 与tensorlist本身的定义有冲突, 在判定tensorlist输入后先做校验, 之后再校验tensorlist信息正确性
        OP_CHECK_IF((contextKeyParams.blockTable != nullptr),
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
            "When tensorlist is used, page attention is not supported!"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((layoutStr == "TND" || layoutStr == "NTD" || layoutStr == "NTD_TND" || layoutStr == "TND_NTD"),
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
            "When tensorlist is used, layout TND/NTD/TND_NTD/NTD_TND is not supported!"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((!CheckTensorList(context, contextKeyParams, layoutStr, batchOfQ)),
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Check Tensorlist failed!"), return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(((contextKeyParams.isKvContinuous == 0) &&
        ((context->GetOptionalInputDesc(QUERY_PADDING_SIZE_INDEX) != nullptr) ||
        (context->GetOptionalInputDesc(KV_PADDING_SIZE_INDEX) != nullptr))),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "When tensorlist is used, left padding is not supported!"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(!CheckKVPaddingCrossover(context, contextKeyParams),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Check KV PaddingSize failed!"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(contextKeyParams.workspaceSize == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "WorkSpaceSize is nullptr"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ConvertContextToParamsIFA(gert::TilingContext &context, IncreFlashAttentionContext &ifaContext,
    bool isMaxWorkspace)
{
    if (context.GetNodeName() == nullptr) {
        OP_LOGE("FusedInferAttentionScore", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    ifaContext.opName = context.GetNodeName();
    ifaContext.platformInfo = context.GetPlatformInfo();
    ifaContext.query.desc = context.GetInputDesc(QUERY_INDEX);
    ifaContext.query.shape = context.GetInputShape(QUERY_INDEX);
    ifaContext.key.desc = context.GetInputDesc(KEY_INDEX);
    ifaContext.key.shape = context.GetInputShape(KEY_INDEX);
    OP_CHECK_IF((ifaContext.query.shape == nullptr) || (ifaContext.key.shape == nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "shape of query of shape of key is null."),
        return ge::GRAPH_FAILED);

    ifaContext.value.desc = context.GetInputDesc(VALUE_INDEX);
    ifaContext.value.shape = context.GetInputShape(VALUE_INDEX);
    ifaContext.pseShift.desc = context.GetOptionalInputDesc(PSE_SHIFT_INDEX);
    ifaContext.pseShift.tensor = context.GetOptionalInputTensor(PSE_SHIFT_INDEX);

    ifaContext.attenMask.desc = context.GetOptionalInputDesc(ATTEN_MASK_INDEX);
    ifaContext.attenMask.tensor = context.GetOptionalInputTensor(ATTEN_MASK_INDEX);
    ifaContext.attenOut.desc = context.GetOutputDesc(ATTENTION_OUT_INDEX);
    ifaContext.attenOut.shape = context.GetOutputShape(ATTENTION_OUT_INDEX);
    ifaContext.lseOut.desc = context.GetOutputDesc(SOFTMAX_LSE_INDEX);
    ifaContext.lseOut.shape = context.GetOutputShape(SOFTMAX_LSE_INDEX);

    ifaContext.actualSeqLengthsQ.tensor = context.GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    ifaContext.actualSeqLengths.tensor = context.GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    ifaContext.deqScale1.tensor = context.GetOptionalInputTensor(DEQUANT_SCALE1_INDEX);
    ifaContext.quantScale1.tensor = context.GetOptionalInputTensor(QUANT_SCALE1_INDEX);
    ifaContext.deqScale2.tensor = context.GetOptionalInputTensor(DEQUANT_SCALE2_INDEX);
    ifaContext.quantScale2.tensor = context.GetOptionalInputTensor(QUANT_SCALE2_INDEX);
    ifaContext.quantOffset2.tensor = context.GetOptionalInputTensor(QUANT_OFFSET2_INDEX);
    ifaContext.quantScale2.desc = context.GetOptionalInputDesc(QUANT_SCALE2_INDEX);
    ifaContext.quantOffset2.desc = context.GetOptionalInputDesc(QUANT_OFFSET2_INDEX);
    ifaContext.antiquantScale.tensor = context.GetOptionalInputTensor(ANTIQUANT_SCALE_INDEX);
    ifaContext.antiquantScale.desc = context.GetOptionalInputDesc(ANTIQUANT_SCALE_INDEX);
    ifaContext.antiquantOffset.tensor = context.GetOptionalInputTensor(ANTIQUANT_OFFSET_INDEX);
    ifaContext.antiquantOffset.desc = context.GetOptionalInputDesc(ANTIQUANT_OFFSET_INDEX);
    ifaContext.blockTable.tensor = context.GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    ifaContext.queryPaddingSize.tensor = context.GetOptionalInputTensor(QUERY_PADDING_SIZE_INDEX);
    ifaContext.kvPaddingSize.tensor = context.GetOptionalInputTensor(KV_PADDING_SIZE_INDEX);
    ifaContext.keyAntiquantScale.tensor = context.GetOptionalInputTensor(KEY_ANTIQUANT_SCALE_INDEX);
    ifaContext.keyAntiquantScale.desc = context.GetOptionalInputDesc(KEY_ANTIQUANT_SCALE_INDEX);
    ifaContext.keyAntiquantOffset.tensor = context.GetOptionalInputTensor(KEY_ANTIQUANT_OFFSET_INDEX);
    ifaContext.keyAntiquantOffset.desc = context.GetOptionalInputDesc(KEY_ANTIQUANT_OFFSET_INDEX);
    ifaContext.valueAntiquantScale.tensor = context.GetOptionalInputTensor(VALUE_ANTIQUANT_SCALE_INDEX);
    ifaContext.valueAntiquantScale.desc = context.GetOptionalInputDesc(VALUE_ANTIQUANT_SCALE_INDEX);
    ifaContext.valueAntiquantOffset.tensor = context.GetOptionalInputTensor(VALUE_ANTIQUANT_OFFSET_INDEX);
    ifaContext.valueAntiquantOffset.desc = context.GetOptionalInputDesc(VALUE_ANTIQUANT_OFFSET_INDEX);
    ifaContext.qStartIdx.tensor = context.GetOptionalInputTensor(Q_START_IDX_INDEX);
    ifaContext.qStartIdx.desc = context.GetOptionalInputDesc(Q_START_IDX_INDEX);
    ifaContext.kvStartIdx.tensor = context.GetOptionalInputTensor(KV_START_IDX_INDEX);
    ifaContext.kvStartIdx.desc = context.GetOptionalInputDesc(KV_START_IDX_INDEX);
    ifaContext.keySharedPrefix.tensor = context.GetOptionalInputTensor(KEY_SHARED_PREFIX_INDEX);
    ifaContext.keySharedPrefix.desc = context.GetOptionalInputDesc(KEY_SHARED_PREFIX_INDEX);
    ifaContext.valueSharedPrefix.tensor = context.GetOptionalInputTensor(VALUE_SHARED_PREFIX_INDEX);
    ifaContext.valueSharedPrefix.desc = context.GetOptionalInputDesc(VALUE_SHARED_PREFIX_INDEX);
    ifaContext.actualSharedPrefixLen.tensor = context.GetOptionalInputTensor(ACTUAL_SHARED_PREFIX_LEN_INDEX);
    ifaContext.actualSharedPrefixLen.desc = context.GetOptionalInputDesc(ACTUAL_SHARED_PREFIX_LEN_INDEX);

    auto attrs = context.GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "attrs got from ge is nullptr"),
        return ge::GRAPH_FAILED);

    ifaContext.numHeads = attrs->GetAttrPointer<uint32_t>(ATTR_N_INDEX);
    ifaContext.sparseMode = attrs->GetAttrPointer<uint32_t>(ATTR_SPARSE_MODE_INDEX);
    ifaContext.preToken = attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKEN_INDEX);
    ifaContext.nextToken = attrs->GetAttrPointer<int64_t>(ATTR_NEXT_TOKEN_INDEX);
    ifaContext.innerPrecise = attrs->GetAttrPointer<uint32_t>(ATTR_INNER_PRECISE_INDEX);
    ifaContext.scaleValue = attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX);
    ifaContext.layOut = attrs->GetStr(ATTR_INPUT_LAYOUT_INDEX);
    ifaContext.kvHeadNums = attrs->GetAttrPointer<uint32_t>(ATTR_NUM_KV_HEADS_INDEX);
    ifaContext.blockSize = attrs->GetAttrPointer<uint32_t>(ATTR_BLOCK_SIZE_INDEX);
    ifaContext.antiquantMode = attrs->GetAttrPointer<int64_t>(ANTIQUANT_MODE_INDEX);
    ifaContext.softmaxLseFlag = attrs->GetAttrPointer<bool>(SOFTMAX_LSE_FLAG_INDEX);
    ifaContext.keyAntiquantMode = attrs->GetAttrPointer<int64_t>(KEY_ANTIQUANT_MODE_INDEX);
    ifaContext.valueAntiquantMode = attrs->GetAttrPointer<int64_t>(VALUE_ANTIQUANT_MODE_INDEX);
    ifaContext.pseType = attrs->GetAttrPointer<int64_t>(PSE_TYPE_INDEX);

    // Tiling下沉场景拿不到actualSeqLengths的个数，所以qk的batch数默认设置为1来规避后续校验
    auto batchOfQuery = 1;
    auto batchOfKey = 1;
    std::string layoutStr(ifaContext.layOut);
    if (layoutStr != "TND" && layoutStr != "NTD") {
        batchOfQuery = ifaContext.query.shape->GetStorageShape().GetDim(QUERY_DIM_0);
        batchOfKey = ifaContext.key.shape->GetStorageShape().GetDim(KV_DIM_0);
    } else {
        if (isMaxWorkspace) {
            if (ifaContext.blockTable.tensor != nullptr) {
                batchOfQuery = ifaContext.blockTable.tensor->GetStorageShape().GetDim(QUERY_DIM_0);
                batchOfKey = ifaContext.blockTable.tensor->GetStorageShape().GetDim(KV_DIM_0);
            }
        } else {
            OP_CHECK_IF(
                (ifaContext.actualSeqLengthsQ.tensor == nullptr || ifaContext.actualSeqLengths.tensor == nullptr),
                OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
                "TND/NTD actualSeqLengths or actualSeqLengthsKv is null."),
                return ge::GRAPH_FAILED);
            batchOfQuery = ifaContext.actualSeqLengthsQ.tensor->GetSize();
            batchOfKey = ifaContext.actualSeqLengths.tensor->GetSize();
        }
    }

    if (batchOfQuery != batchOfKey) {
        ifaContext.kCache.resize(batchOfQuery);
        ifaContext.vCache.resize(batchOfQuery);
        for (int64_t size = 0; size < batchOfQuery; ++size) {
            ifaContext.kCache[size] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(KEY_INDEX, size));
            ifaContext.vCache[size] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(VALUE_INDEX, size));
        }
    } else {
        ifaContext.kCache.resize(1);
        ifaContext.vCache.resize(1);
        ifaContext.kCache[0] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(KEY_INDEX, 0));
        ifaContext.vCache[0] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(VALUE_INDEX, 0));
    }

    OP_CHECK_IF(context.GetWorkspaceSizes(1) == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "workSpaceSize got from ge is nullptr"),
        return ge::GRAPH_FAILED);
    ifaContext.workSpaces = context.GetWorkspaceSizes(1);

    // IFA（伪量化）模板当前不支持 learnable sink 输入特性
    OP_CHECK_IF(context.GetOptionalInputTensor(LEARNABLE_SINK_INDEX) != nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Learnable sink only supports no-quantized GQA mode."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static bool GetMaxWorkspaceFlag(gert::TilingContext &context)
{
    bool res = (context.GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX) != nullptr &&
        context.GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX)->GetData<int64_t>() == nullptr) ||
        (context.GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX) != nullptr &&
        context.GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX)->GetData<int64_t>() == nullptr);
    return res;
}

ge::graphStatus TilingFusedInferAttentionScoreArch38(gert::TilingContext *context)
{
    FusedInferAttentionScoreTilingArch38 FIATilingArch38(context);
    auto ret = FIATilingArch38.DoTiling(nullptr);
    return ret;
}

ge::graphStatus FusedInferAttentionScoreTilingArch38::DoOpTiling()
{
    if (context_ == nullptr) {
        OP_LOGE("FusedInferAttentionScore", "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    bool isMaxWorkspace = GetMaxWorkspaceFlag(*context_);

    auto tempQ = context_->GetInputShape(QUERY_INDEX);
    auto tempV = context_->GetDynamicInputShape(VALUE_INDEX, 0);
    auto tempOut = context_->GetOutputShape(ATTENTION_OUT_INDEX);
    auto tempLse = context_->GetOutputShape(SOFTMAX_LSE_INDEX);
    bool qOutEmptyTensor = false;
    uint32_t queryD = 1U;
    uint32_t valueD = 1U;
    OP_CHECK_IF((tempQ == nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "Query input is null pointer!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF((tempV == nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "Value input is null pointer!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF((tempOut == nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "AttentionOut is null pointer!"), return ge::GRAPH_FAILED);
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF((tempQ->GetStorageShape().GetShapeSize() == 0 && tempOut->GetStorageShape().GetShapeSize() != 0) ||
        (tempQ->GetStorageShape().GetShapeSize() != 0 && tempOut->GetStorageShape().GetShapeSize() == 0),
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
        "query shape size is %llu byte, but attention Out shape size is %llu byte, they cannot be empty while the "
        "other is not",
        tempQ->GetStorageShape().GetShapeSize(), tempOut->GetStorageShape().GetShapeSize()),
        return ge::GRAPH_FAILED);
    if (tempQ->GetStorageShape().GetShapeSize() == 0 && tempOut->GetStorageShape().GetShapeSize() == 0) {
        qOutEmptyTensor = true;
    }

    OP_CHECK_IF((tempQ->GetStorageShape().GetShapeSize() == gert::Shape::kInvalidDimValue),
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "Get the shape size of Query failed!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(attrs == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "Attributes returned from GetAttrs() is a nullptr!"),
        return ge::GRAPH_FAILED);

    uint32_t tempN = *attrs->GetAttrPointer<uint32_t>(ATTR_N_INDEX);
    uint32_t tempKVN = *attrs->GetAttrPointer<uint32_t>(ATTR_NUM_KV_HEADS_INDEX);
    OP_CHECK_IF(tempN == 0, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "Q numhead is 0!"),
        return ge::GRAPH_FAILED);
    if (tempKVN == 0U) {
        tempKVN = tempN;
    }
    const string inputLayoutStr = string(attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
    int64_t s = 0;
    int64_t b = tempQ->GetStorageShape().GetDim(QUERY_DIM_0);
    int64_t t = 0;
    bool lseFlag = *attrs->GetAttrPointer<bool>(SOFTMAX_LSE_FLAG_INDEX);
    bool usingIFA = false;
    if (inputLayoutStr == "BNSD" || inputLayoutStr == "BNSD_BSND") {
        s = tempQ->GetStorageShape().GetDim(QUERY_DIM_2);
    } else if (inputLayoutStr == "TND" || inputLayoutStr == "TND_NTD") {
        if (isMaxWorkspace) {
            t = tempQ->GetStorageShape().GetDim(QUERY_DIM_0);
            s = tempQ->GetStorageShape().GetDim(QUERY_DIM_0);
        } else {
            const gert::Tensor *actualSeqLength = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
            int64_t actSeqLenDims = (actualSeqLength != nullptr) ? actualSeqLength->GetShapeSize() : 0;
            OP_CHECK_IF(((actualSeqLength == nullptr) || (actSeqLenDims == 0) ||
                (actualSeqLength->GetData<int64_t>() == nullptr)),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                "When layout is TND/TND_NTD, actualSequenceLengthQ is required!"),
                return ge::GRAPH_FAILED);
            s = actualSeqLength->GetData<int64_t>()[0];
            for (int i = 1; i < actualSeqLength->GetShapeSize(); ++i) {
                s = std::max(s, actualSeqLength->GetData<int64_t>()[i] - actualSeqLength->GetData<int64_t>()[i - 1]);
            }
            t = actualSeqLength->GetData<int64_t>()[actualSeqLength->GetShapeSize() - 1];
        }
    } else if (inputLayoutStr == "NTD" || inputLayoutStr == "NTD_TND") {
        if (isMaxWorkspace) {
            t = tempQ->GetStorageShape().GetDim(QUERY_DIM_1);
            s = tempQ->GetStorageShape().GetDim(QUERY_DIM_1);
        } else {
            const gert::Tensor *actualSeqLength = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
            int64_t actSeqLenDims = (actualSeqLength != nullptr) ? actualSeqLength->GetShapeSize() : 0;
            OP_CHECK_IF(((actualSeqLength == nullptr) || (actSeqLenDims == 0) ||
                (actualSeqLength->GetData<int64_t>() == nullptr)),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                "When layout is NTD/NTD_TND, actualSequenceLengthQ is required!"),
                return ge::GRAPH_FAILED);
            s = actualSeqLength->GetData<int64_t>()[0];
            for (int i = 1; i < actualSeqLength->GetShapeSize(); ++i) {
                s = std::max(s, actualSeqLength->GetData<int64_t>()[i] - actualSeqLength->GetData<int64_t>()[i - 1]);
            }
            t = actualSeqLength->GetData<int64_t>()[actualSeqLength->GetShapeSize() - 1];
        }
    } else {
        s = tempQ->GetStorageShape().GetDim(1);
    }
    if (inputLayoutStr == "NSD") { // 当前已没有NSD, 回主线后在FIA tiling Arch38内删除
        b = 1;
        OP_CHECK_IF((tempQ->GetStorageShape().GetDimNum() != QUERY_DIM_3),
            OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "input shape dim should be 3!"),
            return ge::GRAPH_FAILED);
        queryD = tempQ->GetStorageShape().GetDim(QUERY_DIM_2);
        valueD = tempV->GetStorageShape().GetDim(VALUE_DIM_2);
        OP_CHECK_IF(((queryD == valueD) && (tempQ->GetStorageShape() != tempOut->GetStorageShape())),
            OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
            "Layout is NSD and Query shape size[%ld, %ld, %ld] does NOT match Attention Out shape size[%ld, %ld, %ld]!",
            tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1),
            tempQ->GetStorageShape().GetDim(QUERY_DIM_2), tempOut->GetStorageShape().GetDim(0),
            tempOut->GetStorageShape().GetDim(1), tempOut->GetStorageShape().GetDim(OUT_DIM_2)),
            return ge::GRAPH_FAILED);
    } else if (inputLayoutStr == "TND" || inputLayoutStr == "TND_NTD") {
        OP_CHECK_IF((tempQ->GetStorageShape().GetDimNum() != QUERY_DIM_3),
            OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
            "The current layout is %s, input query shape dim(%zu) should be 3!", inputLayoutStr.c_str(),
            tempQ->GetStorageShape().GetDimNum()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((tempOut->GetStorageShape().GetDimNum() != QUERY_DIM_3),
            OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
            "The current layout is %s, attention out shape dim(%zu) should be 3!", inputLayoutStr.c_str(),
            tempOut->GetStorageShape().GetDimNum()),
            return ge::GRAPH_FAILED);
        queryD = tempQ->GetStorageShape().GetDim(QUERY_DIM_2);
        valueD = tempV->GetStorageShape().GetDim(VALUE_DIM_2);
        if (inputLayoutStr == "TND") {
            OP_CHECK_IF(((queryD == valueD) && (tempQ->GetStorageShape() != tempOut->GetStorageShape())),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                "Layout is TND and Query shape size[%ld, %ld, %ld] does NOT match Attention Out shape size[%ld, %ld, "
                "%ld]!",
                tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1),
                tempQ->GetStorageShape().GetDim(QUERY_DIM_2), tempOut->GetStorageShape().GetDim(0),
                tempOut->GetStorageShape().GetDim(1), tempOut->GetStorageShape().GetDim(OUT_DIM_2)),
                return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(
                ((queryD == valueD) && ((tempQ->GetStorageShape().GetDim(0) != tempOut->GetStorageShape().GetDim(1)) ||
                (tempQ->GetStorageShape().GetDim(1) != tempOut->GetStorageShape().GetDim(0)) ||
                (tempQ->GetStorageShape().GetDim(QUERY_DIM_2) != tempOut->GetStorageShape().GetDim(QUERY_DIM_2)))),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                "Layout is TND_NTD and Query shape size[%ld, %ld, %ld] does NOT match Attention Out shape size[%ld, "
                "%ld, %ld]!",
                tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1),
                tempQ->GetStorageShape().GetDim(QUERY_DIM_2), tempOut->GetStorageShape().GetDim(0),
                tempOut->GetStorageShape().GetDim(1), tempOut->GetStorageShape().GetDim(OUT_DIM_2)),
                return ge::GRAPH_FAILED);
        }
    } else if (inputLayoutStr == "NTD" || inputLayoutStr == "NTD_TND") {
        OP_CHECK_IF((tempQ->GetStorageShape().GetDimNum() != QUERY_DIM_3),
            OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
            "The current layout is %s, input query shape dim(%zu) should be 3!", inputLayoutStr.c_str(),
            tempQ->GetStorageShape().GetDimNum()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((tempOut->GetStorageShape().GetDimNum() != QUERY_DIM_3),
            OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
            "The current layout is %s, attention out shape dim(%zu) should be 3!", inputLayoutStr.c_str(),
            tempOut->GetStorageShape().GetDimNum()),
            return ge::GRAPH_FAILED);
        queryD = tempQ->GetStorageShape().GetDim(QUERY_DIM_2);
        valueD = tempV->GetStorageShape().GetDim(VALUE_DIM_2);
        if (inputLayoutStr == "NTD") {
            OP_CHECK_IF(((queryD == valueD) && (tempQ->GetStorageShape() != tempOut->GetStorageShape())),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                "Layout is NTD and Query shape size[%ld, %ld, %ld] does NOT match Attention Out shape size[%ld, %ld, "
                "%ld]!",
                tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1),
                tempQ->GetStorageShape().GetDim(QUERY_DIM_2), tempOut->GetStorageShape().GetDim(0),
                tempOut->GetStorageShape().GetDim(1), tempOut->GetStorageShape().GetDim(OUT_DIM_2)),
                return ge::GRAPH_FAILED);
        } else if (inputLayoutStr == "NTD_TND") {
            OP_CHECK_IF(
                ((queryD == valueD) && ((tempQ->GetStorageShape().GetDim(0) != tempOut->GetStorageShape().GetDim(1)) ||
                (tempQ->GetStorageShape().GetDim(1) != tempOut->GetStorageShape().GetDim(0)) ||
                (tempQ->GetStorageShape().GetDim(QUERY_DIM_2) != tempOut->GetStorageShape().GetDim(OUT_DIM_2)))),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                "Layout is NTD_TND and Query shape size[%ld, %ld, %ld] does NOT match Attention Out shape size[%ld, "
                "%ld, %ld]!",
                tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1),
                tempQ->GetStorageShape().GetDim(QUERY_DIM_2), tempOut->GetStorageShape().GetDim(0),
                tempOut->GetStorageShape().GetDim(1), tempOut->GetStorageShape().GetDim(OUT_DIM_2)),
                return ge::GRAPH_FAILED);
        }
    } else if (inputLayoutStr == "BSH") {
        OP_CHECK_IF((tempQ->GetStorageShape().GetDimNum() != QUERY_DIM_3),
            OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
            "The current layout is %s, input query shape dim(%zu) should be 3!", inputLayoutStr.c_str(),
            tempQ->GetStorageShape().GetDimNum()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((tempOut->GetStorageShape().GetDimNum() != QUERY_DIM_3),
            OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
            "The current layout is %s, attention out shape dim(%zu) should be 3!", inputLayoutStr.c_str(),
            tempOut->GetStorageShape().GetDimNum()),
            return ge::GRAPH_FAILED);
        queryD = tempQ->GetStorageShape().GetDim(QUERY_DIM_2) / tempN;
        valueD = tempV->GetStorageShape().GetDim(VALUE_DIM_2) / tempKVN;
        OP_CHECK_IF(((queryD == valueD) && (tempQ->GetStorageShape() != tempOut->GetStorageShape())),
            OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
            "Layout is BSH and Query shape size[%ld, %ld, %ld] does NOT match Attention Out shape size[%ld, %ld, %ld]!",
            tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1),
            tempQ->GetStorageShape().GetDim(QUERY_DIM_2), tempOut->GetStorageShape().GetDim(0),
            tempOut->GetStorageShape().GetDim(1), tempOut->GetStorageShape().GetDim(OUT_DIM_2)),
            return ge::GRAPH_FAILED);
    } else if (inputLayoutStr == "BNSD_BSND" || inputLayoutStr == "BNSD" || inputLayoutStr == "BSND" ||
        inputLayoutStr == "BSND_BNSD") {
        OP_CHECK_IF((tempQ->GetStorageShape().GetDimNum() != QUERY_DIM_4),
            OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
            "The current layout is %s, input query shape dim(%zu) should be 4!", inputLayoutStr.c_str(),
            tempQ->GetStorageShape().GetDimNum()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((tempOut->GetStorageShape().GetDimNum() != QUERY_DIM_4),
            OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
            "The current layout is %s, attention out shape dim(%zu) should be 4!", inputLayoutStr.c_str(),
            tempOut->GetStorageShape().GetDimNum()),
            return ge::GRAPH_FAILED);
        queryD = tempQ->GetStorageShape().GetDim(QUERY_DIM_3);
        valueD = tempV->GetStorageShape().GetDim(VALUE_DIM_3);
        if (inputLayoutStr == "BNSD_BSND" || inputLayoutStr == "BSND_BNSD") {
            OP_CHECK_IF(
                ((queryD == valueD) && ((tempQ->GetStorageShape().GetDim(0) != tempOut->GetStorageShape().GetDim(0)) ||
                (tempQ->GetStorageShape().GetDim(1) != tempOut->GetStorageShape().GetDim(OUT_DIM_2)) ||
                (tempQ->GetStorageShape().GetDim(QUERY_DIM_2) != tempOut->GetStorageShape().GetDim(1)) ||
                (tempQ->GetStorageShape().GetDim(QUERY_DIM_3) != tempOut->GetStorageShape().GetDim(OUT_DIM_3)))),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                "Layout is %s and Query shape size[%ld, %ld, %ld, %ld] does NOT match Attention Out shape size[%ld, "
                "%ld, %ld, %ld]!",
                inputLayoutStr.c_str(), tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1),
                tempQ->GetStorageShape().GetDim(QUERY_DIM_2), tempQ->GetStorageShape().GetDim(QUERY_DIM_3),
                tempOut->GetStorageShape().GetDim(0), tempOut->GetStorageShape().GetDim(1),
                tempOut->GetStorageShape().GetDim(OUT_DIM_2), tempOut->GetStorageShape().GetDim(OUT_DIM_3)),
                return ge::GRAPH_FAILED);
        } else if (inputLayoutStr == "BNSD") {
            OP_CHECK_IF(((queryD == valueD) && (tempQ->GetStorageShape() != tempOut->GetStorageShape())),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                "Layout is BNSD and Query shape size[%ld, %ld, %ld, %ld] does NOT match Attention Out shape size[%ld, "
                "%ld, %ld, %ld]!",
                tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1),
                tempQ->GetStorageShape().GetDim(QUERY_DIM_2), tempQ->GetStorageShape().GetDim(QUERY_DIM_3),
                tempOut->GetStorageShape().GetDim(0), tempOut->GetStorageShape().GetDim(1),
                tempOut->GetStorageShape().GetDim(OUT_DIM_2), tempOut->GetStorageShape().GetDim(OUT_DIM_3)),
                return ge::GRAPH_FAILED);
        }
    } else if (inputLayoutStr == "BSH_BNSD" || inputLayoutStr == "BSH_NBSD") {
        OP_CHECK_IF((tempQ->GetStorageShape().GetDimNum() != QUERY_DIM_3),
            OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
            "The current layout is %s, input query shape dim(%zu) should be 3!", inputLayoutStr.c_str(),
            tempQ->GetStorageShape().GetDimNum()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((tempOut->GetStorageShape().GetDimNum() != QUERY_DIM_4),
            OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
            "The current layout is %s, attention out shape dim(%zu) should be 4!", inputLayoutStr.c_str(),
            tempOut->GetStorageShape().GetDimNum()),
            return ge::GRAPH_FAILED);
        queryD = tempQ->GetStorageShape().GetDim(QUERY_DIM_2) / tempN;
        valueD = tempV->GetStorageShape().GetDim(VALUE_DIM_2) / tempKVN;
        if (inputLayoutStr == "BSH_BNSD") {
            OP_CHECK_IF(((queryD == valueD) &&
                ((tempQ->GetStorageShape().GetDim(QUERY_DIM_0) != tempOut->GetStorageShape().GetDim(0)) ||
                (tempQ->GetStorageShape().GetDim(QUERY_DIM_1) != tempOut->GetStorageShape().GetDim(OUT_DIM_2)) ||
                queryD != tempOut->GetStorageShape().GetDim(OUT_DIM_3)) ||
                tempN != tempOut->GetStorageShape().GetDim(OUT_DIM_1)),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                "Layout is %s and Query shape size[%ld, %ld, %ld] does NOT match Attention Out shape size[%ld, %ld, "
                "%ld, %ld]!",
                inputLayoutStr.c_str(), tempQ->GetStorageShape().GetDim(QUERY_DIM_0),
                tempQ->GetStorageShape().GetDim(QUERY_DIM_1), tempQ->GetStorageShape().GetDim(QUERY_DIM_2),
                tempOut->GetStorageShape().GetDim(0), tempOut->GetStorageShape().GetDim(1),
                tempOut->GetStorageShape().GetDim(OUT_DIM_2), tempOut->GetStorageShape().GetDim(OUT_DIM_3)),
                return ge::GRAPH_FAILED);
        } else if (inputLayoutStr == "BSH_NBSD") {
            OP_CHECK_IF(((queryD == valueD) &&
                ((tempQ->GetStorageShape().GetDim(QUERY_DIM_0) != tempOut->GetStorageShape().GetDim(OUT_DIM_1)) ||
                (tempQ->GetStorageShape().GetDim(QUERY_DIM_1) != tempOut->GetStorageShape().GetDim(OUT_DIM_2)) ||
                queryD != tempOut->GetStorageShape().GetDim(OUT_DIM_3)) ||
                tempN != tempOut->GetStorageShape().GetDim(0)),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                "Layout is %s and Query shape size[%ld, %ld, %ld] does NOT match Attention Out shape size[%ld, %ld, "
                "%ld, %ld]!",
                inputLayoutStr.c_str(), tempQ->GetStorageShape().GetDim(QUERY_DIM_0),
                tempQ->GetStorageShape().GetDim(QUERY_DIM_1), tempQ->GetStorageShape().GetDim(QUERY_DIM_2),
                tempOut->GetStorageShape().GetDim(0), tempOut->GetStorageShape().GetDim(1),
                tempOut->GetStorageShape().GetDim(OUT_DIM_2), tempOut->GetStorageShape().GetDim(OUT_DIM_3)),
                return ge::GRAPH_FAILED);
        }
    } else if (inputLayoutStr == "BSND_NBSD" || inputLayoutStr == "BNSD_NBSD") {
        OP_CHECK_IF((tempQ->GetStorageShape().GetDimNum() != QUERY_DIM_4),
            OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
            "The current layout is %s, input query shape dim(%zu) should be 4!", inputLayoutStr.c_str(),
            tempQ->GetStorageShape().GetDimNum()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((tempOut->GetStorageShape().GetDimNum() != QUERY_DIM_4),
            OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
            "The current layout is %s, attention out shape dim(%zu) should be 4!", inputLayoutStr.c_str(),
            tempOut->GetStorageShape().GetDimNum()),
            return ge::GRAPH_FAILED);
        queryD = tempQ->GetStorageShape().GetDim(QUERY_DIM_3);
        valueD = tempV->GetStorageShape().GetDim(VALUE_DIM_3);
        if (inputLayoutStr == "BSND_NBSD") {
            OP_CHECK_IF(((queryD == valueD) &&
                ((tempQ->GetStorageShape().GetDim(0) != tempOut->GetStorageShape().GetDim(OUT_DIM_1)) ||
                (tempQ->GetStorageShape().GetDim(QUERY_DIM_1) != tempOut->GetStorageShape().GetDim(OUT_DIM_2)) ||
                (tempQ->GetStorageShape().GetDim(QUERY_DIM_2) != tempOut->GetStorageShape().GetDim(0)) ||
                (tempQ->GetStorageShape().GetDim(QUERY_DIM_3) != tempOut->GetStorageShape().GetDim(OUT_DIM_3)))),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                "Layout is %s and Query shape size[%ld, %ld, %ld, %ld] does NOT match Attention Out shape size[%ld, "
                "%ld, %ld, %ld]!",
                inputLayoutStr.c_str(), tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1),
                tempQ->GetStorageShape().GetDim(QUERY_DIM_2), tempQ->GetStorageShape().GetDim(QUERY_DIM_3),
                tempOut->GetStorageShape().GetDim(0), tempOut->GetStorageShape().GetDim(1),
                tempOut->GetStorageShape().GetDim(OUT_DIM_2), tempOut->GetStorageShape().GetDim(OUT_DIM_3)),
                return ge::GRAPH_FAILED);
        } else if (inputLayoutStr == "BNSD_NBSD") {
            OP_CHECK_IF(((queryD == valueD) &&
                ((tempQ->GetStorageShape().GetDim(0) != tempOut->GetStorageShape().GetDim(OUT_DIM_1)) ||
                (tempQ->GetStorageShape().GetDim(QUERY_DIM_1) != tempOut->GetStorageShape().GetDim(0)) ||
                (tempQ->GetStorageShape().GetDim(QUERY_DIM_2) != tempOut->GetStorageShape().GetDim(OUT_DIM_2)) ||
                (tempQ->GetStorageShape().GetDim(QUERY_DIM_3) != tempOut->GetStorageShape().GetDim(OUT_DIM_3)))),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                "Layout is %s and Query shape size[%ld, %ld, %ld, %ld] does NOT match Attention Out shape size[%ld, "
                "%ld, %ld, %ld]!",
                inputLayoutStr.c_str(), tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1),
                tempQ->GetStorageShape().GetDim(QUERY_DIM_2), tempQ->GetStorageShape().GetDim(QUERY_DIM_3),
                tempOut->GetStorageShape().GetDim(0), tempOut->GetStorageShape().GetDim(1),
                tempOut->GetStorageShape().GetDim(OUT_DIM_2), tempOut->GetStorageShape().GetDim(OUT_DIM_3)),
                return ge::GRAPH_FAILED);
        }
    } else {
        OP_LOGE(context_->GetNodeName(),
            "Invalid input layout:%s. Currently only "
            "TND/NTD/BSH/BNSD/BSND/BSND_BNSD/BNSD_BSND/BSH_BNSD/BSND_NBSD/BNSD_NBSD/BSH_NBSD layout are supported!",
            inputLayoutStr.c_str());
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF((queryD > DLIMIT),
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
        "D of query should be less than or equal to 512, but d = %u!", queryD),
        return ge::GRAPH_FAILED);
    bool inputOutputIsNullPtr = (context_->GetInputDesc(QUERY_INDEX) == nullptr) ||
        (context_->GetInputDesc(KEY_INDEX) == nullptr) || (context_->GetInputDesc(VALUE_INDEX) == nullptr) ||
        (context_->GetOutputDesc(ATTENTION_OUT_INDEX) == nullptr) ||
        (context_->GetInputShape(QUERY_INDEX) == nullptr) || (context_->GetInputShape(KEY_INDEX) == nullptr) ||
        (context_->GetInputShape(VALUE_INDEX) == nullptr) || (context_->GetOutputShape(ATTENTION_OUT_INDEX) == nullptr);
    OP_CHECK_IF(inputOutputIsNullPtr,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "q, k, v or attenOut is nullptr!"),
        return ge::GRAPH_FAILED);
    auto qDType = context_->GetInputDesc(QUERY_INDEX)->GetDataType();
    auto kDType = context_->GetInputDesc(KEY_INDEX)->GetDataType();
    // IFA非MLA或伪量化场景走IFA模板
    // IFA的MLA或PFA非伪量化场景走PFA模板
    if ((qDType != kDType) ||
        ((s == 1) && ((inputLayoutStr == "BSH") || (inputLayoutStr == "BNSD") || (inputLayoutStr == "BSND")) &&
        ((context_->GetOptionalInputShape(QUERY_ROPE_INDEX) == nullptr) &&
        (context_->GetOptionalInputShape(KEY_ROPE_INDEX) == nullptr)))) {
        auto platformInfoPtr = context_->GetPlatformInfo();
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        if (qDType == kDType) {
            usingIFA = false;
        } else {
            usingIFA = true;
        }
    }
    if (usingIFA) {
        // IFA tiling path
        IncreFlashAttentionContext ifaContext{};
        auto ret = ConvertContextToParamsIFA(*context_, ifaContext, isMaxWorkspace);
        if (ret != ge::GRAPH_SUCCESS) {
            OP_LOGE(context_->GetNodeName(), "Error occored while convert tilingContext to ifa context!");
            return ret;
        }
        IFATilingArch38 ifaTilingArch38(context_);
        ret = ifaTilingArch38.DoSubOpTiling(ifaContext);
        OP_CHECK_IF(ret == ge::GRAPH_FAILED,
            OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "failed in FIA DoSubOpTiling"),
            return ge::GRAPH_FAILED);
        return ret;
    } else {
        // PFA tiling process
        constexpr int64_t D_ALIGN_32 = 32;
        constexpr int64_t D_ALIGN_16 = 16;

        PromptFlashAttentionTilingData pfaTilingData;
        PromptFlashAttentionTilingArch38 pfa_tiling(context_);
        ContextParamsForPFATiling contextParamsForPFATiling;
        PromptFlashAttentionCompileInfo tempCompileInfoPtr;

        OP_CHECK_IF((attrs->GetAttrPointer<uint64_t>(ANTIQUANT_MODE_INDEX) != nullptr) &&
            (*attrs->GetAttrPointer<uint64_t>(ANTIQUANT_MODE_INDEX) != 0),
            OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "antiquant_mode is not supported!"),
            return ge::GRAPH_FAILED);

        auto platformInfoPtr = context_->GetPlatformInfo();
        OP_CHECK_IF(platformInfoPtr == nullptr,
            OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "platformInfoPtr is null!"), return ge::GRAPH_FAILED);
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        tempCompileInfoPtr.aivNum = ascendcPlatform.GetCoreNumAiv();
        tempCompileInfoPtr.aicNum = ascendcPlatform.GetCoreNumAic();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, tempCompileInfoPtr.ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, tempCompileInfoPtr.l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, tempCompileInfoPtr.l0CSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, tempCompileInfoPtr.l0ASize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, tempCompileInfoPtr.l0BSize);
        tempCompileInfoPtr.socShortName = ascendcPlatform.GetSocVersion();
        tempCompileInfoPtr.defaultSysWorkspaceSize = 0;

        contextParamsForPFATiling.compileInfoPtr = &tempCompileInfoPtr;
        auto ret = ConvertContextToParamsPFA(context_, contextParamsForPFATiling, isMaxWorkspace);
        if (ret != ge::GRAPH_SUCCESS) {
            OP_LOGE(context_->GetNodeName(), "Error occored while convert tilingContext to PFA context!");
            return ret;
        }
        if (lseFlag != false) {
            if (!pfa_tiling.CheckNonEmptyShapeExceptions(contextParamsForPFATiling,
                contextParamsForPFATiling.lseoutputShape, "softmaxLse")) {
                return ge::GRAPH_FAILED;
            }
            OP_CHECK_IF(((tempLse == nullptr)),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                "SoftmaxLse shape is null, but SoftmaxLseFlag is true!"),
                return ge::GRAPH_FAILED);

            if (!qOutEmptyTensor) { // q、out为空时，lse为空则不输出，不为空则输出inf，不做拦截
                if (inputLayoutStr == "TND" || inputLayoutStr == "TND_NTD") {
                    OP_CHECK_IF(((tempLse->GetStorageShape().GetDimNum() != QUERY_DIM_3)), // 3：lse shape TN1
                        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                        "Layout is TND/TND_NTD SoftmaxLse shape dim should be 3, but got %zu!",
                        tempLse->GetStorageShape().GetDimNum()),
                        return ge::GRAPH_FAILED);
                    OP_CHECK_IF((((tempLse->GetStorageShape().GetDim(QUERY_DIM_0) != t) ||
                        (tempLse->GetStorageShape().GetDim(QUERY_DIM_1) !=
                        tempN) || // 0: the first dimension 1: the second dimension
                        (tempLse->GetStorageShape().GetDim(QUERY_DIM_2) != 1))), // 2: the third dimension
                        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                        "Layout is TND SoftmaxLse shape size[%ld, %ld, %ld] does not match TN1[%ld, %u, 1]!",
                        tempLse->GetStorageShape().GetDim(QUERY_DIM_0),
                        tempLse->GetStorageShape().GetDim(
                        QUERY_DIM_1), // 0: the first dimension 1: the second dimension
                        tempLse->GetStorageShape().GetDim(QUERY_DIM_2), t, tempN), // 2: the third dimension
                        return ge::GRAPH_FAILED);
                } else if (inputLayoutStr == "NTD" || inputLayoutStr == "NTD_TND") {
                    OP_CHECK_IF(((tempLse->GetStorageShape().GetDimNum() != QUERY_DIM_3)), // 3：lse shape TN1
                        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                        "Layout is NTD/NTD_TND SoftmaxLse shape dim should be 3, but got %zu!",
                        tempLse->GetStorageShape().GetDimNum()),
                        return ge::GRAPH_FAILED);
                    OP_CHECK_IF((((tempLse->GetStorageShape().GetDim(QUERY_DIM_1) != tempN) ||
                        (tempLse->GetStorageShape().GetDim(QUERY_DIM_0) !=
                        t) || // 0: the first dimension 1: the second dimension
                        (tempLse->GetStorageShape().GetDim(QUERY_DIM_2) != 1))), // 2: the third dimension
                        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                        "Layout is NTD SoftmaxLse shape size[%ld, %ld, %ld] does not match TN1[%ld, %u, 1]!",
                        tempLse->GetStorageShape().GetDim(QUERY_DIM_0),
                        tempLse->GetStorageShape().GetDim(
                        QUERY_DIM_1), // 0: the first dimension 1: the second dimension
                        tempLse->GetStorageShape().GetDim(QUERY_DIM_2), t, tempN), // 2: the third dimension
                        return ge::GRAPH_FAILED);
                } else {
                    OP_CHECK_IF(((tempLse->GetStorageShape().GetDimNum() != QUERY_DIM_4)), // 4：lse shape BNS1
                        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                        "Layout is %s, SoftmaxLse shape dim should be 4, but got %zu!", inputLayoutStr.c_str(),
                        tempLse->GetStorageShape().GetDimNum()),
                        return ge::GRAPH_FAILED);
                    OP_CHECK_IF((((tempLse->GetStorageShape().GetDim(QUERY_DIM_0) != b) ||
                        (tempLse->GetStorageShape().GetDim(QUERY_DIM_1) !=
                        tempN) || // 0: the first dimension 1: the second dimension
                        (tempLse->GetStorageShape().GetDim(QUERY_DIM_2) != s) ||
                        (tempLse->GetStorageShape().GetDim(QUERY_DIM_3) !=
                        1))), // 2: the third dimension 3: the fourth dimension
                        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                        "SoftmaxLse shape size[%ld, %ld, %ld, %ld] does not match BNS1[%ld, %u, %ld, 1]!",
                        tempLse->GetStorageShape().GetDim(QUERY_DIM_0),
                        tempLse->GetStorageShape().GetDim(
                        QUERY_DIM_1), // 0: the first dimension 1: the second dimension
                        tempLse->GetStorageShape().GetDim(QUERY_DIM_2), tempLse->GetStorageShape().GetDim(QUERY_DIM_3),
                        b, tempN, s), // 2: the third dimension 3: the fourth dimension
                        return ge::GRAPH_FAILED);
                }
            }
        }
        if (ascendcPlatform.GetCurNpuArch() != NpuArch::DAV_5102) {
            OP_CHECK_IF((((contextParamsForPFATiling.inputDataType == ge::DT_INT8) ||
                (contextParamsForPFATiling.kDataType == ge::DT_INT8) ||
                (contextParamsForPFATiling.outputDataType == ge::DT_INT8)) &&
                (queryD % D_ALIGN_32 != 0)),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                "D(%u) of query should be 32 elements aligned when int8 is involved!", queryD),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF((queryD % D_ALIGN_16 != 0),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                "D(%u) should be 16 elements aligned when FP16/BF16 dtype!", queryD),
                return ge::GRAPH_FAILED);
        }
        ret = pfa_tiling.DoSubOpTiling(pfaTilingData, contextParamsForPFATiling);
        OP_CHECK_IF(ret == ge::GRAPH_FAILED,
            OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "failed in FIA DoSubOpTiling"),
            return ge::GRAPH_FAILED);
        return ret;
    }
}
} // namespace optiling