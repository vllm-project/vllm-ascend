/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_v2_sink_tiling_v3.cpp
 * \brief
 */
#include "fused_infer_attention_score_v2_sink_tiling_v3.h"
#include "fused_infer_attention_score_v2_sink_tiling_check.h"
#include "fused_infer_attention_score_v2_sink_tiling_info_parser.h"
#include "fia_tiling_nonquant_mla_sink.h"
#include "fia_tiling_nonquant_sink.h"
#include "fia_tiling_templates_registry.h"

using namespace AscendC;
namespace optiling {
// FIA新TilingKey, 18位编码, IFA原有TilingKey是17位, 新的TilingKey只是把最高位从1X->10X
// Gqa NoQuant PA dtype: Q=FP16 KV=FP16 OUT=FP16
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScoreV2Sink, FusedInferAttentionScoreV2SinkTilingData)

constexpr size_t DIM_NZ = 5;
constexpr uint32_t NZ_D1_IDX = 2;
constexpr uint32_t NZ_D0_IDX = 4;
constexpr uint32_t TND_NTD_D_IDX = 2;
constexpr int64_t HEAD_DIM_192 = 192;
constexpr uint64_t DIM_NUM_3 = 3;
constexpr uint64_t DIM_NUM_4 = 4;

FIA_EXTERN_C ge::graphStatus TilingFusedInferAttentionScoreV2SinkV3(gert::TilingContext *context)
{
    FiaTilingInfo fiaInfo;
    FiaInfoParser fiaInfoParser(context);
    if (fiaInfoParser.Parse(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // Check函数只做校验，不能修改fiaInfo中的信息
    if (TilingCheck::Check(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return FiaTilingRegistry::GetInstance().DoTilingImpl(context, &fiaInfo);
}

bool GetPaValueD(gert::TilingContext *context, int64_t &valueD)
{
    auto attrs = context->GetAttrs();
    int64_t numHeads = static_cast<int64_t>(*attrs->GetAttrPointer<uint32_t>(ATTR_N_INDEX));
    int64_t numKvHeads = static_cast<int64_t>(*attrs->GetAttrPointer<uint32_t>(ATTR_NUM_KV_HEADS_INDEX));
    if (numKvHeads == 0) {
        numKvHeads = numHeads;
    }
    auto vStorageShape = context->GetInputShape(VALUE_INDEX)->GetShape();
    if (vStorageShape.GetDimNum() == DIM_BSH) {
        valueD = vStorageShape.GetDim(BSH_H_IDX) / numKvHeads; // BnBsH
    } else if (vStorageShape.GetDimNum() == DIM_BNSD_OR_BSND) {
        valueD = vStorageShape.GetDim(BNSD_D_IDX); // BnNBsD
    } else if (vStorageShape.GetDimNum() == DIM_NZ) {
        valueD = vStorageShape.GetDim(NZ_D1_IDX) * vStorageShape.GetDim(NZ_D0_IDX); // NZ: BnND1BsD0
    } else {
        return false;
    }
    return true;
}

bool GetValueD(gert::TilingContext *context, int64_t &valueD)
{
    auto vShape = context->GetInputShape(VALUE_INDEX);
    if (vShape == nullptr) {
        return false;
    }
    auto vStorageShape = vShape->GetShape();

    bool isPageAttention = context->GetOptionalInputShape(BLOCK_TABLE_INDEX) != nullptr;
    if (isPageAttention) {
        return GetPaValueD(context, valueD);
    }

    auto attrs = context->GetAttrs();
    int64_t numHeads = static_cast<int64_t>(*attrs->GetAttrPointer<uint32_t>(ATTR_N_INDEX));
    int64_t numKvHeads = static_cast<int64_t>(*attrs->GetAttrPointer<uint32_t>(ATTR_NUM_KV_HEADS_INDEX));
    if (numKvHeads == 0) {
        numKvHeads = numHeads;
    }
    const std::string inputLayoutStr = std::string(context->GetAttrs()->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
    if (inputLayoutStr == "BNSD_BSND" ||
        inputLayoutStr == "BSND_BNSD" ||
        inputLayoutStr == "BNSD_NBSD" ||
        inputLayoutStr == "BSND_NBSD" ||
        inputLayoutStr == "BNSD" ||
        inputLayoutStr == "BSND") {
        if (vStorageShape.GetDimNum() != DIM_BNSD_OR_BSND) {
            return false;
        }
        valueD = vStorageShape.GetDim(BNSD_D_IDX);
    } else if (inputLayoutStr == "BSH" ||
        inputLayoutStr == "BSH_NBSD" ||
        inputLayoutStr == "BSH_BNSD") {
        if (vStorageShape.GetDimNum() != DIM_BSH) {
            return false;
        }
        valueD = vStorageShape.GetDim(BSH_H_IDX) / numKvHeads;
    } else if (inputLayoutStr == "TND" ||
        inputLayoutStr == "NTD" ||
        inputLayoutStr == "TND_NTD" ||
        inputLayoutStr == "NTD_TND") {
        if (vStorageShape.GetDimNum() != DIM_TND) {
            return false;
        }
        valueD = vStorageShape.GetDim(TND_NTD_D_IDX);
    } else {
        return false;
    }

    return true;
}

bool GetQkvD(gert::TilingContext *context, int64_t &queryD, int64_t &queryRopeD, int64_t &valueD)
{
    auto qShape = context->GetInputShape(QUERY_INDEX);
    auto qRopeShape = context->GetOptionalInputShape(QUERY_ROPE_INDEX);
    if (qShape == nullptr) {
        return false;
    }
    auto qStorageShape = qShape->GetStorageShape();

    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return false;
    }

    int64_t numHeads = static_cast<int64_t>(*attrs->GetAttrPointer<uint32_t>(ATTR_N_INDEX));
    const std::string inputLayoutStr = std::string(context->GetAttrs()->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
    if (inputLayoutStr == "BNSD_BSND" ||
        inputLayoutStr == "BSND_BNSD" ||
        inputLayoutStr == "BNSD_NBSD" ||
        inputLayoutStr == "BSND_NBSD" ||
        inputLayoutStr == "BNSD" ||
        inputLayoutStr == "BSND") {
        if (qStorageShape.GetDimNum() != DIM_BNSD_OR_BSND) {
            return false;
        }
        queryD = qStorageShape.GetDim(BNSD_D_IDX);
        if (qRopeShape != nullptr) {
            queryRopeD = qRopeShape->GetStorageShape().GetDim(BNSD_D_IDX);
        }
    } else if (inputLayoutStr == "BSH" ||
        inputLayoutStr == "BSH_NBSD" ||
        inputLayoutStr == "BSH_BNSD") {
        if (qStorageShape.GetDimNum() != DIM_BSH) {
            return false;
        }
        queryD = qStorageShape.GetDim(BSH_H_IDX) / numHeads;
        if (qRopeShape != nullptr) {
            queryRopeD = qRopeShape->GetStorageShape().GetDim(BSH_H_IDX) / numHeads;
        }
    } else if (inputLayoutStr == "TND" ||
        inputLayoutStr == "NTD" ||
        inputLayoutStr == "TND_NTD" ||
        inputLayoutStr == "NTD_TND") {
        if (qStorageShape.GetDimNum() != DIM_TND) {
            return false;
        }
        queryD = qStorageShape.GetDim(TND_NTD_D_IDX);
        if (qRopeShape != nullptr) {
            queryRopeD = qRopeShape->GetStorageShape().GetDim(TND_NTD_D_IDX);
        }
    } else {
        return false;
    }

    return GetValueD(context, valueD);
}

bool CheckGqaDSupport(gert::TilingContext *context)
{
    int64_t queryD = 0;
    int64_t queryRopeD = 0;
    int64_t valueD = 0;
    if (GetQkvD(context, queryD, queryRopeD, valueD) != true) {
        return false;
    }

    // D的组合(128+0,128)(64+0,64)(128+64,128)(192+0,128)(192+0,192)
    if ((queryD  == 128 && queryRopeD  == 0 && valueD == 128) || // 128: gqa qkvD
        (queryD  == 64 && queryRopeD  == 0 && valueD == 64) || // 64: gqa qkvD
        (queryD  == 128 && queryRopeD  == 64 && valueD == 128) || // 128: gqa qkvD, 64: mla ropeD
        (queryD  == 192 && queryRopeD  == 0 && valueD == 128) || // 192: gqa qkD, 128: gqa valueD
        (queryD  == 192 && queryRopeD  == 0 && valueD == 192)) { // 192: gqa qkvD(GLM-5.2 DSpark draft qk=v=192)
        return true;
    }

    return false;
}

bool CheckGqaInputLayoutSupport(gert::TilingContext *context)
{
    const std::string inputLayoutStr = std::string(context->GetAttrs()->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
    if (inputLayoutStr == "BNSD_BSND" ||
        inputLayoutStr == "BSND_BNSD" ||
        inputLayoutStr == "BNSD" ||
        inputLayoutStr == "BSND" ||
        inputLayoutStr == "BSH_BNSD" ||
        inputLayoutStr == "BSH" ||
        inputLayoutStr == "TND" ||
        inputLayoutStr == "NTD" ||
        inputLayoutStr == "NTD_TND") {
        return true;
    }
    return false;
}

bool IsEmptyTensor(gert::TilingContext *context)
{
    auto qShape = context->GetInputShape(QUERY_INDEX);
    if ((qShape != nullptr) && (qShape->GetStorageShape().GetShapeSize() == 0)) {
        return true;
    }

    auto attenoutShape = context->GetInputShape(ATTENTION_OUT_INDEX);
    if ((attenoutShape != nullptr) && (attenoutShape->GetStorageShape().GetShapeSize() == 0)) {
        return true;
    }

    bool softmaxLseFlag = *context->GetAttrs()->GetAttrPointer<bool>(SOFTMAX_LSE_FLAG_INDEX);
    if (softmaxLseFlag) {
        auto softmaxLseShape = context->GetOutputShape(SOFTMAX_LSE_INDEX);
        if ((softmaxLseShape != nullptr) && (softmaxLseShape->GetStorageShape().GetShapeSize() == 0)) {
            return true;
        }
    }
    bool softmaxMaxSumFlag = *context->GetAttrs()->GetAttrPointer<bool>(SOFTMAX_MAX_SUM_FLAG_INDEX);
    if (softmaxMaxSumFlag) {
        auto softmaxMaxShape = context->GetOutputShape(SOFTMAX_MAX_INDEX);
        if ((softmaxMaxShape != nullptr) && (softmaxMaxShape->GetStorageShape().GetShapeSize() == 0)) {
            return true;
        }
        auto softmaxSumShape = context->GetOutputShape(SOFTMAX_SUM_INDEX);
        if ((softmaxSumShape != nullptr) && (softmaxSumShape->GetStorageShape().GetShapeSize() == 0)) {
            return true;
        }
    }

    uint32_t keyBIdx = 0;
    while ((context->GetDynamicInputShape(KEY_INDEX, keyBIdx)) != nullptr) {
        const gert::StorageShape *keyShape =
            const_cast<gert::StorageShape *>(context->GetDynamicInputShape(KEY_INDEX, keyBIdx));
        if (keyShape->GetShape().GetShapeSize() == 0) {
            return true;
        }
        keyBIdx++;
    }

    uint32_t valueBIdx = 0;
    while ((context->GetDynamicInputShape(VALUE_INDEX, valueBIdx)) != nullptr) {
        const gert::StorageShape *valueShape =
            const_cast<gert::StorageShape *>(context->GetDynamicInputShape(VALUE_INDEX, valueBIdx));
        if (valueShape->GetShape().GetShapeSize() == 0) {
            return true;
        }
        valueBIdx++;
    }

    return false;
}

bool CheckGqaFeatureSupport(gert::TilingContext *context)
{
    auto pseShift = context->GetOptionalInputTensor(PSE_SHIFT_INDEX);
    auto queryPaddingSize = context->GetOptionalInputTensor(QUERY_PADDING_SIZE_INDEX);
    auto kvPaddingSize = context->GetOptionalInputTensor(KV_PADDING_SIZE_INDEX);
    auto quantScale2 = context->GetOptionalInputTensor(QUANT_SCALE2_INDEX);
    auto quantOffset2 = context->GetOptionalInputTensor(QUANT_OFFSET2_INDEX);
    if (pseShift != nullptr ||
        queryPaddingSize != nullptr ||
        kvPaddingSize != nullptr ||
        quantScale2 != nullptr ||
        quantOffset2 != nullptr) {
        return false;
    }

    return true;
}

bool CheckGqaConstrain(gert::TilingContext *context)
{
    if (CheckGqaInputLayoutSupport(context) &&
        CheckGqaDSupport(context) &&
        CheckGqaFeatureSupport(context)) {
            return true;
    }
    return false;
}

bool isNotLegacyGQA(gert::TilingContext *context)
{
    const std::string inputLayoutStr = std::string(context->GetAttrs()->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
    if (inputLayoutStr != "BSH" &&
        inputLayoutStr != "BSND" &&
        inputLayoutStr != "BNSD" &&
        inputLayoutStr != "BNSD_BSND" &&
        inputLayoutStr != "TND" &&
        inputLayoutStr != "NSD") {
        return true;
    }

    auto queryRope = context->GetOptionalInputTensor(QUERY_ROPE_INDEX);
    auto keyRope = context->GetOptionalInputTensor(KEY_ROPE_INDEX);
    int64_t queryD = 0;
    int64_t queryRopeD = 0;
    int64_t valueD = 0;
    if (GetQkvD(context, queryD, queryRopeD, valueD) != true) {
        return false;
    }

    if (queryD != valueD || queryRope != nullptr || keyRope != nullptr) {
        return true;
    }

    if (inputLayoutStr == "TND" && valueD == HEAD_DIM_192) {
        return false;
    }

    uint32_t keyDimNum = context->GetInputShape(KEY_INDEX)->GetShape().GetDimNum();
    uint32_t valueDimNum = context->GetInputShape(VALUE_INDEX)->GetShape().GetDimNum();
    if ((keyDimNum != DIM_NUM_3 && keyDimNum != DIM_NUM_4) ||
        (valueDimNum != DIM_NUM_3 && valueDimNum != DIM_NUM_4)) {
        return true;
    }

    return false;
}

bool CheckMlaInputLayoutSupport(const gert::TilingContext *context)
{
    const std::string inputLayoutStr = std::string(context->GetAttrs()->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
    if (inputLayoutStr == "BSH" ||
        inputLayoutStr == "BNSD" ||
        inputLayoutStr == "BSND" ||
        inputLayoutStr == "BNSD_NBSD" ||
        inputLayoutStr == "BSND_NBSD" ||
        inputLayoutStr == "BSH_NBSD" ||
        inputLayoutStr == "TND" ||
        inputLayoutStr == "TND_NTD") {
        return true;
    }

    return false;
}

bool CheckMlaDSupport(gert::TilingContext *context)
{
    int64_t queryD = 0;
    int64_t queryRopeD = 0;
    int64_t valueD = 0;
    if (GetQkvD(context, queryD, queryRopeD, valueD) != true) {
        return false;
    }

    if ((queryD  == 512 && queryRopeD  == 64 && valueD == 512)) { // 512: mla qkvD, 64: mla ropeD
        return true;
    }

    return false;
}

bool CheckMlaConstrain(gert::TilingContext *context)
{
    if (isNotLegacyGQA(context)) {
        return true;
    }

    if (CheckMlaInputLayoutSupport(context) &&
        CheckMlaDSupport(context)) {
        return true;
    }

    return false;
}

bool RouteToFia(gert::TilingContext *context)
{
    if ((context == nullptr) || context->GetAttrs() == nullptr ||
    (context->GetInputDesc(QUERY_INDEX) == nullptr) ||
    (context->GetInputDesc(KEY_INDEX) == nullptr)) {
        return false;
    }
    auto platformInfoPtr = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P) {
        return false;
    }

    ge::DataType qDataType = context->GetInputDesc(QUERY_INDEX)->GetDataType();
    ge::DataType kDataType = context->GetInputDesc(KEY_INDEX)->GetDataType();
    bool isRopeSplit = (context->GetOptionalInputTensor(QUERY_ROPE_INDEX) != nullptr &&
        context->GetOptionalInputTensor(KEY_ROPE_INDEX) != nullptr);
    if (isRopeSplit) {
        // MLA非量化
        if ((qDataType == ge::DT_FLOAT16 || qDataType == ge::DT_BF16) && (qDataType == kDataType)) {
            if (CheckGqaConstrain(context)) {
                OPS_LOG_I(context->GetNodeName(), "FIA RopeSplit GQA No quant.");
                return true;
            }
            if (CheckMlaConstrain(context)) {
                OPS_LOG_I(context->GetNodeName(), "FIA RopeSplit MLA No quant.");
                return true;
            }
            return false;
        }
    } else {
        // GQA非量化
        if ((qDataType == ge::DT_FLOAT16 || qDataType == ge::DT_BF16) && (qDataType == kDataType)) {
            OPS_LOG_I(context->GetNodeName(), "FIA GQA No quant.");
            return CheckGqaConstrain(context);
        }
    }
    return false;
}
} // namespace optiling
