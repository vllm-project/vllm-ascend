/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file mask_checker.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "register/op_def_registry.h"
#include "../fused_infer_attention_score_tiling_constants.h"
#include "mask_checker.h"

namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FIA;

// CheckSinglePara
ge::graphStatus MaskChecker::CheckDtypeAndFormat(const FiaTilingInfo &fiaInfo) const
{
    // AttentionMask data type must be int8/uint8/bool, and data format must be ND/NCHW/NHWC/NCDHW.
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(fiaInfo.opParamInfo.attenMask.desc, ATTEN_MASK_NAME)) {
        OP_LOGE_FOR_INVALID_DTYPE(fiaInfo.opName, "atten_mask",
            ToString(fiaInfo.opParamInfo.attenMask.desc->GetDataType()).c_str(), "int8, uint8 or bool");
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckFormatSupport(fiaInfo.opParamInfo.attenMask.desc, ATTEN_MASK_NAME)) {
        OP_LOGE_FOR_INVALID_FORMAT(fiaInfo.opName, ATTEN_MASK_NAME,
            ToString(fiaInfo.opParamInfo.attenMask.desc->GetOriginFormat()).c_str(),
            "ND, NCHW, NHWC or NCDHW");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckSparseMode(const FiaTilingInfo &fiaInfo) const
{
    // SparseMode only supports 0/1/2/3/4/9. SparseMode 9 only support in ascend910B.
    const std::vector<int32_t> sparseModeList = {SPARSE_MODE_NO_MASK, SPARSE_MODE_ALL_MASK, SPARSE_MODE_LEFT_UP,
                                                 SPARSE_MODE_RIGHT_DOWN, SPARSE_MODE_BAND, SPARSE_MODE_TREE};
    OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckValueSupport(fiaInfo.sparseMode, sparseModeList),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "sparse_mode",
                    std::to_string(fiaInfo.sparseMode).c_str(),
                    "SparseMode only supports 0/1/2/3/4/9"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// CheckFeature
ge::graphStatus MaskChecker::CheckAntiquantSparseMode(const FiaTilingInfo &fiaInfo) const
{
    OP_CHECK_IF(
        fiaInfo.s1Size == 1U && fiaInfo.sparseMode != SPARSE_MODE_NO_MASK,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "sparse_mode",
            std::to_string(fiaInfo.sparseMode).c_str(),
            "When S of query equal to 1, sparseMode only supports 0(defaultMask)"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckNoQuantIFAMLA(const FiaTilingInfo &fiaInfo)
{
    // For IFA MLA, input sparse mode only supports 0/3/4.
    enableIFAMLA = (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512);
    if (enableIFAMLA) {
        OP_CHECK_IF(
            fiaInfo.sparseMode != SPARSE_MODE_NO_MASK && fiaInfo.sparseMode != SPARSE_MODE_RIGHT_DOWN &&
                fiaInfo.sparseMode != SPARSE_MODE_BAND && fiaInfo.sparseMode != SPARSE_MODE_TREE,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "sparse_mode",
                std::to_string(fiaInfo.sparseMode).c_str(),
                "Only support sparse 0/3/4/9 when ifa mla is enabled"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckFullQuantIFAMLA(const FiaTilingInfo &fiaInfo)
{
    // For IFA MLA, input sparse mode %d, sparse 0 without mask is supported only when sequence length is 1,
    // and sparse 3 with mask or sparse 0 without mask is supported only when sequence length > 1.
    enableIFAMLA = (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512);
    if (enableIFAMLA) {
        // 先整体对所有全量化进行一次sparsemode的校验
        if (fiaInfo.inputQType == ge::DT_FLOAT8_E4M3FN || fiaInfo.inputQType == ge::DT_HIFLOAT8 ||
            fiaInfo.inputQType == ge::DT_INT8) {
            if (fiaInfo.attenMaskFlag && fiaInfo.sparseMode != SPARSE_MODE_RIGHT_DOWN) {
                std::string reasonMsg = "In MLA FullQuant Scenario, when the datatype of query is "
                    "FLOAT8_E4M3FN, HIFLOAT8, or INT8, and attenMask is not empty, sparseMode must be 3";
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "sparse_mode",
                    std::to_string(fiaInfo.sparseMode).c_str(), reasonMsg.c_str());
                return ge::GRAPH_FAILED;
            }
            if (!fiaInfo.attenMaskFlag && (fiaInfo.sparseMode != SPARSE_MODE_NO_MASK)) {
                std::string reasonMsg = "In MLA FullQuant Scenario, when the datatype of query is "
                    "FLOAT8_E4M3FN, HIFLOAT8, or INT8, and attenMask is empty, sparseMode must be 0";
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "sparse_mode",
                    std::to_string(fiaInfo.sparseMode).c_str(), reasonMsg.c_str());
                return ge::GRAPH_FAILED;
            }
        }

        // 对于Int8场景下，进行以下拦截限制：
        const std::vector<std::string> layoutSupportList = {
            "TND", "TND_NTD",
        };
        std::string layoutStr(fiaInfo.opParamInfo.layOut);
        std::string layout = layoutStr;
        // "TND", "TND_NTD"场景下，不需要对qs进行区分
            // 非TND/TND_NTD场景下，int8场景需要对qs进行区分
            // qs = 1时，仅支持传入sparsemode=0，且不传mask
        if (fiaInfo.inputQType == ge::DT_INT8 &&
            std::find(layoutSupportList.begin(), layoutSupportList.end(), layout) == layoutSupportList.end()) {
            if (fiaInfo.s1Size == 1U) {
                if (fiaInfo.sparseMode != SPARSE_MODE_NO_MASK) {
                    std::string reasonMsg = "In MLA FullQuant Scenario, sparseMode must be 0, when the datatype of "
                        "query is INT8, inputLayout is not TND or TND_NTD and the S axis of query = 1";
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "sparse_mode",
                        std::to_string(fiaInfo.sparseMode).c_str(), reasonMsg.c_str());
                    return ge::GRAPH_FAILED;
                }

                if (fiaInfo.attenMaskFlag) {
                    const gert::Tensor *maskTensor = fiaInfo.opParamInfo.attenMask.tensor;
                    std::string shapeStr =
                        (maskTensor != nullptr) ? ToStringRaw(maskTensor->GetStorageShape()) : "null";
                    std::string reasonMsg = "In MLA FullQuant Scenario, attenMask must be empty, when the datatype of "
                        "query is INT8, inputLayout is not TND or TND_NTD and the S axis of query = 1";
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "atten_mask",
                        shapeStr.c_str(), reasonMsg.c_str());
                    return ge::GRAPH_FAILED;
                }
            } else {
                // 当qs大于1时，仅支持传入sparsemode=3，且传入mask
                if (fiaInfo.sparseMode != SPARSE_MODE_RIGHT_DOWN) {
                    std::string reasonMsg = "In MLA FullQuant Scenario, sparseMode must be 3, when the datatype of "
                        "query is INT8, inputLayout is not TND or TND_NTD and the S axis of query > 1";
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "sparse_mode",
                        std::to_string(fiaInfo.sparseMode).c_str(), reasonMsg.c_str());
                    return ge::GRAPH_FAILED;
                }

                if (!fiaInfo.attenMaskFlag) {
                    const gert::Tensor *maskTensor = fiaInfo.opParamInfo.attenMask.tensor;
                    std::string shapeStr =
                        (maskTensor != nullptr) ? ToStringRaw(maskTensor->GetStorageShape()) : "null";

                    std::string reasonMsg = "In MLA FullQuant Scenario, attenMask cannot be empty, when "
                        "the datatype of query is INT8, inputLayout is not TND or TND_NTD and the S axis of query > 1";
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "atten_mask",
                        shapeStr.c_str(), reasonMsg.c_str());
                    return ge::GRAPH_FAILED;
                }
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckMXFP8FullQuant(const FiaTilingInfo &fiaInfo)
{
    enableMXFP8 = (*fiaInfo.opParamInfo.queryQuantMode == PER_TOKEN_GROUP_MODE &&
                   *fiaInfo.opParamInfo.keyAntiquantMode == PER_TOKEN_GROUP_MODE &&
                   *fiaInfo.opParamInfo.valueAntiquantMode == PER_CHANNEL_GROUP_MODE);
    if (!enableMXFP8) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(!(((fiaInfo.sparseMode == SPARSE_MODE_NO_MASK) && (!fiaInfo.attenMaskFlag)) ||
                ((fiaInfo.sparseMode == SPARSE_MODE_RIGHT_DOWN) && (fiaInfo.attenMaskFlag))),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "sparse_mode",
                        std::to_string(fiaInfo.sparseMode).c_str(),
                        "In MXFP8 fullquant scenario, only sparse 0 without mask or sparse 3 with mask is supported"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckFP8GQAFullQuant(const FiaTilingInfo &fiaInfo)
{
    enableFP8GQAFullQuant = (*fiaInfo.opParamInfo.queryQuantMode == PER_TOKEN_HEAD_MODE &&
                             *fiaInfo.opParamInfo.keyAntiquantMode == PER_TOKEN_HEAD_MODE &&
                             *fiaInfo.opParamInfo.valueAntiquantMode == PER_TENSOR_HEAD_MODE);
    if (!enableFP8GQAFullQuant) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(!(((fiaInfo.sparseMode == SPARSE_MODE_NO_MASK) && (!fiaInfo.attenMaskFlag)) ||
                ((fiaInfo.sparseMode == SPARSE_MODE_RIGHT_DOWN) && (fiaInfo.attenMaskFlag))),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "sparse_mode",
                        std::to_string(fiaInfo.sparseMode).c_str(),
                        "In FP8 GQA fullquant scenario, only support sparse 0 without mask "
                        "or sparse 3 with mask"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckQKVDDifferent(const FiaTilingInfo &fiaInfo) const
{
    // Only support sparse mode 0/2/3 when query and key headdim is not equal to value headdim.
    OP_CHECK_IF(
        fiaInfo.isQKVDDifferent && fiaInfo.sparseMode != SPARSE_MODE_NO_MASK &&
            fiaInfo.sparseMode != SPARSE_MODE_LEFT_UP && fiaInfo.sparseMode != SPARSE_MODE_RIGHT_DOWN,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "sparse_mode",
            std::to_string(fiaInfo.sparseMode).c_str(),
            "when query and key headdim is not equal to value headdim, only sparse mode 0/2/3 is supported"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckFeatureSparseMode(const FiaTilingInfo &fiaInfo) const
{
    int32_t sparseMode = fiaInfo.sparseMode;
    // sparse9 仅在rope分离场景下存在，不支持左padding、PSE、公共前缀、后量化等特性 拦截s2 >= s1
    if (sparseMode == SPARSE_MODE_TREE) {
        // 特性校验
        OP_CHECK_IF(fiaInfo.ropeMode != RopeMode::ROPE_SPLIT,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "sparse_mode",
                std::to_string(sparseMode).c_str(),
                ("In " + std::string(QuantModeToSerialString(fiaInfo.quantMode))
                    + " scenario, when query_rope/key_rope not exist, sparse_mode(9) is not supported").c_str()),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(fiaInfo.qPaddingSizeFlag || fiaInfo.kvPaddingSizeFlag,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName,
                "query_padding_size and kv_padding_size",
                ("In " + std::string(QuantModeToSerialString(fiaInfo.quantMode))
                    + " scenario, when sparse_mode is " + std::to_string(sparseMode)
                    + ", query_padding_size and kv_padding_size must be empty").c_str()),
            return ge::GRAPH_FAILED);
        // 不支持PSE
        OP_CHECK_IF(fiaInfo.pseShiftFlag,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "pse_shift",
                ("In " + std::string(QuantModeToSerialString(fiaInfo.quantMode))
                    + " scenario, when sparse_mode is " + std::to_string(sparseMode)
                    + ", pse_shift must be empty").c_str()),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(fiaInfo.sysPrefixFlag,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "key_shared_prefix and value_shared_prefix",
                ("In " + std::string(QuantModeToSerialString(fiaInfo.quantMode))
                    + " scenario, when sparse_mode is " + std::to_string(sparseMode)
                    + ", key_shared_prefix and value_shared_prefix must be empty").c_str()),
            return ge::GRAPH_FAILED);

        if (fiaInfo.outputType == ge::DT_INT8) {
            std::string reasonMsg = "In " + QuantModeToSerialString(fiaInfo.quantMode) +" scenario,"
                " when sparse is " + std::to_string(sparseMode) + ", the datatype of attentionOut cannot be int8";
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "attention_out",
                ToString(fiaInfo.outputType).c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }

        // s2 >= s1拦截
        // tiling下沉场景 由于actualSeqlen得不到，所以不进行校验
        if (fiaInfo.isMaxWorkspace) {
            return ge::GRAPH_SUCCESS;
        }
        // s2=0 的 batch（入图padding或空tensor场景）不校验 s1<=s2
        int32_t actualSeqSize = std::min(fiaInfo.qSize.size(), fiaInfo.kvSize.size());
        for (int32_t i = 0; i < actualSeqSize; i++) {
            if (fiaInfo.kvSize[i] == 0) {
                continue;
            }
            OP_CHECK_IF(fiaInfo.qSize[i] > fiaInfo.kvSize[i],
                OP_LOGE(fiaInfo.opName,
                        "In %s situation, when sparse is %d, qSize[%d] should less than or equal to kvSize[%d], "
                        "but got qSize %d and kvSize %d.",
                        QuantModeToSerialString(fiaInfo.quantMode).c_str(), sparseMode, i, i,
                        fiaInfo.qSize[i], fiaInfo.kvSize[i]),
            return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckPretokenAndNexttoken(const FiaTilingInfo &fiaInfo)
{
    // In PFA mode, the values of pretoken and nexttoken must ensure the mask range remains valid.
    isIFAFlag = (enableAntiQuant_) && (fiaInfo.s1Size == 1);
    ge::DataType outputType = fiaInfo.opParamInfo.attenOut.desc->GetDataType();
    if (isIFAFlag) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF((fiaInfo.nextToken * (-1)) > fiaInfo.preToken,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(fiaInfo.opName, "pre_tokens and next_tokens",
            (std::to_string(fiaInfo.preToken) + " and " + std::to_string(fiaInfo.nextToken)).c_str(),
            "The following constraint must be met: next_tokens * (-1) > pre_tokens"),
        return ge::GRAPH_FAILED);
    // Check the specific conditions that pretoken and nexttoken must satisfy under the band mode.
    OP_CHECK_IF(
        (fiaInfo.antiQuantFlag && fiaInfo.sparseMode == SPARSE_MODE_BAND && outputType == ge::DT_INT8 &&
         ((fiaInfo.preToken < 0) || fiaInfo.nextToken < 0)),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(fiaInfo.opName, "pre_tokens and next_tokens",
            (std::to_string(fiaInfo.preToken) + " and " + std::to_string(fiaInfo.nextToken)).c_str(),
            "When the datatype of attention_out is int8 and sparse_mode is 4, "
            "preTokens and nextTokens cannot be negative"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckIFAMask2DShape(const FiaTilingInfo &fiaInfo, const std::string &layoutStr) const
{
    if (fiaInfo.socVersion == platform_ascendc::SocVersion::ASCEND910B &&
        (fiaInfo.sparseMode == SPARSE_MODE_NO_MASK || fiaInfo.sparseMode == SPARSE_MODE_ALL_MASK)) {
        if (fiaInfo.ropeMode == RopeMode::NO_ROPE) {
            const std::vector<std::string> layoutSupportList = {
                "BSH", "BSND", "BNSD", "BNSD_BSND",
            };
            if (std::find(layoutSupportList.begin(), layoutSupportList.end(), layoutStr)
                == layoutSupportList.end()) {
                OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(fiaInfo.opName, "attention_mask", layoutStr.c_str(),
                    "In gqa noquant situation, when rope is not enabled, qkHeadDim = vHeadDim and "
                    "sparseMode is 0 or 1, the layout of two dim mask only supports BSH, BSND, BNSD or BNSD_BSND");
                return ge::GRAPH_FAILED;
            }
        } else {
            OP_LOGE(fiaInfo.opName,
                    "In gqa noquant situation, rope exits or qkHeadDim != vHeadDim, "
                    "when sparseMode = 0 or 1, two dim mask is not supported.");
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(fiaInfo.opName, "atten_mask", "2D",
            "Please use 3D mask \[B,QS,KVS\]\/\[1,QS,KVS\] or 4D mask \[B,1,QS,KVS\]\/\[1,1,QS,KVS\]");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckIFAMask3DShape(const FiaTilingInfo &fiaInfo, const gert::Tensor *maskShape,
    uint32_t minAttenMaskSize) const
{
    // attenMask的shape应为(B, >=Q_S, >=KV_S + systemPrefixLen)或(1, >=Q_S, >=KV_S + systemPrefixLen)
    if (((maskShape->GetStorageShape().GetDim(DIM_NUM_0) != fiaInfo.bSize &&
                maskShape->GetStorageShape().GetDim(DIM_NUM_0) != 1) ||
                maskShape->GetStorageShape().GetDim(DIM_NUM_1) < fiaInfo.s1Size ||
                maskShape->GetStorageShape().GetDim(DIM_NUM_2) < minAttenMaskSize)) {
        std::string shapeStr = ToStringRaw(maskShape->GetStorageShape());
        std::string reasonMsg = "The shape of atten_mask must be [B or 1, >=Q_S, >=(KV_S + systemPrefixLen)]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "atten_mask", shapeStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckIFAMask4DShape(const FiaTilingInfo &fiaInfo, const gert::Tensor *maskShape,
    uint32_t minAttenMaskSize) const
{
    // attenMask的shape应为(B, 1, >=Q_S, >=KV_S + systemPrefixLen)或(1, 1, >=Q_S, >=KV_S + systemPrefixLen)
    if (((maskShape->GetStorageShape().GetDim(DIM_NUM_0) != fiaInfo.bSize &&
                maskShape->GetStorageShape().GetDim(DIM_NUM_0) != 1) ||
                maskShape->GetStorageShape().GetDim(DIM_NUM_1) != 1 ||
                maskShape->GetStorageShape().GetDim(DIM_NUM_2) < fiaInfo.s1Size ||
                maskShape->GetStorageShape().GetDim(DIM_NUM_3) < minAttenMaskSize)) {
        std::string shapeStr = ToStringRaw(maskShape->GetStorageShape());
        std::string reasonMsg = "The shape of atten_mask must be [B or 1, 1, >=Q_S, >=(KV_S + systemPrefixLen)]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "atten_mask", shapeStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckIFADimAndShape(const FiaTilingInfo &fiaInfo) const
{
    uint32_t minAttenMaskSize = 0;
    std::string layoutStr(fiaInfo.opParamInfo.layOut);
    const gert::Tensor *maskShape = fiaInfo.opParamInfo.attenMask.tensor;
    size_t attenMaskDim = maskShape->GetStorageShape().GetDimNum();
    uint32_t attenMaskBatch = maskShape->GetStorageShape().GetDim(DIM_NUM_0);
    uint32_t attenMaskSize = maskShape->GetStorageShape().GetDim(maskShape->GetStorageShape().GetDimNum() - 1);
    if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        minAttenMaskSize = fiaInfo.s2Size + fiaInfo.systemPrefixLen;
    } else {
        minAttenMaskSize = fiaInfo.maxActualseq + fiaInfo.systemPrefixLen;
    }
    if (attenMaskDim == MASK_DIM_SS) {
        return CheckIFAMask2DShape(fiaInfo, layoutStr);
    } else if (attenMaskDim == MASK_DIM_BSS &&
                (fiaInfo.sparseMode == SPARSE_MODE_NO_MASK || fiaInfo.sparseMode == SPARSE_MODE_ALL_MASK)) {
        return CheckIFAMask3DShape(fiaInfo, maskShape, minAttenMaskSize);
    } else if (attenMaskDim == MASK_DIM_B1SS &&
                (fiaInfo.sparseMode == SPARSE_MODE_NO_MASK || fiaInfo.sparseMode == SPARSE_MODE_ALL_MASK)) {
        return CheckIFAMask4DShape(fiaInfo, maskShape, minAttenMaskSize);
    } else {
        std::string dimStr = std::to_string(attenMaskDim) + "D";
        OP_LOGE_FOR_INVALID_SHAPEDIM(fiaInfo.opName, "atten_mask", dimStr.c_str(), "2D, 3D or 4D");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckAndProcessTreeMask(const FiaTilingInfo &fiaInfo, MaskInfo &maskInfo) const
{
    const gert::Tensor *maskShape = fiaInfo.opParamInfo.attenMask.tensor;
    size_t attenMaskDim = maskShape->GetStorageShape().GetDimNum();
    if (attenMaskDim != MASK_DIM_S && attenMaskDim != MASK_DIM_BSS) {
        std::string dimStr = std::to_string(attenMaskDim) + "D";
        std::string reasonMsg = "The shape dim of atten_mask must be 1D or 3D when sparse_mode = " +
                                std::to_string(SPARSE_MODE_TREE);
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(fiaInfo.opName, "atten_mask", dimStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    uint64_t sSize = 0;
    if (fiaInfo.qLayout == FiaLayout::TND || fiaInfo.qLayout == FiaLayout::NTD) {
        // tiling下沉场景 由于actualSeqlen得不到，所以不进行校验
        if (fiaInfo.isMaxWorkspace) {
            return ge::GRAPH_SUCCESS;
        }

        for (uint32_t i = 0; i < fiaInfo.qSize.size(); i++) {
            sSize += fiaInfo.qSize[i] * fiaInfo.qSize[i];
        }
    } else {
        sSize = fiaInfo.s1Size;
    }
    maskInfo.attenMaskBatch = maskShape->GetStorageShape().GetDim(DIM_NUM_0);
    maskInfo.attenMaskQSize = static_cast<int64_t>(sSize);
    maskInfo.attenMaskSize = static_cast<int64_t>(sSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckAndProcess2DMask(const FiaTilingInfo &fiaInfo,
                                                   const gert::Tensor *maskShape, MaskInfo &maskInfo) const
{
    if ((fiaInfo.sparseMode == SPARSE_MODE_NO_MASK || fiaInfo.sparseMode == SPARSE_MODE_ALL_MASK) &&
        fiaInfo.socVersion != platform_ascendc::SocVersion::ASCEND910B) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(fiaInfo.opName, "atten_mask", "2D",
            ("Attenmask does not support 2D when sparse mode is "
                + std::to_string(fiaInfo.sparseMode)
                + ". Please use 3D or 4D mask").c_str());
        return ge::GRAPH_FAILED;
    } else {
        if (fiaInfo.socVersion == platform_ascendc::SocVersion::ASCEND910B &&
            (fiaInfo.sparseMode == SPARSE_MODE_NO_MASK || fiaInfo.sparseMode == SPARSE_MODE_ALL_MASK)) {
            if (fiaInfo.ropeMode == RopeMode::NO_ROPE) {
                const std::vector<std::string> layoutSupportList = {
                    "BSH", "BSND", "BNSD", "BNSD_BSND",
                };
                std::string layoutStr(fiaInfo.opParamInfo.layOut);
                std::string layout = layoutStr;
                if (std::find(layoutSupportList.begin(), layoutSupportList.end(), layout) ==
                    layoutSupportList.end()) {
                    OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(fiaInfo.opName, "attention_mask", layout.c_str(),
                        "In gqa noquant situation, when rope is not used and "
                        "qkHeadDim = vHeadDim and sparseMode is 0 or 1, "
                        "the layout of two dim mask only supports BSH, BSND, BNSD or BNSD_BSND");
                    return ge::GRAPH_FAILED;
                }
            } else {
                OP_LOGE(fiaInfo.opName,
                        "In gqa no quant situation, rope exits or qkHeadDim != vHeadDim, "
                        "when sparseMode = 0 or 1, two dim mask is not supported.");
                return ge::GRAPH_FAILED;
            }
        }
    }
    maskInfo.attenMaskQSize = maskShape->GetStorageShape().GetDim(DIM_NUM_0);
    maskInfo.attenMaskSize = maskShape->GetStorageShape().GetDim(DIM_NUM_1);
    maskInfo.strMaskShape = std::to_string(maskInfo.attenMaskQSize) + ", " +
                            std::to_string(maskInfo.attenMaskSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::GetMaskInfo(const FiaTilingInfo &fiaInfo, MaskInfo &maskInfo) const
{
    const gert::Tensor *maskShape = fiaInfo.opParamInfo.attenMask.tensor;
    size_t attenMaskDim = maskShape->GetStorageShape().GetDimNum();
    if (fiaInfo.sparseMode == SPARSE_MODE_TREE) {
        return CheckAndProcessTreeMask(fiaInfo, maskInfo);
    } else {
        if (attenMaskDim == MASK_DIM_SS) {
            return CheckAndProcess2DMask(fiaInfo, maskShape, maskInfo);
        } else if (attenMaskDim == MASK_DIM_BSS) {
            maskInfo.attenMaskBatch = maskShape->GetStorageShape().GetDim(DIM_NUM_0);
            maskInfo.attenMaskQSize = maskShape->GetStorageShape().GetDim(DIM_NUM_1);
            maskInfo.attenMaskSize =
                maskShape->GetStorageShape().GetDim(DIM_NUM_2); // 2: When the dim is 3, the second dimension is S2.
            maskInfo.strMaskShape = std::to_string(maskInfo.attenMaskBatch) + ", " +
                                    std::to_string(maskInfo.attenMaskQSize) + ", " +
                                    std::to_string(maskInfo.attenMaskSize);
        } else if (attenMaskDim == MASK_DIM_B1SS) {
            maskInfo.attenMaskBatch = maskShape->GetStorageShape().GetDim(DIM_NUM_0);
            maskInfo.attenMaskN = maskShape->GetStorageShape().GetDim(DIM_NUM_1);
            maskInfo.attenMaskQSize =
                maskShape->GetStorageShape().GetDim(DIM_NUM_2); // 2: When the dim is 4, the second dimension is S1.
            maskInfo.attenMaskSize = maskShape->GetStorageShape().GetDim(DIM_NUM_3); // 3:The third dimension is S2.
            maskInfo.strMaskShape = std::to_string(maskInfo.attenMaskBatch) + ", " +
                                    std::to_string(maskInfo.attenMaskN) + ", " +
                                    std::to_string(maskInfo.attenMaskQSize) + ", " +
                                    std::to_string(maskInfo.attenMaskSize);
        } else {
            std::string dimStr = std::to_string(attenMaskDim) + "D";
            OP_LOGE_FOR_INVALID_SHAPEDIM(fiaInfo.opName, "atten_mask", dimStr.c_str(), "2D 3D or 4D");
            return ge::GRAPH_FAILED;
        }
    }
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::ValidateMaskDimAndShape(const FiaTilingInfo &fiaInfo, const MaskInfo &maskInfo) const
{
    bool checkMask = false;
    uint32_t minAttenMaskSize = 0;
    if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        minAttenMaskSize = fiaInfo.s2Size + fiaInfo.systemPrefixLen;
    } else {
        minAttenMaskSize = fiaInfo.maxActualseq + fiaInfo.systemPrefixLen;
    }
    if (fiaInfo.sparseMode == SPARSE_MODE_NO_MASK || fiaInfo.sparseMode == SPARSE_MODE_ALL_MASK) {
        checkMask = (maskInfo.attenMaskQSize >= fiaInfo.s1Size) &&
                    (maskInfo.attenMaskSize >= minAttenMaskSize) &&
                    (maskInfo.attenMaskBatch == NUM1 || maskInfo.attenMaskBatch == fiaInfo.bSize) &&
                    (static_cast<uint32_t>(maskInfo.attenMaskN) == NUM1);
    } else if ((fiaInfo.sparseMode == SPARSE_MODE_LEFT_UP) || (fiaInfo.sparseMode == SPARSE_MODE_RIGHT_DOWN) ||
               (fiaInfo.sparseMode == SPARSE_MODE_BAND)) {
        checkMask = (maskInfo.attenMaskBatch == NUM1) && (static_cast<uint32_t>(maskInfo.attenMaskN) == NUM1) &&
                    (maskInfo.attenMaskQSize == SPARSE_OPTIMIZE_ATTENTION_SIZE) &&
                    (maskInfo.attenMaskSize == SPARSE_OPTIMIZE_ATTENTION_SIZE);
    }

    if (fiaInfo.sparseMode == SPARSE_MODE_NO_MASK || fiaInfo.sparseMode == SPARSE_MODE_ALL_MASK) {
        if (!checkMask) {
            const gert::Tensor *maskShape = fiaInfo.opParamInfo.attenMask.tensor;
            std::string shapeStr = ToStringRaw(maskShape->GetStorageShape());
            std::string reasonMsg = "The shape of atten_mask must be [B(" + std::to_string(fiaInfo.bSize) +
                                    ") or 1, >=Q_S(" + std::to_string(fiaInfo.s1Size) + "), >=KV_S(" +
                                    std::to_string(fiaInfo.s2Size) + ") + systemPrefixLen(" +
                                    std::to_string(fiaInfo.systemPrefixLen) + ")]";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "atten_mask", shapeStr.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    if (((fiaInfo.sparseMode == SPARSE_MODE_LEFT_UP) || (fiaInfo.sparseMode == SPARSE_MODE_RIGHT_DOWN) ||
         (fiaInfo.sparseMode == SPARSE_MODE_BAND)) &&
        !checkMask) {
        const gert::Tensor *maskShape = fiaInfo.opParamInfo.attenMask.tensor;
        std::string shapeStr = ToStringRaw(maskShape->GetStorageShape());
        std::string reasonMsg = "The shape of atten_mask must be [2048, 2048], [1, 2048, 2048] or [1, 1, 2048, "
                                "2048] when the sparse_mode is " +
                                std::to_string(fiaInfo.sparseMode);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "atten_mask", shapeStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckDimAndShape(const FiaTilingInfo &fiaInfo)
{
    // In PFA mode, the attenmask dimensions must be 2/3/4.
    // The allowed shape specifications for attenmask vary depending on the sparse mode.
    if ((!fiaInfo.attenMaskFlag) && (fiaInfo.sparseMode != SPARSE_MODE_NO_MASK)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "atten_mask", "empty",
            ("when sparse_mode is " + std::to_string(fiaInfo.sparseMode)
                + " (not 0), atten_mask cannot be empty").c_str());
        return ge::GRAPH_FAILED;
    }
    if ((fiaInfo.isMaxWorkspace && fiaInfo.socVersion != platform_ascendc::SocVersion::ASCEND910B) ||
        !fiaInfo.attenMaskFlag) {
        return ge::GRAPH_SUCCESS;
    }
    if (fiaInfo.socVersion == platform_ascendc::SocVersion::ASCEND910B &&
        (fiaInfo.sparseMode == SPARSE_MODE_NO_MASK ||
        fiaInfo.sparseMode == SPARSE_MODE_ALL_MASK)) {
        if (fiaInfo.isMaxWorkspace) {
            return ge::GRAPH_SUCCESS;
        }
    }
    isIFAFlag = (fiaInfo.antiQuantFlag) && (fiaInfo.s1Size == 1);
    if (isIFAFlag) {
        return CheckIFADimAndShape(fiaInfo);
    }

    MaskInfo maskInfo;
    if (GetMaskInfo(fiaInfo, maskInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ValidateMaskDimAndShape(fiaInfo, maskInfo);
}

ge::graphStatus MaskChecker::CheckSinglePara(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckDtypeAndFormat(fiaInfo) || ge::GRAPH_SUCCESS != CheckSparseMode(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckParaExistence(const FiaTilingInfo &fiaInfo)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckCrossFeature(const FiaTilingInfo &fiaInfo)
{
    if (enableNonQuant_) {
        if (ge::GRAPH_SUCCESS != CheckNoQuantIFAMLA(fiaInfo) || ge::GRAPH_SUCCESS != CheckQKVDDifferent(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
        if (fiaInfo.socVersion == platform_ascendc::SocVersion::ASCEND910B) {
            if (ge::GRAPH_SUCCESS != CheckFeatureSparseMode(fiaInfo)) {
                return ge::GRAPH_FAILED;
            }
        }
    } else if (enableAntiQuant_) {
        if (ge::GRAPH_SUCCESS != CheckAntiquantSparseMode(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    } else if (enableFullQuant_) {
        if (ge::GRAPH_SUCCESS != CheckFullQuantIFAMLA(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
        if (ge::GRAPH_SUCCESS != CheckMXFP8FullQuant(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
        if (ge::GRAPH_SUCCESS != CheckFP8GQAFullQuant(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    } else if (enableAntiQuant_) {
        if (ge::GRAPH_SUCCESS != CheckAntiquantSparseMode(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckPretokenAndNexttoken(fiaInfo) || ge::GRAPH_SUCCESS != CheckDimAndShape(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling
