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
 * \file fused_infer_attention_score_v2_sink_tiling_check_feature.cpp
 * \brief
 */

#include <vector>
#include <string>
#include <utility>
#include <sstream>
#include <numeric>
#include <algorithm>
#include "tiling/tiling_api.h"
#include "fused_infer_attention_score_v2_sink_tiling_check.h"

using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
namespace optiling {
ge::graphStatus FiaTilingCheck::CheckFeatureNoQuantDtype() const
{
    if (quantMode_ != FiaQuantMode::NO_QUANT) {
        return ge::GRAPH_SUCCESS;
    }
    OPS_CHECK(inputQType_ != ge::DT_BF16 && inputQType_ != ge::DT_FLOAT16,
              OPS_LOG_E(opName_,
                        "In %s situation, query dtype only support %s and %s, but got %s",
                        QuantModeToSerialString(quantMode_).c_str(),
                        FusedDataTypeToSerialString(ge::DT_BF16).c_str(),
                        FusedDataTypeToSerialString(ge::DT_FLOAT16).c_str(),
                        FusedDataTypeToSerialString(inputQType_).c_str()),
              return ge::GRAPH_FAILED);

    OPS_CHECK(inputQType_ != inputKvType_,
              OPS_LOG_E(opName_,
                        "In %s situation, key and value dtype(%s) must equal to query dtype(%s)",
                        QuantModeToSerialString(quantMode_).c_str(),
                        FusedDataTypeToSerialString(inputQType_).c_str(),
                        FusedDataTypeToSerialString(inputKvType_).c_str()),
              return ge::GRAPH_FAILED);

    if (ropeMode_ == RopeMode::ROPE_SPLIT) {
        OPS_CHECK((opParamInfo_.queryRope.desc->GetDataType() != opParamInfo_.query.desc->GetDataType()),
                  OPS_LOG_E(opName_,
                            "%s(%s) and %s(%s) must have same dType",
                            QUERY_ROPE_NAME.c_str(),
                            FusedDataTypeToSerialString(opParamInfo_.queryRope.desc->GetDataType()).c_str(),
                            QUERY_NAME.c_str(),
                            FusedDataTypeToSerialString(opParamInfo_.query.desc->GetDataType()).c_str()),
                  return ge::GRAPH_FAILED);

        OPS_CHECK((opParamInfo_.keyRope.desc->GetDataType() != opParamInfo_.key.desc->GetDataType()),
                  OPS_LOG_E(opName_,
                            "%s(%s) and %s(%s) must have same dType",
                            KEY_ROPE_NAME.c_str(),
                            FusedDataTypeToSerialString(opParamInfo_.keyRope.desc->GetDataType()).c_str(),
                            KEY_NAME.c_str(),
                            FusedDataTypeToSerialString(opParamInfo_.key.desc->GetDataType()).c_str()),
                  return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureNoquantBlockSize() const
{
    constexpr int32_t BLOCK_SIZE_ALIGN_SIZE = 16;
    constexpr int32_t BLOCK_SIZE_MAX_SIZE = 1024;
    if (blockSize_ % BLOCK_SIZE_ALIGN_SIZE != 0) {
        OPS_LOG_E(opName_,
                  "In %s situation, %s should aligned to 16, but got %d.",
                  QuantModeToSerialString(quantMode_).c_str(),
                  BLOCK_SIZE_NAME.c_str(),
                  blockSize_);
        return ge::GRAPH_FAILED;
    }

    if (blockSize_ > BLOCK_SIZE_MAX_SIZE) {
        OPS_LOG_E(opName_,
                  "In %s situation, %s should less equal than 1024, but got %d.",
                  QuantModeToSerialString(quantMode_).c_str(),
                  BLOCK_SIZE_NAME.c_str(),
                  blockSize_);
        return ge::GRAPH_FAILED;
    }

    if (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION && blockSize_ == 0) {
        OPS_LOG_E(opName_,
                  "In %s and storage mode is page attention, %s should not be 0",
                  QuantModeToSerialString(quantMode_).c_str(),
                  BLOCK_SIZE_NAME.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureNoquantUnsupported() const
{
    OPS_CHECK(
        fiaInfo_.outputType == ge::DT_INT8,
        OPS_LOG_E(opName_, "In %s situation, postquant is not supported.", QuantModeToSerialString(quantMode_).c_str()),
        return ge::GRAPH_FAILED);

    OPS_CHECK(
        fiaInfo_.pseShiftFlag,
        OPS_LOG_E(opName_, "In %s situation, pseshift is not supported.", QuantModeToSerialString(quantMode_).c_str()),
        return ge::GRAPH_FAILED);

    OPS_CHECK(fiaInfo_.qPaddingSizeFlag || fiaInfo_.kvPaddingSizeFlag,
              OPS_LOG_E(opName_,
                        "In %s situation, left padding is not supported.",
                        QuantModeToSerialString(quantMode_).c_str()),
              return ge::GRAPH_FAILED);

    OPS_CHECK(fiaInfo_.sysPrefixFlag,
              OPS_LOG_E(opName_,
                        "In %s situation, sys prifix is not supported.",
                        QuantModeToSerialString(quantMode_).c_str()),
              return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureMlaNoquantUnsupported() const
{
    if (vHeadDim_ == 512U) {
        if (CheckFeatureNoquantUnsupported() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (kvStorageMode_ == KvStorageMode::TENSOR_LIST) {
            OPS_LOG_E(opName_,
                      "In %s situation, rope exists and query/key head dim = %u, the key/value's storage mode not "
                      "support tensor list",
                      QuantModeToSerialString(quantMode_).c_str(),
                      qkHeadDim_);
            return ge::GRAPH_FAILED;
        }
    } else {
        return CheckFeatureGqaNoquantUnsupported();
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureMlaNoquant()
{
    OPS_CHECK(socVersion_ == platform_ascendc::SocVersion::ASCEND310P,
              OPS_LOG_E(opName_,
                        "In %s %s situation, Ascend310P is not supported",
                        RopeModeToSerialString(ropeMode_).c_str(),
                        QuantModeToSerialString(quantMode_).c_str()),
              return ge::GRAPH_FAILED);
    if (ge::GRAPH_SUCCESS != CheckFeatureMlaNoquantUnsupported() ||
        ge::GRAPH_SUCCESS != CheckFeatureNoquantBlockSize() || ge::GRAPH_SUCCESS != CheckFeatureInOutDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureActualSeqLens() || ge::GRAPH_SUCCESS != CheckFeatureMask() ||
        ge::GRAPH_SUCCESS != CheckFeatureNoQuantDtype() || ge::GRAPH_SUCCESS != CheckFeatureLayout() ||
        ge::GRAPH_SUCCESS != CheckFeatureAxisInfo() || ge::GRAPH_SUCCESS != CheckFeatureHeadDim()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureMla()
{
    if (quantMode_ == FiaQuantMode::NO_QUANT) {
        return CheckFeatureMlaNoquant();
    } else {
        OPS_LOG_E(opName_, "fiaSink Only Support NoQuant, but got %s", QuantModeToSerialString(quantMode_).c_str());
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureMask() const
{
    if ((!attenMaskFlag_) && (fiaInfo_.sparseMode != SPARSE_MODE_NO_MASK)) {
        OPS_LOG_E(opName_,
                  "when %s is %d, it not 0, %s should not be null.",
                  SPARSE_MODE_NAME.c_str(),
                  fiaInfo_.sparseMode,
                  ATTEN_MASK_NAME.c_str());
        return ge::GRAPH_FAILED;
    }

    if (attenMaskFlag_) {
        size_t maskDimNum = opParamInfo_.attenMask.tensor->GetStorageShape().GetDimNum();
        if ((fiaInfo_.sparseMode == SPARSE_MODE_NO_MASK || fiaInfo_.sparseMode == SPARSE_MODE_ALL_MASK) &&
            maskDimNum == DIM_NUM_TWO) {
            OPS_LOG_E(opName_,
                      "In %s situation, rope exits or qkHeadDim != vHeadDim, when sparseMode = 0 or 1, two dim mask is "
                      "not supported.",
                      QuantModeToSerialString(quantMode_).c_str());
            return ge::GRAPH_FAILED;
        }
    }

    if (ropeMode_ == RopeMode::ROPE_SPLIT && vHeadDim_ == 512U) {
        int32_t sparseMode = fiaInfo_.sparseMode;
        if (sparseMode != SPARSE_MODE_NO_MASK && sparseMode != SPARSE_MODE_RIGHT_DOWN &&
            sparseMode != SPARSE_MODE_BAND) {
            OPS_LOG_E(opName_,
                      "In %s situation, when query_rope and key_rope exists and the head dim of value is %u, %s only "
                      "support 0/3/4, but got %d.",
                      QuantModeToSerialString(quantMode_).c_str(),
                      vHeadDim_,
                      SPARSE_MODE_NAME.c_str(),
                      sparseMode);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureLayout() const
{
    const std::vector<std::string> layoutSupportList = {"BSH",
                                                        "BSND",
                                                        "BNSD",
                                                        "TND",
                                                        "NTD",
                                                        "BSH_NBSD",
                                                        "BSND_NBSD",
                                                        "BNSD_NBSD",
                                                        "TND_NTD",
                                                        "NTD_TND",
                                                        "BSH_BNSD",
                                                        "BSND_BNSD",
                                                        "BNSD_BSND"};
    std::string layout = opParamInfo_.layOut;
    OPS_CHECK(std::find(layoutSupportList.begin(), layoutSupportList.end(), layout) == layoutSupportList.end(),
              OPS_LOG_E(opName_,
                        "In %s %s situation, layout only supports BSH, BSND, BNSD, TND, NTD, BSH_NBSD, BSND_NBSD, "
                        "BNSD_NBSD, TND_NTD, NTD_TND, BSH_BNSD, BSND_BNSD, BNSD_BSND, but got %s",
                        QuantModeToSerialString(quantMode_).c_str(),
                        SituationToSerialString(ropeMode_).c_str(),
                        layout.c_str()),
              return ge::GRAPH_FAILED);

    if (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS) {
        OPS_CHECK(kvLayout_ != FiaLayout::BSH && kvLayout_ != FiaLayout::BSND && kvLayout_ != FiaLayout::BNSD &&
                      kvLayout_ != FiaLayout::TND && kvLayout_ != FiaLayout::NTD,
                  OPS_LOG_E(opName_,
                            "In %s %s situation, key/value's layout only support BSH, BSND, BNSD, TND and NTD in batch "
                            "continuous scene, but got %s",
                            QuantModeToSerialString(quantMode_).c_str(),
                            SituationToSerialString(ropeMode_).c_str(),
                            LayoutToSerialString(kvLayout_).c_str()),
                  return ge::GRAPH_FAILED);

        OPS_CHECK(
            kvLayout_ != qLayout_,
            OPS_LOG_E(
                opName_,
                "In %s %s situation, key/value's layout and query's layout should be same in batch continuous scene.",
                QuantModeToSerialString(quantMode_).c_str(),
                SituationToSerialString(ropeMode_).c_str()),
            return ge::GRAPH_FAILED);
    } else if (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION) {
        OPS_CHECK(kvLayout_ == FiaLayout::BnBsH &&
                      (qLayout_ != FiaLayout::BSH && qLayout_ != FiaLayout::BSND && qLayout_ != FiaLayout::BNSD &&
                       qLayout_ != FiaLayout::TND && qLayout_ != FiaLayout::NTD),
                  OPS_LOG_E(opName_,
                            "In %s %s situation, the key/value's layout is BnBsH, %s layout must be BSH, BSND, BNSD "
                            "TND and TND in page attention scene, but got %s",
                            QuantModeToSerialString(quantMode_).c_str(),
                            SituationToSerialString(ropeMode_).c_str(),
                            QUERY_NAME.c_str(),
                            LayoutToSerialString(qLayout_).c_str()),
                  return ge::GRAPH_FAILED);

        OPS_CHECK(kvLayout_ == FiaLayout::BnNBsD &&
                      (qLayout_ != FiaLayout::BSH && qLayout_ != FiaLayout::BSND && qLayout_ != FiaLayout::BNSD &&
                       qLayout_ != FiaLayout::TND && qLayout_ != FiaLayout::NTD),
                  OPS_LOG_E(opName_,
                            "In %s %s situation, the key/value's layout is BnNBsD, %s layout must be BSH, BSND, BNSD "
                            "TND and TND in page attention scene, but got %s",
                            QuantModeToSerialString(quantMode_).c_str(),
                            SituationToSerialString(ropeMode_).c_str(),
                            QUERY_NAME.c_str(),
                            LayoutToSerialString(qLayout_).c_str()),
                  return ge::GRAPH_FAILED);

        OPS_CHECK(kvLayout_ == FiaLayout::NZ &&
                      (qLayout_ != FiaLayout::BSH && qLayout_ != FiaLayout::BSND && qLayout_ != FiaLayout::BNSD &&
                       qLayout_ != FiaLayout::TND && qLayout_ != FiaLayout::NTD),
                  OPS_LOG_E(opName_,
                            "In %s %s situation, the key/value's layout is BnNBsD, %s layout must be BSH, BSND, BNSD "
                            "TND and TND in page attention scene, but got %s",
                            QuantModeToSerialString(quantMode_).c_str(),
                            SituationToSerialString(ropeMode_).c_str(),
                            QUERY_NAME.c_str(),
                            LayoutToSerialString(qLayout_).c_str()),
                  return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureAxisInfo() const
{
    constexpr uint32_t MAX_ACTUAL_SEQ_LEN_BYTE = 64U * 1024U;

    OPS_CHECK(actualSeqLengthsQSize_ > MAX_ACTUAL_SEQ_LEN_BYTE,
              OPS_LOG_E(opName_,
                        "In %s situation, actual sequence length q should be smaller or equal to 64K, but got %u",
                        QuantModeToSerialString(quantMode_).c_str(),
                        actualSeqLengthsQSize_),
              return ge::GRAPH_FAILED);

    OPS_CHECK(actualSeqLengthsKvSize_ > MAX_ACTUAL_SEQ_LEN_BYTE,
              OPS_LOG_E(opName_,
                        "In %s situation, actual sequence length kv should be smaller or equal to 64K, but got %u",
                        QuantModeToSerialString(quantMode_).c_str(),
                        actualSeqLengthsKvSize_),
              return ge::GRAPH_FAILED);

    if (kvStorageMode_ == KvStorageMode::TENSOR_LIST) {
        constexpr uint32_t MAX_B_SIZE = 256U;

        OPS_CHECK(bSize_ > MAX_B_SIZE,
                  OPS_LOG_E(opName_,
                            "In %s situation, batch size(%u) cannot be greater than %u in tensor list scene.",
                            QuantModeToSerialString(quantMode_).c_str(),
                            bSize_,
                            MAX_B_SIZE),
                  return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureGqaNoquantUnsupported() const
{
    if (CheckFeatureNoquantUnsupported() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (kvStorageMode_ == KvStorageMode::TENSOR_LIST) {
        const std::vector<std::string> layoutSupportList = {
            "BSH",
            "BSND",
            "BNSD",
            "BSND_BNSD",
        };
        std::string layout = opParamInfo_.layOut;
        OPS_CHECK((std::find(layoutSupportList.begin(), layoutSupportList.end(), layout) == layoutSupportList.end()) &&
                      ropeMode_ != RopeMode::NO_ROPE,
                  OPS_LOG_E(opName_,
                            "In %s situation, tensor list is not supported.",
                            QuantModeToSerialString(quantMode_).c_str()),
                  return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureGqaNoquantMask() const
{
    if (fiaInfo_.sparseMode == 0) {
        return ge::GRAPH_SUCCESS;
    }

    if (!attenMaskFlag_) {
        OPS_LOG_E(opName_,
                  "In %s situation, when %s = 1/2/3/4, %s should not be null.",
                  QuantModeToSerialString(quantMode_).c_str(),
                  SPARSE_MODE_NAME.c_str(),
                  ATTEN_MASK_NAME.c_str());
        return ge::GRAPH_FAILED;
    }

    size_t maskDimNum = opParamInfo_.attenMask.tensor->GetStorageShape().GetDimNum();
    int64_t maskDim0 = opParamInfo_.attenMask.tensor->GetStorageShape().GetDim(0);
    int32_t sparseMode = *opParamInfo_.sparseMode;

    if (sparseMode == SPARSE_MODE_NO_MASK && maskDimNum == DIM_NUM_TWO && s1Size_ == 1U &&
        maskDim0 == static_cast<int64_t>(bSize_)) {
        OPS_CHECK(qLayout_ == FiaLayout::TND || qLayout_ == FiaLayout::NTD,
                  OPS_LOG_E(opName_,
                            "In %s situation, when %s layout is TND/NTD, %s layout BS2 is not supported.",
                            QuantModeToSerialString(quantMode_).c_str(),
                            QUERY_NAME.c_str(),
                            ATTEN_MASK_NAME.c_str()),
                  return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureGqaNoQuantDtype() const
{
    return CheckFeatureNoQuantDtype();
}

ge::graphStatus FiaTilingCheck::CheckFeatureGqaNoQuantLayout() const
{
    const std::vector<std::string> layoutSupportList = {
        "BSH",
        "BSND",
        "BNSD",
        "TND",
        "NTD",
        "BSH_BNSD",
        "BSND_BNSD",
        "BNSD_BSND",
        "NTD_TND",
    };
    std::string layout = opParamInfo_.layOut;
    OPS_CHECK(std::find(layoutSupportList.begin(), layoutSupportList.end(), layout) == layoutSupportList.end(),
              OPS_LOG_E(opName_,
                        "In %s situation, layout only supports BSH, BSND, BNSD, TND, NTD, BSH_BNSD, BSND_BNSD, "
                        "BNSD_BSND and NTD_TND, but got %s",
                        QuantModeToSerialString(quantMode_).c_str(),
                        layout.c_str()),
              return ge::GRAPH_FAILED);

    if (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS) {
        OPS_CHECK(kvLayout_ != FiaLayout::BSH && kvLayout_ != FiaLayout::BSND && kvLayout_ != FiaLayout::BNSD &&
                      kvLayout_ != FiaLayout::TND && kvLayout_ != FiaLayout::NTD,
                  OPS_LOG_E(opName_,
                            "In %s situation, key/value's layout only support BSH, BSND, BNSD, TND and NTD in batch "
                            "continuous scene, but got %s",
                            QuantModeToSerialString(quantMode_).c_str(),
                            LayoutToSerialString(kvLayout_).c_str()),
                  return ge::GRAPH_FAILED);

        OPS_CHECK(
            kvLayout_ != qLayout_,
            OPS_LOG_E(
                opName_,
                "In %s situation, key/value's layout and query's layout should be same in batch continuous scene.",
                QuantModeToSerialString(quantMode_).c_str()),
            return ge::GRAPH_FAILED);
    } else if (kvStorageMode_ == KvStorageMode::TENSOR_LIST) {
        OPS_CHECK(
            kvLayout_ != FiaLayout::BSH && kvLayout_ != FiaLayout::BSND && kvLayout_ != FiaLayout::BNSD,
            OPS_LOG_E(
                opName_,
                "In %s situation, key/value's layout only support BSH, BSND and BNSD in tensor list scene, but got %s",
                QuantModeToSerialString(quantMode_).c_str(),
                LayoutToSerialString(kvLayout_).c_str()),
            return ge::GRAPH_FAILED);

        OPS_CHECK(
            kvLayout_ != qLayout_,
            OPS_LOG_E(opName_,
                      "In %s situation, key/value's layout and query's layout should be same in tensor list scene.",
                      QuantModeToSerialString(quantMode_).c_str()),
            return ge::GRAPH_FAILED);
    } else if (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION) {
        OPS_CHECK(kvLayout_ == FiaLayout::BnBsH &&
                      (qLayout_ != FiaLayout::BSH && qLayout_ != FiaLayout::BSND && qLayout_ != FiaLayout::BNSD &&
                       qLayout_ != FiaLayout::TND && qLayout_ != FiaLayout::NTD),
                  OPS_LOG_E(opName_,
                            "In %s situation, the key/value's layout is BnBsH, %s layout must be BSH, BSND, BNSD TND "
                            "and TND in page attention scene, but got %s",
                            QuantModeToSerialString(quantMode_).c_str(),
                            QUERY_NAME.c_str(),
                            LayoutToSerialString(qLayout_).c_str()),
                  return ge::GRAPH_FAILED);

        OPS_CHECK(kvLayout_ == FiaLayout::BnNBsD &&
                      (qLayout_ != FiaLayout::BSH && qLayout_ != FiaLayout::BSND && qLayout_ != FiaLayout::BNSD &&
                       qLayout_ != FiaLayout::TND && qLayout_ != FiaLayout::NTD),
                  OPS_LOG_E(opName_,
                            "In %s situation, the key/value's layout is BnNBsD, %s layout must be BSH, BSND, BNSD TND "
                            "and TND in page attention scene, but got %s",
                            QuantModeToSerialString(quantMode_).c_str(),
                            QUERY_NAME.c_str(),
                            LayoutToSerialString(qLayout_).c_str()),
                  return ge::GRAPH_FAILED);

        OPS_CHECK(kvLayout_ == FiaLayout::NZ &&
                      (qLayout_ != FiaLayout::BSH && qLayout_ != FiaLayout::BSND && qLayout_ != FiaLayout::BNSD &&
                       qLayout_ != FiaLayout::TND && qLayout_ != FiaLayout::NTD),
                  OPS_LOG_E(opName_,
                            "In %s situation, the key/value's layout is BnNBsD, %s layout must be BSH, BSND, BNSD TND "
                            "and TND in page attention scene, but got %s",
                            QuantModeToSerialString(quantMode_).c_str(),
                            QUERY_NAME.c_str(),
                            LayoutToSerialString(qLayout_).c_str()),
                  return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureGqaNoQuantShape() const
{
    constexpr uint32_t MAX_ACTUAL_SEQ_LEN_BYTE = 64U * 1024U;
    OPS_CHECK(actualSeqLengthsQSize_ > MAX_ACTUAL_SEQ_LEN_BYTE,
              OPS_LOG_E(opName_,
                        "In %s situation, actual sequence length q should be smaller or equal to 64K, but got %u",
                        QuantModeToSerialString(quantMode_).c_str(),
                        actualSeqLengthsQSize_),
              return ge::GRAPH_FAILED);

    OPS_CHECK(actualSeqLengthsKvSize_ > MAX_ACTUAL_SEQ_LEN_BYTE,
              OPS_LOG_E(opName_,
                        "In %s situation, actual sequence length kv should be smaller or equal to 64K, but got %u",
                        QuantModeToSerialString(quantMode_).c_str(),
                        actualSeqLengthsKvSize_),
              return ge::GRAPH_FAILED);

    if (kvStorageMode_ == KvStorageMode::TENSOR_LIST) {
        constexpr uint32_t MAX_B_SIZE = 256U;
        OPS_CHECK(bSize_ > MAX_B_SIZE,
                  OPS_LOG_E(opName_,
                            "In %s situation, batch size(%u) cannot be greater than %u in tensor list scene.",
                            QuantModeToSerialString(quantMode_).c_str(),
                            bSize_,
                            MAX_B_SIZE),
                  return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureHeadDim() const
{
    constexpr uint32_t MAX_HEAD_DIM = 512;
    constexpr uint32_t MAX_ROPE_DIM = 64;

    OPS_CHECK(vHeadDim_ > MAX_HEAD_DIM,
              OPS_LOG_E(opName_,
                        "In %s situation, headDim of value should be smaller or equal to 512, but got %u",
                        QuantModeToSerialString(quantMode_).c_str(),
                        vHeadDim_),
              return ge::GRAPH_FAILED);

    OPS_CHECK(ropeHeadDim_ > MAX_ROPE_DIM,
              OPS_LOG_E(opName_,
                        "In %s situation, headDim of Rope should be smaller or equal to 64, but got %u",
                        QuantModeToSerialString(quantMode_).c_str(),
                        ropeHeadDim_),
              return ge::GRAPH_FAILED);

    if (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION && kvLayout_ == FiaLayout::NZ) {
        constexpr int32_t D_ALIGN_SIZE = 16;
        OPS_CHECK((vHeadDim_ % D_ALIGN_SIZE != 0) || (qkHeadDim_ % D_ALIGN_SIZE != 0),
                  OPS_LOG_E(opName_,
                            "In %s situation, when the dim of key&value is 5, headDim of query|key|value should be "
                            "align to 16, but got keyHeadDim:%u, queryHeadDim and keyHeadDim:%u",
                            QuantModeToSerialString(quantMode_).c_str(),
                            vHeadDim_,
                            qkHeadDim_),
                  return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureGqaNoquant()
{
    OPS_CHECK(socVersion_ == platform_ascendc::SocVersion::ASCEND310P,
              OPS_LOG_E(opName_,
                        "In %s %s situation, Ascend310P is not supported",
                        RopeModeToSerialString(ropeMode_).c_str(),
                        QuantModeToSerialString(quantMode_).c_str()),
              return ge::GRAPH_FAILED);
    if (ge::GRAPH_SUCCESS != CheckFeatureGqaNoquantUnsupported() ||
        ge::GRAPH_SUCCESS != CheckFeatureNoquantBlockSize() || ge::GRAPH_SUCCESS != CheckFeatureInOutDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureActualSeqLens() || ge::GRAPH_SUCCESS != CheckFeatureGqaNoquantMask() ||
        ge::GRAPH_SUCCESS != CheckFeatureGqaNoQuantDtype() || ge::GRAPH_SUCCESS != CheckFeatureGqaNoQuantLayout() ||
        ge::GRAPH_SUCCESS != CheckFeatureGqaNoQuantShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureGqa()
{
    if (quantMode_ == FiaQuantMode::NO_QUANT) {
        return CheckFeatureGqaNoquant();
    } else {
        OPS_LOG_E(opName_, "fiaSink Only Support NoQuant, but got %s", QuantModeToSerialString(quantMode_).c_str());
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureActualSeqLensExistence() const
{
    if ((qLayout_ == FiaLayout::TND || qLayout_ == FiaLayout::NTD)) {
        OPS_CHECK(opParamInfo_.actualSeqLengthsQ.tensor == nullptr,
                  OPS_LOG_E(opName_,
                            "when %s's layout is %s, %s should not be null.",
                            QUERY_NAME.c_str(),
                            LayoutToSerialString(qLayout_).c_str(),
                            ACTUAL_SEQ_Q_LEN_NAME.c_str()),
                  return ge::GRAPH_FAILED);
        OPS_CHECK(opParamInfo_.actualSeqLengths.tensor == nullptr,
                  OPS_LOG_E(opName_,
                            "when %s's layout is %s, %s should not be null.",
                            QUERY_NAME.c_str(),
                            LayoutToSerialString(qLayout_).c_str(),
                            ACTUAL_SEQ_KV_LEN_NAME.c_str()),
                  return ge::GRAPH_FAILED);
    } else {
        if (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION) {
            OPS_CHECK(
                opParamInfo_.actualSeqLengths.tensor == nullptr,
                OPS_LOG_E(opName_, "In page attention scene, %s should not be null.", ACTUAL_SEQ_KV_LEN_NAME.c_str()),
                return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::GetActualSeqLenSize(uint32_t &size,
                                                    const gert::Tensor *tensor,
                                                    const FiaLayout &layout,
                                                    const std::string &actualSeqLenName,
                                                    const std::string &attrName)
{
    if (tensor == nullptr) {
        OPS_LOG_E(opName_,
                  "when layout of %s is %s, %s must be provided.",
                  attrName.c_str(),
                  LayoutToSerialString(layout).c_str(),
                  actualSeqLenName.c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t shapeSize = tensor->GetShapeSize();
    if (shapeSize <= 0) {
        OPS_LOG_E(opName_, "%s shape size is %ld, it should be greater than 0.", actualSeqLenName.c_str(), shapeSize);
        return ge::GRAPH_FAILED;
    }
    size = static_cast<uint32_t>(shapeSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureActualSeqLensQData()
{
    if (opParamInfo_.actualSeqLengthsQ.tensor == nullptr) {
        qSize.push_back(s1Size_);
        return ge::GRAPH_SUCCESS;
    }

    if (GetActualSeqLenSize(actualSeqLengthsQSize_,
                            opParamInfo_.actualSeqLengthsQ.tensor,
                            qLayout_,
                            ACTUAL_SEQ_Q_LEN_NAME,
                            QUERY_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureActualSeqLensKvData()
{
    if (opParamInfo_.actualSeqLengths.tensor == nullptr) {
        kvSize.push_back(s2Size_);
        return ge::GRAPH_SUCCESS;
    }

    if (GetActualSeqLenSize(actualSeqLengthsKvSize_,
                            opParamInfo_.actualSeqLengths.tensor,
                            kvLayout_,
                            ACTUAL_SEQ_KV_LEN_NAME,
                            KEY_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureInOutDtype() const
{
    const std::vector<std::pair<ge::DataType, ge::DataType>> inOutDtypePairSupported = {
        {ge::DT_INT8, ge::DT_INT8},
        {ge::DT_INT8, ge::DT_FLOAT16},
        {ge::DT_FLOAT16, ge::DT_INT8},
        {ge::DT_FLOAT16, ge::DT_FLOAT16},
        {ge::DT_BF16, ge::DT_BF16},
        {ge::DT_BF16, ge::DT_INT8},
        {ge::DT_INT8, ge::DT_INT8},
    };

    std::pair<ge::DataType, ge::DataType> inOutDtypePair = {inputQType_, outputType_};
    if (!VecContains(inOutDtypePairSupported, inOutDtypePair)) {
        OPS_LOG_E(opName_,
                  "input dtype %d with output dtype %d is not currently supported.",
                  static_cast<int32_t>(inputQType_),
                  static_cast<int32_t>(outputType_));
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureActualSeqLens()
{
    if (ge::GRAPH_SUCCESS != CheckFeatureActualSeqLensExistence() ||
        ge::GRAPH_SUCCESS != CheckFeatureActualSeqLensQData() ||
        ge::GRAPH_SUCCESS != CheckFeatureActualSeqLensKvData()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeature()
{
    if (ropeMode_ == RopeMode::ROPE_SPLIT) {
        return CheckFeatureMla();
    } else {
        return CheckFeatureGqa();
    }

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
