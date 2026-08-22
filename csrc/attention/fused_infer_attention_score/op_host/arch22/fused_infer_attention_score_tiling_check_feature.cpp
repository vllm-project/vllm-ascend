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
 * \file fused_infer_attention_score_tiling_check_feature.cpp
 * \brief
 */

#include <numeric>
#include "tiling/tiling_api.h"
#include "fused_infer_attention_score_tiling_check.h"

using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
namespace optiling {
constexpr int64_t SPARSE_MODE_INT_MAX = 2147483647;
constexpr uint32_t ALIGN_SIZE_16 = 16;
constexpr uint32_t ALIGN_SIZE_32 = 32;
constexpr uint32_t HEAD_NUM_ONE = 1;
constexpr uint32_t HEAD_DIM_64 = 64;
constexpr uint32_t HEAD_DIM_128 = 128;
constexpr uint32_t HEAD_DIM_192 = 192;

ge::graphStatus FiaTilingCheck::CheckFeatureNoQuantDtype() const
{
    if (quantMode_ != FiaQuantMode::NO_QUANT) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(inputQType_ != ge::DT_BF16 && inputQType_ != ge::DT_FLOAT16,
        OP_LOGE(opName_, "In %s situation, query dtype only support %s and %s, but got %s",
            QuantModeToSerialString(quantMode_).c_str(),
            FusedDataTypeToSerialString(ge::DT_BF16).c_str(), FusedDataTypeToSerialString(ge::DT_FLOAT16).c_str(),
            FusedDataTypeToSerialString(inputQType_).c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(inputQType_ != inputKvType_,
        OP_LOGE(opName_, "In %s situation, key and value dtype(%s) must equal to query dtype(%s)",
            QuantModeToSerialString(quantMode_).c_str(),
            FusedDataTypeToSerialString(inputQType_).c_str(),
            FusedDataTypeToSerialString(inputKvType_).c_str()),
        return ge::GRAPH_FAILED);

    if (ropeMode_ == RopeMode::ROPE_SPLIT) {
        OP_CHECK_IF((opParamInfo_.queryRope.desc->GetDataType() != opParamInfo_.query.desc->GetDataType()),
            OP_LOGE(opName_, "%s(%s) and %s(%s) must have same dType",
                QUERY_ROPE_NAME.c_str(), FusedDataTypeToSerialString(opParamInfo_.queryRope.desc->GetDataType()).c_str(),
                QUERY_NAME.c_str(), FusedDataTypeToSerialString(opParamInfo_.query.desc->GetDataType()).c_str()),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF((opParamInfo_.keyRope.desc->GetDataType() != opParamInfo_.key.desc->GetDataType()),
            OP_LOGE(opName_, "%s(%s) and %s(%s) must have same dType",
                KEY_ROPE_NAME.c_str(), FusedDataTypeToSerialString(opParamInfo_.keyRope.desc->GetDataType()).c_str(),
                KEY_NAME.c_str(), FusedDataTypeToSerialString(opParamInfo_.key.desc->GetDataType()).c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureBlockSize() const
{
    constexpr int32_t BLOCK_SIZE_ALIGN_SIZE = 16;
    constexpr int32_t BLOCK_SIZE_MAX_SIZE = 1024;
    if (blockSize_ % BLOCK_SIZE_ALIGN_SIZE != 0) {
        OP_LOGE(opName_, "In %s situation, %s should aligned to 16, but got %d.",
            QuantModeToSerialString(quantMode_).c_str(), BLOCK_SIZE_NAME.c_str(), blockSize_);
            return ge::GRAPH_FAILED;
    }

    if (blockSize_ > BLOCK_SIZE_MAX_SIZE) {
        OP_LOGE(opName_, "In %s situation, %s should less equal than 1024, but got %d.",
            QuantModeToSerialString(quantMode_).c_str(), BLOCK_SIZE_NAME.c_str(), blockSize_);
            return ge::GRAPH_FAILED;
    }

    if (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION && blockSize_ == 0) {
        OP_LOGE(opName_, "In %s and storage mode is page attention, %s should not be 0",
            QuantModeToSerialString(quantMode_).c_str(), BLOCK_SIZE_NAME.c_str());
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureLse() const
{
    if (!fiaInfo_.softmaxLseFlag) {
        return ge::GRAPH_SUCCESS;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureTensorList() const
{
    if (kvStorageMode_ == KvStorageMode::TENSOR_LIST) {
        const std::vector<std::string> layoutSupportList = {
            "BSND", "BNSD", "BSH", "BNSD_BSND",
        };
        std::string layout = opParamInfo_.layOut;
        if (std::find(layoutSupportList.begin(), layoutSupportList.end(), layout) == layoutSupportList.end()) {
            OP_LOGE(opName_,
                "when the tensor number of key/value is greater than 1 (this scenario is called tensor list), "
                "input_layout only supports BSH, BSND, BNSD, and BNSD_BSND, but got %s", layout.c_str());
            return ge::GRAPH_FAILED;
        }

        OP_CHECK_IF(ropeMode_ != RopeMode::NO_ROPE,
            OP_LOGE(opName_,
                "when the tensor number of key/value is greater than 1 (this scenario is called tensor list), "
                "query_rope and key_rope should be not exist and the head_dim(D) dimension of query and key should be "
                "equal to the head_dim(D) dimension of value."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureMlaNoquant()
{
    OP_CHECK_IF(socVersion_ == platform_ascendc::SocVersion::ASCEND310P,
        OP_LOGE(opName_, "In %s %s situation, Ascend310P is not supported",
            RopeModeToSerialString(ropeMode_).c_str(), QuantModeToSerialString(quantMode_).c_str()),
        return ge::GRAPH_FAILED);
    if (ge::GRAPH_SUCCESS != CheckFeatureTensorList() ||
        ge::GRAPH_SUCCESS != CheckFeatureBlockSize() ||
        ge::GRAPH_SUCCESS != CheckFeatureInOutDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureActualSeqLens() ||
        ge::GRAPH_SUCCESS != CheckFeatureMask() ||
        ge::GRAPH_SUCCESS != CheckFeatureNoQuantDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureLse() ||
        ge::GRAPH_SUCCESS != CheckFeatureLayout() ||
        ge::GRAPH_SUCCESS != CheckFeatureAxisInfo() ||
        ge::GRAPH_SUCCESS != CheckFeatureLearnableSink() ||
        ge::GRAPH_SUCCESS != CheckFeatureHeadDim() ||
        ge::GRAPH_SUCCESS != CheckFeaturePostQuant()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureMlaAntiquant() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureMlaFullquant() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureMla()
{
    if (quantMode_ == FiaQuantMode::NO_QUANT) {
        return CheckFeatureMlaNoquant();
    } else if (quantMode_ == FiaQuantMode::ANTI_QUANT) {
        return CheckFeatureMlaAntiquant();
    } else if (quantMode_ == FiaQuantMode::FULL_QUANT) {
        return CheckFeatureMlaFullquant();
    }

    return ge::GRAPH_SUCCESS;
}

// 补充sparse9特性的拦截校验
ge::graphStatus FiaTilingCheck::CheckFeatureSparseMode() const
{
    int32_t sparseMode = fiaInfo_.sparseMode;
    // sparse9 仅在rope分离场景下存在，不支持左padding、PSE、公共前缀、后量化等特性 拦截s2 >= s1
    if (sparseMode == SPARSE_MODE_TREE) {
        // 特性校验
        OP_CHECK_IF(ropeMode_ != RopeMode::ROPE_SPLIT,
            OP_LOGE(opName_,
                    "In %s situation, when query_rope and key_rope not exist, %s does not support sparse(%d).", 
                    QuantModeToSerialString(quantMode_).c_str(), SPARSE_MODE_NAME.c_str(), sparseMode),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(fiaInfo_.qPaddingSizeFlag || fiaInfo_.kvPaddingSizeFlag,
            OP_LOGE(opName_,
                    "In %s situation, when sparse is %d, query_padding_size or kv_padding_size should be not exist.", 
                    QuantModeToSerialString(quantMode_).c_str(), sparseMode),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(fiaInfo_.pseShiftFlag,
            OP_LOGE(opName_, 
                    "In %s situation, when sparse is %d, pse_shift should be not exist.", 
                    QuantModeToSerialString(quantMode_).c_str(), sparseMode),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(fiaInfo_.sysPrefixFlag,
            OP_LOGE(opName_, 
                    "In %s situation, when sparse is %d, key_shared_prefix and key_shared_prefix should be not exist.",
                    QuantModeToSerialString(quantMode_).c_str(), sparseMode),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(outputType_ == ge::DT_INT8,
            OP_LOGE(opName_, 
                    "In %s situation, when sparse is %d, output dtype %d is not currently supported.", 
                    QuantModeToSerialString(quantMode_).c_str(), sparseMode, static_cast<int32_t>(outputType_)),
            return ge::GRAPH_FAILED);

        // s2 >= s1拦截
        // tiling下沉场景 由于actualSeqlen得不到，所以不进行校验
        if (fiaInfo_.isMaxWorkspace) {
            return ge::GRAPH_SUCCESS;
        }
        // s2=0 的 batch（入图padding或空tensor场景）不校验 s1<=s2
        int32_t actualSeqSize = std::min(qSize.size(), kvSize.size());
        for (int32_t i = 0; i < actualSeqSize; i++) {
            if (kvSize[i] == 0) {
                continue;
            }
            OP_CHECK_IF(qSize[i] > kvSize[i],
                OP_LOGE(opName_,
                        "In %s situation, when sparse is %d, qSize[%d] should less than or equal to kvSize[%d],"
                        "but got qSize %d and kvSize %d.",
                        QuantModeToSerialString(quantMode_).c_str(), sparseMode, i, i, qSize[i], kvSize[i]),
            return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureMask() const
{
    if (ge::GRAPH_SUCCESS != CheckFeatureSparseMode()) {
        return ge::GRAPH_FAILED;
    }

    if ((!attenMaskFlag_) && (fiaInfo_.sparseMode != SPARSE_MODE_NO_MASK)) {
        OP_LOGE(opName_, "when %s is %d, it not 0, %s should not be null.",
            SPARSE_MODE_NAME.c_str(), fiaInfo_.sparseMode, ATTEN_MASK_NAME.c_str());
        return ge::GRAPH_FAILED;
    }

    if (attenMaskFlag_) {
        size_t maskDimNum = opParamInfo_.attenMask.tensor->GetStorageShape().GetDimNum();
        if ((fiaInfo_.sparseMode == SPARSE_MODE_NO_MASK || fiaInfo_.sparseMode == SPARSE_MODE_ALL_MASK) && 
            maskDimNum == DIM_NUM_TWO) {
            if (ropeMode_ == RopeMode::NO_ROPE) {
                const std::vector<std::string> layoutSupportList = {
                    "BSH", "BSND", "BNSD", "BNSD_BSND",
                };
                std::string layout = opParamInfo_.layOut;
                OP_CHECK_IF(std::find(layoutSupportList.begin(), layoutSupportList.end(), layout) == layoutSupportList.end(),
                    OP_LOGE(opName_,
                        "In %s situation, rope not exits and qkHeadDim = vHeadDim, when sparseMode = 0 or 1, "
                        "two dim mask only support for layout BSH,BSND,BNSD,BNSD_BSND, but got %s",
                        QuantModeToSerialString(quantMode_).c_str(), layout.c_str()),
                    return ge::GRAPH_FAILED);
            } else {
                OP_LOGE(opName_,
                        "In %s situation, rope exits or qkHeadDim != vHeadDim, when sparseMode = 0 or 1, two dim mask is not supported.",
                        QuantModeToSerialString(quantMode_).c_str());
                return ge::GRAPH_FAILED;
            }
        }
    }

    if (ropeMode_ == RopeMode::ROPE_SPLIT && vHeadDim_ == 512U) {
        int32_t sparseMode = fiaInfo_.sparseMode;
        if (sparseMode != SPARSE_MODE_NO_MASK && sparseMode != SPARSE_MODE_RIGHT_DOWN && 
            sparseMode != SPARSE_MODE_BAND && sparseMode != SPARSE_MODE_TREE) {
            OP_LOGE(opName_,
                    "In %s situation, when query_rope and key_rope exsists and the head dim of value is %u, %s only "
                    "support 0/3/4/9, but got %d.",
                    QuantModeToSerialString(quantMode_).c_str(), vHeadDim_, SPARSE_MODE_NAME.c_str(), sparseMode);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

int64_t FiaTilingCheck::GetQActualSeqLenForPostQuant(uint32_t batchIdx, const gert::Tensor *tempData,
    int64_t actualLenDims) const
{
    if ((actualLenDims == 0) || (tempData == nullptr) || (tempData->GetData<int64_t>() == nullptr)) {
        return fiaInfo_.s1Size;
    }
    if (!fiaInfo_.isAccumQSeq) {
        return (actualLenDims > 1) ? static_cast<uint32_t>(tempData->GetData<int64_t>()[batchIdx]) :
                                     static_cast<uint32_t>(tempData->GetData<int64_t>()[0]);
    }
    if (batchIdx == 0) {
        return static_cast<uint32_t>(tempData->GetData<int64_t>()[0]);
    }
    return static_cast<uint32_t>(tempData->GetData<int64_t>()[batchIdx]) -
           static_cast<uint32_t>(tempData->GetData<int64_t>()[batchIdx - 1]);
}

int64_t FiaTilingCheck::GetKVActualSeqLenForPostQuant(uint32_t batchIdx, const gert::Tensor *tempDataKV,
    int64_t actualLenDimsKV) const
{
    if ((actualLenDimsKV == 0) || (tempDataKV == nullptr) ||
        (tempDataKV->GetData<int64_t>() == nullptr)) {
        if (fiaInfo_.kvStorageMode == KvStorageMode::BATCH_CONTINUOUS) {
            return fiaInfo_.s2Size;
        }
        return fiaInfo_.kvListSeqLens[batchIdx];
    }
    if (!fiaInfo_.isAccumKVSeq) {
        return (actualLenDimsKV > 1) ?
                   static_cast<uint32_t>(tempDataKV->GetData<int64_t>()[batchIdx]) :
                   static_cast<uint32_t>(tempDataKV->GetData<int64_t>()[0]);
    }
    if (batchIdx == 0) {
        return static_cast<uint32_t>(tempDataKV->GetData<int64_t>()[0]);
    }
    return static_cast<uint32_t>(tempDataKV->GetData<int64_t>()[batchIdx]) -
           static_cast<uint32_t>(tempDataKV->GetData<int64_t>()[batchIdx - 1]);
}

void FiaTilingCheck::CalcPostQuantTokensPerBatch(int64_t &preTokensPerbatch, int64_t &nextTokensPerbatch,
    int64_t actualSeqLenKV, int64_t actualSeqLenQ) const
{
    if (fiaInfo_.sparseMode == SPARSE_MODE_RIGHT_DOWN) {
        preTokensPerbatch = static_cast<int64_t>(SPARSE_MODE_INT_MAX);
        nextTokensPerbatch = actualSeqLenKV + static_cast<int64_t>(fiaInfo_.systemPrefixLen) - actualSeqLenQ;
    } else if (fiaInfo_.sparseMode == SPARSE_MODE_BAND) {
        preTokensPerbatch = fiaInfo_.preToken - actualSeqLenKV -
                            static_cast<int64_t>(fiaInfo_.systemPrefixLen) + actualSeqLenQ;
        nextTokensPerbatch = fiaInfo_.nextToken + actualSeqLenKV +
                             static_cast<int64_t>(fiaInfo_.systemPrefixLen) - actualSeqLenQ;
    } else {
        preTokensPerbatch = fiaInfo_.preToken;
        nextTokensPerbatch = fiaInfo_.nextToken;
    }
}

ge::graphStatus FiaTilingCheck::CheckFeaturePostQuantLayout() const
{
    if (fiaInfo_.ropeMode == RopeMode::ROPE_SPLIT) {
        OP_LOGE(opName_, "postquant do not support qkHeadDim = vHeadDim and rope exist");
        return ge::GRAPH_FAILED;
    }
    if (fiaInfo_.ropeMode == RopeMode::ROPE_COMBINE) {
        OP_LOGE(opName_, "postquant do not support qkHeadDim != vHeadDim");
        return ge::GRAPH_FAILED;
    }
    const std::vector<std::string> layoutSupportList = {
        "BSND", "BNSD", "BSH", "BNSD_BSND",
    };
    std::string layout = opParamInfo_.layOut;
    if (std::find(layoutSupportList.begin(), layoutSupportList.end(), layout) == layoutSupportList.end()) {
        OP_LOGE(opName_,
                "when enable postquant, input_layout only supports BSH, BSND, BNSD, "
                "and BNSD_BSND, but got %s",
                layout.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeaturePostQuantOffsetPerBatch(bool checkPostQuantOffset,
    int64_t preTokensPerbatch, int64_t nextTokensPerbatch, int64_t actualSeqLenKV,
    int64_t actualSeqLenQ, int64_t preTokens, int64_t nextTokens) const
{
    OP_CHECK_IF(
        (checkPostQuantOffset && ((preTokensPerbatch + actualSeqLenKV +
                                       static_cast<int64_t>(fiaInfo_.systemPrefixLen) - actualSeqLenQ <
                                   0) ||
                                  (nextTokensPerbatch < 0))),
        OP_LOGE(opName_,
                "When sparse mode = %d, output dtype is int8, the output's dequant offset "
                "is not null or empty tensor, "
                "preTokens = %ld and nextTokens = %ld, some rows of the matrix do not "
                "participate in the calculation, "
                "the accuracy of the final result will be incorrect. Please see the "
                "documentation for more details.",
                fiaInfo_.sparseMode, preTokens, nextTokens),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeaturePostQuant() const
{
    if (!fiaInfo_.isOutQuantEnable) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckFeaturePostQuantLayout()) {
        return ge::GRAPH_FAILED;
    }
    if (fiaInfo_.isLegacyIfa) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(
        (fiaInfo_.sparseMode == SPARSE_MODE_BAND && (fiaInfo_.preToken < 0 || fiaInfo_.nextToken < 0)),
        OP_LOGE(opName_,
                "When output type is int8, sparse mode = 4, preTokens (%ld) or nextTokens (%ld) cannot be negative.",
                fiaInfo_.preToken, fiaInfo_.nextToken),
        return ge::GRAPH_FAILED);
    bool checkPostQuantOffset =
        (fiaInfo_.outputType == ge::DT_INT8) &&
        (opParamInfo_.quantOffset2.tensor != nullptr && opParamInfo_.quantOffset2.desc != nullptr) &&
        (opParamInfo_.quantOffset2.tensor->GetStorageShape().GetShapeSize() != 0);

    if (!fiaInfo_.isMaxWorkspace) {
        const gert::Tensor *tempData = fiaInfo_.opParamInfo.actualSeqLengthsQ.tensor;
        const gert::Tensor *tempDataKV = fiaInfo_.opParamInfo.actualSeqLengths.tensor;
        const int64_t preTokens = fiaInfo_.opParamInfo.preToken == nullptr ? 0 : *opParamInfo_.preToken;
        const int64_t nextTokens = fiaInfo_.opParamInfo.nextToken == nullptr ? 0 : *opParamInfo_.nextToken;
        int64_t actualLenDims = (tempData != nullptr) ? tempData->GetShapeSize() : 0;
        int64_t actualLenDimsKV = (tempDataKV != nullptr) ? tempDataKV->GetShapeSize() : 0;
        for (uint32_t i = 0; i < fiaInfo_.bSize; i++) {
            int64_t actualSeqLenQ = GetQActualSeqLenForPostQuant(i, tempData, actualLenDims);
            int64_t actualSeqLenKV = GetKVActualSeqLenForPostQuant(i, tempDataKV, actualLenDimsKV);
            int64_t preTokensPerbatch = 0;
            int64_t nextTokensPerbatch = 0;
            CalcPostQuantTokensPerBatch(preTokensPerbatch, nextTokensPerbatch, actualSeqLenKV, actualSeqLenQ);
            if (ge::GRAPH_SUCCESS != CheckFeaturePostQuantOffsetPerBatch(checkPostQuantOffset,
                preTokensPerbatch, nextTokensPerbatch, actualSeqLenKV, actualSeqLenQ, preTokens, nextTokens)) {
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus FiaTilingCheck::CheckFeatureLeftPadding() const
{
    if (fiaInfo_.qPaddingSizeFlag || fiaInfo_.kvPaddingSizeFlag) {
        const std::vector<std::string> layoutSupportList = {
            "BSND", "BNSD", "BSH", "BNSD_BSND",
        };
        std::string layout = opParamInfo_.layOut;
        if (std::find(layoutSupportList.begin(), layoutSupportList.end(), layout) == layoutSupportList.end()) {
            OP_LOGE(opName_,
                    "when query_padding_size or kv_padding_size exists, input_layout only supports BSH, BSND, BNSD, "
                    "and BNSD_BSND, but got %s",
                    layout.c_str());
            return ge::GRAPH_FAILED;
        }

        OP_CHECK_IF(ropeMode_ != RopeMode::NO_ROPE,
            OP_LOGE(opName_,
                    "when query_padding_size or kv_padding_size exists, query_rope and key_rope should be not exist "
                    "and the "
                    "head_dim(D) dimension of query and key should be equal to the head_dim(D) dimension of value."),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(kvStorageMode_ == KvStorageMode::TENSOR_LIST,
            OP_LOGE(opName_,
                "when query_padding_size or kv_padding_size exists, key/value tensorlist is not supported; in this "
                "case, the tensor number of key/value should be 1"),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(kvStorageMode_ == KvStorageMode::PAGE_ATTENTION,
            OP_LOGE(opName_,
                "when query_padding_size or kv_padding_size exists, page attention is not supported; in this case, "
                "block_table should exist and block_size is not 0"),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(fiaInfo_.sysPrefixFlag,
            OP_LOGE(opName_,
                    "when query_padding_size exists, key_shared_prefix and key_shared_prefix should be not exist."),
            return ge::GRAPH_FAILED);
    }

    if (fiaInfo_.qPaddingSizeFlag) {
        OP_CHECK_IF(!(fiaInfo_.actualLenQDims != 0U),
            OP_LOGE(opName_, "when query_padding_size exists, the query's actual sequence lengths are required."),
            return ge::GRAPH_FAILED);
    }

    if (fiaInfo_.kvPaddingSizeFlag) {
        OP_CHECK_IF(!(fiaInfo_.actualLenKvDims != 0U),
            OP_LOGE(opName_, "when kv_padding_size exists, the key/value's actual sequence lengths are required."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeaturePSE() const
{
    if (fiaInfo_.pseShiftFlag) {
        const std::vector<std::string> layoutSupportList = {
            "BSND", "BNSD", "BSH", "BNSD_BSND",
        };
        std::string layout = opParamInfo_.layOut;
        if (std::find(layoutSupportList.begin(), layoutSupportList.end(), layout) == layoutSupportList.end()) {
            OP_LOGE(opName_,
                    "when pse_shift exists, input_layout only supports BSH, BSND, BNSD, and BNSD_BSND, but got %s",
                    layout.c_str());
            return ge::GRAPH_FAILED;
        }

        OP_CHECK_IF(ropeMode_ != RopeMode::NO_ROPE,
            OP_LOGE(opName_, "when pse_shift exists, query_rope and key_rope should be not exist and the head_dim(D) "
                             "dimension of query and key should be equal to the head_dim(D) dimension of value."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureLearnableSink() const
{
    if (fiaInfo_.learnableSinkFlag == false) {
        return ge::GRAPH_SUCCESS;
    }

    const std::vector<size_t> sinkDimNumList = {DIM_NUM_ONE};
    if (ge::GRAPH_SUCCESS != CheckDimNumSupport(opParamInfo_.learnableSink.tensor, sinkDimNumList, LEARNABLE_SINK_NAME)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t sinkDim = opParamInfo_.learnableSink.tensor->GetStorageShape().GetDim(0);
    OP_CHECK_IF(sinkDim != fiaInfo_.n1Size,
        OP_LOGE(opName_, "learnable_sink enable, sink shape(%u) must be same equal queryN(%u)!", sinkDim, fiaInfo_.n1Size),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((opParamInfo_.learnableSink.desc->GetDataType() != ge::DT_BF16),
            OP_LOGE(opName_, "When learnable_sink enable, sink dtype must be bf16!"),
            return ge::GRAPH_FAILED);
            
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureLayout() const
{
    const std::vector<std::string> layoutSupportList = {
        "BSH", "BSND", "BNSD", "TND", "NTD", "BSH_NBSD", "BSND_NBSD", "BNSD_NBSD", "TND_NTD", "NTD_TND", "BSH_BNSD", "BSND_BNSD", "BNSD_BSND"
    };
    std::string layout = opParamInfo_.layOut;
    OP_CHECK_IF(std::find(layoutSupportList.begin(), layoutSupportList.end(), layout) == layoutSupportList.end(),
        OP_LOGE(opName_, "In %s %s situation, layout only supports BSH, BSND, BNSD, TND, NTD, BSH_NBSD, BSND_NBSD, BNSD_NBSD, TND_NTD, NTD_TND, BSH_BNSD, BSND_BNSD, BNSD_BSND, but got %s",
            QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), layout.c_str()),
        return ge::GRAPH_FAILED);

    if (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS) {
        OP_CHECK_IF(kvLayout_ != FiaLayout::BSH && kvLayout_ != FiaLayout::BSND && kvLayout_ != FiaLayout::BNSD &&
            kvLayout_ != FiaLayout::TND && kvLayout_ != FiaLayout::NTD,
            OP_LOGE(opName_, "In %s %s situation, key/value's layout only support BSH, BSND, BNSD, TND and NTD in batch continuous scene, but got %s",
                QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), LayoutToSerialString(kvLayout_).c_str()),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(kvLayout_ != qLayout_,
            OP_LOGE(opName_, "In %s %s situation, key/value's layout and query's layout should be same in batch continuous scene.",
                QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str()),
            return ge::GRAPH_FAILED);
    } else if (kvStorageMode_ == KvStorageMode::TENSOR_LIST) {
        OP_CHECK_IF(kvLayout_ != FiaLayout::BSH && kvLayout_ != FiaLayout::BSND && kvLayout_ != FiaLayout::BNSD,
            OP_LOGE(opName_, "In %s %s situation, key/value's layout only support BSH, BSND and BNSD in tensor list scene, but got %s",
                QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), LayoutToSerialString(kvLayout_).c_str()),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(kvLayout_ != qLayout_,
            OP_LOGE(opName_, "In %s %s situation, key/value's layout and query's layout should be same in tensor list scene.",
            QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str()),
            return ge::GRAPH_FAILED);
    } else if (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION) {
        OP_CHECK_IF(kvLayout_ == FiaLayout::BnBsH && (qLayout_ != FiaLayout::BSH && qLayout_ != FiaLayout::BSND &&
                        qLayout_ != FiaLayout::BNSD && qLayout_ != FiaLayout::TND && qLayout_ != FiaLayout::NTD),
            OP_LOGE(opName_, "In %s %s situation, the key/value's layout is BnBsH, %s layout must be BSH, BSND, BNSD TND and TND in page attention scene, but got %s",
                QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), QUERY_NAME.c_str(), LayoutToSerialString(qLayout_).c_str()),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(kvLayout_ == FiaLayout::BnNBsD && (qLayout_ != FiaLayout::BSH && qLayout_ != FiaLayout::BSND &&
                        qLayout_ != FiaLayout::BNSD && qLayout_ != FiaLayout::TND && qLayout_ != FiaLayout::NTD),
            OP_LOGE(opName_, "In %s %s situation, the key/value's layout is BnNBsD, %s layout must be BSH, BSND, BNSD TND and TND in page attention scene, but got %s",
                QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), QUERY_NAME.c_str(), LayoutToSerialString(qLayout_).c_str()),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(kvLayout_ == FiaLayout::NZ && (qLayout_ != FiaLayout::BSH && qLayout_ != FiaLayout::BSND &&
                        qLayout_ != FiaLayout::BNSD && qLayout_ != FiaLayout::TND && qLayout_ != FiaLayout::NTD),
            OP_LOGE(opName_, "In %s %s situation, the key/value's layout is BnNBsD, %s layout must be BSH, BSND, BNSD TND and TND in page attention scene, but got %s",
                QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), QUERY_NAME.c_str(), LayoutToSerialString(qLayout_).c_str()),
            return ge::GRAPH_FAILED);
    }

    // splitRope场景 QK_D=512时支持的layout
    const std::vector<std::string> splitRopeLayoutSupportList = {
        "BSH", "BSND", "BNSD", "TND", "BNSD_NBSD", "BSND_NBSD", "BSH_NBSD", "TND_NTD"
    };
    // 受限制的layout，适用于splitRope且QK_D=128、combine场景
    const std::vector<std::string> restrictedLayoutSupportList = {
        "BSH", "BSND", "BNSD", "BNSD_BSND", "TND", "NTD", "BSH_BNSD", "BSND_BNSD", "NTD_TND"
    };
    // noRope场景 Q_S>1时，仅支持16对齐、在Q_S=1时，D在1~512全泛化的layout
    const std::vector<std::string> noRopeLayoutSupportListA = {
        "BSH", "BSND", "BNSD"
    };
    // noRope场景 BNSD_BSND在Q_S>1时，仅支持16对齐、在Q_S=1的时候，D仅支持64和128的layout
    const std::vector<std::string> noRopeLayoutSupportListB = {
        "BNSD_BSND"
    };
    // noRope场景 不区分Q_S，D仅支持了64和128的layout
    const std::vector<std::string> noRopeLayoutSupportListC = {
        "NTD", "BSH_BNSD", "BSND_BNSD", "NTD_TND"
    };
    // noRope场景 仅支持D为64和128和192的layout
    const std::vector<std::string> noRopeLayoutSupportListD = {
        "TND"
    };

    if (fiaInfo_.ropeMode == RopeMode::ROPE_SPLIT) {
        if (vHeadDim_ == HEAD_DIM_512) {
            OP_CHECK_IF(std::find(splitRopeLayoutSupportList.begin(), splitRopeLayoutSupportList.end(), layout) == splitRopeLayoutSupportList.end(),
            OP_LOGE(opName_, "In %s %s situation, when value headDim = 512, layout only supports BSH, BSND, BNSD, TND, BNSD_NBSD, BSND_NBSD, BSH_NBSD, TND_NTD, but got %s",
                QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), layout.c_str()),
            return ge::GRAPH_FAILED);
        }
        if (vHeadDim_ == HEAD_DIM_128) {
            OP_CHECK_IF(std::find(restrictedLayoutSupportList.begin(), restrictedLayoutSupportList.end(), layout) == restrictedLayoutSupportList.end(),
            OP_LOGE(opName_, "In %s %s situation, when value headDim = 128, layout only supports BSH, BSND, BNSD, BNSD_BSND, TND, NTD, BSH_BNSD, BSND_BNSD, NTD_TND, but got %s",
                QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), layout.c_str()),
            return ge::GRAPH_FAILED);
        }
    } else if (fiaInfo_.ropeMode == RopeMode::NO_ROPE) {
        OP_CHECK_IF(std::find(restrictedLayoutSupportList.begin(), restrictedLayoutSupportList.end(), layout) == restrictedLayoutSupportList.end(),
            OP_LOGE(opName_, "In %s %s situation, layout only supports BSH、BSND、BNSD、BNSD_BSND、TND、NTD、BSH_BNSD、BSND_BNSD、NTD_TND, but got %s.",
                QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), layout.c_str()),
            return ge::GRAPH_FAILED);

        if (!fiaInfo_.isLegacyIfa && (vHeadDim_ % ALIGN_SIZE_16 != 0)) {
            OP_CHECK_IF(std::find(noRopeLayoutSupportListA.begin(), noRopeLayoutSupportListA.end(), layout) != noRopeLayoutSupportListA.end(),
            OP_LOGE(opName_, "In %s %s situation, when Qs>1 and input_layout is %s, headDim of query|key|value should be align to 16.",
                QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), layout.c_str()),
            return ge::GRAPH_FAILED);

            OP_CHECK_IF(std::find(noRopeLayoutSupportListB.begin(), noRopeLayoutSupportListB.end(), layout) != noRopeLayoutSupportListB.end(),
            OP_LOGE(opName_, "In %s %s situation, when Qs>1 and input_layout is %s, headDim of query|key|value should be align to 16.",
                QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), layout.c_str()),
            return ge::GRAPH_FAILED);
        }
        if (fiaInfo_.isLegacyIfa && (vHeadDim_ != HEAD_DIM_64 && vHeadDim_ != HEAD_DIM_128)) {
            OP_CHECK_IF(std::find(noRopeLayoutSupportListB.begin(), noRopeLayoutSupportListB.end(), layout) != noRopeLayoutSupportListB.end(),
            OP_LOGE(opName_, "In %s %s situation, when Qs=1 and input_layout is BNSD_BSND, only query|key|value headDim = 64/128 are supported, but got %u",
                QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), vHeadDim_),
            return ge::GRAPH_FAILED);
        }
        if (std::find(noRopeLayoutSupportListC.begin(), noRopeLayoutSupportListC.end(), layout) != noRopeLayoutSupportListC.end()) {
            OP_CHECK_IF(vHeadDim_ != HEAD_DIM_64 && vHeadDim_ != HEAD_DIM_128,
            OP_LOGE(opName_, "In %s %s situation, when input_layout is NTD, BSH_BNSD, BSND_BNSD, NTD_TND, only query|key|value headDim = 64/128 are supported, but got %u",
                QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), vHeadDim_),
            return ge::GRAPH_FAILED);
        }
        if (std::find(noRopeLayoutSupportListD.begin(), noRopeLayoutSupportListD.end(), layout) != noRopeLayoutSupportListD.end()) {
            OP_CHECK_IF(vHeadDim_ != HEAD_DIM_64 && vHeadDim_ != HEAD_DIM_128 && vHeadDim_ != HEAD_DIM_192,
            OP_LOGE(opName_, "In %s %s situation, when input_layout is TND, only query|key|value headDim = 64/128/192 are supported, but got %u",
                QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), vHeadDim_),
            return ge::GRAPH_FAILED);
        }
    } else if (fiaInfo_.ropeMode == RopeMode::ROPE_COMBINE) {
        if (std::find(restrictedLayoutSupportList.begin(), restrictedLayoutSupportList.end(), layout) != restrictedLayoutSupportList.end()) {
            OP_CHECK_IF(qkHeadDim_ != HEAD_DIM_192 || vHeadDim_ != HEAD_DIM_128,
            OP_LOGE(opName_, "In %s %s situation, when input_layout is BSH, BSND, BNSD, BNSD_BSND, TND, NTD, BSH_BNSD, BSND_BNSD, NTD_TND, and the headDim shared by query and key is not equal to that of value, only query|key headDim = 192, value headDim = 128 are supported, but got query|key headDim: %u, value headDim: %u",
                QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), qkHeadDim_, vHeadDim_),
            return ge::GRAPH_FAILED);
        } else {
            OP_LOGE(opName_, "In %s %s situation, layout only supports BSH、BSND、BNSD、BNSD_BSND、TND、NTD、BSH_BNSD、BSND_BNSD、NTD_TND, but got %s.",
                QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), layout.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureAxisInfo() const
{
    constexpr uint32_t MAX_ACTUAL_SEQ_LEN_BYTE = 64U * 1024U;
    constexpr uint32_t MAX_B_SIZE = 256U;

    OP_CHECK_IF(actualSeqLengthsQSize_ > MAX_ACTUAL_SEQ_LEN_BYTE,
    OP_LOGE(opName_, "In %s situation, actual sequence length q should be smaller or equal to 64K, but got %u",
        QuantModeToSerialString(quantMode_).c_str(), actualSeqLengthsQSize_),
    return ge::GRAPH_FAILED);

    OP_CHECK_IF(actualSeqLengthsKvSize_ > MAX_ACTUAL_SEQ_LEN_BYTE,
    OP_LOGE(opName_, "In %s situation, actual sequence length kv should be smaller or equal to 64K, but got %u",
        QuantModeToSerialString(quantMode_).c_str(), actualSeqLengthsKvSize_),
    return ge::GRAPH_FAILED);

    if (kvStorageMode_ == KvStorageMode::TENSOR_LIST) {
        OP_CHECK_IF(bSize_ > MAX_B_SIZE,
            OP_LOGE(opName_, "In %s situation, batch size(%u) cannot be greater than %u in tensor list scene.",
                QuantModeToSerialString(quantMode_).c_str(), bSize_, MAX_B_SIZE),
            return ge::GRAPH_FAILED);
    }

    const std::vector<std::int32_t> ropeSplitgSizeSupportList = {
        1, 2, 4, 8, 16, 32, 64, 128
    };

    if (fiaInfo_.ropeMode == RopeMode::ROPE_SPLIT && vHeadDim_ == HEAD_DIM_512) {
        OP_CHECK_IF(std::find(ropeSplitgSizeSupportList.begin(), ropeSplitgSizeSupportList.end(), gSize_) == ropeSplitgSizeSupportList.end(),
        OP_LOGE(opName_, "In %s %s situation, when query|key|value headDim = 512, layout only supports 1, 2, 4, 8, 16, 32, 64, 128, but got %u",
            QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), gSize_),
        return ge::GRAPH_FAILED);

        OP_CHECK_IF(n2Size_ != HEAD_NUM_ONE,
        OP_LOGE(opName_, "In %s %s situation, when query|key|value headDim = 512, kvN should be equals to 1.",
            QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str()),
        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureGqaPrefix() const
{
    if (!fiaInfo_.sysPrefixFlag) {
        return ge::GRAPH_SUCCESS;
    }
    const std::vector<std::string> layoutSupportList = {
        "BSND", "BNSD", "BSH", "BNSD_BSND",
    };
    std::string layout = opParamInfo_.layOut;
    if (std::find(layoutSupportList.begin(), layoutSupportList.end(), layout) == layoutSupportList.end()) {
        OP_LOGE(opName_,
                "when system prefix exists, input_layout only supports BSH, BSND, BNSD, and BNSD_BSND, but got %s",
                layout.c_str());
        return ge::GRAPH_FAILED;
    }
    int32_t sparseMode = *opParamInfo_.sparseMode;
    auto *maskTensor = opParamInfo_.attenMask.tensor;
    if (attenMaskFlag_ && (sparseMode == SPARSE_MODE_NO_MASK || sparseMode == SPARSE_MODE_ALL_MASK)) {
        uint32_t maskS2 = maskTensor->GetStorageShape().GetDim(maskTensor->GetStorageShape().GetDimNum() - 1);
        uint32_t totalLen = fiaInfo_.systemPrefixLen + fiaInfo_.maxActualseq;
        if (totalLen > maskS2) {
            OP_LOGE(opName_, "s2Size + systemPrefix (%u) is greater than mask s2 size (%u)", totalLen, maskS2);
            return ge::GRAPH_FAILED;
        }
    }
    if (fiaInfo_.ropeMode !=  RopeMode::NO_ROPE) {
        OP_LOGE(opName_, "system prefix do not support rope");
        return ge::GRAPH_FAILED;
    }

    if (fiaInfo_.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        OP_LOGE(opName_, "system prefix do not support PAGE_ATTENTION");
        return ge::GRAPH_FAILED;
    }
    if (fiaInfo_.kvStorageMode == KvStorageMode::TENSOR_LIST && fiaInfo_.s1Size > 1) {
        OP_LOGE(opName_, "system prefix do not support qs > 1 and enable TENSORLIST");
        return ge::GRAPH_FAILED;
    }

    if (fiaInfo_.pseShiftFlag) {
        if (fiaInfo_.s2Size + fiaInfo_.systemPrefixLen > fiaInfo_.pseShiftS2) {
            OP_LOGE(opName_, "when enable pse and system prefix, pse s2 Size greater than kv s2size + systemPrefixLen");
            return ge::GRAPH_FAILED;   
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureHeadDim() const
{
    constexpr uint32_t MAX_HEAD_DIM = 512;
    constexpr uint32_t MAX_ROPE_DIM = 64;

    OP_CHECK_IF(vHeadDim_ > MAX_HEAD_DIM,
    OP_LOGE(opName_, "In %s situation, headDim of value should be smaller or equal to 512, but got %u",
        QuantModeToSerialString(quantMode_).c_str(), vHeadDim_),
    return ge::GRAPH_FAILED);

    OP_CHECK_IF((fiaInfo_.ropeMode == RopeMode::ROPE_SPLIT || fiaInfo_.ropeMode == RopeMode::ROPE_COMBINE) && ropeHeadDim_ != MAX_ROPE_DIM,
    OP_LOGE(opName_, "In %s situation, headDim of Rope should be equal to 64, but got %u",
        QuantModeToSerialString(quantMode_).c_str(), ropeHeadDim_),
    return ge::GRAPH_FAILED);

    constexpr int32_t D_ALIGN_SIZE = 16;
    if (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION && kvLayout_ == FiaLayout::NZ) {
        OP_CHECK_IF((vHeadDim_ % D_ALIGN_SIZE != 0) || (qkHeadDim_ % D_ALIGN_SIZE != 0),
        OP_LOGE(opName_, "In %s situation, when the dim of key&value is 5, headDim of query|key|value should be align to 16, but got keyHeadDim:%u, queryHeadDim and keyHeadDim:%u",
            QuantModeToSerialString(quantMode_).c_str(), vHeadDim_, qkHeadDim_),
        return ge::GRAPH_FAILED);
    }

    if (fiaInfo_.ropeMode == RopeMode::NO_ROPE) {
        if (!fiaInfo_.isLegacyIfa) {
            OP_CHECK_IF((!fiaInfo_.isOutQuantEnable && (vHeadDim_ % ALIGN_SIZE_16 != 0)),
            OP_LOGE(opName_, "In %s %s situation, when Qs>1, headDim of query|key|value should be align to 16, but got value headDim:%u, query|key headDim:%u",
                QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), vHeadDim_, qkHeadDim_),
            return ge::GRAPH_FAILED);

            OP_CHECK_IF((fiaInfo_.isOutQuantEnable && (vHeadDim_ % ALIGN_SIZE_32 != 0)),
 	        OP_LOGE(opName_, "In %s %s situation, when Qs>1 and enable postquant, headDim of query|key|value should be align to 32, but got value headDim:%u, query|key headDim:%u",
 	            QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), vHeadDim_, qkHeadDim_),
 	        return ge::GRAPH_FAILED);
        }
    } else if (fiaInfo_.ropeMode == RopeMode::ROPE_SPLIT) {
        OP_CHECK_IF(!(vHeadDim_ == HEAD_DIM_512 && ropeHeadDim_ == HEAD_DIM_64) && !(vHeadDim_ == HEAD_DIM_128 && ropeHeadDim_ == HEAD_DIM_64),
        OP_LOGE(opName_, "In %s %s situation, only value matrix headDim = 128/512 and rope headDim = 64 are supported, but got value matrix headDim:%u, rope headDim:%u.",
            QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), vHeadDim_, ropeHeadDim_),
        return ge::GRAPH_FAILED);
    } else if (fiaInfo_.ropeMode == RopeMode::ROPE_COMBINE) {
        OP_CHECK_IF(!(vHeadDim_ == HEAD_DIM_128 && ropeHeadDim_ == HEAD_DIM_64),
        OP_LOGE(opName_, "In %s %s situation, only value matrix headDim = 128 and rope headDim = 64 are supported, but got value matrix headDim:%u, rope headDim:%u.",
            QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), vHeadDim_, ropeHeadDim_),
        return ge::GRAPH_FAILED);
    }
    if (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION && kvLayout_ == FiaLayout::NZ &&
        ropeMode_ == RopeMode::NO_ROPE) {
        const std::vector<std::int32_t> nzNoRopeDSupportList = {64, 128};
        if (std::find(nzNoRopeDSupportList.begin(), nzNoRopeDSupportList.end(), qkHeadDim_) ==
                nzNoRopeDSupportList.end() ||
            std::find(nzNoRopeDSupportList.begin(), nzNoRopeDSupportList.end(), vHeadDim_) ==
                nzNoRopeDSupportList.end()) {
            OP_LOGE(opName_,
                    "In %s %s situation, when the dim of key&value is 5, and the headDim shared by query and key is "
                    "equal to that of value. The headDim of query|key|value should be 64 | "
                    "128, but got valueHeadDim:%u, queryHeadDim and keyHeadDim:%u",
                    QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), vHeadDim_,
                    qkHeadDim_);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureGqaNoquant()
{
    OP_CHECK_IF(socVersion_ == platform_ascendc::SocVersion::ASCEND310P,
        OP_LOGE(opName_, "In %s %s situation, Ascend310P is not supported",
            RopeModeToSerialString(ropeMode_).c_str(), QuantModeToSerialString(quantMode_).c_str()),
        return ge::GRAPH_FAILED);
    if (ge::GRAPH_SUCCESS != CheckFeatureTensorList() ||
        ge::GRAPH_SUCCESS != CheckFeatureBlockSize() ||
        ge::GRAPH_SUCCESS != CheckFeatureInOutDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureActualSeqLens() ||
        ge::GRAPH_SUCCESS != CheckFeatureMask() ||
        ge::GRAPH_SUCCESS != CheckFeatureNoQuantDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureLse() ||
        ge::GRAPH_SUCCESS != CheckFeatureLayout() ||
        ge::GRAPH_SUCCESS != CheckFeatureAxisInfo() ||
        ge::GRAPH_SUCCESS != CheckFeatureLearnableSink() ||
        ge::GRAPH_SUCCESS != CheckFeatureGqaPrefix() ||
        ge::GRAPH_SUCCESS != CheckFeatureLeftPadding() ||
        ge::GRAPH_SUCCESS != CheckFeaturePostQuant() ||
        ge::GRAPH_SUCCESS != CheckFeaturePSE() ||
        ge::GRAPH_SUCCESS != CheckFeatureHeadDim()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureGqaAntiquant() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureGqaFullquant() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureGqa()
{
    if (quantMode_ == FiaQuantMode::NO_QUANT) {
        return CheckFeatureGqaNoquant();
    } else if (quantMode_ == FiaQuantMode::ANTI_QUANT) {
        return CheckFeatureGqaAntiquant();
    } else if (quantMode_ == FiaQuantMode::FULL_QUANT) {
        return CheckFeatureGqaFullquant();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureActualSeqLensExistence() const
{
    if ((qLayout_ == FiaLayout::TND || qLayout_ == FiaLayout::NTD)) {
        OP_CHECK_IF(opParamInfo_.actualSeqLengthsQ.tensor == nullptr,
            OP_LOGE(opName_, "when %s's layout is %s, %s should not be null.", QUERY_NAME.c_str(), LayoutToSerialString(qLayout_).c_str(),
                ACTUAL_SEQ_Q_LEN_NAME.c_str()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.actualSeqLengths.tensor == nullptr,
            OP_LOGE(opName_, "when %s's layout is %s, %s should not be null.", KEY_NAME.c_str(), LayoutToSerialString(kvLayout_).c_str(),
                ACTUAL_SEQ_KV_LEN_NAME.c_str()),
            return ge::GRAPH_FAILED);

        if (!fiaInfo_.isMaxWorkspace) {
            OP_CHECK_IF(opParamInfo_.actualSeqLengthsQ.tensor->GetData<int64_t>() == nullptr,
                OP_LOGE(opName_, "when %s's layout is %s, %s data should not be null.", QUERY_NAME.c_str(), LayoutToSerialString(qLayout_).c_str(),
                    ACTUAL_SEQ_Q_LEN_NAME.c_str()),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(opParamInfo_.actualSeqLengths.tensor->GetData<int64_t>() == nullptr,
                OP_LOGE(opName_, "when %s's layout is %s, %s data should not be null.", KEY_NAME.c_str(), LayoutToSerialString(kvLayout_).c_str(),
                    ACTUAL_SEQ_KV_LEN_NAME.c_str()),
                return ge::GRAPH_FAILED);
        }
    } else {
        if (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION) {
            OP_CHECK_IF(opParamInfo_.actualSeqLengths.tensor == nullptr,
                OP_LOGE(opName_, "In page attention scene, %s should not be null.", ACTUAL_SEQ_KV_LEN_NAME.c_str()),
                return ge::GRAPH_FAILED);
            if (!fiaInfo_.isMaxWorkspace) {
                OP_CHECK_IF(opParamInfo_.actualSeqLengths.tensor->GetData<int64_t>() == nullptr,
                    OP_LOGE(opName_, "In page attention scene, %s data should not be null.", ACTUAL_SEQ_KV_LEN_NAME.c_str()),
                    return ge::GRAPH_FAILED);
            }
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
    const FiaLayout &layout, const std::string &actualSeqLenName, const std::string &attrName)
{
    if (tensor == nullptr) {
        OP_LOGE(opName_, "when layout of %s is %s, %s must be provided.",
            attrName.c_str(), LayoutToSerialString(layout).c_str(), actualSeqLenName.c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t shapeSize = tensor->GetShapeSize();
    if (shapeSize <= 0) {
        OP_LOGE(opName_, "%s shape size is %ld, it should be greater than 0.",
            actualSeqLenName.c_str(), shapeSize);
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

    const int64_t *actualSeq = opParamInfo_.actualSeqLengthsQ.tensor->GetData<int64_t>();
    if (actualSeq == nullptr) {
        qSize.push_back(s1Size_);
        return ge::GRAPH_SUCCESS;
    }

    if (GetActualSeqLenSize(actualSeqLengthsQSize_, opParamInfo_.actualSeqLengthsQ.tensor,
        qLayout_, ACTUAL_SEQ_Q_LEN_NAME, QUERY_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint32_t loop = std::min(actualSeqLengthsQSize_, bSize_);
    for (uint32_t i = 0; i < loop; i++) {
        int64_t tmpS1 = 0;
        if (qLayout_ == FiaLayout::TND || qLayout_ == FiaLayout::NTD) {
            OP_CHECK_IF(actualSeq[i] < 0,
                OP_LOGE(opName_, "when %s's layout is %s, %s[%u] should not be a negative number, but got %ld.",
                    QUERY_NAME.c_str(), LayoutToSerialString(qLayout_).c_str(),
                    ACTUAL_SEQ_Q_LEN_NAME.c_str(), i, actualSeq[i]),
                    return ge::GRAPH_FAILED);

            OP_CHECK_IF(i > 0U && (actualSeq[i] < actualSeq[i - 1U]),
                OP_LOGE(opName_, "when %s's layout is %s, %s[%u](%ld) should not be less than %s[%u](%ld).",
                    QUERY_NAME.c_str(), LayoutToSerialString(qLayout_).c_str(),
                    ACTUAL_SEQ_Q_LEN_NAME.c_str(), i, actualSeq[i],
                    ACTUAL_SEQ_Q_LEN_NAME.c_str(), (i - 1U), actualSeq[i - 1U]),
                    return ge::GRAPH_FAILED);

            tmpS1 = (i == 0U) ? actualSeq[0] : (actualSeq[i] - actualSeq[i - 1U]);
        } else {
            tmpS1 = actualSeq[i];
        }
        if (tmpS1 > static_cast<int64_t>(s1Size_) || tmpS1 < 0) {
            OP_LOGE(opName_,
                "%s[%u] computed is %ld, it should be in range [0, Q_S(%u)].",
                ACTUAL_SEQ_Q_LEN_NAME.c_str(), i, tmpS1, s1Size_);
            return ge::GRAPH_FAILED;
        }
        qSize.push_back(tmpS1);
    }

    OP_CHECK_IF((qLayout_ == FiaLayout::TND) && (qTSize_ != actualSeq[actualSeqLengthsQSize_ - 1]),
        OP_LOGE(opName_, "when %s's layout is %s, T(%u) should be equal to the last element of %s(%ld).",
            QUERY_NAME.c_str(), LayoutToSerialString(qLayout_).c_str(), qTSize_, ACTUAL_SEQ_Q_LEN_NAME.c_str(),
            actualSeq[actualSeqLengthsQSize_ - 1]),
        return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckFeatureActualSeqLensKvData()
{
    if (opParamInfo_.actualSeqLengths.tensor == nullptr) {
        kvSize.push_back(s2Size_);
        return ge::GRAPH_SUCCESS;
    }

    const int64_t *actualSeq = opParamInfo_.actualSeqLengths.tensor->GetData<int64_t>();
    if (actualSeq == nullptr) {
        kvSize.push_back(s2Size_);
        return ge::GRAPH_SUCCESS;
    }

    if(GetActualSeqLenSize(actualSeqLengthsKvSize_, opParamInfo_.actualSeqLengths.tensor,
        kvLayout_, ACTUAL_SEQ_KV_LEN_NAME, KEY_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint32_t loop = std::min(actualSeqLengthsKvSize_, bSize_);
    for (uint32_t i = 0; i < loop; i++) {
        int64_t tmpS2 = 0;
        if (kvLayout_ == FiaLayout::TND || kvLayout_ == FiaLayout::NTD) {
            OP_CHECK_IF(actualSeq[i] < 0,
                OP_LOGE(opName_, "when kv's layout is %s, %s[%u] should not be a negative number, but got %ld.",
                    LayoutToSerialString(kvLayout_).c_str(),
                    ACTUAL_SEQ_KV_LEN_NAME.c_str(), i, actualSeq[i]),
                    return ge::GRAPH_FAILED);

            OP_CHECK_IF(i > 0U && (actualSeq[i] < actualSeq[i - 1U]),
                OP_LOGE(opName_, "when kv's layout is %s, %s[%u](%ld) should not be less than %s[%u](%ld).",
                    LayoutToSerialString(kvLayout_).c_str(),
                    ACTUAL_SEQ_KV_LEN_NAME.c_str(), i, actualSeq[i],
                    ACTUAL_SEQ_KV_LEN_NAME.c_str(), (i - 1U), actualSeq[i - 1U]),
                    return ge::GRAPH_FAILED);

            tmpS2 = (i == 0U) ? actualSeq[0] : (actualSeq[i] - actualSeq[i - 1U]);
        } else {
            tmpS2 = actualSeq[i];
        }

        OP_CHECK_IF(tmpS2 < 0 || tmpS2 > s2Size_,
            OP_LOGE(opName_, "%s(%u) is %ld, it should be in range [0, KV_S(%ld)].",
                ACTUAL_SEQ_KV_LEN_NAME.c_str(), i, tmpS2, s2Size_),
            return ge::GRAPH_FAILED);
        kvSize.push_back(tmpS2);
    }

    OP_CHECK_IF((kvLayout_ == FiaLayout::TND) && (kTSize_ != actualSeq[actualSeqLengthsKvSize_ - 1]),
        OP_LOGE(opName_, "when kv's layout is %s, T(%u) should be equal to the last element of %s(%ld).",
            LayoutToSerialString(kvLayout_).c_str(), kTSize_, ACTUAL_SEQ_KV_LEN_NAME.c_str(),
            actualSeq[actualSeqLengthsKvSize_ - 1]),
        return ge::GRAPH_FAILED);

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
        OP_LOGE(opName_, "input dtype %d with output dtype %d is not currently supported.", static_cast<int32_t>(inputQType_),
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
