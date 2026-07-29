/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file custom_fused_infer_attention_v310_tiling.cc
 * \brief
 */

#include "custom_fused_infer_attention_v310_tiling.h"
#include "custom_fused_infer_attention_v310_tiling_base.h"
#include <vector>
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "error/ops_error.h"

using namespace ge;
namespace optiling {

constexpr uint32_t ATB_INNER_PRECISE = 2;
constexpr uint32_t DIM_NUM_ONE = 1;
constexpr uint32_t DIM_NUM_TWO = 2;

// FP16 BSND paged-attention key: mode=3, layout=BSND(0), q=FP16(0), kv=FP16(0), out=FP16(0), origin=0, pa=2
constexpr uint64_t IFA_TILINGKEY_BSND_FP16 = 30000000000200000UL;
// FP16 TND paged-attention key: mode=3, layout=TND(1), q=FP16(0), kv=FP16(0), out=FP16(0), origin=0, pa=2
constexpr uint64_t IFA_TILINGKEY_TND_FP16  = 30000000000200001UL;

ge::graphStatus CustomFIATiling::GenTilingKey()
{
    switch (inputLayout_) {
        case IfaLayout::BSND:
            context_->tilingKey = IFA_TILINGKEY_BSND_FP16;
            break;
        case IfaLayout::TND:
            context_->tilingKey = IFA_TILINGKEY_TND_FP16;
            break;
        default:
            OPS_LOG_E(context_->opName, "not support inputLayout %u", inputLayout_);
            return ge::GRAPH_FAILED;
    }

    OPS_LOG_I(context_->opName, "IFA tilingKey: %lu.", context_->tilingKey);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CustomFIATiling::ConvertContext(gert::TilingContext &context, IncreFlashAttentionContext &ifaContext)
{
    if (context.GetNodeName() == nullptr) {
        OPS_LOG_E("CustomFusedInferAttentionV310", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    ifaContext.opName = context.GetNodeName();
    ifaContext.platformInfo = context.GetPlatformInfo();
    ifaContext.query.desc = context.GetInputDesc(QUERY_INPUT_INDEX);
    ifaContext.query.shape = context.GetInputShape(QUERY_INPUT_INDEX);
    ifaContext.key.desc = context.GetInputDesc(KEY_INPUT_INDEX);
    ifaContext.key.shape = context.GetInputShape(KEY_INPUT_INDEX);
    OPS_ERR_IF((ifaContext.query.shape == nullptr) || (ifaContext.key.shape == nullptr),
               OPS_LOG_E(context.GetNodeName(), "shape of query or shape of key is null."), return ge::GRAPH_FAILED);
    auto batchOfQuery = ifaContext.query.shape->GetStorageShape().GetDim(0);
    auto batchOfKey = ifaContext.key.shape->GetStorageShape().GetDim(0);
    if (batchOfQuery != batchOfKey) {
        ifaContext.kCache.resize(batchOfQuery);
        ifaContext.vCache.resize(batchOfQuery);
        for (int64_t size = 0; size < batchOfQuery; ++size) {
            ifaContext.kCache[size] =
                const_cast<gert::StorageShape *>(context.GetDynamicInputShape(KEY_INPUT_INDEX, size));
            ifaContext.vCache[size] =
                const_cast<gert::StorageShape *>(context.GetDynamicInputShape(VALUE_INPUT_INDEX, size));
        }
    } else {
        ifaContext.kCache.resize(1);
        ifaContext.vCache.resize(1);
        ifaContext.kCache[0] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(KEY_INPUT_INDEX, 0));
        ifaContext.vCache[0] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(VALUE_INPUT_INDEX, 0));
    }

    ifaContext.value.desc = context.GetInputDesc(VALUE_INPUT_INDEX);
    ifaContext.value.shape = context.GetInputShape(VALUE_INPUT_INDEX);
    ifaContext.attnMask.desc = context.GetOptionalInputDesc(ATTN_MASK_INPUT_INDEX);
    ifaContext.attnMask.tensor = context.GetOptionalInputTensor(ATTN_MASK_INPUT_INDEX);
    ifaContext.attenOut.desc = context.GetOutputDesc(OUTPUT_INDEX);
    ifaContext.attenOut.shape = context.GetOutputShape(OUTPUT_INDEX);
    ifaContext.actualSeqLengthsQ.tensor = context.GetOptionalInputTensor(ACT_SEQ_LEN_Q_INPUT_INDEX);
    ifaContext.actualSeqLengths.tensor = context.GetOptionalInputTensor(ACT_SEQ_LEN_INPUT_INDEX);
    ifaContext.blockTable.tensor = context.GetOptionalInputTensor(BLOCK_TABLE_INPUT_INDEX);
    ifaContext.blockTable.desc = context.GetOptionalInputDesc(BLOCK_TABLE_INPUT_INDEX);

    auto attrs = context.GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_LOG_E(context.GetNodeName(), "attrs got from GE is nullptr"),
               return ge::GRAPH_FAILED);

    ifaContext.numHeads = attrs->GetAttrPointer<uint32_t>(NUM_HEADS_ATTR_INDEX);
    ifaContext.scaleValue = attrs->GetAttrPointer<float>(SCALE_VALUE_ATTR_INDEX);
    ifaContext.layOut = attrs->GetStr(LAYOUT_ATTR_INDEX);
    ifaContext.kvHeadNums = attrs->GetAttrPointer<uint32_t>(KV_NUM_HEADS_ATTR_INDEX);
    ifaContext.blockSize = attrs->GetAttrPointer<uint32_t>(BLOCK_SIZE_ATTR_INDEX);
    ifaContext.innerPrecise = attrs->GetAttrPointer<uint32_t>(INNER_PRECISE_ATTR_INDEX);

    OPS_ERR_IF(context.GetWorkspaceSizes(1) == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "workSpaceSize got from GE is nullptr"),
               return ge::GRAPH_FAILED);
    ifaContext.workSpaces = context.GetWorkspaceSizes(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CustomFIATiling::RunCustomFIATiling(IncreFlashAttentionContext &context)
{
    this->context_ = &context;

    // Step 1: Initialize hardware-platform information.
    if (InitPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Step 2: Check whether the custom operator can handle this input.
    if (CheckIfShouldRunCustomFIA()) {
        return CustomFIATilingProcess();
    }

    return ge::GRAPH_FAILED;
}

ge::graphStatus CustomFIATiling::InitPlatformInfo()
{
    OPS_ERR_IF(context_->platformInfo == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR(context_->opName, "GetPlatformInfo is nullptr."),
               return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->platformInfo);

#ifdef ASCENDC_OP_TEST
    constexpr uint32_t testCoreNum = 8;
    libapiSize_ = 2 * 1024 * 1024;
    context_->blockDim = testCoreNum;
#else
    if (ascendcPlatform.GetSocVersion() != platform_ascendc::SocVersion::ASCEND310P) {
        OPS_LOG_E(context_->opName, "Only ASCEND310P is supported.");
        return ge::GRAPH_FAILED;
    }
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    context_->blockDim = ascendcPlatform.CalcTschBlockDim(aivNum_, aicNum_, aivNum_);
#endif

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CustomFIATiling::ParseTilingAttributes()
{
    // 1. Parse basic attributes.
    numHeads_ = *context_->numHeads;
    numKvHeads_ = (*context_->kvHeadNums == 0) ? numHeads_ : *context_->kvHeadNums;
    blockSize_ = *context_->blockSize;
    
    // 2. Parse data types.
    inputQType_ = context_->query.desc->GetDataType();
    inputKvType_ = context_->key.desc->GetDataType();
    
    // 3. Parse the batch size.
    batchSize_ = static_cast<uint32_t>(context_->blockTable.tensor->GetStorageShape().GetDim(0));

    // 4. Parse the head dimension.
    const auto& qShape = context_->query.shape->GetStorageShape();
    const auto dimNum = qShape.GetDimNum();
    if (dimNum > 0) {
        headDim_ = static_cast<uint32_t>(qShape.GetDim(dimNum - 1));
    } else {
        OPS_LOG_E(context_->opName, "query shape is empty or scalar!");
        return ge::GRAPH_FAILED;
    }

    // 5. Parse the query layout.
    const std::string layout(context_->layOut);
    if (layout == "BSND") {
        inputLayout_ = IfaLayout::BSND;
    } else if (layout == "TND") {
        inputLayout_ = IfaLayout::TND;
    } else {
        OPS_LOG_E(context_->opName, "unsupported layout: %s", layout.c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

bool CustomFIATiling::CheckIfShouldRunCustomFIA()
{
#ifdef ASCENDC_OP_TEST
    return true;
#endif

    bool isPrecise = (*context_->innerPrecise == ATB_INNER_PRECISE);

    if (!isPrecise) {
        OPS_LOG_E(context_->opName, "Only innerPrecise=2 is supported.");
        return false;
    }
    if (context_->blockTable.tensor == nullptr) {
        OPS_LOG_E(context_->opName, "Only paged attention is supported.");
        return false;
    }

    // Check required inputs for null pointers.
    if (CheckBaseInputsNull() != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context_->opName, "Base inputs check failed.");
        return false;
    }

    // Parse operator attributes and initialize the base state required by tiling.
    if (ParseTilingAttributes() != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context_->opName, "Failed to parse tiling attributes.");
        return false;
    }

    // Check 4: Enforce supported features, inputs, and data types.
    if (CheckInputFormatAndLimits() != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context_->opName, "Input format and limits check failed.");
        return false;
    }

    return true;
}

ge::graphStatus CustomFIATiling::IncreFlashAttentionSetTilingData(gert::TilingContext &context)
{
    OPS_ERR_IF(context.GetRawTilingData() == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "RawTilingData got from GE context is nullptr."),
               return GRAPH_FAILED);

    ifaTilingAtbData.SaveToBuffer(context.GetRawTilingData()->GetData(), context.GetRawTilingData()->GetCapacity());
    context.GetRawTilingData()->SetDataSize(ifaTilingAtbData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingCustomFIAAdapter(gert::TilingContext *context, IncreFlashAttentionContext &ifaContext)
{
    CustomFIATiling ifaTilingNew;
    if (ifaTilingNew.RunCustomFIATiling(ifaContext) == ge::SUCCESS) {
        context->SetTilingKey(ifaContext.tilingKey);
        context->SetBlockDim(ifaContext.blockDim);
        ifaTilingNew.IncreFlashAttentionSetTilingData(*context);
        return ge::GRAPH_SUCCESS;
    }

    return ge::GRAPH_FAILED;
}

void CustomFIATiling::ParseMask()
{
    // Enable the compressed mask by default; use MASK_NORM for debugging.
    attenMaskFlag_ = (context_->attnMask.tensor != nullptr) ? IfaMaskType::MASK_COMPRESS : IfaMaskType::NO_MASK;
    if (attenMaskFlag_) {
        OPS_LOG_D(context_->opName, "attenMaskFlag_:%d", attenMaskFlag_);
        auto maskShape = context_->attnMask.tensor;
        auto maxPromptLen = maskShape->GetStorageShape().GetDim(1);
        auto maskDimZero = maskShape->GetStorageShape().GetDim(0);
        maskKvLen_ = maskShape->GetStorageShape().GetDim(DIM_NUM_TWO);
        maskBatchStride_ =
            (maskDimZero == static_cast<int64_t>(batchSize_))
                ? static_cast<uint32_t>(maskKvLen_ * maxPromptLen)
                : 0;
    }
}

ge::graphStatus CustomFIATiling::ParseTndVarlenParams(const gert::Shape& qShape)
{
    if (qShape.GetDimNum() < 1) {
        OPS_LOG_E(context_->opName, "invalid query dim num for TND");
        return ge::GRAPH_FAILED;
    }

    tSeqSize_ = static_cast<uint32_t>(qShape.GetDim(0));

    OPS_ERR_IF(batchSize_ == 0,
        OPS_REPORT_VECTOR_INNER_ERR(context_->opName, "batchSize is zero in TND mode"),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(context_->actualSeqLengthsQ.tensor == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context_->opName, "actualSeqLengthsQ is required in TND varlen concat mode"),
        return ge::GRAPH_FAILED);

    const int64_t *actualLenDataQ = context_->actualSeqLengthsQ.tensor->GetData<int64_t>();
    OPS_ERR_IF(actualLenDataQ == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context_->opName, "actualSeqLengthsQ data is nullptr in TND mode"),
        return ge::GRAPH_FAILED);

    uint32_t actualLenQDims = static_cast<uint32_t>(context_->actualSeqLengthsQ.tensor->GetShapeSize());
    OPS_ERR_IF(actualLenQDims != batchSize_,
        OPS_REPORT_VECTOR_INNER_ERR(context_->opName, "actualSeqLengthsQ size must equal batchSize in TND mode"),
        return ge::GRAPH_FAILED);

    tndQSeqLens_.resize(batchSize_);

    uint32_t totalQTokens = 0;
    for (uint32_t b = 0; b < batchSize_; ++b) {
        uint32_t seqLen = static_cast<uint32_t>(actualLenDataQ[b]);
        tndQSeqLens_[b] = seqLen;
        totalQTokens += seqLen;
    }

    OPS_ERR_IF(totalQTokens != tSeqSize_,
        OPS_REPORT_VECTOR_INNER_ERR(context_->opName, "TND query first dim must equal sum(actualSeqLengthsQ)"),
        return ge::GRAPH_FAILED);

    qTokens_ = 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CustomFIATiling::ParsePagedAttentionParams()
{
    if (context_->kCache.empty()) {
        OPS_LOG_E(context_->opName, "The Key cache is empty in pa situation.");
        return ge::GRAPH_FAILED;
    }

    maxBlockNumPerBatch_ = static_cast<uint32_t>(context_->blockTable.tensor->GetStorageShape().GetDim(1));
    totalBlockNum_ = static_cast<uint32_t>(context_->kCache[0]->GetStorageShape().GetDim(0));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CustomFIATiling::CustomFIAParamGet()
{
    // 1. Read basic attributes.
    scaleValue_ = *context_->scaleValue;
    if (headDim_ == 256) {
        seqStepQ_ = DEFAULT_QUERY_SEQ_STEP_HEAD_DIM_256;
    } else {
        seqStepQ_ = DEFAULT_QUERY_SEQ_STEP_HEAD_DIM_128_LESS;
    }

    const auto &qShape = context_->query.shape->GetStorageShape();

    // 2. Parse and strictly validate layout-specific attributes.
    if (inputLayout_ == IfaLayout::TND) {
        if (ParseTndVarlenParams(qShape) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else if (inputLayout_ == IfaLayout::BSND) {
        // In BSND layout, the S dimension is at index 1.
        qTokens_ = static_cast<uint32_t>(qShape.GetDim(DIM_NUM_ONE));
    }

    // 3. Parse paged-attention KV-cache attributes.
    if (ParsePagedAttentionParams() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 4. Parse mask attributes.
    ParseMask();
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CustomFIATiling::CustomFIAParamSet()
{
    tilingDataBase_->set_batchSize(batchSize_);
    tilingDataBase_->set_headSize(headDim_);
    tilingDataBase_->set_qHeadNum(numHeads_);
    tilingDataBase_->set_kvHeadNum(numKvHeads_);
    tilingDataBase_->set_totalBlockNum(totalBlockNum_);
    tilingDataBase_->set_scaleValue(scaleValue_);
    tilingDataBase_->set_querySeqStep(seqStepQ_);
    tilingDataCore_->set_qTokens(qTokens_);
    tilingDataBase_->set_attenMaskFlag(attenMaskFlag_);
    tilingDataCore_->set_maskHeadStride(0);
    tilingDataCore_->set_maskBatchStride(maskBatchStride_);
    tilingDataCore_->set_maskKvLen(maskKvLen_);
    tilingDataBase_->set_blockSize(blockSize_);
    tilingDataBase_->set_maxBlockNumPerBatch(maxBlockNumPerBatch_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CustomFIATiling::CustomFIASplitBlock()
{
    const uint32_t taskNum = GetTotalQTaskNum();
    context_->blockDim = taskNum < context_->blockDim ? taskNum : context_->blockDim;
    OPS_ERR_IF(context_->blockDim == 0, OPS_LOG_E(context_->opName, "block dim is zero."), return ge::GRAPH_FAILED);

    const uint32_t taskNumPerCore = taskNum / context_->blockDim;
    const uint32_t tailTaskNum = taskNum % context_->blockDim;
    uint32_t taskStart = 0;
    uint32_t taskEnd = 0;
    std::vector<uint32_t> startTaskId(MAX_CORE_NUM, 0U);
    std::vector<uint32_t> endTaskId(MAX_CORE_NUM, 0U);
    std::vector<uint32_t> startBatch(MAX_CORE_NUM, 0U);
    std::vector<uint32_t> endBatch(MAX_CORE_NUM, 0U);

    auto FindBatchByTaskId = [this](uint32_t taskId) -> uint32_t {
        uint32_t acc = 0;
        for (uint32_t b = 0; b < batchSize_; ++b) {
            uint32_t qBlkNum = (tndQSeqLens_[b] + seqStepQ_ - 1) / seqStepQ_;
            uint32_t batchTaskNum = qBlkNum * numHeads_;
            if (taskId < acc + batchTaskNum) {
                return b;
            }
            acc += batchTaskNum;
        }
        return batchSize_ == 0 ? 0 : (batchSize_ - 1);
    };

    for (uint32_t blockIdx = 0; blockIdx < context_->blockDim; blockIdx++) {
        taskStart = taskEnd;
        taskEnd = blockIdx < tailTaskNum ? taskEnd + taskNumPerCore + 1 : taskEnd + taskNumPerCore;
        startTaskId[blockIdx] = taskStart;
        endTaskId[blockIdx] = taskEnd;

        if (inputLayout_ == IfaLayout::TND) {
            startBatch[blockIdx] = FindBatchByTaskId(taskStart);
            endBatch[blockIdx] = (taskEnd == 0) ? 0 : FindBatchByTaskId(taskEnd - 1);
        } else {
            startBatch[blockIdx] = static_cast<uint32_t>(taskStart / numHeads_);
            endBatch[blockIdx] = static_cast<uint32_t>((taskEnd - 1) / numHeads_);
        }
    }

    tilingDataCore_->set_startTaskId(startTaskId.data());
    tilingDataCore_->set_endTaskId(endTaskId.data());
    tilingDataCore_->set_startBatch(startBatch.data());
    tilingDataCore_->set_endBatch(endBatch.data());
    return ge::GRAPH_SUCCESS;
}

uint32_t CustomFIATiling::GetTotalQTaskNum()
{
    OPS_ERR_IF(context_->actualSeqLengthsQ.tensor == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context_->opName,
            "actualSeqLengthsQ is required"),
        return 0);

    uint32_t actualLenQDims = static_cast<uint32_t>(context_->actualSeqLengthsQ.tensor->GetShapeSize());
    const int64_t *actualLenDataQ = context_->actualSeqLengthsQ.tensor->GetData<int64_t>();
    if (actualLenDataQ == nullptr) {
        return 0;
    }

    uint32_t totalQblockSum = 0;
    for (uint32_t bIdx = 0; bIdx < actualLenQDims; ++bIdx) {
        uint32_t curSeqLenQ = static_cast<uint32_t>(actualLenDataQ[bIdx]);
        uint32_t tmpBlkNum = (curSeqLenQ + seqStepQ_ - 1) / seqStepQ_;
        totalQblockSum += tmpBlkNum;
    }

    return totalQblockSum * numHeads_;
}

ge::graphStatus CustomFIATiling::CustomFIATilingProcess()
{
    this->tilingDataBase_ = &ifaTilingAtbData.tilingBase;
    this->tilingDataCore_ = &ifaTilingAtbData.tilingPerCore;

    if (CustomFIAParamGet() != ge::SUCCESS ||
        CustomFIAParamSet() != ge::SUCCESS ||
        CustomFIASplitBlock() != ge::SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    size_t workspaceSize = libapiSize_;
    if (context_->workSpaces) {
        context_->workSpaces[0] = workspaceSize;
    }
    OPS_LOG_D(context_->opName, "IFA block dim:%u aivNum:%u aicNum:%u", context_->blockDim, aivNum_, aicNum_);
    OPS_LOG_D(context_->opName, "batchSize_:%d", batchSize_);

    OPS_LOG_D(context_->opName, "headDim_:%d", headDim_);
    OPS_LOG_D(context_->opName, "numHeads_:%d", numHeads_);
    OPS_LOG_D(context_->opName, "numKvHeads_:%d", numKvHeads_);
    OPS_LOG_D(context_->opName, "maxBlockNumPerBatch_:%d", maxBlockNumPerBatch_);
    OPS_LOG_D(context_->opName, "totalBlockNum_:%d", totalBlockNum_);
    OPS_LOG_D(context_->opName, "scaleValue_:%lf", scaleValue_);

    return GenTilingKey();
}

IFA_EXTERN_C ge::graphStatus TilingIncreFlashAttention(gert::TilingContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("CustomFusedInferAttentionV310", "Context is nullptr."),
               return ge::GRAPH_FAILED);
    IncreFlashAttentionContext ifaContext{.opName = nullptr,
                                          .platformInfo = nullptr,
                                          .query = {nullptr, nullptr},
                                          .key = {nullptr, nullptr},
                                          .value = {nullptr, nullptr},
                                          .attnMask = {nullptr, nullptr},
                                          .actualSeqLengthsQ = {nullptr, nullptr},
                                          .actualSeqLengths = {nullptr, nullptr},
                                          .blockTable = {nullptr, nullptr},
                                          .attenOut = {nullptr, nullptr},
                                          .numHeads = nullptr,
                                          .scaleValue = nullptr,
                                          .kvHeadNums = nullptr,
                                          .layOut = nullptr,
                                          .blockSize = nullptr,
                                          .innerPrecise = nullptr,
                                          .workSpaces = nullptr,
                                          .kCache = {nullptr},
                                          .vCache = {nullptr},
                                          .tilingKey = 0,
                                          .blockDim = 0};
    if (CustomFIATiling::ConvertContext(*context, ifaContext) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context->GetNodeName(), "Error occurred while converting tilingContext to ifa context");
        return ge::GRAPH_FAILED;
    }
    return TilingCustomFIAAdapter(context, ifaContext);
}
} // namespace optiling
