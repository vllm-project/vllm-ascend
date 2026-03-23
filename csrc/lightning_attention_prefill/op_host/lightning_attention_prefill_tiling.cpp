/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lightning_attention_prefill_tiling.h"
#include "register/op_impl_registry.h"

namespace optiling
{

static constexpr uint32_t MAX_BASE_M = 128;
static constexpr uint32_t MAX_BATCH_SIZE = 256;
static constexpr uint32_t MAX_AIV_NUM = 50;
static constexpr uint32_t ATTR_BLOCK_SIZE = 0;
static constexpr uint32_t ATTR_ACTUAL_SEQ_LEN_ARRAY = 1;
static constexpr uint32_t HALF_BYTE_SIZE = 2;
static constexpr uint32_t FLOAT_BYTE_SIZE = 4;
static constexpr size_t DIM_2 = 2;
static constexpr size_t DIM_3 = 3;

bool LightningAttentionPrefillTiling::IsCapable()
{
    return true;
}

ge::graphStatus LightningAttentionPrefillTiling::GetPlatformInfo()
{
    aicNum_ = ascendcPlatform_->GetCoreNumAic();
    aivNum_ = ascendcPlatform_->GetCoreNumAiv();
    actualUsedAivNum_ = aivNum_;
    ascendcPlatform_->GetCoreMemSize(platform_ascendc::CoreMemType::UB, aicoreParams_.ubSize);
    ascendcPlatform_->GetCoreMemSize(platform_ascendc::CoreMemType::L1, aicoreParams_.l1Size);
    ascendcPlatform_->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, aicoreParams_.l0cSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LightningAttentionPrefillTiling::GetShapeAttrsInfo()
{
    auto attrs = context_->GetAttrs();
    auto *blockSize = attrs->GetAttrPointer<int32_t>(ATTR_BLOCK_SIZE);
    blockSize_ = *blockSize;
    tilingData_.laBaseParams.set_blockSize(blockSize_);

    auto *seqLenArray = attrs->GetListInt(ATTR_ACTUAL_SEQ_LEN_ARRAY);
    auto qShape = context_->GetInputShape(0)->GetStorageShape();
    uint32_t batchSize = qShape.GetDim(0);
    uint32_t headNum = qShape.GetDim(1);
    uint32_t maxSeqLen = qShape.GetDim(DIM_2);
    std::vector<uint16_t> blockCountPerBatch(MAX_BATCH_SIZE, maxSeqLen / blockSize_);
    std::vector<uint16_t> tailBlockSize(MAX_BATCH_SIZE);
    if (!seqLenArray || seqLenArray->GetSize() != batchSize) {
        return ge::GRAPH_FAILED;
    }
    for (uint32_t index = 0; index < seqLenArray->GetSize(); ++index) {
        tailBlockSize[index] = seqLenArray->GetData()[index] % blockSize_;
        blockCountPerBatch[index] = (seqLenArray->GetData()[index] + blockSize_ - 1) / blockSize_;
        totalBlockCount_ += blockCountPerBatch[index];
    }
    totalBlockCount_ *= headNum;
    tilingData_.laBaseParams.set_tailBlockSize(tailBlockSize.data());
    tilingData_.laBaseParams.set_blockCountPerBatch(blockCountPerBatch.data());

    if (!AnalyzeDType()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LightningAttentionPrefillTiling::DoOpTiling()
{
    auto qShape = context_->GetInputShape(0)->GetStorageShape();
    // set base params
    tilingData_.laBaseParams.set_batchSize(qShape.GetDim(0));
    tilingData_.laBaseParams.set_headNum(qShape.GetDim(1));
    tilingData_.laBaseParams.set_maxSeqLen(qShape.GetDim(DIM_2));
    tilingData_.laBaseParams.set_headDim(qShape.GetDim(DIM_3));
    tilingData_.laBaseParams.set_eleCountPerHead(qShape.GetDim(DIM_2) * qShape.GetDim(DIM_3));
    tilingData_.laBaseParams.set_eleCountPerBlock(blockSize_ * qShape.GetDim(DIM_3));

    qSBlockSize_ = blockSize_;
    kvSBlockSize_ = blockSize_;

    headDimBlock_ = tilingData_.laBaseParams.get_headDim();

    taskNum_ = tilingData_.laBaseParams.get_batchSize() * tilingData_.laBaseParams.get_headNum();
    if (taskNum_ < actualUsedAivNum_) {
        actualUsedAivNum_ = taskNum_;
    }
    tilingData_.laBaseParams.set_actualUsedAivNum(actualUsedAivNum_);

    SetHeadStartEnd();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LightningAttentionPrefillTiling::DoLibApiTiling()
{
    if (!SetMatmulTiling()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t LightningAttentionPrefillTiling::GetTilingKey() const
{
    return 0;
}

ge::graphStatus LightningAttentionPrefillTiling::GetWorkspaceSize()
{
    uint32_t headNum = tilingData_.laBaseParams.get_headNum();
    // workspace reserved for each core
    // - p
    // - oIntra
    // - updatedKey

    auto dataSize = mm1InDType_ == matmul_tiling::DataType::DT_FLOAT ? FLOAT_BYTE_SIZE : HALF_BYTE_SIZE;
    // workspace to store P, which is type float16/bfloat16/float32 with shape BLOCK_SIZE * BLOCK_SIZE
    uint32_t pWorkspaceSize = dataSize * blockSize_ * blockSize_;
    // workspace to store Ointra, which is type float with shape BLOCK_SIZE * HEAD_DIM
    uint32_t oIntraWorkspaceSize = calcTypeSize_ * tilingData_.laBaseParams.get_eleCountPerBlock();
    // workspace to store Ointer/updated Ki, which is type float16/bfloat16/float32 with shape BLOCK_SIZE * HEAD_DIM
    uint32_t updatedKeyWorkspaceSize = calcTypeSize_ * tilingData_.laBaseParams.get_eleCountPerBlock();
    workspaceSize_ += (pWorkspaceSize + oIntraWorkspaceSize + updatedKeyWorkspaceSize) *
            actualUsedAivNum_;

    // workSpace shared by every core
    // - diagDecay, type float with shape (HEAD, BlockSize, BlockSize)
    uint32_t diagDecayWorkspaceSize = headNum * blockSize_ * blockSize_ * calcTypeSize_;
    workspaceSize_ += diagDecayWorkspaceSize;
    // - (qDecay + kDecay + blockDecay) * HEAD
    uint32_t qDecayWorkspaceSize = blockSize_;
    uint32_t kDecayWorkspaceSize = blockSize_;
    uint32_t blockDecayWorkspaceSize = 8;
    workspaceSize_ += headNum * (qDecayWorkspaceSize + kDecayWorkspaceSize + blockDecayWorkspaceSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LightningAttentionPrefillTiling::PostTiling()
{
    auto blockDim = CalcTschBlockDim(actualUsedAivNum_, aicNum_, aivNum_);
    context_->SetBlockDim(blockDim);
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_ + ascendcPlatform_->GetLibApiWorkSpaceSize();
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
    context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}


bool LightningAttentionPrefillTiling::AnalyzeDType()
{
    inputDType_ = context_->GetInputDesc(0)->GetDataType();
    switch (inputDType_) {
        case ge::DT_FLOAT16:
            mm1InDType_ = matmul_tiling::DataType::DT_FLOAT16;
            mm1OutDType_ = matmul_tiling::DataType::DT_FLOAT;
            mm2InDType_ = matmul_tiling::DataType::DT_FLOAT16;
            mm2OutDType_ = matmul_tiling::DataType::DT_FLOAT;
            mm3InDType_ = matmul_tiling::DataType::DT_FLOAT16;
            mm3OutDType_ = matmul_tiling::DataType::DT_FLOAT;
            mm4InDType_ = matmul_tiling::DataType::DT_FLOAT16;
            mm4OutDType_ = matmul_tiling::DataType::DT_FLOAT;
            calcTypeSize_ = ge::GetSizeByDataType(ge::DT_FLOAT);
            break;
        case ge::DT_BF16:
            mm1InDType_ = matmul_tiling::DataType::DT_BF16;
            mm1OutDType_ = matmul_tiling::DataType::DT_FLOAT;
            mm2InDType_ = matmul_tiling::DataType::DT_BF16;
            mm2OutDType_ = matmul_tiling::DataType::DT_FLOAT;
            mm3InDType_ = matmul_tiling::DataType::DT_BF16;
            mm3OutDType_ = matmul_tiling::DataType::DT_FLOAT;
            mm4InDType_ = matmul_tiling::DataType::DT_BF16;
            mm4OutDType_ = matmul_tiling::DataType::DT_FLOAT;
            calcTypeSize_ = ge::GetSizeByDataType(ge::DT_FLOAT);
            break;
        case ge::DT_FLOAT:
            mm1InDType_ = matmul_tiling::DataType::DT_FLOAT;
            mm1OutDType_ = matmul_tiling::DataType::DT_FLOAT;
            mm2InDType_ = matmul_tiling::DataType::DT_FLOAT;
            mm2OutDType_ = matmul_tiling::DataType::DT_FLOAT;
            mm3InDType_ = matmul_tiling::DataType::DT_FLOAT;
            mm3OutDType_ = matmul_tiling::DataType::DT_FLOAT;
            mm4InDType_ = matmul_tiling::DataType::DT_FLOAT;
            mm4OutDType_ = matmul_tiling::DataType::DT_FLOAT;
            calcTypeSize_ = ge::GetSizeByDataType(ge::DT_FLOAT);
            break;
        default:
            return false;
    }
    return true;
}

void LightningAttentionPrefillTiling::SetHeadStartEnd()
{
    uint32_t headStartIdx = 0;
    uint32_t headEndIdx = 0;
    uint32_t totalBlockCount = totalBlockCount_;
    uint32_t blockCountEachCore;
    std::vector<uint16_t> headStart(MAX_AIV_NUM, 0);
    std::vector<uint16_t> headEnd(MAX_AIV_NUM, 0);
    for (uint32_t coreId = 0, currBlockCount, batchId; coreId < actualUsedAivNum_;
          ++coreId, headStartIdx = ++headEndIdx) {
        blockCountEachCore = totalBlockCount / (actualUsedAivNum_ - coreId);
        for (currBlockCount = 0u; taskNum_ - headEndIdx > actualUsedAivNum_ - coreId;) {
            batchId = headEndIdx / tilingData_.laBaseParams.get_headNum();
            currBlockCount += tilingData_.laBaseParams.get_blockCountPerBatch()[batchId];
            if (currBlockCount >= blockCountEachCore) {
                break;
            } else {
                ++headEndIdx;
            }
        }
        totalBlockCount -= currBlockCount;
        headStart[coreId] = (uint16_t)headStartIdx;
        headEnd[coreId] = (uint16_t)headEndIdx;
    }
    tilingData_.laBaseParams.set_headStart(headStart.data());
    tilingData_.laBaseParams.set_headEnd(headEnd.data());
}

bool LightningAttentionPrefillTiling::SetMatmulTiling()
{
    return SetMatmulTilingForQXK() && SetMatmulTilingForPXV() &&
           SetMatmulTilingForQXKV() && SetMatmulTilingForKXV();
}

bool LightningAttentionPrefillTiling::SetMatmulTilingForQXK()
{
    matmul_tiling::MatmulApiTiling mm1(*ascendcPlatform_);
    mm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mm1InDType_, false);
    mm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mm1InDType_, true);
    mm1.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, mm1OutDType_);
    mm1.SetShape(qSBlockSize_, kvSBlockSize_, headDimBlock_);
    mm1.SetOrgShape(qSBlockSize_, kvSBlockSize_, headDimBlock_, headDimBlock_);
    mm1.SetBias(false);
    if (mm1.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
        return false;
    }
    if (mm1.SetFixSplit(std::min(qSBlockSize_, MAX_BASE_M), std::min(kvSBlockSize_, MAX_BASE_M)) != 0) {
        return false;
    }
    if (mm1.GetTiling(tilingData_.mm1TilingData) != 0) {
        return false;
    }
    tilingData_.mm1TilingData.set_stepM(1);
    tilingData_.mm1TilingData.set_stepN(1);
    return true;
}

bool LightningAttentionPrefillTiling::SetMatmulTilingForPXV()
{
    matmul_tiling::MatmulApiTiling mm2(*ascendcPlatform_);
    mm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mm2InDType_, false);
    mm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mm2InDType_, true);
    mm2.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, mm2OutDType_);
    mm2.SetShape(qSBlockSize_, headDimBlock_, kvSBlockSize_);
    mm2.SetOrgShape(qSBlockSize_, headDimBlock_, kvSBlockSize_, kvSBlockSize_);
    mm2.SetBias(false);
    if (mm2.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
        return false;
    }
    if (mm2.SetFixSplit(std::min(qSBlockSize_, MAX_BASE_M), std::min(headDimBlock_, MAX_BASE_M)) != 0) {
        return false;
    }
    if (mm2.GetTiling(tilingData_.mm2TilingData) != 0) {
        return false;
    }
    tilingData_.mm2TilingData.set_stepM(1);
    tilingData_.mm2TilingData.set_stepN(1);
    return true;
}

bool LightningAttentionPrefillTiling::SetMatmulTilingForQXKV()
{
    matmul_tiling::MatmulApiTiling mm3(*ascendcPlatform_);
    mm3.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mm3InDType_, false);
    mm3.SetBType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, mm3InDType_, false);
    mm3.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, mm3OutDType_);
    mm3.SetShape(qSBlockSize_, headDimBlock_, headDimBlock_);
    mm3.SetOrgShape(qSBlockSize_, headDimBlock_, headDimBlock_, headDimBlock_);
    mm3.SetBias(false);
    if (mm3.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
        return false;
    }
    if (mm3.SetFixSplit(std::min(qSBlockSize_, MAX_BASE_M), std::min(headDimBlock_, MAX_BASE_M)) != 0) {
        return false;
    }
    if (mm3.GetTiling(tilingData_.mm3TilingData) != 0) {
        return false;
    }
    tilingData_.mm3TilingData.set_stepM(1);
    tilingData_.mm3TilingData.set_stepN(1);
    return true;
}

bool LightningAttentionPrefillTiling::SetMatmulTilingForKXV()
{
    matmul_tiling::MatmulApiTiling mm4(*ascendcPlatform_);
    mm4.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mm4InDType_, true);
    mm4.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mm4InDType_, false);
    mm4.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, mm4OutDType_);
    mm4.SetShape(headDimBlock_, headDimBlock_, kvSBlockSize_);
    mm4.SetOrgShape(headDimBlock_, headDimBlock_, kvSBlockSize_, kvSBlockSize_);
    mm4.SetBias(false);
    if (mm4.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
        return false;
    }
    if (mm4.SetFixSplit(std::min(headDimBlock_, MAX_BASE_M), std::min(headDimBlock_, MAX_BASE_M)) != 0) {
        return false;
    }
    if (mm4.GetTiling(tilingData_.mm4TilingData) != 0) {
        return false;
    }
    tilingData_.mm4TilingData.set_stepM(1);
    tilingData_.mm4TilingData.set_stepN(1);
    return true;
}


ASCENDC_EXTERN_C ge::graphStatus TilingLightningAttentionPrefill(gert::TilingContext* context)
{
    LightningAttentionPrefillTiling tiling(context);
    return tiling.DoTiling();
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForLightningAttentionPrefill(gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(LightningAttentionPrefill)
    .Tiling(TilingLightningAttentionPrefill)
    .TilingParse<LightningAttentionPrefillCompileInfo>(TilingPrepareForLightningAttentionPrefill);

}