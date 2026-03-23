/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lightning_attention_decode_tiling.h"
#include "register/op_impl_registry.h"

namespace optiling {

static constexpr uint32_t MAX_BASE_M = 128;
static constexpr size_t DIM_3 = 3;

bool LightningAttentionDecodeTiling::IsCapable()
{
    return true;
}

ge::graphStatus LightningAttentionDecodeTiling::GetPlatformInfo()
{
    aicNum_ = ascendcPlatform_->GetCoreNumAic();
    aivNum_ = ascendcPlatform_->GetCoreNumAiv();
    actualUsedAivNum_ = aivNum_;
    ascendcPlatform_->GetCoreMemSize(platform_ascendc::CoreMemType::UB, aicoreParams_.ubSize);
    ascendcPlatform_->GetCoreMemSize(platform_ascendc::CoreMemType::L1, aicoreParams_.l1Size);
    ascendcPlatform_->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, aicoreParams_.l0cSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LightningAttentionDecodeTiling::GetShapeAttrsInfo()
{
    if (!AnalyzeDType()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LightningAttentionDecodeTiling::DoOpTiling()
{
    auto qShape = context_->GetInputShape(0)->GetStorageShape();
    auto kvCacheShape = context_->GetInputShape(4)->GetStorageShape();
    // set base params
    tilingData_.laBaseParams.set_batchSize(qShape.GetDim(0));
    tilingData_.laBaseParams.set_kvCacheBatchSize(kvCacheShape.GetDim(0));
    tilingData_.laBaseParams.set_headNum(qShape.GetDim(1));
    headDimBlock_ = qShape.GetDim(DIM_3);
    tilingData_.laBaseParams.set_headDim(headDimBlock_);

    taskNum_ = tilingData_.laBaseParams.get_batchSize() * tilingData_.laBaseParams.get_headNum();
    if (taskNum_ < actualUsedAivNum_) {
        actualUsedAivNum_ = taskNum_;
    }
    tilingData_.laBaseParams.set_actualUsedAivNum(actualUsedAivNum_);
    tilingData_.laBaseParams.set_taskNum(taskNum_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LightningAttentionDecodeTiling::DoLibApiTiling()
{
    if (UseMatmul() && !SetMatmulTiling()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t LightningAttentionDecodeTiling::GetTilingKey() const
{
    return 0;
}

ge::graphStatus LightningAttentionDecodeTiling::GetWorkspaceSize()
{
    workspaceSize_ = 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LightningAttentionDecodeTiling::PostTiling()
{
    if (UseMatmul()) {
        auto blockDim = CalcTschBlockDim(actualUsedAivNum_, aicNum_, aivNum_);
        context_->SetBlockDim(blockDim);
    } else {
        context_->SetBlockDim(actualUsedAivNum_);
    }
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_ + ascendcPlatform_->GetLibApiWorkSpaceSize();
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                             context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

bool LightningAttentionDecodeTiling::UseMatmul() const {
    return mm1InDType_ == matmul_tiling::DataType::DT_FLOAT;
}

bool LightningAttentionDecodeTiling::AnalyzeDType()
{
    inputDType_ = context_->GetInputDesc(0)->GetDataType();
    switch (inputDType_) {
        case ge::DT_FLOAT16:
            mm1InDType_ = matmul_tiling::DataType::DT_FLOAT16;
            mm1OutDType_ = matmul_tiling::DataType::DT_FLOAT16;
            break;
        case ge::DT_BF16:
            mm1InDType_ = matmul_tiling::DataType::DT_BF16;
            mm1OutDType_ = matmul_tiling::DataType::DT_BF16;
            break;
        case ge::DT_FLOAT:
            mm1InDType_ = matmul_tiling::DataType::DT_FLOAT;
            mm1OutDType_ = matmul_tiling::DataType::DT_FLOAT;
            break;
        default:
            return false;
    }
    return true;
}

bool LightningAttentionDecodeTiling::SetMatmulTiling()
{
    matmul_tiling::MatmulApiTiling mm1(*ascendcPlatform_);
    mm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mm1InDType_, false);
    mm1.SetBType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, mm1InDType_, false);
    mm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mm1OutDType_);
    mm1.SetShape(1, headDimBlock_, headDimBlock_);
    mm1.SetOrgShape(1, headDimBlock_, headDimBlock_, headDimBlock_);
    mm1.SetBias(false);
    if (mm1.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
        return false;
    }
    if (mm1.SetFixSplit(-1, std::min(headDimBlock_, MAX_BASE_M)) != 0) {
        return false;
    }
    if (mm1.GetTiling(tilingData_.mm1TilingData) != 0) {
        return false;
    }
    return true;
}

ASCENDC_EXTERN_C ge::graphStatus TilingLightningAttentionDecode(gert::TilingContext* context)
{
    LightningAttentionDecodeTiling tiling(context);
    return tiling.DoTiling();
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForLightningAttentionDecode(gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(LightningAttentionDecode)
    .Tiling(TilingLightningAttentionDecode)
    .TilingParse<LightningAttentionDecodeCompileInfo>(TilingPrepareForLightningAttentionDecode);

}
