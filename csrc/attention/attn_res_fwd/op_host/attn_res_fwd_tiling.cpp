/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file attn_res_fwd_tiling.cpp
 * \brief AttnResFwd host tiling — CanResidentAllBlocks → RESIDENT / RELOAD；无 FP32-v WS
 */
#include "attn_res_fwd_tiling.h"

#include <algorithm>

#include "tiling_base/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "err/ops_err.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "util/math_util.h"

namespace optiling {
REGISTER_OPS_TILING_TEMPLATE(AttnResFwd, AttnResFwdTiling, 0);

constexpr size_t PREFIX_SUM_INDEX = 0;
constexpr size_t BLOCK_RESIDUAL_INDEX = 1;
constexpr size_t PROJ_WEIGHT_INDEX = 2;
constexpr size_t NORM_WEIGHT_INDEX = 3;
constexpr size_t HIDDEN_STATES_INDEX = 0;
constexpr size_t INV_RMS_INDEX = 1;
constexpr size_t PROBS_INDEX = 2;

constexpr size_t PREFIX_SUM_DIM = 2;
constexpr size_t BLOCK_RESIDUAL_DIM = 3;

constexpr size_t ATTR_NORM_EPS_INDEX = 0;
constexpr size_t ATTR_NEED_BACKWARD_INDEX = 1;

constexpr int64_t SYS_WORKSPACE_SIZE = 16777216;
constexpr uint32_t UB_AVAIL_BYTES = 192U * 1024U; // DAV_2201 按 192KB 估算
// TPipe/对齐/Que 头开销；过小会导致大 H RESIDENT 运行时 507035
constexpr uint32_t UB_OVERHEAD_BYTES = 32U * 1024U;
constexpr uint32_t ELEM_PER_BLK_BF16 = 16U;
constexpr uint32_t ELEM_PER_BLK_FP32 = 8U;
constexpr uint32_t SCALAR_LOCAL_ELEMS = 8U;
constexpr uint32_t STAGING_ALIGN_BYTES = 512U;

static inline uint32_t AlignUpU32(uint32_t val, uint32_t align)
{
    if (align == 0) {
        return val;
    }
    return (val + align - 1U) / align * align;
}

static inline uint32_t AlignDownU32(uint32_t val, uint32_t align)
{
    if (align == 0) {
        return val;
    }
    return (val / align) * align;
}

void AttnResFwdTiling::InitCompileInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platformInfoPtr is null");
        return;
    }
    const auto &ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
    compileInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
    if (compileInfo_.aivNum <= 0) {
        OP_LOGE(context_->GetNodeName(), "aivNum <= 0");
        return;
    }
    tilingData_.usedCoreNum = static_cast<uint32_t>(compileInfo_.aivNum);
}

ge::graphStatus AttnResFwdTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AttnResFwdTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid context."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(AnalyzeDtype() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid dtypes."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(AnalyzeShapes() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid shapes."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetNormEps() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid norm_eps attr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetNeedBackward() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid need_backward attr."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

uint32_t AttnResFwdTiling::GetMinStagingBytes() const
{
    // inv/probs 均经 Que 逐 token 直写 GM，不再占用 staging
    return 0;
}

uint32_t AttnResFwdTiling::CalcMaxResidentRows(uint32_t hiddenSize) const
{
    // UB ≈ B*H*2 + 4*H*4(scoreWeight/vRow/out/brc) + B*4 + 2*H*2(Que) + overhead (+ minStaging)
    // N_ub_max = floor( (UB - 4*H*4 - 2*H*2 - overhead - minStaging) / (H*2 + 4) )
    // 保持保守估计，避免误选 RESIDENT（大 H 下 B 稍大即可能 UB 打满/精度异常）
    if (hiddenSize == 0) {
        return 0;
    }
    const uint64_t workBytes =
        static_cast<uint64_t>(4U) * hiddenSize * sizeof(float) +
        static_cast<uint64_t>(2U) * hiddenSize * sizeof(uint16_t) + UB_OVERHEAD_BYTES +
        static_cast<uint64_t>(GetMinStagingBytes());
    if (workBytes >= UB_AVAIL_BYTES) {
        return 0;
    }
    const uint64_t remain = UB_AVAIL_BYTES - workBytes;
    const uint64_t perRow = static_cast<uint64_t>(hiddenSize) * sizeof(uint16_t) + sizeof(float);
    return static_cast<uint32_t>(remain / perRow);
}

bool AttnResFwdTiling::CanResidentAllBlocks(uint32_t blockCount, uint32_t hiddenSize) const
{
    return blockCount > 0 && blockCount <= CalcMaxResidentRows(hiddenSize);
}

uint64_t AttnResFwdTiling::EstimateUbComputeBytes(bool resident) const
{
    const uint32_t H = tilingData_.hiddenSize;
    const uint32_t B = tilingData_.blockCount;
    const uint32_t HAlignBf16 = AlignUpU32(H, ELEM_PER_BLK_BF16);
    const uint32_t HAlignFp32 = AlignUpU32(H, ELEM_PER_BLK_FP32);
    // 与 arch22 / kernel 一致：meta 按 32B block（8 fp32）对齐
    const uint32_t metaAlign = AlignUpU32(B, ELEM_PER_BLK_FP32);

    uint64_t ub = 0;
    if (resident) {
        ub += static_cast<uint64_t>(1) * HAlignBf16 * sizeof(uint16_t); // inQue
        ub += static_cast<uint64_t>(1) * HAlignBf16 * sizeof(uint16_t); // outQue
        ub += static_cast<uint64_t>(B) * HAlignBf16 * sizeof(uint16_t); // vBf16 resident
    } else {
        ub += static_cast<uint64_t>(2) * HAlignBf16 * sizeof(uint16_t); // inQue BUFFER_NUM=2
        ub += static_cast<uint64_t>(1) * HAlignBf16 * sizeof(uint16_t); // outQue
    }
    ub += static_cast<uint64_t>(3) * HAlignFp32 * sizeof(float); // scoreWeight + vRow + outFp32
    ub += static_cast<uint64_t>(metaAlign) * sizeof(float); // vecMeta
    ub += static_cast<uint64_t>(metaAlign) * sizeof(float); // metaSoftmax
    ub += static_cast<uint64_t>(metaAlign) * ELEM_PER_BLK_FP32 * sizeof(float); // metaBrc[n*8] after Softmax Brcb
    ub += static_cast<uint64_t>(SCALAR_LOCAL_ELEMS) * sizeof(float);
    if (tilingData_.needBackward != 0) {
        ub += static_cast<uint64_t>(ELEM_PER_BLK_FP32) * sizeof(float); // invQue_ 1 block
        ub += static_cast<uint64_t>(metaAlign) * sizeof(float);         // probsQue_ AlignUp(B, 8)
    }
    ub += UB_OVERHEAD_BYTES;
    return ub;
}

ge::graphStatus AttnResFwdTiling::FillStagingFields()
{
    // 无 staging：inv/probs 均 Que 直写
    tilingData_.stagingBytes = 0;
    tilingData_.tokensPerFlush = 0;
    tilingData_.elemsPerToken = (tilingData_.needBackward != 0) ? tilingData_.blockCount : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AttnResFwdTiling::DoOpTiling()
{
    if (tilingData_.numTokens == 0 || tilingData_.hiddenSize == 0) {
        tilingData_.tokensPerCore = 0;
        tilingData_.blockCount = tilingData_.numBlocks + 1U;
        tilingData_.stagingBytes = 0;
        tilingData_.tokensPerFlush = 0;
        tilingData_.elemsPerToken = 0;
        return ge::GRAPH_SUCCESS;
    }

    const uint32_t coreNum = std::min(tilingData_.numTokens, tilingData_.usedCoreNum);
    tilingData_.usedCoreNum = coreNum;
    tilingData_.tokensPerCore = Ops::Base::CeilDiv(tilingData_.numTokens, coreNum);
    tilingData_.blockCount = tilingData_.numBlocks + 1U;
    tilingData_.invHiddenSize =
        (tilingData_.hiddenSize > 0) ? (1.0f / static_cast<float>(tilingData_.hiddenSize)) : 0.0f;
    tilingData_.wsSizePerToken = 0;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AttnResFwdTiling::DoLibApiTiling()
{
    if (CanResidentAllBlocks(tilingData_.blockCount, tilingData_.hiddenSize)) {
        tilingKey_ = TILING_KEY_BF16_RESIDENT;
        OP_LOGI(context_->GetNodeName(),
                "Select TK-RESIDENT (%lu): B=%u H=%u N_ub_max=%u needBackward=%u",
                static_cast<unsigned long>(tilingKey_), tilingData_.blockCount, tilingData_.hiddenSize,
                CalcMaxResidentRows(tilingData_.hiddenSize), tilingData_.needBackward);
    } else {
        tilingKey_ = TILING_KEY_BF16_RELOAD;
        OP_LOGI(context_->GetNodeName(),
                "Select TK-RELOAD (%lu): B=%u H=%u N_ub_max=%u needBackward=%u",
                static_cast<unsigned long>(tilingKey_), tilingData_.blockCount, tilingData_.hiddenSize,
                CalcMaxResidentRows(tilingData_.hiddenSize), tilingData_.needBackward);
    }
    OP_CHECK_IF(FillStagingFields() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "FillStagingFields failed"), return ge::GRAPH_FAILED);
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

uint64_t AttnResFwdTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus AttnResFwdTiling::GetWorkspaceSize()
{
    // 仅系统 16MB 预留，不追加 FP32-v
    workspaceSize_ = static_cast<uint64_t>(SYS_WORKSPACE_SIZE);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AttnResFwdTiling::PostTiling()
{
    context_->SetBlockDim(tilingData_.usedCoreNum);

    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData);
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData->GetData());

    const auto tilingDataSize = sizeof(AttnResFwd::AttnResFwdTilingData);
    OP_CHECK_IF(rawTilingData->GetCapacity() < tilingDataSize,
                OP_LOGE(context_->GetNodeName(), "raw tiling data capacity %zu < size %zu",
                        rawTilingData->GetCapacity(), tilingDataSize),
                return ge::GRAPH_FAILED);

    errno_t ret = memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    rawTilingData->SetDataSize(tilingDataSize);

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "workspaces is null"),
                return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AttnResFwdTiling::CheckContext()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(PREFIX_SUM_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(PREFIX_SUM_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(BLOCK_RESIDUAL_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(BLOCK_RESIDUAL_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(PROJ_WEIGHT_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(PROJ_WEIGHT_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(NORM_WEIGHT_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(NORM_WEIGHT_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputShape(HIDDEN_STATES_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(HIDDEN_STATES_INDEX));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AttnResFwdTiling::AnalyzeDtype()
{
    auto prefixDtype = context_->GetInputDesc(PREFIX_SUM_INDEX)->GetDataType();
    auto blockDtype = context_->GetInputDesc(BLOCK_RESIDUAL_INDEX)->GetDataType();
    auto projDtype = context_->GetInputDesc(PROJ_WEIGHT_INDEX)->GetDataType();
    auto normDtype = context_->GetInputDesc(NORM_WEIGHT_INDEX)->GetDataType();
    auto outDtype = context_->GetOutputDesc(HIDDEN_STATES_INDEX)->GetDataType();

    OP_CHECK_IF(prefixDtype != ge::DT_BF16, OP_LOGE(context_->GetNodeName(), "prefix_sum dtype must be BF16"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(blockDtype != ge::DT_BF16, OP_LOGE(context_->GetNodeName(), "block_residual dtype must be BF16"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(projDtype != ge::DT_BF16, OP_LOGE(context_->GetNodeName(), "proj_weight dtype must be BF16"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(normDtype != ge::DT_BF16, OP_LOGE(context_->GetNodeName(), "norm_weight dtype must be BF16"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(outDtype != ge::DT_BF16, OP_LOGE(context_->GetNodeName(), "hidden_states dtype must be BF16"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AttnResFwdTiling::AnalyzeShapes()
{
    const auto &prefixShape = context_->GetInputShape(PREFIX_SUM_INDEX)->GetOriginShape();
    const auto &blockShape = context_->GetInputShape(BLOCK_RESIDUAL_INDEX)->GetOriginShape();
    const auto &projShape = context_->GetInputShape(PROJ_WEIGHT_INDEX)->GetOriginShape();
    const auto &normShape = context_->GetInputShape(NORM_WEIGHT_INDEX)->GetOriginShape();

    OP_CHECK_IF(prefixShape.GetDimNum() != PREFIX_SUM_DIM,
                OP_LOGE(context_->GetNodeName(), "prefix_sum dim num must be %zu", PREFIX_SUM_DIM),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(blockShape.GetDimNum() != BLOCK_RESIDUAL_DIM,
                OP_LOGE(context_->GetNodeName(), "block_residual dim num must be %zu", BLOCK_RESIDUAL_DIM),
                return ge::GRAPH_FAILED);

    tilingData_.numTokens = static_cast<uint32_t>(prefixShape.GetDim(0));
    tilingData_.hiddenSize = static_cast<uint32_t>(prefixShape.GetDim(1));
    tilingData_.numBlocks = static_cast<uint32_t>(blockShape.GetDim(1));

    OP_CHECK_IF(blockShape.GetDim(0) != static_cast<int64_t>(tilingData_.numTokens),
                OP_LOGE(context_->GetNodeName(), "T mismatch between prefix_sum and block_residual"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(blockShape.GetDim(2) != static_cast<int64_t>(tilingData_.hiddenSize),
                OP_LOGE(context_->GetNodeName(), "H mismatch between prefix_sum and block_residual"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingData_.numBlocks < 1,
                OP_LOGE(context_->GetNodeName(), "num_blocks must be >= 1, got %u", tilingData_.numBlocks),
                return ge::GRAPH_FAILED);

    int64_t projHidden = projShape.GetDim(projShape.GetDimNum() - 1);
    OP_CHECK_IF(projHidden != static_cast<int64_t>(tilingData_.hiddenSize),
                OP_LOGE(context_->GetNodeName(), "proj_weight hidden dim mismatch"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(normShape.GetDimNum() != 1 || normShape.GetDim(0) != static_cast<int64_t>(tilingData_.hiddenSize),
                OP_LOGE(context_->GetNodeName(), "norm_weight shape must be [H]"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AttnResFwdTiling::GetNormEps()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs is null"), return ge::GRAPH_FAILED);
    const float *normEpsPtr = attrs->GetAttrPointer<float>(ATTR_NORM_EPS_INDEX);
    tilingData_.normEps = (normEpsPtr != nullptr) ? *normEpsPtr : 1e-5f;
    OP_CHECK_IF(tilingData_.normEps <= 0.0f, OP_LOGE(context_->GetNodeName(), "norm_eps must be > 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AttnResFwdTiling::GetNeedBackward()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs is null"), return ge::GRAPH_FAILED);
    const bool *needBackwardPtr = attrs->GetAttrPointer<bool>(ATTR_NEED_BACKWARD_INDEX);
    const bool needBackward = (needBackwardPtr != nullptr) ? *needBackwardPtr : false;
    tilingData_.needBackward = needBackward ? 1U : 0U;

    if (!needBackward) {
        return ge::GRAPH_SUCCESS;
    }

    auto invRmsDesc = context_->GetOutputDesc(INV_RMS_INDEX);
    auto probsDesc = context_->GetOutputDesc(PROBS_INDEX);
    auto invRmsShape = context_->GetOutputShape(INV_RMS_INDEX);
    auto probsShape = context_->GetOutputShape(PROBS_INDEX);
    OP_CHECK_IF(invRmsDesc == nullptr || probsDesc == nullptr || invRmsShape == nullptr || probsShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "need_backward=true requires inv_rms and probs outputs"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(invRmsDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE(context_->GetNodeName(), "inv_rms dtype must be FLOAT"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(probsDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE(context_->GetNodeName(), "probs dtype must be FLOAT"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void AttnResFwdTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(), "numTokens=%u numBlocks=%u hiddenSize=%u tokensPerCore=%u usedCoreNum=%u",
            tilingData_.numTokens, tilingData_.numBlocks, tilingData_.hiddenSize, tilingData_.tokensPerCore,
            tilingData_.usedCoreNum);
    OP_LOGD(context_->GetNodeName(),
            "blockCount=%u wsSizePerToken=%lu normEps=%f tilingKey=%lu N_ub_max=%u "
            "needBackward=%u stagingBytes=%u tokensPerFlush=%u elemsPerToken=%u",
            tilingData_.blockCount, static_cast<unsigned long>(tilingData_.wsSizePerToken), tilingData_.normEps,
            static_cast<unsigned long>(tilingKey_), CalcMaxResidentRows(tilingData_.hiddenSize),
            tilingData_.needBackward, tilingData_.stagingBytes, tilingData_.tokensPerFlush,
            tilingData_.elemsPerToken);
}

static ge::graphStatus AttnResFwdTilingFunc(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("AttnResFwd", "context is null"),
                return ge::GRAPH_FAILED);
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForAttnResFwd(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("AttnResFwd", "context is null"),
                return ge::GRAPH_FAILED);
    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AttnResFwd)
    .Tiling(AttnResFwdTilingFunc)
    .TilingParse<AttnResFwdCompileInfo>(TilingPrepareForAttnResFwd);
} // namespace optiling
