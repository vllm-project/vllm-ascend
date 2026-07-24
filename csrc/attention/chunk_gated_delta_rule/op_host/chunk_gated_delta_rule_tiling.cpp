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
 * \file chunk_gated_delta_rule_tiling.cpp
 * \brief
 */

#include "chunk_gated_delta_rule_tiling.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "err/ops_err.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/chunk_gated_delta_rule_tiling_key.h"

namespace optiling {
REGISTER_OPS_TILING_TEMPLATE(ChunkGatedDeltaRule, ChunkGatedDeltaRuleTiling, 0);

const size_t QUERY_INDEX = 0;
const size_t KEY_INDEX = 1;
const size_t VALUE_INDEX = 2;
const size_t BETA_INDEX = 3;
const size_t STATE_INDEX = 4;
const size_t ACTUAL_SEQ_LENGTHS_INDEX = 5;
const size_t G_INDEX = 6;

const size_t OUTPUT_OUT_IDX = 0;
const size_t OUTPUT_FINAL_STATE_IDX = 1;

const size_t QKV_DIM_NUM = 3;
const size_t BETA_DIM_NUM = 2;
const size_t STATE_DIM_NUM = 4;
const size_t ACTUAL_SEQ_LENGTHS_DIM_NUM = 1;
const size_t G_DIM_NUM = 2;

const size_t DIM_0 = 0;
const size_t DIM_1 = 1;
const size_t DIM_2 = 2;
const size_t DIM_3 = 3;

// 固定系统 workspace 大小（16 MB）
constexpr int64_t SYS_WORKSPACE_SIZE = 16777216;

// Matmul tiling 相关常量
constexpr uint32_t MATMUL_BASE_M = 128;
constexpr uint32_t MATMUL_BASE_K = 128;
constexpr uint32_t MATMUL_BASE_N = 128;

constexpr uint32_t STAGE_ONE_TWO = 2;
constexpr uint32_t STAGE_ONE_THREE = 3;
constexpr uint32_t STAGE_ONE_PARA_NUM = 4;
constexpr uint32_t MASK_NUM = 4;
constexpr int64_t P_NUM = 2;

// 初始化编译信息，读取平台资源，并缓存核数到 tilingData_
void ChunkGatedDeltaRuleTiling::InitCompileInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platformInfoPtr is null");
        return;
    }
    const auto &ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
    compileInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo_.aicNum = ascendcPlatform.GetCoreNumAic();
    socVersion_ = ascendcPlatform.GetSocVersion();

    if (compileInfo_.aivNum <= 0 || compileInfo_.aicNum <= 0) {
        OP_LOGE(context_->GetNodeName(), "aivNum <= 0 or aicNum <= 0");
        return;
    }
    tilingData_.aiCoreNum = compileInfo_.aicNum;
}

ge::graphStatus ChunkGatedDeltaRuleTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
};

// 获取输入输出信息，依次完成上下文、dtype、shape、属性、可选输入和 format 校验
ge::graphStatus ChunkGatedDeltaRuleTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid context."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetOptionalInput() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid GetOptionalInput."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetScale() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid GetScale."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(AnalyzeDtype() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid dtypes."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(AnalyzeShapes() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid shapes."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetStrides() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid GetStrides."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(AnalyzeFormat() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid Format."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleTiling::DoOpTiling()
{
    int64_t c = 64;    // chunk size 取 64
    int64_t p = P_NUM; // 一个 chunk 组中，单核最大处理 chunk 数
    tilingData_.chunkSize = c;
    tilingData_.maxGroupLength = p * tilingData_.aiCoreNum * tilingData_.chunkSize;
    tilingData_.stageOneParaNum = STAGE_ONE_PARA_NUM; // stage1 并行数

    tilingData_.interWorkspaceSz = 0;
    int64_t sizeHigh = ge::GetSizeByDataType(ge::DT_FLOAT);
    int64_t sizeLow = ge::GetSizeByDataType(ge::DT_BF16);
    int64_t nv = tilingData_.nv;
    int64_t dv = tilingData_.dv;
    int64_t dk = tilingData_.dk;
    int64_t s = tilingData_.maxGroupLength;
    tilingData_.interWorkspaceSz += sizeHigh * nv * s;                                   // gCumExp (FP32)
    tilingData_.interWorkspaceSz += sizeLow * nv * s * dk;                               // kCumDecay (BF16)
    if (tilingData_.stateIsFp32) {
        tilingData_.interWorkspaceSz += sizeHigh * nv * s * dv;                           // vInner (FP32)
        tilingData_.interWorkspaceSz += sizeLow * nv * s * dv;                            // vInnerBf16 (BF16, for stage3)
    } else {
        tilingData_.interWorkspaceSz += sizeLow * nv * s * dv;                            // vInner (BF16)
    }
    tilingData_.interWorkspaceSz += sizeLow * nv * s * dk;                               // qPrime (BF16)
    tilingData_.interWorkspaceSz += sizeLow * nv * s * dv;                               // attnInter (BF16, arch22 compat)
    tilingData_.interWorkspaceSz += sizeLow * nv * s * dk;                               // kg (BF16)
    tilingData_.interWorkspaceSz += sizeLow * nv * s * c;                                // qkt (BF16)
    if (tilingData_.stateIsFp32) {
        tilingData_.interWorkspaceSz += sizeLow * nv * dv * dk;                           // stateBf16Wk (BF16, arch35)
    } else if (socVersion_ != platform_ascendc::SocVersion::ASCEND950) {
        tilingData_.interWorkspaceSz += sizeHigh * tilingData_.b * nv * dv * dk;         // highState_ (arch22: kernel unconditionally advances offset)
    }
    tilingData_.interWorkspaceSz += sizeHigh * c * c * tilingData_.aiCoreNum * MASK_NUM; // mask (FP32)

    // stage1 临时变量空间
    tilingData_.stageWorkspaceSz =
        sizeLow * c * (STAGE_ONE_TWO * c + STAGE_ONE_THREE * dk + dv) * tilingData_.stageOneParaNum;
    tilingData_.stageWorkspaceSz *= tilingData_.aiCoreNum;

    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleTiling::DoMatmulTiling()
{
    // 计算 baseM、baseN、baseK
    uint32_t baseM = MATMUL_BASE_M;
    uint32_t baseK = MATMUL_BASE_K;
    uint32_t baseN = MATMUL_BASE_N;

    // ========== MT_FP32: FP32 -> FP32 ==========
    matmul_tiling::MultiCoreMatmulTiling mm_;
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0CSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSize);
    mm_.SetBufferSpace(l1Size, l0CSize, ubSize);
    mm_.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                 matmul_tiling::DataType::DT_BFLOAT16, true);
    mm_.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                 matmul_tiling::DataType::DT_BFLOAT16, true);
    mm_.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                 matmul_tiling::DataType::DT_BFLOAT16);
    mm_.SetBias(false);
    mm_.SetDim(1);
    mm_.SetShape(baseM, baseN, baseK);
    mm_.SetOrgShape(baseM, baseN, baseK);
    mm_.SetFixSplit(baseM, baseN, baseK);
    if (mm_.GetTiling(tilingData_.matmulTilingFp32) == -1) {
        OP_LOGE(context_->GetNodeName(), "CGDR: Get Tiling Failed!");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context_->GetNodeName(), "CGDR: baseM is %d, baseK is %d, baseN is %d.", baseM, baseK, baseN);
    tilingData_.matmulTilingFp32.dbL0C = 1;
    tilingData_.matmulTilingFp32.stepKa = 1;
    tilingData_.matmulTilingFp32.stepKb = 1;
    tilingData_.matmulTilingFp32.depthA1 = 1;
    tilingData_.matmulTilingFp32.depthB1 = 1;
    tilingData_.matmulTilingFp32.stepM = 1;
    tilingData_.matmulTilingFp32.stepN = 1;

    // ========== MT_FP32C: BF16 -> BF16 -> FP32 (for FP32 state path) ==========
    if (tilingData_.stateIsFp32) {
        matmul_tiling::MultiCoreMatmulTiling mmFp32C;
        mmFp32C.SetBufferSpace(l1Size, l0CSize, ubSize);
        mmFp32C.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                         matmul_tiling::DataType::DT_BFLOAT16, true);
        mmFp32C.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                         matmul_tiling::DataType::DT_BFLOAT16, true);
        mmFp32C.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                         matmul_tiling::DataType::DT_FLOAT);
        mmFp32C.SetBias(false);
        mmFp32C.SetDim(1);
        mmFp32C.SetShape(baseM, baseN, baseK);
        mmFp32C.SetOrgShape(baseM, baseN, baseK);
        mmFp32C.SetFixSplit(baseM, baseN, baseK);
        if (mmFp32C.GetTiling(tilingData_.matmulTilingFp32C) == -1) {
            OP_LOGE(context_->GetNodeName(), "CGDR: Get Tiling Fp32C Failed!");
            return ge::GRAPH_FAILED;
        }
        tilingData_.matmulTilingFp32C.dbL0C = 1;
        tilingData_.matmulTilingFp32C.stepKa = 1;
        tilingData_.matmulTilingFp32C.stepKb = 1;
        tilingData_.matmulTilingFp32C.depthA1 = 1;
        tilingData_.matmulTilingFp32C.depthB1 = 1;
        tilingData_.matmulTilingFp32C.stepM = 1;
        tilingData_.matmulTilingFp32C.stepN = 1;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleTiling::DoLibApiTiling()
{
    tilingKey_ = tilingData_.stateIsFp32 ? TILING_KEY_CGDR_FP32_STATE : TILING_KEY_CGDR_BF16_STATE;

    // 执行 matmul tiling
    if (DoMatmulTiling() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "DoMatmulTiling failed");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
};

// 返回 tiling key
uint64_t ChunkGatedDeltaRuleTiling::GetTilingKey() const
{
    return tilingKey_;
};

// 计算 workspace 大小
ge::graphStatus ChunkGatedDeltaRuleTiling::GetWorkspaceSize()
{
    workspaceSize_ = SYS_WORKSPACE_SIZE;
    workspaceSize_ += tilingData_.interWorkspaceSz;
    workspaceSize_ += tilingData_.stageWorkspaceSz;
    return ge::GRAPH_SUCCESS;
};

ge::graphStatus ChunkGatedDeltaRuleTiling::SetScheduleConfig()
{
    constexpr uint32_t batchMode = 1U;
    auto ret = context_->SetScheduleMode(batchMode);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "SetScheduleMode failed, ret=%u", ret),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 写回 tilingData 和 workspace 信息
ge::graphStatus ChunkGatedDeltaRuleTiling::PostTiling()
{
    context_->SetBlockDim(tilingData_.aiCoreNum);

    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData);
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData->GetData());

    auto tilingDataSize = sizeof(ChunkGatedDeltaRule::ChunkGatedDeltaRuleTilingData);
    OP_CHECK_IF(rawTilingData->GetCapacity() < tilingDataSize,
                OP_LOGE(context_->GetNodeName(), "raw tiling data capacity %zu is smaller than tiling data size %zu",
                        rawTilingData->GetCapacity(), tilingDataSize),
                return ge::GRAPH_FAILED);

    errno_t ret = memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    rawTilingData->SetDataSize(tilingDataSize);

    size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
    OP_CHECK_IF(workspaces == nullptr, OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "workspaces is null"),
                return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;

    OP_CHECK_IF(SetScheduleConfig() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "SetScheduleMode failed"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 校验上下文中必选输入、可选输入和输出是否存在
ge::graphStatus ChunkGatedDeltaRuleTiling::CheckContext()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(QUERY_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(QUERY_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(KEY_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(KEY_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(VALUE_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(VALUE_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(BETA_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(BETA_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(STATE_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(STATE_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(ACTUAL_SEQ_LENGTHS_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(ACTUAL_SEQ_LENGTHS_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputShape(OUTPUT_OUT_IDX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(OUTPUT_OUT_IDX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputShape(OUTPUT_FINAL_STATE_IDX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(OUTPUT_FINAL_STATE_IDX));

    auto gDesc = context_->GetOptionalInputDesc(G_INDEX);
    auto gTensor = context_->GetOptionalInputTensor(G_INDEX);
    auto gShape = context_->GetOptionalInputShape(G_INDEX);
    bool hasDesc = (gDesc != nullptr);
    bool hasTensor = (gTensor != nullptr);
    bool hasShape = (gShape != nullptr);
    OP_CHECK_IF((hasDesc != hasTensor) || (hasDesc != hasShape),
                OP_LOGE(context_->GetNodeName(), "gamma desc, tensor and shape should all exist or all be null"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 校验输入输出 dtype：q/k/v/beta/state/out/final_state 为 bf16，actual_seq_lengths 为 int32，可选 g 为 float
ge::graphStatus ChunkGatedDeltaRuleTiling::AnalyzeDtype()
{
    auto queryDtype = context_->GetInputDesc(QUERY_INDEX)->GetDataType();
    auto keyDtype = context_->GetInputDesc(KEY_INDEX)->GetDataType();
    auto valueDtype = context_->GetInputDesc(VALUE_INDEX)->GetDataType();
    OP_CHECK_IF(queryDtype != ge::DT_BF16 || keyDtype != ge::DT_BF16 || valueDtype != ge::DT_BF16,
                OP_LOGE(context_->GetNodeName(), "query dtype, key dtype and value dtype should be bfloat16"),
                return ge::GRAPH_FAILED);

    auto betaDtype = context_->GetInputDesc(BETA_INDEX)->GetDataType();
    auto stateDtype = context_->GetInputDesc(STATE_INDEX)->GetDataType();
    OP_CHECK_IF(betaDtype != ge::DT_BF16,
                OP_LOGE(context_->GetNodeName(), "beta dtype should be bfloat16"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(stateDtype != ge::DT_BF16 && stateDtype != ge::DT_FLOAT,
                OP_LOGE(context_->GetNodeName(), "state dtype should be bfloat16 or float32"),
                return ge::GRAPH_FAILED);
    if (stateDtype == ge::DT_FLOAT && socVersion_ != platform_ascendc::SocVersion::ASCEND950) {
        OP_LOGE(context_->GetNodeName(), "FP32 state is only supported on Ascend950");
        return ge::GRAPH_FAILED;
    }
    tilingData_.stateIsFp32 = (stateDtype == ge::DT_FLOAT) ? 1 : 0;

    auto actualSeqLengthsDtype = context_->GetInputDesc(ACTUAL_SEQ_LENGTHS_INDEX)->GetDataType();
    OP_CHECK_IF(actualSeqLengthsDtype != ge::DT_INT32,
                OP_LOGE(context_->GetNodeName(), "actual_seq_lengths dtype should be int32"), return ge::GRAPH_FAILED);

    if (tilingData_.hasGamma != 0) {
        auto gammaDtype = context_->GetOptionalInputDesc(G_INDEX)->GetDataType();
        OP_CHECK_IF(gammaDtype != ge::DT_FLOAT, OP_LOGE(context_->GetNodeName(), "gamma dtype should be float32"),
                    return ge::GRAPH_FAILED);
    }

    auto outDtype = context_->GetOutputDesc(OUTPUT_OUT_IDX)->GetDataType();
    auto finalStateDtype = context_->GetOutputDesc(OUTPUT_FINAL_STATE_IDX)->GetDataType();
    OP_CHECK_IF(outDtype != ge::DT_BF16, OP_LOGE(context_->GetNodeName(), "output dtype should be bfloat16"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(finalStateDtype != ge::DT_BF16 && finalStateDtype != ge::DT_FLOAT,
                OP_LOGE(context_->GetNodeName(), "final_state dtype should be bfloat16 or float32"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(stateDtype != finalStateDtype,
                OP_LOGE(context_->GetNodeName(),
                        "initial_state and final_state must have the same dtype, got initial_state=%d, final_state=%d",
                        static_cast<int>(stateDtype), static_cast<int>(finalStateDtype)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}


// 校验 shape 的维度数是否符合预期
bool ChunkGatedDeltaRuleTiling::CheckDim(const gert::Shape &shape, const size_t dim, const std::string &dimDesc)
{
    if (shape.GetDimNum() != dim) {
        OP_LOGE(context_->GetNodeName(), "The number of dimensions of %s should be %zu, but it is %zu", dimDesc.c_str(),
                dim, shape.GetDimNum());
        return false;
    }
    return true;
}

// 统一校验所有输入输出 shape 是否与理想 shape 一致
ge::graphStatus ChunkGatedDeltaRuleTiling::CheckExpectedShapes(
    const gert::Shape &queryShape, const gert::Shape &keyShape, const gert::Shape &valueShape,
    const gert::Shape &betaShape, const gert::Shape &stateShape, const gert::Shape &actualSeqLengthsShape,
    const gert::Shape &outShape, const gert::Shape &finalStateShape, const gert::Shape *gShape)
{
    const gert::Shape expectQueryShape = gert::Shape({tilingData_.t, tilingData_.nk, tilingData_.dk});
    const gert::Shape expectKeyShape = gert::Shape({tilingData_.t, tilingData_.nk, tilingData_.dk});
    const gert::Shape expectValueShape = gert::Shape({tilingData_.t, tilingData_.nv, tilingData_.dv});
    const gert::Shape expectBetaShape = gert::Shape({tilingData_.t, tilingData_.nv});
    const gert::Shape expectStateShape = gert::Shape({tilingData_.b, tilingData_.nv, tilingData_.dv, tilingData_.dk});
    const gert::Shape expectActualSeqLengthsShape = gert::Shape({tilingData_.b});
    const gert::Shape expectOutShape = gert::Shape({tilingData_.t, tilingData_.nv, tilingData_.dv});
    const gert::Shape expectFinalStateShape =
        gert::Shape({tilingData_.b, tilingData_.nv, tilingData_.dv, tilingData_.dk});

    OP_CHECK_IF(queryShape != expectQueryShape, OP_LOGE(context_->GetNodeName(), "query shape is invalid"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyShape != expectKeyShape, OP_LOGE(context_->GetNodeName(), "key shape is invalid"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(valueShape != expectValueShape, OP_LOGE(context_->GetNodeName(), "value shape is invalid"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(betaShape != expectBetaShape, OP_LOGE(context_->GetNodeName(), "beta shape is invalid"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(stateShape != expectStateShape, OP_LOGE(context_->GetNodeName(), "state shape is invalid"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(actualSeqLengthsShape != expectActualSeqLengthsShape,
                OP_LOGE(context_->GetNodeName(), "actual_seq_lengths shape is invalid"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(outShape != expectOutShape, OP_LOGE(context_->GetNodeName(), "out shape is invalid"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(finalStateShape != expectFinalStateShape,
                OP_LOGE(context_->GetNodeName(), "final_state shape is invalid"), return ge::GRAPH_FAILED);

    if (gShape != nullptr) {
        const gert::Shape expectGShape = gert::Shape({tilingData_.t, tilingData_.nv});
        OP_CHECK_IF(*gShape != expectGShape, OP_LOGE(context_->GetNodeName(), "g shape is invalid"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

// 校验从 shape 推导出的维度约束
ge::graphStatus ChunkGatedDeltaRuleTiling::CheckDerivedDimConstraints()
{
    OP_CHECK_IF(
        tilingData_.t <= 0 || tilingData_.b <= 0 || tilingData_.nk <= 0 || tilingData_.dk <= 0 || tilingData_.nv <= 0 ||
            tilingData_.dv <= 0,
        OP_LOGE(inputParams_.opName,
                "T, B, Nk, Dk, Nv, Dv should be greater than 0, but T=%ld, B=%ld, Nk=%ld, Dk=%ld, Nv=%ld, Dv=%ld",
                tilingData_.t, tilingData_.b, tilingData_.nk, tilingData_.dk, tilingData_.nv, tilingData_.dv),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(tilingData_.nk > 64 || tilingData_.nv > 64,
                OP_LOGE(inputParams_.opName, "nk and nv should no bigger than 64, but nk is %ld, nv is %ld",
                        tilingData_.nk, tilingData_.nv),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(tilingData_.dv > 128 || tilingData_.dk > 128,
                OP_LOGE(inputParams_.opName, "dv and dk should be no bigger than 128, but dv is %ld, dk is %ld",
                        tilingData_.dv, tilingData_.dk),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(tilingData_.nv % tilingData_.nk != 0,
                OP_LOGE(inputParams_.opName, "nv should be an integer multiple of nk, but nv is %ld, nk is %ld",
                        tilingData_.nv, tilingData_.nk),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 统一校验输入输出 shape/rank 约束，并从锚点 shape 中解析 tilingData 维度
ge::graphStatus ChunkGatedDeltaRuleTiling::AnalyzeShapes()
{
    const auto &queryShape = context_->GetInputShape(QUERY_INDEX)->GetOriginShape();
    const auto &keyShape = context_->GetInputShape(KEY_INDEX)->GetOriginShape();
    const auto &valueShape = context_->GetInputShape(VALUE_INDEX)->GetOriginShape();
    const auto &betaShape = context_->GetInputShape(BETA_INDEX)->GetOriginShape();
    const auto &stateShape = context_->GetInputShape(STATE_INDEX)->GetOriginShape();
    const auto &actualSeqLengthsShape = context_->GetInputShape(ACTUAL_SEQ_LENGTHS_INDEX)->GetOriginShape();
    const auto &outShape = context_->GetOutputShape(OUTPUT_OUT_IDX)->GetOriginShape();
    const auto &finalStateShape = context_->GetOutputShape(OUTPUT_FINAL_STATE_IDX)->GetOriginShape();
    const gert::Shape *gShape = nullptr;

    // 先校验锚点 rank，保证后续 GetDim 安全
    if (!CheckDim(queryShape, QKV_DIM_NUM, "query") || !CheckDim(valueShape, QKV_DIM_NUM, "value") ||
        !CheckDim(stateShape, STATE_DIM_NUM, "state")) {
        return ge::GRAPH_FAILED;
    }

    // 从锚点 shape 中解析公共参数
    tilingData_.t = queryShape.GetDim(DIM_0);
    tilingData_.nk = queryShape.GetDim(DIM_1);
    tilingData_.dk = queryShape.GetDim(DIM_2);
    tilingData_.nv = valueShape.GetDim(DIM_1);
    tilingData_.dv = valueShape.GetDim(DIM_2);
    tilingData_.b = stateShape.GetDim(DIM_0);

    OP_CHECK_IF(CheckDerivedDimConstraints() != ge::GRAPH_SUCCESS,
                OP_LOGE(inputParams_.opName, "Invalid derived dim constraints."), return ge::GRAPH_FAILED);

    if (tilingData_.hasGamma != 0) {
        gShape = &context_->GetOptionalInputShape(G_INDEX)->GetOriginShape();
    }

    OP_CHECK_IF(CheckExpectedShapes(queryShape, keyShape, valueShape, betaShape, stateShape, actualSeqLengthsShape,
                                    outShape, finalStateShape, gShape) != ge::GRAPH_SUCCESS,
                OP_LOGE(inputParams_.opName, "Invalid shape constraints."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// tiling 阶段基于 primary format 做格式校验。
// GetPrimaryFormat() 可以吸收一部分派生格式，但 NCL/NCHW 等布局不会统一折回 ND。
// 因此这里只拦截当前明确不支持的 FRACTAL_NZ，以避免误拦其他合法的 ND 派生布局。
bool ChunkGatedDeltaRuleTiling::CheckFormat(const gert::CompileTimeTensorDesc *desc, const std::string &name)
{
    auto primaryFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(desc->GetStorageFormat()));
    OP_CHECK_IF(primaryFormat == ge::FORMAT_FRACTAL_NZ,
                OP_LOGE(context_->GetNodeName(), "%s format %s is not supported", name.c_str(),
                        ge::TypeUtils::FormatToSerialString(primaryFormat).c_str()),
                return false);
    return true;
}

// 校验输入输出 format，可选 gamma 存在时也需要校验
ge::graphStatus ChunkGatedDeltaRuleTiling::AnalyzeFormat()
{
    if (!CheckFormat(context_->GetInputDesc(QUERY_INDEX), "query") ||
        !CheckFormat(context_->GetInputDesc(KEY_INDEX), "key") ||
        !CheckFormat(context_->GetInputDesc(VALUE_INDEX), "value") ||
        !CheckFormat(context_->GetInputDesc(BETA_INDEX), "beta") ||
        !CheckFormat(context_->GetInputDesc(STATE_INDEX), "state") ||
        !CheckFormat(context_->GetInputDesc(ACTUAL_SEQ_LENGTHS_INDEX), "actual_seq_lengths") ||
        !CheckFormat(context_->GetOutputDesc(OUTPUT_OUT_IDX), "out") ||
        !CheckFormat(context_->GetOutputDesc(OUTPUT_FINAL_STATE_IDX), "final_state")) {
        return ge::GRAPH_FAILED;
    }

    if (tilingData_.hasGamma != 0) {
        if (!CheckFormat(context_->GetOptionalInputDesc(G_INDEX), "gamma")) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleTiling::GetScale()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs is null"), return ge::GRAPH_FAILED);
    auto scalePtr = attrs->GetAttrPointer<float>(0);
    OP_CHECK_IF(scalePtr == nullptr, OP_LOGE(context_->GetNodeName(), "scale attr is null"), return ge::GRAPH_FAILED);
    tilingData_.scale = *scalePtr;
    return ge::GRAPH_SUCCESS;
}

 ge::graphStatus ChunkGatedDeltaRuleTiling::GetStrides()
{
    auto strideState = context_->GetInputStride(STATE_INDEX);
    if (strideState != nullptr && strideState->GetDimNum() == STATE_DIM_NUM) {
        tilingData_.stateStride0 = strideState->GetStride(0);
        tilingData_.stateStride1 = strideState->GetStride(1);
    } else {
        tilingData_.stateStride1 = static_cast<uint64_t>(tilingData_.dk) * static_cast<uint64_t>(tilingData_.dv);
        tilingData_.stateStride0 = static_cast<uint64_t>(tilingData_.nv) * tilingData_.stateStride1;
    }
    return ge::GRAPH_SUCCESS;
}

// 负责判断 gamma 是否存在，并将状态写入 tilingData_.hasGamma（0 或 1）
ge::graphStatus ChunkGatedDeltaRuleTiling::GetOptionalInput()
{
    auto gDesc = context_->GetOptionalInputDesc(G_INDEX);
    auto gTensor = context_->GetOptionalInputTensor(G_INDEX);
    auto gShape = context_->GetOptionalInputShape(G_INDEX);

    tilingData_.hasGamma = (gDesc != nullptr && gTensor != nullptr && gShape != nullptr) ? 1 : 0;
    return ge::GRAPH_SUCCESS;
}


void ChunkGatedDeltaRuleTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(), "aiCoreNum: [%ld]", tilingData_.aiCoreNum);
    OP_LOGD(context_->GetNodeName(), "t: [%ld]", tilingData_.t);
    OP_LOGD(context_->GetNodeName(), "nk: [%ld]", tilingData_.nk);
    OP_LOGD(context_->GetNodeName(), "dk: [%ld]", tilingData_.dk);
    OP_LOGD(context_->GetNodeName(), "nv: [%ld]", tilingData_.nv);
    OP_LOGD(context_->GetNodeName(), "dv: [%ld]", tilingData_.dv);
    OP_LOGD(context_->GetNodeName(), "b: [%ld]", tilingData_.b);
    OP_LOGD(context_->GetNodeName(), "hasGamma: [%ld]", tilingData_.hasGamma);
    OP_LOGD(context_->GetNodeName(), "chunkSize: [%ld]", tilingData_.chunkSize);
    OP_LOGD(context_->GetNodeName(), "maxGroupLength: [%ld]", tilingData_.maxGroupLength);
    OP_LOGD(context_->GetNodeName(), "interWorkspaceSz: [%ld]", tilingData_.interWorkspaceSz);
    OP_LOGD(context_->GetNodeName(), "stageWorkspaceSz: [%ld]", tilingData_.stageWorkspaceSz);
    OP_LOGD(context_->GetNodeName(), "stageOneParaNum: [%ld]", tilingData_.stageOneParaNum);
    OP_LOGD(context_->GetNodeName(), "stateStride0: [%ld]", tilingData_.stateStride0);
    OP_LOGD(context_->GetNodeName(), "stateStride1: [%ld]", tilingData_.stateStride1);
    OP_LOGD(context_->GetNodeName(), "scale: [%f]", tilingData_.scale);
    OP_LOGD(context_->GetNodeName(), "stateIsFp32: [%ld]", tilingData_.stateIsFp32);
}

// tiling 调度入口
static ge::graphStatus ChunkGatedDeltaRuleTilingFunc(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("ChunkGatedDeltaRule", "context is null"),
                return ge::GRAPH_FAILED);
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

// tiling parse 阶段准备，校验编译信息是否存在
static ge::graphStatus TilingPrepareForChunkGatedDeltaRule(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("ChunkGatedDeltaRule", "context is null"),
                return ge::GRAPH_FAILED);

    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<ChunkGatedDeltaRuleCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "compileInfoPtr is null"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkGatedDeltaRule)
    .Tiling(ChunkGatedDeltaRuleTilingFunc)
    .TilingParse<ChunkGatedDeltaRuleCompileInfo>(TilingPrepareForChunkGatedDeltaRule);
} // namespace optiling
