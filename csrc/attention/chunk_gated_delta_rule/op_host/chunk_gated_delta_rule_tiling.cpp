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
#include "tiling_base/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "error/ops_error.h"
#include "log/ops_log.h"
#include "tiling/platform/platform_ascendc.h"

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
// fp32 输入时单元素为 4B，baseK 收敛到 64，保证 L0A/L0B(64KB) 不越界
constexpr uint32_t MATMUL_FP32_BASE_K = 64;

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
        OPS_LOG_E(context_->GetNodeName(), "platformInfoPtr is null");
        return;
    }
    const auto &ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
    compileInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo_.aicNum = ascendcPlatform.GetCoreNumAic();

    if (compileInfo_.aivNum <= 0 || compileInfo_.aicNum <= 0) {
        OPS_LOG_E(context_->GetNodeName(), "aivNum <= 0 or aicNum <= 0");
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
    OPS_CHECK(CheckContext() != ge::GRAPH_SUCCESS, OPS_LOG_E(inputParams_.opName, "Invalid context."),
                return ge::GRAPH_FAILED);

    OPS_CHECK(GetOptionalInput() != ge::GRAPH_SUCCESS, OPS_LOG_E(inputParams_.opName, "Invalid GetOptionalInput."),
                return ge::GRAPH_FAILED);

    OPS_CHECK(GetScale() != ge::GRAPH_SUCCESS, OPS_LOG_E(inputParams_.opName, "Invalid GetScale."),
                return ge::GRAPH_FAILED);

    OPS_CHECK(AnalyzeDtype() != ge::GRAPH_SUCCESS, OPS_LOG_E(inputParams_.opName, "Invalid dtypes."),
                return ge::GRAPH_FAILED);

    OPS_CHECK(AnalyzeShapes() != ge::GRAPH_SUCCESS, OPS_LOG_E(inputParams_.opName, "Invalid shapes."),
                return ge::GRAPH_FAILED);

    OPS_CHECK(AnalyzeFormat() != ge::GRAPH_SUCCESS, OPS_LOG_E(inputParams_.opName, "Invalid Format."),
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
    int64_t nv = tilingData_.nv;
    int64_t dv = tilingData_.dv;
    int64_t dk = tilingData_.dk;
    int64_t s = tilingData_.maxGroupLength;
    int64_t b = tilingData_.b;
    tilingData_.interWorkspaceSz += sizeHigh * nv * s;                                   // gCumExp
    tilingData_.interWorkspaceSz += sizeHigh * nv * s * dk;                              // kCumDecay
    tilingData_.interWorkspaceSz += sizeHigh * nv * s * dv;                              // vInner
    tilingData_.interWorkspaceSz += sizeHigh * nv * s * dk;                              // qPrime
    tilingData_.interWorkspaceSz += sizeHigh * nv * s * dv;                              // attnInter
    tilingData_.interWorkspaceSz += sizeHigh * nv * s * dk;                              // kg
    tilingData_.interWorkspaceSz += sizeHigh * nv * s * c;                               // qkt
    tilingData_.interWorkspaceSz += sizeHigh * b * nv * dv * dk;                         // highState
    tilingData_.interWorkspaceSz += sizeHigh * nv * dv * dk;                            // bf16State (float 路径 state 的 bf16 影子)
    tilingData_.interWorkspaceSz += sizeHigh * nv * s * dv;                             // bf16VNew (float 路径 v_new 的 bf16 影子)
    tilingData_.interWorkspaceSz += sizeHigh * c * c * tilingData_.aiCoreNum * MASK_NUM; // mask

    // stage1 临时变量空间
    tilingData_.stageWorkspaceSz =
        sizeHigh * c * (STAGE_ONE_TWO * c + STAGE_ONE_THREE * dk + dv) * tilingData_.stageOneParaNum;
    tilingData_.stageWorkspaceSz *= tilingData_.aiCoreNum;

    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleTiling::DoMatmulTiling()
{
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0CSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSize);

    // 公共的 matmul tiling 计算流程：固定单核单缓冲(depth=1, step=1, dbL0C=1)，
    // 通过 baseM/baseN/baseK 控制 L0A/L0B/L0C/L1 占用，避免 fp32 输入时越界。
    //   fp32: L0A=baseM*baseK*4, L0B=baseK*baseN*4, L0C=baseM*baseN*4(累加器恒为fp32)
    //   910B3: L0A=L0B=64KB, L0C=128KB, L1=512KB
    auto buildTiling = [&](matmul_tiling::DataType aType, bool aTrans,
                           matmul_tiling::DataType bType, bool bTrans,
                           matmul_tiling::DataType cType,
                           uint32_t baseM, uint32_t baseN, uint32_t baseK,
                           AscendC::tiling::TCubeTiling &out) -> bool {
        matmul_tiling::MultiCoreMatmulTiling mm_;
        mm_.SetBufferSpace(l1Size, l0CSize, ubSize);
        mm_.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, aType, aTrans);
        mm_.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bType, bTrans);
        mm_.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, cType);
        mm_.SetBias(false);
        mm_.SetDim(1);
        mm_.SetShape(baseM, baseN, baseK);
        mm_.SetOrgShape(baseM, baseN, baseK);
        mm_.SetFixSplit(baseM, baseN, baseK);
        if (mm_.GetTiling(out) == -1) {
            OPS_LOG_E(context_->GetNodeName(), "CGDR: Get Tiling Failed!");
            return false;
        }
        out.dbL0C = 1;
        out.stepKa = 1;
        out.stepKb = 1;
        out.depthA1 = 1;
        out.depthB1 = 1;
        out.stepM = 1;
        out.stepN = 1;
        return true;
    };

    // ========== stage2/stage3 复用：bf16,bf16 -> bf16 ==========
    if (!buildTiling(matmul_tiling::DataType::DT_BFLOAT16, true,
                     matmul_tiling::DataType::DT_BFLOAT16, true,
                     matmul_tiling::DataType::DT_BFLOAT16,
                     MATMUL_BASE_M, MATMUL_BASE_N, MATMUL_BASE_K,
                     tilingData_.matmulTilingFp32)) {
        return ge::GRAPH_FAILED;
    }

    // ========== stage1 kk / qk：bf16,bf16 -> fp32 ==========
    if (!buildTiling(matmul_tiling::DataType::DT_BFLOAT16, true,
                     matmul_tiling::DataType::DT_BFLOAT16, true,
                     matmul_tiling::DataType::DT_FLOAT,
                     MATMUL_BASE_M, MATMUL_BASE_N, MATMUL_BASE_K,
                     tilingData_.matmulTilingBf16ToFp32)) {
        return ge::GRAPH_FAILED;
    }

    // ========== stage1 求逆 / attn@v_beta：fp32,fp32 -> fp32 ==========
    if (!buildTiling(matmul_tiling::DataType::DT_FLOAT, false,
                     matmul_tiling::DataType::DT_FLOAT, false,
                     matmul_tiling::DataType::DT_FLOAT,
                     MATMUL_BASE_M, MATMUL_BASE_N, MATMUL_FP32_BASE_K,
                     tilingData_.matmulTilingFp32ToFp32)) {
        return ge::GRAPH_FAILED;
    }

    // ========== stage1 attn@k_cumdecay：fp32,fp32 -> bf16 ==========
    if (!buildTiling(matmul_tiling::DataType::DT_FLOAT, false,
                     matmul_tiling::DataType::DT_FLOAT, false,
                     matmul_tiling::DataType::DT_BFLOAT16,
                     MATMUL_BASE_M, MATMUL_BASE_N, MATMUL_FP32_BASE_K,
                     tilingData_.matmulTilingFp32ToBf16)) {
        return ge::GRAPH_FAILED;
    }

    // ========== stage2 专用：bf16 A/B -> float C ==========
    if (!buildTiling(matmul_tiling::DataType::DT_BFLOAT16, true,
                     matmul_tiling::DataType::DT_BFLOAT16, true,
                     matmul_tiling::DataType::DT_FLOAT,
                     MATMUL_BASE_M, MATMUL_BASE_N, MATMUL_BASE_K,
                     tilingData_.matmulTilingStage2)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleTiling::DoLibApiTiling()
{
    // 单一 float 路径，无 tilingKey 分发
    tilingKey_ = 0;

    // 执行 matmul tiling
    if (DoMatmulTiling() != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context_->GetNodeName(), "DoMatmulTiling failed");
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
    OPS_CHECK(ret != ge::GRAPH_SUCCESS, OPS_LOG_E(context_->GetNodeName(), "SetScheduleMode failed, ret=%u", ret),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 写回 tilingData 和 workspace 信息
ge::graphStatus ChunkGatedDeltaRuleTiling::PostTiling()
{
    context_->SetBlockDim(tilingData_.aiCoreNum);

    auto rawTilingData = context_->GetRawTilingData();
    OPS_LOG_E_IF_NULL(context_->GetNodeName(), rawTilingData, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context_->GetNodeName(), rawTilingData->GetData(), return ge::GRAPH_FAILED);

    auto tilingDataSize = sizeof(ChunkGatedDeltaRule::ChunkGatedDeltaRuleTilingData);
    OPS_CHECK(rawTilingData->GetCapacity() < tilingDataSize,
                OPS_LOG_E(context_->GetNodeName(), "raw tiling data capacity %zu is smaller than tiling data size %zu",
                        rawTilingData->GetCapacity(), tilingDataSize),
                return ge::GRAPH_FAILED);

    errno_t ret = memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OPS_LOG_E(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    rawTilingData->SetDataSize(tilingDataSize);

    size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
    OPS_CHECK(workspaces == nullptr, OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "workspaces is null"),
                return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;

    OPS_CHECK(SetScheduleConfig() != ge::GRAPH_SUCCESS, OPS_LOG_E(context_->GetNodeName(), "SetScheduleMode failed"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 校验上下文中必选输入、可选输入和输出是否存在
ge::graphStatus ChunkGatedDeltaRuleTiling::CheckContext()
{
    OPS_LOG_E_IF_NULL(context_->GetNodeName(), context_->GetInputShape(QUERY_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context_->GetNodeName(), context_->GetInputDesc(QUERY_INDEX), return ge::GRAPH_FAILED);

    OPS_LOG_E_IF_NULL(context_->GetNodeName(), context_->GetInputShape(KEY_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context_->GetNodeName(), context_->GetInputDesc(KEY_INDEX), return ge::GRAPH_FAILED);

    OPS_LOG_E_IF_NULL(context_->GetNodeName(), context_->GetInputShape(VALUE_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context_->GetNodeName(), context_->GetInputDesc(VALUE_INDEX), return ge::GRAPH_FAILED);

    OPS_LOG_E_IF_NULL(context_->GetNodeName(), context_->GetInputShape(BETA_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context_->GetNodeName(), context_->GetInputDesc(BETA_INDEX), return ge::GRAPH_FAILED);

    OPS_LOG_E_IF_NULL(context_->GetNodeName(), context_->GetInputShape(STATE_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context_->GetNodeName(), context_->GetInputDesc(STATE_INDEX), return ge::GRAPH_FAILED);

    OPS_LOG_E_IF_NULL(context_->GetNodeName(), context_->GetInputShape(ACTUAL_SEQ_LENGTHS_INDEX),
                      return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context_->GetNodeName(), context_->GetInputDesc(ACTUAL_SEQ_LENGTHS_INDEX),
                      return ge::GRAPH_FAILED);

    OPS_LOG_E_IF_NULL(context_->GetNodeName(), context_->GetOutputShape(OUTPUT_OUT_IDX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context_->GetNodeName(), context_->GetOutputDesc(OUTPUT_OUT_IDX), return ge::GRAPH_FAILED);

    OPS_LOG_E_IF_NULL(context_->GetNodeName(), context_->GetOutputShape(OUTPUT_FINAL_STATE_IDX),
                      return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context_->GetNodeName(), context_->GetOutputDesc(OUTPUT_FINAL_STATE_IDX),
                      return ge::GRAPH_FAILED);

    auto gDesc = context_->GetOptionalInputDesc(G_INDEX);
    auto gTensor = context_->GetOptionalInputTensor(G_INDEX);
    auto gShape = context_->GetOptionalInputShape(G_INDEX);
    bool hasDesc = (gDesc != nullptr);
    bool hasTensor = (gTensor != nullptr);
    bool hasShape = (gShape != nullptr);
    OPS_CHECK((hasDesc != hasTensor) || (hasDesc != hasShape),
                OPS_LOG_E(context_->GetNodeName(), "gamma desc, tensor and shape should all exist or all be null"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 校验输入输出 dtype：q/k/v/beta/out 恒 bf16；state/final_state 恒 float32（须一致）；
// actual_seq_lengths 为 int32；可选 g 为 float。
ge::graphStatus ChunkGatedDeltaRuleTiling::AnalyzeDtype()
{
    auto queryDtype = context_->GetInputDesc(QUERY_INDEX)->GetDataType();
    auto keyDtype = context_->GetInputDesc(KEY_INDEX)->GetDataType();
    auto valueDtype = context_->GetInputDesc(VALUE_INDEX)->GetDataType();
    OPS_CHECK(queryDtype != ge::DT_BF16 || keyDtype != ge::DT_BF16 || valueDtype != ge::DT_BF16,
                OPS_LOG_E(context_->GetNodeName(), "query dtype, key dtype and value dtype should be bfloat16"),
                return ge::GRAPH_FAILED);

    auto betaDtype = context_->GetInputDesc(BETA_INDEX)->GetDataType();
    auto stateDtype = context_->GetInputDesc(STATE_INDEX)->GetDataType();
    OPS_CHECK(betaDtype != ge::DT_BF16,
                OPS_LOG_E(context_->GetNodeName(), "beta dtype should be bfloat16"),
                return ge::GRAPH_FAILED);
    OPS_CHECK(stateDtype != ge::DT_FLOAT,
                OPS_LOG_E(context_->GetNodeName(), "initial_state dtype should be float32"),
                return ge::GRAPH_FAILED);

    auto actualSeqLengthsDtype = context_->GetInputDesc(ACTUAL_SEQ_LENGTHS_INDEX)->GetDataType();
    OPS_CHECK(actualSeqLengthsDtype != ge::DT_INT32,
                OPS_LOG_E(context_->GetNodeName(), "actual_seq_lengths dtype should be int32"), return ge::GRAPH_FAILED);

    if (tilingData_.hasGamma != 0) {
        auto gammaDtype = context_->GetOptionalInputDesc(G_INDEX)->GetDataType();
        OPS_CHECK(gammaDtype != ge::DT_FLOAT, OPS_LOG_E(context_->GetNodeName(), "gamma dtype should be float32"),
                    return ge::GRAPH_FAILED);
    }

    auto outDtype = context_->GetOutputDesc(OUTPUT_OUT_IDX)->GetDataType();
    auto finalStateDtype = context_->GetOutputDesc(OUTPUT_FINAL_STATE_IDX)->GetDataType();
    OPS_CHECK(outDtype != ge::DT_BF16, OPS_LOG_E(context_->GetNodeName(), "output dtype should be bfloat16"),
                return ge::GRAPH_FAILED);
    OPS_CHECK(finalStateDtype != ge::DT_FLOAT,
                OPS_LOG_E(context_->GetNodeName(), "final_state dtype should be float32"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}


// 校验 shape 的维度数是否符合预期
bool ChunkGatedDeltaRuleTiling::CheckDim(const gert::Shape &shape, const size_t dim, const std::string &dimDesc)
{
    if (shape.GetDimNum() != dim) {
        OPS_LOG_E(context_->GetNodeName(), "The number of dimensions of %s should be %zu, but it is %zu", dimDesc.c_str(),
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

    OPS_CHECK(queryShape != expectQueryShape, OPS_LOG_E(context_->GetNodeName(), "query shape is invalid"),
                return ge::GRAPH_FAILED);
    OPS_CHECK(keyShape != expectKeyShape, OPS_LOG_E(context_->GetNodeName(), "key shape is invalid"),
                return ge::GRAPH_FAILED);
    OPS_CHECK(valueShape != expectValueShape, OPS_LOG_E(context_->GetNodeName(), "value shape is invalid"),
                return ge::GRAPH_FAILED);
    OPS_CHECK(betaShape != expectBetaShape, OPS_LOG_E(context_->GetNodeName(), "beta shape is invalid"),
                return ge::GRAPH_FAILED);
    OPS_CHECK(stateShape != expectStateShape, OPS_LOG_E(context_->GetNodeName(), "state shape is invalid"),
                return ge::GRAPH_FAILED);
    OPS_CHECK(actualSeqLengthsShape != expectActualSeqLengthsShape,
                OPS_LOG_E(context_->GetNodeName(), "actual_seq_lengths shape is invalid"), return ge::GRAPH_FAILED);

    OPS_CHECK(outShape != expectOutShape, OPS_LOG_E(context_->GetNodeName(), "out shape is invalid"),
                return ge::GRAPH_FAILED);
    OPS_CHECK(finalStateShape != expectFinalStateShape,
                OPS_LOG_E(context_->GetNodeName(), "final_state shape is invalid"), return ge::GRAPH_FAILED);

    if (gShape != nullptr) {
        const gert::Shape expectGShape = gert::Shape({tilingData_.t, tilingData_.nv});
        OPS_CHECK(*gShape != expectGShape, OPS_LOG_E(context_->GetNodeName(), "g shape is invalid"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

// 校验从 shape 推导出的维度约束
ge::graphStatus ChunkGatedDeltaRuleTiling::CheckDerivedDimConstraints()
{
    OPS_CHECK(
        tilingData_.t <= 0 || tilingData_.b <= 0 || tilingData_.nk <= 0 || tilingData_.dk <= 0 || tilingData_.nv <= 0 ||
            tilingData_.dv <= 0,
        OPS_LOG_E(inputParams_.opName,
                "T, B, Nk, Dk, Nv, Dv should be greater than 0, but T=%ld, B=%ld, Nk=%ld, Dk=%ld, Nv=%ld, Dv=%ld",
                tilingData_.t, tilingData_.b, tilingData_.nk, tilingData_.dk, tilingData_.nv, tilingData_.dv),
        return ge::GRAPH_FAILED);

    OPS_CHECK(tilingData_.nk > 64 || tilingData_.nv > 64,
                OPS_LOG_E(inputParams_.opName, "nk and nv should no bigger than 64, but nk is %ld, nv is %ld",
                        tilingData_.nk, tilingData_.nv),
                return ge::GRAPH_FAILED);

    OPS_CHECK(tilingData_.dv > 128 || tilingData_.dk > 128,
                OPS_LOG_E(inputParams_.opName, "dv and dk should be no bigger than 128, but dv is %ld, dk is %ld",
                        tilingData_.dv, tilingData_.dk),
                return ge::GRAPH_FAILED);

    OPS_CHECK(tilingData_.nv % tilingData_.nk != 0,
                OPS_LOG_E(inputParams_.opName, "nv should be an integer multiple of nk, but nv is %ld, nk is %ld",
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

    OPS_CHECK(CheckDerivedDimConstraints() != ge::GRAPH_SUCCESS,
                OPS_LOG_E(inputParams_.opName, "Invalid derived dim constraints."), return ge::GRAPH_FAILED);

    if (tilingData_.hasGamma != 0) {
        gShape = &context_->GetOptionalInputShape(G_INDEX)->GetOriginShape();
    }

    OPS_CHECK(CheckExpectedShapes(queryShape, keyShape, valueShape, betaShape, stateShape, actualSeqLengthsShape,
                                    outShape, finalStateShape, gShape) != ge::GRAPH_SUCCESS,
                OPS_LOG_E(inputParams_.opName, "Invalid shape constraints."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// tiling 阶段基于 primary format 做格式校验。
// GetPrimaryFormat() 可以吸收一部分派生格式，但 NCL/NCHW 等布局不会统一折回 ND。
// 因此这里只拦截当前明确不支持的 FRACTAL_NZ，以避免误拦其他合法的 ND 派生布局。
bool ChunkGatedDeltaRuleTiling::CheckFormat(const gert::CompileTimeTensorDesc *desc, const std::string &name)
{
    auto primaryFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(desc->GetStorageFormat()));
    OPS_CHECK(primaryFormat == ge::FORMAT_FRACTAL_NZ,
                OPS_LOG_E(context_->GetNodeName(), "%s format %s is not supported", name.c_str(),
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
    OPS_CHECK(attrs == nullptr, OPS_LOG_E(context_->GetNodeName(), "attrs is null"), return ge::GRAPH_FAILED);
    auto scalePtr = attrs->GetAttrPointer<float>(0);
    OPS_CHECK(scalePtr == nullptr, OPS_LOG_E(context_->GetNodeName(), "scale attr is null"), return ge::GRAPH_FAILED);
    tilingData_.scale = *scalePtr;
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
    OPS_LOG_D(context_->GetNodeName(), "aiCoreNum: [%ld]", tilingData_.aiCoreNum);
    OPS_LOG_D(context_->GetNodeName(), "t: [%ld]", tilingData_.t);
    OPS_LOG_D(context_->GetNodeName(), "nk: [%ld]", tilingData_.nk);
    OPS_LOG_D(context_->GetNodeName(), "dk: [%ld]", tilingData_.dk);
    OPS_LOG_D(context_->GetNodeName(), "nv: [%ld]", tilingData_.nv);
    OPS_LOG_D(context_->GetNodeName(), "dv: [%ld]", tilingData_.dv);
    OPS_LOG_D(context_->GetNodeName(), "b: [%ld]", tilingData_.b);
    OPS_LOG_D(context_->GetNodeName(), "hasGamma: [%ld]", tilingData_.hasGamma);
    OPS_LOG_D(context_->GetNodeName(), "chunkSize: [%ld]", tilingData_.chunkSize);
    OPS_LOG_D(context_->GetNodeName(), "maxGroupLength: [%ld]", tilingData_.maxGroupLength);
    OPS_LOG_D(context_->GetNodeName(), "interWorkspaceSz: [%ld]", tilingData_.interWorkspaceSz);
    OPS_LOG_D(context_->GetNodeName(), "stageWorkspaceSz: [%ld]", tilingData_.stageWorkspaceSz);
    OPS_LOG_D(context_->GetNodeName(), "stageOneParaNum: [%ld]", tilingData_.stageOneParaNum);
    OPS_LOG_D(context_->GetNodeName(), "scale: [%f]", tilingData_.scale);
}

// tiling 调度入口
static ge::graphStatus ChunkGatedDeltaRuleTilingFunc(gert::TilingContext *context)
{
    OPS_CHECK(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("ChunkGatedDeltaRule", "context is null"),
                return ge::GRAPH_FAILED);
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

// tiling parse 阶段准备，校验编译信息是否存在
static ge::graphStatus TilingPrepareForChunkGatedDeltaRule(gert::TilingParseContext *context)
{
    OPS_CHECK(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("ChunkGatedDeltaRule", "context is null"),
                return ge::GRAPH_FAILED);

    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    OPS_CHECK(platformInfo == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<ChunkGatedDeltaRuleCompileInfo>();
    OPS_CHECK(compileInfoPtr == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "compileInfoPtr is null"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkGatedDeltaRule)
    .Tiling(ChunkGatedDeltaRuleTilingFunc)
    .TilingParse<ChunkGatedDeltaRuleCompileInfo>(TilingPrepareForChunkGatedDeltaRule);
} // namespace optiling
