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
 * \file weight_quant_batch_matmul_v2_tiling_splitk.cpp
 * \brief
 */

#include "weight_quant_batch_matmul_v2_tiling_splitk.h"

#include "weight_quant_batch_matmul_v2_compute_matmul_tiling.h"
#include "weight_quant_batch_matmul_v2_white_list.h"
#include "op_host/tiling_key.h"
#include "../../op_kernel/weight_quant_batch_matmul_v2_kernel_tiling_key.h"

using Ops::Transformer::OpTiling::RecursiveSum;

namespace optiling {

const std::set<Mc2WhiteListShape> MIX_SPLIT_K_WHITE_LIST = {
    // JYXC
    {24, 12288, 1792, false, false, false, 1},
    {24, 12288, 7808, false, false, false, 1},
    {24, 3904, 12288, false, false, false, 1},
    {24, 1536, 12288, false, false, false, 1}};

void Mc2WeightQuantBatchMatmulV2TilingSplitK::Reset()
{
    Mc2WeightQuantBatchMatmulV2Tiling::Reset();
    OP_TILING_CHECK(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(), 0,
                             context_->GetRawTilingData()->GetCapacity()) != EOK,
                    OP_LOGE(opName_, "fail to memset tiling data"), return;);
}

/*
The function is limite of splitK
1. bf16*int8=>bf16 without antiquantoffset
2. JYXC white case in pergroup and weightND
*/
bool Mc2WeightQuantBatchMatmulV2TilingSplitK::IsCapable()
{
    OP_LOGI(opName_, "Begin check SplitK");
    // ND切K模板不支持确定性
    if (context_->GetDeterministic() == 1) {
        OP_LOGI(opName_, "ND splitk do not support deterministic");
        return false;
    }
    // 当前仅支持bf16*int8=>bf16
    if (matmulInfoPtr_->aDtype != ge::DT_BF16 || matmulInfoPtr_->bDtype != ge::DT_INT8 ||
        matmulInfoPtr_->cDtype != ge::DT_BF16) {
        OP_LOGI(opName_, "SplitK only support bf16*int8=bf16, current aDtype: [%s], bDtype: [%s], cDtype: [%s]",
                ge::TypeUtils::DataTypeToAscendString(matmulInfoPtr_->aDtype).GetString(),
                ge::TypeUtils::DataTypeToAscendString(matmulInfoPtr_->bDtype).GetString(),
                ge::TypeUtils::DataTypeToAscendString(matmulInfoPtr_->cDtype).GetString());
        return false;
    }
    OP_TILING_CHECK(matmulInfoPtr_->antiQuantScaleDtype == ge::DT_UINT64 ||
                        matmulInfoPtr_->antiQuantScaleDtype == ge::DT_INT64,
                    OP_LOGI(opName_, "SplitK done not support antiquant scale dtype is uint64 or int64"), return false);
    // only support jyxc case
    if (matmulInfoPtr_->antiQuantType == Mc2QuantType::PER_GROUP && matmulInfoPtr_->bFormat != ge::FORMAT_FRACTAL_NZ) {
        Mc2WhiteListShape shape({matmulInfoPtr_->mSize, matmulInfoPtr_->kSize, matmulInfoPtr_->nSize,
                                 matmulInfoPtr_->hasBias, matmulInfoPtr_->transA, matmulInfoPtr_->transB, 1});
        OP_TILING_CHECK(MIX_SPLIT_K_WHITE_LIST.find(shape) == MIX_SPLIT_K_WHITE_LIST.end(),
                        OP_LOGI(opName_, "the case is not match white case for split k"), return false);
        OP_TILING_CHECK(!matmulInfoPtr_->hasAntiQuantOffset,
                        OP_LOGI(opName_, "the white case must with antiquant offset"), return false);
        OP_LOGI(opName_, "Check SplitK succ");
        return true;
    }
    return false;
}

bool Mc2WeightQuantBatchMatmulV2TilingSplitK::GetMatMulTiling()
{
    matmul_tiling::DataType mmInputDtype = Mc2GetMatmulTilingDtype(matmulInfoPtr_->aDtype);
    matmul_tiling::MatmulApiTiling mmTiling;
    mmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmInputDtype,
                      matmulInfoPtr_->transA);
    mmTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmInputDtype,
                      matmulInfoPtr_->transB);
    mmTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                      Mc2GetMatmulTilingDtype(ge::DT_FLOAT));
    mmTiling.SetBias(matmulInfoPtr_->hasBias);
    mmTiling.SetOrgShape(matmulInfoPtr_->mSize, matmulInfoPtr_->nSize, matmulInfoPtr_->kSize);

    uint64_t cubeSingleCoreN = 1024;
    uint64_t cubeSingleCoreK = 64;
    mmTiling.SetShape(matmulInfoPtr_->mSize, cubeSingleCoreN, cubeSingleCoreK);
    mmTiling.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize);
    mmTiling.SetFixSplit(ops::CeilAlign(static_cast<uint32_t>(matmulInfoPtr_->mSize), BLOCK_CUBE), BASIC_BLOCK,
                         cubeSingleCoreK);
    OP_TILING_CHECK(mmTiling.GetTiling(tilingData_->matmulTiling) == -1,
                    OP_LOGE(opName_, "failed to get matmul tiling"), return false);
    tilingData_->matmulTiling.shareL1Size = 0;
    tilingData_->matmulTiling.dbL0C = 2; // 2: db on

    return true;
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2TilingSplitK::DoOpTiling()
{
    OP_TILING_CHECK(InstantiateTilingData() == ge::GRAPH_FAILED,
                    OP_LOGE(opName_, "Unable to get pointer of tiling data"), return ge::GRAPH_FAILED);

    tilingData_->groupSize = matmulInfoPtr_->groupSize;
    uint64_t weightBlockAlignSize = Mc2GetBlockAlignSizeByDataType(matmulInfoPtr_->bDtype);
    if (matmulInfoPtr_->transB) {
        tilingData_->kAlign = ops::CeilAlign(matmulInfoPtr_->kSize, weightBlockAlignSize);
        tilingData_->nAlign = matmulInfoPtr_->nSize;
        tilingData_->kPadSize = static_cast<uint8_t>(tilingData_->kAlign - matmulInfoPtr_->kSize);
    } else {
        tilingData_->kAlign = matmulInfoPtr_->kSize;
        tilingData_->nAlign = ops::CeilAlign(matmulInfoPtr_->nSize, weightBlockAlignSize);
        tilingData_->nPadSize = static_cast<uint8_t>(tilingData_->nAlign - matmulInfoPtr_->nSize);
    }
    tilingData_->kSize = matmulInfoPtr_->kSize;
    tilingData_->nSize = matmulInfoPtr_->nSize;
    tilingData_->mSize = matmulInfoPtr_->mSize;

    // 非转置场景重新实现切分逻辑
    // n方向以1024为最小划分单元
    uint64_t vecSingleN = 512;
    uint64_t nFactor = ops::CeilDiv(matmulInfoPtr_->nSize, vecSingleN * 2);
    uint64_t usedCoreNumMaxResult = 0;
    for (uint64_t cubeDimN = nFactor; cubeDimN >= 1; cubeDimN--) {
        if (nFactor % cubeDimN != 0) {
            continue;
        }
        uint64_t cubeDimK = compileInfoPtr_->aicNum / cubeDimN;
        uint64_t usedCoreNum = cubeDimN * cubeDimK;
        if (usedCoreNum > usedCoreNumMaxResult) {
            tilingData_->cubeBlockDimK = static_cast<uint8_t>(cubeDimK);
            tilingData_->cubeBlockDimN = static_cast<uint8_t>(cubeDimN);
            usedCoreNumMaxResult = usedCoreNum;
        }
    }
    tilingData_->cubeBlockDimM = 1;
    tilingData_->vecBlockDimK = tilingData_->cubeBlockDimK;
    tilingData_->vecSingleK = static_cast<uint32_t>(
        ops::CeilAlign(ops::CeilDiv(matmulInfoPtr_->kSize, static_cast<uint64_t>(tilingData_->cubeBlockDimK)),
                       static_cast<uint64_t>(matmulInfoPtr_->groupSize)));
    tilingData_->vecSingleN = static_cast<uint32_t>(vecSingleN);
    // vec固定保持2倍cube的方式切分
    tilingData_->vecBlockDimN = tilingData_->cubeBlockDimN * 2;
    tilingData_->vecSingleKGroupNum =
        ops::CeilDiv(static_cast<uint64_t>(tilingData_->vecSingleK), matmulInfoPtr_->groupSize);
    OP_TILING_CHECK(!GetMatMulTiling(),
                    OP_LOGE(opName_, "Failed to get mm tiling for mnk[%lu, %lu, %lu]", matmulInfoPtr_->mSize,
                            matmulInfoPtr_->nSize, matmulInfoPtr_->kSize),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2TilingSplitK::InstantiateTilingData()
{
    size_t tilingDataSize = sizeof(Mc2WeightQuantBatchMatmulV2TilingData);
    if (tilingData_ == nullptr) {
        tilingData_ = std::make_unique<Mc2WeightQuantBatchMatmulV2TilingData>();
    }
    OP_TILING_CHECK(tilingData_ == nullptr, OP_LOGE(opName_, "failed to instantiate tilingData"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetRawTilingData()->GetCapacity() < tilingDataSize,
                    OP_LOGE(opName_, "tiling data capacity %zu < actual tiling data size %zu",
                            context_->GetRawTilingData()->GetCapacity(), tilingDataSize),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2TilingSplitK::DoLibApiTiling()
{
    tilingData_->cubeSingleNTailLoop = ops::CeilDiv(matmulInfoPtr_->nSize % tilingData_->matmulTiling.singleCoreN,
                                                    static_cast<uint64_t>(tilingData_->matmulTiling.singleCoreN));
    tilingData_->cubeTailM =
        Mc2CalcTailSize(matmulInfoPtr_->mSize, static_cast<uint64_t>(tilingData_->matmulTiling.singleCoreM));
    tilingData_->cubeTailN =
        Mc2CalcTailSize(matmulInfoPtr_->nSize, static_cast<uint64_t>(tilingData_->matmulTiling.baseN));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2TilingSplitK::GetWorkspaceSize()
{
    uint64_t weightWorkspacesNum = 4;
    uint64_t nF16AlignTo512bSize = ops::CeilDiv(tilingData_->nSize, 256UL) * 256;
    uint64_t weightCacheLen =
        weightWorkspacesNum * tilingData_->cubeBlockDimK * matmulInfoPtr_->groupSize * nF16AlignTo512bSize;
    uint64_t mmResultCache = tilingData_->nSize * tilingData_->mSize;
    workspaceSize_ =
        weightCacheLen * sizeof(matmulInfoPtr_->aDtype) + mmResultCache * sizeof(float) + compileInfoPtr_->workspaceNum;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2TilingSplitK::PostTiling()
{
    size_t tilingDataSize = sizeof(Mc2WeightQuantBatchMatmulV2TilingData);
    OP_LOGD(opName_, "final tiling data size: %zu", tilingDataSize);

    OP_TILING_CHECK(tilingDataSize % sizeof(uint64_t) != 0,
                    OP_LOGE(opName_, "tiling data size[%zu] not aligned to 8", tilingDataSize),
                    return ge::GRAPH_FAILED);
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);
    uint32_t usedAicNum = tilingData_->cubeBlockDimM * tilingData_->cubeBlockDimN;
    uint32_t usedAivNum = tilingData_->vecBlockDimK * tilingData_->vecBlockDimN;
    context_->SetBlockDim(
        std::max(usedAicNum, CalcTschBlockDim(usedAivNum, compileInfoPtr_->aicNum, compileInfoPtr_->aivNum)));
    size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
    workspaces[0] = workspaceSize_;

    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           tilingData_.get(), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t Mc2WeightQuantBatchMatmulV2TilingSplitK::GetTilingKey() const
{
    uint64_t socVersionType = 1UL; // 1 means SUPPORT_L0C_TO_OUT
    uint64_t subSocVersionType = 0UL;
    uint64_t antiquantScenario = 0UL;
    uint64_t algorithm = 3UL; // 3 means CUSTOM tilingkey algorithm
    uint64_t subAlgorithm = 0UL;
    uint64_t subAlgorithmCustom = static_cast<uint64_t>(Mc2KernelTemplateType::MIX_SPLIT_K);
    uint64_t innerPrecise = 0UL;
    uint64_t templateCustom = 0UL;
    uint64_t apiConstexpr = 0UL;
    bool transA = matmulInfoPtr_->transA;
    bool transB = matmulInfoPtr_->transB;
    uint64_t antiquantType = static_cast<uint64_t>(matmulInfoPtr_->antiQuantType);
    uint64_t quantType = static_cast<uint64_t>(matmulInfoPtr_->quantType);
    bool hasAntiquantOffset = matmulInfoPtr_->hasAntiQuantOffset;
    bool hasBias = false;
    bool isBiasFp32 = false;
    bool isWeightNz = false;
    uint64_t templateExtra = 3UL; // 3 means TEMPLATE_EXTRA_NOT_USED
    uint64_t fullLoadMode = 5UL;  // 5 means FULL_LOAD_MODE_NOT_USED
    uint64_t batch = 0UL;
    uint64_t tilingKey_ = GET_TPL_TILING_KEY(socVersionType, subSocVersionType, antiquantScenario, algorithm,
                                             subAlgorithm, subAlgorithmCustom, innerPrecise, templateCustom,
                                             apiConstexpr, transA, transB, antiquantType, quantType, hasAntiquantOffset,
                                             hasBias, isBiasFp32, isWeightNz, templateExtra, fullLoadMode, batch);
    return tilingKey_;
}

} // namespace optiling
