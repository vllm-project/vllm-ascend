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
 * \file weight_quant_batch_matmul_v2_tiling_custom_nz_splitk.cpp
 * \brief
 */

#include "weight_quant_batch_matmul_v2_tiling_custom_nz_splitk.h"
#include "weight_quant_batch_matmul_v2_tiling_tool.h"
#include "weight_quant_batch_matmul_v2_tiling_key.h"
#include "weight_quant_batch_matmul_v2_white_list.h"
#include "../../op_kernel/weight_quant_batch_matmul_v2_kernel_tiling_key.h"

namespace optiling {

constexpr uint64_t M_MIN_LIMIT = 65UL;
constexpr uint64_t M_MAX_LIMIT = 256UL;
constexpr uint64_t MAX_SHAPE_DIM = 65535UL;
constexpr uint64_t SHAPE_ALIGNED_FACTOR = 64UL;
constexpr uint64_t SINGLE_K = 512UL;       // 单次计算K方向数据量
constexpr uint64_t VECTOR_SINGLE_N = 32UL; // 单次计算N方向数据量
constexpr uint64_t CUBE_BASE_M = 128UL;    // cube M方向基本块
constexpr uint64_t CUBE_BASE_N = 128UL;    // cube N方向基本块
constexpr uint64_t CUBE_BASE_K = 128UL;    // cube K方向基本块
constexpr uint64_t CACHE_NUM = 4UL;
constexpr uint64_t VEC_CUBE_RATIO = 2;
constexpr uint64_t DEFAULT_SINGLE_CORE_SIZE = 1024;
constexpr uint64_t L2_SIZE = 67108864;

const std::set<Mc2WhiteListShape> NETWORK_UNALIGN_WHITE_LIST = {
    // m和核数为1，表示该维度不参与匹配
    {1, 8192, 13750, false, false, true, 1},
    {1, 6144, 20708, false, false, true, 1},
    {1, 8192, 16032, false, false, true, 1},
    {1, 3696, 8192, false, false, true, 1},
    {1, 8192, 7392, true, false, true, 1}};

void Mc2WeightQuantBatchMatmulV2CustomNzSplitK::Reset()
{
    Mc2WeightQuantBatchMatmulV2Tiling::Reset();
    cubeSingleN_ = 128UL;
    al1FullLoad_ = false;

    OP_TILING_CHECK(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(), 0,
                             context_->GetRawTilingData()->GetCapacity()) != EOK,
                    OP_LOGE(opName_, "fail to memset tiling data"), return;);
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2CustomNzSplitK::PostTiling()
{
    size_t tilingDataSize = sizeof(Mc2WeightQuantBatchMatmulV2CustomNzSplitKTilingData);
    OP_LOGD(opName_, "final tiling data size: %zu", tilingDataSize);

    OP_TILING_CHECK(tilingDataSize % sizeof(uint64_t) != 0,
                    OP_LOGE(opName_, "tiling data size[%zu] not aligned to 8", tilingDataSize),
                    return ge::GRAPH_FAILED);

    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    uint32_t usedAivNum = compileInfoPtr_->aicNum * 2;
    context_->SetBlockDim(CalcTschBlockDim(usedAivNum, compileInfoPtr_->aicNum, compileInfoPtr_->aivNum));
    errno_t ret =
        memcpy_s(context_->GetTilingData<Mc2WeightQuantBatchMatmulV2CustomNzSplitKTilingData>(),
                 context_->GetRawTilingData()->GetCapacity(), reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool Mc2WeightQuantBatchMatmulV2CustomNzSplitK::IsCapable()
{
    OP_LOGI(opName_, "Start to check Custom SplitK template");
    // NZ切K模板不支持确定性
    if (context_->GetDeterministic() == 1) {
        OP_LOGI(opName_, "NZ splitk do not support deterministic");
        return false;
    }
    OP_TILING_CHECK(matmulInfoPtr_->cDtype == ge::DT_INT8 || matmulInfoPtr_->antiQuantScaleDtype == ge::DT_UINT64 ||
                        matmulInfoPtr_->antiQuantScaleDtype == ge::DT_INT64,
                    OP_LOGI(opName_, "Custom splitK not support quant or antiquant uint64"), return false);
    OP_TILING_CHECK(matmulInfoPtr_->antiQuantType != Mc2QuantType::PER_CHANNEL,
                    OP_LOGI(opName_, "Custom SplitK only support per-channel"), return false);
    OP_TILING_CHECK(matmulInfoPtr_->transA || !matmulInfoPtr_->transB,
                    OP_LOGI(opName_, "Custom SplitK only support not trans_a and trans_b"), return false);
    OP_TILING_CHECK(matmulInfoPtr_->mSize < M_MIN_LIMIT || matmulInfoPtr_->mSize > M_MAX_LIMIT ||
                        matmulInfoPtr_->kSize > MAX_SHAPE_DIM || matmulInfoPtr_->nSize > MAX_SHAPE_DIM,
                    OP_LOGI(opName_, "Custom SplitK only support 64 < m <= 256 and n < 65536 and k < 65536"),
                    return false);
    Mc2WhiteListShape shape({1, matmulInfoPtr_->kSize, matmulInfoPtr_->nSize, matmulInfoPtr_->hasBias,
                             matmulInfoPtr_->transA, matmulInfoPtr_->transB, 1});
    OP_TILING_CHECK(
        NETWORK_UNALIGN_WHITE_LIST.find(shape) == NETWORK_UNALIGN_WHITE_LIST.end() &&
            (matmulInfoPtr_->nSize % SHAPE_ALIGNED_FACTOR != 0 || matmulInfoPtr_->kSize % SHAPE_ALIGNED_FACTOR != 0),
        OP_LOGI(opName_, "Custom SplitK only support n aligned to 64 and k aligned to 64"), return false);
    OP_TILING_CHECK(matmulInfoPtr_->bFormat != ge::FORMAT_FRACTAL_NZ,
                    OP_LOGI(opName_, "Custom SplitK only support weightNz format"), return false);
    OP_LOGI(opName_, "Custom SplitK check success");
    return true;
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2CustomNzSplitK::InstantiateTilingData()
{
    size_t tilingDataSize = sizeof(Mc2WeightQuantBatchMatmulV2CustomNzSplitKTilingData);
    if (tilingData_ == nullptr) {
        tilingData_ = std::unique_ptr<Mc2WeightQuantBatchMatmulV2CustomNzSplitKTilingData>(
            new (std::nothrow) Mc2WeightQuantBatchMatmulV2CustomNzSplitKTilingData());
    }
    OP_TILING_CHECK(tilingData_ == nullptr, OP_LOGE(opName_, "failed to instantiate tilingData"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetRawTilingData()->GetCapacity() < tilingDataSize,
                    OP_LOGE(opName_, "tiling data capacity %zu < actual tiling data size %zu",
                            context_->GetRawTilingData()->GetCapacity(), tilingDataSize),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2CustomNzSplitK::DoOpTiling()
{
    OP_TILING_CHECK(InstantiateTilingData() == ge::GRAPH_FAILED,
                    OP_LOGE(opName_, "unable to get pointer of tiling data"), return ge::GRAPH_FAILED);
    tilingData_->kSize = matmulInfoPtr_->kSize;
    tilingData_->nSize = matmulInfoPtr_->nSize;
    tilingData_->mSize = matmulInfoPtr_->mSize;
    tilingData_->nSizeAlign = ops::CeilAlign(matmulInfoPtr_->nSize, static_cast<uint64_t>(BLOCK_CUBE));
    tilingData_->kSizeAlign =
        ops::CeilAlign(matmulInfoPtr_->kSize, Mc2GetBlockAlignSizeByDataType(matmulInfoPtr_->bDtype));
    tilingData_->hasBias = matmulInfoPtr_->hasBias;
    GetMatMulTiling();
    tilingData_->vecBlockDimN = tilingData_->cubeBlockDimN * VEC_CUBE_RATIO;
    tilingData_->vecBlockDimK = tilingData_->cubeBlockDimK;
    tilingData_->singleK = SINGLE_K;
    tilingData_->vecSingleN = VECTOR_SINGLE_N;
    tilingData_->singleCoreKLoop = ops::CeilDiv(tilingData_->singleCoreK, static_cast<uint64_t>(tilingData_->singleK));
    tilingData_->vectorSingleCoreN = tilingData_->cubeSingleCoreN / VEC_CUBE_RATIO;
    tilingData_->vectorSingleCoreNTail = tilingData_->cubeSingleCoreNTail / VEC_CUBE_RATIO;
    tilingData_->vecSingleCoreNLoop =
        ops::CeilDiv(tilingData_->vectorSingleCoreN, static_cast<uint64_t>(tilingData_->vecSingleN));
    tilingData_->vecSingleCoreNTailLoop =
        ops::CeilDiv(tilingData_->vectorSingleCoreNTail, static_cast<uint64_t>(tilingData_->vecSingleN));
    tilingData_->singleCoreKTailLoop =
        ops::CeilDiv(tilingData_->singleCoreKTail, static_cast<uint64_t>(tilingData_->singleK));
    uint64_t usedAivNum = compileInfoPtr_->aivNum;
    uint64_t postSingleCoreN = ops::CeilAlign(ops::CeilDiv(matmulInfoPtr_->nSize, usedAivNum), SHAPE_ALIGNED_FACTOR);
    uint64_t postSingleN = std::min(1024UL, postSingleCoreN);
    // 160K空间用于类型转换，输入db, 64K, 64K, 32K
    uint32_t postSingleM = std::min(matmulInfoPtr_->mSize, 16 * 1024 / postSingleN);
    tilingData_->postSingleN = postSingleN;
    tilingData_->postSingleM = postSingleM;
    tilingData_->postSingleCoreN = postSingleCoreN;
    return ge::GRAPH_SUCCESS;
}

// 4、计算高阶API的TilingData
ge::graphStatus Mc2WeightQuantBatchMatmulV2CustomNzSplitK::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

// 5、计算TilingKey
uint64_t Mc2WeightQuantBatchMatmulV2CustomNzSplitK::GetTilingKey() const
{
    uint64_t socVersionType = static_cast<uint64_t>(Mc2SocVersionType::SUPPORT_L0C_TO_OUT);
    uint64_t subSocVersionType = 0UL;
    uint64_t antiquantScenario = static_cast<uint64_t>(Mc2QuantizationScenario::DEFAULT);
    uint64_t algorithm = static_cast<uint64_t>(Mc2OptimizationAlgorithmCategory::VECTOR_ANTIQUANT);
    uint64_t subAlgorithm = static_cast<uint64_t>(Mc2OptimizationAlgorithmSubCategory::SPLIT_K);
    uint64_t subAlgorithmCustom = 0UL;
    uint64_t innerPrecise = 0UL;
    uint64_t templateCustom = al1FullLoad_ ? static_cast<uint64_t>(Mc2CustomSplitKConfiguration::A_MK_FULL_LOAD) :
                                             static_cast<uint64_t>(Mc2CustomSplitKConfiguration::A_NORMAL_LOAD);
    uint64_t apiConstexpr = 0UL;
    bool transA = matmulInfoPtr_->transA;
    bool transB = matmulInfoPtr_->transB;
    uint64_t antiquantType = static_cast<uint64_t>(matmulInfoPtr_->antiQuantType);
    uint64_t quantType = static_cast<uint64_t>(Mc2QuantType::NONE);
    bool hasAntiquantOffset = matmulInfoPtr_->hasAntiQuantOffset;
    bool hasBias = false;
    bool isBiasFp32 = false;
    bool isWeightNz = true;       // Mc2WeightFormat::FRACTAL_NZ
    uint64_t templateExtra = 3UL; // 3 means TEMPLATE_EXTRA_NOT_USED
    uint64_t fullLoadMode = 5UL;  // 5 means FULL_LOAD_MODE_NOT_USED
    uint64_t batch = 0UL;
    uint64_t tilingKey_ = GET_TPL_TILING_KEY(socVersionType, subSocVersionType, antiquantScenario, algorithm,
                                             subAlgorithm, subAlgorithmCustom, innerPrecise, templateCustom,
                                             apiConstexpr, transA, transB, antiquantType, quantType, hasAntiquantOffset,
                                             hasBias, isBiasFp32, isWeightNz, templateExtra, fullLoadMode, batch);
    return tilingKey_;
}

// 6、计算Workspace 大小
ge::graphStatus Mc2WeightQuantBatchMatmulV2CustomNzSplitK::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(opName_, "failed to get workspace size"), return ge::GRAPH_FAILED);
    uint64_t workspaceSize =
        CACHE_NUM * CUBE_BASE_N * SINGLE_K * compileInfoPtr_->aicNum * GetSizeByDataType(matmulInfoPtr_->aDtype) +
        sizeof(float) * matmulInfoPtr_->mSize * matmulInfoPtr_->nSize + compileInfoPtr_->workspaceNum;

    workspaces[0] = std::max(workspaceSize, L2_SIZE);
    return ge::GRAPH_SUCCESS;
}

void Mc2WeightQuantBatchMatmulV2CustomNzSplitK::GetMatMulTiling()
{
    uint64_t singleCoreK = DEFAULT_SINGLE_CORE_SIZE;
    uint64_t cubeBlockDimK =
        std::min(ops::CeilDiv(tilingData_->kSizeAlign, singleCoreK), static_cast<uint64_t>(compileInfoPtr_->aicNum));
    singleCoreK = ops::CeilAlign(ops::CeilDiv(tilingData_->kSizeAlign, cubeBlockDimK), DEFAULT_SINGLE_CORE_SIZE);
    cubeBlockDimK = ops::CeilDiv(tilingData_->kSizeAlign, singleCoreK);
    // L1的一半空间256K用来载入A矩阵
    if (matmulInfoPtr_->mSize * singleCoreK * GetSizeByDataType(matmulInfoPtr_->aDtype) <= 256 * 1024) {
        al1FullLoad_ = true;
    }
    uint64_t cubeBlockDimN = compileInfoPtr_->aicNum / cubeBlockDimK;
    uint64_t cubeSingleCoreN =
        ops::CeilAlign(ops::CeilDiv(tilingData_->nSizeAlign, cubeBlockDimN), SHAPE_ALIGNED_FACTOR);
    cubeBlockDimN = ops::CeilDiv(tilingData_->nSizeAlign, cubeSingleCoreN);
    if (cubeSingleCoreN < cubeSingleN_) {
        cubeSingleN_ = cubeSingleN_ >> 1;
    }
    tilingData_->cubeSingleCoreN = cubeSingleCoreN;
    tilingData_->singleCoreK = singleCoreK;
    tilingData_->cubeBlockDimK = cubeBlockDimK;
    tilingData_->cubeBlockDimN = cubeBlockDimN;
    tilingData_->cubeSingleM =
        ops::CeilAlign(std::min(matmulInfoPtr_->mSize, CUBE_BASE_M), static_cast<uint64_t>(AscendC::BLOCK_CUBE));
    tilingData_->cubeSingleN = cubeSingleN_;
    tilingData_->cubeBaseK = CUBE_BASE_K;
    tilingData_->cubeSingleCoreNLoop = ops::CeilDiv(cubeSingleCoreN, cubeSingleN_);
    uint64_t usedAicNum = ops::CeilDiv(tilingData_->nSizeAlign, cubeSingleCoreN);
    uint64_t cubeSingleCoreNTail = tilingData_->nSizeAlign - (usedAicNum - 1) * cubeSingleCoreN;
    tilingData_->cubeSingleCoreNTail = cubeSingleCoreNTail;
    uint64_t cubeSingleCoreNOriTail = tilingData_->nSize - (usedAicNum - 1) * cubeSingleCoreN;
    usedAicNum = ops::CeilDiv(tilingData_->kSizeAlign, singleCoreK);
    uint64_t singleCoreKTail = tilingData_->kSizeAlign - (usedAicNum - 1) * singleCoreK;
    tilingData_->cubeSingleCoreNLoop = ops::CeilDiv(cubeSingleCoreN, cubeSingleN_);
    tilingData_->cubeSingleCoreNOriTail = cubeSingleCoreNOriTail;
    uint64_t singleCoreKOriTail = tilingData_->kSize - (usedAicNum - 1) * singleCoreK;
    tilingData_->singleCoreKTail = singleCoreKTail;
    tilingData_->singleCoreKOriTail = singleCoreKOriTail;
    tilingData_->cubeSingleCoreNTailLoop = ops::CeilDiv(cubeSingleCoreNTail, static_cast<uint64_t>(cubeSingleN_));
}

} // namespace optiling
