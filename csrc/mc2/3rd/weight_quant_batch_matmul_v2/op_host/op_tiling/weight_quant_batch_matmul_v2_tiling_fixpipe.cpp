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
 * \file weight_quant_batch_matmul_v2_tiling_fixpipe.cpp
 * \brief
 */

#include "weight_quant_batch_matmul_v2_tiling_fixpipe.h"

#include "weight_quant_batch_matmul_v2_tiling_key.h"
#include "common/op_host/math_util.h"
#include "../../op_kernel/weight_quant_batch_matmul_v2_kernel_tiling_key.h"

namespace optiling {

constexpr uint64_t INT8_BLOCK_CUBE_TRANSPOSE = 32UL;

ge::graphStatus Mc2WeightQuantBatchMatmulV2TilingFixpipe::PostTiling()
{
    size_t tilingDataSize = sizeof(Mc2WeightQuantBatchMatmulV2FixpipeTilingData);
    OP_LOGD(opName_, "final tiling data size: %zu", tilingDataSize);

    OP_TILING_CHECK(tilingDataSize % sizeof(uint64_t) != 0,
                    OP_LOGE(opName_, "tiling data size[%zu] not aligned to 8", tilingDataSize),
                    return ge::GRAPH_FAILED);

    context_->GetRawTilingData()->SetDataSize(tilingDataSize);
    // 计算aic num n方向分核*m方向分核
    context_->SetBlockDim(tilingData_->nBlockNum *
                          ops::CeilDiv(matmulInfoPtr_->mSize, static_cast<uint64_t>(tilingData_->singleCoreM)));
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           tilingData_.get(), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool Mc2WeightQuantBatchMatmulV2TilingFixpipe::IsCapable()
{
    OP_LOGD(opName_,
            "begin to detect the Fixpipe template limit. MKN[%lu, %lu, %lu], "
            "groupSize_: [%lu]",
            matmulInfoPtr_->mSize, matmulInfoPtr_->kSize, matmulInfoPtr_->nSize, matmulInfoPtr_->groupSize);

    OP_TILING_CHECK(!CheckDtypeIsCapable(),
                    OP_LOGD(opName_, "check mkn finish, the Fixpipe template doesn't "
                                     "support current shape."),
                    return false);

    OP_TILING_CHECK(!CheckShapeIsCapable(),
                    OP_LOGD(opName_, "check mkn finish, the Fixpipe template doesn't "
                                     "support current shape."),
                    return false);
    return true;
}

bool Mc2WeightQuantBatchMatmulV2TilingFixpipe::CheckDtypeIsCapable() const
{
    // 仅支持输出fp16
    OP_TILING_CHECK(matmulInfoPtr_->cDtype != ge::DT_FLOAT16,
                    OP_LOGD(opName_, "the Fixpipe template only support cDtype is FP16, current is [%s].",
                            ge::TypeUtils::DataTypeToAscendString(matmulInfoPtr_->cDtype).GetString()),
                    return false);

    // 只支持W8场景
    OP_TILING_CHECK(matmulInfoPtr_->bDtype == ge::DT_INT4,
                    OP_LOGD(opName_, "the Fixpipe template only support bDtype is int8, "
                                     "current is [int4]."),
                    return false);

    // 仅支持antiquantScale类型是uint64_t/int64_t
    OP_TILING_CHECK(((matmulInfoPtr_->antiQuantScaleDtype != ge::DT_UINT64) &&
                     (matmulInfoPtr_->antiQuantScaleDtype != ge::DT_INT64)),
                    OP_LOGD(opName_,
                            "the Fixpipe template only support antiquantScaleDtype is uint64, "
                            "current is [%s].",
                            ge::TypeUtils::DataTypeToAscendString(matmulInfoPtr_->antiQuantScaleDtype).GetString()),
                    return false);
    return true;
}

bool Mc2WeightQuantBatchMatmulV2TilingFixpipe::CheckShapeIsCapable() const
{
    // 仅支持n轴\k轴都是64的倍数
    OP_TILING_CHECK(matmulInfoPtr_->nSize % 64 != 0 || matmulInfoPtr_->kSize % 64 != 0,
                    OP_LOGD(opName_, "the Fixpipe template only support n aligned to 64 "
                                     "and k aligned to 64."),
                    return false);

    // 仅支持m轴是在1-96的范围
    OP_TILING_CHECK(matmulInfoPtr_->mSize > 96,
                    OP_LOGD(opName_, "the Fixpipe template only support mSize_ in range [1, 96]."), return false);

    // 仅支持b转置场景
    OP_TILING_CHECK(!matmulInfoPtr_->transB,
                    OP_LOGD(opName_,
                            "the Fixpipe template only support weight is "
                            "transposed, current transB : [%s].",
                            matmulInfoPtr_->transB ? "true" : "false"),
                    return false);

    // 仅支持a不转置场景
    OP_TILING_CHECK(matmulInfoPtr_->transA,
                    OP_LOGD(opName_,
                            "the Fixpipe template only support x is not "
                            "transposed, current transA : [%s].",
                            matmulInfoPtr_->transA ? "true" : "false"),
                    return false);

    // 只支持perchannel场景
    OP_TILING_CHECK(matmulInfoPtr_->antiQuantType != Mc2QuantType::PER_CHANNEL,
                    OP_LOGI(opName_, "the Fixpipe template only support per channel."), return false);
    return true;
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2TilingFixpipe::InstantiateTilingData()
{
    size_t tilingDataSize = sizeof(Mc2WeightQuantBatchMatmulV2FixpipeTilingData);
    if (tilingData_ == nullptr) {
        tilingData_ = std::unique_ptr<Mc2WeightQuantBatchMatmulV2FixpipeTilingData>(
            new (std::nothrow) Mc2WeightQuantBatchMatmulV2FixpipeTilingData());
    }
    OP_TILING_CHECK(tilingData_ == nullptr, OP_LOGE(opName_, "failed to instantiate tilingData"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetRawTilingData()->GetCapacity() < tilingDataSize,
                    OP_LOGE(opName_, "tiling data capacity %zu < actual tiling data size %zu",
                            context_->GetRawTilingData()->GetCapacity(), tilingDataSize),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2TilingFixpipe::DoOpTiling()
{
    OP_TILING_CHECK(InstantiateTilingData() == ge::GRAPH_FAILED,
                    OP_LOGE(opName_, "unable to get pointer of tiling data"), return ge::GRAPH_FAILED);

    tilingData_->hasBias = matmulInfoPtr_->hasBias;
    // 保证kernel的数据32对齐，避免为了处理16的尾块而引入其他计算
    uint64_t singleCoreN =
        ops::CeilAlign(ops::CeilDiv(matmulInfoPtr_->nSize, static_cast<uint64_t>(compileInfoPtr_->aicNum)), 32UL);
    uint64_t baseN = std::min(singleCoreN, 128UL);
    if (matmulInfoPtr_->nSize <= compileInfoPtr_->aicNum * BLOCK_CUBE) {
        // 极端场景，n方向按照16的粒度依旧可以分不满，降低n方向切分粒度
        singleCoreN = BLOCK_CUBE;
        baseN = INT8_BLOCK_CUBE_TRANSPOSE;
    }
    uint64_t nBlkNum = ops::CeilDiv(matmulInfoPtr_->nSize, singleCoreN);
    uint64_t mBlkNum = compileInfoPtr_->aicNum / nBlkNum;

    // fixp方案切分的基本块是baseK = 512，
    // 此处根据实际k值缩小基本块的k，防止mmad出错
    uint64_t baseK = matmulInfoPtr_->kSize > 512 ? 512 : matmulInfoPtr_->kSize;
    uint64_t singleCoreM = ops::CeilAlign(ops::CeilDiv(matmulInfoPtr_->mSize, static_cast<uint64_t>(mBlkNum)),
                                          static_cast<uint64_t>(BLOCK_CUBE));

    // fixp基本块切分后，a的最大剩余空间是250 * 1024 byte
    uint64_t aL1MaxSize = 250 * 1024;
    if (singleCoreM * matmulInfoPtr_->kSize * sizeof(matmulInfoPtr_->aDtype) > aL1MaxSize) {
        aFullLoad_ = 0;
    } else {
        aFullLoad_ = 1;
    }
    tilingData_->nBlockNum = nBlkNum;
    tilingData_->baseK = baseK;
    tilingData_->baseM = singleCoreM;
    tilingData_->baseN = baseN;
    tilingData_->singleCoreM = std::min(matmulInfoPtr_->mSize, singleCoreM); // 核内不切m
    tilingData_->singleCoreN = singleCoreN;
    tilingData_->mSize = matmulInfoPtr_->mSize;
    tilingData_->kSize = matmulInfoPtr_->kSize;
    tilingData_->nSize = matmulInfoPtr_->nSize;
    return ge::GRAPH_SUCCESS;
}

// 5、计算TilingKey
uint64_t Mc2WeightQuantBatchMatmulV2TilingFixpipe::GetTilingKey() const
{
    uint64_t socVersionType = static_cast<uint64_t>(Mc2SocVersionType::SUPPORT_L0C_TO_OUT);
    uint64_t subSocVersionType = 0UL;
    uint64_t antiquantScenario = static_cast<uint64_t>(Mc2QuantizationScenario::DEFAULT);
    uint64_t algorithm = static_cast<uint64_t>(Mc2OptimizationAlgorithmCategory::FIXPIPE_ANTIQUANT);
    uint64_t subAlgorithm = 0UL;
    uint64_t subAlgorithmCustom = 0UL;
    uint64_t innerPrecise = 0UL;
    uint64_t templateCustom = aFullLoad_ ?
                                  static_cast<uint64_t>(Mc2FixpipeConfiguration::A_SINGLE_M_SINGLE_K_FULL_LOAD) :
                                  static_cast<uint64_t>(Mc2FixpipeConfiguration::A_NORMAL_LOAD);
    uint64_t apiConstexpr = 0UL;
    bool transA = matmulInfoPtr_->transA;
    bool transB = matmulInfoPtr_->transB;
    uint64_t antiquantType = static_cast<uint64_t>(matmulInfoPtr_->antiQuantType);
    uint64_t quantType = static_cast<uint64_t>(Mc2QuantType::NONE);
    bool hasAntiquantOffset = matmulInfoPtr_->hasAntiQuantOffset;
    bool hasBias = matmulInfoPtr_->hasBias;
    bool isBiasFp32 = false;
    bool isWeightNz = false;      // Mc2WeightFormat::ND
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
ge::graphStatus Mc2WeightQuantBatchMatmulV2TilingFixpipe::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = compileInfoPtr_->workspaceNum;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
