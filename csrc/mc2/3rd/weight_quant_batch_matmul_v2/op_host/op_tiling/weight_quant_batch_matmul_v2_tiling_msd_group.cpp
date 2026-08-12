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
 * \file weight_quant_batch_matmul_v2_tiling_msd_group.cpp
 * \brief
 */
#include "weight_quant_batch_matmul_v2_tiling_msd_group.h"

#include "weight_quant_batch_matmul_v2_white_list.h"

namespace optiling {

constexpr uint64_t DOUBLE_BUFFER_FACTOR = 2UL; // 1: off, 2: on
constexpr int32_t MSD_GROUP_VEC_BASE_BLOCK = 1024;
constexpr int32_t MSD_GROUP_CUBE_BASE_BLOCK = 2048;

const std::set<Mc2WhiteListShape> MSD_GROUP_WHITE_LIST = {
    // JYXC
    {1, 12288, 1792, false, false, false, 1},
    {1, 12288, 7808, false, false, false, 1},
    {1, 3904, 12288, false, false, false, 1},
    {1, 1536, 12288, false, false, false, 1},
    {1, 1536, 12288, false, false, false, 1}};

ge::graphStatus Mc2WeightQuantBatchMatmulV2TilingMsdGroup::PostTiling()
{
    size_t tilingDataSize = sizeof(Mc2WeightQuantBatchMatmulV2MsdGroupTilingData);
    OP_LOGD(opName_, "final tiling data size: %zu", tilingDataSize);

    OP_TILING_CHECK(tilingDataSize % sizeof(uint64_t) != 0,
                    OP_LOGE(opName_, "tiling data size[%zu] not aligned to 8", tilingDataSize),
                    return ge::GRAPH_FAILED);

    context_->GetRawTilingData()->SetDataSize(tilingDataSize);
    uint32_t usedAicNum = tilingData_->cubeBlockDimN * tilingData_->cubeBlockDimK;
    uint32_t usedAivNum = std::max(static_cast<uint32_t>(matmulInfoPtr_->mSize), usedAicNum * 2);
    context_->SetBlockDim(CalcTschBlockDim(usedAivNum, compileInfoPtr_->aicNum, compileInfoPtr_->aivNum));
    errno_t ret = memcpy_s(context_->GetTilingData<Mc2WeightQuantBatchMatmulV2MsdGroupTilingData>(),
                           context_->GetRawTilingData()->GetCapacity(), tilingData_.get(), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

/*
The function is limite of msd-group
1. group_size support:64(int8), 64/128(int4)
2. m<=group_size/8; k<=13952, k%group_size=0; n<=aic_num*2048, n%64=0
3. not trans_a, not trans_b, c_dtype!=int8, antiquan_scale_dtype!=uint64
4. int8: jyxc white case; int4: L1 limit
*/
bool Mc2WeightQuantBatchMatmulV2TilingMsdGroup::IsCapable()
{
    OP_LOGI(opName_, "Begin check msd group");
    // MSD pergroup模板不支持确定性
    if (context_->GetDeterministic() == 1) {
        OP_LOGI(opName_, "MSD pergroup do not support deterministic");
        return false;
    }
    OP_TILING_CHECK(((matmulInfoPtr_->antiQuantScaleDtype == ge::DT_UINT64) ||
                     (matmulInfoPtr_->antiQuantScaleDtype == ge::DT_INT64)),
                    OP_LOGI(opName_, "MSD group do not support antiquant scale dtype is uint64"), return false);
    if (matmulInfoPtr_->bDtype == ge::DT_INT4) {
        OP_TILING_CHECK(matmulInfoPtr_->groupSize != 64 && matmulInfoPtr_->groupSize != 128,
                        OP_LOGI(opName_, "GroupSize support 64 or 128 for W4, bu is [%lu]", matmulInfoPtr_->groupSize),
                        return false);
    } else {
        OP_TILING_CHECK(matmulInfoPtr_->groupSize != 64,
                        OP_LOGI(opName_, "GroupSize support 64 for W8, bu is [%lu]", matmulInfoPtr_->groupSize),
                        return false);
    }
    // 防止N方向分的核数超过aicNum_   2048:N方向cube上切分SingleCoreN固定为2048
    uint64_t maxNSize = compileInfoPtr_->aicNum * 2048UL;
    OP_TILING_CHECK(matmulInfoPtr_->mSize > matmulInfoPtr_->groupSize / 8 || matmulInfoPtr_->kSize > 13952 ||
                        matmulInfoPtr_->nSize > maxNSize,
                    OP_LOGI(opName_, "m <= [%lu]/8, k <= [13952], n <= [%lu], but m is [%lu], k is[%lu] and n is [%lu]",
                            matmulInfoPtr_->groupSize, maxNSize, matmulInfoPtr_->mSize, matmulInfoPtr_->kSize,
                            matmulInfoPtr_->nSize),
                    return false);
    OP_TILING_CHECK(matmulInfoPtr_->kSize % matmulInfoPtr_->groupSize != 0 || matmulInfoPtr_->nSize % 64 != 0,
                    OP_LOGI(opName_,
                            "k should align to GroupSize[%lu], n should align to [64], but k is [%ld] and n is [%lu]",
                            matmulInfoPtr_->groupSize, matmulInfoPtr_->kSize, matmulInfoPtr_->nSize),
                    return false);
    OP_TILING_CHECK(matmulInfoPtr_->transA || matmulInfoPtr_->transB || matmulInfoPtr_->cDtype == ge::DT_INT8,
                    OP_LOGI(opName_, "MSD group not support trans_a, trans_b or quant"), return false);
    if (matmulInfoPtr_->bDtype == ge::DT_INT4) {
        OP_TILING_CHECK(!CheckL1Size(), OP_LOGI(opName_, "L1 size cannot meet the requirement for msd group"),
                        return false);
    } else {
        Mc2WhiteListShape shape({1, matmulInfoPtr_->kSize, matmulInfoPtr_->nSize, false, matmulInfoPtr_->transA,
                                 matmulInfoPtr_->transB, 1});
        if (MSD_GROUP_WHITE_LIST.find(shape) == MSD_GROUP_WHITE_LIST.end()) {
            OP_LOGI(opName_, "MSD group only support jyxc white case when weight dtype is int8.");
            return false;
        }
    }
    OP_LOGI(opName_, "Check msd group succ");
    return true;
}

bool Mc2WeightQuantBatchMatmulV2TilingMsdGroup::CheckL1Size() const
{
    // 当前n方向默认按照2048的基本块切分
    uint64_t cubeNMaxBlockDim = ops::CeilDiv(matmulInfoPtr_->nSize, 2048UL);
    uint64_t cubeKMaxBlockDim = (compileInfoPtr_->aicNum / std::max(1UL, cubeNMaxBlockDim));
    uint64_t cubeSingleCoreK = ops::CeilDiv(matmulInfoPtr_->kSize, cubeKMaxBlockDim);
    // k轴需要对齐到groupSize
    uint64_t cubeSingleCoreKAlign = ops::CeilAlign(cubeSingleCoreK, matmulInfoPtr_->groupSize);
    // m轴按照4次展开的形式(保持和int8一致)，对齐到32
    uint64_t cubeSingleCoreMAlign = ops::CeilAlign(4 * matmulInfoPtr_->mSize, 32UL);
    uint64_t al1Size = (cubeSingleCoreKAlign * cubeSingleCoreMAlign) >> 1;
    // n方向默认按照2048的基本块切分, bl1需要2份空间开pingpong,
    // 以group的维度完成matmul计算
    uint64_t bl1Size = (2 * 2048 * matmulInfoPtr_->groupSize) >> 1;

    return (al1Size + bl1Size <= aicoreParams_.l1Size);
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2TilingMsdGroup::InstantiateTilingData()
{
    size_t tilingDataSize = sizeof(Mc2WeightQuantBatchMatmulV2MsdGroupTilingData);
    if (tilingData_ == nullptr) {
        tilingData_ = std::unique_ptr<Mc2WeightQuantBatchMatmulV2MsdGroupTilingData>(
            new (std::nothrow) Mc2WeightQuantBatchMatmulV2MsdGroupTilingData());
    }
    OP_TILING_CHECK(tilingData_ == nullptr, OP_LOGE(opName_, "failed to instantiate tilingData"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetRawTilingData()->GetCapacity() < tilingDataSize,
                    OP_LOGE(opName_, "tiling data capacity %zu < actual tiling data size %zu",
                            context_->GetRawTilingData()->GetCapacity(), tilingDataSize),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2TilingMsdGroup::DoOpTiling()
{
    OP_TILING_CHECK(InstantiateTilingData() == ge::GRAPH_FAILED,
                    OP_LOGE(opName_, "unable to get pointer of tiling data"), return ge::GRAPH_FAILED);
    tilingData_->groupSize = matmulInfoPtr_->groupSize;
    tilingData_->kSize = matmulInfoPtr_->kSize;
    tilingData_->nSize = matmulInfoPtr_->nSize;
    tilingData_->mSize = matmulInfoPtr_->mSize;
    tilingData_->hasBias = matmulInfoPtr_->hasBias;

    // n方向，cube默认切分基本块为2048
    tilingData_->cubeBlockDimN = ops::CeilDiv(matmulInfoPtr_->nSize, static_cast<uint64_t>(MSD_GROUP_CUBE_BASE_BLOCK));
    // n方向，vector默认切分基本块为1024
    tilingData_->vecSingleCoreN = 1024;
    tilingData_->vecBlockDimN =
        ops::CeilAlign(ops::CeilDiv(matmulInfoPtr_->nSize, static_cast<uint64_t>(MSD_GROUP_VEC_BASE_BLOCK)), 2UL);

    tilingData_->singleCoreK =
        ops::CeilAlign(ops::CeilDiv(matmulInfoPtr_->kSize,
                                    compileInfoPtr_->aicNum / static_cast<uint64_t>(tilingData_->cubeBlockDimN)),
                       matmulInfoPtr_->groupSize);
    tilingData_->cubeBlockDimK = ops::CeilDiv(matmulInfoPtr_->kSize, static_cast<uint64_t>(tilingData_->singleCoreK));
    tilingData_->singleCoreGroup =
        ops::CeilDiv(static_cast<uint64_t>(tilingData_->singleCoreK), matmulInfoPtr_->groupSize);
    tilingData_->vec1SingleCoreM = ops::CeilDiv(matmulInfoPtr_->mSize, static_cast<uint64_t>(compileInfoPtr_->aivNum));
    OP_TILING_CHECK(!GetMatMulTiling(),
                    OP_LOGE(opName_, "failed to get mm tiling for mnk[%lu, %lu, %lu]", matmulInfoPtr_->mSize,
                            matmulInfoPtr_->nSize, matmulInfoPtr_->kSize),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void Mc2WeightQuantBatchMatmulV2TilingMsdGroup::Reset()
{
    OP_TILING_CHECK(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(), 0,
                             context_->GetRawTilingData()->GetCapacity()) != EOK,
                    OP_LOGE(opName_, "fail to memset tiling data"), return;);
}

bool Mc2WeightQuantBatchMatmulV2TilingMsdGroup::GetMatMulTiling()
{
    uint64_t cubeSingleCoreN = 2048;
    uint64_t iteratorTime = 2;
    if (matmulInfoPtr_->bDtype == ge::DT_INT4) {
        // int4场景，m需要展开3次
        iteratorTime = 3;
    }

    matmul_tiling::MatmulApiTiling mmTiling;
    mmTiling.SetAType(matmul_tiling::TPosition::A1, matmul_tiling::CubeFormat::ND,
                      Mc2GetMatmulTilingDtype(matmulInfoPtr_->bDtype), matmulInfoPtr_->transA);
    mmTiling.SetBType(matmul_tiling::TPosition::A1, matmul_tiling::CubeFormat::ND,
                      Mc2GetMatmulTilingDtype(matmulInfoPtr_->bDtype), matmulInfoPtr_->transB);
    mmTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);
    mmTiling.SetBias(false);
    mmTiling.SetOrgShape(iteratorTime * matmulInfoPtr_->mSize, cubeSingleCoreN, matmulInfoPtr_->kSize, cubeSingleCoreN);
    mmTiling.SetShape(iteratorTime * matmulInfoPtr_->mSize, cubeSingleCoreN, matmulInfoPtr_->groupSize);
    mmTiling.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize);
    mmTiling.SetFixSplit(ops::CeilAlign(static_cast<uint32_t>(iteratorTime * matmulInfoPtr_->mSize), BLOCK_CUBE),
                         BASIC_BLOCK, matmulInfoPtr_->groupSize);
    OP_TILING_CHECK(mmTiling.GetTiling(tilingData_->matmulTiling) == -1,
                    OP_LOGE(opName_, "failed to get matmul tiling"), return false);

    tilingData_->matmulTiling.shareL1Size = 0;
    tilingData_->matmulTiling.dbL0C = std::min(
        DOUBLE_BUFFER_FACTOR,
        ops::CeilDiv(aicoreParams_.l0cSize, static_cast<uint64_t>(tilingData_->matmulTiling.shareL0CSize) *
                                                tilingData_->matmulTiling.stepM * tilingData_->matmulTiling.stepN));
    return true;
}

} // namespace optiling
