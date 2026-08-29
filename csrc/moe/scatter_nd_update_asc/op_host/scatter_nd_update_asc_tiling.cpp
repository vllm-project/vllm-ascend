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
 * \file scatter_nd_update_asc_tiling.cpp
 * \brief
 */

#include "scatter_nd_update_asc_tiling.h"

namespace optiling {
constexpr int64_t DEFAULT_UB_FOR_ASCNENDC = 8192;
constexpr int64_t DEFAULT_GM_FOR_ASCNENDC = 16 * 1024 * 1024;
constexpr int64_t VEC_BLOCK_SIZE = 32;
constexpr int64_t INPUT_VAR_IDX = 0;
constexpr int64_t INPUT_INDICES_IDX = 1;
constexpr int64_t INPUT_UPDATE_IDX = 2;
constexpr int64_t OUTPUT_Y_IDX = 0;
constexpr int64_t INPUT_DIM_VALUE = 2;
constexpr int64_t VAR_MIN_DIM_VALUE = 2;
constexpr int64_t VAR_MAX_DIM_VALUE = 4;
constexpr int64_t DB_CONST = 2;
constexpr uint32_t DIM_0 = 0;
constexpr uint32_t DIM_1 = 1;
constexpr int64_t PART_CORE_C_THREAD = 256;
constexpr int64_t PART_CORE_NUM = 16;
constexpr uint64_t ATTR_STRIDE = 0;


ge::graphStatus ScatterNdUpdateAscTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr, OPS_LOG_E(context_->GetNodeName(), "get platformInfo nullptr."),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OPS_ERR_IF(
        coreNum_ <= 0, OPS_LOG_E(context_->GetNodeName(), "coreNum must be greater than 0."),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSize_ = static_cast<int64_t>(ubSizePlatForm);
    OPS_ERR_IF(
        ubSize_ <= 0, OPS_LOG_E(context_->GetNodeName(), "ubSize must be greater than 0."),
        return ge::GRAPH_FAILED);

    // DELETE UB FOR ASCENDC
    ubSize_ = ubSize_ - DEFAULT_UB_FOR_ASCNENDC;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateAscTiling::GetShapeInfo()
{
    OPS_ERR_IF(
        context_ == nullptr, OPS_LOG_E("ScatterNdUpdateAscTiling", "context can not be nullptr."),
        return ge::GRAPH_FAILED);

    if (GetInputShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // dtype校验
    if (GetInputDtypeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // var首轴stride解析（支持首轴非连续，依赖b_），与 ops-nn a2(arch22) 方案一致
    if (HandleViewStride() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateAscTiling::HandleViewStride()
{
    // 与 ops-nn a2(arch22) 的 HandleViewStride 语义一致：
    // firstDimStrideRows = (indexDim > 1) ? indicesMask[0] : 1
    // 本算子 indexDim 仅 1/2 两种，indexDim==2 时按连续处理不走 view，故 firstDimStrideRows 恒为 1
    firstDimStrideRows_ = 1;
    uint64_t stride0Expected = static_cast<uint64_t>(b_) * firstDimStrideRows_;

    // 默认按连续处理
    isViewStride0_ = 0;
    varStride0Elements_ = stride0Expected;
    int64_t stride0 = static_cast<int64_t>(stride0Expected);

    // 不依赖 GetInputStride：torch_npu 环境下 GE 不会把非连续切片的 stride 数组下传，
    // GetInputStride 返回的 strideDimNum 为 0。故改为由 torch 侧显式传入 var.strides()
    // 属性（与 scatter_nd_update_v2 一致），tiling 从属性中读取真实 stride。
    auto attrs = context_->GetAttrs();
    auto stridesPtr = attrs->GetListInt(ATTR_STRIDE);
    if (stridesPtr != nullptr && stridesPtr->GetSize() == varDimNum_) {
        stride0 = stridesPtr->GetData()[DIM_0];

        // 除被索引的前 indexDim 维外，其余轴必须连续（kernel 按整块展平拷贝，
        // b_ = 后 (varDimNum_-indexDim_) 维乘积）。只有首轴可非连续（indexDim==1 时）。
        // 对 var [d0, d1, ..., dk]，dim i (i>=indexDim_) 的连续期望 stride 为 var 第 i 维之后各维的乘积。
        for (int64_t i = indexDim_; i < varDimNum_; ++i) {
            int64_t expectedStride = 1;
            for (int64_t j = i + 1; j < varDimNum_; ++j) {
                expectedStride *= varDims_[j];
            }
            if (varDims_[i] > 1 && stridesPtr->GetData()[i] != expectedStride) {
                OPS_LOG_E(context_->GetNodeName(),
                          "var dim%ld must be contiguous, but got stride %ld, expected %ld. Only dim0 may be non-contiguous.",
                          i, stridesPtr->GetData()[i], expectedStride);
                return ge::GRAPH_FAILED;
            }
        }

        // 仅当实际首轴 stride 严格大于连续期望时才判定为 view；
        // 该判定仅对 indexDim==1 有意义（dim0 与行长 b_ 直接对应）。
        // indexDim==2（DSA block+offset）按连续处理：linearRow 已含 d1 因子，不能套用该 view 逻辑。
        if (indexDim_ == 1 && static_cast<uint64_t>(stride0) > stride0Expected) {
            isViewStride0_ = 1;
            varStride0Elements_ = static_cast<uint64_t>(stride0);
        }
    }

    OPS_LOG_D(context_->GetNodeName(), "var isViewStride0=%lu, varStride0Elements=%lu, firstDimStrideRows=%lu, "
                                       "stride0Expected=%lu.", isViewStride0_, varStride0Elements_, firstDimStrideRows_,
              stride0Expected);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateAscTiling::GetInputShapeInfo()
{
    // var 支持 2~4 维：[d0, d1, ..., dk]。indices 仍固定 [n,1]（单行号 scatter），
    // 后 (k) 维展平作为单行拷贝长度 b_（scatterLength），update 与 var 后 k 维一致。
    auto varInput = context_->GetInputShape(INPUT_VAR_IDX);
    OPS_ERR_IF(varInput == nullptr, OPS_LOG_E(context_->GetNodeName(), "get varInput nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape varShape = varInput->GetStorageShape();
    int64_t dimsN = varShape.GetDimNum();
    OPS_ERR_IF((dimsN < VAR_MIN_DIM_VALUE || dimsN > VAR_MAX_DIM_VALUE),
        OPS_LOG_E(context_->GetNodeName(), "varInput dim:%ld should be in [%ld, %ld].", dimsN,
                  VAR_MIN_DIM_VALUE, VAR_MAX_DIM_VALUE),
        return ge::GRAPH_FAILED);
    varDimNum_ = dimsN;
    for (int64_t i = 0; i < varDimNum_; ++i) {
        varDims_[i] = varShape.GetDim(i);
    }
    a_ = varShape.GetDim(DIM_0);

    auto indicesInput = context_->GetInputShape(INPUT_INDICES_IDX);
    OPS_ERR_IF(indicesInput == nullptr, OPS_LOG_E(context_->GetNodeName(), "get indicesInput nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape indicesShape = indicesInput->GetStorageShape();
    dimsN = indicesShape.GetDimNum();
    OPS_ERR_IF((dimsN != INPUT_DIM_VALUE),
        OPS_LOG_E(context_->GetNodeName(), "indicesInput dim:%ld should be 2.", dimsN),
        return ge::GRAPH_FAILED);
    c_ = indicesShape.GetDim(DIM_0);
    int64_t indicseRank = indicesShape.GetDim(DIM_1);
    // indexDim（scatter_nd 的 K）支持 1/2：
    //   K=1: indices [n,1] 单行号 scatter，update [n, d1..dk]，b_ = d1*...*dk
    //   K=2: indices [n,2] (block, offset) 二维坐标，update [n, d2..dk]，b_ = d2*...*dk
    OPS_ERR_IF((indicseRank != 1 && indicseRank != 2),
        OPS_LOG_E(context_->GetNodeName(), "indicesInput dim1:%ld should be 1 or 2.", indicseRank),
        return ge::GRAPH_FAILED);
    indexDim_ = indicseRank;

    // 行长 b_ = var 后 (varDimNum_-indexDim_) 维乘积（展平）
    b_ = 1;
    for (int64_t i = indexDim_; i < varDimNum_; ++i) {
        b_ *= varShape.GetDim(i);
    }
    OPS_ERR_IF((b_ <= 0),
        OPS_LOG_E(context_->GetNodeName(), "varInput trailing dims product:%ld should be > 0.", b_),
        return ge::GRAPH_FAILED);
    // K=2 时 linearRow = indices[i,0]*varDims_[1] + indices[i,1] 需要 var dim1
    varDim1_ = varDimNum_ > 1 ? varDims_[1] : 1;
    
    auto updateInput = context_->GetInputShape(INPUT_UPDATE_IDX);
    OPS_ERR_IF(updateInput == nullptr, OPS_LOG_E(context_->GetNodeName(), "get updateInput nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape updateShape = updateInput->GetStorageShape();
    dimsN = updateShape.GetDimNum();
    OPS_ERR_IF((dimsN < VAR_MIN_DIM_VALUE || dimsN > VAR_MAX_DIM_VALUE),
        OPS_LOG_E(context_->GetNodeName(), "updateInput dim:%ld should be in [%ld, %ld].", dimsN,
                  VAR_MIN_DIM_VALUE, VAR_MAX_DIM_VALUE),
        return ge::GRAPH_FAILED);
    int64_t updateC = updateShape.GetDim(DIM_0);
    OPS_ERR_IF((updateC != c_),
        OPS_LOG_E(context_->GetNodeName(), "indicesInput dim0:%ld  should be same as update dim0:%ld", updateC, c_),
        return ge::GRAPH_FAILED);
    // update 后 (dimsN-1) 维乘积需与 var 行长 b_ 一致
    int64_t updateB = 1;
    for (int64_t i = 1; i < dimsN; ++i) {
        updateB *= updateShape.GetDim(i);
    }
    OPS_ERR_IF((updateB != b_),
        OPS_LOG_E(context_->GetNodeName(), "updateInput trailing dims product:%ld  should be same as var b:%ld", updateB, b_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateAscTiling::GetInputDtypeInfo()
{
    auto varDesc = context_->GetInputDesc(INPUT_VAR_IDX);
    OPS_ERR_IF(varDesc == nullptr, OPS_LOG_E(context_->GetNodeName(), "get varDesc nullptr."),
        return ge::GRAPH_FAILED);
    auto varDtype = varDesc->GetDataType();
    OPS_ERR_IF(
        (varDtype != ge::DT_FLOAT16 && varDtype != ge::DT_BF16 && varDtype != ge::DT_INT8),
        OPS_LOG_E(context_->GetNodeName(), "varDtype is not supported."),
        return ge::GRAPH_FAILED);
    varDtypeSize_ = varDtype == ge::DT_INT8 ? sizeof(int8_t) : sizeof(uint16_t);
    int64_t bBlockSize = VEC_BLOCK_SIZE / varDtypeSize_;
    bAlign_ = (b_ + bBlockSize - 1) / bBlockSize * bBlockSize;

    auto indicesDesc = context_->GetInputDesc(INPUT_INDICES_IDX);
    OPS_ERR_IF(indicesDesc == nullptr, OPS_LOG_E(context_->GetNodeName(), "get indicesDesc nullptr."),
        return ge::GRAPH_FAILED);
    ge::DataType indicesDtype = indicesDesc->GetDataType();
    OPS_ERR_IF(
        (indicesDtype != ge::DT_INT32 && indicesDtype != ge::DT_INT64),
        OPS_LOG_E(context_->GetNodeName(), "indicesDtype is not supported."),
        return ge::GRAPH_FAILED);
    indicesDtypeSize_ = indicesDtype == ge::DT_INT32 ? sizeof(int32_t) : sizeof(int64_t);

    auto updateDesc = context_->GetInputDesc(INPUT_UPDATE_IDX);
    OPS_ERR_IF(updateDesc == nullptr, OPS_LOG_E(context_->GetNodeName(), "get updateDesc nullptr."),
        return ge::GRAPH_FAILED);
    ge::DataType updateDtype = updateDesc->GetDataType();
    OPS_ERR_IF(
        (updateDtype != varDtype),
        OPS_LOG_E(context_->GetNodeName(), "updateDtype should same with varDtype."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateAscTiling::DoOpTiling()
{
    // block_factor
    if (c_ <= PART_CORE_C_THREAD) {
        coreNum_ = PART_CORE_NUM;
    }
    blockFactor_ = (c_ + coreNum_ - 1) / coreNum_;
    blockNum_ = (c_ + blockFactor_ - 1) / blockFactor_;
    blockFactorTail_ = c_ - (blockNum_ - 1) * blockFactor_;

    // ub_factor condtion
    // ub only constains update
    ubFactor_ = ubSize_ / DB_CONST / bAlign_ / varDtypeSize_;
    OPS_ERR_IF(
    (ubFactor_ < 1),
    OPS_LOG_E(context_->GetNodeName(), "update length to long %ld, not support.", b_),
    return ge::GRAPH_FAILED);
    ubFactor_ = ubFactor_ > blockFactor_ ? blockFactor_ : ubFactor_;

     
    tilingData_.set_a(a_);
    tilingData_.set_b(b_);
    tilingData_.set_bAlign(bAlign_);
    tilingData_.set_c(c_);
    tilingData_.set_blockFactor(blockFactor_);
    tilingData_.set_blockFactorTail(blockFactorTail_);
    tilingData_.set_ubFactor(ubFactor_);
    tilingData_.set_blockNum(blockNum_);
    tilingData_.set_isViewStride0(isViewStride0_);
    tilingData_.set_varStride0Elements(varStride0Elements_);
    tilingData_.set_firstDimStrideRows(firstDimStrideRows_);
    tilingData_.set_indexDim(indexDim_);
    tilingData_.set_varDim1(varDim1_);

    OPS_LOG_I(context_->GetNodeName(), "TilingData ScatterNdUpdateAsc a=%ld, b=%ld, bAlign=%ld, c=%ld, indexDim=%ld, varDim1=%ld, blockFactor=%ld, blockFactorTail=%ld, ubFactor=%ld, blockNum=%ld, isViewStride0=%lu, varStride0Elements=%lu, firstDimStrideRows=%lu.",
     a_, b_, bAlign_, c_, indexDim_, varDim1_, blockFactor_, blockFactorTail_, ubFactor_, blockNum_, isViewStride0_, varStride0Elements_, firstDimStrideRows_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateAscTiling::PostTiling()
{
    context_->SetTilingKey(0);
    context_->SetBlockDim(blockNum_);
    auto workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = DEFAULT_GM_FOR_ASCNENDC;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    OPS_LOG_I(context_->GetNodeName(), "TilingForScatterNdUpdateAsc leaving.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateAscTiling::RunTiling()
{
    ge::graphStatus ret = GetShapeInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = GetPlatformInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = DoOpTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return PostTiling();
}

ge::graphStatus Tiling4ScatterNdUpdateAsc(gert::TilingContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "TilingForScatterNdUpdateAsc running.");
    OPS_ERR_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("TilingForScatterNdUpdateAsc", "Tiling context is null"),
               return ge::GRAPH_FAILED);
    ScatterNdUpdateAscTiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepare4ScatterNdUpdateAsc(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ScatterNdUpdateAsc)
    .Tiling(Tiling4ScatterNdUpdateAsc)
    .TilingParse<ScatterNdUpdateAscCompileInfo>(TilingPrepare4ScatterNdUpdateAsc);

} // namespace optiling