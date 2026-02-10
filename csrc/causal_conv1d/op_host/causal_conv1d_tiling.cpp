/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_tiling.cpp
 * \brief
 */

// #include "error_log.h"
#include "log/ops_log.h"
#include "../tiling_base/tiling_templates_registry.h"
#include "../tiling_base/tiling_util.h"
#include "math_util.h"
#include "causal_conv1d_tiling.h"
#include "../op_kernel/causal_conv1d_tiling_key.h"

#include <set>
#include <limits>

namespace optiling {

using namespace Ops::Transformer::OpTiling;

constexpr uint32_t X_INDEX = 0;
constexpr uint32_t WEIGHT_INDEX = 1;
constexpr uint32_t BIAS_INDEX = 2;
constexpr uint32_t CONV_STATES_INDEX = 3;
constexpr uint32_t QUERY_START_LOC_INDEX = 4;
constexpr uint32_t CACHE_INDICES_INDEX = 5;
constexpr uint32_t HAS_INITIAL_STATE_INDEX = 6;

constexpr int32_t ATTR_ACTIVATION_MODE_INDEX = 0;
constexpr int32_t ATTR_PAD_SLOT_ID_INDEX = 1;



struct DimTileChoice {
    int64_t dimTileSize = 0;
    int64_t blocksPerSeq = 0;
    int64_t gridSize = 0;
};

static inline DimTileChoice ChooseDimTileSize(gert::TilingContext* context, int64_t batch, int64_t dim, uint32_t coreNum)
{

    const int64_t candidates[] = {4096, 2048, 1024, 512};
    DimTileChoice bestOver;
    int64_t bestOverGap = std::numeric_limits<int64_t>::max();
    DimTileChoice bestUnder;

    for (int64_t dimTileSize : candidates) {
        if (dim % dimTileSize != 0) {
            continue;
        }
        const int64_t blocksPerSeq = dim / dimTileSize;
        const int64_t gridSize = batch * blocksPerSeq;
        if (gridSize <= 0) {
            continue;
        }

        if (gridSize >= static_cast<int64_t>(coreNum)) {
            const int64_t gap = gridSize - static_cast<int64_t>(coreNum);
            if (gap < bestOverGap) {
                bestOver.dimTileSize = dimTileSize;
                bestOver.blocksPerSeq = blocksPerSeq;
                bestOver.gridSize = gridSize;
                bestOverGap = gap;
            }
        } else if (gridSize > bestUnder.gridSize ||
                   (gridSize == bestUnder.gridSize && dimTileSize < bestUnder.dimTileSize)) {
                bestUnder.dimTileSize = dimTileSize;
                bestUnder.blocksPerSeq = blocksPerSeq;
                bestUnder.gridSize = gridSize;
        }
    }
    DimTileChoice result = (bestOver.dimTileSize != 0) ? bestOver : bestUnder;

    return result;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, uint32_t& coreNum)
{
    auto compileInfoPtr = context->GetCompileInfo<CausalConv1dCompileInfo>();
    if (compileInfoPtr != nullptr && compileInfoPtr->coreNum != 0 && compileInfoPtr->ubSize != 0) {
        ubSize = compileInfoPtr->ubSize;
        coreNum = compileInfoPtr->coreNum;
        return ge::GRAPH_SUCCESS;
    }
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    if(coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if(ubSize == 0) {
         return ge::GRAPH_FAILED;
    }
     return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAttrsInfo(gert::TilingContext* context, int64_t& activationMode, int64_t& padSlotId)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t* activationModePtr = attrs->GetAttrPointer<int64_t>(ATTR_ACTIVATION_MODE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, activationModePtr);
    activationMode = *activationModePtr;
    if(activationMode != 0 && activationMode != 1){
        return ge::GRAPH_FAILED;
    }
    const int64_t* padSlotIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_PAD_SLOT_ID_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, padSlotIdPtr);
    padSlotId = *padSlotIdPtr;

    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus GetShapeDtypeInfo(gert::TilingContext* context, CausalConv1dTilingData& tiling)
{
    auto xShapePtr = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = EnsureNotScalar(xShapePtr->GetStorageShape());

    int64_t dim = 0;
    int64_t cuSeqlen = 0;
    int64_t seqLen = 0;
    int64_t batch = 0;
    int64_t inputMode = 0;

    if (xShape.GetDimNum() == 2) {
        inputMode = 0;
        cuSeqlen = xShape.GetDim(0);
        dim = xShape.GetDim(1);
        seqLen = 0;
        if(dim <= 0 || cuSeqlen < 0){
            return ge::GRAPH_FAILED;
        }
         
    } else if (xShape.GetDimNum() == 3) {
        inputMode = 1;
        batch = xShape.GetDim(0);
        seqLen = xShape.GetDim(1);
        dim = xShape.GetDim(2);
        cuSeqlen = batch * seqLen;
        if(batch <= 0 || dim <= 0 || seqLen <= 0){
            return ge::GRAPH_FAILED;
        }
    } else {
        return ge::GRAPH_FAILED;
    }

    auto wShapePtr = context->GetInputShape(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, wShapePtr);
    auto wShape = EnsureNotScalar(wShapePtr->GetStorageShape());
    if(wShape.GetDimNum() != 2){
        return ge::GRAPH_FAILED;
    } 
    const int64_t width = wShape.GetDim(0);
    const int64_t wDim = wShape.GetDim(1);
    if(wDim != dim){
        return ge::GRAPH_FAILED;
    }
    if(width != 4){
        return ge::GRAPH_FAILED;
    }
    if(dim != 1024 && dim != 2048 && dim != 4096 && dim != 8192){
        return ge::GRAPH_FAILED;
    }
    
                

    auto sShapePtr = context->GetInputShape(CONV_STATES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sShapePtr);
    auto sShape = EnsureNotScalar(sShapePtr->GetStorageShape());
    if(sShape.GetDimNum() != 3){
       return ge::GRAPH_FAILED;
    }
    const int64_t numCacheLines = sShape.GetDim(0);
    const int64_t stateLen = sShape.GetDim(1);
    const int64_t sDim = sShape.GetDim(2);
    if(numCacheLines <= 0){
         return ge::GRAPH_FAILED;}
    if(sDim != dim){
        return ge::GRAPH_FAILED;}
    if(stateLen < (width - 1)){
        return ge::GRAPH_FAILED;}

    auto qslShapePtr = context->GetInputShape(QUERY_START_LOC_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qslShapePtr);
    auto qslShape = EnsureNotScalar(qslShapePtr->GetStorageShape());
    if(qslShape.GetDimNum() != 1){
        return ge::GRAPH_FAILED;}
    const int64_t qslSize = qslShape.GetDim(0);
    if(qslSize < 1){
        return ge::GRAPH_FAILED;}

    if (inputMode == 0) {
        batch = qslSize - 1;
    }

    if (inputMode == 1) {
        if(qslSize != batch + 1){
            return ge::GRAPH_FAILED;
        }
    }

    auto ciShapePtr = context->GetInputShape(CACHE_INDICES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, ciShapePtr);
    auto ciShape = EnsureNotScalar(ciShapePtr->GetStorageShape());
    if(ciShape.GetDimNum() != 1){return ge::GRAPH_FAILED;}
    if(ciShape.GetDim(0) != batch){return ge::GRAPH_FAILED;}
    
    auto hisShapePtr = context->GetInputShape(HAS_INITIAL_STATE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, hisShapePtr);
    auto hisShape = EnsureNotScalar(hisShapePtr->GetStorageShape());
    if(hisShape.GetDimNum() != 1){
        return ge::GRAPH_FAILED;}
    if(hisShape.GetDim(0) != batch){
        return ge::GRAPH_FAILED;}

    tiling.set_hasBias(0);
    auto biasShapePtr = context->GetOptionalInputShape(BIAS_INDEX);
    if (biasShapePtr != nullptr && biasShapePtr->GetStorageShape().GetDimNum() != 0) {
        auto biasShape = EnsureNotScalar(biasShapePtr->GetStorageShape());
        if(biasShape.GetDimNum() != 1){
            return ge::GRAPH_FAILED;}
        if(biasShape.GetDim(0) != dim){
            return ge::GRAPH_FAILED;}
        tiling.set_hasBias(1);
    }

    const std::set<ge::DataType> supportedXDtype = {ge::DT_BF16, ge::DT_FLOAT16};
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const ge::DataType xDtype = xDesc->GetDataType();
    if(supportedXDtype.count(xDtype) == 0){
        return ge::GRAPH_FAILED;}

    auto wDesc = context->GetInputDesc(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, wDesc);
    if(wDesc->GetDataType() != xDtype){ 
        return ge::GRAPH_FAILED;}

    if (tiling.get_hasBias() == 1) {
        auto biasDesc = context->GetOptionalInputDesc(BIAS_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, biasDesc);
        if(biasDesc->GetDataType() != xDtype){
            return ge::GRAPH_FAILED;}
    }

    auto sDesc = context->GetInputDesc(CONV_STATES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sDesc);
    if(sDesc->GetDataType() != xDtype){
        return ge::GRAPH_FAILED;}

    auto qslDesc = context->GetInputDesc(QUERY_START_LOC_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qslDesc);
    if(qslDesc->GetDataType() != ge::DT_INT32){
        return ge::GRAPH_FAILED;}

    auto ciDesc = context->GetInputDesc(CACHE_INDICES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, ciDesc);
    if(ciDesc->GetDataType() != ge::DT_INT32){
        return ge::GRAPH_FAILED;}

    auto hisDesc = context->GetInputDesc(HAS_INITIAL_STATE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, hisDesc);
    if(hisDesc->GetDataType() != ge::DT_BOOL){
        return ge::GRAPH_FAILED;}

    tiling.set_dim(dim);
    tiling.set_cuSeqlen(cuSeqlen);
    tiling.set_seqLen(seqLen);
    tiling.set_inputMode(inputMode);
    tiling.set_width(width);
    tiling.set_stateLen(stateLen);
    tiling.set_numCacheLines(numCacheLines);
    tiling.set_batch(batch);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CausalConv1dTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    uint32_t coreNum;
    if( GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS){
        return ge::GRAPH_FAILED;
    }

    if(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS){
        return ge::GRAPH_FAILED;
    }
    CausalConv1dTilingData tilingData;

    int64_t activationMode = 0;
    int64_t padSlotId = -1;
    if(GetAttrsInfo(context, activationMode, padSlotId) != ge::GRAPH_SUCCESS){
        return ge::GRAPH_FAILED;
    }
    tilingData.set_activationMode(activationMode);
    tilingData.set_padSlotId(padSlotId);

    if( GetShapeDtypeInfo(context, tilingData) != ge::GRAPH_SUCCESS){
        return ge::GRAPH_FAILED;
    }

    const int64_t dim = tilingData.get_dim();
    const int64_t batch = tilingData.get_batch();
    if(dim <= 0 || batch <= 0){
        return ge::GRAPH_FAILED;
    }
    const DimTileChoice choice = ChooseDimTileSize(context, batch, dim, coreNum);
    const uint32_t blockDim = (choice.gridSize < static_cast<int64_t>(coreNum))
                                  ? static_cast<uint32_t>(choice.gridSize)
                                  : coreNum;
    context->SetBlockDim(blockDim);
    tilingData.set_dimTileSize(choice.dimTileSize);
    tilingData.set_blocksPerSeq(choice.blocksPerSeq);

    const uint64_t tilingKey = GET_TPL_TILING_KEY(CAUSAL_CONV1D_TPL_SCH_MODE_DEFAULT);
    context->SetTilingKey(tilingKey);

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}



static ge::graphStatus TilingParseForCausalConv1d(gert::TilingParseContext* context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto compileInfoPtr = context->GetCompiledInfo<CausalConv1dCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAiv());
    if(compileInfoPtr->coreNum == 0){
      return ge::GRAPH_FAILED;
    }
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    if(compileInfoPtr->ubSize == 0){
          return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CausalConv1d)
    .Tiling(CausalConv1dTilingFunc)
    .TilingParse<CausalConv1dCompileInfo>(TilingParseForCausalConv1d);
} // namespace optiling