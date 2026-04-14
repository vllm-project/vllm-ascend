/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_v310_tiling.cpp
 * \brief
 */

#include "math_util.h"
#include "../tiling_base/tiling_templates_registry.h"
#include "../tiling_base/tiling_util.h"
#include "../op_kernel/causal_conv1d_v310_tiling_data.h"
#include "../op_kernel/causal_conv1d_v310_tiling_key.h"

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
constexpr uint32_t INITIAL_STATE_MODE_INDEX = 6;
constexpr uint32_t NUM_ACCEPTED_TOKENS_INDEX = 7;

constexpr int32_t ATTR_ACTIVATION_MODE_INDEX = 0;
constexpr int32_t ATTR_PAD_SLOT_ID_INDEX = 1;
constexpr int32_t ATTR_RUN_MODE_INDEX = 2;

struct CausalConv1dCompileInfo {
    uint64_t ubSize = 0;
    uint32_t coreNum = 0;
};

struct DimTileChoice {
    int64_t dimTileSize = 0;
    int64_t blocksPerSeq = 0;
    int64_t gridSize = 0;
};

static inline int64_t CeilDivInt64(int64_t x, int64_t y)
{
    return (x + y - 1) / y;
}

static inline bool FitsInInt32(int64_t v)
{
    return v >= static_cast<int64_t>(std::numeric_limits<int32_t>::min()) &&
           v <= static_cast<int64_t>(std::numeric_limits<int32_t>::max());
}

static inline DimTileChoice ChooseDimTileSize(gert::TilingContext *context, int64_t batch, int64_t dim,
                                              uint32_t coreNum)
{
    const int64_t candidates[] = {4096, 2048, 1024, 512, 384, 192};

    auto ChooseOnce = [&](bool requireExactDiv) -> DimTileChoice {
        DimTileChoice bestOver;
        int64_t bestOverGap = std::numeric_limits<int64_t>::max();
        DimTileChoice bestUnder;

        for (int64_t dimTileSize : candidates) {
            if (dimTileSize <= 0) {
                continue;
            }
            if (requireExactDiv && (dim % dimTileSize != 0)) {
                continue;
            }
            const int64_t blocksPerSeq = requireExactDiv ? (dim / dimTileSize) : CeilDivInt64(dim, dimTileSize);
            const int64_t gridSize = batch * blocksPerSeq;
            if (gridSize <= 0) {
                continue;
            }
            OP_LOGD(context, "DimTile candidate[%s]: dimTileSize[%ld], blocksPerSeq[%ld], gridSize[%ld], coreNum[%u].",
                    requireExactDiv ? "exact" : "tail", dimTileSize, blocksPerSeq, gridSize, coreNum);
            if (gridSize >= static_cast<int64_t>(coreNum)) {
                const int64_t gap = gridSize - static_cast<int64_t>(coreNum);
                if (gap < bestOverGap) {
                    //                    bestOver = {dimTileSize, blocksPerSeq, gridSize};
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
        return (bestOver.dimTileSize != 0) ? bestOver : bestUnder;
    };

    DimTileChoice result = ChooseOnce(true /*requireExactDiv*/);
    if (result.dimTileSize == 0) {
        result = ChooseOnce(false /*requireExactDiv*/);
    }
    OP_LOGD(context, "DimTile chosen: dimTileSize[%ld], blocksPerSeq[%ld], gridSize[%ld].", result.dimTileSize,
            result.blocksPerSeq, result.gridSize);
    return result;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext *context, uint64_t &ubSize, uint32_t &coreNum)
{
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext *context)
{
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAttrsInfo(gert::TilingContext *context, int64_t &activationMode, int64_t &padSlotId,
                                    int64_t &runMode)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t *activationModePtr = attrs->GetAttrPointer<int64_t>(ATTR_ACTIVATION_MODE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, activationModePtr);
    activationMode = *activationModePtr;
    OP_CHECK_IF(activationMode != 0 && activationMode != 1, OP_LOGE(context, "activationMode only supports 0/1"),
                return ge::GRAPH_FAILED);

    const int64_t *padSlotIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_PAD_SLOT_ID_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, padSlotIdPtr);
    padSlotId = *padSlotIdPtr;

    const int64_t *runModePtr = attrs->GetAttrPointer<int64_t>(ATTR_RUN_MODE_INDEX);
    runMode = (runModePtr == nullptr) ? 0 : *runModePtr;
    OP_CHECK_IF(runMode != 0 && runMode != 1, OP_LOGE(context, "runMode only supports 0/1"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeDtypeInfo(gert::TilingContext *context, CausalConv1dTilingData &tiling)
{
    const bool isDecodeMode = (tiling.runMode == 1);

    auto xShapePtr = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = EnsureNotScalar(xShapePtr->GetStorageShape());

    int64_t dim = 0;
    int64_t cuSeqlen = 0;
    int64_t seqLen = 0;
    int64_t batch = 0;
    int64_t inputMode = 0;

    if (xShape.GetDimNum() == 2) {
        if (isDecodeMode) {
            inputMode = 2;
            batch = xShape.GetDim(0);
            dim = xShape.GetDim(1);
            seqLen = 1;
            cuSeqlen = batch;
            OP_CHECK_IF(batch <= 0 || dim <= 0, OP_LOGE(context, "invalid x shape for 2D decode mode"),
                        return ge::GRAPH_FAILED);
        } else {
            inputMode = 0;
            cuSeqlen = xShape.GetDim(0);
            dim = xShape.GetDim(1);
            seqLen = 0;
            OP_CHECK_IF(dim <= 0 || cuSeqlen < 0, OP_LOGE(context, "invalid x shape for 2D varlen mode"),
                        return ge::GRAPH_FAILED);
        }
    } else if (xShape.GetDimNum() == 3) {
        inputMode = 1;
        batch = xShape.GetDim(0);
        seqLen = xShape.GetDim(1);
        dim = xShape.GetDim(2);
        cuSeqlen = batch * seqLen;
        OP_CHECK_IF(batch <= 0 || dim <= 0 || seqLen <= 0, OP_LOGE(context, "invalid x shape for 3D batch mode"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(context, "x must be 2D (cu_seqlen, dim) or 3D (batch, seqlen, dim)");
        return ge::GRAPH_FAILED;
    }

    auto wShapePtr = context->GetInputShape(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, wShapePtr);
    auto wShape = EnsureNotScalar(wShapePtr->GetStorageShape());
    OP_CHECK_IF(wShape.GetDimNum() != 2, OP_LOGE(context, "weight must be 2D: (width, dim)"), return ge::GRAPH_FAILED);
    const int64_t width = wShape.GetDim(0);
    const int64_t wDim = wShape.GetDim(1);
    OP_CHECK_IF(wDim != dim, OP_LOGE(context, "weight.shape[1] must equal dim"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(width < 2 || width > 4, OP_LOGE(context, "Only support width in [2,4] now, actually is %ld.", width),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dim % 16 != 0,
                OP_LOGE(context, "dim must be a multiple of 16 for fp16/bf16 alignment, actually is %ld.", dim),
                return ge::GRAPH_FAILED);

    auto sShapePtr = context->GetInputShape(CONV_STATES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sShapePtr);
    auto sShape = EnsureNotScalar(sShapePtr->GetStorageShape());
    OP_CHECK_IF(sShape.GetDimNum() != 3, OP_LOGE(context, "convStates must be 3D: (num_cache_lines, state_len, dim)"),
                return ge::GRAPH_FAILED);
    const int64_t numCacheLines = sShape.GetDim(0);
    const int64_t stateLen = sShape.GetDim(1);
    const int64_t sDim = sShape.GetDim(2);
    OP_CHECK_IF(numCacheLines <= 0, OP_LOGE(context, "convStates.shape[0] (num_cache_lines) must be > 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(sDim != dim, OP_LOGE(context, "convStates.shape[2] must equal dim"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(stateLen < (width - 1), OP_LOGE(context, "convStates.shape[1] must be >= width-1"),
                return ge::GRAPH_FAILED);

    auto qslShapePtr = context->GetOptionalInputShape(QUERY_START_LOC_INDEX);
    const gert::CompileTimeTensorDesc *qslDesc = context->GetOptionalInputDesc(QUERY_START_LOC_INDEX);
    bool qslAbsent = true;
    int64_t qslSize = 0;
    if (qslShapePtr != nullptr) {
        const auto qslStorageShape = qslShapePtr->GetStorageShape();
        const int64_t qslDimNum = qslStorageShape.GetDimNum();
        qslAbsent = (qslDimNum == 0) || (qslDimNum == 1 && qslStorageShape.GetDim(0) <= 0);

        if (!qslAbsent) {
            auto qslShape = EnsureNotScalar(qslStorageShape);
            OP_CHECK_IF(qslShape.GetDimNum() != 1, OP_LOGE(context, "queryStartLoc must be 1D"),
                        return ge::GRAPH_FAILED);
            qslSize = qslShape.GetDim(0);
            OP_CHECK_IF(qslSize < 1, OP_LOGE(context, "queryStartLoc.size must be >= 1"), return ge::GRAPH_FAILED);

            OP_CHECK_NULL_WITH_CONTEXT(context, qslDesc);
            OP_CHECK_IF(qslDesc->GetDataType() != ge::DT_INT64, OP_LOGE(context, "queryStartLoc dtype must be int64"),
                        return ge::GRAPH_FAILED);
        }
    }

    if (qslAbsent) {
        OP_CHECK_IF(inputMode == 0, OP_LOGE(context, "queryStartLoc is required in 2D varlen mode (inputMode=0)"),
                    return ge::GRAPH_FAILED);
        qslSize = batch + 1;
    }

    OP_CHECK_IF(cuSeqlen > static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
                OP_LOGE(context, "cuSeqlen is too large for int32 indexing, got %ld", cuSeqlen),
                return ge::GRAPH_FAILED);

    const int64_t *qslData = nullptr;
    // if (!qslAbsent) {
    //     const gert::Tensor *qslTensor = context->GetOptionalInputTensor(QUERY_START_LOC_INDEX);
    //     qslData = (qslTensor != nullptr) ? qslTensor->GetData<int64_t>() : nullptr;
    //     if (qslData != nullptr) {
    //         OP_CHECK_IF(qslData[0] != 0, OP_LOGE(context, "queryStartLoc[0] must be 0"), return ge::GRAPH_FAILED);
    //         OP_CHECK_IF(qslData[qslSize - 1] != cuSeqlen,
    //                     OP_LOGE(context, "queryStartLoc[last] must equal cuSeqlen, got %ld vs %ld",
    //                             qslData[qslSize - 1], cuSeqlen),
    //                     return ge::GRAPH_FAILED);
    //         for (int64_t i = 0; i + 1 < qslSize; ++i) {
    //             const int64_t cur = qslData[i];
    //             const int64_t nxt = qslData[i + 1];
    //             OP_CHECK_IF(cur < 0 || cur > cuSeqlen,
    //                         OP_LOGE(context, "queryStartLoc[%ld] out of range: %ld (cuSeqlen=%ld)", i, cur, cuSeqlen),
    //                         return ge::GRAPH_FAILED);
    //             OP_CHECK_IF(
    //                 nxt < 0 || nxt > cuSeqlen,
    //                 OP_LOGE(context, "queryStartLoc[%ld] out of range: %ld (cuSeqlen=%ld)", i + 1, nxt, cuSeqlen),
    //                 return ge::GRAPH_FAILED);
    //             OP_CHECK_IF(
    //                 nxt < cur,
    //                 OP_LOGE(context,
    //                         "queryStartLoc must be non-decreasing, got queryStartLoc[%ld]=%ld queryStartLoc[%ld]=%ld",
    //                         i, cur, i + 1, nxt),
    //                 return ge::GRAPH_FAILED);
    //         }
    //     }
    // }

    if (!qslAbsent && isDecodeMode && inputMode == 2) {
        const int64_t batchFromQsl = qslSize - 1;
        if (batchFromQsl != batch) {
            inputMode = 0;
            cuSeqlen = xShape.GetDim(0);
            batch = batchFromQsl;
            seqLen = 0;
            OP_CHECK_IF(dim <= 0 || cuSeqlen < 0 || batch < 0,
                        OP_LOGE(context, "invalid x/queryStartLoc shapes for 2D varlen decode mode"),
                        return ge::GRAPH_FAILED);
        }
    }

    if (inputMode == 0) {
        batch = qslSize - 1;
    }

    if (!qslAbsent && (inputMode == 1 || inputMode == 2)) {
        OP_CHECK_IF(qslSize != batch + 1, OP_LOGE(context, "queryStartLoc.size must equal batch + 1"),
                    return ge::GRAPH_FAILED);
    }

    if (isDecodeMode) {
        const int64_t decodeSeqLen = (inputMode == 1) ? seqLen : 1;
        OP_CHECK_IF(decodeSeqLen < 1, OP_LOGE(context, "decode mode requires seqlen >= 1, actual is %ld", decodeSeqLen),
                    return ge::GRAPH_FAILED);
    }

    tiling.hasCacheIndices = 0;
    bool ciAbsent = true;
    auto ciShapePtr = context->GetOptionalInputShape(CACHE_INDICES_INDEX);
    if (ciShapePtr != nullptr) {
        const auto ciStorageShape = ciShapePtr->GetStorageShape();
        const int64_t ciDimNum = ciStorageShape.GetDimNum();
        ciAbsent = (ciDimNum == 0) || (ciDimNum == 1 && ciStorageShape.GetDim(0) <= 0);
        if (!ciAbsent) {
            auto ciShape = EnsureNotScalar(ciStorageShape);
            OP_CHECK_IF(ciShape.GetDimNum() != 1, OP_LOGE(context, "cacheIndices must be 1D"), return ge::GRAPH_FAILED);
            OP_CHECK_IF(ciShape.GetDim(0) != batch, OP_LOGE(context, "cacheIndices.size must equal batch"),
                        return ge::GRAPH_FAILED);
            tiling.hasCacheIndices = 1;

            // const gert::Tensor *ciTensor = context->GetOptionalInputTensor(CACHE_INDICES_INDEX);
            // const int64_t *ciData = (ciTensor != nullptr) ? ciTensor->GetData<int64_t>() : nullptr;
            // if (ciData != nullptr) {
            //     for (int64_t i = 0; i < batch; ++i) {
            //         const int64_t v = ciData[i];
            //         if (v == tiling.padSlotId) {
            //             continue;
            //         }
            //         OP_CHECK_IF(!FitsInInt32(v), OP_LOGE(context, "cacheIndices[%ld]=%ld does not fit int32", i, v),
            //                     return ge::GRAPH_FAILED);
            //         OP_CHECK_IF(
            //             v < 0 || v >= numCacheLines,
            //             OP_LOGE(context, "cacheIndices[%ld]=%ld out of range [0, num_cache_lines=%ld), padSlotId=%ld",
            //                     i, v, numCacheLines, tiling.padSlotId),
            //             return ge::GRAPH_FAILED);
            //     }
            // }
        }
    }
    if (ciAbsent) {
        OP_CHECK_IF(numCacheLines < batch,
                    OP_LOGE(context,
                            "cacheIndices is absent, requires convStates.shape[0] (num_cache_lines) >= batch for "
                            "identity mapping, got num_cache_lines=%ld batch=%ld",
                            numCacheLines, batch),
                    return ge::GRAPH_FAILED);
    }

    tiling.hasInitialStateMode = 0;
    auto ismShapePtr = context->GetOptionalInputShape(INITIAL_STATE_MODE_INDEX);
    if (ismShapePtr != nullptr) {
        const auto ismStorageShape = ismShapePtr->GetStorageShape();
        const int64_t ismDimNum = ismStorageShape.GetDimNum();
        const bool ismAbsent = (ismDimNum == 0) || (ismDimNum == 1 && ismStorageShape.GetDim(0) <= 0);
        if (!ismAbsent) {
            auto ismShape = EnsureNotScalar(ismStorageShape);
            OP_CHECK_IF(ismShape.GetDimNum() != 1, OP_LOGE(context, "initialStateMode must be 1D"),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(ismShape.GetDim(0) != batch, OP_LOGE(context, "initialStateMode.size must equal batch"),
                        return ge::GRAPH_FAILED);
            tiling.hasInitialStateMode = 1;

            // const gert::Tensor *ismTensor = context->GetOptionalInputTensor(INITIAL_STATE_MODE_INDEX);
            // const int64_t *ismData = (ismTensor != nullptr) ? ismTensor->GetData<int64_t>() : nullptr;
            // if (ismData != nullptr) {
            //     for (int64_t i = 0; i < batch; ++i) {
            //         const int64_t v = ismData[i];
            //         OP_CHECK_IF(v != 0 && v != 1,
            //                     OP_LOGE(context, "initialStateMode[%ld]=%ld is invalid (only supports 0/1)", i, v),
            //                     return ge::GRAPH_FAILED);
            //     }
            // }
        }
    }

    tiling.hasNumAcceptedTokens = 0;
    auto natShapePtr = context->GetOptionalInputShape(NUM_ACCEPTED_TOKENS_INDEX);
    if (natShapePtr != nullptr) {
        const auto natStorageShape = natShapePtr->GetStorageShape();
        const int64_t natDimNum = natStorageShape.GetDimNum();
        const bool natAbsent = (natDimNum == 0) || (natDimNum == 1 && natStorageShape.GetDim(0) <= 0);
        if (!natAbsent) {
            OP_CHECK_IF(!isDecodeMode,
                        OP_LOGE(context, "numAcceptedTokens is only supported in runMode=1 (decode/update)"),
                        return ge::GRAPH_FAILED);
            auto natShape = EnsureNotScalar(natStorageShape);
            OP_CHECK_IF(natShape.GetDimNum() != 1, OP_LOGE(context, "numAcceptedTokens must be 1D"),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(natShape.GetDim(0) != batch, OP_LOGE(context, "numAcceptedTokens.size must equal batch"),
                        return ge::GRAPH_FAILED);

            if (inputMode == 1) {
                const int64_t reqStateLen = (width - 1) + (seqLen - 1);
                OP_CHECK_IF(stateLen < reqStateLen,
                            OP_LOGE(context,
                                    "spec decode requires stateLen >= (width-1) + (seqlen-1), got stateLen=%ld req=%ld",
                                    stateLen, reqStateLen),
                            return ge::GRAPH_FAILED);
            }

            // const gert::Tensor *natTensor = context->GetOptionalInputTensor(NUM_ACCEPTED_TOKENS_INDEX);
            // const int64_t *natData = (natTensor != nullptr) ? natTensor->GetData<int64_t>() : nullptr;
            // if (natData != nullptr) {
            //     for (int64_t i = 0; i < batch; ++i) {
            //         const int64_t a = natData[i];
            //         OP_CHECK_IF(a < 0, OP_LOGE(context, "numAcceptedTokens[%ld]=%ld is invalid (must be >= 0)", i, a),
            //                     return ge::GRAPH_FAILED);
            //         OP_CHECK_IF(!FitsInInt32(a),
            //                     OP_LOGE(context, "numAcceptedTokens[%ld]=%ld does not fit int32", i, a),
            //                     return ge::GRAPH_FAILED);

            //         if (inputMode == 2) {
            //             OP_CHECK_IF(
            //                 a > 1,
            //                 OP_LOGE(context, "numAcceptedTokens[%ld]=%ld exceeds decode 2D token count (1)", i, a),
            //                 return ge::GRAPH_FAILED);
            //         } else if (inputMode == 1) {
            //             OP_CHECK_IF(a > seqLen,
            //                         OP_LOGE(context, "numAcceptedTokens[%ld]=%ld exceeds seqlen=%ld in 3D update", i, a,
            //                                 seqLen),
            //                         return ge::GRAPH_FAILED);
            //         } else if (inputMode == 0) {
            //             if (qslData != nullptr) {
            //                 const int64_t lenI = qslData[i + 1] - qslData[i];
            //                 OP_CHECK_IF(a > lenI,
            //                             OP_LOGE(context, "numAcceptedTokens[%ld]=%ld exceeds varlen segment length=%ld",
            //                                     i, a, lenI),
            //                             return ge::GRAPH_FAILED);
            //             }
            //         }
            //     }
            // }

            tiling.hasNumAcceptedTokens = 1;
        }
    }

    tiling.hasBias = 0;
    auto biasShapePtr = context->GetOptionalInputShape(BIAS_INDEX);
    if (biasShapePtr != nullptr) {
        const auto biasStorageShape = biasShapePtr->GetStorageShape();
        const int64_t biasDimNum = biasStorageShape.GetDimNum();
        const bool biasAbsent = (biasDimNum == 0) || (biasDimNum == 1 && biasStorageShape.GetDim(0) <= 0);
        if (!biasAbsent) {
            auto biasShape = EnsureNotScalar(biasStorageShape);
            OP_CHECK_IF(biasShape.GetDimNum() != 1, OP_LOGE(context, "bias must be 1D: (dim,)"),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(biasShape.GetDim(0) != dim, OP_LOGE(context, "bias.size must equal dim"),
                        return ge::GRAPH_FAILED);
            tiling.hasBias = 1;
        }
    }

    const std::set<ge::DataType> supportedXDtype = {ge::DT_FLOAT16};
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const ge::DataType xDtype = xDesc->GetDataType();
    OP_CHECK_IF(supportedXDtype.count(xDtype) == 0, OP_LOGE(context, "x dtype only supports fp16"),
                return ge::GRAPH_FAILED);

    auto wDesc = context->GetInputDesc(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, wDesc);
    OP_CHECK_IF(wDesc->GetDataType() != xDtype, OP_LOGE(context, "weight dtype must equal x dtype"),
                return ge::GRAPH_FAILED);

    if (tiling.hasBias == 1) {
        auto biasDesc = context->GetOptionalInputDesc(BIAS_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, biasDesc);
        OP_CHECK_IF(biasDesc->GetDataType() != xDtype, OP_LOGE(context, "bias dtype must equal x dtype"),
                    return ge::GRAPH_FAILED);
    }

    auto sDesc = context->GetInputDesc(CONV_STATES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sDesc);
    OP_CHECK_IF(sDesc->GetDataType() != xDtype, OP_LOGE(context, "convStates dtype must equal x dtype"),
                return ge::GRAPH_FAILED);

    if (!qslAbsent) {
        auto qslDesc2 = context->GetOptionalInputDesc(QUERY_START_LOC_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, qslDesc2);
        OP_CHECK_IF(qslDesc2->GetDataType() != ge::DT_INT64, OP_LOGE(context, "queryStartLoc dtype must be int64"),
                    return ge::GRAPH_FAILED);
    }

    if (tiling.hasCacheIndices == 1) {
        auto ciDesc = context->GetOptionalInputDesc(CACHE_INDICES_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, ciDesc);
        OP_CHECK_IF(ciDesc->GetDataType() != ge::DT_INT64, OP_LOGE(context, "cacheIndices dtype must be int64"),
                    return ge::GRAPH_FAILED);
    }

    if (tiling.hasInitialStateMode == 1) {
        auto ismDesc = context->GetOptionalInputDesc(INITIAL_STATE_MODE_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, ismDesc);
        OP_CHECK_IF(ismDesc->GetDataType() != ge::DT_INT64, OP_LOGE(context, "initialStateMode dtype must be int64"),
                    return ge::GRAPH_FAILED);
    }

    if (tiling.hasNumAcceptedTokens == 1) {
        OP_CHECK_IF(width != 4, OP_LOGE(context, "numAcceptedTokens is only supported for width=4 currently"),
                    return ge::GRAPH_FAILED);
        auto natDesc = context->GetOptionalInputDesc(NUM_ACCEPTED_TOKENS_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, natDesc);
        OP_CHECK_IF(natDesc->GetDataType() != ge::DT_INT64, OP_LOGE(context, "numAcceptedTokens dtype must be int64"),
                    return ge::GRAPH_FAILED);
    }

    tiling.dim = dim;
    tiling.cuSeqlen = cuSeqlen;
    tiling.seqLen = seqLen;
    tiling.inputMode = inputMode;
    tiling.width = width;
    tiling.stateLen = stateLen;
    tiling.numCacheLines = numCacheLines;
    tiling.batch = batch;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CausalConv1dTilingFunc(gert::TilingContext *context)
{
    uint64_t ubSize;
    uint32_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);

    CausalConv1dTilingData *tiling = context->GetTilingData<CausalConv1dTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(CausalConv1dTilingData), 0, sizeof(CausalConv1dTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAttrsInfo(context, tiling->activationMode, tiling->padSlotId, tiling->runMode) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetAttrsInfo error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetShapeDtypeInfo(context, *tiling) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeDtypeInfo error"),
                return ge::GRAPH_FAILED);

    const int64_t dim = tiling->dim;
    const int64_t batch = tiling->batch;
    OP_CHECK_IF(dim <= 0 || batch <= 0, OP_LOGE(context, "dim/batch must be positive"), return ge::GRAPH_FAILED);

    const DimTileChoice choice = ChooseDimTileSize(context, batch, dim, coreNum);
    OP_CHECK_IF(choice.dimTileSize <= 0 || choice.blocksPerSeq <= 0 || choice.gridSize <= 0,
                OP_LOGE(context, "invalid dim_tile_size selection"), return ge::GRAPH_FAILED);

    const uint32_t blockDim =
        (choice.gridSize < static_cast<int64_t>(coreNum)) ? static_cast<uint32_t>(choice.gridSize) : coreNum;

    OP_LOGD(context,
            "Tiling result: batch[%ld], dim[%ld], dimTileSize[%ld], blocksPerSeq[%ld], gridSize[%ld], blockDim[%u], "
            "coreNum[%u].",
            batch, dim, choice.dimTileSize, choice.blocksPerSeq, choice.gridSize, blockDim, coreNum);

    context->SetBlockDim(blockDim);
    tiling->dimTileSize = choice.dimTileSize;
    tiling->blocksPerSeq = choice.blocksPerSeq;

    const uint64_t tilingKey = GET_TPL_TILING_KEY(CAUSAL_CONV1D_TPL_SCH_MODE_DEFAULT);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForCausalConv1d(gert::TilingParseContext *context)
{
    OP_LOGD(context, "Enter TilingParseForCausalConv1d.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CausalConv1dV310)
    .Tiling(CausalConv1dTilingFunc)
    .TilingParse<CausalConv1dCompileInfo>(TilingParseForCausalConv1d);
}  // namespace optiling