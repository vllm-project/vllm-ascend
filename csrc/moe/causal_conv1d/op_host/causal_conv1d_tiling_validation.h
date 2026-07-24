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

 #ifndef CAUSAL_CONV1D_TILING_VALIDATION_H
 #define CAUSAL_CONV1D_TILING_VALIDATION_H

 #include "tiling_base/tiling_util.h"
 #include "causal_conv1d_tiling_utils.h"
 #include "../op_kernel/causal_conv1d_tiling_data.h"

 namespace optiling::causal_conv1d_host {

 using namespace Ops::Transformer::OpTiling;

 inline ge::graphStatus GetPlatformInfo(gert::TilingContext *context, uint64_t &ubSize, uint32_t &coreNum)
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

 inline ge::graphStatus SetWorkspaceSize(gert::TilingContext *context, size_t workspaceSize)
 {
     size_t *currentWorkspace = context->GetWorkspaceSizes(1);
     OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
     currentWorkspace[0] = workspaceSize;
     return ge::GRAPH_SUCCESS;
 }

 inline ge::graphStatus GetAttrsInfo(gert::TilingContext *context, CausalConv1dAttrInfo &attrInfo)
 {
     auto attrs = context->GetAttrs();
     OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

     const int64_t *activationModePtr = attrs->GetAttrPointer<int64_t>(ATTR_ACTIVATION_MODE_INDEX);
     OP_CHECK_NULL_WITH_CONTEXT(context, activationModePtr);
     attrInfo.activationMode = *activationModePtr;
     OP_CHECK_IF(attrInfo.activationMode != 0 && attrInfo.activationMode != 1,
                 OP_LOGE(context, "activationMode only supports 0/1"),
                 return ge::GRAPH_FAILED);

     const int64_t *padSlotIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_PAD_SLOT_ID_INDEX);
     OP_CHECK_NULL_WITH_CONTEXT(context, padSlotIdPtr);
     attrInfo.padSlotId = *padSlotIdPtr;

     const int64_t *runModePtr = attrs->GetAttrPointer<int64_t>(ATTR_RUN_MODE_INDEX);
     attrInfo.runMode = (runModePtr == nullptr) ? 0 : *runModePtr;
     OP_CHECK_IF(attrInfo.runMode != 0 && attrInfo.runMode != 1, OP_LOGE(context, "runMode only supports 0/1"),
                 return ge::GRAPH_FAILED);

     const int64_t *headNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_HEAD_NUM_INDEX);
     attrInfo.headNum = (headNumPtr == nullptr) ? 0 : *headNumPtr;
     OP_CHECK_IF(attrInfo.headNum < 0, OP_LOGE(context, "headNum only supports larger than 0"),
         return ge::GRAPH_FAILED);
     return ge::GRAPH_SUCCESS;
 }

 inline ge::graphStatus ValidateAlignedDim(gert::TilingContext *context, int64_t dim)
 {
     OP_CHECK_IF(dim % DIM_ALIGN_ELEMS != 0,
                 OP_LOGE(context,
                         "dim must satisfy dim %% %ld == 0 for causal_conv1d; "
                         "x/weight/convStates last dimension and bias length must all use the same aligned dim, "
                         "got dim=%ld.",
                         DIM_ALIGN_ELEMS, dim),
                 return ge::GRAPH_FAILED);
     return ge::GRAPH_SUCCESS;
 }

 inline ge::graphStatus GetShapeDtypeInfo(gert::TilingContext *context, const CausalConv1dAttrInfo &attrInfo,
                                          CausalConv1dTilingData &tiling, bool &hasBias)
 {
     const bool isDecodeMode = (attrInfo.runMode == 1);
     tiling.activationMode = attrInfo.activationMode;
     tiling.padSlotId = attrInfo.padSlotId;
     tiling.headNum = attrInfo.headNum;
     tiling.isOutReshape = (tiling.headNum > 0);

     auto xShapePtr = context->GetInputShape(X_INDEX);
     OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
     auto xShape = EnsureNotScalar(xShapePtr->GetStorageShape());

     int64_t dim = 0;
     int64_t cuSeqlen = 0;
     int64_t seqLen = 0;
     int64_t batch = 0;
     int64_t inputMode = 0;
     int64_t headNum = tiling.headNum;

     if (xShape.GetDimNum() == 2) {
         if (isDecodeMode) {
             inputMode = 2;
             batch = xShape.GetDim(0);
             dim = xShape.GetDim(1);
             seqLen = 1;
             cuSeqlen = batch;
             OP_CHECK_IF(batch <= 0 || dim <= 0, OP_LOGE(context, "invalid x shape for 2D decode mode"),
                         return ge::GRAPH_FAILED);
             OP_CHECK_IF(headNum < 0 || headNum > dim || (headNum > 0 && dim % headNum != 0),
                 OP_LOGE(context, "invalid headNum: headNum should be in [0, dim] and dim mod headNum = 0, actually headNum=%ld, dim=%ld.", headNum, dim),
                 return ge::GRAPH_FAILED);
             OP_CHECK_IF(headNum > 0 && (dim / headNum) % DIM_ALIGN_ELEMS != 0,
                 OP_LOGE(context, "the headDim (= dim / headNum) should be multiple of 16, but actually headNum=%ld, dim=%ld.", headNum, dim),
                 return ge::GRAPH_FAILED);
         } else {
             inputMode = 0;
             cuSeqlen = xShape.GetDim(0);
             dim = xShape.GetDim(1);
             seqLen = 0;
             OP_CHECK_IF(dim <= 0 || cuSeqlen < 0, OP_LOGE(context, "invalid x shape for 2D varlen mode"),
                         return ge::GRAPH_FAILED);
             OP_CHECK_IF(headNum < 0 || headNum > dim || (headNum > 0 && dim % headNum != 0),
                 OP_LOGE(context, "invalid headNum: headNum should be in [0, dim] and dim mod headNum = 0, actually headNum=%ld, dim=%ld.", headNum, dim),
                 return ge::GRAPH_FAILED);
             OP_CHECK_IF(headNum > 0 && (dim / headNum) % DIM_ALIGN_ELEMS != 0,
                 OP_LOGE(context, "the headDim (= dim / headNum) should be multiple of 16, but actually headNum=%ld, dim=%ld.", headNum, dim),
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
         if (isDecodeMode) {
             OP_CHECK_IF(headNum < 0 || headNum > dim || (headNum > 0 && dim % headNum != 0),
                 OP_LOGE(context, "invalid headNum: headNum should be in [0, dim] and dim mod headNum = 0, actually headNum=%ld, dim=%ld.", headNum, dim),
                 return ge::GRAPH_FAILED);
             OP_CHECK_IF(headNum > 0 && (dim / headNum) % DIM_ALIGN_ELEMS != 0,
                 OP_LOGE(context, "the headDim (= dim / headNum) should be multiple of 16, but actually headNum=%ld, dim=%ld.", headNum, dim),
                 return ge::GRAPH_FAILED);
         } else {
             OP_CHECK_IF(headNum < 0 || headNum > dim || (headNum > 0 && dim % headNum != 0),
                 OP_LOGE(context, "invalid headNum: headNum should be in [0, dim] and dim mod headNum = 0, actually headNum=%ld, dim=%ld.", headNum, dim),
                 return ge::GRAPH_FAILED);
             OP_CHECK_IF(headNum > 0 && (dim / headNum) % DIM_ALIGN_ELEMS != 0,
                 OP_LOGE(context, "the headDim (= dim / headNum) should be multiple of 16, but actually headNum=%ld, dim=%ld.", headNum, dim),
                 return ge::GRAPH_FAILED);
         }
     } else {
         OP_LOGE(context, "x must be 2D (cu_seqlen, dim) or 3D (batch, seqlen, dim)");
         return ge::GRAPH_FAILED;
     }
     OP_CHECK_IF(ValidateAlignedDim(context, dim) != ge::GRAPH_SUCCESS,
                 OP_LOGE(context, "dim alignment validation failed"),
                 return ge::GRAPH_FAILED);

     auto wShapePtr = context->GetInputShape(WEIGHT_INDEX);
     OP_CHECK_NULL_WITH_CONTEXT(context, wShapePtr);
     auto wShape = EnsureNotScalar(wShapePtr->GetStorageShape());
     OP_CHECK_IF(wShape.GetDimNum() != 2, OP_LOGE(context, "weight must be 2D: (width, dim)"), return ge::GRAPH_FAILED);
     const int64_t width = wShape.GetDim(0);
     const int64_t wDim = wShape.GetDim(1);
     OP_CHECK_IF(wDim != dim, OP_LOGE(context, "weight.shape[1] must equal dim"), return ge::GRAPH_FAILED);
     OP_CHECK_IF(width < 2 || width > 4, OP_LOGE(context, "Only support width in [2,4] now, actually is %ld.", width),
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
             const ge::DataType qslDtype = qslDesc->GetDataType();
             OP_CHECK_IF(qslDtype != ge::DT_INT32 && qslDtype != ge::DT_INT64,
                         OP_LOGE(context, "queryStartLoc dtype must be int32 or int64"), return ge::GRAPH_FAILED);
         }
     }

     if (qslAbsent) {
         OP_CHECK_IF(inputMode == 0, OP_LOGE(context, "queryStartLoc is required in 2D varlen mode (inputMode=0)"),
                     return ge::GRAPH_FAILED);
         qslSize = batch + 1;
     }
     tiling.hasQueryStartLoc = qslAbsent ? 0 : 1;

     OP_CHECK_IF(cuSeqlen > static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
                 OP_LOGE(context, "cuSeqlen is too large for int32 indexing, got %ld", cuSeqlen),
                 return ge::GRAPH_FAILED);

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
     tiling.cacheIndicesStride = 1;
     bool ciAbsent = true;
     auto ciShapePtr = context->GetOptionalInputShape(CACHE_INDICES_INDEX);
     if (ciShapePtr != nullptr) {
         const auto ciStorageShape = ciShapePtr->GetStorageShape();
         const int64_t ciDimNum = ciStorageShape.GetDimNum();
         ciAbsent = (ciDimNum == 0) || (ciDimNum == 1 && ciStorageShape.GetDim(0) <= 0);
         if (!ciAbsent) {
             auto ciShape = EnsureNotScalar(ciStorageShape);
             // Spec decode passes cache indices as [batch, num_spec + 1];
             // kernels read the first column with cacheIndicesStride.
             OP_CHECK_IF(ciShape.GetDimNum() != 1 && ciShape.GetDimNum() != 2,
                         OP_LOGE(context, "cacheIndices must be 1D or 2D"), return ge::GRAPH_FAILED);
             OP_CHECK_IF(ciShape.GetDim(0) != batch, OP_LOGE(context, "cacheIndices first dim must equal batch"),
                         return ge::GRAPH_FAILED);
             if (ciShape.GetDimNum() == 2) {
                 OP_CHECK_IF(ciShape.GetDim(1) <= 0, OP_LOGE(context, "cacheIndices second dim must be positive"),
                             return ge::GRAPH_FAILED);
                 tiling.cacheIndicesStride = ciShape.GetDim(1);
             }
             tiling.hasCacheIndices = 1;
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
             OP_CHECK_IF(isDecodeMode,
                         OP_LOGE(context, "initialStateMode is only supported in runMode=0 (fn/prefill)"),
                         return ge::GRAPH_FAILED);
             auto ismShape = EnsureNotScalar(ismStorageShape);
             OP_CHECK_IF(ismShape.GetDimNum() != 1, OP_LOGE(context, "initialStateMode must be 1D"),
                         return ge::GRAPH_FAILED);
             OP_CHECK_IF(ismShape.GetDim(0) != batch, OP_LOGE(context, "initialStateMode.size must equal batch"),
                         return ge::GRAPH_FAILED);

             tiling.hasInitialStateMode = 1;
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

             tiling.hasNumAcceptedTokens = 1;
         }
     }

     tiling.hasBias = 0;
     hasBias = false;
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
             hasBias = true;
         }
     }

     auto xDesc = context->GetInputDesc(X_INDEX);
     OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
     const ge::DataType xDtype = xDesc->GetDataType();
     OP_CHECK_IF(xDtype != ge::DT_BF16 && xDtype != ge::DT_FLOAT16,
                 OP_LOGE(context, "x dtype only supports bf16/fp16"),
                 return ge::GRAPH_FAILED);

     auto wDesc = context->GetInputDesc(WEIGHT_INDEX);
     OP_CHECK_NULL_WITH_CONTEXT(context, wDesc);
     OP_CHECK_IF(wDesc->GetDataType() != xDtype, OP_LOGE(context, "weight dtype must equal x dtype"),
                 return ge::GRAPH_FAILED);

     if (hasBias) {
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
         const ge::DataType qslDtype = qslDesc2->GetDataType();
         OP_CHECK_IF(qslDtype != ge::DT_INT32 && qslDtype != ge::DT_INT64,
                     OP_LOGE(context, "queryStartLoc dtype must be int32 or int64"), return ge::GRAPH_FAILED);
         tiling.queryStartLocUseInt64 = (qslDtype == ge::DT_INT64) ? 1 : 0;
     }
     tiling.cacheIndicesUseInt64 = 0;
     if (tiling.hasCacheIndices == 1) {
         auto ciDesc = context->GetOptionalInputDesc(CACHE_INDICES_INDEX);
         OP_CHECK_NULL_WITH_CONTEXT(context, ciDesc);
         const ge::DataType ciDtype = ciDesc->GetDataType();
         OP_CHECK_IF(ciDtype != ge::DT_INT32 && ciDtype != ge::DT_INT64,
                     OP_LOGE(context, "cacheIndices dtype must be int32 or int64"), return ge::GRAPH_FAILED);
         tiling.cacheIndicesUseInt64 = (ciDtype == ge::DT_INT64) ? 1 : 0;
     }
     if (tiling.hasInitialStateMode == 1) {
         auto ismDesc = context->GetOptionalInputDesc(INITIAL_STATE_MODE_INDEX);
         OP_CHECK_NULL_WITH_CONTEXT(context, ismDesc);
         const ge::DataType ismDtype = ismDesc->GetDataType();
         OP_CHECK_IF(ismDtype != ge::DT_BOOL && ismDtype != ge::DT_INT32 && ismDtype != ge::DT_INT64,
                     OP_LOGE(context, "initialStateMode dtype must be bool, int32 or int64"), return ge::GRAPH_FAILED);
         tiling.initialStateModeDtype =
             (ismDtype == ge::DT_INT64) ? 2 : ((ismDtype == ge::DT_INT32) ? 1 : 0);
     }
     tiling.numAcceptedTokensUseInt64 = 0;
     if (tiling.hasNumAcceptedTokens == 1) {
         OP_CHECK_IF(width != 4, OP_LOGE(context, "numAcceptedTokens is only supported for width=4 currently"),
                     return ge::GRAPH_FAILED);
         auto natDesc = context->GetOptionalInputDesc(NUM_ACCEPTED_TOKENS_INDEX);
         OP_CHECK_NULL_WITH_CONTEXT(context, natDesc);
         const ge::DataType natDtype = natDesc->GetDataType();
         OP_CHECK_IF(natDtype != ge::DT_INT32 && natDtype != ge::DT_INT64,
                     OP_LOGE(context, "numAcceptedTokens dtype must be int32 or int64"), return ge::GRAPH_FAILED);
         tiling.numAcceptedTokensUseInt64 = (natDtype == ge::DT_INT64) ? 1 : 0;
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

 }

 #endif
