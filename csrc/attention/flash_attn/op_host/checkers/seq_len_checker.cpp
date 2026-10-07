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
 * \file actual_seq_len_checker.cpp
 * \brief Checker for cu_seqlens_q/kv (B+1) and seqused_q/kv (B) parameters
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../fa_tiling_info.h"
#include "seq_len_checker_flash_attn.h"

namespace optiling {
namespace flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

ge::graphStatus ActualSeqLenChecker::CheckSingleParaSequsedQ(const FaTilingInfo &faInfo)
{
    auto &sequsedQTensor = faInfo.opParamInfo.sequsedQ.tensor;
    if (sequsedQTensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::CompileTimeTensorDesc *sequsedQDesc = faInfo.opParamInfo.sequsedQ.desc;
    OP_CHECK_IF(sequsedQDesc != nullptr && sequsedQDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(faInfo.opName, "seqused_q",
                                                      DataTypeToSerialString(sequsedQDesc->GetDataType()).c_str(),
                                                      "The dtype of seqused_q must be INT32"),
                return ge::GRAPH_FAILED);

    if (ge::GRAPH_SUCCESS != CheckFormatSupport(sequsedQDesc, SEQUSED_Q_NAME)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t sequsedQDimNum = sequsedQTensor->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        sequsedQDimNum != 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(faInfo.opName, "seqused_q", std::to_string(sequsedQDimNum).c_str(),
                                                 "The shape dim of seqused_q must be 1"),
        return ge::GRAPH_FAILED);

    uint32_t sequsedQShapeSize = sequsedQTensor->GetShapeSize();
    if (sequsedQShapeSize != faInfo.bSize) {
        std::string shapeMsg = std::to_string(sequsedQShapeSize) + ", " + std::to_string(faInfo.bSize);
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(faInfo.opName, "seqused_q and batch", shapeMsg.c_str(),
                                                   "The shape sizes of seqused_q and batch must be the same");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckSingleParaSequsedKv(const FaTilingInfo &faInfo)
{
    auto &sequsedKvTensor = faInfo.opParamInfo.sequsedKv.tensor;
    if (sequsedKvTensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::CompileTimeTensorDesc *sequsedKvDesc = faInfo.opParamInfo.sequsedKv.desc;
    OP_CHECK_IF(sequsedKvDesc != nullptr && sequsedKvDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(faInfo.opName, "seqused_kv",
                                                      DataTypeToSerialString(sequsedKvDesc->GetDataType()).c_str(),
                                                      "The dtype of seqused_kv must be INT32"),
                return ge::GRAPH_FAILED);

    if (ge::GRAPH_SUCCESS != CheckFormatSupport(sequsedKvDesc, SEQUSED_KV_NAME)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t sequsedKvDimNum = sequsedKvTensor->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        sequsedKvDimNum != 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(faInfo.opName, "seqused_kv", std::to_string(sequsedKvDimNum).c_str(),
                                                 "The shape dim of seqused_kv must be 1"),
        return ge::GRAPH_FAILED);

    uint32_t sequsedKvShapeSize = sequsedKvTensor->GetShapeSize();
    if (sequsedKvShapeSize != faInfo.bSize) {
        std::string shapeMsg = std::to_string(sequsedKvShapeSize) + ", " + std::to_string(faInfo.bSize);
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(faInfo.opName, "seqused_kv and batch", shapeMsg.c_str(),
                                                   "The shape sizes of seqused_kv and batch must be the same");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckSingleParaCuSeqlensQ(const FaTilingInfo &faInfo)
{
    auto &cuSeqlensQTensor = faInfo.opParamInfo.cuSeqlensQ.tensor;
    if (cuSeqlensQTensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::CompileTimeTensorDesc *cuSeqlensQDesc = faInfo.opParamInfo.cuSeqlensQ.desc;
    OP_CHECK_IF(cuSeqlensQDesc != nullptr && cuSeqlensQDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(faInfo.opName, "cu_seqlens_q",
                                                      DataTypeToSerialString(cuSeqlensQDesc->GetDataType()).c_str(),
                                                      "The dtype of cu_seqlens_q must be INT32"),
                return ge::GRAPH_FAILED);

    if (ge::GRAPH_SUCCESS != CheckFormatSupport(cuSeqlensQDesc, CU_SEQLENS_Q_NAME)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t cuSeqlensQDimNum = cuSeqlensQTensor->GetStorageShape().GetDimNum();
    OP_CHECK_IF(cuSeqlensQDimNum != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(faInfo.opName, "cu_seqlens_q",
                                                         std::to_string(cuSeqlensQDimNum).c_str(),
                                                         "The shape dim of cu_seqlens_q must be 1"),
                return ge::GRAPH_FAILED);

    uint32_t cuSeqlensQShapeSize = cuSeqlensQTensor->GetShapeSize();
    if (cuSeqlensQShapeSize != faInfo.bSize + 1) {
        std::string shapeMsg = std::to_string(cuSeqlensQShapeSize) + ", " + std::to_string(faInfo.bSize);
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(faInfo.opName, "cu_seqlens_q and batch", shapeMsg.c_str(),
                                                   "The shape size of cu_seqlens_q must be equal to batch + 1");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckSingleParaCuSeqlensKv(const FaTilingInfo &faInfo)
{
    auto &cuSeqlensKvTensor = faInfo.opParamInfo.cuSeqlensKv.tensor;
    if (cuSeqlensKvTensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::CompileTimeTensorDesc *cuSeqlensKvDesc = faInfo.opParamInfo.cuSeqlensKv.desc;
    OP_CHECK_IF(cuSeqlensKvDesc != nullptr && cuSeqlensKvDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(faInfo.opName, "cu_seqlens_kv",
                                                      DataTypeToSerialString(cuSeqlensKvDesc->GetDataType()).c_str(),
                                                      "The dtype of cu_seqlens_kv must be INT32"),
                return ge::GRAPH_FAILED);

    if (ge::GRAPH_SUCCESS != CheckFormatSupport(cuSeqlensKvDesc, CU_SEQLENS_KV_NAME)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t cuSeqlensKvDimNum = cuSeqlensKvTensor->GetStorageShape().GetDimNum();
    OP_CHECK_IF(cuSeqlensKvDimNum != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(faInfo.opName, "cu_seqlens_kv",
                                                         std::to_string(cuSeqlensKvDimNum).c_str(),
                                                         "The shape dim of cu_seqlens_kv must be 1"),
                return ge::GRAPH_FAILED);

    uint32_t cuSeqlensKvShapeSize = cuSeqlensKvTensor->GetShapeSize();
    if (cuSeqlensKvShapeSize != faInfo.bSize + 1) {
        std::string shapeMsg = std::to_string(cuSeqlensKvShapeSize) + ", " + std::to_string(faInfo.bSize);
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(faInfo.opName, "cu_seqlens_kv and batch", shapeMsg.c_str(),
                                                   "The shape size of cu_seqlens_kv must be equal to batch + 1");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckSingleParaMaxSeqlenQ(const FaTilingInfo &faInfo)
{
    OP_CHECK_IF(faInfo.maxSeqQ < -1,
                OP_LOGE_FOR_INVALID_VALUE(faInfo.opName, "max_seqlen_q", std::to_string(faInfo.maxSeqQ).c_str(),
                                          "= -1 or >= 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckSingleParaMaxSeqlenKv(const FaTilingInfo &faInfo)
{
    OP_CHECK_IF(faInfo.maxSeqKv < -1,
                OP_LOGE_FOR_INVALID_VALUE(faInfo.opName, "max_seqlen_kv", std::to_string(faInfo.maxSeqKv).c_str(),
                                          "= -1 or >= 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckSinglePara(const FaTilingInfo &faInfo)
{
    if (CheckSingleParaSequsedQ(faInfo) != ge::GRAPH_SUCCESS || CheckSingleParaSequsedKv(faInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaCuSeqlensQ(faInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaCuSeqlensKv(faInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaMaxSeqlenQ(faInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaMaxSeqlenKv(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckParaExistence(const FaTilingInfo &faInfo)
{
    if (faInfo.pageAttentionFlag) {
        auto &sequsedKvTensor = faInfo.opParamInfo.sequsedKv.tensor;
        OP_CHECK_IF(sequsedKvTensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        faInfo.opName, "seqused_kv", "seqused_kv cannot be empty when paged attention is enabled"),
                    return ge::GRAPH_FAILED);
    }

    auto &cuSeqlensQTensor = faInfo.opParamInfo.cuSeqlensQ.tensor;
    if (faInfo.qLayout == FaLayout::TND) {
        OP_CHECK_IF(cuSeqlensQTensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(faInfo.opName, "cu_seqlens_q",
                                                             "cu_seqlens_q cannot be empty when layout_q is TND"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(cuSeqlensQTensor != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        faInfo.opName, "cu_seqlens_q",
                        "cu_seqlens_q must be empty when layout_q is not TND, only supported in TND layout"),
                    return ge::GRAPH_FAILED);
    }

    auto &cuSeqlensKvTensor = faInfo.opParamInfo.cuSeqlensKv.tensor;
    if (faInfo.kvLayout == FaLayout::TND) {
        OP_CHECK_IF(cuSeqlensKvTensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(faInfo.opName, "cu_seqlens_kv",
                                                             "cu_seqlens_kv cannot be empty when layout_kv is TND"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(cuSeqlensKvTensor != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        faInfo.opName, "cu_seqlens_kv",
                        "cu_seqlens_kv must be empty when layout_kv is not TND, only supported in TND layout"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckFeature(const FaTilingInfo &faInfo)
{
    (void)faInfo;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckMultiPara(const FaTilingInfo &faInfo)
{
    auto &cuSeqlensQTensor = faInfo.opParamInfo.cuSeqlensQ.tensor;
    if (cuSeqlensQTensor != nullptr && faInfo.qLayout == FaLayout::TND) {
    }

    auto &cuSeqlensKvTensor = faInfo.opParamInfo.cuSeqlensKv.tensor;
    if (cuSeqlensKvTensor != nullptr && faInfo.kvLayout == FaLayout::TND) {
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace flash_attn
} // namespace optiling
