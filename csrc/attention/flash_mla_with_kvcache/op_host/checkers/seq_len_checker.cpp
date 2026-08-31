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
 * \brief Checker for cache_seqlens (B), cu_seqlens_q (B+1) and seqused_q (B) parameters
 *
 * Value-level checks (cache_seqlens > 0 / <= max_seqlen_kv, cu_seqlens_q
 * non-decreasing / last == total_q) only run when the host tensor data is
 * actually present (`GetData<T>() != nullptr`, same guard as FIA tiling
 * InitImplParam); during graph compile the data may be absent, in which case
 * the shape/dtype rules still hold and the kernel enforces the values at
 * runtime (ACTLEN_T=uint32_t parser).
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../flash_mla_with_kvcache_tiling_info.h"
#include "seq_len_checker.h"

namespace optiling {
namespace flash_mla_with_kvcache {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35MLA;

ge::graphStatus ActualSeqLenChecker::CheckSingleParaSequsedQ(const FlashMlaWithKvcacheTilingInfo &faInfo)
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
    OP_CHECK_IF(sequsedQDimNum != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(faInfo.opName, "seqused_q",
                                                         std::to_string(sequsedQDimNum).c_str(), "The shape dim of seqused_q must be 1"),
                return ge::GRAPH_FAILED);

    uint32_t sequsedQShapeSize = sequsedQTensor->GetShapeSize();
    if (sequsedQShapeSize != faInfo.bSize) {
        std::string shapeMsg = std::to_string(sequsedQShapeSize) + ", " + std::to_string(faInfo.bSize);
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            faInfo.opName, "seqused_q and batch", shapeMsg.c_str(),
            "The shape sizes of seqused_q and batch must be the same");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckSingleParaCacheSeqlens(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    auto &cacheSeqlensTensor = faInfo.opParamInfo.cacheSeqlens.tensor;
    if (cacheSeqlensTensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::CompileTimeTensorDesc *cacheSeqlensDesc = faInfo.opParamInfo.cacheSeqlens.desc;
    OP_CHECK_IF(cacheSeqlensDesc != nullptr && cacheSeqlensDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(faInfo.opName, "cache_seqlens",
                                                      DataTypeToSerialString(cacheSeqlensDesc->GetDataType()).c_str(),
                                                      "The dtype of cache_seqlens must be INT32"),
                return ge::GRAPH_FAILED);

    if (ge::GRAPH_SUCCESS != CheckFormatSupport(cacheSeqlensDesc, CACHE_SEQLENS_NAME)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t cacheSeqlensDimNum = cacheSeqlensTensor->GetStorageShape().GetDimNum();
    OP_CHECK_IF(cacheSeqlensDimNum != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(faInfo.opName, "cache_seqlens",
                                                         std::to_string(cacheSeqlensDimNum).c_str(), "The shape dim of cache_seqlens must be 1"),
                return ge::GRAPH_FAILED);

    uint32_t cacheSeqlensShapeSize = cacheSeqlensTensor->GetShapeSize();
    if (cacheSeqlensShapeSize != faInfo.bSize) {
        std::string shapeMsg = std::to_string(cacheSeqlensShapeSize) + ", " + std::to_string(faInfo.bSize);
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            faInfo.opName, "cache_seqlens and batch", shapeMsg.c_str(),
            "The shape sizes of cache_seqlens and batch must be the same");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckSingleParaCuSeqlensQ(const FlashMlaWithKvcacheTilingInfo &faInfo)
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
                                                         std::to_string(cuSeqlensQDimNum).c_str(), "The shape dim of cu_seqlens_q must be 1"),
                return ge::GRAPH_FAILED);

    uint32_t cuSeqlensQShapeSize = cuSeqlensQTensor->GetShapeSize();
    if (cuSeqlensQShapeSize != faInfo.bSize + 1) {
        std::string shapeMsg = std::to_string(cuSeqlensQShapeSize) + ", " + std::to_string(faInfo.bSize);
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            faInfo.opName, "cu_seqlens_q and batch", shapeMsg.c_str(),
            "The shape size of cu_seqlens_q must be equal to batch + 1");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckSingleParaMaxSeqlenQ(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    OP_CHECK_IF(faInfo.maxSeqQ < -1,
                OP_LOGE_FOR_INVALID_VALUE(faInfo.opName, "max_seqlen_q",
                                          std::to_string(faInfo.maxSeqQ).c_str(), "= -1 or >= 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckSingleParaMaxSeqlenKv(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    OP_CHECK_IF(faInfo.maxSeqKv < -1,
                OP_LOGE_FOR_INVALID_VALUE(faInfo.opName, "max_seqlen_kv",
                                          std::to_string(faInfo.maxSeqKv).c_str(), "= -1 or >= 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckSinglePara(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    if (CheckSingleParaSequsedQ(faInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaCacheSeqlens(faInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaCuSeqlensQ(faInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaMaxSeqlenQ(faInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaMaxSeqlenKv(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckParaExistence(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    // KV is paged-only -> cache_seqlens always required
    if (faInfo.pageAttentionFlag) {
        auto &cacheSeqlensTensor = faInfo.opParamInfo.cacheSeqlens.tensor;
        OP_CHECK_IF(cacheSeqlensTensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(faInfo.opName, "cache_seqlens",
                                                             "cache_seqlens cannot be empty when paged attention is enabled"),
                    return ge::GRAPH_FAILED);
    }

    // cu_seqlens_q is only meaningful for the TND query layout: required when TND,
    // forbidden otherwise (BNSD/BSND query layouts carry no cumulative seqlens)
    auto &cuSeqlensQTensor = faInfo.opParamInfo.cuSeqlensQ.tensor;
    if (faInfo.qLayout == FlashMlaWithKvcacheLayout::TND) {
        OP_CHECK_IF(cuSeqlensQTensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(faInfo.opName, "cu_seqlens_q",
                                                             "cu_seqlens_q cannot be empty when layout_q is TND"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(cuSeqlensQTensor != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(faInfo.opName, "cu_seqlens_q",
                                                             "cu_seqlens_q must be empty when layout_q is not TND, only supported in TND layout"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckFeature(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    (void)faInfo;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckMultiPara(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    // ACLNN 直调路径下 GetData<int32_t>() 返回设备指针，host 侧值校验会段错误
    // （与 flash_attn 同策略：只做 shape/dtype 校验；值约束由 kernel 的 ACTLEN_T=uint32_t 解析器兜底）。
    return ge::GRAPH_SUCCESS;
}

} // namespace flash_mla_with_kvcache
} // namespace optiling
