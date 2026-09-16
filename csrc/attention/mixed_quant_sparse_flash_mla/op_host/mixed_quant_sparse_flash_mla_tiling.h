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
 * \file mixed_quant_sparse_flash_mla_tiling.h
 * \brief
 */
#ifndef MIXED_QUANT_SPARSE_FLASH_MLA_TILING_H
#define MIXED_QUANT_SPARSE_FLASH_MLA_TILING_H

#include <graph/utils/type_utils.h>
#include <exe_graph/runtime/tiling_context.h>
#include <tiling/platform/platform_ascendc.h>
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "log/log.h"
#include "log/error_code.h"
#include "vendor/c6240b268/include/err/ops_err.h"
#include "platform/platform_info.h"
#include "mixed_quant_sparse_flash_mla_check.h"

namespace optiling {

std::vector<int64_t> MQSMLAC624ToVector(const gert::Shape &shape);
std::string MQSMLAC624ToStringRaw(const gert::Shape &shape);
std::string MQSMLALayoutToSerialString(MQSMLALayout layout);

// -----------算子TilingData定义---------------
BEGIN_TILING_DATA_DEF(MixedQuantSparseFlashMlaBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, batchSize)
TILING_DATA_FIELD_DEF(uint32_t, qSeqSize)
TILING_DATA_FIELD_DEF(uint32_t, kvSeqSize)
TILING_DATA_FIELD_DEF(uint32_t, cmpKvSeqSize)
TILING_DATA_FIELD_DEF(uint32_t, paOriBlockSize)
TILING_DATA_FIELD_DEF(uint32_t, paCmpBlockSize)
TILING_DATA_FIELD_DEF(uint32_t, oriMaxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, cmpMaxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, nNumOfQInOneGroup)
TILING_DATA_FIELD_DEF(uint32_t, oriSparseBlockCount)
TILING_DATA_FIELD_DEF(uint32_t, cmpSparseBlockCount)
TILING_DATA_FIELD_DEF(float, softmaxScale) // 即 scaleValue
TILING_DATA_FIELD_DEF(int32_t, oriKvStride)
TILING_DATA_FIELD_DEF(int32_t, cmpKvStride)
TILING_DATA_FIELD_DEF(uint32_t, tileSize)
TILING_DATA_FIELD_DEF(uint32_t, ropeHeadDim)
TILING_DATA_FIELD_DEF(uint32_t, cmpRatio)
TILING_DATA_FIELD_DEF(uint32_t, oriMaskMode)
TILING_DATA_FIELD_DEF(uint32_t, cmpMaskMode)
TILING_DATA_FIELD_DEF(int32_t, oriWinLeft)
TILING_DATA_FIELD_DEF(int32_t, oriWinRight)
TILING_DATA_FIELD_DEF(uint32_t, sparseBlockSize)
TILING_DATA_FIELD_DEF(uint32_t, dSize)
TILING_DATA_FIELD_DEF(uint32_t, dSizeVInput)
TILING_DATA_FIELD_DEF(uint32_t, returnSoftmaxLse)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(MixedQuantSparseFlashMlaBaseParamsOp, MixedQuantSparseFlashMlaBaseParams)

BEGIN_TILING_DATA_DEF(MixedQuantSparseFlashMlaTilingData)
TILING_DATA_FIELD_DEF_STRUCT(MixedQuantSparseFlashMlaBaseParams, baseParams);
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(MixedQuantSparseFlashMla, MixedQuantSparseFlashMlaTilingData)

// ---------------算子Tiling类---------------
class MixedQuantSparseFlashMlaTiling {
public:
    explicit MixedQuantSparseFlashMlaTiling(gert::TilingContext *context)
        : context_(context) {};
    ge::graphStatus DoOpTiling(MQSMLATilingInfo *tilingInfo);

private:
    gert::TilingContext *context_ = nullptr;
    QSMLATemplateMode perfMode_ = QSMLATemplateMode::SWA_TEMPLATE_MODE;
    MixedQuantSparseFlashMlaTilingData tilingData_;
    uint32_t blockDim_{0};
    uint64_t workspaceSize_{0};
    uint64_t tilingKey_{0};

    MQSMLATilingInfo *qsmlaInfo_ = nullptr;
};

} // namespace optiling
#endif
