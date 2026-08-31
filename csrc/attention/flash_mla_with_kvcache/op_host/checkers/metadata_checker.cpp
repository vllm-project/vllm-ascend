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
 * \file metadata_checker.cpp
 * \brief Checker for metadata parameter (文档: INT32, shape=(max_schedule_size,))
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../flash_mla_with_kvcache_tiling_info.h"
#include "metadata_checker.h"

namespace optiling {
namespace flash_mla_with_kvcache {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35MLA;

ge::graphStatus MetadataChecker::CheckSinglePara(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    auto &metadataTensor = faInfo.opParamInfo.metadata.tensor;
    if (metadataTensor == nullptr) {
        OP_LOGE(faInfo.opName, "metadata is required but is null!");
        return ge::GRAPH_FAILED;
    }

    const gert::CompileTimeTensorDesc *metadataDesc = faInfo.opParamInfo.metadata.desc;
    OP_CHECK_IF(metadataDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(faInfo.opName, "metadata",
            "metadata desc cannot be empty"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(metadataDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(faInfo.opName, "metadata",
            DataTypeToSerialString(metadataDesc->GetDataType()).c_str(),
            "The dtype of metadata must be INT32"),
        return ge::GRAPH_FAILED);

    if (ge::GRAPH_SUCCESS != CheckFormatSupport(metadataDesc, METADATA_NAME)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t dimNum = metadataTensor->GetStorageShape().GetDimNum();
    OP_CHECK_IF(dimNum != 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(faInfo.opName, "metadata",
            std::to_string(dimNum).c_str(), "The shape dim of metadata must be 1"),
        return ge::GRAPH_FAILED);

    int64_t dim0 = metadataTensor->GetStorageShape().GetDim(0);
    OP_CHECK_IF(dim0 <= 0,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(faInfo.opName, "metadata",
            GetShapeStr(metadataTensor->GetStorageShape()).c_str(),
            "The 1st axis of metadata must be greater than 0"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

} // namespace flash_mla_with_kvcache
} // namespace optiling