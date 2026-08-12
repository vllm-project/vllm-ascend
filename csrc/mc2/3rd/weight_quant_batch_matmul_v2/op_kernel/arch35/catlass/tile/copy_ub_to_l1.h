/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ARCH35_CATLASS_TILE_COPY_UB_TO_L1_H
#define ARCH35_CATLASS_TILE_COPY_UB_TO_L1_H

#include "../utils/device_utils.h"

namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass {
using AscendC::DataCopyParams;
using AscendC::LocalTensor;

template <typename T>
DEVICE void CopyUbToL1IntervalDataCopy(const LocalTensor<T> &dst, const LocalTensor<T> &src, uint32_t blockCount,
                                       uint32_t blockLen, uint32_t dstInnerLength, uint32_t srcInnerLength)
{
    DataCopyParams params;
    params.blockLen = blockLen;
    params.blockCount = blockCount;
    params.srcStride = srcInnerLength;
    params.dstStride = dstInnerLength;
    DataCopy(dst, src, params);
}

} // namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass
#endif
