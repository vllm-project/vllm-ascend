/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ARCH35_CATLASS_TILE_COPY_GM_TO_L1_H
#define ARCH35_CATLASS_TILE_COPY_GM_TO_L1_H

#include "../utils/device_utils.h"

namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass {
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::Nd2NzParams;

template <typename T>
DEVICE void CopyGmToL1IntervalDataCopy(const LocalTensor<T> &dst, const GlobalTensor<T> &src, uint32_t nValue,
                                       uint32_t dValue, uint32_t dstInnerLength, uint32_t srcInnerLength)
{
    // ND2NZ
    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = nValue;
    nd2nzParams.dValue = dValue;
    nd2nzParams.srcDValue = srcInnerLength;
    // PS tiling保证是16对齐的
    nd2nzParams.dstNzC0Stride = dstInnerLength;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;
    DataCopy(dst, src, nd2nzParams);
}

template <typename T>
DEVICE void CopyGmToL1(const LocalTensor<T> &dst, const GlobalTensor<T> &src, uint32_t size)
{
    DataCopy(dst, src, size);
}

} // namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass
#endif
