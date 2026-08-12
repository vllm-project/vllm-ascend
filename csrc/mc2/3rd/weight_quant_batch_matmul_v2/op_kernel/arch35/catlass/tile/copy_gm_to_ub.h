/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ARCH35_CATLASS_TILE_COPY_GM_TO_UB_H
#define ARCH35_CATLASS_TILE_COPY_GM_TO_UB_H

#include "../utils/device_utils.h"

namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass {
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPadExtParams;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;

template <typename T>
DEVICE void CopyGmToUbIntervalDataCopy(const LocalTensor<T> &dst, const GlobalTensor<T> &src, uint32_t blockCount,
                                       uint32_t blockLen, uint32_t dstInnerLength, uint32_t srcInnerLength)
{
#if defined(__CCE_KT_TEST__)
    ASCENDC_ASSERT(dstInnerLength >= blockLen,
                   { X_LOG("dstInnerLength[%d] should be larger than blockLen[%d].", dstInnerLength, blockLen); });
#endif
    DataCopyExtParams params;
    params.blockCount = blockCount;
    params.blockLen = blockLen * sizeof(T);
    params.srcStride = (srcInnerLength - blockLen) * sizeof(T);
    params.dstStride = (dstInnerLength - blockLen) * sizeof(T) / ONE_BLK_SIZE;
    DataCopyPadExtParams<T> padParams;
    if (blockLen % (32 / sizeof(T)) != 0) {
        padParams.isPad = true;
        padParams.rightPadding = CeilAlign(blockLen, static_cast<uint32_t>(32 / sizeof(T))) - blockLen;
        padParams.paddingValue = 0;
    }
    if constexpr (IsSameType<T, int4b_t>::value) {
        // int4场景下， 跳转的步长、数据长度等需要除2
        params.blockLen = params.blockLen >> 1;
        params.srcStride = params.srcStride >> 1;
        params.dstStride = params.dstStride >> 1;
        padParams.rightPadding = padParams.rightPadding >> 1;
    }
    X_LOG("DataCopyPad2D blockCount %d blockLen %d srcStride %d dstStride %d", params.blockCount, params.blockLen,
          params.srcStride, params.dstStride);
#if defined(__CCE_KT_TEST__)
    ASCENDC_ASSERT(params.blockLen > 0, { X_LOG("blockLen[%d] should be larger than 0.", params.blockLen); });
#endif
    DataCopyPad(dst, src, params, padParams);
}

} // namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass
#endif
