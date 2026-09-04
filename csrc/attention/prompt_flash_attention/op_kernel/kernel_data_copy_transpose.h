/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_data_copy_transpose.h
 * \brief
 */
#ifndef KERNEL_DATA_COPY_TRANSPOSE_H
#define KERNEL_DATA_COPY_TRANSPOSE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
using namespace AscendC;

enum class CopyTransposeType {
    TRANSPOSE_TYPE_NONE,  // Default value
    TRANSPOSE_NZ2ND_0213, // { shape:[B, A1, A3 / 16, A2 / 16, 16, 16], format:"NZ"} -->{ shape:[B, A2, A1, A3],
                          // ori_shape:[B, A2, A1, A3], format:"ND"}
    TRANSPOSE_NZ2NZ_0213, // { shape:[B, A1, A3 / 16, A2 / 16, 16, 16], format:"NZ"}-->{ shape:[B, A2, A3 / 16, A1 / 16,
                          // 16, 16], origin_shape:[B, A2, A1, A3], format:"NZ"}
    TRANSPOSE_NZ2NZ_012_WITH_N,    // { shape:[B, H / 16, S / 16, 16, 16], format:"NZ"}-->{ shape:[B, N, H/N/16, S / 16,
                                   // 16, 16], ori_shape:[B, N, S, H/N], format:"NZ"}
    TRANSPOSE_NZ2ND_012_WITH_N,    // { shape:[B, H / 16, S / 16, 16, 16], format:"NZ"}-->{ shape:[B, N, S, H/N],
                                   // ori_shape:[B, N, S, H/N], format:"ND"}
    TRANSPOSE_NZ2ND_012_WITHOUT_N, // { shape:[B, N, H/N/16, S/16, 16, 16], format:"NZ"}-->{ shape:[B, S, H],
                                   // ori_shape:[B, S, H], format:"ND"}
    TRANSPOSE_NZ2NZ_012_WITHOUT_N, // { shape:[B, N, H/N/16, S/16, 16, 16], format:"NZ"}-->{ shape:[B, H/16, S/16, 16,
                                   // 16], ori_shape:[B, S, H], format:"NZ"}
    TRANSPOSE_ND2ND_ONLY,          // { shape:[H, W], format:"ND"} -->{ shape:[W, H], format:"ND"}
    TRANSPOSE_ND_UB_GM,            //  [B, N, S, H/N] -> [B, S, H]
};

__aicore__ inline void DataCopyUB2GMAlign(const GlobalTensor<half> &dstGlobal, const LocalTensor<half> &srcLocal,
    uint16_t nBurst, uint32_t lenBurst, uint8_t srcGap, uint8_t dstGap)
{
    if (g_coreType == AIV) {
        DataCopyPad(dstGlobal, srcLocal, DataCopyExtParams(nBurst, lenBurst, srcGap, dstGap, 0));
    }
}

#if (__CCE_AICORE__ > 200)
__aicore__ inline void DataCopyUB2GMAlign(const GlobalTensor<bfloat16_t> &dstGlobal, const LocalTensor<bfloat16_t> &srcLocal,
    uint16_t nBurst, uint32_t lenBurst, uint8_t srcGap, uint8_t dstGap)
{
    if (g_coreType == AIV) {
        DataCopyPad(dstGlobal, srcLocal, DataCopyExtParams(nBurst, lenBurst, srcGap, dstGap, 0));                            
    }
}
#endif

__aicore__ inline void DataCopyUB2GMAlign(const GlobalTensor<float> &dstGlobal, const LocalTensor<float> &srcLocal,
    uint16_t nBurst, uint32_t lenBurst, uint8_t srcGap, uint8_t dstGap)
{
    if (g_coreType == AIV) {
        DataCopyPad(dstGlobal, srcLocal, DataCopyExtParams(nBurst, lenBurst, srcGap, dstGap, 0));
    }
}

__aicore__ inline void DataCopyUB2GMAlign(const GlobalTensor<int8_t> &dstGlobal, const LocalTensor<int8_t> &srcLocal,
    uint16_t nBurst, uint32_t lenBurst, uint8_t srcGap, uint8_t dstGap)
{
    if (g_coreType == AIV) {
        DataCopyPad(dstGlobal, srcLocal, DataCopyExtParams(nBurst, lenBurst, srcGap, dstGap, 0));                            
    }
}

using TransposeParams = struct TransposeParams {
    int64_t bIndex;
    int64_t nIndex;
    int64_t sIndex;
    int64_t hNIndex;
}; // Index the position of the segmented small block within the original large block

template <typename T>
__aicore__ inline void DataCopyTranspose(const GlobalTensor<T> &dstGlobal, const LocalTensor<T> &srcLocal,
    CopyTransposeType transposeType, TransposeParams transposeParams, CopyTransposeTiling tiling)
{
    if (transposeType != CopyTransposeType::TRANSPOSE_ND_UB_GM) {
        return;
    }
    if (tiling.dstShapeB == 0) {
        return;
    }

    int startAddr = transposeParams.bIndex * (tiling.shapeSHValue) + transposeParams.nIndex * tiling.dstShapeHN +
        transposeParams.sIndex * tiling.dstShapeH + transposeParams.hNIndex;

    for (int i = 0; i < (int)tiling.srcShapeB; i++) {
        for (int j = 0; j < (int)tiling.srcShapeN; j++) {
            for (int k = 0; k < (int)tiling.srcShapeS; k++) {
                DataCopyUB2GMAlign(dstGlobal[startAddr + i * (tiling.shapeSHValue) + j * tiling.dstShapeHN
                    + k * tiling.dstShapeH], srcLocal[k * tiling.srcShapeHN + j * (tiling.shapeNsValue)
                    + i * (tiling.shapeNsnValue)], 1,
                    tiling.originalShapeNLen, 0, 0);
            }
        }
    }
}

template <typename T>
__aicore__ inline void DataCopyTranspose2(const GlobalTensor<T> &dstGlobal, const LocalTensor<T> &srcLocal,
    CopyTransposeType transposeType, TransposeParams transposeParams,
    CopyTransposeTiling tiling, int64_t multi_seq_offset)
{
    if (transposeType != CopyTransposeType::TRANSPOSE_ND_UB_GM) {
        return;
    }
    if (tiling.dstShapeB == 0) {
        return;
    }

    int64_t startAddr = multi_seq_offset + transposeParams.nIndex * tiling.dstShapeHN +
        transposeParams.sIndex * tiling.dstShapeH + transposeParams.hNIndex;

    for (int i = 0; i < (int)tiling.srcShapeB; i++) {
        for (int j = 0; j < (int)tiling.srcShapeN; j++) {
            for (int k = 0; k < (int)tiling.srcShapeS; k++) {
                DataCopyUB2GMAlign(dstGlobal[startAddr + i * (tiling.shapeSHValue) + j * tiling.dstShapeHN +
                k * tiling.dstShapeH], srcLocal[k * tiling.srcShapeHN + j * (tiling.shapeNsValue) +
                i * (tiling.shapeNsnValue)], 1, tiling.originalShapeNLen, 0, 0);
            }
        }
    }
}

#endif // KERNEL_DATA_COPY_TRANSPOSE_H