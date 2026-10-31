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
 * \file l0c_to_ub_iterator.h
 * \brief
 */
#ifndef L0C_TO_UB_ITERATOR_H
#define L0C_TO_UB_ITERATOR_H

#include "iterator.h"

/////////////////////////////////////////////////////
// l0c_to_ub
/////////////////////////////////////////////////////

// Partial specialization ZN, half, int32_t
template <ArchType ArchTag, typename ElementIn, typename ElementOut, bool MatrixMode = true> struct l0c_to_ub {
    __aicore__ l0c_to_ub(AscendC::LocalTensor<ElementOut> ubTensor, AscendC::LocalTensor<ElementIn> l0cTensor,
                         uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride)
    {
        constexpr auto mode =
            MatrixMode ? AscendC::BlockMode::BLOCK_MODE_MATRIX : AscendC::BlockMode::BLOCK_MODE_VECTOR;
        AscendC::DataCopy(ubTensor, l0cTensor,
                          AscendC::DataCopyParams(nBurst,                              // count
                                                  lenBurst,                            // len
                                                  srcStride,                           // srcStrideIn
                                                  dstStride),                          // dstStrideIn
                          AscendC::DataCopyEnhancedParams(mode,                        // blockModeIn
                                                          AscendC::DeqScale::DEQ_NONE, // deqScaleIn
                                                          0,                           // deqValueIn
                                                          0,                           // sidStoreModeIn
                                                          false,                       // isReluIn
                                                          pad_t::PAD_NONE,             // padModeIn
                                                          0)                           // padValueIn
        );
    };
};

template <ArchType ArchTag>
struct l0c_to_ub<ArchTag, int32_t, half> {
    __aicore__ l0c_to_ub(AscendC::LocalTensor<half> ubTensor,
                         AscendC::LocalTensor<int32_t> l0cTensor,
                         uint16_t nBurst,
                         uint16_t lenBurst,
                         uint16_t srcStride,
                         uint16_t dstStride)
    {
        AscendC::DataCopy(ubTensor, l0cTensor,
                          AscendC::DataCopyParams(nBurst,                                        // count
                                                  lenBurst,                                      // len
                                                  srcStride,                                     // srcStrideIn
                                                  dstStride),                                    // dstStrideIn
                          AscendC::DataCopyEnhancedParams(AscendC::BlockMode::BLOCK_MODE_MATRIX, // blockModeIn
                                                          AscendC::DeqScale::VDEQ16,             // deqScaleIn
                                                          0,                                     // deqValueIn
                                                          0,                                     // sidStoreModeIn
                                                          false,                                 // isReluIn
                                                          pad_t::PAD_NONE,                       // padModeIn
                                                          0)                                     // padValueIn
        );
    };
};

#endif // L0C_TO_UB_ITERATOR_H