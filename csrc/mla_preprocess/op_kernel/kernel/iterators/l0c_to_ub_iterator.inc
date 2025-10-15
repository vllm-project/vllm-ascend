/* Adapted from
 *      https://gitee.com/ascend/ascend-transformer-boost.git
 *
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../iterator.h"

/////////////////////////////////////////////////////
// l0c_to_ub
/////////////////////////////////////////////////////

// Partial specialization ZN, half, int32_t
template <ArchType ArchTag, typename ElementIn, typename ElementOut, bool MatrixMode = true>
struct l0c_to_ub {
    __aicore__ l0c_to_ub(AscendC::LocalTensor<ElementOut> ubTensor,
                         AscendC::LocalTensor<ElementIn> l0cTensor,
                         uint16_t nBurst,
                         uint16_t lenBurst,
                         uint16_t srcStride,
                         uint16_t dstStride)
    {
        constexpr auto mode =
            MatrixMode ? AscendC::BlockMode::BLOCK_MODE_MATRIX : AscendC::BlockMode::BLOCK_MODE_VECTOR;
        AscendC::DataCopy(ubTensor,
                          l0cTensor,
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
        AscendC::DataCopy(ubTensor,
                          l0cTensor,
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
