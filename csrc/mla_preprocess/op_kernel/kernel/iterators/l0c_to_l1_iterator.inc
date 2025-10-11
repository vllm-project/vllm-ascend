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
// l0c_to_l1
/////////////////////////////////////////////////////

// Partial specialization ZN, half, int32_t
template <ArchType ArchTag>
struct l0c_to_l1<ArchTag, DataFormat::ZN, half, int32_t> {
    using ElementOut = half;
    using ElementIn = int32_t;
    __aicore__ l0c_to_l1(AscendC::LocalTensor<ElementOut> l1Tensor,
                         AscendC::LocalTensor<ElementIn> l0cTensor,
                         AscendC::LocalTensor<uint64_t> deqTensor,
                         uint32_t mTileActual,
                         uint32_t nTileActual,
                         uint32_t mTileCeil,
                         uint32_t nActual)
    {
        constexpr uint32_t BLOCK_NUM = 16;
        constexpr uint32_t BLOCK_SIZE = 32;
        AscendC::FixpipeParams<ElementIn> intriParams(
            (nTileActual + BLOCK_NUM - 1) / AscendC::BLOCK_CUBE,
            static_cast<uint16_t>(mTileActual * BLOCK_NUM * sizeof(float) / BLOCK_SIZE),
            0,
            mTileCeil - static_cast<uint16_t>(mTileActual * BLOCK_NUM * sizeof(float) / BLOCK_SIZE) *
                            sizeof(ElementOut) / sizeof(ElementIn));
        intriParams.nz2ndParams = {false, 1, 0, 0, static_cast<uint16_t>(nTileActual)};
        intriParams.quantParams = {QuantMode_t::VDEQF16};
        AscendC::Fixpipe(l1Tensor, l0cTensor, deqTensor, intriParams);
    };
};
