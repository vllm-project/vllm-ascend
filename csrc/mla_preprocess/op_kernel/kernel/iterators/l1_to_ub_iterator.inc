/*  Adapted from
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
// l1_to_ub
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DataType>
struct l1_to_ub {
    __aicore__ l1_to_ub(AscendC::LocalTensor<DataType> ubTensor,
                        AscendC::LocalTensor<DataType> l1Tensor,
                        uint16_t nBurst,
                        uint16_t lenBurst,
                        uint16_t srcStride,
                        uint16_t dstStride)
    {
        AscendC::DataCopy(ubTensor, l1Tensor, AscendC::DataCopyParams(nBurst, lenBurst, srcStride, dstStride));
    };
};

/////////////////////////////////////////////////////
// ub_to_l1
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DataType>
struct ub_to_l1 {
    __aicore__ ub_to_l1(AscendC::LocalTensor<DataType> l1Tensor,
                        AscendC::LocalTensor<DataType> ubTensor,
                        uint16_t nBurst,
                        uint16_t lenBurst,
                        uint16_t srcStride,
                        uint16_t dstStride)
    {
        AscendC::DataCopy(l1Tensor, ubTensor, AscendC::DataCopyParams(nBurst, lenBurst, srcStride, dstStride));
    };
};
