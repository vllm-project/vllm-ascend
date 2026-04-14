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
 * \file l1_to_bt_iterator.h
 * \brief
 */
#ifndef L1_TO_BT_ITERATOR_H
#define L1_TO_BT_ITERATOR_H

#include "iterator.h"

/////////////////////////////////////////////////////
// l1_to_bt
/////////////////////////////////////////////////////

// Partial specialization for V220
template <ArchType ArchTag, typename DataType>
struct l1_to_bt {
    using HardwareParams = HardwareInfo<ArchTag>;
    static constexpr uint32_t BLOCK_SIZE = HardwareParams::btBlockSize / sizeof(DataType);

    __aicore__ l1_to_bt(AscendC::LocalTensor<DataType> biasTableTensor,
                        AscendC::LocalTensor<DataType> biasL1Tensor,
                        uint32_t ntileActual)
    {
        AscendC::DataCopy(
            biasTableTensor, biasL1Tensor, {1, static_cast<uint16_t>(CeilDiv<BLOCK_SIZE>(ntileActual)), 0, 0});
    };
};

#endif // L1_TO_BT_ITERATOR_H