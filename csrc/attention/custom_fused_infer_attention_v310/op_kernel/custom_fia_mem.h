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
 * \file custom_fia_mem.h
 * \brief 310P-specific local buffer initialization.
 *
 * Why this file exists:
 *   The common mem.h (csrc/common/include/kernel/mem.h) guards the V200 buffer
 *   initialization with #ifndef __clang__. Bisheng CCE defines __clang__, so on
 *   310P the UB/CB/L0A/L0B/L0C local storage is never initialized, causing NaN
 *   outputs and AICore exceptions.
 *
 *   FiaV200Buffer uses InitBuffer() + address_.logicPos (same as the
 *   Ascend_Ops reference), which works under both GCC and Bisheng CCE,
 *   independent of __clang__.
 *
 *   The dedicated type keeps the fix local to CustomFusedInferAttentionV310.
 *   The common AsdopsBuffer and all other operators remain unchanged.
 */

#ifndef CUSTOM_FIA_MEM_H
#define CUSTOM_FIA_MEM_H

#include "mem.h"

class FiaV200Buffer {
public:
    __aicore__ FiaV200Buffer()
    {
        constexpr uint32_t bufferSize[(uint32_t)BufferType::ASCEND_MAX] = {
            HardwareInfo<ArchType::ASCEND_V200>::ubSize,
            HardwareInfo<ArchType::ASCEND_V200>::l1Size,
            HardwareInfo<ArchType::ASCEND_V200>::l0ASize,
            HardwareInfo<ArchType::ASCEND_V200>::l0BSize,
            HardwareInfo<ArchType::ASCEND_V200>::l0CSize};
        tensor[(uint32_t)BufferType::ASCEND_UB].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_UB]);
        tensor[(uint32_t)BufferType::ASCEND_UB].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
        tensor[(uint32_t)BufferType::ASCEND_CB].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_CB]);
        tensor[(uint32_t)BufferType::ASCEND_CB].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::A1);
        tensor[(uint32_t)BufferType::ASCEND_L0A].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_L0A]);
        tensor[(uint32_t)BufferType::ASCEND_L0A].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::A2);
        tensor[(uint32_t)BufferType::ASCEND_L0B].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_L0B]);
        tensor[(uint32_t)BufferType::ASCEND_L0B].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::B2);
        tensor[(uint32_t)BufferType::ASCEND_L0C].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_L0C]);
        tensor[(uint32_t)BufferType::ASCEND_L0C].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::CO1);
    }

    template <BufferType BufferType_, typename DstDataType = half>
    __aicore__ AscendC::LocalTensor<DstDataType> GetBuffer(const uint32_t offset) const
    {
        return tensor[(uint32_t)BufferType_][offset].template ReinterpretCast<DstDataType>();
    }

private:
    AscendC::LocalTensor<uint8_t> tensor[(uint32_t)BufferType::ASCEND_MAX];
};
#endif
