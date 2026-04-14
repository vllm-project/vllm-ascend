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
 * \file mem.h
 * \brief
 */

#ifndef INCLUDE_MEM_H
#define INCLUDE_MEM_H

#include "hardware.h"
#include "kernel_event.h"
#include "kernel_tensor.h"
 
enum class BufferType { ASCEND_UB, ASCEND_CB, ASCEND_L0A, ASCEND_L0B, ASCEND_L0C, ASCEND_MAX };

template <BufferType BufferType_>
__aicore__ constexpr AscendC::TPosition GetPosition()
{
    if constexpr (BufferType_ == BufferType::ASCEND_UB) {
        return AscendC::TPosition::VECIN;
    } else if constexpr (BufferType_ == BufferType::ASCEND_CB) {
        return AscendC::TPosition::A1;
    } else if constexpr (BufferType_ == BufferType::ASCEND_L0A) {
        return AscendC::TPosition::A2;
    } else if constexpr (BufferType_ == BufferType::ASCEND_L0B) {
        return AscendC::TPosition::B2;
    } else if constexpr (BufferType_ == BufferType::ASCEND_L0C) {
        return AscendC::TPosition::CO1;
    }
    return AscendC::TPosition::GM;
}

template <ArchType ArchTag>
struct AsdopsBuffer {
public:
    __aicore__ AsdopsBuffer()
    {
        constexpr uint32_t bufferSize[(uint32_t)BufferType::ASCEND_MAX] = {HardwareInfo<ArchTag>::ubSize,
                                                                        HardwareInfo<ArchTag>::l1Size,
                                                                        HardwareInfo<ArchTag>::l0ASize,
                                                                        HardwareInfo<ArchTag>::l0BSize,
                                                                        HardwareInfo<ArchTag>::l0CSize};
#ifdef __DAV_C220_VEC__
        tensor[(uint32_t)BufferType::ASCEND_UB] = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::VECIN, 0, bufferSize[(uint32_t)BufferType::ASCEND_UB]);
#elif __DAV_C220_CUBE__
        tensor[(uint32_t)BufferType::ASCEND_CB] = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::A1, 0, bufferSize[(uint32_t)BufferType::ASCEND_CB]);
        tensor[(uint32_t)BufferType::ASCEND_L0A] = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::A2, 0, bufferSize[(uint32_t)BufferType::ASCEND_L0A]);
        tensor[(uint32_t)BufferType::ASCEND_L0B] = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::B2, 0, bufferSize[(uint32_t)BufferType::ASCEND_L0B]);
        tensor[(uint32_t)BufferType::ASCEND_L0C] = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::CO1, 0, bufferSize[(uint32_t)BufferType::ASCEND_L0C]);
#else
#ifndef __clang__
        tensor[(uint32_t)BufferType::ASCEND_UB] = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::VECIN, 0, bufferSize[(uint32_t)BufferType::ASCEND_UB]);
        tensor[(uint32_t)BufferType::ASCEND_CB] = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::A1, 0, bufferSize[(uint32_t)BufferType::ASCEND_CB]);
        tensor[(uint32_t)BufferType::ASCEND_L0A] = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::A2, 0, bufferSize[(uint32_t)BufferType::ASCEND_L0A]);
        tensor[(uint32_t)BufferType::ASCEND_L0B] = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::B2, 0, bufferSize[(uint32_t)BufferType::ASCEND_L0B]);
        tensor[(uint32_t)BufferType::ASCEND_L0C] = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::CO1, 0, bufferSize[(uint32_t)BufferType::ASCEND_L0C]);
#endif
#endif
    };

    template <BufferType BufferType_, typename DstDataType = half>
    __aicore__ AscendC::LocalTensor<DstDataType> GetBuffer(const uint32_t offset) const
    {
        return tensor[(uint32_t)BufferType_][offset].template ReinterpretCast<DstDataType>();
    }

public:
    AscendC::LocalTensor<uint8_t> tensor[(uint32_t)BufferType::ASCEND_MAX];
};
#endif