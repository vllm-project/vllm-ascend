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
 * \file mma.h
 * \brief
 */

#ifndef INCLUDE_MMA_H
#define INCLUDE_MMA_H

#include "hardware.h"
#include "kernel_tensor.h"

template <ArchType ArchTag, typename ElementA, typename ElementB, typename AccDTypeC, bool IsTransposeA>
struct mmad {
    __aicore__ mmad(AscendC::LocalTensor<AccDTypeC> l0cTensor,
                    AscendC::LocalTensor<ElementA> l0aTensor,
                    AscendC::LocalTensor<ElementB> l0bTensor,
                    uint32_t mTileActual,
                    uint32_t nTileActual,
                    uint32_t kPartActual,
                    bool initC) {};

    __aicore__ mmad(AscendC::LocalTensor<AccDTypeC> l0cTensor,
                    AscendC::LocalTensor<ElementA> l0aTensor,
                    AscendC::LocalTensor<ElementB> l0bTensor,
                    uint64_t biasBt,
                    uint32_t mTileActual,
                    uint32_t nTileActual,
                    uint32_t kPartActual,
                    bool initC) {};
};

// Partial specialization for V220, int8_t, not_vector_A, not TransposeA
template <ArchType ArchTag, typename AccDTypeC, typename ElementA, typename ElementB>
struct mmad<ArchTag, ElementA, ElementB, AccDTypeC, false> {
    __aicore__ mmad(AscendC::LocalTensor<AccDTypeC> l0cTensor,
                    AscendC::LocalTensor<ElementA> l0aTensor,
                    AscendC::LocalTensor<ElementB> l0bTensor,
                    uint32_t mTileActual,
                    uint32_t nTileActual,
                    uint32_t kPartActual,
                    bool initC)
    {
        AscendC::Mmad(l0cTensor,
                      l0aTensor,
                      l0bTensor,
                      AscendC::MmadParams(mTileActual, nTileActual, kPartActual, 0, false, initC));
    };

    __aicore__ mmad(AscendC::LocalTensor<AccDTypeC> l0cTensor,
                    AscendC::LocalTensor<ElementA> l0aTensor,
                    AscendC::LocalTensor<ElementB> l0bTensor,
                    uint64_t biasBt,
                    uint32_t mTileActual,
                    uint32_t nTileActual,
                    uint32_t kPartActual,
                    bool initC)
    {
        AscendC::LocalTensor<ElementA> biasTensor;
        biasTensor.InitBuffer(biasBt, mTileActual);
        biasTensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::C2);
        AscendC::Mmad(l0cTensor,
                      l0aTensor,
                      l0bTensor,
                      biasTensor,
                      AscendC::MmadParams(mTileActual, nTileActual, kPartActual, 0, false, initC));
    };
};

#endif