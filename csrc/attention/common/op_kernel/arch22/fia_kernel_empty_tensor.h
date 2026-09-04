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
 * \file fia_kernel_empty_tensor.h
 * \brief
 */

#ifndef FIA_KERNEL_EMPTY_TENSOR_H
#define FIA_KERNEL_EMPTY_TENSOR_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../fia_public_define.h"
#include "../vector_common.h"
#include "kernel_common.h"
#include "../memory_copy.h"

static constexpr float FLOAT_INF = 3e+99;

template <typename T>
class FiaKernelEmptyTensor {
public:
    __aicore__ inline FiaKernelEmptyTensor(){};
    __aicore__ inline void Init(__gm__ uint8_t *attentionOut,
                                __gm__ uint8_t *softmaxLse,
                                const FusedInferAttentionScoreEmptyTensorTilingData *__restrict tiling,
                                TPipe *tPipe);
    __aicore__ inline void Process();

protected:
    // ==============================TilingData&TPipe==============================
    const FusedInferAttentionScoreEmptyTensorTilingData *__restrict tilingData = nullptr;
    TPipe *pipe = nullptr;

    // ================================Required Global Tensor=================================
    GlobalTensor<T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;
    // ================================类成员变量====================================
    uint32_t usedCoreNum = 0U;
};

template <typename T>
__aicore__ inline void FiaKernelEmptyTensor<T>::Process()
{
    if ASCEND_IS_AIV {
        uint32_t initOutputEventId = 0U;
        SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        uint32_t tmpBlockIdx = GetBlockIdx();
        usedCoreNum = tilingData->usedCoreNum;
        uint64_t totalOutputSize = tilingData->totalOutputSize;
        uint64_t singleCoreSize = tilingData->singleCoreSize;
        WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        if (totalOutputSize > tmpBlockIdx * singleCoreSize) {
            uint64_t tailSize = totalOutputSize - tmpBlockIdx * singleCoreSize;
            uint64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;
            if (singleInitOutputSize > 0) {
                matmul::InitOutput<T>(attentionOutGm[tmpBlockIdx * singleCoreSize], singleInitOutputSize, 0);
            }
        }
        SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);

        if (tilingData->softmaxLseFlag) {
            float lseInitValue = FLOAT_INF;
            uint64_t totalLseSize = tilingData->totalLseSize;
            uint64_t singleCoreLseSize = tilingData->singleCoreLseSize;
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            if (totalLseSize > tmpBlockIdx * singleCoreLseSize) {
                uint64_t tailLseSize = totalLseSize - tmpBlockIdx * singleCoreLseSize;
                uint64_t singleInitOutputLseSize = tailLseSize < singleCoreLseSize ? tailLseSize : singleCoreLseSize;
                if (singleInitOutputLseSize > 0) {
                matmul::InitOutput<float>(softmaxLseGm[tmpBlockIdx * singleCoreLseSize], singleInitOutputLseSize, lseInitValue);
                }
            }
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
        WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
    }
}

template <typename T>
__aicore__ inline void FiaKernelEmptyTensor<T>::Init(
                                __gm__ uint8_t *attentionOut,
                                __gm__ uint8_t *softmaxLse,
                                const FusedInferAttentionScoreEmptyTensorTilingData *__restrict tiling,
                                TPipe *tPipe)
{
    pipe = tPipe;
    // init tiling data
    tilingData = tiling;

    // init global buffer
    attentionOutGm.SetGlobalBuffer((__gm__ T *)attentionOut);
    if (tilingData->softmaxLseFlag) {
        softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
    }
}

#endif // FIA_KERNEL_EMPTY_TENSOR_H