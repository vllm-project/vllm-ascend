/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file init_output.h
 * \brief
 */

#ifndef INIT_OUTPUT_H
#define INIT_OUTPUT_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace AttentionCommon {

template <typename T, uint32_t SYNC_ID, uint32_t POP_BUF_START_ADDR, uint32_t POP_BUF_ELE_SIZE,
          bool ENABLE_LOCK = false>
__aicore__ inline void InitOutput(GlobalTensor<T> outGm, uint64_t totalElementNum, uint32_t vecCoreNum, T initValue)
{
    if ASCEND_IS_AIV {
        uint64_t singleCoreMaxElementNum = (totalElementNum + vecCoreNum - 1U) / vecCoreNum;
        uint32_t tmpBlockIdx = AscendC::GetBlockIdx();
        uint64_t gmOffset = tmpBlockIdx * singleCoreMaxElementNum;
        if (gmOffset < totalElementNum) {
            uint64_t singleCoreActualElementNum = (gmOffset + singleCoreMaxElementNum > totalElementNum) ?
                                                  (totalElementNum - gmOffset) : singleCoreMaxElementNum;
            LocalTensor<T> popBuffer =
                AscendC::LocalTensor<uint8_t>(TPosition::VECIN, POP_BUF_START_ADDR, POP_BUF_ELE_SIZE * sizeof(T))
                    .template ReinterpretCast<T>();

            if constexpr (ENABLE_LOCK) {
                Mutex::Lock<PIPE_V>(SYNC_ID);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_ID);

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_ID);
            }
            AscendC::Duplicate(popBuffer, initValue, POP_BUF_ELE_SIZE);
            if constexpr (ENABLE_LOCK) {
                Mutex::Unlock<PIPE_V>(SYNC_ID);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_ID);
            }

            if constexpr (ENABLE_LOCK) {
                Mutex::Lock<PIPE_MTE3>(SYNC_ID);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_ID);
            }
            uint64_t loopCnt = singleCoreActualElementNum / POP_BUF_ELE_SIZE;
            uint64_t tailSize = singleCoreActualElementNum - loopCnt * POP_BUF_ELE_SIZE;
            for (uint64_t loop = 0; loop < loopCnt; loop++) {
                AscendC::DataCopy(outGm[gmOffset], popBuffer, POP_BUF_ELE_SIZE);
                gmOffset += POP_BUF_ELE_SIZE;
            }
            if (tailSize > 0) {
                AscendC::DataCopyExtParams dataCopyParams;
                dataCopyParams.blockCount = 1;
                dataCopyParams.blockLen = tailSize * sizeof(T);
                dataCopyParams.srcStride = 0;
                dataCopyParams.dstStride = 0;
                AscendC::DataCopyPad(outGm[gmOffset], popBuffer, dataCopyParams);
            }
            if constexpr (ENABLE_LOCK) {
                Mutex::Unlock<PIPE_MTE3>(SYNC_ID);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_ID);

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_ID);
            }
        }
    }
}

} // namespace AttentionCommon

#endif // INIT_OUTPUT_H