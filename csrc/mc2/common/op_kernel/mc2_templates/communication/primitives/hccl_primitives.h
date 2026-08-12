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
 * \file hccl_primitives.h
 * \brief
 */

#ifndef MC2_HCCL_PRIMITIVES_H
#define MC2_HCCL_PRIMITIVES_H

#include "lib/hccl/hccl.h"

namespace MC2KernelTemplate {
struct MC2AlltoAllContext {
    uint32_t taskCnt;
    GM_ADDR sendBuffer;
    GM_ADDR recvBuffer;
    uint64_t sendOffset;
    uint64_t recvOffset;
    uint64_t sendCount;
    uint64_t strideCount;
    uint64_t hcclDataType;
    uint8_t repeat = 1;
};

template <typename HcclType>
class MC2AlltoAllPrimitives {
public:
    __aicore__ inline AscendC::HcclHandle Prepare(HcclType *hccl, MC2AlltoAllContext *context, uint32_t taskIndex)
    {
        return hccl->template AlltoAll<false>(context->sendBuffer + taskIndex * context->sendOffset,
                                              context->recvBuffer + taskIndex * context->recvOffset, context->sendCount,
                                              (AscendC::HcclDataType)(static_cast<uint8_t>(context->hcclDataType)),
                                              context->strideCount, context->repeat);
    }

    __aicore__ inline AscendC::HcclHandle SyncSend(HcclType *hccl, MC2AlltoAllContext *context, uint32_t taskIndex)
    {
        return hccl->template AlltoAll<true>(context->sendBuffer + taskIndex * context->sendOffset,
                                             context->recvBuffer + taskIndex * context->recvOffset, context->sendCount,
                                             (AscendC::HcclDataType)(static_cast<uint8_t>(context->hcclDataType)),
                                             context->strideCount, context->repeat);
    }
};

}; // namespace MC2KernelTemplate
#endif
