/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef L0_FLASH_ATTN_METADATA_H
#define L0_FLASH_ATTN_METADATA_H

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *FlashAttnMetadata(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKvOptional,
                                   const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional,
                                   int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv, int64_t numHeadsQ,
                                   int64_t numHeadsKv, int64_t headDim, int64_t headDimV, int64_t maskMode,
                                   int64_t winLeft, int64_t winRight, const char *layoutQ, const char *layoutKv,
                                   const char *layoutOut, bool isGradEnabled, const char *socVersion,
                                   int64_t aicCoreNum, int64_t aivCoreNum, const aclTensor *metaData,
                                   aclOpExecutor *executor);
} // namespace l0op

#endif
