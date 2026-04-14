/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_INC_EXTERNAL_ACLNN_KERNELS_TRANSDATA_H
#define COMMON_INC_EXTERNAL_ACLNN_KERNELS_TRANSDATA_H

#include "opdev/op_executor.h"

namespace l0op {

const aclTensor* ReFormat(const aclTensor* x, const op::Format& format, aclOpExecutor* executor = nullptr);

/**
 * TransData
 * Formal Transdata. Set the c0 size strictly based on the data type and chip block size.
 * support data type as follows: fp16,fp32,int32,uint32,int8,uint8
 * fp16: block_size/2
 * fp32/int32/uint32: block_size/4 (this is different from `TransDataSpecial`)
 * int8/uint8: block_size/1
 *
 * @param x : aclTensor need to transpose
 * @param dstPrimaryFormat: dstPrimaryFormat like NC1HWC0
 * @param groups: groups
 * @param executor: executor should not be null
 * @return trans format tensor
 */
const aclTensor* TransData(const aclTensor* x, op::Format dstPrimaryFormat, int64_t groups, aclOpExecutor* executor);
/**
 * Special Transdata. Set the c0 size strictly based on the data type and chip block size.
 * this transdata c0 size rule:
 *     fp16: block_size/2
 *     fp32/int32/uint32: block_size/2
 *     int8/uint8: block_size/1
 * bool not supported, should do:
 *     (NCHW, bool)-> cast -> (NCHW, fp16) -> TransDataSpecial -> (5HD, fp16) -> cast -> (5HD, bool)
 *     (5HD, bool)-> cast -> (5HD, fp16) -> TransDataSpecial -> (NCHW, fp16) -> cast -> (NCHW, bool)
 *
 * @param x : aclTensor need to transpose
 * @param dstPrimaryFormat: dstPrimaryFormat like NC1HWC0
 * @param groups: groups
 * @param executor: executor should not be null
 * @return trans format tensor
 */
const aclTensor* TransDataSpecial(
    const aclTensor* x, op::Format dstPrimaryFormat, int64_t groups, aclOpExecutor* executor);

} // namespace l0op

#endif // COMMON_INC_EXTERNAL_ACLNN_KERNELS_TRANSDATA_H
