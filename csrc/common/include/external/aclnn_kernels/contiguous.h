/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_INC_EXTERNAL_ACLNN_KERNELS_CONTIGUOUS_H
#define COMMON_INC_EXTERNAL_ACLNN_KERNELS_CONTIGUOUS_H

#include "opdev/op_def.h"
#include "opdev/common_types.h"

namespace l0op {

typedef struct {
    // 每个op::Shape 18ns
    int64_t viewOffset;

    // Transpose
    op::Shape transposeSrcShape;
    op::Shape transposeDstShape;
    op::FVector<int64_t, op::MAX_DIM_NUM> perm;

    // broadcast to
    op::Shape broadcastSrcShape;
    op::Shape broadcastDstShape;
    op::FVector<int64_t, op::MAX_DIM_NUM> shape;

    // slice
    op::Shape sliceSrcShape;
    op::Shape sliceDstShape;
    op::FVector<int64_t, op::MAX_DIM_NUM> offset;
    op::FVector<int64_t, op::MAX_DIM_NUM> size;

    // strided slice
    op::Shape stridedsliceSrcShape;
    op::Shape stridedsliceDstShape;
    op::FVector<int64_t, op::MAX_DIM_NUM> begin;
    op::FVector<int64_t, op::MAX_DIM_NUM> end;
    op::FVector<int64_t, op::MAX_DIM_NUM> strides;

    // optimizer
    bool mayBroadcast;
    bool mayTranspose;
    bool maySlice;
    bool mayStridedslice;
} ContiguousParam;

/**
 * @brief 将非连续Tensor转换为连续Tensor
 * @param x
 * @param executor
 * @return aclTensor 转换后的tensor
 */
const aclTensor* Contiguous(const aclTensor* x, aclOpExecutor* executor);

/**
 * @brief 将连续tensor拷贝到非连续的tensor上
 * @param x
 * @param y
 * @param executor
 * @return aclTensor 转换后的tensor
 */
const aclTensor* ViewCopy(const aclTensor* x, const aclTensor* y, aclOpExecutor* executor);

/**
 * @brief 对Tensor创建一个View，要求Tensor满足PickView的条件
 * @param x 输入Tensor，可以是一整块的非连续Tensor
 * @param executor
 * @return 输出Shape是一个连续Tensor
 */
const aclTensor* PickViewAsContiguous(const aclTensor* x, aclOpExecutor* executor);

const aclTensor* ReViewToOut(const aclTensor* x, const aclTensor* y, aclOpExecutor* executor);

// ============内部接口=============
bool CanOptimizeContiguous(
    const op::Shape& viewShape, const op::Strides& strides, int64_t offset, int64_t storageSize,
    ContiguousParam& param);

bool CanOptimizeView(const op::Shape& viewShape, const op::Strides& strides, int64_t offset, ContiguousParam& param);
// ============内部接口=============
} // namespace l0op

#endif // COMMON_INC_EXTERNAL_ACLNN_KERNELS_CONTIGUOUS_H
