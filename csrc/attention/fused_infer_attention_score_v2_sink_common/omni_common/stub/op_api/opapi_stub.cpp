/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file op_legacy_api.cpp
 * \brief
 */

#include "opdev/op_executor.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/pad.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/slice.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"

#include "level0/add.h"
#include "level0/axpy.h"
#include "level0/broadcast_to.h"
#include "level0/dot.h"
#include "level0/fill.h"
#include "level0/mul.h"
#include "level0/muls.h"
#include "level0/reduce_mean.h"
#include "level0/padv3.h"
#include "level0/sort.h"
#include "level0/dilation.h"
#include "level0/zero_op.h"
#include "level0/squeeze.h"
#include "level0/unsqueeze.h"

namespace l0op {
const aclTensor *TensorMove(const aclTensor *x, const aclTensor * /* executor */, aclOpExecutor * /* executor */)
{
    return x;
}
const aclTensor *ZerosLike(const aclTensor *self, aclOpExecutor * /* executor */)
{
    return self;
}
const aclTensor *Maximum(const aclTensor *self, const aclTensor * /* executor */, aclOpExecutor * /* executor */)
{
    return self;
}
const aclTensor *GatherV2(const aclTensor *self, int64_t /* executor */,
                          const aclTensor * /* executor */,
                          aclOpExecutor * /* executor */,
                          int /* batchDims = 0 */,
                          bool /* negativeIndexSupport = false */)
{
    return self;
}

const aclTensor *GatherV2WithImplMode(const aclTensor *self, int64_t /* executor */, const aclTensor * /* executor */,
                                      int64_t /* executor */, aclOpExecutor * /* executor */, int /* batchDims = 0 */,
                                      bool /* negativeIndexSupport = false */)
{
    return self;
}
const aclTensor *GatherElements(const aclTensor *self, const int64_t /* executor */, const aclTensor * /* executor */,
                                aclOpExecutor * /* executor */)
{
    return self;
}
const aclTensor *Minimum(const aclTensor *self, const aclTensor * /* executor */, aclOpExecutor * /* executor */)
{
    return self;
}
const aclTensor *Cast(const aclTensor *self, op::DataType /* executor */, aclOpExecutor * /* executor */)
{
    return self;
}

const aclTensor *CastOnlyForConvBackward(const aclTensor *self,
                                         op::DataType /* executor */,
                                         aclOpExecutor * /* executor */)
{
    return self;
}

const aclTensor *Contiguous(const aclTensor *x, aclOpExecutor * /* executor */)
{
    return x;
}

const aclTensor *ViewCopy(const aclTensor *x, const aclTensor * /* executor */, aclOpExecutor * /* executor */)
{
    return x;
}

const aclTensor *PickViewAsContiguous(const aclTensor *x, aclOpExecutor * /* executor */)
{
    return x;
}

const aclTensor *ReViewToOut(const aclTensor *x, const aclTensor * /* executor */, aclOpExecutor * /* executor */)
{
    return x;
}

const std::tuple<aclTensor *, aclTensor *> Sort(const aclTensor * self, int64_t /* executor */, bool /* executor */,
                                                bool /* executor */, op::DataType /* executor */,
                                                aclOpExecutor * /* executor */)
{
    return std::tuple<aclTensor *, aclTensor *>(const_cast<aclTensor *>(self), const_cast<aclTensor *>(self));
}

bool CanOptimizeContiguous(const op::Shape & /* executor */, const op::Strides & /* executor */, int64_t /* executor */,
                           int64_t /* executor */, ContiguousParam & /* executor */)
{
    return true;
}

bool CanOptimizeView(const op::Shape & /* executor */, const op::Strides & /* executor */, int64_t /* executor */,
                     ContiguousParam & /* executor */)
{
    return true;
}

const aclTensor *Pad(const aclTensor *self, const aclTensor * /* executor */, aclOpExecutor * /* executor */)
{
    return self;
}

const aclTensor *Reshape(const aclTensor *x, const op::Shape & /* executor */, aclOpExecutor * /* executor */)
{
    return x;
}

const aclTensor *Reshape(const aclTensor *x, const aclIntArray * /* executor */, aclOpExecutor * /* executor */)
{
    return x;
}

const aclTensor *Slice(const aclTensor *x, const aclTensor * /* executor */, const aclTensor * /* executor */,
                       const aclTensor * /* executor */, aclOpExecutor * /* executor */)
{
    return x;
}

const aclTensor *Slice(const aclTensor *x, const aclIntArray * /* executor */, const aclIntArray * /* executor */,
                       aclOpExecutor * /* executor */)
{
    return x;
}

const aclTensor *ReFormat(const aclTensor *x, const op::Format & /* executor */, aclOpExecutor * /* executor */)
{
    return x;
}

const aclTensor *TransData(const aclTensor *x, op::Format /* executor */, int64_t /* executor */,
                           aclOpExecutor * /* executor */)
{
    return x;
}

const aclTensor *TransDataSpecial(const aclTensor *x, op::Format /* executor */, int64_t /* executor */,
                                  aclOpExecutor * /* executor */)
{
    return x;
}

const aclTensor *Transpose(const aclTensor *x, const aclTensor * /* executor */, const aclTensor * /* executor */,
                           aclOpExecutor * /* executor */)
{
    return x;
}
const aclTensor *Transpose(const aclTensor *x, const aclIntArray * /* executor */, aclOpExecutor * /* executor */)
{
    return x;
}

const aclTensor *Add(const aclTensor *self, const aclTensor * /* executor */, aclOpExecutor * /* executor */)
{
    return self;
}
const aclTensor *Axpy(const aclTensor *self, const aclTensor * /* executor */,
                      float /* executor */, aclOpExecutor * /* executor */)
{
    return self;
}

const aclTensor *BroadcastTo(const aclTensor *x, const aclTensor * /* executor */, const aclTensor * /* executor */,
                             aclOpExecutor * /* executor */)
{
    return x;
}
const aclTensor *BroadcastTo(const aclTensor *x, const aclIntArray * /* executor */, aclOpExecutor * /* executor */)
{
    return x;
}
const aclTensor *Dot(const aclTensor *self, const aclTensor * /* executor */, aclOpExecutor * /* executor */)
{
    return self;
}
const aclTensor *Fill(const aclTensor * /* executor */, const aclTensor *value, const aclIntArray * /* executor */,
                      aclOpExecutor * /* executor */)
{
    return value;
}
const aclTensor *Mul(const aclTensor *self, const aclTensor * /* executor */, aclOpExecutor * /* executor */)
{
    return self;
}
const aclTensor *Muls(const aclTensor *self, float /* executor */, aclOpExecutor * /* executor */)
{
    return self;
}
const aclTensor *ReduceMean(const aclTensor *self, const aclIntArray * /* executor */, bool /* executor */,
                            aclOpExecutor * /* executor */)
{
    return self;
}
const aclTensor *ReduceMean(const aclTensor *self, const aclIntArray * /* executor */, bool /* executor */,
                            bool /* executor */, aclOpExecutor * /* executor */)
{
    return self;
}
const aclTensor *Dilation(const aclTensor *x, const aclIntArray * /* executor */, const aclIntArray * /* executor */,
                          float /* executor */, aclOpExecutor * /* executor */)
{
    return x;
}
const aclTensor *Shape_op(const aclTensor *x, aclOpExecutor * /* executor */)
{
    return x;
}

const aclTensor *SqueezeNd(const aclTensor *x, const aclIntArray * /* executor */, aclOpExecutor * /* executor */)
{
    return x;
}
const aclTensor *SqueezeNd(const aclTensor *x, int64_t /* executor */, aclOpExecutor * /* executor */)
{
    return x;
}

const aclTensor *PadV3(const aclTensor *self, const aclTensor * /* executor */, const aclTensor * /* executor */,
                       const std::string & /* executor */, const bool /* executor */, aclOpExecutor * /* executor */)
{
    return self;
}

const aclTensor *UnsqueezeNd(const aclTensor *x, const aclIntArray * /* executor */, aclOpExecutor * /* executor */)
{
    return x;
}

const aclTensor *UnsqueezeNd(const aclTensor *x, int64_t /* executor */, aclOpExecutor * /* executor */)
{
    return x;
}

const aclTensor *ReduceSumOp(const aclTensor *x, const aclIntArray * /* executor */, bool /* executor */,
                             aclOpExecutor * /* executor */)
{
    return x;
}

const aclTensor *MaskedScatter(const aclTensor *self,
                               const aclTensor * /* executor */,
                               const aclTensor * /* executor */,
                               aclOpExecutor * /* executor */)
{
    return self;
}

const aclTensor *InplaceIndexAddAiCore(const aclTensor *self,
                                       const int64_t /* executor */,
                                       const aclTensor * /* executor */,
                                       const aclTensor * /* executor */,
                                       const aclTensor * /* executor */,
                                       aclOpExecutor * /* executor */)
{
    return self;
}

const aclTensor *InplaceIndexAddAiCpu(const aclTensor *self,
                                      const int64_t /* executor */,
                                      const aclTensor * /* executor */,
                                      const aclTensor * /* executor */,
                                      const aclTensor * /* executor */,
                                      aclOpExecutor * /* executor */)
{
    return self;
}

const aclTensor *InplaceIndexAddWithSorted(const aclTensor * self, const int64_t /* executor */,
                                           const aclTensor * /* executor */, const aclTensor * /* executor */,
                                           const aclTensor * /* executor */, const aclTensor * /* executor */,
                                           aclOpExecutor * /* executor */)
{
    return self;
}

const aclTensor *GatherV3(const aclTensor *self, int64_t axis, const aclTensor *indices, aclOpExecutor *executor,
                        int batchDims = 0, bool negativeIndexSupport = false)
{
    (void)self;
    (void)axis;
    (void)indices;
    (void)executor;
    (void)batchDims;
    (void)negativeIndexSupport;
    return self;
}

} // namespace l0op
