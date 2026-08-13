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
 * \file mhc_pre_clamp_sinkhorn_proto.h
 * \brief MhcPreClampSinkhorn operator proto definition
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_MHC_PRE_SINKHORN_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_MHC_PRE_SINKHORN_PROTO_H_
#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief MhcPreClampSinkhorn operator performs Sinkhorn normalization with hierarchical combination
 */
REG_OP(MhcPreClampSinkhorn)
    .INPUT(x, TensorType({DT_BF16}))
    .INPUT(phi, TensorType({DT_FLOAT}))
    .INPUT(alpha, TensorType({DT_FLOAT}))
    .INPUT(bias, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16}))
    .OUTPUT(post, TensorType({DT_FLOAT}))
    .OUTPUT(comb_frag, TensorType({DT_FLOAT}))
    .ATTR(hc_mult, Int, 4)
    .ATTR(num_iters, Int, 20)
    .ATTR(hc_eps, Float, 1e-6f)
    .ATTR(norm_eps, Float, 1e-6f)
    .ATTR(need_backward, Bool, true)
    .ATTR(clamp_min, Float, 0.0f)
    .ATTR(clamp_max, Float, 0.0f)
    .OP_END_FACTORY_REG(MhcPreClampSinkhorn)

} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_MHC_PRE_SINKHORN_PROTO_H_
