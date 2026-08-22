

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
 * \file mhc_post_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_MHCPOST_H_
#define OPS_OP_PROTO_INC_MHCPOST_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Fuse the main branch feature h_out with the residual branch feature x
* using the gating mechanism h_post and the doubly stochastic matrix  h_res
* to enable information flow.
* @par Inputs:
* @li x: A Tensor. Represents input data in the mHC layer. Type is:BFloat16 or Float16. \n
* Dataformat:ND. Supports 3D (Shape:[T, n, D]) or 4D (Shape:[B, S, n, D]) tensors.
* @li h_res: A Tensor. Represents the h_res transformation matrix. Type is:Float32. \n 
* Dataformat:ND. Supports 3D (Shape:[T, n, n]) or 4D (Shape:[B, S, n, n]) tensors.
* @li h_out: A Tensor. Represents output of the Atten/MLP layer. Type is:BFloat16 or Float16. \n
* Dataformat:ND. Supports 2D (Shape:[T, D]) or 3D (Shape:[B, S, D]) tensors.
* @li h_post: A Tensor. Represents the h_res transformation matrix. Type is:Float32. \n
* Dataformat:ND. Supports 2D (Shape:[T, n]) or 3D (Shape:[B, S, n]) tensors.
* @par Outputs:
* y: A Tensor. Type is:BFloat16 or Float16. Dataformat:ND. \n
* Supports 3D (Shape:[T, n, D]) or 4D (Shape:[B, S, n, D]) tensors. 
*/
REG_OP(MhcPost)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(h_res, TensorType({DT_FLOAT}))
    .INPUT(h_out, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(h_post, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(MhcPost)

} // namespace ge

#endif // OPS_OP_PROTO_INC_MHCPOST_H_
