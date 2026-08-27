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
 * \file stub_ops.h
 * \brief
 */

#ifndef MATH_COMMON_STUB_OPS_H
#define MATH_COMMON_STUB_OPS_H

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
*@brief Input data for other operators. \n

*@par Inputs:
*x: A tensor. \n

*@par Attributes:
*index: Index of the input tensor.The data type must be int32 or int64.
Assume that net has three data nodes, one should be set 0, another should
be set 1, and the left should be set 2. \n

*@par Outputs:
*y: A tensor. \n

*@par Third-party framework compatibility
*Compatible with the Caffe operator Data.
*/
REG_OP(Data)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(index, Int, 0)
    .OP_END_FACTORY_REG(Data)

/**
*@brief Creates a constant tensor from a tensor-like object. This operator is used for inference.
Operator Const has the same definition as operator Constant. \n

*@par Attributes:
*value: Required. The value and type of the resulting tensor, and no restrictions on type. \n

*@par Outputs:
*y: A constant tensor. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Const.
*/
REG_OP(Const)
    .OUTPUT(y, TensorType::ALL())
    .ATTR(value, Tensor, Tensor())
    .OP_END_FACTORY_REG(Const)

/**
*@brief Cast a tensor from src data type to dst data type.

*@par Inputs:
*One input:
* x:An ND or 5HD tensor. Support 1D~8D. Must be one of the following types: bool, float16, float, int8, int32, uint32, uint8, bfloat16, uint1,
   int64, uint64, int16, uint16, double, complex32, complex64, complex128, qint8, quint8, qint16, quint16, qint32,
   hifloat8, float8_e5m2, float8_e4m3fn, float4_e1m2, float4_e2m1.

*@par Attributes:
*dst_type: A required attribute of type int32, specifying the dst data type.

*@par Outputs:
*y:An ND Tensor with same shape as x, and data type is specified by dst_type.

*@attention Constraints:
* @li In the scenario where the data type is converted from float16 to int16: \n
*     If the input data contains inf, inf is converted into the maximum value of int16. \n
*     If the input data contains -inf, -inf is converted into the minimum value of int16. \n
* @li In the scenarios where the data type is converted from INT32 to INT8: \n
*     It can only guarantee that the input data has no precision errors within the range of (-2048, 1920).
* @li Atlas Inference Series Product in the scenarios where the data type is converted from FLOAT32 to INT8: \n
*     It can only guarantee that the input data has no precision errors within the range of (-2048, 1920).
* @li Atlas Inference Series Product in the scenarios where the data type is converted from FLOAT32 to INT64 and from FLOAT32 to UINT8: \n
*     It can only guarantee that the input data has no precision errors within the range of (-2147483648, 2147483583).
* @li Atlas Inference Series Product in the scenarios where the data type is converted from INT64 to FLOAT32: \n
*     It can only guarantee that the input data has no precision errors within the range of (-2147483648, 2147483647).
*/
REG_OP(Cast)
    .INPUT(x, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8,
                          DT_INT64, DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64,
                          DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32, DT_BF16, DT_UINT1,
                          DT_COMPLEX32, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN,
                          DT_FLOAT4_E1M2, DT_FLOAT4_E2M1}))
    .OUTPUT(y, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8,
                           DT_INT64, DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64,
                           DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32,
                           DT_BF16, DT_COMPLEX32, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN,
                           DT_FLOAT4_E1M2, DT_FLOAT4_E2M1}))
    .REQUIRED_ATTR(dst_type, Int)
    .OP_END_FACTORY_REG(Cast)

/**
* @brief Creates a tensor filled with a scalar value.
* This operation creates a tensor of shape "dims" and fills it with "value".
*
* @par Inputs:
* @li dims: A 1D tensor of types int32 or int64. Represents the shape of the output tensor .
        The size of each dimension must be less than or equal to 8. \n

* @li value: A 0D scalar. Specifies the value to fill the returned tensor.
*    Must be one of the following types:
*    bfloat16, float16, float32, double, int32, uint8, int16, int8, complex64, int64, bool,
*    qint8, quint8, qint32, qint16, quint16, uint16, complex128, uint32, uint64, string.
*
* @par Outputs:
* y: A tensor. Has the same type as "value".
*
* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator Fill.
* @li Compatible with the Caffe operator Filler.
*
*/
REG_OP(Fill)
    .INPUT(dims, TensorType::IndexNumberType())
    .INPUT(value, "T")
    .OUTPUT(y, "T")
    .DATATYPE(T, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16,
                              DT_INT8, DT_COMPLEX64, DT_INT64, DT_BOOL, DT_QINT8,
                              DT_QUINT8, DT_QINT32, DT_QINT16, DT_QUINT16, DT_UINT16,
                              DT_COMPLEX128, DT_FLOAT16, DT_BF16, DT_UINT32, DT_UINT64, DT_STRING}))
    .OP_END_FACTORY_REG(Fill)


}  // namespace ge

#endif  // MATH_COMMON_STUB_OPS_H
