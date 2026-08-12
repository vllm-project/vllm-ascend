/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MATMUL_COMMON_OP_API_CUBE_UTIL_H_
#define MATMUL_COMMON_OP_API_CUBE_UTIL_H_

#include "aclnn/aclnn_base.h"
#include "op_api/op_api_def.h"
#include "opdev/common_types.h"
#include "opdev/platform.h"

namespace Ops {
namespace Transformer {
// 校验针对cube tensor的dtype，cubeMathType的值是否符合预期
bool CheckCubeMathType(const op::DataType cubeTensorDtype, int8_t cubeMathType);

// 校验针对mm算子 tensor的dtype，cubeMathType的值是否符合预期
bool CheckCubeMathTypeForMm(const op::DataType cubeTensorDtype, int8_t cubeMathType);

// 返回芯片对应支持的数据类型
const std::initializer_list<op::DataType> &GetDtypeSupportListBySocVersion();
const std::initializer_list<op::DataType> &GetDtypeSupportListBySocVersion4ConvBackward(bool transposed);

// 根据promote type + cubemathtype的组合算出最终算子应用的dtype
op::DataType CalcPromoteTypeCubemathtype(const op::DataType cubeTensorPromoteType, int8_t cubeMathType);
op::DataType CalcUseFp16PromoteType(const op::DataType cubeTensorPromoteType);
op::DataType CalcUseHf32PromoteType(const op::DataType cubeTensorPromoteType);
op::DataType CalcAllowFp32DownPrecisionPromoteType(const op::DataType cubeTensorPromoteType);
op::DataType CalcKeepDtypePromoteType(const op::DataType cubeTensorPromoteType);
op::DataType CalcPromoteTypeCubeMathTypeNew(const op::DataType cubeTensorPromoteType, int8_t cubeMathType);

// 根据promoteType + cubeMathType 判断是否要走HF32分支
bool NeedCubeGoHF32(const op::DataType cubeTensorPromoteType, int8_t cubeMathType);

// 检查针对x芯片，cube算子是否支持FP32
inline bool IsCubeSupportFp32()
{
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_2201 &&
        op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        return false;
    }
    return true;
}

// 检查针对x芯片，cube算子是否支持HF32
inline bool IsCubeSupportHf32()
{
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_2201 &&
        op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        return false;
    }
    return true;
}

bool CheckUnSupportDtype(const aclTensor *input, const aclTensor *weight);

} // namespace Transformer
} // namespace Ops

#endif
