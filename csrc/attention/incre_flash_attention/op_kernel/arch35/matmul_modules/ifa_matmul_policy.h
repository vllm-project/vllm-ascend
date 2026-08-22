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
 * \file ifa_matmul_policy.h
 * \brief
 */
#ifndef IFA_MATMUL_POLICY_H
#define IFA_MATMUL_POLICY_H

#include "ifa_flag_data.h"
#include "ifa_cube_in_buffer.h"
#include "ifa_copy_cube_in.h"

namespace AscendC {
namespace Impl {
namespace Detail {
template <typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE, const auto& MM_CFG, typename MM_CB>
class IFAMatmulPolicyNormal : public MatmulPolicy<MM_CFG,IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
 public:
  using UserDefDataType = IFAFlagData;
  using CubeInBufferA = AscendC::Impl::Detail::IFACubeInBuffer<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
  using CopyCubeInA = AscendC::Impl::Detail::IFACopyCubeIn<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
};

} // namespace Detail
} // namespace Impl
} // namespace AscendC
#endif  // IFA_MATMUL_POLICY_H
