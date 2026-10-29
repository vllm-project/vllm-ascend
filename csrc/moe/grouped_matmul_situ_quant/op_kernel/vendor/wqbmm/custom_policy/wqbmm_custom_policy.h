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
 * \file wqbmm_custom_policy.h
 * \brief
 */
#ifndef WQBMM_CUSTOM_POLICY_H
#define WQBMM_CUSTOM_POLICY_H

#include "lib/matmul_intf.h"
#include "wqbmm_copy_cube_out.h"

namespace WeightQuantBatchMatmulV2::Arch35 {
template <const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
struct WQBmmCustomPolicy : public AscendC::Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
public:
    using CopyCubeOut = WQBmmCustomCopyCubeOut<IMPL, A_TYPE, B_TYPE, C_TYPE, MM_CFG>;
};
}  // namespace WeightQuantBatchMatmulV2::Arch35
#endif  // WQBMM_CUSTOM_POLICY_H