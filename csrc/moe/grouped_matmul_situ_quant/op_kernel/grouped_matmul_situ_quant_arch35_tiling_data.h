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
 * \file grouped_matmul_situ_quant_arch35_tiling_data.h
 * \brief 本算子自带的 MxA8W4 tiling 结构（已复制到 op_kernel/weight_quant_basic_block/）。
 *
 * 调度契约（groupNum/coreNum/groupSize/kSize/nSize）与 MxA8W4 数据通路同源，
 * 额外承载 SiTU 的 beta/linearBeta，因此不引入独立结构，避免两份定义漂移。
 */
#ifndef GROUPED_MATMUL_SITU_QUANT_ARCH35_TILING_DATA_H
#define GROUPED_MATMUL_SITU_QUANT_ARCH35_TILING_DATA_H

#include "weight_quant_basic_block/grouped_matmul_swiglu_quant_v2_weight_quant_tiling_data.h"

namespace gmm_situ {
using SituTilingData = GMMSQArch35Tiling::GMMSQWeightQuantTilingData;
} // namespace gmm_situ

#endif // GROUPED_MATMUL_SITU_QUANT_ARCH35_TILING_DATA_H
