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
 * \file grouped_matmul_situ_quant.cpp
 * \brief GroupedMatmulSituQuant arch35 (Ascend950) 融合算子入口。
 *
 * 单 launch 完成 GroupedMatmul(MXFP8 x MXFP4) -> SiTU(gate/up) -> 动态 MX 量化，
 * 替代拆链 npu_grouped_matmul + situ_mx_quant。
 *
 * 实现策略：本算子自带一份 MxA8W4 数据通路（op_kernel/weight_quant_basic_block/，
 * 由 arch35 的 GMMSQWeightQuant 实现复制而来，与本算子同目录编译，不依赖其它算子），
 * 唯一差异是激活策略模板实参为 GmmVecActivationSitu（见 situ_activation.h）：
 *   - 调度、L1/L0 搬运、per-32 max-exp 归约、e8m0 scale 与 fp8 量化全部与 MxA8W4 通路一致；
 *   - 激活在 UB 内以寄存器 VF 完成，不引入额外 GM 往返。
 *
 * 入参顺序与本仓库算子原型保持一致（weight/weight_scale 在 group_list 之前），
 * 与 ops-transformer 侧的入口仅差 group_list 的位置；arch35 不读取
 * weight_assist_matrix / bias / smooth_scale 三个槽位（它们是 A2/A3 的 A8W4 MSD 通路需要的）。
 */
#include "kernel_tiling/kernel_tiling.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"

#include "weight_quant_basic_block/grouped_matmul_swiglu_quant_v2_weight_quant_tiling_data.h"
#include "weight_quant_basic_block/gmmsq_weight_quant_resplit_controller.h"
#include "grouped_matmul_situ_quant_tiling_key.h"
#include "situ_activation.h"

#define FLOAT_OVERFLOW_MODE_CTRL 60

using namespace AscendC;

template <int8_t QUANT_B_TRANS, int8_t QUANT_A_TRANS, bool IS_SINGLE_MULTI_SINGLE>
__global__ __aicore__ void grouped_matmul_situ_quant(
    GM_ADDR x, GM_ADDR xScale, GM_ADDR weight, GM_ADDR weightScale, GM_ADDR groupList,
    GM_ADDR weightAssistMatrix, GM_ADDR bias, GM_ADDR smoothScale, GM_ADDR y, GM_ADDR yScale,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)weightAssistMatrix;
    (void)bias;
    (void)smoothScale;
    AscendC::InitSocState();
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    // enable overflow mode to avoid nan/inf value
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(GMMSQArch35Tiling::GMMSQWeightQuantTilingData);
    GET_TILING_DATA_WITH_STRUCT(GMMSQArch35Tiling::GMMSQWeightQuantTilingData, tilingDataIn, tiling);

    GMMSQWeightQuant::GMMSQWeightQuantResplitController<
        DTYPE_X, DTYPE_WEIGHT, DTYPE_WEIGHT_SCALE, DTYPE_X_SCALE, DTYPE_Y, DTYPE_Y_SCALE, GMMSQWeightQuant::MXA8W4_NZNK,
        GMMSQWeightQuant::VEC_ANTIQUANT_CONFIG_DYNAMIC, IS_SINGLE_MULTI_SINGLE, GMMSQWeightQuant::GmmVecActivationSitu>
        controller;

    controller.Init(x, weight, weightScale, groupList, xScale, y, yScale, &tilingDataIn);
    controller.Process();
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
}
