/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_swiglu_quant_weight_nz_tensor_list.cpp
 * \brief
 */
#include "grouped_matmul_swiglu_quant_weight_nz_tensor_list.h"
#include <typeinfo>
#include "grouped_matmul_swiglu_quant_weight_nz_tensor_list_split_ws.h"
using namespace AscendC;
using namespace matmul;
using namespace GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_TENSOR_LIST;
using MM_DTYPE_Y = int32_t;

template <bool trans = false>
using xType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X>;

template <bool trans = false>
using weightType = MatmulType<AscendC::TPosition::GM, CubeFormat::NZ, DTYPE_WEIGHT>;

using yType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, MM_DTYPE_Y>;

#define GMM_CV_SPLIT_IMP(computeClass, dtypeC, transA, transB, sync, cfg, aType, bType, cType)                     \
    do {                                                                                                           \
        using matmulType = MMImplType<aType<transA>, bType<transB>, cType, cType, cfg>;                            \
        matmulType::MT mm;                                                                                         \
        GET_TILING_DATA_MEMBER(GMMSwigluQuantTilingData, gmmSwigluBaseParams, gmmSwigluBaseParams_, tiling);       \
        GET_TILING_DATA_MEMBER(GMMSwigluQuantTilingData, mmTilingData, mmTilingData_, tiling);                     \
        GET_TILING_DATA_MEMBER(GMMSwigluQuantTilingData, gmmSwiglu, gmmSwiglu_, tiling);                           \
        if ASCEND_IS_AIC {                                                                                         \
        mm.SetSubBlockIdx(0);                                                                                      \
        mm.Init(&mmTilingData_, &tPipe);                                                                           \
        }                                                                                                          \
        computeClass<matmulType, sync, dtypeC> computeOp(mm);                                                      \
        computeOp.Init(x, weight, perChannelScale, perTokenScale, groupList, quantOutput, quantScaleOutput,        \
                       user1, &gmmSwigluBaseParams_, &mmTilingData_, &gmmSwiglu_, &tPipe);                         \
        computeOp.Process();                                                                                       \
    } while (0)

extern "C" __global__ __aicore__ void grouped_matmul_swiglu_quant_weight_nz_tensor_list(GM_ADDR x, GM_ADDR weight, GM_ADDR perChannelScale, GM_ADDR perTokenScale, 
                                                                  GM_ADDR groupList, GM_ADDR quantOutput, GM_ADDR quantScaleOutput, 
                                                                  GM_ADDR workspace, GM_ADDR tiling) {
    TPipe tPipe;
    AscendCUtils::SetOverflow(1);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GM_ADDR user1 = GetUserWorkspace(workspace);
    if (TILING_KEY_IS(0)) {  // antiquant msd
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
        GMM_CV_SPLIT_IMP(
            GMMSwigluCompute, // computeClass
            DTYPE_WEIGHT_SCALE,
            false,              // transA
            false,              // transB
            false,              // sync
            NZ_CFG_MDL,         // cfg
            xType,              // aType
            weightType,         // bType
            yType);             // cType
    } else if(TILING_KEY_IS(1)){
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        GMM_CV_SPLIT_IMP(
            GMMSwigluSplitWorkSpaceCompute, // computeClass
            DTYPE_WEIGHT_SCALE,
            false,              // transA
            false,              // transB
            false,              // sync
            NZ_CFG_MDL,         // cfg
            xType,              // aType
            weightType,         // bType
            yType);             // cType
    }
}
