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
 * \brief GroupedMatmul(A8W4 MSD) + dequant + SiTU-GLU + per-token quant 的 kernel 入口。
 *
 * 只实现 Atlas A2 / A3 的 A8W4 MSD 路径（x=INT8, weight=INT4），tiling key = 2。
 */

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "adv_api/matmul/constant_tiling.h"
#include "arch32/grouped_matmul_situ_quant_a8w4_msd_pipeline.h"
#include "arch32/grouped_matmul_situ_quant_utils_kernel.h"

using namespace AscendC;
using namespace matmul;
using namespace GroupedMatmulDequantSituQuant;

#if defined(GMM_SITU_QUANT_A8W4_MSD)
// The report shape uses 18 full 384-column tiles plus a 256-column tail.
constexpr uint32_t STATIC_EXPERT_ROWS = 16;
constexpr uint32_t STATIC_EXPERT_K = 7168;
constexpr uint32_t KIMI_ROUTED_EXPERTS = 7;
constexpr uint32_t STATIC_EXPERT_FULL_TILE_COLUMNS = 384;
constexpr uint32_t STATIC_EXPERT_TAIL_COLUMNS = 256;
constexpr uint32_t STATIC_EXPERT_TAIL_CORE = 18;

template <int32_t COLS>
struct SingleExpertStaticMatmul {
    using AT = MatmulType<TPosition::GM, CubeFormat::ND, int4b_t, false>;
    using BT = MatmulType<TPosition::GM, CubeFormat::NZ, int4b_t, false>;
    using CT = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
    using BiasT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    template <int32_t N>
    using Dim = AscendC::Std::integral_constant<int32_t, N>;

    __aicore__ static constexpr MatmulApiStaticTiling MakeTiling()
    {
        constexpr int32_t BASE_M = 64;
        constexpr int32_t BASE_K = 128;
        constexpr int32_t FULL_K = STATIC_EXPERT_K;
        constexpr int32_t PANEL_K = 1024;

        auto config = CUSTOM_CFG_MDL;
        config.basicM = BASE_M;
        config.basicN = COLS;
        config.basicK = BASE_K;
        constexpr int32_t PACKED_ROWS = DOUBLE_ROW * STATIC_EXPERT_ROWS;
        config.singleCoreM = PACKED_ROWS;
        config.singleCoreN = COLS;
        config.singleCoreK = FULL_K;
        config.enableSetBias = false;
        using Single = AscendC::Std::tuple<Dim<PACKED_ROWS>, Dim<COLS>, Dim<FULL_K>>;
        using L1 = AscendC::Std::tuple<Dim<BASE_M>, Dim<COLS>, Dim<PANEL_K>>;
        using Base = AscendC::Std::tuple<Dim<BASE_M>, Dim<COLS>, Dim<BASE_K>>;
        return AscendC::GetMatmulApiTiling<AT, BT, CT, BiasT, Single, L1, Base>(config);
    }
    static constexpr auto config = MakeTiling();
    using MT = matmul::MatmulImpl<AT, BT, CT, BiasT, config>;
};

template <int32_t COLS>
__aicore__ inline void RunStaticSingleExpert(
    GM_ADDR x, GM_ADDR xScale, GM_ADDR weight, GM_ADDR weightScale, GM_ADDR groupList,
    GM_ADDR weightAssistanceMatrix, GM_ADDR y, GM_ADDR yScale, GM_ADDR workspace, TPipe& pipe,
    GMMSituQuantBaseParams params, const GMMSituQuant& situ, const TCubeTiling& actualTiling, uint32_t expert)
{
    using Matmul = SingleExpertStaticMatmul<COLS>;
    typename Matmul::MT mm;
    GMMSituQuantPipelineSchedule<Matmul> op(mm, &params, &situ, &pipe);
    op.Init(x, weight, weightScale, xScale, weightAssistanceMatrix, groupList,
            y, yScale, workspace, STATIC_EXPERT_ROWS);
    params.baseM = actualTiling.baseM;
    params.baseN = actualTiling.baseN;
    mm.SetSubBlockIdx(0);
    mm.Init(&actualTiling);
    op.ProcessSmallSingleExpert(expert);
}
#endif

extern "C" __global__ __aicore__ void grouped_matmul_situ_quant_msd(GM_ADDR x, GM_ADDR xScale, GM_ADDR weight,
                                                                GM_ADDR weightScale, GM_ADDR groupList,
                                                                GM_ADDR weightAssistanceMatrix, GM_ADDR bias,
                                                                GM_ADDR smoothScale, GM_ADDR y, GM_ADDR yScale,
                                                                GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe tPipe;
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);

#if defined(GMM_SITU_QUANT_A8W4_MSD)
    if (TILING_KEY_IS(2) || TILING_KEY_IS(3) || TILING_KEY_IS(4)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
        KERNEL_TASK_TYPE(3, KERNEL_TYPE_MIX_AIC_1_2);
        KERNEL_TASK_TYPE(4, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_MEMBER(GMMSituQuantTilingData, gmmSituQuantBaseParams, gmmSituQuantBaseParams_, tiling);
        GET_TILING_DATA_MEMBER(GMMSituQuantTilingData, mmTilingData, mmTilingData_, tiling);
        GET_TILING_DATA_MEMBER(GMMSituQuantTilingData, mmTilingDataAlt, mmTilingDataAlt_, tiling);
        GET_TILING_DATA_MEMBER(GMMSituQuantTilingData, mmTilingDataWide, mmTilingDataWide_, tiling);
        GET_TILING_DATA_MEMBER(GMMSituQuantTilingData, gmmSituQuant, gmmSituQuant_, tiling);
        using xType = MatmulType<TPosition::GM, CubeFormat::ND, int4b_t, false>;
        using weightType = MatmulType<TPosition::GM, wFormat, int4b_t, false>;
        using yType = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
        using matmulType = MMImplTypeCustom<xType, weightType, yType>;
        matmulType::MT mm;
        GMMSituQuantBaseParams baseParams = gmmSituQuantBaseParams_;
        GMMSituQuantPipelineSchedule<matmulType> op(mm, &baseParams, &gmmSituQuant_, &tPipe);
        op.Init(x, weight, weightScale, xScale, weightAssistanceMatrix, groupList, y, yScale, userWorkspace);
        if (TILING_KEY_IS(4)) {
            if (op.GetLiveRows() == 1) {
                if ASCEND_IS_AIC {
                    GET_TILING_DATA_MEMBER(GMMSituQuantTilingData, mmTilingDataSingleToken, singleTokenTiling, tiling);
                    baseParams.baseM = singleTokenTiling.baseM;
                    baseParams.baseN = singleTokenTiling.baseN;
                    mm.SetSubBlockIdx(0);
                    mm.Init(&singleTokenTiling);
                }
                op.ProcessSingleToken();
                return;
            }
            const int32_t expert = op.FindSmallSingleExpert();
            if (expert >= 0) {
                if ASCEND_IS_AIC {
                    GET_TILING_DATA_MEMBER(GMMSituQuantTilingData, mmTilingDataSingleToken, smallRowsTiling, tiling);
                    if (baseParams.K == STATIC_EXPERT_K && op.GetLiveRows() == STATIC_EXPERT_ROWS &&
                        GetBlockNum() > STATIC_EXPERT_TAIL_CORE) {
                        if (GetBlockIdx() == STATIC_EXPERT_TAIL_CORE) {
                            RunStaticSingleExpert<STATIC_EXPERT_TAIL_COLUMNS>(x, xScale, weight, weightScale, groupList,
                                weightAssistanceMatrix, y, yScale, userWorkspace, tPipe, baseParams,
                                gmmSituQuant_, smallRowsTiling, expert);
                        } else {
                            RunStaticSingleExpert<STATIC_EXPERT_FULL_TILE_COLUMNS>(x, xScale, weight, weightScale, groupList,
                                weightAssistanceMatrix, y, yScale, userWorkspace, tPipe, baseParams,
                                gmmSituQuant_, smallRowsTiling, expert);
                        }
                        return;
                    }
                    baseParams.baseM = smallRowsTiling.baseM;
                    baseParams.baseN = smallRowsTiling.baseN;
                    mm.SetSubBlockIdx(0);
                    mm.Init(&smallRowsTiling);
                }
                op.ProcessSmallSingleExpert(expert);
                return;
            }
        }
        if ASCEND_IS_AIC {
            const auto* cubeTiling = &mmTilingData_;
            if constexpr (wFormat == CubeFormat::NZ) {
                const int64_t altLimit = baseParams.altRowsPerExpertLimit;
                const int64_t groupNum = gmmSituQuant_.groupListLen;
                if (altLimit > 0 && groupNum > 0 && op.GetLiveRows() <= altLimit * groupNum) {
                    cubeTiling = &mmTilingDataAlt_;
                } else if (mmTilingDataWide_.baseM > 0 &&
                           op.FitsWideTile(mmTilingData_.baseM, mmTilingDataWide_.baseM)) {
                    cubeTiling = &mmTilingDataWide_;
                }
            }
            baseParams.baseM = cubeTiling->baseM;
            baseParams.baseN = cubeTiling->baseN;
            mm.SetSubBlockIdx(0);
            mm.Init(cubeTiling);
        }
        if (TILING_KEY_IS(3) || (TILING_KEY_IS(4) && baseParams.groupNum == KIMI_ROUTED_EXPERTS)) {
            op.ProcessDecode();
        } else {
            op.Process();
        }
    }
#endif
}
