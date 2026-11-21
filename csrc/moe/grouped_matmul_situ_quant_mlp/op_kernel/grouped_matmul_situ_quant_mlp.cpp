#define GMM_SITU_QUANT_A8W4_MSD
#define GMM_SITU_QUANT_MLP

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#if __has_include("../../grouped_matmul_situ_quant/op_kernel/arch32/grouped_matmul_situ_quant_a8w4_msd_pipeline.h")
#include "../../grouped_matmul_situ_quant/op_kernel/arch32/grouped_matmul_situ_quant_a8w4_msd_pipeline.h"
#else
#include "../grouped_matmul_situ_quant/arch32/grouped_matmul_situ_quant_a8w4_msd_pipeline.h"
#endif

using namespace AscendC;
using namespace matmul;
using namespace GroupedMatmulDequantSituQuant;

extern "C" __global__ __aicore__ void grouped_matmul_situ_quant_mlp(
    GM_ADDR x, GM_ADDR xScale, GM_ADDR groupList, GM_ADDR upWeight, GM_ADDR upScale, GM_ADDR upCompensation,
    GM_ADDR downWeight, GM_ADDR downScale, GM_ADDR downCompensation, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (!TILING_KEY_IS(0) && !TILING_KEY_IS(1)) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    auto upParams = tilingData.upParams;
    auto downParams = tilingData.downParams;
    auto downVector = tilingData.downVector;

    TPipe pipe;
    using In = MatmulType<TPosition::GM, CubeFormat::ND, int4b_t, false>;
    using Weight = MatmulType<TPosition::GM, CubeFormat::NZ, int4b_t, false>;
    using Partial = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
    using Matmul = MMImplTypeCustom<In, Weight, Partial>;
    Matmul::MT mm;

    GM_ADDR scratch = GetUserWorkspace(workspace);
    GM_ADDR packed = scratch + upParams.workSpaceOffset1 + upParams.workSpaceOffset2;
    GM_ADDR activationScale = packed + upParams.M * downParams.K;
    GMMSituQuantPipelineSchedule<Matmul> up(mm, &upParams, &tilingData.upVector, &pipe);
    up.Init(x, upWeight, upScale, xScale, upCompensation, groupList, packed, activationScale, scratch);
    const int64_t rows = up.GetLiveRows();
    if (rows == 0) {
        return;
    }

    constexpr int64_t KIMI_ROUTED_EXPERTS = 7;
    constexpr int64_t DECODE_ROWS_PER_EXPERT = 32;
    constexpr int64_t PACKED_DECODE_ROWS = 16;
    constexpr int64_t PREFILL_ROW_THRESHOLD = 2048;
    const bool singleToken = rows == 1;
    const bool routed = upParams.groupNum == KIMI_ROUTED_EXPERTS;
    const bool decode = rows <= DECODE_ROWS_PER_EXPERT * upParams.groupNum;
    int32_t smallExpert = -1;
    if (TILING_KEY_IS(1) && rows > 1 && rows <= PACKED_DECODE_ROWS) {
        smallExpert = up.FindSmallSingleExpert();
    }
    if ASCEND_IS_AIC {
        const auto* tile = &tilingData.upDefault;
        if (singleToken || smallExpert >= 0) {
            tile = &tilingData.upSingleToken;
        } else if (decode) {
            tile = &tilingData.upDecode;
        }
        upParams.baseM = tile->baseM;
        upParams.baseN = tile->baseN;
        mm.SetSubBlockIdx(0);
        mm.Init(tile);
    }
    if (smallExpert >= 0) {
        up.ProcessSmallSingleExpert(smallExpert, true);
    } else if (singleToken) {
        up.ProcessSingleToken(true);
    } else {
        up.ProcessDecode(true);
    }

    GMMSituQuantPipelineSchedule<Matmul> down(mm, &downParams, &downVector, &pipe);
    down.Init(packed, downWeight, downScale, activationScale, downCompensation, groupList, y, nullptr, scratch, rows);
    if ASCEND_IS_AIC {
        mm.End();
        const auto* downTile = &tilingData.downCube;
        if (singleToken && routed) {
            downTile = &tilingData.downSingleToken;
        } else if (routed && rows > PREFILL_ROW_THRESHOLD) {
            downTile = &tilingData.downPrefill;
        }
        downParams.baseM = downTile->baseM;
        downParams.baseN = downTile->baseN;
        mm.SetSubBlockIdx(0);
        mm.Init(downTile);
    }
    // The packed activation and scales must be visible before Cube reads them.
    SyncAll<false>();
    down.ProcessPrepackedDown();
}
