#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "custom_muls_tiling_key.h"
#include "custom_muls.h"

extern "C" __global__ __aicore__ void custom_muls(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(CustomMulsTilingData);
    GET_TILING_DATA_WITH_STRUCT(CustomMulsTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (TILING_KEY_IS(CUSTOM_MULS_KEY_BF16_PROMOTE_FP32)) {
        NsCustomMuls::CustomMuls<bfloat16_t, true> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(CUSTOM_MULS_KEY_FP16_PROMOTE_FP32)) {
        NsCustomMuls::CustomMuls<half, true> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(CUSTOM_MULS_KEY_FP32_DIRECT)) {
        NsCustomMuls::CustomMuls<float, false> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
