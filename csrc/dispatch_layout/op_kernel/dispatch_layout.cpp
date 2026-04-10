#include "kernel_operator.h"
#include "dispatch_layout.h"
#include "dispatch_layout_a2.h"
#include "dispatch_layout_tiling.h"

#define TILING_KEY_INT 23
#define TILING_KEY_A2_INT 123

extern "C" __global__ __aicore__ void dispatch_layout(GM_ADDR topkIdx, GM_ADDR numTokensPerRank,
                                                      GM_ADDR numTokensPerExpert, GM_ADDR isTokenInRank,
                                                      GM_ADDR notifySendData, GM_ADDR sendTokenIdxSmall,
                                                      GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(DispatchLayoutTilingData);
    GET_TILING_DATA_WITH_STRUCT(DispatchLayoutTilingData, tilingData, tiling);

    TPipe pipe;

    if (TILING_KEY_IS(TILING_KEY_INT)) {
        MoeDispatchLayout::DispatchLayout<int32_t> op;
        op.Init(topkIdx, numTokensPerRank, numTokensPerExpert, isTokenInRank, notifySendData, sendTokenIdxSmall,
                workspace, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_A2_INT)) {
        MoeDispatchLayoutA2::DispatchLayoutA2<int32_t> op;
        op.Init(topkIdx, numTokensPerRank, numTokensPerExpert, isTokenInRank, notifySendData, sendTokenIdxSmall,
                workspace, &pipe, &tilingData);
        op.Process();
    }
}
