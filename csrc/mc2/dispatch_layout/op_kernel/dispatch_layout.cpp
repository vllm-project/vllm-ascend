#include "kernel_operator.h"
#include "dispatch_layout.h"
#include "dispatch_layout_tiling.h"


extern "C" __global__ __aicore__ void dispatch_layout(GM_ADDR topkIdx, GM_ADDR numTokensPerRank, GM_ADDR numTokensPerExpert, 
                                                      GM_ADDR isTokenInRank, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(DispatchLayoutTilingData);
    GET_TILING_DATA_WITH_STRUCT(DispatchLayoutTilingData, tilingData, tiling);

    TPipe pipe;

    DispatchLayout<int32_t> op;
    op.Init(topkIdx, numTokensPerRank, numTokensPerExpert, isTokenInRank, workspace, &pipe, &tilingData);
    op.Process();
}
