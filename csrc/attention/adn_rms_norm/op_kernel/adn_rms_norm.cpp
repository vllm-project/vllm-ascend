#include "adn_rms_norm_multi_row.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void adn_rms_norm(
    GM_ADDR x,
    GM_ADDR gamma,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(100)) {
        AdnRmsNorm::MultiRowKernel<128, 32> kernel;
        kernel.Init(x, gamma, y, &tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(200)) {
        AdnRmsNorm::MultiRowKernel<256, 16> kernel;
        kernel.Init(x, gamma, y, &tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(300)) {
        AdnRmsNorm::MultiRowKernel<2048, 2> kernel;
        kernel.Init(x, gamma, y, &tilingData);
        kernel.Process();
    }
}
