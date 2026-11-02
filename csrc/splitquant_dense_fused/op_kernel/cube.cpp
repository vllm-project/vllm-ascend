#include "kernel_operator.h"
#include "cube_adapter.h"

extern "C" __global__ __aicore__ void splitquant_grouped_cube(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t m, uint32_t n, uint32_t k,
    GM_ADDR workspace, GM_ADDR tiling)
{
    SplitQuantDenseA2::GroupedKernel op;
    op.Run(a, b, c, tiling, m, n, k / 128);
}

