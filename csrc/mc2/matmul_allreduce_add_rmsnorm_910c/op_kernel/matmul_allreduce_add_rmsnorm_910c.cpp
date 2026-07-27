#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "matmul_allreduce_add_rmsnorm_910c_aic_kernel.h"
#include "matmul_allreduce_add_rmsnorm_910c_aiv_kernel.h"
#include "matmul_allreduce_add_rmsnorm_910c_tiling.h"

extern "C" __global__ __aicore__ void matmul_allreduce_add_rmsnorm910c(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR residual, GM_ADDR gamma, GM_ADDR y,
    GM_ADDR add_out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MatmulAllreduceAddRmsnorm910cTilingData);
    GET_TILING_DATA(tiling_data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);

    if ASCEND_IS_AIC {
        MatmulAllreduceAddRmsnorm910cAicKernel<DTYPE_X1, DTYPE_Y> op;
        op.Init(x1, x2, y, workspace, &tiling_data);
        op.Process();
        return;
    }

    if ASCEND_IS_AIV {
        auto context = AscendC::GetHcclContext<AscendC::HCCL_GROUP_ID_0>();
        auto windowContext = reinterpret_cast<__gm__ HcclOpResParam *>(context);
        MatmulAllreduceAddRmsnorm910cAivKernel<DTYPE_X1> op;
        op.Init(y, residual, gamma, y, add_out, workspace, &tiling_data, windowContext);
        op.Process();
    }
}
