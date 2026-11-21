#include "../../../grouped_matmul_situ_quant/op_host/op_api/grouped_matmul_situ_quant_packed_tensor.h"

#include "aclnn/aclnn_base.h"

extern "C" aclnnStatus aclnnGroupedMatmulSituQuantMlpGetWorkspaceSize(
    const aclTensor* x, const aclTensor* xScale, const aclTensor* groupList, const aclTensor* upWeight,
    const aclTensor* upScale, const aclTensor* upCompensation, const aclTensor* downWeight,
    const aclTensor* downScale, const aclTensor* downCompensation, int64_t groupListType, double beta,
    double linearBeta, aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor);

extern "C" aclnnStatus aclnnGroupedMatmulSituQuantMlp(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

extern "C" aclnnStatus aclnnGroupedMatmulSituQuantMlpV2GetWorkspaceSize(
    const aclTensor* x, const aclTensor* xScale, const aclTensor* groupList, const aclTensor* upWeight,
    const aclTensor* upScale, const aclTensor* upCompensation, const aclTensor* downWeight,
    const aclTensor* downScale, const aclTensor* downCompensation, int64_t groupListType, double beta,
    double linearBeta, aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    situ_quant::UnpackInt32ToInt4(upWeight);
    situ_quant::UnpackInt32ToInt4(downWeight);
    situ_quant::CastInt64ScaleToUint64(upScale);
    situ_quant::CastInt64ScaleToUint64(downScale);
    return aclnnGroupedMatmulSituQuantMlpGetWorkspaceSize(
        x, xScale, groupList, upWeight, upScale, upCompensation, downWeight, downScale, downCompensation,
        groupListType, beta, linearBeta, y, workspaceSize, executor);
}

extern "C" aclnnStatus aclnnGroupedMatmulSituQuantMlpV2(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    return aclnnGroupedMatmulSituQuantMlp(workspace, workspaceSize, executor, stream);
}
