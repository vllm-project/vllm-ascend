#include <string.h>
#include "graph/types.h"
#include "aclnn/opdev/platform.h"
#include "aclnn_dispatch_gmm_combine_decode.h"
#include "aclnnInner_dispatch_gmm_combine_decode.h"

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnDispatchGmmCombineDecodeGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *expertIds,
    const aclTensor *gmm1PermutedWeight,
    const aclTensor *gmm1PermutedWeightScale,
    const aclTensor *gmm2Weight,
    const aclTensor *gmm2WeightScale,
    const aclTensor *expertSmoothScalesOptional,
    const aclTensor *expertScalesOptional,
    char *groupEp,
    int64_t epRankSize,
    int64_t epRankId,
    int64_t moeExpertNum,
    int64_t shareExpertNum,
    int64_t shareExpertRankNum,
    int64_t quantMode,
    int64_t globalBs,
    const aclTensor *output,
    const aclTensor *epRecvCount,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnInnerDispatchGmmCombineDecodeGetWorkspaceSize(x, expertIds, gmm1PermutedWeight, gmm1PermutedWeightScale,
        gmm2Weight, gmm2WeightScale, expertSmoothScalesOptional, expertScalesOptional, groupEp, epRankSize,
        epRankId, moeExpertNum, shareExpertNum, shareExpertRankNum, quantMode, globalBs,
        output, epRecvCount, workspaceSize, executor);
}

aclnnStatus aclnnDispatchGmmCombineDecode(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910B) {
            NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        } else {
            NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
        }
    }
    return aclnnInnerDispatchGmmCombineDecode(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif


