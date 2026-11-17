/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0.
 */
#include <cstring>
#include "aclnn_moe_token_cut.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include <cstring>
#include "aclnn/aclnn_base.h"

namespace {
constexpr size_t HCCL_GROUP_NAME_MAX = 128;

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};
}  // namespace

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(
    void *executor, NnopbaseHcclServerType serverType);

extern "C" aclnnStatus aclnnInnerMoeTokenCutGetWorkspaceSize(
    const aclTensor *topkIds, const aclTensor *topkWeightsOptional,
    const aclTensor *hidden,
    const aclTensor *cutIdx, const aclTensor *newTopk, const aclTensor *newTopkWeight,
    const aclTensor *newHidden, const aclTensor *newResiduals,
    const aclTensor *actualBs, const aclTensor *prefixIndex, const aclTensor *noCutRankNum,
    const aclTensor *activeMask, const aclTensor *localCutCount, const aclTensor *batchSize,
    const aclTensor *enableCut,
    const aclTensor *residuals, const aclTensor *discardLabel, const aclTensor *discardTokens,
    const aclTensor *discardMoeReqId, const aclTensor *reqIdList,
    const aclTensor *discardResiduals, const aclTensor *discardTopk,
    const aclTensor *discardTopkWeight, const aclTensor *moeCnt,
    const aclTensor *historyBs, const aclTensor *originalBs,
    const char *groupEp, int64_t topK, int64_t rankId, int64_t layerIdx,
    int64_t rankNum,
    int64_t expertNum, int64_t expertPerRank, int64_t levelLimit,
    int64_t cutCountThreshold, int64_t smallExpertTokenThreshold,
    int64_t sendAivNum, int64_t recvAivNum,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern "C" aclnnStatus aclnnInnerMoeTokenCut(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

extern "C" aclnnStatus aclnnMoeTokenCutGetWorkspaceSize(
    const aclTensor *topkIds,
    const aclTensor *topkWeightsOptional,
    const aclTensor *hidden,
    aclTensor *cutIdx,
    aclTensor *newTopk,
    aclTensor *newTopkWeight,
    aclTensor *newHidden,
    aclTensor *newResiduals,
    aclTensor *actualBs,
    aclTensor *prefixIndex,
    aclTensor *noCutRankNum,
    aclTensor *activeMask,
    const aclTensor *localCutCount,
    const aclTensor *batchSize,
    const aclTensor *enableCut,
    const aclTensor *residuals,
    aclTensor *discardLabel,
    const aclTensor *discardTokens,
    const aclTensor *discardMoeReqId,
    aclTensor *reqIdList,
    const aclTensor *discardResiduals,
    const aclTensor *discardTopk,
    const aclTensor *discardTopkWeight,
    aclTensor *moeCnt,
    aclTensor *historyBs,
    const aclTensor *originalBs,
    const char *groupEp,
    int64_t topK,
    int64_t rankId,
    int64_t layerIdx,
    int64_t rankNum,
    int64_t expertNum,
    int64_t expertPerRank,
    int64_t levelLimit,
    int64_t cutCountThreshold,
    int64_t smallExpertTokenThreshold,
    int64_t sendAivNum,
    int64_t recvAivNum,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    // ACLNN 入口只做低成本空指针检查；shape、dtype、rank 拓扑以及
    // workspace/window 容量由 host tiling 统一校验。
    if (topkIds == nullptr || cutIdx == nullptr || newTopk == nullptr ||
        newTopkWeight == nullptr || actualBs == nullptr || prefixIndex == nullptr ||
        noCutRankNum == nullptr || activeMask == nullptr ||
        localCutCount == nullptr ||
        batchSize == nullptr || enableCut == nullptr ||
        workspaceSize == nullptr || executor == nullptr ||
        groupEp == nullptr || strnlen(groupEp, HCCL_GROUP_NAME_MAX) == 0 ||
        strnlen(groupEp, HCCL_GROUP_NAME_MAX) == HCCL_GROUP_NAME_MAX) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    return aclnnInnerMoeTokenCutGetWorkspaceSize(
        topkIds, topkWeightsOptional, hidden, cutIdx, newTopk, newTopkWeight,
        newHidden, newResiduals, actualBs,
        prefixIndex, noCutRankNum, activeMask, localCutCount, batchSize,
        enableCut, residuals, discardLabel, discardTokens, discardMoeReqId,
        reqIdList, discardResiduals, discardTopk, discardTopkWeight, moeCnt,
        historyBs, originalBs, groupEp, topK, rankId, layerIdx,
        rankNum, expertNum, expertPerRank, levelLimit, cutCountThreshold,
        smallExpertTokenThreshold, sendAivNum, recvAivNum,
        workspaceSize, executor);
}

extern "C" aclnnStatus aclnnMoeTokenCut(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    const bool setterAvailable = NnopbaseSetHcclServerType != nullptr;
    // std::fprintf(stderr, "[MTC_HOST_V1] setter_available=%d\n",
        // static_cast<int>(setterAvailable));
    // std::fflush(stderr);

    // 保留原逻辑：仅在弱符号存在时设置 MTE server。
    if (setterAvailable) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
        // std::fprintf(stderr, "[MTC_HOST_V1] MTE setter called\n");
        // std::fflush(stderr);
    }

    // std::fprintf(stderr, "[MTC_HOST_V1] inner before\n");
    // std::fflush(stderr);
    const aclnnStatus status =
        aclnnInnerMoeTokenCut(workspace, workspaceSize, executor, stream);
    // 此处只表示 Host 执行接口返回，不表示设备任务已完成；不额外插入同步。
    // std::fprintf(stderr, "[MTC_HOST_V1] inner returned, status=%d\n",
        // static_cast<int>(status));
    // std::fflush(stderr);
    return status;
}
