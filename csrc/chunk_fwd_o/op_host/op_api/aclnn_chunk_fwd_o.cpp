#include <string.h>
#include "graph/types.h"
#include "aclnn_chunk_fwd_o.h"
#include <iostream>

using namespace std;

#ifdef __cplusplus
extern "C" {
#endif
extern aclnnStatus aclnnInnerChunkFwdO(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);
extern aclnnStatus aclnnInnerChunkFwdOGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *h,
    const aclTensor *g, const aclTensor *cuSeqlensOptional, const aclTensor *chunkOffsetsOptional, double scale, int64_t chunkSize,
    int64_t kStride0, int64_t vStride0, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnChunkFwdOGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *h,
    const aclTensor *g,
    const aclTensor *cuSeqlensOptional,
    const aclTensor *chunkOffsetsOptional,
    double scale,
    int64_t chunkSize,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    int64_t *keyStridesValue = nullptr;
    uint64_t keyStridesNum = 0;

    int64_t *valueStridesValue = nullptr;
    uint64_t valueStridesNum = 0;

    // aclGetViewStrides 获取aclTensor 对应的stride和stride个数
    aclGetViewStrides(k, &keyStridesValue, &keyStridesNum);
    aclGetViewStrides(v, &valueStridesValue, &valueStridesNum);
    
    std::vector<int64_t> sizeData = {0, 0};

    // k和v地址是否连续判断是否需要增加？
    sizeData = {keyStridesValue[0], valueStridesValue[0]};

    aclIntArray *kvStridesOptional = aclCreateIntArray(sizeData.data(), sizeData.size());
    int64_t kStride0 = keyStridesValue[0];
    int64_t vStride0 = valueStridesValue[0];

    aclnnStatus ret = aclnnInnerChunkFwdOGetWorkspaceSize(q, k, v, h, g, cuSeqlensOptional, chunkOffsetsOptional,
        scale, chunkSize, kStride0, vStride0, out, workspaceSize, executor);

    aclDestroyIntArray(kvStridesOptional);

    delete[] keyStridesValue;
    delete[] valueStridesValue;

    return ret;
}

aclnnStatus aclnnChunkFwdO(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    return aclnnInnerChunkFwdO(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
