#include <string.h>
#include "graph/types.h"
#include "aclnn_chunk_gated_delta_rule_fwd_h.h"
#include <iostream>

using namespace std;

#ifdef __cplusplus
extern "C" {
#endif
extern aclnnStatus aclnnInnerChunkGatedDeltaRuleFwdH(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);
extern aclnnStatus aclnnInnerChunkGatedDeltaRuleFwdHGetWorkspaceSize(const aclTensor *k, const aclTensor *w, const aclTensor *u, const aclTensor *g,
    const aclTensor *initialStateOptional, const aclTensor *cuSeqlensOptional, const aclTensor *chunkIndicesOptional, 
    bool outputFinalState, int64_t chunkSize, int64_t kStride0, int64_t vStride0, const aclTensor *h, const aclTensor *vNew, const aclTensor *finalState, 
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *u,
    const aclTensor *g,
    const aclTensor *initialStateOptional,
    const aclTensor *cuSeqlensOptional,
    const aclTensor *chunkIndicesOptional,
    bool outputFinalState,
    int64_t chunkSize,
    const aclTensor *h,
    const aclTensor *vNew,
    const aclTensor *finalState,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    // 增加逻辑
    cout << "step1" << endl;
    int64_t *keyStridesValue = nullptr;
    uint64_t keyStridesNum = 0;

    int64_t *valueStridesValue = nullptr;
    uint64_t valueStridesNum = 0;

    // aclGetViewStrides 获取aclTensor 对应的stride和stride个数
    aclGetViewStrides(k, &keyStridesValue, &keyStridesNum);
    aclGetViewStrides(u, &valueStridesValue, &valueStridesNum);
    cout << "step2 " << endl;
    /*
    // 打印第一次调用结果
    cout << "===== 第一次调用 aclGetViewStrides(k) 结果 =====" << endl;
    cout << "keyStridesValue 指针地址 = " << keyStridesValue << endl;
    cout << "keyStridesNum 步值数量 = " << keyStridesNum << endl;

    // 打印步值数组的每一个元素（关键：数组有多少个就打多少个）
    if (keyStridesValue != nullptr && keyStridesNum > 0) {
        cout << "keyStridesValue 数组内容：";
        for (size_t i = 0; i < keyStridesNum; ++i) {
            cout << keyStridesValue[i] << " ";
        }
        cout << endl;
    }

    // 第二次调用
    aclGetViewStrides(v, &keyStridesValue, &keyStridesNum);
    // 打印第二次调用结果
    cout << "\n===== 第二次调用 aclGetViewStrides(v) 结果 =====" << endl;
    cout << "keyStridesValue 指针地址 = " << keyStridesValue << endl;
    cout << "keyStridesNum 步值数量 = " << keyStridesNum << endl;

    // 打印步值数组
    if (keyStridesValue != nullptr && keyStridesNum > 0) {
        cout << "keyStridesValue 数组内容：";
        for (size_t i = 0; i < keyStridesNum; ++i) {
            cout << keyStridesValue[i] << " ";
        }
        cout << endl;
    }
    */

    std::vector<int64_t> sizeData = {0, 0};

    cout << "step3" << endl;

    // k和v地址是否连续
    // if (!op::IsContiguous(k) && !op::IsContiguous(v)){
        cout << "step4" << endl;
        sizeData = {keyStridesValue[0], valueStridesValue[0]};
    // }

    cout << "step5" << endl;
    for (int i = 0; i < sizeData.size(); ++i) {
        cout << "sizeData[" << i << "] = " << *(sizeData.data() + i) << endl;
    }
    cout << "sizeData.size() = " << sizeData.size() << endl;
    aclIntArray *kvStridesOptional = aclCreateIntArray(sizeData.data(), sizeData.size());


    cout << "step6" << endl;

     if (kvStridesOptional == nullptr) {
        cout << "kvStridesOptional 创建失败" << endl;
    } else {
        cout << "kvStridesOptional 创建成功" << endl;
    }
    int64_t kStride0 = keyStridesValue[0];
    int64_t vStride0 = valueStridesValue[0];

    aclnnStatus ret = aclnnInnerChunkGatedDeltaRuleFwdHGetWorkspaceSize(k, w, u, g, initialStateOptional, cuSeqlensOptional, chunkIndicesOptional, outputFinalState,
        chunkSize, kStride0, vStride0, h, vNew, finalState, workspaceSize, executor);

    cout << "step7" << endl;
    aclDestroyIntArray(kvStridesOptional);

    cout << "step8" << endl;
    delete[] keyStridesValue;
    delete[] valueStridesValue;

    cout << "step9" << endl;
    return ret;
}


aclnnStatus aclnnChunkGatedDeltaRuleFwdH(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    return aclnnInnerChunkGatedDeltaRuleFwdH(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
