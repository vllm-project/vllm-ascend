/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_dispatch_gmm_combine.h"
#include <algorithm>
// #include "aclnn_kernels/common/op_error_check.h"
// #include "opdev/op_log.h"
// #include "opdev/common_types.h"
// #include "opdev/platform.h"
// #include "ophost/matmul_util.h"
#include <unistd.h>
#include <vector>
#include <string>
#include <iostream>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <climits>
#include "../op_host/error_log.h"
// using namespace op;

// using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t TWO_DIMS = 2;
static constexpr int64_t KVALUE_MIN = 256;
static constexpr int64_t KVALUE_MAX = 65535;
static constexpr size_t HCCL_GROUP_NAME_MAX = 128U;
enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};
// static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
//     op::DataType::DT_INT8,
//     op::DataType::DT_FLOAT16,
//     op::DataType::DT_BF16
// };
extern aclnnStatus aclnnInnerDispatchGmmCombineGetWorkspaceSize(const aclTensor* x, const aclTensor* weight1, const aclTensor* weight2,
                                                         const aclTensor* expertId, const aclTensor* scale1, const aclTensor* scale2,
                                                         const aclTensor* probs,
                                                         const char* group, int64_t maxOutputSize,
                                                         bool transB, bool weightNz,
                                                         aclTensor* out,
                                                         uint64_t* workspaceSize, aclOpExecutor** executor);
extern aclnnStatus aclnnInnerDispatchGmmCombine(void *workspace, uint64_t workspaceSize,
                                            aclOpExecutor *executor, aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

// check nullptr
static bool CheckNotNull(const aclTensor* x, const aclTensor* weight1, const aclTensor* weight2, const aclTensor* output) {
    // OP_CHECK_NULL(x, return false);
    // OP_CHECK_NULL(weight1, return false);
    // OP_CHECK_NULL(weight2, return false);
    // OP_CHECK_NULL(output, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* weight1, const aclTensor* weight2, const aclTensor* output) {
    // OP_CHECK_DTYPE_NOT_SUPPORT(x, DTYPE_SUPPORT_LIST, return false);
    // OP_CHECK_DTYPE_NOT_SUPPORT(weight1, DTYPE_SUPPORT_LIST, return false);
    // OP_CHECK_DTYPE_NOT_SUPPORT(weight2, DTYPE_SUPPORT_LIST, return false);
    // OP_CHECK_DTYPE_NOT_SUPPORT(output, DTYPE_SUPPORT_LIST, return false);

    return true;
}

// 入参教验
static aclnnStatus CheckParams(const aclTensor *x, const aclTensor *weight1, const aclTensor *weight2, const aclTensor *output,
                               const char* group)
{
    OP_LOGD("DispatchGmmCombine CheckParams start");
    // CHECK_RET(CheckNotNull(x, weight1, weight2, output), ACLNN_ERR_PARAM_NULLPTR);
    // CHECK_RET(CheckDtypeValid(x, weight1, weight2, output), ACLNN_ERR_PARAM_NULLPTR);
    // if (strnlen(group, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        // OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required group name exceeds %zu", HCCL_GROUP_NAME_MAX);
        // return ACLNN_ERR_PARAM_NULLPTR;
    // }
    OP_LOGD("DispatchGmmCombine CheckParams success");
    return ACL_SUCCESS;
}

// static bool IsWeightNZFormat(const aclTensor* x2) {
//     auto format = ge::GetPrimaryFormat(x2->GetStorageFormat());
//     if (format == Format::FORMAT_FRACTAL_NZ) {
//         return true;
//     }
//     return false;
// }

aclnnStatus aclnnDispatchGmmCombineGetWorkspaceSize(const aclTensor* x, const aclTensor* weight1, const aclTensor* weight2,
                                                    const aclTensor* expertId, const aclTensor* scale1, const aclTensor* scale2,
                                                    const aclTensor* probs,
                                                    const char* group, int64_t maxOutputSize,
                                                    aclTensor* out,
                                                    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGD("aclnnDispatchGmmCombineGetWorkspaceSize start");
    // auto ret_param = CheckParams(x, weight1, weight2, out, group);
    // CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);

    bool transB = false;//IsTransposeLastTwoDims(weight1);
    bool weightNz = true;//IsWeightNZFormat(weight1);

    aclnnStatus ret = aclnnInnerDispatchGmmCombineGetWorkspaceSize(x, weight1, weight2, expertId, scale1, scale2, probs, group, 
                                                                    maxOutputSize, transB, weightNz,
                                                                    out, workspaceSize, executor);
    OP_LOGD("DispatchGmmCombine, aclnnInnerGetWorkspaceSize ret = %d.", ret);
    return ret;
}

aclnnStatus aclnnDispatchGmmCombine(void* workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    OP_LOGD("aclnnDispatchGmmCombine start");
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    aclnnStatus ret = aclnnInnerDispatchGmmCombine(workspace, workspaceSize, executor, stream);
    return ret;
}
#ifdef __cplusplus
}
#endif