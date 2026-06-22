#include <dlfcn.h>
#include "aclnn_fused_gdn_gating_v310.h"
#include "fused_gdn_gating_v310.h"

#include "securec.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

#include "aclnn_kernels/contiguous.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

// 310P 参数解包结构体
struct FusedGdnGatingV310Params {
    const aclTensor *aLog{nullptr};
    const aclTensor *a{nullptr};
    const aclTensor *b{nullptr};
    const aclTensor *dtBias{nullptr};
    float beta{1.0f};
    float threshold{20.0f};
    aclTensor *g{nullptr};
    aclTensor *betaOutput{nullptr};
};

// 严格限定 310P 硬件合规性白名单：全部强制为 FP16，门控输出为 FP32
static const std::initializer_list<DataType> HALF_TYPE_SUPPORT_LIST = {DataType::DT_FLOAT16};
static const std::initializer_list<DataType> FLOAT_TYPE_SUPPORT_LIST = {DataType::DT_FLOAT};

static inline bool CheckNotNull(const FusedGdnGatingV310Params &params)
{
    OP_CHECK_NULL(params.aLog,   return false);
    OP_CHECK_NULL(params.a,      return false);
    OP_CHECK_NULL(params.b,      return false);
    OP_CHECK_NULL(params.dtBias, return false);
    OP_CHECK_NULL(params.g,          return false);
    OP_CHECK_NULL(params.betaOutput, return false);
    return true;
}

static inline bool CheckDtypeValid(const FusedGdnGatingV310Params &params)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(params.aLog,       HALF_TYPE_SUPPORT_LIST,  return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.a,          HALF_TYPE_SUPPORT_LIST,  return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.b,          HALF_TYPE_SUPPORT_LIST,  return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.dtBias,     HALF_TYPE_SUPPORT_LIST,  return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.g,          FLOAT_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.betaOutput, HALF_TYPE_SUPPORT_LIST,  return false);
    return true;
}

static aclnnStatus CheckParams(FusedGdnGatingV310Params &params)
{
    CHECK_RET(CheckDtypeValid(params), ACLNN_ERR_PARAM_INVALID);
    OP_LOGD("FusedGdnGatingV310 check params success.");
    return ACLNN_SUCCESS;
}

static aclnnStatus PreProcess(FusedGdnGatingV310Params &params)
{
    // 强制同步视角 Shape 到原始 Shape 描述符，规避动态 Shape 框架缺陷
    params.aLog->SetOriginalShape(params.aLog->GetViewShape());
    params.a->SetOriginalShape(params.a->GetViewShape());
    params.b->SetOriginalShape(params.b->GetViewShape());
    params.dtBias->SetOriginalShape(params.dtBias->GetViewShape());
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnFusedGdnGatingV310GetWorkspaceSize(
    const aclTensor *aLog, const aclTensor *a, const aclTensor *b,
    const aclTensor *dtBias, float beta, float threshold,
    aclTensor *g, aclTensor *betaOutput,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnFusedGdnGatingV310,
                   DFX_IN(aLog, a, b, dtBias, beta, threshold),
                   DFX_OUT(g, betaOutput));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    FusedGdnGatingV310Params params{aLog, a, b, dtBias, beta, threshold, g, betaOutput};

    CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    
    auto ret = PreProcess(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 强转连续内存块，严丝合缝匹配
    auto aLog_   = l0op::Contiguous(aLog,   uniqueExecutor.get());
    auto a_      = l0op::Contiguous(a,      uniqueExecutor.get());
    auto b_      = l0op::Contiguous(b,      uniqueExecutor.get());
    auto dtBias_ = l0op::Contiguous(dtBias, uniqueExecutor.get());
    
    auto g_          = l0op::Contiguous(g,          uniqueExecutor.get());
    auto betaOutput_ = l0op::Contiguous(betaOutput, uniqueExecutor.get());

    // 调用 310 专属的底层发射中心 (顺序严格保持大一统修正后的排列)
    auto outRet = l0op::FusedGdnGatingV310(aLog_, a_, b_, dtBias_, beta, threshold, uniqueExecutor.get());
    if (outRet.g == nullptr || outRet.beta_output == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto vcG = l0op::ViewCopy(outRet.g, g_, uniqueExecutor.get());
    if (vcG == nullptr) return ACLNN_ERR_INNER_NULLPTR;
    
    auto vcBeta = l0op::ViewCopy(outRet.beta_output, betaOutput_, uniqueExecutor.get());
    if (vcBeta == nullptr) return ACLNN_ERR_INNER_NULLPTR;

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFusedGdnGatingV310(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFusedGdnGatingV310);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
