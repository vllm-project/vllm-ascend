#include "chunk_kda_fwd_finalize.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ChunkKdaFwdFinalize);

namespace {

const aclTensor *ConvertIntArray(const aclIntArray *array,
                                 aclOpExecutor *executor)
{
    if (array == nullptr) {
        return nullptr;
    }
    const aclTensor *tensor = executor->ConvertToTensor(array, DataType::DT_INT64);
    if (tensor == nullptr) {
        return nullptr;
    }
    auto *mutableTensor = const_cast<aclTensor *>(tensor);
    mutableTensor->SetStorageFormat(Format::FORMAT_ND);
    mutableTensor->SetViewFormat(Format::FORMAT_ND);
    mutableTensor->SetOriginalFormat(Format::FORMAT_ND);
    return tensor;
}

} // namespace

const aclTensor *ChunkKdaFwdFinalize(
    const aclTensor *qgScaled,
    const aclTensor *aqk,
    const aclTensor *vNew,
    const aclTensor *h,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const char *outputLayout,
    bool stateVFirst,
    const aclTensor *attnOut,
    aclOpExecutor *executor)
{
    L0_DFX(ChunkKdaFwdFinalize, qgScaled, aqk, vNew, h,
           cuSeqlensOptional, chunkIndicesOptional, outputLayout,
           stateVFirst, attnOut);
    const aclTensor *cuSeqlens = ConvertIntArray(cuSeqlensOptional, executor);
    const aclTensor *chunkIndices = ConvertIntArray(chunkIndicesOptional, executor);
    if ((cuSeqlensOptional != nullptr && cuSeqlens == nullptr) ||
        (chunkIndicesOptional != nullptr && chunkIndices == nullptr)) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                "转换 cu_seqlens/chunk_indices 到 Tensor 失败。");
        return nullptr;
    }
    const auto status = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkKdaFwdFinalize,
        OP_INPUT(qgScaled, aqk, vNew, h, cuSeqlens, chunkIndices),
        OP_OUTPUT(attnOut),
        OP_ATTR(outputLayout, stateVFirst));
    if (status != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "添加 ChunkKdaFwdFinalize AI Core 任务失败。");
        return nullptr;
    }
    return attnOut;
}

} // namespace l0op
