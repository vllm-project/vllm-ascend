#include "aclnn_chunk_kda_fwd_finalize.h"
#include "chunk_kda_fwd_finalize.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {

constexpr int64_t FINALIZE_CHUNK_ROWS = 64;

struct FinalizeParams {
    const aclTensor *qgScaled;
    const aclTensor *aqk;
    const aclTensor *vNew;
    const aclTensor *h;
    const aclIntArray *cuSeqlens;
    const aclIntArray *chunkIndices;
    const char *outputLayout;
    bool stateVFirst;
    const aclTensor *attnOut;
};

struct FinalizeShape {
    int64_t batch = 0;
    int64_t heads = 0;
    int64_t seqLen = 0;
    int64_t totalChunks = 0;
    bool packed = false;
    bool sequenceMajor = false;
};

bool HasShape(const aclTensor *tensor, std::initializer_list<int64_t> dims)
{
    if (tensor == nullptr || tensor->GetViewShape().GetDimNum() != dims.size()) {
        return false;
    }
    size_t index = 0;
    for (int64_t dim : dims) {
        if (tensor->GetViewShape().GetDim(index++) != dim) {
            return false;
        }
    }
    return true;
}

aclnnStatus ResolveLayout(const FinalizeParams &params, FinalizeShape &shape)
{
    CHECK_COND(params.outputLayout != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "outputLayout 不能为 nullptr。");
    shape.packed = std::strcmp(params.outputLayout, "NTD") == 0 ||
                   std::strcmp(params.outputLayout, "TND") == 0;
    shape.sequenceMajor = std::strcmp(params.outputLayout, "BSND") == 0 ||
                          std::strcmp(params.outputLayout, "TND") == 0;
    CHECK_COND(shape.packed || shape.sequenceMajor ||
                   std::strcmp(params.outputLayout, "BNSD") == 0,
               ACLNN_ERR_PARAM_INVALID,
               "outputLayout 只支持大写 BSND、BNSD、TND 或 NTD，当前值=%s。",
               params.outputLayout);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckDtypeAndFormat(const FinalizeParams &params)
{
    const aclTensor *tensors[] = {
        params.qgScaled, params.aqk, params.vNew, params.h, params.attnOut};
    const char *names[] = {
        "qgScaled", "aqk", "vNew", "h", "attnOut"};
    for (size_t index = 0; index < 5; ++index) {
        CHECK_COND(tensors[index]->GetDataType() == DataType::DT_BF16,
                   ACLNN_ERR_PARAM_INVALID,
                   "%s 只支持 BF16。", names[index]);
        const auto storage = tensors[index]->GetStorageFormat();
        const auto view = tensors[index]->GetViewFormat();
        CHECK_COND(storage == Format::FORMAT_ND &&
                       view == Format::FORMAT_ND && !IsPrivateFormat(storage),
                   ACLNN_ERR_PARAM_INVALID,
                   "%s 必须为非私有 ND storage/view format，当前=%d/%d。",
                   names[index], static_cast<int>(storage), static_cast<int>(view));
    }
    CHECK_COND(IsContiguous(params.attnOut), ACLNN_ERR_PARAM_INVALID,
               "attnOut 由 kernel 直接写入，必须连续。");
    return ACLNN_SUCCESS;
}

aclnnStatus ResolveInputShape(const FinalizeParams &params,
                              FinalizeShape &shape)
{
    const auto &q = params.qgScaled->GetViewShape();
    if (shape.packed) {
        CHECK_COND(q.GetDimNum() == 3 && q.GetDim(2) == 128,
                   ACLNN_ERR_PARAM_INVALID,
                   "NTD/TND 的 qgScaled 必须为 head-major [HV,T,128]。");
        shape.batch = 1;
        shape.heads = q.GetDim(0);
        shape.seqLen = q.GetDim(1);
        CHECK_COND(
            HasShape(params.aqk, {shape.heads, shape.seqLen, 64}) &&
                (HasShape(params.vNew, {shape.heads, shape.seqLen, 128}) ||
                 HasShape(params.vNew, {1, shape.heads, shape.seqLen, 128})),
            ACLNN_ERR_PARAM_INVALID,
            "packed 输入要求 Aqk=[HV,T,64]，vNew=[HV,T,128] 或 FwdH 输出 [1,HV,T,128]。");
    } else {
        CHECK_COND(q.GetDimNum() == 4 && q.GetDim(3) == 128,
                   ACLNN_ERR_PARAM_INVALID,
                   "BNSD/BSND 的 qgScaled 必须为 head-major [B,HV,T,128]。");
        shape.batch = q.GetDim(0);
        shape.heads = q.GetDim(1);
        shape.seqLen = q.GetDim(2);
        CHECK_COND(HasShape(params.aqk,
                            {shape.batch, shape.heads, shape.seqLen, 64}) &&
                       HasShape(params.vNew,
                                {shape.batch, shape.heads, shape.seqLen, 128}),
                   ACLNN_ERR_PARAM_INVALID,
                   "dense 输入要求 Aqk=[B,HV,T,64]，vNew=[B,HV,T,128]。");
    }
    CHECK_COND(shape.batch > 0 && shape.heads > 0 && shape.seqLen > 0 &&
                   shape.batch <= std::numeric_limits<uint32_t>::max() &&
                   shape.heads <= std::numeric_limits<uint32_t>::max() &&
                   shape.seqLen <= std::numeric_limits<uint32_t>::max(),
               ACLNN_ERR_PARAM_INVALID,
               "B/HV/T 必须为正且可用 uint32 表示，当前 B=%ld,HV=%ld,T=%ld。",
               shape.batch, shape.heads, shape.seqLen);

    // 设备端以 64 位元素索引计算 head-major 数据和 H 偏移。
    const uint64_t b = static_cast<uint64_t>(shape.batch);
    const uint64_t hv = static_cast<uint64_t>(shape.heads);
    const uint64_t t = static_cast<uint64_t>(shape.seqLen);
    CHECK_COND(b <= std::numeric_limits<uint64_t>::max() / hv / t / 128,
               ACLNN_ERR_PARAM_INVALID,
               "B*HV*T*128 超过 uint64 元素索引范围。");
    return ACLNN_SUCCESS;
}

aclnnStatus ResolveSequenceShape(const FinalizeParams &params,
                                 FinalizeShape &shape)
{
    if (params.cuSeqlens == nullptr) {
        CHECK_COND(params.chunkIndices == nullptr, ACLNN_ERR_PARAM_INVALID,
                   "chunkIndicesOptional 要求同时传入 cuSeqlensOptional。");
        shape.totalChunks = (shape.seqLen - 1) / FINALIZE_CHUNK_ROWS + 1;
    } else {
        CHECK_COND(shape.batch == 1, ACLNN_ERR_PARAM_INVALID,
                   "变长输入要求 B=1，当前 B=%ld。", shape.batch);
        const auto *cu = params.cuSeqlens;
        CHECK_COND(cu->Size() >= 2 && (*cu)[0] == 0 &&
                       (*cu)[cu->Size() - 1] == shape.seqLen,
                   ACLNN_ERR_PARAM_INVALID,
                   "cuSeqlensOptional 必须从 0 到 T=%ld 且至少包含两个边界。",
                   shape.seqLen);
        uint64_t total = 0;
        for (size_t seq = 0; seq + 1 < cu->Size(); ++seq) {
            const int64_t length = (*cu)[seq + 1] - (*cu)[seq];
            CHECK_COND(length > 0 && (*cu)[seq + 1] <= shape.seqLen,
                       ACLNN_ERR_PARAM_INVALID,
                       "cuSeqlensOptional 必须严格递增，错误位置=%zu。", seq);
            total += static_cast<uint64_t>((length - 1) / FINALIZE_CHUNK_ROWS + 1);
            CHECK_COND(total <= std::numeric_limits<uint32_t>::max(),
                       ACLNN_ERR_PARAM_INVALID,
                       "变长 total_chunks 超出 uint32 范围。");
        }
        CHECK_COND(cu->Size() - 1 <= std::numeric_limits<uint32_t>::max(),
                   ACLNN_ERR_PARAM_INVALID,
                   "变长 sequence 数超出 uint32 范围。");
        shape.totalChunks = static_cast<int64_t>(total);
        if (params.chunkIndices != nullptr) {
            CHECK_COND(params.chunkIndices->Size() == total * 2,
                       ACLNN_ERR_PARAM_INVALID,
                       "chunkIndicesOptional 长度应为 2*total_chunks=%lu。",
                       static_cast<unsigned long>(total * 2));
            size_t offset = 0;
            for (size_t seq = 0; seq + 1 < cu->Size(); ++seq) {
                const int64_t length = (*cu)[seq + 1] - (*cu)[seq];
                const int64_t chunks = (length - 1) / FINALIZE_CHUNK_ROWS + 1;
                for (int64_t chunk = 0; chunk < chunks; ++chunk) {
                    CHECK_COND((*params.chunkIndices)[offset] ==
                                   static_cast<int64_t>(seq) &&
                                   (*params.chunkIndices)[offset + 1] == chunk,
                               ACLNN_ERR_PARAM_INVALID,
                               "chunkIndicesOptional 必须为规范的 sequence-major (seq,chunk) 序列。");
                    offset += 2;
                }
            }
        }
    }
    CHECK_COND(HasShape(params.h,
                        {shape.batch, shape.heads, shape.totalChunks, 128, 128}),
               ACLNN_ERR_PARAM_INVALID,
               "h 必须为 [B,HV,C,128,128]，当前 B=%ld,HV=%ld,C=%ld。",
               shape.batch, shape.heads, shape.totalChunks);
    const uint64_t b = static_cast<uint64_t>(shape.batch);
    const uint64_t hv = static_cast<uint64_t>(shape.heads);
    const uint64_t c = static_cast<uint64_t>(shape.totalChunks);
    CHECK_COND(b <= std::numeric_limits<uint64_t>::max() / hv / c / 128 / 128,
               ACLNN_ERR_PARAM_INVALID,
               "B*HV*C*128*128 超过 uint64 元素索引范围。");
    return ACLNN_SUCCESS;
}

aclnnStatus CheckOutputShape(const FinalizeParams &params,
                             const FinalizeShape &shape)
{
    const bool shapeValid = shape.packed
        ? (shape.sequenceMajor
            ? HasShape(params.attnOut, {shape.seqLen, shape.heads, 128})
            : HasShape(params.attnOut, {shape.heads, shape.seqLen, 128}))
        : (shape.sequenceMajor
            ? HasShape(params.attnOut, {shape.batch, shape.seqLen, shape.heads, 128})
            : HasShape(params.attnOut, {shape.batch, shape.heads, shape.seqLen, 128}));
    CHECK_COND(shapeValid, ACLNN_ERR_PARAM_INVALID,
               "attnOut shape 与 outputLayout=%s 不匹配，期望 B=%ld,HV=%ld,T=%ld,V=128。",
               params.outputLayout, shape.batch, shape.heads, shape.seqLen);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const FinalizeParams &params)
{
    CHECK_COND(params.qgScaled != nullptr && params.aqk != nullptr &&
                   params.vNew != nullptr && params.h != nullptr &&
                   params.attnOut != nullptr,
               ACLNN_ERR_PARAM_NULLPTR,
               "qgScaled、aqk、vNew、h 和 attnOut 均不能为空。");
    FinalizeShape shape;
    aclnnStatus status = ResolveLayout(params, shape);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckDtypeAndFormat(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = ResolveInputShape(params, shape);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = ResolveSequenceShape(params, shape);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    return CheckOutputShape(params, shape);
}

aclnnStatus MakeInputsContiguous(FinalizeParams &params,
                                 aclOpExecutor *executor)
{
    const aclTensor **inputs[] = {
        &params.qgScaled, &params.aqk, &params.vNew, &params.h};
    for (const aclTensor **input : inputs) {
        if (IsContiguous(*input)) {
            continue;
        }
        *input = l0op::Contiguous(*input, executor);
        CHECK_RET(*input != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnChunkKdaFwdFinalizeGetWorkspaceSize(
    const aclTensor *qgScaled,
    const aclTensor *aqk,
    const aclTensor *vNew,
    const aclTensor *h,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const char *outputLayout,
    bool stateVFirst,
    const aclTensor *attnOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    CHECK_COND(workspaceSize != nullptr && executor != nullptr,
               ACLNN_ERR_PARAM_NULLPTR,
               "workspaceSize 与 executor 不能为空。");
    L2_DFX_PHASE_1(
        aclnnChunkKdaFwdFinalize,
        DFX_IN(qgScaled, aqk, vNew, h, cuSeqlensOptional,
               chunkIndicesOptional, outputLayout, stateVFirst),
        DFX_OUT(attnOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr,
              ACLNN_ERR_INNER_CREATE_EXECUTOR);
    FinalizeParams params{qgScaled, aqk, vNew, h, cuSeqlensOptional,
                          chunkIndicesOptional, outputLayout, stateVFirst,
                          attnOut};
    aclnnStatus status = CheckParams(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = MakeInputsContiguous(params, uniqueExecutor.get());
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    const aclTensor *result = l0op::ChunkKdaFwdFinalize(
        params.qgScaled, params.aqk, params.vNew, params.h,
        params.cuSeqlens, params.chunkIndices, params.outputLayout,
        params.stateVFirst, params.attnOut, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkKdaFwdFinalize(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkKdaFwdFinalize);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor,
                                   stream) == ACLNN_SUCCESS,
               ACLNN_ERR_INNER,
               "ChunkKdaFwdFinalize AI Core 启动失败。");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
