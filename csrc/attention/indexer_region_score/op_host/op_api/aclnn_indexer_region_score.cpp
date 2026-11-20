// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "aclnn_indexer_region_score.h"
#include <cstdint>
#include "aclnn_kernels/contiguous.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
using namespace op;
namespace l0op {
OP_TYPE_REGISTER(TensorMove);
// Flatten only the logical dense span; caller backing storage may be much larger.
static const aclTensor *MaterializeAligned(const aclTensor *input, const aclTensor *source, aclOpExecutor *executor)
{
    const auto count = [](const Shape &shape, int64_t &elements) {
        elements = 1;
        for (size_t axis = 0; axis < shape.GetDimNum(); ++axis) {
            const int64_t dim = shape.GetDim(axis);
            if (dim < 0 || (dim != 0 && elements > INT64_MAX / dim)) return false;
            elements *= dim;
        }
        return true;
    };
    int64_t elements = 0, capacity = 0;
    const auto *backing = input->GetStorage() == source->GetStorage() ? source : input;
    const int64_t offset = input->GetViewOffset();
    if (!count(input->GetViewShape(), elements) || !count(backing->GetStorageShape(), capacity) ||
        offset < 0 || offset > capacity || elements > capacity - offset) return nullptr;
    const Shape flat{elements};
    const auto *bounded = executor->CreateView(input, flat, Shape{offset + elements}, Strides{1}, offset);
    auto *output = executor->AllocTensor(input->GetViewShape(), input->GetDataType());
    if (!bounded || !output) return nullptr;
    // Match TensorMove's physical 1-D extents, retaining canonical logical output strides.
    const auto *destination = executor->CreateView(output, flat, flat, Strides{1}, 0);
    if (!destination || ADD_TO_LAUNCHER_LIST_AICORE(
        TensorMove, OP_INPUT(bounded), OP_OUTPUT(destination), OP_ATTR()) != ACLNN_SUCCESS) return nullptr;
    return output;
}
} // namespace l0op
namespace {
bool IsAligned(const aclTensor *tensor)
{
    const uint64_t bytes = TypeSize(tensor->GetDataType());
    return bytes > 0 && bytes <= 32 && 32 % bytes == 0 && tensor->GetViewOffset() >= 0 &&
        static_cast<uint64_t>(tensor->GetViewOffset()) % (32 / bytes) == 0 &&
        reinterpret_cast<uintptr_t>(tensor->GetStorageAddr()) % 32 == 0;
}

bool IsPagedKeyView(const aclTensor *key)
{
    const auto &shape = key->GetViewShape();
    const auto &strides = key->GetViewStrides();
    const int64_t bytes = TypeSize(key->GetDataType()), offset = key->GetViewOffset();
    constexpr int64_t DIM = 256, MAX_SUMMARIES = 1024;
    return IsAligned(key) && shape.GetDimNum() == 4 && strides.size() == 4 && shape.GetDim(0) > 0 &&
        shape.GetDim(1) > 0 && shape.GetDim(1) <= MAX_SUMMARIES && shape.GetDim(2) == 1 && shape.GetDim(3) == DIM &&
        strides[3] == 1 && (shape.GetDim(1) == 1 || strides[1] == DIM) &&
        strides[0] >= shape.GetDim(1) * DIM && strides[0] % (32 / bytes) == 0 &&
        offset <= INT64_MAX / bytes - shape.GetDim(1) * DIM &&
        shape.GetDim(0) - 1 <= (INT64_MAX / bytes - shape.GetDim(1) * DIM - offset) / strides[0];
}

const aclTensor *PrepareTensor(const aclTensor *tensor, bool pagedKey, bool output, aclOpExecutor *executor)
{
    if (output && (!IsContiguous(tensor) || !IsAligned(tensor))) {
        // IRS overwrites outputs; do not read uninitialized caller buffers.
        return executor->AllocTensor(tensor->GetViewShape(), tensor->GetDataType());
    }
    const auto *dense = output || (pagedKey && IsPagedKeyView(tensor)) ? tensor : l0op::Contiguous(tensor, executor);
    if (!dense) return nullptr;
    if (!IsAligned(dense) || (pagedKey && !IsPagedKeyView(dense))) {
        dense = l0op::MaterializeAligned(dense, tensor, executor);
        if (!dense) return nullptr;
    }
    const auto *backing = dense->GetStorage() == tensor->GetStorage() ? tensor : dense;
    return executor->CreateView(dense, dense->GetViewShape(), backing->GetStorageShape(),
        dense->GetViewStrides(), dense->GetViewOffset());
}
} // namespace
namespace l0op {
OP_TYPE_REGISTER(IndexerRegionScore);
aclnnStatus IndexerRegionScore(
    const aclTensor *query, const aclTensor *weights, const aclTensor *key,
    const aclTensor *lengths, const aclTensor *table, const aclTensor *cu,
    int64_t regions, int64_t topK, int64_t regionSize, double headScale,
    int64_t cacheMode, int64_t maskMode, int64_t sortMode,
    const aclTensor *output, const aclTensor *ids, aclOpExecutor *executor)
{
    L0_DFX(IndexerRegionScore, query, weights, key, lengths, table, cu,
        regions, topK, regionSize, headScale, cacheMode, maskMode, sortMode, output, ids);
    const bool returnScore = output != nullptr;
    if (!output) output = executor->AllocTensor(Shape{0}, DataType::DT_FLOAT);
    if (!output) return ACLNN_ERR_INNER_NULLPTR;
    return ADD_TO_LAUNCHER_LIST_AICORE(IndexerRegionScore,
        OP_INPUT(query, weights, key, lengths, table, cu), OP_OUTPUT(output, ids),
        OP_ATTR(regions, topK, regionSize, static_cast<float>(headScale), cacheMode, maskMode, sortMode, returnScore));
}
} // namespace l0op
extern "C" aclnnStatus aclnnIndexerRegionScoreGetWorkspaceSize(
    const aclTensor *query, const aclTensor *weights, const aclTensor *key,
    const aclTensor *lengths, const aclTensor *table, const aclTensor *cu,
    int64_t regions, int64_t topK, int64_t regionSize, double headScale,
    int64_t cacheMode, int64_t maskMode, int64_t sortMode,
    const aclTensor *output, const aclTensor *ids, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnIndexerRegionScore,
        DFX_IN(query, weights, key, lengths, table, cu, regions, topK, regionSize, headScale,
            cacheMode, maskMode, sortMode), DFX_OUT(output, ids));
    if (!workspaceSize || !executor) return ACLNN_ERR_PARAM_NULLPTR;
    *workspaceSize = 0;
    *executor = nullptr;
    if (cacheMode != 1 || maskMode != 3 || sortMode != 1) return ACLNN_ERR_PARAM_INVALID;
    for (const aclTensor *tensor : {query, weights, key, lengths, table, cu, ids}) {
        if (!tensor) return ACLNN_ERR_PARAM_NULLPTR;
        if (tensor->GetViewOffset() < 0) return ACLNN_ERR_PARAM_INVALID;
        tensor->SetOriginalShape(tensor->GetViewShape());
    }
    if (output) output->SetOriginalShape(output->GetViewShape());
    if (output && output->GetViewOffset() < 0) return ACLNN_ERR_PARAM_INVALID;
    auto exec = CREATE_EXECUTOR();
    if (!exec.get()) return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    const aclTensor *views[8]{};
    const aclTensor *tensors[] = {query, weights, key, lengths, table, cu, output, ids};
    for (uint32_t i = 0; i < 8; ++i) {
        if (!tensors[i]) continue;
        views[i] = PrepareTensor(tensors[i], i == 2, i >= 6, exec.get());
        if (!views[i]) return ACLNN_ERR_INNER_NULLPTR;
    }
    auto ret = l0op::IndexerRegionScore(views[0], views[1], views[2], views[3], views[4], views[5],
        regions, topK, regionSize, headScale, cacheMode, maskMode, sortMode, views[6], views[7], exec.get());
    if (ret != ACLNN_SUCCESS) return ret;
    if ((output && !l0op::ViewCopy(views[6], output, exec.get())) ||
        !l0op::ViewCopy(views[7], ids, exec.get())) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    *workspaceSize = exec->GetWorkspaceSize();
    exec.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
extern "C" aclnnStatus aclnnIndexerRegionScore(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnIndexerRegionScore);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
