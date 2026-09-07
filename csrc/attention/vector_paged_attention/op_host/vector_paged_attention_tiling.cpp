/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file vector_paged_attention_tiling.cpp
 * \brief Tiling implementation for VectorPagedAttention
 */
#include "vector_paged_attention_tiling.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"

namespace optiling {
namespace {
// The operator's declared domain. Everything outside it is rejected here and,
// earlier and with a better message, by the torch adapter -- a tiling failure
// during aclgraph capture is an error, not a fallback to another kernel.
constexpr int64_t SUPPORTED_HEAD_DIM = 64;
constexpr int64_t MAX_BLOCK_SIZE = 128;
constexpr int64_t MAX_KV_CAPACITY = 4096;
constexpr int64_t MAX_BATCH = 32;
constexpr int64_t MAX_NUM_HEADS = 128;

constexpr size_t QUERY_INDEX = 0;
constexpr size_t KEY_CACHE_INDEX = 1;
constexpr size_t VALUE_CACHE_INDEX = 2;
constexpr size_t BLOCK_TABLE_INDEX = 3;
constexpr size_t SEQ_LENS_INDEX = 4;

constexpr size_t ATTR_NUM_HEADS = 0;
constexpr size_t ATTR_NUM_KV_HEADS = 1;
constexpr size_t ATTR_SCALE = 2;

constexpr size_t KV_CACHE_DIM_NUM = 3;
constexpr size_t QUERY_DIM_NUM = 3;
constexpr size_t BLOCK_TABLE_DIM_NUM = 2;
constexpr size_t SEQ_LENS_DIM_NUM = 1;

struct VectorPagedAttentionParams {
    int64_t batch{0};
    int64_t numHeads{0};
    int64_t headDim{0};
    int64_t blockSize{0};
    int64_t maxBlocks{0};
    int64_t kvStride{0};
    int64_t numBlocks{0};
    int64_t numKvHeads{0};
    float scale{0.0F};
};

bool ReadAttrs(gert::TilingContext* context, VectorPagedAttentionParams& params)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return false;
    }
    const int64_t* numHeads = attrs->GetInt(ATTR_NUM_HEADS);
    const int64_t* numKvHeads = attrs->GetInt(ATTR_NUM_KV_HEADS);
    const float* scale = attrs->GetFloat(ATTR_SCALE);
    if (numHeads == nullptr || numKvHeads == nullptr || scale == nullptr) {
        return false;
    }
    params.numHeads = *numHeads;
    params.numKvHeads = *numKvHeads;
    params.scale = *scale;
    return true;
}

// query [batch, numHeads, headDim]
// keyCache / valueCache [numBlocks, blockSize, numKvHeads * headDim]
// blockTable [batch, maxBlocks], seqLens [batch]
bool ReadShapes(gert::TilingContext* context, VectorPagedAttentionParams& params)
{
    const gert::StorageShape* query = context->GetInputShape(QUERY_INDEX);
    const gert::StorageShape* keyCache = context->GetInputShape(KEY_CACHE_INDEX);
    const gert::StorageShape* valueCache = context->GetInputShape(VALUE_CACHE_INDEX);
    const gert::StorageShape* blockTable = context->GetInputShape(BLOCK_TABLE_INDEX);
    const gert::StorageShape* seqLens = context->GetInputShape(SEQ_LENS_INDEX);
    if (query == nullptr || keyCache == nullptr || valueCache == nullptr ||
        blockTable == nullptr || seqLens == nullptr) {
        return false;
    }
    const gert::Shape& q = query->GetStorageShape();
    const gert::Shape& k = keyCache->GetStorageShape();
    const gert::Shape& v = valueCache->GetStorageShape();
    const gert::Shape& bt = blockTable->GetStorageShape();
    const gert::Shape& sl = seqLens->GetStorageShape();
    if (q.GetDimNum() != QUERY_DIM_NUM || k.GetDimNum() != KV_CACHE_DIM_NUM ||
        v.GetDimNum() != KV_CACHE_DIM_NUM || bt.GetDimNum() != BLOCK_TABLE_DIM_NUM ||
        sl.GetDimNum() != SEQ_LENS_DIM_NUM) {
        return false;
    }
    for (size_t axis = 0; axis < KV_CACHE_DIM_NUM; ++axis) {
        if (k.GetDim(axis) != v.GetDim(axis)) {
            return false;
        }
    }
    params.batch = q.GetDim(0);
    params.headDim = q.GetDim(2);
    params.numBlocks = k.GetDim(0);
    params.blockSize = k.GetDim(1);
    params.kvStride = k.GetDim(2);
    params.maxBlocks = bt.GetDim(1);
    return q.GetDim(1) == params.numHeads && bt.GetDim(0) == params.batch &&
           sl.GetDim(0) == params.batch;
}

bool InDeclaredDomain(const VectorPagedAttentionParams& params)
{
    return params.numHeads >= 1 && params.numHeads <= MAX_NUM_HEADS &&
           // Multi-head only: one core owns one (request, head) and reads that
           // head's own slice of every page, so heads cannot share KV rows.
           params.numKvHeads == params.numHeads &&
           params.headDim == SUPPORTED_HEAD_DIM &&
           params.kvStride == params.numKvHeads * params.headDim &&
           params.batch >= 1 && params.batch <= MAX_BATCH &&
           params.blockSize >= 8 && params.blockSize <= MAX_BLOCK_SIZE &&
           // The value pass folds a page's rows pairwise, which needs no tail
           // handling only when the page holds a power-of-two number of rows.
           (params.blockSize & (params.blockSize - 1)) == 0 &&
           params.numBlocks >= 1 && params.maxBlocks >= 1 &&
           params.blockSize * params.maxBlocks <= MAX_KV_CAPACITY;
}
}  // namespace

static ge::graphStatus VectorPagedAttentionTilingFunc(gert::TilingContext* context)
{
    VectorPagedAttentionParams params;
    if (!ReadAttrs(context, params) || !ReadShapes(context, params)) {
        OP_LOGE(context->GetNodeName(), "VectorPagedAttention got malformed inputs or attributes.");
        return ge::GRAPH_FAILED;
    }
    if (!InDeclaredDomain(params)) {
        OP_LOGE(context->GetNodeName(),
                "VectorPagedAttention outside its declared domain: batch=%ld numHeads=%ld "
                "numKvHeads=%ld headDim=%ld blockSize=%ld maxBlocks=%ld. Requires headDim=64, "
                "numKvHeads==numHeads, batch<=32, a power-of-two blockSize<=128 and "
                "blockSize*maxBlocks<=4096.",
                params.batch, params.numHeads, params.numKvHeads, params.headDim,
                params.blockSize, params.maxBlocks);
        return ge::GRAPH_FAILED;
    }

    auto* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    if (aivCoreNum == 0) {
        aivCoreNum = ascendcPlatform.GetCoreNum();
    }

    // One AI vector core per (request, head). A decode step is tiny, so the
    // split that pays is the one needing no cross-core reduction: a core owns a
    // whole head, keeps its running softmax in UB and writes its own headDim
    // outputs, so the operator needs no workspace of its own.
    const int64_t blockDim = params.batch * params.numHeads;
    if (blockDim > static_cast<int64_t>(aivCoreNum)) {
        OP_LOGE(context->GetNodeName(),
                "VectorPagedAttention needs batch*numHeads (%ld) <= AIV core count (%u).",
                blockDim, aivCoreNum);
        return ge::GRAPH_FAILED;
    }

    VectorPagedAttentionTilingData tilingData;
    tilingData.set_batch(static_cast<uint32_t>(params.batch));
    tilingData.set_numHeads(static_cast<uint32_t>(params.numHeads));
    tilingData.set_headDim(static_cast<uint32_t>(params.headDim));
    tilingData.set_blockSize(static_cast<uint32_t>(params.blockSize));
    tilingData.set_maxBlocks(static_cast<uint32_t>(params.maxBlocks));
    tilingData.set_kvStride(static_cast<uint32_t>(params.kvStride));
    tilingData.set_kvCapacity(static_cast<uint32_t>(params.blockSize * params.maxBlocks));
    tilingData.set_numBlocks(static_cast<uint32_t>(params.numBlocks));
    tilingData.set_scale(params.scale);

    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    context->SetTilingKey(0);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSize);
    workspaceSize[0] = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForVectorPagedAttention(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(VectorPagedAttention)
    .Tiling(VectorPagedAttentionTilingFunc)
    .TilingParse<VectorPagedAttentionCompileInfo>(TilingParseForVectorPagedAttention);

}  // namespace optiling
