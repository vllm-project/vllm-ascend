/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file compress_norm_rope_tiling.cpp
 * \brief CompressNormRope A2/A3 tiling（两阶段重构版）
 *
 *   - C4（coff=2）：按"组"均分任务（组数上界 = T/r + 2B，start_pos 未对齐跨组）
 *   - C128（coff=1）：按"组×dChunk"均分任务（dChunk=64），压缩行经用户 workspace 中转二阶段
 * 保留 vllm-ascend 的 ASCENDC_TPL template key 自动分发（FP16/BF16 × coff 1/2）。
 */

#include <algorithm>
#include "log/log.h"
#include "err/ops_err.h"
#include "compress_norm_rope_tiling.h"

using namespace ge;

// 日志宏兼容层：vllm-ascend 无 OP_LOGE_FOR_INVALID_*_WITH_REASON，
// 映射到 printf 风格的 OP_LOGE（值细节进日志，不影响返回 GRAPH_FAILED 语义）
#define OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, paramName, incorrectValue, reason) \
    OP_LOGE((opName), "invalid " paramName " value, " reason)
#define OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, paramName, reason) \
    OP_LOGE((opName), "invalid " paramName ", " reason)
#define OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, paramName, incorrectValue, reason) \
    OP_LOGE((opName), "invalid " paramName " dim, " reason)

namespace optiling {
namespace {

constexpr uint32_t DIM_INDEX_0 = 0;
constexpr uint32_t DIM_INDEX_1 = 1;
constexpr uint32_t DIM_NUM_2 = 2;
constexpr uint32_t DIM_NUM_3 = 3;

constexpr uint32_t ROTARY_MODE_HALF = 1;
constexpr uint32_t ROTARY_MODE_INTERLEAVE = 2;
constexpr uint32_t FP32_BLOCK = 8; // 32B / sizeof(float)

ge::graphStatus CheckAttrSupport(gert::TilingContext *context, int64_t cmpRatio, int64_t coff, int64_t cacheMode,
                                 int64_t ropeHeadDim, int64_t rotaryMode, uint32_t headDim)
{
    OP_CHECK_IF(cmpRatio != (int64_t)CMP_RATIO_C4 && cmpRatio != (int64_t)CMP_RATIO_C128,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "cmp_ratio",
                                                      std::to_string(cmpRatio), "only supports 4, 128"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(coff != (int64_t)COFF_DISABLE && coff != (int64_t)COFF_OVERLAP,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "coff", std::to_string(coff),
                                                      "only supports 1, 2"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(coff == (int64_t)COFF_OVERLAP && cmpRatio != (int64_t)CMP_RATIO_C4,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "coff/cmp_ratio",
                                                      std::to_string(coff) + "/" + std::to_string(cmpRatio),
                                                      "coff=2 only supports cmp_ratio=4 (C4)"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(coff == (int64_t)COFF_DISABLE && cmpRatio != (int64_t)CMP_RATIO_C128,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "coff/cmp_ratio",
                                                      std::to_string(coff) + "/" + std::to_string(cmpRatio),
                                                      "coff=1 only supports cmp_ratio=128 (C128)"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(cacheMode != 1,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "cache_mode",
                                                      std::to_string(cacheMode), "only supports 1 (LINEAR_BUFFER)"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(rotaryMode != (int64_t)ROTARY_MODE_HALF && rotaryMode != (int64_t)ROTARY_MODE_INTERLEAVE,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "rotary_mode",
                                                      std::to_string(rotaryMode),
                                                      "only supports 1 (HALF), 2 (INTERLEAVE)"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ropeHeadDim <= 0 || ropeHeadDim % (int64_t)(2 * FP32_BLOCK) != 0 ||
                    (uint32_t)ropeHeadDim > headDim,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "rope_head_dim",
                                                      std::to_string(ropeHeadDim),
                                                      "should be positive multiple of 16 and <= headDim"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConvertContext(gert::TilingContext &context, CompressNormRopeContext &c)
{
    if (context.GetNodeName() == nullptr) {
        OP_LOGE("CompressNormRope", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    c.opName = context.GetNodeName();
    c.opType = context.GetNodeType();
    c.platformInfo = context.GetPlatformInfo();
    auto fillRequired = [&](RequiredParaInfo &info, uint32_t idx) {
        info.desc = context.GetRequiredInputDesc(idx);
        info.shape = context.GetRequiredInputShape(idx);
    };
    fillRequired(c.mmKv, MM_KV_INPUT_INDEX);
    fillRequired(c.mmScore, MM_SCORE_INPUT_INDEX);
    fillRequired(c.stateCache, STATE_CACHE_INPUT_INDEX);
    fillRequired(c.ape, APE_INPUT_INDEX);
    fillRequired(c.normWeight, NORM_WEIGHT_INPUT_INDEX);
    fillRequired(c.ropeSin, ROPE_SIN_INPUT_INDEX);
    fillRequired(c.ropeCos, ROPE_COS_INPUT_INDEX);
    auto fillOptional = [&](OptionalParaInfo &info, uint32_t idx) {
        info.desc = context.GetOptionalInputDesc(idx);
        info.shape = context.GetOptionalInputShape(idx);
        info.tensor = context.GetOptionalInputTensor(idx);
    };
    fillOptional(c.stateBlockTable, STATE_BLOCK_TABLE_INPUT_INDEX);
    fillOptional(c.cuSeqlens, CU_SEQ_LEN_INPUT_INDEX);
    fillOptional(c.seqUsed, SEQ_USED_INPUT_INDEX);
    fillOptional(c.startPos, START_POS_INPUT_INDEX);
    c.cmpKv.desc = context.GetOutputDesc(CMP_KV_OUTPUT_INDEX);
    c.cmpKv.shape = context.GetOutputShape(CMP_KV_OUTPUT_INDEX);

    auto attrs = context.GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context.GetNodeName(), "attrs got from ge is nullptr"),
               return ge::GRAPH_FAILED);
    c.ropeHeadDim = attrs->GetAttrPointer<int>(ROPE_HEAD_DIM_ATTR_INDEX);
    c.coff = attrs->GetAttrPointer<int>(COFF_ATTR_INDEX);
    c.cmpRatio = attrs->GetAttrPointer<int>(CMP_RATIO_ATTR_INDEX);
    c.normEps = attrs->GetAttrPointer<float>(NORM_EPS_ATTR_INDEX);
    c.rotaryMode = attrs->GetAttrPointer<int>(ROTARY_MODE_ATTR_INDEX);
    c.cacheMode = attrs->GetAttrPointer<int>(CACHE_MODE_ATTR_INDEX);
    c.stateCacheStrideDim0 = attrs->GetAttrPointer<int>(STATE_CACHE_STRIDE_DIM0_ATTR_INDEX);
    c.mmKvStrideDim0 = attrs->GetAttrPointer<int>(MM_KV_STRIDE_DIM0_ATTR_INDEX);
    c.mmScoreStrideDim0 = attrs->GetAttrPointer<int>(MM_SCORE_STRIDE_DIM0_ATTR_INDEX);

    OP_CHECK_IF(context.GetWorkspaceSizes(1) == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "workSpaceSize got from ge is nullptr"),
               return ge::GRAPH_FAILED);
    c.workSpaces = context.GetWorkspaceSizes(1);
    return ge::GRAPH_SUCCESS;
}

} // namespace

ge::graphStatus TilingCompressNormRope(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("CompressNormRope", "Context is nullptr."),
               return ge::GRAPH_FAILED);
    OP_LOGI("Getting Tiling");

    CompressNormRopeContext c{};
    if (ConvertContext(*context, c) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Error occurred while converting tilingContext to CompressNormRope context");
        return ge::GRAPH_FAILED;
    }
    const char *opName = c.opName;

    // ── 平台信息 ──
    OP_CHECK_IF(c.platformInfo == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "platformInfo", "is nullptr"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(c.platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(aivNum == 0, OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "aivNum", "is 0"),
                return ge::GRAPH_FAILED);
    size_t libapiSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    // ── 输入 shape / dtype ──
    auto mmKvShape = c.mmKv.shape;
    auto mmScoreShape = c.mmScore.shape;
    auto stateCacheShape = c.stateCache.shape;
    auto apeShape = c.ape.shape;
    auto sbtShape = c.stateBlockTable.shape;
    auto cuSeqlensShape = c.cuSeqlens.shape;
    auto normWeightShape = c.normWeight.shape;
    auto ropeSinShape = c.ropeSin.shape;
    auto ropeCosShape = c.ropeCos.shape;
    OP_CHECK_NULL_WITH_CONTEXT(context, mmKvShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, mmScoreShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, stateCacheShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, apeShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, sbtShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, cuSeqlensShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, normWeightShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, ropeSinShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, ropeCosShape);

    OP_CHECK_IF(mmKvShape->GetStorageShape().GetDimNum() != DIM_NUM_2,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, "mm_kv",
                                                         std::to_string(mmKvShape->GetStorageShape().GetDimNum()),
                                                         "only supports 2D [T, coff*headDim] (TH layout)"),
                return ge::GRAPH_FAILED);

    // ── 属性（默认值对齐 def）──
    int64_t ropeHeadDim = (c.ropeHeadDim != nullptr) ? *c.ropeHeadDim : 64;
    int64_t cmpRatio = (c.cmpRatio != nullptr) ? *c.cmpRatio : (int64_t)CMP_RATIO_C4;
    int64_t coff = (c.coff != nullptr) ? *c.coff : (int64_t)COFF_OVERLAP;
    float normEps = (c.normEps != nullptr) ? *c.normEps : 1e-6f;
    int64_t rotaryMode = (c.rotaryMode != nullptr) ? *c.rotaryMode : (int64_t)ROTARY_MODE_INTERLEAVE;
    int64_t cacheMode = (c.cacheMode != nullptr) ? *c.cacheMode : 1;
    int64_t strideDim0Attr = (c.stateCacheStrideDim0 != nullptr) ? *c.stateCacheStrideDim0 : 0;
    int64_t mmKvStrideDim0Attr = (c.mmKvStrideDim0 != nullptr) ? *c.mmKvStrideDim0 : 0;
    int64_t mmScoreStrideDim0Attr = (c.mmScoreStrideDim0 != nullptr) ? *c.mmScoreStrideDim0 : 0;

    uint32_t t = mmKvShape->GetStorageShape().GetDim(DIM_INDEX_0);
    uint32_t outDim = mmKvShape->GetStorageShape().GetDim(DIM_INDEX_1);
    OP_CHECK_IF(outDim % coff != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "mm_kv.dim(1)", std::to_string(outDim),
                                                      "should be divisible by coff"),
                return ge::GRAPH_FAILED);
    uint32_t headDim = outDim / (uint32_t)coff;
    uint64_t mmKvStrideDim0 = mmKvStrideDim0Attr > 0 ? (uint64_t)mmKvStrideDim0Attr : outDim;
    uint64_t mmScoreStrideDim0 = mmScoreStrideDim0Attr > 0 ? (uint64_t)mmScoreStrideDim0Attr : outDim;
    OP_CHECK_IF(mmKvStrideDim0 < outDim || mmKvStrideDim0 > UINT32_MAX,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "mm_kv_stride_dim0",
                                                      std::to_string(mmKvStrideDim0),
                                                      "should be in [mm_kv.dim(1), UINT32_MAX]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(mmScoreStrideDim0 < outDim || mmScoreStrideDim0 > UINT32_MAX,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "mm_score_stride_dim0",
                                                      std::to_string(mmScoreStrideDim0),
                                                      "should be in [mm_score.dim(1), UINT32_MAX]"),
                return ge::GRAPH_FAILED);
    if (CheckAttrSupport(context, cmpRatio, coff, cacheMode, ropeHeadDim, rotaryMode, headDim) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(normWeightShape->GetStorageShape().GetDimNum() != 1 ||
                    normWeightShape->GetStorageShape().GetDim(DIM_INDEX_0) != headDim,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "norm_weight", "shape should be [headDim]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ropeSinShape->GetStorageShape().GetDimNum() != DIM_NUM_2 ||
                    ropeSinShape->GetStorageShape().GetDim(DIM_INDEX_1) != ropeHeadDim,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "rope_sin",
                                                         "shape should be [scNum, rope_head_dim]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ropeCosShape->GetStorageShape().GetDim(DIM_INDEX_0) != ropeSinShape->GetStorageShape().GetDim(0) ||
                    ropeCosShape->GetStorageShape().GetDim(DIM_INDEX_1) != ropeHeadDim,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "rope_cos", "shape mismatch with rope_sin"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(headDim % D_CHUNK_SIZE_C128 != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "headDim", std::to_string(headDim),
                                                      "should be divisible by 64"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(mmScoreShape->GetStorageShape().GetDim(DIM_INDEX_0) != t ||
                    mmScoreShape->GetStorageShape().GetDim(DIM_INDEX_1) != outDim,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "mm_score", "shape mismatch with mm_kv"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(apeShape->GetStorageShape().GetDim(DIM_INDEX_0) != cmpRatio ||
                    apeShape->GetStorageShape().GetDim(DIM_INDEX_1) != outDim,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "ape", "shape should be [cmp_ratio, coff*headDim]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(stateCacheShape->GetStorageShape().GetDimNum() != DIM_NUM_3 ||
                    stateCacheShape->GetStorageShape().GetDim(DIM_INDEX_1) < 1 ||
                    stateCacheShape->GetStorageShape().GetDim(2) != 2 * outDim,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "state_cache",
                                                         "shape should be [blockNum, blockSize, 2*coff*headDim]"),
                return ge::GRAPH_FAILED);

    uint32_t batchSize = cuSeqlensShape->GetStorageShape().GetDim(DIM_INDEX_0) - 1;
    uint32_t blockSize = stateCacheShape->GetStorageShape().GetDim(DIM_INDEX_1);
    uint32_t maxBlockNumPerBatch = sbtShape->GetStorageShape().GetDim(DIM_INDEX_1);
    uint64_t stateStride0 =
        (strideDim0Attr > 0) ? (uint64_t)strideDim0Attr : (uint64_t)blockSize * 2 * outDim;

    // ── 空 tensor（EMPTY_X 模板：kernel 直接返回，仅需合法 key）──
    if (t == 0) {
        c.templateId = TemplateId::EMPTY_X;
        CompressNormRopeTilingData *tilingData = context->GetTilingData<CompressNormRopeTilingData>();
        OP_CHECK_IF(tilingData == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR(opName, "TilingData is nullptr."), return ge::GRAPH_FAILED);
        tilingData->batchSize = batchSize;
        tilingData->tokenSize = t;
        tilingData->headDim = headDim;
        tilingData->cmpRatio = (uint32_t)cmpRatio;
        tilingData->usedCoreNum = 1;
        tilingData->blockSize = blockSize;
        tilingData->maxBlockNumPerBatch = maxBlockNumPerBatch;
        tilingData->stateCacheStrideDim0 = stateStride0;
        tilingData->mmKvStrideDim0 = mmKvStrideDim0;
        tilingData->mmScoreStrideDim0 = mmScoreStrideDim0;
        tilingData->dChunkSize = (coff == (int64_t)COFF_DISABLE) ? D_CHUNK_SIZE_C128 : headDim;
        tilingData->dChunkNum = headDim / tilingData->dChunkSize;
        tilingData->maxGroupTaskNum = 0;
        tilingData->maxTaskNum = 0;
        tilingData->taskPerCore = 0;
        tilingData->taskRem = 0;
        tilingData->ropeHeadDim = (uint32_t)ropeHeadDim;
        tilingData->rotaryMode = (uint32_t)rotaryMode;
        tilingData->normEps = normEps;
        tilingData->maxScNum = 0;
        // GetTilingData<T> 已指向 raw tiling data 内存，直接填充即可（vllm-ascend 惯例，无 SaveToBuffer）

        const uint32_t xType = static_cast<uint32_t>(c.mmKv.desc->GetDataType());
        const uint32_t normType = static_cast<uint32_t>(c.normWeight.desc->GetDataType());
        const uint32_t ropeType = static_cast<uint32_t>(c.ropeSin.desc->GetDataType());
        c.tilingKey = GET_TPL_TILING_KEY(xType, normType, ropeType, static_cast<uint32_t>(coff), 1U);
        context->SetTilingKey(c.tilingKey);
        context->SetBlockDim(1);
        context->SetScheduleMode(BATCH_MODE_SCHEDULE);
        OP_LOGI(opName, "[EMPTY_X] T:%u key:%lu", t, c.tilingKey);
        return ge::GRAPH_SUCCESS;
    }
    c.templateId = TemplateId::NORMAL;

    // ── 任务切分 ──
    uint32_t dChunkSize = (coff == (int64_t)COFF_DISABLE) ? D_CHUNK_SIZE_C128 : headDim;
    uint32_t dChunkNum = headDim / dChunkSize;
    // 组数上界：start_pos 未按 r 对齐且 S 跨组边界时，单 batch 工作组数最坏 = S/r + 2
    // （如 P%r=122, S=8 → 跨 2 组），故上界 = T/r + 2B（T/r + B 会漏组！）
    uint32_t maxGroupTaskNum = t / cmpRatio + 2 * batchSize;
    uint32_t maxTaskNum = maxGroupTaskNum * dChunkNum;
    uint32_t blockDim = aivNum;
    if (maxTaskNum < blockDim) {
        blockDim = maxTaskNum == 0 ? 1 : maxTaskNum;
    }
    uint32_t taskPerCore = maxTaskNum / blockDim;
    uint32_t taskRem = maxTaskNum % blockDim;
    // 压缩行数上界（产出组数 ≤ S/r + 1 每 batch）：c128 二阶段 workspace 行数 & 行均分
    uint32_t maxScNum = std::min(t, (uint32_t)(t / cmpRatio) + batchSize);

    // ── tiling data ──
    CompressNormRopeTilingData *tilingData = context->GetTilingData<CompressNormRopeTilingData>();
    OP_CHECK_IF(tilingData == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName, "TilingData is nullptr."),
                return ge::GRAPH_FAILED);
    tilingData->batchSize = batchSize;
    tilingData->tokenSize = t;
    tilingData->headDim = headDim;
    tilingData->cmpRatio = (uint32_t)cmpRatio;
    tilingData->usedCoreNum = blockDim;
    tilingData->blockSize = blockSize;
    tilingData->maxBlockNumPerBatch = maxBlockNumPerBatch;
    tilingData->stateCacheStrideDim0 = stateStride0;
    tilingData->mmKvStrideDim0 = mmKvStrideDim0;
    tilingData->mmScoreStrideDim0 = mmScoreStrideDim0;
    tilingData->dChunkSize = dChunkSize;
    tilingData->dChunkNum = dChunkNum;
    tilingData->maxGroupTaskNum = maxGroupTaskNum;
    tilingData->maxTaskNum = maxTaskNum;
    tilingData->taskPerCore = taskPerCore;
    tilingData->taskRem = taskRem;
    tilingData->ropeHeadDim = (uint32_t)ropeHeadDim;
    tilingData->rotaryMode = (uint32_t)rotaryMode;
    tilingData->normEps = normEps;
    tilingData->maxScNum = maxScNum;

    // ── workspace：c128 压缩行（未 norm/rope fp32）中转 = maxScNum × headDim × 4B；c4 无用户 workspace ──
    size_t userWs = (coff == (int64_t)COFF_DISABLE) ? (size_t)maxScNum * headDim * sizeof(float) : 0;
    if (c.workSpaces != nullptr) {
        c.workSpaces[0] = libapiSize + userWs;
    }

    // ── template tiling key / blockDim / schedule ──
    const uint32_t xType = static_cast<uint32_t>(c.mmKv.desc->GetDataType());
    const uint32_t normType = static_cast<uint32_t>(c.normWeight.desc->GetDataType());
    const uint32_t ropeType = static_cast<uint32_t>(c.ropeSin.desc->GetDataType());
    c.tilingKey = GET_TPL_TILING_KEY(xType, normType, ropeType, static_cast<uint32_t>(coff), 0U);
    context->SetTilingKey(c.tilingKey);
    context->SetBlockDim(blockDim);
    // c128 二阶段 SyncAll 要求全核同调度（c4 无害）
    context->SetScheduleMode(BATCH_MODE_SCHEDULE);

    OP_LOGI(opName,
            "[TILING] T:%u B:%u headDim:%u cmpRatio:%ld coff:%ld blockSize:%u dChunk:%ux%u maxTask:%u "
            "blockDim:%u taskPerCore:%u rem:%u key:%lu ws:%zu",
            t, batchSize, headDim, cmpRatio, coff, blockSize, dChunkSize, dChunkNum, maxTaskNum, blockDim,
            taskPerCore, taskRem, c.tilingKey, libapiSize + userWs);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForCompressNormRope(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CompressNormRope)
    .Tiling(TilingCompressNormRope)
    .TilingParse<CompressNormRopeCompileInfo>(TilingPrepareForCompressNormRope);
} // namespace optiling
