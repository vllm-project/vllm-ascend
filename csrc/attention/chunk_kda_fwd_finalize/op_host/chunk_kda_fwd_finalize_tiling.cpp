#include "chunk_kda_fwd_finalize_tiling.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>

#include "chunk_kda_fwd_finalize_tiling_processor.h"
#include "../op_kernel/chunk_kda_fwd_finalize_tiling_key.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
namespace {

constexpr uint32_t FINALIZE_MIX_BATCH_MODE = 1;
constexpr uint64_t FINALIZE_AIV_MOVER_DENSE_MIN_CHUNKS_PER_CORE = 4;
constexpr uint64_t FINALIZE_AIV_MOVER_VARLEN_MIN_CHUNKS_PER_CORE = 8;

struct FinalizeShape {
    uint64_t batch = 0;
    uint64_t heads = 0;
    uint64_t seqLen = 0;
    bool packed = false;
    bool sequenceMajor = false;
};

bool HasShape(const gert::Shape &shape, std::initializer_list<int64_t> dims)
{
    if (shape.GetDimNum() != dims.size()) {
        return false;
    }
    size_t index = 0;
    for (int64_t dim : dims) {
        if (shape.GetDim(index++) != dim) {
            return false;
        }
    }
    return true;
}

// 形状不匹配时把实际输入/输出形状写进日志，便于直接定位调用方 shape 问题。
void FormatShape(const gert::Shape &shape, char *buffer, size_t bufferSize)
{
    size_t offset = 0;
    const size_t dimNum = shape.GetDimNum();
    offset += static_cast<size_t>(snprintf(buffer + offset, bufferSize - offset,
                                          "rank%zu[", dimNum));
    for (size_t index = 0; index < dimNum && offset < bufferSize; ++index) {
        offset += static_cast<size_t>(snprintf(buffer + offset, bufferSize - offset, "%s%ld",
                                              index == 0 ? "" : ",", shape.GetDim(index)));
    }
    (void)snprintf(buffer + (offset < bufferSize ? offset : bufferSize - 2),
                   bufferSize - (offset < bufferSize ? offset : bufferSize - 2), "]");
}

void FormatInputShape(gert::TilingContext *context, size_t index, char *buffer,
                      size_t bufferSize)
{
    const auto *shape = context->GetInputShape(index);
    if (shape == nullptr) {
        (void)snprintf(buffer, bufferSize, "null");
        return;
    }
    FormatShape(shape->GetStorageShape(), buffer, bufferSize);
}

void FormatOutputShape(gert::TilingContext *context, size_t index, char *buffer,
                       size_t bufferSize)
{
    const auto *shape = context->GetOutputShape(index);
    if (shape == nullptr) {
        (void)snprintf(buffer, bufferSize, "null");
        return;
    }
    FormatShape(shape->GetStorageShape(), buffer, bufferSize);
}

bool ReadLayout(const char *layout, FinalizeShape &shape)
{
    if (layout == nullptr) {
        return false;
    }
    shape.packed = std::strcmp(layout, "NTD") == 0 ||
                   std::strcmp(layout, "TND") == 0;
    shape.sequenceMajor = std::strcmp(layout, "BSND") == 0 ||
                          std::strcmp(layout, "TND") == 0;
    return shape.packed || shape.sequenceMajor ||
           std::strcmp(layout, "BNSD") == 0;
}

bool ResolveInputs(gert::TilingContext *context, FinalizeShape &info)
{
    const auto *q = context->GetInputShape(FINALIZE_INPUT_QG_SCALED);
    const auto *aqk = context->GetInputShape(FINALIZE_INPUT_AQK);
    const auto *vNew = context->GetInputShape(FINALIZE_INPUT_V_NEW);
    const auto *h = context->GetInputShape(FINALIZE_INPUT_H);
    const auto *out = context->GetOutputShape(0);
    if (q == nullptr || aqk == nullptr || vNew == nullptr ||
        h == nullptr || out == nullptr) {
        return false;
    }
    const gert::Shape &qShape = q->GetStorageShape();
    const gert::Shape &aShape = aqk->GetStorageShape();
    const gert::Shape &vShape = vNew->GetStorageShape();
    const gert::Shape &hShape = h->GetStorageShape();
    const gert::Shape &oShape = out->GetStorageShape();
    if (info.packed) {
        if (qShape.GetDimNum() != 3 || qShape.GetDim(2) != 128) {
            return false;
        }
        info.batch = 1;
        info.heads = static_cast<uint64_t>(qShape.GetDim(0));
        info.seqLen = static_cast<uint64_t>(qShape.GetDim(1));
        const bool vShapeValid =
            HasShape(vShape, {static_cast<int64_t>(info.heads),
                              static_cast<int64_t>(info.seqLen), 128}) ||
            HasShape(vShape, {1, static_cast<int64_t>(info.heads),
                              static_cast<int64_t>(info.seqLen), 128});
        if (!HasShape(aShape, {static_cast<int64_t>(info.heads),
                               static_cast<int64_t>(info.seqLen), 64}) ||
            !vShapeValid) {
            return false;
        }
        return info.sequenceMajor
            ? HasShape(oShape, {static_cast<int64_t>(info.seqLen),
                                static_cast<int64_t>(info.heads), 128})
            : HasShape(oShape, {static_cast<int64_t>(info.heads),
                                static_cast<int64_t>(info.seqLen), 128});
    }
    if (qShape.GetDimNum() != 4 || qShape.GetDim(3) != 128) {
        return false;
    }
    info.batch = static_cast<uint64_t>(qShape.GetDim(0));
    info.heads = static_cast<uint64_t>(qShape.GetDim(1));
    info.seqLen = static_cast<uint64_t>(qShape.GetDim(2));
    if (!HasShape(aShape, {static_cast<int64_t>(info.batch),
                           static_cast<int64_t>(info.heads),
                           static_cast<int64_t>(info.seqLen), 64}) ||
        !HasShape(vShape, {static_cast<int64_t>(info.batch),
                           static_cast<int64_t>(info.heads),
                           static_cast<int64_t>(info.seqLen), 128})) {
        return false;
    }
    return info.sequenceMajor
        ? HasShape(oShape, {static_cast<int64_t>(info.batch),
                            static_cast<int64_t>(info.seqLen),
                            static_cast<int64_t>(info.heads), 128})
        : HasShape(oShape, {static_cast<int64_t>(info.batch),
                            static_cast<int64_t>(info.heads),
                            static_cast<int64_t>(info.seqLen), 128});
}

bool ResolveSequenceInfo(gert::TilingContext *context,
                         const FinalizeShape &info, uint64_t &seqNum,
                         uint64_t &totalChunks, bool &isVarLen)
{
    const auto *cu = context->GetOptionalInputTensor(
        FINALIZE_INPUT_CU_SEQLENS);
    const auto *indices = context->GetOptionalInputTensor(
        FINALIZE_INPUT_CHUNK_INDICES);
    const auto *cuDesc = context->GetOptionalInputDesc(
        FINALIZE_INPUT_CU_SEQLENS);
    const auto *indexDesc = context->GetOptionalInputDesc(
        FINALIZE_INPUT_CHUNK_INDICES);
    const auto *cuShape = context->GetOptionalInputShape(
        FINALIZE_INPUT_CU_SEQLENS);
    const auto *indexShape = context->GetOptionalInputShape(
        FINALIZE_INPUT_CHUNK_INDICES);
    if ((cu != nullptr) != (cuDesc != nullptr) ||
        (cu != nullptr) != (cuShape != nullptr) ||
        (indices != nullptr) != (indexDesc != nullptr) ||
        (indices != nullptr) != (indexShape != nullptr)) {
        OP_LOGE(context->GetNodeName(),
                "可选 metadata 的 descriptor、shape 与常量值必须同时存在。");
        return false;
    }
    isVarLen = cu != nullptr;
    if (!isVarLen) {
        if (indices != nullptr) {
            OP_LOGE(context->GetNodeName(),
                    "chunk_indices 必须和 cu_seqlens 一起传入。");
            return false;
        }
        seqNum = info.batch;
        totalChunks = (info.seqLen - 1) / FINALIZE_CHUNK_ROWS + 1;
        return true;
    }
    if (info.batch != 1) {
        OP_LOGE(context->GetNodeName(),
                "变长输入要求 B=1，当前 B=%lu。",
                static_cast<unsigned long>(info.batch));
        return false;
    }
    const gert::Shape &cuStorageShape = cu->GetStorageShape();
    if (cuStorageShape.GetDimNum() != 1 || cuStorageShape.GetDim(0) < 2) {
        return false;
    }
    const int64_t *lengths = cu->GetData<int64_t>();
    if (lengths == nullptr) {
        return false;
    }
    seqNum = static_cast<uint64_t>(cuStorageShape.GetDim(0) - 1);
    if (lengths[0] != 0 ||
        lengths[seqNum] != static_cast<int64_t>(info.seqLen)) {
        return false;
    }
    totalChunks = 0;
    for (uint64_t seq = 0; seq < seqNum; ++seq) {
        if (lengths[seq + 1] <= lengths[seq] ||
            lengths[seq + 1] > static_cast<int64_t>(info.seqLen)) {
            OP_LOGE(context->GetNodeName(),
                    "cu_seqlens 必须从 0 到 T 严格递增，错误序列=%lu。",
                    static_cast<unsigned long>(seq));
            return false;
        }
        const uint64_t length = static_cast<uint64_t>(
            lengths[seq + 1] - lengths[seq]);
        totalChunks += (length - 1) / FINALIZE_CHUNK_ROWS + 1;
        if (totalChunks > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
    }
    if (indices == nullptr) {
        return true;
    }
    const gert::Shape &indexStorageShape = indices->GetStorageShape();
    if (indexStorageShape.GetDimNum() != 1 ||
        indexStorageShape.GetDim(0) != static_cast<int64_t>(2 * totalChunks)) {
        return false;
    }
    const int64_t *pairs = indices->GetData<int64_t>();
    if (pairs == nullptr) {
        return false;
    }
    uint64_t offset = 0;
    for (uint64_t seq = 0; seq < seqNum; ++seq) {
        const uint64_t length = static_cast<uint64_t>(
            lengths[seq + 1] - lengths[seq]);
        const uint64_t chunks = (length - 1) / FINALIZE_CHUNK_ROWS + 1;
        for (uint64_t chunk = 0; chunk < chunks; ++chunk) {
            if (pairs[offset] != static_cast<int64_t>(seq) ||
                pairs[offset + 1] != static_cast<int64_t>(chunk)) {
                OP_LOGE(context->GetNodeName(),
                        "chunk_indices 必须是规范的 sequence-major (seq,chunk) 序列。");
                return false;
            }
            offset += 2;
        }
    }
    return true;
}

} // namespace

ge::graphStatus Tiling4ChunkKdaFwdFinalize(gert::TilingContext *context)
{
    const auto *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const char *layout = attrs->GetStr(FINALIZE_ATTR_OUTPUT_LAYOUT);
    const bool *stateVFirst = attrs->GetAttrPointer<bool>(
        FINALIZE_ATTR_STATE_V_FIRST);
    FinalizeShape info;
    if (stateVFirst == nullptr || !ReadLayout(layout, info)) {
        OP_LOGE(context->GetNodeName(),
                "output_layout 只支持 BNSD/BSND/NTD/TND，且 state_v_first 必须提供。");
        return ge::GRAPH_FAILED;
    }
    for (size_t index = 0; index <= FINALIZE_INPUT_H; ++index) {
        const auto *desc = context->GetInputDesc(index);
        if (desc == nullptr || desc->GetDataType() != ge::DT_BF16) {
            OP_LOGE(context->GetNodeName(),
                    "qg_scaled/Aqk/v_new/h 均必须为 BF16，错误输入编号=%zu。",
                    index);
            return ge::GRAPH_FAILED;
        }
    }
    const auto *outputDesc = context->GetOutputDesc(0);
    if (outputDesc == nullptr || outputDesc->GetDataType() != ge::DT_BF16 ||
        !ResolveInputs(context, info) || info.batch == 0 || info.heads == 0 ||
        info.seqLen == 0 ||
        info.batch > std::numeric_limits<uint32_t>::max() ||
        info.heads > std::numeric_limits<uint32_t>::max() ||
        info.seqLen > std::numeric_limits<uint32_t>::max()) {
        char qgBuffer[96] = {0};
        char aqkBuffer[96] = {0};
        char vNewBuffer[96] = {0};
        char hBuffer[96] = {0};
        char outBuffer[96] = {0};
        FormatInputShape(context, FINALIZE_INPUT_QG_SCALED, qgBuffer, sizeof(qgBuffer));
        FormatInputShape(context, FINALIZE_INPUT_AQK, aqkBuffer, sizeof(aqkBuffer));
        FormatInputShape(context, FINALIZE_INPUT_V_NEW, vNewBuffer, sizeof(vNewBuffer));
        FormatInputShape(context, FINALIZE_INPUT_H, hBuffer, sizeof(hBuffer));
        FormatOutputShape(context, 0, outBuffer, sizeof(outBuffer));
        OP_LOGE(context->GetNodeName(),
                "输入/输出必须匹配 head-major BF16、K=V=128、Aqk 末维 64 与 output_layout。"
                "当前 layout=%s, qg_scaled=%s, Aqk=%s, v_new=%s, h=%s, attn_out=%s。",
                layout, qgBuffer, aqkBuffer, vNewBuffer, hBuffer, outBuffer);
        return ge::GRAPH_FAILED;
    }
    const auto *cuDesc = context->GetOptionalInputDesc(
        FINALIZE_INPUT_CU_SEQLENS);
    const auto *indexDesc = context->GetOptionalInputDesc(
        FINALIZE_INPUT_CHUNK_INDICES);
    if ((cuDesc != nullptr && cuDesc->GetDataType() != ge::DT_INT64) ||
        (indexDesc != nullptr && indexDesc->GetDataType() != ge::DT_INT64)) {
        OP_LOGE(context->GetNodeName(),
                "cu_seqlens/chunk_indices 仅支持 INT64 host 元数据。");
        return ge::GRAPH_FAILED;
    }

    uint64_t seqNum = 0;
    uint64_t totalChunks = 0;
    bool isVarLen = false;
    if (!ResolveSequenceInfo(context, info, seqNum, totalChunks, isVarLen) ||
        seqNum > std::numeric_limits<uint32_t>::max() ||
        totalChunks > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE(context->GetNodeName(), "变长 metadata 无效或 chunk 数超限。");
        return ge::GRAPH_FAILED;
    }
    const auto &h = context->GetInputShape(FINALIZE_INPUT_H)->GetStorageShape();
    const int64_t hBatch = static_cast<int64_t>(info.batch);
    if (!HasShape(h, {hBatch, static_cast<int64_t>(info.heads),
                      static_cast<int64_t>(totalChunks), 128, 128})) {
        OP_LOGE(context->GetNodeName(),
                "h 必须为 [B,HV,C,128,128]，其中 C=%lu。",
                static_cast<unsigned long>(totalChunks));
        return ge::GRAPH_FAILED;
    }

    const auto platform = platform_ascendc::PlatformAscendC(
        context->GetPlatformInfo());
    ChunkKdaFwdFinalizeScheduleContext scheduleContext;
    scheduleContext.batch = info.batch;
    scheduleContext.valueHeadNum = info.heads;
    scheduleContext.chunksPerSequence = totalChunks;
    scheduleContext.totalVarLenChunks = totalChunks;
    scheduleContext.aicCoreNum = platform.GetCoreNumAic();
    scheduleContext.libApiWorkspaceBytes = platform.GetLibApiWorkSpaceSize();
    scheduleContext.isVarLen = isVarLen;
    ChunkKdaFwdFinalizeSchedule schedule;
    if (!ChunkKdaFwdFinalizeTilingProcessor(scheduleContext).Process(schedule)) {
        OP_LOGE(context->GetNodeName(), "chunk-first/head 分核或 workspace 规划失败。");
        return ge::GRAPH_FAILED;
    }

    ChunkKdaFwdFinalizeTilingData tiling;
    tiling.set_batch(static_cast<uint32_t>(info.batch));
    tiling.set_seqNum(static_cast<uint32_t>(seqNum));
    tiling.set_seqLen(static_cast<uint32_t>(info.seqLen));
    tiling.set_valueHeadNum(static_cast<uint32_t>(info.heads));
    tiling.set_totalChunks(static_cast<uint32_t>(totalChunks));
    tiling.set_usedCoreNum(schedule.usedCoreNum);
    tiling.set_headsPerPartition(schedule.headsPerPartition);
    tiling.set_isVarLen(isVarLen);
    tiling.set_outputSequenceMajor(info.sequenceMajor);
    tiling.set_stateVFirst(*stateVFirst);
    const bool isA5 =
        platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950;
    const uint64_t minimumChunksPerCore =
        schedule.chunkWorkItems / schedule.usedCoreNum;
    const uint64_t moverMinimumChunksPerCore = isVarLen
        ? FINALIZE_AIV_MOVER_VARLEN_MIN_CHUNKS_PER_CORE
        : FINALIZE_AIV_MOVER_DENSE_MIN_CHUNKS_PER_CORE;
    const bool useAivInputMover =
        isA5 && schedule.headsPerPartition == info.heads &&
        minimumChunksPerCore >= moverMinimumChunksPerCore;
    using namespace KdaFinalize;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(
        static_cast<uint64_t>(useAivInputMover ? 1 : 0));
    context->SetTilingKey(tilingKey);
    OP_LOGD(context->GetNodeName(), "tilingKey: %lu, useAivInputMover: %d",
            static_cast<unsigned long>(tilingKey), useAivInputMover);
    context->SetBlockDim(schedule.usedCoreNum);
    if (useAivInputMover &&
        context->SetScheduleMode(FINALIZE_MIX_BATCH_MODE) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(),
                "设置 A5 MIX AIC/AIV batch 调度模式失败。");
        return ge::GRAPH_FAILED;
    }
    size_t *workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr ||
        schedule.workspaceBytes > std::numeric_limits<size_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = static_cast<size_t>(schedule.workspaceBytes);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForChunkKdaFwdFinalize(
    gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkKdaFwdFinalize)
    .Tiling(Tiling4ChunkKdaFwdFinalize)
    .TilingParse<ChunkKdaFwdFinalizeCompileInfo>(
        TilingPrepareForChunkKdaFwdFinalize);

} // namespace optiling
