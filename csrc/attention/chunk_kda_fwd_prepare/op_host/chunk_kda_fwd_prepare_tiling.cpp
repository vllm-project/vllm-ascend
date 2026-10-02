/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#include "chunk_kda_fwd_prepare_tiling.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>

#include "chunk_kda_fwd_prepare_tiling_processor.h"
#include "../op_kernel/chunk_kda_fwd_prepare_tiling_key.h"
#include "platform/soc_spec.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
namespace {

constexpr int64_t PREPARE_CHUNK_SIZE = 64;
constexpr int64_t PREPARE_K_DIM = 128;
constexpr int64_t PREPARE_V_DIM = 128;
constexpr uint32_t PREPARE_MIX_BATCH_MODE = 1;

static_assert(PREPARE_OUTPUT_MODE_NONE == CHUNK_KDA_FWD_PREPARE_OUTPUT_NONE);
static_assert(PREPARE_OUTPUT_MODE_RECOMPUTE ==
              CHUNK_KDA_FWD_PREPARE_OUTPUT_RECOMPUTE);
static_assert(PREPARE_OUTPUT_MODE_SAVE == CHUNK_KDA_FWD_PREPARE_OUTPUT_SAVE);
static_assert(PREPARE_OUTPUT_MODE_FORWARD ==
              CHUNK_KDA_FWD_PREPARE_OUTPUT_FORWARD);

enum class PrepareLayout {
    BNSD,
    BSND,
    NTD,
    TND,
};

struct PrepareShapeInfo {
    bool packed = false;
    bool sequenceMajor = false;
    int64_t batch = 0;
    int64_t seqLen = 0;
    int64_t qkHeadNum = 0;
    int64_t valueHeadNum = 0;
    int64_t kDim = 0;
    int64_t vDim = 0;
};

bool ParseLayout(const char *layout, PrepareLayout &parsed)
{
    if (layout == nullptr || std::strcmp(layout, "BNSD") == 0) {
        parsed = PrepareLayout::BNSD;
        return true;
    }
    if (std::strcmp(layout, "BSND") == 0) {
        parsed = PrepareLayout::BSND;
        return true;
    }
    if (std::strcmp(layout, "NTD") == 0) {
        parsed = PrepareLayout::NTD;
        return true;
    }
    if (std::strcmp(layout, "TND") == 0) {
        parsed = PrepareLayout::TND;
        return true;
    }
    return false;
}

bool SameShape(const gert::Shape &lhs, const gert::Shape &rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t dim = 0; dim < lhs.GetDimNum(); ++dim) {
        if (lhs.GetDim(dim) != rhs.GetDim(dim)) {
            return false;
        }
    }
    return true;
}

bool HasShape(const gert::Shape &shape,
              std::initializer_list<int64_t> expected)
{
    if (shape.GetDimNum() != expected.size()) {
        return false;
    }
    size_t dim = 0;
    for (int64_t value : expected) {
        if (shape.GetDim(dim++) != value) {
            return false;
        }
    }
    return true;
}

bool ResolveShape(gert::TilingContext *context, PrepareLayout layout,
                  PrepareShapeInfo &info)
{
    const auto *qShapePtr = context->GetInputShape(PREPARE_INPUT_Q);
    const auto *kShapePtr = context->GetInputShape(PREPARE_INPUT_K);
    const auto *vShapePtr = context->GetInputShape(PREPARE_INPUT_V);
    const auto *gShapePtr = context->GetInputShape(PREPARE_INPUT_G);
    const auto *betaShapePtr = context->GetInputShape(PREPARE_INPUT_BETA);
    if (qShapePtr == nullptr || kShapePtr == nullptr || vShapePtr == nullptr ||
        gShapePtr == nullptr || betaShapePtr == nullptr) {
        return false;
    }

    const auto &q = qShapePtr->GetStorageShape();
    const auto &k = kShapePtr->GetStorageShape();
    const auto &v = vShapePtr->GetStorageShape();
    const auto &g = gShapePtr->GetStorageShape();
    const auto &beta = betaShapePtr->GetStorageShape();
    if (!SameShape(q, k)) {
        return false;
    }

    info.packed = layout == PrepareLayout::NTD || layout == PrepareLayout::TND;
    info.sequenceMajor = layout == PrepareLayout::BSND || layout == PrepareLayout::TND;
    if (info.packed) {
        if (q.GetDimNum() != 3 || v.GetDimNum() != 3 || g.GetDimNum() != 3) {
            return false;
        }
        info.batch = 1;
        if (layout == PrepareLayout::TND) {
            info.seqLen = q.GetDim(0);
            info.qkHeadNum = q.GetDim(1);
            info.valueHeadNum = v.GetDim(1);
            info.kDim = q.GetDim(2);
            info.vDim = v.GetDim(2);
            if (!HasShape(v, {info.seqLen, info.valueHeadNum, info.vDim}) ||
                !HasShape(g, {info.seqLen, info.valueHeadNum, info.kDim})) {
                return false;
            }
        } else {
            info.qkHeadNum = q.GetDim(0);
            info.valueHeadNum = v.GetDim(0);
            info.seqLen = q.GetDim(1);
            info.kDim = q.GetDim(2);
            info.vDim = v.GetDim(2);
            if (!HasShape(v, {info.valueHeadNum, info.seqLen, info.vDim}) ||
                !HasShape(g, {info.valueHeadNum, info.seqLen, info.kDim})) {
                return false;
            }
        }
        if (layout == PrepareLayout::TND) {
            return HasShape(beta, {info.seqLen, info.valueHeadNum});
        }
        return HasShape(beta, {info.valueHeadNum, info.seqLen});
    }

    if (q.GetDimNum() != 4 || v.GetDimNum() != 4 || g.GetDimNum() != 4) {
        return false;
    }
    info.batch = q.GetDim(0);
    if (layout == PrepareLayout::BSND) {
        info.seqLen = q.GetDim(1);
        info.qkHeadNum = q.GetDim(2);
        info.valueHeadNum = v.GetDim(2);
        info.kDim = q.GetDim(3);
        info.vDim = v.GetDim(3);
        if (!HasShape(v, {info.batch, info.seqLen, info.valueHeadNum, info.vDim}) ||
            !HasShape(g, {info.batch, info.seqLen, info.valueHeadNum, info.kDim})) {
            return false;
        }
    } else {
        info.qkHeadNum = q.GetDim(1);
        info.valueHeadNum = v.GetDim(1);
        info.seqLen = q.GetDim(2);
        info.kDim = q.GetDim(3);
        info.vDim = v.GetDim(3);
        if (!HasShape(v, {info.batch, info.valueHeadNum, info.seqLen, info.vDim}) ||
            !HasShape(g, {info.batch, info.valueHeadNum, info.seqLen, info.kDim})) {
            return false;
        }
    }
    if (layout == PrepareLayout::BSND) {
        return HasShape(beta,
                        {info.batch, info.seqLen, info.valueHeadNum});
    }
    return HasShape(beta, {info.batch, info.valueHeadNum, info.seqLen});
}

bool IsGateDtype(ge::DataType dtype)
{
    return dtype == ge::DT_BF16 || dtype == ge::DT_FLOAT;
}

bool DtypeToTemplateToken(ge::DataType dtype, uint64_t &token)
{
    if (dtype == ge::DT_BF16) {
        token = CHUNK_KDA_FWD_PREPARE_TPL_BF16;
        return true;
    }
    if (dtype == ge::DT_FLOAT) {
        token = CHUNK_KDA_FWD_PREPARE_TPL_FP32;
        return true;
    }
    return false;
}

bool FitsUint32(int64_t value)
{
    return value > 0 &&
        static_cast<uint64_t>(value) <= std::numeric_limits<uint32_t>::max();
}

bool FitsDmaSourceStride(int64_t headNum, uint64_t rowBytes)
{
    return headNum > 0 && rowBytes > 0 &&
        static_cast<uint64_t>(headNum - 1) <=
            std::numeric_limits<uint32_t>::max() / rowBytes;
}

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return value / divisor + static_cast<uint64_t>(value % divisor != 0);
}

bool ResolveSequenceInfo(gert::TilingContext *context,
                         const PrepareShapeInfo &shape, uint64_t &seqNum,
                         uint64_t &totalChunks, bool &isVarLen)
{
    const auto *cuTensor =
        context->GetOptionalInputTensor(PREPARE_INPUT_CU_SEQLENS);
    const auto *chunkTensor =
        context->GetOptionalInputTensor(PREPARE_INPUT_CHUNK_INDICES);
    isVarLen = cuTensor != nullptr;
    if (!isVarLen) {
        if (chunkTensor != nullptr) {
            OP_LOGE(context->GetNodeName(),
                    "chunk_indices 只有与 cu_seqlens 同时传入时才合法。");
            return false;
        }
        seqNum = static_cast<uint64_t>(shape.batch);
        totalChunks = CeilDiv(static_cast<uint64_t>(shape.seqLen),
                              PREPARE_CHUNK_SIZE);
        return totalChunks != 0;
    }

    if (!shape.packed && shape.batch != 1) {
        OP_LOGE(context->GetNodeName(),
                "rank-4 变长输入要求 B=1，当前 B=%ld。", shape.batch);
        return false;
    }
    const auto &cuShape = cuTensor->GetStorageShape();
    if (cuShape.GetDimNum() != 1 || cuShape.GetDim(0) < 2) {
        OP_LOGE(context->GetNodeName(),
                "cu_seqlens 必须是一维且至少包含 [0, total_tokens]。");
        return false;
    }
    const int64_t *cu = cuTensor->GetData<int64_t>();
    if (cu == nullptr) {
        OP_LOGE(context->GetNodeName(), "tiling 阶段无法读取 cu_seqlens。");
        return false;
    }
    seqNum = static_cast<uint64_t>(cuShape.GetDim(0) - 1);
    if (cu[0] != 0 || cu[seqNum] != shape.seqLen) {
        OP_LOGE(context->GetNodeName(),
                "cu_seqlens 首元素必须为 0，末元素必须等于 T=%ld。",
                shape.seqLen);
        return false;
    }

    totalChunks = 0;
    for (uint64_t seq = 0; seq < seqNum; ++seq) {
        if (cu[seq] < 0 || cu[seq + 1] < cu[seq] ||
            cu[seq + 1] > shape.seqLen) {
            OP_LOGE(context->GetNodeName(),
                    "cu_seqlens 必须非递减且元素范围为 [0, T]，错误位置=%lu。",
                    static_cast<unsigned long>(seq));
            return false;
        }
        const uint64_t length = static_cast<uint64_t>(cu[seq + 1] - cu[seq]);
        const uint64_t sequenceChunks = CeilDiv(length, PREPARE_CHUNK_SIZE);
        if (totalChunks > std::numeric_limits<uint64_t>::max() - sequenceChunks) {
            return false;
        }
        totalChunks += sequenceChunks;
    }
    if (totalChunks == 0) {
        OP_LOGE(context->GetNodeName(), "变长输入至少需要一个非空 sequence。");
        return false;
    }

    if (chunkTensor == nullptr) {
        return true;
    }
    const auto &chunkShape = chunkTensor->GetStorageShape();
    if (chunkShape.GetDimNum() != 1 ||
        static_cast<uint64_t>(chunkShape.GetDim(0)) != totalChunks * 2) {
        OP_LOGE(context->GetNodeName(),
                "chunk_indices 必须是一维 [2 * total_chunks]，total_chunks=%lu。",
                static_cast<unsigned long>(totalChunks));
        return false;
    }
    const int64_t *indices = chunkTensor->GetData<int64_t>();
    if (indices == nullptr) {
        OP_LOGE(context->GetNodeName(), "tiling 阶段无法读取 chunk_indices。");
        return false;
    }
    uint64_t offset = 0;
    for (uint64_t seq = 0; seq < seqNum; ++seq) {
        const uint64_t length = static_cast<uint64_t>(cu[seq + 1] - cu[seq]);
        const uint64_t chunks = CeilDiv(length, PREPARE_CHUNK_SIZE);
        for (uint64_t chunk = 0; chunk < chunks; ++chunk) {
            if (indices[offset] != static_cast<int64_t>(seq) ||
                indices[offset + 1] != static_cast<int64_t>(chunk)) {
                OP_LOGE(context->GetNodeName(),
                        "chunk_indices 必须按 sequence-major 顺序保存 (seq_id, local_chunk_id)。");
                return false;
            }
            offset += 2;
        }
    }
    return true;
}

bool CheckOptionalInput(gert::TilingContext *context,
                        const PrepareShapeInfo &shape, bool useGateInKernel,
                        bool safeGate, bool &hasDtBias)
{
    const auto *aLogDesc = context->GetOptionalInputDesc(PREPARE_INPUT_A_LOG);
    const auto *dtBiasDesc = context->GetOptionalInputDesc(PREPARE_INPUT_DT_BIAS);
    const auto *aLogShape = context->GetOptionalInputShape(PREPARE_INPUT_A_LOG);
    const auto *dtBiasShape = context->GetOptionalInputShape(PREPARE_INPUT_DT_BIAS);
    if ((aLogDesc == nullptr) != (aLogShape == nullptr) ||
        (dtBiasDesc == nullptr) != (dtBiasShape == nullptr)) {
        OP_LOGE(context->GetNodeName(),
                "a_log/dt_bias 的 descriptor 与 shape 元数据必须成对存在。");
        return false;
    }
    const bool hasALog = aLogDesc != nullptr;
    hasDtBias = dtBiasDesc != nullptr;

    if (!useGateInKernel) {
        if (hasALog || hasDtBias || safeGate) {
            OP_LOGE(context->GetNodeName(),
                    "use_gate_in_kernel=false 时 a_log、dt_bias 必须为空且 safe_gate 必须为 false。");
            return false;
        }
        return true;
    }
    if (!hasALog) {
        OP_LOGE(context->GetNodeName(),
                "use_gate_in_kernel=true 时必须传入 a_log。");
        return false;
    }
    if (aLogDesc->GetDataType() != ge::DT_FLOAT ||
        (dtBiasDesc != nullptr && dtBiasDesc->GetDataType() != ge::DT_FLOAT)) {
        OP_LOGE(context->GetNodeName(), "a_log 和 dt_bias 只支持 FP32。");
        return false;
    }
    const auto &aLog = aLogShape->GetStorageShape();
    if (!HasShape(aLog, {shape.valueHeadNum})) {
        OP_LOGE(context->GetNodeName(), "a_log 必须为 [HV]，当前 HV=%ld。",
                shape.valueHeadNum);
        return false;
    }
    if (hasDtBias) {
        const auto &dtBias = dtBiasShape->GetStorageShape();
        if (!HasShape(dtBias, {shape.valueHeadNum * shape.kDim})) {
            OP_LOGE(context->GetNodeName(),
                    "dt_bias 必须为展平的 [HV*K]，当前 HV=%ld，K=%ld。",
                    shape.valueHeadNum, shape.kDim);
            return false;
        }
    }
    return true;
}

void PrintTiling(gert::TilingContext *context,
                 ChunkKdaFwdPrepareTilingData &tiling,
                 uint64_t outputMode, uint64_t tilingKey,
                 uint64_t workspaceBytes)
{
    OP_LOGD(context->GetNodeName(),
            "ChunkKdaFwdPrepare tiling: B=%u, N=%u, T=%u, HK=%u, HV=%u, "
            "chunks=%u, cores=%u, headsPerPartition=%u, outputMode=%lu, "
            "varlen=%d, sequenceMajor=%d, tilingKey=%lu, "
            "workspace=%lu.",
            tiling.get_batch(), tiling.get_seqNum(), tiling.get_seqLen(),
            tiling.get_qkHeadNum(), tiling.get_valueHeadNum(),
            tiling.get_totalChunks(), tiling.get_usedCoreNum(),
            tiling.get_headsPerPartition(),
            static_cast<unsigned long>(outputMode),
            static_cast<int>(tiling.get_isVarLen()),
            static_cast<int>(tiling.get_inputSequenceMajor()),
            static_cast<unsigned long>(tilingKey),
            static_cast<unsigned long>(workspaceBytes));
}

} // namespace

ge::graphStatus Tiling4ChunkKdaFwdPrepare(gert::TilingContext *context)
{
    const auto *attrs = context->GetAttrs();
    const auto *qDesc = context->GetInputDesc(PREPARE_INPUT_Q);
    const auto *kDesc = context->GetInputDesc(PREPARE_INPUT_K);
    const auto *vDesc = context->GetInputDesc(PREPARE_INPUT_V);
    const auto *gDesc = context->GetInputDesc(PREPARE_INPUT_G);
    const auto *betaDesc = context->GetInputDesc(PREPARE_INPUT_BETA);
    if (attrs == nullptr || qDesc == nullptr || kDesc == nullptr ||
        vDesc == nullptr || gDesc == nullptr || betaDesc == nullptr) {
        OP_LOGE(context->GetNodeName(), "缺少必选输入 descriptor 或 attrs。");
        return ge::GRAPH_FAILED;
    }

    const char *layout = attrs->GetStr(PREPARE_ATTR_LAYOUT);
    const auto *scalePtr = attrs->GetAttrPointer<float>(PREPARE_ATTR_SCALE);
    const auto *chunkSizePtr =
        attrs->GetAttrPointer<int64_t>(PREPARE_ATTR_CHUNK_SIZE);
    const auto *epsilonPtr = attrs->GetAttrPointer<float>(PREPARE_ATTR_EPSILON);
    const auto *useNormPtr =
        attrs->GetAttrPointer<bool>(PREPARE_ATTR_USE_QK_L2NORM);
    const auto *useGatePtr = attrs->GetAttrPointer<bool>(PREPARE_ATTR_USE_GATE);
    const auto *useBetaPtr =
        attrs->GetAttrPointer<bool>(PREPARE_ATTR_USE_BETA_SIGMOID);
    const auto *allowNegPtr =
        attrs->GetAttrPointer<bool>(PREPARE_ATTR_ALLOW_NEG_EIGVAL);
    const auto *safeGatePtr = attrs->GetAttrPointer<bool>(PREPARE_ATTR_SAFE_GATE);
    const auto *lowerBoundPtr =
        attrs->GetAttrPointer<float>(PREPARE_ATTR_LOWER_BOUND);
    const auto *useExp2Ptr = attrs->GetAttrPointer<bool>(PREPARE_ATTR_USE_EXP2);
    if (scalePtr == nullptr || chunkSizePtr == nullptr || epsilonPtr == nullptr ||
        useNormPtr == nullptr || useGatePtr == nullptr || useBetaPtr == nullptr ||
        allowNegPtr == nullptr || safeGatePtr == nullptr || lowerBoundPtr == nullptr ||
        useExp2Ptr == nullptr) {
        OP_LOGE(context->GetNodeName(), "算子属性不完整。");
        return ge::GRAPH_FAILED;
    }

    PrepareLayout parsedLayout;
    if (!ParseLayout(layout, parsedLayout)) {
        OP_LOGE(context->GetNodeName(),
                "layout 必须为 BNSD、BSND、NTD 或 TND，当前值=%s。",
                layout == nullptr ? "<null>" : layout);
        return ge::GRAPH_FAILED;
    }
    if (*chunkSizePtr != PREPARE_CHUNK_SIZE) {
        OP_LOGE(context->GetNodeName(),
                "chunk_size 仅支持 64，当前值=%ld。", *chunkSizePtr);
        return ge::GRAPH_FAILED;
    }
    if (!std::isfinite(*scalePtr) || !std::isfinite(*epsilonPtr) ||
        *epsilonPtr <= 0.0F || !std::isfinite(*lowerBoundPtr)) {
        OP_LOGE(context->GetNodeName(),
                "scale/lower_bound 必须为有限数，epsilon 必须为正有限数。");
        return ge::GRAPH_FAILED;
    }
    if (*allowNegPtr && !*useBetaPtr) {
        OP_LOGE(context->GetNodeName(),
                "allow_neg_eigval=true 要求 use_beta_sigmoid_in_kernel=true。");
        return ge::GRAPH_FAILED;
    }
    if (*safeGatePtr && (*lowerBoundPtr < -5.0F || *lowerBoundPtr >= 0.0F)) {
        OP_LOGE(context->GetNodeName(),
                "safe_gate=true 时 lower_bound 必须在 [-5, 0) 内，当前值=%f。",
                *lowerBoundPtr);
        return ge::GRAPH_FAILED;
    }

    PrepareShapeInfo shape;
    if (!ResolveShape(context, parsedLayout, shape)) {
        OP_LOGE(context->GetNodeName(),
                "q/k/v/g/beta shape 与 layout=%s 不匹配。",
                layout);
        return ge::GRAPH_FAILED;
    }
    const auto *outputModePtr =
        attrs->GetAttrPointer<int64_t>(PREPARE_ATTR_OUTPUT_MODE);
    if (outputModePtr == nullptr || *outputModePtr < PREPARE_OUTPUT_MODE_NONE ||
        *outputModePtr > PREPARE_OUTPUT_MODE_FORWARD) {
        OP_LOGE(context->GetNodeName(),
                "output_mode 只支持 0(none)、1(recompute)、2(save)、3(forward)。");
        return ge::GRAPH_FAILED;
    }
    const uint64_t outputMode = static_cast<uint64_t>(*outputModePtr);
    if (!FitsUint32(shape.batch) || !FitsUint32(shape.seqLen) ||
        !FitsUint32(shape.qkHeadNum) || !FitsUint32(shape.valueHeadNum) ||
        shape.kDim != PREPARE_K_DIM || shape.vDim != PREPARE_V_DIM ||
        shape.valueHeadNum < shape.qkHeadNum ||
        shape.valueHeadNum % shape.qkHeadNum != 0) {
        OP_LOGE(context->GetNodeName(),
                "要求 B/T/HK/HV 为正且可用 uint32 表示、K=V=128、HV>=HK 且 HV%%HK=0；"
                "当前 B=%ld,T=%ld,HK=%ld,HV=%ld,K=%ld,V=%ld。",
                shape.batch, shape.seqLen, shape.qkHeadNum, shape.valueHeadNum,
                shape.kDim, shape.vDim);
        return ge::GRAPH_FAILED;
    }
    if (qDesc->GetDataType() != ge::DT_BF16 ||
        kDesc->GetDataType() != ge::DT_BF16 ||
        vDesc->GetDataType() != ge::DT_BF16 ||
        !IsGateDtype(gDesc->GetDataType()) ||
        !IsGateDtype(betaDesc->GetDataType())) {
        OP_LOGE(context->GetNodeName(),
                "q/k/v 只支持 BF16，g/beta 分别支持 BF16 或 FP32。当前 dtype=%d/%d/%d/%d/%d。",
                static_cast<int>(qDesc->GetDataType()),
                static_cast<int>(kDesc->GetDataType()),
                static_cast<int>(vDesc->GetDataType()),
                static_cast<int>(gDesc->GetDataType()),
                static_cast<int>(betaDesc->GetDataType()));
        return ge::GRAPH_FAILED;
    }
    if (shape.sequenceMajor) {
        const uint64_t gateElementBytes =
            gDesc->GetDataType() == ge::DT_FLOAT ? sizeof(float) : 2U;
        const uint64_t betaElementBytes =
            betaDesc->GetDataType() == ge::DT_FLOAT ? sizeof(float) : 2U;
        const bool strideValid =
            FitsDmaSourceStride(shape.qkHeadNum,
                                PREPARE_K_DIM * 2U) &&
            FitsDmaSourceStride(shape.valueHeadNum,
                                PREPARE_V_DIM * 2U) &&
            FitsDmaSourceStride(shape.valueHeadNum,
                                PREPARE_K_DIM * gateElementBytes) &&
            FitsDmaSourceStride(shape.valueHeadNum, betaElementBytes);
        if (!strideValid) {
            OP_LOGE(context->GetNodeName(),
                    "sequence-major 输入的跨 head DMA stride 超过 uint32 范围，"
                    "当前 HK=%ld,HV=%ld。",
                    shape.qkHeadNum, shape.valueHeadNum);
            return ge::GRAPH_FAILED;
        }
    }

    bool hasDtBias = false;
    if (!CheckOptionalInput(context, shape, *useGatePtr, *safeGatePtr,
                            hasDtBias)) {
        return ge::GRAPH_FAILED;
    }

    uint64_t seqNum = 0;
    uint64_t totalChunks = 0;
    bool isVarLen = false;
    if (!ResolveSequenceInfo(context, shape, seqNum, totalChunks, isVarLen) ||
        seqNum > std::numeric_limits<uint32_t>::max() ||
        totalChunks > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE(context->GetNodeName(),
                "sequence/chunk 元数据无效或数量超过 uint32 范围。");
        return ge::GRAPH_FAILED;
    }

    const auto platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint64_t aicCoreNum = platform.GetCoreNumAic();
    const bool isAscend950 =
        platform.GetCurNpuArch() == NpuArch::DAV_3510;
    ChunkKdaFwdPrepareScheduleContext scheduleContext;
    scheduleContext.batch = static_cast<uint64_t>(shape.batch);
    scheduleContext.qkHeadNum = static_cast<uint64_t>(shape.qkHeadNum);
    scheduleContext.valueHeadNum = static_cast<uint64_t>(shape.valueHeadNum);
    scheduleContext.chunksPerSequence = totalChunks;
    scheduleContext.totalVarLenChunks = totalChunks;
    scheduleContext.aicCoreNum = aicCoreNum;
    scheduleContext.libApiWorkspaceBytes = platform.GetLibApiWorkSpaceSize();
    scheduleContext.isVarLen = isVarLen;
    scheduleContext.workspaceSlotBytes =
        isAscend950 ? CHUNK_KDA_FWD_PREPARE_ARCH35_SLOT_BYTES
                    : CHUNK_KDA_FWD_PREPARE_ARCH22_SLOT_BYTES;
    ChunkKdaFwdPrepareSchedule schedule;
    if (!ChunkKdaFwdPrepareTilingProcessor(scheduleContext).Process(schedule)) {
        OP_LOGE(context->GetNodeName(),
                "无法生成 chunk-first/GVA 完整头组调度，AIC 核数=%lu。",
                static_cast<unsigned long>(aicCoreNum));
        return ge::GRAPH_FAILED;
    }

    uint64_t gateDtypeToken = 0;
    uint64_t betaDtypeToken = 0;
    if (!DtypeToTemplateToken(gDesc->GetDataType(), gateDtypeToken) ||
        !DtypeToTemplateToken(betaDesc->GetDataType(), betaDtypeToken)) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t normMode = *useNormPtr
        ? CHUNK_KDA_FWD_PREPARE_NORM_L2
        : CHUNK_KDA_FWD_PREPARE_NORM_IDENTITY;
    const uint64_t betaMode = !*useBetaPtr
        ? CHUNK_KDA_FWD_PREPARE_BETA_RAW
        : (*allowNegPtr ? CHUNK_KDA_FWD_PREPARE_BETA_TWO_SIGMOID
                        : CHUNK_KDA_FWD_PREPARE_BETA_SIGMOID);
    const uint64_t gateMode = !*useGatePtr
        ? CHUNK_KDA_FWD_PREPARE_GATE_PRECOMPUTED_STEP
        : (*safeGatePtr ? CHUNK_KDA_FWD_PREPARE_GATE_SAFE_SIGMOID
                        : CHUNK_KDA_FWD_PREPARE_GATE_SOFTPLUS);
    using namespace KdaPrepare;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(
        gateDtypeToken, betaDtypeToken, normMode, betaMode, gateMode,
        static_cast<uint64_t>(*useExp2Ptr),
        static_cast<uint64_t>(*safeGatePtr), outputMode);

    ChunkKdaFwdPrepareTilingData tiling;
    tiling.set_batch(static_cast<uint32_t>(shape.batch));
    tiling.set_seqNum(static_cast<uint32_t>(seqNum));
    tiling.set_seqLen(static_cast<uint32_t>(shape.seqLen));
    tiling.set_qkHeadNum(static_cast<uint32_t>(shape.qkHeadNum));
    tiling.set_valueHeadNum(static_cast<uint32_t>(shape.valueHeadNum));
    tiling.set_totalChunks(static_cast<uint32_t>(totalChunks));
    tiling.set_usedCoreNum(schedule.usedCoreNum);
    tiling.set_headsPerPartition(schedule.headsPerPartition);
    tiling.set_epsilon(*epsilonPtr);
    tiling.set_lowerBound(*lowerBoundPtr);
    tiling.set_scale(*scalePtr);
    tiling.set_isVarLen(isVarLen);
    tiling.set_inputSequenceMajor(shape.sequenceMajor);
    tiling.set_hasDtBias(hasDtBias);

    context->SetTilingKey(tilingKey);
    context->SetBlockDim(schedule.usedCoreNum);
    if (context->SetScheduleMode(PREPARE_MIX_BATCH_MODE) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(),
                "设置 MIX AIC/AIV batch 调度模式失败。");
        return ge::GRAPH_FAILED;
    }
    size_t *workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr ||
        schedule.workspaceBytes > std::numeric_limits<size_t>::max()) {
        OP_LOGE(context->GetNodeName(), "workspace 大小超过平台 size_t 范围。");
        return ge::GRAPH_FAILED;
    }
    workspace[0] = static_cast<size_t>(schedule.workspaceBytes);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    PrintTiling(context, tiling, outputMode, tilingKey,
                schedule.workspaceBytes);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForChunkKdaFwdPrepare(
    gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkKdaFwdPrepare)
    .Tiling(Tiling4ChunkKdaFwdPrepare)
    .TilingParse<ChunkKdaFwdPrepareCompileInfo>(
        TilingPrepareForChunkKdaFwdPrepare);

} // namespace optiling
