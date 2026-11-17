/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0.
 */
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>

#include "error/ops_error.h"
#include "register/op_def_registry.h"
#include "tiling/hccl/hccl_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/moe_token_cut_tiling.h"

namespace {

constexpr uint32_t MAX_BATCH_SIZE = 80;
constexpr uint32_t MAX_RANK_NUM = 384;
constexpr uint32_t MAX_EXPERT_NUM = 512;
constexpr uint32_t CONTROL_LOCAL_BYTES =
    MAX_BATCH_SIZE * (sizeof(int8_t) + sizeof(int8_t) + sizeof(int32_t));
constexpr uint32_t MAX_LOCAL_UB_BYTES = 192U * 1024U;
// Kernel restore ping-pong 从 60KB 开始使用固定 UB 地址，TPipe 管理的
// buffer 必须全部位于该地址之前，避免两套分配方式互相覆盖。
constexpr uint32_t RESTORE_UB_HEAD_OFFSET = 68U * 1024U;
// 与 kernel 保持一致：每个通信 slot 按 64B cache line 对齐。
constexpr uint32_t ALIGN_BYTES = 64;
constexpr uint64_t MB_BYTES = 1024UL * 1024UL;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;
constexpr size_t MAX_GROUP_NAME_LENGTH = 128;

enum AttrIndex : uint32_t {
    ATTR_GROUP_EP = 0,
    ATTR_TOPK,
    ATTR_RANK_ID,
    ATTR_LAYER_IDX,
    ATTR_RANK_NUM,
    ATTR_EXPERT_NUM,
    ATTR_EXPERT_PER_RANK,
    ATTR_LEVEL_LIMIT,
    ATTR_CUT_COUNT_THRESHOLD,
    ATTR_SMALL_EXPERT_TOKEN_THRESHOLD,
    ATTR_SEND_AIV_NUM,
    ATTR_RECV_AIV_NUM
};

uint64_t GetWindowBytes()
{
    uint64_t windowMb = 200;
    const char *value = std::getenv("HCCL_BUFFSIZE");
    if (value != nullptr) {
        try {
            windowMb = static_cast<uint64_t>(std::stoul(value));
        } catch (...) {
            windowMb = 200;
        }
    }
    return windowMb * MB_BYTES;
}

uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return (value + align - 1U) / align * align;
}

bool IsShape(const gert::StorageShape *shape, int64_t dim0)
{
    return shape != nullptr && shape->GetStorageShape().GetDimNum() == 1 &&
        shape->GetStorageShape().GetDim(0) == dim0;
}

bool IsShape(const gert::StorageShape *shape, int64_t dim0, int64_t dim1)
{
    return shape != nullptr && shape->GetStorageShape().GetDimNum() == 2 &&
        shape->GetStorageShape().GetDim(0) == dim0 &&
        shape->GetStorageShape().GetDim(1) == dim1;
}

}  // namespace

namespace optiling {

static ge::graphStatus MoeTokenCutTiling(gert::TilingContext *context)
{
    // host 侧一次性完成参数、固定写回 Tensor、通信 window 和 workspace 校验，
    // 避免把动态边界判断留到每个 AIV。
    const char *nodeName = context->GetNodeName();
    MoeTokenCutTilingData *tiling = context->GetTilingData<MoeTokenCutTilingData>();
    OPS_CHECK(tiling == nullptr, OPS_LOG_E(nodeName, "tiling data is null."), return ge::GRAPH_FAILED);
    const auto *attrs = context->GetAttrs();
    OPS_CHECK(attrs == nullptr, OPS_LOG_E(nodeName, "attrs is null."), return ge::GRAPH_FAILED);

    const char *groupEp = attrs->GetAttrPointer<char>(ATTR_GROUP_EP);
    const int64_t *topK = attrs->GetAttrPointer<int64_t>(ATTR_TOPK);
    const int64_t *rankId = attrs->GetAttrPointer<int64_t>(ATTR_RANK_ID);
    const int64_t *layerIdx = attrs->GetAttrPointer<int64_t>(ATTR_LAYER_IDX);
    const int64_t *rankNum = attrs->GetAttrPointer<int64_t>(ATTR_RANK_NUM);
    const int64_t *expertNum = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_NUM);
    const int64_t *expertPerRank = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_PER_RANK);
    const int64_t *levelLimit = attrs->GetAttrPointer<int64_t>(ATTR_LEVEL_LIMIT);
    const int64_t *cutThreshold = attrs->GetAttrPointer<int64_t>(ATTR_CUT_COUNT_THRESHOLD);
    const int64_t *smallExpertTokenThreshold =
        attrs->GetAttrPointer<int64_t>(ATTR_SMALL_EXPERT_TOKEN_THRESHOLD);
    const int64_t *sendAivNum = attrs->GetAttrPointer<int64_t>(ATTR_SEND_AIV_NUM);
    const int64_t *recvAivNum = attrs->GetAttrPointer<int64_t>(ATTR_RECV_AIV_NUM);

    OPS_CHECK(groupEp == nullptr || strnlen(groupEp, MAX_GROUP_NAME_LENGTH) == 0 ||
        strnlen(groupEp, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH,
        OPS_LOG_E(nodeName, "group_ep is invalid."), return ge::GRAPH_FAILED);
    OPS_CHECK(topK == nullptr || rankId == nullptr || rankNum == nullptr ||
        layerIdx == nullptr || expertNum == nullptr || expertPerRank == nullptr || levelLimit == nullptr ||
        cutThreshold == nullptr || smallExpertTokenThreshold == nullptr ||
        sendAivNum == nullptr || recvAivNum == nullptr,
        OPS_LOG_E(nodeName, "one or more attrs are null."), return ge::GRAPH_FAILED);

    OPS_CHECK(*topK <= 0 || *topK > *expertNum,
        OPS_LOG_E(nodeName, "topk must be in [1, expert_num]."), return ge::GRAPH_FAILED);
    OPS_CHECK(*rankNum <= 0 || *rankNum > MAX_RANK_NUM || *rankId < 0 || *rankId >= *rankNum,
        OPS_LOG_E(nodeName, "rank_id/rank_num is invalid."), return ge::GRAPH_FAILED);
    OPS_CHECK(*layerIdx < 0 || *layerIdx >= INT32_MAX,
        OPS_LOG_E(nodeName, "layer_idx must be in [0, %d).", INT32_MAX),
        return ge::GRAPH_FAILED);
    OPS_CHECK(*expertNum <= 0 || *expertNum > MAX_EXPERT_NUM || *expertPerRank <= 0 ||
        *expertNum != *expertPerRank * *rankNum,
        OPS_LOG_E(nodeName, "expert_num must equal expert_per_rank * rank_num."), return ge::GRAPH_FAILED);
    OPS_CHECK(*levelLimit < 0 || *cutThreshold < 0 || *smallExpertTokenThreshold < 0,
        OPS_LOG_E(nodeName, "level/count attrs must be non-negative."), return ge::GRAPH_FAILED);

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = platform.GetCoreNumAiv();
    OPS_CHECK(*sendAivNum <= 0 || *recvAivNum <= 0 ||
        static_cast<uint64_t>(*sendAivNum + *recvAivNum) > aivNum,
        OPS_LOG_E(nodeName, "send_aiv_num + recv_aiv_num must be <= %u.", aivNum),
        return ge::GRAPH_FAILED);

    const gert::StorageShape *topkIds = context->GetInputShape(0);
    const gert::StorageShape *topkWeights = context->GetInputShape(1);
    const gert::StorageShape *hidden = context->GetInputShape(2);
    const gert::StorageShape *cutIdx = context->GetInputShape(3);
    const gert::StorageShape *newTopk = context->GetInputShape(4);
    const gert::StorageShape *newTopkWeight = context->GetInputShape(5);
    const gert::StorageShape *newHidden = context->GetInputShape(6);
    const gert::StorageShape *newResiduals = context->GetInputShape(7);
    const gert::StorageShape *actualBs = context->GetInputShape(8);
    const gert::StorageShape *prefixIndex = context->GetInputShape(9);
    const gert::StorageShape *noCutRankNum = context->GetInputShape(10);
    const gert::StorageShape *activeMask = context->GetInputShape(11);
    const gert::StorageShape *localCutCount = context->GetInputShape(12);
    const gert::StorageShape *batchSize = context->GetInputShape(13);
    const gert::StorageShape *enableCut = context->GetInputShape(14);
    const gert::StorageShape *residuals = context->GetInputShape(15);
    const gert::StorageShape *discardLabel = context->GetInputShape(16);
    const gert::StorageShape *discardTokens = context->GetInputShape(17);
    const gert::StorageShape *discardMoeReqId = context->GetInputShape(18);
    const gert::StorageShape *reqIdList = context->GetInputShape(19);
    const gert::StorageShape *discardResiduals = context->GetInputShape(20);
    const gert::StorageShape *discardTopk = context->GetInputShape(21);
    const gert::StorageShape *discardTopkWeight = context->GetInputShape(22);
    const gert::StorageShape *moeCnt = context->GetInputShape(23);
    const gert::StorageShape *historyBs = context->GetInputShape(24);
    const gert::StorageShape *originalBs = context->GetInputShape(25);

    // OPS_CHECK(!IsShape(topkIds, MAX_BATCH_SIZE, *topK) || !IsShape(newTopk, MAX_BATCH_SIZE, *topK) ||
    //     !IsShape(cutIdx, MAX_BATCH_SIZE) || !IsShape(actualBs, 1) ||
    //     !IsShape(prefixIndex, MAX_BATCH_SIZE) || !IsShape(noCutRankNum, 1) ||
    //     !IsShape(activeMask, MAX_BATCH_SIZE) ||
    //     !IsShape(localCutCount, 1) || !IsShape(batchSize, 1) ||
    //     !IsShape(enableCut, 1),
    //     OPS_LOG_E(nodeName, "fixed writeback tensor shape is invalid."), return ge::GRAPH_FAILED);
    // OPS_CHECK(topkWeights != nullptr && !IsShape(topkWeights, MAX_BATCH_SIZE, *topK),
    //     OPS_LOG_E(nodeName, "topk_weights shape must be [64, topk]."), return ge::GRAPH_FAILED);
    // OPS_CHECK(!IsShape(newTopkWeight, MAX_BATCH_SIZE, *topK),
    //     OPS_LOG_E(nodeName, "new_topkweight shape must be [64, topk]."), return ge::GRAPH_FAILED);
    const int64_t hiddenDim = hidden->GetStorageShape().GetDim(1);

    const auto *idsDesc = context->GetInputDesc(0);
    const auto *weightDesc = context->GetInputDesc(1);
    const auto *hiddenDesc = context->GetInputDesc(2);
    const auto *cutIdxDesc = context->GetInputDesc(3);
    const auto *newTopkDesc = context->GetInputDesc(4);
    const auto *newWeightDesc = context->GetInputDesc(5);
    const auto *newHiddenDesc = context->GetInputDesc(6);
    const auto *newResidualsDesc = context->GetInputDesc(7);
    const auto *actualBsDesc = context->GetInputDesc(8);
    const auto *prefixIndexDesc = context->GetInputDesc(9);
    const auto *noCutRankNumDesc = context->GetInputDesc(10);
    const auto *activeMaskDesc = context->GetInputDesc(11);
    const auto *localCutCountDesc = context->GetInputDesc(12);
    const auto *batchSizeDesc = context->GetInputDesc(13);
    const auto *enableCutDesc = context->GetInputDesc(14);
    const auto *residualsDesc = context->GetInputDesc(15);
    const auto *discardLabelDesc = context->GetInputDesc(16);
    const auto *discardTokensDesc = context->GetInputDesc(17);
    const auto *discardMoeReqIdDesc = context->GetInputDesc(18);
    const auto *reqIdListDesc = context->GetInputDesc(19);
    const auto *discardResidualsDesc = context->GetInputDesc(20);
    const auto *discardTopkDesc = context->GetInputDesc(21);
    const auto *discardTopkWeightDesc = context->GetInputDesc(22);
    const auto *moeCntDesc = context->GetInputDesc(23);
    const auto *historyBsDesc = context->GetInputDesc(24);
    const auto *originalBsDesc = context->GetInputDesc(25);
    OPS_CHECK(idsDesc == nullptr || weightDesc == nullptr || hiddenDesc == nullptr ||
        cutIdxDesc == nullptr ||
        newTopkDesc == nullptr || newWeightDesc == nullptr ||
        newHiddenDesc == nullptr || newResidualsDesc == nullptr ||
        actualBsDesc == nullptr || prefixIndexDesc == nullptr ||
        noCutRankNumDesc == nullptr || activeMaskDesc == nullptr ||
        localCutCountDesc == nullptr || batchSizeDesc == nullptr ||
        enableCutDesc == nullptr,
        OPS_LOG_E(nodeName, "one or more input descriptors are null."), return ge::GRAPH_FAILED);
    OPS_CHECK(idsDesc->GetDataType() != ge::DT_INT32 ||
        newTopkDesc->GetDataType() != ge::DT_INT32 ||
        actualBsDesc->GetDataType() != ge::DT_INT32 ||
        prefixIndexDesc->GetDataType() != ge::DT_INT32 ||
        noCutRankNumDesc->GetDataType() != ge::DT_INT32 ||
        localCutCountDesc->GetDataType() != ge::DT_INT32 ||
        batchSizeDesc->GetDataType() != ge::DT_INT32 ||
        enableCutDesc->GetDataType() != ge::DT_INT32 ||
        cutIdxDesc->GetDataType() != ge::DT_INT8 ||
        activeMaskDesc->GetDataType() != ge::DT_BOOL,
        OPS_LOG_E(nodeName, "ID/index/count tensors have invalid dtypes."),
        return ge::GRAPH_FAILED);
    OPS_CHECK(weightDesc != nullptr && weightDesc->GetDataType() != newWeightDesc->GetDataType(),
        OPS_LOG_E(nodeName, "topk weight input/writeback dtypes must match."), return ge::GRAPH_FAILED);

    uint32_t weightTypeSize = 2;
    if (newWeightDesc->GetDataType() == ge::DT_FLOAT) {
        weightTypeSize = 4;
    }

    // 与 kernel 的三段 TBuf 布局一致：
    // local counts 和 token indices 跨阶段保留，temp buffer 按阶段峰值复用。
    const uint32_t countsBytes =
        AlignUp(static_cast<uint32_t>(*expertNum) * sizeof(int32_t), ALIGN_BYTES);
    // const uint32_t indicesBytes = AlignUp(
    //     static_cast<uint32_t>(*expertNum) * MAX_BATCH_SIZE * sizeof(int32_t),
    //     ALIGN_BYTES);
    const uint32_t rankBytes =
        AlignUp(static_cast<uint32_t>(*rankNum) * sizeof(int32_t), ALIGN_BYTES);
    // const uint32_t topkElements = MAX_BATCH_SIZE * static_cast<uint32_t>(*topK);
    // const uint32_t topkElementsAligned = AlignUp(topkElements, 64U);
    // const uint32_t topkBytes = topkElementsAligned * sizeof(int32_t);
    // const uint32_t compareMaskBytes =
    //     AlignUp((topkElementsAligned + 7U) / 8U, ALIGN_BYTES);
    // // 每张卡的连续接收区由 P 个 source slot 组成；方案二要求每个 AIV
    // // 一次搬入完整接收区，因此该区域同时计入 temp UB 峰值。
    // const uint32_t slotStride = countsBytes + ALIGN_BYTES;
    // const uint64_t receiveRegionBytes =
    //     static_cast<uint64_t>(*rankNum) * slotStride;
    // const uint32_t buildTempBytes = 2U * topkBytes + compareMaskBytes;
    // const uint64_t decisionTempBytes = receiveRegionBytes +
    //     3U * countsBytes + CONTROL_LOCAL_BYTES + rankBytes;
    // const uint32_t compressTempBytes =
    //     AlignUp(static_cast<uint32_t>(*topK) * sizeof(int32_t), ALIGN_BYTES) +
    //     AlignUp(static_cast<uint32_t>(*topK) * weightTypeSize, ALIGN_BYTES) +
    //     AlignUp(static_cast<uint32_t>(hiddenDim) * weightTypeSize, ALIGN_BYTES) +
    //     AlignUp(static_cast<uint32_t>(hiddenDim) * weightTypeSize, ALIGN_BYTES);
    // const uint32_t communicationTempBytes = countsBytes + ALIGN_BYTES;
    // const uint64_t decisionAndCompressBytes =
    //     AlignUp(static_cast<uint32_t>(decisionTempBytes), ALIGN_BYTES) +
    //     compressTempBytes;
    // const uint64_t tempBytes = std::max<uint64_t>(buildTempBytes,
    //     std::max<uint64_t>(decisionAndCompressBytes, communicationTempBytes));
    // const uint64_t restoreCtrlBytes =
    //     AlignUp(MAX_BATCH_SIZE * sizeof(int32_t), ALIGN_BYTES);
    // const uint64_t requiredLocalBytes =
    //     countsBytes + indicesBytes + tempBytes + restoreCtrlBytes;
    // OPS_CHECK(requiredLocalBytes > MAX_LOCAL_UB_BYTES,
    //     OPS_LOG_E(nodeName, "local buffer requires %lu bytes, limit is %u.",
    //         requiredLocalBytes, MAX_LOCAL_UB_BYTES),
    //     return ge::GRAPH_FAILED);
    // OPS_CHECK(requiredLocalBytes > RESTORE_UB_HEAD_OFFSET,
    //     OPS_LOG_E(nodeName,
    //         "TPipe buffers require %lu bytes and overlap restore UB starting at %u.",
    //         requiredLocalBytes, RESTORE_UB_HEAD_OFFSET),
    //     return ge::GRAPH_FAILED);

    // 每个 source rank 独占一个 64B 对齐的 [E] count 向量以及一个状态
    // cache line；状态区前两个 int32 分别保存 flag 和 layerIdx + 1 magic。
    //const uint64_t requiredWindowBytes = receiveRegionBytes;
    const uint64_t windowBytes = GetWindowBytes();
    // OPS_CHECK(requiredWindowBytes > windowBytes,
    //     OPS_LOG_E(nodeName, "HCCL window requires %lu bytes, only %lu available.",
    //         requiredWindowBytes, windowBytes), return ge::GRAPH_FAILED);

    // counts/token index 由所有 AIV 在各自 UB 中独立生成，无需用户 workspace。
    const uint64_t workspaceIntCount = 0U;
    const uint64_t userWorkspaceBytes = AlignUp(
        static_cast<uint32_t>(workspaceIntCount * sizeof(int32_t)), ALIGN_BYTES);
    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    OPS_CHECK(workspaceSizes == nullptr, OPS_LOG_E(nodeName, "workspace size is null."),
        return ge::GRAPH_FAILED);
    workspaceSizes[0] = platform.GetLibApiWorkSpaceSize() + userWorkspaceBytes;

    MoeTokenCutInfo &info = tiling->info;
    info.topK = static_cast<uint32_t>(*topK);
    info.rankId = static_cast<uint32_t>(*rankId);
    info.layerIdx = static_cast<uint32_t>(*layerIdx);
    info.rankNum = static_cast<uint32_t>(*rankNum);
    info.expertNum = static_cast<uint32_t>(*expertNum);
    info.expertPerRank = static_cast<uint32_t>(*expertPerRank);
    info.levelLimit = static_cast<uint32_t>(*levelLimit);
    info.cutCountThreshold = static_cast<uint32_t>(*cutThreshold);
    info.smallExpertTokenThreshold = static_cast<uint32_t>(*smallExpertTokenThreshold);
    info.aivNum = aivNum;
    info.sendAivNum = static_cast<uint32_t>(*sendAivNum);
    info.recvAivNum = static_cast<uint32_t>(*recvAivNum);
    info.weightTypeSize = weightTypeSize;
    info.hiddenDim = static_cast<uint32_t>(hiddenDim);
    info.windowBytes = windowBytes;
    info.userWorkspaceBytes = userWorkspaceBytes;

    AscendC::Mc2CcTilingConfig mc2Config(
        std::string(groupEp), OP_TYPE_ALL_TO_ALL,
        "AlltoAll=level0:fullmesh;level1:pairwise");
    mc2Config.GetTiling(tiling->mc2InitTiling);
    mc2Config.GetTiling(tiling->mc2CcTiling);

    context->SetBlockDim(platform.CalcTschBlockDim(aivNum, 0, aivNum));
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

struct MoeTokenCutCompileInfo {};

static ge::graphStatus ParseMoeTokenCut(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeTokenCut)
    .Tiling(MoeTokenCutTiling)
    .TilingParse<MoeTokenCutCompileInfo>(ParseMoeTokenCut);

}  // namespace optiling
