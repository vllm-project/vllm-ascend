/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef CHUNK_KDA_FWD_PREPARE_TILING_PROCESSOR_H
#define CHUNK_KDA_FWD_PREPARE_TILING_PROCESSOR_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace optiling {

constexpr uint64_t CHUNK_KDA_FWD_PREPARE_CHUNK_ROWS = 64;
constexpr uint64_t CHUNK_KDA_FWD_PREPARE_ARCH35_SLOT_BYTES = 0x1A400;
constexpr uint64_t CHUNK_KDA_FWD_PREPARE_ARCH22_SLOT_BYTES = 0x1F400;
constexpr uint64_t CHUNK_KDA_FWD_PREPARE_SLOTS_PER_WORKGROUP = 4;
constexpr uint64_t CHUNK_KDA_FWD_PREPARE_ARCH35_WORKGROUP_BYTES =
    CHUNK_KDA_FWD_PREPARE_ARCH35_SLOT_BYTES *
    CHUNK_KDA_FWD_PREPARE_SLOTS_PER_WORKGROUP;
constexpr uint64_t CHUNK_KDA_FWD_PREPARE_ARCH22_WORKGROUP_BYTES =
    CHUNK_KDA_FWD_PREPARE_ARCH22_SLOT_BYTES *
    CHUNK_KDA_FWD_PREPARE_SLOTS_PER_WORKGROUP;

struct ChunkKdaFwdPrepareScheduleContext {
    uint64_t batch = 0;
    uint64_t qkHeadNum = 0;
    uint64_t valueHeadNum = 0;
    uint64_t chunksPerSequence = 0;
    uint64_t totalVarLenChunks = 0;
    uint64_t aicCoreNum = 0;
    uint64_t libApiWorkspaceBytes = 0;
    bool isVarLen = false;
    // 调用方未显式识别平台时按较大的 Arch22 slot 保守分配。
    uint64_t workspaceSlotBytes = CHUNK_KDA_FWD_PREPARE_ARCH22_SLOT_BYTES;
};

struct ChunkKdaFwdPrepareSchedule {
    uint32_t usedCoreNum = 0;
    uint32_t headsPerPartition = 0;
    uint64_t chunkWorkItems = 0;
    uint64_t totalWorkItems = 0;
    uint64_t workspaceBytes = 0;
};

class ChunkKdaFwdPrepareTilingProcessor {
public:
    explicit ChunkKdaFwdPrepareTilingProcessor(
        const ChunkKdaFwdPrepareScheduleContext &context) : context_(context)
    {}

    bool Process(ChunkKdaFwdPrepareSchedule &schedule) const
    {
        if (context_.batch == 0 || context_.qkHeadNum == 0 ||
            context_.valueHeadNum < context_.qkHeadNum ||
            context_.valueHeadNum % context_.qkHeadNum != 0 ||
            context_.aicCoreNum == 0 ||
            (context_.workspaceSlotBytes !=
                 CHUNK_KDA_FWD_PREPARE_ARCH35_SLOT_BYTES &&
             context_.workspaceSlotBytes !=
                 CHUNK_KDA_FWD_PREPARE_ARCH22_SLOT_BYTES)) {
            return false;
        }

        uint64_t chunkWorkItems = context_.totalVarLenChunks;
        if (!context_.isVarLen) {
            if (!CheckedMultiply(context_.batch, context_.chunksPerSequence,
                                 chunkWorkItems)) {
                return false;
            }
        }
        if (chunkWorkItems == 0) {
            return false;
        }
        if (chunkWorkItems > std::numeric_limits<uint32_t>::max()) {
            return false;
        }

        // 先仅按 chunk 分核。chunk 数不足时，才以完整 GVA QK 头组为单位分 head。
        uint64_t headsPerPartition = context_.valueHeadNum;
        uint64_t partitionCount = 1;
        if (chunkWorkItems < context_.aicCoreNum && context_.qkHeadNum > 1) {
            headsPerPartition = context_.valueHeadNum / context_.qkHeadNum;
            partitionCount = context_.qkHeadNum;
        }

        uint64_t totalWorkItems = 0;
        if (!CheckedMultiply(chunkWorkItems, partitionCount, totalWorkItems)) {
            return false;
        }
        if (totalWorkItems > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        const uint64_t usedCoreNum = std::min(context_.aicCoreNum, totalWorkItems);
        uint64_t workgroupBytes = 0;
        uint64_t userWorkspaceBytes = 0;
        if (!CheckedMultiply(context_.workspaceSlotBytes,
                             CHUNK_KDA_FWD_PREPARE_SLOTS_PER_WORKGROUP,
                             workgroupBytes) ||
            !CheckedMultiply(usedCoreNum, workgroupBytes,
                             userWorkspaceBytes) ||
            context_.libApiWorkspaceBytes >
                std::numeric_limits<uint64_t>::max() - userWorkspaceBytes ||
            usedCoreNum > std::numeric_limits<uint32_t>::max() ||
            headsPerPartition > std::numeric_limits<uint32_t>::max()) {
            return false;
        }

        schedule.usedCoreNum = static_cast<uint32_t>(usedCoreNum);
        schedule.headsPerPartition = static_cast<uint32_t>(headsPerPartition);
        schedule.chunkWorkItems = chunkWorkItems;
        schedule.totalWorkItems = totalWorkItems;
        schedule.workspaceBytes = context_.libApiWorkspaceBytes + userWorkspaceBytes;
        return true;
    }

private:
    static bool CheckedMultiply(uint64_t lhs, uint64_t rhs, uint64_t &result)
    {
        if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
            return false;
        }
        result = lhs * rhs;
        return true;
    }

    const ChunkKdaFwdPrepareScheduleContext &context_;
};

} // namespace optiling

#endif // CHUNK_KDA_FWD_PREPARE_TILING_PROCESSOR_H
