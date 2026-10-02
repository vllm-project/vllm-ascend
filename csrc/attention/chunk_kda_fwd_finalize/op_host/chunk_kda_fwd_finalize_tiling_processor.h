#ifndef CHUNK_KDA_FWD_FINALIZE_TILING_PROCESSOR_H
#define CHUNK_KDA_FWD_FINALIZE_TILING_PROCESSOR_H

#include <algorithm>
#include <cstdint>
#include <limits>

namespace optiling {

constexpr uint64_t FINALIZE_CHUNK_ROWS = 64;

struct ChunkKdaFwdFinalizeScheduleContext {
    uint64_t batch = 0;
    uint64_t valueHeadNum = 0;
    uint64_t chunksPerSequence = 0;
    uint64_t totalVarLenChunks = 0;
    uint64_t aicCoreNum = 0;
    uint64_t libApiWorkspaceBytes = 0;
    bool isVarLen = false;
};

struct ChunkKdaFwdFinalizeSchedule {
    uint32_t usedCoreNum = 0;
    uint32_t headsPerPartition = 0;
    uint64_t chunkWorkItems = 0;
    uint64_t totalWorkItems = 0;
    uint64_t workspaceBytes = 0;
};

class ChunkKdaFwdFinalizeTilingProcessor {
public:
    explicit ChunkKdaFwdFinalizeTilingProcessor(
        const ChunkKdaFwdFinalizeScheduleContext &context) : context_(context)
    {}

    bool Process(ChunkKdaFwdFinalizeSchedule &schedule) const
    {
        if (context_.batch == 0 || context_.valueHeadNum == 0 ||
            context_.aicCoreNum == 0 ||
            context_.valueHeadNum > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        uint64_t chunkItems = context_.totalVarLenChunks;
        if (!context_.isVarLen &&
            !CheckedMultiply(context_.batch, context_.chunksPerSequence,
                             chunkItems)) {
            return false;
        }
        if (chunkItems == 0 ||
            chunkItems > std::numeric_limits<uint32_t>::max()) {
            return false;
        }

        const uint64_t headsPerPartition =
            chunkItems < context_.aicCoreNum ? 1 : context_.valueHeadNum;
        const uint64_t partitions = context_.valueHeadNum / headsPerPartition;
        uint64_t workItems = 0;
        if (!CheckedMultiply(chunkItems, partitions, workItems) ||
            workItems > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        const uint64_t cores = std::min(context_.aicCoreNum, workItems);
        if (cores > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        schedule.usedCoreNum = static_cast<uint32_t>(cores);
        schedule.headsPerPartition = static_cast<uint32_t>(headsPerPartition);
        schedule.chunkWorkItems = chunkItems;
        schedule.totalWorkItems = workItems;
        schedule.workspaceBytes = context_.libApiWorkspaceBytes;
        return true;
    }

private:
    static bool CheckedMultiply(uint64_t lhs, uint64_t rhs, uint64_t &out)
    {
        if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
            return false;
        }
        out = lhs * rhs;
        return true;
    }

    const ChunkKdaFwdFinalizeScheduleContext &context_;
};

} // namespace optiling

#endif // CHUNK_KDA_FWD_FINALIZE_TILING_PROCESSOR_H
