// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace vllm_ascend::hisparse {
struct ResolverTilingData {
    uint64_t rows;
    uint64_t topk;
    uint64_t hotSlots;
    uint64_t hostRows;
    uint64_t logicalStride;
    uint64_t stateIndexCount;
    uint64_t tagRows;
    uint64_t sourceRows;
    uint64_t sourceColumns;
    uint64_t sourceStride;
    uint64_t sourceBlockSize;
    uint64_t residentRows;
    uint64_t residentColumns;
    uint64_t residentStride;
    uint64_t residentBlockSize;
    uint64_t residentNullBlock;
    uint64_t hotBlockSize;
    uint64_t hotTableStride;
    uint64_t attentionStride;
    uint64_t attentionBlockStride;
    uint64_t countsStride;
    uint64_t maxUnionRows;
    uint64_t hashSize;
    uint64_t attentionCapacity;
    uint64_t byteMissMask;
    uint32_t ubBytes;
};
}  // namespace vllm_ascend::hisparse
