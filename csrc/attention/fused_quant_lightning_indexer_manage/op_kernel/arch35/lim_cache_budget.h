#ifndef LIM_CACHE_BUDGET_H
#define LIM_CACHE_BUDGET_H

#include <cstdint>
#include "kernel_operator.h"
#include "lim_payload_layout.h"
#include "lim_request_limits.h"

namespace lim_cache_budget {
constexpr uint32_t TOPK = lim_request_limits::TOPK;
constexpr uint32_t BLOCK = lim_request_limits::BLOCK_SIZE;
// Absolute prefix ceiling (zero tail), also used to size/check UB scratch.
// Each request's effective ceiling is reduced by its block-aligned tail.
constexpr uint32_t MAX_CACHE_TOKENS =
    lim_payload_layout::PENDING_MISS_SLOT / BLOCK * BLOCK;
static_assert(MAX_CACHE_TOKENS < lim_payload_layout::PENDING_MISS_SLOT,
              "prefix slots must stay below reserved payload slots");

__aicore__ inline uint32_t MaxBudget(uint32_t tailLength)
{
    const uint64_t tailSlots =
        (static_cast<uint64_t>(tailLength) + BLOCK - 1U) / BLOCK * BLOCK;
    if (tailSlots > lim_payload_layout::PENDING_MISS_SLOT) return 0U;
    return static_cast<uint32_t>(
        (lim_payload_layout::PENDING_MISS_SLOT - tailSlots) / BLOCK * BLOCK);
}

// A3's request-level C contract, with the current A5 Q range supplied by
// the caller. Tail hits occupy consecutive slots after the prefix budget.
__aicore__ inline bool Valid(
    uint32_t budget, uint32_t prefix, uint32_t routes, uint32_t tailLength)
{
    return routes != 0U && budget != 0U && budget % BLOCK == 0U &&
        budget <= MaxBudget(tailLength) && budget <= prefix &&
        (prefix <= routes * TOPK
            ? budget == prefix : budget >= routes * TOPK);
}
} // namespace lim_cache_budget

#endif
