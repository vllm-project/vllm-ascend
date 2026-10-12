#ifndef LIM_CACHE_BUDGET_H
#define LIM_CACHE_BUDGET_H

#include <cstdint>
#include "kernel_operator.h"
#include "lim_payload_layout.h"
#include "lim_request_limits.h"

namespace lim_cache_budget {
constexpr uint32_t TOPK = lim_request_limits::TOPK;
constexpr uint32_t BLOCK = lim_request_limits::BLOCK_SIZE;
constexpr uint32_t MAX_CACHE_TOKENS = 32384U;
static_assert(MAX_CACHE_TOKENS + 255U < lim_payload_layout::PENDING_MISS_SLOT,
              "the largest tail slot must fit below reserved payload slots");
static_assert(MAX_CACHE_TOKENS + BLOCK + 255U >= lim_payload_layout::PENDING_MISS_SLOT,
              "the cache budget must be the largest safe aligned value");

// A3's request-level C contract, with the current A5 Q range supplied by
// the caller. The full-visible A5 LI and tail-hit behavior are unchanged.
__aicore__ inline bool Valid(uint32_t budget, uint32_t prefix, uint32_t routes)
{
    return routes != 0U && budget != 0U && budget % BLOCK == 0U &&
        budget <= MAX_CACHE_TOKENS && budget <= prefix &&
        (prefix <= routes * TOPK
            ? budget == prefix : budget >= routes * TOPK);
}
} // namespace lim_cache_budget

#endif
