#ifndef LIM_PAYLOAD_LAYOUT_H
#define LIM_PAYLOAD_LAYOUT_H

#include <cstdint>

// A5 payload: slot[31:17] | source_low[16:0]. Long-source high bits travel
// in a separate survivor tag; the 16-bit sortable score is never modified.
namespace lim_payload_layout {
constexpr uint32_t SOURCE_BITS = 17U;
constexpr uint32_t SLOT_BITS = 32U - SOURCE_BITS;
constexpr uint32_t SOURCE_MASK = (1U << SOURCE_BITS) - 1U;
constexpr uint32_t SOURCE_TAG_BITS = 4U;
constexpr uint32_t SOURCE_TAG_MASK = (1U << SOURCE_TAG_BITS) - 1U;
constexpr uint32_t LONG_SOURCE_MASK = (1U << (SOURCE_BITS + SOURCE_TAG_BITS)) - 1U;
constexpr uint32_t SLOT_MASK = (1U << SLOT_BITS) - 1U;
constexpr uint32_t MISS_SLOT = SLOT_MASK;
constexpr uint32_t PENDING_MISS_SLOT = MISS_SLOT - 1U;
constexpr uint32_t PACKED_SOURCE_CAPACITY = 1U << SOURCE_BITS;
constexpr uint32_t MAX_SOURCE_CAPACITY = 1U << (SOURCE_BITS + SOURCE_TAG_BITS);
static_assert(SOURCE_BITS == 17U && SLOT_BITS == 15U,
              "A5 short-source payload must use a 17/15 split");
} // namespace lim_payload_layout

#endif // LIM_PAYLOAD_LAYOUT_H
