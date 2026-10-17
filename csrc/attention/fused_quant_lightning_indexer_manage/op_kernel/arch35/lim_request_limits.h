#ifndef LIM_REQUEST_LIMITS_H
#define LIM_REQUEST_LIMITS_H

#include <cstdint>

// Values shared by host tiling, request validation, and device publishers.
namespace lim_request_limits {
constexpr uint32_t BLOCK_SIZE = 128U;
constexpr uint32_t TOPK = 2048U;
constexpr uint32_t MAX_ROUTES = 14U;
constexpr uint32_t MISS_CAPACITY = 32768U;

constexpr int32_t IDENTITY_STATE = -3;
constexpr int32_t FIRST_FILL_STATE = -2;
constexpr int32_t STEADY_STATE = -1;
} // namespace lim_request_limits

#endif // LIM_REQUEST_LIMITS_H
