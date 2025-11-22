
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef UTILS_COMMON_H
#define UTILS_COMMON_H

namespace host_utils {

constexpr uint32_t BLK_SIZE_ALIN_FOR_INT64 = 4;
constexpr uint32_t BLK_SIZE_ALIN_FOR_INT32 = 8;

inline uint64_t alinInt64Count(uint64_t count)
{
    return (count + BLK_SIZE_ALIN_FOR_INT64 - 1) / BLK_SIZE_ALIN_FOR_INT64 * BLK_SIZE_ALIN_FOR_INT64;
}

inline uint64_t alinInt32Count(uint64_t count)
{
    return (count + BLK_SIZE_ALIN_FOR_INT32 - 1) / BLK_SIZE_ALIN_FOR_INT32 * BLK_SIZE_ALIN_FOR_INT32;
}

template <typename T>
inline T CeilDiv(const T dividend, const T divisor)
{
    if (divisor == 0) {
        return UINT32_MAX;
    }
    return (dividend + divisor - 1) / divisor;
}

template <typename T>
inline T RoundUp(const T val, const T align = 16)
{
    if (align == 0 || val + align - 1 < val) {
        return 0;
    }
    return (val + align - 1) / align * align;
}

template <typename T>
inline T RoundDown(const T val, const T align = 16)
{
    if (align == 0) {
        return 0;
    }
    return val / align * align;
}
}  // namespace host_utils
#endif  // UTILS_COMMON_H
