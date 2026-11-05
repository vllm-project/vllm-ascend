#ifndef KERNEL_MATH_UTILS_H
#define KERNEL_MATH_UTILS_H
#include <cstdint>

namespace device_utils {

template <typename T, T roundVal>
__aicore__ __force_inline__ T RoundUp(const T &val)
{
    return (val + roundVal - 1) / roundVal * roundVal;
}

};  // namespace device_utils

#endif
