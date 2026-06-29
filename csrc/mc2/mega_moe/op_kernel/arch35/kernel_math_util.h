#ifndef MEGA_MOE_KERNEL_MATH_UTIL_H
#define MEGA_MOE_KERNEL_MATH_UTIL_H

namespace Ops { namespace Base {

template <typename T>
__aicore__ static constexpr T CeilAlign(T num, T align) {
    return ((num + align - 1) / align) * align;
}
template <typename T>
__aicore__ static constexpr T CeilDiv(T num, T div) {
    return (num + div - 1) / div;
}

} }  // namespace Ops::Base

#endif
