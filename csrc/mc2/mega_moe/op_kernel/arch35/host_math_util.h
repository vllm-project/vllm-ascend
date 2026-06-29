#ifndef MEGA_MOE_HOST_MATH_UTIL_H
#define MEGA_MOE_HOST_MATH_UTIL_H

namespace Ops { namespace Base {

template <typename T>
static constexpr T CeilAlign(T num, T align) {
    return ((num + align - 1) / align) * align;
}
template <typename T>
static constexpr T CeilDiv(T num, T div) {
    return (num + div - 1) / div;
}

} }  // namespace Ops::Base

namespace ops {
using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;

}

#endif