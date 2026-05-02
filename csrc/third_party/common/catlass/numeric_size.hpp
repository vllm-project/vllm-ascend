#ifndef CATLASS_NUMERIC_SIZE_HPP
#define CATLASS_NUMERIC_SIZE_HPP

#include "catlass/catlass.hpp"

namespace Catlass {

#if defined(__CCE__)
using AscendC::SizeOfBits;
#else
template <typename T>
struct SizeOfBits {
    static constexpr size_t value = sizeof(T) * 8;
};
#endif

/// Returns the number of bytes required to hold a specified number of bits
template <typename ReturnType = size_t, typename T>
CATLASS_HOST_DEVICE
constexpr ReturnType BitsToBytes(T bits) 
{
    return (static_cast<ReturnType>(bits) + static_cast<ReturnType>(7)) / static_cast<ReturnType>(8);
}

/// Returns the number of bits required to hold a specified number of bytes
template <typename ReturnType = size_t, typename T>
CATLASS_HOST_DEVICE
constexpr ReturnType BytesToBits(T bytes) 
{
    return static_cast<ReturnType>(bytes) * static_cast<ReturnType>(8);
}

} // namespace Catlass

#endif // CATLASS_NUMERIC_SIZE_HPP
