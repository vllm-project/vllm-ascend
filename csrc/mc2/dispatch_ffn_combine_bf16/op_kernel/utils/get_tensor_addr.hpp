#ifndef GET_TENSOR_ADDR_HPP
#define GET_TENSOR_ADDR_HPP
#include "kernel_operator.h"

#define FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__

template <typename T>
FORCE_INLINE_AICORE __gm__ T* GetTensorAddr(uint32_t index, GM_ADDR tensorPtr) {
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;  // The offset of the data address from the first address.
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(retPtr + index));
}

#endif // GET_TENSOR_ADDR_HPP