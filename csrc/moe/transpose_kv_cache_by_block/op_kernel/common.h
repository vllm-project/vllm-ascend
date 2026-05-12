#include "kernel_operator.h"
using namespace AscendC;

#ifndef __OP_KERNEL_KV_CACHE_TRANSPOSE_H__
#define __OP_KERNEL_KV_CACHE_TRANSPOSE_H__

template <typename T>
__aicore__ inline __gm__ T* GetTensorAddr(uint16_t index, GM_ADDR tensorPtr) {
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
    // The offset of the data address from the first address.
    uint64_t tensorPtrOffset = *dataAddr;
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(retPtr + index));
}
#endif