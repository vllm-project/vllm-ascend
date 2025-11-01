#ifndef POINTER_ARRAY_MANAGER_HPP
#define POINTER_ARRAY_MANAGER_HPP

#include "kernel_operator.h"

#define CATLASS_HOST_DEVICE __forceinline__ [host, aicore]

class PointerArrayManater{
private:
    GM_ADDR* m_ptrArray;
    size_t m_segmentSize;
    int32_t m_rank;
    int32_t m_rankSize;


public:
    CATLASS_HOST_DEVICE
    PointerArrayManater(){}

    CATLASS_HOST_DEVICE
    PointerArrayManater(GM_ADDR* pointers, int32_t rankSize,  int32_t rank, size_t buffSize)
        : m_ptrArray(pointers), m_rankSize(rankSize), m_rank(rank), m_segmentSize(buffSize)
    {
    }

    CATLASS_HOST_DEVICE
    GM_ADDR getSymmetricPtr(int idx) {
        return m_ptrArray[idx];
    }

    CATLASS_HOST_DEVICE
    GM_ADDR shmem_ptr(GM_ADDR ptr, int32_t index) const  {
        int foundIndex = -1;
        for (int i = 0; i < m_rankSize; i++) {
            GM_ADDR start = m_ptrArray[i];
            GM_ADDR end = start + m_segmentSize;
            if (ptr >= start && ptr < end) {
                foundIndex = i;
                break;
            }
        }
        if (foundIndex == -1) {
            return nullptr;
        }

        size_t offset = ptr - m_ptrArray[foundIndex];

        if (index < 0 || index >= m_rankSize) {
            return nullptr;
        }

        return m_ptrArray[index] + offset;
    }

    CATLASS_HOST_DEVICE
    ~PointerArrayManater() {
    }
};
#endif // POINTER_ARRAY_MANAGER_HPP