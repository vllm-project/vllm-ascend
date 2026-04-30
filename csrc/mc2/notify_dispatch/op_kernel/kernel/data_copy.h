#ifndef CAM_DATACOPY_GM2GM_H
#define CAM_DATACOPY_GM2GM_H
#include <type_traits>
#include "comm_args.h"

using namespace AscendC;
using namespace Moe;

template <typename T>
FORCE_INLINE_AICORE void SetAtomicOpType(int op)
{
    switch (op) {
        case ADD:
            AscendC::SetAtomicAdd<T>();
            break;
        case MUL:
            // Ignore setting the atomic register when performing mul
            break;
        case MAX:
            AscendC::SetAtomicMax<T>();
            break;
        case MIN:
            AscendC::SetAtomicMin<T>();
            break;
        default:
            AscendC::SetAtomicNone();
    }
}

template <typename T>
FORCE_INLINE_AICORE void CpUB2GM(__gm__ T *gmAddr, __ubuf__ T *ubAddr, uint32_t size)
{
    LocalTensor<uint8_t> ubTensor;
    GlobalTensor<uint8_t> gmTensor;
    DataCopyExtParams dataCopyParams(1, size, 0, 0, 0);
    ubTensor.address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
    ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(ubAddr);
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(gmAddr));
    DataCopyPad(gmTensor, ubTensor, dataCopyParams);
}

template <typename T>
FORCE_INLINE_AICORE void CpGM2UB(__ubuf__ T *ubAddr, __gm__ T *gmAddr, uint32_t size)
{
    LocalTensor<uint8_t> ubTensor;
    GlobalTensor<uint8_t> gmTensor;
    DataCopyExtParams dataCopyParams(1, size, 0, 0, 0);
    ubTensor.address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
    ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(ubAddr);
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(gmAddr));
    DataCopyPadExtParams<uint8_t> padParams;
    DataCopyPad(ubTensor, gmTensor, dataCopyParams, padParams);
}

template<typename T>
FORCE_INLINE_AICORE void CopyUB2UB(__ubuf__ T *dst, __ubuf__ T *src, const uint32_t calCount)
{
    LocalTensor<T> srcTensor;
    LocalTensor<T> dstTensor;
    TBuffAddr srcAddr, dstAddr;
    srcAddr.bufferAddr = reinterpret_cast<uint64_t>(src);
    dstAddr.bufferAddr = reinterpret_cast<uint64_t>(dst);
    srcTensor.SetAddr(srcAddr);
    dstTensor.SetAddr(dstAddr);
    DataCopy(dstTensor, srcTensor, calCount);
}

#endif // CAM_DATACOPY_GM2GM_H