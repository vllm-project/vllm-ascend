#ifndef SYNC_UTIL_HPP
#define SYNC_UTIL_HPP


#include "kernel_operator.h"
#include "pointer_array_manager.hpp"
#include "const_args.hpp"
using namespace AscendC;

#define FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__

template<typename T>
FORCE_INLINE_AICORE void gm_store(__gm__ T *addr, T val) {
    *((__gm__ T *)addr) = val;
}

template<typename T>
FORCE_INLINE_AICORE T gm_load(__gm__ T *cache) {
    return *((__gm__ T *)cache);
}

FORCE_INLINE_AICORE void gm_dcci(__gm__ uint8_t * addr) {
    using namespace AscendC;
    GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(addr);

    // Important: add hint to avoid dcci being optimized by compiler
    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(global);
    __asm__ __volatile__("");
}

FORCE_INLINE_AICORE int32_t gm_signal_wait_until_eq_for_barrier(__gm__ int32_t *sig_addr, int32_t cmp_val) {
    do {
        gm_dcci((__gm__ uint8_t *)sig_addr);

        if (*sig_addr == cmp_val) {
            return *sig_addr;
        }

        // in case when peer pe enters next barrier
        if (*sig_addr == cmp_val + 1) {
            return *sig_addr;
        }
    } while (true);

    // never reach
    return -1;
}

FORCE_INLINE_AICORE void CrossRankSync(GM_ADDR* symmetricPtr, int32_t rank, int32_t rankSize, uint64_t buffSize=200)
{
    uint64_t flag_offset_mb = buffSize - 1;
    uint64_t flag_offset = flag_offset_mb * MB_SIZE / sizeof(int32_t);

    __gm__ int32_t* sync_counter = (__gm__ int32_t*)symmetricPtr[rank] + flag_offset;
    __gm__ int32_t* sync_base = (__gm__ int32_t*)symmetricPtr[rank] + flag_offset + 1024;
    int count = gm_load(sync_base) + 1;
    int vec_id = AscendC::GetBlockIdx();
    int vec_size = AscendC::GetBlockNum() * AscendC::GetTaskRation();
    for(int i = vec_id; i < rankSize; i += vec_size) {
        __gm__ int32_t* sync_remote = (__gm__ int32_t*)(symmetricPtr[i]) + flag_offset + rank * 16;
        gm_store(sync_remote, count);
        gm_dcci((__gm__ uint8_t*)sync_remote);
        auto sync_check = sync_counter + i * 16;
        gm_signal_wait_until_eq_for_barrier(sync_check, count);
    }

    AscendC::SyncAll<true>();
    gm_store(sync_base, count);
}
#endif // SYNC_UTIL_HPP