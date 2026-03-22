#ifndef SYNC_UTIL_HPP
#define SYNC_UTIL_HPP


#include "kernel_operator.h"
#include "const_args.hpp"

#include "moe_distribute_base.h"

#ifndef HCCL_COMM
#include "shmem_api.h"
using namespace AscendC::HcclContextDef;
#endif

#define FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__
constexpr int32_t MAX_RANK_SIZE = 32;
constexpr int32_t SHMEM_MEM = 1024 * MB_SIZE;

FORCE_INLINE_AICORE void AicSyncAll() {
    AscendC::CrossCoreSetFlag<0x0, PIPE_FIX>(8);
    AscendC::CrossCoreWaitFlag<0x0>(8);
}

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
        if (*sig_addr == cmp_val + 1) {
            return *sig_addr;
        }
    } while (true);
    return -1;
}

FORCE_INLINE_AICORE void gm_signal_wait_until_ne(__gm__ int32_t *sig_addr, int32_t cmp_val) {
    do {
        AscendC::LocalTensor<int32_t> ub;
        ub.address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
        ub.address_.bufferAddr = 0;
        AscendC::GlobalTensor<int32_t> sig;
        sig.SetGlobalBuffer(sig_addr);
        AscendC::DataCopy(ub, sig, 8);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        if (ub(0) != cmp_val) {
            return;
        }
    } while (true);
    return;
}


class HcclShmem {
public:
    #ifdef HCCL_COMM
        __gm__ HcclOpResParamCustom *WinContext_{nullptr};
        Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
        GM_ADDR m_ptrArray[MAX_RANK_SIZE];
        FORCE_INLINE_AICORE
        HcclShmem(){
            auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
            WinContext_ = (__gm__ HcclOpResParamCustom *)contextGM0;

            m_rank = WinContext_->localUsrRankId;
            m_rankSize = WinContext_->rankSize;
            m_segmentSize = WinContext_->winSize;
            for (int i = 0; i < m_rankSize; i++) {
                m_ptrArray[i] = (GM_ADDR)((i == m_rank) ? WinContext_->localWindowsIn :
                                    ((HcclRankRelationResV2Custom *)(WinContext_->remoteRes[i].nextDevicePtr))->windowsIn);
            }
        }
    #else
        FORCE_INLINE_AICORE
        HcclShmem(){
            m_segmentSize = SHMEM_MEM;
        }
        FORCE_INLINE_AICORE
        void initShmem(GM_ADDR symmetricPtr_, size_t rank, size_t rankSize) {
            symmetricPtr = symmetricPtr_;
            m_rank = rank;
            m_rankSize = rankSize;
        }
    #endif

    FORCE_INLINE_AICORE
    GM_ADDR operator() () const {
        #ifdef HCCL_COMM
            return m_ptrArray[m_rank];
        #else
            return reinterpret_cast<GM_ADDR>(shmem_ptr(symmetricPtr, m_rank));
        #endif
    }

    FORCE_INLINE_AICORE
    GM_ADDR operator() (int32_t index) const {
        #ifdef HCCL_COMM
            return m_ptrArray[index];
        #else
            return reinterpret_cast<GM_ADDR>(shmem_ptr(symmetricPtr, index));
        #endif
    }

    FORCE_INLINE_AICORE
    GM_ADDR operator () (int64_t offset, int32_t rankId) const  {
        #ifdef HCCL_COMM
            if (offset < 0 || offset >= m_segmentSize) {
                return nullptr;
            }
            if (rankId < 0 || rankId >= m_rankSize) {
                return nullptr;
            }
            return m_ptrArray[rankId] + offset;
        #else
            return reinterpret_cast<GM_ADDR>(shmem_ptr((symmetricPtr + offset), rankId));
        #endif
    }



    FORCE_INLINE_AICORE
    size_t SegmentSize() const {
        return m_segmentSize;
    }

    FORCE_INLINE_AICORE
    int32_t RankSize() const {
        return m_rankSize;
    }


    FORCE_INLINE_AICORE
    ~HcclShmem() {
    }


    FORCE_INLINE_AICORE
    void CrossRankSync() {
        uint64_t flag_offset = (m_segmentSize - MB_SIZE) / sizeof(int32_t);
        __gm__ int32_t* sync_counter = (__gm__ int32_t*)(*this)() + flag_offset;
        __gm__ int32_t* sync_base = (__gm__ int32_t*)(*this)() + flag_offset + 2048;
        int count = gm_load(sync_base) + 1;
        int vec_id = AscendC::GetBlockIdx();
        int vec_size = AscendC::GetBlockNum() * AscendC::GetTaskRation();
        for(int i = vec_id; i < m_rankSize; i += vec_size) {
            __gm__ int32_t* sync_remote = (__gm__ int32_t*)((*this)(i)) + flag_offset + m_rank * 16;
            gm_store(sync_remote, count);
            gm_dcci((__gm__ uint8_t*)sync_remote);
            auto sync_check = sync_counter + i * 16;
            gm_signal_wait_until_eq_for_barrier(sync_check, count);
        }

        AscendC::SyncAll<true>();
        gm_store(sync_base, count);
    }

    FORCE_INLINE_AICORE
    __gm__ int32_t* SyncBaseAddr() {
        uint64_t flag_offset = (m_segmentSize - MB_SIZE) / sizeof(int32_t);
        return (__gm__ int32_t*)(*this)() + flag_offset + 2048;
    }

private:
    GM_ADDR symmetricPtr;
    int32_t m_rank;
    int32_t m_rankSize;
    size_t m_segmentSize;
};




#endif