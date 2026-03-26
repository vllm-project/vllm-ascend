#ifndef SYNC_UTIL_HPP
#define SYNC_UTIL_HPP


#include "kernel_operator.h"
#include "const_args.hpp"

#ifdef HCCL_COMM
#include "moe_distribute_base.h"
using namespace AscendC::HcclContextDef;

#else
#include "shmem_api.h"
#endif

#define FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__
constexpr int32_t MAX_RANK_SIZE = 32;
constexpr int32_t SHMEM_MEM = 700 * MB_SIZE;

constexpr uint16_t SEND_SYNC_EVENT_ID = 9;
constexpr uint16_t RECV_SYNC_EVENT_ID = 10;

constexpr uint32_t SELF_STATE_OFFSET = 256 * 1024;
constexpr uint32_t STATE_OFFSET = 512;

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

template<typename T>
FORCE_INLINE_AICORE void gm_dcci(__gm__ T * addr) {
    using namespace AscendC;
    GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(reinterpret_cast<GM_ADDR>(addr));

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
        ub.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
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
    #ifdef HCCL_COMM    // HCCL needs to initialize the HCCL context
        __gm__ HcclOpResParamCustom *WinContext_{nullptr};
        Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
        AscendC::LocalTensor<int32_t> ub;
        FORCE_INLINE_AICORE
        HcclShmem(){
            auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
            WinContext_ = (__gm__ HcclOpResParamCustom *)contextGM0;

            m_rank = WinContext_->localUsrRankId;
            m_rankSize = WinContext_->rankSize;
            m_segmentSize = WinContext_->winSize;
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
    GM_ADDR operator() () const {   // No parameters: return pointer to local peermem
        #ifdef HCCL_COMM
            return (GM_ADDR)(WinContext_->localWindowsIn);
        #else
            return reinterpret_cast<GM_ADDR>(shmem_ptr(symmetricPtr, m_rank));
        #endif
    }

    FORCE_INLINE_AICORE
    GM_ADDR operator() (int32_t index) const {  // With index parameter: return pointer to the base address of remote peermem
        #ifdef HCCL_COMM
            return (GM_ADDR)((index == m_rank) ? WinContext_->localWindowsIn :
                                    ((HcclRankRelationResV2Custom *)(WinContext_->remoteRes[index].nextDevicePtr))->windowsIn);
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
            return (GM_ADDR)((rankId == m_rank) ? WinContext_->localWindowsIn :
                                    ((HcclRankRelationResV2Custom *)(WinContext_->remoteRes[rankId].nextDevicePtr))->windowsIn) + offset;
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
    void InitStatusTargetSum()
    {
        using namespace AscendC;
        uint64_t flag_offset = (m_segmentSize - MB_SIZE) + SELF_STATE_OFFSET;
        //uint64_t self_state_offset = (m_segmentSize - 2 * MB_SIZE);
        // ep state
        //uint32_t coreIdx = get_block_idx();;
        uint32_t coreIdx = GetBlockIdx();
        GlobalTensor<int32_t> selfStatusTensor;
        selfStatusTensor.SetGlobalBuffer((__gm__ int32_t *)((*this)() + flag_offset));
        __asm__ __volatile__("");
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfStatusTensor[coreIdx * UB_ALIGN]);
        __asm__ __volatile__("");
        int32_t state = selfStatusTensor(coreIdx * UB_ALIGN);
        if (state == 0) {
            sumTarget_ = static_cast<float>(1.0);
            selfStatusTensor(coreIdx * UB_ALIGN) = 0x3F800000;  // 1.0f
            epStateValue_ = 0x3F800000;                          // 1.0f
        } else {
            sumTarget_ = static_cast<float>(0.0);
            selfStatusTensor(coreIdx * UB_ALIGN) = 0;
            epStateValue_ = 0;
        }
        __asm__ __volatile__("");
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfStatusTensor[coreIdx * UB_ALIGN]);
        __asm__ __volatile__("");
    }

    FORCE_INLINE_AICORE
    void CrossRankSyncV2Set(AscendC::LocalTensor<int32_t> ctrBuffer) {
        //subblockid = 0
        uint32_t stateOffset_ =  STATE_OFFSET;
        // uint32_t epStateOffsetOnWin_ = m_rank * stateOffset_;
        
        uint64_t flag_offset = (m_segmentSize - MB_SIZE) + m_rank * stateOffset_;
        //uint64_t flag_offset = (m_segmentSize - MB_SIZE);
        int vec_size = get_block_num();
        int vec_id = get_block_idx();
 
        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(RECV_SYNC_EVENT_ID);
        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(SEND_SYNC_EVENT_ID);
        AscendC::CrossCoreWaitFlag(SEND_SYNC_EVENT_ID);
        pipe_barrier(PIPE_ALL);
 
        ctrBuffer.SetValue(0, epStateValue_);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        for (uint32_t dstEpIdx = vec_id; dstEpIdx < m_rankSize; dstEpIdx += vec_size) {
            AscendC::GlobalTensor<int32_t> gmDstStates;
            gmDstStates.SetGlobalBuffer((__gm__ int32_t*)((*this)(flag_offset, dstEpIdx)));
            DataCopy(gmDstStates, ctrBuffer, 8);
        }
        AscendC::CrossCoreWaitFlag(RECV_SYNC_EVENT_ID);
    }

    FORCE_INLINE_AICORE
    void CrossRankSyncV2Wait(AscendC::LocalTensor<float> statusTensor, AscendC::LocalTensor<float> gatherMaskOutTensor,
        AscendC::LocalTensor<uint32_t> gatherTmpTensor, AscendC::LocalTensor<float> statusSumOutTensor) {

        uint64_t flag_offset = (m_segmentSize - MB_SIZE);
        int vec_size = get_block_num();
        int vec_id = get_block_idx();
        uint32_t stateOffset_ =  STATE_OFFSET;

        uint32_t sendRankNum_ = m_rankSize / vec_size;
        uint32_t remainderRankNum = m_rankSize % vec_size;
        uint32_t startRankId_ = sendRankNum_ * vec_id;
        if (vec_id < remainderRankNum) {
            sendRankNum_++;
            startRankId_ += vec_id;
        } else {
            startRankId_ += remainderRankNum;
        }
        uint32_t endRankId_ = startRankId_ + sendRankNum_;
        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(SEND_SYNC_EVENT_ID);

        AscendC::GlobalTensor<float> epStatusSpaceGlobalTensor_;
        epStatusSpaceGlobalTensor_.SetGlobalBuffer((__gm__ float *)((*this)() + flag_offset));

        if (startRankId_ < m_rankSize) {
            AscendC::PipeBarrier<PIPE_ALL>();
            gatherTmpTensor.SetValue(0, 1);
            uint32_t mask = 1;  // gatherMask + sum
            uint64_t rsvdCnt = 0;
            // DataCopyParams intriParams{static_cast<uint16_t>(sendRankNum_), 1,
            //                         static_cast<uint16_t>((moeSendNum_ > 512) ? 7 : 15), 0}; 
            AscendC::DataCopyParams intriParams{static_cast<uint16_t>(sendRankNum_), 1,
                                    static_cast<uint16_t>(15), 0}; 

            float sumOfFlag = static_cast<float>(-1.0);
            float minTarget = (sumTarget_ * sendRankNum_) - (float)0.5;
            float maxTarget = (sumTarget_ * sendRankNum_) + (float)0.5;
            AscendC::SumParams sumParams{1, sendRankNum_, sendRankNum_};

            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);

            while ((sumOfFlag < minTarget) || (sumOfFlag > maxTarget)) {
                AscendC::DataCopy<float>(statusTensor, epStatusSpaceGlobalTensor_[startRankId_ * stateOffset_ / sizeof(float)],
                                intriParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

                GatherMask(gatherMaskOutTensor, statusTensor, gatherTmpTensor, true, mask,
                        {1, (uint16_t)sendRankNum_, 1, 0}, rsvdCnt);

                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Sum(statusSumOutTensor, gatherMaskOutTensor, sumParams);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                sumOfFlag = statusSumOutTensor.GetValue(0);
            }
        }

        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(RECV_SYNC_EVENT_ID);
        AscendC::CrossCoreWaitFlag(RECV_SYNC_EVENT_ID);

        //unpermute
        AscendC::CrossCoreWaitFlag(SEND_SYNC_EVENT_ID);
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
    float sumTarget_{0.0};
    int32_t epStateValue_;
};




#endif
