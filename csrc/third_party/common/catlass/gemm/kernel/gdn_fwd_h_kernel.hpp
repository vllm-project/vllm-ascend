#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/debug.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/block/block_scheduler_gdn_fwd_h.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "kernel_operator.h"
using namespace Catlass;

namespace Catlass::Gemm::Kernel {

template<
    class CubeScheduler, 
    class VecScheduler, 
    class BlockMmadWH,
    class BlockMmadKV,
    class EpilogueGDNFwdHVnew,
    class EpilogueGDNFwdHUpdate
>
class GDNFwdHKernel {
public:
    
    using ArchTag = Arch::AtlasA2;

    using GDNFwdHOffsets = Catlass::Gemm::Block::GDNFwdHOffsets;

    using ElementK = typename BlockMmadKV::ElementA;
    using ElementW = typename BlockMmadWH::ElementA;
    using ElementU = typename BlockMmadKV::ElementB;
    using ElementG = float;
    using ElementH = typename BlockMmadWH::ElementB;
    using ElementV = typename BlockMmadKV::ElementB;
    using ElementVWork = half;
    using ElementHWork = half;
    using ElementInitialState = float;
    using ElementFinalState = float;

    using L1TileShape = typename BlockMmadWH::L1TileShape;
    
    using LayoutW = Catlass::layout::RowMajor;
    using LayoutH = Catlass::layout::RowMajor;
    using LayoutV = Catlass::layout::RowMajor;
    using LayoutK = Catlass::layout::ColumnMajor;

    
    uint32_t batch;
    uint32_t seqlen;
    uint32_t kNumHead;
    uint32_t vNumHead;
    uint32_t kHeadDim;
    uint32_t vHeadDim;
    uint32_t chunkSize;
    bool useInitialState;
    bool storeFinalState;
    uint32_t isVariedLen;
    uint32_t shapeBatch;
    uint32_t tokenBatch;
    uint32_t vWorkspaceOffset;
    uint32_t hWorkspaceOffset;
    
    AscendC::GlobalTensor<ElementK> gmK;
    AscendC::GlobalTensor<ElementW> gmW;
    AscendC::GlobalTensor<ElementU> gmU;
    AscendC::GlobalTensor<ElementG> gmG;
    AscendC::GlobalTensor<ElementInitialState> gmInitialState;
    AscendC::GlobalTensor<ElementH> gmH;
    AscendC::GlobalTensor<ElementV> gmV;
    AscendC::GlobalTensor<ElementFinalState> gmFinalState;
    AscendC::GlobalTensor<ElementV> gmVWorkspace;
    AscendC::GlobalTensor<ElementVWork> gmVWorkspaceHalf;
    AscendC::GlobalTensor<ElementH> gmHWorkspace;
    AscendC::GlobalTensor<ElementHWork> gmHWorkspaceHalf;
    
    AscendC::GlobalTensor<int64_t> gmSeqlen;
    AscendC::GlobalTensor<int64_t> gmNumChunks;

    CubeScheduler cubeBlockScheduler;
    VecScheduler vecBlockScheduler;

    Arch::Resource<ArchTag> resource;


    __aicore__ inline GDNFwdHKernel() {}

    __aicore__ inline void Init(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR inital_state, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, 
        GM_ADDR h, GM_ADDR v_new, GM_ADDR final_state, GM_ADDR tiling, GM_ADDR user) {
        
        __gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict gdnFwdHTilingData = reinterpret_cast<__gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict>(tiling);

        batch = gdnFwdHTilingData->batch;
        seqlen = gdnFwdHTilingData->seqlen;
        kNumHead = gdnFwdHTilingData->kNumHead;
        vNumHead = gdnFwdHTilingData->vNumHead;
        kHeadDim = gdnFwdHTilingData->kHeadDim;
        vHeadDim = gdnFwdHTilingData->vHeadDim;
        chunkSize = gdnFwdHTilingData->chunkSize;
        useInitialState = gdnFwdHTilingData->useInitialState;
        storeFinalState = gdnFwdHTilingData->storeFinalState;
        isVariedLen = gdnFwdHTilingData->isVariedLen;
        shapeBatch = gdnFwdHTilingData->shapeBatch;
        tokenBatch = gdnFwdHTilingData->tokenBatch;
        vWorkspaceOffset = gdnFwdHTilingData->vWorkspaceOffset;
        hWorkspaceOffset = gdnFwdHTilingData->hWorkspaceOffset;
        
        gmK.SetGlobalBuffer((__gm__ ElementK *)k);
        gmW.SetGlobalBuffer((__gm__ ElementW *)w);
        gmU.SetGlobalBuffer((__gm__ ElementU *)u);
        gmG.SetGlobalBuffer((__gm__ ElementG *)g);
        gmInitialState.SetGlobalBuffer((__gm__ ElementInitialState *)inital_state);
        gmH.SetGlobalBuffer((__gm__ ElementH *)h);
        gmV.SetGlobalBuffer((__gm__ ElementV *)v_new);
        gmFinalState.SetGlobalBuffer((__gm__ ElementFinalState *)final_state);
        gmVWorkspace.SetGlobalBuffer((__gm__ ElementV *)(user + vWorkspaceOffset));
        gmVWorkspaceHalf.SetGlobalBuffer((__gm__ ElementVWork *)(user + vWorkspaceOffset));
        gmHWorkspace.SetGlobalBuffer((__gm__ ElementH *)(user + hWorkspaceOffset));
        gmHWorkspaceHalf.SetGlobalBuffer((__gm__ ElementHWork *)(user + hWorkspaceOffset));

        gmSeqlen.SetGlobalBuffer((__gm__ int64_t *)cu_seqlens);
        gmNumChunks.SetGlobalBuffer((__gm__ int64_t *)chunk_indices);

        if ASCEND_IS_AIC {
            cubeBlockScheduler.Init(cu_seqlens, chunk_indices, tiling);
        }

        if ASCEND_IS_AIV {
            vecBlockScheduler.Init(cu_seqlens, chunk_indices, tiling);
        }
    }
    
    __aicore__ inline void Process() {

        if ASCEND_IS_AIC {
            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();

            BlockMmadWH blockMmadWH(resource);
            BlockMmadKV blockMmadKV(resource);

            auto wLayout = tla::MakeLayout<ElementW, LayoutW>(shapeBatch * kNumHead * cubeBlockScheduler.totalTokens, kHeadDim);
            auto hLayout = tla::MakeLayout<ElementH, LayoutH>(shapeBatch * vNumHead * cubeBlockScheduler.totalChunks * kHeadDim, vHeadDim);
            auto vLayout = tla::MakeLayout<ElementVWork, LayoutV>(coreNum * chunkSize * PING_PONG_STAGES, vHeadDim);
            
            auto kLayout = tla::MakeLayout<ElementK, LayoutK>(kHeadDim, shapeBatch * kNumHead * cubeBlockScheduler.totalTokens);
            auto vworkLayout = tla::MakeLayout<ElementVWork, LayoutV>(coreNum * chunkSize * PING_PONG_STAGES, vHeadDim);
            auto hworkLayout = tla::MakeLayout<ElementHWork, LayoutH>(coreNum * kHeadDim * PING_PONG_STAGES, vHeadDim);

            while (cubeBlockScheduler.isRunning) {
                cubeBlockScheduler.InitTask();
                // step 1: v_work = w @ h[i]
                GDNFwdHOffsets& cube1Offsets = cubeBlockScheduler.GetStage1Offsets();
                Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done);
                if (cubeBlockScheduler.NeedProcessStage1()) {
                    int64_t cube1OffsetW = cube1Offsets.wOffset;
                    int64_t cube1OffsetH = cube1Offsets.hSrcOffset;
                    int64_t cube1OffsetVwork = cube1Offsets.vWorkOffset;
                    auto tensorW = tla::MakeTensor(gmW[cube1OffsetW], wLayout, Catlass::Arch::PositionGM{});
                    auto tensorH = tla::MakeTensor(gmH[cube1OffsetH], hLayout, Catlass::Arch::PositionGM{});
                    auto tensorV = tla::MakeTensor(gmVWorkspaceHalf[cube1OffsetVwork], vLayout, Catlass::Arch::PositionGM{});
                    GemmCoord cube1Shape {cube1Offsets.blockTokens, vHeadDim, kHeadDim};
                    auto tensorBlockW = GetTile(tensorW, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.k()));
                    auto tensorBlockH = GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.k(), cube1Shape.n()));
                    auto tensorBlockV = GetTile(tensorV, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.n()));
                    blockMmadWH.preSetFlags();
                    blockMmadWH(tensorBlockW, tensorBlockH, tensorBlockV, cube1Shape);
                    blockMmadWH.finalWaitFlags();
                }
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube1Done);

                if (cubeBlockScheduler.iterId > 1) {
                    Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done);
                    GDNFwdHOffsets& cube2Offsets = cubeBlockScheduler.GetStage2Offsets();
                    if (cubeBlockScheduler.NeedProcessStage2()) {
                        // step 3: h[i+1] = k.T @ v_work
                        int64_t cube2OffsetK = cube2Offsets.wkOffset;
                        int64_t cube2OffsetVwork = cube2Offsets.vWorkOffset;
                        int64_t cube2OffsetH = cube2Offsets.hWorkOffset;
                        auto tensorK = tla::MakeTensor(gmK[cube2OffsetK], kLayout, Catlass::Arch::PositionGM{});
                        auto tensorVwork = tla::MakeTensor(gmVWorkspace[cube2OffsetVwork], vworkLayout, Catlass::Arch::PositionGM{});
                        auto tensorHwork = tla::MakeTensor(gmHWorkspaceHalf[cube2OffsetH], hworkLayout, Catlass::Arch::PositionGM{});
                        GemmCoord cube2Shape{kHeadDim, vHeadDim, cube2Offsets.blockTokens};
                        auto tensorBlockK = GetTile(tensorK, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.k()));
                        auto tensorBlockVwork = GetTile(tensorVwork, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.k(), cube2Shape.n()));
                        auto tensorBlockHwork = GetTile(tensorHwork, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.n()));
                        blockMmadKV.preSetFlags();
                        blockMmadKV(tensorBlockK, tensorBlockVwork, tensorBlockHwork, cube2Shape);
                        blockMmadKV.finalWaitFlags();
                    }
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube2Done);
                }
            }
            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done);

        }

        if ASCEND_IS_AIV {
            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();
            uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
            uint32_t subBlockNum = AscendC::GetSubBlockNum();

            EpilogueGDNFwdHVnew epilogueGDNFwdHVnew(resource);

            if (useInitialState) {
                AscendC::LocalTensor<ElementInitialState> floatUbTensorPing = resource.ubBuf.template GetBufferByByte<float>(0);
                AscendC::LocalTensor<ElementInitialState> floatUbTensorPong = resource.ubBuf.template GetBufferByByte<float>(96 * 1024);
                AscendC::LocalTensor<ElementH> hUbTensorPing = resource.ubBuf.template GetBufferByByte<ElementH>(64 * 1024);
                AscendC::LocalTensor<ElementH> hUbTensorPong = resource.ubBuf.template GetBufferByByte<ElementH>(160 * 1024);
                uint32_t totalChunks = isVariedLen ? gmNumChunks.GetValue(tokenBatch) : ((seqlen + chunkSize - 1) / chunkSize);
                uint32_t stateBlockSize = kHeadDim * vHeadDim;
                uint32_t pingpongFlag = 1;
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
                for (uint32_t shapeBatchIdx = 0; shapeBatchIdx < shapeBatch; shapeBatchIdx++) {
                    for (uint32_t vHeadIdx = 0; vHeadIdx < vNumHead; vHeadIdx++) {
                        for (uint32_t tokenBatchIdx = 0; tokenBatchIdx < tokenBatch; tokenBatchIdx++) {
                            uint32_t batchIdx = isVariedLen ? tokenBatchIdx : shapeBatchIdx;
                            uint32_t chunkOffset = isVariedLen ? gmNumChunks.GetValue(tokenBatchIdx) : 0;
                            uint32_t initialStateOffset = (batchIdx * vNumHead + vHeadIdx) * stateBlockSize;
                            uint32_t hOffset = (shapeBatchIdx * vNumHead * totalChunks + vHeadIdx * totalChunks + chunkOffset) * stateBlockSize;
                            AscendC::LocalTensor<ElementInitialState> floatUbTensor = pingpongFlag ? floatUbTensorPing : floatUbTensorPong;
                            AscendC::LocalTensor<ElementH> hUbTensor = pingpongFlag ? hUbTensorPing : hUbTensorPong;
                            auto event_id = pingpongFlag ? EVENT_ID1 : EVENT_ID0;
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
                            AscendC::DataCopy(floatUbTensor, gmInitialState[initialStateOffset], stateBlockSize);
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id);
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);
                            AscendC::Cast(hUbTensor, floatUbTensor, AscendC::RoundMode::CAST_RINT, stateBlockSize);
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event_id);
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event_id);
                            AscendC::DataCopy(gmH[hOffset], hUbTensor, stateBlockSize);
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
                            pingpongFlag = 1 - pingpongFlag;
                        }
                        
                    }
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            }

            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done);
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done);
            while (vecBlockScheduler.isRunning) {
                vecBlockScheduler.InitTask();
                // step 2:
                GDNFwdHOffsets& vec1Offsets = vecBlockScheduler.GetStage1Offsets();
                // gmV = gmU - gmVWorkspace
                // g_buf = gmG[-1] - gmG
                // g_buf = exp(g_buf)
                // gmVWorkspace = g_buf * gmV
                if (vecBlockScheduler.NeedProcessStage1()) {
                    epilogueGDNFwdHVnew(
                        gmV[vec1Offsets.uvOffset], gmVWorkspace[vec1Offsets.vWorkOffset], 
                        gmG[vec1Offsets.gOffset], gmU[vec1Offsets.uvOffset], gmVWorkspaceHalf[vec1Offsets.vWorkOffset], 
                        vec1Offsets.blockTokens, kHeadDim, vHeadDim, vecBlockScheduler.cube1Done
                    );
                } else {
                    Arch::CrossCoreWaitFlag(vecBlockScheduler.cube1Done);
                }
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec1Done);

                if (vecBlockScheduler.iterId > 1) {
                    GDNFwdHOffsets& vec2Offsets = vecBlockScheduler.GetStage2Offsets();
                    if (vecBlockScheduler.NeedProcessStage2()) {
                        // step 4:  h[i+1] += h_work if i < num_chunks - 1 else None
                        EpilogueGDNFwdHUpdate epilogueGDNFwdHUpdate(resource);
                        epilogueGDNFwdHUpdate(
                            gmH[vec2Offsets.hDstOffset], gmFinalState[vec2Offsets.finalStateOffset],
                            gmG[vec2Offsets.gOffset],
                            gmH[vec2Offsets.hSrcOffset], gmHWorkspaceHalf[vec2Offsets.hWorkOffset],
                            vec2Offsets.blockTokens, kHeadDim, vHeadDim, vecBlockScheduler.cube2Done,
                            (vec2Offsets.isFinalState && storeFinalState)
                        );
                    } else {
                        Arch::CrossCoreWaitFlag(vecBlockScheduler.cube2Done);
                    }
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done);
                }
            }

        }
    }

};

}