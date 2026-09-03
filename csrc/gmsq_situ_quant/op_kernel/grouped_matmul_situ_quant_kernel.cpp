/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// AscendC fused kernel for GMM1 (fake-A8W8) + SiTU + per-token INT8 quant.
//
// Production-base fusion (this revision): the GEMM engine is the CANN
// production static-tiling MatmulImpl<A int8 ND, B int8 NZ, C int32 ND> path --
// the same mechanism as production aclnnGroupedMatmulWeightNz fake-A8W8. The
// static tiling is built with GetMatmulApiTiling over the production GenGmmConf
// field set (basicM=128/basicN=256/basicK=128, singleCoreM=128/singleCoreN=1024/
// singleCoreK=8192, enVecND2NZ, INNER_PRODUCT) plus the production 8 static
// tiling overrides (depthA1/B1=8, stepM/N=1, stepKa/Kb=4, dbL0A/B=2, dbL0C=1).
//
// Weight path: the AIV stage expands packed INT4-in-INT32 -> INT8 directly into
// FRACTAL_NZ layout (16 DataCopyPad [16,32] blocks per [32,256] strip), so the
// AIC feeds MatmulImpl a production-compatible NZ B tensor. On a P34 cache hit
// the whole [E,K,N] NZ workspace is reused (skipUnpack=1).
//
// AIC GEMM tiles (expert, M-tile, N-tile) with BM=128 / BN=256, sync
// IterateAll<true>, writing int32 acc [sumMPad, N] into GM before the wave
// handshake (FLAG_WAVE_READY) releases the AIV SiTU+quant epilogue.
//
// All capabilities preserved (regression must pass): FusedMetaCache
// persistent host meta, no per-call H2D / stream sync on host (NPUGraph /
// torch.compile usable), device-side group_list parse (AIC GM-direct read, AIV
// UB parse), ND/NZ weight input (host converts ND->NZ), stacked/TensorList dual
// organization, capacity pruning, group_list_type 0/1, P34 persistent
// expanded-w8 cache (skipUnpack).
//
// Rewritten by mechanism from the CANN Open Software License sources
// (ops_transformer/ascendc/grouped_matmul/*), not copied verbatim.

#include "kernel_operator.h"
#include "adv_api/matmul_intf.h"

using AscendC::DataCopyExtParams;
using AscendC::DataCopyPadExtParams;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::TBuf;
using AscendC::TQue;
using AscendC::TPipe;

namespace {

constexpr int32_t BM = 128;               // GEMM M tile (production baseM)
constexpr int32_t BN = 256;               // GEMM N tile (production baseN / L0B tile)
                                           // (runtime N-block nBlock, multiple of 256, is host-chosen
                                           // for MTE2 B-fetch load balance; default 1024 = singleCoreN)
constexpr int32_t UNPACK_BK = 64;         // unpack K rows per strip
constexpr int32_t UNPACK_SUB_K = 32;      // unpack rows per AIV sub-block
constexpr int32_t FUSED_UNPACK_BN = 256;  // unpack strip width (== BN)
constexpr int32_t FUSED_UNPACK_BNP = 32;  // int32 carriers per strip row
constexpr int32_t ACT_RED_CHUNK = 1024;   // ReduceMax chunk width
constexpr int32_t SCALE_SUB = 256;        // fp32 outputs per de-interleave chunk
constexpr uint16_t FLAG_WAVE_READY = 8;   // AIC -> AIV (PIPE_FIX): wave-batch tiles in GM
constexpr int32_t SYNC_THROTTLE = 14;     // max unconsumed cross-core sets
constexpr int32_t XS_MAX_CHUNK = 2048;    // x_scale floats per on-demand chunk (8 KiB
                                           // UB), independent of capacity C
constexpr int32_t P54_SMALL_M = 64;       // P54 header-opt: mValid <= 64 collapses the
                                           // per-wave cross-core handshake into ONE sync

// ---------------------------------------------------------------------------
// Production MMImplType GEMM types (fake-A8W8: int8 ND x int8 NZ -> half ND).
// C=half enables the MatmulImpl epilogue dequant (int32 L0C * fp32 channel
// scale -> half, VDEQF16 fixpipe) which halves the GM accumulator traffic.
// ---------------------------------------------------------------------------
using AType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int8_t, false>;
using BType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::NZ, int8_t, false>;
using CType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false>;
using BiasType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false>;

// Production GenGmmConf(isND2NZ=true) field set, rebuilt by field assignment
// (C++17-safe; no designated initializers). Every field whose value differs
// from its default member initializer is set explicitly; the rest keep the
// struct defaults, which match the production designated-initializer list.
// MatmulConfig / CubeFormat / BatchMode / IterateMode / IterateOrder /
// ScheduleType are declared at GLOBAL scope in matmul_config.h (no namespace).
__aicore__ inline constexpr MatmulConfig MakeGmmConf()
{
    MatmulConfig conf{};
    conf.doMultiDataLoad = true;
    conf.basicM = 128;
    conf.basicN = 256;
    conf.basicK = 128;
    conf.enVecND2NZ = true;
    conf.singleCoreM = 128;
    conf.singleCoreN = 1024;
    conf.singleCoreK = 8192;
    conf.enUnitFlag = true;
    conf.enableInit = false;
    conf.batchMode = BatchMode::NONE;
    conf.enableEnd = true;
    conf.enableGetTensorC = true;
    conf.enableSetOrgShape = true;
    conf.enableSetBias = false;
    conf.enableSetTail = true;
    conf.enableQuantVector = true;
    conf.enableSetDefineData = false;
    conf.iterateMode = IterateMode::ITERATE_MODE_DEFAULT;
    conf.enableReuse = true;
    conf.enableUBReuse = true;
    conf.iterateOrder = IterateOrder::UNDEF;
    conf.scheduleType = ScheduleType::INNER_PRODUCT;
    conf.isBiasBatch = true;
    return conf;
}

// Production GetGmmMatmulApiTiling(isWeightNZ=true, transB=false) + the 8
// static-tiling overrides applied to the returned MatmulApiStaticTiling.
__aicore__ inline constexpr auto MakeGmmStaticTiling()
{
    auto tiling =
        AscendC::GetMatmulApiTiling<AType, BType, CType, BiasType>(MakeGmmConf());
    tiling.depthA1 = 8;
    tiling.depthB1 = 8;
    tiling.stepM = 1;
    tiling.stepN = 1;
    tiling.stepKa = 4;
    tiling.stepKb = 4;
    tiling.dbL0A = 2;
    tiling.dbL0B = 2;
    // dbL0C must stay 1: the L0C accumulator is STILL int32 (GetMmDstType<int8>),
    // because the int32->half dequant runs in the fixpipe only during the
    // L0C->GM copy-out. baseM(128)*baseN(256)*sizeof(int32) = 128KB already fills
    // the L0C, so dbL0C=2 remains impossible even with a half C output. The C=half
    // win is the halved GM accumulator traffic, not L0C double buffering.
    tiling.dbL0C = 1;
    return tiling;
}
constexpr static auto staticCFG = MakeGmmStaticTiling();

// The production MatmulImpl<A, B, C, Bias, staticCFG> type (MMImplType::MT).
using GmmMt = AscendC::MatmulImpl<AType, BType, CType, BiasType, staticCFG>;

__aicore__ inline float Int32BitsToFloat(int32_t v)
{
    union {
        int32_t i;
        float f;
    } u;
    u.i = v;
    return u.f;
}

__aicore__ inline int32_t AivCoreIdx()
{
    return AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
}

__aicore__ inline int32_t AivSubIdx()
{
    return AscendC::GetBlockIdx() % AscendC::GetSubBlockNum();
}

__aicore__ inline int32_t AivCoreNum()
{
    int32_t n = AscendC::GetBlockNum() / AscendC::GetSubBlockNum();
    return n > 0 ? n : 1;
}

__aicore__ inline int32_t AivSlotIdx()
{
    return AscendC::GetBlockIdx();
}

__aicore__ inline int32_t AivSlotNum()
{
    int32_t n = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
    return n > 0 ? n : 1;
}

// Read a 64-bit GM address stored as two int32 lanes (lo, hi).
__aicore__ inline int64_t ReadGmPtr(const GlobalTensor<int32_t> &tbl, int32_t compactIdx)
{
    uint32_t lo = static_cast<uint32_t>(tbl.GetValue(static_cast<uint64_t>(2 * compactIdx)));
    int32_t hi = tbl.GetValue(static_cast<uint64_t>(2 * compactIdx + 1));
    return (static_cast<int64_t>(hi) << 32) | static_cast<int64_t>(lo);
}

// On-device group_list accessor for the AIC (cube) core. The AIC must NOT touch
// VECCALC UB (scalar SetValue/GetValue on a vector-calc TBuf hangs on the cube
// core), so this variant reads group_list directly from GM on each accessor
// call. E <= 128 -> the per-tile scan is small; the AIV stage does the
// heavyweight on-device parse into local UB.
// Layout follows group_list GM: [E] int64, glType 0 (cumsum) / 1 (count).
class DeviceGlMetaGm {
public:
    static constexpr int32_t MAX_E = 128;     // matches host "E <= 128" assumption
    static constexpr int32_t MAX_MBLK = 256;  // C/128 + E <= 256 (eval max ~21)

    __aicore__ inline DeviceGlMetaGm() {}

    // Per-expert M dynamic split: precompute the whole (expert, padded-row,
    // start-row) mapping ONCE per forward (O(E + M_blocks) GM scalar reads),
    // then resolve each GEMM tile in O(1) local lookups instead of three O(E)
    // GM scalar scans (FindExpert / PadOff / Start) per tile. For decode's 120
    // tiles x ~20 GM reads -> ~2400 GM reads become a single ~35-read pass.
    __aicore__ inline void Init(const GlobalTensor<int64_t> &glGM, int32_t E, int32_t glType)
    {
        E_ = E; glType_ = glType; glGM_ = glGM;
        int64_t prev = 0;
        int32_t sumM = 0;
        int32_t sumP = 0;
        for (int32_t e = 0; e < E_; e++) {
            int64_t gv = glGM_.GetValue(e);
            int32_t cnt = (glType_ == 1) ? static_cast<int32_t>(gv)
                                         : static_cast<int32_t>(gv - prev);
            prev = gv;
            count_[e] = cnt;
            start_[e] = sumM;
            padOff_[e] = sumP;
            sumM += cnt;
            sumP += ((cnt + BM - 1) / BM) * BM;
        }
        padOff_[E_] = sumP;
        sumMPad_ = sumP;
        sumM_ = sumM;
        numMBlocks_ = sumP / BM;
        int32_t e = 0;
        for (int32_t b = 0; b < numMBlocks_; b++) {
            int32_t rowW = b * BM;
            while (e + 1 < E_ && rowW >= padOff_[e + 1]) {
                ++e;
            }
            expertOfBlock_[b] = e;
        }
    }

    __aicore__ inline int32_t Count(int32_t e) const { return count_[e]; }
    __aicore__ inline int32_t Start(int32_t e) const { return start_[e]; }
    __aicore__ inline int32_t PadOff(int32_t e) const { return padOff_[e]; }
    __aicore__ inline int32_t SumMPad() const { return sumMPad_; }
    // P54: real token count (M_valid) for the small-M single-sync decision.
    __aicore__ inline int32_t SumM() const { return sumM_; }
    __aicore__ inline int32_t ExpertOfBlock(int32_t bmG) const { return expertOfBlock_[bmG]; }

private:
    GlobalTensor<int64_t> glGM_;
    int32_t count_[MAX_E];
    int32_t start_[MAX_E];
    int32_t padOff_[MAX_E + 1];
    int32_t expertOfBlock_[MAX_MBLK];
    int32_t E_;
    int32_t glType_;
    int32_t sumMPad_;
    int32_t sumM_;
    int32_t numMBlocks_;
};

// On-device group_list parser + accessor (device-side group_list parsing).
// The AIV stage parses the device group_list tensor into a LOCAL UB meta buffer
// every forward, so the meta is correct for the CURRENT (possibly dynamic)
// production MoE group_list -- no host D2H, no glPtr in any cache key.
// Local layout (E = expert count):
//   [0..E)      mE        : real token count of expert e
//   [E..2E)     startRow  : first real row of expert e inside x / y
//   [2E..3E+1)  mPadOff   : prefix sum of padded rows (mPadOff[E] = sumMPad)
//   [3E+1]      sumMPad
//   [3E+2]      MValid
class DeviceGlMeta {
public:
    __aicore__ inline DeviceGlMeta() {}

    __aicore__ inline void Init(const GlobalTensor<int64_t> &glGM, int32_t E,
                                int32_t glType, TPipe *pipe)
    {
        E_ = E;
        glGM_ = glGM;
        pipe->InitBuffer(metaBuf_, (3 * E + 4) * static_cast<uint32_t>(sizeof(int32_t)));
        Parse(glType);
    }

    __aicore__ inline void Parse(int32_t glType)
    {
        LocalTensor<int32_t> meta = metaBuf_.Get<int32_t>();
        int64_t prev = 0;
        int32_t sumM = 0;
        int32_t sumP = 0;
        for (int32_t e = 0; e < E_; e++) {
            int64_t gv = glGM_.GetValue(e);
            int32_t cnt = (glType == 1) ? static_cast<int32_t>(gv)
                                        : static_cast<int32_t>(gv - prev);
            prev = gv;
            meta.SetValue(e, cnt);           // mE[e]
            meta.SetValue(E_ + e, sumM);     // startRow[e]
            meta.SetValue(2 * E_ + e, sumP); // mPadOff[e]
            int32_t mPad = ((cnt + BM - 1) / BM) * BM;
            sumP += mPad;
            sumM += cnt;
        }
        meta.SetValue(2 * E_ + E_, sumP);    // mPadOff[E]
        meta.SetValue(3 * E_ + 1, sumP);     // sumMPad
        meta.SetValue(3 * E_ + 2, sumM);     // MValid
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline int32_t PadOff(int32_t e)
    {
        return metaBuf_.Get<int32_t>().GetValue(2 * E_ + e);
    }

    __aicore__ inline int32_t Start(int32_t e)
    {
        return metaBuf_.Get<int32_t>().GetValue(E_ + e);
    }

    __aicore__ inline int32_t Count(int32_t e)
    {
        return metaBuf_.Get<int32_t>().GetValue(e);
    }

    __aicore__ inline int32_t SumMPad()
    {
        return metaBuf_.Get<int32_t>().GetValue(3 * E_ + 1);
    }

    __aicore__ inline int32_t MValid()
    {
        return metaBuf_.Get<int32_t>().GetValue(3 * E_ + 2);
    }

    // Owning expert of a global padded row. Empty experts have
    // mPadOff[e] == mPadOff[e+1], so the scan steps over them.
    __aicore__ inline int32_t FindExpert(int32_t rowW)
    {
        LocalTensor<int32_t> meta = metaBuf_.Get<int32_t>();
        int32_t e = 0;
        while (e + 1 < E_ + 1 && rowW >= meta.GetValue(2 * E_ + e + 1)) {
            ++e;
        }
        return e < E_ ? e : E_ - 1;
    }

    // Owning expert of a global REAL row g (0 <= g < MValid). Empty experts
    // have startRow[e] == startRow[e+1], so the scan steps over them.
    __aicore__ inline int32_t FindExpertReal(int32_t g)
    {
        LocalTensor<int32_t> meta = metaBuf_.Get<int32_t>();
        int32_t e = 0;
        while (e + 1 < E_ && g >= meta.GetValue(E_ + e + 1)) {
            ++e;
        }
        return e < E_ ? e : E_ - 1;
    }

private:
    GlobalTensor<int64_t> glGM_;
    TBuf<AscendC::TPosition::VECCALC> metaBuf_;
    int32_t E_;
};

// ---------------------------------------------------------------------------
// Fused GEMM (AIC): production MatmulImpl int8 ND x int8 NZ -> int32 ND.
// Tiles are (expert, M-tile, N-tile) with BM=128 / BN=256; the acc is written
// to [sumMPad, N] int32 GM (cross-expert padded rows). Wave-major tile order
// (t = w*cores + core) with a wave handshake to the local AIV pair.
// ---------------------------------------------------------------------------
class GmsqGemmKernel256 {
public:
    __aicore__ inline GmsqGemmKernel256() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR acc, GM_ADDR scPtrTbl,
                                const GlobalTensor<int64_t> &glGM, int32_t E, int32_t glType,
                                int32_t K, int32_t N, int32_t nBlock, TPipe *pipe)
    {
        K_ = K;
        N_ = N;
        // Runtime N-block size (host-chosen multiple of baseN=256, <= singleCoreN
        // 1024) for MTE2 B-fetch load balance. The MatmulImpl iterates
        // nBlock/baseN internal baseN=256 N-tiles and REUSES A across them in
        // L0A, instead of reloading A per 256-wide tile.
        nBlock_ = nBlock;
        nNum_ = (N + nBlock_ - 1) / nBlock_;
        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(x));
        wGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(w));
        accGM_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(acc));
        scPtrTblGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(scPtrTbl));
        // AIC: GM-direct group_list access (no VECCALC UB on the cube core).
        meta_.Init(glGM, E, glType);
        sumMPad_ = meta_.SumMPad();
        totalTiles_ = (sumMPad_ / BM) * nNum_;
        mm_.SetSubBlockIdx(0);
        mm_.Init(static_cast<TCubeTiling *>(nullptr), pipe);
    }

    __aicore__ inline void ProcessFused(int32_t skipUnpack)
    {
        int32_t core = AscendC::GetBlockIdx();
        int32_t cores = AscendC::GetBlockNum();
        // On a P34 cache MISS the AIV stage fills the whole [E,K,N] NZ workspace
        // first; the AIC waits on the grid-wide barrier so it reads the freshly
        // written NZ weight. On a HIT the persistent NZ workspace is already valid.
        if (skipUnpack == 0) {
            AscendC::SyncAll<false>();
        }
        int32_t totalWaves = (totalTiles_ + cores - 1) / cores;
        // P54 header-opt: small-M (decode/cap, mValid<=64) collapses the per-wave
        // handshake into ONE cross-core sync (decode = 5 waves -> 5 syncs before,
        // 1 now), removing the fixed per-wave sync/launch overhead that dominates
        // small-M. Formula MUST match GmsqFusedAivKernel256::ActPhase (same
        // threshold P54_SMALL_M and same expression).
        int32_t mValid = meta_.SumM();
        int32_t waveBatch = (mValid <= P54_SMALL_M)
                                ? totalWaves
                                : (totalWaves + SYNC_THROTTLE - 1) / SYNC_THROTTLE;
        if (waveBatch < 1) {
            waveBatch = 1;
        }
        for (int32_t w = 0; w < totalWaves; w++) {
            int32_t t = w * cores + core;
            if (t < totalTiles_) {
                int32_t bmG = t / nNum_;
                int32_t bn = t - bmG * nNum_;
                int32_t rowW = bmG * BM;
                // Per-expert M dynamic split: expert is resolved O(1) from the
                // precomputed per-M-block map (no O(E) GM FindExpert scan).
                int32_t e = meta_.ExpertOfBlock(bmG);
                int32_t lr0 = rowW - meta_.PadOff(e);
                int32_t aRow = meta_.Start(e) + lr0;
                int32_t rowsAvail = meta_.Count(e) - lr0;
                int32_t curM = rowsAvail < BM ? rowsAvail : BM;
                if (curM > 0) {
                    int32_t tailN = bn * nBlock_;
                    int32_t curBN = N_ - tailN < nBlock_ ? N_ - tailN : nBlock_;
                    // NZ B-tile base for expert e, N-offset tailN (non-transpose
                    // SetWOffset == tailN * AlignUp16(K) == tailN * K, K mult 16).
                    int64_t wBase = (int64_t)e * K_ * N_ + (int64_t)tailN * K_;
                    mm_.SetOrgShape(meta_.Count(e), N_, K_);
                    mm_.SetSingleShape(curM, curBN, K_);
                    mm_.SetTensorA(xGM_[(int64_t)aRow * K_], false);
                    mm_.SetTensorB(wGM_[wBase], false);
                    // P49 epilogue dequant: int32 L0C -> half C with the per-channel
                    // fp32 channel scale (fp32 bits encoded in the low 32 bits of the
                    // ND int64 weight_scale carrier, reinterpreted as uint64). Offset
                    // the scale base by tailN so the fixpipe reads scale[tailN + col].
                    deqScale_.SetGlobalBuffer(
                        reinterpret_cast<__gm__ uint64_t *>(ReadGmPtr(scPtrTblGM_, e)), N_);
                    mm_.SetQuantVector(deqScale_[tailN]);
                    mm_.IterateAll<true>(accGM_[(int64_t)rowW * N_ + tailN], 0);
                }
            }
            if (((w + 1) % waveBatch == 0) || (w + 1 == totalWaves)) {
                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(FLAG_WAVE_READY);
            }
        }
    }

private:
    GlobalTensor<int8_t> xGM_;
    GlobalTensor<int8_t> wGM_;
    GlobalTensor<half> accGM_;
    GlobalTensor<int32_t> scPtrTblGM_;
    GlobalTensor<uint64_t> deqScale_;
    DeviceGlMetaGm meta_;
    GmmMt mm_;
    int32_t K_;
    int32_t N_;
    int32_t nBlock_;
    int32_t nNum_;
    int32_t totalTiles_;
    int32_t sumMPad_;
};

// ---------------------------------------------------------------------------
// Fused AIV stage: strip-ordered NZ unpack producer + wave-driven act/quant
// consumer on every AI Core's AIV pair.
//   Phase 0: channel-scale de-interleave (int64 carriers -> contiguous fp32),
//     striped over all AIV slots; wave-0 SyncAll guarantees visibility.
//   Phase 1 (P34 cache miss only): unpack the whole [E,K,N] cache into FRACTAL_NZ
//     layout for ALL experts (any future group_list's active set is covered),
//     then grid-sync so the AIC GEMM sees the fresh NZ weight.
//   Phase 2: per wave batch: wait FLAG_WAVE_READY + SyncAll, then act+quant the
//     row-blocks completed by this batch. Rows striped over all slots.
// ---------------------------------------------------------------------------
class GmsqFusedAivKernel256 {
public:
    __aicore__ inline GmsqFusedAivKernel256() {}

    __aicore__ inline void Init(GM_ADDR wPtrTbl, GM_ADDR scPtrTbl, GM_ADDR w8, GM_ADDR acc,
                                GM_ADDR scaleF32, GM_ADDR xScale, GM_ADDR y, GM_ADDR yScale,
                                const GlobalTensor<int64_t> &glGM, int32_t E, int32_t glType,
                                int32_t kNum, int32_t nNum, int32_t NP, int32_t N, int32_t N2,
                                int32_t K, int32_t C, float beta, float invBeta,
                                int32_t hasLinear, float linBeta, float invLinBeta,
                                int32_t nBlock, TPipe *pipe, int32_t skipUnpack,
                                int32_t nzInput)
    {
        skipUnpack_ = skipUnpack;
        nzInput_ = nzInput;
        E_ = E;
        kNum_ = kNum;
        nNum_ = nNum;
        NP_ = NP;
        N_ = N;
        N2_ = N2;
        K_ = K;
        // x_scale is loaded in on-demand chunks (see LoadXsChunk): the UB
        // footprint no longer scales with the input capacity C, which used to
        // overflow the AIV UB (507015) for profile-sized C (>= ~18K rows).
        (void)C;
        beta_ = beta;
        invBeta2_ = 2.0f * invBeta;
        hasLinear_ = hasLinear;
        linBeta_ = linBeta;
        invLinBeta2_ = 2.0f * invLinBeta;
        // device-side group_list parse -> local UB meta
        meta_.Init(glGM, E, glType, pipe);
        // N-block count for the GEMM wave (must MATCH the AIC's nNum_ which now
        // uses the runtime nBlock, not the 256-wide unpack strip count nNum_).
        nBlock_ = nBlock;
        nBlockNum_ = (N + nBlock_ - 1) / nBlock_;
        gemmTiles_ = (meta_.SumMPad() / BM) * nBlockNum_;

        ptrTblGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(wPtrTbl));
        scPtrTblGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(scPtrTbl));
        w8GM_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(w8));
        accGM_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(acc));
        xsGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(xScale));
        yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(y));
        ysGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(yScale));

        pipe->InitBuffer(inQueue_, 2, UNPACK_SUB_K * FUSED_UNPACK_BNP * sizeof(int32_t));
        pipe->InitBuffer(upkH16Buf_, UNPACK_SUB_K * FUSED_UNPACK_BN * sizeof(half));
        pipe->InitBuffer(outQueue_, 2, UNPACK_SUB_K * FUSED_UNPACK_BN * sizeof(int8_t));
        uint32_t rowF32 = (uint32_t)(N2 * sizeof(float));
        uint32_t rowH16 = (uint32_t)(N2 * sizeof(half));
        pipe->InitBuffer(accGQueue_, 1, rowH16);
        pipe->InitBuffer(accUQueue_, 1, rowH16);
        pipe->InitBuffer(xsQueue_, 1, XS_MAX_CHUNK * sizeof(float));
        pipe->InitBuffer(gBuf_, rowF32);
        pipe->InitBuffer(uBuf_, rowF32);
        pipe->InitBuffer(tBuf_, rowF32);
        pipe->InitBuffer(aBuf_, rowF32);
        pipe->InitBuffer(actH16Buf_, (uint32_t)(N2 * sizeof(half)));
        pipe->InitBuffer(yQueue_, 1, (uint32_t)(N2 * sizeof(int8_t)));
        pipe->InitBuffer(ysQueue_, 1, 8 * sizeof(float));
        pipe->InitBuffer(redWorkBuf_, ACT_RED_CHUNK * sizeof(float));
        pipe->InitBuffer(chunkMaxBuf_, 8 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int32_t coreIdx = AivCoreIdx();
        int32_t subIdx = AivSubIdx();
        int32_t slot = AivSlotIdx();
        int32_t slots = AivSlotNum();
        int32_t cores = AscendC::GetBlockNum();
        int32_t totalWaves = (gemmTiles_ + cores - 1) / cores;

        // Phase 1 (P34 cache miss only): fill the whole [E,K,N] NZ cache (all
        // experts), then grid-sync so the AIC GEMM sees the fresh NZ weight.
        if (skipUnpack_ == 0) {
            CacheFillUnpackAll(coreIdx, subIdx, cores);
            AscendC::SyncAll<false>();
        }
        // Phase 2: wave-driven fused act+quant (expert index; no strip gating).
        ActPhase(slot, slots, cores, totalWaves);
    }

private:
    // Phase 1 (P34 cache miss only): fill the whole [E,K,N] FRACTAL_NZ cache
    // for ALL experts. Distribution: (expert, bn) strips across AIV cores; both
    // sub-blocks of a block split the K rows via subIdx.
    __aicore__ inline void CacheFillUnpackAll(int32_t coreIdx, int32_t subIdx, int32_t cores)
    {
        if (nzInput_) {
            CacheFillUnpackAllNz(coreIdx, subIdx, cores);
            return;
        }
        int32_t totalStrips = E_ * nNum_;
        for (int32_t idx = coreIdx; idx < totalStrips; idx += cores) {
            int32_t e = idx / nNum_;
            int32_t bn = idx % nNum_;
            packedGM_.SetGlobalBuffer(
                reinterpret_cast<__gm__ int32_t *>(ReadGmPtr(ptrTblGM_, e)));
            for (int32_t bk = 0; bk < kNum_; bk++) {
                UnpackTile(e, bn, bk, subIdx);
            }
        }
    }

    // Native int4 FRACTAL_NZ input: the int32 carrier views an int8 packed-NZ
    // storage whose logical matrix is [K, N/2] (two INT4 per byte on the output
    // axis, W4A8 process_weights convention: transpose -> maybe_trans_nz ->
    // view(int32)). Unpack consumes the native INT8-NZ geometry directly — the
    // same block tiling as the w8 cache, so each input [16,32] byte block maps
    // to two aligned [16,32] output blocks and no byte is ever transposed.
    __aicore__ inline void CacheFillUnpackAllNz(int32_t coreIdx, int32_t subIdx, int32_t cores)
    {
        int32_t npb = NP_ * 4;      // packed bytes per row (two INT4 per byte)
        int32_t nBlk = npb / 32;    // input N-blocks of 32 bytes (N % 256 == 0 gate)
        int32_t totalStrips = E_ * nNum_;
        for (int32_t idx = coreIdx; idx < totalStrips; idx += cores) {
            int32_t e = idx / nNum_;
            int32_t bn = idx % nNum_;
            packedGM8_.SetGlobalBuffer(
                reinterpret_cast<__gm__ int8_t *>(ReadGmPtr(ptrTblGM_, e)));
            for (int32_t bk = 0; bk < kNum_; bk++) {
                int32_t row0 = bk * UNPACK_BK + subIdx * UNPACK_SUB_K;
                for (int32_t jb = 0; jb < 4; jb++) {
                    // strip covers 256 logical N = 128 packed bytes = 4 blocks;
                    // each (bk, subIdx) spans 32 rows = two 16-row NZ blocks
                    for (int32_t kb = 0; kb < 2; kb++) {
                        UnpackBlockNz(e, row0 + kb * 16, bn * 4 + jb, nBlk);
                    }
                }
            }
        }
    }

    // One input block: int8 FRACTAL_NZ [16, 32] packed bytes at
    //   inOff = ((k/16)*(npb/32) + j/32)*512 + (k%16)*32 + (j%32)
    // -> nibble-expand to logical int8 [16, 64]
    // -> two [16, 32] halves into the w8 NZ cache at
    //   w8Off = ((k/16)*(N/32) + (2*j + h)/32)*512 + (k%16)*32 + (n%32)
    // Both offsets are 512 B block-aligned; the two nibbles of byte j land on
    // logical n = 2*j (low) and 2*j + 1 (high), matching the W4A8 packer.
    __aicore__ inline void UnpackBlockNz(int32_t e, int32_t row0, int32_t jb, int32_t nBlk)
    {
        // Input addressing: torch_npu standard interleaved FRACTAL_NZ,
        //   block(jb, kb16) base = (kb16 * nBlk + jb) * 512.
        // NOTE this is NOT the w8 workspace convention below (the unpacked w8
        // cache is N-block-major, matching the GEMM B operand).
        int32_t kb16 = row0 / 16;
        int64_t inOff = (static_cast<int64_t>(kb16) * nBlk + jb) * 512;
        LocalTensor<int8_t> in8 = inQueue_.AllocTensor<int8_t>();
        AscendC::DataCopy(in8, packedGM8_[inOff], 512);
        inQueue_.EnQue(in8);
        LocalTensor<int8_t> in8l = inQueue_.DeQue<int8_t>();
        LocalTensor<half> h16 = upkH16Buf_.Get<half>();
        LocalTensor<int8_t> out8 = outQueue_.AllocTensor<int8_t>();
        AscendC::Cast(h16, in8l.ReinterpretCast<AscendC::int4b_t>(),
                      AscendC::RoundMode::CAST_NONE, 1024);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(out8, h16, AscendC::RoundMode::CAST_NONE, 1024);
        inQueue_.FreeTensor(in8l);
        outQueue_.EnQue(out8);
        LocalTensor<int8_t> out8l = outQueue_.DeQue<int8_t>();
        // Direct per-run GM stores: each [16, 32] half is 16 contiguous 32 B
        // runs at w8Off + r*32 (no UB->UB staging buffer in between). The
        // expert base eBase keeps every expert inside its own cache slice
        // (same contract as UnpackTile).
        int64_t eBase = (int64_t)e * K_ * N_;
        for (int32_t h = 0; h < 2; h++) {
            int64_t w8Off =
                eBase + static_cast<int64_t>(jb * 2 + h) * (K_ * 32) + kb16 * 512;
            for (int32_t r = 0; r < 16; r++) {
                AscendC::DataCopy(w8GM_[w8Off + r * 32], out8l[r * 64 + h * 32], 32);
            }
        }
        outQueue_.FreeTensor(out8l);
    }

    // One unpack tile: 32 rows x 32 int32 carriers -> 32 x 256 int8, written
    // into FRACTAL_NZ layout. The [32,256] ND tile is 2 K-blocks x 8 N-blocks
    // of [16,32]; each [16,32] block is scattered to its NZ position via one
    // DataCopyPad (16 total). NZ offset for int8 (C0=32, K16=AlignUp16(K)=K):
    //   off(k,n) = (n/32)*K*32 + (k/16)*512 + (k%16)*32 + (n%32).
    __aicore__ inline void UnpackTile(int32_t e, int32_t bn, int32_t bk, int32_t subIdx)
    {
        int32_t row0 = bk * UNPACK_BK + subIdx * UNPACK_SUB_K;
        LocalTensor<int32_t> pk = inQueue_.AllocTensor<int32_t>();
        DataCopyExtParams cp{(uint16_t)UNPACK_SUB_K, (uint32_t)(FUSED_UNPACK_BNP * sizeof(int32_t)),
                             (uint32_t)((NP_ - FUSED_UNPACK_BNP) * sizeof(int32_t)), 0, 0};
        DataCopyPadExtParams<int32_t> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(pk, packedGM_[(int64_t)row0 * NP_ + bn * FUSED_UNPACK_BNP], cp, pp);
        inQueue_.EnQue(pk);

        LocalTensor<half> h16 = upkH16Buf_.Get<half>();
        LocalTensor<int32_t> pkl = inQueue_.DeQue<int32_t>();
        LocalTensor<AscendC::int4b_t> p4 = pkl.ReinterpretCast<AscendC::int4b_t>();
        AscendC::Cast(h16, p4, AscendC::RoundMode::CAST_NONE, UNPACK_SUB_K * FUSED_UNPACK_BN);
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<int8_t> out = outQueue_.AllocTensor<int8_t>();
        AscendC::Cast(out, h16, AscendC::RoundMode::CAST_NONE, UNPACK_SUB_K * FUSED_UNPACK_BN);
        inQueue_.FreeTensor(pkl);
        outQueue_.EnQue(out);

        LocalTensor<int8_t> outl = outQueue_.DeQue<int8_t>();
        // Scatter the ND [32,256] tile into the NZ workspace: 16 [16,32] blocks.
        // DataCopyExtParams in the UB->GM direction: blockLen is BYTES, the
        // srcStride is the UB-side gap in 32-BYTE BLOCKS, and dstStride is the
        // GM-side gap in BYTES (see MOV_UB_TO_OUT_ALIGN: burst_src_stride =
        // ceil32(lenBurst)*32 + srcGap*32; burst_dst_stride = lenBurst + dstGap).
        // Source rows sit at 256B stride (32B written, gap 224B == 7 blocks);
        // NZ destination rows are contiguous 32B (gap 0B).
        int64_t eBase = (int64_t)e * K_ * N_;
        int32_t rowK = row0 / 16;  // K-block base index (16-row blocks)
        DataCopyExtParams nzcp{static_cast<uint16_t>(16), static_cast<uint32_t>(32),
                               static_cast<uint32_t>(7), 0, 0};
        for (int32_t kblk = 0; kblk < 2; kblk++) {
            int32_t srcRowBase = kblk * 16 * FUSED_UNPACK_BN;
            int64_t dstKBase = eBase + (int64_t)(rowK + kblk) * 512;
            for (int32_t nblk = 0; nblk < 8; nblk++) {
                int32_t srcOff = srcRowBase + nblk * 32;
                int64_t dstOff = dstKBase + (int64_t)(bn * 8 + nblk) * K_ * 32;
                AscendC::DataCopyPad(w8GM_[dstOff], outl[srcOff], nzcp);
            }
        }
        outQueue_.FreeTensor(outl);
    }

    // Phase 2: wave-driven fused act+quant.
    // On-demand x_scale chunk: load x_scale[g0 : g0+len) into the fixed 8 KiB
    // chunk buffer. Total MTE2 bytes = M_valid * 4 (padding rows are never read,
    // versus the old full-capacity C * 4 copy on every AIV slot).
    __aicore__ inline LocalTensor<float> LoadXsChunk(int32_t g0, int32_t len)
    {
        LocalTensor<float> xs = xsQueue_.AllocTensor<float>();
        DataCopyExtParams xcp{1, (uint32_t)(len * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> xpp{false, 0, 0, 0};
        AscendC::DataCopyPad(xs, xsGM_[(int64_t)g0], xcp, xpp);
        xsQueue_.EnQue(xs);
        return xsQueue_.DeQue<float>();
    }

    __aicore__ inline void FreeXsChunk(LocalTensor<float> &xs)
    {
        xsQueue_.FreeTensor(xs);
    }

    __aicore__ inline void ActPhase(int32_t slot, int32_t slots, int32_t cores,
                                    int32_t totalWaves)
    {
        g_ = gBuf_.Get<float>();
        u_ = uBuf_.Get<float>();
        t_ = tBuf_.Get<float>();
        a_ = aBuf_.Get<float>();
        h16_ = actH16Buf_.Get<half>();
        redWork_ = redWorkBuf_.Get<float>();
        chunkMax_ = chunkMaxBuf_.Get<float>();

        // P54 header-opt: match GmsqGemmKernel256::ProcessFused (identical
        // P54_SMALL_M threshold and expression) so the AIC flag emission and the
        // AIV flag wait collapse to a single cross-core sync for small M.
        int32_t mValid = meta_.MValid();
        int32_t waveBatch = (mValid <= P54_SMALL_M)
                                ? totalWaves
                                : (totalWaves + SYNC_THROTTLE - 1) / SYNC_THROTTLE;
        if (waveBatch < 1) {
            waveBatch = 1;
        }
        int32_t batches = (totalWaves + waveBatch - 1) / waveBatch;
        // P54 merge-writes: a single sync means every accumulator is visible
        // before the epilogue, so a tiny M (decode/cap: 1-2 rows per expert) no
        // longer needs to be serialized row-block by row-block under BM=128
        // padding. Stripe the REAL rows globally over all slots instead.
        bool globalStripe = (batches == 1);
        int32_t prevRB = 0;
        for (int32_t b = 0; b < batches; b++) {
            AscendC::CrossCoreWaitFlag(FLAG_WAVE_READY);
            AscendC::SyncAll<true>();
            int32_t completed = (b + 1) * waveBatch * cores;
            if (completed > gemmTiles_) {
                completed = gemmTiles_;
            }
            int32_t rbDone = completed / nBlockNum_;
            if (globalStripe) {
                ActPhaseGlobal(slot, slots, mValid);
            } else {
                for (int32_t rb = prevRB; rb < rbDone; rb++) {
                    ActRowBlock(rb, slot, slots);
                }
            }
            prevRB = rbDone;
        }
    }

    // Act+quant ALL real rows in one global stripe (single-sync path). For a
    // tiny M each expert has 1-2 rows, so the per-row-block stripe would
    // serialize ~E blocks; this spreads the MValid real rows across all slots.
    // Chunk-cyclic order (each slot owns contiguous row chunks) keeps the
    // on-demand x_scale load at one small copy per chunk instead of one copy
    // per row, and keeps the load balance identical to the old slot-strided
    // order. accRow mapping is IDENTICAL to ActRowBlock:
    // accRow = PadOff(e) + (g-Start(e)).
    __aicore__ inline void ActPhaseGlobal(int32_t slot, int32_t slots, int32_t mValid)
    {
        int32_t chunk = (mValid + slots - 1) / slots;
        if (chunk > XS_MAX_CHUNK) {
            chunk = XS_MAX_CHUNK;
        }
        if (chunk < 1) {
            chunk = 1;
        }
        for (int32_t g0 = slot * chunk; g0 < mValid; g0 += slots * chunk) {
            int32_t len = mValid - g0 < chunk ? mValid - g0 : chunk;
            LocalTensor<float> xs = LoadXsChunk(g0, len);
            for (int32_t g = g0; g < g0 + len; g++) {
                int32_t e = meta_.FindExpertReal(g);
                int32_t lr = g - meta_.Start(e);
                int32_t accRow = meta_.PadOff(e) + lr;
                ActRow(accRow, g, xs.GetValue(g - g0));
            }
            FreeXsChunk(xs);
        }
    }

    // Act+quant all real rows of padded row-block rb (striped over slots).
    // x_scale for the block (<= BM rows) is loaded on demand once per block.
    __aicore__ inline void ActRowBlock(int32_t rb, int32_t slot, int32_t slots)
    {
        int32_t rowW = rb * BM;
        int32_t e = meta_.FindExpert(rowW);
        int32_t lr0 = rowW - meta_.PadOff(e);
        int32_t valid = meta_.Count(e) - lr0;
        if (valid > BM) {
            valid = BM;
        }
        if (valid <= 0 || slot >= valid) {
            return;
        }
        int32_t outRow0 = meta_.Start(e) + lr0;
        LocalTensor<float> xs = LoadXsChunk(outRow0, valid);
        for (int32_t r = slot; r < valid; r += slots) {
            ActRow(rowW + r, outRow0 + r, xs.GetValue(r));
        }
        FreeXsChunk(xs);
    }

    // Per-row act+quant. The channel scale (sgl/sul) is now applied in the AIC
    // epilogue dequant (int32 acc * fp32 channel scale -> half), so the AIV only
    // reads the half accumulator, casts to float, and applies the x scale.
    __aicore__ inline void ActRow(int32_t accRow, int32_t row, float xsi)
    {
        LocalTensor<half> accG = accGQueue_.AllocTensor<half>();
        AscendC::DataCopy(accG, accGM_[(int64_t)accRow * N_], N2_);
        accGQueue_.EnQue(accG);
        LocalTensor<half> accU = accUQueue_.AllocTensor<half>();
        AscendC::DataCopy(accU, accGM_[(int64_t)accRow * N_ + N2_], N2_);
        accUQueue_.EnQue(accU);

        LocalTensor<half> accGl = accGQueue_.DeQue<half>();
        AscendC::Cast(g_, accGl, AscendC::RoundMode::CAST_NONE, N2_);
        accGQueue_.FreeTensor(accGl);
        LocalTensor<half> accUl = accUQueue_.DeQue<half>();
        AscendC::Cast(u_, accUl, AscendC::RoundMode::CAST_NONE, N2_);
        accUQueue_.FreeTensor(accUl);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(g_, g_, xsi, N2_);
        AscendC::Muls(u_, u_, xsi, N2_);
        AscendC::PipeBarrier<PIPE_V>();

        // SiTU: gate = beta*tanh(g/beta)*sigmoid(g); tanh(z)=2*sig(2z)-1
        AscendC::Muls(t_, g_, invBeta2_, N2_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sigmoid(t_, t_, N2_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sigmoid(g_, g_, N2_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(t_, t_, 2.0f, N2_);
        AscendC::Adds(t_, t_, -1.0f, N2_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(t_, t_, g_, N2_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(t_, t_, beta_, N2_);
        AscendC::PipeBarrier<PIPE_V>();
        if (hasLinear_ != 0) {
            AscendC::Muls(u_, u_, invLinBeta2_, N2_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sigmoid(u_, u_, N2_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(u_, u_, 2.0f, N2_);
            AscendC::Adds(u_, u_, -1.0f, N2_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(u_, u_, linBeta_, N2_);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Mul(t_, t_, u_, N2_);  // act row in t
        AscendC::PipeBarrier<PIPE_V>();

        // rowmax(|act|) via safe ReduceMax API (chunked, barrier-free fire-and-collect)
        AscendC::Abs(a_, t_, N2_);
        AscendC::PipeBarrier<PIPE_V>();
        float rowMax = 0.0f;
        int32_t chunks = (N2_ + ACT_RED_CHUNK - 1) / ACT_RED_CHUNK;
        for (int32_t ch = 0; ch < N2_; ch += ACT_RED_CHUNK) {
            int32_t idx = ch / ACT_RED_CHUNK;
            int32_t len = N2_ - ch < ACT_RED_CHUNK ? N2_ - ch : ACT_RED_CHUNK;
            AscendC::ReduceMax(chunkMax_[idx], a_[ch], redWork_, len);
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (int32_t idx = 0; idx < chunks; idx++) {
            float v = chunkMax_.GetValue(idx);
            rowMax = rowMax > v ? rowMax : v;
        }

        // y_scale + safe inverse (zero rows: inv=0 -> y=0 exactly)
        float yScale = rowMax / 127.0f;
        float inv = rowMax > 0.0f ? 127.0f / rowMax : 0.0f;

        // y = rint(act * inv) -> int8
        AscendC::Muls(t_, t_, inv, N2_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(h16_, t_, AscendC::RoundMode::CAST_RINT, N2_);
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<int8_t> y8 = yQueue_.AllocTensor<int8_t>();
        AscendC::Cast(y8, h16_, AscendC::RoundMode::CAST_NONE, N2_);
        yQueue_.EnQue(y8);
        LocalTensor<int8_t> y8l = yQueue_.DeQue<int8_t>();
        AscendC::DataCopy(yGM_[(int64_t)row * N2_], y8l, N2_);
        yQueue_.FreeTensor(y8l);

        LocalTensor<float> ys = ysQueue_.AllocTensor<float>();
        ys.SetValue(0, yScale);
        AscendC::PipeBarrier<PIPE_ALL>();
        ysQueue_.EnQue(ys);
        LocalTensor<float> ysl = ysQueue_.DeQue<float>();
        DataCopyExtParams ycp{1, (uint32_t)sizeof(float), 0, 0, 0};
        AscendC::DataCopyPad(ysGM_[row], ysl, ycp);
        ysQueue_.FreeTensor(ysl);
    }

    GlobalTensor<int32_t> ptrTblGM_;
    GlobalTensor<int32_t> scPtrTblGM_;
    GlobalTensor<int32_t> packedGM_;
    GlobalTensor<int8_t> packedGM8_;
    GlobalTensor<int8_t> w8GM_;
    GlobalTensor<half> accGM_;
    GlobalTensor<float> xsGM_;
    GlobalTensor<int8_t> yGM_;
    GlobalTensor<float> ysGM_;
    DeviceGlMeta meta_;
    TQue<AscendC::TPosition::VECIN, 1> inQueue_;
    TQue<AscendC::TPosition::VECOUT, 1> outQueue_;
    TBuf<AscendC::TPosition::VECCALC> upkH16Buf_;
    TQue<AscendC::TPosition::VECIN, 1> accGQueue_;
    TQue<AscendC::TPosition::VECIN, 1> accUQueue_;
    TQue<AscendC::TPosition::VECIN, 1> xsQueue_;
    TQue<AscendC::TPosition::VECOUT, 1> yQueue_;
    TQue<AscendC::TPosition::VECOUT, 1> ysQueue_;
    TBuf<AscendC::TPosition::VECCALC> gBuf_;
    TBuf<AscendC::TPosition::VECCALC> uBuf_;
    TBuf<AscendC::TPosition::VECCALC> tBuf_;
    TBuf<AscendC::TPosition::VECCALC> aBuf_;
    TBuf<AscendC::TPosition::VECCALC> actH16Buf_;
    TBuf<AscendC::TPosition::VECCALC> redWorkBuf_;
    TBuf<AscendC::TPosition::VECCALC> chunkMaxBuf_;
    LocalTensor<float> g_;
    LocalTensor<float> u_;
    LocalTensor<float> t_;
    LocalTensor<float> a_;
    LocalTensor<half> h16_;
    LocalTensor<float> redWork_;
    LocalTensor<float> chunkMax_;
    int32_t E_;
    int32_t skipUnpack_;
    int32_t nzInput_;
    int32_t kNum_;
    int32_t nNum_;
    int32_t nBlock_;
    int32_t nBlockNum_;
    int32_t gemmTiles_;
    int32_t NP_;
    int32_t N_;
    int32_t N2_;
    int32_t K_;
    float beta_;
    float invBeta2_;
    int32_t hasLinear_;
    float linBeta_;
    float invLinBeta2_;
};

}  // namespace

// Device-side group_list parse fused single launch (production MatmulImpl GEMM),
// ported to the fused interface. The group_list device tensor is parsed on-device
// every forward, so the operator is correct for DYNAMIC production MoE
// group_list (new tensor or in-place update) with zero host D2H and no glPtr in
// any cache key.
extern "C" __global__ __aicore__ void gmsq_fused_256(GM_ADDR x, GM_ADDR wPtrTbl, GM_ADDR scPtrTbl,
                                                      GM_ADDR w8, GM_ADDR acc, GM_ADDR scaleF32,
                                                      GM_ADDR xScale, GM_ADDR y, GM_ADDR yScale,
                                                      GM_ADDR groupList, int32_t E, int32_t kNum,
                                                      int32_t nNum, int32_t nBlock, int32_t NP,
                                                      int32_t N, int32_t N2, int32_t K, int32_t C,
                                                      int32_t glType, float beta, float invBeta,
                                                      int32_t hasLinear, float linBeta,
                                                      float invLinBeta, int32_t skipUnpack,
                                                      int32_t nzInput)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GlobalTensor<int64_t> glGM;
    glGM.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(groupList));
    if ASCEND_IS_AIC {
        AscendC::AscendCUtils::SetOverflow(1);
        TPipe pipe;
        GmsqGemmKernel256 kernel;
        kernel.Init(x, w8, acc, scPtrTbl, glGM, E, glType, K, N, nBlock, &pipe);
        kernel.ProcessFused(skipUnpack);
    }
    if ASCEND_IS_AIV {
        TPipe pipe;
        GmsqFusedAivKernel256 kernel;
        kernel.Init(wPtrTbl, scPtrTbl, w8, acc, scaleF32, xScale, y, yScale, glGM, E, glType,
                    kNum, nNum, NP, N, N2, K, C, beta, invBeta, hasLinear, linBeta, invLinBeta,
                    nBlock, &pipe, skipUnpack, nzInput);
        kernel.Process();
    }
}



// Host-side launcher exported to the vllm_ascend_C extension (bgmv/sgmv
// style): this ccec-compiled TU owns the kernel entry, so the tripple-chevron
// launch is resolved inside libvllm_ascend_kernels and only this symbol is
// linked out to the host extension.
namespace vllm_ascend {

void gmsq_fused_256_impl(uint32_t blockDim, void *stream, void *x, void *wPtrTbl, void *scPtrTbl,
                          void *w8, void *acc, void *scaleF32, void *xScale, void *y, void *yScale,
                          void *groupList, int32_t E, int32_t kNum, int32_t nNum, int32_t nBlock,
                          int32_t NP, int32_t N, int32_t N2, int32_t K, int32_t C, int32_t glType,
                          float beta, float invBeta, int32_t hasLinear, float linBeta,
                          float invLinBeta, int32_t skipUnpack, int32_t nzInput)
{
    gmsq_fused_256<<<blockDim, nullptr, stream>>>(x, wPtrTbl, scPtrTbl, w8, acc, scaleF32, xScale,
                                                   y, yScale, groupList, E, kNum, nNum, nBlock, NP,
                                                   N, N2, K, C, glType, beta, invBeta, hasLinear,
                                                   linBeta, invLinBeta, skipUnpack, nzInput);
}
}  // namespace vllm_ascend
