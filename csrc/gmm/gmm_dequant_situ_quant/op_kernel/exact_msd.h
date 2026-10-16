/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Exact integer MSD, derived from the official A8W4 decomposition. Included
// inside the adaptive kernel's anonymous namespace to share device metadata and
// its unchanged SiTU/quant epilogue. Runtime shape limits come from alignment,
// metadata/UB capacity, and exact FP32 integer reconstruction, not model shapes.
// ND packed W4 and native INT8-NZ packed W4 viewed as INT32 are read in place.
constexpr int32_t MSD_SLOTS = 8;
constexpr int32_t MSD_RAW_ROWS = 2 * BM + 16;
constexpr int32_t MSD_NBLOCK = 1024;
constexpr int32_t MSD_BASE_K = 256;
constexpr int32_t MSD_BASE_N = 256;
constexpr int32_t MSD_PACK_K = 4096; // bounded activation-only UB tile

using MsdA = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND,
                                AscendC::int4b_t, false>;
using MsdB = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND,
                                AscendC::int4b_t, false>;
using MsdNzB = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::NZ,
                                  AscendC::int4b_t, false>;
using MsdC = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int32_t, false>;
using MsdBias = MsdC;

__aicore__ inline constexpr MatmulConfig MakeMsdConf()
{
    auto conf = MakeGmmConf();
    conf.basicK = MSD_BASE_K; // same L0 operand bytes as A8W8 K128.
    conf.basicN = MSD_BASE_N;
    conf.singleCoreM = 384; // at most 2*128+1 raw rows; internally M128 tiles.
    conf.singleCoreK = 16384; // FP32 exact-reconstruction bound; runtime K via SetSingleShape.
    conf.enableQuantVector = false; // INT32 output; no partial dequantization.
    return conf;
}

__aicore__ inline constexpr auto MakeMsdTiling()
{
    auto tiling = AscendC::GetMatmulApiTiling<MsdA, MsdB, MsdC, MsdBias>(MakeMsdConf());
    tiling.depthA1 = 8;
    tiling.depthB1 = 8;
    tiling.stepM = 1;
    tiling.stepN = 1;
    tiling.stepKa = 4;
    tiling.stepKb = 4;
    tiling.dbL0A = 2;
    tiling.dbL0B = 2;
    tiling.dbL0C = 1;
    return tiling;
}
constexpr static auto msdCFG = MakeMsdTiling();
using MsdMt = AscendC::MatmulImpl<MsdA, MsdB, MsdC, MsdBias, msdCFG>;

// Native packed INT8 NZ is [N/64,K/16,16,32] BYTES, not the K-block-major
// layout assumed by the old fallback's UnpackBlockNz. For logical W4 (k,n):
//   byte = ((n/64)*(K/16)+k/16)*512 + (k%16)*32 + (n%64)/2.
// Reinterpreting each byte as its low/high INT4 pair gives precisely the
// Matmul INT4 NZ layout [N/64,K/16,16,64]. The byte order does not change.
// Native ACL physical-memory reads and CANN 9.1 CopyNZ2NZImpl confirm this
// equivalence. The ordinary NZ Matmul loader can consume the original weight
// directly; no callback, L1 reorder, UB staging or GM weight copy is needed.
__aicore__ inline constexpr auto MakeMsdNzTiling()
{
    auto tiling = AscendC::GetMatmulApiTiling<MsdA, MsdNzB, MsdC, MsdBias>(MakeMsdConf());
    tiling.depthA1 = 8;
    tiling.depthB1 = 8;
    tiling.stepM = 1;
    tiling.stepN = 1;
    tiling.stepKa = 4;
    tiling.stepKb = 4;
    tiling.dbL0A = 2;
    tiling.dbL0B = 2;
    tiling.dbL0C = 1;
    return tiling;
}
constexpr static auto msdNzCFG = MakeMsdNzTiling();
using MsdNzMt = AscendC::MatmulImpl<MsdA, MsdNzB, MsdC, MsdBias, msdNzCFG>;

template <typename MM_TYPE = MsdMt, bool NativeNz = false>
class GmsqExactMsdCube {
public:
    __aicore__ inline void Init(GM_ADDR wt, GM_ADDR packed, GM_ADDR raw,
        const GlobalTensor<int64_t> &gl, int32_t E, int32_t K, int32_t N,
        int32_t glType, TPipe *pipe)
    {
        K_ = K;
        N_ = N;
        meta_.Init(gl, E, glType);
        ptr_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(wt));
        a_.SetGlobalBuffer(reinterpret_cast<__gm__ AscendC::int4b_t *>(packed));
        c_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(raw));
        mm_.SetSubBlockIdx(0);
        mm_.Init(static_cast<TCubeTiling *>(nullptr), pipe);
    }

    __aicore__ inline void Process()
    {
        const int32_t blocks = meta_.SumMPad() / BM;
        const int32_t nTiles = (N_ + MSD_NBLOCK - 1) / MSD_NBLOCK;
        const int32_t core = AscendC::GetBlockIdx();
        const int32_t cores = AscendC::GetBlockNum();
        for (int32_t begin = 0; begin < blocks; begin += MSD_SLOTS) {
            const int32_t count = blocks - begin < MSD_SLOTS ? blocks - begin : MSD_SLOTS;
            AscendC::SyncAll<false>(); // packed A complete on all AIVs
            for (int32_t task = core; task < count * nTiles; task += cores) {
                const int32_t slot = task / nTiles;
                const int32_t col = (task % nTiles) * MSD_NBLOCK;
                const int32_t width = N_ - col < MSD_NBLOCK ? N_ - col : MSD_NBLOCK;
                const int32_t block = begin + slot;
                const int32_t e = meta_.ExpertOfBlock(block);
                const int32_t localRow = block * BM - meta_.PadOff(e);
                const int32_t valid = meta_.Count(e) - localRow < BM ? meta_.Count(e) - localRow : BM;
                const int32_t rawRows = 2 * valid + 1;
                GlobalTensor<AscendC::int4b_t> weight;
                weight.SetGlobalBuffer(reinterpret_cast<__gm__ AscendC::int4b_t *>(ReadGmPtr(ptr_, e)));
                mm_.SetOrgShape(rawRows, N_, K_);
                mm_.SetSingleShape(rawRows, width, K_);
                mm_.SetTensorA(a_[static_cast<int64_t>(slot) * MSD_RAW_ROWS * K_], false);
                if constexpr (NativeNz) {
                    // Native N-block-major slice: col*K/2 bytes.
                    // GlobalTensor<int4b_t> offsets count nibbles, not bytes.
                    mm_.SetTensorB(weight[static_cast<int64_t>(col) * K_], false);
                } else {
                    mm_.SetTensorB(weight[col], false);
                }
                mm_.template IterateAll<true>(c_[static_cast<int64_t>(slot) * MSD_RAW_ROWS * N_ + col], 0);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SyncAll<false>(); // every raw INT32 tile visible
            AscendC::SyncAll<false>(); // consumer finished before either GM bank is reused
        }
    }

private:
    DeviceGlMetaGm meta_;
    GlobalTensor<int32_t> ptr_;
    GlobalTensor<AscendC::int4b_t> a_;
    GlobalTensor<int32_t> c_;
    MM_TYPE mm_;
    int32_t K_;
    int32_t N_;
};

class GmsqExactMsdVector : public GmsqFusedAivKernel256 {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR wt, GM_ADDR scales,
        GM_ADDR packed, GM_ADDR raw, GM_ADDR xs, GM_ADDR y, GM_ADDR ys,
        const GlobalTensor<int64_t> &gl, int32_t E, int32_t K, int32_t N,
        int32_t C, int32_t glType, float beta, float invBeta, int32_t hasLinear,
        float linBeta, float invLinBeta, TPipe *pipe)
    {
        GmsqFusedAivKernel256::Init(wt, scales, packed, raw, nullptr, xs, y, ys,
            gl, E, glType, K / UNPACK_BK, N / BN, N / 8, N, N / 2, K, C,
            beta, invBeta, hasLinear, linBeta, invLinBeta, MSD_NBLOCK, pipe,
            1, 0, nullptr, 0, BOUNDED_MAX_E, true);
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(x));
        packed_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(packed));
        raw_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(raw));
        g_ = gBuf_.Get<float>();
        u_ = uBuf_.Get<float>();
        t_ = tBuf_.Get<float>();
        a_ = aBuf_.Get<float>();
        h16_ = actH16Buf_.Get<half>();
        redWork_ = redWorkBuf_.Get<float>();
        chunkMax_ = chunkMaxBuf_.Get<float>();
    }

    __aicore__ inline void Process()
    {
        const int32_t blocks = meta_.SumMPad() / BM;
        const int32_t lane = AivSlotIdx();
        const int32_t lanes = AivSlotNum();
        for (int32_t begin = 0; begin < blocks; begin += MSD_SLOTS) {
            const int32_t count = blocks - begin < MSD_SLOTS ? blocks - begin : MSD_SLOTS;
            for (int32_t slot = 0; slot < count; ++slot) {
                const int32_t block = begin + slot;
                const int32_t e = meta_.FindExpert(block * BM);
                const int32_t localRow = block * BM - meta_.PadOff(e);
                const int32_t valid = meta_.Count(e) - localRow < BM ? meta_.Count(e) - localRow : BM;
                for (int32_t row = lane; row < valid; row += lanes) {
                    PackRow(meta_.Start(e) + localRow + row, slot, row);
                }
                if (lane == 0) {
                    PackCorrection(slot, valid);
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SyncAll<false>();
            AscendC::SyncAll<false>();
            // Globally stripe REAL rows inside this chunk. This avoids serial
            // low-utilization epilogue rounds for many small expert groups.
            int32_t cumulative = 0;
            for (int32_t slot = 0; slot < count; ++slot) {
                const int32_t block = begin + slot;
                const int32_t e = meta_.FindExpert(block * BM);
                const int32_t localRow = block * BM - meta_.PadOff(e);
                const int32_t valid = meta_.Count(e) - localRow < BM ? meta_.Count(e) - localRow : BM;
                for (int32_t row = (lane + lanes - cumulative % lanes) % lanes;
                     row < valid; row += lanes) {
                    const int32_t outputRow = meta_.Start(e) + localRow + row;
                    LocalTensor<float> xs = LoadXsChunk(outputRow, 1);
                    const float scale = xs.GetValue(0);
                    FreeXsChunk(xs);
                    Reconstruct(slot, row, valid, e, 0, g_);
                    Reconstruct(slot, row, valid, e, N2_, u_);
                    FinishActRow(outputRow, scale);
                }
                cumulative += valid;
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SyncAll<false>();
        }
    }

private:
    __aicore__ inline void PackRow(int32_t inputRow, int32_t slot, int32_t row)
    {
        for (int32_t k = 0; k < K_; k += MSD_PACK_K) {
            const int32_t len = K_ - k < MSD_PACK_K ? K_ - k : MSD_PACK_K;
            PackRowTile(inputRow, slot, row, k, len);
        }
    }

    __aicore__ inline void PackRowTile(int32_t inputRow, int32_t slot,
                                     int32_t row, int32_t k, int32_t len)
    {
        auto input = inQueue_.AllocTensor<int8_t>();
        AscendC::DataCopy(input, x_[static_cast<int64_t>(inputRow) * K_ + k], len);
        inQueue_.EnQue(input);
        auto src = inQueue_.DeQue<int8_t>();
        auto high = upkH16Buf_.Get<half>();
        auto low = high[len];
        auto out = outQueue_.AllocTensor<int8_t>();
        auto out4 = out.ReinterpretCast<AscendC::int4b_t>();
        AscendC::Cast(low, src, AscendC::RoundMode::CAST_NONE, len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(high, low, static_cast<half>(0.0625f), len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(out4, high, AscendC::RoundMode::CAST_FLOOR, len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(high, out4, AscendC::RoundMode::CAST_NONE, len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(high, high, static_cast<half>(-16), len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(low, low, high, len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(low, low, static_cast<half>(-8), len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(out4[len], low, AscendC::RoundMode::CAST_NONE, len);
        inQueue_.FreeTensor(src);
        outQueue_.EnQue(out);
        auto ready = outQueue_.DeQue<int8_t>();
        const int64_t offset = (static_cast<int64_t>(slot) * MSD_RAW_ROWS + 2 * row) * K_ / 2;
        if (len == K_) {
            // Preserve the original single-DMA fast path for K <= 4096.
            AscendC::DataCopy(packed_[offset], ready, len);
        } else {
            AscendC::DataCopy(packed_[offset + k / 2], ready, len / 2);
            AscendC::DataCopy(packed_[offset + (K_ + k) / 2], ready[len / 2], len / 2);
        }
        outQueue_.FreeTensor(ready);
    }

    __aicore__ inline void PackCorrection(int32_t slot, int32_t valid)
    {
        for (int32_t k = 0; k < K_; k += MSD_PACK_K) {
            const int32_t len = K_ - k < MSD_PACK_K ? K_ - k : MSD_PACK_K;
            PackCorrectionTile(slot, valid, k, len);
        }
    }

    __aicore__ inline void PackCorrectionTile(int32_t slot, int32_t valid,
                                            int32_t k, int32_t len)
    {
        auto out = outQueue_.AllocTensor<int8_t>();
        // Two signed INT4 -8 values per byte. cAux = (-8 * ones) @ W.
        AscendC::Duplicate(out.ReinterpretCast<int16_t>(), static_cast<int16_t>(-30584), len / 4);
        outQueue_.EnQue(out);
        auto ready = outQueue_.DeQue<int8_t>();
        AscendC::DataCopy(packed_[(static_cast<int64_t>(slot) * MSD_RAW_ROWS + 2 * valid) * K_ / 2 + k / 2], ready, len / 2);
        outQueue_.FreeTensor(ready);
    }

    __aicore__ inline void ReadRaw(int64_t offset, LocalTensor<float> dst)
    {
        auto buf = accGQueue_.AllocTensor<int32_t>();
        AscendC::DataCopy(buf, raw_[offset], N2_);
        accGQueue_.EnQue(buf);
        auto ready = accGQueue_.DeQue<int32_t>();
        AscendC::Cast(dst, ready, AscendC::RoundMode::CAST_NONE, N2_);
        accGQueue_.FreeTensor(ready);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Reconstruct(int32_t slot, int32_t row, int32_t valid,
        int32_t expert, int32_t col, LocalTensor<float> dst)
    {
        const int64_t base = static_cast<int64_t>(slot) * MSD_RAW_ROWS * N_ + col;
        ReadRaw(base + (2 * row + 1) * N_, dst);
        ReadRaw(base + 2 * valid * N_, t_);
        AscendC::Sub(dst, dst, t_, N2_);
        AscendC::PipeBarrier<PIPE_V>();
        ReadRaw(base + 2 * row * N_, t_);
        AscendC::Muls(t_, t_, 16.0f, N2_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(dst, dst, t_, N2_);
        AscendC::PipeBarrier<PIPE_V>();
        // low-aux is a dot product with unsigned low nibbles (magnitude <=120K).
        // Both 16*high and the final A8W4 dot have magnitude <=1024K.
        // K<=16384 keeps EVERY integer intermediate in [-2^24,2^24].
        // These FP32 operations reconstruct the INT32 result exactly; no
        // partial channel scaling or FP16 cast has occurred.
        GlobalTensor<uint64_t> scale;
        scale.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(ReadGmPtr(scPtrTblGM_, expert)));
        for (int32_t start = 0; start < N2_; start += 1024) {
            const int32_t len = N2_ - start < 1024 ? N2_ - start : 1024;
            auto scales = accUQueue_.AllocTensor<uint64_t>();
            AscendC::DataCopy(scales, scale[col + start], len);
            accUQueue_.EnQue(scales);
            auto ready = accUQueue_.DeQue<uint64_t>();
            uint64_t retained = 0;
            AscendC::GatherMask(t_, ready.ReinterpretCast<float>(), static_cast<uint8_t>(1),
                false, static_cast<uint32_t>(0), {1, static_cast<uint8_t>(len / 32), 8, 0}, retained);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(dst[start], dst[start], t_, len);
            accUQueue_.FreeTensor(ready);
            AscendC::PipeBarrier<PIPE_V>();
        }
        // The baseline's single int32*channelScale -> FP16 boundary is kept.
        AscendC::Cast(h16_, dst, AscendC::RoundMode::CAST_NONE, N2_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(dst, h16_, AscendC::RoundMode::CAST_NONE, N2_);
        AscendC::PipeBarrier<PIPE_V>();
    }

    GlobalTensor<int8_t> x_;
    GlobalTensor<int8_t> packed_;
    GlobalTensor<int32_t> raw_;
};
