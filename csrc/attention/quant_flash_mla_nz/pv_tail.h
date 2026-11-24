// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
#pragma once

namespace BaseApi {
using FullNzCube = QuantFlashMlaBlockCubeFp8<fp8_e4m3fn_t, float, LayOutTypeEnum::LAYOUT_TND,
    S1TemplateType::Aligned64, S2TemplateType::Aligned256, DTemplateType::Aligned576, DTemplateType::Aligned512, 2>;

class DecodeNzCube : public FullNzCube {
public:
    using FullNzCube::FullNzCube;
    __aicore__ inline void IterateBmm2(RunInfoX& tile) {
        if (tile.actMSize != kRows) {
            FullNzCube::IterateBmm2(tile);
            return;
        }
        const uint32_t pSlot = tile.loop % L1_P_BUFCNT;
        const uint32_t resultSlot = tile.loop % UB_MM2_RES_BUFCNT;
        const uint32_t kvSlot = tile.kvL1BufId;
        const uint32_t l0cId = L0C_BUFFER_ID0 + l0cBufId_;
        auto p = l1PBuffers_[pSlot * L1_KV_BUF_BYTES + dVBaseSize * s2BaseSize].ReinterpretCast<Q_T>();
        auto v = l1KvBuffers_[kvSlot * L1_KV_BUF_BYTES].ReinterpretCast<Q_T>();
        auto c = l0CBuffers_[l0cBufId_ * L0C_BUF_BYTES].ReinterpretCast<float>();
        auto out = ubMmResBuffers_[UB_MM1_RES_BUFCNT * UB_MM1_RES_BUF_BYTES + resultSlot * UB_MM2_RES_BUF_BYTES]
                       .ReinterpretCast<float>();
        const uint32_t ready = CROSSCORE_L1P_0 + pSlot;
        const uint32_t free = CROSSCORE_L1P_FREE_0 + pSlot;
        const uint32_t done = CROSSCORE_MM_2 + resultSlot;
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(ready);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(ready + AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(done);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(done + AIV0_AIV1_OFFSET);
        Mutex::Lock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvSlot);
        Mutex::Lock<PIPE_M>(l0cId);
        Multiply(p, v, c, tile.actSingleLoopS2Size);
        Mutex::Unlock<PIPE_M>(l0cId);
        Mutex::Lock<PIPE_FIX>(l0cId);
        Mutex::Unlock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvSlot);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(free);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(free + AIV0_AIV1_OFFSET);

        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fix{};
        fix.nSize = dVBaseSize;
        fix.mSize = kRows;
        fix.srcStride = kRows;
        fix.dstStride = dVBaseSize;
        fix.dualDstCtl = 1;
        fix.params.ndNum = 1;
        Fixpipe<float, float, BMM2_FIXPIPE_CONFIG>(out, c, fix);
        Mutex::Unlock<PIPE_FIX>(l0cId);
        l0cBufId_ = (l0cBufId_ + 1) % L0C_BUFCNT;
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(done);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(done + AIV0_AIV1_OFFSET);
    }

private:
    static constexpr uint32_t kRows = 32;
    __aicore__ inline void Multiply(const LocalTensor<Q_T>& p, const LocalTensor<Q_T>& v,
                                    const LocalTensor<float>& c, uint32_t k) {
        auto a = l0ABuffers_[mBaseSize * dBaseSize].ReinterpretCast<Q_T>();
        Mutex::Lock<PIPE_MTE1>(L0A_BUFFER_ID1);
        // Each vector sub-block stores 16 active rows in its own 32-row half of P.
        // Gather the two NZ stripes into one 32-row L0A operand without rewriting shared L1.
        LoadData2DParamsV2 load{};
        load.mStep = 1;
        load.kStep = (k + 31) / 32;
        load.srcStride = mBaseSize / 16;
        load.dstStride = kRows / 16;
        LoadData(a, p, load);
        LoadData(a[16 * 32], p[32 * 32], load);
        Mutex::Unlock<PIPE_MTE1>(L0A_BUFFER_ID1);
        Mutex::Lock<PIPE_M>(L0A_BUFFER_ID1);
#pragma unroll
        for (uint32_t n = 0; n < 4; ++n) {
            const uint32_t id = L0B_BUFFER_ID0 + l0ABSlot_;
            auto b = l0BBuffers_[l0ABSlot_ * BUFFER_SIZE_BYTE_32K].ReinterpretCast<Q_T>();
            const uint32_t paddedK = (k + 31) / 32 * 32;
            Mutex::Lock<PIPE_MTE1>(id);
            LoadOperand(b, v[n * 128 * paddedK], paddedK, 128, true);
            Mutex::Unlock<PIPE_MTE1>(id);
            Mutex::Lock<PIPE_M>(id);
            MmadParams mm{};
            mm.m = kRows;
            mm.n = 128;
            mm.k = k;
            mm.cmatrixInitVal = true;
            Mmad(c[n * kRows * 128], a, b, mm);
            Mutex::Unlock<PIPE_M>(id);
            l0ABSlot_ ^= 1;
        }
        Mutex::Unlock<PIPE_M>(L0A_BUFFER_ID1);
    }
};
}  // namespace BaseApi
