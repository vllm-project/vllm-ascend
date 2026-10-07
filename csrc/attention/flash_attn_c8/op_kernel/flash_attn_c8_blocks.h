/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the software repository for the full text.
 */
#ifndef FLASH_ATTN_C8_BLOCKS_H
#define FLASH_ATTN_C8_BLOCKS_H

#include "flash_attn_c8_query_scale.h"
#include "../../a5_mla_common/op_kernel/arch35/c8_pipeline/flash_mla_c8_kernel.h"
#include "flash_attn_c8_vec128_packed_p.h"

namespace BaseApi {

using FlashAttnC8CubeBase = FAFullQuantMlaBlockCube<
    fp8_e4m3fn_t, float, FlashMlaC8Layout::LAYOUT_TND,
    S1TemplateType::Aligned128, S2TemplateType::Aligned128,
    static_cast<DTemplateType>(192), DTemplateType::Aligned128,
    true, 0, false, false, true, false>;

// Reuse decode's mixed FP8/BF16 QK pipeline, but expanded MLA has distinct
// K128 and V128 tensors. In particular, PV must never consume the K ring.
class FlashAttnC8BlockCube : public FlashAttnC8CubeBase {
public:
    static constexpr uint32_t L1_BYTES =
        3 * mBaseSize * s2BaseSize + 2 * mBaseSize * (128 + 64 * 2) +
        2 * s2BaseSize * (128 + 64 * 2) + 2 * s2BaseSize * 128;
    static_assert(L1_BYTES <= 512 * 1024, "C8 P/Q/K/V rings exceed L1");
    BuffersPolicyDB<BufferType::L1> l1ValueBuffers;

    __aicore__ inline explicit FlashAttnC8BlockCube(ConstInfoX &info)
        : FlashAttnC8CubeBase(info) {}

    __aicore__ inline void InitBuffers()
    {
        FlashAttnC8CubeBase::InitBuffers();
        l1ValueBuffers.Init(*l1BufferManagerPtr, s2BaseSize * dVBaseSize * sizeof(KV_T));
    }

    __aicore__ inline void FreeEventID()
    {
        l1ValueBuffers.Uninit(*l1BufferManagerPtr);
        FlashAttnC8CubeBase::FreeEventID();
    }

    __aicore__ inline void IterateBmm1(MM1_DBUF_T &outputBuf, RunInfoX &runInfo)
    {
        FlashAttnC8CubeBase::IterateBmm1(outputBuf, runInfo);
        // K's last reader is QK LoadData. Decode delays this token until PV
        // because K==V; doing that here would deadlock the two-slot K ring
        // when the scheduler preloads three C1 tiles before its first C2.
        l1KBuffers.GetPre().Set<HardEvent::MTE1_MTE2>();
    }

    __aicore__ inline void CopyValueTile(const LocalTensor<KV_T> &dst, RunInfoX &runInfo)
    {
        const uint32_t rows = runInfo.actSingleLoopS2Size;
        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{
            .tensor = dst, .rowCount = (rows + 31) / 32 * 32};
        GmKvCoord coord{
            .bIdx = runInfo.bIdx, .n2Idx = runInfo.n2Idx,
            .s2Idx = runInfo.s2Idx, .dIdx = 0,
            .s2DealSize = rows, .dDealSize = constInfo.dSizeV};
        copyKvGmToL1(l1Tensor, valueGm, coord);
    }

    __aicore__ inline void IterateBmm2(mm2ResPos &outputBuf, MM2_ABUF_POLICY_T &inputBuf,
                                      RunInfoX &runInfo)
    {
        auto value = l1ValueBuffers.Get();
        value.Wait<HardEvent::MTE1_MTE2>();
        CopyValueTile(value.GetTensor<KV_T>(), runInfo);
        value.Set<HardEvent::MTE2_MTE1>();

        // V owns a separate L1 ring. Start its DMA while vector softmax
        // prepares P, then wait for both operands before the PV matmul.
        MM2_ABUF_T probability = inputBuf.Get();
        probability.WaitCrossCore();
        outputBuf.WaitCrossCore();
        value.Wait<HardEvent::MTE2_MTE1>();

        auto result = mmL0CBuffers.Get();
        result.Wait<HardEvent::FIX_M>();
        MMParam param = MakeMMParam(mBaseSize, constInfo.dSizeV,
                                    runInfo.actSingleLoopS2Size, false, false);
        MatmulN<Q_T, KV_T, MLA_FULLQUANT_MM2_T, 128, 256, 128, ABLayout::MK, ABLayout::KN>(
            probability.GetTensor<Q_T>(), value.GetTensor<KV_T>(),
            mmL0ABuffers, mmL0BBuffers, result.GetTensor<MLA_FULLQUANT_MM2_T>(), param);
        value.Set<HardEvent::MTE1_MTE2>();
        result.Set<HardEvent::M_FIX>();
        result.Wait<HardEvent::M_FIX>();
        FixpipeMm2(outputBuf.GetTensor<MLA_FULLQUANT_MM2_T>(),
                   result.GetTensor<MLA_FULLQUANT_MM2_T>(), runInfo);
        result.Set<HardEvent::FIX_M>();
        outputBuf.SetCrossCore();
    }
};

template <bool hasMask>
using FlashAttnC8VecBase = FAFullQuantMlaBlockVec<
    fp8_e4m3fn_t, float, bfloat16_t,
    FlashMlaC8Layout::LAYOUT_TND, FlashMlaC8Layout::LAYOUT_TND,
    S1TemplateType::Aligned128, S2TemplateType::Aligned128,
    static_cast<DTemplateType>(192), DTemplateType::Aligned128,
    PseTypeEnum::PSE_NONE_TYPE, hasMask, false, true, 0,
    true, false, false, true, false>;

template <bool hasMask, bool packedP = false>
class FlashAttnC8BlockVec : public FlashAttnC8VecBase<hasMask> {
public:
    using Base = FlashAttnC8VecBase<hasMask>;
    using ConstInfoX = typename Base::ConstInfoX;

    __aicore__ inline explicit FlashAttnC8BlockVec(ConstInfoX &info) : Base(info) {}

    __aicore__ inline void InitBuffers()
    {
        // The decode block reserves P/output/mask space for 32 rows per AIV.
        // Expanded prefill uses M128 and therefore 64 rows on each AIV.
        constexpr uint32_t rows = Base::mBaseSize / CV_RATIO;
        constexpr uint32_t n = Base::s2BaseSize;
        constexpr uint32_t pScaleBytes = packedP ? 512 : 256;
        constexpr uint32_t crossCoreBytes = 3 * rows * n * sizeof(float);
        constexpr uint32_t vectorBytes = 9 * 256 + 2 * 2048 + 3 * 256 +
            rows * 128 * sizeof(float) + 2 * (rows + 1) * n + 512 +
            (hasMask ? rows * n : 0) + 2 * rows * sizeof(float) + 3 * pScaleBytes +
            rows * sizeof(float) * 8 + 128 + 64;
        static_assert(rows <= 64, "The softmax state and shared scratch hold 64 rows");
        static_assert(crossCoreBytes + vectorBytes <= 248 * 1024, "C8 buffers exceed AIV UB");
        this->SoftmaxInitBuffer();
        this->tPipe->InitBuffer(this->preLoopMaxBuf, 256);
        this->tPipe->InitBuffer(this->preLoopSumBuf, 256);
        this->tPipe->InitBuffer(this->firstLoopSumBuf, 256);
        this->tPipe->InitBuffer(this->stage2OutBuf, rows * 128 * sizeof(float));
        this->tPipe->InitBuffer(this->stage1OutQue[0], 1, (rows + 1) * n);
        this->tPipe->InitBuffer(this->stage1OutQue[1], 1, (rows + 1) * n);
        this->tPipe->InitBuffer(this->commonTBuf, 512);
        if constexpr (hasMask) {
            this->tPipe->InitBuffer(this->attenMaskInQue, 1, rows * n);
        }
        this->tPipe->InitBuffer(this->queryAntiqScaleInputQue[0], rows * sizeof(float));
        this->tPipe->InitBuffer(this->queryAntiqScaleInputQue[1], rows * sizeof(float));
        this->tPipe->InitBuffer(this->pScaleBuf[0], pScaleBytes);
        this->tPipe->InitBuffer(this->pScaleBuf[1], pScaleBytes);
        this->tPipe->InitBuffer(this->pScaleBuf[2], pScaleBytes);
        if (this->constInfo.isSoftmaxLseEnable) {
            this->tPipe->InitBuffer(this->softmaxLseQueue, 1, rows * sizeof(float) * 8);
        }
        constexpr int wideIndex = static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX);
        constexpr int narrowIndex = static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX);
        this->tPipe->InitBuffer(this->vselrIndexesBuf[wideIndex], 128);
        this->tPipe->InitBuffer(this->vselrIndexesBuf[narrowIndex], 64);
        auto indexes = this->vselrIndexesBuf[wideIndex].template Get<uint8_t>();
        for (uint32_t i = 0; i < 128; ++i) {
            indexes.SetValue(i, i << 1);
        }
        indexes = this->vselrIndexesBuf[narrowIndex].template Get<uint8_t>();
        for (uint32_t i = 0; i < 64; ++i) {
            indexes.SetValue(i, i << 2);
        }
    }

    __aicore__ inline void ProcessVec1(
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
        RunInfoX runInfo)
    {
        this->deScaleKValue = this->deScaleKGm.GetValue(runInfo.n2Idx);
        if constexpr (packedP) {
            bool fullValid = true;
            if constexpr (hasMask) {
                // Only right-down causal masking and gSize=1 are supported.
                // Later rows admit every key admitted by the first AIV row.
                const int64_t firstQuery = static_cast<int64_t>(runInfo.gS1Idx) + runInfo.vecMbaseIdx;
                const int64_t lastKey = static_cast<int64_t>(runInfo.s2Idx) + runInfo.actSingleLoopS2Size - 1;
                fullValid = lastKey <= firstQuery + runInfo.nextTokensLeftUp;
            }
            if (!runInfo.isFirstS2Loop && runInfo.actSingleLoopS2Size == 128 &&
                runInfo.actVecMSize != 0 && fullValid) {
                ProcessPackedUpdate(outputBuf, bmm1ResBuf, runInfo);
                return;
            }
        }
        Base::ProcessVec1(outputBuf, bmm1ResBuf, runInfo);
    }

    __aicore__ inline void ProcessPackedUpdate(
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
        RunInfoX runInfo)
    {
        bmm1ResBuf.WaitCrossCore();
        auto sum = this->softmaxSumBuf[runInfo.mloop % 3].template Get<float>();
        auto maximum = this->softmaxMaxBuf[runInfo.mloop % 3].template Get<float>();
        auto exponent = this->softmaxExpBuf[runInfo.loop % 3].template Get<float>();
        auto scratch = this->commonTBuf.template Get<uint8_t>();
        auto queryScale = this->queryAntiqScaleInputQue[runInfo.mloop % 2].template Get<float>();
        auto pScale = this->pScaleBuf[runInfo.loop % 3].template Get<float>();
        auto score = bmm1ResBuf.template GetTensor<float>();
        const uint32_t slot = runInfo.loop % 2;
        auto probability = this->stage1OutQue[slot].template AllocTensor<fp8_e4m3fn_t>();
        auto indexes = this->vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)]
            .template Get<uint8_t>();
        FaVectorApi::FlashAttnC8Update128PackedP(probability, indexes, score, maximum,
            scratch, pScale, runInfo.actVecMSize, this->constInfo.scaleValue,
            queryScale, this->deScaleKValue);
        bmm1ResBuf.SetCrossCore();

        this->stage1OutQue[slot].template EnQue(probability);
        this->stage1OutQue[slot].template DeQue<fp8_e4m3fn_t>();
        auto probabilityL1 = outputBuf.template GetTensor<fp8_e4m3fn_t>();
        const uint32_t offset = this->constInfo.subBlockIdx * (Base::mBaseSize * 16);
        DataCopy(probabilityL1[offset], probability,
            {Base::s2BaseSize / 32, static_cast<uint16_t>(runInfo.actVecMSize),
             static_cast<uint16_t>(Base::vec1Srcstride - runInfo.actVecMSize),
             static_cast<uint16_t>(Base::mBaseSize - runInfo.actVecMSize)});
        this->stage1OutQue[slot].template FreeTensor(probability);
        outputBuf.SetCrossCore();
        UpdateExpSumAndExpMax<float>(sum, maximum, exponent, sum, maximum,
                                     scratch, runInfo.actVecMSize);
        if (unlikely(runInfo.isLastS2Loop)) {
            SetFlag<HardEvent::V_MTE2>(this->vToMte2Id[runInfo.mloop % 2]);
            this->SoftmaxDataCopyOut(runInfo, sum, maximum);
        }
    }

    __aicore__ inline void ProcessVec2(typename Base::mm2ResPos &bmm2ResBuf, RunInfoX runInfo)
    {
        this->deScaleVValue = this->deScaleVGm.GetValue(runInfo.n2Idx);
        Base::ProcessVec2(bmm2ResBuf, runInfo);
    }
};

} // namespace BaseApi
#endif
