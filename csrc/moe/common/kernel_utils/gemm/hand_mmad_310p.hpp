/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */
#ifndef M200_HAND_MMAD_310P_HPP
#define M200_HAND_MMAD_310P_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"

namespace M200Gemm {

// Hand-written single-tile matmul for the 310P unified core.
//
// Replaces Catlass BlockMmadTla for the chunk_fwd_o tiles. Every tile here is
// m<=128, n<=128, k<=128, i.e. ONE L1 tile and ONE L0 tile, so BlockMmadTla's
// kL1Loop / mL0Loop / nL0Loop all evaluate to 1 and its preload branch, L1A/L1B/
// L0A/L0B stage rotation, ENABLE_L1_RESIDENT lastAddr+lastCoord comparisons and
// event-list indexing are dead code we still paid scalar cycles for -- scalar was
// 42.6% of kernel runtime. This is the same instruction sequence with the loops
// and the staging removed: GM->L1 x2, L1->L0A/L0B, Mmad, L0C->UB.
//
// Layout math is lifted verbatim from the Catlass path it replaces:
//   zN (L1)  : catlass/layout/matrix.hpp zN::MakeLayout
//              shape (16, ceil(M/16), 16, ceil(K/16))
//              stride(16, 256, 1, roundUp(M,16)*16)
//   zZ (L0A) : stride(16, roundUp(K,16)*16, 1, 256)
//   nZ (L0B) : stride( 1, roundUp(N,16)*16, 16, 256)
//   GM->L1   : gemm/tile/atlasa2/copy_gm_to_l1.hpp  (Nd2NzParams)
//   L1->L0A  : gemm/tile/atlasa2/copy_l1_to_l0a.hpp (zN -> zZ, ifTranspose=false)
//   L1->L0B  : gemm/tile/atlasa2/copy_l1_to_l0b.hpp (zN -> nZ, ifTranspose=true)
// fp16 only: ELE_NUM_PER_C0 = 16, ELE_NUM_PER_FRACTAL = 256, C0_NUM_PER_FRACTAL = 16.

static constexpr uint32_t HM_C0 = 16;      // elements per C0 block (fp16)
static constexpr uint32_t HM_FRAC = 256;   // elements per 16x16 fractal (fp16)

CATLASS_DEVICE constexpr uint32_t HmRoundUp16(uint32_t v) { return (v + 15) / 16 * 16; }

/// One tile of C = A @ B, landing in L0C[l0cOff] and then staged to UB[ubStageOff]
/// in NZ order. All offsets are bytes.
///
///   A : RowMajor [m, k], GM row stride lda.
///       With A_FROM_L1, gmA/lda are ignored and l1AOff must already hold the tile
///       in zN<half>(m, k) -- that is what Vec1's copy_ubuf_to_cbuf leaves behind,
///       since the cube's NZ staging order and zN(64,64) coincide.
///   B : RowMajor [k, n], GM row stride ldb.
///       With B_COL_MAJOR, B is ColumnMajor [k, n] with column stride ldb instead,
///       which is how q @ k^T reads k (stored [seqlen][kHeadDim] row-major, so
///       B[kk][j] == k[j][kk] == gm[j*ldb + kk]). Nd2Nz then lands it in nZ, and
///       L1->L0B is a straight nZ->nZ copy with ifTranspose = false.
///
/// Each cube passes its OWN l0cOff. L0C is 128 KB and the three tiles are
/// 16 + 32 + 32 = 80 KB, so they fit without sharing.
///   With A_COL_MAJOR, A is ColumnMajor [m, k] with column stride lda (i.e. the
///   GM holds A^T row-major, A[i][j] == gm[j*lda + i]) -- how k.T reads k in
///   fwd_h's h_work = k.T @ v_update. Nd2Nz of the stored [k, m] block lands zN;
///   the L0A load walks one m-block-row of fractals per repeat with
///   ifTranspose = true, mirroring the B_COL_MAJOR trick.
///   With B_FROM_L1, gmB/ldb are ignored and l1BOff must already hold the tile
///   in the zN layout the plain path's Nd2Nz would have produced (an earlier
///   body's GM->L1 load, a UB->L1 hand-off, or an L1-resident state).
///   With B_NZ_GM, gmB already holds the tile as a zN image (the cross-op h
///   format): the GM->L1 move is one flat burst, no Nd2Nz row walk.
template <class ArchTag, bool B_COL_MAJOR = false, bool A_FROM_L1 = false, bool A_COL_MAJOR = false,
          bool B_FROM_L1 = false, bool B_NZ_GM = false>
CATLASS_DEVICE void HandMmad(
    Catlass::Arch::Resource<ArchTag> &res,
    AscendC::GlobalTensor<half> const &gmA, uint32_t lda,
    AscendC::GlobalTensor<half> const &gmB, uint32_t ldb,
    uint32_t m, uint32_t n, uint32_t k,
    uint32_t l1AOff, uint32_t l1BOff, uint32_t ubStageOff, uint32_t l0cOff)
{
    auto l1A = res.l1Buf.template GetBufferByByte<half>(l1AOff);
    auto l1B = res.l1Buf.template GetBufferByByte<half>(l1BOff);
    auto l0a = res.l0ABuf.template GetBufferByByte<half>(0);
    auto l0b = res.l0BBuf.template GetBufferByByte<half>(0);
    auto l0c = res.l0CBuf.template GetBufferByByte<float>(l0cOff);

    static_assert(!(A_FROM_L1 && A_COL_MAJOR), "pick one A source");
    const uint32_t mR = HmRoundUp16(m), nR = HmRoundUp16(n), kR = HmRoundUp16(k);
    const uint32_t l1aC0Stride = A_COL_MAJOR ? kR : mR;    // zN of the STORED matrix
    const uint32_t l1bC0Stride = B_COL_MAJOR ? nR : kR;    // nZ(kR,nR).stride(1)/C0 : zN(kR,nR)

    // ---- GM -> L1 ----
    if constexpr (!A_FROM_L1) {
        AscendC::Nd2NzParams pa;
        pa.ndNum = 1;
        pa.nValue = A_COL_MAJOR ? k : m;
        pa.dValue = A_COL_MAJOR ? m : k;
        pa.srcNdMatrixStride = 0;
        pa.srcDValue = lda;  pa.dstNzC0Stride = l1aC0Stride;
        pa.dstNzNStride = 1;  pa.dstNzMatrixStride = 0;
        AscendC::DataCopy(l1A, gmA, pa);
    }
    if constexpr (B_NZ_GM) {
        static_assert(!B_COL_MAJOR && !B_FROM_L1, "B_NZ_GM is a plain zN image");
        AscendC::DataCopy(l1B, gmB, kR * nR);
    } else if constexpr (!B_FROM_L1) {
        // ColumnMajor source swaps the roles: dValue is the ROW count, nValue the
        // COLUMN count.
        AscendC::Nd2NzParams pb;
        pb.ndNum = 1;
        pb.nValue = B_COL_MAJOR ? n : k;
        pb.dValue = B_COL_MAJOR ? k : n;
        pb.srcNdMatrixStride = 0;
        pb.srcDValue = ldb;  pb.dstNzC0Stride = l1bC0Stride;
        pb.dstNzNStride = 1;  pb.dstNzMatrixStride = 0;
        AscendC::DataCopy(l1B, gmB, pb);
    }

    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID7);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID7);

    // ---- L1 -> L0 ----
    {   // -> zZ.  From zN of A it strides fractal columns; from zN of the stored
        // A^T (A_COL_MAJOR) each dst m-row of fractals is one contiguous src run
        // with a per-fractal transpose.
        AscendC::LoadData2DParams p;
        p.startIndex = 0;
        p.repeatTimes = static_cast<uint16_t>(kR / HM_C0);
        p.srcStride = A_COL_MAJOR ? 1 : (l1aC0Stride * HM_C0 / HM_FRAC);
        p.sid = 0;  p.dstGap = 0;  p.ifTranspose = A_COL_MAJOR;  p.addrMode = 0;
        const uint32_t dstRowStride = kR * HM_C0;
        const uint32_t srcRowStride = A_COL_MAJOR ? dstRowStride : HM_FRAC;
        for (uint32_t i = 0; i < mR / HM_C0; ++i) {
            AscendC::LoadData(l0a[i * dstRowStride], l1A[i * srcRowStride], p);
        }
    }
    {   // -> nZ.  From nZ (B_COL_MAJOR) it is a straight copy; from zN it transposes.
        AscendC::LoadData2DParams p;
        p.startIndex = 0;
        p.repeatTimes = static_cast<uint16_t>(nR / HM_C0);
        p.srcStride = B_COL_MAJOR ? 1 : (l1bC0Stride * HM_C0 / HM_FRAC);
        p.sid = 0;  p.dstGap = 0;  p.ifTranspose = !B_COL_MAJOR;  p.addrMode = 0;
        const uint32_t dstRowStride = nR * HM_C0;
        const uint32_t srcRowStride = B_COL_MAJOR ? dstRowStride : HM_FRAC;
        for (uint32_t i = 0; i < kR / HM_C0; ++i) {
            AscendC::LoadData(l0b[i * dstRowStride], l1B[i * srcRowStride], p);
        }
    }

    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID7);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID7);

    // ---- Mmad ----
    AscendC::MmadParams mp;
    mp.m = m;  mp.n = n;  mp.k = k;
    mp.unitFlag = 0;  mp.cmatrixInitVal = true;  mp.cmatrixSource = false;
    AscendC::Mmad(l0c, l0a, l0b, mp);

    // L0A/L0B are free once the mmad retires.
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);

    // ---- L0C -> UB (NZ), the same copy the Catlass block-out used ----
    // copy_matrix_cc_to_ubuf issues on the V pipe (kernel_event.h GetQueEvt, L0C->UB
    // on __NPU_ARCH__ 2002 maps to M_V / V_M), hence the M_V before and V_M after.
    AscendC::SetFlag<AscendC::HardEvent::M_V>(EVENT_ID7);
    AscendC::WaitFlag<AscendC::HardEvent::M_V>(EVENT_ID7);

    auto co2 = res.ubBuf.template GetBufferByByte<float>(ubStageOff);
    AscendC::DataCopyParams cp;
    cp.blockCount = static_cast<uint8_t>(nR / HM_C0);
    cp.blockLen = static_cast<uint16_t>(mR / HM_C0);
    cp.srcStride = 0;  cp.dstStride = 0;
    AscendC::DataCopyEnhancedParams eh;
    eh.blockMode = AscendC::BlockMode::BLOCK_MODE_MATRIX;
    AscendC::DataCopy(co2, l0c, cp, eh);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
    AscendC::SetFlag<AscendC::HardEvent::V_M>(EVENT_ID7);
    AscendC::WaitFlag<AscendC::HardEvent::V_M>(EVENT_ID7);
}

}  // namespace M200Gemm

#endif  // M200_HAND_MMAD_310P_HPP
