// mega_kernel.cpp — GDN Mega-Kernel (group-value / GQA): all PTO stages in one launch
//
// Same pipeline as pto_mega_kernel, but scaled_dot_kkt / wy_fast / chunk_h / chunk_o use
// templates (H, Hg) from dynamic_bsnd_groupvalue; cumsum still uses H (value heads) like
// dynamic_bsnd.
//
// Stages:
//   1. cumsum      (Vec)
//   2. transpose   (Vec)
//   3. kkt         (Cube+Vec)  — K has Hg heads; β,g,A use H value heads
//   4. solve_tril  (Cube)
//   5. wy_fast     (Vec+Cube)
//   6. chunk_h     (Cube+Vec)
//   7. chunk_o     (Cube+Vec)

#ifndef GDN_H
#define GDN_H 16
#endif
#ifndef GDN_HG
#define GDN_HG GDN_H
#endif
#ifndef GDN_D
#define GDN_D 128
#endif
#ifndef GDN_C
#define GDN_C 128
#endif
#ifndef MEMORY_BASE
#define MEMORY_BASE
#endif

#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
#include <type_traits>
using namespace pto;

// ===================================================================
// Device-only helpers (shared with standard mega-kernel)
// ===================================================================
#ifdef __CCE_AICORE__

constexpr uint16_t SYNC_AIV_FLAG = 12;
constexpr uint16_t SYNC_AIC_FLAG = 11;
constexpr uint16_t SYNC_AIC_AIV_FLAG = 13;
constexpr uint16_t SYNC_AIV_ONLY_ALL = 14;
constexpr uint16_t SYNC_MODE_SHIFT_VALUE = 4;
constexpr uint16_t SYNC_FLAG_SHIFT_VALUE = 8;

AICORE inline uint16_t GetffstMsg(uint16_t mode, uint16_t flagId)
{
    return (0x1 + ((mode & 0x3) << SYNC_MODE_SHIFT_VALUE) +
            ((flagId & 0xf) << SYNC_FLAG_SHIFT_VALUE));
}

template <bool isAIVOnly = true>
AICORE inline void SyncAllImpl()
{
    pipe_barrier(PIPE_ALL);
    if constexpr (isAIVOnly) {
        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x0, SYNC_AIV_ONLY_ALL));
        wait_flag_dev(SYNC_AIV_ONLY_ALL);
        return;
    }
#if defined(__DAV_C220_CUBE__)
    wait_flag_dev(SYNC_AIV_FLAG);
    ffts_cross_core_sync(PIPE_FIX, GetffstMsg(0x0, SYNC_AIC_FLAG));
    wait_flag_dev(SYNC_AIC_FLAG);
    ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIC_AIV_FLAG));
#elif defined(__DAV_C220_VEC__)
    ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIV_FLAG));
    wait_flag_dev(SYNC_AIC_AIV_FLAG);
#endif
}

template <typename T, int32_t H_val>
AICORE void mega_transpose_TH_to_HT(
    __gm__ T *src, __gm__ T *dst, int64_t T_len)
{
#if defined(__DAV_C220_VEC__)
    if (get_subblockid() != 0) return;
    set_mask_norm();
    set_vector_mask(-1, -1);

    auto cid = get_block_idx();
    auto block_num = get_block_num();

    constexpr int32_t BLOCK = 128;
    constexpr int32_t H = static_cast<int32_t>(H_val);
    constexpr int32_t ES = static_cast<int32_t>(sizeof(T));
    constexpr int32_t SRC_UB = 0;
    constexpr int32_t DST_UB = SRC_UB + BLOCK * H * ES;
    constexpr int32_t TMP_UB = DST_UB + H * BLOCK * ES;

    using UBSrcFull = Tile<TileType::Vec, T, BLOCK, H, BLayout::RowMajor,
                           BLOCK, H, SLayout::NoneBox, 512, PadValue::Zero>;
    using UBSrcDyn  = Tile<TileType::Vec, T, BLOCK, H, BLayout::RowMajor,
                           DYNAMIC, DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
    using UBDst     = Tile<TileType::Vec, T, H, BLOCK, BLayout::RowMajor,
                           H, BLOCK, SLayout::NoneBox, 512>;
    using UBDstDyn  = Tile<TileType::Vec, T, H, BLOCK, BLayout::RowMajor,
                           DYNAMIC, DYNAMIC, SLayout::NoneBox, 512>;
    using UBTmp     = Tile<TileType::Vec, T, BLOCK, H, BLayout::RowMajor,
                           BLOCK, H, SLayout::NoneBox, 512>;

    using UBRow     = Tile<TileType::Vec, T, 1, BLOCK, BLayout::RowMajor,
                           1, BLOCK, SLayout::NoneBox, 512>;
    using UBRowDyn  = Tile<TileType::Vec, T, 1, BLOCK, BLayout::RowMajor,
                           DYNAMIC, DYNAMIC, SLayout::NoneBox, 512>;

    using Gm2D      = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using Gm1D      = Shape<1, 1, 1, 1, DYNAMIC>;
    using GmSrcS    = Stride<1, 1, 1, H, 1>;
    using GmS1      = Stride<1, 1, 1, 1, 1>;

    UBSrcFull ub_src; TASSIGN(ub_src, SRC_UB);
    UBDst     ub_dst; TASSIGN(ub_dst, DST_UB);
    UBTmp     ub_tmp; TASSIGN(ub_tmp, TMP_UB);

    int64_t num_tok_blocks = (T_len + BLOCK - 1) / BLOCK;

    for (int64_t bi = static_cast<int64_t>(cid); bi < num_tok_blocks;
         bi += static_cast<int64_t>(block_num)) {
        int64_t t0 = bi * BLOCK;
        int32_t valid = (t0 + BLOCK <= T_len)
                            ? BLOCK
                            : static_cast<int32_t>(T_len - t0);

        {
            Gm2D gs; gs.shape[3] = valid; gs.shape[4] = H;
            GlobalTensor<T, Gm2D, GmSrcS> gm(src + t0 * H, gs);
            UBSrcDyn ld(valid, H);
            TASSIGN(ld, SRC_UB);
            TLOAD(ld, gm);
            if (valid != BLOCK) TFILLPAD_INPLACE(ub_src, ld);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        TTRANS(ub_dst, ub_src, ub_tmp);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        for (int32_t h = 0; h < H; ++h) {
            Gm1D gs; gs.shape[4] = valid;
            GlobalTensor<T, Gm1D, GmS1> gm(dst + h * T_len + t0, gs);
            UBRowDyn st(1, valid);
            TASSIGN(st, DST_UB + h * BLOCK * ES);
            TSTORE(gm, st);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
#endif
}

template <int32_t H, int32_t C>
AICORE void mega_cast_fp32_to_fp16_bsnd(
    __gm__ float *src, __gm__ half *dst,
    uint32_t num_matrices, int64_t total_tokens)
{
#if defined(__DAV_C220_VEC__)
    if (get_subblockid() != 0) return;
    set_mask_norm();
    set_vector_mask(-1, -1);

    auto cid = get_block_idx();
    auto block_num = get_block_num();

    constexpr int32_t F32_UB = 0;
    constexpr int32_t F16_UB = C * static_cast<int32_t>(sizeof(float));

    using SrcUB    = Tile<TileType::Vec, float, 1, C, BLayout::RowMajor,
                          1, C, SLayout::NoneBox, 512, PadValue::Zero>;
    using DynSrcUB = Tile<TileType::Vec, float, 1, C, BLayout::RowMajor,
                          DYNAMIC, DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
    using DstUB    = Tile<TileType::Vec, half, 1, C, BLayout::RowMajor,
                          1, C, SLayout::NoneBox, 512>;
    using DynDstUB = Tile<TileType::Vec, half, 1, C, BLayout::RowMajor,
                          DYNAMIC, DYNAMIC, SLayout::NoneBox, 512>;
    using Gm1D     = Shape<1, 1, 1, 1, DYNAMIC>;
    using GmS1     = Stride<1, 1, 1, 1, 1>;

    SrcUB src_ub; TASSIGN(src_ub, F32_UB);
    DstUB dst_ub; TASSIGN(dst_ub, F16_UB);

    for (uint32_t m = cid; m < num_matrices; m += block_num) {
        uint32_t h = m % static_cast<uint32_t>(H);
        uint32_t chunk_idx = m / static_cast<uint32_t>(H);

        for (int64_t t = 0; t < total_tokens; ++t) {
            int64_t off = t * static_cast<int64_t>(H * C) +
                          static_cast<int64_t>(h * C);

            {
                Gm1D gs; gs.shape[4] = C;
                GlobalTensor<float, Gm1D, GmS1> gm(src + off, gs);
                SrcUB ld; TASSIGN(ld, F32_UB);
                TLOAD(ld, gm);
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            TCVT(dst_ub, src_ub, RoundMode::CAST_NONE);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            {
                Gm1D gs; gs.shape[4] = C;
                GlobalTensor<half, Gm1D, GmS1> gm(dst + off, gs);
                DstUB st; TASSIGN(st, F16_UB);
                TSTORE(gm, st);
            }
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
    }
#endif
}

#endif // __CCE_AICORE__

// ===================================================================
// Include original kernel implementations in separate namespaces.
// ===================================================================

#define call_kernel _mk_unused_gv_ck_cumsum
namespace mk_cumsum {
#include "chunk_cumsum.cpp"
}
#undef call_kernel

#define call_kernel _mk_unused_gv_ck_kkt
namespace mk_kkt {
#include "scaled_dot_kkt.cpp"
}
#undef call_kernel

namespace mk_solve {
#include "tri_inverse_impl.cpp"
}

#define call_kernel _mk_unused_gv_ck_wy
namespace mk_wy {
#include "wy_fast.cpp"
}
#undef call_kernel

#define call_kernel _mk_unused_gv_ck_h
namespace mk_h {
#include "chunk_h.cpp"
}
#undef call_kernel

#define call_kernel _mk_unused_gv_ck_o
namespace mk_o {
#include "chunk_o.cpp"
}
#undef call_kernel

AICORE void mega_solve_tril(
    __gm__ half *out, __gm__ half *in, __gm__ half *minus_id,
    uint32_t matrix_size, uint32_t num_matrices,
    uint32_t num_bsnd_heads,
    __gm__ int32_t *cu_seqlens, uint32_t is_lower)
{
    if (num_matrices <= get_block_num())
        mk_solve::runKernelTriInvRecUnroll<half, float, GDN_C, 1, true, half>(
            out, in, minus_id, num_matrices,
            num_bsnd_heads, cu_seqlens, is_lower);
    else if (num_matrices <= 2u * get_block_num())
        mk_solve::runKernelTriInvRecUnroll<half, float, GDN_C, 2, true, half>(
            out, in, minus_id, num_matrices,
            num_bsnd_heads, cu_seqlens, is_lower);
    else
        mk_solve::runKernelTriInvRecUnroll<half, float, GDN_C, 4, true, half>(
            out, in, minus_id, num_matrices,
            num_bsnd_heads, cu_seqlens, is_lower);
}

extern "C" __global__ AICORE void launch_mega_kernel(
    __gm__ uint8_t *q_ptr,
    __gm__ uint8_t *k_ptr,
    __gm__ uint8_t *v_ptr,
    __gm__ uint8_t *g_in_ptr,
    __gm__ uint8_t *beta_ptr,
    __gm__ uint8_t *msk_lower_ptr,
    __gm__ uint8_t *msk_full_ptr,
    __gm__ uint8_t *minus_id_ptr,
    __gm__ uint8_t *cu_seqlens_ptr,
    __gm__ uint8_t *o_ptr,
    __gm__ uint8_t *g_sum_ptr,
    __gm__ uint8_t *g_t_ptr,
    __gm__ uint8_t *beta_t_ptr,
    __gm__ uint8_t *A_ptr,
    __gm__ uint8_t *A_inv_f32_ptr,
    __gm__ uint8_t *A_inv_ptr,
    __gm__ uint8_t *w_ptr,
    __gm__ uint8_t *u_ptr,
    __gm__ uint8_t *s_ptr,
    __gm__ uint8_t *v_new_ptr,
    __gm__ uint8_t *fs_ptr,
    __gm__ uint8_t *kkt_ws_ptr,
    __gm__ uint8_t *wy_ws_a1_ptr,
    __gm__ uint8_t *wy_ws_a2_ptr,
    __gm__ uint8_t *h_ws_ptr,
    __gm__ uint8_t *o_ws_qk_ptr,
    __gm__ uint8_t *o_ws_qs_ptr,
    __gm__ uint8_t *o_ws_gated_ptr,
    int64_t batch_size,
    int64_t seq_len,
    int64_t total_tokens,
    uint32_t num_matrices,
    uint64_t ffts_addr)
{
    set_ffts_base_addr(ffts_addr);

    constexpr int32_t H = GDN_H;
    constexpr int32_t HG = GDN_HG;
    constexpr int32_t D = GDN_D;
    constexpr int32_t C = GDN_C;

    mk_cumsum::cumsum_kernel<H, C>(
        reinterpret_cast<__gm__ float *>(g_in_ptr),
        reinterpret_cast<__gm__ float *>(g_sum_ptr),
        reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr),
        batch_size, seq_len, ffts_addr);

#ifdef MEGA_STOP_AFTER_CUMSUM
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

#ifdef MEGA_STOP_AFTER_SYNC1
    return;
#endif

    mega_transpose_TH_to_HT<float, H>(
        reinterpret_cast<__gm__ float *>(g_sum_ptr),
        reinterpret_cast<__gm__ float *>(g_t_ptr),
        total_tokens);
    mega_transpose_TH_to_HT<half, H>(
        reinterpret_cast<__gm__ half *>(beta_ptr),
        reinterpret_cast<__gm__ half *>(beta_t_ptr),
        total_tokens);

#ifdef MEGA_STOP_AFTER_TRANSPOSE
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

    mk_kkt::kkt_kernel<H, HG, D, C>(
        reinterpret_cast<__gm__ half *>(k_ptr),
        reinterpret_cast<__gm__ half *>(beta_t_ptr),
        reinterpret_cast<__gm__ float *>(g_t_ptr),
        reinterpret_cast<__gm__ float *>(msk_lower_ptr),
        reinterpret_cast<__gm__ half *>(kkt_ws_ptr),
        reinterpret_cast<__gm__ half *>(A_ptr),
        reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr),
        batch_size, seq_len, total_tokens, ffts_addr);

#if defined(__DAV_C220_CUBE__)
    pipe_barrier(PIPE_ALL);
    wait_flag_dev(2);
    wait_flag_dev(3);
#endif

#ifdef MEGA_STOP_AFTER_KKT
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

    mega_solve_tril(
        reinterpret_cast<__gm__ half *>(A_inv_ptr),
        reinterpret_cast<__gm__ half *>(A_ptr),
        reinterpret_cast<__gm__ half *>(minus_id_ptr),
        C, num_matrices, H,
        reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), 1);

#ifdef MEGA_STOP_AFTER_SOLVE
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

#ifdef MEGA_STOP_AFTER_CAST
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

#ifdef MEGA_STOP_AFTER_SYNC_BEFORE_WY
    return;
#endif

    mk_wy::wy_fast_kernel<H, HG, D, C>(
        reinterpret_cast<__gm__ half *>(k_ptr),
        reinterpret_cast<__gm__ half *>(v_ptr),
        reinterpret_cast<__gm__ half *>(beta_t_ptr),
        reinterpret_cast<__gm__ float *>(g_t_ptr),
        reinterpret_cast<__gm__ half *>(A_inv_ptr),
        reinterpret_cast<__gm__ half *>(wy_ws_a1_ptr),
        reinterpret_cast<__gm__ half *>(wy_ws_a2_ptr),
        reinterpret_cast<__gm__ half *>(w_ptr),
        reinterpret_cast<__gm__ half *>(u_ptr),
        reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr),
        batch_size, seq_len, total_tokens, ffts_addr);

#if defined(__DAV_C220_VEC__)
    if (get_block_idx() < num_matrices) {
        pipe_barrier(PIPE_ALL);
        wait_flag_dev(3);
        wait_flag_dev(4);
    }
#endif

#ifdef MEGA_STOP_AFTER_WY
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

    mk_h::chunk_h_kernel<H, HG, D, C>(
        reinterpret_cast<__gm__ half *>(k_ptr),
        reinterpret_cast<__gm__ half *>(w_ptr),
        reinterpret_cast<__gm__ half *>(u_ptr),
        reinterpret_cast<__gm__ float *>(g_t_ptr),
        reinterpret_cast<__gm__ half *>(s_ptr),
        reinterpret_cast<__gm__ half *>(v_new_ptr),
        reinterpret_cast<__gm__ half *>(fs_ptr),
        reinterpret_cast<__gm__ half *>(h_ws_ptr),
        reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr),
        batch_size, seq_len, total_tokens, ffts_addr);

#ifdef MEGA_STOP_AFTER_H
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

    mk_o::chunk_o_kernel<H, HG, D, C>(
        reinterpret_cast<__gm__ half *>(q_ptr),
        reinterpret_cast<__gm__ half *>(k_ptr),
        reinterpret_cast<__gm__ half *>(v_new_ptr),
        reinterpret_cast<__gm__ half *>(s_ptr),
        reinterpret_cast<__gm__ float *>(g_t_ptr),
        reinterpret_cast<__gm__ float *>(msk_full_ptr),
        reinterpret_cast<__gm__ half *>(o_ws_qk_ptr),
        reinterpret_cast<__gm__ half *>(o_ws_qs_ptr),
        reinterpret_cast<__gm__ half *>(o_ws_gated_ptr),
        reinterpret_cast<__gm__ half *>(o_ptr),
        reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr),
        batch_size, seq_len, total_tokens, ffts_addr);

#if defined(__DAV_C220_CUBE__)
    if (get_block_idx() < num_matrices) {
        pipe_barrier(PIPE_ALL);
        wait_flag_dev(3);
    }
#endif
}

extern "C" void call_kernel(
    uint32_t block_dim, void *stream,
    uint8_t *q, uint8_t *k, uint8_t *v,
    uint8_t *g_in, uint8_t *beta,
    uint8_t *msk_lower, uint8_t *msk_full,
    uint8_t *minus_id, uint8_t *cu_seqlens,
    uint8_t *o,
    uint8_t *g_sum, uint8_t *g_t, uint8_t *beta_t,
    uint8_t *A, uint8_t *A_inv_f32, uint8_t *A_inv,
    uint8_t *w, uint8_t *u, uint8_t *s, uint8_t *v_new, uint8_t *fs,
    uint8_t *kkt_ws, uint8_t *wy_ws_a1, uint8_t *wy_ws_a2,
    uint8_t *h_ws,
    uint8_t *o_ws_qk, uint8_t *o_ws_qs, uint8_t *o_ws_gated,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens,
    uint32_t num_matrices)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    launch_mega_kernel<<<block_dim, nullptr, stream>>>(
        q, k, v, g_in, beta, msk_lower, msk_full, minus_id, cu_seqlens,
        o,
        g_sum, g_t, beta_t, A, A_inv_f32, A_inv,
        w, u, s, v_new, fs,
        kkt_ws, wy_ws_a1, wy_ws_a2, h_ws,
        o_ws_qk, o_ws_qs, o_ws_gated,
        batch_size, seq_len, total_tokens, num_matrices,
        fftsAddr);
}
