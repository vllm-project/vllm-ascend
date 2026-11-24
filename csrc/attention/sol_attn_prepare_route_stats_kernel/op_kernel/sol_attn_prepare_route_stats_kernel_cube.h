#pragma once
using namespace ascrip;

__aicore__ inline void sol_attn_prepare_route_stats_kernel_cube(GM_ADDR q_bar_, GM_ADDR kc_, GM_ADDR threshold_, GM_ADDR group_approx_mask_, GM_ADDR flat_exact_cnt_, GM_ADDR flat_exact_idx_, int32_t B, int32_t H, int32_t NT, int32_t NT_Q, int32_t NG, int32_t NPAD, int32_t NT_Q_PAD8, int32_t query_blocks_per_tile, int32_t sink_start, int32_t sink_end, float scale_log2, float scale_sq, float tau, GM_ADDR workspace)
{
    GMTensor<float> q_bar((__gm__ float*)q_bar_);
    GMTensor<bfloat16_t> kc((__gm__ bfloat16_t*)kc_);
    GMTensor<float> threshold((__gm__ float*)threshold_);
    GMTensor<uint64_t> group_approx_mask((__gm__ uint64_t*)group_approx_mask_);
    GMTensor<int32_t> flat_exact_cnt((__gm__ int32_t*)flat_exact_cnt_);
    GMTensor<uint16_t> flat_exact_idx((__gm__ uint16_t*)flat_exact_idx_);
    SetAtomicNone();
    SEvent<PIPE_M, PIPE_FIX, 0, 0> ev_m_fix_ready_1;  // guards result #511
    SEvent<PIPE_FIX, PIPE_M, 1, 0> ev_fix_m_valid_1;  // guards result #512
    SEvent<PIPE_MTE1, PIPE_M, 0, 0> ev_mte1_m_ready_1;  // guards l0_k #517
    SEvent<PIPE_M, PIPE_MTE1, 1, 0> ev_m_mte1_valid_1;  // guards l0_k #518
    SEvent<PIPE_MTE2, PIPE_MTE1, 0, 0> ev_mte2_mte1_ready_1;  // guards l1_k #523
    SEvent<PIPE_MTE1, PIPE_MTE2, 1, 0> ev_mte1_mte2_valid_1;  // guards l1_k #524
    const int32_t mul_1 = (B * H) * NPAD;  // #1 #2
    const int32_t mul_2 = mul_1 * 128;  // #3
    const GMTensor<float> key_f32((__gm__ float*)(workspace + (0)));  // #4
    const GMTensor<float> score_ws((__gm__ float*)(workspace + ((mul_2 * 4) + (mul_1 * 4))));  // #576 #577 #578 #8
    const Tensor<float, Position::L1> l1_q((uint64_t)(0));  // #9
    const Tensor<float, Position::L0A> l0_q((uint64_t)(0));  // #10
    const Buff<float, Position::L1, 2> l1_k((uint64_t)(16384), (uint64_t)(65536));  // #11
    const Tensor<float, Position::L0B> l0_k((uint64_t)(0));  // #12
    const Buff<float, Position::L0C, 2> result((uint64_t)(0), (uint64_t)(16384));  // #13
    SEvent<PIPE_MTE1, PIPE_MTE2, 1, 1> q_l1_free;  // #14
    SEvent<PIPE_MTE2, PIPE_MTE1, 0, 1> q_l1_ready;  // #15
    SEvent<PIPE_M, PIPE_MTE1, 1, 1> q_l0_free;  // #16
    SEvent<PIPE_MTE1, PIPE_M, 0, 1> q_l0_ready;  // #17
    // sync.mutex cv id=0: the producer side; its 2 tokens are drained before the return  // #18
    WAIT_VEC<PIPE_S>(7);  // #190
    const int32_t mul_6 = 2 * query_blocks_per_tile;  // #191
    const int32_t idiv_1 = 32 / mul_6;  // #694
    const int32_t idiv_2 = 32 % mul_6;  // #695
    const bool idiv_3 = idiv_2 != 0;  // #696
    const bool idiv_5 = mul_6 < 0;  // #698
    const bool idiv_6 = false != idiv_5;  // #699
    const bool idiv_7 = idiv_3 && idiv_6;  // #700
    const int32_t idiv_8 = (idiv_7) ? (1) : (0);  // #701
    const int32_t subblock_queries = idiv_1 - idiv_8;  // #192
    const int32_t mul_7 = 2 * subblock_queries;  // #193
    const int32_t idiv_9 = NT_Q_PAD8 / mul_7;  // #702
    const int32_t idiv_10 = NT_Q_PAD8 % mul_7;  // #703
    const bool idiv_11 = idiv_10 != 0;  // #704
    const bool idiv_13 = mul_7 < 0;  // #706
    const bool idiv_14 = false == idiv_13;  // #707
    const bool idiv_15 = idiv_11 && idiv_14;  // #708
    const int32_t idiv_16 = (idiv_15) ? (1) : (0);  // #709
    const int32_t q_chunks = idiv_9 + idiv_16;  // #194
    const int32_t group_id = GetCubeIdx();  // #199
    const int32_t mul_9 = (B * H) * q_chunks;  // #200 #201
    const int32_t cube_count = GetCubeNum();  // #202
    const int32_t cast_2 = (int32_t)(cube_count);  // #203
    int32_t chunk_step_3_state = 0;  // #461
    int32_t k_step_5_state = 0;  // #462
    for (int32_t task = group_id; task < mul_9; task += cast_2) {  // #503
        const int32_t chunk_step_carry = (int32_t)(chunk_step_3_state);  // #463
        const int32_t k_step_carry = (int32_t)(k_step_5_state);  // #464
        const int32_t idiv_17 = task / q_chunks;  // #710
        const int32_t idiv_18 = task % q_chunks;  // #711
        const bool idiv_19 = idiv_18 != 0;  // #712
        const bool idiv_20 = task < 0;  // #713
        const bool idiv_21 = q_chunks < 0;  // #714
        const bool idiv_22 = idiv_20 != idiv_21;  // #715
        const bool idiv_23 = idiv_19 && idiv_22;  // #716
        const int32_t idiv_24 = (idiv_23) ? (1) : (0);  // #717
        const int32_t bh_1 = idiv_17 - idiv_24;  // #204
        const int32_t idiv_25 = bh_1 / H;  // #718
        const int32_t idiv_26 = bh_1 % H;  // #719
        const bool idiv_27 = idiv_26 != 0;  // #720
        const bool idiv_28 = bh_1 < 0;  // #721
        const bool idiv_30 = idiv_28 != false;  // #723
        const bool idiv_31 = idiv_27 && idiv_30;  // #724
        const int32_t idiv_32 = (idiv_31) ? (1) : (0);  // #725
        const int32_t b_1 = idiv_25 - idiv_32;  // #205
        const int32_t h_1 = ascrip::FloorMod<int32_t>(bh_1, H);  // #206
        const int32_t tile_base = ascrip::FloorMod<int32_t>(task, q_chunks) * (2 * subblock_queries);  // #207 #208 #209
        const int32_t q_base = tile_base * query_blocks_per_tile;  // #210
        const int32_t q_rows = Min(32, NT - q_base);  // #211 #212
        q_l1_free.wait();  // #219
        if (q_rows < 32) {  // #220 #223
            set_constant_to_l1(l1_q, (float)(0.0f), 512);  // #221
            PipeBarrier<PIPE_MTE2>();  // #222
        }
        if (q_rows > 0) {  // #224 #229
            const GMTensor<float> view_54 = q_bar[(((b_1 * ((H * NT) * 128)) + (h_1 * (NT * 128))) + (q_base * 128))];  // #227
            gm_to_l1_nd2nz(l1_q, view_54, q_rows, 128, 32, 128);  // #228
        }
        q_l1_ready.set();  // #230
        q_l0_free.wait();  // #231
        q_l1_ready.wait();  // #232
        l1_to_l0<false>(l0_q, l1_q, 32, 128, 32, 128);  // #233
        q_l1_free.set();  // #234
        q_l0_ready.set();  // #235
        q_l0_ready.wait();  // #236
        int32_t chunk_step_2_state = chunk_step_carry;  // #466
        int32_t k_step_4_state = k_step_carry;  // #467
        for (int32_t first_group = 0; first_group < NG; first_group += 4) {  // #498
            const int32_t chunk_step_carry_1 = (int32_t)(chunk_step_2_state);  // #468
            const int32_t k_step_carry_1 = (int32_t)(k_step_4_state);  // #469
            const int32_t groups = Min(4, NG - first_group);  // #243 #244
            const int32_t start = first_group * 64;  // #245
            const int32_t end = start + (groups * 64);  // #246 #247
            WAIT_VEC<PIPE_S>(0);  // #248
            int32_t k_step_3_state = 0;  // #470
            int32_t k_step_2_state = k_step_carry_1;  // #471
            for (int32_t key = start; key < end; key += 128) {  // #474
                const int32_t k_step_carry_2 = (int32_t)(k_step_2_state);  // #472
                const int32_t n_keys = Min(128, end - key);  // #249 #250
                const Tensor<float, Position::L1> l1_keys = l1_k.get(k_step_carry_2);  // #251
                const Tensor<float, Position::L1> view_55 = l1_k.get(k_step_carry_2);  // #252
                const GMTensor<float> view_56 = key_f32[(((b_1 * ((H * NPAD) * 128)) + (h_1 * (NPAD * 128))) + (key * 128))];  // #255
                ev_mte1_mte2_valid_1.wait();  // #525
                gm_to_l1_nd2nz(view_55, view_56, n_keys, 128, 128, 128);  // #256
                const Tensor<float, Position::L0B> view_57 = l0_k;  // #257
                const Tensor<float, Position::L1> view_58 = l1_k.get(k_step_carry_2);  // #258
                ev_m_mte1_valid_1.wait();  // #519
                ev_mte2_mte1_ready_1.set();  // #526
                ev_mte2_mte1_ready_1.wait();  // #527
                l1_to_l0<false>(view_57, view_58, 128, 128, n_keys, 128);  // #259
                const Tensor<float, Position::L0C> acc = result.get(k_step_carry_2);  // #260
                const Tensor<float, Position::L0C> view_59 = result.get(k_step_carry_2);  // #261
                const Tensor<float, Position::L0B> view_60 = l0_k;  // #262
                ev_fix_m_valid_1.wait();  // #513
                ev_mte1_m_ready_1.set();  // #520
                ev_mte1_m_ready_1.wait();  // #521
                mmad(view_59, l0_q, view_60, 32, n_keys, 128, true);  // #263
                const int32_t col_1 = key - start;  // #264
                const int32_t group_id_1 = GetCubeIdx();  // #265
                const int32_t mod_1 = chunk_step_carry_1 & 1;  // #266
                const GMTensor<float> view_61 = score_ws[(((group_id_1 * 16384) + (mod_1 * 8192)) + col_1)];  // #269
                const Tensor<float, Position::L0C> view_62 = result.get(k_step_carry_2);  // #270
                ev_m_fix_ready_1.set();  // #514
                ev_m_fix_ready_1.wait();  // #515
                l0c_to_gm_nz2nd(view_61, view_62, 32, n_keys, 256, 32, false, 0.0f, 0, false, false);  // #271
                k_step_2_state = (int32_t)(k_step_carry_2 + 1);  // #272 #273 #473
                ev_fix_m_valid_1.set();  // #516
                ev_m_mte1_valid_1.set();  // #522
                ev_mte1_mte2_valid_1.set();  // #528
            }
            const int32_t k_step_2 = (int32_t)(k_step_2_state);  // #475
            CUBE_READY<PIPE_FIX>(0);  // #276
            k_step_3_state = k_step_2;  // #476
            const int32_t k_step_3 = (int32_t)(k_step_3_state);  // #477
            chunk_step_2_state = (int32_t)(chunk_step_carry_1 + 1);  // #434 #435 #496
            k_step_4_state = k_step_3;  // #497
        }
        const int32_t chunk_step_2 = (int32_t)(chunk_step_2_state);  // #499
        const int32_t k_step_4 = (int32_t)(k_step_4_state);  // #500
        q_l0_free.set();  // #439
        chunk_step_3_state = chunk_step_2;  // #501
        k_step_5_state = k_step_4;  // #502
    }
    WAIT_VEC<PIPE_S>(0);  // #455
    WAIT_VEC<PIPE_S>(0);  // #455
    return;  // #455
}
