#pragma once
#include "../../common/sol_attn/tensorutils_cce.h"
using namespace ascrip;

__aicore__ inline void sol_attn_fwd_m128_kernel_cube(GM_ADDR q_, GM_ADDR k_pad_, GM_ADDR v_pad_, GM_ADDR kc_padded_, GM_ADDR vc_padded_, GM_ADDR flat_exact_cnt_, GM_ADDR flat_exact_idx_, GM_ADDR group_approx_mask_, GM_ADDR last_group_lens_, GM_ADDR out_, GM_ADDR lse_, int32_t B, int32_t T, int32_t H, int32_t NT, int32_t NT_Q, int32_t NG, int32_t NPAD, int32_t T_PAD_KV, int32_t NT_Q_PAD8, int32_t T_LSE, float scale_log2, GM_ADDR workspace)
{
    GMTensor<bfloat16_t> q((__gm__ bfloat16_t*)q_);
    GMTensor<bfloat16_t> k_pad((__gm__ bfloat16_t*)k_pad_);
    GMTensor<bfloat16_t> v_pad((__gm__ bfloat16_t*)v_pad_);
    GMTensor<bfloat16_t> kc_padded((__gm__ bfloat16_t*)kc_padded_);
    GMTensor<bfloat16_t> vc_padded((__gm__ bfloat16_t*)vc_padded_);
    GMTensor<int32_t> flat_exact_cnt((__gm__ int32_t*)flat_exact_cnt_);
    GMTensor<uint16_t> flat_exact_idx((__gm__ uint16_t*)flat_exact_idx_);
    GMTensor<uint64_t> group_approx_mask((__gm__ uint64_t*)group_approx_mask_);
    GMTensor<float> last_group_lens((__gm__ float*)last_group_lens_);
    GMTensor<bfloat16_t> out((__gm__ bfloat16_t*)out_);
    GMTensor<float> lse((__gm__ float*)lse_);
    SetAtomicNone();
    const GMTensor<float> score_ws((__gm__ float*)(workspace + (0)));  // #2
    const GMTensor<bfloat16_t> p_ws((__gm__ bfloat16_t*)(workspace + (18874368)));  // #3
    const GMTensor<float> pv_ws((__gm__ float*)(workspace + (28311552)));  // #4
    const Buff<bfloat16_t, Position::L1, 1> l1q((uint64_t)(0), (uint64_t)(32768));  // #5
    const Buff<bfloat16_t, Position::L1, 2> l1k((uint64_t)(32768), (uint64_t)(32768));  // #6
    const Buff<bfloat16_t, Position::L1, 4> l1p((uint64_t)(98304), (uint64_t)(32768));  // #7
    const Buff<bfloat16_t, Position::L1, 4> l1v((uint64_t)(229376), (uint64_t)(32768));  // #8
    const Buff<float, Position::L0C, 2> l0c((uint64_t)(0), (uint64_t)(65536));  // #9
    const Buff<bfloat16_t, Position::L0A, 2> l0a((uint64_t)(0), (uint64_t)(32768));  // #10
    const Buff<bfloat16_t, Position::L0B, 2> l0b((uint64_t)(0), (uint64_t)(32768));  // #11
    // sync.mutex cv id=0: the producer side; its 2 tokens are drained before the return  // #33
    CUBE_READY<PIPE_MTE2>(1);  // #34
    CUBE_READY<PIPE_MTE2>(1);  // #34
    // sync.mutex cv id=2: the producer side; its 2 tokens are drained before the return  // #35
    DEvent<PIPE_MTE2, PIPE_MTE1, 0, 0, 1> k_local_ready;  // #36
    DEvent<PIPE_MTE1, PIPE_MTE2, 2, 0, 1> k_local_free;  // #37
    DEvent<PIPE_MTE2, PIPE_MTE1, 0, 2, 3> pv_local_ready;  // #38
    DEvent<PIPE_MTE1, PIPE_MTE2, 2, 2, 3> pv_local_free;  // #39
    DEvent<PIPE_MTE1, PIPE_M, 0, 0, 1> ab_local_ready;  // #40
    DEvent<PIPE_M, PIPE_MTE1, 2, 0, 1> ab_local_free;  // #41
    DEvent<PIPE_M, PIPE_FIX, 0, 0, 1> c_local_ready;  // #42
    DEvent<PIPE_FIX, PIPE_M, 2, 0, 1> c_local_free;  // #43
    const int32_t cube_idx = GetCubeIdx();  // #44
    const int32_t cube_count = GetCubeNum();  // #45
    const int32_t core_step = (int32_t)(cube_count);  // #46
    const int32_t bh_total = B * H;  // #49
    const int32_t total_tasks = bh_total * NT_Q;  // #50
    const int32_t idiv_1 = NG / 4;  // #3441
    const int32_t idiv_2 = NG % 4;  // #3442
    const bool idiv_3 = idiv_2 != 0;  // #3443
    const bool idiv_7 = idiv_3 && true;  // #3447
    const int32_t idiv_8 = (idiv_7) ? (1) : (0);  // #3448
    const int32_t num_abatches = idiv_1 + idiv_8;  // #51
    const int32_t full_waves_lim = (NT_Q / core_step) * core_step;  // #62 #63
    int32_t l0ab_cnt_32_state = 0;  // #2424
    int32_t l0c_cnt_25_state = 0;  // #2425
    int32_t l1k_cnt_18_state = 0;  // #2426
    int32_t l1p_cnt_18_state = 0;  // #2427
    int32_t l1q_cnt_2_state = 0;  // #2428
    int32_t l1v_cnt_18_state = 0;  // #2429
    for (int32_t linear_id = cube_idx; linear_id < total_tasks; linear_id += core_step) {  // #2760
        const int32_t l0ab_cnt_carry = (int32_t)(l0ab_cnt_32_state);  // #2430
        const int32_t l0c_cnt_carry = (int32_t)(l0c_cnt_25_state);  // #2431
        const int32_t l1k_cnt_carry = (int32_t)(l1k_cnt_18_state);  // #2432
        const int32_t l1p_cnt_carry = (int32_t)(l1p_cnt_18_state);  // #2433
        const int32_t l1q_cnt_carry = (int32_t)(l1q_cnt_2_state);  // #2434
        const int32_t l1v_cnt_carry = (int32_t)(l1v_cnt_18_state);  // #2435
        const int32_t idiv_9 = linear_id / NT_Q;  // #3449
        const int32_t idiv_10 = linear_id % NT_Q;  // #3450
        const bool idiv_11 = idiv_10 != 0;  // #3451
        const bool idiv_12 = linear_id < 0;  // #3452
        const bool idiv_14 = idiv_12 != false;  // #3454
        const bool idiv_15 = idiv_11 && idiv_14;  // #3455
        const int32_t idiv_16 = (idiv_15) ? (1) : (0);  // #3456
        const int32_t bh = idiv_9 - idiv_16;  // #64
        const int32_t r = ascrip::FloorMod<int32_t>(linear_id, NT_Q);  // #65
        const bool cmp = r < full_waves_lim;  // #66
        int32_t q_idx_1_state = 0;  // #2436
        if (cmp) {  // #76
            const int32_t idiv_17 = r / core_step;  // #3457
            const int32_t idiv_18 = r % core_step;  // #3458
            const bool idiv_19 = idiv_18 != 0;  // #3459
            const bool idiv_20 = r < 0;  // #3460
            const bool idiv_22 = idiv_20 != false;  // #3462
            const bool idiv_23 = idiv_19 && idiv_22;  // #3463
            const int32_t idiv_24 = (idiv_23) ? (1) : (0);  // #3464
            const int32_t wave = idiv_17 - idiv_24;  // #67
            const int32_t lane = ascrip::FloorMod<int32_t>(r, core_step);  // #68
            q_idx_1_state = (wave * core_step) + ascrip::FloorMod<int32_t>((lane + wave) + bh, core_step);  // #69 #70 #71 #72 #73 #2437
        } else {
            q_idx_1_state = r;  // #2438
        }
        const int32_t q_idx_1 = (int32_t)(q_idx_1_state);  // #2439
        const int32_t idiv_25 = bh / H;  // #3465
        const int32_t idiv_26 = bh % H;  // #3466
        const bool idiv_27 = idiv_26 != 0;  // #3467
        const bool idiv_28 = bh < 0;  // #3468
        const bool idiv_30 = idiv_28 != false;  // #3470
        const bool idiv_31 = idiv_27 && idiv_30;  // #3471
        const int32_t idiv_32 = (idiv_31) ? (1) : (0);  // #3472
        const int32_t b = idiv_25 - idiv_32;  // #77
        const int32_t h = ascrip::FloorMod<int32_t>(bh, H);  // #78
        const int32_t q_start = q_idx_1 * 128;  // #79
        const int32_t valid_m = Min(128, T - q_start);  // #80 #81
        if (valid_m < 128) {  // #85 #89
            const Tensor<bfloat16_t, Position::L1> slot = l1q.get(l1q_cnt_carry);  // #86
            set_constant_to_l1(slot, (bfloat16_t)(0.0f), 1024);  // #87
            PipeBarrier<PIPE_MTE2>();  // #88
        }
        const Tensor<bfloat16_t, Position::L1> slot_1 = l1q.get(l1q_cnt_carry);  // #90
        const GMTensor<bfloat16_t> view = q[(((b * ((T * H) * 128)) + (q_start * (H * 128))) + (h * 128))];  // #93
        gm_to_l1_nd2nz(slot_1, view, valid_m, 128, 128, H * 128);  // #2761 #94
        PipeBarrier<PIPE_ALL>();  // #99
        const int32_t lag_a = Min(3, num_abatches);  // #106
        const int32_t add_3 = num_abatches + lag_a;  // #107
        int32_t l0ab_cnt_13_state = l0ab_cnt_carry;  // #2440
        int32_t l0c_cnt_10_state = l0c_cnt_carry;  // #2441
        int32_t l1k_cnt_7_state = l1k_cnt_carry;  // #2442
        int32_t l1p_cnt_7_state = l1p_cnt_carry;  // #2443
        int32_t l1v_cnt_7_state = l1v_cnt_carry;  // #2444
        for (int32_t ab = 0; ab < add_3; ab += 1) {  // #2603
            const int32_t l0ab_cnt_carry_1 = (int32_t)(l0ab_cnt_13_state);  // #2445
            const int32_t l0c_cnt_carry_1 = (int32_t)(l0c_cnt_10_state);  // #2446
            const int32_t l1k_cnt_carry_1 = (int32_t)(l1k_cnt_7_state);  // #2447
            const int32_t l1p_cnt_carry_1 = (int32_t)(l1p_cnt_7_state);  // #2448
            const int32_t l1v_cnt_carry_1 = (int32_t)(l1v_cnt_7_state);  // #2449
            const bool cmp_2 = ab < num_abatches;  // #108
            int32_t l0ab_cnt_6_state = 0;  // #2450
            int32_t l0c_cnt_6_state = 0;  // #2451
            int32_t l1k_cnt_6_state = 0;  // #2452
            if (cmp_2) {  // #565
                const int32_t slot_a_p = ab & 3;  // #109
                const int32_t g_base_p = ab * 4;  // #110
                const uint64_t m0 = group_approx_mask.load((((b * ((H * NT_Q) * NG)) + (h * (NT_Q * NG))) + (q_idx_1 * NG)) + g_base_p);  // #112 #113 #114 #115 #116 #117 #118 #119 #120 #121
                const int32_t add_7 = g_base_p + 1;  // #122
                const bool cmp_3 = add_7 < NG;  // #123
                uint64_t m1_1_state = 0ULL;  // #2453
                if (cmp_3) {  // #138
                    m1_1_state = group_approx_mask.load((((b * ((H * NT_Q) * NG)) + (h * (NT_Q * NG))) + (q_idx_1 * NG)) + add_7);  // #126 #127 #128 #129 #130 #131 #132 #133 #134 #135 #2454
                } else {
                    m1_1_state = 0ULL;  // #2455
                }
                const uint64_t m1_1 = (uint64_t)(m1_1_state);  // #2456
                const int32_t add_12 = g_base_p + 2;  // #139
                const bool cmp_4 = add_12 < NG;  // #140
                uint64_t m2_1_state = 0ULL;  // #2457
                if (cmp_4) {  // #155
                    m2_1_state = group_approx_mask.load((((b * ((H * NT_Q) * NG)) + (h * (NT_Q * NG))) + (q_idx_1 * NG)) + add_12);  // #143 #144 #145 #146 #147 #148 #149 #150 #151 #152 #2458
                } else {
                    m2_1_state = 0ULL;  // #2459
                }
                const uint64_t m2_1 = (uint64_t)(m2_1_state);  // #2460
                const int32_t add_17 = g_base_p + 3;  // #156
                const bool cmp_5 = add_17 < NG;  // #157
                uint64_t m3_1_state = 0ULL;  // #2461
                if (cmp_5) {  // #172
                    m3_1_state = group_approx_mask.load((((b * ((H * NT_Q) * NG)) + (h * (NT_Q * NG))) + (q_idx_1 * NG)) + add_17);  // #160 #161 #162 #163 #164 #165 #166 #167 #168 #169 #2462
                } else {
                    m3_1_state = 0ULL;  // #2463
                }
                const uint64_t m3_1 = (uint64_t)(m3_1_state);  // #2464
                const bool has_p0_p = (m0 | m1_1) != 0ULL;  // #173 #174
                const bool has_p1_p = (m2_1 | m3_1) != 0ULL;  // #175 #176
                bool choice_state = false;  // #2465
                if (has_p0_p) {  // #179
                    choice_state = true;  // #2466
                } else {
                    choice_state = has_p1_p;  // #2467
                }
                const bool choice = (bool)(choice_state);  // #2468
                int32_t l0ab_cnt_5_state = 0;  // #2469
                int32_t l0c_cnt_5_state = 0;  // #2470
                int32_t l1k_cnt_5_state = 0;  // #2471
                if (choice) {  // #562
                    int32_t l0ab_cnt_2_state = 0;  // #2472
                    int32_t l0c_cnt_2_state = 0;  // #2473
                    int32_t l1k_cnt_2_state = 0;  // #2474
                    if (has_p0_p) {  // #232
                        k_local_free.wait();  // #181
                        const int32_t g0_off = g_base_p * 64;  // #182
                        if ((g_base_p + 1) < NG) {  // #183 #184 #199
                            const Tensor<bfloat16_t, Position::L1> slot_2 = l1k.get(l1k_cnt_carry_1);  // #185
                            const GMTensor<bfloat16_t> view_1 = kc_padded[(((b * ((H * NPAD) * 128)) + (h * (NPAD * 128))) + (g0_off * 128))];  // #188
                            gm_to_l1_nd2nz(slot_2, view_1, 128, 128, 128, 128);  // #189
                        } else {
                            const Tensor<bfloat16_t, Position::L1> slot_3 = l1k.get(l1k_cnt_carry_1);  // #190
                            set_constant_to_l1(slot_3, (bfloat16_t)(0.0f), 1024);  // #191
                            PipeBarrier<PIPE_MTE2>();  // #192
                            const Tensor<bfloat16_t, Position::L1> slot_4 = l1k.get(l1k_cnt_carry_1);  // #193
                            const Tensor<bfloat16_t, Position::L1> view_2 = l1k.get(l1k_cnt_carry_1);  // #194
                            const GMTensor<bfloat16_t> view_3 = kc_padded[(((b * ((H * NPAD) * 128)) + (h * (NPAD * 128))) + (g0_off * 128))];  // #197
                            gm_to_l1_nd2nz(view_2, view_3, 64, 128, 128, 128);  // #198
                        }
                        k_local_ready.set();  // #200
                        ab_local_free.wait();  // #201
                        k_local_ready.wait();  // #202
                        const Tensor<bfloat16_t, Position::L0A> slot_5 = l0a.get(l0ab_cnt_carry_1);  // #203
                        const Tensor<bfloat16_t, Position::L1> slot_6 = l1q.get(l1q_cnt_carry);  // #204
                        l1_to_l0<false>(slot_5, slot_6, 128, 128, 128, 128);  // #205
                        const Tensor<bfloat16_t, Position::L0B> slot_7 = l0b.get(l0ab_cnt_carry_1);  // #206
                        const Tensor<bfloat16_t, Position::L1> slot_8 = l1k.get(l1k_cnt_carry_1);  // #207
                        l1_to_l0<false>(slot_7, slot_8, 128, 128, 128, 128);  // #208
                        k_local_free.set();  // #209
                        ab_local_ready.set();  // #210
                        c_local_free.wait();  // #211
                        ab_local_ready.wait();  // #212
                        const Tensor<float, Position::L0C> slot_9 = l0c.get(l0c_cnt_carry_1);  // #213
                        const Tensor<bfloat16_t, Position::L0A> slot_10 = l0a.get(l0ab_cnt_carry_1);  // #214
                        const Tensor<bfloat16_t, Position::L0B> slot_11 = l0b.get(l0ab_cnt_carry_1);  // #215
                        mmad(slot_9, slot_10, slot_11, 128, 128, 128, true);  // #216
                        ab_local_free.set();  // #217
                        c_local_ready.set();  // #218
                        c_local_ready.wait();  // #219
                        const GMTensor<float> view_4 = score_ws[((cube_idx * 196608) + (slot_a_p * 49152))];  // #220
                        const Tensor<float, Position::L0C> slot_12 = l0c.get(l0c_cnt_carry_1);  // #221
                        l0c_to_gm_nz2nd(view_4, slot_12, 128, 128, 128, 128, false, 0.0f, 0, false, false);  // #222
                        c_local_free.set();  // #223
                        const int32_t l1k_cnt_1 = (int32_t)(l1k_cnt_carry_1 + 1);  // #224 #225
                        const int32_t l0ab_cnt_1 = (int32_t)(l0ab_cnt_carry_1 + 1);  // #226 #227
                        const int32_t l0c_cnt_1 = (int32_t)(l0c_cnt_carry_1 + 1);  // #228 #229
                        l0ab_cnt_2_state = l0ab_cnt_1;  // #2475
                        l0c_cnt_2_state = l0c_cnt_1;  // #2476
                        l1k_cnt_2_state = l1k_cnt_1;  // #2477
                    } else {
                        l0ab_cnt_2_state = l0ab_cnt_carry_1;  // #2478
                        l0c_cnt_2_state = l0c_cnt_carry_1;  // #2479
                        l1k_cnt_2_state = l1k_cnt_carry_1;  // #2480
                    }
                    const int32_t l0ab_cnt_2 = (int32_t)(l0ab_cnt_2_state);  // #2481
                    const int32_t l0c_cnt_2 = (int32_t)(l0c_cnt_2_state);  // #2482
                    const int32_t l1k_cnt_2 = (int32_t)(l1k_cnt_2_state);  // #2483
                    int32_t l0ab_cnt_4_state = 0;  // #2484
                    int32_t l0c_cnt_4_state = 0;  // #2485
                    int32_t l1k_cnt_4_state = 0;  // #2486
                    if (has_p1_p) {  // #286
                        k_local_free.wait();  // #234
                        const int32_t g2_off = (g_base_p + 2) * 64;  // #235 #236
                        if ((g_base_p + 3) < NG) {  // #237 #238 #253
                            const Tensor<bfloat16_t, Position::L1> slot_13 = l1k.get(l1k_cnt_2);  // #239
                            const GMTensor<bfloat16_t> view_5 = kc_padded[(((b * ((H * NPAD) * 128)) + (h * (NPAD * 128))) + (g2_off * 128))];  // #242
                            gm_to_l1_nd2nz(slot_13, view_5, 128, 128, 128, 128);  // #243
                        } else {
                            const Tensor<bfloat16_t, Position::L1> slot_14 = l1k.get(l1k_cnt_2);  // #244
                            set_constant_to_l1(slot_14, (bfloat16_t)(0.0f), 1024);  // #245
                            PipeBarrier<PIPE_MTE2>();  // #246
                            const Tensor<bfloat16_t, Position::L1> slot_15 = l1k.get(l1k_cnt_2);  // #247
                            const Tensor<bfloat16_t, Position::L1> view_6 = l1k.get(l1k_cnt_2);  // #248
                            const GMTensor<bfloat16_t> view_7 = kc_padded[(((b * ((H * NPAD) * 128)) + (h * (NPAD * 128))) + (g2_off * 128))];  // #251
                            gm_to_l1_nd2nz(view_6, view_7, 64, 128, 128, 128);  // #252
                        }
                        k_local_ready.set();  // #254
                        ab_local_free.wait();  // #255
                        k_local_ready.wait();  // #256
                        const Tensor<bfloat16_t, Position::L0A> slot_16 = l0a.get(l0ab_cnt_2);  // #257
                        const Tensor<bfloat16_t, Position::L1> slot_17 = l1q.get(l1q_cnt_carry);  // #258
                        l1_to_l0<false>(slot_16, slot_17, 128, 128, 128, 128);  // #259
                        const Tensor<bfloat16_t, Position::L0B> slot_18 = l0b.get(l0ab_cnt_2);  // #260
                        const Tensor<bfloat16_t, Position::L1> slot_19 = l1k.get(l1k_cnt_2);  // #261
                        l1_to_l0<false>(slot_18, slot_19, 128, 128, 128, 128);  // #262
                        k_local_free.set();  // #263
                        ab_local_ready.set();  // #264
                        c_local_free.wait();  // #265
                        ab_local_ready.wait();  // #266
                        const Tensor<float, Position::L0C> slot_20 = l0c.get(l0c_cnt_2);  // #267
                        const Tensor<bfloat16_t, Position::L0A> slot_21 = l0a.get(l0ab_cnt_2);  // #268
                        const Tensor<bfloat16_t, Position::L0B> slot_22 = l0b.get(l0ab_cnt_2);  // #269
                        mmad(slot_20, slot_21, slot_22, 128, 128, 128, true);  // #270
                        ab_local_free.set();  // #271
                        c_local_ready.set();  // #272
                        c_local_ready.wait();  // #273
                        const GMTensor<float> view_8 = score_ws[(((cube_idx * 196608) + (slot_a_p * 49152)) + 16384)];  // #274
                        const Tensor<float, Position::L0C> slot_23 = l0c.get(l0c_cnt_2);  // #275
                        l0c_to_gm_nz2nd(view_8, slot_23, 128, 128, 128, 128, false, 0.0f, 0, false, false);  // #276
                        c_local_free.set();  // #277
                        const int32_t l1k_cnt_3 = (int32_t)(l1k_cnt_2 + 1);  // #278 #279
                        const int32_t l0ab_cnt_3 = (int32_t)(l0ab_cnt_2 + 1);  // #280 #281
                        const int32_t l0c_cnt_3 = (int32_t)(l0c_cnt_2 + 1);  // #282 #283
                        l0ab_cnt_4_state = l0ab_cnt_3;  // #2487
                        l0c_cnt_4_state = l0c_cnt_3;  // #2488
                        l1k_cnt_4_state = l1k_cnt_3;  // #2489
                    } else {
                        l0ab_cnt_4_state = l0ab_cnt_2;  // #2490
                        l0c_cnt_4_state = l0c_cnt_2;  // #2491
                        l1k_cnt_4_state = l1k_cnt_2;  // #2492
                    }
                    const int32_t l0ab_cnt_4 = (int32_t)(l0ab_cnt_4_state);  // #2493
                    const int32_t l0c_cnt_4 = (int32_t)(l0c_cnt_4_state);  // #2494
                    const int32_t l1k_cnt_4 = (int32_t)(l1k_cnt_4_state);  // #2495
                    CUBE_READY<PIPE_FIX>(0);  // #288
                    l0ab_cnt_5_state = l0ab_cnt_4;  // #2508
                    l0c_cnt_5_state = l0c_cnt_4;  // #2509
                    l1k_cnt_5_state = l1k_cnt_4;  // #2510
                } else {
                    l0ab_cnt_5_state = l0ab_cnt_carry_1;  // #2511
                    l0c_cnt_5_state = l0c_cnt_carry_1;  // #2512
                    l1k_cnt_5_state = l1k_cnt_carry_1;  // #2513
                }
                const int32_t l0ab_cnt_5 = (int32_t)(l0ab_cnt_5_state);  // #2514
                const int32_t l0c_cnt_5 = (int32_t)(l0c_cnt_5_state);  // #2515
                const int32_t l1k_cnt_5 = (int32_t)(l1k_cnt_5_state);  // #2516
                l0ab_cnt_6_state = l0ab_cnt_5;  // #2517
                l0c_cnt_6_state = l0c_cnt_5;  // #2518
                l1k_cnt_6_state = l1k_cnt_5;  // #2519
            } else {
                l0ab_cnt_6_state = l0ab_cnt_carry_1;  // #2520
                l0c_cnt_6_state = l0c_cnt_carry_1;  // #2521
                l1k_cnt_6_state = l1k_cnt_carry_1;  // #2522
            }
            const int32_t l0ab_cnt_6 = (int32_t)(l0ab_cnt_6_state);  // #2523
            const int32_t l0c_cnt_6 = (int32_t)(l0c_cnt_6_state);  // #2524
            const int32_t l1k_cnt_6 = (int32_t)(l1k_cnt_6_state);  // #2525
            const bool cmp_20 = ab >= lag_a;  // #566
            int32_t l0ab_cnt_12_state = 0;  // #2526
            int32_t l0c_cnt_9_state = 0;  // #2527
            int32_t l1p_cnt_6_state = 0;  // #2528
            int32_t l1v_cnt_6_state = 0;  // #2529
            if (cmp_20) {  // #783
                const int32_t ac = ab - lag_a;  // #567
                const int32_t slot_a_c = ac & 3;  // #568
                const int32_t gc_base = ac * 4;  // #569
                const uint64_t mc0 = group_approx_mask.load((((b * ((H * NT_Q) * NG)) + (h * (NT_Q * NG))) + (q_idx_1 * NG)) + gc_base);  // #571 #572 #573 #574 #575 #576 #577 #578 #579 #580
                const int32_t add_46 = gc_base + 1;  // #581
                const bool cmp_21 = add_46 < NG;  // #582
                uint64_t mc1_1_state = 0ULL;  // #2530
                if (cmp_21) {  // #597
                    mc1_1_state = group_approx_mask.load((((b * ((H * NT_Q) * NG)) + (h * (NT_Q * NG))) + (q_idx_1 * NG)) + add_46);  // #585 #586 #587 #588 #589 #590 #591 #592 #593 #594 #2531
                } else {
                    mc1_1_state = 0ULL;  // #2532
                }
                const uint64_t mc1_1 = (uint64_t)(mc1_1_state);  // #2533
                const int32_t add_51 = gc_base + 2;  // #598
                const bool cmp_22 = add_51 < NG;  // #599
                uint64_t mc2_1_state = 0ULL;  // #2534
                if (cmp_22) {  // #614
                    mc2_1_state = group_approx_mask.load((((b * ((H * NT_Q) * NG)) + (h * (NT_Q * NG))) + (q_idx_1 * NG)) + add_51);  // #602 #603 #604 #605 #606 #607 #608 #609 #610 #611 #2535
                } else {
                    mc2_1_state = 0ULL;  // #2536
                }
                const uint64_t mc2_1 = (uint64_t)(mc2_1_state);  // #2537
                const int32_t add_56 = gc_base + 3;  // #615
                const bool cmp_23 = add_56 < NG;  // #616
                uint64_t mc3_1_state = 0ULL;  // #2538
                if (cmp_23) {  // #631
                    mc3_1_state = group_approx_mask.load((((b * ((H * NT_Q) * NG)) + (h * (NT_Q * NG))) + (q_idx_1 * NG)) + add_56);  // #619 #620 #621 #622 #623 #624 #625 #626 #627 #628 #2539
                } else {
                    mc3_1_state = 0ULL;  // #2540
                }
                const uint64_t mc3_1 = (uint64_t)(mc3_1_state);  // #2541
                const bool has_p0_c = (mc0 | mc1_1) != 0ULL;  // #632 #633
                const bool has_p1_c = (mc2_1 | mc3_1) != 0ULL;  // #634 #635
                bool choice_4_state = false;  // #2542
                if (has_p0_c) {  // #638
                    choice_4_state = true;  // #2543
                } else {
                    choice_4_state = has_p1_c;  // #2544
                }
                const bool choice_4 = (bool)(choice_4_state);  // #2545
                int32_t l0ab_cnt_11_state = 0;  // #2546
                int32_t l0c_cnt_8_state = 0;  // #2547
                int32_t l1p_cnt_5_state = 0;  // #2548
                int32_t l1v_cnt_5_state = 0;  // #2549
                if (choice_4) {  // #780
                    WAIT_VEC<PIPE_S>(1);  // #640
                    c_local_free.wait();  // #641
                    int32_t l0ab_cnt_8_state = 0;  // #2550
                    int32_t l1p_cnt_2_state = 0;  // #2551
                    int32_t l1v_cnt_2_state = 0;  // #2552
                    if (has_p0_c) {  // #690
                        pv_local_free.wait();  // #643
                        const int32_t gc0_off = gc_base * 64;  // #644
                        const Tensor<bfloat16_t, Position::L1> slot_24 = l1p.get(l1p_cnt_carry_1);  // #645
                        const GMTensor<bfloat16_t> view_53 = p_ws[((cube_idx * 196608) + (slot_a_c * 49152))];  // #646
                        gm_to_l1_nd2nz(slot_24, view_53, 128, 128, 128, 128);  // #647
                        if ((gc_base + 1) < NG) {  // #648 #649 #664
                            const Tensor<bfloat16_t, Position::L1> slot_25 = l1v.get(l1v_cnt_carry_1);  // #650
                            const GMTensor<bfloat16_t> view_54 = vc_padded[(((b * ((H * NPAD) * 128)) + (h * (NPAD * 128))) + (gc0_off * 128))];  // #653
                            gm_to_l1_nd2nz(slot_25, view_54, 128, 128, 128, 128);  // #654
                        } else {
                            const Tensor<bfloat16_t, Position::L1> slot_26 = l1v.get(l1v_cnt_carry_1);  // #655
                            set_constant_to_l1(slot_26, (bfloat16_t)(0.0f), 1024);  // #656
                            PipeBarrier<PIPE_MTE2>();  // #657
                            const Tensor<bfloat16_t, Position::L1> slot_27 = l1v.get(l1v_cnt_carry_1);  // #658
                            const Tensor<bfloat16_t, Position::L1> view_55 = l1v.get(l1v_cnt_carry_1);  // #659
                            const GMTensor<bfloat16_t> view_56 = vc_padded[(((b * ((H * NPAD) * 128)) + (h * (NPAD * 128))) + (gc0_off * 128))];  // #662
                            gm_to_l1_nd2nz(view_55, view_56, 64, 128, 128, 128);  // #663
                        }
                        pv_local_ready.set();  // #665
                        ab_local_free.wait();  // #666
                        pv_local_ready.wait();  // #667
                        const Tensor<bfloat16_t, Position::L0A> slot_28 = l0a.get(l0ab_cnt_6);  // #668
                        const Tensor<bfloat16_t, Position::L1> slot_29 = l1p.get(l1p_cnt_carry_1);  // #669
                        l1_to_l0<false>(slot_28, slot_29, 128, 128, 128, 128);  // #670
                        const Tensor<bfloat16_t, Position::L0B> slot_30 = l0b.get(l0ab_cnt_6);  // #671
                        const Tensor<bfloat16_t, Position::L1> slot_31 = l1v.get(l1v_cnt_carry_1);  // #672
                        l1_to_l0<true>(slot_30, slot_31, 128, 128, 128, 128);  // #673
                        pv_local_free.set();  // #674
                        ab_local_ready.set();  // #675
                        ab_local_ready.wait();  // #676
                        const Tensor<float, Position::L0C> slot_32 = l0c.get(l0c_cnt_6);  // #677
                        const Tensor<bfloat16_t, Position::L0A> slot_33 = l0a.get(l0ab_cnt_6);  // #678
                        const Tensor<bfloat16_t, Position::L0B> slot_34 = l0b.get(l0ab_cnt_6);  // #679
                        mmad(slot_32, slot_33, slot_34, 128, 128, 128, true);  // #680
                        ab_local_free.set();  // #681
                        const int32_t l0ab_cnt_7 = (int32_t)(l0ab_cnt_6 + 1);  // #682 #683
                        const int32_t l1p_cnt_1 = (int32_t)(l1p_cnt_carry_1 + 1);  // #684 #685
                        const int32_t l1v_cnt_1 = (int32_t)(l1v_cnt_carry_1 + 1);  // #686 #687
                        l0ab_cnt_8_state = l0ab_cnt_7;  // #2553
                        l1p_cnt_2_state = l1p_cnt_1;  // #2554
                        l1v_cnt_2_state = l1v_cnt_1;  // #2555
                    } else {
                        l0ab_cnt_8_state = l0ab_cnt_6;  // #2556
                        l1p_cnt_2_state = l1p_cnt_carry_1;  // #2557
                        l1v_cnt_2_state = l1v_cnt_carry_1;  // #2558
                    }
                    const int32_t l0ab_cnt_8 = (int32_t)(l0ab_cnt_8_state);  // #2559
                    const int32_t l1p_cnt_2 = (int32_t)(l1p_cnt_2_state);  // #2560
                    const int32_t l1v_cnt_2 = (int32_t)(l1v_cnt_2_state);  // #2561
                    int32_t l0ab_cnt_10_state = 0;  // #2562
                    int32_t l1p_cnt_4_state = 0;  // #2563
                    int32_t l1v_cnt_4_state = 0;  // #2564
                    if (has_p1_c) {  // #746
                        pv_local_free.wait();  // #692
                        const int32_t gc2_off = (gc_base + 2) * 64;  // #693 #694
                        const Tensor<bfloat16_t, Position::L1> slot_35 = l1p.get(l1p_cnt_2);  // #695
                        const GMTensor<bfloat16_t> view_57 = p_ws[(((cube_idx * 196608) + (slot_a_c * 49152)) + 16384)];  // #696
                        gm_to_l1_nd2nz(slot_35, view_57, 128, 128, 128, 128);  // #697
                        if ((gc_base + 3) < NG) {  // #698 #699 #714
                            const Tensor<bfloat16_t, Position::L1> slot_36 = l1v.get(l1v_cnt_2);  // #700
                            const GMTensor<bfloat16_t> view_58 = vc_padded[(((b * ((H * NPAD) * 128)) + (h * (NPAD * 128))) + (gc2_off * 128))];  // #703
                            gm_to_l1_nd2nz(slot_36, view_58, 128, 128, 128, 128);  // #704
                        } else {
                            const Tensor<bfloat16_t, Position::L1> slot_37 = l1v.get(l1v_cnt_2);  // #705
                            set_constant_to_l1(slot_37, (bfloat16_t)(0.0f), 1024);  // #706
                            PipeBarrier<PIPE_MTE2>();  // #707
                            const Tensor<bfloat16_t, Position::L1> slot_38 = l1v.get(l1v_cnt_2);  // #708
                            const Tensor<bfloat16_t, Position::L1> view_59 = l1v.get(l1v_cnt_2);  // #709
                            const GMTensor<bfloat16_t> view_60 = vc_padded[(((b * ((H * NPAD) * 128)) + (h * (NPAD * 128))) + (gc2_off * 128))];  // #712
                            gm_to_l1_nd2nz(view_59, view_60, 64, 128, 128, 128);  // #713
                        }
                        pv_local_ready.set();  // #715
                        ab_local_free.wait();  // #716
                        pv_local_ready.wait();  // #717
                        const Tensor<bfloat16_t, Position::L0A> slot_39 = l0a.get(l0ab_cnt_8);  // #718
                        const Tensor<bfloat16_t, Position::L1> slot_40 = l1p.get(l1p_cnt_2);  // #719
                        l1_to_l0<false>(slot_39, slot_40, 128, 128, 128, 128);  // #720
                        const Tensor<bfloat16_t, Position::L0B> slot_41 = l0b.get(l0ab_cnt_8);  // #721
                        const Tensor<bfloat16_t, Position::L1> slot_42 = l1v.get(l1v_cnt_2);  // #722
                        l1_to_l0<true>(slot_41, slot_42, 128, 128, 128, 128);  // #723
                        pv_local_free.set();  // #724
                        ab_local_ready.set();  // #725
                        ab_local_ready.wait();  // #726
                        if (has_p0_c) {  // #736
                            PipeBarrier<PIPE_M>();  // #727
                            const Tensor<float, Position::L0C> slot_43 = l0c.get(l0c_cnt_6);  // #728
                            const Tensor<bfloat16_t, Position::L0A> slot_44 = l0a.get(l0ab_cnt_8);  // #729
                            const Tensor<bfloat16_t, Position::L0B> slot_45 = l0b.get(l0ab_cnt_8);  // #730
                            mmad(slot_43, slot_44, slot_45, 128, 128, 128, false);  // #731
                        } else {
                            const Tensor<float, Position::L0C> slot_46 = l0c.get(l0c_cnt_6);  // #732
                            const Tensor<bfloat16_t, Position::L0A> slot_47 = l0a.get(l0ab_cnt_8);  // #733
                            const Tensor<bfloat16_t, Position::L0B> slot_48 = l0b.get(l0ab_cnt_8);  // #734
                            mmad(slot_46, slot_47, slot_48, 128, 128, 128, true);  // #735
                        }
                        ab_local_free.set();  // #737
                        const int32_t l0ab_cnt_9 = (int32_t)(l0ab_cnt_8 + 1);  // #738 #739
                        const int32_t l1p_cnt_3 = (int32_t)(l1p_cnt_2 + 1);  // #740 #741
                        const int32_t l1v_cnt_3 = (int32_t)(l1v_cnt_2 + 1);  // #742 #743
                        l0ab_cnt_10_state = l0ab_cnt_9;  // #2565
                        l1p_cnt_4_state = l1p_cnt_3;  // #2566
                        l1v_cnt_4_state = l1v_cnt_3;  // #2567
                    } else {
                        l0ab_cnt_10_state = l0ab_cnt_8;  // #2568
                        l1p_cnt_4_state = l1p_cnt_2;  // #2569
                        l1v_cnt_4_state = l1v_cnt_2;  // #2570
                    }
                    const int32_t l0ab_cnt_10 = (int32_t)(l0ab_cnt_10_state);  // #2571
                    const int32_t l1p_cnt_4 = (int32_t)(l1p_cnt_4_state);  // #2572
                    const int32_t l1v_cnt_4 = (int32_t)(l1v_cnt_4_state);  // #2573
                    c_local_ready.set();  // #748
                    c_local_ready.wait();  // #749
                    const GMTensor<float> view_61 = pv_ws[((cube_idx * 65536) + (slot_a_c * 16384))];  // #750
                    const Tensor<float, Position::L0C> slot_49 = l0c.get(l0c_cnt_6);  // #751
                    l0c_to_gm_nz2nd(view_61, slot_49, 128, 128, 128, 128, false, 0.0f, 0, false, false);  // #752
                    c_local_free.set();  // #753
                    CUBE_READY<PIPE_FIX>(2);  // #754
                    const int32_t l0c_cnt_7 = (int32_t)(l0c_cnt_6 + 1);  // #755 #756
                    l0ab_cnt_11_state = l0ab_cnt_10;  // #2574
                    l0c_cnt_8_state = l0c_cnt_7;  // #2575
                    l1p_cnt_5_state = l1p_cnt_4;  // #2576
                    l1v_cnt_5_state = l1v_cnt_4;  // #2577
                } else {
                    l0ab_cnt_11_state = l0ab_cnt_6;  // #2578
                    l0c_cnt_8_state = l0c_cnt_6;  // #2579
                    l1p_cnt_5_state = l1p_cnt_carry_1;  // #2580
                    l1v_cnt_5_state = l1v_cnt_carry_1;  // #2581
                }
                const int32_t l0ab_cnt_11 = (int32_t)(l0ab_cnt_11_state);  // #2582
                const int32_t l0c_cnt_8 = (int32_t)(l0c_cnt_8_state);  // #2583
                const int32_t l1p_cnt_5 = (int32_t)(l1p_cnt_5_state);  // #2584
                const int32_t l1v_cnt_5 = (int32_t)(l1v_cnt_5_state);  // #2585
                l0ab_cnt_12_state = l0ab_cnt_11;  // #2586
                l0c_cnt_9_state = l0c_cnt_8;  // #2587
                l1p_cnt_6_state = l1p_cnt_5;  // #2588
                l1v_cnt_6_state = l1v_cnt_5;  // #2589
            } else {
                l0ab_cnt_12_state = l0ab_cnt_6;  // #2590
                l0c_cnt_9_state = l0c_cnt_6;  // #2591
                l1p_cnt_6_state = l1p_cnt_carry_1;  // #2592
                l1v_cnt_6_state = l1v_cnt_carry_1;  // #2593
            }
            const int32_t l0ab_cnt_12 = (int32_t)(l0ab_cnt_12_state);  // #2594
            const int32_t l0c_cnt_9 = (int32_t)(l0c_cnt_9_state);  // #2595
            const int32_t l1p_cnt_6 = (int32_t)(l1p_cnt_6_state);  // #2596
            const int32_t l1v_cnt_6 = (int32_t)(l1v_cnt_6_state);  // #2597
            l0ab_cnt_13_state = l0ab_cnt_12;  // #2598
            l0c_cnt_10_state = l0c_cnt_9;  // #2599
            l1k_cnt_7_state = l1k_cnt_6;  // #2600
            l1p_cnt_7_state = l1p_cnt_6;  // #2601
            l1v_cnt_7_state = l1v_cnt_6;  // #2602
        }
        const int32_t l0ab_cnt_13 = (int32_t)(l0ab_cnt_13_state);  // #2604
        const int32_t l0c_cnt_10 = (int32_t)(l0c_cnt_10_state);  // #2605
        const int32_t l1k_cnt_7 = (int32_t)(l1k_cnt_7_state);  // #2606
        const int32_t l1p_cnt_7 = (int32_t)(l1p_cnt_7_state);  // #2607
        const int32_t l1v_cnt_7 = (int32_t)(l1v_cnt_7_state);  // #2608
        const int32_t exact_total = flat_exact_cnt.load(((b * (H * NT_Q_PAD8)) + (h * NT_Q_PAD8)) + q_idx_1);  // #787 #788 #789 #790 #791 #792
        const int32_t idiv_33 = exact_total / 6;  // #3473
        const int32_t idiv_34 = exact_total % 6;  // #3474
        const bool idiv_35 = idiv_34 != 0;  // #3475
        const bool idiv_36 = exact_total < 0;  // #3476
        const bool idiv_38 = idiv_36 == false;  // #3478
        const bool idiv_39 = idiv_35 && idiv_38;  // #3479
        const int32_t idiv_40 = (idiv_39) ? (1) : (0);  // #3480
        const int32_t num_ebatches = idiv_33 + idiv_40;  // #793
        const bool cmp_26 = num_ebatches > 0;  // #794
        int32_t l0ab_cnt_31_state = 0;  // #2609
        int32_t l0c_cnt_24_state = 0;  // #2610
        int32_t l1k_cnt_17_state = 0;  // #2611
        int32_t l1p_cnt_17_state = 0;  // #2612
        int32_t l1v_cnt_17_state = 0;  // #2613
        if (cmp_26) {  // #2386
            const int32_t lag_e = Min(3, num_ebatches);  // #795
            const int32_t add_78 = num_ebatches + lag_e;  // #796
            int32_t l0ab_cnt_30_state = l0ab_cnt_13;  // #2614
            int32_t l0c_cnt_23_state = l0c_cnt_10;  // #2615
            int32_t l1k_cnt_16_state = l1k_cnt_7;  // #2616
            int32_t l1p_cnt_16_state = l1p_cnt_7;  // #2617
            int32_t l1v_cnt_16_state = l1v_cnt_7;  // #2618
            for (int32_t bi = 0; bi < add_78; bi += 1) {  // #2733
                const int32_t l0ab_cnt_carry_2 = (int32_t)(l0ab_cnt_30_state);  // #2619
                const int32_t l0c_cnt_carry_2 = (int32_t)(l0c_cnt_23_state);  // #2620
                const int32_t l1k_cnt_carry_2 = (int32_t)(l1k_cnt_16_state);  // #2621
                const int32_t l1p_cnt_carry_2 = (int32_t)(l1p_cnt_16_state);  // #2622
                const int32_t l1v_cnt_carry_2 = (int32_t)(l1v_cnt_16_state);  // #2623
                const bool cmp_27 = bi < num_ebatches;  // #797
                int32_t l0ab_cnt_21_state = 0;  // #2624
                int32_t l0c_cnt_18_state = 0;  // #2625
                int32_t l1k_cnt_15_state = 0;  // #2626
                if (cmp_27) {  // #1778
                    const int32_t slot_b_p = bi & 3;  // #798
                    const int32_t e_base_p = bi * 6;  // #799
                    const int32_t sub_18 = num_ebatches + -1;  // #800
                    const bool cmp_28 = bi < sub_18;  // #801
                    int32_t l0ab_cnt_20_state = 0;  // #2627
                    int32_t l0c_cnt_17_state = 0;  // #2628
                    int32_t l1k_cnt_14_state = 0;  // #2629
                    if (cmp_28) {  // #1775
                        k_local_free.wait();  // #803
                        const int32_t blk0 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx_1 * NPAD)) + e_base_p));  // #805 #806 #807 #808 #809 #810 #811 #812 #813 #814 #815
                        const int32_t add_82 = e_base_p + 1;  // #816
                        const int32_t blk1 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx_1 * NPAD)) + add_82));  // #818 #819 #820 #821 #822 #823 #824 #825 #826 #827 #828
                        const int32_t kv_off0 = blk0 * 64;  // #829
                        if (blk1 == (blk0 + 1)) {  // #830 #831 #850
                            const Tensor<bfloat16_t, Position::L1> slot_51 = l1k.get(l1k_cnt_carry_2);  // #832
                            const GMTensor<bfloat16_t> view_67 = k_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off0 * 128))];  // #835
                            gm_to_l1_nd2nz(slot_51, view_67, 128, 128, 128, 128);  // #836
                        } else {
                            const int32_t kv_off1 = blk1 * 64;  // #837
                            const Tensor<bfloat16_t, Position::L1> slot_52 = l1k.get(l1k_cnt_carry_2);  // #838
                            const Tensor<bfloat16_t, Position::L1> view_68 = l1k.get(l1k_cnt_carry_2);  // #839
                            const GMTensor<bfloat16_t> view_69 = k_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off0 * 128))];  // #842
                            gm_to_l1_nd2nz(view_68, view_69, 64, 128, 128, 128);  // #843
                            const Tensor<bfloat16_t, Position::L1> slot_53 = l1k.get(l1k_cnt_carry_2);  // #844
                            const Tensor<bfloat16_t, Position::L1> view_70 = l1k.get(l1k_cnt_carry_2)[1024];  // #845
                            const GMTensor<bfloat16_t> view_71 = k_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off1 * 128))];  // #848
                            gm_to_l1_nd2nz(view_70, view_71, 64, 128, 128, 128);  // #849
                        }
                        k_local_ready.set();  // #851
                        ab_local_free.wait();  // #852
                        k_local_ready.wait();  // #853
                        const Tensor<bfloat16_t, Position::L0A> slot_54 = l0a.get(l0ab_cnt_carry_2);  // #854
                        const Tensor<bfloat16_t, Position::L1> slot_55 = l1q.get(l1q_cnt_carry);  // #855
                        l1_to_l0<false>(slot_54, slot_55, 128, 128, 128, 128);  // #856
                        const Tensor<bfloat16_t, Position::L0B> slot_56 = l0b.get(l0ab_cnt_carry_2);  // #857
                        const Tensor<bfloat16_t, Position::L1> slot_57 = l1k.get(l1k_cnt_carry_2);  // #858
                        l1_to_l0<false>(slot_56, slot_57, 128, 128, 128, 128);  // #859
                        k_local_free.set();  // #860
                        ab_local_ready.set();  // #861
                        c_local_free.wait();  // #862
                        ab_local_ready.wait();  // #863
                        const Tensor<float, Position::L0C> slot_58 = l0c.get(l0c_cnt_carry_2);  // #864
                        const Tensor<bfloat16_t, Position::L0A> slot_59 = l0a.get(l0ab_cnt_carry_2);  // #865
                        const Tensor<bfloat16_t, Position::L0B> slot_60 = l0b.get(l0ab_cnt_carry_2);  // #866
                        mmad(slot_58, slot_59, slot_60, 128, 128, 128, true);  // #867
                        ab_local_free.set();  // #868
                        c_local_ready.set();  // #869
                        c_local_ready.wait();  // #870
                        const GMTensor<float> view_72 = score_ws[((cube_idx * 196608) + (slot_b_p * 49152))];  // #871
                        const Tensor<float, Position::L0C> slot_61 = l0c.get(l0c_cnt_carry_2);  // #872
                        l0c_to_gm_nz2nd(view_72, slot_61, 128, 128, 128, 128, false, 0.0f, 0, false, false);  // #873
                        c_local_free.set();  // #874
                        k_local_free.wait();  // #875
                        const int32_t add_90 = e_base_p + 2;  // #876
                        const int32_t blk2 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx_1 * NPAD)) + add_90));  // #878 #879 #880 #881 #882 #883 #884 #885 #886 #887 #888
                        const int32_t add_94 = e_base_p + 3;  // #889
                        const int32_t blk3 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx_1 * NPAD)) + add_94));  // #891 #892 #893 #894 #895 #896 #897 #898 #899 #900 #901
                        const int32_t kv_off2 = blk2 * 64;  // #902
                        if (blk3 == (blk2 + 1)) {  // #903 #904 #926
                            const Tensor<bfloat16_t, Position::L1> slot_62 = l1k.get((l1k_cnt_carry_2 + 1));  // #905 #906
                            const GMTensor<bfloat16_t> view_73 = k_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off2 * 128))];  // #909
                            gm_to_l1_nd2nz(slot_62, view_73, 128, 128, 128, 128);  // #910
                        } else {
                            const int32_t kv_off3 = blk3 * 64;  // #911
                            const int32_t add_101 = l1k_cnt_carry_2 + 1;  // #912
                            const Tensor<bfloat16_t, Position::L1> slot_63 = l1k.get(add_101);  // #913
                            const Tensor<bfloat16_t, Position::L1> view_74 = l1k.get(add_101);  // #914
                            const GMTensor<bfloat16_t> view_75 = k_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off2 * 128))];  // #917
                            gm_to_l1_nd2nz(view_74, view_75, 64, 128, 128, 128);  // #918
                            const Tensor<bfloat16_t, Position::L1> slot_64 = l1k.get(add_101);  // #920
                            const Tensor<bfloat16_t, Position::L1> view_76 = l1k.get(add_101)[1024];  // #921
                            const GMTensor<bfloat16_t> view_77 = k_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off3 * 128))];  // #924
                            gm_to_l1_nd2nz(view_76, view_77, 64, 128, 128, 128);  // #925
                        }
                        k_local_ready.set();  // #927
                        ab_local_free.wait();  // #928
                        k_local_ready.wait();  // #929
                        const int32_t add_105 = l0ab_cnt_carry_2 + 1;  // #930
                        const Tensor<bfloat16_t, Position::L0A> slot_65 = l0a.get(add_105);  // #931
                        const Tensor<bfloat16_t, Position::L1> slot_66 = l1q.get(l1q_cnt_carry);  // #932
                        l1_to_l0<false>(slot_65, slot_66, 128, 128, 128, 128);  // #933
                        const Tensor<bfloat16_t, Position::L0B> slot_67 = l0b.get(add_105);  // #935
                        const Tensor<bfloat16_t, Position::L1> slot_68 = l1k.get((l1k_cnt_carry_2 + 1));  // #936 #937
                        l1_to_l0<false>(slot_67, slot_68, 128, 128, 128, 128);  // #938
                        k_local_free.set();  // #939
                        ab_local_ready.set();  // #940
                        c_local_free.wait();  // #941
                        ab_local_ready.wait();  // #942
                        const int32_t add_108 = l0c_cnt_carry_2 + 1;  // #943
                        const Tensor<float, Position::L0C> slot_69 = l0c.get(add_108);  // #944
                        const Tensor<bfloat16_t, Position::L0A> slot_70 = l0a.get(add_105);  // #946
                        const Tensor<bfloat16_t, Position::L0B> slot_71 = l0b.get(add_105);  // #948
                        mmad(slot_69, slot_70, slot_71, 128, 128, 128, true);  // #949
                        ab_local_free.set();  // #950
                        c_local_ready.set();  // #951
                        c_local_ready.wait();  // #952
                        const GMTensor<float> view_78 = score_ws[(((cube_idx * 196608) + (slot_b_p * 49152)) + 16384)];  // #953
                        const Tensor<float, Position::L0C> slot_72 = l0c.get(add_108);  // #955
                        l0c_to_gm_nz2nd(view_78, slot_72, 128, 128, 128, 128, false, 0.0f, 0, false, false);  // #956
                        c_local_free.set();  // #957
                        k_local_free.wait();  // #958
                        const int32_t add_112 = e_base_p + 4;  // #959
                        const int32_t blk4 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx_1 * NPAD)) + add_112));  // #961 #962 #963 #964 #965 #966 #967 #968 #969 #970 #971
                        const int32_t add_116 = e_base_p + 5;  // #972
                        const int32_t blk5 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx_1 * NPAD)) + add_116));  // #974 #975 #976 #977 #978 #979 #980 #981 #982 #983 #984
                        const int32_t kv_off4 = blk4 * 64;  // #985
                        if (blk5 == (blk4 + 1)) {  // #986 #987 #1009
                            const Tensor<bfloat16_t, Position::L1> slot_73 = l1k.get((l1k_cnt_carry_2 + 2));  // #988 #989
                            const GMTensor<bfloat16_t> view_79 = k_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off4 * 128))];  // #992
                            gm_to_l1_nd2nz(slot_73, view_79, 128, 128, 128, 128);  // #993
                        } else {
                            const int32_t kv_off5 = blk5 * 64;  // #994
                            const int32_t add_123 = l1k_cnt_carry_2 + 2;  // #995
                            const Tensor<bfloat16_t, Position::L1> slot_74 = l1k.get(add_123);  // #996
                            const Tensor<bfloat16_t, Position::L1> view_80 = l1k.get(add_123);  // #997
                            const GMTensor<bfloat16_t> view_81 = k_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off4 * 128))];  // #1000
                            gm_to_l1_nd2nz(view_80, view_81, 64, 128, 128, 128);  // #1001
                            const Tensor<bfloat16_t, Position::L1> slot_75 = l1k.get(add_123);  // #1003
                            const Tensor<bfloat16_t, Position::L1> view_82 = l1k.get(add_123)[1024];  // #1004
                            const GMTensor<bfloat16_t> view_83 = k_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off5 * 128))];  // #1007
                            gm_to_l1_nd2nz(view_82, view_83, 64, 128, 128, 128);  // #1008
                        }
                        k_local_ready.set();  // #1010
                        ab_local_free.wait();  // #1011
                        k_local_ready.wait();  // #1012
                        const int32_t add_127 = l0ab_cnt_carry_2 + 2;  // #1013
                        const Tensor<bfloat16_t, Position::L0A> slot_76 = l0a.get(add_127);  // #1014
                        const Tensor<bfloat16_t, Position::L1> slot_77 = l1q.get(l1q_cnt_carry);  // #1015
                        l1_to_l0<false>(slot_76, slot_77, 128, 128, 128, 128);  // #1016
                        const Tensor<bfloat16_t, Position::L0B> slot_78 = l0b.get(add_127);  // #1018
                        const Tensor<bfloat16_t, Position::L1> slot_79 = l1k.get((l1k_cnt_carry_2 + 2));  // #1019 #1020
                        l1_to_l0<false>(slot_78, slot_79, 128, 128, 128, 128);  // #1021
                        k_local_free.set();  // #1022
                        ab_local_ready.set();  // #1023
                        c_local_free.wait();  // #1024
                        ab_local_ready.wait();  // #1025
                        const int32_t add_130 = l0c_cnt_carry_2 + 2;  // #1026
                        const Tensor<float, Position::L0C> slot_80 = l0c.get(add_130);  // #1027
                        const Tensor<bfloat16_t, Position::L0A> slot_81 = l0a.get(add_127);  // #1029
                        const Tensor<bfloat16_t, Position::L0B> slot_82 = l0b.get(add_127);  // #1031
                        mmad(slot_80, slot_81, slot_82, 128, 128, 128, true);  // #1032
                        ab_local_free.set();  // #1033
                        c_local_ready.set();  // #1034
                        c_local_ready.wait();  // #1035
                        const GMTensor<float> view_84 = score_ws[(((cube_idx * 196608) + (slot_b_p * 49152)) + 32768)];  // #1036
                        const Tensor<float, Position::L0C> slot_83 = l0c.get(add_130);  // #1038
                        l0c_to_gm_nz2nd(view_84, slot_83, 128, 128, 128, 128, false, 0.0f, 0, false, false);  // #1039
                        c_local_free.set();  // #1040
                        CUBE_READY<PIPE_FIX>(0);  // #1041
                        const int32_t l1k_cnt_8 = (int32_t)(l1k_cnt_carry_2 + 3);  // #1042 #1043
                        const int32_t l0ab_cnt_14 = (int32_t)(l0ab_cnt_carry_2 + 3);  // #1044 #1045
                        const int32_t l0c_cnt_11 = (int32_t)(l0c_cnt_carry_2 + 3);  // #1046 #1047
                        l0ab_cnt_20_state = l0ab_cnt_14;  // #2630
                        l0c_cnt_17_state = l0c_cnt_11;  // #2631
                        l1k_cnt_14_state = l1k_cnt_8;  // #2632
                    } else {
                        const int32_t b_len_p = exact_total - e_base_p;  // #1196
                        k_local_free.wait();  // #1198
                        if (b_len_p >= 2) {  // #1199 #1269
                            const int32_t blk0_t = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx_1 * NPAD)) + e_base_p));  // #1201 #1202 #1203 #1204 #1205 #1206 #1207 #1208 #1209 #1210 #1211
                            const int32_t add_154 = e_base_p + 1;  // #1212
                            const int32_t blk1_t = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx_1 * NPAD)) + add_154));  // #1214 #1215 #1216 #1217 #1218 #1219 #1220 #1221 #1222 #1223 #1224
                            const int32_t kv_off0_t = blk0_t * 64;  // #1225
                            if (blk1_t == (blk0_t + 1)) {  // #1226 #1227 #1246
                                const Tensor<bfloat16_t, Position::L1> slot_84 = l1k.get(l1k_cnt_carry_2);  // #1228
                                const GMTensor<bfloat16_t> view_115 = k_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off0_t * 128))];  // #1231
                                gm_to_l1_nd2nz(slot_84, view_115, 128, 128, 128, 128);  // #1232
                            } else {
                                const int32_t kv_off1_t = blk1_t * 64;  // #1233
                                const Tensor<bfloat16_t, Position::L1> slot_85 = l1k.get(l1k_cnt_carry_2);  // #1234
                                const Tensor<bfloat16_t, Position::L1> view_116 = l1k.get(l1k_cnt_carry_2);  // #1235
                                const GMTensor<bfloat16_t> view_117 = k_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off0_t * 128))];  // #1238
                                gm_to_l1_nd2nz(view_116, view_117, 64, 128, 128, 128);  // #1239
                                const Tensor<bfloat16_t, Position::L1> slot_86 = l1k.get(l1k_cnt_carry_2);  // #1240
                                const Tensor<bfloat16_t, Position::L1> view_118 = l1k.get(l1k_cnt_carry_2)[1024];  // #1241
                                const GMTensor<bfloat16_t> view_119 = k_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off1_t * 128))];  // #1244
                                gm_to_l1_nd2nz(view_118, view_119, 64, 128, 128, 128);  // #1245
                            }
                        } else {
                            const Tensor<bfloat16_t, Position::L1> slot_87 = l1k.get(l1k_cnt_carry_2);  // #1247
                            set_constant_to_l1(slot_87, (bfloat16_t)(0.0f), 1024);  // #1248
                            PipeBarrier<PIPE_MTE2>();  // #1249
                            const int32_t blk0_t_1 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx_1 * NPAD)) + e_base_p));  // #1251 #1252 #1253 #1254 #1255 #1256 #1257 #1258 #1259 #1260 #1261
                            const int32_t kv_off0_t_1 = blk0_t_1 * 64;  // #1262
                            const Tensor<bfloat16_t, Position::L1> slot_88 = l1k.get(l1k_cnt_carry_2);  // #1263
                            const Tensor<bfloat16_t, Position::L1> view_120 = l1k.get(l1k_cnt_carry_2);  // #1264
                            const GMTensor<bfloat16_t> view_121 = k_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off0_t_1 * 128))];  // #1267
                            gm_to_l1_nd2nz(view_120, view_121, 64, 128, 128, 128);  // #1268
                        }
                        k_local_ready.set();  // #1270
                        ab_local_free.wait();  // #1271
                        k_local_ready.wait();  // #1272
                        const Tensor<bfloat16_t, Position::L0A> slot_89 = l0a.get(l0ab_cnt_carry_2);  // #1273
                        const Tensor<bfloat16_t, Position::L1> slot_90 = l1q.get(l1q_cnt_carry);  // #1274
                        l1_to_l0<false>(slot_89, slot_90, 128, 128, 128, 128);  // #1275
                        const Tensor<bfloat16_t, Position::L0B> slot_91 = l0b.get(l0ab_cnt_carry_2);  // #1276
                        const Tensor<bfloat16_t, Position::L1> slot_92 = l1k.get(l1k_cnt_carry_2);  // #1277
                        l1_to_l0<false>(slot_91, slot_92, 128, 128, 128, 128);  // #1278
                        k_local_free.set();  // #1279
                        ab_local_ready.set();  // #1280
                        c_local_free.wait();  // #1281
                        ab_local_ready.wait();  // #1282
                        const Tensor<float, Position::L0C> slot_93 = l0c.get(l0c_cnt_carry_2);  // #1283
                        const Tensor<bfloat16_t, Position::L0A> slot_94 = l0a.get(l0ab_cnt_carry_2);  // #1284
                        const Tensor<bfloat16_t, Position::L0B> slot_95 = l0b.get(l0ab_cnt_carry_2);  // #1285
                        mmad(slot_93, slot_94, slot_95, 128, 128, 128, true);  // #1286
                        ab_local_free.set();  // #1287
                        c_local_ready.set();  // #1288
                        c_local_ready.wait();  // #1289
                        const GMTensor<float> view_122 = score_ws[((cube_idx * 196608) + (slot_b_p * 49152))];  // #1290
                        const Tensor<float, Position::L0C> slot_96 = l0c.get(l0c_cnt_carry_2);  // #1291
                        l0c_to_gm_nz2nd(view_122, slot_96, 128, 128, 128, 128, false, 0.0f, 0, false, false);  // #1292
                        c_local_free.set();  // #1293
                        const int32_t l1k_cnt_9 = (int32_t)(l1k_cnt_carry_2 + 1);  // #1294 #1295
                        const int32_t l0ab_cnt_15 = (int32_t)(l0ab_cnt_carry_2 + 1);  // #1296 #1297
                        const int32_t l0c_cnt_12 = (int32_t)(l0c_cnt_carry_2 + 1);  // #1298 #1299
                        const bool cmp_34 = b_len_p > 2;  // #1300
                        int32_t l0ab_cnt_17_state = 0;  // #2633
                        int32_t l0c_cnt_14_state = 0;  // #2634
                        int32_t l1k_cnt_11_state = 0;  // #2635
                        if (cmp_34) {  // #1408
                            k_local_free.wait();  // #1302
                            if (b_len_p >= 4) {  // #1303 #1375
                                const int32_t add_169 = e_base_p + 2;  // #1304
                                const int32_t blk2_t = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx_1 * NPAD)) + add_169));  // #1306 #1307 #1308 #1309 #1310 #1311 #1312 #1313 #1314 #1315 #1316
                                const int32_t add_173 = e_base_p + 3;  // #1317
                                const int32_t blk3_t = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx_1 * NPAD)) + add_173));  // #1319 #1320 #1321 #1322 #1323 #1324 #1325 #1326 #1327 #1328 #1329
                                const int32_t kv_off2_t = blk2_t * 64;  // #1330
                                if (blk3_t == (blk2_t + 1)) {  // #1331 #1332 #1351
                                    const Tensor<bfloat16_t, Position::L1> slot_97 = l1k.get(l1k_cnt_9);  // #1333
                                    const GMTensor<bfloat16_t> view_123 = k_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off2_t * 128))];  // #1336
                                    gm_to_l1_nd2nz(slot_97, view_123, 128, 128, 128, 128);  // #1337
                                } else {
                                    const int32_t kv_off3_t = blk3_t * 64;  // #1338
                                    const Tensor<bfloat16_t, Position::L1> slot_98 = l1k.get(l1k_cnt_9);  // #1339
                                    const Tensor<bfloat16_t, Position::L1> view_124 = l1k.get(l1k_cnt_9);  // #1340
                                    const GMTensor<bfloat16_t> view_125 = k_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off2_t * 128))];  // #1343
                                    gm_to_l1_nd2nz(view_124, view_125, 64, 128, 128, 128);  // #1344
                                    const Tensor<bfloat16_t, Position::L1> slot_99 = l1k.get(l1k_cnt_9);  // #1345
                                    const Tensor<bfloat16_t, Position::L1> view_126 = l1k.get(l1k_cnt_9)[1024];  // #1346
                                    const GMTensor<bfloat16_t> view_127 = k_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off3_t * 128))];  // #1349
                                    gm_to_l1_nd2nz(view_126, view_127, 64, 128, 128, 128);  // #1350
                                }
                            } else {
                                const Tensor<bfloat16_t, Position::L1> slot_100 = l1k.get(l1k_cnt_9);  // #1352
                                set_constant_to_l1(slot_100, (bfloat16_t)(0.0f), 1024);  // #1353
                                PipeBarrier<PIPE_MTE2>();  // #1354
                                const int32_t add_181 = e_base_p + 2;  // #1355
                                const int32_t blk2_t_1 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx_1 * NPAD)) + add_181));  // #1357 #1358 #1359 #1360 #1361 #1362 #1363 #1364 #1365 #1366 #1367
                                const int32_t kv_off2_t_1 = blk2_t_1 * 64;  // #1368
                                const Tensor<bfloat16_t, Position::L1> slot_101 = l1k.get(l1k_cnt_9);  // #1369
                                const Tensor<bfloat16_t, Position::L1> view_128 = l1k.get(l1k_cnt_9);  // #1370
                                const GMTensor<bfloat16_t> view_129 = k_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off2_t_1 * 128))];  // #1373
                                gm_to_l1_nd2nz(view_128, view_129, 64, 128, 128, 128);  // #1374
                            }
                            k_local_ready.set();  // #1376
                            ab_local_free.wait();  // #1377
                            k_local_ready.wait();  // #1378
                            const Tensor<bfloat16_t, Position::L0A> slot_102 = l0a.get(l0ab_cnt_15);  // #1379
                            const Tensor<bfloat16_t, Position::L1> slot_103 = l1q.get(l1q_cnt_carry);  // #1380
                            l1_to_l0<false>(slot_102, slot_103, 128, 128, 128, 128);  // #1381
                            const Tensor<bfloat16_t, Position::L0B> slot_104 = l0b.get(l0ab_cnt_15);  // #1382
                            const Tensor<bfloat16_t, Position::L1> slot_105 = l1k.get(l1k_cnt_9);  // #1383
                            l1_to_l0<false>(slot_104, slot_105, 128, 128, 128, 128);  // #1384
                            k_local_free.set();  // #1385
                            ab_local_ready.set();  // #1386
                            c_local_free.wait();  // #1387
                            ab_local_ready.wait();  // #1388
                            const Tensor<float, Position::L0C> slot_106 = l0c.get(l0c_cnt_12);  // #1389
                            const Tensor<bfloat16_t, Position::L0A> slot_107 = l0a.get(l0ab_cnt_15);  // #1390
                            const Tensor<bfloat16_t, Position::L0B> slot_108 = l0b.get(l0ab_cnt_15);  // #1391
                            mmad(slot_106, slot_107, slot_108, 128, 128, 128, true);  // #1392
                            ab_local_free.set();  // #1393
                            c_local_ready.set();  // #1394
                            c_local_ready.wait();  // #1395
                            const GMTensor<float> view_130 = score_ws[(((cube_idx * 196608) + (slot_b_p * 49152)) + 16384)];  // #1396
                            const Tensor<float, Position::L0C> slot_109 = l0c.get(l0c_cnt_12);  // #1397
                            l0c_to_gm_nz2nd(view_130, slot_109, 128, 128, 128, 128, false, 0.0f, 0, false, false);  // #1398
                            c_local_free.set();  // #1399
                            const int32_t l1k_cnt_10 = (int32_t)(l1k_cnt_9 + 1);  // #1400 #1401
                            const int32_t l0ab_cnt_16 = (int32_t)(l0ab_cnt_15 + 1);  // #1402 #1403
                            const int32_t l0c_cnt_13 = (int32_t)(l0c_cnt_12 + 1);  // #1404 #1405
                            l0ab_cnt_17_state = l0ab_cnt_16;  // #2636
                            l0c_cnt_14_state = l0c_cnt_13;  // #2637
                            l1k_cnt_11_state = l1k_cnt_10;  // #2638
                        } else {
                            l0ab_cnt_17_state = l0ab_cnt_15;  // #2639
                            l0c_cnt_14_state = l0c_cnt_12;  // #2640
                            l1k_cnt_11_state = l1k_cnt_9;  // #2641
                        }
                        const int32_t l0ab_cnt_17 = (int32_t)(l0ab_cnt_17_state);  // #2642
                        const int32_t l0c_cnt_14 = (int32_t)(l0c_cnt_14_state);  // #2643
                        const int32_t l1k_cnt_11 = (int32_t)(l1k_cnt_11_state);  // #2644
                        const bool cmp_37 = b_len_p > 4;  // #1409
                        int32_t l0ab_cnt_19_state = 0;  // #2645
                        int32_t l0c_cnt_16_state = 0;  // #2646
                        int32_t l1k_cnt_13_state = 0;  // #2647
                        if (cmp_37) {  // #1517
                            k_local_free.wait();  // #1411
                            if (b_len_p == 6) {  // #1412 #1484
                                const int32_t add_189 = e_base_p + 4;  // #1413
                                const int32_t blk4_t = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx_1 * NPAD)) + add_189));  // #1415 #1416 #1417 #1418 #1419 #1420 #1421 #1422 #1423 #1424 #1425
                                const int32_t add_193 = e_base_p + 5;  // #1426
                                const int32_t blk5_t = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx_1 * NPAD)) + add_193));  // #1428 #1429 #1430 #1431 #1432 #1433 #1434 #1435 #1436 #1437 #1438
                                const int32_t kv_off4_t = blk4_t * 64;  // #1439
                                if (blk5_t == (blk4_t + 1)) {  // #1440 #1441 #1460
                                    const Tensor<bfloat16_t, Position::L1> slot_110 = l1k.get(l1k_cnt_11);  // #1442
                                    const GMTensor<bfloat16_t> view_131 = k_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off4_t * 128))];  // #1445
                                    gm_to_l1_nd2nz(slot_110, view_131, 128, 128, 128, 128);  // #1446
                                } else {
                                    const int32_t kv_off5_t = blk5_t * 64;  // #1447
                                    const Tensor<bfloat16_t, Position::L1> slot_111 = l1k.get(l1k_cnt_11);  // #1448
                                    const Tensor<bfloat16_t, Position::L1> view_132 = l1k.get(l1k_cnt_11);  // #1449
                                    const GMTensor<bfloat16_t> view_133 = k_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off4_t * 128))];  // #1452
                                    gm_to_l1_nd2nz(view_132, view_133, 64, 128, 128, 128);  // #1453
                                    const Tensor<bfloat16_t, Position::L1> slot_112 = l1k.get(l1k_cnt_11);  // #1454
                                    const Tensor<bfloat16_t, Position::L1> view_134 = l1k.get(l1k_cnt_11)[1024];  // #1455
                                    const GMTensor<bfloat16_t> view_135 = k_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off5_t * 128))];  // #1458
                                    gm_to_l1_nd2nz(view_134, view_135, 64, 128, 128, 128);  // #1459
                                }
                            } else {
                                const Tensor<bfloat16_t, Position::L1> slot_113 = l1k.get(l1k_cnt_11);  // #1461
                                set_constant_to_l1(slot_113, (bfloat16_t)(0.0f), 1024);  // #1462
                                PipeBarrier<PIPE_MTE2>();  // #1463
                                const int32_t add_201 = e_base_p + 4;  // #1464
                                const int32_t blk4_t_1 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx_1 * NPAD)) + add_201));  // #1466 #1467 #1468 #1469 #1470 #1471 #1472 #1473 #1474 #1475 #1476
                                const int32_t kv_off4_t_1 = blk4_t_1 * 64;  // #1477
                                const Tensor<bfloat16_t, Position::L1> slot_114 = l1k.get(l1k_cnt_11);  // #1478
                                const Tensor<bfloat16_t, Position::L1> view_136 = l1k.get(l1k_cnt_11);  // #1479
                                const GMTensor<bfloat16_t> view_137 = k_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off4_t_1 * 128))];  // #1482
                                gm_to_l1_nd2nz(view_136, view_137, 64, 128, 128, 128);  // #1483
                            }
                            k_local_ready.set();  // #1485
                            ab_local_free.wait();  // #1486
                            k_local_ready.wait();  // #1487
                            const Tensor<bfloat16_t, Position::L0A> slot_115 = l0a.get(l0ab_cnt_17);  // #1488
                            const Tensor<bfloat16_t, Position::L1> slot_116 = l1q.get(l1q_cnt_carry);  // #1489
                            l1_to_l0<false>(slot_115, slot_116, 128, 128, 128, 128);  // #1490
                            const Tensor<bfloat16_t, Position::L0B> slot_117 = l0b.get(l0ab_cnt_17);  // #1491
                            const Tensor<bfloat16_t, Position::L1> slot_118 = l1k.get(l1k_cnt_11);  // #1492
                            l1_to_l0<false>(slot_117, slot_118, 128, 128, 128, 128);  // #1493
                            k_local_free.set();  // #1494
                            ab_local_ready.set();  // #1495
                            c_local_free.wait();  // #1496
                            ab_local_ready.wait();  // #1497
                            const Tensor<float, Position::L0C> slot_119 = l0c.get(l0c_cnt_14);  // #1498
                            const Tensor<bfloat16_t, Position::L0A> slot_120 = l0a.get(l0ab_cnt_17);  // #1499
                            const Tensor<bfloat16_t, Position::L0B> slot_121 = l0b.get(l0ab_cnt_17);  // #1500
                            mmad(slot_119, slot_120, slot_121, 128, 128, 128, true);  // #1501
                            ab_local_free.set();  // #1502
                            c_local_ready.set();  // #1503
                            c_local_ready.wait();  // #1504
                            const GMTensor<float> view_138 = score_ws[(((cube_idx * 196608) + (slot_b_p * 49152)) + 32768)];  // #1505
                            const Tensor<float, Position::L0C> slot_122 = l0c.get(l0c_cnt_14);  // #1506
                            l0c_to_gm_nz2nd(view_138, slot_122, 128, 128, 128, 128, false, 0.0f, 0, false, false);  // #1507
                            c_local_free.set();  // #1508
                            const int32_t l1k_cnt_12 = (int32_t)(l1k_cnt_11 + 1);  // #1509 #1510
                            const int32_t l0ab_cnt_18 = (int32_t)(l0ab_cnt_17 + 1);  // #1511 #1512
                            const int32_t l0c_cnt_15 = (int32_t)(l0c_cnt_14 + 1);  // #1513 #1514
                            l0ab_cnt_19_state = l0ab_cnt_18;  // #2648
                            l0c_cnt_16_state = l0c_cnt_15;  // #2649
                            l1k_cnt_13_state = l1k_cnt_12;  // #2650
                        } else {
                            l0ab_cnt_19_state = l0ab_cnt_17;  // #2651
                            l0c_cnt_16_state = l0c_cnt_14;  // #2652
                            l1k_cnt_13_state = l1k_cnt_11;  // #2653
                        }
                        const int32_t l0ab_cnt_19 = (int32_t)(l0ab_cnt_19_state);  // #2654
                        const int32_t l0c_cnt_16 = (int32_t)(l0c_cnt_16_state);  // #2655
                        const int32_t l1k_cnt_13 = (int32_t)(l1k_cnt_13_state);  // #2656
                        CUBE_READY<PIPE_FIX>(0);  // #1519
                        l0ab_cnt_20_state = l0ab_cnt_19;  // #2657
                        l0c_cnt_17_state = l0c_cnt_16;  // #2658
                        l1k_cnt_14_state = l1k_cnt_13;  // #2659
                    }
                    const int32_t l0ab_cnt_20 = (int32_t)(l0ab_cnt_20_state);  // #2660
                    const int32_t l0c_cnt_17 = (int32_t)(l0c_cnt_17_state);  // #2661
                    const int32_t l1k_cnt_14 = (int32_t)(l1k_cnt_14_state);  // #2662
                    l0ab_cnt_21_state = l0ab_cnt_20;  // #2663
                    l0c_cnt_18_state = l0c_cnt_17;  // #2664
                    l1k_cnt_15_state = l1k_cnt_14;  // #2665
                } else {
                    l0ab_cnt_21_state = l0ab_cnt_carry_2;  // #2666
                    l0c_cnt_18_state = l0c_cnt_carry_2;  // #2667
                    l1k_cnt_15_state = l1k_cnt_carry_2;  // #2668
                }
                const int32_t l0ab_cnt_21 = (int32_t)(l0ab_cnt_21_state);  // #2669
                const int32_t l0c_cnt_18 = (int32_t)(l0c_cnt_18_state);  // #2670
                const int32_t l1k_cnt_15 = (int32_t)(l1k_cnt_15_state);  // #2671
                const bool cmp_66 = bi >= lag_e;  // #1779
                int32_t l0ab_cnt_29_state = 0;  // #2672
                int32_t l0c_cnt_22_state = 0;  // #2673
                int32_t l1p_cnt_15_state = 0;  // #2674
                int32_t l1v_cnt_15_state = 0;  // #2675
                if (cmp_66) {  // #2381
                    const int32_t ci = bi - lag_e;  // #1780
                    const int32_t slot_b_c = ci & 3;  // #1781
                    const int32_t e_base_c = ci * 6;  // #1782
                    const int32_t sub_56 = num_ebatches + -1;  // #1783
                    const bool cmp_67 = ci < sub_56;  // #1784
                    int32_t l0ab_cnt_28_state = 0;  // #2676
                    int32_t l0c_cnt_21_state = 0;  // #2677
                    int32_t l1p_cnt_14_state = 0;  // #2678
                    int32_t l1v_cnt_14_state = 0;  // #2679
                    if (cmp_67) {  // #2357
                        WAIT_VEC<PIPE_S>(1);  // #1786
                        c_local_free.wait();  // #1787
                        pv_local_free.wait();  // #1788
                        const int32_t vblk0 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx_1 * NPAD)) + e_base_c));  // #1790 #1791 #1792 #1793 #1794 #1795 #1796 #1797 #1798 #1799 #1800
                        const int32_t add_229 = e_base_c + 1;  // #1801
                        const int32_t vblk1 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx_1 * NPAD)) + add_229));  // #1803 #1804 #1805 #1806 #1807 #1808 #1809 #1810 #1811 #1812 #1813
                        const Tensor<bfloat16_t, Position::L1> slot_123 = l1p.get(l1p_cnt_carry_2);  // #1814
                        const GMTensor<bfloat16_t> view_178 = p_ws[((cube_idx * 196608) + (slot_b_c * 49152))];  // #1815
                        gm_to_l1_nd2nz(slot_123, view_178, 128, 128, 128, 128);  // #1816
                        const int32_t v_off0 = vblk0 * 64;  // #1817
                        if (vblk1 == (vblk0 + 1)) {  // #1818 #1819 #1838
                            const Tensor<bfloat16_t, Position::L1> slot_124 = l1v.get(l1v_cnt_carry_2);  // #1820
                            const GMTensor<bfloat16_t> view_179 = v_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (v_off0 * 128))];  // #1823
                            gm_to_l1_nd2nz(slot_124, view_179, 128, 128, 128, 128);  // #1824
                        } else {
                            const int32_t v_off1 = vblk1 * 64;  // #1825
                            const Tensor<bfloat16_t, Position::L1> slot_125 = l1v.get(l1v_cnt_carry_2);  // #1826
                            const Tensor<bfloat16_t, Position::L1> view_180 = l1v.get(l1v_cnt_carry_2);  // #1827
                            const GMTensor<bfloat16_t> view_181 = v_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (v_off0 * 128))];  // #1830
                            gm_to_l1_nd2nz(view_180, view_181, 64, 128, 128, 128);  // #1831
                            const Tensor<bfloat16_t, Position::L1> slot_126 = l1v.get(l1v_cnt_carry_2);  // #1832
                            const Tensor<bfloat16_t, Position::L1> view_182 = l1v.get(l1v_cnt_carry_2)[1024];  // #1833
                            const GMTensor<bfloat16_t> view_183 = v_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (v_off1 * 128))];  // #1836
                            gm_to_l1_nd2nz(view_182, view_183, 64, 128, 128, 128);  // #1837
                        }
                        pv_local_ready.set();  // #1839
                        ab_local_free.wait();  // #1840
                        pv_local_ready.wait();  // #1841
                        const Tensor<bfloat16_t, Position::L0A> slot_127 = l0a.get(l0ab_cnt_21);  // #1842
                        const Tensor<bfloat16_t, Position::L1> slot_128 = l1p.get(l1p_cnt_carry_2);  // #1843
                        l1_to_l0<false>(slot_127, slot_128, 128, 128, 128, 128);  // #1844
                        const Tensor<bfloat16_t, Position::L0B> slot_129 = l0b.get(l0ab_cnt_21);  // #1845
                        const Tensor<bfloat16_t, Position::L1> slot_130 = l1v.get(l1v_cnt_carry_2);  // #1846
                        l1_to_l0<true>(slot_129, slot_130, 128, 128, 128, 128);  // #1847
                        pv_local_free.set();  // #1848
                        ab_local_ready.set();  // #1849
                        ab_local_ready.wait();  // #1850
                        const Tensor<float, Position::L0C> slot_131 = l0c.get(l0c_cnt_18);  // #1851
                        const Tensor<bfloat16_t, Position::L0A> slot_132 = l0a.get(l0ab_cnt_21);  // #1852
                        const Tensor<bfloat16_t, Position::L0B> slot_133 = l0b.get(l0ab_cnt_21);  // #1853
                        mmad(slot_131, slot_132, slot_133, 128, 128, 128, true);  // #1854
                        ab_local_free.set();  // #1855
                        pv_local_free.wait();  // #1856
                        const int32_t add_237 = e_base_c + 2;  // #1857
                        const int32_t vblk2 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx_1 * NPAD)) + add_237));  // #1859 #1860 #1861 #1862 #1863 #1864 #1865 #1866 #1867 #1868 #1869
                        const int32_t add_241 = e_base_c + 3;  // #1870
                        const int32_t vblk3 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx_1 * NPAD)) + add_241));  // #1872 #1873 #1874 #1875 #1876 #1877 #1878 #1879 #1880 #1881 #1882
                        const int32_t add_245 = l1p_cnt_carry_2 + 1;  // #1883
                        const Tensor<bfloat16_t, Position::L1> slot_134 = l1p.get(add_245);  // #1884
                        const GMTensor<bfloat16_t> view_184 = p_ws[(((cube_idx * 196608) + (slot_b_c * 49152)) + 16384)];  // #1885
                        gm_to_l1_nd2nz(slot_134, view_184, 128, 128, 128, 128);  // #1886
                        const int32_t v_off2 = vblk2 * 64;  // #1887
                        if (vblk3 == (vblk2 + 1)) {  // #1888 #1889 #1911
                            const Tensor<bfloat16_t, Position::L1> slot_135 = l1v.get((l1v_cnt_carry_2 + 1));  // #1890 #1891
                            const GMTensor<bfloat16_t> view_185 = v_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (v_off2 * 128))];  // #1894
                            gm_to_l1_nd2nz(slot_135, view_185, 128, 128, 128, 128);  // #1895
                        } else {
                            const int32_t v_off3 = vblk3 * 64;  // #1896
                            const int32_t add_249 = l1v_cnt_carry_2 + 1;  // #1897
                            const Tensor<bfloat16_t, Position::L1> slot_136 = l1v.get(add_249);  // #1898
                            const Tensor<bfloat16_t, Position::L1> view_186 = l1v.get(add_249);  // #1899
                            const GMTensor<bfloat16_t> view_187 = v_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (v_off2 * 128))];  // #1902
                            gm_to_l1_nd2nz(view_186, view_187, 64, 128, 128, 128);  // #1903
                            const Tensor<bfloat16_t, Position::L1> slot_137 = l1v.get(add_249);  // #1905
                            const Tensor<bfloat16_t, Position::L1> view_188 = l1v.get(add_249)[1024];  // #1906
                            const GMTensor<bfloat16_t> view_189 = v_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (v_off3 * 128))];  // #1909
                            gm_to_l1_nd2nz(view_188, view_189, 64, 128, 128, 128);  // #1910
                        }
                        pv_local_ready.set();  // #1912
                        ab_local_free.wait();  // #1913
                        pv_local_ready.wait();  // #1914
                        const int32_t add_253 = l0ab_cnt_21 + 1;  // #1915
                        const Tensor<bfloat16_t, Position::L0A> slot_138 = l0a.get(add_253);  // #1916
                        const Tensor<bfloat16_t, Position::L1> slot_139 = l1p.get(add_245);  // #1918
                        l1_to_l0<false>(slot_138, slot_139, 128, 128, 128, 128);  // #1919
                        const Tensor<bfloat16_t, Position::L0B> slot_140 = l0b.get(add_253);  // #1921
                        const Tensor<bfloat16_t, Position::L1> slot_141 = l1v.get((l1v_cnt_carry_2 + 1));  // #1922 #1923
                        l1_to_l0<true>(slot_140, slot_141, 128, 128, 128, 128);  // #1924
                        pv_local_free.set();  // #1925
                        ab_local_ready.set();  // #1926
                        ab_local_ready.wait();  // #1927
                        PipeBarrier<PIPE_M>();  // #1928
                        const Tensor<float, Position::L0C> slot_142 = l0c.get(l0c_cnt_18);  // #1929
                        const Tensor<bfloat16_t, Position::L0A> slot_143 = l0a.get(add_253);  // #1931
                        const Tensor<bfloat16_t, Position::L0B> slot_144 = l0b.get(add_253);  // #1933
                        mmad(slot_142, slot_143, slot_144, 128, 128, 128, false);  // #1934
                        ab_local_free.set();  // #1935
                        pv_local_free.wait();  // #1936
                        const int32_t add_259 = e_base_c + 4;  // #1937
                        const int32_t vblk4 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx_1 * NPAD)) + add_259));  // #1939 #1940 #1941 #1942 #1943 #1944 #1945 #1946 #1947 #1948 #1949
                        const int32_t add_263 = e_base_c + 5;  // #1950
                        const int32_t vblk5 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx_1 * NPAD)) + add_263));  // #1952 #1953 #1954 #1955 #1956 #1957 #1958 #1959 #1960 #1961 #1962
                        const int32_t add_267 = l1p_cnt_carry_2 + 2;  // #1963
                        const Tensor<bfloat16_t, Position::L1> slot_145 = l1p.get(add_267);  // #1964
                        const GMTensor<bfloat16_t> view_190 = p_ws[(((cube_idx * 196608) + (slot_b_c * 49152)) + 32768)];  // #1965
                        gm_to_l1_nd2nz(slot_145, view_190, 128, 128, 128, 128);  // #1966
                        const int32_t v_off4 = vblk4 * 64;  // #1967
                        if (vblk5 == (vblk4 + 1)) {  // #1968 #1969 #1991
                            const Tensor<bfloat16_t, Position::L1> slot_146 = l1v.get((l1v_cnt_carry_2 + 2));  // #1970 #1971
                            const GMTensor<bfloat16_t> view_191 = v_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (v_off4 * 128))];  // #1974
                            gm_to_l1_nd2nz(slot_146, view_191, 128, 128, 128, 128);  // #1975
                        } else {
                            const int32_t v_off5 = vblk5 * 64;  // #1976
                            const int32_t add_271 = l1v_cnt_carry_2 + 2;  // #1977
                            const Tensor<bfloat16_t, Position::L1> slot_147 = l1v.get(add_271);  // #1978
                            const Tensor<bfloat16_t, Position::L1> view_192 = l1v.get(add_271);  // #1979
                            const GMTensor<bfloat16_t> view_193 = v_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (v_off4 * 128))];  // #1982
                            gm_to_l1_nd2nz(view_192, view_193, 64, 128, 128, 128);  // #1983
                            const Tensor<bfloat16_t, Position::L1> slot_148 = l1v.get(add_271);  // #1985
                            const Tensor<bfloat16_t, Position::L1> view_194 = l1v.get(add_271)[1024];  // #1986
                            const GMTensor<bfloat16_t> view_195 = v_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (v_off5 * 128))];  // #1989
                            gm_to_l1_nd2nz(view_194, view_195, 64, 128, 128, 128);  // #1990
                        }
                        pv_local_ready.set();  // #1992
                        ab_local_free.wait();  // #1993
                        pv_local_ready.wait();  // #1994
                        const int32_t add_275 = l0ab_cnt_21 + 2;  // #1995
                        const Tensor<bfloat16_t, Position::L0A> slot_149 = l0a.get(add_275);  // #1996
                        const Tensor<bfloat16_t, Position::L1> slot_150 = l1p.get(add_267);  // #1998
                        l1_to_l0<false>(slot_149, slot_150, 128, 128, 128, 128);  // #1999
                        const Tensor<bfloat16_t, Position::L0B> slot_151 = l0b.get(add_275);  // #2001
                        const Tensor<bfloat16_t, Position::L1> slot_152 = l1v.get((l1v_cnt_carry_2 + 2));  // #2002 #2003
                        l1_to_l0<true>(slot_151, slot_152, 128, 128, 128, 128);  // #2004
                        pv_local_free.set();  // #2005
                        ab_local_ready.set();  // #2006
                        ab_local_ready.wait();  // #2007
                        PipeBarrier<PIPE_M>();  // #2008
                        const Tensor<float, Position::L0C> slot_153 = l0c.get(l0c_cnt_18);  // #2009
                        const Tensor<bfloat16_t, Position::L0A> slot_154 = l0a.get(add_275);  // #2011
                        const Tensor<bfloat16_t, Position::L0B> slot_155 = l0b.get(add_275);  // #2013
                        mmad(slot_153, slot_154, slot_155, 128, 128, 128, false);  // #2014
                        ab_local_free.set();  // #2015
                        c_local_ready.set();  // #2016
                        c_local_ready.wait();  // #2017
                        const GMTensor<float> view_196 = pv_ws[((cube_idx * 65536) + (slot_b_c * 16384))];  // #2018
                        const Tensor<float, Position::L0C> slot_156 = l0c.get(l0c_cnt_18);  // #2019
                        l0c_to_gm_nz2nd(view_196, slot_156, 128, 128, 128, 128, false, 0.0f, 0, false, false);  // #2020
                        c_local_free.set();  // #2021
                        CUBE_READY<PIPE_FIX>(2);  // #2022
                        const int32_t l0ab_cnt_22 = (int32_t)(l0ab_cnt_21 + 3);  // #2023 #2024
                        const int32_t l1p_cnt_8 = (int32_t)(l1p_cnt_carry_2 + 3);  // #2025 #2026
                        const int32_t l1v_cnt_8 = (int32_t)(l1v_cnt_carry_2 + 3);  // #2027 #2028
                        const int32_t l0c_cnt_19 = (int32_t)(l0c_cnt_18 + 1);  // #2029 #2030
                        l0ab_cnt_28_state = l0ab_cnt_22;  // #2680
                        l0c_cnt_21_state = l0c_cnt_19;  // #2681
                        l1p_cnt_14_state = l1p_cnt_8;  // #2682
                        l1v_cnt_14_state = l1v_cnt_8;  // #2683
                    } else {
                        const int32_t b_len_c = exact_total - e_base_c;  // #2031
                        WAIT_VEC<PIPE_S>(1);  // #2033
                        c_local_free.wait();  // #2034
                        pv_local_free.wait();  // #2035
                        const Tensor<bfloat16_t, Position::L1> slot_157 = l1p.get(l1p_cnt_carry_2);  // #2036
                        const GMTensor<bfloat16_t> view_197 = p_ws[((cube_idx * 196608) + (slot_b_c * 49152))];  // #2037
                        gm_to_l1_nd2nz(slot_157, view_197, 128, 128, 128, 128);  // #2038
                        if (b_len_c >= 2) {  // #2039 #2109
                            const int32_t vblk0_t = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx_1 * NPAD)) + e_base_c));  // #2041 #2042 #2043 #2044 #2045 #2046 #2047 #2048 #2049 #2050 #2051
                            const int32_t add_288 = e_base_c + 1;  // #2052
                            const int32_t vblk1_t = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx_1 * NPAD)) + add_288));  // #2054 #2055 #2056 #2057 #2058 #2059 #2060 #2061 #2062 #2063 #2064
                            const int32_t v_off0_t = vblk0_t * 64;  // #2065
                            if (vblk1_t == (vblk0_t + 1)) {  // #2066 #2067 #2086
                                const Tensor<bfloat16_t, Position::L1> slot_158 = l1v.get(l1v_cnt_carry_2);  // #2068
                                const GMTensor<bfloat16_t> view_198 = v_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (v_off0_t * 128))];  // #2071
                                gm_to_l1_nd2nz(slot_158, view_198, 128, 128, 128, 128);  // #2072
                            } else {
                                const int32_t v_off1_t = vblk1_t * 64;  // #2073
                                const Tensor<bfloat16_t, Position::L1> slot_159 = l1v.get(l1v_cnt_carry_2);  // #2074
                                const Tensor<bfloat16_t, Position::L1> view_199 = l1v.get(l1v_cnt_carry_2);  // #2075
                                const GMTensor<bfloat16_t> view_200 = v_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (v_off0_t * 128))];  // #2078
                                gm_to_l1_nd2nz(view_199, view_200, 64, 128, 128, 128);  // #2079
                                const Tensor<bfloat16_t, Position::L1> slot_160 = l1v.get(l1v_cnt_carry_2);  // #2080
                                const Tensor<bfloat16_t, Position::L1> view_201 = l1v.get(l1v_cnt_carry_2)[1024];  // #2081
                                const GMTensor<bfloat16_t> view_202 = v_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (v_off1_t * 128))];  // #2084
                                gm_to_l1_nd2nz(view_201, view_202, 64, 128, 128, 128);  // #2085
                            }
                        } else {
                            const Tensor<bfloat16_t, Position::L1> slot_161 = l1v.get(l1v_cnt_carry_2);  // #2087
                            set_constant_to_l1(slot_161, (bfloat16_t)(0.0f), 1024);  // #2088
                            PipeBarrier<PIPE_MTE2>();  // #2089
                            const int32_t vblk0_t_1 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx_1 * NPAD)) + e_base_c));  // #2091 #2092 #2093 #2094 #2095 #2096 #2097 #2098 #2099 #2100 #2101
                            const int32_t v_off0_t_1 = vblk0_t_1 * 64;  // #2102
                            const Tensor<bfloat16_t, Position::L1> slot_162 = l1v.get(l1v_cnt_carry_2);  // #2103
                            const Tensor<bfloat16_t, Position::L1> view_203 = l1v.get(l1v_cnt_carry_2);  // #2104
                            const GMTensor<bfloat16_t> view_204 = v_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (v_off0_t_1 * 128))];  // #2107
                            gm_to_l1_nd2nz(view_203, view_204, 64, 128, 128, 128);  // #2108
                        }
                        pv_local_ready.set();  // #2110
                        ab_local_free.wait();  // #2111
                        pv_local_ready.wait();  // #2112
                        const Tensor<bfloat16_t, Position::L0A> slot_163 = l0a.get(l0ab_cnt_21);  // #2113
                        const Tensor<bfloat16_t, Position::L1> slot_164 = l1p.get(l1p_cnt_carry_2);  // #2114
                        l1_to_l0<false>(slot_163, slot_164, 128, 128, 128, 128);  // #2115
                        const Tensor<bfloat16_t, Position::L0B> slot_165 = l0b.get(l0ab_cnt_21);  // #2116
                        const Tensor<bfloat16_t, Position::L1> slot_166 = l1v.get(l1v_cnt_carry_2);  // #2117
                        l1_to_l0<true>(slot_165, slot_166, 128, 128, 128, 128);  // #2118
                        pv_local_free.set();  // #2119
                        ab_local_ready.set();  // #2120
                        ab_local_ready.wait();  // #2121
                        const Tensor<float, Position::L0C> slot_167 = l0c.get(l0c_cnt_18);  // #2122
                        const Tensor<bfloat16_t, Position::L0A> slot_168 = l0a.get(l0ab_cnt_21);  // #2123
                        const Tensor<bfloat16_t, Position::L0B> slot_169 = l0b.get(l0ab_cnt_21);  // #2124
                        mmad(slot_167, slot_168, slot_169, 128, 128, 128, true);  // #2125
                        ab_local_free.set();  // #2126
                        const int32_t l0ab_cnt_23 = (int32_t)(l0ab_cnt_21 + 1);  // #2127 #2128
                        const int32_t l1p_cnt_9 = (int32_t)(l1p_cnt_carry_2 + 1);  // #2129 #2130
                        const int32_t l1v_cnt_9 = (int32_t)(l1v_cnt_carry_2 + 1);  // #2131 #2132
                        const bool cmp_73 = b_len_c > 2;  // #2133
                        int32_t l0ab_cnt_25_state = 0;  // #2684
                        int32_t l1p_cnt_11_state = 0;  // #2685
                        int32_t l1v_cnt_11_state = 0;  // #2686
                        if (cmp_73) {  // #2238
                            pv_local_free.wait();  // #2135
                            const Tensor<bfloat16_t, Position::L1> slot_170 = l1p.get(l1p_cnt_9);  // #2136
                            const GMTensor<bfloat16_t> view_205 = p_ws[(((cube_idx * 196608) + (slot_b_c * 49152)) + 16384)];  // #2137
                            gm_to_l1_nd2nz(slot_170, view_205, 128, 128, 128, 128);  // #2138
                            if (b_len_c >= 4) {  // #2139 #2211
                                const int32_t add_303 = e_base_c + 2;  // #2140
                                const int32_t vblk2_t = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx_1 * NPAD)) + add_303));  // #2142 #2143 #2144 #2145 #2146 #2147 #2148 #2149 #2150 #2151 #2152
                                const int32_t add_307 = e_base_c + 3;  // #2153
                                const int32_t vblk3_t = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx_1 * NPAD)) + add_307));  // #2155 #2156 #2157 #2158 #2159 #2160 #2161 #2162 #2163 #2164 #2165
                                const int32_t v_off2_t = vblk2_t * 64;  // #2166
                                if (vblk3_t == (vblk2_t + 1)) {  // #2167 #2168 #2187
                                    const Tensor<bfloat16_t, Position::L1> slot_171 = l1v.get(l1v_cnt_9);  // #2169
                                    const GMTensor<bfloat16_t> view_206 = v_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (v_off2_t * 128))];  // #2172
                                    gm_to_l1_nd2nz(slot_171, view_206, 128, 128, 128, 128);  // #2173
                                } else {
                                    const int32_t v_off3_t = vblk3_t * 64;  // #2174
                                    const Tensor<bfloat16_t, Position::L1> slot_172 = l1v.get(l1v_cnt_9);  // #2175
                                    const Tensor<bfloat16_t, Position::L1> view_207 = l1v.get(l1v_cnt_9);  // #2176
                                    const GMTensor<bfloat16_t> view_208 = v_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (v_off2_t * 128))];  // #2179
                                    gm_to_l1_nd2nz(view_207, view_208, 64, 128, 128, 128);  // #2180
                                    const Tensor<bfloat16_t, Position::L1> slot_173 = l1v.get(l1v_cnt_9);  // #2181
                                    const Tensor<bfloat16_t, Position::L1> view_209 = l1v.get(l1v_cnt_9)[1024];  // #2182
                                    const GMTensor<bfloat16_t> view_210 = v_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (v_off3_t * 128))];  // #2185
                                    gm_to_l1_nd2nz(view_209, view_210, 64, 128, 128, 128);  // #2186
                                }
                            } else {
                                const Tensor<bfloat16_t, Position::L1> slot_174 = l1v.get(l1v_cnt_9);  // #2188
                                set_constant_to_l1(slot_174, (bfloat16_t)(0.0f), 1024);  // #2189
                                PipeBarrier<PIPE_MTE2>();  // #2190
                                const int32_t add_315 = e_base_c + 2;  // #2191
                                const int32_t vblk2_t_1 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx_1 * NPAD)) + add_315));  // #2193 #2194 #2195 #2196 #2197 #2198 #2199 #2200 #2201 #2202 #2203
                                const int32_t v_off2_t_1 = vblk2_t_1 * 64;  // #2204
                                const Tensor<bfloat16_t, Position::L1> slot_175 = l1v.get(l1v_cnt_9);  // #2205
                                const Tensor<bfloat16_t, Position::L1> view_211 = l1v.get(l1v_cnt_9);  // #2206
                                const GMTensor<bfloat16_t> view_212 = v_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (v_off2_t_1 * 128))];  // #2209
                                gm_to_l1_nd2nz(view_211, view_212, 64, 128, 128, 128);  // #2210
                            }
                            pv_local_ready.set();  // #2212
                            ab_local_free.wait();  // #2213
                            pv_local_ready.wait();  // #2214
                            const Tensor<bfloat16_t, Position::L0A> slot_176 = l0a.get(l0ab_cnt_23);  // #2215
                            const Tensor<bfloat16_t, Position::L1> slot_177 = l1p.get(l1p_cnt_9);  // #2216
                            l1_to_l0<false>(slot_176, slot_177, 128, 128, 128, 128);  // #2217
                            const Tensor<bfloat16_t, Position::L0B> slot_178 = l0b.get(l0ab_cnt_23);  // #2218
                            const Tensor<bfloat16_t, Position::L1> slot_179 = l1v.get(l1v_cnt_9);  // #2219
                            l1_to_l0<true>(slot_178, slot_179, 128, 128, 128, 128);  // #2220
                            pv_local_free.set();  // #2221
                            ab_local_ready.set();  // #2222
                            ab_local_ready.wait();  // #2223
                            PipeBarrier<PIPE_M>();  // #2224
                            const Tensor<float, Position::L0C> slot_180 = l0c.get(l0c_cnt_18);  // #2225
                            const Tensor<bfloat16_t, Position::L0A> slot_181 = l0a.get(l0ab_cnt_23);  // #2226
                            const Tensor<bfloat16_t, Position::L0B> slot_182 = l0b.get(l0ab_cnt_23);  // #2227
                            mmad(slot_180, slot_181, slot_182, 128, 128, 128, false);  // #2228
                            ab_local_free.set();  // #2229
                            const int32_t l0ab_cnt_24 = (int32_t)(l0ab_cnt_23 + 1);  // #2230 #2231
                            const int32_t l1p_cnt_10 = (int32_t)(l1p_cnt_9 + 1);  // #2232 #2233
                            const int32_t l1v_cnt_10 = (int32_t)(l1v_cnt_9 + 1);  // #2234 #2235
                            l0ab_cnt_25_state = l0ab_cnt_24;  // #2687
                            l1p_cnt_11_state = l1p_cnt_10;  // #2688
                            l1v_cnt_11_state = l1v_cnt_10;  // #2689
                        } else {
                            l0ab_cnt_25_state = l0ab_cnt_23;  // #2690
                            l1p_cnt_11_state = l1p_cnt_9;  // #2691
                            l1v_cnt_11_state = l1v_cnt_9;  // #2692
                        }
                        const int32_t l0ab_cnt_25 = (int32_t)(l0ab_cnt_25_state);  // #2693
                        const int32_t l1p_cnt_11 = (int32_t)(l1p_cnt_11_state);  // #2694
                        const int32_t l1v_cnt_11 = (int32_t)(l1v_cnt_11_state);  // #2695
                        const bool cmp_76 = b_len_c > 4;  // #2239
                        int32_t l0ab_cnt_27_state = 0;  // #2696
                        int32_t l1p_cnt_13_state = 0;  // #2697
                        int32_t l1v_cnt_13_state = 0;  // #2698
                        if (cmp_76) {  // #2344
                            pv_local_free.wait();  // #2241
                            const Tensor<bfloat16_t, Position::L1> slot_183 = l1p.get(l1p_cnt_11);  // #2242
                            const GMTensor<bfloat16_t> view_213 = p_ws[(((cube_idx * 196608) + (slot_b_c * 49152)) + 32768)];  // #2243
                            gm_to_l1_nd2nz(slot_183, view_213, 128, 128, 128, 128);  // #2244
                            if (b_len_c == 6) {  // #2245 #2317
                                const int32_t add_323 = e_base_c + 4;  // #2246
                                const int32_t vblk4_t = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx_1 * NPAD)) + add_323));  // #2248 #2249 #2250 #2251 #2252 #2253 #2254 #2255 #2256 #2257 #2258
                                const int32_t add_327 = e_base_c + 5;  // #2259
                                const int32_t vblk5_t = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx_1 * NPAD)) + add_327));  // #2261 #2262 #2263 #2264 #2265 #2266 #2267 #2268 #2269 #2270 #2271
                                const int32_t v_off4_t = vblk4_t * 64;  // #2272
                                if (vblk5_t == (vblk4_t + 1)) {  // #2273 #2274 #2293
                                    const Tensor<bfloat16_t, Position::L1> slot_184 = l1v.get(l1v_cnt_11);  // #2275
                                    const GMTensor<bfloat16_t> view_214 = v_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (v_off4_t * 128))];  // #2278
                                    gm_to_l1_nd2nz(slot_184, view_214, 128, 128, 128, 128);  // #2279
                                } else {
                                    const int32_t v_off5_t = vblk5_t * 64;  // #2280
                                    const Tensor<bfloat16_t, Position::L1> slot_185 = l1v.get(l1v_cnt_11);  // #2281
                                    const Tensor<bfloat16_t, Position::L1> view_215 = l1v.get(l1v_cnt_11);  // #2282
                                    const GMTensor<bfloat16_t> view_216 = v_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (v_off4_t * 128))];  // #2285
                                    gm_to_l1_nd2nz(view_215, view_216, 64, 128, 128, 128);  // #2286
                                    const Tensor<bfloat16_t, Position::L1> slot_186 = l1v.get(l1v_cnt_11);  // #2287
                                    const Tensor<bfloat16_t, Position::L1> view_217 = l1v.get(l1v_cnt_11)[1024];  // #2288
                                    const GMTensor<bfloat16_t> view_218 = v_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (v_off5_t * 128))];  // #2291
                                    gm_to_l1_nd2nz(view_217, view_218, 64, 128, 128, 128);  // #2292
                                }
                            } else {
                                const Tensor<bfloat16_t, Position::L1> slot_187 = l1v.get(l1v_cnt_11);  // #2294
                                set_constant_to_l1(slot_187, (bfloat16_t)(0.0f), 1024);  // #2295
                                PipeBarrier<PIPE_MTE2>();  // #2296
                                const int32_t add_335 = e_base_c + 4;  // #2297
                                const int32_t vblk4_t_1 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx_1 * NPAD)) + add_335));  // #2299 #2300 #2301 #2302 #2303 #2304 #2305 #2306 #2307 #2308 #2309
                                const int32_t v_off4_t_1 = vblk4_t_1 * 64;  // #2310
                                const Tensor<bfloat16_t, Position::L1> slot_188 = l1v.get(l1v_cnt_11);  // #2311
                                const Tensor<bfloat16_t, Position::L1> view_219 = l1v.get(l1v_cnt_11);  // #2312
                                const GMTensor<bfloat16_t> view_220 = v_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (v_off4_t_1 * 128))];  // #2315
                                gm_to_l1_nd2nz(view_219, view_220, 64, 128, 128, 128);  // #2316
                            }
                            pv_local_ready.set();  // #2318
                            ab_local_free.wait();  // #2319
                            pv_local_ready.wait();  // #2320
                            const Tensor<bfloat16_t, Position::L0A> slot_189 = l0a.get(l0ab_cnt_25);  // #2321
                            const Tensor<bfloat16_t, Position::L1> slot_190 = l1p.get(l1p_cnt_11);  // #2322
                            l1_to_l0<false>(slot_189, slot_190, 128, 128, 128, 128);  // #2323
                            const Tensor<bfloat16_t, Position::L0B> slot_191 = l0b.get(l0ab_cnt_25);  // #2324
                            const Tensor<bfloat16_t, Position::L1> slot_192 = l1v.get(l1v_cnt_11);  // #2325
                            l1_to_l0<true>(slot_191, slot_192, 128, 128, 128, 128);  // #2326
                            pv_local_free.set();  // #2327
                            ab_local_ready.set();  // #2328
                            ab_local_ready.wait();  // #2329
                            PipeBarrier<PIPE_M>();  // #2330
                            const Tensor<float, Position::L0C> slot_193 = l0c.get(l0c_cnt_18);  // #2331
                            const Tensor<bfloat16_t, Position::L0A> slot_194 = l0a.get(l0ab_cnt_25);  // #2332
                            const Tensor<bfloat16_t, Position::L0B> slot_195 = l0b.get(l0ab_cnt_25);  // #2333
                            mmad(slot_193, slot_194, slot_195, 128, 128, 128, false);  // #2334
                            ab_local_free.set();  // #2335
                            const int32_t l0ab_cnt_26 = (int32_t)(l0ab_cnt_25 + 1);  // #2336 #2337
                            const int32_t l1p_cnt_12 = (int32_t)(l1p_cnt_11 + 1);  // #2338 #2339
                            const int32_t l1v_cnt_12 = (int32_t)(l1v_cnt_11 + 1);  // #2340 #2341
                            l0ab_cnt_27_state = l0ab_cnt_26;  // #2699
                            l1p_cnt_13_state = l1p_cnt_12;  // #2700
                            l1v_cnt_13_state = l1v_cnt_12;  // #2701
                        } else {
                            l0ab_cnt_27_state = l0ab_cnt_25;  // #2702
                            l1p_cnt_13_state = l1p_cnt_11;  // #2703
                            l1v_cnt_13_state = l1v_cnt_11;  // #2704
                        }
                        const int32_t l0ab_cnt_27 = (int32_t)(l0ab_cnt_27_state);  // #2705
                        const int32_t l1p_cnt_13 = (int32_t)(l1p_cnt_13_state);  // #2706
                        const int32_t l1v_cnt_13 = (int32_t)(l1v_cnt_13_state);  // #2707
                        c_local_ready.set();  // #2346
                        c_local_ready.wait();  // #2347
                        const GMTensor<float> view_221 = pv_ws[((cube_idx * 65536) + (slot_b_c * 16384))];  // #2348
                        const Tensor<float, Position::L0C> slot_196 = l0c.get(l0c_cnt_18);  // #2349
                        l0c_to_gm_nz2nd(view_221, slot_196, 128, 128, 128, 128, false, 0.0f, 0, false, false);  // #2350
                        c_local_free.set();  // #2351
                        CUBE_READY<PIPE_FIX>(2);  // #2352
                        const int32_t l0c_cnt_20 = (int32_t)(l0c_cnt_18 + 1);  // #2353 #2354
                        l0ab_cnt_28_state = l0ab_cnt_27;  // #2708
                        l0c_cnt_21_state = l0c_cnt_20;  // #2709
                        l1p_cnt_14_state = l1p_cnt_13;  // #2710
                        l1v_cnt_14_state = l1v_cnt_13;  // #2711
                    }
                    const int32_t l0ab_cnt_28 = (int32_t)(l0ab_cnt_28_state);  // #2712
                    const int32_t l0c_cnt_21 = (int32_t)(l0c_cnt_21_state);  // #2713
                    const int32_t l1p_cnt_14 = (int32_t)(l1p_cnt_14_state);  // #2714
                    const int32_t l1v_cnt_14 = (int32_t)(l1v_cnt_14_state);  // #2715
                    l0ab_cnt_29_state = l0ab_cnt_28;  // #2716
                    l0c_cnt_22_state = l0c_cnt_21;  // #2717
                    l1p_cnt_15_state = l1p_cnt_14;  // #2718
                    l1v_cnt_15_state = l1v_cnt_14;  // #2719
                } else {
                    l0ab_cnt_29_state = l0ab_cnt_21;  // #2720
                    l0c_cnt_22_state = l0c_cnt_18;  // #2721
                    l1p_cnt_15_state = l1p_cnt_carry_2;  // #2722
                    l1v_cnt_15_state = l1v_cnt_carry_2;  // #2723
                }
                const int32_t l0ab_cnt_29 = (int32_t)(l0ab_cnt_29_state);  // #2724
                const int32_t l0c_cnt_22 = (int32_t)(l0c_cnt_22_state);  // #2725
                const int32_t l1p_cnt_15 = (int32_t)(l1p_cnt_15_state);  // #2726
                const int32_t l1v_cnt_15 = (int32_t)(l1v_cnt_15_state);  // #2727
                l0ab_cnt_30_state = l0ab_cnt_29;  // #2728
                l0c_cnt_23_state = l0c_cnt_22;  // #2729
                l1k_cnt_16_state = l1k_cnt_15;  // #2730
                l1p_cnt_16_state = l1p_cnt_15;  // #2731
                l1v_cnt_16_state = l1v_cnt_15;  // #2732
            }
            const int32_t l0ab_cnt_30 = (int32_t)(l0ab_cnt_30_state);  // #2734
            const int32_t l0c_cnt_23 = (int32_t)(l0c_cnt_23_state);  // #2735
            const int32_t l1k_cnt_16 = (int32_t)(l1k_cnt_16_state);  // #2736
            const int32_t l1p_cnt_16 = (int32_t)(l1p_cnt_16_state);  // #2737
            const int32_t l1v_cnt_16 = (int32_t)(l1v_cnt_16_state);  // #2738
            l0ab_cnt_31_state = l0ab_cnt_30;  // #2739
            l0c_cnt_24_state = l0c_cnt_23;  // #2740
            l1k_cnt_17_state = l1k_cnt_16;  // #2741
            l1p_cnt_17_state = l1p_cnt_16;  // #2742
            l1v_cnt_17_state = l1v_cnt_16;  // #2743
        } else {
            l0ab_cnt_31_state = l0ab_cnt_13;  // #2744
            l0c_cnt_24_state = l0c_cnt_10;  // #2745
            l1k_cnt_17_state = l1k_cnt_7;  // #2746
            l1p_cnt_17_state = l1p_cnt_7;  // #2747
            l1v_cnt_17_state = l1v_cnt_7;  // #2748
        }
        const int32_t l0ab_cnt_31 = (int32_t)(l0ab_cnt_31_state);  // #2749
        const int32_t l0c_cnt_24 = (int32_t)(l0c_cnt_24_state);  // #2750
        const int32_t l1k_cnt_17 = (int32_t)(l1k_cnt_17_state);  // #2751
        const int32_t l1p_cnt_17 = (int32_t)(l1p_cnt_17_state);  // #2752
        const int32_t l1v_cnt_17 = (int32_t)(l1v_cnt_17_state);  // #2753
        PipeBarrier<PIPE_ALL>();  // #2393
        const int32_t l1q_cnt_1 = (int32_t)(l1q_cnt_carry + 1);  // #2419 #2420
        l0ab_cnt_32_state = l0ab_cnt_31;  // #2754
        l0c_cnt_25_state = l0c_cnt_24;  // #2755
        l1k_cnt_18_state = l1k_cnt_17;  // #2756
        l1p_cnt_18_state = l1p_cnt_17;  // #2757
        l1q_cnt_2_state = l1q_cnt_1;  // #2758
        l1v_cnt_18_state = l1v_cnt_17;  // #2759
    }
    WAIT_VEC<PIPE_S>(0);  // #2423
    WAIT_VEC<PIPE_S>(0);  // #2423
    WAIT_VEC<PIPE_S>(2);  // #2423
    WAIT_VEC<PIPE_S>(2);  // #2423
    return;  // #2423
}
