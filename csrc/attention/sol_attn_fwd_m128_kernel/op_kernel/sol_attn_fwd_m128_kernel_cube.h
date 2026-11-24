#pragma once
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
    const GMTensor<float> score_pairs = score_ws;  // #5
    const GMTensor<bfloat16_t> p_pairs = p_ws;  // #6
    const Tensor<bfloat16_t, Position::L1> l1q((uint64_t)(0));  // #7
    const Buff<bfloat16_t, Position::L1, 2> l1k((uint64_t)(32768), (uint64_t)(32768));  // #8
    const Buff<bfloat16_t, Position::L1, 4> l1p((uint64_t)(98304), (uint64_t)(32768));  // #9
    const Buff<bfloat16_t, Position::L1, 4> l1v((uint64_t)(229376), (uint64_t)(32768));  // #10
    const Buff<float, Position::L0C, 2> l0c((uint64_t)(0), (uint64_t)(65536));  // #11
    const Buff<bfloat16_t, Position::L0A, 2> l0a((uint64_t)(0), (uint64_t)(32768));  // #12
    const Buff<bfloat16_t, Position::L0B, 2> l0b((uint64_t)(0), (uint64_t)(32768));  // #13
    // sync.mutex cv id=0: the producer side; its 2 tokens are drained before the return  // #33
    CUBE_READY<PIPE_MTE2>(1);  // #34
    CUBE_READY<PIPE_MTE2>(1);  // #34
    // sync.mutex cv id=2: the producer side; its 2 tokens are drained before the return  // #35
    SEvent<PIPE_MTE1, PIPE_MTE2, 1, 4> q_input_free;  // #36
    SEvent<PIPE_MTE2, PIPE_MTE1, 0, 4> q_input_ready;  // #37
    DEvent<PIPE_MTE2, PIPE_MTE1, 0, 0, 1> k_local_ready;  // #39
    DEvent<PIPE_MTE1, PIPE_MTE2, 2, 0, 1> k_local_free;  // #40
    DEvent<PIPE_MTE2, PIPE_MTE1, 0, 2, 3> pv_local_ready;  // #41
    DEvent<PIPE_MTE1, PIPE_MTE2, 2, 2, 3> pv_local_free;  // #42
    DEvent<PIPE_MTE1, PIPE_M, 0, 0, 1> ab_local_ready;  // #43
    DEvent<PIPE_M, PIPE_MTE1, 2, 0, 1> ab_local_free;  // #44
    DEvent<PIPE_M, PIPE_FIX, 0, 0, 1> c_local_ready;  // #45
    DEvent<PIPE_FIX, PIPE_M, 2, 0, 1> c_local_free;  // #46
    const int32_t cube_idx = GetCubeIdx();  // #50
    const int32_t cube_count = GetCubeNum();  // #51
    const int32_t core_step = (int32_t)(cube_count);  // #52
    const int32_t bh_total = B * H;  // #55
    const int32_t total_tasks = bh_total * NT_Q;  // #56
    int32_t l0ab_cnt_17_state = 0;  // #1557
    int32_t l0c_cnt_12_state = 0;  // #1558
    int32_t l1k_cnt_10_state = 0;  // #1559
    int32_t l1pv_cnt_9_state = 0;  // #1560
    for (int32_t linear_id = cube_idx; linear_id < total_tasks; linear_id += core_step) {  // #1801
        const int32_t l0ab_cnt_carry = (int32_t)(l0ab_cnt_17_state);  // #1561
        const int32_t l0c_cnt_carry = (int32_t)(l0c_cnt_12_state);  // #1562
        const int32_t l1k_cnt_carry = (int32_t)(l1k_cnt_10_state);  // #1563
        const int32_t l1pv_cnt_carry = (int32_t)(l1pv_cnt_9_state);  // #1564
        const int32_t idiv_1 = linear_id / NT_Q;  // #3102
        const int32_t idiv_2 = linear_id % NT_Q;  // #3103
        const bool idiv_3 = idiv_2 != 0;  // #3104
        const bool idiv_4 = linear_id < 0;  // #3105
        const bool idiv_6 = idiv_4 != false;  // #3107
        const bool idiv_7 = idiv_3 && idiv_6;  // #3108
        const int32_t idiv_8 = (idiv_7) ? (1) : (0);  // #3109
        const int32_t bh = idiv_1 - idiv_8;  // #61
        const int32_t q_idx = ascrip::FloorMod<int32_t>(linear_id, NT_Q);  // #62
        const int32_t idiv_9 = bh / H;  // #3110
        const int32_t idiv_10 = bh % H;  // #3111
        const bool idiv_11 = idiv_10 != 0;  // #3112
        const bool idiv_12 = bh < 0;  // #3113
        const bool idiv_14 = idiv_12 != false;  // #3115
        const bool idiv_15 = idiv_11 && idiv_14;  // #3116
        const int32_t idiv_16 = (idiv_15) ? (1) : (0);  // #3117
        const int32_t b = idiv_9 - idiv_16;  // #63
        const int32_t h = ascrip::FloorMod<int32_t>(bh, H);  // #64
        const int32_t q_start = q_idx * 128;  // #65
        const int32_t valid_m = Min(128, T - q_start);  // #66 #67
        q_input_free.wait();  // #73
        if (valid_m < 128) {  // #74 #77
            set_constant_to_l1(l1q, (bfloat16_t)(0.0f), 1024);  // #75
            PipeBarrier<PIPE_MTE2>();  // #76
        }
        const GMTensor<bfloat16_t> view = q[(((b * ((T * H) * 128)) + (q_start * (H * 128))) + (h * 128))];  // #80
        gm_to_l1_nd2nz(l1q, view, valid_m, 128, 128, H * 128);  // #1802 #81
        q_input_ready.set();  // #82
        q_input_ready.wait();  // #83
        const int32_t exact_total = flat_exact_cnt.load(((b * (H * NT_Q_PAD8)) + (h * NT_Q_PAD8)) + q_idx);  // #94 #95 #96 #97 #98 #99
        const int32_t idiv_17 = NG / 6;  // #3118
        const int32_t idiv_18 = NG % 6;  // #3119
        const bool idiv_19 = idiv_18 != 0;  // #3120
        const bool idiv_23 = idiv_19 && true;  // #3124
        const int32_t idiv_24 = (idiv_23) ? (1) : (0);  // #3125
        const int32_t approx_beats_1 = idiv_17 + idiv_24;  // #100
        const int32_t idiv_25 = exact_total / 6;  // #3126
        const int32_t idiv_26 = exact_total % 6;  // #3127
        const bool idiv_27 = idiv_26 != 0;  // #3128
        const bool idiv_28 = exact_total < 0;  // #3129
        const bool idiv_30 = idiv_28 == false;  // #3131
        const bool idiv_31 = idiv_27 && idiv_30;  // #3132
        const int32_t idiv_32 = (idiv_31) ? (1) : (0);  // #3133
        const int32_t ceil_div = idiv_25 + idiv_32;  // #101
        const int32_t total_beats = approx_beats_1 + ceil_div;  // #102
        const int32_t add_3 = total_beats + 3;  // #108
        int32_t accum_step_3_state = 0;  // #1565
        int32_t l0ab_cnt_16_state = l0ab_cnt_carry;  // #1566
        int32_t l0c_cnt_11_state = l0c_cnt_carry;  // #1567
        int32_t l1k_cnt_9_state = l1k_cnt_carry;  // #1568
        int32_t l1pv_cnt_8_state = l1pv_cnt_carry;  // #1569
        int32_t ready_lengths_2_state = 0;  // #1570
        for (int32_t beat = 0; beat < add_3; beat += 1) {  // #1791
            const int32_t accum_step_carry = (int32_t)(accum_step_3_state);  // #1571
            const int32_t l0ab_cnt_carry_1 = (int32_t)(l0ab_cnt_16_state);  // #1572
            const int32_t l0c_cnt_carry_1 = (int32_t)(l0c_cnt_11_state);  // #1573
            const int32_t l1k_cnt_carry_1 = (int32_t)(l1k_cnt_9_state);  // #1574
            const int32_t l1pv_cnt_carry_1 = (int32_t)(l1pv_cnt_8_state);  // #1575
            const int32_t ready_lengths_carry = (int32_t)(ready_lengths_2_state);  // #1576
            const bool cmp_1 = beat < total_beats;  // #110
            int32_t b_len_p_5_state = 0;  // #1577
            int32_t l0ab_cnt_8_state = 0;  // #1578
            int32_t l0c_cnt_8_state = 0;  // #1579
            int32_t l1k_cnt_8_state = 0;  // #1580
            if (cmp_1) {  // #1187
                const bool approx_p = beat < approx_beats_1;  // #111
                const int32_t approx_g_p = beat * 6;  // #112
                const int32_t e_base_p = Max(0, beat - approx_beats_1) * 6;  // #121 #122 #123
                const int32_t b_len_p_1 = Min(6, exact_total - e_base_p);  // #124 #125
                int32_t b_len_p_4_state = 0;  // #1581
                if (approx_p) {  // #244
                    const bool cmp_2 = approx_g_p < NG;  // #128
                    uint64_t mask_0_1_state = 0ULL;  // #1588
                    if (cmp_2) {  // #142
                        mask_0_1_state = group_approx_mask.load((((b * ((H * NT_Q) * NG)) + (h * (NT_Q * NG))) + (q_idx * NG)) + approx_g_p);  // #130 #131 #132 #133 #134 #135 #136 #137 #138 #139 #1589
                    } else {
                        mask_0_1_state = 0ULL;  // #1590
                    }
                    const uint64_t mask_0_1 = (uint64_t)(mask_0_1_state);  // #1591
                    const int32_t add_8 = approx_g_p + 1;  // #143
                    const bool cmp_3 = add_8 < NG;  // #145
                    uint64_t mask_1_1_state = 0ULL;  // #1592
                    if (cmp_3) {  // #159
                        mask_1_1_state = group_approx_mask.load((((b * ((H * NT_Q) * NG)) + (h * (NT_Q * NG))) + (q_idx * NG)) + add_8);  // #147 #148 #149 #150 #151 #152 #153 #154 #155 #156 #1593
                    } else {
                        mask_1_1_state = 0ULL;  // #1594
                    }
                    const uint64_t mask_1_1 = (uint64_t)(mask_1_1_state);  // #1595
                    const int32_t add_12 = approx_g_p + 2;  // #160
                    const bool cmp_4 = add_12 < NG;  // #162
                    uint64_t mask_2_1_state = 0ULL;  // #1596
                    if (cmp_4) {  // #176
                        mask_2_1_state = group_approx_mask.load((((b * ((H * NT_Q) * NG)) + (h * (NT_Q * NG))) + (q_idx * NG)) + add_12);  // #164 #165 #166 #167 #168 #169 #170 #171 #172 #173 #1597
                    } else {
                        mask_2_1_state = 0ULL;  // #1598
                    }
                    const uint64_t mask_2_1 = (uint64_t)(mask_2_1_state);  // #1599
                    const int32_t add_16 = approx_g_p + 3;  // #177
                    const bool cmp_5 = add_16 < NG;  // #179
                    uint64_t mask_3_1_state = 0ULL;  // #1600
                    if (cmp_5) {  // #193
                        mask_3_1_state = group_approx_mask.load((((b * ((H * NT_Q) * NG)) + (h * (NT_Q * NG))) + (q_idx * NG)) + add_16);  // #181 #182 #183 #184 #185 #186 #187 #188 #189 #190 #1601
                    } else {
                        mask_3_1_state = 0ULL;  // #1602
                    }
                    const uint64_t mask_3_1 = (uint64_t)(mask_3_1_state);  // #1603
                    const int32_t add_20 = approx_g_p + 4;  // #194
                    const bool cmp_6 = add_20 < NG;  // #196
                    uint64_t mask_4_1_state = 0ULL;  // #1604
                    if (cmp_6) {  // #210
                        mask_4_1_state = group_approx_mask.load((((b * ((H * NT_Q) * NG)) + (h * (NT_Q * NG))) + (q_idx * NG)) + add_20);  // #198 #199 #200 #201 #202 #203 #204 #205 #206 #207 #1605
                    } else {
                        mask_4_1_state = 0ULL;  // #1606
                    }
                    const uint64_t mask_4_1 = (uint64_t)(mask_4_1_state);  // #1607
                    const int32_t add_24 = approx_g_p + 5;  // #211
                    const bool cmp_7 = add_24 < NG;  // #213
                    uint64_t mask_5_1_state = 0ULL;  // #1608
                    if (cmp_7) {  // #227
                        mask_5_1_state = group_approx_mask.load((((b * ((H * NT_Q) * NG)) + (h * (NT_Q * NG))) + (q_idx * NG)) + add_24);  // #215 #216 #217 #218 #219 #220 #221 #222 #223 #224 #1609
                    } else {
                        mask_5_1_state = 0ULL;  // #1610
                    }
                    const uint64_t mask_5_1 = (uint64_t)(mask_5_1_state);  // #1611
                    const bool cmp_8 = (((((((mask_0_1 | mask_1_1) | mask_2_1) | mask_3_1) | mask_4_1) | mask_5_1) | 0ULL) | 0ULL) != 0ULL;  // #228 #229 #230 #231 #232 #233 #234 #236
                    int32_t b_len_p_3_state = 0;  // #1612
                    if (cmp_8) {  // #241
                        b_len_p_3_state = Min(6, NG - approx_g_p);  // #237 #238 #1613
                    } else {
                        b_len_p_3_state = 0;  // #1614
                    }
                    b_len_p_4_state = (int32_t)(b_len_p_3_state);  // #1615 #1616
                } else {
                    b_len_p_4_state = b_len_p_1;  // #1623
                }
                const int32_t b_len_p_4 = (int32_t)(b_len_p_4_state);  // #1630
                const bool cmp_9 = b_len_p_4 > 0;  // #245
                int32_t l0ab_cnt_7_state = 0;  // #1637
                int32_t l0c_cnt_7_state = 0;  // #1638
                int32_t l1k_cnt_7_state = 0;  // #1639
                if (cmp_9) {  // #1184
                    const int32_t slot_b_p = beat & 3;  // #246
                    const int32_t pair_len = Min(2, b_len_p_4);  // #248
                    const bool cmp_10 = pair_len > 0;  // #249
                    int32_t l0ab_cnt_2_state = 0;  // #1640
                    int32_t l0c_cnt_2_state = 0;  // #1641
                    int32_t l1k_cnt_2_state = 0;  // #1642
                    if (cmp_10) {  // #345
                        k_local_free.wait();  // #250
                        if (approx_p) {  // #314
                            const int32_t first = approx_g_p * 64;  // #252
                            const int32_t cols = pair_len * 64;  // #253
                            if (pair_len < 2) {  // #254 #258
                                const Tensor<bfloat16_t, Position::L1> slot = l1k.get(l1k_cnt_carry_1);  // #255
                                set_constant_to_l1(slot, (bfloat16_t)(0.0f), 1024);  // #256
                                PipeBarrier<PIPE_MTE2>();  // #257
                            }
                            const Tensor<bfloat16_t, Position::L1> slot_1 = l1k.get(l1k_cnt_carry_1);  // #259
                            const Tensor<bfloat16_t, Position::L1> view_1 = l1k.get(l1k_cnt_carry_1);  // #260
                            const int32_t sub_7 = (first + cols) - first;  // #261 #262
                            const GMTensor<bfloat16_t> view_2 = kc_padded[(((b * ((H * NPAD) * 128)) + (h * (NPAD * 128))) + (first * 128))];  // #263
                            gm_to_l1_nd2nz(view_1, view_2, sub_7, 128, 128, 128);  // #264
                        } else {
                            if (pair_len < 2) {  // #265 #269
                                const Tensor<bfloat16_t, Position::L1> slot_2 = l1k.get(l1k_cnt_carry_1);  // #266
                                set_constant_to_l1(slot_2, (bfloat16_t)(0.0f), 1024);  // #267
                                PipeBarrier<PIPE_MTE2>();  // #268
                            }
                            if (0 < pair_len) {  // #270 #291
                                const int32_t blk_p = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + e_base_p));  // #273 #274 #275 #276 #277 #278 #279 #280 #281 #282 #283
                                const int32_t kv_off_p = blk_p * 64;  // #284
                                const Tensor<bfloat16_t, Position::L1> slot_3 = l1k.get(l1k_cnt_carry_1);  // #285
                                const Tensor<bfloat16_t, Position::L1> view_3 = l1k.get(l1k_cnt_carry_1);  // #286
                                const GMTensor<bfloat16_t> view_4 = k_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off_p * 128))];  // #289
                                gm_to_l1_nd2nz(view_3, view_4, 64, 128, 128, 128);  // #290
                            }
                            if (1 < pair_len) {  // #292 #313
                                const int32_t add_35 = e_base_p + 1;  // #293
                                const int32_t blk_p_1 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + add_35));  // #295 #296 #297 #298 #299 #300 #301 #302 #303 #304 #305
                                const int32_t kv_off_p_1 = blk_p_1 * 64;  // #306
                                const Tensor<bfloat16_t, Position::L1> slot_4 = l1k.get(l1k_cnt_carry_1);  // #307
                                const Tensor<bfloat16_t, Position::L1> view_5 = l1k.get(l1k_cnt_carry_1)[1024];  // #308
                                const GMTensor<bfloat16_t> view_6 = k_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off_p_1 * 128))];  // #311
                                gm_to_l1_nd2nz(view_5, view_6, 64, 128, 128, 128);  // #312
                            }
                        }
                        k_local_ready.set();  // #315
                        ab_local_free.wait();  // #316
                        k_local_ready.wait();  // #317
                        const Tensor<bfloat16_t, Position::L0A> q_tile_l0 = l0a.get(l0ab_cnt_carry_1);  // #318
                        l1_to_l0<false>(q_tile_l0, l1q, 128, 128, 128, 128);  // #319
                        const Tensor<bfloat16_t, Position::L0B> slot_5 = l0b.get(l0ab_cnt_carry_1);  // #320
                        const Tensor<bfloat16_t, Position::L1> slot_6 = l1k.get(l1k_cnt_carry_1);  // #321
                        l1_to_l0<false>(slot_5, slot_6, 128, 128, 128, 128);  // #322
                        k_local_free.set();  // #323
                        ab_local_ready.set();  // #324
                        c_local_free.wait();  // #325
                        ab_local_ready.wait();  // #326
                        const Tensor<float, Position::L0C> slot_7 = l0c.get(l0c_cnt_carry_1);  // #327
                        const Tensor<bfloat16_t, Position::L0B> slot_8 = l0b.get(l0ab_cnt_carry_1);  // #328
                        mmad(slot_7, q_tile_l0, slot_8, 128, 128, 128, true);  // #329
                        ab_local_free.set();  // #330
                        const int32_t l0ab_cnt_1 = (int32_t)(l0ab_cnt_carry_1 + 1);  // #331 #332
                        c_local_ready.set();  // #333
                        c_local_ready.wait();  // #334
                        const GMTensor<float> view_7 = score_ws[((cube_idx * 196608) + (slot_b_p * 49152))];  // #335
                        const Tensor<float, Position::L0C> slot_9 = l0c.get(l0c_cnt_carry_1);  // #336
                        l0c_to_gm_nz2nd(view_7, slot_9, 128, 128, 128, 128, false, 0.0f, 0, false, false);  // #337
                        c_local_free.set();  // #338
                        const int32_t l1k_cnt_1 = (int32_t)(l1k_cnt_carry_1 + 1);  // #339 #340
                        const int32_t l0c_cnt_1 = (int32_t)(l0c_cnt_carry_1 + 1);  // #341 #342
                        l0ab_cnt_2_state = l0ab_cnt_1;  // #1643
                        l0c_cnt_2_state = l0c_cnt_1;  // #1644
                        l1k_cnt_2_state = l1k_cnt_1;  // #1645
                    } else {
                        l0ab_cnt_2_state = l0ab_cnt_carry_1;  // #1646
                        l0c_cnt_2_state = l0c_cnt_carry_1;  // #1647
                        l1k_cnt_2_state = l1k_cnt_carry_1;  // #1648
                    }
                    const int32_t l0ab_cnt_2 = (int32_t)(l0ab_cnt_2_state);  // #1649
                    const int32_t l0c_cnt_2 = (int32_t)(l0c_cnt_2_state);  // #1650
                    const int32_t l1k_cnt_2 = (int32_t)(l1k_cnt_2_state);  // #1651
                    const int32_t pair_len_1 = Min(2, b_len_p_4 - 2);  // #346 #347
                    const bool cmp_15 = pair_len_1 > 0;  // #348
                    int32_t l0ab_cnt_4_state = 0;  // #1652
                    int32_t l0c_cnt_4_state = 0;  // #1653
                    int32_t l1k_cnt_4_state = 0;  // #1654
                    if (cmp_15) {  // #444
                        k_local_free.wait();  // #349
                        if (approx_p) {  // #413
                            const int32_t first_1 = (approx_g_p + 2) * 64;  // #350 #351
                            const int32_t cols_1 = pair_len_1 * 64;  // #352
                            if (pair_len_1 < 2) {  // #353 #357
                                const Tensor<bfloat16_t, Position::L1> slot_10 = l1k.get(l1k_cnt_2);  // #354
                                set_constant_to_l1(slot_10, (bfloat16_t)(0.0f), 1024);  // #355
                                PipeBarrier<PIPE_MTE2>();  // #356
                            }
                            const Tensor<bfloat16_t, Position::L1> slot_11 = l1k.get(l1k_cnt_2);  // #358
                            const Tensor<bfloat16_t, Position::L1> view_8 = l1k.get(l1k_cnt_2);  // #359
                            const int32_t sub_11 = (first_1 + cols_1) - first_1;  // #360 #361
                            const GMTensor<bfloat16_t> view_9 = kc_padded[(((b * ((H * NPAD) * 128)) + (h * (NPAD * 128))) + (first_1 * 128))];  // #362
                            gm_to_l1_nd2nz(view_8, view_9, sub_11, 128, 128, 128);  // #363
                        } else {
                            if (pair_len_1 < 2) {  // #364 #368
                                const Tensor<bfloat16_t, Position::L1> slot_12 = l1k.get(l1k_cnt_2);  // #365
                                set_constant_to_l1(slot_12, (bfloat16_t)(0.0f), 1024);  // #366
                                PipeBarrier<PIPE_MTE2>();  // #367
                            }
                            if (0 < pair_len_1) {  // #369 #390
                                const int32_t add_45 = e_base_p + 2;  // #370
                                const int32_t blk_p_2 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + add_45));  // #372 #373 #374 #375 #376 #377 #378 #379 #380 #381 #382
                                const int32_t kv_off_p_2 = blk_p_2 * 64;  // #383
                                const Tensor<bfloat16_t, Position::L1> slot_13 = l1k.get(l1k_cnt_2);  // #384
                                const Tensor<bfloat16_t, Position::L1> view_10 = l1k.get(l1k_cnt_2);  // #385
                                const GMTensor<bfloat16_t> view_11 = k_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off_p_2 * 128))];  // #388
                                gm_to_l1_nd2nz(view_10, view_11, 64, 128, 128, 128);  // #389
                            }
                            if (1 < pair_len_1) {  // #391 #412
                                const int32_t add_50 = e_base_p + 3;  // #392
                                const int32_t blk_p_3 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + add_50));  // #394 #395 #396 #397 #398 #399 #400 #401 #402 #403 #404
                                const int32_t kv_off_p_3 = blk_p_3 * 64;  // #405
                                const Tensor<bfloat16_t, Position::L1> slot_14 = l1k.get(l1k_cnt_2);  // #406
                                const Tensor<bfloat16_t, Position::L1> view_12 = l1k.get(l1k_cnt_2)[1024];  // #407
                                const GMTensor<bfloat16_t> view_13 = k_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off_p_3 * 128))];  // #410
                                gm_to_l1_nd2nz(view_12, view_13, 64, 128, 128, 128);  // #411
                            }
                        }
                        k_local_ready.set();  // #414
                        ab_local_free.wait();  // #415
                        k_local_ready.wait();  // #416
                        const Tensor<bfloat16_t, Position::L0A> q_tile_l0_1 = l0a.get(l0ab_cnt_2);  // #417
                        l1_to_l0<false>(q_tile_l0_1, l1q, 128, 128, 128, 128);  // #418
                        const Tensor<bfloat16_t, Position::L0B> slot_15 = l0b.get(l0ab_cnt_2);  // #419
                        const Tensor<bfloat16_t, Position::L1> slot_16 = l1k.get(l1k_cnt_2);  // #420
                        l1_to_l0<false>(slot_15, slot_16, 128, 128, 128, 128);  // #421
                        k_local_free.set();  // #422
                        ab_local_ready.set();  // #423
                        c_local_free.wait();  // #424
                        ab_local_ready.wait();  // #425
                        const Tensor<float, Position::L0C> slot_17 = l0c.get(l0c_cnt_2);  // #426
                        const Tensor<bfloat16_t, Position::L0B> slot_18 = l0b.get(l0ab_cnt_2);  // #427
                        mmad(slot_17, q_tile_l0_1, slot_18, 128, 128, 128, true);  // #428
                        ab_local_free.set();  // #429
                        const int32_t l0ab_cnt_3 = (int32_t)(l0ab_cnt_2 + 1);  // #430 #431
                        c_local_ready.set();  // #432
                        c_local_ready.wait();  // #433
                        const GMTensor<float> view_14 = score_ws[(((cube_idx * 196608) + (slot_b_p * 49152)) + 16384)];  // #434
                        const Tensor<float, Position::L0C> slot_19 = l0c.get(l0c_cnt_2);  // #435
                        l0c_to_gm_nz2nd(view_14, slot_19, 128, 128, 128, 128, false, 0.0f, 0, false, false);  // #436
                        c_local_free.set();  // #437
                        const int32_t l1k_cnt_3 = (int32_t)(l1k_cnt_2 + 1);  // #438 #439
                        const int32_t l0c_cnt_3 = (int32_t)(l0c_cnt_2 + 1);  // #440 #441
                        l0ab_cnt_4_state = l0ab_cnt_3;  // #1655
                        l0c_cnt_4_state = l0c_cnt_3;  // #1656
                        l1k_cnt_4_state = l1k_cnt_3;  // #1657
                    } else {
                        l0ab_cnt_4_state = l0ab_cnt_2;  // #1658
                        l0c_cnt_4_state = l0c_cnt_2;  // #1659
                        l1k_cnt_4_state = l1k_cnt_2;  // #1660
                    }
                    const int32_t l0ab_cnt_4 = (int32_t)(l0ab_cnt_4_state);  // #1661
                    const int32_t l0c_cnt_4 = (int32_t)(l0c_cnt_4_state);  // #1662
                    const int32_t l1k_cnt_4 = (int32_t)(l1k_cnt_4_state);  // #1663
                    const int32_t pair_len_2 = Min(2, b_len_p_4 - 4);  // #445 #446
                    const bool cmp_20 = pair_len_2 > 0;  // #447
                    int32_t l0ab_cnt_6_state = 0;  // #1664
                    int32_t l0c_cnt_6_state = 0;  // #1665
                    int32_t l1k_cnt_6_state = 0;  // #1666
                    if (cmp_20) {  // #543
                        k_local_free.wait();  // #448
                        if (approx_p) {  // #512
                            const int32_t first_2 = (approx_g_p + 4) * 64;  // #449 #450
                            const int32_t cols_2 = pair_len_2 * 64;  // #451
                            if (pair_len_2 < 2) {  // #452 #456
                                const Tensor<bfloat16_t, Position::L1> slot_20 = l1k.get(l1k_cnt_4);  // #453
                                set_constant_to_l1(slot_20, (bfloat16_t)(0.0f), 1024);  // #454
                                PipeBarrier<PIPE_MTE2>();  // #455
                            }
                            const Tensor<bfloat16_t, Position::L1> slot_21 = l1k.get(l1k_cnt_4);  // #457
                            const Tensor<bfloat16_t, Position::L1> view_15 = l1k.get(l1k_cnt_4);  // #458
                            const int32_t sub_15 = (first_2 + cols_2) - first_2;  // #459 #460
                            const GMTensor<bfloat16_t> view_16 = kc_padded[(((b * ((H * NPAD) * 128)) + (h * (NPAD * 128))) + (first_2 * 128))];  // #461
                            gm_to_l1_nd2nz(view_15, view_16, sub_15, 128, 128, 128);  // #462
                        } else {
                            if (pair_len_2 < 2) {  // #463 #467
                                const Tensor<bfloat16_t, Position::L1> slot_22 = l1k.get(l1k_cnt_4);  // #464
                                set_constant_to_l1(slot_22, (bfloat16_t)(0.0f), 1024);  // #465
                                PipeBarrier<PIPE_MTE2>();  // #466
                            }
                            if (0 < pair_len_2) {  // #468 #489
                                const int32_t add_60 = e_base_p + 4;  // #469
                                const int32_t blk_p_4 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + add_60));  // #471 #472 #473 #474 #475 #476 #477 #478 #479 #480 #481
                                const int32_t kv_off_p_4 = blk_p_4 * 64;  // #482
                                const Tensor<bfloat16_t, Position::L1> slot_23 = l1k.get(l1k_cnt_4);  // #483
                                const Tensor<bfloat16_t, Position::L1> view_17 = l1k.get(l1k_cnt_4);  // #484
                                const GMTensor<bfloat16_t> view_18 = k_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off_p_4 * 128))];  // #487
                                gm_to_l1_nd2nz(view_17, view_18, 64, 128, 128, 128);  // #488
                            }
                            if (1 < pair_len_2) {  // #490 #511
                                const int32_t add_65 = e_base_p + 5;  // #491
                                const int32_t blk_p_5 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + add_65));  // #493 #494 #495 #496 #497 #498 #499 #500 #501 #502 #503
                                const int32_t kv_off_p_5 = blk_p_5 * 64;  // #504
                                const Tensor<bfloat16_t, Position::L1> slot_24 = l1k.get(l1k_cnt_4);  // #505
                                const Tensor<bfloat16_t, Position::L1> view_19 = l1k.get(l1k_cnt_4)[1024];  // #506
                                const GMTensor<bfloat16_t> view_20 = k_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off_p_5 * 128))];  // #509
                                gm_to_l1_nd2nz(view_19, view_20, 64, 128, 128, 128);  // #510
                            }
                        }
                        k_local_ready.set();  // #513
                        ab_local_free.wait();  // #514
                        k_local_ready.wait();  // #515
                        const Tensor<bfloat16_t, Position::L0A> q_tile_l0_2 = l0a.get(l0ab_cnt_4);  // #516
                        l1_to_l0<false>(q_tile_l0_2, l1q, 128, 128, 128, 128);  // #517
                        const Tensor<bfloat16_t, Position::L0B> slot_25 = l0b.get(l0ab_cnt_4);  // #518
                        const Tensor<bfloat16_t, Position::L1> slot_26 = l1k.get(l1k_cnt_4);  // #519
                        l1_to_l0<false>(slot_25, slot_26, 128, 128, 128, 128);  // #520
                        k_local_free.set();  // #521
                        ab_local_ready.set();  // #522
                        c_local_free.wait();  // #523
                        ab_local_ready.wait();  // #524
                        const Tensor<float, Position::L0C> slot_27 = l0c.get(l0c_cnt_4);  // #525
                        const Tensor<bfloat16_t, Position::L0B> slot_28 = l0b.get(l0ab_cnt_4);  // #526
                        mmad(slot_27, q_tile_l0_2, slot_28, 128, 128, 128, true);  // #527
                        ab_local_free.set();  // #528
                        const int32_t l0ab_cnt_5 = (int32_t)(l0ab_cnt_4 + 1);  // #529 #530
                        c_local_ready.set();  // #531
                        c_local_ready.wait();  // #532
                        const GMTensor<float> view_21 = score_ws[(((cube_idx * 196608) + (slot_b_p * 49152)) + 32768)];  // #533
                        const Tensor<float, Position::L0C> slot_29 = l0c.get(l0c_cnt_4);  // #534
                        l0c_to_gm_nz2nd(view_21, slot_29, 128, 128, 128, 128, false, 0.0f, 0, false, false);  // #535
                        c_local_free.set();  // #536
                        const int32_t l1k_cnt_5 = (int32_t)(l1k_cnt_4 + 1);  // #537 #538
                        const int32_t l0c_cnt_5 = (int32_t)(l0c_cnt_4 + 1);  // #539 #540
                        l0ab_cnt_6_state = l0ab_cnt_5;  // #1667
                        l0c_cnt_6_state = l0c_cnt_5;  // #1668
                        l1k_cnt_6_state = l1k_cnt_5;  // #1669
                    } else {
                        l0ab_cnt_6_state = l0ab_cnt_4;  // #1670
                        l0c_cnt_6_state = l0c_cnt_4;  // #1671
                        l1k_cnt_6_state = l1k_cnt_4;  // #1672
                    }
                    const int32_t l0ab_cnt_6 = (int32_t)(l0ab_cnt_6_state);  // #1673
                    const int32_t l0c_cnt_6 = (int32_t)(l0c_cnt_6_state);  // #1674
                    const int32_t l1k_cnt_6 = (int32_t)(l1k_cnt_6_state);  // #1675
                    CUBE_READY<PIPE_FIX>(0);  // #544
                    const int32_t pair_len_6 = Min(2, b_len_p_4);  // #585
                    if (pair_len_6 > 0) {  // #586 #681
                        if (approx_p) {  // #670
                        } else {
                            if (0 < pair_len_6) {  // #603 #635
                                const uint16_t load_6 = flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + e_base_p);  // #607 #608 #609 #610 #611 #612 #613 #614 #615 #616
                            }
                            if (1 < pair_len_6) {  // #637 #669
                                const int32_t blk_p_8 = e_base_p + 1;  // #639
                                const uint16_t load_7 = flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + blk_p_8);  // #641 #642 #643 #644 #645 #646 #647 #648 #649 #650
                            }
                        }
                    }
                    const int32_t pair_len_7 = Min(2, b_len_p_4 - 2);  // #682 #683
                    if (pair_len_7 > 0) {  // #684 #779
                        if (approx_p) {  // #768
                        } else {
                            if (0 < pair_len_7) {  // #701 #733
                                const int32_t add_84 = e_base_p + 2;  // #702
                                const uint16_t load_8 = flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + add_84);  // #705 #706 #707 #708 #709 #710 #711 #712 #713 #714
                            }
                            if (1 < pair_len_7) {  // #735 #767
                                const int32_t blk_p_12 = (e_base_p + 2) + 1;  // #736 #737
                                const uint16_t load_9 = flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + blk_p_12);  // #739 #740 #741 #742 #743 #744 #745 #746 #747 #748
                            }
                        }
                    }
                    const int32_t pair_len_8 = Min(2, b_len_p_4 - 4);  // #780 #781
                    if (pair_len_8 > 0) {  // #782 #877
                        if (approx_p) {  // #866
                        } else {
                            if (0 < pair_len_8) {  // #799 #831
                                const int32_t add_92 = e_base_p + 4;  // #800
                                const uint16_t load_10 = flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + add_92);  // #803 #804 #805 #806 #807 #808 #809 #810 #811 #812
                            }
                            if (1 < pair_len_8) {  // #833 #865
                                const int32_t blk_p_16 = (e_base_p + 4) + 1;  // #834 #835
                                const uint16_t load_11 = flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + blk_p_16);  // #837 #838 #839 #840 #841 #842 #843 #844 #845 #846
                            }
                        }
                    }
                    l0ab_cnt_7_state = l0ab_cnt_6;  // #1724
                    l0c_cnt_7_state = l0c_cnt_6;  // #1725
                    l1k_cnt_7_state = l1k_cnt_6;  // #1726
                } else {
                    l0ab_cnt_7_state = l0ab_cnt_carry_1;  // #1727
                    l0c_cnt_7_state = l0c_cnt_carry_1;  // #1728
                    l1k_cnt_7_state = l1k_cnt_carry_1;  // #1729
                }
                const int32_t l0ab_cnt_7 = (int32_t)(l0ab_cnt_7_state);  // #1730
                const int32_t l0c_cnt_7 = (int32_t)(l0c_cnt_7_state);  // #1731
                const int32_t l1k_cnt_7 = (int32_t)(l1k_cnt_7_state);  // #1732
                b_len_p_5_state = b_len_p_4;  // #1733
                l0ab_cnt_8_state = l0ab_cnt_7;  // #1734
                l0c_cnt_8_state = l0c_cnt_7;  // #1735
                l1k_cnt_8_state = l1k_cnt_7;  // #1736
            } else {
                b_len_p_5_state = 0;  // #1737
                l0ab_cnt_8_state = l0ab_cnt_carry_1;  // #1738
                l0c_cnt_8_state = l0c_cnt_carry_1;  // #1739
                l1k_cnt_8_state = l1k_cnt_carry_1;  // #1740
            }
            const int32_t b_len_p_5 = (int32_t)(b_len_p_5_state);  // #1741
            const int32_t l0ab_cnt_8 = (int32_t)(l0ab_cnt_8_state);  // #1742
            const int32_t l0c_cnt_8 = (int32_t)(l0c_cnt_8_state);  // #1743
            const int32_t l1k_cnt_8 = (int32_t)(l1k_cnt_8_state);  // #1744
            const int32_t ready_lengths_1 = ((ready_lengths_carry & 4095) << 4) | b_len_p_5;  // #1188 #1189 #1190
            const int32_t b_len_c = ready_lengths_1 >> 12;  // #1191
            const bool cmp_64 = b_len_c > 0;  // #1192
            int32_t accum_step_2_state = 0;  // #1745
            int32_t l0ab_cnt_15_state = 0;  // #1746
            int32_t l0c_cnt_10_state = 0;  // #1747
            int32_t l1pv_cnt_7_state = 0;  // #1748
            if (cmp_64) {  // #1519
                const int32_t consumed = beat + -3;  // #1193
                const int32_t slot_30 = consumed & 3;  // #1194
                const bool approx_c = consumed < approx_beats_1;  // #1195
                const int32_t approx_g_c = consumed * 6;  // #1196
                const int32_t e_base_c = Max(0, consumed - approx_beats_1) * 6;  // #1197 #1198 #1199
                WAIT_VEC<PIPE_S>(1);  // #1200
                c_local_free.wait();  // #1201
                const int32_t pair_len_12 = Min(2, b_len_c);  // #1203
                const bool cmp_65 = pair_len_12 > 0;  // #1204
                int32_t l0ab_cnt_10_state = 0;  // #1749
                int32_t l1pv_cnt_2_state = 0;  // #1750
                if (cmp_65) {  // #1296
                    pv_local_free.wait();  // #1205
                    const Tensor<bfloat16_t, Position::L1> slot_31 = l1p.get(l1pv_cnt_carry_1);  // #1206
                    const GMTensor<bfloat16_t> view_52 = p_ws[((cube_idx * 196608) + (slot_30 * 49152))];  // #1207
                    gm_to_l1_nd2nz(slot_31, view_52, 128, 128, 128, 128);  // #1208
                    if (approx_c) {  // #1272
                        const int32_t first_3 = approx_g_c * 64;  // #1210
                        const int32_t cols_3 = pair_len_12 * 64;  // #1211
                        if (pair_len_12 < 2) {  // #1212 #1216
                            const Tensor<bfloat16_t, Position::L1> slot_32 = l1v.get(l1pv_cnt_carry_1);  // #1213
                            set_constant_to_l1(slot_32, (bfloat16_t)(0.0f), 1024);  // #1214
                            PipeBarrier<PIPE_MTE2>();  // #1215
                        }
                        const Tensor<bfloat16_t, Position::L1> slot_33 = l1v.get(l1pv_cnt_carry_1);  // #1217
                        const Tensor<bfloat16_t, Position::L1> view_53 = l1v.get(l1pv_cnt_carry_1);  // #1218
                        const int32_t sub_53 = (first_3 + cols_3) - first_3;  // #1219 #1220
                        const GMTensor<bfloat16_t> view_54 = vc_padded[(((b * ((H * NPAD) * 128)) + (h * (NPAD * 128))) + (first_3 * 128))];  // #1221
                        gm_to_l1_nd2nz(view_53, view_54, sub_53, 128, 128, 128);  // #1222
                    } else {
                        if (pair_len_12 < 2) {  // #1223 #1227
                            const Tensor<bfloat16_t, Position::L1> slot_34 = l1v.get(l1pv_cnt_carry_1);  // #1224
                            set_constant_to_l1(slot_34, (bfloat16_t)(0.0f), 1024);  // #1225
                            PipeBarrier<PIPE_MTE2>();  // #1226
                        }
                        if (0 < pair_len_12) {  // #1228 #1249
                            const int32_t blk_c = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + e_base_c));  // #1231 #1232 #1233 #1234 #1235 #1236 #1237 #1238 #1239 #1240 #1241
                            const int32_t kv_off_c = blk_c * 64;  // #1242
                            const Tensor<bfloat16_t, Position::L1> slot_35 = l1v.get(l1pv_cnt_carry_1);  // #1243
                            const Tensor<bfloat16_t, Position::L1> view_55 = l1v.get(l1pv_cnt_carry_1);  // #1244
                            const GMTensor<bfloat16_t> view_56 = v_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off_c * 128))];  // #1247
                            gm_to_l1_nd2nz(view_55, view_56, 64, 128, 128, 128);  // #1248
                        }
                        if (1 < pair_len_12) {  // #1250 #1271
                            const int32_t add_128 = e_base_c + 1;  // #1251
                            const int32_t blk_c_1 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + add_128));  // #1253 #1254 #1255 #1256 #1257 #1258 #1259 #1260 #1261 #1262 #1263
                            const int32_t kv_off_c_1 = blk_c_1 * 64;  // #1264
                            const Tensor<bfloat16_t, Position::L1> slot_36 = l1v.get(l1pv_cnt_carry_1);  // #1265
                            const Tensor<bfloat16_t, Position::L1> view_57 = l1v.get(l1pv_cnt_carry_1)[1024];  // #1266
                            const GMTensor<bfloat16_t> view_58 = v_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off_c_1 * 128))];  // #1269
                            gm_to_l1_nd2nz(view_57, view_58, 64, 128, 128, 128);  // #1270
                        }
                    }
                    pv_local_ready.set();  // #1273
                    ab_local_free.wait();  // #1274
                    pv_local_ready.wait();  // #1275
                    const Tensor<bfloat16_t, Position::L0A> slot_37 = l0a.get(l0ab_cnt_8);  // #1276
                    const Tensor<bfloat16_t, Position::L1> slot_38 = l1p.get(l1pv_cnt_carry_1);  // #1277
                    l1_to_l0<false>(slot_37, slot_38, 128, 128, 128, 128);  // #1278
                    const Tensor<bfloat16_t, Position::L0B> slot_39 = l0b.get(l0ab_cnt_8);  // #1279
                    const Tensor<bfloat16_t, Position::L1> slot_40 = l1v.get(l1pv_cnt_carry_1);  // #1280
                    l1_to_l0<true>(slot_39, slot_40, 128, 128, 128, 128);  // #1281
                    pv_local_free.set();  // #1282
                    ab_local_ready.set();  // #1283
                    ab_local_ready.wait();  // #1284
                    const Tensor<float, Position::L0C> slot_41 = l0c.get(l0c_cnt_8);  // #1285
                    const Tensor<bfloat16_t, Position::L0A> slot_42 = l0a.get(l0ab_cnt_8);  // #1286
                    const Tensor<bfloat16_t, Position::L0B> slot_43 = l0b.get(l0ab_cnt_8);  // #1287
                    mmad(slot_41, slot_42, slot_43, 128, 128, 128, true);  // #1288
                    ab_local_free.set();  // #1289
                    const int32_t l0ab_cnt_9 = (int32_t)(l0ab_cnt_8 + 1);  // #1290 #1291
                    const int32_t l1pv_cnt_1 = (int32_t)(l1pv_cnt_carry_1 + 1);  // #1292 #1293
                    l0ab_cnt_10_state = l0ab_cnt_9;  // #1751
                    l1pv_cnt_2_state = l1pv_cnt_1;  // #1752
                } else {
                    l0ab_cnt_10_state = l0ab_cnt_8;  // #1753
                    l1pv_cnt_2_state = l1pv_cnt_carry_1;  // #1754
                }
                const int32_t l0ab_cnt_10 = (int32_t)(l0ab_cnt_10_state);  // #1755
                const int32_t l1pv_cnt_2 = (int32_t)(l1pv_cnt_2_state);  // #1756
                const int32_t pair_len_13 = Min(2, b_len_c - 2);  // #1297 #1298
                const bool cmp_70 = pair_len_13 > 0;  // #1299
                int32_t l0ab_cnt_12_state = 0;  // #1757
                int32_t l1pv_cnt_4_state = 0;  // #1758
                if (cmp_70) {  // #1392
                    pv_local_free.wait();  // #1300
                    const Tensor<bfloat16_t, Position::L1> slot_44 = l1p.get(l1pv_cnt_2);  // #1301
                    const GMTensor<bfloat16_t> view_59 = p_ws[(((cube_idx * 196608) + (slot_30 * 49152)) + 16384)];  // #1302
                    gm_to_l1_nd2nz(slot_44, view_59, 128, 128, 128, 128);  // #1303
                    if (approx_c) {  // #1367
                        const int32_t first_4 = (approx_g_c + 2) * 64;  // #1304 #1305
                        const int32_t cols_4 = pair_len_13 * 64;  // #1306
                        if (pair_len_13 < 2) {  // #1307 #1311
                            const Tensor<bfloat16_t, Position::L1> slot_45 = l1v.get(l1pv_cnt_2);  // #1308
                            set_constant_to_l1(slot_45, (bfloat16_t)(0.0f), 1024);  // #1309
                            PipeBarrier<PIPE_MTE2>();  // #1310
                        }
                        const Tensor<bfloat16_t, Position::L1> slot_46 = l1v.get(l1pv_cnt_2);  // #1312
                        const Tensor<bfloat16_t, Position::L1> view_60 = l1v.get(l1pv_cnt_2);  // #1313
                        const int32_t sub_57 = (first_4 + cols_4) - first_4;  // #1314 #1315
                        const GMTensor<bfloat16_t> view_61 = vc_padded[(((b * ((H * NPAD) * 128)) + (h * (NPAD * 128))) + (first_4 * 128))];  // #1316
                        gm_to_l1_nd2nz(view_60, view_61, sub_57, 128, 128, 128);  // #1317
                    } else {
                        if (pair_len_13 < 2) {  // #1318 #1322
                            const Tensor<bfloat16_t, Position::L1> slot_47 = l1v.get(l1pv_cnt_2);  // #1319
                            set_constant_to_l1(slot_47, (bfloat16_t)(0.0f), 1024);  // #1320
                            PipeBarrier<PIPE_MTE2>();  // #1321
                        }
                        if (0 < pair_len_13) {  // #1323 #1344
                            const int32_t add_137 = e_base_c + 2;  // #1324
                            const int32_t blk_c_2 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + add_137));  // #1326 #1327 #1328 #1329 #1330 #1331 #1332 #1333 #1334 #1335 #1336
                            const int32_t kv_off_c_2 = blk_c_2 * 64;  // #1337
                            const Tensor<bfloat16_t, Position::L1> slot_48 = l1v.get(l1pv_cnt_2);  // #1338
                            const Tensor<bfloat16_t, Position::L1> view_62 = l1v.get(l1pv_cnt_2);  // #1339
                            const GMTensor<bfloat16_t> view_63 = v_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off_c_2 * 128))];  // #1342
                            gm_to_l1_nd2nz(view_62, view_63, 64, 128, 128, 128);  // #1343
                        }
                        if (1 < pair_len_13) {  // #1345 #1366
                            const int32_t add_142 = e_base_c + 3;  // #1346
                            const int32_t blk_c_3 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + add_142));  // #1348 #1349 #1350 #1351 #1352 #1353 #1354 #1355 #1356 #1357 #1358
                            const int32_t kv_off_c_3 = blk_c_3 * 64;  // #1359
                            const Tensor<bfloat16_t, Position::L1> slot_49 = l1v.get(l1pv_cnt_2);  // #1360
                            const Tensor<bfloat16_t, Position::L1> view_64 = l1v.get(l1pv_cnt_2)[1024];  // #1361
                            const GMTensor<bfloat16_t> view_65 = v_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off_c_3 * 128))];  // #1364
                            gm_to_l1_nd2nz(view_64, view_65, 64, 128, 128, 128);  // #1365
                        }
                    }
                    pv_local_ready.set();  // #1368
                    ab_local_free.wait();  // #1369
                    pv_local_ready.wait();  // #1370
                    const Tensor<bfloat16_t, Position::L0A> slot_50 = l0a.get(l0ab_cnt_10);  // #1371
                    const Tensor<bfloat16_t, Position::L1> slot_51 = l1p.get(l1pv_cnt_2);  // #1372
                    l1_to_l0<false>(slot_50, slot_51, 128, 128, 128, 128);  // #1373
                    const Tensor<bfloat16_t, Position::L0B> slot_52 = l0b.get(l0ab_cnt_10);  // #1374
                    const Tensor<bfloat16_t, Position::L1> slot_53 = l1v.get(l1pv_cnt_2);  // #1375
                    l1_to_l0<true>(slot_52, slot_53, 128, 128, 128, 128);  // #1376
                    pv_local_free.set();  // #1377
                    ab_local_ready.set();  // #1378
                    ab_local_ready.wait();  // #1379
                    PipeBarrier<PIPE_M>();  // #1380
                    const Tensor<float, Position::L0C> slot_54 = l0c.get(l0c_cnt_8);  // #1381
                    const Tensor<bfloat16_t, Position::L0A> slot_55 = l0a.get(l0ab_cnt_10);  // #1382
                    const Tensor<bfloat16_t, Position::L0B> slot_56 = l0b.get(l0ab_cnt_10);  // #1383
                    mmad(slot_54, slot_55, slot_56, 128, 128, 128, false);  // #1384
                    ab_local_free.set();  // #1385
                    const int32_t l0ab_cnt_11 = (int32_t)(l0ab_cnt_10 + 1);  // #1386 #1387
                    const int32_t l1pv_cnt_3 = (int32_t)(l1pv_cnt_2 + 1);  // #1388 #1389
                    l0ab_cnt_12_state = l0ab_cnt_11;  // #1759
                    l1pv_cnt_4_state = l1pv_cnt_3;  // #1760
                } else {
                    l0ab_cnt_12_state = l0ab_cnt_10;  // #1761
                    l1pv_cnt_4_state = l1pv_cnt_2;  // #1762
                }
                const int32_t l0ab_cnt_12 = (int32_t)(l0ab_cnt_12_state);  // #1763
                const int32_t l1pv_cnt_4 = (int32_t)(l1pv_cnt_4_state);  // #1764
                const int32_t pair_len_14 = Min(2, b_len_c - 4);  // #1393 #1394
                const bool cmp_75 = pair_len_14 > 0;  // #1395
                int32_t l0ab_cnt_14_state = 0;  // #1765
                int32_t l1pv_cnt_6_state = 0;  // #1766
                if (cmp_75) {  // #1488
                    pv_local_free.wait();  // #1396
                    const Tensor<bfloat16_t, Position::L1> slot_57 = l1p.get(l1pv_cnt_4);  // #1397
                    const GMTensor<bfloat16_t> view_66 = p_ws[(((cube_idx * 196608) + (slot_30 * 49152)) + 32768)];  // #1398
                    gm_to_l1_nd2nz(slot_57, view_66, 128, 128, 128, 128);  // #1399
                    if (approx_c) {  // #1463
                        const int32_t first_5 = (approx_g_c + 4) * 64;  // #1400 #1401
                        const int32_t cols_5 = pair_len_14 * 64;  // #1402
                        if (pair_len_14 < 2) {  // #1403 #1407
                            const Tensor<bfloat16_t, Position::L1> slot_58 = l1v.get(l1pv_cnt_4);  // #1404
                            set_constant_to_l1(slot_58, (bfloat16_t)(0.0f), 1024);  // #1405
                            PipeBarrier<PIPE_MTE2>();  // #1406
                        }
                        const Tensor<bfloat16_t, Position::L1> slot_59 = l1v.get(l1pv_cnt_4);  // #1408
                        const Tensor<bfloat16_t, Position::L1> view_67 = l1v.get(l1pv_cnt_4);  // #1409
                        const int32_t sub_61 = (first_5 + cols_5) - first_5;  // #1410 #1411
                        const GMTensor<bfloat16_t> view_68 = vc_padded[(((b * ((H * NPAD) * 128)) + (h * (NPAD * 128))) + (first_5 * 128))];  // #1412
                        gm_to_l1_nd2nz(view_67, view_68, sub_61, 128, 128, 128);  // #1413
                    } else {
                        if (pair_len_14 < 2) {  // #1414 #1418
                            const Tensor<bfloat16_t, Position::L1> slot_60 = l1v.get(l1pv_cnt_4);  // #1415
                            set_constant_to_l1(slot_60, (bfloat16_t)(0.0f), 1024);  // #1416
                            PipeBarrier<PIPE_MTE2>();  // #1417
                        }
                        if (0 < pair_len_14) {  // #1419 #1440
                            const int32_t add_151 = e_base_c + 4;  // #1420
                            const int32_t blk_c_4 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + add_151));  // #1422 #1423 #1424 #1425 #1426 #1427 #1428 #1429 #1430 #1431 #1432
                            const int32_t kv_off_c_4 = blk_c_4 * 64;  // #1433
                            const Tensor<bfloat16_t, Position::L1> slot_61 = l1v.get(l1pv_cnt_4);  // #1434
                            const Tensor<bfloat16_t, Position::L1> view_69 = l1v.get(l1pv_cnt_4);  // #1435
                            const GMTensor<bfloat16_t> view_70 = v_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off_c_4 * 128))];  // #1438
                            gm_to_l1_nd2nz(view_69, view_70, 64, 128, 128, 128);  // #1439
                        }
                        if (1 < pair_len_14) {  // #1441 #1462
                            const int32_t add_156 = e_base_c + 5;  // #1442
                            const int32_t blk_c_5 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + add_156));  // #1444 #1445 #1446 #1447 #1448 #1449 #1450 #1451 #1452 #1453 #1454
                            const int32_t kv_off_c_5 = blk_c_5 * 64;  // #1455
                            const Tensor<bfloat16_t, Position::L1> slot_62 = l1v.get(l1pv_cnt_4);  // #1456
                            const Tensor<bfloat16_t, Position::L1> view_71 = l1v.get(l1pv_cnt_4)[1024];  // #1457
                            const GMTensor<bfloat16_t> view_72 = v_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (kv_off_c_5 * 128))];  // #1460
                            gm_to_l1_nd2nz(view_71, view_72, 64, 128, 128, 128);  // #1461
                        }
                    }
                    pv_local_ready.set();  // #1464
                    ab_local_free.wait();  // #1465
                    pv_local_ready.wait();  // #1466
                    const Tensor<bfloat16_t, Position::L0A> slot_63 = l0a.get(l0ab_cnt_12);  // #1467
                    const Tensor<bfloat16_t, Position::L1> slot_64 = l1p.get(l1pv_cnt_4);  // #1468
                    l1_to_l0<false>(slot_63, slot_64, 128, 128, 128, 128);  // #1469
                    const Tensor<bfloat16_t, Position::L0B> slot_65 = l0b.get(l0ab_cnt_12);  // #1470
                    const Tensor<bfloat16_t, Position::L1> slot_66 = l1v.get(l1pv_cnt_4);  // #1471
                    l1_to_l0<true>(slot_65, slot_66, 128, 128, 128, 128);  // #1472
                    pv_local_free.set();  // #1473
                    ab_local_ready.set();  // #1474
                    ab_local_ready.wait();  // #1475
                    PipeBarrier<PIPE_M>();  // #1476
                    const Tensor<float, Position::L0C> slot_67 = l0c.get(l0c_cnt_8);  // #1477
                    const Tensor<bfloat16_t, Position::L0A> slot_68 = l0a.get(l0ab_cnt_12);  // #1478
                    const Tensor<bfloat16_t, Position::L0B> slot_69 = l0b.get(l0ab_cnt_12);  // #1479
                    mmad(slot_67, slot_68, slot_69, 128, 128, 128, false);  // #1480
                    ab_local_free.set();  // #1481
                    const int32_t l0ab_cnt_13 = (int32_t)(l0ab_cnt_12 + 1);  // #1482 #1483
                    const int32_t l1pv_cnt_5 = (int32_t)(l1pv_cnt_4 + 1);  // #1484 #1485
                    l0ab_cnt_14_state = l0ab_cnt_13;  // #1767
                    l1pv_cnt_6_state = l1pv_cnt_5;  // #1768
                } else {
                    l0ab_cnt_14_state = l0ab_cnt_12;  // #1769
                    l1pv_cnt_6_state = l1pv_cnt_4;  // #1770
                }
                const int32_t l0ab_cnt_14 = (int32_t)(l0ab_cnt_14_state);  // #1771
                const int32_t l1pv_cnt_6 = (int32_t)(l1pv_cnt_6_state);  // #1772
                c_local_ready.set();  // #1489
                c_local_ready.wait();  // #1490
                const GMTensor<float> view_73 = pv_ws[((cube_idx * 65536) + (slot_30 * 16384))];  // #1491
                const Tensor<float, Position::L0C> slot_70 = l0c.get(l0c_cnt_8);  // #1492
                l0c_to_gm_nz2nd(view_73, slot_70, 128, 128, 128, 128, false, 0.0f, 0, false, false);  // #1493
                c_local_free.set();  // #1494
                CUBE_READY<PIPE_FIX>(2);  // #1495
                const int32_t accum_step_1 = (int32_t)(accum_step_carry + 1);  // #1513 #1514
                const int32_t l0c_cnt_9 = (int32_t)(l0c_cnt_8 + 1);  // #1515 #1516
                accum_step_2_state = accum_step_1;  // #1773
                l0ab_cnt_15_state = l0ab_cnt_14;  // #1774
                l0c_cnt_10_state = l0c_cnt_9;  // #1775
                l1pv_cnt_7_state = l1pv_cnt_6;  // #1776
            } else {
                accum_step_2_state = accum_step_carry;  // #1777
                l0ab_cnt_15_state = l0ab_cnt_8;  // #1778
                l0c_cnt_10_state = l0c_cnt_8;  // #1779
                l1pv_cnt_7_state = l1pv_cnt_carry_1;  // #1780
            }
            const int32_t accum_step_2 = (int32_t)(accum_step_2_state);  // #1781
            const int32_t l0ab_cnt_15 = (int32_t)(l0ab_cnt_15_state);  // #1782
            const int32_t l0c_cnt_10 = (int32_t)(l0c_cnt_10_state);  // #1783
            const int32_t l1pv_cnt_7 = (int32_t)(l1pv_cnt_7_state);  // #1784
            accum_step_3_state = accum_step_2;  // #1785
            l0ab_cnt_16_state = l0ab_cnt_15;  // #1786
            l0c_cnt_11_state = l0c_cnt_10;  // #1787
            l1k_cnt_9_state = l1k_cnt_8;  // #1788
            l1pv_cnt_8_state = l1pv_cnt_7;  // #1789
            ready_lengths_2_state = ready_lengths_1;  // #1790
        }
        const int32_t l0ab_cnt_16 = (int32_t)(l0ab_cnt_16_state);  // #1793
        const int32_t l0c_cnt_11 = (int32_t)(l0c_cnt_11_state);  // #1794
        const int32_t l1k_cnt_9 = (int32_t)(l1k_cnt_9_state);  // #1795
        const int32_t l1pv_cnt_8 = (int32_t)(l1pv_cnt_8_state);  // #1796
        PipeBarrier<PIPE_ALL>();  // #1526
        q_input_free.set();  // #1527
        l0ab_cnt_17_state = l0ab_cnt_16;  // #1797
        l0c_cnt_12_state = l0c_cnt_11;  // #1798
        l1k_cnt_10_state = l1k_cnt_9;  // #1799
        l1pv_cnt_9_state = l1pv_cnt_8;  // #1800
    }
    WAIT_VEC<PIPE_S>(0);  // #1556
    WAIT_VEC<PIPE_S>(0);  // #1556
    WAIT_VEC<PIPE_S>(2);  // #1556
    WAIT_VEC<PIPE_S>(2);  // #1556
    return;  // #1556
}
