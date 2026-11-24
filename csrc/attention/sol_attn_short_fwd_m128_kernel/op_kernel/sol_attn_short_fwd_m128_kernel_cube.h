#pragma once
using namespace ascrip;

__aicore__ inline void sol_attn_short_fwd_m128_kernel_cube(GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR out_, GM_ADDR lse_, int32_t B, int32_t T, int32_t H, int32_t NT, int32_t NT_Q, int32_t NT_Q_PAD8, int32_t T_LSE, int32_t sink_start, int32_t sink_end, float scale_log2, float scale_sq, float tau, GM_ADDR workspace)
{
    GMTensor<bfloat16_t> q((__gm__ bfloat16_t*)q_);
    GMTensor<bfloat16_t> k((__gm__ bfloat16_t*)k_);
    GMTensor<bfloat16_t> v((__gm__ bfloat16_t*)v_);
    GMTensor<bfloat16_t> out((__gm__ bfloat16_t*)out_);
    GMTensor<float> lse((__gm__ float*)lse_);
    SetAtomicNone();
    const int32_t mul = B * H;  // #2
    const int32_t mul_2 = (mul * 64) * 128;  // #3 #4
    const int32_t mul_178 = mul_2 * 2;  // #2763
    const GMTensor<bfloat16_t> kc((__gm__ bfloat16_t*)(workspace + (0)));  // #5
    const GMTensor<bfloat16_t> vc((__gm__ bfloat16_t*)(workspace + (mul_178)));  // #9
    const int32_t add_189 = (mul_178 + mul_178) + 256;  // #2765 #2766
    const int32_t mul_7 = mul * NT_Q_PAD8;  // #12
    const int32_t add_190 = add_189 + (mul_7 * 8);  // #2767 #2768
    const GMTensor<uint64_t> approx_mask((__gm__ uint64_t*)(workspace + (add_189)));  // #13
    const int32_t add_191 = add_190 + (mul_7 * 4);  // #2769 #2770
    const GMTensor<int32_t> exact_counts((__gm__ int32_t*)(workspace + (add_190)));  // #16
    const int32_t mul_12 = (mul * NT_Q) * 64;  // #18 #19
    const int32_t add_192 = add_191 + (mul_12 * 2);  // #2771 #2772
    const GMTensor<uint16_t> exact_indices((__gm__ uint16_t*)(workspace + (add_191)));  // #20
    const bool dense = NT <= 2;  // #21
    if (!(dense)) {  // #22 #498
        WAIT_VEC<PIPE_S>(7);  // #497
    }
    const int32_t add_193 = add_192 + 18874368;  // #2773
    const GMTensor<float> score_ws((__gm__ float*)(workspace + (add_192)));  // #500
    const GMTensor<bfloat16_t> p_ws((__gm__ bfloat16_t*)(workspace + (add_193)));  // #501
    const GMTensor<float> pv_ws((__gm__ float*)(workspace + (add_193 + 9437184)));  // #2774 #502
    const GMTensor<float> score_pairs = score_ws;  // #503
    const GMTensor<bfloat16_t> p_pairs = p_ws;  // #504
    const Tensor<bfloat16_t, Position::L1> l1q((uint64_t)(0));  // #505
    const Buff<bfloat16_t, Position::L1, 2> l1k((uint64_t)(32768), (uint64_t)(32768));  // #506
    const Buff<bfloat16_t, Position::L1, 4> l1p((uint64_t)(98304), (uint64_t)(32768));  // #507
    const Buff<bfloat16_t, Position::L1, 4> l1v((uint64_t)(229376), (uint64_t)(32768));  // #508
    const Buff<float, Position::L0C, 2> l0c((uint64_t)(0), (uint64_t)(65536));  // #509
    const Buff<bfloat16_t, Position::L0A, 2> l0a((uint64_t)(0), (uint64_t)(32768));  // #510
    const Buff<bfloat16_t, Position::L0B, 2> l0b((uint64_t)(0), (uint64_t)(32768));  // #511
    // sync.mutex cv id=0: the producer side; its 2 tokens are drained before the return  // #564
    CUBE_READY<PIPE_MTE2>(1);  // #565
    CUBE_READY<PIPE_MTE2>(1);  // #565
    // sync.mutex cv id=2: the producer side; its 2 tokens are drained before the return  // #566
    SEvent<PIPE_MTE1, PIPE_MTE2, 1, 4> q_input_free;  // #567
    SEvent<PIPE_MTE2, PIPE_MTE1, 0, 4> q_input_ready;  // #568
    DEvent<PIPE_MTE2, PIPE_MTE1, 0, 0, 1> k_local_ready;  // #570
    DEvent<PIPE_MTE1, PIPE_MTE2, 2, 0, 1> k_local_free;  // #571
    DEvent<PIPE_MTE2, PIPE_MTE1, 0, 2, 3> pv_local_ready;  // #572
    DEvent<PIPE_MTE1, PIPE_MTE2, 2, 2, 3> pv_local_free;  // #573
    DEvent<PIPE_MTE1, PIPE_M, 0, 0, 1> ab_local_ready;  // #574
    DEvent<PIPE_M, PIPE_MTE1, 2, 0, 1> ab_local_free;  // #575
    DEvent<PIPE_M, PIPE_FIX, 0, 0, 1> c_local_ready;  // #576
    DEvent<PIPE_FIX, PIPE_M, 2, 0, 1> c_local_free;  // #577
    const int32_t cube_idx = GetCubeIdx();  // #583
    const int32_t cube_count = GetCubeNum();  // #584
    const int32_t core_step = (int32_t)(cube_count);  // #585
    const int32_t bh_total = B * H;  // #588
    const int32_t total_tasks = bh_total * NT_Q;  // #589
    int32_t l0ab_cnt_17_state = 0;  // #2268
    int32_t l0c_cnt_12_state = 0;  // #2269
    int32_t l1k_cnt_10_state = 0;  // #2270
    int32_t l1pv_cnt_9_state = 0;  // #2271
    for (int32_t linear_id = cube_idx; linear_id < total_tasks; linear_id += core_step) {  // #2677
        const int32_t l0ab_cnt_carry = (int32_t)(l0ab_cnt_17_state);  // #2272
        const int32_t l0c_cnt_carry = (int32_t)(l0c_cnt_12_state);  // #2273
        const int32_t l1k_cnt_carry = (int32_t)(l1k_cnt_10_state);  // #2274
        const int32_t l1pv_cnt_carry = (int32_t)(l1pv_cnt_9_state);  // #2275
        const int32_t idiv_1 = linear_id / NT_Q;  // #4403
        const int32_t idiv_2 = linear_id % NT_Q;  // #4404
        const bool idiv_3 = idiv_2 != 0;  // #4405
        const bool idiv_4 = linear_id < 0;  // #4406
        const bool idiv_5 = NT_Q < 0;  // #4407
        const bool idiv_6 = idiv_4 != idiv_5;  // #4408
        const bool idiv_7 = idiv_3 && idiv_6;  // #4409
        const int32_t idiv_8 = (idiv_7) ? (1) : (0);  // #4410
        const int32_t bh_1 = idiv_1 - idiv_8;  // #594
        const int32_t q_idx = ascrip::FloorMod<int32_t>(linear_id, NT_Q);  // #595
        const int32_t idiv_9 = bh_1 / H;  // #4411
        const int32_t idiv_10 = bh_1 % H;  // #4412
        const bool idiv_11 = idiv_10 != 0;  // #4413
        const bool idiv_12 = bh_1 < 0;  // #4414
        const bool idiv_14 = idiv_12 != false;  // #4416
        const bool idiv_15 = idiv_11 && idiv_14;  // #4417
        const int32_t idiv_16 = (idiv_15) ? (1) : (0);  // #4418
        const int32_t b_1 = idiv_9 - idiv_16;  // #596
        const int32_t h_1 = ascrip::FloorMod<int32_t>(bh_1, H);  // #597
        const int32_t q_start = q_idx * 128;  // #598
        const int32_t valid_m = Min(128, T - q_start);  // #599 #600
        q_input_free.wait();  // #606
        if (valid_m < 128) {  // #607 #610
            set_constant_to_l1(l1q, (bfloat16_t)(0.0f), 1024);  // #608
            PipeBarrier<PIPE_MTE2>();  // #609
        }
        const GMTensor<bfloat16_t> view_185 = q[(((b_1 * ((T * H) * 128)) + (q_start * (H * 128))) + (h_1 * 128))];  // #613
        gm_to_l1_nd2nz(l1q, view_185, valid_m, 128, 128, H * 128);  // #2684 #614
        q_input_ready.set();  // #615
        q_input_ready.wait();  // #616
        const bool not_5 = !(dense);  // #626
        int32_t approx_beats_1_state = 0;  // #2276
        int32_t exact_total_1_state = 0;  // #2277
        if (not_5) {  // #636
            const int32_t exact_total = exact_counts.load(((b_1 * (H * NT_Q_PAD8)) + (h_1 * NT_Q_PAD8)) + q_idx);  // #628 #629 #630 #631 #632 #633
            approx_beats_1_state = 1;  // #2278
            exact_total_1_state = exact_total;  // #2279
        } else {
            approx_beats_1_state = 0;  // #2280
            exact_total_1_state = NT;  // #2281
        }
        const int32_t approx_beats_1 = (int32_t)(approx_beats_1_state);  // #2282
        const int32_t exact_total_1 = (int32_t)(exact_total_1_state);  // #2283
        const int32_t idiv_17 = exact_total_1 / 6;  // #4419
        const int32_t idiv_18 = exact_total_1 % 6;  // #4420
        const bool idiv_19 = idiv_18 != 0;  // #4421
        const bool idiv_20 = exact_total_1 < 0;  // #4422
        const bool idiv_22 = idiv_20 == false;  // #4424
        const bool idiv_23 = idiv_19 && idiv_22;  // #4425
        const int32_t idiv_24 = (idiv_23) ? (1) : (0);  // #4426
        const int32_t ceil_div_3 = idiv_17 + idiv_24;  // #637
        const int32_t total_beats = approx_beats_1 + ceil_div_3;  // #638
        const int32_t add_13 = total_beats + 3;  // #644
        int32_t accum_step_3_state = 0;  // #2284
        int32_t l0ab_cnt_16_state = l0ab_cnt_carry;  // #2285
        int32_t l0c_cnt_11_state = l0c_cnt_carry;  // #2286
        int32_t l1k_cnt_9_state = l1k_cnt_carry;  // #2287
        int32_t l1pv_cnt_8_state = l1pv_cnt_carry;  // #2288
        int32_t ready_lengths_2_state = 0;  // #2289
        for (int32_t beat = 0; beat < add_13; beat += 1) {  // #2667
            const int32_t accum_step_carry = (int32_t)(accum_step_3_state);  // #2290
            const int32_t l0ab_cnt_carry_1 = (int32_t)(l0ab_cnt_16_state);  // #2291
            const int32_t l0c_cnt_carry_1 = (int32_t)(l0c_cnt_11_state);  // #2292
            const int32_t l1k_cnt_carry_1 = (int32_t)(l1k_cnt_9_state);  // #2293
            const int32_t l1pv_cnt_carry_1 = (int32_t)(l1pv_cnt_8_state);  // #2294
            const int32_t ready_lengths_carry = (int32_t)(ready_lengths_2_state);  // #2295
            const bool cmp_9 = beat < total_beats;  // #646
            int32_t b_len_p_5_state = 0;  // #2296
            int32_t l0ab_cnt_8_state = 0;  // #2297
            int32_t l0c_cnt_8_state = 0;  // #2298
            int32_t l1k_cnt_8_state = 0;  // #2299
            if (cmp_9) {  // #1749
                const bool approx_p = beat < approx_beats_1;  // #647
                const int32_t approx_g_p = beat * 2;  // #648
                const int32_t e_base_p = Max(0, beat - approx_beats_1) * 6;  // #657 #658 #659
                const int32_t b_len_p_1 = Min(6, exact_total_1 - e_base_p);  // #660 #661
                int32_t b_len_p_4_state = 0;  // #2300
                if (approx_p) {  // #706
                    const bool cmp_10 = approx_g_p < 1;  // #664
                    uint64_t mask_0_1_state = 0ULL;  // #2303
                    if (cmp_10) {  // #675
                        mask_0_1_state = approx_mask.load((((b_1 * (H * NT_Q_PAD8)) + (h_1 * NT_Q_PAD8)) + q_idx) + approx_g_p);  // #666 #667 #668 #669 #670 #671 #672 #2304
                    } else {
                        mask_0_1_state = 0ULL;  // #2305
                    }
                    const uint64_t mask_0_1 = (uint64_t)(mask_0_1_state);  // #2306
                    const int32_t add_18 = approx_g_p + 1;  // #676
                    const bool cmp_11 = add_18 < 1;  // #678
                    uint64_t mask_1_1_state = 0ULL;  // #2307
                    if (cmp_11) {  // #689
                        mask_1_1_state = approx_mask.load((((b_1 * (H * NT_Q_PAD8)) + (h_1 * NT_Q_PAD8)) + q_idx) + add_18);  // #680 #681 #682 #683 #684 #685 #686 #2308
                    } else {
                        mask_1_1_state = 0ULL;  // #2309
                    }
                    const bool cmp_12 = (((((((mask_0_1 | 0ULL) | 0ULL) | 0ULL) | 0ULL) | 0ULL) | 0ULL) | 0ULL) != 0ULL;  // #690 #691 #692 #693 #694 #695 #696 #698
                    int32_t b_len_p_3_state = 0;  // #2311
                    if (cmp_12) {  // #703
                        b_len_p_3_state = 1 - approx_g_p;  // #699 #2312
                    } else {
                        b_len_p_3_state = 0;  // #2313
                    }
                    b_len_p_4_state = (int32_t)(b_len_p_3_state);  // #2314 #2315
                } else {
                    b_len_p_4_state = b_len_p_1;  // #2318
                }
                const int32_t b_len_p_4 = (int32_t)(b_len_p_4_state);  // #2321
                const bool cmp_13 = b_len_p_4 > 0;  // #707
                int32_t l0ab_cnt_7_state = 0;  // #2324
                int32_t l0c_cnt_7_state = 0;  // #2325
                int32_t l1k_cnt_7_state = 0;  // #2326
                if (cmp_13) {  // #1746
                    const int32_t slot_b_p = beat & 3;  // #708
                    const int32_t pair_len = Min(2, b_len_p_4);  // #710
                    const bool cmp_14 = pair_len > 0;  // #711
                    int32_t l0ab_cnt_2_state = 0;  // #2327
                    int32_t l0c_cnt_2_state = 0;  // #2328
                    int32_t l1k_cnt_2_state = 0;  // #2329
                    if (cmp_14) {  // #844
                        k_local_free.wait();  // #712
                        if (approx_p) {  // #813
                            const int32_t first = approx_g_p * 64;  // #714
                            const int32_t cols = pair_len * 64;  // #715
                            if (pair_len < 2) {  // #716 #720
                                const Tensor<bfloat16_t, Position::L1> slot = l1k.get(l1k_cnt_carry_1);  // #717
                                set_constant_to_l1(slot, (bfloat16_t)(0.0f), 1024);  // #718
                                PipeBarrier<PIPE_MTE2>();  // #719
                            }
                            const Tensor<bfloat16_t, Position::L1> slot_1 = l1k.get(l1k_cnt_carry_1);  // #721
                            const Tensor<bfloat16_t, Position::L1> view_186 = l1k.get(l1k_cnt_carry_1);  // #722
                            const int32_t sub_30 = (first + cols) - first;  // #723 #724
                            const GMTensor<bfloat16_t> view_187 = kc[(((b_1 * ((H * 64) * 128)) + (h_1 * 8192)) + (first * 128))];  // #725
                            gm_to_l1_nd2nz(view_186, view_187, sub_30, 128, 128, 128);  // #726
                        } else {
                            const Tensor<bfloat16_t, Position::L1> slot_2 = l1k.get(l1k_cnt_carry_1);  // #727
                            const int32_t blk1 = e_base_p + 1;  // #729
                            const bool not_6 = !(dense);  // #730
                            int32_t blk0_1_state = 0;  // #2330
                            int32_t blk1_3_state = 0;  // #2331
                            if (not_6) {  // #762
                                const int32_t blk0 = (int32_t)(exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + e_base_p));  // #732 #733 #734 #735 #736 #737 #738 #739 #740 #741 #742
                                const bool cmp_16 = pair_len == 2;  // #743
                                int32_t blk1_2_state = 0;  // #2332
                                if (cmp_16) {  // #759
                                    const int32_t add_28 = e_base_p + 1;  // #744
                                    blk1_2_state = (int32_t)(exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + add_28));  // #746 #747 #748 #749 #750 #751 #752 #753 #754 #755 #756 #2333
                                } else {
                                    blk1_2_state = blk1;  // #2334
                                }
                                const int32_t blk1_2 = (int32_t)(blk1_2_state);  // #2335
                                blk0_1_state = blk0;  // #2336
                                blk1_3_state = blk1_2;  // #2337
                            } else {
                                blk0_1_state = e_base_p;  // #2338
                                blk1_3_state = blk1;  // #2339
                            }
                            const int32_t blk0_1 = (int32_t)(blk0_1_state);  // #2340
                            const int32_t blk1_3 = (int32_t)(blk1_3_state);  // #2341
                            const int32_t start0 = blk0_1 * 64;  // #763
                            const int32_t rows0 = Min(64, T - start0);  // #764 #765
                            const bool cmp_17 = pair_len == 2;  // #766
                            int32_t rows1_1_state = 0;  // #2342
                            if (cmp_17) {  // #772
                                rows1_1_state = Min(64, T - (blk1_3 * 64));  // #767 #768 #769 #2343
                            } else {
                                rows1_1_state = 64;  // #2344
                            }
                            const int32_t rows1_1 = (int32_t)(rows1_1_state);  // #2345
                            const bool cmp_18 = pair_len < 2;  // #773
                            bool choice_1_state = false;  // #2346
                            if (cmp_18) {  // #781
                                choice_1_state = true;  // #2347
                            } else {
                                const bool cmp_19 = rows0 < 64;  // #774
                                bool choice_state = false;  // #2348
                                if (cmp_19) {  // #778
                                    choice_state = true;  // #2349
                                } else {
                                    choice_state = rows1_1 < 64;  // #775 #2350
                                }
                                choice_1_state = (bool)(choice_state);  // #2351 #2352
                            }
                            const bool choice_1 = (bool)(choice_1_state);  // #2353
                            if (choice_1) {  // #784
                                set_constant_to_l1(slot_2, (bfloat16_t)(0.0f), 1024);  // #782
                                PipeBarrier<PIPE_MTE2>();  // #783
                            }
                            bool choice_2_state = false;  // #2354
                            if (cmp_17) {  // #790
                                choice_2_state = blk1_3 == (blk0_1 + 1);  // #786 #787 #2355
                            } else {
                                choice_2_state = false;  // #2356
                            }
                            const bool choice_2 = (bool)(choice_2_state);  // #2357
                            if (choice_2) {  // #812
                                const int32_t rows_1 = rows0 + rows1_1;  // #791
                                const Tensor<bfloat16_t, Position::L1> view_188 = l1k.get(l1k_cnt_carry_1);  // #792
                                const int32_t sub_33 = (start0 + rows_1) - start0;  // #793 #794
                                const GMTensor<bfloat16_t> view_189 = k[(((b_1 * ((T * H) * 128)) + (start0 * (H * 128))) + (h_1 * 128))];  // #795
                                gm_to_l1_nd2nz(view_188, view_189, sub_33, 128, 128, H * 128);  // #2685 #796
                            } else {
                                const Tensor<bfloat16_t, Position::L1> view_190 = l1k.get(l1k_cnt_carry_1);  // #797
                                const GMTensor<bfloat16_t> view_191 = k[(((b_1 * ((T * H) * 128)) + (start0 * (H * 128))) + (h_1 * 128))];  // #800
                                const int32_t mul_157 = H * 128;  // #2686
                                gm_to_l1_nd2nz(view_190, view_191, rows0, 128, 128, mul_157);  // #801
                                if (cmp_17) {  // #811
                                    const int32_t start1 = blk1_3 * 64;  // #803
                                    const Tensor<bfloat16_t, Position::L1> view_192 = l1k.get(l1k_cnt_carry_1)[1024];  // #806
                                    const GMTensor<bfloat16_t> view_193 = k[(((b_1 * ((T * H) * 128)) + (start1 * (H * 128))) + (h_1 * 128))];  // #809
                                    gm_to_l1_nd2nz(view_192, view_193, rows1_1, 128, 128, mul_157);  // #810
                                }
                            }
                        }
                        k_local_ready.set();  // #814
                        ab_local_free.wait();  // #815
                        k_local_ready.wait();  // #816
                        const Tensor<bfloat16_t, Position::L0A> q_tile_l0 = l0a.get(l0ab_cnt_carry_1);  // #817
                        l1_to_l0<false>(q_tile_l0, l1q, 128, 128, 128, 128);  // #818
                        const Tensor<bfloat16_t, Position::L0B> slot_3 = l0b.get(l0ab_cnt_carry_1);  // #819
                        const Tensor<bfloat16_t, Position::L1> slot_4 = l1k.get(l1k_cnt_carry_1);  // #820
                        l1_to_l0<false>(slot_3, slot_4, 128, 128, 128, 128);  // #821
                        k_local_free.set();  // #822
                        ab_local_ready.set();  // #823
                        c_local_free.wait();  // #824
                        ab_local_ready.wait();  // #825
                        const Tensor<float, Position::L0C> slot_5 = l0c.get(l0c_cnt_carry_1);  // #826
                        const Tensor<bfloat16_t, Position::L0B> slot_6 = l0b.get(l0ab_cnt_carry_1);  // #827
                        mmad(slot_5, q_tile_l0, slot_6, 128, 128, 128, true);  // #828
                        ab_local_free.set();  // #829
                        const int32_t l0ab_cnt_1 = (int32_t)(l0ab_cnt_carry_1 + 1);  // #830 #831
                        c_local_ready.set();  // #832
                        c_local_ready.wait();  // #833
                        const GMTensor<float> view_194 = score_ws[((cube_idx * 196608) + (slot_b_p * 49152))];  // #834
                        const Tensor<float, Position::L0C> slot_7 = l0c.get(l0c_cnt_carry_1);  // #835
                        l0c_to_gm_nz2nd(view_194, slot_7, 128, 128, 128, 128, false, 0.0f, 0, false, false);  // #836
                        c_local_free.set();  // #837
                        const int32_t l1k_cnt_1 = (int32_t)(l1k_cnt_carry_1 + 1);  // #838 #839
                        const int32_t l0c_cnt_1 = (int32_t)(l0c_cnt_carry_1 + 1);  // #840 #841
                        l0ab_cnt_2_state = l0ab_cnt_1;  // #2358
                        l0c_cnt_2_state = l0c_cnt_1;  // #2359
                        l1k_cnt_2_state = l1k_cnt_1;  // #2360
                    } else {
                        l0ab_cnt_2_state = l0ab_cnt_carry_1;  // #2361
                        l0c_cnt_2_state = l0c_cnt_carry_1;  // #2362
                        l1k_cnt_2_state = l1k_cnt_carry_1;  // #2363
                    }
                    const int32_t l0ab_cnt_2 = (int32_t)(l0ab_cnt_2_state);  // #2364
                    const int32_t l0c_cnt_2 = (int32_t)(l0c_cnt_2_state);  // #2365
                    const int32_t l1k_cnt_2 = (int32_t)(l1k_cnt_2_state);  // #2366
                    const int32_t pair_len_1 = Min(2, b_len_p_4 - 2);  // #845 #846
                    const bool cmp_24 = pair_len_1 > 0;  // #847
                    int32_t l0ab_cnt_4_state = 0;  // #2367
                    int32_t l0c_cnt_4_state = 0;  // #2368
                    int32_t l1k_cnt_4_state = 0;  // #2369
                    if (cmp_24) {  // #980
                        k_local_free.wait();  // #848
                        if (approx_p) {  // #949
                            const int32_t first_1 = (approx_g_p + 2) * 64;  // #849 #850
                            const int32_t cols_1 = pair_len_1 * 64;  // #851
                            if (pair_len_1 < 2) {  // #852 #856
                                const Tensor<bfloat16_t, Position::L1> slot_8 = l1k.get(l1k_cnt_2);  // #853
                                set_constant_to_l1(slot_8, (bfloat16_t)(0.0f), 1024);  // #854
                                PipeBarrier<PIPE_MTE2>();  // #855
                            }
                            const Tensor<bfloat16_t, Position::L1> slot_9 = l1k.get(l1k_cnt_2);  // #857
                            const Tensor<bfloat16_t, Position::L1> view_195 = l1k.get(l1k_cnt_2);  // #858
                            const int32_t sub_38 = (first_1 + cols_1) - first_1;  // #859 #860
                            const GMTensor<bfloat16_t> view_196 = kc[(((b_1 * ((H * 64) * 128)) + (h_1 * 8192)) + (first_1 * 128))];  // #861
                            gm_to_l1_nd2nz(view_195, view_196, sub_38, 128, 128, 128);  // #862
                        } else {
                            const Tensor<bfloat16_t, Position::L1> slot_10 = l1k.get(l1k_cnt_2);  // #863
                            const int32_t add_42 = e_base_p + 2;  // #864
                            const int32_t blk1_4 = add_42 + 1;  // #865
                            const bool not_7 = !(dense);  // #866
                            int32_t blk0_3_state = 0;  // #2370
                            int32_t blk1_7_state = 0;  // #2371
                            if (not_7) {  // #898
                                const int32_t blk0_2 = (int32_t)(exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + add_42));  // #868 #869 #870 #871 #872 #873 #874 #875 #876 #877 #878
                                const bool cmp_26 = pair_len_1 == 2;  // #879
                                int32_t blk1_6_state = 0;  // #2372
                                if (cmp_26) {  // #895
                                    const int32_t add_46 = add_42 + 1;  // #880
                                    blk1_6_state = (int32_t)(exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + add_46));  // #882 #883 #884 #885 #886 #887 #888 #889 #890 #891 #892 #2373
                                } else {
                                    blk1_6_state = blk1_4;  // #2374
                                }
                                const int32_t blk1_6 = (int32_t)(blk1_6_state);  // #2375
                                blk0_3_state = blk0_2;  // #2376
                                blk1_7_state = blk1_6;  // #2377
                            } else {
                                blk0_3_state = add_42;  // #2378
                                blk1_7_state = blk1_4;  // #2379
                            }
                            const int32_t blk0_3 = (int32_t)(blk0_3_state);  // #2380
                            const int32_t blk1_7 = (int32_t)(blk1_7_state);  // #2381
                            const int32_t start0_1 = blk0_3 * 64;  // #899
                            const int32_t rows0_1 = Min(64, T - start0_1);  // #900 #901
                            const bool cmp_27 = pair_len_1 == 2;  // #902
                            int32_t rows1_3_state = 0;  // #2382
                            if (cmp_27) {  // #908
                                rows1_3_state = Min(64, T - (blk1_7 * 64));  // #903 #904 #905 #2383
                            } else {
                                rows1_3_state = 64;  // #2384
                            }
                            const int32_t rows1_3 = (int32_t)(rows1_3_state);  // #2385
                            const bool cmp_28 = pair_len_1 < 2;  // #909
                            bool choice_4_state = false;  // #2386
                            if (cmp_28) {  // #917
                                choice_4_state = true;  // #2387
                            } else {
                                const bool cmp_29 = rows0_1 < 64;  // #910
                                bool choice_3_state = false;  // #2388
                                if (cmp_29) {  // #914
                                    choice_3_state = true;  // #2389
                                } else {
                                    choice_3_state = rows1_3 < 64;  // #911 #2390
                                }
                                choice_4_state = (bool)(choice_3_state);  // #2391 #2392
                            }
                            const bool choice_4 = (bool)(choice_4_state);  // #2393
                            if (choice_4) {  // #920
                                set_constant_to_l1(slot_10, (bfloat16_t)(0.0f), 1024);  // #918
                                PipeBarrier<PIPE_MTE2>();  // #919
                            }
                            bool choice_5_state = false;  // #2394
                            if (cmp_27) {  // #926
                                choice_5_state = blk1_7 == (blk0_3 + 1);  // #922 #923 #2395
                            } else {
                                choice_5_state = false;  // #2396
                            }
                            const bool choice_5 = (bool)(choice_5_state);  // #2397
                            if (choice_5) {  // #948
                                const int32_t rows_2 = rows0_1 + rows1_3;  // #927
                                const Tensor<bfloat16_t, Position::L1> view_197 = l1k.get(l1k_cnt_2);  // #928
                                const int32_t sub_41 = (start0_1 + rows_2) - start0_1;  // #929 #930
                                const GMTensor<bfloat16_t> view_198 = k[(((b_1 * ((T * H) * 128)) + (start0_1 * (H * 128))) + (h_1 * 128))];  // #931
                                gm_to_l1_nd2nz(view_197, view_198, sub_41, 128, 128, H * 128);  // #2688 #932
                            } else {
                                const Tensor<bfloat16_t, Position::L1> view_199 = l1k.get(l1k_cnt_2);  // #933
                                const GMTensor<bfloat16_t> view_200 = k[(((b_1 * ((T * H) * 128)) + (start0_1 * (H * 128))) + (h_1 * 128))];  // #936
                                const int32_t mul_160 = H * 128;  // #2689
                                gm_to_l1_nd2nz(view_199, view_200, rows0_1, 128, 128, mul_160);  // #937
                                if (cmp_27) {  // #947
                                    const int32_t start1_1 = blk1_7 * 64;  // #939
                                    const Tensor<bfloat16_t, Position::L1> view_201 = l1k.get(l1k_cnt_2)[1024];  // #942
                                    const GMTensor<bfloat16_t> view_202 = k[(((b_1 * ((T * H) * 128)) + (start1_1 * (H * 128))) + (h_1 * 128))];  // #945
                                    gm_to_l1_nd2nz(view_201, view_202, rows1_3, 128, 128, mul_160);  // #946
                                }
                            }
                        }
                        k_local_ready.set();  // #950
                        ab_local_free.wait();  // #951
                        k_local_ready.wait();  // #952
                        const Tensor<bfloat16_t, Position::L0A> q_tile_l0_1 = l0a.get(l0ab_cnt_2);  // #953
                        l1_to_l0<false>(q_tile_l0_1, l1q, 128, 128, 128, 128);  // #954
                        const Tensor<bfloat16_t, Position::L0B> slot_11 = l0b.get(l0ab_cnt_2);  // #955
                        const Tensor<bfloat16_t, Position::L1> slot_12 = l1k.get(l1k_cnt_2);  // #956
                        l1_to_l0<false>(slot_11, slot_12, 128, 128, 128, 128);  // #957
                        k_local_free.set();  // #958
                        ab_local_ready.set();  // #959
                        c_local_free.wait();  // #960
                        ab_local_ready.wait();  // #961
                        const Tensor<float, Position::L0C> slot_13 = l0c.get(l0c_cnt_2);  // #962
                        const Tensor<bfloat16_t, Position::L0B> slot_14 = l0b.get(l0ab_cnt_2);  // #963
                        mmad(slot_13, q_tile_l0_1, slot_14, 128, 128, 128, true);  // #964
                        ab_local_free.set();  // #965
                        const int32_t l0ab_cnt_3 = (int32_t)(l0ab_cnt_2 + 1);  // #966 #967
                        c_local_ready.set();  // #968
                        c_local_ready.wait();  // #969
                        const GMTensor<float> view_203 = score_ws[(((cube_idx * 196608) + (slot_b_p * 49152)) + 16384)];  // #970
                        const Tensor<float, Position::L0C> slot_15 = l0c.get(l0c_cnt_2);  // #971
                        l0c_to_gm_nz2nd(view_203, slot_15, 128, 128, 128, 128, false, 0.0f, 0, false, false);  // #972
                        c_local_free.set();  // #973
                        const int32_t l1k_cnt_3 = (int32_t)(l1k_cnt_2 + 1);  // #974 #975
                        const int32_t l0c_cnt_3 = (int32_t)(l0c_cnt_2 + 1);  // #976 #977
                        l0ab_cnt_4_state = l0ab_cnt_3;  // #2398
                        l0c_cnt_4_state = l0c_cnt_3;  // #2399
                        l1k_cnt_4_state = l1k_cnt_3;  // #2400
                    } else {
                        l0ab_cnt_4_state = l0ab_cnt_2;  // #2401
                        l0c_cnt_4_state = l0c_cnt_2;  // #2402
                        l1k_cnt_4_state = l1k_cnt_2;  // #2403
                    }
                    const int32_t l0ab_cnt_4 = (int32_t)(l0ab_cnt_4_state);  // #2404
                    const int32_t l0c_cnt_4 = (int32_t)(l0c_cnt_4_state);  // #2405
                    const int32_t l1k_cnt_4 = (int32_t)(l1k_cnt_4_state);  // #2406
                    const int32_t pair_len_2 = Min(2, b_len_p_4 - 4);  // #981 #982
                    const bool cmp_34 = pair_len_2 > 0;  // #983
                    int32_t l0ab_cnt_6_state = 0;  // #2407
                    int32_t l0c_cnt_6_state = 0;  // #2408
                    int32_t l1k_cnt_6_state = 0;  // #2409
                    if (cmp_34) {  // #1116
                        k_local_free.wait();  // #984
                        if (approx_p) {  // #1085
                            const int32_t first_2 = (approx_g_p + 4) * 64;  // #985 #986
                            const int32_t cols_2 = pair_len_2 * 64;  // #987
                            if (pair_len_2 < 2) {  // #988 #992
                                const Tensor<bfloat16_t, Position::L1> slot_16 = l1k.get(l1k_cnt_4);  // #989
                                set_constant_to_l1(slot_16, (bfloat16_t)(0.0f), 1024);  // #990
                                PipeBarrier<PIPE_MTE2>();  // #991
                            }
                            const Tensor<bfloat16_t, Position::L1> slot_17 = l1k.get(l1k_cnt_4);  // #993
                            const Tensor<bfloat16_t, Position::L1> view_204 = l1k.get(l1k_cnt_4);  // #994
                            const int32_t sub_46 = (first_2 + cols_2) - first_2;  // #995 #996
                            const GMTensor<bfloat16_t> view_205 = kc[(((b_1 * ((H * 64) * 128)) + (h_1 * 8192)) + (first_2 * 128))];  // #997
                            gm_to_l1_nd2nz(view_204, view_205, sub_46, 128, 128, 128);  // #998
                        } else {
                            const Tensor<bfloat16_t, Position::L1> slot_18 = l1k.get(l1k_cnt_4);  // #999
                            const int32_t add_60 = e_base_p + 4;  // #1000
                            const int32_t blk1_8 = add_60 + 1;  // #1001
                            const bool not_8 = !(dense);  // #1002
                            int32_t blk0_5_state = 0;  // #2410
                            int32_t blk1_11_state = 0;  // #2411
                            if (not_8) {  // #1034
                                const int32_t blk0_4 = (int32_t)(exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + add_60));  // #1004 #1005 #1006 #1007 #1008 #1009 #1010 #1011 #1012 #1013 #1014
                                const bool cmp_36 = pair_len_2 == 2;  // #1015
                                int32_t blk1_10_state = 0;  // #2412
                                if (cmp_36) {  // #1031
                                    const int32_t add_64 = add_60 + 1;  // #1016
                                    blk1_10_state = (int32_t)(exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + add_64));  // #1018 #1019 #1020 #1021 #1022 #1023 #1024 #1025 #1026 #1027 #1028 #2413
                                } else {
                                    blk1_10_state = blk1_8;  // #2414
                                }
                                const int32_t blk1_10 = (int32_t)(blk1_10_state);  // #2415
                                blk0_5_state = blk0_4;  // #2416
                                blk1_11_state = blk1_10;  // #2417
                            } else {
                                blk0_5_state = add_60;  // #2418
                                blk1_11_state = blk1_8;  // #2419
                            }
                            const int32_t blk0_5 = (int32_t)(blk0_5_state);  // #2420
                            const int32_t blk1_11 = (int32_t)(blk1_11_state);  // #2421
                            const int32_t start0_2 = blk0_5 * 64;  // #1035
                            const int32_t rows0_2 = Min(64, T - start0_2);  // #1036 #1037
                            const bool cmp_37 = pair_len_2 == 2;  // #1038
                            int32_t rows1_5_state = 0;  // #2422
                            if (cmp_37) {  // #1044
                                rows1_5_state = Min(64, T - (blk1_11 * 64));  // #1039 #1040 #1041 #2423
                            } else {
                                rows1_5_state = 64;  // #2424
                            }
                            const int32_t rows1_5 = (int32_t)(rows1_5_state);  // #2425
                            const bool cmp_38 = pair_len_2 < 2;  // #1045
                            bool choice_7_state = false;  // #2426
                            if (cmp_38) {  // #1053
                                choice_7_state = true;  // #2427
                            } else {
                                const bool cmp_39 = rows0_2 < 64;  // #1046
                                bool choice_6_state = false;  // #2428
                                if (cmp_39) {  // #1050
                                    choice_6_state = true;  // #2429
                                } else {
                                    choice_6_state = rows1_5 < 64;  // #1047 #2430
                                }
                                choice_7_state = (bool)(choice_6_state);  // #2431 #2432
                            }
                            const bool choice_7 = (bool)(choice_7_state);  // #2433
                            if (choice_7) {  // #1056
                                set_constant_to_l1(slot_18, (bfloat16_t)(0.0f), 1024);  // #1054
                                PipeBarrier<PIPE_MTE2>();  // #1055
                            }
                            bool choice_8_state = false;  // #2434
                            if (cmp_37) {  // #1062
                                choice_8_state = blk1_11 == (blk0_5 + 1);  // #1058 #1059 #2435
                            } else {
                                choice_8_state = false;  // #2436
                            }
                            const bool choice_8 = (bool)(choice_8_state);  // #2437
                            if (choice_8) {  // #1084
                                const int32_t rows_3 = rows0_2 + rows1_5;  // #1063
                                const Tensor<bfloat16_t, Position::L1> view_206 = l1k.get(l1k_cnt_4);  // #1064
                                const int32_t sub_49 = (start0_2 + rows_3) - start0_2;  // #1065 #1066
                                const GMTensor<bfloat16_t> view_207 = k[(((b_1 * ((T * H) * 128)) + (start0_2 * (H * 128))) + (h_1 * 128))];  // #1067
                                gm_to_l1_nd2nz(view_206, view_207, sub_49, 128, 128, H * 128);  // #2691 #1068
                            } else {
                                const Tensor<bfloat16_t, Position::L1> view_208 = l1k.get(l1k_cnt_4);  // #1069
                                const GMTensor<bfloat16_t> view_209 = k[(((b_1 * ((T * H) * 128)) + (start0_2 * (H * 128))) + (h_1 * 128))];  // #1072
                                const int32_t mul_163 = H * 128;  // #2692
                                gm_to_l1_nd2nz(view_208, view_209, rows0_2, 128, 128, mul_163);  // #1073
                                if (cmp_37) {  // #1083
                                    const int32_t start1_2 = blk1_11 * 64;  // #1075
                                    const Tensor<bfloat16_t, Position::L1> view_210 = l1k.get(l1k_cnt_4)[1024];  // #1078
                                    const GMTensor<bfloat16_t> view_211 = k[(((b_1 * ((T * H) * 128)) + (start1_2 * (H * 128))) + (h_1 * 128))];  // #1081
                                    gm_to_l1_nd2nz(view_210, view_211, rows1_5, 128, 128, mul_163);  // #1082
                                }
                            }
                        }
                        k_local_ready.set();  // #1086
                        ab_local_free.wait();  // #1087
                        k_local_ready.wait();  // #1088
                        const Tensor<bfloat16_t, Position::L0A> q_tile_l0_2 = l0a.get(l0ab_cnt_4);  // #1089
                        l1_to_l0<false>(q_tile_l0_2, l1q, 128, 128, 128, 128);  // #1090
                        const Tensor<bfloat16_t, Position::L0B> slot_19 = l0b.get(l0ab_cnt_4);  // #1091
                        const Tensor<bfloat16_t, Position::L1> slot_20 = l1k.get(l1k_cnt_4);  // #1092
                        l1_to_l0<false>(slot_19, slot_20, 128, 128, 128, 128);  // #1093
                        k_local_free.set();  // #1094
                        ab_local_ready.set();  // #1095
                        c_local_free.wait();  // #1096
                        ab_local_ready.wait();  // #1097
                        const Tensor<float, Position::L0C> slot_21 = l0c.get(l0c_cnt_4);  // #1098
                        const Tensor<bfloat16_t, Position::L0B> slot_22 = l0b.get(l0ab_cnt_4);  // #1099
                        mmad(slot_21, q_tile_l0_2, slot_22, 128, 128, 128, true);  // #1100
                        ab_local_free.set();  // #1101
                        const int32_t l0ab_cnt_5 = (int32_t)(l0ab_cnt_4 + 1);  // #1102 #1103
                        c_local_ready.set();  // #1104
                        c_local_ready.wait();  // #1105
                        const GMTensor<float> view_212 = score_ws[(((cube_idx * 196608) + (slot_b_p * 49152)) + 32768)];  // #1106
                        const Tensor<float, Position::L0C> slot_23 = l0c.get(l0c_cnt_4);  // #1107
                        l0c_to_gm_nz2nd(view_212, slot_23, 128, 128, 128, 128, false, 0.0f, 0, false, false);  // #1108
                        c_local_free.set();  // #1109
                        const int32_t l1k_cnt_5 = (int32_t)(l1k_cnt_4 + 1);  // #1110 #1111
                        const int32_t l0c_cnt_5 = (int32_t)(l0c_cnt_4 + 1);  // #1112 #1113
                        l0ab_cnt_6_state = l0ab_cnt_5;  // #2438
                        l0c_cnt_6_state = l0c_cnt_5;  // #2439
                        l1k_cnt_6_state = l1k_cnt_5;  // #2440
                    } else {
                        l0ab_cnt_6_state = l0ab_cnt_4;  // #2441
                        l0c_cnt_6_state = l0c_cnt_4;  // #2442
                        l1k_cnt_6_state = l1k_cnt_4;  // #2443
                    }
                    const int32_t l0ab_cnt_6 = (int32_t)(l0ab_cnt_6_state);  // #2444
                    const int32_t l0c_cnt_6 = (int32_t)(l0c_cnt_6_state);  // #2445
                    const int32_t l1k_cnt_6 = (int32_t)(l1k_cnt_6_state);  // #2446
                    CUBE_READY<PIPE_FIX>(0);  // #1117
                    const int32_t pair_len_6 = Min(2, b_len_p_4);  // #1158
                    if (pair_len_6 > 0) {  // #1159 #1262
                        if (approx_p) {  // #1251
                        } else {
                            if (0 < pair_len_6) {  // #1176 #1212
                                if (!(dense)) {  // #1179 #1194
                                    const uint16_t load_7 = exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + e_base_p);  // #1181 #1182 #1183 #1184 #1185 #1186 #1187 #1188 #1189 #1190
                                }
                            }
                            if (1 < pair_len_6) {  // #1214 #1250
                                const int32_t blk_p_3 = e_base_p + 1;  // #1216
                                if (!(dense)) {  // #1217 #1232
                                    const uint16_t load_8 = exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + blk_p_3);  // #1219 #1220 #1221 #1222 #1223 #1224 #1225 #1226 #1227 #1228
                                }
                            }
                        }
                    }
                    const int32_t pair_len_7 = Min(2, b_len_p_4 - 2);  // #1263 #1264
                    if (pair_len_7 > 0) {  // #1265 #1368
                        if (approx_p) {  // #1357
                        } else {
                            if (0 < pair_len_7) {  // #1282 #1318
                                const int32_t add_87 = e_base_p + 2;  // #1283
                                if (!(dense)) {  // #1285 #1300
                                    const uint16_t load_9 = exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + add_87);  // #1287 #1288 #1289 #1290 #1291 #1292 #1293 #1294 #1295 #1296
                                }
                            }
                            if (1 < pair_len_7) {  // #1320 #1356
                                const int32_t blk_p_9 = (e_base_p + 2) + 1;  // #1321 #1322
                                if (!(dense)) {  // #1323 #1338
                                    const uint16_t load_10 = exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + blk_p_9);  // #1325 #1326 #1327 #1328 #1329 #1330 #1331 #1332 #1333 #1334
                                }
                            }
                        }
                    }
                    const int32_t pair_len_8 = Min(2, b_len_p_4 - 4);  // #1369 #1370
                    if (pair_len_8 > 0) {  // #1371 #1474
                        if (approx_p) {  // #1463
                        } else {
                            if (0 < pair_len_8) {  // #1388 #1424
                                const int32_t add_95 = e_base_p + 4;  // #1389
                                if (!(dense)) {  // #1391 #1406
                                    const uint16_t load_11 = exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + add_95);  // #1393 #1394 #1395 #1396 #1397 #1398 #1399 #1400 #1401 #1402
                                }
                            }
                            if (1 < pair_len_8) {  // #1426 #1462
                                const int32_t blk_p_15 = (e_base_p + 4) + 1;  // #1427 #1428
                                if (!(dense)) {  // #1429 #1444
                                    const uint16_t load_12 = exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + blk_p_15);  // #1431 #1432 #1433 #1434 #1435 #1436 #1437 #1438 #1439 #1440
                                }
                            }
                        }
                    }
                    l0ab_cnt_7_state = l0ab_cnt_6;  // #2516
                    l0c_cnt_7_state = l0c_cnt_6;  // #2517
                    l1k_cnt_7_state = l1k_cnt_6;  // #2518
                } else {
                    l0ab_cnt_7_state = l0ab_cnt_carry_1;  // #2519
                    l0c_cnt_7_state = l0c_cnt_carry_1;  // #2520
                    l1k_cnt_7_state = l1k_cnt_carry_1;  // #2521
                }
                const int32_t l0ab_cnt_7 = (int32_t)(l0ab_cnt_7_state);  // #2522
                const int32_t l0c_cnt_7 = (int32_t)(l0c_cnt_7_state);  // #2523
                const int32_t l1k_cnt_7 = (int32_t)(l1k_cnt_7_state);  // #2524
                b_len_p_5_state = b_len_p_4;  // #2525
                l0ab_cnt_8_state = l0ab_cnt_7;  // #2526
                l0c_cnt_8_state = l0c_cnt_7;  // #2527
                l1k_cnt_8_state = l1k_cnt_7;  // #2528
            } else {
                b_len_p_5_state = 0;  // #2529
                l0ab_cnt_8_state = l0ab_cnt_carry_1;  // #2530
                l0c_cnt_8_state = l0c_cnt_carry_1;  // #2531
                l1k_cnt_8_state = l1k_cnt_carry_1;  // #2532
            }
            const int32_t b_len_p_5 = (int32_t)(b_len_p_5_state);  // #2533
            const int32_t l0ab_cnt_8 = (int32_t)(l0ab_cnt_8_state);  // #2534
            const int32_t l0c_cnt_8 = (int32_t)(l0c_cnt_8_state);  // #2535
            const int32_t l1k_cnt_8 = (int32_t)(l1k_cnt_8_state);  // #2536
            const int32_t ready_lengths_1 = ((ready_lengths_carry & 4095) << 4) | b_len_p_5;  // #1750 #1751 #1752
            const int32_t b_len_c = ready_lengths_1 >> 12;  // #1753
            const bool cmp_83 = b_len_c > 0;  // #1754
            int32_t accum_step_2_state = 0;  // #2537
            int32_t l0ab_cnt_15_state = 0;  // #2538
            int32_t l0c_cnt_10_state = 0;  // #2539
            int32_t l1pv_cnt_7_state = 0;  // #2540
            if (cmp_83) {  // #2208
                const int32_t consumed = beat + -3;  // #1755
                const int32_t slot_24 = consumed & 3;  // #1756
                const bool approx_c = consumed < approx_beats_1;  // #1757
                const int32_t approx_g_c = consumed * 2;  // #1758
                const int32_t e_base_c = Max(0, consumed - approx_beats_1) * 6;  // #1759 #1760 #1761
                WAIT_VEC<PIPE_S>(1);  // #1762
                c_local_free.wait();  // #1763
                const int32_t pair_len_12 = Min(2, b_len_c);  // #1765
                const bool cmp_84 = pair_len_12 > 0;  // #1766
                int32_t l0ab_cnt_10_state = 0;  // #2541
                int32_t l1pv_cnt_2_state = 0;  // #2542
                if (cmp_84) {  // #1895
                    pv_local_free.wait();  // #1767
                    const Tensor<bfloat16_t, Position::L1> slot_25 = l1p.get(l1pv_cnt_carry_1);  // #1768
                    const GMTensor<bfloat16_t> view_237 = p_ws[((cube_idx * 196608) + (slot_24 * 49152))];  // #1769
                    gm_to_l1_nd2nz(slot_25, view_237, 128, 128, 128, 128);  // #1770
                    if (approx_c) {  // #1871
                        const int32_t first_3 = approx_g_c * 64;  // #1772
                        const int32_t cols_3 = pair_len_12 * 64;  // #1773
                        if (pair_len_12 < 2) {  // #1774 #1778
                            const Tensor<bfloat16_t, Position::L1> slot_26 = l1v.get(l1pv_cnt_carry_1);  // #1775
                            set_constant_to_l1(slot_26, (bfloat16_t)(0.0f), 1024);  // #1776
                            PipeBarrier<PIPE_MTE2>();  // #1777
                        }
                        const Tensor<bfloat16_t, Position::L1> slot_27 = l1v.get(l1pv_cnt_carry_1);  // #1779
                        const Tensor<bfloat16_t, Position::L1> view_238 = l1v.get(l1pv_cnt_carry_1);  // #1780
                        const int32_t sub_84 = (first_3 + cols_3) - first_3;  // #1781 #1782
                        const GMTensor<bfloat16_t> view_239 = vc[(((b_1 * ((H * 64) * 128)) + (h_1 * 8192)) + (first_3 * 128))];  // #1783
                        gm_to_l1_nd2nz(view_238, view_239, sub_84, 128, 128, 128);  // #1784
                    } else {
                        const Tensor<bfloat16_t, Position::L1> slot_28 = l1v.get(l1pv_cnt_carry_1);  // #1785
                        const int32_t blk1_12 = e_base_c + 1;  // #1787
                        const bool not_24 = !(dense);  // #1788
                        int32_t blk0_7_state = 0;  // #2543
                        int32_t blk1_15_state = 0;  // #2544
                        if (not_24) {  // #1820
                            const int32_t blk0_6 = (int32_t)(exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + e_base_c));  // #1790 #1791 #1792 #1793 #1794 #1795 #1796 #1797 #1798 #1799 #1800
                            const bool cmp_86 = pair_len_12 == 2;  // #1801
                            int32_t blk1_14_state = 0;  // #2545
                            if (cmp_86) {  // #1817
                                const int32_t add_132 = e_base_c + 1;  // #1802
                                blk1_14_state = (int32_t)(exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + add_132));  // #1804 #1805 #1806 #1807 #1808 #1809 #1810 #1811 #1812 #1813 #1814 #2546
                            } else {
                                blk1_14_state = blk1_12;  // #2547
                            }
                            const int32_t blk1_14 = (int32_t)(blk1_14_state);  // #2548
                            blk0_7_state = blk0_6;  // #2549
                            blk1_15_state = blk1_14;  // #2550
                        } else {
                            blk0_7_state = e_base_c;  // #2551
                            blk1_15_state = blk1_12;  // #2552
                        }
                        const int32_t blk0_7 = (int32_t)(blk0_7_state);  // #2553
                        const int32_t blk1_15 = (int32_t)(blk1_15_state);  // #2554
                        const int32_t start0_3 = blk0_7 * 64;  // #1821
                        const int32_t rows0_3 = Min(64, T - start0_3);  // #1822 #1823
                        const bool cmp_87 = pair_len_12 == 2;  // #1824
                        int32_t rows1_7_state = 0;  // #2555
                        if (cmp_87) {  // #1830
                            rows1_7_state = Min(64, T - (blk1_15 * 64));  // #1825 #1826 #1827 #2556
                        } else {
                            rows1_7_state = 64;  // #2557
                        }
                        const int32_t rows1_7 = (int32_t)(rows1_7_state);  // #2558
                        const bool cmp_88 = pair_len_12 < 2;  // #1831
                        bool choice_19_state = false;  // #2559
                        if (cmp_88) {  // #1839
                            choice_19_state = true;  // #2560
                        } else {
                            const bool cmp_89 = rows0_3 < 64;  // #1832
                            bool choice_18_state = false;  // #2561
                            if (cmp_89) {  // #1836
                                choice_18_state = true;  // #2562
                            } else {
                                choice_18_state = rows1_7 < 64;  // #1833 #2563
                            }
                            choice_19_state = (bool)(choice_18_state);  // #2564 #2565
                        }
                        const bool choice_19 = (bool)(choice_19_state);  // #2566
                        if (choice_19) {  // #1842
                            set_constant_to_l1(slot_28, (bfloat16_t)(0.0f), 1024);  // #1840
                            PipeBarrier<PIPE_MTE2>();  // #1841
                        }
                        bool choice_20_state = false;  // #2567
                        if (cmp_87) {  // #1848
                            choice_20_state = blk1_15 == (blk0_7 + 1);  // #1844 #1845 #2568
                        } else {
                            choice_20_state = false;  // #2569
                        }
                        const bool choice_20 = (bool)(choice_20_state);  // #2570
                        if (choice_20) {  // #1870
                            const int32_t rows_4 = rows0_3 + rows1_7;  // #1849
                            const Tensor<bfloat16_t, Position::L1> view_240 = l1v.get(l1pv_cnt_carry_1);  // #1850
                            const int32_t sub_87 = (start0_3 + rows_4) - start0_3;  // #1851 #1852
                            const GMTensor<bfloat16_t> view_241 = v[(((b_1 * ((T * H) * 128)) + (start0_3 * (H * 128))) + (h_1 * 128))];  // #1853
                            gm_to_l1_nd2nz(view_240, view_241, sub_87, 128, 128, H * 128);  // #2694 #1854
                        } else {
                            const Tensor<bfloat16_t, Position::L1> view_242 = l1v.get(l1pv_cnt_carry_1);  // #1855
                            const GMTensor<bfloat16_t> view_243 = v[(((b_1 * ((T * H) * 128)) + (start0_3 * (H * 128))) + (h_1 * 128))];  // #1858
                            const int32_t mul_166 = H * 128;  // #2695
                            gm_to_l1_nd2nz(view_242, view_243, rows0_3, 128, 128, mul_166);  // #1859
                            if (cmp_87) {  // #1869
                                const int32_t start1_3 = blk1_15 * 64;  // #1861
                                const Tensor<bfloat16_t, Position::L1> view_244 = l1v.get(l1pv_cnt_carry_1)[1024];  // #1864
                                const GMTensor<bfloat16_t> view_245 = v[(((b_1 * ((T * H) * 128)) + (start1_3 * (H * 128))) + (h_1 * 128))];  // #1867
                                gm_to_l1_nd2nz(view_244, view_245, rows1_7, 128, 128, mul_166);  // #1868
                            }
                        }
                    }
                    pv_local_ready.set();  // #1872
                    ab_local_free.wait();  // #1873
                    pv_local_ready.wait();  // #1874
                    const Tensor<bfloat16_t, Position::L0A> slot_29 = l0a.get(l0ab_cnt_8);  // #1875
                    const Tensor<bfloat16_t, Position::L1> slot_30 = l1p.get(l1pv_cnt_carry_1);  // #1876
                    l1_to_l0<false>(slot_29, slot_30, 128, 128, 128, 128);  // #1877
                    const Tensor<bfloat16_t, Position::L0B> slot_31 = l0b.get(l0ab_cnt_8);  // #1878
                    const Tensor<bfloat16_t, Position::L1> slot_32 = l1v.get(l1pv_cnt_carry_1);  // #1879
                    l1_to_l0<true>(slot_31, slot_32, 128, 128, 128, 128);  // #1880
                    pv_local_free.set();  // #1881
                    ab_local_ready.set();  // #1882
                    ab_local_ready.wait();  // #1883
                    const Tensor<float, Position::L0C> slot_33 = l0c.get(l0c_cnt_8);  // #1884
                    const Tensor<bfloat16_t, Position::L0A> slot_34 = l0a.get(l0ab_cnt_8);  // #1885
                    const Tensor<bfloat16_t, Position::L0B> slot_35 = l0b.get(l0ab_cnt_8);  // #1886
                    mmad(slot_33, slot_34, slot_35, 128, 128, 128, true);  // #1887
                    ab_local_free.set();  // #1888
                    const int32_t l0ab_cnt_9 = (int32_t)(l0ab_cnt_8 + 1);  // #1889 #1890
                    const int32_t l1pv_cnt_1 = (int32_t)(l1pv_cnt_carry_1 + 1);  // #1891 #1892
                    l0ab_cnt_10_state = l0ab_cnt_9;  // #2571
                    l1pv_cnt_2_state = l1pv_cnt_1;  // #2572
                } else {
                    l0ab_cnt_10_state = l0ab_cnt_8;  // #2573
                    l1pv_cnt_2_state = l1pv_cnt_carry_1;  // #2574
                }
                const int32_t l0ab_cnt_10 = (int32_t)(l0ab_cnt_10_state);  // #2575
                const int32_t l1pv_cnt_2 = (int32_t)(l1pv_cnt_2_state);  // #2576
                const int32_t pair_len_13 = Min(2, b_len_c - 2);  // #1896 #1897
                const bool cmp_94 = pair_len_13 > 0;  // #1898
                int32_t l0ab_cnt_12_state = 0;  // #2577
                int32_t l1pv_cnt_4_state = 0;  // #2578
                if (cmp_94) {  // #2028
                    pv_local_free.wait();  // #1899
                    const Tensor<bfloat16_t, Position::L1> slot_36 = l1p.get(l1pv_cnt_2);  // #1900
                    const GMTensor<bfloat16_t> view_246 = p_ws[(((cube_idx * 196608) + (slot_24 * 49152)) + 16384)];  // #1901
                    gm_to_l1_nd2nz(slot_36, view_246, 128, 128, 128, 128);  // #1902
                    if (approx_c) {  // #2003
                        const int32_t first_4 = (approx_g_c + 2) * 64;  // #1903 #1904
                        const int32_t cols_4 = pair_len_13 * 64;  // #1905
                        if (pair_len_13 < 2) {  // #1906 #1910
                            const Tensor<bfloat16_t, Position::L1> slot_37 = l1v.get(l1pv_cnt_2);  // #1907
                            set_constant_to_l1(slot_37, (bfloat16_t)(0.0f), 1024);  // #1908
                            PipeBarrier<PIPE_MTE2>();  // #1909
                        }
                        const Tensor<bfloat16_t, Position::L1> slot_38 = l1v.get(l1pv_cnt_2);  // #1911
                        const Tensor<bfloat16_t, Position::L1> view_247 = l1v.get(l1pv_cnt_2);  // #1912
                        const int32_t sub_92 = (first_4 + cols_4) - first_4;  // #1913 #1914
                        const GMTensor<bfloat16_t> view_248 = vc[(((b_1 * ((H * 64) * 128)) + (h_1 * 8192)) + (first_4 * 128))];  // #1915
                        gm_to_l1_nd2nz(view_247, view_248, sub_92, 128, 128, 128);  // #1916
                    } else {
                        const Tensor<bfloat16_t, Position::L1> slot_39 = l1v.get(l1pv_cnt_2);  // #1917
                        const int32_t add_145 = e_base_c + 2;  // #1918
                        const int32_t blk1_16 = add_145 + 1;  // #1919
                        const bool not_25 = !(dense);  // #1920
                        int32_t blk0_9_state = 0;  // #2579
                        int32_t blk1_19_state = 0;  // #2580
                        if (not_25) {  // #1952
                            const int32_t blk0_8 = (int32_t)(exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + add_145));  // #1922 #1923 #1924 #1925 #1926 #1927 #1928 #1929 #1930 #1931 #1932
                            const bool cmp_96 = pair_len_13 == 2;  // #1933
                            int32_t blk1_18_state = 0;  // #2581
                            if (cmp_96) {  // #1949
                                const int32_t add_149 = add_145 + 1;  // #1934
                                blk1_18_state = (int32_t)(exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + add_149));  // #1936 #1937 #1938 #1939 #1940 #1941 #1942 #1943 #1944 #1945 #1946 #2582
                            } else {
                                blk1_18_state = blk1_16;  // #2583
                            }
                            const int32_t blk1_18 = (int32_t)(blk1_18_state);  // #2584
                            blk0_9_state = blk0_8;  // #2585
                            blk1_19_state = blk1_18;  // #2586
                        } else {
                            blk0_9_state = add_145;  // #2587
                            blk1_19_state = blk1_16;  // #2588
                        }
                        const int32_t blk0_9 = (int32_t)(blk0_9_state);  // #2589
                        const int32_t blk1_19 = (int32_t)(blk1_19_state);  // #2590
                        const int32_t start0_4 = blk0_9 * 64;  // #1953
                        const int32_t rows0_4 = Min(64, T - start0_4);  // #1954 #1955
                        const bool cmp_97 = pair_len_13 == 2;  // #1956
                        int32_t rows1_9_state = 0;  // #2591
                        if (cmp_97) {  // #1962
                            rows1_9_state = Min(64, T - (blk1_19 * 64));  // #1957 #1958 #1959 #2592
                        } else {
                            rows1_9_state = 64;  // #2593
                        }
                        const int32_t rows1_9 = (int32_t)(rows1_9_state);  // #2594
                        const bool cmp_98 = pair_len_13 < 2;  // #1963
                        bool choice_22_state = false;  // #2595
                        if (cmp_98) {  // #1971
                            choice_22_state = true;  // #2596
                        } else {
                            const bool cmp_99 = rows0_4 < 64;  // #1964
                            bool choice_21_state = false;  // #2597
                            if (cmp_99) {  // #1968
                                choice_21_state = true;  // #2598
                            } else {
                                choice_21_state = rows1_9 < 64;  // #1965 #2599
                            }
                            choice_22_state = (bool)(choice_21_state);  // #2600 #2601
                        }
                        const bool choice_22 = (bool)(choice_22_state);  // #2602
                        if (choice_22) {  // #1974
                            set_constant_to_l1(slot_39, (bfloat16_t)(0.0f), 1024);  // #1972
                            PipeBarrier<PIPE_MTE2>();  // #1973
                        }
                        bool choice_23_state = false;  // #2603
                        if (cmp_97) {  // #1980
                            choice_23_state = blk1_19 == (blk0_9 + 1);  // #1976 #1977 #2604
                        } else {
                            choice_23_state = false;  // #2605
                        }
                        const bool choice_23 = (bool)(choice_23_state);  // #2606
                        if (choice_23) {  // #2002
                            const int32_t rows_5 = rows0_4 + rows1_9;  // #1981
                            const Tensor<bfloat16_t, Position::L1> view_249 = l1v.get(l1pv_cnt_2);  // #1982
                            const int32_t sub_95 = (start0_4 + rows_5) - start0_4;  // #1983 #1984
                            const GMTensor<bfloat16_t> view_250 = v[(((b_1 * ((T * H) * 128)) + (start0_4 * (H * 128))) + (h_1 * 128))];  // #1985
                            gm_to_l1_nd2nz(view_249, view_250, sub_95, 128, 128, H * 128);  // #2697 #1986
                        } else {
                            const Tensor<bfloat16_t, Position::L1> view_251 = l1v.get(l1pv_cnt_2);  // #1987
                            const GMTensor<bfloat16_t> view_252 = v[(((b_1 * ((T * H) * 128)) + (start0_4 * (H * 128))) + (h_1 * 128))];  // #1990
                            const int32_t mul_169 = H * 128;  // #2698
                            gm_to_l1_nd2nz(view_251, view_252, rows0_4, 128, 128, mul_169);  // #1991
                            if (cmp_97) {  // #2001
                                const int32_t start1_4 = blk1_19 * 64;  // #1993
                                const Tensor<bfloat16_t, Position::L1> view_253 = l1v.get(l1pv_cnt_2)[1024];  // #1996
                                const GMTensor<bfloat16_t> view_254 = v[(((b_1 * ((T * H) * 128)) + (start1_4 * (H * 128))) + (h_1 * 128))];  // #1999
                                gm_to_l1_nd2nz(view_253, view_254, rows1_9, 128, 128, mul_169);  // #2000
                            }
                        }
                    }
                    pv_local_ready.set();  // #2004
                    ab_local_free.wait();  // #2005
                    pv_local_ready.wait();  // #2006
                    const Tensor<bfloat16_t, Position::L0A> slot_40 = l0a.get(l0ab_cnt_10);  // #2007
                    const Tensor<bfloat16_t, Position::L1> slot_41 = l1p.get(l1pv_cnt_2);  // #2008
                    l1_to_l0<false>(slot_40, slot_41, 128, 128, 128, 128);  // #2009
                    const Tensor<bfloat16_t, Position::L0B> slot_42 = l0b.get(l0ab_cnt_10);  // #2010
                    const Tensor<bfloat16_t, Position::L1> slot_43 = l1v.get(l1pv_cnt_2);  // #2011
                    l1_to_l0<true>(slot_42, slot_43, 128, 128, 128, 128);  // #2012
                    pv_local_free.set();  // #2013
                    ab_local_ready.set();  // #2014
                    ab_local_ready.wait();  // #2015
                    PipeBarrier<PIPE_M>();  // #2016
                    const Tensor<float, Position::L0C> slot_44 = l0c.get(l0c_cnt_8);  // #2017
                    const Tensor<bfloat16_t, Position::L0A> slot_45 = l0a.get(l0ab_cnt_10);  // #2018
                    const Tensor<bfloat16_t, Position::L0B> slot_46 = l0b.get(l0ab_cnt_10);  // #2019
                    mmad(slot_44, slot_45, slot_46, 128, 128, 128, false);  // #2020
                    ab_local_free.set();  // #2021
                    const int32_t l0ab_cnt_11 = (int32_t)(l0ab_cnt_10 + 1);  // #2022 #2023
                    const int32_t l1pv_cnt_3 = (int32_t)(l1pv_cnt_2 + 1);  // #2024 #2025
                    l0ab_cnt_12_state = l0ab_cnt_11;  // #2607
                    l1pv_cnt_4_state = l1pv_cnt_3;  // #2608
                } else {
                    l0ab_cnt_12_state = l0ab_cnt_10;  // #2609
                    l1pv_cnt_4_state = l1pv_cnt_2;  // #2610
                }
                const int32_t l0ab_cnt_12 = (int32_t)(l0ab_cnt_12_state);  // #2611
                const int32_t l1pv_cnt_4 = (int32_t)(l1pv_cnt_4_state);  // #2612
                const int32_t pair_len_14 = Min(2, b_len_c - 4);  // #2029 #2030
                const bool cmp_104 = pair_len_14 > 0;  // #2031
                int32_t l0ab_cnt_14_state = 0;  // #2613
                int32_t l1pv_cnt_6_state = 0;  // #2614
                if (cmp_104) {  // #2161
                    pv_local_free.wait();  // #2032
                    const Tensor<bfloat16_t, Position::L1> slot_47 = l1p.get(l1pv_cnt_4);  // #2033
                    const GMTensor<bfloat16_t> view_255 = p_ws[(((cube_idx * 196608) + (slot_24 * 49152)) + 32768)];  // #2034
                    gm_to_l1_nd2nz(slot_47, view_255, 128, 128, 128, 128);  // #2035
                    if (approx_c) {  // #2136
                        const int32_t first_5 = (approx_g_c + 4) * 64;  // #2036 #2037
                        const int32_t cols_5 = pair_len_14 * 64;  // #2038
                        if (pair_len_14 < 2) {  // #2039 #2043
                            const Tensor<bfloat16_t, Position::L1> slot_48 = l1v.get(l1pv_cnt_4);  // #2040
                            set_constant_to_l1(slot_48, (bfloat16_t)(0.0f), 1024);  // #2041
                            PipeBarrier<PIPE_MTE2>();  // #2042
                        }
                        const Tensor<bfloat16_t, Position::L1> slot_49 = l1v.get(l1pv_cnt_4);  // #2044
                        const Tensor<bfloat16_t, Position::L1> view_256 = l1v.get(l1pv_cnt_4);  // #2045
                        const int32_t sub_100 = (first_5 + cols_5) - first_5;  // #2046 #2047
                        const GMTensor<bfloat16_t> view_257 = vc[(((b_1 * ((H * 64) * 128)) + (h_1 * 8192)) + (first_5 * 128))];  // #2048
                        gm_to_l1_nd2nz(view_256, view_257, sub_100, 128, 128, 128);  // #2049
                    } else {
                        const Tensor<bfloat16_t, Position::L1> slot_50 = l1v.get(l1pv_cnt_4);  // #2050
                        const int32_t add_162 = e_base_c + 4;  // #2051
                        const int32_t blk1_20 = add_162 + 1;  // #2052
                        const bool not_26 = !(dense);  // #2053
                        int32_t blk0_11_state = 0;  // #2615
                        int32_t blk1_23_state = 0;  // #2616
                        if (not_26) {  // #2085
                            const int32_t blk0_10 = (int32_t)(exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + add_162));  // #2055 #2056 #2057 #2058 #2059 #2060 #2061 #2062 #2063 #2064 #2065
                            const bool cmp_106 = pair_len_14 == 2;  // #2066
                            int32_t blk1_22_state = 0;  // #2617
                            if (cmp_106) {  // #2082
                                const int32_t add_166 = add_162 + 1;  // #2067
                                blk1_22_state = (int32_t)(exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + add_166));  // #2069 #2070 #2071 #2072 #2073 #2074 #2075 #2076 #2077 #2078 #2079 #2618
                            } else {
                                blk1_22_state = blk1_20;  // #2619
                            }
                            const int32_t blk1_22 = (int32_t)(blk1_22_state);  // #2620
                            blk0_11_state = blk0_10;  // #2621
                            blk1_23_state = blk1_22;  // #2622
                        } else {
                            blk0_11_state = add_162;  // #2623
                            blk1_23_state = blk1_20;  // #2624
                        }
                        const int32_t blk0_11 = (int32_t)(blk0_11_state);  // #2625
                        const int32_t blk1_23 = (int32_t)(blk1_23_state);  // #2626
                        const int32_t start0_5 = blk0_11 * 64;  // #2086
                        const int32_t rows0_5 = Min(64, T - start0_5);  // #2087 #2088
                        const bool cmp_107 = pair_len_14 == 2;  // #2089
                        int32_t rows1_11_state = 0;  // #2627
                        if (cmp_107) {  // #2095
                            rows1_11_state = Min(64, T - (blk1_23 * 64));  // #2090 #2091 #2092 #2628
                        } else {
                            rows1_11_state = 64;  // #2629
                        }
                        const int32_t rows1_11 = (int32_t)(rows1_11_state);  // #2630
                        const bool cmp_108 = pair_len_14 < 2;  // #2096
                        bool choice_25_state = false;  // #2631
                        if (cmp_108) {  // #2104
                            choice_25_state = true;  // #2632
                        } else {
                            const bool cmp_109 = rows0_5 < 64;  // #2097
                            bool choice_24_state = false;  // #2633
                            if (cmp_109) {  // #2101
                                choice_24_state = true;  // #2634
                            } else {
                                choice_24_state = rows1_11 < 64;  // #2098 #2635
                            }
                            choice_25_state = (bool)(choice_24_state);  // #2636 #2637
                        }
                        const bool choice_25 = (bool)(choice_25_state);  // #2638
                        if (choice_25) {  // #2107
                            set_constant_to_l1(slot_50, (bfloat16_t)(0.0f), 1024);  // #2105
                            PipeBarrier<PIPE_MTE2>();  // #2106
                        }
                        bool choice_26_state = false;  // #2639
                        if (cmp_107) {  // #2113
                            choice_26_state = blk1_23 == (blk0_11 + 1);  // #2109 #2110 #2640
                        } else {
                            choice_26_state = false;  // #2641
                        }
                        const bool choice_26 = (bool)(choice_26_state);  // #2642
                        if (choice_26) {  // #2135
                            const int32_t rows_6 = rows0_5 + rows1_11;  // #2114
                            const Tensor<bfloat16_t, Position::L1> view_258 = l1v.get(l1pv_cnt_4);  // #2115
                            const int32_t sub_103 = (start0_5 + rows_6) - start0_5;  // #2116 #2117
                            const GMTensor<bfloat16_t> view_259 = v[(((b_1 * ((T * H) * 128)) + (start0_5 * (H * 128))) + (h_1 * 128))];  // #2118
                            gm_to_l1_nd2nz(view_258, view_259, sub_103, 128, 128, H * 128);  // #2700 #2119
                        } else {
                            const Tensor<bfloat16_t, Position::L1> view_260 = l1v.get(l1pv_cnt_4);  // #2120
                            const GMTensor<bfloat16_t> view_261 = v[(((b_1 * ((T * H) * 128)) + (start0_5 * (H * 128))) + (h_1 * 128))];  // #2123
                            const int32_t mul_172 = H * 128;  // #2701
                            gm_to_l1_nd2nz(view_260, view_261, rows0_5, 128, 128, mul_172);  // #2124
                            if (cmp_107) {  // #2134
                                const int32_t start1_5 = blk1_23 * 64;  // #2126
                                const Tensor<bfloat16_t, Position::L1> view_262 = l1v.get(l1pv_cnt_4)[1024];  // #2129
                                const GMTensor<bfloat16_t> view_263 = v[(((b_1 * ((T * H) * 128)) + (start1_5 * (H * 128))) + (h_1 * 128))];  // #2132
                                gm_to_l1_nd2nz(view_262, view_263, rows1_11, 128, 128, mul_172);  // #2133
                            }
                        }
                    }
                    pv_local_ready.set();  // #2137
                    ab_local_free.wait();  // #2138
                    pv_local_ready.wait();  // #2139
                    const Tensor<bfloat16_t, Position::L0A> slot_51 = l0a.get(l0ab_cnt_12);  // #2140
                    const Tensor<bfloat16_t, Position::L1> slot_52 = l1p.get(l1pv_cnt_4);  // #2141
                    l1_to_l0<false>(slot_51, slot_52, 128, 128, 128, 128);  // #2142
                    const Tensor<bfloat16_t, Position::L0B> slot_53 = l0b.get(l0ab_cnt_12);  // #2143
                    const Tensor<bfloat16_t, Position::L1> slot_54 = l1v.get(l1pv_cnt_4);  // #2144
                    l1_to_l0<true>(slot_53, slot_54, 128, 128, 128, 128);  // #2145
                    pv_local_free.set();  // #2146
                    ab_local_ready.set();  // #2147
                    ab_local_ready.wait();  // #2148
                    PipeBarrier<PIPE_M>();  // #2149
                    const Tensor<float, Position::L0C> slot_55 = l0c.get(l0c_cnt_8);  // #2150
                    const Tensor<bfloat16_t, Position::L0A> slot_56 = l0a.get(l0ab_cnt_12);  // #2151
                    const Tensor<bfloat16_t, Position::L0B> slot_57 = l0b.get(l0ab_cnt_12);  // #2152
                    mmad(slot_55, slot_56, slot_57, 128, 128, 128, false);  // #2153
                    ab_local_free.set();  // #2154
                    const int32_t l0ab_cnt_13 = (int32_t)(l0ab_cnt_12 + 1);  // #2155 #2156
                    const int32_t l1pv_cnt_5 = (int32_t)(l1pv_cnt_4 + 1);  // #2157 #2158
                    l0ab_cnt_14_state = l0ab_cnt_13;  // #2643
                    l1pv_cnt_6_state = l1pv_cnt_5;  // #2644
                } else {
                    l0ab_cnt_14_state = l0ab_cnt_12;  // #2645
                    l1pv_cnt_6_state = l1pv_cnt_4;  // #2646
                }
                const int32_t l0ab_cnt_14 = (int32_t)(l0ab_cnt_14_state);  // #2647
                const int32_t l1pv_cnt_6 = (int32_t)(l1pv_cnt_6_state);  // #2648
                c_local_ready.set();  // #2162
                c_local_ready.wait();  // #2163
                const GMTensor<float> view_264 = pv_ws[((cube_idx * 65536) + (slot_24 * 16384))];  // #2164
                const Tensor<float, Position::L0C> slot_58 = l0c.get(l0c_cnt_8);  // #2165
                l0c_to_gm_nz2nd(view_264, slot_58, 128, 128, 128, 128, false, 0.0f, 0, false, false);  // #2166
                c_local_free.set();  // #2167
                CUBE_READY<PIPE_FIX>(2);  // #2168
                const int32_t accum_step_1 = (int32_t)(accum_step_carry + 1);  // #2202 #2203
                const int32_t l0c_cnt_9 = (int32_t)(l0c_cnt_8 + 1);  // #2204 #2205
                accum_step_2_state = accum_step_1;  // #2649
                l0ab_cnt_15_state = l0ab_cnt_14;  // #2650
                l0c_cnt_10_state = l0c_cnt_9;  // #2651
                l1pv_cnt_7_state = l1pv_cnt_6;  // #2652
            } else {
                accum_step_2_state = accum_step_carry;  // #2653
                l0ab_cnt_15_state = l0ab_cnt_8;  // #2654
                l0c_cnt_10_state = l0c_cnt_8;  // #2655
                l1pv_cnt_7_state = l1pv_cnt_carry_1;  // #2656
            }
            const int32_t accum_step_2 = (int32_t)(accum_step_2_state);  // #2657
            const int32_t l0ab_cnt_15 = (int32_t)(l0ab_cnt_15_state);  // #2658
            const int32_t l0c_cnt_10 = (int32_t)(l0c_cnt_10_state);  // #2659
            const int32_t l1pv_cnt_7 = (int32_t)(l1pv_cnt_7_state);  // #2660
            accum_step_3_state = accum_step_2;  // #2661
            l0ab_cnt_16_state = l0ab_cnt_15;  // #2662
            l0c_cnt_11_state = l0c_cnt_10;  // #2663
            l1k_cnt_9_state = l1k_cnt_8;  // #2664
            l1pv_cnt_8_state = l1pv_cnt_7;  // #2665
            ready_lengths_2_state = ready_lengths_1;  // #2666
        }
        const int32_t l0ab_cnt_16 = (int32_t)(l0ab_cnt_16_state);  // #2669
        const int32_t l0c_cnt_11 = (int32_t)(l0c_cnt_11_state);  // #2670
        const int32_t l1k_cnt_9 = (int32_t)(l1k_cnt_9_state);  // #2671
        const int32_t l1pv_cnt_8 = (int32_t)(l1pv_cnt_8_state);  // #2672
        PipeBarrier<PIPE_ALL>();  // #2215
        q_input_free.set();  // #2216
        l0ab_cnt_17_state = l0ab_cnt_16;  // #2673
        l0c_cnt_12_state = l0c_cnt_11;  // #2674
        l1k_cnt_10_state = l1k_cnt_9;  // #2675
        l1pv_cnt_9_state = l1pv_cnt_8;  // #2676
    }
    WAIT_VEC<PIPE_S>(0);  // #2249
    WAIT_VEC<PIPE_S>(0);  // #2249
    WAIT_VEC<PIPE_S>(2);  // #2249
    WAIT_VEC<PIPE_S>(2);  // #2249
    return;  // #2249
}
