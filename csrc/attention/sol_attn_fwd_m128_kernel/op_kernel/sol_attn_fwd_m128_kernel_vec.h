#pragma once
using namespace ascrip;

__aicore__ inline void sol_attn_fwd_m128_kernel_vec(GM_ADDR q_, GM_ADDR k_pad_, GM_ADDR v_pad_, GM_ADDR kc_padded_, GM_ADDR vc_padded_, GM_ADDR flat_exact_cnt_, GM_ADDR flat_exact_idx_, GM_ADDR group_approx_mask_, GM_ADDR last_group_lens_, GM_ADDR out_, GM_ADDR lse_, int32_t B, int32_t T, int32_t H, int32_t NT, int32_t NT_Q, int32_t NG, int32_t NPAD, int32_t T_PAD_KV, int32_t NT_Q_PAD8, int32_t T_LSE, float scale_log2, GM_ADDR workspace)
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
    ResetMask();
    SetMaskNorm();
    SEvent<PIPE_V, PIPE_MTE3, 0, 0> ev_v_mte3_ready_1;  // guards ub_p_0, ub_rmax_s #2772
    SEvent<PIPE_MTE3, PIPE_V, 1, 0> ev_mte3_v_valid_1;  // guards ub_p_0, ub_rmax_s #2773
    const float scale_natural = scale_log2 * 0.6931471805599453f;  // #2778
    const GMTensor<float> score_ws((__gm__ float*)(workspace + (0)));  // #2779
    const GMTensor<bfloat16_t> p_ws((__gm__ bfloat16_t*)(workspace + (18874368)));  // #2780
    const GMTensor<float> pv_ws((__gm__ float*)(workspace + (28311552)));  // #2781
    const Tensor<float, Position::UB> ub_score_p0((uint64_t)(0));  // #2789
    const Tensor<float, Position::UB> ub_score_p1((uint64_t)(32768));  // #2790
    const Tensor<float, Position::UB> ub_score_p2((uint64_t)(65536));  // #2791
    const Tensor<bfloat16_t, Position::UB> ub_p_0((uint64_t)(98304));  // #2792
    const Tensor<bfloat16_t, Position::UB> ub_p_1((uint64_t)(114688));  // #2793
    const Tensor<bfloat16_t, Position::UB> ub_p_2((uint64_t)(131072));  // #2794
    const Tensor<float, Position::UB> accum_ub((uint64_t)(147456));  // #2795
    const Tensor<float, Position::UB> ub_red_0((uint64_t)(180224));  // #2796
    const Tensor<float, Position::UB> ub_red_1((uint64_t)(182272));  // #2797
    const Tensor<float, Position::UB> ub_red_2((uint64_t)(184320));  // #2798
    const Tensor<float, Position::UB> ub_red_3((uint64_t)(186368));  // #2799
    const Tensor<float, Position::UB> ub_len_last((uint64_t)(188416));  // #2800
    const Tensor<float, Position::UB> ub_bmax_s((uint64_t)(188672));  // #2801
    const Tensor<float, Position::UB> ub_old_max_s((uint64_t)(188928));  // #2802
    const Tensor<float, Position::UB> ub_rmax_s((uint64_t)(189184));  // #2803
    const Tensor<float, Position::UB> ub_sum_s((uint64_t)(189440));  // #2804
    const Tensor<float, Position::UB> ub_bsum_s((uint64_t)(189696));  // #2805
    const Tensor<float, Position::UB> ub_rsum_s((uint64_t)(189952));  // #2806
    const Tensor<float, Position::UB> ub_zero_s((uint64_t)(190208));  // #2807
    const Buff<float, Position::UB, 4> expdiff_buf((uint64_t)(190464), (uint64_t)(256));  // #2808
    const Tensor<float, Position::UB> ub_max((uint64_t)(191488));  // #2809
    VEC_READY<PIPE_MTE2>(0);  // #2810
    VEC_READY<PIPE_MTE2>(0);  // #2810
    // sync.mutex vc id=1: the producer side; its 2 tokens are drained before the return  // #2811
    VEC_READY<PIPE_MTE2>(2);  // #2812
    VEC_READY<PIPE_MTE2>(2);  // #2812
    const int32_t cube_idx = GetCubeIdx();  // #2813
    const int32_t cube_count = GetCubeNum();  // #2814
    const int32_t core_step = (int32_t)(cube_count);  // #2815
    const int32_t sb = GetSubBlockIdx();  // #47
    const int32_t sb_row = sb * 64;  // #48
    const int32_t bh_total = B * H;  // #2816
    const int32_t total_tasks = bh_total * NT_Q;  // #2817
    const int32_t idiv_41 = NG / 4;  // #3481
    const int32_t idiv_42 = NG % 4;  // #3482
    const bool idiv_43 = idiv_42 != 0;  // #3483
    const bool idiv_47 = idiv_43 && true;  // #3487
    const int32_t idiv_48 = (idiv_47) ? (1) : (0);  // #3488
    const int32_t num_abatches = idiv_41 + idiv_48;  // #2818
    gm_to_ub_pad(ub_len_last, last_group_lens, 1, 256, 0, 0);  // #59
    vector_dup(ub_zero_s.ptr(), (float)(0.0f), (uint8_t)(1), (uint16_t)1, 1, (uint8_t)0, 0);  // #60
    const int32_t full_waves_lim = (NT_Q / core_step) * core_step;  // #2826 #2827
    int32_t l0ab_cnt_32_state = 0;  // #2828
    int32_t l0c_cnt_25_state = 0;  // #2829
    int32_t l1k_cnt_18_state = 0;  // #2830
    int32_t l1p_cnt_18_state = 0;  // #2831
    int32_t l1q_cnt_2_state = 0;  // #2832
    int32_t l1v_cnt_18_state = 0;  // #2833
    for (int32_t linear_id = cube_idx; linear_id < total_tasks; linear_id += core_step) {  // #3439
        const int32_t l0ab_cnt_carry = (int32_t)(l0ab_cnt_32_state);  // #2834
        const int32_t l0c_cnt_carry = (int32_t)(l0c_cnt_25_state);  // #2835
        const int32_t l1k_cnt_carry = (int32_t)(l1k_cnt_18_state);  // #2836
        const int32_t l1p_cnt_carry = (int32_t)(l1p_cnt_18_state);  // #2837
        const int32_t l1q_cnt_carry = (int32_t)(l1q_cnt_2_state);  // #2838
        const int32_t l1v_cnt_carry = (int32_t)(l1v_cnt_18_state);  // #2839
        const int32_t idiv_49 = linear_id / NT_Q;  // #3489
        const int32_t idiv_50 = linear_id % NT_Q;  // #3490
        const bool idiv_51 = idiv_50 != 0;  // #3491
        const bool idiv_52 = linear_id < 0;  // #3492
        const bool idiv_54 = idiv_52 != false;  // #3494
        const bool idiv_55 = idiv_51 && idiv_54;  // #3495
        const int32_t idiv_56 = (idiv_55) ? (1) : (0);  // #3496
        const int32_t bh = idiv_49 - idiv_56;  // #2840
        const int32_t r = ascrip::FloorMod<int32_t>(linear_id, NT_Q);  // #2841
        const bool cmp = r < full_waves_lim;  // #2842
        int32_t q_idx_1_state = 0;  // #2843
        if (cmp) {  // #2853
            const int32_t idiv_57 = r / core_step;  // #3497
            const int32_t idiv_58 = r % core_step;  // #3498
            const bool idiv_59 = idiv_58 != 0;  // #3499
            const bool idiv_60 = r < 0;  // #3500
            const bool idiv_62 = idiv_60 != false;  // #3502
            const bool idiv_63 = idiv_59 && idiv_62;  // #3503
            const int32_t idiv_64 = (idiv_63) ? (1) : (0);  // #3504
            const int32_t wave = idiv_57 - idiv_64;  // #2844
            const int32_t lane = ascrip::FloorMod<int32_t>(r, core_step);  // #2845
            q_idx_1_state = (wave * core_step) + ascrip::FloorMod<int32_t>((lane + wave) + bh, core_step);  // #2846 #2847 #2848 #2849 #2850 #2851
        } else {
            q_idx_1_state = r;  // #2852
        }
        const int32_t q_idx_1 = (int32_t)(q_idx_1_state);  // #2854
        const int32_t idiv_65 = bh / H;  // #3505
        const int32_t idiv_66 = bh % H;  // #3506
        const bool idiv_67 = idiv_66 != 0;  // #3507
        const bool idiv_68 = bh < 0;  // #3508
        const bool idiv_70 = idiv_68 != false;  // #3510
        const bool idiv_71 = idiv_67 && idiv_70;  // #3511
        const int32_t idiv_72 = (idiv_71) ? (1) : (0);  // #3512
        const int32_t b = idiv_65 - idiv_72;  // #2855
        const int32_t h = ascrip::FloorMod<int32_t>(bh, H);  // #2856
        const int32_t q_start = q_idx_1 * 128;  // #2857
        const int32_t valid_m = Min(128, T - q_start);  // #2858 #2859
        const int32_t local_valid_m = Min(64, Max(valid_m - sb_row, 0));  // #82 #83 #84
        if (valid_m < 128) {  // #2860 #2863
            PipeBarrier<PIPE_MTE2>();  // #2862
        }
        ev_mte3_v_valid_1.wait();  // #2774
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #95
        vector_dup(ub_rmax_s.ptr(), (float)(-1e+30f), (uint8_t)(1), (uint16_t)1, 1, (uint8_t)0, 0);  // #95
        vector_dup(ub_rsum_s.ptr(), (float)(0.0f), (uint8_t)(1), (uint16_t)1, 1, (uint8_t)0, 0);  // #96
        vector_dup(accum_ub.ptr(), (float)(0.0f), (uint8_t)(128), (uint16_t)1, 1, (uint8_t)8, 0);  // #97
        PipeBarrier<PIPE_ALL>();  // #2869
        SetFlag<PIPE_V, PIPE_MTE2>(4);  // #100
        SetFlag<PIPE_V, PIPE_MTE2>(5);  // #101
        SetFlag<PIPE_V, PIPE_MTE2>(6);  // #102
        SetFlag<PIPE_MTE3, PIPE_V>(4);  // #103
        SetFlag<PIPE_MTE3, PIPE_V>(5);  // #104
        SetFlag<PIPE_MTE3, PIPE_V>(6);  // #105
        const int32_t lag_a = Min(3, num_abatches);  // #2870
        const int32_t add_3 = num_abatches + lag_a;  // #2871
        int32_t l0ab_cnt_13_state = l0ab_cnt_carry;  // #2872
        int32_t l0c_cnt_10_state = l0c_cnt_carry;  // #2873
        int32_t l1k_cnt_7_state = l1k_cnt_carry;  // #2874
        int32_t l1p_cnt_7_state = l1p_cnt_carry;  // #2875
        int32_t l1v_cnt_7_state = l1v_cnt_carry;  // #2876
        for (int32_t ab = 0; ab < add_3; ab += 1) {  // #3186
            const int32_t l0ab_cnt_carry_1 = (int32_t)(l0ab_cnt_13_state);  // #2877
            const int32_t l0c_cnt_carry_1 = (int32_t)(l0c_cnt_10_state);  // #2878
            const int32_t l1k_cnt_carry_1 = (int32_t)(l1k_cnt_7_state);  // #2879
            const int32_t l1p_cnt_carry_1 = (int32_t)(l1p_cnt_7_state);  // #2880
            const int32_t l1v_cnt_carry_1 = (int32_t)(l1v_cnt_7_state);  // #2881
            const bool cmp_2 = ab < num_abatches;  // #2882
            int32_t l0ab_cnt_6_state = 0;  // #2883
            int32_t l0c_cnt_6_state = 0;  // #2884
            int32_t l1k_cnt_6_state = 0;  // #2885
            if (cmp_2) {  // #3022
                const int32_t slot_a_p = ab & 3;  // #2886
                const int32_t g_base_p = ab * 4;  // #2887
                const uint64_t m0 = group_approx_mask.load((((b * ((H * NT_Q) * NG)) + (h * (NT_Q * NG))) + (q_idx_1 * NG)) + g_base_p);  // #2889 #2890 #2891 #2892 #2893 #2894 #2895 #2896 #2897 #2898
                const int32_t add_7 = g_base_p + 1;  // #2899
                const bool cmp_3 = add_7 < NG;  // #2900
                uint64_t m1_1_state = 0ULL;  // #2901
                if (cmp_3) {  // #2916
                    m1_1_state = group_approx_mask.load((((b * ((H * NT_Q) * NG)) + (h * (NT_Q * NG))) + (q_idx_1 * NG)) + add_7);  // #2904 #2905 #2906 #2907 #2908 #2909 #2910 #2911 #2912 #2913 #2914
                } else {
                    m1_1_state = 0ULL;  // #2915
                }
                const uint64_t m1_1 = (uint64_t)(m1_1_state);  // #2917
                const int32_t add_12 = g_base_p + 2;  // #2918
                const bool cmp_4 = add_12 < NG;  // #2919
                uint64_t m2_1_state = 0ULL;  // #2920
                if (cmp_4) {  // #2935
                    m2_1_state = group_approx_mask.load((((b * ((H * NT_Q) * NG)) + (h * (NT_Q * NG))) + (q_idx_1 * NG)) + add_12);  // #2923 #2924 #2925 #2926 #2927 #2928 #2929 #2930 #2931 #2932 #2933
                } else {
                    m2_1_state = 0ULL;  // #2934
                }
                const uint64_t m2_1 = (uint64_t)(m2_1_state);  // #2936
                const int32_t add_17 = g_base_p + 3;  // #2937
                const bool cmp_5 = add_17 < NG;  // #2938
                uint64_t m3_1_state = 0ULL;  // #2939
                if (cmp_5) {  // #2954
                    m3_1_state = group_approx_mask.load((((b * ((H * NT_Q) * NG)) + (h * (NT_Q * NG))) + (q_idx_1 * NG)) + add_17);  // #2942 #2943 #2944 #2945 #2946 #2947 #2948 #2949 #2950 #2951 #2952
                } else {
                    m3_1_state = 0ULL;  // #2953
                }
                const uint64_t m3_1 = (uint64_t)(m3_1_state);  // #2955
                const bool has_p0_p = (m0 | m1_1) != 0ULL;  // #2956 #2957
                const bool has_p1_p = (m2_1 | m3_1) != 0ULL;  // #2958 #2959
                bool choice_state = false;  // #2960
                if (has_p0_p) {  // #2963
                    choice_state = true;  // #2961
                } else {
                    choice_state = has_p1_p;  // #2962
                }
                const bool choice = (bool)(choice_state);  // #2964
                int32_t l0ab_cnt_5_state = 0;  // #2965
                int32_t l0c_cnt_5_state = 0;  // #2966
                int32_t l1k_cnt_5_state = 0;  // #2967
                if (choice) {  // #3012
                    int32_t l0ab_cnt_2_state = 0;  // #2968
                    int32_t l0c_cnt_2_state = 0;  // #2969
                    int32_t l1k_cnt_2_state = 0;  // #2970
                    if (has_p0_p) {  // #2983
                        const int32_t l1k_cnt_1 = (int32_t)(l1k_cnt_carry_1 + 1);  // #2971 #2972
                        const int32_t l0ab_cnt_1 = (int32_t)(l0ab_cnt_carry_1 + 1);  // #2973 #2974
                        const int32_t l0c_cnt_1 = (int32_t)(l0c_cnt_carry_1 + 1);  // #2975 #2976
                        l0ab_cnt_2_state = l0ab_cnt_1;  // #2977
                        l0c_cnt_2_state = l0c_cnt_1;  // #2978
                        l1k_cnt_2_state = l1k_cnt_1;  // #2979
                    } else {
                        l0ab_cnt_2_state = l0ab_cnt_carry_1;  // #2980
                        l0c_cnt_2_state = l0c_cnt_carry_1;  // #2981
                        l1k_cnt_2_state = l1k_cnt_carry_1;  // #2982
                    }
                    const int32_t l0ab_cnt_2 = (int32_t)(l0ab_cnt_2_state);  // #2984
                    const int32_t l0c_cnt_2 = (int32_t)(l0c_cnt_2_state);  // #2985
                    const int32_t l1k_cnt_2 = (int32_t)(l1k_cnt_2_state);  // #2986
                    int32_t l0ab_cnt_4_state = 0;  // #2987
                    int32_t l0c_cnt_4_state = 0;  // #2988
                    int32_t l1k_cnt_4_state = 0;  // #2989
                    if (has_p1_p) {  // #3002
                        const int32_t l1k_cnt_3 = (int32_t)(l1k_cnt_2 + 1);  // #2990 #2991
                        const int32_t l0ab_cnt_3 = (int32_t)(l0ab_cnt_2 + 1);  // #2992 #2993
                        const int32_t l0c_cnt_3 = (int32_t)(l0c_cnt_2 + 1);  // #2994 #2995
                        l0ab_cnt_4_state = l0ab_cnt_3;  // #2996
                        l0c_cnt_4_state = l0c_cnt_3;  // #2997
                        l1k_cnt_4_state = l1k_cnt_3;  // #2998
                    } else {
                        l0ab_cnt_4_state = l0ab_cnt_2;  // #2999
                        l0c_cnt_4_state = l0c_cnt_2;  // #3000
                        l1k_cnt_4_state = l1k_cnt_2;  // #3001
                    }
                    const int32_t l0ab_cnt_4 = (int32_t)(l0ab_cnt_4_state);  // #3003
                    const int32_t l0c_cnt_4 = (int32_t)(l0c_cnt_4_state);  // #3004
                    const int32_t l1k_cnt_4 = (int32_t)(l1k_cnt_4_state);  // #3005
                    WAIT_CUBE<PIPE_S>(0);  // #290
                    if (has_p0_p) {  // #297
                        WaitFlag<PIPE_V, PIPE_MTE2>(4);  // #291
                        const GMTensor<float> view_9 = score_ws[(((cube_idx * 196608) + (slot_a_p * 49152)) + (sb_row * 128))];  // #294
                        gm_to_ub_pad(ub_score_p0, view_9, 64, 512, 0, 0);  // #295
                        SetFlag<PIPE_MTE2, PIPE_V>(4);  // #296
                    }
                    if (has_p1_p) {  // #306
                        WaitFlag<PIPE_V, PIPE_MTE2>(5);  // #298
                        const GMTensor<float> view_10 = score_ws[(((cube_idx * 196608) + (slot_a_p * 49152)) + ((128 + sb_row) * 128))];  // #299 #303
                        gm_to_ub_pad(ub_score_p1, view_10, 64, 512, 0, 0);  // #304
                        SetFlag<PIPE_MTE2, PIPE_V>(5);  // #305
                    }
                    if (has_p0_p) {  // #309
                        WaitFlag<PIPE_MTE2, PIPE_V>(4);  // #307
                        PipeBarrier<PIPE_V>();  // c220: V-V hazard #308
                        vmuls(ub_score_p0.ptr(), ub_score_p0.ptr(), (float)(scale_natural), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #308
                    }
                    if (has_p1_p) {  // #312
                        WaitFlag<PIPE_MTE2, PIPE_V>(5);  // #310
                        PipeBarrier<PIPE_V>();  // c220: V-V hazard #311
                        vmuls(ub_score_p1.ptr(), ub_score_p1.ptr(), (float)(scale_natural), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #311
                    }
                    PipeBarrier<PIPE_V>();  // #313
                    if (has_p0_p) {  // #329
                        const uint64_t nm0 = ~(m0);  // #314
                        if (nm0 != 0ULL) {  // #315 #320
                            SetVectorMask((uint64_t)(0), (uint64_t)(nm0));  // #316
                            const Tensor<float, Position::UB> view_11 = ub_score_p0;  // #317
                            vector_dup(view_11.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #318
                            ResetMask();  // #319
                        }
                        const uint64_t nm1 = ~(m1_1);  // #321
                        if (nm1 != 0ULL) {  // #322 #328
                            PipeBarrier<PIPE_V>();  // #323
                            SetVectorMask((uint64_t)(0), (uint64_t)(nm1));  // #324
                            const Tensor<float, Position::UB> view_12 = ub_score_p0[64];  // #325
                            vector_dup(view_12.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #326
                            ResetMask();  // #327
                        }
                    }
                    if (has_p1_p) {  // #345
                        const uint64_t nm2 = ~(m2_1);  // #330
                        if (nm2 != 0ULL) {  // #331 #336
                            SetVectorMask((uint64_t)(0), (uint64_t)(nm2));  // #332
                            const Tensor<float, Position::UB> view_13 = ub_score_p1;  // #333
                            vector_dup(view_13.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #334
                            ResetMask();  // #335
                        }
                        const uint64_t nm3 = ~(m3_1);  // #337
                        if (nm3 != 0ULL) {  // #338 #344
                            PipeBarrier<PIPE_V>();  // #339
                            SetVectorMask((uint64_t)(0), (uint64_t)(nm3));  // #340
                            const Tensor<float, Position::UB> view_14 = ub_score_p1[64];  // #341
                            vector_dup(view_14.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #342
                            ResetMask();  // #343
                        }
                    }
                    PipeBarrier<PIPE_V>();  // #346
                    bool choice_1_state = false;  // #2496
                    if (has_p0_p) {  // #349
                        choice_1_state = has_p1_p;  // #2497
                    } else {
                        choice_1_state = false;  // #2498
                    }
                    const bool choice_1 = (bool)(choice_1_state);  // #2499
                    if (choice_1) {  // #382
                        const Tensor<float, Position::UB> view_15 = ub_score_p0;  // #350
                        vcgmax(ub_red_0.ptr(), view_15.ptr(), (uint8_t)(64), 1, 1, 16);  // #351
                        const Tensor<float, Position::UB> view_16 = ub_score_p0[64];  // #352
                        vcgmax(ub_red_1.ptr(), view_16.ptr(), (uint8_t)(64), 1, 1, 16);  // #353
                        const Tensor<float, Position::UB> view_17 = ub_score_p1;  // #354
                        vcgmax(ub_red_2.ptr(), view_17.ptr(), (uint8_t)(64), 1, 1, 16);  // #355
                        const Tensor<float, Position::UB> view_18 = ub_score_p1[64];  // #356
                        vcgmax(ub_red_3.ptr(), view_18.ptr(), (uint8_t)(64), 1, 1, 16);  // #357
                        PipeBarrier<PIPE_V>();  // #358
                        vmax(ub_red_0.ptr(), ub_red_0.ptr(), ub_red_1.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #359
                        vmax(ub_red_2.ptr(), ub_red_2.ptr(), ub_red_3.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #360
                        PipeBarrier<PIPE_V>();  // #361
                        vmax(ub_red_0.ptr(), ub_red_0.ptr(), ub_red_2.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #362
                        PipeBarrier<PIPE_V>();  // #363
                        vcgmax(ub_bmax_s.ptr(), ub_red_0.ptr(), (uint8_t)(8), 1, 1, 8);  // #364
                    } else {
                        if (has_p0_p) {  // #381
                            const Tensor<float, Position::UB> view_19 = ub_score_p0;  // #365
                            vcgmax(ub_red_0.ptr(), view_19.ptr(), (uint8_t)(64), 1, 1, 16);  // #366
                            const Tensor<float, Position::UB> view_20 = ub_score_p0[64];  // #367
                            vcgmax(ub_red_1.ptr(), view_20.ptr(), (uint8_t)(64), 1, 1, 16);  // #368
                            PipeBarrier<PIPE_V>();  // #369
                            vmax(ub_red_0.ptr(), ub_red_0.ptr(), ub_red_1.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #370
                            PipeBarrier<PIPE_V>();  // #371
                            vcgmax(ub_bmax_s.ptr(), ub_red_0.ptr(), (uint8_t)(8), 1, 1, 8);  // #372
                        } else {
                            const Tensor<float, Position::UB> view_21 = ub_score_p1;  // #373
                            vcgmax(ub_red_0.ptr(), view_21.ptr(), (uint8_t)(64), 1, 1, 16);  // #374
                            const Tensor<float, Position::UB> view_22 = ub_score_p1[64];  // #375
                            vcgmax(ub_red_1.ptr(), view_22.ptr(), (uint8_t)(64), 1, 1, 16);  // #376
                            PipeBarrier<PIPE_V>();  // #377
                            vmax(ub_red_0.ptr(), ub_red_0.ptr(), ub_red_1.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #378
                            PipeBarrier<PIPE_V>();  // #379
                            vcgmax(ub_bmax_s.ptr(), ub_red_0.ptr(), (uint8_t)(8), 1, 1, 8);  // #380
                        }
                    }
                    vadd(ub_old_max_s.ptr(), ub_rmax_s.ptr(), ub_zero_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #383
                    PipeBarrier<PIPE_V>();  // #384
                    const Tensor<float, Position::UB> row_diff_a = expdiff_buf.get(ab);  // #385
                    vmax(ub_rmax_s.ptr(), ub_rmax_s.ptr(), ub_bmax_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #386
                    PipeBarrier<PIPE_V>();  // #387
                    vsub(row_diff_a.ptr(), ub_old_max_s.ptr(), ub_rmax_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #388
                    ResetMask();  // #389
                    vbrcb((__ubuf__ uint32_t*)(ub_max.ptr()), (__ubuf__ uint32_t*)(ub_rmax_s.ptr()), (uint16_t)1, (uint16_t)8, (uint8_t)(8));  // #389
                    PipeBarrier<PIPE_V>();  // #390
                    vexp(row_diff_a.ptr(), row_diff_a.ptr(), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint8_t)0, (uint8_t)0);  // #391
                    if (has_p0_p) {  // #395
                        const Tensor<float, Position::UB> view_23 = ub_score_p0;  // #392
                        const Tensor<float, Position::UB> view_24 = ub_score_p0;  // #393
                        vsub(view_23.ptr(), view_24.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #394
                    }
                    if (has_p1_p) {  // #399
                        const Tensor<float, Position::UB> view_25 = ub_score_p1;  // #396
                        const Tensor<float, Position::UB> view_26 = ub_score_p1;  // #397
                        vsub(view_25.ptr(), view_26.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #398
                    }
                    PipeBarrier<PIPE_V>();  // #400
                    if (has_p0_p) {  // #404
                        const Tensor<float, Position::UB> view_27 = ub_score_p0[64];  // #401
                        const Tensor<float, Position::UB> view_28 = ub_score_p0[64];  // #402
                        vsub(view_27.ptr(), view_28.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #403
                    }
                    if (has_p1_p) {  // #408
                        const Tensor<float, Position::UB> view_29 = ub_score_p1[64];  // #405
                        const Tensor<float, Position::UB> view_30 = ub_score_p1[64];  // #406
                        vsub(view_29.ptr(), view_30.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #407
                    }
                    PipeBarrier<PIPE_V>();  // #409
                    if (has_p0_p) {  // #411
                        vexp(ub_score_p0.ptr(), ub_score_p0.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint8_t)8, (uint8_t)8);  // #410
                    }
                    if (has_p1_p) {  // #413
                        vexp(ub_score_p1.ptr(), ub_score_p1.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint8_t)8, (uint8_t)8);  // #412
                    }
                    PipeBarrier<PIPE_V>();  // #414
                    if (has_p0_p) {  // #424
                        WaitFlag<PIPE_MTE3, PIPE_V>(4);  // #415
                        vconv_f322bf16a(ub_p_0.ptr(), ub_score_p0.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)4, (uint16_t)8);  // #416
                        SetFlag<PIPE_V, PIPE_MTE3>(4);  // #417
                        WaitFlag<PIPE_V, PIPE_MTE3>(4);  // #418
                        const GMTensor<bfloat16_t> view_31 = p_ws[(((cube_idx * 196608) + (slot_a_p * 49152)) + (sb_row * 128))];  // #421
                        ub_to_gm_pad(view_31, ub_p_0, 64, 256, 0, 0);  // #422
                        SetFlag<PIPE_MTE3, PIPE_V>(4);  // #423
                    }
                    if (has_p1_p) {  // #436
                        WaitFlag<PIPE_MTE3, PIPE_V>(5);  // #425
                        vconv_f322bf16a(ub_p_1.ptr(), ub_score_p1.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)4, (uint16_t)8);  // #426
                        SetFlag<PIPE_V, PIPE_MTE3>(5);  // #427
                        WaitFlag<PIPE_V, PIPE_MTE3>(5);  // #428
                        const GMTensor<bfloat16_t> view_32 = p_ws[(((cube_idx * 196608) + (slot_a_p * 49152)) + ((128 + sb_row) * 128))];  // #429 #433
                        ub_to_gm_pad(view_32, ub_p_1, 64, 256, 0, 0);  // #434
                        SetFlag<PIPE_MTE3, PIPE_V>(5);  // #435
                    }
                    const int32_t sub_11 = num_abatches + -1;  // #437
                    if (ab < sub_11) {  // #438 #556
                        bool choice_2_state = false;  // #2500
                        if (has_p0_p) {  // #441
                            choice_2_state = has_p1_p;  // #2501
                        } else {
                            choice_2_state = false;  // #2502
                        }
                        const bool choice_2 = (bool)(choice_2_state);  // #2503
                        if (choice_2) {  // #481
                            const Tensor<float, Position::UB> view_33 = ub_score_p0;  // #442
                            vcgadd(ub_red_0.ptr(), view_33.ptr(), (uint8_t)(64), 1, 1, 16);  // #443
                            const Tensor<float, Position::UB> view_34 = ub_score_p0[64];  // #444
                            vcgadd(ub_red_1.ptr(), view_34.ptr(), (uint8_t)(64), 1, 1, 16);  // #445
                            SetFlag<PIPE_V, PIPE_MTE2>(4);  // #446
                            const Tensor<float, Position::UB> view_35 = ub_score_p1;  // #447
                            vcgadd(ub_red_2.ptr(), view_35.ptr(), (uint8_t)(64), 1, 1, 16);  // #448
                            const Tensor<float, Position::UB> view_36 = ub_score_p1[64];  // #449
                            vcgadd(ub_red_3.ptr(), view_36.ptr(), (uint8_t)(64), 1, 1, 16);  // #450
                            SetFlag<PIPE_V, PIPE_MTE2>(5);  // #451
                            vmul(ub_rsum_s.ptr(), ub_rsum_s.ptr(), row_diff_a.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #452
                            PipeBarrier<PIPE_V>();  // #453
                            vadd(ub_red_0.ptr(), ub_red_0.ptr(), ub_red_1.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #454
                            vadd(ub_red_2.ptr(), ub_red_2.ptr(), ub_red_3.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #455
                            PipeBarrier<PIPE_V>();  // #456
                            vadd(ub_red_0.ptr(), ub_red_0.ptr(), ub_red_2.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #457
                            PipeBarrier<PIPE_V>();  // #458
                            vcgadd(ub_bsum_s.ptr(), ub_red_0.ptr(), (uint8_t)(8), 1, 1, 8);  // #459
                        } else {
                            if (has_p0_p) {  // #480
                                const Tensor<float, Position::UB> view_37 = ub_score_p0;  // #460
                                vcgadd(ub_red_0.ptr(), view_37.ptr(), (uint8_t)(64), 1, 1, 16);  // #461
                                const Tensor<float, Position::UB> view_38 = ub_score_p0[64];  // #462
                                vcgadd(ub_red_1.ptr(), view_38.ptr(), (uint8_t)(64), 1, 1, 16);  // #463
                                SetFlag<PIPE_V, PIPE_MTE2>(4);  // #464
                                vmul(ub_rsum_s.ptr(), ub_rsum_s.ptr(), row_diff_a.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #465
                                PipeBarrier<PIPE_V>();  // #466
                                vadd(ub_red_0.ptr(), ub_red_0.ptr(), ub_red_1.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #467
                                PipeBarrier<PIPE_V>();  // #468
                                vcgadd(ub_bsum_s.ptr(), ub_red_0.ptr(), (uint8_t)(8), 1, 1, 8);  // #469
                            } else {
                                const Tensor<float, Position::UB> view_39 = ub_score_p1;  // #470
                                vcgadd(ub_red_0.ptr(), view_39.ptr(), (uint8_t)(64), 1, 1, 16);  // #471
                                const Tensor<float, Position::UB> view_40 = ub_score_p1[64];  // #472
                                vcgadd(ub_red_1.ptr(), view_40.ptr(), (uint8_t)(64), 1, 1, 16);  // #473
                                SetFlag<PIPE_V, PIPE_MTE2>(5);  // #474
                                vmul(ub_rsum_s.ptr(), ub_rsum_s.ptr(), row_diff_a.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #475
                                PipeBarrier<PIPE_V>();  // #476
                                vadd(ub_red_0.ptr(), ub_red_0.ptr(), ub_red_1.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #477
                                PipeBarrier<PIPE_V>();  // #478
                                vcgadd(ub_bsum_s.ptr(), ub_red_0.ptr(), (uint8_t)(8), 1, 1, 8);  // #479
                            }
                        }
                        PipeBarrier<PIPE_V>();  // #482
                        vmuls(ub_bsum_s.ptr(), ub_bsum_s.ptr(), (float)(64.0f), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #483
                    } else {
                        PipeBarrier<PIPE_V>();  // #484
                        const int32_t sub_12 = NG + -1;  // #485
                        const int32_t last_u = sub_12 - g_base_p;  // #486
                        if (last_u == 0) {  // #487 #508
                            if (has_p0_p) {  // #491
                                const Tensor<float, Position::UB> view_41 = ub_score_p0;  // #488
                                const Tensor<float, Position::UB> view_42 = ub_score_p0;  // #489
                                vmul(view_41.ptr(), view_42.ptr(), ub_len_last.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)16, (uint8_t)16, (uint8_t)0);  // #490
                            }
                        } else {
                            if (last_u == 1) {  // #492 #507
                                if (has_p0_p) {  // #496
                                    const Tensor<float, Position::UB> view_43 = ub_score_p0[64];  // #493
                                    const Tensor<float, Position::UB> view_44 = ub_score_p0[64];  // #494
                                    vmul(view_43.ptr(), view_44.ptr(), ub_len_last.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)16, (uint8_t)16, (uint8_t)0);  // #495
                                }
                            } else {
                                if (last_u == 2) {  // #497 #506
                                    if (has_p1_p) {  // #501
                                        const Tensor<float, Position::UB> view_45 = ub_score_p1;  // #498
                                        const Tensor<float, Position::UB> view_46 = ub_score_p1;  // #499
                                        vmul(view_45.ptr(), view_46.ptr(), ub_len_last.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)16, (uint8_t)16, (uint8_t)0);  // #500
                                    }
                                } else {
                                    if (has_p1_p) {  // #505
                                        const Tensor<float, Position::UB> view_47 = ub_score_p1[64];  // #502
                                        const Tensor<float, Position::UB> view_48 = ub_score_p1[64];  // #503
                                        vmul(view_47.ptr(), view_48.ptr(), ub_len_last.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)16, (uint8_t)16, (uint8_t)0);  // #504
                                    }
                                }
                            }
                        }
                        PipeBarrier<PIPE_V>();  // #509
                        if (has_p0_p) {  // #515
                            const Tensor<float, Position::UB> view_49 = ub_score_p0;  // #510
                            vcgadd(ub_red_0.ptr(), view_49.ptr(), (uint8_t)(64), 1, 1, 16);  // #511
                            const Tensor<float, Position::UB> view_50 = ub_score_p0[64];  // #512
                            vcgadd(ub_red_1.ptr(), view_50.ptr(), (uint8_t)(64), 1, 1, 16);  // #513
                            SetFlag<PIPE_V, PIPE_MTE2>(4);  // #514
                        }
                        if (has_p1_p) {  // #521
                            const Tensor<float, Position::UB> view_51 = ub_score_p1;  // #516
                            vcgadd(ub_red_2.ptr(), view_51.ptr(), (uint8_t)(64), 1, 1, 16);  // #517
                            const Tensor<float, Position::UB> view_52 = ub_score_p1[64];  // #518
                            vcgadd(ub_red_3.ptr(), view_52.ptr(), (uint8_t)(64), 1, 1, 16);  // #519
                            SetFlag<PIPE_V, PIPE_MTE2>(5);  // #520
                        }
                        vmul(ub_rsum_s.ptr(), ub_rsum_s.ptr(), row_diff_a.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #522
                        PipeBarrier<PIPE_V>();  // #523
                        if (has_p0_p) {  // #530
                            if (last_u != 0) {  // #524 #526
                                vmuls(ub_red_0.ptr(), ub_red_0.ptr(), (float)(64.0f), (uint8_t)(8), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #525
                            }
                            if (last_u != 1) {  // #527 #529
                                vmuls(ub_red_1.ptr(), ub_red_1.ptr(), (float)(64.0f), (uint8_t)(8), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #528
                            }
                        }
                        if (has_p1_p) {  // #537
                            if (last_u != 2) {  // #531 #533
                                vmuls(ub_red_2.ptr(), ub_red_2.ptr(), (float)(64.0f), (uint8_t)(8), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #532
                            }
                            if (last_u != 3) {  // #534 #536
                                vmuls(ub_red_3.ptr(), ub_red_3.ptr(), (float)(64.0f), (uint8_t)(8), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #535
                            }
                        }
                        PipeBarrier<PIPE_V>();  // #538
                        bool choice_3_state = false;  // #2504
                        if (has_p0_p) {  // #541
                            choice_3_state = has_p1_p;  // #2505
                        } else {
                            choice_3_state = false;  // #2506
                        }
                        const bool choice_3 = (bool)(choice_3_state);  // #2507
                        if (choice_3) {  // #555
                            vadd(ub_red_0.ptr(), ub_red_0.ptr(), ub_red_1.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #542
                            vadd(ub_red_2.ptr(), ub_red_2.ptr(), ub_red_3.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #543
                            PipeBarrier<PIPE_V>();  // #544
                            vadd(ub_red_0.ptr(), ub_red_0.ptr(), ub_red_2.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #545
                            PipeBarrier<PIPE_V>();  // #546
                            vcgadd(ub_bsum_s.ptr(), ub_red_0.ptr(), (uint8_t)(8), 1, 1, 8);  // #547
                        } else {
                            if (has_p0_p) {  // #554
                                vadd(ub_red_0.ptr(), ub_red_0.ptr(), ub_red_1.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #548
                                PipeBarrier<PIPE_V>();  // #549
                                vcgadd(ub_bsum_s.ptr(), ub_red_0.ptr(), (uint8_t)(8), 1, 1, 8);  // #550
                            } else {
                                vadd(ub_red_2.ptr(), ub_red_2.ptr(), ub_red_3.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #551
                                PipeBarrier<PIPE_V>();  // #552
                                vcgadd(ub_bsum_s.ptr(), ub_red_2.ptr(), (uint8_t)(8), 1, 1, 8);  // #553
                            }
                        }
                    }
                    PipeBarrier<PIPE_V>();  // #557
                    vadd(ub_rsum_s.ptr(), ub_rsum_s.ptr(), ub_bsum_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #558
                    VEC_READY<PIPE_MTE3>(1);  // #559
                    l0ab_cnt_5_state = l0ab_cnt_4;  // #3006
                    l0c_cnt_5_state = l0c_cnt_4;  // #3007
                    l1k_cnt_5_state = l1k_cnt_4;  // #3008
                } else {
                    l0ab_cnt_5_state = l0ab_cnt_carry_1;  // #3009
                    l0c_cnt_5_state = l0c_cnt_carry_1;  // #3010
                    l1k_cnt_5_state = l1k_cnt_carry_1;  // #3011
                }
                const int32_t l0ab_cnt_5 = (int32_t)(l0ab_cnt_5_state);  // #3013
                const int32_t l0c_cnt_5 = (int32_t)(l0c_cnt_5_state);  // #3014
                const int32_t l1k_cnt_5 = (int32_t)(l1k_cnt_5_state);  // #3015
                l0ab_cnt_6_state = l0ab_cnt_5;  // #3016
                l0c_cnt_6_state = l0c_cnt_5;  // #3017
                l1k_cnt_6_state = l1k_cnt_5;  // #3018
            } else {
                l0ab_cnt_6_state = l0ab_cnt_carry_1;  // #3019
                l0c_cnt_6_state = l0c_cnt_carry_1;  // #3020
                l1k_cnt_6_state = l1k_cnt_carry_1;  // #3021
            }
            const int32_t l0ab_cnt_6 = (int32_t)(l0ab_cnt_6_state);  // #3023
            const int32_t l0c_cnt_6 = (int32_t)(l0c_cnt_6_state);  // #3024
            const int32_t l1k_cnt_6 = (int32_t)(l1k_cnt_6_state);  // #3025
            const bool cmp_20 = ab >= lag_a;  // #3026
            int32_t l0ab_cnt_12_state = 0;  // #3027
            int32_t l0c_cnt_9_state = 0;  // #3028
            int32_t l1p_cnt_6_state = 0;  // #3029
            int32_t l1v_cnt_6_state = 0;  // #3030
            if (cmp_20) {  // #3176
                const int32_t ac = ab - lag_a;  // #3031
                const int32_t slot_a_c = ac & 3;  // #3032
                const int32_t gc_base = ac * 4;  // #3033
                const uint64_t mc0 = group_approx_mask.load((((b * ((H * NT_Q) * NG)) + (h * (NT_Q * NG))) + (q_idx_1 * NG)) + gc_base);  // #3035 #3036 #3037 #3038 #3039 #3040 #3041 #3042 #3043 #3044
                const int32_t add_46 = gc_base + 1;  // #3045
                const bool cmp_21 = add_46 < NG;  // #3046
                uint64_t mc1_1_state = 0ULL;  // #3047
                if (cmp_21) {  // #3062
                    mc1_1_state = group_approx_mask.load((((b * ((H * NT_Q) * NG)) + (h * (NT_Q * NG))) + (q_idx_1 * NG)) + add_46);  // #3050 #3051 #3052 #3053 #3054 #3055 #3056 #3057 #3058 #3059 #3060
                } else {
                    mc1_1_state = 0ULL;  // #3061
                }
                const uint64_t mc1_1 = (uint64_t)(mc1_1_state);  // #3063
                const int32_t add_51 = gc_base + 2;  // #3064
                const bool cmp_22 = add_51 < NG;  // #3065
                uint64_t mc2_1_state = 0ULL;  // #3066
                if (cmp_22) {  // #3081
                    mc2_1_state = group_approx_mask.load((((b * ((H * NT_Q) * NG)) + (h * (NT_Q * NG))) + (q_idx_1 * NG)) + add_51);  // #3069 #3070 #3071 #3072 #3073 #3074 #3075 #3076 #3077 #3078 #3079
                } else {
                    mc2_1_state = 0ULL;  // #3080
                }
                const uint64_t mc2_1 = (uint64_t)(mc2_1_state);  // #3082
                const int32_t add_56 = gc_base + 3;  // #3083
                const bool cmp_23 = add_56 < NG;  // #3084
                uint64_t mc3_1_state = 0ULL;  // #3085
                if (cmp_23) {  // #3100
                    mc3_1_state = group_approx_mask.load((((b * ((H * NT_Q) * NG)) + (h * (NT_Q * NG))) + (q_idx_1 * NG)) + add_56);  // #3088 #3089 #3090 #3091 #3092 #3093 #3094 #3095 #3096 #3097 #3098
                } else {
                    mc3_1_state = 0ULL;  // #3099
                }
                const uint64_t mc3_1 = (uint64_t)(mc3_1_state);  // #3101
                const bool has_p0_c = (mc0 | mc1_1) != 0ULL;  // #3102 #3103
                const bool has_p1_c = (mc2_1 | mc3_1) != 0ULL;  // #3104 #3105
                bool choice_4_state = false;  // #3106
                if (has_p0_c) {  // #3109
                    choice_4_state = true;  // #3107
                } else {
                    choice_4_state = has_p1_c;  // #3108
                }
                const bool choice_4 = (bool)(choice_4_state);  // #3110
                int32_t l0ab_cnt_11_state = 0;  // #3111
                int32_t l0c_cnt_8_state = 0;  // #3112
                int32_t l1p_cnt_5_state = 0;  // #3113
                int32_t l1v_cnt_5_state = 0;  // #3114
                if (choice_4) {  // #3163
                    int32_t l0ab_cnt_8_state = 0;  // #3115
                    int32_t l1p_cnt_2_state = 0;  // #3116
                    int32_t l1v_cnt_2_state = 0;  // #3117
                    if (has_p0_c) {  // #3130
                        const int32_t l0ab_cnt_7 = (int32_t)(l0ab_cnt_6 + 1);  // #3118 #3119
                        const int32_t l1p_cnt_1 = (int32_t)(l1p_cnt_carry_1 + 1);  // #3120 #3121
                        const int32_t l1v_cnt_1 = (int32_t)(l1v_cnt_carry_1 + 1);  // #3122 #3123
                        l0ab_cnt_8_state = l0ab_cnt_7;  // #3124
                        l1p_cnt_2_state = l1p_cnt_1;  // #3125
                        l1v_cnt_2_state = l1v_cnt_1;  // #3126
                    } else {
                        l0ab_cnt_8_state = l0ab_cnt_6;  // #3127
                        l1p_cnt_2_state = l1p_cnt_carry_1;  // #3128
                        l1v_cnt_2_state = l1v_cnt_carry_1;  // #3129
                    }
                    const int32_t l0ab_cnt_8 = (int32_t)(l0ab_cnt_8_state);  // #3131
                    const int32_t l1p_cnt_2 = (int32_t)(l1p_cnt_2_state);  // #3132
                    const int32_t l1v_cnt_2 = (int32_t)(l1v_cnt_2_state);  // #3133
                    int32_t l0ab_cnt_10_state = 0;  // #3134
                    int32_t l1p_cnt_4_state = 0;  // #3135
                    int32_t l1v_cnt_4_state = 0;  // #3136
                    if (has_p1_c) {  // #3149
                        const int32_t l0ab_cnt_9 = (int32_t)(l0ab_cnt_8 + 1);  // #3137 #3138
                        const int32_t l1p_cnt_3 = (int32_t)(l1p_cnt_2 + 1);  // #3139 #3140
                        const int32_t l1v_cnt_3 = (int32_t)(l1v_cnt_2 + 1);  // #3141 #3142
                        l0ab_cnt_10_state = l0ab_cnt_9;  // #3143
                        l1p_cnt_4_state = l1p_cnt_3;  // #3144
                        l1v_cnt_4_state = l1v_cnt_3;  // #3145
                    } else {
                        l0ab_cnt_10_state = l0ab_cnt_8;  // #3146
                        l1p_cnt_4_state = l1p_cnt_2;  // #3147
                        l1v_cnt_4_state = l1v_cnt_2;  // #3148
                    }
                    const int32_t l0ab_cnt_10 = (int32_t)(l0ab_cnt_10_state);  // #3150
                    const int32_t l1p_cnt_4 = (int32_t)(l1p_cnt_4_state);  // #3151
                    const int32_t l1v_cnt_4 = (int32_t)(l1v_cnt_4_state);  // #3152
                    const int32_t l0c_cnt_7 = (int32_t)(l0c_cnt_6 + 1);  // #3153 #3154
                    WAIT_CUBE<PIPE_S>(2);  // #758
                    WaitFlag<PIPE_V, PIPE_MTE2>(6);  // #759
                    const GMTensor<float> view_62 = pv_ws[(((cube_idx * 65536) + (slot_a_c * 16384)) + (sb_row * 128))];  // #762
                    gm_to_ub_pad(ub_score_p2, view_62, 64, 512, 0, 0);  // #763
                    SetFlag<PIPE_MTE2, PIPE_V>(6);  // #764
                    const Tensor<float, Position::UB> slot_50 = expdiff_buf.get(ac);  // #765
                    PipeBarrier<PIPE_V>();  // c220: V-V hazard #766
                    ResetMask();  // #766
                    vbrcb((__ubuf__ uint32_t*)(ub_max.ptr()), (__ubuf__ uint32_t*)(slot_50.ptr()), (uint16_t)1, (uint16_t)8, (uint8_t)(8));  // #766
                    PipeBarrier<PIPE_V>();  // #767
                    const Tensor<float, Position::UB> view_63 = accum_ub;  // #768
                    const Tensor<float, Position::UB> view_64 = accum_ub;  // #769
                    vmul(view_63.ptr(), view_64.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #770
                    const Tensor<float, Position::UB> view_65 = accum_ub[64];  // #771
                    const Tensor<float, Position::UB> view_66 = accum_ub[64];  // #772
                    PipeBarrier<PIPE_V>();  // c220: V-V hazard #773
                    vmul(view_65.ptr(), view_66.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #773
                    PipeBarrier<PIPE_V>();  // #774
                    WaitFlag<PIPE_MTE2, PIPE_V>(6);  // #775
                    vadd(accum_ub.ptr(), accum_ub.ptr(), ub_score_p2.ptr(), (uint8_t)(128), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #776
                    SetFlag<PIPE_V, PIPE_MTE2>(6);  // #777
                    l0ab_cnt_11_state = l0ab_cnt_10;  // #3155
                    l0c_cnt_8_state = l0c_cnt_7;  // #3156
                    l1p_cnt_5_state = l1p_cnt_4;  // #3157
                    l1v_cnt_5_state = l1v_cnt_4;  // #3158
                } else {
                    l0ab_cnt_11_state = l0ab_cnt_6;  // #3159
                    l0c_cnt_8_state = l0c_cnt_6;  // #3160
                    l1p_cnt_5_state = l1p_cnt_carry_1;  // #3161
                    l1v_cnt_5_state = l1v_cnt_carry_1;  // #3162
                }
                const int32_t l0ab_cnt_11 = (int32_t)(l0ab_cnt_11_state);  // #3164
                const int32_t l0c_cnt_8 = (int32_t)(l0c_cnt_8_state);  // #3165
                const int32_t l1p_cnt_5 = (int32_t)(l1p_cnt_5_state);  // #3166
                const int32_t l1v_cnt_5 = (int32_t)(l1v_cnt_5_state);  // #3167
                l0ab_cnt_12_state = l0ab_cnt_11;  // #3168
                l0c_cnt_9_state = l0c_cnt_8;  // #3169
                l1p_cnt_6_state = l1p_cnt_5;  // #3170
                l1v_cnt_6_state = l1v_cnt_5;  // #3171
            } else {
                l0ab_cnt_12_state = l0ab_cnt_6;  // #3172
                l0c_cnt_9_state = l0c_cnt_6;  // #3173
                l1p_cnt_6_state = l1p_cnt_carry_1;  // #3174
                l1v_cnt_6_state = l1v_cnt_carry_1;  // #3175
            }
            const int32_t l0ab_cnt_12 = (int32_t)(l0ab_cnt_12_state);  // #3177
            const int32_t l0c_cnt_9 = (int32_t)(l0c_cnt_9_state);  // #3178
            const int32_t l1p_cnt_6 = (int32_t)(l1p_cnt_6_state);  // #3179
            const int32_t l1v_cnt_6 = (int32_t)(l1v_cnt_6_state);  // #3180
            l0ab_cnt_13_state = l0ab_cnt_12;  // #3181
            l0c_cnt_10_state = l0c_cnt_9;  // #3182
            l1k_cnt_7_state = l1k_cnt_6;  // #3183
            l1p_cnt_7_state = l1p_cnt_6;  // #3184
            l1v_cnt_7_state = l1v_cnt_6;  // #3185
        }
        const int32_t l0ab_cnt_13 = (int32_t)(l0ab_cnt_13_state);  // #3187
        const int32_t l0c_cnt_10 = (int32_t)(l0c_cnt_10_state);  // #3188
        const int32_t l1k_cnt_7 = (int32_t)(l1k_cnt_7_state);  // #3189
        const int32_t l1p_cnt_7 = (int32_t)(l1p_cnt_7_state);  // #3190
        const int32_t l1v_cnt_7 = (int32_t)(l1v_cnt_7_state);  // #3191
        const int32_t exact_total = flat_exact_cnt.load(((b * (H * NT_Q_PAD8)) + (h * NT_Q_PAD8)) + q_idx_1);  // #3193 #3194 #3195 #3196 #3197 #3198
        const int32_t idiv_73 = exact_total / 6;  // #3513
        const int32_t idiv_74 = exact_total % 6;  // #3514
        const bool idiv_75 = idiv_74 != 0;  // #3515
        const bool idiv_76 = exact_total < 0;  // #3516
        const bool idiv_78 = idiv_76 == false;  // #3518
        const bool idiv_79 = idiv_75 && idiv_78;  // #3519
        const int32_t idiv_80 = (idiv_79) ? (1) : (0);  // #3520
        const int32_t num_ebatches = idiv_73 + idiv_80;  // #3199
        const bool cmp_26 = num_ebatches > 0;  // #3200
        int32_t l0ab_cnt_31_state = 0;  // #3201
        int32_t l0c_cnt_24_state = 0;  // #3202
        int32_t l1k_cnt_17_state = 0;  // #3203
        int32_t l1p_cnt_17_state = 0;  // #3204
        int32_t l1v_cnt_17_state = 0;  // #3205
        if (cmp_26) {  // #3420
            const int32_t lag_e = Min(3, num_ebatches);  // #3206
            const int32_t add_78 = num_ebatches + lag_e;  // #3207
            int32_t l0ab_cnt_30_state = l0ab_cnt_13;  // #3208
            int32_t l0c_cnt_23_state = l0c_cnt_10;  // #3209
            int32_t l1k_cnt_16_state = l1k_cnt_7;  // #3210
            int32_t l1p_cnt_16_state = l1p_cnt_7;  // #3211
            int32_t l1v_cnt_16_state = l1v_cnt_7;  // #3212
            for (int32_t bi = 0; bi < add_78; bi += 1) {  // #3404
                const int32_t l0ab_cnt_carry_2 = (int32_t)(l0ab_cnt_30_state);  // #3213
                const int32_t l0c_cnt_carry_2 = (int32_t)(l0c_cnt_23_state);  // #3214
                const int32_t l1k_cnt_carry_2 = (int32_t)(l1k_cnt_16_state);  // #3215
                const int32_t l1p_cnt_carry_2 = (int32_t)(l1p_cnt_16_state);  // #3216
                const int32_t l1v_cnt_carry_2 = (int32_t)(l1v_cnt_16_state);  // #3217
                const bool cmp_27 = bi < num_ebatches;  // #3218
                int32_t l0ab_cnt_21_state = 0;  // #3219
                int32_t l0c_cnt_18_state = 0;  // #3220
                int32_t l1k_cnt_15_state = 0;  // #3221
                if (cmp_27) {  // #3298
                    const int32_t slot_b_p = bi & 3;  // #3222
                    const int32_t e_base_p = bi * 6;  // #3223
                    const int32_t sub_18 = num_ebatches + -1;  // #3224
                    const bool cmp_28 = bi < sub_18;  // #3225
                    int32_t l0ab_cnt_20_state = 0;  // #3226
                    int32_t l0c_cnt_17_state = 0;  // #3227
                    int32_t l1k_cnt_14_state = 0;  // #3228
                    if (cmp_28) {  // #3288
                        const int32_t l1k_cnt_8 = (int32_t)(l1k_cnt_carry_2 + 3);  // #3229 #3230
                        const int32_t l0ab_cnt_14 = (int32_t)(l0ab_cnt_carry_2 + 3);  // #3231 #3232
                        const int32_t l0c_cnt_11 = (int32_t)(l0c_cnt_carry_2 + 3);  // #3233 #3234
                        WAIT_CUBE<PIPE_S>(0);  // #1049
                        WaitFlag<PIPE_V, PIPE_MTE2>(4);  // #1050
                        const GMTensor<float> view_85 = score_ws[(((cube_idx * 196608) + (slot_b_p * 49152)) + (sb_row * 128))];  // #1053
                        gm_to_ub_pad(ub_score_p0, view_85, 64, 512, 0, 0);  // #1054
                        SetFlag<PIPE_MTE2, PIPE_V>(4);  // #1055
                        WaitFlag<PIPE_V, PIPE_MTE2>(5);  // #1056
                        const GMTensor<float> view_86 = score_ws[(((cube_idx * 196608) + (slot_b_p * 49152)) + ((128 + sb_row) * 128))];  // #1057 #1061
                        gm_to_ub_pad(ub_score_p1, view_86, 64, 512, 0, 0);  // #1062
                        SetFlag<PIPE_MTE2, PIPE_V>(5);  // #1063
                        WaitFlag<PIPE_V, PIPE_MTE2>(6);  // #1064
                        const GMTensor<float> view_87 = score_ws[(((cube_idx * 196608) + (slot_b_p * 49152)) + ((256 + sb_row) * 128))];  // #1065 #1069
                        gm_to_ub_pad(ub_score_p2, view_87, 64, 512, 0, 0);  // #1070
                        SetFlag<PIPE_MTE2, PIPE_V>(6);  // #1071
                        WaitFlag<PIPE_MTE2, PIPE_V>(4);  // #1072
                        PipeBarrier<PIPE_V>();  // c220: V-V hazard #1073
                        vmuls(ub_score_p0.ptr(), ub_score_p0.ptr(), (float)(scale_natural), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #1073
                        WaitFlag<PIPE_MTE2, PIPE_V>(5);  // #1074
                        vmuls(ub_score_p1.ptr(), ub_score_p1.ptr(), (float)(scale_natural), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #1075
                        WaitFlag<PIPE_MTE2, PIPE_V>(6);  // #1076
                        vmuls(ub_score_p2.ptr(), ub_score_p2.ptr(), (float)(scale_natural), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #1077
                        PipeBarrier<PIPE_V>();  // #1078
                        const Tensor<float, Position::UB> view_88 = ub_score_p0;  // #1079
                        vcgmax(ub_red_0.ptr(), view_88.ptr(), (uint8_t)(64), 1, 1, 16);  // #1080
                        const Tensor<float, Position::UB> view_89 = ub_score_p0[64];  // #1081
                        vcgmax(ub_red_1.ptr(), view_89.ptr(), (uint8_t)(64), 1, 1, 16);  // #1082
                        const Tensor<float, Position::UB> view_90 = ub_score_p1;  // #1083
                        vcgmax(ub_red_2.ptr(), view_90.ptr(), (uint8_t)(64), 1, 1, 16);  // #1084
                        const Tensor<float, Position::UB> view_91 = ub_score_p1[64];  // #1085
                        vcgmax(ub_red_3.ptr(), view_91.ptr(), (uint8_t)(64), 1, 1, 16);  // #1086
                        PipeBarrier<PIPE_V>();  // #1087
                        vmax(ub_red_0.ptr(), ub_red_0.ptr(), ub_red_1.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #1088
                        vmax(ub_red_2.ptr(), ub_red_2.ptr(), ub_red_3.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #1089
                        PipeBarrier<PIPE_V>();  // #1090
                        const Tensor<float, Position::UB> view_92 = ub_score_p2;  // #1091
                        vcgmax(ub_red_1.ptr(), view_92.ptr(), (uint8_t)(64), 1, 1, 16);  // #1092
                        const Tensor<float, Position::UB> view_93 = ub_score_p2[64];  // #1093
                        vcgmax(ub_red_3.ptr(), view_93.ptr(), (uint8_t)(64), 1, 1, 16);  // #1094
                        vmax(ub_red_0.ptr(), ub_red_0.ptr(), ub_red_2.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #1095
                        PipeBarrier<PIPE_V>();  // #1096
                        vmax(ub_red_1.ptr(), ub_red_1.ptr(), ub_red_3.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #1097
                        PipeBarrier<PIPE_V>();  // #1098
                        vmax(ub_red_0.ptr(), ub_red_0.ptr(), ub_red_1.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #1099
                        PipeBarrier<PIPE_V>();  // #1100
                        vcgmax(ub_bmax_s.ptr(), ub_red_0.ptr(), (uint8_t)(8), 1, 1, 8);  // #1101
                        vadd(ub_old_max_s.ptr(), ub_rmax_s.ptr(), ub_zero_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1102
                        PipeBarrier<PIPE_V>();  // #1103
                        const Tensor<float, Position::UB> row_diff_e = expdiff_buf.get(bi);  // #1104
                        vmax(ub_rmax_s.ptr(), ub_rmax_s.ptr(), ub_bmax_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1105
                        PipeBarrier<PIPE_V>();  // #1106
                        vsub(row_diff_e.ptr(), ub_old_max_s.ptr(), ub_rmax_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1107
                        ResetMask();  // #1108
                        vbrcb((__ubuf__ uint32_t*)(ub_max.ptr()), (__ubuf__ uint32_t*)(ub_rmax_s.ptr()), (uint16_t)1, (uint16_t)8, (uint8_t)(8));  // #1108
                        PipeBarrier<PIPE_V>();  // #1109
                        vexp(row_diff_e.ptr(), row_diff_e.ptr(), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint8_t)0, (uint8_t)0);  // #1110
                        const Tensor<float, Position::UB> view_94 = ub_score_p0;  // #1111
                        const Tensor<float, Position::UB> view_95 = ub_score_p0;  // #1112
                        vsub(view_94.ptr(), view_95.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #1113
                        const Tensor<float, Position::UB> view_96 = ub_score_p1;  // #1114
                        const Tensor<float, Position::UB> view_97 = ub_score_p1;  // #1115
                        vsub(view_96.ptr(), view_97.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #1116
                        const Tensor<float, Position::UB> view_98 = ub_score_p2;  // #1117
                        const Tensor<float, Position::UB> view_99 = ub_score_p2;  // #1118
                        vsub(view_98.ptr(), view_99.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #1119
                        PipeBarrier<PIPE_V>();  // #1120
                        const Tensor<float, Position::UB> view_100 = ub_score_p0[64];  // #1121
                        const Tensor<float, Position::UB> view_101 = ub_score_p0[64];  // #1122
                        vsub(view_100.ptr(), view_101.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #1123
                        const Tensor<float, Position::UB> view_102 = ub_score_p1[64];  // #1124
                        const Tensor<float, Position::UB> view_103 = ub_score_p1[64];  // #1125
                        vsub(view_102.ptr(), view_103.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #1126
                        const Tensor<float, Position::UB> view_104 = ub_score_p2[64];  // #1127
                        const Tensor<float, Position::UB> view_105 = ub_score_p2[64];  // #1128
                        vsub(view_104.ptr(), view_105.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #1129
                        PipeBarrier<PIPE_V>();  // #1130
                        vexp(ub_score_p0.ptr(), ub_score_p0.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint8_t)8, (uint8_t)8);  // #1131
                        vexp(ub_score_p1.ptr(), ub_score_p1.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint8_t)8, (uint8_t)8);  // #1132
                        vexp(ub_score_p2.ptr(), ub_score_p2.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint8_t)8, (uint8_t)8);  // #1133
                        PipeBarrier<PIPE_V>();  // #1134
                        WaitFlag<PIPE_MTE3, PIPE_V>(4);  // #1135
                        vconv_f322bf16a(ub_p_0.ptr(), ub_score_p0.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)4, (uint16_t)8);  // #1136
                        SetFlag<PIPE_V, PIPE_MTE3>(4);  // #1137
                        WaitFlag<PIPE_V, PIPE_MTE3>(4);  // #1138
                        const GMTensor<bfloat16_t> view_106 = p_ws[(((cube_idx * 196608) + (slot_b_p * 49152)) + (sb_row * 128))];  // #1141
                        ub_to_gm_pad(view_106, ub_p_0, 64, 256, 0, 0);  // #1142
                        SetFlag<PIPE_MTE3, PIPE_V>(4);  // #1143
                        WaitFlag<PIPE_MTE3, PIPE_V>(5);  // #1144
                        vconv_f322bf16a(ub_p_1.ptr(), ub_score_p1.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)4, (uint16_t)8);  // #1145
                        SetFlag<PIPE_V, PIPE_MTE3>(5);  // #1146
                        WaitFlag<PIPE_V, PIPE_MTE3>(5);  // #1147
                        const GMTensor<bfloat16_t> view_107 = p_ws[(((cube_idx * 196608) + (slot_b_p * 49152)) + ((128 + sb_row) * 128))];  // #1148 #1152
                        ub_to_gm_pad(view_107, ub_p_1, 64, 256, 0, 0);  // #1153
                        SetFlag<PIPE_MTE3, PIPE_V>(5);  // #1154
                        WaitFlag<PIPE_MTE3, PIPE_V>(6);  // #1155
                        vconv_f322bf16a(ub_p_2.ptr(), ub_score_p2.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)4, (uint16_t)8);  // #1156
                        SetFlag<PIPE_V, PIPE_MTE3>(6);  // #1157
                        WaitFlag<PIPE_V, PIPE_MTE3>(6);  // #1158
                        const GMTensor<bfloat16_t> view_108 = p_ws[(((cube_idx * 196608) + (slot_b_p * 49152)) + ((256 + sb_row) * 128))];  // #1159 #1163
                        ub_to_gm_pad(view_108, ub_p_2, 64, 256, 0, 0);  // #1164
                        SetFlag<PIPE_MTE3, PIPE_V>(6);  // #1165
                        const Tensor<float, Position::UB> view_109 = ub_score_p0;  // #1166
                        vcgadd(ub_red_0.ptr(), view_109.ptr(), (uint8_t)(64), 1, 1, 16);  // #1167
                        const Tensor<float, Position::UB> view_110 = ub_score_p0[64];  // #1168
                        vcgadd(ub_red_1.ptr(), view_110.ptr(), (uint8_t)(64), 1, 1, 16);  // #1169
                        SetFlag<PIPE_V, PIPE_MTE2>(4);  // #1170
                        const Tensor<float, Position::UB> view_111 = ub_score_p1;  // #1171
                        vcgadd(ub_red_2.ptr(), view_111.ptr(), (uint8_t)(64), 1, 1, 16);  // #1172
                        const Tensor<float, Position::UB> view_112 = ub_score_p1[64];  // #1173
                        vcgadd(ub_red_3.ptr(), view_112.ptr(), (uint8_t)(64), 1, 1, 16);  // #1174
                        SetFlag<PIPE_V, PIPE_MTE2>(5);  // #1175
                        PipeBarrier<PIPE_V>();  // #1176
                        vadd(ub_red_0.ptr(), ub_red_0.ptr(), ub_red_1.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #1177
                        vadd(ub_red_2.ptr(), ub_red_2.ptr(), ub_red_3.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #1178
                        PipeBarrier<PIPE_V>();  // #1179
                        const Tensor<float, Position::UB> view_113 = ub_score_p2;  // #1180
                        vcgadd(ub_red_1.ptr(), view_113.ptr(), (uint8_t)(64), 1, 1, 16);  // #1181
                        const Tensor<float, Position::UB> view_114 = ub_score_p2[64];  // #1182
                        vcgadd(ub_red_3.ptr(), view_114.ptr(), (uint8_t)(64), 1, 1, 16);  // #1183
                        SetFlag<PIPE_V, PIPE_MTE2>(6);  // #1184
                        vadd(ub_red_0.ptr(), ub_red_0.ptr(), ub_red_2.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #1185
                        vmul(ub_rsum_s.ptr(), ub_rsum_s.ptr(), row_diff_e.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1186
                        PipeBarrier<PIPE_V>();  // #1187
                        vadd(ub_red_1.ptr(), ub_red_1.ptr(), ub_red_3.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #1188
                        PipeBarrier<PIPE_V>();  // #1189
                        vadd(ub_red_0.ptr(), ub_red_0.ptr(), ub_red_1.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #1190
                        PipeBarrier<PIPE_V>();  // #1191
                        vcgadd(ub_bsum_s.ptr(), ub_red_0.ptr(), (uint8_t)(8), 1, 1, 8);  // #1192
                        PipeBarrier<PIPE_V>();  // #1193
                        vadd(ub_rsum_s.ptr(), ub_rsum_s.ptr(), ub_bsum_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1194
                        VEC_READY<PIPE_MTE3>(1);  // #1195
                        l0ab_cnt_20_state = l0ab_cnt_14;  // #3235
                        l0c_cnt_17_state = l0c_cnt_11;  // #3236
                        l1k_cnt_14_state = l1k_cnt_8;  // #3237
                    } else {
                        const int32_t b_len_p = exact_total - e_base_p;  // #3238
                        const int32_t l1k_cnt_9 = (int32_t)(l1k_cnt_carry_2 + 1);  // #3239 #3240
                        const int32_t l0ab_cnt_15 = (int32_t)(l0ab_cnt_carry_2 + 1);  // #3241 #3242
                        const int32_t l0c_cnt_12 = (int32_t)(l0c_cnt_carry_2 + 1);  // #3243 #3244
                        const bool cmp_34 = b_len_p > 2;  // #3245
                        int32_t l0ab_cnt_17_state = 0;  // #3246
                        int32_t l0c_cnt_14_state = 0;  // #3247
                        int32_t l1k_cnt_11_state = 0;  // #3248
                        if (cmp_34) {  // #3261
                            const int32_t l1k_cnt_10 = (int32_t)(l1k_cnt_9 + 1);  // #3249 #3250
                            const int32_t l0ab_cnt_16 = (int32_t)(l0ab_cnt_15 + 1);  // #3251 #3252
                            const int32_t l0c_cnt_13 = (int32_t)(l0c_cnt_12 + 1);  // #3253 #3254
                            l0ab_cnt_17_state = l0ab_cnt_16;  // #3255
                            l0c_cnt_14_state = l0c_cnt_13;  // #3256
                            l1k_cnt_11_state = l1k_cnt_10;  // #3257
                        } else {
                            l0ab_cnt_17_state = l0ab_cnt_15;  // #3258
                            l0c_cnt_14_state = l0c_cnt_12;  // #3259
                            l1k_cnt_11_state = l1k_cnt_9;  // #3260
                        }
                        const int32_t l0ab_cnt_17 = (int32_t)(l0ab_cnt_17_state);  // #3262
                        const int32_t l0c_cnt_14 = (int32_t)(l0c_cnt_14_state);  // #3263
                        const int32_t l1k_cnt_11 = (int32_t)(l1k_cnt_11_state);  // #3264
                        const bool cmp_37 = b_len_p > 4;  // #3265
                        int32_t l0ab_cnt_19_state = 0;  // #3266
                        int32_t l0c_cnt_16_state = 0;  // #3267
                        int32_t l1k_cnt_13_state = 0;  // #3268
                        if (cmp_37) {  // #3281
                            const int32_t l1k_cnt_12 = (int32_t)(l1k_cnt_11 + 1);  // #3269 #3270
                            const int32_t l0ab_cnt_18 = (int32_t)(l0ab_cnt_17 + 1);  // #3271 #3272
                            const int32_t l0c_cnt_15 = (int32_t)(l0c_cnt_14 + 1);  // #3273 #3274
                            l0ab_cnt_19_state = l0ab_cnt_18;  // #3275
                            l0c_cnt_16_state = l0c_cnt_15;  // #3276
                            l1k_cnt_13_state = l1k_cnt_12;  // #3277
                        } else {
                            l0ab_cnt_19_state = l0ab_cnt_17;  // #3278
                            l0c_cnt_16_state = l0c_cnt_14;  // #3279
                            l1k_cnt_13_state = l1k_cnt_11;  // #3280
                        }
                        const int32_t l0ab_cnt_19 = (int32_t)(l0ab_cnt_19_state);  // #3282
                        const int32_t l0c_cnt_16 = (int32_t)(l0c_cnt_16_state);  // #3283
                        const int32_t l1k_cnt_13 = (int32_t)(l1k_cnt_13_state);  // #3284
                        WAIT_CUBE<PIPE_S>(0);  // #1521
                        WaitFlag<PIPE_V, PIPE_MTE2>(4);  // #1522
                        const GMTensor<float> view_139 = score_ws[(((cube_idx * 196608) + (slot_b_p * 49152)) + (sb_row * 128))];  // #1525
                        gm_to_ub_pad(ub_score_p0, view_139, 64, 512, 0, 0);  // #1526
                        SetFlag<PIPE_MTE2, PIPE_V>(4);  // #1527
                        if (cmp_34) {  // #1537
                            WaitFlag<PIPE_V, PIPE_MTE2>(5);  // #1529
                            const GMTensor<float> view_140 = score_ws[(((cube_idx * 196608) + (slot_b_p * 49152)) + ((128 + sb_row) * 128))];  // #1530 #1534
                            gm_to_ub_pad(ub_score_p1, view_140, 64, 512, 0, 0);  // #1535
                            SetFlag<PIPE_MTE2, PIPE_V>(5);  // #1536
                        }
                        if (cmp_37) {  // #1547
                            WaitFlag<PIPE_V, PIPE_MTE2>(6);  // #1539
                            const GMTensor<float> view_141 = score_ws[(((cube_idx * 196608) + (slot_b_p * 49152)) + ((256 + sb_row) * 128))];  // #1540 #1544
                            gm_to_ub_pad(ub_score_p2, view_141, 64, 512, 0, 0);  // #1545
                            SetFlag<PIPE_MTE2, PIPE_V>(6);  // #1546
                        }
                        WaitFlag<PIPE_MTE2, PIPE_V>(4);  // #1548
                        PipeBarrier<PIPE_V>();  // c220: V-V hazard #1549
                        vmuls(ub_score_p0.ptr(), ub_score_p0.ptr(), (float)(scale_natural), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #1549
                        if (b_len_p > 2) {  // #1550 #1553
                            WaitFlag<PIPE_MTE2, PIPE_V>(5);  // #1551
                            vmuls(ub_score_p1.ptr(), ub_score_p1.ptr(), (float)(scale_natural), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #1552
                        }
                        if (b_len_p > 4) {  // #1554 #1557
                            WaitFlag<PIPE_MTE2, PIPE_V>(6);  // #1555
                            vmuls(ub_score_p2.ptr(), ub_score_p2.ptr(), (float)(scale_natural), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #1556
                        }
                        PipeBarrier<PIPE_V>();  // #1558
                        if (b_len_p == 1) {  // #1559 #1570
                            const Tensor<float, Position::UB> view_142 = ub_score_p0[64];  // #1560
                            vector_dup(view_142.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #1561
                        } else {
                            if (b_len_p == 3) {  // #1562 #1569
                                const Tensor<float, Position::UB> view_143 = ub_score_p1[64];  // #1563
                                vector_dup(view_143.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #1564
                            } else {
                                if (b_len_p == 5) {  // #1565 #1568
                                    const Tensor<float, Position::UB> view_144 = ub_score_p2[64];  // #1566
                                    vector_dup(view_144.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #1567
                                }
                            }
                        }
                        const int32_t tail_n = T - ((NT - 1) * 64);  // #1571 #1572 #1573
                        if (tail_n < 64) {  // #1574 #1622
                            const int32_t sub_50 = exact_total - 1;  // #1575
                            const int32_t last_blk = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx_1 * NPAD)) + sub_50));  // #1577 #1578 #1579 #1580 #1581 #1582 #1583 #1584 #1585 #1586 #1587
                            if (last_blk == (NT - 1)) {  // #1588 #1589 #1621
                                PipeBarrier<PIPE_V>();  // #1590
                                const uint64_t low = 1ULL << tail_n;  // #1593
                                const uint64_t suffix_mask = 18446744073709551615ULL ^ (low - 1ULL);  // #1595 #1596
                                SetVectorMask((uint64_t)(0), (uint64_t)(suffix_mask));  // #1597
                                if (b_len_p == 1) {  // #1598 #1619
                                    const Tensor<float, Position::UB> view_145 = ub_score_p0;  // #1599
                                    vector_dup(view_145.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #1600
                                } else {
                                    if (b_len_p == 2) {  // #1601 #1618
                                        const Tensor<float, Position::UB> view_146 = ub_score_p0[64];  // #1602
                                        vector_dup(view_146.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #1603
                                    } else {
                                        if (b_len_p == 3) {  // #1604 #1617
                                            const Tensor<float, Position::UB> view_147 = ub_score_p1;  // #1605
                                            vector_dup(view_147.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #1606
                                        } else {
                                            if (b_len_p == 4) {  // #1607 #1616
                                                const Tensor<float, Position::UB> view_148 = ub_score_p1[64];  // #1608
                                                vector_dup(view_148.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #1609
                                            } else {
                                                if (b_len_p == 5) {  // #1610 #1615
                                                    const Tensor<float, Position::UB> view_149 = ub_score_p2;  // #1611
                                                    vector_dup(view_149.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #1612
                                                } else {
                                                    const Tensor<float, Position::UB> view_150 = ub_score_p2[64];  // #1613
                                                    vector_dup(view_150.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #1614
                                                }
                                            }
                                        }
                                    }
                                }
                                ResetMask();  // #1620
                            }
                        }
                        PipeBarrier<PIPE_V>();  // #1623
                        const Tensor<float, Position::UB> view_151 = ub_score_p0;  // #1624
                        vcgmax(ub_red_0.ptr(), view_151.ptr(), (uint8_t)(64), 1, 1, 16);  // #1625
                        const Tensor<float, Position::UB> view_152 = ub_score_p0[64];  // #1626
                        vcgmax(ub_red_1.ptr(), view_152.ptr(), (uint8_t)(64), 1, 1, 16);  // #1627
                        if (b_len_p > 2) {  // #1628 #1635
                            const Tensor<float, Position::UB> view_153 = ub_score_p1;  // #1629
                            vcgmax(ub_red_2.ptr(), view_153.ptr(), (uint8_t)(64), 1, 1, 16);  // #1630
                            const Tensor<float, Position::UB> view_154 = ub_score_p1[64];  // #1631
                            vcgmax(ub_red_3.ptr(), view_154.ptr(), (uint8_t)(64), 1, 1, 16);  // #1632
                        } else {
                            vector_dup(ub_red_2.ptr(), (float)(-1e+30f), (uint8_t)(8), (uint16_t)1, 1, (uint8_t)8, 0);  // #1633
                            vector_dup(ub_red_3.ptr(), (float)(-1e+30f), (uint8_t)(8), (uint16_t)1, 1, (uint8_t)8, 0);  // #1634
                        }
                        PipeBarrier<PIPE_V>();  // #1636
                        vmax(ub_red_0.ptr(), ub_red_0.ptr(), ub_red_1.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #1637
                        vmax(ub_red_2.ptr(), ub_red_2.ptr(), ub_red_3.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #1638
                        PipeBarrier<PIPE_V>();  // #1639
                        if (b_len_p > 4) {  // #1640 #1647
                            const Tensor<float, Position::UB> view_155 = ub_score_p2;  // #1641
                            vcgmax(ub_red_1.ptr(), view_155.ptr(), (uint8_t)(64), 1, 1, 16);  // #1642
                            const Tensor<float, Position::UB> view_156 = ub_score_p2[64];  // #1643
                            vcgmax(ub_red_3.ptr(), view_156.ptr(), (uint8_t)(64), 1, 1, 16);  // #1644
                        } else {
                            vector_dup(ub_red_1.ptr(), (float)(-1e+30f), (uint8_t)(8), (uint16_t)1, 1, (uint8_t)8, 0);  // #1645
                            vector_dup(ub_red_3.ptr(), (float)(-1e+30f), (uint8_t)(8), (uint16_t)1, 1, (uint8_t)8, 0);  // #1646
                        }
                        vmax(ub_red_0.ptr(), ub_red_0.ptr(), ub_red_2.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #1648
                        PipeBarrier<PIPE_V>();  // #1649
                        vmax(ub_red_1.ptr(), ub_red_1.ptr(), ub_red_3.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #1650
                        PipeBarrier<PIPE_V>();  // #1651
                        vmax(ub_red_0.ptr(), ub_red_0.ptr(), ub_red_1.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #1652
                        PipeBarrier<PIPE_V>();  // #1653
                        vcgmax(ub_bmax_s.ptr(), ub_red_0.ptr(), (uint8_t)(8), 1, 1, 8);  // #1654
                        vadd(ub_old_max_s.ptr(), ub_rmax_s.ptr(), ub_zero_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1655
                        PipeBarrier<PIPE_V>();  // #1656
                        const Tensor<float, Position::UB> row_diff_t = expdiff_buf.get(bi);  // #1657
                        vmax(ub_rmax_s.ptr(), ub_rmax_s.ptr(), ub_bmax_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1658
                        PipeBarrier<PIPE_V>();  // #1659
                        vsub(row_diff_t.ptr(), ub_old_max_s.ptr(), ub_rmax_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1660
                        ResetMask();  // #1661
                        vbrcb((__ubuf__ uint32_t*)(ub_max.ptr()), (__ubuf__ uint32_t*)(ub_rmax_s.ptr()), (uint16_t)1, (uint16_t)8, (uint8_t)(8));  // #1661
                        PipeBarrier<PIPE_V>();  // #1662
                        vexp(row_diff_t.ptr(), row_diff_t.ptr(), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint8_t)0, (uint8_t)0);  // #1663
                        const Tensor<float, Position::UB> view_157 = ub_score_p0;  // #1664
                        const Tensor<float, Position::UB> view_158 = ub_score_p0;  // #1665
                        vsub(view_157.ptr(), view_158.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #1666
                        if (b_len_p > 2) {  // #1667 #1671
                            const Tensor<float, Position::UB> view_159 = ub_score_p1;  // #1668
                            const Tensor<float, Position::UB> view_160 = ub_score_p1;  // #1669
                            vsub(view_159.ptr(), view_160.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #1670
                        }
                        if (b_len_p > 4) {  // #1672 #1676
                            const Tensor<float, Position::UB> view_161 = ub_score_p2;  // #1673
                            const Tensor<float, Position::UB> view_162 = ub_score_p2;  // #1674
                            vsub(view_161.ptr(), view_162.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #1675
                        }
                        PipeBarrier<PIPE_V>();  // #1677
                        const Tensor<float, Position::UB> view_163 = ub_score_p0[64];  // #1678
                        const Tensor<float, Position::UB> view_164 = ub_score_p0[64];  // #1679
                        vsub(view_163.ptr(), view_164.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #1680
                        if (b_len_p > 2) {  // #1681 #1685
                            const Tensor<float, Position::UB> view_165 = ub_score_p1[64];  // #1682
                            const Tensor<float, Position::UB> view_166 = ub_score_p1[64];  // #1683
                            vsub(view_165.ptr(), view_166.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #1684
                        }
                        if (b_len_p > 4) {  // #1686 #1690
                            const Tensor<float, Position::UB> view_167 = ub_score_p2[64];  // #1687
                            const Tensor<float, Position::UB> view_168 = ub_score_p2[64];  // #1688
                            vsub(view_167.ptr(), view_168.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #1689
                        }
                        PipeBarrier<PIPE_V>();  // #1691
                        vexp(ub_score_p0.ptr(), ub_score_p0.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint8_t)8, (uint8_t)8);  // #1692
                        if (b_len_p > 2) {  // #1693 #1695
                            vexp(ub_score_p1.ptr(), ub_score_p1.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint8_t)8, (uint8_t)8);  // #1694
                        }
                        if (b_len_p > 4) {  // #1696 #1698
                            vexp(ub_score_p2.ptr(), ub_score_p2.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint8_t)8, (uint8_t)8);  // #1697
                        }
                        PipeBarrier<PIPE_V>();  // #1699
                        WaitFlag<PIPE_MTE3, PIPE_V>(4);  // #1700
                        vconv_f322bf16a(ub_p_0.ptr(), ub_score_p0.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)4, (uint16_t)8);  // #1701
                        SetFlag<PIPE_V, PIPE_MTE3>(4);  // #1702
                        WaitFlag<PIPE_V, PIPE_MTE3>(4);  // #1703
                        const GMTensor<bfloat16_t> view_169 = p_ws[(((cube_idx * 196608) + (slot_b_p * 49152)) + (sb_row * 128))];  // #1706
                        ub_to_gm_pad(view_169, ub_p_0, 64, 256, 0, 0);  // #1707
                        SetFlag<PIPE_MTE3, PIPE_V>(4);  // #1708
                        if (b_len_p > 2) {  // #1709 #1721
                            WaitFlag<PIPE_MTE3, PIPE_V>(5);  // #1710
                            vconv_f322bf16a(ub_p_1.ptr(), ub_score_p1.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)4, (uint16_t)8);  // #1711
                            SetFlag<PIPE_V, PIPE_MTE3>(5);  // #1712
                            WaitFlag<PIPE_V, PIPE_MTE3>(5);  // #1713
                            const GMTensor<bfloat16_t> view_170 = p_ws[(((cube_idx * 196608) + (slot_b_p * 49152)) + ((128 + sb_row) * 128))];  // #1714 #1718
                            ub_to_gm_pad(view_170, ub_p_1, 64, 256, 0, 0);  // #1719
                            SetFlag<PIPE_MTE3, PIPE_V>(5);  // #1720
                        }
                        if (b_len_p > 4) {  // #1722 #1734
                            WaitFlag<PIPE_MTE3, PIPE_V>(6);  // #1723
                            vconv_f322bf16a(ub_p_2.ptr(), ub_score_p2.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)4, (uint16_t)8);  // #1724
                            SetFlag<PIPE_V, PIPE_MTE3>(6);  // #1725
                            WaitFlag<PIPE_V, PIPE_MTE3>(6);  // #1726
                            const GMTensor<bfloat16_t> view_171 = p_ws[(((cube_idx * 196608) + (slot_b_p * 49152)) + ((256 + sb_row) * 128))];  // #1727 #1731
                            ub_to_gm_pad(view_171, ub_p_2, 64, 256, 0, 0);  // #1732
                            SetFlag<PIPE_MTE3, PIPE_V>(6);  // #1733
                        }
                        const Tensor<float, Position::UB> view_172 = ub_score_p0;  // #1735
                        vcgadd(ub_red_0.ptr(), view_172.ptr(), (uint8_t)(64), 1, 1, 16);  // #1736
                        const Tensor<float, Position::UB> view_173 = ub_score_p0[64];  // #1737
                        vcgadd(ub_red_1.ptr(), view_173.ptr(), (uint8_t)(64), 1, 1, 16);  // #1738
                        SetFlag<PIPE_V, PIPE_MTE2>(4);  // #1739
                        if (b_len_p > 2) {  // #1740 #1748
                            const Tensor<float, Position::UB> view_174 = ub_score_p1;  // #1741
                            vcgadd(ub_red_2.ptr(), view_174.ptr(), (uint8_t)(64), 1, 1, 16);  // #1742
                            const Tensor<float, Position::UB> view_175 = ub_score_p1[64];  // #1743
                            vcgadd(ub_red_3.ptr(), view_175.ptr(), (uint8_t)(64), 1, 1, 16);  // #1744
                            SetFlag<PIPE_V, PIPE_MTE2>(5);  // #1745
                        } else {
                            vector_dup(ub_red_2.ptr(), (float)(0.0f), (uint8_t)(8), (uint16_t)1, 1, (uint8_t)8, 0);  // #1746
                            vector_dup(ub_red_3.ptr(), (float)(0.0f), (uint8_t)(8), (uint16_t)1, 1, (uint8_t)8, 0);  // #1747
                        }
                        PipeBarrier<PIPE_V>();  // #1749
                        vadd(ub_red_0.ptr(), ub_red_0.ptr(), ub_red_1.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #1750
                        vadd(ub_red_2.ptr(), ub_red_2.ptr(), ub_red_3.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #1751
                        PipeBarrier<PIPE_V>();  // #1752
                        if (b_len_p > 4) {  // #1753 #1761
                            const Tensor<float, Position::UB> view_176 = ub_score_p2;  // #1754
                            vcgadd(ub_red_1.ptr(), view_176.ptr(), (uint8_t)(64), 1, 1, 16);  // #1755
                            const Tensor<float, Position::UB> view_177 = ub_score_p2[64];  // #1756
                            vcgadd(ub_red_3.ptr(), view_177.ptr(), (uint8_t)(64), 1, 1, 16);  // #1757
                            SetFlag<PIPE_V, PIPE_MTE2>(6);  // #1758
                        } else {
                            vector_dup(ub_red_1.ptr(), (float)(0.0f), (uint8_t)(8), (uint16_t)1, 1, (uint8_t)8, 0);  // #1759
                            vector_dup(ub_red_3.ptr(), (float)(0.0f), (uint8_t)(8), (uint16_t)1, 1, (uint8_t)8, 0);  // #1760
                        }
                        vadd(ub_red_0.ptr(), ub_red_0.ptr(), ub_red_2.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #1762
                        vmul(ub_rsum_s.ptr(), ub_rsum_s.ptr(), row_diff_t.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1763
                        PipeBarrier<PIPE_V>();  // #1764
                        vadd(ub_red_1.ptr(), ub_red_1.ptr(), ub_red_3.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #1765
                        PipeBarrier<PIPE_V>();  // #1766
                        vadd(ub_red_0.ptr(), ub_red_0.ptr(), ub_red_1.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #1767
                        PipeBarrier<PIPE_V>();  // #1768
                        vcgadd(ub_bsum_s.ptr(), ub_red_0.ptr(), (uint8_t)(8), 1, 1, 8);  // #1769
                        PipeBarrier<PIPE_V>();  // #1770
                        vadd(ub_rsum_s.ptr(), ub_rsum_s.ptr(), ub_bsum_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1771
                        VEC_READY<PIPE_MTE3>(1);  // #1772
                        l0ab_cnt_20_state = l0ab_cnt_19;  // #3285
                        l0c_cnt_17_state = l0c_cnt_16;  // #3286
                        l1k_cnt_14_state = l1k_cnt_13;  // #3287
                    }
                    const int32_t l0ab_cnt_20 = (int32_t)(l0ab_cnt_20_state);  // #3289
                    const int32_t l0c_cnt_17 = (int32_t)(l0c_cnt_17_state);  // #3290
                    const int32_t l1k_cnt_14 = (int32_t)(l1k_cnt_14_state);  // #3291
                    l0ab_cnt_21_state = l0ab_cnt_20;  // #3292
                    l0c_cnt_18_state = l0c_cnt_17;  // #3293
                    l1k_cnt_15_state = l1k_cnt_14;  // #3294
                } else {
                    l0ab_cnt_21_state = l0ab_cnt_carry_2;  // #3295
                    l0c_cnt_18_state = l0c_cnt_carry_2;  // #3296
                    l1k_cnt_15_state = l1k_cnt_carry_2;  // #3297
                }
                const int32_t l0ab_cnt_21 = (int32_t)(l0ab_cnt_21_state);  // #3299
                const int32_t l0c_cnt_18 = (int32_t)(l0c_cnt_18_state);  // #3300
                const int32_t l1k_cnt_15 = (int32_t)(l1k_cnt_15_state);  // #3301
                const bool cmp_66 = bi >= lag_e;  // #3302
                int32_t l0ab_cnt_29_state = 0;  // #3303
                int32_t l0c_cnt_22_state = 0;  // #3304
                int32_t l1p_cnt_15_state = 0;  // #3305
                int32_t l1v_cnt_15_state = 0;  // #3306
                if (cmp_66) {  // #3394
                    const int32_t ci = bi - lag_e;  // #3307
                    const int32_t slot_b_c = ci & 3;  // #3308
                    const int32_t e_base_c = ci * 6;  // #3309
                    const int32_t sub_56 = num_ebatches + -1;  // #3310
                    const bool cmp_67 = ci < sub_56;  // #3311
                    int32_t l0ab_cnt_28_state = 0;  // #3312
                    int32_t l0c_cnt_21_state = 0;  // #3313
                    int32_t l1p_cnt_14_state = 0;  // #3314
                    int32_t l1v_cnt_14_state = 0;  // #3315
                    if (cmp_67) {  // #3381
                        const int32_t l0ab_cnt_22 = (int32_t)(l0ab_cnt_21 + 3);  // #3316 #3317
                        const int32_t l1p_cnt_8 = (int32_t)(l1p_cnt_carry_2 + 3);  // #3318 #3319
                        const int32_t l1v_cnt_8 = (int32_t)(l1v_cnt_carry_2 + 3);  // #3320 #3321
                        const int32_t l0c_cnt_19 = (int32_t)(l0c_cnt_18 + 1);  // #3322 #3323
                        l0ab_cnt_28_state = l0ab_cnt_22;  // #3324
                        l0c_cnt_21_state = l0c_cnt_19;  // #3325
                        l1p_cnt_14_state = l1p_cnt_8;  // #3326
                        l1v_cnt_14_state = l1v_cnt_8;  // #3327
                    } else {
                        const int32_t b_len_c = exact_total - e_base_c;  // #3328
                        const int32_t l0ab_cnt_23 = (int32_t)(l0ab_cnt_21 + 1);  // #3329 #3330
                        const int32_t l1p_cnt_9 = (int32_t)(l1p_cnt_carry_2 + 1);  // #3331 #3332
                        const int32_t l1v_cnt_9 = (int32_t)(l1v_cnt_carry_2 + 1);  // #3333 #3334
                        const bool cmp_73 = b_len_c > 2;  // #3335
                        int32_t l0ab_cnt_25_state = 0;  // #3336
                        int32_t l1p_cnt_11_state = 0;  // #3337
                        int32_t l1v_cnt_11_state = 0;  // #3338
                        if (cmp_73) {  // #3351
                            const int32_t l0ab_cnt_24 = (int32_t)(l0ab_cnt_23 + 1);  // #3339 #3340
                            const int32_t l1p_cnt_10 = (int32_t)(l1p_cnt_9 + 1);  // #3341 #3342
                            const int32_t l1v_cnt_10 = (int32_t)(l1v_cnt_9 + 1);  // #3343 #3344
                            l0ab_cnt_25_state = l0ab_cnt_24;  // #3345
                            l1p_cnt_11_state = l1p_cnt_10;  // #3346
                            l1v_cnt_11_state = l1v_cnt_10;  // #3347
                        } else {
                            l0ab_cnt_25_state = l0ab_cnt_23;  // #3348
                            l1p_cnt_11_state = l1p_cnt_9;  // #3349
                            l1v_cnt_11_state = l1v_cnt_9;  // #3350
                        }
                        const int32_t l0ab_cnt_25 = (int32_t)(l0ab_cnt_25_state);  // #3352
                        const int32_t l1p_cnt_11 = (int32_t)(l1p_cnt_11_state);  // #3353
                        const int32_t l1v_cnt_11 = (int32_t)(l1v_cnt_11_state);  // #3354
                        const bool cmp_76 = b_len_c > 4;  // #3355
                        int32_t l0ab_cnt_27_state = 0;  // #3356
                        int32_t l1p_cnt_13_state = 0;  // #3357
                        int32_t l1v_cnt_13_state = 0;  // #3358
                        if (cmp_76) {  // #3371
                            const int32_t l0ab_cnt_26 = (int32_t)(l0ab_cnt_25 + 1);  // #3359 #3360
                            const int32_t l1p_cnt_12 = (int32_t)(l1p_cnt_11 + 1);  // #3361 #3362
                            const int32_t l1v_cnt_12 = (int32_t)(l1v_cnt_11 + 1);  // #3363 #3364
                            l0ab_cnt_27_state = l0ab_cnt_26;  // #3365
                            l1p_cnt_13_state = l1p_cnt_12;  // #3366
                            l1v_cnt_13_state = l1v_cnt_12;  // #3367
                        } else {
                            l0ab_cnt_27_state = l0ab_cnt_25;  // #3368
                            l1p_cnt_13_state = l1p_cnt_11;  // #3369
                            l1v_cnt_13_state = l1v_cnt_11;  // #3370
                        }
                        const int32_t l0ab_cnt_27 = (int32_t)(l0ab_cnt_27_state);  // #3372
                        const int32_t l1p_cnt_13 = (int32_t)(l1p_cnt_13_state);  // #3373
                        const int32_t l1v_cnt_13 = (int32_t)(l1v_cnt_13_state);  // #3374
                        const int32_t l0c_cnt_20 = (int32_t)(l0c_cnt_18 + 1);  // #3375 #3376
                        l0ab_cnt_28_state = l0ab_cnt_27;  // #3377
                        l0c_cnt_21_state = l0c_cnt_20;  // #3378
                        l1p_cnt_14_state = l1p_cnt_13;  // #3379
                        l1v_cnt_14_state = l1v_cnt_13;  // #3380
                    }
                    const int32_t l0ab_cnt_28 = (int32_t)(l0ab_cnt_28_state);  // #3382
                    const int32_t l0c_cnt_21 = (int32_t)(l0c_cnt_21_state);  // #3383
                    const int32_t l1p_cnt_14 = (int32_t)(l1p_cnt_14_state);  // #3384
                    const int32_t l1v_cnt_14 = (int32_t)(l1v_cnt_14_state);  // #3385
                    WAIT_CUBE<PIPE_S>(2);  // #2359
                    WaitFlag<PIPE_V, PIPE_MTE2>(6);  // #2360
                    const GMTensor<float> view_222 = pv_ws[(((cube_idx * 65536) + (slot_b_c * 16384)) + (sb_row * 128))];  // #2363
                    gm_to_ub_pad(ub_score_p2, view_222, 64, 512, 0, 0);  // #2364
                    SetFlag<PIPE_MTE2, PIPE_V>(6);  // #2365
                    const Tensor<float, Position::UB> slot_197 = expdiff_buf.get(ci);  // #2366
                    PipeBarrier<PIPE_V>();  // c220: V-V hazard #2367
                    ResetMask();  // #2367
                    vbrcb((__ubuf__ uint32_t*)(ub_max.ptr()), (__ubuf__ uint32_t*)(slot_197.ptr()), (uint16_t)1, (uint16_t)8, (uint8_t)(8));  // #2367
                    PipeBarrier<PIPE_V>();  // #2368
                    const Tensor<float, Position::UB> view_223 = accum_ub;  // #2369
                    const Tensor<float, Position::UB> view_224 = accum_ub;  // #2370
                    vmul(view_223.ptr(), view_224.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #2371
                    const Tensor<float, Position::UB> view_225 = accum_ub[64];  // #2372
                    const Tensor<float, Position::UB> view_226 = accum_ub[64];  // #2373
                    PipeBarrier<PIPE_V>();  // c220: V-V hazard #2374
                    vmul(view_225.ptr(), view_226.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #2374
                    PipeBarrier<PIPE_V>();  // #2375
                    WaitFlag<PIPE_MTE2, PIPE_V>(6);  // #2376
                    vadd(accum_ub.ptr(), accum_ub.ptr(), ub_score_p2.ptr(), (uint8_t)(128), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #2377
                    SetFlag<PIPE_V, PIPE_MTE2>(6);  // #2378
                    l0ab_cnt_29_state = l0ab_cnt_28;  // #3386
                    l0c_cnt_22_state = l0c_cnt_21;  // #3387
                    l1p_cnt_15_state = l1p_cnt_14;  // #3388
                    l1v_cnt_15_state = l1v_cnt_14;  // #3389
                } else {
                    l0ab_cnt_29_state = l0ab_cnt_21;  // #3390
                    l0c_cnt_22_state = l0c_cnt_18;  // #3391
                    l1p_cnt_15_state = l1p_cnt_carry_2;  // #3392
                    l1v_cnt_15_state = l1v_cnt_carry_2;  // #3393
                }
                const int32_t l0ab_cnt_29 = (int32_t)(l0ab_cnt_29_state);  // #3395
                const int32_t l0c_cnt_22 = (int32_t)(l0c_cnt_22_state);  // #3396
                const int32_t l1p_cnt_15 = (int32_t)(l1p_cnt_15_state);  // #3397
                const int32_t l1v_cnt_15 = (int32_t)(l1v_cnt_15_state);  // #3398
                l0ab_cnt_30_state = l0ab_cnt_29;  // #3399
                l0c_cnt_23_state = l0c_cnt_22;  // #3400
                l1k_cnt_16_state = l1k_cnt_15;  // #3401
                l1p_cnt_16_state = l1p_cnt_15;  // #3402
                l1v_cnt_16_state = l1v_cnt_15;  // #3403
            }
            const int32_t l0ab_cnt_30 = (int32_t)(l0ab_cnt_30_state);  // #3405
            const int32_t l0c_cnt_23 = (int32_t)(l0c_cnt_23_state);  // #3406
            const int32_t l1k_cnt_16 = (int32_t)(l1k_cnt_16_state);  // #3407
            const int32_t l1p_cnt_16 = (int32_t)(l1p_cnt_16_state);  // #3408
            const int32_t l1v_cnt_16 = (int32_t)(l1v_cnt_16_state);  // #3409
            l0ab_cnt_31_state = l0ab_cnt_30;  // #3410
            l0c_cnt_24_state = l0c_cnt_23;  // #3411
            l1k_cnt_17_state = l1k_cnt_16;  // #3412
            l1p_cnt_17_state = l1p_cnt_16;  // #3413
            l1v_cnt_17_state = l1v_cnt_16;  // #3414
        } else {
            l0ab_cnt_31_state = l0ab_cnt_13;  // #3415
            l0c_cnt_24_state = l0c_cnt_10;  // #3416
            l1k_cnt_17_state = l1k_cnt_7;  // #3417
            l1p_cnt_17_state = l1p_cnt_7;  // #3418
            l1v_cnt_17_state = l1v_cnt_7;  // #3419
        }
        const int32_t l0ab_cnt_31 = (int32_t)(l0ab_cnt_31_state);  // #3421
        const int32_t l0c_cnt_24 = (int32_t)(l0c_cnt_24_state);  // #3422
        const int32_t l1k_cnt_17 = (int32_t)(l1k_cnt_17_state);  // #3423
        const int32_t l1p_cnt_17 = (int32_t)(l1p_cnt_17_state);  // #3424
        const int32_t l1v_cnt_17 = (int32_t)(l1v_cnt_17_state);  // #3425
        WaitFlag<PIPE_V, PIPE_MTE2>(4);  // #2387
        WaitFlag<PIPE_V, PIPE_MTE2>(5);  // #2388
        WaitFlag<PIPE_V, PIPE_MTE2>(6);  // #2389
        WaitFlag<PIPE_MTE3, PIPE_V>(4);  // #2390
        WaitFlag<PIPE_MTE3, PIPE_V>(5);  // #2391
        WaitFlag<PIPE_MTE3, PIPE_V>(6);  // #2392
        PipeBarrier<PIPE_ALL>();  // #3426
        ResetMask();  // #2394
        vbrcb((__ubuf__ uint32_t*)(ub_max.ptr()), (__ubuf__ uint32_t*)(ub_rsum_s.ptr()), (uint16_t)1, (uint16_t)8, (uint8_t)(8));  // #2394
        const Tensor<float, Position::UB> view_227 = accum_ub;  // #3427
        const Tensor<float, Position::UB> view_228 = accum_ub;  // #3428
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #2397
        vdiv(view_227.ptr(), view_228.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #2397
        const Tensor<float, Position::UB> view_229 = accum_ub[64];  // #3429
        const Tensor<float, Position::UB> view_230 = accum_ub[64];  // #3430
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #2400
        vdiv(view_229.ptr(), view_230.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #2400
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #2401
        vconv_f322bf16a(ub_p_0.ptr(), accum_ub.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)4, (uint16_t)8);  // #2401
        vln(ub_sum_s.ptr(), ub_rsum_s.ptr(), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint8_t)0, (uint8_t)0);  // #2402
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #2403
        vadd(ub_rmax_s.ptr(), ub_rmax_s.ptr(), ub_sum_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #2403
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #2404
        vmuls(ub_rmax_s.ptr(), ub_rmax_s.ptr(), (float)(1.4426950408889634f), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #2404
        ev_v_mte3_ready_1.set();  // #2775
        ev_v_mte3_ready_1.wait();  // #2776
        if (local_valid_m > 0) {  // #2405 #2417
            const int32_t out_row = q_start + sb_row;  // #2406
            const GMTensor<bfloat16_t> view_231 = out[(((b * ((T * H) * 128)) + (out_row * (H * 128))) + (h * 128))];  // #2409
            const Tensor<bfloat16_t, Position::UB> view_232 = ub_p_0;  // #2410
            ub_to_gm_pad(view_231, view_232, local_valid_m, 256, 0, ((128 + ((H * 128) - 128)) * 2) - 256);  // #2762 #2763 #2764 #2765 #2766 #2411
            const GMTensor<float> view_233 = lse[(((b * (H * T_LSE)) + (h * T_LSE)) + out_row)];  // #2414
            const Tensor<float, Position::UB> view_234 = ub_rmax_s;  // #2415
            const int32_t sub_84 = 64 - local_valid_m;  // #2770
            const int32_t idiv_81 = sub_84 / 8;  // #3521
            const int32_t idiv_82 = sub_84 % 8;  // #3522
            const bool idiv_83 = idiv_82 != 0;  // #3523
            const bool idiv_84 = sub_84 < 0;  // #3524
            const bool idiv_86 = idiv_84 != false;  // #3526
            const bool idiv_87 = idiv_83 && idiv_86;  // #3527
            const int32_t idiv_88 = (idiv_87) ? (1) : (0);  // #3528
            const int32_t div_1 = idiv_81 - idiv_88;  // #2771
            ub_to_gm_pad(view_233, view_234, 1, local_valid_m * 4, div_1, 0);  // #2767 #2416
        }
        const int32_t l1q_cnt_1 = (int32_t)(l1q_cnt_carry + 1);  // #3431 #3432
        l0ab_cnt_32_state = l0ab_cnt_31;  // #3433
        l0c_cnt_25_state = l0c_cnt_24;  // #3434
        l1k_cnt_18_state = l1k_cnt_17;  // #3435
        l1p_cnt_18_state = l1p_cnt_17;  // #3436
        l1q_cnt_2_state = l1q_cnt_1;  // #3437
        l1v_cnt_18_state = l1v_cnt_17;  // #3438
        ev_mte3_v_valid_1.set();  // #2777
    }
    WAIT_CUBE<PIPE_S>(1);  // #3440
    WAIT_CUBE<PIPE_S>(1);  // #3440
    return;  // #3440
}
