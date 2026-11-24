#pragma once
using namespace ascrip;

__aicore__ inline void sol_attn_short_fwd_m128_kernel_vec(GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR out_, GM_ADDR lse_, int32_t B, int32_t T, int32_t H, int32_t NT, int32_t NT_Q, int32_t NT_Q_PAD8, int32_t T_LSE, int32_t sink_start, int32_t sink_end, float scale_log2, float scale_sq, float tau, GM_ADDR workspace)
{
    GMTensor<bfloat16_t> q((__gm__ bfloat16_t*)q_);
    GMTensor<bfloat16_t> k((__gm__ bfloat16_t*)k_);
    GMTensor<bfloat16_t> v((__gm__ bfloat16_t*)v_);
    GMTensor<bfloat16_t> out((__gm__ bfloat16_t*)out_);
    GMTensor<float> lse((__gm__ float*)lse_);
    SetAtomicNone();
    ResetMask();
    SetMaskNorm();
    SEvent<PIPE_MTE2, PIPE_MTE3, 0, 0> ev_mte2_mte3_ready_1;  // guards arena #2713
    SEvent<PIPE_MTE3, PIPE_MTE2, 1, 0> ev_mte3_mte2_valid_1;  // guards arena #2714
    SEvent<PIPE_MTE2, PIPE_V, 0, 0> ev_mte2_v_ready_d0_1;  // guards arena #2723
    SEvent<PIPE_V, PIPE_MTE2, 1, 0> ev_v_mte2_valid_d0_1;  // guards arena #2724
    SEvent<PIPE_MTE2, PIPE_V, 0, 1> ev_mte2_v_ready_d1_1;  // guards arena #2725
    SEvent<PIPE_V, PIPE_MTE2, 1, 1> ev_v_mte2_valid_d1_1;  // guards arena #2726
    SEvent<PIPE_V, PIPE_MTE3, 0, 0> ev_v_mte3_ready_1;  // guards arena #2749
    SEvent<PIPE_MTE3, PIPE_V, 1, 0> ev_mte3_v_valid_1;  // guards arena #2750
    const Tensor<uint8_t, Position::UB> arena((uint64_t)(0));  // #2776
    const int32_t mul = B * H;  // #2777
    const int32_t mul_2 = (mul * 64) * 128;  // #2778 #2779
    const int32_t mul_178 = mul_2 * 2;  // #2780
    const GMTensor<bfloat16_t> kc((__gm__ bfloat16_t*)(workspace + (0)));  // #2781
    const int32_t add_188 = mul_178 + mul_178;  // #2786
    const GMTensor<bfloat16_t> vc((__gm__ bfloat16_t*)(workspace + (mul_178)));  // #2787
    const int32_t add_189 = add_188 + 256;  // #2788
    const GMTensor<float> last_group_lens((__gm__ float*)(workspace + (add_188)));  // #2789
    const int32_t mul_7 = mul * NT_Q_PAD8;  // #2791
    const int32_t add_190 = add_189 + (mul_7 * 8);  // #2792 #2793
    const GMTensor<uint64_t> approx_mask((__gm__ uint64_t*)(workspace + (add_189)));  // #2794
    const int32_t add_191 = add_190 + (mul_7 * 4);  // #2797 #2798
    const GMTensor<int32_t> exact_counts((__gm__ int32_t*)(workspace + (add_190)));  // #2799
    const int32_t mul_12 = (mul * NT_Q) * 64;  // #2801 #2802
    const int32_t add_192 = add_191 + (mul_12 * 2);  // #2803 #2804
    const GMTensor<uint16_t> exact_indices((__gm__ uint16_t*)(workspace + (add_191)));  // #2805
    const bool dense = NT <= 2;  // #2806
    if (!(dense)) {  // #2807 #2808
        const Tensor<uint8_t, Position::UB> storage = arena;  // #24
        const Tensor<bfloat16_t, Position::UB> view = arena.as<bfloat16_t>();  // #25
        const Tensor<bfloat16_t, Position::UB> q_block = arena.as<bfloat16_t>();  // #26
        const Tensor<uint8_t, Position::UB> storage_1 = arena[16384];  // #27
        const Tensor<bfloat16_t, Position::UB> view_1 = arena.as<bfloat16_t>()[8192];  // #28
        const Tensor<bfloat16_t, Position::UB> k_block = arena.as<bfloat16_t>()[8192];  // #29
        const Tensor<uint8_t, Position::UB> storage_2 = arena[32768];  // #30
        const Tensor<bfloat16_t, Position::UB> view_2 = arena.as<bfloat16_t>()[16384];  // #31
        const Tensor<bfloat16_t, Position::UB> v_block = arena.as<bfloat16_t>()[16384];  // #32
        const Tensor<uint8_t, Position::UB> storage_3 = arena[49152];  // #33
        const Tensor<float, Position::UB> view_3 = arena.as<float>()[12288];  // #34
        const Tensor<float, Position::UB> sums = arena.as<float>()[12288];  // #35
        const Tensor<uint8_t, Position::UB> storage_4 = arena[81920];  // #36
        const Tensor<float, Position::UB> view_4 = arena.as<float>()[20480];  // #37
        const Tensor<float, Position::UB> q_rows = arena.as<float>()[20480];  // #38
        const Tensor<uint8_t, Position::UB> storage_5 = arena[114688];  // #39
        const Tensor<bfloat16_t, Position::UB> view_5 = arena.as<bfloat16_t>()[57344];  // #40
        const Tensor<bfloat16_t, Position::UB> k_rows = arena.as<bfloat16_t>()[57344];  // #41
        const Tensor<uint8_t, Position::UB> storage_6 = arena[131072];  // #42
        const Tensor<bfloat16_t, Position::UB> view_6 = arena.as<bfloat16_t>()[65536];  // #43
        const Tensor<bfloat16_t, Position::UB> v_rows = arena.as<bfloat16_t>()[65536];  // #44
        const Tensor<uint8_t, Position::UB> storage_7 = arena[147456];  // #45
        const Tensor<float, Position::UB> view_7 = arena.as<float>()[36864];  // #46
        const Tensor<float, Position::UB> keys = arena.as<float>()[36864];  // #47
        const Tensor<uint8_t, Position::UB> storage_8 = arena[180224];  // #48
        const Tensor<float, Position::UB> view_8 = arena.as<float>()[45056];  // #49
        const Tensor<float, Position::UB> mean_k = arena.as<float>()[45056];  // #50
        const Tensor<uint8_t, Position::UB> storage_9 = arena[180736];  // #51
        const Tensor<float, Position::UB> view_9 = arena.as<float>()[45184];  // #52
        const Tensor<float, Position::UB> var_k = arena.as<float>()[45184];  // #53
        const Tensor<uint8_t, Position::UB> storage_10 = arena[181248];  // #54
        const Tensor<float, Position::UB> view_10 = arena.as<float>()[45312];  // #55
        const Tensor<float, Position::UB> moment_k = arena.as<float>()[45312];  // #56
        const Tensor<uint8_t, Position::UB> storage_11 = arena[181760];  // #57
        const Tensor<float, Position::UB> view_11 = arena.as<float>()[45440];  // #58
        const Tensor<float, Position::UB> query_mean = arena.as<float>()[45440];  // #59
        const Tensor<uint8_t, Position::UB> storage_12 = arena[182016];  // #60
        const Tensor<float, Position::UB> view_12 = arena.as<float>()[45504];  // #61
        const Tensor<float, Position::UB> query_var = arena.as<float>()[45504];  // #62
        const Tensor<uint8_t, Position::UB> storage_13 = arena[182272];  // #63
        const Tensor<float, Position::UB> view_13 = arena.as<float>()[45568];  // #64
        const Tensor<float, Position::UB> partial = arena.as<float>()[45568];  // #65
        const Tensor<uint8_t, Position::UB> storage_14 = arena[182528];  // #66
        const Tensor<float, Position::UB> view_14 = arena.as<float>()[45632];  // #67
        const Tensor<float, Position::UB> lens = arena.as<float>()[45632];  // #68
        const Tensor<uint8_t, Position::UB> storage_15 = arena[182784];  // #69
        const Tensor<float, Position::UB> view_15 = arena.as<float>()[45696];  // #70
        const Tensor<float, Position::UB> scores = arena.as<float>()[45696];  // #71
        const Tensor<uint8_t, Position::UB> storage_16 = arena[183040];  // #72
        const Tensor<uint8_t, Position::UB> view_16 = arena[183040];  // #73
        const Tensor<uint8_t, Position::UB> cmp = arena[183040];  // #74
        const Tensor<uint64_t, Position::UB> cmp_bits = arena.as<uint64_t>()[22880];  // #75
        const Tensor<uint8_t, Position::UB> storage_17 = arena[183072];  // #76
        const Tensor<uint64_t, Position::UB> view_17 = arena.as<uint64_t>()[22884];  // #77
        const Tensor<uint64_t, Position::UB> route_pattern = arena.as<uint64_t>()[22884];  // #78
        const Tensor<uint8_t, Position::UB> storage_18 = arena[183104];  // #79
        const Tensor<uint16_t, Position::UB> view_18 = arena.as<uint16_t>()[91552];  // #80
        const Tensor<uint16_t, Position::UB> source_indices = arena.as<uint16_t>()[91552];  // #81
        const Tensor<uint8_t, Position::UB> storage_19 = arena[183360];  // #82
        const Tensor<uint64_t, Position::UB> view_19 = arena.as<uint64_t>()[22920];  // #83
        const Tensor<uint64_t, Position::UB> route_masks = arena.as<uint64_t>()[22920];  // #84
        const Tensor<uint8_t, Position::UB> storage_20 = arena[183872];  // #85
        const Tensor<int32_t, Position::UB> view_20 = arena.as<int32_t>()[45968];  // #86
        const Tensor<int32_t, Position::UB> counts = arena.as<int32_t>()[45968];  // #87
        const Tensor<uint16_t, Position::UB> compact_indices((uint64_t)(194560));  // #88
        SEvent<PIPE_S, PIPE_V, 0, 0> scalar_publish;  // #89
        for (int32_t i = 0; i < 128; i += 1) {  // #2251
            const Tensor<uint16_t, Position::UB> point = arena.as<uint16_t>()[(91552 + i)];  // #91
            point.store(0, (uint16_t)((uint16_t)(i)));  // #90 #92
        }
        scalar_publish.set();  // #95
        scalar_publish.wait();  // #96
        const uint64_t valid = 18446744073709551615ULL >> (64 - NT);  // #98 #99
        const int32_t sub_1 = H + -1;  // #100
        const int32_t src_gap = sub_1 * 128;  // #101
        const float inv_nt = 1.0f / (float)(NT);  // #102 #103
        const int32_t vec_id = GetVecIdx();  // #104
        const int32_t mul_13 = B * H;  // #105
        const int32_t _launch_cubes = GetCubeNum();  // #2250
        const int32_t vec_count = _launch_cubes * 2;  // #106
        for (int32_t bh = vec_id; bh < mul_13; bh += vec_count) {  // #2266
            const int32_t idiv_25 = bh / H;  // #4427
            const int32_t idiv_26 = bh % H;  // #4428
            const bool idiv_27 = idiv_26 != 0;  // #4429
            const bool idiv_28 = bh < 0;  // #4430
            const bool idiv_30 = idiv_28 != false;  // #4432
            const bool idiv_31 = idiv_27 && idiv_30;  // #4433
            const int32_t idiv_32 = (idiv_31) ? (1) : (0);  // #4434
            const int32_t b = idiv_25 - idiv_32;  // #107
            const int32_t h = ascrip::FloorMod<int32_t>(bh, H);  // #108
            ev_v_mte2_valid_d0_1.wait();  // #2727
            ev_mte3_v_valid_1.wait();  // #2751
            if (NT < 64) {  // #109 #115
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #110
                vector_dup(q_rows.ptr(), (float)(0.0f), (uint8_t)(128), (uint16_t)1, 1, (uint8_t)8, 0);  // #110
                const Tensor<uint32_t, Position::UB> view_21 = arena.as<uint32_t>()[28672];  // #111
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #112
                vector_dup(view_21.ptr(), (uint32_t)(0U), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)8, 0);  // #112
                const Tensor<uint32_t, Position::UB> view_22 = arena.as<uint32_t>()[32768];  // #113
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #114
                vector_dup(view_22.ptr(), (uint32_t)(0U), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)8, 0);  // #114
            }
            ev_mte3_mte2_valid_1.wait();  // #2715
            ev_v_mte2_valid_d0_1.set();  // #2728
            ev_v_mte2_valid_d0_1.wait();  // #2729
            for (int32_t block = 0; block < NT; block += 1) {  // #2252
                const int32_t tok = block * 64;  // #116
                const int32_t rows = Min(T - tok, 64);  // #117 #118
                ev_v_mte2_valid_d1_1.wait();  // #2733
                if (rows < 64) {  // #119 #126
                    const Tensor<uint32_t, Position::UB> view_23 = arena.as<uint32_t>();  // #120
                    PipeBarrier<PIPE_V>();  // c220: V-V hazard #121
                    vector_dup(view_23.ptr(), (uint32_t)(0U), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)8, 0);  // #121
                    const Tensor<uint32_t, Position::UB> view_24 = arena.as<uint32_t>()[4096];  // #122
                    PipeBarrier<PIPE_V>();  // c220: V-V hazard #123
                    vector_dup(view_24.ptr(), (uint32_t)(0U), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)8, 0);  // #123
                    const Tensor<uint32_t, Position::UB> view_25 = arena.as<uint32_t>()[8192];  // #124
                    PipeBarrier<PIPE_V>();  // c220: V-V hazard #125
                    vector_dup(view_25.ptr(), (uint32_t)(0U), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)8, 0);  // #125
                }
                const GMTensor<bfloat16_t> view_26 = q[(((b * ((T * H) * 128)) + (tok * (H * 128))) + (h * 128))];  // #129
                const int32_t mul_149 = src_gap * 2;  // #2678
                ev_v_mte2_valid_d1_1.set();  // #2734
                ev_v_mte2_valid_d1_1.wait();  // #2735
                gm_to_ub_pad(q_block, view_26, rows, 256, mul_149, 0);  // #130
                const GMTensor<bfloat16_t> view_27 = k[(((b * ((T * H) * 128)) + (tok * (H * 128))) + (h * 128))];  // #133
                gm_to_ub_pad(k_block, view_27, rows, 256, mul_149, 0);  // #134
                const GMTensor<bfloat16_t> view_28 = v[(((b * ((T * H) * 128)) + (tok * (H * 128))) + (h * 128))];  // #137
                gm_to_ub_pad(v_block, view_28, rows, 256, mul_149, 0);  // #138
                const float inv_rows = 1.0f / (float)(rows);  // #139 #140
                ev_mte2_v_ready_d1_1.set();  // #2736
                ev_mte2_v_ready_d1_1.wait();  // #2737
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #141
                vconv_bf162f32(sums.ptr(), q_block.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)4);  // #141
                const Tensor<float, Position::UB> view_29 = arena.as<float>()[12288];  // #142
                const Tensor<float, Position::UB> view_30 = arena.as<float>()[12288];  // #143
                const Tensor<float, Position::UB> view_31 = arena.as<float>()[16384];  // #144
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #145
                vadd(view_29.ptr(), view_30.ptr(), view_31.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #145
                const Tensor<float, Position::UB> view_32 = arena.as<float>()[12288];  // #146
                const Tensor<float, Position::UB> view_33 = arena.as<float>()[12288];  // #147
                const Tensor<float, Position::UB> view_34 = arena.as<float>()[14336];  // #148
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #149
                vadd(view_32.ptr(), view_33.ptr(), view_34.ptr(), (uint8_t)(32), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #149
                const Tensor<float, Position::UB> view_35 = arena.as<float>()[12288];  // #150
                const Tensor<float, Position::UB> view_36 = arena.as<float>()[12288];  // #151
                const Tensor<float, Position::UB> view_37 = arena.as<float>()[13312];  // #152
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #153
                vadd(view_35.ptr(), view_36.ptr(), view_37.ptr(), (uint8_t)(16), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #153
                const Tensor<float, Position::UB> view_38 = arena.as<float>()[12288];  // #154
                const Tensor<float, Position::UB> view_39 = arena.as<float>()[12288];  // #155
                const Tensor<float, Position::UB> view_40 = arena.as<float>()[12800];  // #156
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #157
                vadd(view_38.ptr(), view_39.ptr(), view_40.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #157
                const Tensor<float, Position::UB> view_41 = arena.as<float>()[12288];  // #158
                const Tensor<float, Position::UB> view_42 = arena.as<float>()[12288];  // #159
                const Tensor<float, Position::UB> view_43 = arena.as<float>()[12544];  // #160
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #161
                vadd(view_41.ptr(), view_42.ptr(), view_43.ptr(), (uint8_t)(4), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #161
                const Tensor<float, Position::UB> view_44 = arena.as<float>()[12288];  // #162
                const Tensor<float, Position::UB> view_45 = arena.as<float>()[12288];  // #163
                const Tensor<float, Position::UB> view_46 = arena.as<float>()[12416];  // #164
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #165
                vadd(view_44.ptr(), view_45.ptr(), view_46.ptr(), (uint8_t)(2), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #165
                const Tensor<float, Position::UB> view_47 = arena.as<float>()[(20480 + (block * 128))];  // #168
                const Tensor<float, Position::UB> view_48 = arena.as<float>()[12288];  // #169
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #172
                vmuls(view_47.ptr(), view_48.ptr(), (float)(inv_rows), (uint8_t)(2), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #172
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #173
                vconv_bf162f32(sums.ptr(), k_block.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)4);  // #173
                const Tensor<float, Position::UB> view_49 = arena.as<float>()[12288];  // #174
                const Tensor<float, Position::UB> view_50 = arena.as<float>()[12288];  // #175
                const Tensor<float, Position::UB> view_51 = arena.as<float>()[16384];  // #176
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #177
                vadd(view_49.ptr(), view_50.ptr(), view_51.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #177
                const Tensor<float, Position::UB> view_52 = arena.as<float>()[12288];  // #178
                const Tensor<float, Position::UB> view_53 = arena.as<float>()[12288];  // #179
                const Tensor<float, Position::UB> view_54 = arena.as<float>()[14336];  // #180
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #181
                vadd(view_52.ptr(), view_53.ptr(), view_54.ptr(), (uint8_t)(32), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #181
                const Tensor<float, Position::UB> view_55 = arena.as<float>()[12288];  // #182
                const Tensor<float, Position::UB> view_56 = arena.as<float>()[12288];  // #183
                const Tensor<float, Position::UB> view_57 = arena.as<float>()[13312];  // #184
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #185
                vadd(view_55.ptr(), view_56.ptr(), view_57.ptr(), (uint8_t)(16), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #185
                const Tensor<float, Position::UB> view_58 = arena.as<float>()[12288];  // #186
                const Tensor<float, Position::UB> view_59 = arena.as<float>()[12288];  // #187
                const Tensor<float, Position::UB> view_60 = arena.as<float>()[12800];  // #188
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #189
                vadd(view_58.ptr(), view_59.ptr(), view_60.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #189
                const Tensor<float, Position::UB> view_61 = arena.as<float>()[12288];  // #190
                const Tensor<float, Position::UB> view_62 = arena.as<float>()[12288];  // #191
                const Tensor<float, Position::UB> view_63 = arena.as<float>()[12544];  // #192
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #193
                vadd(view_61.ptr(), view_62.ptr(), view_63.ptr(), (uint8_t)(4), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #193
                const Tensor<float, Position::UB> view_64 = arena.as<float>()[12288];  // #194
                const Tensor<float, Position::UB> view_65 = arena.as<float>()[12288];  // #195
                const Tensor<float, Position::UB> view_66 = arena.as<float>()[12416];  // #196
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #197
                vadd(view_64.ptr(), view_65.ptr(), view_66.ptr(), (uint8_t)(2), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #197
                const Tensor<float, Position::UB> view_67 = arena.as<float>()[12288];  // #198
                const Tensor<float, Position::UB> view_68 = arena.as<float>()[12288];  // #199
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #200
                vmuls(view_67.ptr(), view_68.ptr(), (float)(inv_rows), (uint8_t)(2), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #200
                const Tensor<bfloat16_t, Position::UB> view_69 = arena.as<bfloat16_t>()[(57344 + (block * 128))];  // #203
                const Tensor<float, Position::UB> view_70 = arena.as<float>()[12288];  // #204
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #207
                vconv_f322bf16r(view_69.ptr(), view_70.ptr(), (uint8_t)(2), (uint16_t)1, (uint16_t)1, (uint16_t)4, (uint16_t)8);  // #207
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #208
                vconv_bf162f32(sums.ptr(), v_block.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)4);  // #208
                const Tensor<float, Position::UB> view_71 = arena.as<float>()[12288];  // #209
                const Tensor<float, Position::UB> view_72 = arena.as<float>()[12288];  // #210
                const Tensor<float, Position::UB> view_73 = arena.as<float>()[16384];  // #211
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #212
                vadd(view_71.ptr(), view_72.ptr(), view_73.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #212
                const Tensor<float, Position::UB> view_74 = arena.as<float>()[12288];  // #213
                const Tensor<float, Position::UB> view_75 = arena.as<float>()[12288];  // #214
                const Tensor<float, Position::UB> view_76 = arena.as<float>()[14336];  // #215
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #216
                vadd(view_74.ptr(), view_75.ptr(), view_76.ptr(), (uint8_t)(32), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #216
                const Tensor<float, Position::UB> view_77 = arena.as<float>()[12288];  // #217
                const Tensor<float, Position::UB> view_78 = arena.as<float>()[12288];  // #218
                const Tensor<float, Position::UB> view_79 = arena.as<float>()[13312];  // #219
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #220
                vadd(view_77.ptr(), view_78.ptr(), view_79.ptr(), (uint8_t)(16), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #220
                const Tensor<float, Position::UB> view_80 = arena.as<float>()[12288];  // #221
                const Tensor<float, Position::UB> view_81 = arena.as<float>()[12288];  // #222
                const Tensor<float, Position::UB> view_82 = arena.as<float>()[12800];  // #223
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #224
                vadd(view_80.ptr(), view_81.ptr(), view_82.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #224
                const Tensor<float, Position::UB> view_83 = arena.as<float>()[12288];  // #225
                const Tensor<float, Position::UB> view_84 = arena.as<float>()[12288];  // #226
                const Tensor<float, Position::UB> view_85 = arena.as<float>()[12544];  // #227
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #228
                vadd(view_83.ptr(), view_84.ptr(), view_85.ptr(), (uint8_t)(4), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #228
                const Tensor<float, Position::UB> view_86 = arena.as<float>()[12288];  // #229
                const Tensor<float, Position::UB> view_87 = arena.as<float>()[12288];  // #230
                const Tensor<float, Position::UB> view_88 = arena.as<float>()[12416];  // #231
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #232
                vadd(view_86.ptr(), view_87.ptr(), view_88.ptr(), (uint8_t)(2), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #232
                const Tensor<bfloat16_t, Position::UB> view_89 = arena.as<bfloat16_t>()[(65536 + (block * 128))];  // #235
                const Tensor<float, Position::UB> view_90 = arena.as<float>()[12288];  // #236
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #239
                vconv_f322bf16r(view_89.ptr(), view_90.ptr(), (uint8_t)(2), (uint16_t)1, (uint16_t)1, (uint16_t)4, (uint16_t)8);  // #239
                ev_v_mte2_valid_d1_1.set();  // #2741
            }
            ev_mte2_v_ready_d0_1.set();  // #2743
            ev_mte2_v_ready_d0_1.wait();  // #2744
            const GMTensor<bfloat16_t> view_91 = kc[((b * ((H * 64) * 128)) + (h * 8192))];  // #242
            ev_mte2_mte3_ready_1.set();  // #2716
            ev_mte2_mte3_ready_1.wait();  // #2717
            ev_v_mte3_ready_1.set();  // #2752
            ev_v_mte3_ready_1.wait();  // #2753
            ub_to_gm_pad(view_91, k_rows, 64, 256, 0, 0);  // #243
            const GMTensor<bfloat16_t> view_92 = vc[((b * ((H * 64) * 128)) + (h * 8192))];  // #244
            ub_to_gm_pad(view_92, v_rows, 64, 256, 0, 0);  // #245
            ev_mte3_v_valid_1.set();  // #2754
            ev_mte3_v_valid_1.wait();  // #2755
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #246
            vconv_bf162f32(sums.ptr(), k_rows.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)4);  // #246
            const Tensor<float, Position::UB> view_93 = arena.as<float>()[12288];  // #247
            const Tensor<float, Position::UB> view_94 = arena.as<float>()[12288];  // #248
            const Tensor<float, Position::UB> view_95 = arena.as<float>()[16384];  // #249
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #250
            vadd(view_93.ptr(), view_94.ptr(), view_95.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #250
            const Tensor<float, Position::UB> view_96 = arena.as<float>()[12288];  // #251
            const Tensor<float, Position::UB> view_97 = arena.as<float>()[12288];  // #252
            const Tensor<float, Position::UB> view_98 = arena.as<float>()[14336];  // #253
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #254
            vadd(view_96.ptr(), view_97.ptr(), view_98.ptr(), (uint8_t)(32), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #254
            const Tensor<float, Position::UB> view_99 = arena.as<float>()[12288];  // #255
            const Tensor<float, Position::UB> view_100 = arena.as<float>()[12288];  // #256
            const Tensor<float, Position::UB> view_101 = arena.as<float>()[13312];  // #257
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #258
            vadd(view_99.ptr(), view_100.ptr(), view_101.ptr(), (uint8_t)(16), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #258
            const Tensor<float, Position::UB> view_102 = arena.as<float>()[12288];  // #259
            const Tensor<float, Position::UB> view_103 = arena.as<float>()[12288];  // #260
            const Tensor<float, Position::UB> view_104 = arena.as<float>()[12800];  // #261
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #262
            vadd(view_102.ptr(), view_103.ptr(), view_104.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #262
            const Tensor<float, Position::UB> view_105 = arena.as<float>()[12288];  // #263
            const Tensor<float, Position::UB> view_106 = arena.as<float>()[12288];  // #264
            const Tensor<float, Position::UB> view_107 = arena.as<float>()[12544];  // #265
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #266
            vadd(view_105.ptr(), view_106.ptr(), view_107.ptr(), (uint8_t)(4), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #266
            const Tensor<float, Position::UB> view_108 = arena.as<float>()[12288];  // #267
            const Tensor<float, Position::UB> view_109 = arena.as<float>()[12288];  // #268
            const Tensor<float, Position::UB> view_110 = arena.as<float>()[12416];  // #269
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #270
            vadd(view_108.ptr(), view_109.ptr(), view_110.ptr(), (uint8_t)(2), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #270
            const Tensor<float, Position::UB> view_111 = arena.as<float>()[12288];  // #271
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #272
            vmuls(mean_k.ptr(), view_111.ptr(), (float)(inv_nt), (uint8_t)(2), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #272
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #273
            vconv_bf162f32(sums.ptr(), k_rows.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)4);  // #273
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #274
            vmul(sums.ptr(), sums.ptr(), sums.ptr(), (uint8_t)(128), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #274
            const Tensor<float, Position::UB> view_112 = arena.as<float>()[12288];  // #275
            const Tensor<float, Position::UB> view_113 = arena.as<float>()[12288];  // #276
            const Tensor<float, Position::UB> view_114 = arena.as<float>()[16384];  // #277
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #278
            vadd(view_112.ptr(), view_113.ptr(), view_114.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #278
            const Tensor<float, Position::UB> view_115 = arena.as<float>()[12288];  // #279
            const Tensor<float, Position::UB> view_116 = arena.as<float>()[12288];  // #280
            const Tensor<float, Position::UB> view_117 = arena.as<float>()[14336];  // #281
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #282
            vadd(view_115.ptr(), view_116.ptr(), view_117.ptr(), (uint8_t)(32), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #282
            const Tensor<float, Position::UB> view_118 = arena.as<float>()[12288];  // #283
            const Tensor<float, Position::UB> view_119 = arena.as<float>()[12288];  // #284
            const Tensor<float, Position::UB> view_120 = arena.as<float>()[13312];  // #285
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #286
            vadd(view_118.ptr(), view_119.ptr(), view_120.ptr(), (uint8_t)(16), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #286
            const Tensor<float, Position::UB> view_121 = arena.as<float>()[12288];  // #287
            const Tensor<float, Position::UB> view_122 = arena.as<float>()[12288];  // #288
            const Tensor<float, Position::UB> view_123 = arena.as<float>()[12800];  // #289
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #290
            vadd(view_121.ptr(), view_122.ptr(), view_123.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #290
            const Tensor<float, Position::UB> view_124 = arena.as<float>()[12288];  // #291
            const Tensor<float, Position::UB> view_125 = arena.as<float>()[12288];  // #292
            const Tensor<float, Position::UB> view_126 = arena.as<float>()[12544];  // #293
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #294
            vadd(view_124.ptr(), view_125.ptr(), view_126.ptr(), (uint8_t)(4), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #294
            const Tensor<float, Position::UB> view_127 = arena.as<float>()[12288];  // #295
            const Tensor<float, Position::UB> view_128 = arena.as<float>()[12288];  // #296
            const Tensor<float, Position::UB> view_129 = arena.as<float>()[12416];  // #297
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #298
            vadd(view_127.ptr(), view_128.ptr(), view_129.ptr(), (uint8_t)(2), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #298
            const Tensor<float, Position::UB> view_130 = arena.as<float>()[12288];  // #299
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #300
            vmuls(moment_k.ptr(), view_130.ptr(), (float)(inv_nt), (uint8_t)(2), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #300
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #301
            vmul(var_k.ptr(), mean_k.ptr(), mean_k.ptr(), (uint8_t)(2), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #301
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #302
            vsub(var_k.ptr(), moment_k.ptr(), var_k.ptr(), (uint8_t)(2), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #302
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #303
            vmaxs(var_k.ptr(), var_k.ptr(), (float)(0.0f), (uint8_t)(2), (uint16_t)1, (uint16_t)1, (uint8_t)8, (uint8_t)8, false, false);  // #303
            const Tensor<float, Position::UB> view_131 = arena.as<float>()[12288];  // #304
            const Tensor<float, Position::UB> view_132 = arena.as<float>()[20480];  // #305
            const Tensor<float, Position::UB> view_133 = arena.as<float>()[45056];  // #306
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #307
            vmul(view_131.ptr(), view_132.ptr(), view_133.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)16, (uint8_t)16, (uint8_t)0);  // #307
            const Tensor<float, Position::UB> view_134 = arena.as<float>()[12352];  // #308
            const Tensor<float, Position::UB> view_135 = arena.as<float>()[20544];  // #309
            const Tensor<float, Position::UB> view_136 = arena.as<float>()[45120];  // #310
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #311
            vmul(view_134.ptr(), view_135.ptr(), view_136.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)16, (uint8_t)16, (uint8_t)0);  // #311
            const Tensor<float, Position::UB> view_137 = arena.as<float>()[12288];  // #312
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #313
            vcadd(query_mean.ptr(), view_137.ptr(), (uint8_t)(64), 1, 1, 16, 0);  // #313
            const Tensor<float, Position::UB> view_138 = arena.as<float>()[12352];  // #314
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #315
            vcadd(partial.ptr(), view_138.ptr(), (uint8_t)(64), 1, 1, 16, 0);  // #315
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #316
            vadd(query_mean.ptr(), query_mean.ptr(), partial.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #316
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #317
            vmuls(query_mean.ptr(), query_mean.ptr(), (float)(scale_log2), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #317
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #318
            vmul(sums.ptr(), q_rows.ptr(), q_rows.ptr(), (uint8_t)(128), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #318
            const Tensor<float, Position::UB> view_139 = arena.as<float>()[12288];  // #319
            const Tensor<float, Position::UB> view_140 = arena.as<float>()[12288];  // #320
            const Tensor<float, Position::UB> view_141 = arena.as<float>()[45184];  // #321
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #322
            vmul(view_139.ptr(), view_140.ptr(), view_141.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)16, (uint8_t)16, (uint8_t)0);  // #322
            const Tensor<float, Position::UB> view_142 = arena.as<float>()[12352];  // #323
            const Tensor<float, Position::UB> view_143 = arena.as<float>()[12352];  // #324
            const Tensor<float, Position::UB> view_144 = arena.as<float>()[45248];  // #325
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #326
            vmul(view_142.ptr(), view_143.ptr(), view_144.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)16, (uint8_t)16, (uint8_t)0);  // #326
            const Tensor<float, Position::UB> view_145 = arena.as<float>()[12288];  // #327
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #328
            vcadd(query_var.ptr(), view_145.ptr(), (uint8_t)(64), 1, 1, 16, 0);  // #328
            const Tensor<float, Position::UB> view_146 = arena.as<float>()[12352];  // #329
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #330
            vcadd(partial.ptr(), view_146.ptr(), (uint8_t)(64), 1, 1, 16, 0);  // #330
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #331
            vadd(query_var.ptr(), query_var.ptr(), partial.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #331
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #332
            vmuls(query_var.ptr(), query_var.ptr(), (float)(scale_sq), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #332
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #333
            vmaxs(query_var.ptr(), query_var.ptr(), (float)(0.0f), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint8_t)0, (uint8_t)0, false, false);  // #333
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #334
            vadds(query_var.ptr(), query_var.ptr(), (float)(1e-06f), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #334
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #335
            vsqrt(query_var.ptr(), query_var.ptr(), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint8_t)0, (uint8_t)0);  // #335
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #336
            vmuls(query_var.ptr(), query_var.ptr(), (float)(tau), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #336
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #337
            vadd(query_mean.ptr(), query_mean.ptr(), query_var.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #337
            PipeBarrier<PIPE_ALL>();  // #339
            vconv_bf162f32(keys.ptr(), k_rows.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)4);  // #340
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #341
            vector_dup(counts.ptr(), (int32_t)(0), (uint8_t)(1), (uint16_t)1, 1, (uint8_t)0, 0);  // #341
            SetFlag<PIPE_V, PIPE_S>(6);  // #342
            WaitFlag<PIPE_V, PIPE_S>(6);  // #343
            for (int32_t qi = 0; qi < NT_Q; qi += 1) {  // #2265
                const int32_t q0 = qi * 2;  // #344
                const int32_t q1 = Min((q0 + 2) - 1, NT - 1);  // #345 #346 #347 #348
                const Tensor<float, Position::UB> point_1 = arena.as<float>()[(45440 + q0)];  // #349
                const float th0 = point_1.load(0);  // #350
                const Tensor<float, Position::UB> view_147 = arena.as<float>()[(20480 + (q0 * 128))];  // #353
                const Tensor<float, Position::UB> view_148 = arena.as<float>()[12288];  // #354
                const Tensor<float, Position::UB> view_149 = arena.as<float>()[36864];  // #355
                const Tensor<float, Position::UB> view_150 = arena.as<float>()[(20480 + (q0 * 128))];  // #356
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #357
                vmul(view_148.ptr(), view_149.ptr(), view_150.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)16, (uint8_t)16, (uint8_t)0);  // #357
                const Tensor<float, Position::UB> view_151 = arena.as<float>()[12352];  // #358
                const Tensor<float, Position::UB> view_152 = arena.as<float>()[36928];  // #359
                const Tensor<float, Position::UB> view_153 = arena.as<float>()[(20480 + ((q0 * 128) + 64))];  // #360
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #361
                vmul(view_151.ptr(), view_152.ptr(), view_153.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)16, (uint8_t)16, (uint8_t)0);  // #361
                const Tensor<float, Position::UB> view_154 = arena.as<float>()[12288];  // #362
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #363
                vcadd(scores.ptr(), view_154.ptr(), (uint8_t)(64), 1, 1, 16, 0);  // #363
                const Tensor<float, Position::UB> view_155 = arena.as<float>()[12352];  // #364
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #365
                vcadd(partial.ptr(), view_155.ptr(), (uint8_t)(64), 1, 1, 16, 0);  // #365
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #366
                vadd(scores.ptr(), scores.ptr(), partial.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #366
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #367
                vmuls(scores.ptr(), scores.ptr(), (float)(scale_log2), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #367
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #368
                vcmpvs_gt((__ubuf__ uint8_t*)(cmp.ptr()), scores.ptr(), (float)(th0), (uint8_t)(1), (uint16_t)0, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #368
                SetFlag<PIPE_V, PIPE_S>(6);  // #369
                WaitFlag<PIPE_V, PIPE_S>(6);  // #370
                const Tensor<uint64_t, Position::UB> point_2 = arena.as<uint64_t>()[22880];  // #371
                const uint64_t exact = point_2.load(0);  // #372
                const bool cmp_3 = q1 != q0;  // #373
                uint64_t exact_2_state = 0ULL;  // #2253
                if (cmp_3) {  // #401
                    const Tensor<float, Position::UB> point_3 = arena.as<float>()[(45440 + q1)];  // #374
                    const float th1 = point_3.load(0);  // #375
                    const Tensor<float, Position::UB> view_156 = arena.as<float>()[(20480 + (q1 * 128))];  // #378
                    const Tensor<float, Position::UB> view_157 = arena.as<float>()[12288];  // #379
                    const Tensor<float, Position::UB> view_158 = arena.as<float>()[36864];  // #380
                    const Tensor<float, Position::UB> view_159 = arena.as<float>()[(20480 + (q1 * 128))];  // #381
                    PipeBarrier<PIPE_V>();  // c220: V-V hazard #382
                    vmul(view_157.ptr(), view_158.ptr(), view_159.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)16, (uint8_t)16, (uint8_t)0);  // #382
                    const Tensor<float, Position::UB> view_160 = arena.as<float>()[12352];  // #383
                    const Tensor<float, Position::UB> view_161 = arena.as<float>()[36928];  // #384
                    const Tensor<float, Position::UB> view_162 = arena.as<float>()[(20480 + ((q1 * 128) + 64))];  // #385
                    PipeBarrier<PIPE_V>();  // c220: V-V hazard #386
                    vmul(view_160.ptr(), view_161.ptr(), view_162.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)16, (uint8_t)16, (uint8_t)0);  // #386
                    const Tensor<float, Position::UB> view_163 = arena.as<float>()[12288];  // #387
                    PipeBarrier<PIPE_V>();  // c220: V-V hazard #388
                    vcadd(scores.ptr(), view_163.ptr(), (uint8_t)(64), 1, 1, 16, 0);  // #388
                    const Tensor<float, Position::UB> view_164 = arena.as<float>()[12352];  // #389
                    PipeBarrier<PIPE_V>();  // c220: V-V hazard #390
                    vcadd(partial.ptr(), view_164.ptr(), (uint8_t)(64), 1, 1, 16, 0);  // #390
                    PipeBarrier<PIPE_V>();  // c220: V-V hazard #391
                    vadd(scores.ptr(), scores.ptr(), partial.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #391
                    PipeBarrier<PIPE_V>();  // c220: V-V hazard #392
                    vmuls(scores.ptr(), scores.ptr(), (float)(scale_log2), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #392
                    PipeBarrier<PIPE_V>();  // c220: V-V hazard #393
                    vcmpvs_gt((__ubuf__ uint8_t*)(cmp.ptr()), scores.ptr(), (float)(th1), (uint8_t)(1), (uint16_t)0, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #393
                    SetFlag<PIPE_V, PIPE_S>(6);  // #394
                    WaitFlag<PIPE_V, PIPE_S>(6);  // #395
                    const Tensor<uint64_t, Position::UB> point_4 = arena.as<uint64_t>()[22880];  // #396
                    exact_2_state = exact | point_4.load(0);  // #397 #398 #2254
                } else {
                    exact_2_state = exact;  // #2255
                }
                const uint64_t exact_2 = (uint64_t)(exact_2_state);  // #2256
                const int32_t lo = Max(q0 - 1, 0);  // #402 #405
                const int32_t hi = Min(q1 + 2, 64);  // #403 #407
                const bool cmp_4 = lo < hi;  // #409
                uint64_t mask_2_state = 0ULL;  // #2257
                if (cmp_4) {  // #421
                    mask_2_state = (18446744073709551615ULL >> (64 - hi)) & ~((1ULL << lo) - 1ULL);  // #411 #412 #414 #416 #417 #418 #2258
                } else {
                    mask_2_state = 0ULL;  // #2259
                }
                const uint64_t mask_2 = (uint64_t)(mask_2_state);  // #2260
                const uint64_t or_ = exact_2 | mask_2;  // #422
                const int32_t lo_1 = Max(sink_start, 0);  // #424
                const int32_t hi_1 = Min(sink_end, 64);  // #426
                const bool cmp_5 = lo_1 < hi_1;  // #428
                uint64_t mask_5_state = 0ULL;  // #2261
                if (cmp_5) {  // #440
                    mask_5_state = (18446744073709551615ULL >> (64 - hi_1)) & ~((1ULL << lo_1) - 1ULL);  // #430 #431 #433 #435 #436 #437 #2262
                } else {
                    mask_5_state = 0ULL;  // #2263
                }
                const uint64_t mask_5 = (uint64_t)(mask_5_state);  // #2264
                const uint64_t exact_3 = (or_ | mask_5) & valid;  // #441 #442
                const Tensor<uint64_t, Position::UB> point_5 = arena.as<uint64_t>()[(22920 + qi)];  // #445
                point_5.store(0, (uint64_t)(valid & ~(exact_3)));  // #443 #444 #446
                const Tensor<uint64_t, Position::UB> point_6 = arena.as<uint64_t>()[22884];  // #447
                point_6.store(0, (uint64_t)(exact_3));  // #448
                const Tensor<uint64_t, Position::UB> point_7 = arena.as<uint64_t>()[22885];  // #450
                point_7.store(0, (uint64_t)(0ULL));  // #451
                scalar_publish.set();  // #452
                scalar_publish.wait();  // #453
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #454
                SetMaskNorm();  // #454
                ResetMask();  // #454
                vreducev2((__ubuf__ uint16_t*)(compact_indices.ptr()), (__ubuf__ uint16_t*)(source_indices.ptr()), (__ubuf__ uint16_t*)(route_pattern.ptr()), (uint8_t)(1), 1, 0, 8, 1);  // #454
                SetFlag<PIPE_V, PIPE_S>(0);  // #454
                WaitFlag<PIPE_V, PIPE_S>(0);  // #454
                const int32_t num_exact = (int32_t)get_rsvd_cnt();  // #454
                const Tensor<int32_t, Position::UB> point_8 = arena.as<int32_t>()[(45968 + qi)];  // #455
                point_8.store(0, (int32_t)(num_exact));  // #456
                if (num_exact > 0) {  // #457 #464
                    SetFlag<PIPE_V, PIPE_MTE3>(7);  // #458
                    WaitFlag<PIPE_V, PIPE_MTE3>(7);  // #459
                    const GMTensor<uint16_t> view_165 = exact_indices[(((b * ((H * NT_Q) * 64)) + (h * (NT_Q * 64))) + (qi * 64))];  // #460
                    ub_to_gm_pad(view_165, compact_indices, 1, num_exact * 2, 0, 0);  // #2681 #461
                    SetFlag<PIPE_MTE3, PIPE_V>(7);  // #462
                    WaitFlag<PIPE_MTE3, PIPE_V>(7);  // #463
                }
            }
            SetFlag<PIPE_S, PIPE_MTE3>(0);  // #467
            WaitFlag<PIPE_S, PIPE_MTE3>(0);  // #468
            const GMTensor<uint64_t> view_166 = approx_mask[((b * (H * NT_Q_PAD8)) + (h * NT_Q_PAD8))];  // #469
            ub_to_gm_pad(view_166, route_masks, 1, NT_Q * 8, 0, 0);  // #2682 #470
            const GMTensor<int32_t> view_167 = exact_counts[((b * (H * NT_Q_PAD8)) + (h * NT_Q_PAD8))];  // #471
            ub_to_gm_pad(view_167, counts, 1, NT_Q_PAD8 * 4, 0, 0);  // #2683 #472
            SetFlag<PIPE_MTE3, PIPE_S>(0);  // #473
            WaitFlag<PIPE_MTE3, PIPE_S>(0);  // #474
            ev_mte3_mte2_valid_1.set();  // #2721
            ev_v_mte2_valid_d0_1.set();  // #2745
            ev_v_mte3_ready_1.set();  // #2759
            ev_v_mte3_ready_1.wait();  // #2760
            ev_mte3_v_valid_1.set();  // #2761
        }
        const int32_t vec_id_1 = GetVecIdx();  // #477
        if (vec_id_1 == 0) {  // #478 #491
            for (int32_t col = 0; col < 64; col += 1) {  // #2267
                const int32_t length = Min(64, Max(0, T - (col * 64)));  // #479 #480 #481 #482
                const Tensor<float, Position::UB> point_9 = arena.as<float>()[(45632 + col)];  // #484
                point_9.store(0, (float)((float)(length)));  // #483 #485
            }
            SetFlag<PIPE_S, PIPE_MTE3>(0);  // #488
            WaitFlag<PIPE_S, PIPE_MTE3>(0);  // #489
            ub_to_gm_pad(last_group_lens, lens, 1, 256, 0, 0);  // #490
        }
        PipeBarrier<PIPE_ALL>();  // #492
        ALLVEC_READY<PIPE_MTE3>(6);  // #493
        ALLVEC_WAIT<PIPE_S>(6);  // #494
        VEC_READY<PIPE_MTE3>(7);  // #495
    }
    const float scale_natural = scale_log2 * 0.6931471805599453f;  // #2809
    const int32_t add_193 = add_192 + 18874368;  // #2810
    const GMTensor<float> score_ws((__gm__ float*)(workspace + (add_192)));  // #2811
    const GMTensor<bfloat16_t> p_ws((__gm__ bfloat16_t*)(workspace + (add_193)));  // #2813
    const GMTensor<float> pv_ws((__gm__ float*)(workspace + (add_193 + 9437184)));  // #2812 #2815
    const GMTensor<float> score_pairs = score_ws;  // #2816
    const GMTensor<bfloat16_t> p_pairs = p_ws;  // #2817
    const Tensor<uint8_t, Position::UB> storage_21 = arena;  // #2825
    const Tensor<float, Position::UB> view_168 = arena.as<float>();  // #2826
    const Tensor<float, Position::UB> score_tiles = arena.as<float>();  // #2827
    const Tensor<uint8_t, Position::UB> storage_22 = arena[98304];  // #2828
    const Tensor<bfloat16_t, Position::UB> view_169 = arena.as<bfloat16_t>()[49152];  // #2829
    const Tensor<bfloat16_t, Position::UB> probability_tiles = arena.as<bfloat16_t>()[49152];  // #2830
    const Tensor<uint8_t, Position::UB> storage_23 = arena[114688];  // #2831
    const Tensor<float, Position::UB> view_170 = arena.as<float>()[28672];  // #2832
    const Tensor<float, Position::UB> accum_pair = arena.as<float>()[28672];  // #2833
    const Tensor<uint8_t, Position::UB> storage_24 = arena[180224];  // #2834
    const Tensor<float, Position::UB> view_171 = arena.as<float>()[45056];  // #2835
    const Tensor<float, Position::UB> ub_len_last = arena.as<float>()[45056];  // #2836
    const Tensor<uint8_t, Position::UB> storage_25 = arena[180480];  // #2837
    const Tensor<float, Position::UB> view_172 = arena.as<float>()[45120];  // #2838
    const Tensor<float, Position::UB> ub_max_s = arena.as<float>()[45120];  // #2839
    const Tensor<uint8_t, Position::UB> storage_26 = arena[180736];  // #2840
    const Tensor<float, Position::UB> view_173 = arena.as<float>()[45184];  // #2841
    const Tensor<float, Position::UB> ub_bmax_s = arena.as<float>()[45184];  // #2842
    const Tensor<uint8_t, Position::UB> storage_27 = arena[180992];  // #2843
    const Tensor<float, Position::UB> view_174 = arena.as<float>()[45248];  // #2844
    const Tensor<float, Position::UB> ub_old_max_s = arena.as<float>()[45248];  // #2845
    const Tensor<uint8_t, Position::UB> storage_28 = arena[181248];  // #2846
    const Tensor<float, Position::UB> view_175 = arena.as<float>()[45312];  // #2847
    const Tensor<float, Position::UB> ub_rmax_s = arena.as<float>()[45312];  // #2848
    const Tensor<uint8_t, Position::UB> storage_29 = arena[181504];  // #2849
    const Tensor<float, Position::UB> view_176 = arena.as<float>()[45376];  // #2850
    const Tensor<float, Position::UB> ub_sum_s = arena.as<float>()[45376];  // #2851
    const Tensor<uint8_t, Position::UB> storage_30 = arena[181760];  // #2852
    const Tensor<float, Position::UB> view_177 = arena.as<float>()[45440];  // #2853
    const Tensor<float, Position::UB> ub_bsum_s = arena.as<float>()[45440];  // #2854
    const Tensor<uint8_t, Position::UB> storage_31 = arena[182016];  // #2855
    const Tensor<float, Position::UB> view_178 = arena.as<float>()[45504];  // #2856
    const Tensor<float, Position::UB> ub_rsum_s = arena.as<float>()[45504];  // #2857
    const Tensor<uint8_t, Position::UB> storage_32 = arena[182272];  // #2858
    const Tensor<float, Position::UB> view_179 = arena.as<float>()[45568];  // #2859
    const Tensor<float, Position::UB> ub_zero_s = arena.as<float>()[45568];  // #2860
    const Tensor<uint8_t, Position::UB> storage_33 = arena[182528];  // #2861
    const Tensor<float, Position::UB> view_180 = arena.as<float>()[45632];  // #2862
    const Tensor<float, Position::UB> expdiff_buf = arena.as<float>()[45632];  // #2863
    const Tensor<uint8_t, Position::UB> storage_34 = arena[183552];  // #2864
    const Tensor<float, Position::UB> view_181 = arena.as<float>()[45888];  // #2865
    const Tensor<float, Position::UB> ub_max = arena.as<float>()[45888];  // #2866
    const Tensor<uint8_t, Position::UB> storage_35 = arena[185600];  // #2867
    const Tensor<float, Position::UB> view_182 = arena.as<float>()[46400];  // #2868
    const Tensor<float, Position::UB> ub_reduce_pair = arena.as<float>()[46400];  // #2869
    const Tensor<float, Position::UB> ub_reduce = arena.as<float>()[46400];  // #2870
    const Tensor<uint8_t, Position::UB> storage_36 = arena[189696];  // #2871
    const Tensor<float, Position::UB> view_183 = arena.as<float>()[47424];  // #2872
    const Tensor<float, Position::UB> ub_expdiff = arena.as<float>()[47424];  // #2873
    const Tensor<uint8_t, Position::UB> storage_37 = arena[191744];  // #2874
    const Tensor<float, Position::UB> view_184 = arena.as<float>()[47936];  // #2875
    const Tensor<float, Position::UB> ub_rowsum = arena.as<float>()[47936];  // #2876
    VEC_READY<PIPE_MTE2>(0);  // #2877
    VEC_READY<PIPE_MTE2>(0);  // #2877
    // sync.mutex vc id=1: the producer side; its 2 tokens are drained before the return  // #2878
    VEC_READY<PIPE_MTE2>(2);  // #2879
    VEC_READY<PIPE_MTE2>(2);  // #2879
    SEvent<PIPE_MTE2, PIPE_V, 0, 2> lengths_ready;  // #569
    if (!(dense)) {  // #2880 #2881
        gm_to_ub_pad(ub_len_last, last_group_lens, 1, 256, 0, 0);  // #579
        lengths_ready.set();  // #580
        lengths_ready.wait();  // #581
    }
    const int32_t cube_idx = GetCubeIdx();  // #2882
    const int32_t cube_count = GetCubeNum();  // #2883
    const int32_t core_step = (int32_t)(cube_count);  // #2884
    const int32_t sb = GetSubBlockIdx();  // #586
    const int32_t sb_row = sb * 64;  // #587
    const int32_t bh_total = B * H;  // #2885
    const int32_t total_tasks = bh_total * NT_Q;  // #2886
    int32_t l0ab_cnt_17_state = 0;  // #2891
    int32_t l0c_cnt_12_state = 0;  // #2892
    int32_t l1k_cnt_10_state = 0;  // #2893
    int32_t l1pv_cnt_9_state = 0;  // #2894
    for (int32_t linear_id = cube_idx; linear_id < total_tasks; linear_id += core_step) {  // #4397
        const int32_t l0ab_cnt_carry = (int32_t)(l0ab_cnt_17_state);  // #2895
        const int32_t l0c_cnt_carry = (int32_t)(l0c_cnt_12_state);  // #2896
        const int32_t l1k_cnt_carry = (int32_t)(l1k_cnt_10_state);  // #2897
        const int32_t l1pv_cnt_carry = (int32_t)(l1pv_cnt_9_state);  // #2898
        const int32_t idiv_33 = linear_id / NT_Q;  // #4435
        const int32_t idiv_34 = linear_id % NT_Q;  // #4436
        const bool idiv_35 = idiv_34 != 0;  // #4437
        const bool idiv_36 = linear_id < 0;  // #4438
        const bool idiv_37 = NT_Q < 0;  // #4439
        const bool idiv_38 = idiv_36 != idiv_37;  // #4440
        const bool idiv_39 = idiv_35 && idiv_38;  // #4441
        const int32_t idiv_40 = (idiv_39) ? (1) : (0);  // #4442
        const int32_t bh_1 = idiv_33 - idiv_40;  // #2899
        const int32_t q_idx = ascrip::FloorMod<int32_t>(linear_id, NT_Q);  // #2900
        const int32_t idiv_41 = bh_1 / H;  // #4443
        const int32_t idiv_42 = bh_1 % H;  // #4444
        const bool idiv_43 = idiv_42 != 0;  // #4445
        const bool idiv_44 = bh_1 < 0;  // #4446
        const bool idiv_46 = idiv_44 != false;  // #4448
        const bool idiv_47 = idiv_43 && idiv_46;  // #4449
        const int32_t idiv_48 = (idiv_47) ? (1) : (0);  // #4450
        const int32_t b_1 = idiv_41 - idiv_48;  // #2901
        const int32_t h_1 = ascrip::FloorMod<int32_t>(bh_1, H);  // #2902
        const int32_t q_start = q_idx * 128;  // #2903
        const int32_t valid_m = Min(128, T - q_start);  // #2904 #2905
        const int32_t local_valid_m = Min(64, Max(valid_m - sb_row, 0));  // #601 #602 #603
        ev_v_mte2_valid_d0_1.wait();  // #2730
        ev_mte3_v_valid_1.wait();  // #2756
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #617
        vector_dup(ub_zero_s.ptr(), (float)(0.0f), (uint8_t)(1), (uint16_t)1, 1, (uint8_t)0, 0);  // #617
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #618
        vector_dup(ub_max_s.ptr(), (float)(-1e+30f), (uint8_t)(1), (uint16_t)1, 1, (uint8_t)0, 0);  // #618
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #619
        vector_dup(ub_sum_s.ptr(), (float)(0.0f), (uint8_t)(1), (uint16_t)1, 1, (uint8_t)0, 0);  // #619
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #620
        vector_dup(ub_rmax_s.ptr(), (float)(-1e+30f), (uint8_t)(1), (uint16_t)1, 1, (uint8_t)0, 0);  // #620
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #621
        vector_dup(ub_rsum_s.ptr(), (float)(0.0f), (uint8_t)(1), (uint16_t)1, 1, (uint8_t)0, 0);  // #621
        const Tensor<float, Position::UB> tile = arena.as<float>()[28672];  // #2907
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #623
        vector_dup(tile.ptr(), (float)(0.0f), (uint8_t)(128), (uint16_t)1, 1, (uint8_t)8, 0);  // #623
        const bool not_5 = !(dense);  // #2909
        int32_t approx_beats_1_state = 0;  // #2910
        int32_t exact_total_1_state = 0;  // #2911
        if (not_5) {  // #2923
            const int32_t exact_total = exact_counts.load(((b_1 * (H * NT_Q_PAD8)) + (h_1 * NT_Q_PAD8)) + q_idx);  // #2913 #2914 #2915 #2916 #2917 #2918
            approx_beats_1_state = 1;  // #2919
            exact_total_1_state = exact_total;  // #2920
        } else {
            approx_beats_1_state = 0;  // #2921
            exact_total_1_state = NT;  // #2922
        }
        const int32_t approx_beats_1 = (int32_t)(approx_beats_1_state);  // #2924
        const int32_t exact_total_1 = (int32_t)(exact_total_1_state);  // #2925
        const int32_t idiv_49 = exact_total_1 / 6;  // #4451
        const int32_t idiv_50 = exact_total_1 % 6;  // #4452
        const bool idiv_51 = idiv_50 != 0;  // #4453
        const bool idiv_52 = exact_total_1 < 0;  // #4454
        const bool idiv_54 = idiv_52 == false;  // #4456
        const bool idiv_55 = idiv_51 && idiv_54;  // #4457
        const int32_t idiv_56 = (idiv_55) ? (1) : (0);  // #4458
        const int32_t ceil_div_3 = idiv_49 + idiv_56;  // #2926
        const int32_t total_beats = approx_beats_1 + ceil_div_3;  // #2927
        SetFlag<PIPE_V, PIPE_MTE2>(4);  // #639
        SetFlag<PIPE_V, PIPE_MTE2>(5);  // #640
        SetFlag<PIPE_V, PIPE_MTE2>(6);  // #641
        SetFlag<PIPE_MTE3, PIPE_V>(4);  // #642
        const int32_t add_13 = total_beats + 3;  // #2929
        int32_t accum_step_3_state = 0;  // #2930
        int32_t l0ab_cnt_16_state = l0ab_cnt_carry;  // #2931
        int32_t l0c_cnt_11_state = l0c_cnt_carry;  // #2932
        int32_t l1k_cnt_9_state = l1k_cnt_carry;  // #2933
        int32_t l1pv_cnt_8_state = l1pv_cnt_carry;  // #2934
        int32_t ready_lengths_2_state = 0;  // #2935
        ev_mte3_mte2_valid_1.wait();  // #2718
        ev_v_mte2_valid_d0_1.set();  // #2731
        ev_v_mte2_valid_d0_1.wait();  // #2732
        for (int32_t beat = 0; beat < add_13; beat += 1) {  // #4377
            const int32_t accum_step_carry = (int32_t)(accum_step_3_state);  // #2936
            const int32_t l0ab_cnt_carry_1 = (int32_t)(l0ab_cnt_16_state);  // #2937
            const int32_t l0c_cnt_carry_1 = (int32_t)(l0c_cnt_11_state);  // #2938
            const int32_t l1k_cnt_carry_1 = (int32_t)(l1k_cnt_9_state);  // #2939
            const int32_t l1pv_cnt_carry_1 = (int32_t)(l1pv_cnt_8_state);  // #2940
            const int32_t ready_lengths_carry = (int32_t)(ready_lengths_2_state);  // #2941
            const bool cmp_9 = beat < total_beats;  // #2943
            int32_t b_len_p_5_state = 0;  // #2944
            int32_t l0ab_cnt_8_state = 0;  // #2945
            int32_t l0c_cnt_8_state = 0;  // #2946
            int32_t l1k_cnt_8_state = 0;  // #2947
            if (cmp_9) {  // #3890
                const bool approx_p = beat < approx_beats_1;  // #2948
                const int32_t approx_g_p = beat * 2;  // #2949
                const int32_t e_base_p = Max(0, beat - approx_beats_1) * 6;  // #2958 #2959 #2960
                const int32_t b_len_p_1 = Min(6, exact_total_1 - e_base_p);  // #2961 #2962
                int32_t b_len_p_4_state = 0;  // #2963
                uint64_t mask_0_2_state = 0ULL;  // #2964
                uint64_t mask_1_2_state = 0ULL;  // #2965
                if (approx_p) {  // #3020
                    const bool cmp_10 = approx_g_p < 1;  // #2968
                    uint64_t mask_0_1_state = 0ULL;  // #2969
                    if (cmp_10) {  // #2980
                        mask_0_1_state = approx_mask.load((((b_1 * (H * NT_Q_PAD8)) + (h_1 * NT_Q_PAD8)) + q_idx) + approx_g_p);  // #2971 #2972 #2973 #2974 #2975 #2976 #2977 #2978
                    } else {
                        mask_0_1_state = 0ULL;  // #2979
                    }
                    const uint64_t mask_0_1 = (uint64_t)(mask_0_1_state);  // #2981
                    const int32_t add_18 = approx_g_p + 1;  // #2982
                    const bool cmp_11 = add_18 < 1;  // #2984
                    uint64_t mask_1_1_state = 0ULL;  // #2985
                    if (cmp_11) {  // #2996
                        mask_1_1_state = approx_mask.load((((b_1 * (H * NT_Q_PAD8)) + (h_1 * NT_Q_PAD8)) + q_idx) + add_18);  // #2987 #2988 #2989 #2990 #2991 #2992 #2993 #2994
                    } else {
                        mask_1_1_state = 0ULL;  // #2995
                    }
                    const bool cmp_12 = (((((((mask_0_1 | 0ULL) | 0ULL) | 0ULL) | 0ULL) | 0ULL) | 0ULL) | 0ULL) != 0ULL;  // #2998 #2999 #3000 #3001 #3002 #3003 #3004 #3006
                    int32_t b_len_p_3_state = 0;  // #3007
                    if (cmp_12) {  // #3012
                        b_len_p_3_state = 1 - approx_g_p;  // #3008 #3010
                    } else {
                        b_len_p_3_state = 0;  // #3011
                    }
                    b_len_p_4_state = (int32_t)(b_len_p_3_state);  // #3013 #3014
                    mask_0_2_state = mask_0_1;  // #3015
                    mask_1_2_state = 0ULL;  // #3016
                } else {
                    b_len_p_4_state = b_len_p_1;  // #3017
                    mask_0_2_state = 0ULL;  // #3018
                    mask_1_2_state = 0ULL;  // #3019
                }
                const int32_t b_len_p_4 = (int32_t)(b_len_p_4_state);  // #3021
                const uint64_t mask_0_2 = (uint64_t)(mask_0_2_state);  // #3022
                const bool cmp_13 = b_len_p_4 > 0;  // #3024
                int32_t l0ab_cnt_7_state = 0;  // #3025
                int32_t l0c_cnt_7_state = 0;  // #3026
                int32_t l1k_cnt_7_state = 0;  // #3027
                const int32_t beat_slot_4_1 = beat & 3;  // #4400
                if (cmp_13) {  // #3878
                    const int32_t pair_len = Min(2, b_len_p_4);  // #3030
                    const bool cmp_14 = pair_len > 0;  // #3031
                    int32_t l0ab_cnt_2_state = 0;  // #3032
                    int32_t l0c_cnt_2_state = 0;  // #3033
                    int32_t l1k_cnt_2_state = 0;  // #3034
                    if (cmp_14) {  // #3168
                        if (approx_p) {  // #3148
                            if (pair_len < 2) {  // #3038 #3041
                                PipeBarrier<PIPE_MTE2>();  // #3040
                            }
                        } else {
                            const int32_t blk1 = e_base_p + 1;  // #3049
                            const bool not_6 = !(dense);  // #3050
                            int32_t blk0_1_state = 0;  // #3051
                            int32_t blk1_3_state = 0;  // #3052
                            if (not_6) {  // #3088
                                const int32_t blk0 = (int32_t)(exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + e_base_p));  // #3054 #3055 #3056 #3057 #3058 #3059 #3060 #3061 #3062 #3063 #3064
                                const bool cmp_16 = pair_len == 2;  // #3065
                                int32_t blk1_2_state = 0;  // #3066
                                if (cmp_16) {  // #3082
                                    const int32_t add_28 = e_base_p + 1;  // #3067
                                    blk1_2_state = (int32_t)(exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + add_28));  // #3069 #3070 #3071 #3072 #3073 #3074 #3075 #3076 #3077 #3078 #3079 #3080
                                } else {
                                    blk1_2_state = blk1;  // #3081
                                }
                                const int32_t blk1_2 = (int32_t)(blk1_2_state);  // #3083
                                blk0_1_state = blk0;  // #3084
                                blk1_3_state = blk1_2;  // #3085
                            } else {
                                blk0_1_state = e_base_p;  // #3086
                                blk1_3_state = blk1;  // #3087
                            }
                            const int32_t blk0_1 = (int32_t)(blk0_1_state);  // #3089
                            const int32_t blk1_3 = (int32_t)(blk1_3_state);  // #3090
                            const int32_t start0 = blk0_1 * 64;  // #3091
                            const int32_t rows0 = Min(64, T - start0);  // #3092 #3093
                            const bool cmp_17 = pair_len == 2;  // #3094
                            int32_t rows1_1_state = 0;  // #3095
                            if (cmp_17) {  // #3101
                                rows1_1_state = Min(64, T - (blk1_3 * 64));  // #3096 #3097 #3098 #3099
                            } else {
                                rows1_1_state = 64;  // #3100
                            }
                            const int32_t rows1_1 = (int32_t)(rows1_1_state);  // #3102
                            const bool cmp_18 = pair_len < 2;  // #3103
                            bool choice_1_state = false;  // #3104
                            if (cmp_18) {  // #3114
                                choice_1_state = true;  // #3105
                            } else {
                                const bool cmp_19 = rows0 < 64;  // #3106
                                bool choice_state = false;  // #3107
                                if (cmp_19) {  // #3111
                                    choice_state = true;  // #3108
                                } else {
                                    choice_state = rows1_1 < 64;  // #3109 #3110
                                }
                                choice_1_state = (bool)(choice_state);  // #3112 #3113
                            }
                            const bool choice_1 = (bool)(choice_1_state);  // #3115
                            if (choice_1) {  // #3117
                                PipeBarrier<PIPE_MTE2>();  // #3116
                            }
                        }
                        const int32_t l0ab_cnt_1 = (int32_t)(l0ab_cnt_carry_1 + 1);  // #3154 #3155
                        const int32_t l1k_cnt_1 = (int32_t)(l1k_cnt_carry_1 + 1);  // #3158 #3159
                        const int32_t l0c_cnt_1 = (int32_t)(l0c_cnt_carry_1 + 1);  // #3160 #3161
                        l0ab_cnt_2_state = l0ab_cnt_1;  // #3162
                        l0c_cnt_2_state = l0c_cnt_1;  // #3163
                        l1k_cnt_2_state = l1k_cnt_1;  // #3164
                    } else {
                        l0ab_cnt_2_state = l0ab_cnt_carry_1;  // #3165
                        l0c_cnt_2_state = l0c_cnt_carry_1;  // #3166
                        l1k_cnt_2_state = l1k_cnt_carry_1;  // #3167
                    }
                    const int32_t l0ab_cnt_2 = (int32_t)(l0ab_cnt_2_state);  // #3169
                    const int32_t l0c_cnt_2 = (int32_t)(l0c_cnt_2_state);  // #3170
                    const int32_t l1k_cnt_2 = (int32_t)(l1k_cnt_2_state);  // #3171
                    const int32_t pair_len_1 = Min(2, b_len_p_4 - 2);  // #3172 #3173
                    const bool cmp_24 = pair_len_1 > 0;  // #3174
                    int32_t l0ab_cnt_4_state = 0;  // #3175
                    int32_t l0c_cnt_4_state = 0;  // #3176
                    int32_t l1k_cnt_4_state = 0;  // #3177
                    if (cmp_24) {  // #3311
                        if (approx_p) {  // #3291
                            if (pair_len_1 < 2) {  // #3181 #3184
                                PipeBarrier<PIPE_MTE2>();  // #3183
                            }
                        } else {
                            const int32_t add_42 = e_base_p + 2;  // #3191
                            const int32_t blk1_4 = add_42 + 1;  // #3192
                            const bool not_7 = !(dense);  // #3193
                            int32_t blk0_3_state = 0;  // #3194
                            int32_t blk1_7_state = 0;  // #3195
                            if (not_7) {  // #3231
                                const int32_t blk0_2 = (int32_t)(exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + add_42));  // #3197 #3198 #3199 #3200 #3201 #3202 #3203 #3204 #3205 #3206 #3207
                                const bool cmp_26 = pair_len_1 == 2;  // #3208
                                int32_t blk1_6_state = 0;  // #3209
                                if (cmp_26) {  // #3225
                                    const int32_t add_46 = add_42 + 1;  // #3210
                                    blk1_6_state = (int32_t)(exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + add_46));  // #3212 #3213 #3214 #3215 #3216 #3217 #3218 #3219 #3220 #3221 #3222 #3223
                                } else {
                                    blk1_6_state = blk1_4;  // #3224
                                }
                                const int32_t blk1_6 = (int32_t)(blk1_6_state);  // #3226
                                blk0_3_state = blk0_2;  // #3227
                                blk1_7_state = blk1_6;  // #3228
                            } else {
                                blk0_3_state = add_42;  // #3229
                                blk1_7_state = blk1_4;  // #3230
                            }
                            const int32_t blk0_3 = (int32_t)(blk0_3_state);  // #3232
                            const int32_t blk1_7 = (int32_t)(blk1_7_state);  // #3233
                            const int32_t start0_1 = blk0_3 * 64;  // #3234
                            const int32_t rows0_1 = Min(64, T - start0_1);  // #3235 #3236
                            const bool cmp_27 = pair_len_1 == 2;  // #3237
                            int32_t rows1_3_state = 0;  // #3238
                            if (cmp_27) {  // #3244
                                rows1_3_state = Min(64, T - (blk1_7 * 64));  // #3239 #3240 #3241 #3242
                            } else {
                                rows1_3_state = 64;  // #3243
                            }
                            const int32_t rows1_3 = (int32_t)(rows1_3_state);  // #3245
                            const bool cmp_28 = pair_len_1 < 2;  // #3246
                            bool choice_4_state = false;  // #3247
                            if (cmp_28) {  // #3257
                                choice_4_state = true;  // #3248
                            } else {
                                const bool cmp_29 = rows0_1 < 64;  // #3249
                                bool choice_3_state = false;  // #3250
                                if (cmp_29) {  // #3254
                                    choice_3_state = true;  // #3251
                                } else {
                                    choice_3_state = rows1_3 < 64;  // #3252 #3253
                                }
                                choice_4_state = (bool)(choice_3_state);  // #3255 #3256
                            }
                            const bool choice_4 = (bool)(choice_4_state);  // #3258
                            if (choice_4) {  // #3260
                                PipeBarrier<PIPE_MTE2>();  // #3259
                            }
                        }
                        const int32_t l0ab_cnt_3 = (int32_t)(l0ab_cnt_2 + 1);  // #3297 #3298
                        const int32_t l1k_cnt_3 = (int32_t)(l1k_cnt_2 + 1);  // #3301 #3302
                        const int32_t l0c_cnt_3 = (int32_t)(l0c_cnt_2 + 1);  // #3303 #3304
                        l0ab_cnt_4_state = l0ab_cnt_3;  // #3305
                        l0c_cnt_4_state = l0c_cnt_3;  // #3306
                        l1k_cnt_4_state = l1k_cnt_3;  // #3307
                    } else {
                        l0ab_cnt_4_state = l0ab_cnt_2;  // #3308
                        l0c_cnt_4_state = l0c_cnt_2;  // #3309
                        l1k_cnt_4_state = l1k_cnt_2;  // #3310
                    }
                    const int32_t l0ab_cnt_4 = (int32_t)(l0ab_cnt_4_state);  // #3312
                    const int32_t l0c_cnt_4 = (int32_t)(l0c_cnt_4_state);  // #3313
                    const int32_t l1k_cnt_4 = (int32_t)(l1k_cnt_4_state);  // #3314
                    const int32_t pair_len_2 = Min(2, b_len_p_4 - 4);  // #3315 #3316
                    const bool cmp_34 = pair_len_2 > 0;  // #3317
                    int32_t l0ab_cnt_6_state = 0;  // #3318
                    int32_t l0c_cnt_6_state = 0;  // #3319
                    int32_t l1k_cnt_6_state = 0;  // #3320
                    if (cmp_34) {  // #3454
                        if (approx_p) {  // #3434
                            if (pair_len_2 < 2) {  // #3324 #3327
                                PipeBarrier<PIPE_MTE2>();  // #3326
                            }
                        } else {
                            const int32_t add_60 = e_base_p + 4;  // #3334
                            const int32_t blk1_8 = add_60 + 1;  // #3335
                            const bool not_8 = !(dense);  // #3336
                            int32_t blk0_5_state = 0;  // #3337
                            int32_t blk1_11_state = 0;  // #3338
                            if (not_8) {  // #3374
                                const int32_t blk0_4 = (int32_t)(exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + add_60));  // #3340 #3341 #3342 #3343 #3344 #3345 #3346 #3347 #3348 #3349 #3350
                                const bool cmp_36 = pair_len_2 == 2;  // #3351
                                int32_t blk1_10_state = 0;  // #3352
                                if (cmp_36) {  // #3368
                                    const int32_t add_64 = add_60 + 1;  // #3353
                                    blk1_10_state = (int32_t)(exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + add_64));  // #3355 #3356 #3357 #3358 #3359 #3360 #3361 #3362 #3363 #3364 #3365 #3366
                                } else {
                                    blk1_10_state = blk1_8;  // #3367
                                }
                                const int32_t blk1_10 = (int32_t)(blk1_10_state);  // #3369
                                blk0_5_state = blk0_4;  // #3370
                                blk1_11_state = blk1_10;  // #3371
                            } else {
                                blk0_5_state = add_60;  // #3372
                                blk1_11_state = blk1_8;  // #3373
                            }
                            const int32_t blk0_5 = (int32_t)(blk0_5_state);  // #3375
                            const int32_t blk1_11 = (int32_t)(blk1_11_state);  // #3376
                            const int32_t start0_2 = blk0_5 * 64;  // #3377
                            const int32_t rows0_2 = Min(64, T - start0_2);  // #3378 #3379
                            const bool cmp_37 = pair_len_2 == 2;  // #3380
                            int32_t rows1_5_state = 0;  // #3381
                            if (cmp_37) {  // #3387
                                rows1_5_state = Min(64, T - (blk1_11 * 64));  // #3382 #3383 #3384 #3385
                            } else {
                                rows1_5_state = 64;  // #3386
                            }
                            const int32_t rows1_5 = (int32_t)(rows1_5_state);  // #3388
                            const bool cmp_38 = pair_len_2 < 2;  // #3389
                            bool choice_7_state = false;  // #3390
                            if (cmp_38) {  // #3400
                                choice_7_state = true;  // #3391
                            } else {
                                const bool cmp_39 = rows0_2 < 64;  // #3392
                                bool choice_6_state = false;  // #3393
                                if (cmp_39) {  // #3397
                                    choice_6_state = true;  // #3394
                                } else {
                                    choice_6_state = rows1_5 < 64;  // #3395 #3396
                                }
                                choice_7_state = (bool)(choice_6_state);  // #3398 #3399
                            }
                            const bool choice_7 = (bool)(choice_7_state);  // #3401
                            if (choice_7) {  // #3403
                                PipeBarrier<PIPE_MTE2>();  // #3402
                            }
                        }
                        const int32_t l0ab_cnt_5 = (int32_t)(l0ab_cnt_4 + 1);  // #3440 #3441
                        const int32_t l1k_cnt_5 = (int32_t)(l1k_cnt_4 + 1);  // #3444 #3445
                        const int32_t l0c_cnt_5 = (int32_t)(l0c_cnt_4 + 1);  // #3446 #3447
                        l0ab_cnt_6_state = l0ab_cnt_5;  // #3448
                        l0c_cnt_6_state = l0c_cnt_5;  // #3449
                        l1k_cnt_6_state = l1k_cnt_5;  // #3450
                    } else {
                        l0ab_cnt_6_state = l0ab_cnt_4;  // #3451
                        l0c_cnt_6_state = l0c_cnt_4;  // #3452
                        l1k_cnt_6_state = l1k_cnt_4;  // #3453
                    }
                    const int32_t l0ab_cnt_6 = (int32_t)(l0ab_cnt_6_state);  // #3455
                    const int32_t l0c_cnt_6 = (int32_t)(l0c_cnt_6_state);  // #3456
                    const int32_t l1k_cnt_6 = (int32_t)(l1k_cnt_6_state);  // #3457
                    WAIT_CUBE<PIPE_S>(0);  // #1118
                    const int32_t pair_len_3 = Min(2, b_len_p_4);  // #3459
                    if (pair_len_3 > 0) {  // #3460 #3462
                        const Tensor<float, Position::UB> ub_sc = arena.as<float>();  // #3461
                        WaitFlag<PIPE_V, PIPE_MTE2>(4);  // #1124
                        const GMTensor<float> view_213 = score_ws[(((cube_idx * 196608) + (beat_slot_4_1 * 49152)) + (sb_row * 128))];  // #1127
                        gm_to_ub_pad(ub_sc, view_213, 64, 512, 0, 0);  // #1128
                        SetFlag<PIPE_MTE2, PIPE_V>(4);  // #1129
                    }
                    const int32_t pair_len_4 = Min(2, b_len_p_4 - 2);  // #3463 #3464
                    if (pair_len_4 > 0) {  // #3465 #3467
                        const int32_t pair_row_v_1 = 128 + sb_row;  // #1134
                        const Tensor<float, Position::UB> ub_sc_1 = arena.as<float>()[8192];  // #3466
                        WaitFlag<PIPE_V, PIPE_MTE2>(5);  // #1136
                        const GMTensor<float> view_214 = score_ws[(((cube_idx * 196608) + (beat_slot_4_1 * 49152)) + (pair_row_v_1 * 128))];  // #1139
                        gm_to_ub_pad(ub_sc_1, view_214, 64, 512, 0, 0);  // #1140
                        SetFlag<PIPE_MTE2, PIPE_V>(5);  // #1141
                    }
                    const int32_t pair_len_5 = Min(2, b_len_p_4 - 4);  // #3468 #3469
                    if (pair_len_5 > 0) {  // #3470 #3472
                        const int32_t pair_row_v_2 = 256 + sb_row;  // #1146
                        const Tensor<float, Position::UB> ub_sc_2 = arena.as<float>()[16384];  // #3471
                        WaitFlag<PIPE_V, PIPE_MTE2>(6);  // #1148
                        const GMTensor<float> view_215 = score_ws[(((cube_idx * 196608) + (beat_slot_4_1 * 49152)) + (pair_row_v_2 * 128))];  // #1151
                        gm_to_ub_pad(ub_sc_2, view_215, 64, 512, 0, 0);  // #1152
                        SetFlag<PIPE_MTE2, PIPE_V>(6);  // #1153
                    }
                    PipeBarrier<PIPE_V>();  // #1155
                    SetMaskCount();  // #1156
                    SetVectorMask(64);  // #1156
                    vector_dup(ub_bmax_s.ptr(), (float)(-1e+30f), (uint8_t)(1), (uint16_t)1, 1, (uint8_t)8, 0);  // #1156
                    SetMaskNorm();  // #1156
                    ResetMask();  // #1156
                    const int32_t pair_len_6 = Min(2, b_len_p_4);  // #3474
                    if (pair_len_6 > 0) {  // #3475 #3552
                        const Tensor<float, Position::UB> ub_sc_3 = arena.as<float>();  // #3476
                        WaitFlag<PIPE_MTE2, PIPE_V>(4);  // #1161
                        PipeBarrier<PIPE_V>();  // c220: V-V hazard #1162
                        vmuls(ub_sc_3.ptr(), ub_sc_3.ptr(), (float)(scale_natural), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #1162
                        if (approx_p) {  // #3549
                            SetVectorMask((uint64_t)(0), (uint64_t)(~(mask_0_2)));  // #3477 #1164
                            PipeBarrier<PIPE_V>();  // #1165
                            const Tensor<float, Position::UB> view_216 = arena.as<float>();  // #3478
                            vector_dup(view_216.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #1167
                            ResetMask();  // #1168
                            SetVectorMask((uint64_t)(0), (uint64_t)(~(0)));  // #3479 #1170
                            PipeBarrier<PIPE_V>();  // #1171
                            const Tensor<float, Position::UB> view_217 = arena.as<float>()[64];  // #3480
                            vector_dup(view_217.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #1173
                            ResetMask();  // #1174
                        } else {
                            const Tensor<float, Position::UB> ub_cols = arena.as<float>();  // #3481
                            if (0 < pair_len_6) {  // #3482 #3514
                                const bool not_11 = !(dense);  // #3485
                                int32_t blk_p_2_state = 0;  // #3486
                                if (not_11) {  // #3501
                                    blk_p_2_state = (int32_t)(exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + e_base_p));  // #3488 #3489 #3490 #3491 #3492 #3493 #3494 #3495 #3496 #3497 #3498 #3499
                                } else {
                                    blk_p_2_state = e_base_p;  // #3500
                                }
                                const int32_t blk_p_2 = (int32_t)(blk_p_2_state);  // #3502
                                const int32_t valid_n = Min(64, T - (blk_p_2 * 64));  // #3503 #3504 #3505
                                if (valid_n < 64) {  // #3506 #3513
                                    const uint64_t low = 1ULL << valid_n;  // #3509
                                    SetVectorMask((uint64_t)(0), (uint64_t)(18446744073709551615ULL ^ (low - 1ULL)));  // #3511 #3512 #1205
                                    PipeBarrier<PIPE_V>();  // #1206
                                    vector_dup(ub_cols.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #1207
                                    ResetMask();  // #1208
                                }
                            } else {
                                PipeBarrier<PIPE_V>();  // #1210
                                vector_dup(ub_cols.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #1211
                            }
                            const Tensor<float, Position::UB> ub_cols_1 = arena.as<float>()[64];  // #3515
                            if (1 < pair_len_6) {  // #3516 #3548
                                const int32_t blk_p_3 = e_base_p + 1;  // #3518
                                const bool not_12 = !(dense);  // #3519
                                int32_t blk_p_5_state = 0;  // #3520
                                if (not_12) {  // #3535
                                    blk_p_5_state = (int32_t)(exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + blk_p_3));  // #3522 #3523 #3524 #3525 #3526 #3527 #3528 #3529 #3530 #3531 #3532 #3533
                                } else {
                                    blk_p_5_state = blk_p_3;  // #3534
                                }
                                const int32_t blk_p_5 = (int32_t)(blk_p_5_state);  // #3536
                                const int32_t valid_n_1 = Min(64, T - (blk_p_5 * 64));  // #3537 #3538 #3539
                                if (valid_n_1 < 64) {  // #3540 #3547
                                    const uint64_t low_1 = 1ULL << valid_n_1;  // #3543
                                    SetVectorMask((uint64_t)(0), (uint64_t)(18446744073709551615ULL ^ (low_1 - 1ULL)));  // #3545 #3546 #1243
                                    PipeBarrier<PIPE_V>();  // #1244
                                    vector_dup(ub_cols_1.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #1245
                                    ResetMask();  // #1246
                                }
                            } else {
                                PipeBarrier<PIPE_V>();  // #1248
                                vector_dup(ub_cols_1.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #1249
                            }
                        }
                        PipeBarrier<PIPE_V>();  // #1252
                        vcgmax(ub_reduce_pair.ptr(), ub_sc_3.ptr(), (uint8_t)(128), 1, 1, 8);  // #1253
                        PipeBarrier<PIPE_V>();  // #1254
                        const Tensor<float, Position::UB> view_218 = arena.as<float>()[46400];  // #3550
                        const Tensor<float, Position::UB> view_219 = arena.as<float>()[46408];  // #3551
                        vmax(ub_max.ptr(), view_218.ptr(), view_219.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)2, (uint8_t)2, (uint8_t)8, (uint8_t)16, (uint8_t)16);  // #1257
                        PipeBarrier<PIPE_V>();  // #1258
                        vcgmax(ub_max_s.ptr(), ub_max.ptr(), (uint8_t)(8), 1, 1, 8);  // #1259
                        PipeBarrier<PIPE_V>();  // #1260
                        SetMaskCount();  // #1261
                        SetVectorMask(64);  // #1261
                        vmax(ub_bmax_s.ptr(), ub_bmax_s.ptr(), ub_max_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1261
                        SetMaskNorm();  // #1261
                        ResetMask();  // #1261
                    }
                    const int32_t pair_len_7 = Min(2, b_len_p_4 - 2);  // #3553 #3554
                    if (pair_len_7 > 0) {  // #3555 #3632
                        const Tensor<float, Position::UB> ub_sc_4 = arena.as<float>()[8192];  // #3556
                        WaitFlag<PIPE_MTE2, PIPE_V>(5);  // #1267
                        PipeBarrier<PIPE_V>();  // c220: V-V hazard #1268
                        vmuls(ub_sc_4.ptr(), ub_sc_4.ptr(), (float)(scale_natural), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #1268
                        if (approx_p) {  // #3629
                            SetVectorMask((uint64_t)(0), (uint64_t)(~(0)));  // #3557 #1270
                            PipeBarrier<PIPE_V>();  // #1271
                            const Tensor<float, Position::UB> view_220 = arena.as<float>()[8192];  // #3558
                            vector_dup(view_220.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #1273
                            ResetMask();  // #1274
                            SetVectorMask((uint64_t)(0), (uint64_t)(~(0)));  // #3559 #1276
                            PipeBarrier<PIPE_V>();  // #1277
                            const Tensor<float, Position::UB> view_221 = arena.as<float>()[8256];  // #3560
                            vector_dup(view_221.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #1279
                            ResetMask();  // #1280
                        } else {
                            const Tensor<float, Position::UB> ub_cols_2 = arena.as<float>()[8192];  // #3561
                            if (0 < pair_len_7) {  // #3562 #3594
                                const int32_t add_87 = e_base_p + 2;  // #3563
                                const bool not_15 = !(dense);  // #3565
                                int32_t blk_p_8_state = 0;  // #3566
                                if (not_15) {  // #3581
                                    blk_p_8_state = (int32_t)(exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + add_87));  // #3568 #3569 #3570 #3571 #3572 #3573 #3574 #3575 #3576 #3577 #3578 #3579
                                } else {
                                    blk_p_8_state = add_87;  // #3580
                                }
                                const int32_t blk_p_8 = (int32_t)(blk_p_8_state);  // #3582
                                const int32_t valid_n_2 = Min(64, T - (blk_p_8 * 64));  // #3583 #3584 #3585
                                if (valid_n_2 < 64) {  // #3586 #3593
                                    const uint64_t low_2 = 1ULL << valid_n_2;  // #3589
                                    SetVectorMask((uint64_t)(0), (uint64_t)(18446744073709551615ULL ^ (low_2 - 1ULL)));  // #3591 #3592 #1311
                                    PipeBarrier<PIPE_V>();  // #1312
                                    vector_dup(ub_cols_2.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #1313
                                    ResetMask();  // #1314
                                }
                            } else {
                                PipeBarrier<PIPE_V>();  // #1316
                                vector_dup(ub_cols_2.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #1317
                            }
                            const Tensor<float, Position::UB> ub_cols_3 = arena.as<float>()[8256];  // #3595
                            if (1 < pair_len_7) {  // #3596 #3628
                                const int32_t blk_p_9 = (e_base_p + 2) + 1;  // #3597 #3598
                                const bool not_16 = !(dense);  // #3599
                                int32_t blk_p_11_state = 0;  // #3600
                                if (not_16) {  // #3615
                                    blk_p_11_state = (int32_t)(exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + blk_p_9));  // #3602 #3603 #3604 #3605 #3606 #3607 #3608 #3609 #3610 #3611 #3612 #3613
                                } else {
                                    blk_p_11_state = blk_p_9;  // #3614
                                }
                                const int32_t blk_p_11 = (int32_t)(blk_p_11_state);  // #3616
                                const int32_t valid_n_3 = Min(64, T - (blk_p_11 * 64));  // #3617 #3618 #3619
                                if (valid_n_3 < 64) {  // #3620 #3627
                                    const uint64_t low_3 = 1ULL << valid_n_3;  // #3623
                                    SetVectorMask((uint64_t)(0), (uint64_t)(18446744073709551615ULL ^ (low_3 - 1ULL)));  // #3625 #3626 #1349
                                    PipeBarrier<PIPE_V>();  // #1350
                                    vector_dup(ub_cols_3.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #1351
                                    ResetMask();  // #1352
                                }
                            } else {
                                PipeBarrier<PIPE_V>();  // #1354
                                vector_dup(ub_cols_3.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #1355
                            }
                        }
                        PipeBarrier<PIPE_V>();  // #1358
                        vcgmax(ub_reduce_pair.ptr(), ub_sc_4.ptr(), (uint8_t)(128), 1, 1, 8);  // #1359
                        PipeBarrier<PIPE_V>();  // #1360
                        const Tensor<float, Position::UB> view_222 = arena.as<float>()[46400];  // #3630
                        const Tensor<float, Position::UB> view_223 = arena.as<float>()[46408];  // #3631
                        vmax(ub_max.ptr(), view_222.ptr(), view_223.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)2, (uint8_t)2, (uint8_t)8, (uint8_t)16, (uint8_t)16);  // #1363
                        PipeBarrier<PIPE_V>();  // #1364
                        vcgmax(ub_max_s.ptr(), ub_max.ptr(), (uint8_t)(8), 1, 1, 8);  // #1365
                        PipeBarrier<PIPE_V>();  // #1366
                        SetMaskCount();  // #1367
                        SetVectorMask(64);  // #1367
                        vmax(ub_bmax_s.ptr(), ub_bmax_s.ptr(), ub_max_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1367
                        SetMaskNorm();  // #1367
                        ResetMask();  // #1367
                    }
                    const int32_t pair_len_8 = Min(2, b_len_p_4 - 4);  // #3633 #3634
                    if (pair_len_8 > 0) {  // #3635 #3712
                        const Tensor<float, Position::UB> ub_sc_5 = arena.as<float>()[16384];  // #3636
                        WaitFlag<PIPE_MTE2, PIPE_V>(6);  // #1373
                        PipeBarrier<PIPE_V>();  // c220: V-V hazard #1374
                        vmuls(ub_sc_5.ptr(), ub_sc_5.ptr(), (float)(scale_natural), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #1374
                        if (approx_p) {  // #3709
                            SetVectorMask((uint64_t)(0), (uint64_t)(~(0)));  // #3637 #1376
                            PipeBarrier<PIPE_V>();  // #1377
                            const Tensor<float, Position::UB> view_224 = arena.as<float>()[16384];  // #3638
                            vector_dup(view_224.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #1379
                            ResetMask();  // #1380
                            SetVectorMask((uint64_t)(0), (uint64_t)(~(0)));  // #3639 #1382
                            PipeBarrier<PIPE_V>();  // #1383
                            const Tensor<float, Position::UB> view_225 = arena.as<float>()[16448];  // #3640
                            vector_dup(view_225.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #1385
                            ResetMask();  // #1386
                        } else {
                            const Tensor<float, Position::UB> ub_cols_4 = arena.as<float>()[16384];  // #3641
                            if (0 < pair_len_8) {  // #3642 #3674
                                const int32_t add_95 = e_base_p + 4;  // #3643
                                const bool not_19 = !(dense);  // #3645
                                int32_t blk_p_14_state = 0;  // #3646
                                if (not_19) {  // #3661
                                    blk_p_14_state = (int32_t)(exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + add_95));  // #3648 #3649 #3650 #3651 #3652 #3653 #3654 #3655 #3656 #3657 #3658 #3659
                                } else {
                                    blk_p_14_state = add_95;  // #3660
                                }
                                const int32_t blk_p_14 = (int32_t)(blk_p_14_state);  // #3662
                                const int32_t valid_n_4 = Min(64, T - (blk_p_14 * 64));  // #3663 #3664 #3665
                                if (valid_n_4 < 64) {  // #3666 #3673
                                    const uint64_t low_4 = 1ULL << valid_n_4;  // #3669
                                    SetVectorMask((uint64_t)(0), (uint64_t)(18446744073709551615ULL ^ (low_4 - 1ULL)));  // #3671 #3672 #1417
                                    PipeBarrier<PIPE_V>();  // #1418
                                    vector_dup(ub_cols_4.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #1419
                                    ResetMask();  // #1420
                                }
                            } else {
                                PipeBarrier<PIPE_V>();  // #1422
                                vector_dup(ub_cols_4.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #1423
                            }
                            const Tensor<float, Position::UB> ub_cols_5 = arena.as<float>()[16448];  // #3675
                            if (1 < pair_len_8) {  // #3676 #3708
                                const int32_t blk_p_15 = (e_base_p + 4) + 1;  // #3677 #3678
                                const bool not_20 = !(dense);  // #3679
                                int32_t blk_p_17_state = 0;  // #3680
                                if (not_20) {  // #3695
                                    blk_p_17_state = (int32_t)(exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + blk_p_15));  // #3682 #3683 #3684 #3685 #3686 #3687 #3688 #3689 #3690 #3691 #3692 #3693
                                } else {
                                    blk_p_17_state = blk_p_15;  // #3694
                                }
                                const int32_t blk_p_17 = (int32_t)(blk_p_17_state);  // #3696
                                const int32_t valid_n_5 = Min(64, T - (blk_p_17 * 64));  // #3697 #3698 #3699
                                if (valid_n_5 < 64) {  // #3700 #3707
                                    const uint64_t low_5 = 1ULL << valid_n_5;  // #3703
                                    SetVectorMask((uint64_t)(0), (uint64_t)(18446744073709551615ULL ^ (low_5 - 1ULL)));  // #3705 #3706 #1455
                                    PipeBarrier<PIPE_V>();  // #1456
                                    vector_dup(ub_cols_5.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #1457
                                    ResetMask();  // #1458
                                }
                            } else {
                                PipeBarrier<PIPE_V>();  // #1460
                                vector_dup(ub_cols_5.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #1461
                            }
                        }
                        PipeBarrier<PIPE_V>();  // #1464
                        vcgmax(ub_reduce_pair.ptr(), ub_sc_5.ptr(), (uint8_t)(128), 1, 1, 8);  // #1465
                        PipeBarrier<PIPE_V>();  // #1466
                        const Tensor<float, Position::UB> view_226 = arena.as<float>()[46400];  // #3710
                        const Tensor<float, Position::UB> view_227 = arena.as<float>()[46408];  // #3711
                        vmax(ub_max.ptr(), view_226.ptr(), view_227.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)2, (uint8_t)2, (uint8_t)8, (uint8_t)16, (uint8_t)16);  // #1469
                        PipeBarrier<PIPE_V>();  // #1470
                        vcgmax(ub_max_s.ptr(), ub_max.ptr(), (uint8_t)(8), 1, 1, 8);  // #1471
                        PipeBarrier<PIPE_V>();  // #1472
                        SetMaskCount();  // #1473
                        SetVectorMask(64);  // #1473
                        vmax(ub_bmax_s.ptr(), ub_bmax_s.ptr(), ub_max_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1473
                        SetMaskNorm();  // #1473
                        ResetMask();  // #1473
                    }
                    const int32_t mod = beat & 3;  // #3713
                    const Tensor<float, Position::UB> tile_1 = arena.as<float>()[(45632 + (mod * 64))];  // #3717
                    const Tensor<float, Position::UB> row_diff = arena.as<float>()[(45632 + (mod * 64))];  // #3718
                    PipeBarrier<PIPE_V>();  // c220: V-V hazard #1481
                    SetMaskCount();  // #1481
                    SetVectorMask(64);  // #1481
                    vadd(ub_old_max_s.ptr(), ub_rmax_s.ptr(), ub_zero_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1481
                    SetMaskNorm();  // #1481
                    ResetMask();  // #1481
                    PipeBarrier<PIPE_V>();  // #1482
                    SetMaskCount();  // #1483
                    SetVectorMask(64);  // #1483
                    vmax(ub_rmax_s.ptr(), ub_rmax_s.ptr(), ub_bmax_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1483
                    SetMaskNorm();  // #1483
                    ResetMask();  // #1483
                    PipeBarrier<PIPE_V>();  // #1484
                    SetMaskCount();  // #1485
                    SetVectorMask(64);  // #1485
                    vsub(row_diff.ptr(), ub_old_max_s.ptr(), ub_rmax_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1485
                    SetMaskNorm();  // #1485
                    ResetMask();  // #1485
                    PipeBarrier<PIPE_V>();  // #1486
                    SetMaskCount();  // #1487
                    SetVectorMask(64);  // #1487
                    vexp(row_diff.ptr(), row_diff.ptr(), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint8_t)0, (uint8_t)0);  // #1487
                    SetMaskNorm();  // #1487
                    ResetMask();  // #1487
                    PipeBarrier<PIPE_V>();  // c220: V-V hazard #1488
                    SetMaskCount();  // #1488
                    SetVectorMask(64);  // #1488
                    vector_dup(ub_bsum_s.ptr(), (float)(0.0f), (uint8_t)(1), (uint16_t)1, 1, (uint8_t)8, 0);  // #1488
                    SetMaskNorm();  // #1488
                    ResetMask();  // #1488
                    PipeBarrier<PIPE_V>();  // c220: V-V hazard #1489
                    ResetMask();  // #1489
                    vbrcb((__ubuf__ uint32_t*)(ub_max.ptr()), (__ubuf__ uint32_t*)(ub_rmax_s.ptr()), (uint16_t)1, (uint16_t)8, (uint8_t)(8));  // #1489
                    PipeBarrier<PIPE_V>();  // #1490
                    const int32_t pair_len_9 = Min(2, b_len_p_4);  // #3720
                    if (pair_len_9 > 0) {  // #3721 #3767
                        const Tensor<float, Position::UB> ub_sc_6 = arena.as<float>();  // #3722
                        const Tensor<float, Position::UB> ub_cols_6 = arena.as<float>();  // #3723
                        vsub(ub_cols_6.ptr(), ub_cols_6.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #1497
                        const Tensor<float, Position::UB> ub_cols_7 = arena.as<float>()[64];  // #3724
                        PipeBarrier<PIPE_V>();  // c220: V-V hazard #1499
                        vsub(ub_cols_7.ptr(), ub_cols_7.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #1499
                        PipeBarrier<PIPE_V>();  // #1500
                        vexp(ub_sc_6.ptr(), ub_sc_6.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint8_t)8, (uint8_t)8);  // #1501
                        PipeBarrier<PIPE_V>();  // #1502
                        WaitFlag<PIPE_MTE3, PIPE_V>(4);  // #1503
                        vconv_f322bf16a(probability_tiles.ptr(), ub_sc_6.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)4, (uint16_t)8);  // #1504
                        SetFlag<PIPE_V, PIPE_MTE3>(4);  // #1505
                        WaitFlag<PIPE_V, PIPE_MTE3>(4);  // #1506
                        const GMTensor<bfloat16_t> view_228 = p_ws[(((cube_idx * 196608) + (beat_slot_4_1 * 49152)) + (sb_row * 128))];  // #1509
                        ub_to_gm_pad(view_228, probability_tiles, 64, 256, 0, 0);  // #1510
                        SetFlag<PIPE_MTE3, PIPE_V>(4);  // #1511
                        PipeBarrier<PIPE_V>();  // c220: V-V hazard #1512
                        vcgadd(ub_reduce_pair.ptr(), ub_sc_6.ptr(), (uint8_t)(128), 1, 1, 8);  // #1512
                        if (0 < pair_len_9) {  // #3739 #3752
                            PipeBarrier<PIPE_V>();  // #1525
                            bool choice_10_state = false;  // #3740
                            if (approx_p) {  // #3746
                                choice_10_state = approx_g_p == 0;  // #3743 #3744
                            } else {
                                choice_10_state = false;  // #3745
                            }
                            const bool choice_10 = (bool)(choice_10_state);  // #3747
                            if (choice_10) {  // #3751
                                const Tensor<float, Position::UB> ub_cols_8 = arena.as<float>();  // #3748
                                vmul(ub_cols_8.ptr(), ub_cols_8.ptr(), ub_len_last.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)16, (uint8_t)16, (uint8_t)0);  // #1533
                                PipeBarrier<PIPE_V>();  // #1534
                                vcgadd(ub_reduce.ptr(), ub_cols_8.ptr(), (uint8_t)(64), 1, 1, 16);  // #1535
                                PipeBarrier<PIPE_V>();  // #1536
                                vcgadd(ub_sum_s.ptr(), ub_reduce.ptr(), (uint8_t)(8), 1, 1, 8);  // #1537
                            } else {
                                const Tensor<float, Position::UB> view_229 = arena.as<float>()[46400];  // #3749
                                vcgadd(ub_sum_s.ptr(), view_229.ptr(), (uint8_t)(8), 1, 2, 16);  // #1539
                                if (approx_p) {  // #3750
                                    PipeBarrier<PIPE_V>();  // #1540
                                    SetMaskCount();  // #1541
                                    SetVectorMask(64);  // #1541
                                    vmuls(ub_sum_s.ptr(), ub_sum_s.ptr(), (float)(64.0f), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #1541
                                    SetMaskNorm();  // #1541
                                    ResetMask();  // #1541
                                }
                            }
                            PipeBarrier<PIPE_V>();  // #1544
                            SetMaskCount();  // #1545
                            SetVectorMask(64);  // #1545
                            vadd(ub_bsum_s.ptr(), ub_bsum_s.ptr(), ub_sum_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1545
                            SetMaskNorm();  // #1545
                            ResetMask();  // #1545
                        }
                        if (1 < pair_len_9) {  // #3753 #3766
                            PipeBarrier<PIPE_V>();  // #1548
                            bool choice_11_state = false;  // #3754
                            if (approx_p) {  // #3760
                                choice_11_state = (approx_g_p + 1) == 0;  // #3756 #3757 #3758
                            } else {
                                choice_11_state = false;  // #3759
                            }
                            const Tensor<float, Position::UB> view_230 = arena.as<float>()[46408];  // #3763
                            vcgadd(ub_sum_s.ptr(), view_230.ptr(), (uint8_t)(8), 1, 2, 16);  // #1562
                            if (approx_p) {  // #3764
                                PipeBarrier<PIPE_V>();  // #1563
                                SetMaskCount();  // #1564
                                SetVectorMask(64);  // #1564
                                vmuls(ub_sum_s.ptr(), ub_sum_s.ptr(), (float)(64.0f), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #1564
                                SetMaskNorm();  // #1564
                                ResetMask();  // #1564
                            }
                            PipeBarrier<PIPE_V>();  // #1567
                            SetMaskCount();  // #1568
                            SetVectorMask(64);  // #1568
                            vadd(ub_bsum_s.ptr(), ub_bsum_s.ptr(), ub_sum_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1568
                            SetMaskNorm();  // #1568
                            ResetMask();  // #1568
                        }
                        SetFlag<PIPE_V, PIPE_MTE2>(4);  // #1570
                    }
                    const int32_t pair_len_10 = Min(2, b_len_p_4 - 2);  // #3768 #3769
                    if (pair_len_10 > 0) {  // #3770 #3816
                        const int32_t pair_row_v_4 = 128 + sb_row;  // #1575
                        const Tensor<float, Position::UB> ub_sc_7 = arena.as<float>()[8192];  // #3771
                        const Tensor<float, Position::UB> ub_cols_10 = arena.as<float>()[8192];  // #3772
                        PipeBarrier<PIPE_V>();  // c220: V-V hazard #1578
                        vsub(ub_cols_10.ptr(), ub_cols_10.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #1578
                        const Tensor<float, Position::UB> ub_cols_11 = arena.as<float>()[8256];  // #3773
                        PipeBarrier<PIPE_V>();  // c220: V-V hazard #1580
                        vsub(ub_cols_11.ptr(), ub_cols_11.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #1580
                        PipeBarrier<PIPE_V>();  // #1581
                        vexp(ub_sc_7.ptr(), ub_sc_7.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint8_t)8, (uint8_t)8);  // #1582
                        PipeBarrier<PIPE_V>();  // #1583
                        WaitFlag<PIPE_MTE3, PIPE_V>(4);  // #1584
                        vconv_f322bf16a(probability_tiles.ptr(), ub_sc_7.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)4, (uint16_t)8);  // #1585
                        SetFlag<PIPE_V, PIPE_MTE3>(4);  // #1586
                        WaitFlag<PIPE_V, PIPE_MTE3>(4);  // #1587
                        const GMTensor<bfloat16_t> view_231 = p_ws[(((cube_idx * 196608) + (beat_slot_4_1 * 49152)) + (pair_row_v_4 * 128))];  // #1590
                        ub_to_gm_pad(view_231, probability_tiles, 64, 256, 0, 0);  // #1591
                        SetFlag<PIPE_MTE3, PIPE_V>(4);  // #1592
                        PipeBarrier<PIPE_V>();  // c220: V-V hazard #1593
                        vcgadd(ub_reduce_pair.ptr(), ub_sc_7.ptr(), (uint8_t)(128), 1, 1, 8);  // #1593
                        if (0 < pair_len_10) {  // #3788 #3801
                            PipeBarrier<PIPE_V>();  // #1606
                            bool choice_13_state = false;  // #3789
                            if (approx_p) {  // #3795
                                choice_13_state = (approx_g_p + 2) == 0;  // #3790 #3792 #3793
                            } else {
                                choice_13_state = false;  // #3794
                            }
                            const Tensor<float, Position::UB> view_232 = arena.as<float>()[46400];  // #3798
                            vcgadd(ub_sum_s.ptr(), view_232.ptr(), (uint8_t)(8), 1, 2, 16);  // #1620
                            if (approx_p) {  // #3799
                                PipeBarrier<PIPE_V>();  // #1621
                                SetMaskCount();  // #1622
                                SetVectorMask(64);  // #1622
                                vmuls(ub_sum_s.ptr(), ub_sum_s.ptr(), (float)(64.0f), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #1622
                                SetMaskNorm();  // #1622
                                ResetMask();  // #1622
                            }
                            PipeBarrier<PIPE_V>();  // #1625
                            SetMaskCount();  // #1626
                            SetVectorMask(64);  // #1626
                            vadd(ub_bsum_s.ptr(), ub_bsum_s.ptr(), ub_sum_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1626
                            SetMaskNorm();  // #1626
                            ResetMask();  // #1626
                        }
                        if (1 < pair_len_10) {  // #3802 #3815
                            PipeBarrier<PIPE_V>();  // #1629
                            bool choice_14_state = false;  // #3803
                            if (approx_p) {  // #3809
                                choice_14_state = ((approx_g_p + 2) + 1) == 0;  // #3804 #3805 #3806 #3807
                            } else {
                                choice_14_state = false;  // #3808
                            }
                            const Tensor<float, Position::UB> view_233 = arena.as<float>()[46408];  // #3812
                            vcgadd(ub_sum_s.ptr(), view_233.ptr(), (uint8_t)(8), 1, 2, 16);  // #1643
                            if (approx_p) {  // #3813
                                PipeBarrier<PIPE_V>();  // #1644
                                SetMaskCount();  // #1645
                                SetVectorMask(64);  // #1645
                                vmuls(ub_sum_s.ptr(), ub_sum_s.ptr(), (float)(64.0f), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #1645
                                SetMaskNorm();  // #1645
                                ResetMask();  // #1645
                            }
                            PipeBarrier<PIPE_V>();  // #1648
                            SetMaskCount();  // #1649
                            SetVectorMask(64);  // #1649
                            vadd(ub_bsum_s.ptr(), ub_bsum_s.ptr(), ub_sum_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1649
                            SetMaskNorm();  // #1649
                            ResetMask();  // #1649
                        }
                        SetFlag<PIPE_V, PIPE_MTE2>(5);  // #1651
                    }
                    const int32_t pair_len_11 = Min(2, b_len_p_4 - 4);  // #3817 #3818
                    if (pair_len_11 > 0) {  // #3819 #3865
                        const int32_t pair_row_v_5 = 256 + sb_row;  // #1656
                        const Tensor<float, Position::UB> ub_sc_8 = arena.as<float>()[16384];  // #3820
                        const Tensor<float, Position::UB> ub_cols_14 = arena.as<float>()[16384];  // #3821
                        PipeBarrier<PIPE_V>();  // c220: V-V hazard #1659
                        vsub(ub_cols_14.ptr(), ub_cols_14.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #1659
                        const Tensor<float, Position::UB> ub_cols_15 = arena.as<float>()[16448];  // #3822
                        PipeBarrier<PIPE_V>();  // c220: V-V hazard #1661
                        vsub(ub_cols_15.ptr(), ub_cols_15.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #1661
                        PipeBarrier<PIPE_V>();  // #1662
                        vexp(ub_sc_8.ptr(), ub_sc_8.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint8_t)8, (uint8_t)8);  // #1663
                        PipeBarrier<PIPE_V>();  // #1664
                        WaitFlag<PIPE_MTE3, PIPE_V>(4);  // #1665
                        vconv_f322bf16a(probability_tiles.ptr(), ub_sc_8.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)4, (uint16_t)8);  // #1666
                        SetFlag<PIPE_V, PIPE_MTE3>(4);  // #1667
                        WaitFlag<PIPE_V, PIPE_MTE3>(4);  // #1668
                        const GMTensor<bfloat16_t> view_234 = p_ws[(((cube_idx * 196608) + (beat_slot_4_1 * 49152)) + (pair_row_v_5 * 128))];  // #1671
                        ub_to_gm_pad(view_234, probability_tiles, 64, 256, 0, 0);  // #1672
                        SetFlag<PIPE_MTE3, PIPE_V>(4);  // #1673
                        PipeBarrier<PIPE_V>();  // c220: V-V hazard #1674
                        vcgadd(ub_reduce_pair.ptr(), ub_sc_8.ptr(), (uint8_t)(128), 1, 1, 8);  // #1674
                        if (0 < pair_len_11) {  // #3837 #3850
                            PipeBarrier<PIPE_V>();  // #1687
                            bool choice_16_state = false;  // #3838
                            if (approx_p) {  // #3844
                                choice_16_state = (approx_g_p + 4) == 0;  // #3839 #3841 #3842
                            } else {
                                choice_16_state = false;  // #3843
                            }
                            const Tensor<float, Position::UB> view_235 = arena.as<float>()[46400];  // #3847
                            vcgadd(ub_sum_s.ptr(), view_235.ptr(), (uint8_t)(8), 1, 2, 16);  // #1701
                            if (approx_p) {  // #3848
                                PipeBarrier<PIPE_V>();  // #1702
                                SetMaskCount();  // #1703
                                SetVectorMask(64);  // #1703
                                vmuls(ub_sum_s.ptr(), ub_sum_s.ptr(), (float)(64.0f), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #1703
                                SetMaskNorm();  // #1703
                                ResetMask();  // #1703
                            }
                            PipeBarrier<PIPE_V>();  // #1706
                            SetMaskCount();  // #1707
                            SetVectorMask(64);  // #1707
                            vadd(ub_bsum_s.ptr(), ub_bsum_s.ptr(), ub_sum_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1707
                            SetMaskNorm();  // #1707
                            ResetMask();  // #1707
                        }
                        if (1 < pair_len_11) {  // #3851 #3864
                            PipeBarrier<PIPE_V>();  // #1710
                            bool choice_17_state = false;  // #3852
                            if (approx_p) {  // #3858
                                choice_17_state = ((approx_g_p + 4) + 1) == 0;  // #3853 #3854 #3855 #3856
                            } else {
                                choice_17_state = false;  // #3857
                            }
                            const Tensor<float, Position::UB> view_236 = arena.as<float>()[46408];  // #3861
                            vcgadd(ub_sum_s.ptr(), view_236.ptr(), (uint8_t)(8), 1, 2, 16);  // #1724
                            if (approx_p) {  // #3862
                                PipeBarrier<PIPE_V>();  // #1725
                                SetMaskCount();  // #1726
                                SetVectorMask(64);  // #1726
                                vmuls(ub_sum_s.ptr(), ub_sum_s.ptr(), (float)(64.0f), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #1726
                                SetMaskNorm();  // #1726
                                ResetMask();  // #1726
                            }
                            PipeBarrier<PIPE_V>();  // #1729
                            SetMaskCount();  // #1730
                            SetVectorMask(64);  // #1730
                            vadd(ub_bsum_s.ptr(), ub_bsum_s.ptr(), ub_sum_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1730
                            SetMaskNorm();  // #1730
                            ResetMask();  // #1730
                        }
                        SetFlag<PIPE_V, PIPE_MTE2>(6);  // #1732
                    }
                    const int32_t mod_1 = beat & 3;  // #3866
                    const Tensor<float, Position::UB> tile_2 = arena.as<float>()[(45632 + (mod_1 * 64))];  // #3870
                    const Tensor<float, Position::UB> row_diff_1 = arena.as<float>()[(45632 + (mod_1 * 64))];  // #3871
                    PipeBarrier<PIPE_V>();  // c220: V-V hazard #1740
                    SetMaskCount();  // #1740
                    SetVectorMask(64);  // #1740
                    vmul(ub_rsum_s.ptr(), ub_rsum_s.ptr(), row_diff_1.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1740
                    SetMaskNorm();  // #1740
                    ResetMask();  // #1740
                    PipeBarrier<PIPE_V>();  // #1741
                    SetMaskCount();  // #1742
                    SetVectorMask(64);  // #1742
                    vadd(ub_rsum_s.ptr(), ub_rsum_s.ptr(), ub_bsum_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1742
                    SetMaskNorm();  // #1742
                    ResetMask();  // #1742
                    VEC_READY<PIPE_MTE3>(1);  // #1743
                    l0ab_cnt_7_state = l0ab_cnt_6;  // #3872
                    l0c_cnt_7_state = l0c_cnt_6;  // #3873
                    l1k_cnt_7_state = l1k_cnt_6;  // #3874
                } else {
                    l0ab_cnt_7_state = l0ab_cnt_carry_1;  // #3875
                    l0c_cnt_7_state = l0c_cnt_carry_1;  // #3876
                    l1k_cnt_7_state = l1k_cnt_carry_1;  // #3877
                }
                const int32_t l0ab_cnt_7 = (int32_t)(l0ab_cnt_7_state);  // #3879
                const int32_t l0c_cnt_7 = (int32_t)(l0c_cnt_7_state);  // #3880
                const int32_t l1k_cnt_7 = (int32_t)(l1k_cnt_7_state);  // #3881
                b_len_p_5_state = b_len_p_4;  // #3882
                l0ab_cnt_8_state = l0ab_cnt_7;  // #3883
                l0c_cnt_8_state = l0c_cnt_7;  // #3884
                l1k_cnt_8_state = l1k_cnt_7;  // #3885
            } else {
                b_len_p_5_state = 0;  // #3886
                l0ab_cnt_8_state = l0ab_cnt_carry_1;  // #3887
                l0c_cnt_8_state = l0c_cnt_carry_1;  // #3888
                l1k_cnt_8_state = l1k_cnt_carry_1;  // #3889
            }
            const int32_t b_len_p_5 = (int32_t)(b_len_p_5_state);  // #3891
            const int32_t l0ab_cnt_8 = (int32_t)(l0ab_cnt_8_state);  // #3892
            const int32_t l0c_cnt_8 = (int32_t)(l0c_cnt_8_state);  // #3893
            const int32_t l1k_cnt_8 = (int32_t)(l1k_cnt_8_state);  // #3894
            const int32_t ready_lengths_1 = ((ready_lengths_carry & 4095) << 4) | b_len_p_5;  // #3895 #3896 #3897
            const int32_t b_len_c = ready_lengths_1 >> 12;  // #3898
            const bool cmp_83 = b_len_c > 0;  // #3899
            int32_t accum_step_2_state = 0;  // #3900
            int32_t l0ab_cnt_15_state = 0;  // #3901
            int32_t l0c_cnt_10_state = 0;  // #3902
            int32_t l1pv_cnt_7_state = 0;  // #3903
            if (cmp_83) {  // #4366
                const int32_t consumed = beat + -3;  // #3904
                const int32_t slot_24 = consumed & 3;  // #3905
                const bool approx_c = consumed < approx_beats_1;  // #3906
                const int32_t e_base_c = Max(0, consumed - approx_beats_1) * 6;  // #3908 #3909 #3910
                const int32_t pair_len_12 = Min(2, b_len_c);  // #3912
                const bool cmp_84 = pair_len_12 > 0;  // #3913
                int32_t l0ab_cnt_10_state = 0;  // #3914
                int32_t l1pv_cnt_2_state = 0;  // #3915
                if (cmp_84) {  // #4047
                    if (approx_c) {  // #4031
                        if (pair_len_12 < 2) {  // #3921 #3924
                            PipeBarrier<PIPE_MTE2>();  // #3923
                        }
                    } else {
                        const int32_t blk1_12 = e_base_c + 1;  // #3932
                        const bool not_24 = !(dense);  // #3933
                        int32_t blk0_7_state = 0;  // #3934
                        int32_t blk1_15_state = 0;  // #3935
                        if (not_24) {  // #3971
                            const int32_t blk0_6 = (int32_t)(exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + e_base_c));  // #3937 #3938 #3939 #3940 #3941 #3942 #3943 #3944 #3945 #3946 #3947
                            const bool cmp_86 = pair_len_12 == 2;  // #3948
                            int32_t blk1_14_state = 0;  // #3949
                            if (cmp_86) {  // #3965
                                const int32_t add_132 = e_base_c + 1;  // #3950
                                blk1_14_state = (int32_t)(exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + add_132));  // #3952 #3953 #3954 #3955 #3956 #3957 #3958 #3959 #3960 #3961 #3962 #3963
                            } else {
                                blk1_14_state = blk1_12;  // #3964
                            }
                            const int32_t blk1_14 = (int32_t)(blk1_14_state);  // #3966
                            blk0_7_state = blk0_6;  // #3967
                            blk1_15_state = blk1_14;  // #3968
                        } else {
                            blk0_7_state = e_base_c;  // #3969
                            blk1_15_state = blk1_12;  // #3970
                        }
                        const int32_t blk0_7 = (int32_t)(blk0_7_state);  // #3972
                        const int32_t blk1_15 = (int32_t)(blk1_15_state);  // #3973
                        const int32_t start0_3 = blk0_7 * 64;  // #3974
                        const int32_t rows0_3 = Min(64, T - start0_3);  // #3975 #3976
                        const bool cmp_87 = pair_len_12 == 2;  // #3977
                        int32_t rows1_7_state = 0;  // #3978
                        if (cmp_87) {  // #3984
                            rows1_7_state = Min(64, T - (blk1_15 * 64));  // #3979 #3980 #3981 #3982
                        } else {
                            rows1_7_state = 64;  // #3983
                        }
                        const int32_t rows1_7 = (int32_t)(rows1_7_state);  // #3985
                        const bool cmp_88 = pair_len_12 < 2;  // #3986
                        bool choice_19_state = false;  // #3987
                        if (cmp_88) {  // #3997
                            choice_19_state = true;  // #3988
                        } else {
                            const bool cmp_89 = rows0_3 < 64;  // #3989
                            bool choice_18_state = false;  // #3990
                            if (cmp_89) {  // #3994
                                choice_18_state = true;  // #3991
                            } else {
                                choice_18_state = rows1_7 < 64;  // #3992 #3993
                            }
                            choice_19_state = (bool)(choice_18_state);  // #3995 #3996
                        }
                        const bool choice_19 = (bool)(choice_19_state);  // #3998
                        if (choice_19) {  // #4000
                            PipeBarrier<PIPE_MTE2>();  // #3999
                        }
                    }
                    const int32_t l0ab_cnt_9 = (int32_t)(l0ab_cnt_8 + 1);  // #4039 #4040
                    const int32_t l1pv_cnt_1 = (int32_t)(l1pv_cnt_carry_1 + 1);  // #4041 #4042
                    l0ab_cnt_10_state = l0ab_cnt_9;  // #4043
                    l1pv_cnt_2_state = l1pv_cnt_1;  // #4044
                } else {
                    l0ab_cnt_10_state = l0ab_cnt_8;  // #4045
                    l1pv_cnt_2_state = l1pv_cnt_carry_1;  // #4046
                }
                const int32_t l0ab_cnt_10 = (int32_t)(l0ab_cnt_10_state);  // #4048
                const int32_t l1pv_cnt_2 = (int32_t)(l1pv_cnt_2_state);  // #4049
                const int32_t pair_len_13 = Min(2, b_len_c - 2);  // #4050 #4051
                const bool cmp_94 = pair_len_13 > 0;  // #4052
                int32_t l0ab_cnt_12_state = 0;  // #4053
                int32_t l1pv_cnt_4_state = 0;  // #4054
                if (cmp_94) {  // #4186
                    if (approx_c) {  // #4170
                        if (pair_len_13 < 2) {  // #4060 #4063
                            PipeBarrier<PIPE_MTE2>();  // #4062
                        }
                    } else {
                        const int32_t add_145 = e_base_c + 2;  // #4070
                        const int32_t blk1_16 = add_145 + 1;  // #4071
                        const bool not_25 = !(dense);  // #4072
                        int32_t blk0_9_state = 0;  // #4073
                        int32_t blk1_19_state = 0;  // #4074
                        if (not_25) {  // #4110
                            const int32_t blk0_8 = (int32_t)(exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + add_145));  // #4076 #4077 #4078 #4079 #4080 #4081 #4082 #4083 #4084 #4085 #4086
                            const bool cmp_96 = pair_len_13 == 2;  // #4087
                            int32_t blk1_18_state = 0;  // #4088
                            if (cmp_96) {  // #4104
                                const int32_t add_149 = add_145 + 1;  // #4089
                                blk1_18_state = (int32_t)(exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + add_149));  // #4091 #4092 #4093 #4094 #4095 #4096 #4097 #4098 #4099 #4100 #4101 #4102
                            } else {
                                blk1_18_state = blk1_16;  // #4103
                            }
                            const int32_t blk1_18 = (int32_t)(blk1_18_state);  // #4105
                            blk0_9_state = blk0_8;  // #4106
                            blk1_19_state = blk1_18;  // #4107
                        } else {
                            blk0_9_state = add_145;  // #4108
                            blk1_19_state = blk1_16;  // #4109
                        }
                        const int32_t blk0_9 = (int32_t)(blk0_9_state);  // #4111
                        const int32_t blk1_19 = (int32_t)(blk1_19_state);  // #4112
                        const int32_t start0_4 = blk0_9 * 64;  // #4113
                        const int32_t rows0_4 = Min(64, T - start0_4);  // #4114 #4115
                        const bool cmp_97 = pair_len_13 == 2;  // #4116
                        int32_t rows1_9_state = 0;  // #4117
                        if (cmp_97) {  // #4123
                            rows1_9_state = Min(64, T - (blk1_19 * 64));  // #4118 #4119 #4120 #4121
                        } else {
                            rows1_9_state = 64;  // #4122
                        }
                        const int32_t rows1_9 = (int32_t)(rows1_9_state);  // #4124
                        const bool cmp_98 = pair_len_13 < 2;  // #4125
                        bool choice_22_state = false;  // #4126
                        if (cmp_98) {  // #4136
                            choice_22_state = true;  // #4127
                        } else {
                            const bool cmp_99 = rows0_4 < 64;  // #4128
                            bool choice_21_state = false;  // #4129
                            if (cmp_99) {  // #4133
                                choice_21_state = true;  // #4130
                            } else {
                                choice_21_state = rows1_9 < 64;  // #4131 #4132
                            }
                            choice_22_state = (bool)(choice_21_state);  // #4134 #4135
                        }
                        const bool choice_22 = (bool)(choice_22_state);  // #4137
                        if (choice_22) {  // #4139
                            PipeBarrier<PIPE_MTE2>();  // #4138
                        }
                    }
                    const int32_t l0ab_cnt_11 = (int32_t)(l0ab_cnt_10 + 1);  // #4178 #4179
                    const int32_t l1pv_cnt_3 = (int32_t)(l1pv_cnt_2 + 1);  // #4180 #4181
                    l0ab_cnt_12_state = l0ab_cnt_11;  // #4182
                    l1pv_cnt_4_state = l1pv_cnt_3;  // #4183
                } else {
                    l0ab_cnt_12_state = l0ab_cnt_10;  // #4184
                    l1pv_cnt_4_state = l1pv_cnt_2;  // #4185
                }
                const int32_t l0ab_cnt_12 = (int32_t)(l0ab_cnt_12_state);  // #4187
                const int32_t l1pv_cnt_4 = (int32_t)(l1pv_cnt_4_state);  // #4188
                const int32_t pair_len_14 = Min(2, b_len_c - 4);  // #4189 #4190
                const bool cmp_104 = pair_len_14 > 0;  // #4191
                int32_t l0ab_cnt_14_state = 0;  // #4192
                int32_t l1pv_cnt_6_state = 0;  // #4193
                if (cmp_104) {  // #4325
                    if (approx_c) {  // #4309
                        if (pair_len_14 < 2) {  // #4199 #4202
                            PipeBarrier<PIPE_MTE2>();  // #4201
                        }
                    } else {
                        const int32_t add_162 = e_base_c + 4;  // #4209
                        const int32_t blk1_20 = add_162 + 1;  // #4210
                        const bool not_26 = !(dense);  // #4211
                        int32_t blk0_11_state = 0;  // #4212
                        int32_t blk1_23_state = 0;  // #4213
                        if (not_26) {  // #4249
                            const int32_t blk0_10 = (int32_t)(exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + add_162));  // #4215 #4216 #4217 #4218 #4219 #4220 #4221 #4222 #4223 #4224 #4225
                            const bool cmp_106 = pair_len_14 == 2;  // #4226
                            int32_t blk1_22_state = 0;  // #4227
                            if (cmp_106) {  // #4243
                                const int32_t add_166 = add_162 + 1;  // #4228
                                blk1_22_state = (int32_t)(exact_indices.load((((b_1 * ((H * NT_Q) * 64)) + (h_1 * (NT_Q * 64))) + (q_idx * 64)) + add_166));  // #4230 #4231 #4232 #4233 #4234 #4235 #4236 #4237 #4238 #4239 #4240 #4241
                            } else {
                                blk1_22_state = blk1_20;  // #4242
                            }
                            const int32_t blk1_22 = (int32_t)(blk1_22_state);  // #4244
                            blk0_11_state = blk0_10;  // #4245
                            blk1_23_state = blk1_22;  // #4246
                        } else {
                            blk0_11_state = add_162;  // #4247
                            blk1_23_state = blk1_20;  // #4248
                        }
                        const int32_t blk0_11 = (int32_t)(blk0_11_state);  // #4250
                        const int32_t blk1_23 = (int32_t)(blk1_23_state);  // #4251
                        const int32_t start0_5 = blk0_11 * 64;  // #4252
                        const int32_t rows0_5 = Min(64, T - start0_5);  // #4253 #4254
                        const bool cmp_107 = pair_len_14 == 2;  // #4255
                        int32_t rows1_11_state = 0;  // #4256
                        if (cmp_107) {  // #4262
                            rows1_11_state = Min(64, T - (blk1_23 * 64));  // #4257 #4258 #4259 #4260
                        } else {
                            rows1_11_state = 64;  // #4261
                        }
                        const int32_t rows1_11 = (int32_t)(rows1_11_state);  // #4263
                        const bool cmp_108 = pair_len_14 < 2;  // #4264
                        bool choice_25_state = false;  // #4265
                        if (cmp_108) {  // #4275
                            choice_25_state = true;  // #4266
                        } else {
                            const bool cmp_109 = rows0_5 < 64;  // #4267
                            bool choice_24_state = false;  // #4268
                            if (cmp_109) {  // #4272
                                choice_24_state = true;  // #4269
                            } else {
                                choice_24_state = rows1_11 < 64;  // #4270 #4271
                            }
                            choice_25_state = (bool)(choice_24_state);  // #4273 #4274
                        }
                        const bool choice_25 = (bool)(choice_25_state);  // #4276
                        if (choice_25) {  // #4278
                            PipeBarrier<PIPE_MTE2>();  // #4277
                        }
                    }
                    const int32_t l0ab_cnt_13 = (int32_t)(l0ab_cnt_12 + 1);  // #4317 #4318
                    const int32_t l1pv_cnt_5 = (int32_t)(l1pv_cnt_4 + 1);  // #4319 #4320
                    l0ab_cnt_14_state = l0ab_cnt_13;  // #4321
                    l1pv_cnt_6_state = l1pv_cnt_5;  // #4322
                } else {
                    l0ab_cnt_14_state = l0ab_cnt_12;  // #4323
                    l1pv_cnt_6_state = l1pv_cnt_4;  // #4324
                }
                const int32_t l0ab_cnt_14 = (int32_t)(l0ab_cnt_14_state);  // #4326
                const int32_t l1pv_cnt_6 = (int32_t)(l1pv_cnt_6_state);  // #4327
                WAIT_CUBE<PIPE_S>(2);  // #2169
                const int32_t mod_2 = (accum_step_carry + 1) & 1;  // #4330 #4331
                const int32_t start_2 = mod_2 * 64;  // #4332
                const Tensor<float, Position::UB> next_accum = arena.as<float>()[(28672 + (start_2 * 128))];  // #4335
                const GMTensor<float> view_265 = pv_ws[(((cube_idx * 65536) + (slot_24 * 16384)) + (sb_row * 128))];  // #2178
                ev_v_mte2_valid_d1_1.wait();  // #2738
                gm_to_ub_pad(next_accum, view_265, 64, 512, 0, 0);  // #2179
                const int32_t mod_3 = consumed & 3;  // #4336
                const Tensor<float, Position::UB> tile_3 = arena.as<float>()[(45632 + (mod_3 * 64))];  // #4340
                ev_mte2_v_ready_d1_1.set();  // #2739
                ev_mte2_v_ready_d1_1.wait();  // #2740
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #2185
                ResetMask();  // #2185
                vbrcb((__ubuf__ uint32_t*)(ub_expdiff.ptr()), (__ubuf__ uint32_t*)(tile_3.ptr()), (uint16_t)1, (uint16_t)8, (uint8_t)(8));  // #2185
                const int32_t mod_4 = accum_step_carry & 1;  // #4341
                const int32_t start_4 = mod_4 * 64;  // #4342
                const Tensor<float, Position::UB> previous_accum = arena.as<float>()[(28672 + (start_4 * 128))];  // #4345
                const Tensor<float, Position::UB> view_266 = arena.as<float>()[(28672 + (start_2 * 128))];  // #4346
                const Tensor<float, Position::UB> view_267 = arena.as<float>()[(28672 + (start_4 * 128))];  // #4347
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #2195
                vmla(view_266.ptr(), view_267.ptr(), ub_expdiff.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #2195
                const Tensor<float, Position::UB> view_268 = arena.as<float>()[(28672 + ((start_2 * 128) + 64))];  // #4350
                const Tensor<float, Position::UB> view_269 = arena.as<float>()[(28672 + ((start_4 * 128) + 64))];  // #4351
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #2200
                vmla(view_268.ptr(), view_269.ptr(), ub_expdiff.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #2200
                const int32_t accum_step_1 = (int32_t)(accum_step_carry + 1);  // #4354 #4355
                const int32_t l0c_cnt_9 = (int32_t)(l0c_cnt_8 + 1);  // #4356 #4357
                accum_step_2_state = accum_step_1;  // #4358
                l0ab_cnt_15_state = l0ab_cnt_14;  // #4359
                l0c_cnt_10_state = l0c_cnt_9;  // #4360
                l1pv_cnt_7_state = l1pv_cnt_6;  // #4361
                ev_v_mte2_valid_d1_1.set();  // #2742
            } else {
                accum_step_2_state = accum_step_carry;  // #4362
                l0ab_cnt_15_state = l0ab_cnt_8;  // #4363
                l0c_cnt_10_state = l0c_cnt_8;  // #4364
                l1pv_cnt_7_state = l1pv_cnt_carry_1;  // #4365
            }
            const int32_t accum_step_2 = (int32_t)(accum_step_2_state);  // #4367
            const int32_t l0ab_cnt_15 = (int32_t)(l0ab_cnt_15_state);  // #4368
            const int32_t l0c_cnt_10 = (int32_t)(l0c_cnt_10_state);  // #4369
            const int32_t l1pv_cnt_7 = (int32_t)(l1pv_cnt_7_state);  // #4370
            accum_step_3_state = accum_step_2;  // #4371
            l0ab_cnt_16_state = l0ab_cnt_15;  // #4372
            l0c_cnt_11_state = l0c_cnt_10;  // #4373
            l1k_cnt_9_state = l1k_cnt_8;  // #4374
            l1pv_cnt_8_state = l1pv_cnt_7;  // #4375
            ready_lengths_2_state = ready_lengths_1;  // #4376
        }
        ev_mte2_v_ready_d0_1.set();  // #2746
        ev_mte2_v_ready_d0_1.wait();  // #2747
        const int32_t accum_step_3 = (int32_t)(accum_step_3_state);  // #4378
        const int32_t l0ab_cnt_16 = (int32_t)(l0ab_cnt_16_state);  // #4379
        const int32_t l0c_cnt_11 = (int32_t)(l0c_cnt_11_state);  // #4380
        const int32_t l1k_cnt_9 = (int32_t)(l1k_cnt_9_state);  // #4381
        const int32_t l1pv_cnt_8 = (int32_t)(l1pv_cnt_8_state);  // #4382
        WaitFlag<PIPE_V, PIPE_MTE2>(4);  // #2211
        WaitFlag<PIPE_V, PIPE_MTE2>(5);  // #2212
        WaitFlag<PIPE_V, PIPE_MTE2>(6);  // #2213
        WaitFlag<PIPE_MTE3, PIPE_V>(4);  // #2214
        PipeBarrier<PIPE_ALL>();  // #4383
        const int32_t mod_5 = accum_step_3 & 1;  // #4384
        const int32_t start_5 = mod_5 * 64;  // #4385
        const Tensor<float, Position::UB> accum_ub = arena.as<float>()[(28672 + (start_5 * 128))];  // #4388
        ResetMask();  // #2222
        vbrcb((__ubuf__ uint32_t*)(ub_rowsum.ptr()), (__ubuf__ uint32_t*)(ub_rsum_s.ptr()), (uint16_t)1, (uint16_t)8, (uint8_t)(8));  // #2222
        const Tensor<float, Position::UB> view_270 = arena.as<float>()[(28672 + (start_5 * 128))];  // #4389
        const Tensor<float, Position::UB> view_271 = arena.as<float>()[(28672 + (start_5 * 128))];  // #4390
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #2225
        vdiv(view_270.ptr(), view_271.ptr(), ub_rowsum.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #2225
        const Tensor<float, Position::UB> view_272 = arena.as<float>()[(28672 + ((start_5 * 128) + 64))];  // #4391
        const Tensor<float, Position::UB> view_273 = arena.as<float>()[(28672 + ((start_5 * 128) + 64))];  // #4392
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #2228
        vdiv(view_272.ptr(), view_273.ptr(), ub_rowsum.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #2228
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #2229
        vconv_f322bf16a(probability_tiles.ptr(), accum_ub.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)4, (uint16_t)8);  // #2229
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #2230
        vln(ub_sum_s.ptr(), ub_rsum_s.ptr(), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint8_t)0, (uint8_t)0);  // #2230
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #2231
        vadd(ub_rmax_s.ptr(), ub_rmax_s.ptr(), ub_sum_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #2231
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #2232
        vmuls(ub_rmax_s.ptr(), ub_rmax_s.ptr(), (float)(1.4426950408889634f), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #2232
        ev_mte2_mte3_ready_1.set();  // #2719
        ev_mte2_mte3_ready_1.wait();  // #2720
        ev_v_mte3_ready_1.set();  // #2757
        ev_v_mte3_ready_1.wait();  // #2758
        if (local_valid_m > 0) {  // #2233 #2245
            const int32_t out_row = q_start + sb_row;  // #2234
            const GMTensor<bfloat16_t> view_274 = out[(((b_1 * ((T * H) * 128)) + (out_row * (H * 128))) + (h_1 * 128))];  // #2237
            const Tensor<bfloat16_t, Position::UB> view_275 = arena.as<bfloat16_t>()[49152];  // #2238
            ub_to_gm_pad(view_274, view_275, local_valid_m, 256, 0, ((128 + ((H * 128) - 128)) * 2) - 256);  // #2703 #2704 #2705 #2706 #2707 #2239
            const GMTensor<float> view_276 = lse[(((b_1 * (H * T_LSE)) + (h_1 * T_LSE)) + out_row)];  // #2242
            const Tensor<float, Position::UB> view_277 = arena.as<float>()[45312];  // #2243
            const int32_t sub_117 = 64 - local_valid_m;  // #2711
            const int32_t idiv_57 = sub_117 / 8;  // #4459
            const int32_t idiv_58 = sub_117 % 8;  // #4460
            const bool idiv_59 = idiv_58 != 0;  // #4461
            const bool idiv_60 = sub_117 < 0;  // #4462
            const bool idiv_62 = idiv_60 != false;  // #4464
            const bool idiv_63 = idiv_59 && idiv_62;  // #4465
            const int32_t idiv_64 = (idiv_63) ? (1) : (0);  // #4466
            const int32_t div_1 = idiv_57 - idiv_64;  // #2712
            ub_to_gm_pad(view_276, view_277, 1, local_valid_m * 4, div_1, 0);  // #2708 #2244
        }
        l0ab_cnt_17_state = l0ab_cnt_16;  // #4393
        l0c_cnt_12_state = l0c_cnt_11;  // #4394
        l1k_cnt_10_state = l1k_cnt_9;  // #4395
        l1pv_cnt_9_state = l1pv_cnt_8;  // #4396
        ev_mte3_mte2_valid_1.set();  // #2722
        ev_v_mte2_valid_d0_1.set();  // #2748
        ev_mte3_v_valid_1.set();  // #2762
    }
    WAIT_CUBE<PIPE_S>(1);  // #4398
    WAIT_CUBE<PIPE_S>(1);  // #4398
    return;  // #4398
}
