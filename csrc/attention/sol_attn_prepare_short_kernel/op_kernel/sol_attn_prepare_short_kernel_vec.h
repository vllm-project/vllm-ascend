#pragma once
using namespace ascrip;

__aicore__ inline void sol_attn_prepare_short_kernel_vec(GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR k_pad_, GM_ADDR v_pad_, GM_ADDR kc_, GM_ADDR vc_, GM_ADDR last_group_lens_, GM_ADDR threshold_, GM_ADDR approx_mask_, GM_ADDR exact_counts_, GM_ADDR exact_indices_, int32_t B, int32_t T, int32_t H, int32_t NT, int32_t T_PAD_KV, int32_t NT_Q, int32_t NT_Q_PAD8, int32_t query_blocks_per_tile, int32_t sink_start, int32_t sink_end, float scale_log2, float scale_sq, float tau, GM_ADDR workspace)
{
    GMTensor<bfloat16_t> q((__gm__ bfloat16_t*)q_);
    GMTensor<bfloat16_t> k((__gm__ bfloat16_t*)k_);
    GMTensor<bfloat16_t> v((__gm__ bfloat16_t*)v_);
    GMTensor<bfloat16_t> k_pad((__gm__ bfloat16_t*)k_pad_);
    GMTensor<bfloat16_t> v_pad((__gm__ bfloat16_t*)v_pad_);
    GMTensor<bfloat16_t> kc((__gm__ bfloat16_t*)kc_);
    GMTensor<bfloat16_t> vc((__gm__ bfloat16_t*)vc_);
    GMTensor<float> last_group_lens((__gm__ float*)last_group_lens_);
    GMTensor<float> threshold((__gm__ float*)threshold_);
    GMTensor<uint64_t> approx_mask((__gm__ uint64_t*)approx_mask_);
    GMTensor<int32_t> exact_counts((__gm__ int32_t*)exact_counts_);
    GMTensor<uint16_t> exact_indices((__gm__ uint16_t*)exact_indices_);
    SetAtomicNone();
    ResetMask();
    SetMaskNorm();
    SEvent<PIPE_MTE2, PIPE_MTE3, 0, 0> ev_mte2_mte3_ready_1;  // guards k_block, v_block #465
    SEvent<PIPE_MTE3, PIPE_MTE2, 1, 0> ev_mte3_mte2_valid_1;  // guards k_block, v_block #466
    SEvent<PIPE_MTE2, PIPE_V, 0, 0> ev_mte2_v_ready_1;  // guards k_block, q_block, v_block #471
    SEvent<PIPE_V, PIPE_MTE2, 1, 0> ev_v_mte2_valid_1;  // guards k_block, q_block, v_block #472
    SEvent<PIPE_V, PIPE_MTE3, 0, 0> ev_v_mte3_ready_d0_1;  // guards k_block, k_rows, threshold_rows, v_block, v_rows #479
    SEvent<PIPE_MTE3, PIPE_V, 1, 0> ev_mte3_v_valid_d0_1;  // guards k_block, k_rows, threshold_rows, v_block, v_rows #480
    SEvent<PIPE_V, PIPE_MTE3, 0, 1> ev_v_mte3_ready_d1_1;  // guards k_block, k_rows, threshold_rows, v_block, v_rows #481
    SEvent<PIPE_MTE3, PIPE_V, 1, 1> ev_mte3_v_valid_d1_1;  // guards k_block, k_rows, threshold_rows, v_block, v_rows #482
    const Tensor<bfloat16_t, Position::UB> q_block((uint64_t)(0));  // #1
    const Tensor<bfloat16_t, Position::UB> k_block((uint64_t)(16384));  // #2
    const Tensor<bfloat16_t, Position::UB> v_block((uint64_t)(32768));  // #3
    const Tensor<float, Position::UB> sums((uint64_t)(49152));  // #4
    const Tensor<float, Position::UB> q_rows((uint64_t)(81920));  // #5
    const Tensor<bfloat16_t, Position::UB> k_rows((uint64_t)(114688));  // #6
    const Tensor<bfloat16_t, Position::UB> v_rows((uint64_t)(131072));  // #7
    const Tensor<float, Position::UB> keys((uint64_t)(147456));  // #8
    const Tensor<float, Position::UB> mean_k((uint64_t)(180224));  // #9
    const Tensor<float, Position::UB> var_k((uint64_t)(180736));  // #10
    const Tensor<float, Position::UB> moment_k((uint64_t)(181248));  // #11
    const Tensor<float, Position::UB> query_mean((uint64_t)(181760));  // #12
    const Tensor<float, Position::UB> query_var((uint64_t)(182016));  // #13
    const Tensor<float, Position::UB> partial((uint64_t)(182272));  // #14
    const Tensor<float, Position::UB> lens((uint64_t)(182528));  // #15
    const Tensor<float, Position::UB> scores((uint64_t)(182784));  // #16
    const Tensor<uint8_t, Position::UB> cmp((uint64_t)(183040));  // #17
    const Tensor<uint64_t, Position::UB> cmp_bits = cmp.as<uint64_t>();  // #18
    const Tensor<uint64_t, Position::UB> route_pattern((uint64_t)(183072));  // #19
    const Tensor<uint16_t, Position::UB> source_indices((uint64_t)(183104));  // #20
    const Tensor<uint64_t, Position::UB> route_masks((uint64_t)(183360));  // #21
    const Tensor<int32_t, Position::UB> counts((uint64_t)(183872));  // #22
    const Tensor<uint16_t, Position::UB> compact_indices((uint64_t)(184128));  // #23
    const Tensor<float, Position::UB> threshold_rows((uint64_t)(184384));  // #24
    SEvent<PIPE_S, PIPE_V, 0, 0> scalar_publish;  // #25
    for (int32_t i = 0; i < 128; i += 1) {  // #441
        source_indices.store(i, (uint16_t)((uint16_t)(i)));  // #26 #28
    }
    scalar_publish.set();  // #31
    scalar_publish.wait();  // #32
    const uint64_t valid = 18446744073709551615ULL >> (64 - NT);  // #34 #35
    const int32_t sub_1 = H + -1;  // #36
    const int32_t src_gap = sub_1 * 128;  // #37
    const float inv_nt = 1.0f / (float)(NT);  // #38 #39
    const int32_t vec_id = GetVecIdx();  // #40
    const int32_t mul = B * H;  // #41
    const int32_t vec_count = GetVecNum();  // #42
    for (int32_t bh = vec_id; bh < mul; bh += vec_count) {  // #456
        const int32_t idiv_1 = bh / H;  // #499
        const int32_t idiv_2 = bh % H;  // #500
        const bool idiv_3 = idiv_2 != 0;  // #501
        const bool idiv_4 = bh < 0;  // #502
        const bool idiv_6 = idiv_4 != false;  // #504
        const bool idiv_7 = idiv_3 && idiv_6;  // #505
        const int32_t idiv_8 = (idiv_7) ? (1) : (0);  // #506
        const int32_t b = idiv_1 - idiv_8;  // #43
        const int32_t h = ascrip::FloorMod<int32_t>(bh, H);  // #44
        ev_mte3_v_valid_d0_1.wait();  // #483
        if (NT < 64) {  // #45 #51
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #46
            vector_dup(q_rows.ptr(), (float)(0.0f), (uint8_t)(128), (uint16_t)1, 1, (uint8_t)8, 0);  // #46
            const Tensor<uint32_t, Position::UB> view = k_rows.as<uint32_t>();  // #47
            vector_dup(view.ptr(), (uint32_t)(0U), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)8, 0);  // #48
            const Tensor<uint32_t, Position::UB> view_1 = v_rows.as<uint32_t>();  // #49
            vector_dup(view_1.ptr(), (uint32_t)(0U), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)8, 0);  // #50
        }
        for (int32_t block = 0; block < NT; block += 1) {  // #442
            const int32_t tok = block * 64;  // #52
            const int32_t rows = Min(T - tok, 64);  // #53 #54
            ev_v_mte2_valid_1.wait();  // #473
            ev_mte3_v_valid_d1_1.wait();  // #488
            if (rows < 64) {  // #55 #62
                const Tensor<uint32_t, Position::UB> view_2 = q_block.as<uint32_t>();  // #56
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #57
                vector_dup(view_2.ptr(), (uint32_t)(0U), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)8, 0);  // #57
                const Tensor<uint32_t, Position::UB> view_3 = k_block.as<uint32_t>();  // #58
                vector_dup(view_3.ptr(), (uint32_t)(0U), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)8, 0);  // #59
                const Tensor<uint32_t, Position::UB> view_4 = v_block.as<uint32_t>();  // #60
                vector_dup(view_4.ptr(), (uint32_t)(0U), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)8, 0);  // #61
            }
            const GMTensor<bfloat16_t> view_5 = q[(((b * ((T * H) * 128)) + (tok * (H * 128))) + (h * 128))];  // #65
            const int32_t mul_5 = src_gap * 2;  // #458
            ev_v_mte2_valid_1.set();  // #474
            ev_v_mte2_valid_1.wait();  // #475
            gm_to_ub_pad(q_block, view_5, rows, 256, mul_5, 0);  // #66
            const GMTensor<bfloat16_t> view_6 = k[(((b * ((T * H) * 128)) + (tok * (H * 128))) + (h * 128))];  // #69
            ev_mte3_mte2_valid_1.wait();  // #467
            gm_to_ub_pad(k_block, view_6, rows, 256, mul_5, 0);  // #70
            const GMTensor<bfloat16_t> view_7 = v[(((b * ((T * H) * 128)) + (tok * (H * 128))) + (h * 128))];  // #73
            gm_to_ub_pad(v_block, view_7, rows, 256, mul_5, 0);  // #74
            const GMTensor<bfloat16_t> view_8 = k_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (tok * 128))];  // #77
            ev_mte2_mte3_ready_1.set();  // #468
            ev_mte2_mte3_ready_1.wait();  // #469
            ev_v_mte3_ready_d1_1.set();  // #489
            ev_v_mte3_ready_d1_1.wait();  // #490
            ub_to_gm_pad(view_8, k_block, 64, 256, 0, 0);  // #78
            const GMTensor<bfloat16_t> view_9 = v_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (tok * 128))];  // #81
            ub_to_gm_pad(view_9, v_block, 64, 256, 0, 0);  // #82
            const float inv_rows = 1.0f / (float)(rows);  // #83 #84
            ev_mte2_v_ready_1.set();  // #476
            ev_mte2_v_ready_1.wait();  // #477
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #85
            vconv_bf162f32(sums.ptr(), q_block.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)4);  // #85
            const Tensor<float, Position::UB> view_10 = sums;  // #86
            const Tensor<float, Position::UB> view_11 = sums;  // #87
            const Tensor<float, Position::UB> view_12 = sums[4096];  // #88
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #89
            vadd(view_10.ptr(), view_11.ptr(), view_12.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #89
            const Tensor<float, Position::UB> view_13 = sums;  // #90
            const Tensor<float, Position::UB> view_14 = sums;  // #91
            const Tensor<float, Position::UB> view_15 = sums[2048];  // #92
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #93
            vadd(view_13.ptr(), view_14.ptr(), view_15.ptr(), (uint8_t)(32), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #93
            const Tensor<float, Position::UB> view_16 = sums;  // #94
            const Tensor<float, Position::UB> view_17 = sums;  // #95
            const Tensor<float, Position::UB> view_18 = sums[1024];  // #96
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #97
            vadd(view_16.ptr(), view_17.ptr(), view_18.ptr(), (uint8_t)(16), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #97
            const Tensor<float, Position::UB> view_19 = sums;  // #98
            const Tensor<float, Position::UB> view_20 = sums;  // #99
            const Tensor<float, Position::UB> view_21 = sums[512];  // #100
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #101
            vadd(view_19.ptr(), view_20.ptr(), view_21.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #101
            const Tensor<float, Position::UB> view_22 = sums;  // #102
            const Tensor<float, Position::UB> view_23 = sums;  // #103
            const Tensor<float, Position::UB> view_24 = sums[256];  // #104
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #105
            vadd(view_22.ptr(), view_23.ptr(), view_24.ptr(), (uint8_t)(4), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #105
            const Tensor<float, Position::UB> view_25 = sums;  // #106
            const Tensor<float, Position::UB> view_26 = sums;  // #107
            const Tensor<float, Position::UB> view_27 = sums[128];  // #108
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #109
            vadd(view_25.ptr(), view_26.ptr(), view_27.ptr(), (uint8_t)(2), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #109
            const Tensor<float, Position::UB> view_28 = q_rows[(block * 128)];  // #112
            const Tensor<float, Position::UB> view_29 = sums;  // #113
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #116
            vmuls(view_28.ptr(), view_29.ptr(), (float)(inv_rows), (uint8_t)(2), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #116
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #117
            vconv_bf162f32(sums.ptr(), k_block.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)4);  // #117
            const Tensor<float, Position::UB> view_30 = sums;  // #118
            const Tensor<float, Position::UB> view_31 = sums;  // #119
            const Tensor<float, Position::UB> view_32 = sums[4096];  // #120
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #121
            vadd(view_30.ptr(), view_31.ptr(), view_32.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #121
            const Tensor<float, Position::UB> view_33 = sums;  // #122
            const Tensor<float, Position::UB> view_34 = sums;  // #123
            const Tensor<float, Position::UB> view_35 = sums[2048];  // #124
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #125
            vadd(view_33.ptr(), view_34.ptr(), view_35.ptr(), (uint8_t)(32), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #125
            const Tensor<float, Position::UB> view_36 = sums;  // #126
            const Tensor<float, Position::UB> view_37 = sums;  // #127
            const Tensor<float, Position::UB> view_38 = sums[1024];  // #128
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #129
            vadd(view_36.ptr(), view_37.ptr(), view_38.ptr(), (uint8_t)(16), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #129
            const Tensor<float, Position::UB> view_39 = sums;  // #130
            const Tensor<float, Position::UB> view_40 = sums;  // #131
            const Tensor<float, Position::UB> view_41 = sums[512];  // #132
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #133
            vadd(view_39.ptr(), view_40.ptr(), view_41.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #133
            const Tensor<float, Position::UB> view_42 = sums;  // #134
            const Tensor<float, Position::UB> view_43 = sums;  // #135
            const Tensor<float, Position::UB> view_44 = sums[256];  // #136
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #137
            vadd(view_42.ptr(), view_43.ptr(), view_44.ptr(), (uint8_t)(4), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #137
            const Tensor<float, Position::UB> view_45 = sums;  // #138
            const Tensor<float, Position::UB> view_46 = sums;  // #139
            const Tensor<float, Position::UB> view_47 = sums[128];  // #140
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #141
            vadd(view_45.ptr(), view_46.ptr(), view_47.ptr(), (uint8_t)(2), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #141
            const Tensor<float, Position::UB> view_48 = sums;  // #142
            const Tensor<float, Position::UB> view_49 = sums;  // #143
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #144
            vmuls(view_48.ptr(), view_49.ptr(), (float)(inv_rows), (uint8_t)(2), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #144
            const Tensor<bfloat16_t, Position::UB> view_50 = k_rows[(block * 128)];  // #147
            const Tensor<float, Position::UB> view_51 = sums;  // #148
            ev_mte3_v_valid_d1_1.set();  // #491
            ev_mte3_v_valid_d1_1.wait();  // #492
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #151
            vconv_f322bf16r(view_50.ptr(), view_51.ptr(), (uint8_t)(2), (uint16_t)1, (uint16_t)1, (uint16_t)4, (uint16_t)8);  // #151
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #152
            vconv_bf162f32(sums.ptr(), v_block.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)4);  // #152
            const Tensor<float, Position::UB> view_52 = sums;  // #153
            const Tensor<float, Position::UB> view_53 = sums;  // #154
            const Tensor<float, Position::UB> view_54 = sums[4096];  // #155
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #156
            vadd(view_52.ptr(), view_53.ptr(), view_54.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #156
            const Tensor<float, Position::UB> view_55 = sums;  // #157
            const Tensor<float, Position::UB> view_56 = sums;  // #158
            const Tensor<float, Position::UB> view_57 = sums[2048];  // #159
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #160
            vadd(view_55.ptr(), view_56.ptr(), view_57.ptr(), (uint8_t)(32), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #160
            const Tensor<float, Position::UB> view_58 = sums;  // #161
            const Tensor<float, Position::UB> view_59 = sums;  // #162
            const Tensor<float, Position::UB> view_60 = sums[1024];  // #163
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #164
            vadd(view_58.ptr(), view_59.ptr(), view_60.ptr(), (uint8_t)(16), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #164
            const Tensor<float, Position::UB> view_61 = sums;  // #165
            const Tensor<float, Position::UB> view_62 = sums;  // #166
            const Tensor<float, Position::UB> view_63 = sums[512];  // #167
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #168
            vadd(view_61.ptr(), view_62.ptr(), view_63.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #168
            const Tensor<float, Position::UB> view_64 = sums;  // #169
            const Tensor<float, Position::UB> view_65 = sums;  // #170
            const Tensor<float, Position::UB> view_66 = sums[256];  // #171
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #172
            vadd(view_64.ptr(), view_65.ptr(), view_66.ptr(), (uint8_t)(4), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #172
            const Tensor<float, Position::UB> view_67 = sums;  // #173
            const Tensor<float, Position::UB> view_68 = sums;  // #174
            const Tensor<float, Position::UB> view_69 = sums[128];  // #175
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #176
            vadd(view_67.ptr(), view_68.ptr(), view_69.ptr(), (uint8_t)(2), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #176
            const Tensor<bfloat16_t, Position::UB> view_70 = v_rows[(block * 128)];  // #179
            const Tensor<float, Position::UB> view_71 = sums;  // #180
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #183
            vconv_f322bf16r(view_70.ptr(), view_71.ptr(), (uint8_t)(2), (uint16_t)1, (uint16_t)1, (uint16_t)4, (uint16_t)8);  // #183
            ev_mte3_mte2_valid_1.set();  // #470
            ev_v_mte2_valid_1.set();  // #478
            ev_v_mte3_ready_d1_1.set();  // #493
            ev_v_mte3_ready_d1_1.wait();  // #494
            ev_mte3_v_valid_d1_1.set();  // #495
        }
        ev_v_mte3_ready_d0_1.set();  // #496
        ev_v_mte3_ready_d0_1.wait();  // #497
        const GMTensor<bfloat16_t> view_72 = kc[((b * ((H * 64) * 128)) + (h * 8192))];  // #186
        ub_to_gm_pad(view_72, k_rows, 64, 256, 0, 0);  // #187
        const GMTensor<bfloat16_t> view_73 = vc[((b * ((H * 64) * 128)) + (h * 8192))];  // #188
        ub_to_gm_pad(view_73, v_rows, 64, 256, 0, 0);  // #189
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #190
        vconv_bf162f32(sums.ptr(), k_rows.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)4);  // #190
        const Tensor<float, Position::UB> view_74 = sums;  // #191
        const Tensor<float, Position::UB> view_75 = sums;  // #192
        const Tensor<float, Position::UB> view_76 = sums[4096];  // #193
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #194
        vadd(view_74.ptr(), view_75.ptr(), view_76.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #194
        const Tensor<float, Position::UB> view_77 = sums;  // #195
        const Tensor<float, Position::UB> view_78 = sums;  // #196
        const Tensor<float, Position::UB> view_79 = sums[2048];  // #197
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #198
        vadd(view_77.ptr(), view_78.ptr(), view_79.ptr(), (uint8_t)(32), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #198
        const Tensor<float, Position::UB> view_80 = sums;  // #199
        const Tensor<float, Position::UB> view_81 = sums;  // #200
        const Tensor<float, Position::UB> view_82 = sums[1024];  // #201
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #202
        vadd(view_80.ptr(), view_81.ptr(), view_82.ptr(), (uint8_t)(16), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #202
        const Tensor<float, Position::UB> view_83 = sums;  // #203
        const Tensor<float, Position::UB> view_84 = sums;  // #204
        const Tensor<float, Position::UB> view_85 = sums[512];  // #205
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #206
        vadd(view_83.ptr(), view_84.ptr(), view_85.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #206
        const Tensor<float, Position::UB> view_86 = sums;  // #207
        const Tensor<float, Position::UB> view_87 = sums;  // #208
        const Tensor<float, Position::UB> view_88 = sums[256];  // #209
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #210
        vadd(view_86.ptr(), view_87.ptr(), view_88.ptr(), (uint8_t)(4), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #210
        const Tensor<float, Position::UB> view_89 = sums;  // #211
        const Tensor<float, Position::UB> view_90 = sums;  // #212
        const Tensor<float, Position::UB> view_91 = sums[128];  // #213
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #214
        vadd(view_89.ptr(), view_90.ptr(), view_91.ptr(), (uint8_t)(2), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #214
        const Tensor<float, Position::UB> view_92 = sums;  // #215
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #216
        vmuls(mean_k.ptr(), view_92.ptr(), (float)(inv_nt), (uint8_t)(2), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #216
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #217
        vconv_bf162f32(sums.ptr(), k_rows.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)4);  // #217
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #218
        vmul(sums.ptr(), sums.ptr(), sums.ptr(), (uint8_t)(128), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #218
        const Tensor<float, Position::UB> view_93 = sums;  // #219
        const Tensor<float, Position::UB> view_94 = sums;  // #220
        const Tensor<float, Position::UB> view_95 = sums[4096];  // #221
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #222
        vadd(view_93.ptr(), view_94.ptr(), view_95.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #222
        const Tensor<float, Position::UB> view_96 = sums;  // #223
        const Tensor<float, Position::UB> view_97 = sums;  // #224
        const Tensor<float, Position::UB> view_98 = sums[2048];  // #225
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #226
        vadd(view_96.ptr(), view_97.ptr(), view_98.ptr(), (uint8_t)(32), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #226
        const Tensor<float, Position::UB> view_99 = sums;  // #227
        const Tensor<float, Position::UB> view_100 = sums;  // #228
        const Tensor<float, Position::UB> view_101 = sums[1024];  // #229
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #230
        vadd(view_99.ptr(), view_100.ptr(), view_101.ptr(), (uint8_t)(16), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #230
        const Tensor<float, Position::UB> view_102 = sums;  // #231
        const Tensor<float, Position::UB> view_103 = sums;  // #232
        const Tensor<float, Position::UB> view_104 = sums[512];  // #233
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #234
        vadd(view_102.ptr(), view_103.ptr(), view_104.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #234
        const Tensor<float, Position::UB> view_105 = sums;  // #235
        const Tensor<float, Position::UB> view_106 = sums;  // #236
        const Tensor<float, Position::UB> view_107 = sums[256];  // #237
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #238
        vadd(view_105.ptr(), view_106.ptr(), view_107.ptr(), (uint8_t)(4), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #238
        const Tensor<float, Position::UB> view_108 = sums;  // #239
        const Tensor<float, Position::UB> view_109 = sums;  // #240
        const Tensor<float, Position::UB> view_110 = sums[128];  // #241
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #242
        vadd(view_108.ptr(), view_109.ptr(), view_110.ptr(), (uint8_t)(2), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #242
        const Tensor<float, Position::UB> view_111 = sums;  // #243
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #244
        vmuls(moment_k.ptr(), view_111.ptr(), (float)(inv_nt), (uint8_t)(2), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #244
        vmul(var_k.ptr(), mean_k.ptr(), mean_k.ptr(), (uint8_t)(2), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #245
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #246
        vsub(var_k.ptr(), moment_k.ptr(), var_k.ptr(), (uint8_t)(2), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #246
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #247
        vmaxs(var_k.ptr(), var_k.ptr(), (float)(0.0f), (uint8_t)(2), (uint16_t)1, (uint16_t)1, (uint8_t)8, (uint8_t)8, false, false);  // #247
        const Tensor<float, Position::UB> view_112 = sums;  // #248
        const Tensor<float, Position::UB> view_113 = q_rows;  // #249
        const Tensor<float, Position::UB> view_114 = mean_k;  // #250
        vmul(view_112.ptr(), view_113.ptr(), view_114.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)16, (uint8_t)16, (uint8_t)0);  // #251
        const Tensor<float, Position::UB> view_115 = sums[64];  // #252
        const Tensor<float, Position::UB> view_116 = q_rows[64];  // #253
        const Tensor<float, Position::UB> view_117 = mean_k[64];  // #254
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #255
        vmul(view_115.ptr(), view_116.ptr(), view_117.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)16, (uint8_t)16, (uint8_t)0);  // #255
        const Tensor<float, Position::UB> view_118 = sums;  // #256
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #257
        vcadd(query_mean.ptr(), view_118.ptr(), (uint8_t)(64), 1, 1, 16, 0);  // #257
        const Tensor<float, Position::UB> view_119 = sums[64];  // #258
        vcadd(partial.ptr(), view_119.ptr(), (uint8_t)(64), 1, 1, 16, 0);  // #259
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #260
        vadd(query_mean.ptr(), query_mean.ptr(), partial.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #260
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #261
        vmuls(query_mean.ptr(), query_mean.ptr(), (float)(scale_log2), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #261
        vmul(sums.ptr(), q_rows.ptr(), q_rows.ptr(), (uint8_t)(128), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #262
        const Tensor<float, Position::UB> view_120 = sums;  // #263
        const Tensor<float, Position::UB> view_121 = sums;  // #264
        const Tensor<float, Position::UB> view_122 = var_k;  // #265
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #266
        vmul(view_120.ptr(), view_121.ptr(), view_122.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)16, (uint8_t)16, (uint8_t)0);  // #266
        const Tensor<float, Position::UB> view_123 = sums[64];  // #267
        const Tensor<float, Position::UB> view_124 = sums[64];  // #268
        const Tensor<float, Position::UB> view_125 = var_k[64];  // #269
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #270
        vmul(view_123.ptr(), view_124.ptr(), view_125.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)16, (uint8_t)16, (uint8_t)0);  // #270
        const Tensor<float, Position::UB> view_126 = sums;  // #271
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #272
        vcadd(query_var.ptr(), view_126.ptr(), (uint8_t)(64), 1, 1, 16, 0);  // #272
        const Tensor<float, Position::UB> view_127 = sums[64];  // #273
        vcadd(partial.ptr(), view_127.ptr(), (uint8_t)(64), 1, 1, 16, 0);  // #274
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #275
        vadd(query_var.ptr(), query_var.ptr(), partial.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #275
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #276
        vmuls(query_var.ptr(), query_var.ptr(), (float)(scale_sq), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #276
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #277
        vmaxs(query_var.ptr(), query_var.ptr(), (float)(0.0f), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint8_t)0, (uint8_t)0, false, false);  // #277
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #278
        vadds(query_var.ptr(), query_var.ptr(), (float)(1e-06f), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #278
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #279
        vsqrt(query_var.ptr(), query_var.ptr(), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint8_t)0, (uint8_t)0);  // #279
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #280
        vmuls(query_var.ptr(), query_var.ptr(), (float)(tau), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #280
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #281
        vadd(query_mean.ptr(), query_mean.ptr(), query_var.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #281
        ev_mte3_v_valid_d0_1.set();  // #484
        ev_mte3_v_valid_d0_1.wait();  // #485
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #282
        ResetMask();  // #282
        vbrcb((__ubuf__ uint32_t*)(threshold_rows.ptr()), (__ubuf__ uint32_t*)(query_mean.ptr()), (uint16_t)1, (uint16_t)8, (uint8_t)(8));  // #282
        const GMTensor<float> view_128 = threshold[((b * (NT * H)) + h)];  // #283
        const int32_t sub_11 = H + -1;  // #284
        ev_v_mte3_ready_d0_1.set();  // #486
        ev_v_mte3_ready_d0_1.wait();  // #487
        ub_to_gm_pad(view_128, threshold_rows, NT, 4, 0, sub_11 * 4);  // #461 #285
        PipeBarrier<PIPE_ALL>();  // #287
        vconv_bf162f32(keys.ptr(), k_rows.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)4);  // #288
        vector_dup(counts.ptr(), (int32_t)(0), (uint8_t)(1), (uint16_t)1, 1, (uint8_t)0, 0);  // #289
        SetFlag<PIPE_V, PIPE_S>(6);  // #290
        WaitFlag<PIPE_V, PIPE_S>(6);  // #291
        for (int32_t qi = 0; qi < NT_Q; qi += 1) {  // #455
            const int32_t q0 = qi * query_blocks_per_tile;  // #292
            const int32_t sub_13 = NT + -1;  // #295
            const int32_t q1 = Min((q0 + query_blocks_per_tile) - 1, sub_13);  // #293 #294 #296
            const float th0 = query_mean.load(q0);  // #298
            const Tensor<float, Position::UB> view_129 = q_rows[(q0 * 128)];  // #301
            const Tensor<float, Position::UB> view_130 = sums;  // #302
            const Tensor<float, Position::UB> view_131 = keys;  // #303
            const Tensor<float, Position::UB> view_132 = q_rows[(q0 * 128)];  // #304
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #305
            vmul(view_130.ptr(), view_131.ptr(), view_132.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)16, (uint8_t)16, (uint8_t)0);  // #305
            const Tensor<float, Position::UB> view_133 = sums[64];  // #306
            const Tensor<float, Position::UB> view_134 = keys[64];  // #307
            const Tensor<float, Position::UB> view_135 = q_rows[((q0 * 128) + 64)];  // #308
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #309
            vmul(view_133.ptr(), view_134.ptr(), view_135.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)16, (uint8_t)16, (uint8_t)0);  // #309
            const Tensor<float, Position::UB> view_136 = sums;  // #310
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #311
            vcadd(scores.ptr(), view_136.ptr(), (uint8_t)(64), 1, 1, 16, 0);  // #311
            const Tensor<float, Position::UB> view_137 = sums[64];  // #312
            vcadd(partial.ptr(), view_137.ptr(), (uint8_t)(64), 1, 1, 16, 0);  // #313
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #314
            vadd(scores.ptr(), scores.ptr(), partial.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #314
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #315
            vmuls(scores.ptr(), scores.ptr(), (float)(scale_log2), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #315
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #316
            vcmpvs_gt((__ubuf__ uint8_t*)(cmp.ptr()), scores.ptr(), (float)(th0), (uint8_t)(1), (uint16_t)0, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #316
            SetFlag<PIPE_V, PIPE_S>(6);  // #317
            WaitFlag<PIPE_V, PIPE_S>(6);  // #318
            const Tensor<uint64_t, Position::UB> point_2 = cmp.as<uint64_t>();  // #319
            const uint64_t exact = point_2.load(0);  // #320
            const bool cmp_3 = q1 != q0;  // #321
            uint64_t exact_2_state = 0ULL;  // #443
            if (cmp_3) {  // #349
                const float th1 = query_mean.load(q1);  // #323
                const Tensor<float, Position::UB> view_138 = q_rows[(q1 * 128)];  // #326
                const Tensor<float, Position::UB> view_139 = sums;  // #327
                const Tensor<float, Position::UB> view_140 = keys;  // #328
                const Tensor<float, Position::UB> view_141 = q_rows[(q1 * 128)];  // #329
                vmul(view_139.ptr(), view_140.ptr(), view_141.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)16, (uint8_t)16, (uint8_t)0);  // #330
                const Tensor<float, Position::UB> view_142 = sums[64];  // #331
                const Tensor<float, Position::UB> view_143 = keys[64];  // #332
                const Tensor<float, Position::UB> view_144 = q_rows[((q1 * 128) + 64)];  // #333
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #334
                vmul(view_142.ptr(), view_143.ptr(), view_144.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)16, (uint8_t)16, (uint8_t)0);  // #334
                const Tensor<float, Position::UB> view_145 = sums;  // #335
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #336
                vcadd(scores.ptr(), view_145.ptr(), (uint8_t)(64), 1, 1, 16, 0);  // #336
                const Tensor<float, Position::UB> view_146 = sums[64];  // #337
                vcadd(partial.ptr(), view_146.ptr(), (uint8_t)(64), 1, 1, 16, 0);  // #338
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #339
                vadd(scores.ptr(), scores.ptr(), partial.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #339
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #340
                vmuls(scores.ptr(), scores.ptr(), (float)(scale_log2), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #340
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #341
                vcmpvs_gt((__ubuf__ uint8_t*)(cmp.ptr()), scores.ptr(), (float)(th1), (uint8_t)(1), (uint16_t)0, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #341
                SetFlag<PIPE_V, PIPE_S>(6);  // #342
                WaitFlag<PIPE_V, PIPE_S>(6);  // #343
                const Tensor<uint64_t, Position::UB> point_4 = cmp.as<uint64_t>();  // #344
                exact_2_state = exact | point_4.load(0);  // #345 #346 #444
            } else {
                exact_2_state = exact;  // #445
            }
            const uint64_t exact_2 = (uint64_t)(exact_2_state);  // #446
            const int32_t lo = Max(q0 - 1, 0);  // #350 #353
            const int32_t hi = Min(q1 + 2, 64);  // #351 #355
            const bool cmp_4 = lo < hi;  // #357
            uint64_t mask_2_state = 0ULL;  // #447
            if (cmp_4) {  // #369
                mask_2_state = (18446744073709551615ULL >> (64 - hi)) & ~((1ULL << lo) - 1ULL);  // #359 #360 #362 #364 #365 #366 #448
            } else {
                mask_2_state = 0ULL;  // #449
            }
            const uint64_t mask_2 = (uint64_t)(mask_2_state);  // #450
            const uint64_t or_ = exact_2 | mask_2;  // #370
            const int32_t lo_1 = Max(sink_start, 0);  // #372
            const int32_t hi_1 = Min(sink_end, 64);  // #374
            const bool cmp_5 = lo_1 < hi_1;  // #376
            uint64_t mask_5_state = 0ULL;  // #451
            if (cmp_5) {  // #388
                mask_5_state = (18446744073709551615ULL >> (64 - hi_1)) & ~((1ULL << lo_1) - 1ULL);  // #378 #379 #381 #383 #384 #385 #452
            } else {
                mask_5_state = 0ULL;  // #453
            }
            const uint64_t mask_5 = (uint64_t)(mask_5_state);  // #454
            const uint64_t exact_3 = (or_ | mask_5) & valid;  // #389 #390
            route_masks.store(qi, (uint64_t)(valid & ~(exact_3)));  // #391 #392 #394
            route_pattern.store(0, (uint64_t)(exact_3));  // #396
            route_pattern.store(1, (uint64_t)(0ULL));  // #399
            scalar_publish.set();  // #400
            scalar_publish.wait();  // #401
            SetMaskNorm();  // #402
            ResetMask();  // #402
            vreducev2((__ubuf__ uint16_t*)(compact_indices.ptr()), (__ubuf__ uint16_t*)(source_indices.ptr()), (__ubuf__ uint16_t*)(route_pattern.ptr()), (uint8_t)(1), 1, 0, 8, 1);  // #402
            SetFlag<PIPE_V, PIPE_S>(0);  // #402
            WaitFlag<PIPE_V, PIPE_S>(0);  // #402
            const int32_t num_exact = (int32_t)get_rsvd_cnt();  // #402
            counts.store(qi, (int32_t)(num_exact));  // #404
            if (num_exact > 0) {  // #405 #412
                SetFlag<PIPE_V, PIPE_MTE3>(7);  // #406
                WaitFlag<PIPE_V, PIPE_MTE3>(7);  // #407
                const GMTensor<uint16_t> view_147 = exact_indices[(((b * ((H * NT_Q) * 64)) + (h * (NT_Q * 64))) + (qi * 64))];  // #408
                ub_to_gm_pad(view_147, compact_indices, 1, num_exact * 2, 0, 0);  // #462 #409
                SetFlag<PIPE_MTE3, PIPE_V>(7);  // #410
                WaitFlag<PIPE_MTE3, PIPE_V>(7);  // #411
            }
        }
        SetFlag<PIPE_S, PIPE_MTE3>(0);  // #415
        WaitFlag<PIPE_S, PIPE_MTE3>(0);  // #416
        const GMTensor<uint64_t> view_148 = approx_mask[((b * (H * NT_Q)) + (h * NT_Q))];  // #417
        ub_to_gm_pad(view_148, route_masks, 1, NT_Q * 8, 0, 0);  // #463 #418
        const GMTensor<int32_t> view_149 = exact_counts[((b * (H * NT_Q_PAD8)) + (h * NT_Q_PAD8))];  // #419
        ub_to_gm_pad(view_149, counts, 1, NT_Q_PAD8 * 4, 0, 0);  // #464 #420
        SetFlag<PIPE_MTE3, PIPE_S>(0);  // #421
        WaitFlag<PIPE_MTE3, PIPE_S>(0);  // #422
        ev_mte3_v_valid_d0_1.set();  // #498
    }
    const int32_t vec_id_1 = GetVecIdx();  // #425
    if (vec_id_1 == 0) {  // #426 #439
        for (int32_t col = 0; col < 64; col += 1) {  // #457
            const int32_t length = Min(64, Max(0, T - (col * 64)));  // #427 #428 #429 #430
            lens.store(col, (float)((float)(length)));  // #431 #433
        }
        SetFlag<PIPE_S, PIPE_MTE3>(0);  // #436
        WaitFlag<PIPE_S, PIPE_MTE3>(0);  // #437
        ub_to_gm_pad(last_group_lens, lens, 1, 256, 0, 0);  // #438
    }
    return;  // #440
}
