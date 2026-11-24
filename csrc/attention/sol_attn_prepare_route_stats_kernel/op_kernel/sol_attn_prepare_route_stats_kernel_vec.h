#pragma once
using namespace ascrip;

__aicore__ inline void sol_attn_prepare_route_stats_kernel_vec(GM_ADDR q_bar_, GM_ADDR kc_, GM_ADDR threshold_, GM_ADDR group_approx_mask_, GM_ADDR flat_exact_cnt_, GM_ADDR flat_exact_idx_, int32_t B, int32_t H, int32_t NT, int32_t NT_Q, int32_t NG, int32_t NPAD, int32_t NT_Q_PAD8, int32_t query_blocks_per_tile, int32_t sink_start, int32_t sink_end, float scale_log2, float scale_sq, float tau, GM_ADDR workspace)
{
    GMTensor<float> q_bar((__gm__ float*)q_bar_);
    GMTensor<bfloat16_t> kc((__gm__ bfloat16_t*)kc_);
    GMTensor<float> threshold((__gm__ float*)threshold_);
    GMTensor<uint64_t> group_approx_mask((__gm__ uint64_t*)group_approx_mask_);
    GMTensor<int32_t> flat_exact_cnt((__gm__ int32_t*)flat_exact_cnt_);
    GMTensor<uint16_t> flat_exact_idx((__gm__ uint16_t*)flat_exact_idx_);
    SetAtomicNone();
    ResetMask();
    SetMaskNorm();
    SEvent<PIPE_MTE2, PIPE_V, 0, 0> ev_mte2_v_ready_d0_1;  // guards k_in, q_in, scores0, scores1 #529
    SEvent<PIPE_V, PIPE_MTE2, 1, 0> ev_v_mte2_valid_d0_1;  // guards k_in, q_in, scores0, scores1 #530
    SEvent<PIPE_MTE2, PIPE_V, 0, 1> ev_mte2_v_ready_d1_1;  // guards k_in, q_in, scores0, scores1 #531
    SEvent<PIPE_V, PIPE_MTE2, 1, 1> ev_v_mte2_valid_d1_1;  // guards k_in, q_in, scores0, scores1 #532
    SEvent<PIPE_V, PIPE_MTE3, 0, 0> ev_v_mte3_ready_1;  // guards indices, query_mean, stat_tile, threshold_rows #555
    SEvent<PIPE_MTE3, PIPE_V, 1, 0> ev_mte3_v_valid_1;  // guards indices, query_mean, stat_tile, threshold_rows #556
    SEvent<PIPE_V, PIPE_S, 0, 1> ev_v_s_ready_1;  // guards cmp0, cmp1 #573
    const int32_t mul_1 = (B * H) * NPAD;  // #580 #581
    const int32_t mul_2 = mul_1 * 128;  // #582
    const int32_t mul_31 = mul_2 * 4;  // #583
    const GMTensor<float> key_f32((__gm__ float*)(workspace + (0)));  // #584
    const GMTensor<float> thresh_bh((__gm__ float*)(workspace + (mul_31)));  // #589
    const GMTensor<float> score_ws((__gm__ float*)(workspace + (mul_31 + (mul_1 * 4))));  // #587 #588 #591
    VEC_READY<PIPE_MTE2>(0);  // #597
    VEC_READY<PIPE_MTE2>(0);  // #597
    const Tensor<bfloat16_t, Position::UB> k_in((uint64_t)(0));  // #598
    const Tensor<float, Position::UB> q_in((uint64_t)(8192));  // #599
    const Tensor<float, Position::UB> sum_k((uint64_t)(24576));  // #600
    const Tensor<float, Position::UB> sum_k_sq((uint64_t)(40960));  // #601
    const Tensor<float, Position::UB> stat_tile((uint64_t)(57344));  // #602
    const Tensor<float, Position::UB> var_k((uint64_t)(73728));  // #603
    const Tensor<float, Position::UB> query_mean((uint64_t)(74240));  // #604
    const Tensor<float, Position::UB> query_var((uint64_t)(74368));  // #605
    const Tensor<float, Position::UB> partial((uint64_t)(74496));  // #606
    const Tensor<float, Position::UB> threshold_rows((uint64_t)(74624));  // #607
    const Tensor<float, Position::UB> scores0((uint64_t)(75648));  // #608
    const Tensor<float, Position::UB> scores1((uint64_t)(76672));  // #609
    const Tensor<uint8_t, Position::UB> cmp0((uint64_t)(77696));  // #610
    const Tensor<uint8_t, Position::UB> cmp1((uint64_t)(77728));  // #611
    const Tensor<uint64_t, Position::UB> bits0 = cmp0.as<uint64_t>();  // #612
    const Tensor<uint64_t, Position::UB> bits1 = cmp1.as<uint64_t>();  // #613
    const Tensor<uint64_t, Position::UB> masks((uint64_t)(77760));  // #614
    const Tensor<uint16_t, Position::UB> indices((uint64_t)(77792));  // #615
    const Tensor<int32_t, Position::UB> counts((uint64_t)(78304));  // #616
    const Tensor<int16_t, Position::UB> dense_ids((uint64_t)(78368));  // #617
    const Tensor<int16_t, Position::UB> offset_ids((uint64_t)(78880));  // #618
    const Tensor<uint64_t, Position::UB> exact_bits((uint64_t)(79392));  // #619
    SEvent<PIPE_S, PIPE_V, 0, 0> mask_ready;  // #41
    for (int32_t lane = 0; lane < 128; lane += 1) {  // #457
        dense_ids.store(lane, (int16_t)((int16_t)(lane)));  // #43 #45
    }
    mask_ready.set();  // #48
    mask_ready.wait();  // #49
    const Tensor<int16_t, Position::UB> view = dense_ids[128];  // #50
    const Tensor<int16_t, Position::UB> view_1 = dense_ids;  // #51
    vadds(view.ptr(), view_1.ptr(), (int16_t)((int16_t)(128)), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #52
    PipeBarrier<PIPE_V>();  // #53
    const int32_t vec_id = GetVecIdx();  // #55
    const int32_t mul_5 = B * H;  // #56
    const int32_t _launch_cubes = GetCubeNum();  // #456
    const int32_t vec_count = _launch_cubes * 2;  // #57
    for (int32_t bh = vec_id; bh < mul_5; bh += vec_count) {  // #460
        const int32_t idiv_33 = bh / H;  // #726
        const int32_t idiv_34 = bh % H;  // #727
        const bool idiv_35 = idiv_34 != 0;  // #728
        const bool idiv_36 = bh < 0;  // #729
        const bool idiv_38 = idiv_36 != false;  // #731
        const bool idiv_39 = idiv_35 && idiv_38;  // #732
        const int32_t idiv_40 = (idiv_39) ? (1) : (0);  // #733
        const int32_t b = idiv_33 - idiv_40;  // #58
        const int32_t h = ascrip::FloorMod<int32_t>(bh, H);  // #59
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #60
        vector_dup(sum_k.ptr(), (float)(0.0f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)8, 0);  // #60
        vector_dup(sum_k_sq.ptr(), (float)(0.0f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)8, 0);  // #61
        for (int32_t col = 0; col < NPAD; col += 32) {  // #458
            const GMTensor<bfloat16_t> view_2 = kc[(((b * ((H * NPAD) * 128)) + (h * (NPAD * 128))) + (col * 128))];  // #64
            ev_v_mte2_valid_d0_1.wait();  // #533
            gm_to_ub_pad(k_in, view_2, 32, 256, 0, 0);  // #65
            ev_mte2_v_ready_d0_1.set();  // #534
            ev_mte2_v_ready_d0_1.wait();  // #535
            ev_mte3_v_valid_1.wait();  // #557
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #66
            vconv_bf162f32(stat_tile.ptr(), k_in.ptr(), (uint8_t)(64), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)4);  // #66
            const GMTensor<float> view_3 = key_f32[(((b * ((H * NPAD) * 128)) + (h * (NPAD * 128))) + (col * 128))];  // #69
            ev_v_mte3_ready_1.set();  // #558
            ev_v_mte3_ready_1.wait();  // #559
            ub_to_gm_pad(view_3, stat_tile, 32, 512, 0, 0);  // #70
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #71
            vadd(sum_k.ptr(), sum_k.ptr(), stat_tile.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #71
            ev_mte3_v_valid_1.set();  // #560
            ev_mte3_v_valid_1.wait();  // #561
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #72
            vmul(stat_tile.ptr(), stat_tile.ptr(), stat_tile.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #72
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #73
            vadd(sum_k_sq.ptr(), sum_k_sq.ptr(), stat_tile.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #73
            ev_v_mte2_valid_d0_1.set();  // #550
            ev_v_mte3_ready_1.set();  // #568
            ev_v_mte3_ready_1.wait();  // #569
            ev_mte3_v_valid_1.set();  // #570
        }
        const Tensor<float, Position::UB> view_4 = sum_k;  // #76
        const Tensor<float, Position::UB> view_5 = sum_k;  // #77
        const Tensor<float, Position::UB> view_6 = sum_k[2048];  // #78
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #79
        vadd(view_4.ptr(), view_5.ptr(), view_6.ptr(), (uint8_t)(32), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #79
        const Tensor<float, Position::UB> view_7 = sum_k;  // #80
        const Tensor<float, Position::UB> view_8 = sum_k;  // #81
        const Tensor<float, Position::UB> view_9 = sum_k[1024];  // #82
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #83
        vadd(view_7.ptr(), view_8.ptr(), view_9.ptr(), (uint8_t)(16), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #83
        const Tensor<float, Position::UB> view_10 = sum_k;  // #84
        const Tensor<float, Position::UB> view_11 = sum_k;  // #85
        const Tensor<float, Position::UB> view_12 = sum_k[512];  // #86
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #87
        vadd(view_10.ptr(), view_11.ptr(), view_12.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #87
        const Tensor<float, Position::UB> view_13 = sum_k;  // #88
        const Tensor<float, Position::UB> view_14 = sum_k;  // #89
        const Tensor<float, Position::UB> view_15 = sum_k[256];  // #90
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #91
        vadd(view_13.ptr(), view_14.ptr(), view_15.ptr(), (uint8_t)(4), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #91
        const Tensor<float, Position::UB> view_16 = sum_k;  // #92
        const Tensor<float, Position::UB> view_17 = sum_k;  // #93
        const Tensor<float, Position::UB> view_18 = sum_k[128];  // #94
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #95
        vadd(view_16.ptr(), view_17.ptr(), view_18.ptr(), (uint8_t)(2), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #95
        const Tensor<float, Position::UB> view_19 = sum_k_sq;  // #96
        const Tensor<float, Position::UB> view_20 = sum_k_sq;  // #97
        const Tensor<float, Position::UB> view_21 = sum_k_sq[2048];  // #98
        vadd(view_19.ptr(), view_20.ptr(), view_21.ptr(), (uint8_t)(32), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #99
        const Tensor<float, Position::UB> view_22 = sum_k_sq;  // #100
        const Tensor<float, Position::UB> view_23 = sum_k_sq;  // #101
        const Tensor<float, Position::UB> view_24 = sum_k_sq[1024];  // #102
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #103
        vadd(view_22.ptr(), view_23.ptr(), view_24.ptr(), (uint8_t)(16), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #103
        const Tensor<float, Position::UB> view_25 = sum_k_sq;  // #104
        const Tensor<float, Position::UB> view_26 = sum_k_sq;  // #105
        const Tensor<float, Position::UB> view_27 = sum_k_sq[512];  // #106
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #107
        vadd(view_25.ptr(), view_26.ptr(), view_27.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #107
        const Tensor<float, Position::UB> view_28 = sum_k_sq;  // #108
        const Tensor<float, Position::UB> view_29 = sum_k_sq;  // #109
        const Tensor<float, Position::UB> view_30 = sum_k_sq[256];  // #110
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #111
        vadd(view_28.ptr(), view_29.ptr(), view_30.ptr(), (uint8_t)(4), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #111
        const Tensor<float, Position::UB> view_31 = sum_k_sq;  // #112
        const Tensor<float, Position::UB> view_32 = sum_k_sq;  // #113
        const Tensor<float, Position::UB> view_33 = sum_k_sq[128];  // #114
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #115
        vadd(view_31.ptr(), view_32.ptr(), view_33.ptr(), (uint8_t)(2), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #115
        const float inv_nt = 1.0f / (float)(NT);  // #116 #117
        const Tensor<float, Position::UB> mean_k = sum_k;  // #118
        const Tensor<float, Position::UB> moment_k = sum_k_sq;  // #119
        vmuls(mean_k.ptr(), mean_k.ptr(), (float)(inv_nt), (uint8_t)(2), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #120
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #121
        vmuls(moment_k.ptr(), moment_k.ptr(), (float)(inv_nt), (uint8_t)(2), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #121
        vmul(var_k.ptr(), mean_k.ptr(), mean_k.ptr(), (uint8_t)(2), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #122
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #123
        vsub(var_k.ptr(), moment_k.ptr(), var_k.ptr(), (uint8_t)(2), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #123
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #124
        vmaxs(var_k.ptr(), var_k.ptr(), (float)(0.0f), (uint8_t)(2), (uint16_t)1, (uint16_t)1, (uint8_t)8, (uint8_t)8, false, false);  // #124
        for (int32_t row = 0; row < NT; row += 32) {  // #459
            const int32_t n = Min(32, NT - row);  // #125 #126
            ev_v_mte2_valid_d0_1.wait();  // #536
            if (n < 32) {  // #127 #129
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #128
                vector_dup(q_in.ptr(), (float)(0.0f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)8, 0);  // #128
            }
            const Tensor<float, Position::UB> view_34 = q_in;  // #130
            const GMTensor<float> view_35 = q_bar[(((b * ((H * NT) * 128)) + (h * (NT * 128))) + (row * 128))];  // #133
            ev_v_mte2_valid_d0_1.set();  // #537
            ev_v_mte2_valid_d0_1.wait();  // #538
            gm_to_ub_pad(view_34, view_35, n, 512, 0, 0);  // #134
            const Tensor<float, Position::UB> view_36 = stat_tile;  // #135
            const Tensor<float, Position::UB> view_37 = q_in;  // #136
            const Tensor<float, Position::UB> view_38 = sum_k;  // #137
            ev_mte2_v_ready_d0_1.set();  // #539
            ev_mte2_v_ready_d0_1.wait();  // #540
            ev_mte3_v_valid_1.wait();  // #562
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #138
            vmul(view_36.ptr(), view_37.ptr(), view_38.ptr(), (uint8_t)(32), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)16, (uint8_t)16, (uint8_t)0);  // #138
            const Tensor<float, Position::UB> view_39 = stat_tile[64];  // #139
            const Tensor<float, Position::UB> view_40 = q_in[64];  // #140
            const Tensor<float, Position::UB> view_41 = sum_k[64];  // #141
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #142
            vmul(view_39.ptr(), view_40.ptr(), view_41.ptr(), (uint8_t)(32), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)16, (uint8_t)16, (uint8_t)0);  // #142
            const Tensor<float, Position::UB> view_42 = stat_tile;  // #143
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #144
            vcadd(query_mean.ptr(), view_42.ptr(), (uint8_t)(32), 1, 1, 16, 0);  // #144
            const Tensor<float, Position::UB> view_43 = stat_tile[64];  // #145
            vcadd(partial.ptr(), view_43.ptr(), (uint8_t)(32), 1, 1, 16, 0);  // #146
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #147
            SetMaskCount();  // #147
            SetVectorMask(32);  // #147
            vadd(query_mean.ptr(), query_mean.ptr(), partial.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #147
            SetMaskNorm();  // #147
            ResetMask();  // #147
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #148
            SetMaskCount();  // #148
            SetVectorMask(32);  // #148
            vmuls(query_mean.ptr(), query_mean.ptr(), (float)(scale_log2), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #148
            SetMaskNorm();  // #148
            ResetMask();  // #148
            vmul(stat_tile.ptr(), q_in.ptr(), q_in.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #149
            const Tensor<float, Position::UB> view_44 = stat_tile;  // #150
            const Tensor<float, Position::UB> view_45 = stat_tile;  // #151
            const Tensor<float, Position::UB> view_46 = var_k;  // #152
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #153
            vmul(view_44.ptr(), view_45.ptr(), view_46.ptr(), (uint8_t)(32), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)16, (uint8_t)16, (uint8_t)0);  // #153
            const Tensor<float, Position::UB> view_47 = stat_tile[64];  // #154
            const Tensor<float, Position::UB> view_48 = stat_tile[64];  // #155
            const Tensor<float, Position::UB> view_49 = var_k[64];  // #156
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #157
            vmul(view_47.ptr(), view_48.ptr(), view_49.ptr(), (uint8_t)(32), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)16, (uint8_t)16, (uint8_t)0);  // #157
            const Tensor<float, Position::UB> view_50 = stat_tile;  // #158
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #159
            vcadd(query_var.ptr(), view_50.ptr(), (uint8_t)(32), 1, 1, 16, 0);  // #159
            const Tensor<float, Position::UB> view_51 = stat_tile[64];  // #160
            vcadd(partial.ptr(), view_51.ptr(), (uint8_t)(32), 1, 1, 16, 0);  // #161
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #162
            SetMaskCount();  // #162
            SetVectorMask(32);  // #162
            vadd(query_var.ptr(), query_var.ptr(), partial.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #162
            SetMaskNorm();  // #162
            ResetMask();  // #162
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #163
            SetMaskCount();  // #163
            SetVectorMask(32);  // #163
            vmuls(query_var.ptr(), query_var.ptr(), (float)(scale_sq), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #163
            SetMaskNorm();  // #163
            ResetMask();  // #163
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #164
            SetMaskCount();  // #164
            SetVectorMask(32);  // #164
            vmaxs(query_var.ptr(), query_var.ptr(), (float)(0.0f), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint8_t)8, (uint8_t)8, false, false);  // #164
            SetMaskNorm();  // #164
            ResetMask();  // #164
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #165
            SetMaskCount();  // #165
            SetVectorMask(32);  // #165
            vadds(query_var.ptr(), query_var.ptr(), (float)(1e-06f), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #165
            SetMaskNorm();  // #165
            ResetMask();  // #165
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #166
            SetMaskCount();  // #166
            SetVectorMask(32);  // #166
            vsqrt(query_var.ptr(), query_var.ptr(), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint8_t)8, (uint8_t)8);  // #166
            SetMaskNorm();  // #166
            ResetMask();  // #166
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #167
            SetMaskCount();  // #167
            SetVectorMask(32);  // #167
            vmuls(query_var.ptr(), query_var.ptr(), (float)(tau), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #167
            SetMaskNorm();  // #167
            ResetMask();  // #167
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #168
            SetMaskCount();  // #168
            SetVectorMask(32);  // #168
            vadd(query_mean.ptr(), query_mean.ptr(), query_var.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #168
            SetMaskNorm();  // #168
            ResetMask();  // #168
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #169
            ResetMask();  // #169
            vbrcb((__ubuf__ uint32_t*)(threshold_rows.ptr()), (__ubuf__ uint32_t*)(query_mean.ptr()), (uint16_t)1, (uint16_t)8, (uint8_t)(4));  // #169
            const GMTensor<float> view_52 = thresh_bh[(((b * (H * NPAD)) + (h * NPAD)) + row)];  // #172
            ev_v_mte3_ready_1.set();  // #563
            ev_v_mte3_ready_1.wait();  // #564
            ub_to_gm_pad(view_52, query_mean, 1, n * 4, 0, 0);  // #504 #173
            const GMTensor<float> view_53 = threshold[(((b * (NT * H)) + (row * H)) + h)];  // #176
            const int32_t sub_6 = H + -1;  // #177
            ub_to_gm_pad(view_53, threshold_rows, n, 4, 0, sub_6 * 4);  // #505 #178
            ev_v_mte2_valid_d0_1.set();  // #551
            ev_mte3_v_valid_1.set();  // #571
        }
    }
    PipeBarrier<PIPE_ALL>();  // #184
    ALLVEC_READY<PIPE_MTE3>(6);  // #185
    ALLVEC_WAIT<PIPE_S>(6);  // #186
    DataCacheCleanAndInvalid(q_bar, (uint64_t)AscendC::CacheLine::ENTIRE_DATA_CACHE, (uint64_t)AscendC::DcciDst::CACHELINE_OUT);  // #187
    VEC_READY<PIPE_MTE3>(7);  // #188
    const int32_t mul_6 = 2 * query_blocks_per_tile;  // #620
    const int32_t idiv_41 = 32 / mul_6;  // #734
    const int32_t idiv_42 = 32 % mul_6;  // #735
    const bool idiv_43 = idiv_42 != 0;  // #736
    const bool idiv_45 = mul_6 < 0;  // #738
    const bool idiv_46 = false != idiv_45;  // #739
    const bool idiv_47 = idiv_43 && idiv_46;  // #740
    const int32_t idiv_48 = (idiv_47) ? (1) : (0);  // #741
    const int32_t subblock_queries = idiv_41 - idiv_48;  // #621
    const int32_t mul_7 = 2 * subblock_queries;  // #622
    const int32_t idiv_49 = NT_Q_PAD8 / mul_7;  // #742
    const int32_t idiv_50 = NT_Q_PAD8 % mul_7;  // #743
    const bool idiv_51 = idiv_50 != 0;  // #744
    const bool idiv_53 = mul_7 < 0;  // #746
    const bool idiv_54 = false == idiv_53;  // #747
    const bool idiv_55 = idiv_51 && idiv_54;  // #748
    const int32_t idiv_56 = (idiv_55) ? (1) : (0);  // #749
    const int32_t q_chunks = idiv_49 + idiv_56;  // #623
    const int32_t subblock = GetSubBlockIdx();  // #196
    const int32_t group_id = GetCubeIdx();  // #627
    const int32_t mul_9 = (B * H) * q_chunks;  // #628 #629
    const int32_t cube_count = GetCubeNum();  // #630
    const int32_t cast_2 = (int32_t)(cube_count);  // #631
    int32_t chunk_step_3_state = 0;  // #632
    int32_t k_step_5_state = 0;  // #633
    for (int32_t task = group_id; task < mul_9; task += cast_2) {  // #692
        const int32_t chunk_step_carry = (int32_t)(chunk_step_3_state);  // #634
        const int32_t k_step_carry = (int32_t)(k_step_5_state);  // #635
        const int32_t idiv_57 = task / q_chunks;  // #750
        const int32_t idiv_58 = task % q_chunks;  // #751
        const bool idiv_59 = idiv_58 != 0;  // #752
        const bool idiv_60 = task < 0;  // #753
        const bool idiv_61 = q_chunks < 0;  // #754
        const bool idiv_62 = idiv_60 != idiv_61;  // #755
        const bool idiv_63 = idiv_59 && idiv_62;  // #756
        const int32_t idiv_64 = (idiv_63) ? (1) : (0);  // #757
        const int32_t bh_1 = idiv_57 - idiv_64;  // #636
        const int32_t idiv_65 = bh_1 / H;  // #758
        const int32_t idiv_66 = bh_1 % H;  // #759
        const bool idiv_67 = idiv_66 != 0;  // #760
        const bool idiv_68 = bh_1 < 0;  // #761
        const bool idiv_70 = idiv_68 != false;  // #763
        const bool idiv_71 = idiv_67 && idiv_70;  // #764
        const int32_t idiv_72 = (idiv_71) ? (1) : (0);  // #765
        const int32_t b_1 = idiv_65 - idiv_72;  // #637
        const int32_t h_1 = ascrip::FloorMod<int32_t>(bh_1, H);  // #638
        const int32_t tile_base = ascrip::FloorMod<int32_t>(task, q_chunks) * (2 * subblock_queries);  // #639 #640 #641
        const int32_t q_base = tile_base * query_blocks_per_tile;  // #642
        const int32_t store_base = tile_base + (subblock * subblock_queries);  // #213 #214
        const int32_t valid_tiles = Max(0, Min(subblock_queries, NT_Q - store_base));  // #215 #216 #217
        for (int32_t qi = 0; qi < subblock_queries; qi += 1) {  // #465
            counts.store(qi, (int32_t)(0));  // #240
        }
        int32_t chunk_step_2_state = chunk_step_carry;  // #645
        int32_t k_step_4_state = k_step_carry;  // #646
        for (int32_t first_group = 0; first_group < NG; first_group += 4) {  // #687
            const int32_t chunk_step_carry_1 = (int32_t)(chunk_step_2_state);  // #647
            const int32_t k_step_carry_1 = (int32_t)(k_step_4_state);  // #648
            const int32_t groups = Min(4, NG - first_group);  // #649 #650
            const int32_t start = first_group * 64;  // #651
            const int32_t end = start + (groups * 64);  // #652 #653
            int32_t k_step_3_state = 0;  // #654
            int32_t k_step_2_state = k_step_carry_1;  // #655
            for (int32_t key = start; key < end; key += 128) {  // #679
                const int32_t k_step_carry_2 = (int32_t)(k_step_2_state);  // #656
                k_step_2_state = (int32_t)(k_step_carry_2 + 1);  // #676 #677 #678
            }
            k_step_3_state = (int32_t)(k_step_2_state);  // #680 #681
            const int32_t k_step_3 = (int32_t)(k_step_3_state);  // #682
            WAIT_CUBE<PIPE_S>(0);  // #279
            for (int32_t qi_1 = 0; qi_1 < valid_tiles; qi_1 += 1) {  // #495
                const int32_t q_tile = store_base + qi_1;  // #281
                const int32_t q0 = q_tile * query_blocks_per_tile;  // #282
                const int32_t sub_15 = NT + -1;  // #285
                const int32_t q1 = Min((q0 + query_blocks_per_tile) - 1, sub_15);  // #283 #284 #286
                const int32_t local_q0 = q0 - q_base;  // #287
                const int32_t local_q1 = q1 - q_base;  // #288
                const float threshold0 = thresh_bh.load(((b_1 * (H * NPAD)) + (h_1 * NPAD)) + q0);  // #290 #291 #292 #293 #294 #295
                const float threshold1 = thresh_bh.load(((b_1 * (H * NPAD)) + (h_1 * NPAD)) + q1);  // #297 #298 #299 #300 #301 #302
                const int32_t exact_count = counts.load(qi_1);  // #304
                const int32_t group_id_2 = GetCubeIdx();  // #305
                const int32_t mod_2 = chunk_step_carry_1 & 1;  // #306
                const GMTensor<float> view_63 = score_ws[(((group_id_2 * 16384) + (mod_2 * 8192)) + (local_q0 * 256))];  // #308 #307
                ev_v_mte2_valid_d0_1.wait();  // #541
                gm_to_ub_pad(scores0, view_63, 1, (groups * 64) * 4, 0, 0);  // #309 #506 #310
                ev_mte2_v_ready_d0_1.set();  // #542
                ev_mte2_v_ready_d0_1.wait();  // #543
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #311
                vmuls(scores0.ptr(), scores0.ptr(), (float)(scale_log2), (uint8_t)(groups), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #311
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #312
                vcmpvs_gt((__ubuf__ uint8_t*)(cmp0.ptr()), scores0.ptr(), (float)(threshold0), (uint8_t)(groups), (uint16_t)0, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #312
                ev_v_mte2_valid_d0_1.set();  // #544
                ev_v_mte2_valid_d0_1.wait();  // #545
                if (q0 != q1) {  // #313 #322
                    const int32_t group_id_3 = GetCubeIdx();  // #314
                    const int32_t mod_3 = chunk_step_carry_1 & 1;  // #315
                    const GMTensor<float> view_64 = score_ws[(((group_id_3 * 16384) + (mod_3 * 8192)) + (local_q1 * 256))];  // #317 #316
                    ev_v_mte2_valid_d1_1.wait();  // #546
                    gm_to_ub_pad(scores1, view_64, 1, (groups * 64) * 4, 0, 0);  // #318 #507 #319
                    ev_mte2_v_ready_d1_1.set();  // #547
                    ev_mte2_v_ready_d1_1.wait();  // #548
                    vmuls(scores1.ptr(), scores1.ptr(), (float)(scale_log2), (uint8_t)(groups), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #320
                    PipeBarrier<PIPE_V>();  // c220: V-V hazard #321
                    vcmpvs_gt((__ubuf__ uint8_t*)(cmp1.ptr()), scores1.ptr(), (float)(threshold1), (uint8_t)(groups), (uint16_t)0, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #321
                    ev_v_mte2_valid_d1_1.set();  // #549
                }
                ev_mte2_v_ready_d0_1.set();  // #552
                ev_mte2_v_ready_d0_1.wait();  // #553
                ev_v_s_ready_1.set();  // #574
                ev_v_s_ready_1.wait();  // #575
                for (int32_t gg = 0; gg < groups; gg += 1) {  // #490
                    const int32_t group_start = start + (gg * 64);  // #323 #324
                    const int32_t valid = Min(64, NT - group_start);  // #325 #326
                    const uint64_t valid_bits = 18446744073709551615ULL >> (64 - valid);  // #327 #328
                    const Tensor<uint64_t, Position::UB> point_5 = cmp0.as<uint64_t>()[gg];  // #329
                    const uint64_t exact = point_5.load(0);  // #330
                    const bool cmp_4 = q0 != q1;  // #331
                    uint64_t exact_2_state = 0ULL;  // #478
                    if (cmp_4) {  // #337
                        const Tensor<uint64_t, Position::UB> point_6 = cmp1.as<uint64_t>()[gg];  // #332
                        exact_2_state = exact | point_6.load(0);  // #333 #334 #479
                    } else {
                        exact_2_state = exact;  // #480
                    }
                    const uint64_t exact_2 = (uint64_t)(exact_2_state);  // #481
                    const int32_t lo = Max((q0 - 1) - group_start, 0);  // #338 #340 #341
                    const int32_t hi = Min((q1 + 2) - group_start, 64);  // #339 #342 #343
                    const bool cmp_5 = lo < hi;  // #345
                    uint64_t mask_2_state = 0ULL;  // #482
                    if (cmp_5) {  // #357
                        mask_2_state = (18446744073709551615ULL >> (64 - hi)) & ~((1ULL << lo) - 1ULL);  // #347 #348 #350 #352 #353 #354 #483
                    } else {
                        mask_2_state = 0ULL;  // #484
                    }
                    const uint64_t mask_2 = (uint64_t)(mask_2_state);  // #485
                    const uint64_t exact_3 = exact_2 | mask_2;  // #358
                    const int32_t lo_1 = Max(sink_start - group_start, 0);  // #359 #360
                    const int32_t hi_1 = Min(sink_end - group_start, 64);  // #361 #362
                    const bool cmp_6 = lo_1 < hi_1;  // #364
                    uint64_t mask_5_state = 0ULL;  // #486
                    if (cmp_6) {  // #376
                        mask_5_state = (18446744073709551615ULL >> (64 - hi_1)) & ~((1ULL << lo_1) - 1ULL);  // #366 #367 #369 #371 #372 #373 #487
                    } else {
                        mask_5_state = 0ULL;  // #488
                    }
                    const uint64_t mask_5 = (uint64_t)(mask_5_state);  // #489
                    const uint64_t exact_4 = exact_3 | mask_5;  // #377
                    const uint64_t exact_5 = exact_4 & valid_bits;  // #378
                    masks.store(gg, (uint64_t)(valid_bits & ~(exact_5)));  // #379 #380 #382
                    const int32_t mod_4 = gg & 1;  // #385
                    const int32_t pattern = ((gg / 2) * 4) + mod_4;  // #383 #384 #386
                    exact_bits.store(pattern, (uint64_t)(exact_5));  // #388
                }
                const int32_t mod_5 = groups & 1;  // #391
                if (mod_5 != 0) {  // #392 #399
                    const int32_t idiv_73 = groups / 2;  // #766
                    const int32_t idiv_74 = groups % 2;  // #767
                    const bool idiv_75 = idiv_74 != 0;  // #768
                    const bool idiv_76 = groups < 0;  // #769
                    const bool idiv_78 = idiv_76 != false;  // #771
                    const bool idiv_79 = idiv_75 && idiv_78;  // #772
                    const int32_t idiv_80 = (idiv_79) ? (1) : (0);  // #773
                    const int32_t div_1 = idiv_73 - idiv_80;  // #394
                    exact_bits.store((div_1 * 4) + 1, (uint64_t)(0ULL));  // #395 #396 #398
                }
                mask_ready.set();  // #400
                mask_ready.wait();  // #401
                const int32_t idiv_81 = groups / 2;  // #774
                const int32_t idiv_82 = groups % 2;  // #775
                const bool idiv_83 = idiv_82 != 0;  // #776
                const bool idiv_84 = groups < 0;  // #777
                const bool idiv_86 = idiv_84 == false;  // #779
                const bool idiv_87 = idiv_83 && idiv_86;  // #780
                const int32_t idiv_88 = (idiv_87) ? (1) : (0);  // #781
                const int32_t gather_repeats = idiv_81 + idiv_88;  // #402
                const bool cmp_8 = start == 0;  // #403
                int32_t chunk_count_2_state = 0;  // #491
                ev_mte3_v_valid_1.wait();  // #565
                if (cmp_8) {  // #412
                    const Tensor<uint16_t, Position::UB> view_65 = dense_ids.as<uint16_t>();  // #404
                    SetMaskNorm();  // #405
                    ResetMask();  // #405
                    vreducev2((__ubuf__ uint16_t*)(indices.ptr()), (__ubuf__ uint16_t*)(view_65.ptr()), (__ubuf__ uint16_t*)(exact_bits.ptr()), (uint8_t)(gather_repeats), 1, 0, 8, 1);  // #405
                    SetFlag<PIPE_V, PIPE_S>(0);  // #405
                    WaitFlag<PIPE_V, PIPE_S>(0);  // #405
                    const int32_t chunk_count = (int32_t)get_rsvd_cnt();  // #405
                    chunk_count_2_state = chunk_count;  // #492
                } else {
                    vadds(offset_ids.ptr(), dense_ids.ptr(), (int16_t)((int16_t)(start)), (uint8_t)(gather_repeats), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #406 #407
                    const Tensor<uint16_t, Position::UB> view_66 = offset_ids.as<uint16_t>();  // #408
                    PipeBarrier<PIPE_V>();  // c220: V-V hazard #409
                    SetMaskNorm();  // #409
                    ResetMask();  // #409
                    vreducev2((__ubuf__ uint16_t*)(indices.ptr()), (__ubuf__ uint16_t*)(view_66.ptr()), (__ubuf__ uint16_t*)(exact_bits.ptr()), (uint8_t)(gather_repeats), 1, 0, 8, 1);  // #409
                    SetFlag<PIPE_V, PIPE_S>(0);  // #409
                    WaitFlag<PIPE_V, PIPE_S>(0);  // #409
                    const int32_t chunk_count_1 = (int32_t)get_rsvd_cnt();  // #409
                    chunk_count_2_state = chunk_count_1;  // #493
                }
                const int32_t chunk_count_2 = (int32_t)(chunk_count_2_state);  // #494
                SetFlag<PIPE_S, PIPE_MTE3>(0);  // #413
                WaitFlag<PIPE_S, PIPE_MTE3>(0);  // #414
                const GMTensor<uint64_t> view_67 = group_approx_mask[((((b_1 * ((H * NT_Q) * NG)) + (h_1 * (NT_Q * NG))) + (q_tile * NG)) + first_group)];  // #417
                ub_to_gm_pad(view_67, masks, 1, groups * 8, 0, 0);  // #508 #418
                ev_v_mte3_ready_1.set();  // #566
                ev_v_mte3_ready_1.wait();  // #567
                if (chunk_count_2 > 0) {  // #419 #424
                    const GMTensor<uint16_t> view_68 = flat_exact_idx[((((b_1 * ((H * NT_Q) * NPAD)) + (h_1 * (NT_Q * NPAD))) + (q_tile * NPAD)) + exact_count)];  // #422
                    ub_to_gm_pad(view_68, indices, 1, chunk_count_2 * 2, 0, 0);  // #509 #423
                }
                SetFlag<PIPE_MTE3, PIPE_S>(0);  // #425
                WaitFlag<PIPE_MTE3, PIPE_S>(0);  // #426
                counts.store(qi_1, (int32_t)(exact_count + chunk_count_2));  // #427 #429
                ev_v_mte2_valid_d0_1.set();  // #554
                ev_mte3_v_valid_1.set();  // #572
            }
            VEC_READY<PIPE_MTE2>(0);  // #433
            chunk_step_2_state = (int32_t)(chunk_step_carry_1 + 1);  // #683 #684 #685
            k_step_4_state = k_step_3;  // #686
        }
        const int32_t chunk_step_2 = (int32_t)(chunk_step_2_state);  // #688
        const int32_t k_step_4 = (int32_t)(k_step_4_state);  // #689
        if (store_base < NT_Q_PAD8) {  // #441 #452
            SetFlag<PIPE_S, PIPE_MTE3>(0);  // #442
            WaitFlag<PIPE_S, PIPE_MTE3>(0);  // #443
            const int32_t rows = Min(subblock_queries, NT_Q_PAD8 - store_base);  // #444 #445
            const GMTensor<int32_t> view_69 = flat_exact_cnt[(((b_1 * (H * NT_Q_PAD8)) + (h_1 * NT_Q_PAD8)) + store_base)];  // #448
            ub_to_gm_pad(view_69, counts, 1, rows * 4, 0, 0);  // #510 #449
            SetFlag<PIPE_MTE3, PIPE_S>(0);  // #450
            WaitFlag<PIPE_MTE3, PIPE_S>(0);  // #451
        }
        chunk_step_3_state = chunk_step_2;  // #690
        k_step_5_state = k_step_4;  // #691
    }
    return;  // #693
}
