#pragma once
using namespace ascrip;

__aicore__ inline void sol_attn_prepare_route_pack_kernel_vec(GM_ADDR scores_, GM_ADDR threshold_, GM_ADDR group_approx_mask_, GM_ADDR flat_exact_cnt_, GM_ADDR flat_exact_idx_, int32_t HEADS, int32_t NT, int32_t NT_Q, int32_t NG, int32_t NPAD, int32_t NT_Q_PAD8, int32_t query_blocks_per_tile, int32_t sink_start, int32_t sink_end, float scale_log2, GM_ADDR workspace)
{
    GMTensor<float> scores((__gm__ float*)scores_);
    GMTensor<float> threshold((__gm__ float*)threshold_);
    GMTensor<uint64_t> group_approx_mask((__gm__ uint64_t*)group_approx_mask_);
    GMTensor<int32_t> flat_exact_cnt((__gm__ int32_t*)flat_exact_cnt_);
    GMTensor<uint16_t> flat_exact_idx((__gm__ uint16_t*)flat_exact_idx_);
    SetAtomicNone();
    ResetMask();
    SetMaskNorm();
    SEvent<PIPE_MTE2, PIPE_V, 0, 0> ev_mte2_v_ready_d0_1;  // guards scores0, scores1 #236
    SEvent<PIPE_V, PIPE_MTE2, 1, 0> ev_v_mte2_valid_d0_1;  // guards scores0, scores1 #237
    SEvent<PIPE_MTE2, PIPE_V, 0, 1> ev_mte2_v_ready_d1_1;  // guards scores0, scores1 #238
    SEvent<PIPE_V, PIPE_MTE2, 1, 1> ev_v_mte2_valid_d1_1;  // guards scores0, scores1 #239
    SEvent<PIPE_V, PIPE_MTE3, 0, 0> ev_v_mte3_ready_d0_1;  // guards counts, indices #252
    SEvent<PIPE_MTE3, PIPE_V, 1, 0> ev_mte3_v_valid_d0_1;  // guards counts, indices #253
    SEvent<PIPE_V, PIPE_MTE3, 0, 1> ev_v_mte3_ready_d1_1;  // guards counts, indices #254
    SEvent<PIPE_MTE3, PIPE_V, 1, 1> ev_mte3_v_valid_d1_1;  // guards counts, indices #255
    SEvent<PIPE_V, PIPE_S, 0, 1> ev_v_s_ready_d0_1;  // guards cmp0, cmp1, counts #264
    SEvent<PIPE_V, PIPE_S, 0, 2> ev_v_s_ready_d1_1;  // guards cmp0, cmp1, counts #265
    SEvent<PIPE_V, PIPE_S, 0, 3> ev_v_s_ready_d2_1;  // guards cmp0, cmp1, counts #266
    const Tensor<float, Position::UB> scores0((uint64_t)(0));  // #1
    const Tensor<float, Position::UB> scores1((uint64_t)(8192));  // #2
    const Tensor<uint8_t, Position::UB> cmp0((uint64_t)(16384));  // #3
    const Tensor<uint8_t, Position::UB> cmp1((uint64_t)(16640));  // #4
    const Tensor<uint64_t, Position::UB> bits0 = cmp0.as<uint64_t>();  // #5
    const Tensor<uint64_t, Position::UB> bits1 = cmp1.as<uint64_t>();  // #6
    const Tensor<uint64_t, Position::UB> masks((uint64_t)(16896));  // #7
    const Tensor<uint16_t, Position::UB> indices((uint64_t)(17152));  // #8
    const Tensor<int32_t, Position::UB> counts((uint64_t)(21248));  // #9
    const Tensor<int16_t, Position::UB> dense_ids((uint64_t)(21280));  // #10
    const Tensor<int16_t, Position::UB> offset_ids((uint64_t)(25376));  // #11
    const Tensor<uint64_t, Position::UB> exact_bits((uint64_t)(29472));  // #12
    SEvent<PIPE_S, PIPE_V, 0, 0> mask_ready;  // #13
    for (int32_t lane = 0; lane < 128; lane += 1) {  // #202
        dense_ids.store(lane, (int16_t)((int16_t)(lane)));  // #14 #16
    }
    mask_ready.set();  // #19
    mask_ready.wait();  // #20
    const Tensor<int16_t, Position::UB> view = dense_ids[128];  // #21
    const Tensor<int16_t, Position::UB> view_1 = dense_ids;  // #22
    vadds(view.ptr(), view_1.ptr(), (int16_t)((int16_t)(128)), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #23
    const Tensor<int16_t, Position::UB> view_2 = dense_ids[256];  // #24
    const Tensor<int16_t, Position::UB> view_3 = dense_ids;  // #25
    PipeBarrier<PIPE_V>();  // c220: V-V hazard #26
    vadds(view_2.ptr(), view_3.ptr(), (int16_t)((int16_t)(256)), (uint8_t)(2), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #26
    const Tensor<int16_t, Position::UB> view_4 = dense_ids[512];  // #27
    const Tensor<int16_t, Position::UB> view_5 = dense_ids;  // #28
    PipeBarrier<PIPE_V>();  // c220: V-V hazard #29
    vadds(view_4.ptr(), view_5.ptr(), (int16_t)((int16_t)(512)), (uint8_t)(4), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #29
    const Tensor<int16_t, Position::UB> view_6 = dense_ids[1024];  // #30
    const Tensor<int16_t, Position::UB> view_7 = dense_ids;  // #31
    PipeBarrier<PIPE_V>();  // c220: V-V hazard #32
    vadds(view_6.ptr(), view_7.ptr(), (int16_t)((int16_t)(1024)), (uint8_t)(8), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #32
    PipeBarrier<PIPE_V>();  // #33
    const int32_t q_chunks = NT_Q_PAD8 / 8;  // #34
    const int32_t vec_id = GetVecIdx();  // #36
    const int32_t mul = HEADS * q_chunks;  // #37
    const int32_t vec_count = GetVecNum();  // #38
    for (int32_t task = vec_id; task < mul; task += vec_count) {  // #226
        const int32_t idiv_1 = task / q_chunks;  // #273
        const int32_t idiv_2 = task % q_chunks;  // #274
        const bool idiv_3 = idiv_2 != 0;  // #275
        const bool idiv_4 = task < 0;  // #276
        const bool idiv_6 = idiv_4 != false;  // #278
        const bool idiv_7 = idiv_3 && idiv_6;  // #279
        const int32_t idiv_8 = (idiv_7) ? (1) : (0);  // #280
        const int32_t h = idiv_1 - idiv_8;  // #39
        const int32_t q_base = ascrip::FloorMod<int32_t>(task, q_chunks) * 8;  // #40 #41
        const int32_t valid_q = Min(8, NT_Q - q_base);  // #42 #43
        ev_mte3_v_valid_d0_1.wait();  // #256
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #44
        SetMaskCount();  // #44
        SetVectorMask(8);  // #44
        vector_dup(counts.ptr(), (int32_t)(0), (uint8_t)(1), (uint16_t)1, 1, (uint8_t)8, 0);  // #44
        SetMaskNorm();  // #44
        ResetMask();  // #44
        for (int32_t qi = 0; qi < valid_q; qi += 1) {  // #225
            const int32_t q_tile = q_base + qi;  // #45
            const int32_t q0 = q_tile * query_blocks_per_tile;  // #46
            const int32_t sub_2 = NT + -1;  // #49
            const int32_t q1 = Min((q0 + query_blocks_per_tile) - 1, sub_2);  // #47 #48 #50
            const float threshold0 = threshold.load((h * NT) + q0);  // #52 #53 #54
            const float threshold1 = threshold.load((h * NT) + q1);  // #56 #57 #58
            int32_t exact_count_2_state = 0;  // #203
            for (int32_t first_group = 0; first_group < NG; first_group += 32) {  // #223
                const int32_t exact_count_carry = (int32_t)(exact_count_2_state);  // #204
                const int32_t groups = Min(32, NG - first_group);  // #60 #61
                const int32_t start = first_group * 64;  // #62
                const int32_t mul_3 = groups * 64;  // #63
                const int32_t end = start + mul_3;  // #64
                const GMTensor<float> view_8 = scores[(((h * (NT * NPAD)) + (q0 * NPAD)) + start)];  // #66 #65
                ev_v_mte2_valid_d0_1.wait();  // #240
                gm_to_ub_pad(scores0, view_8, 1, mul_3 * 4, 0, 0);  // #227 #68
                ev_mte2_v_ready_d0_1.set();  // #241
                ev_mte2_v_ready_d0_1.wait();  // #242
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #69
                vmuls(scores0.ptr(), scores0.ptr(), (float)(scale_log2), (uint8_t)(groups), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #69
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #70
                vcmpvs_gt((__ubuf__ uint8_t*)(cmp0.ptr()), scores0.ptr(), (float)(threshold0), (uint8_t)(groups), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #70
                ev_v_mte2_valid_d0_1.set();  // #243
                ev_v_mte2_valid_d0_1.wait();  // #244
                if (q0 != q1) {  // #71 #78
                    const GMTensor<float> view_9 = scores[(((h * (NT * NPAD)) + (q1 * NPAD)) + start)];  // #73 #72
                    ev_v_mte2_valid_d1_1.wait();  // #245
                    gm_to_ub_pad(scores1, view_9, 1, (groups * 64) * 4, 0, 0);  // #74 #228 #75
                    ev_mte2_v_ready_d1_1.set();  // #246
                    ev_mte2_v_ready_d1_1.wait();  // #247
                    vmuls(scores1.ptr(), scores1.ptr(), (float)(scale_log2), (uint8_t)(groups), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #76
                    PipeBarrier<PIPE_V>();  // c220: V-V hazard #77
                    vcmpvs_gt((__ubuf__ uint8_t*)(cmp1.ptr()), scores1.ptr(), (float)(threshold1), (uint8_t)(groups), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #77
                    ev_v_mte2_valid_d1_1.set();  // #248
                }
                ev_mte2_v_ready_d0_1.set();  // #249
                ev_mte2_v_ready_d0_1.wait();  // #250
                ev_v_s_ready_d2_1.set();  // #267
                ev_v_s_ready_d2_1.wait();  // #268
                for (int32_t gg = 0; gg < groups; gg += 1) {  // #217
                    const int32_t group_start = start + (gg * 64);  // #79 #80
                    const int32_t valid = Min(64, NT - group_start);  // #81 #82
                    const uint64_t valid_bits = 18446744073709551615ULL >> (64 - valid);  // #83 #84
                    const Tensor<uint64_t, Position::UB> point_3 = cmp0.as<uint64_t>()[gg];  // #85
                    const uint64_t exact = point_3.load(0);  // #86
                    const bool cmp_1 = q0 != q1;  // #87
                    uint64_t exact_2_state = 0ULL;  // #205
                    if (cmp_1) {  // #93
                        const Tensor<uint64_t, Position::UB> point_4 = cmp1.as<uint64_t>()[gg];  // #88
                        exact_2_state = exact | point_4.load(0);  // #89 #90 #206
                    } else {
                        exact_2_state = exact;  // #207
                    }
                    const uint64_t exact_2 = (uint64_t)(exact_2_state);  // #208
                    const int32_t lo = Max((q0 - 1) - group_start, 0);  // #94 #96 #97
                    const int32_t hi = Min((q1 + 2) - group_start, 64);  // #95 #98 #99
                    const bool cmp_2 = lo < hi;  // #101
                    uint64_t mask_2_state = 0ULL;  // #209
                    if (cmp_2) {  // #113
                        mask_2_state = (18446744073709551615ULL >> (64 - hi)) & ~((1ULL << lo) - 1ULL);  // #103 #104 #106 #108 #109 #110 #210
                    } else {
                        mask_2_state = 0ULL;  // #211
                    }
                    const uint64_t mask_2 = (uint64_t)(mask_2_state);  // #212
                    const uint64_t exact_3 = exact_2 | mask_2;  // #114
                    const int32_t lo_1 = Max(sink_start - group_start, 0);  // #115 #116
                    const int32_t hi_1 = Min(sink_end - group_start, 64);  // #117 #118
                    const bool cmp_3 = lo_1 < hi_1;  // #120
                    uint64_t mask_5_state = 0ULL;  // #213
                    if (cmp_3) {  // #132
                        mask_5_state = (18446744073709551615ULL >> (64 - hi_1)) & ~((1ULL << lo_1) - 1ULL);  // #122 #123 #125 #127 #128 #129 #214
                    } else {
                        mask_5_state = 0ULL;  // #215
                    }
                    const uint64_t mask_5 = (uint64_t)(mask_5_state);  // #216
                    const uint64_t exact_4 = exact_3 | mask_5;  // #133
                    const uint64_t exact_5 = exact_4 & valid_bits;  // #134
                    masks.store(gg, (uint64_t)(valid_bits & ~(exact_5)));  // #135 #136 #138
                    const int32_t mod_1 = gg & 1;  // #141
                    const int32_t pattern = ((gg / 2) * 4) + mod_1;  // #139 #140 #142
                    exact_bits.store(pattern, (uint64_t)(exact_5));  // #144
                }
                const int32_t mod_2 = groups & 1;  // #147
                if (mod_2 != 0) {  // #148 #155
                    const int32_t idiv_9 = groups / 2;  // #281
                    const int32_t idiv_10 = groups % 2;  // #282
                    const bool idiv_11 = idiv_10 != 0;  // #283
                    const bool idiv_12 = groups < 0;  // #284
                    const bool idiv_14 = idiv_12 != false;  // #286
                    const bool idiv_15 = idiv_11 && idiv_14;  // #287
                    const int32_t idiv_16 = (idiv_15) ? (1) : (0);  // #288
                    const int32_t div_1 = idiv_9 - idiv_16;  // #150
                    exact_bits.store((div_1 * 4) + 1, (uint64_t)(0ULL));  // #151 #152 #154
                }
                mask_ready.set();  // #156
                mask_ready.wait();  // #157
                const int32_t idiv_17 = groups / 2;  // #289
                const int32_t idiv_18 = groups % 2;  // #290
                const bool idiv_19 = idiv_18 != 0;  // #291
                const bool idiv_20 = groups < 0;  // #292
                const bool idiv_22 = idiv_20 == false;  // #294
                const bool idiv_23 = idiv_19 && idiv_22;  // #295
                const int32_t idiv_24 = (idiv_23) ? (1) : (0);  // #296
                const int32_t gather_repeats = idiv_17 + idiv_24;  // #158
                const bool cmp_5 = start == 0;  // #159
                int32_t chunk_count_2_state = 0;  // #218
                ev_mte3_v_valid_d1_1.wait();  // #257
                if (cmp_5) {  // #168
                    const Tensor<uint16_t, Position::UB> view_10 = dense_ids.as<uint16_t>();  // #160
                    SetMaskNorm();  // #161
                    ResetMask();  // #161
                    vreducev2((__ubuf__ uint16_t*)(indices.ptr()), (__ubuf__ uint16_t*)(view_10.ptr()), (__ubuf__ uint16_t*)(exact_bits.ptr()), (uint8_t)(gather_repeats), 1, 0, 8, 1);  // #161
                    SetFlag<PIPE_V, PIPE_S>(0);  // #161
                    WaitFlag<PIPE_V, PIPE_S>(0);  // #161
                    const int32_t chunk_count = (int32_t)get_rsvd_cnt();  // #161
                    chunk_count_2_state = chunk_count;  // #219
                } else {
                    vadds(offset_ids.ptr(), dense_ids.ptr(), (int16_t)((int16_t)(start)), (uint8_t)(gather_repeats), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #162 #163
                    const Tensor<uint16_t, Position::UB> view_11 = offset_ids.as<uint16_t>();  // #164
                    PipeBarrier<PIPE_V>();  // c220: V-V hazard #165
                    SetMaskNorm();  // #165
                    ResetMask();  // #165
                    vreducev2((__ubuf__ uint16_t*)(indices.ptr()), (__ubuf__ uint16_t*)(view_11.ptr()), (__ubuf__ uint16_t*)(exact_bits.ptr()), (uint8_t)(gather_repeats), 1, 0, 8, 1);  // #165
                    SetFlag<PIPE_V, PIPE_S>(0);  // #165
                    WaitFlag<PIPE_V, PIPE_S>(0);  // #165
                    const int32_t chunk_count_1 = (int32_t)get_rsvd_cnt();  // #165
                    chunk_count_2_state = chunk_count_1;  // #220
                }
                const int32_t chunk_count_2 = (int32_t)(chunk_count_2_state);  // #221
                SetFlag<PIPE_S, PIPE_MTE3>(0);  // #169
                WaitFlag<PIPE_S, PIPE_MTE3>(0);  // #170
                const GMTensor<uint64_t> view_12 = group_approx_mask[(((h * (NT_Q * NG)) + (q_tile * NG)) + first_group)];  // #173
                ub_to_gm_pad(view_12, masks, 1, groups * 8, 0, 0);  // #229 #174
                ev_v_mte3_ready_d1_1.set();  // #258
                ev_v_mte3_ready_d1_1.wait();  // #259
                if (chunk_count_2 > 0) {  // #175 #180
                    const GMTensor<uint16_t> view_13 = flat_exact_idx[(((h * (NT_Q * NPAD)) + (q_tile * NPAD)) + exact_count_carry)];  // #178
                    ub_to_gm_pad(view_13, indices, 1, chunk_count_2 * 2, 0, 0);  // #230 #179
                }
                SetFlag<PIPE_MTE3, PIPE_S>(0);  // #181
                WaitFlag<PIPE_MTE3, PIPE_S>(0);  // #182
                exact_count_2_state = exact_count_carry + chunk_count_2;  // #183 #222
                ev_v_mte2_valid_d0_1.set();  // #251
                ev_mte3_v_valid_d1_1.set();  // #260
            }
            ev_v_s_ready_d1_1.set();  // #269
            ev_v_s_ready_d1_1.wait();  // #270
            const int32_t exact_count_2 = (int32_t)(exact_count_2_state);  // #224
            counts.store(qi, (int32_t)(exact_count_2));  // #187
        }
        ev_v_mte3_ready_d0_1.set();  // #261
        ev_v_mte3_ready_d0_1.wait();  // #262
        ev_v_s_ready_d0_1.set();  // #271
        ev_v_s_ready_d0_1.wait();  // #272
        SetFlag<PIPE_S, PIPE_MTE3>(0);  // #190
        WaitFlag<PIPE_S, PIPE_MTE3>(0);  // #191
        const GMTensor<int32_t> view_14 = flat_exact_cnt[((h * NT_Q_PAD8) + q_base)];  // #194
        ub_to_gm_pad(view_14, counts, 1, 32, 0, 0);  // #195
        SetFlag<PIPE_MTE3, PIPE_S>(0);  // #196
        WaitFlag<PIPE_MTE3, PIPE_S>(0);  // #197
        ev_mte3_v_valid_d0_1.set();  // #263
    }
    return;  // #201
}
