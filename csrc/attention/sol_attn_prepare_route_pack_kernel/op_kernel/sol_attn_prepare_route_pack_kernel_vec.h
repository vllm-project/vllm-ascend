#pragma once
using namespace ascrip;

__aicore__ inline void sol_attn_prepare_route_pack_kernel_vec(GM_ADDR raw_scores_, GM_ADDR thresh_bh_, GM_ADDR group_approx_mask_, GM_ADDR flat_exact_cnt_, GM_ADDR flat_exact_idx_, int32_t HC, int32_t NT, int32_t NT_Q, int32_t NG, int32_t NPAD, int32_t NT_Q_PAD8, int32_t TILE_M, int32_t sink_s0, int32_t sink_s1, float scale_log2, GM_ADDR workspace)
{
    GMTensor<float> raw_scores((__gm__ float*)raw_scores_);
    GMTensor<float> thresh_bh((__gm__ float*)thresh_bh_);
    GMTensor<uint64_t> group_approx_mask((__gm__ uint64_t*)group_approx_mask_);
    GMTensor<int32_t> flat_exact_cnt((__gm__ int32_t*)flat_exact_cnt_);
    GMTensor<uint16_t> flat_exact_idx((__gm__ uint16_t*)flat_exact_idx_);
    SetAtomicNone();
    ResetMask();
    SetMaskNorm();
    const Tensor<float, Position::UB> ub_row0((uint64_t)(0));  // #1
    const Tensor<float, Position::UB> ub_row1((uint64_t)(16384));  // #2
    const Tensor<uint8_t, Position::UB> ub_cmp0_u8((uint64_t)(32768));  // #3
    const Tensor<uint8_t, Position::UB> ub_cmp1_u8((uint64_t)(33280));  // #4
    const Tensor<uint64_t, Position::UB> ub_cmp0_u64 = ub_cmp0_u8.as<uint64_t>();  // #5
    const Tensor<uint64_t, Position::UB> ub_cmp1_u64 = ub_cmp1_u8.as<uint64_t>();  // #6
    const Tensor<uint64_t, Position::UB> ub_mask_flat((uint64_t)(33792));  // #7
    const Tensor<uint64_t, Position::UB> ub_mask_out((uint64_t)(37888));  // #8
    const Tensor<uint16_t, Position::UB> ub_idx((uint64_t)(38400));  // #9
    const Tensor<int32_t, Position::UB> ub_cnt((uint64_t)(46720));  // #10
    const int32_t vid = GetVecIdx();  // #11
    const int32_t vnum = GetVecNum();  // #12
    const int32_t q_chunks = NT_Q_PAD8 / 8;  // #13
    const int32_t total_chunks = HC * q_chunks;  // #14
    const GMTensor<uint64_t> mask_flat = group_approx_mask;  // #25 #24
    for (int32_t chunk_id = vid; chunk_id < total_chunks; chunk_id += vnum) {  // #502
        const int32_t idiv_1 = chunk_id / q_chunks;  // #538
        const int32_t idiv_2 = chunk_id % q_chunks;  // #539
        const bool idiv_3 = idiv_2 != 0;  // #540
        const bool idiv_4 = chunk_id < 0;  // #541
        const bool idiv_6 = idiv_4 != false;  // #543
        const bool idiv_7 = idiv_3 && idiv_6;  // #544
        const int32_t idiv_8 = (idiv_7) ? (1) : (0);  // #545
        const int32_t hc_i = idiv_1 - idiv_8;  // #26
        const int32_t q8 = ascrip::FloorMod<int32_t>(chunk_id, q_chunks);  // #27
        const int32_t q_base = q8 * 8;  // #28
        const int32_t valid_q = Min(8, NT_Q - q_base);  // #29 #30
        for (int32_t zi = 0; zi < 8; zi += 1) {  // #370
            ub_cnt.store(zi, (int32_t)(0));  // #33
        }
        for (int32_t qi = 0; qi < valid_q; qi += 1) {  // #501
            const int32_t q_idx = q_base + qi;  // #36
            const bool cmp = TILE_M == 128;  // #37
            int32_t qb0_1_state = 0;  // #371
            int32_t qb1_1_state = 0;  // #372
            if (cmp) {  // #44
                const int32_t qb0 = 2 * q_idx;  // #38
                const int32_t sub_1 = NT + -1;  // #40
                const int32_t qb1 = Min(qb0 + 1, sub_1);  // #39 #41
                qb0_1_state = qb0;  // #373
                qb1_1_state = qb1;  // #374
            } else {
                qb0_1_state = q_idx;  // #375
                qb1_1_state = q_idx;  // #376
            }
            const int32_t qb0_1 = (int32_t)(qb0_1_state);  // #377
            const int32_t qb1_1 = (int32_t)(qb1_1_state);  // #378
            const float thresh0 = thresh_bh.load((hc_i * NT) + qb0_1);  // #46 #47 #48
            const bool cmp_1 = qb1_1 != qb0_1;  // #49
            float thresh1_1_state = 0.0f;  // #379
            if (cmp_1) {  // #56
                thresh1_1_state = thresh_bh.load((hc_i * NT) + qb1_1);  // #51 #52 #53 #380
            } else {
                thresh1_1_state = thresh0;  // #381
            }
            const float thresh1_1 = (float)(thresh1_1_state);  // #382
            int32_t buf_len_8_state = 0;  // #383
            int32_t write_pos_4_state = 0;  // #384
            for (int32_t g_base = 0; g_base < NG; g_base += 64) {  // #497
                const int32_t buf_len_carry = (int32_t)(buf_len_8_state);  // #385
                const int32_t write_pos_carry = (int32_t)(write_pos_4_state);  // #386
                const int32_t cur_g = Min(64, NG - g_base);  // #59 #60
                const int32_t col0 = g_base * 64;  // #61
                const int32_t cur_cols = cur_g * 64;  // #62
                const Tensor<float, Position::UB> view = ub_row0;  // #63
                const int32_t add_3 = col0 + cur_cols;  // #64
                const int32_t sub_3 = add_3 - col0;  // #65
                const GMTensor<float> view_1 = raw_scores[(((hc_i * (NT * NPAD)) + (qb0_1 * NPAD)) + col0)];  // #66
                const int32_t sub_22 = 4096 - sub_3;  // #506
                const int32_t idiv_9 = sub_22 / 8;  // #546
                const int32_t idiv_10 = sub_22 % 8;  // #547
                const bool idiv_11 = idiv_10 != 0;  // #548
                const bool idiv_12 = sub_22 < 0;  // #549
                const bool idiv_14 = idiv_12 != false;  // #551
                const bool idiv_15 = idiv_11 && idiv_14;  // #552
                const int32_t idiv_16 = (idiv_15) ? (1) : (0);  // #553
                const int32_t div_2 = idiv_9 - idiv_16;  // #507
                gm_to_ub_pad(view, view_1, 1, sub_3 * 4, 0, div_2);  // #503 #67
                if (qb1_1 != qb0_1) {  // #68 #74
                    const Tensor<float, Position::UB> view_2 = ub_row1;  // #69
                    const int32_t sub_4 = add_3 - col0;  // #71
                    const GMTensor<float> view_3 = raw_scores[(((hc_i * (NT * NPAD)) + (qb1_1 * NPAD)) + col0)];  // #72
                    const int32_t sub_24 = 4096 - sub_4;  // #511
                    const int32_t idiv_17 = sub_24 / 8;  // #554
                    const int32_t idiv_18 = sub_24 % 8;  // #555
                    const bool idiv_19 = idiv_18 != 0;  // #556
                    const bool idiv_20 = sub_24 < 0;  // #557
                    const bool idiv_22 = idiv_20 != false;  // #559
                    const bool idiv_23 = idiv_19 && idiv_22;  // #560
                    const int32_t idiv_24 = (idiv_23) ? (1) : (0);  // #561
                    const int32_t div_3 = idiv_17 - idiv_24;  // #512
                    gm_to_ub_pad(view_2, view_3, 1, sub_4 * 4, 0, div_3);  // #508 #73
                }
                SetFlag<PIPE_MTE2, PIPE_V>(0);  // #75
                WaitFlag<PIPE_MTE2, PIPE_V>(0);  // #76
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #77
                vmuls(ub_row0.ptr(), ub_row0.ptr(), (float)(scale_log2), (uint8_t)(cur_g), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #77
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #78
                vcmpvs_gt((__ubuf__ uint8_t*)(ub_cmp0_u8.ptr()), ub_row0.ptr(), (float)(thresh0), (uint8_t)(cur_g), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #78
                if (qb1_1 != qb0_1) {  // #79 #82
                    vmuls(ub_row1.ptr(), ub_row1.ptr(), (float)(scale_log2), (uint8_t)(cur_g), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #80
                    PipeBarrier<PIPE_V>();  // c220: V-V hazard #81
                    vcmpvs_gt((__ubuf__ uint8_t*)(ub_cmp1_u8.ptr()), ub_row1.ptr(), (float)(thresh1_1), (uint8_t)(cur_g), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #81
                }
                SetFlag<PIPE_V, PIPE_MTE2>(0);  // #83
                WaitFlag<PIPE_V, PIPE_MTE2>(0);  // #84
                SetFlag<PIPE_V, PIPE_S>(0);  // #85
                WaitFlag<PIPE_V, PIPE_S>(0);  // #86
                int32_t buf_len_4_state = buf_len_carry;  // #387
                for (int32_t g_loc = 0; g_loc < cur_g; g_loc += 1) {  // #476
                    const int32_t buf_len_carry_1 = (int32_t)(buf_len_4_state);  // #388
                    const int32_t g = g_base + g_loc;  // #87
                    const int32_t g_start = g * 64;  // #88
                    const int32_t valid_g = Min(64, Max(NT - g_start, 0));  // #89 #90 #91
                    const bool cmp_4 = valid_g <= 0;  // #92
                    uint64_t valid_u64_3_state = 0ULL;  // #389
                    if (cmp_4) {  // #104
                        valid_u64_3_state = 0ULL;  // #390
                    } else {
                        const bool cmp_5 = valid_g < 64;  // #94
                        uint64_t valid_u64_2_state = 0ULL;  // #391
                        if (cmp_5) {  // #101
                            valid_u64_2_state = (1ULL << valid_g) - 1ULL;  // #96 #98 #392
                        } else {
                            valid_u64_2_state = 18446744073709551615ULL;  // #393
                        }
                        valid_u64_3_state = (uint64_t)(valid_u64_2_state);  // #394 #395
                    }
                    const uint64_t valid_u64_3 = (uint64_t)(valid_u64_3_state);  // #396
                    const int32_t s_lo = Max(sink_s0 - g_start, 0);  // #105 #106
                    const int32_t s_hi = Min(sink_s1 - g_start, valid_g);  // #107 #108
                    const bool cmp_6 = s_lo < s_hi;  // #110
                    uint64_t sink_m_2_state = 0ULL;  // #397
                    if (cmp_6) {  // #132
                        const bool cmp_7 = s_hi < 64;  // #111
                        uint64_t s_hi_m_1_state = 0ULL;  // #398
                        if (cmp_7) {  // #118
                            s_hi_m_1_state = (1ULL << s_hi) - 1ULL;  // #113 #115 #399
                        } else {
                            s_hi_m_1_state = 18446744073709551615ULL;  // #400
                        }
                        const uint64_t s_hi_m_1 = (uint64_t)(s_hi_m_1_state);  // #401
                        const bool cmp_8 = s_lo > 0;  // #120
                        uint64_t s_lo_m_2_state = 0ULL;  // #402
                        if (cmp_8) {  // #127
                            s_lo_m_2_state = (1ULL << s_lo) - 1ULL;  // #122 #124 #403
                        } else {
                            s_lo_m_2_state = 0ULL;  // #404
                        }
                        const uint64_t s_lo_m_2 = (uint64_t)(s_lo_m_2_state);  // #405
                        sink_m_2_state = s_hi_m_1 & ~(s_lo_m_2);  // #128 #129 #406
                    } else {
                        sink_m_2_state = 0ULL;  // #407
                    }
                    const uint64_t sink_m_2 = (uint64_t)(sink_m_2_state);  // #408
                    const int32_t d0_lo = Max((qb0_1 - 1) - g_start, 0);  // #133 #134 #135
                    const int32_t d0_hi = Min((qb0_1 + 2) - g_start, valid_g);  // #136 #137 #138
                    const bool cmp_9 = d0_lo < d0_hi;  // #140
                    uint64_t diag0_2_state = 0ULL;  // #409
                    if (cmp_9) {  // #162
                        const bool cmp_10 = d0_hi < 64;  // #141
                        uint64_t d0_hi_m_1_state = 0ULL;  // #410
                        if (cmp_10) {  // #148
                            d0_hi_m_1_state = (1ULL << d0_hi) - 1ULL;  // #143 #145 #411
                        } else {
                            d0_hi_m_1_state = 18446744073709551615ULL;  // #412
                        }
                        const uint64_t d0_hi_m_1 = (uint64_t)(d0_hi_m_1_state);  // #413
                        const bool cmp_11 = d0_lo > 0;  // #150
                        uint64_t d0_lo_m_2_state = 0ULL;  // #414
                        if (cmp_11) {  // #157
                            d0_lo_m_2_state = (1ULL << d0_lo) - 1ULL;  // #152 #154 #415
                        } else {
                            d0_lo_m_2_state = 0ULL;  // #416
                        }
                        const uint64_t d0_lo_m_2 = (uint64_t)(d0_lo_m_2_state);  // #417
                        diag0_2_state = d0_hi_m_1 & ~(d0_lo_m_2);  // #158 #159 #418
                    } else {
                        diag0_2_state = 0ULL;  // #419
                    }
                    const uint64_t diag0_2 = (uint64_t)(diag0_2_state);  // #420
                    const Tensor<uint64_t, Position::UB> point_3 = ub_cmp0_u8.as<uint64_t>()[g_loc];  // #163
                    const uint64_t gt0 = point_3.load(0);  // #164
                    const uint64_t exact0 = ((gt0 | diag0_2) | sink_m_2) & valid_u64_3;  // #165 #166 #167
                    const uint64_t approx_final = valid_u64_3 & ~(exact0);  // #168 #169
                    const bool cmp_12 = qb1_1 != qb0_1;  // #170
                    uint64_t approx_final_2_state = 0ULL;  // #421
                    if (cmp_12) {  // #211
                        const int32_t d1_lo = Max((qb1_1 - 1) - g_start, 0);  // #171 #172 #173
                        const int32_t d1_hi = Min((qb1_1 + 2) - g_start, valid_g);  // #174 #175 #176
                        const bool cmp_13 = d1_lo < d1_hi;  // #178
                        uint64_t diag1_2_state = 0ULL;  // #422
                        if (cmp_13) {  // #200
                            const bool cmp_14 = d1_hi < 64;  // #179
                            uint64_t d1_hi_m_1_state = 0ULL;  // #423
                            if (cmp_14) {  // #186
                                d1_hi_m_1_state = (1ULL << d1_hi) - 1ULL;  // #181 #183 #424
                            } else {
                                d1_hi_m_1_state = 18446744073709551615ULL;  // #425
                            }
                            const uint64_t d1_hi_m_1 = (uint64_t)(d1_hi_m_1_state);  // #426
                            const bool cmp_15 = d1_lo > 0;  // #188
                            uint64_t d1_lo_m_2_state = 0ULL;  // #427
                            if (cmp_15) {  // #195
                                d1_lo_m_2_state = (1ULL << d1_lo) - 1ULL;  // #190 #192 #428
                            } else {
                                d1_lo_m_2_state = 0ULL;  // #429
                            }
                            const uint64_t d1_lo_m_2 = (uint64_t)(d1_lo_m_2_state);  // #430
                            diag1_2_state = d1_hi_m_1 & ~(d1_lo_m_2);  // #196 #197 #431
                        } else {
                            diag1_2_state = 0ULL;  // #432
                        }
                        const uint64_t diag1_2 = (uint64_t)(diag1_2_state);  // #433
                        const Tensor<uint64_t, Position::UB> point_4 = ub_cmp1_u8.as<uint64_t>()[g_loc];  // #201
                        const uint64_t gt1 = point_4.load(0);  // #202
                        const uint64_t exact1 = ((gt1 | diag1_2) | sink_m_2) & valid_u64_3;  // #203 #204 #205
                        const uint64_t approx1 = valid_u64_3 & ~(exact1);  // #206 #207
                        approx_final_2_state = approx_final & approx1;  // #208 #434
                    } else {
                        approx_final_2_state = approx_final;  // #435
                    }
                    const uint64_t approx_final_2 = (uint64_t)(approx_final_2_state);  // #436
                    if (NG <= 64) {  // #212 #219
                        ub_mask_flat.store((qi * NG) + g_loc, (uint64_t)(approx_final_2));  // #213 #214 #216
                    } else {
                        ub_mask_out.store(g_loc, (uint64_t)(approx_final_2));  // #218
                    }
                    const uint64_t exact_final = valid_u64_3 & ~(approx_final_2);  // #220 #221
                    const bool cmp_17 = exact_final != 0ULL;  // #222
                    int32_t buf_len_3_state = 0;  // #437
                    if (cmp_17) {  // #274
                        int32_t buf_len_2_state = buf_len_carry_1;  // #438
                        uint64_t rem_state = exact_final;  // #439
                        for (int32_t _ = 0; _ < 64; _ += 1) {  // #470
                            const int32_t buf_len_carry_2 = (int32_t)(buf_len_2_state);  // #440
                            const uint64_t rem_carry = (uint64_t)(rem_state);  // #441
                            if (rem_carry == 0ULL) {  // #223 #225
                                buf_len_2_state = buf_len_carry_2;  // #442
                                rem_state = rem_carry;  // #443
                                break;  // #224
                            }
                            const uint64_t rem_next = rem_carry & (rem_carry - 1ULL);  // #226 #227
                            const uint64_t lsb = rem_carry ^ rem_next;  // #228
                            const bool cmp_19 = (lsb & 18446744069414584320ULL) != 0ULL;  // #229 #230
                            int32_t b5_state = 0;  // #444
                            if (cmp_19) {  // #233
                                b5_state = 32;  // #445
                            } else {
                                b5_state = 0;  // #446
                            }
                            const int32_t b5 = (int32_t)(b5_state);  // #447
                            const bool cmp_20 = (lsb & 18446462603027742720ULL) != 0ULL;  // #234 #235
                            int32_t b4_state = 0;  // #448
                            if (cmp_20) {  // #238
                                b4_state = 16;  // #449
                            } else {
                                b4_state = 0;  // #450
                            }
                            const int32_t b4 = (int32_t)(b4_state);  // #451
                            const bool cmp_21 = (lsb & 18374966859414961920ULL) != 0ULL;  // #239 #240
                            int32_t b3_state = 0;  // #452
                            if (cmp_21) {  // #243
                                b3_state = 8;  // #453
                            } else {
                                b3_state = 0;  // #454
                            }
                            const int32_t b3 = (int32_t)(b3_state);  // #455
                            const bool cmp_22 = (lsb & 17361641481138401520ULL) != 0ULL;  // #244 #245
                            int32_t b2_state = 0;  // #456
                            if (cmp_22) {  // #248
                                b2_state = 4;  // #457
                            } else {
                                b2_state = 0;  // #458
                            }
                            const int32_t b2 = (int32_t)(b2_state);  // #459
                            const bool cmp_23 = (lsb & 14757395258967641292ULL) != 0ULL;  // #249 #250
                            int32_t b1_state = 0;  // #460
                            if (cmp_23) {  // #253
                                b1_state = 2;  // #461
                            } else {
                                b1_state = 0;  // #462
                            }
                            const int32_t b1 = (int32_t)(b1_state);  // #463
                            const bool cmp_24 = (lsb & 12297829382473034410ULL) != 0ULL;  // #254 #255
                            int32_t b0_state = 0;  // #464
                            if (cmp_24) {  // #258
                                b0_state = 1;  // #465
                            } else {
                                b0_state = 0;  // #466
                            }
                            const int32_t b0 = (int32_t)(b0_state);  // #467
                            const int32_t bit_p = ((((b5 + b4) + b3) + b2) + b1) + b0;  // #259 #260 #261 #262 #263
                            ub_idx.store(buf_len_carry_2, (uint16_t)((uint16_t)(g_start + bit_p)));  // #264 #265 #267
                            buf_len_2_state = (int32_t)(buf_len_carry_2 + 1);  // #268 #269 #468
                            rem_state = rem_next;  // #469
                        }
                        buf_len_3_state = (int32_t)(buf_len_2_state);  // #471 #472
                    } else {
                        buf_len_3_state = buf_len_carry_1;  // #473
                    }
                    buf_len_4_state = (int32_t)(buf_len_3_state);  // #474 #475
                }
                const int32_t buf_len_4 = (int32_t)(buf_len_4_state);  // #477
                SetFlag<PIPE_S, PIPE_V>(0);  // #277
                WaitFlag<PIPE_S, PIPE_V>(0);  // #278
                if (NG > 64) {  // #279 #289
                    SetFlag<PIPE_S, PIPE_MTE3>(0);  // #280
                    WaitFlag<PIPE_S, PIPE_MTE3>(0);  // #281
                    const GMTensor<uint64_t> view_4 = group_approx_mask[(((hc_i * (NT_Q * NG)) + (q_idx * NG)) + g_base)];  // #284
                    const Tensor<uint64_t, Position::UB> view_5 = ub_mask_out;  // #285
                    const int32_t sub_26 = 64 - cur_g;  // #516
                    const int32_t idiv_25 = sub_26 / 4;  // #562
                    const int32_t idiv_26 = sub_26 % 4;  // #563
                    const bool idiv_27 = idiv_26 != 0;  // #564
                    const bool idiv_28 = sub_26 < 0;  // #565
                    const bool idiv_30 = idiv_28 != false;  // #567
                    const bool idiv_31 = idiv_27 && idiv_30;  // #568
                    const int32_t idiv_32 = (idiv_31) ? (1) : (0);  // #569
                    const int32_t div_4 = idiv_25 - idiv_32;  // #517
                    ub_to_gm_pad(view_4, view_5, 1, cur_g * 8, div_4, 0);  // #513 #286
                    SetFlag<PIPE_MTE3, PIPE_S>(0);  // #287
                    WaitFlag<PIPE_MTE3, PIPE_S>(0);  // #288
                }
                const bool cmp_26 = (g_base + cur_g) < NG;  // #290 #291
                int32_t buf_len_7_state = 0;  // #478
                int32_t write_pos_3_state = 0;  // #479
                if (cmp_26) {  // #320
                    const int32_t idiv_33 = buf_len_4 / 64;  // #570
                    const int32_t idiv_34 = buf_len_4 % 64;  // #571
                    const bool idiv_35 = idiv_34 != 0;  // #572
                    const bool idiv_36 = buf_len_4 < 0;  // #573
                    const bool idiv_38 = idiv_36 != false;  // #575
                    const bool idiv_39 = idiv_35 && idiv_38;  // #576
                    const int32_t idiv_40 = (idiv_39) ? (1) : (0);  // #577
                    const int32_t div = idiv_33 - idiv_40;  // #292
                    const int32_t full64 = div * 64;  // #293
                    const bool cmp_27 = full64 > 0;  // #294
                    int32_t buf_len_6_state = 0;  // #480
                    int32_t write_pos_2_state = 0;  // #481
                    if (cmp_27) {  // #317
                        SetFlag<PIPE_S, PIPE_MTE3>(0);  // #295
                        WaitFlag<PIPE_S, PIPE_MTE3>(0);  // #296
                        const int32_t sub_16 = (write_pos_carry + full64) - write_pos_carry;  // #297 #298
                        const GMTensor<uint16_t> view_6 = flat_exact_idx[(((hc_i * (NT_Q * NPAD)) + (q_idx * NPAD)) + write_pos_carry)];  // #299
                        const Tensor<uint16_t, Position::UB> view_7 = ub_idx;  // #300
                        const int32_t sub_28 = 4160 - sub_16;  // #521
                        const int32_t idiv_41 = sub_28 / 16;  // #578
                        const int32_t idiv_42 = sub_28 % 16;  // #579
                        const bool idiv_43 = idiv_42 != 0;  // #580
                        const bool idiv_44 = sub_28 < 0;  // #581
                        const bool idiv_46 = idiv_44 != false;  // #583
                        const bool idiv_47 = idiv_43 && idiv_46;  // #584
                        const int32_t idiv_48 = (idiv_47) ? (1) : (0);  // #585
                        const int32_t div_5 = idiv_41 - idiv_48;  // #522
                        ub_to_gm_pad(view_6, view_7, 1, sub_16 * 2, div_5, 0);  // #518 #301
                        SetFlag<PIPE_MTE3, PIPE_S>(0);  // #302
                        WaitFlag<PIPE_MTE3, PIPE_S>(0);  // #303
                        const int32_t rem_len = buf_len_4 & 63;  // #304
                        for (int32_t ri = 0; ri < rem_len; ri += 1) {  // #482
                            const uint16_t rem_v = ub_idx.load(full64 + ri);  // #305 #307
                            ub_idx.store(ri, (uint16_t)(rem_v));  // #309
                        }
                        const int32_t write_pos_1 = (int32_t)(write_pos_carry + full64);  // #312 #313
                        buf_len_6_state = (int32_t)(rem_len);  // #314 #483
                        write_pos_2_state = write_pos_1;  // #484
                    } else {
                        buf_len_6_state = buf_len_4;  // #485
                        write_pos_2_state = write_pos_carry;  // #486
                    }
                    const int32_t buf_len_6 = (int32_t)(buf_len_6_state);  // #487
                    const int32_t write_pos_2 = (int32_t)(write_pos_2_state);  // #488
                    buf_len_7_state = buf_len_6;  // #489
                    write_pos_3_state = write_pos_2;  // #490
                } else {
                    buf_len_7_state = buf_len_4;  // #491
                    write_pos_3_state = write_pos_carry;  // #492
                }
                const int32_t buf_len_7 = (int32_t)(buf_len_7_state);  // #493
                const int32_t write_pos_3 = (int32_t)(write_pos_3_state);  // #494
                buf_len_8_state = buf_len_7;  // #495
                write_pos_4_state = write_pos_3;  // #496
            }
            const int32_t buf_len_8 = (int32_t)(buf_len_8_state);  // #498
            const int32_t write_pos_4 = (int32_t)(write_pos_4_state);  // #499
            const int32_t total_exact = write_pos_4 + buf_len_8;  // #323
            if (buf_len_8 > 0) {  // #324 #343
                const int32_t sub_17 = (buf_len_8 + 64) - 1;  // #325 #326
                const int32_t idiv_49 = sub_17 / 64;  // #586
                const int32_t idiv_50 = sub_17 % 64;  // #587
                const bool idiv_51 = idiv_50 != 0;  // #588
                const bool idiv_52 = sub_17 < 0;  // #589
                const bool idiv_54 = idiv_52 != false;  // #591
                const bool idiv_55 = idiv_51 && idiv_54;  // #592
                const int32_t idiv_56 = (idiv_55) ? (1) : (0);  // #593
                const int32_t div_1 = idiv_49 - idiv_56;  // #327
                const int32_t flush64 = div_1 * 64;  // #328
                for (int32_t pad_c = buf_len_8; pad_c < flush64; pad_c += 1) {  // #500
                    ub_idx.store(pad_c, (uint16_t)((uint16_t)(0)));  // #331
                }
                SetFlag<PIPE_S, PIPE_MTE3>(0);  // #334
                WaitFlag<PIPE_S, PIPE_MTE3>(0);  // #335
                const int32_t sub_18 = (write_pos_4 + flush64) - write_pos_4;  // #336 #337
                const GMTensor<uint16_t> view_8 = flat_exact_idx[(((hc_i * (NT_Q * NPAD)) + (q_idx * NPAD)) + write_pos_4)];  // #338
                const Tensor<uint16_t, Position::UB> view_9 = ub_idx;  // #339
                const int32_t sub_30 = 4160 - sub_18;  // #526
                const int32_t idiv_57 = sub_30 / 16;  // #594
                const int32_t idiv_58 = sub_30 % 16;  // #595
                const bool idiv_59 = idiv_58 != 0;  // #596
                const bool idiv_60 = sub_30 < 0;  // #597
                const bool idiv_62 = idiv_60 != false;  // #599
                const bool idiv_63 = idiv_59 && idiv_62;  // #600
                const int32_t idiv_64 = (idiv_63) ? (1) : (0);  // #601
                const int32_t div_6 = idiv_57 - idiv_64;  // #527
                ub_to_gm_pad(view_8, view_9, 1, sub_18 * 2, div_6, 0);  // #523 #340
                SetFlag<PIPE_MTE3, PIPE_S>(0);  // #341
                WaitFlag<PIPE_MTE3, PIPE_S>(0);  // #342
            }
            ub_cnt.store(qi, (int32_t)(total_exact));  // #345
        }
        SetFlag<PIPE_S, PIPE_MTE3>(1);  // #348
        WaitFlag<PIPE_S, PIPE_MTE3>(1);  // #349
        if (NG <= 64) {  // #350 #360
            if (valid_q > 0) {  // #351 #359
                const int32_t flat_off = q_base * NG;  // #352
                const int32_t flat_len = valid_q * NG;  // #353
                const GMTensor<uint64_t> view_10 = group_approx_mask[((hc_i * (NT_Q * NG)) + flat_off)];  // #356
                const Tensor<uint64_t, Position::UB> view_11 = ub_mask_flat;  // #357
                const int32_t sub_32 = 512 - flat_len;  // #531
                const int32_t idiv_65 = sub_32 / 4;  // #602
                const int32_t idiv_66 = sub_32 % 4;  // #603
                const bool idiv_67 = idiv_66 != 0;  // #604
                const bool idiv_68 = sub_32 < 0;  // #605
                const bool idiv_70 = idiv_68 != false;  // #607
                const bool idiv_71 = idiv_67 && idiv_70;  // #608
                const int32_t idiv_72 = (idiv_71) ? (1) : (0);  // #609
                const int32_t div_7 = idiv_65 - idiv_72;  // #532
                ub_to_gm_pad(view_10, view_11, 1, flat_len * 8, div_7, 0);  // #528 #358
            }
        }
        const GMTensor<int32_t> view_12 = flat_exact_cnt[((hc_i * NT_Q_PAD8) + q_base)];  // #363
        ub_to_gm_pad(view_12, ub_cnt, 1, 32, 0, 0);  // #364
        SetFlag<PIPE_MTE3, PIPE_S>(1);  // #365
        WaitFlag<PIPE_MTE3, PIPE_S>(1);  // #366
    }
    return;  // #369
}
