#pragma once
using namespace ascrip;

__aicore__ inline void sol_attn_prepare_qkv_kernel_vec(GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR k_pad_, GM_ADDR v_pad_, GM_ADDR kc_padded_, GM_ADDR vc_padded_, GM_ADDR kc_pad_f32_, GM_ADDR q_bar_f32_, GM_ADDR last_group_lens_, int32_t B, int32_t T, int32_t H, int32_t NT, int32_t NG, int32_t NPAD, int32_t T_PAD_KV, float inv_tail_len, GM_ADDR workspace)
{
    GMTensor<bfloat16_t> q((__gm__ bfloat16_t*)q_);
    GMTensor<bfloat16_t> k((__gm__ bfloat16_t*)k_);
    GMTensor<bfloat16_t> v((__gm__ bfloat16_t*)v_);
    GMTensor<bfloat16_t> k_pad((__gm__ bfloat16_t*)k_pad_);
    GMTensor<bfloat16_t> v_pad((__gm__ bfloat16_t*)v_pad_);
    GMTensor<bfloat16_t> kc_padded((__gm__ bfloat16_t*)kc_padded_);
    GMTensor<bfloat16_t> vc_padded((__gm__ bfloat16_t*)vc_padded_);
    GMTensor<float> kc_pad_f32((__gm__ float*)kc_pad_f32_);
    GMTensor<float> q_bar_f32((__gm__ float*)q_bar_f32_);
    GMTensor<float> last_group_lens((__gm__ float*)last_group_lens_);
    SetAtomicNone();
    ResetMask();
    SetMaskNorm();
    SEvent<PIPE_MTE2, PIPE_MTE3, 0, 0> ev_mte2_mte3_ready_d0_1;  // guards ub_k_bf16, ub_v_bf16 #223
    SEvent<PIPE_MTE3, PIPE_MTE2, 1, 0> ev_mte3_mte2_valid_d0_1;  // guards ub_k_bf16, ub_v_bf16 #224
    SEvent<PIPE_MTE2, PIPE_MTE3, 0, 0> ev_mte2_mte3_ready_d1_1;  // guards ub_k_bf16, ub_v_bf16 #225
    SEvent<PIPE_MTE3, PIPE_MTE2, 1, 1> ev_mte3_mte2_valid_d1_1;  // guards ub_k_bf16, ub_v_bf16 #226
    SEvent<PIPE_MTE2, PIPE_V, 0, 0> ev_mte2_v_ready_d0_1;  // guards ub_k_bf16, ub_q_bf16, ub_v_bf16 #235
    SEvent<PIPE_V, PIPE_MTE2, 1, 0> ev_v_mte2_valid_d0_1;  // guards ub_k_bf16, ub_q_bf16, ub_v_bf16 #236
    SEvent<PIPE_MTE2, PIPE_V, 0, 0> ev_mte2_v_ready_d1_1;  // guards ub_k_bf16, ub_q_bf16, ub_v_bf16 #237
    SEvent<PIPE_V, PIPE_MTE2, 1, 1> ev_v_mte2_valid_d1_1;  // guards ub_k_bf16, ub_q_bf16, ub_v_bf16 #238
    SEvent<PIPE_V, PIPE_MTE3, 0, 0> ev_v_mte3_ready_1;  // guards ub_f32, ub_k_bf16, ub_kc_bf16, ub_kc_f32, ub_qbar_f32, ub_v_bf16, ub_vc_bf16 #249
    SEvent<PIPE_MTE3, PIPE_V, 1, 0> ev_mte3_v_valid_1;  // guards ub_f32, ub_k_bf16, ub_kc_bf16, ub_kc_f32, ub_qbar_f32, ub_v_bf16, ub_vc_bf16 #250
    const Tensor<bfloat16_t, Position::UB> ub_k_bf16((uint64_t)(0));  // #1
    const Tensor<bfloat16_t, Position::UB> ub_v_bf16((uint64_t)(16384));  // #2
    const Tensor<bfloat16_t, Position::UB> ub_q_bf16((uint64_t)(32768));  // #3
    const Tensor<float, Position::UB> ub_f32((uint64_t)(49152));  // #4
    const Tensor<bfloat16_t, Position::UB> ub_kc_bf16((uint64_t)(81920));  // #5
    const Tensor<bfloat16_t, Position::UB> ub_vc_bf16((uint64_t)(82176));  // #6
    const Tensor<float, Position::UB> ub_kc_f32((uint64_t)(82432));  // #7
    const Tensor<float, Position::UB> ub_qbar_f32((uint64_t)(82944));  // #8
    const Tensor<float, Position::UB> ub_lens((uint64_t)(83456));  // #9
    const int32_t vid = GetVecIdx();  // #10
    const int32_t vnum = GetVecNum();  // #11
    const int32_t total_active = (B * NT) * H;  // #12 #13
    ev_mte3_mte2_valid_d0_1.wait();  // #227
    ev_v_mte2_valid_d0_1.wait();  // #239
    for (int32_t tid = vid; tid < total_active; tid += vnum) {  // #205
        const int32_t idiv_1 = tid / H;  // #271
        const int32_t idiv_2 = tid % H;  // #272
        const bool idiv_3 = idiv_2 != 0;  // #273
        const bool idiv_4 = tid < 0;  // #274
        const bool idiv_6 = idiv_4 != false;  // #276
        const bool idiv_7 = idiv_3 && idiv_6;  // #277
        const int32_t idiv_8 = (idiv_7) ? (1) : (0);  // #278
        const int32_t b_nt = idiv_1 - idiv_8;  // #14
        const int32_t h = ascrip::FloorMod<int32_t>(tid, H);  // #15
        const int32_t idiv_9 = b_nt / NT;  // #279
        const int32_t idiv_10 = b_nt % NT;  // #280
        const bool idiv_11 = idiv_10 != 0;  // #281
        const bool idiv_12 = b_nt < 0;  // #282
        const bool idiv_14 = idiv_12 != false;  // #284
        const bool idiv_15 = idiv_11 && idiv_14;  // #285
        const int32_t idiv_16 = (idiv_15) ? (1) : (0);  // #286
        const int32_t b = idiv_9 - idiv_16;  // #16
        const int32_t blk = ascrip::FloorMod<int32_t>(b_nt, NT);  // #17
        const int32_t tok_start = blk * 64;  // #18
        const int32_t valid_rows = Min(64, T - tok_start);  // #19 #20
        const float inv_len = (float)(0.015625f);  // #21
        const bool cmp = valid_rows < 64;  // #22
        float inv_len_1_state = 0.0f;  // #201
        if (cmp) {  // #25
            inv_len_1_state = inv_tail_len;  // #202
        } else {
            inv_len_1_state = inv_len;  // #203
        }
        const float inv_len_1 = (float)(inv_len_1_state);  // #204
        ev_v_mte2_valid_d1_1.wait();  // #241
        ev_mte3_v_valid_1.wait();  // #251
        if (valid_rows < 64) {  // #26 #31
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #27
            vector_dup(ub_f32.ptr(), (float)(0.0f), (uint8_t)(128), (uint16_t)1, 1, (uint8_t)8, 0);  // #27
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #28
            vconv_f322bf16r(ub_k_bf16.ptr(), ub_f32.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)4, (uint16_t)8);  // #28
            vconv_f322bf16r(ub_v_bf16.ptr(), ub_f32.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)4, (uint16_t)8);  // #29
            vconv_f322bf16r(ub_q_bf16.ptr(), ub_f32.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)4, (uint16_t)8);  // #30
        }
        const Tensor<bfloat16_t, Position::UB> view = ub_k_bf16;  // #32
        const GMTensor<bfloat16_t> view_1 = k[(((b * ((T * H) * 128)) + (tok_start * (H * 128))) + (h * 128))];  // #35
        const int32_t sub_16 = ((128 + ((H * 128) - 128)) * 2) - 256;  // #208 #209 #210 #211 #212
        ev_mte3_mte2_valid_d1_1.wait();  // #229
        ev_v_mte2_valid_d1_1.set();  // #242
        ev_v_mte2_valid_d1_1.wait();  // #243
        gm_to_ub_pad(view, view_1, valid_rows, 256, sub_16, 0);  // #36
        const Tensor<bfloat16_t, Position::UB> view_2 = ub_v_bf16;  // #37
        const GMTensor<bfloat16_t> view_3 = v[(((b * ((T * H) * 128)) + (tok_start * (H * 128))) + (h * 128))];  // #40
        gm_to_ub_pad(view_2, view_3, valid_rows, 256, sub_16, 0);  // #41
        const Tensor<bfloat16_t, Position::UB> view_4 = ub_q_bf16;  // #42
        const GMTensor<bfloat16_t> view_5 = q[(((b * ((T * H) * 128)) + (tok_start * (H * 128))) + (h * 128))];  // #45
        gm_to_ub_pad(view_4, view_5, valid_rows, 256, sub_16, 0);  // #46
        const GMTensor<bfloat16_t> view_6 = k_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (tok_start * 128))];  // #49
        ev_mte2_mte3_ready_d1_1.set();  // #230
        ev_mte2_mte3_ready_d1_1.wait();  // #231
        ev_v_mte3_ready_1.set();  // #252
        ev_v_mte3_ready_1.wait();  // #253
        ub_to_gm_pad(view_6, ub_k_bf16, 64, 256, 0, 0);  // #50
        const GMTensor<bfloat16_t> view_7 = v_pad[(((b * ((H * T_PAD_KV) * 128)) + (h * (T_PAD_KV * 128))) + (tok_start * 128))];  // #53
        ub_to_gm_pad(view_7, ub_v_bf16, 64, 256, 0, 0);  // #54
        ev_mte2_v_ready_d1_1.set();  // #244
        ev_mte2_v_ready_d1_1.wait();  // #245
        ev_mte3_v_valid_1.set();  // #254
        ev_mte3_v_valid_1.wait();  // #255
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #55
        vconv_bf162f32(ub_f32.ptr(), ub_k_bf16.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)4);  // #55
        const Tensor<float, Position::UB> view_8 = ub_f32;  // #56
        const Tensor<float, Position::UB> view_9 = ub_f32;  // #57
        const Tensor<float, Position::UB> view_10 = ub_f32[4096];  // #58
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #59
        vadd(view_8.ptr(), view_9.ptr(), view_10.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #59
        const Tensor<float, Position::UB> view_11 = ub_f32;  // #60
        const Tensor<float, Position::UB> view_12 = ub_f32;  // #61
        const Tensor<float, Position::UB> view_13 = ub_f32[2048];  // #62
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #63
        vadd(view_11.ptr(), view_12.ptr(), view_13.ptr(), (uint8_t)(32), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #63
        const Tensor<float, Position::UB> view_14 = ub_f32;  // #64
        const Tensor<float, Position::UB> view_15 = ub_f32;  // #65
        const Tensor<float, Position::UB> view_16 = ub_f32[1024];  // #66
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #67
        vadd(view_14.ptr(), view_15.ptr(), view_16.ptr(), (uint8_t)(16), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #67
        const Tensor<float, Position::UB> view_17 = ub_f32;  // #68
        const Tensor<float, Position::UB> view_18 = ub_f32;  // #69
        const Tensor<float, Position::UB> view_19 = ub_f32[512];  // #70
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #71
        vadd(view_17.ptr(), view_18.ptr(), view_19.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #71
        const Tensor<float, Position::UB> view_20 = ub_f32;  // #72
        const Tensor<float, Position::UB> view_21 = ub_f32;  // #73
        const Tensor<float, Position::UB> view_22 = ub_f32[256];  // #74
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #75
        vadd(view_20.ptr(), view_21.ptr(), view_22.ptr(), (uint8_t)(4), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #75
        const Tensor<float, Position::UB> view_23 = ub_f32;  // #76
        const Tensor<float, Position::UB> view_24 = ub_f32;  // #77
        const Tensor<float, Position::UB> view_25 = ub_f32[128];  // #78
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #79
        vadd(view_23.ptr(), view_24.ptr(), view_25.ptr(), (uint8_t)(2), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #79
        const Tensor<float, Position::UB> view_26 = ub_f32;  // #80
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #81
        vmuls(ub_kc_f32.ptr(), view_26.ptr(), (float)(inv_len_1), (uint8_t)(2), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #81
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #82
        vconv_f322bf16r(ub_kc_bf16.ptr(), ub_kc_f32.ptr(), (uint8_t)(2), (uint16_t)1, (uint16_t)1, (uint16_t)4, (uint16_t)8);  // #82
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #83
        vconv_bf162f32(ub_kc_f32.ptr(), ub_kc_bf16.ptr(), (uint8_t)(2), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)4);  // #83
        const GMTensor<bfloat16_t> view_27 = kc_padded[(((b * ((H * NPAD) * 128)) + (h * (NPAD * 128))) + (blk * 128))];  // #86
        ev_v_mte3_ready_1.set();  // #256
        ev_v_mte3_ready_1.wait();  // #257
        ub_to_gm_pad(view_27, ub_kc_bf16, 1, 256, 0, 0);  // #87
        const GMTensor<float> view_28 = kc_pad_f32[(((b * ((H * NPAD) * 128)) + (h * (NPAD * 128))) + (blk * 128))];  // #90
        ub_to_gm_pad(view_28, ub_kc_f32, 1, 512, 0, 0);  // #91
        ev_mte3_v_valid_1.set();  // #258
        ev_mte3_v_valid_1.wait();  // #259
        vconv_bf162f32(ub_f32.ptr(), ub_v_bf16.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)4);  // #92
        const Tensor<float, Position::UB> view_29 = ub_f32;  // #93
        const Tensor<float, Position::UB> view_30 = ub_f32;  // #94
        const Tensor<float, Position::UB> view_31 = ub_f32[4096];  // #95
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #96
        vadd(view_29.ptr(), view_30.ptr(), view_31.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #96
        const Tensor<float, Position::UB> view_32 = ub_f32;  // #97
        const Tensor<float, Position::UB> view_33 = ub_f32;  // #98
        const Tensor<float, Position::UB> view_34 = ub_f32[2048];  // #99
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #100
        vadd(view_32.ptr(), view_33.ptr(), view_34.ptr(), (uint8_t)(32), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #100
        const Tensor<float, Position::UB> view_35 = ub_f32;  // #101
        const Tensor<float, Position::UB> view_36 = ub_f32;  // #102
        const Tensor<float, Position::UB> view_37 = ub_f32[1024];  // #103
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #104
        vadd(view_35.ptr(), view_36.ptr(), view_37.ptr(), (uint8_t)(16), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #104
        const Tensor<float, Position::UB> view_38 = ub_f32;  // #105
        const Tensor<float, Position::UB> view_39 = ub_f32;  // #106
        const Tensor<float, Position::UB> view_40 = ub_f32[512];  // #107
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #108
        vadd(view_38.ptr(), view_39.ptr(), view_40.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #108
        const Tensor<float, Position::UB> view_41 = ub_f32;  // #109
        const Tensor<float, Position::UB> view_42 = ub_f32;  // #110
        const Tensor<float, Position::UB> view_43 = ub_f32[256];  // #111
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #112
        vadd(view_41.ptr(), view_42.ptr(), view_43.ptr(), (uint8_t)(4), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #112
        const Tensor<float, Position::UB> view_44 = ub_f32;  // #113
        const Tensor<float, Position::UB> view_45 = ub_f32;  // #114
        const Tensor<float, Position::UB> view_46 = ub_f32[128];  // #115
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #116
        vadd(view_44.ptr(), view_45.ptr(), view_46.ptr(), (uint8_t)(2), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #116
        const Tensor<float, Position::UB> view_47 = ub_f32;  // #117
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #118
        vconv_f322bf16r(ub_vc_bf16.ptr(), view_47.ptr(), (uint8_t)(2), (uint16_t)1, (uint16_t)1, (uint16_t)4, (uint16_t)8);  // #118
        const GMTensor<bfloat16_t> view_48 = vc_padded[(((b * ((H * NPAD) * 128)) + (h * (NPAD * 128))) + (blk * 128))];  // #121
        ev_v_mte3_ready_1.set();  // #260
        ev_v_mte3_ready_1.wait();  // #261
        ub_to_gm_pad(view_48, ub_vc_bf16, 1, 256, 0, 0);  // #122
        ev_mte3_v_valid_1.set();  // #262
        ev_mte3_v_valid_1.wait();  // #263
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #123
        vconv_bf162f32(ub_f32.ptr(), ub_q_bf16.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)4);  // #123
        const Tensor<float, Position::UB> view_49 = ub_f32;  // #124
        const Tensor<float, Position::UB> view_50 = ub_f32;  // #125
        const Tensor<float, Position::UB> view_51 = ub_f32[4096];  // #126
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #127
        vadd(view_49.ptr(), view_50.ptr(), view_51.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #127
        const Tensor<float, Position::UB> view_52 = ub_f32;  // #128
        const Tensor<float, Position::UB> view_53 = ub_f32;  // #129
        const Tensor<float, Position::UB> view_54 = ub_f32[2048];  // #130
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #131
        vadd(view_52.ptr(), view_53.ptr(), view_54.ptr(), (uint8_t)(32), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #131
        const Tensor<float, Position::UB> view_55 = ub_f32;  // #132
        const Tensor<float, Position::UB> view_56 = ub_f32;  // #133
        const Tensor<float, Position::UB> view_57 = ub_f32[1024];  // #134
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #135
        vadd(view_55.ptr(), view_56.ptr(), view_57.ptr(), (uint8_t)(16), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #135
        const Tensor<float, Position::UB> view_58 = ub_f32;  // #136
        const Tensor<float, Position::UB> view_59 = ub_f32;  // #137
        const Tensor<float, Position::UB> view_60 = ub_f32[512];  // #138
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #139
        vadd(view_58.ptr(), view_59.ptr(), view_60.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #139
        const Tensor<float, Position::UB> view_61 = ub_f32;  // #140
        const Tensor<float, Position::UB> view_62 = ub_f32;  // #141
        const Tensor<float, Position::UB> view_63 = ub_f32[256];  // #142
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #143
        vadd(view_61.ptr(), view_62.ptr(), view_63.ptr(), (uint8_t)(4), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #143
        const Tensor<float, Position::UB> view_64 = ub_f32;  // #144
        const Tensor<float, Position::UB> view_65 = ub_f32;  // #145
        const Tensor<float, Position::UB> view_66 = ub_f32[128];  // #146
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #147
        vadd(view_64.ptr(), view_65.ptr(), view_66.ptr(), (uint8_t)(2), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #147
        const Tensor<float, Position::UB> view_67 = ub_f32;  // #148
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #149
        vmuls(ub_qbar_f32.ptr(), view_67.ptr(), (float)(inv_len_1), (uint8_t)(2), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #149
        const GMTensor<float> view_68 = q_bar_f32[(((b * ((H * NT) * 128)) + (h * (NT * 128))) + (blk * 128))];  // #152
        ev_v_mte3_ready_1.set();  // #264
        ev_v_mte3_ready_1.wait();  // #265
        ub_to_gm_pad(view_68, ub_qbar_f32, 1, 512, 0, 0);  // #153
        ev_mte3_mte2_valid_d1_1.set();  // #232
        ev_v_mte2_valid_d1_1.set();  // #246
        ev_mte3_v_valid_1.set();  // #269
    }
    ev_mte2_mte3_ready_d0_1.set();  // #233
    ev_mte2_mte3_ready_d0_1.wait();  // #234
    ev_mte2_v_ready_d0_1.set();  // #247
    ev_mte2_v_ready_d0_1.wait();  // #248
    const int32_t pad_blocks = NPAD - NT;  // #157
    if (pad_blocks > 0) {  // #158 #180
        const int32_t total_bh = B * H;  // #159
        ev_mte3_v_valid_1.wait();  // #266
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #160
        vector_dup(ub_f32.ptr(), (float)(0.0f), (uint8_t)(128), (uint16_t)1, 1, (uint8_t)8, 0);  // #160
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #161
        vconv_f322bf16r(ub_k_bf16.ptr(), ub_f32.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)4, (uint16_t)8);  // #161
        ev_v_mte3_ready_1.set();  // #267
        ev_v_mte3_ready_1.wait();  // #268
        for (int32_t bh = vid; bh < total_bh; bh += vnum) {  // #206
            const int32_t idiv_17 = bh / H;  // #287
            const int32_t idiv_18 = bh % H;  // #288
            const bool idiv_19 = idiv_18 != 0;  // #289
            const bool idiv_20 = bh < 0;  // #290
            const bool idiv_22 = idiv_20 != false;  // #292
            const bool idiv_23 = idiv_19 && idiv_22;  // #293
            const int32_t idiv_24 = (idiv_23) ? (1) : (0);  // #294
            const int32_t pb = idiv_17 - idiv_24;  // #163
            const int32_t ph = ascrip::FloorMod<int32_t>(bh, H);  // #164
            const int32_t sub_10 = NPAD - NT;  // #165
            const GMTensor<bfloat16_t> view_69 = kc_padded[(((pb * ((H * NPAD) * 128)) + (ph * (NPAD * 128))) + (NT * 128))];  // #166
            const Tensor<bfloat16_t, Position::UB> view_70 = ub_k_bf16;  // #167
            ub_to_gm_pad(view_69, view_70, sub_10, 256, 0, 0);  // #168
            const int32_t sub_11 = NPAD - NT;  // #169
            const GMTensor<bfloat16_t> view_71 = vc_padded[(((pb * ((H * NPAD) * 128)) + (ph * (NPAD * 128))) + (NT * 128))];  // #170
            const Tensor<bfloat16_t, Position::UB> view_72 = ub_k_bf16;  // #171
            ub_to_gm_pad(view_71, view_72, sub_11, 256, 0, 0);  // #172
            const int32_t sub_12 = NPAD - NT;  // #173
            const GMTensor<float> view_73 = kc_pad_f32[(((pb * ((H * NPAD) * 128)) + (ph * (NPAD * 128))) + (NT * 128))];  // #174
            const Tensor<float, Position::UB> view_74 = ub_f32;  // #175
            ub_to_gm_pad(view_73, view_74, sub_12, 512, 0, 0);  // #176
        }
        ev_mte3_v_valid_1.set();  // #270
    }
    if (vid == 0) {  // #181 #199
        const int32_t g_last_start = (NG - 1) * 64;  // #182 #183
        for (int32_t c = 0; c < 64; c += 1) {  // #207
            const int32_t rem = Min(64, Max(T - ((g_last_start + c) * 64), 0));  // #184 #185 #186 #187 #188
            ub_lens.store(c, (float)((float)(rem)));  // #189 #191
        }
        SetFlag<PIPE_S, PIPE_MTE3>(0);  // #194
        WaitFlag<PIPE_S, PIPE_MTE3>(0);  // #195
        ub_to_gm_pad(last_group_lens, ub_lens, 1, 256, 0, 0);  // #196
        SetFlag<PIPE_MTE3, PIPE_S>(0);  // #197
        WaitFlag<PIPE_MTE3, PIPE_S>(0);  // #198
    }
    ev_mte3_mte2_valid_d0_1.set();  // #228
    ev_v_mte2_valid_d0_1.set();  // #240
    return;  // #200
}
