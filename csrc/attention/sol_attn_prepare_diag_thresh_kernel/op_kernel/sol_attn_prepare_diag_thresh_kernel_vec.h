#pragma once
#include "../../common/sol_attn/tensorutils_cce.h"
using namespace ascrip;

__aicore__ inline void sol_attn_prepare_diag_thresh_kernel_vec(GM_ADDR q_bar_f32_, GM_ADDR kc_pad_f32_, GM_ADDR thresh_bh_, int32_t B, int32_t H, int32_t NT, int32_t NG, int32_t NPAD, float scale_log2, float tau, float inv_nt, GM_ADDR workspace)
{
    GMTensor<float> q_bar_f32((__gm__ float*)q_bar_f32_);
    GMTensor<float> kc_pad_f32((__gm__ float*)kc_pad_f32_);
    GMTensor<float> thresh_bh((__gm__ float*)thresh_bh_);
    SetAtomicNone();
    ResetMask();
    SetMaskNorm();
    SEvent<PIPE_MTE2, PIPE_V, 0, 0> ev_mte2_v_ready_1;  // guards ub_tile #151
    SEvent<PIPE_V, PIPE_MTE2, 1, 0> ev_v_mte2_valid_1;  // guards ub_tile #152
    SEvent<PIPE_V, PIPE_MTE3, 0, 0> ev_v_mte3_ready_1;  // guards ub_thresh #163
    SEvent<PIPE_MTE3, PIPE_V, 1, 0> ev_mte3_v_valid_1;  // guards ub_thresh #164
    const Tensor<float, Position::UB> ub_tile((uint64_t)(0));  // #1
    const Tensor<float, Position::UB> ub_sq((uint64_t)(32768));  // #2
    const Tensor<float, Position::UB> ub_sum1((uint64_t)(65536));  // #3
    const Tensor<float, Position::UB> ub_sum2((uint64_t)(66048));  // #4
    const Tensor<float, Position::UB> ub_mean((uint64_t)(66560));  // #5
    const Tensor<float, Position::UB> ub_var((uint64_t)(67072));  // #6
    const Tensor<float, Position::UB> ub_mean_l((uint64_t)(67584));  // #7
    const Tensor<float, Position::UB> ub_mean_r((uint64_t)(67840));  // #8
    const Tensor<float, Position::UB> ub_var_l((uint64_t)(68096));  // #9
    const Tensor<float, Position::UB> ub_var_r((uint64_t)(68352));  // #10
    const Tensor<float, Position::UB> ub_acc_l((uint64_t)(68608));  // #11
    const Tensor<float, Position::UB> ub_acc_r((uint64_t)(84992));  // #12
    const Tensor<float, Position::UB> ub_mu((uint64_t)(101376));  // #13
    const Tensor<float, Position::UB> ub_vproj((uint64_t)(101632));  // #14
    const Tensor<float, Position::UB> ub_std((uint64_t)(101888));  // #15
    const Tensor<float, Position::UB> ub_thresh((uint64_t)(102144));  // #16
    const int32_t vid = GetVecIdx();  // #17
    const int32_t vnum = GetVecNum();  // #18
    const int32_t total_bh = B * H;  // #19
    const float scale_log2_sq = scale_log2 * scale_log2;  // #20
    for (int32_t bh = vid; bh < total_bh; bh += vnum) {  // #145
        const int32_t idiv_1 = bh / H;  // #169
        const int32_t idiv_2 = bh % H;  // #170
        const bool idiv_3 = idiv_2 != 0;  // #171
        const bool idiv_4 = bh < 0;  // #172
        const bool idiv_6 = idiv_4 != false;  // #174
        const bool idiv_7 = idiv_3 && idiv_6;  // #175
        const int32_t idiv_8 = (idiv_7) ? (1) : (0);  // #176
        const int32_t b = idiv_1 - idiv_8;  // #21
        const int32_t h = ascrip::FloorMod<int32_t>(bh, H);  // #22
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #23
        vector_dup(ub_sum1.ptr(), (float)(0.0f), (uint8_t)(2), (uint16_t)1, 1, (uint8_t)8, 0);  // #23
        vector_dup(ub_sum2.ptr(), (float)(0.0f), (uint8_t)(2), (uint16_t)1, 1, (uint8_t)8, 0);  // #24
        for (int32_t g = 0; g < NG; g += 1) {  // #143
            const int32_t g_start = g * 64;  // #26
            const GMTensor<float> view = kc_pad_f32[(((b * ((H * NPAD) * 128)) + (h * (NPAD * 128))) + (g_start * 128))];  // #29
            ev_v_mte2_valid_1.wait();  // #153
            gm_to_ub_pad(ub_tile, view, 64, 512, 0, 0);  // #30
            ev_mte2_v_ready_1.set();  // #154
            ev_mte2_v_ready_1.wait();  // #155
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #31
            vmul(ub_sq.ptr(), ub_tile.ptr(), ub_tile.ptr(), (uint8_t)(128), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #31
            const Tensor<float, Position::UB> view_1 = ub_tile;  // #32
            const Tensor<float, Position::UB> view_2 = ub_tile;  // #33
            const Tensor<float, Position::UB> view_3 = ub_tile[4096];  // #34
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #35
            vadd(view_1.ptr(), view_2.ptr(), view_3.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #35
            const Tensor<float, Position::UB> view_4 = ub_tile;  // #36
            const Tensor<float, Position::UB> view_5 = ub_tile;  // #37
            const Tensor<float, Position::UB> view_6 = ub_tile[2048];  // #38
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #39
            vadd(view_4.ptr(), view_5.ptr(), view_6.ptr(), (uint8_t)(32), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #39
            const Tensor<float, Position::UB> view_7 = ub_tile;  // #40
            const Tensor<float, Position::UB> view_8 = ub_tile;  // #41
            const Tensor<float, Position::UB> view_9 = ub_tile[1024];  // #42
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #43
            vadd(view_7.ptr(), view_8.ptr(), view_9.ptr(), (uint8_t)(16), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #43
            const Tensor<float, Position::UB> view_10 = ub_tile;  // #44
            const Tensor<float, Position::UB> view_11 = ub_tile;  // #45
            const Tensor<float, Position::UB> view_12 = ub_tile[512];  // #46
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #47
            vadd(view_10.ptr(), view_11.ptr(), view_12.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #47
            const Tensor<float, Position::UB> view_13 = ub_tile;  // #48
            const Tensor<float, Position::UB> view_14 = ub_tile;  // #49
            const Tensor<float, Position::UB> view_15 = ub_tile[256];  // #50
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #51
            vadd(view_13.ptr(), view_14.ptr(), view_15.ptr(), (uint8_t)(4), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #51
            const Tensor<float, Position::UB> view_16 = ub_tile;  // #52
            const Tensor<float, Position::UB> view_17 = ub_tile;  // #53
            const Tensor<float, Position::UB> view_18 = ub_tile[128];  // #54
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #55
            vadd(view_16.ptr(), view_17.ptr(), view_18.ptr(), (uint8_t)(2), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #55
            const Tensor<float, Position::UB> view_19 = ub_tile;  // #56
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #57
            vadd(ub_sum1.ptr(), ub_sum1.ptr(), view_19.ptr(), (uint8_t)(2), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #57
            const Tensor<float, Position::UB> view_20 = ub_sq;  // #58
            const Tensor<float, Position::UB> view_21 = ub_sq;  // #59
            const Tensor<float, Position::UB> view_22 = ub_sq[4096];  // #60
            vadd(view_20.ptr(), view_21.ptr(), view_22.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #61
            const Tensor<float, Position::UB> view_23 = ub_sq;  // #62
            const Tensor<float, Position::UB> view_24 = ub_sq;  // #63
            const Tensor<float, Position::UB> view_25 = ub_sq[2048];  // #64
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #65
            vadd(view_23.ptr(), view_24.ptr(), view_25.ptr(), (uint8_t)(32), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #65
            const Tensor<float, Position::UB> view_26 = ub_sq;  // #66
            const Tensor<float, Position::UB> view_27 = ub_sq;  // #67
            const Tensor<float, Position::UB> view_28 = ub_sq[1024];  // #68
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #69
            vadd(view_26.ptr(), view_27.ptr(), view_28.ptr(), (uint8_t)(16), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #69
            const Tensor<float, Position::UB> view_29 = ub_sq;  // #70
            const Tensor<float, Position::UB> view_30 = ub_sq;  // #71
            const Tensor<float, Position::UB> view_31 = ub_sq[512];  // #72
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #73
            vadd(view_29.ptr(), view_30.ptr(), view_31.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #73
            const Tensor<float, Position::UB> view_32 = ub_sq;  // #74
            const Tensor<float, Position::UB> view_33 = ub_sq;  // #75
            const Tensor<float, Position::UB> view_34 = ub_sq[256];  // #76
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #77
            vadd(view_32.ptr(), view_33.ptr(), view_34.ptr(), (uint8_t)(4), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #77
            const Tensor<float, Position::UB> view_35 = ub_sq;  // #78
            const Tensor<float, Position::UB> view_36 = ub_sq;  // #79
            const Tensor<float, Position::UB> view_37 = ub_sq[128];  // #80
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #81
            vadd(view_35.ptr(), view_36.ptr(), view_37.ptr(), (uint8_t)(2), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #81
            const Tensor<float, Position::UB> view_38 = ub_sq;  // #82
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #83
            vadd(ub_sum2.ptr(), ub_sum2.ptr(), view_38.ptr(), (uint8_t)(2), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #83
            ev_v_mte2_valid_1.set();  // #161
        }
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #87
        vmuls(ub_mean.ptr(), ub_sum1.ptr(), (float)(inv_nt), (uint8_t)(2), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #87
        vmuls(ub_var.ptr(), ub_sum2.ptr(), (float)(inv_nt), (uint8_t)(2), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #88
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #89
        vmul(ub_sum1.ptr(), ub_mean.ptr(), ub_mean.ptr(), (uint8_t)(2), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #89
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #90
        vsub(ub_var.ptr(), ub_var.ptr(), ub_sum1.ptr(), (uint8_t)(2), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #90
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #91
        vmaxs(ub_var.ptr(), ub_var.ptr(), (float)(0.0f), (uint8_t)(2), (uint16_t)1, (uint16_t)1, (uint8_t)8, (uint8_t)8, false, false);  // #91
        vmuls(ub_mean.ptr(), ub_mean.ptr(), (float)(scale_log2), (uint8_t)(2), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #92
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #93
        vmuls(ub_var.ptr(), ub_var.ptr(), (float)(scale_log2_sq), (uint8_t)(2), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #93
        const Tensor<float, Position::UB> view_39 = ub_mean;  // #94
        ub_to_ub(ub_mean_l, view_39, 1, 8, 8, 0);  // #95
        const Tensor<float, Position::UB> view_40 = ub_mean[64];  // #96
        ub_to_ub(ub_mean_r, view_40, 1, 8, 8, 0);  // #97
        const Tensor<float, Position::UB> view_41 = ub_var;  // #98
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #99
        ub_to_ub(ub_var_l, view_41, 1, 8, 8, 0);  // #99
        const Tensor<float, Position::UB> view_42 = ub_var[64];  // #100
        ub_to_ub(ub_var_r, view_42, 1, 8, 8, 0);  // #101
        for (int32_t g_1 = 0; g_1 < NG; g_1 += 1) {  // #144
            const int32_t g_start_1 = g_1 * 64;  // #103
            const int32_t valid_q = Min(64, NT - g_start_1);  // #104 #105
            ev_v_mte2_valid_1.wait();  // #156
            if (valid_q < 64) {  // #106 #108
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #107
                vector_dup(ub_tile.ptr(), (float)(0.0f), (uint8_t)(128), (uint16_t)1, 1, (uint8_t)8, 0);  // #107
            }
            const Tensor<float, Position::UB> view_43 = ub_tile;  // #109
            const GMTensor<float> view_44 = q_bar_f32[(((b * ((H * NT) * 128)) + (h * (NT * 128))) + (g_start_1 * 128))];  // #112
            ev_v_mte2_valid_1.set();  // #157
            ev_v_mte2_valid_1.wait();  // #158
            gm_to_ub_pad(view_43, view_44, valid_q, 512, 0, 0);  // #113
            const Tensor<float, Position::UB> view_45 = ub_tile;  // #114
            ev_mte2_v_ready_1.set();  // #159
            ev_mte2_v_ready_1.wait();  // #160
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #115
            vmul(ub_acc_l.ptr(), view_45.ptr(), ub_mean_l.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)16, (uint8_t)0);  // #115
            const Tensor<float, Position::UB> view_46 = ub_tile[64];  // #116
            vmul(ub_acc_r.ptr(), view_46.ptr(), ub_mean_r.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)16, (uint8_t)0);  // #117
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #118
            vadd(ub_acc_l.ptr(), ub_acc_l.ptr(), ub_acc_r.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #118
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #119
            vcadd(ub_mu.ptr(), ub_acc_l.ptr(), (uint8_t)(64), 1, 1, 8, 0);  // #119
            vmul(ub_sq.ptr(), ub_tile.ptr(), ub_tile.ptr(), (uint8_t)(128), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #120
            const Tensor<float, Position::UB> view_47 = ub_sq;  // #121
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #122
            vmul(ub_acc_l.ptr(), view_47.ptr(), ub_var_l.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)16, (uint8_t)0);  // #122
            const Tensor<float, Position::UB> view_48 = ub_sq[64];  // #123
            vmul(ub_acc_r.ptr(), view_48.ptr(), ub_var_r.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)16, (uint8_t)0);  // #124
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #125
            vadd(ub_acc_l.ptr(), ub_acc_l.ptr(), ub_acc_r.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)8, (uint8_t)8, (uint8_t)8);  // #125
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #126
            vcadd(ub_vproj.ptr(), ub_acc_l.ptr(), (uint8_t)(64), 1, 1, 8, 0);  // #126
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #127
            vmaxs(ub_vproj.ptr(), ub_vproj.ptr(), (float)(0.0f), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint8_t)0, (uint8_t)0, false, false);  // #127
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #128
            vadds(ub_vproj.ptr(), ub_vproj.ptr(), (float)(1e-06f), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #128
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #129
            vsqrt(ub_std.ptr(), ub_vproj.ptr(), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint8_t)0, (uint8_t)0);  // #129
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #130
            vmuls(ub_std.ptr(), ub_std.ptr(), (float)(tau), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #130
            ev_mte3_v_valid_1.wait();  // #165
            PipeBarrier<PIPE_V>();  // c220: V-V hazard #131
            vadd(ub_thresh.ptr(), ub_mu.ptr(), ub_std.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #131
            const GMTensor<float> view_49 = thresh_bh[(((b * (H * NT)) + (h * NT)) + g_start_1)];  // #134
            const Tensor<float, Position::UB> view_50 = ub_thresh;  // #135
            const int32_t sub_5 = 64 - valid_q;  // #149
            const int32_t idiv_9 = sub_5 / 8;  // #177
            const int32_t idiv_10 = sub_5 % 8;  // #178
            const bool idiv_11 = idiv_10 != 0;  // #179
            const bool idiv_12 = sub_5 < 0;  // #180
            const bool idiv_14 = idiv_12 != false;  // #182
            const bool idiv_15 = idiv_11 && idiv_14;  // #183
            const int32_t idiv_16 = (idiv_15) ? (1) : (0);  // #184
            const int32_t div_1 = idiv_9 - idiv_16;  // #150
            ev_v_mte3_ready_1.set();  // #166
            ev_v_mte3_ready_1.wait();  // #167
            ub_to_gm_pad(view_49, view_50, 1, valid_q * 4, div_1, 0);  // #146 #136
            ev_v_mte2_valid_1.set();  // #162
            ev_mte3_v_valid_1.set();  // #168
        }
    }
    return;  // #142
}
