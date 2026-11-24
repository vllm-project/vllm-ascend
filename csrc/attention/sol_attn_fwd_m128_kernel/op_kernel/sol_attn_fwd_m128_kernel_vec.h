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
    SEvent<PIPE_MTE2, PIPE_V, 0, 0> ev_mte2_v_ready_d0_1;  // guards accum_pair #1813
    SEvent<PIPE_V, PIPE_MTE2, 1, 0> ev_v_mte2_valid_d0_1;  // guards accum_pair #1814
    SEvent<PIPE_MTE2, PIPE_V, 0, 1> ev_mte2_v_ready_d1_1;  // guards accum_pair #1815
    SEvent<PIPE_V, PIPE_MTE2, 1, 1> ev_v_mte2_valid_d1_1;  // guards accum_pair #1816
    SEvent<PIPE_V, PIPE_MTE3, 0, 0> ev_v_mte3_ready_1;  // guards probability_tiles, ub_rmax_s #1827
    SEvent<PIPE_MTE3, PIPE_V, 1, 0> ev_mte3_v_valid_1;  // guards probability_tiles, ub_rmax_s #1828
    const float scale_natural = scale_log2 * 0.6931471805599453f;  // #1833
    const GMTensor<float> score_ws((__gm__ float*)(workspace + (0)));  // #1834
    const GMTensor<bfloat16_t> p_ws((__gm__ bfloat16_t*)(workspace + (18874368)));  // #1835
    const GMTensor<float> pv_ws((__gm__ float*)(workspace + (28311552)));  // #1836
    const GMTensor<float> score_pairs = score_ws;  // #1837
    const GMTensor<bfloat16_t> p_pairs = p_ws;  // #1838
    const Buff<float, Position::UB, 3> score_tiles((uint64_t)(0), (uint64_t)(32768));  // #1846
    const Buff<bfloat16_t, Position::UB, 1> probability_tiles((uint64_t)(98304), (uint64_t)(16384));  // #1847
    const Buff<float, Position::UB, 2> accum_pair((uint64_t)(114688), (uint64_t)(32768));  // #1848
    const Tensor<bfloat16_t, Position::UB> ub_out = probability_tiles.slot[0];  // #1849
    const Tensor<float, Position::UB> ub_len_last((uint64_t)(180224));  // #1850
    const Tensor<float, Position::UB> ub_max_s((uint64_t)(180480));  // #1851
    const Tensor<float, Position::UB> ub_bmax_s((uint64_t)(180736));  // #1852
    const Tensor<float, Position::UB> ub_old_max_s((uint64_t)(180992));  // #1853
    const Tensor<float, Position::UB> ub_rmax_s((uint64_t)(181248));  // #1854
    const Tensor<float, Position::UB> ub_sum_s((uint64_t)(181504));  // #1855
    const Tensor<float, Position::UB> ub_bsum_s((uint64_t)(181760));  // #1856
    const Tensor<float, Position::UB> ub_rsum_s((uint64_t)(182016));  // #1857
    const Tensor<float, Position::UB> ub_zero_s((uint64_t)(182272));  // #1858
    const Buff<float, Position::UB, 4> expdiff_buf((uint64_t)(182528), (uint64_t)(256));  // #1859
    const Tensor<float, Position::UB> ub_max((uint64_t)(183552));  // #1860
    const Tensor<float, Position::UB> ub_reduce_pair((uint64_t)(185600));  // #1861
    const Tensor<float, Position::UB> ub_reduce = ub_reduce_pair;  // #1862
    const Tensor<float, Position::UB> ub_expdiff((uint64_t)(189696));  // #1863
    const Tensor<float, Position::UB> ub_rowsum((uint64_t)(191744));  // #1864
    VEC_READY<PIPE_MTE2>(0);  // #1865
    VEC_READY<PIPE_MTE2>(0);  // #1865
    // sync.mutex vc id=1: the producer side; its 2 tokens are drained before the return  // #1866
    VEC_READY<PIPE_MTE2>(2);  // #1867
    VEC_READY<PIPE_MTE2>(2);  // #1867
    SEvent<PIPE_MTE2, PIPE_V, 0, 0> lengths_ready;  // #38
    gm_to_ub_pad(ub_len_last, last_group_lens, 1, 256, 0, 0);  // #47
    lengths_ready.set();  // #48
    lengths_ready.wait();  // #49
    const int32_t cube_idx = GetCubeIdx();  // #1868
    const int32_t cube_count = GetCubeNum();  // #1869
    const int32_t core_step = (int32_t)(cube_count);  // #1870
    const int32_t sb = GetSubBlockIdx();  // #53
    const int32_t sb_row = sb * 64;  // #54
    const int32_t bh_total = B * H;  // #1871
    const int32_t total_tasks = bh_total * NT_Q;  // #1872
    int32_t l0ab_cnt_17_state = 0;  // #1877
    int32_t l0c_cnt_12_state = 0;  // #1878
    int32_t l1k_cnt_10_state = 0;  // #1879
    int32_t l1pv_cnt_9_state = 0;  // #1880
    for (int32_t linear_id = cube_idx; linear_id < total_tasks; linear_id += core_step) {  // #3100
        const int32_t l0ab_cnt_carry = (int32_t)(l0ab_cnt_17_state);  // #1881
        const int32_t l0c_cnt_carry = (int32_t)(l0c_cnt_12_state);  // #1882
        const int32_t l1k_cnt_carry = (int32_t)(l1k_cnt_10_state);  // #1883
        const int32_t l1pv_cnt_carry = (int32_t)(l1pv_cnt_9_state);  // #1884
        const int32_t idiv_33 = linear_id / NT_Q;  // #3134
        const int32_t idiv_34 = linear_id % NT_Q;  // #3135
        const bool idiv_35 = idiv_34 != 0;  // #3136
        const bool idiv_36 = linear_id < 0;  // #3137
        const bool idiv_38 = idiv_36 != false;  // #3139
        const bool idiv_39 = idiv_35 && idiv_38;  // #3140
        const int32_t idiv_40 = (idiv_39) ? (1) : (0);  // #3141
        const int32_t bh = idiv_33 - idiv_40;  // #1885
        const int32_t q_idx = ascrip::FloorMod<int32_t>(linear_id, NT_Q);  // #1886
        const int32_t idiv_41 = bh / H;  // #3142
        const int32_t idiv_42 = bh % H;  // #3143
        const bool idiv_43 = idiv_42 != 0;  // #3144
        const bool idiv_44 = bh < 0;  // #3145
        const bool idiv_46 = idiv_44 != false;  // #3147
        const bool idiv_47 = idiv_43 && idiv_46;  // #3148
        const int32_t idiv_48 = (idiv_47) ? (1) : (0);  // #3149
        const int32_t b = idiv_41 - idiv_48;  // #1887
        const int32_t h = ascrip::FloorMod<int32_t>(bh, H);  // #1888
        const int32_t q_start = q_idx * 128;  // #1889
        const int32_t valid_m = Min(128, T - q_start);  // #1890 #1891
        const int32_t local_valid_m = Min(64, Max(valid_m - sb_row, 0));  // #68 #69 #70
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #84
        vector_dup(ub_zero_s.ptr(), (float)(0.0f), (uint8_t)(1), (uint16_t)1, 1, (uint8_t)0, 0);  // #84
        vector_dup(ub_max_s.ptr(), (float)(-1e+30f), (uint8_t)(1), (uint16_t)1, 1, (uint8_t)0, 0);  // #85
        vector_dup(ub_sum_s.ptr(), (float)(0.0f), (uint8_t)(1), (uint16_t)1, 1, (uint8_t)0, 0);  // #86
        ev_mte3_v_valid_1.wait();  // #1829
        vector_dup(ub_rmax_s.ptr(), (float)(-1e+30f), (uint8_t)(1), (uint16_t)1, 1, (uint8_t)0, 0);  // #87
        vector_dup(ub_rsum_s.ptr(), (float)(0.0f), (uint8_t)(1), (uint16_t)1, 1, (uint8_t)0, 0);  // #88
        const Tensor<float, Position::UB> tile = accum_pair.slot[0];  // #1893
        ev_v_mte2_valid_d0_1.wait();  // #1817
        vector_dup(tile.ptr(), (float)(0.0f), (uint8_t)(128), (uint16_t)1, 1, (uint8_t)8, 0);  // #90
        const int32_t exact_total = flat_exact_cnt.load(((b * (H * NT_Q_PAD8)) + (h * NT_Q_PAD8)) + q_idx);  // #1896 #1897 #1898 #1899 #1900 #1901
        const int32_t idiv_49 = NG / 6;  // #3150
        const int32_t idiv_50 = NG % 6;  // #3151
        const bool idiv_51 = idiv_50 != 0;  // #3152
        const bool idiv_55 = idiv_51 && true;  // #3156
        const int32_t idiv_56 = (idiv_55) ? (1) : (0);  // #3157
        const int32_t approx_beats_1 = idiv_49 + idiv_56;  // #1902
        const int32_t idiv_57 = exact_total / 6;  // #3158
        const int32_t idiv_58 = exact_total % 6;  // #3159
        const bool idiv_59 = idiv_58 != 0;  // #3160
        const bool idiv_60 = exact_total < 0;  // #3161
        const bool idiv_62 = idiv_60 == false;  // #3163
        const bool idiv_63 = idiv_59 && idiv_62;  // #3164
        const int32_t idiv_64 = (idiv_63) ? (1) : (0);  // #3165
        const int32_t ceil_div = idiv_57 + idiv_64;  // #1903
        const int32_t total_beats = approx_beats_1 + ceil_div;  // #1904
        SetFlag<PIPE_V, PIPE_MTE2>(4);  // #103
        SetFlag<PIPE_V, PIPE_MTE2>(5);  // #104
        SetFlag<PIPE_V, PIPE_MTE2>(6);  // #105
        SetFlag<PIPE_MTE3, PIPE_V>(4);  // #106
        const int32_t add_3 = total_beats + 3;  // #1906
        int32_t accum_step_3_state = 0;  // #1907
        int32_t l0ab_cnt_16_state = l0ab_cnt_carry;  // #1908
        int32_t l0c_cnt_11_state = l0c_cnt_carry;  // #1909
        int32_t l1k_cnt_9_state = l1k_cnt_carry;  // #1910
        int32_t l1pv_cnt_8_state = l1pv_cnt_carry;  // #1911
        int32_t ready_lengths_2_state = 0;  // #1912
        ev_v_mte2_valid_d0_1.set();  // #1818
        ev_v_mte2_valid_d0_1.wait();  // #1819
        for (int32_t beat = 0; beat < add_3; beat += 1) {  // #3084
            const int32_t accum_step_carry = (int32_t)(accum_step_3_state);  // #1913
            const int32_t l0ab_cnt_carry_1 = (int32_t)(l0ab_cnt_16_state);  // #1914
            const int32_t l0c_cnt_carry_1 = (int32_t)(l0c_cnt_11_state);  // #1915
            const int32_t l1k_cnt_carry_1 = (int32_t)(l1k_cnt_9_state);  // #1916
            const int32_t l1pv_cnt_carry_1 = (int32_t)(l1pv_cnt_8_state);  // #1917
            const int32_t ready_lengths_carry = (int32_t)(ready_lengths_2_state);  // #1918
            const bool cmp_1 = beat < total_beats;  // #1920
            int32_t b_len_p_5_state = 0;  // #1921
            int32_t l0ab_cnt_8_state = 0;  // #1922
            int32_t l0c_cnt_8_state = 0;  // #1923
            int32_t l1k_cnt_8_state = 0;  // #1924
            if (cmp_1) {  // #2778
                const bool approx_p = beat < approx_beats_1;  // #1925
                const int32_t approx_g_p = beat * 6;  // #1926
                const int32_t e_base_p = Max(0, beat - approx_beats_1) * 6;  // #1935 #1936 #1937
                const int32_t b_len_p_1 = Min(6, exact_total - e_base_p);  // #1938 #1939
                int32_t b_len_p_4_state = 0;  // #1940
                uint64_t mask_0_2_state = 0ULL;  // #1941
                uint64_t mask_1_2_state = 0ULL;  // #1942
                uint64_t mask_2_2_state = 0ULL;  // #1943
                uint64_t mask_3_2_state = 0ULL;  // #1944
                uint64_t mask_4_2_state = 0ULL;  // #1945
                uint64_t mask_5_2_state = 0ULL;  // #1946
                if (approx_p) {  // #2091
                    const bool cmp_2 = approx_g_p < NG;  // #1949
                    uint64_t mask_0_1_state = 0ULL;  // #1950
                    if (cmp_2) {  // #1964
                        mask_0_1_state = group_approx_mask.load((((b * ((H * NT_Q) * NG)) + (h * (NT_Q * NG))) + (q_idx * NG)) + approx_g_p);  // #1952 #1953 #1954 #1955 #1956 #1957 #1958 #1959 #1960 #1961 #1962
                    } else {
                        mask_0_1_state = 0ULL;  // #1963
                    }
                    const uint64_t mask_0_1 = (uint64_t)(mask_0_1_state);  // #1965
                    const int32_t add_8 = approx_g_p + 1;  // #1966
                    const bool cmp_3 = add_8 < NG;  // #1968
                    uint64_t mask_1_1_state = 0ULL;  // #1969
                    if (cmp_3) {  // #1983
                        mask_1_1_state = group_approx_mask.load((((b * ((H * NT_Q) * NG)) + (h * (NT_Q * NG))) + (q_idx * NG)) + add_8);  // #1971 #1972 #1973 #1974 #1975 #1976 #1977 #1978 #1979 #1980 #1981
                    } else {
                        mask_1_1_state = 0ULL;  // #1982
                    }
                    const uint64_t mask_1_1 = (uint64_t)(mask_1_1_state);  // #1984
                    const int32_t add_12 = approx_g_p + 2;  // #1985
                    const bool cmp_4 = add_12 < NG;  // #1987
                    uint64_t mask_2_1_state = 0ULL;  // #1988
                    if (cmp_4) {  // #2002
                        mask_2_1_state = group_approx_mask.load((((b * ((H * NT_Q) * NG)) + (h * (NT_Q * NG))) + (q_idx * NG)) + add_12);  // #1990 #1991 #1992 #1993 #1994 #1995 #1996 #1997 #1998 #1999 #2000
                    } else {
                        mask_2_1_state = 0ULL;  // #2001
                    }
                    const uint64_t mask_2_1 = (uint64_t)(mask_2_1_state);  // #2003
                    const int32_t add_16 = approx_g_p + 3;  // #2004
                    const bool cmp_5 = add_16 < NG;  // #2006
                    uint64_t mask_3_1_state = 0ULL;  // #2007
                    if (cmp_5) {  // #2021
                        mask_3_1_state = group_approx_mask.load((((b * ((H * NT_Q) * NG)) + (h * (NT_Q * NG))) + (q_idx * NG)) + add_16);  // #2009 #2010 #2011 #2012 #2013 #2014 #2015 #2016 #2017 #2018 #2019
                    } else {
                        mask_3_1_state = 0ULL;  // #2020
                    }
                    const uint64_t mask_3_1 = (uint64_t)(mask_3_1_state);  // #2022
                    const int32_t add_20 = approx_g_p + 4;  // #2023
                    const bool cmp_6 = add_20 < NG;  // #2025
                    uint64_t mask_4_1_state = 0ULL;  // #2026
                    if (cmp_6) {  // #2040
                        mask_4_1_state = group_approx_mask.load((((b * ((H * NT_Q) * NG)) + (h * (NT_Q * NG))) + (q_idx * NG)) + add_20);  // #2028 #2029 #2030 #2031 #2032 #2033 #2034 #2035 #2036 #2037 #2038
                    } else {
                        mask_4_1_state = 0ULL;  // #2039
                    }
                    const uint64_t mask_4_1 = (uint64_t)(mask_4_1_state);  // #2041
                    const int32_t add_24 = approx_g_p + 5;  // #2042
                    const bool cmp_7 = add_24 < NG;  // #2044
                    uint64_t mask_5_1_state = 0ULL;  // #2045
                    if (cmp_7) {  // #2059
                        mask_5_1_state = group_approx_mask.load((((b * ((H * NT_Q) * NG)) + (h * (NT_Q * NG))) + (q_idx * NG)) + add_24);  // #2047 #2048 #2049 #2050 #2051 #2052 #2053 #2054 #2055 #2056 #2057
                    } else {
                        mask_5_1_state = 0ULL;  // #2058
                    }
                    const uint64_t mask_5_1 = (uint64_t)(mask_5_1_state);  // #2060
                    const bool cmp_8 = (((((((mask_0_1 | mask_1_1) | mask_2_1) | mask_3_1) | mask_4_1) | mask_5_1) | 0ULL) | 0ULL) != 0ULL;  // #2061 #2062 #2063 #2064 #2065 #2066 #2067 #2069
                    int32_t b_len_p_3_state = 0;  // #2070
                    if (cmp_8) {  // #2075
                        b_len_p_3_state = Min(6, NG - approx_g_p);  // #2071 #2072 #2073
                    } else {
                        b_len_p_3_state = 0;  // #2074
                    }
                    b_len_p_4_state = (int32_t)(b_len_p_3_state);  // #2076 #2077
                    mask_0_2_state = mask_0_1;  // #2078
                    mask_1_2_state = mask_1_1;  // #2079
                    mask_2_2_state = mask_2_1;  // #2080
                    mask_3_2_state = mask_3_1;  // #2081
                    mask_4_2_state = mask_4_1;  // #2082
                    mask_5_2_state = mask_5_1;  // #2083
                } else {
                    b_len_p_4_state = b_len_p_1;  // #2084
                    mask_0_2_state = 0ULL;  // #2085
                    mask_1_2_state = 0ULL;  // #2086
                    mask_2_2_state = 0ULL;  // #2087
                    mask_3_2_state = 0ULL;  // #2088
                    mask_4_2_state = 0ULL;  // #2089
                    mask_5_2_state = 0ULL;  // #2090
                }
                const int32_t b_len_p_4 = (int32_t)(b_len_p_4_state);  // #2092
                const uint64_t mask_0_2 = (uint64_t)(mask_0_2_state);  // #2093
                const uint64_t mask_1_2 = (uint64_t)(mask_1_2_state);  // #2094
                const uint64_t mask_2_2 = (uint64_t)(mask_2_2_state);  // #2095
                const uint64_t mask_3_2 = (uint64_t)(mask_3_2_state);  // #2096
                const uint64_t mask_4_2 = (uint64_t)(mask_4_2_state);  // #2097
                const uint64_t mask_5_2 = (uint64_t)(mask_5_2_state);  // #2098
                const bool cmp_9 = b_len_p_4 > 0;  // #2099
                int32_t l0ab_cnt_7_state = 0;  // #2100
                int32_t l0c_cnt_7_state = 0;  // #2101
                int32_t l1k_cnt_7_state = 0;  // #2102
                if (cmp_9) {  // #2766
                    const int32_t slot_b_p = beat & 3;  // #2103
                    const int32_t pair_len = Min(2, b_len_p_4);  // #2105
                    const bool cmp_10 = pair_len > 0;  // #2106
                    int32_t l0ab_cnt_2_state = 0;  // #2107
                    int32_t l0c_cnt_2_state = 0;  // #2108
                    int32_t l1k_cnt_2_state = 0;  // #2109
                    if (cmp_10) {  // #2188
                        if (approx_p) {  // #2168
                            if (pair_len < 2) {  // #2113 #2116
                                PipeBarrier<PIPE_MTE2>();  // #2115
                            }
                        } else {
                            if (pair_len < 2) {  // #2122 #2125
                                PipeBarrier<PIPE_MTE2>();  // #2124
                            }
                            if (0 < pair_len) {  // #2126 #2146
                                const uint16_t load = flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + e_base_p);  // #2129 #2130 #2131 #2132 #2133 #2134 #2135 #2136 #2137 #2138
                            }
                            if (1 < pair_len) {  // #2147 #2167
                                const int32_t add_35 = e_base_p + 1;  // #2148
                                const uint16_t load_1 = flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + add_35);  // #2150 #2151 #2152 #2153 #2154 #2155 #2156 #2157 #2158 #2159
                            }
                        }
                        const int32_t l0ab_cnt_1 = (int32_t)(l0ab_cnt_carry_1 + 1);  // #2174 #2175
                        const int32_t l1k_cnt_1 = (int32_t)(l1k_cnt_carry_1 + 1);  // #2178 #2179
                        const int32_t l0c_cnt_1 = (int32_t)(l0c_cnt_carry_1 + 1);  // #2180 #2181
                        l0ab_cnt_2_state = l0ab_cnt_1;  // #2182
                        l0c_cnt_2_state = l0c_cnt_1;  // #2183
                        l1k_cnt_2_state = l1k_cnt_1;  // #2184
                    } else {
                        l0ab_cnt_2_state = l0ab_cnt_carry_1;  // #2185
                        l0c_cnt_2_state = l0c_cnt_carry_1;  // #2186
                        l1k_cnt_2_state = l1k_cnt_carry_1;  // #2187
                    }
                    const int32_t l0ab_cnt_2 = (int32_t)(l0ab_cnt_2_state);  // #2189
                    const int32_t l0c_cnt_2 = (int32_t)(l0c_cnt_2_state);  // #2190
                    const int32_t l1k_cnt_2 = (int32_t)(l1k_cnt_2_state);  // #2191
                    const int32_t pair_len_1 = Min(2, b_len_p_4 - 2);  // #2192 #2193
                    const bool cmp_15 = pair_len_1 > 0;  // #2194
                    int32_t l0ab_cnt_4_state = 0;  // #2195
                    int32_t l0c_cnt_4_state = 0;  // #2196
                    int32_t l1k_cnt_4_state = 0;  // #2197
                    if (cmp_15) {  // #2276
                        if (approx_p) {  // #2256
                            if (pair_len_1 < 2) {  // #2201 #2204
                                PipeBarrier<PIPE_MTE2>();  // #2203
                            }
                        } else {
                            if (pair_len_1 < 2) {  // #2210 #2213
                                PipeBarrier<PIPE_MTE2>();  // #2212
                            }
                            if (0 < pair_len_1) {  // #2214 #2234
                                const int32_t add_45 = e_base_p + 2;  // #2215
                                const uint16_t load_2 = flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + add_45);  // #2217 #2218 #2219 #2220 #2221 #2222 #2223 #2224 #2225 #2226
                            }
                            if (1 < pair_len_1) {  // #2235 #2255
                                const int32_t add_50 = e_base_p + 3;  // #2236
                                const uint16_t load_3 = flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + add_50);  // #2238 #2239 #2240 #2241 #2242 #2243 #2244 #2245 #2246 #2247
                            }
                        }
                        const int32_t l0ab_cnt_3 = (int32_t)(l0ab_cnt_2 + 1);  // #2262 #2263
                        const int32_t l1k_cnt_3 = (int32_t)(l1k_cnt_2 + 1);  // #2266 #2267
                        const int32_t l0c_cnt_3 = (int32_t)(l0c_cnt_2 + 1);  // #2268 #2269
                        l0ab_cnt_4_state = l0ab_cnt_3;  // #2270
                        l0c_cnt_4_state = l0c_cnt_3;  // #2271
                        l1k_cnt_4_state = l1k_cnt_3;  // #2272
                    } else {
                        l0ab_cnt_4_state = l0ab_cnt_2;  // #2273
                        l0c_cnt_4_state = l0c_cnt_2;  // #2274
                        l1k_cnt_4_state = l1k_cnt_2;  // #2275
                    }
                    const int32_t l0ab_cnt_4 = (int32_t)(l0ab_cnt_4_state);  // #2277
                    const int32_t l0c_cnt_4 = (int32_t)(l0c_cnt_4_state);  // #2278
                    const int32_t l1k_cnt_4 = (int32_t)(l1k_cnt_4_state);  // #2279
                    const int32_t pair_len_2 = Min(2, b_len_p_4 - 4);  // #2280 #2281
                    const bool cmp_20 = pair_len_2 > 0;  // #2282
                    int32_t l0ab_cnt_6_state = 0;  // #2283
                    int32_t l0c_cnt_6_state = 0;  // #2284
                    int32_t l1k_cnt_6_state = 0;  // #2285
                    if (cmp_20) {  // #2364
                        if (approx_p) {  // #2344
                            if (pair_len_2 < 2) {  // #2289 #2292
                                PipeBarrier<PIPE_MTE2>();  // #2291
                            }
                        } else {
                            if (pair_len_2 < 2) {  // #2298 #2301
                                PipeBarrier<PIPE_MTE2>();  // #2300
                            }
                            if (0 < pair_len_2) {  // #2302 #2322
                                const int32_t add_60 = e_base_p + 4;  // #2303
                                const uint16_t load_4 = flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + add_60);  // #2305 #2306 #2307 #2308 #2309 #2310 #2311 #2312 #2313 #2314
                            }
                            if (1 < pair_len_2) {  // #2323 #2343
                                const int32_t add_65 = e_base_p + 5;  // #2324
                                const uint16_t load_5 = flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + add_65);  // #2326 #2327 #2328 #2329 #2330 #2331 #2332 #2333 #2334 #2335
                            }
                        }
                        const int32_t l0ab_cnt_5 = (int32_t)(l0ab_cnt_4 + 1);  // #2350 #2351
                        const int32_t l1k_cnt_5 = (int32_t)(l1k_cnt_4 + 1);  // #2354 #2355
                        const int32_t l0c_cnt_5 = (int32_t)(l0c_cnt_4 + 1);  // #2356 #2357
                        l0ab_cnt_6_state = l0ab_cnt_5;  // #2358
                        l0c_cnt_6_state = l0c_cnt_5;  // #2359
                        l1k_cnt_6_state = l1k_cnt_5;  // #2360
                    } else {
                        l0ab_cnt_6_state = l0ab_cnt_4;  // #2361
                        l0c_cnt_6_state = l0c_cnt_4;  // #2362
                        l1k_cnt_6_state = l1k_cnt_4;  // #2363
                    }
                    const int32_t l0ab_cnt_6 = (int32_t)(l0ab_cnt_6_state);  // #2365
                    const int32_t l0c_cnt_6 = (int32_t)(l0c_cnt_6_state);  // #2366
                    const int32_t l1k_cnt_6 = (int32_t)(l1k_cnt_6_state);  // #2367
                    WAIT_CUBE<PIPE_S>(0);  // #545
                    const int32_t pair_len_3 = Min(2, b_len_p_4);  // #2369
                    if (pair_len_3 > 0) {  // #2370 #2372
                        const Tensor<float, Position::UB> ub_sc = score_tiles.slot[0];  // #2371
                        WaitFlag<PIPE_V, PIPE_MTE2>(4);  // #551
                        const GMTensor<float> view_22 = score_ws[(((cube_idx * 196608) + (slot_b_p * 49152)) + (sb_row * 128))];  // #554
                        gm_to_ub_pad(ub_sc, view_22, 64, 512, 0, 0);  // #555
                        SetFlag<PIPE_MTE2, PIPE_V>(4);  // #556
                    }
                    const int32_t pair_len_4 = Min(2, b_len_p_4 - 2);  // #2373 #2374
                    if (pair_len_4 > 0) {  // #2375 #2377
                        const int32_t pair_row_v_1 = 128 + sb_row;  // #561
                        const Tensor<float, Position::UB> ub_sc_1 = score_tiles.slot[1];  // #2376
                        WaitFlag<PIPE_V, PIPE_MTE2>(5);  // #563
                        const GMTensor<float> view_23 = score_ws[(((cube_idx * 196608) + (slot_b_p * 49152)) + (pair_row_v_1 * 128))];  // #566
                        gm_to_ub_pad(ub_sc_1, view_23, 64, 512, 0, 0);  // #567
                        SetFlag<PIPE_MTE2, PIPE_V>(5);  // #568
                    }
                    const int32_t pair_len_5 = Min(2, b_len_p_4 - 4);  // #2378 #2379
                    if (pair_len_5 > 0) {  // #2380 #2382
                        const int32_t pair_row_v_2 = 256 + sb_row;  // #573
                        const Tensor<float, Position::UB> ub_sc_2 = score_tiles.slot[2];  // #2381
                        WaitFlag<PIPE_V, PIPE_MTE2>(6);  // #575
                        const GMTensor<float> view_24 = score_ws[(((cube_idx * 196608) + (slot_b_p * 49152)) + (pair_row_v_2 * 128))];  // #578
                        gm_to_ub_pad(ub_sc_2, view_24, 64, 512, 0, 0);  // #579
                        SetFlag<PIPE_MTE2, PIPE_V>(6);  // #580
                    }
                    PipeBarrier<PIPE_V>();  // #582
                    SetMaskCount();  // #583
                    SetVectorMask(64);  // #583
                    vector_dup(ub_bmax_s.ptr(), (float)(-1e+30f), (uint8_t)(1), (uint16_t)1, 1, (uint8_t)8, 0);  // #583
                    SetMaskNorm();  // #583
                    ResetMask();  // #583
                    const int32_t pair_len_6 = Min(2, b_len_p_4);  // #2384
                    if (pair_len_6 > 0) {  // #2385 #2450
                        const Tensor<float, Position::UB> ub_sc_3 = score_tiles.slot[0];  // #2386
                        WaitFlag<PIPE_MTE2, PIPE_V>(4);  // #588
                        vmuls(ub_sc_3.ptr(), ub_sc_3.ptr(), (float)(scale_natural), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #589
                        if (approx_p) {  // #2447
                            SetVectorMask((uint64_t)(0), (uint64_t)(~(mask_0_2)));  // #2387 #591
                            PipeBarrier<PIPE_V>();  // #592
                            const Tensor<float, Position::UB> view_25 = score_tiles.slot[0];  // #2388
                            vector_dup(view_25.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #594
                            ResetMask();  // #595
                            SetVectorMask((uint64_t)(0), (uint64_t)(~(mask_1_2)));  // #2389 #597
                            PipeBarrier<PIPE_V>();  // #598
                            const Tensor<float, Position::UB> view_26 = score_tiles.slot[0][64];  // #2390
                            vector_dup(view_26.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #600
                            ResetMask();  // #601
                        } else {
                            const Tensor<float, Position::UB> ub_cols = score_tiles.slot[0];  // #2391
                            if (0 < pair_len_6) {  // #2392 #2418
                                const int32_t blk_p_7 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + e_base_p));  // #2396 #2397 #2398 #2399 #2400 #2401 #2402 #2403 #2404 #2405 #2406
                                const int32_t valid_n = Min(64, T - (blk_p_7 * 64));  // #2407 #2408 #2409
                                if (valid_n < 64) {  // #2410 #2417
                                    const uint64_t low = 1ULL << valid_n;  // #2413
                                    SetVectorMask((uint64_t)(0), (uint64_t)(18446744073709551615ULL ^ (low - 1ULL)));  // #2415 #2416 #628
                                    PipeBarrier<PIPE_V>();  // #629
                                    vector_dup(ub_cols.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #630
                                    ResetMask();  // #631
                                }
                            } else {
                                PipeBarrier<PIPE_V>();  // #633
                                vector_dup(ub_cols.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #634
                            }
                            const Tensor<float, Position::UB> ub_cols_1 = score_tiles.slot[0][64];  // #2419
                            if (1 < pair_len_6) {  // #2420 #2446
                                const int32_t blk_p_8 = e_base_p + 1;  // #2422
                                const int32_t blk_p_9 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + blk_p_8));  // #2424 #2425 #2426 #2427 #2428 #2429 #2430 #2431 #2432 #2433 #2434
                                const int32_t valid_n_1 = Min(64, T - (blk_p_9 * 64));  // #2435 #2436 #2437
                                if (valid_n_1 < 64) {  // #2438 #2445
                                    const uint64_t low_1 = 1ULL << valid_n_1;  // #2441
                                    SetVectorMask((uint64_t)(0), (uint64_t)(18446744073709551615ULL ^ (low_1 - 1ULL)));  // #2443 #2444 #662
                                    PipeBarrier<PIPE_V>();  // #663
                                    vector_dup(ub_cols_1.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #664
                                    ResetMask();  // #665
                                }
                            } else {
                                PipeBarrier<PIPE_V>();  // #667
                                vector_dup(ub_cols_1.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #668
                            }
                        }
                        PipeBarrier<PIPE_V>();  // #671
                        vcgmax(ub_reduce_pair.ptr(), ub_sc_3.ptr(), (uint8_t)(128), 1, 1, 8);  // #672
                        PipeBarrier<PIPE_V>();  // #673
                        const Tensor<float, Position::UB> view_27 = ub_reduce_pair;  // #2448
                        const Tensor<float, Position::UB> view_28 = ub_reduce_pair[8];  // #2449
                        vmax(ub_max.ptr(), view_27.ptr(), view_28.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)2, (uint8_t)2, (uint8_t)8, (uint8_t)16, (uint8_t)16);  // #676
                        PipeBarrier<PIPE_V>();  // #677
                        vcgmax(ub_max_s.ptr(), ub_max.ptr(), (uint8_t)(8), 1, 1, 8);  // #678
                        PipeBarrier<PIPE_V>();  // #679
                        SetMaskCount();  // #680
                        SetVectorMask(64);  // #680
                        vmax(ub_bmax_s.ptr(), ub_bmax_s.ptr(), ub_max_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #680
                        SetMaskNorm();  // #680
                        ResetMask();  // #680
                    }
                    const int32_t pair_len_7 = Min(2, b_len_p_4 - 2);  // #2451 #2452
                    if (pair_len_7 > 0) {  // #2453 #2518
                        const Tensor<float, Position::UB> ub_sc_4 = score_tiles.slot[1];  // #2454
                        WaitFlag<PIPE_MTE2, PIPE_V>(5);  // #686
                        vmuls(ub_sc_4.ptr(), ub_sc_4.ptr(), (float)(scale_natural), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #687
                        if (approx_p) {  // #2515
                            SetVectorMask((uint64_t)(0), (uint64_t)(~(mask_2_2)));  // #2455 #689
                            PipeBarrier<PIPE_V>();  // #690
                            const Tensor<float, Position::UB> view_29 = score_tiles.slot[1];  // #2456
                            vector_dup(view_29.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #692
                            ResetMask();  // #693
                            SetVectorMask((uint64_t)(0), (uint64_t)(~(mask_3_2)));  // #2457 #695
                            PipeBarrier<PIPE_V>();  // #696
                            const Tensor<float, Position::UB> view_30 = score_tiles.slot[1][64];  // #2458
                            vector_dup(view_30.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #698
                            ResetMask();  // #699
                        } else {
                            const Tensor<float, Position::UB> ub_cols_2 = score_tiles.slot[1];  // #2459
                            if (0 < pair_len_7) {  // #2460 #2486
                                const int32_t add_84 = e_base_p + 2;  // #2461
                                const int32_t blk_p_11 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + add_84));  // #2464 #2465 #2466 #2467 #2468 #2469 #2470 #2471 #2472 #2473 #2474
                                const int32_t valid_n_2 = Min(64, T - (blk_p_11 * 64));  // #2475 #2476 #2477
                                if (valid_n_2 < 64) {  // #2478 #2485
                                    const uint64_t low_2 = 1ULL << valid_n_2;  // #2481
                                    SetVectorMask((uint64_t)(0), (uint64_t)(18446744073709551615ULL ^ (low_2 - 1ULL)));  // #2483 #2484 #726
                                    PipeBarrier<PIPE_V>();  // #727
                                    vector_dup(ub_cols_2.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #728
                                    ResetMask();  // #729
                                }
                            } else {
                                PipeBarrier<PIPE_V>();  // #731
                                vector_dup(ub_cols_2.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #732
                            }
                            const Tensor<float, Position::UB> ub_cols_3 = score_tiles.slot[1][64];  // #2487
                            if (1 < pair_len_7) {  // #2488 #2514
                                const int32_t blk_p_12 = (e_base_p + 2) + 1;  // #2489 #2490
                                const int32_t blk_p_13 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + blk_p_12));  // #2492 #2493 #2494 #2495 #2496 #2497 #2498 #2499 #2500 #2501 #2502
                                const int32_t valid_n_3 = Min(64, T - (blk_p_13 * 64));  // #2503 #2504 #2505
                                if (valid_n_3 < 64) {  // #2506 #2513
                                    const uint64_t low_3 = 1ULL << valid_n_3;  // #2509
                                    SetVectorMask((uint64_t)(0), (uint64_t)(18446744073709551615ULL ^ (low_3 - 1ULL)));  // #2511 #2512 #760
                                    PipeBarrier<PIPE_V>();  // #761
                                    vector_dup(ub_cols_3.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #762
                                    ResetMask();  // #763
                                }
                            } else {
                                PipeBarrier<PIPE_V>();  // #765
                                vector_dup(ub_cols_3.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #766
                            }
                        }
                        PipeBarrier<PIPE_V>();  // #769
                        vcgmax(ub_reduce_pair.ptr(), ub_sc_4.ptr(), (uint8_t)(128), 1, 1, 8);  // #770
                        PipeBarrier<PIPE_V>();  // #771
                        const Tensor<float, Position::UB> view_31 = ub_reduce_pair;  // #2516
                        const Tensor<float, Position::UB> view_32 = ub_reduce_pair[8];  // #2517
                        vmax(ub_max.ptr(), view_31.ptr(), view_32.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)2, (uint8_t)2, (uint8_t)8, (uint8_t)16, (uint8_t)16);  // #774
                        PipeBarrier<PIPE_V>();  // #775
                        vcgmax(ub_max_s.ptr(), ub_max.ptr(), (uint8_t)(8), 1, 1, 8);  // #776
                        PipeBarrier<PIPE_V>();  // #777
                        SetMaskCount();  // #778
                        SetVectorMask(64);  // #778
                        vmax(ub_bmax_s.ptr(), ub_bmax_s.ptr(), ub_max_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #778
                        SetMaskNorm();  // #778
                        ResetMask();  // #778
                    }
                    const int32_t pair_len_8 = Min(2, b_len_p_4 - 4);  // #2519 #2520
                    if (pair_len_8 > 0) {  // #2521 #2586
                        const Tensor<float, Position::UB> ub_sc_5 = score_tiles.slot[2];  // #2522
                        WaitFlag<PIPE_MTE2, PIPE_V>(6);  // #784
                        vmuls(ub_sc_5.ptr(), ub_sc_5.ptr(), (float)(scale_natural), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);  // #785
                        if (approx_p) {  // #2583
                            SetVectorMask((uint64_t)(0), (uint64_t)(~(mask_4_2)));  // #2523 #787
                            PipeBarrier<PIPE_V>();  // #788
                            const Tensor<float, Position::UB> view_33 = score_tiles.slot[2];  // #2524
                            vector_dup(view_33.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #790
                            ResetMask();  // #791
                            SetVectorMask((uint64_t)(0), (uint64_t)(~(mask_5_2)));  // #2525 #793
                            PipeBarrier<PIPE_V>();  // #794
                            const Tensor<float, Position::UB> view_34 = score_tiles.slot[2][64];  // #2526
                            vector_dup(view_34.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #796
                            ResetMask();  // #797
                        } else {
                            const Tensor<float, Position::UB> ub_cols_4 = score_tiles.slot[2];  // #2527
                            if (0 < pair_len_8) {  // #2528 #2554
                                const int32_t add_92 = e_base_p + 4;  // #2529
                                const int32_t blk_p_15 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + add_92));  // #2532 #2533 #2534 #2535 #2536 #2537 #2538 #2539 #2540 #2541 #2542
                                const int32_t valid_n_4 = Min(64, T - (blk_p_15 * 64));  // #2543 #2544 #2545
                                if (valid_n_4 < 64) {  // #2546 #2553
                                    const uint64_t low_4 = 1ULL << valid_n_4;  // #2549
                                    SetVectorMask((uint64_t)(0), (uint64_t)(18446744073709551615ULL ^ (low_4 - 1ULL)));  // #2551 #2552 #824
                                    PipeBarrier<PIPE_V>();  // #825
                                    vector_dup(ub_cols_4.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #826
                                    ResetMask();  // #827
                                }
                            } else {
                                PipeBarrier<PIPE_V>();  // #829
                                vector_dup(ub_cols_4.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #830
                            }
                            const Tensor<float, Position::UB> ub_cols_5 = score_tiles.slot[2][64];  // #2555
                            if (1 < pair_len_8) {  // #2556 #2582
                                const int32_t blk_p_16 = (e_base_p + 4) + 1;  // #2557 #2558
                                const int32_t blk_p_17 = (int32_t)(flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + blk_p_16));  // #2560 #2561 #2562 #2563 #2564 #2565 #2566 #2567 #2568 #2569 #2570
                                const int32_t valid_n_5 = Min(64, T - (blk_p_17 * 64));  // #2571 #2572 #2573
                                if (valid_n_5 < 64) {  // #2574 #2581
                                    const uint64_t low_5 = 1ULL << valid_n_5;  // #2577
                                    SetVectorMask((uint64_t)(0), (uint64_t)(18446744073709551615ULL ^ (low_5 - 1ULL)));  // #2579 #2580 #858
                                    PipeBarrier<PIPE_V>();  // #859
                                    vector_dup(ub_cols_5.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #860
                                    ResetMask();  // #861
                                }
                            } else {
                                PipeBarrier<PIPE_V>();  // #863
                                vector_dup(ub_cols_5.ptr(), (float)(-1e+30f), (uint8_t)(64), (uint16_t)1, 1, (uint8_t)16, 0);  // #864
                            }
                        }
                        PipeBarrier<PIPE_V>();  // #867
                        vcgmax(ub_reduce_pair.ptr(), ub_sc_5.ptr(), (uint8_t)(128), 1, 1, 8);  // #868
                        PipeBarrier<PIPE_V>();  // #869
                        const Tensor<float, Position::UB> view_35 = ub_reduce_pair;  // #2584
                        const Tensor<float, Position::UB> view_36 = ub_reduce_pair[8];  // #2585
                        vmax(ub_max.ptr(), view_35.ptr(), view_36.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)2, (uint8_t)2, (uint8_t)8, (uint8_t)16, (uint8_t)16);  // #872
                        PipeBarrier<PIPE_V>();  // #873
                        vcgmax(ub_max_s.ptr(), ub_max.ptr(), (uint8_t)(8), 1, 1, 8);  // #874
                        PipeBarrier<PIPE_V>();  // #875
                        SetMaskCount();  // #876
                        SetVectorMask(64);  // #876
                        vmax(ub_bmax_s.ptr(), ub_bmax_s.ptr(), ub_max_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #876
                        SetMaskNorm();  // #876
                        ResetMask();  // #876
                    }
                    const Tensor<float, Position::UB> row_diff = expdiff_buf.get(beat);  // #2587
                    SetMaskCount();  // #879
                    SetVectorMask(64);  // #879
                    vadd(ub_old_max_s.ptr(), ub_rmax_s.ptr(), ub_zero_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #879
                    SetMaskNorm();  // #879
                    ResetMask();  // #879
                    PipeBarrier<PIPE_V>();  // #880
                    SetMaskCount();  // #881
                    SetVectorMask(64);  // #881
                    vmax(ub_rmax_s.ptr(), ub_rmax_s.ptr(), ub_bmax_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #881
                    SetMaskNorm();  // #881
                    ResetMask();  // #881
                    PipeBarrier<PIPE_V>();  // #882
                    SetMaskCount();  // #883
                    SetVectorMask(64);  // #883
                    vsub(row_diff.ptr(), ub_old_max_s.ptr(), ub_rmax_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #883
                    SetMaskNorm();  // #883
                    ResetMask();  // #883
                    PipeBarrier<PIPE_V>();  // #884
                    SetMaskCount();  // #885
                    SetVectorMask(64);  // #885
                    vexp(row_diff.ptr(), row_diff.ptr(), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint8_t)0, (uint8_t)0);  // #885
                    SetMaskNorm();  // #885
                    ResetMask();  // #885
                    SetMaskCount();  // #886
                    SetVectorMask(64);  // #886
                    vector_dup(ub_bsum_s.ptr(), (float)(0.0f), (uint8_t)(1), (uint16_t)1, 1, (uint8_t)8, 0);  // #886
                    SetMaskNorm();  // #886
                    ResetMask();  // #886
                    ResetMask();  // #887
                    vbrcb((__ubuf__ uint32_t*)(ub_max.ptr()), (__ubuf__ uint32_t*)(ub_rmax_s.ptr()), (uint16_t)1, (uint16_t)8, (uint8_t)(8));  // #887
                    PipeBarrier<PIPE_V>();  // #888
                    const int32_t pair_len_9 = Min(2, b_len_p_4);  // #2589
                    if (pair_len_9 > 0) {  // #2590 #2644
                        const Tensor<float, Position::UB> ub_sc_6 = score_tiles.slot[0];  // #2591
                        const Tensor<bfloat16_t, Position::UB> ub_prob = probability_tiles.slot[0];  // #2592
                        const Tensor<float, Position::UB> ub_cols_6 = score_tiles.slot[0];  // #2593
                        vsub(ub_cols_6.ptr(), ub_cols_6.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #896
                        const Tensor<float, Position::UB> ub_cols_7 = score_tiles.slot[0][64];  // #2594
                        PipeBarrier<PIPE_V>();  // c220: V-V hazard #898
                        vsub(ub_cols_7.ptr(), ub_cols_7.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #898
                        PipeBarrier<PIPE_V>();  // #899
                        vexp(ub_sc_6.ptr(), ub_sc_6.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint8_t)8, (uint8_t)8);  // #900
                        PipeBarrier<PIPE_V>();  // #901
                        WaitFlag<PIPE_MTE3, PIPE_V>(4);  // #902
                        vconv_f322bf16a(ub_prob.ptr(), ub_sc_6.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)4, (uint16_t)8);  // #903
                        SetFlag<PIPE_V, PIPE_MTE3>(4);  // #904
                        WaitFlag<PIPE_V, PIPE_MTE3>(4);  // #905
                        const GMTensor<bfloat16_t> view_37 = p_ws[(((cube_idx * 196608) + (slot_b_p * 49152)) + (sb_row * 128))];  // #908
                        ub_to_gm_pad(view_37, ub_prob, 64, 256, 0, 0);  // #909
                        SetFlag<PIPE_MTE3, PIPE_V>(4);  // #910
                        vcgadd(ub_reduce_pair.ptr(), ub_sc_6.ptr(), (uint8_t)(128), 1, 1, 8);  // #911
                        const bool cmp_44 = pair_len_9 == 2;  // #2595
                        bool full_pair_state = false;  // #2596
                        if (cmp_44) {  // #2608
                            const bool not_6 = !(approx_p);  // #2597
                            bool choice_state = false;  // #2598
                            if (not_6) {  // #2604
                                choice_state = not_6;  // #2599
                            } else {
                                choice_state = (approx_g_p + 2) < NG;  // #2601 #2602 #2603
                            }
                            full_pair_state = (bool)(choice_state);  // #2605 #2606
                        } else {
                            full_pair_state = cmp_44;  // #2607
                        }
                        const bool full_pair = (bool)(full_pair_state);  // #2609
                        if (full_pair) {  // #2643
                            PipeBarrier<PIPE_V>();  // #923
                            const Tensor<float, Position::UB> view_38 = ub_reduce_pair;  // #2610
                            const Tensor<float, Position::UB> view_39 = ub_reduce_pair[8];  // #2611
                            vadd(ub_rowsum.ptr(), view_38.ptr(), view_39.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)2, (uint8_t)2, (uint8_t)8, (uint8_t)16, (uint8_t)16);  // #926
                            PipeBarrier<PIPE_V>();  // #927
                            vcgadd(ub_sum_s.ptr(), ub_rowsum.ptr(), (uint8_t)(8), 1, 1, 8);  // #928
                            if (approx_p) {  // #2612
                                PipeBarrier<PIPE_V>();  // #929
                                SetMaskCount();  // #930
                                SetVectorMask(64);  // #930
                                vmuls(ub_sum_s.ptr(), ub_sum_s.ptr(), (float)(64.0f), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #930
                                SetMaskNorm();  // #930
                                ResetMask();  // #930
                            }
                            PipeBarrier<PIPE_V>();  // #932
                            SetMaskCount();  // #933
                            SetVectorMask(64);  // #933
                            vadd(ub_bsum_s.ptr(), ub_bsum_s.ptr(), ub_sum_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #933
                            SetMaskNorm();  // #933
                            ResetMask();  // #933
                        } else {
                            if (0 < pair_len_9) {  // #2613 #2627
                                PipeBarrier<PIPE_V>();  // #935
                                bool choice_1_state = false;  // #2614
                                if (approx_p) {  // #2621
                                    const int32_t sub_41 = NG + -1;  // #2617
                                    choice_1_state = approx_g_p == sub_41;  // #2618 #2619
                                } else {
                                    choice_1_state = false;  // #2620
                                }
                                const bool choice_1 = (bool)(choice_1_state);  // #2622
                                if (choice_1) {  // #2626
                                    const Tensor<float, Position::UB> ub_cols_8 = score_tiles.slot[0];  // #2623
                                    vmul(ub_cols_8.ptr(), ub_cols_8.ptr(), ub_len_last.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)16, (uint8_t)16, (uint8_t)0);  // #944
                                    PipeBarrier<PIPE_V>();  // #945
                                    vcgadd(ub_reduce.ptr(), ub_cols_8.ptr(), (uint8_t)(64), 1, 1, 16);  // #946
                                    PipeBarrier<PIPE_V>();  // #947
                                    vcgadd(ub_sum_s.ptr(), ub_reduce.ptr(), (uint8_t)(8), 1, 1, 8);  // #948
                                } else {
                                    const Tensor<float, Position::UB> view_40 = ub_reduce_pair;  // #2624
                                    vcgadd(ub_sum_s.ptr(), view_40.ptr(), (uint8_t)(8), 1, 2, 16);  // #950
                                    if (approx_p) {  // #2625
                                        PipeBarrier<PIPE_V>();  // #951
                                        SetMaskCount();  // #952
                                        SetVectorMask(64);  // #952
                                        vmuls(ub_sum_s.ptr(), ub_sum_s.ptr(), (float)(64.0f), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #952
                                        SetMaskNorm();  // #952
                                        ResetMask();  // #952
                                    }
                                }
                                PipeBarrier<PIPE_V>();  // #955
                                SetMaskCount();  // #956
                                SetVectorMask(64);  // #956
                                vadd(ub_bsum_s.ptr(), ub_bsum_s.ptr(), ub_sum_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #956
                                SetMaskNorm();  // #956
                                ResetMask();  // #956
                            }
                            if (1 < pair_len_9) {  // #2628 #2642
                                PipeBarrier<PIPE_V>();  // #959
                                bool choice_2_state = false;  // #2629
                                if (approx_p) {  // #2636
                                    const int32_t sub_42 = NG + -1;  // #2632
                                    choice_2_state = (approx_g_p + 1) == sub_42;  // #2631 #2633 #2634
                                } else {
                                    choice_2_state = false;  // #2635
                                }
                                const bool choice_2 = (bool)(choice_2_state);  // #2637
                                if (choice_2) {  // #2641
                                    const Tensor<float, Position::UB> ub_cols_9 = score_tiles.slot[0][64];  // #2638
                                    vmul(ub_cols_9.ptr(), ub_cols_9.ptr(), ub_len_last.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)16, (uint8_t)16, (uint8_t)0);  // #968
                                    PipeBarrier<PIPE_V>();  // #969
                                    vcgadd(ub_reduce.ptr(), ub_cols_9.ptr(), (uint8_t)(64), 1, 1, 16);  // #970
                                    PipeBarrier<PIPE_V>();  // #971
                                    vcgadd(ub_sum_s.ptr(), ub_reduce.ptr(), (uint8_t)(8), 1, 1, 8);  // #972
                                } else {
                                    const Tensor<float, Position::UB> view_41 = ub_reduce_pair[8];  // #2639
                                    vcgadd(ub_sum_s.ptr(), view_41.ptr(), (uint8_t)(8), 1, 2, 16);  // #974
                                    if (approx_p) {  // #2640
                                        PipeBarrier<PIPE_V>();  // #975
                                        SetMaskCount();  // #976
                                        SetVectorMask(64);  // #976
                                        vmuls(ub_sum_s.ptr(), ub_sum_s.ptr(), (float)(64.0f), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #976
                                        SetMaskNorm();  // #976
                                        ResetMask();  // #976
                                    }
                                }
                                PipeBarrier<PIPE_V>();  // #979
                                SetMaskCount();  // #980
                                SetVectorMask(64);  // #980
                                vadd(ub_bsum_s.ptr(), ub_bsum_s.ptr(), ub_sum_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #980
                                SetMaskNorm();  // #980
                                ResetMask();  // #980
                            }
                        }
                        SetFlag<PIPE_V, PIPE_MTE2>(4);  // #983
                    }
                    const int32_t pair_len_10 = Min(2, b_len_p_4 - 2);  // #2645 #2646
                    if (pair_len_10 > 0) {  // #2647 #2701
                        const int32_t pair_row_v_4 = 128 + sb_row;  // #988
                        const Tensor<float, Position::UB> ub_sc_7 = score_tiles.slot[1];  // #2648
                        const Tensor<bfloat16_t, Position::UB> ub_prob_1 = probability_tiles.slot[0];  // #2649
                        const Tensor<float, Position::UB> ub_cols_10 = score_tiles.slot[1];  // #2650
                        PipeBarrier<PIPE_V>();  // c220: V-V hazard #992
                        vsub(ub_cols_10.ptr(), ub_cols_10.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #992
                        const Tensor<float, Position::UB> ub_cols_11 = score_tiles.slot[1][64];  // #2651
                        PipeBarrier<PIPE_V>();  // c220: V-V hazard #994
                        vsub(ub_cols_11.ptr(), ub_cols_11.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #994
                        PipeBarrier<PIPE_V>();  // #995
                        vexp(ub_sc_7.ptr(), ub_sc_7.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint8_t)8, (uint8_t)8);  // #996
                        PipeBarrier<PIPE_V>();  // #997
                        WaitFlag<PIPE_MTE3, PIPE_V>(4);  // #998
                        vconv_f322bf16a(ub_prob_1.ptr(), ub_sc_7.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)4, (uint16_t)8);  // #999
                        SetFlag<PIPE_V, PIPE_MTE3>(4);  // #1000
                        WaitFlag<PIPE_V, PIPE_MTE3>(4);  // #1001
                        const GMTensor<bfloat16_t> view_42 = p_ws[(((cube_idx * 196608) + (slot_b_p * 49152)) + (pair_row_v_4 * 128))];  // #1004
                        ub_to_gm_pad(view_42, ub_prob_1, 64, 256, 0, 0);  // #1005
                        SetFlag<PIPE_MTE3, PIPE_V>(4);  // #1006
                        vcgadd(ub_reduce_pair.ptr(), ub_sc_7.ptr(), (uint8_t)(128), 1, 1, 8);  // #1007
                        const bool cmp_51 = pair_len_10 == 2;  // #2652
                        bool full_pair_1_state = false;  // #2653
                        if (cmp_51) {  // #2665
                            const bool not_7 = !(approx_p);  // #2654
                            bool choice_3_state = false;  // #2655
                            if (not_7) {  // #2661
                                choice_3_state = not_7;  // #2656
                            } else {
                                choice_3_state = ((approx_g_p + 2) + 2) < NG;  // #2657 #2658 #2659 #2660
                            }
                            full_pair_1_state = (bool)(choice_3_state);  // #2662 #2663
                        } else {
                            full_pair_1_state = cmp_51;  // #2664
                        }
                        const bool full_pair_1 = (bool)(full_pair_1_state);  // #2666
                        if (full_pair_1) {  // #2700
                            PipeBarrier<PIPE_V>();  // #1019
                            const Tensor<float, Position::UB> view_43 = ub_reduce_pair;  // #2667
                            const Tensor<float, Position::UB> view_44 = ub_reduce_pair[8];  // #2668
                            vadd(ub_rowsum.ptr(), view_43.ptr(), view_44.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)2, (uint8_t)2, (uint8_t)8, (uint8_t)16, (uint8_t)16);  // #1022
                            PipeBarrier<PIPE_V>();  // #1023
                            vcgadd(ub_sum_s.ptr(), ub_rowsum.ptr(), (uint8_t)(8), 1, 1, 8);  // #1024
                            if (approx_p) {  // #2669
                                PipeBarrier<PIPE_V>();  // #1025
                                SetMaskCount();  // #1026
                                SetVectorMask(64);  // #1026
                                vmuls(ub_sum_s.ptr(), ub_sum_s.ptr(), (float)(64.0f), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #1026
                                SetMaskNorm();  // #1026
                                ResetMask();  // #1026
                            }
                            PipeBarrier<PIPE_V>();  // #1028
                            SetMaskCount();  // #1029
                            SetVectorMask(64);  // #1029
                            vadd(ub_bsum_s.ptr(), ub_bsum_s.ptr(), ub_sum_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1029
                            SetMaskNorm();  // #1029
                            ResetMask();  // #1029
                        } else {
                            if (0 < pair_len_10) {  // #2670 #2684
                                PipeBarrier<PIPE_V>();  // #1031
                                bool choice_4_state = false;  // #2671
                                if (approx_p) {  // #2678
                                    const int32_t sub_45 = NG + -1;  // #2674
                                    choice_4_state = (approx_g_p + 2) == sub_45;  // #2672 #2675 #2676
                                } else {
                                    choice_4_state = false;  // #2677
                                }
                                const bool choice_4 = (bool)(choice_4_state);  // #2679
                                if (choice_4) {  // #2683
                                    const Tensor<float, Position::UB> ub_cols_12 = score_tiles.slot[1];  // #2680
                                    vmul(ub_cols_12.ptr(), ub_cols_12.ptr(), ub_len_last.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)16, (uint8_t)16, (uint8_t)0);  // #1040
                                    PipeBarrier<PIPE_V>();  // #1041
                                    vcgadd(ub_reduce.ptr(), ub_cols_12.ptr(), (uint8_t)(64), 1, 1, 16);  // #1042
                                    PipeBarrier<PIPE_V>();  // #1043
                                    vcgadd(ub_sum_s.ptr(), ub_reduce.ptr(), (uint8_t)(8), 1, 1, 8);  // #1044
                                } else {
                                    const Tensor<float, Position::UB> view_45 = ub_reduce_pair;  // #2681
                                    vcgadd(ub_sum_s.ptr(), view_45.ptr(), (uint8_t)(8), 1, 2, 16);  // #1046
                                    if (approx_p) {  // #2682
                                        PipeBarrier<PIPE_V>();  // #1047
                                        SetMaskCount();  // #1048
                                        SetVectorMask(64);  // #1048
                                        vmuls(ub_sum_s.ptr(), ub_sum_s.ptr(), (float)(64.0f), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #1048
                                        SetMaskNorm();  // #1048
                                        ResetMask();  // #1048
                                    }
                                }
                                PipeBarrier<PIPE_V>();  // #1051
                                SetMaskCount();  // #1052
                                SetVectorMask(64);  // #1052
                                vadd(ub_bsum_s.ptr(), ub_bsum_s.ptr(), ub_sum_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1052
                                SetMaskNorm();  // #1052
                                ResetMask();  // #1052
                            }
                            if (1 < pair_len_10) {  // #2685 #2699
                                PipeBarrier<PIPE_V>();  // #1055
                                bool choice_5_state = false;  // #2686
                                if (approx_p) {  // #2693
                                    const int32_t sub_46 = NG + -1;  // #2689
                                    choice_5_state = ((approx_g_p + 2) + 1) == sub_46;  // #2687 #2688 #2690 #2691
                                } else {
                                    choice_5_state = false;  // #2692
                                }
                                const bool choice_5 = (bool)(choice_5_state);  // #2694
                                if (choice_5) {  // #2698
                                    const Tensor<float, Position::UB> ub_cols_13 = score_tiles.slot[1][64];  // #2695
                                    vmul(ub_cols_13.ptr(), ub_cols_13.ptr(), ub_len_last.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)16, (uint8_t)16, (uint8_t)0);  // #1064
                                    PipeBarrier<PIPE_V>();  // #1065
                                    vcgadd(ub_reduce.ptr(), ub_cols_13.ptr(), (uint8_t)(64), 1, 1, 16);  // #1066
                                    PipeBarrier<PIPE_V>();  // #1067
                                    vcgadd(ub_sum_s.ptr(), ub_reduce.ptr(), (uint8_t)(8), 1, 1, 8);  // #1068
                                } else {
                                    const Tensor<float, Position::UB> view_46 = ub_reduce_pair[8];  // #2696
                                    vcgadd(ub_sum_s.ptr(), view_46.ptr(), (uint8_t)(8), 1, 2, 16);  // #1070
                                    if (approx_p) {  // #2697
                                        PipeBarrier<PIPE_V>();  // #1071
                                        SetMaskCount();  // #1072
                                        SetVectorMask(64);  // #1072
                                        vmuls(ub_sum_s.ptr(), ub_sum_s.ptr(), (float)(64.0f), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #1072
                                        SetMaskNorm();  // #1072
                                        ResetMask();  // #1072
                                    }
                                }
                                PipeBarrier<PIPE_V>();  // #1075
                                SetMaskCount();  // #1076
                                SetVectorMask(64);  // #1076
                                vadd(ub_bsum_s.ptr(), ub_bsum_s.ptr(), ub_sum_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1076
                                SetMaskNorm();  // #1076
                                ResetMask();  // #1076
                            }
                        }
                        SetFlag<PIPE_V, PIPE_MTE2>(5);  // #1079
                    }
                    const int32_t pair_len_11 = Min(2, b_len_p_4 - 4);  // #2702 #2703
                    if (pair_len_11 > 0) {  // #2704 #2758
                        const int32_t pair_row_v_5 = 256 + sb_row;  // #1084
                        const Tensor<float, Position::UB> ub_sc_8 = score_tiles.slot[2];  // #2705
                        const Tensor<bfloat16_t, Position::UB> ub_prob_2 = probability_tiles.slot[0];  // #2706
                        const Tensor<float, Position::UB> ub_cols_14 = score_tiles.slot[2];  // #2707
                        PipeBarrier<PIPE_V>();  // c220: V-V hazard #1088
                        vsub(ub_cols_14.ptr(), ub_cols_14.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #1088
                        const Tensor<float, Position::UB> ub_cols_15 = score_tiles.slot[2][64];  // #2708
                        PipeBarrier<PIPE_V>();  // c220: V-V hazard #1090
                        vsub(ub_cols_15.ptr(), ub_cols_15.ptr(), ub_max.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #1090
                        PipeBarrier<PIPE_V>();  // #1091
                        vexp(ub_sc_8.ptr(), ub_sc_8.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint8_t)8, (uint8_t)8);  // #1092
                        PipeBarrier<PIPE_V>();  // #1093
                        WaitFlag<PIPE_MTE3, PIPE_V>(4);  // #1094
                        vconv_f322bf16a(ub_prob_2.ptr(), ub_sc_8.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)4, (uint16_t)8);  // #1095
                        SetFlag<PIPE_V, PIPE_MTE3>(4);  // #1096
                        WaitFlag<PIPE_V, PIPE_MTE3>(4);  // #1097
                        const GMTensor<bfloat16_t> view_47 = p_ws[(((cube_idx * 196608) + (slot_b_p * 49152)) + (pair_row_v_5 * 128))];  // #1100
                        ub_to_gm_pad(view_47, ub_prob_2, 64, 256, 0, 0);  // #1101
                        SetFlag<PIPE_MTE3, PIPE_V>(4);  // #1102
                        vcgadd(ub_reduce_pair.ptr(), ub_sc_8.ptr(), (uint8_t)(128), 1, 1, 8);  // #1103
                        const bool cmp_58 = pair_len_11 == 2;  // #2709
                        bool full_pair_2_state = false;  // #2710
                        if (cmp_58) {  // #2722
                            const bool not_8 = !(approx_p);  // #2711
                            bool choice_6_state = false;  // #2712
                            if (not_8) {  // #2718
                                choice_6_state = not_8;  // #2713
                            } else {
                                choice_6_state = ((approx_g_p + 4) + 2) < NG;  // #2714 #2715 #2716 #2717
                            }
                            full_pair_2_state = (bool)(choice_6_state);  // #2719 #2720
                        } else {
                            full_pair_2_state = cmp_58;  // #2721
                        }
                        const bool full_pair_2 = (bool)(full_pair_2_state);  // #2723
                        if (full_pair_2) {  // #2757
                            PipeBarrier<PIPE_V>();  // #1115
                            const Tensor<float, Position::UB> view_48 = ub_reduce_pair;  // #2724
                            const Tensor<float, Position::UB> view_49 = ub_reduce_pair[8];  // #2725
                            vadd(ub_rowsum.ptr(), view_48.ptr(), view_49.ptr(), (uint8_t)(8), (uint8_t)1, (uint8_t)2, (uint8_t)2, (uint8_t)8, (uint8_t)16, (uint8_t)16);  // #1118
                            PipeBarrier<PIPE_V>();  // #1119
                            vcgadd(ub_sum_s.ptr(), ub_rowsum.ptr(), (uint8_t)(8), 1, 1, 8);  // #1120
                            if (approx_p) {  // #2726
                                PipeBarrier<PIPE_V>();  // #1121
                                SetMaskCount();  // #1122
                                SetVectorMask(64);  // #1122
                                vmuls(ub_sum_s.ptr(), ub_sum_s.ptr(), (float)(64.0f), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #1122
                                SetMaskNorm();  // #1122
                                ResetMask();  // #1122
                            }
                            PipeBarrier<PIPE_V>();  // #1124
                            SetMaskCount();  // #1125
                            SetVectorMask(64);  // #1125
                            vadd(ub_bsum_s.ptr(), ub_bsum_s.ptr(), ub_sum_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1125
                            SetMaskNorm();  // #1125
                            ResetMask();  // #1125
                        } else {
                            if (0 < pair_len_11) {  // #2727 #2741
                                PipeBarrier<PIPE_V>();  // #1127
                                bool choice_7_state = false;  // #2728
                                if (approx_p) {  // #2735
                                    const int32_t sub_49 = NG + -1;  // #2731
                                    choice_7_state = (approx_g_p + 4) == sub_49;  // #2729 #2732 #2733
                                } else {
                                    choice_7_state = false;  // #2734
                                }
                                const bool choice_7 = (bool)(choice_7_state);  // #2736
                                if (choice_7) {  // #2740
                                    const Tensor<float, Position::UB> ub_cols_16 = score_tiles.slot[2];  // #2737
                                    vmul(ub_cols_16.ptr(), ub_cols_16.ptr(), ub_len_last.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)16, (uint8_t)16, (uint8_t)0);  // #1136
                                    PipeBarrier<PIPE_V>();  // #1137
                                    vcgadd(ub_reduce.ptr(), ub_cols_16.ptr(), (uint8_t)(64), 1, 1, 16);  // #1138
                                    PipeBarrier<PIPE_V>();  // #1139
                                    vcgadd(ub_sum_s.ptr(), ub_reduce.ptr(), (uint8_t)(8), 1, 1, 8);  // #1140
                                } else {
                                    const Tensor<float, Position::UB> view_50 = ub_reduce_pair;  // #2738
                                    vcgadd(ub_sum_s.ptr(), view_50.ptr(), (uint8_t)(8), 1, 2, 16);  // #1142
                                    if (approx_p) {  // #2739
                                        PipeBarrier<PIPE_V>();  // #1143
                                        SetMaskCount();  // #1144
                                        SetVectorMask(64);  // #1144
                                        vmuls(ub_sum_s.ptr(), ub_sum_s.ptr(), (float)(64.0f), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #1144
                                        SetMaskNorm();  // #1144
                                        ResetMask();  // #1144
                                    }
                                }
                                PipeBarrier<PIPE_V>();  // #1147
                                SetMaskCount();  // #1148
                                SetVectorMask(64);  // #1148
                                vadd(ub_bsum_s.ptr(), ub_bsum_s.ptr(), ub_sum_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1148
                                SetMaskNorm();  // #1148
                                ResetMask();  // #1148
                            }
                            if (1 < pair_len_11) {  // #2742 #2756
                                PipeBarrier<PIPE_V>();  // #1151
                                bool choice_8_state = false;  // #2743
                                if (approx_p) {  // #2750
                                    const int32_t sub_50 = NG + -1;  // #2746
                                    choice_8_state = ((approx_g_p + 4) + 1) == sub_50;  // #2744 #2745 #2747 #2748
                                } else {
                                    choice_8_state = false;  // #2749
                                }
                                const bool choice_8 = (bool)(choice_8_state);  // #2751
                                if (choice_8) {  // #2755
                                    const Tensor<float, Position::UB> ub_cols_17 = score_tiles.slot[2][64];  // #2752
                                    vmul(ub_cols_17.ptr(), ub_cols_17.ptr(), ub_len_last.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)16, (uint8_t)16, (uint8_t)0);  // #1160
                                    PipeBarrier<PIPE_V>();  // #1161
                                    vcgadd(ub_reduce.ptr(), ub_cols_17.ptr(), (uint8_t)(64), 1, 1, 16);  // #1162
                                    PipeBarrier<PIPE_V>();  // #1163
                                    vcgadd(ub_sum_s.ptr(), ub_reduce.ptr(), (uint8_t)(8), 1, 1, 8);  // #1164
                                } else {
                                    const Tensor<float, Position::UB> view_51 = ub_reduce_pair[8];  // #2753
                                    vcgadd(ub_sum_s.ptr(), view_51.ptr(), (uint8_t)(8), 1, 2, 16);  // #1166
                                    if (approx_p) {  // #2754
                                        PipeBarrier<PIPE_V>();  // #1167
                                        SetMaskCount();  // #1168
                                        SetVectorMask(64);  // #1168
                                        vmuls(ub_sum_s.ptr(), ub_sum_s.ptr(), (float)(64.0f), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #1168
                                        SetMaskNorm();  // #1168
                                        ResetMask();  // #1168
                                    }
                                }
                                PipeBarrier<PIPE_V>();  // #1171
                                SetMaskCount();  // #1172
                                SetVectorMask(64);  // #1172
                                vadd(ub_bsum_s.ptr(), ub_bsum_s.ptr(), ub_sum_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1172
                                SetMaskNorm();  // #1172
                                ResetMask();  // #1172
                            }
                        }
                        SetFlag<PIPE_V, PIPE_MTE2>(6);  // #1175
                    }
                    const Tensor<float, Position::UB> row_diff_1 = expdiff_buf.get(beat);  // #2759
                    SetMaskCount();  // #1178
                    SetVectorMask(64);  // #1178
                    vmul(ub_rsum_s.ptr(), ub_rsum_s.ptr(), row_diff_1.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1178
                    SetMaskNorm();  // #1178
                    ResetMask();  // #1178
                    PipeBarrier<PIPE_V>();  // #1179
                    SetMaskCount();  // #1180
                    SetVectorMask(64);  // #1180
                    vadd(ub_rsum_s.ptr(), ub_rsum_s.ptr(), ub_bsum_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1180
                    SetMaskNorm();  // #1180
                    ResetMask();  // #1180
                    VEC_READY<PIPE_MTE3>(1);  // #1181
                    l0ab_cnt_7_state = l0ab_cnt_6;  // #2760
                    l0c_cnt_7_state = l0c_cnt_6;  // #2761
                    l1k_cnt_7_state = l1k_cnt_6;  // #2762
                } else {
                    l0ab_cnt_7_state = l0ab_cnt_carry_1;  // #2763
                    l0c_cnt_7_state = l0c_cnt_carry_1;  // #2764
                    l1k_cnt_7_state = l1k_cnt_carry_1;  // #2765
                }
                const int32_t l0ab_cnt_7 = (int32_t)(l0ab_cnt_7_state);  // #2767
                const int32_t l0c_cnt_7 = (int32_t)(l0c_cnt_7_state);  // #2768
                const int32_t l1k_cnt_7 = (int32_t)(l1k_cnt_7_state);  // #2769
                b_len_p_5_state = b_len_p_4;  // #2770
                l0ab_cnt_8_state = l0ab_cnt_7;  // #2771
                l0c_cnt_8_state = l0c_cnt_7;  // #2772
                l1k_cnt_8_state = l1k_cnt_7;  // #2773
            } else {
                b_len_p_5_state = 0;  // #2774
                l0ab_cnt_8_state = l0ab_cnt_carry_1;  // #2775
                l0c_cnt_8_state = l0c_cnt_carry_1;  // #2776
                l1k_cnt_8_state = l1k_cnt_carry_1;  // #2777
            }
            const int32_t b_len_p_5 = (int32_t)(b_len_p_5_state);  // #2779
            const int32_t l0ab_cnt_8 = (int32_t)(l0ab_cnt_8_state);  // #2780
            const int32_t l0c_cnt_8 = (int32_t)(l0c_cnt_8_state);  // #2781
            const int32_t l1k_cnt_8 = (int32_t)(l1k_cnt_8_state);  // #2782
            const int32_t ready_lengths_1 = ((ready_lengths_carry & 4095) << 4) | b_len_p_5;  // #2783 #2784 #2785
            const int32_t b_len_c = ready_lengths_1 >> 12;  // #2786
            const bool cmp_64 = b_len_c > 0;  // #2787
            int32_t accum_step_2_state = 0;  // #2788
            int32_t l0ab_cnt_15_state = 0;  // #2789
            int32_t l0c_cnt_10_state = 0;  // #2790
            int32_t l1pv_cnt_7_state = 0;  // #2791
            if (cmp_64) {  // #3073
                const int32_t consumed = beat + -3;  // #2792
                const int32_t slot_30 = consumed & 3;  // #2793
                const bool approx_c = consumed < approx_beats_1;  // #2794
                const int32_t e_base_c = Max(0, consumed - approx_beats_1) * 6;  // #2796 #2797 #2798
                const int32_t pair_len_12 = Min(2, b_len_c);  // #2800
                const bool cmp_65 = pair_len_12 > 0;  // #2801
                int32_t l0ab_cnt_10_state = 0;  // #2802
                int32_t l1pv_cnt_2_state = 0;  // #2803
                if (cmp_65) {  // #2880
                    if (approx_c) {  // #2864
                        if (pair_len_12 < 2) {  // #2809 #2812
                            PipeBarrier<PIPE_MTE2>();  // #2811
                        }
                    } else {
                        if (pair_len_12 < 2) {  // #2818 #2821
                            PipeBarrier<PIPE_MTE2>();  // #2820
                        }
                        if (0 < pair_len_12) {  // #2822 #2842
                            const uint16_t load_12 = flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + e_base_c);  // #2825 #2826 #2827 #2828 #2829 #2830 #2831 #2832 #2833 #2834
                        }
                        if (1 < pair_len_12) {  // #2843 #2863
                            const int32_t add_128 = e_base_c + 1;  // #2844
                            const uint16_t load_13 = flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + add_128);  // #2846 #2847 #2848 #2849 #2850 #2851 #2852 #2853 #2854 #2855
                        }
                    }
                    const int32_t l0ab_cnt_9 = (int32_t)(l0ab_cnt_8 + 1);  // #2872 #2873
                    const int32_t l1pv_cnt_1 = (int32_t)(l1pv_cnt_carry_1 + 1);  // #2874 #2875
                    l0ab_cnt_10_state = l0ab_cnt_9;  // #2876
                    l1pv_cnt_2_state = l1pv_cnt_1;  // #2877
                } else {
                    l0ab_cnt_10_state = l0ab_cnt_8;  // #2878
                    l1pv_cnt_2_state = l1pv_cnt_carry_1;  // #2879
                }
                const int32_t l0ab_cnt_10 = (int32_t)(l0ab_cnt_10_state);  // #2881
                const int32_t l1pv_cnt_2 = (int32_t)(l1pv_cnt_2_state);  // #2882
                const int32_t pair_len_13 = Min(2, b_len_c - 2);  // #2883 #2884
                const bool cmp_70 = pair_len_13 > 0;  // #2885
                int32_t l0ab_cnt_12_state = 0;  // #2886
                int32_t l1pv_cnt_4_state = 0;  // #2887
                if (cmp_70) {  // #2964
                    if (approx_c) {  // #2948
                        if (pair_len_13 < 2) {  // #2893 #2896
                            PipeBarrier<PIPE_MTE2>();  // #2895
                        }
                    } else {
                        if (pair_len_13 < 2) {  // #2902 #2905
                            PipeBarrier<PIPE_MTE2>();  // #2904
                        }
                        if (0 < pair_len_13) {  // #2906 #2926
                            const int32_t add_137 = e_base_c + 2;  // #2907
                            const uint16_t load_14 = flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + add_137);  // #2909 #2910 #2911 #2912 #2913 #2914 #2915 #2916 #2917 #2918
                        }
                        if (1 < pair_len_13) {  // #2927 #2947
                            const int32_t add_142 = e_base_c + 3;  // #2928
                            const uint16_t load_15 = flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + add_142);  // #2930 #2931 #2932 #2933 #2934 #2935 #2936 #2937 #2938 #2939
                        }
                    }
                    const int32_t l0ab_cnt_11 = (int32_t)(l0ab_cnt_10 + 1);  // #2956 #2957
                    const int32_t l1pv_cnt_3 = (int32_t)(l1pv_cnt_2 + 1);  // #2958 #2959
                    l0ab_cnt_12_state = l0ab_cnt_11;  // #2960
                    l1pv_cnt_4_state = l1pv_cnt_3;  // #2961
                } else {
                    l0ab_cnt_12_state = l0ab_cnt_10;  // #2962
                    l1pv_cnt_4_state = l1pv_cnt_2;  // #2963
                }
                const int32_t l0ab_cnt_12 = (int32_t)(l0ab_cnt_12_state);  // #2965
                const int32_t l1pv_cnt_4 = (int32_t)(l1pv_cnt_4_state);  // #2966
                const int32_t pair_len_14 = Min(2, b_len_c - 4);  // #2967 #2968
                const bool cmp_75 = pair_len_14 > 0;  // #2969
                int32_t l0ab_cnt_14_state = 0;  // #2970
                int32_t l1pv_cnt_6_state = 0;  // #2971
                if (cmp_75) {  // #3048
                    if (approx_c) {  // #3032
                        if (pair_len_14 < 2) {  // #2977 #2980
                            PipeBarrier<PIPE_MTE2>();  // #2979
                        }
                    } else {
                        if (pair_len_14 < 2) {  // #2986 #2989
                            PipeBarrier<PIPE_MTE2>();  // #2988
                        }
                        if (0 < pair_len_14) {  // #2990 #3010
                            const int32_t add_151 = e_base_c + 4;  // #2991
                            const uint16_t load_16 = flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + add_151);  // #2993 #2994 #2995 #2996 #2997 #2998 #2999 #3000 #3001 #3002
                        }
                        if (1 < pair_len_14) {  // #3011 #3031
                            const int32_t add_156 = e_base_c + 5;  // #3012
                            const uint16_t load_17 = flat_exact_idx.load((((b * ((H * NT_Q) * NPAD)) + (h * (NT_Q * NPAD))) + (q_idx * NPAD)) + add_156);  // #3014 #3015 #3016 #3017 #3018 #3019 #3020 #3021 #3022 #3023
                        }
                    }
                    const int32_t l0ab_cnt_13 = (int32_t)(l0ab_cnt_12 + 1);  // #3040 #3041
                    const int32_t l1pv_cnt_5 = (int32_t)(l1pv_cnt_4 + 1);  // #3042 #3043
                    l0ab_cnt_14_state = l0ab_cnt_13;  // #3044
                    l1pv_cnt_6_state = l1pv_cnt_5;  // #3045
                } else {
                    l0ab_cnt_14_state = l0ab_cnt_12;  // #3046
                    l1pv_cnt_6_state = l1pv_cnt_4;  // #3047
                }
                const int32_t l0ab_cnt_14 = (int32_t)(l0ab_cnt_14_state);  // #3049
                const int32_t l1pv_cnt_6 = (int32_t)(l1pv_cnt_6_state);  // #3050
                WAIT_CUBE<PIPE_S>(2);  // #1496
                const Tensor<float, Position::UB> next_accum = accum_pair.get((accum_step_carry + 1));  // #3053 #3054
                const GMTensor<float> view_74 = pv_ws[(((cube_idx * 65536) + (slot_30 * 16384)) + (sb_row * 128))];  // #1501
                ev_v_mte2_valid_d1_1.wait();  // #1820
                gm_to_ub_pad(next_accum, view_74, 64, 512, 0, 0);  // #1502
                const Tensor<float, Position::UB> tile_1 = expdiff_buf.get(consumed);  // #3055
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #1504
                ResetMask();  // #1504
                vbrcb((__ubuf__ uint32_t*)(ub_expdiff.ptr()), (__ubuf__ uint32_t*)(tile_1.ptr()), (uint16_t)1, (uint16_t)8, (uint8_t)(8));  // #1504
                const Tensor<float, Position::UB> previous_accum = accum_pair.get(accum_step_carry);  // #3056
                const Tensor<float, Position::UB> view_75 = accum_pair.get((accum_step_carry + 1));  // #3057
                const Tensor<float, Position::UB> view_76 = accum_pair.get(accum_step_carry);  // #3058
                ev_mte2_v_ready_d1_1.set();  // #1821
                ev_mte2_v_ready_d1_1.wait();  // #1822
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #1508
                vmla(view_75.ptr(), view_76.ptr(), ub_expdiff.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #1508
                const Tensor<float, Position::UB> view_77 = accum_pair.get((accum_step_carry + 1))[64];  // #3059
                const Tensor<float, Position::UB> view_78 = accum_pair.get(accum_step_carry)[64];  // #3060
                PipeBarrier<PIPE_V>();  // c220: V-V hazard #1511
                vmla(view_77.ptr(), view_78.ptr(), ub_expdiff.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #1511
                const int32_t accum_step_1 = (int32_t)(accum_step_carry + 1);  // #3061 #3062
                const int32_t l0c_cnt_9 = (int32_t)(l0c_cnt_8 + 1);  // #3063 #3064
                accum_step_2_state = accum_step_1;  // #3065
                l0ab_cnt_15_state = l0ab_cnt_14;  // #3066
                l0c_cnt_10_state = l0c_cnt_9;  // #3067
                l1pv_cnt_7_state = l1pv_cnt_6;  // #3068
                ev_v_mte2_valid_d1_1.set();  // #1823
            } else {
                accum_step_2_state = accum_step_carry;  // #3069
                l0ab_cnt_15_state = l0ab_cnt_8;  // #3070
                l0c_cnt_10_state = l0c_cnt_8;  // #3071
                l1pv_cnt_7_state = l1pv_cnt_carry_1;  // #3072
            }
            const int32_t accum_step_2 = (int32_t)(accum_step_2_state);  // #3074
            const int32_t l0ab_cnt_15 = (int32_t)(l0ab_cnt_15_state);  // #3075
            const int32_t l0c_cnt_10 = (int32_t)(l0c_cnt_10_state);  // #3076
            const int32_t l1pv_cnt_7 = (int32_t)(l1pv_cnt_7_state);  // #3077
            accum_step_3_state = accum_step_2;  // #3078
            l0ab_cnt_16_state = l0ab_cnt_15;  // #3079
            l0c_cnt_11_state = l0c_cnt_10;  // #3080
            l1k_cnt_9_state = l1k_cnt_8;  // #3081
            l1pv_cnt_8_state = l1pv_cnt_7;  // #3082
            ready_lengths_2_state = ready_lengths_1;  // #3083
        }
        ev_mte2_v_ready_d0_1.set();  // #1824
        ev_mte2_v_ready_d0_1.wait();  // #1825
        const int32_t accum_step_3 = (int32_t)(accum_step_3_state);  // #3085
        const int32_t l0ab_cnt_16 = (int32_t)(l0ab_cnt_16_state);  // #3086
        const int32_t l0c_cnt_11 = (int32_t)(l0c_cnt_11_state);  // #3087
        const int32_t l1k_cnt_9 = (int32_t)(l1k_cnt_9_state);  // #3088
        const int32_t l1pv_cnt_8 = (int32_t)(l1pv_cnt_8_state);  // #3089
        WaitFlag<PIPE_V, PIPE_MTE2>(4);  // #1522
        WaitFlag<PIPE_V, PIPE_MTE2>(5);  // #1523
        WaitFlag<PIPE_V, PIPE_MTE2>(6);  // #1524
        WaitFlag<PIPE_MTE3, PIPE_V>(4);  // #1525
        PipeBarrier<PIPE_ALL>();  // #3090
        const Tensor<float, Position::UB> accum_ub = accum_pair.get(accum_step_3);  // #3091
        ResetMask();  // #1529
        vbrcb((__ubuf__ uint32_t*)(ub_rowsum.ptr()), (__ubuf__ uint32_t*)(ub_rsum_s.ptr()), (uint16_t)1, (uint16_t)8, (uint8_t)(8));  // #1529
        const Tensor<float, Position::UB> view_79 = accum_pair.get(accum_step_3);  // #3092
        const Tensor<float, Position::UB> view_80 = accum_pair.get(accum_step_3);  // #3093
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #1532
        vdiv(view_79.ptr(), view_80.ptr(), ub_rowsum.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #1532
        const Tensor<float, Position::UB> view_81 = accum_pair.get(accum_step_3)[64];  // #3094
        const Tensor<float, Position::UB> view_82 = accum_pair.get(accum_step_3)[64];  // #3095
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #1535
        vdiv(view_81.ptr(), view_82.ptr(), ub_rowsum.ptr(), (uint8_t)(64), (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)16, (uint8_t)16, (uint8_t)1);  // #1535
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #1536
        vconv_f322bf16a(ub_out.ptr(), accum_ub.ptr(), (uint8_t)(128), (uint16_t)1, (uint16_t)1, (uint16_t)4, (uint16_t)8);  // #1536
        vln(ub_sum_s.ptr(), ub_rsum_s.ptr(), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint8_t)0, (uint8_t)0);  // #1537
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #1538
        vadd(ub_rmax_s.ptr(), ub_rmax_s.ptr(), ub_sum_s.ptr(), (uint8_t)(1), (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0);  // #1538
        PipeBarrier<PIPE_V>();  // c220: V-V hazard #1539
        vmuls(ub_rmax_s.ptr(), ub_rmax_s.ptr(), (float)(1.4426950408889634f), (uint8_t)(1), (uint16_t)1, (uint16_t)1, (uint16_t)0, (uint16_t)0);  // #1539
        ev_v_mte3_ready_1.set();  // #1830
        ev_v_mte3_ready_1.wait();  // #1831
        if (local_valid_m > 0) {  // #1540 #1552
            const int32_t out_row = q_start + sb_row;  // #1541
            const GMTensor<bfloat16_t> view_83 = out[(((b * ((T * H) * 128)) + (out_row * (H * 128))) + (h * 128))];  // #1544
            const Tensor<bfloat16_t, Position::UB> view_84 = probability_tiles.slot[0];  // #1545
            ub_to_gm_pad(view_83, view_84, local_valid_m, 256, 0, ((128 + ((H * 128) - 128)) * 2) - 256);  // #1803 #1804 #1805 #1806 #1807 #1546
            const GMTensor<float> view_85 = lse[(((b * (H * T_LSE)) + (h * T_LSE)) + out_row)];  // #1549
            const Tensor<float, Position::UB> view_86 = ub_rmax_s;  // #1550
            const int32_t sub_70 = 64 - local_valid_m;  // #1811
            const int32_t idiv_65 = sub_70 / 8;  // #3166
            const int32_t idiv_66 = sub_70 % 8;  // #3167
            const bool idiv_67 = idiv_66 != 0;  // #3168
            const bool idiv_68 = sub_70 < 0;  // #3169
            const bool idiv_70 = idiv_68 != false;  // #3171
            const bool idiv_71 = idiv_67 && idiv_70;  // #3172
            const int32_t idiv_72 = (idiv_71) ? (1) : (0);  // #3173
            const int32_t div_1 = idiv_65 - idiv_72;  // #1812
            ub_to_gm_pad(view_85, view_86, 1, local_valid_m * 4, div_1, 0);  // #1808 #1551
        }
        l0ab_cnt_17_state = l0ab_cnt_16;  // #3096
        l0c_cnt_12_state = l0c_cnt_11;  // #3097
        l1k_cnt_10_state = l1k_cnt_9;  // #3098
        l1pv_cnt_9_state = l1pv_cnt_8;  // #3099
        ev_v_mte2_valid_d0_1.set();  // #1826
        ev_mte3_v_valid_1.set();  // #1832
    }
    WAIT_CUBE<PIPE_S>(1);  // #3101
    WAIT_CUBE<PIPE_S>(1);  // #3101
    return;  // #3101
}
