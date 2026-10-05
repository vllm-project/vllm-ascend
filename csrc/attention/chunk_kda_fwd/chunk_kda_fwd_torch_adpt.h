/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */
#ifndef CHUNK_KDA_FWD_TORCH_ADPT_H
#define CHUNK_KDA_FWD_TORCH_ADPT_H

#include <string>
#include <tuple>

#include "../kda_torch_adpt_common.h"

namespace vllm_ascend {

std::tuple<at::Tensor, c10::optional<at::Tensor>, c10::optional<at::Tensor>, at::Tensor, at::Tensor,
           c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>,
           c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>,
           c10::optional<at::Tensor>>
chunk_kda_fwd(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &g,
    const at::Tensor &beta,
    double scale,
    int64_t chunk_size,
    c10::string_view layout,
    const c10::optional<at::Tensor> &initial_state,
    c10::optional<bool> output_final_state,
    c10::optional<at::IntArrayRef> cu_seqlens,
    c10::optional<at::IntArrayRef> chunk_indices,
    c10::optional<bool> safe_gate,
    c10::optional<double> lower_bound,
    c10::optional<bool> use_gate_in_kernel,
    const c10::optional<at::Tensor> &A_log,
    const c10::optional<at::Tensor> &dt_bias,
    c10::optional<bool> disable_recompute,
    c10::optional<bool> return_intermediate_states,
    c10::optional<bool> state_v_first)
{
    std::string layout_str(layout.data(), layout.size());
    TORCH_CHECK(layout_str == "BSND" || layout_str == "BNSD" || layout_str == "TND" || layout_str == "NTD",
                "chunk_kda_fwd: layout must be one of BSND, BNSD, TND, NTD and must be uppercase.");
    TORCH_CHECK(chunk_size == 64 || chunk_size == 128, "chunk_kda_fwd: chunk_size must be 64 or 128.");
    bool output_final_state_ = output_final_state.value_or(false);
    bool safe_gate_ = safe_gate.value_or(false);
    double lower_bound_ = lower_bound.value_or(-5.0);
    bool use_gate_in_kernel_ = use_gate_in_kernel.value_or(false);
    bool disable_recompute_ = disable_recompute.value_or(false);
    bool return_intermediate_states_ = return_intermediate_states.value_or(false);
    bool state_v_first_ = state_v_first.value_or(false);

    bool is_tnd = layout_str == "TND";
    bool is_ntd = layout_str == "NTD";
    bool is_bsnd = layout_str == "BSND";
    bool is_bnsd = layout_str == "BNSD";
    bool is_rank3 = is_tnd || is_ntd;
    TORCH_CHECK(
        (is_rank3 && q.dim() == 3 && k.dim() == 3 && v.dim() == 3 && g.dim() == 3 && beta.dim() == 2) ||
            (!is_rank3 && q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && g.dim() == 4 && beta.dim() == 3),
        "chunk_kda_fwd: input ranks do not match layout.");
    TORCH_CHECK(q.sizes() == k.sizes(), "chunk_kda_fwd: q and k must have identical shape.");

    auto q_sizes = q.sizes();
    auto v_sizes = v.sizes();
    int64_t B = is_rank3 ? 1 : q_sizes[0];
    int64_t T = is_tnd ? q_sizes[0] : (is_ntd ? q_sizes[1] : (is_bnsd ? q_sizes[2] : q_sizes[1]));
    int64_t H = is_tnd ? q_sizes[1] : (is_ntd ? q_sizes[0] : (is_bnsd ? q_sizes[1] : q_sizes[2]));
    int64_t K = is_rank3 ? q_sizes[2] : q_sizes[3];
    int64_t HV = is_tnd ? v_sizes[1] : (is_ntd ? v_sizes[0] : (is_bnsd ? v_sizes[1] : v_sizes[2]));
    int64_t V = is_rank3 ? v_sizes[2] : v_sizes[3];
    TORCH_CHECK(H > 0 && HV >= H && HV % H == 0 && H <= 128 && HV <= 128,
                "chunk_kda_fwd: H/HV must satisfy 0 < H <= HV <= 128 and HV % H == 0.");
    TORCH_CHECK(K >= 16 && K <= 256 && K % 16 == 0 && V >= 16 && V <= 256 && V % 16 == 0,
                "chunk_kda_fwd: K/V must be multiples of 16 and no greater than 256.");
    TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
                "chunk_kda_fwd: q/k/v must use float16 or bfloat16.");
    TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(),
                "chunk_kda_fwd: q/k/v dtype must match.");
    TORCH_CHECK((g.scalar_type() == at::kFloat || g.scalar_type() == at::kBFloat16) &&
                    (beta.scalar_type() == at::kFloat || beta.scalar_type() == at::kBFloat16),
                "chunk_kda_fwd: g and beta must be float32 or bfloat16.");

    check_kda_cu_seqlens(cu_seqlens, T, "chunk_kda_fwd");
    check_kda_chunk_indices(chunk_indices, cu_seqlens, chunk_size, "chunk_kda_fwd");
    TORCH_CHECK(!cu_seqlens.has_value() || is_rank3 || B == 1,
                "chunk_kda_fwd: rank4 varlen input requires B=1.");
    auto g_sizes = g.sizes();
    TORCH_CHECK((is_tnd && beta.sizes()[0] == T && beta.sizes()[1] == HV) ||
                    (is_ntd && beta.sizes()[0] == HV && beta.sizes()[1] == T) ||
                    (is_bsnd && beta.sizes()[0] == B && beta.sizes()[1] == T && beta.sizes()[2] == HV) ||
                    (is_bnsd && beta.sizes()[0] == B && beta.sizes()[1] == HV && beta.sizes()[2] == T),
                "chunk_kda_fwd: beta shape mismatch.");
    TORCH_CHECK((is_tnd && v_sizes == at::IntArrayRef({T, HV, V}) && g_sizes == at::IntArrayRef({T, HV, K})) ||
                    (is_ntd && v_sizes == at::IntArrayRef({HV, T, V}) &&
                               g_sizes == at::IntArrayRef({HV, T, K})) ||
                    (is_bsnd && v_sizes == at::IntArrayRef({B, T, HV, V}) &&
                                g_sizes == at::IntArrayRef({B, T, HV, K})) ||
                    (is_bnsd && v_sizes == at::IntArrayRef({B, HV, T, V}) &&
                                g_sizes == at::IntArrayRef({B, HV, T, K})),
                "chunk_kda_fwd: v/g shapes do not match layout.");

    int64_t seq_num = get_kda_seq_num(B, cu_seqlens);
    TORCH_CHECK(seq_num <= 1024, "chunk_kda_fwd: at most 1024 sequences are supported.");
    std::vector<int64_t> state_shape = state_v_first_ ? std::vector<int64_t>{seq_num, HV, V, K}
                                                      : std::vector<int64_t>{seq_num, HV, K, V};
    if (initial_state.has_value() && initial_state->defined()) {
        TORCH_CHECK(initial_state->scalar_type() == at::kFloat,
                    "chunk_kda_fwd: initial_state must be float32 when provided.");
        TORCH_CHECK(initial_state->sizes() == at::IntArrayRef(state_shape),
                    "chunk_kda_fwd: initial_state shape does not match state_v_first.");
    }
    if (use_gate_in_kernel_) {
        TORCH_CHECK(A_log.has_value() && A_log->defined() && A_log->scalar_type() == at::kFloat &&
                        A_log->sizes() == at::IntArrayRef({HV}),
                    "chunk_kda_fwd: A_log must be float32 [HV] when use_gate_in_kernel=True.");
        if (dt_bias.has_value() && dt_bias->defined()) {
            TORCH_CHECK(dt_bias->scalar_type() == at::kFloat && dt_bias->sizes() == at::IntArrayRef({HV * K}),
                        "chunk_kda_fwd: dt_bias must be float32 [HV*K].");
        }
        if (safe_gate_) {
            TORCH_CHECK(lower_bound_ >= -5.0 && lower_bound_ < 0.0,
                        "chunk_kda_fwd: lower_bound must be in [-5, 0).");
        }
    }

    std::vector<int64_t> generated_chunk_indices;
    c10::optional<at::IntArrayRef> chunk_indices_for_call;
    if (chunk_indices.has_value()) {
        chunk_indices_for_call = chunk_indices.value();
    } else if (cu_seqlens.has_value()) {
        generated_chunk_indices = build_kda_chunk_indices(cu_seqlens.value(), chunk_size);
        chunk_indices_for_call = at::IntArrayRef(generated_chunk_indices);
    } else {
        chunk_indices_for_call = c10::nullopt;
    }

    int64_t total_chunks = get_kda_total_chunks(B, T, chunk_size, cu_seqlens, chunk_indices_for_call);
    std::vector<int64_t> attn_shape = is_rank3 ? std::vector<int64_t>{T, HV, V}
                                               : std::vector<int64_t>{B, T, HV, V};
    std::vector<int64_t> matrix_shape = is_rank3 ? std::vector<int64_t>{HV, T, chunk_size}
                                                 : std::vector<int64_t>{B, HV, T, chunk_size};
    std::vector<int64_t> k_shape = is_rank3 ? std::vector<int64_t>{HV, T, K}
                                            : std::vector<int64_t>{B, HV, T, K};
    std::vector<int64_t> v_shape = is_rank3 ? std::vector<int64_t>{HV, T, V}
                                            : std::vector<int64_t>{B, HV, T, V};
    std::vector<int64_t> h_shape =
        is_rank3 ? (state_v_first_ ? std::vector<int64_t>{total_chunks, HV, V, K}
                                   : std::vector<int64_t>{total_chunks, HV, K, V})
                 : (state_v_first_ ? std::vector<int64_t>{B, total_chunks, HV, V, K}
                                   : std::vector<int64_t>{B, total_chunks, HV, K, V});

    at::Tensor attn_out = at::empty(attn_shape, v.options());
    at::Tensor final_state =
        output_final_state_ ? at::empty(state_shape, q.options().dtype(at::kFloat)) : at::Tensor();
    at::Tensor gk_out = (!use_gate_in_kernel_ || disable_recompute_)
                            ? at::empty(k_shape, q.options().dtype(at::kFloat))
                            : at::Tensor();
    at::Tensor aqk = at::empty(matrix_shape, q.options());
    at::Tensor akk = at::empty_like(aqk);
    at::Tensor w = disable_recompute_ ? at::empty(k_shape, q.options()) : at::Tensor();
    at::Tensor u = disable_recompute_ ? at::empty(v_shape, q.options()) : at::Tensor();
    at::Tensor qg = disable_recompute_ ? at::empty(k_shape, q.options()) : at::Tensor();
    at::Tensor kg = disable_recompute_ ? at::empty(k_shape, q.options()) : at::Tensor();
    at::Tensor v_new = disable_recompute_ ? at::empty(v_shape, q.options()) : at::Tensor();
    at::Tensor h = (disable_recompute_ || return_intermediate_states_)
                       ? at::empty(h_shape, q.options())
                       : at::Tensor();

    const at::Tensor &initial_state_ = c10::value_or_else(initial_state, [] { return at::Tensor(); });
    const at::Tensor &A_log_ = c10::value_or_else(A_log, [] { return at::Tensor(); });
    const at::Tensor &dt_bias_ = c10::value_or_else(dt_bias, [] { return at::Tensor(); });
    const char *layout_cstr = layout_str.c_str();
    EXEC_NPU_CMD(
        aclnnChunkKdaFwd,
        q, k, v, g, beta, A_log_, dt_bias_, initial_state_, cu_seqlens, chunk_indices_for_call,
        layout_cstr, scale, chunk_size, safe_gate_, lower_bound_, use_gate_in_kernel_, state_v_first_,
        attn_out, final_state, gk_out, aqk, akk, w, u, qg, kg, v_new, h
    );

    c10::optional<at::Tensor> final_state_out =
        final_state.defined() ? c10::optional<at::Tensor>(final_state) : c10::nullopt;
    c10::optional<at::Tensor> gk_optional =
        gk_out.defined() ? c10::optional<at::Tensor>(gk_out) : c10::nullopt;
    c10::optional<at::Tensor> w_optional = w.defined() ? c10::optional<at::Tensor>(w) : c10::nullopt;
    c10::optional<at::Tensor> u_optional = u.defined() ? c10::optional<at::Tensor>(u) : c10::nullopt;
    c10::optional<at::Tensor> qg_optional = qg.defined() ? c10::optional<at::Tensor>(qg) : c10::nullopt;
    c10::optional<at::Tensor> kg_optional = kg.defined() ? c10::optional<at::Tensor>(kg) : c10::nullopt;
    c10::optional<at::Tensor> v_new_optional =
        v_new.defined() ? c10::optional<at::Tensor>(v_new) : c10::nullopt;
    c10::optional<at::Tensor> h_optional = h.defined() ? c10::optional<at::Tensor>(h) : c10::nullopt;
    c10::optional<at::Tensor> initial_state_out =
        initial_state.has_value() && initial_state->defined() ? initial_state : c10::nullopt;
    return std::make_tuple(attn_out, final_state_out, gk_optional, aqk, akk, w_optional, u_optional,
                           qg_optional, kg_optional, v_new_optional, h_optional, initial_state_out);
}
} // namespace vllm_ascend

#endif
