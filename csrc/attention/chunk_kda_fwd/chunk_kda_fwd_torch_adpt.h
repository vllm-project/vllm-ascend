/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */
#ifndef CHUNK_KDA_FWD_TORCH_ADPT_H
#define CHUNK_KDA_FWD_TORCH_ADPT_H

#include <string>
#include <tuple>

#include "attention/kda_torch_adpt_common.h"

namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
chunk_kda_fwd(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &gk,
    const at::Tensor &beta,
    double scale,
    int64_t chunk_size,
    c10::string_view layout,
    const c10::optional<at::Tensor> &initial_state,
    c10::optional<bool> output_final_state,
    c10::optional<at::IntArrayRef> cu_seqlens,
    c10::optional<at::IntArrayRef> chunk_indices,
    c10::optional<bool> return_intermediate,
    c10::optional<bool> safe_gate,
    c10::optional<bool> transpose_state_layout)
{
    std::string layout_str(layout.data(), layout.size());
    TORCH_CHECK(layout_str == "BSND" || layout_str == "BNSD" || layout_str == "TND" || layout_str == "NTD",
                "chunk_kda_fwd: layout must be one of BSND, BNSD, TND, NTD and must be uppercase.");
    TORCH_CHECK(!safe_gate.value_or(false), "chunk_kda_fwd: safe_gate=True is not supported.");
    TORCH_CHECK(!transpose_state_layout.value_or(false),
                "chunk_kda_fwd: transpose_state_layout=True is not supported.");
    TORCH_CHECK(chunk_size == 32 || chunk_size == 64 || chunk_size == 128,
                "chunk_kda_fwd: chunk_size must be 32, 64 or 128.");

    bool is_tnd = layout_str == "TND";
    bool is_ntd = layout_str == "NTD";
    bool is_bsnd = layout_str == "BSND";
    bool is_bnsd = layout_str == "BNSD";
    bool is_rank3 = is_tnd || is_ntd;
    TORCH_CHECK((is_rank3 && q.dim() == 3 && k.dim() == 3 && v.dim() == 3 && gk.dim() == 3 && beta.dim() == 2) ||
                    (!is_rank3 && q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && gk.dim() == 4 && beta.dim() == 3),
                "chunk_kda_fwd: layout/rank mismatch.");
    TORCH_CHECK(q.sizes() == k.sizes(), "chunk_kda_fwd: q and k must have identical shape.");

    auto q_sizes = q.sizes();
    auto v_sizes = v.sizes();
    bool is_internal_layout = is_bnsd || is_ntd;
    int64_t B = is_rank3 ? 1 : q_sizes[0];
    int64_t T = is_tnd ? q_sizes[0] : (is_ntd ? q_sizes[1] : (is_bnsd ? q_sizes[2] : q_sizes[1]));
    int64_t H = is_tnd ? q_sizes[1] : (is_ntd ? q_sizes[0] : (is_bnsd ? q_sizes[1] : q_sizes[2]));
    int64_t K = is_rank3 ? q_sizes[2] : q_sizes[3];
    int64_t HV = is_tnd ? v_sizes[1] : (is_ntd ? v_sizes[0] : (is_bnsd ? v_sizes[1] : v_sizes[2]));
    int64_t V = is_rank3 ? v_sizes[2] : v_sizes[3];
    TORCH_CHECK(H > 0 && HV >= H, "chunk_kda_fwd: H and HV must be positive and H must be <= HV.");
    TORCH_CHECK(H <= 128 && HV <= 128, "chunk_kda_fwd: H and HV must be <= 128.");
    TORCH_CHECK(!is_tnd || H == 1,
                "chunk_kda_fwd: TND layout with H > 1 is not supported; use NTD for multi-head rank3 input.");
    check_kda_cu_seqlens(cu_seqlens, T, "chunk_kda_fwd");
    check_kda_chunk_indices(chunk_indices, cu_seqlens, chunk_size, "chunk_kda_fwd");
    TORCH_CHECK(!cu_seqlens.has_value() || is_rank3 || B == 1,
                "chunk_kda_fwd: rank4 varlen input with cu_seqlens currently requires B=1.");
    TORCH_CHECK(HV % H == 0, "chunk_kda_fwd: HV must be divisible by H.");
    TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
                "chunk_kda_fwd: q/k/v must use float16 or bfloat16.");
    TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(),
                "chunk_kda_fwd: q/k/v dtype must match.");
    TORCH_CHECK((chunk_size == 64 || chunk_size == 128) && K >= 16 && V >= 16 && K % 16 == 0 && V % 16 == 0 && V <= 256 &&
                    K * V >= 4 * 64 * 64 && K * V >= chunk_size * (K + V),
                "chunk_kda_fwd: shape is outside the supported split cube/vector template.");

    int64_t seq_num = get_kda_seq_num(B, cu_seqlens);
    at::Tensor initial_state_tensor = initial_state.value_or(at::Tensor());
    if (initial_state_tensor.defined()) {
        TORCH_CHECK(initial_state_tensor.scalar_type() == at::kFloat,
                    "chunk_kda_fwd: initial_state must be float32 when provided.");
        TORCH_CHECK(initial_state_tensor.dim() == 4 && initial_state_tensor.size(0) == seq_num &&
                        initial_state_tensor.size(1) == HV && initial_state_tensor.size(2) == K &&
                        initial_state_tensor.size(3) == V,
                    "chunk_kda_fwd: initial_state must be [seq_num,Hv,K,V].");
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
    at::Tensor o = at::empty_like(v);
    at::Tensor final_state_work = at::empty({seq_num, HV, K, V}, q.options().dtype(at::kFloat));
    at::Tensor aqk = is_rank3 ? (is_internal_layout ? at::empty({HV, T, chunk_size}, q.options()) :
        at::empty({T, HV, chunk_size}, q.options())) : (is_internal_layout ?
        at::empty({B, HV, T, chunk_size}, q.options()) : at::empty({B, T, HV, chunk_size}, q.options()));
    at::Tensor akk = at::empty_like(aqk);
    at::Tensor w = is_rank3 ? (is_internal_layout ? at::empty({HV, T, K}, q.options()) :
        at::empty({T, HV, K}, q.options())) : (is_internal_layout ?
        at::empty({B, HV, T, K}, q.options()) : at::empty({B, T, HV, K}, q.options()));
    at::Tensor u = at::empty_like(v);
    at::Tensor qg = at::empty_like(w);
    at::Tensor kg = at::empty_like(w);
    at::Tensor v_new = at::empty_like(v);
    at::Tensor h = is_rank3 ? (is_internal_layout ? at::empty({HV, total_chunks, K, V}, q.options()) :
        at::empty({total_chunks, HV, K, V}, q.options())) : (is_internal_layout ?
        at::empty({B, HV, total_chunks, K, V}, q.options()) :
        at::empty({B, total_chunks, HV, K, V}, q.options()));

    bool recompute_output_final_state = true;
    char *layout_cstr = const_cast<char *>(layout_str.c_str());
    EXEC_NPU_CMD(
        aclnnChunkKdaFwd,
        q, k, v, gk, beta, initial_state_tensor,
        cu_seqlens, chunk_indices_for_call,
        layout_cstr, scale, chunk_size, recompute_output_final_state, total_chunks,
        o, final_state_work, aqk, akk, w, u, qg, kg, v_new, h
    );

    at::Tensor final_state = output_final_state.value_or(false) ?
        final_state_work : at::empty({0}, q.options().dtype(at::kFloat));
    at::Tensor empty = at::empty({0}, q.options());
    at::Tensor g = gk.scalar_type() == at::kFloat ? gk : gk.to(at::kFloat);
    at::Tensor initial_state_out = initial_state_tensor.defined() ? initial_state_tensor : empty;
    (void)return_intermediate;
    return std::make_tuple(o, final_state, g, aqk, akk, w, u, qg, kg, v_new, h, initial_state_out);
}

} // namespace vllm_ascend

#endif
