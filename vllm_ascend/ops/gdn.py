#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import torch_npu
from einops import rearrange
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fla.ops import (
    fused_recurrent_gated_delta_rule,
)
from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd
from vllm.model_executor.layers.mamba.gdn_linear_attn import GatedDeltaNetAttention
from vllm.triton_utils import triton
from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.attention.utils import maybe_save_kv_layer_to_connector
from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule
from vllm_ascend.ops.triton.fla.fused_qkvzba_split_reshape import fused_qkvzba_split_reshape_cat
from vllm_ascend.ops.triton.fla.sigmoid_gating import fused_sigmoid_gating_delta_rule_update
from vllm_ascend.ops.triton.fla.utils import clear_ssm_states
from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch
from vllm_ascend.ops.triton.mamba.causal_conv1d import causal_conv1d_update_npu
from vllm_ascend.utils import enable_sp
from vllm_ascend.compilation.acl_graph import (
    get_draft_graph_params,
    get_graph_params,
)
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.utils import weak_ref_tensors


def to_int64_tuple(tensor: torch.Tensor) -> tuple[int, ...]:
    tensor = tensor.to(torch.int64)
    if tensor.dim() == 0:
        return (tensor.item(),)
    return tuple(tensor.tolist())


def _require_non_spec_prefill_fallback_meta(attn_metadata, field_name: str, sub_field_name: str):
    fallback_meta = getattr(attn_metadata, field_name, None)
    if fallback_meta is None:
        raise RuntimeError(
                f"Expected attn_metadata.non_spec_prefill_fallback_meta.{sub_field_name} for patched GDN non-spec prefill path."
        )
    return fallback_meta


def get_non_spec_causal_conv1d_host_args(attn_metadata) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    fallback_meta = _require_non_spec_prefill_fallback_meta(attn_metadata, "non_spec_prefill_fallback_meta",
                                                            "causal_conv1d")
    causal_conv1d_meta = fallback_meta.causal_conv1d
    return (
        to_int64_tuple(causal_conv1d_meta.query_start_loc_cpu),
        to_int64_tuple(causal_conv1d_meta.cache_indices_cpu),
        to_int64_tuple(causal_conv1d_meta.has_initial_state_cpu),
    )


def get_non_spec_decode_causal_conv1d_host_args(attn_metadata) -> tuple[
    tuple[int, ...], tuple[int, ...]]:
    fallback_meta = _require_non_spec_prefill_fallback_meta(attn_metadata, "non_spec_decode_fallback_meta",
                                                            "causal_conv1d")
    causal_conv1d_meta = fallback_meta.causal_conv1d
    return (
        to_int64_tuple(causal_conv1d_meta.query_start_loc_cpu),
        to_int64_tuple(causal_conv1d_meta.cache_indices_cpu),
    )


def get_spec_causal_conv1d_host_args(attn_metadata) -> tuple[
    tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    fallback_meta = _require_non_spec_prefill_fallback_meta(attn_metadata, "spec_decode_fallback_meta",
                                                            "spec_causal_conv1d")
    causal_conv1d_meta = fallback_meta.spec_causal_conv1d

    return (
        to_int64_tuple(causal_conv1d_meta.query_start_loc_cpu),
        to_int64_tuple(causal_conv1d_meta.cache_indices_cpu),
        to_int64_tuple(causal_conv1d_meta.num_accepted_tokens_cpu)
    )

def _pad_conv1d_host_args_to_capture(
        qsl_host: tuple[int, ...],
        cidx_host: tuple[int, ...],
        num_accepted_host: tuple[int, ...] | tuple,
        cap_x_dim0: int,
        q_per_seq: int,
        with_num_accepted: bool,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...] | tuple]:
    """把 runtime host 参数补尾部虚拟 req, 对齐到 capture 期 x.shape[0].

    capture 期 ACL Graph 记录了 ``cuSeqlen = mixed_qkv.shape[0]`` (= ``cap_x_dim0``),
    tiling 强校验 ``qsl[last] == cuSeqlen``. runtime 实际 req 数可能少于 capture
    期(典型场景: FIA 给 query_start_loc 补了一个虚拟 req, 但 vllm 把它判定成
    非 spec/非 decode, 于是 spec/non_spec_decode host 参数只覆盖真实 req).
    这里按 ``q_per_seq`` 步长补虚拟 req, 让 ``qsl[last] == cap_x_dim0``,
    虚拟 req 的 cache_index 设为 ``PAD_SLOT_ID``, kernel 走到时会跳过 state 写回.
    """
    if not qsl_host:
        return qsl_host, cidx_host, num_accepted_host
    runtime_qsl_last = int(qsl_host[-1])
    pad_tokens = cap_x_dim0 - runtime_qsl_last
    if pad_tokens <= 0 or q_per_seq <= 0:
        return qsl_host, cidx_host, num_accepted_host
    pad_seqs = pad_tokens // q_per_seq
    if pad_seqs <= 0:
        return qsl_host, cidx_host, num_accepted_host
    extra_qsl = tuple(runtime_qsl_last + (i + 1) * q_per_seq for i in range(pad_seqs))
    qsl_host = qsl_host + extra_qsl
    cidx_host = cidx_host + (PAD_SLOT_ID,) * pad_seqs
    if with_num_accepted:
        num_accepted_host = tuple(num_accepted_host) + (1,) * pad_seqs
    return qsl_host, cidx_host, num_accepted_host


def update_conv1d_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        is_draft_model=False,
        draft_attn_metadatas=None,
):
    """更新conv1d的Host侧参数"""
    from vllm_ascend.compilation.acl_graph import get_graph_params, get_draft_graph_params

    graph_params = get_draft_graph_params() if is_draft_model else get_graph_params()

    if num_tokens not in graph_params.conv1d_params or len(graph_params.conv1d_params[num_tokens]) == 0:
        return

    attn_metadata = forward_context.attn_metadata
    if is_draft_model and draft_attn_metadatas is not None:
        attn_metadata = draft_attn_metadatas

    with torch.npu.stream(update_stream):
        for param, handle, event in zip(
                graph_params.conv1d_params[num_tokens],
                graph_params.conv1d_handles[num_tokens],
                graph_params.conv1d_events[num_tokens],
        ):
            # 解包捕获时的参数
            (output, mixed_qkv, conv_weights_T, conv_state, bias, activation_num,
             pad_slot_id, run_mode, branch, layer_prefix, _, _, _, q_per_seq) = param

            new_query_start_loc = ()
            new_cache_indices = ()
            new_num_accepted = ()

            if run_mode == 1 and attn_metadata is not None:
                # get gdn metadata by captrued layer_prefix 
                meta = attn_metadata
                if isinstance(meta, dict):
                    meta = meta.get(layer_prefix, None)
                    assert isinstance(meta, GDNAttentionMetadata)

                if meta is None:
                    continue

                cap_x_dim0 = int(mixed_qkv.size(0))
                if "spec" == branch:
                    qsl_host, cidx_host, num_accepted_host = get_spec_causal_conv1d_host_args(meta)
                    new_query_start_loc, new_cache_indices, new_num_accepted = _pad_conv1d_host_args_to_capture(
                            qsl_host, cidx_host, num_accepted_host,
                            cap_x_dim0=cap_x_dim0,
                            q_per_seq=q_per_seq,
                            with_num_accepted=True,
                    )
                elif "non_spec_decode" == branch:
                    non_sdq_host, non_sd_cidx_host = get_non_spec_decode_causal_conv1d_host_args(meta)
                    new_query_start_loc, new_cache_indices, _ = _pad_conv1d_host_args_to_capture(
                            non_sdq_host, non_sd_cidx_host, (),
                            cap_x_dim0=cap_x_dim0,
                            q_per_seq=q_per_seq,
                            with_num_accepted=False,
                    )
                    new_num_accepted = ()

            torch.npu.graph_task_update_begin(update_stream, handle)
            torch.ops._C_ascend.npu_causal_conv1d_custom(
                    output,
                    mixed_qkv,
                    conv_weights_T,
                    conv_state=conv_state,
                    bias_opt=bias,
                    query_start_loc_opt=new_query_start_loc,
                    cache_indices_opt=new_cache_indices,
                    initial_state_mode_opt=(),
                    num_accepted_tokens_opt=new_num_accepted,
                    activation_mode=activation_num,
                    pad_slot_id=pad_slot_id,
                    run_mode=run_mode,
            )
            torch.npu.graph_task_update_end(update_stream)
            event.record(update_stream)


def get_non_spec_chunked_prefill_meta(attn_metadata):
    fallback_meta = _require_non_spec_prefill_fallback_meta(attn_metadata, "non_spec_prefill_fallback_meta", "chunk")
    return fallback_meta.chunk


class AscendGatedDeltaNetAttention(GatedDeltaNetAttention):
    def forward(
            self,
            hidden_states: torch.Tensor,
            output: torch.Tensor,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        if not self.gqa_interleaved_layout:
            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            num_tokens = mixed_qkvz.size(0)
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            ba, _ = self.in_proj_ba(hidden_states)
            b, a = ba.chunk(2, dim=-1)

            b = b.contiguous()
            a = a.contiguous()
        else:
            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
            projected_states_ba, _ = self.in_proj_ba(hidden_states)
            num_tokens = projected_states_qkvz.size(0)

            mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat(
                    projected_states_qkvz,
                    projected_states_ba,
                    triton.cdiv(self.num_k_heads, self.tp_size),
                    triton.cdiv(self.num_v_heads, self.tp_size),
                    self.head_k_dim,
                    self.head_v_dim,
            )

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        # Note: we should not use torch.empty here like other attention backends,
        # see discussions in https://github.com/vllm-project/vllm/pull/28182
        core_attn_out = torch.zeros(
                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
        )

        torch.ops.vllm.gdn_attention_core(
                mixed_qkv,
                b,
                a,
                core_attn_out,
                self.prefix,
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        maybe_save_kv_layer_to_connector("", [])
        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)

    def _forward_core(
            self,
            mixed_qkv: torch.Tensor,
            b: torch.Tensor,
            a: torch.Tensor,
            core_attn_out: torch.Tensor,
    ):
        """
        Core attention computation (called by custom op).
        """
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
            # V1 profile run
            return

        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        conv_state = self_kv_cache[0].transpose(-1, -2)
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        if not enable_sp():
            mixed_qkv = mixed_qkv[:num_actual_tokens]
            b = b[:num_actual_tokens]
            a = a[:num_actual_tokens]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        # 1.1: Process the multi-query part
        if spec_sequence_masks is not None:
            conv_weights_T = conv_weights.transpose(0, 1)
            activation_num = 1 if self.activation else 0
            (
                spec_query_start_loc_opt,
                spec_cache_indices_opt,
                num_accepted_tokens_cpu
            ) = get_spec_causal_conv1d_host_args(attn_metadata)
            # capturing branch for conv1d update
            if _EXTRA_CTX.capturing:
                stream = torch_npu.npu.current_stream()
                event = torch.npu.ExternalEvent()
                event.wait(stream)
                event.reset(stream)
                graph_params = get_graph_params() if not _EXTRA_CTX.is_draft_model else get_draft_graph_params()
                graph_params.conv1d_events[num_actual_tokens].append(event)

                output_spec = torch.empty_like(mixed_qkv_spec)
                # capture 时每条 spec req 的 query 长度 (= num_spec + 1)。update 时
                # 用它把 host 参数对齐到 capture 的 x.shape[0]，避免 runtime 实际
                # spec 序列数 < capture 期 spec 序列数时 tiling 校验失败。
                spec_q_per_seq = int(attn_metadata.spec_state_indices_tensor.size(-1))
                # 保存参数引用（tensor用weak_ref，Host变量直接保存tuple）
                graph_params.conv1d_params[num_actual_tokens].append((
                    weak_ref_tensors(output_spec),
                    weak_ref_tensors(mixed_qkv_spec),
                    weak_ref_tensors(conv_weights_T),
                    weak_ref_tensors(self_kv_cache[0]),
                    self.conv1d.bias,
                    activation_num,
                    PAD_SLOT_ID,
                    1,  # run_mode
                    "spec",
                    self.prefix,
                    spec_query_start_loc_opt,
                    spec_cache_indices_opt,
                    num_accepted_tokens_cpu,
                    spec_q_per_seq
                ))

                torch.npu.graph_task_group_begin(stream)
                torch.ops._C_ascend.npu_causal_conv1d_custom(
                        output_spec,
                        mixed_qkv_spec,
                        conv_weights_T,
                        conv_state=self_kv_cache[0],
                        bias_opt=self.conv1d.bias,
                        query_start_loc_opt=spec_query_start_loc_opt,
                        cache_indices_opt=spec_cache_indices_opt,
                        initial_state_mode_opt=(),
                        num_accepted_tokens_opt=num_accepted_tokens_cpu,
                        activation_mode=activation_num,
                        pad_slot_id=PAD_SLOT_ID,
                        run_mode=1,
                )
                handle = torch.npu.graph_task_group_end(stream)
                graph_params.conv1d_handles[num_actual_tokens].append(handle)
                mixed_qkv_spec = output_spec
            else:
                # for enforce eager
                output_spec = torch.empty_like(mixed_qkv_spec)
                torch.ops._C_ascend.npu_causal_conv1d_custom(
                        output_spec,
                        mixed_qkv_spec,
                        conv_weights_T,
                        conv_state=self_kv_cache[0],
                        bias_opt=self.conv1d.bias,
                        query_start_loc_opt=spec_query_start_loc_opt,
                        cache_indices_opt=spec_cache_indices_opt,
                        initial_state_mode_opt=(),
                        num_accepted_tokens_opt=num_accepted_tokens_cpu,
                        activation_mode=activation_num,
                        pad_slot_id=PAD_SLOT_ID,
                        run_mode=1,
                )
                mixed_qkv_spec = output_spec

        # 1.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            if mixed_qkv_non_spec is not None:
                conv_weights_T = conv_weights.transpose(0, 1)
                activation_num = 1 if self.activation else 0
                (
                    query_start_loc_opt,
                    cache_indices_opt,
                    initial_state_mode_opt,
                ) = get_non_spec_causal_conv1d_host_args(attn_metadata)
                mixed_qkv_non_spec_output = torch.empty_like(mixed_qkv_non_spec)
                torch.ops._C_ascend.npu_causal_conv1d_custom(
                        mixed_qkv_non_spec_output,
                        mixed_qkv_non_spec,
                        conv_weights_T,
                        conv_state=self_kv_cache[0],
                        bias_opt=self.conv1d.bias,
                        query_start_loc_opt=query_start_loc_opt,
                        cache_indices_opt=cache_indices_opt,
                        initial_state_mode_opt=initial_state_mode_opt,
                        num_accepted_tokens_opt=[],
                        activation_mode=activation_num,
                        pad_slot_id=PAD_SLOT_ID,
                        run_mode=0,
                )
                mixed_qkv_non_spec = mixed_qkv_non_spec_output
        elif attn_metadata.num_decodes > 0:
            conv_weights_T = conv_weights.transpose(0, 1)
            activation_num = 1 if self.activation else 0
            non_spec_qsl_host, non_spec_ci_host = get_non_spec_decode_causal_conv1d_host_args(attn_metadata)
            # 图捕获分支
            if _EXTRA_CTX.capturing:
                stream = torch_npu.npu.current_stream()
                event = torch.npu.ExternalEvent()
                event.wait(stream)
                event.reset(stream)
                graph_params = get_graph_params() if not _EXTRA_CTX.is_draft_model else get_draft_graph_params()
                graph_params.conv1d_events[num_actual_tokens].append(event)

                output_non_spec = torch.empty_like(mixed_qkv_non_spec)
                # non_spec_decode 每条 req 固定 1 个 token，update 时按 1 个步长补尾部虚拟 req。
                non_spec_q_per_seq = 1
                # 保存参数引用
                graph_params.conv1d_params[num_actual_tokens].append((
                    weak_ref_tensors(output_non_spec),
                    weak_ref_tensors(mixed_qkv_non_spec),
                    weak_ref_tensors(conv_weights_T),
                    weak_ref_tensors(self_kv_cache[0]),
                    self.conv1d.bias,
                    activation_num,
                    PAD_SLOT_ID,
                    1,  # run_mode
                    "non_spec_decode",
                    self.prefix,
                    non_spec_qsl_host,
                    non_spec_ci_host,
                    [],
                    non_spec_q_per_seq
                ))

                torch.npu.graph_task_group_begin(stream)
                torch.ops._C_ascend.npu_causal_conv1d_custom(
                        output_non_spec,
                        mixed_qkv_non_spec,
                        conv_weights_T,
                        conv_state=self_kv_cache[0],
                        bias_opt=self.conv1d.bias,
                        query_start_loc_opt=non_spec_qsl_host,
                        cache_indices_opt=non_spec_ci_host,
                        initial_state_mode_opt=(),
                        num_accepted_tokens_opt=[],
                        activation_mode=activation_num,
                        pad_slot_id=PAD_SLOT_ID,
                        run_mode=1,
                )
                handle = torch.npu.graph_task_group_end(stream)
                graph_params.conv1d_handles[num_actual_tokens].append(handle)
                mixed_qkv_non_spec = output_non_spec
            else:
                output_non_spec = torch.empty_like(mixed_qkv_non_spec)
                torch.ops._C_ascend.npu_causal_conv1d_custom(
                        output_non_spec,
                        mixed_qkv_non_spec,
                        conv_weights_T,
                        conv_state=self_kv_cache[0],
                        bias_opt=self.conv1d.bias,
                        query_start_loc_opt=to_int64_tuple(non_spec_query_start_loc[:num_actual_tokens + 1]),
                        cache_indices_opt=to_int64_tuple(non_spec_state_indices_tensor[:num_actual_tokens]),
                        initial_state_mode_opt=to_int64_tuple((non_spec_state_indices_tensor != PAD_SLOT_ID).to(torch.int32)),
                        num_accepted_tokens_opt=[],
                        activation_mode=activation_num,
                        pad_slot_id=PAD_SLOT_ID,
                        run_mode=1,
                )
                mixed_qkv_non_spec = output_non_spec
        else:
            mixed_qkv_non_spec = None

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(mixed_qkv_non_spec)

        # 2. Recurrent attention
        if self.gqa_interleaved_layout:
            # Qwen3Next: torch_npu ops support float16/bf16 ssm_state.
            # g/beta are needed for both spec-decode and decode, so compute unconditionally.
            g, beta = fused_gdn_gating_patch(self.A_log, a, b, self.dt_bias)
            if spec_sequence_masks is not None:
                if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                    g_spec = g
                    beta_spec = beta
                    g_non_spec = None
                    beta_non_spec = None
                else:
                    g_spec = g.index_select(1, spec_token_indx)
                    beta_spec = beta.index_select(1, spec_token_indx)
                    g_non_spec = g.index_select(1, non_spec_token_indx)
                    beta_non_spec = beta.index_select(1, non_spec_token_indx)
            else:
                g_spec = None
                beta_spec = None
                g_non_spec = g
                beta_non_spec = beta

            # 2.1: Process the multi-query part
            if spec_sequence_masks is not None:
                cu_seqlens = spec_query_start_loc[: attn_metadata.num_spec_decodes + 1]
                actual_seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                query_spec = l2norm_fwd(query_spec)
                key_spec = l2norm_fwd(key_spec)
                core_attn_out_spec = torch_npu.npu_recurrent_gated_delta_rule(
                        query=query_spec.squeeze(0),
                        key=key_spec.squeeze(0),
                        value=value_spec.squeeze(0),
                        g=g_spec.squeeze(0),
                        beta=beta_spec.squeeze(0),
                        state=ssm_state,
                        scale=key_spec.shape[-1] ** -0.5,
                        actual_seq_lengths=actual_seq_lengths,
                        ssm_state_indices=spec_state_indices_tensor.flatten(),
                        num_accepted_tokens=num_accepted_tokens.to(torch.int32),
                ).unsqueeze(0)
            else:
                core_attn_out_spec, last_recurrent_state = None, None

            # 2.2: Process the remaining part
            if attn_metadata.num_prefills > 0:
                initial_state = ssm_state[non_spec_state_indices_tensor].transpose(-1, -2).contiguous()
                clear_ssm_states(initial_state, has_initial_state)
                (core_attn_out_non_spec, last_recurrent_state) = chunk_gated_delta_rule(
                        q=query_non_spec,
                        k=key_non_spec,
                        v=value_non_spec,
                        g=g_non_spec,
                        beta=beta_non_spec,
                        initial_state=initial_state,
                        output_final_state=True,
                        cu_seqlens=non_spec_query_start_loc,
                        prebuilt_meta=get_non_spec_chunked_prefill_meta(attn_metadata),
                        head_first=False,
                        use_qk_l2norm_in_kernel=True,
                )
                ssm_state[non_spec_state_indices_tensor] = (
                    last_recurrent_state.transpose(-1, -2).contiguous().to(ssm_state.dtype)
                )
            elif attn_metadata.num_decodes > 0:
                cu_seqlens = non_spec_query_start_loc[: attn_metadata.num_decodes + 1]
                actual_seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                query_non_spec = l2norm_fwd(query_non_spec)
                key_non_spec = l2norm_fwd(key_non_spec)
                core_attn_out_non_spec = torch_npu.npu_recurrent_gated_delta_rule(
                        query=query_non_spec.squeeze(0),
                        key=key_non_spec.squeeze(0),
                        value=value_non_spec.squeeze(0),
                        g=g_non_spec.squeeze(0) if g_non_spec is not None else g_non_spec,
                        beta=beta_non_spec.squeeze(0) if beta_non_spec is not None else beta_non_spec,
                        state=ssm_state,
                        scale=key_non_spec.shape[-1] ** -0.5,
                        actual_seq_lengths=actual_seq_lengths,
                        ssm_state_indices=non_spec_state_indices_tensor,
                ).unsqueeze(0)
            else:
                core_attn_out_non_spec, last_recurrent_state = None, None
        else:
            # Qwen3.5: torch_npu ops do not support float32 ssm_state, use FLA ops instead.
            # NOTE: Once torch_npu supports float32 ssm_state, this branch can be removed.
            if attn_metadata.num_prefills > 0 or spec_sequence_masks is not None:
                g, beta = fused_gdn_gating_patch(self.A_log, a, b, self.dt_bias)
                if spec_sequence_masks is not None:
                    if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                        g_spec = g
                        beta_spec = beta
                        g_non_spec = None
                        beta_non_spec = None
                    else:
                        g_spec = g.index_select(1, spec_token_indx)
                        beta_spec = beta.index_select(1, spec_token_indx)
                        g_non_spec = g.index_select(1, non_spec_token_indx)
                        beta_non_spec = beta.index_select(1, non_spec_token_indx)
                else:
                    g_spec = None
                    beta_spec = None
                    g_non_spec = g
                    beta_non_spec = beta

                # 2.1: Process the multi-query part
                if spec_sequence_masks is not None:
                    core_attn_out_spec, last_recurrent_state = fused_recurrent_gated_delta_rule(
                            q=query_spec,
                            k=key_spec,
                            v=value_spec,
                            g=g_spec,
                            beta=beta_spec,
                            initial_state=ssm_state,
                            inplace_final_state=True,
                            cu_seqlens=spec_query_start_loc[: attn_metadata.num_spec_decodes + 1],
                            ssm_state_indices=spec_state_indices_tensor,
                            num_accepted_tokens=num_accepted_tokens,
                            use_qk_l2norm_in_kernel=True,
                    )
                else:
                    core_attn_out_spec, last_recurrent_state = None, None

                # 2.2: Process the remaining part
                if attn_metadata.num_prefills > 0:
                    initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
                    clear_ssm_states(initial_state, has_initial_state)
                    (core_attn_out_non_spec, last_recurrent_state) = chunk_gated_delta_rule(
                            q=query_non_spec,
                            k=key_non_spec,
                            v=value_non_spec,
                            g=g_non_spec,
                            beta=beta_non_spec,
                            initial_state=initial_state,
                            output_final_state=True,
                            cu_seqlens=non_spec_query_start_loc,
                            prebuilt_meta=get_non_spec_chunked_prefill_meta(attn_metadata),
                            head_first=False,
                            use_qk_l2norm_in_kernel=True,
                    )
                    ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(ssm_state.dtype)
                elif attn_metadata.num_decodes > 0:
                    core_attn_out_non_spec, last_recurrent_state = fused_recurrent_gated_delta_rule(
                            q=query_non_spec,
                            k=key_non_spec,
                            v=value_non_spec,
                            g=g_non_spec,
                            beta=beta_non_spec,
                            initial_state=ssm_state,
                            inplace_final_state=True,
                            cu_seqlens=non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                            ssm_state_indices=non_spec_state_indices_tensor,
                            use_qk_l2norm_in_kernel=True,
                    )
                else:
                    core_attn_out_non_spec, last_recurrent_state = None, None
            elif attn_metadata.num_decodes > 0:
                core_attn_out_spec = None
                core_attn_out_non_spec = fused_sigmoid_gating_delta_rule_update(
                        A_log=self.A_log.contiguous(),
                        dt_bias=self.dt_bias.contiguous(),
                        q=query_non_spec.contiguous(),
                        k=key_non_spec.contiguous(),
                        v=value_non_spec.contiguous(),
                        a=a.contiguous(),
                        b=b.contiguous(),
                        initial_state_source=ssm_state,
                        initial_state_indices=non_spec_state_indices_tensor,
                        cu_seqlens=non_spec_query_start_loc,
                        use_qk_l2norm_in_kernel=True,
                        softplus_beta=1.0,
                        softplus_threshold=20.0,
                )
            else:
                core_attn_out_spec, core_attn_out_non_spec = None, None
            maybe_save_kv_layer_to_connector("", [])

        # 3. Merge core attention output
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                    (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                    dtype=core_attn_out_non_spec.dtype,
                    device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            if not enable_sp():
                core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
            else:
                core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)[:num_actual_tokens]
        elif spec_sequence_masks is not None:
            if not enable_sp():
                core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
            else:
                core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)[:num_actual_tokens]
        else:
            if not enable_sp():
                core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)
            else:
                core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)[:num_actual_tokens]
