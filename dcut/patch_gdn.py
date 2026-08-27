# SPDX-License-Identifier: Apache-2.0
"""Patch QwenGatedDeltaNetAttention._forward_core for D-Cut."""
from __future__ import annotations
import os

from .globals import logger, ENABLE_GDN_MAIN_PIECEWISE_GRAPH, ENV_CONFIG
from .gdn_buffers import _dcut_alloc_gdn_spec_bufs, _dcut_alloc_gdn_nonspec_bufs
from .gdn_eager import _conv1d_spec_varlen_eager

def _patch_gdn_dcut() -> None:
    """Patch AscendGatedDeltaNetAttention._forward_core for D-Cut.

    Two changes:
    1. Spec Conv1D eager path: use _conv1d_spec_varlen_eager fallback instead
       of CANN op (which crashes on variable query_len from D-Cut truncation).
    2. Recurrent GDN spec kernel call: align ssm_state_indices with actual
       token positions (boolean mask) and clamp num_accepted_tokens to actual
       seq lengths.
    """
    try:
        import torch
        import torch_npu
        from einops import rearrange
        from vllm.distributed import get_pcp_group
        from vllm.forward_context import get_forward_context
        from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd
        from vllm.v1.attention.backend import AttentionMetadata
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
        from vllm.v1.attention.backends.utils import PAD_SLOT_ID

        from vllm_ascend.ascend_forward_context import _EXTRA_CTX
        from vllm_ascend.attention.utils import maybe_save_kv_layer_to_connector
        from vllm_ascend.compilation.acl_graph import (
            get_draft_graph_params,
            get_graph_params,
        )
        from vllm_ascend.device.device_op import DeviceOperator
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
        from vllm_ascend.ops.gdn import (
            AscendGatedDeltaNetAttention,
            to_int64_tuple,
            get_non_spec_causal_conv1d_host_args,
            get_causal_conv1d_update_host_args,
            get_spec_causal_conv1d_update_host_args,
            get_non_spec_chunked_prefill_meta,
        )
        from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule
        from vllm_ascend.ops.triton.fla.utils import clear_ssm_states
        from vllm_ascend.ops.triton.mamba.causal_conv1d import causal_conv1d_fn
        from vllm_ascend.utils import weak_ref_tensors
    except Exception as e:
        import sys as _dbg2
        logger.warning("D-Cut: cannot import GDN ops for patching: %s", e)
        return


    if getattr(QwenGatedDeltaNetAttention, "_dcut_gdn_patched", False):
        return

    # Monkeypatch _pad_conv1d_host_args_to_capture to handle pad_tokens < q_per_seq.
    # When D-Cut truncation reduces spec tokens, the padding falls short.
    from vllm_ascend.ops.gdn import _pad_conv1d_host_args_to_capture as _orig_pad
    if not getattr(_orig_pad, '_dcut_patched', False):
        def _dcut_pad_conv1d_host_args(qsl_host, cidx_host, num_accepted_host,
                                        cap_x_dim0, q_per_seq, with_num_accepted):
            result = _orig_pad(qsl_host, cidx_host, num_accepted_host,
                               cap_x_dim0, q_per_seq, with_num_accepted)
            qsl, cidx, nat = result
            # If still short (pad_tokens < q_per_seq case), add one final dummy
            if qsl and int(qsl[-1]) != cap_x_dim0:
                qsl = tuple(qsl) + (int(cap_x_dim0),)
                cidx = tuple(cidx) + (PAD_SLOT_ID,)
                if with_num_accepted:
                    nat = tuple(nat) + (1,)
            # Clamp num_accepted_tokens to not exceed segment lengths.
            # D-Cut truncation can make nat[i] > (qsl[i+1] - qsl[i]),
            # causing EZ9999: "numAcceptedTokens[i]=X exceeds varlen segment length=Y".
            if with_num_accepted and nat:
                clamped = []
                for i in range(len(nat)):
                    if i + 1 < len(qsl):
                        seg_len = int(qsl[i + 1]) - int(qsl[i])
                        clamped.append(min(int(nat[i]), seg_len))
                    else:
                        clamped.append(int(nat[i]))
                nat = tuple(clamped)
            return qsl, cidx, nat
        _dcut_pad_conv1d_host_args._dcut_patched = True
        # Patch in the gdn module so update_conv1d_graph_params uses the fixed version
        import vllm_ascend.ops.gdn as _gdn_mod
        _gdn_mod._pad_conv1d_host_args_to_capture = _dcut_pad_conv1d_host_args
        logger.warning("D-Cut: patched _pad_conv1d_host_args_to_capture for sub-q_per_seq padding")


    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """Core attention computation (called by custom op). D-Cut patched."""
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
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
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

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
            (spec_qsl_host, spec_ci_host, spec_nat_host) = get_spec_causal_conv1d_update_host_args(attn_metadata)
            if _EXTRA_CTX.capturing or torch.compiler.is_compiling():
                stream = torch_npu.npu.current_stream()
                event = torch.npu.ExternalEvent()
                event.wait(stream)
                event.reset(stream)
                graph_params = get_graph_params() if not _EXTRA_CTX.is_draft_model else get_draft_graph_params()
                graph_params.conv1d_events[num_actual_tokens].append(event)

                output_spec = torch.empty_like(mixed_qkv_spec)
                spec_q_per_seq = int(attn_metadata.spec_state_indices_tensor.size(-1))
                graph_params.conv1d_params[num_actual_tokens].append(
                    (
                        weak_ref_tensors(output_spec),
                        weak_ref_tensors(mixed_qkv_spec),
                        weak_ref_tensors(conv_weights_T),
                        weak_ref_tensors(self_kv_cache[0]),
                        self.conv1d.bias,
                        activation_num,
                        PAD_SLOT_ID,
                        1,
                        "spec",
                        self.prefix,
                        spec_qsl_host,
                        spec_ci_host,
                        spec_nat_host,
                        spec_q_per_seq,
                    )
                )

                torch.npu.graph_task_group_begin(stream)
                torch.ops._C_ascend.npu_causal_conv1d_custom(
                    output_spec,
                    mixed_qkv_spec,
                    conv_weights_T,
                    conv_state=self_kv_cache[0],
                    bias_opt=self.conv1d.bias,
                    query_start_loc_opt=spec_qsl_host,
                    cache_indices_opt=spec_ci_host,
                    initial_state_mode_opt=(),
                    num_accepted_tokens_opt=spec_nat_host,
                    activation_mode=activation_num,
                    pad_slot_id=PAD_SLOT_ID,
                    run_mode=1,
                )
                handle = torch.npu.graph_task_group_end(stream)
                graph_params.conv1d_handles[num_actual_tokens].append(handle)
                mixed_qkv_spec = output_spec
            else:
                # D-Cut: per-request F.conv1d fallback for variable query_len.
                num_spec_decodes = attn_metadata.num_spec_decodes
                use_cann = True  # Use CANN op (fast); padding fix handles variable query_len

                if use_cann:
                    output_spec = torch.empty_like(mixed_qkv_spec)
                    # D-Cut: clamp num_accepted_tokens to segment lengths.
                    # D-Cut truncation can make nat[i] > (qsl[i+1] - qsl[i]),
                    # causing EZ9999: numAcceptedTokens[i]=X exceeds varlen segment length=Y
                    if spec_nat_host:
                        _clamped = []
                        for _i in range(len(spec_nat_host)):
                            if _i + 1 < len(spec_qsl_host):
                                _seg = int(spec_qsl_host[_i + 1]) - int(spec_qsl_host[_i])
                                _clamped.append(min(int(spec_nat_host[_i]), _seg))
                            else:
                                _clamped.append(int(spec_nat_host[_i]))
                        spec_nat_host = tuple(_clamped)
                    torch.ops._C_ascend.npu_causal_conv1d_custom(
                        output_spec,
                        mixed_qkv_spec,
                        conv_weights_T,
                        conv_state=self_kv_cache[0],
                        bias_opt=self.conv1d.bias,
                        query_start_loc_opt=spec_qsl_host,
                        cache_indices_opt=spec_ci_host,
                        initial_state_mode_opt=(),
                        num_accepted_tokens_opt=spec_nat_host,
                        activation_mode=activation_num,
                        pad_slot_id=PAD_SLOT_ID,
                        run_mode=1,
                    )
                    mixed_qkv_spec = output_spec
                else:
                    output_spec = torch.empty_like(mixed_qkv_spec)
                    _conv1d_spec_varlen_eager(
                        output_spec,
                        mixed_qkv_spec,
                        conv_weights,
                        self_kv_cache[0],
                        self.conv1d.bias,
                        self.activation,
                        self.num_spec,
                        spec_query_start_loc,
                        spec_state_indices_tensor,
                        num_accepted_tokens,
                        num_spec_decodes,
                    )
                    mixed_qkv_spec = output_spec

        # 1.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            if mixed_qkv_non_spec is not None:
                if get_pcp_group().world_size > 1:
                    mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
                    has_initial_state = attn_metadata.has_initial_state
                    non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
                    conv_state = self_kv_cache[0].transpose(-1, -2)
                    mixed_qkv_non_spec = causal_conv1d_fn(
                        mixed_qkv_non_spec_T,
                        conv_weights,
                        self.conv1d.bias,
                        activation=self.activation,
                        conv_states=conv_state,
                        has_initial_state=has_initial_state,
                        cache_indices=non_spec_state_indices_tensor,
                        query_start_loc=non_spec_query_start_loc,
                        metadata=attn_metadata,
                    ).transpose(0, 1)
                else:
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
            non_spec_qsl_host, non_spec_ci_host = get_causal_conv1d_update_host_args(attn_metadata)
            if _EXTRA_CTX.capturing or torch.compiler.is_compiling():
                stream = torch_npu.npu.current_stream()
                event = torch.npu.ExternalEvent()
                event.wait(stream)
                event.reset(stream)
                graph_params = get_graph_params() if not _EXTRA_CTX.is_draft_model else get_draft_graph_params()
                graph_params.conv1d_events[num_actual_tokens].append(event)

                output_non_spec = torch.empty_like(mixed_qkv_non_spec)
                non_spec_q_per_seq = 1
                graph_params.conv1d_params[num_actual_tokens].append(
                    (
                        weak_ref_tensors(output_non_spec),
                        weak_ref_tensors(mixed_qkv_non_spec),
                        weak_ref_tensors(conv_weights_T),
                        weak_ref_tensors(self_kv_cache[0]),
                        self.conv1d.bias,
                        activation_num,
                        PAD_SLOT_ID,
                        1,
                        "non_spec_decode",
                        self.prefix,
                        non_spec_qsl_host,
                        non_spec_ci_host,
                        [],
                        non_spec_q_per_seq,
                    )
                )

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
                    query_start_loc_opt=to_int64_tuple(non_spec_query_start_loc[: num_actual_tokens + 1]),
                    cache_indices_opt=to_int64_tuple(non_spec_state_indices_tensor[:num_actual_tokens]),
                    initial_state_mode_opt=[],
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
        g, beta = DeviceOperator.fused_gdn_gating(self.A_log, a, b, self.dt_bias)
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
            query_spec = l2norm_fwd(query_spec)
            key_spec = l2norm_fwd(key_spec)
            if _EXTRA_CTX.capturing or torch.compiler.is_compiling():
                # Graph capture OR Dynamo tracing: use pre-allocated static
                # buffers (stable data_ptr).  _model_forward fills them
                # graph-externally before each replay via _dcut_fill_gdn_spec_bufs.
                # This replaces the old torch.cat/flatten/.to() which created
                # new tensors each call → stale data_ptr at replay → 0% accuracy.
                # NOTE: Must check torch.compiler.is_compiling() too because
                # _EXTRA_CTX.capturing is False during Dynamo tracing.
                _gdn_bufs = _dcut_alloc_gdn_spec_bufs(
                    self.prefix, num_actual_tokens,
                    spec_state_indices_tensor, spec_query_start_loc.device)
                actual_seq_lengths = _gdn_bufs["asl"]
                aligned_ssm_indices = _gdn_bufs["ssi"]
                clamped_nat = _gdn_bufs["nat"]
            else:
                # Eager mode: D-Cut per-request fallback (torch.cat is safe
                # here — no graph, data_ptr stability irrelevant).
                cu_seqlens = spec_query_start_loc[: attn_metadata.num_spec_decodes + 1]
                actual_seq_lengths = torch.cat([cu_seqlens[:1], cu_seqlens[1:] - cu_seqlens[:-1]])
                per_seq_lens = actual_seq_lengths[1:]
                max_tokens = spec_state_indices_tensor.size(1)
                col_idx = torch.arange(max_tokens, device=spec_state_indices_tensor.device)
                mask = col_idx.unsqueeze(0) < per_seq_lens.unsqueeze(1)
                aligned_ssm_indices = spec_state_indices_tensor[mask].clone()
                clamped_nat = torch.minimum(
                    num_accepted_tokens.to(torch.int32),
                    per_seq_lens.to(torch.int32)
                )
            core_attn_out_spec = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
                query=query_spec.squeeze(0),
                key=key_spec.squeeze(0),
                value=value_spec.squeeze(0),
                g=g_spec.squeeze(0),
                beta=beta_spec.squeeze(0),
                state=ssm_state,
                scale=key_spec.shape[-1] ** -0.5,
                actual_seq_lengths=actual_seq_lengths,
                ssm_state_indices=aligned_ssm_indices,
                num_accepted_tokens=clamped_nat,
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
            query_non_spec = l2norm_fwd(query_non_spec)
            key_non_spec = l2norm_fwd(key_non_spec)
            if _EXTRA_CTX.capturing or torch.compiler.is_compiling():
                # Graph capture: use pre-allocated static buffers (stable
                # data_ptr).  _model_forward fills them before each replay.
                _gdn_ns_bufs = _dcut_alloc_gdn_nonspec_bufs(
                    self.prefix, num_actual_tokens,
                    non_spec_state_indices_tensor,
                    non_spec_query_start_loc.device)
                actual_seq_lengths = _gdn_ns_bufs["asl"]
                _ns_ssi = _gdn_ns_bufs["ssi"]
            else:
                cu_seqlens = non_spec_query_start_loc[: attn_metadata.num_decodes + 1]
                actual_seq_lengths = torch.cat([cu_seqlens[:1], cu_seqlens[1:] - cu_seqlens[:-1]])
                _ns_ssi = non_spec_state_indices_tensor
            core_attn_out_non_spec = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
                query=query_non_spec.squeeze(0),
                key=key_non_spec.squeeze(0),
                value=value_non_spec.squeeze(0),
                g=g_non_spec.squeeze(0) if g_non_spec is not None else g_non_spec,
                beta=beta_non_spec.squeeze(0) if beta_non_spec is not None else beta_non_spec,
                state=ssm_state,
                scale=key_non_spec.shape[-1] ** -0.5,
                actual_seq_lengths=actual_seq_lengths,
                ssm_state_indices=_ns_ssi,
            ).unsqueeze(0)
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None

        # 3. Merge core attention output
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)

    QwenGatedDeltaNetAttention._forward_core = _forward_core
    QwenGatedDeltaNetAttention._dcut_gdn_patched = True
    _check_fn = getattr(QwenGatedDeltaNetAttention, '_forward_core', None)
    logger.warning(
        "D-Cut: patched AscendGatedDeltaNetAttention._forward_core "
        "(D-Cut conv1d varlen eager + ssm_state_indices alignment + NAT clamping)."
    )
