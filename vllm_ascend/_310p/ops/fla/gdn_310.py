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
# from collections.abc import Iterable
# mypy: ignore-errors


import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import chunk_gated_delta_rule_310
from vllm_ascend._310p.ops.fla.fused_gdn_gating import fused_gdn_gating_pytorch
from vllm_ascend._310p.ops.fla.l2norm import l2norm_310p
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.utils import maybe_save_kv_layer_to_connector
from vllm_ascend.utils import enable_sp


def _zero_padded_tokens(
    tensor: torch.Tensor,
    valid_tokens: torch.Tensor,
    token_dim: int,
) -> torch.Tensor:
    if tensor.numel() == 0:
        return tensor

    token_count = tensor.shape[token_dim]
    if token_count == 0:
        return tensor

    positions = torch.arange(
        token_count,
        device=tensor.device,
        dtype=valid_tokens.dtype,
    )
    valid_mask = positions < valid_tokens.to(device=tensor.device)
    mask_shape = [1] * tensor.ndim
    mask_shape[token_dim] = token_count
    return tensor * valid_mask.reshape(mask_shape).to(dtype=tensor.dtype)


def _flatten_state_indices(
    ssm_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_tokens: int,
) -> torch.Tensor:
    if ssm_state_indices.ndim == 1:
        return ssm_state_indices[:total_tokens].to(torch.int32).contiguous()

    num_seqs = (cu_seqlens[1:] - cu_seqlens[:-1]).shape[0]
    seq_lens = cu_seqlens[1 : num_seqs + 1] - cu_seqlens[:num_seqs]
    ssm_state_indices = ssm_state_indices[:num_seqs]

    # Uniform spec-decode ACL graph uses fixed q_len per request; reshape avoids
    # NPU masked_select which breaks stream capture (aclnnMaskedSelect / 107027).
    if _EXTRA_CTX.capturing or (seq_lens.numel() > 0 and torch.all(seq_lens == seq_lens[0])):
        q_per_seq = ssm_state_indices.shape[1]
        flat = ssm_state_indices[:, :q_per_seq].reshape(-1)
        return flat[:total_tokens].to(torch.int32).contiguous()

    # Eager mixed batches with variable seq_lens: compact on CPU, copy back async.
    ssm_cpu = ssm_state_indices.cpu()
    seq_lens_cpu = seq_lens.cpu()
    q_per_seq = ssm_cpu.shape[1]
    positions = torch.arange(q_per_seq)
    valid = positions.unsqueeze(0) < seq_lens_cpu.unsqueeze(1)
    flat_cpu = ssm_cpu.masked_select(valid).to(torch.int32).contiguous()[:total_tokens]
    if not flat_cpu.is_pinned:
        flat_cpu = flat_cpu.pin_memory()
    flat_dev = torch.empty(flat_cpu.numel(), dtype=torch.int32, device=ssm_state_indices.device)
    flat_dev.copy_(flat_cpu, non_blocking=True)
    return flat_dev.contiguous()


def _mask_padded_recurrent_accepted_tokens(
    num_accepted_tokens: torch.Tensor,
    actual_seq_lengths: torch.Tensor,
) -> torch.Tensor:
    accepted_tokens = num_accepted_tokens[: actual_seq_lengths.shape[0]].to(torch.int32).contiguous()
    return torch.where(
        actual_seq_lengths > 0,
        accepted_tokens,
        torch.zeros_like(accepted_tokens),
    ).contiguous()


def npu_recurrent_gated_delta_rule_310(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor,
    state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
) -> torch.Tensor:
    if use_qk_l2norm_in_kernel:
        q = l2norm_310p(q)
        k = l2norm_310p(k)

    total_tokens = v.shape[1]
    flat_state_indices = _flatten_state_indices(ssm_state_indices, cu_seqlens, total_tokens)
    actual_seq_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32).contiguous()
    flat_state_indices = torch.clamp_min(
        flat_state_indices,
        0,
    ).contiguous()
    accepted_tokens = None
    if num_accepted_tokens is not None:
        accepted_tokens = _mask_padded_recurrent_accepted_tokens(
            num_accepted_tokens,
            actual_seq_lengths,
        )

    out = torch.ops._C_ascend.npu_recurrent_gated_delta_rule_310(
        query=q.squeeze(0).to(torch.float16).contiguous(),
        key=k.squeeze(0).to(torch.float16).contiguous(),
        value=v.squeeze(0).to(torch.float16).contiguous(),
        g=None if g is None else g.squeeze(0).to(torch.float32).contiguous(),
        gk=None,
        beta=beta.squeeze(0).to(torch.float16).contiguous(),
        state=state,
        actual_seq_lengths=actual_seq_lengths,
        ssm_state_indices=flat_state_indices,
        num_accepted_tokens=accepted_tokens,
        scale_value=k.shape[-1] ** -0.5,
    ).unsqueeze(0)
    return out


def _310p_get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
    conv_state_dtype, _ = _original_get_state_dtype(self)
    return conv_state_dtype, torch.float16


_original_get_state_dtype = GatedDeltaNetAttention.get_state_dtype


def _merge_spec_and_non_spec_outputs_310(
    core_attn_out: torch.Tensor,
    num_actual_tokens: int,
    spec_token_indx: torch.Tensor,
    non_spec_token_indx: torch.Tensor,
    core_attn_out_spec: torch.Tensor,
    core_attn_out_non_spec: torch.Tensor,
) -> None:
    """Merge spec/non-spec GDN outputs back into the batch layout.

    Avoid NPU ``index_copy_`` (IndexPutV2) which fails on some layouts; use
    direct indexing instead. Validate lengths so mixed prefill+spec batches
    do not pass mismatched tensors from spec ops.
    """
    spec_out = core_attn_out_spec.squeeze(0)
    non_spec_out = core_attn_out_non_spec.squeeze(0)
    n_spec = spec_token_indx.numel()
    n_non_spec = non_spec_token_indx.numel()
    if spec_out.shape[0] != n_spec:
        raise RuntimeError(f"GDN spec output length {spec_out.shape[0]} != spec_token_indx {n_spec}")
    if non_spec_out.shape[0] != n_non_spec:
        raise RuntimeError(f"GDN non-spec output length {non_spec_out.shape[0]} != non_spec_token_indx {n_non_spec}")
    out = core_attn_out[:num_actual_tokens]
    out[spec_token_indx] = spec_out
    out[non_spec_token_indx] = non_spec_out


class AscendGatedDeltaNetAttention310(GatedDeltaNetAttention):
    get_state_dtype = _310p_get_state_dtype

    def get_attn_backend(self):
        from vllm_ascend._310p.ops.gdn_attn_builder_310 import (
            AscendGDNAttentionBackend310,
        )

        return AscendGDNAttentionBackend310

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        # Core attention computation (called by custom op).

        # NOTE: The processing logic of Qwen3_5GatedDeltaNet is the same as Qwen3NextGatedDeltaNet.
        # However, because the ops `torch_npu.npu_recurrent_gated_delta_rule`
        # currently does not support `ssm_state` inputs in float32 format,
        # we temporarily retain the current _forward_core implementation.
        # Once the ops supports float32 `ssm_state`, this patch should be removed.

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
        conv_state = self_kv_cache[0]
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens

        if not enable_sp():
            mixed_qkv = mixed_qkv[:num_actual_tokens]
            b = b[:num_actual_tokens]
            a = a[:num_actual_tokens]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2)).transpose(0, 1)
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
        activation_num = 1 if self.activation else 0

        # 1.1: Process the multi-query part
        if spec_sequence_masks is not None:
            spec_causal_conv1d_meta = attn_metadata.spec_decode_metadata.spec_causal_conv1d
            spec_query_start_loc_device = spec_causal_conv1d_meta.query_start_loc
            uniform_spec_only = attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0
            # The final entry remains the runtime token count even when
            # graph metadata includes padded requests.
            spec_valid_tokens = spec_query_start_loc_device[-1]
            if uniform_spec_only:
                mixed_qkv_spec = _zero_padded_tokens(
                    mixed_qkv_spec,
                    spec_valid_tokens,
                    token_dim=0,
                )
            mixed_qkv_spec = torch.ops._C_ascend.npu_causal_conv1d_310(
                mixed_qkv_spec,
                conv_weights,
                bias=self.conv1d.bias,
                conv_states=conv_state,
                query_start_loc=spec_query_start_loc_device,
                cache_indices=spec_causal_conv1d_meta.cache_indices,
                initial_state_mode=None,
                num_accepted_tokens=spec_causal_conv1d_meta.num_accepted_tokens,
                activation_mode=activation_num,
                pad_slot_id=PAD_SLOT_ID,
                run_mode=1,
            )

        # 1.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            if mixed_qkv_non_spec is not None:
                mixed_qkv_non_spec = torch.ops._C_ascend.npu_causal_conv1d_310(
                    mixed_qkv_non_spec,
                    conv_weights,
                    bias=self.conv1d.bias,
                    conv_states=conv_state,
                    query_start_loc=non_spec_query_start_loc,
                    cache_indices=non_spec_state_indices_tensor,
                    initial_state_mode=has_initial_state,
                    num_accepted_tokens=None,
                    activation_mode=activation_num,
                    pad_slot_id=PAD_SLOT_ID,
                    run_mode=0,
                )
        elif attn_metadata.num_decodes > 0:
            mixed_qkv_non_spec = torch.ops._C_ascend.npu_causal_conv1d_310(
                mixed_qkv_non_spec,
                conv_weights,
                bias=self.conv1d.bias,
                conv_states=conv_state,
                query_start_loc=None,
                cache_indices=non_spec_state_indices_tensor[: attn_metadata.num_actual_tokens],
                initial_state_mode=None,
                num_accepted_tokens=None,
                activation_mode=activation_num,
                pad_slot_id=PAD_SLOT_ID,
                run_mode=1,
            )
        else:
            mixed_qkv_non_spec = None
        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(mixed_qkv_non_spec)

        g, beta = fused_gdn_gating_pytorch(self.A_log, a, b, self.dt_bias)
        if attn_metadata.num_prefills > 0 or spec_sequence_masks is not None:
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

            # 2. Recurrent attention

            # 2.1: Process the multi-query part
            if spec_sequence_masks is not None:
                core_attn_out_spec = npu_recurrent_gated_delta_rule_310(
                    q=query_spec,
                    k=key_spec,
                    v=value_spec,
                    g=g_spec,
                    beta=beta_spec,
                    state=ssm_state,
                    cu_seqlens=spec_query_start_loc[: attn_metadata.num_spec_decodes + 1],
                    ssm_state_indices=spec_state_indices_tensor,
                    num_accepted_tokens=spec_causal_conv1d_meta.num_accepted_tokens,
                    use_qk_l2norm_in_kernel=True,
                )
            else:
                core_attn_out_spec = None

            # 2.2: Process the remaining part
            if attn_metadata.num_prefills > 0:
                initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
                initial_state[~has_initial_state, ...] = 0
                (
                    core_attn_out_non_spec,
                    last_recurrent_state,
                ) = chunk_gated_delta_rule_310(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=initial_state,
                    output_final_state=True,
                    cu_seqlens=non_spec_query_start_loc,
                    head_first=False,
                    use_qk_l2norm_in_kernel=True,
                )

                # Init cache
                ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(ssm_state.dtype)
            elif attn_metadata.num_decodes > 0:
                core_attn_out_non_spec = npu_recurrent_gated_delta_rule_310(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    state=ssm_state,
                    cu_seqlens=non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    use_qk_l2norm_in_kernel=True,
                )
            else:
                core_attn_out_non_spec = None

        elif attn_metadata.num_decodes > 0:
            core_attn_out_non_spec = npu_recurrent_gated_delta_rule_310(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g,
                beta=beta,
                state=ssm_state,
                cu_seqlens=non_spec_query_start_loc,
                ssm_state_indices=non_spec_state_indices_tensor,
                use_qk_l2norm_in_kernel=True,
            )
        # 3. Merge core attention output
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            _merge_spec_and_non_spec_outputs_310(
                core_attn_out,
                num_actual_tokens,
                spec_token_indx,
                non_spec_token_indx,
                core_attn_out_spec,
                core_attn_out_non_spec,
            )
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
        if spec_sequence_masks is not None and uniform_spec_only:
            core_attn_out.copy_(
                _zero_padded_tokens(
                    core_attn_out,
                    spec_valid_tokens,
                    token_dim=0,
                )
            )
        maybe_save_kv_layer_to_connector("", [])
