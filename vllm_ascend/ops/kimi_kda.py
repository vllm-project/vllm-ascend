#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.

"""Ascend implementation of Kimi's gated delta attention.

The vLLM implementation provides the projections, cache specification, and
opaque ``kda_attention`` custom op.  This OOT replacement keeps that public
surface while routing prefill through the Kimi AscendC kernels and decode
through the recurrent KDA AscendC kernel.
"""

from collections.abc import Callable
from functools import partial, wraps

import torch
from einops import rearrange
from vllm.config import VllmConfig
from vllm.distributed import get_pcp_group, get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context

try:
    from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd  # type: ignore[import-not-found]
except ModuleNotFoundError:
    from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
from vllm.model_executor.layers.linear import ColumnParallelLinear, QKVParallelLinear
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import replace_parameter
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionBackend
from vllm_ascend.ops.kimi_kda_state import kimi_kda_state_shape
from vllm_ascend.ops.triton.fla.utils import clear_ssm_states
from vllm_ascend.ops.triton.kda.kda import fused_kda_gate
from vllm_ascend.utils import is_vl_model, parse_layer_idx

apply_kda_rms_norm_sigmoid_gate: (
    Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, float],
        torch.Tensor,
    ]
    | None
) = None
if HAS_TRITON:
    from vllm_ascend.ops.triton.kda.fused_norm_gate import (
        apply_kda_rms_norm_sigmoid_gate as triton_apply_kda_rms_norm_sigmoid_gate,
    )

    apply_kda_rms_norm_sigmoid_gate = triton_apply_kda_rms_norm_sigmoid_gate

_KDA_CHUNK_SIZE = 64
_PACKED_CONV_WEIGHT_NAME = "packed_conv_weights"
_FUSED_QKV_NAME = "fused_qkv"


def _zero_padded_spec_output(
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    """Zero graph-padding rows skipped by the recurrent KDA kernel.

    ``recurrent_kda`` leaves the output for zero-length sequences
    uninitialized. FULL graph replay keeps those rows in the static output
    shape, so explicitly clear the uncovered tail before it reaches the
    residual and MoE layers.
    """
    token_indices = torch.arange(
        output.shape[1],
        dtype=query_start_loc.dtype,
        device=output.device,
    )
    valid_tokens = token_indices < query_start_loc[-1]
    return torch.where(
        valid_tokens.view(1, -1, 1, 1),
        output,
        0.0,
    )


def uses_kimi_k3_global_inputs_embeds(vllm_config: VllmConfig) -> bool:
    model_config = vllm_config.model_config
    if model_config.enable_prompt_embeds:
        return True
    if not is_vl_model(vllm_config) or model_config.multimodal_config is None:
        return False
    multimodal_config = model_config.multimodal_config
    return bool(multimodal_config.enable_mm_embeds or multimodal_config.get_limit_per_prompt("image") > 0)


def _load_a_log(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    *,
    num_heads: int,
) -> None:
    """Normalize supported A_log layouts and then TP-shard heads."""
    if loaded_weight.ndim == 1:
        if loaded_weight.shape[0] < num_heads:
            raise ValueError(f"A_log has fewer checkpoint heads than the model: {loaded_weight.shape[0]} < {num_heads}")
        # Some checkpoints pad the logical heads in a one-dimensional tensor.
        loaded_weight = loaded_weight[:num_heads].reshape(1, 1, num_heads, 1)
    elif loaded_weight.ndim == 4:
        if loaded_weight.shape[0] != 1 or loaded_weight.shape[1] != 1 or loaded_weight.shape[3] != 1:
            raise ValueError(f"A_log 4-D checkpoint must have shape [1, 1, H, 1], got {tuple(loaded_weight.shape)}")
        if tuple(loaded_weight.shape) == tuple(param.shape):
            default_weight_loader(param, loaded_weight)
            return
        if loaded_weight.shape[2] < num_heads:
            raise ValueError(f"A_log has fewer checkpoint heads than the model: {loaded_weight.shape[2]} < {num_heads}")
        loaded_weight = loaded_weight[:, :, :num_heads, :]
    else:
        raise ValueError(f"A_log checkpoint must be 1-D or 4-D, got {loaded_weight.ndim}-D")

    local_heads = param.shape[2]
    if local_heads <= 0 or num_heads % local_heads != 0:
        raise ValueError(
            "A_log parameter shape is incompatible with logical heads: "
            f"param={tuple(param.shape)}, num_heads={num_heads}"
        )
    tp_rank = get_tensor_model_parallel_rank()
    start = tp_rank * local_heads
    if start + local_heads > num_heads:
        raise ValueError(f"A_log TP rank {tp_rank} exceeds {num_heads} logical heads")
    default_weight_loader(
        param,
        loaded_weight.narrow(2, start, local_heads),
    )


def _require_ascendc_prefill_ops() -> None:
    required_ops = ("kda_gate_cumsum", "chunk_kda_fwd")
    missing_ops = [name for name in required_ops if not hasattr(torch.ops._C_ascend, name)]
    if missing_ops:
        qualified_ops = ", ".join(f"torch.ops._C_ascend.{name}" for name in missing_ops)
        raise RuntimeError(
            "Kimi KDA prefill requires the PR141 AscendC operators, but the "
            f"following schemas are missing: {qualified_ops}. Rebuild and install "
            "the vllm-ascend custom operators with KDA support."
        )


class AscendKimiGatedDeltaNetAttention(KimiGatedDeltaNetAttention):
    """Kimi KDA with Ascend prefill/decode kernels.

    Kimi K3 adds two details that are absent from vLLM 0.23's base layer:
    a full-rank output gate (``g_proj``) and a bounded sigmoid decay gate.
    """

    def __init__(self, config, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(config, vllm_config, prefix)

        kda_config = config.linear_attn_config
        assert kda_config is not None, "linear_attn_config must be set"
        self.use_full_rank_gate = bool(kda_config.get("use_full_rank_gate", False))
        gate_lower_bound = kda_config.get("gate_lower_bound")
        self.gate_lower_bound = float(gate_lower_bound) if gate_lower_bound is not None else None

        # KDA uses the same hidden states and TP head layout for Q, K, and V.
        # Pack their checkpoint shards into one standard QKV linear so MXFP8
        # performs one dynamic quantization and one quantized matmul.
        fused_qkv = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_heads,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.{_FUSED_QKV_NAME}",
        )
        del self.q_proj
        del self.k_proj
        del self.v_proj
        self.fused_qkv = fused_qkv

        self.A_log.weight_loader = partial(
            _load_a_log,
            num_heads=self.num_heads,
        )

        # vLLM 0.23 builds the legacy low-rank output gate unconditionally.
        # Replace it with the checkpoint-compatible full-rank projection for K3.
        if self.use_full_rank_gate:
            del self.g_a_proj
            del self.g_b_proj
            self.g_proj = ColumnParallelLinear(
                self.hidden_size,
                self.head_dim * self.num_heads,
                bias=False,
                quant_config=self.quant_config,
                prefix=f"{prefix}.g_proj",
            )

        # The upstream class used FusedRMSNormGated's default epsilon.  K3's
        # checkpoint config is authoritative and uses the sigmoid gate path.
        self.o_norm.eps = config.rms_norm_eps

        # Multimodal inputs_embeds are built before the Ascend forward context,
        # so the first decoder layer receives the full token sequence.  Every
        # later layer receives a FlashComm token shard.  Keep this decision
        # static so Dynamo does not need to infer the layout from tensor shapes.
        self.is_vl_first_layer = bool(uses_kimi_k3_global_inputs_embeds(vllm_config) and parse_layer_idx(prefix) == 0)

        # The checkpoint stores three fp32 convolution weights as [C, 1, W],
        # while the AscendC kernel consumes one activation-dtype [W, 3 * C]
        # tensor. Keep the derived kernel-format weight on q_conv1d so it uses
        # the same parameter load/reload lifecycle as other repacked weights.
        self.q_conv1d.register_parameter(
            _PACKED_CONV_WEIGHT_NAME,
            torch.nn.Parameter(
                torch.empty(
                    self._packed_conv_shape(),
                    dtype=self.model_config.dtype,
                ),
                requires_grad=False,
            ),
        )
        for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d):
            self._wrap_conv_process_weights(conv)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return AscendGDNAttentionBackend

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return kimi_kda_state_shape(
            self.tp_size,
            self.num_heads,
            self.head_dim,
            self.conv_size,
            self.num_spec,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        del positions
        # KDA metadata and its recurrent state describe the complete sequence.
        # KDA's gate projections do not match SequenceColumnParallelOp's prefix
        # whitelist, so gather the token shard once before every projection.
        # The fused module deliberately uses the ``fused_qkv`` prefix instead
        # of ``qkv_proj`` to avoid a second automatic gather inside the linear.
        # The multimodal first layer is already full-sized and must not gather.
        hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            hidden_states.contiguous(),
            not self.is_vl_first_layer,
        )
        num_tokens = hidden_states.size(0)
        qkv = self.fused_qkv(hidden_states)[0]
        projection_size = self.local_num_heads * self.head_dim
        q, k, v = qkv.split([projection_size] * 3, dim=-1)

        beta = self.b_proj(hidden_states)[0].float().sigmoid().unsqueeze(0)
        raw_gate = self.f_b_proj(self.f_a_proj(hidden_states)[0])[0]
        raw_gate = rearrange(raw_gate, "n (h d) -> 1 n h d", d=self.head_dim)

        if self.use_full_rank_gate:
            output_gate = self.g_proj(hidden_states)[0]
        else:
            output_gate = self.g_b_proj(self.g_a_proj(hidden_states)[0])[0]
        output_gate = rearrange(output_gate, "n (h d) -> n h d", d=self.head_dim)

        core_attn_out = torch.zeros(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.vllm.kda_attention(
            q,
            k,
            v,
            raw_gate,
            beta,
            core_attn_out,
            self.prefix,
        )
        core_attn_out = self._apply_output_norm_gate(
            core_attn_out,
            output_gate,
            num_valid_rows=getattr(self, "_kda_norm_num_valid_rows", None),
        )
        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        output[:] = self.o_proj(core_attn_out)[0]

    def _apply_output_norm_gate(
        self,
        core_attn_out: torch.Tensor,
        output_gate: torch.Tensor,
        num_valid_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if apply_kda_rms_norm_sigmoid_gate is not None:
            return apply_kda_rms_norm_sigmoid_gate(
                core_attn_out,
                output_gate,
                self.o_norm.weight,
                self.o_norm.eps,
                num_valid_rows=num_valid_rows,
            )
        return self.o_norm(core_attn_out, output_gate)

    @staticmethod
    def _run_causal_conv1d(
        mixed_qkv: torch.Tensor,
        conv_weights_t: torch.Tensor,
        conv_state: torch.Tensor,
        metadata,
        *,
        run_mode: int,
        num_accepted_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = torch.empty_like(mixed_qkv)
        torch.ops._C_ascend.npu_causal_conv1d_custom(
            out,
            mixed_qkv,
            conv_weights_t,
            conv_state=conv_state,
            bias_opt=None,
            query_start_loc_opt=metadata.query_start_loc,
            cache_indices_opt=metadata.cache_indices,
            initial_state_mode_opt=getattr(metadata, "initial_state_mode", None),
            num_accepted_tokens_opt=num_accepted_tokens,
            activation_mode=1,
            pad_slot_id=PAD_SLOT_ID,
            run_mode=run_mode,
        )
        return out

    def _packed_conv_shape(self) -> tuple[int, int]:
        local_channels = self.local_num_heads * self.head_dim
        return self.conv_size, 3 * local_channels

    def _wrap_conv_process_weights(
        self,
        conv: ColumnParallelLinear,
    ) -> None:
        """Refresh the packed weight after a complete checkpoint load.

        Kernel-format reloads address ``packed_conv_weights`` directly. They
        must include that parameter instead of relying on these source-weight
        post-load hooks.
        """
        original_process_weights = conv.quant_method.process_weights_after_loading

        @wraps(original_process_weights)
        def wrapped_process_weights(*args, **kwargs):
            result = original_process_weights(*args, **kwargs)
            self._pack_conv_weights()
            return result

        conv.quant_method.process_weights_after_loading = wrapped_process_weights  # type: ignore[method-assign]

    @torch.no_grad()
    def _pack_conv_weights(self) -> None:
        source_weights = tuple(conv.weight for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d))
        if any(weight.is_meta for weight in source_weights):
            return

        packed_param = self.q_conv1d.get_parameter(_PACKED_CONV_WEIGHT_NAME)
        packed_weights = torch.cat(
            [
                weight.view(weight.size(0), weight.size(2))
                .transpose(0, 1)
                .to(device=packed_param.device, dtype=packed_param.dtype)
                for weight in source_weights
            ],
            dim=1,
        ).contiguous()
        replace_parameter(
            self.q_conv1d,
            _PACKED_CONV_WEIGHT_NAME,
            packed_weights,
            prefer_copy=True,
        )

    def _conv_weights_t(self) -> torch.Tensor:
        return self.q_conv1d.get_parameter(_PACKED_CONV_WEIGHT_NAME)

    def _recurrent_gate(self, raw_gate: torch.Tensor) -> torch.Tensor:
        flat_gate = rearrange(raw_gate, "1 n h d -> n (h d)")
        gate = fused_kda_gate(
            flat_gate,
            self.A_log,
            self.head_dim,
            g_bias=self.dt_bias,
            safe_gate=self.gate_lower_bound is not None,
            lower_bound=self.gate_lower_bound if self.gate_lower_bound is not None else -5.0,
        )
        return gate.unsqueeze(0)

    def _run_recurrent(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        state_indices: torch.Tensor,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = torch.ops._C_ascend.recurrent_kda(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            raw_gate.contiguous(),
            beta.contiguous(),
            recurrent_state,
            cu_seqlens,
            state_indices,
            self.A_log.reshape(-1).contiguous(),
            self.dt_bias.contiguous(),
            num_accepted_tokens=num_accepted_tokens,
            scale=self.head_dim**-0.5,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=False,
            allow_neg_eigval=False,
            safe_gate=self.gate_lower_bound is not None,
            lower_bound=self.gate_lower_bound if self.gate_lower_bound is not None else -5.0,
        )
        return out

    def _run_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        prebuilt_metadata,
    ) -> torch.Tensor:
        if get_pcp_group().world_size > 1:
            raise NotImplementedError("Kimi KDA prefill does not yet support PCP.")
        _require_ascendc_prefill_ops()

        cu_seqlens_kern = prebuilt_metadata.cu_seqlens_kern
        cu_seqlens = prebuilt_metadata.cu_seqlens_host if cu_seqlens_kern is None else cu_seqlens_kern
        keep = prebuilt_metadata.keep_meta
        if keep is not None:
            if keep.numel() != state_indices.shape[0] or keep.numel() != has_initial_state.numel():
                raise ValueError(
                    "Kimi KDA prefill metadata is inconsistent: keep_meta must have "
                    "one entry per uncompressed sequence."
                )
            state_indices = state_indices[keep]
            has_initial_state = has_initial_state[keep]

        num_sequences = (cu_seqlens.numel() if isinstance(cu_seqlens, torch.Tensor) else len(cu_seqlens)) - 1
        if state_indices.shape[0] != num_sequences or has_initial_state.numel() != num_sequences:
            raise ValueError(
                "Kimi KDA prefill metadata is inconsistent: compact cu_seqlens, "
                "state_indices, and has_initial_state must describe the same number of sequences."
            )

        # The recurrent cache uses [H,V,K].  PR141's AscendC prefill operator
        # uses [H,K,V], so transpose only at that operator boundary.
        initial_state_vk = recurrent_state[state_indices].contiguous()
        clear_ssm_states(initial_state_vk, has_initial_state)

        initial_state_kv = initial_state_vk.transpose(-1, -2).contiguous()
        cu_seqlens_ascendc = (
            tuple(cu_seqlens.detach().cpu().tolist()) if isinstance(cu_seqlens, torch.Tensor) else cu_seqlens
        )

        q = l2norm_fwd(q.contiguous())
        k = l2norm_fwd(k.contiguous())

        if self.gate_lower_bound is not None:
            gate_cumsum = torch.ops._C_ascend.kda_gate_cumsum(
                raw_gate.contiguous(),
                _KDA_CHUNK_SIZE,
                A_log=self.A_log.reshape(-1).contiguous(),
                dt_bias=self.dt_bias.contiguous(),
                cu_seqlens=cu_seqlens_ascendc,
                use_gate_in_kernel=True,
                safe_gate=True,
                lower_bound=self.gate_lower_bound,
                layout="BSND",
            )
        else:
            gate = self._recurrent_gate(raw_gate)
            gate_cumsum = torch.ops._C_ascend.kda_gate_cumsum(
                gate.contiguous(),
                _KDA_CHUNK_SIZE,
                cu_seqlens=cu_seqlens_ascendc,
                layout="BSND",
            )

        result = torch.ops._C_ascend.chunk_kda_fwd(
            q,
            k,
            v.contiguous(),
            gate_cumsum,
            beta.contiguous(),
            self.head_dim**-0.5,
            _KDA_CHUNK_SIZE,
            layout="BSND",
            initial_state=initial_state_kv,
            output_final_state=True,
            cu_seqlens=cu_seqlens_ascendc,
            chunk_indices=prebuilt_metadata.chunk_indices_chunk64_host,
            return_intermediate=False,
        )
        recurrent_state[state_indices] = result[1].transpose(-1, -2).contiguous().to(recurrent_state.dtype)
        return result[0]

    def _forward(
        self,
        q_proj_states: torch.Tensor,
        k_proj_states: torch.Tensor,
        v_proj_states: torch.Tensor,
        g1: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw: AttentionMetadata | None = forward_context.attn_metadata
        if attn_metadata_raw is None:
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        num_actual_tokens = attn_metadata.num_actual_tokens
        q_proj_states = q_proj_states[:num_actual_tokens]
        k_proj_states = k_proj_states[:num_actual_tokens]
        v_proj_states = v_proj_states[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]

        conv_state, recurrent_state = self.kv_cache
        mixed_qkv = torch.cat((q_proj_states, k_proj_states, v_proj_states), dim=-1)
        conv_weights_t = self._conv_weights_t()

        spec_masks = attn_metadata.spec_sequence_masks
        spec_token_indices = attn_metadata.spec_token_indx
        non_spec_token_indices = attn_metadata.non_spec_token_indx

        if spec_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_spec = mixed_qkv
                raw_gate_spec = g1
                beta_spec = beta
                mixed_non_spec = raw_gate_non_spec = beta_non_spec = None
            else:
                mixed_spec = mixed_qkv.index_select(0, spec_token_indices)
                raw_gate_spec = g1.index_select(1, spec_token_indices)
                beta_spec = beta.index_select(1, spec_token_indices)
                mixed_non_spec = mixed_qkv.index_select(0, non_spec_token_indices)
                raw_gate_non_spec = g1.index_select(1, non_spec_token_indices)
                beta_non_spec = beta.index_select(1, non_spec_token_indices)
        else:
            mixed_spec = raw_gate_spec = beta_spec = None
            mixed_non_spec = mixed_qkv
            raw_gate_non_spec = g1
            beta_non_spec = beta

        core_spec = None
        if mixed_spec is not None:
            spec_meta = attn_metadata.spec_decode_metadata
            assert spec_meta is not None
            spec_conv_meta = spec_meta.spec_causal_conv1d
            mixed_spec = self._run_causal_conv1d(
                mixed_spec,
                conv_weights_t,
                conv_state,
                spec_conv_meta,
                run_mode=1,
                num_accepted_tokens=spec_conv_meta.num_accepted_tokens,
            )
            q_spec, k_spec, v_spec = mixed_spec.chunk(3, dim=-1)
            q_spec, k_spec, v_spec = (
                rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim) for x in (q_spec, k_spec, v_spec)
            )
            assert raw_gate_spec is not None and beta_spec is not None
            assert attn_metadata.spec_query_start_loc is not None
            assert attn_metadata.spec_state_indices_tensor is not None
            core_spec = self._run_recurrent(
                q_spec,
                k_spec,
                v_spec,
                raw_gate_spec,
                beta_spec,
                recurrent_state,
                attn_metadata.spec_query_start_loc,
                attn_metadata.spec_state_indices_tensor,
                num_accepted_tokens=spec_conv_meta.num_accepted_tokens,
            )
            # Padding rows the recurrent kernel left uninitialized are zeroed
            # downstream: the contiguous path folds the zeroing into the
            # gated-norm kernel (via ``self._kda_norm_num_valid_rows``), so the
            # Range/Less/Fill/SelectV2 pre-pass is skipped; the merge path still
            # zeroes explicitly because spec tokens are scattered (not a row
            # range the norm can threshold).

        core_non_spec = None
        if mixed_non_spec is not None and mixed_non_spec.shape[0] > 0:
            if attn_metadata.num_prefills > 0:
                prefill_meta = attn_metadata.non_spec_prefill_metadata
                assert prefill_meta is not None
                mixed_non_spec = self._run_causal_conv1d(
                    mixed_non_spec,
                    conv_weights_t,
                    conv_state,
                    prefill_meta.causal_conv1d,
                    run_mode=0,
                )
            elif attn_metadata.num_decodes > 0:
                decode_meta = attn_metadata.non_spec_decode_metadata
                assert decode_meta is not None
                mixed_non_spec = self._run_causal_conv1d(
                    mixed_non_spec,
                    conv_weights_t,
                    conv_state,
                    decode_meta.causal_conv1d,
                    run_mode=1,
                )

            q_non_spec, k_non_spec, v_non_spec = mixed_non_spec.chunk(3, dim=-1)
            q_non_spec, k_non_spec, v_non_spec = (
                rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim) for x in (q_non_spec, k_non_spec, v_non_spec)
            )
            assert raw_gate_non_spec is not None and beta_non_spec is not None

            split_non_spec = spec_masks is None and attn_metadata.num_prefills > 0 and attn_metadata.num_decodes > 0
            num_decode_tokens = attn_metadata.num_decode_tokens
            core_decode = None
            if split_non_spec:
                assert attn_metadata.non_spec_query_start_loc is not None
                assert attn_metadata.non_spec_state_indices_tensor is not None
                core_decode = self._run_recurrent(
                    q_non_spec[:, :num_decode_tokens],
                    k_non_spec[:, :num_decode_tokens],
                    v_non_spec[:, :num_decode_tokens],
                    raw_gate_non_spec[:, :num_decode_tokens],
                    beta_non_spec[:, :num_decode_tokens],
                    recurrent_state,
                    attn_metadata.non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                    attn_metadata.non_spec_state_indices_tensor[: attn_metadata.num_decodes],
                )

            if attn_metadata.num_prefills > 0:
                if split_non_spec:
                    q_non_spec = q_non_spec[:, num_decode_tokens:]
                    k_non_spec = k_non_spec[:, num_decode_tokens:]
                    v_non_spec = v_non_spec[:, num_decode_tokens:]
                    raw_gate_non_spec = raw_gate_non_spec[:, num_decode_tokens:]
                    beta_non_spec = beta_non_spec[:, num_decode_tokens:]

                assert attn_metadata.prefill_state_indices is not None
                assert attn_metadata.prefill_has_initial_state is not None
                prefill_meta = attn_metadata.non_spec_prefill_metadata
                assert prefill_meta is not None
                core_prefill = self._run_prefill(
                    q_non_spec,
                    k_non_spec,
                    v_non_spec,
                    raw_gate_non_spec,
                    beta_non_spec,
                    recurrent_state,
                    attn_metadata.prefill_state_indices,
                    attn_metadata.prefill_has_initial_state,
                    prefill_meta.chunk,
                )
                core_non_spec = (
                    torch.cat((core_decode, core_prefill), dim=1) if core_decode is not None else core_prefill
                )
            elif attn_metadata.num_decodes > 0:
                assert attn_metadata.non_spec_query_start_loc is not None
                assert attn_metadata.non_spec_state_indices_tensor is not None
                core_non_spec = self._run_recurrent(
                    q_non_spec,
                    k_non_spec,
                    v_non_spec,
                    raw_gate_non_spec,
                    beta_non_spec,
                    recurrent_state,
                    attn_metadata.non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                    attn_metadata.non_spec_state_indices_tensor,
                )

        if core_spec is not None and core_non_spec is not None:
            # Merge path: spec tokens are scattered to ``spec_token_indices``,
            # so the norm's row-range threshold can't express validity — zero
            # the dummy spec rows explicitly and leave the threshold unset.
            core_spec = _zero_padded_spec_output(
                core_spec, attn_metadata.spec_query_start_loc)
            self._kda_norm_num_valid_rows = None
            merged = torch.empty(
                (1, num_actual_tokens, self.local_num_heads, self.head_dim),
                dtype=core_non_spec.dtype,
                device=core_non_spec.device,
            )
            merged.index_copy_(1, spec_token_indices, core_spec)
            merged.index_copy_(1, non_spec_token_indices, core_non_spec)
            core_attn_out[:, :num_actual_tokens] = merged
        elif core_spec is not None:
            # Contiguous spec path: defer padding-row zeroing to the gated-norm
            # kernel via ``num_valid_rows`` (RMSNorm drives padding rows to 0
            # anyway), so the Range/Less/Fill/SelectV2 pre-pass is skipped.
            # The non-triton fallback keeps the explicit zeroing.
            if apply_kda_rms_norm_sigmoid_gate is not None:
                self._kda_norm_num_valid_rows = attn_metadata.spec_query_start_loc[-1]
            else:
                core_spec = _zero_padded_spec_output(
                    core_spec, attn_metadata.spec_query_start_loc)
                self._kda_norm_num_valid_rows = None
            core_attn_out[:, :num_actual_tokens] = core_spec
        elif core_non_spec is not None:
            self._kda_norm_num_valid_rows = None
            core_attn_out[:, :num_actual_tokens] = core_non_spec
