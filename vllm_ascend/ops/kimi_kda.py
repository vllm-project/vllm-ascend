# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend backend for the vLLM 0.27 Kimi K3 delta-attention layer.

The projections, weight loading, and cache specification stay owned by
upstream vLLM.  Only the CUDA-specific convolution and KDA execution is
replaced here with the Ascend metadata builder and AscendC operators.
"""

from dataclasses import dataclass
from functools import wraps

import torch
import torch_npu
from einops import rearrange
from fla_npu.ops.ascendc import causal_conv1d_fn, causal_conv1d_update
from torch import nn
from vllm.compilation.breakable_cudagraph import eager_break_during_capture, is_breakable_cudagraph_enabled
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.utils import replace_parameter
from vllm.models.kimi_k3.nvidia.kda import (
    KimiK3DeltaAttention,
)
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.ascend_config import validate_additional_config_bool
from vllm_ascend.attention.utils import (
    maybe_save_kv_layer_to_connector,
    wait_for_kv_layer_from_connector,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import (
    record_attention_compute_start,
)
from vllm_ascend.ops.gdn_attn_builder import AscendGDNHostMetadataBackend, AscendMegaKDAHostMetadataBackend
from vllm_ascend.ops.kda import run_chunk_kda, run_recurrent_kda
from vllm_ascend.ops.kimi_mega_kda import (
    MegaKDAWeights,
    ensure_mega_kda_bf16,
    ensure_mega_kda_nd,
    load_mega_kda_operator,
    mega_kda_nz_operand,
    run_mega_recurrent_kda,
)
from vllm_ascend.ops.linear_op import KimiOProjMMReduceScatterOp
from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import (
    AscendW4A8MXFPDynamicLinearMethod,
)
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import (
    AscendW8A8MXFP8DynamicLinearMethod,
)
from vllm_ascend.utils import npu_stream_switch

_PACKED_CONV_WEIGHT_NAME = "ascend_conv1d_weight"
_F_PROJ_SHARD_ID = 1
_MEGA_KDA_MAX_BATCH_MULTIPLIER = 32
_KDA_BFG_STREAM: torch.npu.Stream | None = None


def _kda_bfg_stream() -> torch.npu.Stream:
    global _KDA_BFG_STREAM
    if _KDA_BFG_STREAM is None:
        _KDA_BFG_STREAM = torch_npu.npu.Stream()
    return _KDA_BFG_STREAM


class _KDAFusedBFGLinear(MergedColumnParallelLinear):
    """Pack beta, an offline-composed F projection, and the output gate."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        tp_size: int,
        quant_config,
        prefix: str,
    ) -> None:
        projection_size = num_heads * head_dim
        super().__init__(
            input_size=hidden_size,
            output_sizes=[num_heads, projection_size, projection_size],
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )
        if self.tp_size != tp_size:
            raise ValueError(f"KDA fused BFG TP mismatch: layer={self.tp_size}, attention={tp_size}")
        local_projection_size = projection_size // tp_size
        self.f_a_weight = nn.Parameter(
            self.weight.new_empty((head_dim, hidden_size)),
            requires_grad=False,
        )
        self.f_b_weight = nn.Parameter(
            self.weight.new_empty((local_projection_size, head_dim)),
            requires_grad=False,
        )
        self.f_a_weight.weight_loader = self._load_f_a_weight
        self.f_b_weight.weight_loader = self._load_f_b_weight
        self._f_a_loaded = False
        self._f_b_loaded = False

    def _load_f_a_weight(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        del loaded_shard_id
        if param.shape != loaded_weight.shape:
            raise ValueError(
                "KDA f_a_proj checkpoint shape mismatch: "
                f"expected {tuple(param.shape)}, got {tuple(loaded_weight.shape)}"
            )
        param.data.copy_(loaded_weight)
        self._f_a_loaded = True
        self._maybe_fuse_f_proj()

    def _load_f_b_weight(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        del loaded_shard_id
        if loaded_weight.shape == param.shape:
            local_weight = loaded_weight
        else:
            expected_shape = (param.shape[0] * self.tp_size, param.shape[1])
            if loaded_weight.shape != expected_shape:
                raise ValueError(
                    "KDA f_b_proj checkpoint shape mismatch: "
                    f"expected {expected_shape} or {tuple(param.shape)}, "
                    f"got {tuple(loaded_weight.shape)}"
                )
            local_weight = loaded_weight.narrow(
                0,
                self.tp_rank * param.shape[0],
                param.shape[0],
            )
        param.data.copy_(local_weight)
        self._f_b_loaded = True
        self._maybe_fuse_f_proj()

    @torch.no_grad()
    def _maybe_fuse_f_proj(self) -> None:
        if not self._f_a_loaded or not self._f_b_loaded:
            return
        output_dim = getattr(self.weight, "output_dim", None)
        if output_dim is None:
            raise ValueError("KDA fused f_proj requires an output-sharded parameter")
        shard_offset = sum(self.output_sizes[:_F_PROJ_SHARD_ID]) // self.tp_size
        shard_size = self.output_sizes[_F_PROJ_SHARD_ID] // self.tp_size
        param_shard = self.weight.narrow(output_dim, shard_offset, shard_size)
        fused_weight = torch.matmul(
            self.f_b_weight.float(),
            self.f_a_weight.float(),
        ).to(dtype=param_shard.dtype)
        if fused_weight.shape != param_shard.shape:
            raise ValueError(
                "KDA composed f_proj shape mismatch: "
                f"expected {tuple(param_shard.shape)}, got {tuple(fused_weight.shape)}"
            )
        param_shard.copy_(fused_weight)


def _normalize_causal_cache_indices(cache_indices: torch.Tensor) -> torch.Tensor:
    """Normalize vLLM cache indices to one index per request."""
    if cache_indices.dim() == 1:
        return cache_indices.contiguous()
    if cache_indices.dim() == 2:
        return cache_indices[:, 0].contiguous()
    raise ValueError(f"KDA causal cache_indices must be 1D or 2D, got shape={tuple(cache_indices.shape)}")


def _prepare_beta(
    beta: torch.Tensor,
    num_actual_tokens: int,
    *,
    is_preprocessed: bool = False,
) -> torch.Tensor:
    """Slice beta and apply sigmoid unless the auxiliary stream already did."""
    beta = beta[:, :num_actual_tokens]
    return beta if is_preprocessed else beta.float().sigmoid()


def _restore_token_rows(output: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Restore graph padding rows after Mega emits only live verify rows."""
    if output.shape[0] == num_tokens:
        return output
    restored = output.new_zeros((num_tokens, output.shape[-1]))
    restored[: output.shape[0]].copy_(output)
    return restored


@dataclass(frozen=True)
class _MegaKDAPlan:
    batch: int
    spec_len: int
    conv_state_block_ids: torch.Tensor
    ssm_state_indices: torch.Tensor
    conv_num_accepted_tokens: torch.Tensor
    num_accepted_tokens: torch.Tensor


def _kimi_kda_with_output(
    hidden_states: torch.Tensor, positions: torch.Tensor, output: torch.Tensor, layer_name: str
) -> None:
    # Keep runtime metadata and real cache pointers outside Dynamo/FakeTensor.
    layer = get_forward_context().no_compile_layers[layer_name]
    plan = layer._plan_mega_kda(hidden_states)
    if plan is None:
        output.copy_(layer._forward_default(hidden_states, positions))
    else:
        layer._forward_mega_kda(hidden_states, positions, plan, output)


def _kimi_kda_with_output_fake(
    hidden_states: torch.Tensor, positions: torch.Tensor, output: torch.Tensor, layer_name: str
) -> None:
    return


def _kimi_kda(hidden_states: torch.Tensor, positions: torch.Tensor, layer_name: str, output_rows: int) -> torch.Tensor:
    # The ordinary FULL/eager path can return the external operator's fresh
    # result directly. Only breakable segments need a caller-owned buffer.
    layer = get_forward_context().no_compile_layers[layer_name]
    plan = layer._plan_mega_kda(hidden_states)
    if plan is None:
        return layer._forward_default(hidden_states, positions)
    return layer._execute_mega_kda(hidden_states, positions, plan)


def _kimi_kda_fake(
    hidden_states: torch.Tensor, positions: torch.Tensor, layer_name: str, output_rows: int
) -> torch.Tensor:
    return hidden_states.new_empty((output_rows, hidden_states.shape[-1]))


direct_register_custom_op(
    op_name="ascend_kimi_kda",
    op_func=_kimi_kda,
    mutates_args=[],
    fake_impl=_kimi_kda_fake,
    dispatch_key="PrivateUse1",
)


direct_register_custom_op(
    op_name="ascend_kimi_kda_with_output",
    op_func=_kimi_kda_with_output,
    mutates_args=["output"],
    fake_impl=_kimi_kda_with_output_fake,
    dispatch_key="PrivateUse1",
)


class AscendKimiK3DeltaAttention(KimiK3DeltaAttention):
    """Kimi K3 KDA using AscendC prefill and recurrent kernels."""

    # Worker startup prepares this layer after KV-cache binding.
    _requires_kda_state_copy = True

    def __init__(self, config, vllm_config, prefix: str = "") -> None:
        # Upstream asks for the backend during its constructor. Select a
        # separate metadata group so Qwen/GLM and draft layers are untouched.
        self.use_mega_kda = validate_additional_config_bool(
            (getattr(vllm_config, "additional_config", None) or {}).get("enable_mega_kda", False),
            "additional_config.enable_mega_kda",
        )
        self._mega_operator = load_mega_kda_operator() if self.use_mega_kda else None
        quant_config = getattr(vllm_config, "quant_config", None)
        uses_mixed_projection = bool(
            quant_config is not None
            and getattr(
                quant_config,
                "uses_kimi_k3_mixed_kda_projection",
                lambda _prefix: False,
            )(f"{prefix}.in_proj_qkvgfab")
        )
        super().__init__(config, vllm_config, prefix)
        self.uses_mixed_projection = uses_mixed_projection
        self._ascend_kda_state_copy = None
        if uses_mixed_projection:
            # vLLM 0.27 packs all KDA input projections into one linear.  A
            # QuaRot checkpoint instead stores q/k/v as W8A8 and keeps B/F/G
            # in floating point. Split those precision groups so DynamicQuant
            # can overlap the composed BFG projection.
            self.in_proj_qkvgfab = MergedColumnParallelLinear(
                self.hidden_size,
                [self.projection_size] * 3,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_qkv",
            )
            del self.f_b_proj
            self.fused_bfg_proj = _KDAFusedBFGLinear(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                tp_size=self.tp_size,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_gfab",
            )
            self._fused_bfg_output_sizes = (
                self.local_num_heads,
                self.local_projection_size,
                self.local_projection_size,
            )
        # Upstream's FusedRMSNormGated constructor defaults to 1e-5, while
        # Kimi K3 checkpoints use the model-configured RMS epsilon (1e-6 for
        # the production checkpoint). Preserve the checkpoint contract used
        # by the validated v0.26 implementation.
        self.o_norm.eps = config.rms_norm_eps
        self._mega_weights: MegaKDAWeights | None = None
        self._mega_weight_signature: tuple | None = None
        # vLLM keeps the checkpoint-compatible FP32 [3C, 1, W] weight, while
        # fla_npu causal_conv1d consumes an activation-dtype [W, 3C]
        # tensor. Materialize that kernel layout once after weight loading.
        self.register_parameter(
            _PACKED_CONV_WEIGHT_NAME,
            nn.Parameter(
                torch.empty(
                    self.conv_size,
                    3 * self.local_projection_size,
                    dtype=self.model_config.dtype,
                    device=self.conv1d.weight.device,
                ),
                requires_grad=False,
            ),
        )
        original_process_weights = self.conv1d.quant_method.process_weights_after_loading

        @wraps(original_process_weights)
        def process_weights_and_pack(*args, **kwargs):
            result = original_process_weights(*args, **kwargs)
            self._pack_conv_weights()
            self._mega_weights = None
            self._mega_weight_signature = None
            return result

        self.conv1d.quant_method.process_weights_after_loading = process_weights_and_pack

    def get_attn_backend(self) -> type[AttentionBackend]:
        if self.use_mega_kda:
            return AscendMegaKDAHostMetadataBackend
        return AscendGDNHostMetadataBackend

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if getattr(self, "use_mega_kda", False) and self._mega_weights is not None:
            custom_op = getattr(self.o_proj, "custom_op", None)
            if custom_op is not None and type(custom_op) is not KimiOProjMMReduceScatterOp:
                return self._forward_default(hidden_states, positions)
            if torch.compiler.is_compiling():
                if not is_breakable_cudagraph_enabled():
                    rows = hidden_states.shape[0]
                    if type(custom_op) is KimiOProjMMReduceScatterOp:
                        rows = (rows + self.tp_size - 1) // self.tp_size
                    return torch.ops.vllm.ascend_kimi_kda(hidden_states, positions, self.prefix, rows)
                output = self._allocate_mega_output(hidden_states)
                torch.ops.vllm.ascend_kimi_kda_with_output(hidden_states, positions, output, self.prefix)
                return output
            plan = self._plan_mega_kda(hidden_states)
            if plan is not None:
                if not is_breakable_cudagraph_enabled():
                    return self._execute_mega_kda(hidden_states, positions, plan)
                output = self._allocate_mega_output(hidden_states)
                self._forward_mega_kda(hidden_states, positions, plan, output)
                return output
        return self._forward_default(hidden_states, positions)

    def _allocate_mega_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        rows = hidden_states.shape[0]
        if type(getattr(self.o_proj, "custom_op", None)) is KimiOProjMMReduceScatterOp:
            rows = (rows + self.tp_size - 1) // self.tp_size
        return hidden_states.new_empty((rows, self.hidden_size))

    def _forward_default(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if self.uses_mixed_projection:
            num_tokens = hidden_states.size(0)
            mixed_qkv, beta, g1, g2 = self._run_overlapped_qkv_bfg(hidden_states)
            core_attn_out = torch.empty(
                (1, num_tokens, self.local_num_heads, self.head_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            self._forward(
                mixed_qkv=mixed_qkv,
                g1=g1,
                g2=g2,
                beta=beta,
                core_attn_out=core_attn_out,
                beta_is_preprocessed=True,
            )
            core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
            return self.o_proj(core_attn_out)[0]
        return super().forward(hidden_states, positions)

    def _plan_mega_kda(self, hidden_states: torch.Tensor) -> _MegaKDAPlan | None:
        """Resolve a supported decode/verify batch before touching state."""
        if self._mega_weights is None or hidden_states.dtype != torch.bfloat16:
            return None
        context = get_forward_context()
        raw = context.attn_metadata
        if not isinstance(raw, dict) or self.prefix not in raw:
            return None
        metadata = raw[self.prefix]
        if not isinstance(metadata, GDNAttentionMetadata) or metadata.num_prefills:
            return None
        if metadata.num_spec_decodes + metadata.num_decodes <= 0:
            return None
        if getattr(metadata, "mega_has_folded_prefills", True):
            return None
        custom_op = getattr(self.o_proj, "custom_op", None)
        if custom_op is not None and type(custom_op) is not KimiOProjMMReduceScatterOp:
            return None

        is_spec = metadata.num_spec_decodes > 0
        if is_spec and metadata.num_decodes:
            return None
        if is_spec:
            spec_meta = metadata.spec_decode_metadata
            state_indices = metadata.spec_state_indices_tensor
            query_start_loc = None if spec_meta is None else spec_meta.spec_causal_conv1d.query_start_loc
            accepted = metadata.num_accepted_tokens
            if query_start_loc is None or state_indices is None or accepted is None:
                return None
        else:
            query_start_loc = metadata.non_spec_query_start_loc
            state_indices = metadata.non_spec_state_indices_tensor
            if query_start_loc is None or state_indices is None:
                return None
            accepted = torch.ones(state_indices.shape[0], dtype=torch.int32, device=state_indices.device)

        live_batch = metadata.num_spec_decodes if is_spec else metadata.num_decodes
        # FULL graphs use graph-sized stable state tables. Padding is a suffix
        # of NULL_BLOCK_ID rows, exactly as in the source PR's runtime B/S ABI.
        batch = query_start_loc.numel() - 1
        query_lens = getattr(metadata, "mega_query_lens", ())
        if live_batch <= 0 or batch < live_batch or len(query_lens) != live_batch:
            return None
        spec_len = self.num_spec + 1
        if any(length != spec_len for length in query_lens):
            return None
        if (
            hidden_states.ndim != 2
            or hidden_states.shape[-1] != self.hidden_size
            or metadata.num_actual_tokens != live_batch * spec_len
            or hidden_states.shape[0] != batch * spec_len
        ):
            return None
        if batch % self.tp_size != 0 or batch > self.tp_size * _MEGA_KDA_MAX_BATCH_MULTIPLIER:
            return None
        if state_indices.ndim not in (1, 2) or state_indices.shape[0] != batch:
            return None
        if state_indices.ndim == 1 and spec_len != 1:
            return None
        if state_indices.ndim == 2 and state_indices.shape[1] < spec_len:
            return None
        if accepted.numel() < batch:
            return None
        if not self._mega_cache_supported(spec_len):
            logger.warning_once(
                "MegaKDA requires BF16 Conv / FP32 recurrent caches with sufficient verify capacity "
                "using the existing KDA path for %s. Cache layouts: %s",
                self.prefix,
                [(tuple(state.shape), state.stride(), state.dtype) for state in self.kv_cache],
            )
            return None

        accepted = accepted[:batch]
        first_verify = getattr(metadata, "mega_first_verify", None)
        if is_spec:
            # V2 tracks request identity across physical block migration.
            # Without this flag the first round could read the wrong Conv tail.
            if first_verify is None or first_verify.numel() != batch:
                return None
            conv_accepted = getattr(metadata, "mega_conv_accepted_tokens", None)
            if conv_accepted is None:
                conv_accepted = torch.where(
                    first_verify[:batch] > 0,
                    torch.full_like(accepted, spec_len),
                    accepted,
                )
            elif conv_accepted.numel() < batch:
                return None
            else:
                conv_accepted = conv_accepted[:batch]
        else:
            conv_accepted = accepted
        conv_state_block_ids = getattr(metadata, "mega_conv_state_block_ids", None)
        ssm_state_indices = getattr(metadata, "mega_ssm_state_indices", None)
        if conv_state_block_ids is not None and ssm_state_indices is not None:
            conv_state_block_ids = conv_state_block_ids[:batch]
            ssm_state_indices = ssm_state_indices[: batch * spec_len]
        elif state_indices.ndim == 1:
            conv_state_block_ids = state_indices[:batch]
            ssm_state_indices = state_indices[:batch]
        else:
            conv_state_block_ids = state_indices[:batch, 0]
            ssm_state_indices = state_indices[:batch, :spec_len].reshape(-1)
        return _MegaKDAPlan(
            batch=batch,
            spec_len=spec_len,
            conv_state_block_ids=conv_state_block_ids,
            ssm_state_indices=ssm_state_indices,
            conv_num_accepted_tokens=conv_accepted,
            num_accepted_tokens=accepted,
        )

    def _mega_cache_supported(self, spec_len: int) -> bool:
        """Validate payload shapes/dtypes without changing runner-owned views.

        The external operator must honor the original tensors' actual strides.
        This adapter does not make a dense copy or reinterpret their storage.
        """
        if len(self.kv_cache) != 2:
            return False
        conv, recurrent = self.kv_cache
        return (
            conv.ndim == 3
            and conv.shape[1] >= self.conv_size - 1 + spec_len - 1
            and conv.shape[2] == 3 * self.local_projection_size
            and conv.dtype == torch.bfloat16
            and recurrent.ndim == 4
            and recurrent.shape[1:] == (self.local_num_heads, self.head_dim, self.head_dim)
            and recurrent.dtype == torch.float32
        )

    def _build_mega_weights(self) -> MegaKDAWeights:
        """Prepare the external operator operands once after weight loading."""
        o_proj_weight = ensure_mega_kda_nd(self.o_proj.weight.detach())
        o_norm_weight = self.o_norm.weight.detach()
        if o_norm_weight.dtype != torch.float32:
            o_norm_weight = o_norm_weight.to(torch.float32)
        conv = ensure_mega_kda_bf16(ensure_mega_kda_nd(self.get_parameter(_PACKED_CONV_WEIGHT_NAME).detach()))
        if self.uses_mixed_projection:
            weight = ensure_mega_kda_nd(self.in_proj_qkvgfab.weight.detach())
            fused = self.fused_bfg_proj
            fused_weight = ensure_mega_kda_nd(fused.weight.detach())
            output_dim = getattr(fused.weight, "output_dim", 0)
            qkv_src = weight[: 3 * self.local_projection_size]
            f_a_src = ensure_mega_kda_nd(fused.f_a_weight.detach())
            f_b_src = ensure_mega_kda_nd(fused.f_b_weight.detach())
            beta_src = fused_weight.narrow(output_dim, 0, self.local_num_heads)
            gate_src = fused_weight.narrow(
                output_dim,
                self.local_num_heads + self.local_projection_size,
                self.local_projection_size,
            )
        else:
            weight = ensure_mega_kda_nd(self.in_proj_qkvgfab.weight.detach())
            local_projection = self.local_projection_size
            qkv_end = 3 * local_projection
            gate_end = qkv_end + local_projection
            f_a_end = gate_end + self.head_dim
            qkv_src = weight[:qkv_end]
            gate_src = weight[qkv_end:gate_end]
            f_a_src = weight[gate_end:f_a_end]
            beta_src = weight[f_a_end : f_a_end + self.local_num_heads]
            f_b_src = ensure_mega_kda_nd(self.f_b_proj.weight.detach())
        return MegaKDAWeights(
            qkv=mega_kda_nz_operand(qkv_src),
            f_a=mega_kda_nz_operand(f_a_src),
            f_b=mega_kda_nz_operand(f_b_src),
            beta=mega_kda_nz_operand(beta_src),
            output_gate=mega_kda_nz_operand(gate_src),
            o_norm=o_norm_weight,
            o_proj=mega_kda_nz_operand(o_proj_weight),
            conv=conv.view(self.conv_size, 3, self.local_num_heads, self.head_dim).permute(1, 2, 0, 3).contiguous(),
        )

    def _prepare_mega_weights(self) -> bool:
        """Called by the model post-load hook after *all* quant finalizers."""
        self._mega_weights = None
        self._mega_weight_signature = None
        if not self.use_mega_kda:
            return False
        layers = (self.in_proj_qkvgfab, self.o_proj)
        layers += (self.fused_bfg_proj,) if self.uses_mixed_projection else (self.f_b_proj,)
        if (
            self.gate_lower_bound is None
            or getattr(self.conv1d, "bias", None) is not None
            or any(
                not isinstance(layer.quant_method, UnquantizedLinearMethod)
                or layer.weight.dtype != torch.bfloat16
                or layer.weight.ndim != 2
                or getattr(layer, "bias", None) is not None
                for layer in layers
            )
        ):
            logger.warning_once(
                "MegaKDA requires bias-free, unquantized BF16 KDA projections and a bounded gate; "
                "using the existing KDA path."
            )
            return False
        if self.uses_mixed_projection and any(
            weight.dtype != torch.bfloat16
            for weight in (self.fused_bfg_proj.f_a_weight, self.fused_bfg_proj.f_b_weight)
        ):
            return False
        self._mega_weights = self._build_mega_weights()
        self._mega_weight_signature = self._current_mega_weight_signature()
        logger.info_once("MegaKDA BF16/NZ operands prepared after loading for %s", self.prefix)
        return True

    def _current_mega_weight_signature(self) -> tuple:
        weights = [
            self.in_proj_qkvgfab.weight,
            self.o_proj.weight,
            self.o_norm.weight,
            self.get_parameter(_PACKED_CONV_WEIGHT_NAME),
            self.A_log,
            self.dt_bias,
        ]
        if self.uses_mixed_projection:
            weights.extend((self.fused_bfg_proj.weight, self.fused_bfg_proj.f_a_weight, self.fused_bfg_proj.f_b_weight))
        else:
            weights.append(self.f_b_proj.weight)
        signature = []
        for weight in weights:
            try:
                version = weight._version
            except RuntimeError:
                version = None  # Inference tensors have no version counter.
            signature.append((weight.data_ptr(), tuple(weight.shape), weight.dtype, version))
        return tuple(signature)

    @eager_break_during_capture
    def _forward_mega_kda(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        plan: _MegaKDAPlan,
        output: torch.Tensor,
    ) -> None:
        # Breakable graph replays need the same caller-owned output address.
        output.copy_(self._execute_mega_kda(hidden_states, positions, plan))

    def _execute_mega_kda(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        plan: _MegaKDAPlan,
    ) -> torch.Tensor:
        if self._mega_weight_signature != self._current_mega_weight_signature():
            raise RuntimeError("MegaKDA weights changed; run the model process_weights_after_loading hook again")
        conv_state, recurrent_state = self.kv_cache
        required_conv_state_len = self.conv_size - 1 + plan.spec_len - 1
        if (
            conv_state.ndim != 3
            or conv_state.shape[1] < required_conv_state_len
            or conv_state.shape[2] != 3 * self.local_num_heads * self.head_dim
            or recurrent_state.ndim != 4
            or recurrent_state.shape[1:] != (self.local_num_heads, self.head_dim, self.head_dim)
            or conv_state.dtype != torch.bfloat16
            or recurrent_state.dtype != torch.float32
        ):
            raise RuntimeError(
                "MegaKDA requires KDA cache shapes "
                f"conv=[C,>={required_conv_state_len},{3 * self.local_num_heads * self.head_dim}] BF16 and "
                f"recurrent=[R,{self.local_num_heads},{self.head_dim},{self.head_dim}] FP32; "
                f"got conv={tuple(conv_state.shape)}/{conv_state.dtype}, "
                f"recurrent={tuple(recurrent_state.shape)}/{recurrent_state.dtype}"
            )
        wait_for_kv_layer_from_connector(self.prefix)
        record_attention_compute_start()
        # Pass the original cache views and scheduler block IDs to the operator.
        output = run_mega_recurrent_kda(
            hidden_states=hidden_states[: plan.batch * plan.spec_len].reshape(plan.batch, plan.spec_len, -1),
            weights=self._mega_weights,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            conv_state_block_ids=plan.conv_state_block_ids,
            ssm_state_indices=plan.ssm_state_indices,
            conv_num_accepted_tokens=plan.conv_num_accepted_tokens,
            num_accepted_tokens=plan.num_accepted_tokens,
            a_log=self.A_log.detach().reshape(-1).contiguous(),
            dt_bias=self.dt_bias.view(self.local_num_heads, self.head_dim),
            scale=self.head_dim**-0.5,
            gate_lower_bound=float(self.gate_lower_bound if self.gate_lower_bound is not None else -5.0),
            o_norm_variance_epsilon=float(self.o_norm.eps),
            operator=self._mega_operator,
        )
        maybe_save_kv_layer_to_connector("", [])
        # Restore token rows before the collective, so the custom OProj's
        # shard contract remains identical to the original MM + RS path.
        output = _restore_token_rows(output, hidden_states.size(0))
        logger.info_once("mega_recurrent_kda executed: batch=%d, spec_len=%d", plan.batch, plan.spec_len)
        custom_op = getattr(self.o_proj, "custom_op", None)
        if type(custom_op) is KimiOProjMMReduceScatterOp:
            return custom_op.reduce_projected_output(output)
        if self.tp_size > 1 and self.o_proj.reduce_results:
            output = tensor_model_parallel_all_reduce(output)
        return output

    def _run_overlapped_qkv_bfg(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fork BFG work to an auxiliary stream and join it back to main."""
        main_stream = torch.npu.current_stream()
        bfg_stream = _kda_bfg_stream()

        hidden_states_ready = main_stream.record_event()
        hidden_states.record_stream(bfg_stream)
        with npu_stream_switch(bfg_stream):
            bfg_stream.wait_event(hidden_states_ready)
            fused_bfg = self._project_bfg(hidden_states)
            bfg_projection_ready = bfg_stream.record_event()

        quantized_qkv = self._quantize_fused_qkv(hidden_states)
        quant_ready = main_stream.record_event()

        # Stage 1 join: DynamicQuant on main overlaps the BFG GEMM on the
        # auxiliary stream, but the two Cube matmuls remain serialized.
        main_stream.wait_event(bfg_projection_ready)
        mixed_qkv = self._matmul_fused_qkv(quantized_qkv)

        with npu_stream_switch(bfg_stream):
            # Stage 2: after both first-stage branches complete, overlap the
            # QKV Cube matmul with beta's FP32 conversion and sigmoid vector
            # work. Split and reshape the F/output gates here as well so all
            # BFG output handling occurs after the QKV matmul is enqueued.
            bfg_stream.wait_event(quant_ready)
            beta, g1, g2 = self._postprocess_bfg(fused_bfg)
            bfg_ready = bfg_stream.record_event()

        for tensor in (beta, g1, g2):
            tensor.record_stream(main_stream)
        # bfg_ready is the auxiliary stream tail. Joining that exact event is
        # required for multi-stream ACL graph capture as well as eager reuse.
        main_stream.wait_event(bfg_ready)
        return mixed_qkv, beta, g1, g2

    def _project_bfg(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.fused_bfg_proj(hidden_states)[0]

    def _postprocess_bfg(
        self,
        fused_bfg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        beta, raw_gate, output_gate = fused_bfg.split(
            self._fused_bfg_output_sizes,
            dim=-1,
        )
        beta = beta.float().sigmoid().unsqueeze(0)
        raw_gate = rearrange(raw_gate, "n (h d) -> 1 n h d", d=self.head_dim)
        output_gate = rearrange(output_gate, "n (h d) -> n h d", d=self.head_dim)
        return beta, raw_gate, output_gate

    def _quantize_fused_qkv(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        quant_method = self.in_proj_qkvgfab.quant_method
        inner_quant_method = getattr(quant_method, "quant_method", quant_method)
        if (
            isinstance(
                inner_quant_method,
                (
                    AscendW4A8MXFPDynamicLinearMethod,
                    AscendW8A8MXFP8DynamicLinearMethod,
                ),
            )
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.ndim == 2
        ):
            if isinstance(inner_quant_method, AscendW8A8MXFP8DynamicLinearMethod):
                return torch_npu.npu_dynamic_mx_quant(
                    hidden_states,
                    dst_type=torch.float8_e4m3fn,
                    scale_alg=inner_quant_method.dynamic_mx_quant_scale_alg,
                )
            return torch_npu.npu_dynamic_mx_quant(
                hidden_states,
                dst_type=torch.float8_e4m3fn,
            )
        return hidden_states

    def _matmul_fused_qkv(
        self,
        qkv_input: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        quant_method = self.in_proj_qkvgfab.quant_method
        if quant_method is None:
            raise RuntimeError("KDA fused QKV quantization method is not initialized")
        return quant_method.apply(
            self.in_proj_qkvgfab,
            qkv_input,
            bias=None,
        )

    @staticmethod
    def _run_causal_conv1d(
        mixed_qkv: torch.Tensor,
        conv_weights_t: torch.Tensor,
        conv_state: torch.Tensor,
        query_start_loc: torch.Tensor,
        cache_indices: torch.Tensor,
        initial_state_mode: torch.Tensor | None,
        *,
        run_mode: int,
        max_query_len: int = -1,
        num_accepted_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cache_indices = _normalize_causal_cache_indices(cache_indices)
        if run_mode == 0:
            return causal_conv1d_fn(
                mixed_qkv,
                conv_weights_t,
                None,
                conv_states=conv_state,
                query_start_loc=query_start_loc,
                cache_indices=cache_indices,
                has_initial_state=initial_state_mode,
                activation="silu",
                pad_slot_id=PAD_SLOT_ID,
                null_block_id=0,
            )
        if run_mode != 1:
            raise ValueError(f"Unsupported causal_conv1d run_mode: {run_mode}")

        output = torch.empty_like(mixed_qkv)
        return causal_conv1d_update(
            mixed_qkv,
            conv_state,
            conv_weights_t,
            bias=None,
            activation="silu",
            conv_state_indices=cache_indices,
            num_accepted_tokens=num_accepted_tokens,
            query_start_loc=query_start_loc,
            max_query_len=max_query_len,
            null_block_id=0,
            out=output,
        )

    @torch.no_grad()
    def _pack_conv_weights(self) -> None:
        if self.conv1d.weight.is_meta:
            return
        packed_param = self.get_parameter(_PACKED_CONV_WEIGHT_NAME)
        packed_weight = (
            self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
            .transpose(0, 1)
            .to(device=packed_param.device, dtype=packed_param.dtype)
            .contiguous()
        )
        replace_parameter(
            self,
            _PACKED_CONV_WEIGHT_NAME,
            packed_weight,
            prefer_copy=True,
        )

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
        return run_recurrent_kda(
            q,
            k,
            v,
            raw_gate,
            beta,
            recurrent_state,
            cu_seqlens,
            state_indices,
            self.A_log,
            self.dt_bias,
            lower_bound=self.gate_lower_bound,
            num_accepted_tokens=num_accepted_tokens,
        )

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
        cu_seqlens = (
            prebuilt_metadata.cu_seqlens_host
            if prebuilt_metadata.cu_seqlens_kern is None
            else prebuilt_metadata.cu_seqlens_kern
        )
        # Unlike the standalone copy API, prefill requires per-request flags.
        # Missing flags must not silently preserve potentially stale cache rows.
        if has_initial_state is None:
            raise ValueError("KDA prefill requires has_initial_state metadata")
        keep = prebuilt_metadata.keep_meta
        if keep is not None:
            state_indices = state_indices[keep]
            has_initial_state = has_initial_state[keep]

        # The recurrent cache uses [H,V,K]. The fused prefill operator accepts
        # that state layout directly through state_v_first.
        state_copy = getattr(self, "_ascend_kda_state_copy", None)
        if state_copy is None or not getattr(self, "_kda_state_copy_ready", False):
            raise RuntimeError("KDA state copy requires worker cache initialization before prefill")
        initial_state_vk = state_copy.gather(recurrent_state, state_indices, has_initial_state)

        output, final_state = run_chunk_kda(
            q,
            k,
            v,
            raw_gate,
            beta,
            initial_state_vk,
            cu_seqlens,
            prebuilt_metadata.chunk_indices_chunk64_host,
            self.A_log,
            self.dt_bias,
            lower_bound=self.gate_lower_bound,
        )
        state_copy.scatter(recurrent_state, final_state, state_indices)
        return output

    @eager_break_during_capture
    def _forward(
        self,
        mixed_qkv: torch.Tensor,
        g1: torch.Tensor,
        g2: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
        *,
        beta_is_preprocessed: bool = False,
    ) -> None:
        """Dispatch speculative, prefill, and decode tokens through KDA kernels."""
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        if attn_metadata_raw is None:
            core_attn_out.zero_()
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        # Layerwise KV pool hooks must stay inside this eager-break region:
        # the forward() caller may be traced, and these side effects (thread
        # locks, connector waits) would break the graph. Waiting here still
        # orders the deferred mamba state copy and the layer load before the
        # conv/recurrent kernels touch mamba state.
        wait_for_kv_layer_from_connector(self.prefix)
        record_attention_compute_start()

        num_actual_tokens = attn_metadata.num_actual_tokens
        mixed_qkv = mixed_qkv[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        g2 = g2[:num_actual_tokens]
        beta = _prepare_beta(
            beta,
            num_actual_tokens,
            is_preprocessed=beta_is_preprocessed,
        )

        conv_state, recurrent_state = self.kv_cache
        conv_weights_t = self.get_parameter(_PACKED_CONV_WEIGHT_NAME)
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
                assert spec_token_indices is not None
                assert non_spec_token_indices is not None
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
                spec_conv_meta.query_start_loc,
                spec_conv_meta.cache_indices,
                None,
                run_mode=1,
                max_query_len=self.num_spec + 1,
                num_accepted_tokens=spec_conv_meta.num_accepted_tokens,
            )
            q_spec, k_spec, v_spec = (
                rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim) for x in mixed_spec.chunk(3, dim=-1)
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

        core_non_spec = None
        if mixed_non_spec is not None and mixed_non_spec.shape[0] > 0:
            if attn_metadata.num_prefills > 0:
                prefill_meta = attn_metadata.non_spec_prefill_metadata
                assert prefill_meta is not None
                mixed_non_spec = self._run_causal_conv1d(
                    mixed_non_spec,
                    conv_weights_t,
                    conv_state,
                    prefill_meta.causal_conv1d.query_start_loc,
                    prefill_meta.causal_conv1d.cache_indices,
                    prefill_meta.causal_conv1d.initial_state_mode,
                    run_mode=0,
                )
            elif attn_metadata.num_decodes > 0:
                decode_meta = attn_metadata.non_spec_decode_metadata
                assert decode_meta is not None
                mixed_non_spec = self._run_causal_conv1d(
                    mixed_non_spec,
                    conv_weights_t,
                    conv_state,
                    decode_meta.causal_conv1d.query_start_loc,
                    decode_meta.causal_conv1d.cache_indices,
                    None,
                    run_mode=1,
                    max_query_len=1,
                )

            q_non_spec, k_non_spec, v_non_spec = (
                rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim) for x in mixed_non_spec.chunk(3, dim=-1)
            )
            assert raw_gate_non_spec is not None
            assert beta_non_spec is not None

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

        if core_spec is None and core_non_spec is None:
            # Idle DP dummy runs carry graph-shaped metadata with no live work.
            # Do not feed a previous replay's output through the norm gate.
            core_attn_out.zero_()
            maybe_save_kv_layer_to_connector("", [])
            return

        # Reuse the caller-owned result buffer. FULL graphs can leave rows
        # outside the live spec/non-spec index sets, so define them before the
        # two index copies rather than allocating a temporary merged tensor.
        core_attn_out[:, :num_actual_tokens].zero_()
        if core_spec is not None and core_non_spec is not None:
            assert spec_token_indices is not None
            assert non_spec_token_indices is not None
            assert spec_token_indices.numel() + non_spec_token_indices.numel() <= num_actual_tokens
            core_attn_out[:, :num_actual_tokens].index_copy_(1, spec_token_indices, core_spec)
            core_attn_out[:, :num_actual_tokens].index_copy_(1, non_spec_token_indices, core_non_spec)
        elif core_spec is not None:
            core_attn_out[:, :num_actual_tokens] = core_spec
        elif core_non_spec is not None:
            core_attn_out[:, :num_actual_tokens] = core_non_spec

        # Keep the registered norm's parameters while writing into the
        # caller-owned attention output directly.
        norm_out = core_attn_out[:, :num_actual_tokens]
        self.o_norm.forward_oot(norm_out, g2, out=norm_out)
        core_attn_out[:, num_actual_tokens:].zero_()
        maybe_save_kv_layer_to_connector("", [])
