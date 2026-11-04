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


from functools import wraps

import torch
import torch.nn.functional as F
import torch_npu
from vllm.config import VllmConfig
from vllm.distributed import get_tp_group, tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention as _GDNBaseCls
from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention, Qwen3NextSparseMoeBlock

try:
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MultiTokenPredictor
    from vllm.sequence import IntermediateTensors
except ImportError:
    Qwen3_5MultiTokenPredictor = None
    IntermediateTensors = None

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.models.common.ops.sequence_parallel import sp_all_gather, sp_reduce_scatter
from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention, initialize_packed_conv_weight
from vllm_ascend.ops.linear_op import CustomColumnParallelOp, CustomRowParallelOp
from vllm_ascend.utils import enable_dense_sp

_GDN_PATCH_TARGET = _GDNBaseCls
_GDN_ORIGINAL_INIT = _GDN_PATCH_TARGET.__init__


# torch_npu's fused matmul-collective kernels are restricted to these TP
# sizes (npu_mm_reduce_scatter_base rejects larger ones on the installed
# torch_npu builds; see the tp_size >= 16 TODO in ascend_forward_context.py).
DENSE_SP_FUSED_TP_SIZES = (2, 4, 8)
_DENSE_SP_FUSED_DTYPES = (torch.bfloat16, torch.float16)


def _pure_prefill_batch() -> bool:
    """Whether the running batch is prefill-only.

    The fused collectives cannot allocate their comm resources inside decode
    graph capture (the same constraint KimiOProjMMReduceScatterOp guards
    against), so decode-sized batches fall back to the unfused
    gather / GEMM / reduce-scatter layout, which returns the same shapes.
    """
    if not is_forward_context_available():
        return False
    attn_metadata = get_forward_context().attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = next(
            (meta for meta in attn_metadata.values() if hasattr(meta, "num_prefills") and hasattr(meta, "num_decodes")),
            None,
        )
    if (
        attn_metadata is None
        or getattr(attn_metadata, "num_prefills", 0) == 0
        or getattr(attn_metadata, "num_decodes", 0) != 0
    ):
        return False
    return True


class DenseSpAllGatherMmOp(CustomColumnParallelOp):
    """Fused AllGather + Matmul for a dense-SP column projection.

    Under dense sequence parallelism the residual stream stays on the
    rank-local token shard while a column-parallel GEMM needs every token
    row: ``torch_npu.npu_all_gather_base_mm`` folds the all-gather into the
    GEMM, so the gathered activation is never materialized and the
    communication overlaps the matmul. The output carries the padded
    full-token rows (``shard_rows * tp``); the caller trims back to the real
    token count. Configurations that cannot take the fused kernel fall back
    to the unfused ``sp_all_gather`` + GEMM layout with identical numerics.
    """

    def __init__(self, layer):
        super().__init__(layer)
        self.update_attrs()
        tp_group = get_tp_group()
        backend = tp_group.device_group._get_backend(torch.device("npu"))
        self.hcom = backend.get_hccl_comm_name(tp_group.rank_in_group)
        self.tp_size = tp_group.world_size

    def _can_fuse(self, input_: torch.Tensor) -> bool:
        weight = self.layer.weight
        return (
            _pure_prefill_batch()
            and input_.ndim == 2
            and input_.dtype in _DENSE_SP_FUSED_DTYPES
            and weight.ndim == 2
            and weight.dtype == input_.dtype
            and input_.shape[0] > 0
        )

    def apply_impl(self, input_):
        bias = self.bias if not self.skip_bias_add else None
        assert self.quant_method is not None
        if self._can_fuse(input_):
            output = DeviceOperator.npu_all_gather_base_mm(
                input_.contiguous(),
                self.layer.weight.t(),
                self.hcom,
                self.tp_size,
                bias=bias,
            )
        else:
            input_parallel = sp_all_gather(input_)
            output = self.quant_method.apply(self.layer, input_parallel, bias)
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class DenseSpMmReduceScatterOp(CustomRowParallelOp):
    """Fused Matmul + ReduceScatter for a dense-SP row projection.

    Completes the TP partial sums of the paired column projection with
    ``torch_npu.npu_mm_reduce_scatter_base``: the per-rank GEMM partials are
    reduced and scattered in one kernel, so neither the partial output nor a
    separate collective is materialized. The kernel requires the M axis to
    be divisible by the TP size, so the input is zero-padded first (padding
    a GEMM's input with zero rows equals padding its output). Bias, when
    present, is added once per token row on this rank's shard after the
    scatter; passing it into the fused GEMM would count it once per TP rank.
    Configurations that cannot take the fused kernel fall back to the
    unfused partial-GEMM + ``sp_reduce_scatter`` layout with identical
    numerics.
    """

    def __init__(self, layer):
        super().__init__(layer)
        self.update_attrs()
        tp_group = get_tp_group()
        backend = tp_group.device_group._get_backend(torch.device("npu"))
        self.hcom = backend.get_hccl_comm_name(tp_group.rank_in_group)
        self.tp_size = tp_group.world_size

    def _can_fuse(self, input_parallel: torch.Tensor) -> bool:
        weight = self.layer.weight
        return (
            _pure_prefill_batch()
            and input_parallel.ndim == 2
            and input_parallel.dtype in _DENSE_SP_FUSED_DTYPES
            and weight.ndim == 2
            and weight.dtype == input_parallel.dtype
            and input_parallel.shape[0] > 0
        )

    def apply_impl(self, input_):
        input_parallel = self.get_input_parallel(input_)
        assert self.quant_method is not None
        if self._can_fuse(input_parallel):
            sp_pad = (-input_parallel.shape[0]) % self.tp_size
            if sp_pad:
                input_parallel = F.pad(input_parallel, (0, 0, 0, sp_pad))
            output = DeviceOperator.npu_mm_reduce_scatter_base(
                input_parallel,
                self.layer.weight.t(),
                self.hcom,
                self.tp_size,
                reduce_op="sum",
            )
            if self.bias is not None and not self.skip_bias_add:
                output = output + self.bias
        else:
            bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
            output_parallel = self.quant_method.apply(self.layer, input_parallel, bias=bias_)
            output = sp_reduce_scatter(output_parallel)
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


def _owns_sequence_collective(proj, op_cls) -> bool:
    """Whether ``proj`` carries the fused custom op that owns its boundary collective."""
    return isinstance(getattr(proj, "custom_op", None), op_cls)


def _install_dense_sp_fused_op(proj, op_cls) -> bool:
    """Attach a fused matmul-collective custom op to ``proj`` when eligible.

    Construction-time guards cover the configuration that cannot change
    after the layer is built (an already-installed custom op, TP size,
    quantization, weight layout); the per-forward prefill / dtype guards
    live in the op classes so decode batches fall back without re-tracing.
    """
    if getattr(proj, "custom_op", None) is not None:
        return False
    if not isinstance(getattr(proj, "quant_method", None), UnquantizedLinearMethod):
        return False
    if get_tp_group().world_size not in DENSE_SP_FUSED_TP_SIZES:
        return False
    if op_cls is DenseSpAllGatherMmOp and getattr(proj, "gather_output", False):
        # The fused path has no separate output to gather.
        return False
    weight = getattr(proj, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.dtype not in _DENSE_SP_FUSED_DTYPES:
        return False
    fused_op_name = "npu_all_gather_base_mm" if op_cls is DenseSpAllGatherMmOp else "npu_mm_reduce_scatter_base"
    if not callable(getattr(torch_npu, fused_op_name, None)):
        return False
    proj.custom_op = op_cls(proj)
    return True


def _uses_multimodal_rope(attention: Qwen3NextAttention) -> bool:
    """Return whether a Qwen3.5 attention layer exposes multimodal RoPE."""
    return "qwen3_5" in attention.config.model_type and hasattr(attention.rotary_emb, "mrope_section")


class AscendQwen3NextAttention(Qwen3NextAttention):
    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, output: torch.Tensor = None):
        qkv, _ = self.qkv_proj(hidden_states)
        if _owns_sequence_collective(self.qkv_proj, DenseSpAllGatherMmOp):
            # The projection owns the sequence all-gather (fused into its
            # GEMM, or the unfused fallback inside the same custom op), so
            # it returns the padded full-token rows; trim back to the real
            # token count that RoPE and the attention kernels expect.
            qkv = qkv[: positions.shape[-1]]
        if _uses_multimodal_rope(self):
            # MRV2 MTP uses 1D text positions; the fused kernel reads three planes.
            if positions.ndim == 1:
                positions = positions.unsqueeze(0).expand(3, -1)
            cos_sin = self.rotary_emb.cos_sin_cache[positions]
            if cos_sin.device != qkv.device:
                cos_sin = cos_sin.to(qkv.device)
            if cos_sin.dtype != qkv.dtype:
                cos_sin = cos_sin.to(qkv.dtype)

            q, k, v, gate = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
                qkv=qkv,
                q_weight=1.0 + self.q_norm.weight,
                k_weight=1.0 + self.k_norm.weight,
                cos_sin=cos_sin,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_dim,
                eps=self.config.rms_norm_eps,
                mrope_section=self.rotary_emb.mrope_section,
                is_interleaved=self.rotary_emb.mrope_interleaved,
                rope_dim=self.rotary_emb.rotary_dim,
                has_gate=self.attn_output_gate,
            )
        else:
            if self.attn_output_gate:
                q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
                orig_shape = q_gate.shape[:-1]
                q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
                q, gate = torch.chunk(q_gate, 2, dim=-1)
                q = q.reshape(*orig_shape, -1)
                gate = gate.reshape(*orig_shape, -1)
            else:
                q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(-1, self.num_heads * self.head_dim)
            k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(-1, self.num_kv_heads * self.head_dim)

            q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        out, _ = self.o_proj(attn_output)
        if output is not None:
            output[:] = out
        return out


def _sequence_parallel_dense_mlp_forward(mlp: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Run a TP-sharded dense MLP on the sequence-parallel token shard.

    ``gate_up_proj``/``down_proj`` keep their TP weight shards, so every rank
    must see the same (padded) token rows: all-gather the shard, run the
    column-parallel / row-parallel GEMMs (``down_proj.reduce_results=False``
    leaves partial sums) and reduce-scatter back to the rank-local shard.
    When both projections carry fused matmul-collective custom ops, each
    collective lives inside its projection and this helper reduces to the
    plain GEMM chain on the padded full-token rows.
    """
    if _owns_sequence_collective(mlp.down_proj, DenseSpMmReduceScatterOp):
        gate_up, _ = mlp.gate_up_proj(hidden_states)
        hidden_states = mlp.act_fn(gate_up)
        hidden_states, _ = mlp.down_proj(hidden_states)
        return hidden_states
    hidden_states = sp_all_gather(hidden_states)
    gate_up, _ = mlp.gate_up_proj(hidden_states)
    hidden_states = mlp.act_fn(gate_up)
    hidden_states, _ = mlp.down_proj(hidden_states)
    return sp_reduce_scatter(hidden_states)


class AscendQwen3_5DecoderLayer(Qwen3_5DecoderLayer):
    """Sequence-parallel decoder layer for the Qwen3.5 family.

    Replicates upstream ``Qwen3NextDecoderLayer.forward`` and extends it for
    Ascend sequence parallelism: under SP the attention output projections
    (``o_proj`` / GDN ``out_proj``) leave TP partial sums that this layer
    completes with a reduce-scatter, and a dense ``Qwen3NextMLP`` gathers the
    full sequence around its TP-sharded weights. MoE blocks keep the upstream
    ``already_sequence_parallel=True`` contract. The layout is driven by
    ``use_attn_reduce_scatter_for_moe``, which the patched ``__init__`` below
    also raises for the dense Qwen3.5 family when the FlashComm switch
    enables dense SP.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor = None,
        **kwargs: object,
    ):
        use_sp = getattr(self, "use_attn_reduce_scatter_for_moe", False)
        full_num_tokens = positions.shape[-1]

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if use_sp:
            # Attention (full and GDN) needs the full token sequence; the
            # sequence-parallel output projection above it leaves partial
            # sums that the reduce-scatter below completes on the shard.
            # A qkv projection carrying the fused AllGather+Matmul op owns
            # the gather, so the decoder skips its own collective (the
            # patched attention trims the padded rows the op returns). The
            # GDN in-projections consume the gathered activation in several
            # places inside the mixer core, so GDN layers always gather here.
            qkv_gathers = self.layer_type == "full_attention" and _owns_sequence_collective(
                self.self_attn.qkv_proj, DenseSpAllGatherMmOp
            )
            if not qkv_gathers:
                hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states=hidden_states)
        elif self.layer_type == "full_attention":
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (self.attn_layer_scale.to(hidden_states.dtype)[0] + 1)
            else:
                hidden_states = hidden_states * (self.attn_layer_scale.to(hidden_states.dtype) + 1)

        if use_sp:
            # An output projection carrying the fused Matmul+ReduceScatter
            # op completes the TP partial sums itself and returns the
            # rank-local shard, so the decoder skips its reduce-scatter.
            out_proj = self.linear_attn.out_proj if self.layer_type == "linear_attention" else self.self_attn.o_proj
            if not _owns_sequence_collective(out_proj, DenseSpMmReduceScatterOp):
                hidden_states = sp_reduce_scatter(hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if use_sp:
            if isinstance(self.mlp, Qwen3NextSparseMoeBlock):
                hidden_states = self.mlp(hidden_states, already_sequence_parallel=True)
            else:
                # TP-sharded dense weights must see the same token rows on
                # every rank; restore the padded full sequence for the MLP,
                # then reduce its TP partial output back to the local shard.
                hidden_states = _sequence_parallel_dense_mlp_forward(self.mlp, hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1)
            else:
                assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                    f"shape must be the same {len(hidden_states.shape)}, {len(self.ffn_layer_scale.shape)}"
                )
                hidden_states = hidden_states * (self.ffn_layer_scale.to(hidden_states.dtype) + 1)

        return hidden_states, residual


_QWEN3_5_LAYER_ORIGINAL_INIT = Qwen3_5DecoderLayer.__init__


def _qwen3_5_decoder_layer_init_with_dense_sp(
    self,
    vllm_config: VllmConfig,
    layer_type: str,
    prefix: str = "",
) -> None:
    _QWEN3_5_LAYER_ORIGINAL_INIT(self, vllm_config, layer_type, prefix=prefix)
    if not enable_dense_sp(vllm_config) or self.use_attn_reduce_scatter_for_moe:
        return
    # Dense FlashComm SP reuses the upstream SP-MoE layer contract: the flag
    # drives Qwen3NextModel's entry chunk / exit all-gather, the MTP head's
    # chunk / gather, and this patch's gather / scatter decoder layout.
    self.use_attn_reduce_scatter_for_moe = True
    # The output projections were built with reduce_results=True (the upstream
    # flag is False for dense checkpoints); under SP the decoder layer's
    # reduce-scatter completes the sum instead.
    if layer_type == "linear_attention":
        self.linear_attn.out_proj.reduce_results = False
    else:
        self.self_attn.o_proj.reduce_results = False
    if not isinstance(self.mlp, Qwen3NextSparseMoeBlock):
        self.mlp.down_proj.reduce_results = False

    # Fused matmul-collective ops for prefill-sized unquantized batches: the
    # column side folds the sequence all-gather into the GEMM and the row
    # side folds the reduce-scatter in, so the decoder-level collectives
    # disappear for the layers whose boundary projections took them. The
    # GDN in-projections are deliberately left stock (their gather feeds
    # several consumers inside the mixer core).
    if layer_type == "linear_attention":
        _install_dense_sp_fused_op(self.linear_attn.out_proj, DenseSpMmReduceScatterOp)
    else:
        _install_dense_sp_fused_op(self.self_attn.qkv_proj, DenseSpAllGatherMmOp)
        _install_dense_sp_fused_op(self.self_attn.o_proj, DenseSpMmReduceScatterOp)
    if not isinstance(self.mlp, Qwen3NextSparseMoeBlock):
        gate_up_installed = _install_dense_sp_fused_op(self.mlp.gate_up_proj, DenseSpAllGatherMmOp)
        down_installed = _install_dense_sp_fused_op(self.mlp.down_proj, DenseSpMmReduceScatterOp)
        if gate_up_installed != down_installed:
            # The dense-MLP helper assumes the pair either both own their
            # collective or neither does; unwind a half-installed pair.
            if gate_up_installed:
                self.mlp.gate_up_proj.custom_op = None
            if down_installed:
                self.mlp.down_proj.custom_op = None


if Qwen3_5MultiTokenPredictor is not None:

    def qwen3_5_mtp_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        # Backport upstream Qwen3.5 MTP behavior: the local drafter runs on the
        # last PP stage and should always combine token embeddings with the
        # target hidden states instead of consuming PP intermediate tensors.
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
        inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
        hidden_states = self.fc(hidden_states)
        residual = None

        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[current_step_idx]
        if mtp_layer.use_attn_reduce_scatter_for_moe:
            # SP chunk before decoder to keep residual shape consistent with
            # reduced hidden_states after Qwen3NextDecoderLayer's reduce_scatter.
            from vllm.model_executor.models.utils import sequence_parallel_chunk

            assert hidden_states.shape[0] == positions.shape[-1]
            hidden_states = sequence_parallel_chunk(hidden_states)
            assert residual is None
        hidden_states, residual = mtp_layer(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        if mtp_layer.use_attn_reduce_scatter_for_moe:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states[: positions.shape[-1]]
        return hidden_states

    Qwen3_5MultiTokenPredictor.forward = qwen3_5_mtp_forward


Qwen3NextAttention.forward = AscendQwen3NextAttention.forward
# Sequence-parallel decoder layout (upstream SP-MoE contract plus the dense
# Qwen3.5-family FlashComm SP extension) and the dense-SP layer wiring.
Qwen3_5DecoderLayer.__init__ = _qwen3_5_decoder_layer_init_with_dense_sp
Qwen3_5DecoderLayer.forward = AscendQwen3_5DecoderLayer.forward
_GDN_PATCH_TARGET._split_ba_for_tp = AscendGatedDeltaNetAttention._split_ba_for_tp
_GDN_PATCH_TARGET.get_state_shape = AscendGatedDeltaNetAttention.get_state_shape
_GDN_PATCH_TARGET.get_attn_backend = AscendGatedDeltaNetAttention.get_attn_backend

if get_current_hardware_profile().supports(HardwareCapability.GDN_COMPATIBILITY):
    from vllm_ascend._310p.ops.fla.gdn_310 import AscendGatedDeltaNetAttention310

    _GDN_PATCH_TARGET._forward_core = AscendGatedDeltaNetAttention310._forward_core
    _GDN_PATCH_TARGET.get_state_dtype = AscendGatedDeltaNetAttention310.get_state_dtype
else:
    _GDN_PATCH_TARGET._pack_conv_weights = AscendGatedDeltaNetAttention._pack_conv_weights

    @wraps(_GDN_ORIGINAL_INIT)
    def _gdn_init_with_packed_weight(self, *args, **kwargs):
        _GDN_ORIGINAL_INIT(self, *args, **kwargs)
        initialize_packed_conv_weight(self)

    _GDN_PATCH_TARGET.__init__ = _gdn_init_with_packed_weight
    _GDN_PATCH_TARGET.forward = AscendGatedDeltaNetAttention.forward
    _GDN_PATCH_TARGET._forward_core = AscendGatedDeltaNetAttention._forward_core
    _GDN_PATCH_TARGET._warmup_prefill_kernels = AscendGatedDeltaNetAttention._warmup_prefill_kernels
