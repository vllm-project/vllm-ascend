# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark Open on Ascend: correctness-first bring-up over the upstream ROCm stack.

vLLM owns the Muse Spark configuration, weight mapping and decoder structure
(``vllm.models.muse_spark``). This module reuses the ROCm text stack, which is
the smaller of the two vendor stacks, and replaces each piece that only runs
on CUDA or ROCm with its PyTorch reference:

* RMSNorm: upstream's own fused Triton kernel runs here unchanged, so the
  norms need no subclass. Its one CUDA-specific line sits behind a
  ``tl.constexpr`` that its launcher sets from ``current_platform.is_cuda()``,
  and the rest is plain Triton. The two kernels that fold the residual as well
  stay out of reach: upstream gates those on ``Tensor.is_cuda``, which is
  False for an NPU tensor. They fall back to a sequence of single norms, which
  the kernel above still serves.
* QK-norm and RoPE: upstream rotates through ``RotaryEmbedding.forward_cuda``
  and can insert K/V from a compiled kernel that has no Ascend counterpart.
  Here QK-norm takes the fused RMSNorm kernel with every other norm, and RoPE
  goes to ``AscendRotaryEmbedding``, which ``REGISTERED_ASCEND_OPS`` has
  already substituted. Both are handed fp32 and the result rounds once, as the
  upstream kernel does, and the Ascend attention backend writes the KV cache.
* Attention output gate: the weightless RMSNorm with a sigmoid gate runs
  through the Ascend ``rms_norm_gated`` kernel, as in GLM-5.3-Flash KDA.
* MoE: Muse Spark normalizes every expert's output before the weighted top-k
  sum. The Ascend combine fuses the weighting and the sum, so the norm runs at
  the end of the MLP stage instead, on the rows the combine is about to weigh.
  That is the same function, because the norm reads one row at a time.

Expert parallelism works, because the per-route norm is a single shared
row-wise norm: normalizing a rank's local rows is the same function as
normalizing per route, exactly as it is without expert parallelism. The fused
dispatch/MLP/combine operator (``enable_fused_mc2``) is the one MoE path that
cannot express it, and is rejected.

Tensor parallelism without expert parallelism slices every expert along its
intermediate size, so each rank's down-projection rows are partial sums, and
RMS is not linear in them. As upstream does, the rows are all-reduced over TP
before the norm, and the MoE output then skips its final all-reduce. That
holds whether the checkpoint is quantized or not, because the norm and the
all-reduce travel together in one mixin.

ModelSlim quantized checkpoints are accepted, but no accuracy claim is made
and none has been run: the per-route norm is mixed into whichever Ascend MoE
scheme the quantization description selects, and the fused QKVOGate projection
is declared in this model's packed-module mapping so the scheme lookup can
resolve it from the four tensors the checkpoint keeps separate. The other
quantization configs, EPLB, sequence-parallel MoE, data or prefill context
parallelism without expert parallelism, and the DSpark draft are not
supported yet.
"""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import get_ep_group
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, GateLinear
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.models.muse_spark.amd.model import (
    MuseSpark1xAttention,
    MuseSpark1xDecoderLayer,
    MuseSpark1xForCausalLM,
    MuseSpark1xMoE,
    MuseSparkTorchLinearMethod,
)
from vllm.models.muse_spark.common.layers import (
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from vllm.models.muse_spark.common.moe import (
    MuseSpark1xRouter,
    MuseSparkConsolidatedRoutedExperts,
    needs_pre_norm_all_reduce,
)
from vllm.models.muse_spark.common.omni import (
    OMNI_WEIGHTS_MAPPER,
    MuseSpark1xOmniMixin,
    register_omni_processor,
)

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
from vllm_ascend.ops.triton.kda.kda import rms_norm_gated
from vllm_ascend.quantization.configs.modelslim_config import AscendModelSlimConfig
from vllm_ascend.utils import ASCEND_QUANTIZATION_METHOD

ROUTER_DTYPE = torch.float32
ROUTER_SCORE_NORMALIZATION = "sigmoid"
ROUTER_SCORE_DIVISION_EPS = 1e-15
ATTENTION_OUTPUT_GATE_ACTIVATION = "sigmoid"


class AscendMuseSpark1xAttention(MuseSpark1xAttention):
    """Muse Spark attention with PyTorch QK-norm/RoPE and the Ascend gated norm."""

    def _qk_norm_rope_kv_insert(
        self, qkv: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.use_o_gate:
            q, k, v, og = qkv.split([self.q_size, self.kv_size, self.kv_size, self.q_size], dim=-1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            og = None

        # Like the upstream kernel, norm and rotate in fp32 and round once at
        # the end; rounding the norm output before RoPE would round twice.
        if self.qk_norm is not None:
            q = self.qk_norm(q.reshape(-1, self.head_dim), out_dtype=torch.float32).reshape(-1, self.q_size)
            k = self.qk_norm(k.reshape(-1, self.head_dim), out_dtype=torch.float32).reshape(-1, self.kv_size)

        if self.use_attn_temperature_tuning:
            q = q * self._get_attn_temperature_scale(positions)

        if self.rotary_emb is not None:
            # Called, not forward_native: REGISTERED_ASCEND_OPS substitutes
            # AscendRotaryEmbedding for RotaryEmbedding, so this reaches
            # torch.ops.vllm.npu_rotary_embedding, where the reference is a
            # dozen elementwise kernels in every one of 62 layers.
            #
            # fp32 in, fp32 out, and the rounding below is still the only one.
            # The Triton kernel underneath loads cos/sin as fp32, casts q and k
            # to that, and stores back into their own buffers, so it rotates in
            # fp32 whatever it is handed and the output dtype is the input's.
            # Handing it bf16 instead would round before the rotation and again
            # at the store; fp32 here keeps the single rounding, and also keeps
            # the shared fp32 cos/sin cache in fp32.
            q, k = self.rotary_emb(positions, q.float(), k.float())
        return q.to(v.dtype), k.to(v.dtype), v, og

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.fuse_o_norm_gate:
            # The unfused output norm and gate upstream are plain PyTorch.
            return super().forward(positions, hidden_states)

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, og = self._qk_norm_rope_kv_insert(qkv, positions)
        assert og is not None
        attn_output = self.attn(q, k, v)

        # Same contract as upstream's FLA call: weightless RMSNorm over each
        # head, then multiplied by sigmoid(og).
        attn_output = rms_norm_gated(
            attn_output.reshape(-1, self.num_heads, self.head_dim),
            og.reshape(-1, self.num_heads, self.head_dim).contiguous(),
            None,
            None,
            ATTENTION_OUTPUT_GATE_ACTIVATION,
            eps=self.rms_norm_eps,
        )
        output, _ = self.o_proj(attn_output.reshape(-1, self.num_heads * self.head_dim))
        return output


# Upstream builds these modules inside its own constructors; the Ascend model
# switches each instance to the Ascend subclass once construction finishes.
# The subclasses add no state, so the switch keeps every parameter and buffer.
ASCEND_MODULE_CLASSES: dict[type[nn.Module], type[nn.Module]] = {
    MuseSpark1xAttention: AscendMuseSpark1xAttention,
}


def adopt_ascend_modules(model: nn.Module) -> None:
    """Move every upstream Muse Spark attention onto its Ascend path.

    The norms are not in the table: upstream's own fused Triton kernel serves
    them here, so they need no subclass of ours.
    """
    for module in model.modules():
        ascend_cls = ASCEND_MODULE_CLASSES.get(type(module))
        if ascend_cls is None:
            continue
        module.__class__ = ascend_cls
        if isinstance(module, AscendMuseSpark1xAttention):
            # The Ascend attention backend owns the KV cache write.
            module.fused_qk_norm_rope_kv_insert = False
            module.attn.kv_cache_update_bypass = False


_POST_EXPERT_NORM_MARKER = "_muse_spark_post_expert_norm"


def has_post_expert_norm(obj: Any) -> bool:
    """Whether the Muse Spark per-route norm is installed on *obj*."""
    return bool(getattr(obj, _POST_EXPERT_NORM_MARKER, False))


def adopt_post_expert_norm(method: Any) -> Any:
    """Normalize every route's output before the combine weights and sums it.

    Muse Spark normalizes each expert's output and only then takes the weighted
    top-k sum. The Ascend combine, ``npu_moe_token_unpermute``, fuses the
    weighting and the sum, so there is no seam inside it to normalize at. There
    does not need to be one: the norm is row-wise, ``apply_gmm2`` returns one
    row per route, and a row-wise function commutes with the permutation the
    combine undoes. Normalizing here is therefore the per-route norm, whatever
    order the rows are grouped in.

    The rows are handed back in the activation dtype, not fp32. The combine
    reads ``probs`` in the dtype of its input, so fp32 rows would make it
    return fp32 into a bf16 unpooler. Upstream holds fp32 until after the
    weighted sum, so this rounds once more than upstream does: measured at
    0.21% mean relative error over random rows, against bf16's own 0.39% unit
    in the last place. The error is rounding noise of that order, not a bias.

    Without expert parallelism every expert is sliced along its intermediate
    size, so each TP rank's rows are partial sums, and RMS is not linear in
    them. The rows are then all-reduced over TP before the norm, as upstream's
    ``needs_pre_norm_all_reduce`` path does, in the activation dtype the down
    projection already rounded them to. Every TP rank routes the same tokens
    through the same permutation, so the rows line up across ranks.

    Which object this lands on depends on the checkpoint, so it is chosen here
    rather than fixed by a subclass: the MoE method itself when the weights are
    unquantized, and the quantization scheme the Ascend adapter wraps when they
    are not. Both get the norm, and both get the all-reduce, which a quantized
    checkpoint needs just as much.

    The wrapper is installed on the instance and not by reassigning
    ``__class__``, which the method chain's object layout does not allow.
    ``QuantizeMethodBase`` derives from ``abc.ABC``, whose ``__slots__`` is
    empty, so it is the first class in the chain to add an instance dict;
    ``UnquantizedFusedMoEMethod`` then also derives from ``CustomOp``, an
    ``nn.Module``, which brings a second. A class built by putting any further
    base in front of that pair is not layout-compatible with it, and CPython
    refuses the assignment with "object layout differs" -- whether or not the
    added base declares empty slots of its own. A bound wrapper needs no new
    class, so it does not care.

    The MoE methods that reach here differ by hardware, and the assignment
    does succeed on some of them: the one-card e2e test builds the model and
    passes on A2 while the same code raises on A5. Wrapping works on both, and
    is not a layout away from working on the next one.

    Returns the object the norm was installed on, which is the one that then
    takes ``post_expert_norm_fn``.
    """
    target = method
    if not hasattr(type(target), "apply_gmm2"):
        # The adapter keeps the scheme under the same attribute name it is
        # itself held by, so one step in reaches the object that runs.
        target = getattr(target, "quant_method", None)
        if target is None or not hasattr(type(target), "apply_gmm2"):
            raise RuntimeError(
                "Muse Spark could not find the MoE method that runs the MLP stage, so its "
                f"per-route output norm has nowhere to go; got {type(method).__name__}."
            )
    if has_post_expert_norm(target):
        return target

    # The un-normed stage, bound once. Held in the closure rather than looked
    # up again, so the wrapper cannot find itself if this ever runs twice.
    inner = target.apply_gmm2

    def apply_gmm2(mlp_compute_input, hidden_states, act_out_scale, _owner=target, _inner=inner):
        routed_rows = _inner(mlp_compute_input, hidden_states, act_out_scale)
        if _owner.post_expert_norm_fn is None:
            raise RuntimeError("Muse Spark routed experts ran without their post-expert norm.")
        if _owner.pre_norm_all_reduce:
            routed_rows = tensor_model_parallel_all_reduce(routed_rows)
        return _owner.post_expert_norm_fn(routed_rows, out_dtype=routed_rows.dtype)

    # Attached by the owning MoE once the factory has built the layer, as
    # upstream attaches its own post_expert_norm_fn to the routed experts.
    # The bound method is held, not the module: the host can be an
    # ``nn.Module``, so holding the norm itself would register its weight a
    # second time and put a key in the state dict no checkpoint carries.
    target.post_expert_norm_fn = None
    # Set by AscendMuseSparkRoutedExperts, which is the one place holding both
    # the MoE config and this object.
    target.pre_norm_all_reduce = False
    target.apply_gmm2 = apply_gmm2
    setattr(target, _POST_EXPERT_NORM_MARKER, True)
    return target


class AscendMuseSparkRoutedExperts(MuseSparkConsolidatedRoutedExperts, AscendRoutedExperts):
    """Routed experts whose MLP stage ends in the Muse Spark per-route norm.

    The checkpoint stores each layer's experts as two stacked tensors,
    ``w13_weight [E, 2I, H]`` and ``w2_weight [E, H, I]``. The generic
    ``RoutedExperts.load_weights`` only knows ``gate_up_proj`` / ``down_proj``
    and per-expert names, so it skips these without an error, and the loader's
    unloaded-weight check exempts MoE parameters: the experts would run on
    uninitialized memory. ``MuseSparkConsolidatedRoutedExperts`` comes first
    for its ``load_weights``, which hands each expert's gate, up and down
    slices to the expert weight loader, as upstream's routed experts do; that
    loader keeps this rank's experts and TP slice. Everything else is
    ``AscendRoutedExperts``.

    The MoE method is not replaced, because for a quantized checkpoint the
    quantization description picks it: the per-route norm is mixed into
    whichever method is already installed, and the pre-norm all-reduce is set
    on that same object. Setting it here rather than in the mixin is what
    makes the all-reduce reach a quantized checkpoint at all -- the mixin is
    installed by reassigning ``__class__``, so it has no constructor to read
    the MoE config in, and this is the one place holding both that config and
    the object the norm landed on.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adopt_mlp_stage()

    def adopt_mlp_stage(self) -> Any:
        """Install the per-route norm on the live MLP stage and tell it whether
        its rows are TP partials.

        Both belong on the same object and neither can be set in the mixin:
        it is installed by reassigning ``__class__`` on a method the factory
        already built, so it has no constructor of ours to run. This is the
        one place that holds the MoE config and the object the norm landed on.
        """
        method = adopt_post_expert_norm(self.quant_method)
        method.pre_norm_all_reduce = needs_pre_norm_all_reduce(self.moe_config)
        return method

    def mlp_stage_method(self) -> Any:
        """The object whose ``apply_gmm2`` the MLP stage calls.

        Resolved on each call rather than stored: the hosts can be
        ``nn.Module``s, and holding one as an attribute here would register it
        as a submodule and duplicate its parameters in the state dict.
        """
        return adopt_post_expert_norm(self.quant_method)

    def attach_post_expert_norm(self, norm_fn: Callable[..., torch.Tensor]) -> None:
        """Hand the per-route norm to the MLP stage; it raises without one."""
        self.mlp_stage_method().post_expert_norm_fn = norm_fn


class AscendMuseSparkMoERunner(AscendMoERunner):
    """The Ascend MoE runner, without the final all-reduce the experts already did.

    Once the routed rows are summed over TP ahead of the per-route norm, the
    combined output is complete on every rank, and reducing it again would
    multiply it by the TP size. Upstream's ``MuseMoERunner`` skips that
    all-reduce the same way. Everything else, the comm methods and the
    combine included, is the Ascend runner's. Both overrides read only the
    MoE config, so they are static properties of the layer and safe to
    capture in a graph.
    """

    @property
    def _fused_output_is_reduced(self) -> bool:
        return needs_pre_norm_all_reduce(self.moe_config) or super()._fused_output_is_reduced

    def _maybe_reduce_final_output(
        self,
        states: torch.Tensor,
        trunc_size: int | None,
        output_is_reduced: bool | None = None,
    ) -> torch.Tensor:
        if not needs_pre_norm_all_reduce(self.moe_config):
            return super()._maybe_reduce_final_output(states, trunc_size, output_is_reduced)
        if trunc_size is not None and trunc_size > 0:
            return states[..., :trunc_size]
        return states


class MuseSparkReferenceExperts(nn.Module):
    """Routed experts as a dense PyTorch reference with the per-route norm.

    No longer on the model path: the MoE runs through ``FusedMoEFactory``. This
    stays as the executable statement of what the routed experts compute, which
    the Ascend path is checked against, and it is small enough to step through
    when that path disagrees.

    Parameter names and shapes follow the consolidated checkpoint tensors,
    ``w13_weight [E, 2I, H]`` (gate rows, then up rows) and
    ``w2_weight [E, H, I]``, the same layout ``FusedMoEFactory`` allocates.
    Every token runs through every expert, so compute grows with the expert
    count; the shapes stay static and no device-to-host sync is needed.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.w13_weight = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size, dtype=params_dtype),
            requires_grad=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        post_expert_norm: MuseSpark1xRMSNorm,
    ) -> torch.Tensor:
        # hidden_states [T, D], topk_weights/topk_ids [T, K]; D is the expert width.
        gate_up = torch.einsum("td,eod->teo", hidden_states, self.w13_weight)  # [T, E, 2I]
        gate, up = gate_up.chunk(2, dim=-1)
        activated = (F.silu(gate.float()) * up.float()).to(hidden_states.dtype)  # [T, E, I]
        expert_rows = torch.einsum("tei,edi->ted", activated, self.w2_weight)  # [T, E, D]

        route_index = topk_ids.long().unsqueeze(-1).expand(-1, -1, self.hidden_size)
        route_rows = expert_rows.gather(1, route_index)  # [T, K, D]
        # Normalize each route's output before the weighted sum.
        normalized = post_expert_norm(route_rows, out_dtype=torch.float32)
        combined = (normalized * topk_weights.float().unsqueeze(-1)).sum(dim=1)  # [T, D]
        return combined.to(hidden_states.dtype)


class AscendMuseSpark1xMoE(MuseSpark1xMoE):
    """Muse Spark latent MoE on the Ascend fused-MoE path.

    The upstream constructor is not called: it pins a CUDA/ROCm runner and a
    MoE config that defers the top-k finalize, neither of which applies here.
    The factory is built again with the Ascend routed experts and the Ascend
    runner. Router, pooler, unpooler and the two expert norms keep their
    upstream names, so checkpoints map onto the same keys.
    """

    def __init__(self, vllm_config: VllmConfig, layer_idx: int, prefix: str = "") -> None:
        nn.Module.__init__(self)

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe
        if self.is_sequence_parallel:
            raise NotImplementedError("Muse Spark on Ascend does not support sequence-parallel MoE yet.")
        # The per-route norm lives at the end of the MLP stage, which the fused
        # dispatch/MLP/combine operator does not expose. Until that norm can run
        # inside the fused kernel, the fused path would silently drop it.
        if get_ascend_config().enable_fused_mc2 == 1:
            raise NotImplementedError(
                "Muse Spark on Ascend does not support enable_fused_mc2 yet: that path fuses "
                "dispatch, the MLP stage and the combine into a single operator, leaving nowhere "
                "to apply the per-expert output norm."
            )
        if parallel_config.enable_eplb:
            raise NotImplementedError("Muse Spark on Ascend does not support EPLB yet.")

        # vLLM builds the EP group over all TP x PCP x DP ranks of every MoE
        # model; only enable_expert_parallel shards the experts across it, so
        # the bookkeeping below reads the flag rather than the group size.
        self.use_ep = parallel_config.enable_expert_parallel
        ep_group = get_ep_group()
        self.ep_group = ep_group.device_group
        self.ep_rank = ep_group.rank_in_group if self.use_ep else 0
        self.ep_size = self.ep_group.size() if self.use_ep else 1
        # Without expert parallelism the experts are sliced over TP x DP x PCP,
        # but the rows are summed before the norm over TP only, as upstream's
        # are; upstream rejects DP and PCP here too.
        if not self.use_ep and (
            parallel_config.data_parallel_size > 1 or parallel_config.prefill_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "Muse Spark on Ascend needs expert parallelism under data or prefill context parallelism."
            )

        self.layer_idx = layer_idx
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.n_routed_experts = config.num_local_experts
        self.n_logical_experts = config.num_local_experts
        # EPLB is rejected above, so there are no redundant experts to place.
        self.n_redundant_experts = 0
        self.n_physical_experts = self.n_logical_experts
        if self.n_physical_experts % self.ep_size:
            raise ValueError(
                f"Muse Spark on Ascend needs the {self.n_physical_experts} experts to divide "
                f"evenly across the {self.ep_size}-rank expert-parallel group."
            )
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = self.physical_expert_start + self.n_local_physical_experts
        self.n_shared_experts = config.num_shared_experts

        self.router_dtype = ROUTER_DTYPE
        self.router = GateLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            out_dtype=self.router_dtype,
            params_dtype=self.router_dtype,
            prefix=f"{prefix}.router",
        )
        self.register_buffer(
            "expert_bias",
            torch.zeros(config.num_local_experts, dtype=torch.float32),
        )
        self.route_scale = config.routed_scaling_factor
        self.router_output_multiplier = config.output_multiplier
        self._router_multiplier_folded = False
        self.router.weight.weight_loader = self._load_router_weight
        self.moe_router = MuseSpark1xRouter(
            top_k=config.num_experts_per_tok,
            num_experts=config.num_local_experts,
            expert_bias=self.expert_bias,
            route_scale=self.route_scale,
            router_output_multiplier=self.router_output_multiplier,
            router_score_normalization=ROUTER_SCORE_NORMALIZATION,
            router_score_division_eps=ROUTER_SCORE_DIVISION_EPS,
        )

        moe_hidden_size = config.moe_hidden_size
        self.pooler = ReplicatedLinear(
            config.hidden_size,
            moe_hidden_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.pooler",
        )
        self.pooler.quant_method = MuseSparkTorchLinearMethod()
        self.unpooler = ReplicatedLinear(
            moe_hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.unpooler",
        )
        self.unpooler.quant_method = MuseSparkTorchLinearMethod()
        self.pre_expert_norm = MuseSpark1xRMSNorm(
            hidden_size=moe_hidden_size,
            eps=config.rms_norm_eps,
            has_weight=True,
            is_post_norm=False,
            zero_centered_gamma=True,
            post_norm_gain_center_type="one",
            residual_id=2 * layer_idx + 2,
            output_norm_gain_center_type="zero",
        )
        self.post_expert_norm = MuseSpark1xRMSNormWithInputScale(
            hidden_size=moe_hidden_size,
            eps=config.post_norm_eps,
            has_weight=True,
            is_post_norm=True,
            zero_centered_gamma=True,
            post_norm_gain_center_type="one",
            residual_id=None,
            output_norm_gain_center_type="zero",
        )

        self.intermediate_size = config.moe_intermediate_size
        # The experts run at the pooled width; the router still reads the full
        # hidden size, so its logits are computed here and passed in.
        self.experts = FusedMoEFactory(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=moe_hidden_size,
            intermediate_size=self.intermediate_size,
            renormalize=True,
            router=self.moe_router,
            # The experts are the one module here that honours the checkpoint's
            # quantization. Passing None instead builds them through the
            # unquantized method, which allocates bf16 for every expert it was
            # handed: 3GiB a layer, 186GiB a card for this model, against the
            # 50GiB its w4a8 weights occupy. Nothing reports it -- the layer
            # simply asks for memory no card has. The pooler and unpooler above
            # keep None on purpose, because their method is replaced outright
            # on the next line.
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.experts",
            is_sequence_parallel=self.is_sequence_parallel,
            router_logits_dtype=self.router_dtype,
            scoring_func=ROUTER_SCORE_NORMALIZATION,
            e_score_correction_bias=self.expert_bias,
            routed_scaling_factor=self.route_scale,
            routed_experts_cls=AscendMuseSparkRoutedExperts,
            runner_cls=AscendMuseSparkMoERunner,
        )
        self.experts.routed_experts.attach_post_expert_norm(self.post_expert_norm.forward)

    def _pool(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.pooler(hidden_states)
        return self.pre_expert_norm(pooled)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits = self._router_logits(hidden_states)  # [T, E] fp32, full width
        pooled = self._pool(hidden_states)  # [T, moe_hidden]
        experts_out = self.experts(hidden_states=pooled, router_logits=router_logits)
        output, _ = self.unpooler(experts_out)
        return output


class AscendMuseSpark1xDecoderLayer(MuseSpark1xDecoderLayer):
    moe_cls = AscendMuseSpark1xMoE


class AscendMuseSpark1xForCausalLM(MuseSpark1xForCausalLM):
    decoder_layer_cls = AscendMuseSpark1xDecoderLayer
    # Upstream's __init__ refuses any quantization outside this tuple, and fills
    # it with the ROCm Quark configs. It reads the tuple off ``self``, so naming
    # the ModelSlim config here is how a port declares what it accepts. The
    # check below is the one that produces the message, but without this a
    # ModelSlim checkpoint never reaches it: super().__init__() raises first,
    # and talks about Quark. The two lists agree by construction, because
    # AscendModelSlimConfig.get_name() is what returns
    # ASCEND_QUANTIZATION_METHOD.
    supported_quantization_configs = (AscendModelSlimConfig,)
    supported_quantization_description = f"ModelSlim ({ASCEND_QUANTIZATION_METHOD})"

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[MuseSpark1xDecoderLayer] | None = None,
    ) -> None:
        quant_config = vllm_config.quant_config
        if quant_config is not None and quant_config.get_name() != ASCEND_QUANTIZATION_METHOD:
            # ModelSlim checkpoints are the only quantized ones wired up: the
            # per-route norm is mixed into whichever Ascend MoE scheme the
            # description selects (see adopt_post_expert_norm), and the
            # fused QKVOGate projection is resolved through this model's
            # packed-module mapping. The other quantization configs reach
            # different method classes and have had neither done for them.
            raise NotImplementedError(
                "Muse Spark on Ascend supports unquantized (BF16) checkpoints and ModelSlim "
                f"({ASCEND_QUANTIZATION_METHOD!r}) ones; got {quant_config.get_name()!r}."
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)
        adopt_ascend_modules(self)


@register_omni_processor
class AscendMuseSparkForConditionalGeneration(MuseSpark1xOmniMixin, AscendMuseSpark1xForCausalLM, SupportsMultiModal):
    """Muse Spark Omni: the upstream vision and audio towers over the Ascend text stack."""

    hf_to_vllm_mapper = MuseSpark1xForCausalLM.hf_to_vllm_mapper | OMNI_WEIGHTS_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # The towers are built after the text stack; cover their norms too.
        adopt_ascend_modules(self)


__all__ = [
    "AscendMuseSpark1xForCausalLM",
    "AscendMuseSparkForConditionalGeneration",
]
