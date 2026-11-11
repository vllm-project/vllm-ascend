#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Ascend adapter for Mistral Large 4.

Upstream builds the model for CUDA and ROCm. Two groups of things do not
carry over to NPU, and this module replaces both.

**The hand-fused Triton kernels need no work.** Upstream gates every one of
them on ``current_platform.is_rocm()``
(``vllm.models.mistral.amd.ops.fusion_enabled``), so on Ascend ``ssmax``,
``mla_rope_ssmax``, ``indexer_q``, ``indexer_k``, ``indexer_q_norm`` and
``indexer_k_cache`` are already off and the torch reference paths run. In
particular ``SSMaxScaling.forward`` falls back to ``q * (log_pos * s + b)``,
so scalable softmax is correct here without a new kernel.

What this module does replace:

* **The indexer / QKV stream overlap.** Upstream forks the indexer onto an aux
  CUDA stream and joins it with ``torch.cuda.Event``s. Neither exists on NPU,
  so :class:`AscendMistralMLAAttention` runs the two serially. The math is
  identical; only the overlap is lost.
* **``record_logical_topk_ready``.** Upstream calls it on the attention impl
  for sparse layers. The Ascend SFA impl has no such method, so the call is
  made only where it exists.
* **The KV-sharing write.** See :func:`adopt_shared_kv_skip`.

Everything else — MLA, the DSA indexer, the shared-indexer schedule, NoPE,
YaRN, the MoE block and the FP8 block-scaled checkpoint — is carried by paths
vLLM-Ascend already ships. Backend selection needs no patch: a config with
``index_topk`` and no ``compress_ratios`` resolves to ``AscendSFABackend``.
"""

from collections.abc import Sequence
from typing import Any, cast

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_ep_group
from vllm.logger import logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mla import MLAModules
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.models.mistral.amd.attention import MistralMLAAttention
from vllm.models.mistral.amd.indexer import MistralIndexer
from vllm.models.mistral.amd.ssmax import SSMaxScaling
from vllm.models.mistral.common.layer_spec import MistralLayerSpec
from vllm.models.mistral.mistral_large4.amd.dspark import (
    MistralLarge4DSparkForCausalLM,
)
from vllm.models.mistral.mistral_large4.amd.model import (
    MistralLarge4ForCausalLM,
)
from vllm.v1.attention.backends.mla.index_group import SparseMLAIndexGroupBuilder

from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
from vllm_ascend.ops.mla import AscendMultiHeadLatentAttention
from vllm_ascend.ops.rotary_embedding import (
    _record_cos_and_sin_cache_interleaved,
    _record_cos_sin_cache,
    gather_mla_rope,
)

__all__ = [
    "AscendMistralIndexerBackend",
    "AscendMistralIndexer",
    "AscendMistralLarge4DSparkForCausalLM",
    "AscendMistralLarge4ForCausalLM",
    "AscendMistralMLAAttention",
    "adopt_ascend_modules",
    "adopt_shared_kv_skip",
    "has_shared_kv_skip",
    "prepare_indexer_for_ascend",
    "validate_moe_parallelism",
]


_SHARED_KV_SKIP_MARKER = "_ascend_shared_kv_skip"


def _produces_topk_for_shared_layers(config: Any, spec: Any) -> bool:
    """Whether this indexer layer feeds a following shared-index layer."""
    index_sharing = getattr(config, "index_sharing", None)
    next_layer = spec.layer_idx + 1
    return bool(
        spec.has_indexer
        and index_sharing is not None
        and next_layer < len(index_sharing)
        and index_sharing[next_layer] == "shared"
    )


def has_shared_kv_skip(impl: Any) -> bool:
    """Whether the KV-sharing write skip is installed on *impl*."""
    return bool(getattr(impl, _SHARED_KV_SKIP_MARKER, False))


def adopt_shared_kv_skip(attn: nn.Module) -> None:
    """Stop a KV-sharing layer from writing over the cache it borrows.

    Nine of the 72 layers hold no KV compression weights: they read layer 35's
    576-wide latent and expand it with their own ``kv_b_proj``. Upstream
    expresses that by handing those layers ``kv_sharing_target_layer_name`` and
    feeding them a never-written zero latent.

    On Ascend that argument is only half honoured. ``AscendSFAImpl`` reads it
    when it resolves *indexer* metadata, which is what we want — the borrowing
    layer should read the source layer's indexer cache. But the main latent
    write runs unconditionally through ``_get_sfa_kv_slot_mapping``, so the
    layer would scatter its zero latent into the slots it is supposed to be
    reading. The failure is silent: the model runs and produces fluent, wrong
    output.

    (``AscendMLAImpl`` is worse: it accepts ``kv_sharing_target_layer_name``
    in its constructor and never reads it at all. Mistral Large 4 selects the
    SFA backend, so that path is not taken here.)

    Narrowing the slot mapping to zero rows is the one hook that covers every
    write site, since all of them route through this method.

    The override is installed on the instance, not by reassigning ``__class__``
    to a subclass carrying a mixin. That reassignment is refused for this
    chain: ``AttentionImplBase`` derives from ``abc.ABC`` and ``Generic`` and
    declares no slots, so it is the class that adds the instance dict, and a
    mixin in front of it adds one too. CPython compares the two layouts and
    raises::

        TypeError: __class__ assignment:
        'AscendSharedKVAscendSFAImpl' object layout differs from 'AscendSFAImpl'

    A function in the instance dict shadows the method for this impl alone and
    needs no class. Stored unbound on purpose: attribute lookup from ``self``
    finds it before the class attribute and does not bind it, so it receives
    the single argument the call site passes.
    """
    inner = getattr(attn, "attn", None)
    impl = getattr(inner, "impl", None)
    if impl is None:
        impl = getattr(getattr(inner, "mla_attn", None), "impl", None)
    if impl is None:
        raise ValueError(
            "KV-sharing layer has no attention impl to install the write skip on; "
            "refusing to run, because the layer would otherwise overwrite the "
            "latent cache it borrows."
        )
    if has_shared_kv_skip(impl):
        return

    def _get_sfa_kv_slot_mapping(attn_metadata: Any) -> torch.Tensor:
        # A zero-row mapping, not a sentinel value: the Ascend scatter writes
        # at the indices it is given, so -1 would land on the last slot.
        return attn_metadata.slot_mapping[:0]

    impl._get_sfa_kv_slot_mapping = _get_sfa_kv_slot_mapping
    setattr(impl, _SHARED_KV_SKIP_MARKER, True)


class AscendMistralMLAAttention(MistralMLAAttention):
    """Upstream attention with the CUDA stream overlap and indexer call removed.

    Two changes, and the second is a correctness one.

    Upstream runs the indexer on an aux stream so it overlaps the Q/KV
    projections, joining on two ``torch.cuda.Event``s. vLLM-Ascend uses
    ``torch.npu.Event`` and ``npu_stream_switch``; upstream's helper
    (``vllm.utils.multi_stream_utils.execute_in_parallel``) has no Ascend
    counterpart.

    More importantly, **the model must not call the indexer at all here.** On
    CUDA, ``MLAAttention`` does not run it, so upstream's attention module
    does. ``AscendSFAImpl`` does run it, from inside ``self.attn``. Calling it
    in both places would run the k path and the cache write twice per step.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Any,
        spec: Any,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None,
        index_group_builder: SparseMLAIndexGroupBuilder | None,
        zero_kv_inputs: tuple[torch.Tensor, torch.Tensor],
        quant_config: QuantizationConfig | None = None,
        non_causal: bool = False,
        sliding_window: int | None = None,
        reduce_results: bool = True,
    ) -> None:
        # Do not call MistralMLAAttention.__init__: it constructs the inner
        # generic MLAAttention before this adapter can supply the outer MLA
        # modules required by AscendMLAImpl/AscendSFAImpl.
        nn.Module.__init__(self)
        if quant_config is None:
            quant_config = vllm_config.quant_config
        tp_size = get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % tp_size == 0
        assert config.q_lora_rank is None, "q-LoRA is not supported"

        self.spec = spec
        self.num_local_heads = config.num_attention_heads // tp_size
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim if spec.use_rope else self.qk_head_dim
        self.v_head_dim = config.v_head_dim
        self.zero_kv_c, self.zero_k_pe = zero_kv_inputs

        self.q_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        if not spec.shares_kv:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                config.hidden_size,
                self.kv_lora_rank + (self.qk_rope_head_dim if spec.use_rope else 0),
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )
            self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            config.num_attention_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.v_head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.o_proj",
        )
        self.ssmax = (
            SSMaxScaling(self.num_local_heads, tp_sharded=True) if getattr(config, "ssmax_scaling", False) else None
        )
        self.fuse_rope_ssmax = False
        self.indexer = (
            AscendMistralIndexer(
                vllm_config,
                config,
                spec,
                cast(torch.Tensor, topk_indices_buffer),
                prefix=f"{prefix}.indexer",
            )
            if spec.has_indexer
            else None
        )
        # Upstream fills these with an aux CUDA stream and its events; neither
        # exists on NPU, so they stay empty and the two run serially. Annotated
        # because nothing here ever puts an element in to infer from.
        self.indexer_stream = None
        self.indexer_events: list[Any] = []

        kv_sharing_target_layer_name = None
        if spec.shares_kv:
            layers_prefix = prefix.rsplit(".", 2)[0]
            kv_sharing_target_layer_name = f"{layers_prefix}.{spec.kv_source_layer}.self_attn.attn"

        mla_modules = MLAModules(
            kv_a_layernorm=getattr(self, "kv_a_layernorm", None),
            kv_b_proj=self.kv_b_proj,
            rotary_emb=None,
            o_proj=self.o_proj,
            fused_qkv_a_proj=None,
            kv_a_proj_with_mqa=getattr(self, "kv_a_proj_with_mqa", None),
            q_a_layernorm=None,
            q_b_proj=None,
            q_proj=self.q_proj,
            indexer=self.indexer,
            is_sparse=spec.use_sparse,
            topk_indices_buffer=topk_indices_buffer,
            index_group_builder=index_group_builder,
        )
        self.attn = AscendMultiHeadLatentAttention(
            hidden_size=config.hidden_size,
            num_heads=self.num_local_heads,
            scale=self.qk_head_dim**-0.5,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=None,
            kv_lora_rank=self.kv_lora_rank,
            mla_modules=mla_modules,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=prefix,
            skip_topk=spec.use_sparse and not spec.has_indexer,
            non_causal_multi_token_decode=non_causal,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            sliding_window=sliding_window,
        )
        impl = self.attn.mla_attn.impl
        impl.mistral_ssmax = self.ssmax
        impl.mistral_use_rope = spec.use_rope
        impl.mistral_shares_kv = spec.shares_kv
        if _produces_topk_for_shared_layers(config, spec):
            if topk_indices_buffer is None:
                raise ValueError(
                    f"Mistral shared-index layers require topk_indices_buffer from producer layer {spec.layer_idx}."
                )
            # Ascend's indexer returns its TopK tensor directly. Persist only
            # the producer whose next layer reuses it; otherwise the shared
            # model-wide buffer remains uninitialized when skip_topk reads it.
            impl.use_index_cache = True

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        log_positions: torch.Tensor,
        rotary_emb: Any,
    ) -> torch.Tensor:
        del rotary_emb
        impl = self.attn.mla_attn.impl
        # The Ascend SFA impl runs the indexer itself and its generic contract
        # carries no log positions. Hand both SSMax users the current step's
        # values before entering the custom op.
        impl.mistral_log_positions = log_positions
        backend = getattr(getattr(impl, "indexer", None), "impl", None)
        if isinstance(backend, AscendMistralIndexerBackend):
            backend.log_positions = log_positions
        return self.attn(positions, hidden_states)


class AscendMistralIndexer(MistralIndexer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Any,
        spec: MistralLayerSpec,
        topk_indices_buffer: torch.Tensor,
        prefix: str,
    ) -> None:
        super().__init__(vllm_config, config, spec, topk_indices_buffer, prefix)
        self.indexer_use_rope = spec.indexer_use_rope
        prepare_indexer_for_ascend(self, config)

        def _get_attn_backend():
            return AscendSFAIndexerBackend

        self.k_cache.get_attn_backend = _get_attn_backend

    def get_ascend_indexer_backend_cls(self):
        return AscendMistralIndexerBackend


class AscendMistralIndexerBackend(AscendSFAIndexerBackend):
    """The Ascend DSA indexer, driven from Mistral's indexer module.

    ``AscendSFAIndexerBackend`` is not a reimplementation of the indexer: it
    borrows the upstream module's weights by attribute name and calls them.
    ``MistralIndexer`` already carries ``n_head``, ``head_dim``, ``wq_b``,
    ``wk_weights_proj`` and ``k_norm`` under exactly those names, because both
    designs descend from DeepSeek-V3.2's DSA. So the k path needs nothing: the
    base class calls ``wk_weights_proj`` and splits at ``head_dim``, which is
    Mistral's layout, and it calls ``k_norm`` as a module, so Mistral's
    ``LayerNorm`` with a bias costs nothing.

    Two things differ, and this subclass covers both.

    **The query.** DeepSeek's indexer shares the MLA's ``q_c`` — the output of
    ``q_a_layernorm`` over a q-LoRA projection. Mistral has no q-LoRA, so the
    fused prolog yields no ``q_c``; its indexer instead owns a ``q_input_norm``
    over the hidden states. The two are therefore decoupled, which is simpler:
    this backend computes ``q_input_norm(hidden_states)`` and ignores whatever
    ``q_c`` it is handed.

    **Scalable softmax.** Upstream scales the indexer query by
    ``s * log(pos + 1) + b`` after ``wq_b``. The Ascend ``forward`` has no
    ``log_positions`` parameter, so there is no hook; the scale is applied
    here, in the same place and the same order as upstream.
    """

    def __init__(self, vllm_indexer: AscendMistralIndexer, qk_rope_head_dim: int) -> None:
        super().__init__(vllm_indexer, qk_rope_head_dim if vllm_indexer.indexer_use_rope else 0)
        self.is_rope_neox_style = False
        # Kept so the q path and SSMax can run from the upstream module.
        self.q_input_norm = vllm_indexer.q_input_norm
        self.ssmax = vllm_indexer.ssmax
        # Set once per step by the attention module, which is the last place
        # that holds the step's log positions before the Ascend impl takes
        # over. The Ascend indexer signature has nowhere to carry them.
        self.log_positions: torch.Tensor | None = None

    def mistral_q(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``q_input_norm``, then ``wq_b``, then SSMax — upstream's order."""
        q = self.wq_b(self.q_input_norm(hidden_states))[0]
        q = q.view(-1, self.n_head, self.head_dim)
        if self.ssmax is None:
            return q
        if self.log_positions is None:
            raise RuntimeError(
                "Mistral Large 4 on Ascend: scalable softmax needs the step's "
                "log(positions + 1), which the attention module sets before "
                "the indexer runs. It was not set, so the query would go "
                "unscaled and long-context attention would flatten."
            )
        return self.ssmax(q, self.log_positions)

    def project_query(
        self,
        q_c: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        del q_c
        return self.apply_indexer_rope(self.mistral_q(hidden_states), cos, sin)


def prepare_indexer_for_ascend(indexer: nn.Module, config: Any) -> nn.Module:
    """Give ``MistralIndexer`` the three scalars the Ascend backend reads.

    ``AscendSFAIndexerBackend.__init__`` reads ``topk_tokens``,
    ``softmax_scale`` and ``q_lora_rank`` off the module it wraps. Mistral's
    indexer defines none of them:

    * ``topk_tokens`` is genuinely used (it sizes the top-k output and clamps
      the per-token lengths), and is ``config.index_topk``.
    * ``softmax_scale`` and ``q_lora_rank`` are assigned in the backend's
      constructor and never read anywhere in the file. They are set here so
      construction succeeds, with values that stay correct if that changes:
      the usual DSA scale, and ``None`` for a model with no q-LoRA — which is
      the value ``AscendSFAImpl`` already branches on.
    """
    indexer.topk_tokens = config.index_topk
    indexer.q_lora_rank = None
    indexer.softmax_scale = indexer.head_dim**-0.5
    return indexer


def adopt_ascend_modules(model: nn.Module) -> int:
    """Switch every attention module to the Ascend subclass, in place.

    Returns the number of KV-sharing layers the write skip was installed on, so
    a caller can check it against the config rather than trusting the walk.

    The subclass adds no state, so switching ``__class__`` after construction
    leaves parameters and buffers alone. The aux stream and its events are
    dropped here as well: they are built in upstream's constructor and are
    never used once the forward is serial.
    """
    shared = 0
    for module in model.modules():
        if not isinstance(module, MistralMLAAttention):
            continue
        if not isinstance(module, AscendMistralMLAAttention):
            module.__class__ = AscendMistralMLAAttention
        module.indexer_stream = None
        module.indexer_events = []
        if module.spec.shares_kv:
            adopt_shared_kv_skip(module)
            shared += 1
    return shared


def _reject_unsupported(vllm_config: VllmConfig) -> None:
    """Refuse the deployments this adapter has not been built for.

    Each of these would otherwise fail somewhere less obvious, or silently
    produce wrong numbers.
    """
    parallel_config = vllm_config.parallel_config
    compilation_config = vllm_config.compilation_config

    if parallel_config.enable_eplb:
        raise NotImplementedError(
            "EPLB is not supported for Mistral Large 4 on Ascend: redundant "
            "expert placement is bookkeeping this adapter does not do."
        )
    if getattr(compilation_config.pass_config, "fuse_allreduce_rms", False):
        # defers_allreduce() would then leave sublayer outputs TP-partial and
        # fold the reduction into a fused CUDA/ROCm collective RMSNorm.
        raise NotImplementedError(
            "fuse_allreduce_rms is not supported for Mistral Large 4 on "
            "Ascend: the fused all-reduce RMSNorm it defers to is a "
            "CUDA/ROCm kernel."
        )
    if parallel_config.use_sequence_parallel_moe:
        raise NotImplementedError("Sequence-parallel MoE is not supported for Mistral Large 4 on Ascend.")


def validate_moe_parallelism(vllm_config: VllmConfig) -> None:
    """Check the MoE block against the layout it will actually run under.

    The MoE itself needs no rewiring: vLLM-Ascend overrides ``MoERunner`` and
    ``RoutedExperts`` in the out-of-tree custom-op registry, so upstream's
    ``FusedMoEFactory`` already builds the Ascend experts, router and
    dispatch/combine. Unlike Muse Spark there is no per-route norm to thread
    through the MLP stage, so ``enable_fused_mc2`` is left alone here: nothing
    needs an ``apply_gmm2`` boundary to survive.

    What is worth checking up front is the expert count against the
    expert-parallel group, because the failure otherwise surfaces inside the
    dispatch as a shape error.
    """
    parallel_config = vllm_config.parallel_config
    hf_config = vllm_config.model_config.hf_config
    num_experts = getattr(hf_config, "n_routed_experts", None)
    if not num_experts:
        return

    # vLLM builds the EP group over every MoE model's ranks either way; only
    # enable_expert_parallel actually shards the experts across it.
    if parallel_config.enable_expert_parallel:
        ep_size = get_ep_group().world_size
        if num_experts % ep_size:
            raise ValueError(
                f"Mistral Large 4 on Ascend needs its {num_experts} experts to "
                f"divide evenly across the {ep_size}-rank expert-parallel "
                f"group."
            )

    # Expert width 6144 is outside the MegaMoe window ([256, 4096] aligned to
    # 128), so the fused MoE declines and the generic grouped-matmul path runs.
    # That is correct, only slower, and the warning MegaMoe logs on its own
    # says nothing about why this model in particular misses it.
    moe_width = getattr(hf_config, "moe_intermediate_size", None)
    if moe_width is not None and not (256 <= int(moe_width) <= 4096):
        logger.info(
            "Mistral Large 4: expert width %s is outside the MegaMoe window "
            "[256, 4096], so the routed experts run on the generic "
            "grouped-matmul path rather than the fused one.",
            moe_width,
        )


@support_torch_compile
class AscendMistralLarge4ForCausalLM(MistralLarge4ForCausalLM):
    """Mistral Large 4 with the Ascend paths adopted after construction.

    Upstream builds the attention modules inside its own constructors and
    offers no hook for the attention class, so they are switched afterwards.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        _reject_unsupported(vllm_config)
        validate_moe_parallelism(vllm_config)
        # Upstream currently hard-codes MistralMLAAttention in the AMD decoder
        # layer and provides no class hook. Replace only that module-global
        # constructor for the duration of this model build; every created
        # layer is already the Ascend class when its inner attention backend
        # is selected, so construction cannot fail before the adapter runs.
        from vllm.models.mistral.mistral_large4.amd import model as upstream_model

        original_attention_cls = upstream_model.MistralMLAAttention
        upstream_model.MistralMLAAttention = AscendMistralMLAAttention
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            upstream_model.MistralMLAAttention = original_attention_cls
        # Upstream fails fast when an unsupported deployment asks for the
        # specialized decode; on Ascend it is never built, since its runtime is
        # HIP IPC and peer memory.
        self.model.specialized_decode_config = None
        self.model.specialized_decode = None

        shared = adopt_ascend_modules(self)
        expected = sum(spec.shares_kv for spec in self.model.layer_specs)
        if shared != expected:
            raise ValueError(
                f"Installed the KV-sharing write skip on {shared} layers, but "
                f"the config declares {expected}. Refusing to run: a missed "
                f"layer would overwrite the latent cache it borrows."
            )

    def get_model_state_cls(self):
        # Upstream returns MistralLarge4ModelState solely to construct the
        # HIP-only specialized decode after weight loading. Ascend disables
        # that path above and must retain its own MRV2 attention-metadata
        # builder for graph capture and execution.
        from vllm_ascend.worker.v2.model_states.default import AscendModelState

        return AscendModelState

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        # Mistral owns a shared MistralRotaryEmbedding rather than one of the
        # classes replaced by vLLM-Ascend's RoPE registry. Publish its final
        # LeRoPE-adjusted cache only after weight loading has rebuilt it;
        # attention metadata is constructed before the first model forward.
        rotary_emb = self.model.rotary_emb
        _record_cos_sin_cache(rotary_emb.cos_sin_cache)
        _record_cos_and_sin_cache_interleaved(rotary_emb, rotary_emb.cos_sin_cache)
        return loaded


class AscendMistralLarge4DSparkForCausalLM(MistralLarge4DSparkForCausalLM):
    """Build Mistral Large 4's dense MLA DSpark draft with Ascend attention.

    The upstream draft imports the same AMD decoder layer as the target, but
    constructs it outside :class:`AscendMistralLarge4ForCausalLM`. Without the
    scoped constructor replacement below it falls back to the generic upstream
    MLA wrapper, which does not pass the Ascend backend's required MLA module
    contract and fails while looking up ``rotary_emb``.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        from vllm.models.mistral.mistral_large4.amd import model as upstream_model

        original_attention_cls = upstream_model.MistralMLAAttention
        upstream_model.MistralMLAAttention = AscendMistralMLAAttention
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            upstream_model.MistralMLAAttention = original_attention_cls

        # Draft attention must rotate with the same plain YaRN table as its
        # context KV writer. The target publishes a separate LeRoPE table.
        for layer in self.model.layers:
            layer.self_attn.attn.impl.rope_metadata_source = self.model.rotary_emb

    @staticmethod
    def _normalize_draft_kv_cache(
        kv_cache: Any,
        kv_lora_rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the NoPE/RoPE component views used by Ascend MLA writers."""
        while isinstance(kv_cache, (list, tuple)) and len(kv_cache) == 1:
            kv_cache = kv_cache[0]
        if isinstance(kv_cache, torch.Tensor):
            if kv_cache.shape[-1] <= kv_lora_rank:
                raise RuntimeError(
                    "Mistral DSpark fused MLA cache is missing its RoPE component: "
                    f"shape={tuple(kv_cache.shape)}, kv_lora_rank={kv_lora_rank}."
                )
            return (
                kv_cache[..., :kv_lora_rank],
                kv_cache[..., kv_lora_rank:],
            )
        if isinstance(kv_cache, (list, tuple)) and len(kv_cache) == 2:
            nope_cache, rope_cache = kv_cache
            if isinstance(nope_cache, torch.Tensor) and isinstance(rope_cache, torch.Tensor):
                return nope_cache, rope_cache
        raise RuntimeError(
            "Mistral DSpark expected a fused MLA cache or a two-component "
            f"NoPE/RoPE cache, got {type(kv_cache).__name__}."
        )

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: (
            torch.Tensor | list[torch.Tensor | None] | tuple[torch.Tensor | None, ...] | None
        ) = None,
    ) -> None:
        """Write draft context KV with Ascend's component-cache operators.

        Upstream calls ``MLAAttentionImpl.do_kv_cache_update``, whose CUDA
        contract is one fused cache tensor. Model Runner V2 binds Ascend MLA
        as separate NoPE/RoPE cache views, so that helper cannot be used here.
        """
        if context_states.numel() == 0 or context_slot_mapping is None:
            return
        if context_positions.numel() != context_states.shape[0]:
            raise ValueError("Mistral DSpark context positions must match context rows.")
        if isinstance(context_slot_mapping, (list, tuple)) and len(context_slot_mapping) != len(self.model.layers):
            raise ValueError("context_slot_mapping must contain one entry per Mistral DSpark layer.")

        # Use the draft's plain-YaRN cache. The target has LeRoPE, so the
        # process-global target table is not valid for these dense draft layers.
        rotary = self.model.rotary_emb
        cos, sin = gather_mla_rope(rotary.cos_sin_cache, context_positions)

        # Resolve the one-tensor form into a per-layer sequence up front. A
        # bool flag plus a ternary leaves one branch holding the whole union,
        # so the tensor calls below have no type to resolve against.
        per_layer_slots: Sequence[torch.Tensor | None]
        if isinstance(context_slot_mapping, (list, tuple)):
            per_layer_slots = context_slot_mapping
        else:
            per_layer_slots = [context_slot_mapping] * len(self.model.layers)
        for layer_idx, layer in enumerate(self.model.layers):
            slots = per_layer_slots[layer_idx]
            if slots is None or slots.numel() == 0:
                continue
            if slots.numel() != context_states.shape[0]:
                raise ValueError("Mistral DSpark slot mapping must match context rows.")

            attn = layer.self_attn
            mla = attn.attn
            impl = mla.impl
            kv_cache = self._normalize_draft_kv_cache(mla.kv_cache, attn.kv_lora_rank)
            if impl._cache_requires_prolog_v3_writer(kv_cache):
                impl.exec_context_kv_prolog_v3(
                    context_states,
                    cos,
                    sin,
                    kv_cache,
                    slots,
                )
                continue

            kv_no_split = attn.kv_a_proj_with_mqa(context_states)[0]
            impl.exec_kv_prefill(
                kv_no_split,
                cos,
                sin,
                kv_cache,
                slots,
            )
