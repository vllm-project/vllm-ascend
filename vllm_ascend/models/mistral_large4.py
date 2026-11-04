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
* **The KV-sharing write.** See :class:`AscendSharedKVSlotMappingMixin`.

Everything else — MLA, the DSA indexer, the shared-indexer schedule, NoPE,
YaRN, the MoE block and the FP8 block-scaled checkpoint — is carried by paths
vLLM-Ascend already ships. Backend selection needs no patch: a config with
``index_topk`` and no ``compress_ratios`` resolves to ``AscendSFABackend``.
"""

from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_ep_group
from vllm.logger import init_logger
from vllm.models.mistral.amd.attention import MistralMLAAttention
from vllm.models.mistral.mistral_large4.amd.model import (
    MistralLarge4ForCausalLM,
)

from vllm_ascend.attention.indexer import AscendSFAIndexerBackend

logger = init_logger(__name__)

__all__ = [
    "AscendMistralIndexerBackend",
    "AscendMistralLarge4ForCausalLM",
    "AscendMistralMLAAttention",
    "AscendSharedKVSlotMappingMixin",
    "adopt_ascend_modules",
    "prepare_indexer_for_ascend",
    "validate_moe_parallelism",
]


class AscendSharedKVSlotMappingMixin:
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
    """

    def _get_sfa_kv_slot_mapping(self, attn_metadata: Any) -> torch.Tensor:
        # A zero-row mapping, not a sentinel value: the Ascend scatter writes
        # at the indices it is given, so -1 would land on the last slot.
        return attn_metadata.slot_mapping[:0]


def _shared_kv_impl_class(base: type) -> type:
    """Return ``base`` with the shared-KV mixin ahead of it in the MRO."""
    name = f"AscendSharedKV{base.__name__}"
    cached = getattr(base, "_ascend_shared_kv_cls", None)
    if cached is not None and cached.__name__ == name:
        return cached
    cls = type(name, (AscendSharedKVSlotMappingMixin, base), {})
    base._ascend_shared_kv_cls = cls  # type: ignore[attr-defined]
    return cls


def adopt_shared_kv_skip(attn: nn.Module) -> None:
    """Install the shared-KV write skip on one attention module's impl.

    The impl is built by the attention layer's constructor, so the mixin is
    installed afterwards by reassigning ``__class__``. The mixin adds no state,
    so parameters and buffers are untouched.
    """
    impl = getattr(getattr(attn, "attn", None), "impl", None)
    if impl is None:
        raise ValueError(
            "KV-sharing layer has no attention impl to install the write skip on; "
            "refusing to run, because the layer would otherwise overwrite the "
            "latent cache it borrows."
        )
    if isinstance(impl, AscendSharedKVSlotMappingMixin):
        return
    impl.__class__ = _shared_kv_impl_class(type(impl))


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

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        log_positions: torch.Tensor,
        rotary_emb: Any,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        # The Ascend SFA impl runs the indexer itself, from inside self.attn,
        # and its signature carries no log positions. This is the last place
        # that holds them, so hand them to the backend here.
        backend = getattr(getattr(self.attn.impl, "indexer", None), "impl", None)
        if isinstance(backend, AscendMistralIndexerBackend):
            backend.log_positions = log_positions
        q, kv_c_normed, k_pe = self._project_qkv(positions, hidden_states, log_positions, rotary_emb)
        if self.spec.use_sparse:
            # Upstream's MLA impls use this to order the top-k read against the
            # indexer's stream. Serial execution already guarantees that, and
            # the Ascend SFA impl does not define it.
            record = getattr(self.attn.impl, "record_logical_topk_ready", None)
            if record is not None:
                record()

        attn_out = self.attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(num_tokens, self.num_local_heads * self.v_head_dim),
        )
        return self.o_proj(attn_out)[0]


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

    def __init__(self, vllm_indexer: nn.Module, qk_rope_head_dim: int) -> None:
        super().__init__(vllm_indexer, qk_rope_head_dim)
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


class AscendMistralLarge4ForCausalLM(MistralLarge4ForCausalLM):
    """Mistral Large 4 with the Ascend paths adopted after construction.

    Upstream builds the attention modules inside its own constructors and
    offers no hook for the attention class, so they are switched afterwards.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        _reject_unsupported(vllm_config)
        validate_moe_parallelism(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
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
