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

logger = init_logger(__name__)

__all__ = [
    "AscendMistralLarge4ForCausalLM",
    "AscendMistralMLAAttention",
    "AscendSharedKVSlotMappingMixin",
    "adopt_ascend_modules",
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
    """Upstream attention with the CUDA stream overlap removed.

    Upstream runs the indexer on an aux stream so it overlaps the Q/KV
    projections, joining on two ``torch.cuda.Event``s. vLLM-Ascend uses
    ``torch.npu.Event`` and ``npu_stream_switch``; upstream's helper
    (``vllm.utils.multi_stream_utils.execute_in_parallel``) has no Ascend
    counterpart. Running the two serially keeps the same math.
    """

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        log_positions: torch.Tensor,
        rotary_emb: Any,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        indexer = self.indexer
        if indexer is not None:
            indexer(
                hidden_states,
                positions,
                log_positions,
                rotary_emb if self.spec.indexer_use_rope else None,
            )
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
