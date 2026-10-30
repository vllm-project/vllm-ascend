# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen-specific CPU-consumer proof for deferred decode length waits.

The shared queue controller owns scheduling and snapshot readiness. This
capability only identifies the model implementations and attention consumers
whose CPU metadata can use a conservative length bound. Unknown consumers
retain the normal synchronized path.
"""

from typing import Any

from vllm_ascend.models.qwen4_exp.mm_decode_queue import ObjectIdentity


class QwenDecodeQueueCapability:
    """Stateless proof supplied by the Qwen model-state implementation."""

    __slots__ = ()

    def supports(self, runner: Any) -> bool:
        """Inspect both target and draft consumers without caching old proofs."""
        try:
            # These dependencies belong to the initialized model, rather than
            # every shared runner (or CPU-only import of this module).
            from vllm.models.qwen4_exp.amd.qsa import Qwen4ExpQSAMetadataBuilder
            from vllm.models.qwen4_exp.common.qsa_cache import QSAMetadataBuilder, QSAStateBackend
            from vllm.v1.attention.backends.short_conv_attn import (
                PleShortConvAttentionBackend,
                PleShortConvAttentionMetadataBuilder,
            )

            from vllm_ascend.models.qwen4_exp import qsa as ascend_qsa
            from vllm_ascend.models.qwen4_exp.model_state import AscendQwen4ExpModelState
            from vllm_ascend.models.qwen4_exp.qsa import AscendQSABackend, AscendQSAImpl
            from vllm_ascend.ops.gdn_attn_builder import (
                AscendGDNAttentionBackend,
                AscendGDNAttentionMetadataBuilder,
                AscendGDNFusedAttentionBackend,
                AscendGDNHostMetadataBackend,
                AscendGDNHostMetadataBuilder,
            )
            from vllm_ascend.worker.v2.spec_decode.mtp.speculator import AscendMTPSpeculator

            if (
                type(runner.model_state) is not AscendQwen4ExpModelState
                or type(runner.speculator) is not AscendMTPSpeculator
                or runner.model_config.architecture not in ("Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration")
            ):
                return False

            qsa_state_builders: tuple[type, ...] = (QSAMetadataBuilder,)
            # A5's optional exported builder delegates MRV2 state-cache work
            # to the same producer. Admit the exact class, not its subclasses.
            ascend_state_builder = getattr(ascend_qsa, "AscendQSAStateMetadataBuilder", None)
            if isinstance(ascend_state_builder, type) and issubclass(ascend_state_builder, QSAMetadataBuilder):
                qsa_state_builders += (ascend_state_builder,)
            allowed_builders = {
                AscendQSABackend: (Qwen4ExpQSAMetadataBuilder,),
                QSAStateBackend: qsa_state_builders,
                AscendGDNAttentionBackend: (AscendGDNAttentionMetadataBuilder,),
                AscendGDNHostMetadataBackend: (AscendGDNHostMetadataBuilder,),
                AscendGDNFusedAttentionBackend: (AscendGDNAttentionMetadataBuilder, AscendGDNHostMetadataBuilder),
                PleShortConvAttentionBackend: (PleShortConvAttentionMetadataBuilder,),
            }
            target_pairs = tuple(
                (group.backend, type(group.get_metadata_builder(0)))
                for cache_groups in runner.attn_groups
                for group in cache_groups
            )
            # The speculator owns a separate inventory built with its draft
            # config. A supported target never establishes draft eligibility.
            draft_pairs = tuple(
                (group.backend, type(group.get_metadata_builder(0)))
                for cache_groups in runner.speculator.attn_groups
                for group in cache_groups
            )
            return (
                bool(target_pairs)
                and bool(draft_pairs)
                and all(
                    builder in allowed_builders.get(backend, ()) for backend, builder in (*target_pairs, *draft_pairs)
                )
                and any(backend is AscendQSABackend for backend, _ in target_pairs)
                and any(backend is AscendQSABackend for backend, _ in draft_pairs)
                and runner.speculator.attn_architecture == "QSA"
                and runner.speculator.attn_backend is AscendQSABackend
                and AscendQSABackend.get_impl_cls() is AscendQSAImpl
            )
        except (AttributeError, TypeError, IndexError, AssertionError, NotImplementedError, ImportError):
            return False

    def mm_context(self, runner: Any) -> tuple | None:
        """Bind the audited media consumers and their caches to a certificate."""
        try:
            from vllm_ascend.models.qwen4_exp.model_state import AscendQwen4ExpModelState

            state = runner.model_state
            encoder_runner = getattr(state, "encoder_runner", None)
            cache = runner.encoder_cache
            if (
                type(state) is not AscendQwen4ExpModelState
                or not runner.supports_mm_inputs
                or not state.supports_mm_inputs
                or encoder_runner is None
                or cache is None
                or encoder_runner.encoder_cache is not cache
                or state.encoder_cache is not cache
                or encoder_runner.is_realtime is not False
                or runner.model_config.enable_prompt_embeds
                or runner.model_config.is_mm_prefix_lm
                or runner.model_config.is_encoder_decoder
                or getattr(state, "mm_pruner", None) is not None
                or getattr(state, "prompt_embeds_state", None) is not None
                or getattr(runner.model, "deepstack_input_embeds_num_tokens", 0) != 0
            ):
                return None
            return (
                ObjectIdentity(runner.req_states),
                ObjectIdentity(state),
                ObjectIdentity(runner.model),
                ObjectIdentity(cache),
                ObjectIdentity(encoder_runner),
                ObjectIdentity(getattr(state, "rope_state", None)),
                ObjectIdentity(runner.model_config),
                ObjectIdentity(runner.model_config.hf_text_config),
                type(state).prepare_inputs_embeds,
                type(state).gather_mm_embeddings,
                type(encoder_runner).gather_mm_embeddings,
            )
        except (AttributeError, TypeError, ImportError):
            return None
