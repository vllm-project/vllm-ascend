# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Identify Qwen consumers that accept optimistic CPU sequence lengths.

This is an initialization-time capability, not per-batch queue admission.
The runner skips both the exact-count D2H producer and its synchronized
consumer; rejection-corrected positions and execution lengths stay on device.
"""

from typing import Any


def can_skip_qwen_cpu_length_sync(runner: Any) -> bool:
    """Return False for consumers that still require exact CPU lengths."""
    config = getattr(runner, "ascend_config", None)
    if config is None or getattr(config, "enable_decode_early_queue", False) is not True:
        return False
    try:
        # The inventories and these worker-only dependencies exist only after
        # model/KV initialization. Keep CPU-only module imports lightweight.
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

        state = runner.model_state
        if (
            type(state) is not AscendQwen4ExpModelState
            or type(runner.speculator) is not AscendMTPSpeculator
            or runner.model_config.architecture not in ("Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration")
        ):
            return False

        # CP partitions CPU length metadata into local cache rows. Sparse KV
        # offload uses exact computed-token boundaries to restore cache tails.
        parallel = runner.parallel_config
        if (
            parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or config.sparse_kv_offload_config.enabled
        ):
            return False

        if runner.supports_mm_inputs:
            # Ordinary media embeddings live only in the prompt. Their existing
            # gather path skips completed prompts using the CPU upper bound;
            # real-time/interleaved media can still need an exact token window.
            encoder = getattr(state, "encoder_runner", None)
            if encoder is None or encoder.is_realtime is not False:
                return False

        qsa_state_builders: tuple[type, ...] = (QSAMetadataBuilder,)
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
        draft_pairs = tuple(
            (group.backend, type(group.get_metadata_builder(0)))
            for cache_groups in runner.speculator.attn_groups
            for group in cache_groups
        )
        return (
            bool(target_pairs)
            and bool(draft_pairs)
            and all(builder in allowed_builders.get(backend, ()) for backend, builder in (*target_pairs, *draft_pairs))
            and any(backend is AscendQSABackend for backend, _ in target_pairs)
            and any(backend is AscendQSABackend for backend, _ in draft_pairs)
            and runner.speculator.attn_architecture == "QSA"
            and runner.speculator.attn_backend is AscendQSABackend
            and AscendQSABackend.get_impl_cls() is AscendQSAImpl
        )
    except Exception:
        # Optional capability inspection must not make an otherwise supported
        # service fail to initialize; unknown consumers keep exact CPU counts.
        return False
