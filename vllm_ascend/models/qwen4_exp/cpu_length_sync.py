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
    if config is None:
        return False
    try:
        # The inventories and these worker-only dependencies exist only after
        # model/KV initialization. Keep CPU-only module imports lightweight.
        # main renamed the QSA side-cache module and its builder/backend; the
        # release tree still ships qsa_cache.py.
        import vllm.models.qwen4_exp.common.qsa_indexer as qsa_indexer
        from vllm.models.qwen4_exp.amd.qsa import Qwen4ExpQSAMetadataBuilder

        QSAIndexerMetadataBuilder = qsa_indexer.QSAIndexerMetadataBuilder
        QSAIndexerBackend = qsa_indexer.QSAIndexerBackend
        from vllm.v1.attention.backends.short_conv_attn import (
            PleShortConvAttentionBackend,
            PleShortConvAttentionMetadataBuilder,
        )

        from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionMetadataBuilder
        from vllm_ascend.attention.gdn_attn_builder import (
            AscendGDNAttentionBackend,
            AscendGDNAttentionMetadataBuilder,
            AscendGDNFusedAttentionBackend,
            AscendGDNHostMetadataBackend,
            AscendGDNHostMetadataBuilder,
        )
        from vllm_ascend.models.qwen4_exp import qsa as ascend_qsa
        from vllm_ascend.models.qwen4_exp.model_state import AscendQwen4ExpModelState
        from vllm_ascend.models.qwen4_exp.qsa import (
            AscendQSABackend,
            AscendQSAImpl,
            AscendQSAMetadataBuilder,
        )
        from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import AscendDFlash2Speculator
        from vllm_ascend.worker.v2.spec_decode.mtp.speculator import AscendMTPSpeculator

        state = runner.model_state
        is_mtp = type(runner.speculator) is AscendMTPSpeculator
        is_dflash2 = type(runner.speculator) is AscendDFlash2Speculator
        if (
            type(state) is not AscendQwen4ExpModelState
            or not (is_mtp or is_dflash2)
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

        qsa_state_builders: tuple[type, ...] = (QSAIndexerMetadataBuilder,)
        ascend_state_builder = getattr(ascend_qsa, "AscendQSAStateMetadataBuilder", None)
        if isinstance(ascend_state_builder, type) and issubclass(ascend_state_builder, QSAIndexerMetadataBuilder):
            qsa_state_builders += (ascend_state_builder,)
        allowed_builders = {
            AscendAttentionBackend: (AscendAttentionMetadataBuilder,),
            AscendQSABackend: (
                Qwen4ExpQSAMetadataBuilder,
                AscendQSAMetadataBuilder,
            ),
            QSAIndexerBackend: qsa_state_builders,
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
        target_supported = (
            bool(target_pairs)
            and all(builder in allowed_builders.get(backend, ()) for backend, builder in target_pairs)
            and any(backend is AscendQSABackend for backend, _ in target_pairs)
            and AscendQSABackend.get_impl_cls() is AscendQSAImpl
        )
        if not target_supported or not draft_pairs:
            return False
        if is_dflash2:
            # DFlash2's dense draft uses the CPU per-request upper bound in
            # _select_seq_lens; its target uses the same Qwen QSA/GDN path.
            return all(
                backend is AscendAttentionBackend and builder is AscendAttentionMetadataBuilder
                for backend, builder in draft_pairs
            )
        return (
            all(builder in allowed_builders.get(backend, ()) for backend, builder in draft_pairs)
            and any(backend is AscendQSABackend for backend, _ in draft_pairs)
            and runner.speculator.attn_architecture == "QSA"
            and runner.speculator.attn_backend is AscendQSABackend
        )
    except Exception:
        # Optional capability inspection must not make an otherwise supported
        # service fail to initialize; unknown consumers keep exact CPU counts.
        return False
