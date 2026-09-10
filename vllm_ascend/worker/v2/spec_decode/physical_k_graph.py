# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Width-specific draft graph descriptors and capture-time physical-K scopes.

The target remains PIECEWISE. Draft FULL graphs only replay matching widths;
uncaptured widths retain the upstream eager fallback.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from functools import wraps
from typing import Any

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.worker.v2.spec_decode.physical_k import (
    configured_capture_k,
    physical_k_scope,
    query_width,
)


def _sample_from_anchor(self) -> bool:
    speculative_config = self.vllm_config.speculative_config
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    hf_config = getattr(draft_model_config, "hf_config", None)
    if getattr(speculative_config, "use_dspark", lambda: False)():
        return bool(getattr(hf_config, "sample_from_anchor", True))
    return False


def extend_capture_descriptors(self) -> None:
    """Capture one FULL descriptor for each configured physical K.

    The upstream manager only expands capture widths for its native
    dynamic-spec configuration.  Ascend's hardware-aware policy is an
    independent scheduler path, so add the same descriptor matrix here.
    If a width is not captured, normal dispatch falls back to eager mode;
    it never reuses a graph with a different query width.
    """

    decode_mode = self.cudagraph_mode.decode_mode()
    if decode_mode == CUDAGraphMode.NONE:
        return
    capture_sizes = sorted(self.compilation_config.cudagraph_capture_sizes or [])
    if not capture_sizes:
        return

    speculative_config = self.vllm_config.speculative_config
    max_k = int(getattr(speculative_config, "num_speculative_tokens", 0))
    if max_k <= 0:
        return
    sample_from_anchor = _sample_from_anchor(self)
    max_capture_size = self.compilation_config.max_cudagraph_capture_size or (1 << 60)
    max_decode_tokens = self.max_num_reqs * self.decode_query_len

    capture_descs = self._capture_descs.setdefault(decode_mode, [])
    for raw_tokens in capture_sizes:
        for draft_k in configured_capture_k(self.vllm_config, max_k):
            width = query_width(sample_from_anchor, draft_k)
            rounded_tokens = ((raw_tokens + width - 1) // width) * width
            num_reqs = rounded_tokens // width
            if rounded_tokens > max_decode_tokens or rounded_tokens > max_capture_size or num_reqs > self.max_num_reqs:
                continue
            for num_active_loras in self.lora_capture_cases:
                desc = BatchExecutionDescriptor(
                    cg_mode=decode_mode,
                    num_tokens=rounded_tokens,
                    num_reqs=num_reqs,
                    uniform_token_count=width,
                    num_active_loras=num_active_loras,
                )
                if desc not in capture_descs:
                    capture_descs.append(desc)
                self._candidates.setdefault((rounded_tokens, num_active_loras), []).append(desc)
                for token_count in range(0, rounded_tokens + 1):
                    self._candidates.setdefault((token_count, num_active_loras), []).append(desc)

    capture_descs.sort(key=lambda item: item.num_tokens, reverse=True)

    # Ascend's FIA graph-parameter table is keyed by ``num_tokens`` only.
    # Keeping two FULL graphs with the same token count but different
    # physical K would make them share one workspace/event/parameter bucket
    # (for example, N=16 with K=1, 2 and 4), which can feed a workspace
    # captured for one query width to another kernel.  That combination
    # fails asynchronously in the FIA MTE with an invalid DDR address.
    # Keep the widest graph for each token count; narrower K values remain
    # valid dynamic decisions and dispatch eagerly when no width-matching
    # graph exists.
    widest_by_tokens: dict[int, BatchExecutionDescriptor] = {}
    for desc in capture_descs:
        current = widest_by_tokens.get(desc.num_tokens)
        if current is None or (desc.uniform_token_count or 0) > (current.uniform_token_count or 0):
            widest_by_tokens[desc.num_tokens] = desc
    kept_descs = set(widest_by_tokens.values())
    capture_descs[:] = sorted(
        kept_descs,
        key=lambda item: item.num_tokens,
        reverse=True,
    )
    for candidates in self._candidates.values():
        candidates[:] = list(
            dict.fromkeys(desc for desc in candidates if desc.cg_mode != decode_mode or desc in kept_descs)
        )
    for key, candidates in self._candidates.items():
        unique = list(dict.fromkeys(candidates))
        candidates[:] = unique


@contextmanager
def physical_k_capture_scope(self, forward_fn: Callable):
    """Keep metadata preparation and forward on the same K, restoring on error."""
    # The upstream DFlash graph manager builds ``attn_state`` before it
    # invokes the supplied forward function.  For a variable-width graph,
    # that preparation must observe the same physical K as the forward
    # itself: the Ascend metadata builder derives TND
    # ``actual_seq_lengths_q`` and query padding from the prepared input
    # batch.  Temporarily wrap the upstream helper so capture-time
    # metadata and the model forward use one width.  This also avoids
    # changing the upstream vLLM checkout or the shared attention backend.
    dflash_cudagraph_module = None
    original_prepare_inputs = None
    if self._v2_varlen_physical_k:
        import vllm.v1.worker.gpu.spec_decode.dflash.cudagraph as dflash_cudagraph_module

        original_prepare_inputs = dflash_cudagraph_module._prepare_dflash_inputs_to_capture

        def prepare_inputs_with_runtime_width(
            num_reqs: int,
            num_tokens: int,
            input_buffers: InputBuffers,
            block_tables: BlockTables,
            attn_groups: list[list[AttentionGroup]],
            kv_cache_config: KVCacheConfig,
            max_model_len: int,
            skip_attn: bool,
            causal: bool | Mapping[int, bool],
        ):
            width = num_tokens // num_reqs
            draft_k = width if _sample_from_anchor(self) else width - 1
            with physical_k_scope(self.speculator, draft_k=draft_k):
                attn_state = original_prepare_inputs(
                    num_reqs,
                    num_tokens,
                    input_buffers,
                    block_tables,
                    attn_groups,
                    kv_cache_config,
                    max_model_len,
                    skip_attn,
                    causal,
                )

            # ``make_dummy`` normally produces this sequence, but an
            # Ascend graph capture may add padding requests.  Make the
            # exact descriptor contract explicit for FIA's TND layout:
            # the final cumulative query length must equal num_tokens.
            if attn_state.attn_metadata is not None:
                query_lens = [(idx + 1) * width for idx in range(num_reqs)]
                for metadata in attn_state.attn_metadata.values():
                    metadata.actual_seq_lengths_q = query_lens
            return attn_state

        dflash_cudagraph_module._prepare_dflash_inputs_to_capture = prepare_inputs_with_runtime_width

    def forward_with_runtime_width(
        num_reqs: int,
        num_tokens: int,
        attn_metadata: Any,
        slot_mappings: Any,
        num_tokens_across_dp: Any,
        cg_mode: CUDAGraphMode,
    ):
        if not self._v2_varlen_physical_k or num_reqs <= 0:
            return forward_fn(
                num_reqs,
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cg_mode,
            )
        width = num_tokens // num_reqs
        draft_k = width if _sample_from_anchor(self) else width - 1
        with physical_k_scope(self.speculator, draft_k=draft_k):
            return forward_fn(
                num_reqs,
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cg_mode,
            )

    try:
        yield forward_with_runtime_width
    finally:
        if dflash_cudagraph_module is not None:
            dflash_cudagraph_module._prepare_dflash_inputs_to_capture = original_prepare_inputs


def enable_draft_graph_debug(manager, logger) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    original = manager.dispatch
    logger.debug("ASCEND_DRAFT_CAPTURES descriptors=%s", manager._capture_descs)

    @wraps(original)
    def traced(*args, **kwargs):
        result = original(*args, **kwargs)
        # Only descriptor metadata is logged, never device tensor contents.
        logger.debug(
            "ASCEND_DRAFT_DISPATCH mode=%s tokens=%s batch=%s width=%s",
            getattr(result, "cg_mode", None),
            getattr(result, "num_tokens", None),
            getattr(result, "num_reqs", None),
            getattr(result, "uniform_token_count", None),
        )
        return result

    manager.dispatch = traced
