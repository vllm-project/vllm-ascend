# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
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
# This file is a part of the vllm-ascend project.
#
# AscendMTPSpeculator is a sibling of AscendEagleSpeculator: both mix in
# AscendSpecDecodeMixin for the shared flat NPU draft loop and layer it on their
# own upstream base (MTPSpeculator vs EagleSpeculator). MTP's draft uses MLA
# attention, so it overrides only the MLA-specific pieces; the flat loop is
# inherited unchanged. Nothing MLA-specific lives in the mixin or Eagle.
from contextlib import contextmanager
from copy import copy
from typing import Any

import torch
import vllm.v1.worker.gpu.spec_decode.speculator as _upstream_speculator
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendAutoRegressiveSpeculator


@contextmanager
def build_wrapper(positions, pad):
    """Temporarily wrap build_attn_metadata to forward MLA rotary positions.

    The flat draft-metadata path (upstream ``_build_draft_attn_metadata``) calls
    ``build_attn_metadata`` without ``positions``, but MLA reads positions
    *inside* ``build_decode_metadata`` for rotary cos/sin. This caches the
    original ``build_attn_metadata``, replaces it for the duration of the block
    with one that forwards ``positions[:pad]``, and restores it on exit -- so
    MTP can reuse the flat ``super()._build_draft_attn_metadata()`` path instead
    of duplicating the ``build_attn_metadata`` call and all its arguments.

    Must run inside ``build_attn_metadata_wrapper()``, which has already
    replaced the module-level ``build_attn_metadata`` with the Ascend builder.
    """
    raw = _upstream_speculator.build_attn_metadata  # cache

    def build_attn_metadata(*args, **kwargs):
        kwargs["positions"] = positions[:pad]
        return raw(*args, **kwargs)

    try:
        _upstream_speculator.build_attn_metadata = build_attn_metadata
        yield
    finally:
        _upstream_speculator.build_attn_metadata = raw  # restore


class AscendMTPSpeculator(AscendAutoRegressiveSpeculator, MTPSpeculator):
    """Ascend MTP speculator (MLA draft).

    Sibling of AscendEagleSpeculator: inherits the shared flat NPU draft loop
    from AscendSpecDecodeMixin (layered on upstream MTPSpeculator) and overrides
    only the MLA-specific pieces. MLA differs from flat in three ways: rotary
    ``positions`` must reach ``build_attn_metadata``, the decode aclgraph pads
    by ``num_tokens`` (FIA TND) instead of ``num_reqs``, and MLA metadata
    carries a ``.decode`` sub-object needing per-step upkeep. Flat (non-MLA)
    MTP models fall back to the inherited flat loop via the
    ``if not self.draft_uses_mla`` guards. ``draft_uses_mla`` is read from
    ``model_config.is_deepseek_mla`` in ``__init__`` (config-level, available
    upfront -- same check as v1's llm_base_proposer); the MLA block-table
    buffer is pre-allocated in ``set_attn``. No lazy init, no getattr.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self.draft_uses_mla = self.vllm_config.model_config.is_deepseek_mla

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
    ) -> None:
        """Ascend setup: build attn_backends (via super), then pre-allocate the
        stable MLA block-table buffer (max-sized, so the captured decode graph
        can reference it by identity). ``draft_uses_mla`` is set in __init__."""
        super().set_attn(model_state, kv_cache_config, block_tables)
        if self.draft_uses_mla:
            # MLA draft has a single KV cache group; allocate a stable zero
            # buffer matching its block table for the captured decode graph.
            block_table = block_tables.input_block_tables[0]
            self._draft_block_table_buf = torch.zeros_like(block_table)

    def _build_draft_attn_metadata(
        self,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens_padded: int,
        num_query_per_req: int = 1,
        causal: bool = True,
    ) -> dict[str, Any] | None:
        # Flat MTP reuses the flat super() build (positions are not needed).
        if not self.draft_uses_mla:
            return super()._build_draft_attn_metadata(
                num_reqs, num_reqs_padded, num_tokens_padded, num_query_per_req, causal
            )
        # MLA: rotary positions must reach build_attn_metadata, but the flat
        # super() path does not forward them. Wrap build_attn_metadata to inject
        # positions[:num_tokens_padded] and reuse super() (no arg duplication).
        with build_wrapper(self.input_buffers.positions, num_tokens_padded):
            return super()._build_draft_attn_metadata(
                num_reqs, num_reqs_padded, num_tokens_padded, num_query_per_req, causal
            )

    def get_draft_decode_num_reqs_padded(self, desc: BatchExecutionDescriptor) -> int:
        # MLA decode aclgraph pads by num_tokens (FIA TND); flat pads by num_reqs.
        return desc.num_tokens if self.draft_uses_mla else desc.num_reqs

    def _init_decode_draft_attn_metadatas(self, attn_metadata, num_reqs_padded):
        # MLA: if the live base is a stale prefill residual (.decode=None),
        # rebuild it fresh BEFORE super() copies it per step.
        if self.draft_uses_mla:
            attn_metadata = self._get_decode_base_metadata(attn_metadata, num_reqs_padded)
        # MLA: snapshot each step's .decode block_table into the stable buffer
        # after super builds per-step copies. Flat uses the base as-is.
        draft_attn_metadatas = super()._init_decode_draft_attn_metadatas(attn_metadata, num_reqs_padded)
        if self.draft_uses_mla:
            for per_step_attn_metadata in draft_attn_metadatas:
                for metadata in per_step_attn_metadata.values():
                    self._prepare_step_decode_fields(metadata, num_reqs_padded)
        return draft_attn_metadatas

    def _update_decode_attn_metadata(self, attn_metadata, step, num_reqs=None):
        # Flat: super updates the base fields. MLA: also mirror them onto .decode.
        super()._update_decode_attn_metadata(attn_metadata, step, num_reqs)
        if self.draft_uses_mla:
            for metadata in attn_metadata.values():
                self._write_decode_step_fields(metadata, metadata.actual_seq_lengths_q, metadata.seq_lens_list)

    def _get_decode_base_metadata(self, attn_metadata, num_reqs_padded):
        # Rebuild fresh if live metadata is a stale prefill residual (.decode=None).
        need_fresh = any(m.decode is None for m in attn_metadata.values())
        if need_fresh:
            assert self.input_batch is not None
            return self._build_draft_attn_metadata(
                num_reqs=self.input_batch.num_reqs,
                num_reqs_padded=num_reqs_padded,
                num_tokens_padded=num_reqs_padded,
            )
        return attn_metadata

    def _prepare_step_decode_fields(self, metadata, num_reqs_padded):
        # Snapshot the MLA .decode block_table into the stable pre-allocated
        # buffer (the captured decode graph references it by identity; async
        # add_requests mutates input_block_tables in place between steps).
        if metadata.num_decodes > 0:
            metadata.decode = copy(metadata.decode)
            block_table = self.block_tables.input_block_tables[0]
            buf = self._draft_block_table_buf
            buf[:num_reqs_padded].copy_(block_table[:num_reqs_padded])
            metadata.decode.block_table = buf[:num_reqs_padded]

    def _write_decode_step_fields(self, metadata, query_lens_list, seq_lens_list):
        # Mirror per-step seq-len fields onto the MLA .decode sub-metadata.
        if metadata.num_decodes > 0:
            metadata.decode.actual_seq_lengths_q = query_lens_list
            metadata.decode.seq_lens_list = seq_lens_list
