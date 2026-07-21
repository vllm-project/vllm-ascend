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
# AscendMTPSpeculator: standalone Ascend MTP speculator. All Ascend overrides
# (init_cudagraph_manager, propose, set_attn, capture, _run_model,
# _generate_draft, _multi_step_decode, _build_draft_attn_metadata,
# build_draft_attn_metadatas, ...) live here. The MLA ``.decode`` hooks handle
# MLA draft models; flat MTP auto-adapts via ``_draft_uses_mla``.
import logging
from contextlib import contextmanager
from copy import copy
from typing import Any, cast

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import AttentionStatePair, BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata, build_attn_metadata_wrapper
from vllm_ascend.worker.v2.input_batch import AscendInputBuffers

logger = logging.getLogger(__name__)


class AscendMTPSpeculator(MTPSpeculator):
    """Ascend MTP speculator.

    Self-contained Ascend overrides for the MTP draft path. Unlike Eagle,
    MTP installs its own MLA-aware aclgraph managers (see
    ``init_cudagraph_manager``) and provides MLA ``.decode`` hooks; flat MTP
    models auto-adapt via ``_draft_uses_mla``.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        """Override MTPSpeculator.__init__ for Ascend NPUs.
        Ascend attention metadata building needs more information, such as
        seq_lens_cpu from input_batch, so we need to override __init__.
        """
        super().__init__(vllm_config, device)

        del self.input_buffers
        # AscendInputBuffers has extra `seq_lens_cpu` attribute.
        # so reinitialize input_buffers here.
        self.input_buffers: AscendInputBuffers = AscendInputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=device,
        )

        # add more attributes for `input_buffers` in graph mode
        cudagraph_mode = self.vllm_config.compilation_config.cudagraph_mode
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            self.input_buffers.draft_seq_lens_cpus = [
                torch.zeros(self.max_num_reqs, dtype=torch.int32, device="cpu")
                for _ in range(self.num_speculative_steps - 1)
            ]

        # we need to update full graph params in run_fullgraph,
        # so create a stream to update full graph params.
        if cudagraph_mode.has_full_cudagraphs():
            self.update_stream: torch.npu.Stream = torch.npu.Stream()

        # when in decode phase of speculator, we need some value in
        # draft model's input_batch. so we keep a reference here.
        self.input_batch: InputBatch | None = None

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        """Install MTP's own MLA-aware aclgraph managers.

        Unlike Eagle (which relies on the globally-patched managers installed
        by ``patch_eagle_speculator`` via ``super().init_cudagraph_manager()``),
        MTP instantiates ``PrefillMTPAclGraphManager`` / ``DecodeMTPAclGraphManager``
        directly. The decode manager is MLA-aware (its ``run_fullgraph`` calls
        ``get_draft_decode_num_reqs_padded``), which the Eagle managers are not.

        We deliberately do not call ``super()``: doing so would first create the
        patched Eagle managers (triggering ``set_draft_graph_params``) only to
        discard them, and we would then trigger it again below.
        """
        from vllm_ascend.worker.v2.spec_decode.mtp.aclgraph import (
            DecodeMTPAclGraphManager,
            PrefillMTPAclGraphManager,
        )

        # Initialize cudagraph manager for draft prefill (draft position 0).
        self.prefill_cudagraph_manager = PrefillMTPAclGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            self.num_speculative_steps + 1,
            speculator=self,
        )

        # PIECEWISE cudagraphs are not supported for draft decodes.
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            decode_cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            decode_cudagraph_mode = CUDAGraphMode.NONE

        # Initialize cudagraph manager for draft decodes (draft positions > 0).
        self.decode_cudagraph_manager = DecodeMTPAclGraphManager(
            self.vllm_config,
            self.device,
            decode_cudagraph_mode,
            decode_query_len=1,
            speculator=self,
        )

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: Any = None,
    ):
        """Override propose for Ascend NPUs, because npu attention metadata
        needs more information, we need to cache input_batch, so we can use it
        later in generate_draft.
        """
        self.input_batch = input_batch
        # wrap build_attn_metadata to use Ascend attention metadata building.
        # so we can call super().propose() directly.
        with build_attn_metadata_wrapper(), torch_gather_wrapper():
            return super().propose(
                input_batch,
                attn_metadata,
                slot_mappings,
                last_hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                num_tokens_across_dp,
                dummy_run,
                skip_attn_for_dummy_run,
                mm_inputs,
                is_profile=is_profile,
            )

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
    ) -> None:
        super().set_attn(model_state, kv_cache_config, block_tables)

        # npu needs attn_backends to update graph params
        attn_backends: dict[str, type[AttentionBackend]] = {}

        active_layer_names = self.draft_attn_layer_names
        for kv_cache_group_id, kv_cache_group_spec in enumerate(kv_cache_config.kv_cache_groups):
            layer_names = kv_cache_group_spec.layer_names
            if active_layer_names is not None:
                layer_names = list(active_layer_names.intersection(layer_names))

            layer_type = cast(type[Any], AttentionLayerBase)
            attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type, layer_names)

            for layer_name in layer_names:
                attn_backend = attn_layers[layer_name].get_attn_backend()
                attn_backends[layer_name] = attn_backend

        self.attn_backends = attn_backends

    def capture(
        self,
        attn_states: dict[BatchExecutionDescriptor, AttentionStatePair],
    ) -> None:
        logger.info("Capturing model for speculator...")
        # Reset indices to zeros to prevent stale values from prior
        # dummy runs to cause out-of-bounds indexing during capture.
        self.last_token_indices.zero_()

        # Capture the prefill routine (model forward + compute_logits +
        # sample).
        # For FULL graphs, the entire routine is recorded as one graph.
        # For PIECEWISE, only the model's compiled regions are captured
        # and the rest (compute_logits, gumbel_sample) runs eagerly.
        assert self.prefill_cudagraph_manager is not None
        if self.prefill_cudagraph_manager.use_breakable_cg:
            self.prefill_cudagraph_manager.init_breakable_cg_runner(self.model)
        self.prefill_cudagraph_manager.capture(
            self._prefill,
            attn_states,
            progress_bar_desc="Capturing prefill CUDA graphs",
        )

        if self.num_speculative_steps == 1:
            return

        # Capture all decode draft generation steps as a single graph.
        assert self.decode_cudagraph_manager is not None
        with build_attn_metadata_wrapper():
            self.decode_cudagraph_manager.capture(
                self._multi_step_decode,
                self.model_state,
                self.input_buffers,
                self.block_tables,
                self.attn_groups,
                self.kv_cache_config,
                progress_bar_desc="Capturing decode CUDA graphs",
            )

    @torch.inference_mode()
    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Override AutoRegressiveSpeculator._run_model for Ascend NPUs."""
        last_hidden_states, hidden_states = super()._run_model(
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
            mm_inputs,
        )
        self._ascend_update_seq_lens(attn_metadata)
        return last_hidden_states, hidden_states

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        """Thin override: delegate to upstream single-step ``_generate_draft``,
        then apply Ascend-specific attention-metadata updates required by the
        FIA operator."""
        super()._generate_draft(
            num_reqs,
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        if attn_metadata is not None:
            self._update_decode_attn_metadata(attn_metadata, 1, num_reqs)

    def _multi_step_decode(
        self,
        num_reqs: int,
        skip_attn: bool,
        batch_desc: BatchExecutionDescriptor,
        num_tokens_across_dp: torch.Tensor | None,
    ) -> None:
        """FULL mode: replay the merged multi-step draft graph once.

        Ascend captures ``_multi_step_decode`` (all speculative steps) as a
        single merged aclgraph, so FULL mode replays it once via
        ``run_fullgraph`` -- unlike upstream which captures per-step
        ``_generate_draft`` graphs and loops ``run_fullgraph`` per step.
        The per-step FIA params are updated afterwards by
        ``update_full_graph_params`` (fed by ``build_draft_attn_metadatas``).
        For PIECEWISE / NONE modes we delegate to upstream (per-step
        ``_generate_draft``).
        """
        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            assert self.decode_cudagraph_manager is not None
            self.decode_cudagraph_manager.run_fullgraph(batch_desc)
            return
        super()._multi_step_decode(num_reqs, skip_attn, batch_desc, num_tokens_across_dp)

    def _build_draft_attn_metadata(
        self,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens_padded: int,
        num_query_per_req: int = 1,
        causal: bool = True,
    ) -> dict[str, Any] | None:
        # Call build_attn_metadata directly to pass positions (MLA needs them
        # for rotary cos/sin; flat ignores).
        query_start_loc_cpu = torch.clamp(self.arange[: num_reqs_padded + 1], max=num_reqs) * num_query_per_req
        block_tables = [x[:num_reqs_padded] for x in self.block_tables.input_block_tables]
        slot_mappings = self.block_tables.slot_mappings[:, :num_tokens_padded]
        attn_metadata = build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=num_reqs_padded,
            num_tokens=num_tokens_padded,
            query_start_loc_gpu=self.input_buffers.query_start_loc[: num_reqs_padded + 1],
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=num_query_per_req,
            seq_lens=self.input_buffers.seq_lens[:num_reqs_padded],
            max_seq_len=self.draft_max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=self.kv_cache_config,
            causal=causal,
            positions=self.input_buffers.positions[:num_tokens_padded],
        )
        if attn_metadata is not None:
            # Ascend-specific: force DecodeOnly attention state for the draft model.
            for metadata in attn_metadata.values():
                metadata.attn_state = AscendAttentionState.DecodeOnly
        return attn_metadata

    def build_draft_attn_metadatas(self, num_reqs_padded, is_draft_model_prefill):
        """Build draft_attn_metadatas for partial-merged draft graph."""
        attn_metadata = self.model_state.attn_metadata
        attn_metadata = {
            name: metadata for name, metadata in attn_metadata.items() if name in self.draft_attn_layer_names
        }

        if is_draft_model_prefill:
            return [attn_metadata]

        attn_metadata = self._get_decode_base_metadata(attn_metadata, num_reqs_padded)

        draft_attn_metadatas = self._init_decode_draft_attn_metadatas(attn_metadata, num_reqs_padded)

        for i, per_step_attn_metadata in enumerate(draft_attn_metadatas):
            step = i + 1
            assert self.input_batch is not None
            self._update_decode_attn_metadata(per_step_attn_metadata, step, self.input_batch.num_reqs)

        return draft_attn_metadatas

    def _get_decode_base_metadata(self, attn_metadata, num_reqs_padded):
        # Rebuild fresh if live metadata is a stale prefill residual (.decode=None).
        need_fresh = any(getattr(m, "decode", "MISSING") is None for m in attn_metadata.values())
        if need_fresh:
            assert self.input_batch is not None
            return self._build_draft_attn_metadata(
                num_reqs=self.input_batch.num_reqs,
                num_reqs_padded=num_reqs_padded,
                num_tokens_padded=num_reqs_padded,
            )
        return attn_metadata

    def get_draft_decode_num_reqs_padded(self, desc: BatchExecutionDescriptor) -> int:
        # MLA draft: padded num_tokens (FIA TND). Flat: num_reqs.
        return desc.num_tokens if self._draft_uses_mla() else desc.num_reqs

    def _draft_uses_mla(self) -> bool:
        # Cached: draft attn backend is MLA vs flat.
        cache = getattr(self, "_draft_uses_mla_cache", None)
        if cache is not None:
            return cache
        from vllm_ascend.attention.mla_v1 import AscendMLABackend

        result = any(isinstance(b, type) and issubclass(b, AscendMLABackend) for b in self.attn_backends.values())
        self._draft_uses_mla_cache = result
        return result

    def _ascend_update_seq_lens(self, attn_metadata: dict[str, Any] | None) -> None:
        if attn_metadata is not None:
            for attn_meta in attn_metadata.values():
                attn_meta.seq_lens = attn_meta.seq_lens + 1
                attn_meta.seq_len_list = attn_meta.seq_lens.tolist()

    def _init_decode_draft_attn_metadatas(self, attn_metadata: dict[str, Any] | None, num_reqs_padded: int):
        """Initialize attention metadata for decode phase in graph mode on Ascend NPUs."""
        if attn_metadata is None:
            return

        attn_state = AscendAttentionState.DecodeOnly

        draft_attn_metadatas = []
        # attn_metadata is build in vllm's super class.
        # We need to update attn_state for each layer's metadata.
        for seq_lens_cpu in self.input_buffers.draft_seq_lens_cpus:
            per_step_attn_metadata = {k: copy(v) for k, v in attn_metadata.items()}

            seq_lens_cpu = seq_lens_cpu[:num_reqs_padded]
            for metadata in per_step_attn_metadata.values():
                metadata.attn_state = attn_state
                metadata.seq_lens_cpu = seq_lens_cpu
                self._prepare_step_decode_fields(metadata, num_reqs_padded)
            draft_attn_metadatas.append(per_step_attn_metadata)

        return draft_attn_metadatas

    def _prepare_step_decode_fields(self, metadata, num_reqs_padded: int) -> None:
        # Per-step .decode copy + persistent block_table snapshot (async update
        # races with next add_requests mutating input_block_tables in place).
        decode_meta = getattr(metadata, "decode", None)
        if decode_meta is not None:
            metadata.decode = copy(decode_meta)
            input_bts = self.block_tables.input_block_tables
            assert len(input_bts) == 1, "MTP/MLA draft expects a single KV cache group"
            src = input_bts[0]
            buf = getattr(self, "_draft_block_table_buf", None)
            if buf is None or buf.shape[0] < src.shape[0]:
                buf = src.new_zeros(src.shape)
                self._draft_block_table_buf = buf
            buf[:num_reqs_padded].copy_(src[:num_reqs_padded])
            metadata.decode.block_table = buf[:num_reqs_padded]

    def _update_decode_attn_metadata(
        self, attn_metadata: dict[str, Any] | None, step: int, num_reqs: int | None = None
    ):
        """Update attention metadata for decode phase on Ascend NPUs."""
        if attn_metadata is None:
            return

        num_reqs_padded = next(iter(attn_metadata.values())).seq_lens_cpu.shape[0]
        seq_lens_cpu = self._get_seq_lens_cpu()[:num_reqs_padded]
        if num_reqs is None:
            num_reqs = num_reqs_padded
        next_seq_lens_cpu = self._calc_next_seq_lens_cpu(seq_lens_cpu, num_reqs, num_reqs_padded, step)

        query_lens_list = [i for i in range(1, num_reqs_padded + 1)]
        seq_lens_list = next_seq_lens_cpu.tolist()
        # attn_metadata is build in vllm's super class.
        # We need to update attn_state for each layer's metadata.
        for metadata in attn_metadata.values():
            metadata.actual_seq_lengths_q = query_lens_list
            metadata.seq_lens_cpu.copy_(next_seq_lens_cpu)
            metadata.seq_lens_list = seq_lens_list
            self._write_decode_step_fields(metadata, query_lens_list, seq_lens_list)

    def _write_decode_step_fields(self, metadata, query_lens_list, seq_lens_list) -> None:
        decode_meta = getattr(metadata, "decode", None)
        if decode_meta is not None:
            decode_meta.actual_seq_lengths_q = query_lens_list
            decode_meta.seq_lens_list = seq_lens_list

    def _calc_next_seq_lens_cpu(self, seq_lens_cpu, num_reqs, num_reqs_padded, step):
        # NOTE(drslark) to achieve fully alignment with vllm, `num_rejected` should be subtracted from `seq_lens`
        # to avoid extra sync overhead, `v2` is currently aligned with NPU `v1` only

        # follows the logic in `prepare_eagle_decode` and `update_eagle_inputs`
        next_seqs_cpu = torch.clamp(seq_lens_cpu[:num_reqs_padded] + step, max=self.max_model_len)
        next_seqs_cpu[num_reqs:].fill_(0)
        return next_seqs_cpu

    def _get_seq_lens_cpu(self) -> torch.Tensor:
        """Get seq_lens_cpu from input_batch."""
        assert self.input_batch is not None
        seq_lens_cpu = torch.from_numpy(self.input_batch.seq_lens_np)
        return seq_lens_cpu


# TODO Remove this patch when cann fix the gather bug.
# NOTE(Ronald1995): torch.gather will pollute the cache such as self.input_buffers.positions
# the bug is reported to huawei CANN team, but not fixed yet.
# NOTE(drslark): make a temporary patch only for `torch.gather`
_original_gather = torch.gather


def gather(input, dim, index, *, sparse_grad=False, out=None):
    if out is None:
        return _original_gather(input, dim, index, sparse_grad=sparse_grad)
    out[:] = _original_gather(input, dim, index, sparse_grad=sparse_grad)
    return out


@contextmanager
def torch_gather_wrapper():
    """Context manager to override torch.gather for Ascend NPUs."""
    original_gather = torch.gather
    try:
        torch.gather = gather
        yield
    finally:
        torch.gather = original_gather
