# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/aclgraph_utils.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
import os
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
import torch_npu
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, ModelCudaGraphManager
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.ubatch_utils import UBatchRunner
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.compilation.acl_graph import (
    set_graph_params,
    update_full_graph_params,
)
from vllm_ascend.compilation.breakable_aclgraph import BreakableACLGraphWrapper
from vllm_ascend.compilation.updatable_graph import (
    ContextSource,
    UpdatableGraph,
)
from vllm_ascend.utils import use_updatable_graph
from vllm_ascend.worker.v2.utils import communicator_switch


def collect_sorted_captured_token_sizes(capture_descs: dict) -> list[int]:
    """Collect the actual per-graph token counts that will be captured.

    With speculative decoding under FULL_DECODE_ONLY, each raw
    ``cudagraph_capture_size`` is rounded up to a multiple of
    ``decode_query_len`` (see ``CudaGraphManager._init_candidates``), so the
    real graph sizes differ from ``compilation_config.cudagraph_capture_sizes``.
    The attention backend keys its per-size graph params (events/handles/...)
    by these rounded token counts, so they must be derived from the actual
    capture descriptors, not the raw config sizes.
    """
    return sorted({desc.num_tokens for descs in capture_descs.values() for desc in descs})


def _get_graph_update_backend(
    attn_groups: list[list[AttentionGroup]],
) -> type[AttentionBackend]:
    for groups in attn_groups:
        for group in groups:
            backend = group.backend
            try:
                impl_cls = backend.get_impl_cls()
            except NotImplementedError:
                # Metadata-only backends such as GDN have no attention impl.
                continue
            if impl_cls is not None:
                return backend
    raise RuntimeError("No executable attention backend is available for full-graph parameter updates.")


class ModelAclGraphManager(ModelCudaGraphManager):
    """ACL Model Cuda Graph Manager for Ascend NPUs."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int | Mapping[int, int],
        model_runner: Any,
        lora_capture_cases: list[int] | None = None,
        varlen_decode: bool = False,
        ubatch_runner: UBatchRunner | None = None,
    ):
        # vLLM main (#51700) passes the microbatch runner into the graph
        # manager.
        super().__init__(
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_len,
            lora_capture_cases=lora_capture_cases,
            varlen_decode=varlen_decode,
            ubatch_runner=ubatch_runner,
        )
        self.breakable_cg_runner: BreakableACLGraphWrapper | None = None
        self._cached_graph_update_backend: type[AttentionBackend] | None = None
        self._cached_use_updatable: bool | None = None
        self.model_runner = model_runner
        self.update_stream = self.model_runner.update_stream
        self.capture_sizes = collect_sorted_captured_token_sizes(self._capture_descs)
        is_npu = device.type == "npu"
        self._update_sync_event = torch.npu.Event() if is_npu else None
        self._update_sync_ready = False
        self._bg_enabled = is_npu and os.environ.get("VLLM_ASCEND_DISABLE_BG_TASK0") != "1"
        self._bg_req = threading.Event()
        self._bg_done = threading.Event()
        self._bg_done.set()
        self._bg_job: tuple[UpdatableGraph, Any, Any, torch.Tensor] | None = None
        self._bg_result: tuple[UpdatableGraph, Any, Any, torch.Tensor] | None = None
        self._bg_error: Exception | None = None
        self._bg_stop = False
        self._bg_thread: threading.Thread | None = None
        if self._bg_enabled:
            self._bg_thread = threading.Thread(
                target=self._bg_update_loop,
                name="aclgraph-bg-t0",
                daemon=True,
            )
            self._bg_thread.start()
        if super().needs_capture():
            set_graph_params(self.capture_sizes)

    def init_breakable_cg_runner(self, model: nn.Module) -> None:
        if self.breakable_cg_runner is None:
            self.breakable_cg_runner = BreakableACLGraphWrapper(model, self.vllm_config)

    def run_fullgraph(self, desc: BatchExecutionDescriptor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Override run_fullgraph to update full graph params in run_fullgraph."""
        num_tokens = desc.num_tokens
        assert self.update_stream is not None
        if getattr(self, "_cached_use_updatable", None) is None:
            with set_current_vllm_config(self.vllm_config):
                attn_backend = _get_graph_update_backend(self.model_runner.attn_groups)
            self._cached_graph_update_backend = attn_backend
            self._cached_use_updatable = use_updatable_graph(attn_backend)
        else:
            cached_backend = getattr(self, "_cached_graph_update_backend", None)
            if cached_backend is None:
                with set_current_vllm_config(self.vllm_config):
                    attn_backend = _get_graph_update_backend(self.model_runner.attn_groups)
                self._cached_graph_update_backend = attn_backend
            else:
                attn_backend = cached_backend
        attn_metadata = self.model_runner.model_state.attn_metadata

        metadata = getattr(self.model_runner.model_state, "device_metadata", None)
        try:
            prepare_engram = getattr(self.model_runner.model_state, "prepare_engram", None)
            if prepare_engram is not None:
                # Match FULL replay's DP padding before lookup consumes it.
                with (
                    set_current_vllm_config(self.vllm_config),
                    set_forward_context(
                        attn_metadata,
                        self.vllm_config,
                        num_tokens=num_tokens,
                        cudagraph_runtime_mode=desc.cg_mode,
                        num_tokens_across_dp=torch.full([self.model_runner.dp_size], num_tokens),
                        batch_descriptor=None,
                        slot_mapping=None,
                    ),
                ):
                    prepare_engram()
            if self._cached_use_updatable:
                return self._updatable_graph_replay(desc, attn_metadata)
            else:
                # This will be removed once the refactoring is fully complete.
                return self._graph_relay(attn_backend, desc, num_tokens, attn_metadata)
        finally:
            if metadata is not None:
                metadata.finish_replay()

    def _bg_update_loop(self) -> None:
        if self.device.type == "npu":
            torch.npu.set_device(self.device)
        while True:
            self._bg_req.wait()
            self._bg_req.clear()
            if self._bg_stop:
                self._bg_done.set()
                return
            job = self._bg_job
            self._bg_job = None
            if job is None:
                self._bg_done.set()
                continue
            graph, aq, akv, bt = job
            try:
                t0 = graph.tasks[0].bind(
                    {
                        "actual_seq_lengths": aq,
                        "actual_seq_lengths_kv": akv,
                        "block_table": bt,
                    }
                )
                with torch.npu.stream(self.update_stream):
                    t0.apply(self.update_stream)
                self._bg_result = (graph, aq, akv, bt)
                self._bg_error = None
            except Exception as exc:
                self._bg_error = exc
                self._bg_result = None
            finally:
                self._bg_done.set()

    def _join_bg(self) -> tuple[UpdatableGraph, Any, Any, torch.Tensor] | None:
        bg_done = getattr(self, "_bg_done", None)
        if bg_done is None:
            return None
        bg_done.wait()
        err = getattr(self, "_bg_error", None)
        self._bg_error = None
        if err is not None:
            self._bg_result = None
            raise err
        return getattr(self, "_bg_result", None)

    def release_graphs(self) -> None:
        self._join_bg()
        self._bg_result = None
        self._update_sync_ready = False
        super().release_graphs()

    def start_bg_first_task(self, desc: BatchExecutionDescriptor, input_batch: Any) -> None:
        self._join_bg()
        self._bg_result = None
        if not getattr(self, "_bg_enabled", False) or self.update_stream is None:
            return
        if desc.cg_mode != CUDAGraphMode.FULL or input_batch.has_prefill:
            return
        graph = self.graphs.get(desc)
        if not isinstance(graph, UpdatableGraph) or not graph.tasks:
            return
        t0 = graph.tasks[0]
        if t0.operation is not torch_npu.npu_fused_infer_attention_score.out:
            return
        prov = t0.provider
        if getattr(prov, "is_non_pa", False) or getattr(prov, "sliding_window", None):
            return
        block_tables_all = self.model_runner.block_tables.input_block_tables
        if not block_tables_all:
            return
        prov_width = getattr(prov, "width", None)
        if prov_width is not None:
            capacity = prov.capacity  # type: ignore[attr-defined]
            real_rows = input_batch.num_reqs
            bt_mgr = self.model_runner.block_tables
            if (
                real_rows <= 0
                or real_rows > capacity
                or input_batch.num_tokens != real_rows * prov_width
                or input_batch.num_tokens_after_padding != capacity * prov_width
                or bt_mgr.has_sliding_window_groups
                or bt_mgr.cp_size != 1
                or block_tables_all[0].shape[0] < capacity
            ):
                return
            aq = None
            akv = input_batch.seq_lens_np[:real_rows].tolist() + [prov_width] * (capacity - real_rows)
            bt = block_tables_all[0][:capacity]
        else:
            num_reqs_padded = input_batch.num_reqs_after_padding
            num_input_tokens = input_batch.num_tokens_after_padding
            aq = input_batch.query_start_loc_np[1:].tolist()
            max_query_len = int(input_batch.num_scheduled_tokens.max()) if input_batch.num_reqs > 0 else 1
            if max_query_len > 1 and aq and num_input_tokens > aq[-1]:
                aq.append(num_input_tokens)
            akv = input_batch.seq_lens_np[:num_reqs_padded].tolist()
            if len(akv) < len(aq):
                akv.extend([1] * (len(aq) - len(akv)))
            bt = block_tables_all[0][:num_reqs_padded]
            if bt.shape[0] < len(aq):
                return
        self._bg_job = (graph, aq, akv, bt)
        self._bg_done.clear()
        self._bg_req.set()
        time.sleep(0)

    def _graph_relay(self, attn_backend, desc, num_tokens, attn_metadata):
        self._join_bg()
        self._bg_result = None
        self.update_stream.wait_stream(torch.npu.current_stream())
        ret = super().run_fullgraph(desc)

        # refer to vllm.v1.worker.gpu.dp_utils.sync_cudagraph_and_dp_padding to
        # calculate num_tokens_across_dp.
        num_tokens_across_dp = torch.full([self.model_runner.dp_size], num_tokens)
        # Attention graph updates must see the target ModelConfig.
        # set_forward_context() does not update the current VllmConfig.
        # TODO: Remove this explicit current-config scope once ACL graph replay
        # passes VllmConfig directly through the graph-update interfaces.
        with (
            set_current_vllm_config(self.vllm_config),
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=desc.cg_mode,
                num_tokens_across_dp=num_tokens_across_dp,
                batch_descriptor=None,  # Full graph model don't need batch_descriptor
                slot_mapping=None,
            ),
        ):
            forward_context = get_forward_context()
            update_full_graph_params(
                # FIXME(Ronald1995): support hybrid attn backend
                attn_backend,
                self.update_stream,
                forward_context,
                num_tokens,
                self.vllm_config,
                self.model_runner.speculative_config,
            )
        logger.info_once("ACL graph replay is active for the V2 target model (logged once).")
        return ret

    def _updatable_graph_replay(self, desc, attn_metadata):
        graph = self.graphs[desc]
        assert isinstance(graph, UpdatableGraph)
        prefetched = self._join_bg()
        self._bg_result = None
        if self._update_sync_event is not None and not self._update_sync_ready:
            self._update_sync_event.record(torch.npu.current_stream())
            self.update_stream.wait_event(self._update_sync_event)
            self._update_sync_ready = True

        resolved_tasks = None
        if graph.tasks:
            first_prov = graph.tasks[0].provider
            if getattr(first_prov, "width", None) is not None:
                bt_mgr = self.model_runner.block_tables
                can_defer = False
                if (
                    not bt_mgr.has_sliding_window_groups
                    and bt_mgr.cp_size == 1
                    and bt_mgr.input_block_tables
                    and isinstance(attn_metadata, dict)
                ):
                    meta0 = attn_metadata.get(getattr(first_prov, "layer_name", ""))
                    full_bt = bt_mgr.input_block_tables[0]
                    if meta0 is not None:
                        meta0._suffix_bsh_full_block_table = full_bt
                        cap = getattr(first_prov, "capacity", 0)
                        if full_bt.shape[0] >= cap and full_bt.is_contiguous():
                            can_defer = all(v is meta0 for v in attn_metadata.values())
                if not can_defer:
                    # Resolve before run_fullgraph so any fallback tensor op precedes replay.
                    resolved_tasks = graph.resolve_tasks(ContextSource(attn_metadata))
        ret = super().run_fullgraph(desc)
        if resolved_tasks is None:
            resolved_tasks = graph.resolve_tasks(ContextSource(attn_metadata))
        tasks_to_update = resolved_tasks
        if prefetched is not None and resolved_tasks:
            p_graph, p_aq, p_akv, p_bt = prefetched
            t0_kw = resolved_tasks[0].kwargs
            t0_bt = t0_kw.get("block_table")
            bt_ok = (p_bt is t0_bt) or (
                t0_bt is not None
                and p_bt.data_ptr() == t0_bt.data_ptr()
                and p_bt.shape == t0_bt.shape
                and p_bt.stride() == t0_bt.stride()
            )
            if (
                p_graph is graph
                and p_aq == t0_kw.get("actual_seq_lengths")
                and p_akv == t0_kw.get("actual_seq_lengths_kv")
                and bt_ok
            ):
                tasks_to_update = resolved_tasks[1:]

        if tasks_to_update:
            graph.update(self.update_stream, tasks_to_update)
        if self._update_sync_event is not None:
            self._update_sync_event.record(torch.npu.current_stream())
            self.update_stream.wait_event(self._update_sync_event)
        return ret

    def capture(
        self,
        model: nn.Module,
        model_state: ModelState,
        input_buffers: InputBuffers,
        intermediate_tensors: IntermediateTensors | None,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        has_lora: bool = False,
        use_aux_hidden_state_outputs: bool = False,
        lora_capture_hook: Callable[[int, int, int], None] | None = None,
        progress_bar_desc: str = "Capturing CUDA graphs",
        # vLLM #53869 supplies PCP slot mappings during graph capture.
        pcp_manager: Any = None,
    ) -> None:
        """Capture CUDA graphs for model forward pass."""
        metadata = getattr(model_state, "device_metadata", None)
        model = ModelWithContext(model, device_metadata=metadata)
        try:
            with communicator_switch():
                return super().capture(
                    model,
                    model_state,
                    input_buffers,
                    intermediate_tensors,
                    block_tables,
                    attn_groups,
                    kv_cache_config,
                    pcp_manager=pcp_manager,
                    has_lora=has_lora,
                    use_aux_hidden_state_outputs=use_aux_hidden_state_outputs,
                    lora_capture_hook=lora_capture_hook,
                    progress_bar_desc=progress_bar_desc,
                )
        finally:
            # Input preparation can fail before ModelWithContext.forward runs.
            if metadata is not None:
                metadata.finish()


class ModelWithContext(nn.Module):
    """Define a wrapper model to inject forward context.
    so we can inherit vllm's CudaGraphManager._capture_full_graph.
    """

    def __init__(self, original_model, is_draft_model=False, is_draft_model_prefill=False, device_metadata=None):
        super().__init__()
        self.original_model = original_model
        self.is_draft_model = is_draft_model
        self.is_draft_model_prefill = is_draft_model_prefill
        self.device_metadata = device_metadata

    def forward(self, *args, **kwargs):
        forward_context = get_forward_context()
        # In warmup phase, capturing=False by default.
        # when capturing, we need to set capturing=True in forward context.
        _EXTRA_CTX.capturing = (
            torch.npu.is_current_stream_capturing()
            and forward_context.cudagraph_runtime_mode != CUDAGraphMode.PIECEWISE
        )
        if self.is_draft_model:
            _EXTRA_CTX.is_draft_model = True
        if self.is_draft_model_prefill:
            _EXTRA_CTX.is_draft_model_prefill = True

        try:
            if self.device_metadata is not None:
                self.device_metadata.begin_forward()
            return self.original_model(*args, **kwargs)
        finally:
            # Warmup and actual capture receive separate prepare_attn calls.
            # Retire each submission, including unused producers, after its
            # consumers; do not wait for metadata at graph entry.
            if self.device_metadata is not None:
                self.device_metadata.finish()

    def get_original_model(self):
        return self.original_model

    def __getattr__(self, name: str):
        # vLLM #57632 moved the draft context-K/V precompute into the
        # capture path, which reaches model methods through this wrapper.
        try:
            return super().__getattr__(name)
        except AttributeError:
            original_model = self._modules.get("original_model")
            if original_model is None:
                raise
            return getattr(original_model, name)


@contextmanager
def model_capture_wrapper(speculator, is_draft_model_prefill):
    """Context manager to override speculator's model for speculator capturing."""
    try:
        speculator.model = ModelWithContext(speculator.model, True, is_draft_model_prefill)
        yield
    finally:
        speculator.model = speculator.model.get_original_model()
