# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
    prepare_inputs_to_capture,
)
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.compilation.acl_graph import (
    get_draft_graph_params,
    set_draft_graph_params,
    set_draft_graph_prefill_params,
    update_full_graph_params,
)
from vllm_ascend.compilation.updatable_graph import (
    SharedSource,
    UpdatableGraph,
)
from vllm_ascend.utils import use_updatable_graph, vllm_version_is
from vllm_ascend.worker.v2.aclgraph_utils import (
    collect_sorted_captured_token_sizes,
    model_capture_wrapper,
)
from vllm_ascend.worker.v2.spec_decode.upstream import SpeculatorCudaGraphManager
from vllm_ascend.worker.v2.utils import communicator_switch


@dataclass(frozen=True)
class DraftBatchDescriptor(BatchExecutionDescriptor):
    num_speculative_tokens: int = 0


class AutoRegressiveAclGraphManager(SpeculatorCudaGraphManager):
    """ACL graph manager for autoregressive speculative decoding."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
        lora_capture_cases: list[int] | None = None,
    ):
        super().__init__(
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_len,
            lora_capture_cases=lora_capture_cases,
        )

        # Upstream constructs graph managers without a speculator reference.
        # AscendAutoRegressiveSpeculator attaches it after construction so replay
        # can rebuild draft metadata and update graph parameters.
        self.speculator: Any = None
        # The attention backend keys its per-size graph params by the actual
        # captured token counts (rounded up to decode_query_len when using
        # speculative decoding), so derive them from the capture descriptors
        # instead of the raw config sizes.
        self.capture_sizes = collect_sorted_captured_token_sizes(self._capture_descs)
        # Upstream uses num_speculative_steps + 1 as the draft-prefill query
        # length and 1 for draft decode.
        self.is_draft_model_prefill = decode_query_len > 1
        if super().needs_capture():
            if self.is_draft_model_prefill:
                set_draft_graph_prefill_params(self.capture_sizes)
            else:
                set_draft_graph_params(self.capture_sizes)

    def capture(
        self,
        forward_fn: Callable,
        model_state: ModelState,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        progress_bar_desc: str = "Capturing CUDA graphs",
        specialize_spec_tokens: bool = False,
    ) -> None:
        """Capture ACL graphs for autoregressive speculative decoding."""

        if specialize_spec_tokens and not self.is_draft_model_prefill and not vllm_version_is("0.30.0"):
            config = self.speculator.speculative_config
            schedule = build_dynamic_sd_schedule_lookup(
                config.num_speculative_tokens_per_batch_size,
                self.max_num_reqs,
                self.speculator.num_speculative_steps,
            )
            self._capture_descs = {
                mode: [
                    DraftBatchDescriptor(**vars(desc), num_speculative_tokens=k)
                    for desc in descs
                    for k in sorted(set(schedule) - {0, 1}, reverse=True)
                ]
                for mode, descs in self._capture_descs.items()
            }
            self._draft_graph_params_by_desc: dict[BatchExecutionDescriptor, dict[str, Any]] = {}

        with communicator_switch(), model_capture_wrapper(self.speculator, self.is_draft_model_prefill):
            if self.is_draft_model_prefill:
                super().capture(
                    forward_fn,
                    model_state,
                    input_buffers,
                    block_tables,
                    attn_groups,
                    kv_cache_config,
                    progress_bar_desc=progress_bar_desc,
                )
                return

            def create_forward_fn(desc: BatchExecutionDescriptor, warmup: bool):
                num_tokens = desc.num_tokens
                num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)
                num_tokens_across_dp = (
                    torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                    if self.dp_size > 1
                    else None
                )
                prepare_inputs_to_capture(
                    num_reqs,
                    num_tokens,
                    model_state,
                    input_buffers,
                    block_tables,
                    attn_groups,
                    kv_cache_config,
                    full_cudagraph=(desc.cg_mode == CUDAGraphMode.FULL),
                    # vLLM #58275 makes mixed FULL graphs capture prefill kernels
                    # by bounding the dummy query length to the capture
                    # descriptor. v0.30.0 has no such default and keeps its own
                    # (unbounded) behaviour.
                    **(
                        {"max_query_len": desc.max_query_len or desc.uniform_token_count}
                        if not vllm_version_is("0.30.0")
                        else {}
                    ),
                )
                seq_lens_cpu_upper_bound = input_buffers.seq_lens_cpu[:num_reqs]
                forward_kwargs = {}
                if not vllm_version_is("0.30.0"):
                    forward_kwargs["num_speculative_steps"] = (
                        desc.num_speculative_tokens
                        if isinstance(desc, DraftBatchDescriptor)
                        else self.speculator.num_speculative_steps
                    )

                def run(cg_mode):
                    with self._draft_graph_params_scope(desc):
                        forward_fn(
                            num_reqs,
                            cg_mode == CUDAGraphMode.PIECEWISE,
                            BatchExecutionDescriptor(cg_mode=cg_mode, num_tokens=num_tokens, num_reqs=num_reqs),
                            num_tokens_across_dp,
                            seq_lens_cpu_upper_bound,
                            **forward_kwargs,
                        )

                return run

            CudaGraphManager.capture(self, create_forward_fn, progress_bar_desc=progress_bar_desc)

    def specialize_spec_tokens(self, desc: BatchExecutionDescriptor, num_speculative_tokens: int):
        if desc.cg_mode != CUDAGraphMode.FULL or self.is_draft_model_prefill:
            return desc
        specialized = (
            replace(desc, num_speculative_tokens=num_speculative_tokens)
            if isinstance(desc, DraftBatchDescriptor)
            else DraftBatchDescriptor(**vars(desc), num_speculative_tokens=num_speculative_tokens)
        )
        if specialized in self.graphs:
            return specialized
        if desc in self.graphs and num_speculative_tokens == self.speculator.num_speculative_steps:
            return desc
        # A merged graph contains a fixed number of steps. Never replay a
        # different width when the selected graph has not been captured.
        return replace(desc, cg_mode=CUDAGraphMode.NONE)

    @contextmanager
    def _draft_graph_params_scope(self, desc):
        if not isinstance(desc, DraftBatchDescriptor):
            yield
            return
        # Legacy MLA stores task handles by token count. Keep separate lists
        # for graphs with equal batch shapes but different draft widths.
        pool = get_draft_graph_params()
        if pool is None:
            yield
            return
        num_tokens = desc.num_tokens
        fields = ("events", "handles", "attn_params", "workspaces")
        previous = {name: getattr(pool, name)[num_tokens] for name in fields}
        params = self._draft_graph_params_by_desc.setdefault(
            desc, {"events": [], "handles": [], "attn_params": [], "workspaces": previous["workspaces"]}
        )
        for name in fields:
            getattr(pool, name)[num_tokens] = params[name]
        try:
            yield
        finally:
            for name in fields:
                params[name] = getattr(pool, name)[num_tokens]
                getattr(pool, name)[num_tokens] = previous[name]

    def run_fullgraph(self, desc: BatchExecutionDescriptor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Replay the draft ACL graph and update its attention parameters."""
        num_tokens = desc.num_tokens
        logger.info_once(
            "ACL graph replay is active for the V2 draft model (%s, logged once per phase).",
            "prefill" if self.is_draft_model_prefill else "decode",
        )
        assert self.update_stream is not None

        attn_backend = self.speculator.attn_backend
        draft_vllm_config = self.speculator.draft_vllm_config
        draft_attn_metadatas = self.speculator.build_draft_attn_metadatas(
            desc.num_reqs,
            desc.num_tokens,
            self.is_draft_model_prefill,
            **(
                {"num_speculative_steps": desc.num_speculative_tokens} if isinstance(desc, DraftBatchDescriptor) else {}
            ),
        )
        with self._draft_graph_params_scope(desc):
            if use_updatable_graph(attn_backend):
                return self._updatable_graph_replay(desc, draft_attn_metadatas)
            else:
                # This will be removed once the refactoring is fully complete.
                return self._graph_replay(desc, attn_backend, num_tokens, draft_vllm_config, draft_attn_metadatas)

    def _graph_replay(self, desc, attn_backend, num_tokens, draft_vllm_config, draft_attn_metadatas):
        self.update_stream.wait_stream(torch.npu.current_stream())
        ret = super().run_fullgraph(desc)
        # Mirror vLLM's DP graph-replay token-count metadata.
        num_tokens_across_dp = torch.full([self.speculator.dp_size], num_tokens)
        attn_metadata = self.speculator.model_state.attn_metadata
        # sfa_v1.py:AscendSFABackend.get_impl_cls reaches
        # sfa_cp.py:resolve_sfa_impl, whose SFA CP selector reads the current
        # ModelConfig. Publish the draft config because set_forward_context()
        # does not update it.
        # TODO: Remove this explicit current-config scope once ACL graph replay
        # passes VllmConfig directly through the graph-update interfaces.
        with (
            set_current_vllm_config(draft_vllm_config),
            set_forward_context(
                attn_metadata,
                draft_vllm_config,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=desc.cg_mode,
                num_tokens_across_dp=num_tokens_across_dp,
                batch_descriptor=None,  # Full graph model don't need batch_descriptor
                slot_mapping=None,
            ),
        ):
            # Select the draft prefill/decode graph-parameter pool.
            _EXTRA_CTX.is_draft_model = True
            _EXTRA_CTX.is_draft_model_prefill = self.is_draft_model_prefill

            forward_context = get_forward_context()
            assert attn_backend is not None, "Speculator attention backend is not initialized"
            update_full_graph_params(
                # FIXME(Ronald1995): support hybrid attn backend
                attn_backend,
                self.update_stream,
                forward_context,
                num_tokens,
                draft_vllm_config,
                self.speculator.speculative_config,
                draft_attn_metadatas=draft_attn_metadatas,
            )
        return ret

    def _updatable_graph_replay(self, desc, draft_attn_metadatas):
        graph = self.graphs[desc]
        assert isinstance(graph, UpdatableGraph)
        fia_params = self.speculator.build_fia_params(
            desc.num_reqs,
            draft_attn_metadatas[0],
            self.is_draft_model_prefill,
            **(
                {"num_speculative_steps": desc.num_speculative_tokens} if isinstance(desc, DraftBatchDescriptor) else {}
            ),
        )
        resolved_tasks = graph.resolve_tasks(SharedSource(fia_params))
        self.update_stream.wait_stream(torch.npu.current_stream())
        ret = super().run_fullgraph(desc)
        graph.update(self.update_stream, resolved_tasks)
        return ret
