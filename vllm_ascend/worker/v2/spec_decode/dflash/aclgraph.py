from collections.abc import Callable, Mapping
from copy import copy
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import logger
from vllm.utils.math_utils import round_up
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (  # type: ignore[import-not-found]
    BatchExecutionDescriptor,
)
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.compilation.acl_graph import (
    set_draft_graph_params,
    update_full_graph_params,
)
from vllm_ascend.compilation.updatable_graph import (
    ContextSource,
    UpdatableGraph,
)
from vllm_ascend.core.kv_cache_interface import uses_kimi_k31_gqa_draft
from vllm_ascend.utils import use_updatable_graph
from vllm_ascend.worker.v2.aclgraph_utils import collect_sorted_captured_token_sizes, model_capture_wrapper
from vllm_ascend.worker.v2.utils import communicator_switch


class DFlashAclGraphManager(DFlashCudaGraphManager):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
        speculator: Any = None,
    ):
        self.is_kimi_gqa_draft = uses_kimi_k31_gqa_draft(vllm_config)
        self._logged_serving_replay = False
        compilation = vllm_config.compilation_config
        sizes = compilation.cudagraph_capture_sizes
        if self.is_kimi_gqa_draft and cudagraph_mode.decode_mode() == CUDAGraphMode.FULL and sizes:
            # Each TP rank processes every draft token with a shard of the heads.
            # Fit whole draft requests inside the token and request limits.
            limit = min(
                max(sizes),
                compilation.max_cudagraph_capture_size,
                vllm_config.scheduler_config.max_num_seqs * decode_query_len,
                vllm_config.scheduler_config.max_num_batched_tokens,
            )
            limit -= limit % decode_query_len
            draft_sizes = sorted({min(round_up(size, decode_query_len), limit) for size in sizes}) if limit else []
            vllm_config = copy(vllm_config)
            vllm_config.compilation_config = copy(compilation)
            vllm_config.compilation_config.cudagraph_capture_sizes = draft_sizes
            vllm_config.compilation_config.max_cudagraph_capture_size = limit
        super().__init__(
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_len,
        )

        # It is set by AscendDFlashSpeculator.init_cudagraph_manager after creation,
        # because upstream's init_cudagraph_manager creates the manager without it.
        self.speculator = speculator
        # The attention backend keys its per-size graph params by the actual
        # captured token counts (rounded up to decode_query_len when using
        # speculative decoding), so derive them from the capture descriptors
        # instead of the raw config sizes.
        self.capture_sizes = collect_sorted_captured_token_sizes(self._capture_descs)
        # DFlash's parallel drafting forward has its own dedicated draft graph
        # path, independent of Eagle's prefill/decode split, so it always uses
        # the default draft params bucket (is_draft_model_prefill stays False in
        # both capture and replay to keep them consistent).
        if super().needs_capture():
            set_draft_graph_params(self.capture_sizes)

    def capture(
        self,
        forward_fn: Callable,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        causal: bool | Mapping[int, bool] = False,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """Capture ACL graphs for DFlash."""
        with communicator_switch(), model_capture_wrapper(self.speculator, False):
            super().capture(
                forward_fn,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                max_model_len,
                causal,
                progress_bar_desc,
            )
        if self.is_kimi_gqa_draft and self.graphs:
            logger.info(
                "Kimi K3.1 DSpark draft graph stage=capture captured_token_sizes=%s", self.captured_token_counts()
            )

    def run_fullgraph(self, desc: BatchExecutionDescriptor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Override run_fullgraph to update full graph params in run_fullgraph."""
        num_tokens = desc.num_tokens
        attn_backend = list(self.speculator.attn_backends.values())[0]
        draft_attn_metadatas = self.speculator.build_draft_attn_metadatas(
            desc.num_reqs,
            self.speculator.input_batch.seq_lens_cpu_upper_bound,
        )
        if use_updatable_graph(attn_backend):
            result = self._updatable_graph_replay(desc, draft_attn_metadatas)
        else:
            # This will be removed once the refactoring is fully complete.
            result = self._graph_replay(desc, attn_backend, num_tokens, draft_attn_metadatas)
        if self.is_kimi_gqa_draft and not self._logged_serving_replay and self.speculator._serving_draft_batch:
            logger.info(
                "Kimi K3.1 DSpark draft graph stage=replay mode=FULL requests=%d padded_requests=%d tokens=%d",
                self.speculator.input_batch.num_reqs,
                desc.num_reqs,
                desc.num_tokens,
            )
            self._logged_serving_replay = True
        return result

    def _graph_replay(self, desc, attn_backend, num_tokens, draft_attn_metadatas):
        self.update_stream.wait_stream(torch.npu.current_stream())
        ret = super().run_fullgraph(desc)
        # refer to vllm.v1.worker.gpu.dp_utils.sync_cudagraph_and_dp_padding to
        # calculate num_tokens_across_dp.
        num_tokens_across_dp = torch.full([self.speculator.dp_size], num_tokens)

        with set_forward_context(
            self.speculator.model_state.attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=desc.cg_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            batch_descriptor=None,  # Full graph model don't need batch_descriptor
            slot_mapping=None,
        ):
            _EXTRA_CTX.is_draft_model = True
            _EXTRA_CTX.is_draft_model_prefill = False
            forward_context = get_forward_context()
            update_full_graph_params(
                # FIXME(Ronald1995): support hybrid attn backend
                attn_backend,
                self.update_stream,
                forward_context,
                num_tokens,
                self.vllm_config,
                self.speculator.speculative_config,
                draft_attn_metadatas=draft_attn_metadatas,
            )
        return ret

    def _updatable_graph_replay(self, desc, draft_attn_metadatas):
        graph = self.graphs[desc]
        assert isinstance(graph, UpdatableGraph)
        resolved_tasks = graph.resolve_tasks(ContextSource(draft_attn_metadatas[0]))
        self.update_stream.wait_stream(torch.npu.current_stream())
        ret = super().run_fullgraph(desc)
        graph.update(self.update_stream, resolved_tasks)
        return ret
