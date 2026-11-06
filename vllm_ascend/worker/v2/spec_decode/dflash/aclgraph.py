from collections.abc import Callable, Mapping
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (  # type: ignore[import-not-found]
    BatchExecutionDescriptor,
)
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.fa_v1 import AscendFlashAttentionBackend
from vllm_ascend.compilation.acl_graph import (
    set_draft_graph_params,
    update_full_graph_params,
)
from vllm_ascend.compilation.updatable_graph import (
    ContextSource,
    UpdatableGraph,
)
from vllm_ascend.utils import use_updatable_graph
from vllm_ascend.worker.v2.aclgraph_utils import collect_sorted_captured_token_sizes, model_capture_wrapper
from vllm_ascend.worker.v2.spec_decode.dflash.capture import draft_window_dflash_capture_scope
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import draft_window_all_continuing
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

    def dispatch(
        self,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        num_active_loras: int,
        max_query_len: int | None = None,
        num_ubatches: int = 1,
    ) -> BatchExecutionDescriptor:
        if draft_window_all_continuing(self.speculator):
            # This explicit phase keeps Full reads; never a missing-bucket or UPDATE fallback.
            return BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                num_active_loras=self._resolve_effective_loras(num_active_loras),
                num_ubatches=num_ubatches,
            )
        return super().dispatch(
            num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len, num_ubatches
        )

    def capture(
        self,
        forward_fn: Callable,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        causal: bool | Mapping[int, bool] = False,
        precompute_context_kv: Callable[[int], None] | None = None,
        prepare_forward: Callable[[int, dict[str, Any] | None], None] | None = None,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """Capture ACL graphs for DFlash.

        vLLM #60692 renamed ``precompute_context_kv(num_reqs)`` to
        ``prepare_forward(num_reqs, attn_metadata)``; accept both and forward the
        one the installed vLLM expects.
        """
        prepare_kwargs: dict[str, Any] = {"prepare_forward": prepare_forward}
        with (
            draft_window_dflash_capture_scope(self.speculator) as context,
            communicator_switch(),
            model_capture_wrapper(self.speculator, False),
        ):
            super().capture(
                forward_fn,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                max_model_len,
                causal=causal,
                progress_bar_desc=progress_bar_desc,
                **prepare_kwargs,
            )
            if context is not None:
                context.validate_captured_graphs(self.graphs)

    def run_fullgraph(self, desc: BatchExecutionDescriptor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Override run_fullgraph to update full graph params in run_fullgraph."""
        num_tokens = desc.num_tokens
        dp_dummy = getattr(self.speculator, "_draft_window_dp_dummy_active", False)
        window_provider = None if dp_dummy else getattr(self.speculator, "_draft_window_graph_binding_provider", None)
        attn_backend = list(self.speculator.attn_backends.values())[0]
        if window_provider is not None and not use_updatable_graph(attn_backend):
            raise NotImplementedError("Draft Window Graph requires the admitted updatable backend binding contract")
        if all(backend is AscendFlashAttentionBackend for backend in self.speculator.attn_backends.values()):
            # Length preparation and FlashAttn tiling run inside the graph,
            # reading the draft's persistent device buffers on every replay.
            return super().run_fullgraph(desc)
        if window_provider is None:
            draft_attn_metadatas = self.speculator.build_draft_attn_metadatas(
                desc.num_reqs,
                self.speculator.input_batch.seq_lens_cpu_upper_bound,
            )
        else:
            window_provider.bind_current(desc)
            draft_attn_metadatas = None
        if use_updatable_graph(attn_backend):
            ret = self._updatable_graph_replay(
                desc,
                draft_attn_metadatas,
                window_source=window_provider,
            )
            if window_provider is not None:
                window_provider.finish_update()
            return ret
        else:
            # Ordinary OFF/eager Target policy keeps the original fallback.
            return self._graph_replay(desc, attn_backend, num_tokens, draft_attn_metadatas)

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

    def _updatable_graph_replay(self, desc, draft_attn_metadatas, *, window_source=None):
        graph = self.graphs[desc]
        assert isinstance(graph, UpdatableGraph)
        resolved_tasks = graph.resolve_tasks(
            ContextSource(draft_attn_metadatas[0]) if window_source is None else window_source
        )
        self.update_stream.wait_stream(torch.npu.current_stream())
        ret = super().run_fullgraph(desc)
        graph.update(self.update_stream, resolved_tasks)
        return ret
