# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
from collections.abc import Callable, Mapping, Sequence
from copy import copy
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
    prepare_inputs_to_capture,
)
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import SpeculatorCudaGraphManager
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_config import draft_config_context
from vllm_ascend.ascend_forward_context import (
    _EXTRA_CTX,
    MoECommType,
    resolve_draft_moe_quant_type,
    select_moe_comm_method,
)
from vllm_ascend.compilation.acl_graph import (
    set_draft_graph_params,
    set_draft_graph_prefill_params,
    update_full_graph_params,
)
from vllm_ascend.compilation.updatable_graph import UpdatableGraph
from vllm_ascend.utils import is_drafter_moe_model, use_updatable_graph
from vllm_ascend.worker.v2.aclgraph_utils import (
    AscendGraphManagerMixin,
    collect_sorted_captured_token_sizes,
    model_capture_wrapper,
)
from vllm_ascend.worker.v2.spec_decode import resolve_speculator_draft_vllm_config
from vllm_ascend.worker.v2.utils import communicator_switch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.pcp_manager import PCPManager


def _draft_batch_supports_full_aclgraph(num_tokens: int, draft_config: VllmConfig) -> bool:
    with draft_config_context():
        return (
            select_moe_comm_method(
                num_tokens,
                draft_config,
                is_draft_model=True,
                draft_moe_quant_type=resolve_draft_moe_quant_type(draft_config, is_draft_model=True),
            )
            is not MoECommType.ALLTOALL
        )


def filter_draft_full_aclgraph_sizes(
    vllm_config: VllmConfig,
    cudagraph_mode: CUDAGraphMode,
    decode_query_lens: Mapping[int, int] | None,
    decode_request_sizes: Sequence[int] | None = None,
    *,
    varlen_decode: bool = False,
) -> tuple[VllmConfig, CUDAGraphMode, Sequence[int] | None]:
    # Both FULL and PIECEWISE capture MoE forward, where ALLTOALL synchronizes
    # dynamic split counts on the host.
    if (
        cudagraph_mode == CUDAGraphMode.NONE
        or getattr(vllm_config, "speculative_config", None) is None
        or not is_drafter_moe_model(vllm_config)
    ):
        return vllm_config, cudagraph_mode, decode_request_sizes

    draft_config = resolve_speculator_draft_vllm_config(None, vllm_config)
    assert draft_config is not None
    # Keep the target capture grids intact.
    compilation = vllm_config.compilation_config
    graph_config = copy(vllm_config)
    graph_config.compilation_config = copy(compilation)
    graph_compilation = graph_config.compilation_config
    mixed_mode = cudagraph_mode.mixed_mode()
    if mixed_mode != CUDAGraphMode.NONE:
        graph_compilation.prefill_cudagraph_capture_token_sizes = [
            size
            for size in compilation.prefill_cudagraph_capture_token_sizes or ()
            if _draft_batch_supports_full_aclgraph(size, draft_config)
        ]
        if not graph_compilation.prefill_cudagraph_capture_token_sizes:
            mixed_mode = CUDAGraphMode.NONE

    has_decode = False
    if cudagraph_mode.separate_routine():
        assert decode_query_lens is not None
        if varlen_decode:
            sizes = [
                size
                for size in vllm_config.get_varlen_decode_cudagraph_token_sizes(max(decode_query_lens))
                if _draft_batch_supports_full_aclgraph(size, draft_config)
            ]
            if sizes:
                graph_compilation.max_decode_cudagraph_capture_tokens = max(sizes)
            has_decode = bool(sizes)
        else:
            candidate_sizes = (
                compilation.decode_cudagraph_capture_request_sizes
                if decode_request_sizes is None
                else decode_request_sizes
            )
            max_decode_tokens = compilation.max_cudagraph_capture_size or 0
            if compilation.max_decode_cudagraph_capture_tokens:
                max_decode_tokens = (
                    min(max_decode_tokens, compilation.max_decode_cudagraph_capture_tokens)
                    if max_decode_tokens > 0
                    else compilation.max_decode_cudagraph_capture_tokens
                )
            effective_limits = [
                (query_len, min(max_reqs, max_decode_tokens // query_len) if max_decode_tokens > 0 else max_reqs)
                for query_len, max_reqs in decode_query_lens.items()
            ]
            effective_limits = [(query_len, limit) for query_len, limit in effective_limits if limit >= 1]
            decode_request_sizes = (
                [
                    size
                    for size in candidate_sizes or ()
                    if all(
                        _draft_batch_supports_full_aclgraph(min(size, limit) * query_len, draft_config)
                        for query_len, limit in effective_limits
                    )
                ]
                if effective_limits
                else []
            )
            has_decode = bool(decode_request_sizes)

    if not has_decode:
        cudagraph_mode = mixed_mode
    elif mixed_mode == CUDAGraphMode.NONE:
        cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    return graph_config, cudagraph_mode, decode_request_sizes


def get_draft_prefill_cudagraph_mode(vllm_config: VllmConfig, cudagraph_mode: CUDAGraphMode) -> CUDAGraphMode:
    """Keep PCP-sharded draft prefill out of FULL graphs.

    An autoregressive draft keeping the target's PCP size shards its prefill
    through the upstream PCP hooks. Their rank-local input staging and the
    all-gather in ``restore_draft_prefill`` run outside the model forward,
    which a FULL graph would skip on replay.
    """
    if vllm_config.parallel_config.prefill_context_parallel_size <= 1:
        return cudagraph_mode
    return CUDAGraphMode.PIECEWISE if cudagraph_mode.has_piecewise_cudagraphs() else CUDAGraphMode.NONE


class AutoRegressiveAclGraphManager(AscendGraphManagerMixin, SpeculatorCudaGraphManager):
    """ACL graph manager for autoregressive speculative decoding."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_lens: Mapping[int, int],
        lora_capture_cases: list[int] | None = None,
        varlen_decode: bool = False,
        *,
        decode_request_sizes: Sequence[int] | None = None,
    ):
        # Upstream uses num_speculative_steps + 1 as the draft-prefill query
        # length and 1 for draft decode.
        # vLLM #63406 keys uniform decode graphs by request count.
        decode_query_len = max(decode_query_lens)
        if decode_query_len > 1:
            cudagraph_mode = get_draft_prefill_cudagraph_mode(vllm_config, cudagraph_mode)
        vllm_config, cudagraph_mode, decode_request_sizes = filter_draft_full_aclgraph_sizes(
            vllm_config, cudagraph_mode, decode_query_lens, decode_request_sizes, varlen_decode=varlen_decode
        )
        super().__init__(
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_lens,
            lora_capture_cases=lora_capture_cases,
            varlen_decode=varlen_decode,
            decode_request_sizes=decode_request_sizes,
        )

        # Upstream constructs graph managers without a speculator reference.
        # AscendTargetDependentARSpeculator attaches it after construction so replay
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
        pcp_manager: "PCPManager | None" = None,
    ) -> None:
        """Capture ACL graphs for autoregressive speculative decoding.

        Draft prefill reuses the target's PCP-local metadata at runtime, so
        its capture stages the dummy through ``pcp_manager`` like the target
        capture does. TODO: drop once upstream SpeculatorCudaGraphManager
        forwards ``pcp_manager`` to ``prepare_inputs_to_capture``.
        """

        with communicator_switch(), model_capture_wrapper(self.speculator, self.is_draft_model_prefill):

            def create_forward_fn(desc: BatchExecutionDescriptor, warmup: bool):
                num_tokens = desc.num_tokens
                num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)
                num_tokens_across_dp = (
                    torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                    if self.dp_size > 1
                    else None
                )
                attn_metadata, slot_mappings = prepare_inputs_to_capture(
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
                    # descriptor.
                    max_query_len=desc.max_query_len or desc.uniform_token_count,
                    pcp_manager=pcp_manager,
                )
                if self.is_draft_model_prefill:
                    return lambda cg_mode: forward_fn(
                        num_reqs, num_tokens, attn_metadata, slot_mappings, num_tokens_across_dp, cg_mode
                    )
                seq_lens_cpu_upper_bound = input_buffers.seq_lens_cpu[:num_reqs]
                return lambda cg_mode: forward_fn(
                    num_reqs,
                    cg_mode == CUDAGraphMode.PIECEWISE,
                    BatchExecutionDescriptor(cg_mode=cg_mode, num_tokens=num_tokens, num_reqs=num_reqs),
                    num_tokens_across_dp,
                    seq_lens_cpu_upper_bound,
                    # vLLM #60335 added `num_speculative_steps` on main.
                    *(self.speculator.num_speculative_steps,),
                )

            CudaGraphManager.capture(self, create_forward_fn, progress_bar_desc=progress_bar_desc)

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
        )
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
        # Attention graph updates must see the draft ModelConfig.
        # set_forward_context() does not update the current VllmConfig.
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
        source = self.speculator.build_param_source(
            desc.num_reqs,
            draft_attn_metadatas[0],
            self.is_draft_model_prefill,
            num_tokens=desc.num_tokens,
        )
        resolved_tasks = graph.resolve_tasks(source)
        self.update_stream.wait_stream(torch.npu.current_stream())
        ret = super().run_fullgraph(desc)
        graph.update(self.update_stream, resolved_tasks)
        return ret
