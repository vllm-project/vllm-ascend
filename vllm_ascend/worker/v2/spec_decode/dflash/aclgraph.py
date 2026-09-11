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
from vllm_ascend.compilation.acl_graph import (
    get_draft_graph_params,
    set_draft_graph_params,
    update_full_graph_params,
)
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

    def run_fullgraph(self, desc: BatchExecutionDescriptor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Override run_fullgraph to update full graph params in run_fullgraph."""
        num_tokens = desc.num_tokens

        if self.speculator is None:
            raise RuntimeError("Draft ACL graph manager has no bound speculator.")
        if self.update_stream is None:
            raise RuntimeError("Draft ACL graph manager has no bound update stream.")
        expected_num_tokens = desc.num_reqs * self.speculator.num_query_per_req
        if num_tokens != expected_num_tokens:
            raise RuntimeError(
                "Draft ACL graph descriptor mismatch: "
                f"num_tokens={num_tokens}, num_reqs={desc.num_reqs}, "
                f"num_query_per_req={self.speculator.num_query_per_req}."
            )

        draft_attn_metadatas = self.speculator.build_draft_attn_metadatas(
            desc.num_reqs,
            self.speculator.input_batch.seq_lens_cpu_upper_bound,
        )
        self._validate_graph_param_cardinality(get_draft_graph_params(), num_tokens)
        if hasattr(self.speculator, "get_draft_graph_backend"):
            draft_backend = self.speculator.get_draft_graph_backend()
        else:
            backends = set(self.speculator.attn_backends.values())
            if len(backends) != 1:
                raise NotImplementedError("Draft ACL graph requires one attention backend.")
            draft_backend = next(iter(backends))
        self.update_stream.wait_stream(torch.npu.current_stream())
        ret = super().run_fullgraph(desc)

        # refer to vllm.v1.worker.gpu.dp_utils.sync_cudagraph_and_dp_padding to
        # calculate num_tokens_across_dp.
        # DPMetadata validates these counts on the host. An NPU tensor would
        # synchronize graph replay before the parameter-update events are recorded.
        num_tokens_across_dp = torch.full([self.speculator.dp_size], num_tokens, device="cpu")

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
                draft_backend,
                self.update_stream,
                forward_context,
                num_tokens,
                self.vllm_config,
                self.speculator.speculative_config,
                draft_attn_metadatas=draft_attn_metadatas,
            )
        return ret

    @staticmethod
    def _validate_graph_param_cardinality(graph_params: Any, num_tokens: int) -> int:
        """Prevent backend task updates from silently dropping graph entries."""
        if graph_params is None:
            raise RuntimeError("Draft ACL graph parameters have not been initialized.")
        captured_params = graph_params.attn_params.get(num_tokens)
        handles = graph_params.handles.get(num_tokens)
        events = graph_params.events.get(num_tokens)
        if captured_params is None or handles is None or events is None:
            raise RuntimeError(f"Draft ACL graph has no captured parameter bucket for {num_tokens} tokens.")
        counts = (len(captured_params), len(handles), len(events))
        if counts[0] == 0:
            raise RuntimeError(f"Draft ACL graph captured no attention update tasks for {num_tokens} tokens.")
        if len(set(counts)) != 1:
            raise RuntimeError(
                "Draft ACL graph update cardinality mismatch for "
                f"{num_tokens} tokens: params={counts[0]}, handles={counts[1]}, events={counts[2]}."
            )
        return counts[0]
