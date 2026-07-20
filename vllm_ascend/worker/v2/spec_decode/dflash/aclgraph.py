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
    set_draft_graph_params,
    update_full_graph_params,
)
from vllm_ascend.utils import vllm_version_is
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
        # vllm v0.25.1 passes ``causal=self.dflash_causal`` here while the
        # vllm main branch removed it from ``init_cudagraph_manager`` and moved
        # it into ``capture()`` instead. Accepting ``causal`` via ``**kwargs``
        # keeps us compatible with both pinned versions; it is simply forwarded
        # to the upstream ``__init__`` which only consumes it on v0.25.1.
        **kwargs: Any,
    ):
        super().__init__(
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_len,
            **kwargs,
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
            # On vllm v0.25.1, ``causal`` is forwarded via ``__init__`` and the
            # upstream ``DFlashCudaGraphManager.capture`` does not accept it.
            # On vllm main, ``causal`` was moved into ``capture()``, so forward
            # it there. Gate on the pinned vllm version to stay compatible.
            if vllm_version_is("0.25.1"):
                super().capture(
                    forward_fn,
                    input_buffers,
                    block_tables,
                    attn_groups,
                    kv_cache_config,
                    max_model_len,
                    progress_bar_desc,
                )
            else:
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

        draft_attn_metadatas = self.speculator.build_draft_attn_metadatas(desc.num_reqs)

        ret = super().run_fullgraph(desc)

        positions = self.speculator.input_buffers.positions[:num_tokens]

        # refer to vllm.v1.worker.gpu.dp_utils.sync_cudagraph_and_dp_padding to
        # calculate num_tokens_across_dp.
        num_tokens_across_dp = torch.full([self.speculator.dp_size], num_tokens, device=self.device)

        with set_forward_context(
            self.speculator.model_state.attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=desc.cg_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            batch_descriptor=None,  # Full graph model don't need batch_descriptor
            slot_mapping=None,
        ):
            # decide to update draft graph params
            _EXTRA_CTX.is_draft_model = True

            _EXTRA_CTX.is_draft_model_prefill = False

            forward_context = get_forward_context()

            update_full_graph_params(
                # FIXME(Ronald1995): support hybrid attn backend
                list(self.speculator.attn_backends.values())[0],
                self.speculator.update_stream,
                forward_context,
                num_tokens,
                self.vllm_config,
                self.speculator.speculative_config,
                positions.shape[0],
                draft_attn_metadatas=draft_attn_metadatas,
            )
        return ret
