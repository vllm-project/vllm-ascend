from collections.abc import Callable, Mapping
from typing import Any

import torch
import vllm.v1.worker.gpu.spec_decode.dflash.cudagraph as dflash_cudagraph_module
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
from vllm_ascend.worker.v2.attn_utils import build_draft_attn_metadata_factory
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
        if self.speculator is None:
            raise RuntimeError("The draft speculator must be attached before ACLGraph capture.")

        def capture_is_prefilling(num_reqs: int) -> torch.Tensor:
            # Parallel-draft query graphs contain decode/speculative rows only.
            # Use an explicit request-sized tensor so padded capture rows cannot
            # inherit stale prefill flags from a previous runtime batch.
            return torch.zeros(num_reqs, dtype=torch.bool)

        with (
            communicator_switch(),
            model_capture_wrapper(self.speculator, False),
            build_draft_attn_metadata_factory(
                input_buffers.positions,
                pad=None,
                is_prefilling=capture_is_prefilling,
                module=dflash_cudagraph_module,
            ),
        ):
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
            raise RuntimeError("The draft speculator must be attached before ACLGraph replay.")
        if desc.num_reqs is None:
            raise ValueError("DFlash full graphs require a padded request count.")
        expected_num_tokens = desc.num_reqs * self.decode_query_len
        if num_tokens != expected_num_tokens:
            raise ValueError(
                "DFlash graph descriptor is not a uniform draft batch: "
                f"num_tokens={num_tokens}, num_reqs={desc.num_reqs}, "
                f"decode_query_len={self.decode_query_len}."
            )

        draft_attn_metadatas = self.speculator.build_draft_attn_metadatas(
            desc.num_reqs,
            self.speculator.input_batch.seq_lens_cpu_upper_bound,
        )
        self._validate_graph_param_cardinality(get_draft_graph_params(), num_tokens, draft_attn_metadatas)
        self.update_stream.wait_stream(torch.npu.current_stream())
        ret = super().run_fullgraph(desc)

        # DPMetadata validates these counts on the CPU. Keep them off the
        # replay stream so validation cannot wait for the graph whose attention
        # parameters still need to be updated below.
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
            # decide to update draft graph params
            _EXTRA_CTX.is_draft_model = True

            _EXTRA_CTX.is_draft_model_prefill = False

            forward_context = get_forward_context()

            attn_backend = getattr(self.speculator, "attn_backend", None)
            if attn_backend is None:
                unique_backends = set(self.speculator.attn_backends.values())
                if len(unique_backends) != 1:
                    backend_names = sorted(backend.__name__ for backend in unique_backends)
                    raise RuntimeError(
                        "DFlash ACLGraph requires one homogeneous draft "
                        "attention backend, but found "
                        f"{backend_names or ['none']}."
                    )
                attn_backend = next(iter(unique_backends))
            update_full_graph_params(
                attn_backend,
                self.update_stream,
                forward_context,
                num_tokens,
                self.vllm_config,
                self.speculator.speculative_config,
                draft_attn_metadatas=draft_attn_metadatas,
            )
        return ret

    @staticmethod
    def _validate_graph_param_cardinality(graph_params: Any, num_tokens: int, metadatas: Any) -> None:
        """Reject incomplete update buckets before submitting graph replay."""
        if graph_params is None:
            raise RuntimeError("Draft ACLGraph parameters have not been initialized.")
        buckets = [getattr(graph_params, field).get(num_tokens) for field in ("attn_params", "handles", "events")]
        if any(bucket is None for bucket in buckets):
            raise RuntimeError(f"Draft ACLGraph has no complete parameter bucket for {num_tokens} tokens.")
        counts = [len(bucket) for bucket in buckets]
        if not counts[0] or len(set(counts)) != 1:
            raise RuntimeError(f"Draft ACLGraph update cardinality mismatch: params/handles/events={counts}.")
        steps = [metadatas] if isinstance(metadatas, dict) else metadatas
        if not steps or any(not step for step in steps):
            raise RuntimeError("Draft ACLGraph requires nonempty attention metadata.")
        # MLA updates only decode attention entries; hybrid state metadata is
        # not an FIA task. Match the backend's filtering rather than counting
        # every state entry as an attention layer.
        uses_mla_metadata = any(hasattr(meta, "decode") for step in steps for meta in step.values())
        layer_count = sum(
            sum(getattr(meta, "decode", None) is not None for meta in step.values()) if uses_mla_metadata else len(step)
            for step in steps
        )
        if not layer_count:
            raise RuntimeError("Draft ACLGraph has no attention layers to update.")
        if counts[0] % layer_count:
            raise RuntimeError(
                f"Draft ACLGraph task count {counts[0]} is not divisible by metadata layer count {layer_count}."
            )
