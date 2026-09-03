import os
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
from vllm_ascend.worker.v2.aclgraph_utils import collect_sorted_captured_token_sizes, model_capture_wrapper
from vllm_ascend.worker.v2.utils import communicator_switch
from vllm_ascend.worker.v2.spec_decode.physical_k import (
    configured_capture_k,
    physical_k_scope,
    query_width,
    v2_varlen_physical_k_enabled,
)


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
        self._v2_varlen_physical_k = v2_varlen_physical_k_enabled(vllm_config)
        if self._v2_varlen_physical_k:
            self._extend_varlen_capture_descriptors()
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

    def _sample_from_anchor(self) -> bool:
        speculative_config = self.vllm_config.speculative_config
        draft_model_config = getattr(speculative_config, "draft_model_config", None)
        hf_config = getattr(draft_model_config, "hf_config", None)
        if getattr(speculative_config, "use_dspark", lambda: False)():
            return bool(getattr(hf_config, "sample_from_anchor", True))
        return False

    def _extend_varlen_capture_descriptors(self) -> None:
        """Capture one FULL descriptor for each configured physical K.

        The upstream manager only expands capture widths for its native
        dynamic-spec configuration.  Ascend's hardware-aware policy is an
        independent scheduler path, so add the same descriptor matrix here.
        If a width is not captured, normal dispatch falls back to eager mode;
        it never reuses a graph with a different query width.
        """

        decode_mode = self.cudagraph_mode.decode_mode()
        if decode_mode == CUDAGraphMode.NONE:
            return
        capture_sizes = sorted(self.compilation_config.cudagraph_capture_sizes or [])
        if not capture_sizes:
            return

        speculative_config = self.vllm_config.speculative_config
        max_k = int(getattr(speculative_config, "num_speculative_tokens", 0))
        if max_k <= 0:
            return
        sample_from_anchor = self._sample_from_anchor()
        max_capture_size = (
            self.compilation_config.max_cudagraph_capture_size or (1 << 60)
        )
        max_decode_tokens = self.max_num_reqs * self.decode_query_len

        capture_descs = self._capture_descs.setdefault(decode_mode, [])
        for raw_tokens in capture_sizes:
            for draft_k in configured_capture_k(self.vllm_config, max_k):
                width = query_width(sample_from_anchor, draft_k)
                rounded_tokens = ((raw_tokens + width - 1) // width) * width
                num_reqs = rounded_tokens // width
                if (
                    rounded_tokens > max_decode_tokens
                    or rounded_tokens > max_capture_size
                    or num_reqs > self.max_num_reqs
                ):
                    continue
                for num_active_loras in self.lora_capture_cases:
                    desc = BatchExecutionDescriptor(
                        cg_mode=decode_mode,
                        num_tokens=rounded_tokens,
                        num_reqs=num_reqs,
                        uniform_token_count=width,
                        num_active_loras=num_active_loras,
                    )
                    if desc not in capture_descs:
                        capture_descs.append(desc)
                    self._candidates.setdefault(
                        (rounded_tokens, num_active_loras), []
                    ).append(desc)
                    for token_count in range(0, rounded_tokens + 1):
                        self._candidates.setdefault(
                            (token_count, num_active_loras), []
                        ).append(desc)

        capture_descs.sort(key=lambda item: item.num_tokens, reverse=True)
        for key, candidates in self._candidates.items():
            unique = list(dict.fromkeys(candidates))
            candidates[:] = unique

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
        debug_capture_sync = os.environ.get("VLLM_ASCEND_DFLASH_CAPTURE_SYNC") == "1"
        previous_desc: tuple[int, int, int] | None = None

        def forward_with_runtime_width(
            num_reqs: int,
            num_tokens: int,
            attn_metadata: Any,
            slot_mappings: Any,
            num_tokens_across_dp: Any,
            cg_mode: CUDAGraphMode,
        ):
            if not self._v2_varlen_physical_k or num_reqs <= 0:
                return forward_fn(
                    num_reqs,
                    num_tokens,
                    attn_metadata,
                    slot_mappings,
                    num_tokens_across_dp,
                    cg_mode,
                )
            width = num_tokens // num_reqs
            draft_k = width if self._sample_from_anchor() else width - 1
            is_capturing = torch.npu.is_current_stream_capturing()
            nonlocal previous_desc
            current_desc = (num_tokens, num_reqs, draft_k)
            # The ACL graph capture API reports many device-side failures only
            # at the next stream synchronization.  Sync before the next
            # descriptor's warmup so the failing descriptor can be identified.
            if debug_capture_sync and not is_capturing:
                logger.warning(
                    "DFlash V2 graph-capture sync before desc=%s; previous=%s",
                    current_desc,
                    previous_desc,
                )
                torch.npu.current_stream().synchronize()
            with physical_k_scope(self.speculator, draft_k=draft_k):
                result = forward_fn(
                    num_reqs,
                    num_tokens,
                    attn_metadata,
                    slot_mappings,
                    num_tokens_across_dp,
                    cg_mode,
                )
            if debug_capture_sync and is_capturing:
                previous_desc = current_desc
            return result

        with communicator_switch(), model_capture_wrapper(self.speculator, False):
            super().capture(
                forward_with_runtime_width,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                max_model_len,
                causal,
                progress_bar_desc,
            )
            if debug_capture_sync:
                logger.warning(
                    "DFlash V2 graph-capture final sync; last_desc=%s",
                    previous_desc,
                )
                torch.npu.current_stream().synchronize()

    def run_fullgraph(self, desc: BatchExecutionDescriptor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Override run_fullgraph to update full graph params in run_fullgraph."""
        num_tokens = desc.num_tokens

        draft_attn_metadatas = self.speculator.build_draft_attn_metadatas(
            desc.num_reqs,
            self.speculator.input_batch.seq_lens_cpu_upper_bound,
            num_tokens_padded=num_tokens,
        )
        self.update_stream.wait_stream(torch.npu.current_stream())
        ret = super().run_fullgraph(desc)

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
                self.update_stream,
                forward_context,
                num_tokens,
                self.vllm_config,
                self.speculator.speculative_config,
                draft_attn_metadatas=draft_attn_metadatas,
            )
        return ret
