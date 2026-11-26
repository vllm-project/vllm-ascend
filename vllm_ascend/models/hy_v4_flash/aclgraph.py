# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full ACLGraphs for uniform HYV4 Flash decode batches."""

from vllm.config import CUDAGraphMode
from vllm.logger import logger
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager

from .decode_fusions import uses_decode_fusions


class FlashAclGraphManager(ModelAclGraphManager):
    def capture(self, model, model_state, *args, **kwargs):
        super().capture(model, model_state, *args, **kwargs)
        self._short_graphs = {}
        if not uses_decode_fusions(self.vllm_config):
            return
        layers = [
            module
            for module in model.modules()
            if getattr(module, "indexer", None) is not None and hasattr(module, "_short_indexpool_capture")
        ]
        if not layers:
            return
        self._short_limit = min((layer.indexer.budget + 1) * layer.indexer.ratio for layer in layers)
        long_graphs = self.graphs
        self.graphs = {}
        try:
            for layer in layers:
                layer._short_indexpool_capture = True
            super().capture(model, model_state, *args, **kwargs)
            self._short_graphs = self.graphs
        finally:
            self.graphs = long_graphs
            for layer in layers:
                layer._short_indexpool_capture = False

    def _init_candidates(self):
        super()._init_candidates()
        decode_graphs = [
            desc for desc in self._capture_descs.get(CUDAGraphMode.FULL, ()) if desc.uniform_token_count is not None
        ]
        self._capture_descs.clear()
        self._candidates.clear()
        if decode_graphs:
            self._capture_descs[CUDAGraphMode.FULL] = decode_graphs

    def dispatch(self, num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len=None, num_ubatches=1):
        if uniform_token_count is not None:
            desc = super().dispatch(
                num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len, num_ubatches
            )
            if desc in self.graphs:
                return desc
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_active_loras=num_active_loras,
            num_ubatches=num_ubatches,
        )

    def run_fullgraph(self, desc):
        metadata = self.model_runner.model_state.attn_metadata
        batch = next(iter(metadata.values()), None) if isinstance(metadata, dict) else None
        short_graph = getattr(self, "_short_graphs", {}).get(desc)
        use_short = (
            short_graph is not None and batch is not None and max(batch.seq_lens, default=0) <= self._short_limit
        )
        original = self.graphs[desc]
        try:
            if use_short:
                self.graphs[desc] = short_graph
            output = super().run_fullgraph(desc)
        finally:
            self.graphs[desc] = original
        logger.info_once(
            "FULL ACLGraph replay is active for HYV4 Flash target (%d requests, %d query tokens).",
            desc.num_reqs,
            desc.num_tokens,
        )
        return output
