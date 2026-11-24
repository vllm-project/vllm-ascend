# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full ACLGraphs for uniform HYV4 Flash decode batches."""

from vllm.config import CUDAGraphMode
from vllm.logger import logger
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager


class FlashAclGraphManager(ModelAclGraphManager):
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
        output = super().run_fullgraph(desc)
        logger.info_once(
            "FULL ACLGraph replay is active for HYV4 Flash target (%d requests, %d query tokens).",
            desc.num_reqs,
            desc.num_tokens,
        )
        return output
