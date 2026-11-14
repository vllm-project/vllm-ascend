# SPDX-License-Identifier: Apache-2.0
"""Keep platform graph-selection assertions on the active runner's real API.

This CPU fixture adapts the old key-initialization/tuple interface to MRv2's
descriptor interface. It builds real native dispatch tables; it does not
capture or replay any device graph.
"""

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

from vllm_ascend.utils import vllm_version_is


class _ModelRunnerV2Dispatcher:
    def __init__(self, vllm_config):
        self.vllm_config = vllm_config

    def initialize_cudagraph_keys(self, mode):
        query_len = self.vllm_config.uniform_decode_query_len
        pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
        with (
            patch("vllm.v1.worker.gpu.cudagraph_utils.get_pp_group", return_value=pp_group),
            patch("vllm.v1.worker.gpu.cudagraph_utils.current_platform.get_global_graph_pool", return_value=None),
        ):
            self.manager = CudaGraphManager(
                self.vllm_config,
                torch.device("cpu"),
                mode,
                {query_len: self.vllm_config.scheduler_config.max_num_seqs},
                decode_request_sizes=[
                    tokens // query_len
                    for tokens in self.vllm_config.compilation_config.cudagraph_capture_sizes
                    if tokens % query_len == 0
                ],
            )
        # dispatch() only selects initialized descriptors after this gate.
        # No graph is captured, installed in manager.graphs, or replayed here.
        self.manager._graphs_captured = True

    def dispatch(self, num_tokens, uniform_decode=False):
        query_len = self.vllm_config.uniform_decode_query_len
        desc = self.manager.dispatch(
            num_reqs=num_tokens // query_len if uniform_decode else 1,
            num_tokens=num_tokens,
            uniform_token_count=query_len if uniform_decode else None,
            num_active_loras=0,
        )
        return desc.cg_mode, desc


if vllm_version_is("0.33.0"):
    CudagraphDispatcher = import_module("vllm.v1.cudagraph_dispatcher").CudagraphDispatcher
else:
    CudagraphDispatcher = _ModelRunnerV2Dispatcher
