# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config.compilation import CompilationConfig, CUDAGraphMode
from vllm.v1.worker.gpu import cudagraph_utils
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import SpeculatorCudaGraphManager

from vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph import AutoRegressiveAclGraphManager


@pytest.mark.parametrize("query_len", [1, 4])
@pytest.mark.parametrize("dynamic", [False, True])
def test_draft_decode_does_not_inherit_target_verification_widths(query_len, dynamic):
    schedule = [[1, 8, 3], [9, 32, 0]] if dynamic else None
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            num_speculative_tokens_per_batch_size=schedule,
            uses_dynamic_speculative_decoding=lambda: dynamic,
        ),
    )
    received = []

    def init(manager, graph_config, *args, **kwargs):
        received.append(graph_config)
        manager._capture_descs = {}

    with (
        patch.object(SpeculatorCudaGraphManager, "__init__", init),
        patch.object(SpeculatorCudaGraphManager, "needs_capture", return_value=False),
    ):
        AutoRegressiveAclGraphManager(config, torch.device("cpu"), CUDAGraphMode.FULL_DECODE_ONLY, query_len)

    graph_config = received[0]
    if dynamic and query_len == 1:
        assert graph_config is not config
        assert graph_config.speculative_config is not config.speculative_config
        assert graph_config.speculative_config.num_speculative_tokens_per_batch_size is None
    else:
        assert graph_config is config
    # Never mutate the target or the draft-prefill schedule.
    assert config.speculative_config.num_speculative_tokens_per_batch_size == schedule


def test_dynamic_draft_decode_capture_descriptors_have_positive_single_token_width():
    class SpecConfig(SimpleNamespace):
        def uses_dynamic_speculative_decoding(self):
            return self.num_speculative_tokens_per_batch_size is not None

    compilation = CompilationConfig(cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32])
    compilation.max_cudagraph_capture_size = 32
    config = SimpleNamespace(
        speculative_config=SpecConfig(num_speculative_tokens_per_batch_size=[(1, 8, 3), (9, 32, 0)]),
        num_speculative_tokens=3,
        scheduler_config=SimpleNamespace(max_num_seqs=32),
        parallel_config=SimpleNamespace(data_parallel_size=1, tensor_parallel_size=1),
        compilation_config=compilation,
    )
    with (
        patch.object(
            cudagraph_utils, "get_pp_group", return_value=SimpleNamespace(is_first_rank=True, is_last_rank=True)
        ),
        patch.object(cudagraph_utils.current_platform, "get_global_graph_pool", return_value=None),
        patch("vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph.set_draft_graph_params"),
    ):
        manager = AutoRegressiveAclGraphManager(config, torch.device("cpu"), CUDAGraphMode.FULL_DECODE_ONLY, 1)

    descriptors = manager._capture_descs[CUDAGraphMode.FULL]
    assert descriptors
    assert all(desc.uniform_token_count == 1 and 0 < desc.num_reqs == desc.num_tokens for desc in descriptors)
