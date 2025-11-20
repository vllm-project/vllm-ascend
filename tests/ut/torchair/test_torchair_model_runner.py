import os
import unittest
import pytest
from unittest.mock import Mock, MagicMock, patch

import torch
from vllm_ascend.torchair_model_runner import NPUTorchairModelRunner
from vllm.config import VllmConfig


class TestNPUTorchairModelRunner(PytestBase):

    @pytest.fixture
    def setup_npu_torchair_model_runner(self, mocker: MockerFixture):
        vllm_config = MagicMock(spec=VllmConfig)
        vllm_config.model_config = MagicMock()
        vllm_config.model_config.hf_config = MagicMock()
        vllm_config.model_config.hf_config.index_topk = 2

        device = torch.device("npu:0")

        ascend_config = MagicMock()
        ascend_config = enable_shared_expert_dp = False
        ascend_config.max_num_batched_tokens = 2048
        ascend_config.max_model_len = 1024
        ascend_config.torchair_graph_config = MagicMock()
        ascend_config.torchair_graph_config.use_cached_graph = True
        ascend_config.torchair_graph_config.use_cached_kv_cache_bytes = False
        ascend_config.torchair_graph_config.graph_batch_sizes = [1, 2, 4]
        ascend_config.torchair_graph_config.graph_batch_sizes_init = True

        mocker.patch("vllm_ascend.worker.model_runner_v1.NPUModelRunner.__init__",
                    return_value=None)

        mocker.patch("vllm_ascend.get_ascend_config", return_value=ascend_config)
        mocker.patch("vllm_ascend.torchair.utils.register_torchair_model")
        mocker.patch("vllm_ascend.torchair.utils.torchair_ops_patch")
        mocker.patch("vllm_ascend.torchair.utils.torchair_quant_method_register")
        mocker.patch("vllm_ascend.envs.VLLM_ASCEND_TRACE_RECOMPILES", return_value=False)

        mock_attn_builder = Mock()
        mock_attn_backend = Mock()
        mock_attn_backend.get_builder_cls.return_value = lambda *args, **kwargs: mock_attn_builder
        with patch.object(NPUTorchairModelRunner, 'attn_backend', mock_attn_backend):
            with patch.object(NPUTorchairModelRunner, 'speculative_config', MagicMock()):
                NPUTorchairModelRunner.decode_token_per_req = 1
                NPUTorchairModelRunner.max_num_tokens = 10

                runner = NPUTorchairModelRunner(vllm_config, device)
                runner.vllm_config = vllm_config
                runner.device = device
                runner.attn_backend = mock_attn_backend
        
        return runner

    def test_init(self, mocker: MockerFixture, setup_npu_torchair_model_runner):
        runner = setup_npu_torchair_model_runner
        assert isinstance(runner, NPUTorchairModelRunner)