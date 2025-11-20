from unittest.mock import MagicMock, Mock

import pytest
import torch
from pytest_mock import MockerFixture
from vllm.config import CacheConfig, VllmConfig

from tests.ut.base import PytestBase
from vllm_ascend.torchair.torchair_model_runner import NPUTorchairModelRunner


class TestNPUTorchairModelRunner(PytestBase):

    # @pytest.fixture
    # def setup_npu_torchair_model_runner(self, mocker: MockerFixture):
    #     vllm_config = MagicMock(spec=VllmConfig)
    #     vllm_config.model_config = MagicMock()
    #     vllm_config.model_config.hf_config = MagicMock()
    #     vllm_config.model_config.hf_config.index_topk = 2
    #     cache_config = CacheConfig(block_size=16)
    #     vllm_config.cache_config = cache_config
    #     speculative_config = MagicMock()
    #     speculative_config.num_speculative_tokens = 4
    #     vllm_config.speculative_config = speculative_config
    #     vllm_config.compilation_config = MagicMock()

    #     device = torch.device("npu:0")

    #     ascend_config = MagicMock()
    #     ascend_config.max_num_batched_tokens = 2048
    #     ascend_config.max_model_len = 1024
    #     ascend_config.torchair_graph_config = MagicMock()
    #     ascend_config.torchair_graph_config.use_cached_graph = True
    #     ascend_config.torchair_graph_config.use_cached_kv_cache_bytes = False
    #     ascend_config.torchair_graph_config.graph_batch_sizes = [1, 2, 4]
    #     ascend_config.torchair_graph_config.graph_batch_sizes_init = True

    #     # mocker.patch(
    #     #     "vllm_ascend.worker.model_runner_v1.NPUModelRunner.__init__",
    #     #     return_value=None)

    #     mocker.patch("vllm_ascend.utils.get_ascend_config",
    #                  return_value=ascend_config)
    #     mocker.patch("vllm_ascend.torchair.utils.register_torchair_model")
    #     mocker.patch("vllm_ascend.torchair.utils.torchair_ops_patch")
    #     mocker.patch(
    #         "vllm_ascend.torchair.utils.torchair_quant_method_register")
    #     mocker.patch("vllm_ascend.envs.VLLM_ASCEND_TRACE_RECOMPILES",
    #                  return_value=False)

    #     mock_attn_builder = Mock()
    #     mock_attn_backend = Mock()
    #     mock_attn_backend.get_builder_cls.return_value = lambda *args, **kwargs: mock_attn_builder

    #     NPUTorchairModelRunner.decode_token_per_req = 1
    #     NPUTorchairModelRunner.max_num_tokens = 10

    #     runner = NPUTorchairModelRunner(vllm_config, device)
    #     runner.vllm_config = vllm_config
    #     runner.device = device
    #     runner.attn_backend = mock_attn_backend
    #     runner.ascend_config = ascend_config
    #     runner.model_config = vllm_config.model_config

    #     return runner
    @pytest.fixture
    def setup_npu_torchair_model_runner(self, mocker: MockerFixture):
        vllm_config = MagicMock(spec=VllmConfig)
        vllm_config.model_config = MagicMock()
        cache_config = CacheConfig(block_size=16)
        vllm_config.cache_config = cache_config
        vllm_config.model_config.hf_config = MagicMock()
        vllm_config.model_config.hf_config.index_topk = 2
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.use_mla = False
        vllm_config.model_config.get_hidden_size.return_value = 512
        vllm_config.model_config.pooler_config = None
        vllm_config.model_config.logits_processors = []
        cache_config = MagicMock()
        cache_config.block_size = 16
        cache_config.cache_dtype = "auto"
        vllm_config.cache_config = cache_config

        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config

        vllm_config.compilation_config = MagicMock()
        vllm_config.compilation_config.cudagraph_mode = Mock()
        vllm_config.compilation_config.cudagraph_capture_sizes = [1, 2, 4]

        vllm_config.lora_config = MagicMock()
        vllm_config.parallel_config = MagicMock()
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.parallel_config.data_parallel_rank = 0
        vllm_config.parallel_config.cp_kv_cache_interleave_size = 1

        scheduler_config = MagicMock()
        scheduler_config.max_num_batched_tokens = 2048
        scheduler_config.max_num_seqs = 64
        scheduler_config.chunked_prefill_enabled = True
        scheduler_config.async_scheduling = False
        vllm_config.scheduler_config = scheduler_config

        vllm_config.load_config = MagicMock()
        
        vllm_config.kv_transfer_config = None

        mocker.patch(
            "vllm_ascend.worker.model_runner_v1.is_pin_memory_available",
            return_value=True)
        mocker.patch("vllm_ascend.worker.model_runner_v1.cdiv",
                     return_value=64)
        mocker.patch(
            "vllm_ascend.worker.model_runner_v1.prefill_context_parallel_enable",
            return_value=False)
        mocker.patch("vllm_ascend.worker.model_runner_v1.get_dcp_group"
                     ).return_value.world_size = 1
        mocker.patch(
            "vllm_ascend.torchair.torchair_model_runner.get_attn_backend",
            autospec=True)
        mocker.patch(
            "vllm_ascend.torchair.torchair_model_runner._set_up_drafter")
        mocker.patch(
            "vllm_ascend.torchair.torchair_model_runner._may_pad_kv_consumer_num_seq"
        )


        device = torch.device("npu:0")
        ascend_config = MagicMock()

        ascend_config.ascend_scheduler_config.enabled = False

        ascend_config.weight_prefetch_config = Mock()
        ascend_config.dynamic_eplb = False
        ascend_config.expert_map_record_path = None

        mocker.patch("vllm_ascend.utils.get_ascend_config",
                     return_value=ascend_config)
        mocker.patch("vllm_ascend.torchair.utils.register_torchair_model")
        mocker.patch("vllm_ascend.torchair.utils.torchair_ops_patch")
        mocker.patch(
            "vllm_ascend.torchair.utils.torchair_quant_method_register")
        mocker.patch("vllm_ascend.envs.VLLM_ASCEND_TRACE_RECOMPILES",
                     return_value=False)
        mock_attn_builder = Mock()
        mock_attn_backend = Mock()
        mock_attn_backend.get_builder_cls.return_value = lambda *args, **kwargs: mock_attn_builder

        NPUTorchairModelRunner.decode_token_per_req = 1
        NPUTorchairModelRunner.max_num_tokens = 10

        runner = NPUTorchairModelRunner(vllm_config, device)
        runner.vllm_config = vllm_config
        runner.device = device
        runner.attn_backend = mock_attn_backend
        runner.ascend_config = ascend_config
        runner.model_config = vllm_config.model_config

        return runner

    def test_init(self, mocker: MockerFixture,
                  setup_npu_torchair_model_runner):
        runner = setup_npu_torchair_model_runner
        assert isinstance(runner, NPUTorchairModelRunner)
