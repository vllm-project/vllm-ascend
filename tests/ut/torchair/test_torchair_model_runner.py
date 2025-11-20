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
        # 核心配置对象
        vllm_config = MagicMock(spec=VllmConfig)
        
        # --- 必需的配置对象 (用于初始化属性) ---
        vllm_config.model_config = MagicMock()
        vllm_config.model_config.hf_config = MagicMock()
        vllm_config.model_config.hf_config.index_topk = 2
        vllm_config.model_config.max_model_len = 1024 # 模拟 model_config.max_model_len
        vllm_config.model_config.use_mla = False # 模拟 model_config.use_mla
        vllm_config.model_config.get_hidden_size.return_value = 512 # 模拟 get_hidden_size
        vllm_config.model_config.pooler_config = None # 模拟 is_pooling_model
        vllm_config.model_config.logits_processors = [] # 模拟 build_logitsprocs

        cache_config = MagicMock()
        cache_config.block_size = 16
        cache_config.cache_dtype = "auto" # 模拟 cache_config.cache_dtype
        vllm_config.cache_config = cache_config
        
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        
        vllm_config.compilation_config = MagicMock()
        vllm_config.compilation_config.cudagraph_mode = Mock() # 模拟 compilation_config
        # 模拟 aclgraph_batch_sizes
        vllm_config.compilation_config.cudagraph_capture_sizes = [1, 2, 4] 
        
        vllm_config.lora_config = MagicMock()
        vllm_config.parallel_config = MagicMock()
        vllm_config.parallel_config.data_parallel_size = 1 # 模拟 dp_size
        vllm_config.parallel_config.data_parallel_rank = 0 # 模拟 dp_rank
        vllm_config.parallel_config.cp_kv_cache_interleave_size = 1 # 模拟 cp_kv_cache_interleave_size

        scheduler_config = MagicMock()
        scheduler_config.max_num_batched_tokens = 2048 # 模拟 max_num_tokens
        scheduler_config.max_num_seqs = 64 # 模拟 max_num_reqs (decode_max_num_seqs 默认为 0)
        scheduler_config.chunked_prefill_enabled = True # 模拟 chunked_prefill_enabled
        scheduler_config.async_scheduling = False # 模拟 use_async_scheduling
        vllm_config.scheduler_config = scheduler_config

        # --- 修复 'load_config' 报错 ---
        vllm_config.load_config = MagicMock() 
        
        # --- 模拟 kv_transfer_config (用于判断 kv role) ---
        vllm_config.kv_transfer_config = None

        # --- 模拟其他缺失的函数/常量 ---
        mocker.patch("vllm_ascend.worker.model_runner_v1.is_pin_memory_available",
                     return_value=True) # 模拟 pin_memory
        mocker.patch("vllm_ascend.worker.model_runner_v1.cdiv",
                     return_value=64) # 模拟 max_num_blocks_per_req
        mocker.patch("vllm_ascend.worker.model_runner_v1.prefill_context_parallel_enable",
                     return_value=False) # 模拟 pcp_size/rank
        mocker.patch("vllm_ascend.worker.model_runner_v1.get_dcp_group").return_value.world_size = 1 # 模拟 dcp_size/rank
        mocker.patch("vllm_ascend.torchair.torchair_model_runner.get_attn_backend",
                     autospec=True) # 模拟 Attention 设置
        mocker.patch("vllm_ascend.torchair.torchair_model_runner._set_up_drafter") # 模拟 drafter 设置
        mocker.patch("vllm_ascend.torchair.torchair_model_runner._may_pad_kv_consumer_num_seq") # 模拟 kv 填充

        # NPU特定的配置
        device = torch.device("npu:0")
        ascend_config = MagicMock()
        # 确保 ascend_scheduler_config.enabled 被设置，否则 chunked_prefill_enabled 会被设置为 True
        ascend_config.ascend_scheduler_config.enabled = False 
        # 其他 NPU/Ascend 特有配置
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

        # 设置类属性（如果需要）
        NPUTorchairModelRunner.decode_token_per_req = 1
        NPUTorchairModelRunner.max_num_tokens = 10 # 这行在实际测试中可能被覆盖

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
