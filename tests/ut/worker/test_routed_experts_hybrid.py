from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.ut.base import TestBase
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.worker import NPUWorker


class FakeAttentionSpec:

    def __init__(self, block_size):
        self.block_size = block_size


class FakeMambaSpec:

    def __init__(self, block_size):
        self.block_size = block_size


class TestRoutedExpertsHybrid(TestBase):

    def test_get_attention_kv_cache_gid_prefers_attention_group(self):
        runner = object.__new__(NPUModelRunner)
        runner.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=FakeMambaSpec(block_size=262144)),
                SimpleNamespace(kv_cache_spec=FakeAttentionSpec(block_size=128)),
            ]
        )

        with patch("vllm_ascend.worker.model_runner_v1.AttentionSpec", FakeAttentionSpec):
            self.assertEqual(runner._get_attention_kv_cache_gid(), 1)

    def test_npu_model_runner_reuses_parent_init_routed_experts_capturer(self):
        self.assertNotIn("init_routed_experts_capturer", NPUModelRunner.__dict__)

    def test_worker_initializes_routed_experts_after_kv_cache(self):
        worker = object.__new__(NPUWorker)
        worker.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(enable_sleep_mode=False)
        )
        worker.model_config = SimpleNamespace(enable_return_routed_experts=True)
        worker.model_runner = MagicMock()

        with patch("vllm_ascend.worker.worker.ensure_kv_transfer_initialized"), patch.object(
            GPUModelRunner,
            "init_routed_experts_capturer",
        ) as mock_parent_init:
            worker.initialize_from_config(kv_cache_config=MagicMock())

        worker.model_runner.initialize_kv_cache.assert_called_once()
        mock_parent_init.assert_called_once_with(worker.model_runner)
