from unittest.mock import MagicMock, Mock

import pytest
import torch
from pytest_mock import MockerFixture
from vllm.config import VllmConfig

from tests.ut.base import PytestBase
from vllm_ascend.torchair.torchair_model_runner import NPUTorchairModelRunner


class TestNPUTorchairModelRunner(PytestBase):

    @pytest.fixture
    def setup_npu_torchair_model_runner(self, mocker: MockerFixture):
        mocker.patch.object(NPUTorchairModelRunner, "__init__",
                            lambda self, *args, **kwargs: None)
        runner = NPUTorchairModelRunner(Mock(), Mock())

        runner.device = torch.device("cpu")
        runner.vllm_config = MagicMock(spec=VllmConfig)

        runner.speculative_config = MagicMock(
            method="deepseek_mtp",
            num_speculative_tokens=4,
            disable_padded_drafter_batch=False)

        runner.ascend_config = MagicMock(enable_shared_expert_dp=False,
                                         torchair_graph_config=MagicMock(
                                             use_cached_graph=True,
                                             graph_batch_sizes=[1, 2, 4]))

        runner.decode_token_per_req = 2
        runner.is_kv_consumer = True
        runner.max_num_reqs = 100

        runner.model_config = MagicMock(hf_config=MagicMock(index_topk=2))
        runner.attn_backend = MagicMock(get_builder_cls=lambda: Mock())

        return runner

    def test_init(self, mocker: MockerFixture,
                  setup_npu_torchair_model_runner):
        runner = setup_npu_torchair_model_runner
        assert isinstance(runner, NPUTorchairModelRunner)
