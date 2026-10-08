# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import AutoConfig


def pytest_addoption(parser):
    parser.addoption("--dcp-q-replicate-model", help="Dense MLA checkpoint fitting on four NPUs")
    parser.addoption("--dcp-q-replicate-sparse-model", help="RoPE SFA/DSA checkpoint with Q-LoRA fitting on four NPUs")


@pytest.fixture
def dcp_q_replicate_model(request, model_kind):
    option = "--dcp-q-replicate-sparse-model" if model_kind == "sparse" else "--dcp-q-replicate-model"
    model = request.config.getoption(option)
    if model is None:
        pytest.skip(f"Supply {option} with a supported checkpoint")
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    if model_kind == "sparse":
        assert getattr(config, "index_topk", 0) > 0, "A sparse SFA/DSA checkpoint is required"
        assert getattr(config, "q_lora_rank", None) is not None, "Sparse native preprocessing requires Q-LoRA"
        assert getattr(config, "qk_rope_head_dim", 0) > 0, "NoPE sparse backends are outside this test's scope"
    else:
        assert not hasattr(config, "index_topk"), "A dense MLA checkpoint is required"
    return model
