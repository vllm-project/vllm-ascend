# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import AutoConfig


def pytest_addoption(parser):
    parser.addoption(
        "--dcp-qrep-sparse-model",
        default=None,
        help="Unquantized RoPE SFA/DSA checkpoint with Q-LoRA, fitting on four NPUs",
    )
    parser.addoption(
        "--dcp-qrep-sparse-mtp-model",
        default=None,
        help="Unquantized RoPE SFA/DSA checkpoint including MTP weights, fitting on four NPUs",
    )
    parser.addoption(
        "--dcp-qrep-model",
        default=None,
        help="Unquantized dense DeepSeek MLA checkpoint (direct Q or Q-LoRA), fitting on four NPUs",
    )

    parser.addoption(
        "--dcp-qrep-mtp-model",
        default=None,
        help="Unquantized dense MLA checkpoint with MTP weights, fitting on four NPUs",
    )


@pytest.fixture
def dcp_qrep_mtp_model(request):
    model = request.config.getoption("--dcp-qrep-mtp-model")
    if model is None:
        pytest.skip("Supply --dcp-qrep-mtp-model with an unquantized dense MLA MTP checkpoint")
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    assert not hasattr(config, "index_topk"), "Sparse MLA is outside this test's scope"
    assert not getattr(config, "quantization_config", None), "The checkpoint must be unquantized"
    assert getattr(config, "num_nextn_predict_layers", 0) > 0, "The checkpoint must include MTP layers"
    return model


@pytest.fixture
def dcp_qrep_model(request):
    model = request.config.getoption("--dcp-qrep-model")
    if model is None:
        pytest.skip("Supply --dcp-qrep-model with a supported unquantized dense MLA checkpoint")
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    assert not hasattr(config, "index_topk"), "Sparse MLA is outside this test's scope"
    assert not getattr(config, "quantization_config", None), "The checkpoint must be unquantized"
    return model


def _sparse_qrep_checkpoint(request, option, *, mtp=False):
    model = request.config.getoption(option)
    if model is None:
        pytest.skip(f"Supply {option} with a supported unquantized sparse checkpoint")
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    assert getattr(config, "index_topk", 0) > 0, "A sparse SFA/DSA checkpoint is required"
    assert getattr(config, "q_lora_rank", None) is not None, "Sparse native preprocessing requires Q-LoRA"
    assert getattr(config, "qk_rope_head_dim", 0) > 0, "NoPE sparse backends are outside this test's scope"
    assert not getattr(config, "quantization_config", None), "The checkpoint must be unquantized"
    if mtp:
        assert getattr(config, "num_nextn_predict_layers", 0) > 0, "The checkpoint must include MTP layers"
    return model


@pytest.fixture
def dcp_qrep_sparse_model(request):
    return _sparse_qrep_checkpoint(request, "--dcp-qrep-sparse-model")


@pytest.fixture
def dcp_qrep_sparse_mtp_model(request):
    return _sparse_qrep_checkpoint(request, "--dcp-qrep-sparse-mtp-model", mtp=True)
