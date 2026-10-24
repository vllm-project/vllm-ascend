# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm_ascend.ascend_config import AscendConfig, SparseKVOffloadConfig
from vllm_ascend.models.glm5next.recoverssm_config import use_glm_recoverssm, validate_glm_recoverssm


def make_config():
    return SimpleNamespace(
        additional_config={"kda_recoverssm": True},
        speculative_config=SimpleNamespace(),
        cache_config=SimpleNamespace(use_kda_recoverssm=False, mamba_cache_mode="none", enable_prefix_caching=False),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
        kv_transfer_config=None,
    )


def test_ascend_configuration_registers_default_off_recoverssm_switch():
    assert AscendConfig(sparse_kv_offload_config=SparseKVOffloadConfig()).kda_recoverssm is False
    config = AscendConfig(sparse_kv_offload_config=SparseKVOffloadConfig(), kda_recoverssm=True)
    assert config.kda_recoverssm is True


def test_default_does_not_change_native_cache_contract():
    config = make_config()
    config.additional_config = {}
    assert not use_glm_recoverssm(config)
    assert not config.cache_config.use_kda_recoverssm


@pytest.mark.parametrize("connector", [None, "MooncakeConnectorV2", "MooncakePullConnector"])
def test_configuration_validation_leaves_worker_activation_to_state_initialization(connector):
    config = make_config()
    if connector is not None:
        config.kv_transfer_config = SimpleNamespace(kv_connector=connector)
    assert use_glm_recoverssm(config)
    assert not config.cache_config.use_kda_recoverssm
    validate_glm_recoverssm(config)
    assert not config.cache_config.use_kda_recoverssm


def test_initialization_requires_speculative_decoding():
    config = make_config()
    config.speculative_config = None
    ascend_config = AscendConfig(sparse_kv_offload_config=SparseKVOffloadConfig(), kda_recoverssm=True)
    with pytest.raises(ValueError, match="requires speculative decoding"):
        ascend_config.derive_and_validate(config)
    assert not config.cache_config.use_kda_recoverssm


@pytest.mark.parametrize(
    "section,field,value,message",
    [
        ("cache_config", "mamba_cache_mode", "align", "Mamba cache mode none"),
        ("cache_config", "enable_prefix_caching", True, "prefix caching disabled"),
        ("parallel_config", "pipeline_parallel_size", 2, "PP=1 and PCP=1"),
        ("parallel_config", "prefill_context_parallel_size", 2, "PP=1 and PCP=1"),
        ("parallel_config", "decode_context_parallel_size", 2, "DCP=1"),
        (None, "kv_transfer_config", SimpleNamespace(kv_connector="AscendStoreConnector"), "requires Mooncake"),
    ],
)
def test_initialization_rejects_unsupported_execution(section, field, value, message):
    config = make_config()
    setattr(getattr(config, section) if section else config, field, value)
    ascend_config = AscendConfig(sparse_kv_offload_config=SparseKVOffloadConfig(), kda_recoverssm=True)
    with pytest.raises(ValueError, match=message):
        ascend_config.derive_and_validate(config)
    assert not config.cache_config.use_kda_recoverssm
