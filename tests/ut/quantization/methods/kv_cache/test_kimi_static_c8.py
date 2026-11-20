from types import SimpleNamespace as Ns

import pytest
from vllm.config.compilation import CUDAGraphMode as Mode

from vllm_ascend.ascend_config import AscendConfig


@pytest.fixture
def configuration():
    config = Ns(
        model_config=Ns(
            hf_config=Ns(model_type="kimi_linear", kv_lora_rank=512, qk_rope_head_dim=64, mla_use_output_gate=False)
        ),
        mla_cache_mode="C8",
        compilation_config=Ns(cudagraph_mode=Mode.NONE),
        parallel_config=Ns(
            prefill_context_parallel_size=1, decode_context_parallel_size=8, cp_kv_cache_interleave_size=1664
        ),
        kv_transfer_config=Ns(is_kv_producer=False, kv_connector="MooncakePullConnector"),
        cache_config=Ns(block_size=1664),
        use_v2_model_runner=True,
        speculative_config=None,
    )
    config.model_config.hf_text_config = config.model_config.hf_config
    return config


@pytest.mark.parametrize("value,expected", [("default", "C16"), ("c8", "C8"), (" C16 ", "C16")])
def test_configuration_mode_normalization_and_c16_passthrough(value, expected):
    kwargs = {} if value == "default" else {"mla_cache_mode": value}
    options = AscendConfig(sparse_kv_offload_config=Ns(enabled=False), **kwargs)
    assert options.mla_cache_mode == expected
    if expected == "C16":
        options._validate_mla_cache_mode(None)  # Existing models must not be inspected.


@pytest.mark.parametrize("value", ["INT8", "FP8", "", None, True, 8])
def test_configuration_rejects_invalid_mode(value):
    with pytest.raises(ValueError, match="mla_cache_mode"):
        AscendConfig(mla_cache_mode=value, sparse_kv_offload_config=Ns(enabled=False))


@pytest.mark.parametrize(
    "owner,field,value,error",
    [
        (None, None, None, None),
        ("root", "speculative_config", Ns(method="dspark"), None),
        ("model", "model_type", "deepseek_v3", "requires Kimi K3"),
        ("model", "kv_lora_rank", 256, "requires Kimi K3"),
        ("model", "qk_rope_head_dim", 0, "requires Kimi K3"),
        ("root", "use_v2_model_runner", False, "RunnerV2"),
        ("root", "speculative_config", Ns(method="eagle"), "DSpark"),
        ("parallel", "prefill_context_parallel_size", 2, "prefill_context_parallel"),
        ("parallel", "decode_context_parallel_size", 4, "decode_context_parallel"),
        ("parallel", "cp_kv_cache_interleave_size", 128, "interleave"),
        ("transfer", "is_kv_producer", True, "KV producer"),
        ("transfer", "kv_connector", "MooncakeConnectorV1", "MooncakeConnectorV2"),
        ("cache", "block_size", 128, "block_size=1664"),
    ],
)
def test_configuration_checks_model_runner_pd_and_parallel_boundaries(configuration, owner, field, value, error):
    if owner:
        target = {
            "root": configuration,
            "model": configuration.model_config.hf_text_config,
            "parallel": configuration.parallel_config,
            "transfer": configuration.kv_transfer_config,
            "cache": configuration.cache_config,
        }[owner]
        setattr(target, field, value)
    options = AscendConfig(mla_cache_mode="C8", sparse_kv_offload_config=Ns(enabled=False))
    if error:
        with pytest.raises(ValueError, match=error):
            options._validate_mla_cache_mode(configuration)
    else:
        before = vars(configuration.model_config.hf_text_config).copy()
        options._validate_mla_cache_mode(configuration)
        assert vars(configuration.model_config.hf_text_config) == before
