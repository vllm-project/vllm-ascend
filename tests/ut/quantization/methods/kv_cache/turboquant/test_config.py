# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.quantization.methods.kv_cache.turboquant import config
from vllm_ascend.utils import AscendDeviceType


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setattr(config, "get_ascend_device_type", lambda: AscendDeviceType.A3)
    monkeypatch.setattr(config.ctypes, "CDLL", lambda _: SimpleNamespace(NnopbaseSupportTensorV2=lambda: True))
    return SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="turboquant_4bit_nc"),
        attention_config=SimpleNamespace(indexer_kv_dtype="int8"),
        use_v2_model_runner=True,
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_text_config=SimpleNamespace(
                model_type="deepseek_v4",
                head_dim=512,
                qk_rope_head_dim=64,
                index_topk=512,
                num_attention_heads=64,
            ),
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        additional_config={},
        kv_transfer_config=None,
    )


def test_valid_runner_v2_configuration(runtime):
    config.validate_turboquant(runtime)
    assert runtime.cache_config.cache_dtype == "turboquant_4bit_nc"


def test_hf_config_fallback(runtime):
    runtime.model_config.hf_config = runtime.model_config.hf_text_config
    del runtime.model_config.hf_text_config

    config.validate_turboquant(runtime)


def test_missing_hf_config_fails_with_validation_error(runtime):
    del runtime.model_config.hf_text_config

    with pytest.raises(ValueError, match="requires DeepSeek V4"):
        config.validate_turboquant(runtime)


@pytest.mark.parametrize(
    "path,value,message",
    [
        ("use_v2_model_runner", False, "VLLM_USE_V2_MODEL_RUNNER"),
        ("model_config.dtype", torch.float16, "BF16"),
        ("model_config.hf_text_config.model_type", "deepseek_v3", "requires DeepSeek V4"),
        ("model_config.hf_text_config.head_dim", 256, "head_dim=512"),
        ("model_config.hf_text_config.index_topk", 256, "index_topk"),
        ("attention_config.indexer_kv_dtype", "auto", "indexer_kv_dtype='int8'"),
        ("parallel_config.tensor_parallel_size", 64, "multiple of 4"),
        ("parallel_config.decode_context_parallel_size", 2, "context parallelism"),
        ("parallel_config.prefill_context_parallel_size", 2, "context parallelism"),
        ("kv_transfer_config", object(), "KV transfer"),
    ],
)
def test_invalid_configuration_fails_early(runtime, path, value, message):
    owner = runtime
    parts = path.split(".")
    for part in parts[:-1]:
        owner = getattr(owner, part)
    setattr(owner, parts[-1], value)
    with pytest.raises(ValueError, match=message):
        config.validate_turboquant(runtime)


def test_a5_is_rejected(runtime, monkeypatch):
    monkeypatch.setattr(config, "get_ascend_device_type", lambda: AscendDeviceType.A5)
    with pytest.raises(ValueError, match="A2/A3"):
        config.validate_turboquant(runtime)


def test_legacy_opbase_cannot_silently_remove_compression(runtime, monkeypatch):
    monkeypatch.setattr(config.ctypes, "CDLL", lambda _: SimpleNamespace())
    with pytest.raises(ValueError, match="NnopbaseSupportTensorV2"):
        config.validate_turboquant(runtime)


@pytest.mark.parametrize("cache_dtype", ["auto", "bfloat16", "bf16", "float16", "fp8", "fp8_e4m3"])
def test_non_turboquant_configuration_is_unchanged(cache_dtype, monkeypatch):
    # Ordinary configurations must not require DS V4, Runner V2, or new CANN APIs.
    runtime = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype=cache_dtype))

    def unexpected_probe(*args):
        pytest.fail("Non-TurboQuant configuration probed TurboQuant capabilities")

    monkeypatch.setattr(config, "get_ascend_device_type", unexpected_probe)
    monkeypatch.setattr(config.ctypes, "CDLL", unexpected_probe)
    config.validate_turboquant(runtime)
    assert vars(runtime) == {"cache_config": SimpleNamespace(cache_dtype=cache_dtype)}
