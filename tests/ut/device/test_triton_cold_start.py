# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Exercise real vLLM ModelConfig and registry with an empty model-info cache."""

import inspect
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize(
    "architecture",
    [
        "Qwen4ExpForCausalLM",
        "Step5ForCausalLM",
        "Glm5NextForCausalLM",
        "DeepseekV41ForCausalLM",
        "MiniMaxM3SparseForCausalLM",
    ],
)
def test_cold_model_config_rejects_before_lazy_registry_import(monkeypatch, tmp_path, architecture):
    from vllm.config import ModelConfig
    from vllm.model_executor.models import registry as registry_module

    from vllm_ascend.device import triton_gate
    from vllm_ascend.models import register_model
    from vllm_ascend.platform import NPUPlatform

    # Start from an unwrapped registry, with both memory and disk caches cold.
    registry_type = type(registry_module.ModelRegistry)
    monkeypatch.setattr(registry_type, "inspect_model_cls", inspect.unwrap(registry_type.inspect_model_cls))
    monkeypatch.setattr(registry_module.ModelRegistry, "models", dict(registry_module.ModelRegistry.models))
    registry_module._try_inspect_model_cls.cache_clear()
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(
        triton_gate,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(supports=lambda capability: False),
    )
    lazy_inspect = Mock(side_effect=AssertionError("Cold startup reached a model-module import"))
    monkeypatch.setattr(registry_module._LazyRegisteredModel, "inspect_model_cls", lazy_inspect)
    later_hook = Mock(side_effect=AssertionError("The late config hook must not be needed"))
    monkeypatch.setattr(NPUPlatform, "check_and_update_config", later_hook)

    # Model registration is a general plugin entry point, also used by the
    # registry subprocess. Do not call the later platform config hook here.
    register_model()
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": [architecture],
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "vocab_size": 128,
                "max_position_embeddings": 128,
            }
        )
    )

    with pytest.raises(ValueError, match=rf"requires Triton-Ascend.*{architecture}.*native 310P implementation"):
        ModelConfig(model=str(tmp_path), tokenizer=str(tmp_path), dtype="float16", max_model_len=64)

    lazy_inspect.assert_not_called()
    later_hook.assert_not_called()
    assert not (tmp_path / "cache" / "modelinfos").exists()
