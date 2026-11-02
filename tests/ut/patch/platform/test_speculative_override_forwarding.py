# SPDX-License-Identifier: Apache-2.0
"""Keep upstream override arguments intact without initializing an accelerator."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("style", ["legacy", "positional", "keyword"])
def test_normalizer_forwards_upstream_mtp_override(style):
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/platform/patch_speculative_config.py"
    node = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_normalize_legacy_qwen3_dspark_config"
    )
    calls = []

    def upstream(config, *args, **kwargs):
        calls.append((config, args, kwargs))
        config.overrides = args[0] if args else kwargs.get("transformers_mtp_overrides")
        return config

    namespace = {
        "PretrainedConfig": object,
        "_orig_hf_config_override": upstream,
        "_normalize_kimi_dflash_rope": lambda config: None,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    config = SimpleNamespace(model_type="unrelated", architectures=[])
    override = {"model_type": "draft", "num_hidden_layers": 1}
    args = (override,) if style == "positional" else ()
    kwargs = {"transformers_mtp_overrides": override} if style == "keyword" else {}
    assert namespace[node.name](config, *args, **kwargs) is config
    assert calls == [(config, args, kwargs)]
    assert config.overrides is (None if style == "legacy" else override)
