# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_ascend.patch.platform import patch_reasoning_parser_plugin as patch


def _args(plugin: str | None = "/tmp/custom.py"):
    return SimpleNamespace(reasoning_parser_plugin=plugin)


def test_imports_plugin_before_headless_engine_config(monkeypatch):
    calls = []

    monkeypatch.setattr(
        patch.ReasoningParserManager,
        "import_reasoning_parser",
        lambda path: calls.append(path),
    )
    monkeypatch.setattr(
        patch,
        "_ORIGINAL_RUN_HEADLESS",
        lambda args: calls.append("run") or "result",
    )

    result = patch.serve.run_headless(_args())

    assert calls == ["/tmp/custom.py", "run"]
    assert result == "result"


@pytest.mark.parametrize(
    "args",
    [_args(plugin=None), _args(plugin="x")],
)
def test_skips_incomplete_plugin_config(monkeypatch, args):
    monkeypatch.setattr(
        patch.ReasoningParserManager,
        "import_reasoning_parser",
        pytest.fail,
    )

    patch._import_reasoning_parser_plugin(args)


def test_rejects_vllm_run_headless_contract_drift():
    def changed_run_headless(args, extra):
        del args, extra

    with pytest.raises(RuntimeError, match="run_headless signature changed"):
        patch._validate_run_headless_contract(changed_run_headless)


def test_patches_only_headless_entrypoint():
    assert patch.serve.run_headless is patch._patched_run_headless
