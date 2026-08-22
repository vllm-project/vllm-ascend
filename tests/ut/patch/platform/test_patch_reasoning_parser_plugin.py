# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from vllm.config import VllmConfig
from vllm.reasoning import ReasoningParserManager

from vllm_ascend.patch.platform import patch_reasoning_parser_plugin


def _config(parser_name: str = "custom", plugin_path: str = "/tmp/custom.py"):
    return SimpleNamespace(
        reasoning_config=SimpleNamespace(reasoning_parser=parser_name),
        structured_outputs_config=SimpleNamespace(
            reasoning_parser_plugin=plugin_path,
        ),
    )


def test_imports_reasoning_parser_plugin_before_vllm_config_post_init(monkeypatch):
    calls = []
    config = _config()
    registered = []

    def import_plugin(path):
        calls.append(("import", path))
        registered.append("custom")

    def original_post_init(self):
        assert "custom" in registered
        calls.append(("post_init", self))

    monkeypatch.setattr(
        ReasoningParserManager,
        "list_registered",
        lambda: registered,
    )
    monkeypatch.setattr(
        ReasoningParserManager,
        "import_reasoning_parser",
        import_plugin,
    )
    monkeypatch.setattr(
        patch_reasoning_parser_plugin,
        "_ORIGINAL_VLLM_CONFIG_POST_INIT",
        original_post_init,
    )

    VllmConfig.__post_init__(config)

    assert calls == [
        ("import", "/tmp/custom.py"),
        ("post_init", config),
    ]


def test_skips_plugin_import_when_reasoning_parser_is_registered(monkeypatch):
    import_plugin = pytest.fail
    monkeypatch.setattr(
        ReasoningParserManager,
        "list_registered",
        lambda: ["custom"],
    )
    monkeypatch.setattr(
        ReasoningParserManager,
        "import_reasoning_parser",
        import_plugin,
    )

    patch_reasoning_parser_plugin._import_reasoning_parser_plugin(_config())


@pytest.mark.parametrize(
    ("parser_name", "plugin_path"),
    [("", "/tmp/custom.py"), ("custom", "")],
)
def test_skips_plugin_import_without_complete_config(
    monkeypatch,
    parser_name,
    plugin_path,
):
    monkeypatch.setattr(
        ReasoningParserManager,
        "import_reasoning_parser",
        pytest.fail,
    )

    patch_reasoning_parser_plugin._import_reasoning_parser_plugin(
        _config(parser_name, plugin_path),
    )


def test_rejects_vllm_post_init_contract_drift():
    def changed_post_init(self, extra):
        pass

    with pytest.raises(RuntimeError, match="VllmConfig.__post_init__ signature changed"):
        patch_reasoning_parser_plugin._validate_post_init_contract(
            changed_post_init,
        )
