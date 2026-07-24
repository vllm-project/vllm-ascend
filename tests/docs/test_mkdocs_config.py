from __future__ import annotations

from pathlib import Path

import markdown  # type: ignore[import-untyped]
import pytest
from mkdocs.config import load_config  # type: ignore[import-not-found]

REPO_ROOT = Path(__file__).resolve().parents[2]
MKDOCS_CONFIG = REPO_ROOT / "mkdocs.yml"


@pytest.mark.parametrize(
    ("smart_enable", "source", "expected"),
    [
        ("underscore", "通常**按需**发布", "<p>通常<strong>按需</strong>发布</p>"),
        ("underscore", "配置 foo_bar_baz 即可", "<p>配置 foo_bar_baz 即可</p>"),
        ("all", "Typically **on demand** released", "<p>Typically <strong>on demand</strong> released</p>"),
    ],
)
def test_betterem_configuration_renders_emphasis(
    monkeypatch: pytest.MonkeyPatch,
    smart_enable: str,
    source: str,
    expected: str,
) -> None:
    monkeypatch.setenv("DOCS_BETTEREM_SMART_ENABLE", smart_enable)
    config = load_config(config_file=str(MKDOCS_CONFIG))
    betterem_config = config["mdx_configs"]["pymdownx.betterem"]

    rendered = markdown.markdown(
        source,
        extensions=["pymdownx.betterem"],
        extension_configs={"pymdownx.betterem": betterem_config},
    )

    assert betterem_config["smart_enable"] == smart_enable
    assert rendered == expected
