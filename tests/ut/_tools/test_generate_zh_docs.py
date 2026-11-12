import pytest

from tools import generate_zh_docs


@pytest.mark.parametrize(
    "source, translations, expected",
    [
        pytest.param(
            "Steps:\n\n1. Read ``config``\n   carefully.\n2. Load `path`.\n",
            {
                "Steps:": "步骤：",
                "1. Read ``config``\n   carefully.\n2. Load `path`.": "1. 仔细读取 ``config``。\n2. 加载 `path`。",
            },
            "步骤：\n\n1. 仔细读取 ``config``。\n2. 加载 `path`。\n",
            id="lists-and-code-spans",
        ),
        pytest.param(
            "`get()` finds `name` in `handlers`.",
            {"`get()` finds `name` in `handlers`.": "`get()` 从 `handlers` 中查找 `name`。"},
            "`get()` 从 `handlers` 中查找 `name`。",
            id="translated-code-order",
        ),
        pytest.param(
            "Use mode with `mode`, ``mode``, [mode](https://example.com/mode), and [mode](#mode).",
            {"mode": "模式"},
            "Use 模式 with `mode`, ``mode``, [模式](https://example.com/mode), and [模式](#mode).",
            id="protected-code-and-link-targets",
        ),
        pytest.param(
            "Read the [guide](#guide) for setup.",
            {"Read the [guide](#guide)\nfor setup.": "安装方法请参阅[指南](#guide)。"},
            "安装方法请参阅[指南](#guide)。",
            id="reflowed-prose-with-links",
        ),
    ],
)
def test_translation_preserves_markdown(source, translations, expected):
    assert generate_zh_docs.apply_translations(source, translations) == expected


def test_reflow_respects_code_blocks_and_paragraph_boundaries():
    wrapped = "Read this\nparagraph."
    prose = "Read this paragraph."
    code = f"```text\n{prose}\n```"
    separate_paragraphs = "Read this\n\nparagraph."
    source = "\n\n".join([prose, code, wrapped, separate_paragraphs])

    assert generate_zh_docs.apply_translations(source, {wrapped: "阅读这一段。"}) == "\n\n".join(
        ["阅读这一段。", code, "阅读这一段。", separate_paragraphs]
    )


def test_generation_refreshes_pages_and_assets(tmp_path, monkeypatch):
    pytest.importorskip("polib")
    source_dir = tmp_path / "source"
    locale_dir = source_dir / "locale" / "zh_CN" / "LC_MESSAGES"
    zh_dir = source_dir / "zh"
    inputs = {
        "guide.md": "Hello\n",
        "locale/zh_CN/LC_MESSAGES/guide.po": 'msgid "Hello"\nmsgstr "你好"\n',
        "untranslated.md": "No translation yet\n",
        "locale/zh_CN/LC_MESSAGES/untranslated.po": "",
        "nested/shared.inc.md": "Shared content\n",
        "javascripts/example.js": "script",
        "stylesheets/extra.css": "style",
        "zh/nested/shared.inc.md": "Old content",
        "zh/removed.md": "stale",
        "zh/javascripts/removed.js": "stale",
    }
    for name, content in inputs.items():
        path = source_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(generate_zh_docs, "SOURCE_DIR", source_dir)
    monkeypatch.setattr(generate_zh_docs, "LOCALE_DIR", locale_dir)
    monkeypatch.setattr(generate_zh_docs, "ZH_DIR", zh_dir)

    generate_zh_docs.main()

    output = {
        path.relative_to(zh_dir).as_posix(): path.read_text(encoding="utf-8")
        for path in zh_dir.rglob("*")
        if path.is_file()
    }
    assert output == {
        "guide.md": "你好\n",
        "untranslated.md": "No translation yet\n",
        "nested/shared.inc.md": "Shared content\n",
        "javascripts/example.js": "script",
        "stylesheets/extra.css": "style",
    }
