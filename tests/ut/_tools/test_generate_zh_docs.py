from tools import generate_zh_docs
from tools.generate_zh_docs import apply_translations


def test_reset_output_dir_removes_stale_generated_files(tmp_path, monkeypatch):
    output_dir = tmp_path / "zh"
    stale_file = output_dir / "_snippets" / "stale.md"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("stale", encoding="utf-8")
    monkeypatch.setattr(generate_zh_docs, "ZH_DIR", output_dir)

    generate_zh_docs._reset_output_dir()

    assert output_dir.is_dir()
    assert not stale_file.exists()


def test_preserves_translated_list_structure_with_multi_backtick_code():
    source = """The current pipeline is:

1. `BlockScanner` parses ``model-code`` fences and accepts only options listed
   in `MODEL_CODE_OPTION_NAMES`.
2. `YamlLoader` loads `test_case_path`.
"""
    translations = {
        "The current pipeline is:": "当前的流程如下：",
        (
            "1. `BlockScanner` parses ``model-code`` fences and accepts only options listed\n"
            "   in `MODEL_CODE_OPTION_NAMES`.\n"
            "2. `YamlLoader` loads `test_case_path`."
        ): (
            "1. `BlockScanner` 解析 ``model-code`` 围栏，仅接受 `MODEL_CODE_OPTION_NAMES` 中列出的选项。\n"
            "2. `YamlLoader` 加载 `test_case_path`。"
        ),
    }

    result = apply_translations(source, translations)

    assert (
        result
        == """当前的流程如下：

1. `BlockScanner` 解析 ``model-code`` 围栏，仅接受 `MODEL_CODE_OPTION_NAMES` 中列出的选项。
2. `YamlLoader` 加载 `test_case_path`。
"""
    )


def test_preserves_code_order_from_translated_paragraph():
    source = "`get_converter()` looks up `block.converter_tag` from `build_default_converters()`."
    translation = "`get_converter()` 从 `build_default_converters()` 中查找 `block.converter_tag`。"

    assert apply_translations(source, {source: translation}) == translation


def test_short_translations_do_not_modify_protected_spans():
    source = "Use mode with `mode`, [mode](https://example.com/mode), and ``mode``."

    result = apply_translations(source, {"mode": "模式"})

    assert result == "Use 模式 with `mode`, [模式](https://example.com/mode), and ``mode``."


def test_matches_translated_prose_after_source_line_reflow():
    source = "Phase 3 removes the direct environment-variable read. Details in [Phased rollout](#phased-rollout)."
    msgid = "Phase 3 removes the direct environment-variable read. Details in\n[Phased rollout](#phased-rollout)."
    translation = "阶段 3 移除了直接环境变量读取。详情请参见[分阶段推出](#phased-rollout)。"

    assert apply_translations(source, {msgid: translation}) == translation


def test_copy_untranslated_markdown_refreshes_existing_file(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    locale_dir = source_dir / "locale" / "zh_CN" / "LC_MESSAGES"
    zh_dir = source_dir / "zh"
    source_file = source_dir / "getting_started" / "quick_start" / "atlas-a2.inc.md"
    translated_source = source_dir / "translated.md"

    source_file.parent.mkdir(parents=True)
    locale_dir.mkdir(parents=True)
    zh_dir.mkdir(parents=True)
    source_file.write_text("当前中文片段\n")
    translated_source.write_text("English source\n")
    (locale_dir / "translated.po").write_text("")

    copied_file = zh_dir / source_file.relative_to(source_dir)
    copied_file.parent.mkdir(parents=True)
    copied_file.write_text("旧内容\n")
    translated_output = zh_dir / "translated.md"
    translated_output.write_text("已有翻译\n")

    monkeypatch.setattr(generate_zh_docs, "SOURCE_DIR", source_dir)
    monkeypatch.setattr(generate_zh_docs, "LOCALE_DIR", locale_dir)
    monkeypatch.setattr(generate_zh_docs, "ZH_DIR", zh_dir)

    generate_zh_docs.copy_untranslated_markdown()

    assert copied_file.read_text() == "当前中文片段\n"
    assert translated_output.read_text() == "已有翻译\n"


def test_copy_assets_copies_javascripts_and_removes_stale_files(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    zh_dir = source_dir / "zh"
    javascript_dir = source_dir / "javascripts"
    javascript_dir.mkdir(parents=True)
    (javascript_dir / "mathjax.js").write_text("mathjax", encoding="utf-8")
    (javascript_dir / "tabbed-toc.js").write_text("tabbed-toc", encoding="utf-8")

    copied_javascript_dir = zh_dir / "javascripts"
    copied_javascript_dir.mkdir(parents=True)
    (copied_javascript_dir / "stale.js").write_text("stale", encoding="utf-8")

    monkeypatch.setattr(generate_zh_docs, "SOURCE_DIR", source_dir)
    monkeypatch.setattr(generate_zh_docs, "ZH_DIR", zh_dir)

    generate_zh_docs.copy_assets()

    assert (copied_javascript_dir / "mathjax.js").read_text(encoding="utf-8") == "mathjax"
    assert (copied_javascript_dir / "tabbed-toc.js").read_text(encoding="utf-8") == "tabbed-toc"
    assert not (copied_javascript_dir / "stale.js").exists()
