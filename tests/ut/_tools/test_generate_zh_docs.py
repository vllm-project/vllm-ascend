from tools.generate_zh_docs import apply_translations


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
