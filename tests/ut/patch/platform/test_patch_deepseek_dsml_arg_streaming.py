# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for ``patch_deepseek_dsml_arg_streaming``.

The patch module is executed against stubbed ``vllm`` modules, so these tests
run without a vLLM or NPU build.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

PATCH_PATH = (
    Path(__file__).resolve().parents[4] / "vllm_ascend" / "patch" / "platform" / "patch_deepseek_dsml_arg_streaming.py"
)

GATE = frozenset(">")


class FakeParserEngine:
    """Stand-in for ``vllm.parser.engine.parser_engine.ParserEngine``."""

    def _compute_arg_delta(self, idx: int, raw_delta: str) -> str | None:
        self.calls.append((idx, raw_delta, self._arg_structural_chars))
        # Mirrors upstream: no arguments delta while the structural gate is on.
        return "delta" if self._arg_structural_chars is None else None


def _make_engine(name: str, args: str) -> FakeParserEngine:
    engine = FakeParserEngine()
    engine.parser_engine_config = SimpleNamespace(name=name)
    engine._arg_structural_chars = GATE
    engine._tool_slots = [SimpleNamespace(args=args)]
    engine.calls = []
    return engine


def load_patch():
    """Execute the patch module with stubbed ``vllm`` modules."""
    vllm = ModuleType("vllm")
    logger_module = ModuleType("vllm.logger")
    parser_module = ModuleType("vllm.parser")
    engine_module = ModuleType("vllm.parser.engine")
    parser_engine_module = ModuleType("vllm.parser.engine.parser_engine")

    logger_module.logger = Mock()
    parser_engine_module.ParserEngine = FakeParserEngine
    parser_module.engine = engine_module
    engine_module.parser_engine = parser_engine_module
    vllm.logger = logger_module
    vllm.parser = parser_module

    stubs = {
        "vllm": vllm,
        "vllm.logger": logger_module,
        "vllm.parser": parser_module,
        "vllm.parser.engine": engine_module,
        "vllm.parser.engine.parser_engine": parser_engine_module,
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("_test_dsml_streaming_patch", PATCH_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def test_detects_open_parameter_value():
    module = load_patch()
    open_v4 = '<｜DSML｜parameter name="content" string="true">正文'
    assert module._in_open_parameter_value(open_v4) is True
    assert module._in_open_parameter_value(open_v4 + "</｜DSML｜parameter>") is False

    open_v41 = '<｜DSML｜ parameter name="content" string="true">正文'
    assert module._in_open_parameter_value(open_v41) is True
    assert module._in_open_parameter_value(open_v41 + "</｜DSML｜ parameter>") is False

    assert module._in_open_parameter_value("<｜DSML｜tool_calls>") is False


def test_partial_tag_tail_is_detected():
    module = load_patch()
    value = '<｜DSML｜parameter name="content" string="true">正文'
    assert module._ends_with_partial_tag(value + "<") is True
    assert module._ends_with_partial_tag(value + "正文</｜DSML｜par") is True
    assert module._ends_with_partial_tag(value + "正文</｜DSML｜ parameter") is True
    assert module._ends_with_partial_tag(value + "正文") is False


def test_value_body_delta_bypasses_gate_and_restores_it():
    load_patch()
    args = '<｜DSML｜parameter name="content" string="true">正文'
    engine = _make_engine("deepseek_v4", args)

    assert engine._compute_arg_delta(0, "正文正文") == "delta"

    _, _, structural_during_call = engine.calls[-1]
    assert structural_during_call is None
    assert engine._arg_structural_chars == GATE


def test_v41_spaced_dialect_bypasses_gate():
    load_patch()
    args = '<｜DSML｜ parameter name="content" string="true">正文'
    engine = _make_engine("deepseek_v41", args)

    assert engine._compute_arg_delta(0, "正文正文") == "delta"
    assert engine.calls[-1][2] is None


def test_other_parsers_keep_the_gate():
    load_patch()
    args = '<｜DSML｜parameter name="content" string="true">正文'
    engine = _make_engine("qwen3", args)

    assert engine._compute_arg_delta(0, "正文正文") is None
    assert engine.calls[-1][2] == GATE


def test_delta_with_structural_char_keeps_upstream_path():
    load_patch()
    args = '<｜DSML｜parameter name="content" string="true">正文'
    engine = _make_engine("deepseek_v4", args)

    assert engine._compute_arg_delta(0, "正文>") is None
    assert engine.calls[-1][2] == GATE


def test_partial_tag_delta_keeps_upstream_path():
    load_patch()
    args = '<｜DSML｜parameter name="content" string="true">正文<'
    engine = _make_engine("deepseek_v4", args)

    assert engine._compute_arg_delta(0, "正文") is None
    assert engine.calls[-1][2] == GATE


def test_closed_parameter_keeps_upstream_path():
    load_patch()
    args = '<｜DSML｜parameter name="content" string="true">正文</｜DSML｜parameter>'
    engine = _make_engine("deepseek_v4", args)

    assert engine._compute_arg_delta(0, "正文") is None
    assert engine.calls[-1][2] == GATE


def test_patch_is_idempotent():
    load_patch()
    patched = FakeParserEngine._compute_arg_delta
    load_patch()
    assert FakeParserEngine._compute_arg_delta is patched
