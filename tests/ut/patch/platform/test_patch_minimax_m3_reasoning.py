# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import runpy
from pathlib import Path

import pytest
from vllm.reasoning.minimax_m3_reasoning_parser import MiniMaxM3ReasoningParser

PATCH_PATH = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/platform/patch_minimax_m3_reasoning.py"
PROMPT = (
    "]~b]system\nWrap reasoning in <mm:think></mm:think> tags or begin after </mm:think>.[e~[\n"
    "]~b]user\nCapital of France?[e~[\n"
    "]~b]ai\n</mm:think>Paris.[e~[\n"
    "]~b]user\nAnd Germany?[e~[\n"
    "]~b]ai\n"
)


class MiniMaxTokenizer:
    """Tiny tokenizer; message boundary ID zero also checks truthiness errors."""

    def __init__(self, message_boundary=True, split_markers=False):
        self.special_tokens = (("]~b]",) if message_boundary else ()) + ("<mm:think>", "</mm:think>")
        self.vocab = {token: i for i, token in enumerate(self.special_tokens)}
        self.split_markers = split_markers

    def get_vocab(self):
        return self.vocab.copy()

    def encode(self, text, **kwargs):
        ids = []
        while text:
            token = text[0]
            for special in self.special_tokens:
                if text.startswith(special) and (special == "]~b]" or not self.split_markers):
                    token = special
                    break
            ids.append(self.vocab.setdefault(token, len(self.vocab)))
            text = text[len(token) :]
        return ids

    def decode(self, ids, **kwargs):
        tokens = {value: key for key, value in self.vocab.items()}
        return "".join(tokens[token_id] for token_id in ids)


@pytest.fixture
def patched_parser():
    # Load only this patch, without unrelated device/platform initialization.
    names = ("_last_message_ids", "is_reasoning_end", "is_reasoning_end_streaming")
    originals = {name: getattr(MiniMaxM3ReasoningParser, name, None) for name in names}
    namespace = runpy.run_path(str(PATCH_PATH))
    try:
        yield MiniMaxM3ReasoningParser, namespace
    finally:
        for name, original in originals.items():
            if original is None:
                delattr(MiniMaxM3ReasoningParser, name)
            else:
                setattr(MiniMaxM3ReasoningParser, name, original)


@pytest.mark.parametrize("split_markers", [False, True])
@pytest.mark.parametrize(
    ("suffix", "expected"),
    [("", False), ("<mm:think>ab", False), ("<mm:think>a</mm:think>", True), ("</mm:think>Berlin", True)],
)
def test_prompt_markers_do_not_end_reasoning(patched_parser, split_markers, suffix, expected):
    parser_cls, _ = patched_parser
    tokenizer = MiniMaxTokenizer(split_markers=split_markers)
    parser = parser_cls(tokenizer)
    assert parser.is_reasoning_end(tokenizer.encode(PROMPT + suffix)) is expected


@pytest.mark.parametrize("split_markers", [False, True])
def test_streaming_end_check_ignores_prompt_markers(patched_parser, split_markers):
    parser_cls, _ = patched_parser
    tokenizer = MiniMaxTokenizer(split_markers=split_markers)
    parser = parser_cls(tokenizer)
    open_ids = tokenizer.encode("<mm:think>ab")
    closed_ids = tokenizer.encode("</mm:think>")
    input_ids = tokenizer.encode(PROMPT) + open_ids
    assert not parser.is_reasoning_end_streaming(input_ids, iter(open_ids[-1:]))
    assert parser.is_reasoning_end_streaming(input_ids + closed_ids, iter(closed_ids))


@pytest.mark.parametrize("message_boundary", [False, True])
@pytest.mark.parametrize(
    ("text", "expected"),
    [("", False), ("<mm:think>", False), ("<mm:think>abc", False), ("</mm:think>", True)],
)
def test_output_only_ids_unchanged(patched_parser, message_boundary, text, expected):
    parser_cls, _ = patched_parser
    tokenizer = MiniMaxTokenizer(message_boundary=message_boundary)
    parser = parser_cls(tokenizer)
    ids = tokenizer.encode(text)
    assert parser.is_reasoning_end(ids) is expected
    assert parser.is_reasoning_end_streaming(ids, ids[-1:]) is expected


def test_missing_boundary_token_preserves_prompt_behavior(patched_parser):
    parser_cls, _ = patched_parser
    tokenizer = MiniMaxTokenizer(message_boundary=False)
    parser = parser_cls(tokenizer)
    assert parser.is_reasoning_end(tokenizer.encode(PROMPT))


def test_continued_message_and_empty_last_message(patched_parser):
    parser_cls, _ = patched_parser
    tokenizer = MiniMaxTokenizer()
    parser = parser_cls(tokenizer)
    assert parser.is_reasoning_end(tokenizer.encode("]~b]ai\n</mm:think>partial answer"))
    ids = tokenizer.encode(PROMPT + "</mm:think>answer[e~[]~b]")
    assert not parser.is_reasoning_end(ids)
    assert not parser.is_reasoning_end_streaming(ids, [])


@pytest.mark.parametrize(
    "attribute", ["_reasoning_ended_streaming", "_reasoning_active_streaming", "_pending_marker_streaming"]
)
def test_streaming_state_takes_precedence(patched_parser, attribute, monkeypatch):
    parser_cls, _ = patched_parser
    tokenizer = MiniMaxTokenizer()
    parser = parser_cls(tokenizer)
    setattr(parser, attribute, True)
    monkeypatch.setattr(parser, "_last_message_ids", lambda _: pytest.fail("State fast paths must not scan token IDs"))
    ids = tokenizer.encode(PROMPT + "</mm:think>")
    assert parser.is_reasoning_end_streaming(ids, []) is (attribute == "_reasoning_ended_streaming")


def test_nonstreaming_extraction_unchanged(patched_parser):
    parser_cls, _ = patched_parser
    parser = parser_cls(MiniMaxTokenizer())
    assert parser.extract_reasoning("<mm:think>plan</mm:think>Berlin", None) == ("plan", "Berlin")
    assert parser.extract_reasoning("</mm:think>Berlin", None) == (None, "Berlin")


@pytest.mark.parametrize(
    ("chunks", "expected_reasoning"),
    [(["<mm:think>", "plan", "</mm:think>", "Berlin"], "plan"), (["</mm:think>", "Berlin"], "")],
)
def test_streaming_extraction_after_prompt_check(patched_parser, chunks, expected_reasoning):
    parser_cls, _ = patched_parser
    tokenizer = MiniMaxTokenizer()
    parser = parser_cls(tokenizer)
    assert not parser.is_reasoning_end(tokenizer.encode(PROMPT))
    previous_text, previous_ids = "", []
    reasoning, content = "", ""
    for chunk in chunks:
        delta_ids = tokenizer.encode(chunk)
        current_text, current_ids = previous_text + chunk, previous_ids + delta_ids
        delta = parser.extract_reasoning_streaming(
            previous_text, current_text, chunk, previous_ids, current_ids, delta_ids
        )
        if delta is not None:
            reasoning += delta.reasoning or ""
            content += delta.content or ""
        previous_text, previous_ids = current_text, current_ids
    assert reasoning == expected_reasoning
    assert content == "Berlin"


def test_missing_reasoning_token_still_raises(patched_parser):
    parser_cls, _ = patched_parser
    tokenizer = MiniMaxTokenizer()
    del tokenizer.vocab["</mm:think>"]
    with pytest.raises(RuntimeError, match="could not locate"):
        parser_cls(tokenizer)


def test_patch_is_idempotent(patched_parser):
    parser_cls, namespace = patched_parser
    methods = (parser_cls.is_reasoning_end, parser_cls.is_reasoning_end_streaming)
    namespace["_patch_minimax_m3_reasoning"]()
    runpy.run_path(str(PATCH_PATH))
    assert (parser_cls.is_reasoning_end, parser_cls.is_reasoning_end_streaming) == methods


def test_upstream_fix_is_not_overwritten(patched_parser, monkeypatch):
    _, namespace = patched_parser

    class FixedParser:
        def _last_message_ids(self, input_ids):
            return input_ids

        def is_reasoning_end(self, input_ids):
            return False

        def is_reasoning_end_streaming(self, input_ids, delta_ids):
            return False

    patch = namespace["_patch_minimax_m3_reasoning"]
    monkeypatch.setitem(patch.__globals__, "MiniMaxM3ReasoningParser", FixedParser)
    methods = (FixedParser._last_message_ids, FixedParser.is_reasoning_end, FixedParser.is_reasoning_end_streaming)
    patch()
    assert (
        FixedParser._last_message_ids,
        FixedParser.is_reasoning_end,
        FixedParser.is_reasoning_end_streaming,
    ) == methods
