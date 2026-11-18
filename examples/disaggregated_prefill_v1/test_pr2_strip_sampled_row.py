# Regression tests for PR #21815 v2 -- strip_sampled_row (pd_proxy cloze shortcut)
#
# Covers the two reviewer-reported issues fixed in v2:
#   P2-1 (Big2Wheel): v1 trimmed only choices[0] -> v2 loops every choice and
#        recomputes usage.total_tokens behind an isinstance(dict) guard.
#   P2-2 (Big2Wheel): v1 left text_offset untouched -> v2 trims it in lockstep
#        with tokens/token_logprobs/top_logprobs.
#
# Upstream shape reproduced here = vLLM completions echo + logprobs semantics:
# one array row per echoed token, plus exactly one forced sampled row appended
# by the engine (min_tokens=1 on a max_tokens=0 echo request), and
# text_offset carrying one entry per row (char start of each token).
#
# The function is extracted from pd_proxy_pr2_cloze_shortcut.py via AST so the
# tests run standalone on any CPU box (no fastapi/httpx/uvloop imports).
#   python test_pr2_strip_sampled_row.py        (plain runner)
#   pytest    test_pr2_strip_sampled_row.py     (pytest, if available)

import ast
import pathlib

SRC_CANDIDATES = [
    "pd_proxy_pr2_cloze_shortcut.py",   # local PR working copy (LF)
    "load_balance_proxy_server_example.py",  # vllm-ascend examples/ copy
    "pd_proxy.py",                       # device runtime copy
    "pd_proxy_pr2_cloze_shortcut_CRLF.py",
]


def _find_src():
    here = pathlib.Path(__file__).parent
    for name in SRC_CANDIDATES:
        p = here / name
        if p.exists():
            return p
    raise FileNotFoundError(f"none of {SRC_CANDIDATES} found next to this test")


SRC_PATH = _find_src()


def _load_strip_sampled_row():
    src = SRC_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "strip_sampled_row":
            seg = ast.get_source_segment(src, node)
            ns = {}
            exec(compile(seg, "<strip_sampled_row>", "exec"), ns)
            return ns["strip_sampled_row"]
    raise RuntimeError("strip_sampled_row not found in source")


strip = _load_strip_sampled_row()


def _echo_response(choices):
    """Build an upstream echo response: N echo rows + 1 forced sampled row.

    choices: list of dicts with keys:
      echo_tokens   -- list of echoed token strings
      forced_token  -- the single sampled token appended by min_tokens=1
      logprobs_rows -- list of logprob floats, one per row (echo + forced)
      offsets       -- text_offset list, one entry per row (vLLM semantics)
    """
    out_choices = []
    for spec in choices:
        rows = list(spec["echo_tokens"]) + [spec["forced_token"]]
        out_choices.append({
            "index": len(out_choices),
            "text": "".join(rows),
            "finish_reason": "length",  # max_tokens=0 + min_tokens=1 -> length
            "logprobs": {
                "tokens": rows,
                "token_logprobs": spec["logprobs_rows"],
                "top_logprobs": [{t: p} for t, p in zip(rows, spec["logprobs_rows"])],
                "text_offset": spec["offsets"],
            },
        })
    return {
        "id": "cmpl-test",
        "object": "text_completion",
        "choices": out_choices,
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
    }


def test_batch_two_choices_all_trimmed():
    """P2-1: both choices trimmed + usage recomputed (was: only choices[0])."""
    resp = _echo_response([
        {   # 3 echo rows + 1 forced row
            "echo_tokens": ["Hello", "!", " world"],
            "forced_token": " world",
            "logprobs_rows": [-0.1, -0.2, -0.3, -0.7],
            "offsets": [0, 5, 6, 12],
        },
        {
            "echo_tokens": ["World", "!", "!"],
            "forced_token": "!",
            "logprobs_rows": [-0.05, -0.15, -0.25, -0.9],
            "offsets": [0, 5, 6, 7],
        },
    ])
    out = strip(resp)

    a, b = out["choices"]
    # choice 0: 4 rows -> 3, text rebuilt from trimmed tokens only
    assert a["text"] == "Hello! world"
    assert a["logprobs"]["tokens"] == ["Hello", "!", " world"]
    assert a["logprobs"]["token_logprobs"] == [-0.1, -0.2, -0.3]
    assert len(a["logprobs"]["top_logprobs"]) == 3
    # text_offset trimmed in lockstep (P2-2), one entry per remaining token
    assert a["logprobs"]["text_offset"] == [0, 5, 6]
    # choice 1 (v1 left this untouched): same treatment
    assert b["text"] == "World!!"
    assert b["logprobs"]["tokens"] == ["World", "!", "!"]
    assert b["logprobs"]["token_logprobs"] == [-0.05, -0.15, -0.25]
    assert b["logprobs"]["text_offset"] == [0, 5, 6]
    # forced finish_reason 'length' restored to native-echo 'stop' on both
    assert a["finish_reason"] == "stop"
    assert b["finish_reason"] == "stop"
    # usage recomputed: completion zeroed, total == prompt (isinstance guard fix)
    assert out["usage"] == {"prompt_tokens": 10, "completion_tokens": 0,
                            "total_tokens": 10}


def test_text_offset_aligned_two_token_echo():
    """P2-2: reviewer's minimal repro -- offsets stay locked to token count."""
    resp = _echo_response([{
        "echo_tokens": ["Hello", " world"],
        "forced_token": " world",
        "logprobs_rows": [-0.01, -0.02, -0.7],
        "offsets": [0, 5, 11],
    }])
    out = strip(resp)
    (choice,) = out["choices"]
    assert choice["text"] == "Hello world"
    lp = choice["logprobs"]
    assert lp["tokens"] == ["Hello", " world"]
    assert lp["token_logprobs"] == [-0.01, -0.02]
    assert lp["text_offset"] == [0, 5]          # was [0, 5, 11] in v1
    assert len(lp["text_offset"]) == len(lp["tokens"])


def test_no_logprobs_choice_still_fixed():
    """Choice without logprobs: no trim possible, but finish_reason still fixed."""
    resp = {
        "choices": [{"index": 0, "text": "echo text", "finish_reason": "length"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8},
    }
    out = strip(resp)
    (choice,) = out["choices"]
    assert choice["finish_reason"] == "stop"
    assert out["usage"]["completion_tokens"] == 0
    assert out["usage"]["total_tokens"] == 7


def test_non_dict_usage_untouched():
    """High fix: `usage or {}` created a temp dict -- malformed usage must not
    be silently replaced; non-dict usage passes through untouched."""
    resp = {
        "choices": [{"index": 0, "text": "x", "finish_reason": "stop"}],
        "usage": None,
    }
    out = strip(resp)
    assert out["usage"] is None


def test_single_row_response_unchanged():
    """Degenerate 1-row echo (no separable forced row): nothing trimmed."""
    resp = {
        "choices": [{
            "index": 0,
            "text": "Hello",
            "finish_reason": "stop",
            "logprobs": {
                "tokens": ["Hello"],
                "token_logprobs": [-0.3],
                "top_logprobs": [{"Hello": -0.3}],
                "text_offset": [0],
            },
        }],
        "usage": {"prompt_tokens": 2, "completion_tokens": 0, "total_tokens": 2},
    }
    out = strip(resp)
    (choice,) = out["choices"]
    assert choice["logprobs"]["tokens"] == ["Hello"]
    assert choice["logprobs"]["token_logprobs"] == [-0.3]
    assert choice["logprobs"]["text_offset"] == [0]
    assert out["usage"]["total_tokens"] == 2


def test_usage_missing_key_safe():
    """usage dict missing prompt_tokens: defaults to 0, no KeyError."""
    resp = {
        "choices": [{"index": 0, "text": "x", "finish_reason": "stop"}],
        "usage": {"completion_tokens": 1},
    }
    out = strip(resp)
    assert out["usage"] == {"completion_tokens": 0, "total_tokens": 0}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
