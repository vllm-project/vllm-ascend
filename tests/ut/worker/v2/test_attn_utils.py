import ast
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_ATTN_UTILS_PATH = _REPO_ROOT / "vllm_ascend" / "worker" / "v2" / "attn_utils.py"
_ATTENTION_V1_PATH = _REPO_ROOT / "vllm_ascend" / "attention" / "attention_v1.py"


class _NumpySpy:
    ndarray = np.ndarray

    def __init__(self):
        self.all_inputs = []

    def array_equal(self, *args, **kwargs):
        return np.array_equal(*args, **kwargs)

    def all(self, condition, *args, **kwargs):
        self.all_inputs.append(np.asarray(condition).copy().tolist())
        return np.all(condition, *args, **kwargs)


def _extract_node(path: Path, node_type: type[ast.AST], name: str) -> ast.AST:
    assert path.is_file()
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, node_type) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


def _load_build_attn_state():
    class EncoderOnlyAttentionSpec:
        pass

    enum_node = _extract_node(_ATTENTION_V1_PATH, ast.ClassDef, "AscendAttentionState")
    enum_module = ast.Module(body=[enum_node], type_ignores=[])
    enum_globals = {"Enum": Enum}
    exec(compile(ast.fix_missing_locations(enum_module), str(_ATTENTION_V1_PATH), "exec"), enum_globals)

    func_node = _extract_node(_ATTN_UTILS_PATH, ast.FunctionDef, "build_attn_state")
    func_module = ast.Module(body=[func_node], type_ignores=[])
    func_globals = {
        "np": np,
        "VllmConfig": object,
        "EncoderOnlyAttentionSpec": EncoderOnlyAttentionSpec,
        "AscendAttentionState": enum_globals["AscendAttentionState"],
    }
    exec(compile(ast.fix_missing_locations(func_module), str(_ATTN_UTILS_PATH), "exec"), func_globals)
    return (
        func_globals["build_attn_state"],
        func_globals["AscendAttentionState"],
        EncoderOnlyAttentionSpec,
    )


def _make_config(
    encoder_spec_cls,
    *,
    runner_type: str = "generate",
    enable_chunked_prefill: bool = False,
    spec_method: str | None = None,
    pooling_encoder: bool = False,
):
    kv_cache_spec = encoder_spec_cls() if pooling_encoder else object()
    return SimpleNamespace(
        model_config=SimpleNamespace(runner_type=runner_type),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=kv_cache_spec)]),
        speculative_config=SimpleNamespace(method=spec_method) if spec_method is not None else None,
        scheduler_config=SimpleNamespace(enable_chunked_prefill=enable_chunked_prefill),
    )


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_exact_plain_ndarray_alias_skips_valid_token_reduction(dtype):
    build_attn_state, states, encoder_spec_cls = _load_build_attn_state()
    spy = _NumpySpy()
    build_attn_state.__globals__["np"] = spy
    num_tokens = np.array([2, 3], dtype=dtype)

    state = build_attn_state(
        _make_config(encoder_spec_cls, enable_chunked_prefill=True),
        np.array([8, 9], dtype=np.int32),
        2,
        num_tokens,
        num_tokens,
    )

    assert state is states.ChunkedPrefill
    assert spy.all_inputs == [[False, False]]


def test_same_object_strided_plain_ndarray_alias_skips_valid_token_reduction():
    build_attn_state, states, encoder_spec_cls = _load_build_attn_state()
    spy = _NumpySpy()
    build_attn_state.__globals__["np"] = spy
    num_tokens = np.array([2, 8, 3, 9], dtype=np.int32)[::2]

    state = build_attn_state(
        _make_config(encoder_spec_cls),
        np.array([8, 9], dtype=np.int32),
        2,
        num_tokens,
        num_tokens,
    )

    assert type(num_tokens) is np.ndarray
    assert state is states.PrefillCacheHit
    assert spy.all_inputs == [[False, False]]


def test_equal_but_distinct_arrays_keep_valid_token_reduction():
    build_attn_state, states, encoder_spec_cls = _load_build_attn_state()
    spy = _NumpySpy()
    build_attn_state.__globals__["np"] = spy

    state = build_attn_state(
        _make_config(encoder_spec_cls),
        np.array([8, 9], dtype=np.int32),
        2,
        np.array([2, 3], dtype=np.int32),
        np.array([1, 1], dtype=np.int32),
    )

    assert state is states.ChunkedPrefill
    assert spy.all_inputs == [[False, False], [True, True]]


def test_distinct_views_sharing_storage_keep_valid_token_reduction():
    build_attn_state, states, encoder_spec_cls = _load_build_attn_state()
    spy = _NumpySpy()
    build_attn_state.__globals__["np"] = spy
    base = np.array([2, 1, 1], dtype=np.int32)
    scheduled = base[:2]
    valid = base[1:]

    state = build_attn_state(
        _make_config(encoder_spec_cls),
        np.array([8, 9], dtype=np.int32),
        2,
        scheduled,
        valid,
    )

    assert np.shares_memory(scheduled, valid)
    assert scheduled is not valid
    assert state is states.ChunkedPrefill
    assert spy.all_inputs == [[False, True], [True, True]]


def test_exact_alias_ndarray_subclass_keeps_valid_token_reduction():
    build_attn_state, states, encoder_spec_cls = _load_build_attn_state()
    spy = _NumpySpy()
    build_attn_state.__globals__["np"] = spy

    class TokenArray(np.ndarray):
        pass

    num_tokens = np.array([2, 3], dtype=np.int32).view(TokenArray)

    state = build_attn_state(
        _make_config(encoder_spec_cls),
        np.array([8, 9], dtype=np.int32),
        2,
        num_tokens,
        num_tokens,
    )

    assert type(num_tokens) is not np.ndarray
    assert state is states.PrefillCacheHit
    assert spy.all_inputs == [[False, False], [False, False]]


def test_classifier_priority_prefill_no_cache_returns_before_token_reductions():
    build_attn_state, states, encoder_spec_cls = _load_build_attn_state()
    spy = _NumpySpy()
    build_attn_state.__globals__["np"] = spy

    state = build_attn_state(
        _make_config(encoder_spec_cls),
        np.array([2, 3], dtype=np.int32),
        2,
        np.array([2, 3], dtype=np.int32),
        np.array([1, 1], dtype=np.int32),
    )

    assert state is states.PrefillNoCache
    assert spy.all_inputs == []


@pytest.mark.parametrize(
    ("spec_method", "expected_state"),
    [(None, "DecodeOnly"), ("mtp", "SpecDecoding")],
)
def test_scheduled_all_ones_decode_branch_returns_before_valid_token_reduction(spec_method, expected_state):
    build_attn_state, states, encoder_spec_cls = _load_build_attn_state()
    spy = _NumpySpy()
    build_attn_state.__globals__["np"] = spy

    state = build_attn_state(
        _make_config(encoder_spec_cls, spec_method=spec_method),
        np.array([8, 9], dtype=np.int32),
        2,
        np.array([1, 1], dtype=np.int32),
        np.array([1, 1], dtype=np.int32),
    )

    assert state is getattr(states, expected_state)
    assert spy.all_inputs == [[True, True]]


@pytest.mark.parametrize(
    ("pooling_encoder", "expected_state"),
    [(True, "PrefillNoCache"), (False, "PrefillCacheHit")],
)
def test_pooling_handling_returns_before_token_reductions(pooling_encoder, expected_state):
    build_attn_state, states, encoder_spec_cls = _load_build_attn_state()
    spy = _NumpySpy()
    build_attn_state.__globals__["np"] = spy

    state = build_attn_state(
        _make_config(encoder_spec_cls, runner_type="pooling", pooling_encoder=pooling_encoder),
        np.array([1, 1], dtype=np.int32),
        2,
        np.array([1, 1], dtype=np.int32),
        np.array([1, 1], dtype=np.int32),
    )

    assert state is getattr(states, expected_state)
    assert spy.all_inputs == []


@pytest.mark.parametrize(
    ("enable_chunked_prefill", "expected_state"),
    [(True, "ChunkedPrefill"), (False, "PrefillCacheHit")],
)
def test_chunked_prefill_fallback_on_and_off(enable_chunked_prefill, expected_state):
    build_attn_state, states, encoder_spec_cls = _load_build_attn_state()

    state = build_attn_state(
        _make_config(encoder_spec_cls, enable_chunked_prefill=enable_chunked_prefill),
        np.array([8, 9], dtype=np.int32),
        2,
        np.array([2, 3], dtype=np.int32),
        np.array([2, 3], dtype=np.int32),
    )

    assert state is getattr(states, expected_state)


def test_build_attn_state_does_not_mutate_inputs():
    build_attn_state, states, encoder_spec_cls = _load_build_attn_state()
    seq_lens = np.array([8, 9], dtype=np.int32)
    base = np.array([2, 1, 3, 1], dtype=np.int32)
    scheduled = base[::2]
    valid = base[1::2]
    seq_lens_before = seq_lens.copy()
    base_before = base.copy()
    scheduled_before = scheduled.copy()
    valid_before = valid.copy()

    state = build_attn_state(
        _make_config(encoder_spec_cls),
        seq_lens,
        2,
        scheduled,
        valid,
    )

    assert state is states.ChunkedPrefill
    assert np.array_equal(seq_lens, seq_lens_before)
    assert np.array_equal(base, base_before)
    assert np.array_equal(scheduled, scheduled_before)
    assert np.array_equal(valid, valid_before)
