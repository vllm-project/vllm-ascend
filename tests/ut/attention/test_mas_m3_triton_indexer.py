"""CPU unit tests for the MiniMax M3 Triton indexer.

The public entry points are tested with launch recorders, while the Python
functions wrapped by ``triton.jit`` are executed with a small tensor-backed
subset of the Triton language.  This keeps the tests hardware independent and
still exercises the indexing and masking logic inside the kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.attention import msa_m3_triton_indexer as indexer


def _unwrap_triton_function(kernel):
    """Return the Python function below autotune/heuristics/JIT wrappers."""
    while hasattr(kernel, "fn"):
        kernel = kernel.fn
    return kernel


@dataclass(frozen=True)
class _Pointer:
    tensor: torch.Tensor
    offsets: object = 0

    @property
    def dtype(self):
        return SimpleNamespace(element_ty=self.tensor.dtype)

    def __add__(self, offsets):
        return _Pointer(self.tensor, torch.as_tensor(self.offsets) + offsets)

    __radd__ = __add__


def _ptr(tensor: torch.Tensor) -> _Pointer:
    return _Pointer(tensor)


class _FakeCuda:
    def __init__(self) -> None:
        self.wait_count = 0
        self.launch_count = 0

    def gdc_wait(self) -> None:
        self.wait_count += 1

    def gdc_launch_dependents(self) -> None:
        self.launch_count += 1


class _FakeTL:
    """Tensor-backed subset of ``triton.language`` used by these kernels."""

    int64 = torch.int64
    float32 = torch.float32
    constexpr = object()

    def __init__(self) -> None:
        self.program_ids = (0, 0, 0)
        self.extra = SimpleNamespace(cuda=_FakeCuda())

    def set_program_ids(self, *program_ids: int) -> None:
        self.program_ids = (*program_ids, 0, 0, 0)[:3]

    def program_id(self, axis: int) -> int:
        return self.program_ids[axis]

    @staticmethod
    def static_assert(condition, message=None) -> None:
        assert condition, message

    @staticmethod
    def arange(start: int, end: int) -> torch.Tensor:
        return torch.arange(start, end)

    @staticmethod
    def range(start, end, step=1):
        return range(int(start), int(end), int(step))

    @staticmethod
    def _mask_for(offsets: torch.Tensor, mask) -> torch.Tensor:
        if mask is None:
            return torch.ones_like(offsets, dtype=torch.bool)
        return torch.as_tensor(mask, dtype=torch.bool).broadcast_to(offsets.shape)

    @classmethod
    def load(cls, pointer: _Pointer, mask=None, other=0):
        offsets = torch.as_tensor(pointer.offsets, dtype=torch.long)
        if mask is None:
            return pointer.tensor.reshape(-1)[offsets]
        access_mask = cls._mask_for(offsets, mask)
        safe_offsets = torch.where(access_mask, offsets, 0)
        values = pointer.tensor.reshape(-1)[safe_offsets]
        fallback = torch.as_tensor(other, dtype=values.dtype).broadcast_to(values.shape)
        return torch.where(access_mask, values, fallback)

    @classmethod
    def store(cls, pointer: _Pointer, value, mask=None) -> None:
        offsets = torch.as_tensor(pointer.offsets, dtype=torch.long)
        access_mask = cls._mask_for(offsets, mask)
        values = torch.as_tensor(value, dtype=pointer.tensor.dtype).broadcast_to(offsets.shape)
        pointer.tensor.reshape(-1)[offsets[access_mask]] = values[access_mask]

    @staticmethod
    def minimum(left, right):
        return torch.minimum(torch.as_tensor(left), torch.as_tensor(right))

    @staticmethod
    def maximum(left, right):
        return torch.maximum(torch.as_tensor(left), torch.as_tensor(right))

    @staticmethod
    def where(condition, left, right):
        return torch.where(condition, left, right)

    @staticmethod
    def dot(left, right, out_dtype=None):
        result = torch.matmul(left, right)
        return result.to(out_dtype) if out_dtype is not None else result

    @staticmethod
    def max(value, axis: int):
        return torch.max(value, dim=axis).values

    @staticmethod
    def floor(value):
        return torch.floor(value)


@pytest.fixture
def fake_tl(monkeypatch: pytest.MonkeyPatch) -> _FakeTL:
    language = _FakeTL()
    monkeypatch.setattr(indexer, "tl", language)
    return language


class _LaunchRecorder:
    def __init__(self, callback=None) -> None:
        self.callback = callback
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
            if self.callback is not None:
                self.callback(*args, **kwargs)

        return launch


@pytest.mark.parametrize(
    ("cache_factory", "expected_shape"),
    [
        (lambda: torch.zeros(3, 128, 4), (3, 128, 4)),
        (lambda: torch.zeros(3, 128, 1, 4), (3, 128, 4)),
        (lambda: torch.zeros(2, 3, 128, 1, 4), (3, 128, 4)),
        (lambda: (torch.zeros(3, 128, 4), torch.ones(1)), (3, 128, 4)),
        (lambda: [torch.zeros(3, 128, 4)], (3, 128, 4)),
    ],
)
def test_as_triton_index_kv_cache_accepts_supported_layouts(cache_factory, expected_shape) -> None:
    assert indexer._as_triton_index_kv_cache(cache_factory()).shape == expected_shape


@pytest.mark.parametrize(
    "cache",
    [
        torch.zeros(3, 128, 2, 4),
        torch.zeros(3, 128),
        torch.zeros(1, 2, 3, 4, 5, 6),
    ],
)
def test_as_triton_index_kv_cache_rejects_unsupported_layouts(cache: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="Unexpected index cache"):
        indexer._as_triton_index_kv_cache(cache)


@pytest.mark.parametrize("value", ["bad", "0", "-2"])
def test_read_positive_int_env_rejects_invalid_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("M3_TEST_POSITIVE", value)
    with pytest.raises(ValueError, match="must be a positive integer"):
        indexer._read_positive_int_env("M3_TEST_POSITIVE", 7)


def test_environment_readers_handle_defaults_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    assert indexer._read_positive_int_env("M3_TEST_MISSING", 7) == 7
    assert indexer._read_int_choice_env("M3_TEST_MISSING", (1, 2)) is None

    monkeypatch.setenv("M3_TEST_POSITIVE", "9")
    monkeypatch.setenv("M3_TEST_CHOICE", "2")
    assert indexer._read_positive_int_env("M3_TEST_POSITIVE", 7) == 9
    assert indexer._read_int_choice_env("M3_TEST_CHOICE", (1, 2)) == 2


@pytest.mark.parametrize("value", ["bad", "3"])
def test_read_int_choice_env_rejects_invalid_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("M3_TEST_CHOICE", value)
    with pytest.raises(ValueError, match="must be one of 1, 2"):
        indexer._read_int_choice_env("M3_TEST_CHOICE", (1, 2))


def test_prune_decode_score_autotune_configs() -> None:
    configs = [SimpleNamespace(kwargs={"num_kv_chunks": count}) for count in (1, 2, 4, 8)]
    assert [
        config.kwargs["num_kv_chunks"]
        for config in indexer._prune_decode_score_autotune_configs(configs, {"num_reqs": 200})
    ] == [1, 2]
    assert indexer._prune_decode_score_autotune_configs(configs, {"num_reqs": 9999}) == configs[:1]
    assert indexer._prune_decode_score_autotune_configs(configs, {"num_reqs": 0}) == configs


def test_choose_prefill_finalize_query_tile_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(indexer.ENV_PREFILL_FINALIZE_QUERY_TILE, "32")
    assert indexer._choose_prefill_finalize_query_tile_size(4096, 4, 2) == 32
    monkeypatch.delenv(indexer.ENV_PREFILL_FINALIZE_QUERY_TILE)

    assert indexer._choose_prefill_finalize_query_tile_size(0, 0, 0) == 8
    assert indexer._choose_prefill_finalize_query_tile_size(1024, 4, 2) == 64
    assert indexer._choose_prefill_finalize_query_tile_size(8192, 4, 2) == 64


def test_choose_prefill_invalid_mask_query_tile_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(indexer.ENV_PREFILL_INVALID_MASK_QUERY_TILE, "16")
    assert indexer._choose_prefill_invalid_mask_query_tile_size(1024, 2, 2, 16) == 16
    monkeypatch.delenv(indexer.ENV_PREFILL_INVALID_MASK_QUERY_TILE)

    assert indexer._choose_prefill_invalid_mask_query_tile_size(0, 0, 0, 0) == 1
    assert indexer._choose_prefill_invalid_mask_query_tile_size(256, 2, 1, 16) == 4
    monkeypatch.setenv(indexer.ENV_PREFILL_INVALID_MASK_MAX_TILE_ELEMENTS, "16")
    assert indexer._choose_prefill_invalid_mask_query_tile_size(8192, 4, 2, 64) == 1


def test_copy_topk_indices_fast_path_padding_and_output() -> None:
    raw = torch.tensor([[[2, 1], [0, 3]]], dtype=torch.int64)
    fast = indexer._copy_topk_indices(raw, 2, None)
    assert fast.dtype == torch.int32
    assert torch.equal(fast, raw.to(torch.int32))

    padded = indexer._copy_topk_indices(raw, 4, None)
    assert padded.tolist() == [[[2, 1, -1, -1], [0, 3, -1, -1]]]

    output = torch.full((1, 5, 5), 99, dtype=torch.int32)
    result = indexer._copy_topk_indices(raw, 3, output)
    assert result.data_ptr() == output.data_ptr()
    assert result.tolist() == [[[2, 1, -1], [0, 3, -1]]]
    assert torch.all(output[:, 2:] == 99)


def test_prefill_score_kernel_causal_full_boundary_and_empty_tile(fake_tl: _FakeTL) -> None:
    kernel = _unwrap_triton_function(indexer._prefill_index_block_score_kernel)
    query = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]])
    cache = torch.tensor([[[1.0, 0.0], [2.0, 0.0]], [[3.0, 0.0], [4.0, 0.0]]])
    score = torch.full((1, 2, 2), -99.0)
    block_table = torch.tensor([[0, 1]])
    cu_seqlens = torch.tensor([0, 2])
    seq_lens = torch.tensor([4])
    prefix_lens = torch.tensor([2])

    kernel(
        _ptr(query),
        _ptr(cache),
        _ptr(score),
        _ptr(block_table),
        _ptr(cu_seqlens),
        _ptr(seq_lens),
        _ptr(prefix_lens),
        1,
        2,
        *query.stride(),
        *cache.stride(),
        *score.stride(),
        block_table.stride(0),
        BLOCK_SIZE_Q=2,
        BLOCK_SIZE_K=2,
    )
    assert torch.equal(score, torch.tensor([[[2.0, 3.0], [2.0, 4.0]]]))

    fake_tl.set_program_ids(1, 0)
    kernel(
        _ptr(query),
        _ptr(cache),
        _ptr(score),
        _ptr(block_table),
        _ptr(cu_seqlens),
        _ptr(seq_lens),
        _ptr(prefix_lens),
        1,
        2,
        *query.stride(),
        *cache.stride(),
        *score.stride(),
        block_table.stride(0),
        BLOCK_SIZE_Q=2,
        BLOCK_SIZE_K=2,
    )


def test_decode_score_kernel_priorities_pdl_and_empty_chunk(fake_tl: _FakeTL) -> None:
    kernel = _unwrap_triton_function(indexer._decode_index_score_kernel)
    query = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]])
    cache = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])
    score = torch.full((1, 2, 1), -99.0)
    init_mask = torch.tensor([[True], [False]])
    local_mask = torch.tensor([[False], [True]])
    block_table = torch.tensor([[0]])
    seq_lens = torch.tensor([2])

    kernel(
        _ptr(query),
        _ptr(cache),
        _ptr(score),
        _ptr(init_mask),
        _ptr(local_mask),
        _ptr(block_table),
        _ptr(seq_lens),
        1,
        2,
        1,
        2,
        *query.stride(),
        *cache.stride(),
        *score.stride(),
        *init_mask.stride(),
        block_table.stride(0),
        BLOCK_SIZE_K=2,
        BLOCK_SIZE_Q=2,
        num_kv_chunks=1,
        USE_PDL=True,
    )
    assert score[0, 0, 0] == pytest.approx(1e30)
    assert score[0, 1, 0] == pytest.approx(1e29)
    assert fake_tl.extra.cuda.wait_count == fake_tl.extra.cuda.launch_count == 1

    fake_tl.set_program_ids(0, 1)
    kernel(
        _ptr(query),
        _ptr(cache),
        _ptr(score),
        _ptr(init_mask),
        _ptr(local_mask),
        _ptr(block_table),
        _ptr(seq_lens),
        1,
        2,
        1,
        2,
        *query.stride(),
        *cache.stride(),
        *score.stride(),
        *init_mask.stride(),
        block_table.stride(0),
        BLOCK_SIZE_K=2,
        BLOCK_SIZE_Q=2,
        num_kv_chunks=2,
        USE_PDL=False,
    )


def test_decode_tail_fill_and_invalid_index_mask_kernels(fake_tl: _FakeTL) -> None:
    tail_kernel = _unwrap_triton_function(indexer._decode_score_tail_fill_kernel)
    score = torch.zeros((1, 2, 4))
    seq_lens = torch.tensor([1])
    for query_id in (0, 1):
        fake_tl.set_program_ids(query_id, 0, 0)
        tail_kernel(_ptr(score), _ptr(seq_lens), 2, 4, 2, 4, *score.stride(), BLOCK_SIZE_K=4)
    assert torch.isneginf(score[0, 0]).all()
    assert score[0, 1, 0] == 0
    assert torch.isneginf(score[0, 1, 1:]).all()
    fake_tl.set_program_ids(0, 0, 1)
    tail_kernel(_ptr(score), _ptr(seq_lens), 2, 4, 2, 4, *score.stride(), BLOCK_SIZE_K=4)

    invalid_kernel = _unwrap_triton_function(indexer._decode_topk_invalid_index_mask_kernel)
    indices = torch.tensor([[[0, 1, 2], [0, 3, -1]]], dtype=torch.int32)
    for query_id in (0, 1):
        fake_tl.set_program_ids(query_id, 0)
        invalid_kernel(_ptr(indices), _ptr(seq_lens), 2, 3, 2, *indices.stride(), BLOCK_SIZE_T=4)
    assert indices.tolist() == [[[-1, -1, -1], [0, -1, -1]]]


def test_prefill_finalize_kernel_forced_blocks_tail_and_empty_tile(fake_tl: _FakeTL) -> None:
    kernel = _unwrap_triton_function(indexer._prefill_score_finalize_kernel)
    score = torch.zeros((1, 2, 4))
    cu_seqlens = torch.tensor([0, 2])
    prefix_lens = torch.tensor([1])

    kernel(
        _ptr(score),
        _ptr(cu_seqlens),
        _ptr(prefix_lens),
        1,
        1,
        1,
        4,
        *score.stride(),
        2,
        BLOCK_SIZE_Q=2,
        BLOCK_SIZE_FORCE=2,
        BLOCK_SIZE_TAIL=2,
    )
    assert score[0, 0, 0] == pytest.approx(1e29)
    assert torch.isneginf(score[0, 0, 1:]).all()
    assert score[0, 1, 0] == pytest.approx(1e30)
    assert score[0, 1, 1] == pytest.approx(1e29)
    assert torch.isneginf(score[0, 1, 2:]).all()

    fake_tl.set_program_ids(1, 0)
    kernel(
        _ptr(score),
        _ptr(cu_seqlens),
        _ptr(prefix_lens),
        1,
        0,
        0,
        4,
        *score.stride(),
        2,
        BLOCK_SIZE_Q=2,
        BLOCK_SIZE_FORCE=1,
        BLOCK_SIZE_TAIL=2,
    )


def test_prefill_invalid_index_mask_kernel_validates_rank_and_id(fake_tl: _FakeTL) -> None:
    kernel = _unwrap_triton_function(indexer._prefill_topk_invalid_index_mask_kernel)
    indices = torch.tensor([[[0, 1, 2], [1, 0, 3]]], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 2])
    prefix_lens = torch.tensor([1])

    kernel(
        _ptr(indices),
        _ptr(cu_seqlens),
        _ptr(prefix_lens),
        1,
        2,
        3,
        *indices.stride(),
        BLOCK_SIZE_Q=2,
        BLOCK_SIZE_T=4,
    )
    assert indices.tolist() == [[[0, -1, -1], [1, 0, -1]]]

    fake_tl.set_program_ids(1, 0)
    kernel(
        _ptr(indices),
        _ptr(cu_seqlens),
        _ptr(prefix_lens),
        1,
        2,
        3,
        *indices.stride(),
        BLOCK_SIZE_Q=2,
        BLOCK_SIZE_T=4,
    )


def test_decode_init_local_mask_kernel_and_empty_chunk(fake_tl: _FakeTL) -> None:
    kernel = _unwrap_triton_function(indexer._decode_init_local_mask_kernel)
    init_mask = torch.zeros((2, 3), dtype=torch.bool)
    local_mask = torch.zeros_like(init_mask)
    seq_lens = torch.tensor([2])

    for query_id in (0, 1):
        fake_tl.set_program_ids(query_id, 0)
        kernel(
            _ptr(init_mask),
            _ptr(local_mask),
            _ptr(seq_lens),
            2,
            3,
            2,
            3,
            1,
            1,
            *init_mask.stride(),
            BLOCK_SIZE_K=4,
        )
    assert init_mask.tolist() == [[True, False, False], [True, False, False]]
    assert local_mask.tolist() == [[True, False, False], [True, False, False]]

    fake_tl.set_program_ids(0, 1)
    kernel(
        _ptr(init_mask),
        _ptr(local_mask),
        _ptr(seq_lens),
        2,
        3,
        2,
        3,
        1,
        1,
        *init_mask.stride(),
        BLOCK_SIZE_K=4,
    )


def test_minimax_m3_index_score_launches_expected_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    launch = _LaunchRecorder()
    monkeypatch.setattr(indexer, "_prefill_index_block_score_kernel", launch)
    query = torch.zeros((3, 2, 4))
    cache = torch.zeros((5, 128, 1, 4))
    block_table = torch.zeros((2, 2), dtype=torch.int32)

    score = indexer.minimax_m3_index_score(
        query,
        cache,
        block_table,
        torch.tensor([0, 2, 3]),
        torch.tensor([2, 129]),
        torch.tensor([0, 128]),
        max_query_len=2,
        max_seq_len=129,
        num_kv_heads=2,
    )
    assert score.shape == (2, 3, 16)
    assert launch.calls[0][0] == (1, 4)
    assert launch.calls[0][1][1].shape == (5, 128, 4)
    assert launch.calls[0][2] == {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 128}


def test_minimax_m3_index_topk_launches_finalize_and_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    finalize = _LaunchRecorder()
    invalid_mask = _LaunchRecorder()
    monkeypatch.setattr(indexer, "_prefill_score_finalize_kernel", finalize)
    monkeypatch.setattr(indexer, "_prefill_topk_invalid_index_mask_kernel", invalid_mask)
    score = torch.tensor([[[1.0, 3.0], [4.0, 2.0]]])

    result = indexer.minimax_m3_index_topk(
        score,
        torch.tensor([0, 2]),
        torch.tensor([0]),
        max_query_len=2,
        topk=3,
        init_blocks=0,
        local_blocks=1,
    )
    assert result.tolist() == [[[1, 0, -1], [0, 1, -1]]]
    assert finalize.calls[0][0] == (1, 1)
    assert invalid_mask.calls[0][0] == (2, 1)
    with pytest.raises(AssertionError):
        indexer.minimax_m3_index_topk(score, torch.tensor([0, 2]), torch.tensor([0]), 2, 0, 0, 0)


@pytest.mark.parametrize("use_pdl", [False, True])
def test_minimax_m3_index_decode_launches_all_stages(monkeypatch: pytest.MonkeyPatch, use_pdl: bool) -> None:
    launches = {
        name: _LaunchRecorder()
        for name in (
            "_decode_init_local_mask_kernel",
            "_decode_index_score_kernel",
            "_decode_score_tail_fill_kernel",
            "_decode_topk_invalid_index_mask_kernel",
        )
    }
    for name, launch in launches.items():
        monkeypatch.setattr(indexer, name, launch)
    platform = MagicMock()
    platform.is_arch_support_pdl.return_value = use_pdl
    monkeypatch.setattr(indexer, "current_platform", platform)

    result = indexer.minimax_m3_index_decode(
        torch.zeros((4, 1, 2)),
        torch.zeros((4, 128, 2)),
        torch.zeros((2, 2), dtype=torch.int32),
        torch.tensor([2, 129]),
        max_seq_len=129,
        topk=3,
        init_blocks=1,
        local_blocks=1,
        num_kv_heads=1,
        decode_query_len=2,
        max_decode_query_len=4,
    )
    assert result.shape == (1, 4, 3)
    assert torch.all(result[..., 2] == -1)
    assert launches["_decode_init_local_mask_kernel"].calls[0][0] == (4, 16)
    score_call = launches["_decode_index_score_kernel"].calls[0]
    assert score_call[0]({"num_kv_chunks": 2}) == (2, 2)
    assert score_call[2]["BLOCK_SIZE_Q"] == 4
    assert score_call[2]["USE_PDL"] is use_pdl
    assert ("launch_pdl" in score_call[2]) is use_pdl
    assert launches["_decode_score_tail_fill_kernel"].calls[0][0] == (4, 1, 16)
    assert launches["_decode_topk_invalid_index_mask_kernel"].calls[0][0] == (4, 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"topk": 0},
        {"num_kv_heads": 2},
        {"decode_query_len": 3},
        {"decode_query_len": 2, "max_decode_query_len": 1},
    ],
)
def test_minimax_m3_index_decode_validates_arguments(kwargs) -> None:
    arguments = dict(
        idx_q=torch.zeros((4, 1, 2)),
        index_kv_cache=torch.zeros((4, 128, 2)),
        block_table=torch.zeros((2, 2), dtype=torch.int32),
        seq_lens=torch.tensor([2, 2]),
        max_seq_len=129,
        topk=2,
        init_blocks=0,
        local_blocks=1,
        num_kv_heads=1,
        decode_query_len=2,
    )
    arguments.update(kwargs)
    with pytest.raises(AssertionError):
        indexer.minimax_m3_index_decode(**arguments)
