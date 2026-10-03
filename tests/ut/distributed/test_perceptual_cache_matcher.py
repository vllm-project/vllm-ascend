import math
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from vllm_ascend.distributed.ec_transfer import perceptual_ssim_cache_matcher as matcher_module
from vllm_ascend.distributed.ec_transfer.perceptual_ssim_cache_matcher import (
    ImagePixelTensorMetadata,
    MemcacheResizedTensorStore,
    PhashMatchItem,
    PhashSSIMCacheMatcher,
    SimilarityCacheConfig,
)
from vllm_ascend.distributed.ec_transfer.phash import _dct_matrix
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import MmcDirect


class _InMemoryResizedTensorStore:
    def __init__(self) -> None:
        self.values: dict[str, torch.Tensor] = {}
        self.get_calls: list[str] = []
        self.batch_get_calls: list[list[str]] = []

    @staticmethod
    def get_resized_key(mm_hash: str) -> str:
        return "resized_" + mm_hash

    def put(self, mm_hash: str, tensor: torch.Tensor) -> None:
        self.values[mm_hash] = tensor.clone()

    def get(self, mm_hash: str, metadata: ImagePixelTensorMetadata) -> torch.Tensor | None:
        return self.batch_get([mm_hash], [metadata])[0]

    def batch_get(
        self,
        mm_hashes: list[str],
        metadatas: list[ImagePixelTensorMetadata],
    ) -> list[torch.Tensor | None]:
        assert len(mm_hashes) == len(metadatas)
        self.batch_get_calls.append(list(mm_hashes))
        self.get_calls.extend(mm_hashes)
        return [
            None if (tensor := self.values.get(mm_hash)) is None else tensor.clone()
            for mm_hash in mm_hashes
        ]


def _item(resized: torch.Tensor, grid: tuple[int, int, int]) -> dict:
    return {
        "pixel_values": SimpleNamespace(data=resized),
        "image_grid_thw": SimpleNamespace(data=torch.tensor([grid])),
    }


def _make_matcher() -> PhashSSIMCacheMatcher:
    backend = MagicMock()
    backend.exists.side_effect = lambda keys: [1] * len(keys)
    matcher = PhashSSIMCacheMatcher(
        backend,
        SimilarityCacheConfig(enabled=True, matcher="phash_ssim"),
        merge_size=2,
    )
    matcher._tensor_store = _InMemoryResizedTensorStore()
    return matcher


def test_matcher_owns_similarity_compute_executor() -> None:
    put_executor = MagicMock()
    similarity_compute_executor = MagicMock()

    with patch.object(
        matcher_module,
        "ThreadPoolExecutor",
        side_effect=[put_executor, similarity_compute_executor],
    ) as executor_cls:
        matcher = PhashSSIMCacheMatcher(
            MagicMock(),
            SimilarityCacheConfig(enabled=True, matcher="phash_ssim"),
            merge_size=2,
        )

    assert executor_cls.call_args_list == [
        call(max_workers=1, thread_name_prefix="ec-resized-put"),
        call(
            max_workers=matcher_module._MAX_SIMILARITY_COMPUTE_WORKERS,
            thread_name_prefix="ec-similarity-compute",
        ),
    ]
    assert matcher._similarity_compute_executor is similarity_compute_executor

    matcher.shutdown()

    similarity_compute_executor.shutdown.assert_called_once_with(wait=True)
    put_executor.shutdown.assert_called_once_with(wait=True)


def test_dct_matrix_uses_frequency_rows() -> None:
    matrix = _dct_matrix()

    assert torch.allclose(matrix[0], torch.full((32,), 1.0 / math.sqrt(2.0)))
    expected_first_frequency = torch.cos(math.pi / 32 * (torch.arange(32) + 0.5))
    assert torch.allclose(matrix[1], expected_first_frequency)


def test_get_or_update_similarity_match_returns_candidate_and_records_score() -> None:
    matcher = _make_matcher()
    resized = torch.arange(16 * 12, dtype=torch.float32).reshape(16, 12)
    item = _item(resized, (1, 4, 4))

    matcher.register("candidate", item)
    matcher.shutdown()
    matcher.prewarm_candidate_resized_tensors("query", item)

    hit = matcher.get_or_update_similarity_match("query", item)

    assert hit == "candidate"
    assert matcher.phash_match_results["query"].candidate_mm_hash_group_by_hamming == {
        0: {"candidate": 1.0}
    }
    assert matcher.phash_match_results["candidate"].candidate_mm_hash_group_by_hamming == {
        0: {"query": 1.0}
    }


def test_get_or_update_similarity_match_returns_none_without_prewarmed_result() -> None:
    matcher = _make_matcher()
    item = _item(torch.zeros(16, 12), (1, 4, 4))

    assert matcher.get_or_update_similarity_match("query", item) is None
    assert matcher._tensor_store.get_calls == []
    cached = matcher.phash_match_results["query"]
    assert cached.image_pixel_tensor_metadata.grid == (1, 4, 4)
    assert cached.candidate_mm_hash_group_by_hamming == {}
    matcher.shutdown()


def test_register_persists_item_already_cached_by_phash_lookup() -> None:
    matcher = _make_matcher()
    resized = torch.zeros(16, 12)
    grid = (1, 4, 4)
    item = _item(resized, grid)

    with patch.object(matcher_module, "compute_phash", return_value=123) as compute:
        assert matcher._cache_or_compute_phash("query", resized, grid) == (123, grid)
        matcher.register("query", item)
    matcher.shutdown()

    compute.assert_called_once_with(resized, grid, merge_size=2)
    assert matcher._phash_to_mm_hashes[123] == {"query"}
    assert torch.equal(matcher._tensor_store.values["query"], resized)


def test_get_or_update_similarity_match_only_scores_minimum_hamming_group() -> None:
    matcher = _make_matcher()
    grid = (1, 4, 4)
    query_resized = torch.zeros(16, 12)
    candidate_resized = torch.ones(16, 12)
    metadata = ImagePixelTensorMetadata(tuple(candidate_resized.shape), candidate_resized.dtype, grid)
    matcher.phash_match_results["query"] = PhashMatchItem(
        image_pixel_tensor_metadata=ImagePixelTensorMetadata(
            tuple(query_resized.shape), query_resized.dtype, grid
        ),
        current_phash=100,
        candidate_mm_hash_group_by_hamming={
            1: {"near-low": None, "near-high": None},
            2: {"far": None},
        },
    )
    for candidate_mm_hash, candidate_phash in (
        ("near-low", 101),
        ("near-high", 102),
        ("far", 103),
    ):
        matcher.phash_match_results[candidate_mm_hash] = PhashMatchItem(
            image_pixel_tensor_metadata=metadata,
            current_phash=candidate_phash,
            candidate_mm_hash_group_by_hamming={},
        )
        matcher._phash_to_mm_hashes[candidate_phash] = {candidate_mm_hash}
        matcher._tensor_store.values[candidate_mm_hash] = candidate_resized

    with patch.object(matcher_module, "ssim_score", side_effect=[0.80, 0.999]):
        hit = matcher.get_or_update_similarity_match("query", _item(query_resized, grid))

    assert hit == "near-high"
    assert matcher._tensor_store.get_calls == ["near-low", "near-high"]
    assert matcher.phash_match_results["query"].candidate_mm_hash_group_by_hamming == {
        1: {"near-low": 0.80, "near-high": 0.999},
        2: {"far": None},
    }
    assert matcher.phash_match_results["near-high"].candidate_mm_hash_group_by_hamming == {
        1: {"query": 0.999}
    }
    matcher.shutdown()


def test_prewarm_candidate_resized_tensors_only_checks_candidate_keys() -> None:
    matcher = _make_matcher()
    resized = torch.arange(16 * 12, dtype=torch.float32).reshape(16, 12)
    item = _item(resized, (1, 4, 4))
    matcher.register("candidate", item)
    matcher.shutdown()
    matcher._backend.reset_mock()

    candidates = matcher.prewarm_candidate_resized_tensors("query", item)

    assert candidates == ["candidate"]
    match_item = matcher.phash_match_results["query"]
    assert isinstance(match_item.current_phash, int)
    assert match_item.candidate_mm_hash_group_by_hamming == {0: {"candidate": None}}
    matcher._backend.exists.assert_called_once_with(["resized_candidate"])
    assert matcher._tensor_store.get_calls == []


def test_batch_ensure_similarity_cache_available_keeps_order_and_batches_exists() -> None:
    matcher = _make_matcher()
    resized = torch.arange(16 * 12, dtype=torch.float32).reshape(16, 12)
    item = _item(resized, (1, 4, 4))
    matcher.register("candidate", item)
    matcher.shutdown()
    matcher._backend.reset_mock()

    with patch.object(matcher_module, "ThreadPoolExecutor") as executor:
        candidate_batches = matcher.batch_ensure_similarity_cache_available(
            ["query-1", "query-2", "query-3"],
            [item, item, item],
        )

    assert candidate_batches == [["candidate"]] * 3
    executor.assert_not_called()
    matcher._backend.exists.assert_called_once_with(["resized_candidate"])
    assert matcher._tensor_store.get_calls == []


def test_batch_get_candidate_phash_items_reuses_executor_and_keeps_alignment() -> None:
    matcher = _make_matcher()
    mm_hashes = ["query-1", "query-2", "query-3", "query-4"]
    resized_tensors = [object() for _ in mm_hashes]
    missing_features = [SimpleNamespace(data=tensor) for tensor in resized_tensors]
    matcher._get_candidate_phash_item = MagicMock(
        side_effect=lambda mm_hash, tensor: (mm_hash, tensor)
    )
    similarity_compute_executor = MagicMock()
    similarity_compute_executor.map.side_effect = lambda fn, hashes, tensors: map(
        fn, hashes, tensors
    )
    matcher._similarity_compute_executor.shutdown(wait=True)
    matcher._similarity_compute_executor = similarity_compute_executor

    first_result = matcher._batch_get_candidate_phash_items(mm_hashes, missing_features)
    second_result = matcher._batch_get_candidate_phash_items(mm_hashes, missing_features)

    expected = list(zip(mm_hashes, resized_tensors))
    assert first_result == expected
    assert second_result == expected
    assert similarity_compute_executor.map.call_count == 2

    matcher.shutdown()
    similarity_compute_executor.shutdown.assert_called_once_with(wait=True)


def test_score_candidates_batches_loads_and_reuses_similarity_executor() -> None:
    matcher = _make_matcher()
    grid = (1, 4, 4)
    metadata = ImagePixelTensorMetadata((16, 12), torch.float32, grid)
    candidate_mm_hashes = [f"candidate-{index}" for index in range(4)]
    for index, candidate_mm_hash in enumerate(candidate_mm_hashes):
        matcher.phash_match_results[candidate_mm_hash] = PhashMatchItem(
            image_pixel_tensor_metadata=metadata,
            current_phash=index,
            candidate_mm_hash_group_by_hamming={},
        )
        matcher._tensor_store.values[candidate_mm_hash] = torch.zeros(metadata.shape)

    similarity_compute_executor = MagicMock()
    similarity_compute_executor.map.side_effect = lambda fn, *iterables: map(fn, *iterables)
    matcher._similarity_compute_executor.shutdown(wait=True)
    matcher._similarity_compute_executor = similarity_compute_executor
    matcher._score_candidate = MagicMock(
        side_effect=lambda candidate_mm_hash, tensor, item_metadata, gray_plane: (
            0.99,
            candidate_mm_hash,
            1.0,
        )
    )

    scored = matcher._score_candidates(
        "query",
        candidate_mm_hashes,
        torch.zeros(1, 1, 4, 4),
        grid,
    )

    assert scored == [(0.99, candidate_mm_hash) for candidate_mm_hash in candidate_mm_hashes]
    assert matcher._tensor_store.batch_get_calls == [candidate_mm_hashes]
    similarity_compute_executor.map.assert_called_once()
    assert matcher.ssim_comparison_count == 4
    assert matcher.ssim_comparison_total_ms == 4.0
    assert matcher.resized_tensor_batch_get_count == 1

    matcher.shutdown()
    similarity_compute_executor.shutdown.assert_called_once_with(wait=True)


def test_batch_ensure_similarity_cache_available_rejects_misaligned_inputs() -> None:
    matcher = _make_matcher()

    with pytest.raises(ValueError, match="same number"):
        matcher.batch_ensure_similarity_cache_available(["query"], [])

    matcher.shutdown()


def test_phash_cache_survives_clear_step_state() -> None:
    matcher = _make_matcher()
    resized = torch.zeros(16, 12)
    grid = (1, 4, 4)

    with patch.object(matcher_module, "compute_phash", return_value=123) as compute:
        assert matcher._cache_or_compute_phash("query", resized, grid) == (123, grid)
        matcher.phash_match_results["query"] = PhashMatchItem(
            image_pixel_tensor_metadata=ImagePixelTensorMetadata(
                tuple(resized.shape), resized.dtype, grid
            ),
            current_phash=123,
            candidate_mm_hash_group_by_hamming={1: {"candidate": None}},
        )
        matcher.clear_step_state()
        assert matcher._cache_or_compute_phash("query", resized, grid) == (123, grid)

    compute.assert_called_once_with(resized, grid, merge_size=2)
    cached = matcher.phash_match_results["query"]
    assert cached.current_phash == 123
    assert cached.image_pixel_tensor_metadata.grid == grid
    assert cached.candidate_mm_hash_group_by_hamming == {}
    matcher.shutdown()


def test_phash_cache_rejects_reused_mm_hash_with_different_grid() -> None:
    matcher = _make_matcher()
    resized = torch.zeros(16, 12)
    original_grid = (1, 4, 4)

    with patch.object(matcher_module, "compute_phash", return_value=123) as compute:
        assert matcher._cache_or_compute_phash("query", resized, original_grid) == (123, original_grid)
        assert matcher._cache_or_compute_phash("query", resized, (1, 2, 8)) is None

    compute.assert_called_once_with(resized, original_grid, merge_size=2)
    matcher.shutdown()


def test_phash_cache_rejects_reused_mm_hash_with_different_tensor_shape() -> None:
    matcher = _make_matcher()
    original = torch.zeros(16, 12)
    different_shape = torch.zeros(16, 24)
    grid = (1, 4, 4)

    with patch.object(matcher_module, "compute_phash", return_value=123) as compute:
        assert matcher._cache_or_compute_phash("query", original, grid) == (123, grid)
        assert matcher._cache_or_compute_phash("query", different_shape, grid) is None

    compute.assert_called_once_with(original, grid, merge_size=2)
    matcher.shutdown()


def test_get_or_update_similarity_match_records_below_threshold_score() -> None:
    matcher = _make_matcher()
    resized = torch.arange(16 * 12, dtype=torch.float32).reshape(16, 12)
    item = _item(resized, (1, 4, 4))
    matcher.register("candidate", item)
    matcher.shutdown()
    matcher.prewarm_candidate_resized_tensors("query", item)

    with patch.object(matcher_module, "ssim_score", return_value=0.98):
        hit = matcher.get_or_update_similarity_match("query", item)

    assert hit is None
    assert matcher.phash_match_results["query"].candidate_mm_hash_group_by_hamming == {
        0: {"candidate": 0.98}
    }
    assert matcher.phash_match_results["candidate"].candidate_mm_hash_group_by_hamming == {
        0: {"query": 0.98}
    }


def test_get_or_update_similarity_match_rejects_different_grid() -> None:
    matcher = _make_matcher()
    resized = torch.zeros(16, 12)
    matcher.register("candidate", _item(resized, (1, 4, 4)))
    matcher.shutdown()
    query_item = _item(resized, (1, 2, 8))
    matcher.prewarm_candidate_resized_tensors("query", query_item)

    hit = matcher.get_or_update_similarity_match("query", query_item)

    assert hit is None


def test_memcache_resized_store_uses_host_transfer_directions() -> None:
    backend = MagicMock()
    backend.exists.return_value = [0]
    backend.put_from.return_value = 0
    tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    store = MemcacheResizedTensorStore(backend)

    store.put("hash", tensor)

    backend.put_from.assert_called_once_with(
        "resized_hash",
        tensor.data_ptr(),
        tensor.nbytes,
        MmcDirect.COPY_H2G.value,
    )

    key_info = MagicMock()
    key_info.size.return_value = tensor.nbytes
    backend.batch_get_key_info.return_value = [key_info]
    backend.batch_get_into_buffers.return_value = [0]
    metadata = ImagePixelTensorMetadata(tuple(tensor.shape), tensor.dtype, (1, 2, 4))

    output = store.get("hash", metadata)

    assert output is not None
    get_args = backend.batch_get_into_buffers.call_args.args
    assert get_args[0] == ["resized_hash"]
    assert get_args[2] == [tensor.nbytes]
    assert get_args[3] == MmcDirect.COPY_G2H.value


def test_memcache_resized_store_batch_get_preserves_partial_results() -> None:
    backend = MagicMock()
    first = torch.empty(2, 4, dtype=torch.float32)
    second = torch.empty(4, 4, dtype=torch.float32)
    first_info = MagicMock()
    first_info.size.return_value = first.nbytes
    second_info = MagicMock()
    second_info.size.return_value = second.nbytes
    backend.batch_get_key_info.return_value = [first_info, second_info]
    backend.batch_get_into_buffers.return_value = [0, 7]
    store = MemcacheResizedTensorStore(backend)

    outputs = store.batch_get(
        ["first", "second"],
        [
            ImagePixelTensorMetadata(tuple(first.shape), first.dtype, (1, 2, 4)),
            ImagePixelTensorMetadata(tuple(second.shape), second.dtype, (1, 4, 4)),
        ],
    )

    assert outputs[0] is not None
    assert outputs[0].shape == first.shape
    assert outputs[1] is None
    backend.batch_get_key_info.assert_called_once_with(["resized_first", "resized_second"])
    get_args = backend.batch_get_into_buffers.call_args.args
    assert get_args[0] == ["resized_first", "resized_second"]
    assert get_args[2] == [first.nbytes, second.nbytes]
    assert get_args[3] == MmcDirect.COPY_G2H.value
