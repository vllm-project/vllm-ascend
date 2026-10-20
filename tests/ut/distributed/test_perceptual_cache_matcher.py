import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.distributed.ec_transfer import memcache_resized_tensor_store as resized_store_module
from vllm_ascend.distributed.ec_transfer import perceptual_ssim_cache_matcher as matcher_module
from vllm_ascend.distributed.ec_transfer.memcache_resized_tensor_store import (
    MemcacheResizedTensorStore,
    ResizedTensorLoadRequest,
)
from vllm_ascend.distributed.ec_transfer.perceptual_ssim_cache_matcher import (
    PhashSSIMCacheMatcher,
    SimilarityCacheItem,
)
from vllm_ascend.distributed.ec_transfer.similarity_cache_config import (
    MIN_RESIZED_IMAGE_RECV_BUFFER_SIZE,
    SimilarityCacheConfig,
)
from vllm_ascend.distributed.ec_transfer.phash import _dct_matrix
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import MmcDirect


class _InMemoryResizedTensorStore:
    def __init__(self) -> None:
        self.values: dict[str, torch.Tensor] = {}

    @staticmethod
    def get_resized_key(
        mm_hash: str,
        grid: tuple[int, ...],
        shape: tuple[int, ...] | torch.Size,
        dtype: torch.dtype,
    ) -> str:
        grid_suffix = "x".join(str(dimension) for dimension in grid[:3])
        shape_suffix = "x".join(str(dimension) for dimension in shape)
        dtype_suffix = str(dtype).removeprefix("torch.")
        return (
            f"resized_{mm_hash}_grid:{grid_suffix}_shape:{shape_suffix}_"
            f"{dtype_suffix}_test-model"
        )

    @staticmethod
    def get_mm_hash_from_resized_key(key: str) -> str | None:
        if not key.startswith("resized_"):
            return None
        mm_hash, separator, _ = key[len("resized_") :].partition("_grid:")
        return mm_hash if separator else None

    @staticmethod
    def get_grid_from_resized_key(key: str) -> tuple[int, int, int] | None:
        _, separator, suffix = key.partition("_grid:")
        if not separator:
            return None
        grid_text, separator, _ = suffix.partition("_shape:")
        if not separator:
            return None
        return tuple(int(value) for value in grid_text.split("x"))

    def is_compatible_resized_key(
        self,
        key: str,
        grid: tuple[int, ...],
        tensor: torch.Tensor,
    ) -> bool:
        mm_hash = self.get_mm_hash_from_resized_key(key)
        return mm_hash is not None and key == self.get_resized_key(
            mm_hash, grid, tensor.shape, tensor.dtype
        )

    def put(
        self,
        mm_hash: str,
        tensor: torch.Tensor,
        grid: tuple[int, ...],
    ) -> bool:
        key = self.get_resized_key(mm_hash, grid, tensor.shape, tensor.dtype)
        self.values[key] = tensor.clone()
        return True

    def shutdown(self) -> None:
        pass


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
        model_id="test-model",
    )
    matcher._tensor_store = _InMemoryResizedTensorStore()
    return matcher


def test_matcher_owns_similarity_compute_executors() -> None:
    matcher_compute_executor = MagicMock()
    store_compute_executor = MagicMock()

    with (
        patch.object(
            matcher_module,
            "ThreadPoolExecutor",
            return_value=matcher_compute_executor,
        ) as matcher_executor_cls,
        patch.object(
            resized_store_module,
            "ThreadPoolExecutor",
            return_value=store_compute_executor,
        ) as store_executor_cls,
    ):
        matcher = PhashSSIMCacheMatcher(
            MagicMock(),
            SimilarityCacheConfig(enabled=True, matcher="phash_ssim"),
            merge_size=2,
            model_id="test-model",
        )

    matcher_executor_cls.assert_called_once_with(
        max_workers=matcher_module._MAX_SIMILARITY_COMPUTE_WORKERS,
        thread_name_prefix="ec-similarity-compute",
    )
    store_executor_cls.assert_called_once_with(
        max_workers=matcher_module._MAX_SIMILARITY_COMPUTE_WORKERS,
        thread_name_prefix="ec-similarity-compute",
    )
    assert matcher._similarity_compute_executor is matcher_compute_executor
    assert matcher._tensor_store._similarity_compute_executor is store_compute_executor

    matcher.shutdown()

    matcher_compute_executor.shutdown.assert_called_once_with(wait=True)
    store_compute_executor.shutdown.assert_called_once_with(wait=True)


def test_dct_matrix_uses_frequency_rows() -> None:
    matrix = _dct_matrix()

    assert torch.allclose(matrix[0], torch.full((32,), 1.0 / math.sqrt(2.0)))
    expected_first_frequency = torch.cos(math.pi / 32 * (torch.arange(32) + 0.5))
    assert torch.allclose(matrix[1], expected_first_frequency)


def test_cached_phash_is_none_without_prewarmed_result() -> None:
    matcher = _make_matcher()
    item = _item(torch.zeros(16, 12), (1, 4, 4))

    assert matcher.get_cached_phash("query") is None
    matcher.shutdown()


def test_batch_ensure_similarity_cache_available_uses_ranked_direct_ssim_hit() -> None:
    matcher = _make_matcher()
    matcher.similarity_cache["query"] = SimilarityCacheItem(
        phash=123,
        ssim_similar_items=matcher._sort_ssim_similar_items(
            {"lower": 0.991, "best": 0.999}
        ),
    )
    matcher._backend.reset_mock()

    matcher.batch_ensure_similarity_cache_available(
        ["query"],
        [SimpleNamespace(data=object())],
    )

    matcher._backend.exists.assert_called_once_with(["best"])
    matcher.shutdown()


def test_record_similarity_score_caches_every_calculated_result() -> None:
    matcher = _make_matcher()
    matcher.similarity_cache["query"] = SimilarityCacheItem(phash=100)
    matcher.similarity_cache["high"] = SimilarityCacheItem(phash=101)
    matcher.similarity_cache["low"] = SimilarityCacheItem(phash=102)

    matcher._record_similarity_score(
        "query",
        [
            (0.80, "low"),
            (0.99, "high"),
            (0.70, "not-indexed"),
        ],
    )

    assert list(
        matcher.similarity_cache["query"].ssim_similar_items.items()
    ) == [
        ("high", 0.99),
        ("low", 0.80),
        ("not-indexed", 0.70),
    ]
    assert matcher.similarity_cache["high"].ssim_similar_items == {
        "query": 0.99
    }
    assert matcher.similarity_cache["low"].ssim_similar_items == {
        "query": 0.80
    }
    matcher.shutdown()


def test_batch_ensure_similarity_cache_available_computes_and_persists_missing_phashes() -> None:
    matcher = _make_matcher()
    resized = torch.arange(16 * 12, dtype=torch.float32).reshape(16, 12)
    item = _item(resized, (1, 4, 4))
    mm_hashes = [f"query-{index}" for index in range(4)]

    with patch.object(matcher_module, "compute_phash", return_value=123):
        matcher.batch_ensure_similarity_cache_available(
            mm_hashes,
            [SimpleNamespace(data=item)] * len(mm_hashes),
        )

    assert [matcher.similarity_cache[mm_hash].phash for mm_hash in mm_hashes] == [
        123
    ] * len(mm_hashes)
    expected_keys = [
        matcher._tensor_store.get_resized_key(
            mm_hash, (1, 4, 4), resized.shape, resized.dtype
        )
        for mm_hash in mm_hashes
    ]
    assert matcher.phash_to_resized_mm_hashes[123] == expected_keys
    assert set(matcher._tensor_store.values) == set(expected_keys)
    matcher.shutdown()


def test_batch_ensure_similarity_cache_available_rejects_misaligned_inputs() -> None:
    matcher = _make_matcher()

    with pytest.raises(ValueError, match="same number"):
        matcher.batch_ensure_similarity_cache_available(["query"], [])

    matcher.shutdown()


def test_memcache_resized_store_put_uses_host_transfer_direction() -> None:
    backend = MagicMock()
    backend.exists.return_value = [0]
    backend.put_from.return_value = 0
    tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    grid = (1, 1, 2)
    store = MemcacheResizedTensorStore(
        backend,
        2,
        "test-model",
        MIN_RESIZED_IMAGE_RECV_BUFFER_SIZE,
    )
    expected_key = store.get_resized_key("hash", grid, tensor.shape, tensor.dtype)

    store.put("hash", tensor, grid)

    backend.put_from.assert_called_once_with(
        expected_key,
        tensor.data_ptr(),
        tensor.nbytes,
        MmcDirect.COPY_H2G.value,
    )
    backend.register_buffer.assert_called_once_with(
        [tensor.data_ptr()],
        [tensor.nbytes],
    )
    backend.unregister_buffer.assert_called_once_with(
        [tensor.data_ptr()],
        [tensor.nbytes],
    )


def test_memcache_resized_store_key_includes_shape_dtype_and_model() -> None:
    store = MemcacheResizedTensorStore(
        MagicMock(), 2, "Qwen/Qwen3-VL-8B-Instruct"
    )
    grid = (1, 10, 10)

    resized_key = store.get_resized_key(
        "image-hash", grid, (100, 99), torch.float32
    )
    assert resized_key == (
        "resized_image-hash_grid:1x10x10_shape:100x99_"
        "float32_Qwen%2FQwen3-VL-8B-Instruct"
    )
    assert store.get_mm_hash_from_resized_key(resized_key) == "image-hash"
    assert store.get_grid_from_resized_key(resized_key) == grid
    assert store.get_resized_key("image-hash", grid, (99, 100), torch.float32) != (
        store.get_resized_key("image-hash", grid, (100, 99), torch.float32)
    )
    assert store.get_resized_key("image-hash", grid, (100, 99), torch.float16) != (
        store.get_resized_key("image-hash", grid, (100, 99), torch.float32)
    )
    assert store.get_resized_key("image-hash", (1, 5, 20), (100, 99), torch.float32) != (
        store.get_resized_key("image-hash", grid, (100, 99), torch.float32)
    )
    other_model_store = MemcacheResizedTensorStore(MagicMock(), 2, "other-model")
    assert other_model_store.get_resized_key(
        "image-hash", grid, (100, 99), torch.float32
    ) != store.get_resized_key("image-hash", grid, (100, 99), torch.float32)


def test_memcache_resized_store_calculates_ssim_in_parallel_in_key_order() -> None:
    store = MemcacheResizedTensorStore(MagicMock(), 2, "test-model")
    current = torch.zeros(16, 12, dtype=torch.float32)
    grid = (1, 4, 4)
    candidate_tensors = [torch.full_like(current, index) for index in range(4)]
    resized_keys = [
        store.get_resized_key(
            f"candidate-{index}",
            grid,
            candidate.shape,
            candidate.dtype,
        )
        for index, candidate in enumerate(candidate_tensors)
    ]
    request_batch = [
        ResizedTensorLoadRequest(
            output_index=index,
            resized_key=resized_key,
            nbytes=candidate.nbytes,
        )
        for index, (resized_key, candidate) in enumerate(
            zip(resized_keys, candidate_tensors)
        )
    ]

    similarity_compute_executor = MagicMock()
    similarity_compute_executor.map.side_effect = (
        lambda function, *iterables: map(function, *iterables)
    )
    store._similarity_compute_executor.shutdown(wait=True)
    store._similarity_compute_executor = similarity_compute_executor
    store._score_candidate = MagicMock(
        side_effect=lambda current_tensor, resized_key, candidate_tensor: (
            float(candidate_tensor[0, 0]),
            store.get_mm_hash_from_resized_key(resized_key),
        )
    )

    results = store.calculate_ssim_results(
        current,
        candidate_tensors,
        request_batch,
    )

    assert results == [
        (float(index), f"candidate-{index}") for index in range(4)
    ]
    similarity_compute_executor.map.assert_called_once()
    map_args = similarity_compute_executor.map.call_args.args
    assert len(map_args[1]) == 4
    assert all(plane.shape == (1, 1, 4, 4) for plane in map_args[1])
    assert map_args[2] == resized_keys
    assert map_args[3] == candidate_tensors
    store.shutdown()
    similarity_compute_executor.shutdown.assert_called_once_with(wait=True)


def test_regular_ssim_batches_stop_after_first_qualified_batch() -> None:
    store = MemcacheResizedTensorStore(MagicMock(), 2, "test-model")
    current = torch.zeros(16, 12, dtype=torch.float32)
    request_batches = [
        [ResizedTensorLoadRequest(0, "first", current.nbytes)],
        [
            ResizedTensorLoadRequest(1, "second", current.nbytes),
            ResizedTensorLoadRequest(2, "best", current.nbytes),
        ],
        [ResizedTensorLoadRequest(3, "must-not-run", current.nbytes)],
    ]
    store._ensure_recv_buffer = MagicMock(return_value=MagicMock())
    store._load_batch_into_buffer = MagicMock()
    store.calculate_ssim_results = MagicMock(
        side_effect=[
            [(0.90, "first")],
            [(0.95, "second"), (0.99, "best")],
        ]
    )

    results = store.select_ssim_from_regular_batches(
        request_batches,
        current,
        ssim_threshold=0.90,
    )

    assert results == [
        (0.99, "best"),
        (0.95, "second"),
        (0.90, "first"),
    ]
    assert store._load_batch_into_buffer.call_count == 2
    assert store.calculate_ssim_results.call_count == 2
    store.shutdown()


def test_oversized_ssim_batches_stop_and_unregister_temporary_buffer() -> None:
    backend = MagicMock()
    store = MemcacheResizedTensorStore(backend, 2, "test-model")
    current = torch.zeros(16, 12, dtype=torch.float32)
    request_batches = [
        [ResizedTensorLoadRequest(0, "first", current.nbytes)],
        [ResizedTensorLoadRequest(1, "best", current.nbytes)],
        [ResizedTensorLoadRequest(2, "must-not-run", current.nbytes)],
    ]
    temporary_buffer = MagicMock()
    temporary_buffer.data_ptr.return_value = 1234
    temporary_buffer.nbytes = current.nbytes
    store._load_batch_into_buffer = MagicMock()
    store.calculate_ssim_results = MagicMock(
        side_effect=[[(0.90, "first")], [(0.99, "best")]]
    )

    with patch.object(resized_store_module.torch, "empty", return_value=temporary_buffer):
        results = store.select_ssim_from_oversized_batches(
            request_batches,
            current,
            ssim_threshold=0.90,
        )

    assert results == [(0.99, "best"), (0.90, "first")]
    assert store._load_batch_into_buffer.call_count == 2
    assert store.calculate_ssim_results.call_count == 2
    backend.register_buffer.assert_called_once_with([1234], [current.nbytes])
    backend.unregister_buffer.assert_called_once_with([1234], [current.nbytes])
    store.shutdown()


def test_memcache_resized_store_builds_capacity_batches_in_order() -> None:
    requests = [
        ResizedTensorLoadRequest(0, "first", 4),
        ResizedTensorLoadRequest(1, "second", 6),
        ResizedTensorLoadRequest(2, "third", 5),
    ]

    batches = MemcacheResizedTensorStore._build_capacity_batches(
        requests,
        capacity_bytes=10,
    )

    assert batches == [requests[:2], requests[2:]]


def test_memcache_resized_store_classifies_named_load_requests() -> None:
    store = MemcacheResizedTensorStore(
        MagicMock(),
        2,
        "test-model",
        MIN_RESIZED_IMAGE_RECV_BUFFER_SIZE,
    )
    key_infos = [
        SimpleNamespace(size=lambda: MIN_RESIZED_IMAGE_RECV_BUFFER_SIZE),
        SimpleNamespace(size=lambda: MIN_RESIZED_IMAGE_RECV_BUFFER_SIZE + 1),
    ]

    regular_requests, oversized_requests = store.clarify_exist_requests(
        ["regular", "oversized"],
        key_infos,
    )

    assert regular_requests == [
        ResizedTensorLoadRequest(
            output_index=0,
            resized_key="regular",
            nbytes=MIN_RESIZED_IMAGE_RECV_BUFFER_SIZE,
        )
    ]
    assert oversized_requests == [
        ResizedTensorLoadRequest(
            output_index=1,
            resized_key="oversized",
            nbytes=MIN_RESIZED_IMAGE_RECV_BUFFER_SIZE + 1,
        )
    ]
    store.shutdown()
