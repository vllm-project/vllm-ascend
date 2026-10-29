import math
import random
from concurrent.futures import wait
from threading import Event, Thread, get_ident
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.distributed.ec_transfer import memcache_resized_tensor_store as resized_store_module
from vllm_ascend.distributed.ec_transfer import perceptual_ssim_cache_matcher as matcher_module
from vllm_ascend.distributed.ec_transfer import phash as phash_module
from vllm_ascend.distributed.ec_transfer.memcache_resized_tensor_store import (
    MemcacheResizedTensorStore,
    ResizedTensorLoadRequest,
)
from vllm_ascend.distributed.ec_transfer.perceptual_ssim_cache_matcher import (
    PhashSSIMCacheMatcher,
    SimilarityCacheItem,
)
from vllm_ascend.distributed.ec_transfer.phash import PHASH_NUM_BANDS, _dct_matrix
from vllm_ascend.distributed.ec_transfer.similarity_cache_config import (
    MIN_RESIZED_IMAGE_RECV_BUFFER_SIZE,
    SimilarityCacheConfig,
)
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
        return f"resized_{mm_hash}_grid:{grid_suffix}_shape:{shape_suffix}_{dtype_suffix}_test-model"

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
        return mm_hash is not None and key == self.get_resized_key(mm_hash, grid, tensor.shape, tensor.dtype)

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


def _make_matcher(max_hamming: int = PHASH_NUM_BANDS - 1) -> PhashSSIMCacheMatcher:
    backend = MagicMock()
    backend.exists.side_effect = lambda keys: [1] * len(keys)
    matcher = PhashSSIMCacheMatcher(
        backend,
        SimilarityCacheConfig(enabled=True, matcher="phash_ssim", max_hamming=max_hamming),
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

    assert matcher.get_cached_phash("query") is None
    matcher.shutdown()


def test_batch_ensure_similarity_cache_available_uses_ranked_direct_ssim_hit() -> None:
    matcher = _make_matcher()
    matcher.similarity_cache["query"] = SimilarityCacheItem(
        phash=123,
        ssim_similar_items=matcher._sort_ssim_similar_items({"lower": 0.991, "best": 0.999}),
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

    assert list(matcher.similarity_cache["query"].ssim_similar_items.items()) == [
        ("high", 0.99),
        ("low", 0.80),
        ("not-indexed", 0.70),
    ]
    assert matcher.similarity_cache["high"].ssim_similar_items == {"query": 0.99}
    assert matcher.similarity_cache["low"].ssim_similar_items == {"query": 0.80}
    matcher.shutdown()


def test_batch_ensure_similarity_cache_available_registers_phash_without_persisting() -> None:
    matcher = _make_matcher()
    resized = torch.arange(16 * 12, dtype=torch.float32).reshape(16, 12)
    item = _item(resized, (1, 4, 4))
    mm_hashes = [f"query-{index}" for index in range(4)]

    with (
        patch.object(matcher_module, "compute_phash", return_value=123),
        patch.object(matcher._tensor_store, "put", wraps=matcher._tensor_store.put) as put,
    ):
        matcher.batch_ensure_similarity_cache_available(
            mm_hashes,
            [SimpleNamespace(data=item)] * len(mm_hashes),
        )

    assert [matcher.similarity_cache[mm_hash].phash for mm_hash in mm_hashes] == [123] * len(mm_hashes)
    put.assert_not_called()
    expected_keys = [
        matcher._tensor_store.get_resized_key(mm_hash, (1, 4, 4), resized.shape, resized.dtype) for mm_hash in mm_hashes
    ]
    assert set(matcher.phash_to_resized_mm_hashes[123]) == set(expected_keys)
    assert matcher._tensor_store.values == {}
    matcher.shutdown()


def test_batch_ensure_similarity_cache_available_rejects_misaligned_inputs() -> None:
    matcher = _make_matcher()

    with pytest.raises(ValueError, match="same number"):
        matcher.batch_ensure_similarity_cache_available(["query"], [])

    matcher.shutdown()


def test_persist_missed_resized_tensor_persists_only_after_matching() -> None:
    matcher = _make_matcher()
    resized = torch.arange(16 * 12, dtype=torch.float32).reshape(16, 12)
    feature = _item(resized, (1, 4, 4))
    expected_key = matcher._tensor_store.get_resized_key(
        "query",
        (1, 4, 4),
        resized.shape,
        resized.dtype,
    )

    with patch.object(matcher_module, "compute_phash", return_value=123):
        matcher.persist_missed_resized_tensor("query", feature)

    assert matcher.get_cached_phash("query") == 123
    assert matcher.phash_to_resized_mm_hashes[123] == [expected_key]
    assert matcher._tensor_store.values == {expected_key: resized}
    matcher.shutdown()


def test_persist_missed_resized_tensor_reuses_prewarmed_phash() -> None:
    matcher = _make_matcher()
    matcher.similarity_cache["query"] = SimilarityCacheItem(phash=123)
    resized = torch.zeros(16, 12)
    expected_key = matcher._tensor_store.get_resized_key(
        "query",
        (1, 4, 4),
        resized.shape,
        resized.dtype,
    )

    with patch.object(matcher_module, "compute_phash") as compute:
        matcher.persist_missed_resized_tensor("query", _item(resized, (1, 4, 4)))

    compute.assert_not_called()
    assert matcher.get_cached_phash("query") == 123
    assert matcher.phash_to_resized_mm_hashes[123] == [expected_key]
    assert set(matcher._tensor_store.values) == {expected_key}
    matcher.shutdown()


def _register_phash(matcher: PhashSSIMCacheMatcher, mm_hash: str, image_phash: int) -> str:
    tensor = torch.zeros(16, 12)
    grid = (1, 4, 4)
    with patch.object(matcher_module, "compute_phash", return_value=image_phash):
        matcher.persist_missed_resized_tensor(mm_hash, _item(tensor, grid))
    return matcher._tensor_store.get_resized_key(mm_hash, grid, tensor.shape, tensor.dtype)


class _DirectLookupOnly(dict):
    """Fail if lookup visits unrelated reverse-index entries."""

    def __iter__(self):
        raise AssertionError("reverse index must not be traversed")

    def items(self):
        raise AssertionError("reverse index must not be traversed")

    def keys(self):
        raise AssertionError("reverse index must not be traversed")

    def values(self):
        raise AssertionError("reverse index must not be traversed")


def test_zero_hamming_uses_direct_lookup_without_bands_or_hamming() -> None:
    matcher = _make_matcher(max_hamming=0)
    try:
        with (
            patch.object(matcher_module, "bands", create=True) as bands_spy,
            patch.object(matcher_module, "hamming") as hamming_spy,
        ):
            second_key = _register_phash(matcher, "second", 0)
            first_key = _register_phash(matcher, "first", 0)
            _register_phash(matcher, "unrelated", 1 << 63)
            matcher.phash_to_resized_mm_hashes = _DirectLookupOnly(matcher.phash_to_resized_mm_hashes)

            matches = matcher.find_resized_mm_hashes_by_hamming(0)
            assert matches == [second_key, first_key]
            matches.clear()
            assert matcher.find_resized_mm_hashes_by_hamming(0) == [second_key, first_key]
            assert matcher.find_resized_mm_hashes_by_hamming(12345) is None
            assert matcher._phash_band_index == {}
            bands_spy.assert_not_called()
            hamming_spy.assert_not_called()
    finally:
        matcher.shutdown()


@pytest.mark.parametrize("score", [0.8, 0.99, 0.999, float("nan")])
def test_exact_phash_collision_still_requires_ssim_threshold(score: float) -> None:
    matcher = _make_matcher(max_hamming=0)
    query = torch.ones(16, 12)
    try:
        _register_phash(matcher, "query", 123)
        candidate = _register_phash(matcher, "candidate", 123)
        with patch.object(matcher, "_get_ssim_score_list", return_value=[(score, "candidate")]):
            result = matcher.find_and_cache_matched_ssim("query", _item(query, (1, 4, 4)), [candidate])
        assert result == ("candidate" if score > matcher._config.ssim_threshold else None)
        assert "candidate" in matcher.similarity_cache["query"].ssim_similar_items
    finally:
        matcher.shutdown()


@pytest.mark.parametrize("max_hamming", range(PHASH_NUM_BANDS))
def test_phash_lookup_matches_ordered_brute_force(max_hamming: int) -> None:
    matcher = _make_matcher(max_hamming=max_hamming)
    rng = random.Random(20261020)
    queries = [0, 1 << 63, rng.getrandbits(64)]
    registered_hashes = [rng.getrandbits(64) for _ in range(48)]
    for query in queries:
        registered_hashes.extend(
            query ^ sum(1 << bit for bit in rng.sample(range(64), distance)) for distance in range(PHASH_NUM_BANDS + 1)
        )
    # One differing bit in every band is outside the supported thresholds.
    all_bands_differ = sum(1 << (index * (64 // PHASH_NUM_BANDS)) for index in range(PHASH_NUM_BANDS))
    registered_hashes.extend([all_bands_differ, 0, 0, 1 << 63])
    rng.shuffle(registered_hashes)
    expected_reverse: dict[int, list[str]] = {}
    try:
        for index, image_phash in enumerate(registered_hashes):
            key = _register_phash(matcher, f"image-{index}", image_phash)
            expected_reverse.setdefault(image_phash, []).append(key)
        for query in queries + registered_hashes:
            expected = [
                key
                for image_phash, keys in sorted(expected_reverse.items())
                if (query ^ image_phash).bit_count() <= max_hamming
                for key in keys
            ]
            assert matcher.find_resized_mm_hashes_by_hamming(query) == (expected or None)
        assert matcher.find_resized_mm_hashes_by_hamming(0xFFFFFFFFFFFFFFFF) is None
    finally:
        matcher.shutdown()


def test_bands_recall_only_shared_bands_and_filter_false_positives() -> None:
    matcher = _make_matcher(max_hamming=PHASH_NUM_BANDS - 1)
    try:
        near = _register_phash(matcher, "near", 1)
        exact = _register_phash(matcher, "exact", 0)
        far_hash = ((1 << 64) - 1) ^ ((1 << (64 // PHASH_NUM_BANDS)) - 1)
        _register_phash(matcher, "shared-band-far", far_hash)
        _register_phash(matcher, "no-shared-band", 0x0101010101010101)
        matcher.phash_to_resized_mm_hashes = _DirectLookupOnly(matcher.phash_to_resized_mm_hashes)
        with patch.object(matcher_module, "hamming", wraps=phash_module.hamming) as hamming_spy:
            assert matcher.find_resized_mm_hashes_by_hamming(0) == [exact, near]
        assert {call.args[1] for call in hamming_spy.call_args_list} == {0, 1, far_hash}
        assert hamming_spy.call_count == 3
    finally:
        matcher.shutdown()


def test_band_registration_is_idempotent_for_colliding_hashes() -> None:
    matcher = _make_matcher(max_hamming=PHASH_NUM_BANDS - 1)
    try:
        with patch.object(matcher_module, "bands", wraps=phash_module.bands, create=True) as bands_spy:
            first = _register_phash(matcher, "first", 0)
            second = _register_phash(matcher, "second", 0)
            assert _register_phash(matcher, "first", 0) == first
            high = _register_phash(matcher, "high", 1 << 63)
        assert bands_spy.call_count == 2
        assert matcher.phash_to_resized_mm_hashes == {0: [first, second], 1 << 63: [high]}
        expected_bands: dict[tuple[int, int], set[int]] = {}
        for image_phash in (0, 1 << 63):
            for band in phash_module.bands(image_phash):
                expected_bands.setdefault(band, set()).add(image_phash)
        assert matcher._phash_band_index == expected_bands
    finally:
        matcher.shutdown()


@pytest.mark.parametrize("put_raises", [False, True])
def test_failed_persistence_does_not_publish_phash_indexes(
    put_raises: bool,
) -> None:
    matcher = _make_matcher()

    def failed_put(*args) -> bool:
        assert matcher.phash_to_resized_mm_hashes == {}
        assert matcher._phash_band_index == {}
        assert matcher.similarity_cache == {}
        if put_raises:
            raise RuntimeError("persistence failed")
        return False

    try:
        with patch.object(matcher._tensor_store, "put", side_effect=failed_put):
            if put_raises:
                with pytest.raises(RuntimeError, match="persistence failed"):
                    _register_phash(matcher, "failed", 0)
            else:
                _register_phash(matcher, "failed", 0)
        assert matcher.phash_to_resized_mm_hashes == {}
        assert matcher._phash_band_index == {}
        assert matcher.similarity_cache == {}
    finally:
        matcher.shutdown()


def test_batch_hamming_lookup_memoizes_queries_and_preserves_multiplicity() -> None:
    matcher = _make_matcher(max_hamming=0)
    try:
        one = _register_phash(matcher, "one", 1)
        zero = _register_phash(matcher, "zero", 0)
        with patch.object(
            matcher,
            "find_resized_mm_hashes_by_hamming",
            wraps=matcher.find_resized_mm_hashes_by_hamming,
        ) as lookup_spy:
            misses, candidates = matcher._collect_candidate_resized_mm_hashes_by_hamming(
                [
                    ("query-one", 1),
                    ("query-zero", 0),
                    ("missing", 99),
                    ("query-one-again", 1),
                    ("missing-again", 99),
                ]
            )
        assert misses == ["missing", "missing-again"]
        assert candidates == [one, zero, one]
        assert [call.args[0] for call in lookup_spy.call_args_list] == [1, 0, 99]
    finally:
        matcher.shutdown()


@pytest.mark.parametrize("worker_raises", [False, True])
def test_parallel_phash_compute_waits_for_all_metadata_writes_before_return_or_error(
    worker_raises: bool,
) -> None:
    matcher = _make_matcher()
    count = matcher_module._PARALLEL_SIMILARITY_COMPUTE_THRESHOLD + 1
    mm_hashes = [f"query-{index}" for index in range(count)]
    feature = SimpleNamespace(data=_item(torch.zeros(16, 12), (1, 4, 4)))
    features = dict.fromkeys(mm_hashes, feature)
    worker_started = {mm_hash: Event() for mm_hash in mm_hashes}
    release_workers = Event()
    waiting_or_returned = Event()
    returned = Event()
    writes: list[str] = []
    errors: list[BaseException] = []
    worker_error = RuntimeError("worker failed")

    def compute(mm_hash, feature) -> None:
        worker_started[mm_hash].set()
        if worker_raises and mm_hash == mm_hashes[0]:
            raise worker_error
        assert release_workers.wait(5), "test did not release workers"
        writes.append(mm_hash)

    def observed_wait(futures):
        assert len(futures) == count
        waiting_or_returned.set()
        return wait(futures)

    def run() -> None:
        try:
            matcher._compute_and_cache_missing_phashes(mm_hashes, features)
        except BaseException as error:
            errors.append(error)
        finally:
            returned.set()
            waiting_or_returned.set()

    caller = Thread(target=run)
    try:
        with (
            patch.object(
                matcher,
                "_compute_phash_and_cache_metadata",
                side_effect=compute,
            ),
            patch.object(matcher_module, "wait", side_effect=observed_wait, create=True),
        ):
            caller.start()
            try:
                assert waiting_or_returned.wait(5), "caller did not start"
                assert not returned.is_set()
                assert all(started.wait(5) for started in worker_started.values())
                assert writes == []
            finally:
                release_workers.set()
                caller.join(5)
            assert not caller.is_alive()
            assert returned.is_set()
        assert errors == ([worker_error] if worker_raises else [])
        assert set(writes) == set(mm_hashes[1:] if worker_raises else mm_hashes)
    finally:
        release_workers.set()
        matcher.shutdown()


@pytest.mark.parametrize("count", [1, 3, 4])
def test_cpu_preprocessing_schedules_by_batch_size(count: int) -> None:
    matcher = _make_matcher()
    mm_hashes = [f"query-{index}" for index in range(count)]
    features = {
        mm_hash: SimpleNamespace(data=_item(torch.zeros(16, 12), (1, 4, 4))) for mm_hash in mm_hashes
    }
    caller_thread = get_ident()
    compute_threads: list[int] = []
    try:
        with patch.object(
            matcher,
            "_compute_phash_and_cache_metadata",
            side_effect=lambda *args: compute_threads.append(get_ident()),
        ):
            matcher._compute_and_cache_missing_phashes(mm_hashes, features)
        assert len(compute_threads) == count
        serial = count <= matcher_module._PARALLEL_SIMILARITY_COMPUTE_THRESHOLD
        assert all((thread == caller_thread) == serial for thread in compute_threads)
    finally:
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
    store = MemcacheResizedTensorStore(MagicMock(), 2, "Qwen/Qwen3-VL-8B-Instruct")
    grid = (1, 10, 10)

    resized_key = store.get_resized_key("image-hash", grid, (100, 99), torch.float32)
    assert resized_key == ("resized_image-hash_grid:1x10x10_shape:100x99_float32_Qwen%2FQwen3-VL-8B-Instruct")
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
    assert other_model_store.get_resized_key("image-hash", grid, (100, 99), torch.float32) != store.get_resized_key(
        "image-hash", grid, (100, 99), torch.float32
    )


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
        ResizedTensorLoadRequest(index, resized_key, candidate.nbytes)
        for index, (resized_key, candidate) in enumerate(zip(resized_keys, candidate_tensors))
    ]

    store._score_candidate = MagicMock(
        side_effect=lambda current_tensor, resized_key, candidate_tensor: (
            float(candidate_tensor[0, 0]),
            store.get_mm_hash_from_resized_key(resized_key),
        )
    )

    with patch.object(
        store._similarity_compute_executor, "submit", wraps=store._similarity_compute_executor.submit
    ) as submit_spy:
        results = store.calculate_ssim_results(
            current,
            candidate_tensors,
            request_batch,
        )

    assert results == [(float(index), f"candidate-{index}") for index in range(4)]
    assert submit_spy.call_count == 4
    assert all(call.args[1].shape == (1, 1, 4, 4) for call in submit_spy.call_args_list)
    assert [call.args[2] for call in submit_spy.call_args_list] == resized_keys
    assert all(call.args[3] is tensor for call, tensor in zip(submit_spy.call_args_list, candidate_tensors))
    store.shutdown()


def test_regular_ssim_batches_stop_after_first_qualified_batch() -> None:
    store = MemcacheResizedTensorStore(MagicMock(), 2, "test-model")
    current = torch.zeros(16, 12, dtype=torch.float32)
    request_batches = [
        [ResizedTensorLoadRequest(0, "first", current.nbytes)],
        [ResizedTensorLoadRequest(1, "second", current.nbytes), ResizedTensorLoadRequest(2, "best", current.nbytes)],
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
    store.calculate_ssim_results = MagicMock(side_effect=[[(0.90, "first")], [(0.99, "best")]])

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


@pytest.mark.parametrize("mode", ["regular", "oversized"])
@pytest.mark.parametrize("followup_hit", [False, True])
def test_ssim_batches_handle_empty_scores(mode: str, followup_hit: bool) -> None:
    store = MemcacheResizedTensorStore(MagicMock(), 2, "test-model")
    current = torch.zeros(16, 12)
    recv_buffer = torch.empty(current.nbytes, dtype=torch.uint8)
    requests = [[ResizedTensorLoadRequest(i, f"candidate-{i}", current.nbytes)] for i in range(3)]
    store._ensure_recv_buffer = MagicMock(return_value=recv_buffer)
    store._load_batch_into_buffer = MagicMock()
    scores = [[], [(0.999, "best")]] if followup_hit else [[], [], []]
    store.calculate_ssim_results = MagicMock(side_effect=scores)
    try:
        with patch.object(resized_store_module.torch, "empty", return_value=recv_buffer):
            result = getattr(store, f"select_ssim_from_{mode}_batches")(requests, current, 0.99)
        assert result == ([(0.999, "best")] if followup_hit else [])
        assert store.calculate_ssim_results.call_count == (2 if followup_hit else 3)
        if mode == "oversized":
            store._backend.unregister_buffer.assert_called_once_with([recv_buffer.data_ptr()], [recv_buffer.nbytes])
    finally:
        store.shutdown()


@pytest.mark.parametrize("first_score", [0.8, 0.995])
def test_regular_best_candidate_skips_oversized_loading(first_score: float) -> None:
    matcher = PhashSSIMCacheMatcher(
        MagicMock(), SimilarityCacheConfig(enabled=True), merge_size=2, model_id="test-model"
    )
    current = torch.zeros(16, 12)
    store = matcher._tensor_store
    regular = [ResizedTensorLoadRequest(i, key, current.nbytes) for i, key in enumerate(("first", "best"))]
    oversized = [ResizedTensorLoadRequest(2, "oversized", current.nbytes)]
    store.is_compatible_resized_key = MagicMock(return_value=True)
    store.batch_get_resized_key_infos = MagicMock(return_value=([], ["first", "best", "oversized"], [object()] * 3))
    store.clarify_exist_requests = MagicMock(return_value=(regular, oversized))
    store.build_request_batches = MagicMock(return_value=([regular], [oversized]))
    store._ensure_recv_buffer = MagicMock(return_value=MagicMock())
    store._load_batch_into_buffer = MagicMock()
    store.calculate_ssim_results = MagicMock(return_value=[(first_score, "first"), (0.999, "best")])
    store.select_ssim_from_oversized_batches = MagicMock(return_value=[])
    matcher._record_similarity_score = MagicMock()
    try:
        assert matcher.find_and_cache_matched_ssim(
            "current", _item(current, (1, 4, 4)), ["first", "best", "oversized"]
        ) == "best"
        store.select_ssim_from_oversized_batches.assert_not_called()
    finally:
        matcher.shutdown()


@pytest.mark.parametrize("mode", ["regular", "oversized"])
def test_ssim_batches_break_score_ties_by_key(mode: str) -> None:
    store = MemcacheResizedTensorStore(MagicMock(), 2, "test-model")
    current = torch.zeros(16, 12)
    recv_buffer = torch.empty(current.nbytes, dtype=torch.uint8)
    requests = [[ResizedTensorLoadRequest(i, key, current.nbytes) for i, key in enumerate(("z", "a"))]]
    store._ensure_recv_buffer = MagicMock(return_value=recv_buffer)
    store._load_batch_into_buffer = MagicMock()
    store.calculate_ssim_results = MagicMock(return_value=[(0.999, "z"), (0.999, "a")])
    try:
        with patch.object(resized_store_module.torch, "empty", return_value=recv_buffer):
            assert getattr(store, f"select_ssim_from_{mode}_batches")(requests, current, 0.99) == [
                (0.999, "a"),
                (0.999, "z"),
            ]
    finally:
        store.shutdown()



def test_four_band_lookup_keeps_distance_three_boundary():
    matcher = _make_matcher(max_hamming=3)
    try:
        hit = _register_phash(matcher, "three-bands", 0x0000000100010001)
        _register_phash(matcher, "four-bands", 0x0001000100010001)
        assert matcher.find_resized_mm_hashes_by_hamming(0) == [hit]
        assert sum(len(bucket) for bucket in matcher._phash_band_index.values()) == 2 * PHASH_NUM_BANDS
    finally:
        matcher.shutdown()


def _make_capacity_matcher(
    max_capacity: int,
    eviction_threshold: float,
    lower_watermark: float,
) -> PhashSSIMCacheMatcher:
    matcher = PhashSSIMCacheMatcher(
        MagicMock(),
        SimilarityCacheConfig(
            enabled=True,
            matcher="phash_ssim",
            max_capacity=max_capacity,
            eviction_threshold=eviction_threshold,
            lower_watermark=lower_watermark,
        ),
        merge_size=2,
        model_id="test-model",
    )
    matcher._tensor_store = _InMemoryResizedTensorStore()
    return matcher


def test_similarity_cache_capacity_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="max_capacity"):
        SimilarityCacheConfig(enabled=True, max_capacity=0)
    with pytest.raises(ValueError, match="max_capacity"):
        SimilarityCacheConfig(enabled=True, max_capacity=True)
    with pytest.raises(ValueError, match="eviction_threshold"):
        SimilarityCacheConfig(enabled=True, eviction_threshold=0.0)
    with pytest.raises(ValueError, match="lower_watermark"):
        SimilarityCacheConfig(enabled=True, lower_watermark=0.95)
    with pytest.raises(ValueError, match="lower_watermark"):
        SimilarityCacheConfig(enabled=True, lower_watermark=1.0, eviction_threshold=1.0)


def test_similarity_cache_random_eviction_syncs_reverse_and_band_index() -> None:
    matcher = _make_capacity_matcher(max_capacity=20, eviction_threshold=0.95, lower_watermark=0.5)
    random.seed(20261026)
    try:
        for index in range(19):
            _register_phash(matcher, f"image-{index}", 0x1000 + index)
        assert len(matcher.similarity_cache) == 10  # max_capacity * lower_watermark

        surviving_phashes = {item.phash for item in matcher.similarity_cache.values()}
        assert len(surviving_phashes) == 10
        assert set(matcher.phash_to_resized_mm_hashes) == surviving_phashes
        for mm_hash, item in matcher.similarity_cache.items():
            resized_keys = matcher.phash_to_resized_mm_hashes[item.phash]
            assert [
                MemcacheResizedTensorStore.get_mm_hash_from_resized_key(key) for key in resized_keys
            ] == [mm_hash]
        for phash_set in matcher._phash_band_index.values():
            assert phash_set
            assert phash_set <= surviving_phashes
    finally:
        matcher.shutdown()


def test_similarity_cache_keeps_growing_below_eviction_threshold() -> None:
    matcher = _make_capacity_matcher(max_capacity=20, eviction_threshold=0.95, lower_watermark=0.5)
    try:
        for index in range(18):
            _register_phash(matcher, f"image-{index}", 0x1000 + index)
        assert len(matcher.similarity_cache) == 18
        assert len(matcher.phash_to_resized_mm_hashes) == 18
    finally:
        matcher.shutdown()
