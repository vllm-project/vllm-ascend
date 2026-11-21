import os
import tempfile
import time
from pathlib import Path

import pytest
import torch

from vllm_ascend.worker.mm_global_cache import GlobalMMEmbeddingCache


@pytest.fixture
def cache_dir():
    with tempfile.TemporaryDirectory() as path:
        yield Path(path)


def test_put_get_roundtrip(cache_dir):
    cache = GlobalMMEmbeddingCache(str(cache_dir), 1024 * 1024)
    value = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    cache.put("hash-a", value)
    cache.flush()
    loaded = cache.get("hash-a")
    assert loaded is not None
    torch.testing.assert_close(loaded, value)


def test_miss_and_corrupt_entry(cache_dir):
    cache = GlobalMMEmbeddingCache(str(cache_dir), 1024 * 1024)
    assert cache.get("missing") is None
    cache.put("bad", torch.ones(2))
    cache.flush()
    path = next(cache_dir.glob("*.pt"))
    path.write_bytes(b"not a tensor")
    assert cache.get("bad") is None
    assert not path.exists()

    cache.put("bad", torch.full((2,), 3.0))
    cache.flush()
    loaded = cache.get("bad")
    assert loaded is not None
    torch.testing.assert_close(loaded, torch.full((2,), 3.0))


def test_zero_capacity_disables_put(cache_dir):
    cache = GlobalMMEmbeddingCache(str(cache_dir), 0)
    cache.put("hash", torch.ones(2))
    cache.flush()
    assert not list(cache_dir.glob("*.pt"))
    assert cache.get("hash") is None


def test_lru_eviction(cache_dir):
    cache = GlobalMMEmbeddingCache(str(cache_dir), 2000)
    big = torch.ones(600, dtype=torch.bfloat16)
    cache.put("a", big)
    cache.put("b", big)
    cache.put("c", big)
    cache.flush()
    assert len(list(cache_dir.glob("*.pt"))) < 3


def test_eviction_counts_files_written_by_other_processes(cache_dir):
    cache = GlobalMMEmbeddingCache(str(cache_dir), 2000)
    external = cache_dir / "external.pt"
    external.write_bytes(b"x" * 1500)
    cache.put("a", torch.ones(400, dtype=torch.bfloat16))
    cache.flush()
    assert not external.exists()


def test_stale_tmp_files_are_removed_but_recent_ones_are_kept(cache_dir):
    stale = cache_dir / "stale.tmp"
    fresh = cache_dir / "fresh.tmp"
    stale.write_bytes(b"stale")
    fresh.write_bytes(b"fresh")
    old = time.time() - 7200
    os.utime(stale, (old, old))

    GlobalMMEmbeddingCache(str(cache_dir), 1024 * 1024)

    assert not stale.exists()
    assert fresh.exists()
