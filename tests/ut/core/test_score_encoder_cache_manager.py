from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.config import EncoderCacheManagerConfig
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager

from vllm_ascend.ascend_config import is_score_encoder_cache_manager
from vllm_ascend.ec_manager.score_ec_manager import (
    CacheEntry,
    ScoreEncoderCacheManager,
)

SCORE_MANAGER_CLS = "vllm_ascend.ec_manager.score_ec_manager.ScoreEncoderCacheManager"


def _build_manager(
    *,
    npu_cache_size: int = 10,
    cpu_cache_size: int = 10,
) -> ScoreEncoderCacheManager:
    manager = ScoreEncoderCacheManager.__new__(ScoreEncoderCacheManager)
    EncoderCacheManager.__init__(manager, npu_cache_size)
    manager.npu_num_free_slots = npu_cache_size
    manager.npu_num_freeable_slots = npu_cache_size
    manager.cpu_cache_size = cpu_cache_size
    manager.cpu_num_free_slots = cpu_cache_size
    manager.cpu_num_freeable_slots = cpu_cache_size
    manager.npu_cache = {}
    manager.cpu_cache = {}
    manager.npu_freeable = {}
    manager.cpu_freeable = OrderedDict()
    manager.req_cnt = 0
    manager.watermark = 0.2
    manager.promote_percentile = 0.2
    manager.max_clock = 15
    manager.clock_decay_every = 64
    manager.promoting = []
    manager.cpu_get_encoder_mm_hashes = []
    manager.npu_freed = []
    manager.cpu_freed = []
    manager.alpha = 1
    manager.beta = 1
    manager.hardware_flops = 1
    return manager


def _build_request(request_id: str, mm_hash: str, num_embeds: int):
    request = MagicMock()
    request.request_id = request_id
    request.mm_features = [SimpleNamespace(identifier=mm_hash)]
    request.get_num_encoder_embeds.return_value = num_embeds
    return request


def test_qualified_class_name_resolves_score_manager():
    config = EncoderCacheManagerConfig(encoder_cache_manager_cls=SCORE_MANAGER_CLS)

    assert config.get_encoder_cache_manager_obj() is ScoreEncoderCacheManager
    assert is_score_encoder_cache_manager(SimpleNamespace(ec_manager_config=config))


def test_other_managers_do_not_enable_score_cache():
    vllm_config = SimpleNamespace(
        ec_manager_config=SimpleNamespace(get_encoder_cache_manager_obj=lambda: EncoderCacheManager)
    )

    assert not is_score_encoder_cache_manager(vllm_config)


def test_factory_reads_score_parameters_from_vllm_config():
    manager_config = {
        "cpu_cache_slots": 12,
        "max_clock": 7,
        "clock_decay_every": 8,
        "watermark": 0.3,
        "promote_percentile": 0.4,
    }
    vision_config = SimpleNamespace(
        num_heads=2,
        hidden_size=4,
        intermediate_size=8,
    )
    vllm_config = SimpleNamespace(
        ec_manager_config=EncoderCacheManagerConfig(
            encoder_cache_manager_cls=SCORE_MANAGER_CLS,
            manager_config=manager_config,
        ),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(vision_config=vision_config)),
    )

    manager = ScoreEncoderCacheManager.create_manager(
        cache_size=10,
        vllm_config=vllm_config,
    )

    assert manager.cpu_cache_size == 12
    assert manager.max_clock == 7
    assert manager.clock_decay_every == 8
    assert manager.watermark == 0.3
    assert manager.promote_percentile == 0.4


def test_cpu_evict_preserves_npu_residency():
    manager = _build_manager(npu_cache_size=2, cpu_cache_size=4)
    entry = CacheEntry(
        mm_hash="shared",
        freq=1,
        clock=1,
        num_embeds=2,
        cal_cost=1,
    )
    manager.cached = {"shared": set()}
    manager.npu_cache = {"shared": entry}
    manager.cpu_cache = {"shared": entry}
    manager.npu_freeable = {"shared": entry}
    manager.cpu_freeable = OrderedDict([("shared", entry)])
    manager.cpu_num_free_slots = 2
    manager.cpu_num_freeable_slots = 4
    manager.npu_num_free_slots = 0
    manager.npu_num_freeable_slots = 2

    request = _build_request("request", "new", 3)

    assert manager.can_allocate(request, 0, 3, 0)
    manager._check_invariant()
    assert "shared" in manager.cached
    assert "shared" in manager.npu_cache
    assert "shared" not in manager.cpu_cache
    assert manager.get_freed_mm_hashes() == []

    metadata = manager.get_manager_metadata()
    assert metadata.npu_freed == []
    assert metadata.cpu_freed == ["shared"]
    assert manager.cpu_freed == []


def test_npu_evict_removes_last_residency():
    manager = _build_manager(npu_cache_size=2)
    entry = CacheEntry(
        mm_hash="npu-only",
        freq=1,
        clock=1,
        num_embeds=2,
        cal_cost=1,
    )
    manager.cached = {"npu-only": set()}
    manager.npu_cache = {"npu-only": entry}
    manager.npu_num_free_slots = 0

    manager.evict_from_npu(entry)
    manager._check_invariant()

    assert "npu-only" not in manager.cached
    assert "npu-only" not in manager.npu_cache
    assert manager.npu_freed == ["npu-only"]
    assert manager.npu_num_free_slots == 2
    assert entry.clock == 0


def test_rejects_encoder_output_larger_than_cpu_cache():
    manager = _build_manager(cpu_cache_size=2)
    request = _build_request("request", "too-large", 3)

    with pytest.raises(
        ValueError,
        match="manager_config.cpu_cache_slots",
    ):
        manager.can_allocate(request, 0, 3, 0)


def test_allocate_reuse_promote_free_and_reset_preserve_invariants():
    manager = _build_manager(npu_cache_size=4, cpu_cache_size=4)
    request = _build_request("request", "image", 2)

    assert manager.can_allocate(request, 0, 2, 0)
    manager.allocate(request, 0)
    manager._check_invariant()

    manager.free_encoder_input(request, 0)
    manager._check_invariant()

    assert manager.check_and_update_cache(request, 0)
    manager._check_invariant()
    assert "image" in manager.npu_cache

    metadata = manager.get_manager_metadata()
    assert metadata.promoting_mm_hashes == ["image"]
    assert manager.get_manager_metadata().promoting_mm_hashes == []

    manager.free_encoder_input(request, 0)
    manager._check_invariant()

    manager.reset()
    manager._check_invariant()
    metadata = manager.get_manager_metadata()
    assert metadata.npu_freed == []
    assert metadata.cpu_freed == []


def test_on_request_does_not_run_full_invariant_check():
    manager = _build_manager()
    manager.req_cnt = 999
    manager._check_invariant = MagicMock()

    manager.on_request()

    manager._check_invariant.assert_not_called()


def test_should_promote_caps_watermark_eviction_to_freeable_capacity():
    manager = _build_manager(npu_cache_size=10, cpu_cache_size=1)
    candidate = CacheEntry(
        mm_hash="candidate",
        freq=2,
        clock=1,
        num_embeds=1,
        cal_cost=1,
    )
    victim = CacheEntry(
        mm_hash="victim",
        freq=1,
        clock=1,
        num_embeds=1,
        cal_cost=1,
    )
    pinned = CacheEntry(
        mm_hash="pinned",
        freq=1,
        clock=1,
        num_embeds=9,
        cal_cost=1,
    )
    manager.cached = {
        "candidate": {"request"},
        "victim": set(),
        "pinned": {"other-request"},
    }
    manager.cpu_cache = {"candidate": candidate}
    manager.cpu_num_free_slots = 0
    manager.cpu_num_freeable_slots = 0
    manager.npu_cache = {
        "victim": victim,
        "pinned": pinned,
    }
    manager.npu_freeable = {"victim": victim}
    manager.npu_num_free_slots = 0
    manager.npu_num_freeable_slots = 1
    manager.watermark = 0.2
    manager.promote_percentile = 0

    assert manager.should_promote("candidate")
    manager._check_invariant()
    assert manager.npu_freed == ["victim"]
    assert manager.npu_num_free_slots == 1


def test_cpu_promotion_score_ignores_clock():
    manager = _build_manager(npu_cache_size=1, cpu_cache_size=1)
    candidate = CacheEntry(
        mm_hash="candidate",
        freq=4,
        clock=manager.max_clock,
        num_embeds=1,
        cal_cost=1,
    )
    victim = CacheEntry(
        mm_hash="victim",
        freq=5,
        clock=0,
        num_embeds=1,
        cal_cost=1,
    )
    manager.cached = {
        "candidate": {"request"},
        "victim": set(),
    }
    manager.cpu_cache = {"candidate": candidate}
    manager.cpu_num_free_slots = 0
    manager.cpu_num_freeable_slots = 0
    manager.npu_cache = {"victim": victim}
    manager.npu_freeable = {"victim": victim}
    manager.npu_num_free_slots = 0
    manager.npu_num_freeable_slots = 1
    manager.promote_percentile = 0

    assert not manager.should_promote("candidate")
    manager._check_invariant()
    assert "victim" in manager.npu_cache
    assert manager.npu_freed == []


def test_cpu_temporary_hit_does_not_get_npu_clock():
    manager = _build_manager(npu_cache_size=1, cpu_cache_size=2)
    first_request = _build_request("first", "candidate", 1)

    assert manager.can_allocate(first_request, 0, 1, 0)
    manager.allocate(first_request, 0)
    manager.free_encoder_input(first_request, 0)

    blocker = CacheEntry(
        mm_hash="blocker",
        freq=1,
        clock=manager.max_clock,
        num_embeds=1,
        cal_cost=1,
    )
    manager.cached["blocker"] = {"active-request"}
    manager.npu_cache["blocker"] = blocker
    manager.npu_num_free_slots = 0
    manager.npu_num_freeable_slots = 0

    second_request = _build_request("second", "candidate", 1)
    assert manager.check_and_update_cache(second_request, 0)
    manager._check_invariant()

    candidate = manager.cpu_cache["candidate"]
    assert candidate.clock == 0
    assert "candidate" not in manager.npu_cache
    assert manager.cpu_get_encoder_mm_hashes == ["candidate"]


def test_clock_tracks_npu_residency_lifecycle():
    manager = _build_manager(npu_cache_size=2, cpu_cache_size=2)
    first_request = _build_request("first", "image", 1)

    assert manager.can_allocate(first_request, 0, 1, 0)
    manager.allocate(first_request, 0)
    entry = manager.cpu_cache["image"]
    assert entry.clock == 0

    manager.free_encoder_input(first_request, 0)
    second_request = _build_request("second", "image", 1)
    assert manager.check_and_update_cache(second_request, 0)
    manager._check_invariant()
    assert entry.clock == manager.max_clock

    entry.clock = 1
    third_request = _build_request("third", "image", 1)
    assert manager.check_and_update_cache(third_request, 0)
    manager._check_invariant()
    assert entry.clock == manager.max_clock

    manager.free_encoder_input(second_request, 0)
    manager.free_encoder_input(third_request, 0)
    manager._check_invariant()
    manager.npu_freeable.pop("image")
    manager.evict_from_npu(entry)
    manager._check_invariant()
    assert entry.clock == 0


@pytest.mark.parametrize("seq_len", [1, 4, 16])
def test_theory_cost_is_normalized_by_storage_cost(seq_len):
    manager = _build_manager()
    manager.alpha = 2
    manager.beta = 3
    manager.hardware_flops = 4

    expected = 32 * (manager.alpha * seq_len + manager.beta) / manager.hardware_flops

    assert manager.cal_theory_cost_storage_cost(seq_len) == pytest.approx(expected)
