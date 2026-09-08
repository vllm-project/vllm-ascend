from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from transformers import CLIPVisionConfig, Qwen2_5_VLConfig
from vllm.config import EncoderCacheManagerConfig
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager

from vllm_ascend.ec_manager.score_ec_manager import (
    CacheEntry,
    ScoreEncoderCacheConfig,
    ScoreEncoderCacheManager,
)
from vllm_ascend.utils import is_score_encoder_cache_manager, vllm_version_is

SCORE_MANAGER_CLS = "vllm_ascend.ec_manager.score_ec_manager.ScoreEncoderCacheManager"


def _build_manager(
    *,
    npu_cache_size: int = 10,
    cpu_cache_size: int = 10,
) -> ScoreEncoderCacheManager:
    vision_config = SimpleNamespace(
        num_heads=1,
        hidden_size=1,
        intermediate_size=1,
    )
    vllm_config = SimpleNamespace(
        ec_manager_config=SimpleNamespace(manager_config={"cpu_cache_slots": cpu_cache_size}),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(vision_config=vision_config)),
    )
    return ScoreEncoderCacheManager(
        cache_size=npu_cache_size,
        vllm_config=vllm_config,
    )


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


@pytest.mark.skipif(
    vllm_version_is("0.27.1"),
    reason=("ScoreEncoderCacheManager configuration requires vllm-project/vllm#51251."),
)
@pytest.mark.parametrize(
    ("vision_config", "expected_attn_heads"),
    [
        (CLIPVisionConfig(num_attention_heads=3), 3),
        (Qwen2_5_VLConfig(vision_config={"num_heads": 4}).vision_config, 4),
    ],
)
def test_factory_reads_score_parameters_from_vllm_config(
    vision_config,
    expected_attn_heads,
):
    vllm_config = SimpleNamespace(
        ec_manager_config=EncoderCacheManagerConfig(
            encoder_cache_manager_cls=SCORE_MANAGER_CLS,
            manager_config={"cpu_cache_slots": 12},
        ),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(vision_config=vision_config)),
    )

    manager = ScoreEncoderCacheManager.create_manager(
        cache_size=10,
        vllm_config=vllm_config,
    )

    assert manager.cpu_cache_size == 12
    assert manager.attn_heads == expected_attn_heads
    manager._check_invariant()


def test_score_config_accepts_valid_boundary_values():
    config = ScoreEncoderCacheConfig.from_dict(
        {
            "cpu_cache_slots": 1,
            "max_clock": 0,
            "clock_decay_every": 1,
            "watermark": 0,
            "promote_percentile": 1,
        }
    )

    assert config.cpu_cache_slots == 1
    assert config.max_clock == 0
    assert config.clock_decay_every == 1
    assert config.watermark == 0
    assert config.promote_percentile == 1


@pytest.mark.parametrize(
    ("field", "user_config"),
    [
        ("cpu_cache_slots", {"cpu_cache_slots": 0}),
        ("cpu_cache_slots", {"cpu_cache_slots": 1.5}),
        ("cpu_cache_slots", {"cpu_cache_slots": True}),
        ("max_clock", {"max_clock": -1}),
        ("max_clock", {"max_clock": 1.5}),
        ("max_clock", {"max_clock": True}),
        ("clock_decay_every", {"clock_decay_every": 0}),
        ("clock_decay_every", {"clock_decay_every": True}),
        ("watermark", {"watermark": -0.1}),
        ("watermark", {"watermark": 1.1}),
        ("watermark", {"watermark": float("nan")}),
        ("watermark", {"watermark": True}),
        ("promote_percentile", {"promote_percentile": -0.1}),
        ("promote_percentile", {"promote_percentile": 1.1}),
        ("promote_percentile", {"promote_percentile": True}),
    ],
)
def test_score_config_rejects_invalid_values(field: str, user_config: dict[str, object]):
    with pytest.raises(ValueError, match=field):
        ScoreEncoderCacheConfig.from_dict(user_config)


def test_score_config_rejects_non_dict_config():
    with pytest.raises(ValueError, match="manager_config must be a dict"):
        ScoreEncoderCacheConfig.from_dict([])


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
