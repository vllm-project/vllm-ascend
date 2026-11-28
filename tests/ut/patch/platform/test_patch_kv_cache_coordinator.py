# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the hybrid coordinator MTP + prefix-cache fix.

The patch ``0001-fix-mtp-prefix-cache-hit-hybrid-pd`` makes three changes to
``AscendHybridKVCacheCoordinator``:

1. ``eagle_group_ids`` fallback flags **only FullAttention** groups when no
   group carries ``is_eagle_group`` (Mamba/GDN groups must stay out because
   draft models have no mamba layers).
2. The standard ``find_longest_cache_hit`` skips the EAGLE last-block drop on
   the PD prefill producer and on standalone instances. The role is derived
   from ``kv_transfer_config`` (``is_kv_producer and not is_kv_consumer``),
   attached onto ``KVCacheConfig`` by the ``get_kv_cache_config_from_groups``
   builder in ``vllm_ascend.core.kv_cache_config_builder``, and read back by
   the coordinator - no role environment variable is involved.
3. ``find_longest_cache_hit_per_group`` applies the same drop gating and
   keeps the ``(block_hashes, max_cache_hit_length)`` call convention used by
   RecomputeScheduler / DyntraLB / BalanceScheduler.
4. The producer/standalone EAGLE block-drop suppression is applied inline
   inside ``Scheduler._mamba_block_aligned_split`` (see
   ``patch_mamba_block_aligned_split``): on a pure producer or a standalone
   instance the EAGLE block-drop bit (``use_eagle`` on vLLM 0.28.x,
   ``use_eagle_block_drop`` on newer revisions) is cleared for the duration
   of the original call. Otherwise the scheduler's one-page backoff
   suppresses the final full mamba-align chunk split and the boundary state
   is never materialized. Consumers / kv_both instances pass through
   unchanged. The scheduler-side behavior itself is exercised in
   ``test_patch_mamba_block_aligned_split``.

These exercises run CPU-only: the heavy ``__init__`` is exercised with a
lightweight BlockPool/manager factory, lookup tests build the coordinator
with ``__new__`` and drive it through recording fake managers, and the
scheduler wrapper is exercised against one-off scheduler double classes.
"""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec

from vllm_ascend.core.kv_cache_compat import FullAttentionSpec, KVCacheGroupSpec
from vllm_ascend.patch.platform import patch_kv_cache_coordinator as mod
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import (
    AscendHybridKVCacheCoordinator,
)
from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.34.0"):
    from vllm.v1.core.kv_cache_coordinator import SpecGroup
else:
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager
    from vllm.v1.kv_cache_interface import KVCacheGroup, ManagerConfig, PrefixLookupGroup

HASH_BLOCK_SIZE = 128

FA_SPEC = FullAttentionSpec(
    block_size=128,
    num_kv_heads=1,
    head_size=8,
    dtype=torch.float32,
)
MAMBA_SPEC = MambaSpec(
    block_size=1536,
    shapes=((1,),),
    dtypes=(torch.float32,),
    **({"mamba_cache_mode": "align"} if vllm_version_is("0.34.0") else {}),
)


# ---------------------------------------------------------------------------
# Lightweight doubles for the real __init__ path (fix ① / role detection)
# ---------------------------------------------------------------------------


class _FakeBlockPool:
    def __init__(self, *args, **kwargs):
        pass


class _FakeFAManager:
    def __init__(self):
        self.use_eagle = False


class _FakeMambaManager:
    def __init__(self):
        self.use_eagle = False
        self.drop_eagle_checkpoint_block = False


def _fake_manager_factory(**kwargs):
    spec = kwargs["kv_cache_spec"]
    if isinstance(spec, MambaSpec):
        return _FakeMambaManager()
    return _FakeFAManager()


def _hybrid_config(
    *, mamba_eagle: bool = False, retention_interval: int | None = None, extra_mamba: bool = False
) -> KVCacheConfig:
    if vllm_version_is("0.34.0"):
        groups = [
            KVCacheGroupSpec(["fa-layer"], FA_SPEC, False),
            KVCacheGroupSpec(["mamba-layer"], MAMBA_SPEC, mamba_eagle),
        ]
        if extra_mamba:
            groups.append(KVCacheGroupSpec(["other-mamba-layer"], MAMBA_SPEC, False))
    else:
        groups = [
            KVCacheGroup.from_layers({"fa-layer": FullAttentionManager.layer_config(FA_SPEC)}),
            KVCacheGroup.from_layers(
                {
                    "mamba-layer": MambaManager.layer_config(
                        replace(MAMBA_SPEC, block_size=HASH_BLOCK_SIZE), enable_prefix_caching=True
                    )
                },
            ),
        ]
        if extra_mamba:
            groups.append(
                KVCacheGroup.from_layers(
                    {
                        "other-mamba-layer": MambaManager.layer_config(
                            replace(MAMBA_SPEC, block_size=HASH_BLOCK_SIZE), enable_prefix_caching=True
                        )
                    }
                )
            )
    return KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
        prefix_cache_retention_interval=retention_interval,
    )


def _make_coordinator(
    monkeypatch,
    *,
    use_eagle: bool,
    kv_transfer_config=None,
    mamba_eagle: bool = False,
    retention_interval: int | None = None,
    extra_mamba: bool = False,
):
    if vllm_version_is("0.34.0"):
        monkeypatch.setattr(mod, "BlockPool", _FakeBlockPool)
        monkeypatch.setattr(mod, "MambaManager", _FakeMambaManager)
        monkeypatch.setattr(mod, "get_manager_for_kv_cache_spec", _fake_manager_factory)
    kv_cache_config = _hybrid_config(
        mamba_eagle=mamba_eagle, retention_interval=retention_interval, extra_mamba=extra_mamba
    )
    # The kv-transfer config is attached by the
    # get_kv_cache_config_from_groups builder in real engine startup.
    kv_cache_config.kv_transfer_config = kv_transfer_config
    coordinator_options: dict[str, Any] = (
        {"use_eagle": use_eagle, "pcp_world_size": 1}
        if vllm_version_is("0.34.0")
        else {"max_in_flight_tokens": 4096, "num_hash_lookahead_tokens": int(use_eagle)}
    )
    return AscendHybridKVCacheCoordinator(
        kv_cache_config=kv_cache_config,
        max_model_len=4096,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        hash_block_size=HASH_BLOCK_SIZE,
        scheduler_block_size=HASH_BLOCK_SIZE,
        **coordinator_options,
    )


@pytest.mark.parametrize("retention_interval", [None, 0, 384])
def test_coordinator_uses_resolved_retention_interval(monkeypatch, retention_interval):
    coordinator = _make_coordinator(monkeypatch, use_eagle=False, retention_interval=retention_interval)
    assert coordinator.retention_interval == retention_interval


# ---------------------------------------------------------------------------
# Fix ①: eagle fallback must not flag mamba groups
# ---------------------------------------------------------------------------


def test_eagle_fallback_flags_only_full_attention_group(monkeypatch):
    coordinator = _make_coordinator(
        monkeypatch,
        use_eagle=True,
        kv_transfer_config=SimpleNamespace(is_kv_producer=False, is_kv_consumer=True),
    )
    if vllm_version_is("0.34.0"):
        assert coordinator.eagle_group_ids == {0}
        assert coordinator.single_type_managers[0].use_eagle is True
        assert coordinator.single_type_managers[1].use_eagle is False
    else:
        assert coordinator.block_pool.num_hash_lookahead_tokens == 1
        assert [group.group_ids for group in coordinator.attention_groups] == [[0], [1]]
        assert all(not hasattr(manager, "use_eagle") for manager in coordinator.single_type_managers)


def test_no_eagle_group_when_speculative_decoding_disabled(monkeypatch):
    coordinator = _make_coordinator(monkeypatch, use_eagle=False)
    if vllm_version_is("0.34.0"):
        assert coordinator.eagle_group_ids == set()
        assert all(not manager.use_eagle for manager in coordinator.single_type_managers)
    else:
        assert coordinator.block_pool.num_hash_lookahead_tokens == 0
        assert all(not hasattr(manager, "use_eagle") for manager in coordinator.single_type_managers)


def test_explicit_eagle_group_marker_takes_precedence(monkeypatch):
    # DeepSeek-V4-style annotation path: a flagged group must be respected
    # even when it is the mamba group (fallback must not overwrite it).
    coordinator = _make_coordinator(monkeypatch, use_eagle=True, mamba_eagle=True)
    if vllm_version_is("0.34.0"):
        assert coordinator.eagle_group_ids == {1}
    else:
        assert coordinator.block_pool.num_hash_lookahead_tokens == 1
        assert coordinator.attention_groups[1].group_ids == [1]
        assert coordinator.attention_groups[1].manager_config.params.enable_prefix_caching


# ---------------------------------------------------------------------------
# Fix ② role detection: kv_transfer_config (attached by the kv-cache-utils
# builder) -> coordinator
# ---------------------------------------------------------------------------


def _kv_transfer_config(*, is_kv_producer: bool, is_kv_consumer: bool):
    return SimpleNamespace(
        is_kv_producer=is_kv_producer,
        is_kv_consumer=is_kv_consumer,
    )


def test_coordinator_reads_producer_tag(monkeypatch):
    coordinator = _make_coordinator(
        monkeypatch,
        use_eagle=True,
        kv_transfer_config=_kv_transfer_config(is_kv_producer=True, is_kv_consumer=False),
    )
    if vllm_version_is("0.34.0"):
        assert coordinator.skips_eagle_block_drop is True
    else:
        assert coordinator.kv_cache_config.kv_transfer_config.is_kv_producer
        assert coordinator.block_pool.num_hash_lookahead_tokens == 1


def test_coordinator_non_producer_tag(monkeypatch):
    coordinator = _make_coordinator(
        monkeypatch,
        use_eagle=True,
        kv_transfer_config=_kv_transfer_config(is_kv_producer=False, is_kv_consumer=True),
    )
    if vllm_version_is("0.34.0"):
        assert coordinator.skips_eagle_block_drop is False
    else:
        assert coordinator.kv_cache_config.kv_transfer_config.is_kv_consumer
        assert coordinator.block_pool.num_hash_lookahead_tokens == 1


def test_coordinator_reads_standalone_drop_exemption(monkeypatch):
    # No kv-transfer config attached: a standalone instance. Not a PD
    # producer, but every content-hash match is a verified local prompt
    # block, so the EAGLE drop is suppressed just like on the producer.
    coordinator = _make_coordinator(monkeypatch, use_eagle=True, kv_transfer_config=None)
    if vllm_version_is("0.34.0"):
        assert coordinator.skips_eagle_block_drop is True
    else:
        assert coordinator.kv_cache_config.kv_transfer_config is None
        assert coordinator.block_pool.num_hash_lookahead_tokens == 1


@pytest.mark.parametrize(
    ("kv_transfer_config", "expected"),
    [
        (_kv_transfer_config(is_kv_producer=True, is_kv_consumer=False), True),
        (_kv_transfer_config(is_kv_producer=False, is_kv_consumer=True), False),
        (_kv_transfer_config(is_kv_producer=True, is_kv_consumer=True), False),
        (_kv_transfer_config(is_kv_producer=False, is_kv_consumer=False), False),
        (None, True),
    ],
)
def test_skips_eagle_block_drop_role_semantics(kv_transfer_config, expected):
    assert mod._skips_eagle_block_drop(kv_transfer_config) is expected


@pytest.mark.parametrize(
    ("kv_transfer_config", "use_eagle", "mamba_eagle", "expected"),
    [
        pytest.param(None, True, True, False, id="standalone"),
        pytest.param(_kv_transfer_config(is_kv_producer=True, is_kv_consumer=False), True, True, False, id="producer"),
        pytest.param(_kv_transfer_config(is_kv_producer=False, is_kv_consumer=True), True, True, True, id="consumer"),
        pytest.param(
            _kv_transfer_config(is_kv_producer=True, is_kv_consumer=True), True, False, True, id="both-unmarked"
        ),
        pytest.param(_kv_transfer_config(is_kv_producer=True, is_kv_consumer=True), False, True, False, id="no-drop"),
    ],
)
def test_mamba_checkpoint_drop_follows_lookup_policy(monkeypatch, kv_transfer_config, use_eagle, mamba_eagle, expected):
    coordinator = _make_coordinator(
        monkeypatch,
        use_eagle=use_eagle,
        kv_transfer_config=kv_transfer_config,
        mamba_eagle=mamba_eagle,
        extra_mamba=True,
    )
    if vllm_version_is("0.34.0"):
        # Lookup EAGLE markers do not limit checkpoint writes: even an unmarked
        # target Mamba group must share the scheduler's replay boundary.
        for manager in coordinator.single_type_managers[1:]:
            assert manager.drop_eagle_checkpoint_block is expected
        assert not hasattr(coordinator.single_type_managers[0], "drop_eagle_checkpoint_block")
    else:
        for manager, group in zip(
            coordinator.single_type_managers[1:], coordinator.kv_cache_config.kv_cache_groups[1:], strict=True
        ):
            assert not hasattr(manager, "drop_eagle_checkpoint_block")
            assert manager.block_pool.num_hash_lookahead_tokens == int(use_eagle)
            assert group.manager_config.params.enable_prefix_caching


@pytest.mark.parametrize(
    "kv_transfer_config",
    [
        SimpleNamespace(is_kv_producer=True, is_kv_consumer=False),
        None,
    ],
)
def test_kv_cache_config_builder_mounts_kv_transfer_config(monkeypatch, kv_transfer_config):
    if not vllm_version_is("0.34.0"):
        from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder

        from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder

        sentinel_config = SimpleNamespace()
        monkeypatch.setattr(
            DefaultKVCacheConfigBuilder, "get_kv_cache_config_from_groups", lambda *args: sentinel_config
        )
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="kimi_k3")),
            kv_transfer_config=kv_transfer_config,
        )
        result = AscendKVCacheConfigBuilder().get_kv_cache_config_from_groups(vllm_config, [], 0)
        assert result is sentinel_config
        assert result.kv_transfer_config is kv_transfer_config
        return
    from vllm_ascend.core import kv_cache_config_builder as builder_module

    sentinel_config = SimpleNamespace()
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="kimi_k3")),
        kv_transfer_config=kv_transfer_config,
    )
    monkeypatch.setattr(
        builder_module.DefaultKVCacheConfigBuilder,
        "get_kv_cache_config_from_groups",
        lambda *args, **kwargs: sentinel_config,
    )
    result = builder_module.get_ascend_kv_cache_config_from_groups(vllm_config, [], num_blocks=0)
    # The builder owns every config it materializes, so the coordinator-side
    # getattr in __init__ sees the attached role on every real engine path.
    assert result is sentinel_config
    assert result.kv_transfer_config is kv_transfer_config


@pytest.mark.parametrize(
    "kv_transfer_config",
    [
        SimpleNamespace(is_kv_producer=False, is_kv_consumer=True),
        None,
    ],
)
def test_public_builder_mounts_kv_transfer_config_on_default_path(monkeypatch, kv_transfer_config):
    from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder

    from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder

    # A generic layout matches no Ascend divisor, so the public builder
    # returns through the upstream default materializer. The PD consumer
    # role must ride on that result too: the hybrid coordinator treats a
    # config without it as standalone and would skip the EAGLE replay block
    # drop used for external cache loads.
    sentinel_config = SimpleNamespace()
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="llama")),
        kv_transfer_config=kv_transfer_config,
    )
    monkeypatch.setattr(
        DefaultKVCacheConfigBuilder,
        "get_kv_cache_config_from_groups",
        lambda self, vllm_config, groups, num_blocks: sentinel_config,
    )
    result = AscendKVCacheConfigBuilder().get_kv_cache_config_from_groups(
        vllm_config, [KVCacheGroupSpec(["fa-layer"], FA_SPEC, False)], 8
    )
    assert result is sentinel_config
    assert result.kv_transfer_config is kv_transfer_config


# ---------------------------------------------------------------------------
# Recording fake managers for the lookup paths (fix ② / ③)
# ---------------------------------------------------------------------------


class _RecordingManager:
    """Reports a hit equal to the offered max_length (never shrinks it) and
    records the ``drop_eagle_block`` flag the coordinator passed."""

    supports_fine_grained_hash_lookup = False
    Params = type(None)
    calls: list[dict] = []

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes,
        max_length,
        kv_cache_group_ids,
        block_pool,
        alignment_tokens,
        drop_eagle_block=None,
        dcp_world_size=1,
        pcp_world_size=1,
        **kwargs,
    ):
        cls.calls.append(
            {"max_length": max_length, "drop_eagle_block": drop_eagle_block, "dcp_world_size": dcp_world_size}
        )
        return (([],), max_length)


class _RecordingFA(_RecordingManager):
    calls = []


class _RecordingMamba(_RecordingManager):
    calls = []


def _make_lookup_coordinator(*, producer: bool, standalone: bool = False):
    coordinator = AscendHybridKVCacheCoordinator.__new__(AscendHybridKVCacheCoordinator)
    coordinator.skips_eagle_block_drop = producer or standalone
    coordinator.dcp_world_size = 1
    coordinator.hash_block_size = HASH_BLOCK_SIZE
    coordinator.scheduler_block_size = None
    coordinator.lcm_block_size = MAMBA_SPEC.block_size
    coordinator.enable_partial_hash_hits = False
    coordinator.enable_caching = True
    coordinator.block_pool = object()
    coordinator.kv_cache_config = SimpleNamespace(kv_cache_groups=[object(), object()])
    # FA group is the EAGLE group; the mamba group never is.
    if vllm_version_is("0.34.0"):
        coordinator.attention_groups = [
            SpecGroup(FA_SPEC, [0], _RecordingFA, True),
            SpecGroup(MAMBA_SPEC, [1], _RecordingMamba, False),
        ]
    else:
        coordinator.scheduler_block_size = HASH_BLOCK_SIZE
        coordinator.num_reprefillable_tokens = 0
        coordinator.kv_cache_config.kv_cache_groups = [
            SimpleNamespace(prefix_cacheable=True, kv_cache_specs={}) for _ in range(2)
        ]
        coordinator.single_type_managers = [
            SimpleNamespace(
                block_size=spec.block_size, block_pool=coordinator.block_pool, dcp_world_size=1, pcp_world_size=1
            )
            for spec in (FA_SPEC, MAMBA_SPEC)
        ]
        coordinator.attention_groups = [
            PrefixLookupGroup(ManagerConfig(_RecordingFA, None), FA_SPEC.block_size, [0]),
            PrefixLookupGroup(ManagerConfig(_RecordingMamba, None), MAMBA_SPEC.block_size, [1]),
        ]
    # Newer PD/partial-hit revisions cap the offered hit length at lookup
    # entry. That behavior is outside this regression's scope; install a
    # pass-through stub so the drop-gating logic runs with the original
    # length. On the baseline revision the method does not exist and no stub
    # is needed.
    if hasattr(AscendHybridKVCacheCoordinator, "_producer_hit_cap"):
        coordinator.pd_has_state_groups = False
        coordinator.enable_partial_hash_hits = False
        coordinator._producer_hit_cap = lambda length: length
    return coordinator


# ---------------------------------------------------------------------------
# Fix ②: standard find_longest_cache_hit drop gating
# ---------------------------------------------------------------------------


def test_standard_lookup_drops_last_block_on_consumer():
    coordinator = _make_lookup_coordinator(producer=False)
    block_hashes = [] if vllm_version_is("0.34.0") else [bytes(32)] * (2048 // HASH_BLOCK_SIZE)
    coordinator.find_longest_cache_hit(block_hashes=block_hashes, max_cache_hit_length=2048)
    assert _RecordingFA.calls[-1]["drop_eagle_block"] is (True if vllm_version_is("0.34.0") else None)
    # Mamba group is never eagle.
    assert _RecordingMamba.calls[-1]["drop_eagle_block"] is (False if vllm_version_is("0.34.0") else None)


def test_standard_lookup_skips_drop_on_producer():
    coordinator = _make_lookup_coordinator(producer=True)
    coordinator.find_longest_cache_hit(block_hashes=[], max_cache_hit_length=2048)
    assert _RecordingFA.calls[-1]["drop_eagle_block"] is (False if vllm_version_is("0.34.0") else None)
    assert _RecordingMamba.calls[-1]["drop_eagle_block"] is (False if vllm_version_is("0.34.0") else None)


def test_standard_lookup_skips_drop_on_standalone():
    coordinator = _make_lookup_coordinator(producer=False, standalone=True)
    coordinator.find_longest_cache_hit(block_hashes=[], max_cache_hit_length=2048)
    assert _RecordingFA.calls[-1]["drop_eagle_block"] is (False if vllm_version_is("0.34.0") else None)
    assert _RecordingMamba.calls[-1]["drop_eagle_block"] is (False if vllm_version_is("0.34.0") else None)


# ---------------------------------------------------------------------------
# Fix ③: per-group lookup drop gating + call-site signature
# ---------------------------------------------------------------------------


def test_per_group_lookup_drops_last_block_on_consumer():
    coordinator = _make_lookup_coordinator(producer=False)
    block_hashes = [] if vllm_version_is("0.34.0") else [bytes(32)] * (2048 // HASH_BLOCK_SIZE)
    _, hit_lengths = coordinator.find_longest_cache_hit_per_group(block_hashes, 2048)
    assert _RecordingFA.calls[-1]["drop_eagle_block"] is (True if vllm_version_is("0.34.0") else None)
    assert _RecordingMamba.calls[-1]["drop_eagle_block"] is (False if vllm_version_is("0.34.0") else None)
    assert hit_lengths == (2048, 2048)


def test_per_group_lookup_skips_drop_on_producer():
    coordinator = _make_lookup_coordinator(producer=True)
    _, hit_lengths = coordinator.find_longest_cache_hit_per_group([], 2048)
    assert _RecordingFA.calls[-1]["drop_eagle_block"] is (False if vllm_version_is("0.34.0") else None)
    assert _RecordingMamba.calls[-1]["drop_eagle_block"] is (False if vllm_version_is("0.34.0") else None)
    assert hit_lengths == (2048, 2048)


def test_per_group_lookup_skips_drop_on_standalone():
    coordinator = _make_lookup_coordinator(producer=False, standalone=True)
    _, hit_lengths = coordinator.find_longest_cache_hit_per_group([], 2048)
    assert _RecordingFA.calls[-1]["drop_eagle_block"] is (False if vllm_version_is("0.34.0") else None)
    assert _RecordingMamba.calls[-1]["drop_eagle_block"] is (False if vllm_version_is("0.34.0") else None)
    assert hit_lengths == (2048, 2048)


def test_per_group_lookup_matches_scheduler_call_convention():
    # RecomputeScheduler / DyntraLBScheduler / BalanceScheduler all call with
    # exactly two positional args: (request.block_hashes, request.num_tokens-1).
    coordinator = _make_lookup_coordinator(producer=True)
    block_hashes = ["hash-0", "hash-1"]
    blocks, hit_lengths = coordinator.find_longest_cache_hit_per_group(block_hashes, 4095)
    assert len(blocks) == 2
    assert _RecordingFA.calls[-1]["max_length"] == 4095
    assert hit_lengths == (4095, 4095)


def test_per_group_lookup_preserves_replicated_draft_positions():
    coordinator = _make_lookup_coordinator(producer=True)
    coordinator.dcp_world_size = 2
    draft_spec = replace(FA_SPEC, is_draft=True, dcp_sharded=False)
    if vllm_version_is("0.34.0"):
        coordinator.attention_groups = [
            SpecGroup(FA_SPEC, [0], _RecordingFA, False),
            SpecGroup(draft_spec, [1], _RecordingMamba, True),
        ]
    else:
        target_manager, draft_manager = coordinator.single_type_managers
        target_manager.dcp_world_size = coordinator.dcp_world_size
        draft_manager.block_size = draft_spec.block_size
        coordinator.attention_groups[1] = PrefixLookupGroup(
            ManagerConfig(_RecordingMamba, None), draft_spec.block_size, [1]
        )
    coordinator.find_longest_cache_hit_per_group([], 2048)
    assert _RecordingFA.calls[-1]["dcp_world_size"] == 2
    assert _RecordingMamba.calls[-1]["dcp_world_size"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
