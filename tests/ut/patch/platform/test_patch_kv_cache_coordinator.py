# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for effective Eagle drop, lookahead and legacy role exemptions.

Current main follows the native resolved policy for every role. Only v0.32
keeps the historical producer/standalone exemption. No model or NPU execution.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.kv_cache_coordinator import SpecGroup
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupRole,
    KVCacheGroupSpec,
    MambaSpec,
)

from vllm_ascend.patch.platform import patch_kv_cache_coordinator as mod
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import (
    AscendHybridKVCacheCoordinator,
)


@pytest.fixture(autouse=True)
def current_main(monkeypatch):
    monkeypatch.setattr(mod, "vllm_version_is", lambda version: False)


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
    mamba_cache_mode="align",
)


# ---------------------------------------------------------------------------
# Lightweight doubles for the real __init__ path (fix ① / role detection)
# ---------------------------------------------------------------------------


class _FakeBlockPool:
    def __init__(self, *args, **kwargs):
        self.num_hash_lookahead_tokens = kwargs.get("num_hash_lookahead_tokens", 0)


class _FakeFAManager:
    def __init__(self, **kwargs):
        self.use_eagle = False
        self.role = kwargs.get("role")
        self.block_size = kwargs["kv_cache_spec"].block_size


class _FakeMambaManager:
    def __init__(self, **kwargs):
        self.drop_eagle_checkpoint_block = False
        self.use_eagle = False
        self.role = kwargs.get("role")
        self.block_size = kwargs["kv_cache_spec"].block_size


def _fake_manager_factory(**kwargs):
    spec = kwargs["kv_cache_spec"]
    if isinstance(spec, MambaSpec):
        return _FakeMambaManager(**kwargs)
    return _FakeFAManager(**kwargs)


def _hybrid_config(
    *,
    mamba_eagle: bool = False,
    fa_eagle: bool = False,
    retention_interval: int | None = None,
) -> KVCacheConfig:
    groups = [
        KVCacheGroupSpec(["fa-layer"], FA_SPEC, fa_eagle),
        KVCacheGroupSpec(["mamba-layer"], MAMBA_SPEC, mamba_eagle),
    ]
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
    fa_eagle: bool = False,
    num_hash_lookahead_tokens: int = 0,
    retention_interval: int | None = None,
    scheduler_block_size: int = MAMBA_SPEC.block_size,
    kv_cache_config: KVCacheConfig | None = None,
):
    monkeypatch.setattr(mod, "BlockPool", _FakeBlockPool)
    monkeypatch.setattr(mod, "MambaManager", _FakeMambaManager)
    monkeypatch.setattr(mod, "get_manager_for_kv_cache_spec", _fake_manager_factory)
    if kv_cache_config is None:
        kv_cache_config = _hybrid_config(
            mamba_eagle=mamba_eagle,
            fa_eagle=fa_eagle,
            retention_interval=retention_interval,
        )
    # The kv-transfer config is attached by the
    # get_kv_cache_config_from_groups builder in real engine startup.
    kv_cache_config.kv_transfer_config = kv_transfer_config
    return AscendHybridKVCacheCoordinator(
        kv_cache_config=kv_cache_config,
        max_model_len=4096,
        use_eagle=use_eagle,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=HASH_BLOCK_SIZE,
        scheduler_block_size=scheduler_block_size,
        num_hash_lookahead_tokens=num_hash_lookahead_tokens,
    )


@pytest.mark.parametrize("retention_interval", [None, 0, 384])
def test_coordinator_uses_resolved_retention_interval(monkeypatch, retention_interval):
    coordinator = _make_coordinator(
        monkeypatch,
        use_eagle=False,
        retention_interval=retention_interval,
        scheduler_block_size=HASH_BLOCK_SIZE,
    )
    assert coordinator.retention_interval == retention_interval


# ---------------------------------------------------------------------------
# Eagle identity and version-specific fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("legacy", [False, True], ids=["main", "v032"])
def test_unmarked_eagle_fallback_matches_paired_version(monkeypatch, legacy):
    monkeypatch.setattr(mod, "vllm_version_is", lambda version: legacy and version == "0.32.0")
    coordinator = _make_coordinator(monkeypatch, use_eagle=True)
    expected = {0} if legacy else {0, 1}
    assert coordinator.eagle_group_ids == expected
    assert coordinator.drop_eagle_group_ids == expected
    assert coordinator.single_type_managers[0].use_eagle is True
    assert coordinator.single_type_managers[1].use_eagle is (not legacy)


def test_no_eagle_group_when_speculative_decoding_disabled(monkeypatch):
    coordinator = _make_coordinator(monkeypatch, use_eagle=False)
    assert coordinator.eagle_group_ids == set()
    assert all(not manager.use_eagle for manager in coordinator.single_type_managers)


def test_explicit_eagle_group_marker_takes_precedence(monkeypatch):
    # DeepSeek-V4-style annotation path: a flagged group must be respected
    # even when it is the mamba group (fallback must not overwrite it).
    coordinator = _make_coordinator(monkeypatch, use_eagle=True, mamba_eagle=True)
    assert coordinator.eagle_group_ids == {1}


# ---------------------------------------------------------------------------
# Fix ② role detection: kv_transfer_config (attached by the kv-cache-utils
# builder) -> coordinator
# ---------------------------------------------------------------------------


def _kv_transfer_config(*, is_kv_producer: bool, is_kv_consumer: bool):
    return SimpleNamespace(
        is_kv_producer=is_kv_producer,
        is_kv_consumer=is_kv_consumer,
    )


def test_main_coordinator_does_not_exempt_producer(monkeypatch):
    coordinator = _make_coordinator(
        monkeypatch,
        use_eagle=True,
        kv_transfer_config=_kv_transfer_config(is_kv_producer=True, is_kv_consumer=False),
    )
    assert coordinator.skips_eagle_block_drop is False


def test_coordinator_non_producer_tag(monkeypatch):
    coordinator = _make_coordinator(
        monkeypatch,
        use_eagle=True,
        kv_transfer_config=_kv_transfer_config(is_kv_producer=False, is_kv_consumer=True),
    )
    assert coordinator.skips_eagle_block_drop is False


def test_main_coordinator_does_not_exempt_standalone(monkeypatch):
    coordinator = _make_coordinator(monkeypatch, use_eagle=True, kv_transfer_config=None)
    assert coordinator.skips_eagle_block_drop is False


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
@pytest.mark.parametrize("legacy", [False, True], ids=["main", "v032"])
def test_skips_eagle_block_drop_role_semantics(monkeypatch, kv_transfer_config, expected, legacy):
    monkeypatch.setattr(mod, "vllm_version_is", lambda version: legacy and version == "0.32.0")
    assert mod._skips_eagle_block_drop(kv_transfer_config) is (legacy and expected)


@pytest.mark.parametrize(
    ("kv_transfer_config", "drop_for_role"),
    [
        pytest.param(None, False, id="standalone"),
        pytest.param(_kv_transfer_config(is_kv_producer=True, is_kv_consumer=False), False, id="producer"),
        pytest.param(_kv_transfer_config(is_kv_producer=False, is_kv_consumer=True), True, id="consumer"),
        pytest.param(_kv_transfer_config(is_kv_producer=True, is_kv_consumer=True), True, id="both"),
    ],
)
@pytest.mark.parametrize("use_eagle", [False, True])
@pytest.mark.parametrize("mamba_eagle", [False, True])
def test_mamba_checkpoint_drop_follows_lookup_policy(
    monkeypatch, kv_transfer_config, drop_for_role, use_eagle, mamba_eagle
):
    monkeypatch.setattr(mod, "vllm_version_is", lambda version: version == "0.32.0")
    coordinator = _make_coordinator(
        monkeypatch,
        use_eagle=use_eagle,
        kv_transfer_config=kv_transfer_config,
        mamba_eagle=mamba_eagle,
    )
    manager = coordinator.single_type_managers[1]
    assert manager.drop_eagle_checkpoint_block is (use_eagle and mamba_eagle and drop_for_role)


@pytest.mark.parametrize(
    "kv_transfer_config",
    [
        SimpleNamespace(is_kv_producer=True, is_kv_consumer=False),
        None,
    ],
)
def test_kv_cache_config_builder_mounts_kv_transfer_config(monkeypatch, kv_transfer_config):
    from vllm_ascend.patch.platform import patch_kv_cache_utils as kcu_patch

    sentinel_config = SimpleNamespace()
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="kimi_k3")),
        kv_transfer_config=kv_transfer_config,
    )
    monkeypatch.setattr(
        kcu_patch,
        "_orig_get_kv_cache_config_from_groups",
        lambda *args, **kwargs: sentinel_config,
    )
    result = kcu_patch._ascend_get_kv_cache_config_from_groups(vllm_config, [], 0)
    # The builder replaces the upstream entry point, so the coordinator-side
    # getattr in __init__ sees the attached role on every real engine path.
    live_module = kcu_patch._kv_cache_planning
    assert live_module.get_kv_cache_config_from_groups is kcu_patch._ascend_get_kv_cache_config_from_groups
    assert result is sentinel_config
    assert result.kv_transfer_config is kv_transfer_config


# ---------------------------------------------------------------------------
# Recording fake managers for the lookup paths (fix ② / ③)
# ---------------------------------------------------------------------------


class _RecordingManager:
    """Reports a hit equal to the offered max_length (never shrinks it) and
    records the ``drop_eagle_block`` flag the coordinator passed."""

    supports_fine_grained_hash_lookup = False
    calls: list[dict] = []

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes,
        max_length,
        kv_cache_group_ids,
        block_pool,
        kv_cache_spec,
        drop_eagle_block,
        alignment_tokens,
        dcp_world_size=1,
        pcp_world_size=1,
    ):
        cls.calls.append({"max_length": max_length, "drop_eagle_block": drop_eagle_block})
        return (([],), max_length)


class _RecordingFA(_RecordingManager):
    calls = []


class _RecordingMamba(_RecordingManager):
    calls = []


def _make_legacy_lookup_coordinator(*, producer: bool, standalone: bool = False):
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
    coordinator.attention_groups = [
        SpecGroup(FA_SPEC, [0], _RecordingFA, True),
        SpecGroup(MAMBA_SPEC, [1], _RecordingMamba, False),
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
    coordinator = _make_legacy_lookup_coordinator(producer=False)
    coordinator.find_longest_cache_hit(block_hashes=[], max_cache_hit_length=2048)
    assert _RecordingFA.calls[-1]["drop_eagle_block"] is True
    # Mamba group is never eagle.
    assert _RecordingMamba.calls[-1]["drop_eagle_block"] is False


def test_legacy_standard_lookup_skips_drop_on_producer():
    coordinator = _make_legacy_lookup_coordinator(producer=True)
    coordinator.find_longest_cache_hit(block_hashes=[], max_cache_hit_length=2048)
    assert _RecordingFA.calls[-1]["drop_eagle_block"] is False
    assert _RecordingMamba.calls[-1]["drop_eagle_block"] is False


def test_legacy_standard_lookup_skips_drop_on_standalone():
    coordinator = _make_legacy_lookup_coordinator(producer=False, standalone=True)
    coordinator.find_longest_cache_hit(block_hashes=[], max_cache_hit_length=2048)
    assert _RecordingFA.calls[-1]["drop_eagle_block"] is False
    assert _RecordingMamba.calls[-1]["drop_eagle_block"] is False


# ---------------------------------------------------------------------------
# Fix ③: per-group lookup drop gating + call-site signature
# ---------------------------------------------------------------------------


def test_per_group_lookup_drops_last_block_on_consumer():
    coordinator = _make_legacy_lookup_coordinator(producer=False)
    _, hit_lengths = coordinator.find_longest_cache_hit_per_group([], 2048)
    assert _RecordingFA.calls[-1]["drop_eagle_block"] is True
    assert _RecordingMamba.calls[-1]["drop_eagle_block"] is False
    assert hit_lengths == (2048, 2048)


def test_legacy_per_group_lookup_skips_drop_on_producer():
    coordinator = _make_legacy_lookup_coordinator(producer=True)
    _, hit_lengths = coordinator.find_longest_cache_hit_per_group([], 2048)
    assert _RecordingFA.calls[-1]["drop_eagle_block"] is False
    assert _RecordingMamba.calls[-1]["drop_eagle_block"] is False
    assert hit_lengths == (2048, 2048)


def test_legacy_per_group_lookup_skips_drop_on_standalone():
    coordinator = _make_legacy_lookup_coordinator(producer=False, standalone=True)
    _, hit_lengths = coordinator.find_longest_cache_hit_per_group([], 2048)
    assert _RecordingFA.calls[-1]["drop_eagle_block"] is False
    assert _RecordingMamba.calls[-1]["drop_eagle_block"] is False
    assert hit_lengths == (2048, 2048)


def test_per_group_lookup_matches_scheduler_call_convention():
    # RecomputeScheduler / DyntraLBScheduler / BalanceScheduler all call with
    # exactly two positional args: (request.block_hashes, request.num_tokens-1).
    coordinator = _make_legacy_lookup_coordinator(producer=True)
    block_hashes = ["hash-0", "hash-1"]
    blocks, hit_lengths = coordinator.find_longest_cache_hit_per_group(block_hashes, 4095)
    assert len(blocks) == 2
    assert _RecordingFA.calls[-1]["max_length"] == 4095
    assert hit_lengths == (4095, 4095)


@pytest.mark.parametrize(
    "lookahead, effective_drop",
    [(0, True), (1, False), (0, False)],
    ids=["legacy-hashes-or-disable-lookahead", "lookahead", "disable-eagle-block-drop"],
)
@pytest.mark.parametrize("role", ["standalone", "producer", "consumer", "both"])
def test_main_resolved_drop_preserves_identity_and_lookup(monkeypatch, lookahead, effective_drop, role):
    config = (
        None
        if role == "standalone"
        else _kv_transfer_config(
            is_kv_producer=role in ("producer", "both"),
            is_kv_consumer=role in ("consumer", "both"),
        )
    )
    coordinator = _make_coordinator(
        monkeypatch,
        use_eagle=effective_drop,
        fa_eagle=True,
        kv_transfer_config=config,
        num_hash_lookahead_tokens=lookahead,
    )
    assert coordinator.eagle_group_ids == {0}
    assert coordinator.drop_eagle_group_ids == ({0} if effective_drop else set())
    assert coordinator.block_pool.num_hash_lookahead_tokens == lookahead
    assert coordinator.skips_eagle_block_drop is False
    assert [g.use_eagle for g in coordinator.attention_groups] == [effective_drop, False]
    assert [m.use_eagle for m in coordinator.single_type_managers] == [effective_drop, False]
    # Mamba is not an EAGLE group here, but follows the global replay boundary.
    assert coordinator.single_type_managers[1].drop_eagle_checkpoint_block is effective_drop
    # Keep the constructor's resolved group policy while recording both lookup APIs.
    coordinator.attention_groups = [
        g._replace(manager_cls=_RecordingFA if isinstance(g.spec, FullAttentionSpec) else _RecordingMamba)
        for g in coordinator.attention_groups
    ]
    coordinator.find_longest_cache_hit([], 2048)
    assert _RecordingFA.calls[-1]["drop_eagle_block"] is effective_drop
    assert _RecordingMamba.calls[-1]["drop_eagle_block"] is False
    coordinator.find_longest_cache_hit_per_group([], 2048)
    assert _RecordingFA.calls[-1]["drop_eagle_block"] is effective_drop
    assert _RecordingMamba.calls[-1]["drop_eagle_block"] is False


@pytest.mark.parametrize("legacy", [False, True], ids=["main", "v032"])
@pytest.mark.parametrize("retention", [0, MAMBA_SPEC.block_size])
def test_retention_uses_resolved_cache_config(monkeypatch, retention, legacy):
    monkeypatch.setattr(mod, "vllm_version_is", lambda version: legacy and version == "0.32.0")
    # A conflicting environment value must not override the resolved config.
    monkeypatch.setenv("VLLM_PREFIX_CACHE_RETENTION_INTERVAL", "3072")
    coordinator = _make_coordinator(
        monkeypatch,
        use_eagle=False,
        fa_eagle=True,
        num_hash_lookahead_tokens=1,
        retention_interval=retention,
    )
    assert coordinator.retention_interval == retention


@pytest.mark.parametrize("initial_retention", [0, 3072])
def test_inherits_native_store_checkpoint_retention(monkeypatch, initial_retention):
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store import (  # type: ignore
            scheduler as store_mod,
        )
    except ModuleNotFoundError:
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (  # type: ignore
            scheduler as store_mod,
        )

    # Exercise the real Store retention policy, stubbing only lookup transport
    # and unrelated coordinator construction. No store client/network is opened.
    monkeypatch.setattr(store_mod, "LookupKeyClient", lambda config: None)
    monkeypatch.setattr(
        store_mod,
        "resolve_kv_cache_block_sizes",
        lambda kv_config, config: (MAMBA_SPEC.block_size, HASH_BLOCK_SIZE),
    )
    monkeypatch.setattr(
        store_mod,
        "MooncakeStoreCoordinator",
        SimpleNamespace(from_kv_cache_config=lambda *args: SimpleNamespace(enable_partial_hash_hits=True)),
    )
    kv_config = _hybrid_config(fa_eagle=True)
    kv_config.prefix_cache_retention_interval = initial_retention
    transfer = SimpleNamespace(
        kv_role="kv_both",
        is_kv_producer=True,
        is_kv_consumer=True,
        kv_connector_extra_config={"save_decode_cache": True},
    )
    config = SimpleNamespace(
        kv_transfer_config=transfer,
        kv_events_config=None,
        parallel_config=SimpleNamespace(world_size=1),
    )
    store = store_mod.MooncakeStoreScheduler(config, kv_config)
    # Store resolves its version's checkpoint policy; Ascend must preserve it.
    expected = store.kv_cache_config.prefix_cache_retention_interval
    assert store.kv_cache_config is kv_config
    coordinator = _make_coordinator(
        monkeypatch,
        use_eagle=False,
        num_hash_lookahead_tokens=1,
        kv_transfer_config=transfer,
        kv_cache_config=kv_config,
    )
    assert coordinator.kv_cache_config is kv_config
    assert coordinator.retention_interval == expected
    assert coordinator.retention_interval is not None  # Never silently fall back to dense.


def test_main_forwards_cache_group_role_to_manager(monkeypatch):
    config = _hybrid_config(fa_eagle=True)
    config.kv_cache_groups[0].role = KVCacheGroupRole.HISPARSE_SOURCE
    coordinator = _make_coordinator(
        monkeypatch,
        use_eagle=False,
        num_hash_lookahead_tokens=1,
        kv_cache_config=config,
    )
    assert [manager.role for manager in coordinator.single_type_managers] == [
        group.role for group in config.kv_cache_groups
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
