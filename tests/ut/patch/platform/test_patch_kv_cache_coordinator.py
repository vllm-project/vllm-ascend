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
   builder in ``patch_kv_cache_utils``, and read back by the coordinator -
   no role environment variable is involved.
3. ``find_longest_cache_hit_per_group`` applies the same drop gating and
   keeps the ``(block_hashes, max_cache_hit_length)`` call convention used by
   RecomputeScheduler / DyntraLB / BalanceScheduler.
4. ``Scheduler._mamba_block_aligned_split`` is wrapped unconditionally (see
   ``patch_mamba_block_aligned_split``); on a pure producer or a standalone
   instance the EAGLE block-drop bit (``use_eagle`` on vLLM 0.28.x,
   ``use_eagle_block_drop`` on newer revisions) is cleared for the duration
   of the original call. Otherwise the scheduler's one-page backoff
   suppresses the final full mamba-align chunk split and the boundary state
   is never materialized. Consumers / kv_both instances pass through
   unchanged.

These exercises run CPU-only: the heavy ``__init__`` is exercised with a
lightweight BlockPool/manager factory, lookup tests build the coordinator
with ``__new__`` and drive it through recording fake managers, and the
scheduler wrapper is exercised against one-off scheduler double classes.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.kv_cache_coordinator import SpecGroup
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

from vllm_ascend.patch.platform import patch_kv_cache_coordinator as mod
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import (
    AscendHybridKVCacheCoordinator,
)
from vllm_ascend.patch.platform.patch_mamba_block_aligned_split import (
    _install_producer_mamba_block_aligned_split_patch,
)

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
        pass


class _FakeFAManager:
    def __init__(self, **kwargs):
        self.use_eagle = False


class _FakeMambaManager:
    def __init__(self, **kwargs):
        self.use_eagle = False


def _fake_manager_factory(**kwargs):
    spec = kwargs["kv_cache_spec"]
    if isinstance(spec, MambaSpec):
        return _FakeMambaManager()
    return _FakeFAManager()


def _hybrid_config(*, mamba_eagle: bool = False) -> KVCacheConfig:
    groups = [
        KVCacheGroupSpec(["fa-layer"], FA_SPEC, False),
        KVCacheGroupSpec(["mamba-layer"], MAMBA_SPEC, mamba_eagle),
    ]
    return KVCacheConfig(num_blocks=8, kv_cache_tensors=[], kv_cache_groups=groups)


def _make_coordinator(
    monkeypatch,
    *,
    use_eagle: bool,
    kv_transfer_config=None,
    role_tagged: bool = True,
    mamba_eagle: bool = False,
):
    monkeypatch.setattr(mod, "BlockPool", _FakeBlockPool)
    monkeypatch.setattr(mod, "get_manager_for_kv_cache_spec", _fake_manager_factory)
    kv_cache_config = _hybrid_config(mamba_eagle=mamba_eagle)
    if role_tagged:
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
    )


# ---------------------------------------------------------------------------
# Fix ①: eagle fallback must not flag mamba groups
# ---------------------------------------------------------------------------


def test_eagle_fallback_flags_only_full_attention_group(monkeypatch):
    coordinator = _make_coordinator(monkeypatch, use_eagle=True)
    # Only the FA group is flagged; the mamba group stays out.
    assert coordinator.eagle_group_ids == {0}
    # Bit propagation after verify_and_split: FA manager eagle, mamba not.
    assert coordinator.single_type_managers[0].use_eagle is True
    assert coordinator.single_type_managers[1].use_eagle is False


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


def test_coordinator_reads_producer_tag(monkeypatch):
    coordinator = _make_coordinator(
        monkeypatch,
        use_eagle=True,
        kv_transfer_config=_kv_transfer_config(is_kv_producer=True, is_kv_consumer=False),
    )
    assert coordinator.is_kv_producer is True
    assert coordinator.skips_eagle_block_drop is True


def test_coordinator_non_producer_tag(monkeypatch):
    coordinator = _make_coordinator(
        monkeypatch,
        use_eagle=True,
        kv_transfer_config=_kv_transfer_config(is_kv_producer=False, is_kv_consumer=True),
    )
    assert coordinator.is_kv_producer is False
    assert coordinator.skips_eagle_block_drop is False


def test_coordinator_defaults_non_producer_without_tag(monkeypatch):
    coordinator = _make_coordinator(monkeypatch, use_eagle=True, role_tagged=False)
    assert coordinator.is_kv_producer is False


def test_coordinator_reads_standalone_drop_exemption(monkeypatch):
    # No kv-transfer config attached: a standalone instance. Not a PD
    # producer, but every content-hash match is a verified local prompt
    # block, so the EAGLE drop is suppressed just like on the producer.
    coordinator = _make_coordinator(monkeypatch, use_eagle=True, kv_transfer_config=None)
    assert coordinator.is_kv_producer is False
    assert coordinator.skips_eagle_block_drop is True


@pytest.mark.parametrize(
    ("kv_transfer_config", "expected"),
    [
        (_kv_transfer_config(is_kv_producer=True, is_kv_consumer=False), True),
        (_kv_transfer_config(is_kv_producer=False, is_kv_consumer=True), False),
        (_kv_transfer_config(is_kv_producer=True, is_kv_consumer=True), False),
        (_kv_transfer_config(is_kv_producer=False, is_kv_consumer=False), False),
        (None, False),
    ],
)
def test_is_kv_producer_role_semantics(kv_transfer_config, expected):
    assert mod._is_kv_producer(kv_transfer_config) is expected


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
    "kv_transfer_config",
    [
        SimpleNamespace(is_kv_producer=True, is_kv_consumer=False),
        None,
    ],
)
def test_kv_cache_config_builder_mounts_kv_transfer_config(monkeypatch, kv_transfer_config):
    import vllm.v1.core.kv_cache_utils as kcu

    from vllm_ascend.patch.platform import patch_kv_cache_utils as kcu_patch

    sentinel_config = SimpleNamespace()
    vllm_config = SimpleNamespace(kv_transfer_config=kv_transfer_config)
    monkeypatch.setattr(
        kcu_patch,
        "_orig_get_kv_cache_config_from_groups",
        lambda *args, **kwargs: sentinel_config,
    )
    result = kcu_patch._ascend_get_kv_cache_config_from_groups(vllm_config, [], 0)
    # The builder replaces the upstream entry point, so the coordinator-side
    # getattr in __init__ sees the attached role on every real engine path.
    assert kcu.get_kv_cache_config_from_groups is kcu_patch._ascend_get_kv_cache_config_from_groups
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


def _make_lookup_coordinator(*, producer: bool, standalone: bool = False):
    coordinator = AscendHybridKVCacheCoordinator.__new__(AscendHybridKVCacheCoordinator)
    coordinator.is_kv_producer = producer
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
    coordinator = _make_lookup_coordinator(producer=False)
    coordinator.find_longest_cache_hit(block_hashes=[], max_cache_hit_length=2048)
    assert _RecordingFA.calls[-1]["drop_eagle_block"] is True
    # Mamba group is never eagle.
    assert _RecordingMamba.calls[-1]["drop_eagle_block"] is False


def test_standard_lookup_skips_drop_on_producer():
    coordinator = _make_lookup_coordinator(producer=True)
    coordinator.find_longest_cache_hit(block_hashes=[], max_cache_hit_length=2048)
    assert _RecordingFA.calls[-1]["drop_eagle_block"] is False
    assert _RecordingMamba.calls[-1]["drop_eagle_block"] is False


def test_standard_lookup_skips_drop_on_standalone():
    coordinator = _make_lookup_coordinator(producer=False, standalone=True)
    coordinator.find_longest_cache_hit(block_hashes=[], max_cache_hit_length=2048)
    assert _RecordingFA.calls[-1]["drop_eagle_block"] is False
    assert _RecordingMamba.calls[-1]["drop_eagle_block"] is False


# ---------------------------------------------------------------------------
# Fix ③: per-group lookup drop gating + call-site signature
# ---------------------------------------------------------------------------


def test_per_group_lookup_drops_last_block_on_consumer():
    coordinator = _make_lookup_coordinator(producer=False)
    _, hit_lengths = coordinator.find_longest_cache_hit_per_group([], 2048)
    assert _RecordingFA.calls[-1]["drop_eagle_block"] is True
    assert _RecordingMamba.calls[-1]["drop_eagle_block"] is False
    assert hit_lengths == (2048, 2048)


def test_per_group_lookup_skips_drop_on_producer():
    coordinator = _make_lookup_coordinator(producer=True)
    _, hit_lengths = coordinator.find_longest_cache_hit_per_group([], 2048)
    assert _RecordingFA.calls[-1]["drop_eagle_block"] is False
    assert _RecordingMamba.calls[-1]["drop_eagle_block"] is False
    assert hit_lengths == (2048, 2048)


def test_per_group_lookup_skips_drop_on_standalone():
    coordinator = _make_lookup_coordinator(producer=False, standalone=True)
    _, hit_lengths = coordinator.find_longest_cache_hit_per_group([], 2048)
    assert _RecordingFA.calls[-1]["drop_eagle_block"] is False
    assert _RecordingMamba.calls[-1]["drop_eagle_block"] is False
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


# ---------------------------------------------------------------------------
# Fix ④: producer scheduler companion patch - the EAGLE one-page backoff in
# _mamba_block_aligned_split must be suppressed on the producer, otherwise the
# final full mamba-align state page is never materialized across a chunk
# boundary and hybrid hits stay one page (1536 tokens) short.
# ---------------------------------------------------------------------------


class _RecordingSplitScheduler:
    """Minimal scheduler double: records the drop bit seen by the original
    split implementation and echoes its arguments."""

    def __init__(
        self,
        *,
        drop_attr: str = "use_eagle",
        drop_value: bool = True,
        raise_in_split: bool = False,
        is_kv_producer: bool = False,
        is_kv_consumer: bool = False,
        kv_transfer_config_present: bool = True,
        vllm_config_present: bool = True,
    ):
        # Explicitly do NOT carry the other-era attribute: the wrapper must
        # pick the knob the running vLLM revision actually exposes.
        setattr(self, drop_attr, drop_value)
        self.raise_in_split = raise_in_split
        self.observed_drop_bits: list[bool] = []
        self.calls: list[tuple] = []
        if vllm_config_present:
            vllm_config = SimpleNamespace(model_config=SimpleNamespace())
            if kv_transfer_config_present:
                vllm_config.kv_transfer_config = SimpleNamespace(
                    is_kv_producer=is_kv_producer,
                    is_kv_consumer=is_kv_consumer,
                )
            else:
                vllm_config.kv_transfer_config = None
            self.vllm_config = vllm_config


def _split_observed_bit(scheduler) -> bool:
    if hasattr(scheduler, "use_eagle_block_drop"):
        return scheduler.use_eagle_block_drop
    return scheduler.use_eagle


def _fresh_split_scheduler(**kwargs):
    # One-off subclass so binding/replacing the method never pollutes the
    # shared base class used by the other cases.
    cls = type("_IsolatedSplitScheduler", (_RecordingSplitScheduler,), {})

    def _record(self, request, num_new_tokens, nlc=0, nec=0):
        self.observed_drop_bits.append(_split_observed_bit(self))
        self.calls.append((num_new_tokens, nlc, nec))
        if self.raise_in_split:
            raise RuntimeError("boom")
        return ("split", num_new_tokens)

    cls._mamba_block_aligned_split = _record  # type: ignore[attr-defined]
    return cls, cls(**kwargs)


def test_scheduler_split_patch_clears_use_eagle_for_producer():
    cls, scheduler = _fresh_split_scheduler(drop_value=True, is_kv_producer=True)
    _install_producer_mamba_block_aligned_split_patch(cls)
    result = scheduler._mamba_block_aligned_split("req", 1600)
    # The original implementation observes the drop as disabled...
    assert scheduler.observed_drop_bits == [False]
    assert result == ("split", 1600)
    # ...and the scheduler's own bit is restored afterwards.
    assert scheduler.use_eagle is True


def test_scheduler_split_patch_clears_use_eagle_block_drop_on_newer_vllm():
    cls, scheduler = _fresh_split_scheduler(
        drop_attr="use_eagle_block_drop",
        drop_value=True,
        is_kv_producer=True,
    )
    _install_producer_mamba_block_aligned_split_patch(cls)
    scheduler._mamba_block_aligned_split("req", 3200, 1536, 0)
    assert scheduler.observed_drop_bits == [False]
    assert scheduler.calls == [(3200, 1536, 0)]
    assert scheduler.use_eagle_block_drop is True
    assert not hasattr(scheduler, "use_eagle")


def test_scheduler_split_patch_restores_bit_on_exception():
    cls, scheduler = _fresh_split_scheduler(
        drop_value=True,
        raise_in_split=True,
        is_kv_producer=True,
    )
    _install_producer_mamba_block_aligned_split_patch(cls)
    with pytest.raises(RuntimeError, match="boom"):
        scheduler._mamba_block_aligned_split("req", 1600)
    assert scheduler.use_eagle is True


@pytest.mark.parametrize(
    "scheduler_kwargs",
    [
        # Pure decode consumer.
        {"is_kv_producer": False, "is_kv_consumer": True},
        # kv_both serves both roles: keep upstream drop behavior.
        {"is_kv_producer": True, "is_kv_consumer": True},
        # Connector configured but neither PD role: not a standalone, so the
        # drop stays on upstream behavior.
        {"is_kv_producer": False, "is_kv_consumer": False},
    ],
)
def test_scheduler_split_patch_passes_through_non_producer(scheduler_kwargs):
    cls, scheduler = _fresh_split_scheduler(drop_value=True, **scheduler_kwargs)
    _install_producer_mamba_block_aligned_split_patch(cls)
    result = scheduler._mamba_block_aligned_split("req", 1600)
    # The original sees the untouched bit and runs exactly once.
    assert scheduler.observed_drop_bits == [True]
    assert result == ("split", 1600)
    assert scheduler.use_eagle is True


@pytest.mark.parametrize(
    "scheduler_kwargs",
    [
        # Connector configured but kv_transfer_config None: standalone.
        {"kv_transfer_config_present": False},
        # No vllm_config at all: also reads as standalone.
        {"vllm_config_present": False},
    ],
)
def test_scheduler_split_patch_suppresses_drop_on_standalone(scheduler_kwargs):
    # A standalone instance has no connector: matched blocks are always
    # verified local prompt blocks, so the EAGLE backoff is suppressed (the
    # single-instance counterpart of the producer case) and the scheduler's
    # own bit is restored afterwards.
    cls, scheduler = _fresh_split_scheduler(drop_value=True, **scheduler_kwargs)
    _install_producer_mamba_block_aligned_split_patch(cls)
    result = scheduler._mamba_block_aligned_split("req", 1600)
    assert scheduler.observed_drop_bits == [False]
    assert result == ("split", 1600)
    assert scheduler.use_eagle is True


def test_scheduler_split_patch_is_idempotent():
    cls, _ = _fresh_split_scheduler()
    _install_producer_mamba_block_aligned_split_patch(cls)
    wrapped_once = cls._mamba_block_aligned_split
    _install_producer_mamba_block_aligned_split_patch(cls)
    assert cls._mamba_block_aligned_split is wrapped_once


def test_scheduler_split_patch_wraps_consumer_early_return_path():
    # Model the neighboring consumer/sparse-index wrapper's early return: it
    # consults use_eagle but deliberately never calls its saved original.
    # The producer wrapper must be outermost for this path to see False.
    cls, scheduler = _fresh_split_scheduler(drop_value=True, is_kv_producer=True)
    inner = cls._mamba_block_aligned_split

    def _consumer_early_return(self, request, num_new_tokens, nlc=0, nec=0):
        self.observed_drop_bits.append(self.use_eagle)
        return ("consumer-early-return", num_new_tokens)

    _consumer_early_return.__wrapped__ = inner  # type: ignore[attr-defined]
    cls._mamba_block_aligned_split = _consumer_early_return
    _install_producer_mamba_block_aligned_split_patch(cls)

    registered = cls._mamba_block_aligned_split
    assert registered is not _consumer_early_return
    assert registered.__wrapped__ is _consumer_early_return  # type: ignore[attr-defined]
    assert getattr(registered, "_ascend_producer_no_eagle_drop", False)
    assert scheduler._mamba_block_aligned_split("req", 1600) == (
        "consumer-early-return",
        1600,
    )
    assert scheduler.observed_drop_bits == [False]
    assert scheduler.use_eagle is True


def test_scheduler_split_patch_clears_and_restores_both_drop_attributes():
    cls, scheduler = _fresh_split_scheduler(drop_value=True, is_kv_producer=True)
    scheduler.use_eagle_block_drop = "dedicated-original"

    def _observe_both(self, request, num_new_tokens, nlc=0, nec=0):
        self.calls.append((self.use_eagle, self.use_eagle_block_drop))
        return num_new_tokens

    cls._mamba_block_aligned_split = _observe_both
    _install_producer_mamba_block_aligned_split_patch(cls)

    assert scheduler._mamba_block_aligned_split("req", 3200) == 3200
    assert scheduler.calls == [(False, False)]
    assert scheduler.use_eagle is True
    assert scheduler.use_eagle_block_drop == "dedicated-original"


def test_scheduler_split_patch_handles_producer_without_drop_attribute():
    cls, scheduler = _fresh_split_scheduler(drop_value=True, is_kv_producer=True)
    del scheduler.use_eagle

    def _attribute_free_split(self, request, num_new_tokens, nlc=0, nec=0):
        self.calls.append((num_new_tokens, nlc, nec))
        return ("split", num_new_tokens)

    cls._mamba_block_aligned_split = _attribute_free_split
    _install_producer_mamba_block_aligned_split_patch(cls)

    assert scheduler._mamba_block_aligned_split("req", 1600) == ("split", 1600)
    assert scheduler.observed_drop_bits == []
    assert scheduler.calls == [(1600, 0, 0)]
    assert not hasattr(scheduler, "use_eagle")
    assert not hasattr(scheduler, "use_eagle_block_drop")


def test_scheduler_split_patch_noop_without_split_method():
    class _BareScheduler:
        pass

    # Must neither bind anything nor raise.
    _install_producer_mamba_block_aligned_split_patch(_BareScheduler)
    assert not hasattr(_BareScheduler, "_mamba_block_aligned_split")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
