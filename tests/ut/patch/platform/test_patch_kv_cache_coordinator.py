# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the hybrid coordinator MTP + prefix-cache fix.

The Ascend patch keeps hybrid MTP prefix-cache read and write boundaries
consistent:

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
5. Replay retention and Mamba tail/checkpoint registration apply the same
   role-dependent drop policy. Sparse-retention tests use real managers and
   a BlockPool so a missing state or hash entry cannot be hidden by doubles.

These exercises run CPU-only: the heavy ``__init__`` is exercised with a
lightweight BlockPool/manager factory, lookup tests build the coordinator
with ``__new__`` and drive it through recording fake managers, and sparse
retention tests supply synthetic state blocks to the real manager/pool
implementation. The scheduler integration uses its actual split method.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.core.kv_cache_coordinator import SpecGroup
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
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
    _mamba_block_aligned_split,
)
from vllm_ascend.patch.platform.patch_mamba_manager import AscendMambaManager

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
        self.block_size = kwargs["kv_cache_spec"].block_size


class _FakeMambaManager:
    def __init__(self, **kwargs):
        self.use_eagle = False
        self.block_size = kwargs["kv_cache_spec"].block_size


def _fake_manager_factory(**kwargs):
    spec = kwargs["kv_cache_spec"]
    if isinstance(spec, MambaSpec):
        return _FakeMambaManager(**kwargs)
    return _FakeFAManager(**kwargs)


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
    mamba_eagle: bool = False,
):
    monkeypatch.setattr(mod, "BlockPool", _FakeBlockPool)
    monkeypatch.setattr(mod, "get_manager_for_kv_cache_spec", _fake_manager_factory)
    kv_cache_config = _hybrid_config(mamba_eagle=mamba_eagle)
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
    assert coordinator.skips_eagle_block_drop is True


def test_coordinator_non_producer_tag(monkeypatch):
    coordinator = _make_coordinator(
        monkeypatch,
        use_eagle=True,
        kv_transfer_config=_kv_transfer_config(is_kv_producer=False, is_kv_consumer=True),
    )
    assert coordinator.skips_eagle_block_drop is False


def test_coordinator_reads_standalone_drop_exemption(monkeypatch):
    # No kv-transfer config attached: a standalone instance. Not a PD
    # producer, but every content-hash match is a verified local prompt
    # block, so the EAGLE drop is suppressed just like on the producer.
    coordinator = _make_coordinator(monkeypatch, use_eagle=True, kv_transfer_config=None)
    assert coordinator.skips_eagle_block_drop is True


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


# Retention must be explicitly enabled here: Ascend's default None would
# conceal the replay-boundary mismatch by retaining every materialized state.
REPLAY_BLOCK_SIZE = 384
REPLAY_ROLES = [
    pytest.param(None, False, id="standalone"),
    pytest.param(_kv_transfer_config(is_kv_producer=True, is_kv_consumer=False), False, id="producer"),
    pytest.param(_kv_transfer_config(is_kv_producer=False, is_kv_consumer=True), True, id="consumer"),
    pytest.param(_kv_transfer_config(is_kv_producer=True, is_kv_consumer=True), True, id="both"),
]


def _make_retention_coordinator(
    monkeypatch, kv_transfer_config, use_eagle, mamba_block_size=REPLAY_BLOCK_SIZE, fa_block_size=REPLAY_BLOCK_SIZE
):
    # Exercise actual manager retention, hash registration and lookup. Only
    # bypass the global spec registry, which is initialized by engine startup.
    def real_manager_factory(kv_cache_spec, max_in_flight_tokens, max_model_len, **kwargs):
        manager_cls = AscendMambaManager if isinstance(kv_cache_spec, MambaSpec) else FullAttentionManager
        return manager_cls(kv_cache_spec=kv_cache_spec, **kwargs)

    monkeypatch.setattr(mod, "get_manager_for_kv_cache_spec", real_manager_factory)
    fa_spec = FullAttentionSpec(block_size=fa_block_size, num_kv_heads=1, head_size=8, dtype=torch.float32)
    mamba_spec = MambaSpec(
        block_size=mamba_block_size,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    config = KVCacheConfig(
        num_blocks=1024,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["fa"], fa_spec, False), KVCacheGroupSpec(["mamba"], mamba_spec, False)],
    )
    config.kv_transfer_config = kv_transfer_config
    coordinator = AscendHybridKVCacheCoordinator(
        kv_cache_config=config,
        max_model_len=150000,
        use_eagle=use_eagle,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=REPLAY_BLOCK_SIZE,
        scheduler_block_size=mamba_block_size,
    )
    coordinator.retention_interval = 0
    return coordinator


def _replay_request(prompt_tokens):
    return SimpleNamespace(
        request_id="prime",
        num_prompt_tokens=prompt_tokens,
        num_tokens=prompt_tokens,
        num_computed_tokens=0,
        shared_prefix_boundary=0,
        block_hashes=[BlockHash(i.to_bytes(8, "big")) for i in range(prompt_tokens // REPLAY_BLOCK_SIZE)],
    )


def _materialize_replay_states(coordinator, request, state_boundaries):
    full_manager, state_manager = coordinator.single_type_managers
    pool = coordinator.block_pool
    full_manager.req_to_blocks[request.request_id] = pool.get_new_blocks(
        (request.num_prompt_tokens + full_manager.block_size - 1) // full_manager.block_size
    )
    num_state_slots = (request.num_prompt_tokens + state_manager.block_size - 1) // state_manager.block_size
    state_blocks = [pool.null_block] * num_state_slots
    for boundary in state_boundaries:
        state_blocks[(boundary - 1) // state_manager.block_size] = pool.get_new_blocks(1)[0]
    state_manager.req_to_blocks[request.request_id] = state_blocks


@pytest.mark.parametrize(("kv_transfer_config", "role_needs_drop"), REPLAY_ROLES)
@pytest.mark.parametrize("use_eagle", [False, True], ids=["no-mtp", "mtp"])
@pytest.mark.parametrize("prompt_tokens", [131082, 131328], ids=["unaligned", "aligned"])
def test_sparse_retention_first_replay_uses_effective_drop(
    monkeypatch, kv_transfer_config, role_needs_drop, use_eagle, prompt_tokens
):
    coordinator = _make_retention_coordinator(monkeypatch, kv_transfer_config, use_eagle)
    request = _replay_request(prompt_tokens)
    replay_boundary = (prompt_tokens - 1) // REPLAY_BLOCK_SIZE * REPLAY_BLOCK_SIZE
    effective_drop = use_eagle and role_needs_drop
    if effective_drop:
        replay_boundary -= REPLAY_BLOCK_SIZE
    # Represent the state that the role-aware scheduler actually materializes;
    # a state at the neighboring boundary must not be invented by retention.
    _materialize_replay_states(coordinator, request, [replay_boundary])
    coordinator.cache_blocks(request, prompt_tokens)

    _, hit_tokens, uncached_tokens = coordinator.find_longest_cache_hit(request.block_hashes, prompt_tokens - 1)
    assert hit_tokens == replay_boundary
    assert uncached_tokens == 0
    assert coordinator.eagle_group_ids == ({0} if use_eagle else set())
    assert coordinator.single_type_managers[0].use_eagle is use_eagle
    assert coordinator.single_type_managers[1].drop_eagle_checkpoint_block is effective_drop


@pytest.mark.parametrize(("kv_transfer_config", "role_needs_drop"), REPLAY_ROLES)
def test_sparse_retention_aligned_prompt_retains_resend_and_extension(monkeypatch, kv_transfer_config, role_needs_drop):
    prompt_tokens = 131328
    coordinator = _make_retention_coordinator(monkeypatch, kv_transfer_config, True)
    request = _replay_request(prompt_tokens)
    resend_boundary = prompt_tokens - REPLAY_BLOCK_SIZE * (2 if role_needs_drop else 1)
    extension_boundary = prompt_tokens - (REPLAY_BLOCK_SIZE if role_needs_drop else 0)
    _materialize_replay_states(coordinator, request, [resend_boundary, extension_boundary])
    coordinator.cache_blocks(request, prompt_tokens)

    _, resend_hit, _ = coordinator.find_longest_cache_hit(request.block_hashes, prompt_tokens - 1)
    # A longer sibling has identical hashes through the original prompt. Its
    # P-1 cap is above that prompt, so its final aligned state is reusable.
    _, extension_hit, _ = coordinator.find_longest_cache_hit(
        request.block_hashes, prompt_tokens + REPLAY_BLOCK_SIZE - 1
    )
    assert resend_hit == resend_boundary
    assert extension_hit == extension_boundary


@pytest.mark.parametrize("kv_transfer_config", [None, _kv_transfer_config(is_kv_producer=True, is_kv_consumer=False)])
def test_sparse_retention_registers_non_dropped_partial_mamba_tail(monkeypatch, kv_transfer_config):
    coordinator = _make_retention_coordinator(monkeypatch, kv_transfer_config, True, mamba_block_size=1536)
    request = _replay_request(131082)
    boundary = 130944
    full_manager, state_manager = coordinator.single_type_managers
    pool = coordinator.block_pool
    full_manager.req_to_blocks[request.request_id] = pool.get_new_blocks(boundary // REPLAY_BLOCK_SIZE)
    state_blocks = [pool.null_block] * (boundary // state_manager.block_size + 1)
    state_block = pool.get_new_blocks(1)[0]
    state_blocks[-1] = state_block
    state_manager.req_to_blocks[request.request_id] = state_blocks

    coordinator.cache_blocks(request, boundary)

    assert state_block.block_hash_num_tokens == boundary
    _, hit_tokens, uncached_tokens = coordinator.find_longest_cache_hit(
        request.block_hashes, request.num_prompt_tokens - 1
    )
    assert hit_tokens == boundary
    assert uncached_tokens == 0


@pytest.mark.parametrize("fa_block_size", [REPLAY_BLOCK_SIZE, 1536], ids=["hash-sized-fa", "coarse-fa"])
def test_sparse_retention_hash_aligned_partial_tail_keeps_resend_and_extension(monkeypatch, fa_block_size):
    coordinator = _make_retention_coordinator(
        monkeypatch, None, True, mamba_block_size=1536, fa_block_size=fa_block_size
    )
    request = _replay_request(131328)
    resend_boundary = request.num_prompt_tokens - REPLAY_BLOCK_SIZE
    full_manager, state_manager = coordinator.single_type_managers
    pool = coordinator.block_pool
    full_manager.req_to_blocks[request.request_id] = pool.get_new_blocks(
        (request.num_prompt_tokens + full_manager.block_size - 1) // full_manager.block_size
    )
    tail_idx = resend_boundary // state_manager.block_size
    state_blocks = [pool.null_block] * (tail_idx + 1)
    resend_state = pool.get_new_blocks(1)[0]
    state_blocks[tail_idx] = resend_state
    state_manager.req_to_blocks[request.request_id] = state_blocks
    coordinator.cache_blocks(request, resend_boundary)
    assert resend_state.block_hash_num_tokens == resend_boundary

    # The next chunk writes a private state while the cached resend state
    # remains immutable, as in the manager's copy-on-write path.
    extension_state = pool.get_new_blocks(1)[0]
    state_blocks[tail_idx] = extension_state
    request.num_computed_tokens = resend_boundary
    coordinator.cache_blocks(request, request.num_prompt_tokens)
    assert extension_state.block_hash_num_tokens == request.num_prompt_tokens
    _, resend_hit, _ = coordinator.find_longest_cache_hit(request.block_hashes, request.num_prompt_tokens - 1)
    _, extension_hit, _ = coordinator.find_longest_cache_hit(
        request.block_hashes, request.num_prompt_tokens + REPLAY_BLOCK_SIZE - 1
    )
    assert resend_hit == resend_boundary
    assert extension_hit == request.num_prompt_tokens


@pytest.mark.parametrize("kv_transfer_config", [None, _kv_transfer_config(is_kv_producer=True, is_kv_consumer=False)])
@pytest.mark.parametrize("prompt_tokens", [131082, 131328], ids=["unaligned", "aligned"])
def test_scheduler_materializes_sparse_retention_first_replay_state(monkeypatch, kv_transfer_config, prompt_tokens):
    coordinator = _make_retention_coordinator(monkeypatch, kv_transfer_config, True)
    request = _replay_request(prompt_tokens)
    scheduler = SimpleNamespace(
        vllm_config=SimpleNamespace(kv_transfer_config=kv_transfer_config, model_config=SimpleNamespace()),
        cache_config=SimpleNamespace(block_size=REPLAY_BLOCK_SIZE),
        block_size=REPLAY_BLOCK_SIZE,
        hash_block_size=REPLAY_BLOCK_SIZE,
        use_eagle=True,
        use_eagle_block_drop=True,
        max_num_scheduled_tokens=16384,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=False,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_prefill_checkpoint_alignment=REPLAY_BLOCK_SIZE,
        mamba_shared_prefix_checkpoint=False,
        mamba_fine_grained_prefix_cache=False,
    )
    chunk_ends = []
    while request.num_computed_tokens < prompt_tokens:
        num_new_tokens = min(scheduler.max_num_scheduled_tokens, prompt_tokens - request.num_computed_tokens)
        num_new_tokens = _mamba_block_aligned_split(scheduler, request, num_new_tokens)
        assert num_new_tokens > 0
        request.num_computed_tokens += num_new_tokens
        if request.num_computed_tokens % REPLAY_BLOCK_SIZE == 0:
            chunk_ends.append(request.num_computed_tokens)
    replay_boundary = (prompt_tokens - 1) // REPLAY_BLOCK_SIZE * REPLAY_BLOCK_SIZE
    assert replay_boundary in chunk_ends
    # Give retention only states produced at actual scheduler chunk ends.
    # This catches a correct-looking mask retaining a nonexistent state.
    _materialize_replay_states(coordinator, request, chunk_ends)
    coordinator.cache_blocks(request, prompt_tokens)
    _, hit_tokens, _ = coordinator.find_longest_cache_hit(request.block_hashes, prompt_tokens - 1)
    assert hit_tokens == replay_boundary
    assert scheduler.use_eagle_block_drop is True


@pytest.mark.parametrize("skips_drop", [False, True], ids=["consumer", "standalone"])
def test_module_helper_cache_entry_preserves_role_policy_and_finalized_tokens(monkeypatch, skips_drop):
    # vLLM #61012 moved the replay helper outside the class. Emulate that
    # entry point even while the declared main2main pair uses the old method.
    helper = MagicMock(return_value=(130560,))
    monkeypatch.setattr(mod.vllm_kv_cache_coordinator, "replay_boundaries", helper, raising=False)
    coordinator = AscendHybridKVCacheCoordinator.__new__(AscendHybridKVCacheCoordinator)
    coordinator.skips_eagle_block_drop = skips_drop
    coordinator.scheduler_block_size = REPLAY_BLOCK_SIZE
    coordinator.eagle_group_ids = {0}
    coordinator.retention_interval = 0
    coordinator.single_type_managers = (
        SimpleNamespace(enable_caching=True, cache_blocks=MagicMock()),
        SimpleNamespace(enable_caching=True, cache_blocks=MagicMock()),
        SimpleNamespace(enable_caching=False, cache_blocks=MagicMock()),
    )
    coordinator.get_num_cacheable_tokens = MagicMock(side_effect=[130940, 130560])
    request = _replay_request(131082)
    coordinator.cache_blocks(request, 130944)

    boundaries = (131081, 131082) if skips_drop else (130560,)
    for manager, finalized_tokens in zip(coordinator.single_type_managers, (130940, 130560)):
        manager.cache_blocks.assert_called_once_with(
            request, finalized_tokens, retention_interval=0, replay_boundaries=boundaries
        )
    coordinator.single_type_managers[2].cache_blocks.assert_not_called()
    assert coordinator.get_num_cacheable_tokens.call_count == 2
    if skips_drop:
        helper.assert_not_called()
    else:
        helper.assert_called_once_with(request.num_prompt_tokens, REPLAY_BLOCK_SIZE, use_eagle=True)
    assert coordinator.eagle_group_ids == {0}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
