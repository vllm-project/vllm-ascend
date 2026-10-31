# SPDX-License-Identifier: Apache-2.0
# FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-runnable unit tests for ``offload_shared_indexer_only``.

Covers the three pure pieces of the shared-indexer-only offload mode:

1. ``SparseKVOffloadConfig`` validation (requires ``fused_op_type=fused_copy_sfa``);
2. the manager layer split (offloaded hot-row layers vs device-resident full
   layers) and the LIM group runtime (side-stream state, per-member onboarding
   dispatch through the ``npu_lim_copy_descriptors`` operator);
3. ``AscendSFAKVOffloadImpl`` gating: resident layers never take the fused
   exec_kv / resident-attention paths, and only owners with shared members
   install the LIM top-k selector.

``offload.sparse_copy`` and NPU streams are monkeypatched so no memfabric or
NPU runtime is needed.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ascend_config import SparseKVOffloadConfig
from vllm_ascend.attention.sfa_kv_offload import AscendSFAKVOffloadImpl
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    LIM_MISS_PLAN_MAX_WIDTH,
    SparseKVOffloadManager,
)

BLOCK = 128

_MANAGER_MODULE = "vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager"


class _nullstream:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@contextmanager
def _npu_stubs():
    """Stub the torch_npu stream/event surface used by the group runtime."""
    with (
        patch(
            f"{_MANAGER_MODULE}.torch_npu.npu.Stream",
            side_effect=lambda device=None: SimpleNamespace(wait_event=lambda ev: None),
        ),
        patch(
            f"{_MANAGER_MODULE}.torch_npu.npu.Event",
            side_effect=lambda: SimpleNamespace(record=lambda: None),
        ),
        patch(
            f"{_MANAGER_MODULE}.torch_npu.npu.current_stream",
            side_effect=lambda: SimpleNamespace(
                record_event=lambda event: None,
                wait_event=lambda event: None,
            ),
        ),
        patch(f"{_MANAGER_MODULE}.torch_npu.npu.stream", side_effect=lambda s: _nullstream()),
    ):
        yield


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def test_shared_indexer_only_requires_fused_pipeline():
    # The baseline removed fused_op_type: the fused Copy-SFA path is the
    # default and use_fused_overlap selects the alternative operator, so the
    # requirement is expressed as mutual exclusion with use_fused_overlap.
    with pytest.raises(ValueError, match="offload_shared_indexer_only requires"):
        SparseKVOffloadConfig(
            enabled=True,
            offload_shared_indexer_only=True,
            use_fused_overlap=True,
        )
    with pytest.raises(ValueError, match="miss_onboard_mode"):
        SparseKVOffloadConfig(
            enabled=True,
            miss_onboard_mode="side_stream",
        )


def test_shared_indexer_only_accepts_fused_pipeline():
    config = SparseKVOffloadConfig(
        enabled=True,
        offload_shared_indexer_only=True,
        topk_buffer_size=4096,
        dram_size_per_dp_GB=256,
    )
    assert config.offload_shared_indexer_only
    assert config.use_fused_copy_sfa
    assert not config.use_side_stream_onboard
    config_side = SparseKVOffloadConfig(
        enabled=True,
        offload_shared_indexer_only=True,
        miss_onboard_mode="side_stream",
        topk_buffer_size=4096,
        dram_size_per_dp_GB=256,
    )
    assert config_side.use_side_stream_onboard


def test_shared_indexer_only_defaults_off():
    config = SparseKVOffloadConfig(enabled=True)
    assert not config.offload_shared_indexer_only


# ---------------------------------------------------------------------------
# Manager: layer split + LIM group runtime
# ---------------------------------------------------------------------------
def _make_manager():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.block_size = BLOCK
    manager.topk_buffer_size = 4096
    manager.max_num_tokens = 8192
    manager.num_target_layers = 5
    manager.tp_rank = 0
    manager.layer_name_to_offload_id = {}
    manager.full_resident_layer_names = []
    manager.offload_layer_names = []
    # The baseline derives host-stored (offload) layers from the KV cache
    # groups' specs; default to no groups for the manager-level tests.
    manager.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    # Main-KV row dims published by register_kv_caches and forwarded to the
    # onboarding operator through params.
    manager.head_dim_k = 512
    manager.head_dim_v = 64
    return manager


def _host_spec_group(*layer_names):
    return SimpleNamespace(layer_names=list(layer_names), kv_cache_spec=SimpleNamespace(store_on_host=True))


def _device_spec_group(*layer_names):
    return SimpleNamespace(layer_names=list(layer_names), kv_cache_spec=SimpleNamespace(store_on_host=False))


def test_register_layers_splits_full_resident():
    manager = _make_manager()
    # Production shape: full-resident layers carry store_on_host=False specs
    # (their own KV cache group), offloaded layers store_on_host=True.
    manager.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            _device_spec_group("model.layers.0.self_attn.attn"),
            _host_spec_group("model.layers.1.self_attn.attn", "model.layers.2.self_attn.attn"),
        ]
    )
    kv_caches = {
        "model.layers.0.self_attn.attn": (None, None),
        "model.layers.1.self_attn.attn": (None, None),
        "model.layers.2.self_attn.attn": (None, None),
        "model.layers.0.self_attn.indexer.k_cache": (None,),
    }
    manager._register_offload_layers(
        kv_caches,
        full_resident_layer_names=["model.layers.0.self_attn.attn"],
    )
    assert manager.offload_layer_names == [
        "model.layers.1.self_attn.attn",
        "model.layers.2.self_attn.attn",
    ]
    assert manager.full_resident_layer_names == ["model.layers.0.self_attn.attn"]
    assert manager.layer_name_to_offload_id == {
        "model.layers.1.self_attn.attn": 0,
        "model.layers.2.self_attn.attn": 1,
    }


def test_register_layers_without_resident_keeps_legacy_order():
    manager = _make_manager()
    manager.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[_host_spec_group("model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn")]
    )
    kv_caches = {
        "model.layers.0.self_attn.attn": (None, None),
        "model.layers.1.self_attn.attn": (None, None),
    }
    manager._register_offload_layers(kv_caches)
    assert manager.offload_layer_names == list(kv_caches)
    assert manager.full_resident_layer_names == []


def _group_owner_and_members(hot=4096, k_dim=512, v_dim=64, miss_rows=2):
    """Build an owner SimpleNamespace plus manager bases for two members."""
    owner = SimpleNamespace(
        layer_name="model.layers.2.self_attn.attn",
        _current_layer_name=None,
        lim_miss_src=torch.full((miss_rows, LIM_MISS_PLAN_MAX_WIDTH), -1, dtype=torch.int32),
        lim_miss_dst=torch.full((miss_rows, LIM_MISS_PLAN_MAX_WIDTH), -1, dtype=torch.int32),
        lim_misses=torch.zeros(miss_rows, dtype=torch.int32),
    )
    host_k = torch.zeros(1000 * BLOCK, k_dim, dtype=torch.bfloat16)
    host_v = torch.zeros(1000 * BLOCK, v_dim, dtype=torch.bfloat16)
    dev_k = torch.zeros(64 * (hot + 2 * BLOCK), k_dim, dtype=torch.bfloat16)
    dev_v = torch.zeros(64 * (hot + 2 * BLOCK), v_dim, dtype=torch.bfloat16)
    token_bytes = torch.tensor([[k_dim * 2], [v_dim * 2]], dtype=torch.int64).view(2, 1, 1)
    metadata = SimpleNamespace(
        copy_sfa_pool_entries=torch.tensor([0, 1], dtype=torch.int32),
        copy_sfa_source_block_table=torch.tensor([[5, 7, -1], [1, 2, 3]], dtype=torch.int32),
        copy_sfa_token_bytes=token_bytes,
    )
    manager = _make_manager()
    manager.head_dim_k = k_dim
    manager.head_dim_v = v_dim
    manager.copy_sfa_host_bases = [
        torch.tensor([[host_k.data_ptr()], [host_v.data_ptr()]], dtype=torch.int64),
        torch.tensor([[host_k.data_ptr()], [host_v.data_ptr()]], dtype=torch.int64),
    ]
    manager.copy_sfa_device_bases = [
        torch.tensor([[dev_k.data_ptr()], [dev_v.data_ptr()]], dtype=torch.int64),
        torch.tensor([[dev_k.data_ptr() + 1], [dev_v.data_ptr() + 1]], dtype=torch.int64),
    ]
    manager._lim_group_states = {}
    manager._lim_side_stream = None
    manager._lim_group_ksc_params = None
    manager.layer_name_to_offload_id = {
        "model.layers.3.self_attn.attn": 0,
        "model.layers.4.self_attn.attn": 1,
    }
    # configure_lim_groups probes the device of a registered buffer; a plain
    # CPU tensor keeps the CPU-runnable contract of this suite.
    manager.topk_buffers_k = [torch.zeros(4)]
    return manager, owner, metadata


def test_configure_lim_groups_builds_state():
    manager, owner, _ = _group_owner_and_members()
    with _npu_stubs(), patch(f"{_MANAGER_MODULE}._LIM_DESC_OP", lambda *a, **k: None):
        manager.configure_lim_groups(
            {
                "model.layers.2.self_attn.attn": [
                    "model.layers.3.self_attn.attn",
                    "model.layers.4.self_attn.attn",
                ],
                "model.layers.0.self_attn.attn": [],  # singleton owner: no state
            }
        )
    assert set(manager._lim_group_states) == {"model.layers.2.self_attn.attn"}
    state = manager._lim_group_states["model.layers.2.self_attn.attn"]
    assert state.member_names == [
        "model.layers.3.self_attn.attn",
        "model.layers.4.self_attn.attn",
    ]
    assert set(state.done_events) == set(state.member_names)
    assert manager._lim_group_state(owner) is state
    # Deployment constants are created before capture, with the model's own
    # KV dims rather than the kernel's compile-time maxima.
    assert manager._lim_group_ksc_params is not None
    assert manager._lim_group_ksc_params.tolist() == [
        BLOCK,
        manager.topk_buffer_size,
        LIM_MISS_PLAN_MAX_WIDTH,
        manager.head_dim_k,
        manager.head_dim_v,
    ]
    # The per-step geometry tensor starts zeroed and is filled on launch;
    # it is owned by the group, not the manager.
    assert state.ksc_geometry is not None
    assert state.ksc_geometry.tolist() == [0, 0]


def test_configure_lim_groups_geometry_is_per_group():
    """Each group owns a private geometry tensor: the side-stream kernel
    reads it asynchronously, so a later group's launch-time fill must never
    overwrite the values through a manager-level shared tensor while an
    earlier group's kernel is still in flight."""
    manager, _, _ = _group_owner_and_members()
    with _npu_stubs(), patch(f"{_MANAGER_MODULE}._LIM_DESC_OP", lambda *a, **k: None):
        manager.configure_lim_groups(
            {
                "model.layers.2.self_attn.attn": ["model.layers.3.self_attn.attn"],
                "model.layers.0.self_attn.attn": ["model.layers.4.self_attn.attn"],
            }
        )
    first = manager._lim_group_states["model.layers.2.self_attn.attn"]
    second = manager._lim_group_states["model.layers.0.self_attn.attn"]
    assert first.ksc_geometry.data_ptr() != second.ksc_geometry.data_ptr()


def test_configure_lim_groups_rejects_oversized_head_dims():
    manager, _, _ = _group_owner_and_members()
    manager.head_dim_k = 1024  # > LIM_KV_DIM_MAX
    with (
        _npu_stubs(),
        patch(f"{_MANAGER_MODULE}._LIM_DESC_OP", lambda *a, **k: None),
        pytest.raises(RuntimeError, match="kv_lora_rank <="),
    ):
        manager.configure_lim_groups({"model.layers.2.self_attn.attn": ["model.layers.3.self_attn.attn"]})


def test_configure_lim_groups_empty_is_noop():
    manager, _, _ = _group_owner_and_members()
    manager.configure_lim_groups({})
    assert manager._lim_group_states == {}
    assert manager._lim_side_stream is None
    assert manager._lim_group_ksc_params is None


def test_launch_lim_group_onboard_inline_fallback_without_op():
    """Without the KSC binding the launch is a no-op: members fall back to
    inline miss resolution inside the fused attention operator."""
    manager, owner, metadata = _group_owner_and_members()
    with _npu_stubs(), patch(f"{_MANAGER_MODULE}._LIM_DESC_OP", None):
        manager.configure_lim_groups(
            {
                "model.layers.2.self_attn.attn": [
                    "model.layers.3.self_attn.attn",
                    "model.layers.4.self_attn.attn",
                ]
            }
        )
        assert manager.lim_group_onboard_ready is False
        assert manager._lim_side_stream is None
        # No group states without the operator: nothing to pre-copy.
        assert manager._lim_group_states == {}
        # The launch is a safe no-op even with a bound state-like owner.
        manager.launch_lim_group_onboard(owner, metadata)
        assert manager._lim_group_ksc_params is None


def test_launch_lim_group_ksc_per_member():
    """KSC path: one operator launch + one done event per member, side stream."""
    manager, owner, metadata = _group_owner_and_members()
    metadata.num_decode_tokens = 1
    metadata.token_to_req = torch.zeros(1, dtype=torch.int32)
    metadata.copy_sfa_seq_lens = torch.tensor([1000], dtype=torch.int32)
    metadata.copy_sfa_prefix_lens = torch.tensor([896], dtype=torch.int32)
    metadata.copy_sfa_cache_tokens = torch.tensor([4096], dtype=torch.int32)
    metadata.copy_sfa_query_ends = torch.tensor([1], dtype=torch.int32)
    owner.lim_topk_src = torch.zeros((1, 1, 2048), dtype=torch.int32)
    owner.lim_topk_dst = torch.zeros((1, 1, 2048), dtype=torch.int32)
    recorded = []
    op_calls = []

    def fake_op(*args, **kwargs):
        op_calls.append(args)

    with (
        _npu_stubs(),
        patch(f"{_MANAGER_MODULE}._LIM_DESC_OP", fake_op),
    ):
        manager.configure_lim_groups(
            {"model.layers.2.self_attn.attn": ["model.layers.3.self_attn.attn", "model.layers.4.self_attn.attn"]}
        )
        state = manager._lim_group_states["model.layers.2.self_attn.attn"]
        for name in state.member_names:
            state.done_events[name].record = (lambda n: lambda: recorded.append(n))(name)
        manager._lim_group_state = lambda impl: state
        manager.launch_lim_group_onboard(owner, metadata)
    # One launch per member, in member order, each followed by its event.
    assert len(op_calls) == 2
    assert recorded == state.member_names
    # ABI: (miss_src, miss_dst, miss_counts, src_table, pool_entries,
    #       host_ptrs, dev_ptrs, params, active_geometry)
    assert torch.equal(op_calls[0][5], manager.copy_sfa_host_bases[0].view(2))
    assert torch.equal(op_calls[1][6], manager.copy_sfa_device_bases[1].view(2))
    assert op_calls[0][7].tolist() == [
        BLOCK,
        manager.topk_buffer_size,
        LIM_MISS_PLAN_MAX_WIDTH,
        manager.head_dim_k,
        manager.head_dim_v,
    ]
    # active_geometry = [active_rows, block_table_width]: the row bound comes
    # from the current batch, the stride from the live block table (not from
    # max_model_len / block_size).
    geometry = op_calls[0][8]
    assert geometry.tolist() == [
        metadata.copy_sfa_pool_entries.numel(),
        metadata.copy_sfa_source_block_table.shape[1],
    ]
    assert geometry.data_ptr() == state.ksc_geometry.data_ptr()


def test_launch_lim_group_onboard_bounds_kernel_by_active_count():
    """Batch shrink: stale counts beyond the active batch stay untouched but
    are unreachable — the kernel loop is bounded by active_geometry[0] (the
    current batch count), which travels to every launch, so no per-step
    zeroing launch is needed."""
    manager, owner, metadata = _group_owner_and_members(miss_rows=4)
    owner.lim_misses[2:] = 7  # stale counts from a larger previous step
    token = SimpleNamespace(step="current-pass")
    with (
        _npu_stubs(),
        patch(f"{_MANAGER_MODULE}._LIM_DESC_OP", lambda *a, **k: None),
        patch(f"{_MANAGER_MODULE}.is_forward_context_available", return_value=True),
        patch(f"{_MANAGER_MODULE}.get_forward_context", return_value=token),
    ):
        manager.configure_lim_groups({"model.layers.2.self_attn.attn": ["model.layers.3.self_attn.attn"]})
        manager._lim_group_state = lambda impl: manager._lim_group_states["model.layers.2.self_attn.attn"]
        manager.launch_lim_group_onboard(owner, metadata)
        assert manager.lim_group_launched_for(owner) is True
        # A different forward pass (or no context at all): not launched.
        with patch(f"{_MANAGER_MODULE}.get_forward_context", return_value=SimpleNamespace(step="later")):
            assert manager.lim_group_launched_for(owner) is False
        with patch(f"{_MANAGER_MODULE}.is_forward_context_available", return_value=False):
            assert manager.lim_group_launched_for(owner) is False
    assert metadata.copy_sfa_pool_entries.numel() == 2
    # The stale tail is not rewritten; the kernel never reads it because the
    # geometry's row bound is the current batch count.
    assert owner.lim_misses.tolist() == [0, 0, 7, 7]
    state = manager._lim_group_states["model.layers.2.self_attn.attn"]
    assert state.ksc_geometry.tolist() == [2, 3]
    # The launch token is the forward context, not the attention metadata
    # (owners and members get distinct metadata objects per KV cache group).
    assert state.last_launch_token is token


def test_wait_lim_group_onboard_unknown_member_is_noop():
    manager, owner, _ = _group_owner_and_members()
    waited = []

    def _current_stream():
        return SimpleNamespace(record_event=lambda event: None, wait_event=waited.append)

    with (
        _npu_stubs(),
        patch(f"{_MANAGER_MODULE}._LIM_DESC_OP", lambda *a, **k: None),
        patch(f"{_MANAGER_MODULE}.torch_npu.npu.current_stream", side_effect=_current_stream),
    ):
        manager.configure_lim_groups({"model.layers.2.self_attn.attn": ["model.layers.3.self_attn.attn"]})
        known = SimpleNamespace(layer_name="model.layers.3.self_attn.attn", _current_layer_name=None)
        unknown = SimpleNamespace(layer_name="model.layers.9.self_attn.attn", _current_layer_name=None)
        unknown_owner = SimpleNamespace(layer_name="model.layers.77.self_attn.attn", _current_layer_name=None)
        manager.wait_lim_group_onboard(owner, unknown)
        assert waited == []
        manager.wait_lim_group_onboard(owner, known)
        assert len(waited) == 1
        manager.wait_lim_group_onboard(unknown_owner, known)
        assert len(waited) == 1


def test_stage_group_d2h_prefix_flush_and_step_cache():
    """The [part, member, token] staging layout keeps per-member writes
    contiguous, flushes via a sliced reshape, and the per-step geometry
    cache is keyed by slot_mapping tensor identity."""
    manager = _make_manager()
    manager.token_size_bytes_k = 1024
    manager.token_size_bytes_v = 128
    manager.d2h_token_indices_npu = torch.arange(manager.max_num_tokens, dtype=torch.int64)
    # configure_lim_groups probes the device of a registered buffer; a plain
    # CPU tensor keeps the CPU-runnable contract of this suite.
    manager.topk_buffers_k = [torch.zeros(4)]
    with _npu_stubs(), patch(f"{_MANAGER_MODULE}._LIM_DESC_OP", None):
        # Inline mode (no op) still allocates the per-group staging buffers.
        manager.configure_lim_groups({"owner.attn": ["m0.attn", "m1.attn"]})
    src_buf, dst_buf, len_buf = manager._group_d2h_buffers["owner.attn"]
    assert src_buf.shape == (2, 2, manager.max_num_tokens)

    copies = []

    def fake_sparse_copy(src, dst, lengths, count, device):
        copies.append((src, dst, lengths, count))
        return 0

    k0 = torch.randn(3, 256, dtype=torch.float32)
    v0 = torch.randn(3, 32, dtype=torch.float32)
    k1 = torch.randn(3, 256, dtype=torch.float32)
    v1 = torch.randn(3, 32, dtype=torch.float32)
    slots = torch.tensor([4, 5, -1], dtype=torch.int64)
    cpu_k = torch.zeros(16, 256, dtype=torch.float32)
    cpu_v = torch.zeros(16, 32, dtype=torch.float32)

    with patch(f"{_MANAGER_MODULE}.offload.sparse_copy", side_effect=fake_sparse_copy):
        # Unknown groups fail loudly instead of staging into a wrong buffer.
        with pytest.raises(RuntimeError, match="not configured"):
            manager.stage_group_d2h("missing.attn", 0, True, k0, v0, slots, cpu_k, cpu_v)
        # Member 0 only stages; the flush waits for the last member.
        manager.stage_group_d2h("owner.attn", 0, False, k0, v0, slots, cpu_k, cpu_v)
        assert copies == []
        manager.stage_group_d2h("owner.attn", 1, True, k1, v1, slots, cpu_k, cpu_v)
    assert len(copies) == 1
    flushed_src, flushed_dst, flushed_len, count = copies[0]
    # The flush passes contiguous views of the sliced group prefix.
    assert flushed_src.is_contiguous()
    assert flushed_dst.is_contiguous()
    assert flushed_len.is_contiguous()
    assert flushed_src.numel() == 12 and count.item() == 12
    # Layout [part, member, token]: the flush carries every member's
    # per-token source addresses; verify as a multiset (exact ordering
    # is an implementation detail of the slice+reshape).
    k0p, k1p, v0p, v1p = (int(t.data_ptr()) for t in (k0, k1, v0, v1))
    expected_src = sorted(
        [k0p, k0p + 1024, k0p + 2048]
        + [k1p, k1p + 1024, k1p + 2048]
        + [v0p, v0p + 128, v0p + 256]
        + [v1p, v1p + 128, v1p + 256]
    )
    assert sorted(flushed_src.tolist()) == expected_src
    base_k, base_v = int(cpu_k.data_ptr()), int(cpu_v.data_ptr())
    # [2,m,t] layout: k_m0(t0,t1,t2), k_m1(t0,t1,t2), v_m0(...), v_m1(...)
    _dst = flushed_dst.tolist()
    assert _dst[0:3] == [base_k + 4 * 1024, base_k + 5 * 1024, base_k + 0]
    assert _dst[3:6] == [base_k + 4 * 1024, base_k + 5 * 1024, base_k + 0]
    assert _dst[6:9] == [base_v + 4 * 128, base_v + 5 * 128, base_v + 0]
    assert _dst[9:12] == [base_v + 4 * 128, base_v + 5 * 128, base_v + 0]
    # Slot -1 is invalid: zero length (its clamped destination is harmless).
    _len = flushed_len.tolist()
    assert _len[0:3] == [1024, 1024, 0]  # k_m0: slots 4,5 valid, -1 invalid
    assert _len[3:6] == [1024, 1024, 0]  # k_m1
    assert _len[6:9] == [128, 128, 0]  # v_m0
    assert _len[9:12] == [128, 128, 0]  # v_m1

    # Step cache: identity-keyed on the slot tensor, invalidated by writes.
    state = manager._group_d2h_step
    assert state["slot_mapping"] is slots
    assert manager._group_d2h_step_state(slots, 3, 16, slots.device) is state
    slots[0] = 6  # in-place write bumps the tensor version
    assert manager._group_d2h_step_state(slots, 3, 16, slots.device) is not state


# ---------------------------------------------------------------------------
# PD read thread: per-rank paged destinations
# ---------------------------------------------------------------------------
def _make_pd_layer(per_rank_dest=False, topk_base_index=None):
    layer = {
        "layer_name": "model.layers.2.self_attn",
        "pool_idx": 0,
        "offload_id": 0,
        "per_rank_dest": per_rank_dest,
        "p_k_base": 1000,
        "p_v_base": 2000,
        "p_k_len": 10,
        "p_v_len": 20,
        "k_cpu_ptr": 3000,
        "v_cpu_ptr": 4000,
        "indexer": None,
        "scale": None,
    }
    if topk_base_index is not None:
        layer["topk_base_index"] = topk_base_index
    return layer


def _make_pd_thread(tp_rank=1, tp_size=2):
    from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.read_thread import (
        ConsumerReadState,
        MembPullReadThread,
    )

    thread = MembPullReadThread.__new__(MembPullReadThread)
    thread.tp_rank = tp_rank
    thread._state = ConsumerReadState(
        num_blocks=16,
        tp_size=tp_size,
        layer_metadata={},
        main_name_to_idx={"model.layers.2.self_attn": 0},
        cpu_pools=[],
        main_gva_bases=[(3000, 4000)],
        main_block_lens=[(10, 20)],
        indexer_tensors=[],
        indexer_scale_tensors=[],
        dest_blocks_by_req={"req-0": ([0, 1, 2, 3], [])},
    )
    return thread


def test_per_rank_dest_pulls_all_main_blocks_on_every_rank():
    # A device-resident full layer on tp_rank 1 of 2: the whole main block
    # list is pulled into this rank's HBM, unlike the shared CPU pool split.
    thread = _make_pd_thread(tp_rank=1, tp_size=2)
    layer = _make_pd_layer(per_rank_dest=True)
    local, peer, lengths, info = thread._build_req_descriptors(
        layer,
        "req-0",
        p_main_block_ids=[0, 1, 2, 3],
        p_indexer_block_ids=[],
        want_info=True,
    )
    assert local == [3000, 4000]
    assert peer == [1000, 2000]
    assert lengths == [40, 80]
    assert info is not None and info["n_main"] == 4


def test_shared_pool_layer_still_splits_main_across_tp_ranks():
    thread = _make_pd_thread(tp_rank=1, tp_size=2)
    layer = _make_pd_layer(per_rank_dest=False)
    local, peer, lengths, _ = thread._build_req_descriptors(
        layer,
        "req-0",
        p_main_block_ids=[0, 1, 2, 3],
        p_indexer_block_ids=[],
        want_info=False,
    )
    # Rank 1 owns blocks 2..3 -> HBM-split destination offsets 3020/4040.
    assert local == [3020, 4040]
    assert peer == [1020, 2040]
    assert lengths == [20, 40]


def test_tail_d2d_skipped_when_layer_has_no_hot_rows():
    """Device-resident full layers have no hot rows, so the tail D2D skip is
    keyed on topk_base_index == -1 (not on per_rank_dest itself: a per-rank
    destination with a valid hot-row base still appends tail descriptors)."""
    from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.protocol import (
        CopySfaTailDest,
    )

    thread = _make_pd_thread()
    thread._state.copy_sfa_tail_by_req["req-0"] = CopySfaTailDest(pool_slot=1, tail_tokens=3, tail_block_index=0)
    thread._state.topk_k_bases = [100_000]
    thread._state.topk_v_bases = [200_000]
    thread._state.topk_row_tokens = 256
    thread._state.topk_hot_tokens = 128
    thread._state.block_size = 128

    def _run(layer):
        peer_chunks: list = []
        local_chunks: list = []
        length_chunks: list = []
        thread._append_copy_sfa_tail_descriptors(layer, "req-0", [5], 0, peer_chunks, local_chunks, length_chunks)
        return peer_chunks, local_chunks, length_chunks

    # Device-resident layer: no hot-row base -> skip.
    layer = _make_pd_layer(per_rank_dest=True, topk_base_index=-1)
    layer["p_k_len"] = 1280
    layer["p_v_len"] = 2560
    assert all(chunk == [] for chunk in _run(layer))
    # Control: per_rank_dest alone does not gate the skip — a valid hot-row
    # base still produces tail descriptors.
    layer = _make_pd_layer(per_rank_dest=True, topk_base_index=0)
    layer["p_k_len"] = 1280
    layer["p_v_len"] = 2560
    peer_chunks, local_chunks, length_chunks = _run(layer)
    assert peer_chunks and local_chunks and length_chunks


# ---------------------------------------------------------------------------
# Impl gating
# ---------------------------------------------------------------------------
def _make_impl(*, skip_topk, is_mtp, shared_only, members=()):
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.skip_topk = skip_topk
    impl._is_mtp_layer = is_mtp
    impl.lim_shared_only = shared_only
    # Exercise the same production predicate as __init__ instead of copying it.
    impl.lim_full_resident = AscendSFAKVOffloadImpl.is_full_resident_layer(
        offload_shared_indexer_only=shared_only,
        skip_topk=skip_topk,
        is_mtp_layer=is_mtp,
    )
    impl._lim_share_members = list(members)
    # __init__ also allocates this lazily-used pad block; mirror the default.
    impl._lim_topk_pad_zeros = None
    return impl


def test_full_resident_classification():
    # Full non-MTP owner in shared-only mode: resident.
    assert _make_impl(skip_topk=False, is_mtp=False, shared_only=True).lim_full_resident
    # Shared (skip_topk) layers are never resident.
    assert not _make_impl(skip_topk=True, is_mtp=False, shared_only=True).lim_full_resident
    # The MTP layer keeps the offloaded LIM flow.
    assert not _make_impl(skip_topk=False, is_mtp=True, shared_only=True).lim_full_resident
    # Legacy mode: nothing is resident.
    assert not _make_impl(skip_topk=False, is_mtp=False, shared_only=False).lim_full_resident


def test_prepare_indexer_metadata_selector_gating():
    fused_metadata = SimpleNamespace(fused_copy_sfa_enabled=True)

    class FakeIndexerMetadata:
        topk_selector = "unset"

    # Resident singleton owner: standard selection (no LIM).
    impl = _make_impl(skip_topk=False, is_mtp=False, shared_only=True)
    indexer_metadata = FakeIndexerMetadata()
    impl._prepare_indexer_metadata(indexer_metadata, fused_metadata)
    assert indexer_metadata.topk_selector is None

    # Resident owner with shared members: LIM still runs for the group.
    impl = _make_impl(
        skip_topk=False,
        is_mtp=False,
        shared_only=True,
        members=[SimpleNamespace()],
    )
    indexer_metadata = FakeIndexerMetadata()
    impl._prepare_indexer_metadata(indexer_metadata, fused_metadata)
    assert indexer_metadata.topk_selector == impl._lim_select

    # MTP owner: LIM as in legacy mode.
    impl = _make_impl(skip_topk=False, is_mtp=True, shared_only=True)
    indexer_metadata = FakeIndexerMetadata()
    impl._prepare_indexer_metadata(indexer_metadata, fused_metadata)
    assert indexer_metadata.topk_selector == impl._lim_select

    # Legacy owner (mode off): LIM.
    impl = _make_impl(skip_topk=False, is_mtp=False, shared_only=False)
    indexer_metadata = FakeIndexerMetadata()
    impl._prepare_indexer_metadata(indexer_metadata, fused_metadata)
    assert indexer_metadata.topk_selector == impl._lim_select

    # Shared layer: never selects.
    impl = _make_impl(skip_topk=True, is_mtp=False, shared_only=True)
    indexer_metadata = FakeIndexerMetadata()
    impl._prepare_indexer_metadata(indexer_metadata, fused_metadata)
    assert indexer_metadata.topk_selector is None


def test_exec_kv_resident_layer_takes_standard_path():
    impl = _make_impl(skip_topk=False, is_mtp=False, shared_only=True)
    impl._current_layer_name = "model.layers.2.self_attn.attn"
    impl.layer_name = impl._current_layer_name
    import vllm_ascend.attention.sfa_v1 as sfa_v1

    def fail_offload(*args, **kwargs):
        raise AssertionError("offload exec_kv path taken for a resident layer")

    with (
        patch.object(type(impl), "_is_decode_only", fail_offload),
        patch.object(sfa_v1.AscendSFAImpl, "exec_kv", return_value=("k_pe", "k_nope")) as standard_exec,
    ):
        result = impl.exec_kv(None, None, None, (None, None), None, None)
    assert result == ("k_pe", "k_nope")
    assert standard_exec.called


def test_attention_resident_layer_takes_standard_path():
    impl = _make_impl(skip_topk=False, is_mtp=False, shared_only=True)
    impl._current_layer_name = "model.layers.2.self_attn.attn"
    impl.layer_name = impl._current_layer_name
    sentinel = object()
    import vllm_ascend.attention.sfa_v1 as sfa_v1

    with (
        patch(
            "vllm_ascend.attention.sfa_kv_offload.get_sparse_kv_offload_manager",
            lambda: SimpleNamespace(_get_offload_layer_id=lambda name: 0),
        ),
        patch.object(
            sfa_v1.AscendSFAImpl,
            "_execute_sparse_flash_attention_process",
            return_value=sentinel,
        ) as standard_attn,
    ):
        topk = torch.zeros(1, 1, 2048, dtype=torch.int32)
        query = torch.zeros(1, 1, 128)
        result = impl._execute_sparse_flash_attention_process(query, query, (None, None), topk, None, None, None)
    assert result is sentinel
    assert standard_attn.called


def test_attention_resident_layer_pads_sp_dummy_topk_rows():
    """Regression: SP-MoE/DSA-CP pad the query rows to the TP size while the
    lightning indexer only returns rows for real tokens — the resident path
    must hand the standard attention a matching 3D selection whose trailing
    dummy rows are zero indices, and the zero block must be reused across
    calls instead of reallocated per forward."""
    impl = _make_impl(skip_topk=False, is_mtp=False, shared_only=True)
    impl._current_layer_name = "model.layers.2.self_attn.attn"
    impl.layer_name = impl._current_layer_name
    import vllm_ascend.attention.sfa_v1 as sfa_v1

    forwarded = []
    sentinel = object()

    def fake_standard(self, ql_nope, q_pe, kv_cache, topk_indices, *args, **kwargs):
        forwarded.append(topk_indices)
        return sentinel

    with (
        patch(
            "vllm_ascend.attention.sfa_kv_offload.get_sparse_kv_offload_manager",
            lambda: SimpleNamespace(_get_offload_layer_id=lambda name: 0),
        ),
        patch.object(sfa_v1.AscendSFAImpl, "_execute_sparse_flash_attention_process", fake_standard),
    ):
        # 2D selection (rows, topk) with 3 real rows vs 8 padded query rows.
        topk_2d = torch.full((3, 6), 7, dtype=torch.int32)
        query = torch.zeros(8, 1, 128)
        result = impl._execute_sparse_flash_attention_process(query, query, (None, None), topk_2d, None, None, None)
        assert result is sentinel
        padded = forwarded[-1]
        assert padded.dim() == 3
        assert padded.shape == (8, 1, 6)
        assert torch.equal(padded[:3], torch.full((3, 1, 6), 7, dtype=torch.int32))
        assert torch.equal(padded[3:], torch.zeros(5, 1, 6, dtype=torch.int32))
        # The cached zero block is reused on the next padded forward.
        pad_ptr = impl._lim_topk_pad_zeros.data_ptr()
        topk_3d = torch.full((3, 1, 6), 9, dtype=torch.int32)
        impl._execute_sparse_flash_attention_process(query, query, (None, None), topk_3d, None, None, None)
        assert impl._lim_topk_pad_zeros.data_ptr() == pad_ptr
        assert forwarded[-1].shape == (8, 1, 6)
        assert torch.equal(forwarded[-1][:3], torch.full((3, 1, 6), 9, dtype=torch.int32))


def test_side_stream_attention_dispatch():
    """Side-stream members wait then use the zero-miss fused operator."""
    impl = _make_impl(skip_topk=True, is_mtp=False, shared_only=True)
    impl.layer_name = "model.layers.3.self_attn.attn"
    impl._current_layer_name = impl.layer_name
    impl.use_side_stream_onboard = True
    owner = _make_impl(skip_topk=False, is_mtp=False, shared_only=True)
    owner.layer_name = "model.layers.2.self_attn.attn"
    owner._lim_share_members = [impl]
    impl.lim_indexer_owner = owner

    waited = []
    launched = {"flag": False}
    manager = SimpleNamespace(
        wait_lim_group_onboard=lambda o, m: waited.append(True),
        _get_offload_layer_id=lambda name: 0,
        lim_group_onboard_ready=True,
        lim_group_launched_for=lambda o: launched["flag"],
    )
    metadata = SimpleNamespace(
        fused_copy_sfa_enabled=True,
    )
    query = torch.zeros(1, 8, 128)
    query_rope = torch.zeros(1, 8, 64)
    topk = torch.zeros(1, 1, 2048, dtype=torch.int32)

    calls = []
    impl._copy_sfa_attention = lambda *args, **kwargs: calls.append(kwargs) or torch.zeros(1, 8, 128)
    # The owner's LIM launched this pass's onboarding: resident path.
    launched["flag"] = True
    with (
        patch(
            "vllm_ascend.attention.sfa_kv_offload.get_sparse_kv_offload_manager",
            return_value=manager,
        ),
        patch(
            "vllm_ascend.attention.sfa_kv_offload.AscendSFAKVOffloadImpl._offload_layer_name",
            return_value=impl.layer_name,
        ),
    ):
        impl._execute_sparse_flash_attention_process(
            query,
            query_rope,
            (None, None),
            topk,
            metadata,
            None,
            None,
        )
    assert waited == [True]
    assert calls and calls[0].get("misses_already_resident") is True

    # Same member, but the owner never launched this pass's onboarding (e.g.
    # a draft reuse step): no wait, and the fused operator resolves the
    # owner's miss plan itself (inline fallback).
    calls.clear()
    waited.clear()
    launched["flag"] = False
    with (
        patch(
            "vllm_ascend.attention.sfa_kv_offload.get_sparse_kv_offload_manager",
            return_value=manager,
        ),
        patch(
            "vllm_ascend.attention.sfa_kv_offload.AscendSFAKVOffloadImpl._offload_layer_name",
            return_value=impl.layer_name,
        ),
    ):
        impl._execute_sparse_flash_attention_process(
            query,
            query_rope,
            (None, None),
            topk,
            metadata,
            None,
            None,
        )
    assert waited == []
    assert calls and calls[0].get("misses_already_resident") is not True

    # Non-side-stream members call the operator with real misses.
    calls.clear()
    waited.clear()
    launched["flag"] = True
    impl2 = _make_impl(skip_topk=True, is_mtp=False, shared_only=True)
    impl2.layer_name = "model.layers.3.self_attn.attn"
    impl2._current_layer_name = impl2.layer_name
    impl2.use_side_stream_onboard = False
    impl2.lim_indexer_owner = owner
    impl2._copy_sfa_attention = lambda *args, **kwargs: calls.append(kwargs) or torch.zeros(1, 8, 128)
    with (
        patch(
            "vllm_ascend.attention.sfa_kv_offload.get_sparse_kv_offload_manager",
            return_value=manager,
        ),
        patch(
            "vllm_ascend.attention.sfa_kv_offload.AscendSFAKVOffloadImpl._offload_layer_name",
            return_value=impl2.layer_name,
        ),
    ):
        impl2._execute_sparse_flash_attention_process(
            query,
            query_rope,
            (None, None),
            topk,
            metadata,
            None,
            None,
        )
    assert waited == []
    assert calls and calls[0].get("misses_already_resident") is not True


# ---------------------------------------------------------------------------
# V2 model runner: layer binding + registration wiring
# ---------------------------------------------------------------------------
def _make_v2_runner(sparse_cfg):
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner as V2ModelRunner

    runner = V2ModelRunner.__new__(V2ModelRunner)
    runner.ascend_config = SimpleNamespace(sparse_kv_offload_config=sparse_cfg)
    runner.model_config = SimpleNamespace(hf_config=SimpleNamespace(num_hidden_layers=3))
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    # Layers 0/2 are indexer owners (skip_topk=False); layer 1 shares layer
    # 0's selection (skip_topk=True, same topk buffer); layer 5 is the MTP
    # layer (index >= num_hidden_layers).
    shared_buffer = torch.zeros(4)
    other_buffer = torch.zeros(4)

    def _impl(name, skip_topk, buffer):
        impl = _make_impl(skip_topk=skip_topk, is_mtp=False, shared_only=sparse_cfg.offload_shared_indexer_only)
        impl.layer_name = None
        impl._current_layer_name = None
        impl.topk_indices_buffer = buffer
        return impl

    layers = {
        "model.layers.0.self_attn.attn": _impl("l0", False, shared_buffer),
        "model.layers.1.self_attn.attn": _impl("l1", True, shared_buffer),
        "model.layers.2.self_attn.attn": _impl("l2", False, other_buffer),
        "model.layers.5.self_attn.attn": _impl("l5", False, other_buffer),
    }
    runner.compilation_config.static_forward_context = {
        name: SimpleNamespace(impl=impl) for name, impl in layers.items()
    }
    return runner, layers


def test_v2_bind_sparse_offload_layer_groups():
    sparse_cfg = SimpleNamespace(
        enabled=True,
        use_fused_copy_sfa=True,
        offload_shared_indexer_only=True,
    )
    runner, layers = _make_v2_runner(sparse_cfg)
    runner._bind_sparse_offload_layer_groups()
    # Names bound from the forward context for every offload impl.
    assert all(layers[n].layer_name == n for n in layers)
    # The MTP layer (index 5 >= num_hidden_layers=3) is reclassified.
    assert layers["model.layers.5.self_attn.attn"]._is_mtp_layer is True
    assert layers["model.layers.5.self_attn.attn"].lim_full_resident is False
    # Owners with members are resident but keep running LIM for the group;
    # singleton owners are resident without members; shared members are not.
    owner0 = layers["model.layers.0.self_attn.attn"]
    assert owner0.lim_full_resident is True
    assert [m.layer_name for m in owner0._lim_share_members] == ["model.layers.1.self_attn.attn"]
    assert layers["model.layers.1.self_attn.attn"].lim_full_resident is False
    assert layers["model.layers.2.self_attn.attn"]._lim_share_members == []
    assert runner._lim_shared_only_groups == {"model.layers.0.self_attn.attn": ["model.layers.1.self_attn.attn"]}
    # The full-resident list feeds register_kv_caches: owners 0/2 only.
    assert sorted(runner._full_resident_offload_layer_names()) == [
        "model.layers.0.self_attn.attn",
        "model.layers.2.self_attn.attn",
    ]


def test_v2_bind_skips_when_fused_disabled():
    sparse_cfg = SimpleNamespace(
        enabled=True,
        use_fused_copy_sfa=False,
        offload_shared_indexer_only=False,
    )
    runner, layers = _make_v2_runner(sparse_cfg)
    runner._bind_sparse_offload_layer_groups()
    # No binding and no grouping without the fused pipeline.
    assert layers["model.layers.0.self_attn.attn"].layer_name is None
    assert runner._lim_shared_only_groups == {}


def test_v2_register_sparse_kv_caches_wires_full_resident_and_groups():
    sparse_cfg = SimpleNamespace(
        enabled=True,
        use_fused_copy_sfa=True,
        offload_shared_indexer_only=True,
    )
    runner, layers = _make_v2_runner(sparse_cfg)
    runner._bind_sparse_offload_layer_groups()
    calls = []
    manager = SimpleNamespace(
        offload_layer_names=["model.layers.1.self_attn.attn", "model.layers.5.self_attn.attn"],
        register_kv_caches=lambda kv, full_resident_layer_names=None: calls.append(
            ("register", full_resident_layer_names)
        ),
        configure_lim_groups=lambda groups: calls.append(("configure", groups)),
        _get_offload_layer_id=lambda name: 0,
    )
    runner.sparse_kv_offload_manager = manager
    runner._register_sparse_kv_caches({})
    assert calls[0][0] == "register"
    assert sorted(calls[0][1]) == ["model.layers.0.self_attn.attn", "model.layers.2.self_attn.attn"]
    assert calls[1] == (
        "configure",
        {"model.layers.0.self_attn.attn": ["model.layers.1.self_attn.attn"]},
    )
