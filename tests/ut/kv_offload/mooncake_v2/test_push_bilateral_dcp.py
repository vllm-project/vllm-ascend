# SPDX-License-Identifier: Apache-2.0
"""Multiple P shards write real CPU buffers for every D rank, without an NPU."""

import ctypes
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import base_worker, push_worker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import MooncakeConnectorMetadata

from . import test_push_sending
from .test_push_dcp import configure, metadata

worker = test_push_sending.worker


@pytest.fixture
def producers(worker, monkeypatch):
    created = []

    def make(dcp, tp, remote, interleave, scale, cache_kind, pcp=1):
        for physical_rank in range(tp * pcp):
            pcp_rank, tp_rank = divmod(physical_rank, tp)
            config = SimpleNamespace(
                parallel_config=SimpleNamespace(
                    decode_context_parallel_size=dcp,
                    tensor_parallel_size=tp,
                    prefill_context_parallel_size=pcp,
                )
            )
            obj = push_worker.MooncakePushConnectorWorker(config, "p", None)
            created.append(obj)
            obj.register_kv_caches({})
            obj.dcp_size, obj.tp_size, obj.tp_rank = dcp, tp, tp_rank
            obj.pcp_size, obj.pcp_rank = pcp, pcp_rank
            obj.dcp_rank = (tp_rank * pcp + pcp_rank) % dcp
            obj.transfer_metadata = replace(
                worker.transfer_metadata,
                cp_kv_cache_interleave_size=interleave,
                layer_block_sizes=[4 * scale],
                block_size_scales=[[scale]],
                kv_caches_base_addr=[[0]],
            )
            if cache_kind in ("mla", "p_unsharded", "d_unsharded", "both_unsharded"):
                spec = Mock(
                    spec=push_worker.MLAAttentionSpec,
                    dcp_sharded=cache_kind not in ("p_unsharded", "both_unsharded"),
                    sliding_window=None,
                    is_circular=False,
                )
            else:
                spec = Mock(
                    spec=push_worker.AscendSFAIndexerCacheSpec,
                    dcp_sharded=True,
                    sliding_window=None,
                    is_circular=False,
                    sfa_dcp_replicated_indexer_size=dcp if cache_kind in ("p_replicated", "both_replicated") else 1,
                )
            obj.kv_cache_specs = [spec]
            monkeypatch.setattr(obj, "_get_remote_metadata", lambda _: remote)
            monkeypatch.setattr(obj, "_send_done", Mock(return_value=True))
        return created

    yield make
    for obj in created:
        obj.shutdown()


@pytest.mark.parametrize(
    "p_dcp,d_dcp,p_tp,d_tp,p_pcp,d_pcp",
    [
        (1, 4, 2, 8, 1, 1),
        (4, 1, 8, 2, 1, 1),
        (2, 4, 4, 8, 1, 1),
        (4, 2, 8, 4, 1, 1),
        (4, 4, 8, 4, 1, 1),
        (8, 8, 8, 8, 1, 1),
        (2, 1, 4, 2, 2, 1),
        (8, 1, 4, 2, 2, 1),
        (1, 2, 2, 4, 1, 2),
        (1, 8, 2, 4, 1, 2),
        (2, 8, 4, 4, 2, 2),
        (8, 2, 4, 4, 2, 2),
        (1, 8, 4, 4, 2, 2),
        (4, 8, 2, 4, 4, 2),
    ],
)
@pytest.mark.parametrize(
    "p_interleave,d_interleave,p_scale,d_scale",
    [
        (4, 4, 1, 2),
        (4, 8, 2, 4),
        (8, 4, 4, 1),
        (8, 8, 2, 2),
    ],
)
@pytest.mark.parametrize("prefix", [0, 5, 16])
@pytest.mark.parametrize(
    "cache_kind",
    ["mla", "p_replicated", "d_replicated", "both_replicated", "p_unsharded", "d_unsharded", "both_unsharded"],
)
def test_bilateral_chunks_copy_each_token_once(
    worker,
    monkeypatch,
    producers,
    p_dcp,
    d_dcp,
    p_tp,
    d_tp,
    p_pcp,
    d_pcp,
    p_interleave,
    d_interleave,
    p_scale,
    d_scale,
    prefix,
    cache_kind,
):
    remote = configure(worker, monkeypatch, d_dcp, d_tp, d_interleave, p_scale, d_scale, pcp=d_pcp)
    pp = remote.metadata_by_pp_rank[0]
    p_effective = 1 if cache_kind in ("p_replicated", "both_replicated", "p_unsharded", "both_unsharded") else p_dcp
    d_effective = 1 if cache_kind in ("d_replicated", "both_replicated", "d_unsharded", "both_unsharded") else d_dcp
    pp.dcp_sharded[:] = [cache_kind not in ("d_unsharded", "both_unsharded")]
    if cache_kind in ("d_replicated", "both_replicated"):
        pp.indexer_replication_sizes[:] = [d_dcp]
    senders = producers(p_dcp, p_tp, remote, p_interleave, p_scale, cache_kind, pcp=p_pcp)
    # Physical workers are PCP-major; DCP groups enumerate PCP first within TP.
    p_order = [pcp * p_tp + tp for tp in range(p_tp) for pcp in range(p_pcp)]
    d_order = [pcp * d_tp + tp for tp in range(d_tp) for pcp in range(d_pcp)]
    source_blocks = tuple(1 + 2 * i for i in range(16))
    target_blocks = tuple(10 + 3 * i for i in range(16))
    sources = [(ctypes.c_ubyte * 1024)(*([254] * 1024)) for _ in senders]
    targets = [(ctypes.c_ubyte * 1024)(*([254] * 1024)) for _ in range(d_tp * d_pcp)]
    for rank, (sender, source) in enumerate(zip(senders, sources)):
        owned = [
            token for token in range(32) if (token // p_interleave) % p_effective == p_order.index(rank) % p_effective
        ]
        for position, token in enumerate(owned):
            page, offset = divmod(position, 4 * p_scale)
            source[source_blocks[page] * 4 * p_scale + offset] = token + 1
        sender.transfer_metadata.kv_caches_base_addr[:] = [[ctypes.addressof(source)]]
    for rank, target in enumerate(targets):
        pcp_rank, tp_rank = divmod(rank, d_tp)
        pp.metadata_by_pcp_rank[pcp_rank].metadata_by_tp_rank[tp_rank].kv_caches_base_addr[:] = [
            [ctypes.addressof(target)]
        ]
    writes = []

    def copy(session, src, dst, lengths):
        for source_addr, target_addr, size in zip(src, dst, lengths):
            assert any(
                ctypes.addressof(buf) <= source_addr < source_addr + size <= ctypes.addressof(buf) + 1024
                for buf in sources
            )
            assert any(
                ctypes.addressof(buf) <= target_addr < target_addr + size <= ctypes.addressof(buf) + 1024
                for buf in targets
            )
            writes.extend(range(target_addr, target_addr + size))
            ctypes.memmove(target_addr, source_addr, size)
        return 0

    for sender in senders:
        sender.engine.batch_transfer_sync_write.side_effect = copy
    previous = 0
    for end in (3, 9, 17, 29):
        for sender in senders:
            meta = metadata(previous, end, prefix=prefix, remote_blocks=target_blocks, tp=d_tp, pcp=d_pcp)
            meta.push_send_requests["p-request"] = replace(
                meta.push_send_requests["p-request"], local_block_ids=(source_blocks,)
            )
            sender.start_load_kv(meta)
            sender.save_kv_layer("model.layers.0.attn")
            sender.wait_for_save()
            assert all(call.args[1] is None for call in sender._send_done.call_args_list)
            if end < 29:
                for destination in sender._reqs_send_state["d-request"].destinations.values():
                    if destination.required_layers:
                        assert destination.completion is None
        ready = end // 4 * 4 if end < 29 else 32
        for rank, target in enumerate(targets):
            expected = [254] * 1024
            owned = [
                token
                for token in range(32)
                if (token // d_interleave) % d_effective == d_order.index(rank) % d_effective
            ]
            for position, token in enumerate(owned):
                if prefix // 4 * 4 <= token < ready:
                    page, offset = divmod(position, 4 * d_scale)
                    expected[target_blocks[page] * 4 * d_scale + offset] = token + 1
            assert list(target) == expected
        previous = end
    assert len(writes) == len(set(writes)), "Replicas/chunks must not overwrite the same D byte twice"
    assert all(sender._send_done.call_count == d_tp * d_pcp for sender in senders)


@pytest.mark.parametrize(
    "p_tp,d_tp,p_dcp,d_dcp,p_pcp,d_pcp",
    [
        (2, 2, 1, 1, 1, 1),
        (4, 2, 4, 1, 1, 1),
        (2, 4, 1, 4, 1, 1),
        (4, 4, 4, 2, 1, 1),
        (4, 4, 2, 4, 1, 1),
        (4, 2, 2, 2, 1, 1),
        (2, 4, 2, 2, 1, 1),
        (2, 4, 2, 8, 2, 2),
        (4, 2, 8, 2, 2, 2),
        (4, 4, 2, 8, 2, 2),
    ],
)
@pytest.mark.parametrize("mamba_type", ["MAMBA1", "MAMBA2", "GDN_ATTN"])
@pytest.mark.parametrize("replicated", [False, True])
def test_mamba_dcp_preserves_tp_state_bytes(
    worker, monkeypatch, producers, p_tp, d_tp, p_dcp, d_dcp, p_pcp, d_pcp, mamba_type, replicated
):
    """DCP must neither drop TP state shards nor apply attention token stripes."""
    backends = SimpleNamespace(MAMBA1="MAMBA1", MAMBA2="MAMBA2", GDN_ATTN="GDN_ATTN")
    monkeypatch.setattr(base_worker, "MambaAttentionBackendEnum", backends)
    monkeypatch.setattr(
        base_worker, "torch", SimpleNamespace(tensor=lambda *args, **kwargs: SimpleNamespace(element_size=lambda: 1))
    )
    widths = {"MAMBA1": (16,), "MAMBA2": (16, 8, 8), "GDN_ATTN": (8, 8, 16)}[mamba_type]

    def shard(tp, rank):
        # Build each TP shard from global projection/state bytes independently
        # of the connector's address slicing, including distinct conv rows.
        if replicated:
            tp, rank = 1, 0
        conv = []
        for row in range(2):
            offset = row * sum(widths)
            for width in widths:
                conv.extend(range(offset + rank * width // tp, offset + (rank + 1) * width // tp))
                offset += width
        state = list(range(128 + rank * 32 // tp, 128 + (rank + 1) * 32 // tp))
        return (bytes(conv), bytes(state)), [(2, sum(widths) // tp), (8 // tp, 2, 2)]

    remote = configure(worker, monkeypatch, dcp=d_dcp, tp=d_tp, pcp=d_pcp)
    pp = remote.metadata_by_pp_rank[0]
    pp.dcp_sharded[:] = [False]
    target_values, pp.block_shapes[0] = shard(d_tp * d_pcp, 0)
    pp.block_lens[0] = list(map(len, target_values))
    pp.block_strides[0] = [length + 7 for length in pp.block_lens[0]]
    pp.block_size_scales[0] = [1, 1]
    targets = [
        [ctypes.create_string_buffer(b"\xfe" * (12 * stride)) for stride in pp.block_strides[0]]
        for _ in range(d_tp * d_pcp)
    ]
    for rank, buffers in enumerate(targets):
        pcp_rank, tp_rank = divmod(rank, d_tp)
        pp.metadata_by_pcp_rank[pcp_rank].metadata_by_tp_rank[tp_rank].kv_caches_base_addr[0] = list(
            map(ctypes.addressof, buffers)
        )
    senders = producers(p_dcp, p_tp, remote, 1, 1, "mla", pcp=p_pcp)
    sources, writes = [], []

    def copy(session, src, dst, lengths):
        for source, target, length in zip(src, dst, lengths, strict=True):
            writes.extend(range(target, target + length))
            ctypes.memmove(target, source, length)
        return 0

    for sender in senders:
        values, shapes = shard(p_tp * p_pcp, sender.tp_rank * p_pcp + sender.pcp_rank)
        strides = [len(value) + 5 for value in values]
        buffers = [ctypes.create_string_buffer(b"\xfe" * (3 * stride)) for stride in strides]
        sources.append(buffers)
        for buf, value, stride in zip(buffers, values, strides, strict=True):
            ctypes.memmove(ctypes.addressof(buf) + 2 * stride, value, len(value))
        sender.transfer_metadata = replace(
            sender.transfer_metadata,
            kv_caches_base_addr=[list(map(ctypes.addressof, buffers))],
            block_shapes=[shapes],
            block_lens=[list(map(len, values))],
            block_strides=[strides],
            block_size_scales=[[1, 1]],
        )
        for field in ("layer_names", "kv_caches_base_addr", "block_shapes", "block_lens", "block_strides"):
            setattr(sender, field, getattr(sender.transfer_metadata, field))
        sender.kv_cache_specs = [
            Mock(
                spec=push_worker.MambaSpec,
                dcp_sharded=False,
                tp_replicated=replicated,
                mamba_type=mamba_type,
                dtypes=("byte", "byte"),
                sliding_window=None,
                is_circular=False,
            )
        ]
        sender.engine.batch_transfer_sync_write.side_effect = copy

    for computed, end in ((0, 4), (4, 8)):
        for sender in senders:
            meta = metadata(computed, end, prompt=8, tp=d_tp, pcp=d_pcp, remote_blocks=(10, 11))
            meta.push_send_requests["p-request"] = replace(
                meta.push_send_requests["p-request"], local_block_ids=((1, 2),)
            )
            sender.start_load_kv(meta)
            sender.wait_for_save()
            if end < 8:
                sender.engine.batch_transfer_sync_write.assert_not_called()
                # Redundant replicas may finish early; required state shards may not.
                assert all(
                    dest.completion is None
                    for dest in sender._reqs_send_state["d-request"].destinations.values()
                    if dest.required_layers
                )
        if end < 8:
            assert not writes
    for rank, buffers in enumerate(targets):
        pcp_rank, tp_rank = divmod(rank, d_tp)
        values, _ = shard(d_tp * d_pcp, tp_rank * d_pcp + pcp_rank)
        for buf, value, stride in zip(buffers, values, pp.block_strides[0], strict=True):
            expected = bytearray(b"\xfe" * (12 * stride) + b"\0")
            expected[11 * stride : 11 * stride + len(value)] = value
            assert buf.raw == expected
    assert len(writes) == len(set(writes)), "Each destination state byte must have exactly one TP owner"
    assert all(sender._send_done.call_count == d_tp * d_pcp for sender in senders)
    assert all(call.args[1] is None for sender in senders for call in sender._send_done.call_args_list)


@pytest.mark.parametrize("p_dcp,d_dcp", [(2, 4), (4, 2), (4, 1)])
def test_invalid_producer_interleave_is_rejected(worker, monkeypatch, p_dcp, d_dcp):
    configure(worker, monkeypatch, dcp=d_dcp, tp=d_dcp)
    worker.tp_size = worker.dcp_size = p_dcp
    worker.transfer_metadata = replace(worker.transfer_metadata, cp_kv_cache_interleave_size=2)
    worker.start_load_kv(metadata(0, 8, prompt=8, tp=d_dcp))
    worker.wait_for_save()
    worker.engine.batch_transfer_sync_write.assert_not_called()
    assert worker._send_done.call_count == d_dcp
    assert all("P page size" in call.args[1] for call in worker._send_done.call_args_list)


@pytest.mark.parametrize("write_fails", [False, True])
@pytest.mark.parametrize("p_pcp", [1, 2])
def test_receiver_waits_for_last_source_shard(worker, monkeypatch, producers, write_fails, p_pcp):
    remote = configure(worker, monkeypatch, dcp=1, tp=1)
    senders = producers(2, 4, remote, 4, 1, "mla", pcp=p_pcp)
    delayed_rank = 1 if p_pcp == 1 else 4
    worker.kv_transfer_config = SimpleNamespace(is_kv_producer=False)
    recv_meta = MooncakeConnectorMetadata()
    recv_meta.add_new_push_req("d-request", ([10, 11],), 8, {"num_computed_tokens": 0})
    worker.start_load_kv(recv_meta)
    entered, release = threading.Event(), threading.Event()

    def delayed_write(*args):
        entered.set()
        assert release.wait(timeout=5)
        return -1 if write_fails else 0

    def done(params, error):
        payload = dict(params, producer_sizes=[1, p_pcp, 4], status="error" if error else "ok")
        worker._result_queue.put(("done", "d-request", payload))
        return True

    senders[delayed_rank].engine.batch_transfer_sync_write.side_effect = delayed_write
    for sender in senders:
        monkeypatch.setattr(sender, "_send_done", Mock(side_effect=done))
    try:
        for sender in senders:
            sender.start_load_kv(metadata(0, 8, prompt=8, tp=1))
            sender._reqs_send_state["d-request"].destinations[(0, 0, 0)].metadata_future.result(timeout=3)
            sender.save_kv_layer("model.layers.0.attn")
        assert entered.wait(timeout=3)
        for rank, sender in enumerate(senders):
            if rank != delayed_rank:
                sender.wait_for_save()
        # Non-owner replicas send DONE too, but cannot substitute for shard 1.
        senders[delayed_rank]._send_done.assert_not_called()
        assert not worker.get_transfer_results().finished_recving
        assert worker._reqs_recv_state["d-request"].finished_ranks == {
            (0, sender.pcp_rank, sender.tp_rank) for rank, sender in enumerate(senders) if rank != delayed_rank
        }
    finally:
        release.set()
    senders[delayed_rank].wait_for_save()
    result = worker.get_transfer_results()
    assert result.finished_recving == {"d-request"}
    assert result.failed_recving == ({"d-request"} if write_fails else set())
