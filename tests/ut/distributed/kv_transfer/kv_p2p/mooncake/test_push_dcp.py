# SPDX-License-Identifier: Apache-2.0
"""CPU-only DCP WRITE tests: real host buffers, mocked NPU fences."""

import ctypes
from dataclasses import replace
from unittest.mock import Mock

import msgspec
import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import push_worker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import MooncakeConnectorMetadata

from . import test_push_sending
from .test_push_sending import drive, multi_cache_remote, remote_metadata, target_params

worker = test_push_sending.worker


def configure(worker, monkeypatch, dcp=2, tp=2, interleave=4, local_scale=1, remote_scale=1, pcp=1):
    spec = Mock(spec=push_worker.MLAAttentionSpec, dcp_sharded=True, sliding_window=None, is_circular=False)
    worker.kv_cache_specs = [spec]
    local = worker.transfer_metadata
    local.layer_block_sizes[:] = [4 * local_scale]
    local.block_size_scales[:] = [[local_scale]]
    local.block_shapes[:] = [[(1, 4, 1)]]
    local.block_lens[:] = [[4]]
    local.block_strides[:] = [[4]]
    remote = replace(remote_metadata(tp_size=tp, heads=1, pcp_size=pcp), dcp_size=dcp)
    pp = replace(remote.metadata_by_pp_rank[0], cp_kv_cache_interleave_size=interleave)
    remote.metadata_by_pp_rank[0] = pp
    pp.layer_block_sizes[:] = [4 * remote_scale]
    pp.block_size_scales[:] = [[remote_scale]]
    pp.block_shapes[:] = [[(1, 4, 1)]]
    pp.block_lens[:] = [[4]]
    pp.block_strides[:] = [[4]]
    monkeypatch.setattr(worker, "_get_remote_metadata", lambda _: remote)
    return remote


def metadata(computed, end, prompt=29, prefix=0, remote_blocks=(10, 11, 12, 13), tp=2, pcp=1):
    target = target_params()
    target.update(
        remote_num_computed_tokens=prefix,
        remote_num_external_tokens=prompt - prefix,
        remote_block_ids=[list(remote_blocks)],
        remote_worker_addresses=[
            (0, pcp_rank, tp_rank, f"tcp://127.0.0.1:{9001 + pcp_rank * tp + tp_rank}")
            for pcp_rank in range(pcp)
            for tp_rank in range(tp)
        ],
    )
    meta = MooncakeConnectorMetadata()
    meta.add_new_push_send_req("p-request", (list(range(1, 10)),), prompt, computed, end - computed, target)
    return meta


@pytest.mark.parametrize("dcp,tp", [(2, 2), (2, 4), (4, 4)])
@pytest.mark.parametrize("interleave,local_scale,remote_scale", [(4, 1, 1), (4, 2, 4), (8, 4, 2)])
@pytest.mark.parametrize("prefix", [0, 8])
@pytest.mark.parametrize("cache_kind", ["mla", "sharded_indexer", "replicated_indexer"])
def test_chunked_write_matches_token_oracle(
    worker, monkeypatch, dcp, tp, interleave, local_scale, remote_scale, prefix, cache_kind
):
    remote = configure(worker, monkeypatch, dcp, tp, interleave, local_scale, remote_scale)
    effective_dcp = dcp
    if cache_kind != "mla":
        worker.kv_cache_specs = [
            Mock(
                spec=push_worker.AscendSFAIndexerCacheSpec,
                dcp_sharded=True,
                sliding_window=None,
                is_circular=False,
                sfa_dcp_replicated_indexer_size=1,
            )
        ]
        if cache_kind == "replicated_indexer":
            remote.metadata_by_pp_rank[0].indexer_replication_sizes[:] = [dcp]
            effective_dcp = 1
    source = (ctypes.c_ubyte * 256)(*([255] * 256))
    targets = [(ctypes.c_ubyte * 256)(*([255] * 256)) for _ in range(tp)]
    for token in range(32):
        page, offset = divmod(token, 4 * local_scale)
        source[(page + 1) * 4 * local_scale + offset] = token + 1
    worker.transfer_metadata.kv_caches_base_addr[:] = [[ctypes.addressof(source)]]
    pp = remote.metadata_by_pp_rank[0]
    for rank, target in enumerate(targets):
        pp.metadata_by_pcp_rank[0].metadata_by_tp_rank[rank].kv_caches_base_addr[:] = [[ctypes.addressof(target)]]
    copies = []

    def copy(session, src, dst, lengths):
        for source_addr, target_addr, size in zip(src, dst, lengths):
            copies.append((source_addr, target_addr, size))
            ctypes.memmove(target_addr, source_addr, size)
        return 0

    worker.engine.batch_transfer_sync_write.side_effect = copy
    previous = 0
    for end in [3, 9, 17, 29]:
        worker.start_load_kv(metadata(previous, end, prefix=prefix, tp=tp, remote_blocks=tuple(range(10, 18))))
        worker.save_kv_layer(worker.transfer_metadata.layer_names[0])
        worker.wait_for_save()
        ready = end // 4 * 4 if end < 29 else 32
        # Independent token-by-token oracle, including final block padding.
        for rank, target in enumerate(targets):
            expected = [255] * 256
            local_token = 0
            for token in range(32):
                if (token // interleave) % effective_dcp != rank % effective_dcp:
                    continue
                if prefix <= token < ready:
                    page, offset = divmod(local_token, 4 * remote_scale)
                    expected[(10 + page) * 4 * remote_scale + offset] = token + 1
                local_token += 1
            assert list(target) == expected
        if end < 29:
            worker._send_done.assert_not_called()
        previous = end
    assert worker._send_done.call_count == tp
    assert all(call.args[1] is None for call in worker._send_done.call_args_list)
    assert len(copies) == len(set(copies)), "A later chunk must not resend a completed block"


def test_empty_final_stripe_completes_without_write(worker, monkeypatch):
    configure(worker, monkeypatch, dcp=4, tp=4)
    worker.start_load_kv(metadata(0, 3, prompt=3, tp=4))
    worker.wait_for_save()
    assert worker.engine.batch_transfer_sync_write.call_count == 1
    assert worker._send_done.call_count == 4
    assert all(call.args[1] is None for call in worker._send_done.call_args_list)


@pytest.mark.parametrize("p_tp,d_tp,dcp", [(4, 2, 2), (2, 4, 2), (4, 4, 4)])
def test_one_producer_replica_per_destination(worker, monkeypatch, p_tp, d_tp, dcp):
    remote = configure(worker, monkeypatch, dcp=dcp, tp=d_tp)
    worker.tp_size = p_tp
    for d_rank in range(d_tp):
        owners = []
        for p_rank in range(p_tp):
            worker.tp_rank = p_rank
            if worker._get_transfer_partition("model.layers.0.attn", remote, (0, 0, d_rank)) is not None:
                owners.append(p_rank)
        assert owners == [d_rank % p_tp]


@pytest.mark.parametrize("interleave", [0, 2, 6, 8])
def test_invalid_interleave_fails_request(worker, monkeypatch, interleave):
    configure(worker, monkeypatch, interleave=interleave)
    worker.start_load_kv(metadata(0, 8, prompt=8))
    worker.wait_for_save()
    worker.engine.batch_transfer_sync_write.assert_not_called()
    assert worker._send_done.call_count == 2
    assert all("interleave" in call.args[1] for call in worker._send_done.call_args_list)


def test_short_destination_table_fails_without_writing(worker, monkeypatch):
    configure(worker, monkeypatch)
    worker.start_load_kv(metadata(0, 29, remote_blocks=()))
    worker.wait_for_save()
    worker.engine.batch_transfer_sync_write.assert_not_called()
    assert all("block tables" in call.args[1] for call in worker._send_done.call_args_list)


@pytest.mark.parametrize("p_tp,d_tp", [(3, 2), (2, 3)])
def test_mamba_dcp_rejects_noninteger_tp_ratio_before_write(worker, monkeypatch, p_tp, d_tp):
    remote = configure(worker, monkeypatch, dcp=d_tp, tp=d_tp)
    remote.metadata_by_pp_rank[0].dcp_sharded[:] = [False]
    worker.tp_size = worker.dcp_size = p_tp
    worker.kv_cache_specs = [Mock(spec=push_worker.MambaSpec, dcp_sharded=False, tp_replicated=False)]
    worker.start_load_kv(metadata(0, 8, prompt=8, tp=d_tp))
    worker.wait_for_save()
    worker.engine.batch_transfer_sync_write.assert_not_called()
    assert worker._send_done.call_count == d_tp
    assert all("integer ratio" in call.args[1] for call in worker._send_done.call_args_list)


@pytest.mark.parametrize("kind", ["swa", "circular", "replicated_indexer", "sharded_indexer"])
def test_effective_dcp_depends_on_cache_layout(worker, monkeypatch, kind):
    remote = configure(worker, monkeypatch)
    spec_type = {
        "swa": push_worker.SlidingWindowMLASpec,
        "circular": push_worker.MLAAttentionSpec,
        "replicated_indexer": push_worker.AscendSFAIndexerCacheSpec,
        "sharded_indexer": push_worker.AscendSFAIndexerCacheSpec,
    }[kind]
    worker.kv_cache_specs = [
        Mock(
            spec=spec_type, dcp_sharded=kind != "swa", is_circular=kind == "circular", sfa_dcp_replicated_indexer_size=1
        )
    ]
    remote.metadata_by_pp_rank[0].dcp_sharded[:] = [kind != "swa"]
    remote.metadata_by_pp_rank[0].indexer_replication_sizes[:] = [2 if kind == "replicated_indexer" else 1]
    assert worker._get_layer_dcp_sizes("model.layers.0.attn", remote, (0, 0, 0)) == (
        1,
        2 if kind == "sharded_indexer" else 1,
    )


@pytest.mark.parametrize("p_sharded,d_sharded", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("p_dcp,d_dcp", [(1, 4), (2, 4), (4, 2)])
def test_dcp_sharding_flags_are_independent(worker, monkeypatch, p_sharded, d_sharded, p_dcp, d_dcp):
    remote = configure(worker, monkeypatch, dcp=d_dcp, tp=4)
    worker.dcp_size = p_dcp
    worker.kv_cache_specs[0].dcp_sharded = p_sharded
    remote.metadata_by_pp_rank[0].dcp_sharded[:] = [d_sharded]
    assert worker._get_layer_dcp_sizes("model.layers.0.attn", remote, (0, 0, 0)) == (
        p_dcp if p_sharded else 1,
        d_dcp if d_sharded else 1,
    )


@pytest.mark.parametrize("worker", ["dspark"], indirect=True)
def test_dcp_flags_follow_layer_names_across_peers(worker):
    worker.dcp_size = 2
    sharded, _, unsharded = worker.transfer_metadata.layer_names
    worker.kv_cache_specs = [
        Mock(spec=push_worker.MLAAttentionSpec, dcp_sharded=flag, is_circular=False) for flag in (True, False)
    ]
    worker.layer_name_to_spec_index[unsharded] = 1
    remote = replace(multi_cache_remote(worker), dcp_size=4)
    pp = remote.metadata_by_pp_rank[0]
    # Remote layer order and layout need not match the local spec order.
    remote.metadata_by_pp_rank[0] = replace(
        pp, layer_names=[unsharded, pp.layer_names[1], sharded], dcp_sharded=[True, True, False]
    )
    assert worker._get_layer_dcp_sizes(sharded, remote, (0, 0, 0)) == (2, 1)
    assert worker._get_layer_dcp_sizes(unsharded, remote, (0, 0, 0)) == (1, 4)


@pytest.mark.parametrize("p_sharded,d_sharded", [(False, False), (False, True), (True, False), (True, True)])
def test_full_attention_with_dcp_requires_unsharded_layout(worker, monkeypatch, p_sharded, d_sharded):
    remote = configure(worker, monkeypatch, dcp=4, tp=4)
    worker.tp_size = 4
    worker.dcp_size = 2
    worker.kv_cache_specs = [
        Mock(spec=push_worker.FullAttentionSpec, dcp_sharded=p_sharded, sliding_window=None, is_circular=False)
    ]
    pp = remote.metadata_by_pp_rank[0]
    pp.dcp_sharded[:] = [d_sharded]
    worker.transfer_metadata.block_shapes[:] = [[(2, 4, 1)]]
    pp.block_shapes[:] = [[(2, 4, 1)]]
    if p_sharded or d_sharded:
        with pytest.raises(ValueError, match="Push DCP does not support"):
            worker._get_transfer_partition("model.layers.0.attn", remote, (0, 0, 0))
    else:
        for p_rank in range(4):
            worker.tp_rank = p_rank
            for d_rank in range(4):
                assert worker._get_transfer_partition("model.layers.0.attn", remote, (0, 0, d_rank)) == (
                    (0, 0, 2) if p_rank == d_rank else None
                )


def test_dcp_keeps_layer_event_snapshot(worker, monkeypatch):
    configure(worker, monkeypatch)
    worker.start_load_kv(metadata(0, 4, prompt=8))
    state = worker._reqs_send_state["d-request"]
    state.destinations[(0, 0, 0)].metadata_future.result(timeout=3)
    # Capture submission without launching a WRITE until the event is checked.
    monkeypatch.setattr(worker, "_process_sending_requests", Mock())
    worker.save_kv_layer("model.layers.0.attn")
    destination = state.destinations[(0, 0, 0)]
    request, event = worker._build_layer_transfer_request("d-request", state, destination, "model.layers.0.attn")
    state.metadata = replace(state.metadata, num_computed_tokens=4, num_scheduled_tokens=4)
    worker.save_kv_layer("model.layers.0.attn")
    assert event is not state.available_layers["model.layers.0.attn"][1]
    assert request.end_token == 4
    assert request.local_block_ids == (1,)
    event.synchronize()


@pytest.mark.parametrize("worker", ["dspark"], indirect=True)
@pytest.mark.parametrize("p_dcp", [1, 2, 4])
def test_dcp_draft_waits_for_its_own_hook(worker, monkeypatch, p_dcp):
    worker.dcp_size = worker.tp_size = p_dcp
    worker.kv_cache_specs = [
        Mock(spec=push_worker.MLAAttentionSpec, dcp_sharded=True, sliding_window=None, is_circular=False)
    ]
    remote = replace(multi_cache_remote(worker), dcp_size=2, tp_size=2)
    pp = remote.metadata_by_pp_rank[0]
    pp.metadata_by_pcp_rank[0].metadata_by_tp_rank[1] = replace(pp.metadata_by_pcp_rank[0].metadata_by_tp_rank[0])
    monkeypatch.setattr(worker, "_get_remote_metadata", lambda _: remote)
    meta = metadata(0, 8, prompt=8)
    meta.push_send_requests["p-request"] = replace(
        meta.push_send_requests["p-request"], local_block_ids=((1, 2), (3, 4)), remote_block_ids=((10, 11), (12, 13))
    )
    worker.start_load_kv(meta)
    attention, indexer, draft = worker.transfer_metadata.layer_names
    state = worker._reqs_send_state["d-request"]
    worker.save_kv_layer(attention)
    drive(worker, lambda: all(len(dest.sent_tokens) == 2 for dest in state.destinations.values()))
    assert draft not in state.available_layers
    worker._send_done.assert_not_called()
    worker.save_kv_layer(draft)
    worker.wait_for_save()
    assert worker._send_done.call_count == 2
    assert all(call.args[1] is None for call in worker._send_done.call_args_list)


@pytest.mark.parametrize(
    "dcp,tp,pcp",
    [
        (1, 1, 1),
        (2, 4, 1),
        (4, 4, 1),
        (1, 4, 2),
        (2, 4, 2),
        (8, 4, 2),
    ],
)
@pytest.mark.parametrize("p_dcp,p_tp,p_pcp", [(1, 1, 1), (2, 2, 1), (1, 4, 2), (2, 4, 2), (8, 4, 2)])
def test_metadata_handshake_accepts_pcp_dcp(worker, monkeypatch, dcp, tp, pcp, p_dcp, p_tp, p_pcp):
    remote = replace(remote_metadata(tp_size=tp, pcp_size=pcp), dcp_size=dcp)
    worker.pcp_size = p_pcp
    worker.dcp_size, worker.tp_size = p_dcp, p_tp
    monkeypatch.setattr(push_worker, "make_zmq_socket", Mock(return_value=Mock()))
    monkeypatch.setattr(push_worker, "ensure_zmq_send", Mock())
    monkeypatch.setattr(push_worker, "ensure_zmq_recv", Mock(return_value=msgspec.msgpack.encode(remote)))
    params = dict(scheduler_address="tcp://127.0.0.1:9000", consumer_engine_id="d")
    assert push_worker.MooncakePushConnectorWorker._get_remote_metadata(worker, params) == remote
