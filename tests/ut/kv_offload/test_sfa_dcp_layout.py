# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import msgspec
import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dcp_layout import (
    dcp_block_segments,
    validate_dcp_layout_metadata,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.protocol import (
    MF_META_ACK,
    LayerMetadata,
    SendTask,
    parse_read_request,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.read_thread import (
    MembPullReadThread,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.send_thread import (
    MembPullSendingThread,
    ProducerSendState,
)


@pytest.mark.parametrize("tokens", [1, 11, 127, 128, 129, 1023, 1024, 1025, 8192, 8257])
@pytest.mark.parametrize("cp", [2, 4, 8])
def test_main_shards_reconstruct_exact_global_sequence(tokens, cp):
    block = 128
    source = list(range(17, 17 + (tokens + cp * block - 1) // (cp * block)))
    dest = list(reversed(range(100, 100 + (tokens + block - 1) // block)))
    seen = {}
    for rank in range(cp):
        for sb, so, db, do, count in dcp_block_segments(
            source,
            dest,
            start_block=0,
            token_start=0,
            token_end=tokens,
            block_size=block,
            dcp_size=cp,
            owner_rank=rank,
        ):
            for offset in range(count):
                global_token = dest.index(db) * block + do + offset
                value = (source.index(sb) * cp + rank) * block + so + offset
                assert global_token not in seen
                seen[global_token] = value
    assert seen == dict(enumerate(range(tokens)))


@pytest.mark.parametrize("start,end", [(0, 11), (0, 8192), (8192, 8257), (128, 1153), (1536, 2733)])
def test_indexer_replicated_view_and_partial_chunks(start, end):
    cp, block = 8, 128
    all_src = list(range(29, 29 + (end + 1023) // 1024))
    first = start // 1024
    src = all_src[first:]
    dst = list(range(90, 90 + (end + 127) // 128))
    # Each D TP receives the entire indexer from two P contributors.
    seen = {}
    for member in range(2):
        for sb, so, db, do, count in dcp_block_segments(
            src,
            dst,
            start_block=first,
            token_start=start,
            token_end=end,
            block_size=block,
            dcp_size=cp,
            contributor=member,
            contributors=2,
        ):
            for i in range(count):
                target = dst.index(db) * block + do + i
                source = (all_src.index(sb // cp) * cp + sb % cp) * block + so + i
                assert target not in seen
                seen[target] = source
    assert seen == {i: i for i in range(start, end)}


def test_reject_missing_source_and_destination():
    kwargs = dict(start_block=0, token_start=0, token_end=1025, block_size=128, dcp_size=8)
    with pytest.raises(ValueError, match="destination"):
        list(dcp_block_segments([1, 2], [3], **kwargs))
    with pytest.raises(ValueError, match="source"):
        list(dcp_block_segments([1], list(range(9)), **kwargs))


@pytest.mark.parametrize("prefix_tokens", [0, 7])
@pytest.mark.parametrize("d_tp", [1, 2, 4, 8])
@pytest.mark.parametrize("with_indexer", [False, True])
def test_receiver_reconstructs_main_and_every_rank_indexer(with_indexer, d_tp, prefix_tokens):
    class Tensor:
        shape = (100, 4, 1)

        def __init__(self, ptr):
            self.ptr = ptr

        def data_ptr(self):
            return self.ptr

        def element_size(self):
            return 1

    name = "model.layers.0.self_attn.attn"
    block, cp, end = 4, 8, 71
    pids, dids = [9, 3, 15], list(range(30, 48))
    expected = {}
    source = {}
    actual = {}
    # Encode distinct per-token values independently of the descriptor helper.
    for rank in range(cp):
        base = 100000 + rank * 10000
        for token in range(end):
            logical, off = divmod(token, block)
            superblock = logical // cp
            if logical % cp == rank:
                for component, rowsize in [(0, 2), (1, 1)]:
                    for byte in range(rowsize):
                        addr = base + component * 1000 + pids[superblock] * block * rowsize + off * rowsize + byte
                        source[addr] = (component, token, byte)
            if with_indexer:
                for component in [2, 3]:
                    addr = base + component * 1000 + (pids[superblock] * cp + logical % cp) * block + off
                    source[addr] = (component, token, 0)
    for token in range(prefix_tokens, end):
        db, off = dids[token // block], token % block
        for component, rowsize in [(0, 2), (1, 1)]:
            for byte in range(rowsize):
                expected[component * 1000 + db * block * rowsize + off * rowsize + byte] = (component, token, byte)
        if with_indexer:
            for d_rank in range(d_tp):
                for component in [2, 3]:
                    expected[10000 + d_rank * 10000 + component * 1000 + db * block + off] = (component, token, 0)
    for start, stop in [(prefix_tokens, 19), (19, 64), (64, end)]:
        for rank in range(cp):
            ratio = cp // d_tp
            d_rank = rank // ratio
            base = 100000 + rank * 10000
            state = SimpleNamespace(
                tp_size=d_tp,
                main_block_size=block,
                indexer_block_size=block,
                main_name_to_idx={name: 0},
                get_offload_layer_id=lambda _: 0,
                main_gva_bases=[(0, 1000)],
                main_block_lens=[(8, 4)],
                num_blocks=100,
                indexer_tensors=[Tensor(12000 + d_rank * 10000) if with_indexer else None],
                indexer_scale_tensors=[Tensor(13000 + d_rank * 10000) if with_indexer else None],
            )
            thread = MembPullReadThread.__new__(MembPullReadThread)
            thread._state, thread.tp_rank = state, d_rank
            thread._wait_for_dest_blocks = lambda *_: (dids, dids)

            def copy(session, local, peer, sizes):
                for dst, src, length in zip(local, peer, sizes):
                    for byte in range(length):
                        assert dst + byte not in actual
                        actual[dst + byte] = source[src + byte]
                return 0

            thread.engine = SimpleNamespace(batch_transfer_sync_read=copy)
            meta = dict(
                dcp=dict(size=cp, rank=rank, block_size=block, indexer_block_size=block),
                base_addrs=[base, base + 1000] + ([base + 2000, base + 3000] if with_indexer else []),
                block_len=[8, 4] + ([4, 4] if with_indexer else []),
                block_size_scale=[1, 1] + ([cp, cp] if with_indexer else []),
                main_tensor_count=2,
                has_indexer=with_indexer,
            )
            first, last = start // (block * cp), (stop + block * cp - 1) // (block * cp)
            entry = (
                "req",
                pids[first:last],
                pids[first:last],
                first,
                first,
                start,
                stop,
            )
            thread._do_read_dcp_batch(name, [entry], "session", meta, rank % ratio, ratio)
    assert actual == expected


@pytest.mark.parametrize("finished,computed", [(False, 8257), (True, 8257), (True, None), (True, 0)])
def test_sender_uses_virtual_block_ranges_and_exact_tokens(finished, computed):
    name = "model.layers.0.self_attn.attn"
    thread = MembPullSendingThread.__new__(MembPullSendingThread)
    thread._state = ProducerSendState(
        last_layer_idx=0,
        layer_metadata={name: LayerMetadata([], [], [], [], has_indexer=True)},
        p_session="P",
        main_group_idx=0,
        indexer_group_idx=1,
        block_sizes=(128, 128),
        layer_storage_slots={},
        producer_dcp_size=8,
        producer_dcp_rank=3,
    )
    thread.last_layer_idx = 0
    thread._p_save_events = {}
    thread._pending_reads_by_layer = {}
    thread.mark_layer_pending = MagicMock()
    thread._mf_meta_sent_paths = set()
    thread._send_mf_meta = MagicMock()
    dealer = MagicMock()
    thread._ensure_peer_dealer = MagicMock(return_value=dealer)
    thread._pending_read_paths_by_layer = {}
    encoder = MagicMock()
    encoder.encode.side_effect = lambda value: value
    req = SimpleNamespace(
        local_block_ids=[list(range(9)), list(range(20, 29))],
        remote_cache_tokens=0,
        local_transed_tokens=8192 if computed else 0,
        local_computed_tokens=computed,
        chunk_finish=finished,
        remote_host="D",
        remote_engine_id="D-engine",
        remote_port=18050,
        group_member_idx=1,
        tp_ratio=2,
    )
    thread._process_send_task(SendTask(send_request={"req": req}, layer_idx=0, layer_name=name), encoder)
    msg = dealer.send.call_args.args[0]
    assert msg[3] == ([("req", [8], [28], 8, 8, 8192, 8257)] if computed else [])
    assert msg[4] == (["req"] if finished else [])


def test_dcp_sender_rejects_legacy_ack():
    thread = MembPullSendingThread.__new__(MembPullSendingThread)
    thread.timeout = 0.01
    thread._mf_meta_sent_paths = set()
    thread._state = ProducerSendState(
        last_layer_idx=0,
        layer_metadata={"layer": LayerMetadata([], [], [], [])},
        p_session="P",
        main_group_idx=0,
        indexer_group_idx=0,
        block_sizes=(128,),
        layer_storage_slots={},
        producer_dcp_size=8,
        producer_dcp_rank=1,
    )
    dealer = MagicMock()
    dealer.poll.return_value = True
    dealer.recv_multipart.return_value = [b"ACK"]
    with pytest.raises(RuntimeError, match="DCP-aware D"):
        thread._send_mf_meta("D", dealer, msgspec.msgpack.Encoder())
    dealer.recv_multipart.return_value = [msgspec.msgpack.encode((MF_META_ACK, 2))]
    thread._send_mf_meta("D", dealer, msgspec.msgpack.Encoder())
    msg = msgspec.msgpack.decode(dealer.send.call_args.args[0])
    assert msg[3] == 2
    layout = msgspec.msgpack.decode(msg[2])["layer"]["dcp"]
    assert layout == dict(size=8, rank=1, block_size=128, indexer_block_size=128)


@pytest.mark.parametrize("failure", [-1, RuntimeError("transport failure")])
def test_memfabric_dcp_read_failure_is_not_success(failure):
    thread = MembPullReadThread.__new__(MembPullReadThread)
    thread.tp_rank = 0
    thread._state = SimpleNamespace(
        tp_size=1,
        main_block_size=4,
        indexer_block_size=4,
        main_name_to_idx={"layer": 0},
        get_offload_layer_id=lambda _: 0,
        main_gva_bases=[(10000, 20000)],
        main_block_lens=[(8, 4)],
        indexer_tensors=[None],
        indexer_scale_tensors=[None],
    )
    thread._wait_for_dest_blocks = lambda *_: ([3], [])
    transfer = Mock(return_value=failure) if isinstance(failure, int) else Mock(side_effect=failure)
    thread.engine = SimpleNamespace(batch_transfer_sync_read=transfer)
    meta = dict(
        dcp=dict(size=8, rank=0, block_size=4, indexer_block_size=4),
        base_addrs=[1000, 2000],
        block_len=[8, 4],
        block_size_scale=[1, 1],
        main_tensor_count=2,
        has_indexer=False,
    )
    with pytest.raises(RuntimeError):
        thread._do_read_dcp_batch("layer", [("req", [2], [], 0, 0, 0, 3)], "P", meta, 0, 8)
    transfer.assert_called_once()


@pytest.mark.parametrize("start", [0, 1, 127, 128, 1025])
def test_empty_interval_needs_no_allocated_blocks(start):
    assert (
        list(dcp_block_segments([], [], start_block=0, token_start=start, token_end=start, block_size=128, dcp_size=4))
        == []
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"ratio": 0},
        {"ratio": 3},
        {"member": 2},
        {"tp_rank": 1},
        {"main_block_size": 64},
        {"indexer_block_size": 64},
    ],
)
def test_invalid_receiver_topology_or_block_size(changes):
    kwargs = dict(tp_size=2, tp_rank=0, member=1, ratio=2, main_block_size=128, indexer_block_size=128)
    kwargs.update(changes)
    with pytest.raises(ValueError):
        validate_dcp_layout_metadata(dict(size=4, rank=1, block_size=128, indexer_block_size=128), **kwargs)


@pytest.mark.parametrize("field,value", [("topk_k_bases", [1000]), ("dspark_context_receiver", object())])
def test_unsupported_offload_path_fails_before_transfer(field, value):
    thread = MembPullReadThread.__new__(MembPullReadThread)
    thread._state = SimpleNamespace(**{field: value})
    thread.engine = Mock()
    with pytest.raises(ValueError, match="fused-overlap"):
        thread._do_read_dcp_batch("layer", [], "P", {}, 0, 1)
    thread.engine.batch_transfer_sync_read.assert_not_called()


@pytest.mark.parametrize("fields", [2, 3, 4, 5, 7])
def test_parse_wire_read_descriptor(fields):
    wire = ["request", [3], [4], 2, 1, 257, 300][:fields]
    parsed = parse_read_request(wire)
    assert parsed == (
        "request",
        [3],
        [4] if fields >= 3 else [3],
        2 if fields >= 4 else 0,
        1 if fields >= 5 else 0,
    ) + ((257, 300) if fields == 7 else ())


@pytest.mark.parametrize(
    "wire", [[], ["req"], ["req", [], [], 0, 0, 5], ["req", [], [], 0, 0, 1, 2, 3], [None, []], [123, []]]
)
def test_parse_invalid_wire_read_descriptor(wire):
    with pytest.raises(ValueError):
        parse_read_request(wire)
