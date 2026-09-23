"""Unit tests for MooncakeHeterogeneousXferAdapter: wire protocol handshake + transfer state machine.

Tests three scenarios (design section 6.3 three-state response):
  1. Single request, all layers done -> FINISH.
  2. Partial layers done -> CONTINUE (mark_task_done returns None).
  3. Some tasks failed -> ERROR.

Uses real zmq ROUTER/DEALER on localhost (no mock) to validate the actual
wire protocol path. No NPU/GPU required — pure protocol + state logic.
"""

import msgspec
import pytest
import zmq

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_heterogeneous_adapter import (
    MooncakeHeterogeneousXferAdapter,
    MooncakeXferMetadata,
    MooncakeXferResponse,
    MooncakeXferResponseStatus,
)


def _make_metadata(num_layers: int = 2) -> MooncakeXferMetadata:
    """Construct a metadata shaped like upstream D's receive_kv_from_single_worker."""
    return MooncakeXferMetadata(
        remote_hostname="127.0.0.1",
        remote_port=9999,
        remote_tp_size=1,
        remote_tp_rank=0,
        req_blocks={
            "req-001": ("transfer-001", [[0, 1, 2]]),
        },
        kv_caches_base_addr=[0x1000, 0x2000],
        block_lens=[16384, 16384],
        kv_block_lens=[4096, 4096],
        registered_layer_names=[f"model.layers.{i}.self_attn" for i in range(num_layers)],
        registered_layer_indices=list(range(num_layers)),
        registered_group_indices=[0] * num_layers,
    )


def _round_trip(adapter: MooncakeHeterogeneousXferAdapter, port: int, metadata: MooncakeXferMetadata):
    """Simulate upstream D: DEALER connect, send metadata, recv response."""
    ctx = zmq.Context.instance()  # type: ignore
    sock = ctx.socket(zmq.DEALER)  # type: ignore
    sock.setsockopt(zmq.RCVTIMEO, 5000)  # type: ignore
    sock.connect(f"tcp://127.0.0.1:{port}")
    encoder = msgspec.msgpack.Encoder()
    sock.send(encoder.encode(metadata))
    resp_bytes = sock.recv()
    sock.close()
    return msgspec.msgpack.Decoder(MooncakeXferResponse).decode(resp_bytes)


def _free_port() -> int:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def adapter():
    port = _free_port()
    a = MooncakeHeterogeneousXferAdapter("127.0.0.1", port, num_layers=2)
    a.register_request("transfer-001", "p-req-001")
    a.start_listener()
    yield a, port
    a.stop()


def test_on_metadata_builds_tasks():
    """on_metadata builds one SendTask per registered layer (TP=1, 1:1 block map)."""
    a = MooncakeHeterogeneousXferAdapter("127.0.0.1", _free_port(), num_layers=2)
    a.register_request("transfer-001", "p-req-001")
    tasks = a.on_metadata(_make_metadata(num_layers=2))
    assert len(tasks) == 2  # one per layer
    assert all(t.transfer_id == "transfer-001" for t in tasks)
    assert {t.layer_index for t in tasks} == {0, 1}
    # TP=1: src block ids == dst block ids (1:1 mapping)
    for t in tasks:
        assert t.src_block_ids == t.dst_block_ids == [0, 1, 2]
        assert t.d_rank == 0


def test_partial_done_returns_none(adapter):
    """All layers not yet done -> mark_task_done returns None (CONTINUE)."""
    a, _ = adapter
    a.on_metadata(_make_metadata(num_layers=2))
    # Only layer 0 done
    resp = a.mark_task_done("transfer-001", layer_index=0, group_index=0)
    assert resp is None  # not all done -> no FINISH yet


def test_all_done_returns_finish(adapter):
    """All layers done -> FINISH with ok_reqs."""
    a, port = adapter
    a.on_metadata(_make_metadata(num_layers=2))
    a.mark_task_done("transfer-001", layer_index=0, group_index=0)
    resp = a.mark_task_done("transfer-001", layer_index=1, group_index=0)
    assert resp is not None
    assert resp.status == MooncakeXferResponseStatus.FINISH
    assert resp.ok_reqs == ["req-001"]


def test_failed_returns_error(adapter):
    """Any task failed -> ERROR with err_reqs."""
    a, _ = adapter
    a.on_metadata(_make_metadata(num_layers=2))
    a.mark_task_done("transfer-001", layer_index=0, group_index=0, ok=False)
    resp = a.mark_task_done("transfer-001", layer_index=1, group_index=0)
    assert resp is not None
    assert resp.status == MooncakeXferResponseStatus.ERROR
    assert resp.err_reqs == ["req-001"]


def test_wire_protocol_roundtrip(adapter):
    """End-to-end: D sends MooncakeXferMetadata, adapter ROUTER recv + decode + CONTINUE."""
    a, port = adapter
    metadata = _make_metadata(num_layers=2)
    resp = _round_trip(a, port, metadata)
    # Listener immediately replies CONTINUE WITHOUT ok_reqs: ok_reqs means
    # "KV transfer done for this req", but transfer hasn't started yet (tasks
    # execute async by connector). D's process_pulling_result decrements
    # pull_tasks_count on ok_reqs; CONTINUE must not carry them or D decodes
    # before KV arrives. Only the FINISH response (post-transfer) carries ok_reqs.
    assert resp.status == MooncakeXferResponseStatus.CONTINUE
    assert resp.ok_reqs is None


def test_full_handshake_finish(adapter):
    """End-to-end FINISH: D sends metadata, stays recv; connector marks all layers done;
    D receives CONTINUE then FINISH on the same connection.

    This mirrors upstream D's receive_kv_from_single_worker loop:
      send metadata -> while True: recv -> CONTINUE (process) -> FINISH (break).
    """
    a, port = adapter
    ctx = zmq.Context.instance()  # type: ignore
    sock = ctx.socket(zmq.DEALER)  # type: ignore
    sock.setsockopt(zmq.RCVTIMEO, 5000)  # type: ignore
    sock.connect(f"tcp://127.0.0.1:{port}")
    encoder = msgspec.msgpack.Encoder()
    resp_decoder = msgspec.msgpack.Decoder(MooncakeXferResponse)

    metadata = _make_metadata(num_layers=2)
    sock.send(encoder.encode(metadata))

    # D blocks recv — first reply is CONTINUE (listener immediate ack)
    first = resp_decoder.decode(sock.recv())
    assert first.status == MooncakeXferResponseStatus.CONTINUE

    # Simulate connector completing all layers (async, after CONTINUE)
    a.mark_task_done("transfer-001", layer_index=0, group_index=0)
    final_resp = a.mark_task_done("transfer-001", layer_index=1, group_index=0)
    assert final_resp is not None

    # D receives the FINISH pushed back by mark_task_done on same connection
    pushed = resp_decoder.decode(sock.recv())
    assert pushed.status == MooncakeXferResponseStatus.FINISH
    assert pushed.ok_reqs == ["req-001"]
    sock.close()


def test_full_handshake_error(adapter):
    """End-to-end ERROR: a failed task surfaces ERROR to D instead of FINISH."""
    a, port = adapter
    ctx = zmq.Context.instance()  # type: ignore
    sock = ctx.socket(zmq.DEALER)  # type: ignore
    sock.setsockopt(zmq.RCVTIMEO, 5000)  # type: ignore
    sock.connect(f"tcp://127.0.0.1:{port}")
    encoder = msgspec.msgpack.Encoder()
    resp_decoder = msgspec.msgpack.Decoder(MooncakeXferResponse)

    sock.send(encoder.encode(_make_metadata(num_layers=2)))
    assert resp_decoder.decode(sock.recv()).status == MooncakeXferResponseStatus.CONTINUE

    a.mark_task_done("transfer-001", layer_index=0, group_index=0, ok=False)
    final_resp = a.mark_task_done("transfer-001", layer_index=1, group_index=0)
    assert final_resp is not None
    assert final_resp.status == MooncakeXferResponseStatus.ERROR

    pushed = resp_decoder.decode(sock.recv())
    assert pushed.status == MooncakeXferResponseStatus.ERROR
    sock.close()


def test_to_agent_metadata_mapping():
    """to_agent_metadata maps upstream MooncakeXferMetadata -> P-side agent metadata shape.

    D (GPU FA vllm 0.28) 4D HND layout: 1 region/layer, shape
    (num_blocks, num_kv_heads, block_size, 2*head_size), k/v interleaved in last
    dim. block_len is the full k+v block, kv_block_len = block_len//2.
    to_agent_metadata keeps D's 1 region (does not expand) so the connector
    merges P's k/v into this 4D HND layout.
    """
    a = MooncakeHeterogeneousXferAdapter("127.0.0.1", _free_port(), num_layers=2)
    # D 4D HND: block_len=16384 (k+v), kv_block_len=8192 (each half).
    metadata = MooncakeXferMetadata(
        remote_hostname="127.0.0.1",
        remote_port=9999,
        remote_tp_size=1,
        remote_tp_rank=0,
        req_blocks={"req-001": ("transfer-001", [[0, 1]])},
        kv_caches_base_addr=[0x1000, 0x2000],
        block_lens=[16384, 16384],
        kv_block_lens=[8192, 8192],
        registered_layer_names=["model.layers.0.self_attn", "model.layers.1.self_attn"],
        registered_layer_indices=[0, 1],
        registered_group_indices=[0, 0],
    )
    agent = a.to_agent_metadata(metadata)

    assert agent["te_rpc_port"] == 9999  # remote_port
    assert set(agent["layer_metadata"].keys()) == {
        "model.layers.0.self_attn",
        "model.layers.1.self_attn",
    }
    lm0 = agent["layer_metadata"]["model.layers.0.self_attn"]
    # D is 1 region (k/v interleaved 4D), not expanded into 2.
    assert lm0["kv_caches_base_addr"] == [0x1000]
    assert lm0["block_len"] == [16384]  # merged block stride
    assert lm0["kv_block_len"] == 8192
    assert lm0["tensor_group_idx"] == [0]
    assert lm0["block_size_scale"] == [1]
    lm1 = agent["layer_metadata"]["model.layers.1.self_attn"]
    assert lm1["kv_caches_base_addr"] == [0x2000]
    assert lm1["block_len"] == [16384]


def test_remote_metadata_cache_after_on_metadata():
    """on_metadata stores D-side metadata in _remote_metadata_cache; get_remote_metadata returns it."""
    a = MooncakeHeterogeneousXferAdapter("127.0.0.1", _free_port(), num_layers=2)
    a.register_request("transfer-001", "p-req-001")
    # Before on_metadata: cache empty
    assert a.get_remote_metadata("transfer-001") is None
    assert a.get_transfer_id_for_request("p-req-001") == "transfer-001"

    a.on_metadata(_make_metadata(num_layers=2))
    cached = a.get_remote_metadata("transfer-001")
    assert cached is not None
    assert cached["te_rpc_port"] == 9999  # remote_port
    assert cached["remote_hostname"] == "127.0.0.1"
    assert "model.layers.0.self_attn" in cached["layer_metadata"]
    # req_blocks preserved so the connector can retrieve D's target block ids.
    assert "req_blocks" in cached
    assert "req-001" in cached["req_blocks"]
