"""Unit tests for MooncakeHeterogeneousConnector.

Tests the request-triggered (non-layerwise) P-side connector that talks to
upstream vLLM MooncakeConnector's wire protocol.

Strategy:
  - Scheduler: pure logic, no mock needed.
  - Worker: mock TransferEngine + global_te + torch.npu (no NPU required).
  - End-to-end: real zmq (D sends MooncakeXferMetadata -> adapter recv ->
    worker _do_transfer -> mark_task_done -> D receives FINISH).
"""

from unittest.mock import MagicMock, patch

import msgspec
import pytest
import torch
import zmq
from vllm.v1.kv_cache_interface import MambaSpec, MLAAttentionSpec

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_heterogeneous_adapter import (
    MooncakeHeterogeneousXferAdapter,
    MooncakeXferMetadata,
    MooncakeXferResponse,
    MooncakeXferResponseStatus,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_heterogeneous_connector import (
    LayerMeta,
    MooncakeHeterogeneousConnector,
    MooncakeHeterogeneousConnectorMetadata,
    MooncakeHeterogeneousConnectorScheduler,
    MooncakeHeterogeneousConnectorWorker,
)

# ---- Scheduler tests (pure logic) ----


def _mock_request(req_id="req-1", transfer_id="transfer-1"):
    req = MagicMock()
    req.request_id = req_id
    req.kv_transfer_params = {"transfer_id": transfer_id}
    return req


def test_scheduler_get_num_new_matched_tokens():
    sched = MagicMock(spec=MooncakeHeterogeneousConnectorScheduler)
    sched.get_num_new_matched_tokens = MooncakeHeterogeneousConnectorScheduler.get_num_new_matched_tokens.__get__(sched)
    assert sched.get_num_new_matched_tokens(MagicMock(), 0) == (0, False)


def test_scheduler_update_state_after_alloc_records_transfer():
    # Construct scheduler without full VllmConfig (use object trick)
    sched = MooncakeHeterogeneousConnectorScheduler.__new__(MooncakeHeterogeneousConnectorScheduler)
    sched._reqs_need_send = {}
    req = _mock_request()
    sched.update_state_after_alloc(req, MagicMock(), 0)
    assert "req-1" in sched._reqs_need_send
    assert sched._reqs_need_send["req-1"][0] == "transfer-1"


def test_scheduler_request_finished_fills_block_ids():
    sched = MooncakeHeterogeneousConnectorScheduler.__new__(MooncakeHeterogeneousConnectorScheduler)
    sched._reqs_need_send = {}
    req = _mock_request()
    sched.update_state_after_alloc(req, MagicMock(), 0)
    # Before request_finished: local_block_ids empty
    assert sched._reqs_need_send["req-1"][2] == []
    result = sched.request_finished(req, [0, 1, 2])
    assert result == (True, None)  # delayed release
    assert sched._reqs_need_send["req-1"][2] == [0, 1, 2]


def test_scheduler_request_finished_untracked_returns_false():
    """[P1] Untracked request (no transfer_id / not in _reqs_need_send) must
    return False so the scheduler frees blocks immediately — otherwise the
    worker never reports completion and blocks leak permanently."""
    sched = MooncakeHeterogeneousConnectorScheduler.__new__(MooncakeHeterogeneousConnectorScheduler)
    sched._reqs_need_send = {}  # req not registered via update_state_after_alloc
    req = _mock_request()
    result = sched.request_finished(req, [0, 1, 2])
    assert result == (False, None)  # immediate release, no leak


def test_scheduler_request_finished_empty_blocks_returns_false():
    """[P1] Empty block_ids must return False (immediate release); delaying
    release of a blockless request leaks blocks."""
    sched = MooncakeHeterogeneousConnectorScheduler.__new__(MooncakeHeterogeneousConnectorScheduler)
    sched._reqs_need_send = {}
    req = _mock_request()
    sched.update_state_after_alloc(req, MagicMock(), 0)
    result = sched.request_finished(req, [])
    assert result == (False, None)  # immediate release


def test_scheduler_build_connector_meta_emits_ready_reqs():
    sched = MooncakeHeterogeneousConnectorScheduler.__new__(MooncakeHeterogeneousConnectorScheduler)
    sched._reqs_need_send = {}
    req = _mock_request()
    sched.update_state_after_alloc(req, MagicMock(), 0)
    sched.request_finished(req, [0, 1, 2])
    meta = sched.build_connector_meta(MagicMock())
    assert isinstance(meta, MooncakeHeterogeneousConnectorMetadata)
    assert "req-1" in meta.reqs_to_send
    assert meta.reqs_to_send["req-1"] == ("transfer-1", [0, 1, 2])
    # req popped from _reqs_need_send after build
    assert "req-1" not in sched._reqs_need_send


def test_scheduler_request_finished_all_groups_single_group():
    sched = MooncakeHeterogeneousConnectorScheduler.__new__(MooncakeHeterogeneousConnectorScheduler)
    sched._reqs_need_send = {}
    req = _mock_request()
    sched.update_state_after_alloc(req, MagicMock(), 0)
    result = sched.request_finished_all_groups(req, ([0, 1],))
    assert result == (True, None)
    assert sched._reqs_need_send["req-1"][2] == [0, 1]


def test_requires_piecewise_for_cudagraph_false():
    assert MooncakeHeterogeneousConnector.requires_piecewise_for_cudagraph({}) is False


# ---- Worker transfer logic (mock TransferEngine) ----


def _make_worker_with_mock_engine():
    """Build a worker without NPU by mocking global_te and TransferEngine.

    P (NPU) KV layout: k/v split, 2 regions per layer, each block_len=4096
    (= block_size * num_kv_heads * head_dim * element_size).
    """
    worker = MooncakeHeterogeneousConnectorWorker.__new__(MooncakeHeterogeneousConnectorWorker)
    worker.total_layers = 2
    # Empty spec mapping: _do_transfer dispatch treats a None spec as standard attention.
    worker.kv_cache_specs_by_name = {}
    worker.engine = MagicMock()
    worker.engine.batch_transfer_sync_write.return_value = 0
    # P: 2 regions/layer (k, v separated). NHD k_cache/v_cache shape
    # (num_blocks, block_size, num_kv_heads, head_size) = (8, 4, 2, 128).
    # d=128 makes the hnd_buf block bytes (h*bs*2d*float32_elem = 2*4*256*4 = 8192) == D block_len,
    # aligning the same-model-config precondition (P stride == D block_len; see BUG-2 transfer_len).
    # layer block_len (the else fallback) = bs*h*d*2 = 1024; the standard branch does not use it.
    bs, h, d = 4, 2, 128
    worker.layer_metadata = {
        "model.layers.0.self_attn": LayerMeta(
            kv_caches_base_addr=[0x100000, 0x140000], block_len=[bs * h * d * 2, bs * h * d * 2]
        ),
        "model.layers.1.self_attn": LayerMeta(
            kv_caches_base_addr=[0x200000, 0x240000], block_len=[bs * h * d * 2, bs * h * d * 2]
        ),
    }
    # mock P k/v caches (NHD) — _do_transfer does index_select + permute + cat on these.
    worker.kv_caches = {
        "model.layers.0.self_attn": [torch.zeros(8, bs, h, d), torch.zeros(8, bs, h, d)],
        "model.layers.1.self_attn": [torch.zeros(8, bs, h, d), torch.zeros(8, bs, h, d)],
    }
    # mock global_te.register_buffer (imported in connector as global_te)
    import vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_heterogeneous_connector as conn_mod

    conn_mod.global_te = MagicMock()
    worker._completed_transfers = set()
    worker._failed_transfers = set()
    worker._transfer_to_req = {}
    worker._pending_hnd_bufs = {}
    worker._reported_finished = set()
    # mock adapter: mark_all_tasks_done returns None (no FINISH sent in unit context)
    worker.adapter = MagicMock()
    worker.adapter.mark_all_tasks_done.return_value = None
    return worker


def _make_remote_meta(d_block_ids=(0, 1), transfer_id="transfer-1"):
    """D-side metadata as stored by adapter.get_remote_metadata.

    D (GPU FA vllm 0.28) 4D HND layout: 1 region/layer, shape
    (num_blocks, num_kv_heads, block_size, 2*head_size). k/v interleaved in last
    dim. block_len = 2*block_size*num_kv_heads*head_size*elem, kv_block_len =
    block_len//2. req_blocks carries D's target block ids.
    """
    return {
        "remote_hostname": "10.0.0.1",
        "te_rpc_port": 9999,
        "layer_metadata": {
            "model.layers.0.self_attn": {
                "kv_caches_base_addr": [0x500000],
                "block_len": [8192],
                "kv_block_len": 4096,
                "tensor_group_idx": [0],
                "block_size_scale": [1],
            },
            "model.layers.1.self_attn": {
                "kv_caches_base_addr": [0x600000],
                "block_len": [8192],
                "kv_block_len": 4096,
                "tensor_group_idx": [0],
                "block_size_scale": [1],
            },
        },
        "req_blocks": {"d-req-1": (transfer_id, [list(d_block_ids)])},
    }


def test_worker_do_transfer_address_computation():
    """_do_transfer merges P k/v into D 4D HND and transfers 1 region per layer.

    P k/v split (NHD) -> D merged (4D HND, k/v interleaved). Each layer produces
    1 transfer entry per D block (merged k+v). P blocks from local_block_ids,
    D blocks from req_blocks.
    """
    worker = _make_worker_with_mock_engine()
    # P local blocks [0,1], D blocks [10,11] (independent allocation, not 1:1)
    worker._do_transfer("req-1", "transfer-1", [0, 1], _make_remote_meta(d_block_ids=(10, 11)))
    call_args = worker.engine.batch_transfer_sync_write.call_args
    session_id, src_list, dst_list, length_list = call_args[0]
    assert session_id == "10.0.0.1:9999"
    # 2 layers * 1 merged region * 2 D blocks = 4 transfer entries (was 8 with k/v split)
    assert len(src_list) == 4
    # transfer_len = P hnd_buf stride = 8192 (== D block_len under same-model config;
    # BUG-2 uses P stride rather than assuming == D block_len).
    assert all(length == 8192 for length in length_list)
    # layer0 region: dst = D_base + d_bid * d_block_len; D block 10 -> 0x500000+10*8192
    assert dst_list[0] == 0x500000 + 10 * 8192
    assert dst_list[1] == 0x500000 + 11 * 8192
    # layer1 region: D block 10 -> 0x600000+10*8192
    assert dst_list[2] == 0x600000 + 10 * 8192
    assert "transfer-1" in worker._completed_transfers


def test_worker_do_transfer_stride_mismatch_uses_p_stride_and_warns():
    """BUG-2 regression: when P hnd_buf stride != D block_len, transfer_len uses the actual P stride
    (not an assumption of == D block_len) and warns on the mismatch, avoiding RDMA misreads/overruns.

    Under the same model config stride == d_block_len; they differ when P/D dtype or head config differ.
    Using P stride keeps the src address on the real memory layout; the warning does not interrupt serve.
    """
    import vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_heterogeneous_connector as conn_mod

    worker = _make_worker_with_mock_engine()
    # Shrink P k/v head_size to 16: hnd_buf block bytes = h*bs*2d*elem = 2*4*32*4 = 1024,
    # while D block_len stays 8192 (the _make_remote_meta default) -> the stride != d_block_len case.
    bs, h, d = 4, 2, 16
    for ln in worker.kv_caches:
        worker.kv_caches[ln] = [torch.zeros(8, bs, h, d), torch.zeros(8, bs, h, d)]
    # vllm logger propagate=False (custom stdout handler); caplog cannot capture it,
    # so patch.object intercepts logger.warning directly.
    with patch.object(conn_mod.logger, "warning") as mock_warn:
        worker._do_transfer("req-1", "transfer-1", [0, 1], _make_remote_meta(d_block_ids=(10, 11)))
    call_args = worker.engine.batch_transfer_sync_write.call_args
    _, _, _, length_list = call_args[0]
    # transfer_len uses P hnd_buf stride = 1024, not D block_len 8192.
    assert all(length == 1024 for length in length_list), length_list
    # Warns on the stride mismatch (one per layer, 2 layers).
    warn_msgs = [str(c.args) for c in mock_warn.call_args_list]
    assert any("stride" in m and "block_len" in m for m in warn_msgs), warn_msgs


def test_worker_do_transfer_empty_d_blocks_maps_and_reports():
    """BUG-1 regression: the _do_transfer early-return path (empty D block ids) still sets _transfer_to_req
    so get_finished reports the req_id and frees P blocks. Before the fix this path skipped the assignment

    and blocks leaked forever. The entry-level _transfer_to_req[transfer_id]=req_id covers all early-return
    paths (empty d_block_ids, P<D, empty src_list), not a per-return assignment — this test covers the empty path.
    """
    worker = _make_worker_with_mock_engine()
    # Empty D block ids -> _do_transfer early-returns (mark_failed + return), no batch_transfer.
    worker._do_transfer("req-3", "transfer-3", [0, 1], _make_remote_meta(d_block_ids=(), transfer_id="transfer-3"))
    assert "transfer-3" in worker._failed_transfers
    assert worker._transfer_to_req.get("transfer-3") == "req-3"  # BUG-1: entry-level assignment
    # batch_transfer not called (early return).
    assert worker.engine.batch_transfer_sync_write.call_args is None
    _, save_fin = worker.get_finished(set())
    assert save_fin == {"req-3"}  # blocks freed via report


def test_worker_do_transfer_mla_reuses_standard_branch():
    """Stage 2: MLA (kv_c, k_pe) 2-tuple has the same structure as standard attention (k, v),
    so _do_transfer reuses the standard attention hnd_buf branch (cat + permute); no raise.

    P NPU MLA cache = 2-tuple (kv_c, k_pe); D GPU = single tensor kv_c_and_k_pe_cache
    (kv_c first + k_pe after, concat on head_dim). The cat logic matches standard attention (2-tuple -> cat last dim).
    This test verifies the MLA spec does not raise NotImplementedError and transfers via the standard branch.
    Layout correctness (kv_lora_rank/qk_rope_head_dim head_dim) is pending DeepSeek hardware verification.
    """
    worker = _make_worker_with_mock_engine()
    # MLA spec (build a minimal mock; only the isinstance check matters; frozen dataclass).
    mla_spec = MLAAttentionSpec(block_size=4, num_kv_heads=2, head_size=16, dtype=torch.float16)
    worker.kv_cache_specs_by_name = {
        "model.layers.0.self_attn": mla_spec,
        "model.layers.1.self_attn": mla_spec,
    }
    # kv_caches stays a 2-tuple ([kv_c, k_pe]), same structure as standard attention [k, v]
    # (_make_worker already sets kv_caches to a 2-tuple 4D tensor; MLA reuses it).
    worker._do_transfer("req-1", "transfer-1", [0, 1], _make_remote_meta(d_block_ids=(10, 11)))
    # No NotImplementedError + normal transfer (like standard attention, 4 entries).
    call_args = worker.engine.batch_transfer_sync_write.call_args
    _, src_list, _, _ = call_args[0]
    assert len(src_list) == 4
    assert "transfer-1" in worker._completed_transfers


def test_worker_do_transfer_mamba_raises():
    """Stage 4: Mamba (SSM state) is not implemented; _do_transfer raises NotImplementedError on MambaSpec
    (explicit error, not silent).

    Mamba cache = (conv, ssm); D only registers conv (upstream get_transfer_cache_regions for
    Mamba returns [conv]). The transfer logic involves mamba_cache_mode/num_speculative_blocks,
    very different from standard attention; deferred to a Mamba-specific hardware implementation.
    """
    worker = _make_worker_with_mock_engine()
    mamba_spec = MambaSpec(
        block_size=4,
        shapes=((4, 2, 16), (4, 2, 16)),
        dtypes=(torch.float16,),
    )
    worker.kv_cache_specs_by_name = {"model.layers.0.self_attn": mamba_spec}
    # Mamba cache = (conv, ssm)
    worker.kv_caches = {
        "model.layers.0.self_attn": [torch.zeros(8, 4, 2, 16), torch.zeros(8, 4, 2, 16)],
        "model.layers.1.self_attn": [torch.zeros(8, 4, 2, 16), torch.zeros(8, 4, 2, 16)],
    }
    with pytest.raises(NotImplementedError, match="Mamba"):
        worker._do_transfer("req-1", "transfer-1", [0, 1], _make_remote_meta())


def test_get_finished_returns_completed_reqs():
    """get_finished returns req_ids of completed transfers (for block release)."""
    worker = _make_worker_with_mock_engine()
    worker._do_transfer("req-1", "transfer-1", [0, 1], _make_remote_meta())
    load_fin, save_fin = worker.get_finished(set())
    assert save_fin == {"req-1"}
    # second call: already reported, not returned again
    _, save_fin2 = worker.get_finished(set())
    assert save_fin2 is None


def test_get_finished_clears_transfer_state():
    """P1-a regression: get_finished clears transfer state after reporting, so a long-running P service
    does not grow with request count. Reported transfer_ids are removed from _completed/_failed_transfers
    and _transfer_to_req, and _reported_finished is cleared too."""
    worker = _make_worker_with_mock_engine()
    worker._do_transfer("req-1", "transfer-1", [0, 1], _make_remote_meta())
    worker._do_transfer("req-2", "transfer-2", [0, 1], _make_remote_meta(transfer_id="transfer-2"))
    assert len(worker._completed_transfers) == 2
    _, save_fin = worker.get_finished(set())
    assert save_fin == {"req-1", "req-2"}
    # After cleanup the sets are empty (no accumulation).
    assert worker._completed_transfers == set()
    assert worker._failed_transfers == set()
    assert worker._transfer_to_req == {}
    assert worker._reported_finished == set()
    # Second call does not re-report (transfer_ids cleared; iterates an empty set).
    _, save_fin2 = worker.get_finished(set())
    assert save_fin2 is None


def test_get_finished_returns_failed_reqs():
    """get_finished also returns req_ids of failed transfers."""
    worker = _make_worker_with_mock_engine()
    worker.engine.batch_transfer_sync_write.return_value = -1
    worker._do_transfer("req-2", "transfer-2", [0, 1], _make_remote_meta(transfer_id="transfer-2"))
    _, save_fin = worker.get_finished(set())
    assert save_fin == {"req-2"}


def test_worker_do_transfer_failure_marks_failed():
    """Failed batch_transfer_sync_write marks transfer as failed."""
    worker = _make_worker_with_mock_engine()
    worker.engine.batch_transfer_sync_write.return_value = -1
    worker._do_transfer("req-1", "transfer-1", [0, 1], _make_remote_meta())
    assert "transfer-1" in worker._failed_transfers
    assert "transfer-1" not in worker._completed_transfers


def test_start_load_kv_timeout_marks_failed_and_reports(monkeypatch):
    """[P1] When D pull metadata never arrives (timeout), the transfer must be
    marked failed + mapped so get_finished reports it and frees P blocks.
    Otherwise request_finished's delayed release leaks blocks forever."""
    import vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_heterogeneous_connector as conn_mod

    monkeypatch.setattr(conn_mod.time, "sleep", lambda *_a, **_kw: None)
    worker = _make_worker_with_mock_engine()
    # adapter never produces remote metadata -> start_load_kv times out
    worker.adapter.get_remote_metadata.return_value = None
    worker._pending_hnd_bufs = {}
    meta = MooncakeHeterogeneousConnectorMetadata()
    meta.reqs_to_send["req-9"] = ("transfer-9", [0, 1])
    worker.start_load_kv(meta)
    assert "transfer-9" in worker._failed_transfers
    assert worker._transfer_to_req.get("transfer-9") == "req-9"
    _, save_fin = worker.get_finished(set())
    assert save_fin == {"req-9"}  # blocks freed via report


def test_worker_save_kv_layer_is_noop():
    worker = _make_worker_with_mock_engine()
    worker.save_kv_layer()  # should not raise
    worker.wait_for_layer_load("any")
    worker.wait_for_save()


# ---- End-to-end: adapter + worker transfer + FINISH ----


def _free_port():
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_end_to_end_d_push_worker_transfer_d_finish():
    """Full flow: D sends MooncakeXferMetadata -> adapter recv+cache ->
    worker _do_transfer (mock engine) -> mark_task_done -> D receives FINISH.

    This validates the request-triggered model end-to-end (protocol + transfer
    trigger + completion), with only the NPU transport mocked.
    """
    port = _free_port()
    adapter = MooncakeHeterogeneousXferAdapter("127.0.0.1", port, num_layers=2)
    adapter.register_request("transfer-1", "req-1")
    adapter.start_listener()

    worker = _make_worker_with_mock_engine()
    worker.adapter = adapter

    try:
        # D sends MooncakeXferMetadata (merged blocks-first layout: block_len=8192,
        # kv_block_len=4096). req_blocks carries D's target block ids [0,1].
        metadata = MooncakeXferMetadata(
            remote_hostname="10.0.0.1",
            remote_port=9999,
            remote_tp_size=1,
            remote_tp_rank=0,
            req_blocks={"req-1": ("transfer-1", [[0, 1]])},
            kv_caches_base_addr=[0x500000, 0x600000],
            block_lens=[8192, 8192],
            kv_block_lens=[4096, 4096],
            registered_layer_names=["model.layers.0.self_attn", "model.layers.1.self_attn"],
            registered_layer_indices=[0, 1],
            registered_group_indices=[0, 0],
        )
        ctx = zmq.Context.instance()  # type: ignore
        sock = ctx.socket(zmq.DEALER)  # type: ignore
        sock.setsockopt(zmq.RCVTIMEO, 5000)  # type: ignore
        sock.connect(f"tcp://127.0.0.1:{port}")
        encoder = msgspec.msgpack.Encoder()
        resp_decoder = msgspec.msgpack.Decoder(MooncakeXferResponse)
        sock.send(encoder.encode(metadata))

        # D receives CONTINUE (adapter immediate ack)
        first = resp_decoder.decode(sock.recv())
        assert first.status == MooncakeXferResponseStatus.CONTINUE

        # Wait for adapter to process metadata + cache it
        import time

        time.sleep(0.5)

        # Worker triggers transfer (simulating start_load_kv)
        remote_meta = adapter.get_remote_metadata("transfer-1")
        assert remote_meta is not None, "adapter should have cached remote metadata"
        worker._do_transfer("req-1", "transfer-1", [0, 1], remote_meta)

        # D receives FINISH (pushed by mark_task_done)
        pushed = resp_decoder.decode(sock.recv())
        assert pushed.status == MooncakeXferResponseStatus.FINISH
        assert "transfer-1" in worker._completed_transfers
        sock.close()
    finally:
        adapter.stop()
