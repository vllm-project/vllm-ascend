# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm import SamplingParams
from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.utils.hashing import sha256
from vllm.v1.core import kv_cache_utils as kv_utils
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.sched.output import KVConnectorBlockState, NewRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from tests.ut.distributed.ascend_store import test_request_lifecycle as lc
from tests.ut.distributed.ascend_store.cache_config_helpers import make_resolved_cache_config
from tests.ut.distributed.ascend_store.test_request_lifecycle import devices_and_store  # noqa: F401
from tests.ut.kv_offload.utils import assert_scheduler_empty, create_model_runner_output, create_vllm_config
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import metadata as md_mod
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import layerwise_cache_layout as lw_layout
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_scheduler import LayerwiseKVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler

PKG = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store"
MOONCAKE_CLIENT = f"{PKG}.backend.mooncake_backend.MooncakeBackend.create_scheduler_client"
MEMCACHE_CLIENT = f"{PKG}.backend.memcache_backend.MemcacheBackend.create_scheduler_client"
pytestmark = pytest.mark.usefixtures("devices_and_store")


def _make_cfg(
    backend: str = "cpu",
    kv_role: str = "kv_both",
    block_size: int = 128,
    num_blocks: int = 32,
    prefix_match_unit: int | None = None,
    extra_groups: list[KVCacheGroupSpec] | None = None,
    extra_config: dict[str, Any] | None = None,
) -> tuple[VllmConfig, KVCacheConfig]:
    kv_cfg = KVTransferConfig(
        kv_connector="AscendStoreConnector",
        kv_role=kv_role,
        kv_connector_module_path=f"{PKG}.ascend_store_connector",
        kv_connector_extra_config={"backend": backend, "lookup_rpc_port": "0", **(extra_config or {})},
    )
    cfg = create_vllm_config(max_num_batched_tokens=2048, block_size=block_size, kv_transfer_config=kv_cfg)
    cfg.cache_config.num_gpu_blocks, cfg.cache_config.prefix_match_unit = num_blocks, prefix_match_unit
    attn = FullAttentionSpec(block_size=block_size, num_kv_heads=1, head_size=16, dtype=torch.float16)
    groups = [KVCacheGroupSpec(["model.layers.0.kv"], attn), *(extra_groups or [])]
    cfg.scheduler_config.disable_hybrid_kv_cache_manager = len(groups) <= 1
    plan = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups)
    plan.kv_transfer_config = kv_cfg
    return cfg, make_resolved_cache_config(cfg, plan)  # type: ignore[return-value]


def _make_req(req_id: str, tokens: list[int], cap: Any = ..., hash_bs: int = 128) -> Request:
    kv_utils.init_none_hash(sha256)
    extra = None if cap is ... else {"kv_transfer_params": {"max_offload_tokens": cap}}
    sp = SamplingParams(max_tokens=2, extra_args=extra)
    return Request(
        req_id, tokens, sp, pooling_params=None, block_hasher=kv_utils.get_request_block_hasher(hash_bs, sha256)
    )


def _run_batch(sched: Scheduler, worker: Any, raw: torch.Tensor, reqs: list[tuple[Request, int]]) -> None:
    pool, req_map, verified = worker.connector_worker, {r.request_id: r for r, _ in reqs}, set()
    for r, _ in reqs:
        sched.add_request(r)
    for _ in range(16):
        out = sched.schedule()
        worker.handle_preemptions(out.kv_connector_metadata)
        worker.bind_connector_metadata(out.kv_connector_metadata)
        worker.start_load_kv(SimpleNamespace() if out.total_num_scheduled_tokens else None)
        if pool.kv_recv_thread and any(r.status == RequestStatus.WAITING_FOR_REMOTE_KVS for r in req_map.values()):
            deadline = time.monotonic() + 5.0
            while not pool.kv_recv_thread.finished_requests:
                pool.kv_recv_thread.raise_if_failed()
                assert time.monotonic() < deadline
                time.sleep(0.001)
        for req, hit in reqs:
            if (rid := req.request_id) not in sched.requests:
                continue
            bids = sched.kv_cache_manager.get_blocks(rid).get_block_ids()[0]
            meta = out.kv_connector_metadata
            if hit and meta and any(m.req_id == rid and m.load_spec and m.load_spec.can_load for m in meta.requests):
                ptoks = req.prompt_token_ids
                for idx, bid in enumerate(bids):
                    if bid and (idx + 1) * 128 <= hit:
                        for p, nm in ((0, "k"), (1, "v")):
                            assert torch.all(raw[p, bid] == lc.payload_value(ptoks, (idx + 1) * 128, nm, 0))
                verified.add(rid)
            if sched_toks := out.num_scheduled_tokens.get(rid, 0):
                end, toks = req.num_computed_tokens, list(req.all_token_ids)
                for idx, bid in enumerate(bids):
                    if bid and (end - sched_toks) < (idx + 1) * 128 <= end:
                        for p, nm in ((0, "k"), (1, "v")):
                            raw[p, bid].fill_(lc.payload_value(toks, (idx + 1) * 128, nm, 0))
        worker.wait_for_save()
        pool.wait_for_previous_save()
        sending, recving = worker.get_finished(out.finished_req_ids)
        s_reqs = [req_map[rid] for rid, cnt in out.num_scheduled_tokens.items() if cnt > 0]
        res = create_model_runner_output(s_reqs)
        res.sampled_token_ids = [[] if r.num_computed_tokens < len(r.prompt_token_ids) else [0] for r in s_reqs]
        res.kv_connector_output = KVConnectorOutput(sending, recving, worker.build_connector_worker_meta())
        sched.update_from_output(out, res)
        worker.clear_connector_metadata()
        if all(r.is_finished() and rid not in sched.requests for rid, r in req_map.items()):
            if (drain := sched.schedule()).finished_req_ids:
                worker.bind_connector_metadata(drain.kv_connector_metadata)
                worker.get_finished(drain.finished_req_ids)
                worker.clear_connector_metadata()
            break
    assert verified == {r.request_id for r, hit in reqs if hit > 0}


def test_validation_and_req_meta_boundaries(caplog: pytest.LogCaptureFixture) -> None:
    """Validate param parsing and ReqMeta block rounding, zero disablement, and None default."""
    assert md_mod.get_max_offload_tokens(None) is None and md_mod.get_max_offload_tokens({}) is None
    bad_good = ((None, None, False), (0, 0, False), (100, 100, False), (False, None, True))
    for val, exp, warn in bad_good + ((-1, None, True), (1.5, None, True), ("0", None, True)):
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            res = md_mod.get_max_offload_tokens({"max_offload_tokens": val})
        assert res == exp and (exp is None or type(res) is int)
        assert any("max_offload_tokens" in r.message for r in caplog.records) is warn

    hashes: list[kv_utils.BlockHash] = [f"h{i}".encode() for i in range(16)]  # type: ignore[list-item]
    z_tr = md_mod.RequestTracker("z", 36, allocated_block_ids=[1, 2, 3], last_block_gva=4096, max_offload_tokens=0)
    assert md_mod.ReqMeta.from_request_tracker(z_tr, 16, block_hashes=hashes[:2]) is None
    ls = md_mod.LoadSpec(0, 32, True)
    z_ld = md_mod.ReqMeta.from_request_tracker(z_tr, 16, load_spec=ls, block_hashes=hashes[:2])
    assert z_ld is not None and (z_ld.can_save, z_ld.target_token_len) == (False, 36)
    bids_c = list(range(1, 18))
    for cap, exp_pre, exp_dec in ((15, 0, 0), (20, 16, 0), (100, 96, 0), (200, 128, 192), (None, 128, 208)):
        tr = md_mod.RequestTracker("c", 134, allocated_block_ids=bids_c, last_block_gva=4096, max_offload_tokens=cap)
        meta = md_mod.ReqMeta.from_request_tracker(tr, 16, block_hashes=hashes)
        assert (meta.token_len_chunk if meta else 0) == exp_pre
        assert meta is None or cap is None or meta.partial_block_index is None
        tr.token_len = 208
        dec = md_mod.ReqMeta.from_request_tracker(tr, 16, block_hashes=hashes)
        assert (dec.token_len_chunk if dec else 0) == exp_dec


@pytest.mark.parametrize("load_async", [False, True])
def test_real_scheduler_offload_limits_and_mixed_batch(monkeypatch: pytest.MonkeyPatch, load_async: bool) -> None:
    """Real Scheduler + AscendStoreConnector honors 0, positive cap, None, read invariance, and mixed batches."""
    puts: list[str] = []
    orig_put = lc.MemoryBackend.put

    def _rec_put(self: lc.MemoryBackend, ks: list[str], a: Any, sz: Any, **kw: Any) -> list[bool]:
        res = orig_put(self, ks, a, sz, **kw)
        puts.extend(k for k, ok in zip(ks, res) if ok)
        return res

    monkeypatch.setattr(lc.MemoryBackend, "put", _rec_put)
    cfg, plan = _make_cfg(extra_config={"load_async": load_async, "save_decode_cache": True})
    cfg.cache_config.prefix_cache_retention_interval = None
    bs, hs = kv_utils.resolve_kv_cache_block_sizes(plan, cfg)
    worker = KVConnectorFactory.create_connector(cfg, KVConnectorRole.WORKER, plan)
    raw = lc.aligned_buffer(32 * 2 * 128 * 16 * 2).view(torch.float16).view(2, 32, 128, 1, 16)
    worker.register_kv_caches({"model.layers.0.kv": (raw[0], raw[1])})
    sched = Scheduler(cfg, plan, StructuredOutputManager(cfg), block_size=bs, hash_block_size=hs)

    def run(items: list[tuple[str, list[int], int | None, int]], clear: bool = False) -> None:
        assert sched.reset_prefix_cache()
        raw.zero_()
        puts.clear()
        if clear:
            lc.MemoryBackend.payloads.clear()
        _run_batch(sched, worker, raw, [(_make_req(rid, toks, cap, hs), hit) for rid, toks, cap, hit in items])

    try:
        run([("zero", list(range(300)), 0, 0)])
        assert not lc.MemoryBackend.payloads
        run([("cap200", list(range(300)), 200, 0)])
        assert len(lc.MemoryBackend.payloads) == 1
        run([("uncap", list(range(1000, 1300)), None, 0)], clear=True)
        assert len(lc.MemoryBackend.payloads) == 2
        run([("zero-read", list(range(1000, 1450)), 0, 256)])
        assert puts == []
        mixed = [("m0", list(range(4000, 4300)), 0, 0), ("mc", list(range(5000, 5300)), 200, 0)]
        run(mixed + [("mu", list(range(6000, 6300)), None, 0), ("mr", list(range(1000, 1420)), 0, 256)])
        assert len(puts) == 3 and assert_scheduler_empty(sched) is None
        if not load_async:
            raw.fill_(7.0)
            b_hashes: list[kv_utils.BlockHash] = [b"\x01" * 16, b"\x02" * 16]  # type: ignore[list-item]
            b_puts = [md_mod.BoundaryPut(0, 1, 128), md_mod.BoundaryPut(0, 2, 256)]
            cw = worker.connector_worker
            pfx = cw.token_database._get_key_prefix(0)
            for cap, exp_idx in ((0, []), (128, [0]), (None, [0, 1])):
                lc.MemoryBackend.payloads.clear()
                bm = md_mod.ReqMeta("s", block_hashes=b_hashes, boundary_puts=b_puts, max_offload_tokens=cap)
                assert cw.kv_send_thread._store_boundary_puts(bm)
                exp_k = [pfx + md_mod.block_hash_to_str(b_hashes[i]) for i in exp_idx]
                assert sorted(lc.MemoryBackend.payloads) == exp_k
            cw.dspark_prefix_cache = (dsp := MagicMock())
            for cap, exp in ((0, None), (128, 128), (None, 256)):
                dsp.reset_mock()
                rm, md = md_mod.ReqMeta("d", max_offload_tokens=cap), md_mod.AscendConnectorMetadata(set())
                md.add_request(rm)
                worker.save_dspark_prefix(md, "d", 256, {0: [1, 2]})
                assert (dsp.save.call_args.args[1] if dsp.save.called else None) == exp
    finally:
        if sched.connector is not None:
            sched.connector.shutdown()
        worker.shutdown()


def test_boundary_puts_and_layerwise_offload_fallback(caplog: pytest.LogCaptureFixture) -> None:
    """Fine-grained Mamba/partial-tail boundary puts and layerwise_offload fallback honor offload limits."""
    mspec = MambaSpec(shapes=((4,),), dtypes=(torch.float32,), block_size=32, mamba_cache_mode="align")
    mamba = KVCacheGroupSpec(["model.layers.1.mamba"], mspec)
    cfg, plan = _make_cfg("mooncake", block_size=16, num_blocks=16, prefix_match_unit=8, extra_groups=[mamba])
    with patch(MOONCAKE_CLIENT, return_value=MagicMock()):
        sched = KVPoolScheduler(cfg, use_layerwise=False, kv_cache_config=plan)
    sched.bind_gpu_block_pool(BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=8))
    tokens, bids, full = list(range(28)), ([1, 2], [3]), [(0, 1, 16), (0, 2, 24), (1, 3, 16), (1, 4, 24)]
    for cap, exp_puts, exp_fin in ((0, [], False), (16, [(1, 3, 16)], False), (24, full, True), (None, full, True)):
        req = _make_req(f"b-{cap}", tokens, cap, hash_bs=8)
        rid = req.request_id
        sched.update_state_after_alloc(req, SimpleNamespace(get_block_ids=lambda: bids), 0)  # type: ignore[arg-type]
        out = SchedulerOutput.make_empty()
        out.preempted_req_ids, out.scheduled_new_reqs = set(), [NewRequestData.from_request(req, bids, tokens)]
        out.num_scheduled_tokens = {rid: 28}
        off = {rid: [(1, 3, 16), (1, 4, 24)]}
        out.kv_connector_block_state = KVConnectorBlockState({rid}, {rid: bids}.__getitem__, off)
        b_meta = sched.build_connector_meta(out)
        assert [tuple(p) for m in b_meta.requests if m.req_id == rid for p in (m.boundary_puts or ())] == exp_puts
        sched.register_finished_partial_tail(req, bids, [(1, 3, 24)])
        assert (rid in sched._finished_partial_tail_metas) is exp_fin
        sched._finished_partial_tail_metas.clear()

    lw_cfg, lw_ext = ("memcache", "kv_producer"), {"use_layerwise": True, lw_layout.LAYERWISE_REUSE_BACKING_BYTES: 4096}
    rw_cfg, rw_plan = _make_cfg(*lw_cfg, extra_config=lw_ext)
    with patch(MEMCACHE_CLIENT, return_value=MagicMock()):
        rw_sched = LayerwiseKVPoolScheduler(rw_cfg, use_layerwise=True, kv_cache_config=rw_plan)
    with caplog.at_level(logging.WARNING):
        assert rw_sched._parse_max_offload_tokens(_make_req("lw-reuse", list(range(32)), 0)) is None
    assert any("layerwise buffer reuse" in r.message for r in caplog.records)
