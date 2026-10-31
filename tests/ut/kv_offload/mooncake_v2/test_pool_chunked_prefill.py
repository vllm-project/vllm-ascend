# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import CacheConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.outputs import ModelRunnerOutput

from vllm_ascend.ascend_config import SchedulerConfig as AscendSchedulerConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import pool_worker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.layerwise_pool_connector import LayerwisePoolConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pool_control import pool_control_request
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pool_metadata import (
    LayerwisePoolConnectorMetadata,
    PoolSaveRequest,
    PoolSourceProfile,
    PoolTransferMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.key_codec import PoolKeyIdentity

from .test_pool_object_adapter import memory_backend
from .test_pool_prefix_schedule import request as real_request
from .test_pool_prefix_schedule import scheduler_config
from .test_pool_scheduler import output, request, scheduler
from .test_pool_state_snapshot import kda


def worker(monkeypatch, backend, caches, spec=None, *, rank=0, tp=1, dcp=1):
    spec = spec or FullAttentionSpec(block_size=4, num_kv_heads=2, head_size=2, dtype=torch.float16)
    first = next(iter(caches.values()))
    first = first[0] if isinstance(first, tuple) else first
    config = SimpleNamespace(
        num_blocks=first.shape[0],
        kv_cache_groups=[SimpleNamespace(layer_names=list(caches), kv_cache_spec=spec)],
    )
    monkeypatch.setattr(
        pool_worker,
        "collect_configured_register_regions",
        lambda *args: SimpleNamespace(ptrs=[], lengths=[]),
    )
    result = pool_worker.LayerwisePoolWorker(
        backend,
        config,
        worker_id=f"pp0/pcp0/tp{rank}",
        tp_rank=rank,
        tp_size=tp,
        dcp_rank=rank % dcp,
        dcp_size=dcp,
        lease_ttl_ms=10000,
        total_num_kv_heads=2,
        base_layers=len(caches),
    )
    result.register_kv_caches(caches)
    result._test_caches = caches
    return result


@pytest.mark.parametrize("cancel_after", [None, 1, 2])
def test_chunks_commit_before_final_and_final_manifest_restores_whole_history(monkeypatch, cancel_after):
    backend, _, committed = memory_backend()
    source = {f"model.layers.{i}.attn": torch.full((4, 2, 4, 2), -1.0, dtype=torch.float16) for i in range(2)}
    p = worker(monkeypatch, backend, source)
    d = worker(monkeypatch, backend, {name: torch.full_like(tensor, -1) for name, tensor in source.items()})
    profile = PoolSourceProfile("source", "engine", PoolKeyIdentity("memcache-v1", "model", 1, 1))
    handshake = p.enable_manifest_control(profile, "127.0.0.1", retention_seconds=5)
    current = scheduler(client=pool_control_request)
    req = request()
    req.prompt_token_ids = req._all_token_ids = list(range(9))
    req.num_prompt_tokens = 9
    req.block_hashes = [b"a", b"b", b"c", b"d"]
    current.set_xfer_handshake_metadata({0: handshake})
    current.on_new_request(req)
    previous_end = 0
    try:
        for step, end in enumerate((3, 7, 9), 1):
            blocks = tuple(range(1, (end + 3) // 4 + 1))
            meta = current.build_connector_meta(output(count=end - previous_end, blocks=(blocks,)))
            save = meta.saves[req.request_id]
            assert save.chunk_start == previous_end and save.token_end == end
            assert save.is_final == (end == 9)
            p.prepare_save_request(req.request_id, save, handshake.profile)
            previous_full = {
                key
                for key, ref in p._prefill_progress[req.request_id].objects.items()
                if ref.object_id.startswith("pool-full-v1:")
            }
            assert not previous_full.intersection(p._writes[req.request_id][0].objects)
            for layer, (name, tensor) in enumerate(source.items()):
                for token in range(previous_end, end):
                    tensor[blocks[token // 4], :, token % 4].fill_(layer * 100 + token)
                p.save_kv_layer(name, MagicMock())
            p.wait_for_save()
            assert committed  # Even the first, incomplete page is already stored.
            state = current._source[req.request_id]
            query = {
                "op": "manifest",
                "incarnation": "engine",
                "transfer_id": state.transfer_id,
                "attempt_id": state.attempt_id,
            }
            if end < 9:
                with pytest.raises(RuntimeError, match="not committed"):
                    pool_control_request(handshake.control_endpoint, query)
                assert p.executor._readers  # Cross-chunk source protection is live.
                assert not p.take_write_reports()
            if step == cancel_after:
                released = LayerwisePoolConnectorMetadata()
                released.released_sources = {req.request_id}
                p.start_load_kv(released)
                assert not p._prefill_progress and not p._retained and not p.executor._readers
                return
            req.num_computed_tokens = end
            previous_end = end
        _, params = current.request_finished(req, (list(blocks),))
        transfer = PoolTransferMetadata.from_wire(params["pool_transfer"])
        assert transfer.token_start == 0 and transfer.token_end == 9
        d.restore_request("decode", transfer, (list(blocks),))
        d.reader._active["decode"].future.result(timeout=5)
        assert d.get_transfer_results().finished_recving == {"decode"}
        # Every real byte needed by all three chunks reached D.
        for name, target in d._test_caches.items():
            for token in range(9):
                assert torch.equal(
                    target[blocks[token // 4], :, token % 4], source[name][blocks[token // 4], :, token % 4]
                )
        pool_control_request(handshake.control_endpoint, query | {"op": "release"})
        assert not p._retained and not p.executor._readers
    finally:
        current.shutdown()
        p.shutdown()
        d.shutdown()


def test_cp_rank_without_tokens_in_an_early_chunk_still_joins_final_handoff(monkeypatch):
    backend, _, _ = memory_backend()
    name = "model.layers.0.attn"
    caches = [{name: torch.full((4, 2, 4, 2), float(rank + 1), dtype=torch.float16)} for rank in range(2)]
    workers = [worker(monkeypatch, backend, data, rank=rank, tp=2, dcp=2) for rank, data in enumerate(caches)]
    target = {name: torch.full((4, 2, 4, 2), -1.0, dtype=torch.float16)}
    d = worker(monkeypatch, backend, target)
    profiles = [
        PoolSourceProfile(
            f"p{rank}", f"engine{rank}", PoolKeyIdentity("memcache-v1", "model", 1, 1), group_block_sizes=(8,)
        )
        for rank in range(2)
    ]
    handshakes = [
        p.enable_manifest_control(profile, "127.0.0.1", retention_seconds=5)
        for p, profile in zip(workers, profiles, strict=True)
    ]
    previous = 0
    try:
        for end in (3, 7, 9):
            blocks = tuple(range(1, (end + 7) // 8 + 1))
            save = PoolSaveRequest(
                "transfer", "attempt", "input", 0, end, (blocks,), (("h0",),), chunk_start=previous, is_final=end == 9
            )
            for p, handshake in zip(workers, handshakes, strict=True):
                p.prepare_save_request("request", save, handshake.profile)
                p.save_kv_layer(name, MagicMock())
                p.wait_for_save()
            if end == 3:
                assert not workers[1]._prefill_progress["request"].objects
                assert not workers[1].take_write_reports()
            previous = end
        reports = [p.take_write_reports()[0] for p in workers]
        assert all(report.committed for report in reports)
        transfer = PoolTransferMetadata(
            "transfer",
            "attempt",
            "input",
            0,
            9,
            "attempt",
            tuple(h.profile for h in handshakes),
            tuple(ref for report in reports for ref in report.objects),
        )
        d.restore_request("decode", transfer, ([1, 2, 3],))
        d.reader._active["decode"].future.result(timeout=5)
        assert d.get_transfer_results().finished_recving == {"decode"}
        assert torch.all(target[name][1] == 1)
        assert torch.all(target[name][2] == 2)
        assert torch.all(target[name][3, :, 0] == 1)
    finally:
        for p in workers:
            p.shutdown()
        d.shutdown()


def test_actual_scheduler_emits_a_save_for_every_prefill_chunk(monkeypatch):
    """Use installed Scheduler and KVCacheManager, not supplied token counts."""
    from vllm.platforms import current_platform
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

    backend, _, _ = memory_backend()
    caches = {"model.layers.0.attn": torch.zeros((16, 2, 4, 2), dtype=torch.float16)}
    tensor = next(iter(caches.values()))
    spec = FullAttentionSpec(block_size=4, num_kv_heads=2, head_size=2, dtype=torch.float16)
    cache = KVCacheConfig(
        16,
        [
            KVCacheTensor(
                size=tensor.untyped_storage().nbytes(),
                layers=list(caches),
                layer_stride=tensor.untyped_storage().nbytes(),
                block_stride=tensor.stride(0) * tensor.element_size(),
            )
        ],
        [KVCacheGroupSpec(list(caches), spec)],
    )
    config = scheduler_config(CacheConfig(block_size=4, enable_prefix_caching=True))
    config.cache_config.num_gpu_blocks = 16
    config.scheduler_config.max_num_batched_tokens = 3
    config.scheduler_config.max_num_seqs = 1
    monkeypatch.setattr(current_platform, "register_custom_kv_cache_specs", lambda config: None)
    monkeypatch.setattr(
        "vllm_ascend.patch.platform.patch_balance_schedule.init_ascend_config",
        lambda config: SimpleNamespace(scheduler_config=AscendSchedulerConfig()),
    )
    register_all_kvcache_specs(config)
    init_none_hash(sha256)
    connector = LayerwisePoolConnector(config, KVConnectorRole.SCHEDULER, cache)
    monkeypatch.setattr("vllm.v1.core.sched.scheduler.KVConnectorFactory.create_connector", lambda **kwargs: connector)
    engine = Scheduler(config, cache, MagicMock(), block_size=4, hash_block_size=4)
    # The worker helper uses four blocks; replace that description with the
    # real sixteen-block registration used by the installed scheduler.
    p = worker(monkeypatch, backend, caches)
    p.cache_config = cache
    p.bindings = pool_worker.describe_cache_bindings(cache, caches)
    profile = PoolSourceProfile("source", "engine", PoolKeyIdentity("memcache-v1", "model", 1, 1))
    handshake = p.enable_manifest_control(profile, "127.0.0.1", retention_seconds=5)
    connector.set_xfer_handshake_metadata({0: handshake})
    req = real_request("request")
    engine.add_request(req)
    try:
        ends = []
        for step in range(3):
            scheduled = engine.schedule()
            assert scheduled.num_scheduled_tokens == {req.request_id: 3}
            save = scheduled.kv_connector_metadata.saves[req.request_id]
            ends.append(save.token_end)
            assert save.chunk_start == step * 3
            p.prepare_save_request(req.request_id, save, handshake.profile)
            p.save_kv_layer(next(iter(caches)), MagicMock())
            p.wait_for_save()
            if step < 2:
                # No sampled token on an unfinished prefill chunk.
                result = ModelRunnerOutput(
                    req_ids=[req.request_id],
                    req_id_to_index={req.request_id: 0},
                    sampled_token_ids=[[]],
                    logprobs=None,
                    prompt_logprobs_dict={},
                    pooler_output=None,
                )
                engine.update_from_output(scheduled, result)
        assert ends == [3, 6, 9]
        assert p.take_write_reports()[0].committed
    finally:
        p.shutdown()
        engine.shutdown()


@pytest.mark.parametrize("kind", ["kda", "tail"])
def test_only_final_state_checkpoint_enters_the_accumulated_handoff(monkeypatch, kind):
    from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec

    backend, _, committed = memory_backend()
    name = "model.layers.0.state"
    if kind == "kda":
        spec, tensors, _ = kda(2, 0, mode="align")
        caches = {name: tensors}
    else:
        spec = AscendIndexerKPoolTailSpec(
            block_size=7, num_kv_heads=1, head_size=2, dtype=torch.float32, sliding_window=4, compress_ratio=4
        )
        caches = {name: torch.full((4, 7, 4), -1.0)}
    targets = {
        name: tuple(torch.full_like(t, -1) for t in tensors) if kind == "kda" else torch.full_like(caches[name], -1)
    }
    p, d = worker(monkeypatch, backend, caches, spec), worker(monkeypatch, backend, targets, spec)
    profile = PoolSourceProfile("source", "engine", PoolKeyIdentity("memcache-v1", "model", 1, 1))
    handshake = p.enable_manifest_control(profile, "127.0.0.1", retention_seconds=5)
    try:
        previous = 0
        for end in (3, 7, 9):
            blocks = tuple(range(1, (end + 3) // 4 + 1)) if kind == "kda" else (1,)
            save = PoolSaveRequest(
                "transfer",
                "attempt",
                "input",
                0,
                end,
                (blocks,),
                (("h0", "h1"),) if kind == "kda" else ((),),
                chunk_start=previous,
                is_final=end == 9,
            )
            p.prepare_save_request("request", save, handshake.profile)
            for tensor in tensors if kind == "kda" else (caches[name],):
                tensor.fill_(end)
            p.save_kv_layer(name, MagicMock())
            p.wait_for_save()
            assert committed
            previous = end
        report = p.take_write_reports()[0]
        assert report.committed
        assert all(part.cache.token_end == 9 for ref in report.objects for part in ref.slices)
        final_keys = {part.key for ref in report.objects for part in ref.resolve(handshake.profile.key_identity)}
        assert set(p._retained["transfer", "attempt"]) == final_keys
        assert set(p.executor._readers) == final_keys
        transfer = PoolTransferMetadata(
            "transfer", "attempt", "input", 0, 9, "attempt", (handshake.profile,), report.objects
        )
        d.restore_request("decode", transfer, (list(blocks),))
        d.reader._active["decode"].future.result(timeout=5)
        assert d.get_transfer_results().finished_recving == {"decode"}
        for tensor in targets[name] if kind == "kda" else (targets[name],):
            assert torch.all(tensor[blocks[-1]] == 9)
    finally:
        p.shutdown()
        d.shutdown()


def test_failed_later_chunk_never_publishes_a_successful_handoff(monkeypatch):
    backend, _, committed = memory_backend()
    caches = {"model.layers.0.attn": torch.ones((4, 2, 4, 2), dtype=torch.float16)}
    p = worker(monkeypatch, backend, caches)
    profile = PoolSourceProfile("source", "engine", PoolKeyIdentity("memcache-v1", "model", 1, 1))
    handshake = p.enable_manifest_control(profile, "127.0.0.1", retention_seconds=5)
    name = next(iter(caches))
    try:
        first = PoolSaveRequest("transfer", "attempt", "input", 0, 3, ((1,),), ((),), is_final=False)
        p.prepare_save_request("request", first, handshake.profile)
        p.save_kv_layer(name, MagicMock())
        p.wait_for_save()
        earlier = set(committed)
        backend.batch_copy_gva.side_effect = lambda *args: -7
        later = PoolSaveRequest(
            "transfer", "attempt", "input", 0, 7, ((1, 2),), (("h0",),), chunk_start=3, is_final=False
        )
        p.prepare_save_request("request", later, handshake.profile)
        p.save_kv_layer(name, MagicMock())
        p.wait_for_save()
        assert committed == earlier
        report = p.take_write_reports()[0]
        assert not report.committed and report.error
        final = PoolSaveRequest("transfer", "attempt", "input", 0, 9, ((1, 2, 3),), (("h0", "h1"),), chunk_start=7)
        p.prepare_save_request("request", final, handshake.profile)
        p.wait_for_save()
        wire = pool_control_request(
            handshake.control_endpoint,
            {"op": "manifest", "incarnation": "engine", "transfer_id": "transfer", "attempt_id": "attempt"},
        )
        assert not wire["committed"] and not wire["objects"]["rows"]
        released = LayerwisePoolConnectorMetadata()
        released.released_sources = {"request"}
        p.start_load_kv(released)
        assert not p._retained and not p.executor._readers
    finally:
        p.shutdown()
