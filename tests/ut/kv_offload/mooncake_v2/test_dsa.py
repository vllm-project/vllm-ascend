# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPU integration of shared Main destinations with the real V2 planner."""

import ctypes
import queue
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_dsa
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import base_worker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakePCPTransferMetadata,
    MooncakeTPTransferMetadata,
    MooncakeTransferMetadata,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa import (
    MooncakeDsaRecvingThread,
    MooncakeDsaWorker,
    validate_dsa_config,
    validate_shared_host_views,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_transfer import (
    TransferRegionIndex,
    bound_register_regions,
    split_transfer_ranges,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_host_pool import (
    HostMemoryRegion,
    HostPoolTopology,
    MooncakeHostPool,
)
from vllm_ascend.worker.v2.attn_utils import _reshape_combined_mla_kv_cache

from .helpers import (
    make_kv_cache_tensor,
    make_metadata_groups,
    make_pp_metadata,
    make_sfa_indexer_spec,
    make_transfer_metadata,
)

MAIN = "model.layers.0.self_attn"
INDEXER = MAIN + ".indexer"
DRAFT = "mtp.layers.0.self_attn"
BLOCK = 4
CAPACITY = 32


def main_spec(*, host=False, block_size=BLOCK):
    return AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=12,
        dtype=torch.bfloat16,
        store_on_host=host,
    )


def make_host_pool():
    sizes = [CAPACITY * BLOCK * dim * 2 for dim in (8, 4)]
    pool = MooncakeHostPool(
        HostMemoryRegion(torch.zeros(sum(sizes) + 128, dtype=torch.int8)),
        HostPoolTopology(tp_rank=0, tp_size=1),
    )
    pair = tuple(
        raw.view(torch.bfloat16).reshape(CAPACITY, BLOCK, 1, dim)
        for raw, dim in zip(pool.allocate_tensors(sizes, 1), (8, 4), strict=True)
    )
    return pool, pair


def make_worker(monkeypatch, rank, pool, pair, *, producer=False):
    specs = [main_spec(host=not producer), make_sfa_indexer_spec(BLOCK, 1), main_spec()]
    names = [MAIN, INDEXER, DRAFT]
    caches = {
        MAIN: pair if producer else [torch.empty((1, BLOCK, 1, 8), dtype=torch.bfloat16)] * 6,
        INDEXER: [torch.zeros((CAPACITY, BLOCK, 1, 8), dtype=torch.float16)],
        DRAFT: [torch.zeros((CAPACITY, BLOCK, 1, dim), dtype=torch.bfloat16) for dim in (8, 4)],
    }
    config = KVCacheConfig(
        num_blocks=CAPACITY,
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec) for name, spec in zip(names, specs, strict=True)
        ],
        kv_cache_tensors=[
            make_kv_cache_tensor(
                size=CAPACITY * spec.page_size_bytes,
                layers=[name],
                layer_stride=CAPACITY * spec.page_size_bytes,
                block_stride=spec.page_size_bytes,
            )
            for name, spec in zip(names, specs, strict=True)
        ],
    )
    manager = SimpleNamespace(
        offload_layer_names=[MAIN],
        kv_cache_config=config,
        tp_size=1,
        get_mooncake_host_pool=lambda: pool,
        get_local_host_kv_views=lambda name: pair,
    )
    monkeypatch.setattr(mooncake_dsa, "get_sparse_kv_offload_manager", lambda: manager)
    monkeypatch.setattr(mooncake_dsa, "global_te", MagicMock())
    monkeypatch.setattr(base_worker, "global_te", MagicMock())
    worker = MooncakeDsaWorker.__new__(MooncakeDsaWorker)
    worker.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=BLOCK))
    worker.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
    worker.kv_transfer_config = SimpleNamespace(is_kv_consumer=not producer, is_kv_producer=producer)
    worker.kv_cache_config = config
    worker.tp_rank, worker.tp_size, worker.pcp_rank = rank, 2, 0
    worker.engine_id, worker.te_rpc_port = "engine-d", 9000
    worker.block_size, worker.side_channel_host, worker.handshake_port = BLOCK, "127.0.0.1", 5000 + rank
    worker._recving_thread = None
    monkeypatch.setattr(
        worker,
        "_build_recving_thread",
        MagicMock(
            side_effect=lambda cls, event, **kwargs: MagicMock(
                start=MagicMock(side_effect=event.set), is_alive=MagicMock(return_value=True)
            )
        ),
    )
    return worker, caches, manager


@pytest.mark.parametrize("rank", [0, 1])
def test_registration_only_tp0_receives_main_each_rank_keeps_indexer_and_draft(monkeypatch, rank):
    pool, pair = make_host_pool()
    worker, caches, _ = make_worker(monkeypatch, rank, pool, pair)
    original_config = worker.kv_cache_config
    worker.register_kv_caches(caches)
    metadata = worker.xfer_handshake_metadata
    register = mooncake_dsa.global_te.register_buffer
    assert isinstance(register, MagicMock)
    register.assert_called_once_with(
        [ptr for ptr, _ in metadata.registered_regions],
        [length for _, length in metadata.registered_regions],
    )
    assert metadata.layer_names == ([MAIN, INDEXER, DRAFT] if rank == 0 else [INDEXER, DRAFT])
    assert worker.kv_cache_config is original_config
    assert original_config.kv_cache_groups[0].layer_names == [MAIN]
    assert metadata.group_indices == ([0, 1, 2] if rank == 0 else [1, 2])
    assert metadata.kv_caches_base_addr[-1] == [tensor.data_ptr() for tensor in caches[DRAFT]]
    assert metadata.kv_caches_base_addr[-2] == [caches[INDEXER][0].data_ptr()]
    if rank == 0:
        assert metadata.kv_caches_base_addr[0] == [tensor.data_ptr() for tensor in pair]
    for ptr, length in metadata.registered_regions:
        assert ptr + length <= pool.data_ptr or ptr >= pool.data_ptr + pool.nbytes
    assert worker._build_recving_thread.call_args.args[0] is MooncakeDsaRecvingThread
    assert worker._build_recving_thread.call_args.kwargs["direct_host_regions"] == (
        [(pool.data_ptr, pool.nbytes)] if rank == 0 else []
    )


def test_source_keeps_components_separate_and_publishes_physical_mrs(monkeypatch):
    pool, pair = make_host_pool()
    worker, caches, _ = make_worker(monkeypatch, 0, pool, pair, producer=True)
    worker.register_kv_caches(caches)
    metadata = worker.xfer_handshake_metadata
    assert metadata.cache_dtypes[0] == ["torch.bfloat16", "torch.bfloat16"]
    assert metadata.kv_caches_base_addr[0] == [tensor.data_ptr() for tensor in pair]
    assert metadata.registered_regions
    assert worker._recving_thread is None


@pytest.mark.parametrize("remote_tp,remote_pcp", [(8, 1), (4, 2), (8, 2), (2, 4), (1, 8)])
@pytest.mark.parametrize("producer_pp", [1, 2])
@pytest.mark.parametrize("computed", [0, 5, 32])
@pytest.mark.parametrize("transfer_mode", ["batch", "request"])
def test_real_v2_mapper_reassembles_p_dcp8_into_tp0_shared_host(
    monkeypatch, remote_tp, remote_pcp, producer_pp, computed, transfer_mode
):
    """Reconstruct real bytes with the inherited planner, buckets and executor.

    Only the hardware READ is replaced with CPU memmove. Source fixtures retain
    their storage while PP/PCP/DCP and prefix offsets are mapped by V2 itself.
    """
    pool, host_pair = make_host_pool()
    for tensor in host_pair:
        tensor.fill_(-1)
    source_owners: list[torch.Tensor] = []
    pcp_metadata = {}
    token_count = 96
    remote_blocks, source_scale = 4, 8
    source_slot_elements = BLOCK * (8 + 4) + 16
    for pcp in range(remote_pcp):
        ranks = {}
        for tp in range(remote_tp):
            shard = (tp * remote_pcp + pcp) % 8
            # Use the actual allocator's page-strided FIA layout, including
            # padding and both components sharing the same physical backing.
            raw = torch.empty(remote_blocks * source_scale * source_slot_elements, dtype=torch.bfloat16)
            pair = _reshape_combined_mla_kv_cache(
                raw,
                (remote_blocks * source_scale, BLOCK, 1, 8),
                (remote_blocks * source_scale, BLOCK, 1, 4),
                torch.bfloat16,
                source_slot_elements * 2 * source_scale,
                source_scale,
            )
            for tensor in pair:
                for token in range(tensor.shape[0] * BLOCK):
                    global_token = (token // BLOCK * 8 + shard) * BLOCK + token % BLOCK
                    tensor[token // BLOCK, token % BLOCK].fill_(global_token)
            source_owners.extend((raw, *pair))
            ranks[tp] = MooncakeTPTransferMetadata(
                te_rpc_port=9000 + tp,
                local_ip=f"p{pcp}",
                handshake_port=5000 + tp,
                layer_indices=[0],
                kv_caches_base_addr=[[tensor.data_ptr() for tensor in pair]],
                registered_regions=[(raw.data_ptr(), raw.nbytes)],
            )
        pcp_metadata[pcp] = MooncakePCPTransferMetadata(metadata_by_tp_rank=ranks)
    pp = replace(
        make_pp_metadata(
            layer_names=[MAIN],
            layer_block_sizes=[BLOCK * source_scale],
            block_size_scales=[[source_scale, source_scale]],
            block_lens=[[BLOCK * dim * 2 for dim in (8, 4)]],
            block_strides=[[source_slot_elements * 2] * 2],
            block_shapes=[[(BLOCK, 1, dim) for dim in (8, 4)]],
        ),
        num_blocks=remote_blocks,
        metadata_by_pcp_rank=pcp_metadata,
        cp_kv_cache_interleave_size=BLOCK,
        cache_dtypes=[["torch.bfloat16"] * 2],
    )
    remote = replace(make_metadata_groups(tp_size=remote_tp, pcp_size=remote_pcp, pp_metadata=pp), dcp_size=8)
    if producer_pp > 1:
        remote = replace(
            remote,
            pp_size=producer_pp,
            metadata_by_pp_rank={
                0: replace(pp, layer_names=["model.layers.99.self_attn"]),
                1: pp,
            },
        )
    metadata = replace(
        make_transfer_metadata(
            layer_names=[MAIN],
            layer_block_sizes=[BLOCK],
            base_addrs=[[t.data_ptr() for t in host_pair]],
            block_lens=[[BLOCK * dim * 2 for dim in (8, 4)]],
            block_strides=[[BLOCK * dim * 2 for dim in (8, 4)]],
            block_shapes=[[(BLOCK, 1, dim) for dim in (8, 4)]],
            block_size_scales=[[1, 1]],
        ),
        num_blocks=CAPACITY,
        cache_dtypes=[["torch.bfloat16"] * 2],
        cp_kv_cache_interleave_size=BLOCK,
    )
    config = KVCacheConfig(
        num_blocks=CAPACITY,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[MAIN], kv_cache_spec=main_spec(host=True))],
    )
    engine = MagicMock()

    def read(session, local_addrs, remote_addrs, lengths):
        for local, source, length in zip(local_addrs, remote_addrs, lengths, strict=True):
            ctypes.memmove(local, source, length)
        return 0

    engine.batch_transfer_sync_read.side_effect = read
    thread = MooncakeDsaRecvingThread(
        engine=engine,
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(is_deepseek_mla=True, hf_text_config=SimpleNamespace()),
            speculative_config=None,
            parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=BLOCK),
            kv_transfer_config=SimpleNamespace(
                kv_connector_extra_config={"pull_transfer_mode": transfer_mode},
                get_from_extra_config=lambda key, default: {"pull_transfer_mode": transfer_mode}.get(key, default),
            ),
        ),
        kv_cache_config=config,
        kv_cache_specs=[main_spec(host=True)],
        layer_name_to_group_index={MAIN: 0},
        layer_name_to_spec_index={MAIN: 0},
        local_metadata=metadata,
        tp_rank=0,
        tp_size=2,
        pp_rank=0,
        pp_size=1,
        dp_rank=0,
        dp_size=1,
        pcp_rank=0,
        pcp_size=1,
        dcp_rank=0,
        dcp_size=1,
        device=0,
        ready_event=threading.Event(),
        direct_host_regions=[(pool.data_ptr, pool.nbytes)],
    )
    groups, pairs = thread._build_remote_transfer_layout(remote)
    thread.remote_tp_rank_groups["engine-p"] = groups
    thread.remote_layer_index_pairs["engine-p"] = pairs
    monkeypatch.setattr(thread, "_get_remote_metadata", lambda *args: remote)
    request = ReqMeta(
        local_block_ids=([*range(computed // BLOCK, token_count // BLOCK)],),
        local_full_block_ids=([*range(token_count // BLOCK)],),
        remote_block_ids=([*range(remote_blocks)],),
        local_num_prompt_tokens=token_count + 1,
        remote_num_prompt_tokens=token_count,
        num_computed_tokens=computed,
        num_external_tokens=token_count - computed,
        remote_host="p",
        remote_port=6000,
        remote_engine_id="engine-p",
        remote_request_id="p-request",
    )
    try:
        thread.request_queue.put(("engine-p", {"request": request}))
        thread._process_next()
        assert thread.finished_requests.get(timeout=5) == ("request", False)
        assert thread.request_queue.unfinished_tasks == 0
    finally:
        thread.executor.shutdown(wait=True)
    for tensor in host_pair:
        rows = tensor.reshape(-1, tensor.shape[-1])
        start = computed // BLOCK * BLOCK
        assert torch.all(rows[:start] == -1)
        assert torch.all(rows[token_count:] == -1)
        actual = rows[start:token_count]
        expected = torch.arange(start, token_count, dtype=torch.bfloat16).unsqueeze(1).expand_as(actual)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected_calls = min(8, (token_count - computed // BLOCK * BLOCK + BLOCK - 1) // BLOCK)
    assert engine.batch_transfer_sync_read.call_count == expected_calls
    assert (
        sum(sum(call.args[3]) for call in engine.batch_transfer_sync_read.call_args_list)
        == (token_count - computed // BLOCK * BLOCK) * 12 * 2
    )


def test_physical_range_split_validates_before_any_read():
    assert bound_register_regions([100], [25], max_region_bytes=10) == ([100, 110, 120], [10, 10, 5])
    local = TransferRegionIndex([(100, 6), (106, 6)])
    remote = TransferRegionIndex([(200, 4), (204, 8)])
    assert split_transfer_ranges([100], [200], [12], local_regions=local, remote_regions=remote) == (
        [100, 104, 106],
        [200, 204, 206],
        [4, 2, 6],
    )
    with pytest.raises(ValueError, match="outside"):
        split_transfer_ranges([100], [200], [13], local_regions=local, remote_regions=remote)


def test_shared_offsets_compare_local_mappings_not_virtual_addresses(monkeypatch):
    pool, pair = make_host_pool()
    descriptor = pool.describe_local_views([(MAIN, i, tensor) for i, tensor in enumerate(pair)], CAPACITY)
    manager = MagicMock(
        offload_layer_names=[MAIN],
        kv_cache_config=SimpleNamespace(num_blocks=CAPACITY),
        tp_size=2,
        tp_group=SimpleNamespace(cpu_group="tp"),
        get_local_host_kv_views=lambda name: pair,
    )

    def gather(output, value, group):
        assert group == "tp"
        output[:] = [value, (descriptor, None)]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    validate_shared_host_views(manager, pool)

    def mismatch(output, value, group):
        output[:] = [value, ((1, *descriptor[1:]), None)]

    monkeypatch.setattr(torch.distributed, "all_gather_object", mismatch)
    with pytest.raises(ValueError, match="layout mismatch"):
        validate_shared_host_views(manager, pool)


@pytest.mark.parametrize("side", ["local", "local_full", "remote"])
@pytest.mark.parametrize("transfer_mode", ["batch", "request"])
def test_rejects_out_of_capacity_blocks_before_read(monkeypatch, side, transfer_mode):
    thread = MooncakeDsaRecvingThread.__new__(MooncakeDsaRecvingThread)
    thread.local_metadata = make_transfer_metadata()
    remote = make_metadata_groups()
    monkeypatch.setattr(thread, "_get_remote_metadata", lambda *args: remote)
    request = SimpleNamespace(
        local_block_ids=([32] if side == "local" else [0],),
        local_full_block_ids=([32] if side == "local_full" else [0],),
        remote_block_ids=([32] if side == "remote" else [0],),
        remote_host="p",
        remote_port=6000,
    )
    thread.engine = MagicMock()
    thread.executor = MagicMock()
    thread.can_report_invalid_block_ids = False
    thread.request_queue = queue.Queue()
    thread.finished_requests = queue.SimpleQueue()
    thread.request_queue.put(("engine-p", {"request": request}))
    process_next = thread._process_next_request if transfer_mode == "request" else thread._process_next_batch
    process_next()
    assert thread.finished_requests.get(timeout=5) == ("request", True)
    assert thread.finished_requests.empty()
    assert thread.request_queue.unfinished_tasks == 0
    thread.executor.submit.assert_not_called()
    thread.engine.batch_transfer_sync_read.assert_not_called()


def test_failed_physical_range_validation_submits_no_partial_read():
    thread = MooncakeDsaRecvingThread.__new__(MooncakeDsaRecvingThread)
    thread._local_regions = TransferRegionIndex([(100, 8)])
    thread.engine = MagicMock()
    peer = MooncakeTPTransferMetadata(
        te_rpc_port=9000,
        layer_indices=[0],
        kv_caches_base_addr=[[200]],
        local_ip="p",
        handshake_port=1,
        registered_regions=[(200, 8)],
    )
    with pytest.raises(ValueError, match="outside"):
        thread._read_transfer_bucket("p:9000", [100, 104], [200, 204], [4, 8], peer)
    thread.engine.batch_transfer_sync_read.assert_not_called()


@pytest.mark.parametrize("producer", [False, True])
@pytest.mark.parametrize("invalid", ["layerwise", "role", "offload", "policy", "dcp", "pcp"])
def test_configuration_rejected_before_threads_start(monkeypatch, producer, invalid):
    transfer = SimpleNamespace(
        is_kv_producer=producer,
        is_kv_consumer=not producer,
        kv_load_failure_policy="fail",
        kv_connector_extra_config={},
    )
    sparse = SimpleNamespace(enabled=not producer, host_backend="mooncake", use_fused_overlap=True)
    parallel = SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1)
    if invalid == "layerwise":
        transfer.kv_connector_extra_config["use_layerwise"] = True
    elif invalid == "role":
        transfer.is_kv_consumer = transfer.is_kv_producer
    elif invalid == "offload":
        sparse.enabled = producer
    elif invalid == "policy":
        transfer.kv_load_failure_policy = "recompute"
    elif invalid == "dcp":
        parallel.decode_context_parallel_size = 8
    elif invalid == "pcp":
        parallel.prefill_context_parallel_size = 2
    monkeypatch.setattr(
        mooncake_dsa, "init_ascend_config", lambda config: SimpleNamespace(sparse_kv_offload_config=sparse)
    )
    config = SimpleNamespace(kv_transfer_config=transfer, parallel_config=parallel)
    if producer and invalid in ("policy", "dcp", "pcp"):
        validate_dsa_config(config)
    else:
        with pytest.raises(ValueError):
            validate_dsa_config(config)


def test_physical_metadata_survives_pp_scheduler_aggregation():
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler import MooncakeMetadataThread

    scheduler = MooncakeMetadataThread.__new__(MooncakeMetadataThread)
    scheduler.engine_id, scheduler.tp_size, scheduler.pp_size, scheduler.pcp_size = "engine-p", 2, 2, 1
    scheduler.use_kv_pp = False
    workers: dict[int | tuple[int, ...], MooncakeTransferMetadata] = {}
    for pp in range(2):
        for tp in range(2):
            workers[(pp, 0, tp)] = replace(
                make_transfer_metadata(layer_names=[f"model.layers.{pp}.self_attn"]),
                registered_regions=[(1000 + 100 * tp, 64)],
                cache_dtypes=[["torch.bfloat16"]],
            )
    merged = scheduler._merge_metadata_by_pp_rank(workers)
    assert set(merged) == {0, 1}
    for merged_pp in merged.values():
        assert merged_pp.cache_dtypes == [["torch.bfloat16"]]
        for tp, peer in merged_pp.metadata_by_pcp_rank[0].metadata_by_tp_rank.items():
            assert peer.registered_regions == [(1000 + 100 * tp, 64)]
    workers[(0, 0, 1)] = replace(workers[(0, 0, 1)], cache_dtypes=[["torch.float16"]])
    with pytest.raises(ValueError, match="cache_dtypes"):
        scheduler._merge_metadata_by_pp_rank(workers)


def test_dtype_mismatch_rejected_before_read():
    thread = MooncakeDsaRecvingThread.__new__(MooncakeDsaRecvingThread)
    thread.tp_size, thread.layer_names = 1, [MAIN]
    thread.local_metadata = replace(make_transfer_metadata(), cache_dtypes=[["torch.bfloat16"]])
    remote = make_metadata_groups(pp_metadata=replace(make_pp_metadata(), cache_dtypes=[["torch.float16"]]))
    with pytest.raises(ValueError, match="dtype mismatch"):
        thread._build_remote_transfer_layout(remote)


@pytest.mark.parametrize("failed_rank", [None, 0, 1])
def test_core_aggregation_waits_for_tp0_main_before_scheduler_ack(failed_rank):
    from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
    from vllm.v1.outputs import KVConnectorOutput

    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_scheduler import MooncakePullConnectorScheduler
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa import MooncakeDsaConnector

    connector = MooncakeDsaConnector.__new__(MooncakeDsaConnector)
    aggregate = KVOutputAggregator.from_connector(connector, world_size=2)
    scheduler = MooncakePullConnectorScheduler.__new__(MooncakePullConnectorScheduler)
    scheduler._heartbeat_thread = scheduler._sending_thread = None
    scheduler._recving_thread = MagicMock()
    scheduler._reqs_recv_info = {"request": ("p", 6000, "p-request")}
    scheduler.kv_transfer_config = SimpleNamespace(kv_load_failure_policy="fail")

    def output(rank, finished):
        return MagicMock(
            kv_connector_output=KVConnectorOutput(
                finished_recving={"request"} if finished else None,
                failed_recving={"request"} if finished and failed_rank == rank else set(),
            )
        )

    early = aggregate.aggregate([output(0, False), output(1, True)])
    assert early.kv_connector_output.finished_recving is None
    assert not early.kv_connector_output.failed_recving
    scheduler.update_connector_output(early.kv_connector_output)
    scheduler._recving_thread.add_request.assert_not_called()
    final = aggregate.aggregate([output(0, True), output(1, False)])
    assert final.kv_connector_output.finished_recving == {"request"}
    assert final.kv_connector_output.failed_recving == (set() if failed_rank is None else {"request"})
    scheduler.update_connector_output(final.kv_connector_output)
    scheduler._recving_thread.add_request.assert_called_once_with("p", 6000, "p-request")
