from collections import Counter
from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import MooncakeConnectorWorker


def _worker(rank):
    worker = object.__new__(MooncakeConnectorWorker)
    worker.use_mla = True
    worker.use_sparse = False
    worker._is_hma_required = True
    worker.use_hybrid = True
    worker.tp_size = worker.dcp_size = 16
    worker.tp_rank = worker.dcp_rank = rank
    worker.pcp_size = 1
    worker.pcp_rank = 0
    worker._prefill_tp_size = 16  # Deliberately stale: request metadata is authoritative.
    worker._prefill_pp_size = 1
    worker.num_key_value_heads = 96
    worker.tp_num_need_pulls = 1
    worker.side_channel_port = 60000
    worker.handshake_port = 60000 + rank
    worker.block_size = 384
    worker.block_size_scale = [[3], [1]]
    worker.kv_group2layeridx = {
        0: ({"kv_cache_spec_type": "AscendMLAAttentionSpec"}, [0]),
        1: ({"kv_cache_spec_type": "MambaSpec"}, [1]),
    }
    worker.local_remote_block_port_mapping = {}
    worker.remote_port_send_num = {}
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(is_deepseek_mla=True),
        kv_transfer_config=SimpleNamespace(kv_port=60000),
    )
    return worker


@pytest.mark.parametrize("prefill_tp", [16, 32])
def test_hybrid_mla_dcp_pulls_all_state_shards_and_counts_completions(prefill_tp):
    workers = [_worker(rank) for rank in range(16)]
    # Reuse the same engine metadata cache while the final shard changes.
    for prompt_blocks in [1, 17, 33, 63]:
        meta = SimpleNamespace(
            remote_ptp_size=prefill_tp, remote_pcp_size=1, remote_dcp_size=prefill_tp,
            remote_port=50012, remote_block_size=384, num_prompt_blocks=prompt_blocks,
            num_external_tokens=prompt_blocks * 384 - 1, num_computed_tokens=0,
            local_block_ids=(list(range(100, 100 + (prompt_blocks + 15) // 16)), [200]),
            remote_block_ids=(list(range(10, 10 + (prompt_blocks + prefill_tp - 1) // prefill_tp)), [20]),
            remote_engine_id="same_prefill_engine", remote_host="80.5.17.106",
            remote_multi_nodes_meta_mapping={str(i): {"host": "80.5.17.106" if i < 16 else "80.5.17.107",
                                                     "handshake_port": 50012 + i} for i in range(prefill_tp)},
        )
        all_ports = []
        for worker in workers:
            ports, local_ids, remote_ids = worker._get_kv_split_metadata("req", meta)
            pulls = worker._get_group_pulls_metadata("req", ports, prefill_tp, 50012, 1, prefill_tp)
            state_sources = []
            attention_blocks = []
            for shard_idx, shard_ports in enumerate(ports):
                for port_idx, port in enumerate(shard_ports):
                    all_ports.append(port)
                    for pull in pulls[shard_idx][port_idx]:
                        if pull.group_id == 0:
                            assert len(local_ids[shard_idx][0]) == len(remote_ids[shard_idx][0])
                            attention_blocks.extend(
                                (port - 50012, local_id, remote_id)
                                for local_id, remote_id in zip(local_ids[shard_idx][0], remote_ids[shard_idx][0])
                            )
                        if pull.group_id == 1:
                            assert pull.is_group_transfer_end == (pull.remote_tp_offset == pull.num_group_pulls - 1)
                            assert local_ids[shard_idx][1] == [200]
                            assert remote_ids[shard_idx][1] == [20]
                            state_sources.append((port - 50012, pull.remote_tp_offset, pull.num_group_pulls))
            # Independent oracle: global logical blocks are distributed round
            # robin; each C384 logical block expands to three C128 kernel blocks.
            expected_blocks = [
                (block % prefill_tp, (100 + block // 16) * 3 + offset,
                 (10 + block // prefill_tp) * 3 + offset)
                for block in range(worker.dcp_rank, prompt_blocks, 16)
                for offset in range(3)
            ]
            assert sorted(attention_blocks) == sorted(expected_blocks)
            ratio = prefill_tp // 16
            assert sorted(state_sources) == [(worker.tp_rank * ratio + i, i, ratio) for i in range(ratio)]
        expected = Counter(all_ports)
        for worker in workers:
            info = worker.remote_port_send_num[meta.remote_engine_id]
            assert {p: v["num"] for p, v in info.items()} == dict(expected)
