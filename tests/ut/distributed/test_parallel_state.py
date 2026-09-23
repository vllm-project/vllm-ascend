from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import ParallelConfig

from vllm_ascend.distributed.parallel_state import (
    _LMTP,
    _MC2,
    _OTP,
    _P_TP,
    destroy_ascend_model_parallel,
    get_global_rank,
    get_lmhead_tp_group,
    get_mc2_group,
    get_otp_group,
    get_p_tp_group,
    init_ascend_model_parallel,
)


@pytest.fixture
def parallel_config():
    return ParallelConfig(
        data_parallel_size=2,
        tensor_parallel_size=4,
        pipeline_parallel_size=2,
    )


@pytest.fixture
def mock_distributed():
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=16),
        patch("torch.distributed.get_backend", return_value="nccl"),
        patch("vllm_ascend.distributed.parallel_state.get_world_group") as mock_group,
    ):
        mock_group.return_value.local_rank = 0
        mock_group.return_value.device_group = MagicMock()
        yield


def test_init_ascend_model_parallel(mock_distributed, parallel_config):
    mock_ascend_config = MagicMock()
    mock_ascend_config.kvpp_config.size = 1
    mock_ascend_config.finegrained_tp_config.lmhead_tensor_parallel_size = 2
    mock_ascend_config.finegrained_tp_config.oproj_tensor_parallel_size = 2
    mock_ascend_config.finegrained_tp_config.embedding_tensor_parallel_size = 2
    mock_ascend_config.finegrained_tp_config.mlp_tensor_parallel_size = 2
    mock_ascend_config.pd_tp_ratio = 2
    mock_ascend_config.num_head_replica = 0
    mock_ascend_config.pd_head_ratio = 2
    mock_ascend_config.enable_context_parallel = False
    mock_vllm_config = MagicMock()
    mock_vllm_config.kv_transfer_config.is_kv_producer = True
    with (
        patch("vllm_ascend.distributed.parallel_state.model_parallel_initialized", return_value=False),
        patch("vllm_ascend.distributed.parallel_state.init_model_parallel_group"),
        patch("vllm_ascend.distributed.parallel_state.get_current_vllm_config", return_value=mock_vllm_config),
        patch("vllm_ascend.distributed.parallel_state.get_ascend_config", return_value=mock_ascend_config),
        patch("vllm_ascend.utils.get_ascend_config", return_value=mock_ascend_config),
    ):
        init_ascend_model_parallel(parallel_config)

        mc2_group = get_mc2_group()
        lmheadtp_group = get_lmhead_tp_group()
        otp_group = get_otp_group()
        p_tp_group = get_p_tp_group()
        assert mc2_group is not None
        assert otp_group is not None
        assert lmheadtp_group is not None
        assert p_tp_group is not None

        destroy_ascend_model_parallel()
        assert _MC2 is None
        assert _LMTP is None
        assert _OTP is None
        assert _P_TP is None


def _build_parallel_config(
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    prefill_context_parallel_size=1,
    data_parallel_index=0,
):
    return SimpleNamespace(
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        prefill_context_parallel_size=prefill_context_parallel_size,
        data_parallel_index=data_parallel_index,
    )


@pytest.mark.parametrize(
    "parallel_config_kwargs, rank_in_group, expected",
    [
        # No parallelism at all (single card): replica_size == 1.
        (dict(tensor_parallel_size=1), 0, 0),
        # TP only: rank_in_group is the local rank within the single replica.
        (dict(tensor_parallel_size=4), 0, 0),
        (dict(tensor_parallel_size=4), 3, 3),
        # Dense DP: world group spans one replica, rank_in_group is local and
        # data_parallel_index supplies the DP offset.
        (dict(tensor_parallel_size=4, data_parallel_index=0), 2, 2),
        (dict(tensor_parallel_size=4, data_parallel_index=1), 2, 6),
        # MoE DP / external_launcher: world group spans all DP ranks, so
        # rank_in_group is already global; the modulo strips the DP offset and
        # data_parallel_index re-adds it (result equals rank_in_group).
        (dict(tensor_parallel_size=4, data_parallel_index=1), 6, 6),
        (dict(tensor_parallel_size=4, data_parallel_index=1), 7, 7),
        # TP * PP * prefill-CP all contribute to replica_size; DCP/EP do not.
        (dict(tensor_parallel_size=2, pipeline_parallel_size=2, data_parallel_index=1), 1, 5),
        (
            dict(
                tensor_parallel_size=2, pipeline_parallel_size=2, prefill_context_parallel_size=2, data_parallel_index=1
            ),
            3,
            11,
        ),
    ],
)
def test_get_global_rank(parallel_config_kwargs, rank_in_group, expected):
    parallel_config = _build_parallel_config(**parallel_config_kwargs)
    with patch("vllm_ascend.distributed.parallel_state.get_world_group") as mock_group:
        mock_group.return_value.rank_in_group = rank_in_group
        assert get_global_rank(parallel_config) == expected


def test_get_global_rank_defaults_to_current_config():
    parallel_config = _build_parallel_config(tensor_parallel_size=4, data_parallel_index=1)
    mock_vllm_config = MagicMock()
    mock_vllm_config.parallel_config = parallel_config
    with (
        patch(
            "vllm_ascend.distributed.parallel_state.get_current_vllm_config",
            return_value=mock_vllm_config,
        ),
        patch("vllm_ascend.distributed.parallel_state.get_world_group") as mock_group,
    ):
        mock_group.return_value.rank_in_group = 3
        # data_parallel_index(1) * replica_size(4) + 3 == 7
        assert get_global_rank() == 7


@pytest.mark.parametrize("size", [1, 4])
@pytest.mark.parametrize("pcp_size", [1, 2])
def test_kvpp_group_stays_inside_pipeline_stage(monkeypatch, size, pcp_size):
    from vllm_ascend.distributed import parallel_state

    for name in ("_KVPP", "_MC2", "_P_TP", "_OTP", "_LMTP", "_EMBED_TP", "_MLP_TP", "_DYNAMIC_EPLB"):
        monkeypatch.setattr(parallel_state, name, None)
    config = SimpleNamespace(
        kvpp_config=SimpleNamespace(size=size),
        pd_tp_ratio=1,
        pd_head_ratio=1,
        eplb_config=SimpleNamespace(dynamic_eplb=False),
        finegrained_tp_config=SimpleNamespace(
            oproj_tensor_parallel_size=0,
            lmhead_tensor_parallel_size=0,
            embedding_tensor_parallel_size=0,
            mlp_tensor_parallel_size=0,
        ),
    )
    calls, groups = {}, {}

    def init_group(ranks, local_rank, backend, *, group_name):
        calls[group_name] = (ranks, local_rank, backend)
        groups[group_name] = MagicMock()
        return groups[group_name]

    monkeypatch.setattr(parallel_state.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_state.torch.distributed, "get_world_size", lambda: 8)
    monkeypatch.setattr(parallel_state.torch.distributed, "get_backend", lambda _: "hccl")
    monkeypatch.setattr(parallel_state, "get_world_group", lambda: SimpleNamespace(local_rank=4, device_group=object()))
    monkeypatch.setattr(parallel_state, "get_ascend_config", lambda: config)
    monkeypatch.setattr(parallel_state, "init_model_parallel_group", init_group)
    parallel_state.init_ascend_model_parallel(
        SimpleNamespace(
            tensor_parallel_size=4 // pcp_size,
            pipeline_parallel_size=2,
            data_parallel_size=1,
            prefill_context_parallel_size=pcp_size,
        )
    )
    if size == 1:
        assert "kvpp" not in calls
        assert parallel_state._KVPP is None
    else:
        assert calls["kvpp"] == ([[0, 1, 2, 3], [4, 5, 6, 7]], 4, "hccl")
        assert parallel_state.get_kvpp_group() is groups["kvpp"]
        assert groups["kvpp"] is not groups["mc2"]
    parallel_state.destroy_ascend_model_parallel()
    if size > 1:
        groups["kvpp"].destroy.assert_called_once_with()
    assert parallel_state._KVPP is None


@pytest.mark.parametrize(
    "world_size,data_parallel_size,replica_size,node_count,placement,expected",
    [
        # Single A3, TP4/DP4: one node, so all four local replicas shard.
        (16, 4, 4, 1, "contiguous", 4),
        # Dual A3, global TP8/DP4: two whole replicas per node, 16 ranks each.
        (32, 4, 8, 2, "contiguous", 2),
        # Dual A3, global TP4/DP8: four replicas per node.
        (32, 8, 4, 2, "contiguous", 4),
        # Four A2, global TP8/DP4: one replica per node, nothing to shard.
        (32, 4, 8, 4, "contiguous", 1),
        # Dual A2, global TP4/DP4.
        (16, 4, 4, 2, "contiguous", 2),
        # Nodes cannot be filled with whole replicas.
        (24, 4, 8, 2, "contiguous", 1),
        # Replicas are spread across the nodes instead of grouped by node.
        (32, 4, 8, 2, "interleaved", 1),
    ],
)
def test_engram_dp_shard_size(
    monkeypatch, world_size, data_parallel_size, replica_size, node_count, placement, expected
):
    from vllm_ascend.distributed import parallel_state

    ranks_per_node = world_size // node_count

    def same_node_as(pg, source_rank):
        if placement == "interleaved":
            return [rank % node_count == source_rank % node_count for rank in range(world_size)]
        start = source_rank // ranks_per_node * ranks_per_node
        return [start <= rank < start + ranks_per_node for rank in range(world_size)]

    monkeypatch.setattr(parallel_state, "get_node_count", lambda: node_count)
    monkeypatch.setattr(parallel_state, "in_the_same_node_as", same_node_as)
    monkeypatch.setattr(parallel_state, "get_world_group", lambda: SimpleNamespace(cpu_group=object()))
    assert parallel_state._engram_dp_shard_size(world_size, data_parallel_size, replica_size) == expected


def test_engram_dp_group_ranks_stay_inside_one_node():
    from vllm_ascend.distributed import parallel_state

    # Dual A3, global TP8/DP4: global ranks 0-15 are on node 0, 16-31 on node 1.
    all_ranks = torch.arange(32).reshape(-1, 4, 1, 1, 8)
    groups = parallel_state._engram_dp_group_ranks(all_ranks, 2)
    assert groups[:2] == [[0, 8], [16, 24]]
    assert len(groups) == 16
    assert sorted(rank for ranks in groups for rank in ranks) == list(range(32))
    for ranks in groups:
        assert max(ranks) - min(ranks) < 16


@pytest.mark.parametrize("upstream", [False, True])
def test_engram_dp_group_created_for_cluster_wide_dp(monkeypatch, upstream):
    from vllm_ascend.distributed import parallel_state

    for name in (
        "_KVPP",
        "_MC2",
        "_P_TP",
        "_OTP",
        "_LMTP",
        "_EMBED_TP",
        "_MLP_TP",
        "_DYNAMIC_EPLB",
        "_ENGRAM_DP",
    ):
        monkeypatch.setattr(parallel_state, name, None)
    ascend_config = SimpleNamespace(
        kvpp_config=SimpleNamespace(size=1),
        pd_tp_ratio=1,
        pd_head_ratio=1,
        eplb_config=SimpleNamespace(dynamic_eplb=False),
        finegrained_tp_config=SimpleNamespace(
            oproj_tensor_parallel_size=0,
            lmhead_tensor_parallel_size=0,
            embedding_tensor_parallel_size=0,
            mlp_tensor_parallel_size=0,
        ),
    )
    vllm_config = SimpleNamespace(
        engram_config=SimpleNamespace(dp_shared_memory=True),
        model_config=SimpleNamespace(architecture="DeepseekV41ForCausalLM"),
    )
    calls, groups = {}, {}

    def init_group(ranks, local_rank, backend, *, group_name):
        calls[group_name] = ranks
        groups[group_name] = MagicMock()
        return groups[group_name]

    monkeypatch.setattr(parallel_state.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_state.torch.distributed, "get_world_size", lambda: 32)
    monkeypatch.setattr(parallel_state.torch.distributed, "get_backend", lambda _: "hccl")
    monkeypatch.setattr(
        parallel_state,
        "get_world_group",
        lambda: SimpleNamespace(local_rank=0, device_group=object(), cpu_group=object()),
    )
    monkeypatch.setattr(parallel_state, "get_ascend_config", lambda: ascend_config)
    monkeypatch.setattr(parallel_state, "init_model_parallel_group", init_group)
    monkeypatch.setattr(parallel_state, "get_current_vllm_config_or_none", lambda: vllm_config)
    monkeypatch.setattr(parallel_state, "get_node_count", lambda: 2)
    upstream_group = MagicMock()
    monkeypatch.setattr(parallel_state, "_UPSTREAM_ENGRAM_DP_GETTER", (lambda: upstream_group) if upstream else None)
    monkeypatch.setattr(
        parallel_state, "in_the_same_node_as", lambda pg, rank: [ranks // 16 == rank // 16 for ranks in range(32)]
    )
    config = SimpleNamespace(
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        data_parallel_size=4,
        enable_elastic_ep=False,
    )
    # Per TP position the global DP runs [tp, tp+8, tp+16, tp+24] split into the
    # two node-local pairs of that node.
    expected_edp = [[tp + 8 * half, tp + 8 * half + 8] for tp in range(8) for half in (0, 2)]
    for _ in range(2):
        parallel_state.init_ascend_model_parallel(config)
        if upstream:
            assert "edp" not in calls
            assert parallel_state.get_engram_dp_group() is upstream_group
        else:
            assert calls["edp"] == expected_edp
            assert parallel_state.get_engram_dp_group() is groups["edp"]
        parallel_state.destroy_ascend_model_parallel()
        if upstream:
            upstream_group.destroy.assert_not_called()
        else:
            groups["edp"].destroy.assert_called_once_with()
        assert parallel_state._ENGRAM_DP is None
