from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.config import ParallelConfig

from vllm_ascend.distributed.parallel_state import (
    _FLASHCOMM2_ODP,
    _FLASHCOMM2_OTP,
    _LMTP,
    _MC2,
    _MEGA_MOE,
    _OTP,
    _P_TP,
    _initialize_mega_moe_communicator,
    destroy_ascend_model_parallel,
    get_flashcomm2_odp_group,
    get_flashcomm2_otp_group,
    get_global_rank,
    get_lmhead_tp_group,
    get_mc2_group,
    get_mega_moe_group,
    get_otp_group,
    get_p_tp_group,
    init_ascend_model_parallel,
)
from vllm_ascend.utils import AscendDeviceType


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
        patch("vllm_ascend.distributed.parallel_state.get_tp_group") as mock_tp_group,
    ):
        mock_group.return_value.local_rank = 0
        mock_group.return_value.device_group = MagicMock()
        mock_tp_group.return_value.world_size = 4
        yield


@pytest.mark.parametrize(
    ("device_type", "expect_mega_moe_group"),
    [(AscendDeviceType.A5, True), (AscendDeviceType.A3, False)],
)
def test_init_ascend_model_parallel(mock_distributed, parallel_config, device_type, expect_mega_moe_group):
    mock_ascend_config = MagicMock()
    mock_ascend_config.finegrained_tp_config.lmhead_tensor_parallel_size = 2
    mock_ascend_config.finegrained_tp_config.oproj_tensor_parallel_size = 2
    mock_ascend_config.finegrained_tp_config.embedding_tensor_parallel_size = 2
    mock_ascend_config.finegrained_tp_config.mlp_tensor_parallel_size = 2
    mock_ascend_config.flashcomm2_oproj_tensor_parallel_size = 2
    mock_ascend_config.pd_tp_ratio = 2
    mock_ascend_config.num_head_replica = 0
    mock_ascend_config.pd_head_ratio = 2
    mock_ascend_config.enable_flashcomm2_parallel_size = 2
    mock_ascend_config.enable_fused_mc2 = 1
    mock_ascend_config.enable_context_parallel = False
    mock_vllm_config = MagicMock()
    mock_vllm_config.kv_transfer_config.is_kv_producer = True
    with (
        patch("vllm_ascend.distributed.parallel_state.model_parallel_initialized", return_value=False),
        patch("vllm_ascend.distributed.parallel_state.init_model_parallel_group") as mock_init_group,
        patch("vllm_ascend.distributed.parallel_state.get_current_vllm_config", return_value=mock_vllm_config),
        patch("vllm_ascend.distributed.parallel_state.get_ascend_config", return_value=mock_ascend_config),
        patch("vllm_ascend.distributed.parallel_state.get_ascend_device_type", return_value=device_type),
        patch("vllm_ascend.distributed.parallel_state._initialize_mega_moe_communicator") as mock_init_communicator,
        patch("vllm_ascend.utils.get_ascend_config", return_value=mock_ascend_config),
    ):
        init_ascend_model_parallel(parallel_config)

        mc2_group = get_mc2_group()
        lmheadtp_group = get_lmhead_tp_group()
        otp_group = get_otp_group()
        flashcomm2_otp_group = get_flashcomm2_otp_group()
        flashcomm2_odp_group = get_flashcomm2_odp_group()
        p_tp_group = get_p_tp_group()
        assert mc2_group is not None
        assert otp_group is not None
        assert flashcomm2_otp_group is not None
        assert flashcomm2_odp_group is not None
        assert lmheadtp_group is not None
        assert p_tp_group is not None
        mega_moe_group_created = any(
            call.kwargs.get("group_name") == "mega_moe" for call in mock_init_group.call_args_list
        )
        assert mega_moe_group_created is expect_mega_moe_group
        if expect_mega_moe_group:
            mega_moe_group = get_mega_moe_group()
            assert mega_moe_group is not None
            mock_init_communicator.assert_called_once_with(mega_moe_group)
        else:
            mock_init_communicator.assert_not_called()
            with pytest.raises(AssertionError, match="MegaMoE group is not initialized"):
                get_mega_moe_group()

        destroy_ascend_model_parallel()
        assert _MC2 is None
        assert _MEGA_MOE is None
        assert _LMTP is None
        assert _OTP is None
        assert _FLASHCOMM2_OTP is None
        assert _FLASHCOMM2_ODP is None
        assert _P_TP is None


def test_initialize_mega_moe_communicator():
    group = MagicMock()
    backend = group.device_group._get_backend.return_value
    backend.get_hccl_comm_name.return_value = "mega-moe-hccl-group"

    with patch("torch.distributed.get_rank", return_value=2) as mock_get_rank:
        _initialize_mega_moe_communicator(group)

    mock_get_rank.assert_called_once_with(group=group.device_group)
    backend.get_hccl_comm_name.assert_called_once_with(2)


def test_initialize_mega_moe_communicator_rejects_empty_group_name():
    group = MagicMock()
    backend = group.device_group._get_backend.return_value
    backend.get_hccl_comm_name.return_value = ""

    with (
        patch("torch.distributed.get_rank", return_value=0),
        pytest.raises(RuntimeError, match="Failed to initialize the MegaMoE HCCL communicator"),
    ):
        _initialize_mega_moe_communicator(group)


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
