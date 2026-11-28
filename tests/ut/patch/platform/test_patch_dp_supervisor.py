# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace

import pytest
from vllm.entrypoints.launchers import dp_supervisor


@pytest.fixture
def supervisor_args():
    # tests/ut/conftest.py registers the Ascend platform patches before collection.
    return Namespace(
        port=8000,
        data_parallel_start_rank=2,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=2,
        device_ids=None,
    )


@pytest.mark.parametrize(
    "pcp_size,expected_shards",
    [(1, [[0, 1], [2, 3]]), (2, [[0, 1, 2, 3], [4, 5, 6, 7]])],
)
def test_default_device_ids_are_env_relative(supervisor_args, monkeypatch, pcp_size, expected_shards):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "7,6,5,4,3,2,1,0")
    supervisor_args.prefill_context_parallel_size = pcp_size

    for local_rank, expected in enumerate(expected_shards):
        child = dp_supervisor._build_vllm_dp_server_args(supervisor_args, local_rank)
        assert child.device_ids == expected


def test_explicit_device_ids_preserve_order(supervisor_args):
    supervisor_args.tensor_parallel_size = 1
    supervisor_args.pipeline_parallel_size = 2
    supervisor_args.device_ids = [6, 2, 7, 4, 1, 5, 3, 0]

    assert dp_supervisor._build_vllm_dp_server_args(supervisor_args, 0).device_ids == [6, 2, 7, 4]
    assert dp_supervisor._build_vllm_dp_server_args(supervisor_args, 1).device_ids == [1, 5, 3, 0]


def test_short_device_list_rejects_partial_pcp_shard(supervisor_args):
    supervisor_args.device_ids = list(range(7))

    with pytest.raises(ValueError, match=r"DP rank 1 needs devices \[4, 8\)"):
        dp_supervisor._build_vllm_dp_server_args(supervisor_args, 1)
