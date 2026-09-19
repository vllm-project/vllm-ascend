# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Default graph coverage for speculative disaggregated decode."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.config import VllmConfig
from vllm.config.compilation import CompilationConfig, CUDAGraphMode
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

from vllm_ascend.platform import NPUPlatform, _get_default_max_cudagraph_capture_size


def make_config(seqs=256, drafts=3, role="kv_consumer", mode=CUDAGraphMode.FULL_DECODE_ONLY, eager=False):
    return SimpleNamespace(
        compilation_config=SimpleNamespace(
            max_cudagraph_capture_size=None,
            cudagraph_capture_sizes=None,
            cudagraph_mode=mode,
            pass_config=SimpleNamespace(enable_sp=False),
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=seqs),
        speculative_config=None if drafts is None else SimpleNamespace(num_speculative_tokens=drafts),
        kv_transfer_config=None if role is None else SimpleNamespace(kv_role=role),
        model_config=SimpleNamespace(enforce_eager=eager),
    )


@pytest.mark.parametrize("seqs,expected", [(1, 4), (128, 512), (129, 516), (200, 800), (256, 1024)])
def test_uniform_decode_capacity(seqs, expected):
    config = make_config(seqs=seqs)
    with patch("vllm_ascend.platform.logger.warning") as warning:
        NPUPlatform.apply_config_platform_defaults(config)
    assert config.compilation_config.max_cudagraph_capture_size == expected
    warning.assert_not_called()


@pytest.mark.parametrize("seqs,drafts,expected", [(256, 1, 512), (128, 7, 1024), (256, 7, 1024), (512, 3, 1024)])
def test_speculative_length_and_resource_cap(seqs, drafts, expected):
    config = make_config(seqs=seqs, drafts=drafts)
    with patch("vllm_ascend.platform.logger.warning") as warning:
        assert _get_default_max_cudagraph_capture_size(config) == expected
    if seqs * (drafts + 1) > expected:
        warning.assert_called_once()
        assert warning.call_args.args[1:] == (expected, seqs, drafts + 1, seqs * (drafts + 1))
    else:
        warning.assert_not_called()


@pytest.mark.parametrize("maximum,sizes", [(512, None), (768, None), (2048, None), (None, [4, 512]), (None, [])])
def test_explicit_settings_are_preserved(maximum, sizes):
    config = make_config(seqs=512)
    config.compilation_config.max_cudagraph_capture_size = maximum
    config.compilation_config.cudagraph_capture_sizes = sizes
    with patch("vllm_ascend.platform.logger.warning") as warning:
        NPUPlatform.apply_config_platform_defaults(config)
    assert config.compilation_config.max_cudagraph_capture_size == maximum
    assert config.compilation_config.cudagraph_capture_sizes == sizes
    warning.assert_not_called()


@pytest.mark.parametrize("role", [None, "kv_producer", "kv_both"])
def test_other_kv_roles_keep_existing_cap(role):
    assert _get_default_max_cudagraph_capture_size(make_config(role=role)) == 512


@pytest.mark.parametrize(
    "mode",
    [None, CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL, CUDAGraphMode.FULL_AND_PIECEWISE],
)
def test_other_graph_modes_keep_existing_cap(mode):
    assert _get_default_max_cudagraph_capture_size(make_config(mode=mode)) == 512


def test_eager_keeps_existing_cap():
    config = make_config(eager=True)
    with patch("vllm_ascend.platform.logger.warning") as warning:
        assert _get_default_max_cudagraph_capture_size(config) == 512
    warning.assert_not_called()


@pytest.mark.parametrize("drafts", [None, 0])
@pytest.mark.parametrize("seqs,expected", [(256, 256), (1024, 512)])
def test_non_speculative_decode_keeps_existing_default(drafts, seqs, expected):
    assert _get_default_max_cudagraph_capture_size(make_config(seqs=seqs, drafts=drafts)) == expected


@pytest.mark.parametrize("scheduler", [None, SimpleNamespace(max_num_seqs=None)])
def test_missing_scheduler_capacity_defers_to_upstream(scheduler):
    config = make_config()
    config.scheduler_config = scheduler
    assert _get_default_max_cudagraph_capture_size(config) is None


@pytest.mark.parametrize(
    "explicit_max,token_budget,expected_max,boundary_mode,capacity_mode",
    [
        (None, 32768, 1024, CUDAGraphMode.FULL, CUDAGraphMode.FULL),
        (512, 32768, 512, CUDAGraphMode.NONE, CUDAGraphMode.NONE),
        (None, 768, 768, CUDAGraphMode.FULL, CUDAGraphMode.NONE),
    ],
)
def test_final_sizes_and_uniform_decode_dispatch(
    explicit_max, token_budget, expected_max, boundary_mode, capacity_mode
):
    config = make_config()
    config.compilation_config = CompilationConfig(
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        max_cudagraph_capture_size=explicit_max,
    )
    config.scheduler_config.max_num_batched_tokens = token_budget
    config.parallel_config = SimpleNamespace(tensor_parallel_size=1)
    config.performance_mode = "throughput"
    config.num_speculative_tokens = 3
    config.uniform_decode_query_len = 4
    config.lora_config = None

    NPUPlatform.apply_config_platform_defaults(config)
    VllmConfig._set_cudagraph_sizes(config)
    assert config.compilation_config.max_cudagraph_capture_size == expected_max

    dispatcher = CudagraphDispatcher(config)
    dispatcher.initialize_cudagraph_keys(CUDAGraphMode.FULL_DECODE_ONLY, uniform_decode_query_len=4)
    assert dispatcher.dispatch(128 * 4, uniform_decode=True)[0] == CUDAGraphMode.FULL
    assert dispatcher.dispatch(129 * 4, uniform_decode=True)[0] == boundary_mode
    assert dispatcher.dispatch(256 * 4, uniform_decode=True)[0] == capacity_mode
