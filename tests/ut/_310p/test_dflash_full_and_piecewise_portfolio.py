# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

from vllm_ascend._310p.dflash_full_and_piecewise import (
    apply_dflash_full_and_piecewise_capture_config,
    get_310p_dflash_graph_capabilities,
    initialize_dflash_full_and_piecewise_cudagraph_keys,
    is_310p_dflash_effective_full,
    is_310p_dflash_effective_piecewise,
    is_310p_dflash_full_and_piecewise,
)
from vllm_ascend.ascend_config import AscendCompilationConfig

PORTFOLIO_KEY = "dflash_full_and_piecewise_capture_config"


def _config(
    *,
    mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    method="dflash",
    piecewise=32,
    full=80,
    k=7,
    max_num_seqs=10,
    max_num_batched_tokens=3584,
    capture_sizes=None,
):
    portfolio = {
        "piecewise_capture_size": piecewise,
        "full_capture_size": full,
    }
    return SimpleNamespace(
        speculative_config=(
            SimpleNamespace(method=method, num_speculative_tokens=k)
            if method is not None
            else None
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=mode,
            cudagraph_capture_sizes=capture_sizes,
            max_cudagraph_capture_size=(
                max(capture_sizes) if capture_sizes else None
            ),
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        additional_config={
            "ascend_compilation_config": {PORTFOLIO_KEY: portfolio}
        },
    )


class _FakeDispatcher:
    def __init__(self, vllm_config):
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.uniform_decode_query_len = (
            1 + vllm_config.speculative_config.num_speculative_tokens
        )
        self.cudagraph_mode = CUDAGraphMode.NONE
        self.cudagraph_keys = {
            CUDAGraphMode.PIECEWISE: set(),
            CUDAGraphMode.FULL: set(),
        }
        self.keys_initialized = False
        self.specialize_lora_count = False

    def _compute_bs_to_padded_graph_size(self):
        sizes = self.compilation_config.cudagraph_capture_sizes
        max_size = self.compilation_config.max_cudagraph_capture_size
        self._bs_to_padded_graph_size = [0] * (max_size + 1)
        for value in range(max_size + 1):
            self._bs_to_padded_graph_size[value] = next(
                (size for size in sizes if size >= value),
                max_size,
            )

    @staticmethod
    def _get_lora_cases():
        return [0]

    def _create_padded_batch_descriptor(
        self,
        num_tokens,
        uniform_decode,
        has_lora,
        num_active_loras=0,
    ):
        padded = self._bs_to_padded_graph_size[num_tokens]
        num_reqs = (
            padded // self.uniform_decode_query_len
            if uniform_decode
            else min(
                padded,
                self.vllm_config.scheduler_config.max_num_seqs,
            )
        )
        return BatchDescriptor(
            num_tokens=padded,
            num_reqs=num_reqs,
            uniform=uniform_decode,
            has_lora=has_lora,
            num_active_loras=num_active_loras,
        )

    def add_cudagraph_key(self, runtime_mode, descriptor):
        self.cudagraph_keys[runtime_mode].add(descriptor)


def _sizes(dispatcher, mode):
    return {desc.num_tokens for desc in dispatcher.cudagraph_keys[mode]}


def test_config_parses_one_piecewise_and_one_full_capacity():
    config = AscendCompilationConfig(
        **{
            PORTFOLIO_KEY: {
                "piecewise_capture_size": 32,
                "full_capture_size": 80,
            }
        }
    )

    assert getattr(config, PORTFOLIO_KEY) == {
        "piecewise_capture_size": 32,
        "full_capture_size": 80,
    }


def test_absent_config_does_not_add_a_new_ascend_config_field():
    config = AscendCompilationConfig()

    assert not hasattr(config, PORTFOLIO_KEY)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"piecewise_capture_size": [32, 64], "full_capture_size": 80},
         "only supports one PIECEWISE"),
        ({"piecewise_capture_size": 32, "full_capture_size": [80, 96]},
         "only supports one FULL"),
        ({"piecewise_capture_size": 0, "full_capture_size": 80},
         "positive integer"),
        ({"piecewise_capture_size": 32}, "requires exactly"),
        ({"piecewise_capture_size": 32, "full_capture_size": 80, "extra": 1},
         "requires exactly"),
    ],
)
def test_config_rejects_unvalidated_portfolios(raw, message):
    with pytest.raises(ValueError, match=message):
        AscendCompilationConfig(**{PORTFOLIO_KEY: raw})


def test_absent_config_preserves_existing_capture_sizes():
    config = _config(capture_sizes=[16, 32, 80])
    del config.additional_config["ascend_compilation_config"][PORTFOLIO_KEY]

    assert not apply_dflash_full_and_piecewise_capture_config(config)
    assert config.compilation_config.cudagraph_capture_sizes == [16, 32, 80]


def test_absent_config_disables_private_hybrid_capability():
    config = _config()
    del config.additional_config["ascend_compilation_config"][PORTFOLIO_KEY]

    with patch(
        "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
        return_value=True,
    ):
        assert not is_310p_dflash_full_and_piecewise(config)
        assert not get_310p_dflash_graph_capabilities(config).any
        assert not is_310p_dflash_effective_full(
            config,
            CUDAGraphMode.FULL,
        )
        assert not is_310p_dflash_effective_piecewise(
            config,
            CUDAGraphMode.PIECEWISE,
        )


@pytest.mark.parametrize(
    ("platform_310p", "method", "mode"),
    [
        (False, "dflash", CUDAGraphMode.FULL_AND_PIECEWISE),
        (True, "mtp", CUDAGraphMode.FULL_AND_PIECEWISE),
        (True, None, CUDAGraphMode.FULL_AND_PIECEWISE),
        (True, "dflash", CUDAGraphMode.PIECEWISE),
        (True, "dflash", CUDAGraphMode.FULL_DECODE_ONLY),
    ],
)
def test_explicit_config_does_not_mutate_other_scopes(
    platform_310p,
    method,
    mode,
):
    config = _config(
        method=method,
        mode=mode,
        capture_sizes=[16, 24],
    )

    with patch(
        "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
        return_value=platform_310p,
    ):
        assert not apply_dflash_full_and_piecewise_capture_config(config)

    assert config.compilation_config.cudagraph_capture_sizes == [16, 24]


def test_platform_planner_forms_descriptor_union_without_ownership():
    config = _config(capture_sizes=None)

    with patch(
        "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
        return_value=True,
    ):
        assert apply_dflash_full_and_piecewise_capture_config(config)

    assert config.compilation_config.cudagraph_capture_sizes == [32, 80]
    assert config.compilation_config.max_cudagraph_capture_size == 80


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"full": 82}, "divisible"),
        ({"full": 88}, "logical deployment bound"),
        ({"piecewise": 4096}, "max_num_batched_tokens"),
    ],
)
def test_platform_planner_rejects_unsafe_capacity_contracts(kwargs, message):
    config = _config(**kwargs)

    with (
        patch(
            "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
            return_value=True,
        ),
        pytest.raises(ValueError, match=message),
    ):
        apply_dflash_full_and_piecewise_capture_config(config)


def test_target_inventory_has_strict_mode_ownership():
    config = _config(capture_sizes=[32, 80])
    dispatcher = _FakeDispatcher(config)

    with patch(
        "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
        return_value=True,
    ):
        handled = initialize_dflash_full_and_piecewise_cudagraph_keys(
            dispatcher,
            CUDAGraphMode.FULL_AND_PIECEWISE,
            uniform_decode_query_len=8,
        )

    assert handled
    assert _sizes(dispatcher, CUDAGraphMode.PIECEWISE) == {32}
    assert _sizes(dispatcher, CUDAGraphMode.FULL) == {80}


def test_existing_dispatcher_routes_full_piecewise_and_safe_fallback():
    config = _config(capture_sizes=[32, 80])
    dispatcher = _FakeDispatcher(config)

    with patch(
        "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
        return_value=True,
    ):
        initialize_dflash_full_and_piecewise_cudagraph_keys(
            dispatcher,
            CUDAGraphMode.FULL_AND_PIECEWISE,
            uniform_decode_query_len=8,
        )

    dispatch = CudagraphDispatcher.dispatch.__get__(dispatcher)

    mode, descriptor = dispatch(80, uniform_decode=True)
    assert mode == CUDAGraphMode.FULL
    assert descriptor.num_tokens == 80

    mode, descriptor = dispatch(32, uniform_decode=False)
    assert mode == CUDAGraphMode.PIECEWISE
    assert descriptor.num_tokens == 32

    # A uniform workload below the FULL bucket can safely use the configured
    # PIECEWISE bucket without adding FULL32 to the capture inventory.
    mode, descriptor = dispatch(16, uniform_decode=True)
    assert mode == CUDAGraphMode.PIECEWISE
    assert descriptor.num_tokens == 32

    mode, descriptor = dispatch(40, uniform_decode=False)
    assert mode == CUDAGraphMode.NONE
    assert descriptor.num_tokens == 40

    mode, descriptor = dispatch(88, uniform_decode=True)
    assert mode == CUDAGraphMode.NONE
    assert descriptor.num_tokens == 88


def test_draft_outer_dispatcher_keeps_only_piecewise_capacity():
    config = _config(capture_sizes=[32, 80])
    dispatcher = _FakeDispatcher(config)

    with patch(
        "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
        return_value=True,
    ):
        handled = initialize_dflash_full_and_piecewise_cudagraph_keys(
            dispatcher,
            CUDAGraphMode.PIECEWISE,
            uniform_decode_query_len=8,
        )

    assert handled
    assert _sizes(dispatcher, CUDAGraphMode.PIECEWISE) == {32}
    assert _sizes(dispatcher, CUDAGraphMode.FULL) == set()


def test_initializer_defers_to_upstream_without_explicit_portfolio():
    config = _config(capture_sizes=[32, 80])
    del config.additional_config["ascend_compilation_config"][PORTFOLIO_KEY]
    dispatcher = _FakeDispatcher(config)

    with patch(
        "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
        return_value=True,
    ):
        assert not initialize_dflash_full_and_piecewise_cudagraph_keys(
            dispatcher,
            CUDAGraphMode.FULL_AND_PIECEWISE,
            uniform_decode_query_len=8,
        )

    assert not dispatcher.keys_initialized
