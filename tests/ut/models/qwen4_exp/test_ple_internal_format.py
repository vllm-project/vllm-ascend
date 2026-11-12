# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.models.qwen4_exp import ple


def test_ple_builder_scopes_generic_recoverssm_to_gdn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def base_init(
        self: object,
        kv_cache_spec: object,
        layer_names: list[str],
        vllm_config: object,
        device: torch.device,
    ) -> None:
        observed.update(
            kv_cache_spec=kv_cache_spec,
            layer_names=layer_names,
            vllm_config=vllm_config,
            device=device,
        )
        self.vllm_config = vllm_config  # type: ignore[attr-defined]

    monkeypatch.setattr(
        ple.PleShortConvAttentionMetadataBuilder,
        "__init__",
        base_init,
    )
    cache_config = SimpleNamespace(use_replayssm=True)
    vllm_config = SimpleNamespace(cache_config=cache_config)
    kv_cache_spec = object()
    device = torch.device("cpu")

    builder = ple.AscendQwen4ExpPLEMetadataBuilder(
        kv_cache_spec,  # type: ignore[arg-type]
        ["model.layers.0.ple"],
        vllm_config,  # type: ignore[arg-type]
        device,
    )

    assert cache_config.use_replayssm is True
    assert observed["vllm_config"] is not vllm_config
    assert observed["vllm_config"].cache_config is not cache_config
    assert observed["vllm_config"].cache_config.use_replayssm is False
    assert builder.vllm_config.cache_config.use_replayssm is False
    assert observed["kv_cache_spec"] is kv_cache_spec
    assert observed["layer_names"] == ["model.layers.0.ple"]
    assert observed["device"] == device
    assert ple.AscendQwen4ExpPLEBackend.get_builder_cls() is ple.AscendQwen4ExpPLEMetadataBuilder


def test_compact_prefill_output_matches_dense_reference() -> None:
    torch.manual_seed(0)
    source = torch.randn(3, 5, 4)
    conv_output = source.clone()
    q_starts = torch.tensor([0, 2, 5, 6], dtype=torch.int64)
    positions = torch.arange(6, dtype=torch.int64)
    req_indices = torch.searchsorted(q_starts[1:], positions, right=True)
    col_indices = positions - q_starts[req_indices]
    valid_state = torch.tensor([True, False, True])

    dense = F.silu(source).transpose(1, 2).contiguous()
    valid_tokens = torch.arange(4).view(1, 4) < torch.tensor([2, 3, 1]).view(3, 1)
    dense.masked_fill_(~(valid_tokens & valid_state.view(3, 1)).unsqueeze(-1), 0)
    expected = dense[req_indices, col_indices]
    output = torch.empty(6, 5)

    output.zero_()
    storage = conv_output.untyped_storage().data_ptr()
    for request_start in range(0, 3, 2):
        ple._accumulate_compact_prefill_output(
            conv_output[request_start : request_start + 2],
            output,
            req_indices,
            col_indices,
            valid_state,
            request_start,
        )

    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(conv_output, F.silu(source))
    assert conv_output.untyped_storage().data_ptr() == storage


def test_compact_prefill_path_matches_upstream_state_semantics() -> None:
    from vllm.models.qwen4_exp.amd.ple_layer import Qwen4ExpPLELayer

    torch.manual_seed(1)
    num_prefills = 2
    num_prefill_tokens = 5
    hidden_size = 4
    conv_state_len = 9
    metadata = SimpleNamespace(
        query_start_loc_p=torch.tensor([0, 2, 5], dtype=torch.int32),
        non_spec_query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        has_initial_states_p=torch.tensor([False, True]),
        max_prefill_query_len=3,
    )
    layer = SimpleNamespace(
        conv_state_len=conv_state_len,
        short_conv_dilation=3,
    )
    x_p = torch.randn(num_prefill_tokens, hidden_size)
    conv_weights = torch.randn(hidden_size, 4)
    state_indices = torch.tensor([0, 2], dtype=torch.int32)
    reference_state = torch.randn(4, hidden_size, conv_state_len)
    compact_state = reference_state.clone()

    original_attr = "_ascend_original_short_conv_dilated_prefill_batched"
    original = getattr(Qwen4ExpPLELayer, original_attr, None)
    if original is None:
        original = getattr(
            Qwen4ExpPLELayer,
            "_short_conv_dilated_prefill_batched",
            None,
        )
    if original is None:
        pytest.skip("upstream uses the fused PLE prefill dispatcher")
    assert original is not None
    expected = original(
        layer,
        x_p,
        metadata,
        reference_state,
        conv_weights,
        state_indices,
        num_prefills,
        0,
        num_prefill_tokens,
    )
    actual = ple._short_conv_dilated_prefill_batched(
        layer,
        x_p,
        metadata,
        compact_state,
        conv_weights,
        state_indices,
        num_prefills,
        0,
        num_prefill_tokens,
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(compact_state, reference_state)


def test_compact_prefill_path_bounds_convolution_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(2)
    num_prefills = 9
    hidden_size = 4
    query_len = 3
    num_prefill_tokens = num_prefills * query_len
    metadata = SimpleNamespace(
        query_start_loc_p=torch.arange(
            0,
            num_prefill_tokens + 1,
            query_len,
            dtype=torch.int32,
        ),
        non_spec_query_start_loc=torch.arange(
            0,
            num_prefill_tokens + 1,
            query_len,
            dtype=torch.int32,
        ),
        has_initial_states_p=torch.ones(num_prefills, dtype=torch.bool),
        max_prefill_query_len=query_len,
    )
    layer = SimpleNamespace(conv_state_len=9, short_conv_dilation=3)
    x_p = torch.randn(num_prefill_tokens, hidden_size)
    conv_weights = torch.randn(hidden_size, 4)
    conv_state = torch.randn(num_prefills, hidden_size, layer.conv_state_len)
    state_indices = torch.arange(num_prefills, dtype=torch.int32)
    batch_sizes: list[int] = []
    conv1d = F.conv1d

    def record_batch_size(*args: object, **kwargs: object) -> torch.Tensor:
        assert isinstance(args[0], torch.Tensor)
        batch_sizes.append(args[0].shape[0])
        return conv1d(*args, **kwargs)

    monkeypatch.setattr(ple.F, "conv1d", record_batch_size)
    ple._short_conv_dilated_prefill_batched(
        layer,
        x_p,
        metadata,
        conv_state,
        conv_weights,
        state_indices,
        num_prefills,
        0,
        num_prefill_tokens,
    )

    assert batch_sizes == [4, 4, 1]


def test_set_internal_format_prefers_high_level_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(allow_internal_format=True)
    low_level_calls: list[dict] = []
    monkeypatch.setattr(ple, "torch", SimpleNamespace(npu=SimpleNamespace(config=config)))
    monkeypatch.setattr(ple, "torch_npu", SimpleNamespace(_C=SimpleNamespace(_npu_setOption=low_level_calls.append)))

    ple._set_allow_internal_format(False)

    assert config.allow_internal_format is False
    assert low_level_calls == []


def test_set_internal_format_falls_back_to_low_level_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # torch_npu 2.10 (910B): torch.npu.config exists but carries no attribute.
    low_level_calls: list[dict] = []
    monkeypatch.setattr(ple, "torch", SimpleNamespace(npu=SimpleNamespace(config=SimpleNamespace())))
    monkeypatch.setattr(ple, "torch_npu", SimpleNamespace(_C=SimpleNamespace(_npu_setOption=low_level_calls.append)))

    ple._set_allow_internal_format(False)

    assert low_level_calls == [{"ALLOW_INTERNAL_FORMAT": "disable"}]


def test_set_internal_format_without_any_api_only_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ple, "torch", SimpleNamespace(npu=SimpleNamespace()))
    monkeypatch.setattr(ple, "torch_npu", SimpleNamespace(_C=SimpleNamespace()))

    ple._set_allow_internal_format(False)  # must not raise


class _DummyPLEMetadata:
    pass


def _install_ple_forward_context(monkeypatch: pytest.MonkeyPatch, attn_metadata: object) -> object:
    import vllm.forward_context as forward_context_module
    import vllm.model_executor.layers.mamba.mamba_utils as mamba_utils
    import vllm.v1.attention.backends.short_conv_attn as short_conv_attn

    monkeypatch.setattr(short_conv_attn, "PleShortConvAttentionMetadata", _DummyPLEMetadata)
    monkeypatch.setattr(
        forward_context_module,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=attn_metadata),
    )
    return mamba_utils


def _make_ple_layer(state: torch.Tensor, dispatch: object) -> SimpleNamespace:
    return SimpleNamespace(
        prefix="layers.1.ple",
        kv_cache=[state],
        conv_state_len=9,
        num_spec_tokens=3,
        conv1d=SimpleNamespace(weight=torch.randn(4, 1, 4)),
        _short_conv_dilated_dispatch=dispatch,
    )


def test_ple_wrap_dispatches_sd_state(monkeypatch: pytest.MonkeyPatch) -> None:
    updates: list[bool] = []
    monkeypatch.setattr(ple, "_get_allow_internal_format", lambda: True)
    monkeypatch.setattr(ple, "_set_allow_internal_format", updates.append)
    wait = Mock()
    monkeypatch.setattr(ple, "wait_for_kv_layer_from_connector", wait)
    original = Mock()
    calls: list[dict[str, Any]] = []
    # SD layout: [slots, window, channels]. The trailing capacity is kept.
    state = torch.randn(3, 14, 4)
    layer = _make_ple_layer(state, lambda **kwargs: calls.append(kwargs))
    metadata = _DummyPLEMetadata()
    mamba_utils = _install_ple_forward_context(monkeypatch, {layer.prefix: metadata})
    monkeypatch.setattr(mamba_utils, "is_conv_state_dim_first", lambda: False)
    inputs = torch.randn(2, 4, dtype=torch.bfloat16)
    residual = torch.randn(2, 4)
    outer_residual = torch.randn(2, 4)

    wrapped = ple._wrap_ple_short_conv(original)

    assert wrapped(layer, inputs, residual, outer_residual) is None
    original.assert_not_called()
    assert updates == [False, True]
    wait.assert_called_once_with(layer.prefix)
    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["inputs"] is inputs
    assert kwargs["residual"] is residual
    assert kwargs["outer_residual"] is outer_residual
    assert kwargs["metadata"] is metadata
    assert kwargs["conv_state"].shape == (3, 12, 4)
    assert kwargs["conv_weights"].shape == (4, 4)
    assert kwargs["conv_weights"].dtype == inputs.dtype


def test_ple_wrap_transposes_dim_first_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ple, "_get_allow_internal_format", lambda: True)
    monkeypatch.setattr(ple, "_set_allow_internal_format", lambda _enabled: None)
    monkeypatch.setattr(ple, "wait_for_kv_layer_from_connector", Mock())
    calls: list[dict[str, Any]] = []
    # DS layout: [slots, channels, window].
    state = torch.randn(3, 4, 14)
    layer = _make_ple_layer(state, lambda **kwargs: calls.append(kwargs))
    metadata = _DummyPLEMetadata()
    mamba_utils = _install_ple_forward_context(monkeypatch, {layer.prefix: metadata})
    monkeypatch.setattr(mamba_utils, "is_conv_state_dim_first", lambda: True)

    wrapped = ple._wrap_ple_short_conv(Mock())

    wrapped(layer, torch.randn(2, 4), torch.randn(2, 4), None)

    conv_state = calls[0]["conv_state"]
    assert conv_state.shape == (3, 12, 4)
    assert conv_state.untyped_storage().data_ptr() == state.untyped_storage().data_ptr()


def test_ple_wrap_is_a_noop_without_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    updates: list[bool] = []
    monkeypatch.setattr(ple, "_get_allow_internal_format", lambda: True)
    monkeypatch.setattr(ple, "_set_allow_internal_format", updates.append)
    monkeypatch.setattr(ple, "wait_for_kv_layer_from_connector", Mock())
    dispatch = Mock()
    layer = _make_ple_layer(torch.randn(3, 14, 4), dispatch)
    _install_ple_forward_context(monkeypatch, None)
    residual = torch.zeros(2, 4)
    outer_residual = torch.ones(2, 4)

    wrapped = ple._wrap_ple_short_conv(Mock())

    assert wrapped(layer, torch.randn(2, 4), residual, outer_residual) is None
    dispatch.assert_not_called()
    torch.testing.assert_close(residual, outer_residual)
    assert updates == [False, True]


def test_ple_internal_format_is_restored_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    updates: list[bool] = []
    monkeypatch.setattr(ple, "_get_allow_internal_format", lambda: True)
    monkeypatch.setattr(ple, "_set_allow_internal_format", updates.append)
    monkeypatch.setattr(ple, "wait_for_kv_layer_from_connector", Mock())
    layer = _make_ple_layer(torch.randn(3, 14, 4), Mock(side_effect=RuntimeError("expected short-conv failure")))
    metadata = _DummyPLEMetadata()
    mamba_utils = _install_ple_forward_context(monkeypatch, {layer.prefix: metadata})
    monkeypatch.setattr(mamba_utils, "is_conv_state_dim_first", lambda: False)
    wrapped = ple._wrap_ple_short_conv(Mock())

    with pytest.raises(RuntimeError, match="expected short-conv failure"):
        wrapped(layer, torch.randn(2, 4), torch.randn(2, 4), None)
    assert updates == [False, True]


def test_installing_ple_patch_is_idempotent() -> None:
    from vllm.models.qwen4_exp.amd.ple_layer import Qwen4ExpPLELayer

    original_attr = "_ascend_original_short_conv"
    initial_method = Qwen4ExpPLELayer._short_conv
    had_original = hasattr(Qwen4ExpPLELayer, original_attr)
    initial_original = getattr(Qwen4ExpPLELayer, original_attr, None)

    def original(_layer: object, inputs: torch.Tensor) -> torch.Tensor:
        return inputs

    try:
        if had_original:
            delattr(Qwen4ExpPLELayer, original_attr)
        Qwen4ExpPLELayer._short_conv = original
        ple.patch_upstream_ple_short_conv()
        installed = Qwen4ExpPLELayer._short_conv
        ple.patch_upstream_ple_short_conv()

        assert Qwen4ExpPLELayer._short_conv is installed
        assert getattr(Qwen4ExpPLELayer, original_attr) is original
    finally:
        Qwen4ExpPLELayer._short_conv = initial_method
        if had_original:
            setattr(Qwen4ExpPLELayer, original_attr, initial_original)
        elif hasattr(Qwen4ExpPLELayer, original_attr):
            delattr(Qwen4ExpPLELayer, original_attr)


def test_installing_ple_prefill_patch_is_idempotent() -> None:
    original_attr = "_ascend_original_short_conv_dilated_prefill_batched"

    def original(_layer: object, *_args: object) -> torch.Tensor:
        return torch.empty(0)

    class LegacyPLELayer:
        _short_conv_dilated_prefill_batched = original

    ple._patch_legacy_prefill(LegacyPLELayer)
    installed = LegacyPLELayer._short_conv_dilated_prefill_batched
    ple._patch_legacy_prefill(LegacyPLELayer)

    assert installed is ple._short_conv_dilated_prefill_batched
    assert LegacyPLELayer._short_conv_dilated_prefill_batched is installed
    assert getattr(LegacyPLELayer, original_attr) is original


def test_legacy_prefill_patch_skips_removed_upstream_method() -> None:
    class FusedPLELayer:
        pass

    ple._patch_legacy_prefill(FusedPLELayer)

    assert not hasattr(
        FusedPLELayer,
        "_ascend_original_short_conv_dilated_prefill_batched",
    )
    assert not hasattr(FusedPLELayer, "_short_conv_dilated_prefill_batched")
