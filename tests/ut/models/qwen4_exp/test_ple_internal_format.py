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
    checkpoint = None


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


@pytest.mark.parametrize(
    "mode,history,dtype,model_alignment,grid,expected",
    [
        ("align", 9, torch.bfloat16, 64, 128, 128),
        ("align", 9, torch.float32, 64, 128, 128),
        ("align", 9, torch.float16, 64, 128, None),
        ("align", 9, torch.bfloat16, None, 128, None),
        ("align", 9, torch.bfloat16, 1, 8, None),
        ("align", 0, torch.bfloat16, 64, 128, None),
        ("all", 9, torch.bfloat16, 64, 128, None),
    ],
)
def test_ple_checkpoint_spec(monkeypatch, mode, history, dtype, model_alignment, grid, expected):
    spec = ple.MambaSpec(shapes=((12, 4),), dtypes=(dtype,), block_size=128)
    layer = ple.AscendQwen4ExpPLELayer.__new__(ple.AscendQwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer.cache_config = SimpleNamespace(mamba_cache_mode=mode)
    layer.model_config = SimpleNamespace(dtype=torch.bfloat16)
    layer.conv_state_len = history
    monkeypatch.setattr(ple, "get_ascend_config", lambda: SimpleNamespace(gdn_internal_prefill_checkpoints=True))
    monkeypatch.setattr(ple, "get_pp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(layer, "get_state_dtype", lambda: (dtype,))
    monkeypatch.setattr(ple.Qwen4ExpPLELayer, "get_kv_cache_spec", lambda self, config: spec)
    monkeypatch.setattr(ple, "get_mamba_prefill_checkpoint_alignment", lambda config: model_alignment)
    monkeypatch.setattr(ple, "get_mamba_prefill_checkpoint_grid", lambda config: grid)

    actual = layer.get_kv_cache_spec(SimpleNamespace())
    if expected is None:
        assert actual is spec
    else:
        assert actual.num_prefill_checkpoint_blocks == 1
        assert actual.prefill_checkpoint_alignment == expected
        assert actual.shapes == spec.shapes


@pytest.mark.parametrize("opt_in,pp_size,expected", [(True, 1, 1), (False, 1, None), (True, 2, None)])
def test_ple_checkpoint_capability_requires_opt_in_and_single_pp_stage(monkeypatch, opt_in, pp_size, expected):
    layer = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        conv_state_len=9,
        get_state_dtype=lambda: (torch.bfloat16,),
    )
    monkeypatch.setattr(ple, "get_ascend_config", lambda: SimpleNamespace(gdn_internal_prefill_checkpoints=opt_in))
    monkeypatch.setattr(ple, "get_pp_group", lambda: SimpleNamespace(world_size=pp_size))

    assert ple.AscendQwen4ExpPLELayer.get_prefill_checkpoint_alignment(layer) == expected


def test_ple_wrap_exports_history_with_non_spec_token_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ple, "_get_allow_internal_format", lambda: True)
    monkeypatch.setattr(ple, "_set_allow_internal_format", lambda _enabled: None)
    monkeypatch.setattr(ple, "wait_for_kv_layer_from_connector", Mock())
    exporter = Mock()
    monkeypatch.setattr(ple, "MambaPrefillCheckpointExporter", lambda: exporter)
    state = torch.randn(3, 14, 4)
    layer = _make_ple_layer(state, Mock())
    metadata = _DummyPLEMetadata()
    metadata.checkpoint = object()
    metadata.non_spec_query_start_loc = torch.tensor([0, 1, 12])
    metadata.non_spec_token_indx = torch.tensor([0, *range(4, 15)])
    mamba_utils = _install_ple_forward_context(monkeypatch, {layer.prefix: metadata})
    monkeypatch.setattr(mamba_utils, "is_conv_state_dim_first", lambda: False)
    inputs = torch.randn(15, 4)

    ple._wrap_ple_short_conv(Mock())(layer, inputs, torch.zeros_like(inputs), None)

    args, kwargs = exporter.export.call_args
    assert args == (metadata.checkpoint,)
    assert kwargs["raw_qkv"] is inputs
    assert kwargs["cu_seqlens"] is metadata.non_spec_query_start_loc
    assert kwargs["token_indices"] is metadata.non_spec_token_indx
    cache_window = kwargs["conv_state"]
    expected = state[:, 2:11].transpose(1, 2)
    assert cache_window.data_ptr() == expected.data_ptr()
    assert cache_window.shape == expected.shape
    assert cache_window.stride() == expected.stride()


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
