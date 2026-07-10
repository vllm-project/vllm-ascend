from unittest import mock

import pytest
import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.device.device_op import (
    A5DeviceAdaptor,
    Ascend310PDeviceAdaptor,
    BaseDeviceAdaptor,
)


def _gdn_causal_conv1d_inputs():
    return {
        "x": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "weight": torch.ones(2, 3),
        "conv_state": torch.zeros(2, 2, 3),
        "bias": None,
        "query_start_loc": torch.tensor([0, 1, 2], dtype=torch.int32),
        "cache_indices": torch.tensor([2, 3], dtype=torch.int32),
        "initial_state_mode": torch.tensor([False, True]),
        "num_accepted_tokens": torch.tensor([1, 2], dtype=torch.int32),
        "activation_mode": 1,
        "run_mode": 0,
    }


def test_base_gdn_causal_conv1d_uses_common_operator(monkeypatch):
    captured = {}

    def fake_causal_conv1d(output, x, weight, **kwargs):
        captured.update(kwargs)
        captured["weight"] = weight
        output.copy_(x + 1)

    monkeypatch.setattr(
        torch.ops._C_ascend,
        "npu_causal_conv1d_custom",
        fake_causal_conv1d,
        raising=False,
    )
    inputs = _gdn_causal_conv1d_inputs()

    output = BaseDeviceAdaptor.gdn_causal_conv1d(**inputs)

    assert torch.equal(output, inputs["x"] + 1)
    assert captured["weight"] is inputs["weight"]
    assert captured["conv_state"] is inputs["conv_state"]
    assert captured["bias_opt"] is None
    assert captured["query_start_loc_opt"] is inputs["query_start_loc"]
    assert captured["cache_indices_opt"] is inputs["cache_indices"]
    assert captured["initial_state_mode_opt"] is inputs["initial_state_mode"]
    assert captured["num_accepted_tokens_opt"] is inputs["num_accepted_tokens"]
    assert captured["activation_mode"] == 1
    assert captured["pad_slot_id"] == PAD_SLOT_ID
    assert captured["run_mode"] == 0


def test_310p_gdn_causal_conv1d_uses_310p_operator(monkeypatch):
    captured = {}

    def fake_causal_conv1d(x, weight, **kwargs):
        captured.update(kwargs)
        captured["weight"] = weight
        return x + 1

    monkeypatch.setattr(
        torch.ops._C_ascend,
        "npu_causal_conv1d_310",
        fake_causal_conv1d,
        raising=False,
    )
    inputs = _gdn_causal_conv1d_inputs()

    output = Ascend310PDeviceAdaptor.gdn_causal_conv1d(**inputs)

    assert torch.equal(output, inputs["x"] + 1)
    assert captured["weight"] is inputs["weight"]
    assert captured["conv_states"] is inputs["conv_state"]
    assert captured["bias"] is None
    assert captured["query_start_loc"] is inputs["query_start_loc"]
    assert captured["cache_indices"] is inputs["cache_indices"]
    assert captured["initial_state_mode"] is inputs["initial_state_mode"]
    assert captured["num_accepted_tokens"] is inputs["num_accepted_tokens"]
    assert captured["activation_mode"] == 1
    assert captured["pad_slot_id"] == PAD_SLOT_ID
    assert captured["run_mode"] == 0


def test_base_gdn_recurrent_decode_normalizes_operator_contract(monkeypatch):
    captured = {}

    def fake_recurrent(**kwargs):
        captured.update(kwargs)
        return kwargs["value"]

    monkeypatch.setattr(
        torch.ops._C_ascend,
        "npu_recurrent_gated_delta_rule",
        fake_recurrent,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.fla.ops.l2norm.l2norm_fwd",
        lambda tensor: tensor,
    )
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 2, 3, 4)
    value = torch.randn(1, 2, 3, 5)
    g = torch.randn(1, 2, 3, 4)
    beta = torch.randn(1, 2, 3)
    state = torch.randn(4, 3, 4, 5)
    actual_seq_lengths = torch.tensor([0, 1, 1], dtype=torch.int32)
    state_indices = torch.tensor([4, 5], dtype=torch.int32)
    accepted_tokens = torch.tensor([1, 2], dtype=torch.int64)

    output = BaseDeviceAdaptor.gdn_recurrent_decode(
        query,
        key,
        value,
        g,
        beta,
        state,
        actual_seq_lengths,
        state_indices,
        accepted_tokens,
    )

    assert output.shape == value.shape
    assert captured["query"].shape == query.shape[1:]
    assert captured["key"].shape == key.shape[1:]
    assert captured["value"].shape == value.shape[1:]
    assert captured["g"].shape == g.shape[1:]
    assert captured["beta"].shape == beta.shape[1:]
    assert captured["scale"] == key.shape[-1] ** -0.5
    assert captured["actual_seq_lengths"] is actual_seq_lengths
    assert captured["ssm_state_indices"] is state_indices
    assert captured["num_accepted_tokens"].dtype == torch.int32


def test_310p_gdn_recurrent_metadata_adaptation():
    accepted_tokens = torch.tensor([2, 3, 4], dtype=torch.int64)
    actual_seq_lengths = torch.tensor([0, 4, 0, 1], dtype=torch.int32)
    masked = Ascend310PDeviceAdaptor._mask_gdn_accepted_tokens(
        accepted_tokens,
        actual_seq_lengths,
    )
    state_indices = torch.tensor(
        [[10, 11], [20, 21], [30, 31]],
        dtype=torch.int32,
    )
    uniform_seq_lengths = torch.tensor([0, 2, 2, 2], dtype=torch.int32)
    flattened = Ascend310PDeviceAdaptor._flatten_gdn_state_indices(
        state_indices,
        uniform_seq_lengths,
        total_tokens=6,
    )

    assert masked.dtype == torch.int32
    assert masked.tolist() == [2, 0, 4]
    assert flattened.tolist() == [10, 11, 20, 21, 30, 31]


def test_310p_gdn_recurrent_decode_uses_adapted_operator_contract(monkeypatch):
    captured = {}

    def fake_recurrent(**kwargs):
        captured.update(kwargs)
        return kwargs["value"]

    monkeypatch.setattr(
        torch.ops._C_ascend,
        "npu_recurrent_gated_delta_rule_310",
        fake_recurrent,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm_ascend._310p.ops.fla.l2norm.l2norm_310p",
        lambda tensor: tensor,
    )
    query = torch.randn(1, 4, 2, 3)
    key = torch.randn(1, 4, 2, 3)
    value = torch.randn(1, 4, 2, 5)
    g = torch.randn(1, 4, 2, 3)
    beta = torch.randn(1, 4, 2)
    state = torch.randn(32, 2, 5, 3)
    actual_seq_lengths = torch.tensor([0, 2, 2], dtype=torch.int64)
    state_indices = torch.tensor([[10, 11], [20, 21]], dtype=torch.int64)
    accepted_tokens = torch.tensor([2, 1], dtype=torch.int64)

    output = Ascend310PDeviceAdaptor.gdn_recurrent_decode(
        query,
        key,
        value,
        g,
        beta,
        state,
        actual_seq_lengths,
        state_indices,
        accepted_tokens,
    )

    assert output.shape == value.shape
    assert captured["query"].dtype == torch.float16
    assert captured["key"].dtype == torch.float16
    assert captured["value"].dtype == torch.float16
    assert captured["g"].dtype == torch.float32
    assert captured["beta"].dtype == torch.float16
    assert captured["actual_seq_lengths"].dtype == torch.int32
    assert captured["actual_seq_lengths"].tolist() == [0, 2, 2]
    assert captured["ssm_state_indices"].dtype == torch.int32
    assert captured["ssm_state_indices"].tolist() == [10, 11, 20, 21]
    assert captured["num_accepted_tokens"].dtype == torch.int32
    assert captured["num_accepted_tokens"].tolist() == [2, 1]
    assert captured["scale_value"] == key.shape[-1] ** -0.5


def test_310p_fused_gdn_gating_uses_pytorch_fallback():
    A_log = torch.randn(2)
    a = torch.randn(3, 2)
    b = torch.randn(3, 2)
    dt_bias = torch.randn(2)
    expected = (torch.randn(1, 3, 2), torch.randn(1, 3, 2))

    with mock.patch(
        "vllm_ascend._310p.ops.fla.fused_gdn_gating.fused_gdn_gating_pytorch",
        return_value=expected,
    ) as mock_gating:
        output = Ascend310PDeviceAdaptor.fused_gdn_gating(A_log, a, b, dt_bias)

    assert output is expected
    mock_gating.assert_called_once_with(A_log, a, b, dt_bias)


def test_npu_flash_attention_uses_fusion_attention_for_fp32():
    query = torch.randn(5, 4, 64, dtype=torch.float32)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    seq_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)
    expected = torch.randn_like(query)

    with (
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu.npu_fusion_attention",
            return_value=(expected,),
        ) as mock_fusion_attention,
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu._npu_flash_attention_unpad",
            create=True,
        ) as mock_flash_attention,
    ):
        output = BaseDeviceAdaptor.npu_flash_attention(
            query=query,
            key=key,
            value=value,
            seq_lens_cpu=seq_lens_cpu,
            head_num=4,
            scale_value=0.125,
            num_kv_heads=4,
        )

    assert output is expected
    mock_flash_attention.assert_not_called()
    mock_fusion_attention.assert_called_once()
    call_kwargs = mock_fusion_attention.call_args.kwargs
    assert call_kwargs["query"] is query
    assert call_kwargs["key"] is key
    assert call_kwargs["value"] is value
    assert call_kwargs["actual_seq_qlen"] == [2, 5]
    assert all(isinstance(seq_len, int) for seq_len in call_kwargs["actual_seq_qlen"])
    assert call_kwargs["actual_seq_kvlen"] is call_kwargs["actual_seq_qlen"]
    assert call_kwargs["head_num"] == 4
    assert call_kwargs["scale"] == 0.125
    assert call_kwargs["input_layout"] == "TND"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_npu_flash_attention_uses_unpad_attention_for_low_precision(dtype):
    query = torch.randn(5, 4, 64, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    seq_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)

    def fake_flash_attention(*, query, key, value, seq_len, scale_value, num_heads, num_kv_heads, out):
        out.copy_(query + 1)

    with (
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu.npu_fusion_attention",
        ) as mock_fusion_attention,
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu._npu_flash_attention_unpad",
            side_effect=fake_flash_attention,
            create=True,
        ) as mock_flash_attention,
    ):
        output = BaseDeviceAdaptor.npu_flash_attention(
            query=query,
            key=key,
            value=value,
            seq_lens_cpu=seq_lens_cpu,
            head_num=4,
            scale_value=0.125,
            num_kv_heads=4,
        )

    mock_fusion_attention.assert_not_called()
    mock_flash_attention.assert_called_once()
    call_kwargs = mock_flash_attention.call_args.kwargs
    assert call_kwargs["query"] is query
    assert call_kwargs["key"] is key
    assert call_kwargs["value"] is value
    assert call_kwargs["seq_len"] is seq_lens_cpu
    assert call_kwargs["num_heads"] == 4
    assert call_kwargs["num_kv_heads"] == 4
    assert call_kwargs["scale_value"] == 0.125
    torch.testing.assert_close(output, query + 1)


def test_a5_npu_flash_attention_uses_python_sequence_lengths():
    query = torch.randn(5, 4, 64, dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    seq_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)
    expected = torch.randn_like(query)

    with mock.patch(
        "vllm_ascend.device.device_op.torch_npu.npu_fusion_attention",
        return_value=(expected,),
    ) as mock_fusion_attention:
        output = A5DeviceAdaptor.npu_flash_attention(
            query=query,
            key=key,
            value=value,
            seq_lens_cpu=seq_lens_cpu,
            head_num=4,
            scale_value=0.125,
            num_kv_heads=4,
        )

    assert output is expected
    call_kwargs = mock_fusion_attention.call_args.kwargs
    assert call_kwargs["actual_seq_qlen"] == [2, 5]
    assert all(isinstance(seq_len, int) for seq_len in call_kwargs["actual_seq_qlen"])
    assert call_kwargs["actual_seq_kvlen"] is call_kwargs["actual_seq_qlen"]
