# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

enable_custom_op()
torch_npu.npu.config.allow_internal_format = True


def _tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: str = "meta",
) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=device)


def _sfa_inputs(device: str = "meta") -> tuple[torch.Tensor, ...]:
    return (
        _tensor((2, 4, 576), torch.bfloat16, device),
        _tensor((2, 16, 1, 386), torch.int8, device),
        _tensor((2, 16, 1, 386), torch.int8, device),
        _tensor((2, 1, 8), torch.int32, device),
    )


def _sfa_kwargs(device: str = "meta") -> dict[str, object]:
    return {
        "block_table": _tensor((1, 2), torch.int32, device),
        "actual_seq_lengths_query": _tensor((1,), torch.int32, device),
        "actual_seq_lengths_kv": _tensor((1,), torch.int32, device),
        "scale_value": 0.125,
        "key_quant_mode": 3,
        "value_quant_mode": 3,
        "sparse_block_size": 1,
        "layout_query": "TND",
        "layout_kv": "PA_BSND",
        "sparse_mode": 3,
        "attention_mode": 2,
        "quant_scale_repo_mode": 1,
        "tile_size": 128,
        "rope_head_dim": 64,
        "return_softmax_lse": True,
    }


def test_turboquant_sfa_schema_hides_cann_window_sentinels() -> None:
    schema = str(torch.ops._C_ascend.turboquant_sparse_flash_attention.default._schema)

    assert "pre_tokens" not in schema
    assert "next_tokens" not in schema


@pytest.mark.parametrize(
    "required_input",
    [
        "block_table",
        "actual_seq_lengths_query",
        "actual_seq_lengths_kv",
    ],
)
def test_turboquant_sfa_schema_requires_paged_inputs(required_input: str) -> None:
    kwargs = _sfa_kwargs()
    kwargs.pop(required_input)

    with pytest.raises(RuntimeError, match=required_input):
        torch.ops._C_ascend.turboquant_sparse_flash_attention(*_sfa_inputs(), **kwargs)


def test_turboquant_sfa_meta_shapes() -> None:
    output, softmax_max, softmax_sum = torch.ops._C_ascend.turboquant_sparse_flash_attention(
        *_sfa_inputs(),
        **_sfa_kwargs(),
    )

    assert output.shape == (2, 4, 512)
    assert output.dtype == torch.bfloat16
    assert softmax_max.shape == softmax_sum.shape == (1, 2, 4)
    assert softmax_max.dtype == softmax_sum.dtype == torch.float32


@pytest.mark.parametrize("scale_name", ["key_dequant_scale", "value_dequant_scale"])
def test_turboquant_sfa_meta_rejects_invalid_dequant_scale_dtype(scale_name: str) -> None:
    kwargs = _sfa_kwargs()
    kwargs[scale_name] = _tensor((1,), torch.bfloat16)

    with pytest.raises(RuntimeError, match=f"{scale_name} must have float32 dtype"):
        torch.ops._C_ascend.turboquant_sparse_flash_attention(*_sfa_inputs(), **kwargs)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("sparse_head_mismatch", "token/KV-head dimensions"),
        ("empty_kv_blocks", "block count and block size must be non-zero"),
        ("empty_kv_block_size", "block count and block size must be non-zero"),
        ("unaligned_kv_block_size", "KV block size must be aligned"),
        ("empty_block_batch", "block_table must match the sequence batch"),
        ("empty_block_columns", "block_table must match the sequence batch"),
        ("empty_query_lengths", "actual_seq_lengths_query must be non-empty"),
        ("length_batch_mismatch", "sequence-length tensors must have the same batch size"),
        ("block_batch_mismatch", "block_table must match the sequence batch"),
    ],
)
def test_turboquant_sfa_meta_rejects_invalid_paged_contract(
    case: str,
    message: str,
) -> None:
    inputs = list(_sfa_inputs())
    kwargs = _sfa_kwargs()

    if case == "sparse_head_mismatch":
        inputs[3] = _tensor((2, 4, 8), torch.int32)
    elif case == "empty_kv_blocks":
        inputs[1] = inputs[2] = _tensor((0, 16, 1, 386), torch.int8)
    elif case == "empty_kv_block_size":
        inputs[1] = inputs[2] = _tensor((2, 0, 1, 386), torch.int8)
    elif case == "unaligned_kv_block_size":
        inputs[1] = inputs[2] = _tensor((2, 8, 1, 386), torch.int8)
    elif case == "empty_block_batch":
        kwargs["block_table"] = _tensor((0, 2), torch.int32)
    elif case == "empty_block_columns":
        kwargs["block_table"] = _tensor((1, 0), torch.int32)
    elif case == "empty_query_lengths":
        kwargs["actual_seq_lengths_query"] = _tensor((0,), torch.int32)
    elif case == "length_batch_mismatch":
        kwargs["actual_seq_lengths_query"] = _tensor((2,), torch.int32)
        kwargs["block_table"] = _tensor((2, 2), torch.int32)
    elif case == "block_batch_mismatch":
        kwargs["block_table"] = _tensor((2, 2), torch.int32)
    else:
        raise AssertionError(f"Unhandled test case: {case}")

    with pytest.raises(RuntimeError, match=message):
        torch.ops._C_ascend.turboquant_sparse_flash_attention(*inputs, **kwargs)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("empty_kv_blocks", "block count and block size must be non-zero"),
        ("sparse_head_mismatch", "token/KV-head dimensions"),
    ],
)
def test_turboquant_sfa_runtime_adapter_rejects_before_aclnn(
    case: str,
    message: str,
) -> None:
    inputs = list(_sfa_inputs("npu"))
    kwargs = _sfa_kwargs("npu")
    if case == "empty_kv_blocks":
        inputs[1] = inputs[2] = _tensor((0, 16, 1, 386), torch.int8, "npu")
    else:
        inputs[3] = _tensor((2, 4, 8), torch.int32, "npu")

    with pytest.raises(RuntimeError, match=message):
        torch.ops._C_ascend.turboquant_sparse_flash_attention(*inputs, **kwargs)


@pytest.mark.parametrize(
    ("query_shape", "key_shape", "message"),
    [
        ((2, 576), (2, 16, 1, 386), "query dimension must be 3"),
        ((2, 4, 576), (2, 16, 386), "key dimension must be 4"),
        ((2, 4, 576), (2, 16, 0, 386), "exactly one KV head"),
    ],
)
def test_turboquant_sfa_meta_rejects_invalid_ranks_and_heads(
    query_shape: tuple[int, ...],
    key_shape: tuple[int, ...],
    message: str,
) -> None:
    query = _tensor(query_shape, torch.bfloat16)
    key = _tensor(key_shape, torch.int8)
    value = _tensor(key_shape, torch.int8)
    sparse_indices = _tensor((2, 1, 8), torch.int32)

    with pytest.raises(RuntimeError, match=message):
        torch.ops._C_ascend.turboquant_sparse_flash_attention(
            query,
            key,
            value,
            sparse_indices,
            **_sfa_kwargs(),
        )


def test_turboquant_compress_meta_shape() -> None:
    slot = torch.ops._C_ascend.turbo_quant_compress_latent(
        _tensor((3, 512), torch.float32),
        _tensor((16,), torch.float32),
    )

    assert slot.shape == (3, 320)
    assert slot.dtype == torch.uint8


@pytest.mark.parametrize(
    ("latent_shape", "latent_dtype", "centroid_shape", "centroid_dtype", "message"),
    [
        ((512,), torch.float32, (16,), torch.float32, r"\[N, 512\]"),
        ((3, 511), torch.float32, (16,), torch.float32, r"\[N, 512\]"),
        ((3, 512), torch.bfloat16, (16,), torch.float32, "float32 inputs"),
        ((3, 512), torch.float32, (15,), torch.float32, "exactly 16 centroids"),
        ((3, 512), torch.float32, (16,), torch.bfloat16, "float32 inputs"),
    ],
)
def test_turboquant_compress_meta_rejects_invalid_contract(
    latent_shape: tuple[int, ...],
    latent_dtype: torch.dtype,
    centroid_shape: tuple[int, ...],
    centroid_dtype: torch.dtype,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        torch.ops._C_ascend.turbo_quant_compress_latent(
            _tensor(latent_shape, latent_dtype),
            _tensor(centroid_shape, centroid_dtype),
        )
