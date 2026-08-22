import gc
import math
import os
from pathlib import Path

import pytest
import torch

from vllm_ascend.utils import enable_custom_op

DTYPE = torch.bfloat16
BLOCK_SIZE = 128
# Per-rank geometries from the validated TP16 Kimi K3 + DSpark deployment.
# Keep the test at the same local-rank boundary used by Attention v1/MLA v1
# so it exercises the corresponding FIA tiling keys.
TP_SIZE = 16
DSPARK_NUM_QUERY_HEADS = 64 // TP_SIZE
DSPARK_NUM_KV_HEADS = 16 // TP_SIZE
DSPARK_HEAD_DIM = 64
K3_NUM_LOCAL_QUERY_HEADS = 96 // TP_SIZE
# MLA v1 pads K3's six local heads to the next power of two before FIA.
K3_NUM_QUERY_HEADS = 1 << (K3_NUM_LOCAL_QUERY_HEADS - 1).bit_length()
K3_NUM_KV_HEADS = 1
K3_LATENT_DIM = 512
K3_ROPE_DIM = 64
K3_QK_HEAD_DIM = 128 + K3_ROPE_DIM
BF16_ATOL = 5e-2
BF16_RTOL = 8e-3
REFERENCE_ATOL = 8e-2
REFERENCE_RTOL = 1.5e-2


def _assert_vllm_fia_plugin_available():
    """Require the vllm-ascend _C_ascend FIA path, never torch_npu's wrapper."""
    if not enable_custom_op():
        pytest.skip("requires the vllm-ascend custom-op extension")

    try:
        fia_op = torch.ops._C_ascend.npu_fused_infer_attention_score_v2
    except AttributeError:
        pytest.skip("requires the vllm-ascend _C_ascend FIA binding")

    package_root = Path(__file__).resolve()
    package_root = next(parent for parent in package_root.parents if (parent / "vllm_ascend").is_dir()) / "vllm_ascend"
    custom_root = (package_root / "_cann_ops_custom").resolve()
    expected_vendor = custom_root / "vendors" / "custom_transformer"
    custom_opp_paths = [
        Path(path).resolve() for path in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":") if path
    ]
    assert custom_opp_paths and custom_opp_paths[0] == expected_vendor, (
        "enable_custom_op() did not put the vllm-ascend custom transformer "
        f"first in ASCEND_CUSTOM_OPP_PATH: {custom_opp_paths}"
    )
    assert (expected_vendor / "op_api" / "lib" / "libcust_opapi.so").is_file(), (
        f"vllm-ascend custom transformer has no libcust_opapi.so: {expected_vendor}"
    )
    assert fia_op is not None


def _assert_first_axis_noncontiguous(tensor):
    expected_strides = [1] * tensor.ndim
    for axis in range(tensor.ndim - 2, -1, -1):
        expected_strides[axis] = expected_strides[axis + 1] * tensor.shape[axis + 1]
    assert tensor.stride(0) != expected_strides[0]
    for axis in range(1, tensor.ndim):
        assert tensor.stride(axis) == expected_strides[axis]


def _run_plugin(
    query,
    key,
    value,
    *,
    query_rope=None,
    key_rope=None,
    block_table,
    actual_seq_qlen,
    actual_seq_kvlen,
    input_layout,
    num_query_heads,
    num_key_value_heads,
    softmax_scale,
):
    # This explicit namespace is the point of the test.  Calling
    # torch_npu.npu_fused_infer_attention_score_v2 here would only test the
    # installed op-plugin/CANN wrapper and could not prove the vllm binding.
    return torch.ops._C_ascend.npu_fused_infer_attention_score_v2(
        query,
        key,
        value,
        query_rope=query_rope,
        key_rope=key_rope,
        num_query_heads=num_query_heads,
        num_key_value_heads=num_key_value_heads,
        input_layout=input_layout,
        softmax_scale=softmax_scale,
        block_table=block_table,
        block_size=BLOCK_SIZE,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        sparse_mode=0,
    )[0]


def _paged_kv(cache, block_table, request_index, kv_len):
    page_count = math.ceil(kv_len / BLOCK_SIZE)
    page_ids = block_table[request_index, :page_count].tolist()
    if cache.dim() == 4:
        pages = [cache[int(page_id), 0] for page_id in page_ids]
    else:
        # Attention v1 reshapes the cache to [blocks, block_size, Nkv * D].
        pages = [cache[int(page_id)] for page_id in page_ids]
    return torch.cat(pages, dim=0)[:kv_len]


def _reference_gqa(query, key, value, block_table, actual_seq_qlen, actual_seq_kvlen):
    """Independent FP32 reference for TND GQA PA, including page selection."""
    # Keep the reference independent of the NPU FIA implementation: perform
    # the page gather, matmul, softmax, and value reduction on CPU FP32.
    query_f = query.float().cpu()
    key_f = key.float().cpu()
    value_f = value.float().cpu()
    block_table_cpu = block_table.cpu()
    q_ends = [int(end) for end in actual_seq_qlen]
    kv_lens = [int(length) for length in actual_seq_kvlen]
    outputs = []
    q_start = 0
    for request_index, q_end in enumerate(q_ends):
        request_query = query_f[q_start:q_end]
        request_key = _paged_kv(key_f, block_table_cpu, request_index, kv_lens[request_index])
        request_value = _paged_kv(value_f, block_table_cpu, request_index, kv_lens[request_index])
        scores = torch.einsum("qhd,kd->qhk", request_query, request_key) * (DSPARK_HEAD_DIM**-0.5)
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("qhk,kd->qhd", probabilities, request_value))
        q_start = q_end
    return torch.cat(outputs, dim=0)


def _reference_mla(query_nope, query_rope, key_nope, key_rope, value, block_table, kv_len):
    """Independent FP32 reference for Kimi K3 MLA BNSD_NBSD PA."""
    # Keep the reference independent of the NPU FIA implementation: perform
    # the page gather, split latent/RoPE dot products, softmax, and value
    # reduction on CPU FP32.
    query_nope_f = query_nope.float().cpu()
    query_rope_f = query_rope.float().cpu()
    key_nope_f = key_nope.float().cpu()
    key_rope_f = key_rope.float().cpu()
    value_f = value.float().cpu()
    block_table_cpu = block_table.cpu()
    outputs = []
    for request_index in range(query_nope.shape[0]):
        request_key_nope = _paged_kv(key_nope_f, block_table_cpu, request_index, kv_len)
        request_key_rope = _paged_kv(key_rope_f, block_table_cpu, request_index, kv_len)
        request_value = _paged_kv(value_f, block_table_cpu, request_index, kv_len)
        scores = torch.einsum("nsd,ld->nsl", query_nope_f[request_index], request_key_nope)
        scores += torch.einsum("nsd,ld->nsl", query_rope_f[request_index], request_key_rope)
        scores *= K3_QK_HEAD_DIM**-0.5
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("nsl,ld->nsd", probabilities, request_value))
    # FIA's BNSD_NBSD output is [num_heads, batch, seq, value_dim].
    return torch.stack(outputs, dim=0).permute(1, 0, 2, 3)


def _assert_precision(actual, reference, label):
    actual_f = actual.float().cpu()
    reference_f = reference.float().cpu()
    max_abs = (actual_f - reference_f).abs().max().item()
    max_rel = ((actual_f - reference_f).abs() / reference_f.abs().clamp_min(1e-6)).max().item()
    print(f"{label}: max_abs={max_abs:.6e}, max_rel={max_rel:.6e}")
    torch.testing.assert_close(
        actual_f,
        reference_f,
        atol=REFERENCE_ATOL,
        rtol=REFERENCE_RTOL,
        msg=(f"{label} differs from the independent FP32 reference: max_abs={max_abs:.6g}, max_rel={max_rel:.6g}"),
    )


@torch.inference_mode()
def test_fia_gqa_pa_first_axis_noncontiguous():
    """Kimi K3/DSpark Attention v1 GQA PA with a strided block axis."""
    _assert_vllm_fia_plugin_available()
    torch.manual_seed(20260807)

    query = torch.randn((3, DSPARK_NUM_QUERY_HEADS, DSPARK_HEAD_DIM), dtype=DTYPE, device="npu")
    # DSpark Attention v1 uses Qwen3's global 64/16 GQA geometry. Under the
    # TP16 deployment each rank sees Q=4, KV=1, D=64 and the cache view is
    # [blocks, block_size, Nkv * D].
    storage_shape = (10, BLOCK_SIZE, DSPARK_NUM_KV_HEADS, DSPARK_HEAD_DIM)
    key_storage = torch.randn(storage_shape, dtype=DTYPE, device="npu")
    value_storage = torch.randn(storage_shape, dtype=DTYPE, device="npu")
    key_noncontiguous = key_storage[::2].view(5, BLOCK_SIZE, DSPARK_NUM_KV_HEADS * DSPARK_HEAD_DIM)
    value_noncontiguous = value_storage[::2].view(5, BLOCK_SIZE, DSPARK_NUM_KV_HEADS * DSPARK_HEAD_DIM)
    assert not key_noncontiguous.is_contiguous()
    assert not value_noncontiguous.is_contiguous()
    _assert_first_axis_noncontiguous(key_noncontiguous)
    _assert_first_axis_noncontiguous(value_noncontiguous)

    block_table = torch.tensor([[0, 1, 0], [2, 3, 4]], dtype=torch.int32, device="npu")
    actual_seq_qlen = [1, 3]
    actual_seq_kvlen = [129, 257]

    # Execute the strided cache first.  This keeps the test from accidentally
    # reusing a contiguous tiling result in CANN's executor cache.
    noncontiguous_output = _run_plugin(
        query,
        key_noncontiguous,
        value_noncontiguous,
        block_table=block_table,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        input_layout="TND",
        num_query_heads=DSPARK_NUM_QUERY_HEADS,
        num_key_value_heads=DSPARK_NUM_KV_HEADS,
        softmax_scale=DSPARK_HEAD_DIM**-0.5,
    )
    torch.npu.synchronize()
    contiguous_output = _run_plugin(
        query,
        key_noncontiguous.contiguous(),
        value_noncontiguous.contiguous(),
        block_table=block_table,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        input_layout="TND",
        num_query_heads=DSPARK_NUM_QUERY_HEADS,
        num_key_value_heads=DSPARK_NUM_KV_HEADS,
        softmax_scale=DSPARK_HEAD_DIM**-0.5,
    )
    torch.npu.synchronize()
    reference = _reference_gqa(
        query,
        key_noncontiguous,
        value_noncontiguous,
        block_table,
        actual_seq_qlen,
        actual_seq_kvlen,
    )

    assert noncontiguous_output.shape == (3, DSPARK_NUM_QUERY_HEADS, DSPARK_HEAD_DIM)
    torch.testing.assert_close(
        noncontiguous_output.float(),
        contiguous_output.float(),
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
    )
    _assert_precision(noncontiguous_output, reference, "GQA non-contiguous FIA")
    _assert_precision(contiguous_output, reference, "GQA contiguous FIA")

    gc.collect()
    torch.npu.empty_cache()


@torch.inference_mode()
def test_fia_mla_pa_first_axis_noncontiguous():
    """Kimi K3 MLA v1 latent-512/rope-64 PA with a strided block axis."""
    _assert_vllm_fia_plugin_available()
    torch.manual_seed(20260808)

    # K3 has six local heads at TP16; MLA v1 pads them to eight for FIA and
    # removes the two padded heads after the operator returns.
    query_nope = torch.cat(
        [
            torch.randn((1, K3_NUM_LOCAL_QUERY_HEADS, 1, K3_LATENT_DIM), dtype=DTYPE, device="npu"),
            torch.zeros(
                (1, K3_NUM_QUERY_HEADS - K3_NUM_LOCAL_QUERY_HEADS, 1, K3_LATENT_DIM),
                dtype=DTYPE,
                device="npu",
            ),
        ],
        dim=1,
    )
    query_rope = torch.cat(
        [
            torch.randn((1, K3_NUM_LOCAL_QUERY_HEADS, 1, K3_ROPE_DIM), dtype=DTYPE, device="npu"),
            torch.zeros(
                (1, K3_NUM_QUERY_HEADS - K3_NUM_LOCAL_QUERY_HEADS, 1, K3_ROPE_DIM),
                dtype=DTYPE,
                device="npu",
            ),
        ],
        dim=1,
    )

    key_storage = torch.randn((6, K3_NUM_KV_HEADS, BLOCK_SIZE, K3_LATENT_DIM), dtype=DTYPE, device="npu")
    key_rope_storage = torch.randn((6, K3_NUM_KV_HEADS, BLOCK_SIZE, K3_ROPE_DIM), dtype=DTYPE, device="npu")
    key_nope_noncontiguous = key_storage[::2]
    # Kimi K3 MLA v1 passes the latent cache as both key and value.
    value_noncontiguous = key_nope_noncontiguous
    # MLA v1 has two independent cache planes.  The migration under test is
    # the latent KV cache stride; keep the 64-wide RoPE plane in its normal
    # contiguous layout, as the K3 wrapper does.
    key_rope_contiguous = key_rope_storage[::2].contiguous()
    assert not key_nope_noncontiguous.is_contiguous()
    assert not value_noncontiguous.is_contiguous()
    _assert_first_axis_noncontiguous(key_nope_noncontiguous)
    _assert_first_axis_noncontiguous(value_noncontiguous)

    block_table = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="npu")
    actual_seq_qlen = [1]
    actual_seq_kvlen = [257]

    noncontiguous_output = _run_plugin(
        query_nope,
        key_nope_noncontiguous,
        value_noncontiguous,
        query_rope=query_rope,
        key_rope=key_rope_contiguous,
        block_table=block_table,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        input_layout="BNSD_NBSD",
        num_query_heads=K3_NUM_QUERY_HEADS,
        num_key_value_heads=K3_NUM_KV_HEADS,
        softmax_scale=K3_QK_HEAD_DIM**-0.5,
    )
    torch.npu.synchronize()
    contiguous_output = _run_plugin(
        query_nope,
        key_nope_noncontiguous.contiguous(),
        value_noncontiguous.contiguous(),
        query_rope=query_rope,
        key_rope=key_rope_contiguous,
        block_table=block_table,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        input_layout="BNSD_NBSD",
        num_query_heads=K3_NUM_QUERY_HEADS,
        num_key_value_heads=K3_NUM_KV_HEADS,
        softmax_scale=K3_QK_HEAD_DIM**-0.5,
    )
    torch.npu.synchronize()
    reference = _reference_mla(
        query_nope,
        query_rope,
        key_nope_noncontiguous,
        key_rope_contiguous,
        value_noncontiguous,
        block_table,
        actual_seq_kvlen[0],
    )
    # BNSD_NBSD returns the head-major NBSD output consumed by MLA decode.
    assert noncontiguous_output.shape == (K3_NUM_QUERY_HEADS, 1, 1, K3_LATENT_DIM)
    _assert_precision(noncontiguous_output, reference, "MLA non-contiguous FIA")
    _assert_precision(contiguous_output, reference, "MLA contiguous FIA")
    torch.testing.assert_close(
        noncontiguous_output.float(),
        contiguous_output.float(),
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
    )

    gc.collect()
    torch.npu.empty_cache()
