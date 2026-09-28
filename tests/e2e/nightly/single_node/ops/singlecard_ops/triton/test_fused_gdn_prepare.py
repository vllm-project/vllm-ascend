import pytest
import torch
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

from vllm_ascend._310p.ops.fla.fused_gdn_gating import fused_gdn_gating_pytorch
from vllm_ascend.ops.triton.fused_gdn_prepare import fused_gdn_prepare_impl
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _reference_prepare(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    key_dim: int,
    value_dim: int,
    tp_size: int,
    index: torch.Tensor | None = None,
):
    """The legacy chain the fused kernel replaces, as an independent reference.

    rearrange_mixed_qkv (split + reshape + contiguous) + l2norm_fwd(q/k) +
    fused_gdn_gating_pytorch (+ index_select of g/beta on the spec path).
    """
    nk = num_k_heads // tp_size
    nv = num_v_heads // tp_size
    num_tokens = mixed_qkv.shape[0]
    q_dim = key_dim // tp_size
    v_dim = value_dim // tp_size

    query, key, value = torch.split(mixed_qkv, [q_dim, q_dim, v_dim], dim=-1)
    query = query.reshape(1, num_tokens, nk, head_k_dim).contiguous()
    key = key.reshape(1, num_tokens, nk, head_k_dim).contiguous()
    value = value.reshape(1, num_tokens, nv, head_v_dim).contiguous()
    query = l2norm_fwd(query)
    key = l2norm_fwd(key)

    g, beta = fused_gdn_gating_pytorch(
        A_log=a_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        beta=1.0,
        threshold=20.0,
    )
    if index is not None:
        g = g.index_select(1, index)
        beta = beta.index_select(1, index)
    return query, key, value, g, beta


def _run_and_compare(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    key_dim: int,
    value_dim: int,
    tp_size: int,
    index: torch.Tensor | None,
):
    num_tokens = mixed_qkv.shape[0]
    nk = num_k_heads // tp_size
    nv = num_v_heads // tp_size

    q, k, v, g, beta = fused_gdn_prepare_impl(
        mixed_qkv,
        a,
        b,
        a_log,
        dt_bias,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        key_dim,
        value_dim,
        tp_size,
        index,
    )

    assert q.shape == (1, num_tokens, nk, head_k_dim)
    assert k.shape == (1, num_tokens, nk, head_k_dim)
    assert v.shape == (1, num_tokens, nv, head_v_dim)
    assert g.shape == (1, num_tokens, nv) and g.dtype == torch.float32
    assert beta.shape == (1, num_tokens, nv) and beta.dtype == b.dtype

    q_ref, k_ref, v_ref, g_ref, beta_ref = _reference_prepare(
        mixed_qkv,
        a,
        b,
        a_log,
        dt_bias,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        key_dim,
        value_dim,
        tp_size,
        index,
    )

    torch.testing.assert_close(q.cpu(), q_ref.cpu(), rtol=1e-2, atol=1e-2, equal_nan=True)
    torch.testing.assert_close(k.cpu(), k_ref.cpu(), rtol=1e-2, atol=1e-2, equal_nan=True)
    torch.testing.assert_close(v.cpu(), v_ref.cpu(), rtol=1e-2, atol=1e-2, equal_nan=True)
    torch.testing.assert_close(
        g.to(torch.float32).cpu(), g_ref.to(torch.float32).cpu(), rtol=1e-2, atol=1e-2, equal_nan=True
    )
    torch.testing.assert_close(
        beta.to(torch.float32).cpu(), beta_ref.to(torch.float32).cpu(), rtol=1e-2, atol=1e-2, equal_nan=True
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [8, 333, 4096])
def test_fused_gdn_prepare_parity(dtype, num_tokens):
    """Non-spec path parity across both BLOCK_T tiers and dtypes."""
    init_device_properties_triton()
    torch.manual_seed(0)
    device = "npu"

    # Asymmetric head counts so a cross-wired q/k/v section shows up in parity.
    num_k_heads, num_v_heads = 4, 8
    head_k_dim, head_v_dim = 128, 128
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    tp_size = 1

    qkv_dim = 2 * key_dim // tp_size + value_dim // tp_size
    mixed_qkv = torch.randn(num_tokens, qkv_dim, dtype=dtype, device=device)
    a = torch.randn(num_tokens, num_v_heads // tp_size, dtype=dtype, device=device)
    b = torch.randn(num_tokens, num_v_heads // tp_size, dtype=dtype, device=device)
    a_log = torch.randn(num_v_heads // tp_size, dtype=dtype, device=device)
    dt_bias = torch.randn(num_v_heads // tp_size, dtype=dtype, device=device)

    _run_and_compare(
        mixed_qkv,
        a,
        b,
        a_log,
        dt_bias,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        key_dim,
        value_dim,
        tp_size,
        index=None,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_gdn_prepare_spec_index_parity(dtype):
    """Spec path: mixed_qkv is a gathered subset, a/b stay whole; the kernel
    must gather the same g/beta rows the legacy index_select produced."""
    init_device_properties_triton()
    torch.manual_seed(0)
    device = "npu"

    num_k_heads, num_v_heads = 4, 8
    head_k_dim, head_v_dim = 128, 128
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    tp_size = 1
    num_tokens_all = 64

    nv = num_v_heads // tp_size
    a = torch.randn(num_tokens_all, nv, dtype=dtype, device=device)
    b = torch.randn(num_tokens_all, nv, dtype=dtype, device=device)
    a_log = torch.randn(nv, dtype=dtype, device=device)
    dt_bias = torch.randn(nv, dtype=dtype, device=device)

    # Unsorted, with repeats and boundary rows, like a real spec subset.
    index = torch.tensor([5, 3, 5, 60, 0, 17], dtype=torch.int32, device=device)
    qkv_dim = 2 * key_dim // tp_size + value_dim // tp_size
    mixed_qkv = torch.randn(num_tokens_all, qkv_dim, dtype=dtype, device=device).index_select(0, index)

    _run_and_compare(
        mixed_qkv,
        a,
        b,
        a_log,
        dt_bias,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        key_dim,
        value_dim,
        tp_size,
        index=index,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("row_offset", [0, 7])
@pytest.mark.parametrize("num_tokens", [8, 333])
def test_fused_gdn_prepare_head_first_parity(dtype, row_offset, num_tokens):
    """Head-first parity: q/k/v arrive as [T, H, D] views of FLA's [H, T, D]
    convolution output instead of a packed mixed_qkv row, and gating rows can
    start at ``row_offset`` for split mixed batches."""
    init_device_properties_triton()
    torch.manual_seed(0)
    device = "npu"

    num_k_heads, num_v_heads = 4, 8
    head_k_dim, head_v_dim = 128, 128
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    tp_size = 1
    nk = num_k_heads // tp_size
    nv = num_v_heads // tp_size
    num_tokens_all = num_tokens + row_offset

    # Build the packed rows first so both layouts feed identical values.
    qkv_dim = 2 * key_dim // tp_size + value_dim // tp_size
    mixed_qkv = torch.randn(num_tokens, qkv_dim, dtype=dtype, device=device)
    a = torch.randn(num_tokens_all, nv, dtype=dtype, device=device)
    b = torch.randn(num_tokens_all, nv, dtype=dtype, device=device)
    a_log = torch.randn(nv, dtype=dtype, device=device)
    dt_bias = torch.randn(nv, dtype=dtype, device=device)

    # q/k/v reference: same values the packed rows carry.
    q_ref, k_ref, v_ref, _, _ = _reference_prepare(
        mixed_qkv,
        a,
        b,
        a_log,
        dt_bias,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        key_dim,
        value_dim,
        tp_size,
    )

    # Repack q/k/v into FLA's head-first [H, T, D] buffer and take the
    # token-major views the way _split_head_first_conv_output does.
    q_pack = q_ref.squeeze(0).permute(1, 0, 2)  # [H, T, D]
    k_pack = k_ref.squeeze(0).permute(1, 0, 2)
    v_pack = v_ref.squeeze(0).permute(1, 0, 2)
    q_hf = q_pack.transpose(0, 1)  # [T, H, D] view of the [H, T, D] buffer
    k_hf = k_pack.transpose(0, 1)
    v_hf = v_pack.transpose(0, 1)
    assert not q_hf.is_contiguous()

    q, k, v, g, beta = fused_gdn_prepare_impl(
        None,
        a,
        b,
        a_log,
        dt_bias,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        key_dim,
        value_dim,
        tp_size,
        None,
        head_first_qkv=(q_hf, k_hf, v_hf),
        row_offset=row_offset,
    )

    assert q.shape == (1, num_tokens, nk, head_k_dim)
    assert v.shape == (1, num_tokens, nv, head_v_dim)
    torch.testing.assert_close(q.cpu(), q_ref.cpu(), rtol=1e-2, atol=1e-2, equal_nan=True)
    torch.testing.assert_close(k.cpu(), k_ref.cpu(), rtol=1e-2, atol=1e-2, equal_nan=True)
    torch.testing.assert_close(v.cpu(), v_ref.cpu(), rtol=1e-2, atol=1e-2, equal_nan=True)

    # Gating rows come from a[row_offset : row_offset + T].
    g_expected, beta_expected = fused_gdn_gating_pytorch(
        A_log=a_log,
        a=a[row_offset : row_offset + num_tokens],
        b=b[row_offset : row_offset + num_tokens],
        dt_bias=dt_bias,
        beta=1.0,
        threshold=20.0,
    )
    torch.testing.assert_close(
        g.to(torch.float32).cpu(), g_expected.to(torch.float32).cpu(), rtol=1e-2, atol=1e-2, equal_nan=True
    )
    torch.testing.assert_close(
        beta.to(torch.float32).cpu(), beta_expected.to(torch.float32).cpu(), rtol=1e-2, atol=1e-2, equal_nan=True
    )
