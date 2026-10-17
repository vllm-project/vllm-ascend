import gc

import pytest
import torch

from vllm_ascend.ops.triton.fla.fused_qkvzba_split_reshape import fused_qkvzba_split_reshape_cat


def fused_qkvzba_split_reshape_cat_torch_ref(
    mixed_qkvz,
    mixed_ba,
    num_heads_qk,
    num_heads_v,
    head_qk_dim,
    head_v_dim,
):
    """Pure-torch reference of fused_qkvzba_split_reshape_cat.

    mixed_qkvz layout: [seq, num_heads_qk, (Q, K, V, Z)] where Q/K have
    head_qk_dim elements and V/Z have (num_heads_v // num_heads_qk) * head_v_dim.
    mixed_ba layout:  [seq, num_heads_qk, (B, A)] with num_heads_v // num_heads_qk
    elements each.
    """
    v_heads_per_qk = num_heads_v // num_heads_qk
    qkvz_dim_t = head_qk_dim * 2 + v_heads_per_qk * head_v_dim * 2
    ba_dim_t = v_heads_per_qk * 2

    seq_len = mixed_qkvz.shape[0]
    mixed_qkvz = mixed_qkvz.view(seq_len, num_heads_qk, qkvz_dim_t)
    mixed_ba = mixed_ba.view(seq_len, num_heads_qk, ba_dim_t)

    q = mixed_qkvz[..., :head_qk_dim]
    k = mixed_qkvz[..., head_qk_dim : 2 * head_qk_dim]
    v = mixed_qkvz[..., 2 * head_qk_dim : 2 * head_qk_dim + v_heads_per_qk * head_v_dim]
    z = mixed_qkvz[..., 2 * head_qk_dim + v_heads_per_qk * head_v_dim :]
    b = mixed_ba[..., :v_heads_per_qk]
    a = mixed_ba[..., v_heads_per_qk:]

    v = v.reshape(seq_len, num_heads_v, head_v_dim)
    z = z.reshape(seq_len, num_heads_v, head_v_dim)
    b = b.reshape(seq_len, num_heads_v)
    a = a.reshape(seq_len, num_heads_v)

    q = q.reshape(seq_len, num_heads_qk * head_qk_dim)
    k = k.reshape(seq_len, num_heads_qk * head_qk_dim)
    v_flat = v.reshape(seq_len, num_heads_v * head_v_dim)

    mixed_qkv = torch.cat((q, k, v_flat), dim=-1)
    return mixed_qkv, z, b, a


def validate_cmp(y_cal, y_ref, device="npu"):
    y_cal = y_cal.to(device)
    y_ref = y_ref.to(device)
    # Pure data-movement op (load/store only, no arithmetic): results must be
    # bit-exact, which is the unified precision tolerance for this operator type.
    torch.testing.assert_close(y_ref, y_cal, rtol=0, atol=0, equal_nan=True)


# Real inference shapes used by Qwen3-GDN-10B (gqa_interleaved_layout=True),
# with num_heads_qk / num_heads_v divided by tp_size (here tp_size = 1):
#   num_k_heads=16, num_v_heads=128, head_k_dim=512, head_v_dim=512
# Plus a generic case covering a wider parameter space (v_heads_per_qk >= 1).
TEST_CASES = [
    (64, 16, 128, 512, 512),
    (2048, 16, 128, 512, 512),
    (128, 8, 32, 128, 64),
]


@pytest.mark.parametrize(
    "seq_len, num_heads_qk, num_heads_v, head_qk_dim, head_v_dim",
    TEST_CASES,
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_qkvzba_split_reshape_cat(
    seq_len,
    num_heads_qk,
    num_heads_v,
    head_qk_dim,
    head_v_dim,
    dtype,
):
    if num_heads_v % num_heads_qk != 0:
        pytest.skip("num_heads_v must be divisible by num_heads_qk")

    torch.random.manual_seed(0)
    device = "npu"

    projected_states_qkvz = torch.randn(
        seq_len, 2 * head_qk_dim * num_heads_qk + 2 * head_v_dim * num_heads_v, dtype=dtype, device=device
    )

    projected_states_ba = torch.randn(seq_len, 2 * num_heads_v, dtype=dtype, device=device)

    mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat(
        projected_states_qkvz.clone(),
        projected_states_ba.clone(),
        num_heads_qk,
        num_heads_v,
        head_qk_dim,
        head_v_dim,
    )

    mixed_qkv_ref, z_ref, b_ref, a_ref = fused_qkvzba_split_reshape_cat_torch_ref(
        projected_states_qkvz,
        projected_states_ba,
        num_heads_qk,
        num_heads_v,
        head_qk_dim,
        head_v_dim,
    )

    validate_cmp(mixed_qkv, mixed_qkv_ref)
    validate_cmp(z, z_ref)
    validate_cmp(b, b_ref)
    validate_cmp(a, a_ref)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
