import pytest
import torch
from einops import rearrange
from vllm.model_executor.models.qwen3_next import Qwen3NextGatedDeltaNet

from vllm_ascend.ops.triton.fla.fused_qkvzba_split_reshape import \
    fused_qkvzba_split_reshape_cat


def validate_cmp(y_cal, y_ref, dtype, device='npu'):
    y_cal = y_cal.to(device)
    y_ref = y_ref.to(device)
    if dtype == torch.float16:
        torch.testing.assert_close(y_ref,
                                   y_cal,
                                   rtol=5e-03,
                                   atol=5e-03,
                                   equal_nan=True)
    elif dtype == torch.bfloat16:
        torch.testing.assert_close(y_ref,
                                   y_cal,
                                   rtol=5e-03,
                                   atol=5e-03,
                                   equal_nan=True)
    elif dtype == torch.float32:
        torch.testing.assert_close(y_ref,
                                   y_cal,
                                   rtol=1e-03,
                                   atol=1e-03,
                                   equal_nan=True)
    elif dtype == torch.int32 or dtype == torch.int64 or dtype == torch.int16 or dtype == torch.int8 or dtype == torch.uint32:
        assert torch.equal(y_cal, y_ref)
    elif dtype == torch.bool:
        assert torch.equal(y_cal, y_ref)
    else:
        raise ValueError(
            'Invalid parameter \"dtype\" is found : {}'.format(dtype))


@pytest.mark.parametrize("seq_len", [1, 16, 64, 128, 256, 1024, 2048, 3567])
@pytest.mark.parametrize("num_heads_qk", [2, 4, 8, 16])
@pytest.mark.parametrize("num_heads_v", [2, 4, 8])
@pytest.mark.parametrize("head_qk_dim", [64, 128, 256])
@pytest.mark.parametrize("head_v_dim", [64, 128])
@pytest.mark.parametrize("dtype",
                         [torch.float32, torch.float16, torch.bfloat16])
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

    projected_states_qkvz = torch.randn(seq_len,
                                        2 * head_qk_dim * num_heads_qk +
                                        2 * head_v_dim * num_heads_v,
                                        dtype=dtype,
                                        device=device)

    projected_states_ba = torch.randn(seq_len,
                                      2 * num_heads_v,
                                      dtype=dtype,
                                      device=device)

    projected_states_qkvz_copy = projected_states_qkvz.clone()
    projected_states_ba_copy = projected_states_ba.clone()

    mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat(
        projected_states_qkvz_copy,
        projected_states_ba_copy,
        num_heads_qk,
        num_heads_v,
        head_qk_dim,
        head_v_dim,
    )

    gdn = Qwen3NextGatedDeltaNet.__new__(Qwen3NextGatedDeltaNet)
    gdn.num_k_heads = num_heads_qk
    gdn.num_v_heads = num_heads_v
    gdn.head_k_dim = head_qk_dim
    gdn.head_v_dim = head_v_dim
    gdn.tp_size = 1

    query, key, value, z_ref, b_ref, a_ref = gdn.fix_query_key_value_ordering(
        mixed_qkvz=projected_states_qkvz, mixed_ba=projected_states_ba)
    query, key, value = map(lambda x: rearrange(x, 'l p d -> l (p d)'),
                            (query, key, value))
    mixed_qkv_ref = torch.cat((query, key, value), dim=-1)

    validate_cmp(mixed_qkv, mixed_qkv_ref, dtype)
    validate_cmp(z, z_ref, dtype)
    validate_cmp(b, b_ref, dtype)
    validate_cmp(a, a_ref, dtype)
