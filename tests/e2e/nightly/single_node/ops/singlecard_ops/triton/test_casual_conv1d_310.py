import torch

from tests.ut.base import PytestBase
from vllm_ascend._310p.ops.causal_conv1d import causal_conv1d_fn_pytorch, causal_conv1d_update_pytorch
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


class TestCausalConv1d(PytestBase):
    def test_triton_matches_torch_reference(self):

        from vllm_ascend.ops.triton.mamba.causal_conv1d import causal_conv1d_update_npu

        torch.manual_seed(0)
        device = torch.device("npu")

        total_tokens = 25
        hidden_size = 8192
        kernel_size = 4
        conv_state_blocks = 366
        bs = 1

        mixed_qkv_prefill = torch.randn(total_tokens, hidden_size, dtype=torch.bfloat16, device=device)
        mixed_qkv_decode = torch.randn(1, hidden_size, dtype=torch.bfloat16, device=device)
        conv_weights = torch.randn(hidden_size, kernel_size, dtype=torch.bfloat16, device=device)
        activation = "silu"
        # Create conv_state with contiguous memory layout where dim dimension is stride 1
        conv_state = torch.randn(
            conv_state_blocks, kernel_size - 1, hidden_size, dtype=torch.bfloat16, device=device
        ).contiguous()
        conv_state = conv_state.transpose(1, 2)
        fn_has_initial_state = torch.tensor([False])
        cache_indices = torch.randint(low=0, high=conv_state_blocks, size=(bs,), dtype=torch.int32, device=device)
        query_start_loc = torch.tensor([0, total_tokens], dtype=torch.int32, device=device)

        fn_triton_out = torch.ops._C_ascend.causal_conv1d_fn(
            mixed_qkv_prefill,
            conv_weights,
            bias=None,
            activation=activation,
            conv_states=conv_state,
            has_initial_state=fn_has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )
        mixed_qkv_prefill_T = mixed_qkv_prefill.transpose(0, 1)
        fn_ref_out = causal_conv1d_fn_pytorch(
            mixed_qkv_prefill_T,
            conv_weights,
            bias=None,
            activation=activation,
            conv_states=conv_state,
            has_initial_state=fn_has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )

        update_triton_out = causal_conv1d_update_npu(
            mixed_qkv_decode,
            conv_state,
            conv_weights,
            bias=None,
            activation=activation,
            conv_state_indices=cache_indices,
            validate_data=True,
        )
        mixed_qkv_decode_T = mixed_qkv_decode.transpose(0, 1)
        update_ref_out = causal_conv1d_update_pytorch(
            mixed_qkv_decode_T,
            conv_state,
            conv_weights,
            bias=None,
            activation=activation,
            conv_state_indices=cache_indices,
        )

        torch.testing.assert_close(fn_triton_out, fn_ref_out, rtol=1e-2, atol=1e-2, equal_nan=True)
        torch.testing.assert_close(update_triton_out, update_ref_out, rtol=1e-2, atol=1e-2, equal_nan=True)
