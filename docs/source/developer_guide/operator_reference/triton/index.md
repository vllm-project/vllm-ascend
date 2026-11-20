# Triton operators

Find an operator by its callable name. Each entry links to the wrapper, a call or registration site and the operator description. To add or update an entry, follow the [contribution guide](../../contribution/triton_operators.md).

| Operator | Purpose |
| --- | --- |
| [`copy_conv_state`](copy_conv_state.md) | Stage convolution state in contiguous storage or write it back. |
| [`fused_qkvzba_split_reshape_cat`](fused_qkvzba_split_reshape_cat.md) | Unpack fused GDN projections into type-contiguous layouts. |
| [`gather_initial_states`](gather_initial_states.md) | Gather recurrent state and zero rows for fresh sequences. |
| [`gather_ssm_states`](gather_ssm_states.md) | Gather selected physical SSM cache rows. |
| [`scatter_ssm_states_`](scatter_ssm_states_.md) | Write updated SSM rows back into cache. |
| [`write_recurrent_output`](write_recurrent_output.md) | Write valid recurrent KDA rows and clear destination padding. |

Earlier descriptions awaiting review are in the [package documentation directory](https://github.com/vllm-project/vllm-ascend/tree/main/vllm_ascend/ops/triton/docs).
