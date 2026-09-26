# AttnResFwd with residual addition

This A5 entry shares PR #16068's resident/reload implementation with
`AttnResFwd`. It forms `BF16(prefix_sum + addend)` once inside the kernel,
uses that rounded value for both residual scores and the weighted output,
and returns it separately for the MLP residual connection.

```python
output, new_prefix = torch.ops._C_ascend.attn_res_fwd_with_add(
    prefix_sum, addend, block_residual, proj_weight, norm_weight, norm_eps
)
```

The original `attn_res_fwd` binding remains unchanged for calls without an
addition. Both entries accept BF16 tensor views through ACLNN's existing
contiguous conversion. `block_residual` contains only initialized blocks;
the with-add entry also accepts zero blocks and then returns the rounded
sum. Inputs are never modified, including prefixes retained by DSpark.

The fused variant allocates one additional BF16 hidden-size row in UB.
Tiling includes this buffer when selecting resident versus reload mode.
Only A5 is registered for the fused variant; the ordinary reference
operator retains its A3/A5 implementations. No Triton residual fallback
is used by K3.
