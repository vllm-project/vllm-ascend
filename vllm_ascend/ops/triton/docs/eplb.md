# EPLB mapping and expert-load kernels

The existing EPLB wrappers and routing/dispatch integration are unchanged.

```text
router logical IDs [T,K]
  -> map_to_physical_triton
  -> physical IDs [T,K]
  -> existing MoE dispatch and computation
  -> MoE-provided expert_tokens
  -> record_expert_tokens_triton
  -> cumulative expert_load
```

## Mapping

Each program owns a balanced, contiguous token range. The grid is
`min(T, vector_core_num)`, independent of tile size. Quotient/remainder
partitioning gives every token exactly one owner; ownership differs by at most
one token between programs. Each program processes its flattened assignments
in bounded tiles of at most 512 elements.

Logical IDs retain their input integer width rather than being unconditionally
widened to INT64. Invalid logical IDs produce `-1`, as in the existing NPU map.
Routing-table contents remain device-side state and can change during graph
replay. Token count and grid count are not value-specialized; TopK and the
power-of-two tile size describe the intended compile-time variants.

## Recording

The record kernel continues to consume counts returned by the existing MoE
path. It does not count map outputs or change source/destination-rank ownership.
Group-list type 1 contains counts; type 0 contains cumulative counts and is
converted by subtracting the previous *global* element, including at block
boundaries.

Independent blocks contain at most 256 experts. When counts have a wider
integer type than the load destination, operands are narrowed before integer
arithmetic. Addition and subtraction commute with this narrowing modulo the
destination width. When counts are narrower, subtraction retains the input
width, preserving the existing overflow semantics. Recording disabled on the
device leaves the load unchanged.

## Validation boundary

The NPU tests cover mapping ownership/tails, invalid IDs, integer widths,
runtime routing-table and record-flag updates, and record block boundaries.
Performance and model-profile results belong in the PR description, with the
measured source and environment retained in task artifacts.

Padding-aware load correction is a separate correctness change. This kernel
optimization does not remove padding assignments from upstream MoE counts and
does not change communication, gating, routing weights or model support.
