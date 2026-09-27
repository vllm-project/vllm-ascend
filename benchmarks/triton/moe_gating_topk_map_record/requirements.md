# MoE gating TopK + map + record: experiment contract

This experiment targets upstream main `8d4409d6`: CANN gating TopK,
Triton EPLB logical-to-physical mapping, then a separate Triton load-record
kernel using local counts supplied by the downstream MoE operator.
The private ASC reference fuses TopK with a one-dimensional mapping but does
not record load; PR #17574 is a separate Triton design reference, not code to
copy.

Inputs: contiguous logits `[T, E]`; optional correction bias `[E]`;
periodic replica table `[R, E]` (int32); scalar device tensors for recording
enable and valid-token count; mutable physical-expert load view and its local
expert range; static `K`,
scoring kind, grouping parameters, renormalization, and scaling factor.
Outputs: TopK weights `[T, K]` (same dtype as logits), physical IDs `[T, K]` (int32), and
in-place increments to the local load-view slice for valid rows only. IDs must match the
reference exactly; weights use a documented floating-point tolerance; counts
must match exactly. Empty and padded rows, repeated physical experts, ties,
recording-off and table updates are mandatory functional cases.

Primary business cases: T=64/128/256/512 and 65536/131072/262144/524288;
E=8/16/32; K=6 or 8 when K<=E; softmax and sigmoid. Grouping defaults to
one group pending a grouped-routing requirement. Float32 logits are primary;
fp16/bf16 are compatibility checks. Measure the complete routing segment,
each kernel with `msprof op`, and wrapper/launch latency separately.

Performance is compared on the same NPU, environment, inputs and recording
mode, including the three-kernel main-path baseline and a one-kernel
candidate. The record kernel is downstream of MoE in production, so summing
durations is only a routing-segment cost model, not an end-to-end graph claim.
Accuracy and functionality gate performance claims. At most seven
optimization rounds; each round gets one signed commit and an entry in
`performance.md` with the shape matrix, method, outcome and explanation.
