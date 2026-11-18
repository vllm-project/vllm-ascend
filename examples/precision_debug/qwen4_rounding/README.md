# Qwen4Exp BF16 rounding diagnostics

These opt-in, eager replay functions preserve the numerical boundaries used
in the A3 versus H20 first-four-layer investigation. They do not patch installed
models, update weights, or change serving defaults. Run them outside
`torch.compile`: kernel fusion can remove the materialization being tested.

## GDN convolution

```python
from conv import conv_product_bf16

# All inputs are BF16; history is None for a fresh prefill.
aligned = conv_product_bf16(x, weight, history)
control = conv_product_bf16(x, weight, history, round_products=False)
```

The aligned path rounds each product to BF16, then accumulates in FP32 and
applies SiLU before the output BF16 conversion. It does not round the running
sum to BF16. Only a bias-free, four-tap, single-sequence replay is supported.
It returns [tokens, channels]; the native prefill layout and native cache
updates must be preserved by the caller. Do not replace a cache-updating
production operator with this function alone.

Run the CPU regression tests from this directory with `python -m pytest -q`.
The helpers require PyTorch; no model weights are bundled.

## Shared-expert SiLU

```python
from shared_silu import silu_bf16_then_mul

activation = silu_bf16_then_mul(gate_up_projection)
```

This materializes SiLU(gate) in BF16 before multiplying by the up projection.
The investigation applied it to the first layer's shared expert, not to every
activation or the routed experts. Applying it globally would be a different
experiment. The later down projection and shared-expert sigmoid gate are
unchanged.

## PLE gate

```python
from ple_gate import ple_gate_materialized

gated, conv_input, scalars = ple_gate_materialized(
    key, value, hidden, norm_key, norm_query, norm_conv, hc=4, eps=1e-6
)
```

This Ascend-only replay uses five separate kernels and four actual BF16 scratch
buffers: dot reduction, scaling by sqrt(hidden size), signed square root, and
sigmoid. A cast inside a fused kernel is not a substitute for these observed
memory boundaries. It returns the gate output and the following grouped
RMSNorm output. It does not change KV projection, embedding lookup, GR Read,
or convolution. Inputs are read-only; key/value can share a strided KV storage.

Tests in `test_ple_gate.py` require an allocated NPU and the experiment's
vLLM/triton-ascend runtime. Skip them on CPU with
`python -m pytest -q test_conv.py test_shared_silu.py`.
Kernel specializations depend on hidden width and residual group count;
this offline diagnostic does not register serving warmup variants.

## PLE residual additions

```python
from ple_residual import ple_residual_bf16

output = ple_residual_bf16(gated, activated_conv_output, outer_residual)
```

The first sum is written to BF16 before adding the outer residual. This function
is for replay **after** the native convolution and SiLU; it does not reapply
either operation or touch convolution state. In the model-level diagnostic,
the equivalent change split the first residual addition from the outer one
using a second launch. Do not add either residual twice when integrating it.

## Evidence and limits

The source experiments used a BF16 first-four-layer export, one fixed 2841-token
prefill followed by 15 decode steps where reported, TP1/eager, routed experts
disabled and shared experts retained. Core commit was
`5df63a7203cdb9328d824a27b682e77c7b9928be`; Ascend commit was
`5108abeab33fd75b77f5834ee5cba9597e951b73`.

- Convolution plus shared SiLU replay reduced first shared-expert maximum ABS
  from 0.015625 to 0.0087890625 in that experiment. Second GDN maximum ABS
  increased from 0.11328125 to 0.126953125 despite a lower mean error.
- With fixed PLE KV/hidden inputs, all four gate boundaries made 29,091,840
  gated elements bitwise equal to the frozen H20 reference. Grouped norm still
  differed at 211 elements, maximum ABS 0.0078125.
- Adding the PLE residual boundary in that fixed-input experiment reduced PLE
  maximum ABS from 0.0078125 to 0.0009765625. Different upstream inputs still
  produce different PLE outputs; this is not an end-to-end equality claim.

These are historical experiment results, not evidence that arbitrary serving
shapes, graph capture, speculative decode, model accuracy, or performance have
passed for this packaged example. Re-run the tests in the target environment.
The extra kernels and allocations are intentional diagnostic costs. The
examples are not installed into the model runtime and do not assert that A3
is less accurate or that the long-reasoning issue is fixed.

No H20 FP32 down changes, GR Read overrides, fixed-output injection, TP1 cache
adapter, or FP32 beta experiments are included. In particular, native decode
rejected FP32 beta in the earlier test. No production default changes are
proposed by this draft.

Packaging validation on Ascend (2026-11-18): the nine tests passed. Replaying
the archived 2841-token case through the packaged PLE gate also reproduced
both historical output hashes exactly (29,091,840 elements each):

- gated: `a961ab9bfc65090f3425519964db135ecf070821d0f6d98c96944610d12009c2`
- normed: `70f3f70d9146de67d18b7e4264c13689a1df2270c51ac77b07a45d01e6781696`

The archived input tensors are not distributed with this example. The included
tests can be run without them; the hashes document the additional historical
fixture check, not a substitute for testing other workloads.
