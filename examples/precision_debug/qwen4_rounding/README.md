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
