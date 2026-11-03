# SplitQuant MoE support

This branch includes the SplitQuant Qwen3 MoE integration for Ascend. The
vLLM-Ascend plugin registers the model and uses the fused Ascend router and
MoE path when the SplitQuant environment is enabled.

## Runtime configuration

For the 30B SplitQuant model, source the repository profile before starting a
new vLLM process:

```bash
source /path/to/vllm-ascend/use_splitquant.sh 30b graph
```

The profile enables graph execution, packed SplitQuant weights, the fused MoE
kernel, and the fused router. `SPLITQUANT_FUSED_ROUTER` defaults to `1` in the
model implementation. Set it to `0` only when diagnosing the reference
`softmax + topk` route.

The fused router is selected only for its supported shape (BF16 input, 128
experts, top-8 routing, and 1–32 tokens). Other shapes use the reference route
and are counted as fallbacks in the evaluation snapshot.

## Validation

The reproducible PIQA evaluator is:

```bash
python experiments/eval_multimodel_current_vllm.py \
  --size 30b --mode fused --tasks piqa --aclgraph \
  --model-path /path/to/Qwen3-30B-A3B-splitquant-w4a4-fixed \
  --architecture SplitQuantQwen3MoeForCausalLM \
  --expected-quantized-projections 192 \
  --out experiments/current_validation/30b
```

Record the vLLM-Ascend, PyTorch, CANN, model, dataset checksum, native library
checksum, and environment flags with the result. A valid run should report
`router_calls` increasing during PIQA; zero means that the reference router
was used.

## Pull requests

Follow the repository contribution guide and PR template. SplitQuant changes
normally use `[Kernel]` for Ascend kernel changes, `[Core]` for runtime/model
integration, `[Bugfix]` for correctness fixes, and `[Doc]` for documentation.
Commits must include a DCO sign-off (`git commit -s`). PR descriptions must
state user impact and include copy-pasteable validation commands.
