# PrefixLM decision pooling

The standard Ascend attention backend supports complete pooling requests with
bidirectional blocks and causal visibility between blocks. The model provides
inclusive token ranges through Model Runner V2. The hardware plugin preserves
those ranges and evaluates an explicit attention mask; it does not register or
patch a model implementation.

This requires the matching vLLM model-state hook and D3 prefix implementation,
validated at [d74c7f54aefa0f4be5fafb2362950290886328b5](https://github.com/Liuchenbing-2026/vllm/commit/d74c7f54aefa0f4be5fafb2362950290886328b5).
The validation runtime used the `quay.nju.edu.cn/ascend/vllm-ascend:v0.30.0rc2`
image with matching source checkouts and a separate dependency environment;
the unmodified image alone does not include these changes.

## Launch

With the matching sources installed, use a local `vllm-sr/d3-turbo` checkpoint:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve /models/d3-turbo \
  --served-model-name d3-turbo --runner pooling --dtype bfloat16 \
  --pooler-config '{"task":"classify","use_activation":false}' \
  --max-model-len 512 --max-num-batched-tokens 512 --max-num-seqs 1 \
  --gpu-memory-utilization 0.9 --no-enable-prefix-caching \
  --no-enable-chunked-prefill --enforce-eager
```

For the validated PIECEWISE path, replace `--enforce-eager` with:

```bash
--compilation-config '{"mode":3,"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[128,256,512]}'
```

Send typed text questions through `/v1/systemone`. State and question boundaries
are supplied by the model strategy. State tokens cannot read question tokens;
question tokens can read the complete state and their own question block.

## Validation and limits

- Real d3-turbo weights were tested on a 910B4-1 with BF16 in Eager and
  PIECEWISE. Text choice and predicate outputs were compared with a same-weight
  CPU FP32 checkpoint-runtime reference. Reordered and repeated questions were
  also checked. This is smoke coverage, not a leaderboard reproduction.
- `tests/e2e/pull_request/one_card/test_prefix_attention_graph.py` compares
  FP16/BF16 MHA and GQA against independent dense FP32 attention, including
  changed inputs and block boundaries during actual NPU graph replay.
- Cached/chunked prefill, prefill context parallelism and sliding attention
  are explicitly rejected for this path. Masks are dense and quadratic in
  request length. Long context, multimodal inference, FULL graph mode and
  shared state computation across questions have not been validated.
