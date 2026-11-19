# PrefixLM decision pooling

The standard Ascend attention backend supports complete pooling requests with
bidirectional blocks and causal visibility between blocks. The model provides
inclusive token ranges through Model Runner V2. The hardware plugin preserves
those ranges and evaluates an explicit attention mask; it does not register or
patch a model implementation.

This requires the matching vLLM model-state hook and D3 prefix implementation:

- vLLM base `c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15` with
  `Liuchenbing-2026/vllm` branch `startlux_decision_eager_pr-g`
  (`5ead79dded61ee73f8c5419c1bb3dcc71e682f09`).
- vLLM-Ascend base `a38556f69529595e70598d0d6203009f5ac3293e` with this
  branch (`026_new_laya_model-g`).

**Validation status on this refreshed baseline:** only static syntax checks have
been run locally; CPU unit tests, CI/lint, and NPU model/E2E evaluation on this
paired baseline are pending. Historical `v0.30.0rc2` smoke results below do not
establish compatibility or acceptance for this refreshed main pair.

## Launch

With matching paired sources installed in a fresh container, use a local
`vllm-sr/d3-turbo` checkpoint:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve <WEIGHTS_DIR>/d3-turbo \
  --served-model-name d3-turbo --runner pooling --dtype bfloat16 \
  --pooler-config '{"task":"classify","use_activation":false}' \
  --max-model-len 512 --max-num-batched-tokens 512 --max-num-seqs 1 \
  --gpu-memory-utilization 0.9 --no-enable-prefix-caching \
  --no-enable-chunked-prefill --enforce-eager
```

For the historical PIECEWISE path, replace `--enforce-eager` with:

```bash
--compilation-config '{"mode":3,"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[128,256,512]}'
```

Send typed text questions through `/v1/decisions`. State and question
boundaries are supplied by the model strategy. State tokens cannot read
question tokens; question tokens can read the complete state and their own
question block.

## Historical validation and limits

- Historical smoke check (`quay.nju.edu.cn/ascend/vllm-ascend:v0.30.0rc2` with
  paired development checkouts): real `d3-turbo` weights were tested on an
  Ascend 910B4-1 with BF16 in Eager and PIECEWISE against a same-weight CPU
  FP32 checkpoint-runtime reference. This is historical smoke coverage, not
  refreshed-main acceptance or a leaderboard reproduction.
- `tests/e2e/pull_request/one_card/test_prefix_attention_graph.py` compares
  FP16/BF16 MHA and GQA against independent dense FP32 attention, including
  changed inputs and block boundaries during actual NPU graph replay.
- Cached/chunked prefill, prefill context parallelism, and sliding attention
  are explicitly rejected for this path. Masks are dense and quadratic in
  request length. Long context, multimodal inference, FULL graph mode, and
  shared state computation across questions have not been validated.
